# Frozen-Sim Decode Performance Round 2

Date: 2026-07-10

## Scope and measured anchor

This pass targets only frozen-sim decode `select`. It does not change PQ
construction, committed-set semantics, the policy walk, RNG, or golden CSV
generation.

The measured 32k anchor is job 53231495 arm C: 32 generated tokens, 32
layers, 8 KV groups/layer, hence 8,192 group iterations. `select` took
248.8 s (30.37 ms/group), of which the deferred wait observed 170.4 s of
already-enqueued GPU work (20.80 ms/group). The remaining 78.4 s is about
9.57 ms/group of Python, allocator, launch, and non-waiting work.

The standard frozen-sim wrapper still uses the CUDA extension for the PQ
fullscan, but deliberately forces the score grid, precision V-prefix, and
policy-delta path to Torch. The new native precision helpers below are invoked
only by explicit default-off flags.

## Static per-group decomposition at 32k

The times below partition the measured 20.80 GPU-ms/group. They are estimates
from tensor shapes and operation/launch structure, not independently timed
CUDA events. The launch column counts approximate CUDA launches, including
multi-kernel ATen reductions/sorts; exact counts vary by Torch/CUDA version.

| Work per KV-group iteration | Shape / operation | Est. GPU ms | Est. launches before | Scaling |
|---|---|---:|---:|---|
| Transient V-PQ reconstruction | code lookup to `[ctx,128]` vhat, then residual | 1.2 | 6-9 | context rows (4x at 128k) |
| Frozen K/V lo plane and commit | two rowwise absmax QDQs, appended error, commit compare, lo-QK | 2.0 | 18-24 | context rows; lo-QK also heads |
| Four/six K-rung score composition | exact fill, PQ indexed fill, exact/lo prefix overwrites | 1.4 | 24-32 | `K * heads * context` |
| Softmax and V-PQ base output | row softmax plus base reduction | 2.0 | 2-5 | `K * heads * context`; reduction order stays per row |
| Risk construction | `p^2 * code_error` | 0.3 | 2 | `K * heads * context` |
| Risk ordering | sorted top-k/full stable order per K/head row | 7.0 | 5-10 | scans context; roughly `K * heads * context` plus sort complexity |
| V-prefix gathers | probabilities, hi residual, lo residual, commit bits | 2.8 | 4-7 | selected V-prefix length; 4x when the fractional V schedule reaches full context |
| Two fp32 residual cumsums + int32 commit cumsum | `[K,heads,Vprefix,128]` twice plus counts | 5.3 | 8-13 | selected V-prefix length; reductions remain within each row |
| Seven V-budget projections | prefix-cell reads, hi/lo composition, base add, stacking | 0.4 | 24-34 | V-grid size only once cumsums exist |
| Policy-grid relative norms and packing | adjacent K/V output-grid deltas in fp64 | 0.4 | 7-11 | `K * V * heads * head_dim`, not context |
| **Total** | | **20.8** | **about 100 (85-120)** | |

The launch estimate is consistent with the prior 15k-25k launches/token
audit: 256 group iterations/token times roughly 85-100 hot-path launches is
about 22k-26k, with variation from sort/cumsum internals and cache hits.

At the canonical fractional schedules, the score/risk/sort/reconstruction
work scales with all context rows. The gathered residual and cumsum work also
scales at 128k because the final V rung is full context. V-budget projection
and policy norms scale only with the 6-by-7 grid, four query heads, and head
dimension.

## Default pure-execution changes

1. The four/six K score rows are filled together in the existing shared
   `[K,heads,context]` score workspace. Exact/PQ/lo assignments are unchanged
   element by element; there is no new reduction.
2. Seven V-budget outputs are gathered together from the existing fp32/int32
   cumsums. The arithmetic remains `(hi + lo_exact) - lo_hi`, matching the
   eager sequence. Direct CPU tests compare these helpers bit-for-bit with the
   old per-rung loops.
3. V-PQ reconstruction uses one model-level, stream-ordered vhat/residual
   workspace. Immutable lookup rows are cached with only the current pack;
   old pack layouts are discarded to avoid decode-length growth.
4. `SELECTOR_PQ_JOINT_VPREFIX_TRANSIENT_BUDGET_MB` defaults to 1600. The
   precision V-prefix estimates four simultaneously live fp32
   residual/product/cumsum planes and selects a K chunk accordingly.

For 32k, the four-plane estimate is 256 MiB per K row. Six rows require
1,536 MiB, so the default retains the original `[6,4,32768,128]` sort/cumsum
batch shapes through the 32-token identity run. For 128k, the estimate is
1,024 MiB per K row, so the default runs `[1,4,131072,128]` K slices in
sequence. The cumsum is still along the same dimension 2 and is independent
per row, but the changed outer batch shape at 128k remains an empirical GPU
identity gate rather than a claimed bit proof.

### Projected 128k residency

The measured warmed base was 32,800 MiB allocated. A one-K 128k V-prefix has
four dominant fp32 planes totaling 1.00 GiB. Shared vhat/residual scratch is
0.125 GiB; risk/order, commit prefix, score/probability grids, and small output
grids add roughly 0.15-0.35 GiB. Allowing for other live lo-plane tensors and
allocator fragmentation gives a conservative select transient of about
1.5-2.0 GiB.

Projected peak is therefore approximately **34.3-34.8 GiB**
(`32.8 + 1.5-2.0`), conservatively below 35.5 GiB and the requested 37 GiB
ceiling. This leaves about 3.7 GiB or more below a 39.25 GiB MIG. This is a
static projection; the 128k rerun is the gate.

## Default-off math-exact flags

All flags below default to zero and are rejected by the canonical-GPU guard.
They may change bits and require the same identity/quality A/B used for prior
key-domain decisions.

- `SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX=1`: keep Torch's risk order, but
  use one CUDA kernel to stream hi/lo residual prefixes and emit all V rungs
  without materializing the large gathered/cumsum planes. It changes prefix
  accumulation associativity/FMA behavior and requires rebuilding the CUDA
  extension.
- `SELECTOR_PQ_JOINT_FROZENSIM_FUSED_RISK_SORT=1`: requires fused V-prefix;
  construct risk, run a CUB stable segmented full sort, and consume the order
  in the fused prefix kernel. It changes the top-k/full-sort tie domain and
  reduction math.
- `SELECTOR_PQ_JOINT_FROZENSIM_COMPILE=1`: apply dynamic `torch.compile` to
  risk construction, rowwise int8 QDQ, and adjacent policy-grid relative-L2
  chains. Inductor may reassociate reductions, and first-use compilation is
  included in an end-to-end A/B unless separately warmed.
- `SELECTOR_PQ_JOINT_FROZENSIM_BF16_VPREFIX=1`: compute risk and gathered
  residual products in bf16 while retaining fp32 cumsum outputs. It changes
  risk order and products; it is incompatible with the fused native flags.
- `SELECTOR_PQ_JOINT_FROZENSIM_TF32_QK=1`: permit TF32 only while dispatching
  exact-QK and lo-QK matmuls, then restore the prior global setting.

## Expected runtime, not promotion evidence

These projections start from 264.3 s attention/sample at 32k. The 128k
column includes the extra launches and possible utilization loss from one-K
chunking. Midpoints are planning estimates; ranges are more meaningful until
the same-GPU trace runs.

| Configuration | 32k s/sample | Projected 128k s/sample | What remains GPU-bound |
|---|---:|---:|---|
| Pure execution only (default) | **258** (250-265) | **820** (760-950) | unchanged risk ordering and three cumsums; sequential K chunks at 128k |
| Pure + compiled chains | **252** (240-265) | **800** (720-930) | sort/cumsums unchanged; compile warm-up can erase gains |
| Pure + bf16 V-prefix | **235** (215-260) | **720** (620-880) | ordering plus fp32 cumsum output; bandwidth reduced |
| Pure + TF32 QK | **256** (248-264) | **810** (750-945) | V-prefix dominates, so QK is a small lever |
| Pure + fused V-prefix | **205** (175-260) | **650** (520-900) | Torch top-k/full sort remains dominant; streamed random reads may regress |
| Pure + fused V-prefix + fused risk sort | **195** (160-270) | **600** (450-950) | CUB full sort and streamed residual reads; native shape may regress |

The default pure pass removes dozens of launches and allocations per group,
but round 1 showed that CPU enqueue work can overlap the GPU. Therefore the
headline expectation is deliberately near-neutral to about 1.06x, not the
sum of nominal CPU savings. The fused/reduced-precision flags attack the GPU
bottleneck; their ranges are deliberately wide because the fused precision
kernel performs streamed, risk-ordered residual reads and earlier native
V-prefix shapes have sometimes regressed in the HF loop.

## Local validation

- Python compilation passed for all touched Python modules and tests.
- `verify_memory_bounded_vpq.py` passed, including direct eager-vs-batched
  score-grid/V-projection bit checks and reconstruction workspace/cache checks.
- CPU-mode joint K/V parity smoke passed on the 16k saved trace at decode 500,
  head 0: four rows, no failures, max attention/o-proj relative L2 0.0.
- Wrapper audit and shell syntax checks are part of the final validation.
- No Slurm, GPU execution, or extension build was performed in this pass.
