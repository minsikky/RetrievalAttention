# Frozen-sim V-PQ Sidecar Memory Audit (2026-07-09)

Scope: Llama-3.1-8B, 32 layers, 8 KV heads, head dim 128, context
131072, page size 5632, static prefix/suffix 128, K-PQ S4B8, V-PQ
S1B4, float32 probability/grid domain, precision tiers enabled, and the
standard all-head K scan. This gives 23 sealed pages per layer and KV head
(`sealed_end=129664`). Physical sizes below use CUDA tensor dtypes, not the
logical byte-cost model.

## Audit result

The original hypothesis identified the right tensors but the wrong dtype.
`vpq_values_for_tokens_gpu` explicitly allocates float32 reconstruction, and
`residual_t = values_t.float() - vhat_all_t.float()` is also float32. With the
default 256-row grow pad, each persistent plane has physical shape
`[131328, 128]`, not just the returned `[131072, 128]` view.

| retained tensor per layer x KV head | dtype | physical shape | MiB/head | GiB, 32 layers x 8 heads |
| --- | --- | --- | ---: | ---: |
| K page codebooks (`GPUIndex.native_codebooks`; `PagePQ.codebooks` are views) | float32 | `[23,4,256,32]` | 2.875 | 0.718750 |
| K page codes (`GPUIndex.native_codes`; `PagePQ.codes` are views) | uint8 | `[23,5632,4]` | 0.494 | 0.123535 |
| K page starts | int64 | `[23]` | 0.00018 | 0.000044 |
| grouped all-head K codebook copy | float32 | attributed `[23,4,256,32]` | 2.875 | 0.718750 |
| grouped all-head K code copy | uint8 | attributed `[23,5632,4]` | 0.494 | 0.123535 |
| V-PQ codebooks | float32 | `[23,1,16,128]` | 0.180 | 0.044922 |
| V-PQ codes | uint8 | `[23,5632,1]` | 0.124 | 0.030884 |
| V-PQ page starts | int64 | `[23]` | 0.00018 | 0.000044 |
| old persistent `vhat` | float32 | `[131328,128]` | 64.125 | 16.031250 |
| old persistent residual | float32 | `[131328,128]` | 64.125 | 16.031250 |
| `code_error` | float64 | `[131328]` | 1.002 | 0.250488 |
| append-only V int8 commit error | float16 | `[131328]` | 0.250 | 0.062622 |

The grouped K pack is a real contiguous copy made by `torch.stack`; its fast
and normal cache dictionaries reference the same copy. Per-head `PagePQ`
objects and prefix-index objects are views/references and do not duplicate the
underlying K tensors. The V pack is not grouped on the frozen torch-grid path.

The two old float32 planes alone are 32.0625 GiB including grow pad (32.0 GiB
without it). This exactly explains the observed +8 GiB at 32k. The prior
fp16 estimate of about 17 GiB undercounted them by approximately 2x. Adding
the PQ packs, scalar caches, and grouped K copy gives about 34.14 GiB of old
retained decode sidecars, so the projected 128k steady allocation was about
65 GiB before ordinary decode workspaces, not 48.8 GiB.

`code_error` also needs a correction: the logical cost model charges two
bytes, and the commit comparison casts it to fp16, but the resident tensor is
float64 because float32 risk construction consumes it. Changing its stored
dtype would change risk/order decisions, so the memory-bounded path retains
the float64 tensor unchanged.

Other context-sized state is transient or shared:

- HF K/V tensors are views of DynamicCache and remain bf16; there is no KV
  duplicate in `torch_k_cache`/`torch_v_cache`.
- `residual_lo_commit` is a float32 `[context,128]` transient for the current
  KV head, not retained across layers.
- The torch score-grid workspace is one model-shared float32
  `[4,4,context+256]` buffer (about 8.0 MiB at 128k), and the shared position
  arange is about 1.0 MiB.
- PQ logits, exact/lo logits, probabilities, risk, sort indices, gathered
  residuals, cumsums, QDQ K/V planes, and K/V build tiles are per-call
  transients. `FRONTIER_DENSE_KEY_T_CACHE=0`, so no float32 K-transpose plane
  is present.

## Memory-bounded representation

`SELECTOR_PQ_JOINT_MEMORY_BOUNDED_VPQ=1` is the new default. It is active
only for CUDA precision-tier runs; CPU and non-tier paths keep their previous
operation sequence.

Warm now builds and retains only the existing V-PQ codes/codebooks plus the
exact float64 `code_error` vector. The statistic is computed one page at a
time with the same float32 residual, float64 square/sum, and page/code bucket
mean operations as the old full-plane builder. Page starts come from CPU
index geometry, avoiding one CUDA synchronization per page. Exact unsealed
suffix rows append zero `code_error`, as before.

At decode, each active KV head reconstructs one float32 `vhat` transient from
the unchanged codes/codebooks, then executes the old expression
`values_t.float() - vhat_all_t.float()` to make a transient residual. The old
`probs @ vhat`, risk ordering, gathers, cumsums, precision-tier commit mask,
`residual_lo_commit`, policy, accounting, and deferred D2H paths are unchanged.
The transients die when that KV-head group finishes, so at most one head's
planes are live rather than all 256 layer-head planes being retained.

The native histogram/code-aggregation base output was deliberately not used:
it changes reduction order and therefore cannot establish bit identity with
the old torch matmul. Pinned-CPU planes were also rejected because they would
stream 16+ GiB over PCIe per token at 128k.

## Projection

Using the measured job-53226865 base (`15316 MiB` model load and `31605 MiB`
after prefill), the new 128k projection is:

| component | 128k GiB |
| --- | ---: |
| weights | 14.957 |
| bf16 KV (measured delta) | 15.907 |
| retained K-PQ original + grouped copy | 1.685 |
| retained V-PQ pack | 0.076 |
| retained float64 code error + fp16 commit error | 0.313 |
| shared persistent workspaces | about 0.009 |
| current-head decode transient peak (analytic upper bound) | at most about 1.2 |
| **projected decode peak** | **at most about 34.15** |

Warm does not create the grouped K copy or commit cache yet. Its final retained
sidecars are about 1.17 GiB, and `PAGEDPQ_BUILD_TEMP_BUDGET_MB=512` still caps
the dominant K/V build tile. Streaming risk needs only page-sized value,
reconstruction, residual, and reduction temporaries (well under 32 MiB), so
the projected warm allocation is about 32.6 GiB. The historical 38.45 GiB
prefill max counter remains visible but is not concurrent allocated residency.

For a literal 1,000,000-token extrapolation, measured KV is scaled linearly
and page state uses 177 sealed pages:

| component | 1M GiB |
| --- | ---: |
| weights | 14.957 |
| bf16 KV | 121.363 |
| retained sidecars/scalars | 15.932 |
| decode transient/workspace upper bound | about 3.2 |
| **projected decode peak** | **about 155.5** |

For 1,048,576 tokens, the corresponding projection is about 162.2 GiB (186
sealed pages).

## Expected decode cost

The memory saving moves deterministic V reconstruction from once-per-sample
warm into every decode step. At 128k, all 32 layers x 8 KV heads reconstruct
16 GiB of float32 `vhat` and 16 GiB of float32 residual per token. Including
codebook gathers, the reconstruction copy, bf16-to-float conversion, and the
residual subtraction, the conservative gross extra device traffic is about
136 GiB/token. That is roughly a 0.09--0.17 second pure-HBM lower bound over a
0.8--1.5 TB/s effective range, before 256 small reconstruction launches; a
practical 128k delta around 0.15--0.35 second/token is plausible but requires
the requested GPU A/B to measure. At 1M the traffic scales to about 1.0 TiB per
token. This cost is preferable to CPU streaming and is the minimum
bit-identical route while the dense torch `probs @ vhat` base remains part of
the frozen simulator.

## Validation contract

`benchmark/selector_eval/runners/verify_memory_bounded_vpq.py` checks exact
CPU equality for old full-plane versus streaming `code_error`, exact-suffix
cache growth, old versus transient V-PQ reconstruction/residual, final
precision-tier V-prefix outputs, and lo-read commit counts. The CUDA path uses
the same elementwise values and preserves the output-producing torch matmul,
sort, gather, cumsum, policy, and accounting order; the requested same-GPU
old/new task A/B remains the final empirical identity gate.
