# Decode Attention Engine — Hardware Architecture v0.1 (2026-07-06)

Scope: a decode-only sparse-attention engine for long-context LLM inference,
implementing the CPU/GPU-validated frontier algorithm stack as of 2026-07-06
(see `current_status.md`). Prefill is out of scope (host/GPU produces the KV
cache and the engine's index structures, or a seal unit builds them online).
Numbers assume Llama-3.1-8B geometry: d=128, 32 q heads / 8 kv heads (GQA 4:1),
fp16 base precision, contexts to 128k+. Everything marked **[measured]** comes
from the trace runs in `attention_efficiency_result/`; **[derived]** is
arithmetic on those; **[estimate]** is a sizing guess to be validated in RTL.

## 1. Algorithm freeze (what the hardware implements)

1. **K-PQ fullscan selector**: pages of 5632 tokens, 4 subvecs x 8 bits,
   per-page codebooks (per-page adaptivity is quality-load-bearing — global
   codebooks demonstrably break the budget controller). Scan produces a PQ
   logit per token.
2. **Budget controller**: predict start rung from proxy softmax mass (m0.9),
   then escalate/de-escalate walk over rung grid K in {0.10,0.30,0.50,0.70,
   0.90,1.0} x ctx, V in {0.05,...,1.0} x ctx, stability test = output relL2
   delta between adjacent rungs vs sqrt-scaled tau. De-escalation validated
   as a Pareto improvement and makes the controller start-insensitive.
3. **Exact-K refinement**: gather K rows for the selected prefix, replace PQ
   logits with exact logits; tail keeps PQ logits scaled 1/sqrt(d) (mixed
   probabilities).
4. **Exact-V by global residual risk**: risk_i = p_i^2 * err_i with err from a
   2B/token V-PQ code-error sidecar; two-pass threshold variant is
   tile-compatible. Non-selected V uses V-PQ (1 subvec x 4 bits).
5. **Progressive precision (int8 MSB planes)**: low-PQ-rank K read as int8
   plane; low-risk V read int8 only where the per-token commit test passes
   (int8_err < vpq_code_error, both 2B sidecar stats sealed at page close).
6. **var4 compressed-domain bounds** (optional): certify a stability test
   without reading the band.
7. NOT implemented: temporal selection reuse (trace-MB negative under every
   guarded budget policy — see FINAL TEMPORAL VERDICT), page-skip scanning
   (loses needles), global codebooks (breaks ranking fidelity).

## 2. DRAM layout (per kv head)

| Region | Size @128k ctx | Layout notes |
|---|---|---|
| K plane A (absmax int8) | 16.8 MB | row-major, 128 B/token; the lo-tier read; separate contiguous region |
| K plane B (int8 residual) | 16.8 MB | read only for hi/exact tokens (exact = A+B); separate contiguous region |
| V plane A / plane B | 16.8 + 16.8 MB | same split as K |
| K/V scale arrays | 4 x 0.5 MB | 2 fp16 scales/row (A,B), dense separate array, sidecar stream |
| K-PQ codes | 0.5 MB | 4 B/token, page-contiguous for streaming |
| K-PQ codebooks | 64 KB/page (1.5 MB) | 4 x 256 x 32 x fp16, page header |
| V-PQ codes | 64 KB | 0.5 B/token |
| V-PQ codebooks + code-error sidecar | 4 KB/page + 256 KB | err 2B/token |
| int8-err sidecar (V commit test) | 256 KB | 2B/token, written at page seal |

Total index overhead over raw fp16 KV: ~2.5 MB / 67 MB ~= **3.7% [derived]**
(codes + codebooks + 2 sidecars), plus ~1.6% for the scale arrays.
**OPEN-2 RESOLVED (2026-07-06, M6 + full-spectrum job 53003123)**: the
layout above is the validated **int8 dual-plane** format — plane A = per-row
absmax int8 of x (lo tier), plane B = per-row absmax int8 of the residual
(exact tier reads A+B, error ~absmax/127² < fp16 rounding). fp16-equivalent
capacity, +0.13% trace MB, quality identical (relL2 0.00875 vs 0.00871 at
tau=0.004). The rejected fp16-MSB-plane read fails hard (relL2 0.087).
Bandwidth accounting NOTE: the golden model's trace-MB curves charge zero
bytes for scales — RTL bandwidth budgets must add the ~1.6% scale-sidecar
adder (2 B/row lo-tier reads, 4 B/row hi-tier) on top of every trace-MB
number in this document.

## 3. Engine pipeline (one decode step, one kv-head lane)

```
      q (4 q-heads of the GQA group)
        |
  [S1 LUT]      4 x 256 x 32-MAC per q-head -> 2 KB LUT        (per page)
        |
  [S2 SCAN]     stream codes 4B/token, 4 LUT adds/token -> PQ logit
        |       codes+codebooks read ONCE per kv head, shared by 4 q heads
  [S3 RANK]     histogram/radix select over 2B logits -> rung prefixes
        |       (no full sort; 256-bin exponent histogram, 2 passes)
  [S4 K-GATHER] fetch K rows for marginal band (int8 plane; +LSB for hi tier)
        |       exact logits d=128 MAC/row -> patch logit buffer
  [S5 SOFTMAX]  per-band partial (max, sum, acc[128]) fp32, flash-style;
        |       band partials kept per rung -> combine tree
  [S6 V-PATH]   risk = p^2 * err threshold select -> V gather
                (int8 plane if commit test passes, else fp16; else V-PQ LUT)
```

Controller FSM wraps S4-S6: PREDICT -> {VERIFY, WALK-UP, WALK-DOWN} -> COMMIT.
A walk step = process one marginal band through S4-S5, compare combined
output vs previous rung (relL2 on 128-dim fp32, trivial). **Down-probes are
free of DRAM traffic**: recombine the band-partial tree without the top band
— this is why the de-escalating controller costs nothing extra in bytes.
var4 certify (optional) sits before S4 and can skip a band read entirely.

Page seal unit (background, off critical path): 3-iter k-means on the sealed
page (5632 x 128 fp16), V-PQ codebook, err + int8-err sidecars. Amortized over
5632 decode steps; at 100 tok/s that is a ~56 s budget per page **[derived]**.

## 4. On-chip SRAM budget (per kv-head lane) [estimate]

| Buffer | Size | Notes |
|---|---|---|
| PQ LUT | 2 KB x 4 q-heads | rebuilt per page |
| Codebook staging | 64 KB double-buffered = 128 KB | stream from DRAM |
| Logit buffer | 2 B x 128k = 256 KB x 4 q-heads = 1 MB | the big one; scales with ctx |
| Rank histogram | 1 KB | 256 bins x 4B |
| Band partials | ~6 KB x 4 q-heads | <=12 rungs x (max,sum,acc[128] fp32) |
| Output/accum | ~2 KB x 4 q-heads | |
| Sidecar staging | 32 KB | err values for selected tokens |
| **Total / lane** | **~1.3 MB** | x 8 lanes ~= **10.5 MB** chip-wide |

The logit buffer dominates and is the main context-length scaling limit; a
1 MB/lane buffer caps at 128k ctx per resident query. Options if 256k+ needed:
spill logits to DRAM (adds 2B/token/step re-read — this is exactly the "stale
logit reread" cost class measured in the temporal experiments), or compress
tail logits to 1B (untested — flag for a trace experiment).

## 5. Bandwidth budget per decode step [measured, per q-head-query]

De-escalating controller, proxy-mass start, thr = tau, 288-position trace
(`ladder_deescalate` run). Components: scan = codes+codebooks (UNSHARED trace
accounting), exactK = selected K rows at fp16, vPath = exact V + sidecars.

| ctx bucket | tau | scan | exactK | vPath | total MB | K frac | V frac |
|---|---|---|---|---|---|---|---|
| <=16k  | 0.002 | 0.10 | 1.83 | 0.63 | 2.55 | 0.80 | 0.27 |
| <=40k  | 0.002 | 0.34 | 4.74 | 1.35 | 6.44 | 0.76 | 0.22 |
| <=80k  | 0.002 | 0.83 | 7.78 | 1.67 | 10.3 | 0.56 | 0.11 |
| <=140k | 0.002 | 1.57 | 9.67 | 3.14 | 14.4 | 0.37 | 0.11 |
| <=140k | 0.004 | 1.57 | 7.37 | 2.18 | 11.1 | 0.28 | 0.07 |

Chip-level corrections to the trace numbers [derived]:
- **GQA scan sharing**: codes (4B/tok) + codebooks (11.6B/tok) are read once
  per kv head and serve 4 q heads -> effective scan ~3.9 B/token/q-head, i.e.
  the 1.57 MB scan term becomes ~0.4 MB per q-head equivalent.
- **Progressive precision** (int8 tiers, hi-frac 0.1): -31% on the
  exact-read terms at identical trace relL2 [measured on the precision grid]
  -> at tau=0.004 + precision the 140k total is ~**7.5 MB/q-head** vs dense
  67 MB = **~9x dense reduction** [derived], before GQA scan sharing.
- Adaptive floor: short contexts settle near the behavioral floor (~10% K /
  5% V), so the reduction ratio *grows* with context — the engine gets more
  valuable exactly where decode is most bandwidth-bound.

Energy sanity check [estimate]: at 7.5 MB/step/q-head x 32 q-heads x 32
layers ~= 7.7 GB per generated token... this is wrong by construction — the
trace charges per *sampled* layer/head; real per-token traffic = per-layer
sum over kv-lanes with GQA union reads. Marked open (Sec. 8, M2): need a
union-read measurement across the 4 q heads of a group before quoting a
per-token energy number.

## 6. Why this wants a custom chip (GPU-unfriendliness inventory)

1. **Sequential rung walk with data-dependent early stop** — each band's
   fetch depends on the previous band's softmax delta. On GPU this serializes
   kernel launches or wastes speculative bandwidth (the trace's lookahead
   accounting showed wasted-band cost); an FSM pipelines it naturally.
2. **Per-token branchy reads** — the V commit test (int8 vs fp16 vs V-PQ per
   token) and risk-threshold gather are divergence disasters on SIMT, trivial
   for a gather engine with a 3-way branch per descriptor.
3. **Bit-plane partial reads** — reading the MSB half of rows at 128B
   granularity wrecks GPU coalescing assumptions; a custom DRAM scheduler
   issues them as native bursts.
4. **Band-partial combine tree** — down-probes recombine stored partials
   instead of recomputing; GPUs would keep these in global memory across
   kernel boundaries.
5. **On-line page seal** — background k-means co-resident with decode.
6. What does NOT need the chip (honest list): the PQ scan itself (GPUs do LUT
   scans well), dense prefill, and anything the trace showed negative.

## 7. Positioning vs KV-compression accelerators

Sparsity (which tokens to read) and precision (bits per read) are orthogonal;
this engine occupies sparsity + *recoverable* precision. Because exact KV
remains in DRAM (bit-planes), worst case degrades to "read more bytes," never
"answer from corrupted state" — the escalation path is the guarantee static
quantization cannot offer. Capacity cost is +3.7% index overhead, traded for
5-9x bandwidth at calibrated-task-safe error (noise-injection: RULER flat to
relL2 0.05 at 32k, 0.01 at 128k). The composition with quantization is
already inside the design (int8 tiers with commit tests), so a 4-bit-KV
competitor is a *tier*, not an alternative.

## 8. Open questions / measurement TODOs (ordered)

- **M1 DONE (2026-07-06)**: tau sweep all-100.0 -> operating point FROZEN at
  deescalate+precision @ tau=0.004 = 2.857 MB/head-query (288-pos spectrum).
- **M2 DONE (job 52989540)**: GQA union/sum across the 4-head group: K
  0.35-0.44, V 0.49-0.57 (short ctx -> 128k). Gather-engine DRAM traffic =
  per-head bytes x union factor; ~5x system-level reduction at 128k.
- **M3 DONE (job 52987561)**: composition holds; precision remains free on
  top of de-escalation at both thresholds.
- **M4**: logit-buffer precision — can tail logits live at 1B in SRAM?
  (affects the dominant SRAM buffer; one CPU trace experiment).
- **M5.5 (was implicit)**: M6 below folded here.
- **M6 — DONE (2026-07-06)**: fp16-MSB-plane lo tier FAILS (relL2 0.087); int8 dual-plane storage validated at full spectrum (job 53003123: +0.13% MB, quality identical). DRAM layout in Sec. 2 is final; see algorithm_spec_v1.md Sec. 6.
- **M5**: rank-select rung granularity for the histogram select vs the
  budget grid (hardware wants power-of-2 bins; grid is fractional).
- **RTL order**: S2 scan + S3 rank first (fixed-function, testable against
  trace CSVs bit-exactly), then S5 partials + controller FSM (the novel
  part), then gather engines, seal unit last (host can seal in v0).

## 9. Cross-references

- Algorithm validation: `notes/current_status.md` sections dated 2026-07-05/06.
- Trace data: `attention_efficiency_result/joint_kv_ladder_grid_20260706/`,
  `joint_kv_progressive_precision_20260705/`, `joint_kv_temporal_reuse_20260705/`.
- Task-level safety: `benchmark_suite_result/attn_noise_calibration_20260705/`,
  `frontier_tau_sweep_20260706/` (in flight).
