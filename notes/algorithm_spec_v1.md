# Frontier Decode Attention — Algorithm Specification v1 (RTL handoff)

Date: 2026-07-06. This is the *contract* between the algorithm side (this
repo, CPU/GPU reference implementations) and the RTL side. RTL implements
exactly what is specified here; anything marked OPEN is not frozen and must
not be silently assumed. Companion architecture sketch: `hw_arch_v0.md`.
Validation history and evidence: `current_status.md`.

Reference implementation (golden model): 
`benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py`
(CPU, trace-driven; function/line pointers below refer to this file unless
stated). GPU end-to-end: `benchmark/ruler/pred/call_pagedpq_streaming.py` +
`benchmark/selector_eval/cuda_ext/`.

## 0. Fixed model geometry

- head_dim d = 128, fp16 base precision for K/V rows.
- GQA: G = 4 query heads per kv head (Llama-3.1-8B: 32 q / 8 kv heads).
- Softmax logit scale: 1/sqrt(d) applied to ALL logits (exact and PQ tail).

## 1. Index structures (built at page seal; seal = background, off hot path)

- **Page**: 5632 contiguous tokens. Static prefix = first 128 tokens and
  static suffix = last 128 tokens of the sequence are always resident
  ("base") and never PQ-indexed. Pages seal when
  `sealed_end = dynamic_start + floor((indexed_end - dynamic_start)/5632)*5632`
  advances (see `build_page_pq_gpu` in `run_gpu_paged_pq_eval.py`).
- **K-PQ**: per page, 4 subvectors x 8 bits (256 centroids of 32 dims each),
  k-means 3 iterations, per-page codebooks (fp16). Codes: 4 B/token.
  Codebook: 4*256*32*2 B = 64 KB/page. PER-PAGE codebooks are load-bearing;
  a global codebook is a validated NEGATIVE (breaks the budget controller).
- **V-PQ**: per page, 1 subvector x 4 bits (16 centroids of 128 dims).
  Codes 0.5 B/token; codebook 4 KB/page.
- **Sidecars** (2 B/token each, computed at seal):
  - `vpq_code_error[i]`: squared-L2 reconstruction error of token i's V row
    under V-PQ (code-table estimate; 0 for non-indexed tokens). See
    `value_vpq_code_stat_risk` in `run_value_exact_strategy_eval.py`.
  - `int8_err[i]`: squared-L2 error of the int8 tier for token i's V row
    (needed by the precision commit test, Sec. 6).

## 2. Selector scan (per decode step, per kv head; shared by G q heads)

For each q head: LUT[s][c] = dot(q_subvec_s, centroid_c) (4x256 fp16 dots of
32 dims). PQ logit per token = sum of 4 LUT lookups. Tail logits enter the
softmax scaled 1/sqrt(d), NO affine calibration (tail_score_calibration=none).
Full scan every step: all sealed pages. Page-skip variants are validated
NEGATIVE (accepted tokens are spread across all pages).

## 3. Ranking and rung grid

Tokens ranked by PQ logit descending (base tokens excluded, always selected).
Budget rungs are fractions of context length:
- K: {0.10, 0.30, 0.50, 0.70, 0.90, 1.0}
- V: {0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0}
Finer grids are Pareto-NEUTRAL (validated); do not add rungs for quality.
A selected-K prefix at rung ki = base ∪ prefix, where prefix = the first
min(ceil(K_frac·ctx), n_eligible) tokens of the ranked list AFTER
excluding base tokens (base rides ON TOP of the budget, never inside
it — `_selected_for_budget`, run_layer_quality_eval.py). V analog:
v_exact_count = min(ceil(V_frac·ctx), ctx) raw, chosen globally by risk.
Confirmed against RTL 12/12 golden fit, issue #7 (2026-07-07).

## 4. Budget controller (the novel control block)

State: (ki, vi) rung indices. All output comparisons use
relL2(a,b) = ||a-b||_2 / max(||b||_2, eps) on the 128-dim attention output.

1. **Predict**: start rung from proxy softmax mass: smallest count whose
   softmax (over PQ logits/sqrt(d)) prefix mass >= 0.9
   (`_softmax_prefix_count`, line ~775). v_start = max(v_budgets[0],
   0.25 * k_start_target). Start strategy is a latency optimization only —
   the controller is start-insensitive (validated).
2. **Stability test** between rung r and r+1 on one axis: delta =
   relL2(out(r), out(r+1)); threshold = tau * clamp(sqrt(band_frac / 0.2),
   0, 1.5) where band_frac = (budget(r+1)-budget(r))/ctx
   (`scaled_threshold`, line ~812). **tau = 0.004 (FROZEN 2026-07-06)**: task-validated end-to-end on the
   GPU frontier (RULER niah_multikey_2/vt/niah_single_1 32k, niah_single_1
   128k - all 100.0); 0.008 also passed everywhere tested and is headroom,
   not adopted. OPEN-1 is RESOLVED.
3. **Escalate walk**: k_first_alternating — alternate preferred axis each
   step, escalate the preferred failing axis, stop when both axes pass
   (`choose_action` line ~531, `simulate_policy` line ~576).
4. **De-escalate walk** (after stop): repeatedly step DOWN any axis whose
   adjacent-band delta is within its scaled threshold (same formula). The
   same pair-delta governs both directions => no oscillation. Validated
   Pareto improvement (-5.7% MB @0.002, -16.6% @0.004). simulate_policy
   `deescalate=True` branch.
5. Output at settle = out(ki, vi). Down-probes MUST be implemented via
   stored per-band partial accumulators (max, sum, acc[128]) — recombine,
   do not re-read DRAM.

## 5. Mixed attention output at rung (ki, vi)

- Selected-K tokens: exact logits (K row read; fp16 or int8 tier per Sec. 6).
- Non-selected tokens: PQ logit / sqrt(d) (raw, uncalibrated).
- Softmax over base + all tokens; flash-style accumulation is REQUIRED to be
  numerically equivalent to the reference within fp32 accumulation.
- V side at rung vi: tokens ordered by risk_i = p_i^2 * vpq_code_error_i
  (p = current softmax prob). Top-(V_frac*ctx) get exact V rows; the rest of
  the SELECTED set uses V-PQ reconstruction; base tokens always exact.
  Two-pass threshold variant (threshold from pass-1 stats, tile-stream
  pass-2) is validated equivalent — the tile-compatible form RTL should use
  (`two_pass_risk` rule; see current_status.md 2026-07-05 Two-Pass section).

## 6. Progressive precision (int8 tier)

Validated form: **per-row symmetric absmax int8** — q8(x_row) =
round(x_row/s)*s, s = absmax(x_row)/127, stored as int8 codes + fp16 scale
(`_quantize_rows_symmetric`, line ~1045).
- K: ranked prefix beyond the top hi_frac=0.1 of the selected set reads the
  int8 tier (128 B + scale instead of 256 B). No commit test needed.
- V: exact-V reads beyond the top hi_frac=0.1 by risk read int8 ONLY where
  `int8_err[i] < vpq_code_error[i]` (commit test, per token); tokens failing
  the test keep V-PQ (no row read at all). Naive int8-for-all-V is a
  validated NEGATIVE (outlier-absmax rows).
- Measured: -31 to -34% total MB at identical trace relL2; 4-bit lo tier is
  NOT free (relL2 0.007-0.018) — do not implement without new validation.
- **OPEN-2 RESOLVED (2026-07-06, M6): int8 dual-plane storage.** K/V rows
  are stored as two int8 planes at fp16-equivalent capacity: plane A =
  per-row symmetric absmax int8 of x (the lo tier, read alone); plane B =
  per-row symmetric absmax int8 of the residual x − A (hi/exact tier reads
  A+B; reconstruction error ~absmax/127², below fp16 rounding in practice).
  Two fp16 scales per row (A-scale, B-scale) in a separate dense scale
  array. Rejected alternative: reading the literal fp16 high byte as the lo
  tier FAILS (relL2 0.087 vs 0.004 — 2 effective mantissa bits; job
  `precision_fp16msb_smoke`). Full-spectrum confirmation (job 53003123,
  `storage_dualplane_deesc_prec/`): 2.8611 MB/head-query @ tau=0.004, max
  relL2 0.00875, vs fp16-storage golden 2.8573 / 0.00871 — +0.13% MB, quality
  identical. Layout details (plane placement, scale array) per issue #2:
  separate contiguous A and B regions (lo-tier stream stays dense 1 B/elem;
  interleaving would double real lo-tier bandwidth vs the trace-MB curves);
  scales as a separate dense array. NOTE: the golden model charges ZERO
  bytes for scales — add ~1.6% (2 B/row lo, 4 B/row hi at d=128) as an
  explicit sidecar adder in any RTL bandwidth budget.

## 7. Explicit non-goals (validated negatives — do not build)

- Temporal selection reuse (any budget policy) — stale tail logits poison
  the stability test.
- Page-skip / coarse-tier scanning; global or shared PQ codebooks.
- Fine rung grids; escalate-only warm starts (ratchet).
- Blind budget carry (frozen budgets): task-unsafe at long staleness.

## 8. Golden vectors for RTL verification

Reproduce the golden CSV on this machine (CPU, deterministic given trace +
seed; beware torch-bundled-numpy interop if editing the runner):

```
scripts/run_joint_kv_budget_policy_eval_one.sh with:
  DECODE_LENGTHS=all HEADS=0,8,16,24 STABILITY_THRESHOLDS=0.002,0.004
  POLICIES=k_first_alternating START_STRATEGIES=proxy_mass_m0p9
  BUDGET_DEESCALATE=1 [PRECISION_K_HI_FRAC=0.1 PRECISION_V_HI_FRAC=0.1]
```
Golden runs already in-repo (gitignored artifacts, regenerate as needed):
`attention_efficiency_result/joint_kv_ladder_grid_20260706/ladder_deescalate/`
and (composition) `deesc_precision_compose/`. Per-row fields to match:
`selected_k_tokens`, `v_exact_reads`, settled `k_budget`/`v_budget`,
`policy_trace` (rung walk sequence), `step_MB_per_head` (byte accounting),
`head_attention_relative_L2` (output correctness, fp tolerance 1e-3 rel).
Block-level goldens: PQ logits and ranked order can be dumped by
instrumenting the runner at the `rank_paged_pq` call — coordinate with the
algorithm side rather than re-deriving.

## 9. Change control

The algorithm side owns this file. RTL discrepancies against the golden
model are bugs on whichever side diverges from THIS document; if the
document is ambiguous, fix the document first.
CALIBRATION BOUNDARY: tau=0.004 and the rung-grid fractions are
calibrated on evidence up to ctx 134.8k; hardware targets 1M+ (issue #8).
Interim risk assessment: `notes/ctx_scaling_1m_memo.md` (e4m3 range
low-risk; proxy-mass clamp becomes the mainline path at 1M; boundary
ties ~linear). Real 1M validation planned:
`benchmark_differentiation_plan.md` Phase E. OPEN items: none.
RESOLVED: logit-buffer FORMAT = **fp8-e4m3 FROZEN** (issue #6 ack
2026-07-07; job 53008051: quality equal-or-better than absmax int8 —
max relL2 0.00784 vs 0.00864 @ tau=0.004 — at +0.93% aggregate MB
concentrated in head 0 +7.5%; scale-free write-during-scan, monotone
code doubles as the 256-bin histogram index; absmax int8 remains the
validated fallback); M4 (8-bit logit buffer FREE at full spectrum, job 53003124:
±0.05% MB, relL2 identical; buffer is 1 B/token); M5 (256-bin histogram
select + exact boundary-bin refine, exact fractional prefix counts, golden
CSVs authoritative — issue #4); OPEN-1 (tau=0.004 frozen); OPEN-2 (int8 dual-plane storage, Sec.
6 — fp16-equivalent capacity, +0.13% MB, quality identical); M2 (GQA union
factors: K 0.35-0.44, V 0.49-0.57 across the 4-head group, rising with
context); M3 (deesc x precision composition golden run
`deesc_precision_compose/`: 2.857 MB/head-query at tau=0.004 on the
288-position spectrum).
