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
4. **De-escalation: REMOVED from the frozen algorithm (2026-07-07).**
   The walk is escalation-only; the step ends at the escalate-walk stop
   state. Rationale: under faithful walk accounting the down-walk saves
   zero DRAM (the bands it abandons were already read during the climb
   — the previously quoted -5.7%/-16.6% MB savings were a settled-state
   accounting artifact), and its quality is slightly WORSE than
   escalation-only at the same tau (relL2 0.00358 vs 0.00274 at 0.004)
   — strictly dominated on the (logical traffic, quality) frontier. Its
   only surviving property (start-insensitivity for warm-start designs)
   belongs to an unspecced budget-carry extension, not the frozen
   per-step algorithm. `--budget_deescalate` remains in the runner for
   reproduction of historical arms; canonical configs and goldens are
   escalation-only. kd/vd probes no longer exist.
5. Output at stop = out(ki, vi) at the escalate-walk stop state.
5b. **Probe-delta producers (pinned 2026-07-07, #7 thread)**. The
   compared quantity is out(ki,vi) = base_output(ki) + Vcorr(ki,vi),
   Vcorr = Σ_hi p·(v_fp16 − v_pq) + Σ_lo p·(v_int8 − v_pq), p = softmax
   row at ki. relL2 denominator = the RICHER rung's output (= the next
   rung; de-escalation removed 2026-07-07). V probes (fixed ki) are acc-only
   (numerator) differences over TWO risk-rank intervals: the marginal
   band (N_lo, N_hi] (commit-winners contribute v_int8 − v_pq; losers
   contribute zero) and the hi-boundary band
   (ceil(0.1·N_lo), ceil(0.1·N_hi)] (v_fp16 minus int8-or-pq per the
   static commit bit). Risk = p²·code_error is frozen across the V walk
   at a given ki and REBUILT whenever ki changes — so K probes (dk)
   change the K band, the normalization, AND the V correction (re-rank
   + p rescale); a frozen-V-set approximation during K probes is a
   different delta and is NOT covered by the eps_band clause without a
   measured decision-flip study. The int8 commit test is inside every
   compared output; the commit bit is per-token static.
   **K-move crossing structure (pinned 2026-07-07)**: on a K move, s_i
   changes on exactly two K-rank intervals — the marginal band
   (B_lo, B_hi] (e4m3 tail → int8 lo; the hi boundary cannot reach it
   for adjacent rungs) and the hi-boundary band
   (ceil(0.1·B_lo), ceil(0.1·B_hi)] (int8 → exact). All other exp(s_i)
   are bit-identical, so p rescales by one scalar (den ratio), the risk
   order among non-crossers is invariant, and the exact Vcorr rebuild =
   scalar rescale + crossing fixups + rank-boundary deltas. Exact — not
   an approximation (so the frozen-V-set study above is moot; RTL
   adopted the exact rebuild). [2026-07-07: kd removed with
   de-escalation; the rebuild applies to K UP-moves only.]
   **Exact-domain weight budget**: exact-tier w = exp(s − band_max) must
   land within 2^-17 relative of the fp64 reference after a single RNE
   quantization (same budget as bin-domain w17; requires fp32-class
   pre-quantization exp error). The eps_band = 2e-5 study bounds
   per-token weight error domain-independently, so no new guard band.
   Golden reference: fp32 exp at run level, fp64 in Item-2 partials.
6. **Decision guard band (FROZEN 2026-07-07, issue #7 thread)**: the RTL
   probe path quantizes the per-token weight w = exp(s − s_maxbin) once
   to 17 bits (RNE, exact fixed-point accumulation), giving hardware
   probe deltas within ≤ ~2e-5 absolute of the fp64 reference.
   Comparisons with |delta − threshold| < eps_band = **2e-5** are
   implementation-defined: either decision is legal; an in-band flip
   moves the walk at most one rung on that axis, and every legal settled
   state satisfies the stopping predicate within eps_band (quality
   impact bounded by ~tau by construction — at-threshold rungs are
   equivalence-class members, not errors). Measured at tau = 0.004
   across 4,704 canonical decode steps (ladder/compose/e4m3 sweeps +
   96-row scan): 2.00% of steps contain >= 1 in-band comparison; a 1e-4
   band was REJECTED as needlessly loose (7.93% of steps). The 12
   stage-2 golden rows have min margin 1.69e-4 — every golden decision
   is deterministic under the band; trace-replay validation is
   unaffected. Any future quantization change that widens the hardware
   delta error beyond 2e-5 must renegotiate this clause explicitly.

## 5. Mixed attention output at rung (ki, vi)

- Selected-K tokens: exact logits (K row read; fp16 or int8 tier per Sec. 6).
- Non-selected tokens: PQ logit / sqrt(d) (raw, uncalibrated).
- Softmax over base + all tokens; flash-style accumulation is REQUIRED to be
  numerically equivalent to the reference within fp32 accumulation.
- V side at rung vi: tokens ordered by risk_i = p_i^2 * vpq_code_error_i
  (p = current softmax prob). Top-(V_frac*ctx) get exact V rows; the rest of
  the SELECTED set uses V-PQ reconstruction; base tokens always exact.
  **Two-pass scan-domain cutoff ADOPTED as the RTL V-selection rule (issue #9,
  2026-07-09).** Pass-1 rides the selector scan and keeps a scalar cutoff =
  the (V_frac·ctx)-th largest log-risk `2·logit + log(vpq_code_error)`
  (scan/approximate logits: PQ for non-resident, exact for resident base;
  zero-error tokens → −∞, never committed). Pass-2 commits exact-V tile-locally
  iff `log_risk ≥ cutoff`. **Fixed-point cutoff: Q7.6 — int_bits=7 (integer
  field width, sign included: clamp ±2^(int_bits−1) = ±64), frac_bits=6 (LSB
  2⁻⁶, additional), round-to-nearest-even → total register = int_bits +
  frac_bits = 13-bit signed.** Clamp dead on observed data (log-risk range
  [−42.2, +3.8]). Precision from the cutoff sweeps: frac=6 job 53138777,
  int-width job 53183387 (±16 over-commits +722% reads, ±32 empirical floor,
  ±64 adopted with 1M margin; re-validate at Phase E 1M).
  Ranks identically to `p²·err` (`log_risk = log(p²·err) + const_step`), so it
  selects the same set as the item-6 key while moving the cutoff to scan time
  and pass-2 to a tile-local compare (breaks the post-walk RANK serialization,
  tiles for multi-page). Quality-neutral: cutoff-precision sweep job 53138777
  (f=6: relL2 within 0.1%, walk-MB within 0.06%, exact-V reads +0.28% vs fp64);
  rule validation jobs 52950302/52950295 (+0.5% MB, relL2 identical vs global
  residual-risk). Golden fields `two_pass_*` per §8 / golden README item-8
  (per-step cutoff + committed set + pass-1/2 operands + bit-exact rebuild
  gate).

## 6. Progressive precision (int8 tier)

Validated form: **per-row symmetric absmax int8** — q8(x_row) =
round(x_row/s)*s, s = absmax(x_row)/127, stored as int8 codes + fp16 scale
(`_quantize_rows_symmetric`, line ~1045).
- K: ranked prefix beyond the top hi_frac=0.1 of the selected set reads the
  int8 tier (128 B + scale instead of 256 B). No commit test needed.
  **Hi/lo split is FROZEN once per step** at the start rung's hi_count =
  ceil(hi_frac·kb[start]) (`--precision_split_freeze start`, adopted
  2026-07-07 for RTL #2 Q1), NOT recomputed per K rung. This removes the
  mid-walk plane-B upgrade re-fetch path (hardware simplification) at no
  quality cost: on the escalation-only population (job 53070218, 96
  head-steps) freezing is quality-neutral (meanL2 +0.06%) and slightly
  cheaper on walk traffic (−2.40%; the growing split accrues plane-B
  upgrades on deep low-weight rungs that do not affect the output).
  Freezing at ceil(hi_frac·max(k_budgets)) (`kbmax`) is dominated
  (+6.4% walk for a noise-level quality change). Golden impact of the
  switch (job 53070220 vs the escalation-only goldens): 11/12 rows move
  only `probe_dk` (recorded stability margins shift with the int8-QDQ
  ranking; no escalation decision flips, masks/outputs bit-identical); 1
  row (q159_h8) has a selected token cross the frozen-vs-growing boundary
  → outputs + Vcorr move, key-selection masks unchanged, dv 2.1e-7.
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
  PRECISION_K_HI_FRAC=0.1 PRECISION_V_HI_FRAC=0.1
```
Golden runs already in-repo (gitignored artifacts, regenerate as needed).
Historical de-escalation runs such as `ladder_deescalate/` and
`deesc_precision_compose/` are reproduction artifacts only; current stage-2
RTL goldens are escalation-only under
`benchmark/selector_eval/golden_vectors/stage2_20260707/`. Per-row fields to match:
`selected_k_tokens`, `v_exact_reads`, settled `k_budget`/`v_budget`,
`policy_trace` (rung walk sequence), `step_MB_per_head` (byte accounting of
the SETTLED state — well-defined as a golden match target, but NOT faithful
walk traffic; `walk_step_MB_per_head` charges the deepest band read per axis
and is the field for any DRAM/bandwidth claim, commit 48e9fd9),
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
RESOLVED: **de-escalation REMOVED (2026-07-07, user decision)** — walk is
escalation-only (§4 item 4): zero walk-basis DRAM value, slightly worse
quality, strictly dominated; kd/vd probes and the golden kd rows retired,
stage-2 goldens regenerated escalation-only;
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
context); M3 (precision composition measured on the historical
`deesc_precision_compose/` artifact; de-escalation subsequently removed):
settled 2.857 MB/head-query at tau=0.004 on the 288-position spectrum;
faithful walk traffic = **4.509 MB/head-query** (job 53051141), see
hw_arch Sec. 5 correction).
