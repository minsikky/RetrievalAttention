# V-select calibration — issues #13/#14/#16 offline analyses (2026-07-11)

`report_calib_full.json` holds the tables behind the calibrated / predicted /
banded absolute-V-threshold verdicts. Pure offline reduction of the same
V-threshold lab logs behind #12 — no model, no torch, no GPU.

Producer: `scripts/analyze_v_select_calib.py` over
`attention_efficiency_result/v_threshold_lab_20260711/{sweep_A_0_199,sweep_B_199_288}/lab`
(the same 9,216 head-steps = 288 decode positions x 32 heads, layer 16, contexts
6,839-134,838, frozen operating point; risk = deployable scan-domain
`p_pq^2 * V_error`; canonical = global residual-risk top-B).

## Method
- Per head-step we log-theta-interpolate the measured 20-point theta grid
  (1e-14..1e-6) to score any per-head-step theta. `bytes = 256 * count`
  (head_dim 128 x fp16, constant). Interpolation fidelity (leave-one-grid-point-out,
  in `meta.interp_fidelity_leave_one_out`): relL2 rel-err mean 4.1% / p95 15.6%;
  count rel-err mean 12.5% / p95 49.5% (count is steep in theta at half-decade
  grid — byte ratios are ensemble means so this partly averages out; none of the
  verdicts are close calls at this noise).
- Sanity: the harness reproduces #12 exactly — global theta 2e-11 -> 68.5% of
  head-steps within 1.05x canonical relL2 at 1.05x bytes; theta 1e-14 (flood) ->
  98.7% at 4.67x bytes.
- **Splits are BY QIDX** (never split the 32 heads of one decode step; they share
  the prompt/context). qidx is context-sorted; lag-1 autocorrelation of the mean
  log cutoff is ~0.91, so:
  - **primary `blocked`**: contiguous blocks of 6 qidx alternated train/test ->
    both splits span the full 6.8k-135k range and all four context buckets, with
    train/test adjacency held to ~16% of qidx pairs (block boundaries only).
  - `even_odd` (balanced but fully autocorrelated) and `contiguous` (first 60% /
    last 40% qidx = a hard short->long context-extrapolation test) are reported
    alongside. Blocked and even_odd agree to ~2pp, confirming the low-capacity
    models (percentile tables, 22-feature linear regression) cannot exploit the
    autocorrelation; the contiguous split is the informative stressor.

## Headlines (primary blocked split, held-out test)
- **Non-stationarity budget** (`meta.variance_decomposition`): log10 canonical
  cutoff has std 1.23 decades, p10-p90 = 2.88 decades. Group-mean R^2: kv_head
  **0.116**, context_bucket 0.195, kv_head x context_bucket **0.332**, query_head
  0.280. Static identity+context explains ~1/3 of the non-stationarity.
- **#13 static tables** (matched bytes, frac within 1.05x canonical): global 0.650,
  kv_head 0.701, kv_head x ctx 0.543 (worse — over-fits under matched-byte
  calibration), query_head 0.748. Coverage ceiling 98.5% (int8-split flooding);
  98% coverage costs 2.9-4.1x bytes. Under the `contiguous` context-extrapolation
  split every static table **collapses to 0.16-0.20**.
- **#14 predictor** (linear on log-cutoff, 22 cheap scan features): test R^2 0.684,
  |err| mean 0.35 / p95 0.85 decade; matched-bytes frac **0.778**, recall 0.92 /
  precision 0.87 vs canonical. Beats every static table. Under `contiguous` it
  holds at **0.842** while static tables collapse. Adding genuine risk-distribution
  quantile features (histogram/CDF barrier) moves R^2 only 0.684->0.702 and does
  not improve matched-bytes coverage — the extra barrier buys nothing.
- **#16 two-tier band**: NOT achievable. No (theta_lo, theta_hi, M) hits >=99%
  within 1.05x at <=1.2x bytes. Centering the band on the #14 prediction, bracketing
  the cutoff for >=99% of head-steps needs a >=1.5-decade band = 47.6% of tokens
  (band count p95 ~20k, M for 99% no-overflow ~35k). At a 0.5-decade band
  (cutoff bracketed 76.5%) the band already holds a median ~1k / p95 6.8k tokens.
  Overflow->include reaches 0.97-0.99 quality but 1.25-2.5x bytes; overflow->drop
  holds bytes but quality falls to 0.35-0.77. The ambiguous population is 1-2
  orders above the 1-2% RTL target — the band relabels most of the array as
  ambiguous rather than removing the global-ranking barrier. Fixed global-center
  bands are strictly worse (cutoff bracketed only 62% at a 1-decade band).

## JSON layout
`meta` (interp fidelity, variance decomposition, canonical stats) and
`splits_reported.{blocked,even_odd,contiguous}` each with `issue13_static_tables`
(per-granularity calibrated tables, matched-bytes + bucket breakdown + coverage
ceiling), `issue14_predictor` (cheap + histogram models, error/recall/precision/
coverage), `issue16_band_predictor_center` and `issue16_band_fixed_center`
(half-width x M sweeps, band population, M-for-no-overflow, include/drop overflow).

Raw per-head-step logs stay in
`attention_efficiency_result/v_threshold_lab_20260711/` (gitignored, on-request).
