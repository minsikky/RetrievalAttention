# V-threshold lab — issue #12 full-sweep report (2026-07-11)

`report_12_full.json` is the committed table behind the issue #12 verdict on
absolute-risk V-selection (fixed theta on deployable scan-domain risk
`p_pq^2 * V_error` vs the canonical global top-B relative ranking).

Substrate: 9,216 head-steps = 288 decode positions x 32 heads, layer 16,
contexts 6,839-134,838, frozen operating point. Producer: the V-threshold lab
in `run_joint_kv_budget_policy_eval.py` (commits a3d0cd9, 4f9f67f), sweep jobs
53292148/53292497 + analyzer 53294376 (`scripts/run_v_threshold_lab_sweep.sbatch`,
`scripts/analyze_v_threshold_lab.py`).

Contents: canonical baseline stats (`canonical_global_topB`), 20-point theta
grid 1e-14..1e-6 (`per_theta`: counts, logical V bytes, relL2 incl. noSplit
variant, per-head-step ratio vs canonical, false-negative/positive risk mass),
matched-bytes and matched-quality operating points, per-octave threshold
sensitivity, count variance at matched bytes, and the context-bucket breakdown
at the matched-bytes theta.

Verdict (full text on issue #12): fixed-theta CONFIRMED NEGATIVE — mean quality
is matchable at matched bytes (theta 2e-11) but the per-head-step distribution
breaks (68.5% within 1.05x canonical; long-context buckets starved 1.35-1.51x
while short contexts are over-provisioned). The same logs support the #13
(per-KV-head calibration; per-layer NOT fittable from this single-layer trace),
#14, and #16 offline analyses.

Raw per-head-step logs (~GBs) stay in
`attention_efficiency_result/v_threshold_lab_20260711/` (gitignored, on-request).
