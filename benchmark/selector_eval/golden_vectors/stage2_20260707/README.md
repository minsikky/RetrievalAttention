# Stage-2 goldens: controller / S5 / S6 (issue #7) — 2026-07-07

Same 12 trace rows as `../s2_s3_20260706/` (heads 0/8/16/24 × ctx
14.8k/38.8k/134.8k). **Producing config = the frozen operating point**:
de-escalating controller, k_first_alternating, proxy_mass_m0p9 start,
τ = 0.004 (sqrt-scaled), progressive precision (0.1, 0.1), logit buffer
**fp8-e4m3** (frozen per issue #6). One npz per row
(`golden2_q<qidx>_h<head>.npz`) plus one V-PQ page block per
(kv_head, ctx) (`page_v_ctx<ctx>_kv<kv>.npz`).

## golden2 npz fields

Identity/config: `qidx head kv_head position context_len dynamic_start
sealed_end page_size head_dim threshold policy start_strategy k_budgets
v_budgets`.

**Item 1 — controller trace** (FSM transition golden): parallel arrays
`probe_kind` (k / v / kd / vd / stop), `probe_ki, probe_vi` (rung state
AT PROBE TIME, i.e. before the action the entry records), `probe_dk,
probe_dv` (the adjacent-band relL2 deltas the test read), `probe_tk,
probe_tv` (the sqrt-scaled thresholds compared against).
`start_ki/start_vi` → escalation walk → `stop` → de-escalation walk →
`settled_ki/settled_vi`. Escalation entries (k/v/stop) read both axis
deltas; de-escalation entries (kd/vd) read only their own axis — the
other axis's delta/threshold are NaN. A `kd` at probe state (ki, vi)
moves ki→ki−1 (vd analogously); the state after the last entry equals
`settled_ki/settled_vi`.
**Coverage:** 8/12 rows de-escalate at the frozen operating point
(de-escalation is the COMMON case, not the exception: the escalation
phase stops one rung above where the down-walk settles whenever the
last band it read contributed under threshold). All five probe kinds
appear in this set.

**Item 2 — band partials** at the settled K state (S5 recombine golden):
`band_labels` = [base, band0..band<settled_ki>, tail]; per band:
`band_count`, `band_max` (max mixed logit in the band, −inf if empty),
`band_sumexp` (Σ exp(s − band_max)), `band_acc` (Σ exp(s − band_max) ·
vhat_row, fp64, shape n_bands × 128). Bands partition the context: base =
resident tokens outside [dynamic_start, sealed_end); band_i = ranked
selection rung i minus rung i−1; tail = never-read sealed tokens (their
logits are the e4m3-quantized, calibrated PQ values — the S2 buffer
domain). Logit domain = the golden model's mixed score row: exact logits
for hi-tier selected, plane-A int8 logits for lo-tier selected, quantized
PQ for the tail; all in the post-1/sqrt(d) domain.
Combine identity (self-checked at dump time, `combine_rel_err` stored,
asserted < 1e-5): with M = max_b band_max,
  output = Σ_b band_acc_b·exp(band_max_b − M) / Σ_b band_sumexp_b·exp(band_max_b − M)
must equal `base_output_fp32` (also stored, = probs @ vhat). A
de-escalation replays this with one band REMOVED — subtraction, not
accumulation.

**Item 3 — tail sum**: the `tail` row of the band arrays. The histogram-
derived tail unit (bin counts × exp table) must match `band_sumexp[tail]`
after aligning on `band_max[tail]`; fp32 tolerance per the #5 contract.

**Item 4 — proxy-mass scalars**: `proxy_mass_c` (the softmax-0.9 crossing
count from the literal `_softmax_prefix_count`), `start_ki`, `start_vi`.
Reminder: k_target = max(k_budgets[0], c); v_target = 0.25 × the CLAMPED
k_target (mainline path at 1M — see ctx_scaling_1m_memo.md).

**Item 5 — V path** at the settled pair: `risk_scores` (fp64, full
context; = probs² × vpq_code_error at settled ki), `v_exact_count`,
`v_risk_cutoff` (min risk inside the selected set), and np.packbits masks
`v_exact_mask_packed` (global top-count by risk), `v_hi_mask_packed`
(top 10% of that set by risk rank — exact fp16 reads),
`v_lo_mask_packed` (int8 commit-test winners), `v_dropped_reads`
(commit-test losers — they keep V-PQ and read nothing). The EFFECTIVE
exact-read set is hi ∪ lo, which may be smaller than the top-count mask.
Unpack: np.unpackbits(x)[:context_len].astype(bool).

**Item 6 — key-domain V selection** (frozen E=6/M=12 risk-rank key,
RTL-acked 2026-07-07; the contract the hardware ranker implements —
`validate_stage2` should hard-assert THESE masks, the item-5 fp-domain
masks stay as reference). Inputs, pinned to the golden's quantized-weight
domain and stored per token:
`v_w17_fp64` = RNE_17(probs / max(probs)) — floating 17-bit-significand
RNE in the value domain, 1.0 exact at the max token, common denominator
scaled out; `v_code_error_fp16` = fp16(code_error) (RNE cast;
range-verified: no overflow/underflow on these contexts). Key:
`v_risk_key_q_fp64` = E6M12 window quantization of w17² · ce16 — the fp64
product is EXACT (≤ 45 mantissa bits), so this equals a single RNE of the
exact integer product to 12 fraction bits, window = 64 octaves anchored
at floor(log2(max positive risk)); below-window positives share one
bottom-bin value 2^floor_e; zeros (w17 == 0 or ce16 == 0) stay 0 — the
zero set equals the structural non-paged set exactly. Ranking: key
descending, stable ⇒ ties in stream order. Masks/scalars mirror item 5 in
the key domain: `v_exact_mask_key_packed`, `v_hi_mask_key_packed`,
`v_lo_mask_key_packed`, `v_dropped_reads_key`, `v_risk_key_cutoff_q`,
plus `v_risk_key_exp_bits`/`v_risk_key_mantissa_bits` (= 6/12). The lo
commit test is unchanged (fp-domain static bit). fp-vs-key mask diffs on
these 12 rows: 2-token boundary swaps on 2 rows (q223_h16 hi<->lo,
q287_h16 exact-set), all at near-equal-risk cutoffs (in-band by the #7
uniform-bound argument).

## page_v npz fields (one per kv_head × ctx, LAST sealed page)

`value_codebook_fp32` (subvecs × 2^subbits × subdim = 1 × 16 × 128),
`value_codes_u8` (page_size × 1), `code_error_fp64` and `int8_err_fp64`
(the two 2 B/token sidecars, dumped fp64 over the page range; hardware
stores them fp16 — these are the pre-rounding reference values),
`page_start`, geometry. Reconstruction: vhat[t] = codebook[0, code[t]];
risk uses code_error; the V commit test is int8_err < code_error.

## Provenance / regeneration

`run_joint_kv_budget_policy_eval.py --golden_dump_stage2_dir <dir>` with
the frozen-config flags (see `notes/algorithm_spec_v1.md` §8/§9), or
`sbatch scripts/run_stage2_key_regen.sbatch` which also runs
`verify_stage2_key_regen.py` (hard-asserts the regen is a bit-exact
superset of this dir). The V-selection recompute in the dump helper
mirrors the selection block in run(); both live in the same file with a
keep-in-sync comment. History: key-domain fields (item 6) added
2026-07-07 as a pure superset — every pre-existing field verified
bit-identical to the original delivery.
