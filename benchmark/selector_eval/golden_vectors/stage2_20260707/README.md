# Stage-2 goldens: controller / S5 / S6 (issue #7) — 2026-07-07

Same 12 trace rows as `../s2_s3_20260706/` (heads 0/8/16/24 × ctx
14.8k/38.8k/134.8k). **Producing config = the frozen operating point**:
ESCALATION-ONLY controller (de-escalation REMOVED from the frozen
algorithm 2026-07-07 — see spec §4; regen job 53061660), k_first_alternating,
proxy_mass_m0p9 start, τ = 0.004 (sqrt-scaled), progressive precision
(0.1, 0.1), logit buffer **fp8-e4m3** (frozen per issue #6). One npz per
row (`golden2_q<qidx>_h<head>.npz`) plus one V-PQ page block per
(kv_head, ctx) (`page_v_ctx<ctx>_kv<kv>.npz`).
Regen provenance: the escalation-only traces are byte-identical PREFIXES
of the previous (de-escalating) goldens truncated at the escalate stop —
verified per row on every probe array; only kd/vd tails were removed.
The 4 rows that never de-escalated are bit-identical to the previous
set; page blocks bit-identical.

## golden2 npz fields

Identity/config: `qidx head kv_head position context_len dynamic_start
sealed_end page_size head_dim threshold policy start_strategy k_budgets
v_budgets`.

**Item 1 — controller trace** (FSM transition golden): parallel arrays
`probe_kind` (k / v / stop), `probe_ki, probe_vi` (rung state
AT PROBE TIME, i.e. before the action the entry records), `probe_dk,
probe_dv` (the adjacent-band relL2 deltas the test read), `probe_tk,
probe_tv` (the sqrt-scaled thresholds compared against).
`start_ki/start_vi` → escalation walk → `stop` = `settled_ki/settled_vi`
(the walk is escalation-only; kd/vd probe kinds no longer exist —
de-escalation removed 2026-07-07). All entries read both axis deltas.

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
must equal `base_output_fp32` (also stored, = probs @ vhat).

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
`v_risk_key_q_fp64` = E6M12 map of risk_hw = w17² · ce16 — the fp64
product is EXACT (≤ 45 mantissa bits). The built hardware map (E=6, M=12;
`e_pre = floor(log2(v))`, `e_post = floor(log2(RNE13(v)))`,
`ebase = e_pre(max positive risk) − 62`):

| e-field | m-field | meaning | membership / value |
|---|---|---|---|
| e=0 | m=0 | structural zero bin | v == 0 (w17==0 or ce16==0) → key 0 |
| e=0 | m=1 | below-window bottom tie class | `e_post ≤ ebase` → key `2^ebase` (one class) |
| e∈[1..62] | mantissa-bearing | in-window octaves | key `q = RNE13(v)` at v's own exponent |
| e=63 | top tie class | anchor-octave carry | `q = 2^(e_pre_max+1)`, ranks top, order-exact |

Membership is decided POST-round (`e_post`), so the 2^E−2 = 62
mantissa-bearing octaves span `e_post ∈ (ebase, e_pre_max]`; the bottom
bin `2^ebase` is strictly below the smallest in-window key `2^(ebase+1)`
(no aliasing), and a mantissa carry at the anchor octave yields
`2^(e_pre_max+1)` with no special case. Ranking: key descending, stable ⇒
ties in stream order. Commit domain (fp16-vs-fp16, hardware-pinned): the
lo-tier int8 plane commits where `fp16(int8_err) < fp16(code_error)`,
STRICT, both RNE casts of the fp64 sidecars. Edges (natural under IEEE
fp16 compare on nonnegatives): `ce16 == 0` loses; an `int8_err` that
underflows fp16 to 0 with `ce16 > 0` wins; an exact fp16 tie loses
(strict <). Masks/scalars mirror item 5 in the key domain:
`v_exact_mask_key_packed`, `v_hi_mask_key_packed`,
`v_lo_mask_key_packed`, `v_dropped_reads_key`, `v_risk_key_cutoff_q`,
plus `v_risk_key_exp_bits`/`v_risk_key_mantissa_bits` (= 6/12).

## page_v npz fields (one per kv_head × ctx, LAST sealed page)

`value_codebook_fp32` (subvecs × 2^subbits × subdim = 1 × 16 × 128),
`value_codes_u8` (page_size × 1), `code_error_fp64` and `int8_err_fp64`
(the two 2 B/token sidecars, dumped fp64 over the page range; hardware
stores them fp16 — these are the pre-rounding reference values),
`page_start`, geometry. Reconstruction: vhat[t] = codebook[0, code[t]];
risk uses code_error; the V commit test is
`fp16(int8_err) < fp16(code_error)` (strict; the stored sidecars are the
pre-rounding fp64 reference values).

## Item 7 — Vcorr (V-correction) goldens

New fields the dump helper asserts at write time; the RTL harness pins the
hardware V-correction against them. Emitted only when the lo tier is live.

**Change 2b — full-context commit sidecars**: `v_commit_mask_packed`
(np.packbits of `fp16(int8_err) < fp16(code_error)` over the whole
context), `v_int8_err_fp16` (fp16 cast of the per-token int8-error stat;
`v_code_error_fp16` from item 6 is the code-error twin).

**Dual operand domain (RTL pin 2026-07-07).** Every Vcorr accumulator and
dv is dumped in two operand domains: `_ref` uses fp64 raw-fp16 / plane-A
operands (the domain the controller trace was produced in); `_hw` uses one
fp16 RNE cast per (token, dim, tier) — hi-tier reads
`fp16(dualplane_recon − vhat)` (plane-A+B dequant summed fp64, single
cast), lo-tier reads `fp16(dequantA − vhat)`. Hi-boundary per-token
contribution rounds each tier before subtracting (`diff16_hi −
diff16_prev`, never one rounding of the algebraic difference).

**3a — per-probe band partials** (ragged, flat arrays + offsets). One
record per probe entry that READS the V axis: k/v/stop compare
(ki,vi)→(ki,vi+1), vd compares (ki,vi−1)→(ki,vi). Per record (length R):
`vcorr_probe_record_kind`, `vcorr_probe_ki`, `vcorr_probe_vi_lo`,
`vcorr_probe_vi_hi`, `vcorr_dv_ref_fp64`, `vcorr_dv_hw_fp64`;
`vcorr_acc_marginal_{ref,hw}` and `vcorr_acc_hiboundary_{ref,hw}` (R×128).
Marginal band = key-rank positions (N_lo, N_hi] (commit winners read the lo
operand, losers 0); hi-boundary band = positions
(ceil(0.1·N_lo), ceil(0.1·N_hi)] (tier upgrade lo/pq→hi). Band members are
flat with offsets: `vcorr_marginal_tokens` / `vcorr_marginal_p` indexed by
`vcorr_marginal_offsets` (R+1; record r spans [off[r], off[r+1])), same
trio for `vcorr_hiboundary_*`. Self-checks (raise on fail): in EACH domain
out(vi_hi) − out(vi_lo) == acc_marginal + acc_hiboundary (rel ≤ 1e-5); and
|REF dv − trace probe value| ≤ eps_band = 2e-5. The trace dv came from the
run's RISK-domain selection while item-7 composes the KEY-domain sets, so
boundary-token swaps legitimately perturb dv inside the band — the signed
delta is stored per record as `vcorr_dv_trace_delta` (length R) and the
regen verifier reports each row's max.

**3b — settled total**: `vcorr_settled_acc_{ref,hw}` (128) =
out(settled) − base(settled). Self-check: REF + base, cast fp32, matches
the item-5 `output_from_base_and_split_masks` path with the key-domain
masks.

**3c — settled exact-V operands** (key-rank order, length n_exact):
`vexact_tokens` (int64), `vexact_v_fp16` (raw rows, float16),
`vexact_int8_codes` (int8) + `vexact_int8_scale` (float16) /
`vexact_int8_scale_fp64` — plane A is the literal
`_quantize_rows_symmetric(values_np, 8)` (per-row absmax int8, **float32
scale**; the fp64 twin is that float32 value widened, so
codes·scale reproduces plane A bit-exactly only in float32);
`vexact_int8_err_fp16`, `vexact_commit` (uint8); plane B (residual plane,
mirrors `_int8_dualplane_rows`) `vexact_residual_codes` (int8) +
`vexact_residual_scale` (float16) / `_fp64`; `vexact_recon_max_abs_err`.
Self-check: dequant(A)+dequant(B) reconstructs v_fp16 within
0.5·scaleB + fp32 slack per element.

**3d — K-move fixup: RETIRED with de-escalation (2026-07-07).** The
`kmove_*` fields (den_old/den_new scalar rescale, crossing token lists,
post-move Vcorr) were gated on the first kd probe and are absent from the
escalation-only goldens. The K-move Vcorr REBUILD structure itself
survives in the algorithm for K UP-moves (same scalar rescale + crossing
fixups, direction reversed); if RTL wants golden vectors for the up-move
rebuild, the 3d block will be re-pointed at the first accepted K
escalation in a follow-up regen (pending their answer on #7).

The verifier (`verify_stage2_key_regen.py`) requires every item-7 field
(3d conditional on the row's trace having kd — vacuous in this set),
reports moved item-6 key-mask token diffs (frozen vs regen) instead of
asserting them, and flags (nonfatal) any record with |dv_hw − dv_ref| >
1e-5.

## Provenance / regeneration

`run_joint_kv_budget_policy_eval.py --golden_dump_stage2_dir <dir>` with
the frozen-config flags (see `notes/algorithm_spec_v1.md` §8/§9), or
`sbatch scripts/run_stage2_key_regen.sbatch` which also runs
`verify_stage2_key_regen.py` (hard-asserts the regen is a bit-exact
superset of this dir). The V-selection recompute in the dump helper
mirrors the selection block in run(); both live in the same file with a
keep-in-sync comment. History: key-domain fields (item 6) added
2026-07-07 as a pure superset — every pre-existing field verified
bit-identical to the original delivery. 2026-07-07 (later): regenerated
escalation-only after de-escalation was removed from the frozen
algorithm (job 53061660): 8/12 rows had kd/vd tails and their
settled-state fields moved accordingly; per-row prefix-identity check
confirmed every probe array is the previous trace truncated at the
escalate stop (fatal=0); 4/12 rows and all 12 page blocks bit-identical;
max |dv_hw − dv_ref| = 4.0e-7, flagged records 0.
