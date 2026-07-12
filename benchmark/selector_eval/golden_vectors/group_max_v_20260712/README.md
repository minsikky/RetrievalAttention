# Group-max V-selection evaluation — issue #20 (2026-07-12)

Offline evaluation of the RTL **group-max ranking** proposal on the #20 thread:
rank tail V tokens by **group-max risk = max over the 4 GQA group heads of the
per-head frozen scan-domain risk** (`p_pq^2 * V_error`), take ONE rank cutoff at
a single group budget `B_grp`, and commit the top-`B_grp` set **apply-to-all**
(any committed token applies to all 4 heads — ratified #20 union-commit
consumption). Goal under test: collapse the 4 per-head V rank histograms to ONE
per group without losing quality at matched bytes.

## Producer & substrate

- **Producer:** `benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py`
  with the additive flag `--gqa_group_max_eval` (requires `--gqa_union_commit`).
  Device = **cpu**, no GPU, no model forward — reads the cached QKV trace
  `attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz`
  (Llama-3.1-8B-Instruct, layer 16; keys/values 8 kv-heads × 137,909 × 128,
  queries 32 × 288 × 128). Job: `scripts/run_group_max_v_eval.sbatch`
  (standard partition, account zhengya0), Slurm 53345653/54/55/56.
- **Frozen operating point** = the `stage2_union_commit_20260710` config verbatim
  (escalation-only `k_first_alternating`, `proxy_mass_m0p9`, τ=0.004 sqrt-scaled,
  progressive precision (0.1,0.1), `--precision_split_freeze start`, fp8-e4m3
  logit buffer, Q7.6 two-pass grid) PLUS `--gqa_union_commit --gqa_group_max_eval`.
- **Positions:** qidx {137, 159, 223, 262, 283, 287} = ctx {12,000; 14,838;
  38,838; 83,225; 126,580; 134,838}, **all 32 q-heads** per position (the ratified
  #20 envelope population). Measurement-1 positions per the RTL ask are
  q137/q262/q283/q287 + the 135k boundary (=q287); q159/q223 add the
  golden-cross-check positions.
- **GQA mapping (verified from `trace.py:33`):** kv_head = head // 4; group = 4
  consecutive q-heads (0-3→kv0, 4-7→kv1, …).
- **Risk (verified from producer):** `risk = p_pq^2 * V_error`. `p_pq` = per-head
  scan-domain softmax weight; `V_error` = V-PQ code-averaged reconstruction error,
  **group-shared** across the 4 heads. The frozen per-head committed-V set is
  exactly `top-B_h` of this risk (rank cutoff); `B_h = v_budgets[settled_vi]`.
  Group-max risk = `max_h(p_pq,h^2) * V_error`.
- **relL2:** per-head attention-output relL2 vs the exact dense head
  (`dense_attention_output`: exact softmax weights, exact V), reusing
  `gqa_union_commit_head_output` verbatim with a group-max V set in place of the
  union V set (committed tokens → exact V, others → PQ-reconstructed V, weighted
  by the head's scan-domain probs). Same machinery as the #20 ratification run.

## Sanity gates (both PASS)

1. **#12 reproduction** (static theta-lab CSVs): global theta 2e-11 → **0.6846**
   of 9,216 head-steps within 1.05× canonical relL2 at 1.046× aggregate bytes
   (matches the committed #12 headline 68.5% / 1.05×).
2. **Machinery gate** (2,756 curve rows): group-max harness reproduces the frozen
   per-head baseline **bit-for-bit** (max |baseline − repro| = 0.0) and recovers
   each frozen committed-V set as `top-B_h` of its own recomputed risk
   (top-B_h recall = **1.0000**). Baseline arm/group-max arm therefore share one
   per-run codebook; the comparison is internally exact.
   *Cross-check vs `stage2_union_commit_20260710`:* committed-set sizes match the
   d3c9501 goldens exactly (e.g. q159 group0 [5936,1484,2968,742]); token
   identities drift 0.47% (union 6323 vs 6353), shifting absolute baseline relL2
   ≤5.5e-4 — a build-order V/K-PQ codebook artifact (isolated single-qidx runs
   seed the sealed-page cache differently than the golden's sequential
   multi-position run), at the documented-acceptable V-PQ level. Does not affect
   the group-max-vs-baseline delta (both arms use the identical per-run codebook).

## Files

- `group_max_report.json` — full report (meta, sanity, meas 1-4, OR-theta).
- `group_max_tables.csv` — per-position headline (coverage, envelope, miss mass, OR-theta).
- `group_max_budget_rule.csv` — measurement-3 budget-rule candidates.
- `gqa_group_max_curve.csv` — raw per (position, head, B_grp-candidate) rows:
  `coverage_recall`, `groupmax_relL2_frozenK` (binding: frozen-K + group-max-V),
  `groupmax_relL2_unionK` (deployment: union-K + group-max-V, headline budgets
  only), `baseline_relL2`, `repro_baseline_relL2`, `repro_topBh_recall`,
  `missed_tokens`, `missed_weight_mass`, `committed_weight_mass`,
  `total_weight_mass`, `Bh/B_union/B_sum/B_max`. Budget grid: `union` (=|group V
  union|, matched bytes), `sum` (ΣB_h), `max` (max_h B_h), `max_x{1.1..4}`,
  `sum_x{0.6..1.0}`.
- `gqa_group_max_ortheta.csv` — OR-theta curiosity (single global theta OR'd
  across the group == threshold on group-max risk; θ ∈ {1,2,3}e-11).
- `scripts/run_group_max_v_eval.sbatch`, `scripts/analyze_group_max_v_select.py`
  — producer job + reducer (both on branch `group-max-v-20`).

## Verdicts (see the #20 thread post)

- **Coverage — conditional.** Group-max top-|union| recovers the frozen per-head
  committed sets well in aggregate (macro recall 0.90–0.96, micro 0.92–0.97) but
  not per-head (min 0.065–0.44): some heads lose most of their committed tail.
- **Envelope (binding gate) — NEGATIVE.** Group-max apply-to-all at matched bytes
  FAILS the ratified #20 envelope: aggregate mean/p95 non-monotone at 3/6
  positions (39k, 83k, 135k) and **55 above-tie per-head violations across 29/32
  heads** (worst +1.22e-3 / +248% at 39k h21), well outside the ctx<32k /
  +6e-4 / +15% band. The single group cutoff starves "quiet" heads whose
  high-risk tokens are outranked by louder group-mates.
- **Budget rule — NEGATIVE for a tight single-rung rule.** `|union|` is the only
  byte-matched point but presupposes the union; `ΣB_h` is safe at **2.11×** bytes;
  a fitted rule from the four B_h under-provisions ~half of held-out groups
  (p95 residual ~6,000 tokens). No single-rung rule reliably hits |union|.
- **Prefetch fallback — POSITIVE (#21 shape).** As a prefetch ranking with a
  per-head commit backstop, group-max is strong: 93.1% mean recall, missed tokens
  are ≤0.15% of total attention mass on average (though up to 90% of an outlier
  head's *committed* correction mass — load-bearing, not pure tail), and the
  group-max + union-K arm improves aggregate relL2 −8.9e-4. Group-max is a good
  prefetch score; exactness needs the fetch-on-miss backstop.
- **OR-theta (curiosity, threshold family) — NEGATIVE, as expected.** A fixed
  global theta OR'd across the group is non-stationary: 1.6–1.7× bytes at short
  ctx, starves to 0.61–0.65× at long ctx with quality regressing (83k/126k/135k).
  Inherits the #12 negative. Note: a *per-group rank-matched* OR-theta is
  identically group-max ranking — the threshold negative is specifically about
  FIXING theta globally, not about the group composition.
