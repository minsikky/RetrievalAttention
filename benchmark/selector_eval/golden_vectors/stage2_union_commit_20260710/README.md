# Stage-2 goldens under UNION-COMMIT execution (issue #20, item 3) — 2026-07-10

Supersedes `../stage2_20260707/` as the RTL gate target; the old set REMAINS
checked in as the pre-union reference (do not delete). Producing config = the
frozen operating point of `../stage2_20260707/` (escalation-only controller,
k_first_alternating, proxy_mass_m0p9 start, τ = 0.004 sqrt-scaled, progressive
precision (0.1, 0.1), `--precision_split_freeze start`, fp8-e4m3 logit buffer,
adopted Q7.6 two-pass cutoff grid) PLUS `--gqa_union_commit`: the issue #20
union-commit execution policy, ratified 2026-07-10 on the #20 thread.
Regen: `scripts/run_stage2_union_commit_regen.sbatch` (job 53296998), branch
`union-commit-20`.

## Ratified semantics (issue #20, as confirmed 2026-07-10)

1. Per-head selection walks, budgets, and escalation are UNCHANGED (frozen
   contract). The union applies at execution only.
2. Exact-K set per head := union of the group's 4 committed K sets. Exact
   logits replace PQ logits for every union token in every head; softmax /
   Vcorr denominators update accordingly (same machinery, larger set).
3. (AMENDED) Per-token K tier := max tier across the group (any head hi →
   load planes A+B). Explicitly: a head can receive MORE hi-precision K rows
   than its own frozen split committed (observed e.g. 1336→1524); this is the
   intended broadcast behavior and changes that head's base output even where
   its committed K set is unchanged.
4. Exact-V set per head := union of the group's committed V sets; per-head
   Vcorr weights as usual over the larger set (committed V sets frozen from
   the walk — the denominator change reweights Vcorr but does NOT re-select).
5. (RESTATED) Union-commit is quality-monotone IN AGGREGATE, not per-head.
   **Numeric envelope (part of the frozen contract; re-validates on any
   operating-point or config change):**
   - aggregate mean AND p95 o-proj relL2 must improve (union < baseline) at
     every validated position;
   - per-head regressions permitted only at ctx < 32k, bounded at
     **≤ +6e-4 absolute and ≤ +15% relative** vs the frozen baseline;
   - **tie floor (proposed numeric amendment, pending RTL sign-off on the
     #20 thread):** deltas ≤ 1e-7 absolute relL2 are ties, not regressions,
     at any ctx. The strict-zero clause at ctx ≥ 32k is not falsifiable at
     fp32 metric precision — observed q223 h14/kv3: baseline 5.041846e-4 →
     union 5.042063e-4 (+2.17e-8, +0.004%), the only positive delta in the
     96-row standard set. Ties are printed by the gate, never silent.
   Encoded as a hard assertion in `scripts/verify_union_envelope.py`, run by
   the regen job on its own `gqa_union_commit.csv` (all 32 heads per
   position — the ratified population; see Inventory).

## Inventory

ALL 32 q heads ran at every position (walks are head-independent and unions
are within-group, so the dumped rows are unaffected — but the item-5 envelope
statistics are DEFINED on the full 32-head population per position, matching
the #20 ratification basis; a violation-enriched subset run fails the p95
clause spuriously). Only the contract heads dump golden rows.

Standard rows (the blessed contract heads):

    golden2_q159_h{0,8,16,24}.npz   ctx  14,838
    golden2_q223_h{0,8,16,24}.npz   ctx  38,838
    golden2_q287_h{0,8,16,24}.npz   ctx 134,838

**Honest-case rows (RTL ask):** the two envelope-permitted regressions from
the #20 quality validation, included so the gates encode the amended contract
honestly rather than only its favorable cases:

    golden2_q137_h0.npz             ctx 12,000, kv group 0 (heads 0-3)
    golden2_q137_h5.npz             ctx 12,000, kv group 1 (heads 4-7)

q137_h0: baseline relL2 3.998e-3 → union 4.142e-3 (+1.44e-4, +3.6%);
group-max-K head, union adds only 2 K tokens, V set 4800→6119, hi-K 1336→1524.
q137_h5: baseline 5.084e-3 → union 5.673e-3 (+5.88e-4, +11.6%); group-max-K
head, +197 K tokens, V 2400→3225, hi-K 1336→1619. Both inside the envelope.
Consumers regression-testing quality MUST use the per-head envelope, not
per-head monotonicity.

Page blocks (unchanged semantics, V-PQ is selection-independent):
`page_v_ctx{14838,38838,134838}_kv{0,2,4,6}.npz` byte-identical to the
pre-union set, plus new `page_v_ctx12000_kv{0,1}.npz` for the q137 rows.

## Field-change list vs `../stage2_20260707/` (the old contract)

**WALK-DOMAIN — bit-identical to the pre-union goldens** (selection frozen;
asserted by `scripts/classify_union_golden_moves.py` on the verifier report):
identity/config fields, `proxy_mass_c`, `start_ki/vi`, `settled_ki/vi`,
`probe_*` (item 1), item-7 3a per-probe fields (`vcorr_probe_*`,
`vcorr_dv_*`, `vcorr_acc_marginal_*`, `vcorr_acc_hiboundary_*`,
`vcorr_marginal_*`, `vcorr_hiboundary_*` — the walk's V probes read the
frozen per-head sets), `v_commit_mask_packed`, `v_int8_err_fp16`,
`v_code_error_fp16`, and the item-8 `two_pass_*` block (two-pass is a
SELECTION mechanism; selection is frozen per head — the union applies to what
is loaded/executed afterwards).

**EXECUTION-DOMAIN — recomputed under the union (changed meaning):**

- Items 2/3 `band_labels/count/max/sumexp/acc`, `combined_output_fp32`,
  `base_output_fp32`, `combine_rel_err`: the score row is the union
  execution's mixed row (exact logits for every union-K token; lo plane per
  the GROUP-MAX tier, i.e. tokens outside `union_k_hi_tokens` read this
  head's int8-QDQ logit). One extra band labeled **`union`** sits between
  `band<settled_ki>` and `tail`: group-mates' committed K tokens beyond this
  head's own rungs. Combine identity unchanged (self-checked < 1e-5).
- Item 5 `risk_scores`, `v_exact_count`, `v_risk_cutoff`,
  `v_exact_mask_packed`, `v_hi/lo_mask_packed`, `v_dropped_reads`: probs are
  the union softmax weights; the exact-V set is **GIVEN** (= the group
  committed-V union, `v_exact_count = |union_v_tokens|`) — NOT a top-count
  re-selection; `v_risk_cutoff` degrades to min-risk-inside-the-set
  (informative). The hi/lo split + fp-domain commit test run as before,
  inside the given set.
- Item 6 `v_w17_fp64` (union weights), `v_risk_key_q_fp64`,
  `v_risk_key_cutoff_q`, `v_exact_mask_key_packed`, `v_hi/lo_mask_key_packed`,
  `v_dropped_reads_key`: the E6M12 ranker's SELECT role is frozen out (set
  given); its SPLIT role remains the hardware contract — hi/lo ordering and
  the fp16 commit test over the given union set in the union-weight key
  domain.
- Item 7 3b/3c/3e `vcorr_settled_acc_{ref,hw}`, `vexact_*`,
  `vexact_band_*` (incl. `vexact_band_p_settled_fp64` = union weights): the
  settled commit state is the union execution. The G3 operand rebuild is
  UNCHANGED and stays bit-tight: settled accs rebuild from
  `vexact_band_p_settled_fp64` + the union key-domain masks +
  `base_output_fp32`; the 3a probe accs rebuild from the (frozen)
  `vcorr_marginal_p`/`vcorr_hiboundary_p`. Mixed-domain by design.

**NEW union_* fields (group context; per golden2 row):**

    gqa_union_commit             True (presence flag)
    union_group_heads            int64[4]  q-head ids of this kv group
    union_k_tokens               int64[Nk] group exact-K union (sorted ids)
    union_k_hi_tokens            int64[Nh] group hi-K union (sorted ids) —
                                 the RTL tier-merge verification target
                                 (max-tier: hi in ANY head → planes A+B)
    union_v_tokens               int64[Nv] group exact-V union (sorted ids)
    group_committed_k_flat/_offsets   ragged per-head committed K sets,
    group_committed_v_flat/_offsets   ordered by union_group_heads (head g
    group_committed_hi_k_flat/_offsets  spans [off[g], off[g+1]))
    union_score_max_fp64         max of the union mixed-score row
    union_softmax_sumexp_fp64    Σ exp(s − max) — the union softmax
                                 denominator pair (with the max) that every
                                 head's p row normalizes by
    union_baseline_rel_l2_fp64   this head's frozen-contract relL2
    union_rel_l2_fp64            this head's union-commit relL2

Union rebuild identity (consumer contract): `union_k_tokens` =
sorted(∪_g committed_k[g]); `union_k_hi_tokens` = sorted(∪_g hi_k[g]);
`union_v_tokens` = sorted(∪_g committed_v[g]) — all three rebuildable from
the ragged group_committed_* fields; the per-head committed sets equal the
pre-union contract's `selected_by_k[settled_ki]` / settled exact-V sets.

## Gates run at regen (all must pass)

(a) `verify_stage2_key_regen.py` vs `stage2_20260707`: G3 operand rebuild
bit-tight (≤1e-9, observed 0.0), two-pass rebuild exact, page blocks
byte-identical; execution-field moves EXPECTED and enumerated.
(b) `classify_union_golden_moves.py`: mismatch enumeration ⊆ the documented
execution-domain list above; any walk-domain move is fatal.
(c) `verify_union_envelope.py`: the ratified numeric envelope on the regen
run's own `gqa_union_commit.csv`.
