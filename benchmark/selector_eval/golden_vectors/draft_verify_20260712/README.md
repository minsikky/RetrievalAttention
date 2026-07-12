# Draft-then-verify goldens (issue #21, items 1/2/4) — 2026-07-12

RTL-side deliverables for the draft-then-verify decode proposal (issue #21).
Items **1** (draft-selection block), **2** (verify replay trace, k=4 AND k=8),
and **4** (acceptance distribution) are here. **Item 3 is HELD** per the #21
thread (streamed-consumer occupancy re-spec) and is NOT produced.

Branch `draft-verify-goldens`, cut from `draft-gpu` (which carries the
`SELECTOR_PQ_JOINT_DRAFT_MODE` code). Producer scripts committed alongside;
format follows `../stage2_union_commit_20260710/` conventions (npz + README,
producers in `scripts/`, combiners here).

## What "draft mode" is (the selection under test)

Env flag `SELECTOR_PQ_JOINT_DRAFT_MODE={off,start1,start2}`.

- `off` — the frozen escalation walk: proxy-mass start rung, then the stability
  escalation walk. Unchanged / byte-identical to the frozen algorithm.
- `start1` / `start2` — **skip the walk**; pin each head's K rung to
  `start_rung + 1` (start1) / `+2` (start2), clamped to the ladder, then apply
  the frozen `v_target` rule `v = max(v_budgets[0], 0.25*k_target)`. Top tokens
  come from the existing risk ranking. This is the one-shot **draft** selection.

The flag is honored in two places: the GPU decode controller
`select_joint_kv_budgets` (`hf_paged_pq_intervention_joint_policy.py`, branch
`draft-gpu`, used by the acceptance runs) and — added on this branch, for the
CPU stage-2 dump path — a mirrored override in
`run_joint_kv_budget_policy_eval.py` (`_stage2_draft_bump`; the settled point is
replaced with the pinned rung right after the walk, so the dump emits the DRAFT
committed set from the SAME scan state the frozen walk saw). Default off leaves
the dump byte-identical to the frozen goldens.

## Provenance note (read this before consuming item 2)

All committed-set operands here come from the **layer-16 decode trace**
(`real_qkv_llama31_l16_..._q288_window32...`) via the stage-2 dump machinery —
the SAME trace every stage-2 golden is built on, so RTL cross-references it
directly. The GPU acceptance runs
(`benchmark_suite_result/draft_verify_gpu_20260711/`) saved greedy `pred` text
but **not** per-decode-step committed sets, and the GPU decode path has no
committed-set dump hook and no teacher-forcing (a heavier instrumentation than
this delivery warranted). So item 2 is assembled from two ground truths, by
role:

- **committed-set operands (a,b,c)** — from the layer-16 trace segment (this
  is what sizes the RTL streamed-consumer datapath: union rows/bytes, per-
  position membership, fetch-on-miss stream). Single layer, as all stage-2
  goldens are.
- **accepted-prefix / rollback (d)** — from the full-model 32-layer GPU
  acceptance runs, reformatted from `acceptance.json` (the token-agreement
  ground truth). See `golden2_accepted_prefix_rounds.csv`.

The RTL verify-FSM shape (draft union → per-position membership → fetch-on-miss
→ accepted-prefix → rollback) is fully exercised; the two data sources are
noted because they are different samples. A same-sample GPU replay would need a
new per-step committed-set dump + teacher-forcing in the decode path — a
follow-up, not this delivery.

---

## Item 1 — draft-selection block

`golden1_q{qidx}_h{head}.npz`, one per (position, head) on the standard stage-2
positions × contract heads:

    golden1_q159_h{0,8,16,24}.npz   ctx  14,838
    golden1_q223_h{0,8,16,24}.npz   ctx  38,838
    golden1_q287_h{0,8,16,24}.npz   ctx 134,838

Per row: the scan/start inputs, the frozen escalation-walk committed set, the
two one-shot draft committed sets (start+1, start+2), and the superset/miss
bitmap of each draft vs frozen.

Producer: `scripts/run_draft_verify_golden1.sbatch` (off/start1/start2 dumps) →
`combine_golden1.py`. Summary CSV: `golden1_draft_selection.csv`.

### Field inventory (`golden1_*.npz`)

| field | dtype/shape | meaning |
|---|---|---|
| `qidx`,`head`,`kv_head`,`position`,`context_len` | scalar | identity |
| `proxy_mass_c` | scalar | proxy-mass count (the start-rung driver) |
| `start_ki`,`start_vi` | scalar | proxy-mass start rungs |
| `k_budgets`,`v_budgets` | int64[·] | the K/V ladders (rung → budget) |
| `risk_scores` | f64[ctx] | the risk ranking over the whole context |
| `frozen_settled_ki/_vi` | scalar | frozen walk's settled rungs |
| `frozen_committed_k_tokens` | int64[·] | frozen committed K set (sorted ids) |
| `frozen_committed_v_tokens` | int64[·] | frozen committed exact-V set |
| `draft_start1_settled_ki/_vi` | scalar | pinned draft rung = start+1 |
| `draft_start1_committed_k_tokens` | int64[·] | start+1 one-shot committed K set |
| `draft_start1_committed_v_tokens` | int64[·] | start+1 committed exact-V set |
| `draft_start2_*` | — | same for start+2 |
| `draft_start1_contains_frozen_k_packed` | uint8[·] | packbits over the SORTED frozen K set: 1 = that frozen token is in the start+1 draft set (the superset bitmap) |
| `draft_start2_contains_frozen_k_packed` | uint8[·] | same for start+2 |
| `draft_start1_missed_frozen_k_tokens` | int64[·] | frozen K tokens the start+1 draft MISSES |
| `draft_start1_extra_k_tokens` | int64[·] | draft-only K tokens (over-provision) |
| `draft_start{1,2}_contains_frozen_v_packed` | uint8[·] | V-set superset bitmap |
| `draft_start{1,2}_k_recall`,`_v_recall` | scalar | |frozen∩draft|/|frozen| |
| `frozen_matches_stage2` | bool | gate 1 (see below) |

`draft_start2_missed_frozen_k_tokens` / `_extra_k_tokens` likewise.

### Measured draft recall (item-1 superset stats)

Recall = fraction of frozen-selected K tokens contained in the draft set, per
(position, head). Full table in `golden1_draft_selection.csv`.

| qidx | ctx | head | start_ki | frozen settled_ki | \|K frozen\| | \|K start+1\| | \|K start+2\| | recall s1 | recall s2 | miss s1 | miss s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 159 | 14,838 | 0  | 2 | 2 | 10,993 | 13,961 | 14,838 | 1.000 | 1.000 | 0 | 0 |
| 159 | 14,838 | 8  | 2 | 3 | 10,993 | 10,993 | 13,961 | 1.000 | 1.000 | 0 | 0 |
| 159 | 14,838 | 16 | 2 | 2 | 10,993 | 13,961 | 14,838 | 1.000 | 1.000 | 0 | 0 |
| 159 | 14,838 | 24 | 2 | 2 | 10,993 | 13,961 | 14,838 | 1.000 | 1.000 | 0 | 0 |
| 223 | 38,838 | 0  | 3 | 3 | 24,465 | 32,233 | 38,838 | 1.000 | 1.000 | 0 | 0 |
| 223 | 38,838 | 8  | 3 | 3 | 24,465 | 32,233 | 38,838 | 1.000 | 1.000 | 0 | 0 |
| 223 | 38,838 | 16 | 3 | 3 | 24,465 | 32,233 | 38,838 | 1.000 | 1.000 | 0 | 0 |
| 223 | 38,838 | 24 | 3 | 2 | 16,698 | 24,465 | 32,233 | 1.000 | 1.000 | 0 | 0 |
| 287 | 134,838 | 0  | 1 | 1 | 45,754 | 72,721 | 99,689 | 1.000 | 1.000 | 0 | 0 |
| 287 | 134,838 | 8  | 1 | 1 | 45,754 | 72,721 | 99,689 | 1.000 | 1.000 | 0 | 0 |
| 287 | 134,838 | 16 | 1 | 0 | 18,786 | 45,754 | 72,721 | 1.000 | 1.000 | 0 | 0 |
| 287 | 134,838 | 24 | 1 | 1 | 45,754 | 72,721 | 99,689 | 1.000 | 1.000 | 0 | 0 |

(q159 h8's frozen walk settled one rung ABOVE its start, so its start+1 draft
equals the frozen set exactly; exact per-row values in the CSV.)

**Measured recall is 1.000 at every (position, head), for both start+1 and
start+2 — stronger than the ~0.916 per-layer start+1 figure from the offline
study, and structural rather than lucky:** selection is rung-NESTED (same
risk ranking; a larger budget's set contains a smaller one's), so the draft
can only miss frozen tokens when the frozen walk escalates ABOVE the draft's
pinned rung. On the standard golden positions the frozen walk settles at
start-1/start/start+1 everywhere, so the start+1 set already contains every
frozen-committed token. The offline 0.916 averaged over positions where the
walk escalated further; none of those land on the blessed positions. The
superset bitmaps are consequently all-ones and the miss lists empty here —
the fields and the FSM they gate are fully exercised, but a consumer wanting
a NONZERO miss stream must pick positions where the walk escalates >= start+2
(none exist in the standard set at this operating point). Draft
over-provision (the byte cost of one-shot): |K start+1| / |K frozen| mean
~1.4x, worst 2.44x (q287 h16, whose walk settled below its start rung).

---

## Item 2 — verify replay trace (k=4 AND k=8)

Over the L16 decode segment (qidx 208..223 = 16 sampled decode positions, ctx
~31k..39k) × kv-group-0 heads {0,1,2,3}. A **round** = a window of k consecutive
positions whose 4k verify consumers share one gather. Delivered **start+1-only**
(the #14 predicted-theta union component is not reproduced here — the FSM shape
is identical, as the #21 thread permits).

`golden2_k{k}_round{r}.npz` for k∈{4,8}. Producer:
`scripts/run_draft_verify_golden2.sbatch` (off + start1 dumps) →
`combine_golden2.py`. Round table: `golden2_verify_rounds.csv`.

### Field inventory (`golden2_k{k}_round{r}.npz`)

| field | meaning |
|---|---|
| `layer_id`(=16), `kv_group`(=0), `group_heads` | context (single layer) |
| `k`, `round_index`, `round_positions_qidx`, `round_context_lens` | round id |
| `drafted_union_k_tokens` / `_v_tokens` | (a) union over the round's positions AND the 4 heads of the start+1 committed set — the streamed committed-KV list |
| `drafted_union_k_rows`/`_bytes`, `_v_rows`/`_bytes` | per-round union size (bytes = rows·128·2, fp16) |
| `frozen_committed_k_flat`/`_offsets` | (b) per-position frozen committed set (group union), ragged, ordered by `round_positions_qidx` |
| `membership_packed_flat`/`_offsets` | (c) per-position packbits over that position's sorted frozen set: 1 = token present in the drafted union |
| `fetch_on_miss_flat`/`_offsets` | (c) per-position fetch-on-miss stream: frozen tokens ABSENT from the drafted union |
| `per_position_draft_union_recall` | |frozen_p ∩ drafted_union| / |frozen_p| |

### Measured per-round union sizes (kv-group 0, layer 16, fp16 rows)

| k | round | positions (qidx) | union-K rows | union-K MB | union-V rows | recall/pos |
|---|---|---|---|---|---|---|
| 4 | 0 | 208–211 | 32,644 | 8.36 | 18,693 | 1.000 ×4 |
| 4 | 1 | 212–215 | 34,709 | 8.89 | 24,665 | 1.000 ×4 |
| 4 | 2 | 216–219 | 36,437 | 9.33 | 25,980 | 1.000 ×4 |
| 4 | 3 | 220–223 | 38,838 | 9.94 | 26,426 | 1.000 ×4 |
| 8 | 0 | 208–215 | 34,709 | 8.89 | 26,755 | 1.000 ×8 |
| 8 | 1 | 216–223 | 38,838 | 9.94 | 29,202 | 1.000 ×8 |

Per-position draft-union recall of frozen is 1.000 throughout, for the same
structural reason as item 1 (rung-nested selection; the frozen walk never
escalated above start+1 in this segment) — so the **fetch-on-miss streams are
EMPTY in this delivery**. The membership bitmaps, ragged offsets, and miss
fields are populated and typed (RTL can wire the FSM against them); a segment
with nonzero misses requires positions where the walk escalates ≥ start+2,
which do not occur here. Note the L16 trace's decode positions are SAMPLED
(non-consecutive; the segment spans ctx 31,096→38,838), so a round models the
k-position gather sharing shape, not literally adjacent tokens — union sizes
are therefore slight over-estimates of the adjacent-token case (queries drift
more between sampled positions).

### Accepted-prefix / rollback (part d) — from the GPU acceptance ground truth

`golden2_accepted_prefix_rounds.csv`, one row per (arm, sample, k, round):

| column | meaning |
|---|---|
| `arm`,`ctx`,`mode`,`sample_index` | GPU acceptance arm + sample |
| `k`,`round_index` | length-k verify round over the decode stream |
| `round_start_pos`,`round_end_pos` | decode-token window [start,end) |
| `accepted_prefix_len` | leading matching tokens before first reject (0..k) |
| `full_accept` | 1 iff the whole round matched frozen |
| `rollback_pos` | absolute decode position the verify rolls back to (= start + accepted_prefix_len) |
| `n_rejected_in_round` | k − accepted_prefix_len |

Gate: these accepted-prefix lengths ARE `acceptance.json`'s agreement structure
(same samples), so they match it by construction — see item 4 for the same data
aggregated.

---

## Item 4 — acceptance distribution

Pure reformatting of `acceptance.json` (arms off/start1/start2 for qa_1 32k n=4
and mk3 128k n=2; token agreement from `analyze_draft_verify.py`). Producer:
`make_acceptance_csvs.py`. Source copied here as `acceptance_source.json`.

- `golden4_acceptance_aggregate.csv` — one row per (mode × ctx) arm:
  `n_samples, pooled_aligned_tokens, pooled_agreement_rate,
  acceptance_at_{4,8,16}, expected_accepted_prefix_k8, mean_first_divergence,
  n_fully_identical_streams, draft_standalone_correct, frozen_correct`.
- `golden4_acceptance_per_sample.csv` — one row per (arm, sample):
  `frozen_len, draft_len, aligned_len, agreement_rate, first_divergence,
  answer_token_pos, answer_region_passed_before_fork, run_lengths (;-joined),
  n_runs, frozen_correct, draft_correct`. (first-divergence vs answer position.)
- `golden4_acceptance_runlengths.csv` — one row per agreement run
  (`arm,ctx,mode,sample_index,run_ordinal,run_length`): the run-length
  distribution.

Aggregate (from acceptance.json, unchanged):

| arm | agree | acc@4 | acc@8 | acc@16 | eap@8 |
|---|---|---|---|---|---|
| 32k start1 | 0.938 | 0.936 | 0.934 | 0.929 | 7.70 |
| 32k start2 | 0.938 | 0.936 | 0.934 | 0.929 | 7.70 |
| 128k start1 | 0.922 | 0.896 | 0.860 | 0.779 | 7.11 |
| 128k start2 | 0.922 | 0.896 | 0.860 | 0.779 | 7.11 |

---

## Consistency gates (self-checked)

All three gates PASSED at delivery (producer jobs 53343710 golden1 5m55s,
53343711 golden2 8m05s, smoke 53343658 44s; standard partition, zhengya0):

1. **Stage-2 cross-check (bit-tight).** For every standard (q, h): the off-mode
   dump's committed K AND V sets equal the checked-in
   `stage2_union_commit_20260710` goldens' per-head `group_committed_{k,v}`
   span, and ALL walk-domain fields (`proxy_mass_c`, `start_ki/vi`,
   `settled_ki/vi`, `probe_*`, `v_commit_mask_packed`, `v_int8_err_fp16`,
   `v_code_error_fp16`, the `two_pass_*` selection block) are bit-identical.
   12/12 rows OK. Proves the draft-mode dump plumbing did not disturb frozen
   selection semantics.
2. **Off-mode identity (structural + verified).** `SELECTOR_PQ_JOINT_DRAFT_MODE=off`
   never enters the draft override (`_stage2_draft_bump` returns None before
   any state is touched), and gate 1 verifies the resulting dump against the
   blessed goldens bit-tight.
3. **Acceptance cross-check.** For every (arm, sample, k): the sum of
   `accepted_prefix_len` over the disjoint verify rounds in
   `golden2_accepted_prefix_rounds.csv` equals the leading agreement-run
   length from `acceptance.json` (truncated to full-round coverage), and the
   reconstruction of each sample's match array asserts the recorded
   run-lengths and agreement rate exactly. ALL MATCH.

## Reproduce

```
# item 4 (pure reformat; no model)
python3 make_acceptance_csvs.py

# items 1 & 2 (CPU stage-2 dumps, standard partition, account zhengya0)
sbatch scripts/run_draft_verify_golden1.sbatch     # off/start1/start2, std positions
sbatch scripts/run_draft_verify_golden2.sbatch     # off/start1, qidx 208..223 segment
# then, with .venv + LD_LIBRARY_PATH=/sw/pkgs/arc/python/3.10.4/lib:
python combine_golden1.py --dump_root <scratch>/draft_verify_golden1_20260712
python combine_golden2.py --dump_root <scratch>/draft_verify_golden2_20260712
```
