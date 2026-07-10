# Dependency epoch trace (issue #11, Phase 1)

Pure-observer trace of the realized escalation walk in the golden NumPy sim
(`run_joint_kv_budget_policy_eval.py`). Enabling `--epoch_trace_dir` adds these
files and changes NO other output (byte-identical with the flag off/on).

## Files
- `epoch_q{qidx}_h{head}.npz` -- one per (qidx, head); compact arrays below.
- `epoch_trace_index.csv` / `.jsonl` -- one row per (qidx, head) file.
- `README.md` -- this file.

## Epoch model
One epoch per realized walk segment (the SAME segments the runner's
`walk_step_MB_per_head` accumulator iterates). `event_kind_code`:
`0=start_eval, 1=k_up, 2=v_up, 3=commit`. `event_kind_code` is `start_eval`
for the first segment (start-rung evaluation) and otherwise equals the
segment action; `true_action_kind_code` always carries the raw segment action
(so a first-segment K up-move reads `event_kind=start_eval`,
`true_action_kind=k_up`, `is_start_eval=True`). The terminal `stop` segment is
`commit`. De-escalation (`kd`/`vd`) and `frozen_budget` segments are OFF the
escalate-only walk basis and are skipped, matching the runner.

Each segment READS its lookahead bands with no refund (deepest band per axis):
the K stability pair `(ki, ki+1)` and the marginal V band `(vi, vi+1)`.

## Walk-MB reconciliation (gate 2)
`epoch_walk_mb_contribution` is the incremental max of the K/V band costs,
seeded at the settled rung exactly as the runner seeds it; the settled
committed cost `total_MB` is attached to the commit epoch. Therefore
`sum(epoch_walk_mb_contribution) == walk_step_MB_per_head` to float roundoff.
`walk_mb_reconstructed_sum` and `walk_mb_reconstruction_abs_err` record the
realized reconstruction error per file (expected ~1e-12 MB or 0).

## GQA union reproduction (gate 3)
`committed_k_tokens` = `selected_by_k[settled_ki]` and `committed_v_tokens` =
`flatnonzero(exact_mask[(settled_ki, settled_vi)])` are the EXACT arrays the
runner's `--gqa_union_stats` accumulates, so K/V union-over-sum recomputed per
4-head group from these reproduces `gqa_union_stats.csv` exactly.

## npz fields
File meta: qidx, head, kv_head, position, context_len, page_size, head_dim,
policy, threshold, start_strategy, start_ki/vi, settled_ki/vi,
settled_k/v_budget, n_epochs, total_MB, v_path_MB, v_state_MB, k_exact_MB,
walk_step_MB_per_head, walk_mb_reconstructed_sum, walk_mb_reconstruction_abs_err,
precision_frozen_hi_count (-1 == not frozen), precision_k/v_hi_frac,
precision_lo_bits/bytes, v_lo_reads, v_dropped_reads, region_names,
region_byte_widths, n_pages.

Per-epoch columns (prefix `epoch_`, length n_epochs): epoch_id,
parent_epoch_id, event_kind_code, true_action_kind_code, is_start_eval,
ki_before, vi_before, ki_after, vi_after, k_read_rung, v_read_ki, v_read_vi,
k_read_mb, v_read_mb, walk_mb_contribution, rank_candidate_set_size,
n_dot_products, n_band_accumulations, n_k_marginal, n_v_marginal,
n_k_exact_cum, n_v_exact_cum, n_hi_boundary, kmove_den_old, kmove_den_new
(NaN unless a realized K up-move).

Compute-op definitions: `n_dot_products` = exact q.k logits newly computed for
the epoch's K lookahead band (marginal K set size); `n_band_accumulations` =
prob*V-row accumulations newly performed for the epoch's V lookahead band
(marginal V set size). `rank_candidate_set_size` = selected-K count at the
read rung.

K-move crossing (per §4 item 5b): for a realized K up-move, the marginal band
`(B_lo, B_hi]` (new exact-K tokens) is `k_marginal_tokens` for that epoch and
enters as int8 lo under the frozen split; the hi-boundary band
`(ceil(0.1*B_lo), ceil(0.1*B_hi)]` that lifts int8->exact is
`hi_boundary_tokens` (empty under `--precision_split_freeze start`, which
freezes the hi count for the whole walk). `kmove_den_old`/`kmove_den_new` are
the softmax denominators the Vcorr scalar rescale uses.

Region logical bytes: `epoch_region_ntokens` and `epoch_region_logical_bytes`
(shape n_epochs x len(region_names)) give the per-region MARGINAL-band token
counts and LOGICAL bytes (token count * region_byte_widths). These are pre-
physical-line-mapping (Phase 2); no line/burst widths are invented here.

CSR token sets (concat + offsets, length n_epochs+1): k_marginal_tokens,
v_marginal_tokens, hi_boundary_tokens. File-level sets: start_k_tokens,
committed_k_tokens, committed_v_tokens.

## Frozen run config
```json
{
  "decode_lengths": "128000",
  "heads": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31
  ],
  "k_budget_fracs": [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
    1.0
  ],
  "logit_buffer_bits": 8,
  "logit_buffer_format": "e4m3",
  "max_qidx_per_decode": 1,
  "n_files": 32,
  "page_size": 5632,
  "policies": [
    "k_first_alternating"
  ],
  "precision_k_hi_frac": 0.1,
  "precision_lo_bits": 8,
  "precision_lo_mode": "int8",
  "precision_split_freeze": "start",
  "precision_v_hi_frac": 0.1,
  "qidx_traced": [
    287
  ],
  "qkv_trace": "attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz",
  "score_proxy_variants": [
    "baseline"
  ],
  "start_strategies": [
    "proxy_mass_m0p9"
  ],
  "static_prefix": 128,
  "static_suffix": 128,
  "temporal_reuse_budget": "ladder",
  "temporal_reuse_mode": "frozen",
  "threshold_max_scale": 1.5,
  "threshold_min_scale": 0.0,
  "threshold_mode": "budget_delta_frac",
  "threshold_reference_frac": 0.2,
  "threshold_scale_shape": "sqrt",
  "thresholds": [
    0.004
  ],
  "v_budget_fracs": [
    0.05,
    0.1,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
  ],
  "v_selection_rules": [
    "global_residual_risk"
  ]
}
```
