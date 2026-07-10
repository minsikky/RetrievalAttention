# Epoch replay package (issue #11, Phase 2)

Physical-line replay deliverable built on the dependency epoch traces of the
frozen escalation controller (escalation-only, k_first_alternating,
proxy_mass_m0p9, tau=0.004, precision 0.1/0.1 int8 with frozen split, e4m3
logit buffer, page_size 5632). Companion to the Phase-1 package
`../epoch_trace_20260710/` (qidx 287 only, trace_format_version 1).

## Layout

- `traces/` -- realized epoch traces, trace_format_version 2 (adds
  `start_v_tokens`, `k_hi_tokens`, `k_hi_tokens_start` for physical replay),
  ALL 32 heads at four positions: q137 (ctx 12,000, high-escalation tail),
  q262 (ctx 83,225, mid-context), q283 (ctx 126,580, near cross-position
  mean), q287 (ctx 134,838, MVD -- Phase-1 position regenerated). Field docs
  in `traces/README.md`; per-file index in `traces/epoch_trace_index.{csv,jsonl}`;
  gates 2+3 results in `traces/validation_report.json`.
- `gold_run/` -- the producing golden-sim run's own outputs
  (per_head/layer/gqa_union CSVs, args.json, summary.json) for cross-checks.
- `replay/` -- physical replay outputs: `replay_sweep.csv` (position x
  reuse-window {0B, 64KiB, 256KiB, 1MiB, unlimited} x order {head_serial,
  interleaved} -> physical bytes + 32B-transaction requests), per-position
  epochs JSONs in the RTL schema (oracle + bounded 256KiB), and
  `summary.md` (headline numbers, gates, interpretation notes).

## Producers

- traces: `benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py
  --epoch_trace_dir` via `scripts/run_epoch_trace_phase2.sbatch`
  (job 53288389; observer-neutrality gate job 53288388).
- replay: `scripts/replay_epoch_trace_physical.py` (RTL physical contract
  constants; `--self_test` for the unit tests).

Exact regeneration:

```
sbatch scripts/run_epoch_trace_phase2.sbatch
.venv/bin/python scripts/replay_epoch_trace_physical.py \
  --trace_dir attention_efficiency_result/epoch_trace_phase2_20260710/trace \
  --qidx 137,262,283,287 --out_dir <out>
```
