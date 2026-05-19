# Benchmark Readiness Checklist

Current objective: get the current frontier attention algorithm into a state where real LongBench/RULER benchmarks can be run comfortably and trusted. That means the full frontier path is correct enough, fast enough on GPU, and honestly accounted for; once those gates pass, run the paired dense/frontier benchmark matrix and quantify how relL2/cosine/logit drift maps to task-level accuracy.

## Final Status

Status: complete for selected-scope benchmark readiness.

`bash scripts/check_frontier_benchmark_readiness.sh` passes. The implementation is ready to run real selected-scope LongBench/RULER benchmark evaluations with the current frontier preset. Broader/full-suite coverage is the next phase, not a remaining blocker for this readiness objective.

## Prompt-To-Artifact Audit

| objective requirement | concrete evidence | status |
| --- | --- | --- |
| Full frontier algorithm implemented on GPU, not isolated subcomponents | Frontier wrappers `scripts/run_frontier_ruler_batched_one.sh` and `scripts/run_frontier_longbench_v2_one.sh`; wrapper audit `notes/wrapper_config_audit_20260516.md` shows `cuda_ext`, `torch_gpu`, `torch_matmul`, native selected/tail attention, V-PQ selected values, and `spgpu` / `zhengya98`. Matrix audit `notes/frontier_benchmark_matrix_afterok_20260516_audit.md` shows real RULER and LongBench frontier runs with zero passthrough. | Complete for selected benchmark path. |
| Correctness against dense/reference attention | CUDA unit audit `notes/cuda_unit_audit_20260516.md` shows job `50321548` passed selector top-k, GPU V-PQ helper, and online-page append tests. Wrapper smoke audit `notes/wrapper_smoke_audit_20260516.md` shows dense/frontier RULER and LongBench smokes all `ok`. RULER matrix scores match dense on all tested tasks. | Complete for selected benchmark path. |
| Per-head/per-layer relL2, cosine, and output-level checks | Trace/layer q288 validation artifacts are summarized in `notes/selector_eval_latest_results.md` and include attention/layer relL2 and cosine. Changed-row dense-reference diagnostics in `notes/frontier_benchmark_matrix_afterok_20260516_longbench_drift.md` include logit relL2/cosine, hidden relL2/cosine, KL, top-1 agreement, and choice-top agreement for every changed LongBench prediction. | Complete for selected benchmark path. |
| GPU performance fast enough for real evaluation | Matrix jobs `50321549`-`50321558` completed inside normal Slurm limits. RULER frontier finished in `32.08-40.58 s/sample`; LongBench-v2 frontier finished in `36.70 s/example`, total Slurm elapsed `37:19` for 59 examples. | Complete for selected benchmark path. |
| Honest accounting: no oracle mass, no dense ranking, no hidden fallback, no mixed snapshot/online assumptions | Strict artifact audit passes. Frontier matrix rows report separated selector/exact/tail/update MB and passthrough `0`. Wrapper audit confirms deployable `pq_ranked_mass_budget` and expected GPU backends. | Complete for current preset. |
| Heavy builds/tests/benchmarks run through Slurm on `spgpu` | CUDA unit job `50321548`, wrapper smoke jobs `50320647`-`50320650`, matrix jobs `50321549`-`50321558`, and changed-row diagnostic jobs `50322156`-`50322158` ran through Slurm on `spgpu` with account `zhengya98`. | Complete. |
| Real downstream benchmarks executed | RULER ctx8k n=4 per task and LongBench-v2 short/easy n=59 were run with paired dense/frontier settings. Audit output: `notes/frontier_benchmark_matrix_afterok_20260516_audit.md`. | Complete for selected scope. |
| RelL2/cosine/logit drift mapped to task-level accuracy | LongBench comparison `notes/frontier_benchmark_matrix_afterok_20260516_longbench_compare.txt` reports dense `21/59 = 35.59%`, frontier `24/59 = 40.68%`, with 3 changed/gained-correct rows. Drift report `notes/frontier_benchmark_matrix_afterok_20260516_longbench_drift.md` links all changed rows to dense-reference logit/hidden diagnostics. | Complete for selected scope. |

## Benchmark Results

| task | dense | frontier | frontier sec/ex | frontier MB/head-query | passthrough |
| --- | ---: | ---: | ---: | ---: | ---: |
| RULER `niah_single_1`, ctx8192, n=4 | 100.00 | 100.00 | 37.94 | 2.452 | 0 |
| RULER `niah_multikey_2`, ctx8192, n=4 | 100.00 | 100.00 | 40.58 | 2.434 | 0 |
| RULER `vt`, ctx8192, n=4 | 100.00 | 100.00 | 32.08 | 2.479 | 0 |
| RULER `fwe`, ctx8192, n=4 | 75.00 | 75.00 | 33.20 | 2.474 | 0 |
| LongBench-v2 short/easy, max input 8192 | 35.59 | 40.68 | 36.70 | 2.555 | 0 |

LongBench prediction comparison:

- Dense: `21/59 = 35.59%`.
- Frontier: `24/59 = 40.68%`.
- Prediction agreement: `56/59`.
- Judge agreement: `56/59`.
- All three changed rows were `gained_correct`.

## Completion Criteria

1. Dense and frontier wrappers complete clean smoke runs through Slurm: complete.
2. Fresh CUDA selector/V-PQ unit tests complete through Slurm and write a passing `summary.json`: complete.
3. The audit tool reports no passthrough, no missing costs, no missing config, and expected dense/frontier modes for smoke artifacts: complete.
4. At least one paired RULER validation matrix and one paired LongBench-v2 validation matrix are run with the same wrapper presets: complete.
5. GPU runtime is practical enough for the selected benchmark scope to finish inside normal Slurm jobs without manual babysitting: complete.
6. Dense-reference diagnostics are linked for enough rows to explain observed task agreement/disagreement in terms of relL2/cosine/logit drift: complete for all changed LongBench rows.
7. The final reported benchmark table includes task accuracy, runtime, separated modeled MB fields, and linked relL2/cosine/logit diagnostics: complete.
8. `bash scripts/check_frontier_benchmark_readiness.sh` passes: complete.
