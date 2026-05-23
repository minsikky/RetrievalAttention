# Selector-Eval Latest Results

Keep this page compact. Archive full tables and stale variants instead of appending indefinitely.

## Active Goal

End state: one benchmark-ready canonical GPU implementation of the current CPU frontier decode algorithm. It keeps dense prefill, approximates decode only after sealed PQ pages are active, matches CPU frontier semantics, and is fast enough to run real dense-vs-frontier task benchmarks.

Canonical algorithm semantics to preserve:

- Prefill attention is dense. Prefill may build/update PQ sidecars, but it must not use sparse/frontier attention.
- Decode uses the CPU frontier algorithm: paged K-PQ fullscan selector, affine-calibrated approximate logits, candidate ranking, exact-K refinement, mixed exact-K/K-PQ logits and probabilities, adaptive K/V output-stability confidence, global residual-risk exact-V selection, and V-PQ reconstruction for non-exact V rows.
- Exact V selection is global residual risk: `risk_i = p_i^2 * V_PQ_error_stat_i`, where `p_i` comes from the mixed exact-K/K-PQ probability distribution.
- Accepted K budgets, accepted V budgets, selected-token counts, logical MB/query, attention outputs, and o-proj outputs must match the CPU reference within parity tolerance.
- Any change to selector scoring, calibration, ranking, budget ladder, confidence/stopping rule, probability construction, exact-V rule, V-PQ reconstruction, or dense-prefill/decode-only scope is a separate algorithm variant, not a canonical optimization.

Implementation approach:

- The benchmark decode hot path must be fused/native CUDA. The target is not a PyTorch-heavy or hybrid PyTorch+CUDA implementation.
- Native CUDA should own decode-critical work: K-PQ scoring, candidate ranking/top-k, exact-K logits, mixed probability construction, adaptive K/V confidence, residual-risk ranking/prefixes, V-PQ aggregation, and final output construction.
- PyTorch is allowed for model orchestration, tensor allocation/lifetime, QKV/O-projection plumbing, extension launch glue, lightweight metadata setup, assertions, reporting, and offline parity/debug checks.
- PyTorch must not dominate the decode loop through repeated hot-path `topk`, `sort`, `gather`, `scatter`, `softmax`, matmul, masking, prefix-sum, or per-budget tensor materialization.
- Dense-logit simulator paths, PyTorch-heavy paths, and partially native paths are useful diagnostics, but they are not the final benchmark path unless profiling proves the remaining PyTorch work is outside the decode-critical path and semantics still match the CPU reference.

Performance goals:

- Overall runtime target: canonical frontier decode should run within `2-3x` dense decode on representative active-path benchmarks while preserving semantics.
- Primary short-decode gate: RULER `niah_single_1`, 32k context, 1 sample, 128 decode tokens, all layers/all heads, Slurm `spgpu`. Dense reference job `50645768` decoded in `17.99s`; required frontier target is `<=54s` (`<=3x`), stretch target `<=36s` (`<=2x`).
- Sustained-decode gate: forced LongGen-style workload with sealed pages active for at least `8192` generated tokens. Dense reference job `50718580` generated 8192 tokens in `323.65s`; required frontier target is `<=971s` (`<=3x`), stretch target `<=647s` (`<=2x`).
- Secondary sustained-scale gate: forced `16384` generated tokens. Dense reference is `702.13s`; required frontier target is `<=2106s` (`<=3x`), stretch target `<=1404s` (`<=2x`).
- Report prefill latency, decode/generation latency, profiling overhead, cost-stat overhead, sidecar/update overhead, logical frontier MB, and physical GPU MB separately.
- After runtime gates pass, run dense-vs-frontier task-quality comparisons on at least one reasoning benchmark, one coding benchmark, and one long-generation benchmark.

Success criteria:

- CUDA unit tests pass for every native helper used by the canonical path.
- Saved-trace CPU-vs-GPU parity passes on real Q/K/V traces at decode lengths `32000`, `64000`, and `128000` on multiple heads.
- Parity covers accepted K/V budgets, selected-token counts, logical MB, attention outputs, and o-proj outputs.
- HF/RULER with `FRONTIER_CANONICAL_GPU=1` exercises the approximation path, not dense fallback, inactive sidecars, or diagnostic shortcuts.
- A paired primary gate exists: one no-stats timing run for latency and one accounting run for logical/physical MB.
- The primary 32k/128 gate reaches `<=54s` decode without semantic changes.
- The sustained 8192-token gate reaches `<=971s` generation time without semantic changes.
- Cost reports separate logical frontier MB from physical GPU execution MB and include selector MB, exact-K/V MB, V-PQ/tail MB, update/sidecar MB, and total step MB/query.
- Profiling shows the decode hot path is native-CUDA dominated and identifies any remaining non-native bottlenecks.
- Benchmark-quality runs report dense and frontier accuracy/score, decode latency, active approximation coverage, logical MB, and physical MB for matched model/task settings.

Constraints / invalid shortcuts:

- Do not replace adaptive confidence with fixed budgets, fixed top-k, selected-mass V, selector-rank exactness, hand-calibrated schedules, context-length-specific knobs, or benchmark-specific knobs and call it canonical.
- Do not use oracle attention probabilities, dense top-k rankings, true achieved mass, relL2 against dense output, task labels, future tokens, generated answers, or post-hoc dense outputs inside selector, compression, confidence, or budget logic.
- Do not compute dense attention output and then mask/prune only for reporting. Dense compute is acceptable only as a clearly labeled diagnostic/simulator baseline, not as the canonical benchmark hot path.
- Do not count physical dense reads as logical sparse reads, and do not hide dense K/V reads, sidecar rebuilds, PQ refreshes, exact probes, PyTorch sort/top-k/gather traffic, online-update work, or calibration/sidecar costs.
- Do not promote an optimization that changes accepted budgets, selected-token statistics, logical MB, or outputs beyond parity tolerance; record it as a separate variant.
- Do not claim benchmark readiness from short smokes, single-layer/head traces, disabled cost stats without paired accounting, inactive approximation paths, runs where sealed PQ pages never activate, or runs that only pass because the context is too short.
- Do not run heavy GPU jobs or extension builds on login nodes; use Slurm `spgpu` with account `zhengya98`.

## Canonical Frontier Path

Current reference frontier path as of 2026-05-22:

- dense prefill;
- decode-only fullscan paged-PQ selector;
- K-PQ approximate logits for all tokens, ranked token candidates, then exact K logits for selected tokens;
- mixed attention probabilities from exact selected-K logits plus K-PQ tail logits;
- adaptive K and V budgets by output-stability confidence;
- global exact-V selection by residual-risk, `risk_i = p_i^2 * V_PQ_error_stat_i`;
- V-PQ reconstruction for non-exact V rows;
- exact accepted-budget logical accounting separated from physical GPU reads.

Reference trace result: `k_first_alternating`, threshold `0.001`, layer 16/all heads, decode lengths `500..128000`, mean logical cost `4.779 MB/head-query`, mean o-proj relL2 `0.001118`, max o-proj relL2 `0.002082`.

Implementation status: the trace runner `run_joint_kv_budget_policy_eval.py` implements the adaptive K/V residual-risk policy. The HF benchmark wrapper now exposes `online_confidence_rule=joint_kv_stability` plus `selected_value_exact_rule=global_residual_risk`, and `FRONTIER_CANONICAL_GPU=1` requires those settings. CPU-vs-CUDA trace parity smokes passed. The benchmark-facing wrapper defaults now enable the validated native V-prefix helper plus prewarmed persistent V-PQ sidecars, which brings the 32k/4 decode diagnostic inside the `2-3x` dense target. Representative 128-token frontier validation and dense-vs-frontier task slices are still pending.

`FRONTIER_CANONICAL_GPU=1` rejects noncanonical fixed-budget, selector-rank, selected-mass shortcuts, segmented V-prefix, and exact all-head precompute. It now requires native V-prefix plus prewarmed persistent V-PQ sidecars for the benchmark-ready path.

Latest 2026-05-23 status:

- Promoted canonical defaults now include grouped residual-risk prefix and exact-fullbudget score-grid rows, while rejected diagnostics remain guarded off: all-head rank-prefix, full-budget sort skipping, and grouped V-PQ residual cache.
- Exact-fullbudget 32k/16 accounting run `50708291` preserved canonical logical stats and improved decode to `48.02s` on the profiled 16-token slice: logical step `4.081499 MB/head-query`, selected `12455.15625`.
- A compile-only Slurm build for the new grouped-flat CUDA policy/output helper, `50709650`, passed on `standard` in `4:18`.
- Decode-side persistent V-PQ sidecar cache extension was changed from per-token `torch.cat` growth to capacity-backed in-place append. This preserves the prior semantics for unsealed suffix tokens (`vhat=exact V`, zero residual/error) but removes O(N) sidecar copying from long decode. Local `py_compile`, shell `bash -n`, and `benchmark/audit_benchmark_wrappers.py` checks pass after the change.
- Added a diagnostic fused grouped-risk policy CUDA path, `SELECTOR_PQ_JOINT_FUSED_RISK_POLICY=1`, which reuses the same residual-risk sort but selects the adaptive K/V policy output directly from sorted risk rows instead of materializing the full grouped V-budget output grid first. It is not promoted yet and canonical wrappers now default it off; `FRONTIER_CANONICAL_GPU=1` rejects it until parity/profile validation promotes it. Compile-only Slurm job `50710431` passed on `standard` in `244s`; local extension import confirms `joint_select_policy_grouped_flat` and `joint_select_policy_from_grouped_risk` are exported; `test_gpu_vpq_helpers.py` now includes parity coverage against the existing grouped-grid + grouped-policy path.
- Direct benchmark scripts `benchmark/run_longbench_v2_hf.sh` and `benchmark/run_public_longdecode_hf.sh` now expose the same canonical joint-KV defaults as the Slurm wrappers, including grouped risk-prefix, native score/risk/policy/V-prefix, exact-fullbudget grid, prewarmed persistent V-PQ sidecars, and fused-risk default-off. `benchmark/audit_benchmark_wrappers.py` now checks these direct scripts too, and the audit passes.
- Local decode-hot-path cleanup avoids materializing selected-token tensors for every K budget when native score-grid construction is active. The score grid only needs take counts, base tokens, and ranked prefixes, while accounting uses selected counts. This is intended to reduce PyTorch allocation/gather overhead without changing semantics; local `py_compile`, wrapper audit, and `git diff --check` pass, but GPU parity/runtime validation is pending.
- Native mixed score-grid construction now handles all K-budget rows, including full-budget exact rows, in one native call. This removes Python-side partial-row `index_copy_` and exact-row fill work; the native kernel overwrites base/ranked selected tokens with exact logits, so a full-budget row becomes exact without Python post-processing. Trace parity job `50711939` passed on decodes `500,1000`, heads `0,8`, with no failures, max CPU/native attention/o-proj relL2 `4.44e-09` / `5.62e-09`, and max Torch/GPU-policy attention/o-proj relL2 `1.85e-06` / `1.65e-06`. HF profile job `50711825` is still pending.
- Batched grouped residual-risk native helpers were added for the canonical grouped V-prefix path. The promoted grouped-risk path can now pass `[groups, k, heads, ...]` tensors into CUDA and derive row groups in-kernel, avoiding explicit row-group ID construction and flattened `torch.cat` buffers in Python. Unit parity coverage was added against the existing flat grouped-risk reference. Local checks and extension symbol import pass; compile-only Slurm job `50713062` passed in `261s` at `cuda_unit_result/frontier_cuda_ext_build_only_20260523_batched_grouped_risk`. Full GPU unit/parity/runtime validation is pending on `spgpu`.
- Targeted CUDA unit job `50709435` failed from a Python unit-test ordering bug (`policy_ids` referenced before assignment), not a kernel mismatch. The test was fixed. Retry job `50713581` passed in `4:33` with exit `0`, output `cuda_unit_result/frontier_cuda_grouped_flat_policy_20260523_retry1`. Dependent fused-risk diagnostic profile `50713697` completed and is negative: score `100.0`, decode `66.74s`, logical step `4.0833 MB/head-query`, selected `12462.66`. It preserves current stats but is slower than the `50711825` post-cachemetadata baseline (`61.67s`), so keep `SELECTOR_PQ_JOINT_FUSED_RISK_POLICY=0`.
- Dense-prefill V-PQ sidecar prewarm no longer forces `torch.cuda.synchronize()` unless profiling is enabled. This preserves sidecar contents and accounting semantics, but removes timing-only synchronization from no-profile benchmark runs. The pending 32k/128 gates will pick up this worktree change when they start.
- An opt-in diagnostic score-grid no-fill helper, `SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL=1`, now exists for the case where indexed pages plus base tokens cover the full context. It is explicitly rejected by `FRONTIER_CANONICAL_GPU=1` until parity/runtime validation promotes it. Compile-only Slurm build `50712158` passed on `standard` in `286s`; long-context trace parity job `50712317` is queued on `spgpu`.
- The no-fill helper is guarded by an explicit full-context coverage check. In both the HF benchmark path and saved-trace parity runner this guard now uses CPU-side index intervals and base-token metadata, avoiding the previous GPU coverage-mask plus `.item()` sync artifact while preserving the same semantic precondition. Lower-memory long parity duplicate `50714326` passed over decodes `32000,64000,128000`, heads `0,8`, with no failures, max CPU/native attention/o-proj relL2 `4.25e-09` / `1.70e-08`, and max Torch/GPU-policy attention/o-proj relL2 `2.62e-06` / `2.14e-06`. Runtime profile `50714329` completed: score `100.0`, decode `52.48s`, logical step `4.0836 MB/head-query`, selected `12463.75`. It is faster but changes accepted stats, so no-fill alone remains diagnostic.
- The native score-grid now treats rows with `k_take_count > ranked_prefix_width` as full exact rows. This repairs the earlier skipped-full-sort diagnostic failure mode where the full-budget row became partly PQ-scored. Unit coverage was added for normal and no-fill helpers; compile-only Slurm build `50712396` passed on `standard` in `264s`. Long-context trace parity job `50712458` passed for `SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT=1` over decodes `32000,64000,128000`, heads `0,8`, with no failures, max CPU/native attention/o-proj relL2 `4.25e-09` / `1.70e-08`, and max Torch/GPU-policy attention/o-proj relL2 `2.62e-06` / `2.14e-06`. Noncanonical 32k/16 profile job `50712539` completed: decode `54.82s`, rank-prefix `1.77s`, logical step `4.0836 MB/head-query`, selected `12463.75`. This is faster than current profile `50711825` but still changes accepted stats slightly, so keep it diagnostic for now.
- Decode-index cache metadata now refreshes `pending_start` and `indexed_end` on cached indexes every decode step. This preserves online pending-token coverage while sealed pages are reused; otherwise the static suffix can slide while cached index metadata stays stale. Canonical 32k/16 profile job `50711825` completed: score `100.0`, decode `61.67s`, logical step `4.0833 MB/head-query`, selected `12462.66`. This is slower than older exact-fullbudget profile `50708291` (`48.02s`) and shows slight accepted-stat drift, so treat it as the current post-cachemetadata baseline/regression to beat.
- The diagnostic grouped V-PQ sidecar cache now caches grouped `vhat`, residuals, and residual-risk stats together, so enabling it no longer forces a second per-head V-PQ sidecar pass in the decode loop. Fresh profile job `50711661` completed, but the repaired path is still negative: 32k/16 decode `58.07s`, logical step `4.0836 MB/head-query`, selected `12463.75`, worse and slightly different from the canonical exact-fullbudget point `48.02s`, `4.0815 MB`, selected `12455.16`. Keep `SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE=0`.
- `benchmark/audit_benchmark_readiness.py` now treats the current CPU-frontier semantics as the readiness contract. It flags non-`joint_kv_stability` confidence, non-`global_residual_risk` exact-V selection, non-fullscan selectors, approximate prefill, missing canonical guard, and non-affine tail-score calibration. It no longer flags exact logical cost accounting as a readiness failure.
- `benchmark/audit_benchmark_readiness.py` now reports decode seconds per example and physical GPU step MB, and accepts `--max-frontier-decode-seconds`. It also flags no-stats timing-only runs as `cost-stats-disabled` / `missing-step-cost`, and inactive sealed-page smokes as `approx-path-inactive` / `selector-inactive`. With a `54s` threshold, the old 32k/128 no-stats artifact `50707936` is correctly not ready: `decode=127.85s`, `cost-stats-disabled`, `missing-step-cost`, `decode>54.000s`.
- Benchmark summaries now include the native joint-K/V CUDA flag state in `pagedpq_config`, and readiness audit requires the promoted native CUDA flags while rejecting unpromoted diagnostic flags. This means older pre-metadata artifacts no longer prove readiness by themselves, even if they were run with canonical wrapper defaults. A synthetic current-format copy of the canonical 32k/16 artifact passes `--strict`; setting `fused_risk_policy=1` is correctly reported as `diagnostic-cuda-flags:fused_risk_policy`.
- Runtime jobs `50708850`, `50709244`, and `50709265` were canceled while still pending so CUDA unit validation could run first.
- Current-canonical 32k/128 gates are now queued:
  - `50711076`: no-stats timing, `ruler_eval_result/frontier_jointkv_profile_20260523/pagedpq_batched_niah_single_1_32768_n1_t128_exactfull_current_noprofile_nostats`;
  - `50711080`: paired accounting, `ruler_eval_result/frontier_jointkv_profile_20260523/pagedpq_batched_niah_single_1_32768_n1_t128_exactfull_current_accounting`.
- Current-canonical 32k/128 gates completed:
  - `50711076` no-stats timing: score `100.0`, generated `128`, prefill `28.65s`, decode `114.36s`, `893 ms/token`. This improves over prior `127.85s` but is still `6.36x` dense decode and above the `36-54s` target.
  - `50711080` paired accounting: score `100.0`, decode `116.99s`, logical step `3.8355 MB/head-query`, physical step `8.9179 MB/head-query`, selected `11728.16`, selector `0.4199 MB`, exact K/V `3.3357 MB`, tail `0.0799 MB`, update `0.0278 MB/head-query`.
  - Combined no-fill plus skip-full-sort profile `50714540` completed: score `100.0`, decode `52.84s`, logical step `4.0833 MB/head-query`, selected `12462.66`. This matches current post-cachemetadata selected/logical stats while improving 32k/16 runtime, so it is the next promotion candidate.
  - Combined long trace parity `50714942` passed: no failures over decodes `32000,64000,128000`, heads `0,8`; max CPU/native attention/o-proj relL2 `4.25e-09` / `1.70e-08`; max Torch/GPU-policy attention/o-proj relL2 `2.62e-06` / `2.14e-06`.
  - Combined 32k/128 accounting gate `50714944` rejected promotion: score `100.0`, decode `125.98s`, logical step `3.8366 MB/head-query`, physical step `8.9183 MB/head-query`, selected `11730.86`. This is slower and slightly higher-cost than current canonical `50711080` (`116.99s`, `3.8355 MB`, selected `11728.16`). Keep no-fill and skip-full-sort as diagnostics; pending no-stats/detail-profile jobs `50714945` and `50715090` were canceled.
  - Added profiler-only timing buckets for canonical HF runs: joint precompute, token-layout construction, grouped tensor packing, and grouped accounting/output selection. Local `py_compile`, wrapper audit, and `git diff --check` pass. Submitted canonical 32k/16 detail profile `50715246`.
  - Submitted fresh current-worktree canonical 32k/128 gates with diagnostics off: no-stats timing `50715264` and paired accounting `50715265`.
  - Fresh current-worktree canonical gates completed. `50715264` no-stats timing: score `100.0`, generated `128`, prefill `32.37s`, decode `107.84s` (`842.5 ms/token`). `50715265` accounting: score `100.0`, decode `123.08s`, logical step `3.8366 MB/head-query`, physical step `8.9183 MB/head-query`, selected `11730.67`, selector `0.4199 MB`, exact K/V `3.3368 MB`, tail `0.0799 MB`, update `0.0278 MB/head-query`. This is the current representative canonical gate and remains above the `36-54s` target.
  - Canonical 32k/16 detail profile `50715546`: score `100.0`, generated `16`, decode `55.98s`, logical step `4.0833 MB/head-query`, selected `12462.66`; native buckets were rank-prefix `3.94s`, residual-risk prefix `4.53s`, layout `1.66s`, score-grid `0.85s`, prob/base `0.75s`, policy `0.23s`, accounting `0.41s`.
  - Native rank-prefix diagnostic `SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX=1` passed unit (`50715889`), short parity (`50715928`), and long parity (`50715943` over decodes `32000,64000,128000`, heads `0,8`), but runtime lost: 32k/16 profile `50715944` decoded in `56.69s`, and 32k/128 no-stats `50715949` decoded in `118.93s` versus current canonical `107.84s`. Keep it off.
  - Residual-risk top-k prefix diagnostic `SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK=1` passed compile/unit (`50716260`, `50716266`) but did not improve the target subcomponent: 32k/16 profile `50716270` decoded in `55.12s`, preserved logical stats, but risk-prefix time worsened to `5.04s` versus canonical `4.53s`. Dependent 128-token jobs `50716271` and `50716492` were canceled; keep it off.
  - Parallel interval accumulation for the batched grouped residual-risk V-prefix helper is promoted as the current canonical implementation. It keeps the same sorted residual-risk order and V-budget semantics while parallelizing each V-budget interval across tokens and dimensions. Compile `50717368` and VPQ unit `50717376` passed. Batched microbench `50717536` improved the canonical helper shape from `8.39ms` grouped to `2.86ms` batched (`2.93x` versus grouped, `11.88x` versus repeated) with max abs diff `2.91e-05` versus the serial reference. Long trace parity `50717386` passed over decodes `32000,64000,128000`, heads `0,8`, no failures, max Torch/GPU-policy attention/o-proj relL2 `2.62e-06` / `2.14e-06`.
  - Runtime gate now passes. 32k/16 profile `50717396`: score `100.0`, decode `11.36s`, risk-prefix `1.71s`, rank-prefix `1.78s`, layout `1.72s`, score-grid `0.93s`. 32k/128 no-stats timing `50717408`: score `100.0`, prefill `15.04s`, decode `48.25s` (`376.9 ms/token`), `2.68x` the dense `17.99s` decode reference and inside the `<=54s` target. Paired accounting `50717414`: score `100.0`, decode `51.31s`, logical step `3.831 MB/head-query`, physical step `8.916 MB/head-query`, selected `11716.6`, selector `0.420 MB`, exact K/V `3.331 MB`, tail `0.080 MB`, update `0.0278 MB/head-query`. Strict readiness audit passes with no canonical warnings.
- Fast-token-layout is promoted into the canonical wrapper/guard. It preserves canonical token ordering for the contiguous sealed-page layout and removes Python token-list construction from the hot loop. Canonical 32k/128 fast-layout accounting job `50721909`: score `100.0`, decode `43.05s`, logical step `3.832 MB/head-query`, physical step `8.916 MB/head-query`, selected `11718.2`, selector `0.420 MB`, exact K/V `3.332 MB`, tail `0.080 MB`, update `0.0278 MB/head-query`; strict readiness audit passes. Paired no-stats job `50722253` decoded in `43.98s`. This is the current 32k/128 representative gate and supersedes `50717408`/`50720410` for latency.
- Sustained long-decode fast-layout validation now passes the 8192-token timing and accounting gates. Canonical guard-on no-stats job `50721920` generated `8192` forced tokens in `750.89s` versus dense `323.65s` (`2.32x`). Paired accounting job `50723383` generated `8192` forced tokens in `842.58s`, with logical step `1.762 MB/head-query`, physical step `1.872 MB/head-query`, selector `0.030 MB`, exact K/V `1.726 MB`, tail `0.0058 MB`, selected `4281.1` tokens, active fraction `0.359`, and strict readiness passing under the `971s` sustained threshold. The 16384 no-stats timing job `50723400` generated `16384` forced tokens in `2650.05s`, about `3.77x` the existing dense 16384 reference `702.13s`, so 16k sustained decode still needs more runtime work. The 6144-token wall-profile fast-layout diagnostic decoded in `490.41s` and reduced token-layout wall time from `23.98s` to `4.49s`; remaining runtime is distributed across score-grid, V-PQ sidecar, exact logits, prob/base, rank-prefix, and accounting.
- Current 8192-token wall profile `50728570` completed in `751.90s` generation time. The largest summed layer buckets are patched attention `499.48s`, score-grid `74.68s`, V-PQ sidecar `65.50s`, exact logits `45.76s`, prob/base `41.41s`, rank-prefix `36.83s`, and risk-prefix `13.90s`.
- Native `joint_softmax_base_outputs` is promoted into the canonical wrapper/guard as `SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE=1`. Build `50729310`, VPQ/unit `50729328`, and long trace parity `50729386` passed. Representative RULER 32k/128 gates passed: no-stats job `50730117` decoded in `44.02s`, and paired accounting job `50730118` decoded in `47.12s` with logical step `3.836 MB/head-query`, physical step `8.918 MB/head-query`, and selected `11730.3` tokens/head-query. Sustained LongGen 8192 gates also pass: no-stats job `50730291` generated in `710.06s` (`2.19x` dense), and accounting job `50730303` generated in `815.34s` with logical step `1.762 MB/head-query`. The current wall profile `50730305` shows prob/base reduced to `16.63s`, with the remaining larger buckets in score-grid, V-PQ sidecar, exact logits, and rank-prefix.
- V-PQ sidecar grow-pad diagnostic is running as job `50729770` with `SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD=8192`. This preserves semantics and tests whether avoiding repeated persistent sidecar realloc/copy cycles improves sustained decode. Manifest: `notes/slurm_manifests/vpq_growpad_validation_20260523.tsv`.
- Incremental V-PQ sidecar refresh across sealed-page cache keys is rejected. Job `50730919` completed on forced LongGenBench SGT-short 8192 with generation `791.29s`, joint total `422.76s`, V-PQ sidecar `69.12s`, score-grid `77.33s`, exact logits `52.90s`, prob/base `15.95s`, rank-prefix `39.42s`, and risk-prefix `18.28s`, which is not better than the native-softmax wall reference `50730305` (`788.07s`, V-PQ sidecar `69.71s`). The diagnostic is now behind `SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR=1`, default off and rejected by the canonical guard/readiness checks.
- Submitted current-canonical native-softmax sustained 16k validation with the incremental V-PQ diagnostic explicitly off. Manifest: `notes/slurm_manifests/current_native_softmax_16k_validation_20260523.tsv`; jobs `50730973` no-stats timing and `50730974` wall profile. Use these to replace the older fast-layout-only 16k timing result and identify the real 16k bottlenecks.
- Submitted one noncanonical grouped V-PQ cache diagnostic because V-PQ sidecar handling remains a large long-decode wall bucket. Manifest: `notes/slurm_manifests/grouped_vpq_cache_diagnostic_20260523.tsv`; job `50730991`, forced LongGenBench SGT-short 8192 wall profile, `FRONTIER_CANONICAL_GPU=0`, `SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE=1`. This is diagnostic only and must not be promoted without parity/accounting validation.
- Added diagnostic rank-position score-grid helper `SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID=1`. It keeps the same mixed score-grid semantics but replaces the large selected-token mask with rank positions plus a base mask. Canonical mode rejects it until validation. Local syntax/config checks pass. Manifest: `notes/slurm_manifests/rankpos_scoregrid_validation_20260523.tsv`; jobs `50731751` unit, `50731755` long parity, `50731756` 32k/128 timing, `50731869` 32k/128 accounting, and `50731891` LongGen 8192 wall profile.
- Rank-position score-grid is validated but not promoted on the 32k/128 gate so far: unit `50731751` passed, long parity `50731755` passed with no failures, no-stats `50731756` decoded in `45.14s`, and accounting `50731869` decoded in `48.72s` with logical step `3.8361 MB/head-query`. This is slightly slower than current canonical native-softmax (`44.02s` no-stats, `47.12s` accounting). LongGen 8192 wall job `50731891` is still running.
- Rank-position 8192 wall `50731891` completed in `734.36s`, but score-grid time was `82.33s` versus canonical native-softmax `76.48s`, so it does not solve the long-decode bottleneck. Keep `SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID=0`.
- Current-canonical native-softmax 16k sustained validation still fails the `<=3x` dense target: no-stats `50730973` generated 16384 forced tokens in `2565.56s` (`3.65x` dense `702.13s`), and wall `50730974` generated in `2686.69s`. Main 16k wall buckets: score-grid `497.84s`, rank-prefix `322.28s`, V-PQ sidecar `261.92s`, exact logits `198.98s`, prob/base `60.91s`.
- Grouped V-PQ cache remains rejected: accounting `50731290` generated 8192 tokens in `854.99s`, slower than canonical accounting `50730303` (`815.34s`), despite wall-profile V-PQ sidecar reduction in `50730991`.
- Added follow-up fused rank-position mixed-softmax/base helper under `SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID=1` plus `SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE=1`. It preserves the same rank-position selected-token semantics while avoiding full score-grid materialization before softmax/base output. Canonical mode still rejects fused softmax until validation. Manifest: `notes/slurm_manifests/rankpos_fused_softmax_validation_20260523.tsv`; unit `50732614` passed in `4:23`, long parity `50732809` passed with no failures and max Torch/GPU-policy attention/o-proj relL2 `3.00e-06` / `2.17e-06`, and runtime jobs `50732822`, `50732842`, `50732851` are pending.
- Fused rank-position is not a promotion candidate. On 32k/128, no-stats `50732822` decoded in `46.79s`, and accounting `50732842` decoded in `50.55s` with logical step `3.8361 MB/head-query`; both are slower than current canonical native-softmax. LongGen 8192 wall `50732851` generated in `750.18s`; prob/base dropped to `0.62s`, but score-grid rose to `93.13s`, so end-to-end remained worse. Keep `SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE=0`.
- Long-bottleneck diagnostics from `notes/slurm_manifests/long_bottleneck_diagnostics_20260523.tsv`: exact-logit dense-sim 8192 wall `50733642` generated in `784.42s` and is not better than canonical; native-rank-prefix 8192 wall `50733647` generated in `908.41s` and is negative; no-fill/skip score-grid 8192 wall `50733665` generated in `755.43s`, only a small noncanonical improvement. Combined native-rank-prefix plus no-fill/skip parity `50733673` passed, but pending runtime job `50733675` was canceled after native-rank proved negative. Exact dense-sim 16k timing `50733608` is still running.
- No-fill-only score-grid validation is now queued as the next promotion candidate because it should preserve canonical score semantics under full indexed/base coverage. Manifest: `notes/slurm_manifests/nofill_candidate_validation_20260523.tsv`; jobs `50734367` 32k/128 timing, `50734368` paired 32k/128 accounting, and `50734366` forced LongGenBench 16k timing. It remains diagnostic until runtime and logical stats are checked.
- A normal fused mixed-softmax/base diagnostic, without rank-position metadata, is also running. Manifest: `notes/slurm_manifests/fused_softmax_base_validation_20260523.tsv`; jobs `50733865` parity, `50733868` 32k/128 timing, `50733887` accounting, and `50733889` LongGen 8192 wall. This tests whether fusing the canonical selected-mask score-grid path helps, after the rank-position fused path proved negative.
- Current-code validation after the fast-layout state passed. Manifest `notes/slurm_manifests/frontier_current_validation_20260523.tsv`: CUDA unit job `50727996` passed all extension tests in `82s`; long native CPU-vs-GPU parity job `50728005` passed over decodes `32000,64000,128000`, heads `0,8`, with no failures and max Torch/GPU-policy attention/o-proj relL2 `2.62e-06` / `2.14e-06`.
- Reporting cleanup: no-stats benchmark summaries now include `approx_path_active_fraction` and fall back to approximate-call coverage for selector/tail active fractions when detailed cost counters are disabled. `benchmark/audit_benchmark_readiness.py` also reports the metric name for public benchmark quality, e.g. substring accuracy or pass@1. This only changes summary/reporting; it does not alter selector logic, accepted budgets, outputs, or MB accounting.
- Current passing-gate readiness audit is archived at `notes/archive/benchmark_audits_2026-05/readiness_fastlayout_gates_20260523.md`. It includes the canonical 32k/128 accounting gate and the canonical 8192 forced long-decode accounting gate, both with readiness `ok`.

## Current Task-Quality Validation

- Public task-quality validation is in progress for the passing canonical CUDA path.
- Active-path forced 8192-token AIME24 and LiveCodeBench dense/frontier slices were submitted in `notes/slurm_manifests/public_task_active_fastlayout_20260523.tsv` (`50723500`, `50723501`, `50723511`, `50723512`). These are intentionally forced to exercise sealed PQ pages; interpret task scores with that caveat.
- Forced active-path AIME24 completed with matching dense/frontier accuracy `100.0`. Dense job `50723500` generated `8192` forced tokens in `323.80s`; frontier job `50723501` generated `8192` forced tokens in `663.66s` (`2.05x` dense), with approximation active fraction `0.302`. This is active reasoning-quality smoke evidence, but only one forced sample.
- Forced active-path LiveCodeBench completed but is weak quality evidence because both dense and frontier pass@1 are `0.0`. Dense job `50723511`: forced `8192` generated tokens, generation `320.84s`. Frontier job `50723512`: forced `8192` generated tokens, generation `840.88s` (`2.62x` dense), approximation active fraction `0.328`.
- Follow-up forced-8192 LiveCodeBench paired shard set completed for offsets 1-4 while keeping sealed pages active: `notes/slurm_manifests/public_task_livecode_offsets1_4_force8192_fastlayout_20260523.tsv` (`50728180`-`50728187`). Offset 2 is the useful coding-quality pair: dense pass@1 `1.0` in `319.22s`; frontier pass@1 `1.0` in `773.82s` (`2.42x` dense), active fraction `0.312`. Offsets 1, 3, and 4 are runtime-only coding smokes because dense and frontier both failed pass@1; frontier generation times were `804.42s`, `801.19s`, and `879.70s`.
- Manifest `notes/slurm_manifests/public_longdecode_parallel_vprefix_smoke_20260523.tsv` launched dense/frontier AIME24, LiveCodeBench, and LongGenBench SGT-short smokes.
- AIME24 and LiveCodeBench completed but generated only `2048` tokens and had `approx_attention_calls_total=0`, so they prove wrapper execution only, not active frontier quality.
- LongGenBench dense/frontier jobs `50717675` / `50717676` are still running and should exercise sealed pages.
- A follow-up long-cap valid-quality slice was submitted in `notes/slurm_manifests/public_longdecode_longcap_parallel_vprefix_20260523.tsv`: AIME24 and LiveCodeBench with `MAX_NEW_TOKENS=8192`, no forced generation, dense jobs `50717945` / `50717947`, frontier jobs `50717946` / `50717948`.
- Long-cap AIME24 completed with matching dense/frontier accuracy `100.0`, but the model stopped at `4885` generated tokens, below the canonical `page_size=5632`, so frontier approximation remained inactive (`approx_attention_calls_total=0`). Dense generation took `206.11s`; frontier wrapper generation took `274.03s`.
- Long-cap LiveCodeBench completed with matching dense/frontier pass@1 `1.0`, but the model stopped at `2222` generated tokens, so frontier approximation also remained inactive (`approx_attention_calls_total=0`). Dense generation took `81.48s`; frontier wrapper generation took `81.96s`.
- LongGenBench dense/frontier SGT-short completed with forced `16384` generated tokens. Both had completion rate `1.0` and substring accuracy `0.0`, so this is active-path runtime/cost evidence but weak task-quality evidence because dense also failed the string-retrieval metric. Dense generation took `702.13s` (`13:19` Slurm elapsed). Frontier generation took `4059.36s` (`01:09:35` Slurm elapsed), about `5.8x` dense, so long continuous decode currently fails the `2-3x` dense runtime goal even though the 32k/128 RULER gate passes. Frontier cost: `400680` approximate attention calls, active fraction `0.679`, logical step `2.388 MB/head-query`, physical step `2.873 MB/head-query`, selected `6838.95` tokens/head-query, selector `0.085 MB`, exact K/V `2.286 MB`, tail `0.017 MB`, online update `0.0021 MB/attention-call`.
- Forced LongGen SGT-short 8192-token no-stats timing now has a clean dense/frontier pair. Dense job `50718580` generated `8192` tokens in `323.65s`; canonical frontier job `50718512` generated `8192` tokens in `1297.15s`, about `4.0x` dense, with `105768` approximate attention calls and logical step `1.592 MB/head-query`. This still misses the sustained-decode `<=3x` target.
- Forced LongGen SGT-short 6144-token wall profile job `50718677` completed in `460.73s` generation time. Wall-profile buckets over `32040` approximate attention calls: rank-prefix `40.99s`, policy `32.00s`, score-grid `25.19s`, V-PQ sidecar `21.99s`, layout `19.52s`, exact logits `14.57s`, prob/base `12.55s`, risk-prefix `4.79s`, accounting `3.55s`, selector `1.21s`. The bottleneck is distributed; prob/base is worth optimizing but cannot alone close the sustained-decode gap.
- Diagnostic native V-PQ base-output aggregation behind `SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE=1` was implemented and rejected. Compile `50719507` and targeted V-PQ unit `50719508` passed; active 4k/2 smoke `50719781` passed with `64` approximate calls. The 32k/16 profile `50719782` scored `100.0` but decoded in `13.17s`, slower than the current passing 32k/16 reference `50717396` (`11.36s`), with prob/base `3.38s` and slight logical-stat drift (`4.080 MB/head-query`, selected `12452.7`). Keep the flag default-off and rejected by canonical guard.
- Native no-MB grouped policy helper added for non-sensitivity policies. It preserves the same adaptive output-stability policy while avoiding per-call K/V MB tensors in canonical `k_first_alternating`; sensitivity-greedy still uses real MB tensors. Build `50720311` and targeted V-PQ/unit `50720312` passed. The 32k/16 profile `50720313` scored `100.0`, decoded in `10.59s`, logical step `4.081 MB/head-query`, selected `12453.5`, and policy bucket dropped to `0.033s`. The 32k/128 timing repeat `50720410` scored `100.0` and decoded in `53.88s`, barely inside the `<=54s` gate; paired accounting `50720347` preserved logical stats (`3.831 MB/head-query`, physical `8.916 MB/head-query`, selected `11716.7`) but decoded in `59.01s`. Treat this as a semantics-preserving native-boundary cleanup, not enough to close the sustained long-decode gap.

## KV-Compression Trace Comparison

Goal: compare selector-side frontier bandwidth reduction against KV-cache compression families on the same saved real Q/K/V trace, using mean step MB/head-query versus mean o-proj relL2 over layer 16, all heads, decode lengths `500..128000`.

Important caveat: these are paper-inspired compression proxies, not faithful KIVI/KVQuant/TurboQuant kernels. They are useful for first-pass MB-vs-relL2 positioning; task-level and faithful implementation comparisons are still required.

Artifacts:

- runner: `benchmark/selector_eval/runners/run_kv_compression_rel_l2_eval.py`;
- Slurm wrapper: `scripts/run_kv_compression_rel_l2_eval_one.sh`;
- manifest: `notes/slurm_manifests/kv_compression_rel_l2_20260522.tsv`;
- plot/table: `attention_efficiency_result/plots/kv_compression_only_vs_frontier_20260522/`;
- historical-overlay plot/table: `attention_efficiency_result/plots/kv_compression_vs_frontier_20260522/`.
- broad SOTA/proxy overlay: `attention_efficiency_result/plots/frontier_pareto_sota_existing_algorithms_0to10mb_20260523/`.

Best current comparison points:

| method | family | mean MB/head-query | mean o-proj relL2 | max o-proj relL2 | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Current adaptive K/V confidence | selector + V-PQ | `4.779` | `0.001118` | `0.002082` | best quality at similar bandwidth, but does not reduce KV-cache capacity |
| KIVI-like b4, 2048 exact window | scalar KV compression proxy | `5.224` | `0.013028` | `0.036735` | closest compression proxy quality, still much worse relL2 than current frontier |
| KIVI-like b4, 128 exact window | scalar KV compression proxy | `4.528` | `0.021423` | `0.041800` | similar MB to frontier but much worse relL2 |
| KVQuant-like b4 clipped, 128 exact window | clipped scalar proxy | `4.395` | `0.025512` | `0.060401` | lower MB than frontier, worse relL2 |
| PQ-like s8b6, 128 exact window | PQ/VQ compression proxy | `0.557` | `0.158352` | `0.270666` | very low MB but large output distortion |
| Dense fp16, mean over decode suite | dense | `17.201` | `0.0` | `0.0` | exact reference |

Conclusion so far: compression alone can drive MB extremely low, but the low-MB proxy points have high relL2. The current frontier remains much closer to dense output at comparable MB. The main competitive threat is not these naive proxies; it is faithful SOTA KV compression that may achieve lower relL2 while also reducing cache capacity, which our current bandwidth-only frontier does not.

2026-05-23 broader Awesome-KV-cache / SOTA coverage:

- Added trace-safe proxy curves for retention/pruning (`recent_k*`, `sink_recent_*`, `l2ret_*`, `h2o_k*`, `snapkv_k*`, `kvzip_k*`, `rocket_snap_k*`, KVPress-style `expected_attn_*`, `critical_snap_*`, `chunk_snap_*`, `keydiff_*`, `tova_*`, `cur_*`, `lagkv_*`, `compactor_*`), quantization (`pmkvq_like_*`, `kitty_like_*`, `tada_like_*`, `tiered_quant_*`), transform coding (`kvtc_like_*`, `freqkv_like_*`), sparse channels (`lookat_like_*`), merging (`zeromerge_like_*`, `cam_like_*`), VQ/dictionary (`commvq_like_*`, `lexico_like_*`), GEAR-style residual compression (`gear_like_*`), and low-rank compression (`lowrank_svd_*`).
- Coverage and caveats are tracked in `notes/kv_cache_compression_coverage.md`.
- Combined 0-10 MB plots and tables: `attention_efficiency_result/plots/frontier_pareto_sota_existing_algorithms_0to10mb_20260523/`.
- Slurm manifests: `notes/slurm_manifests/kvcomp_more_sota_curves_v2_20260523.tsv`, `notes/slurm_manifests/kvcomp_h2o_curves_20260523.tsv`, `notes/slurm_manifests/kvcomp_kvpress_curves_20260523.tsv`, `notes/slurm_manifests/kvcomp_kvpress_queryaware_split_20260523.tsv`, `notes/slurm_manifests/kvcomp_tada_curves_20260523.tsv`, `notes/slurm_manifests/kvcomp_remaining_sota_curves_20260523.tsv`.

Best representative points under `5 MB/head-query`:

| family | best point under 5 MB | mean MB/head-query | mean o-proj relL2 | max o-proj relL2 |
| --- | --- | ---: | ---: | ---: |
| Current frontier | `frontier_current_tau0.001` | `4.785` | `0.001034` | `0.002168` |
| Kitty-like channel-promoted scalar | `kitty_like_k4v4_p0.1_pb8_buf128_s32` | `4.828` | `0.009785` | `0.018728` |
| TaDA-like mean-centered scalar | `tada_like_b4_g64_w128` | `4.928` | `0.013670` | `0.031462` |
| KIVI-like scalar | `kivi_like_b4_w128` | `4.528` | `0.021423` | `0.041800` |
| KVQuant-like scalar | `kvquant_like_b4_clip0.1_w128` | `4.395` | `0.025512` | `0.060401` |
| PM-KVQ-like scalar | `pmkvq_like_b4_s1_g128_w128` | `4.616` | `0.037347` | `0.050926` |
| ZeroMerge-like merging | `zeromerge_like_k8192_tail4096_dense512_s0_obs64_ker5` | `3.950` | `0.066768` | `0.159438` |
| H2O trace retention | `h2o_k8192_obs288_ker1` | `3.935` | `0.073053` | `0.190629` |
| CaM-like merge/prune | `cam_like_k8192_merge128_obs288_ker5_w128` | `3.935` | `0.073330` | `0.184544` |
| Expected-attention trace retention | `expected_attn_k8192_obs288_cov0_v1_ker5_w128` | `3.935` | `0.074632` | `0.188648` |
| ChunkKV/SnapKV trace retention | `chunk_snap_k8192_c256_obs288_ker5_w128` | `3.805` | `0.079254` | `0.169599` |
| CommVQ-like residual VQ | `commvq_like_m4b8_w128` | `1.026` | `0.094960` | `0.243492` |
| KVTC-like transform coding | `kvtc_like_b5_r64_w128` | `2.826` | `0.095139` | `0.157750` |
| TurboQuant paper proxy | `tqpaperprod_k4v3_w128` | `4.595` | `0.124182` | `0.199471` |
| PQ-like fullscan compression | `pq_like_s8b6_w128` | `0.557` | `0.158352` | `0.270666` |
| LOOKAT-like sparse channels | `lookat_like_p0.25_b4_mean1_w128` | `2.527` | `0.232982` | `0.381224` |

Interpretation: the added SOTA/proxy families do not erase the gap. Kitty and TaDA are the strongest external compression-style points near `5 MB/head-query`, but they are still about an order of magnitude worse than the current frontier in mean o-proj relL2. Pruning/retention/merge methods cluster around `0.07-0.08` relL2 at about `4 MB`; transform coding is useful at low MB but not near the high-quality frontier. Compression-only families still matter because they reduce KV-cache capacity, while the current frontier is mainly a bandwidth-reduction algorithm.

2026-05-23 distribution/logit metric update:

- Added trace metrics for routing distortion: logit relL2, `KL(P_dense || P_approx)`, JS divergence, total variation, missing mass, and top-k probability recall. Summary rows now include p95/p99 for these metrics where applicable.
- Added the same metrics to the joint K/V frontier policy runner, measured on the mixed exact-K/K-PQ probability distribution used by the frontier algorithm.
- Representative plots: `attention_efficiency_result/plots/distortion_metrics_representative_20260523/`.
- Current frontier at threshold `0.001`: `4.785 MB/head-query`, mean o-proj relL2 `0.001034`, mean logit relL2 `0.03664`, mean probability JS `0.00012095`, mean KL `0.00067055`, top-512 mass recall `0.999999`.
- Strong compression baselines near the same MB are still worse on probability distortion: Kitty `4.828 MB`, JS `0.000252`, KL `0.001008`, o-proj relL2 `0.009785`; TaDA `4.928 MB`, JS `0.002257`, KL `0.009067`, o-proj relL2 `0.013670`; KIVI b4 `4.528 MB`, JS `0.002772`, KL `0.011147`, o-proj relL2 `0.021423`.
- Merge/prune methods that collapse multiple source tokens into one cache row do not always have a one-to-one token probability distribution. Probability/logit plots skip rows where `mean_token_probability_comparable < 1`; output-relL2 plots still include them.

## Frontier Pareto Sweep

Goal: test whether relaxing the current frontier quality target can beat low-MB PQ-like compression points. The sweep uses deployable knobs only: confidence threshold plus K/V budget ladders. Offline relL2 is only used for measurement, not for stopping.

Artifacts:

- manifest: `notes/slurm_manifests/frontier_pareto_20260522.tsv`;
- plot/table: `attention_efficiency_result/plots/frontier_pareto_20260522/`;
- plot/table with finer PQ-like points: `attention_efficiency_result/plots/frontier_pareto_with_fine_pq_20260522/`;
- plot/table with targeted 3-5 MB PQ-like points: `attention_efficiency_result/plots/frontier_pareto_with_pq_3to5mb_20260522/`;
- sweeps: `attention_efficiency_result/frontier_pareto_20260522/frontier_pareto_{current,low,tiny,ultra}_20260522/`.

Representative non-dominated points:

| point | mean MB/head-query | mean o-proj relL2 | max o-proj relL2 | note |
| --- | ---: | ---: | ---: | --- |
| PQ-like s8b6, 128 exact window | `0.557` | `0.158352` | `0.270666` | full compressed sidecar scan, much lower MB but higher error |
| Frontier ultra, `tau=0.512` | `1.374` | `0.085406` | `0.211079` | best relaxed frontier point near `0.1` mean relL2 |
| Frontier ultra, `tau=0.064` | `1.392` | `0.074518` | `0.165586` | lower error with similar MB |
| Frontier tiny, `tau=0.064` | `1.434` | `0.053015` | `0.098936` | good middle-quality point |
| Frontier low, `tau=0.016` | `1.633` | `0.026058` | `0.048455` | still much lower MB than high-quality point |
| Frontier low, `tau=0.004` | `2.416` | `0.008142` | `0.013446` | sub-`0.01` mean relL2 |
| Current reference, `tau=0.001` | `4.779` | `0.001118` | `0.002082` | high-quality reference |

Conclusion: relaxing confidence and budgets gives a useful frontier, but it does not beat PQ-like compression in raw MB at high-error targets. The relaxed frontier is better quality for higher MB; PQ-like is lower MB because it full-scans a much smaller compressed representation. The current frontier has a selector/sidecar floor around `1.37 MB/head-query` even with near-zero exact K/V budget. To compete with PQ-like at `relL2 ~= 0.1-0.15`, the next algorithmic target is reducing the fullscan K-PQ/V-PQ sidecar floor, not just relaxing confidence.

Fine-grained PQ-like follow-up:

| point | mean MB/head-query | mean o-proj relL2 | max o-proj relL2 | note |
| --- | ---: | ---: | ---: | --- |
| PQ-like s4b8, 128 exact window | `0.517` | `0.119938` | `0.226194` | better than s8b6 at similar MB |
| PQ-like s8b8, 128 exact window | `0.784` | `0.112429` | `0.193160` | more MB, modest quality gain |
| PQ-like s16b8, 128 exact window | `1.318` | `0.087796` | `0.143266` | best fine PQ point near frontier high-error region |
| Frontier ultra, `tau=0.512` | `1.374` | `0.085406` | `0.211079` | similar MB and slightly lower mean error, worse max error than s16b8 |
| Frontier ultra, `tau=0.064` | `1.392` | `0.074518` | `0.165586` | better mean error than fine PQ at similar MB |
| PQ-like s32b6, 128 exact window | `1.758` | `0.096740` | `0.141604` | dominated on mean error by s16b8 and frontier; more subvecs is not automatically better |

Updated conclusion: making PQ-like finer does produce a real quality/cost curve. Around `1.3-1.4 MB/head-query`, the relaxed frontier is competitive or better on mean o-proj relL2, while fine PQ can have better max relL2 for some points. This is a more favorable comparison for our approach than comparing only against low-cost s8b6.

Targeted `3-5 MB/head-query` PQ-like sweep:

| point | mean MB/head-query | mean o-proj relL2 | max o-proj relL2 |
| --- | ---: | ---: | ---: |
| PQ-like s64b4, 2048 exact window | `3.088` | `0.070970` | `0.195114` |
| PQ-like s32b8, 2048 exact window | `3.205` | `0.040817` | `0.099943` |
| PQ-like s64b6, 128 exact window | `3.358` | `0.060065` | `0.105036` |
| Frontier current ladder, `tau=0.004` | `3.394` | `0.003390` | `0.008249` |
| Frontier low ladder, `tau=0.001` | `3.640` | `0.002345` | `0.005075` |
| PQ-like s64b6, 2048 exact window | `4.120` | `0.040509` | `0.092187` |
| Frontier current ladder, `tau=0.002` | `4.120` | `0.001822` | `0.003737` |
| Current high-quality reference | `4.779` | `0.001118` | `0.002082` |

Conclusion for the region of interest: in the `3-5 MB/head-query` band, the adaptive frontier dominates these PQ-like fullscan compression proxies by a wide margin on output error. PQ-like remains compelling below about `1.5 MB/head-query`, but at moderate bandwidth our selector-plus-compression approach preserves much more attention output fidelity.

Current validation and runtime jobs:

| run | job | result | output |
| --- | ---: | --- | --- |
| CPU-vs-CUDA joint-K/V trace parity, decode 500/head 0 | `50645563` | passed; max attention/o-proj relL2 `0.0` | `attention_efficiency_result/joint_kv_cpu_gpu_parity_20260522/smoke_d500_h0` |
| CPU-vs-CUDA joint-K/V trace parity, decode 500/1000, heads 0/8 | `50645581` | passed; max attention relL2 `4.44e-09`, max o-proj relL2 `5.62e-09` | `attention_efficiency_result/joint_kv_cpu_gpu_parity_20260522/default_d500_1000_h0_8` |
| CPU-vs-Torch/GPU joint-K/V policy parity, decode 500/head 0 | `50665664` | passed; exact K/V budget and logical MB matches; max Torch/GPU attention relL2 `1.33e-07`, max o-proj relL2 `1.53e-07` | `attention_efficiency_result/joint_kv_cpu_gpu_parity_20260522/torch_gpu_policy_smoke_d500_h0_v2` |
| CPU-vs-Torch/GPU joint-K/V policy parity, decode 500/1000, heads 0/8 | `50665954` | passed; exact K/V budget and logical MB matches; max Torch/GPU attention relL2 `4.46e-07`, max o-proj relL2 `2.00e-07` | `attention_efficiency_result/joint_kv_cpu_gpu_parity_20260522/torch_gpu_policy_default_d500_1000_h0_8` |
| First HF canonical smoke, layer 16, 4k, 2 decode tokens | `50645631` | passed; logical step `1.229 MB/head-query` | `ruler_eval_result/frontier_jointkv_smoke_20260522/jointkv_l16_ctx4096_ps2048_n1_t2_v3` |
| Dense RULER reference, 32k, 128 decode tokens | `50645768` | passed; decode `17.99s`, score `100.0` | `ruler_eval_result/dense_jointkv_compare_20260522/dense_ctx32768_n1_t128` |
| Canonical HF 32k, 4 decode tokens, fp64-style pre-fp32 profile | `50645811` | passed but too slow; decode `282.33s`, score `100.0`, logical step `4.394 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_lazy_topk_profile` |
| Canonical HF 32k, layer 16, 1 decode token, fp32 probabilities | `50645834` | passed but too slow per layer; decode `3.59s`, logical step `3.192 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_l16_ctx32768_n1_t1_fp32_profile` |
| Canonical HF 32k, all layers, 4 decode tokens, fp32 probabilities | `50645835` | passed but too slow; decode `225.50s`, score `100.0`, logical step `4.393 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_fp32_profile` |
| Canonical HF 32k, all layers, 4 decode tokens, GQA-batched + fast index-cache | `50645917` | passed but still too slow; decode `100.44s`, score `100.0`, logical step `4.399 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_gqa_batched_fastindex_v2_profile` |
| Canonical HF 32k, all layers, 4 decode tokens, grid artifacts + vectorized residual risk | `50650042` | passed; decode `58.12s`, score `100.0`, logical step `4.397 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_gqa_vectorrisk_lazyv_profile` |
| Canonical HF 32k, all layers, 4 decode tokens, residual reuse | `50651432` | passed; decode `56.60s`, score `100.0`, logical step `4.397 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_gqa_residreuse_profile` |
| Canonical HF 32k, all layers, 4 decode tokens, prewarmed V-PQ sidecars | `50652948` | passed; decode `55.30s`, score `100.0`, logical step `4.397 MB/head-query`; diagnostic, not default | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_gqa_prewarmvpq_profile` |
| Canonical HF 32k, all layers, 4 decode tokens, residual reuse no-profile | `50653944` | passed; decode `60.45s`, score `100.0`, logical step `4.397 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_gqa_residreuse_noprofile` |
| Canonical HF 32k, all layers, 4 decode tokens, direct-gather affine calibration | `50664352` | negative; score `100.0`, decode `70.39s`, selected `13437.25`, changed accepted-budget stats; reverted | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_fastaffine_profile` |
| Canonical HF 32k, all layers, 4 decode tokens, fine-grained joint timing | `50664465` | passed; score `100.0`, logical step `4.397 MB/head-query`, selected `13446.875`; residual-risk prefix dominates | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_jointdetail_profile` |
| Canonical HF 32k, all layers, 4 decode tokens, segmented V-prefix diagnostic | `50664601` | negative for canonical parity; score `100.0`, decode `55.97s`, selected `13430.0`; keep opt-in/off | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_segmented_vprefix_profile` |
| Canonical HF 32k, all layers, 4 decode tokens, indexed calibration-mask shortcut | `50664906` | negative; score `100.0`, decode `69.50s`, selected `13437.25`; removed | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_indexed_calib_mask_v2_profile` |
| Native residual-risk V-prefix lower-resource unit gate | `50667861` | passed in `5:13`; `test_gpu_vpq_helpers.py` passed, including native V-prefix output-grid reference check | `cuda_unit_result/frontier_cuda_joint_vprefix_20260522_vpqonly_small` |
| Native residual-risk V-prefix lower-resource trace parity | `50667862` | passed; no failures, max CPU/native attention relL2 `4.44e-09`, max CPU/native o-proj relL2 `5.62e-09`, max Torch/GPU-policy attention relL2 `1.85e-06`, max Torch/GPU-policy o-proj relL2 `1.65e-06` | `attention_efficiency_result/joint_kv_cpu_gpu_parity_20260522/native_vprefix_after_50667861` |
| Native residual-risk V-prefix 32k/4 HF diagnostic | `50667903` | not promoted; score `100.0`, decode `63.38s`, logical step `4.391 MB/head-query`, physical step `9.044 MB/head-query`, selected `13424.5`; slower and budget-shifted versus historical canonical joint-detail `61.86s`, selected `13446.875` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_native_vprefix_after_50667862` |
| Native residual-risk V-prefix long-context trace parity | `50669624` | passed; no failures over decodes `32000,64000,128000`, heads `0,8`; max CPU/native attention relL2 `4.25e-09`, max CPU/native o-proj relL2 `1.70e-08`, max Torch/GPU-policy attention relL2 `2.72e-06`, max Torch/GPU-policy o-proj relL2 `2.18e-06` | `attention_efficiency_result/joint_kv_cpu_gpu_parity_20260522/native_vprefix_long_after_50667903` |
| Native V-prefix + prewarmed persistent V-PQ 32k/4 | `50669920` | diagnostic; score `100.0`, decode `51.30s`, prefill `29.57s`, logical step `4.391 MB/head-query`, physical step `9.044 MB/head-query`, selected `13424.5`; `2.85x` dense decode reference | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t4_native_vprefix_prewarmvpq_profile` |
| Promoted default RULER 32k/128 with accounting | `50671405` | failed runtime target; score `100.0`, decode `376.33s`, prefill `56.79s`, logical step `3.821 MB/head-query`, physical step `8.918 MB/head-query`; `20.9x` dense decode reference | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t128_promoted_default` |
| Promoted default RULER 32k/128 no-stats throughput | `50671500` | failed runtime target; score `100.0`, decode `368.43s`; `20.5x` dense decode reference | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t128_promoted_default_nostats` |
| Promoted default RULER 32k/16 profiled diagnostic | `50674457` | score `100.0`, decode `100.68s`; bottleneck is `native_geometric=35.86s`, especially `score_grid=15.11s` and `risk_prefix=17.06s` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_promoted_default_profile` |
| Native score-grid + risk-prefix short trace parity | `50701018` | passed; no CPU/native failures, max Torch/GPU-policy attention/o-proj relL2 `1.85e-06` / `1.66e-06` | `attention_efficiency_result/joint_kv_cpu_gpu_parity_20260522/native_score_grid_risk_prefix_smoke_lowmem` |
| Native score-grid + risk-prefix long trace parity | `50702996` | passed over decodes `32000,64000,128000`, heads `0,8`; max CPU/native attention/o-proj relL2 `8.44e-09` / `2.62e-08` | `attention_efficiency_result/joint_kv_cpu_gpu_parity_20260522/native_score_grid_risk_prefix_long_lowmem` |
| Native score-grid + risk-prefix 32k/16 profile | `50702455` | score `100.0`, decode `85.14s`, logical step `4.082 MB/head-query`; score-grid fell to `1.35s`, residual-risk prefix still `15.05s` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_native_score_grid_risk_prefix_profile_reuse_32g` |
| Lazy native score-grid + risk-prefix 32k/16 profile | `50703103` | negative; score `100.0`, decode `110.75s`, logical step `4.082 MB/head-query`; work moved into `native_joint_policy=54.70s` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_native_lazy_score_risk_profile_reuse_32g_t30` |
| Promoted default 32k/16, grid artifacts disabled | `50676531` | negative; score `100.0`, decode `169.31s`, `native_joint_policy=107.46s` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_promoted_default_nogrid_profile` |
| Promoted default 32k/16, lazy adaptive path | `50676533` | negative; score `100.0`, decode `145.28s`, `native_joint_policy=89.35s` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_promoted_default_lazy_policy_profile` |
| Coarse schedule 32k/16 profile | `50679445` | modest runtime win but higher MB; score `100.0`, decode `94.83s`, logical step `5.136 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_schedule_coarse_profile` |
| Coarse2 schedule 32k/16 profile | `50679449` | fastest 32k/16 joint-K/V point so far but much higher MB; score `100.0`, decode `87.74s`, logical step `6.806 MB/head-query`, selected `20131.3` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_schedule_coarse2_profile` |
| Coarse V8k schedule 32k/16 profile | `50680169` | fastest 32k/16 so far but high MB; score `100.0`, decode `83.60s`, logical step `6.522 MB/head-query`, risk-prefix `8.39s` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_schedule_coarse_v8k_profile` |
| Mid V8k schedule 32k/16 profile | `50680209` | negative; score `100.0`, decode `105.53s`, logical step `4.496 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_schedule_mid_v8k_profile` |
| Unsorted per-budget V-prefix 32k/16 profile | `50684318` | negative; canceled after `13:29` with no output artifact because it already exceeded the default 32k/16 runtime | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_unsorted_vprefix_profile` |
| Fast affine-selected score-grid 32k/16 profile | `50684812` | negative; score `100.0`, decode `98.88s`, logical step `4.079 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_fast_affine_selected_profile` |
| Coarse V8k + unsorted V-prefix 32k/16 | `50684985` | canceled stale diagnostic | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_coarse_v8k_unsorted_vprefix_profile` |
| Coarse V8k + fast affine-selected 32k/16 | `50685038` | faster only by paying higher logical MB; score `100.0`, decode `84.45s`, logical step `6.524 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_coarse_v8k_fast_affine_profile` |
| On-demand V-prefix adaptive walk 32k/16 | `50687944` | negative; score `100.0`, decode `101.72s`, logical step `4.080 MB/head-query` | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_ondemand_vprefix_profile` |
| Cached-logit canonical default 32k/16 | `50688219` | canceled stale diagnostic before native risk-prefix edits | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_cached_logits_default_profile` |
| Incremental V-grid adaptive walk 32k/16 | `50688781` | canceled stale diagnostic before native risk-prefix edits | `ruler_eval_result/frontier_jointkv_profile_20260522/jointkv_ctx32768_n1_t16_incremental_vgrid_profile` |
| Incremental V-grid 4k/2 smoke | `50689164` | canceled as redundant after lowmem duplicate started | `ruler_eval_result/frontier_jointkv_smoke_20260522/jointkv_ctx4096_n1_t2_incremental_vgrid_smoke` |
| Incremental V-grid 4k/2 smoke lowmem | `50689320` | completed but invalid as a frontier-path smoke: `PAGE_SIZE=5632` > context, so `approx_attention_calls=0` and `selector_active_fraction=0.0` | `ruler_eval_result/frontier_jointkv_smoke_20260522/jointkv_ctx4096_n1_t2_incremental_vgrid_smoke_lowmem` |
| Incremental V-grid 4k/2 smoke, page 2048 | `50691391` | pending; corrected smoke using a sealed PQ page and data reuse | `ruler_eval_result/frontier_jointkv_smoke_20260522/jointkv_ctx4096_ps2048_n1_t2_incremental_vgrid_smoke_lowmem` |

Local checks after wiring:

- `py_compile` passed for the HF runner, public/LongBench/RULER wrappers, and joint K/V trace runners.
- `bash -n` passed for the frontier wrapper scripts and the new parity Slurm wrapper.
- `benchmark/audit_benchmark_wrappers.py` passes with the new canonical defaults.
- Local CPU-mode parity-harness smoke on the small 16k trace passed with `max_attention_relative_L2=0.0` and no failures. This checks harness logic only.
- CUDA selector parity smoke `50645563` passed with `max_attention_relative_L2=0.0`, `max_oproj_relative_L2=0.0`, and no failures.
- Broader CUDA selector parity `50645581` passed with `max_attention_relative_L2=4.44e-09`, `max_oproj_relative_L2=5.62e-09`, and no failures.
- Torch/GPU policy parity `50665954` passed on the same decode `500,1000` / heads `0,8` slice. This now checks the benchmark-style mixed-logit grid, residual-risk exact-V ordering, V-PQ reconstruction, adaptive policy decisions, accepted K/V budgets, attention outputs, o-proj subset outputs, and logical MB against the CPU reference. Max Torch/GPU attention relL2 was `4.46e-07`; max Torch/GPU o-proj relL2 was `2.00e-07`.

## Joint-K/V HF Runtime Optimization Status

Positive semantics-preserving optimizations landed:

- Lazy joint K/V confidence simulation: compute only the adaptive path probes instead of the full K-budget by V-budget grid.
- V-risk prefix sums: one residual-risk ordering per K budget, then prefix residual outputs for all V budgets.
- GPU-side K-budget construction: avoid full ranked-token GPU-to-CPU copies in the canonical joint path.
- Bounded per-forward V-PQ cache: reuse V-PQ reconstruction/risk sidecars across GQA heads for one decode forward, without accumulating one cache entry per generated token.
- Split selector top-K from unsorted PQ tail scores: the selector no longer needs a full sorted ranking just because K-PQ tail logits need every token's approximate score.
- `SELECTOR_PQ_JOINT_FP32_PROBS=1` default for benchmark execution: same logical policy with fp32 score/probability/risk tensors instead of slow fp64 GPU tensors.
- GQA-batched canonical joint K/V path: process heads sharing one KV head together for K-PQ scores, mixed probabilities, residual-risk V selection, and V-PQ output construction.
- Fast decode index-cache reuse now applies to `joint_kv_stability`, not just the older selected-mass native path.
- Grid-artifact path: computes K/V budget output grids once per GQA group and runs the same adaptive policy over the small grid.
- Vectorized residual-risk code statistics: replaces Python/CPU page-code loops with GPU bucket reductions for `p_i^2 * V_PQ_error_stat_i`.
- Reduced V-PQ fallback gather: reconstructs PQ rows first and only gathers exact V for invalid/non-indexed fallback rows.
- Residual reuse: avoids recomputing full `V - V_PQ` inside the residual-risk statistic.
- Native residual-risk V-prefix: parity-validated against the CPU/Torch policy on small and long trace slices; avoids materializing the large gathered residual/cumsum tensors.
- Prewarmed persistent V-PQ sidecars: move V-PQ reconstruction sidecars before decode and keep them persistent for benchmark execution. This increases upfront/prefill-side work but is required for the current decode-runtime target.
- Score-only all-head K-PQ score precompute: removes repeated per-KV-head selector scoring and preserves accepted budgets/logical MB on the 32k/4 smoke, but total runtime remains dominated by sidecar/orchestration and adaptive grid work.
- Fine-grained joint-K/V timing counters: split score-grid, probability/base-output, residual-risk prefix, and policy work. These counters are diagnostic only and do not change canonical outputs or accounting.

Runtime evidence:

| run | decode | dominant costs | interpretation |
| --- | ---: | --- | --- |
| 4k all layers, 2 decode tokens, original profiled joint path | `53.77s` | geometric `33.56s`, output `9.43s`, sidecar `5.27s` | baseline HF canonical path was far too slow |
| 4k all layers, lazy joint path | `27.45s` | geometric `8.12s`, output `9.12s`, sidecar `5.07s` | lazy path cut unused K/V-grid work |
| 4k all layers, lazy + V-PQ cache | `20.70s` | output `2.62s`, geometric `8.01s` | bounded V-PQ cache is positive |
| 4k all layers, lazy + cache + K-budget dedupe | `18.10s` | geometric `4.09s`, output `2.87s` | dedupe helps when small contexts make multiple K budgets identical |
| 32k all layers, 4 decode tokens, pre-fp32 profile | `282.33s` | sidecar `117.35s`, geometric `87.92s`, output `20.18s` | current per-head torch simulator is not benchmark-ready |
| 32k layer 16, 1 decode token, fp32 probabilities | `3.59s` | geometric `0.995s`, output `0.910s`, sidecar `0.878s` | per-layer cost is still too high; need all-head/GQA-batched or native CUDA path |
| 32k all layers, 4 decode tokens, fp32 probabilities | `225.50s` | sidecar `86.33s`, geometric `78.01s`, output `15.40s` | fp32 helps but the benchmark path still needs batched/native canonical execution |
| 32k all layers, 4 decode tokens, GQA-batched + fast index-cache | `100.44s` | sidecar `27.67s`, geometric `31.98s`, output `24.62s` | current best canonical path, but still outside the `2-3x` dense target |
| 32k all layers, 4 decode tokens, grid artifacts | `72.63s` | sidecar `28.07s`, geometric `10.14s`, output `18.63s` | vectorized budget-grid construction cuts geometric policy work |
| 32k all layers, 4 decode tokens, vectorized residual risk + lazy exact-V fallback | `58.12s` | sidecar `26.89s`, geometric `11.51s`, output `2.87s` | removes the largest V-output/risk Python loop |
| 32k all layers, 4 decode tokens, residual reuse | `56.60s` | sidecar `29.08s`, geometric `10.08s`, output `2.02s` | current best non-prewarm profile |
| 32k all layers, 4 decode tokens, prewarmed V-PQ sidecars | `55.30s` | sidecar `29.86s` upfront/profiled, geometric `10.02s`, output `0.22s` | positive but just outside the `3x` dense target |
| 32k all layers, 4 decode tokens, native V-prefix + prewarmed persistent V-PQ | `51.30s` | sidecar `33.77s` upfront/profiled, geometric `7.33s`, output `0.22s`, risk-prefix `4.22s` | within `2-3x` dense for this short diagnostic only; superseded by 32k/128 failure |
| 32k all layers, 128 decode tokens, promoted defaults | `376.33s` | logical step `3.821 MB/head-query`, physical step `8.918 MB/head-query`, selected `11667.0` | failed representative runtime target; about `20.9x` dense decode despite correct task score |
| 32k all layers, 4 decode tokens, residual reuse no-profile | `60.45s` | profiling disabled | no hidden profiling-sync win; still too slow |
| 32k all layers, 4 decode tokens, all-head selector score precompute | `57.74s` | sidecar `27.78s`, geometric `10.97s`, output `2.39s`, selector `0.17s` | safe selector optimization, but not enough; preserves score/logical MB/selected tokens |
| 32k all layers, 4 decode tokens, all-head exact+selector precompute | `56.02s` | sidecar `26.39s`, geometric `10.69s`, output `2.63s`, exact `0.37s` | not promoted; accepted selected-token mean changed slightly (`13439.125` vs `13446.875`) |
| 32k all layers, 4 decode tokens, selected-gather affine calibration | `58.54s` | sidecar `29.01s`, geometric `10.82s`, output `2.45s` | negative; removed after profiling |
| 32k all layers, 4 decode tokens, direct-gather affine calibration | `70.39s` | sidecar `25.74s`, geometric `10.77s`, output `2.74s` | negative; reverted because selected-token mean changed (`13437.25` vs canonical `13446.875`) |
| 32k all layers, 4 decode tokens, fine-grained joint timing | `83.72s` | sidecar `28.77s`, geometric `11.80s`, score-grid `2.99s`, prob/base `0.93s`, risk-prefix `7.10s`, policy `0.71s` | diagnostic; preserves canonical selected/logical MB and identifies residual-risk top-k/prefix as the next native target |
| 32k all layers, 4 decode tokens, segmented V-prefix diagnostic | `55.97s` | sidecar `21.36s`, geometric `4.64s`, score-grid `2.31s`, prob/base `0.19s`, risk-prefix `1.66s`, policy `0.43s` | fast but not canonical; selected-token mean changed to `13430.0`, so keep `SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX=0` |
| 32k all layers, 4 decode tokens, indexed calibration-mask shortcut | `69.50s` | sidecar `22.98s`, geometric `10.99s`, score-grid `3.15s`, risk-prefix `7.06s` | negative; removed because selected-token mean changed to `13437.25`; likely duplicate indexed-token positions make token-to-ordinal shortcuts non-equivalent |
| 32k all layers, 4 decode tokens, count-only stats | `60.05s` | sidecar `28.35s`, geometric `12.06s`, output `3.10s` | no measured win; kept only where needed for avoiding task-run accounting overhead |
| 32k all layers, 16 decode tokens, normal accounting | `109.51s` | no-profile | score `100.0`, logical step `4.082 MB/head-query`, selected `12457.4`; amortizes to `6.84s/token`, still too slow |
| 32k all layers, 16 decode tokens, `DISABLE_COST_STATS=1` | `87.07s` | no-profile | score `100.0`; task-quality speed improves to `5.44s/token`, but MB fields are intentionally absent/zero |
| 32k all layers, 16 decode tokens, native score-grid + native risk-prefix | `85.14s` | job `50702455`, profiled | score `100.0`, logical step `4.082 MB/head-query`, physical step `8.973 MB/head-query`, selected `12456.6`; score-grid time drops from `15.11s` to `1.35s`; risk-prefix remains `15.05s` |
| 32k all layers, 16 decode tokens, native score-grid + native risk-prefix + native policy, stats disabled | `71.49s` | job `50704367`, profiled | policy time drops to `0.41s`; direct GPU final-output gather avoids the native-policy CPU index copy in no-stats task runs; MB fields are intentionally absent/zero |
| 32k all layers, 128 decode tokens, promoted native defaults, stats disabled | `270.75s` | job `50704671`, no-profile | improves over old no-stats promoted default `368.43s`, but still about `15.0x` dense decode (`17.99s`) and outside the `36-54s` target |
| Grouped residual-risk V-prefix microbenchmark, 32k-shaped | `8.38ms` | job `50705919` | exact match versus repeated helper (`max_abs_diff=0.0`); repeated per-group helper took `33.96ms`, so grouping is `4.05x` faster in isolation |
| 4k all layers, 2 decode tokens, opt-in grouped risk-prefix smoke | `7.06s` | job `50706019`, profiled, stats disabled | exercised approximation path (`approx_attention_calls_total=64`); grouped risk-prefix `0.099s`; correctness-only because 2-token score is not meaningful |
| 32k all layers, 16 decode tokens, opt-in grouped risk-prefix | `62.77s` | job `50706105`, profiled, stats disabled | positive; score `100.0`, risk-prefix drops to `4.53s`, geometric `17.12s`; still above the dense-runtime target |
| 32k all layers, 16 decode tokens, all-head selector top-k reuse | `64.02s` | job `50706486`, profiled, stats disabled | negative; selector time rose to `3.12s`; reverted |
| 32k all layers, 16 decode tokens, all-head selector top-k reuse accounting | `66.58s` | job `50706487`, profiled | negative; logical step `4.081 MB/head-query`, selected `12455.2`; reverted |
| 32k all layers, 16 decode tokens, grouped + layout cache + cached-V no full cast, stats disabled | `52.80s` | job `50706841`, profiled | positive; score `100.0`, patched attention `43.30s`, geometric `9.87s`, risk-prefix `4.95s`; MB fields intentionally zero |
| 32k all layers, 16 decode tokens, grouped + layout cache + cached-V no full cast, accounting | `53.63s` | job `50706843`, profiled | positive; score `100.0`, logical step `4.081 MB/head-query`, physical step `8.973 MB/head-query`, selected `12455.2`; same semantics/accounting as grouped baseline |
| 32k all layers, 16 decode tokens, full-budget sort elimination, stats disabled | `48.20s` | job `50707166`, profiled | faster but not promoted; it perturbs tie-sensitive accepted-token statistics in the paired accounting run |
| 32k all layers, 16 decode tokens, full-budget sort elimination, accounting | `56.51s` | job `50707182`, profiled | not canonical; logical step changed to `4.081782 MB/head-query` and selected mean to `12456.6` versus canonical `4.081499` / `12455.2` |
| 32k all layers, 16 decode tokens, ranked-prefix timing / all-head rank diagnostic | `55.20s` | job `50707496`, profiled accounting | not canonical; rank-prefix cost was only `3.27s`, and all-head top-k batching perturbed accepted stats (`4.081766 MB/head-query`, selected `12456.25`); follow-up all-head jobs `50707545` / `50707560` / `50707702` canceled |
| 32k all layers, 16 decode tokens, canonical no-profile/no-stats runtime | `54.64s` | job `50707774` | score `100.0`; no logical MB because cost stats disabled; shows instrumentation sync is not the main bottleneck |
| 32k all layers, 128 decode tokens, canonical no-profile/no-stats runtime | `127.85s` | job `50707936` | score `100.0`; improves over previous canonical no-stats `270.75s`, but still `7.1x` dense decode (`17.99s`) and outside the `36-54s` target |
| 32k all layers, 16 decode tokens, exact full-budget score-grid row, accounting | `48.02s` | job `50708291`, profiled | promoted; keeps full ranked-prefix sort for tie semantics, selected `12455.15625` and logical step `4.081499 MB/head-query` exactly match canonical |
| 32k all layers, 16 decode tokens, exact full-budget score-grid row, 32 GB duplicate | `55.80s` / `60.16s` | jobs `50708383` / `50708387`, profiled | same accounting on the accounting run; runtime varies by node/launch, but no semantic drift |
| 32k all layers, 16 decode tokens, grouped V-PQ residual cache + exact full-budget row | `58.35s` / `58.39s` | jobs `50708523` / `50708524`, profiled | negative; accounting matches canonical, but runtime worsens, so `SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE` is diagnostic-only and default off |
| 32k all layers, 16 decode tokens, vectorized final output gather | `109.32s` | no-profile, stats disabled | negative; removed after profiling |
| 8k all layers, 8 decode tokens, persistent V-PQ cache off/on | `59.82s` / `62.50s` | output `22.47s` / `23.03s`, geometric `22.01s` / `22.85s` | older isolated persistent-cache test; superseded by native V-prefix + prewarm result |

Current conclusion: canonical semantics are wired, and the native score-grid/risk-prefix/policy path has strong trace parity, but GPU benchmark readiness is still not achieved. Grouped risk-prefix plus layout/cache cleanup is positive: the current 32k/128 no-profile gate is `127.85s`, down from `270.75s`, but still above the `36-54s` target. Exact full-budget score-grid rows are now promoted because they preserve accepted stats and logical MB. Full-budget sort elimination and all-head rank-prefix batching remain diagnostics only because they change tie-sensitive accepted budgets. Grouped V-PQ residual caching also remains diagnostic-only because it slowed the HF profile despite matching accounting. The next canonical target is broader decode overhead and larger native fusion across score-grid/probability/V-prefix/policy boundaries.

## Pre-Joint-K/V GPU Baseline Results

These runs used the older geometric selected-mass exact-V path. They are useful runtime baselines and kernel-optimization history, but they are not evidence that the current residual-risk adaptive K/V frontier is benchmark-ready.

| run | job | result | logical step MB/head-query | selected tokens/head | runtime |
| --- | ---: | --- | ---: | ---: | ---: |
| RULER 32k, all layers, 128 decode tokens, wrapper defaults after native-threshold update, no profiling | `50585704` | score `100.0` | `5.920` | `15218` | decode `46.82s`; prefill `12.51s` |
| RULER 32k, all layers, 128 decode tokens, native threshold topk16384 + 512 threads, no profiling | `50585691` | score `100.0` | `5.921` | `15220` | decode `45.93s`; prefill `11.94s` |
| RULER 32k, all layers, 128 decode tokens, native threshold topk16384 + 512 threads, profiled | `50585696` | score `100.0` | `5.919` | `15214` | decode `49.82s`; prefill `11.08s` |
| RULER 32k, all layers, 128 decode tokens, native threshold topk16384 + 256 threads, no profiling | `50585687` | score `100.0` | `5.920` | `15219` | decode `46.07s`; prefill `13.86s` |
| RULER 32k, all layers, 128 decode tokens, native threshold topk16384 + 128 threads, no profiling | `50585667` | score `100.0` | `5.920` | `15218` | decode `47.19s`; prefill `10.85s` |
| RULER 32k, all layers, 128 decode tokens, native threshold topk16384 + 128 threads, profiled | `50585666` | score `100.0` | `5.920` | `15215` | decode `50.00s`; prefill `10.80s` |
| RULER 32k, all layers, 128 decode tokens, int32 exact lists + topk16384 + 128 threads, no profiling | `50584731` | score `100.0` | `5.922` | `15224` | decode `53.17s`; prefill `10.95s` |
| RULER 32k, all layers, 128 decode tokens, fixed wrapper defaults, no profiling | `50585404` | score `100.0` | `5.921` | `15221` | decode `53.28s`; prefill `12.29s` |
| RULER 32k, all layers, 128 decode tokens, exact lists + topk16384 + 128 threads, no profiling | `50584382` | score `100.0` | `5.922` | `15224` | decode `55.21s`; prefill `13.62s` |
| RULER 32k, all layers, 128 decode tokens, exact lists + topk8192 + 128 threads, no profiling | `50584224` | score `100.0` | `5.922` | `15222` | decode `55.97s`; prefill `15.72s` |
| RULER 32k, all layers, 128 decode tokens, exact lists + topk8192 + 128 threads, profiled | `50583015` | score `100.0` | `5.920` | `15217` | decode `56.54s`; prefill `12.49s` |
| RULER 32k, all layers, 128 decode tokens, exact lists + topk16384 + 128 threads, profiled | `50584324` | score `100.0` | `5.922` | `15222` | decode `54.84s`; prefill `10.93s` |
| RULER 32k, all layers, 128 decode tokens, exact lists + threshold topk8192, no profiling | `50582421` | score `100.0` | `5.919` | `15215` | decode `59.54s`; prefill `13.71s` |
| RULER 32k, all layers, 128 decode tokens, exact lists + threshold topk8192, profiled | `50582412` | score `100.0` | `5.922` | `15224` | decode `61.04s`; prefill `14.50s` |
| RULER 32k, all layers, 128 decode tokens, exact lists + ranked-gather logits, profiled | `50582426` | score `100.0` | `5.923` | `15225` | decode `65.56s`; prefill `10.94s` |
| RULER 32k, all layers, 128 decode tokens, selected code deltas, no profiling | `50577348` | score `100.0` | `5.922` | `15225` | decode `75.59s`; prefill `16.67s` |
| RULER 32k, all layers, 128 decode tokens, threshold topk8192, profiled | `50577734` | score `100.0` | `5.923` | `15225` | decode `75.06s`; prefill `15.47s` |
| RULER 32k, all layers, 128 decode tokens, threshold topk8192, no profiling | `50582374` | score `100.0` | `5.919` | `15212` | decode `75.94s`; prefill `11.06s` |
| RULER 32k, all layers, 128 decode tokens, selected code deltas, profiled | `50577178` | score `100.0` | `5.920` | `15216` | decode `76.61s`; prefill `12.24s` |
| RULER 32k, all layers, 128 decode tokens, precompute + 64 threads, no profiling | `50575165` | score `100.0` | `5.921` | `15220` | decode `79.57s`; prefill `12.72s` |
| RULER 32k, all layers, 128 decode tokens, precompute + 64 threads, profiled | `50574902` | score `100.0` | `5.922` | `15221` | decode `84.10s`; prefill `14.53s` |
| RULER 32k, all layers, 128 decode tokens, precomputed rank weights, no profiling | `50573421` | score `100.0` | `5.917` | `15208` | decode `84.55s`; prefill `15.51s` |
| RULER 32k, all layers, 128 decode tokens, precomputed rank weights, profiled | `50573204` | score `100.0` | `5.916` | `15206` | decode `86.79s`; prefill `15.20s` |
| RULER 32k, all layers, 128 decode tokens, fused dim-scan output, no profiling | `50567428` | score `100.0` | `5.916` | `15205` | decode `90.91s`; prefill `15.54s` |
| RULER 32k, all layers, 128 decode tokens, fused dim-scan output, profiled | `50567374` | score `100.0` | `5.919` | `15213` | decode `92.24s`; prefill `13.16s` |
| RULER 32k, all layers, 128 decode tokens, threshold-loop diagnostic | `50571660` | score `100.0` | `5.921` | `15220` | decode `111.78s`; prefill `14.55s` |
| RULER 32k, all layers, 128 decode tokens, dim-scan confidence only | `50566501` | score `100.0` | `5.922` | `15223` | decode `103.83s`; prefill `12.52s` |
| RULER 32k, all layers, 128 decode tokens, pre-dim-scan canonical | `50481532` | score `100.0` | `11.933` before accounting/fusion fixes | `15209` | decode `255.13s` |
| RULER 32k, layer 16 only, 1 token | `50460673` | completed | `49.370` before accounting fix | `11666` | native attention `65.15s` |

Interpretation:

- The canonical path no longer selects the full 32k context by accident.
- Corrected accounting removed exact-K double counting from the confidence/final-softmax path; logical frontier step cost is now about `5.9 MB/head-query`, while the GPU host physically reads about `11.9 MB/head-query`.
- The dim-scan CUDA confidence kernel plus fused candidate-output path reduced 32k/128-token decode from `255.13s` to `90.91s` without changing the logical frontier algorithm or task score.
- Native selected-mass threshold construction plus 512 geometric threads reduced the best no-profile 32k/128-token decode to `45.93s` with unchanged score/logical MB. A wrapper-default validation landed at `46.82s`, so the reference is now at the target boundary but still needs broader benchmark validation before broad claims.
- Dense 32k decode reference is about `18-23s`; the reference target is `27-46s`.

## Pre-Joint-K/V Broad Benchmark Baselines

These benchmark slices were run before the residual-risk adaptive K/V wrapper transition. They are old-frontier baselines, not current canonical task-quality evidence.

| run | dense result | frontier result | frontier cost | interpretation |
| --- | ---: | ---: | ---: | --- |
| RULER `niah_single_1`, 32k, 1 sample | score `100.0`, `28.20s/ex` | score `100.0`, `60.56s/ex` | `5.915` logical MB/head-query; selected `15202` | task quality matches dense; canonical path active |
| RULER `niah_multikey_2`, 32k, 1 sample | score `0.0`, `26.20s/ex` | score `0.0`, `63.34s/ex` | `7.448` logical MB/head-query; selected `20378` | dense and frontier fail the one-sample slice equally |
| LongBench-v2 short/easy, 4 examples, 8k input cap | accuracy `50.0`, `3.67s/ex` | accuracy `50.0`, `8.92s/ex` | `3.063` logical MB/head-query; selected `8213` | task quality matches dense on the slice |
| LongGenBench `sgt_short`, 8k forced decode, 1 example | generation `321.07s`; completion rate `0.538` | generation `720.83s`; completion rate `0.538` | `1.983` logical MB/head-query; selected `4730` | long-decode path runs end-to-end and exercises approximation (`105768` approx attention calls) |

Evidence:

- benchmark matrix manifest: `notes/slurm_manifests/frontier_benchmark_matrix_canonical_gpu_broad_20260521.tsv`;
- long-decode manifest: `notes/slurm_manifests/public_longdecode_canonical_gpu_20260521_8k.tsv`;
- long-decode frontier summary: `public_longdecode_result/canonical_gpu_20260521_8k/frontier_longgenbench_sgt_short_8k/summary.json`.

The broad slices are not a current task-quality claim. They prove the older selected-mass/geometric path can run RULER, LongBench-v2, and a public long-decode slice end-to-end with dense prefill, decode-only paged-PQ, V-PQ tail estimation, and separated logical-vs-physical accounting.

## Pre-Joint-K/V Runtime Bottleneck

Latest useful profiled run for the older selected-mass path (`50585696`, native threshold topk16384 + 512 threads):

- native attention path: `32.18s` total;
- geometric confidence/output fused portion: `16.91s`;
- selected-mass threshold generation: `5.93s`;
- exact ranked-logit materialization: `8.98s`;
- selector fullscan/top-k: `4.32s`;
- final output time is folded into geometric confidence (`native_output_seconds_total = 0.0`).

The current runtime target has shifted to the `joint_kv_stability` residual-risk path. Do not reuse the selected-mass native-path timings as proof that the new canonical path is within the dense `2-3x` runtime band.

Recent low-level checks:

- CUDA unit after the geometric-thread patch passed (`50567968`).
- A concurrent Slurm build race was found while sweeping; unit and exact-logit bench scripts now use the same CUDA extension build lock as the other CUDA scripts.
- The first geometric thread-count sweep was invalid because the benchmarked entry points still had hard-coded `256` threads; this was patched and rerun.
- Corrected microbench: `64` threads improved the synthetic 32k-shaped fused-output time from `37.5 ms` to `23.4 ms`, but the full 32k RULER path regressed. Full-path profiled decode: `64` threads `108.21s`, `128` threads `98.61s`, `256` threads `93.02s`. Keep the default at `256`.
- Threshold-loop selected-mass generation is a negative result: it preserved score/logical MB but increased threshold time from about `12.5s` to `29.4s` and decode from about `93.0s` to `111.8s`. Keep `SELECTOR_PQ_THRESHOLD_LOOP=0` except as a diagnostic.
- Precomputing ranked selected/PQ weights before fused dim-scan is enabled by default in the frontier wrappers via `SELECTOR_PQ_PRECOMPUTE_RANK_WEIGHTS=1`. CUDA unit passed (`50572242`); synthetic 32k-shaped fused-output microbench improved from `31.67 ms` to `24.56 ms`; canonical RULER improved from `90.91s` to `84.55s` no-profile and from `92.24s` to `86.79s` profiled with unchanged score/logical MB.
- After precompute, the synthetic thread-count microbench favored `64` threads (`14.26 ms`) over `128` (`18.63 ms`), `256` (`24.56 ms`), and `512` (`26.81 ms`). Full-path RULER confirmed the win: profiled decode `86.79s` -> `84.10s`, no-profile decode `84.55s` -> `79.57s`. Frontier wrappers now default `SELECTOR_PQ_GEOMETRIC_THREADS=64`.
- Selected compressed-V codeweight deltas are enabled by default in frontier wrappers via `SELECTOR_PQ_SELECTED_CODEWEIGHT_DELTAS=1`. The synthetic 32k-shaped fused-output microbench improved `14.26 ms` -> `10.51 ms`; full RULER improved profiled decode `84.10s` -> `76.61s` and no-profile decode `79.57s` -> `75.59s` with unchanged score/logical MB.
- Exact-preserving selected-mass threshold `topk` fast path (`SELECTOR_PQ_THRESHOLD_TOPK=8192`) preserved score/logical MB and reduced profiled threshold time `12.25s` -> `11.21s`; profiled decode was `75.06s`. The no-profile run was `75.94s`, so this is correct but within run-to-run noise rather than a standalone breakthrough.
- Selected exact-V numerator deltas are a negative result. CUDA unit/build passed (`50580029`), but the 32k-shaped fused-output microbench regressed from `10.51 ms` to `100.49 ms` (`50580037`). The shared-atomic exact-V aggregation costs more than the dim-scan work it replaces; keep `SELECTOR_PQ_SELECTED_EXACT_DELTAS=0`.
- Selected exact-V compact lists are geometry-sensitive and useful for the canonical page size. CUDA unit/build passed (`50581753`); on the generic 16x2048-page microbench they were neutral (`10.62 ms` vs `10.51 ms`, `50582368`), but on the actual RULER 5632-token page geometry they improved fused-output time `19.47 ms` -> `11.22 ms` (`50582378`, `50582379`). Full canonical RULER with `SELECTOR_PQ_SELECTED_EXACT_LISTS=1` plus threshold topk improved profiled decode `75.06s` -> `61.04s` and no-profile decode `75.59s` -> `59.54s`, with unchanged score/logical MB (`50582412`, `50582421`).
- Exact-logit backend A/B: ranked-gather is worse than dense-sim for the 32k/max-budget reference. It increased exact-logit time `9.59s` -> `14.72s` and decode `61.04s` -> `65.56s` (`50582426`). Keep `FRONTIER_EXACT_LOGIT_BACKEND=auto` / dense-sim behavior.
- Exact-list thread sweep on actual 5632-token pages: `32` threads `11.22 ms`, `64` threads `11.22 ms`, `128` threads `9.54 ms`, `256` threads `10.18 ms` (`50582694`, `50582379`, `50582695`, `50582716`). Full RULER with `128` threads improved profiled decode `61.04s` -> `56.54s`, no-profile decode `59.54s` -> `55.97s`, and geometric time `24.06s` -> `18.91s` (`50583015`, `50584224`). Frontier wrappers now default `SELECTOR_PQ_GEOMETRIC_THREADS=128`.
- Threshold topk sweep under exact-list/128-thread path: `topk2048` decode `59.93s`, `topk4096` `61.87s`, `topk8192` `56.54s`, `topk16384` `54.84s` (`50584338`, `50584320`, `50583015`, `50584324`). No-profile `topk16384` was `55.21s` (`50584382`). Smaller topk values likely fall back too often; wrappers now default `SELECTOR_PQ_THRESHOLD_TOPK=16384`.
- Diagnostic no-sync threshold topk (`SELECTOR_PQ_THRESHOLD_TOPK_ASSUME_SUFFICIENT=1`) is not useful: decode was `57.12s`, worse than safe `topk16384`, despite threshold time `9.71s` (`50584423`). Keep the safe fallback-proving path.
- Exact-list token storage as int32 is positive overall despite profile noise: actual-page microbench improved `9.54 ms` -> `9.13 ms` (`50584602`, `50584718`), the profiled RULER run was noisy/worse (`56.46s`, `50584722`), but no-profile improved to the current best `53.17s` with unchanged score/logical MB (`50584731`). Keep the int32 list representation.
- TF32 selector/simulator matmuls are a negative result for the canonical path: score stayed `100.0`, but decode regressed to `58.99s` and selector/threshold slices were slower (`50584734`). Keep `ALLOW_TF32_SELECTOR=0`.
- Wrapper default fix: `scripts/run_frontier_ruler_batched_one.sh` and `scripts/run_frontier_longbench_v2_one.sh` now default `ENABLE_FUSED_GEOMETRIC_OUTPUT=1` and `SELECTOR_PQ_FUSED_DIM_SCAN_OUTPUT=1`. Without this, targeted reruns can silently fall back to the slow separate-output path. No-profile wrapper-default validation preserved score/logical MB and matched the current best band: decode `53.28s`, logical step `5.921 MB/head-query`, selected `15221` (`50585404`).
- Exact-list capacity caps are negative for the current geometry. CUDA unit passed after the cap implementation (`50584789`), but actual-page microbench stayed best with uncapped/default lists: cap `0` `9.54 ms`, cap `2048` `24.14 ms`, cap `4096` `23.88 ms`, cap `8192` `23.03 ms` (`50584800`, `50584801`, `50584799`, `50584802`). The overflow fallback preserves semantics but loses the exact-list benefit.
- Threshold min-top proof fast path is negative so far. The fused-path profile preserved score/logical MB but regressed decode to `66.91s` and threshold time to `16.01s` (`50585113`). Keep `SELECTOR_PQ_THRESHOLD_MIN_TOP_FAST=0`.
- Max-budget/context cap attempts are negative at 32k. Host-side effective cap, context cap, and `GEOMETRIC_MAX_BUDGET=32768` preserved score but did not improve runtime: `56.55s`, `59.13s`, and `58.83s` decode respectively (`50585318`, `50585337`, `50585367`). Keep the existing max-budget defaults for the canonical path.
- Tail pages-per-block grouping is neutral/slightly negative on the actual six-page 32k geometry. Microbench: ppb `1` `9.16 ms`, ppb `2` `9.20 ms`, ppb `4` `9.26 ms`, ppb `8` `9.34 ms` (`50585394`, `50585396`, `50585393`, `50585395`). Keep `SELECTOR_PQ_DECODE_TAIL_PAGES_PER_BLOCK=1`.
- Geometric two-pass final-output reconstruction is a negative result. CUDA unit passed (`50585508`), but actual-geometry microbench regressed `9.14 ms -> 18.58 ms`, and RULER no-profile decode regressed to `55.61s` with unchanged score/logical MB (`50585550`, `50585551`, `50585552`). Keep `SELECTOR_PQ_GEOMETRIC_TWO_PASS_OUTPUT=0`.
- Proxy-gate reduction is safe but neutral. CUDA unit passed (`50585562`); actual-geometry microbench was effectively unchanged (`9.143 ms -> 9.134 ms`), and RULER no-profile decode was `53.47s` (`50585598`, `50585604`, `50585608`).
- Native selected-mass threshold construction from sorted top-k exact logits is positive. CUDA unit passed after exporting the new helper (`50585664`). With topk16384/128 threads, profiled threshold time dropped `10.11s -> 5.63s`, profiled decode improved `54.43s -> 50.00s`, and no-profile decode improved `53.47s -> 47.19s` with unchanged score/logical MB (`50585666`, `50585667`).
- Full-rank native top-k thresholds are a negative safety variant: guaranteed sufficient but too expensive. It preserved score/logical MB, but no-profile decode was `54.44s` and profiled threshold time was `11.88s` (`50585678`, `50585679`).
- Diagnostic no-sync native topk16384 is fastest but relies on a sufficiency assumption. It preserved score/logical MB on the reference and reached no-profile decode `44.87s`, but should remain diagnostic unless broader sufficiency is proven (`50585671`, `50585673`).
- Under native threshold topk16384, 512 geometric threads are the best current safe setting on the reference. No-profile decode: 64 threads `54.94s`, 128 threads `47.19s`, 256 threads `46.07s`, 512 threads `45.93s`; 512-thread profile has geometric time `16.91s` (`50585684`, `50585667`, `50585687`, `50585691`, `50585696`). Frontier RULER/LongBench wrappers now default `SELECTOR_PQ_GEOMETRIC_THREADS=512` and `SELECTOR_PQ_THRESHOLD_NATIVE_TOPK=1`.

## Current Next Tests

- For larger claims, scale sample counts/context lengths beyond the smoke slices above.
- Keep the safe native-threshold path as the default; do not promote the diagnostic no-sync threshold path unless broader sufficiency is proven.
- Keep logical-vs-physical MB reporting separate in all benchmark summaries.

## Joint K/V Confidence Policy

CPU trace diagnostic added in `benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py`.

Setup:

- trace: `real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz`;
- decode lengths: `500,1000,2000,4000,8000,16000,32000,64000,128000`;
- all 32 heads, layer 16;
- policy grid: K-first/V-first priority, K-first/V-first alternating, and sensitivity-greedy;
- V exact rows selected by global residual code-stat risk.

Main policy result:

| threshold | preferred policy | mean MB/head-query | max MB/head-query | mean o-proj relL2 | max o-proj relL2 | mean K budget | mean V budget |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0005` | K-first alternating | `5.621` | `19.543` | `0.000703` | `0.001338` | `13582` | `4880` |
| `0.001` | K-first alternating | `4.779` | `15.558` | `0.001118` | `0.002082` | `11492` | `3175` |
| `0.002` | K-first alternating | `3.856` | `15.558` | `0.002430` | `0.005383` | `8875` | `1664` |
| `0.004` | K-first alternating | `2.947` | `12.570` | `0.004885` | `0.011473` | `5618` | `956` |

Interpretation:

- K-first alternating, K-first priority, V-first alternating, and sensitivity-greedy are nearly identical on this trace. The simple alternating escalation is not a catastrophic policy.
- The practical slide point is threshold `0.001`: `4.779 MB/head-query` mean logical cost with `0.001118` mean o-proj relL2.
- The strict max-over-suite point for the same setting is `15.558 MB/head-query` and `0.002082` o-proj relL2.

Artifacts:

- `attention_efficiency_result/joint_kv_budget_policy_20260522/joint_kv_policy_full_20260522/summary.json`;
- `attention_efficiency_result/plots/mb_vs_relL2_current_20260522/mb_vs_relL2_slide.png`;
- `attention_efficiency_result/plots/mb_vs_relL2_current_20260522/mb_vs_relL2_strict_max.png`.

## V Exact-Set Strategy Diagnostic

CPU trace diagnostic added in `benchmark/selector_eval/runners/run_value_exact_strategy_eval.py`.

Question: after K selection, how much quality is lost by choosing exact V rows with the current selected-probability-mass heuristic, compared with better same-budget V exact-set choices?

Setup:

- trace: `real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz`;
- decode lengths: `500,1000,2000,4000,8000,16000,32000,64000,128000`;
- all heads, layer 16;
- V exact budget is fixed to the count selected by current `selected_mass=0.99`;
- strategies with `oracle` in the name are diagnostics only; they use exact V residuals to rank exact-V rows.

Summary, mixed probability source (`exact selected K logits + PQ tail logits`):

| K budget | V strategy | mean o-proj relL2 | max o-proj relL2 | mean exact V rows/head | exact V outside K |
| ---: | --- | ---: | ---: | ---: | ---: |
| `14336` | current selected-mass | `0.002092` | `0.003598` | `5104` | `0.0%` |
| `14336` | selected residual oracle | `0.001354` | `0.003552` | `5104` | `0.0%` |
| `14336` | global residual oracle | `0.001256` | `0.003264` | `5104` | `3.0%` |
| `14336` | all selected V exact | `0.001237` | `0.003549` | `13438` | `0.0%` |
| `32768` | current selected-mass | `0.001289` | `0.002025` | `6604` | `0.0%` |
| `32768` | selected residual oracle | `0.000425` | `0.001141` | `6604` | `0.0%` |
| `32768` | global residual oracle | `0.000400` | `0.001077` | `6604` | `0.5%` |
| `32768` | all selected V exact | `0.000264` | `0.001139` | `20493` | `0.0%` |

Interpretation:

- The current selected-probability-mass V heuristic leaves measurable quality on the table.
- Most of the gain comes from choosing exact V rows by V reconstruction residual inside the K-selected set, not from exact-reading many outside-K V rows.
- With a wider K prior (`32768`), outside-K exact V rows nearly disappear, but residual-aware same-budget V selection still improves o-proj relL2 by about `3x`.
- This supports a deployable next step: rank selected exact-V rows by a sidecar residual-risk estimate rather than probability mass alone.

Artifacts:

- `attention_efficiency_result/value_exact_strategy_20260522/vexact_strategy_full_k14336_20260522/summary.json`;
- `attention_efficiency_result/value_exact_strategy_20260522/vexact_strategy_full_k32768_20260522/summary.json`.

Follow-up deployable metadata sweep:

| K budget | V strategy | mean o-proj relL2 | max o-proj relL2 | metadata MB/head | interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| `14336` | current selected-mass | `0.002092` | `0.003598` | `0.0000` | baseline |
| `14336` | selected residual code-stat | `0.001350` | `0.003550` | `0.0130` | deployable, nearly matches selected scalar residual |
| `14336` | selected residual scalar | `0.001354` | `0.003552` | `0.0256` | deployable per-token scalar, same-budget upper target |
| `14336` | global residual scalar | `0.001256` | `0.003264` | `0.0672` | stronger but needs global metadata reads |
| `32768` | current selected-mass | `0.001289` | `0.002025` | `0.0000` | baseline |
| `32768` | selected residual code-stat | `0.000425` | `0.001141` | `0.0197` | deployable, closes most of the gap |
| `32768` | selected residual scalar | `0.000425` | `0.001141` | `0.0391` | deployable per-token scalar |
| `32768` | global residual scalar | `0.000400` | `0.001077` | `0.0672` | modest extra gain over selected-only |

Interpretation update:

- Cheap residual-risk metadata is enough to recover most of the V-PQ exact-set quality gap.
- Per-code residual stats are unexpectedly competitive with per-token scalar residuals at much lower metadata traffic.
- Post-projection diagonal weighting is almost identical to plain residual weighting in this run, so top-M `W_O` sensitivity is not urgent.
- The immediate practical rule should be selected-only residual code-stat or selected-only residual scalar, not global V routing.

Artifacts:

- `attention_efficiency_result/value_exact_strategy_20260522/vexact_metadata_full_k14336_20260522/summary.json`;
- `attention_efficiency_result/value_exact_strategy_20260522/vexact_metadata_full_k32768_20260522/summary.json`.

## Historical Source

The full pre-cleanup result log is preserved at `notes/archive/status_history/selector_eval_latest_results_2026-05-20_full.md`.
