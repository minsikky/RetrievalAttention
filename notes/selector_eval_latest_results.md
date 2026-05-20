# Selector-Eval Latest Results

Keep this page compact. Replace stale variants instead of appending every experiment.

## Active Evaluation Goal

Goal: get the current frontier algorithm into a state where we can comfortably run real benchmarks and trust the results. This phase is benchmark readiness and execution, not open-ended selector/compression search.

Benchmark-ready end state:

- The complete frontier algorithm runs end-to-end in real GPU inference through the benchmark wrappers with dense prefill and decode-only approximation: paged-PQ selection, online confidence/budgeting, selected attention, K/V compression, tail estimation, and online accounting are enabled for decode.
- Correctness is established well enough to trust benchmark results: CUDA/unit/parity tests pass, dense-vs-frontier smokes are stable, relL2/cosine/logit/hidden-state diagnostics are understood, and there is no passthrough masking, hidden dense fallback, stale cross-sample state, oracle leakage, or unexplained output drift.
- GPU performance is practical enough for real evaluation: sampled validation shows the selected LongBench/RULER benchmark scope can finish inside normal Slurm jobs without manual babysitting. If runtime is too high, optimized CUDA/backend work is the blocker and benchmark claims should wait.
- Accounting is honest and separated: selector MB, exact KV MB, compressed KV MB, tail-estimator MB, online-update MB, runtime, and task accuracy are reported with explicit units. Snapshot and online costs must not be mixed.
- All deployed decisions use only online inference-time information. The selector/confidence/compression path must not use oracle attention probabilities, dense rankings, achieved mass, dense-reference outputs, or task answers.
- Once the correctness, speed, and accounting gates pass, actually run the benchmark matrix: dense/reference, RetroInfer-style where available, and frontier evaluations on LongBench and RULER. The output should answer the research question: how do attention/layer relL2, cosine, and logit drift translate to task-level accuracy?

New variants are useful only if they remove a blocker to benchmark readiness: correctness, deployability, GPU speed, or accounting.

Current status: the benchmark-facing GPU frontier path is now the CPU-frontier semantic path, not the old fast selector-rank preset. `FRONTIER_CANONICAL_GPU=1` is enabled by default in the frontier wrappers and fails fast unless the run uses decode-only fullscan paged-PQ on CUDA, `geometric_probe_tail_switch`, V-PQ tail, selected-mass selected-V exactness, proxy-mass/PQ-correlation gates, `native_decode_tail`, `index_build_backend=torch_gpu`, and exact accepted-budget accounting. The old selector-rank/fixed native accept-count variants remain diagnostic only and are not the canonical benchmark path.

Canonical GPU smoke status, 2026-05-19:

| run | job | status | key result | interpretation |
| --- | ---: | --- | --- | --- |
| RULER canonical GPU, 4k, n=1, 1 token | `50460026` | completed | guard accepted config, but no sealed page existed; `approx_attention_calls_total=0` | Valid guard smoke only; did not exercise selector/tail. |
| RULER canonical GPU, 8k, n=1, 1 token | `50460136` | completed | `approx_attention_calls_total=32`, `confidence_active_fraction=1.0`, mean selected `7885`, step `13.700 MB/head-query`, native attention `55.05s` | Confirms the canonical selected-mass/proxy-gated GPU path runs end-to-end; also confirms it is slow. |
| RULER canonical GPU, 32k, all layers, n=1, 1 token | `50460266` | canceled after `20.5 min` | no summary before cancellation | Redundant after layer-16 result; canceled to release GPU. Confirms all-layer strict selected-mass geometric is still not benchmark-comfortable. |
| RULER canonical GPU, 32k, layer 16 only, n=1, 1 token | `50460673` | completed | mean selected `11666` tokens/head, step `49.370 MB/head-query`, selector `0.420`, exact KV `4.298`, tail `0.260`, confidence `44.392`, native attention `65.15s` | Confirms canonical CPU-frontier semantics do not select all `32k`; the remaining blocker is confidence runtime/cost, not wrong native selector-rank semantics. |
| CUDA unit, fused geometric output reuse | `50481456` | passed | fullscan top-k, V-PQ helper, fused geometric output parity, online append all passed in `80s` | The fused final-output implementation is functionally valid. |
| RULER canonical GPU, 32k, all layers, n=1, 32 decode tokens, corrected accounting | `50481630` | completed | score `100.0`; mean selected `15333`; step `11.948 MB/head-query`; selector `0.420`, confidence `9.731`, exact KV `1.538`, tail `0.259`, update `0.111`; decode `69.14s` | Exact selected K is no longer double-counted when exact ranked logits are already computed for confidence/final softmax. Runtime remains essentially unchanged from the prior canonical run; this is an accounting/cost-model correction, not a speedup. |
| RULER canonical GPU, 32k, all layers, n=1, 128 decode tokens, corrected accounting | `50481532` | completed | score `100.0`; mean selected `15209`; step `11.933 MB/head-query`; decode `255.13s` | Longer decode stress gives the same corrected per-query cost shape; runtime still scales with the current canonical attention kernel. |

Fused-output runtime status, 2026-05-19: `gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds` reuses exact ranked logits and tail code-weight state in one CUDA final-output call, but the first implementation is slower than the existing canonical path on the 32k RULER smoke. Job `50481621` was canceled after `>6 min` without a prediction row, while the old path completed in `2:40`. The fused path is now opt-in via `ENABLE_FUSED_GEOMETRIC_OUTPUT=1`; default benchmark runs use the faster canonical path plus the corrected exact-K accounting.

## Benchmark Readiness, 2026-05-18

Latest 32k downstream smoke:

| run | job | status | score | runtime | key accounting | interpretation |
| --- | ---: | --- | ---: | ---: | --- | --- |
| Dense RULER niah_single_1, 32k, n=1 | `50417590` | completed | `100.0` | generation `112.40s`; prefill `93.60s`; decode `18.72s` | dense reference | Valid reference row. |
| Strict geometric frontier RULER niah_single_1, 32k, n=1 | `50417591` | timed out | n/a | `01:00:29`, no prediction | selected-mass V exactness + proxy-gated geometric confidence | Not benchmark-ready. It misses the fast native accept-count path and falls back to repeated confidence tail attention. |
| Native geometric frontier RULER niah_single_1, 32k, n=1 | `50422703` | completed output, then canceled to release GPU | answer found | generation `340.50s`; prefill `85.04s`; decode `254.75s` | step `18.606 MB/head-query`; selector `1.055`; exact KV `8.454`; tail `0.322`; confidence `8.776`; update `0.032`; selected tokens `32545.5` | Native accept-count path works, but the strict geometric threshold accepts almost the whole 32k context and is slower/more expensive than dense. |
| Fast frontier RULER niah_single_1, 32k, n=1 | `50420978` | completed | `100.0` | generation `125.05s`; prefill `105.34s`; decode `15.91s` | step `2.362 MB/head-query`; selector `1.055`; exact KV `0.985`; tail estimator `0.322`; update `0.032`; selected tokens `2017.5` | Benchmark-usable fast preset. This is not strict geometric CPU parity; it is fixed-budget ranked-mass with V-PQ selected/tail compression. |
| Dense LongBench-v2 short/easy, 32k, n=2 | `50417592` | completed | `100.0` | generation `142.30s` avg | dense reference | Valid small reference row; one prompt truncated to 32k. |
| Fast frontier LongBench-v2 short/easy, 32k, n=2 | `50421408` | completed | `50.0` | generation `40.86s` avg | step `1.662 MB/head-query`; selector `0.858`; exact KV `0.542`; tail `0.262`; update `0.096` | Not reliable on this smoke: one dense-correct row flips from `B` to `A`. |

Fast frontier preset used here: dense prefill, decode-only paged-PQ fullscan selector, `pq_ranked_mass_budget`, `BUDGET=192`, `PAGE_SIZE=2048`, `PREFILL_CHUNK_SIZE=512`, native prefill LUT path, `selected_value_mode=vpq_value`, `selected_value_exact_rule=selector_rank`, `selected_value_exact_top=256`, V-PQ tail estimator enabled.

Strict geometric blocker: the CPU-parity configuration uses `geometric_probe_tail_switch` with `selected_value_exact_rule=selected_mass` and proxy gates. The current GPU wrapper only uses the native geometric accept-count path for selector-rank/fixed exactness without proxy gates; the strict path therefore repeatedly evaluates tail-producing attention and is too slow for 32k benchmark execution.

Follow-up diagnostics:

- Patched the decode wrapper so native geometric accept-counts can propose a budget even when proxy gates are enabled, then the proxy gate validates that proposed budget once. This targets selector-rank/fixed V exactness variants; it does not make strict selected-mass parity fast yet.
- `50422703`: 32k RULER geometric confidence, selector-rank exact V, no proxy gate. It produced the right NIAH answer, but selected `32545.5` tokens on average and cost `18.606 MB/head-query`; canceled after summary write to release the GPU.
- Canceled `50422881`: proxy-gated geometric confidence was redundant after `50422703`; the 0.99 proxy gate is expected to be at least as conservative as the no-proxy run that already selected nearly the full context.
- Submitted `50424415`: geometric accept-count CUDA microbench at 32k-like dimensions to isolate native confidence-kernel runtime from model generation overhead.
- Submitted loosened geometric threshold sweep, reusing the same 32k NIAH data:
  - `50424705`: `tail_probe_rel_l2_max=0.08`; completed, score `100.0`, but cost stayed near-dense: step `18.606 MB/head-query`, selector `1.055`, exact KV `8.454`, tail `0.322`, confidence `8.776`, selected tokens `32545.5`.
  - `50424706`: `tail_probe_rel_l2_max=0.12`; completed, score `100.0`, and matched the same near-dense cost/selection as rel `0.08`.
  - `50424707`: `tail_probe_rel_l2_max=0.20`; completed, score `100.0`, with the same near-dense cost/selection as rel `0.08/0.12`: step `18.606 MB/head-query`, selected tokens `32545.5`.

LongBench-v2 fast preset result: job `50421408` completed predictions for 2 rows and scored `50.0%` vs dense `100.0%`. Row `66f36490821e116aacb2cc22` matched dense (`D`, correct). Row `66fcf2f2bb02136c067c9169` flipped from dense `B` correct to frontier `A` wrong. Aggregate cost was step `1.662 MB/head-query` with selector `0.858`, exact KV `0.542`, tail estimator `0.262`, update `0.096`, selected tokens `1109.6`.

Changed-row diagnostics submitted for `66fcf2f2bb02136c067c9169`:

- `50423188`: budget `384`, selected V exact top `512`; completed with usable summary before wrapper shutdown error; predicted `B` and scored correct. Cost: step `1.407 MB/head-query`, selector `0.773`, exact KV `0.397`, tail `0.236`, selected tokens `813.5`.
- `50423193`: budget `512`, selected V exact top `512`; completed with usable summary before wrapper shutdown error; predicted `A` and stayed wrong. Cost: step `1.467 MB/head-query`, selector `0.773`, exact KV `0.458`, tail `0.236`, selected tokens `937.0`.
- `50423280`: budget `384`, selected V exact; canceled after prolonged slow generation. This path was redundant because `b384_v512` already makes all selected dynamic V exact (`exact_top=512 > budget=384`) while still using the production V-PQ selected/tail code path.
- `50423521`: budget `192`, selected V exact; canceled as redundant. The original fast frontier row already had `budget=192` and `selected_value_exact_top=256`, so selected dynamic V was exact there too.
- `50428055`: submitted LongBench-v2 short/easy `n=2`, budget `384`, selected V exact top `512`, `MAX_INPUT_TOKENS=32768`, with `STAGE_MODEL_TO_TMP=1`. It failed before model load because staged HF snapshot symlinks pointed to missing `/tmp/.../blobs` paths. Fixed `benchmark/run_longbench_v2_hf.sh` to dereference symlinks during staging (`rsync -aL` / `cp -aL`) and resubmitted as `50433117`.
- `50433117`: LongBench-v2 short/easy `n=2`, budget `384`, completed with accuracy `100.0%`, recovering the previously flipped `66fc...` row (`pred=B`, `answer=B`) while preserving the first row (`pred=D`, `answer=D`). Cost: step `1.719 MB/head-query`, selector `0.849`, exact KV `0.611`, tail `0.259`, update `0.085`, selected tokens `1250.8`, avg generation `42.14s/example`. Audit saved at `notes/lbv2_budget384_n2_rerun_audit_20260518.md`.
- Broader budget384 validation completed: paired LongBench-v2 short/easy `n=16`, max input `32768`, dense `50433593` and frontier `50433618`. Dense and frontier both scored `11/16 = 68.75%`; predictions matched `16/16`, judge labels matched `16/16`, and responses matched `14/16`. Frontier cost: step `1.858 MB/head-query`, total including online update `1.954`, selector `0.736`, exact KV `0.897`, tail `0.225`, update `0.097`, selected tokens `1837.1`, avg generation `13.95s/example` vs dense `11.35s/example`. Audit: `notes/lbv2_budget384_n16_32k_audit_20260518.md`; comparison: `notes/lbv2_budget384_n16_32k_compare_20260518.txt`.
- Full short/easy budget384 validation completed: LongBench-v2 `MAX_EXAMPLES=64`, max input `32768`, dense `50434203` and frontier `50434204`. Dense and frontier both scored `27/59 = 45.76%`; predictions matched `59/59`, judge labels matched `59/59`, and responses matched `48/59`. Frontier cost: step `2.057 MB/head-query`, total including online update `2.141`, selector `0.820`, exact KV `0.986`, tail `0.250`, update `0.0838`, selected tokens `2019.4`, avg generation `13.66s/example` vs dense `10.20s/example`. Audit: `notes/lbv2_budget384_n64_32k_audit_20260518.md`; comparison: `notes/lbv2_budget384_n64_32k_compare_20260518.txt`.
- RULER 32k budget384 matrix completed: paired dense/frontier runs for `niah_single_1`, `niah_multikey_2`, `vt`, and `fwe`, `n=4`. The first staged-model batch `50434947-50434954` was canceled because each job spent GPU time copying the model to `/tmp` without producing artifacts. Resubmitted no-stage jobs `50435100-50435107`; all completed cleanly. Results: `niah_single_1` dense/frontier `100/100`, `vt` `95/100`, but `niah_multikey_2` drops `75 -> 50` and `fwe` drops `100 -> 91.67`. Frontier cost range: step `2.213-2.512 MB/head-query`, selector about `1.02-1.055`, exact KV `0.882-1.136`, tail `0.311-0.322`, update `0.0315-0.1345`, selected tokens `1806-2326`. Audit: `notes/ruler32k_b384_n4_nostage_audit_20260518.md`; comparison: `notes/ruler32k_b384_n4_nostage_compare_20260518.txt`.
- RULER 32k targeted follow-up completed for the two degraded tasks: frontier-only `BUDGET=512/768` with exact selected-V top equal to budget, reusing the already generated dense data files. Larger budget did not recover quality: `niah_multikey_2` stayed `50.0` at both `512` and `768`; `fwe` stayed `91.67` at both `512` and `768`. Audit: `notes/ruler32k_b384_failure_budget_sweep_audit_20260518.md`.
- RULER 32k failure diagnostics completed: `BUDGET=2048` with current V-PQ tail, and `BUDGET=768` with exact selected values plus tail disabled, for `niah_multikey_2` and `fwe`. `niah_multikey_2` recovered to dense score at `BUDGET=2048` (`75.0`) but stayed `50.0` with no-tail `BUDGET=768`, so the multikey failure is selected-K coverage, not tail/V compression. `fwe` stayed `91.67` for both `BUDGET=2048` with tail and no-tail `BUDGET=768`, so its drop is not fixed by the current tail toggle or moderate budget increase. Audit: `notes/ruler32k_failure_diagnostic_audit_20260518.md`.
- Output inspection: the FWE regression is one missed target word out of 12 total target words across 4 examples, not a broad generation collapse. Dense contains all three target words for each example. Frontier misses one target on FWE example 1 under `BUDGET=384`, `768` no-tail, and `2048` tail, although the missed word differs for no-tail. Multikey has the same miss as dense on example 0; the extra frontier miss on example 3 disappears at `BUDGET=2048`.
- RULER 32k threshold refinement submitted: `niah_multikey_2` at `BUDGET=1024/1536` with tail, plus `fwe` at `BUDGET=2048` no-tail and `BUDGET=4096` with tail. Jobs `50438609-50438612`; manifest `notes/slurm_manifests/ruler32k_threshold_refine_20260518.tsv`.

Geometric confidence follow-up:

- `50424887`: native accept-count microbench completed. The fused confidence-count kernel took about `55.45 ms` for a 32k-shaped query with 32 heads, but strict geometric still accepted full `32768` for every head at `tail_probe_rel_l2_max=0.04`.
- Attribution update, 2026-05-19: the near-full strict-geometric behavior is not explained by flat attention or by PQ rank alone. On the saved real trace, replacing PQ rank with oracle rank barely changed budgets: strict V-PQ at `tail_probe_rel_l2_max=0.20` selected `8.4k` tokens/head mean for both PQ and oracle, while exact-delta selected `15.9k` oracle vs `16.1k` PQ. On the exact 32k RULER first decode step, oracle exact-delta selected `5.9k` tokens/head and PQ exact-delta selected `7.9k`, but the production native strict V-PQ confidence path selected `32.5k`. Native tail-stability also selected `32.5k`. Conclusion: the immediate culprit is the compressed-tail confidence/estimator path: exact selected-output convergence is much cheaper, but the V-PQ residual-tail estimate is not stable enough under the geometric budget check, so the rule escalates to max. This is not inherent full-context flatness and not primarily PQ ranking.
- GPU CPU-style parity check, 2026-05-19: forced the RULER wrapper through `geometric_probe_tail_switch` with `selected_value_exact_rule=selected_mass`, `selected_value_exact_mass=0.99`, `selected_value_min_exact_top=1024`, proxy mass `>=0.990`, and PQ-corr `>=0.70`, which bypasses the native geometric accept-count shortcut. On the same 32k RULER first decode step, layer 16 selected `12.0k` tokens/head at `tail_probe<=0.005` and `10.5k` at `tail_probe<=0.20`, not `32k`. This confirms the earlier near-full GPU result is a native fast-path semantic/accounting issue, not a failure of CPU-style geometric selection. The parity path is currently too slow/expensive because it recomputes confidence probes; it is a correctness diagnostic, not yet the optimized benchmark path.
- Native accounting correction, 2026-05-19: `ranked_confidence_cost_mode` now defaults to `exact` in the runner and public wrappers. Exact-cost reruns show native strict/tail-stability at 32k selected about `8.0k` / `7.9k` dynamic tokens/head, not `32k`; the old `32k` summaries were using `upper_bound` accounting that intentionally charged `geometric_max_budget`. The remaining native issue is semantic robustness, not literal full-context retrieval.
- Native proxy-gate fix, 2026-05-19: proxy confidence now accepts a per-head budget tensor. The native path no longer collapses all heads to `max(accepted_budget_counts)` when proxy gates are enabled; only heads that fail the proxy gate are escalated to `max_budget`. Validation job `50456860` completed on the 32k one-token RULER case. With proxy gates `mass>=0.990` and `corr>=0.70`, mean selected tokens were `13.6k`/head, not `32k`, with layer range `7.9k-31.7k`. This confirms the per-head fix works. It is still not benchmark-practical as the final strict rule: decode was `25.0s` for one token because exact-logit proxy calibration dominates, versus `2.15s` for no-proxy native strict.
- Native CUDA proxy-gate speed fix, 2026-05-19: added `gqa_decode_geometric_accept_counts_vpq_proxy`, which applies proxy mass/correlation gates inside the CUDA accept-count path and reuses the exact ranked logits already computed there. CUDA unit job `50459134` passed. Rerun job `50459586` on the same 32k one-token case matched the Python proxy selected budget (`13.6k` tokens/head) but reduced decode from `25.0s` to `2.50s`. Cost accounting also dropped from `17.244` to `9.314 MB/head-query` step because the separate Python exact-logit proxy pass is gone. No-proxy native strict remains cheaper at `8.0k` selected and `6.611 MB/head-query`, so the proxy gate is now fast enough to test but still stricter/more expensive.
- Implemented a new CUDA confidence primitive, `gqa_decode_geometric_accept_counts_vpq_tail_stability`, that compares `tail_budget exact + compressed residual tail` against `probe_budget exact + compressed residual tail`. This is less conservative than the earlier strict check, which compared compressed-tail output against a no-tail larger exact probe.
- `50425964`: tail-stability native microbench completed after rebuilding the CUDA extension. At 32k-shaped dimensions, runtime was `70.72 ms`; accepted counts were mostly still full context (`mean=30272`, `min=6144`, `max=32768`, `29/32` heads at full `32768`). Interpretation: the comparator is less conservative for a few heads, but not enough by itself to assume dynamic confidence will avoid near-dense reads.
- CUDA unit coverage for the new primitive passed in Slurm job `50425744`: full-scan PQ top-k, V-PQ helpers, and online page append tests all passed after extension rebuild. This includes native/reference coverage for `gqa_decode_geometric_accept_counts_vpq_tail_stability`.
- Updated `benchmark/audit_benchmark_readiness.py` for the current decode-only benchmark phase: RULER quality now falls back to `pred/summary-*.csv` when the JSON summary was written before evaluation, dense-prefill passthrough is no longer treated as a readiness failure when `approx_prefill=false`, and GPU/native prefill selector names are accepted. Current rel-sweep audit is saved at `notes/geometric_rel_sweep_audit_20260518.md`.
- Submitted tail-stability RULER 32k sweep reusing the existing NIAH data:
  - `50425803`: `tail_probe_rel_l2_max=0.02`; failed before model load because the RULER wrapper parser did not yet accept `geometric_tail_stability_switch`.
  - `50425804`: `tail_probe_rel_l2_max=0.04`; failed with the same stale parser issue.
  - `50425805`: `tail_probe_rel_l2_max=0.08`; failed with the same stale parser issue.
- Parser fix: added `geometric_tail_stability_switch` to RULER, LongBench-v2, and public long-decode benchmark argument choices. Resubmitted rel `0.02` as `50433012`, rel `0.04` as `50433030`, and rel `0.08` as `50433184` with manifest `notes/slurm_manifests/geometric_tail_stability_rerun_20260518.tsv`.
- Tail-stability rerun outcome: canceled `50433012/50433030/50433184` after `40.3/38.6/29.7 min` without a summary. They were past parser/model load and inside the 32k sample, so this is a runtime-negative result. Tail-stability is not benchmark-practical in the current implementation; strict geometric is already near-dense, and tail-stability is slower before producing a usable cost/quality point.

## Decode Runtime Push, 2026-05-17 Night

Objective: keep the decode-only frontier algorithm fixed and reduce GPU runtime overhead without changing selector/compression semantics.

Latest update, 2026-05-18:

- Added active-fraction accounting fields so short-context runs cannot be mistaken for selector/tail benchmarks:
  - `selector_active_fraction`
  - `tail_active_fraction`
  - `confidence_active_fraction`
- Added `STAGE_MODEL_TO_TMP=1` support to the LongBench-v2 wrapper. This is off by default, but future repeated Qwen diagnostics can stage a local model copy on the Slurm node to reduce GPFS shard-load overhead.
- Fixed a real early-decode speed bug: when no dynamic PQ page is sealed yet, the frontier output is dense-equivalent even with `vpq_value + tail`; the patch now bypasses to the original dense attention instead of entering the slow per-head fallback.
- AIME24 n=1, Qwen3-8B, 512 forced tokens, profiled:
  - before bypass: `131.41s`, `patched_attention_seconds_total=111.26s`, selector/tail/update all zero because no 512-token dynamic page ever sealed.
  - after bypass: `27.32s`, `approx_attention_calls_total=0`, `passthrough_attention_calls_total=18432`, `selector_active_fraction=0`, `tail_active_fraction=0`.
  - interpretation: this was a static-window artifact, not a selector-runtime profile.
- LongGenBench SGT-short n=1, Qwen3-8B, 1024 forced tokens, normal accounting after bypass:
  - generation `61.88s`
  - step/total `0.3670 / 0.3672 MB/head-query`
  - selector/exactKV/tail `0.0799 / 0.2769 / 0.0102 MB/head-query`
  - selected tokens `567.1`
  - update `220.92 MB` cumulative, `0.000187 MB/head-query`
  - selector/tail active fraction `0.870`
  - approx/passthrough attention calls `32040 / 4824`; the bypass covers the first `134` decode steps per layer before the first dynamic page seals.
- `DISABLE_COST_STATS=1` is not a useful speed lever for this path: LongGenBench SGT-short 1024 took `62.77s`, slightly slower than normal accounting, and its cost fields are intentionally incomplete because native decode returns before cost aggregation.
- Hot-path pack/cache cleanup after the bypass:
  - Added fast GQA K-PQ/V-PQ pack caches keyed by sealed page state and moved V-PQ pack cache checks ahead of per-page validation loops.
  - Added a decode-only sidecar cache for the strict native fullscan path, so unchanged sealed-page intervals do not loop over all KV heads every token.
  - LongGenBench SGT-short n=1, Qwen3-8B, 1024 forced tokens, normal accounting:
    - post-bypass baseline: `61.88s`
    - pack-fast: `56.43s`
    - pack-fast + sidecar-fast: `54.51s`
    - modeled cost unchanged: step/total `0.3670 / 0.3672 MB/head-query`, selected tokens `567.1`, selector/tail active fraction `0.870`.
  - Profiled 1024-token run:
    - pack-fast: generation `73.19s`, `patched_attention_seconds_total=40.59s`, `native_pack_seconds_total=1.76s`, `index_sidecar_seconds_total=3.16s`.
    - pack-fast + sidecar-fast: generation `70.82s`, `patched_attention_seconds_total=38.43s`, `native_pack_seconds_total=1.75s`, `index_sidecar_seconds_total=1.15s`.
  - Longer 4096-token scaling check, Slurm `50389683`:
    - generation `222.12s`
    - modeled cost unchanged from the prior 4096 run: step/total `0.5883 / 0.5887 MB/head-query`, selected tokens `573.4`.
    - prior post-hotpath 4096 run was `347.55s`; prior dense 4096 reference was `174.57s`.
    - interpretation: static bypass plus pack/sidecar cache cleanup moves this smoke from roughly `2x` dense runtime to roughly `1.27x` dense runtime without changing modeled algorithmic cost.
  - 8192-token dense/frontier scaling pair queued:
    - manifest `notes/slurm_manifests/ldecode8192_dense_vs_frontier_20260518_004907.tsv`
    - dense `50390139`: generation `325.43s`, completion `0.538`.
    - frontier `50390140`: generation `433.84s`, completion `0.673`, step/total `0.8799 / 0.8805 MB/head-query`, selected tokens `574.4`, selector/tail active fraction `0.984`.
    - interpretation: the path is practical for long-decode sweeps, but runtime is still `1.33x` dense at this prompt/context. The next speed target is native tail/score workspace, not selector top-k or wrapper pack caching.
  - Follow-up speed diagnostics queued:
    - C++ fused-call retest at 4096, Slurm `50390971`: generation `230.09s` vs unfused `222.12s`, identical modeled MB. Keep fused-call disabled by default.
    - 8192-token profiled frontier run, Slurm `50390973`: generation `715.77s` with profiling overhead. Timing slices: `patched_attention_seconds_total=449.74s`, `qkv_cache_seconds_total=179.30s`, `native_selector_seconds_total=50.29s`, `native_attention_seconds_total=83.80s`, `native_pack_seconds_total=19.45s`, `index_sidecar_seconds_total=11.09s`, `output_projection_seconds_total=33.68s`.
    - interpretation: after wrapper caching, the main algorithm-specific kernel cost is selector fullscan plus V-PQ tail attention. QKV/cache/projection are dense-model baseline costs, and profiling sync inflates all slices.
  - Added experimental kernel knob `SELECTOR_PQ_DECODE_TAIL_PAGES_PER_BLOCK` and validated CUDA unit tests in Slurm `50391188`.
    - 4096-token sweep: ppb1 `222.12s`, ppb2 `227.48s`, ppb4 `219.33s`, ppb8 `220.44s`, all with identical modeled MB.
    - conclusion: ppb4/8 are only about `1%` faster and change reduction order/output slightly; keep default ppb1 for trusted benchmark runs.

Current runtime read: after the static-window fix and cache cleanup, the next bottleneck is not the existing native selector kernel. In the latest profiled 1024-token smoke, `native_selector_seconds_total=2.26s`, `native_attention_seconds_total=8.21s`, `qkv_cache_seconds_total=15.04s`, `output_projection_seconds_total=3.50s`, and `patched_attention_seconds_total=38.43s`. Remaining work is mostly QKV/cache/projection plus native tail attention and residual Python orchestration.

Implemented:

- Added native decode fused-call entry points:
  - `gqa_decode_fullscan_vpq_selected_tail_agg`
  - `gqa_decode_fullscan_vpq_selected_tail_agg_mass_min`
  - `gqa_decode_scoreless_fullscan_vpq_tail`
- Added CUDA unit coverage for the fused-call and scoreless decode wrappers.
- Made `selector_paged_pq.__init__` tolerate optional/missing extension symbols so stale local `_C` binaries do not break import before rebuild.
- Added benchmark flags:
  - `--enable_native_decode_fused` / `--disable_native_decode_fused`
  - `--native_decode_scoreless_fused`
  - `--native_decode_scoreless_force_mode`
- Important default: native decode fused-call is now disabled by default. It is opt-in only because the measured runtime was worse.

Validation:

| run | job | result | interpretation |
| --- | ---: | --- | --- |
| CUDA unit, fused decode | `50384520` | passed, `139s` | Fused-call selector+tail wrappers match separate selector+attention. |
| CUDA unit, scoreless decode | `50387162` | passed, `73s` | Scoreless wrapper matches fused top-k plus scoreless tail reference. |

1024-token public-longdecode smoke, Qwen3-8B, AIME24 n=1, dense prefill + approximate decode:

| variant | job | generation sec | step MB/head-query | total MB/head-query | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| old unfused native path | `50387189` | `130.87` | `0.2718` | `0.2719` | Best of this A/B/C; keep as default path. |
| scoreless fused top-k + tail | `50387190` | `146.54` | `0.2970` | `0.2971` | Slower and higher modeled MB because it rescans K-PQ codes for tail scoring. |
| C++ fused-call wrapper | `50387188` | `175.65` | `0.2718` | `0.2719` | Slower despite same modeled MB; likely worse allocator/lifetime/dispatch behavior around internal tensor creation. Do not use by default. |

Default guard check: `50387265` completed a 256-token public-longdecode smoke with no fused env override; summary confirms `disable_native_decode_fused=true`, scoreless disabled, generation `47.28s`, and step/total `0.1450 MB/head-query`.

Conclusion: wrapping existing kernels in a single C++ call is not enough and can regress. The next real speed target must be a true kernel-level redesign that avoids Python dispatch, dense score materialization, and redundant tail work together; otherwise keep the old unfused native path for benchmark execution.

## Decode-Only Prefill Correction, 2026-05-17 Evening

Decision: do not apply K/V selection or compression during batched prefill. Dense prefill has high K/V reuse and efficient dense kernels; per-query sparse masks create irregular K/V holes and destroy matmul reuse. Frontier approximation is now decode-only.

Implementation changes:

- `pagedpq` benchmark mode now defaults to dense prefill plus approximate decode. Approximate prefill is opt-in only via `--approx_prefill` / `APPROX_PREFILL=1`.
- Dense prefill now builds the decode sidecars after each layer's dense prefill update: K-PQ page selector sidecars, V-PQ sidecars when enabled, static prefix/recent suffix metadata, and online append state.
- First decode no longer needs to lazily build the initial prefill index unless `--skip_prefill_index_build` is explicitly set.
- RULER default smoke mode switched from token-stream prompt feeding to batched prefill (`pagedpq_batched`), so prompt processing is dense by default.

Validation:

| run | job | result | key cost/timing | interpretation |
| --- | ---: | --- | --- | --- |
| CUDA extension unit/build | `50362841` | passed | `434s` Slurm elapsed | Native selector/V-PQ extension still builds and unit tests pass after the decode-only wrapper changes. |
| LongBench-v2 short/easy n=1, Qwen3-8B | `50368758` | accuracy `100%`, `approx_prefill=false` | generation `4.33s`, step `0.503 MB/head-query`, total `0.519 MB/head-query`, update `275.48 MB` cumulative | One dense prefill passthrough per layer (`36`), then decode approximation (`540` head-query calls). Sidecar warmup works. |
| RULER niah_single_1 ctx2k n=1, Llama-3.1-8B | `50368791` | score `100%`, `approx_prefill=false` | prefill `2.82s`, decode `1.72s`, step `0.424 MB/head-query`, total `0.438 MB/head-query` | Batched dense prefill + decode-only frontier path works through the RULER wrapper. |
| LongBench-v2 dense-reference diag n=1, Qwen3-8B | `50370776` | dense/frontier accuracy `100%/100%`, exact text match `100%` | mean/max logit relL2 `0.0136/0.0266`, mean/max hidden relL2 `0.0169/0.0268`, KL mean/max `9.4e-6/1.5e-4`, total `0.511 MB/head-query` | Decode-only frontier has low logit/hidden drift on this short/easy diagnostic row. |

Public long-decode smoke coverage:

| benchmark | dense job/result | frontier job/result | frontier cost | interpretation |
| --- | --- | --- | ---: | --- |
| AIME24 n=1, Qwen3-8B, 128 generated tokens | `50369408`, accuracy `0%`, `8.89s` | `50369404`, accuracy `0%`, `31.82s` | `0.114 MB/head-query` | Dense also fails; output shares first `257` chars then diverges. Useful as wrapper/runtime smoke, not quality evidence. |
| LiveCodeBench codegen n=1, Qwen3-8B, 128 generated tokens | `50369409`, `9.92s` | `50369405`, `32.62s` | `0.217 MB/head-query` | Frontier response exactly matches dense for this sample. Code execution was not enabled in this smoke. |
| LongGenBench SGT-short n=1, Qwen3-8B, 128 generated tokens | `50369410`, completion `0.019`, substring once `0`, `5.91s` | `50369406`, completion `0.019`, substring once `0`, `35.69s` | `0.341 MB/head-query` | Metrics match dense but both are uninformative at only 128 generated tokens; output shares first `299` chars then diverges. |

Notes:

- Initial public frontier jobs `50369135-50369137` completed generation but failed while writing `summary.json` because `public_longdecode_eval.py` lacked the `allow_tf32_selector` argument consumed by `pagedpq_config`; fixed and retried as `50369404-50369406`.
- At 4k context and 128 decode tokens, modeled MB is low but wall-clock is slower than dense because selector/compressed-tail kernels dominate. These are smoke tests; speed claims need longer decode/context where dense decode bandwidth dominates.

Longer decode smoke:

| benchmark | dense | frontier | frontier cost/timing | interpretation |
| --- | ---: | ---: | ---: | --- |
| LongGenBench SGT-short n=1, Qwen3-8B, 1024 forced tokens | `50369489`, `43.85s`, completion `0.038`, substring once/range/periodic `0` | `50369490`, `135.25s`, completion `0.019`, substring once/range/periodic `0` | `0.367 MB/head-query`, selected `567`, update `0.00019 MB/head-query`, native selector `2.73s`, native attention `23.89s`, patched attention `96.95s` | Decode-only path works for 1024 tokens, but runtime is still ~3.1x slower than dense at 4k context. Metrics are not useful yet because dense also fails the LongGenBench smoke badly. |
| LongGenBench SGT-short n=1, Qwen3-8B, 4096 forced tokens | `50369998`, `174.57s`, completion `0.154`, substring once/range/periodic `0` | `50369999`, `456.11s`, completion `0.115`, substring once/range/periodic `0` | `0.589 MB/head-query`, selected `573`, update `0.00038 MB/head-query`, native selector `16.62s`, native attention `98.36s`, patched attention `323.62s` | Longer decode confirms algorithmic MB is low but current GPU implementation is still ~2.6x slower than dense at this prompt/context scale. Quality metrics remain weak for both dense and frontier; frontier diverges after `299` chars. |

## GPU Benchmark Execution, 2026-05-17

Current practical benchmark preset:

```text
paged-PQ fullscan selector
online rule: pq_ranked_mass_budget
budget cap: 192 for robust 16k RULER, 128 is acceptable on current LongBench-v2 short/easy
selected V: V-PQ with selector-rank exact top 256
tail: V-PQ tail estimator
index build: torch_gpu
prefill selector: native CUDA LUT score path
prefill tail score reuse: enabled by default in benchmark wrappers
```

Important correction: the strict CPU-style geometric rule is functional on GPU but not benchmark-fast. Current ctx2k strict smoke with selected-mass/min-exact V-PQ tail passes quality (`100%`) but takes `~284-288s` for one RULER sample (`~90-93s` prefill, `~193-195s` decode). CUDA reuse of precomputed ranked logits and probe-only tail-skip passes unit tests, but runtime does not improve materially because the dominant cost is repeated tail-producing confidence attention. The fast ranked-mass preset remains the benchmark path.

Latest downstream results:

| benchmark | setting | dense/reference | frontier | frontier cost | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| RULER 16k, n=4, four tasks | budget 192, proxy mass 0.97 | `100 / 75 / 100 / 91.67` | `100 / 75 / 100 / 91.67` | `4.843-5.220 MB/head-query` | Matches dense on the selected task set; budget 192 fixes the FWE margin failure seen at budget 128. |
| RULER 16k FWE only | budget sweep 192/256/384 | `91.67` | all `91.67` | `4.843 / 4.873 / 4.902 MB/head-query` | FWE failure at budget 128 is a selector-margin issue, not a task/setup issue. |
| RULER 64k, n=1, four tasks | native-LUT, page 2048, chunk 512, budget 192 | `100 / 0 / 100 / 66.67` | `100 / 0 / 100 / 66.67` | `2.227-2.359 MB/head-query` | Matches dense on all tested 64k rows; `niah_multikey_2` is uninformative because dense also fails. Runtime is still much slower than dense. |
| RULER 64k single-needle smoke | native-LUT vs old torch-matmul path | `100` | `100` | `2.351 MB/head-query` | Native-LUT reduces modeled step cost from `17.543` to `2.351 MB/head-query` and runtime from roughly `19.1-20.0 min` to `14.4 min` for one sample. |
| LongBench-v2 short/easy n=16, Qwen3-8B | native-LUT, max input 8192, budget 192 | `9/16 = 56.25%` | `8/16 = 50.00%` | `0.870 MB/head-query` | One dense-correct row flips from `D` to `A`; diagnostic shows a fragile first answer-letter margin (`D-A = +0.25` dense, `A-D = +0.25` frontier) despite high cosine (`mean hidden/logit cosine > 0.999`). |
| LongBench-v2 short/easy n=59, Qwen3-8B | budget 128, proxy mass 0.99, selected-mass min exact 1024 | `29/59 = 49.15%` | `29/59 = 49.15%` | `2.592 MB/head-query` | Predictions match dense on `59/59`; responses differ on `6/59` but judge labels match. |
| LongBench-v2 short/easy n=59, Qwen3-8B | budget 192, proxy mass 0.97 | `29/59 = 49.15%` | `29/59 = 49.15%` | `2.621 MB/head-query` | Same judge labels as dense and budget-128; responses differ on `5/59`. |
| RULER 32k, n=2, four tasks | budget 192, proxy mass 0.97, chunk 512 | `100 / 50 / 90 / 100` | `100 / 50 / 100 / 100` | `9.679-10.427 MB/head-query` | Matches or exceeds dense on the selected four-task 32k smoke. Initial chunk-2048 attempt OOMed in prefill selector logsumexp; chunk 512 fixed it. |
| RULER 16k, n=4, budget 128 proxy mass 0.99 | selected-mass min exact 1024 | `100 / 75 / 100 / 91.67` | `100 / 75 / 100 / 83.33` | `4.814-5.190 MB/head-query` | Too tight/low-margin for FWE; do not use as robust RULER default. |

Implementation updates in this checkpoint:

- Added CUDA selected-value mass+min native path for V-PQ selected/tail attention and fixed min-exact semantics so static/base tokens are always exact but do not consume ranked dynamic `min_exact_top`.
- Added decode probe-only tail-skip and precomputed-ranked-logit CUDA entry points. Unit tests pass, but strict geometric runtime remains dominated by repeated tail-producing attention.
- Fixed frontier wrapper defaults so `GEOMETRIC_MIN_BUDGET` and `GEOMETRIC_MAX_BUDGET` default to `BUDGET`; previous wrappers silently capped `pq_ranked_mass_budget` at 64 even when `BUDGET=128`.
- 32k prefill needs smaller chunks on A40. `PREFILL_CHUNK_SIZE=2048` OOMed while materializing/logsumexping dense selector scores; `PREFILL_CHUNK_SIZE=512` completed.
- Added a native CUDA LUT score kernel for `gqa_causal_fullscan_pq_topk_scores`. It computes page-local PQ centroid LUTs once per query/head/page instead of recomputing `q dot centroid` per token.
- Native-LUT selector validation: CUDA unit job `50334161` passed. 32k profiled NIAH improved from old native `509.22s` to native-LUT `324.15s`, beating torch-matmul `379.08s` while preserving much lower selector cost (`0.747` vs `8.034 MB/head-query`).
- Added and recorded explicit TF32 selector flag. TF32 helps torch-matmul modestly at 16k (`112.35s -> 105.06s`) but native-LUT is equally fast and lower-cost (`105.17s`, `1.034 MB/head-query`).
- Fixed confidence fast-path guards so `selected_value_mode=exact` can use the native selected-exact + V-PQ-tail path instead of falling through to the unsupported per-head ranked-mass fallback.
- Latest CUDA unit rebuilds passed: `50330167`, `50330517`, `50333985`, and `50334161`.
- LongBench-v2 short/easy n=16 native-LUT check completed (`50334667`, `50334668`). Dense accuracy was `56.25%`; frontier was `50.00%`. The single changed row is under targeted rerun diagnostics with higher selector budgets and selected-V exactness variants: manifest `notes/slurm_manifests/lbv2_flip_diag_20260517_102314.tsv`.

LongBench changed-row diagnostic, `_id=66f3918f821e116aacb2d8b7`:

| variant | prediction | step MB/head-query | mean selected | mean hidden relL2 | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| budget 192, exact-V top 256, original n16 setting | `A` wrong | `0.877` | `671.1` | `0.0412` | Baseline flip; dense first answer-letter margin is only `0.25` logit. |
| budget 192, exact-V top 512 | `D` correct | `0.877` | `671.1` | `0.0401` | Recovers free-run answer without changing K budget; selected-V compression is implicated. |
| budget 384, exact-V top 256 | `D` correct | `0.934` | `845.4` | `0.0348` | More selected K can also recover the row at modest extra cost. |
| budget 512, exact-V top 256 | `A` wrong | `0.962` | `961.6` | `0.0330` | More K alone is not monotonic under compressed selected V/tail weighting. |
| budget 384, exact-V top 512 | `D` correct | `0.962` | `845.4` | `0.0291` | Stronger V exactness improves drift and answer stability. |
| budget 512, exact-V top 512 | `D` correct | `1.018` | `961.6` | `0.0242` | Best of this diagnostic set; still a fragile-margin row, not a broad collapse. |
| budget 192, `selected_value_mode=exact`, patched native route | `D` correct | `0.877` | `671.1` | `0.0420` | Validates exact-selected mode now stays on the native fast path; runtime `3:02` Slurm elapsed. |

Latest GPU implementation update:

- Added chunked prefill score reuse for the fast frontier path in `benchmark/selector_eval/runners/run_hf_paged_pq_intervention_eval.py`.
- Before: prefill used `torch_matmul` for top-k but the native V-PQ tail kernel rescored PQ codes again, avoiding OOM but making long prefill slow.
- Now: each query chunk materializes dense PQ selector scores, immediately reuses them in `gqa_causal_vpq_selected_tail_from_scores`, then frees the chunk. This avoids full `[query, head, token]` score OOM while removing redundant tail rescoring.
- Smoke: `50328581`, ctx2k RULER n=1, completed, score `100`, prefill `7.86s`, decode `9.18s`, step `0.548 MB/head-query`.
- 16k probe: `50328842`/`50328843`, ctx16k n=1, completed, scores `100/100`, prefill `~73s`, decode `4-10s`, step `4.94-5.04 MB/head-query`. Previous no-reuse 16k frontier prefill was `~260-295s/sample`, so this is a practical `~3.5-4x` prefill speedup at the same algorithmic preset.
- Wrapper defaults updated: frontier RULER, LongBench-v2, and public-longdecode wrappers now enable `PREFILL_TAIL_SCORE_REUSE=1` unless explicitly overridden.

Implementation fixes in this checkpoint:

- `benchmark/selector_eval/runners/run_hf_paged_pq_intervention_eval.py`: dense-equivalent bypass now only applies when selected values are exact and tail blend is zero; it no longer hides compressed selected-V/tail paths.
- `benchmark/longbench_v2_hf_eval.py` and `benchmark/public_longdecode_eval.py`: summaries now include `cost_proxy_aggregate` in addition to per-layer cost.
- `benchmark/public_longdecode_eval.py`: LiveCodeBench evaluation works on cluster Python 3.10.4 via a compatibility shim for `sys.set_int_max_str_digits`.
- `scripts/submit_frontier_benchmark_matrix.sh`: supports `RUN_RULER=0` for LongBench-only submissions.
- `scripts/submit_public_longdecode_matrix.sh`: passes LiveCodeBench code-eval controls so coding runs can produce pass@1.

Completed so far:

| benchmark | run | dense | frontier | frontier cost | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| RULER 8k, n=4, four tasks | `frontier_benchmark_matrix_frontier_patch_budget64_20260517_ruler8k` | `100/100/100/75` | `100/100/100/83.33` | `2.38-2.43 MB/head-query` | Fast preset preserves small RULER task scores; no passthrough. |
| RULER 16k no-reuse frontier | `frontier_gpu_bench_20260517_015903_ruler16k` | `100/75/100/91.67` | `100/75/100/83.33` | `4.67-5.04 MB/head-query` | Quality mostly holds, but old prefill was too slow; kept as pre-optimization reference. |
| RULER 16k score-reuse probes | `chunked_tail_reuse_16k_20260517` | n/a | `100/100` on n=1 probes | `4.94-5.04 MB/head-query` | Same preset, much faster prefill; full optimized RULER matrix is running. |
| LongBench-v2 short/easy n=59, Qwen3-8B dense | `50327161` | `49.15%` | running optimized | n/a | Old no-reuse frontier was cancelled after optimized score-reuse row started. |
| AIME24 n=2, Qwen3-8B | `frontier_gpu_bench_20260517_015903_public` | `0%` | `0%` | `0.335 MB/head-query` | Not useful as a quality comparator because dense also fails. |
| GPQA no-thinking n=4, Qwen3-8B | `frontier_gpu_bench_20260517_015903_public_gpqa_nothink` | `25%` | `50%` | `0.266 MB/head-query` | Tiny sample, but no evidence of task collapse. |
| LiveCodeBench codegen n=2 | `frontier_gpu_bench_20260517_015903_public_codeeval_retry` | `0%` | `0%` | `0.355 MB/head-query` | Not useful as a quality comparator because dense also fails. |
| LongGenBench SGT short n=2 dense | `frontier_gpu_bench_20260517_015903_public` | substring metrics `0` | frontier running | n/a | Current LongGenBench setup may be a poor comparator because dense already scores zero. |
| CUDA unit tests | `50327521` | n/a | passed | n/a | Selector top-k, V-PQ helper, and online append tests pass after current patches. |

Queued/running benchmark manifests:

- Optimized RULER 16k frontier: `notes/slurm_manifests/frontier_optimized_ruler16k_score_reuse_20260517.tsv`
- Optimized LongBench-v2 n64 frontier: `notes/slurm_manifests/frontier_optimized_lbv2_score_reuse_20260517.tsv`
- RULER 16k paired matrix, pre-optimization: `notes/slurm_manifests/frontier_benchmark_matrix_frontier_gpu_bench_20260517_015903_ruler16k.tsv`
- LongBench-v2 n64 clean pair, pre-optimization: `notes/slurm_manifests/frontier_gpu_bench_20260517_015903_lbv2_n64_clean.tsv`
- Public long decode: `notes/slurm_manifests/public_longdecode_frontier_gpu_bench_20260517_015903_public.tsv`
- LiveCodeBench code-eval retry: `notes/slurm_manifests/public_longdecode_frontier_gpu_bench_20260517_015903_public_codeeval_retry.tsv`
- GPQA disable-thinking pair: `notes/slurm_manifests/public_longdecode_frontier_gpu_bench_20260517_015903_public_gpqa_nothink.tsv`

GPU parity update, 2026-05-16:

- Algorithm-level parity is now the required gate, not just final selected-output parity:
  - GPU NumPy page-PQ builder now uses CPU-compatible page seeds (`seed + 7919 + page_start`) so CPU `PagedLocalPQIndex` and GPU state can be compared directly.
  - current canonical-state passing run: `50325448`, output `cuda_unit_result/gpu_cpu_paged_pq_vpqtail_long_canonical_20260517_004407`
  - coverage: real Q/K/V trace, decode `4000/8000/16000`, heads `0/8/16`, budget `1024`, page size `2048`, selected-V compression and V-PQ tail path enabled.
  - checked intermediates: page count/start/size, `pending_start`/`indexed_end`, pending tokens, static+pending base tokens, per-page PQ codebooks, per-page PQ codes, full ranked candidates, selected set, selector MB accounting, selected attention output, and native V-PQ selected/tail aggregation.
  - result: all page/base/pending/code checks pass exactly, max codebook diff `0.0`, min top-k overlap `1.0`, max score diff `3.05e-05`, max selector MB CPU/GPU diff `0.0`, max selected-output relL2 `1.28e-06`, max V-PQ tail-output relL2 `4.21e-06`.
  - the gate now canonicalizes the CPU-built page-PQ state onto GPU before testing CUDA ranking/attention. This avoids treating repeated NumPy k-means near-tie nondeterminism as a CUDA mismatch.
  - follow-up CUDA unit run after the seeded `torch_gpu` builder change also passed: `50325449`, output `cuda_unit_result/frontier_cuda_unit_after_torch_seed_20260517_004414`.
- Added a trace-level CPU/GPU paged-PQ selector parity gate:
  - script: `benchmark/selector_eval/gpu/run_gpu_cpu_parity_eval.py`
  - Slurm wrapper: `scripts/run_gpu_cpu_paged_pq_parity.sh`
  - current passing run: `50325448`, output `cuda_unit_result/gpu_cpu_paged_pq_vpqtail_long_canonical_20260517_004407`
  - coverage: real Q/K/V trace, decode `4000/8000/16000`, heads `0/8/16`, budget `1024`, page size `2048`
  - timing signal on the current longer gate: native CUDA selector is `~11.25x` faster than the Torch page-loop reference and `~236x` faster than the CPU verifier for the tested selector slice.
  - scope: this validates the fullscan selector/page-policy/exact-selected-output boundary plus the selected-V/V-PQ-tail native path. The online confidence policy still needs separate end-to-end benchmark validation.
- The fast `torch_gpu` page-PQ builder now uses the same per-page/per-subvector seeded random initialization schedule as CPU `build_pq_index`, instead of first-row initialization. This keeps benchmark runs algorithmically aligned with the CPU selector family while still building the page-PQ state on device.
- The stricter CPU-trace-style geometric confidence path is now implemented on GPU for the native V-PQ tail path, including proxy-mass and PQ-correlation gates. It no longer silently falls back when `tail_proxy_mass_min`, `tail_pq_corr_min`, or `tail_pq_relrmse_max` are enabled.
- Dense prefill PQ-score reuse is disabled for long contexts; materializing `[queries, heads, indexed_tokens]` scores OOMs at 16k+ (`50322925`, `50322927`, `50322929`). The long-prefill route now uses tiled `torch_matmul` selector scoring plus native codebook-scanning tail attention, avoiding the dense score matrix.
- Cost accounting now separates the additional exact-K reads used by confidence calibration and divides prefill-call calibration traffic by head-query count. The earlier `14143 MB/head-query` smoke artifact was an accounting bug, not a real cost.
- Functional smoke results:
  - ctx2k strict-geometric no-dense run `50323046`: completed, `2.751 MB/head-query`, prefill `75.8s`; selector time `0.78s`, native tail/confidence attention `72.4s`.
  - ctx4k strict-geometric tiled run `50323090`: completed, `3.289 MB/head-query`, prefill `193.3s`; selector time `2.63s`, native tail/confidence attention `183.3s`.
  - ctx16k strict-geometric tiled run `50323086`: completed, `7.501 MB/head-query`, prefill `2540.9s`; selector time `25.8s`, native tail/confidence attention `2480.5s`.
- Interpretation: strict CPU-style geometric confidence is now functional and memory-safe on GPU, but not benchmark-fast. The blocker is not selector top-k; it is repeated irregular compressed-tail/confidence attention. The fast benchmark preset remains the practical route for broader RULER/LongBench runs until that tail/confidence kernel is redesigned.

Benchmark-readiness result, 2026-05-16:

- Strict gate `scripts/check_frontier_benchmark_readiness.sh` passes.
- CUDA unit job `50321548` passed selector top-k, GPU V-PQ helper, and online-page append tests through Slurm on `spgpu`.
- Wrapper smokes all pass: frontier RULER, frontier LongBench-v2, dense RULER, and dense LongBench-v2.
- Matrix jobs `50321549`-`50321558` completed on `spgpu`; all rows in `notes/frontier_benchmark_matrix_afterok_20260516_audit.md` audit `ok`.
- RULER ctx8k, n=4 per task: frontier matches dense on all tested tasks:
  - `niah_single_1`: dense `100`, frontier `100`, frontier `37.94 s/sample`, `2.452 MB/head-query`.
  - `niah_multikey_2`: dense `100`, frontier `100`, frontier `40.58 s/sample`, `2.434 MB/head-query`.
  - `vt`: dense `100`, frontier `100`, frontier `32.08 s/sample`, `2.479 MB/head-query`.
  - `fwe`: dense `75`, frontier `75`, frontier `33.20 s/sample`, `2.474 MB/head-query`.
- LongBench-v2 short/easy, max input 8192: dense `21/59 = 35.59%`, frontier `24/59 = 40.68%`; frontier runtime `36.70 s/example`, `2.555 MB/head-query`.
- LongBench row-level drift: predictions agree on `56/59`; all three changed rows are `gained_correct`. Changed-row diagnostics report mean logit relL2 `0.0670`, `0.0883`, `0.0776`; mean hidden relL2 `0.0726`, `0.1007`, `0.0884`; no `n/a` diagnostics remain.

Latest backend-readiness update, 2026-05-16:

- Fixed a native fast-path gate: `pq_proxy_mass_budget` / `pq_ranked_mass_budget` no longer get rejected just because `tail_probe_rel_l2_max` is finite. That threshold is probe-only and should not disable score-based confidence.
- Fixed the RULER smoke wrapper default for native selected-V compression: `selected_value_min_exact_top` and `selected_value_max_exact_top` now default to `0`, matching the native selector-rank exact-top path instead of forcing a per-head fallback.
- Fixed the RULER benchmark entrypoint default for native selected-V compression as well: direct `call_pagedpq_streaming.py` runs now default `selected_value_min_exact_top=0` and `selected_value_max_exact_top=0`.
- Added explicit frontier launch presets:
  - `scripts/run_frontier_ruler_batched_one.sh`
  - `scripts/run_frontier_longbench_v2_one.sh`
  These encode the current full frontier settings so benchmark runs do not accidentally use streaming prefill, exact-V-only mode, profiling-on mode, dense mode, or non-native selected-V fallbacks.
- Added matching dense/reference launch presets:
  - `scripts/run_dense_ruler_batched_one.sh`
  - `scripts/run_dense_longbench_v2_one.sh`
  These use valid dense modes (`dense_batched` for RULER and `ATTENTION_MODE=dense` for LongBench-v2), include `spgpu`/`zhengya98` Slurm headers, and prevent the earlier invalid `MODE=dense` failure.
- Added `benchmark/audit_benchmark_readiness.py`, a small artifact-level audit tool that normalizes RULER and LongBench-v2 summaries into one table and flags missing cost, passthrough fallback, streaming-prefill mode, missing config, no confidence rule, and non-compressed selected-V runs. It accepts explicit run paths or Slurm manifests. Current audit output is saved at `notes/readiness_audit_20260516.md`.
- Tightened `benchmark/audit_benchmark_readiness.py` to also flag non-`cuda_ext` selector backends, non-`torch_gpu` index builds, non-GPU prefill selector backends, forced selected-V exact fallback windows, and dense-reference diagnostic artifacts. The stricter checker still marks the current successful RULER/LongBench frontier artifacts `ok`.
- Added `notes/slurm_manifests/ruler_ctx8192_batched_success_20260516.tsv`, a curated manifest for the successful ctx8k RULER dense/frontier artifacts. The older initial pair manifest includes failed dense paths and should not be used for the latest readiness audit.
- Added `notes/benchmark_readiness_checklist.md`, which maps the objective to concrete evidence, current gaps, pending Slurm jobs, and the completion criteria for this phase.
- Added `benchmark/audit_benchmark_wrappers.py` and `notes/wrapper_config_audit_20260516.md` to statically check benchmark wrapper defaults before submission. Current wrapper audit passes for dense/frontier RULER and LongBench wrappers, including `spgpu` / `zhengya98`, `cuda_ext`, `torch_gpu`, `vpq_value`, and the dense modes.
- Added and submitted fresh CUDA unit-test wrapper `scripts/run_frontier_cuda_unit_tests.sh`; Slurm job `50320626` will build the extension on `spgpu`, run selector top-k, V-PQ reconstruction, and online-page append tests, and write `cuda_unit_result/frontier_cuda_unit_tests_20260516/summary.json`. Manifest: `notes/slurm_manifests/frontier_cuda_unit_tests_20260516.tsv`. Audit script/output: `benchmark/audit_cuda_unit_tests.py` and `notes/cuda_unit_audit_20260516.md`.
- Added `scripts/submit_frontier_benchmark_matrix.sh` to submit paired dense/frontier RULER plus LongBench-v2 matrices through the validated wrappers, with manifest output and `DRY_RUN=1` support. Use it only after wrapper smokes pass.
- Added `SBATCH_DEPENDENCY` support to the matrix submitter and queued the actual selected matrix as jobs `50320632`-`50320641`, with `afterok` dependencies on `50320133`, `50320141`, `50320284`, `50320285`, and CUDA unit job `50320626`. Manifest: `notes/slurm_manifests/frontier_benchmark_matrix_afterok_20260516.tsv`.
- Verified the benchmark matrix submitter with `DRY_RUN=1`: it produces the expected 10-job plan for the current selected scope, covering dense/frontier RULER on four ctx8k tasks and dense/frontier LongBench-v2 short/easy n=64 at max input 8192.
- Added `notes/benchmark_runtime_projection_20260516.md`. Based on current successful artifacts, the selected validation matrix projects to about `44.0 min` for the longest frontier LongBench job, `2.9 min` for the longest frontier RULER job, and `64.2 min` if all 10 matrix jobs ran serially, excluding Slurm queue wait.
- Added `benchmark/report_longbench_drift.py` and current reports:
  - `notes/longbench_drift_report_20260516.md`: changed prediction/correctness rows from the LongBench-v2 n=59 pair, joined with available dense-reference drift diagnostics.
  - `notes/longbench_drift_diagnosed_20260516.md`: all currently diagnosed rows, including preserved-correct and preserved-wrong controls.
- Pending validation:
  - `50320083`: profile-off ctx8k RULER n=4 timing check, cancelled while pending to prioritize benchmark-wrapper validation.
  - `50320133`, `50320141`, `50320284`, `50320285`: first wrapper-smoke chain failed fast with exit `126` because the wrappers used `exec` on helper scripts without executable bits.
  - Fixed wrapper launch bug by changing the four benchmark wrappers to `exec bash ...`.
  - `50320647`: fixed frontier RULER wrapper smoke completed cleanly: score `100%`, `pagedpq_batched`, zero passthrough, step `0.548 MB/head-query`.
  - `50320648`-`50320650`: remaining fixed wrapper-smoke chain jobs are pending/dependent.
  - `50320626`: first CUDA unit-test job is invalid as readiness evidence because the wrapper masked a failing V-PQ helper test and wrote a false pass.
  - Fixed CUDA unit wrapper to fail on the first failing command and patched `test_gpu_vpq_helpers.py` to avoid the NumPy `allclose` dispatch failure.
  - `50321442`: corrected CUDA unit-test job is pending; `benchmark/audit_cuda_unit_tests.py` now rejects stale summaries whose embedded `slurm_job_id` does not match the manifest job ID.
  - `50321443`-`50321452`: resubmitted benchmark matrix gated on `afterok:50320647:50320648:50320649:50320650:50321442`.
  - `50321453` / `50321454`: resubmitted LongBench changed-row drift diagnostics for the two missing changed rows in `notes/longbench_drift_report_20260516.md`.
- Added strict completion gate `scripts/check_frontier_benchmark_readiness.sh`. It currently fails as intended because corrected CUDA unit job `50321442` has not produced a matching summary yet; stale summary `50320626` is rejected.
- Added `scripts/submit_longbench_changed_row_diagnostics.sh` so the actual matrix's changed LongBench rows can be diagnosed after predictions exist, instead of assuming the earlier n=59 changed-row set is sufficient.
- Corrected streaming-mode full-frontier RULER smoke completes but is not the benchmark path: ctx8k n=1 score `100%`, generation `650.1s`. This was slow because `pagedpq_stream` feeds the prompt through token-by-token approximation.
- Corrected batched-prefill full-frontier RULER smoke is the right path: `niah_single_1`, ctx2k, n=1, score `100%`, generation `10.3s`, step `0.554 MB/head-query`, selector/exact/tail `0.312/0.238/0.0044 MB`, zero passthrough calls.
- Corrected batched-prefill full-frontier RULER smoke at ctx8k also completes: `niah_single_1`, n=1, score `100%`, generation `36.5s`, step `2.466 MB/head-query`, selector/exact/tail `2.164/0.272/0.0299 MB`, zero passthrough calls.
- Selector microbench says current fused top-k is not the general speed solution: at 2048 positions, 512-token pages, fused helps mainly at `k=16` (`1.2-1.4x`), is parity around `k=32/64/512` in auto mode, and forced local-topk is often slower.

Interpretation: the complete frontier path is functionally alive on GPU and accounting is separated. The correct benchmark mode is `pagedpq_batched`, not `pagedpq_stream`. At ctx8k, small RULER validation batches are now feasible; broader task/context coverage is still needed before calling the benchmark path ready.

RULER ctx8k dense vs frontier-batched validation, n=4 per task:

| task | dense score | frontier score | dense s/sample | frontier s/sample | frontier step MB/head-query | selector/exact/tail MB | passthrough |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `niah_single_1` | 100.0 | 100.0 | 9.07 | 43.28 | 2.452 | 2.150 / 0.271 / 0.031 | 0 |
| `niah_multikey_2` | 100.0 | 100.0 | 8.38 | 43.70 | 2.434 | 2.132 / 0.272 / 0.031 | 0 |
| `vt` | 100.0 | 100.0 | 3.98 | 35.60 | 2.479 | 2.177 / 0.272 / 0.030 | 0 |
| `fwe` | 75.0 | 75.0 | 5.06 | 36.51 | 2.474 | 2.173 / 0.271 / 0.030 | 0 |

This is the first clean downstream evidence for the benchmark path after the native fast-path/config fixes: task scores are preserved on this small RULER set, accounting is separated, and there is no passthrough fallback. Runtime is feasible for validation batches but still slower than dense by a large factor, so GPU optimization remains necessary before broad full-suite runs.

LongBench-v2 short/easy paired validation, Llama-3.1-8B, max input 8192, temp 0:

| run | examples | accuracy | avg generation s/example | step MB/head-query | selector/exact/tail MB | passthrough |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| dense | 59 | 35.59 | 7.38 | n/a | n/a | n/a |
| frontier batched | 59 | 40.68 | 41.21 | 2.555 | 2.252 / 0.271 / 0.032 | 0 |

Row-level comparison: predictions agree on `54/59`; frontier gains four dense-wrong rows and loses one dense-correct row on this subset. This is not proof of quality improvement, but it is useful evidence that the approximation is not causing obvious aggregate degradation at this scale.

Current readiness audit, 2026-05-16:

| requirement | evidence | status |
| --- | --- | --- |
| Full GPU path exists for prefill+decode | HF all-layer `--approx_prefill` smoke `50311722` has zero passthrough and exact target text match. RULER all-layer ranked exact path also has no passthrough. | Partially ready: exact selected-K/V path works; full compressed-tail/selected-V frontier is still too slow. |
| Correctness vs dense/reference | Trace q288/layer-diversity results exist; HF all-layer prefill+decode smoke reports top-1 agreement `1.0`, logit relL2 mean/max `0.098/0.167`, hidden relL2 `0.079/0.113`. | Needs broader validation across tasks, layers, and contexts before benchmark claims. |
| GPU runtime comfortable for benchmarks | Reset-safe RULER `niah_multikey_2` ctx8192 n=4 with explicit geometric `8..512` budget takes `38.14s/sample`, close to the old `35.62s/sample`. Reset-safe LongBench fixed1024 n=8/n=16/n=32 takes `38.15/37.91/36.90s/sample`; fp16 temporary selector scores improve LongBench n=8 to `34.46s/sample` and corrected n=32 to `33.61s/sample` with the same predictions as fp32 `torch_lut`. RULER 8k fixed1024 fp16 also matches fp32 outputs `4/4` and is slightly faster (`41.07s/sample` vs `42.28s/sample`). RULER ctx16k batched fixed1024 matches dense n=2 quality but takes `99.69s/sample` with `96.71s` prefill; the selected-attention weight-cache plus accounting-vectorization path reduces the n=1 profile from `108.08s` to `95.87s`, and fp16 selector scores reduce it further to `93.11s`. | Not ready for full suites; limited 8k RULER/LongBench validation batches are feasible, but 16k RULER exposes prefill as the blocker. |
| Honest accounting | Decode-tail score-I/O cost fix added in `50310869`; summaries now separate selector, exact KV, tail estimator, index-build/update, passthrough, and runtime. LongBench now also writes `args.json` and `pagedpq_config` for future runs. | Mostly ready for current paths; continue auditing any new backend before promotion. |
| No oracle/dense selector leakage | `pq_ranked_mass_budget` uses selector scores and conservative upper-bound cost, not achieved true mass. | Ready for ranked exact path; geometric/tail rules still need fast deployable implementation. |
| Downstream benchmark readiness | Post-reset LongBench n=4 shows adaptive confidence under-retrieves (`25%`) while fixed budgets recover the dense subset score. LongBench n=8/n=16/n=32 show fixed1024 matches or exceeds dense aggregate accuracy at `1.134 MB/head-query`; n=32 still has row-level drift (`1` dense-correct lost and `1` dense-wrong gained). Four paired row diagnostics now show harmful drift has much worse logit/hidden relL2 and min-cosine than preserved rows. Ranked floor/escalation to 2048 does not improve n=8 score. Reset-safe RULER 8k hard-task smoke keeps score `100%` at `38.14s/sample`; RULER 16k batched n=2 matches dense but is too slow. | Not complete: 8k RULER smoke and LongBench validation batches are usable, but full-suite/long-context readiness still needs broader task coverage, faster prefill, and a faster/complete compressed-tail path. |

Current best trace-eval deployable selector rule plus recommended compression safety gate:

```text
recommended online confidence rule:
  geometric_probe_tail_switch
  tail_probe_rel_l2 <= 0.020 for current layer-diversity checked frontier
  calibrated proxy selected mass >= 0.990
  selected-token PQ score corr >= 0.70
  selected V exact until selected-set mass=0.99 below 90k; 0.98 with an easy-head cap above 90k
  selected V min_exact_top=1024
  selected V exact-all when selected_tokens/context_len >= 0.95

full q288 layer-16 result:
  rows = 288
  max attn-concat relL2 = 0.001580
  max layer-output relL2 = 0.000666
  mean step MB/head = 6.889
  max mean-step MB/head = 21.779

layer-diversity result for the selector rule:
  tailprobe020 layer 8 q288 max layer-output relL2 = 0.001656
  tailprobe020 layer 24 q288 max layer-output relL2 = 0.000359

exact-all selected-V gate result on early all-selected layer8/layer24 chunks:
  max layer-output relL2 ~= 1e-6
  max mean-step MB/head = 6.416
```

The exact-all selected-V gate is a deployable compression confidence rule: it uses only selected-set size, not oracle mass. It fixes the observed early-decode case where almost every token is selected and the remaining error is purely V compression.

The fixed algorithm family being improved is:

`routed paged-PQ selector + mixed exact/compressed selected V + compressed tail estimator`

Evaluate this approximation beyond attention-output relL2, while keeping unit-explicit cost accounting:

- Attention-output quality on saved real Q/K/V traces.
- Layer-output quality by replacing one layer's dense attention output and measuring post-`o_proj`, post-attention residual, and full transformer-layer output drift.
- Decode/logit quality under real-model execution: hidden-state drift, logit similarity, next-token agreement, and generated text stability.
- Task-level quality on at least one benchmark-style run before making paper-style claims.

Rules:

- Keep the algorithm fixed unless a run is explicitly labeled as an ablation.
- Do not use oracle/dense attention probabilities, dense rankings, or achieved mass inside deployable selector logic.
- Do not hide costs: report `selector_MB_per_query`, `exact_KV_MB_per_query`, `tail_estimator_MB_per_query`, online update cost when applicable, and total step cost.
- Report quality and cost together; `attention_mass` is now a diagnostic, not the primary success metric.
- Use Slurm for nontrivial runs, preferably account `zhengya98`.
- Keep this page focused on the latest evidence table and trim stale algorithm-search details as they become irrelevant.

## Downstream RULER Smoke 2026-05-14

### Native V-PQ Tail CUDA Optimization 2026-05-15

Current implementation target:

`paged-PQ K selector + exact selected-K logits + mixed exact/compressed selected V + compressed V-PQ tail`

Recent CUDA changes:

- Decode compressed-tail now uses aggregated V-PQ code-weight sums instead of scanning every tail token for every output dimension.
- Decode and prefill native tail paths consume model-dtype fp16/bf16 K/V directly; the full-cache fp32 K/V cast is removed from the hot path.
- Prefill compressed-tail now computes selected logits once, aggregates tail weights by V-PQ code in shared memory, and reconstructs output from code-weight sums.
- Parity passed in Slurm `50243864` and `50243951`, including fp16/bf16 K/V coverage for the optimized decode path and the rewritten causal prefill tail path.

Measured RULER smoke results, `niah_single_1`, one sample, 128 generated tokens, all layers patched:

| run | Slurm | context | score | stream seconds | prefill / decode seconds | native selector seconds | native attention seconds | mean MB/head-query | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pre-aggregation selected-tail native path | `50243283` | 1024 | 100.0 | 73.29 | 18.07 / 55.22 | 0.69 | 60.87 | 0.248 | Correct but decode-tail kernel was not viable. |
| Aggregated decode tail + bf16 K/V | `50243872` | 1024 | 100.0 | 33.00 | 18.17 / 14.83 | 0.77 | 21.23 | 0.248 | Decode bottleneck largely fixed; prefill still slow. |
| Aggregated prefill+decode tail + bf16 K/V | `50243961` | 1024 | 100.0 | 21.36 | 6.38 / 14.98 | 0.87 | 9.10 | 0.248 | Current 1k native-tail runtime checkpoint. |
| Aggregated prefill+decode tail + bf16 K/V | `50243977` | 4096 | 100.0 | 46.55 | 30.57 / 15.98 | 8.81 | 24.33 | 1.008 | 4k works; prefill selector and prefill tail attention dominate. |

Fresh benchmark-readiness validation jobs:

| run | Slurm | status | output | purpose |
| --- | ---: | --- | --- | --- |
| Dense batched baseline, ctx4096, n=4 | `50296001` | pending on `spgpu` priority | `ruler_eval_result/frontier_readiness_20260515/dense_batched_ctx4096_n4` | Dense task score/runtime reference. |
| Fixed-budget CUDA frontier, ctx4096, n=4 | `50296004` | pending on `spgpu` priority | `ruler_eval_result/frontier_readiness_20260515/frontier_fixed_stride8_ctx4096_n4` | Strict `cuda_ext` full GPU path with native compressed-tail/V-PQ decode; no auto fallback. |
| Dense batched baseline, ctx4096, n=4, reduced resources | `50296031` | completed | `ruler_eval_result/frontier_readiness_20260515/dense_batched_ctx4096_n4_reduced` | Score `100.0`, mean total `8.49s/sample`, prefill/decode `2.48/6.02s`. |
| Fixed-budget CUDA frontier, ctx4096, n=4, reduced resources | `50296034` | completed | `ruler_eval_result/frontier_readiness_20260515/frontier_fixed_stride8_ctx4096_n4_reduced` | Score `100.0`, mean total `38.68s/sample`, prefill/decode `22.13/16.55s`, mean total MB/head-query `0.369`. |
| Dense batched baseline, ctx8192, n=4, reduced resources | `50296166` | completed | `ruler_eval_result/frontier_readiness_20260515/dense_batched_ctx8192_n4_reduced` | Score `100.0`, mean total `9.54s/sample`, prefill/decode `1.69/7.84s`. |
| Fixed-budget CUDA frontier, ctx8192, n=4, reduced resources | `50296167` | completed | `ruler_eval_result/frontier_readiness_20260515/frontier_fixed_stride8_ctx8192_n4_reduced` | Score `100.0`, mean total `110.13s/sample`, prefill/decode `91.93/18.21s`, mean total MB/head-query `0.572`. |
| Fixed-budget exact selected-K/V runtime probe, ctx8192, n=4 | `50303565` | completed | `ruler_eval_result/frontier_readiness_20260515/frontier_exact_lut_ctx8192_ps512_b8_n4_current` | Score `100.0`, mean total `31.91s/sample`, prefill/decode `23.78/8.13s`, mean total MB/head-query `0.688`. Re-runs the older fastest `page_size=512`, `budget=8`, `torch_lut` prefill, strict `cuda_ext` decode path in current code. This is a runtime probe/ablation, not the full confidence frontier. |
| Ranked-mass exact selected-K/V, ctx1024, n=1 | `50304904` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_ranked_mass0p97_ctx1024_all_n1` | Score `100.0`, mean total `14.71s/sample`, prefill/decode `3.76/10.95s`, mean total MB/head-query `0.213`, selected tokens `413`, no passthrough. First all-layer deployable ranked-budget smoke with exact selected K/V and no tail. |
| Ranked-mass exact selected-K/V, hard ctx8192, n=4, profiled | `50305024` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_ranked_mass0p97_ctx8192_max512_n4_niah_multikey_2` | Score `100.0`, mean total `57.86s/sample`, prefill/decode `44.27/13.60s`, mean total MB/head-query `0.876`, selector/exactKV `0.438/0.438`, selected tokens `896`, no passthrough. It recovers the hard task with lower modeled MB than fixed `budget=512`, but the profiled runtime is not yet benchmark-comfortable. |
| Ranked-mass exact selected-K/V, hard ctx8192, n=4, no profile | `50305367` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_ranked_mass0p97_ctx8192_max512_n4_niah_multikey_2_noprof` | Score `100.0`, mean total `55.42s/sample`, prefill/decode `44.19/11.24s`, mean total MB/head-query `0.876`, selected tokens `896`. Profiling sync was not the main slowdown. |
| Ranked-mass exact selected-K/V, hard ctx8192, n=4, accounting sync fix | `50306279` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_ranked_mass0p97_ctx8192_max512_n4_niah_multikey_2_syncfix` | Score `100.0`, mean total `40.27s/sample`, prefill/decode `29.04/11.23s`, no passthrough. Removing per-query accepted-budget CPU sync made runtime comparable to fixed `budget=512`; cost is conservatively upper-bound charged at `0.907 MB/head-query` and `961` selected tokens. |
| Ranked-mass compressed-tail exact-V safety path, hard ctx8192, n=1, sync fix | `50306378` | failed fast-path gate | `ruler_eval_result/frontier_readiness_20260516/frontier_ranked_mass0p97_ctx8192_max512_exactv1024_n1_syncfix_niah_multikey_2` | Failed before generation because ranked-mass compressed-tail prefill requires the exact-probe gate disabled; default `tail_probe_rel_l2_max=0.020` pushed it into the forbidden slow per-head fallback. |
| Ranked-mass compressed-tail exact-V safety path, hard ctx8192, n=1, sync fix, probe gate off | `50306409` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_ranked_mass0p97_ctx8192_max512_exactv1024_n1_syncfix2_niah_multikey_2` | Score `100.0`, mean total `194.32s/sample`, prefill/decode `162.96/31.36s`, mean total MB/head-query `0.937`. Sync-safe accounting helps versus old `224s`, but full compressed-tail prefill remains far too slow. |
| Stage-split tail, hard ctx8192, n=1, prefill tail off / decode tail on | `50306727` | failed fast-path gate | `ruler_eval_result/frontier_readiness_20260516/frontier_ranked_mass0p97_ctx8192_max512_exactv1024_n1_prefill0_decode1_niah_multikey_2` | Exposed a gating bug: ranked confidence was allowed for exact prefill and full tail prefill, but not selected-VPQ prefill with tail off. |
| Stage-split tail, hard ctx8192, n=1, prefill tail off / decode tail on, selected-VPQ gate fixed | `50306923` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_ranked_mass0p97_ctx8192_max512_exactv1024_n1_prefill0_decode1_v2_niah_multikey_2` | Score `100.0`, but mean total `502.36s/sample`, prefill/decode `470.98/31.38s`. Rejected: selected-VPQ prefill with exact-value safety is much slower than full tail and exact selected. |
| Ranked-budget exact selected-K/V RULER batch, ctx8192, n=4 | `50308266` / `50308267` / `50308268` | completed | `frontier_exact_ranked_mass0p97_ctx8192_max512_n4_{niah_single_1,vt,fwe}_syncfix` | Completed the limited downstream comparison for the current benchmark-practical deployable tier, alongside the hard `niah_multikey_2` run. |
| Stage-split tail exact-all bypass, hard ctx8192, n=1 | `50309863` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_ranked_mass0p97_ctx8192_max512_exactv1024_n1_prefill0_decode1_exactallbypass_niah_multikey_2` | Score `100.0`, mean total `67.79s/sample`, prefill/decode `36.36/31.43s`, mean total MB/head-query `0.909`. The bypass skips unnecessary V-PQ sidecar construction when prefill tail is off and `exact_value_top >= budget`, reducing the stage-split path from `289.58s` to `67.79s/sample`. |
| Stage-split tail exact-all bypass, hard ctx8192, n=4 | `50309872` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_ranked_mass0p97_ctx8192_max512_exactv1024_n4_prefill0_decode1_exactallbypass_niah_multikey_2` | Score `100.0`, mean total `66.77s/sample`, prefill/decode `35.16/31.60s`, mean total MB/head-query `0.908`. The n=4 runtime matches n=1, so the bypass is stable. It is still `1.66x` slower than ranked exact selected-K/V because decode tail adds about `20s/sample` while adding only `0.001 MB/head-query` in modeled traffic. |
| Profiled exact selected-K/V vs stage-split decode tail, hard ctx8192, n=1 | `50310741` / `50310830` | completed | `frontier_*_profile*_niah_multikey_2` | Exact selected no-tail: `49.33s/sample`, prefill/decode `35.95/13.38s`, native selector/attention `17.92/14.95s`. Stage-split decode-tail: `79.49s/sample`, prefill/decode `46.76/32.73s`, native selector/attention `24.21/36.30s`. Tail path index-build time is small (`2.22s`), so the runtime blocker is per-query native selector/tail-attention work, not V-PQ construction. |
| Decode-tail score-I/O cost-model fix, hard ctx8192, n=1 | `50310869` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_ranked_mass0p97_ctx8192_max512_exactv1024_n1_prefill0_decode1_exactallbypass_costfix_niah_multikey_2` | Score `100.0`, mean total `67.23s/sample`, prefill/decode `35.83/31.40s`. Tail estimator MB/head-query increased from `0.00099` to `0.00188` after charging dense PQ score write/read traffic. Total MB/head-query is now `0.9094`, still close to exact selected because the tail path only affects decode queries in this short-output RULER setting. |
| RULER hard task with causal chunked prefill | `50312466` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_ranked_mass0p97_ctx8192_max512_n4_niah_multikey_2_chunk2048` | Score `100.0`, `4/4` samples, no passthrough, mean total `35.62s/sample`, prefill/decode `24.62/11.00s`, total MB/head-query `0.907`. This improves the previous unchunked exact-selected hard-task run from `40.27s/sample` to `35.62s/sample` without changing cost accounting or output quality. |
| RULER subset with causal chunked prefill | `50312482` / `50312483` / `50312484` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_ranked_mass0p97_ctx8192_max512_n4_{niah_single_1,vt,fwe}_chunk2048` | Scores unchanged versus unchunked: `niah_single_1=100`, `vt=100`, `fwe=75`, all with zero passthrough and unchanged MB/head-query. Runtime improves: `niah_single_1` `41.97s -> 35.44s`, `vt` `32.12s -> 26.20s`, `fwe` `35.69s -> 27.57s`; the gain is almost entirely prefill. |
| HF/logit all-layer exact selected-K/V, synthetic needle, 9,771-token prompt | `50311699` / `50311722` | completed | `attention_efficiency_result/frontier_readiness_20260516_hf_exact_ranked_alllayers_f256*` | Added `--approx_prefill` to the HF intervention runner. Decode-only smoke matched dense text with top-1 agreement `1.0`, but had `64` passthrough prefill calls. True prefill+decode smoke has zero passthrough and still matches target text, but is slow: dense `2.62s`, approximate `85.17s`. Mean/max logit relL2 `0.098/0.167`, hidden relL2 `0.079/0.113`, mean/min logit cosine `0.9948/0.9866`, KL mean/max `0.0038/0.0148`. |
| HF/logit prefill selector-stride sweep, same prompt/config | `50311764` / `50311765` / `50311752` | completed | `attention_efficiency_result/frontier_readiness_20260516_hf_exact_ranked_alllayers_f256_prefilldecode_stride{2,4,8}` | Selector reuse reduces selector MB and runtime but hurts quality too much. Stride1: `85.17s`, total `1.027 MB/head-query`, logit relL2 mean/max `0.098/0.167`. Stride2: `80.32s`, `0.751 MB`, `0.191/0.288`. Stride4: `61.71s`, `0.613 MB`, `0.222/0.316`. Stride8: `46.78s`, `0.544 MB`, `0.217/0.374`. All kept top-1/text match on this easy needle prompt, but the relL2/KL degradation rejects stride reuse as a correctness-preserving readiness path. |
| HF/logit prefill selector backend check, same prompt/config | `50311845` / `50311861` / `50311862` / `50311872` / `50311882` / `50311893` | completed | `attention_efficiency_result/frontier_readiness_20260516_hf_exact_ranked_alllayers_f256_prefilldecode_{profile_mn1,torch_matmul,torch_lut_batched*,native_fused}` | Profiled stride1 max-new=1 shows selector is the largest measured prefill component: native selector `51.3s`, selected attention `29.8s`, index build `2.1s`. Full max-new=8 backend comparison: `torch_lut` `85.17s`, `1.03 MB/head-query`; `native_fused` `109.14s`, same MB/quality, rejected; `torch_matmul` `57.06s`, similar quality, but `3.17 MB/head-query` because it reconstructs/reads PQ-approximated K. Untiled `torch_lut_batched` OOMs; tiled `torch_lut_batched` with 512-query tiles avoids OOM and preserves quality but is slower (`98.34s`); page-blocked tiled LUT is slower again (`118.02s`). Conclusion: keep `torch_lut` as the honest low-MB path and `torch_matmul` only as a labeled runtime-enabling path. |
| HF/logit causal chunked prefill, same prompt/config | `50311909` / `50311910` / `50311930` / `50311931` / `50311944` | completed | `attention_efficiency_result/frontier_readiness_20260516_hf_exact_ranked_alllayers_f256_prefilldecode_lut_chunk{1024,2048,4096,8192}` | Added chunked prefill support for the `torch_lut` selector while preserving `pq_ranked_mass_budget` masking. This keeps zero passthrough, exact text match, and the same modeled traffic (`1.027 MB/head-query`). Runtime improves without the stride-reuse quality loss: chunk1024 `67.90s`, chunk2048 `67.38s`, chunk4096 `70.22s`, chunk8192 `78.70s`, versus unchunked `85.17s`. Logit relL2 mean/max for chunked runs is `0.097/0.157`; hidden relL2 mean/max `0.080/0.115`; top-1 agreement `1.0`. Current sweet spot is chunk2048. Profiled max-new=1 chunk2048 (`50311944`) reduced selector time from `51.31s` to `31.87s` and selected-attention time from `29.84s` to `24.83s`; selector remains the larger bottleneck. |
| HF/logit chunk2048 selector backend re-check | `50311954` / `50311955` / `50311956` / `50312463` | completed | `attention_efficiency_result/frontier_readiness_20260516_hf_exact_ranked_alllayers_f256_prefilldecode_{lutbatched_chunk2048,lutbatched_chunk2048_pb*,nativefused_chunk2048}` | Chunking did not rescue the alternate selector backends. `torch_lut_batched` is `69.99s`, page-blocked batched pb4 is `78.05s`, page-blocked batched pb1 is `102.28s`, and `native_fused` is `102.24s`, all with the same quality/MB/passthrough as chunked `torch_lut`. Keep chunk2048 + `torch_lut` as the current honest low-MB runtime path. |
| LongBench-v2 runner integration | `50312496` / `50312497` / `50312504` / `50312505` / `50312516` / `50312517` | latest completed | `longbench_v2_hf_result/frontier_readiness_20260516_{dense,pagedpq}_llama8b_l8k_n1_v3` | Added `--attention_mode pagedpq` to `benchmark/longbench_v2_hf_eval.py`, exposed selector flags through `benchmark/run_longbench_v2_hf.sh`, repaired the venv dataset stack, and bypassed the failing HF `datasets` streaming parser with a direct JSON-array scanner for `THUDM/LongBench-v2`. The first dense vs paged-PQ smoke now completes. It is not a useful accuracy datapoint because the first example is a long truncated translation case and both dense/paged-PQ score `0.0`, but it proves LongBench can execute the frontier path: paged-PQ has zero passthrough, `0.923 MB/head-query`, and `35.48s` generation time versus dense `5.27s` for 8k input / 16 decode tokens. |
| LongBench-v2 short/easy accuracy smoke | `50312605` / `50312606` / `50312641` / `50312819` / `50312963` | completed | `longbench_v2_hf_result/frontier_readiness_20260516_*_llama8b_short_easy_n4*` | First useful LongBench quality signal is negative for the current exact-selected path. Dense scores `50%` on 4 examples with `2.54s` average generation. Paged-PQ `target=0.97`, max-budget-512 scores `25%` with `29.96s`, zero passthrough, and `0.923 MB/head-query`. Raising to `target=0.99`, max-budget-1024 still scores `25%`, with `39.29s`, `1.134 MB/head-query`, and `1392` mean selected tokens. Raising to `target=0.995`, max-budget-2048 still scores `25%`, with `52.45s`, `1.509 MB/head-query`, and `2161` mean selected tokens. Raising again to `target=0.999`, max-budget-4096 still scores `25%`, with `94.26s`, `2.073 MB/head-query`, and `3314` mean selected tokens. The same dense-correct example stays flipped from `B` to `C`, so the next blocker is not just a small budget increase; we need paired dense-vs-frontier logit/hidden diagnostics on the failing LongBench prompts. |
| LongBench-v2 paired dense diagnostic, dense-correct flipped row `66fc...` | `50313213` / `50313302` / `50313378` | completed | `longbench_v2_hf_result/frontier_readiness_20260516_diag_pagedpq_llama8b_id66fc_*` | Added `--diagnose_dense_reference` to the LongBench runner. Dense greedy is correct (`B`), while adaptive paged-PQ predicts `C`. Whole-vocab metrics look deceptively acceptable: top-1 agreement `0.9375`, mean/max logit relL2 `0.099/0.167`, mean/min logit cosine `0.9948/0.9865`. The task failure is a narrow answer-letter margin flip at step 5: dense `B=30.5`, `C=29.875` (`B` ahead by `0.625`); adaptive paged-PQ `B=29.625`, `C=29.75` (`C` ahead by `0.125`). Fixed-budget 4096 recovers exact dense text and answer: top-1 agreement `1.0`, mean logit relL2 `0.0606`, choice margin error max `0.375`, `2.073 MB/head-query`, `3314` selected tokens. Conclusion: this failure is primarily confidence/ranked-budget under-selection, not an irrecoverable selector-ranking failure. Normal generation fixed-4096 confirmation jobs are queued (`50313507`, `50313508`). |
| Cross-sample paged-PQ cache reset fix | `50313507` / `50313508` / `50313688` / `50313689` / `50315265` / `50315266` / `50315267` | completed | `longbench_v2_hf_result/frontier_readiness_20260516_pagedpq_llama8b_*_resetfix` | Found and fixed a benchmark-harness correctness bug: page/PQ caches persisted across independent examples when a runner kept one patched attention context open. Added `reset_paged_pq_attention_state(model)` and call it before each independent LongBench/RULER sample. Previous multi-sample paged-PQ results are therefore suspect unless rerun after this fix. Post-fix LongBench n=4: adaptive `target=0.999/max4096` now selects only `960` tokens, costs `0.923 MB/head-query`, and scores `25%`. Fixed budgets recover the dense subset score `50%`: fixed1024 selects `1392`, costs `1.134 MB/head-query`, and takes `39.56s/sample`; fixed2048 selects `2161`, costs `1.509 MB/head-query`, and takes `53.47s/sample`; fixed3072 selects `2801`, costs `1.822 MB/head-query`, and takes `70.19s/sample`; fixed4096 selects `3314`, costs `2.073 MB/head-query`, and takes `92.10s/sample`. The cheapest fixed1024 already restores the dense-correct `66fc...` answer to `B`, so the current blocker is the adaptive confidence stop rule under-selecting for LongBench, not selector capacity at modest budget. |
| LongBench-v2 short/easy n=8 reset-safe validation | `50315533` / `50315534` / `50315535` | completed | `longbench_v2_hf_result/frontier_readiness_20260516_*_short_easy_n8_*` | Dense reference scores `25%` on the first 8 short/easy rows. Fixed1024 also scores `25%`, with zero passthrough, `38.15s/sample`, `1.134 MB/head-query`, and `1392` mean selected tokens. Ranked-mass floor/escalation (`min=1024`, `max=2048`, target `0.999`, exact budget accounting) also scores `25%`, but costs more: `54.39s/sample`, `1.476 MB/head-query`, `2092` mean selected tokens. Row-level predictions show fixed1024 preserves the two dense-correct rows; escalation changes one dense-wrong row but does not recover additional correct answers. Current interpretation: fixed1024 is the practical LongBench validation setting for this subset, while simple ranked escalation above that floor is not justified yet. |
| LongBench-v2 short/easy n=16 reset-safe validation | `50315555` / `50315556` | completed | `longbench_v2_hf_result/frontier_readiness_20260516_*_short_easy_n16_*` | Dense reference scores `37.5%`. Fixed1024 scores `43.75%`, with zero passthrough, `37.91s/sample`, `1.134 MB/head-query`, and `1392` mean selected tokens. Row-level agreement is `14/16 = 87.5%`; fixed1024 preserves all 6 dense-correct rows and flips 1 dense-wrong row to correct. Treat the higher accuracy as noise/trajectory change, not proof of superiority, but this is positive readiness evidence that fixed1024 does not lose dense-correct answers on this small LongBench subset. This run also validates the new `pagedpq_config`/`args.json` audit artifacts. |
| LongBench-v2 short/easy n=32 reset-safe validation | `50315605` / `50315606` | completed | `longbench_v2_hf_result/frontier_readiness_20260516_*_short_easy_n32_*` | Dense reference scores `43.75%`. Fixed1024 also scores `43.75%`, with zero passthrough, `36.90s/sample`, `1.134 MB/head-query`, and `1392` mean selected tokens. Prediction agreement is `29/32 = 90.6%`: fixed1024 preserves `13/14` dense-correct rows, loses one dense-correct row, and flips one dense-wrong row to correct. This is usable validation evidence but not full correctness: task accuracy matches, yet row-level drift remains. |
| LongBench-v2 paired diagnostic, fixed1024 lost dense-correct row `66f9625f...` | `50315680` | completed | `longbench_v2_hf_result/frontier_readiness_20260516_diag_pagedpq_llama8b_id66f9625f_fixedk1024_resetfix` | Dense predicts correct `B`; fixed1024 predicts wrong `C`. This is not a tiny benign perturbation: top-1 agreement `0.9375`, mean/max logit relL2 `0.171/0.579`, mean/min logit cosine `0.979/0.833`, mean/max hidden relL2 `0.184/0.401`, mean/min hidden cosine `0.980/0.920`, choice top agreement `0.9375`, max choice-margin error `1.0`. The answer flip occurs at generation step 5: dense `B=27.625`, `C=27.5` (B ahead by `0.125`), while fixed1024 `B=27.875`, `C=28.625` (C ahead by `0.75`). This is direct evidence that benchmark-row failures correlate with much worse relL2/cosine and narrow answer margins. |
| LongBench-v2 paired diagnostics, fixed1024 outcome strata | `50315858` / `50315859` / `50315860` | completed | `longbench_v2_hf_result/frontier_readiness_20260516_diag_pagedpq_llama8b_{preserved_correct,unchanged_wrong,gained_correct}_*_fixedk1024_resetfix` | Comparator rows show the lost dense-correct row is an outlier in error severity. Preserved-correct: exact text match, top1 `1.0`, mean/max logit relL2 `0.091/0.161`, min logit cosine `0.988`. Unchanged-wrong: exact text match, top1 `1.0`, mean/max logit relL2 `0.087/0.234`, min logit cosine `0.977`. Gained-correct: no exact text match, top1 `0.9375`, mean/max logit relL2 `0.111/0.196`, min logit cosine `0.981`. Lost-correct from `50315680`: no exact text match, top1 `0.9375`, mean/max logit relL2 `0.171/0.579`, min logit cosine `0.833`. Initial conclusion: aggregate accuracy hides row-level drift, but severe logit/hidden relL2 and low min-cosine are concentrated in the harmful flip. |
| RULER hard-task reset-fix reruns | `50313846` / `50314715` / `50314975` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_ranked_mass0p97_ctx8192_*_niah_multikey_2*_resetfix*` | Added reset-safe RULER execution and config-auditable summaries. The first reset audit (`50313846`) scored `100%` but used the script defaults `geometric_min=8192`, `geometric_max=65536`, so it selected `3977` tokens and took `164.30s/sample`; this was a config audit, not the intended max512 comparison. The corrected apples-to-apples run (`50314975`) uses `geometric_min=8`, `geometric_max=512`, `granularity=8`: score `100.0`, `0/4` nulls, zero passthrough, mean total `38.14s/sample`, prefill/decode `27.99/10.14s`, `0.907 MB/head-query`, `961` selected tokens. This validates the reset fix and confirms the RULER max512 smoke remains viable. |
| RULER longer-context exact-selected readiness | `50315901` / `50315902`; replacement `50315961` / `50315962` | stream pair cancelled; batched pair completed | `ruler_eval_result/frontier_readiness_20260516/{dense_batched_ctx16384_n2_niah_multikey_2_mn32,frontier_exact_fixed1024_batched_ctx16384_n2_niah_multikey_2_mn32}` | Initial ctx16k streaming dense/fixed1024 pair was cancelled after dense streaming took about `15 min` for the first sample, making the harness itself impractical. Batched ctx16k n=2 with `max_new_tokens=32` works: dense score `50%`, `6.92s/sample`; fixed1024 exact-selected score `50%`, zero passthrough, `99.69s/sample`, prefill/decode `96.71/2.98s`, `1.662 MB/head-query`, `1463` mean selected tokens. Row predictions match exactly. This is a useful correctness smoke, but not runtime-ready: longer-context prefill is now the dominant blocker. |
| RULER ctx16k fixed1024 prefill profile | `50316010` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_fixed1024_batched_ctx16384_n1_profile_mn1` | n=1/max-new=1 with `PROFILE_NATIVE_OPS=1`: total `108.08s/sample`, prefill/decode `107.96/0.12s`, zero passthrough, `1.661 MB/head-query`, `1463` mean selected tokens. Native selector time is `40.84s`, native selected-attention time `38.42s`, index build only `1.88s`; patched attention total `101.40s`. Conclusion: ctx16k runtime is split between selector and selected-attention, with extra orchestration overhead. Optimizing only one kernel will not make long-context RULER comfortable. |
| RULER ctx16k fixed1024 prefill backend check | `50316029` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_fixed1024_batched_ctx16384_n1_profile_matmul_mn1` | `torch_matmul` prefill reduces total `108.08s -> 86.66s` and selector time `40.84s -> 18.07s`, but selected-attention remains `38.44s` and modeled traffic jumps `1.661 -> 5.333 MB/head-query` because selector MB rises to `4.618`. This backend is useful for runtime debugging but is not the honest low-MB frontier. Long-context prefill still needs selected-attention optimization and lower-overhead selector execution. |
| RULER ctx16k fixed1024 prefill profile sweep | `50316057` / `50316058` / `50316059` | completed | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_fixed1024_batched_ctx16384_n1_profile_{chunk4096,chunk8192,page1024_chunk4096}` | Honest `torch_lut` variants do not fix ctx16k prefill. Baseline chunk2048/page512: `108.08s`, selector/attention `40.84/38.42s`, `1.661 MB/head-query`. Chunk4096/page512: `110.18s`, `42.43/39.12s`, same MB. Chunk8192/page512: `120.26s`, `50.03/39.88s`, same MB. Page1024/chunk4096: `108.56s`, selector/attention `37.02/48.68s`, lower modeled MB `1.298` but no runtime gain. Conclusion: simple chunk/page tuning cannot make ctx16k comfortable; selected-attention and selector execution both need deeper optimization. |
| CUDA selected-attention weight-cache optimization | build/test `50316130`; microbench `50316147` / `50316152` / `50316153`; RULER profiles `50316148` / `50316158` / `50316161` / `50316353` | completed | `slurm_out/selpq_build_test-50316130.out`; `ruler_eval_result/frontier_readiness_20260516/frontier_exact_fixed1024_batched_ctx16384_n1_profile_weightcache*` | Changed the exact and V-PQ selected-attention CUDA kernels to compute softmax weights once per selected token in shared memory instead of recomputing `exp(logit-max)/denom` for every output dimension. CUDA parity passed. Synthetic selected-attention microbench at 2048 positions / 32 heads / bf16 K/V reports `74.75/122.42/212.74 ms` for selected `512/1024/2048`. End-to-end ctx16k fixed1024 profile improves total `108.08s -> 97.54s`, prefill `107.96s -> 97.16s`, selected-attention `38.42s -> 30.38s`; selector remains about `40.19s`. Vectorized mean-only prefill accounting then reduces the same clean path to `95.87s` total / `95.52s` prefill, with selected-attention `30.26s`, selector `40.98s`, and unchanged modeled cost `1.661 MB/head-query`. A follow-up Python `torch_lut` q-sub/code-cast precompute attempt worsened selector time to about `60.9s` and total to `118-120s`, so that patch was reverted. Net conclusion: the CUDA attention fix is real and accounting overhead was measurable, but ctx16k is still not comfortable; selector execution is now the largest measured kernel slice. |
| Streaming page-wise `torch_lut` selector attempt | smoke `50316190` / `50316195`; ctx16k profile `50316234` | rejected | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_fixed1024_batched_ctx16384_n1_profile_streamlut` | Added a `torch_lut_streaming` backend that keeps a running top-k over pages instead of materializing all page scores then doing one large top-k. The first smoke failed only because RULER/LongBench wrapper parsers did not know the new backend; parser lists were fixed and smoke `50316195` ran. At ctx16k fixed1024 this is much worse than the current `torch_lut`: total `159.38s`, prefill `159.10s`, selector `99.12s`, selected-attention `30.29s`, same `1.661 MB/head-query` and `1463` selected tokens. Conclusion: repeated per-page top-k is more expensive than the large final top-k; do not use this backend for benchmark-readiness runs. |
| fp16 temporary-score `torch_lut` selector | smoke `50316378`; ctx16k profile `50316385`; LongBench n=8 `50316401`; LongBench n=20 `50316493`; RULER A/B `50316499` / `50316522`; corrected LongBench n=32 `50316553` | promote for fixed1024 exact-selected validation tier | `ruler_eval_result/frontier_readiness_20260516/frontier_exact_fixed1024_batched_ctx16384_n1_profile_lutfp16`; `longbench_v2_hf_result/frontier_readiness_20260516_pagedpq_lutfp16_llama8b_short_easy_n32_scan1000`; `ruler_eval_result/frontier_readiness_20260516/frontier_exact_fixed1024_lutfp16_ctx8192_n4_niah_multikey_2` | Added `torch_lut_fp16`, which still computes PQ LUT scores from the same online codebooks/codes but stores concatenated temporary selector scores in fp16 before `topk`. For fixed exact-selected attention, selected K logits are recomputed exactly, so fp16 affects only selector ordering/ties, not final logits for selected tokens. ctx16k fixed1024 improves over the current clean path: `95.87s -> 93.11s` total, `95.52s -> 92.78s` prefill, selector `40.98s -> 37.58s`, selected-attention `30.26s -> 30.50s`, same modeled `1.661 MB/head-query`. LongBench-v2 short/easy n=8 passes the first quality check: score remains `25%`, exact prediction agreement with the previous fp32 `torch_lut` fixed1024 run is `8/8`, response exact match is `7/8`, and average generation improves `38.15s -> 34.46s`. The first attempted n=32 fp16 run used `DATASET_SCAN_LIMIT=200`, so it produced only `20` examples; on those same IDs it exactly matches fp32 `torch_lut` predictions and judge outcomes (`20/20`), with `40%` accuracy and `33.99s/example`. Corrected LongBench n=32 with `DATASET_SCAN_LIMIT=1000` also passes: score `43.75%`, fp16 vs fp32 `torch_lut` prediction agreement `32/32`, judge agreement `32/32`, response exact match `29/32`, average generation `33.61s` versus fp32 `36.90s`, same modeled cost `1.134 MB/head-query`, and zero passthrough. RULER 8k `niah_multikey_2` apples-to-apples fixed1024 A/B also passes: fp16 and fp32 both score `100%`, outputs match `4/4`, modeled cost is identical at `1.117 MB/head-query`, and fp16 is slightly faster (`41.07s/sample` vs `42.28s/sample`). Conclusion: use `torch_lut_fp16` as the default prefill selector backend for the fixed1024 exact-selected validation tier. This does not solve the separate full compressed-tail frontier runtime blocker. |
| Full compressed-tail prefill smokes | geometric unpatched `50316679`; bitset-patched build `50316705`; geometric bitset `50316728`; ranked-mass bitset `50316771` | rejected as benchmark-ready path | `ruler_eval_result/frontier_readiness_20260516/frontier_complete_geometric_tail_ctx4096_n1_profile*`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_ranked_tail_ctx4096_n1_profile`; `slurm_out/tailbit_build-50316705.out` | Ran strict small smokes for the complete frontier path at ctx4096, n=1, max_new=1: selected V-PQ, compressed V-PQ tail, native prefill score reuse, and native decode tail. Geometric confidence unpatched stayed inside the single sample for `20:36` and was cancelled with no prediction. Inspection showed the from-scores V-PQ tail kernel checked ranked-token membership by scanning all ranked tokens for every tail token, so a shared-memory ranked-token bitset was implemented and CUDA parity passed in `50316705`. Geometric confidence with the bitset created an empty prediction file but still stayed inside the single sample for `15:59` and was cancelled before summary. To separate confidence-loop cost from the tail kernel, `pq_ranked_mass_budget` was also tested with the bitset; it still stayed inside the single sample for `10:42` with an empty prediction file and was cancelled. Conclusion: the bitset removes one obvious O(tail * ranked) check, but the complete prefill path remains structurally too expensive. The blocker is now the full prefill-query/head selected+tail kernel formulation itself, not only geometric confidence. Fixed1024 exact-selected validation remains a practical readiness tier, not the final compressed-tail frontier algorithm. |
| Selector-rank exact selected-V policy | build/parity `50317079`; microbench `50317080`; ctx4096 one-token smoke `50317178`; ctx4096 quality smokes `50317263` / `50317296` | useful MB fix, not runtime fix | `slurm_out/selrank_build2-50317079.out`; `slurm_out/selrank_tailbench2-50317080.out`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_ranked_tail_selrank_ctx4096_n1_profile_v2`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_ranked_tail_selrank_ctx4096_n1_quality*` | Added `selected_value_exact_rule=selector_rank`, encoded in native CUDA as negative `exact_value_top`. This keeps exact V for the first selected tokens in selector order and avoids the serial exact-logit top-k over the selected set. CUDA parity passed, including a new selector-rank mixed V-PQ unit case. The synthetic tail microbench improved selected `4096` from the previous `4801 ms` cliff to `158 ms`, and selected `2048` to `57 ms`. In the real ctx4096 one-token RULER smoke, modeled MB improved (`1.054 -> 0.918 MB/head-query`; exact KV `0.861 -> 0.724`), but wall time barely moved (`58.7s -> 55.6s`) and native attention stayed about `44s`. The normal 128-token quality smoke scores `100%`; profiled runtime is `128.46s/sample` (`58.22s` prefill, `70.24s` decode), and no-profile runtime is still `126.10s/sample` (`57.78s` prefill, `68.33s` decode) at the same `0.945 MB/head-query`. Conclusion: selector-rank fixes selected-V subset overhead and cost accounting, and quality is not obviously broken on this RULER sample, but the benchmark blocker is still the compressed-tail prefill/decode kernel formulation itself. |
| Full compressed-tail proxy-mass sweep | `50317404` / `50317405` / `50317406` / `50317443` / `50317444` / `50317478` | confirms tail runtime floor | `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_selrank_ctx4096_n1_mass_*`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_selrank_ctx4096_n1_tailfloor` | Same ctx4096 hard RULER sample with `selected_value_exact_rule=selector_rank`, compressed selected V, compressed V-PQ tail, and ranked proxy-mass budgets. Lowering the proxy-mass target reduces runtime but does not make the full tail path benchmark-comfortable: target `0.97/0.90/0.80/0.70/0.50/0.30` gives score `100%` with total seconds `126.10/104.22/89.92/81.56/67.79/62.45`. The tail-floor target `0.0` takes `61.11s` and drops score to `0%`. Current summaries use conservative upper-bound accounting, so selected-token MB is not the accepted-budget truth for this sweep. Runtime evidence is still clear: even when the exact selected work is driven toward the minimum, compressed-tail prefill/decode has a large per-query/head scan floor. This path needs a different tail formulation or a substantially faster CUDA implementation before real LongBench/RULER suites. |
| Full compressed-tail exact-accounting check | `50317528` | diagnostic only; not promoted | `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_selrank_ctx4096_n1_mass_0p30_exactacct` | Re-ran the target `0.30` case with `RANKED_CONFIDENCE_COST_MODE=exact` to report accepted budgets instead of conservative upper-bound cost. The measured accepted selected-token count falls from the upper-bound-reported `1826.5` to `645.3`, with modeled step MB `0.815 -> 0.491 MB/head-query`. However, runtime jumps to `223.41s/sample` because exact accounting syncs budget counts during decode, and the generated answer flips from the correct `1742169` to wrong `4645172`. Since this flag should be accounting-only, do not use exact-accounting runs for quality/runtime claims until the cost-mode side effect is isolated. Upper-bound accounting remains the deployable fast path; exact mode is currently an intrusive diagnostic. |
| Selected-weight cache attempt in tail prefill kernel | build/parity `50317561`; bench-only `50317600`; cancelled RULER `50317607`; revert rebuild `50317618` | rejected | `slurm_out/tail_weightcache_build_test-50317561.out`; `slurm_out/tail_weightcache_bench-50317600.out` | Tried caching selected softmax weights in shared memory inside the causal selected+V-PQ-tail kernel to avoid recomputing `exp(selected_logit - max)` for every output dimension. CUDA parity still passed, but the microbench regressed badly: selected `2048/4096` worsened from previous selector-rank `57.45/158.10 ms` to `81.91/299.16 ms`. The likely cause is lower occupancy / larger shared-memory footprint dominating the saved exponentials. The patched RULER repeat was cancelled, source was reverted, and a rebuild was submitted to restore the known-good extension. Do not pursue this exact weight-cache formulation. |
| Warp-tiled selected logits in tail prefill kernel | build/parity `50317618`; bench `50317643`; RULER `50317654` | rejected for end-to-end | `slurm_out/tail_revert_build_test-50317618.out`; `slurm_out/tail_warp_bench-50317643.out`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_selrank_warp_ctx4096_n1_mass_0p30` | The microbench looked neutral/low-budget positive, but the end-to-end ctx4096 target-0.30 RULER repeat failed: score `0.0`, generated wrong repeated answer `4645172` instead of `1742169`, total `226.03s/sample`, prefill/decode `39.16/186.87s`, step `0.812 MB/head-query`. Treat this variant as a regression, not a promotion. Follow-up audit found the prior warp-tiled change did not cover the score-reuse from-scores kernel used by this RULER path, so a separate from-scores patch must be validated before any further claim. |
| From-scores warp-tiled selected logits | build/parity/bench `50317691` / `50317743`; RULER `50317744` / `50317757` | promoted for further validation | `slurm_out/tail_fromscores_warp-50317691.out`; `slurm_out/tail_fromscores_warp_bench2-50317743.out`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_fromscoreswarp_ctx4096_n1_mass_0p30b`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_fromscoreswarp_ctx4096_n1_mass_0p97` | Moved the warp-tiled selected-logit computation into the actual score-reuse from-scores prefill kernel. CUDA parity passed. Tail microbench improved materially at all selected counts: selected `512/1024/2048/4096` now `4.96/11.12/31.85/85.94 ms`, versus previous `9.32/19.22/57.44/158.12 ms`. End-to-end ctx4096 target-0.30 RULER is clean: score `100%`, no passthrough, total `53.16s/sample`, prefill/decode `28.92/24.24s`, step `0.815 MB/head-query`. High-quality target `0.97` is also clean: score `100%`, no passthrough, total `102.68s/sample`, prefill/decode `39.02/63.66s`, step `0.815 MB/head-query`. This is a real prefill-tail speedup versus the prior `126.10s/sample` high-target run, but decode remains too slow for comfortable full-suite benchmarks. |
| Decode ranked-logit warp tiling | build/parity `50317800`; RULER `50317821` | promoted for further validation | `slurm_out/decode_rankwarp_build_test-50317800.out`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_decode_rankwarp_ctx4096_n1_mass_0p97` | Parallelized the decode aggregated-tail ranked/base exact-K logit kernel across warps instead of launching one scalar thread per head. CUDA parity passed, including the selected-tail aggregated decode test. End-to-end ctx4096 high-quality target `0.97` stays correct: score `100%`, no passthrough, same modeled step `0.815 MB/head-query`, and total runtime improves `102.68 -> 54.92s/sample`. Decode drops sharply `63.66 -> 16.77s`; prefill is now the dominant remaining cost at `38.15s`. |
| Selected compressed-V code-weight aggregation | build/parity `50318047` / `50318194`; RULER `50318332` | promoted for further validation | `slurm_out/selected_vpq_agg_build_test-50318047.out`; `slurm_out/fromscores_selected_vpq_agg_test-50318194.out`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_vpqagg_ctx4096_n1_mass_0p97` | Aggregates compressed selected-V weights into the same V-PQ code-weight table used for compressed tail, so the output loop scans exact selected V plus compact code buckets instead of expanding every compressed selected token per output dimension. CUDA parity passed for both regular and from-scores prefill kernels. End-to-end ctx4096 high-quality target `0.97` remains correct: score `100%`, no passthrough, same modeled step `0.815 MB/head-query`, and total runtime improves `54.92 -> 48.31s/sample`. Prefill drops `38.15 -> 31.55s`; decode is unchanged at `16.76s`. |
| Compressed-tail cleanup: skip compressed selected exp + visible causal tail cap | build/parity `50318497` / `50318593`; RULER `50318538` / `50318624`; profile `50318658` | promoted, minor runtime win | `slurm_out/vpqagg_skip_exp_build_test-50318497.out`; `slurm_out/visible_tail_build_test-50318593.out`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_vpqagg_skip_ctx4096_n1_mass_0p97`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_visible_ctx4096_n1_mass_0p97`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_visible_ctx4096_n1_mass_0p97_profile_mn1` | Removed redundant exponentiation for compressed selected-V tokens once their weights are aggregated by V-PQ code, then capped the causal tail scan to rows visible to the current prefill query. CUDA parity passed for both changes. End-to-end ctx4096 target `0.97` remains correct: score `100%`, no passthrough, modeled step `0.815 MB/head-query`. Runtime improves slightly from `48.31 -> 47.91 -> 47.80s/sample`; prefill is `31.20s` and decode is `16.60s`. The one-token profile reports total `33.56s`, prefill/decode `33.17/0.39s`, native selector `5.25s`, native attention `19.66s`, patched attention `30.27s`, and index build `1.19s`. This confirms the remaining benchmark-readiness blocker is still prefill selected/tail attention, not decode or index construction. |
| Active-page code-weight cleanup and ranked-cap sweep | build/parity `50318718`; RULER `50318801` / `50318802` / `50318803` / `50318854` / `50318855` / `50318898` / `50318899`; n=4 `50318913` / `50318914` | cap64 promoted to broader RULER validation | `slurm_out/activecw_build_test-50318718.out`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_activecw_ctx4096_n1_mass_0p97_cap*`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_activecw_ctx4096_n4_mass_0p97_cap*` | Restricted from-scores V-PQ code-weight zero/output scans to visible complete pages; CUDA parity passed. Then swept deployable `GEOMETRIC_MAX_BUDGET` caps on the same ctx4096 hard RULER sample with target `0.97`. All single-sample runs scored `100%` with zero passthrough. Cap `4096/2048/1024/512/256/128/64` gives total seconds `47.71/39.31/30.65/26.70/22.28/25.16/24.82`, modeled step MB/head-query `0.815/0.775/0.665/0.584/0.534/0.485/0.460`, and mean selected tokens `1826/1664/1214/881/679/578/527`. Multi-sample hard-task validation at ctx4096, n=4 also scores `100%` for cap64 and cap256. Cap64 reports `21.72s/sample`, prefill/decode `12.50/9.22s`, step `0.460 MB/head-query`, and selected tokens `527`; cap256 reports `22.38s/sample`, `0.534 MB/head-query`, and selected tokens `679`. Interpretation: cap64 is the current fastest complete compressed-tail readiness candidate for broader task validation, but it is still only one RULER task. |
| RULER ctx4096 paired cap64 validation | dense `50318952` / `50318954` / `50318956`; frontier `50318953` / `50318955` / `50318957` | clean limited validation | `ruler_eval_result/frontier_readiness_20260516/dense_batched_ctx4096_n4_*_paired_cap64`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_activecw_ctx4096_n4_mass_0p97_cap64_*` | Ran dense vs complete-frontier cap64 on three additional self-contained RULER tasks, ctx4096, n=4. Dense scores are all `100%` at `6.58/6.78/6.87s/sample` for `niah_single_1/vt/fwe`. Frontier cap64 also scores `100%` on all three with zero passthrough. Frontier total seconds are `22.24/22.82/23.89`; step MB/head-query is `0.475/0.487/0.498`; selected tokens are about `532-535`. Interpretation: cap64 is no longer just a one-sample hard-task success, but this is still a small 4k RULER subset, not final benchmark readiness. |
| RULER ctx8192 hard-task cap64 validation | `50318998`; profile `50319033` | correct but not full-suite comfortable | `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_activecw_ctx8192_n4_mass_0p97_cap64_niah_multikey_2`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_activecw_ctx8192_n1_mass_0p97_cap64_profile_mn1` | Ran complete-frontier cap64 on `niah_multikey_2`, ctx8192, n=4. Score is `100%`, nulls `0/4`, zero passthrough. Runtime is `53.26s/sample`, with prefill/decode `43.16/10.11s`; modeled step is `0.793 MB/head-query`, split into selector `0.490`, exact KV `0.272`, tail estimator `0.031`, with `556` selected tokens. One-token profile reports total `47.35s`, prefill/decode `46.97/0.38s`, native selector `22.63s`, native attention `7.92s`, patched attention `42.15s`, index build `1.37s`. Interpretation: cap64 survives this longer hard task, but 8k prefill is still too slow for comfortable full RULER sweeps; at 8k the selector score path, not selected/tail attention, is now the dominant measured kernel slice. |
| RULER ctx8192 cap64 selector-backend A/B | profile `50319033` / `50319052`; n=4 `50318998` / `50319143` | matmul is faster but higher-MB | `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_activecw_ctx8192_n1_mass_0p97_cap64_profile_mn1`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_activecw_ctx8192_n1_mass_0p97_cap64_matmul_profile_mn1`; `ruler_eval_result/frontier_readiness_20260516/frontier_complete_tail_activecw_ctx8192_n4_mass_0p97_cap64_matmul_niah_multikey_2` | Compared `native` prefill selector to `torch_matmul` for complete-frontier cap64 at ctx8192. One-token profiles: native total `47.35s`, prefill `46.97s`, selector `22.63s`, attention `7.92s`, step `0.784 MB/head-query`; matmul total `36.35s`, prefill `35.96s`, selector `11.36s`, attention `8.21s`, step `2.452 MB/head-query`. Normal n=4 task evaluation also holds: both score `100%` on `niah_multikey_2`; native is `53.26s/sample` with `0.793 MB/head-query`, while matmul is `41.91s/sample` with `2.434 MB/head-query`. Interpretation: use native for honest low-MB algorithmic accounting, but matmul is the current practical backend when wall-clock benchmark throughput is the immediate blocker. |
| RULER ctx8192 paired cap64 validation | dense `50319346` / `50319348` / `50319350` / `50319352`; frontier `50319347` / `50319349` / `50319351` / `50319353` | clean limited validation | `ruler_eval_result/frontier_readiness_20260516/*ctx8192_n4*cap64*` | Ran dense vs complete-frontier cap64 on four self-contained RULER tasks, ctx8192, n=4, using the practical `torch_matmul` selector path. Scores match dense on all four tasks: `niah_single_1` `100/100`, `niah_multikey_2` `100/100`, `vt` `100/100`, `fwe` `75/75`, with zero frontier nulls. Frontier runtimes are `40.87/37.26/34.33/35.02s/sample` for `niah_single_1/niah_multikey_2/vt/fwe`; dense is `8.87/8.82/3.14/4.68s/sample`. Frontier modeled step MB/head-query is `2.43-2.48`, mostly selector MB `2.13-2.18`, with exact KV about `0.27` and tail about `0.03`. Interpretation: cap64 complete-frontier now has clean RULER 8k task preservation across a small task set, but wall-clock is still `4.2x-10.9x` dense depending on task and the practical selector backend has high modeled selector traffic. |
| LongBench-v2 complete-frontier cap64 smoke, ctx8192 n=8 | frontier `50319170`; temp0 pair `50319203` / `50319204`; diagnostics `50319234` / `50319235` | clean limited validation | `longbench_v2_hf_result/frontier_readiness_20260516_fulltail_cap64_matmul_llama8b_short_easy_n8`; `longbench_v2_hf_result/frontier_readiness_20260516_{dense,fulltail_cap64_matmul}_llama8b_short_easy_n8_temp0*`; `longbench_v2_hf_result/frontier_readiness_20260516_diag_fulltail64_temp0_*` | The complete compressed-tail frontier path now runs LongBench-v2 short/easy n=8 with zero passthrough. Deterministic paired rerun with identical filters and `TEMPERATURE=0.0`: dense `50319203` scores `25%` at `2.63s/sample`; frontier cap64 `50319204` scores `50%` at `31.73s/sample`, step `2.567 MB/head-query`, selector/exact/tail `2.266/0.270/0.031 MB/head-query`, selected tokens `554`, tail samples `3550`. Row-level comparison: frontier and dense have the same correctness on `6/8`; frontier changes two dense-wrong rows into correct answers (`66fa7f1d...`, `66ec3d1d...`) and loses no dense-correct rows. The earlier `temperature=0.1` run `50319170` produced the same predictions/judges. Dense-reference diagnostics on the two changed rows show nontrivial but not catastrophic drift: `66fa7f1d...` mean/max logit relL2 `0.0536/0.0877`, min logit cosine `0.9962`, mean/max hidden relL2 `0.0596/0.1098`, min hidden cosine `0.9940`, top1 agreement `0.9375`; `66ec3d1d...` mean/max logit relL2 `0.0663/0.1455`, min logit cosine `0.9894`, mean/max hidden relL2 `0.0804/0.2058`, min hidden cosine `0.9789`, top1 agreement `0.9375`. In both rows, the frontier drift changes a dense-wrong answer into the correct answer, so this is evidence that task accuracy is not monotonic in dense-output similarity on a tiny subset; broader paired diagnostics are still required. |
| LongBench-v2 complete-frontier cap64, ctx8192 n=32 | dense `50319246`; frontier `50319247` | clean limited validation | `longbench_v2_hf_result/frontier_readiness_20260516_dense_llama8b_short_easy_n32_temp0_scan1000`; `longbench_v2_hf_result/frontier_readiness_20260516_fulltail_cap64_matmul_llama8b_short_easy_n32_temp0` | Broadened the deterministic paired LongBench-v2 short/easy run to n=32. Dense scores `43.75%` at `2.26s/sample`; complete-frontier cap64 scores `50.0%` at `30.58s/sample`, zero passthrough, step `2.567 MB/head-query`, selector/exact/tail `2.266/0.270/0.031 MB/head-query`, selected tokens `554`, tail samples `3550`. Row-level comparison: dense vs frontier has same correctness on `30/32`; frontier recovers the same two dense-wrong rows from the n=8 run and loses no dense-correct rows. Compared with fixed1024 exact-selected (`43.75%`, `33.61s/sample`, `1.134 MB/head-query`), cap64 full-tail is more accurate and slightly faster wall-clock, but its modeled MB is higher because the current `torch_matmul` selector path dominates selector traffic. Interpretation: this is the first clean downstream evidence that the complete frontier path can run a nontrivial LongBench validation batch and preserve/improve task accuracy, but n=32 short/easy is still not full benchmark readiness. |
| LongBench-v2 dense-reference diagnostics, cap64 controls | changed rows `50319234` / `50319235`; controls `50319321` / `50319322` / `50319323` | initial relL2-to-task mapping | `longbench_v2_hf_result/frontier_readiness_20260516_diag_fulltail64_temp0_*` | Ran dense-reference teacher-forced diagnostics for the two answer-improving rows plus three controls. The two improved rows have mean/max logit relL2 `0.0536/0.0877` and `0.0663/0.1455`, min logit cosine `0.9962` and `0.9894`, mean/max hidden relL2 `0.0596/0.1098` and `0.0804/0.2058`, top1 agreement `0.9375` for both. Controls show comparable or even larger drift without task change: unchanged-correct `66f36490...` has mean/max logit relL2 `0.0715/0.1391`, min logit cosine `0.9903`, mean/max hidden relL2 `0.0759/0.1274`, top1 agreement `1.0`; unchanged-wrong `66f78ecf...` has `0.0483/0.0859`, min logit cosine `0.9966`, hidden `0.0613/0.0968`, top1 `1.0`. The changed-prediction/still-wrong control from the batch did not reproduce the changed prediction in diagnostic free-run, so use it only as a stability caveat. Current conclusion: on this tiny diagnostic set, relL2/cosine magnitude alone does not determine task correctness; answer margin, affected-token identity, and generation trajectory matter. Need broader diagnostics before claiming a threshold. |

Benchmark hygiene update: `benchmark/run_longbench_v2_hf.sh` and `benchmark/longbench_v2_hf_eval.py` now default `DATASET_SCAN_LIMIT=1000` instead of `200`, because the lower default silently produced only `20` short/easy examples for a requested `MAX_EXAMPLES=32`. Added `benchmark/compare_longbench_runs.py` to make row-level LongBench agreement checks reproducible instead of relying on one-off inline scripts.

Ctx4096/8192 readiness interpretation: the strict CUDA path is functionally valid on the tested RULER tasks, and cap64 makes 4k limited validation practical. The older fixed-budget path is `4.55x` dense at 4k (`38.68s` vs `8.49s` per sample). The conservative complete compressed-tail path is `47.80s/sample` at cap4096; cap64 gives `21.72s/sample` on ctx4096 `niah_multikey_2` n=4 with score `100%` and `0.460 MB/head-query`. At ctx8192, cap64 still scores `100%` on `niah_multikey_2` n=4; native costs `53.26s/sample` with `0.793 MB/head-query`, while matmul costs `41.91s/sample` with `2.434 MB/head-query`. This is enough for targeted validation, not enough for full RULER/LongBench benchmark readiness.

Ctx8192 readiness interpretation: the practical path has shifted. The deployable ranked-budget exact-selected path recovers the hard `niah_multikey_2` case at `35.62s/sample` with causal chunked prefill, about `3.7x` the dense hard-task reference and faster than the previous unchunked `40.27s/sample`. This is usable for limited RULER/LongBench validation, but it is selector-only with exact selected K/V. The stage-split prefill-exact/decode-tail path is now correct and stable at `66.77s/sample`, but decode tail is not runtime-worthwhile yet: it adds little modeled MB and about `20s/sample` unprofiled / `30s` profiled over ranked exact selected-K/V. The full compressed-tail prefill path remains not benchmark-ready (`194.32s/sample`). A true all-layer HF prefill+decode approximation smoke now has zero passthrough, task text match, and causal chunked prefill down to `67.38s` for a 9,771-token prompt / 8 decode tokens, but this is still not benchmark-comfortable for full suites. The active blocker for the full frontier-family algorithm is making prefill selected attention and selected/tail-value compression useful enough, and fast enough, to justify enabling them in real benchmark sweeps.

Additional RULER downstream smoke batch submitted under `notes/slurm_manifests/ruler_downstream_smoke_20260515.tsv`: dense vs fixed-budget exact selected-K/V runtime probe at 8k, 4 samples. Initial `niah_single_2` and `niah_multivalue` jobs failed because the local RULER essay fixture `PaulGrahamEssays.json` is missing, so the self-contained replacements are `niah_multikey_2`, `vt`, and `fwe`. This is a task-level ablation batch while the full confidence frontier remains blocked by GPU runtime.

Fresh CUDA correctness gate: Slurm `50303686` rebuilt `benchmark/selector_eval/cuda_ext` on `spgpu` and passed `test_fullscan_pq_topk.py`, `test_gpu_vpq_helpers.py`, and `test_online_page_append.py`. This covers native fullscan PQ top-k/scored top-k parity, selected-attention CUDA parity in the fullscan test, GPU V-PQ reconstruction parity, and online page append vs snapshot build parity.

Downstream smoke result, 8k, 4 samples:

| task | dense score | fixed-budget exact-LUT score | dense sec/sample | exact-LUT sec/sample | exact-LUT MB/head-query | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `niah_single_1` | 100.0 | 100.0 | 9.54 | 31.91 | 0.688 | Runtime-feasible ablation reproduces prior single-needle quality. |
| `vt` | 100.0 | 100.0 | 3.49 | 24.32 | 0.685 | Variable-tracking smoke passes. |
| `niah_multikey_2` | 100.0 | 75.0 | 9.18 | 31.57 | 0.685 | Fixed budget loses task accuracy; not robust enough for benchmark-ready frontier claims. |
| `fwe` | 75.0 | 83.33 | 4.38 | 25.67 | 0.685 | Noisy 4-sample task; sparse does not obviously degrade, but dense baseline is already below 100. |

Interpretation: the exact-LUT path is useful for runtime and task-level ablations, but the `niah_multikey_2` drop confirms that a fixed `budget=8` selector is not the frontier algorithm we should use for final benchmark claims. The full confidence/tail path still needs GPU-performance work before real LongBench/RULER validation.

Ranked-budget exact-selected RULER result, 8k, 4 samples, `pq_ranked_mass_budget`, max budget `512`, conservative upper-bound cost accounting:

| task | dense score | ranked exact score | dense sec/sample | ranked exact sec/sample | ranked exact MB/head-query | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `niah_single_1` | 100.0 | 100.0 | 9.54 | 41.97 | 0.911 | Passes single-needle smoke with deployable online budget. |
| `vt` | 100.0 | 100.0 | 3.49 | 32.12 | 0.907 | Passes variable-tracking smoke. |
| `niah_multikey_2` | 100.0 | 100.0 | 9.18 | 40.27 | 0.907 | Fixes the fixed-budget under-retrieval failure. |
| `fwe` | 75.0 | 75.0 | 4.38 | 35.69 | 0.907 | Matches the weak dense baseline on this noisy 4-sample setting. |

Interpretation: ranked-budget exact-selected is the current benchmark-practical deployable tier. It is not the full compressed-tail frontier, but it has no oracle stop rule, no passthrough, and can run short RULER validation batches in minutes rather than hours. Use it for initial relL2-to-task-accuracy mapping while full selected-V/tail compression remains blocked on GPU runtime.

Follow-up budget recovery for `niah_multikey_2`:

| budget | Slurm | score | sec/sample | MB/head-query | selected tokens | interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | `50303704` | 75.0 | 31.57 | 0.685 | 506 | Low fixed budget fails. |
| 16 | `50303724` | 75.0 | 30.85 | 0.688 | 513 | Tiny budget increase does not help. |
| 32 | `50303725` | 75.0 | 31.53 | 0.696 | 528 | Still fails. |
| 128 | `50303748` | 75.0 | 31.90 | 0.738 | 614 | Still fails despite moderate extra exact KV. |
| 512 | `50303749` | 100.0 | 39.33 | 0.907 | 961 | Recovers dense score, so the task failure is budget/coverage-sensitive. |

Interpretation: `niah_multikey_2` is a concrete robustness case where fixed low budget under-retrieves. A benchmark-ready frontier path needs an online budget/confidence rule that escalates hard queries toward the `budget=512` regime when needed, without paying that cost for easy tasks.

Hard-task confidence check: Slurm `50303900` and `50303901` tested `pq_ranked_mass_budget` on `niah_multikey_2` with max budget `512` and ranked-mass targets `0.97`/`0.99`. Both escalated to the same selected count as fixed `budget=512` (`961` tokens) but scored `0.0` with all-compressed selected V plus compressed tail. Runtime was also not benchmark-comfortable (`222.91-241.75s/sample`). This says confidence/budget escalation can reach the hard-query token count, but the compressed-value path is not task-ready under the all-compressed selected-V setting.

Follow-up isolation: Slurm `50303989` reran ranked-mass confidence with `selected_value_exact_top=1024`, making all selected V exact for this hard task while keeping the compressed tail. It recovered score `100.0` with selected tokens `961`, mean total `224.56s/sample`, and mean total MB/head-query `0.936` (`selector 0.438`, `exactKV 0.469`, `tail 0.028`). Conclusion: the `0.0` score in `50303900/50303901` was selected-V compression failure, not tail-estimator failure. The path is still far too slow for benchmark sweeps.

Profile result: Slurm `50304144` repeated the ranked-mass exact-selected-V hard-task path for one sample with `PROFILE_NATIVE_OPS=1`. Score was `100.0`; total stream time `224.22s`, prefill/decode `157.18/67.04s`, mean total MB/head-query `0.936`, selected tokens `961`. Profile totals: native selector `22.99s`, native attention `125.48s`, patched attention `217.43s`. Interpretation: selector time is not the main blocker; selected/tail attention plus Python/HF patch overhead dominate. The next GPU-performance target should be the selected/tail attention path and reducing per-layer/per-query patch overhead, not just fullscan top-k.

Negative optimization check: a shared-memory ranked-token bitmap was tested in the prefill V-PQ tail kernel to avoid scanning the selected list for every tail token. CUDA validation passed (`50304277`), but the hard-task profile was unchanged/slightly worse (`50304295` vs `50304144`): score `100.0` both, total `227.55s` vs `224.22s`, prefill/decode `161.02/66.53s` vs `157.18/67.04s`, native attention `125.45s` vs `125.48s`, native selector `22.95s` vs `22.99s`, patched attention `217.49s` vs `217.43s`. This rules out selected-membership checks as the dominant bottleneck; the source change was not kept. The reverted CUDA source was rebuilt and validated in Slurm `50304683`. The core cost is still per-query tail scoring/aggregation, selected/tail output construction, and HF patch overhead.

### Causal Prefill Score Reuse 2026-05-15

Implemented and validated a strict CUDA path that materializes causal PQ selector scores once and reuses them in the compressed-tail attention kernel:

- New selector API: `gqa_causal_fullscan_pq_topk_scores`.
- New prefill tail APIs: `gqa_causal_vpq_tail_from_scores` and `gqa_causal_vpq_selected_tail_from_scores`.
- HF/RULER flag: `--prefill_tail_score_reuse`, exposed through `PREFILL_TAIL_SCORE_REUSE=1`.
- Safety: the flag fails fast with `prefill_selector_stride > 1`, because using anchor-query scores for non-anchor queries would silently change the algorithm.
- Accounting: modeled selector MB includes dense score write traffic and two tail-score read passes when score reuse is enabled.
- CUDA build/unit validation passed on `spgpu`: Slurm `50300872`.

Controlled one-sample RULER A/B, `niah_single_1`, strict `cuda_ext`, page size `128`, budget `64`, selected V-PQ with exact top `32`, all layers patched:

| run | Slurm | context | score | stream seconds | prefill / decode seconds | native selector seconds | native attention seconds | mean total MB/head-query | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stride1 regular tail | `50300911` | 4096 | 100.0 | 50.81 | 34.00 / 16.81 | 8.63 | 24.49 | 1.008 | Control without score reuse. |
| stride1 score reuse | `50300934` | 4096 | 100.0 | 38.72 | 22.87 / 15.85 | 8.60 | 14.38 | 1.032 | Attention improves; extra score IO is counted. |
| stride1 torch-matmul score reuse | `50301134` | 4096 | 100.0 | 34.54 | 17.67 / 16.86 | 3.34 | 14.26 | 1.756 | Explicit PQ-reconstructed-K/cuBLAS selector backend; faster runtime, higher modeled selector MB. |
| stride1 regular tail | `50300962` | 8192 | 100.0 | 154.64 | 136.53 / 18.11 | 53.15 | 80.63 | 2.109 | Control at 8k; prefill tail is very expensive. |
| stride1 score reuse | `50300963` | 8192 | 100.0 | 94.85 | 77.86 / 16.99 | 52.95 | 26.42 | 2.165 | Large attention win, but selector/dense-score path is now dominant. |
| stride1 torch-matmul score reuse | `50302159` | 8192 | 100.0 | 54.47 | 37.09 / 17.37 | 9.37 | 26.47 | 3.909 | Fastest ctx8192 smoke so far, but it is a runtime-enabling backend with worse modeled MB. |
| stride1 torch-matmul score reuse, n=4 | `50302872` | 8192 | 100.0 | 50.03 | 32.84 / 17.19 | 33.21 total | 105.63 total | 3.908 | Clean multi-sample run; still `5.25x` slower than dense ctx8192 n=4. |
| stride8 top-k + score-only tail reuse | `50301052` | 8192 | 100.0 | 106.87 | 88.63 / 18.23 | 57.93 | 26.84 | 2.385 | Rejected: anchor top-k amortization does not offset the full per-query score pass. |

Interpretation: score reuse is a real runtime improvement for strict stride1 prefill, cutting ctx8192 one-sample native attention time by `3.05x` (`80.63s` to `26.42s`) and total sample time by `1.63x` (`154.64s` to `94.85s`). The explicit torch-matmul PQ selector backend cuts ctx8192 n=4 total time to `50.03s/sample`, but it reconstructs/reads PQ-approximated K and therefore worsens modeled selector MB. Treat it as a benchmark-enabling runtime path, not as evidence for algorithmic memory efficiency. It is still not comfortable for full RULER/LongBench: dense ctx8192 n=4 is `9.54s/sample`, so this path is `5.25x` slower.

### HF/RULER Geometric Confidence Wiring 2026-05-15

Implemented the deployable confidence-rule controls in the HF/RULER path:

- New HF/RULER args: `--online_confidence_rule geometric_probe_tail_switch`, `--tail_score_calibration`, `--tail_proxy_mass_min/max`, `--tail_pq_corr_min`, `--tail_pq_relrmse_max`, and geometric budget controls.
- Non-`none` confidence rules explicitly disable fixed-budget native fast paths, so the runner does not silently ignore the confidence rule.
- Added selected-V exact-all safety gate controls: `--selected_value_exact_all_context_max` and `--selected_value_exact_all_fraction_min`.
- Cost accounting now has a separate `mean_confidence_MB_per_head_query` field instead of burying confidence probes inside selector MB.

Correctness/runtime smokes:

| run | Slurm | context | layers | score | stream seconds | prefill / decode seconds | total MB/head-query | selector / exactKV / tail / confidence MB | interpretation |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| geometric confidence smoke | `50302892` | 1024 | `16` | 100.0 | 98.19 | 75.60 / 22.59 | 0.414 | 0.098 / 0.156 / 0.005 / 0.154 | Functional end-to-end wiring, but far too slow unfused. |
| native prefill confidence | `50302998` | 1024 | `16` | 100.0 | 35.54 | 6.84 / 28.70 | 1.196 | 0.162 / 0.164 / 0.006 / 0.864 | Batched prefill confidence works; decode still fell back slow. |
| native prefill+decode confidence | `50303049` | 1024 | `16` | 100.0 | 16.18 | 10.42 / 5.76 | 1.259 | 0.162 / 0.167 / 0.006 / 0.924 | Native confidence path works for one layer; decode bottleneck is removed for this smoke. |
| native prefill+decode confidence | `50303084` | 1024 | all | 100.0 | 56.26 | 9.07 / 47.19 | 1.274 | 0.162 / 0.167 / 0.006 / 0.940 | All-layer path is functional, but decode attention/confidence overhead dominates. |
| native prefill+decode confidence | `50303085` | 4096 | `16` | 100.0 | 12.68 | 4.85 / 7.83 | 3.369 | 1.535 / 0.212 / 0.050 / 1.572 | Longer-context one-layer path is functional; selector and confidence MB rise as expected. |
| native confidence, no profiling | `50303130` | 1024 | all | 100.0 | 57.29 | 10.60 / 46.69 | 1.275 | 0.162 / 0.167 / 0.006 / 0.940 | Profiling sync was not the runtime problem. |
| native confidence, profiled | `50303126` | 4096 | all | 100.0 | 164.22 | 73.36 / 90.86 | 3.369 | 1.535 / 0.212 / 0.050 / 1.572 | Profiled bottleneck check; native attention/confidence dominates. |
| native confidence, no profiling | `50303131` | 4096 | all | 100.0 | 148.33 | 71.19 / 77.13 | 3.369 | 1.535 / 0.212 / 0.050 / 1.572 | No-profile runtime is still too slow for comfortable benchmark sweeps. |
| PQ-proxy-mass budget | `50303193` | 1024 | all | 100.0 | 30.97 | 8.91 / 22.07 | 0.332 | 0.162 / 0.163 / 0.006 / 0.001 | Cheaper confidence works; removes repeated attention-pass overhead. |
| PQ-proxy-mass budget | `50303194` | 4096 | all | 100.0 | 51.10 | 28.92 / 22.18 | 1.802 | 1.535 / 0.211 / 0.050 / 0.006 | Much faster than exact-probe confidence but still not clearly benchmark-comfortable. |
| PQ-proxy-mass budget, no profiling | `50303220` | 4096 | all | 100.0 | 47.52 | 25.85 / 21.67 | 1.802 | 1.535 / 0.211 / 0.050 / 0.006 | Profiling overhead is modest; proxy is still `5.6x` dense on this smoke. |

Conclusion: the confidence rule is now connected to real-model benchmark execution, and the exact-probe L2 confidence subset has a native prefill+decode implementation. The one-layer ctx1024 smoke improved from `98.19s` to `16.18s`, but this is not yet full benchmark readiness. All-layer ctx1024 is still `56.26s/sample` versus a dense ctx1024 reference of roughly `4.92s/sample`; native attention/confidence calls, not selector time, dominate. The fast path also currently supports fixed selected-V exact-top and the exact-probe L2 gate, while the trace-frontier recommendation used selected-mass/exact-all safety and proxy/correlation gates. The immediate gate is reducing confidence/attention overhead or explicitly labeling this as an expensive correctness path.

New cheaper confidence variant under test: `pq_proxy_mass_budget`. It uses no dense/oracle attention. It computes a per-query/head budget from PQ approximate score mass (`TAIL_PROXY_MASS_MIN=0.99`) and then runs selected/tail attention once. It is much faster than exact-probe confidence (`148.33s -> 47.52s` at ctx4096 all layers, no profiling), but it is a weaker confidence signal and must be validated numerically before being promoted.

Trace validation for the proxy rule:

| run | Slurm | context point | proxy target | attn-concat relL2 | layer-output relL2 | mean / min attention mass | mean step MB/head | selected K tokens/head | interpretation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Proxy denominator over ranked prefix | `50303255` | layer16, one long-context query | 0.99 | 0.001062 | 0.000388 | 0.9975 / 0.9877 | 18.23 | 47,321 | Numerically strong, but denominator was optimistic if rank budget omitted low-score tokens. |
| Proxy denominator over full ranked dynamic set | `50303296` | layer16, one long-context query | 0.99 | 0.001011 | 0.000372 | 0.9980 / 0.9913 | 18.71 | 48,825 | Honest trace check; quality is excellent but cost is high. |

Next trace/runtime sweep: lower proxy targets (`0.95`, `0.97`) to create quality/cost points for downstream task runs.

Proxy target sweep, ctx4096 RULER smoke plus full-denominator layer-16 trace:

| proxy target | RULER score | RULER seconds | RULER MB/head-query | RULER selected tokens | trace attn relL2 | trace layer relL2 | trace mean/min mass | trace MB/head | trace selected K |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.95 | 100.0 | 47.87 | 1.800 | 521 | 0.00535 | 0.00210 | 0.9907 / 0.9610 | 13.28 | 30,713 |
| 0.97 | 100.0 | 47.67 | 1.801 | 525 | 0.00373 | 0.00148 | 0.9944 / 0.9745 | 15.24 | 36,857 |
| 0.99 | 100.0 | 47.52 | 1.802 | 529 | 0.00101 | 0.00037 | 0.9980 / 0.9913 | 18.71 | 48,825 |

Interpretation: lower proxy targets create useful trace-quality/cost points, but the ctx4096 RULER runtime barely changes because this configuration is dominated by fixed selector/tail machinery and static/base tokens rather than the small difference in selected budget.

Ctx8192 proxy confidence scaling:

| run | score | seconds | prefill / decode seconds | MB/head-query | selected tokens | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dense reference, n=4 | 100.0 | 9.54 | 1.69 / 7.84 | n/a | n/a | Dense runtime reference. |
| Fixed matmul path, n=4 | 100.0 | 50.03 | 32.84 / 17.19 | 3.908 | 374 | Current fastest 8k sparse smoke. |
| Proxy target 0.97, max64 | 100.0 | 69.73 | 51.90 / 17.83 | 3.909 | 374 | Same modeled cost/selection as fixed, but proxy confidence computation adds runtime. |
| Proxy target 0.97, max128 | 100.0 | 75.54 | 56.87 / 18.67 | 3.924 | 434 | Larger cap increases selected work and runtime. |
| Proxy target 0.97, max256 | 100.0 | 105.97 | 83.57 / 22.40 | 3.952 | 551 | Too slow; larger top-k budget dominates. |

Conclusion: `pq_proxy_mass_budget` is numerically credible and removes exact-probe overhead, but the current PyTorch/HF implementation is not benchmark-comfortable at 8k. The proxy denominator/top-k budget logic itself adds about `20s/sample` at max64. For near-term downstream experiments, either use the fixed-budget CUDA path as an explicitly labeled runtime-feasible quality tier, or implement a cheaper confidence rule that does not require a full proxy softmax denominator over the dense PQ score matrix.

Cheaper confidence rule under test: `pq_ranked_mass_budget`.

- It uses only the already-ranked PQ candidate scores and normalizes within the top `geometric_max_budget` candidates.
- It is deployable and cheap, but it is not a true full-tail proxy mass. Treat it as a ranked-candidate concentration/confidence rule.
- Pending checks:
  - RULER ctx8192 all-layer smoke: Slurm `50303491`, `ruler_eval_result/frontier_readiness_20260515/frontier_ranked_mass0p97_ctx8192_all`.
  - Saved trace layer-16 relL2/cosine check: Slurm `50303492`, `attention_efficiency_result/frontier_readiness_20260515_trace/trace_ranked_mass0p97_l16_q1`.

Stride-compatible score reuse was also tested. With `PREFILL_SELECTOR_STRIDE=8`, the top-k list still comes from anchor queries, while dense PQ scores are recomputed for every actual query only for tail reuse. This is deployable and not anchor-score cheating, but it was slower than stride1 score reuse because it pays both the anchor top-k pass and the full score-only pass. Do not promote this path.

Next active optimization target: prefill attention kernel quality. Selector amortization is promising on the one-sample 4k smoke. The original fused causal PQ top-k selector was not useful because it only covered very small budgets and effectively missed the real `k=64` path.

| selector variant | Slurm | context | score | stream seconds | prefill / decode seconds | native selector seconds | native attention seconds | mean MB/head-query | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline per-query selector | `50243977` | 4096 | 100.0 | 46.55 | 30.57 / 15.98 | 8.81 | 24.33 | 1.008 | Reference for the optimized prefill+decode tail path. |
| Existing fused selector | `50244001` | 4096 | 100.0 | 48.04 | 31.09 / 16.95 | 8.50 | 24.30 | 1.008 | Rejected for now: tiny selector improvement, worse wall time. |
| Selector stride 4 | `50244113` | 4096 | 100.0 | 44.49 | 27.80 / 16.69 | 5.29 | 24.31 | 0.460 | Positive scheduling ablation; selector traffic/time drops. |
| Selector stride 8 | `50244114` | 4096 | 100.0 | 42.14 | 25.66 / 16.48 | 2.94 | 24.16 | 0.369 | Best 4k smoke so far; needs multi-sample and harder-task validation because it reuses selections across nearby prefill queries. |

Current fused-selector implementation status:

- Implemented a real `k<=64` fused causal GQA page-PQ selector path instead of falling back to the non-fused score-matrix + `at::topk` implementation.
- Added a second smallscan fused kernel for `total_tokens <= 4096` that keeps candidate scores in shared memory. Dispatch is conservative: use it only when `page_size >= 1024`; otherwise `k>16` falls back to the non-fused score-matrix + `at::topk` path to avoid making small-page selectors slower.
- Added Slurm-only build/test wrapper `scripts/run_cuda_ext_fullscan_test.sh` with a shared repo lock, plus `benchmark/selector_eval/cuda_ext/bench_fused_selector.py` for native-vs-fused selector timing.
- Build-only Slurm validation passed: `50264910`.
- CUDA parity on a non-target V100 run passed for the new `budget=64` fused path, but V100 timing is not a paper/result source. Use only `spgpu`/A40 timing for decisions.
- Representative forced A40 selector microbenchmark completed on `spgpu`: Slurm `50295884`, elapsed `00:01:28`, parity passed. This run explicitly compared `auto`, forced `smallscan`, and forced `localtopk`; the earlier auto-only small-page results were not evidence for fused speed because auto intentionally fell back to native for slow regimes.

Forced A40 selector timing, `positions=128`, `heads=32`, `kv_heads=8`, `dim=128`, total indexed tokens fixed at `4096`:

| pages x page | budget | native ms | auto ms | forced smallscan ms | forced localtopk ms | best valid path | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `16x256` | 8 | 15.42 | 24.51 | 27.19 | 24.75 | native | Real fused kernels are slower for small pages. |
| `16x256` | 16 | 15.44 | 19.78 | 28.76 | 19.78 | native | Auto/localtopk slower; native fallback is right. |
| `16x256` | 32 | 15.42 | 15.41 | 29.45 | 27.89 | native/auto | Auto is native fallback; forced fused is bad. |
| `16x256` | 64 | 15.45 | 15.45 | 30.68 | 56.28 | native/auto | Real `k=64` fused is much slower here. |
| `8x512` | 8 | 15.43 | 12.14 | 14.37 | 12.19 | auto/localtopk | Small-budget localtopk helps. |
| `8x512` | 16 | 15.47 | 10.94 | 14.51 | 10.94 | auto/localtopk | Small-budget localtopk helps. |
| `8x512` | 32 | 15.43 | 15.43 | 14.99 | 18.21 | smallscan | Smallscan gives only a small win. |
| `8x512` | 64 | 15.47 | 15.47 | 15.82 | 40.37 | native/auto | Native fallback is right for `k=64`. |
| `4x1024` | 8 | 15.43 | 7.20 | 7.20 | 6.24 | localtopk | Fused helps once page-LUT reuse is larger. |
| `4x1024` | 16 | 15.48 | 7.31 | 7.32 | 6.89 | localtopk | Fused helps. |
| `4x1024` | 32 | 15.43 | 7.56 | 7.56 | 13.01 | auto/smallscan | Smallscan is the right path. |
| `4x1024` | 64 | 15.48 | 9.22 | 9.22 | 33.72 | auto/smallscan | Smallscan gives `1.68x`. |
| `2x2048` | 8 | 15.43 | 3.74 | 3.74 | 3.27 | localtopk | Fused strongly helps. |
| `2x2048` | 16 | 15.48 | 3.95 | 3.95 | 4.73 | auto/smallscan | Fused strongly helps. |
| `2x2048` | 32 | 15.44 | 4.72 | 4.71 | 10.50 | smallscan | Fused strongly helps. |
| `2x2048` | 64 | 15.48 | 7.35 | 7.35 | 30.98 | smallscan | Fused gives `2.11x`. |

Conclusion: fused selector is useful for large pages and small/medium budgets, but it is not a universal replacement for native score-matrix + `at::topk`. The conservative dispatcher is justified for page-256/page-512 `k=64`. This selector kernel work does not by itself make long prefill runtime-ready; the next readiness gate is full-algorithm GPU runtime and quality on benchmark-style RULER/LongBench runs.

Latest GPU implementation checkpoint:

| path | evidence | score | stream seconds | prefill / decode seconds | mean MB/head-query | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Dense HF baseline | `dense_batched_ctx1024_n1_v2` | 100.0 | 4.917 | 0.500 / 4.418 | n/a | Reference for active-selector runtime. |
| Exact selected, raw bf16 K/V | Slurm `50192714`, `pagedpq_exact_rawbf16_ctx1024_n1_all_layers_bulkacct` | 100.0 | 9.463 | 1.506 / 7.958 | 0.246 | Current exact-selected functional baseline; no full-cache fp32 K/V cast. |
| Exact selected, vectorized all-head decode | Slurm `50207566`, `pagedpq_exact_rawbf16_vecdecode_ctx1024_n1_all_layers` | 100.0 | 8.542 | 1.744 / 6.798 | 0.246 | Current exact-selected runtime frontier; removes the remaining per-KV Python loop after GQA selector. |
| Exact selected, vectorized all-head decode | Slurm `50210243`, `pagedpq_exact_vec_ctx2048_n2_ps128` | 100.0 | 11.918 | 5.014 / 6.904 | 0.520 | Higher-context smoke completed cleanly on two samples; score copied from RULER CSV after wrapper summary fix. |
| Exact selected, vectorized all-head decode | Slurm `50210244`, `pagedpq_exact_vec_ctx4096_n1_ps256` | 100.0 | 18.293 | 11.255 / 7.038 | 0.587 | Higher-context smoke completed cleanly; page size 256 keeps selector traffic lower than earlier ps128 runs. |
| Dense HF baseline | Slurm `50212070`, `dense_batched_ctx2048_n2_veccompare` | 100.0 | 6.702 | 1.665 / 5.037 | n/a | Fresh dense comparison: active exact-selected is `1.78x` slower at ctx2048 despite lower modeled MB. |
| Dense HF baseline | Slurm `50212072`, `dense_batched_ctx4096_n1_veccompare` | 100.0 | 7.042 | 1.635 / 5.407 | n/a | Fresh dense comparison: active exact-selected is `2.60x` slower at ctx4096. |
| Exact selected, vectorized all-head decode, profiled | Slurm `50212623`, `pagedpq_exact_vec_ctx4096_n1_ps256_prof` | 100.0 | 22.214 | 14.193 / 8.020 | 0.587 | Profile only: patched attention `16.12s`, native selector `6.84s`, native attention `4.79s`; synchronized profiling inflates wall time but shows launch/selector/selected-attention cost dominates, not page-PQ build. |
| Exact selected, LUT prefill selector | Slurm `50217384`, `pagedpq_exact_lut_ctx4096_ps256_b64` | 100.0 | 17.309 | 10.316 / 6.993 | 0.587 | PQ LUT scoring preserves quality and gives a small ctx4096 speedup versus native scalar selector (`18.293 -> 17.309s`). |
| Exact selected, warp-tiled selected-attention + LUT selector | Slurm `50232710`, `pagedpq_warptiled_lut_ctx4096_ps256_b64` | 100.0 | 14.545 | 7.590 / 6.955 | 0.587 | Warp-tiled selected attention improves ctx4096 substantially with no cost or score change. |
| FlashInfer token-block sparse prefill | Slurm `50226126`, `pagedpq_flashinfer_token_ctx4096_ps256_b64` | 100.0 | 18.661 | 11.445 / 7.216 | 0.770 | Negative: token-level block sparse is slower than the native/LUT path and reads more KV after unioning tokens across heads. |
| FlashInfer page-block sparse prefill | Slurm `50228804`, `pagedpq_flashinfer_page_ctx4096_ps256_b64` | 100.0 | 16.743 | 9.390 / 7.353 | 1.320 | Partial positive: faster than native/LUT at ctx4096, but exact-KV traffic jumps to `0.938 MB/head-query`. |
| Dense HF baseline | Slurm `50212625`, `dense_batched_ctx8192_n1_veccompare` | 100.0 | 9.915 | 2.815 / 7.101 | n/a | Fresh dense comparison for ctx8192. |
| Exact selected, vectorized all-head decode | Slurm `50212624`, `pagedpq_exact_vec_ctx8192_n1_ps512` | 100.0 | 45.417 | 37.468 / 7.948 | 0.713 | Negative runtime scaling result: score is preserved, but sparse prefill is far slower than dense; page-PQ build is only `0.98s`. |
| Exact selected, chunked prefill | Slurm `50213928`, `pagedpq_exact_chunk_ctx8192_n1_ps512_c1024_b` | 100.0 | 41.133 | 33.354 / 7.780 | 0.713 | Chunking avoids some future-page scoring and improves ctx8192 by ~9%, but remains far from dense. |
| Exact selected, lower budget | Slurm `50216881`, `pagedpq_exact_ctx8192_ps512_b32` | 100.0 | 43.024 | 35.330 / 7.694 | 0.699 | Budget 32 lowers exact KV work slightly; selector floor remains dominant enough that runtime improves only modestly. |
| Exact selected, LUT prefill selector | Slurm `50217385`, `pagedpq_exact_lut_ctx8192_ps512_b32` | 100.0 | 38.336 | 30.341 / 7.994 | 0.699 | Best ctx8192 exact-selected runtime so far, but still `3.87x` dense due sparse prefill overhead. |
| Exact selected, warp-tiled selected-attention + LUT selector | Slurm `50232711`, `pagedpq_warptiled_lut_ctx8192_ps512_b32` | 100.0 | 30.566 | 22.857 / 7.709 | 0.699 | Current viable ctx8192 runtime frontier; warp-tiled QK logits remove a large selected-attention bottleneck. |
| Exact selected, warp-tiled selected-attention + LUT selector | Slurm `50233352`, `pagedpq_warptiled_lut_ctx8192_ps512_b8` | 100.0 | 30.338 | 22.692 / 7.646 | 0.688 | Budget 8 preserves this smoke's score and is slightly faster/lower-MB than budget 32; needs multi-sample validation. |
| Exact selected, warp-tiled selected-attention + LUT selector | Slurm `50234006`, `pagedpq_warptiled_lut_ctx8192_ps512_b8_n4` | 100.0 | 29.602 | 21.828 / 7.774 | 0.688 | Four-sample validation preserves score, but runtime remains far from dense. |
| Dense HF baseline | Slurm `50234007`, `dense_batched_ctx8192_n4_veccompare` | 100.0 | 8.335 | 0.731 / 7.604 | n/a | Four-sample dense comparison; active frontier is still `3.55x` slower. |
| Exact selected, warp-tiled selected-attention + LUT selector | Slurm `50233351`, `pagedpq_warptiled_lut_ctx8192_ps512_b0` | 0.0 | 16.321 | 9.281 / 7.040 | 0.685* | Runtime floor if indexed retrieval is removed; score fails. `*` historical row overcharged selector MB before the budget-zero accounting fix. |
| Exact selected, LUT prefill selector, larger page | Slurm `50219148`, `pagedpq_exact_lut_ctx8192_ps1024_b32` | 100.0 | 39.638 | 31.635 / 8.003 | 0.583 | Lower modeled MB than ps512, but slower because exact KV work grows with larger pending pages. |
| FlashInfer page-block sparse prefill | Slurm `50231041`, `pagedpq_flashinfer_page_ctx8192_ps512_b32` | 0.0 | 35.688 | 27.599 / 8.088 | 2.462 | Rejected: slightly faster than token-selected LUT but quality collapses and KV traffic is too high. |
| FlashInfer page-block sparse prefill | Slurm `50231767`, `pagedpq_flashinfer_page_ctx8192_ps128_b32` | 100.0 | 42.826 | 34.448 / 8.379 | 3.613 | Rejected: smaller blocks recover quality, but runtime and modeled MB are worse than token-selected LUT. |
| Selected V-PQ before native decode | Slurm `50194735`, `pagedpq_vpqv6_noexactv_ctx1024_n1_all_layers` | 100.0 | 48.143 | 3.346 / 44.796 | 0.262 | Negative control: GPU helpers worked, but Python/PyTorch per-KV decode dominated. |
| Selected V-PQ native decode | Slurm `50201343`, `pagedpq_vpqv6_native_decode2_prof_ctx1024_n1_all_layers` | 100.0 | 12.811 | 2.211 / 10.600 | 0.262 | Native GQA V-PQ decode path works and removes the huge Python decode bottleneck, but remains slower/costlier than exact selected. |
| Selected V-PQ all-compressed, 6-bit page-local V | Slurm `50241551`, `pagedpq_selectedvpq_warp_ctx2048_ps128_b8` | 0.0 | 14.730 | 4.361 / 10.337 | 0.583 | Native prefill+decode path runs, but 6-bit all-compressed selected V is too lossy for this NIAH sample. |
| Selected V-PQ all-compressed, 7-bit page-local V | Slurm `50242084`, `pagedpq_selectedvpq_warp_ctx2048_ps128_b8_v7_noexactload` | 100.0 | 12.897 | 4.397 / 8.491 | 0.670 | Quality recovers at 7 bits, but ps128 value-codebook traffic makes MB worse than exact selected. |
| Selected V-PQ all-compressed, 8-bit page-local V | Slurm `50241949`, `pagedpq_selectedvpq_warp_ctx2048_ps128_b8_v8_noexactload` | 100.0 | 16.615 | 8.090 / 8.516 | 0.843 | Quality recovers, but modeled MB/runtime are worse than exact selected because page-local V codebooks dominate value-side traffic. |
| Exact selected ps256/budget8 control | Slurm `50242187`, `pagedpq_exact_ctx2048_ps256_b8_current` | 100.0 | 16.398 | 8.594 / 7.795 | 0.336 | Apples-to-apples exact-V control for ps256/budget8. Lower modeled MB than V-PQ, but slower in this one-sample profiled smoke. |
| Selected V-PQ all-compressed, 7-bit page-local V, ps256 | Slurm `50242136`, `pagedpq_selectedvpq_warp_ctx2048_ps256_b8_v7_noexactload` | 100.0 | 13.451 | 4.928 / 8.513 | 0.415 | Best page-local V-PQ smoke so far: faster than the ps256 exact-V control, but still higher modeled MB. |
| Selected V-PQ all-compressed, 7-bit page-local V, ps512 | Slurm `50242279`, `pagedpq_selectedvpq_warp_ctx2048_ps512_b8_v7_noexactload` | 0.0 | 14.256 | 5.583 / 8.664 | 0.324 | Lower modeled MB from fewer pages, but quality collapses. Not usable. |
| Selected V-PQ grouped values, K ps128, V group 2, 7-bit | Slurm `50242721`, `pagedpq_selectedvpq_warp_ctx2048_kps128_vg2_b8_v7_mem64` | 100.0 | 15.991 | 6.678 / 9.302 | 0.576 | Grouped value codebooks reduce MB versus ps128 page-local V-PQ, but are slower and still worse than ps256 V-PQ. |
| Selected V-PQ grouped values, K ps128, V group 4, 7-bit | Slurm `50242722`, `pagedpq_selectedvpq_warp_ctx2048_kps128_vg4_b8_v7_mem64` | 0.0 | 16.053 | 7.200 / 8.844 | 0.530 | More grouping lowers MB but quality collapses. Not usable. |
| Selected V-PQ grouped values, K ps256, V group 2, 7-bit | Slurm `50242774`, `pagedpq_selectedvpq_warp_ctx2048_kps256_vg2_b8_v7_mem64` | 0.0 | 13.828 | 4.649 / 9.170 | 0.369 | Lower modeled MB than page-local ps256 V-PQ, but task quality collapses. Grouped all-compressed selected V is not a safe frontier. |
| Native V-PQ tail, one patched layer | Slurm `50194736`, `pagedpq_native_tail_l0_inf_ctx1024_n1` | 100.0 | 9.945 | 3.083 / 6.862 | 0.271 | Functional coverage for compressed-tail GPU path; not a runtime frontier. |
| Native V-PQ tail, all layers | Slurm `50208187`, `pagedpq_native_tail_all_inf_ctx1024_n1` | 100.0 | 68.608 | 15.705 / 52.903 | 0.271 | All-layer native tail integration works but is far too slow; tail remains an opt-in correctness path, not the active runtime path. |

Latest correctness/build evidence:

- Slurm `50201342`: updated CUDA extension test passed, including bf16/fp16 K/V coverage for GQA exact selected attention and GQA V-PQ selected attention.
- Slurm `50241891`: updated CUDA extension test passed after changing the selected-V-PQ kernel to avoid loading exact V before overwriting it with a PQ-reconstructed value for sealed-page tokens.
- Slurm `50187756`: raw bf16/fp16 exact-selected attention parity passed before RULER integration.
- The attempted decode shortcut through `gqa_causal_exact_selected_attention` was rejected: it preserved quality/cost but increased native attention time and worsened stream runtime.
- Current bottleneck is still HF patched-forward overhead plus selector/selected-attention launch structure, not page-PQ build. In the ctx4096 profiled smoke before the warp-tiled kernel, patched attention was `16.12s`, native selector `6.84s`, native attention `4.79s`, and page-PQ index build only `0.32s`. Chunking and PQ LUT scoring improved ctx8192 from `45.42s` to `38.34s`; the warp-tiled selected-attention kernel further improved it to `30.34s` at budget 8. Sparse prefill is still much slower than dense (`9.92s`). FlashInfer block-sparse variants did not solve this: token blocks were slower, coarse page blocks failed quality, and smaller page blocks recovered quality only by increasing modeled traffic and runtime.

Native selector backend update:

| item | status | evidence |
| --- | --- | --- |
| CUDA fullscan PQ top-k extension | passed | Slurm `50148302`: `test_fullscan_pq_topk.py` matched torch reference for uint8/int64 codes and multiple budgets. |
| CUDA fullscan PQ score-matrix API | passed | Slurm `50149250`: updated `test_fullscan_pq_topk.py` matched torch reference for top-k tokens, top-k scores, and full approximate PQ scores. |
| CUDA exact selected-attention API | passed | Slurm `50152505`: `test_fullscan_pq_topk.py` matched torch reference for exact selected-token attention. |
| CUDA GQA exact-attention + selector APIs | passed | Slurm `50153406` and `50156657`: GQA selected attention and GQA fullscan PQ top-k matched torch references. |
| CUDA causal prefill GQA selector + attention APIs | passed | Slurm `50161439`: causal batched prefill selector and exact selected-token attention matched torch references for uint8/int64 PQ codes and multiple budgets. |
| Device-side fullscan page-PQ builder | passed | Slurm `50171329`: `build_page_pq_torch` plus CUDA selector matched dense QK scores/top-k for the lossless `page_size <= centroids` regime. |
| CUDA causal prefill V-PQ selected attention | passed | Slurm `50176993`: `gqa_causal_vpq_selected_attention` matched a torch reference for uint8/int64 V-PQ codes and multiple budgets. |
| CUDA causal prefill V-PQ compressed-tail attention | passed | Slurm `50177731`: `gqa_causal_vpq_tail_attention` matched a torch reference for uint8/int64 K/V-PQ codes and multiple budgets. |
| CUDA decode V-PQ tail from precomputed selector scores | passed | Slurm `50182030`: `gqa_decode_vpq_tail_from_scores` matched the torch reference and avoids recomputing K-PQ tail logits inside the output loop. |
| CUDA GQA fullscan PQ top-k + score matrix | passed | Slurm `50182126`: `gqa_fullscan_pq_topk_scores` matched per-KV-head torch references for top-k tokens, top-k scores, and dense approximate scores. |
| Selector-eval native vs torch parity | passed | Same trace/config at decode 500, head 0, budget 64: identical mass/L2/MB; selector time `0.00012s` native vs `0.00918s` torch page-loop after warmup. |
| GPU V-PQ reconstruction helper | passed | Slurm `50148710`: `test_gpu_vpq_helpers.py` matched CPU/reference V-PQ reconstruction. |
| GPU all-token V-PQ reconstruction helper | passed | Slurm `50149240`: extended helper matched CPU/reference all-token V-PQ reconstruction for compressed-tail use. |
| Online page append parity | passed | Slurm `50148940`: append-built pages match snapshot full-build pages with the same page-id seed offset. |
| HF/RULER native selector smoke | passed | Slurm `50148451`: `niah_single_1`, context 512, one patched layer, 128 generated tokens, score `100.0`; generation `21.2s`, mean step `0.108 MB/head-query`. |

Current native-backend scope:

- Implemented: batched CUDA fullscan PQ selector for `[heads, dim]` queries.
- Implemented: GQA-wide CUDA fullscan PQ selector for `[num_heads, dim]` queries and per-KV-head page-PQ codebooks.
- Implemented: CUDA fullscan PQ score-matrix return for compressed-tail logic.
- Implemented: batched exact selected-token attention fast path for decode when `selector_backend=cuda_ext`, `selector_mode=fullscan`, `tail_blend=0`, no rerank, no per-head budget map.
- Implemented: GQA-wide exact selected-token attention fast path for decode.
- Implemented: raw fp16/bf16 K/V support for GQA exact selected-token attention and GQA V-PQ selected-token attention; exact selected no longer needs a full-cache fp32 K/V cast.
- Implemented: causal batched GQA fullscan PQ selector and causal exact selected-token attention for prefill positions.
- Implemented: HF/RULER `pagedpq_batched` exact-selected fast path using one native causal selector call and one native causal selected-attention call per patched layer, instead of Python per-query/per-head loops.
- Implemented: `INDEX_BUILD_BACKEND=torch_gpu`, a fullscan page-PQ builder that keeps sealed-page codebook/code construction on device and avoids the CPU NumPy K round-trip.
- Implemented: causal prefill selected-V compression path for `selected_value_mode=vpq_value`, `selected_value_exact_rule=fixed`, exact top/min/max all zero, and `INDEX_BUILD_BACKEND=torch_gpu`. It keeps K exact for selected logits, reconstructs selected V from page-local V-PQ on GPU for sealed-page tokens, and keeps static/pending base tokens exact.
- Implemented: native GQA selected-V-PQ decode path for the all-compressed selected-V case, avoiding the previous Python/PyTorch per-KV decode loop.
- Implemented: vectorized exact-selected decode path that constructs the all-head selected token tensor once after the GQA selector and bypasses the remaining per-KV Python loop.
- Implemented: GPU selected V-PQ reconstruction helper and cost accounting support for the fast path.
- Implemented: native prefill V-PQ compressed-tail fast path for `selector_backend=cuda_ext`, exact selected V, `tail_mode=vpq_value`, `INDEX_BUILD_BACKEND=torch_gpu`, and no probe gate.
- Experimental: native decode V-PQ compressed-tail path gated by `NATIVE_DECODE_TAIL=1`. The score-reuse kernel is correct, but current runtime is not yet a frontier, so it stays opt-in.
- Still incomplete: all active confidence rules without fallback Python per-head logic, GQA-native selected V-PQ/tail for prefill, and online page refresh/rebalance beyond append-only CPU/NumPy page sealing.

Scope:

- Task: `niah_single_1`, context setting `512`, one sample, 128 generated tokens.
- Mode: prefill+decode approximation through HF Llama attention patching, not decode-only.
- Model: local `Llama-3.1-8B-Instruct` snapshot.
- Stress setting: `page_size=128`, `budget=256`, `selector_mode=fullscan`, layer 16 or all layers.
- Note: the first streaming smokes before the pending-page fix are now historical only; online pending-page tokens must remain exact until a page seals.

Current smoke results:

| run | layers patched | score | generation seconds | prefill seconds | decode seconds | approx calls | passthrough calls | mean step MB/head-query | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dense_batched_ctx512_n1_reuse` | 0 | 100.0 | 6.028 | 1.737 | 4.291 | n/a | n/a | n/a | dense HF reference; model load was 73.1s and is excluded from generation time |
| `dense_batched_ctx1024_n1_g16` | 0 | 100.0 | 6.053 | 5.390 | 0.663 | n/a | n/a | n/a | same-context dense batched baseline for the causal prefill smoke |
| `dense_stream_ctx512_n1` | 0 | 100.0 | 15.033 | 10.799 | 4.235 | n/a | n/a | n/a | apples-to-apples dense streaming baseline for native selector smoke |
| `native_cudaext_l0_ctx512_n1` | 1 | 100.0 | 21.179 | 15.338 | 5.840 | 66 | 383 | 0.108 | native CUDA fullscan selector + batched exact selected-token attention for one patched layer; functional but still above dense runtime |
| `native_cudaext_all_ctx512_n1_g128` | 32 | 100.0 | 35.625 | 17.449 | 18.176 | 2112 | 12256 | 0.108 | all-layer active native path; much faster than old Python active path but still `2.4x` dense stream |
| `native_cudaext_all_exact_ctx512_n1_g128_b64_nosync` | 32 | 100.0 | 38.801 | 15.657 | 23.144 | 2112 | 12256 | 0.115 | active exact-selected baseline after stats CPU-sync removal; still had selector timing sync and PyTorch selected attention |
| `native_cudaext_all_exact_ctx512_n1_g128_b64_fusedattn` | 32 | 100.0 | 34.601 | 15.480 | 19.120 | 2112 | 12256 | 0.115 | one-KV-group fused exact selected-attention op; improves decode but launch count remains high |
| `native_cudaext_all_exact_ctx512_n1_g128_b64_fusedattn_nosync2` | 32 | 100.0 | 29.390 | 14.111 | 15.279 | 2112 | 12256 | 0.115 | disabled selector timing sync in HF generation; largest runtime fix in this round |
| `native_cudaext_all_exact_ctx512_n1_g128_b64_gqa_fused` | 32 | 100.0 | 28.132 | 14.091 | 14.041 | 2112 | 12256 | 0.115 | one native GQA selected-attention call per layer; modest additional gain |
| `native_cudaext_all_exact_ctx512_n1_g128_b64_gqa_selector` | 32 | 100.0 | 27.813 | 14.146 | 13.667 | 2112 | 12256 | 0.115 | one native GQA selector call plus one GQA exact-attention call per layer; current active exact-selected runtime frontier |
| `native_cudaext_all_vpq_ctx512_n1_g128` | 32 | 100.0 | 51.835 | 16.481 | 35.354 | 2112 | 12256 | 0.106 | selected V-PQ fast path works functionally but is slower and saves little at context 512; not current runtime frontier |
| `native_cudaext_all_vpqtail_ctx512_n1_g128_b32_mem64` | 32 | 100.0 | 58.339 | 20.803 | 37.535 | 2112 | 12256 | 0.113 | native V-PQ compressed-tail path is active (`tail MB/head-query=0.00059`, `tail samples=14.1`) but slower than exact-selected; use as correctness evidence, not a runtime frontier |
| `native_cudaext_all_vpqtail_ctx512_n1_g128_b32_nosync` | 32 | 100.0 | 46.561 | 15.422 | 31.139 | 2112 | 12256 | 0.113 | same compressed-tail path after stats CPU-sync fix; still slower than exact-selected |
| `pagedpq_batched_prefill_decode_ctx1024_n1_b64_m32` | 32 | 100.0 | 44.779 | 43.624 | 1.156 | 544 | 0 | 0.217 | first all-layer causal prefill+decode approximation smoke; native causal prefill path functional but CPU/NumPy page-PQ build dominated |
| `pagedpq_batched_prefill_decode_ctx1024_n1_b64_m32_buildtiming` | 32 | 100.0 | 51.869 | n/a | n/a | 544 | 0 | 0.217 | build accounting run: page-PQ index build took `37.413s`, read `82.1 MB`, wrote `64.5 MB`; confirms index construction was the bottleneck |
| `pagedpq_batched_prefill_decode_ctx1024_n1_b64_m32_torchbuild` | 32 | 100.0 | 4.457 | n/a | n/a | 544 | 0 | 0.217 | same config with `INDEX_BUILD_BACKEND=torch_gpu`; index build dropped to `1.206s`, read `32.0 MB`, wrote `64.5 MB`; faster than same-context dense smoke |
| `pagedpq_batched_prefill_decode_ctx1024_n1_b64_m32_torchbuild_vpqv` | 32 | 0.0 | 10.681 | 3.359 | 7.322 | 544 | 0 | 0.213 | selected V-PQ with `value_subbits=4`; functional but too lossy for task quality |
| `pagedpq_batched_prefill_decode_ctx1024_n1_b64_m32_torchbuild_vpqv8` | 32 | 100.0 | 36.275 | 3.320 | 32.954 | 544 | 0 | 0.283 | lossless-ish V-PQ control before decode sidecar fix; score preserved but decode used slow CPU-side V-PQ path |
| `pagedpq_batched_prefill_decode_ctx1024_n1_b64_m32_torchbuild_vpqv8_fastdecode` | 32 | 100.0 | 9.048 | 3.172 | 5.875 | 544 | 0 | 0.283 | same lossless-ish V-PQ after torch-GPU decode sidecar path; confirms integration but still slower/costlier than exact-selected |
| `pagedpq_batched_prefill_decode_ctx1024_n1_b64_m32_torchbuild_vpqv6` | 32 | 100.0 | 8.679 | 2.917 | 5.761 | 544 | 0 | 0.227 | intermediate selected V-PQ compression keeps score on this smoke, but page128 V-codebook overhead means it is not yet a cost/runtime win |
| `pagedpq_batched_native_tail_ctx1024_n1_v2` | 1 | 100.0 | 7.733 | 1.957 | 5.775 | 129 | 0 | 0.271 total incl. update | native prefill compressed-tail path works for one layer; split cost is `0.098 selector`, `0.148 exact KV`, `0.025 tail`, `0.00018 update` MB/head-query |
| `pagedpq_batched_native_tail_ctx1024_n1_all_layers` | 32 | 100.0 | 46.739 | 14.782 | 31.957 | 4128 | 0 | 0.271 total incl. update | all-layer compressed-tail path after removing avoidable CPU V-sidecar transfer; still far from dense-close because tail decode/prefill work dominates |
| `pagedpq_batched_tail_scores_decode_l0_ctx1024` | 1 | 100.0 | 7.694 | 1.627 | 6.067 | 129 | 0 | 0.271 total incl. update | score-reuse native decode-tail op is correct and faster than the first native decode-tail kernel (`native_attention 1.87s` vs `3.02s` on layer 0), but not faster than the torch-tail path overall |
| `pagedpq_batched_tail_gqa_scores_l0_ctx1024` | 1 | 100.0 | 15.055 | 8.572 | 6.484 | 129 | 0 | 0.271 total incl. update | negative result: replacing the per-KV score loop with one GQA score-matrix selector is correct but slower here (`native_selector 0.59s`); keep it experimental, not default |
| `dense_batched_ctx2048_n2_g16` | 0 | 100.0 | 3.023 | 2.114 | 0.909 | n/a | n/a | n/a | two-sample dense batched reference at mean prompt `1884` tokens |
| `pagedpq_batched_prefill_decode_ctx2048_n2_b64_m32_torchbuild` | 32 | 100.0 | 5.670 | 4.536 | 1.135 | 1088 | 0 | 0.492 | two-sample prefill+decode approximation with device page-PQ build; index build only `0.137s`, so remaining overhead is selector/exact-attention work |
| `dense_batched_ctx4096_n1_g16` | 0 | 100.0 | 3.032 | 2.162 | 0.871 | n/a | n/a | n/a | one-sample dense batched reference at prompt `3681` tokens |
| `pagedpq_batched_prefill_decode_ctx4096_n1_b64_m32_torchbuild` | 32 | 100.0 | 14.537 | 13.359 | 1.179 | 544 | 0 | 0.937 | page size 128 is now selector dominated: index build `0.159s`, selector `0.759 MB/head-query`; page-size ablations are running |
| `pagedpq_batched_prefill_decode_ctx4096_n1_b64_m32_torchbuild_ps256` | 32 | 100.0 | 12.940 | 11.900 | 1.040 | 544 | 0 | 0.573 | page size 256 halves selector traffic to `0.368 MB/head-query`; fastest ctx4096 variant so far |
| `pagedpq_batched_prefill_decode_ctx4096_n1_b64_m32_torchbuild_ps512` | 32 | 100.0 | 13.169 | 11.885 | 1.284 | 544 | 0 | 0.431 | lower bandwidth than ps256 (`0.173 selector`, `0.258 exact KV`) but slightly slower due larger exact/pending reads and build overhead |
| `pagedpq_batched_prefill_decode_ctx4096_n1_b64_m32_torchbuild_ps512_profile` | 32 | 100.0 | 24.357 | 22.815 | 1.542 | 544 | 0 | 0.431 | synchronized profiling run only: native selector `4.364s`, native selected-attention `4.971s`, index build `2.893s`; sync/launch/HF overhead is also material |
| `dense_batched_ctx8192_n1_g16` | 0 | 100.0 | 3.253 | 2.292 | 0.961 | n/a | n/a | n/a | one-sample dense batched reference at prompt `7884` tokens |
| `pagedpq_batched_prefill_decode_ctx8192_n1_b64_m32_torchbuild_ps512` | 32 | 100.0 | 39.105 | 37.325 | 1.780 | 544 | 0 | 0.706 | negative runtime scaling result: index build only `0.636s`; sparse prefill selector/selected-attention path is the bottleneck |
| `pagedpq_batched_all_ctx512_ps128_n1_denseequiv` | 32 | 100.0 | 7.220 | 1.068 | 6.152 | 0 | 4128 | 0.110 | dense-equivalent fast path; budget covers all sealed tokens, so selector is correctly skipped |
| `pagedpq_batched_all_ctx512_ps128_k64_active` | 32 | 100.0 | 218.318 | 1.056 | 217.262 | 2112 | 2016 | 0.114 | active selector path; Python per-head/per-query overhead is the bottleneck |
| `pagedpq_batched_l16_ctx512_ps128_n1_fastpath` | 1 | 100.0 | 44.534 | 20.083 | 24.451 | 66 | 63 | 0.115 | one-layer active-selector reference before dense-equivalent budget fast path |
| `pagedpq_batched_all_ctx512_ps128_n1_fastpath` | 32 | 100.0 | 199.263 | 1.052 | 198.211 | 2112 | 2016 | 0.115 | historical; this should have been dense-equivalent and is superseded |
| `pagedpq_batched_l16_ctx512_ps128_n1` | 1 | 100.0 | 95.424 | 55.984 | 39.440 | 129 | 0 | 0.115 | pre-fastpath reference |

Implementation fixes from the smoke:

- RULER wrapper now passes `prefill_method=full` correctly instead of accidentally passing `synthetic`.
- RULER native tail smokes need explicit Slurm memory; `50150246` used `--mem=64G`, while the first attempt `50149308` requested only `768M` and was killed before validating the algorithm path.
- Selector timing synchronization was a real runtime bug in HF generation. Disabling per-selector-call CUDA sync reduced active exact-selected decode from `19.120s` to `15.279s` after selected-attention fusion.
- Optional NLTK/pandas/wonderwords fallbacks prevent environment-only failures in synthetic data/evaluation.
- Streaming/batched RULER script can reuse generated data and optionally stage model files to node-local scratch.
- HF paged-PQ intervention now includes unsealed pending-page tokens as exact reads.
- Page-PQ codebook seeds no longer depend on current context length, so sealed pages are stable as decode grows.
- Added a faithful dense fast path before the first page seals: prefix + pending + suffix covers the full context, so exact dense attention is both correct and much faster.
- Added a second dense-equivalent fast path when `budget >= sealed_indexed_tokens`: fullscan would retrieve every indexed token, so selector scoring can be skipped and dense attention is exact.
- Added per-layer sealed-page index caching and per-forward V NumPy caching.

Current downstream conclusion:

- The plumbing is correct enough for smoke: task score is preserved on this simple NIAH sample and prefill+decode patching is active.
- Runtime is improved but still not dense-close for all layers. Native CUDA selector brings the isolated selector step from `0.00918s` to `0.00012s` on the small selector-eval smoke. All-layer active streaming is now `27.8s`, down from the earlier Python active path `218.3s`, but dense streaming is still `15.0s`.
- The first real all-layer `pagedpq_batched` prefill+decode smoke at context 1024 completed with score `100.0`, mean prompt tokens `801`, generation `44.779s`, prefill `43.624s`, decode `1.156s`, and mean step `0.217 MB/head-query` (`0.0746 selector`, `0.1423 exact KV`, `0 tail`). Instrumentation showed CPU/NumPy page-PQ build was the bottleneck: `37.4s` of the `51.9s` build-timing run.
- The new device-side page-PQ build path changes the runtime frontier: the same ctx1024 all-layer smoke is now `4.457s` generation with index build `1.206s`, compared with dense ctx1024 at `6.053s`. Ctx2048 two-sample validation remains correct (`100.0` score) and practical: dense `3.023s/sample`, approximate `5.670s/sample`, with index build only `0.137s` total. Ctx4096 remains slower than dense, but page-size tuning confirms the bottleneck shifted from index build to selector/exact-attention balance: ps128 `0.937 MB/head-query`, ps256 `0.573`, ps512 `0.431`.
- Ctx8192 is a negative runtime result for the current sparse prefill implementation: dense prefill is `2.292s`, while sparse prefill is `37.325s` even though page-PQ index build is only `0.636s`. This means the current CUDA selected-token prefill path is functionally correct but not runtime-scalable against optimized dense/SDPA prefill. Further GPU work should target the prefill sparse-attention kernel/selector scheduling, not more page-PQ build optimization.
- Compressed-tail GPU integration is now functionally validated, but not runtime-frontier. Ctx1024 all-layer compressed-tail is `46.7s` generation versus exact-selected `4.46s` and dense `6.05s`. The problem is no longer CPU page-PQ construction; it is the cost/launch structure of tail estimation and selected/tail attention during real HF generation.
- GQA-wide score-matrix selector did not improve the native decode-tail path in the RULER smoke. It is useful for API completeness and parity, but the current runtime bottleneck is not the KV-head Python loop alone.
- The latest gains were mostly engineering overhead removal: fused selected attention, removal of hot-path selector timing syncs, GQA-wide exact attention, and GQA-wide selector. The remaining gap is not solved by selector top-k alone.
- Next engineering targets are bf16/native selected+tail kernels that avoid full-cache float32 conversion, a fused tail estimator that reuses selector scores without per-token PyTorch matmuls, ctx4096+ validation, and online page-PQ refresh/rebalance beyond append-only sealed pages.

## Overnight Update 2026-05-14

Question: can previous selector frontends or selected-K compression improve the current deployable stack?

Current reference remains:

```text
fullscan paged-PQ selector
+ geometric_probe_tail_switch, tail_probe_rel_l2 <= 0.020
+ selected V-PQ with selected-mass 0.98 above 90k, easy-head cap 11264, heads 0/1 uncapped
+ V-PQ tail estimator

q288 layer16:
  max layer relL2 = 0.000666
  mean step MB/head = 6.889
  max mean-step MB/head = 21.779

128k endpoint:
  layer relL2 = 0.000375
  mean step MB/head = 17.365
```

Selector frontend sweep at 128k, all using the same V-PQ/tail/confidence backend:

| selector frontend | best 128k result | interpretation |
| --- | --- | --- |
| SparQ full ranking | rank32: `23.436 MB/head`, layer relL2 `0.000614` | Quality can be acceptable, but selector traffic is too high. |
| QUEST page bounds | `22.9-23.5 MB/head`, layer relL2 `0.006-0.013` | Page bounds are too loose; false positives and missed high-impact tokens hurt. |
| QUEST -> PQ scan | `26.3-28.6 MB/head`, layer relL2 `0.0056-0.0164` | More expensive and still low quality. |
| SparQ rerank/audit on PQ shortlist | `18.5-20.6 MB/head`, layer relL2 `0.00039-0.00048` | Does not pay for itself versus baseline `17.365 MB/head`. |

Conclusion: SparQ/QUEST are useful references, but neither is a better first-stage selector once plugged into the current V-PQ/tail backend. SparQ has good signal but high selector bandwidth; QUEST page-level routing is too coarse.

Selected-K compression sweep:

| variant | scope | layer relL2 | mean step MB/head | result |
| --- | --- | ---: | ---: | --- |
| compress all selected K with selector logits, no exact mass guard | 128k | `0.036-0.057` | `7.4-8.4` | Rejected: selected K logits are too sensitive. |
| keep exact K for top `0.998` selector-mass, compress the rest | 128k | `0.000404` | `15.928` | Endpoint looked close, so q288 validation was run. |
| keep exact K for top `0.999` selector-mass, compress the rest | 128k | `0.000384` | `16.451` | Endpoint looked close, so q288 validation was run. |
| exact selected K, same global selected-V `0.98` control | q288 | `0.000883` | `7.322` | Apples-to-apples control for the selected-K runs; not the current scheduled default. |
| top `0.998` selected-K mass | q288 | `0.002190` | `6.890` | Rejected: decode-1 quality outlier. |
| top `0.999` selected-K mass | q288 | `0.000900` | `7.022` | Potential endpoint/max-step optimization: `7.322 -> 7.022 MB/head` vs same global control, but slightly worse quality. |

Important caveat: the selected-K compression runs are best treated as compression-potential tests. The current confidence and selected-V decisions still use exact selected logits internally. A deployable selected-K-compressed rule would need to remove or explicitly charge those hidden exact-logit dependencies. Also, this q288 selected-K sweep used a global selected-V `0.98` setting, while the current best reference is context-scheduled. Do not promote selected-K compression yet; the clean next test is a context-scheduled selected-K gate, e.g. exact selected K below the long-context threshold and top `0.999` selected-K mass only above it.

Post-hoc context-gated recombination from the exact/global and selected-K/global q288 rows:

| selected-K gate | max layer relL2 | mean step MB/head | max mean-step MB/head | interpretation |
| --- | ---: | ---: | ---: | --- |
| exact selected K everywhere, global selected-V `0.98` | `0.000883` | `7.322` | `20.708` | apples-to-apples control |
| top `0.999` selected-K mass only for decode >= 32k | `0.000883` | `7.164` | `19.612` | preserves the global-control quality while reducing long-context cost |
| top `0.998` selected-K mass only for decode >= 32k | `0.000883` | `7.090` | `19.233` | cheaper and still avoids the decode-1 outlier in this recombination |

This is only a recombination of already-run rows, not a separately executed single policy. It suggests selected-K compression may be useful as a long-context-only cost cap, but only after the hidden exact-logit dependency is cleaned up.

Executed context-scheduled selected-K policy, output root:
`attention_efficiency_result/selected_key_schedule_20260514_q288_v2`

Policy for this run:

- Below decode `90k`: selected-V mass `0.99`, no selected-V cap, geometric max budget `65536`.
- At/above decode `90k`: selected-V mass `0.98`, easy-head selected-V cap `11264`, heads `0,1` uncapped, geometric max `90000` with heads `0,1` max `120000`.
- Selected-K compression starts at decode `32k`; below that selected K stays exact.

| policy | rows | max layer relL2 | max attn relL2 | mean step MB/head | max mean-step MB/head | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| scheduled exact selected K | 288 | `0.000671` | `0.002169` | `7.412` | `20.807` | apples-to-apples exact-K control for this schedule implementation |
| scheduled selected-K top `0.999` mass above 32k | 288 | `0.000715` | `0.002229` | `7.251` | `19.848` | Better cost, but quality regression is measurable. |
| scheduled selected-K top `0.9995` mass above 32k | 288 | `0.000682` | `0.002198` | `7.306` | `20.115` | Best selected-K tradeoff so far: almost preserves exact-K quality while cutting mean step by `1.4%` and max step by `3.3%`. |

Conclusion: selected-K compression is not a major frontier shift, but a long-context-only exact-mass gate can shave peak/endpoint cost with small quality loss. The clean candidate for any follow-up is `0.9995` above decode `32k`, not broad selected-K compression. The hidden exact-logit caveat still applies.

Slope-stability confidence check:

Question: should the online confidence rule verify local convergence with three nested exact prefixes `k-delta`, `k`, `k+delta` instead of only comparing a compressed-tail estimate against a larger exact probe?

Implemented `geometric_slope_stability` in `run_layer_quality_eval.py`. It records:

- `slope_forward_rel_l2`: exact-output change from `k` to `k+delta`.
- `slope_backward_rel_l2`: exact-output change from `k-delta` to `k`.
- `slope_ratio`: forward/backward slope ratio.
- `slope_curvature_rel_l2`: second-difference magnitude.

Targeted Slurm result on decode lengths `76387, 92903, 128000`, output root:
`attention_efficiency_result/slope_confidence_20260514_smoke`

| confidence rule | max layer relL2 | mean step MB/head across slice | max mean-step MB/head | interpretation |
| --- | ---: | ---: | ---: | --- |
| current `geometric_probe_tail_switch` | `0.000577` | `17.386` | `18.432` | Baseline. |
| slope metrics only, no gate | `0.000577` | `17.386` | `18.432` | Same decisions as baseline; the recorded slopes mostly indicate diminishing returns. |
| strict slope gate: forward `<=0.02`, backward `<=0.05`, ratio `<=1`, curvature `<=0.05` | `0.000579` | `18.329` | `19.546` | Rejected: raises cost without improving layer-output quality. |
| relaxed slope gate: forward `<=0.03`, backward `<=0.08`, ratio `<=1.25`, curvature `<=0.08` | `0.000579` | `17.953` | `18.987` | Rejected: still adds cost without quality gain. |
| forward-only slope `<=0.01` | `0.000551` | `19.713` | `22.157` | Quality improves slightly but cost is too high for this frontier. |
| forward-only slope `<=0.02` | `0.000579` | `18.090` | `19.546` | Rejected: cost increase, no robust quality gain. |
| forward-only slope `<=0.03` | `0.000579` | `17.798` | `18.987` | Rejected: cost increase, no robust quality gain. |

Conclusion: three-point slope stability is useful as a diagnostic but not as the next deployable confidence rule. The backward/curvature gates mostly punish heads where the previous budget increment was important, not necessarily heads with worse final layer-output error. A forward-only gate is cleaner, but the useful quality gain only appears at an expensive threshold. Keep the current geometric probe confidence rule for now.

Important caveat: these probe-style confidence rules still use exact selected-prefix outputs internally. That is acceptable for diagnostic sweeps, but a fully deployable confidence rule must either charge those probe V reads explicitly or replace the probe with a compressed/sample-based uncertainty estimate.

## Morning Update 2026-05-13

Completed since the previous checkpoint:

| experiment | scope | result | interpretation |
| --- | --- | --- | --- |
| Correct q288 QKV conversion for layers 8 and 24 | 128k X traces -> q288 Q/K/V NPZ | `attention_efficiency_result/layer_diversity_qkv_20260513` | Previous layer8/layer24 q288 files only had the first 32 query positions; this is fixed. |
| Current rule, layer diversity | layers 8 and 24, q288 c0-c3 | layer8 max layer relL2 `0.001656`, layer24 max layer relL2 `0.000359` | The layer16-tuned selector confidence generalizes reasonably at layer-output level, but layer8 early decode exposes selected-V compression sensitivity. |
| Exact-all selected-V gate | layers 8/24, q288 c0-c1 | max layer relL2 about `1e-6`; max step `6.416 MB/head` for early chunks | If selected fraction is >= `0.95`, using exact selected V removes early all-selected compression error with small absolute cost. |
| Residual-risk-only selected V | layer16 targeted + partial full | targeted looked good, but full early chunks hit max layer relL2 `0.006212` | Reject risk-only: residual risk without selected probability-mass guard is not deployable. |
| Selected mass + residual-risk V | layer16 q288 c0-c2 complete, c3 running | `risk0.90`: max layer `0.000669`, max step `10.444`; `risk0.99`: max layer `0.000432`, max step `11.391` on c0-c2 | Promising compression-quality candidate, but not final until c3 completes. |

Important implementation fix:

- The residual-risk selected-V cost model now reads a residual-norm sidecar for selected tokens, then reads compressed V only for selected tokens not escalated to exact V. The previous implementation reconstructed/charged V-PQ for all selected tokens before exact escalation, overcharging risk-based variants and not matching a deployable sidecar design.

Final selected-mass + residual-risk selected-V result:

| selected-V rule | q288 rows | max attn relL2 | max layer relL2 | mean step MB/head | max mean-step MB/head | per-head max attn relL2 | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| selected-set mass `0.99` + residual-risk `0.90` | 288 | 0.001667 | 0.000669 | 7.281 | 25.963 | 0.006732 | Better per-head attention output than selected-mass-only, but slightly worse layer max and cost than current cost-quality rule. |
| selected-set mass `0.99` + residual-risk `0.99` | 288 | 0.001679 | 0.000638 | 8.045 | 27.014 | 0.005211 | Best layer max and per-head compression robustness, but costs more. Use as high-quality compression reference, not default. |

Final recommendation:

- Keep `geometric_probe_tail_switch` with proxy mass and PQ-correlation gates as the selector confidence rule.
- Add `selected_value_exact_all_fraction_min=0.95` as a compression safety gate.
- Do not promote residual-risk-only selected V.
- Keep selected-mass-only V compression as the default cost-quality rule.
- Use selected-mass + residual-risk `0.99` as a high-quality compression rule when reducing per-head attention-output outliers is worth the extra bandwidth.

## Frontier Push 2026-05-13 Afternoon

Question: can we reduce bandwidth further than the current deployable rule without weakening the layer-output quality target?

Baseline for comparison:

| rule | scope | max layer relL2 | mean step MB/head | max mean-step MB/head | per-head max attn relL2 | note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| current default: `tail_probe<=0.005`, proxy mass `>=0.990`, PQ corr `>=0.70`, selected-V mass `0.99`, min exact top `1024` | q288 all | 0.000666 | 7.160 | 25.797 | 0.009887 | Current cost-quality baseline. |

Completed endpoint tests at decode `128000`:

| variant | layer relL2 | mean step MB/head | max head step MB | interpretation |
| --- | ---: | ---: | ---: | --- |
| base with geometric min `4096` | 0.000165 | 21.201 | 52.797 | Similar quality/cost to default endpoint. |
| selected-V mass `0.98` | 0.000234 | 19.873 | 48.402 | Good endpoint tradeoff. |
| selected-V cap `32768`, mass `0.99` | 0.000188 | 20.453 | 41.655 | Reduces worst-head V traffic. |
| selected-V mass `0.98`, cap `32768` | 0.000251 | 19.564 | 41.655 | Better endpoint cost, small quality hit. |
| selected-V mass `0.98`, cap `32768`, max budget `90000` | 0.000241 | 19.110 | 34.378 | Endpoint improvement over `120k` emergency cap. |
| selected-V mass `0.98`, cap `16384`, max budget `90000` | 0.000362 | 18.412 | 30.551 | Strong endpoint cost point, but needs full-context validation. |
| SparQ rerank top `65536`, rank `8/16` | 0.000409 / 0.000307 | 24.778 / 25.800 | 53.797 / 54.801 | Rejected: better ordering signal did not lower budget enough to pay for partial-K reads. |

Full q288 validation results:

| variant | scope | max layer relL2 | mean step MB/head | max mean-step MB/head | per-head max attn relL2 | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| selected-V mass `0.98`, cap `32768`, max budget `120000` | q288 all | 0.000764 | 7.956 | 25.770 | 0.010064 | Rejected as global rule: endpoint/worst-head V traffic improves, but average cost worsens. |
| selected-V mass `0.98`, cap `32768`, max budget `90000` | q288 all | 0.001716 | 7.956 | 25.756 | 0.032647 | Rejected as global robust rule: lower emergency budget creates quality outliers. |
| selected-V mass `0.98`, cap `16384`, max budget `90000` | q288 all, using c3 split shards | 0.001698 | 7.956 | 24.381 | 0.032647 | Rejected as global robust rule; useful lower-quality/cost tier only. |

Context-scheduled analysis:

- Applying caps only at very long contexts is deployable because it depends only on context length, not oracle quality.
- `cap32768/max120k` at `decode>=123k` preserves current max layer relL2 but barely changes mean step: `7.160 -> 7.154 MB/head`.
- `cap16384/max90k` at `decode>=90k` improves mean step to `6.910 MB/head`, but max layer relL2 rises to `0.001698` and per-head max attention relL2 to `0.032647`.
- Conclusion: selected-V caps are useful for endpoint/worst-head bandwidth but are not a clean robust frontier improvement under the current quality bar.

Current pending follow-up:

- `frontier_split_c3_cap16384_max90000_rescorr_20260513`: selected-V cap `16384` plus `selected_value_residual_correction=exact_mean` on c3 shards.
- Hypothesis: cap16 failures are selected-V compression outliers, mostly late head 0/1. Residual correction may keep cap16's bandwidth reduction while reducing the `0.0017` layer-output outlier.

Follow-up result:

| variant | scope | max layer relL2 | mean step MB/head | max mean-step MB/head | per-head max attn relL2 | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cap16/max90 + selected-V residual correction | c3 only | 0.004008 | 16.287 | 24.479 | 0.035992 | Rejected: residual-bias correction made selected-V compression outliers worse. |
| cap16/max90, heads `0,1` uncapped V but still max90 | c3 `>=90k` only | 0.001698 | 18.412 | 20.803 | 0.032647 | Rejected: V cap was not the main problem; hard heads still needed larger selected-token budget. |
| cap16/max90 globally, heads `0,1` use max120 and uncapped V | c3 `>=90k` only | 0.000554 | 19.460 | 21.800 | 0.007021 | Promising: restores long-context quality by spending budget only on hard heads. |
| scheduled: current default below 90k, above 90k use cap16/max90 with heads `0,1` max120 + uncapped V | q288 all | 0.000666 | 7.011 | 21.800 | 0.009887 | New best deployable cost-quality point so far. Mean step improves `7.160 -> 7.011 MB/head`; max mean-step improves `25.797 -> 21.800`. |
| scheduled: same as above but easy-head selected-V cap `8192` | q288 all | 0.000806 | 6.963 | 21.779 | 0.009887 | Better cost, but slightly above the current robust max-layer error. Treat as lower-quality tier unless cap12k lands better. |
| scheduled: same as above but easy-head selected-V cap `12288` | q288 all | 0.000666 | 6.990 | 21.779 | 0.009887 | New best robust point: preserves baseline max layer relL2 while cutting mean step `7.160 -> 6.990 MB/head`. |

Current best policy:

- For decode/context below the long-context threshold, keep the previous default rule.
- For long contexts (`decode_length >= 90000` in this trace), use selected-V cap `16384` and geometric max budget `90000` for most heads.
- Override hard heads `0` and `1` to geometric max budget `120000` and no selected-V cap.
- This is deployable in the narrow sense: it uses only context length and static head IDs, not oracle mass or dense rankings.

Open caveat:

- The hard-head IDs are currently calibrated on layer 16 of this trace. This needs layer-diversity validation before claiming generality.
- A middle cap `12288` for easy heads is the current best robust point.
- A cap `10240` interpolation is queued in `frontier_split_c3_cap10240_max90000_hhbudget_20260513`.

Evening frontier update:

Important reproducibility fix: for geometric confidence runs, `scripts/run_confidence_budget_rule_one.sh` now defaults `TAIL_PROXY_MASS_MIN` to `PROXY_MASS_TARGET`. A newly submitted batch accidentally set only `PROXY_MASS_TARGET`, which allowed `tail_confidence_pass=True` with proxy mass as low as `0.936`; those runs were discarded as incomparable.

Confirmed layer-16 scheduled results, q288 full suite:

| scheduled long-context rule | max attn relL2 | max layer relL2 | mean step MB/head | max mean-step MB/head | min true selected mass | min proxy mass | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline before evening: tailprobe `0.005`, easy-head V cap `11264`, heads `0,1` max120 + uncapped V above 90k | 0.001544 | 0.000666 | 6.984 | 21.779 | 0.990935 | 0.990057 | Previous best robust point. |
| tailprobe `0.010`, same schedule | 0.001566 | 0.000666 | 6.920 | 21.779 | 0.990466 | 0.990056 | Confirmed improvement; also layer-diversity checked on layers 8 and 24. |
| tailprobe `0.015`, same schedule | 0.001569 | 0.000666 | 6.894 | 21.779 | 0.989621 | 0.990039 | Slightly cheaper; true dense mass can dip below 0.990, but layer-output quality still holds on layer 16. |
| tailprobe `0.020`, same schedule | 0.001580 | 0.000666 | 6.889 | 21.779 | 0.989621 | 0.990039 | Current layer-diversity checked frontier. |
| tailprobe `0.030`, same schedule | 0.001580 | 0.000666 | 6.886 | 21.779 | 0.989340 | 0.990039 | Marginal layer-16 cost gain; per-head attention relL2 rises to `0.011539`, so not promoted yet. |
| tailprobe `0.050`, same schedule | 0.001580 | 0.000666 | 6.884 | 21.779 | 0.989340 | 0.990039 | Cheapest layer-16 point found tonight, but only `0.005 MB/head` below tailprobe020 and has higher per-head attention outliers. |

Layer-diversity check for tailprobe `0.020`:

| layer | schedule | max attn relL2 | max layer relL2 | mean step MB/head | max mean-step MB/head | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 8 | default below 90k, tailprobe020 cap11 hard-head schedule above 90k | 0.000885 | 0.001656 | 6.505 | 19.336 | Preserves the existing layer-8 max layer error while reducing mean step from `6.673`. |
| 24 | default below 90k, tailprobe020 cap11 hard-head schedule above 90k | 0.003011 | 0.000359 | 6.648 | 18.995 | Preserves the existing layer-24 max layer error while reducing mean step from `6.997`. |

Rejected/diagnostic evening results:

- Global `max120k` without static hard-head IDs is not better: layer-16 c3 cost rises to `20.149 MB/head` and the full schedule is worse than the hard-head schedule.
- Lowering proxy mass to `0.985/0.980` is cheaper but fails the current quality bar on layer 16 (`max layer relL2 ~= 0.00094`), so the proxy mass gate should stay at `0.990` for now.
- `tailprobe030/050` are not yet layer-diversity checked and have higher per-head attention-output outliers; treat them as exploratory layer-16-only points.

## Validation Status 2026-05-12

Current deployable low/mid/high rules have been revalidated on the saved layer-16 real Q/K/V/X trace over the 9 standard decode lengths up to 128k and all heads. q36 has 36 qidx rows, but only 9 unique query positions; repeated positions are byte-identical, so this is effectively 9 unique decode positions.

| config | selection rule | max attn relL2 | max layer relL2 | min layer cosine | max step MB/head | max exact KV MB/head | max selected V MB/head | max tail MB/head | max exact V tokens | source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `b12288/p12288` low-cost | selected-set mass `0.90` | 0.012191 | 0.003905 | 0.999992 | 8.253 | 6.072 | 1.945 | 0.113 | 3147 | `val_q36_low_selmass090_b12288_gpu_v1` |
| `b14336/p14336` mid-quality | selected V-PQ residual-risk mass `0.90` | 0.005157 | 0.002075 | 0.999998 | 8.571 | 6.421 | 2.224 | 0.106 | 3676 | `val_q36_mid_risk090_b14336_cpu_v1` |
| `b14336/p14336` high-quality | selected-set mass `0.99` | 0.005097 | 0.001813 | 0.999998 | 9.638 | 7.530 | 3.290 | 0.106 | 9994 | `val_q36_high_selmass099_b14336_cpu_v1` |

Important caveats still under validation:

- q288 window validation is queued/running in chunks to cover 288 unique decode positions, not just the 9 standard endpoints. The first q288 submission was invalid because comma-separated `DECODE_LENGTHS` was passed through `sbatch --export` and only the first decode value survived. Use the corrected v2 CPU manifest or the clean combined manifest: `attention_efficiency_result/validation_deployable_20260512_173454/q288_cpu_v2_manifest.tsv`, `attention_efficiency_result/validation_deployable_20260512_173454/clean_validation_manifest.tsv`.
- HF/logit-level intervention validation is queued for the same low/mid/high rules with `filler_repeats=1024`: `attention_efficiency_result/validation_deployable_20260512_173454/hf_v2_manifest.tsv`. The first HF submission failed because it forced `--local_files_only` against an empty `.hf_cache`; v2 lets the standard model-loading path resolve the model.
- Layer-diversity validation is queued for layers 8 and 24 using new 128k X traces, followed by q36 conversion and the same low/mid/high rules: `attention_efficiency_result/validation_deployable_20260512_173454/layer_diversity_manifest.tsv`.
- Aggregated validation tables are generated by `benchmark/selector_eval/reports/aggregate_layer_validation.py` into `attention_efficiency_result/validation_deployable_20260512_173454/aggregate/`.

q288 result after corrected v2 chunks 0-3:

| config | q288 rows | max attn relL2 | max layer relL2 | max step MB/head | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| low selected-mass `0.90` | 288 | 0.046831 | 0.018593 | 9.455 | Not robust; early selected-V compression and a long-context c3 failure both show up. |
| mid residual-risk `0.90` | 288 | 0.032440 | 0.016662 | 9.567 | Not robust; converges to the same c3 failure as high. |
| high selected-mass `0.99` | 288 | 0.032276 | 0.016611 | 11.493 | q36 endpoint validation missed a long-context c3 failure around decode `76387`. |

Targeted c0 ablation shows the early q288 failure is mostly selected-V compression being too aggressive, not selector-token failure:

| c0 ablation | max layer relL2 | max step MB/head |
| --- | ---: | ---: |
| low selected-mass `0.90`, no min exact | 0.017686 | 2.617 |
| low + `min_exact_top=1024` | 0.002614 | 2.673 |
| low + `min_exact_top=2048` | 0.002127 | 2.800 |
| low + `min_exact_top=4096` | 0.001638 | 3.137 |
| mid residual-risk `0.90`, no min exact | 0.013017 | 2.917 |
| mid + `min_exact_top=1024` | 0.002904 | 2.997 |
| mid + `min_exact_top=2048` | 0.002758 | 3.203 |
| mid + `min_exact_top=4096` | 0.001439 | 3.624 |

Follow-up queued/running: full q288 `min_exact_top=2048` for low/mid across chunks 0-3, manifest `attention_efficiency_result/validation_deployable_20260512_173454/q288_minexact2048_full_manifest.tsv`.

Partial `min_exact_top=2048` result:

| config | chunks done | max layer relL2 | max step MB/head | interpretation |
| --- | ---: | ---: | ---: | --- |
| low + `min_exact_top=2048` | c0-c3 | 0.016840 | 9.551 | Fixes early/medium q288 outliers but not the c3 budget insufficiency. |
| mid + `min_exact_top=2048` | c0-c3 | 0.016631 | 9.683 | Same: selected-V safeguard is useful but cannot solve c3. |

Diagnostic for the c3 failure at decode `76387`:

| variant | attention mass | attn relL2 | layer relL2 | step MB/head | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| high current, budget `16k` | 0.977574 | 0.032072 | 0.016611 | 9.735 | Reproduces q288 c3 failure. |
| high tail-off, budget `16k` | 0.984979 | 0.033009 | 0.016127 | 12.327 | Tail is not the root cause. |
| exact selected V + tail-off, budget `16k` | 0.984979 | 0.033008 | 0.016104 | 12.180 | Selected-V compression is not the root cause. |
| exact selected V + tail-on, budget `16k` | 0.977574 | 0.032071 | 0.016590 | 10.368 | Exact selected V still fails. |
| high current, budget `32k` | 0.984979 | 0.029753 | 0.013080 | 10.498 | More selected tokens help but not enough. |

Follow-up queued: larger-budget and dense sanity diagnostics for decode `76387`, manifest `attention_efficiency_result/validation_deployable_20260512_173454/q288_c3_decode76387_diag2_manifest.tsv`.

Note: the first larger-budget diagnostic was accidentally capped by the wrapper's fixed `CONF_BUDGETS` max of `32768`; those rows are useful as `32k` evidence, not true `65k/100k`. The wrapper now allows `CONF_BUDGETS` override. Corrected large-budget rerun manifest: `attention_efficiency_result/validation_deployable_20260512_173454/q288_c3_decode76387_diag3_manifest.tsv`.

Corrected large-budget result:

| variant | attention mass | attn relL2 | layer relL2 | step MB/head | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| high current, true budget `65k` | 0.999286 | 0.000391 | 0.000157 | 24.103 | Large budget fixes the c3 failure. |
| high tail-off, true budget `65k` | 0.999286 | 0.002778 | 0.001475 | 24.103 | Tail estimate helps but budget is the main issue. |
| dense sanity, exact V, budget `100k` | 1.000000 | 0.000014 | 0.000005 | 41.813 | Runner sanity check passes. |

Interpretation: the decode-76387 failure is selected-token budget insufficiency, not selected-V compression. A budget-knee sweep is queued for `40960/49152/57344`: `attention_efficiency_result/validation_deployable_20260512_173454/q288_c3_decode76387_budget_knee_manifest.tsv`.

Budget-knee result for decode `76387`, high selected-mass `0.99`:

| selected budget | attention mass | attn relL2 | layer relL2 | step MB/head |
| ---: | ---: | ---: | ---: | ---: |
| 40960 | 0.994074 | 0.001204 | 0.000524 | 17.170 |
| 49152 | 0.996584 | 0.000796 | 0.000347 | 19.567 |
| 57344 | 0.998247 | 0.000543 | 0.000235 | 21.879 |
| 65536 | 0.999286 | 0.000391 | 0.000157 | 24.103 |

Interpretation: the long-context c3 outlier is fixable, but it requires a much larger selected-token budget for that query. This weakens the simple fixed-budget frontier; the next algorithmic question is whether an online confidence rule can detect this high-tail-risk query without always paying 40k+ budget.

Online confidence/budget rules tested after the q288 robustness failure:

- `geometric_probe_tail_switch`: start at a small selected budget, compare compressed-tail output against a larger exact probe, and geometrically escalate when the probe check fails. This uses selected/probed exact K/V, PQ scores, and V-PQ tail sidecars only; it does not use dense/oracle mass.
- `geometric_stable_tail_switch`: same as above, plus a second compressed-tail stability check at the probe budget. This was not materially better than the simpler probe-tail rule.
- `geometric_exact_delta`: compare exact selected-output prefixes at `k` and `probe(k)` and escalate until the prefix delta is small. This is robust and tail-free, but more expensive.

Hard-query results:

| rule | decode | max layer relL2 | mean step MB/head | budget behavior | interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| fixed high selected-mass `0.99` | 76387 | 0.016611 | 9.735 | fixed `16k` | Original q288 c3 failure. |
| `geometric_probe_tail_switch`, threshold `0.05` | 76387 | 0.002519 | 12.246 | mean `22.2k`, max `63.5k` | Detects the original hard query cheaply. |
| `geometric_stable_tail_switch`, threshold `0.05` | 76387 | 0.002318 | 12.603 | mean `22.7k`, max `63.5k` | Slight quality gain, small extra confidence cost. |
| `geometric_exact_delta`, delta `0.02` | 76387 | 0.002564 | 16.383 | mean `36.0k`, max `65.5k` | Robust fallback, but more exact-read heavy. |
| `geometric_probe_tail_switch`, threshold `0.05` | 92903 | 0.008589 | 12.693 | tail passed all heads | Reveals tail-overtrust; threshold too loose. |
| `geometric_probe_tail_switch`, threshold `0.02` | 92903 | 0.003209 | 15.237 | mean `30.8k`, max `65.5k` | Stricter probe catches the second hard query. |
| `geometric_exact_delta`, delta `0.02` | 92903 | 0.003197 | 15.594 | mean `33.7k`, max `65.5k` | Similar quality/cost, no tail estimator. |

Full q288 validation for the current best online rule, `geometric_probe_tail_switch` with tail-probe threshold `0.02` and max budget `65k`:

| scope | rows | max attn relL2 | max layer relL2 | max step MB/head | min head mass | source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| q288 chunks c0-c3 | 288 | 0.009079 | 0.004510 | 17.156 | 0.884493 | `attention_efficiency_result/confidence_budget_rules_20260512/aggregate_q288_l020_merged/validation_by_config.md` |

Comparison against the fixed q288 rules:

| config | q288 rows | max attn relL2 | max layer relL2 | max step MB/head | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| fixed low selected-mass `0.90` | 288 | 0.046831 | 0.018593 | 9.455 | Not robust. |
| fixed mid residual-risk `0.90` | 288 | 0.032440 | 0.016662 | 9.567 | Not robust. |
| fixed high selected-mass `0.99` | 288 | 0.032276 | 0.016611 | 11.493 | Not robust; misses c3 hard query. |
| online geometric probe-tail `0.02` | 288 | 0.009079 | 0.004510 | 17.156 | Best robust online rule so far; costs more but removes the large q288 outliers. |

Current recommendation: use `geometric_probe_tail_switch` with threshold `0.02`, max budget `65k`, and selected-mass selected-V allocation as the robust online candidate. Keep `geometric_exact_delta` as a conservative fallback/ablation because it verifies that the remaining errors are mostly tail-confidence issues, not a fundamentally broken selector.

Confidence-rule refinement 2026-05-13:

| rule | q288 scope | max layer relL2 | mean step MB/head | max mean-step MB/head | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| geometric probe-tail `l020/max65k` | c0-c3 | 0.004510 | 5.886 | 17.156 | Previous robust candidate; remaining failures mix tail-overtrust and max-budget caps. |
| `l020/max65k` + rank-prefix K audit `2048`, mass gate `0.10` | c0-c3 | 0.004452 | 6.250 | 18.994 | Audit improves mean error but not worst-case; not frontier. |
| `l020/max65k` + rank-prefix K audit `2048`, mass gate `0.05` | c0-c3 | 0.004475 | 6.415 | 20.876 | More expensive than `m010` with no worst-case gain; reject. |
| geometric probe-tail `l020/max98k` | c3 only | 0.002459 | 11.608 | 17.581 | Raising max budget fixes the dominant `103226` cap-limited failure more cleanly than audit. |
| geometric probe-tail `l010/max98k` | c3 only | 0.001593 | 13.347 | 20.490 | Stricter probe catches more tail-overtrust heads; remaining failures are mostly max-cap or accepted very-low-threshold tail cases. |
| geometric probe-tail `l010/max120k` | c3 only | 0.001593 | 13.420 | 20.726 | Higher cap removes the `123871/125935` cap-limited head failures, but the c3 layer max is now from lower-budget tail-accepted heads. |

Interpretation: confidence is useful up to a point, but the rank-prefix exact-K audit is not a frontier rule. The cleaner direction is still a probe-centered cascade: stricter `tail_probe_rel_l2` plus a larger emergency max budget. Once a query already fails confidence and hits the max budget, extra confidence checks cannot help; the bottleneck becomes selector/budget/selected-token recall.

Selected-V and confidence refinement, 2026-05-13:

Targeted hard-query sweeps on decode positions `30451/45419/60903/74323/115613/125935` separated the remaining failures into two causes:

- Selected-V compression failures: very sharp selected distributions sometimes exact-fetch only tens of selected V vectors under `selected_mass=0.99`, then compress thousands of selected values. A deployable fix is `selected_value_min_exact_top=1024`; `2048` gives only small extra benefit on these hard points.
- Selector/tail-confidence failures: some heads have low true selected mass even when the tail probe is small. A deployable fix is to gate tail acceptance on calibrated proxy mass and selected-token PQ-score calibration quality.

Targeted hard-query result, all rows use `geometric_probe_tail_switch`, `max_budget=120k`, V-PQ tail, selected V exact until selected-set mass `0.99`, and selected-V `min_exact_top=1024` unless noted:

| rule | hard rows | max attn relL2 | max layer relL2 | mean step MB/head | max mean-step MB/head | min head mass | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no selected-V min, `tail_probe<=0.003` | 6 | 0.001853 | 0.000887 | 17.106 | 24.477 | 0.977686 | Still has selected-V and proxy-overtrust failures. |
| selected-V `min_exact_top=1024`, `tail_probe<=0.003` | 6 | 0.001853 | 0.000887 | 17.106 | 24.477 | 0.977686 | Fixes selected-V sharp-head failures but not proxy-overtrust. |
| `tail_probe<=0.003`, proxy mass `>=0.995` | 6 | 0.001763 | 0.000869 | 17.997 | 26.665 | 0.988161 | Catches some low-mass tails, but misses a PQ-overconfident head. |
| strict `tail_probe<=0.001` | 6 | 0.000560 | 0.000233 | 19.962 | 29.616 | 0.994032 | Strong quality, but more expensive. Useful conservative reference. |
| `tail_probe<=0.003`, proxy mass `>=0.995`, PQ corr `>=0.70` | 6 | 0.000633 | 0.000290 | 18.348 | 26.949 | 0.996783 | PQ-correlation gate catches the proxy-overconfident head. |
| `tail_probe<=0.003`, proxy mass `>=0.995`, PQ corr `>=0.80` | 6 | 0.000632 | 0.000237 | 19.000 | 27.520 | 0.996783 | Slightly better than corr `0.70`, but costs more. |
| `tail_probe<=0.005`, proxy mass `>=0.990`, PQ corr `>=0.70` | 6 | 0.000845 | 0.000305 | 16.788 | 25.797 | 0.994269 | Current best targeted cost-quality candidate. |

Current hypothesis: the confidence rule should not only compare tail estimate vs exact probe. It should also reject tail acceptance when the selector's compressed score model is poorly calibrated on already-fetched selected tokens. `tail_pq_corr` is a cheap deployable signal because it is fit from selected PQ scores and exact logits that were already read.

Full q288 validation result, layer 16, 288 positions, all heads:

| rule | rows | max attn relL2 | mean attn relL2 | max layer relL2 | mean layer relL2 | mean step MB/head | max mean-step MB/head | min head mass | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strict `tail_probe<=0.001`, selected-V `min_exact_top=1024` | 288 | 0.001611 | 0.000669 | 0.000718 | 0.000329 | 7.756 | 29.616 | 0.990241 | Strong but not best; strict probe alone still accepts a low-mass/corr-borderline head. |
| `tail_probe<=0.003`, proxy mass `>=0.995` | 288 | 0.002287 | 0.000692 | 0.000948 | 0.000339 | 7.396 | 26.665 | 0.988161 | Proxy mass gate helps, but without PQ-corr gate it misses proxy-overconfident low-corr heads. |
| `tail_probe<=0.003`, proxy mass `>=0.995`, PQ corr `>=0.70` | 288 | 0.001392 | 0.000646 | 0.000666 | 0.000310 | 7.526 | 26.949 | 0.995061 | Best quality among tested full q288 rules. |
| `tail_probe<=0.005`, proxy mass `>=0.990`, PQ corr `>=0.70` | 288 | 0.001668 | 0.000679 | 0.000666 | 0.000321 | 7.160 | 25.797 | 0.989337 | Current recommended cost-quality rule: same worst layer relL2 as stricter corr-gated rule, lower cost. |

Current recommendation: promote `tail_probe<=0.005`, proxy selected mass `>=0.990`, selected-token PQ corr `>=0.70`, and selected-V `min_exact_top=1024` as the current deployable online confidence rule. Keep the stricter `0.003/0.995/corr0.70` rule as the quality reference.

Output root: `attention_efficiency_result/confidence_full_validation_20260513`. Final aggregate: `attention_efficiency_result/confidence_full_validation_20260513/aggregate_final/validation_by_config.md`.

## Latest Algorithm-Search Update

128k, layer 16, all heads, real Llama-3.1-8B trace. Costs are per head/query and include selector, exact K/V, compressed-tail estimator, and confidence/audit traffic.

Latest cost-model fix: equal-budget probe/tail paths now reuse the compressed tail estimate computed for the confidence check instead of reading the same V-PQ tail twice. The layer-quality runner also reports modeled online sidecar update MB/token. For these page-local PQ configs the update term is tiny at 128k, about `0.00013 MB/head/token`; query-side selector/exact/tail reads still dominate.

Current corrected frontier:

| config | K-PQ | V-PQ tail | max attn relL2 | max layer relL2 | max step MB/head | endpoint exact MB | endpoint tail MB | endpoint confidence MB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `b6144/p6144` | `4x8` | `1x4` | 0.013849 | 0.006625 | 8.681 | 6.542 | 0.188 | 0.019 |
| `b6144/p6144` | `4x8` | `2x4` | 0.013887 | 0.006626 | 8.799 | 6.542 | 0.295 | 0.030 |
| `b6144/p6144` | `4x8` | `1x6` | 0.013705 | 0.006521 | 8.951 | 6.542 | 0.432 | 0.045 |
| `b8192/p8192` | `4x8` | `1x4` | 0.008694 | 0.005382 | 9.273 | 7.136 | 0.193 | 0.013 |
| `b8192/p8192` | `4x8` | `2x4` | 0.008679 | 0.005367 | 9.389 | 7.136 | 0.301 | 0.020 |
| `b8192/p8192` | `4x8` | `1x6` | 0.008637 | 0.005297 | 9.543 | 7.136 | 0.445 | 0.030 |
| `b12288/p12288` | `4x8` | `1x4` | 0.004782 | 0.002135 | 10.453 | 8.401 | 0.113 | 0.006 |
| `b12288/p12288` | `4x8` | `2x6` | 0.004719 | 0.002067 | 10.678 | 8.401 | 0.327 | 0.018 |
| `b12288/p12288` | `4x8` | `4x4` | 0.004783 | 0.002122 | 10.652 | 8.401 | 0.302 | 0.017 |
| `b6144/p6144` | `4x8` | `4x6` | 0.013702 | 0.006522 | 9.304 | 6.542 | 0.752 | 0.078 |
| `b8192/p8192` | `4x8` | `4x6` | 0.008635 | 0.005298 | 9.890 | 7.136 | 0.771 | 0.051 |
| `b12288/p12288` | `4x8` | `4x6` | 0.004719 | 0.002070 | 10.811 | 8.401 | 0.453 | 0.025 |
| `b6144/p6144` | `4x8` | `4x8` | 0.013613 | 0.006472 | 10.382 | 7.604 | 1.729 | 0.179 |
| `b8192/p8192` | `4x8` | `4x8` | 0.008611 | 0.005276 | 10.968 | 7.136 | 1.782 | 0.119 |
| `b12288/p12288` | `4x8` | `4x8` | 0.004695 | 0.002048 | 11.448 | 8.401 | 1.056 | 0.059 |

Failure-boundary check:

| config | V-PQ tail | max layer relL2 | max step MB/head | interpretation |
| --- | --- | ---: | ---: | --- |
| `b6144/p6144` | `1x3` | 0.006738 | 8.636 | Slightly cheaper than `1x4`, modest quality loss. |
| `b6144/p6144` | `1x2` | 0.010523 | 8.614 | Too lossy. |
| `b6144/p6144` | page mean | 0.009048 | 9.229 | Too lossy and not cheaper than `1x4` because exact fallback dominates. |
| `b8192/p8192` | `1x3` | 0.005641 | 9.228 | Slightly cheaper than `1x4`, modest quality loss. |
| `b8192/p8192` | `1x2` | 0.008149 | 9.206 | Too lossy. |
| `b12288/p12288` | `1x3` | 0.002188 | 10.426 | Slightly cheaper than `1x4`, small quality loss. |
| `b12288/p12288` | `1x2` | 0.002817 | 10.413 | Probably too lossy for the high-quality point. |
| `b12288/p12288` | page mean | 0.007736 | 10.336 | Too lossy. |

Interpretation: K selector precision matters more than V-tail precision. Reducing K-PQ from `4x8` to `4x6` or `8x6` saves selector MB but causes large ranking-quality loss. Reducing only V-PQ tail precision is a clean win: `4x8 -> 4x6 -> 4x4 -> 1x4` preserves layer error closely while cutting roughly `1.7-2.3 MB/head` from the endpoint. The failure boundary is around `1x3`; `1x2` and page-mean tails degrade clearly.

Next active experiment: selected-value compression. Exact selected K is still needed for logits, but selected V can potentially come from the same V-PQ sidecar as the tail. This directly targets the dominant exact read term.

Selected-value compression first result:

| config | selected V mode | max layer relL2 | max step MB/head | interpretation |
| --- | --- | ---: | ---: | --- |
| `b6144/p6144` | V-PQ `1x4` | 0.035607 | 6.672 | Too lossy despite large cost drop. |
| `b8192/p8192` | V-PQ `1x4` | 0.035877 | 7.148 | Too lossy. |
| `b12288/p12288` | V-PQ `1x4` | 0.035686 | 7.466 | Too lossy. |
| `b6144/p6144` | V-PQ `4x8` | 0.019296 | 9.162 | Still too lossy and not cheaper than exact V tail-only frontier. |
| `b8192/p8192` | V-PQ `4x8` | 0.018791 | 9.743 | Still too lossy. |
| `b12288/p12288` | V-PQ `4x8` | 0.018483 | 9.472 | Still too lossy. |

Interpretation: low-precision V-PQ is acceptable for the low-mass tail but not for selected/head tokens. Even `4x8` selected-V reconstruction is too lossy. The active follow-up is mixed selected values: exact V for the highest-logit selected tokens, V-PQ only for the lower selected ranks.

Mixed selected-value compression:

| config | exact selected V top | max attn relL2 | max layer relL2 | max step MB/head | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `b6144/p6144` | 1024 | 0.018110 | 0.007831 | 7.069 | Lowest-cost usable point so far. |
| `b6144/p6144` | 2048 | 0.014786 | 0.006912 | 7.280 | Better layer error at modest extra cost. |
| `b8192/p8192` | 1024 | 0.014461 | 0.005929 | 7.325 | Dominates b6144/top2048 on quality at similar cost. |
| `b8192/p8192` | 2048 | 0.009676 | 0.004339 | 7.530 | Strong new middle frontier. |
| `b12288/p12288` | 2048 | 0.007417 | 0.003447 | 8.035 | High-quality/cost tradeoff. |
| `b12288/p12288` | 3072 | 0.006183 | 0.002956 | 8.241 | Better high-quality/cost tradeoff. |
| `b12288/p12288` | 4096 | 0.005619 | 0.002773 | 8.447 | Current best high-quality compressed-selected-V point. |
| `b12288/p12288` | 6144 | 0.005202 | 0.002512 | 8.860 | Higher quality, still below exact-V tail-only cost. |
| `b12288/p12288` | 8192 | 0.005007 | 0.002347 | 9.267 | Near exact-V quality at lower cost. |
| `b14336/p14336` | 6144 | 0.005406 | 0.002214 | 8.888 | Higher-budget selector recovers quality at similar cost. |
| `b14336/p14336` | 8192 | 0.005269 | 0.002021 | 9.284 | Strong high-quality point. |
| `b14336/p14336` | 10240 | 0.005222 | 0.001894 | 9.666 | Best high-quality point so far below 10 MB/head. |
| `b14336/p14336` | 12288 | 0.005184 | 0.001833 | 10.003 | Slightly better quality, just over 10 MB/head. |
| `b14336/p14336`, V-PQ `2x4` | 8192 | 0.005264 | 0.001997 | 9.348 | Tiny quality gain over `1x4`, higher cost. |
| `b14336/p14336`, V-PQ `2x4` | 10240 | 0.005216 | 0.001888 | 9.728 | Tiny quality gain over `1x4`, higher cost. |
| `b16384/p16384` | 8192 | 0.005149 | 0.002071 | 9.507 | Does not dominate b14336/top10240. |
| `b16384/p16384` | 10240 | 0.005109 | 0.001942 | 9.890 | Does not dominate b14336/top10240. |
| `b15360/p15360` | 10240 | 0.005170 | 0.001966 | 9.772 | Does not dominate b14336/top10240. |
| `b15360/p15360` | 12288 | 0.005134 | 0.001910 | 10.109 | Does not dominate b14336/top12288. |

Interpretation: selected V cannot be fully quantized, but exact-top + compressed-tail/low-selected-rank V works. This is now the most promising compression-side direction. It recovers nearly the old exact-V b14336 quality (`0.00175` layer relL2 at `11.55 MB/head`) at `9.67 MB/head`. b15360/b16384 do not dominate. V-PQ `2x4` gives only tiny gains over `1x4`; `1x4` is the cleaner default unless the target is an extreme high-quality corner. Active follow-up: replace fixed exact-top counts with an online selected-softmax-mass rule.

Online selected-value exact allocation:

Instead of a fixed `exact_top`, keep exact V until the already-selected tokens' own softmax mass reaches a target. This uses only exact logits for tokens already selected by PQ/probe, so it is deployable and not an oracle over unselected dense attention.

| config | selected V exact rule | max attn relL2 | max layer relL2 | max step MB/head | mean exact V tokens at 128k | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `b8192/p8192` | selected mass 0.90 | 0.012191 | 0.006071 | 7.677 | 2839 | Cheap but worse than the better selector budget. |
| `b8192/p8192` | selected mass 0.95 | 0.009293 | 0.005701 | 8.035 | 4560 | Not frontier. |
| `b8192/p8192` | selected mass 0.98 | 0.008816 | 0.005469 | 8.521 | 6854 | Not frontier. |
| `b12288/p12288` | selected mass 0.90 | 0.012191 | 0.003905 | 8.253 | 3147 | Good lower-cost point. |
| `b12288/p12288` | selected mass 0.95 | 0.006453 | 0.002223 | 8.655 | 5126 | Close to b14336/m0.95 but slightly worse. |
| `b12288/p12288` | selected mass 0.98 | 0.004778 | 0.002157 | 9.230 | 7880 | Not better than b14336/m0.95 or m0.98. |
| `b14336/p14336` | selected mass 0.90 | 0.012191 | 0.003905 | 8.302 | 3178 | Same quality as b12288/m0.90 at slightly higher cost. |
| `b14336/p14336` | selected mass 0.95 | 0.006453 | 0.002104 | 8.682 | 5172 | New low/mid-quality frontier. |
| `b14336/p14336` | selected mass 0.98 | 0.005092 | 0.001829 | 9.240 | 7989 | New high-quality frontier below 10 MB/head. |
| `b14336/p14336` | risk mass 0.90 | 0.005157 | 0.002075 | 8.571 | 3343 | New mid-quality knee; slightly dominates selected-mass 0.95. |
| `b14336/p14336` | risk mass 0.95 | 0.005152 | 0.001865 | 8.978 | 5008 | Near high-quality point; selected-mass 0.98 is still slightly better quality. |
| `b14336/p14336` | selected mass 0.96 | 0.005239 | 0.001890 | 8.816 | 5855 | Good transition point between risk0.90 and risk0.95. |
| `b14336/p14336` | selected mass 0.97 | 0.005132 | 0.001849 | 8.992 | 6745 | Slightly better than risk0.95, slightly higher cost. |
| `b14336/p14336` | selected mass 0.99 | 0.005097 | 0.001813 | 9.638 | 9994 | New high-quality point below 10 MB/head. |

Interpretation: selected-mass allocation dominates fixed exact-top. It adapts the exact V count per head/query: at 128k, mass 0.95 uses about 5.2k exact V tokens on average and mass 0.99 uses about 10.0k, rather than hard-coding a token count calibrated to 128k. Risk-mass allocation is also useful: it uses a stored per-token V-PQ residual-norm sidecar and selects exact V by `selected_prob * residual_norm`. It does not beat selected-mass at the highest-quality point, but it improves the mid-quality knee.

Negative selected-V correction check:

| variant | scope | result |
| --- | --- | --- |
| Residual mean correction from exact selected V | 128k smoke | Negative. It increased selected-V MB and worsened layer relL2; likely the mean residual over high-logit exact tokens is not a stable bias estimate for lower-rank selected tokens. |
| Selected-mass union risk-mass | 128k smoke | Not frontier. It improved 128k quality versus selected-mass alone at the same mass target, but the union exact set cost too much and did not beat selected-mass 0.98/0.99. |

Active follow-up: decide whether risk-mass is worth keeping despite its extra residual-norm sidecar and slower current simulator path; then move the strongest selected-mass/risk-mass points into broader paper-style quality checks.

| config | attn relL2 | layer relL2 | step MB/head | exact MB | tail MB | confidence MB | tail-pass heads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `probe_tail_p17408_l050_costfix` | 0.019265 | 0.008875 | 12.643 | 10.589 | 0.586 | 0.615 | 24 |
| `probe_tail_p18432_l050_costfix2` | 0.013231 | 0.005978 | 12.991 | 10.964 | 0.559 | 0.615 | 23 |
| `probe_tail_p18432_l050_blend085_fullcurve` | 0.012876 | 0.005853 | 12.991 | 10.964 | 0.559 | 0.615 | 23 |
| `probe_tail_p18432_l050_vpq_blend100` endpoint | 0.012165 | 0.004941 | 12.991 | 10.964 | 0.559 | 0.615 | 23 |
| `probe_tail_p18432_l050_vpq_blend100_fullcurve` max | 0.012165 | 0.005429 | 12.991 | 10.964 | 0.559 at 128k / 0.086 mean | 0.615 | 23 |
| `probe_tail_p18432_l050_vpq_s8b6_blend100` endpoint | 0.011570 | 0.004805 | 12.755 | 10.308 | 0.527 | 0.572 | 14 |
| `probe_tail_b14336_p16384_l100_vpq_s4b8_blend100` endpoint | 0.003179 | 0.001379 | 13.915 | 9.651 | 1.162 | 1.169 | 20 |
| `probe_tail_p18432_l050_vpq_s4b8_blend100` endpoint | 0.003946 | 0.001748 | 13.799 | 10.183 | 0.810 | 0.874 | 14 |
| `probe_tail_b12288_p12288_l100_vpq_s4b8_blend100` endpoint | 0.004695 | 0.002048 | 12.504 | 8.401 | 1.056 | 1.115 | 18 |
| `probe_tail_b8192_p8192_l100_vpq_s4b8_blend100` endpoint | 0.007498 | 0.003399 | 12.749 | 7.136 | 1.782 | 1.900 | 30 |
| `probe_tail_b14336_p14336_l100_vpq_s4b8_blend100` endpoint | 0.003423 | 0.001508 | 13.296 | 9.026 | 1.169 | 1.169 | 20 |
| `probe_tail_p17408_l050_vpq_blend085_fullcurve` max | 0.018392 | 0.008040 | 12.643 | 10.589 | 0.586 at 128k / 0.090 mean | 0.615 | 24 |
| `probe_tail_p19456_l050_costfix` | 0.012826 | 0.005684 | 13.363 | 11.339 | 0.555 | 0.615 | 23 |
| `probe_tail_p19456_l050_blend085_fullcurve` | 0.012577 | 0.005601 | 13.363 | 11.339 | 0.555 | 0.615 | 23 |
| `probe_tail_p20480_l050_costfix` | 0.012062 | 0.005431 | 13.898 | 11.901 | 0.528 | 0.615 | 22 |
| `probe_tail_p21504_l050_costfix` | 0.011781 | 0.005236 | 14.254 | 12.261 | 0.525 | 0.615 | 22 |
| `sparq_rerank_r8_c32768_b16384_p18432` | 0.012135 | 0.006083 | 14.229 | 12.089 | 0.293 | 0.493 | 12 |
| `sparq_rerank_r16_c32768_b12288_p14336` | 0.012079 | 0.005957 | 15.267 | 12.214 | 0.446 | 0.753 | 18 |
| `sparq_audit_r8_c1024_b12288_p14336` | 0.016288 | 0.007940 | 15.263 | 11.214 | 0.519 | 2.677 | 21 |
| `sparq_audit_r16_c4096_b8192_p12288` | 0.008508 | 0.004664 | 18.184 | 12.058 | 0.566 | 4.707 | 23 |
| `probe_tail_p18432_l050_pm985` | 0.024066 | 0.011393 | 13.663 | 12.245 | 0.145 | 0.419 | 6 |

Interpretation:

- Increasing fixed probe budget gives a clean quality/cost curve, but no algorithmic breakthrough. p18432 remains the best cost-quality point; p21504 is a higher-quality, higher-cost point.
- Global SparQ audit catches some extra signal, but its full-context channel scan costs too much. It only beats p18432 quality at much higher step MB.
- SparQ reranking of the PQ shortlist is cheaper than global audit, but still not a frontier improvement. It indicates ordering within the PQ shortlist is not enough; the hard part is which exact tokens/tail correction to trust.
- Proxy-mass gating is negative. It disables useful tail estimates and leaves exact-only fallback with too little budget. Low proxy mass is not by itself a reliable tail-safety signal.
- Fixed partial tail blending gives a small same-cost win (`blend=0.85`), but probe-optimal and proxy-extrapolated blend rules under-correct because the paid exact probe is not the dense target.
- Separate per-page V-PQ tail reconstruction is the first substantial same-cost improvement after p18432. It keeps query-side tail MB the same size in this proxy but uses a stronger V approximation than key-code value means. Caveat: extra V-PQ sidecar update/build cost still needs explicit online accounting before claiming a final deployable cost.
- PQ parameter sweep shows a real compression-quality frontier: `8 subvecs x 6 bits` improves both quality and endpoint step MB versus `4 x 6`, while `4 x 8` buys much lower error at higher tail/confidence MB. This is now the most promising path.
- Cost bottleneck is still exact K/V reads. Selector MB is secondary; better algorithms need to reduce exact tokens or make compressed-tail quality strong enough to lower the probe budget.

## Paper-Style Quality Stack

Current fixed config:

`routed_paged_pq_k4096+strat_systematic_tail_b8_s12288`, page `5632`, layer `16`, Llama-3.1-8B real Q/K/V/X trace.

Important fix: `benchmark/selector_eval/gpu/run_gpu_paged_pq_eval.py` no longer sorts unscored tail tokens by dense/oracle attention scores. Tail bands are now based on selector-ranked candidates first, then deterministic token order for unscored tail. Older tail-estimator runs before this fix are optimistic and should not be used as deployable evidence.

Robustness update:

All-head diagnostics showed the earlier selector/tail stack is not robust enough: failures are head-specific and include routing sensitivity, PQ-ranking false negatives, and tail-estimator overcorrection. We added charged exact-K rerank diagnostics and a no-tail exact-selected fallback. Current robust frontier:

Online-confidence update:

The static `hybrid_v4_refined_pqtail` policy is cheap, but it uses a hand-calibrated per-head mask. The best deployable confidence rule found so far is:

```text
probe_tail_switch:
1. Try exact-small early exit with selector proxy mass + exact marginal-block checks.
2. If not easy, fetch k=16384 plus a bounded diagnostic probe to k=18432.
3. Build the PQ-value tail estimate from k=16384.
4. Trust the tail path only if the tail estimate is close to the k=18432 exact-probe output.
5. Otherwise fall back to exact-only and keep increasing budget using the same marginal rule.

Current thresholds:
proxy_mass_target=0.99
marginal_mass_max=0.010
marginal_score_gap_max=-6.0
tail_confidence_budget=16384
tail_probe_budget=18432
tail_probe_rel_l2_max=0.050
tail_score_calibration=affine_selected
```

This rule uses only selected/probed exact K/V, PQ selector scores, and compressed PQ-value tail sidecars. It does not use dense attention probabilities, dense rankings, achieved mass, or final dense output inside the selector logic. The diagnostic probe traffic is included in `exact_KV_MB_per_query`; failed tail probes are charged in `confidence_MB_per_query`.

Full decode-curve comparison after charging the compressed-tail confidence probe:

| config | online? | max attn-concat relL2 | max layer-output relL2 | mean step MB/head over suite | 128k step MB/head | source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `probe_tail_switch_p18432_l050` | yes | 0.013231 | 0.005978 | 6.682 | 12.991 | `probe_tail_p18432_l050_fullcurve_costfix_v1` |
| `hybrid_v4_refined_pqtail` | no, static head mask | 0.016837 | 0.007369 | 6.873 | 11.335 | `hybrid_v4_refined_pqtail_fullcurve_debug_v1` |
| `marginal_exact_m010_gap-6_t099` | yes | 0.014482 | 0.007022 | 13.536 at 128k only | 13.536 | `online_marg_m010_gm6_t099_128k_v1` |
| `proxy_mass_exact_m098` | yes | 0.041802 | 0.016937 | 9.942 at 128k only | 9.942 | `online_proxy_mass_m098_cal_128k_v1` |
| `alltail_k16384_calibrated` | no confidence gate | 0.019686 | 0.009107 | 12.230 at 128k only | 12.230 | `alltail_k16384_cal_128k_v1` |

Entropy-confidence check:

We tested an uncertainty-aware entropy rule. It fits an affine PQ-logit to exact-logit calibration on already fetched tokens, inflates unseen PQ tail logits by `z * residual_std`, computes an effective support from the resulting proxy distribution, and uses that to pick/probe the budget. This is deployable, but the tested entropy signal did not beat `probe_tail_switch`.

| config | 128k attn-concat relL2 | 128k layer-output relL2 | 128k step MB/head | mean effective support | mean required budget | source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `entropy_z0_scale1` | 0.017158 | 0.007774 | 13.419 | 1562 | 4384 | `entropy_conf_z0_s1_128k_v1` |
| `entropy_z1_scale1` | 0.016471 | 0.007507 | 15.052 | 1636 | 4384 | `entropy_conf_z1_s1_128k_v1` |
| `entropy_z2_scale1` | 0.009243 | 0.004182 | 16.475 | 1742 | 4448 | `entropy_conf_z2_s1_128k_v1` |
| `probe_tail_switch_p18432_l050` | 0.013231 | 0.005978 | 12.991 | n/a | n/a | `probe_tail_p18432_l050_128k_costfix_v1` |

Adaptive entropy budget-controller check:

We also tested a stricter adaptive controller that starts from a small budget, recomputes calibrated entropy after each exact block, grows budget geometrically, and triggers the tail probe as soon as `k >= scale * exp(H_ucb)`. This removes fixed `16k/18432` budgets, but still did not beat the fixed probe-tail rule.

| config | 128k attn-concat relL2 | 128k layer-output relL2 | 128k step MB/head | tail-pass heads | mean required budget | source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `adaptive_entropy_z1_scale1` | 0.044936 | 0.018101 | 10.544 | 32 | 4640 | `adapt_entropy_v2_z1_s1_g1p5_tm001_128k_v1` |
| `adaptive_entropy_z1_scale8_l030` | 0.031831 | 0.014759 | 15.947 | 23 | 13280 | `adapt_entropy_v3_z1_s8_l030_128k_v1` |
| `adaptive_entropy_z1_scale12_l030` | 0.012645 | 0.007106 | 16.549 | 19 | 15424 | `adapt_entropy_v3_z1_s12_l030_128k_v1` |
| `adaptive_entropy_z1_scale16_l030` | 0.012241 | 0.007235 | 16.966 | 18 | 17152 | `adapt_entropy_v4_z1_s16_l030_128k_v1` |
| `probe_tail_switch_p18432_l050` | 0.013231 | 0.005978 | 12.991 | 23 | fixed 16384/18432 | `probe_tail_p18432_l050_128k_costfix_v1` |

Interpretation: entropy is useful as a diagnostic but not yet as the primary budget controller. Raw or lightly scaled entropy is overconfident and accepts tail from too few exact tokens. Large entropy multipliers recover quality, but cost rises above the fixed probe-tail switch while still giving worse layer-output relL2. Best current online rule remains `probe_tail_switch_p18432_l050`: it is about `1.66 MB/head` more expensive than static `hybrid_v4` at 128k after cost-accounting fix, but it improves max layer-output relL2 from `0.00737` to `0.00598` without a static bad-head table.

New best candidates:

`hybrid_pqtail_exactbad`: fullscan paged-PQ selector, `k16384` with deterministic PQ-value compressed tail for most heads, but exact-only higher budgets for heads where compressed tail is biased:

```text
tail off / exact only: heads 15,24,26,27
budgets: head 15=24576, head 24=24576, head 26=26624, head 27=30720, all others=16384
```

This is not random tail sampling. It uses PQ-reconstructed tail values and PQ selector logits for the compressed tail, with compressed value-code traffic charged in `tail_estimator_MB_per_query`.

`hybrid_v4_refined_pqtail`: cheaper refinement of the same idea. It uses exact-small attention for heads that stayed stable over the full decode curve, PQ-value tail for medium/fragile heads, and high-budget exact for biased heads:

```text
exact small/off heads: 3,4,15,17,22,24,26,27,29,30,31
PQ-tail heads: all remaining heads
budgets: head 4=8192; heads 15,24=24576; head 26=26624; head 27=30720; exact-small heads=4096; PQ-tail heads=16384
```

128k layer-quality comparison:

| config | max per-head attn relL2 | attn concat relL2 | post-`o_proj` relL2 | layer-output relL2 | layer cosine | mean step MB/head | max step MB/head | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hybrid_v4_refined_pqtail` | 0.04221 | 0.01684 | 0.02156 | 0.00737 | 0.9999729 | 11.335 | 18.442 | `hybrid_v4_refined_pqtail_fullcurve_debug_v1` |
| `hybrid_v1_pqtail_exactbad` | 0.04221 | 0.01398 | 0.01998 | 0.00681 | 0.9999768 | 12.757 | 18.442 | `hybrid_pqtail_exactbad_fullcurve_debug_v1` |
| `k30720_exactonly` | 0.04221 | 0.01195 | 0.01847 | 0.00657 | 0.9999784 | 18.523 | 18.535 | `layer_exactonly_k30720_fullcurve_v1` |
| `k24576_exactonly` | 0.05293 | 0.01542 | 0.02307 | 0.00808 | 0.9999675 | 15.523 | 15.535 | `layer_exactonly_k24576_fullcurve_v1` |

Full decode-curve comparison for `hybrid_v4_refined_pqtail` against `k30720_exactonly`:

| decode | v4 layer relL2 | k30720 layer relL2 | v4 attn relL2 | k30720 attn relL2 | v4 step MB/head | k30720 step MB/head |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.000034 | 0.000000 | 0.000076 | 0.000001 | 3.479 | 3.624 |
| 1000 | 0.000070 | 0.000000 | 0.000172 | 0.000001 | 3.724 | 3.868 |
| 2000 | 0.000021 | 0.000001 | 0.000041 | 0.000001 | 4.212 | 4.356 |
| 4000 | 0.000019 | 0.000000 | 0.000045 | 0.000001 | 5.188 | 5.333 |
| 8000 | 0.000292 | 0.000001 | 0.000776 | 0.000001 | 6.616 | 7.327 |
| 16000 | 0.001730 | 0.000001 | 0.004609 | 0.000002 | 7.478 | 11.315 |
| 32000 | 0.001662 | 0.000498 | 0.004414 | 0.000896 | 10.167 | 17.709 |
| 64000 | 0.002175 | 0.005525 | 0.004751 | 0.009940 | 9.660 | 17.078 |
| 128000 | 0.007369 | 0.006574 | 0.016837 | 0.011947 | 11.335 | 18.523 |

Interpretation: `hybrid_v1` is the best quality-preserving point; it cuts endpoint mean step cost by `31%` vs `k30720_exactonly` while preserving the same worst-head attention relL2 and nearly the same layer-output relL2. `hybrid_v4` is the best cost-quality tradeoff so far; it cuts endpoint mean step cost by `39%` vs `k30720_exactonly` and is still better than `k24576_exactonly` on both cost and max layer relL2. The remaining caveat is that both use static per-head policy choices calibrated from diagnostics.

| config | max per-head attn relL2 over decode suite | 128k seed worst max per-head relL2 | max attn-concat relL2 | max layer-output relL2 | max step MB/head | source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `routed_pq_k16384_exactonly` | 0.0751 | deterministic/no tail | 0.024213 | 0.012007 | 11.535 | `layer_cheap_k16384_exactonly_fullcurve_v1` |
| `routed_pq_k4096+exactK_rerank32768+tail_s24576` | 0.0312 | 0.0312 across seeds 0-4 | 0.011072 | 0.005762 | 25.535 | `layer_robust_k4096_r32768_s24576_fullcurve_v1` |
| previous `routed_pq_k4096+tail_s16384` | 0.1417 at 128k | seed-sensitive | 0.046219 at 128k | 0.019686 at 128k | 12.821 | `layer_quality_routed_k4096_s16384_128k_v1` |

Interpretation: the most robust low-cost method found so far is actually the no-tail exact-selected fallback with larger head budget (`k16384`). It has higher per-head attention error than the rerank+tail candidate, but lower layer-output drift than the previous tail-based config and no estimator variance. The lower-error robust method is exact-K reranking of the top PQ shortlist before selecting `k4096`, then systematic stratified tail correction with `24576` samples. This charges rerank K reads explicitly and remains below dense attention bandwidth, but the cost is much higher than exact-only `k16384`.

Supporting robustness artifacts:

- Full all-head decode diagnostics: `attention_efficiency_result/robust_candidate_fullcurve_manifest.tsv`
- Seed robustness for tail candidates: `attention_efficiency_result/robust_candidate_seed_manifest.tsv`
- Layer-quality validation: `attention_efficiency_result/robust_layer_quality_manifest.tsv`

Runtime smoke on synthetic needle prompt, layer `16` intervention only, prompt length from HF run, `8` generated tokens:

| config | exact text match | affected top-1 agreement | affected logit relL2 mean / max | affected hidden relL2 mean / max | affected KL mean / max | mean step MB/head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hf_hybrid_v1`, prompt `38955` tokens | true | 1.000 | 0.02036 / 0.04341 | 0.01711 / 0.03069 | 0.000680 / 0.003901 | 9.113 |
| `hf_hybrid_v4`, prompt `38955` tokens | true | 1.000 | 0.02155 / 0.04256 | 0.01720 / 0.03106 | 0.000314 / 0.001949 | 7.828 |
| `hf_hybrid_v4`, prompt `9771` tokens | true | 1.000 | 0.02343 / 0.04077 | 0.02153 / 0.04094 | 0.000457 / 0.002700 | 2.771 |
| `hf_k16384_exactonly` | true | 1.000 | 0.024367 / 0.043000 | 0.021432 / 0.040747 | 0.000675 / 0.003857 | 2.916 |
| `hf_k4096_rerank32768_tail_s24576` | true | 1.000 | 0.020812 / 0.034023 | 0.018842 / 0.034279 | 0.000429 / 0.001955 | 4.858 |

Runtime smoke sources: `hf_hybrid_v1_layer16_needle_f1024_v1`, `hf_hybrid_v4_layer16_needle_f1024_v1`, `hf_hybrid_v4_layer16_needle_spgpu_v1`, `hf_cheap_k16384_exactonly_v1`, `hf_robust_k4096_r32768_s24576_v1`. Dense and approximate free-run outputs generated ` ZEBRA-4729. `. The `38955`-token hybrid smokes actually exercise compressed tail (`v4`: `mean_tail_estimator_MB_per_head_query=0.105`, `mean_tail_samples=11424` compressed tail tokens; `v1`: `0.140`, `15232` compressed tail tokens), unlike the shorter `9771`-token prompt where budgets covered the full context. These are still smoke tests, not task-level proof.

Deeper diagnosis and current recommendation:

The failure decomposition is now clearer.

- For small selected budgets (`k4096`), PQ approximate ordering is still the main selector-quality bottleneck. Full routing coverage alone does not fix bad heads; exact-K reranking of a PQ shortlist is what raises top-4096 recall close to 1.0.
- Once selection is good, the tail estimator is the main robustness risk. It can greatly improve selected-only outputs, but it can also overcorrect specific heads and is seed-sensitive.
- Correction clipping based on `||tail - selected|| / ||selected||` is not useful in the tested form: most caps do not activate, and aggressive caps hurt heads where a large tail correction is actually needed.
- Repeated independent tail estimates help somewhat. At equal total tail reads, `s12288 x2` improves over single `s24576` on seed 0, but seed 4 still reaches max per-head relL2 `0.0842`, so this is not robust enough to replace exact-only.
- The cleanest robust direction is currently larger exact selected sets with no tail estimator.

128k exact-only selected-budget frontier:

| config | mean per-head relL2 | p95 per-head relL2 | max per-head relL2 | mean mass | min mass | step MB/head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `k24576_exactonly` | 0.0156 | 0.0441 | 0.0529 | 0.9836 | 0.9173 | 15.523 |
| `k26624_exactonly` | 0.0132 | 0.0345 | 0.0507 | 0.9867 | 0.9208 | 16.523 |
| `k28672_exactonly` | 0.0123 | 0.0325 | 0.0487 | 0.9878 | 0.9246 | 17.523 |
| `k30720_exactonly` | 0.0099 | 0.0273 | 0.0422 | 0.9912 | 0.9444 | 18.523 |
| `k32768_exactonly` | 0.0091 | 0.0267 | 0.0382 | 0.9920 | 0.9489 | 19.523 |

Layer-quality exact-only frontier:

| config | max attn-concat relL2 | max post-`o_proj` relL2 | max layer-output relL2 | min layer cosine | max step MB/head |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k24576_exactonly` | 0.015421 | 0.024506 | 0.008079 | 0.9999675 | 15.535 |
| `k30720_exactonly` | 0.011947 | 0.018465 | 0.006574 | 0.9999784 | 18.535 |
| `k32768_exactonly` | 0.011388 | 0.017994 | 0.006412 | 0.9999795 | 19.535 |

Current recommendation: use `k30720_exactonly` as the robust paper-style candidate unless the objective is maximum quality regardless of cost. It is nearly tied with `k32768_exactonly` in layer quality, has max per-head relL2 `0.0422` at 128k, avoids tail-estimator variance entirely, and costs 1 MB/head less than `k32768`.

Layer-quality smoke after the fix:

Source: `attention_efficiency_result/layer_quality_routed_k4096_s12288_smoke_v2`

| decode | mean head mass | attn concat relL2 | post-`o_proj` relL2 | post-attn residual relL2 | layer output relL2 | layer output cosine | mean step MB/head |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8000 | 0.984183 | 0.008389 | 0.016415 | 0.005137 | 0.005185 | 0.9999866 | 5.546 |

Corrected full decode-length layer suite:

Source: `attention_efficiency_result/layer_quality_routed_k4096_s12288_full_v2`, Slurm `49792148`.

| decode | mean mass | min mass | attn relL2 | post-`o_proj` relL2 | post-attn residual relL2 | layer output relL2 | layer output cosine | mean step MB/head |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.998295 | 0.970732 | 0.001176 | 0.002469 | 0.000790 | 0.000780 | 0.9999997 | 3.248 |
| 1000 | 0.997862 | 0.969999 | 0.002019 | 0.004355 | 0.001373 | 0.001331 | 0.9999991 | 3.493 |
| 2000 | 0.998838 | 0.990509 | 0.000251 | 0.000476 | 0.000181 | 0.000179 | 1.0000000 | 3.981 |
| 4000 | 0.999436 | 0.996049 | 0.000117 | 0.000234 | 0.000080 | 0.000084 | 1.0000000 | 4.957 |
| 8000 | 0.984183 | 0.833248 | 0.008389 | 0.016415 | 0.005137 | 0.005185 | 0.9999866 | 5.546 |
| 16000 | 0.933387 | 0.712474 | 0.010289 | 0.018881 | 0.005601 | 0.005328 | 0.9999860 | 5.976 |
| 32000 | 0.944121 | 0.744347 | 0.013381 | 0.015219 | 0.004339 | 0.004190 | 0.9999913 | 9.377 |
| 64000 | 0.915800 | 0.539574 | 0.026633 | 0.044726 | 0.014123 | 0.013579 | 0.9999080 | 9.719 |
| 128000 | 0.859972 | 0.672997 | 0.104879 | 0.178448 | 0.065043 | 0.061472 | 0.9981522 | 10.732 |

Interpretation: corrected layer-output quality is good through 64k but not clean at 128k. The endpoint degradation was hidden by the earlier tail-sampling leak, so do not use pre-fix tail-estimator summaries as paper evidence. Next rung is model/runtime evaluation, but it should be run with awareness that the fixed `s12288` endpoint is already a quality risk.

128k tail-budget stress checks within the same algorithm family:

| config | mean mass | attn relL2 | post-`o_proj` relL2 | layer output relL2 | layer output cosine | mean step MB/head | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `s12288` | 0.859972 | 0.104879 | 0.178448 | 0.061472 | 0.9981522 | 10.732 | `layer_quality_routed_k4096_s12288_full_v2` |
| `s16384` | 0.859972 | 0.046219 | 0.058567 | 0.019686 | 0.9998065 | 12.732 | `layer_quality_routed_k4096_s16384_128k_v1` |
| `s32768` | 0.859972 | 0.071701 | 0.102429 | 0.032715 | 0.9994648 | 20.360 | `layer_quality_routed_k4096_s32768_128k_v1` |

Interpretation: spending more tail-estimator reads can reduce endpoint layer drift, but the current systematic stratified estimator is not monotonic (`s32768` is worse than `s16384`).

Old CPU/proxy alignment check:

Question: previous CPU/proxy results had much lower relL2, e.g. `gated_paged_pq_k4096+strat_exp_tail_b8_s4096` around `0.011-0.014` and `k16384+strat_exp_tail_b8_s4096` around `0.004-0.009`. To check whether that carries over, we reran the old-like random exponential-band tail estimator through the corrected all-head layer-quality path at 128k.

Source manifest: `attention_efficiency_result/layer_quality_oldlike_ab_manifest.tsv`, Slurm `49804469`-`49804472`, account `zhengya98`.

| config | mean mass | min mass | attn relL2 | post-`o_proj` relL2 | layer output relL2 | layer output cosine | mean step MB/head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `k4096`, page `2048`, random `s4096` | 0.837629 | 0.467575 | 0.060933 | 0.091774 | 0.031430 | 0.999510 | 5.126 |
| `k4096`, page `5632`, random `s4096` | 0.859972 | 0.672997 | 0.054071 | 0.073056 | 0.023859 | 0.999715 | 6.732 |
| `k16384`, page `2048`, random `s4096` | 0.937262 | 0.809007 | 0.047181 | 0.062618 | 0.021118 | 0.999778 | 11.539 |
| `k16384`, page `5632`, random `s4096` | 0.952839 | 0.850547 | 0.044148 | 0.054777 | 0.018456 | 0.999830 | 13.000 |
| current `k4096`, page `5632`, systematic `s16384` | 0.859972 | 0.672997 | 0.046219 | 0.058567 | 0.019686 | 0.999807 | 12.732 |

Conclusion: the very low old CPU/proxy relL2 does not directly carry over to the corrected all-head layer-quality path. The old-like random `strat_exp` setting still gives attention relL2 around `0.044-0.061` at 128k, while full layer-output relL2 is lower at `0.018-0.031` due to projection/residual attenuation. Treat the old CPU/proxy numbers as useful selector/tail diagnostics, not paper-style layer-quality evidence.

128k `s16384` seed sensitivity:

| seed | attn relL2 | post-`o_proj` relL2 | residual relL2 | layer output relL2 | layer output cosine | mean step MB/head |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.046219 | 0.058567 | 0.021347 | 0.019686 | 0.9998065 | 12.732 |
| 1 | 0.041968 | 0.052089 | 0.018986 | 0.017618 | 0.9998448 | 12.732 |
| 2 | 0.135125 | 0.183267 | 0.066799 | 0.059083 | 0.9982728 | 12.732 |
| 3 | 0.044939 | 0.056296 | 0.020519 | 0.018991 | 0.9998197 | 12.732 |
| 4 | 0.046104 | 0.059242 | 0.021593 | 0.020138 | 0.9997972 | 12.732 |

Seed-sensitivity conclusion: four seeds are acceptable-ish at 128k, but one seed regresses to nearly the `s12288` failure level. The current tail estimator should be treated as promising but not yet paper-robust.

### Runtime Logit / Task-Style Evidence

Implemented HF runtime intervention:

`benchmark/selector_eval/runners/run_hf_paged_pq_intervention_eval.py`

What it does:

- Loads Llama-3.1-8B from the local HF snapshot.
- Keeps prefill dense.
- During single-token decode, patches selected layer attention to use routed paged-PQ selection plus systematic stratified tail estimation.
- Runs dense greedy decode, then runs approximate teacher-forced decode on the dense token sequence for logit/hidden drift.
- Runs approximate free decode for a small synthetic needle task.

Current fixed runtime config:

`hf_routed_paged_pq_k4096+strat_systematic_tail_b8_s16384`, layer `16` only, prompt length `9771`, synthetic target `ZEBRA-4729`, `8` generated tokens.

Source: `attention_efficiency_result/hf_paged_pq_intervention_fixed_k4096_s16384_v3`, Slurm `49792215`.

| stack level | source | metric | result |
| --- | --- | --- | --- |
| Attention output | `layer_quality_routed_k4096_s16384_128k_v1` | 128k attention concat relL2 / cosine | `0.046219` / `0.999135` |
| Layer output | `layer_quality_routed_k4096_s16384_128k_v1` | 128k post-`o_proj` relL2 | `0.058567` |
| Layer output | `layer_quality_routed_k4096_s16384_128k_v1` | 128k full layer-output relL2 / cosine | `0.019686` / `0.9998065` |
| Model/logit | `hf_paged_pq_intervention_fixed_k4096_s16384_v3` | affected top-1 agreement | `1.000` over `7` affected decode steps |
| Model/logit | `hf_paged_pq_intervention_fixed_k4096_s16384_v3` | affected logit relL2 mean / max | `0.021324` / `0.033731` |
| Model/logit | `hf_paged_pq_intervention_fixed_k4096_s16384_v3` | affected logit cosine mean / min | `>=0.999431` min, mean summary `0.999779` over all steps |
| Model/logit | `hf_paged_pq_intervention_fixed_k4096_s16384_v3` | affected dense-to-approx KL mean / max | `0.000430` / `0.002018` |
| Model/logit | `hf_paged_pq_intervention_fixed_k4096_s16384_v3` | affected final-hidden relL2 mean / max | `0.018574` / `0.033912` |
| Task-style | `hf_paged_pq_intervention_fixed_k4096_s16384_v3` | synthetic needle exact dense-vs-approx text match | `true`: both generated ` ZEBRA-4729. ` |
| Task-style | `hf_paged_pq_intervention_fixed_k4096_s16384_v3` | target contained in output | dense `true`, approx `true` |

Runtime cost proxy for the patched layer only:

| layer | selected tokens/head | tail samples/head | selector MB/head | exact KV MB/head | tail MB/head | step MB/head |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | `4352` | `2697.4` | `0.0408` | `2.125` | `1.317` | `3.483` |

Limitations:

- Runtime eval currently patches only layer `16`, not all layers.
- The task-style result is a single synthetic needle example, not RULER/LongBench.
- Prefill is dense; this evaluates decode-time approximation quality.
- The 128k offline layer result still shows seed sensitivity for `s16384`, so the tail estimator is not yet robust enough for broad paper claims.

## Current Explicit Snapshot/Online Run

Important correction: many older `target_mass=0.98` rows are **oracle-frontier diagnostics**, not deployable selectors, because they stop by accumulating true attention probability mass inside `select()`. Treat those rows as "how good the ranking could be if we knew where to stop." Deployable rows must use selector-only stop logic and then report externally measured `attention_mass`.

Source:

```text
attention_efficiency_result/selector_eval_snapshot_online_variants_t098_h0_v3/summary.csv
```

Plots:

```text
attention_efficiency_result/selector_eval_snapshot_online_variants_t098_h0_v3/plots/
```

This is the current corrected-cost run with explicit snapshot and online selector variants at target mass `0.98`, head `0`, static prefix/suffix `128/128`.

Cost columns are unit-explicit:

- `selectorMB/query`: selector/index/router/scoring traffic for one query at this decode length.
- `exactKVMB/query`: exact K/V traffic for one query at this decode length.
- `updateCumMB`: cumulative online maintenance traffic up to this decode length.
- `updateMB/token`: `updateCumMB / decode_length`.
- `stepMB/query`: `selectorMB/query + exactKVMB/query + updateMB/token`.

### Two Comparison Modes

Use two tables, not one mixed table:

- Snapshot/query-only: assumes the index already exists. Compare `selectorMB/query` and `exactKVMB/query` separately.
- Online/realistic: includes index maintenance when that maintenance is modeled. Compare by `stepMB/query`, but only across methods with comparable online assumptions.

Current online realism status:

| algorithm | snapshot/query-only status | online/realistic status |
| --- | --- | --- |
| top_mass_oracle | oracle lower bound | not implementable |
| gated_paged_pq_snapshot / gated_paged_pq_online | valid | modeled: page sealing + router/PQ maintenance |
| paged_local_pq_snapshot / paged_local_pq_online | valid | modeled: page sealing + local PQ maintenance |
| pqcache_full_scan_snapshot / pqcache_full_scan_online_proxy | valid selector proxy | partially modeled: current-context PQ build, not optimized online maintenance |
| ivfpq_periodic_rebuild | valid selector proxy | modeled but rebuild traffic is very large |
| sparq_r16 | valid | no persistent index update modeled/needed in this proxy |
| retroinfer_style / retroinfer_online_proxy | valid snapshot/query proxy | online memory proxy for generated-token clustering/update; not full RetroInfer |
| magicpig_k10_l150 | valid hash-sidecar proxy | hash index update not modeled |
| retrievalattention_graph | valid traversal proxy | graph construction/update cost not production-faithful |

### Current Best Frontier / Deployable Status

At 128k, head `0`, target `0.98`:

| status | algorithm/config | mass | selectorMB/query | exactKVMB/query | updateMB/token | stepMB/query | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle lower bound | top_mass_oracle | 0.980001 | 0.000 | 29.963 | 0.000000 | 29.963 | `selector_eval_snapshot_online_variants_t098_h0_v3` |
| oracle-frontier diagnostic | paged_local_pq_online, page `3072`, PQ `s4b6` | 0.980000 | 1.848 | 31.097 | 0.000254 | 32.945 | `selector_eval_pagedpq_ps3072_s4b6_t098_h0_v3` |
| deployable full-suite best | `paged_local_pq_approx_sched_v2`, page `5632`, PQ `s4b6` | min mass 0.980031 | endpoint selector 1.213 | endpoint exact 31.304 | endpoint update 0.000245 | max step 32.517 | `selector_eval_pagedpq_approx_sched_v2_ps5632_s4b6_full_t098_h0_v1` |
| deployable 128k-only best | `paged_local_pq_approx_mbp8`, page `5632`, PQ `s4b6` | 0.980031 | 1.213 | 31.304 | 0.000245 | 32.517 | `selector_eval_pagedpq_approx_ps5632_s4b6_margin_fine_t098_h0_v1` |
| deployable near miss | `paged_local_pq_approx`, page `3072`, PQ `s4b6` | 0.979418 | 1.848 | 30.688 | 0.000254 | 32.536 | `selector_eval_pagedpq_approx_ps3072_s4b6_margins_smoke_t098_h0_v1` |
| deployable in progress | none currently queued | - | - | - | - | - | - |

Do not claim the `32.945 MB/query` row as a deployable result. It is the current best PQ ranking frontier. The deployable approx-stop path currently beats the old `39.914 MB/query` target at margin `6 bp`; `5 bp` misses by only `0.000079` mass.

Current full-suite deployable schedule `paged_local_pq_approx_sched_v1` uses selector-only decode-length margins: `100 bp` for decode `<=1000`, `75 bp` for decode `<=64000`, and `8 bp` for decode `128000`. It does not use achieved mass inside selector logic.

Candidate `sched_v2` uses tighter selector-only margins from calibration: `85 bp` for decode `<=500`, `80 bp` for `<=1000`, `60 bp` for `<=4000`, `50 bp` for `<=64000`, and `8 bp` for `128000`.

PQ-shape sweep under `sched_v2`, page `5632`: `s4b6` remains best. `s4b5`, `s8b4`, and `s8b6` miss the full-suite mass target; `s4b7` is valid but worse at `33.120 MB/query`.

Do not continue blind page-size or margin tuning as the main strategy.

## Active Pivot: Output Metrics Instead Of Mass-Only

Motivation from literature:

- Value-aware Approximate Attention (`arXiv:2103.09857`) argues that approximating attention weights alone ignores value vectors; the target should be the attention sub-layer output.
- CAOTE (`arXiv:2504.14051`) uses attention-output error for KV eviction and explicitly combines attention scores with values.
- Output Perturbation KV selection (`arXiv:2502.03805`) formalizes KV criticality via perturbation of attention outputs and finds that values and downstream projection matrices matter beyond attention weights.
- CurDKV (`arXiv:2509.15038`) argues that attention-score preservation does not guarantee output preservation, and selects tokens to preserve the dominant subspace of `softmax(QK^T)V`.
- Delta Attention (`arXiv:2505.11254`) frames sparse-attention degradation as attention-output distribution shift and corrects sparse outputs rather than only improving the selector.
- vAttention (`arXiv:2510.05688`) supports a top-k plus sampling direction: top-k is good for peaked distributions, sampling/estimation is better for flatter tails.

Implication for this repo: keep `attention_mass` as an interpretable proxy, but start evaluating `output_cosine` and `output_relative_L2` as first-class metrics. The most interesting next family is selected exact attention plus a cheap tail estimator/correction, not only better high-mass selectors.

Metric-stack update:

- Added `output_rmsnorm_relative_L2`.
- Added `output_centered_cosine`.
- Added channel-relative error summaries: `output_mean_abs_relative_error`, `output_p95_abs_relative_error`, `output_p99_abs_relative_error`, `output_max_abs_relative_error`.
- Added `output_linf_relative`.
- These are emitted for both exact selected sparse outputs and tail-estimated outputs.

Latest metric-stack run:

- Slurm: `49698176`
- Raw: `attention_efficiency_result/selector_eval_metric_stack_v1/summary.csv`

Full-suite max / worst-style metrics:

| algorithm | max step | max relL2 | max RMSNorm relL2 | min cosine | max mean abs rel err | max p99 abs rel err | max linf rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `top_mass_oracle` | 29.962 | 0.031111 | 0.026172 | 0.999658 | 0.133968 | 3.029125 | 0.047547 |
| `retroinfer_style` | 42.721 | 0.022886 | 0.022056 | 0.999757 | 0.080039 | 1.001096 | 0.029175 |
| `paged_local_pq_approx_sched_v2` | 32.517 | 0.031081 | 0.028323 | 0.999599 | 0.105683 | 2.038350 | 0.053965 |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s4096` | 6.851 | 0.014454 | 0.012927 | 0.999916 | 0.038098 | 0.321212 | 0.014758 |
| `gated_paged_pq_budget_k16384+strat_exp_tail_b8_s4096` | 13.137 | 0.008824 | 0.008281 | 0.999965 | 0.023592 | 0.266296 | 0.008220 |

Initial read: the stratified-tail variants also look better under normalized-space and channel-relative metrics, not only raw relL2.

Completed output-metric sweep:

- Job: `49642290`
- Output: `attention_efficiency_result/selector_eval_output_metrics_fraction_sweep_v1`
- Report: `attention_efficiency_result/selector_eval_output_metrics_fraction_sweep_v1/output_metric_fraction_results.md`
- Plots: `mass_vs_decode.png`, `output_cosine_vs_decode.png`, `output_relL2_vs_decode.png`, `stepMB_vs_decode.png`
- Selectors: `top_fraction_oracle_f10/f20/f30/f40`, `paged_local_pq_fraction_f10/f20/f30/f40`, and `paged_local_pq_approx_sched_v2`
- Decode lengths: `500,1000,2000,4000,8000,16000,32000,64000,128000`
- Target column is kept at `0.98` only so oracle-FN diagnostics remain comparable; fixed-fraction selectors do not stop using true mass.

Smoke result at decode `500`:

| algorithm | mass | output_cosine | output_relative_L2 | selectorMB/query | exactKVMB/query | stepMB/query |
| --- | --- | --- | --- | --- | --- | --- |
| top_fraction_oracle_f10 | 0.806562 | 0.965006 | 0.318621 | 0.000 | 0.358 | 0.358 |
| paged_local_pq_fraction_f10 | 0.809798 | 0.969821 | 0.291309 | 0.053 | 0.833 | 0.886 |

Interpretation: low mass can still have high cosine, but relative L2 is not yet good enough. This supports measuring both cosine and magnitude-sensitive error; cosine alone is too weak for residual-stream fidelity.

128k endpoint highlights:

| algorithm | mass | output_cosine | output_relative_L2 | stepMB/query |
| --- | ---: | ---: | ---: | ---: |
| top_fraction_oracle_f40 | 0.974002 | 0.999852 | 0.017713 | 26.336 |
| paged_local_pq_fraction_f40 | 0.971739 | 0.999831 | 0.019252 | 27.549 |
| paged_local_pq_approx_sched_v2 | 0.980031 | 0.999900 | 0.014894 | 32.517 |

Full-curve conclusion: fixed-fraction truncation is not stable enough. At 128k, 40% PQ is cheaper and close in output metrics, but across all lengths its worst-case `output_relative_L2` is `0.187403`. Even oracle 40% has worst-case `0.113919`. This means we should not simply lower the mass target. The stronger direction is exact selected top tokens plus tail correction/estimation.

Active tail-correction implementation:

- Added evaluator-side tail rows named `<selector>+<tail_estimator>`.
- Implemented `uniform_tail_s<N>_seed<S>` as a deployable importance-sampling estimator: exact selected head plus uniform samples from unselected tokens; charges sampled exact K/V reads as `tail_estimator_MB_per_query`.
- Implemented `oracle_prob_tail_s<N>_seed<S>` as diagnostic only: samples tail by true attention probability; rows are marked oracle-diagnostic and must not be counted as deployable wins.
- Implemented `rank_tail_s<N>_seed<S>` smoke-only: samples tail by selector candidate rank. Initial smoke was worse than uniform and is not yet promoted.

Smoke at decode `500`, PQ f20:

| algorithm | mass | output_cosine | output_relative_L2 | tailMB/query | stepMB/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| `paged_local_pq_fraction_f20` | 0.8098 | 0.96982 | 0.2913 | 0.0000 | 0.8857 |
| `+uniform_tail_s128_seed0` | 0.8098 | 0.99318 | 0.1149 | 0.0625 | 0.9482 |
| `+oracle_prob_tail_s128_seed0` | 0.8098 | 0.99661 | 0.0839 | 0.0625 | 0.9482 |
| `+rank_tail_s128_seed0` | 0.8098 | 0.95513 | 0.2774 | 0.0625 | 0.9482 |

Active Slurm jobs:

- Full curve: job `49642374`, output `attention_efficiency_result/selector_eval_tail_sampling_full_v1`
- 128k endpoint speculative run: job `49642386`, output `attention_efficiency_result/selector_eval_tail_sampling_128k_v1`, completed
- Narrow full-curve deployable run: job `49642399`, output `attention_efficiency_result/selector_eval_tail_sampling_full_fast_v1`, completed

128k endpoint result:

| algorithm group | mass | output_relative_L2 mean/max | stepMB/query | interpretation |
| --- | ---: | ---: | ---: | --- |
| `paged_local_pq_approx_sched_v2` | 0.980031 | 0.014894 / 0.014894 | 32.517 | current baseline |
| `paged_local_pq_fraction_f40+uniform_tail_s128` | 0.971739 | 0.003053 / 0.003162 | 27.612 | endpoint win if full-curve holds |
| `paged_local_pq_fraction_f40+uniform_tail_s512` | 0.971739 | 0.001975 / 0.002115 | 27.799 | stronger endpoint quality, still cheaper |
| `paged_local_pq_fraction_f30+uniform_tail_s512` | 0.953280 | 0.004220 / 0.004918 | 21.215 | very cheap endpoint, likely needs full-curve check |

Endpoint conclusion: tail sampling changes the picture. At 128k, low-mass PQ plus uniform tail correction beats the 0.98-mass baseline in both cost and output relative L2. Do not call this final until the full decode-length curve completes, because earlier fixed-fraction results were unstable at shorter lengths.

Full-curve deployable result:

Source: `attention_efficiency_result/selector_eval_tail_sampling_full_v1/summary.csv`

Report: `attention_efficiency_result/selector_eval_tail_sampling_full_v1/tail_sampling_summary.md`

Fast sanity run: `attention_efficiency_result/selector_eval_tail_sampling_full_fast_v1`

| algorithm group | max relL2 mean/worst seed | min cosine worst seed | max stepMB/query | endpoint relL2 mean | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| `paged_local_pq_approx_sched_v2` | 0.031081 / 0.031081 | 0.999599 | 32.517 | 0.014894 | current baseline |
| `paged_local_pq_fraction_f40+uniform_tail_s4096` | 0.011350 / 0.012558 | 0.999925 | 29.549 | 0.000625 | best conservative deployable |
| `paged_local_pq_fraction_f40+uniform_tail_s2048` | 0.021321 / 0.022121 | 0.999761 | 28.549 | 0.000999 | cheaper conservative deployable |
| `paged_local_pq_fraction_f30+uniform_tail_s4096` | 0.022184 / 0.024378 | 0.999742 | 22.965 | 0.001595 | best aggressive deployable |
| `paged_local_pq_fraction_f30+uniform_tail_s2048` | 0.032475 / 0.033360 | 0.999433 | 21.965 | 0.002173 | very cheap but slightly worse worst-case relL2 than baseline |

Current conclusion: exact selected PQ head plus uniform sampled tail correction clearly improves the cost-quality frontier under output-relative-L2. The `f40+s4096` variant beats the previous 0.98-mass PQ baseline both in full-curve worst-case relative L2 (`0.012558` worst seed vs `0.031081`) and max step MB/query (`29.549` vs `32.517`). The `f30+s4096` variant is more aggressive: it is also better than baseline on both metrics while cutting max step to `22.965` MB/query. This is the first satisfactory positive result for dropping mass as the primary target.

## Sublinear Head + Tail Experiments

Question: can the output-preserving direction reduce exact attention complexity sublinearly instead of using fixed fractions such as 30% or 40%?

Implemented budget-rule selectors:

- `top_budget_oracle_<rule>`: oracle diagnostic top-probability head under a budget rule.
- `paged_local_pq_budget_<rule>`: full-scan page-local PQ ranking, then keep a sublinear head budget.
- `gated_paged_pq_budget_<rule>`: routed/gated page-local PQ ranking, then keep a sublinear head budget.
- Budget rules include `sqrt_x8`, `log_x512`, and `n067_x1`.
- Tail estimators now support dynamic budgets such as `uniform_tail_log_x512_seed0` and `uniform_tail_sqrt_x8_seed0`, plus fixed budgets such as `uniform_tail_s16384_seed0`.

Key outputs:

- Main full sublinear sweep: `attention_efficiency_result/selector_eval_sublinear_full_fast_v1/summary.csv`
- Routed heavy-tail sweep: `attention_efficiency_result/selector_eval_sublinear_gated_tail_heavy_v1/summary.csv`
- Routed seed-check sweep: `attention_efficiency_result/selector_eval_sublinear_gated_tail_seedcheck_v1/summary.csv`
- Final merged report: `attention_efficiency_result/selector_eval_sublinear_gated_tail_seedcheck_v1/sublinear_final_summary.md`
- Plots: `sublinear_relL2_vs_decode.png`, `sublinear_stepMB_vs_decode.png`, `sublinear_selectorMB_vs_decode.png`, `sublinear_selected_tokens_vs_decode.png`

Baseline:

| algorithm | max relL2 | min cosine | max stepMB/query | selected alpha | exact+tail alpha | selector alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `paged_local_pq_approx_sched_v2` | 0.031081 | 0.999599 | 32.517 | 0.452 | 0.452 | 0.600 |

Best deployable sublinear results:

| algorithm | max relL2 | min cosine | max stepMB/query | selected alpha | exact+tail alpha | selector alpha | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `paged_local_pq_budget_log_x512+uniform_tail_log_x256` | 0.013067 | 0.999940 | 10.193 | 0.100 | 0.170 | 0.600 | best full-scan PQ sublinear exact-read result; selector still conceptually full scan |
| `paged_local_pq_budget_sqrt_x8+uniform_tail_s4096` | 0.027325 | 0.999651 | 7.237 | 0.155 | 0.083 | 0.600 | cheaper full-scan PQ result, still beats baseline quality/cost |
| `gated_paged_pq_budget_log_x512+uniform_tail_s16384` | 0.022415 | 0.999831 | 10.999 | 0.094 | 0.228 | 0.286 | best true routed/sublinear-selector result |
| `gated_paged_pq_budget_log_x512+uniform_tail_log_x512` | 0.041027 | 0.999546 | 7.260 | 0.094 | 0.108 | 0.286 | cheaper routed variant, but misses baseline relL2 on worst seed |

Conclusion: there is a credible sublinear path. The strongest overall result is full-scan PQ with sublinear exact reads, but its selector still scans PQ codes. The more important algorithmic result is routed PQ plus a heavier fixed tail (`s16384`): it beats the baseline on full-curve output relative L2 and step MB while both selector traffic and exact+tail traffic grow sublinearly in this suite. This supports a concrete next direction: improve routed candidate quality so tail budget can drop from 16k toward 4k or log-scaled while preserving output error.

### Head/Tail Split Sweep

Motivation: `gated_paged_pq_budget_log_x512+uniform_tail_s16384` was tail-dominated, so we swept fixed selected-head budgets against fixed tail budgets to test whether the same total MB is better spent on exact selected tokens or tail samples.

Source:

- Endpoint grid: `attention_efficiency_result/selector_eval_head_tail_split_128k_v1/summary.csv`
- Endpoint report: `attention_efficiency_result/selector_eval_head_tail_split_128k_v1/head_tail_split_summary.md`
- Full-curve grid: `attention_efficiency_result/selector_eval_head_tail_split_full_v1/summary.csv`
- Full-curve report: `attention_efficiency_result/selector_eval_head_tail_split_full_v1/head_tail_split_full_summary.md`

Corrected conclusion: the previous tail-heavy routed result was not optimal. Under similar total MB, moving budget from tail samples into selected head tokens improves output error.

Full-curve routed comparison:

| algorithm | max relL2 | min cosine | max stepMB/query | selector alpha | exact+tail alpha | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gated_paged_pq_budget_log_x512+uniform_tail_s16384` | 0.022415 | 0.999831 | 10.999 | 0.286 | 0.228 | older tail-heavy result |
| `gated_paged_pq_budget_k4096+uniform_tail_s12288` | 0.015435 | 0.999886 | 10.851 | 0.316 | 0.223 | better split at similar cost |
| `gated_paged_pq_budget_k16384+uniform_tail_s2048` | 0.008123 | 0.999969 | 12.137 | 0.462 | 0.246 | best routed quality/cost among tested splits |
| `gated_paged_pq_budget_k512+uniform_tail_s8192` | 0.022874 | 0.999784 | 7.053 | 0.286 | 0.109 | lower-cost routed option |

Full-scan PQ split comparison:

| algorithm | max relL2 | max stepMB/query | note |
| --- | ---: | ---: | --- |
| `paged_local_pq_budget_k4096+uniform_tail_s2048` | 0.025025 | 6.802 | very cheap, still beats baseline relL2 |
| `paged_local_pq_budget_k8192+uniform_tail_s2048` | 0.015369 | 8.802 | strong quality/cost, but selector is full scan |
| `paged_local_pq_budget_k16384+uniform_tail_s2048` | 0.006197 | 12.802 | best full-scan split in this sweep |

Takeaway: tail correction is valuable, but not as a tail-only solution. The best split gives enough exact selected tokens to stabilize the head, then uses a smaller tail sample to correct the residual. Next algorithmic target should be improving routed candidate quality for `k4k`-to-`k16k` heads, not increasing tail samples further.

### Iso-Budget Allocation Curve

Important correction: the first head/tail split sweep was a grid, not a true iso-budget allocation curve. We then ran explicit fixed-budget curves at 128k where `dynamic_head_tokens + tail_samples` is constant and the split moves from `0%` head / `100%` tail to `100%` head / `0%` tail.

Source:

- Manifest: `attention_efficiency_result/selector_eval_iso_head_tail_jobs/manifest.tsv`
- Report: `attention_efficiency_result/selector_eval_iso_head_tail_jobs/iso_head_tail_summary.md`
- Plots: `iso_head_tail_l2_gated_paged_pq.png`, `iso_head_tail_l2_paged_local_pq.png`

Routed PQ best points at 128k:

| total dynamic budget | best head/tail split | relL2 | stepMB/query | selectorMB | exactKVMB | tailMB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 tokens / 8 MB | 80% head / 20% tail | 0.006499 | 11.137 | 0.548 | 8.989 | 1.600 |
| 24,576 tokens / 12 MB | 50% head / 50% tail | 0.003704 | 15.137 | 0.548 | 8.589 | 6.000 |
| 32,768 tokens / 16 MB | 80% head / 20% tail | 0.002094 | 19.375 | 0.786 | 15.389 | 3.200 |
| 49,152 tokens / 24 MB | 90% head / 10% tail | 0.000725 | 27.771 | 1.182 | 24.189 | 2.400 |

Full-scan PQ best points at 128k:

| total dynamic budget | best head/tail split | relL2 | stepMB/query | selectorMB | exactKVMB | tailMB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 tokens / 8 MB | 70% head / 30% tail | 0.007027 | 11.802 | 1.213 | 8.189 | 2.400 |
| 24,576 tokens / 12 MB | 70% head / 30% tail | 0.003347 | 15.802 | 1.213 | 10.989 | 3.600 |
| 32,768 tokens / 16 MB | 80% head / 20% tail | 0.002140 | 19.802 | 1.213 | 15.389 | 3.200 |
| 49,152 tokens / 24 MB | 80% head / 20% tail | 0.000832 | 27.802 | 1.213 | 21.789 | 4.800 |

Conclusion: for a fixed MB budget, the optimum is not `0%` PQ head / `100%` tail. Quality generally improves as more budget goes to PQ-selected head tokens, with a smaller tail correction. The best allocations are usually head-heavy (`70-90%` head) except one routed 12 MB case where `50/50` wins. This reinforces that the algorithm should improve head selection, then use tail sampling as residual correction.

### Flipped Head/Tail Diagnostic

Question: what if we exact-read the long tail and estimate the selected head instead of exact-reading the head and estimating the tail?

Implementation:

- Added `uniform_head_s<N>` diagnostic estimator.
- For `uniform_head`, `exact_KV_MB/query` is the exact read of the unselected tail, and `tail_estimator_MB/query` is the sampled selected-head read.
- This is deployable as a Monte Carlo estimator but intentionally tests the bad/flipped direction.

Source:

- `attention_efficiency_result/selector_eval_flipped_head_tail_v2/summary.csv`
- `attention_efficiency_result/selector_eval_flipped_head_tail_v2/flipped_head_tail_summary.md`

Full-suite result:

| algorithm | max relL2 | max stepMB/query | max exactKVMB | max estimatorMB | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| `gated_paged_pq_budget_k8192+uniform_tail_s2048` | 0.022285 | 7.938 | 6.589 | 1.000 | normal direction works |
| `paged_local_pq_budget_k8192+uniform_tail_s2048` | 0.015369 | 8.802 | 6.589 | 1.000 | normal direction works |
| `gated_paged_pq_budget_k16384+uniform_tail_s2048` | 0.008123 | 12.137 | 10.589 | 1.000 | best routed normal direction in this test |
| `gated_paged_pq_budget_k16384+uniform_head_s8192` | 0.697058 | 59.798 | 55.250 | 4.000 | flipped direction fails |
| `paged_local_pq_budget_k8192+uniform_head_s8192` | 0.359136 | 64.463 | 59.250 | 4.000 | flipped direction fails |

Conclusion: flipping is not competitive. The exact tail is nearly dense, so memory cost jumps to `~55-59 MB/query` at 128k, and estimating the high-impact head has high variance even with 8192 samples. This confirms the useful decomposition is exact high-impact head plus estimated low-impact tail.

### Stratified / Multi-Resolution Tail Estimation

Question: can we improve over two-level `exact head + uniform tail estimate` by giving the tail multiple estimation resolutions?

Implemented first pass:

- `strat_tail_b<B>_s<N>`: split unselected tail into `B` selector-rank bands and allocate samples evenly.
- `strat_exp_tail_b<B>_s<N>`: rank bands with geometrically more samples for higher-ranked tail bands.
- `strat_neyman_tail_b<B>_s<N>`: proxy-Neyman allocation using per-band `exp(qk_approx) * ||v||` variance.
- These remain deployable diagnostics: no oracle probabilities or true mass are used by the estimator. The current proxy-Neyman uses exact trace scores in the simulator for the variance proxy, so treat it as less clean than `strat_exp` until rewritten to use approximate selector scores only.

Source:

- Slurm: `49690233`
- Raw: `attention_efficiency_result/selector_eval_stratified_tail_v1/summary.csv`
- Report: `attention_efficiency_result/selector_eval_stratified_tail_v1/stratified_tail_summary.md`

Key full-suite comparisons:

| algorithm | max relL2 | max stepMB/query | min cosine | interpretation |
| --- | ---: | ---: | ---: | --- |
| `gated_paged_pq_budget_k4096+uniform_tail_s2048` | 0.049478 | 5.851 | 0.998679 | baseline at 2k samples |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s2048` | 0.019839 | 5.851 | 0.999832 | same MB, much lower error |
| `gated_paged_pq_budget_k4096+uniform_tail_s4096` | 0.068940 | 6.851 | 0.998682 | uniform can be noisy/worse even with more samples |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s4096` | 0.011162 | 6.851 | 0.999936 | strong new frontier point |
| `gated_paged_pq_budget_k8192+uniform_tail_s2048` | 0.022285 | 7.938 | 0.999761 | previous routed point |
| `gated_paged_pq_budget_k8192+strat_exp_tail_b8_s2048` | 0.012044 | 7.938 | 0.999937 | same MB, nearly half relL2 |
| `gated_paged_pq_budget_k8192+uniform_tail_s8192` | 0.038656 | 10.938 | 0.999643 | uniform degrades due to variance/seed |
| `gated_paged_pq_budget_k8192+strat_exp_tail_b8_s8192` | 0.008201 | 10.938 | 0.999987 | strong routed point |
| `gated_paged_pq_budget_k16384+uniform_tail_s2048` | 0.008123 | 12.137 | 0.999969 | previous strong routed point |
| `gated_paged_pq_budget_k16384+strat_neyman_tail_b8_s2048` | 0.006263 | 12.137 | 0.999987 | better, but proxy-Neyman cleanliness needs fix |

Current answer: yes, multi-resolution tail estimation opens a promising path. The clean `strat_exp` estimator improves the cost-quality frontier at the same MB, especially for smaller exact-head budgets. Best clean point so far is `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s4096`: full-suite max relL2 `0.011162` at only `6.851 MB/query`, compared with `0.013067` at `10.193 MB/query` for the previous full-scan sublinear PQ result and `0.008123` at `12.137 MB/query` for the previous routed result.

Next work:

- Run seed sweeps for `strat_exp` to separate allocation signal from sampling variance.
- Replace proxy-Neyman's exact-score variance proxy with selector-score/PQ-score variance so it is fully deployable.
- Implement true control-variate tail estimation: cheap PQ aggregate for all bands plus sampled exact residual correction.

Seed-sweep update:

- Slurm: `49694915`
- Raw: `attention_efficiency_result/selector_eval_stratified_tail_seed_sweep_v1/summary.csv`
- Report: `attention_efficiency_result/selector_eval_stratified_tail_seed_sweep_v1/stratified_tail_seed_summary.md`
- Changed `strat_neyman_tail` to use selector/PQ approximate scores when available, not exact trace scores. It is now cleaner, but not better than `strat_exp`.

Robust seeded results, 3 seeds:

| algorithm | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | --- |
| `gated_paged_pq_budget_k4096+uniform_tail_s2048` | 0.047704 | 0.051255 | 5.851 | uniform baseline |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s2048` | 0.027293 | 0.044049 | 5.851 | better than uniform but still noisy |
| `gated_paged_pq_budget_k4096+uniform_tail_s4096` | 0.040204 | 0.068940 | 6.851 | uniform remains noisy |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s4096` | 0.012278 | 0.014454 | 6.851 | best robust low-cost point |
| `gated_paged_pq_budget_k8192+uniform_tail_s2048` | 0.025585 | 0.027655 | 7.938 | uniform baseline |
| `gated_paged_pq_budget_k8192+strat_exp_tail_b8_s2048` | 0.012734 | 0.013724 | 7.938 | robust improvement |
| `gated_paged_pq_budget_k16384+uniform_tail_s2048` | 0.009406 | 0.010404 | 12.137 | still strong at higher exact-head budget |
| `gated_paged_pq_budget_k8192+strat_exp_tail_b8_s8192` | 0.011376 | 0.013306 | 10.938 | strong, but worse cost than k4096+s4096 |

Updated conclusion: `strat_exp_tail` is a valid new path. It is not just a lucky seed. It gives the best low-cost robust frontier so far: worst-seed full-suite relL2 `0.014454` at `6.851 MB/query`, with routed selector traffic. The simple rank-geometric allocation currently beats selector-score Neyman in robustness, likely because the variance proxy is still too noisy.

Control-variate tail update:

- Implemented `strat_exp_cv_tail_b<B>_s<N>`: for each rank band, use a cheap page-mean value baseline for all tail tokens, then sample exact residuals from that band.
- Cost model charges both sampled exact K/V reads and page-mean sidecar reads; it does not get free aggregate tail information.
- Slurm: `49698665`
- Raw: `attention_efficiency_result/selector_eval_control_variate_tail_v1/summary.csv`
- Report: `attention_efficiency_result/selector_eval_control_variate_tail_v1/control_variate_seed_summary.md`

Seeded results, 3 seeds:

| algorithm | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | --- |
| `paged_local_pq_budget_k16384+strat_exp_tail_b8_s4096` | 0.003006 | 0.003363 | 13.802 | best high-quality full-scan point in this sweep |
| `paged_local_pq_budget_k16384+strat_exp_cv_tail_b8_s4096` | 0.003358 | 0.003762 | 13.808 | CV is slightly worse at this sample budget |
| `paged_local_pq_budget_k16384+strat_exp_tail_b8_s2048` | 0.005083 | 0.006180 | 12.802 | non-CV lower-sample baseline |
| `paged_local_pq_budget_k16384+strat_exp_cv_tail_b8_s2048` | 0.004152 | 0.004678 | 12.808 | CV helps when sample budget is tighter |
| `gated_paged_pq_budget_k16384+strat_exp_tail_b8_s4096` | 0.008471 | 0.008824 | 13.137 | routed high-head baseline |
| `gated_paged_pq_budget_k16384+strat_exp_cv_tail_b8_s4096` | 0.008342 | 0.009796 | 13.139 | mean slightly better, worst seed worse |
| `gated_paged_pq_budget_k8192+strat_exp_tail_b8_s4096` | 0.015981 | 0.023276 | 8.938 | non-CV medium-head baseline |
| `gated_paged_pq_budget_k8192+strat_exp_cv_tail_b8_s4096` | 0.011340 | 0.012078 | 8.939 | strongest routed CV improvement |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s4096` | 0.012278 | 0.014454 | 6.851 | current best low-cost routed point |
| `gated_paged_pq_budget_k4096+strat_exp_cv_tail_b8_s4096` | 0.022548 | 0.034690 | 6.851 | CV hurts badly at very small exact head |

Conclusion: the first control-variate version is useful but not a new default. Page-mean residual correction helps when the exact head or sample budget is moderately sized, especially `gated_paged_pq_k8192+s4096`, but it is worse for the current best low-cost `k4096+s4096` point. The page-mean baseline is not consistently correlated enough with the exact tail contribution. Next CV variant should use a stronger cheap baseline, such as PQ-code value centroids or per-band code/value aggregates, not only page means.

PQ-code value-centroid CV update:

- Implemented `strat_exp_pqcv_tail_b<B>_s<N>`.
- This reuses each page's existing K-PQ token codes and adds a V-side centroid table per page/subvector/subcode. The estimator uses PQ-score weights and reconstructed value centroids as the cheap baseline, then samples exact residuals.
- Cost model charges sampled exact K/V, V-centroid codebook reads, and amortized V-side sidecar construction. It does not charge a second per-token code stream because this variant reuses the selector's existing page PQ codes.
- Smoke: `49698934`, `attention_efficiency_result/selector_eval_pqcv_tail_smoke_v2`
- Full seeded run: `49699192`, `attention_efficiency_result/selector_eval_pqcv_tail_full_v1`
- Report: `attention_efficiency_result/selector_eval_pqcv_tail_full_v1/pqcv_seed_summary.md`

Full-suite routed seeded results, 3 seeds:

| algorithm | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | --- |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s2048` | 0.027293 | 0.044049 | 5.851 | cheap but noisy |
| `gated_paged_pq_budget_k4096+strat_exp_pqcv_tail_b8_s2048` | 0.018285 | 0.018855 | 5.882 | PQCV stabilizes this very cheap point |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s4096` | 0.012278 | 0.014454 | 6.851 | still the best low-cost point |
| `gated_paged_pq_budget_k4096+strat_exp_pqcv_tail_b8_s4096` | 0.019165 | 0.023493 | 6.882 | worse than plain stratified tail |
| `gated_paged_pq_budget_k8192+strat_exp_tail_b8_s2048` | 0.012734 | 0.013724 | 7.938 | strong plain stratified point |
| `gated_paged_pq_budget_k8192+strat_exp_pqcv_tail_b8_s2048` | 0.014047 | 0.015537 | 8.001 | worse than plain stratified tail |
| `gated_paged_pq_budget_k8192+strat_exp_tail_b8_s4096` | 0.015981 | 0.023276 | 8.938 | noisy plain stratified point |
| `gated_paged_pq_budget_k8192+strat_exp_pqcv_tail_b8_s4096` | 0.014793 | 0.022168 | 9.001 | slight mean improvement, not enough |
| `gated_paged_pq_budget_k16384+strat_exp_tail_b8_s4096` | 0.008471 | 0.008824 | 13.137 | best high-quality routed point |
| `gated_paged_pq_budget_k16384+strat_exp_pqcv_tail_b8_s4096` | 0.008561 | 0.010214 | 13.278 | worse worst-case |

Conclusion: V centroids keyed by K-PQ subcodes are a better baseline than page means for one very cheap noisy corner (`k4096+s2048`), but they do not move the current frontier. The likely issue is correlation: K-PQ subcodes are good enough for scoring/ranking, but the induced V centroids are not consistently accurate enough as an output baseline. Do not keep pushing this exact PQCV form unless paired with a stronger V grouping or per-band aggregate.

### Compression-Side Experiments

Question: after selector-side exploration, can selected tokens be attended using compressed/quantized K/V instead of retrieving exact K/V?

Implemented estimator variants:

- `pq_head_only`: selected head uses selector PQ scores as approximate K logits and K-code-derived V centroids.
- `pq_head_strat_exp_tail_b<B>_s<N>`: same compressed selected head plus stratified exact tail sampling.
- `vpq_head_only`: selected head keeps exact K logits but uses a true per-page V-PQ codebook/codes for selected V.
- `vpq_head_strat_exp_tail_b<B>_s<N>`: V-PQ selected head plus stratified exact tail sampling.
- `vpq_after_d<D>_strat_exp_tail_b<B>_s<N>`: exact selected K/V for decode lengths `<=D`, V-PQ selected head after `D`, with the same stratified tail estimator.

Cost model:

- `pq_head_*` reuses selector PQ scores/codes and charges V-centroid sidecar reads plus exact fallback K/V for static/pending tokens.
- `vpq_head_*` charges exact selected K reads, V-PQ codebook/code reads, exact fallback K/V for static/pending tokens, and amortized V-PQ sidecar build traffic.
- Tail samples remain exact K/V reads.

Key runs:

- Raw K/V compression smoke: `49700002`, `attention_efficiency_result/selector_eval_kv_compression_smoke_v2`
- Full V-PQ selected-head sweep: `49700006`, `attention_efficiency_result/selector_eval_vpq_head_full_v1`
- Scheduled V-PQ sweep: `49700017`, `attention_efficiency_result/selector_eval_vpq_schedule_full_v1`
- 128k-only schedule check: `49700033`, `attention_efficiency_result/selector_eval_vpq_schedule_d64000_v1`
- K-logit residual compression smoke: `49700098`, `attention_efficiency_result/selector_eval_kcomp_smoke_v1`
- K-logit residual full sweep: `49700100`, `attention_efficiency_result/selector_eval_kcomp_full_v1`
- K-PQ subspace permutation smoke: `49700103`, `49700104`, `attention_efficiency_result/selector_eval_kcomp_perm_smoke_v1`
- Top-64 residual smoke: `49700112`, `attention_efficiency_result/selector_eval_kcomp_topr64_smoke_v1`
- Query-local K-logit calibration smoke: `49700127`, `attention_efficiency_result/selector_eval_kcalib_smoke_v1`
- Query-local K-logit calibration full sweep: `49700133`, `attention_efficiency_result/selector_eval_kcalib_full_v1`
- Scheduled K-logit calibration: `49700138`, `attention_efficiency_result/selector_eval_kcalib_schedule_full_v2`
- Rank-band K-logit calibration smoke: `49700142`, `attention_efficiency_result/selector_eval_kbandcalib_smoke_v1`
- Scheduled rank-band K-logit calibration: `49700145`, `attention_efficiency_result/selector_eval_kbandcalib_schedule_full_v1`
- Scheduled rank-band K-logit calibration with smaller tail: `49700163`, `attention_efficiency_result/selector_eval_kbandcalib_tail2048_full_v1`

128k smoke, seed 0:

| algorithm | relL2 | stepMB/query | interpretation |
| --- | ---: | ---: | --- |
| `gated_k4096+strat_exp_tail_s4096` | 0.010143 | 6.851 | exact selected K/V baseline |
| `gated_k4096+pq_head_strat_exp_tail_s4096` | 0.076149 | 4.882 | K-PQ logits are too poorly calibrated |
| `gated_k4096+vpq_head_strat_exp_tail_s4096` | 0.013593 | 5.898 | true V-PQ is much more plausible |
| `gated_k8192+vpq_head_strat_exp_tail_s2048` | 0.014330 | 6.032 | also plausible at endpoint |

Full-suite seeded results:

| algorithm | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | --- |
| `gated_k4096+strat_exp_tail_s4096` | 0.012278 | 0.014454 | 6.851 | current exact-head low-cost baseline |
| `gated_k4096+vpq_head_strat_exp_tail_s4096` | 0.049948 | 0.049948 | 5.898 | endpoint looked good, but short/mid decode degrades |
| `gated_k4096+vpq_after_d32000_strat_exp_tail_s4096` | 0.014167 | 0.015121 | 6.655 | best scheduled compression point so far; small MB win, slight relL2 loss |
| `gated_k4096+vpq_after_d64000_strat_exp_tail_s4096` | 0.021660 | 0.025639 | 6.655 | worse; 128k V-PQ has seed sensitivity |
| `gated_k8192+vpq_after_d8000_strat_exp_tail_s4096` | 0.020156 | 0.021491 | 7.367 | lower MB than exact k8192, but worse quality |

Conclusion: compression has potential, but not with raw K-PQ logits. Quantizing K for attention logits causes large output error, likely because selector PQ scores are ranking-useful but not softmax-calibrated. True V-PQ with exact K logits is the viable direction. It can reduce endpoint MB, but full-suite quality is sensitive at shorter/mid decode lengths. The best current scheduled V-PQ point is a marginal tradeoff, not a decisive new frontier.

K-logit reconstruction variants tested:

- `kpqv_tail_b8_s4096`: base K-PQ logits plus exact selected V plus stratified tail.
- `ktopr<R>_tail_b8_s4096`: base K-PQ logits plus exact residual correction on top-`R` query dimensions, exact selected V, stratified tail.
- `krpq_tail_b8_s4096`: base K-PQ logits plus residual-PQ correction, exact selected V, stratified tail.
- `PAGED_PQ_PERMUTATION=interleave|variance_balanced`: OPQ-lite subspace reshaping smoke, not a full learned OPQ rotation.

Endpoint smoke at 128k, seed 0:

| algorithm | relL2 | stepMB/query | interpretation |
| --- | ---: | ---: | --- |
| `gated_k4096+strat_exp_tail_s4096` | 0.010143 | 6.851 | exact selected K/V reference |
| `gated_k4096+kpqv_tail_s4096` | 0.056577 | 5.851 | raw K-PQ logits are not accurate enough |
| `gated_k4096+krpq_tail_s4096` | 0.068899 | 5.898 | residual PQ did not help in this setup |
| `gated_k4096+ktopr32_tail_s4096` | 0.056033 | 6.101 | top-dim residual helps little |
| `gated_k4096+ktopr64_tail_s4096` | 0.054385 | 6.351 | more residual dims still far from exact |

Full-suite seeded K-compression result:

| algorithm | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | --- |
| `gated_k4096+strat_exp_tail_s4096` | 0.012278 | 0.014454 | 6.851 | exact selected K/V baseline |
| `gated_k4096+kpqv_tail_s4096` | 0.346309 | 0.347423 | 5.851 | fails badly at short/mid decode |
| `gated_k4096+krpq_tail_s4096` | 0.345571 | 0.349953 | 5.898 | residual-PQ correction does not fix calibration |
| `gated_k4096+ktopr32_tail_s4096` | 0.345594 | 0.346429 | 6.101 | top-dim residual correction also fails full-suite |

OPQ-lite smoke:

- `interleave` and `variance_balanced` subspace permutations did not materially change the endpoint conclusion.
- At 128k, both gave `~0.055-0.056 relL2` for `ktopr32/kpqv`, still far from the exact-head `~0.010 relL2`.
- This does not rule out true learned OPQ or random-rotation quantizers, but simple subspace reshaping is not enough.

K-logit calibration update:

- `kcalib<P>_tail_b8_s4096`: read exact K for `P` selected-head probe tokens, fit one query-local affine map from PQ logits to exact logits, then use calibrated PQ logits for selected-head attention.
- `kbandcalib<P>x<B>_tail_b8_s4096`: same idea, but fit separate affine maps across `B` rank bands.
- `kcalib_after_d<D>_p<P>_...` and `kbandcalib_after_d<D>_p<P>x<B>_...`: use exact selected K/V for decode lengths `<=D`, then switch to calibrated K-PQ logits.
- These variants are deployable in principle: probes are exact key reads charged in the cost model; no oracle mass or dense probabilities are used inside the estimator.

128k smoke:

| algorithm | relL2 | stepMB/query | interpretation |
| --- | ---: | ---: | --- |
| `gated_k4096+strat_exp_tail_s4096` | 0.010143 | 6.851 | exact selected K/V reference |
| `gated_k4096+kpqv_tail_s4096` | 0.056577 | 5.851 | uncalibrated K-PQ logits are too noisy |
| `gated_k4096+kcalib128_tail_s4096` | 0.010610 | 5.882 | global affine calibration nearly fixes endpoint |
| `gated_k4096+kbandcalib128x4_tail_s4096` | 0.007509 | 5.882 | rank-band calibration beats exact-head endpoint |
| `gated_k8192+kbandcalib1024x8_tail_s4096` | 0.008240 | 7.188 | strong endpoint quality/cost |

Full-suite scheduled rank-band result:

| algorithm | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | --- |
| `gated_k4096+strat_exp_tail_s4096` | 0.012278 | 0.014454 | 6.851 | current low-cost exact-head baseline |
| `gated_k4096+kbandcalib_after_d64000_p1024x8_tail_s4096` | 0.013740 | 0.014563 | 6.655 | nearly same worst relL2, lower MB |
| `gated_k8192+strat_exp_tail_s4096` | 0.015981 | 0.023276 | 8.938 | exact-head k8192 baseline |
| `gated_k8192+kbandcalib_after_d64000_p1024x8_tail_s4096` | 0.010180 | 0.011945 | 8.707 | clear K-compression win for k8192 |
| `gated_k4096+kbandcalib_after_d64000_p1024x8_tail_s2048` | 0.019606 | 0.021178 | 5.655 | cheaper, but worse quality |
| `gated_k8192+kbandcalib_after_d64000_p1024x8_tail_s2048` | 0.031112 | 0.064349 | 7.707 | unstable with smaller tail |

Conclusion from K calibration: K compression is not dead. Plain K-PQ reconstruction fails, residual-PQ/top-dim residuals do not fix it, but query-local rank-band calibration is effective. The best current K-compressed low-cost point is a near-frontier tradeoff: `k4096+kbandcalib_after_d64000_p1024x8_tail_s4096` cuts max step from `6.851` to `6.655 MB/query` with almost unchanged worst relL2 (`0.014454 -> 0.014563`). For a higher-quality operating point, `k8192+kbandcalib_after_d64000_p1024x8_tail_s4096` beats exact-head k8192 in both quality and cost.

Next compression-side direction:

- Improve V compression quality at similar bytes, for example larger V-PQ codebooks, residual V-PQ, or per-band/per-head V codebooks.
- For K compression, focus on calibrated inner-product approximation. The next concrete step is making rank-band calibration less hand-scheduled and testing learned/statistical confidence rules for when calibration is safe.

V-PQ shape sweep update:

- Added independent V-PQ shape controls: `VALUE_PQ_SUBVECS`, `VALUE_PQ_SUBBITS`.
- Selector K-PQ shape remains `s4b6`; V-PQ sidecar can now use separate `s4b6`, `s4b7`, `s8b6`, or `s8b7`.
- Shape sweep jobs: `49700663-49700666`, `attention_efficiency_result/selector_eval_vpq_shape_jobs_v1`.
- Combined comparison with best V-PQ shape `s8b7`: `49700672`, `attention_efficiency_result/selector_eval_kband_vpq_combo_v1`.

Best V-PQ shape results:

| V-PQ shape | algorithm | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| exact V | `gated_k4096+strat_exp_tail_s4096` | 0.012278 | 0.014454 | 6.851 | exact selected K/V baseline |
| `s4b6` | `gated_k4096+vpq_after_d32000_tail_s4096` | 0.014167 | 0.015121 | 6.655 | original V-PQ |
| `s8b7` | `gated_k4096+vpq_after_d32000_tail_s4096` | 0.013644 | 0.014965 | 6.655 | best tested V-PQ shape |
| `s8b7` | `gated_k8192+vpq_after_d64000_tail_s4096` | 0.016340 | 0.023223 | 8.707 | similar worst-case to exact k8192, slightly cheaper |

Comparison against calibrated K:

| algorithm | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | --- |
| `gated_k4096+strat_exp_tail_s4096` | 0.012278 | 0.014454 | 6.851 | exact selected K/V baseline |
| `gated_k4096+vpq_after_d32000_tail_s4096`, V-PQ `s8b7` | 0.013644 | 0.014965 | 6.655 | small V-compression tradeoff |
| `gated_k4096+kbandcalib_after_d64000_p1024x8_tail_s4096` | 0.013740 | 0.014563 | 6.655 | small K-compression tradeoff, slightly better worst-case |
| `gated_k8192+strat_exp_tail_s4096` | 0.015981 | 0.023276 | 8.938 | exact k8192 baseline |
| `gated_k8192+vpq_after_d32000_tail_s4096`, V-PQ `s8b7` | 0.017356 | 0.025372 | 8.707 | V-PQ worsens worst-case at k8192 |
| `gated_k8192+kbandcalib_after_d64000_p1024x8_tail_s4096` | 0.010180 | 0.011945 | 8.707 | calibrated K dominates V-PQ for k8192 |

Conclusion from V compression: V-PQ is viable but modest. Increasing V-PQ capacity from `s4b6` to `s8b7` improves quality at the same reported step MB because selected V read is still dominated by exact selected K plus V-PQ codebook/code traffic. However, V-PQ does not yet beat calibrated K. The best V-PQ point is a small cost/quality tradeoff near the k4096 frontier, not a major new frontier.

Next V-compression direction:

- Residual V-PQ may be more useful than simply increasing codebook size.
- Consider mixed exact/V-PQ by rank band: exact V for the highest selected tokens, V-PQ for lower selected tokens. Current scheduled V-PQ is decode-length based, not rank/importance based.

### relL2 Oracle Diagnostics

Question: if the primary target is output-relative-L2 rather than attention mass, how much room exists below the 98%-mass oracle?

Implemented diagnostics:

- `top_prob_oracle`: same probability-ranking oracle as the mass oracle, evaluated by output relL2.
- `rel_l2_contribution_oracle`: value-aware one-shot ranking by `p_i * ||v_i - dense_output||`.
- `rel_l2_greedy_batch_oracle`: offline greedy diagnostic over a top contribution candidate pool; this uses dense output/value information and is not deployable.

Source:

- Raw samples: `attention_efficiency_result/selector_eval_rel_l2_oracle_v1/samples.csv`
- Frontier: `attention_efficiency_result/selector_eval_rel_l2_oracle_v1/frontier.csv`
- Report: `attention_efficiency_result/selector_eval_rel_l2_oracle_v1/summary.md`
- Slurm: `49678846` ran successfully through the evaluator; Slurm marked it failed only because the first postprocessor used a broken/shadowed `pandas` import. The postprocessor was fixed and the report was regenerated.

128k comparison against the 98%-mass oracle endpoint:

| target | algorithm | selected tokens | attention mass | relL2 | cosine | exactKVMB/query | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| mass >= 0.98 | `top_mass_oracle` | 61,363 | 0.980001 | 0.014649 | 0.999900 | 29.963 | current mass lower bound |
| relL2 <= 0.014649 | `top_prob_oracle` | 61,696 | 0.980235 | 0.014528 | 0.999902 | 30.125 | probability ranking gives essentially the same cost |
| relL2 <= 0.014649 | `rel_l2_contribution_oracle` | 53,504 | 0.972383 | 0.014555 | 0.999895 | 26.125 | value-aware one-shot ranking saves ~13% KV traffic |
| relL2 <= 0.014649 | `rel_l2_greedy_batch_oracle` | 33,024 | 0.902548 | 0.014541 | 0.999895 | 16.125 | offline upper-bound suggests large room if selector becomes output-aware |

Other 128k greedy diagnostic frontiers:

| relL2 target | selected tokens | attention mass | exactKVMB/query |
| ---: | ---: | ---: | ---: |
| 0.031111 | 17,152 | 0.832977 | 8.375 |
| 0.020000 | 25,856 | 0.879113 | 12.625 |
| 0.010000 | 43,264 | 0.922607 | 21.125 |
| 0.005000 | not reached with this greedy pool | 0.952119 | 31.375 |

Conclusion: probability mass is too conservative for output fidelity. A deployable selector cannot use the greedy oracle, but the gap is large enough to justify output-aware selector work: approximate value-aware ranking, residual/tail correction, and learned/deployable output-error predictors.

Negative algorithmic attempts after `sched_v2`:

- Residual-radius bounded PQ selection (`selector_eval_pagedpq_bound_ps8192_s4b6_full_t098_h0_v2`): selector-side lower/upper mass bound is valid but far too conservative, max step `66.599 MB/query`.
- Boundary exact-guard PQ (`selector_eval_pagedpq_guard_boundary_ps5632_s4b6_full_t098_h0_v1`): exact guard around PQ cutoff is valid but degenerates to near-dense selection at 128k, max step `67.052 MB/query`.
- Ignore `selector_eval_pagedpq_bound_*_v1`; those runs used a buggy stop loop that exhausted all candidates.

Next direction should be a cleaner algorithmic change, not further scalar margin/page tuning. Promising options: learn a selector-side calibration from approximate-score statistics without true mass at runtime, or redesign the index so selection quality improves directly (for example multi-resolution/product-code routing with a small exact verification budget that does not require near-dense prefix reads).

Active algorithmic test:

- `paged_local_pq_probe_ucb_k<N>_q<PCT>`: page-local PQ full scan plus a charged exact-key probe set. The selector fits a query-local affine calibration from PQ score to exact score, estimates a probe residual quantile, and uses that residual as a conservative upper/lower confidence term in the mass stop rule. This is intended to replace hand-fit decode-length margins with runtime-observable uncertainty.
- Result: valid but too conservative. At 128k, `k128_q90` reaches mass `0.990351` but costs `42.089 MB/query`; `k128_q95` reaches mass `0.992028` and costs `44.341 MB/query`. The per-token worst-case bound overpays, so this should not be the active path.
- Current Slurm run: `49577730` completed cleanly.
- Output: `attention_efficiency_result/selector_eval_pagedpq_probe_ucb_ps5632_s4b6_full_t098_h0_v1`
- Compared selectors: `top_mass_oracle`, `paged_local_pq_probe_k128`, `paged_local_pq_probe_ucb_k128_q90`, `paged_local_pq_probe_ucb_k128_q95`, `paged_local_pq_probe_ucb_k256_q95`, `paged_local_pq_approx_sched_v2`

New active algorithmic test:

- `paged_local_pq_probe_ratio_k<N>`: same charged exact probes, but estimates aggregate exact/approx weight ratios for selected-prefix and tail regions instead of applying a per-token residual bound. This should be much less conservative and is the cleaner version of probe-calibrated stopping.
- Result: `k256` is full-suite valid but worse than the current best (`34.072 MB/query` at 128k). `k128` is closer (`33.574 MB/query`) but misses the 500-token point with mass `0.979632`. Probe-ratio is useful for deployable stopping but is not enough by itself.
- Current Slurm run: `49578390` completed cleanly.
- Output: `attention_efficiency_result/selector_eval_pagedpq_probe_ratio_ps5632_s4b6_full_t098_h0_v1`

New active score-quality test:

- `paged_local_pq_resid_r<N>`: page-local PQ full scan plus exact residual correction on the top-`N` query channels. Cost model charges normal PQ codebook/code reads plus `N * tokens * score_key_bytes` for exact key-channel reads. This tests whether we can keep PQ's cheap full scan while borrowing SparQ's strongest signal only where it most reduces PQ score error.
- `paged_local_pq_resid_probe_ratio_r<N>_k<M>` combines the residual scorer with aggregate probe-ratio stopping.
- Result before exact-channel cost correction: score quality improved (`r8` exactKV `30.147 MB` at 128k), but selector traffic dominated. `r2` nearly matched current best cost (`32.618 MB`) but missed mass (`0.979571` at 128k and lower at short decodes).
- Cost-model correction: exact key probes/residual channels should be charged at KV key precision, not PQ codebook precision. Re-running residual-SparQ with that fix.
- Result after correction (`selector_eval_pagedpq_residual_sparq_ps5632_s4b6_full_t098_h0_v2`): full-context residual correction improves ranking but not enough. `paged_local_pq_resid_r2` reaches `32.124 MB/query` at 128k but misses mass (`0.979571`); `paged_local_pq_resid_probe_ratio_r2_k128_z0p5` is full-suite valid but costs `32.698 MB/query`, worse than `sched_v2`.
- Current Slurm run: `49579294` completed cleanly.
- Output: `attention_efficiency_result/selector_eval_pagedpq_residual_sparq_ps5632_s4b6_full_t098_h0_v2`

Selective residual-pool test:

- `paged_local_pq_resid_pool_r<N>`: apply exact top-query-channel residual correction only to the PQ prefix whose approximate mass reaches target, not to the full context.
- Result (`selector_eval_pagedpq_residual_pool_ps5632_s4b6_full_t098_h0_v1`): lower selector traffic, but selection misses mass. Best endpoint-like result `paged_local_pq_resid_pool_probe_ratio_r2_k128_z0p5` costs `32.577 MB/query` at 128k but is invalid (`min_mass=0.978305`, mass `0.979865` at 128k). Not a valid replacement.

Next direction:

- Probe-ratio calibration is too global: PQ/residual score bias varies by rank. Implement rank-band probe calibration so the selector estimates exact/approx weight ratio per rank band, then stops from band-corrected mass. This is a cleaner deployable stop rule than decode-length margin schedules and should avoid the over/under-selection seen with global ratio calibration.

Active rank-band calibration test:

- `paged_local_pq_probe_bands_k<N>_b<B>_z<Z>`: charged exact probes are split by rank band; each band gets its own exact/approx weight ratio and confidence interval.
- Residual variants: `paged_local_pq_resid_probe_bands_r<R>_k<N>_b<B>_z<Z>` and `paged_local_pq_resid_pool_probe_bands_r<R>_k<N>_b<B>_z<Z>`.
- Result: completed but not a replacement. Best valid banded variant `paged_local_pq_resid_probe_bands_r2_k128_b8_z0p5` costs `33.461 MB/query` at 128k. Lower-cost variants miss mass; e.g. `paged_local_pq_resid_pool_probe_bands_r2_k128_b8_z0` costs `32.115 MB/query` but min mass is `0.975853`.
- Current Slurm run: `49582611` completed cleanly.
- Output: `attention_efficiency_result/selector_eval_pagedpq_probe_bands_ps5632_s4b6_full_t098_h0_v1`

Next direction:

- The failure mode is now cutoff ranking quality, not whole-list mass estimation. Try boundary residual refinement: use PQ full scan to find the approximate cutoff, then read only top-query exact K channels for a window around that cutoff and rerank that boundary region. This targets false negatives just below the cutoff without paying residual correction over the full prefix or full context.

Active boundary-refinement test:

- `paged_local_pq_resid_window_r<R>_w<W>`: PQ full scan plus exact top-query-channel residual only in a rank window around the approximate PQ cutoff.
- `paged_local_pq_resid_window_probe_bands_r<R>_w<W>_k<N>_b<B>_z<Z>` combines boundary residual refinement with banded probe stopping.
- Result: completed, not a replacement. Best valid boundary variant `paged_local_pq_resid_window_probe_bands_r2_w8192_k128_b8_z0p5` costs `33.667 MB/query` at 128k. Lower-cost variants miss mass; e.g. `w8192_z0` costs `32.774 MB/query` but min mass is `0.978751`.
- Current Slurm run: `49583196` completed cleanly.
- Output: `attention_efficiency_result/selector_eval_pagedpq_residual_window_ps5632_s4b6_full_t098_h0_v1`

Active index-quality test:

- `paged_local_pq_varperm_*`: page-local PQ with variance-balanced dimension assignment before product quantization. This tries to reduce PQ score error without extra token-level selector reads; selector cost only adds small per-page permutation metadata reads/writes.
- Result: negative. Variance-balanced permutation worsened ranking; `paged_local_pq_varperm_approx` reaches only `0.978179` mass at 128k while costing `35.665 MB/query`, and banded-probe variants are also invalid/expensive.
- Current Slurm run: `49583633` completed cleanly.
- Output: `attention_efficiency_result/selector_eval_pagedpq_varperm_ps5632_s4b6_full_t098_h0_v1`

Active deterministic permutation check:

- `paged_local_pq_interleave_*`: deterministic interleaving of dimensions across PQ subvectors. This is a cheaper/less data-dependent alternative to variance-balanced permutation.
- Result: negative. Interleaving also worsens ranking; `paged_local_pq_interleave_approx` costs `36.272 MB/query` at 128k and misses mass (`0.977402`), and banded-probe variants are invalid/expensive.
- Current Slurm run: `49584494` completed cleanly.
- Output: `attention_efficiency_result/selector_eval_pagedpq_interleave_ps5632_s4b6_full_t098_h0_v1`

Next direction:

- Exact key-only boundary verification: use PQ to find the cutoff, read full exact K only for a narrow boundary window, rerank that window by true QK score, and keep final attention KV reads separate. This is a classic ANN verify step and may recover cutoff false negatives more reliably than top-dim residual correction.

Active exact-verification test:

- `paged_local_pq_verify_window_w<W>`: PQ full scan, exact-key verify/rerank only a boundary window around the approximate cutoff.
- `paged_local_pq_verify_window_probe_bands_w<W>_k<N>_b<B>_z<Z>` combines exact boundary verification with rank-banded probe stopping.
- Result: negative. Exact boundary verification improves ranking but selector key-read overhead is too high. No tested variant is full-suite valid below baseline; e.g. `w4096_z0p5` still misses mass (`min_mass=0.979160`) while costing `33.488 MB/query` at 128k.
- Current Slurm run: `49584771` completed cleanly.
- Output: `attention_efficiency_result/selector_eval_pagedpq_verify_window_ps5632_s4b6_full_t098_h0_v1`

Next direction:

- Low-dimensional projection verification: maintain a small random/JL projection sidecar of K. Around the PQ cutoff, rerank by projected QK instead of full exact QK. This is between top-dim residual and exact-key verification: more isotropic than SparQ/top-dim, much cheaper than reading full K.

Active projection-verification test:

- `paged_local_pq_proj_window_d<D>_w<W>`: PQ full scan plus projected-QK rerank in a boundary window using a random sign projection sidecar.
- `paged_local_pq_proj_window_probe_bands_d<D>_w<W>_k<N>_b<B>_z<Z>` combines projected boundary rerank with banded probe stopping.
- Current Slurm run: `49585129`
- Output: `attention_efficiency_result/selector_eval_pagedpq_proj_window_ps5632_s4b6_full_t098_h0_v1`

Recent negative / diagnostic attempts:

- `gated_paged_pq_online`: consistently worse than page-local PQ because router overhead did not reduce exact K/V enough.
- SparQ-rerank over paged PQ: recovered SparQ-like selected tokens but paid both PQ and SparQ selector traffic.
- Page-SparQ and postings gates: did not beat page-local PQ; restrictive postings missed mass, permissive postings overpaid.
- PQ shape sweep: page `3072`, `s4b6` is the best frontier so far; `s4b7` lowered exact K/V but selector traffic dominated.

### Snapshot / Query-Only At 128k

This table answers: if the index already exists, how much memory traffic does one query need?

| algorithm | mass | FN | FP | cos | relL2 | selectorMB/query | exactKVMB/query | stepMB/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_mass_oracle | 0.980001 | 0.000000 | 0.000000 | 0.999900 | 0.014652 | 0.000 | 29.963 | 29.963 |
| paged_local_pq_snapshot | 0.980000 | 0.009994 | 0.009994 | 0.999979 | 0.006494 | 2.285 | 37.628 | 39.914 |
| gated_paged_pq_snapshot | 0.980000 | 0.009994 | 0.009994 | 0.999979 | 0.006494 | 2.543 | 37.628 | 40.172 |
| sparq_r16 | 0.980000 | 0.005924 | 0.005924 | 0.999881 | 0.015875 | 8.214 | 33.608 | 41.822 |
| retroinfer_style | 0.980077 | 0.012717 | 0.012793 | 0.999986 | 0.005225 | 0.132 | 42.589 | 42.721 |
| pqcache_full_scan_snapshot | 0.980006 | 0.012334 | 0.012339 | 0.999983 | 0.006404 | 0.288 | 43.396 | 43.684 |
| magicpig_k10_l150 | 0.304605 | 0.675503 | 0.000108 | 0.982046 | 0.590283 | 77.905 | 1.151 | 79.057 |
| retrievalattention_graph | 0.319170 | 0.660831 | 0.000000 | 0.977670 | 0.629017 | 1.176 | 1.125 | 2.301 |

Do not over-interpret `retrievalattention_graph` here: its selector/exact traffic is low because it fails to reach the target mass.

### Online / Realistic At 128k

| algorithm | mode | update modeled | mass | FN | FP | cos | relL2 | selectorMB/query | exactKVMB/query | updateCumMB | updateMB/token | stepMB/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparq_r16 | snapshot | False | 0.980000 | 0.005924 | 0.005924 | 0.999881 | 0.015875 | 8.214 | 33.608 | 0.000 | 0.000000 | 41.822 |
| paged_local_pq_online | online | True | 0.980000 | 0.009994 | 0.009994 | 0.999979 | 0.006494 | 2.285 | 37.628 | 33.180 | 0.000259 | 39.914 |
| gated_paged_pq_online | online | True | 0.980000 | 0.009994 | 0.009994 | 0.999979 | 0.006494 | 2.543 | 37.628 | 35.400 | 0.000277 | 40.172 |
| retroinfer_online_proxy | online_proxy | True | 0.980077 | 0.012717 | 0.012793 | 0.999986 | 0.005225 | 0.132 | 42.589 | 31.987 | 0.000250 | 42.722 |
| pqcache_full_scan_online_proxy | online_proxy | True | 0.980006 | 0.012334 | 0.012339 | 0.999983 | 0.006404 | 0.288 | 43.396 | 33.145 | 0.000259 | 43.684 |
| ivfpq_periodic_rebuild | online_proxy | True | 0.980004 | 0.015248 | 0.015252 | 0.999987 | 0.008982 | 0.864 | 51.078 | 215261.542 | 1.681731 | 53.624 |

Online/realistic interpretation:

- Gated/local paged PQ should be compared with `stepMB/query` when online page maintenance is modeled.
- RetroInfer online should be read as `retroinfer_online_proxy`: update memory is charged, but wave-buffer/cache behavior is still not faithful.
- SparQ has no persistent index update in this proxy, so `stepMB/query = selectorMB/query + exactKVMB/query`; this is fair for this proxy but does not capture prefill-side cost.
- PQCache full-scan online cost is still a framework-port proxy, not the optimized implementation.

### Compression-Side Frontier

Active framing: selection lowers how many tokens are touched; compression lowers bytes per touched token. The primary quality metric here is attention-output error (`output_relative_L2`, plus cosine), not attention mass.

Current exact-head frontier before compression:

| selector + estimator | full-suite mean max relL2 | worst-seed max relL2 | mean stepMB/query | note |
| --- | ---: | ---: | ---: | --- |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s4096` | 0.012278 | 0.014454 | 6.851 | exact selected K/V + stratified tail |
| `gated_paged_pq_budget_k8192+strat_exp_tail_b8_s4096` | 0.015981 | 0.023276 | 8.938 | more head tokens, but worse seed robustness in this run |

Completed compression tests:

| selector + estimator | full-suite mean max relL2 | worst-seed max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | --- |
| `k4096+vpq_after_d32000_strat_exp_tail_b8_s4096` | 0.013644 | 0.014965 | 6.655 | V-PQ alone gives a small MB win with slight error increase |
| `k4096+kbandcalib_after_d64000_p1024x8_tail_b8_s4096` | 0.013740 | 0.014563 | 6.655 | compressed K with rank-band calibration roughly ties exact-head quality at lower MB |
| `k4096+kbandvpq_after_d64000_p1024x8_tail_b8_s4096` | 0.014353 | 0.015815 | 6.655 | compressing calibrated K and all selected V is near-valid but slightly worse |
| `k8192+kbandcalib_after_d64000_p1024x8_tail_b8_s4096` | 0.010180 | 0.011945 | 8.707 | strong high-quality point; K calibration helps more than V-PQ |
| `k8192+kbandvmix_after_d64000_p1024x8_e2048_tail_b8_s4096` | 0.009545 | 0.010187 | 8.707 | best completed high-quality compression result; exact V for top selected ranks, V-PQ for lower ranks |
| `k8192+kbandvpq_after_d64000_p1024x8_tail_b8_s4096` | 0.026924 | 0.034876 | 8.707 | all-VPQ selected V is too aggressive at this budget |

Endpoint-only refinement at 128k/head0:

| selector + estimator | relL2 | stepMB/query | note |
| --- | ---: | ---: | --- |
| `k4096+kbandvmix_after_d64000_p512x8_e1024_tail_b8_s4096` | 0.011277 | 5.312 | endpoint looked good, but full-suite worst relL2 rose to 0.025596 |
| `k4096+kbandvmix_after_d64000_p512x8_e512_tail_b8_s4096` | 0.014116 | 5.191 | full-suite worst relL2 0.022772, not a clean replacement |
| `k8192+kbandvmix_after_d64000_p1024x8_e1536_tail_b8_s4096` | 0.009973 | 5.739 | endpoint good, but full-suite worst relL2 0.021565 |
| `k8192+kbandvmix_after_d64000_p1024x8_e2048_tail_b8_s4096` | 0.008197 | 5.860 | best endpoint quality and also best full-suite high-quality compression point |

Threshold sweep:

| selector + estimator | full-suite mean max relL2 | worst-seed max relL2 | mean stepMB/query | conclusion |
| --- | ---: | ---: | ---: | --- |
| `k4096+kbandcalib_after_d32000_p1024x8_tail_b8_s4096` | 0.020501 | 0.026327 | 6.655 | enabling K compression at 32k is too early |
| `k4096+kbandvmix_after_d32000_p512x8_e512_tail_b8_s4096` | 0.028381 | 0.035937 | 6.655 | early mixed K/V compression is worse |
| `k8192+kbandcalib_after_d32000_p1024x8_tail_b8_s4096` | 0.026912 | 0.028420 | 8.707 | early K compression hurts high-quality regime |
| `k8192+kbandvmix_after_d32000_p1024x8_e2048_tail_b8_s4096` | 0.027936 | 0.028696 | 8.707 | early mixed K/V compression is not acceptable |

Interpretation:

- Plain K-PQ logits are not usable by themselves; logit distortion hurts softmax. Rank-band calibration is the most credible K-compression path so far.
- V-PQ is safer than K-PQ because it does not affect token ranking, but all-VPQ selected V can still create output error. Mixed V is better: keep top selected ranks exact, compress lower selected ranks.
- The compression-side frontier is now `calibrated K + mixed exact/PQ V`, not raw PQ reconstruction.
- Enabling compression earlier than 64k is currently not safe; the 32k threshold sweep worsened relL2 substantially.
- Result roots: `attention_efficiency_result/selector_eval_compression_full_v1`, `attention_efficiency_result/selector_eval_compression_refine_full_v1`, and `attention_efficiency_result/selector_eval_compression_threshold_full_v1`.

TurboQuant-inspired proxy:

- Implemented `tqmse/tqprod` compression proxies using fixed sign-Hadamard rotation, scalar Gaussian codebooks, and an optional 1-bit rotated residual correction. This is not a faithful TurboQuant kernel, but it tests the paper's central idea: inner-product-preserving online vector quantization can be better than learned page-local PQ for K scoring.
- Implemented `tqmse_selector_k<B>_budget_<rule>` and `tqprod_selector_k<B>_budget_<rule>` selectors. These rank tokens by TurboQuant-compressed K scores and use a fixed deployable budget; they do not use oracle mass.

Full-suite result at `k4096`, head 0:

| algorithm | seeds | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `tqprod_selector_k3_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.007428 | 0.007883 | 12.853 | best quality, but full-scan selectorMB is too high |
| `tqprod_selector_k4_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.008054 | 0.008740 | 14.906 | more bits do not beat 3-bit enough to justify cost |
| `tqmse_selector_k3_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.009274 | 0.011509 | 10.542 | good quality, still high selector traffic |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.012278 | 0.014454 | 6.851 | current low-MB frontier remains better on cost |
| `gated_paged_pq_budget_k4096+tqprod_after_d64000_k3v3_tail_b8_s4096` | 1 | 0.057209 | 0.057209 | 6.655 | selected K/V TurboQuant proxy is not accurate enough |

Interpretation:

- TurboQuant-style K scores are a better selector signal than our current paged-PQ scores, but the current implementation scans all compressed K codes, so the selector traffic dominates.
- The useful next direction is not "use TurboQuant reconstruction for selected exact attention"; that failed. The useful direction is "use TurboQuant as the first-stage scoring/index representation, then avoid full scan."
- Candidate follow-up: IVF/TurboQuant or bucketed TurboQuant, where coarse routing limits how many TurboQuant codes are scanned. This keeps TurboQuant's online/no-kmeans benefit while attacking its full-scan selectorMB.
- Result root: `attention_efficiency_result/selector_eval_turboquant_full_v1`.

IVF/TurboQuant selector:

- Implemented `ivftqprod_c<C>_k<B>[_m<M>]_budget_<rule>` and `ivftqmse_c<C>_k<B>[_m<M>]_budget_<rule>`.
- Algorithm: score all coarse centroids, visit buckets until the candidate pool reaches `M * budget`, then rank only those bucket members by TurboQuant-compressed K score and select the fixed budget. This is deployable: no oracle mass or dense ranking is used.
- Cost model charges centroid scan, bucket offsets/postings, compressed TQ key sidecar reads, exact selected KV reads, stratified tail reads, and online sidecar/posting writes for generated tokens.

Full-suite result at `k4096`, head 0:

| algorithm | seeds | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `tqprod_selector_k3_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.007428 | 0.007883 | 12.853 | full-scan TQ: excellent quality, too expensive |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.012278 | 0.014454 | 6.851 | current low-MB frontier |
| `ivftqprod_c512_k3_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.018522 | 0.027864 | 5.423 | lower MB, but quality regression at 128k |
| `ivftqprod_c1024_k3_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.020731 | 0.025674 | 5.718 | lower MB, still not robust |
| `ivftqprod_c1024_k3_budget_k8192+strat_exp_tail_b8_s4096` | 3 | 0.017044 | 0.023446 | 9.778 | more selected tokens, too expensive and still worse than full-scan TQ |

Endpoint route-multiplier sweep at 128k:

| algorithm | relL2 | stepMB/query | candidate tokens | interpretation |
| --- | ---: | ---: | ---: | --- |
| `ivftqprod_c512_k3_budget_k4096+strat_exp_tail_b8_s4096` | 0.013396 | 5.383 | 11028 | best endpoint low-cost IVF-TQ point |
| `ivftqprod_c1024_k3_m4_budget_k4096+strat_exp_tail_b8_s4096` | 0.015838 | 6.898 | 25781 | better routing coverage but cost reaches baseline range |
| `ivftqprod_c512_k3_m4_budget_k4096+strat_exp_tail_b8_s4096` | 0.040577 | 6.690 | 30059 | more candidates did not monotonically improve output error |
| `ivftqprod_c512_k3_m8_budget_k4096+strat_exp_tail_b8_s4096` | 0.007075 | 13.868 | 134582 | degenerates to full-scan TQ quality and cost |

Interpretation:

- Bucketed TQ confirms the expected tradeoff: routing can lower selectorMB substantially, but coarse false negatives become the dominant error.
- `ivftqprod_c512_k3_budget_k4096` is an interesting lower-cost point, not a replacement frontier yet.
- Route multiplier alone is not a robust fix; once it recovers full-scan quality, it has effectively scanned the full context.
- Next plausible route is better coarse routing, not just more buckets: multi-assignment/coarse residual routing, query-adaptive fallback buckets, or learned/hash buckets aligned to TQ score residual rather than K-space centroid similarity.
- Result roots: `attention_efficiency_result/selector_eval_ivftq_full_v1` and `attention_efficiency_result/selector_eval_ivftq_mult_smoke_v1`.

Faithful TurboQuant scoring:

- Downloaded and inspected the `fastvq` PyPI implementation under `third_party/fastvq_pkg`; useful as a reference, but it stores radii and uses simplified/uniform angle quantization, so it is not a strict paper-faithful scorer.
- Implemented a paper-style scorer directly:
  - random sign + Hadamard rotation,
  - Lloyd-Max scalar codebook for the Beta coordinate distribution on the unit sphere,
  - optional QJL-style residual inner-product correction,
  - deployable fixed-budget selector names `papertqmse_selector_k<B>_budget_<rule>` and `papertqprod_selector_k<B>_budget_<rule>`.

Full-scan faithful scorer result:

| algorithm | seeds | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `tqprod_selector_k3_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.007428 | 0.007883 | 12.853 | old proxy remains best full-scan quality/cost |
| `papertqmse_selector_k4_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.008031 | 0.008776 | 12.596 | faithful MSE scoring is strong but still full-scan expensive |
| `papertqprod_selector_k4_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.008713 | 0.008949 | 14.906 | QJL correction did not improve enough to justify cost |
| `papertqmse_selector_k3_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.012873 | 0.014784 | 10.542 | quality near current baseline, cost much higher |
| `papertqmse_selector_k2_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.022223 | 0.024898 | 8.489 | cheaper full-scan TQ, but not good enough |

Faithful IVF-TQ and multi-assignment:

- Added `ivfpapertqmse_c<C>[_r<R>]_k<B>[_m<M>]_budget_<rule>` and `ivfpapertqprod_*`.
- `r<R>` means each token is inserted into its top-`R` coarse buckets. This targets coarse-routing false negatives at the cost of extra postings.

| algorithm | seeds | mean max relL2 | worst max relL2 | mean stepMB/query | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `gated_paged_pq_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.012278 | 0.014454 | 6.851 | current low-error baseline |
| `ivfpapertqmse_c512_r2_k3_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.019155 | 0.022051 | 5.301 | best lower-MB faithful routed point; quality regression remains |
| `ivfpapertqmse_c512_r2_k4_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.019235 | 0.022178 | 5.499 | more bits do not fix routing error |
| `ivfpapertqmse_c512_k3_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.022665 | 0.027810 | 5.208 | single-assignment routing is worse |
| `ivfpapertqmse_c512_r4_k3_budget_k4096+strat_exp_tail_b8_s4096` | 3 | 0.025315 | 0.030434 | 5.078 | too many replicas caused earlier/smaller nprobe choices and worse recall |

Interpretation:

- Faithful TQ scoring is good, but full-scan TQ is still too expensive.
- IVF/bucketed TQ gets the cost down, but routing false negatives dominate. Multi-assignment helps versus single-assignment, but not enough to replace the current baseline.
- The best new point is a cost-quality tradeoff, not a strict win: `5.301 MB/query` at worst relL2 `0.022051`.
- The next nontrivial idea should change routing quality, not only code quality: e.g. query-adaptive secondary buckets, two independent coarse partitions, or a small full-scan TQ rescue set merged with IVF candidates.
- Result roots: `attention_efficiency_result/selector_eval_papertq_full_v1`, `attention_efficiency_result/selector_eval_ivfpapertq_full_v1`, and `attention_efficiency_result/selector_eval_ivfpapertq_replicas_full_v1`.

## RULER GPU Integration Status

## Current Decode-Only Frontier Status (2026-05-17 Night)

Current objective:

- Make the real HF/RULER/public-benchmark path benchmark-ready for the frontier algorithm in **dense prefill + approximate decode** mode.
- During prefill, do normal dense attention and build decode sidecars only: page-local K-PQ, optional V-PQ, static prefix/suffix metadata, and online append state.
- During decode, use the frontier GPU selector/attention path: paged K-PQ selection, selected exact K, mixed exact/compressed V, and V-PQ tail estimation.
- Do not approximate prefill by default. Sparse per-prefill-query selection destroys dense matmul reuse and is not the algorithmic target for long decode.
- Preserve task score while reporting unit-explicit bandwidth, trace/output diagnostics, and wall-clock runtime well enough to decide whether full validation is practical.

Latest implementation changes:

- `approx_prefill` now defaults false in LongBench-v2, public long-decode, and RULER wrappers.
- Dense prefill now populates exact HF KV cache, then builds decode sidecars from the cache.
- Removed `torch.cuda.empty_cache()` from the native decode hot path unless `--debug_empty_cache_native` is explicitly set.
- Added persistent GQA V-PQ pack caching so per-token decode no longer re-stacks V-PQ codebooks/codes when the page set is unchanged.
- Added timing fields: `qkv_cache_seconds`, `index_sidecar_seconds`, `native_pack_seconds`, and `output_projection_seconds`.
- CUDA unit tests passed after the V-PQ pack-cache change: Slurm `50380829`, `cuda_unit_result/frontier_cuda_unit_tests_vpackcache_20260517_230014`.

Latest decode-only runtime points:

| run | decode tokens | avg generation sec | dense sec | total MB/head-query | selector MB | exact KV MB | tail MB | selected tokens | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense LongGen SGT-short | 1024 | 43.846 | 43.846 | n/a | n/a | n/a | n/a | n/a | previous dense reference |
| pagedPQ before hot-path cache cleanup | 1024 | 135.245 | 43.846 | 0.367 | 0.080 | 0.277 | 0.010 | 567.1 | previous profiled frontier run |
| pagedPQ after hot-path cache cleanup | 1024 | 81.701 | 43.846 | 0.367 | 0.080 | 0.277 | 0.010 | 567.1 | `public_longdecode_result/ppq_vpackcache_runtime1024_20260517_225952` |
| pagedPQ after hot-path cache cleanup, repro | 1024 | 111.081 | 43.846 | 0.367 | 0.080 | 0.277 | 0.010 | 567.1 | `public_longdecode_result/ppq_vpackcache_runtime1024_repro_20260517_231453`; same cost, different node/runtime |
| pagedPQ after hot-path cache cleanup, cost stats disabled | 1024 | 101.710 | 43.846 | n/a | n/a | n/a | n/a | n/a | `public_longdecode_result/ppq_vpackcache_nostats1024_20260517_232256`; no meaningful speed win, cost summary intentionally incomplete |
| pagedPQ with fixed-confidence bypass | 1024 | 76.107 | 43.846 | 0.367 | 0.080 | 0.277 | 0.010 | 567.1 | `public_longdecode_result/ppq_fixedconf_bypass1024_20260517_232900`; same fixed-budget algorithm, skips dead adaptive-confidence math |
| dense LongGen SGT-short | 4096 | 174.567 | 174.567 | n/a | n/a | n/a | n/a | n/a | previous dense reference |
| pagedPQ before hot-path cache cleanup | 4096 | 456.109 | 174.567 | 0.589 | n/a | n/a | n/a | 573.4 | previous profiled frontier run |
| pagedPQ after hot-path cache cleanup | 4096 | 347.545 | 174.567 | 0.589 | 0.273 | 0.280 | 0.035 | 573.4 | `public_longdecode_result/ppq_vpackcache_runtime4096_20260517_230439` |

Timing diagnosis:

- The 256-token profile crosses the first sealed page and exercises the native path: `public_longdecode_result/ppq_vpackcache_profile256_20260517_230306`.
- Profile totals across 36 layers: patched attention `30.816s`, QKV/cache `4.452s`, output projection `1.061s`, sidecar maintenance `1.185s`, native pack `0.970s`, native selector `0.454s`, native attention `2.400s`.
- Interpretation: native selector/attention kernels are not the dominant wall-clock cost at short/medium decode. The remaining gap is mostly Python/orchestration/kernel-launch/synchronization overhead around many per-layer per-token calls.

Current conclusion:

- The decode-only algorithmic cost remains attractive: about `0.37 MB/head-query` at 1024 generated tokens and `0.59 MB/head-query` at 4096 generated tokens.
- Runtime is improved but still not benchmark-ready for broad validation: current best 1024-token decode-only smoke is `76.107s`, about `1.74x` dense; 4096-token decode-only smoke is `347.545s`, about `2.0x` dense.
- Greedy outputs are not bitwise reproducible across the 1024 after-cache-cleanup runs (`common_prefix_chars=852`), although modeled costs match exactly. Treat LongGen single-sample text/substring metrics as smoke-only until deterministic kernels/settings are audited.
- Disabling per-token cost accounting did not produce a clear runtime win (`101.710s` on one 1024-token run), so Python accounting is not the primary remaining bottleneck. Use `DISABLE_COST_STATS=1` only for task-quality sweeps where cost metrics are already known, not for cost reporting.
- Bypassing adaptive-confidence math when `min_budget == max_budget == budget` is valid and useful. It preserves cost/quality for fixed-budget runs and is now part of the active path.
- Next optimization should not focus on the existing native selector kernel alone. The higher-leverage target is fusing more of the decode step or reducing Python/kernel-launch orchestration around selector + selected/tail attention.

## Previous Prefill+Decode Findings

Latest dense and active-selector smoke points:

| run | ctx | samples | score | stream s/sample | prefill s | decode s | total MB/head-query | selector MB | exact KV MB | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dense_batched_ctx8192_n4_veccompare` | 8192 | 4 | 100 | 8.335 | 0.731 | 7.604 | n/a | n/a | n/a | dense reference |
| `pagedpq_warptiled_lut_ctx8192_ps512_b8_n4` | 8192 | 4 | 100 | 29.602 | 21.828 | 7.774 | 0.688 | 0.442 | 0.246 | current quality-preserving active path, still 3.55x slower than dense |
| `pagedpq_warptiled_lut_ctx8192_ps512_b8` | 8192 | 1 | 100 | 30.338 | 22.692 | 7.646 | 0.688 | 0.442 | 0.246 | same setting, single-sample |
| `pagedpq_stridebatched_ctx8192_ps512_b8_s8` | 8192 | 1 | 0 | 24.658 | 15.928 | 8.730 | 0.315 | 0.068 | 0.246 | selector reuse still lost task quality even at stride 8 |
| `pagedpq_stridebatched_ctx8192_ps512_b8_s64` | 8192 | 1 | 0 | 24.325 | 15.631 | 8.695 | 0.268 | 0.022 | 0.246 | selector reuse cut MB/runtime but lost task quality |
| `pagedpq_nativefused_ctx8192_ps512_b8` | 8192 | 1 | 100 | 40.047 | 31.332 | 8.715 | 0.688 | 0.442 | 0.246 | fused small-budget selector is correct but slower than Torch LUT |
| `pagedpq_nativefused_ctx4096_ps256_b8` | 4096 | 1 | 100 | 22.011 | 13.854 | 8.157 | 0.564 | 0.382 | 0.182 | also slower than Torch LUT baseline |
| `dense_batched_ctx16384_n1_veccompare` | 16384 | 1 | 100 | 15.105 | 4.258 | 10.848 | n/a | n/a | n/a | dense reference at higher context |
| `pagedpq_warptiled_lut_ctx16384_ps512_b8` | 16384 | 1 | 100 | 84.388 | 75.992 | 8.396 | 1.182 | 0.932 | 0.249 | quality OK, prefill runtime not practical |
| `pagedpq_warptiled_lut_ctx16384_ps1024_b8` | 16384 | 1 | 100 | 86.654 | 78.546 | 8.108 | 0.835 | 0.464 | 0.371 | lower selector MB but higher exact suffix; runtime still dominated by sparse prefill |

Implementation status:

- `benchmark/selector_eval/cuda_ext` now exposes `gqa_causal_fullscan_pq_topk_fused`.
- Parity after the fused-selector change passed: `selector_fused_parity4` / Slurm `50238991`.
- The fused selector avoids the full score matrix and `at::topk`, but the first row-block implementation is slower in full RULER smoke. Do not take it forward as the active runtime path without a different kernel design.
- Selector stride reuse is bandwidth-attractive but not currently deployable. `stride=64` at ctx4096 kept score 100, but ctx8192 collapsed to 0; ctx8192 `stride=8` also collapsed to 0. Pending `stride=16/32` were cancelled after `stride=8` failed.
- V-PQ selected-value and V-PQ tail paths pass CUDA unit/parity coverage, but the current all-layer RULER integration is not a runtime/quality frontier. Failed/cancelled or rejected smokes:
  - `pagedpq_vpqtail_ctx2048_ps128_b8`: combined selected V-PQ + tail fell to the slow Python path and was cancelled after no sample progress.
  - `pagedpq_vpqtail_ctx512_ps64_b8`: same combined path was still too slow at ctx512.
  - `pagedpq_tailnative_ctx2048_ps128_b8`: native tail with exact selected V was cancelled after no sample progress.
  - `pagedpq_selectedvpq_ctx2048_ps128_b8`: native selected V-PQ before the warp-tiled/no-exact-load fix was cancelled after no sample progress.
  - `pagedpq_selectedvpq_warp_ctx2048_ps128_b8`: 6-bit all-compressed page-local V-PQ ran but failed task quality.
  - `pagedpq_selectedvpq_warp_ctx2048_ps128_b8_v8_noexactload`: 8-bit all-compressed page-local V-PQ recovered task quality but cost/runtime were worse than exact selected.
  - `pagedpq_selectedvpq_warp_ctx2048_ps256_b8_v7_noexactload`: 7-bit all-compressed page-local V-PQ recovered task quality and was faster than the ps256 exact-V control, but modeled MB was higher (`0.415` vs `0.336 MB/head-query`).
  - `pagedpq_selectedvpq_warp_ctx2048_ps512_b8_v7_noexactload`: page-size 512 reduced modeled MB to `0.324 MB/head-query`, but task quality collapsed.
  - `pagedpq_selectedvpq_warp_ctx2048_kps128_vg2_b8_v7_mem64`: grouping V codebooks across two K pages preserved quality and reduced MB versus ps128 page-local V-PQ (`0.576` vs `0.670`), but did not beat ps256 V-PQ or exact selected.
  - `pagedpq_selectedvpq_warp_ctx2048_kps128_vg4_b8_v7_mem64`: grouping across four K pages reduced MB further but failed quality.
- Treat compression/tail as implemented at kernel/proxy level, with limited RULER functional coverage. The only currently useful RULER frontier remains selector + exact selected K/V attention.
- Current compression diagnosis: page-local/grouped V-PQ is not yet a modeled-MB frontier. Increasing V page/group size reduces codebook traffic but quickly hurts quality. It may still help runtime by avoiding exact V gathers. The next plausible value-compression direction is a native mixed exact/compressed V rule rather than all-compressed V.

Current bottleneck:

- The quality-preserving active path is still too slow for broad validation: ctx8192 active prefill is about `21.8s` vs dense prefill `0.73s`.
- At ctx16384, the gap worsens: active prefill is about `76-79s` vs dense prefill `4.26s`.
- The selector and selected-attention kernels are GPU-resident, but the prefill algorithm still does per-query sparse selection/attention work and does not yet approach FlashAttention-class dense prefill runtime.

## Archived / Older Tables

## GPU Selector Prototype Status

Implemented:

- `benchmark/selector_eval/gpu/run_gpu_paged_pq_eval.py`
- `benchmark/selector_eval/gpu/print_gpu_summary.py`

Scope:

- PyTorch GPU prototype for page-local PQ selector plus exact selected-head attention and stratified tail estimation.
- Modes: `fullscan` and `routed`.
- Online realism: page-local PQ sidecar is built from the saved trace; this is a selector/attention GPU proxy, not a full model decode kernel.
- Cost accounting reports selector MB, exact K/V MB, tail-estimator MB, and step MB separately in `samples.csv` / `summary.csv`.
- Timings report median steady-state GPU query time after warmup.

Important implementation fixes:

- Partial pending page is no longer sealed into PQ; it remains exact/pending, matching the CPU online semantics.
- Tail estimator uses stratum-scaled numerator/denominator estimation instead of treating sampled tail tokens as normal softmax tokens.
- Sparse attention timing now computes selected/tail QK logits on GPU instead of reusing precomputed dense logits.
- Added deterministic `--tail_sampling linspace` to reduce random tail-estimator variance.

GPU smoke results at 128k, head 0, k4096, tail s4096:

| mode | relL2 | cosine | mass | selectorMB | exactKVMB | tailMB | stepMB | median total ms | median selector ms | median attn ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fullscan | 0.012142 | 0.999970 | 0.758812 | 0.854 | 4.589 | 2.000 | 7.442 | 5.880 | 5.024 | 0.851 |
| routed adaptive | 0.012470 | 0.999985 | 0.698233 | 0.140 | 4.589 | 2.000 | 6.729 | 11.142 | 10.291 | 0.851 |
| routed fixed nprobe 16 | 0.012470 | 0.999985 | 0.698233 | 0.140 | 4.589 | 2.000 | 6.729 | 3.275 | 2.507 | 0.790 |

Interpretation:

- Routed PQ gives the intended algorithmic selector-MB reduction (`0.854 -> 0.140 MB/query`) at similar quality for this head.
- Adaptive nprobe is currently a Python/prototype overhead problem. Fixed nprobe separates algorithmic cost from policy-search overhead and is much faster.
- The PyTorch sparse path is still slower than dense single-head timing on GPU because this is irregular gather-heavy code, not a fused sparse-attention kernel.

Full decode-curve GPU k4096 results over decode `500..128000`, heads `0,8,16,24`, one qidx per decode:

| mode | mean relL2 | max relL2 | min cosine | min mass | mean stepMB | max stepMB | mean median ms | max median ms | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fullscan | 0.012215 | 0.080746 | 0.996772 | 0.733934 | 5.035 | 7.442 | 2.105 | 5.916 | `attention_efficiency_result/gpu_paged_pq_fullcurve_k4096_v1` |
| routed adaptive | 0.014930 | 0.280880 | 0.960261 | 0.666981 | 4.905 | 6.796 | 2.603 | 4.873 | `attention_efficiency_result/gpu_routed_paged_pq_fullcurve_k4096_v1` |

Outlier diagnosis:

- Worst endpoint head24/128k is highly sensitive to tail sampling.
- Fullscan random tail seeds: relL2 ranged from `0.071` to `0.284` at fixed selected mass `0.8825`.
- Routed random tail seeds: relL2 ranged from `0.077` to `0.321` at fixed selected mass `0.8689`.
- Deterministic rank-band sampling improves the catastrophic case: fullscan `0.0658`, routed `0.0798`.
- Increasing deterministic tail samples to `s16384` improves to fullscan `0.0430`, routed `0.0474`, but does not fully solve the weak-head case.
- Full-curve tail-policy comparison shows systematic rank-band sampling is best so far:
  - fullscan random: mean relL2 `0.012215`, max relL2 `0.080746`
  - fullscan linspace: mean relL2 `0.012846`, max relL2 `0.177786`
  - fullscan systematic: mean relL2 `0.008728`, max relL2 `0.072039`
  - routed random: mean relL2 `0.014930`, max relL2 `0.280880`
  - routed linspace: mean relL2 `0.008363`, max relL2 `0.079796`
  - routed systematic: mean relL2 `0.007979`, max relL2 `0.076295`

Current GPU conclusion:

- The GPU prototype validates that routed PQ can reduce selector memory traffic, but the current k4096+s4096 tail estimator is not robust across heads.
- Increasing selected-head budget alone does not fix the weak head: routed k8192+s4096 has max relL2 `0.069904`, while routed k16384+s4096 regresses to max relL2 `0.144104`.
- Increasing systematic tail budget is more effective for the weak head:
  - routed k4096+s4096: mean relL2 `0.007979`, max relL2 `0.076295`, mean/max stepMB `4.905/6.796`
  - routed k4096+s8192: mean relL2 `0.005466`, max relL2 `0.077646`, mean/max stepMB `5.716/8.796`
  - routed k4096+s12288: mean relL2 `0.004606`, max relL2 `0.051174`, mean/max stepMB `6.339/10.796`
  - routed k4096+s16384: mean relL2 `0.003456`, max relL2 `0.047229`, mean/max stepMB `6.912/12.796`
- Current best cost-quality knee in this GPU proxy is routed k4096 + systematic stratified tail s12288: it significantly improves worst-head robustness versus s4096 while staying cheaper than s16384.
- Next algorithmic target is an adaptive tail-budget policy or stronger deployable tail estimator that gets the s12288/s16384 robustness without always paying that tail budget.

Older summaries under `attention_efficiency_result/threeway_*`, `proxy_*`, and `online_ivfpq_*` are useful historical artifacts, but they use older proxy schemas and some pre-cleanup MB accounting. Do not mix them into the main comparison without regenerating under `benchmark/selector_eval`.
