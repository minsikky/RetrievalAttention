# Selector-Eval Latest Results

Keep this page compact. Archive full tables and stale variants instead of appending indefinitely.

## Active Goal

End state: one benchmark-ready canonical GPU implementation of the current CPU frontier decode algorithm. It keeps dense prefill, approximates decode only after sealed PQ pages are active, matches CPU frontier semantics, and is fast enough to run real dense-vs-frontier task benchmarks.

## Streaming Exact-V Read-Union Variant - 2026-06-02

Question: can exact-V selection be made FlashAttention-style by scanning blocks, maintaining a running top-k residual-risk set, immediately reading newly-entered exact V rows, and discarding each probability/logit block without rereading it later?

Implementation/result:

- Added `streaming_global_risk_b<size>` to the CPU trace runner. Semantics are read-union: if a token ever enters the running top-k risk set, its exact-V correction remains in the output even if later blocks evict it.
- The runner now accounts actual exact V reads for streaming rules, not just the active final V budget. A batched prefix-rank helper computes all V budgets for a row and was checked against a brute-force running top-k implementation.
- Representative-head long subset (`heads 0,8,16,24`, decodes `32k,128k`) output: `attention_efficiency_result/joint_kv_streaming_risk_20260602/streaming_risk_long_subset_heads_0_8_16_24_batched`.
- Plot/report: `attention_efficiency_result/joint_kv_streaming_risk_20260602/plots_long_subset_heads_0_8_16_24_batched/local_block_commit_summary.md`.

Headline representative-head results:

| rule | threshold | MB/head-q | mean o-proj relL2 | max o-proj relL2 | active V target | exact V reads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| global residual risk | `0.002` | `11.355` | `0.001639` | `0.001898` | `9655` | `9655` |
| streaming risk b2048 | `0.002` | `14.594` | `0.001466` | `0.001690` | `9655` | `23201` |
| streaming risk b8192 | `0.002` | `13.918` | `0.001477` | `0.001700` | `9655` | `20422` |

Decision: the corrected FlashAttention-style streaming read-union is semantically valid but not a better logical frontier in this proxy. It avoids storing/rereading global P/S for exact-V choice, but it reads many extra V rows because tokens that enter early running top-k are kept even after later eviction. Quality improves slightly, but MB increases materially versus global residual-risk final top-k.

Canonical algorithm semantics to preserve:

- Prefill attention is dense. Prefill may build/update PQ sidecars, but it must not use sparse/frontier attention.
- Decode uses the CPU frontier algorithm: paged K-PQ fullscan selector, raw K-PQ approximate logits, candidate ranking, exact-K refinement, mixed exact-K/K-PQ logits and probabilities, adaptive K/V output-stability confidence, global residual-risk exact-V selection, and V-PQ reconstruction for non-exact V rows.
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
- Optimization and promotion gates must use accounting/profile runs, not `DISABLE_COST_STATS=1` timing-only runs. Report prefill latency, decode/generation latency, profiling overhead, cost-stat overhead, sidecar/update overhead, logical frontier MB, and physical GPU MB in the same artifact.
- After runtime gates pass, run dense-vs-frontier task-quality comparisons on at least one reasoning benchmark, one coding benchmark, and one long-generation benchmark.

Success criteria:

- CUDA unit tests pass for every native helper used by the canonical path.
- Saved-trace CPU-vs-GPU parity passes on real Q/K/V traces at decode lengths `32000`, `64000`, and `128000` on multiple heads.
- Parity covers accepted K/V budgets, selected-token counts, logical MB, attention outputs, and o-proj outputs.
- HF/RULER with `FRONTIER_CANONICAL_GPU=1` exercises the approximation path, not dense fallback, inactive sidecars, or diagnostic shortcuts.
- A primary gate artifact exists with cost stats and profiling enabled; no-stats timing-only runs are not promotion evidence.
- The primary 32k/128 accounting/profile gate reaches `<=54s` decode without semantic changes.
- The sustained 8192-token accounting/profile gate reaches `<=971s` generation time without semantic changes.
- Cost reports separate logical frontier MB from physical GPU execution MB and include selector MB, exact-K/V MB, V-PQ/tail MB, update/sidecar MB, and total step MB/query.
- Profiling shows the decode hot path is native-CUDA dominated and identifies any remaining non-native bottlenecks.
- Benchmark-quality runs report dense and frontier accuracy/score, decode latency, active approximation coverage, logical MB, and physical MB for matched model/task settings.

Constraints / invalid shortcuts:

- Do not replace adaptive confidence with fixed budgets, fixed top-k, selected-mass V, selector-rank exactness, hand-calibrated schedules, context-length-specific knobs, or benchmark-specific knobs and call it canonical.
- Do not use oracle attention probabilities, dense top-k rankings, true achieved mass, relL2 against dense output, task labels, future tokens, generated answers, or post-hoc dense outputs inside selector, compression, confidence, or budget logic.
- Do not compute dense attention output and then mask/prune only for reporting. Dense compute is acceptable only as a clearly labeled diagnostic/simulator baseline, not as the canonical benchmark hot path.
- Do not count physical dense reads as logical sparse reads, and do not hide dense K/V reads, sidecar rebuilds, PQ refreshes, exact probes, PyTorch sort/top-k/gather traffic, online-update work, or calibration/sidecar costs.
- Do not promote an optimization that changes accepted budgets, selected-token statistics, logical MB, or outputs beyond parity tolerance; record it as a separate variant.
- Do not claim benchmark readiness from no-stats/disabled-cost-stat runs, short smokes, single-layer/head traces, inactive approximation paths, runs where sealed PQ pages never activate, or runs that only pass because the context is too short.
- Do not run heavy GPU jobs or extension builds on login nodes; use Slurm `spgpu` with account `zhengya98`.

## 128K GPU Memory Feasibility - 2026-05-27

Question: can the canonical frontier benchmark run at 128K context on smaller GPU partitions if prefill stays dense?

Answer: yes for A40 and MIG40, but only with chunked dense prefill and per-chunk CUDA cache release. This does not reduce the final KV cache; it reduces transient SDPA/prefill workspace and allocator slack.

| device / partition | job | chunk | result | peak allocated | peak reserved | decode-start allocated | score | stream time |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| A40 / `spgpu` | `51022861` | 16K | passed | 43.31 GiB | 44.08 GiB | 31.86 GiB | 100.0 | 256.97s |
| A100 MIG40 / `gpu_mig40` | `51022862` | 8K | passed | 38.07 GiB | 38.91 GiB | 31.86 GiB | 100.0 | 309.26s |
| A40 / `spgpu` | `51022686` | 32K | failed | OOM before final prefill chunk | - | - | - | - |
| A100 MIG40 / `gpu_mig40` | `51022689` | 16K | failed | OOM before final prefill chunk | - | - | - | - |

Runtime/cost from passing runs:

| device | prefill | decode | decode ms/token | logical step MB/head-query | physical GPU step MB/head-query | selected tokens/head-query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A40 | 158.00s | 98.97s | 773.21 | 9.790 | 35.700 | 24823.2 |
| MIG40 | 189.98s | 119.28s | 931.89 | 9.778 | 35.696 | 24785.0 |

Implementation changes:

- `PREFILL_CHUNK_SIZE` now drives dense prefill chunking in the RULER streaming runner.
- Dense-prefill sidecar warming is deferred until the full prompt has been prefetched, avoiding sidecar rebuilds after every prefill chunk.
- `FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1` releases transient prefill workspace between chunks.
- Memory tracing now reports allocated, reserved, peak allocated, peak reserved, free, and total GPU memory at model load, chunk boundaries, sidecar warmup, and decode.

Recommended 128K settings:

- A40: `PREFILL_CHUNK_SIZE=16384 FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1 FRONTIER_EMPTY_CACHE_AFTER_PREFILL=1`.
- MIG40: `PREFILL_CHUNK_SIZE=8192 FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1 FRONTIER_EMPTY_CACHE_AFTER_PREFILL=1`.

## Canonical Frontier Path

Current reference frontier path as of 2026-05-25:

- dense prefill;
- decode-only fullscan paged-PQ selector;
- K-PQ approximate logits for all tokens, ranked token candidates, then exact K logits for selected tokens;
- mixed attention probabilities from exact selected-K logits plus raw K-PQ tail logits (`a=1,b=0`);
- adaptive K and V budgets by output-stability confidence;
- global exact-V selection by residual-risk, `risk_i = p_i^2 * V_PQ_error_stat_i`;
- V-PQ reconstruction for non-exact V rows;
- exact accepted-budget logical accounting separated from physical GPU reads.

Current canonical trace policy as of 2026-05-28:

- K budget grid is relative: `10%,30%,50%,70%,90%,100%`.
- V budget grid is relative: `5%,10%,20%,40%,60%,80%,100%`.
- Initial budget hint is `proxy_mass_m0p9`.
- Confidence threshold is budget-jump-aware: base `0.002`, mode `budget_delta_frac`, shape `sqrt`, reference fraction `0.2`, max scale `1.5`.

Reference trace result: hybrid relative K/V grid, `k_first_alternating`, layer 16/all heads, decode lengths `500..128000`, raw K-PQ tail logits, mean logical cost `4.519 MB/head-query`, mean o-proj relL2 `0.001301`, max o-proj relL2 `0.001761`.

Local block-commit exact-V selection, 2026-06-01: implemented `v_selection_rules=local_block_b<size>` in the joint K/V trace runner and compared against current global residual-risk exact-V selection. The local rule selects exact V independently inside each contiguous block with a proportional quota, so it does not need to retain a global survivor list/logit state before committing V for that block. Full all-head sweep over decode lengths `500..128000`, thresholds `0.0002..0.032`, block sizes `512,1024,2048,4096,8192`, and state-inclusive accounting completed as Slurm job `51252564`. Result: local block commit is not a Pareto improvement. The avoided global survivor state is tiny (`~0.014-0.032 MB/head-query` in this trace), while block-local quotas lose global residual-risk ordering and require substantially more exact V reads. At the canonical `threshold=0.002`, global costs `4.537 MB/head-query` with mean/max o-proj relL2 `0.001301/0.001761`; best local block is `b8192`, costing `5.579 MB/head-query` with `0.001442/0.001963`. Cheapest points satisfying mean relL2 `<=0.002`: global `4.300 MB`, best local `b8192` `5.259 MB`. Artifacts: `attention_efficiency_result/joint_kv_local_block_commit_20260601/local_block_full_allheads_dense_thresholds` and plots under `attention_efficiency_result/joint_kv_local_block_commit_20260601/plots_full_allheads_dense_thresholds`.

V-error-only exact-V selection, 2026-06-02: implemented `v_selection_rules=v_error_only` and `local_v_error_b<size>` to test exact-V ranking by stored V-PQ residual/error metadata alone, without multiplying by attention probability `p_i`. This is attractive for streaming because the exact-V priority is query-independent and does not require keeping/re-reading the full probability row. Full all-head trace sweep over decode lengths `500..128000`, thresholds `0.0002..0.032`, and rules `global_residual_risk,v_error_only,local_v_error_b2048,local_v_error_b8192` completed as Slurm job `51266285`. Result: strongly negative. At `threshold=0.002`, global residual-risk is `4.537 MB/head-query`, mean/max o-proj relL2 `0.001301/0.001761`; `v_error_only` is `6.299 MB/head-query`, `0.01334/0.04560`. Cheapest mean relL2 `<=0.002`: global `4.300 MB`, while global `v_error_only` does not reach the target in the tested sweep; local V-error reaches it only by reading `~27K-28K` exact V rows and costs `11.5-11.9 MB/head-query`. Conclusion: `p_i` is essential for deciding which V reconstruction errors matter. Artifacts: `attention_efficiency_result/joint_kv_v_error_only_20260602/v_error_only_full_allheads_dense_thresholds` and plots under `attention_efficiency_result/joint_kv_v_error_only_20260602/plots_full_allheads_dense_thresholds`.

Page-size sensitivity, 2026-05-28: swept current canonical trace policy over page sizes `512,1024,...,5632` with all heads and decode lengths `500..128000`. Output: `attention_efficiency_result/joint_kv_page_size_sweep_20260528`. Smaller pages increase cost and do not improve output error: `ps512` costs `7.610 MB/head-query` with mean o-proj relL2 `0.001102`, while `ps5632` costs `4.813 MB/head-query` with mean o-proj relL2 `0.001026`. Best mean cost in the sweep is `ps4608` at `4.804 MB/head-query`, but `ps5632` has the best mean o-proj relL2 and remains a sensible canonical setting.

Budget-ladder and initial-trial sensitivity, 2026-05-28:

- Budget-ladder sweep output: `attention_efficiency_result/joint_kv_budget_ladder_sweep_20260528`. Current coarse ladder remains the high-quality point: `4.813 MB/head-query`, mean/max o-proj relL2 `0.001026/0.002083`, mean K/V budgets `12231/2581`, mean iterations `2.02`. Finer ladders expose lower-cost but lower-quality points: `fine_abs` costs `2.999 MB/head-query` with mean/max relL2 `0.004465/0.012873`; `fine_tiny` costs `2.894 MB/head-query` with mean/max relL2 `0.005683/0.014478`. Conclusion: the coarse geometric ladder is conservative, and finer ladders are useful Pareto points, but they are not drop-in high-quality replacements.
- Initial-trial sweep output: `attention_efficiency_result/joint_kv_start_strategy_sweep_20260528`. All-head anchor sweep with the fine-tiny grid shows the expected tradeoff. Starting from `min` gives `2.894 MB/head-query` but poor max relL2 `0.014478` and `9.21` iterations. PQ proxy-mass starts improve quality and reduce iterations with modest cost: `proxy_mass_m0p7` gives `3.134 MB/head-query`, mean/max relL2 `0.002995/0.005254`, and `4.68` iterations. `temporal_prev_low` gives `3.516 MB/head-query`, mean/max relL2 `0.002685/0.005016`, and `2.82` iterations. Raw `temporal_prev` over-reads: `6.195 MB/head-query`, mean/max relL2 `0.000999/0.002579`, and `0.88` iterations.
- Entropy start is a direct flat-vs-spiky PQ-score heuristic. On all heads, `proxy_entropy_f0p25` gives `4.301 MB/head-query`, mean/max relL2 `0.001756/0.002724`; `proxy_entropy_f0p50` over-reads to `5.690 MB/head-query` for mean/max relL2 `0.001079/0.001831`.
- Temporal locality diagnostic over all 288 sampled decode queries on heads `0,8` shows direct previous-budget reuse is too conservative. `temporal_prev` starts near full budget (`start K/V 32182/16205`) and costs `7.972 MB/head-query`. Damped reuse (`temporal_prev_low`) is cheaper (`5.176 MB/head-query`) but still not clearly better than PQ-score proxy starts. `proxy_mass_m0p7` gives `5.251 MB/head-query` with lower max relL2 `0.00920` than `temporal_prev_low` (`0.02101`).
- Relative geometric budget ladder, 2026-05-28: added per-query percentage budget support and tested K fractions `0.2%,0.4%,0.8%,1.6%,3.2%,6.4%,12.8%,25.6%,51.2%,100%`, V fractions `0.1%,0.2%,0.4%,0.8%,1.6%,3.2%,6.4%,12.8%,25.6%,51.2%,100%`, start strategies `min,proxy_mass_m0p9`. Output: `attention_efficiency_result/joint_kv_relative_budget_sweep_20260528`. Relative `proxy_mass_m0p9` is high quality but higher cost: `5.309 MB/head-query`, mean/max o-proj relL2 `0.000930/0.001130`, mean K/V budgets `13121/3523`, mean iterations `1.39`. The relative `min` control costs `4.555 MB/head-query` but has poor mean/max relL2 `0.009053/0.014597`. Conclusion: relative geometric `proxy_mass_m0p9` is not a bandwidth win over current coarse absolute budgets; it mainly buys stricter quality by starting/escalating larger at long contexts.
- Budget-jump-aware confidence thresholds, 2026-05-28: added `threshold_mode=budget_delta_frac`, where the effective stability threshold scales with the fractional budget jump. This directly tests whether small budget increments should require a smaller output delta while large increments can use the base threshold. On the `5%,10%,20%,40%,60%,80%,100%` relative grid with `proxy_mass_m0p9`, threshold sweeps over `0.0005,0.001,0.002,0.004,0.008` show the scaled rules improve the target `4.25-4.75 MB/head-query` region versus fixed threshold. Around `4.4-4.6 MB/head-query`, fixed threshold `0.002` gives `4.363 MB`, mean/max relL2 `0.001690/0.003074`; scaled `sqrt` gives `4.475 MB`, `0.001319/0.001909`; scaled `log` gives `4.546 MB`, `0.001200/0.001878`; scaled `linear` gives `4.614 MB`, `0.001155/0.001836`. Conclusion: budget-jump-aware thresholds are a real quality/MB improvement around the useful operating region, not only a higher-cost quality buy; at the very lowest-cost point near `4.0 MB`, fixed threshold is still competitive/noisy. Plots: `attention_efficiency_result/joint_kv_relative_budget_sweep_20260528/plots_threshold_curves/threshold_scaling_mean_relL2.png` and `attention_efficiency_result/joint_kv_relative_budget_sweep_20260528/plots_threshold_curves/threshold_scaling_max_relL2.png`.
- Grid-shape diagnostic, 2026-05-28: on the current `5-100%` grid with `sqrt` threshold `0.002`, final K uses the `5%` bucket only `11/288` head-queries (`3.8%`), while final V uses `5%` for `121/288` (`42.0%`). Apples-to-apples jobs tested `K/V 10,30,50,70,90,100%` and hybrid `K 10,30,50,70,90,100%` with `V 5,10,20,40,60,80,100%`. Full `K/V 10-100` is worse overall (`4.763 MB`, mean/max relL2 `0.001323/0.001974`) than current `K/V 5-100` (`4.475 MB`, `0.001318/0.001909`), though it uses fewer iterations. Hybrid `K10-100,V5-100` is the better candidate: `4.519 MB`, mean/max relL2 `0.001301/0.001761`, and fewer iterations `0.62` vs current `0.79`. Conclusion: do not remove `5%` from V; removing it from K may be useful.
- Canonicalization update, 2026-05-28: the trace and HF benchmark defaults now use the hybrid relative grid (`K 10-100%, V 5-100%`), `proxy_mass_m0p9`, budget-jump-aware `sqrt` threshold `0.002`, raw/no-affine K-PQ tail logits, and global residual-risk exact V. Relative budgets are derived from each query's current context length. HF runtime falls back from the fixed-threshold native policy kernel to the Torch policy evaluator when scaled thresholds or nonzero proxy starts are enabled; this preserves algorithm semantics while leaving native score-grid/risk/V-prefix helpers active.
- Proxy-start oracle diagnostic, 2026-05-28: added offline-only cheapest-grid oracle reporting to the joint K/V policy runner. On `relative 5-100% + proxy_mass_m0p9 + sqrt-scaled threshold 0.002`, the proxy initializer is well matched to a moderate per-head relL2 target `0.002`: start cost is `0.86x` oracle MB on average, covers both oracle K/V budgets for `52.1%` of head-queries, and the adaptive loop ends at `1.01x` oracle MB with `87.2%` oracle K/V coverage. It is not self-calibrating across quality regimes: for target `0.001`, start/final are too low (`0.72x` / `0.84x` oracle MB); for target `0.004`, start/final overread (`1.04x` / `1.22x` oracle MB). Artifact: `attention_efficiency_result/joint_kv_oracle_budget_policy_20260528/relative_5to100_sqrt_proxy_m0p9_oracle/oracle_policy_report.md`.
- Current conclusion: initial trial should be treated as an online runtime/iteration hint, not as a replacement for adaptive confidence. It can change logical MB only when it overshoots because the loop never moves downward. PQ-score sharpness/proxy-mass is the more deployable start heuristic; temporal warm-start may help if damped, but raw previous-budget reuse is too conservative.

Implementation status: the trace runner `run_joint_kv_budget_policy_eval.py` implements the adaptive K/V residual-risk policy. The HF benchmark wrapper now exposes `online_confidence_rule=joint_kv_stability` plus `selected_value_exact_rule=global_residual_risk`, and `FRONTIER_CANONICAL_GPU=1` requires those settings. CPU-vs-CUDA trace parity smokes passed. The benchmark-facing wrapper defaults now enable the validated native V-prefix helper plus prewarmed persistent V-PQ sidecars, which brings the 32k/4 decode diagnostic inside the `2-3x` dense target. Representative 128-token frontier validation and dense-vs-frontier task slices are still pending.

`FRONTIER_CANONICAL_GPU=1` rejects noncanonical fixed-budget, selector-rank, selected-mass shortcuts, segmented V-prefix, and exact all-head precompute. It now requires native V-prefix plus prewarmed persistent V-PQ sidecars for the benchmark-ready path.

Latest 2026-05-24 status:

- 2026-05-26 current canonical all-head path now passes the accounting/profile runtime gates with cost stats enabled. RULER 32k/128 job `50903294` scored `100.0` and decoded in `43.98s` (`2.44x` dense `17.99s`), with logical `4.0243 MB/head-query`, physical `8.9173 MB/head-query`, and selected `12503.5`. Sustained LongGen8192 job `50903295` generated `8192` tokens in `894.78s` (`2.76x` dense `323.65s`, target `<=971s`), with logical `1.7629 MB/head-query`, physical `1.8716 MB/head-query`, selected `4284.7`, and active fraction `0.359`. Main LongGen8192 wall buckets: score-grid `94.35s`, exact logits `60.47s`, QKV cache `60.79s`, accounting `59.24s`, V-PQ sidecar `24.26s`, prob/base `23.35s`, risk-prefix `17.12s`, rank-prefix `6.82s`. Sparse-direct diagnostic LongGen8192 retry `50903299` generated in `896.56s`, so it has no sustained speed advantage and remains noncanonical because of the prior benchmark-style parity miss.
- 2026-05-26 no-calib score-grid workspace is rejected. CUDA unit `50904418` passed, and native CPU/GPU parity was nearly exact, but benchmark-style Torch/GPU parity still missed tolerance slightly (`~5.8e-4`). RULER 32k/128 `50904609` decoded in `57.83s`, slower than canonical `43.98s`; sustained LongGen8192 `50904613` generated in `1190.48s`, much worse than canonical `894.78s`. Wall buckets regressed broadly: score-grid `137.04s`, exact logits `91.46s`, prob/base `60.94s`, risk-prefix `69.66s`. Keep `SELECTOR_PQ_JOINT_NOCALIB_SCORE_GRID_WORKSPACE=0`.
- 2026-05-26 native accounting accumulation is useful but not promoted yet. CUDA unit `50904656` and explicit accumulator unit `50904682` passed. RULER 32k/128 `50904671` reduced accounting wall time slightly (`~1.00s` scale to `0.88s`) but did not beat the canonical short gate. LongGen8192 `50904672` reduced accounting wall time versus canonical (`59.24s` to `25.62s`), but total generation was slower (`1155.35s`) because unrelated buckets were also slower on that run (`score-grid 121.75s`, exact logits `87.89s`, QKV `74.65s`). Keep `SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING_ACCUMULATE=0` until a paired same-code/same-run validation shows stable total-runtime benefit without hiding stats.
- 2026-05-26 wall-profile-only benchmark mode is positive for runtime gates. RULER 32k/128 job `50904953` kept cost stats and wall profiling enabled but used `PROFILE_NATIVE_OPS=0`, avoiding per-bucket CUDA synchronization. It scored `100.0`, decoded in `41.42s` (`2.30x` dense `17.99s`), and preserved logical/physical accounting (`4.0249` / `8.9173 MB/head-query`, selected `12506.2`). Use this mode for benchmark runtime plus MB accounting. Caveat: no-sync wall buckets are useful for coarse timing only; synchronized `PROFILE_NATIVE_OPS=1` runs are still needed for precise kernel-attribution diagnostics.
- 2026-05-26 clean no-sync/accounting baseline: after reverting rejected QKV fast-view edits, clean canonical LongGen8192 job `50906767` generated in `762.62s`, preserving logical/physical accounting (`1.7628` / `1.8715 MB/head-query`, selected `4284.6`). Main wall buckets were score-grid `76.78s`, accounting `58.30s`, exact logits `49.93s`, QKV cache `48.34s`, prob/base `19.36s`, risk-prefix `14.70s`, and rank-prefix `6.10s`. LongGen16384 job `50905126` generated in `2588.76s`, about `3.69x` dense `702.13s`, so the 16k sustained `<=3x` gate still fails. In the no-sync candidate matrix, accounting accumulation `50906696` is noise-level (`758.24s`), native cuBLAS exact `50906745` is negative (`776.64s`), grouped output workspace `50906707` is negative (`919.59s`), softmax workspace `50906705` is negative (`773.83s`), and grouped exact logits `50906698` remains noncanonical despite `733.85s` because parity failed. A new no-calibration scatter-fill score-grid candidate is tracked in `notes/slurm_manifests/nocalib_scatter_scoregrid_20260526.tsv`.
- 2026-05-26 sparse-direct/all-head workspace follow-up: manifest `notes/slurm_manifests/nocalib_sparse_direct_ws_20260526.tsv`. RULER 32k/128 base diagnostic `50894877` scored `100.0`, decoded in `56.91s`, logical `4.0256 MB/head-query`, physical `8.9177 MB/head-query`, selected `12507.5`, rank `2.14s`, score-grid `4.64s`, exact-logit `4.59s`, prob/base `8.28s`, and risk-prefix `13.58s`. Workspace variants did not improve enough: output workspace `50894879` decoded in `56.83s`, risk workspace `50894878` `56.91s`, softmax workspace `50894880` `57.24s`, strided workspace `50894881` `58.39s`. Long trace parity `50894882` matched CPU/native nearly exactly (`4.25e-9` attention, `1.70e-8` o-proj) but exceeded the benchmark-style Torch/GPU policy tolerance slightly (`5.8e-4` attention, `5.7e-4` o-proj), so this remains diagnostic/off. Follow-up fused sparse-direct softmax/base validation is in `notes/slurm_manifests/fused_sparse_direct_softmax_20260526.tsv`.
- 2026-05-26 no-calib rank/exact-logit diagnostics: `notes/slurm_manifests/nocalib_rank_exact_diag_20260526.tsv` and follow-ups `notes/slurm_manifests/nocalib_allhead_rank_followups_20260526.tsv`. Baseline RULER 32k/128: decode `58.73s`, logical `4.025355 MB/head-query`, physical `8.917617 MB/head-query`, selected `12506.793`, rank-prefix `7.35s`, exact-logit `4.37s`, score-grid `4.06s`, prob/base `8.07s`, risk-prefix `13.50s`. `SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX=1` is promoted after same-run audit job `50894782` computed the old per-group rank-prefix and all-head batched rank-prefix together and completed without mismatch. Diagnostic RULER `50894730`: decode `52.98s`, rank-prefix `2.04s`, logical `4.025212 MB/head-query`, physical `8.917361 MB/head-query`, selected `12507.258`. Current-code baseline repeat `50894772`: decode `64.57s`, selected `12506.93`, confirming the small selected-count differences are run noise rather than rank-prefix mismatch. Canonical promoted-default validation `50894796` passed with score `100.0`, decode `59.11s`, rank-prefix `2.75s`, logical `4.025067 MB/head-query`, physical `8.917334 MB/head-query`, selected `12506.773`; total decode is noisy, so only claim rank-prefix reduction, not a clean end-to-end RULER win. Sustained LongGen diagnostic `50894774` generated 8192 tokens in `1072.61s`, logical `1.8437 MB/head-query`, physical `1.9297 MB/head-query`, selected `4387.0`, rank-prefix `12.88s`, joint total `619.78s`; this modestly improves the no-calib one-pass reference (`~1078s`, joint `675.5s`) but remains above the `<=971s` sustained target. Selector-produced top-k reuse is negative: `50894773` removed rank-prefix time but raised selector time and decoded in `57.32s`. Other variants are negative: native rank-prefix fullsort `65.92s`, rank workspace `66.72s`, budget-prefix `79.30s`, budget-prefix workspace `74.40s`, native exact logits `63.60s`; native exact plus risk workspace OOMed before summary.
- 2026-05-26 merge-risk all-in-one diagnostic is rejected. `SELECTOR_PQ_JOINT_MERGE_RISK_POLICY=1` avoids the canonical grouped risk path and does too much per-KV-group recomputation. RULER 32k/128 job `50894641` scored `100.0` but decoded in `443.27s`, with `392.81s` in the merged risk-prefix bucket. Accepted stats drifted to `3.731 MB/head-query` and `11288.6` selected tokens, compared with the no-calib baseline `4.025 MB/head-query` and `12506.8` selected tokens. Keep the flag diagnostic/off.
- 2026-05-26 grouped merge-risk-prefix diagnostic is rejected. `SELECTOR_PQ_JOINT_MERGE_RISK_PREFIX=1` passed CUDA unit job `50894666`, but RULER 32k/128 job `50894675` decoded in `111.72s`; risk-prefix rose to `58.28s` versus no-calib baseline `13.50s`. Logical stats were close but still shifted (`4.030 MB/head-query`, `12509.6` selected). Existing single grouped risk sort is faster than this two-sort-plus-merge approach.
- 2026-05-26 no-calib native diagnostic batch found no useful existing-flag win. Manifest: `notes/slurm_manifests/nocalib_native_diag_20260526.tsv`. Baseline: decode `58.73s`, step `4.025 MB/head-query`, selected `12506.8`. Completed variants: `risk_ws` `58.39s`, `output_ws` `58.47s`, `risk_topk_ws` `59.77s`, `output_risk_ws` `64.74s`, `risk_topk` `67.31s`, `score_direct_ws` `63.20s`, `score_direct_vprefix` `70.06s`. The score-direct variants prove prob/base can be removed, but residual-risk work grows to `~26.8s`, so end-to-end regresses.
- 2026-05-26 score-grid follow-up: direct gridless recomputation and score-grid workspace reuse are not promotable. The recompute variant is guarded by `SELECTOR_PQ_JOINT_MIXED_POLICY_RECOMPUTE_SCORES=1`; it preserved logical stats but RULER 32k/128 job `50889069` decoded in `97.74s`, much worse than canonical. The fused probability-reuse variant job `50889523` also preserved logical stats (`4.025 MB/head-query`, selected `12507.3`) but decoded in `66.98s` with fused work reported as `score-grid=31.51s`, versus canonical decode `58.73s`. The no-calib score-grid workspace job `50889439` was neutral/slower: decode `59.20s`, joint total `50.15s`, score-grid `4.20s`, versus canonical `58.73s` / `49.68s` / `4.06s`. CUDA unit job `50889524` passed after the guarded diagnostic code. Keep these flags diagnostic/off. Conclusion: score-grid fill/allocation is no longer the main lever; runtime work should target residual-risk prefix, probability/base output, rank-prefix, and exact-logit feeding cost.
- 2026-05-25 score-grid no-calibration optimization: canonical tail logits are raw (`a=1,b=0`), so native score-grid now bypasses affine selected-mask fitting. Added a one-pass no-calibration fill that writes each score-grid element once from `token_to_indexed`, `base_mask`, and `rank_pos`. Validation passed: CUDA unit `50888377`; native saved-trace parity `50888418` over 32k/64k/128k heads 0/8 with max attention/o-proj relL2 `4.25e-9` / `1.70e-8`; RULER 32k/128 `50888423` scored `100.0`, elapsed `73.50s`, decode `58.73s`, logical/physical stats preserved. Sustained LongGen 8192 improved modestly from `1094.62s` / score-grid `106.39s` (`50888244`) to `1078.22s` / score-grid `101.13s` (`50888422`), with logical step still `1.763 MB/head-query` and selected `4284.7`. Safe small win; next bottleneck is whole-grid rank-prefix/exact-logit/prob-base/risk-prefix work, not score-grid fill.
- 2026-05-25 staged K/V policy follow-up: implemented native staged policy helper `joint_select_policy_grouped_flat_staged_no_mb`, returning final outputs/indices plus a CUDA per-group boundary mask. Staged K already used max-prefix rank retrieval; staged V now automatically uses top-k V-prefix construction when available, so a staged V grid reads/sorts only the largest staged exact-V prefix and slices smaller budgets. Targeted CUDA unit job `50885519` passed. RULER 32k/128 staged `k3/v5` job `50885531` scored `100.0`, decoded in `76.91s`, logical `3.836 MB/head-query`, physical `8.918 MB/head-query`, selected `11730.3`, staged accept fraction `0.6685`. This is not promotable: canonical is still about `43s`, and staging still duplicates stage plus full fallback for roughly one-third of KV groups.
- 2026-05-25 canonical joint K/V cleanup: canonical joint decode no longer keeps grouped-risk record construction, adaptive K/V policy walking, fused-policy diagnostics, value-cost arithmetic, V-prefix construction, and final cost/output writeback inside the top-level runner. Legacy fast decode, approximate prefill, per-head fallback, and helper re-export modules were removed. Benchmark wrappers import through `hf_paged_pq_intervention_api.py`; CUDA/parity tests import low-level helpers directly. Current hot-path sizes: `run_hf_paged_pq_intervention_eval.py` 429 lines, `hf_paged_pq_intervention_index_sidecars.py` 205, `hf_paged_pq_intervention_joint.py` 533, `hf_paged_pq_intervention_joint_grouped_risk.py` 956, and `hf_paged_pq_intervention_joint_one_group.py` 1598. Validation: local `py_compile`, import checks, wrapper audit, and `git diff --check` passed. Slurm smoke `50879927` completed cleanly in `00:01:47` but stayed dense-equivalent below page size; active-path Slurm smoke `50879957` completed cleanly in `00:01:49`, with `approx_path_active_fraction=0.667`, `approx_attention_calls_total=64`, and mean logical frontier step `2.092 MB/head-query`.
- 2026-05-25 CUDA kernel split: `paged_pq_kernel.cu` is no longer a 22k-line implementation file. It now contains the common includes/helpers plus ordered includes for 18 fragments in `benchmark/selector_eval/cuda_ext/paged_pq_kernel_parts/`; largest fragment is 2,935 lines. Slurm CUDA build/unit gate passed as job `50811572` with output `cuda_unit_result/kernel_split_20260525`, return code `0`, elapsed `241s`.
- Maintainability refactor landed for canonical config drift control. The single source of truth is now `benchmark/selector_eval/frontier_config.py`; generated shell fragments are `scripts/frontier_canonical_env.sh` and `scripts/frontier_direct_runtime_env.sh`. The HF runner guard, wrapper audit, and readiness audit share the same required/disallowed canonical flags, and the wrapper audit checks generated fragments for staleness.
- Active validation policy update: stop launching `no-stats` / `DISABLE_COST_STATS=1` timing-only jobs. Existing no-stats rows below are historical diagnostics only; future optimization and promotion decisions require accounting/profile artifacts with logical MB, physical MB, selected counts, and wall buckets.
- 2026-05-24 batched optimization workflow update: for long-queue GPU optimization, use matrix runs rather than one edit per diagnostic. Current batch manifest is `notes/slurm_manifests/frontier_cuda_batched_v5_20260524.tsv`. Clearly redundant known-negative jobs were canceled immediately (`50804049`-`50804052`, `50804059`-`50804062`); active/pending candidates remain only as controls or plausible interactions.
- 2026-05-24 next batched CUDA workspace diagnostic submitted after strided-workspace CUDA unit `50805254` passed. Manifest: `notes/slurm_manifests/frontier_cuda_batched_v6_20260524.tsv`; jobs `50805280`-`50805293` plus plain no-risk controls `50805309`-`50805314`. The batch compares plain output-workspace controls, risk-workspace controls, `pq_scale`, duplicate-V collapse, and the new `SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE=1` interactions. This is diagnostic-only with `FRONTIER_CANONICAL_GPU=0`; promotion requires matching selected counts/logical MB/outputs and a real runtime win.
- 2026-05-24 v6 RULER 32k/128 results: the only useful interaction is plain strided output workspace. `strided_output_ws` jobs `50805286` / `50805287` scored `100.0`, decoded in `45.53s` no-stats and `58.42s` accounting, and preserved logical `3.8361 MB/head-query`, physical `8.9179 MB/head-query`, selected `11730.3`. Plain no-risk output workspace control `50805309` / `50805310` decoded in `46.10s` / `58.78s`; `pq_scale` plain control `50805311` / `50805312` decoded in `46.48s` / `58.72s`; duplicate-V collapse was noisy/negative (`46.03s` / `65.42s`). Strided plus `pq_scale` or duplicate-V collapse was not better (`59.73s`, `59.26s`, `60.46s` accounting). Keep `pq_scale` and collapse diagnostic/off; strided workspace is a small runtime win but still needs sustained LongGen validation before promotion.
- 2026-05-24 sustained follow-up submitted for the first promising v6 candidate: `notes/slurm_manifests/frontier_cuda_strided_output_ws_longgen_20260524.tsv`, jobs `50805333` no-stats and `50805334` accounting/wall-profile. These test forced LongGen 8192 with `SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE=1`, grouped output workspace, and risk-prefix workspace.
- 2026-05-24 sustained strided output workspace result: job `50805333` generated forced LongGen 8192 in `724.25s`, logical/physical `1.5923 MB/head-query`, selected `3261.0`, active fraction `0.359`. Paired accounting/profile job `50805334` generated in `1077.43s`, logical `1.7618 MB/head-query`, physical `1.8715 MB/head-query`, selected `4281.0`, with wall buckets `score-grid=104.10s`, `rank-prefix=91.92s`, `prob/base=67.21s`, `risk-prefix=64.15s`, `V-PQ sidecar=37.48s`. This is worse than the best recent sustained no-stats result (`688.21s`), so strided workspace is not promotable.
- 2026-05-24 sustained allocation-policy follow-up: grow-pad job `50806060` used larger `8192` grow pads for strided output workspace, risk-prefix workspace, and persistent V-PQ cache. It generated forced LongGen 8192 in `733.67s`, worse than normal strided (`724.25s`), with unchanged logical/physical `1.5923 MB/head-query` and selected `3261.0`. Pending grow accounting job `50806061` was canceled because the no-stats timing already rejected the candidate.
- 2026-05-24 sparse-direct interaction matrix completed. Manifest: `notes/slurm_manifests/frontier_cuda_sparse_direct_interactions_20260524.tsv`. RULER 32k/128 no-stats: sparse-direct control `50806815` scored `100.0`, decoded in `44.79s`; grouped output workspace `50806816` scored `100.0`, decoded in `44.70s`; strided output workspace `50806817` scored `100.0`, decoded in `44.42s`; strided grow-pad `50806818` failed with CUDA OOM. Forced LongGen 8192 no-stats: sparse-direct control `50806819` generated in `703.46s`; grouped output workspace `50806820` generated in `710.08s`; strided output workspace `50806821` generated in `713.39s`; all three preserved logical/physical `1.5923 MB/head-query`, selected `3261.0`, active fraction `0.359`. Grow-pad LongGen `50806822` was canceled after the RULER OOM. Conclusion: sparse-direct remains the best sustained diagnostic family, but output/strided workspace interactions are not promotable because they do not improve sustained runtime.
- 2026-05-24 matched current-code baseline for batched comparison: `default_baseline_20260524_nostats` job `50804038` scored `100.0` and decoded RULER 32k/128 in `50.45s`; `default_baseline_20260524_accounting` job `50804039` scored `100.0`, decoded in `60.76s`, and reported logical `3.8365 MB/head-query`, physical `8.9180 MB/head-query`, selected `11731.7`, with wall buckets `risk-prefix=13.52s`, `prob/base=8.24s`, `rank-prefix=7.81s`, `score-grid=6.07s`, `exact-logit=4.28s`, `group-pack=2.45s`. Treat this as same-batch normalization only; older guard-on jobs remain the stronger reference.
- 2026-05-24 v5 batched diagnostic summary: all remaining jobs completed after canceling known negatives. Best no-stats RULER 32k/128 result was `interval_risk_output_ws_nostats` job `50804055`, score `100.0`, decode `45.64s`. Best accounting/profile result was `pq_scale_output_ws_accounting` job `50804068`, score `100.0`, decode `58.78s`, logical `3.8361 MB/head-query`, physical `8.9179 MB/head-query`, selected `11730.3`, wall buckets `risk-prefix=13.22s`, `prob/base=8.46s`, `rank-prefix=6.90s`, `score-grid=6.42s`, `exact-logit=4.27s`, `group-pack=2.17s`. `collapse_v_output_ws` was similar (`45.83s` no-stats, `59.23s` accounting). `selector_topk_output_ws` and `native_exact_output_ws` were negative. Keep these as diagnostic interactions only; no v5 candidate removes the dominant risk/prob/rank bottlenecks decisively.
- 2026-05-24 v3/v4 diagnostic summary: repaired materialized grouped output workspace (`50803986` / `50803987`) preserved quality and logical stats, but only reached `45.71s` no-stats and `59.05s` accounting, so it is still diagnostic/off. Risk-prefix top-k/workspace v4 variants (`50804021`-`50804028`) were negative or noise-level: best no-stats was `46.38s`, best accounting `60.63s`, and risk-prefix remained about `13.3-14.7s`. Do not promote `SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK` or `SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE`.
- 2026-05-24 sparse-direct batched v1 result: removing the sparse exact-score table and feeding sparse exact logits directly into the tokenfit/no-fill score-grid path is promising for sustained decode but not yet promotable. CUDA unit `50794524` and long trace parity `50794523` passed (`max_attention_relative_L2=4.25e-09`, `max_oproj_relative_L2=1.70e-08`). RULER 32k/128 no-stats `50794519` decoded in `48.17s`, slower than the current best short gate (`43.17s`), while accounting `50794521` decoded in `60.97s` with logical step `3.8406 MB/head-query`, physical `8.9192 MB/head-query`, selected `11743.4`, and wall buckets `risk-prefix=13.53s`, `prob/base=8.23s`, `rank-prefix=8.06s`, `score-grid=5.67s`, `exact-logit=4.42s`. Sustained LongGen 8192 no-stats `50794520` generated in `688.21s`, improving over the prior canonical sustained run (`750.89s`) and inside the `<=3x` dense target (`323.65s`). Paired accounting `50794522` generated in `1067.12s`, with logical step `1.7621 MB/head-query`, physical `1.8717 MB/head-query`, selected `4281.1`; profiling overhead is large, but wall buckets show the remaining long-decode targets: `score-grid=104.72s`, `rank-prefix=91.62s`, `exact-logit=74.06s`, `risk-prefix=66.12s`, `prob/base=55.56s`, `group-pack=20.26s`.
- 2026-05-24 CUDA-native fused-policy diagnostic result: `SELECTOR_PQ_JOINT_FUSED_MIXED_POLICY=1` is semantically correct but not promotable. Unit job `50776548` passed. RULER 32k/128 retry jobs `50776566` / `50776567` scored `100.0`, but decoded in `72.23s` no-stats and `90.25s` accounting/profile; fused work collapsed into `score_grid=52.03s`, joint total `80.49s`. LongGen 8192 job `50776568` generated in `823.06s`, slower than the current canonical `704.96s`.
- 2026-05-24 CUDA-native fused-policy probability-workspace result: changing the fused helper to materialize probabilities once internally passed unit job `50776584` and fixed the worst recomputation (`score_grid=32.57s` on RULER accounting/profile), but still was not promotable. RULER jobs `50776588` / `50776589` scored `100.0`, decoded in `55.03s` no-stats and `72.60s` accounting/profile, preserving logical stats (`3.8365 MB/head-query`, physical `8.9179 MB/head-query`, selected `11732.0`). LongGen `50776591` generated in `835.59s`. A follow-up exact-read ordering cleanup passed unit `50776606`, but RULER `50776609` / `50776610` stayed negative (`53.22s` no-stats, `71.52s` accounting/profile, fused bucket `33.10s`). Keep the flag off. The useful Amdahl conclusion is that monolithic fusion has reached parity with canonical score+prob+risk/group work, but it does not reduce the underlying residual-risk/prefix and rank/exact costs.
- 2026-05-24 CUDA-native progress: added native sparse exact-K building blocks `gqa_decode_token_exact_logits` and `joint_sparse_exact_score_table`. CUDA unit jobs `50766314` (`00:04:03`) and `50774602` (`00:00:52`) passed on `spgpu`.
- 2026-05-24 CUDA-native negative result: cuBLAS softmax/base-output diagnostics are not promotable. Plain cuBLAS helper passed CUDA unit `50783646` and long trace parity `50785705`, but RULER 32k/128 no-stats/accounting `50785729` / `50786259` decoded in `45.11s` / `57.11s`, and sustained LongGen 8192 regressed badly: no-stats `50789040` generated in `840.73s`, accounting `50789458` generated in `1217.81s` with wall prob/base `60.56s`. Grouped cuBLAS passed unit `50791573` and long parity `50791657`, but RULER v2 no-stats/accounting `50791702` / `50791703` decoded in `47.03s` / `61.93s`; adding grouped score workspace `50791717` / `50791718` worsened to `48.53s` / `63.52s`. TF32 cuBLAS passed unit `50791782` and long parity `50791835`, and no-stats `50791838` was near canonical (`43.39s`), but accounting `50791841` drifted accepted stats (`11753.5` selected, `3.8426 MB/head-query`) and had no decisive runtime win (`57.95s`). Keep `SELECTOR_PQ_JOINT_SOFTMAX_BASE_CUBLAS`, `SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS`, and `SELECTOR_PQ_JOINT_SOFTMAX_BASE_CUBLAS_TF32` diagnostic/off.
- 2026-05-24 CUDA-native sparse exact-K diagnostic: `SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID=1` computes exact QK only for base plus ranked-prefix tokens, scatters those logits into a sparse exact table, and feeds the no-fill tokenfit score-grid path. It falls back to dense exact logits when a K row needs the full context, so semantics are preserved but the diagnostic only helps partial-budget long contexts. It is not canonical yet. Trace parity passed: small-budget active smoke `50775638` matched budgets/MB with max Torch/GPU attention/o-proj relL2 `6.21e-06` / `5.01e-06`; long gate `50775733` passed on decodes `32000,64000,128000`, heads `0,8`, max Torch/GPU attention/o-proj relL2 `3.00e-06` / `2.17e-06`. HF 4k/2 active smoke `50775950` completed cleanly (`approx_path_active_fraction=0.667`). RULER 64k/16 `50776077` failed before decode with A40 OOM in dense prefill MLP. RULER 40k/16 matched comparison: sparse exact `50776161` scored `100.0`, decode `9.73s`, joint wall `8.331s`; dense-exact tokenfit/no-fill control `50776299` scored `100.0`, decode `9.84s`, joint wall `8.449s`; canonical `50776231` scored `100.0`, decode `9.97s`, joint wall `8.539s`. Sparse exact is correct but only marginally faster on this short diagnostic, so keep it diagnostic/off until sustained long-decode evidence justifies promotion. Manifest: `notes/slurm_manifests/sparse_exact_score_grid_20260524.tsv`.
- 2026-05-24 CUDA-native negative result: `SELECTOR_PQ_JOINT_GROUPED_SCORE_WORKSPACE=1` is not promotable. It writes native score-grid outputs into per-KV-head slices of a growable flat grouped score/mask/fit workspace, preserving contiguous KV-head slices and logical semantics. Local syntax/audit checks passed, but RULER 32k/128 regressed: no-stats `50766178` scored `100.0` and decoded in `48.92s`; accounting/profile `50766179` scored `100.0`, decoded in `65.69s`, preserved logical stats (`3.8365 MB/head-query`, physical `8.9180 MB/head-query`, selected `11731.7`), with wall score-grid `8.11s` and joint total `56.19s`. Keep the flag off. Manifest: `notes/slurm_manifests/grouped_score_workspace_20260524.tsv`.
- 2026-05-24 CUDA-native audit update: `benchmark/audit_frontier_cuda_native_hotpath.py` now has `--fail-on-full-native-blockers`. The regenerated audit reports the production full-CUDA gate as blocked by five canonical gaps: paged K-PQ score fullscan still feeding ATen/PyTorch-dependent ranking, rank-prefix `torch.topk`, exact-K PyTorch matmul, V-PQ sidecar/residual packing, and grouped tensor packing. Diagnostic/off alternatives do not count as production blockers. Artifact: `notes/archive/benchmark_audits_2026-05/frontier_cuda_native_hotpath_latest.md`.
- 2026-05-24 CUDA-native negative result: `SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX=1` is not promotable. Unit retry `50758669` passed after fixing the helper to emit per-budget residual deltas instead of cumulative intervals. RULER 32k/128 no-stats `50758672` scored `100.0` but decoded in `50.66s`. Paired accounting/profile `50758673` scored `100.0`, decoded in `69.77s`, and preserved logical stats (`3.8365 MB/head-query`, physical `8.9179 MB/head-query`, selected `11732.1`), but risk-prefix rose to `26.78s` while prob/base fell to only `0.47s`; joint total was `60.09s`. Keep the flag off. This rules out the current score-direct shape; the useful native target is an all-KV-head/grouped entry that avoids Python stacking and repeated risk-prefix work.
- 2026-05-24 CUDA-native negative result: score-direct interval-policy path `SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX=1` plus `SELECTOR_PQ_JOINT_SCORE_DIRECT_INTERVAL_POLICY=1` is also not promotable. Compile-only `50760230`, CUDA unit `50760247`, and long trace parity `50760262` passed over decodes `32000,64000,128000`, heads `0,8`, with no failures. RULER 32k/128 no-stats `50760376` decoded in `50.17s`; accounting/profile `50760375` decoded in `68.78s`, preserved logical stats (`3.8364 MB/head-query`, physical `8.9179 MB/head-query`, selected `11731.5`), but still spent `26.92s` in risk-prefix and `4.37s` in group packing. Keep diagnostic/off. This strengthens the conclusion that the next native target is one grouped entry that writes/consumes grouped buffers directly, not another score-direct V-prefix variant.
- 2026-05-24 CUDA-native negative result: native rank-prefix workspace is semantically correct but slower than the canonical PyTorch `topk` path. Long trace parity `50760468` passed over decodes `32000,64000,128000`, heads `0,8`, max Torch/GPU o-proj relL2 `2.14e-6`. RULER 32k/128 no-stats `50760474` decoded in `47.31s` versus current canonical `43.31s`; accounting/profile `50760475` decoded in `62.06s`, preserved average logical stats (`3.8364 MB/head-query`, physical `8.9179 MB/head-query`, selected `11731.5`), but native rank-prefix cost `10.53s`. Keep `SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX=0`. The current native helper is a full segmented radix sort; the actual full-native target is partial top-k/prefix or score+prefix fusion.
- 2026-05-24 CUDA-native combined diagnostic: native rank-prefix plus grouped native exact logits is also not promotable. Long trace parity `50760540` passed, max Torch/GPU o-proj relL2 `2.17e-6`. RULER 32k/128 no-stats `50760542` decoded in `45.61s`; accounting/profile `50760544` decoded in `60.16s`, preserved average logical stats (`3.8363 MB/head-query`, physical `8.9182 MB/head-query`, selected `11729.8`), and reduced exact-logit wall to `2.26s`, but rank-prefix stayed at `10.48s`. This is useful evidence: grouped exact helps, but it cannot compensate for full-sort native rank-prefix.
- 2026-05-24 CUDA-native negative result: partial budget-prefix rank helper `SELECTOR_PQ_JOINT_NATIVE_BUDGET_PREFIX=1` is correct but not promotable. Unit retry `50760993` passed after fixing shared radix-threshold state, and long parity `50761312` passed over decodes `32000,64000,128000`, heads `0,8`, max Torch/GPU attention/o-proj relL2 `2.62e-6` / `2.14e-6`. RULER 32k/128 no-stats `50761475` decoded in `60.65s`; accounting/profile `50761727` decoded in `74.85s`, preserved logical stats (`3.8385 MB/head-query`, physical `8.9188 MB/head-query`, selected `11736.7`), but rank-prefix wall rose to `23.10s`. Keep the flag off. The 32-pass threshold-selection approach is worse than canonical PyTorch `topk` and the full native segmented sort.
- 2026-05-24 CUDA-native negative result: unsorted per-budget K-prefix diagnostic `SELECTOR_PQ_JOINT_UNSORTED_K_PREFIX=1` is not promotable. First parity attempt `50762098` failed because the native score-grid contract is prefix-based; with an empty ranked prefix it marked the wrong exact-K rows. Corrected selected-token fallback parity `50762122` passed over decodes `32000,64000,128000`, heads `0,8`, max Torch/GPU attention/o-proj relL2 `2.72e-6` / `2.18e-6`. The valid RULER selected-fallback jobs are much slower than canonical: no-stats `50763353` decoded in `109.40s`; accounting/profile `50763354` decoded in `129.76s`, with logical stats preserved (`3.8384 MB/head-query`, physical `8.9187 MB/head-query`, selected `11736.4`) but score-grid `82.24s` and rank-prefix `12.10s`. Keep the flag off. This rules out unordered per-budget selected-token fallback; the full-native rank target still needs canonical nested-prefix/tie-preserving selection fused into the score-grid path.
- 2026-05-24 CUDA-native guard cleanup: explicit `SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT=1` now passes the long saved-trace parity gate on current code and is allowed by the canonical guard when exact full-budget score-grid rows are enabled. Job `50758789` covered decodes `32000,64000,128000`, heads `0,8`, with no failures and max Torch/GPU attention/o-proj relL2 `7.08e-7` / `4.33e-7`. RULER 32k/128 no-stats `50758785` scored `100.0` and decoded in `43.31s`; accounting/profile `50758786` scored `100.0`, decoded in `63.51s`, logical step `3.8364 MB/head-query`, physical `8.9179 MB/head-query`, selected `11731.5`, rank-prefix `8.80s`, joint total `54.00s`. This is mostly a guard/default consistency cleanup because canonical duplicate-row collapse already avoids full-budget rank sorting. Redundant sustained LongGen jobs `50758797` and `50758798` were canceled.
- 2026-05-24 CUDA-native negative result: cache-assisted PyTorch exact-logit fallback is not promotable. Reusing the existing transposed fp32 dense-K cache preserved trace semantics, but hurt RULER runtime. Long trace parity `50758835` passed over decodes `32000,64000,128000`, heads `0,8`, max Torch/GPU o-proj relL2 `2.07e-7`. RULER 32k/128 no-stats `50758836` scored `100.0` and decoded in `47.28s`; accounting/profile `50758837` scored `100.0`, decoded in `64.00s`, and exact-logit wall rose to `9.36s`. The code was reverted; exact-K remains a PyTorch-hotpath blocker until a native backend wins in the HF loop. Manifest: `notes/slurm_manifests/exact_key_cache_reuse_20260524.tsv`.
- 2026-05-24 CUDA-native negative sustained result: grouped-GQA native exact-logit backend behind `SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS_BACKEND=grouped` is correct but not promotable. CUDA unit `50758889` passed, long trace parity `50758894` passed over decodes `32000,64000,128000`, heads `0,8`, with no failures and max Torch/GPU o-proj relL2 `2.17e-6`. RULER 32k/128 no-stats `50758893` scored `100.0` and decoded in `41.24s`, faster than current canonical `43.31s`; accounting/profile `50758891` scored `100.0`, decoded in `57.34s`, preserved logical step `3.8366 MB/head-query`, physical step `8.9186 MB/head-query`, selected `11729.6`, and reduced exact-logit wall to `2.28s` with joint total `48.31s`. However sustained LongGen 8192 did not generalize: no-stats `50759782` generated in `738.33s`, slower than the stronger canonical sustained references, and accounting/wall `50759793` generated in `977.67s`, just over the `<=3x` dense gate and slower than canonical accounting. Accounting logical step was `1.7620 MB/head-query`, physical `1.8716 MB/head-query`, selected `4281.0`, with wall exact-logit `35.34s`, rank-prefix `86.17s`, risk-prefix `65.46s`, score-grid `103.37s`, prob/base `50.39s`, group-pack `19.80s`, and joint total `580.26s`. Keep grouped exact diagnostic/off; the next native target must reduce sustained score/rank/risk/prob work, not only short-gate exact logits. Manifest: `notes/slurm_manifests/grouped_exact_logits_20260524.tsv`.
- 2026-05-24 CUDA-native negative result in isolated worktree: `worktrees/opt-fullnative-grouped-pack` adds `SELECTOR_PQ_JOINT_NATIVE_GROUP_PACK=1`, a native CUDA pack helper for grouped `base_output_grid`/`probs_grid` and residual/error records. This targets the grouped tensor-packing blocker by replacing Python `torch.stack` packing with extension calls while preserving frontier semantics. Compile-only Slurm job `50759467` passed on `standard` in `186s`; targeted GPU unit job `50759337` passed in `218s`. The RULER 32k/128 no-stats retry `50759853` completed generation and scored `100.0` before a post-summary worktree `.venv` wrapper failure; its timing summary is usable and decoded in `44.36s`, slower than current canonical `43.31s` and grouped-exact `41.24s`. Accounting/profile retry `50759855` completed cleanly, scored `100.0`, decoded in `68.04s`, preserved logical stats (`3.8365 MB/head-query`, physical `8.9180 MB/head-query`, selected `11731.7`), but regressed joint wall total to `58.24s` and left group-pack at `2.42s`. Keep the flag diagnostic/off; this per-group copy-kernel shape does not remove the grouped-packing bottleneck. Manifest: `notes/slurm_manifests/fullnative_group_pack_20260524.tsv`.
- 2026-05-24 CUDA-native negative result: current-code recheck of `SELECTOR_PQ_JOINT_GROUPED_OUTPUT_WORKSPACE=1` is not promotable. RULER 32k/128 quality is preserved (`50758563` / `50758565` both score `100.0`) and accounting logical stats match (`3.8361 MB/head-query`, selected `11730.1`), but runtime is not better: no-stats decode `43.62s` versus current guard-on append `43.17s`, and accounting/profile decode regresses to `63.19s` with wall joint total `53.61s`. LongGen diagnostic `50758566` was canceled after the negative short-gate result. Manifest: `notes/slurm_manifests/grouped_output_workspace_recheck_20260524.tsv`. Keep the flag diagnostic/off.
- 2026-05-24 CUDA-native negative result: `SELECTOR_PQ_JOINT_SOFTMAX_BASE_WORKSPACE=1` should stay diagnostic/off. First attempt reused one slot and corrupted delayed grouped-risk records: RULER no-stats `50758412` decoded in `43.97s` but score was `0.0`; invalid jobs `50758413`, `50758414`, and `50758415` were canceled. The semantic fix keys workspaces by KV-head slot. Corrected RULER retry preserved quality but did not improve runtime: no-stats `50758443` scored `100.0` and decoded in `44.07s`; accounting/profile `50758444` scored `100.0`, decoded in `58.52s`, logical step `3.8361 MB/head-query`, physical step `8.9179 MB/head-query`, selected `11730.3`, wall prob/base `8.15s`, score-grid `6.56s`, risk-prefix `13.49s`, joint total `49.67s`. Since short timing is worse than canonical, LongGen retries `50758445` and `50758446` were canceled. Manifest: `notes/slurm_manifests/softmax_base_workspace_validation_20260524.tsv`.
- 2026-05-24 CUDA-native negative result: `SELECTOR_PQ_JOINT_SCORE_GRID_WORKSPACE=1` remains diagnostic/off. RULER retry completed (`50757845` no-stats `43.43s`, `50757844` accounting/profile `58.46s`, logical step `3.8361 MB/head-query`), but sustained LongGen is negative. No-stats retry `50758032` generated 8192 tokens in `657.61s`, slightly faster than canonical `50754219` (`668.32s`), but paired accounting/profile `50758033` generated in `1024.59s`, logical step `1.7620 MB/head-query`, physical step `1.8716 MB/head-query`, selected `4281.1`, wall score-grid `104.49s`, rank-prefix `86.97s`, risk-prefix `65.60s`, exact-logit `75.19s`, joint total `624.12s`. Because accounting/profile regresses badly versus canonical accounting (`842.58s`) and prior score-grid workspace validation was negative, keep it off.
- 2026-05-24 CUDA-native update: native aggregate accounting is now canonical and now accumulates compatible grouped records on device before a single reporting-time flush. Wrapper defaults and `FRONTIER_CANONICAL_GPU=1` require `SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING=1`, while diagnostic verification mode remains rejected. Unit job `50756040` passed, same-process verification smoke `50756107` passed, batched smoke `50756188` preserved the active 4k/2 cost stats, and batched guard-on RULER 32k/128 job `50756199` passed readiness: score `100.0`, decode `46.90s`, logical step `3.8363 MB/head-query`, physical step `8.9179 MB/head-query`, selected `11731.4`, active fraction `0.992`. Batched profile job `50756244` scored `100.0` with decode `64.58s`; accounting dropped to `1.00s` from `2.36s` in the previous deferred-accounting profile `50756178` and `3.13s` in the pre-deferred profile `50756126`. Largest remaining wall buckets are risk-prefix `13.56s`, rank-prefix `9.27s`, prob/base `8.15s`, score-grid `7.25s`, exact logits `4.58s`, and group packing `2.48s`. Hot-path audit now counts accounting as custom CUDA; remaining PyTorch-heavy targets are rank-prefix `torch.topk`, exact-K matmul, and grouped tensor packing.
- 2026-05-24 CUDA-native negative result: cuBLAS TF32 exact-logit precompute is diagnostic only. Unit `50757008` passed and microbenches `50757020` / `50757021` showed full all-head cuBLAS-TF32 exact logits about `1.06x` faster than Torch full exact at 32k/128k, but real RULER 32k/128 regressed. Per-KV native exact decoded `48.26s` no-stats / `65.73s` profile-accounting (`50757024` / `50757027`), and all-head native exact decoded `48.14s` no-stats / `64.37s` profile-accounting (`50757030` / `50757031`). The all-head profile kept logical step `3.8342 MB/head-query`, physical step `8.9177 MB/head-query`, selected `11723.6`, and exact-logit wall `7.99s`. Keep `SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS=0`; isolated GEMM speedup is not enough.
- 2026-05-24 CUDA-native negative result: interval-risk policy should stay diagnostic/off. The helper reuses residual-risk interval sums and selects the adaptive K/V policy output without first materializing the full V-prefix output grid. Build `50757573` and CUDA unit `50757566` passed. Long saved-trace parity `50757587` passed over decodes `32000,64000,128000`, heads `0,8`, max Torch/GPU o-proj relL2 `4.28e-7`. RULER 32k/128 no-stats `50757588` decoded in `43.70s`; accounting/profile `50757589` decoded in `64.08s`, logical step `3.8361 MB/head-query`, physical step `8.9179 MB/head-query`, selected `11730.3`, wall risk-prefix `13.66s`. This is not faster than canonical and does not attack the real risk-prefix cost, so keep `SELECTOR_PQ_JOINT_INTERVAL_RISK_POLICY=0`.
- 2026-05-24 CUDA-native negative result: opt-in native sealed-page V-PQ sidecar construction is not promotable. Unit parity passed (`50756439`) and short RULER 32k/128 was neutral/slightly positive (`50756451` no-stats `43.40s`; `50756450` accounting `45.33s`, logical step `3.8361 MB/head-query`), but sustained LongGen 8192 regressed versus canonical native V-PQ append: no-stats `50756465` generated in `781.68s` versus canonical `668.32s`; accounting/profile `50756468` generated in `766.04s`, logical step `1.7620 MB/head-query`, physical step `1.8716 MB/head-query`, selected `4281.1`, wall joint V-PQ sidecar `19.74s`. Keep `SELECTOR_PQ_JOINT_NATIVE_VPQ_SIDECAR=0`; it remains guarded as diagnostic/off.
- 2026-05-24 CUDA-native negative result: risk-prefix workspace reuse should stay off. Unit job `50755693` passed, but diagnostic profile `50756268` with `SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE=1` kept risk-prefix at `13.59s` versus `13.56s` for baseline `50756244`, with only noise-level total-runtime movement. The next risk-prefix optimization needs to reduce sorting/prefix work itself, not only reuse CUB buffers.
- 2026-05-24 CUDA-native negative result: fused grouped-risk policy should stay off. Added no-MB fused grouped-risk entry points and batched accounting for non-MB policies; CUDA unit `50756320` passed and active 4k/2 HF smoke `50756368` completed with logical step `1.2383 MB/head-query`. However 32k/128 diagnostics were much slower than canonical: no-stats `50756374` decoded in `130.17s`; accounting `50756377` decoded in `131.43s`, logical step `3.8345 MB/head-query`, physical step `8.9175 MB/head-query`, selected `11725.5`. Conclusion: the fused policy shape is wrong because it recomputes residual-risk output values on demand; avoiding dummy MB tensors is not enough.
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
- `benchmark/audit_benchmark_readiness.py` now treats the current CPU-frontier semantics as the readiness contract. It flags non-`joint_kv_stability` confidence, non-`global_residual_risk` exact-V selection, non-fullscan selectors, approximate prefill, missing canonical guard, and non-raw tail-score calibration. It no longer flags exact logical cost accounting as a readiness failure.
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
- Native V-PQ exact-suffix append plus grouped V-PQ sidecar cache is promoted into the canonical wrapper/guard. The helper updates unsealed suffix sidecars with exact V, zero residual, and zero residual-risk error without PyTorch slice-copy/zero ops. Fresh guard-on RULER 32k/128 validation passed: no-stats `50755834` decoded in `43.17s`, and accounting `50755835` decoded in `47.19s` with logical step `3.836063 MB/head-query`, selected `11730.289`, active fraction `0.992`, readiness `ok` for the accounting run. Sustained LongGen 8192 no-stats `50754219` generated in `668.32s` (`2.06x` dense), better than native-softmax canonical `710.06s`. Manifest: `notes/slurm_manifests/native_vpq_append_validation_20260524.tsv` and `notes/slurm_manifests/native_vpq_append_promotion_20260524.tsv`; readiness audit: `notes/archive/benchmark_audits_2026-05/readiness_native_vpq_append_promotion_20260524.md`.
- Rejected diagnostic grouped-output softmax/base workspace for now. It lets `joint_softmax_base_outputs` write probabilities and base outputs into caller-provided grouped buffers, but RULER 32k/128 no-stats/accounting `50755882` / `50755883` decoded in `44.01s` / `47.31s`, slower or neutral versus guard-on append (`43.17s` / `47.19s`), with selected-token drift (`11732.03` vs `11730.29`). CUDA unit `50755881` passed, so the helper stays available behind `SELECTOR_PQ_JOINT_GROUPED_OUTPUT_WORKSPACE=1`, but canonical guard rejects it. Manifest: `notes/slurm_manifests/grouped_output_workspace_validation_20260524.tsv`.
- Dense-prefill grouped V-PQ sidecar prewarm was tested and rejected. It moved grouped sidecar packing out of first active decode, but canonical RULER 32k/128 validation did not improve runtime materially (`43.56s` no-stats, `47.15s` accounting) and shifted selected-token stats (`11731.32` vs guard-on append `11730.29`). The patch was removed to preserve canonical semantics. Manifest retained as a negative result: `notes/slurm_manifests/post_grouped_vpq_prewarm_validation_20260524.tsv`.
- Noncanonical all-native hot-path diagnostic uses native rank-prefix workspace, native exact logits, grouped output workspace, and the current native score/prob/risk/policy/V-PQ paths. Long parity `50755904` passed over decodes `32000,64000,128000`, heads `0,8`, with max Torch/GPU o-proj relL2 `4.25e-7`. RULER 32k/128 no-stats `50755905` decoded in `46.48s`; accounting `50755906` decoded in `49.73s`, logical step `3.835286 MB/head-query`, selected `11726.20`. LongGen 8192 no-stats `50755907` generated in `765.75s`, worse than canonical native V-PQ append (`668.32s`) and native-softmax (`710.06s`). This proves the stricter native path is feasible but is not promotable because it is slower than the guard-on append path and has small accepted-stat drift. Manifest: `notes/slurm_manifests/all_native_hotpath_validation_20260524.tsv`.
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
- No-fill-only validation completed. RULER 32k/128 no-stats `50734367` decoded in `43.94s`, but paired accounting `50734368` decoded in `47.96s` with logical `3.8361 MB/head-query`, selected `11730.29`, slightly slower than canonical accounting. LongGen 16k no-stats `50734366` generated in `2505.63s`, a small improvement over canonical `2565.56s` but still above target. Keep `SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL=0`.
- Fused mixed-softmax/base without rank-position is not a promotion candidate. Long parity `50733865` passed, but RULER 32k/128 no-stats/accounting `50733868`/`50733887` decoded in `47.38s`/`51.70s`, and LongGen 8192 wall generated in `731.86s`; all are slower than canonical.
- Native exact-logit custom-kernel validation is not a promotion candidate. Manifest `notes/slurm_manifests/native_exact_logits_validation_20260523.tsv`: unit `50743664` passed; RULER 32k/128 no-stats `50745177` decoded in `44.34s` and accounting `50745198` decoded in `46.91s` with logical step `3.8351 MB/head-query`; these are effectively tied with canonical native-softmax. Sustained LongGen 8192 no-stats `50745230` generated in `809.19s`, worse than canonical `50730291` (`710.06s`); accounting `50745231` generated in `795.96s` with logical step `1.7617 MB/head-query`. Keep `SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS=0` for canonical runs.
- Added a cuBLAS-backed transposed-K native exact-logit helper behind `gqa_decode_full_exact_logits_t_cublas` and `SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS_BACKEND=cublas_t`. Unit retry `50745999` passed. RULER 32k/128 cuBLAS exact-logit jobs `50746024`/`50746021` scored `100.0` but decoded in `48.61s`/`51.28s`, slower than canonical. LongGen 8192 no-stats/accounting `50746022`/`50746023` generated in `725.71s`/`1002.36s`; accounting logical step was `1.7619 MB/head-query`. This is worse than canonical native-softmax (`710.06s` no-stats, `815.34s` accounting), so keep the cuBLAS exact-logit backend diagnostic-only.
- Added diagnostic selector-topk-prefix reuse behind `SELECTOR_PQ_JOINT_SELECTOR_TOPK_PREFIX=1`: native all-head PQ precompute now can return the maximum needed rank prefix, avoiding a later Python `torch.topk` over the same scores. RULER 32k/128 jobs `50746035`/`50746037` scored `100.0` but decoded in `45.22s`/`49.31s` with matching logical stats, so it is not a short-decode promotion candidate. LongGen 8192 no-stats `50746036` generated in `856.40s`, and accounting `50746038` generated in `1018.00s` with logical step `1.7619 MB/head-query`; this is also worse than canonical native-softmax, so keep `SELECTOR_PQ_JOINT_SELECTOR_TOPK_PREFIX=0`.
- Grouped top-k residual-risk V-prefix helper is not promotable. Manifest: `notes/slurm_manifests/risk_prefix_topk_validation_20260523.tsv`. CUDA V-PQ unit `50746264` passed, and long trace parity `50746265` passed over decodes `32000,64000,128000`, heads `0,8`, with no failures; max CPU/native attention/o-proj relL2 `4.25e-09` / `1.70e-08`, max Torch/GPU-policy attention/o-proj relL2 `2.50e-06` / `4.28e-07`. RULER 32k/128 no-stats/accounting `50746266`/`50746269` decoded in `46.60s`/`48.34s`, slower than canonical native-softmax (`44.02s`/`47.12s`), with logical step `3.8365 MB/head-query`. Sustained LongGen 8192 no-stats `50746271` generated in `862.86s`, also worse than canonical native-softmax `50730291` (`710.06s`). Keep `SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK=0`.
- Submitted one sustained native V-PQ base-output diagnostic because V-PQ sidecar/base handling remains a large 16k wall bucket. Manifest: `notes/slurm_manifests/native_vpq_base_sustained_20260523.tsv`; job `50746350`, forced LongGenBench SGT-short 8192 wall profile with `FRONTIER_CANONICAL_GPU=0` and `SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE=1`. This is diagnostic only; the path was already negative on short RULER, so promotion requires sustained runtime improvement plus separate parity/accounting validation.
- Fused token-list affine-fit mixed-softmax/base diagnostic was implemented behind `SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE=1`, with in-kernel PQ-logit scaling support. Manifest `notes/slurm_manifests/fused_tokenfit_softmax_validation_20260524.tsv`: unit `50748814` passed, long parity `50748820` passed over 32k/64k/128k with no failures, and RULER 32k/128 no-stats/accounting `50748821`/`50748822` scored `100.0` but decoded in `47.30s`/`48.62s`. It is slower than canonical native-softmax (`44.02s`/`47.12s`) and shows slight accepted-stat drift (`3.8396 MB/head-query`, selected `11739.7`), so keep `SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE=0`.
- Added `benchmark/audit_frontier_cuda_native_hotpath.py` to keep the native-backend gap explicit. Latest output is archived at `notes/archive/benchmark_audits_2026-05/frontier_cuda_native_hotpath_latest.md`. Current map: custom CUDA owns score-grid, softmax/base, residual-risk V-prefix, and policy; K-PQ fullscan scoring is mostly native but still feeds ATen/PyTorch rank-prefix work; remaining PyTorch/ATen hot-path components are rank-prefix `torch.topk`, exact-K matmul, V-PQ sidecar maintenance, and grouped tensor packing. Treat this as the active CUDA-native checklist.
- Current-code validation after the fast-layout state passed. Manifest `notes/slurm_manifests/frontier_current_validation_20260523.tsv`: CUDA unit job `50727996` passed all extension tests in `82s`; long native CPU-vs-GPU parity job `50728005` passed over decodes `32000,64000,128000`, heads `0,8`, with no failures and max Torch/GPU-policy attention/o-proj relL2 `2.62e-06` / `2.14e-06`.
- Reporting cleanup: no-stats benchmark summaries now include `approx_path_active_fraction` and fall back to approximate-call coverage for selector/tail active fractions when detailed cost counters are disabled. `benchmark/audit_benchmark_readiness.py` also reports the metric name for public benchmark quality, e.g. substring accuracy or pass@1. This only changes summary/reporting; it does not alter selector logic, accepted budgets, outputs, or MB accounting.
- Current passing-gate readiness audit is archived at `notes/archive/benchmark_audits_2026-05/readiness_fastlayout_gates_20260523.md`. It includes the canonical 32k/128 accounting gate and the canonical 8192 forced long-decode accounting gate, both with readiness `ok`.

## Current Task-Quality Validation

- Public task-quality validation is in progress for the passing canonical CUDA path.
- Current active-path validation batch completed: `notes/slurm_manifests/public_longdecode_active_validation_active_validation_20260526_current.tsv`. These forced-8192 slices compare dense vs canonical frontier with sealed pages active. AIME24: dense accuracy `0.333`, generation `336.17s`; frontier accuracy `0.333`, generation `662.35s` (`1.97x` dense), logical `1.5952 MB/head-query`, physical `1.6945 MB/head-query`, selected `3830.9`, active fraction `0.299`. LiveCodeBench codegen: dense pass@1 `0.333`, generation `336.02s`; frontier pass@1 `0.333`, generation `706.85s` (`2.10x` dense), logical `1.6608 MB/head-query`, physical `1.7734 MB/head-query`, selected `4041.0`, active fraction `0.331`. LongGenBench SGT-short: dense completion `0.538` and substring accuracy `0.0`, generation `344.26s`; frontier completion `0.538` and substring accuracy `0.0`, generation `884.51s` (`2.57x` dense), logical `1.7629 MB/head-query`, physical `1.8716 MB/head-query`, selected `4284.6`, active fraction `0.359`. This is active-path quality smoke evidence, not full benchmark evidence.
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

Important caveat: most rows here are paper-inspired compression proxies, not official method implementations. KIVI is being rerun with the official quantization layout before being used in new comparisons.

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
| KIVI b4, group 32, 2048 residual | scalar KV compression | `5.903` | `0.004388` | `0.013980` | best tested KIVI quality, higher MB than frontier |
| KIVI b4, group 32, 128 residual | scalar KV compression | `5.407` | `0.007119` | `0.014053` | closer MB to frontier, higher relL2 |
| KIVI b2, group 32, 2048 residual | scalar KV compression | `3.849` | `0.085225` | `0.180118` | lower MB, too much output distortion |
| KVQuant-like b4 clipped, 128 exact window | clipped scalar proxy | `4.395` | `0.025512` | `0.060401` | lower MB than frontier, worse relL2 |
| PQ-like s8b6, 128 exact window | PQ/VQ compression proxy | `0.557` | `0.158352` | `0.270666` | very low MB but large output distortion |
| Dense fp16, mean over decode suite | dense | `17.201` | `0.0` | `0.0` | exact reference |

KIVI rows use the `jy-yuan/KIVI` quantization layout and are produced by `kivi_b*_g*_w*`.

2026-05-29 update: ran a denser ours-new operating sweep for the current hybrid relative policy (`K 10,30,50,70,90,100%`, `V 5,10,20,40,60,80,100%`, `proxy_mass_m0p9`, sqrt budget-delta confidence). Slurm job `51123321` swept 21 thresholds from `0.0002` to `0.032` in `15:35`. The curve spans `4.028 MB/head-query` at mean o-proj relL2 `0.00337` to `7.780 MB/head-query` at `0.000172`; canonical `tau=0.002` is `4.519 MB/head-query`, mean/max relL2 `0.001301/0.001761`. Plot: `attention_efficiency_result/plots/kivi_vs_frontier_new_20260529/kivi_vs_ours_new_dense_pareto.png`; raw plotted points: `attention_efficiency_result/plots/kivi_vs_frontier_new_20260529/kivi_vs_ours_new_dense_pareto_points.csv`.

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
| KIVI scalar KV compression | `kivi_b4_g32_w128` | `5.407` | `0.007119` | `0.014053` |
| Kitty-like channel-promoted scalar | `kitty_like_k4v4_p0.1_pb8_buf128_s32` | `4.828` | `0.009785` | `0.018728` |
| TaDA-like mean-centered scalar | `tada_like_b4_g64_w128` | `4.928` | `0.013670` | `0.031462` |
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

## Score-Proxy K-Logit Variant Sweep

CPU trace diagnostic added to `benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py`.

Question: can Kitty-inspired selector-side corrections reduce K-logit distortion without giving up the current low output error and MB frontier?

Setup:

- trace: `real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz`;
- decode lengths: `500,1000,2000,4000,8000,16000,32000,64000,128000`;
- all heads, layer 16;
- policy: `k_first_alternating`, threshold `0.001`;
- baseline frontier: paged K-PQ selector + exact K correction + residual-risk V exact set + V-PQ remainder.

Four variant families tested:

- SparQ-channel correction: read exact K on top-|q| channels for every token and correct PQ selector logits.
- Promoted K residual sidecar: store quantized exact-PQ residuals for high-residual channels per page and add them to selector logits.
- Residual/additive K-PQ: train a second PQ on the K residual and add residual-PQ scores to selector logits.
- Band calibration: fit separate affine PQ-logit calibration per rank band using selected exact K and extra probes.

Top results:

| variant | mean MB/head-query | mean o-proj relL2 | max o-proj relL2 | mean logit relL2 | mean prob JS | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | `4.785` | `0.001034` | `0.002168` | `0.036639` | `0.000121` | current reference |
| sparq_r4 | `4.629` | `0.000967` | `0.001868` | `0.034344` | `0.000120` | best output error / MB tradeoff |
| sparq_r16 | `4.783` | `0.000996` | `0.001863` | `0.026385` | `0.000111` | best logit error, costs ~`0.974` MB extra side reads |
| promoted_p0p2_b8 | `5.344` | `0.000996` | `0.001849` | `0.030885` | `0.000091` | best probability JS, higher MB |
| residual_pq_m1b4_s8 | `4.757` | `0.000976` | `0.001895` | `0.033131` | `0.000115` | good deployable sidecar tradeoff |
| bandcal_b8_p16 | `4.891` | `0.001001` | `0.002342` | `0.036608` | `0.000113` | little logit gain despite probe reads |

Interpretation:

- The selector can be made more logit-faithful, but the gain is modest unless we pay nontrivial sidecar traffic.
- SparQ-style channel correction is the strongest direct logit correction. `sparq_r16` cuts mean logit relL2 by about `28%` vs baseline with similar total MB because it reduces the K budget enough to offset side reads.
- Residual/additive PQ is the cleanest deployable path if we want a compact K-residual sidecar: it improves logit error and preserves output error without per-query exact-channel reads.
- Band-wise affine calibration is not compelling; it adds probe reads and does not materially improve logit relL2.
- None of these changes overturn the earlier conclusion that probability/output metrics are already robust; this is mainly a selector-risk hardening path.

Artifacts:

- `attention_efficiency_result/score_proxy_variants_20260523/summary.md`;
- `attention_efficiency_result/score_proxy_variants_20260523/summary.csv`;
- `attention_efficiency_result/plots/score_proxy_variants_20260523/o_proj_relL2/frontier_pareto.png`;
- `attention_efficiency_result/plots/score_proxy_variants_20260523/logit_relL2/frontier_pareto.png`;
- `attention_efficiency_result/plots/score_proxy_variants_20260523/prob_JS/frontier_pareto.png`.

## Quest Under Current Joint K/V Backend

CPU trace diagnostic added to `benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py`.

Question: does Quest become competitive if evaluated under the latest backend: mixed exact-K/K-PQ probabilities, residual-risk exact-V selection, V-PQ reconstruction, and joint K/V confidence?

Setup:

- trace: `real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz`;
- decode lengths: `500,1000,2000,4000,8000,16000,32000,64000,128000`;
- all heads, layer 16;
- policy: `k_first_alternating`, threshold `0.001`;
- current reference from the same metric stack: `4.785 MB/head-query`, mean o-proj relL2 `0.001034`.

Results:

| selector | mean MB/head-query | mean o-proj relL2 | max o-proj relL2 | mean logit relL2 | mean prob JS | mean K budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current frontier | `4.785` | `0.001034` | `0.002168` | `0.036639` | `0.000121` | `12096` |
| Quest page rank 8 | `5.586` | `0.202628` | `0.642262` | `0.084513` | `0.037829` | `19143` |
| Quest page rank 16 | `5.552` | `0.264499` | `0.931622` | `0.125085` | `0.089750` | `19029` |
| Quest page rank 32 | `5.589` | `0.290652` | `1.126522` | `0.166140` | `0.107327` | `19179` |
| Quest -> PQ rank 8 nprobe 4 | `4.363` | `0.328233` | `1.197398` | `0.161058` | `0.100111` | `11492` |
| Quest -> PQ rank 16 nprobe 4 | `3.725` | `0.404976` | `1.382961` | `0.314668` | `0.192208` | `8832` |
| Quest -> PQ rank 16 nprobe 8 | `3.867` | `0.244061` | `1.124626` | `0.200580` | `0.136993` | `9259` |
| Quest -> PQ rank 32 nprobe 8 | `3.760` | `0.255227` | `1.173731` | `0.278133` | `0.147290` | `8903` |

Interpretation:

- Quest page bounds remain noncompetitive. They spend more MB than the current frontier while producing roughly `200x` higher mean output error.
- Quest -> PQ can reduce MB, but quality collapses because the page-bound router discards too many important tokens before PQ reranking.
- This confirms the older Quest negative result under the latest residual-risk V / joint-confidence backend.

Artifacts:

- `attention_efficiency_result/joint_kv_quest_20260523/quest_vs_frontier/summary.md`;
- `attention_efficiency_result/joint_kv_quest_20260523/quest_vs_frontier/summary.csv`;
- `attention_efficiency_result/joint_kv_quest_20260523/quest_vs_frontier/mb_vs_oproj_relL2.png`;
- `attention_efficiency_result/joint_kv_quest_20260523/quest_vs_frontier/mb_vs_logit_relL2.png`;
- `attention_efficiency_result/joint_kv_quest_20260523/quest_vs_frontier/mb_vs_prob_js.png`.

## Tail-Logit Calibration Ablation

Question: what happens if the K-PQ tail logits are not affine-calibrated, i.e. `tail_logit = raw_pq_logit` with `a=1, b=0`?

Setup:

- same full CPU trace suite as the current joint K/V reference;
- decode lengths `500..128000`, all heads, layer 16;
- selector: fullscan paged K-PQ;
- policy: `k_first_alternating`, threshold `0.001`;
- V path: residual-risk exact V plus V-PQ reconstruction.

Results:

| tail calibration | mean MB/head-query | max MB/head-query | mean o-proj relL2 | max o-proj relL2 | mean logit relL2 | mean prob JS | mean K budget | mean V budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine selected | `4.785` | `14.562` | `0.001034` | `0.002168` | `0.036639` | `0.000121` | `12096` | `2581` |
| none (`a=1,b=0`) | `4.813` | `14.562` | `0.001026` | `0.002083` | `0.035484` | `0.000114` | `12231` | `2581` |

Interpretation:

- The affine calibration is not justified by this trace.
- No-affine slightly improves output, logit, and probability metrics, with a tiny MB increase from a slightly larger accepted K budget.
- Raw K-PQ tail logits are now the canonical frontier default; affine calibration should be treated as an explicit ablation.

Artifacts:

- `attention_efficiency_result/joint_kv_tail_calibration_20260525/tail_calibration_ablation/summary.md`;
- `attention_efficiency_result/joint_kv_tail_calibration_20260525/tail_calibration_ablation/summary.csv`;
- `attention_efficiency_result/joint_kv_tail_calibration_20260525/tail_calibration_ablation/per_decode.csv`.

## Historical Source

The full pre-cleanup result log is preserved at `notes/archive/status_history/selector_eval_latest_results_2026-05-20_full.md`.

## CUDA-Native Backend Progress - 2026-05-24

Current objective: move the canonical frontier decode path from PyTorch-heavy orchestration toward fused CUDA-native kernels while preserving CPU frontier semantics.

Validated evidence so far:

| candidate | job(s) | result | promotion status |
| --- | --- | --- | --- |
| rank-prefix workspace | `50755740`, `50755743`, `50755744`, `50755746` | CUDA unit passed; long saved-trace parity passed over `32000,64000,128000`, heads `0,8`, max Torch/GPU o-proj relL2 `4.33e-7`; RULER 32k/128 no-stats/accounting decoded `48.11s` / `51.06s`, logical `3.836 MB/head-query` | diagnostic only; slower than clean current canonical `45.36s` / `47.05s` |
| native exact cuBLAS logits | `50755751`, `50755752`, `50755753` | long saved-trace parity passed, max Torch/GPU o-proj relL2 `4.12e-7`; RULER 32k/128 no-stats/accounting decoded `50.43s` / `52.29s`, logical `3.834 MB/head-query`, selected `11724.15` | diagnostic only; slower and slight selected-stat drift |
| rank-prefix workspace + native exact cuBLAS | `50755755`, `50755756` | long saved-trace parity passed, max Torch/GPU o-proj relL2 `4.12e-7`; RULER 32k/128 no-stats decoded `52.06s` | diagnostic only; slower than each component alone |
| score-grid workspace | `50754735`, `50755449`, `50755695`, `50755697`, `50756502`, `50756503` | CUDA unit passed; long saved-trace parity passed over `32000,64000,128000`, heads `0,8`; max Torch/GPU o-proj relL2 `2.17e-6` | not promotable: RULER 32k/128 no-stats `44.27s` and accounting `47.18s`, neutral/slower than canonical native V-PQ append (`43.17s` / `47.19s`); sustained LongGen 8192 no-stats `790.69s` and accounting `744.96s`, slower than canonical `668.32s` |
| native V-PQ suffix append + grouped V-PQ cache | `50753996`, `50754219`, `50755546`, `50755547`, `50755834`, `50755835` | LongGen 8192 no-stats `668.32s` vs dense `323.65s` (`2.06x`); RULER 32k/128 no-stats/accounting `43.71s` / `46.97s`; accounting matches current canonical logical stats (`3.836073 MB/head-query`, selected `11730.289`) | promoted; fresh guard-on jobs `50755834` / `50755835` queued |
| grouped output workspace | `50755881`, `50755882`, `50755883` | diagnostic path writes softmax/base outputs into grouped caller buffers before grouped residual-risk processing | queued; canonical guard rejects until validation |
| risk-prefix workspace | `50755693`, `50756268` | CUDA unit passed; 32k/128 diagnostic profile left risk-prefix unchanged (`13.59s` vs `13.56s` baseline) | diagnostic only; keep off |
| fused grouped-risk no-MB policy | `50756320`, `50756368`, `50756374`, `50756377` | CUDA unit and active 4k/2 smoke passed, but 32k/128 no-stats/accounting decoded `130.17s` / `131.43s`; logical `3.8345 MB/head-query`, physical `8.9175 MB/head-query`, selected `11725.5` | diagnostic only; keep off because on-demand output reconstruction is far slower than materialized V-prefix grid |

Native V-PQ append is now canonical only together with grouped V-PQ sidecar cache, because that validated combination matched the current canonical accounting exactly while improving both RULER 32k/128 and LongGen 8192 timing. The next open native-hot-path targets are rank-prefix top-k, exact-K logits, grouped output packing/workspaces, and sealed-page V-PQ sidecar construction.

## CUDA Maintainability Audit - 2026-05-25

Thermo-nuclear review cleanup focused on the CUDA-native extension structure, not selector algorithm changes.

| item | outcome |
| --- | --- |
| `paged_pq_kernel.cu` monolith | reduced to a 172-line ordered include shell |
| CUDA fragments | split at kernel/wrapper boundaries; all but one fragment are below 1k LOC |
| fragment contract | documented in `benchmark/selector_eval/cuda_ext/paged_pq_kernel_parts/README.md` |
| CUDA wrapper contract | grouped-risk policy wrappers restored to return `{outputs, indices}` with grouped shapes |
| validation | Slurm job `50813654` passed full CUDA unit set in `272s` |

Follow-up thermo cleanup:

| item | outcome |
| --- | --- |
| `paged_pq_ext.cpp` pybind wall | reduced to an 18-line ordered include shell with fragments under `benchmark/selector_eval/cuda_ext/paged_pq_ext_parts/`; all fragments below 1k LOC |
| remaining oversized geometric-output CUDA fragment | split dim-scan wrappers into `paged_pq_geometric_output_dimscan_wrappers.cu.inc`; both fragments below 1k LOC |
| HF intervention helpers | extracted from the main runner and decomposed into focused modules for common/cache, stats, value/V-PQ, geometric GPU helpers, and trace summaries |
| validation | Slurm job `50816437` passed full CUDA unit set in `67s`; `py_compile`, wrapper audit, and `git diff --check` passed |

Additional runner cleanup extracted `PagedPQPatchState`, which owns the shared decode caches and dense-prefill sidecar warming that used to live as nested closure state inside `patched_paged_pq_attention`. Slurm job `50817085` passed the full CUDA unit set in `59s` after this extraction.

Latest runner decomposition:

| item | outcome |
| --- | --- |
| CLI/parser/reporting | moved into `hf_paged_pq_intervention_cli.py` |
| prefill Torch selector helpers | moved into `hf_paged_pq_prefill_torch.py` |
| per-forward state | moved into `hf_paged_pq_intervention_forward_state.py` |
| joint K/V workspaces and exact-logit helper | moved into `hf_paged_pq_intervention_joint_workspace.py` |
| joint K/V budget parsing | moved into `hf_paged_pq_intervention_joint_budget.py` |
| V-PQ sidecars | split into `hf_paged_pq_intervention_vpq_sidecars.py` and `hf_paged_pq_intervention_vpq_grouped.py` |
| runner size | `run_hf_paged_pq_intervention_eval.py` reduced to `9658` lines; extracted Python modules are below `1k` LOC |
| validation | Slurm jobs `50818766` and `50820581` passed the full CUDA unit set; latest output `cuda_unit_result/thermo_runner_decompose_final_20260525` |

Remaining structural debt: the 9.7k-line `patched_paged_pq_attention` context manager in the HF intervention runner. It is still the next serious thermo blocker; the remaining split needs real decomposition of the prefill, decode, and joint K/V all-head paths without changing canonical frontier semantics.

## CUDA Staged V-Grid Diagnostic - 2026-05-25

Question: can the GPU path evaluate only the first few V-budget grid points, accept early when confidence is stable, and fall back to the full V grid only when the staged pass reaches the boundary?

Implementation:

- Added trace-parity support for `--use_staged_risk_prefix` and `--staged_risk_prefix_v_steps`.
- The diagnostic computes a staged fullsort V-prefix grid for the first `4` V budgets, runs the same native policy on that staged grid, and falls back to the full canonical grid when the selected staged V index is the boundary.
- The top-k staged variant remains diagnostic/off; the tested path is fullsort staged risk prefix.

Results:

| gate | job | runtime | logical MB/head-query | physical MB/head-query | selected tokens | quality/parity | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| long trace parity, decodes `32000,64000,128000`, heads `0,8` | `50883761` | `72.84s` | matched CPU | matched CPU | matched CPU | no failures; max Torch/GPU o-proj relL2 `2.14e-6` | semantically valid |
| RULER 32k/128 staged fullsort | `50883766` | decode `50.37s` | `3.836` | `8.918` | `11731.2` | score `100.0` | passes short gate, but only marginally useful |
| LongGen 8192 staged fullsort | `50883762` | generation `836.04s` | `1.762` | `1.872` | `4281.1` | completed forced 8192 tokens | not promotable |
| LongGen 8192 guarded current reference | `50883671` | generation `824.28s` | `1.762` | `1.872` | `4281.0` | completed forced 8192 tokens | current canonical remains better |

Interpretation:

- Staged-grid execution is logically compatible with the CPU frontier policy when it uses fullsort risk-prefix rows and full-grid fallback.
- It does not improve sustained decode. On LongGen 8192 it preserves logical MB but increases wall time because many rows hit the staged V-boundary and pay both staged risk-prefix work and full-grid fallback work.
- Keep `SELECTOR_PQ_JOINT_STAGED_RISK_PREFIX=0` in canonical defaults. Staged risk prefix remains a diagnostic/off path unless a future version can predict early-accept rows cheaply enough to avoid duplicate full-grid work.

Artifacts:

- Manifest: `notes/slurm_manifests/staged_risk_prefix_20260525.tsv`.
- Parity summary: `attention_efficiency_result/joint_kv_cpu_gpu_parity_20260525/staged_fullsort_long/summary.json`.
- RULER summary: `ruler_eval_result/frontier_cuda_opt_20260525_staged_fullsort_ruler32k128_current/pagedpq_batched_niah_single_1_32768_n1/summary/niah_single_1.json`.
- LongGen summary: `public_longdecode_result/frontier_cuda_opt_20260525_staged_fullsort_longgen8192_nosync/pagedpq_longgenbench_sgt_short_smoke/summary.json`.

## CUDA Score-Prob Interval Diagnostic - 2026-05-26

| gate | job | result | decision |
| --- | --- | --- | --- |
| CUDA VPQ/unit | `50890848` | passed in `222s`; output `cuda_unit_result/score_prob_interval_policy_20260526` | valid build/unit |
| long saved-trace parity | `50893278` | native path matched CPU over `32000,64000,128000`, heads `0,8`; max CPU/native attention/o-proj relL2 `4.25e-09` / `1.70e-08`; auxiliary Torch-GPU path hit `5.8e-4` and tripped the old `5e-4` tolerance | native semantics valid |
| RULER 32k/128 accounting | `50892121` | score `100.0`; decode `74.51s`; joint total `64.72s`; logical `4.026 MB/head-query`; physical `8.918 MB/head-query`; selected `12507.5` | not promotable |
| score-direct top-k CUDA VPQ/unit | `50894450` | passed in `229s`; output `cuda_unit_result/score_direct_topk_interval_retry_20260526` | valid build/unit |
| score-direct top-k long parity | `50894451` | native path matched CPU over `32000,64000,128000`, heads `0,8`; max CPU/native attention/o-proj relL2 `4.25e-09` / `1.70e-08`; auxiliary Torch-GPU path tripped the old tolerance | native semantics valid |
| score-direct top-k RULER 32k/128 accounting | `50894452` | score `100.0`; decode `66.35s`; joint total `57.05s`; logical `4.024 MB/head-query`; physical `8.917 MB/head-query`; selected `12503.5`; prob/base `0.45s`, risk-prefix/top-k `28.15s` | not promotable |
| grouped native softmax/base CUDA VPQ/unit | `50894511` | passed in `220s` after fixing a grouped-value stride bug caught by failed unit job `50894503` | valid build/unit |
| grouped native softmax/base long parity | `50894513` | native path matched CPU over `32000,64000,128000`, heads `0,8`; max CPU/native attention/o-proj relL2 `4.25e-09` / `1.70e-08`; auxiliary Torch-GPU path tripped the old tolerance | native semantics valid |
| grouped native softmax/base RULER 32k/128 accounting | `50894514` | score `100.0`; decode `68.71s`; joint total `59.92s`; logical `4.025 MB/head-query`; physical `8.917 MB/head-query`; selected `12506.8`; prob/base worsened to `18.90s` | not promotable |
| native budget + risk workspace | `50894552` | score `100.0`; decode `65.33s`; joint total `55.20s`; logical `4.025 MB/head-query`; physical `8.918 MB/head-query`; selected `12505.8` | not promotable |
| native budget + grouped output workspace | `50894553` | score `100.0`; decode `57.98s`; joint total `49.16s`; logical `4.026 MB/head-query`; physical `8.918 MB/head-query`; selected `12507.7` | noise-level, selected/logical drift |
| native budget + sparse exact score | `50894554` | score `100.0`; decode `58.48s`; joint total `49.55s`; logical `4.027 MB/head-query`; physical `8.918 MB/head-query`; selected `12512.7` | noise-level, selected/logical drift |

Interpretation: materializing grouped probabilities once eliminates most prob/base time (`0.52s` vs canonical `8.07s`) but inflates residual-risk/V-prefix work (`31.67s` risk-prefix vs canonical `13.50s`). Total decode regresses from canonical `58.73s` to `74.51s`, so keep `SELECTOR_PQ_JOINT_SCORE_PROB_INTERVAL_POLICY=0`.

The score-direct top-k variant proves the same point from the opposite direction: avoiding full probability materialization drops prob/base to `0.45s`, but exact top-k residual-risk ordering is still expensive when the maximum exact-V prefix is large. It decodes in `66.35s`, still slower than canonical `58.73s`, so keep `SELECTOR_PQ_JOINT_SCORE_DIRECT_TOPK_INTERVAL_POLICY=0`.

The grouped native softmax/base variant preserves semantics but also regresses: deferring softmax/base into one grouped native call increases prob/base to `18.90s`, likely from large grouped score/Vhat packing plus a less favorable grouped-kernel shape. Keep `SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE=0`.

The workspace/native-budget combos do not remove enough work to matter. The only apparent speedup (`57.98s` vs `58.73s`) is within noise and changes selected/logical stats, so it is not a clean promotion candidate.

## CUDA Score-Grid / Accounting Diagnostics - 2026-05-26

| candidate | job(s) | result | decision |
| --- | --- | --- | --- |
| no-calib scatter score-grid | `50909143`, `50909522`, `50909523`, `50909524` | CUDA unit passed; partial parity passed for `32000/head0`; RULER 32k/128 was flat (`37.75s` patched vs clean `37.60s`); LongGen8192 regressed (`511.37s` patched vs clean `505.50s`, score-grid `80.64s` vs clean `76.78s`) | not promotable; full parity `50910087` canceled after runtime rejection |
| fused policy accounting | `50911480`, `50911602`, `50911603`, `50911611` | CUDA unit passed in `233s`; native CPU/GPU parity matched tightly (`4.25e-09` attention relL2, `1.70e-08` o-proj relL2), with only auxiliary Torch-GPU tolerance failures; RULER 32k/128 decoded `39.09s`; LongGen8192 generated `508.17s` | not promotable; slightly slower than clean baselines |
| native-op accounting profile | `50910465` | RULER 32k/128 decoded `53.20s` with `PROFILE_NATIVE_OPS=1`; true native accounting time was `0.96s`, not the earlier apparent `13s+` wall bucket | accounting was mostly async attribution noise |

Interpretation: the no-calib scatter score-grid reduced some rank-prefix overhead on short RULER but increased score-grid time on sustained LongGen, so the score-grid bottleneck is not simply rank-position/base-mask traffic. The fused policy-accounting experiment removed the apparent accounting bucket (`58.30s` LongGen8192 clean wall bucket to `1.00s`) but did not improve end-to-end runtime (`508.17s` vs clean `505.50s`). That means the accounting bucket was mostly async attribution noise. Do not target accounting next; target real compute/memory buckets: score-grid, exact logits/QKV handling, prob/base, risk-prefix, and V-PQ sidecar.

## Benchmark Reporting Rule - 2026-05-27

All task reports must include bandwidth savings, not only raw frontier MB. `benchmark/audit_benchmark_readiness.py` now reports:

| field | meaning |
| --- | --- |
| `dense step` | estimated dense K/V-read MB per head-query, using average decode context tokens and bf16 K+V with head_dim 128 |
| `logical save %` | savings from the logical frontier cost model vs dense |
| `physical save %` | savings from the actual GPU-emulation memory traffic vs dense |

Current archived reports:

| artifact | note |
| --- | --- |
| `notes/archive/benchmark_audits_2026-05/ruler64k_partial_bandwidth_20260527.md` | RULER 64k partial results; frontier logical savings are about `66-81%`, but physical GPU-emulation savings are only about `41-44%` |
| `notes/archive/benchmark_audits_2026-05/public_longdecode_partial_bandwidth_20260527.md` | public long-decode partial results; current LongGenBench rows are still substring/completion smoke metrics, not official LLM-judge LongGenBench accuracy |

## CPU/GPU Frontier Hyperparameter Audit - 2026-05-27

The current CPU sweet-spot headline `4.813 MB/head-query` is a mean over decode lengths `500..128000`, not the 128k point. The same CPU trace has substantially higher per-decode logical cost at long contexts:

| decode length | CPU raw-tail mean MB/head-query | mean K budget | mean V budget | mean selected K tokens | mean o-proj relL2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | `1.947` | `4864` | `1472` | `6090` | `0.000493` |
| 1000 | `2.140` | `5120` | `1664` | `6686` | `0.000583` |
| 2000 | `2.253` | `4352` | `1408` | `7398` | `0.000602` |
| 4000 | `2.668` | `4096` | `1184` | `9302` | `0.000538` |
| 8000 | `3.387` | `7616` | `2528` | `10518` | `0.000824` |
| 16000 | `5.248` | `20160` | `3904` | `15990` | `0.001137` |
| 32000 | `7.386` | `20096` | `2656` | `25142` | `0.001022` |
| 64000 | `8.294` | `22848` | `3008` | `26102` | `0.001949` |
| 128000 | `9.998` | `20928` | `5408` | `26230` | `0.002083` |

Static audit:

- RULER64 frontier summaries use the same core policy/budget settings as the current CPU raw-tail frontier: `joint_kv_stability`, `k_first_alternating`, threshold `0.001`, K budgets `4096,8192,14336,32768`, V budgets `1024,2048,4096,6144,8192,12288,16384`, `tail_score_calibration=none`, and `selected_value_exact_rule=global_residual_risk`.
- The RULER64 run summaries omitted some metadata (`subvecs`, `subbits`, `value_subvecs`, `value_subbits`, `kmeans_iters`, `compact_vpq_risk_prefix`) even though the wrapper passes those settings. This was a reporting/audit issue, not an algorithm change; future RULER/LongBench summaries now record `compact_vpq_risk_prefix`, and future RULER summaries also record the PQ parameter fields.
- The strongest existing saved CPU-vs-GPU parity artifact before this audit (`50728005`) used the old `tail_score_calibration=affine_selected` setting. A new raw-tail parity job was submitted as `51050181` to verify current canonical raw-tail GPU semantics directly over the long saved trace.
