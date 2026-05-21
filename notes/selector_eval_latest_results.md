# Selector-Eval Latest Results

Keep this page compact. Archive full tables and stale variants instead of appending indefinitely.

## Active Goal

Get the current frontier algorithm into a state where real benchmarks can run comfortably and produce trustworthy results.

This phase is benchmark readiness and execution:

- complete frontier decode path enabled;
- correctness/parity tests passing;
- benchmark runtime practical enough for RULER/LongBench slices;
- honest logical-vs-physical MB accounting;
- no oracle leakage or hidden dense fallback in selector logic.

## Canonical Frontier Path

Benchmark-facing path:

- dense prefill;
- decode-only fullscan paged-PQ selector;
- online geometric confidence/budgeting;
- selected exact K logits;
- mixed exact/compressed V;
- V-PQ residual tail estimation;
- exact accepted-budget logical accounting.

`FRONTIER_CANONICAL_GPU=1` should reject noncanonical fixed-budget shortcuts.

## Current GPU Canonical Results

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

## Broad Benchmark Validation

Latest canonical benchmark-slice evidence:

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

The broad slices are not a full task-quality claim. They prove the current canonical path can run RULER, LongBench-v2, and a public long-decode slice end-to-end with dense prefill, decode-only paged-PQ, geometric confidence, selected exact/compressed V, V-PQ tail estimation, and separated logical-vs-physical accounting.

## Current Bottleneck

Latest useful profiled canonical run (`50585696`, native threshold topk16384 + 512 threads):

- native attention path: `32.18s` total;
- geometric confidence/output fused portion: `16.91s`;
- selected-mass threshold generation: `5.93s`;
- exact ranked-logit materialization: `8.98s`;
- selector fullscan/top-k: `4.32s`;
- final output time is folded into geometric confidence (`native_output_seconds_total = 0.0`).

The next runtime target is benchmark-readiness rather than another algorithm change: keep this canonical path fixed, confirm the target band across RULER/LongBench/long-decode slices, and only optimize runtime if broader runs are too slow.

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

## Historical Source

The full pre-cleanup result log is preserved at `notes/archive/status_history/selector_eval_latest_results_2026-05-20_full.md`.
