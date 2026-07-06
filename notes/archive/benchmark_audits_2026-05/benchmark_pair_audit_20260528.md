# Benchmark Pair Audit

Dense/frontier pairs are matched by stripping `dense_` / `pagedpq_` prefixes from artifact directories.

## Completed Pair Table

| pair | metric | dense | frontier | delta | frontier s/ex | runtime x | logical MB/hq | physical MB/hq | dense MB/hq | logical save | physical save | selected | active | pred same | judge same | text same | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aime24_off0_n30 | accuracy_pct | 53.33 | n/a | n/a | n/a | n/a | n/a | n/a | 4.082 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | missing-frontier-run |
| gpqa_off0_n25 | accuracy_pct | 60.00 | 56.00 | -4.00 | 1307.81 | 4.29 | 2.065 | 2.435 | 1.759 | -17.4 | -38.4 | 5399.3 | 0.399 | n/a | n/a | 1/25 | logical-bandwidth-worse-than-dense, physical-bandwidth-worse-than-dense, frontier-quality-lower, frontier-hit-max-new-tokens |
| gpqa_off25_n25 | accuracy_pct | 52.00 | 56.00 | 4.00 | 769.81 | 2.53 | 1.604 | 1.703 | 1.806 | 11.2 | 5.7 | 3894.3 | 0.264 | n/a | n/a | 0/25 | ok |
| lbv2_medium_easy_n16_l120000 | accuracy_pct | 31.25 | n/a | n/a | n/a | n/a | n/a | n/a | 51.857 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | missing-frontier-run |
| lbv2_short_easy_n16_l120000 | accuracy_pct | 62.50 | 62.50 | 0.00 | 36.41 | 2.59 | 5.371 | 9.624 | 15.478 | 65.3 | 37.8 | 16298.9 | 0.975 | 16/16 | 16/16 | 15/16 | ok |
| lbv2_short_hard_n16_l120000 | accuracy_pct | 43.75 | 43.75 | 0.00 | 42.48 | 3.20 | 5.407 | 9.739 | 16.223 | 66.7 | 40.0 | 16492.2 | 0.979 | 15/15 | 16/16 | 15/16 | ok |
| livecodebench_codegen_off0_n10 | pass@1_pct | 60.00 | n/a | n/a | n/a | n/a | n/a | n/a | 3.411 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | missing-frontier-run |
| ruler_ctx131072_n1_cwe | score | 0.00 | n/a | n/a | n/a | n/a | n/a | n/a | 63.865 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | missing-frontier-run |
| ruler_ctx131072_n1_niah_multikey_1 | score | 100.00 | 100.00 | 0.00 | 260.83 | 1.57 | 12.502 | 36.403 | 63.904 | 80.4 | 43.0 | 32980.3 | 0.941 | 0/1 | n/a | n/a | ok |
| ruler_ctx131072_n1_niah_multikey_2 | score | 0.00 | 0.00 | 0.00 | 262.85 | 1.60 | 11.692 | 36.111 | 63.967 | 81.7 | 43.5 | 30988.6 | 0.941 | 0/1 | n/a | n/a | ok |
| ruler_ctx131072_n1_niah_multikey_3 | score | 100.00 | 100.00 | 0.00 | 255.26 | 1.58 | 11.114 | 35.835 | 63.626 | 82.5 | 43.7 | 29048.6 | 0.941 | 1/1 | n/a | n/a | ok |
| ruler_ctx131072_n1_niah_multiquery | score | 100.00 | 100.00 | 0.00 | 261.51 | 1.57 | 12.358 | 36.336 | 63.921 | 80.7 | 43.2 | 32697.8 | 0.941 | 1/1 | n/a | n/a | ok |
| ruler_ctx131072_n1_niah_multivalue | score | 100.00 | 100.00 | 0.00 | 260.38 | 1.57 | 12.529 | 36.440 | 63.906 | 80.4 | 43.0 | 32943.3 | 0.941 | 0/1 | n/a | n/a | ok |
| ruler_ctx131072_n1_niah_single_1 | score | 100.00 | 100.00 | 0.00 | 258.90 | 1.59 | 9.526 | 35.593 | 63.941 | 85.1 | 44.3 | 24177.6 | 0.941 | 0/1 | n/a | n/a | ok |
| ruler_ctx131072_n1_niah_single_2 | score | 100.00 | 100.00 | 0.00 | 259.65 | 1.61 | 12.554 | 36.460 | 63.875 | 80.3 | 42.9 | 32898.2 | 0.941 | 0/1 | n/a | n/a | ok |
| ruler_ctx131072_n1_niah_single_3 | score | 100.00 | 100.00 | 0.00 | 260.39 | 1.57 | 12.562 | 36.478 | 63.885 | 80.3 | 42.9 | 32877.2 | 0.941 | 0/1 | n/a | n/a | ok |
| ruler_ctx131072_n1_vt | score | 20.00 | 20.00 | 0.00 | 255.84 | 1.57 | 8.491 | 35.262 | 63.902 | 86.7 | 44.8 | 21220.2 | 0.941 | 1/1 | n/a | n/a | ok |
| ruler_ctx32768_n1_cwe | score | 80.00 | 80.00 | 0.00 | 79.06 | 4.02 | 6.887 | 9.438 | 15.815 | 56.5 | 40.3 | 21942.7 | 0.985 | 1/1 | n/a | n/a | ok |
| ruler_ctx32768_n1_niah_multikey_1 | score | 100.00 | 100.00 | 0.00 | 78.79 | 3.98 | 7.782 | 9.729 | 15.740 | 50.6 | 38.2 | 24258.8 | 0.985 | 1/1 | n/a | n/a | ok |
| ruler_ctx32768_n1_niah_multikey_2 | score | 0.00 | 100.00 | 100.00 | 80.31 | 4.07 | 7.371 | 9.737 | 15.833 | 53.4 | 38.5 | 22954.1 | 0.985 | 0/1 | n/a | n/a | ok |
| ruler_ctx32768_n1_niah_multikey_3 | score | 100.00 | 100.00 | 0.00 | 76.85 | 3.99 | 6.373 | 9.295 | 15.655 | 59.3 | 40.6 | 20097.8 | 0.985 | 1/1 | n/a | n/a | ok |
| ruler_ctx32768_n1_niah_multiquery | score | 100.00 | 100.00 | 0.00 | 79.42 | 4.04 | 7.267 | 9.534 | 15.765 | 53.9 | 39.5 | 22999.5 | 0.985 | 1/1 | n/a | n/a | ok |
| ruler_ctx32768_n1_niah_multivalue | score | 100.00 | 100.00 | 0.00 | 78.58 | 4.03 | 7.757 | 9.679 | 15.745 | 50.7 | 38.5 | 24372.3 | 0.985 | 0/1 | n/a | n/a | ok |
| ruler_ctx32768_n1_niah_single_1 | score | 100.00 | 100.00 | 0.00 | 96.87 | 3.31 | 4.022 | 8.918 | 15.891 | 74.7 | 43.9 | 12493.3 | 0.985 | 1/1 | n/a | n/a | ok |
| ruler_ctx32768_n1_niah_single_2 | score | 100.00 | 100.00 | 0.00 | 78.64 | 3.98 | 7.917 | 9.730 | 15.714 | 49.6 | 38.1 | 24756.2 | 0.985 | 1/1 | n/a | n/a | ok |
| ruler_ctx32768_n1_niah_single_3 | score | 100.00 | 100.00 | 0.00 | 80.86 | 4.15 | 7.928 | 9.729 | 15.727 | 49.6 | 38.1 | 24830.7 | 0.985 | 1/1 | n/a | n/a | ok |
| ruler_ctx32768_n1_vt | score | 100.00 | 100.00 | 0.00 | 80.96 | 4.04 | 5.059 | 9.189 | 15.970 | 68.3 | 42.5 | 15794.1 | 0.985 | 1/1 | n/a | n/a | ok |
| ruler_ctx65536_n1_cwe | score | 0.00 | 0.00 | 0.00 | 157.43 | 2.57 | 9.233 | 18.272 | 31.919 | 71.1 | 42.8 | 28349.2 | 0.970 | 1/1 | n/a | n/a | ok |
| ruler_ctx65536_n1_niah_multikey_1 | score | 100.00 | 100.00 | 0.00 | 125.67 | 2.20 | 10.642 | 18.814 | 31.748 | 66.5 | 40.7 | 31549.8 | 0.970 | 0/1 | n/a | n/a | ok |
| ruler_ctx65536_n1_niah_multikey_2 | score | 0.00 | 0.00 | 0.00 | 130.18 | 2.34 | 9.564 | 18.488 | 31.835 | 70.0 | 41.9 | 28647.4 | 0.970 | 0/1 | n/a | n/a | ok |
| ruler_ctx65536_n1_niah_multikey_3 | score | 100.00 | 100.00 | 0.00 | 124.52 | 2.19 | 8.901 | 18.020 | 31.123 | 71.4 | 42.1 | 26388.2 | 0.970 | 1/1 | n/a | n/a | ok |
| ruler_ctx65536_n1_niah_multiquery | score | 100.00 | 100.00 | 0.00 | 135.70 | 1.98 | 10.305 | 18.670 | 31.760 | 67.6 | 41.2 | 30783.4 | 0.970 | 1/1 | n/a | n/a | ok |
| ruler_ctx65536_n1_niah_multivalue | score | 100.00 | 100.00 | 0.00 | 135.06 | 2.27 | 9.712 | 18.428 | 31.747 | 69.4 | 42.0 | 29319.8 | 0.970 | 1/1 | n/a | n/a | ok |
| ruler_ctx65536_n1_niah_single_1 | score | 100.00 | 100.00 | 0.00 | 127.10 | 2.21 | 6.090 | 17.706 | 31.711 | 80.8 | 44.2 | 17366.8 | 0.970 | 1/1 | n/a | n/a | ok |
| ruler_ctx65536_n1_niah_single_2 | score | 100.00 | 100.00 | 0.00 | 124.33 | 2.20 | 10.481 | 18.729 | 31.722 | 67.0 | 41.0 | 31185.5 | 0.970 | 1/1 | n/a | n/a | ok |
| ruler_ctx65536_n1_niah_single_3 | score | 100.00 | 100.00 | 0.00 | 127.27 | 2.22 | 10.710 | 18.809 | 31.732 | 66.2 | 40.7 | 31813.0 | 0.970 | 1/1 | n/a | n/a | ok |
| ruler_ctx65536_n1_vt | score | 100.00 | 100.00 | 0.00 | 147.01 | 2.37 | 7.668 | 18.124 | 31.910 | 76.0 | 43.2 | 22527.7 | 0.970 | 1/1 | n/a | n/a | ok |

## LongGenBench SGT Metrics

For SGT, `periodic` and `range` are harder smoke checks than `once`. Official SGT paper numbers require an LLM yes/no judge; these substring metrics are artifact-only checks.

_No LongGenBench-SGT pairs detected._

## Incomplete / Failed Runs

| label | reason | jobid | output_dir | slurm_out |
| --- | --- | --- | --- | --- |
| longbench_v2/pagedpq_lbv2_medium_easy_n16_l120000 | partial-artifact-no-summary | n/a | /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/benchmark_suite_result/coalesced_benchmark_suite_20260527_lbv2_medium_retry_spgpu/longbench_v2/pagedpq_lbv2_medium_easy_n16_l120000 | n/a |
| public/pagedpq_aime24_off0_n30 | partial-artifact-no-summary | n/a | /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/benchmark_suite_result/coalesced_benchmark_suite_20260527_aime24_qwen3_official/public/pagedpq_aime24_off0_n30 | n/a |
| public/pagedpq_aime24_off0_n30 | partial-artifact-no-summary | n/a | /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/benchmark_suite_result/coalesced_benchmark_suite_20260528_aime24_frontier_retry/public/pagedpq_aime24_off0_n30 | n/a |
| public/pagedpq_livecodebench_codegen_off0_n10 | partial-artifact-no-summary | n/a | /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/pagedpq_livecodebench_codegen_off0_n10 | n/a |
| pagedpq_lbv2_medium_easy_n16_l120000 | cuda-no-kernel-image-partial-predictions-no-summary | 51051349 | benchmark_suite_result/coalesced_benchmark_suite_20260528_lbv2_medium_rtx6000_missing/longbench_v2/pagedpq_lbv2_medium_easy_n16_l120000 | slurm_out/coalesced_benchmark_suite_20260528_lbv2_medium_rtx6000_missing/pagedpq_medium_easy-51051349.out |
| dense_lbv2_medium_hard_n16_l120000 | cuda-no-kernel-image-partial-predictions-no-summary | 51051351 | benchmark_suite_result/coalesced_benchmark_suite_20260528_lbv2_medium_rtx6000_missing/longbench_v2/dense_lbv2_medium_hard_n16_l120000 | slurm_out/coalesced_benchmark_suite_20260528_lbv2_medium_rtx6000_missing/dense_medium_hard-51051351.out |
| pagedpq_lbv2_medium_hard_n16_l120000 | cuda-no-kernel-image-partial-predictions-no-summary | 51051352 | benchmark_suite_result/coalesced_benchmark_suite_20260528_lbv2_medium_rtx6000_missing/longbench_v2/pagedpq_lbv2_medium_hard_n16_l120000 | slurm_out/coalesced_benchmark_suite_20260528_lbv2_medium_rtx6000_missing/pagedpq_medium_hard-51051352.out |
| pagedpq_aime24_off0_n30 | timeout-partial-predictions-no-summary | 51026456 | benchmark_suite_result/coalesced_benchmark_suite_20260527_aime24_qwen3_official/public/pagedpq_aime24_off0_n30 | slurm_out/coalesced_benchmark_suite_20260527_aime24_qwen3_official/public_group_000-51026456.out |
| pagedpq_aime24_off0_n30 | partial-predictions-no-summary | 51083241 | benchmark_suite_result/coalesced_benchmark_suite_20260528_aime24_frontier_retry/public/pagedpq_aime24_off0_n30 | slurm_out/coalesced_benchmark_suite_20260528_aime24_frontier_retry/public_group_000-51083241.out |
| pagedpq_livecodebench_codegen_off0_n10 | partial-predictions-no-summary | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/pagedpq_livecodebench_codegen_off0_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| dense_livecodebench_codegen_off10_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/dense_livecodebench_codegen_off10_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| pagedpq_livecodebench_codegen_off10_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/pagedpq_livecodebench_codegen_off10_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| dense_livecodebench_codegen_off20_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/dense_livecodebench_codegen_off20_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| pagedpq_livecodebench_codegen_off20_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/pagedpq_livecodebench_codegen_off20_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| dense_livecodebench_codegen_off30_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/dense_livecodebench_codegen_off30_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| pagedpq_livecodebench_codegen_off30_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/pagedpq_livecodebench_codegen_off30_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| dense_livecodebench_codegen_off40_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/dense_livecodebench_codegen_off40_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| pagedpq_livecodebench_codegen_off40_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/pagedpq_livecodebench_codegen_off40_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| dense_livecodebench_codegen_off50_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/dense_livecodebench_codegen_off50_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| pagedpq_livecodebench_codegen_off50_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/pagedpq_livecodebench_codegen_off50_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| dense_livecodebench_codegen_off60_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/dense_livecodebench_codegen_off60_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| pagedpq_livecodebench_codegen_off60_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/pagedpq_livecodebench_codegen_off60_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| dense_livecodebench_codegen_off70_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/dense_livecodebench_codegen_off70_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| pagedpq_livecodebench_codegen_off70_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/pagedpq_livecodebench_codegen_off70_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| dense_livecodebench_codegen_off80_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/dense_livecodebench_codegen_off80_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| pagedpq_livecodebench_codegen_off80_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/pagedpq_livecodebench_codegen_off80_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| dense_livecodebench_codegen_off90_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/dense_livecodebench_codegen_off90_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| pagedpq_livecodebench_codegen_off90_n10 | running-or-incomplete-check-slurm | 51031495 | benchmark_suite_result/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public/pagedpq_livecodebench_codegen_off90_n10 | slurm_out/coalesced_benchmark_suite_20260527_gpqa_lcb_qwen3_official/public_group_000-51031495.out |
| ruler_group_000 | running-or-incomplete-check-slurm | 51083998 | benchmark_suite_result/coalesced_benchmark_suite_20260527_ruler128k_retry/ruler | slurm_out/coalesced_benchmark_suite_20260527_ruler128k_retry/ruler_group_000-51083998.out |
| lbv2_short_easy_n4_paired_diag | oom | 51051434 | benchmark_suite_result/lbv2_short_paired_diag_20260528/pagedpq_short_easy_n4_l120000_diag | slurm_out/lbv2_short_paired_diag_20260528/paired_diag-51051434.out |
