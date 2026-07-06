# Benchmark Pair Audit

Dense/frontier pairs are matched by stripping `dense_` / `pagedpq_` prefixes from artifact directories.

## Completed Pair Table

| pair | metric | dense | frontier | delta | frontier s/ex | runtime x | logical MB/hq | physical MB/hq | dense MB/hq | logical save | physical save | selected | active | pred same | judge same | text same | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lbv2_short_easy_n16_l120000 | accuracy_pct | 68.75 | n/a | n/a | n/a | n/a | n/a | n/a | 15.478 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | missing-frontier-run |
| livecodebench_codegen_off50_n10 | pass@1_pct | 60.00 | 60.00 | 0.00 | 11.83 | 0.99 | 0.449 | 0.449 | 0.403 | -11.5 | -11.5 | 919.0 | 0.000 | n/a | n/a | 10/10 | frontier-approx-path-inactive, frontier-zero-approx-calls, logical-bandwidth-worse-than-dense, physical-bandwidth-worse-than-dense |
| livecodebench_codegen_off60_n10 | pass@1_pct | 20.00 | 20.00 | 0.00 | 25.05 | 1.01 | 0.881 | 0.881 | 0.395 | -122.9 | -122.9 | 1804.4 | 0.000 | n/a | n/a | 10/10 | frontier-approx-path-inactive, frontier-zero-approx-calls, logical-bandwidth-worse-than-dense, physical-bandwidth-worse-than-dense |
| livecodebench_codegen_off70_n10 | pass@1_pct | 50.00 | 50.00 | 0.00 | 26.14 | 1.01 | 0.882 | 0.882 | 0.409 | -115.5 | -115.5 | 1805.5 | 0.000 | n/a | n/a | 10/10 | frontier-approx-path-inactive, frontier-zero-approx-calls, logical-bandwidth-worse-than-dense, physical-bandwidth-worse-than-dense |
| livecodebench_codegen_off80_n10 | pass@1_pct | 20.00 | 20.00 | 0.00 | 44.22 | 1.00 | 1.218 | 1.218 | 0.686 | -77.7 | -77.7 | 2495.3 | 0.000 | n/a | n/a | 10/10 | frontier-approx-path-inactive, frontier-zero-approx-calls, logical-bandwidth-worse-than-dense, physical-bandwidth-worse-than-dense |
| livecodebench_codegen_off90_n10 | pass@1_pct | 0.00 | 0.00 | 0.00 | 57.51 | 1.00 | 1.182 | 1.182 | 0.747 | -58.3 | -58.3 | 2421.3 | 0.000 | n/a | n/a | 10/10 | frontier-approx-path-inactive, frontier-zero-approx-calls, logical-bandwidth-worse-than-dense, physical-bandwidth-worse-than-dense |
| longgenbench_sgt_long_off0_n2 | substring_periodic_pct | 9.13 | 9.13 | 0.00 | 11382.44 | 7.24 | 3.744 | 5.172 | 8.186 | 54.3 | 36.8 | 10917.0 | 0.832 | n/a | n/a | 0/2 | ok |
| longgenbench_sgt_long_off2_n2 | substring_periodic_pct | 6.35 | 6.35 | 0.00 | 11465.53 | 7.28 | 3.710 | 5.150 | 8.188 | 54.7 | 37.1 | 10871.8 | 0.832 | n/a | n/a | 0/2 | ok |
| longgenbench_sgt_long_off4_n2 | substring_periodic_pct | 8.33 | 8.33 | 0.00 | 11440.49 | 7.27 | 3.724 | 5.167 | 8.186 | 54.5 | 36.9 | 10857.0 | 0.832 | n/a | n/a | 0/2 | ok |
| longgenbench_sgt_short_off0_n4 | substring_periodic_pct | 0.00 | 0.00 | 0.00 | 4083.65 | 5.97 | 2.381 | 2.874 | 4.313 | 44.8 | 33.4 | 6815.3 | 0.680 | n/a | n/a | 0/4 | ok |
| longgenbench_sgt_short_off12_n4 | substring_periodic_pct | 2.78 | 2.78 | 0.00 | 4072.50 | 5.95 | 2.365 | 2.869 | 4.315 | 45.2 | 33.5 | 6772.3 | 0.680 | n/a | n/a | 2/4 | ok |
| longgenbench_sgt_short_off28_n4 | substring_periodic_pct | n/a | 0.00 | n/a | 2528.41 | n/a | 2.404 | 2.884 | 4.314 | 44.3 | 33.2 | 6872.3 | 0.680 | n/a | n/a | n/a | missing-dense-run |
| longgenbench_sgt_short_off4_n4 | substring_periodic_pct | 1.67 | 1.67 | 0.00 | 4063.44 | 5.94 | 2.401 | 2.880 | 4.314 | 44.3 | 33.2 | 6871.9 | 0.680 | n/a | n/a | 0/4 | ok |
| longgenbench_sgt_short_off8_n4 | substring_periodic_pct | 2.78 | 0.00 | -2.78 | 4125.76 | 6.03 | 2.434 | 2.893 | 4.313 | 43.6 | 32.9 | 6955.3 | 0.680 | n/a | n/a | 0/4 | frontier-quality-lower |
| ruler_ctx32768_n1_niah_single_1 | score | 100.00 | n/a | n/a | n/a | n/a | n/a | n/a | 15.891 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | missing-frontier-run |

## LongGenBench SGT Metrics

For SGT, `periodic` and `range` are harder smoke checks than `once`. Official SGT paper numbers require an LLM yes/no judge; these substring metrics are artifact-only checks.

| pair | dense completion | frontier completion | dense once | frontier once | dense range | frontier range | dense periodic | frontier periodic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longgenbench_sgt_long_off0_n2 | 69.83 | 70.17 | 60.00 | 60.00 | 0.00 | 0.00 | 9.13 | 9.13 |
| longgenbench_sgt_long_off2_n2 | 79.17 | 79.00 | 90.00 | 90.00 | 28.33 | 58.33 | 6.35 | 6.35 |
| longgenbench_sgt_long_off4_n2 | 75.50 | 75.83 | 80.00 | 80.00 | 62.50 | 75.00 | 8.33 | 8.33 |
| longgenbench_sgt_short_off0_n4 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| longgenbench_sgt_short_off12_n4 | 100.00 | 100.00 | 0.00 | 0.00 | 25.00 | 25.00 | 2.78 | 2.78 |
| longgenbench_sgt_short_off28_n4 | n/a | 100.00 | n/a | 0.00 | n/a | 12.50 | n/a | 0.00 |
| longgenbench_sgt_short_off4_n4 | 100.00 | 100.00 | 0.00 | 0.00 | 25.00 | 25.00 | 1.67 | 1.67 |
| longgenbench_sgt_short_off8_n4 | 100.00 | 96.63 | 0.00 | 0.00 | 12.50 | 12.50 | 2.78 | 0.00 |

## Incomplete / Failed Runs

| label | reason | jobid | output_dir | slurm_out |
| --- | --- | --- | --- | --- |
| longbench_v2/pagedpq_lbv2_short_easy_n16_l120000 | partial-artifact-no-summary | n/a | /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/benchmark_suite_result/coalesced_benchmark_suite_20260526_172000_repack5/longbench_v2/pagedpq_lbv2_short_easy_n16_l120000 | n/a |
