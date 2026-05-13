# Selector Proxy Results on 128k Real Trace

Source trace: `attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz`
QKV proxy input: `attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q36_graphall_s16.npz`
Output root: `attention_efficiency_result/threeway_self_indexing_selectors_g131072_nograph_v1_fixed`
MB plot: `attention_efficiency_result/plots/self_indexing_selectors_g131072_v1_fixed/final_selector_mb_vs_decode.png`
Candidate-frontier plot: `attention_efficiency_result/plots/self_indexing_selectors_g131072_v1_fixed/candidate_frontier_oracle_mass_vs_cost.png`

Cell format: `MB / mass / reach`. Methods with `oracle` use true mass to select the first sufficient budget, so they are lower-envelope selector frontiers.

## Target Mass 0.95

| Decode | PQCache full scan | IVF + global PQ oracle | Raw binary gate oracle | Weighted Hamming oracle | Sign-VQ LUT oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.563 / 0.962 / 1.00 | 0.602 / 0.962 / 1.00 | 0.649 / 0.962 / 1.00 | 0.593 / 0.962 / 1.00 | 0.603 / 0.962 / 1.00 |
| 1000 | 0.831 / 0.956 / 1.00 | 0.860 / 0.956 / 1.00 | 0.944 / 0.956 / 1.00 | 0.846 / 0.956 / 1.00 | 0.823 / 0.956 / 1.00 |
| 2000 | 0.852 / 0.955 / 1.00 | 0.877 / 0.955 / 1.00 | 0.962 / 0.955 / 1.00 | 0.878 / 0.955 / 1.00 | 0.860 / 0.955 / 1.00 |
| 4000 | 1.050 / 0.953 / 1.00 | 1.087 / 0.953 / 1.00 | 1.196 / 0.953 / 1.00 | 1.120 / 0.953 / 1.00 | 1.125 / 0.953 / 1.00 |
| 8000 | 1.609 / 0.951 / 1.00 | 1.619 / 0.951 / 1.00 | 1.856 / 0.951 / 1.00 | 1.704 / 0.951 / 1.00 | 1.692 / 0.951 / 1.00 |
| 16000 | 2.413 / 0.951 / 1.00 | 2.415 / 0.951 / 1.00 | 2.768 / 0.952 / 1.00 | 2.501 / 0.951 / 1.00 | 2.477 / 0.952 / 1.00 |
| 32000 | 3.846 / 0.952 / 1.00 | 3.647 / 0.952 / 1.00 | 4.334 / 0.952 / 1.00 | 3.900 / 0.952 / 1.00 | 4.004 / 0.952 / 1.00 |
| 64000 | 5.276 / 0.951 / 1.00 | 4.987 / 0.951 / 1.00 | 7.385 / 0.938 / 0.75 | 6.019 / 0.951 / 0.94 | 5.703 / 0.951 / 1.00 |
| 128000 | 8.970 / 0.953 / 1.00 | 7.928 / 0.952 / 1.00 | 13.461 / 0.880 / 0.41 | 13.672 / 0.881 / 0.41 | 9.482 / 0.947 / 0.84 |

## Target Mass 0.98

| Decode | PQCache full scan | IVF + global PQ oracle | Raw binary gate oracle | Weighted Hamming oracle | Sign-VQ LUT oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.931 / 0.983 / 1.00 | 0.927 / 0.983 / 1.00 | 1.026 / 0.982 / 1.00 | 0.916 / 0.982 / 1.00 | 0.921 / 0.982 / 1.00 |
| 1000 | 1.360 / 0.981 / 1.00 | 1.385 / 0.981 / 1.00 | 1.470 / 0.981 / 1.00 | 1.296 / 0.981 / 1.00 | 1.264 / 0.981 / 1.00 |
| 2000 | 1.359 / 0.981 / 1.00 | 1.385 / 0.981 / 1.00 | 1.486 / 0.981 / 1.00 | 1.373 / 0.981 / 1.00 | 1.351 / 0.981 / 1.00 |
| 4000 | 1.687 / 0.980 / 1.00 | 1.711 / 0.980 / 1.00 | 1.871 / 0.980 / 1.00 | 1.747 / 0.980 / 1.00 | 1.717 / 0.980 / 1.00 |
| 8000 | 2.583 / 0.980 / 1.00 | 2.602 / 0.980 / 1.00 | 2.809 / 0.980 / 1.00 | 2.636 / 0.980 / 1.00 | 2.616 / 0.980 / 1.00 |
| 16000 | 4.030 / 0.980 / 1.00 | 4.037 / 0.980 / 1.00 | 4.286 / 0.980 / 1.00 | 4.000 / 0.980 / 1.00 | 3.921 / 0.980 / 1.00 |
| 32000 | 6.584 / 0.980 / 1.00 | 6.364 / 0.980 / 1.00 | 7.075 / 0.980 / 0.94 | 6.355 / 0.980 / 1.00 | 6.480 / 0.980 / 1.00 |
| 64000 | 10.218 / 0.980 / 1.00 | 9.574 / 0.980 / 1.00 | 10.535 / 0.958 / 0.66 | 9.361 / 0.974 / 0.75 | 9.278 / 0.978 / 0.84 |
| 128000 | 14.236 / 0.981 / 1.00 | 13.075 / 0.982 / 1.00 | 15.751 / 0.891 / 0.28 | 16.630 / 0.890 / 0.19 | 12.291 / 0.970 / 0.72 |

## First Budget Reaching Candidate Oracle Mass

| Decode | Selector | cost@0.95 | budget@0.95 | cost@0.98 | budget@0.98 |
| --- | --- | ---: | ---: | ---: | ---: |
| 500 | IVF + global PQ | 0.066 | 16 | 0.076 | 64 |
| 500 | Raw binary gate | 0.102 | 512 | 0.102 | 4096 |
| 500 | Weighted Hamming | 0.102 | 512 | 0.102 | 4096 |
| 500 | Sign-VQ LUT | 0.110 | 512 | 0.110 | 4096 |
| 1000 | IVF + global PQ | 0.070 | 32 | 0.090 | 128 |
| 1000 | Raw binary gate | 0.110 | 4096 | 0.110 | 7198 |
| 1000 | Weighted Hamming | 0.110 | 2048 | 0.110 | 4096 |
| 1000 | Sign-VQ LUT | 0.118 | 2048 | 0.118 | 4096 |
| 2000 | IVF + global PQ | 0.071 | 32 | 0.079 | 64 |
| 2000 | Raw binary gate | 0.125 | 2048 | 0.125 | 8192 |
| 2000 | Weighted Hamming | 0.125 | 2048 | 0.125 | 4096 |
| 2000 | Sign-VQ LUT | 0.133 | 2048 | 0.133 | 4096 |
| 4000 | IVF + global PQ | 0.073 | 32 | 0.102 | 128 |
| 4000 | Raw binary gate | 0.156 | 4096 | 0.156 | 8192 |
| 4000 | Weighted Hamming | 0.156 | 4096 | 0.156 | 8192 |
| 4000 | Sign-VQ LUT | 0.163 | 2048 | 0.163 | 8192 |
| 8000 | IVF + global PQ | 0.089 | 64 | 0.117 | 128 |
| 8000 | Raw binary gate | 0.217 | 8192 | 0.217 | 14198 |
| 8000 | Weighted Hamming | 0.217 | 4096 | 0.217 | 8192 |
| 8000 | Sign-VQ LUT | 0.224 | 4096 | 0.224 | 8192 |
| 16000 | IVF + global PQ | 0.107 | 64 | 0.148 | 128 |
| 16000 | Raw binary gate | 0.339 | 16384 | 0.339 | 22198 |
| 16000 | Weighted Hamming | 0.339 | 8192 | 0.339 | 16384 |
| 16000 | Sign-VQ LUT | 0.347 | 8192 | 0.347 | 16384 |
| 32000 | IVF + global PQ | 0.135 | 64 | 0.209 | 128 |
| 32000 | Raw binary gate | 0.583 | 16384 | 0.583 | 32768 |
| 32000 | Weighted Hamming | 0.583 | 16384 | 0.583 | 32768 |
| 32000 | Sign-VQ LUT | 0.591 | 16384 | 0.591 | 32768 |
| 64000 | IVF + global PQ | 0.196 | 64 | 0.331 | 128 |
| 64000 | Raw binary gate | - | - | - | - |
| 64000 | Weighted Hamming | 1.071 | 32768 | - | - |
| 64000 | Sign-VQ LUT | 1.079 | 16384 | 1.079 | 32768 |
| 128000 | IVF + global PQ | 0.188 | 32 | 0.320 | 64 |
| 128000 | Raw binary gate | - | - | - | - |
| 128000 | Weighted Hamming | - | - | - | - |
| 128000 | Sign-VQ LUT | 2.056 | 32768 | - | - |

## Raw Files

- Merged summary CSV: `attention_efficiency_result/threeway_self_indexing_selectors_g131072_nograph_v1_fixed/merged_summary.csv`
- Candidate frontier CSV: `attention_efficiency_result/threeway_self_indexing_selectors_g131072_nograph_v1_fixed/merged_candidate_frontier_summary.csv`
