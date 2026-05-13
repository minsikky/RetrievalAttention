# Self-Indexing-Style Selector Results, Fixed Cost Model

This regenerates the selector-focused nograph run after fixing compressed-sidecar accounting. Binary-gated and weighted-Hamming selectors now use prebuilt packed sidecar indices; Sign-VQ LUT charges packed sign-pattern bits. Methods with `oracle` in the name still use true attention mass to choose the first sufficient budget, so they are lower-envelope selector frontiers rather than deployable stopping policies.

Output root: `attention_efficiency_result/threeway_self_indexing_selectors_nograph_v2_fixed`
Frontier plot: `attention_efficiency_result/plots/self_indexing_selectors_v2_fixed/candidate_frontier_oracle_mass_vs_cost.png`

Cell format: `MB / mass / reach`.

## Final Selector MB, Target Mass 0.95

| Decode | PQCache full scan | IVF + global PQ oracle | Raw binary gate oracle | Weighted Hamming oracle | Sign-VQ LUT oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.937 / 0.955 / 1.00 | 0.969 / 0.955 / 1.00 | 1.054 / 0.955 / 1.00 | 0.930 / 0.955 / 1.00 | 0.930 / 0.955 / 1.00 |
| 1000 | 0.976 / 0.951 / 1.00 | 1.001 / 0.951 / 1.00 | 1.142 / 0.951 / 1.00 | 1.011 / 0.951 / 1.00 | 1.010 / 0.951 / 1.00 |
| 2000 | 1.168 / 0.953 / 1.00 | 1.193 / 0.953 / 1.00 | 1.311 / 0.953 / 1.00 | 1.236 / 0.953 / 1.00 | 1.198 / 0.953 / 1.00 |
| 4000 | 1.405 / 0.952 / 1.00 | 1.415 / 0.952 / 1.00 | 1.575 / 0.952 / 1.00 | 1.452 / 0.952 / 1.00 | 1.460 / 0.952 / 1.00 |
| 8000 | 1.591 / 0.957 / 1.00 | 1.609 / 0.957 / 1.00 | 1.821 / 0.957 / 1.00 | 1.720 / 0.957 / 1.00 | 1.727 / 0.957 / 1.00 |
| 16000 | 1.911 / 0.953 / 1.00 | 1.871 / 0.953 / 1.00 | 2.219 / 0.953 / 1.00 | 2.114 / 0.953 / 1.00 | 2.073 / 0.953 / 1.00 |

## Final Selector MB, Target Mass 0.98

| Decode | PQCache full scan | IVF + global PQ oracle | Raw binary gate oracle | Weighted Hamming oracle | Sign-VQ LUT oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 1.520 / 0.981 / 1.00 | 1.531 / 0.981 / 1.00 | 1.620 / 0.980 / 1.00 | 1.433 / 0.981 / 1.00 | 1.448 / 0.981 / 1.00 |
| 1000 | 1.614 / 0.980 / 1.00 | 1.641 / 0.980 / 1.00 | 1.768 / 0.980 / 1.00 | 1.591 / 0.980 / 0.97 | 1.554 / 0.980 / 0.97 |
| 2000 | 1.845 / 0.980 / 1.00 | 1.885 / 0.980 / 1.00 | 1.988 / 0.980 / 1.00 | 1.891 / 0.980 / 1.00 | 1.854 / 0.980 / 1.00 |
| 4000 | 2.258 / 0.980 / 1.00 | 2.261 / 0.980 / 1.00 | 2.392 / 0.980 / 1.00 | 2.229 / 0.980 / 1.00 | 2.238 / 0.980 / 1.00 |
| 8000 | 2.551 / 0.981 / 1.00 | 2.558 / 0.981 / 1.00 | 2.782 / 0.981 / 1.00 | 2.662 / 0.981 / 1.00 | 2.579 / 0.981 / 1.00 |
| 16000 | 3.457 / 0.980 / 1.00 | 3.451 / 0.980 / 1.00 | 3.837 / 0.980 / 1.00 | 3.573 / 0.980 / 1.00 | 3.467 / 0.980 / 1.00 |

## First Budget Reaching Candidate Oracle Mass

This is pre-PQ rerank: cost to form the candidate set versus exact/oracle attention mass contained in that candidate set.

| Decode | Selector | cost@0.95 | budget@0.95 | cost@0.98 | budget@0.98 |
| --- | --- | ---: | ---: | ---: | ---: |
| 500 | IVF + global PQ | 0.076 | 64 | 0.089 | 128 |
| 500 | Raw binary gate | 0.102 | 4096 | 0.102 | 6698 |
| 500 | Weighted Hamming | 0.102 | 2048 | 0.102 | 4096 |
| 500 | Sign-VQ LUT | 0.110 | 2048 | 0.110 | 4096 |
| 1000 | IVF + global PQ | 0.069 | 32 | 0.090 | 128 |
| 1000 | Raw binary gate | 0.110 | 4096 | 0.110 | 7198 |
| 1000 | Weighted Hamming | 0.110 | 2048 | 0.110 | 4096 |
| 1000 | Sign-VQ LUT | 0.118 | 2048 | 0.118 | 4096 |
| 2000 | IVF + global PQ | 0.079 | 64 | 0.094 | 128 |
| 2000 | Raw binary gate | 0.125 | 4096 | 0.125 | 8192 |
| 2000 | Weighted Hamming | 0.125 | 4096 | 0.125 | 8192 |
| 2000 | Sign-VQ LUT | 0.133 | 4096 | 0.133 | 8192 |
| 4000 | IVF + global PQ | 0.083 | 64 | 0.102 | 128 |
| 4000 | Raw binary gate | 0.156 | 4096 | 0.156 | 8192 |
| 4000 | Weighted Hamming | 0.156 | 4096 | 0.156 | 8192 |
| 4000 | Sign-VQ LUT | 0.163 | 4096 | 0.163 | 8192 |
| 8000 | IVF + global PQ | 0.089 | 64 | 0.117 | 128 |
| 8000 | Raw binary gate | 0.217 | 8192 | 0.217 | 14198 |
| 8000 | Weighted Hamming | 0.217 | 8192 | 0.217 | 14198 |
| 8000 | Sign-VQ LUT | 0.224 | 8192 | 0.224 | 14198 |
| 16000 | IVF + global PQ | 0.085 | 32 | 0.148 | 128 |
| 16000 | Raw binary gate | 0.339 | 8192 | 0.339 | 16384 |
| 16000 | Weighted Hamming | 0.339 | 8192 | 0.339 | 16384 |
| 16000 | Sign-VQ LUT | 0.347 | 8192 | 0.347 | 16384 |

## Raw Files

- Merged summary CSV: `attention_efficiency_result/threeway_self_indexing_selectors_nograph_v2_fixed/merged_summary.csv`
- Merged summary JSON: `attention_efficiency_result/threeway_self_indexing_selectors_nograph_v2_fixed/merged_summary.json`
- Candidate frontier CSV: `attention_efficiency_result/threeway_self_indexing_selectors_nograph_v2_fixed/merged_candidate_frontier_summary.csv`
- Candidate frontier JSON: `attention_efficiency_result/threeway_self_indexing_selectors_nograph_v2_fixed/merged_candidate_frontier_summary.json`
