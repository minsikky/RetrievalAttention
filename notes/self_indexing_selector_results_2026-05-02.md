# Self-Indexing-Style Selector Results

Reference: Self-Indexing KVCache, arXiv:2603.14224. The current `sign_vq_lut_pqcache_g4` proxy implements the selector idea only: 4-d sign-pattern codes, per-pattern centroids from actual K subvectors, query LUT scoring, candidate selection, then PQ rerank. It is not a full paper replication.

Output root: `attention_efficiency_result/threeway_self_indexing_selectors_nograph_v1`
Frontier plot: `attention_efficiency_result/plots/self_indexing_selectors/candidate_frontier_oracle_mass_vs_cost.png`

## Final Selector MB

Cell format: `MB / mass / reach`.

### Target Mass 0.95

| Decode | PQCache full scan | IVF-PQ | Raw binary gate | Weighted Hamming | Sign-VQ LUT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.937 / 0.955 / 1.00 | 0.969 / 0.955 / 1.00 | 1.053 / 0.955 / 1.00 | 0.930 / 0.955 / 1.00 | 1.032 / 0.955 / 1.00 |
| 1000 | 0.976 / 0.951 / 1.00 | 1.001 / 0.951 / 1.00 | 1.143 / 0.951 / 1.00 | 1.011 / 0.951 / 1.00 | 1.120 / 0.951 / 1.00 |
| 2000 | 1.168 / 0.953 / 1.00 | 1.193 / 0.953 / 1.00 | 1.338 / 0.953 / 1.00 | 1.236 / 0.953 / 1.00 | 1.323 / 0.953 / 1.00 |
| 4000 | 1.405 / 0.952 / 1.00 | 1.415 / 0.952 / 1.00 | 1.549 / 0.952 / 1.00 | 1.452 / 0.952 / 1.00 | 1.615 / 0.952 / 1.00 |
| 8000 | 1.591 / 0.957 / 1.00 | 1.609 / 0.957 / 1.00 | 1.869 / 0.957 / 1.00 | 1.720 / 0.957 / 1.00 | 1.944 / 0.957 / 1.00 |
| 16000 | 1.911 / 0.953 / 1.00 | 1.871 / 0.953 / 1.00 | 2.268 / 0.953 / 1.00 | 2.114 / 0.953 / 1.00 | 2.412 / 0.953 / 1.00 |

### Target Mass 0.98

| Decode | PQCache full scan | IVF-PQ | Raw binary gate | Weighted Hamming | Sign-VQ LUT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 1.520 / 0.981 / 1.00 | 1.531 / 0.981 / 1.00 | 1.621 / 0.981 / 1.00 | 1.433 / 0.981 / 1.00 | 1.550 / 0.981 / 1.00 |
| 1000 | 1.614 / 0.980 / 1.00 | 1.640 / 0.980 / 1.00 | 1.775 / 0.980 / 1.00 | 1.591 / 0.980 / 0.97 | 1.664 / 0.980 / 0.97 |
| 2000 | 1.845 / 0.980 / 1.00 | 1.885 / 0.980 / 1.00 | 2.012 / 0.980 / 1.00 | 1.891 / 0.980 / 1.00 | 1.979 / 0.980 / 1.00 |
| 4000 | 2.258 / 0.980 / 1.00 | 2.260 / 0.980 / 1.00 | 2.390 / 0.980 / 1.00 | 2.229 / 0.980 / 1.00 | 2.393 / 0.980 / 1.00 |
| 8000 | 2.551 / 0.981 / 1.00 | 2.558 / 0.981 / 1.00 | 2.766 / 0.981 / 0.97 | 2.662 / 0.981 / 1.00 | 2.796 / 0.981 / 1.00 |
| 16000 | 3.457 / 0.980 / 1.00 | 3.451 / 0.980 / 1.00 | 3.942 / 0.980 / 1.00 | 3.573 / 0.980 / 1.00 | 3.806 / 0.980 / 1.00 |

## First Budget Reaching Candidate Oracle Mass

This is pre-PQ rerank: cost to form the candidate set vs exact/oracle mass contained in that set.

| Decode | Selector | cost@0.95 | budget@0.95 | cost@0.98 | budget@0.98 |
| --- | --- | ---: | ---: | ---: | ---: |
| 500 | IVF-PQ | 0.075 | 64 | 0.088 | 128 |
| 500 | Raw binary | 0.118 | 4096 | 0.128 | 6698 |
| 500 | Weighted Hamming | 0.110 | 2048 | 0.118 | 4096 |
| 500 | Sign-VQ LUT | 0.220 | 2048 | 0.228 | 4096 |
| 1000 | IVF-PQ | 0.069 | 32 | 0.090 | 128 |
| 1000 | Raw binary | 0.125 | 4096 | 0.137 | 7198 |
| 1000 | Weighted Hamming | 0.118 | 2048 | 0.125 | 4096 |
| 1000 | Sign-VQ LUT | 0.235 | 2048 | 0.243 | 4096 |
| 2000 | IVF-PQ | 0.079 | 64 | 0.094 | 128 |
| 2000 | Raw binary | 0.141 | 4096 | 0.156 | 8192 |
| 2000 | Weighted Hamming | 0.141 | 4096 | 0.156 | 8192 |
| 2000 | Sign-VQ LUT | 0.274 | 4096 | 0.289 | 8192 |
| 4000 | IVF-PQ | 0.082 | 64 | 0.101 | 128 |
| 4000 | Raw binary | 0.171 | 4096 | 0.187 | 8192 |
| 4000 | Weighted Hamming | 0.171 | 4096 | 0.187 | 8192 |
| 4000 | Sign-VQ LUT | 0.335 | 4096 | 0.350 | 8192 |
| 8000 | IVF-PQ | 0.089 | 64 | 0.117 | 128 |
| 8000 | Raw binary | 0.248 | 8192 | 0.271 | 14198 |
| 8000 | Weighted Hamming | 0.248 | 8192 | 0.271 | 14198 |
| 8000 | Sign-VQ LUT | 0.472 | 8192 | 0.495 | 14198 |
| 16000 | IVF-PQ | 0.085 | 32 | 0.147 | 128 |
| 16000 | Raw binary | 0.401 | 16384 | 0.401 | 16384 |
| 16000 | Weighted Hamming | 0.370 | 8192 | 0.401 | 16384 |
| 16000 | Sign-VQ LUT | 0.716 | 8192 | 0.748 | 16384 |

## Interpretation

- The previous `binary_gated_pqcache_b128` should be labeled raw random-projection/Hamming gate, not Self-Indexing.
- Query-weighted Hamming is a strong improvement over raw binary gating and sometimes beats full PQCache at short decode, but it loses to IVF-PQ at long decode.
- Sign-VQ LUT improves candidate mass versus raw binary gating for a fixed candidate budget, but its code/LUT scan cost is higher in this proxy, so final MB is not competitive yet.
- IVF-PQ remains the best first-stage selector frontier: much lower pre-PQ cost for the same candidate oracle mass, especially at long decode.
