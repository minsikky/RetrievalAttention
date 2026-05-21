# PQCache Variant Results

Variants implemented in `benchmark/attention_efficiency_threeway_eval.py`:
- `ivfpq_adaptive_c128`: score 128 coarse centroids, scan PQ codes only in selected buckets, increase `nprobe` until target mass is reached.
- `binary_gated_pqcache_b128`: scan 128-bit binary sidecar, keep top candidate budgets, PQ-rerank only retained candidates.

Main output root: `attention_efficiency_result/threeway_pqcache_variants_frontier_nograph_v1`
Candidate frontier plot: `attention_efficiency_result/plots/pqcache_variants/candidate_frontier_oracle_mass_vs_cost.png`

## Final MB at Target Mass 0.95

| Decode | RetroInfer | PQCache full scan | IVF-PQ adaptive | Binary-gated PQ |
| --- | ---: | ---: | ---: | ---: |
| 500 | 1.373 / 0.957 / 1.00 | 0.937 / 0.955 / 1.00 | 0.969 / 0.955 / 1.00 | 1.053 / 0.955 / 1.00 |
| 1000 | 1.331 / 0.954 / 1.00 | 0.976 / 0.951 / 1.00 | 1.001 / 0.951 / 1.00 | 1.143 / 0.951 / 1.00 |
| 2000 | 1.564 / 0.955 / 1.00 | 1.168 / 0.953 / 1.00 | 1.193 / 0.953 / 1.00 | 1.338 / 0.953 / 1.00 |
| 4000 | 2.028 / 0.954 / 1.00 | 1.405 / 0.952 / 1.00 | 1.415 / 0.952 / 1.00 | 1.549 / 0.952 / 1.00 |
| 8000 | 2.137 / 0.959 / 1.00 | 1.591 / 0.957 / 1.00 | 1.609 / 0.957 / 1.00 | 1.869 / 0.957 / 1.00 |
| 16000 | 2.603 / 0.955 / 1.00 | 1.911 / 0.953 / 1.00 | 1.871 / 0.953 / 1.00 | 2.268 / 0.953 / 1.00 |

## Final MB at Target Mass 0.98

| Decode | RetroInfer | PQCache full scan | IVF-PQ adaptive | Binary-gated PQ |
| --- | ---: | ---: | ---: | ---: |
| 500 | 2.086 / 0.982 / 1.00 | 1.520 / 0.981 / 1.00 | 1.531 / 0.981 / 1.00 | 1.621 / 0.981 / 1.00 |
| 1000 | 2.189 / 0.981 / 1.00 | 1.614 / 0.980 / 1.00 | 1.640 / 0.980 / 1.00 | 1.775 / 0.980 / 1.00 |
| 2000 | 2.466 / 0.982 / 1.00 | 1.845 / 0.980 / 1.00 | 1.885 / 0.980 / 1.00 | 2.012 / 0.980 / 1.00 |
| 4000 | 3.232 / 0.981 / 1.00 | 2.258 / 0.980 / 1.00 | 2.260 / 0.980 / 1.00 | 2.390 / 0.980 / 1.00 |
| 8000 | 3.453 / 0.981 / 1.00 | 2.551 / 0.981 / 1.00 | 2.558 / 0.981 / 1.00 | 2.766 / 0.981 / 0.97 |
| 16000 | 4.914 / 0.981 / 1.00 | 3.457 / 0.980 / 1.00 | 3.451 / 0.980 / 1.00 | 3.942 / 0.980 / 1.00 |

Cell format: `MB / mass / reach`.

## Pre-PQ Candidate Frontier

This measures your proposed diagnostic: how much oracle attention mass is inside the candidate set before PQ rerank, and how much routing memory was read to obtain that candidate set.

| Decode | IVF cost@0.95 | IVF budget@0.95 | Binary cost@0.95 | Binary budget@0.95 | IVF cost@0.98 | IVF budget@0.98 | Binary cost@0.98 | Binary budget@0.98 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.075 | 64 | 0.118 | 4096 | 0.088 | 128 | 0.128 | 6698 |
| 1000 | 0.069 | 32 | 0.125 | 4096 | 0.090 | 128 | 0.137 | 7198 |
| 2000 | 0.079 | 64 | 0.141 | 4096 | 0.094 | 128 | 0.156 | 8192 |
| 4000 | 0.082 | 64 | 0.171 | 4096 | 0.101 | 128 | 0.187 | 8192 |
| 8000 | 0.089 | 64 | 0.248 | 8192 | 0.117 | 128 | 0.271 | 14198 |
| 16000 | 0.085 | 32 | 0.401 | 16384 | 0.147 | 128 | 0.401 | 16384 |

## Interpretation

- IVF-PQ has the better pre-rerank routing frontier: it reaches 0.95/0.98 candidate oracle mass with lower routing MB than binary gating.
- Binary gating preserves token-level flexibility, but its hash false-negative/ordering issue forces much larger candidate budgets.
- Final target-mass MB is close to full PQCache because final exact K/V reads dominate at high mass; reducing routing cost alone is not enough.
- Next useful direction is improving candidate precision within IVF buckets or reducing final K/V bytes, not only reducing pre-PQ routing bytes.
