# Multi-target RA vs RetroInfer-style proxy, 2026-05-01

## Setup

- Output: `attention_efficiency_result/proxy_cpu_v12_multi_targets_budget25_q16`
- Prefill tokens: `8192`
- Decode lengths: `0, 4096, 8192, 12288, 16384`
- Budget policy: `linear`, total token budget `25%` of current causal length
- RetroInfer target: `retroinfer_style`
- RetroInfer cluster scope: `prefill`
- Static pattern: prefix `128`, suffix `512`
- Cluster size: `128`
- Graph degree: `16`
- Queries: `16`
- Note: standard/spgpu Slurm partitions were down; this was run foreground on CPU.

## Raised RetroInfer target

Increasing the total budget from `10%` to `25%` raises RetroInfer-style mass:

| Decode | Retro alg read | Retro mass | Retro output cos |
| ---: | ---: | ---: | ---: |
| 0 | 0.2578 | 0.3881 | 0.4801 |
| 4096 | 0.2552 | 0.3014 | 0.3276 |
| 8192 | 0.2539 | 0.2573 | 0.2293 |
| 12288 | 0.2531 | 0.3129 | 0.3059 |
| 16384 | 0.2526 | 0.2606 | 0.3013 |

## RA absolute mass targets

| Decode | RA mass 0.1 | RA mass 0.2 | RA mass 0.4 | RA mass 0.6 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1602 | 0.1789 | 0.1791 | 0.3101 |
| 4096 | 0.1200 | 0.1202 | 0.1296 | 0.2563 |
| 8192 | 0.1029 | 0.1030 | 0.1077 | 0.3355 |
| 12288 | 0.0796 | 0.0863 | 0.1461 | 0.6157 |
| 16384 | 0.0661 | 0.0690 | 0.1463 | 0.7792 |

Entries are algorithmic read ratios. All listed targets reached in the q16 run.

## RA absolute cosine targets

| Decode | RA cos 0.2 | RA cos 0.4 | RA cos 0.6 | RA cos 0.8 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1789 | 0.1789 | 0.1789 | 0.1789 |
| 4096 | 0.1082 | 0.1202 | 0.1202 | 0.1202 |
| 8192 | 0.0834 | 0.1030 | 0.1030 | 0.1030 |
| 12288 | 0.0742 | 0.0803 | 0.0803 | 0.0803 |
| 16384 | 0.0609 | 0.0682 | 0.0682 | 0.0682 |

The repeated values mean one traversal checkpoint jumped past multiple cosine
thresholds at once.

## Interpretation

- Yes, raising the RetroInfer-style mass target is possible by increasing the
  total budget cap; at `25%`, RetroInfer-style mass is roughly `0.26-0.39`
  instead of the previous `~0.10-0.14`.
- RA remains algorithmically cheaper than RetroInfer-style at long decode for
  moderate targets (`mass <= 0.4`, `cos <= 0.8` in this q16 proxy).
- Very high mass (`0.6`) is not a free win: RA becomes more expensive than
  RetroInfer-style at long decode because it has to traverse much deeper.
- This suggests the strongest claim should be target-qualified:
  RetrievalAttention-style has an algorithmic advantage at iso-quality for
  moderate attention mass/output targets, but the advantage can disappear for
  high mass targets.
