# RetrievalAttention vs RetroInfer-style proxy, 2026-04-30

## Setup

- Output: `attention_efficiency_result/proxy_cpu_v10_long_decode_ra_vs_retro_style`
- Optimized adaptive-stop output: `attention_efficiency_result/proxy_cpu_v11_ra_match_retro_style_stop1`
- Prefill tokens: `8192`
- Decode lengths: `0, 4096, 8192, 12288, 16384`
- Budget policy: `linear`, total token budget `10%` of current causal length
- RetroInfer target: `retroinfer_style`
- RetroInfer cluster scope: `prefill`
- Static pattern: prefix `128`, suffix `512`
- Cluster size: `128`
- Graph degree: `16`
- Queries: `64`

## Main result

Fixed-budget RetrievalAttention-style traversal reads much more metadata than
RetroInfer-style, but has much higher attention quality. The cleaner comparison
is adaptive RA: stop when it reaches the RetroInfer-style mass or cosine target.

With per-visit adaptive checks (`ADAPTIVE_CHECK_INTERVAL=1`), RA becomes
algorithmically cheaper than RetroInfer-style once decode is nontrivial:

| Decode | Retro alg read | RA mass-match alg read | Mass-match / Retro | RA cos-match alg read | Cos-match / Retro |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1079 | 0.1395 | 1.2929 | 0.1164 | 1.0788 |
| 4096 | 0.1052 | 0.1045 | 0.9927 | 0.0741 | 0.7038 |
| 8192 | 0.1039 | 0.1013 | 0.9747 | 0.0638 | 0.6142 |
| 12288 | 0.1031 | 0.0829 | 0.8040 | 0.0589 | 0.5710 |
| 16384 | 0.1026 | 0.0847 | 0.8250 | 0.0537 | 0.5235 |

The traversal itself also stays small in the mass/cos match setting:

| Decode | RA mass-match visited | RA cos-match visited |
| ---: | ---: | ---: |
| 0 | 28.20 | 17.69 |
| 4096 | 28.58 | 11.91 |
| 8192 | 41.91 | 16.56 |
| 12288 | 42.84 | 22.53 |
| 16384 | 58.84 | 27.89 |

## Interpretation

- This supports the algorithmic-efficiency hypothesis in the long-decode regime:
  RetrievalAttention-style dynamic traversal can match RetroInfer-style quality
  while reading fewer algorithmic units.
- The result is not true for short/prefill-only context; RA is still more
  expensive at decode length `0`.
- Fixed-budget RA remains expensive in this proxy (`~1.8x` algorithmic read
  ratio), so the win depends on adaptive stopping, not blindly spending the
  full 10% budget.
- This is still a proxy using synthetic QKV geometry. The next credible step is
  the same comparison with real model/oracle diagnostics on GPU decode.
