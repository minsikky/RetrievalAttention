# PQCache Quantized-K Attention Results

This is intentionally not a pure selector-only comparison. `pqcache_quantized_k_m2_b6` uses PQ codes both to select tokens and to provide approximate dynamic-token logits, so selected dynamic tokens read V only instead of exact K+V. Static windows remain exact K/V.

Cell format: `MB / mass / output_cos / reach`.

## Target Mass 0.95

| Decode | RetroInfer | PQCache selector + exact K/V | PQCache selector + quantized K |
| --- | ---: | ---: | ---: |
| 500 | 1.373 / 0.957 / 0.996 / 1.00 | 0.937 / 0.955 / 0.995 / 1.00 | 0.647 / 0.955 / 0.982 / 1.00 |
| 1000 | 1.331 / 0.954 / 0.994 / 1.00 | 0.976 / 0.951 / 0.994 / 1.00 | 0.667 / 0.951 / 0.980 / 1.00 |
| 2000 | 1.564 / 0.955 / 0.993 / 1.00 | 1.168 / 0.953 / 0.992 / 1.00 | 0.764 / 0.953 / 0.979 / 1.00 |
| 4000 | 2.028 / 0.954 / 0.991 / 1.00 | 1.405 / 0.952 / 0.991 / 1.00 | 0.884 / 0.952 / 0.975 / 1.00 |
| 8000 | 2.137 / 0.959 / 0.993 / 1.00 | 1.591 / 0.957 / 0.992 / 1.00 | 0.981 / 0.957 / 0.984 / 1.00 |
| 16000 | 2.603 / 0.955 / 0.991 / 1.00 | 1.911 / 0.953 / 0.990 / 1.00 | 1.149 / 0.953 / 0.976 / 1.00 |

## Target Mass 0.98

| Decode | RetroInfer | PQCache selector + exact K/V | PQCache selector + quantized K |
| --- | ---: | ---: | ---: |
| 500 | 2.086 / 0.982 / 0.999 / 1.00 | 1.520 / 0.981 / 0.999 / 1.00 | 0.938 / 0.981 / 0.986 / 1.00 |
| 1000 | 2.189 / 0.981 / 0.999 / 1.00 | 1.614 / 0.980 / 0.999 / 1.00 | 0.986 / 0.980 / 0.984 / 1.00 |
| 2000 | 2.466 / 0.982 / 0.999 / 1.00 | 1.845 / 0.980 / 0.998 / 1.00 | 1.102 / 0.980 / 0.985 / 1.00 |
| 4000 | 3.232 / 0.981 / 0.998 / 1.00 | 2.258 / 0.980 / 0.999 / 1.00 | 1.311 / 0.980 / 0.980 / 1.00 |
| 8000 | 3.453 / 0.981 / 0.998 / 1.00 | 2.551 / 0.981 / 0.998 / 1.00 | 1.461 / 0.981 / 0.988 / 1.00 |
| 16000 | 4.914 / 0.981 / 0.999 / 1.00 | 3.457 / 0.980 / 0.999 / 1.00 | 1.921 / 0.980 / 0.982 / 1.00 |

## Interpretation

- Quantized-K attention significantly reduces MB because selected dynamic tokens no longer read exact K.
- The cost is lower output cosine: the selected token set can have the same exact attention mass, but PQ logits distort the softmax weights.
- This should be reported separately from selector-only methods. Mass measures token-set coverage; output cosine/logit error measures whether approximate K is good enough for attention computation.
- Next useful knobs: more PQ subspaces, more centroid bits, residual PQ, norm correction, or exact-K escape tier for high-error selected tokens.
