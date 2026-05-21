# Real QKV Proxy Results - 2026-05-01

## Trace

- Source trace: `attention_efficiency_result/real_xtrace_qkv_llama31_8b_layer16_8k_g16384_full.npz`
- Model/layer: `meta-llama/Llama-3.1-8B-Instruct`, layer 16
- Prompt/input length: 6838 tokens
- Generated trace length captured for queries: 16383 decode-input positions
- Stored tensors: `keys=(8, 23221, 128)`, `values=(8, 23221, 128)`, `queries=(32, 16383, 128)`, plus layer inputs

## q192 CPU Proxy

- Job: `49089436`
- Output: `attention_efficiency_result/real_qkv_proxy_llama31_l16_6838_g16384_full_cutoffs`
- Status: completed, elapsed `01:07:49`
- Source NPZ: `attention_efficiency_result/real_qkv_llama31_l16_6838_cutoffs_from_g16384_full.npz`
- Query samples: 192 total, 32 heads x 6 decode cutoffs
- Settings: total budget, linear ratio `0.25`, static prefix/suffix `128/512`, graph degree `16`, `RA_VISIT_BUDGET=0`, adaptive check interval `1`

Key summary:

| Decode | Retro algo | Retro mass | Retro cos | RA mass-match algo | RA token | RA mass-match cos | RA mass reached |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.257 | 0.285 | 0.963 | 3.097 | 0.240 | 0.932 | 1.00 |
| 1000 | 0.257 | 0.282 | 0.980 | 2.993 | 0.233 | 0.914 | 1.00 |
| 2000 | 0.256 | 0.274 | 0.995 | 3.224 | 0.237 | 0.952 | 1.00 |
| 4000 | 0.255 | 0.265 | 0.976 | 3.319 | 0.232 | 0.968 | 1.00 |
| 8000 | 0.254 | 0.259 | 0.920 | 3.487 | 0.229 | 0.969 | 1.00 |
| 16000 | 0.252 | 0.256 | 0.838 | 3.754 | 0.226 | 0.931 | 0.91 |

Interpretation:

- RA mass-match uses fewer selected tokens than RetroInfer-style (`~0.226-0.240` vs `~0.252-0.257` token-read ratio).
- RA mass-match algorithmic read is much worse (`~3.0-3.8x` causal context) because graph traversal metadata dominates.
- The mass target sweep confirms moderate fixed mass (`0.2`) is attainable but expensive (`~1.6-2.7x` algorithmic read). Higher mass (`0.4`, `0.6`) is not reached under the 25% total token budget.
- Real-QKV q192 confirms the q24 directional result, so this is not just small-sample noise.

## GPU Attempt

Implemented `PrecomputedOnlineKnnGraph` in `benchmark/attention_efficiency_eval.py`:

- Builds the prefill KNN graph and generated-token online overlay using batched matmuls.
- Selectable with `--ra_graph_backend precomputed`; wrappers expose `RA_GRAPH_BACKEND` and `RA_PRECOMPUTE_CHUNK`.
- Lazy/precomputed semantics matched exactly on a small CPU smoke test.

Jobs:

- `49093516`: GPU precomputed, chunk 512. Canceled after `00:30:48`; GPU utilization stayed around `1-2%`, no rows emitted.
- `49094125`: GPU precomputed, chunk 4096. Canceled after `00:32:24`; GPU memory increased to ~2.3 GB but utilization still stayed around `2-3%`, no rows emitted.

Stack sample for `49094125` showed the job in Python heap traversal (`long_richcompare`/`heapq` path), not GPU scoring. Conclusion: GPU precomputing neighbor discovery alone is insufficient. To make GPU useful, traversal must move out of Python, likely into C++/CUDA or a vectorized frontier expansion kernel.

## Byte-Cost Model Update

Implemented a byte-weighted proxy cost in `benchmark/attention_efficiency_eval.py` and wrappers:

- Main summary metric is now `estimated_read_mb`, not token-equivalent `algorithmic_read_ratio`.
- Final attention cost: selected tokens x `(K_attn_bytes + V_bytes)` x head dimension.
- Search cost: candidate/centroid K-score reads x `score_key_bytes` x head dimension.
- Graph metadata cost: edge IDs and row offsets as byte-sized metadata, not token-equivalent reads.
- Raw sample rows still record detailed counters such as `key_score_reads`, `rerank_key_reads`, `edge_index_reads`, and byte components.

Small real-QKV validation:

- Job: `49094631`
- Output: `attention_efficiency_result/real_qkv_proxy_llama31_l16_6838_g16384_full_q24_bytecost`
- Status: completed, elapsed `00:11:20`
- Defaults: score K fp32 (`4 B/elem`), attention K/V bf16/fp16 (`2 B/elem`), edge ID `4 B`, offset `4 B`, rerank cost included.

| Decode | Retro MB | Retro mass | RA mass-match MB | RA mass | RA token ratio |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.92 | 0.285 | 2.69 | 0.285 | 0.244 |
| 1000 | 0.98 | 0.275 | 2.82 | 0.276 | 0.237 |
| 2000 | 1.11 | 0.280 | 3.30 | 0.280 | 0.238 |
| 4000 | 1.35 | 0.265 | 4.10 | 0.265 | 0.228 |
| 8000 | 1.84 | 0.253 | 5.68 | 0.254 | 0.223 |
| 16000 | 2.81 | 0.258 | 9.15 | 0.258 | 0.228 |

Interpretation: byte-weighting removes the bogus edge-token equivalence, but RA mass-match is still ~3x RetroInfer-style bytes on q24 because K scoring/rerank reads dominate.
