# Real layer-input trace plan, 2026-05-01

## Rationale

For long-decode experiments, prefill-only QKV is insufficient. Decode queries
depend on generated tokens and hidden states, so we need to run real decode at
least once.

The reusable artifact should be the target layer input hidden states `X_l`, not
only Q/K/V. With `X_l` plus the layer's RMSNorm/projection/RoPE metadata, we can
reconstruct Q/K/V offline for arbitrary cutoffs:

- decode `500`
- decode `1000`
- decode `2000`
- decode `4000`
- decode `8000`
- decode `16000`
- decode `32000`

Then the same trace can support RetroInfer-style, RetrievalAttention-style, and
online graph-update studies without rerunning the model.

## Implemented

- `scripts/dump_real_qkv_trace.py`
  - can save `layer_inputs`
  - can skip full QKV cache save via `--skip_qkv`
  - saves projection weights, layernorm weight/eps, and RoPE cache
  - supports `Full_Flash_Attn`, `RetroInfer`, and `RetrievalAttention`
- `scripts/run_dump_real_qkv_trace.sh`
  - Slurm wrapper for trace dumps
- `scripts/convert_layer_trace_to_qkv_npz.py`
  - converts saved `X_l` trace into the existing QKV NPZ format for requested decode cutoffs
- `benchmark/attention_efficiency_eval.py`
  - NPZ loader now respects trace metadata and real decode positions

## Queued jobs

Current blocker: Slurm partitions were down at submission time.

| Job | Purpose | Output |
| ---: | --- | --- |
| `49083536` | 16k Full_Flash_Attn layer-16 `X_l` trace | `attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g16384_full.npz` |
| `49083540` | convert 16k trace to QKV cutoff NPZ | `attention_efficiency_result/real_qkv_llama31_l16_8k_cutoffs_from_g16384_full.npz` |
| `49083542` | proxy sweep on 16k real QKV cutoffs | `attention_efficiency_result/real_qkv_proxy_llama31_l16_8k_g16384_full_cutoffs` |
| `49083537` | 32k RetroInfer-generated layer-16 `X_l` trace | `attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g32768_retro.npz` |
| `49083543` | convert 32k trace to QKV cutoff NPZ | `attention_efficiency_result/real_qkv_llama31_l16_8k_cutoffs_from_g32768_retro.npz` |
| `49083544` | proxy sweep on 32k real QKV cutoffs | `attention_efficiency_result/real_qkv_proxy_llama31_l16_8k_g32768_retro_cutoffs` |

## Notes

- The 16k dense/full trace is the cleanest target if it fits on one A40.
- The 32k trace uses RetroInfer generation because full dense KV for all layers
  at 8k+32k is likely too large for a single A40.
- The saved `X_l` trace is compact: for Llama-8B, `40960 x 4096 x fp16` is about
  `320 MB` before NPZ overhead.
