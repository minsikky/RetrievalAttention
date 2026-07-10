#!/usr/bin/env python3
from __future__ import annotations

import contextlib
from pathlib import Path
import sys

import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.ruler.pred.dense_kv_offload import (
    DenseKVOffloadCache,
    patched_qwen2_dense_kv_offload,
)


@torch.inference_mode()
def _chunked_greedy(
    model: Qwen2ForCausalLM,
    input_ids: torch.Tensor,
    *,
    chunk_tokens: int,
    decode_steps: int,
    offload: bool,
) -> tuple[list[int], list[torch.Tensor], DenseKVOffloadCache | None]:
    cache = (
        DenseKVOffloadCache(
            num_layers=model.config.num_hidden_layers,
            max_cache_len=int(input_ids.shape[1]) + decode_steps,
            kv_block_tokens=5,
            decode_kv_block_tokens=17,
            staging_buffers=2,
            query_block_tokens=3,
            device="cpu",
        )
        if offload
        else None
    )
    context = patched_qwen2_dense_kv_offload(model) if offload else contextlib.nullcontext()
    past_key_values = cache
    out = None
    with context:
        for start in range(0, int(input_ids.shape[1]), chunk_tokens):
            stop = min(int(input_ids.shape[1]), start + chunk_tokens)
            kwargs = {"use_cache": True, "logits_to_keep": 1}
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            out = model(input_ids[:, start:stop], **kwargs)
            past_key_values = out.past_key_values
        if out is None:
            raise RuntimeError("empty test prompt")

        tokens: list[int] = []
        logits: list[torch.Tensor] = []
        for _ in range(decode_steps):
            step_logits = out.logits[:, -1, :].float()
            logits.append(step_logits.cpu())
            next_token = torch.argmax(step_logits, dim=-1, keepdim=True)
            tokens.append(int(next_token.item()))
            out = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = out.past_key_values
    return tokens, logits, cache


def main() -> None:
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
    )
    config.sliding_window = None
    config._attn_implementation = "sdpa"
    stock = Qwen2ForCausalLM(config).eval().to(dtype=torch.bfloat16)
    offload = Qwen2ForCausalLM(config).eval().to(dtype=torch.bfloat16)
    offload.load_state_dict(stock.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (1, 23), dtype=torch.long)
    stock_tokens, stock_logits, _ = _chunked_greedy(
        stock,
        input_ids,
        chunk_tokens=7,
        decode_steps=16,
        offload=False,
    )
    offload_tokens, offload_logits, cache = _chunked_greedy(
        offload,
        input_ids,
        chunk_tokens=7,
        decode_steps=16,
        offload=True,
    )

    if cache is None:
        raise AssertionError("offload run did not return its cache")
    for layer_idx, layer in enumerate(cache._layers):
        if layer is None:
            raise AssertionError(f"layer {layer_idx} was not cached")
        if layer.key.dtype != torch.bfloat16 or layer.value.dtype != torch.bfloat16:
            raise AssertionError(f"layer {layer_idx} KV dtype changed")
    if stock_tokens != offload_tokens:
        raise AssertionError(
            f"greedy token mismatch\nstock={stock_tokens}\noffload={offload_tokens}"
        )

    deltas = [
        float((stock_step - offload_step).abs().max().item())
        for stock_step, offload_step in zip(stock_logits, offload_logits)
    ]
    print(f"torch={torch.__version__} transformers={transformers.__version__}")
    print(f"greedy_tokens={stock_tokens}")
    print("max_abs_logit_diff_by_step=" + ",".join(f"{delta:.8f}" for delta in deltas))
    print(f"max_abs_logit_diff={max(deltas):.8f}")
    print("PASS: 16/16 greedy token IDs match; CPU KV storage remained bf16")


if __name__ == "__main__":
    main()
