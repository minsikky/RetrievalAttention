#!/usr/bin/env python3
"""Dump real-model Q/K/V traces for offline attention-efficiency experiments.

This runs a normal Full_Flash_Attn prefill+decode once, captures post-RoPE
Q/K/V for one layer, and saves an NPZ compatible with
benchmark/attention_efficiency_eval.py --source_npz.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from model_hub import LlamaModel, QwenModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump real QKV decode trace as NPZ.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data_path", default="attention_efficiency_result/gpu_smoke_8192.json")
    parser.add_argument("--output_npz", default="attention_efficiency_result/real_qkv_trace_layer16.npz")
    parser.add_argument("--layer_idx", type=int, default=16)
    parser.add_argument("--gen_len", type=int, default=128)
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument(
        "--attention_type",
        choices=("Full_Flash_Attn", "RetroInfer", "RetrievalAttention"),
        default="Full_Flash_Attn",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--include_prefill_queries", action="store_true")
    parser.add_argument(
        "--mask_stop_tokens",
        action="store_true",
        help="Mask tokenizer EOS and common chat EOT tokens during generation so long traces do not collapse into terminators.",
    )
    parser.add_argument(
        "--save_layer_inputs",
        action="store_true",
        help="Save layer input hidden states X_l for prefill and decoded tokens.",
    )
    parser.add_argument(
        "--skip_qkv",
        action="store_true",
        help="Only save layer inputs/weights, not the full K/V cache and captured queries.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))


def load_prompt(path: Path, prompt_index: int) -> str:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"expected non-empty list in {path}")
    item = data[int(prompt_index)]
    if not isinstance(item, dict) or "input" not in item:
        raise ValueError(f"expected item with 'input' field in {path}")
    return str(item["input"])


def load_model(model_name: str, max_len: int, dtype: torch.dtype, device: str):
    if "Llama" in model_name:
        return LlamaModel(model_name, max_length=max_len, dtype=dtype, device_map=device)
    if "Qwen" in model_name:
        return QwenModel(model_name, max_length=max_len, dtype=dtype, device_map=device)
    raise ValueError(f"unsupported model: {model_name}")


def generate_attn_config(model_name: str, context_len: int, attention_type: str) -> dict:
    if attention_type == "Full_Flash_Attn":
        return {}
    config_file = PROJECT_ROOT / "config" / f"{model_name.split('/')[-1]}.json"
    config = json.loads(config_file.read_text())
    if attention_type == "RetroInfer":
        retro_core_override = os.environ.get("RETROINFER_CORE", "").strip()
        if retro_core_override:
            config[attention_type]["core"] = max(int(retro_core_override), 1)
        n_clusters = max(int(context_len / 16), 1)
        n_segments = max(int(context_len / 8192), 1)
        lower = (n_clusters // (n_segments * 32)) * (n_segments * 32)
        upper = lower + (n_segments * 32)
        n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper
        nprobe = max(int(n_clusters * 0.018), 1)
        config[attention_type]["n_centroids"] = n_clusters
        config[attention_type]["n_segment"] = n_segments
        config[attention_type]["nprobe"] = nprobe
        config[attention_type]["cache_cluster_num"] = nprobe * 3
        config[attention_type]["max_compute_cluster_num"] = max(int(n_clusters / 4), nprobe)
    if attention_type == "RetrievalAttention" and attention_type not in config:
        config[attention_type] = {
            "static_pattern_start": 128,
            "static_pattern_end": 512,
            "q_knn": 8,
            "key_degree": 8,
            "token_budget": int(os.environ.get("TOKEN_BUDGET_OVERRIDE", "100")),
        }
    return config


def install_trace_hooks(llm, layer_idx: int, include_prefill_queries: bool, save_layer_inputs: bool) -> dict:
    trace = {
        "decode_queries": [],
        "decode_positions": [],
        "prefill_queries": None,
        "prefill_layer_input": None,
        "decode_layer_inputs": [],
    }
    original_layer_prefill = llm.layer_prefill
    original_layer_decode = llm.layer_decode

    def traced_layer_prefill(self, current_layer_idx, start_bdx, hidden_states):
        if int(current_layer_idx) != int(layer_idx):
            return original_layer_prefill(current_layer_idx, start_bdx, hidden_states)

        bsz, seq_len, dim = hidden_states.shape
        if bsz != 1:
            raise ValueError("trace dumping currently expects batch_size=1")
        layer = self.layers[current_layer_idx]

        if save_layer_inputs:
            trace["prefill_layer_input"] = hidden_states[0].detach().to(torch.float16).cpu()

        temp_hidden_states = hidden_states.clone()
        temp_hidden_states = self.layernorm(
            temp_hidden_states,
            layer.input_layernorm_variance_epsilon,
            layer.input_layernorm_weight,
        )
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        if include_prefill_queries:
            trace["prefill_queries"] = query_states[0].detach().float().cpu()

        key_states, value_states = self.kv_cache.prefill_update_kv_cache(
            query_states,
            key_states,
            value_states,
            current_layer_idx,
            start_bdx,
        )
        temp_attn_out = self.prefill_attention(query_states, key_states, value_states, current_layer_idx)
        self.kv_cache.sync(current_layer_idx, start_bdx)
        del query_states, key_states, value_states

        hidden_states += self.wo(temp_attn_out, layer, bsz, seq_len, dim)
        del temp_attn_out

        residual = hidden_states.clone()
        hidden_states = self.layernorm(
            hidden_states,
            layer.post_attention_layernorm_variance_epsilon,
            layer.post_attention_layernorm_weight,
        )
        for batch_idx in range(0, bsz, 1):
            for start_idx in range(0, seq_len, 65536):
                end_idx = min(seq_len, start_idx + 65536)
                hidden_states[batch_idx:batch_idx + 1, start_idx:end_idx, :] = self.mlp(
                    hidden_states[batch_idx:batch_idx + 1, start_idx:end_idx, :],
                    layer,
                )
        hidden_states += residual
        del residual
        return hidden_states

    def traced_layer_decode(self, current_layer_idx, hidden_states):
        if int(current_layer_idx) != int(layer_idx):
            return original_layer_decode(current_layer_idx, hidden_states)

        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        if bsz != 1 or seq_len != 1:
            raise ValueError("trace dumping currently expects batch_size=1 and decode seq_len=1")
        layer = self.layers[current_layer_idx]

        if save_layer_inputs:
            trace["decode_layer_inputs"].append(hidden_states[0, 0].detach().to(torch.float16).cpu())

        hidden_states = self.layernorm(
            hidden_states,
            layer.input_layernorm_variance_epsilon,
            layer.input_layernorm_weight,
        )
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        trace["decode_queries"].append(query_states[0, 0].detach().float().cpu())
        trace["decode_positions"].append(int(self.kv_cache.context))

        key_states, value_states = self.kv_cache.decode_update_kv_cache(
            key_states,
            value_states,
            current_layer_idx,
        )
        attn_out = self.decode_attention(query_states, key_states, value_states, current_layer_idx)
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm(
            hidden_states,
            layer.post_attention_layernorm_variance_epsilon,
            layer.post_attention_layernorm_weight,
        )
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states
        return hidden_states

    llm.layer_prefill = types.MethodType(traced_layer_prefill, llm)
    llm.layer_decode = types.MethodType(traced_layer_decode, llm)
    return trace


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompt = load_prompt(PROJECT_ROOT / args.data_path, args.prompt_index)
    encoded = tokenizer([prompt], return_tensors="pt", padding=True)
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    if args.max_input_tokens > 0 and input_ids.shape[1] > args.max_input_tokens:
        input_ids = input_ids[:, -int(args.max_input_tokens):]
        attention_mask = attention_mask[:, -int(args.max_input_tokens):]
    input_len = int(input_ids.shape[1])
    max_len = input_len + int(args.gen_len)

    print(f"[dump_real_qkv_trace] model={args.model_name}")
    print(f"[dump_real_qkv_trace] input_len={input_len} gen_len={args.gen_len} layer={args.layer_idx}")
    llm = load_model(args.model_name, max_len=max_len, dtype=dtype, device=args.device)
    stop_token_ids = []
    if args.mask_stop_tokens:
        candidates = []
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            if isinstance(eos_id, (list, tuple)):
                candidates.extend(int(x) for x in eos_id if x is not None)
            else:
                candidates.append(int(eos_id))
        for token in ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>"):
            tok_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(tok_id, int) and tok_id >= 0 and tok_id != tokenizer.unk_token_id:
                candidates.append(int(tok_id))
        stop_token_ids = sorted(set(candidates))
        llm.forbidden_token_ids = stop_token_ids
        print(f"[dump_real_qkv_trace] masked stop tokens={stop_token_ids}")
    trace = install_trace_hooks(llm, args.layer_idx, args.include_prefill_queries, args.save_layer_inputs)

    input_ids = input_ids.to(args.device)
    attention_mask = attention_mask.to(args.device)
    attn_config = generate_attn_config(args.model_name, input_len, args.attention_type)
    outputs = llm.generate(
        attention_type=args.attention_type,
        inputs_ids=input_ids,
        attention_masks=attention_mask,
        max_new_length=int(args.gen_len),
        attn_config=attn_config,
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        ignore_eos=True,
        prefill_bsz=1,
        prefill_method="full",
    )
    del outputs
    torch.cuda.synchronize()

    if hasattr(llm.kv_cache, "valid_length") and llm.kv_cache.valid_length is not None:
        valid_len = int(llm.kv_cache.valid_length.detach().cpu().max().item())
    else:
        valid_len = int(getattr(llm.kv_cache, "context", input_len + max(0, int(args.gen_len) - 1)))
    output_path = PROJECT_ROOT / args.output_npz
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer = llm.layers[int(args.layer_idx)]
    meta = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "layer_idx": int(args.layer_idx),
        "input_len": input_len,
        "gen_len": int(args.gen_len),
        "attention_type": args.attention_type,
        "valid_len": valid_len,
        "captured_decode_inputs": len(trace["decode_layer_inputs"]),
        "dtype": args.dtype,
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "mask_stop_tokens": bool(args.mask_stop_tokens),
        "masked_stop_token_ids": stop_token_ids,
        "num_heads": int(llm.num_heads),
        "num_key_value_heads": int(llm.num_key_value_heads),
        "head_dim": int(llm.head_dim),
        "hidden_size": int(llm.hidden_size),
        "norm_eps": float(layer.input_layernorm_variance_epsilon),
    }
    payload = {
        "metadata": json.dumps(meta),
        "wq": layer.wqkv[: llm.hidden_size].detach().to(torch.float16).cpu().numpy(),
        "wk": layer.wqkv[
            llm.hidden_size : llm.hidden_size + llm.hidden_size // llm.num_key_value_groups
        ].detach().to(torch.float16).cpu().numpy(),
        "wv": layer.wqkv[
            llm.hidden_size + llm.hidden_size // llm.num_key_value_groups :
        ].detach().to(torch.float16).cpu().numpy(),
        "input_layernorm_weight": layer.input_layernorm_weight.detach().to(torch.float16).cpu().numpy(),
        "cos_sin_cache": llm.cos_sin_cache[:valid_len].detach().to(torch.float16).cpu().numpy(),
    }
    if args.save_layer_inputs:
        if trace["prefill_layer_input"] is None:
            raise RuntimeError("save_layer_inputs requested but prefill layer input was not captured")
        if trace["decode_layer_inputs"]:
            decode_inputs = torch.stack(trace["decode_layer_inputs"], dim=0)
            layer_inputs = torch.cat((trace["prefill_layer_input"], decode_inputs), dim=0)
        else:
            layer_inputs = trace["prefill_layer_input"]
        payload["layer_inputs"] = layer_inputs.contiguous().numpy()
        payload["layer_input_positions"] = np.arange(int(layer_inputs.shape[0]), dtype=np.int64)

    if not args.skip_qkv:
        key_cache = llm.kv_cache.key_cache[int(args.layer_idx)][0, :valid_len].detach().float().cpu()
        value_cache = llm.kv_cache.value_cache[int(args.layer_idx)][0, :valid_len].detach().float().cpu()
        keys = key_cache.permute(1, 0, 2).contiguous().numpy()
        values = value_cache.permute(1, 0, 2).contiguous().numpy()

        if trace["decode_queries"]:
            queries_t = torch.stack(trace["decode_queries"], dim=0)
            positions = np.asarray(trace["decode_positions"], dtype=np.int64)
        elif trace["prefill_queries"] is not None:
            queries_t = trace["prefill_queries"]
            positions = np.arange(int(queries_t.shape[0]), dtype=np.int64)
        else:
            raise RuntimeError("no queries captured; use gen_len >= 2 or --include_prefill_queries")
        queries = queries_t.permute(1, 0, 2).contiguous().numpy()
        payload.update(keys=keys, values=values, queries=queries, positions=positions)

    np.savez(output_path, **payload)
    print(f"[dump_real_qkv_trace] wrote {output_path}")
    if args.save_layer_inputs:
        print(f"[dump_real_qkv_trace] layer_inputs={payload['layer_inputs'].shape}")
    if not args.skip_qkv:
        print(f"[dump_real_qkv_trace] keys={keys.shape} values={values.shape} queries={queries.shape}")


if __name__ == "__main__":
    main()
