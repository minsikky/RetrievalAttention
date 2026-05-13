#!/usr/bin/env python3
"""Convert a saved layer-input trace into Q/K/V NPZ slices.

The input is produced by scripts/dump_real_qkv_trace.py with
--save_layer_inputs --skip_qkv. The output is compatible with
benchmark/attention_efficiency_eval.py --source_npz.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch


def parse_int_list(text: str) -> list[int]:
    vals = []
    for part in re.split(r"[,;:\s]+", str(text)):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise ValueError(f"empty int list: {text!r}")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert layer X trace to QKV NPZ.")
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument(
        "--decode_lengths",
        default="0,500,1000,2000,4000,8000,16000",
        help="Decode cutoffs to include as query positions. 0 means last prefill token.",
    )
    parser.add_argument(
        "--repeat_positions",
        type=int,
        default=32,
        help=(
            "Repeat each cutoff position this many times in repeat mode, or select this many "
            "distinct positions per cutoff interval in window mode."
        ),
    )
    parser.add_argument(
        "--query_position_mode",
        choices=("repeat", "window"),
        default="repeat",
        help=(
            "repeat preserves the old behavior by duplicating each cutoff query. "
            "window selects distinct positions uniformly from the interval after the previous cutoff."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk_tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--include_graph_prefill_queries",
        action="store_true",
        help="Also save Q vectors as graph_queries for Roar-style Q-K graph construction.",
    )
    parser.add_argument(
        "--graph_query_scope",
        choices=("prefill", "all"),
        default="prefill",
        help="Which layer positions to save as graph_queries when --include_graph_prefill_queries is set.",
    )
    parser.add_argument(
        "--graph_query_stride",
        type=int,
        default=1,
        help="Keep every Nth prefill query position when --include_graph_prefill_queries is set.",
    )
    return parser.parse_args()


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + float(eps))
    return y * weight.float()


def apply_neox_rope(x: torch.Tensor, cos_sin_cache: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    # x: [tokens, heads, dim]. cache stores [cos, sin] with dim/2 each.
    dim = int(x.shape[-1])
    half = dim // 2
    cache = cos_sin_cache.index_select(0, positions.to(cos_sin_cache.device)).float()
    cos = cache[:, :half].unsqueeze(1)
    sin = cache[:, half:half + half].unsqueeze(1)
    x = x.float()
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def main() -> None:
    args = parse_args()
    data = np.load(args.input_npz)
    meta = json.loads(str(data["metadata"].item()))
    layer_inputs_np = data["layer_inputs"]
    input_len = int(meta["input_len"])
    total_tokens = int(layer_inputs_np.shape[0])

    requested_decode = parse_int_list(args.decode_lengths)
    query_positions = []
    decode_for_positions = []
    prev_decode = 0
    for decode_len in requested_decode:
        decode_len = int(decode_len)
        if decode_len <= 0:
            pos = input_len - 1
            if pos >= total_tokens:
                print(
                    f"[convert_layer_trace_to_qkv_npz] skip decode_len={decode_len}: "
                    f"position {pos} >= trace tokens {total_tokens}",
                    flush=True,
                )
                continue
            query_positions.extend([pos] * max(1, int(args.repeat_positions)))
            decode_for_positions.extend([decode_len] * max(1, int(args.repeat_positions)))
            prev_decode = max(prev_decode, decode_len)
            continue

        hi = input_len + decode_len - 1
        if hi >= total_tokens:
            print(
                f"[convert_layer_trace_to_qkv_npz] skip decode_len={decode_len}: "
                f"position {hi} >= trace tokens {total_tokens}",
                flush=True,
            )
            continue
        if args.query_position_mode == "repeat":
            selected = [hi] * max(1, int(args.repeat_positions))
        else:
            lo_decode = max(1, int(prev_decode) + 1)
            hi_decode = int(decode_len)
            count = min(max(1, int(args.repeat_positions)), max(1, hi_decode - lo_decode + 1))
            decode_points = np.linspace(lo_decode, hi_decode, num=count, dtype=np.int64)
            selected = [input_len + int(x) - 1 for x in decode_points.tolist()]
        query_positions.extend(selected)
        decode_for_positions.extend([decode_len] * len(selected))
        prev_decode = max(prev_decode, decode_len)
    if not query_positions:
        raise RuntimeError("no query positions selected")

    device = torch.device(args.device)
    out_dtype = np.float16 if args.dtype == "fp16" else np.float32
    hidden_size = int(meta["hidden_size"])
    num_heads = int(meta["num_heads"])
    num_kv_heads = int(meta["num_key_value_heads"])
    head_dim = int(meta["head_dim"])
    norm_eps = float(meta["norm_eps"])

    wq = torch.as_tensor(data["wq"], device=device).float()
    wk = torch.as_tensor(data["wk"], device=device).float()
    wv = torch.as_tensor(data["wv"], device=device).float()
    norm_weight = torch.as_tensor(data["input_layernorm_weight"], device=device)
    cos_sin_cache = torch.as_tensor(data["cos_sin_cache"], device=device)

    keys = np.empty((num_kv_heads, total_tokens, head_dim), dtype=out_dtype)
    values = np.empty((num_kv_heads, total_tokens, head_dim), dtype=out_dtype)
    query_pos_tensor = torch.as_tensor(query_positions, dtype=torch.long, device=device)
    queries = np.empty((num_heads, len(query_positions), head_dim), dtype=out_dtype)
    graph_query_positions = None
    graph_queries = None
    if args.include_graph_prefill_queries:
        stride = max(1, int(args.graph_query_stride))
        graph_stop = input_len if args.graph_query_scope == "prefill" else total_tokens
        graph_query_positions = np.arange(0, graph_stop, stride, dtype=np.int64)
        graph_queries = np.empty((num_heads, len(graph_query_positions), head_dim), dtype=out_dtype)

    # Build all K/V once for the full trace.
    chunk = max(1, int(args.chunk_tokens))
    for start in range(0, total_tokens, chunk):
        end = min(total_tokens, start + chunk)
        x = torch.as_tensor(layer_inputs_np[start:end], device=device)
        x_norm = rmsnorm(x, norm_weight, norm_eps)
        k = torch.matmul(x_norm, wk.t()).view(end - start, num_kv_heads, head_dim)
        v = torch.matmul(x_norm, wv.t()).view(end - start, num_kv_heads, head_dim)
        pos = torch.arange(start, end, dtype=torch.long, device=device)
        k = apply_neox_rope(k, cos_sin_cache, pos)
        keys[:, start:end, :] = k.permute(1, 0, 2).detach().cpu().numpy().astype(out_dtype, copy=False)
        values[:, start:end, :] = v.float().permute(1, 0, 2).detach().cpu().numpy().astype(out_dtype, copy=False)

    # Build Q only for selected cutoff positions.
    for start in range(0, len(query_positions), chunk):
        end = min(len(query_positions), start + chunk)
        pos = query_pos_tensor[start:end]
        x = torch.as_tensor(layer_inputs_np[np.asarray(query_positions[start:end])], device=device)
        x_norm = rmsnorm(x, norm_weight, norm_eps)
        q = torch.matmul(x_norm, wq.t()).view(end - start, num_heads, head_dim)
        q = apply_neox_rope(q, cos_sin_cache, pos)
        queries[:, start:end, :] = q.permute(1, 0, 2).detach().cpu().numpy().astype(out_dtype, copy=False)

    if graph_queries is not None and graph_query_positions is not None:
        graph_pos_tensor = torch.as_tensor(graph_query_positions, dtype=torch.long, device=device)
        for start in range(0, len(graph_query_positions), chunk):
            end = min(len(graph_query_positions), start + chunk)
            pos = graph_pos_tensor[start:end]
            x = torch.as_tensor(layer_inputs_np[np.asarray(graph_query_positions[start:end])], device=device)
            x_norm = rmsnorm(x, norm_weight, norm_eps)
            q = torch.matmul(x_norm, wq.t()).view(end - start, num_heads, head_dim)
            q = apply_neox_rope(q, cos_sin_cache, pos)
            graph_queries[:, start:end, :] = (
                q.permute(1, 0, 2).detach().cpu().numpy().astype(out_dtype, copy=False)
            )

    out_meta = dict(meta)
    out_meta.update(
        source_trace=str(args.input_npz),
        selected_decode_lengths=requested_decode,
        selected_query_positions=query_positions,
        selected_query_decode_cutoffs=decode_for_positions,
        query_position_mode=args.query_position_mode,
        qkv_dtype=args.dtype,
    )
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "keys": keys,
        "values": values,
        "queries": queries,
        "positions": np.asarray(query_positions, dtype=np.int64),
        "metadata": json.dumps(out_meta),
    }
    if graph_queries is not None and graph_query_positions is not None:
        payload["graph_queries"] = graph_queries
        payload["graph_positions"] = graph_query_positions
    np.savez(output_path, **payload)
    print(f"[convert_layer_trace_to_qkv_npz] wrote {output_path}")
    print(f"[convert_layer_trace_to_qkv_npz] keys={keys.shape} values={values.shape} queries={queries.shape}")
    if graph_queries is not None and graph_query_positions is not None:
        print(
            f"[convert_layer_trace_to_qkv_npz] "
            f"graph_queries={graph_queries.shape} graph_positions={graph_query_positions.shape}"
        )


if __name__ == "__main__":
    main()
