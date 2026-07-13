#!/usr/bin/env python3
"""Build a QKV trace with N *consecutive* decode query positions.

Issue #24 (task #40) needs a genuine serial-decode trajectory: 8 consecutive
decode positions p-7 .. p ending at an existing golden context. The existing
capture (`real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz`)
subsamples the 131,071 decode positions, so it has no run of 8 consecutive
positions. But the source X-trace
(`real_xtrace_..._sampled_maskstop.npz`) carries the layer inputs for ALL
137,909 positions plus wq/norm/RoPE, so Q at any position is reproducible on
CPU with the SAME arithmetic as scripts/convert_layer_trace_to_qkv_npz.py.

This builder produces a small qkv NPZ with exactly the requested consecutive
positions:
  * keys/values are SLICED verbatim from the existing golden qkv trace
    (byte-identical fp16), truncated to max(position)+1 tokens;
  * for the ANCHOR position (already present in the golden trace at
    `--anchor_qidx`) the Q column is COPIED verbatim from the golden trace, so
    a dump at the anchor reproduces the golden committed sets bit-for-bit;
  * the remaining consecutive positions get Q recomputed from the X-trace
    (rmsnorm -> wq -> NeoX RoPE), fp16, identical to the golden pipeline.

The token stream driving the N steps is the model's own captured decode
trajectory (do_sample greedy/sampled continuation recorded in the X-trace);
nothing is synthetic. Every step k uses keys[:, :position_k+1], so the KV
state grows by exactly one real token per step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    # Matches scripts/convert_layer_trace_to_qkv_npz.py (fp32 accumulation).
    xf = x.astype(np.float32, copy=False)
    var = np.mean(xf * xf, axis=-1, keepdims=True)
    y = xf * (1.0 / np.sqrt(var + float(eps)))
    return y * weight.astype(np.float32, copy=False)


def apply_neox_rope(x: np.ndarray, cos_sin_cache: np.ndarray, positions: np.ndarray) -> np.ndarray:
    # x: [tokens, heads, dim]. cache stores [cos, sin] with dim/2 each.
    dim = int(x.shape[-1])
    half = dim // 2
    cache = cos_sin_cache[positions].astype(np.float32)
    cos = cache[:, :half][:, None, :]
    sin = cache[:, half:half + half][:, None, :]
    xf = x.astype(np.float32, copy=False)
    x1 = xf[..., :half]
    x2 = xf[..., half:]
    return np.concatenate((x1 * cos - x2 * sin, x2 * cos + x1 * sin), axis=-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build consecutive-position QKV trace.")
    p.add_argument("--src_qkv", required=True, help="existing golden qkv NPZ (keys/values/queries source).")
    p.add_argument("--x_trace", required=True, help="layer-input X-trace NPZ (wq/norm/RoPE + layer_inputs).")
    p.add_argument("--output_npz", required=True)
    p.add_argument("--end_position", type=int, required=True,
                   help="absolute position of the LAST (newest) decode step.")
    p.add_argument("--n_positions", type=int, default=8, help="number of consecutive positions.")
    p.add_argument("--anchor_qidx", type=int, default=-1,
                   help="qidx in src_qkv whose position equals --end_position; its Q is copied verbatim. "
                        "-1 disables the verbatim copy (all Q recomputed).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk_tokens", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = np.load(args.src_qkv, allow_pickle=True)
    src_meta = json.loads(str(src["metadata"].item()))

    end = int(args.end_position)
    n_pos = int(args.n_positions)
    positions = list(range(end - n_pos + 1, end + 1))
    assert positions[-1] == end
    if positions[0] < 0:
        raise SystemExit(f"positions start below 0: {positions[0]}")
    max_ctx = end + 1

    src_keys = src["keys"]
    src_values = src["values"]
    total_tokens = int(src_keys.shape[1])
    if max_ctx > total_tokens:
        raise SystemExit(f"end_position {end} exceeds src trace tokens {total_tokens}")

    num_kv_heads = int(src_keys.shape[0])
    num_heads = int(src["queries"].shape[0])
    head_dim = int(src_keys.shape[2])

    # keys/values: verbatim fp16 slice.
    keys = np.ascontiguousarray(src_keys[:, :max_ctx, :])
    values = np.ascontiguousarray(src_values[:, :max_ctx, :])

    # X-trace projection inputs.
    xd = np.load(args.x_trace)
    xmeta = json.loads(str(xd["metadata"].item()))
    norm_eps = float(xmeta["norm_eps"])
    input_len = int(xmeta["input_len"])
    wq = np.asarray(xd["wq"], dtype=np.float32)
    norm_weight = np.asarray(xd["input_layernorm_weight"], dtype=np.float32)
    cos_sin_cache = np.asarray(xd["cos_sin_cache"], dtype=np.float32)
    layer_inputs_np = xd["layer_inputs"]

    queries = np.empty((num_heads, n_pos, head_dim), dtype=np.float16)

    # Recompute Q for every requested position (CPU, golden arithmetic).
    pos_arr = np.asarray(positions, dtype=np.int64)
    for start in range(0, n_pos, int(args.chunk_tokens)):
        stop = min(n_pos, start + int(args.chunk_tokens))
        pos_chunk = pos_arr[start:stop]
        x = np.asarray(layer_inputs_np[pos_chunk], dtype=np.float32)
        x_norm = rmsnorm(x, norm_weight, norm_eps)
        q = (x_norm @ wq.T).reshape(len(pos_chunk), num_heads, head_dim)
        q = apply_neox_rope(q, cos_sin_cache, pos_chunk)
        queries[:, start:stop, :] = np.transpose(q, (1, 0, 2)).astype(np.float16, copy=False)

    # Anchor: copy the golden Q verbatim so the last step reproduces the
    # golden committed sets bit-for-bit. Report the fp16 delta vs recompute.
    anchor_recompute_maxabs = None
    if int(args.anchor_qidx) >= 0:
        aq = int(args.anchor_qidx)
        if int(src["positions"][aq]) != end:
            raise SystemExit(
                f"anchor_qidx {aq} position {int(src['positions'][aq])} != end_position {end}"
            )
        golden_q = np.asarray(src["queries"][:, aq, :], dtype=np.float16)
        recomputed = queries[:, n_pos - 1, :].astype(np.float32)
        anchor_recompute_maxabs = float(np.max(np.abs(recomputed - golden_q.astype(np.float32))))
        queries[:, n_pos - 1, :] = golden_q  # verbatim golden

    out_meta = dict(src_meta)
    out_meta.update(
        source_trace=str(args.x_trace),
        built_from_qkv=str(args.src_qkv),
        selected_query_positions=positions,
        consecutive_positions=True,
        n_consecutive=n_pos,
        anchor_qidx=int(args.anchor_qidx),
        anchor_position=int(end) if int(args.anchor_qidx) >= 0 else None,
        anchor_verbatim_from_golden=int(args.anchor_qidx) >= 0,
        anchor_recompute_fp16_maxabs=anchor_recompute_maxabs,
        keys_values_sliced_from_golden=True,
        operating_point="proxy_mass_m0p9 (OP-0.9, CPU parser, post-#23)",
        builder="scripts/build_consecutive_qkv_trace.py",
    )
    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        keys=keys,
        values=values,
        queries=queries,
        positions=np.asarray(positions, dtype=np.int64),
        metadata=json.dumps(out_meta),
    )
    print(f"[build_consecutive_qkv_trace] wrote {out}")
    print(f"  positions={positions}")
    print(f"  keys={keys.shape} values={values.shape} queries={queries.shape}")
    print(f"  input_len={input_len} max_ctx={max_ctx}")
    if anchor_recompute_maxabs is not None:
        print(f"  anchor fp16 recompute maxabs delta vs golden Q = {anchor_recompute_maxabs}")


if __name__ == "__main__":
    main()
