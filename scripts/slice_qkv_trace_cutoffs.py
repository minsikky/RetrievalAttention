#!/usr/bin/env python3
"""Slice a full real-QKV trace down to repeated decode cutoff query positions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


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
    parser = argparse.ArgumentParser(description="Slice QKV trace to selected decode cutoffs.")
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--decode_lengths", default="500,1000,2000,4000,8000,16000")
    parser.add_argument("--repeat_positions", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input_npz)
    meta = json.loads(str(data["metadata"].item()))
    input_len = int(meta["input_len"])
    all_positions = np.asarray(data["positions"], dtype=np.int64)
    selected_decode_lengths = parse_int_list(args.decode_lengths)

    selected_indices = []
    selected_positions = []
    for decode_len in selected_decode_lengths:
        target_pos = input_len + int(decode_len) - 1
        matches = np.nonzero(all_positions == target_pos)[0]
        if len(matches) == 0:
            print(f"[slice_qkv_trace_cutoffs] skip decode_len={decode_len}: no position {target_pos}")
            continue
        idx = int(matches[0])
        for _ in range(max(1, int(args.repeat_positions))):
            selected_indices.append(idx)
            selected_positions.append(target_pos)
    if not selected_indices:
        raise RuntimeError("no query positions selected")

    queries = np.asarray(data["queries"])[:, selected_indices, :]
    out_meta = dict(meta)
    out_meta.update(
        source_trace=str(args.input_npz),
        selected_decode_lengths=selected_decode_lengths,
        selected_query_positions=selected_positions,
        sliced_from_qkv=True,
    )

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        keys=np.asarray(data["keys"]),
        values=np.asarray(data["values"]),
        queries=queries,
        positions=np.asarray(selected_positions, dtype=np.int64),
        metadata=json.dumps(out_meta),
    )
    print(f"[slice_qkv_trace_cutoffs] wrote {output_path}")
    print(
        f"[slice_qkv_trace_cutoffs] keys={data['keys'].shape} "
        f"values={data['values'].shape} queries={queries.shape}"
    )


if __name__ == "__main__":
    main()
