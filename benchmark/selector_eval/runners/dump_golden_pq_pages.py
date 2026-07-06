"""Dump PQ page blocks (codebooks + codes) for the S1/S2 golden vectors.

Issue #5 item (a): the K/V trace is a large gitignored artifact, so the RTL
side cannot rebuild pages locally. This dumps the LAST sealed page per
(context, kv_head) for the same 12 rows as the golden_q*_h*.npz dumps —
distinct pages across contexts, covering all 4 dumped kv-heads — small
enough to track in-repo next to the goldens.

Each page npz self-checks before writing: page logits recomputed from the
dumped codebooks+codes via the reference pq_page_scores must match the
golden ranked_scores_raw_fp32 bit-for-bit at that page's token range.

Usage (defaults match the s2_s3_20260706 goldens):
  python benchmark/selector_eval/runners/dump_golden_pq_pages.py \
    --golden_dir benchmark/selector_eval/golden_vectors/s2_s3_20260706
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.data.trace import load_trace  # noqa: E402
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (  # noqa: E402
    build_page_pq_gpu,
    pq_page_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qkv_trace",
        default="attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz",
    )
    parser.add_argument(
        "--golden_dir",
        default="benchmark/selector_eval/golden_vectors/s2_s3_20260706",
    )
    parser.add_argument("--qidx_heads", default="159:0,159:8,159:16,159:24,223:0,223:8,223:16,223:24,287:0,287:8,287:16,287:24")
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=8)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    args = parser.parse_args()

    trace = load_trace(args.qkv_trace)
    golden_dir = Path(args.golden_dir)
    pairs = [tuple(int(v) for v in tok.split(":")) for tok in str(args.qidx_heads).split(",")]

    built: dict[tuple[int, int], object] = {}
    for qidx, head in pairs:
        golden = np.load(golden_dir / f"golden_q{qidx}_h{head}.npz")
        kv_head = int(trace.kv_head_for(head))
        assert kv_head == int(golden["kv_head"]), (kv_head, int(golden["kv_head"]))
        context_len = int(golden["context_len"])
        dynamic_start = int(golden["dynamic_start"])
        sealed_end = int(golden["sealed_end"])
        indexed_end = max(
            min(max(0, int(args.static_prefix)), int(trace.input_len)),
            min(context_len - max(0, int(args.static_suffix)), trace.keys.shape[1]),
        )
        key = (kv_head, context_len)
        if key not in built:
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            built[key] = build_page_pq_gpu(
                keys_np,
                dynamic_start=dynamic_start,
                indexed_end=indexed_end,
                page_size=int(args.page_size),
                subvecs=int(args.subvecs),
                subbits=int(args.subbits),
                kmeans_iters=int(args.kmeans_iters),
                seed=2025 + 2027 * int(kv_head),
                key_bytes=2,
                router_enabled=False,
                router_prototypes=0,
                router_merge_rel=0.0,
                router_merge_var=0.0,
                router_max_groups=0,
                device=torch.device("cpu"),
            )
        index = built[key]
        page = index.pages[-1]
        assert int(page.start) + int(page.size) == sealed_end, (page.start, page.size, sealed_end)

        # Self-check: reference logits from the dumped blocks must equal the
        # golden ranked scores bit-for-bit on this page's token range.
        # torch-bundled-numpy interop in this venv rejects from_numpy /
        # .numpy(); round-trip through Python lists (bit-exact for fp32).
        query = torch.tensor(golden["query_fp32"].tolist(), dtype=torch.float32)
        tokens_t, scores_t = pq_page_scores(query, page)
        tokens = np.fromiter((int(x) for x in tokens_t.tolist()), dtype=np.int64, count=int(tokens_t.numel()))
        scores = np.fromiter((float(x) for x in scores_t.tolist()), dtype=np.float32, count=int(scores_t.numel()))
        g_idx = np.asarray(golden["ranked_idx"], dtype=np.int64)
        g_scores = np.asarray(golden["ranked_scores_raw_fp32"], dtype=np.float32)
        golden_by_token = np.empty(context_len, dtype=np.float32)
        golden_by_token[g_idx] = g_scores
        if not np.array_equal(scores, golden_by_token[tokens]):
            raise AssertionError(f"self-check FAILED for q{qidx} h{head} page@{page.start}")

        out = golden_dir / f"page_ctx{context_len}_kv{kv_head}.npz"
        np.savez_compressed(
            out,
            context_len=context_len,
            kv_head=kv_head,
            page_start=int(page.start),
            page_size=int(page.size),
            subvecs=int(args.subvecs),
            subbits=int(args.subbits),
            kmeans_iters=int(args.kmeans_iters),
            seed=2025 + 2027 * int(kv_head),
            codebooks_fp32=np.asarray(page.codebooks.tolist(), dtype=np.float32),
            codes_u8=np.asarray(page.codes.tolist(), dtype=np.uint8),
        )
        print(f"OK q{qidx} h{head} -> {out.name} (page@{int(page.start)}, self-check bit-exact)")


if __name__ == "__main__":
    main()
