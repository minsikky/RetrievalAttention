#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import build_page_pq_gpu  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rng = np.random.default_rng(20260514)
    page_size = 32
    pages = 4
    dim = 16
    keys_np = rng.normal(size=(page_size * pages, dim)).astype(np.float32)
    kwargs = dict(
        page_size=page_size,
        subvecs=4,
        subbits=4,
        kmeans_iters=2,
        seed=777,
        key_bytes=2,
        router_enabled=False,
        router_prototypes=1,
        router_merge_rel=0.0,
        router_merge_var=0.0,
        router_max_groups=0,
        device=torch.device("cuda"),
    )
    full = build_page_pq_gpu(keys_np, dynamic_start=0, indexed_end=page_size * pages, **kwargs)
    prefix = build_page_pq_gpu(keys_np, dynamic_start=0, indexed_end=page_size * (pages - 1), **kwargs)
    suffix = build_page_pq_gpu(
        keys_np,
        dynamic_start=page_size * (pages - 1),
        indexed_end=page_size * pages,
        page_id_offset=len(prefix.pages),
        **kwargs,
    )
    appended_pages = [*prefix.pages, *suffix.pages]
    if len(appended_pages) != len(full.pages):
        raise AssertionError(f"page count mismatch: {len(appended_pages)} vs {len(full.pages)}")
    for idx, (lhs, rhs) in enumerate(zip(appended_pages, full.pages, strict=True)):
        if int(lhs.start) != int(rhs.start) or int(lhs.size) != int(rhs.size):
            raise AssertionError(f"page metadata mismatch at {idx}")
        if not torch.equal(lhs.codes, rhs.codes):
            raise AssertionError(f"page codes mismatch at {idx}")
        if not torch.allclose(lhs.codebooks, rhs.codebooks, atol=1e-6, rtol=1e-6):
            diff = torch.max(torch.abs(lhs.codebooks - rhs.codebooks)).item()
            raise AssertionError(f"page codebook mismatch at {idx}: max_diff={diff}")
    print("online page append matches snapshot full build")


if __name__ == "__main__":
    main()
