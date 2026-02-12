import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Keep K_TOP constexpr-friendly for Triton static loops.
_SUPPORTED_K_TOP = (8, 16, 32, 64, 128)
_MAX_SORT_BLOCK_K = 1024
_MAX_EFFECTIVE_BLOCK_Q = 64
_MAX_EFFECTIVE_BLOCK_K = 256


def _resolve_kernel_topk(k_top: int) -> int:
    if k_top <= 0:
        raise ValueError(f"k_top must be positive, got {k_top}")
    for candidate in _SUPPORTED_K_TOP:
        if k_top <= candidate:
            return candidate
    raise ValueError(
        f"k_top={k_top} is larger than max supported {_SUPPORTED_K_TOP[-1]} for this prototype"
    )


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(x - 1).bit_length())


def _resolve_block_k(block_k: int, k_top_kernel: int) -> int:
    """
    Resolve a valid BLOCK_K for Triton sort-based selection.
    Never raises on large user input; clamps to hardware/kernel-safe bounds.
    """
    block_k_kernel = _next_power_of_two(block_k)
    if block_k_kernel < k_top_kernel:
        block_k_kernel = k_top_kernel
    if block_k_kernel > _MAX_SORT_BLOCK_K:
        block_k_kernel = _MAX_SORT_BLOCK_K
    return block_k_kernel


@triton.jit
def _float_to_ordered_i32(x):
    """
    Order-preserving float32 -> int32 transform for signed integer comparisons.
    """
    bits = x.to(tl.int32, bitcast=True)
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


@triton.jit
def _ordered_i32_to_float(x):
    bits = tl.where(x >= 0, x, x ^ 0x7FFFFFFF)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _fused_qk_running_topk_kernel(
    q_ptr,
    k_ptr,
    out_scores_ptr,
    out_indices_ptr,
    num_q,
    num_k,
    head_dim,
    stride_qn,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_on,
    stride_ok,
    stride_in,
    stride_ik,
    K_TOP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert((BLOCK_K & (BLOCK_K - 1)) == 0, "BLOCK_K must be power-of-two for tl.sort")
    tl.static_assert((K_TOP & (K_TOP - 1)) == 0, "K_TOP must be power-of-two")

    pid_q = tl.program_id(0)
    q_start = pid_q * BLOCK_Q
    q_idx = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_idx < num_q

    top_packed = (tl.full((BLOCK_Q, K_TOP), -2147483648, tl.int64) << 32)
    local_cols = tl.arange(0, BLOCK_K)

    for k_start in tl.range(0, num_k, BLOCK_K):
        k_idx = k_start + local_cols
        k_mask = k_idx < num_k

        scores = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        for d_start in tl.range(0, head_dim, BLOCK_D):
            d_idx = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_idx < head_dim

            q_ptrs = q_ptr + q_idx[:, None] * stride_qn + d_idx[None, :] * stride_qd
            k_ptrs = k_ptr + d_idx[:, None] * stride_kd + k_idx[None, :] * stride_kn

            q_tile = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
            k_tile = tl.load(k_ptrs, mask=d_mask[:, None] & k_mask[None, :], other=0.0)
            scores += tl.dot(q_tile, k_tile)

        # Convert scores to order-preserving int32 and pack with absolute key index.
        ordered = _float_to_ordered_i32(scores)
        ordered = tl.where(k_mask[None, :], ordered, -2147483648)
        idx_i64 = k_idx[None, :].to(tl.int64) & 0xFFFFFFFF
        packed = (ordered.to(tl.int64) << 32) | idx_i64

        # Block-local top-k via bitonic sort (power-of-two BLOCK_K).
        packed = tl.sort(packed, descending=1)

        # Keep only first K_TOP entries from BLOCK_K using repeated halving.
        block_top = packed
        if BLOCK_K > K_TOP:
            tmp = tl.reshape(block_top, (BLOCK_Q, 2, BLOCK_K // 2))
            tmp = tl.permute(tmp, (0, 2, 1))
            block_top, _ = tl.split(tmp)
        if BLOCK_K > 2 * K_TOP:
            tmp = tl.reshape(block_top, (BLOCK_Q, 2, BLOCK_K // 4))
            tmp = tl.permute(tmp, (0, 2, 1))
            block_top, _ = tl.split(tmp)
        if BLOCK_K > 4 * K_TOP:
            tmp = tl.reshape(block_top, (BLOCK_Q, 2, BLOCK_K // 8))
            tmp = tl.permute(tmp, (0, 2, 1))
            block_top, _ = tl.split(tmp)
        if BLOCK_K > 8 * K_TOP:
            tmp = tl.reshape(block_top, (BLOCK_Q, 2, BLOCK_K // 16))
            tmp = tl.permute(tmp, (0, 2, 1))
            block_top, _ = tl.split(tmp)
        if BLOCK_K > 16 * K_TOP:
            tmp = tl.reshape(block_top, (BLOCK_Q, 2, BLOCK_K // 32))
            tmp = tl.permute(tmp, (0, 2, 1))
            block_top, _ = tl.split(tmp)
        if BLOCK_K > 32 * K_TOP:
            tmp = tl.reshape(block_top, (BLOCK_Q, 2, BLOCK_K // 64))
            tmp = tl.permute(tmp, (0, 2, 1))
            block_top, _ = tl.split(tmp)
        if BLOCK_K > 64 * K_TOP:
            tmp = tl.reshape(block_top, (BLOCK_Q, 2, BLOCK_K // 128))
            tmp = tl.permute(tmp, (0, 2, 1))
            block_top, _ = tl.split(tmp)

        # Merge running top-k with block top-k by sorting 2*K_TOP entries.
        # Triton 3.1 cat only supports rank-1, so use join+reshape.
        merged = tl.join(top_packed, block_top)
        merged = tl.reshape(merged, (BLOCK_Q, 2 * K_TOP), can_reorder=True)
        merged = tl.sort(merged, descending=1)
        tmp = tl.reshape(merged, (BLOCK_Q, 2, K_TOP))
        tmp = tl.permute(tmp, (0, 2, 1))
        top_packed, _ = tl.split(tmp)

    out_k = tl.arange(0, K_TOP)
    out_ordered = (top_packed >> 32).to(tl.int32)
    out_scores = _ordered_i32_to_float(out_ordered)
    out_indices = (top_packed & 0xFFFFFFFF).to(tl.int32)
    out_scores_ptrs = out_scores_ptr + q_idx[:, None] * stride_on + out_k[None, :] * stride_ok
    out_indices_ptrs = out_indices_ptr + q_idx[:, None] * stride_in + out_k[None, :] * stride_ik
    tl.store(out_scores_ptrs, out_scores, mask=q_mask[:, None])
    tl.store(out_indices_ptrs, out_indices, mask=q_mask[:, None])


def fused_qk_topk_triton(
    queries: torch.Tensor,
    keys: torch.Tensor,
    k_top: int,
    *,
    normalize: bool = True,
    block_q: int = 64,
    block_k: int = 128,
    block_d: int = 32,
    launch_q_chunk: int = 4096,
    verbose: bool = False,
    return_scores: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standalone prototype: fused QK matmul + running top-k in one Triton kernel.
    Returns:
      scores: [num_q, k_top] float32 (sorted descending)
      indices: [num_q, k_top] int32

    Notes:
      - This module is intentionally not integrated into RetrievalAttention code paths.
      - For Tensor-Core style matmul, inputs are cast to fp16 if needed.
      - k_top is capped by _SUPPORTED_K_TOP for this prototype.
    """
    if queries.ndim != 2 or keys.ndim != 2:
        raise ValueError(
            f"queries and keys must be 2D tensors, got {queries.ndim}D and {keys.ndim}D"
        )
    if queries.shape[1] != keys.shape[1]:
        raise ValueError(
            f"head_dim mismatch: queries={queries.shape[1]} keys={keys.shape[1]}"
        )
    if queries.device.type != "cuda" or keys.device.type != "cuda":
        raise ValueError("queries and keys must both be CUDA tensors")

    k_top_kernel = _resolve_kernel_topk(int(k_top))
    if block_q <= 0 or block_k <= 0 or block_d <= 0:
        raise ValueError(
            f"block sizes must be positive, got block_q={block_q}, block_k={block_k}, block_d={block_d}"
        )

    q = queries.contiguous()
    k = keys.contiguous()
    if normalize:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

    # Prototype path: prefer fp16 input to nudge Tensor Core utilization.
    if q.dtype != torch.float16:
        q = q.to(torch.float16)
    if k.dtype != torch.float16:
        k = k.to(torch.float16)

    num_q, head_dim = q.shape
    num_k = k.shape[0]
    if k_top > num_k:
        raise ValueError(f"k_top={k_top} exceeds num_k={num_k}")
    if block_q > _MAX_EFFECTIVE_BLOCK_Q:
        block_q = _MAX_EFFECTIVE_BLOCK_Q
    block_k_kernel = _resolve_block_k(block_k, k_top_kernel)
    block_k_kernel = min(block_k_kernel, _MAX_EFFECTIVE_BLOCK_K)
    scores_full = torch.empty((num_q, k_top_kernel), device=q.device, dtype=torch.float32)
    indices_full = torch.empty((num_q, k_top_kernel), device=q.device, dtype=torch.int32)

    num_warps = 4 if block_q <= 64 and block_k_kernel <= 256 else 8
    num_stages = 2 if block_k_kernel <= 256 else 1

    q_chunk = num_q if launch_q_chunk <= 0 else min(num_q, int(launch_q_chunk))
    q_chunk = max(block_q, q_chunk)
    num_chunks = triton.cdiv(num_q, q_chunk)
    for chunk_idx, q_start in enumerate(range(0, num_q, q_chunk), start=1):
        q_end = min(num_q, q_start + q_chunk)
        q_sub = q[q_start:q_end]
        scores_sub = scores_full[q_start:q_end]
        indices_sub = indices_full[q_start:q_end]
        if verbose and num_chunks > 1:
            print(
                f"[RetrievalAttention] gpu_topk(custom_fused) chunk {chunk_idx}/{num_chunks} "
                f"q_range=[{q_start},{q_end})",
                flush=True,
            )

        grid = (triton.cdiv(q_sub.shape[0], block_q),)
        _fused_qk_running_topk_kernel[grid](
            q_sub,
            k,
            scores_sub,
            indices_sub,
            q_sub.shape[0],
            num_k,
            head_dim,
            q_sub.stride(0),
            q_sub.stride(1),
            k.stride(0),
            k.stride(1),
            scores_sub.stride(0),
            scores_sub.stride(1),
            indices_sub.stride(0),
            indices_sub.stride(1),
            K_TOP=k_top_kernel,
            BLOCK_Q=block_q,
            BLOCK_K=block_k_kernel,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if k_top_kernel != k_top:
        scores_full = scores_full[:, :k_top]
        indices_full = indices_full[:, :k_top]
    if return_scores:
        return scores_full, indices_full
    return torch.empty((0,), device=q.device, dtype=torch.float32), indices_full


@torch.no_grad()
def benchmark_fused_qk_topk(
    queries: torch.Tensor,
    keys: torch.Tensor,
    k_top: int,
    *,
    warmup: int = 3,
    iters: int = 10,
) -> Dict[str, float]:
    """
    Microbenchmark helper for isolated experiments.
    Compares this prototype kernel with torch matmul+topk baseline.
    """
    if queries.device.type != "cuda" or keys.device.type != "cuda":
        raise ValueError("benchmark requires CUDA tensors")

    # Warmup
    for _ in range(max(0, warmup)):
        fused_qk_topk_triton(queries, keys, k_top, normalize=True)
    torch.cuda.synchronize(device=queries.device)

    t0 = time.time()
    for _ in range(max(1, iters)):
        fused_qk_topk_triton(queries, keys, k_top, normalize=True)
    torch.cuda.synchronize(device=queries.device)
    fused_ms = (time.time() - t0) * 1000.0 / float(max(1, iters))

    q = F.normalize(queries.float(), dim=-1)
    k = F.normalize(keys.float(), dim=-1)
    for _ in range(max(0, warmup)):
        scores = torch.matmul(q, k.transpose(0, 1))
        torch.topk(scores, k=k_top, dim=1, sorted=False)
    torch.cuda.synchronize(device=queries.device)

    t0 = time.time()
    for _ in range(max(1, iters)):
        scores = torch.matmul(q, k.transpose(0, 1))
        torch.topk(scores, k=k_top, dim=1, sorted=False)
    torch.cuda.synchronize(device=queries.device)
    baseline_ms = (time.time() - t0) * 1000.0 / float(max(1, iters))

    return {
        "fused_ms": fused_ms,
        "baseline_ms": baseline_ms,
        "speedup_x": (baseline_ms / fused_ms) if fused_ms > 0 else 0.0,
    }


__all__ = [
    "fused_qk_topk_triton",
    "benchmark_fused_qk_topk",
]
