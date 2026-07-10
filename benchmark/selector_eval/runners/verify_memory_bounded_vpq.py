#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import build_page_pq_torch
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_one_group import (
    _rowwise_int8_qdq,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_vprefix import (
    JointVPrefixGridRuntime,
    build_joint_vprefix_grid,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_value import (
    reconstruct_vpq_values_from_pack_torch,
    value_vpq_code_stat_risk_from_pack_streaming_torch,
    value_vpq_code_stat_risk_torch,
    value_vpq_pack_torch,
    vpq_values_for_tokens_gpu,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_vpq_sidecars import (
    memory_bounded_vpq_code_error_for,
)


def main() -> None:
    torch.manual_seed(17)
    context_len = 48
    dim = 8
    dynamic_start = 4
    indexed_end = 44
    page_size = 8
    values = torch.randn((context_len, dim), dtype=torch.float32).to(torch.bfloat16)
    index = build_page_pq_torch(
        values,
        dynamic_start=dynamic_start,
        indexed_end=indexed_end,
        page_size=page_size,
        subvecs=2,
        subbits=2,
        kmeans_iters=1,
        seed=29,
        key_bytes=2,
        router_enabled=False,
        router_prototypes=0,
        router_merge_rel=0.0,
        router_merge_var=0.0,
        router_max_groups=0,
        device=torch.device("cpu"),
    )
    tokens = torch.arange(context_len, dtype=torch.long)
    vhat, valid, page_ids, actual_subbits = vpq_values_for_tokens_gpu(
        index=index,
        values=values,
        values_np=None,
        tokens=tokens,
        subbits=2,
        value_subvecs=1,
        value_subbits=2,
        prefer_torch=True,
        value_bytes=2,
        kmeans_iters=1,
    )
    residual = values.float() - vhat.float()
    code_error_full, _ = value_vpq_code_stat_risk_torch(
        index=index,
        values=values,
        vhat_all=vhat,
        residual_all=residual,
        valid=valid,
        page_ids=page_ids,
        subbits=2,
        value_subvecs=1,
        value_subbits=2,
        value_bytes=2,
        kmeans_iters=1,
    )
    pack = value_vpq_pack_torch(
        index=index,
        values=values,
        value_subvecs=1,
        value_subbits=2,
        key_bytes=2,
        device=torch.device("cpu"),
        kmeans_iters=1,
    )
    if pack is None:
        raise AssertionError("missing V-PQ pack")
    code_error_stream, _ = value_vpq_code_stat_risk_from_pack_streaming_torch(
        values=values,
        pack=pack,
        context_len=context_len,
        dynamic_start=dynamic_start,
    )
    if not torch.equal(code_error_stream, code_error_full):
        raise AssertionError("streaming code_error changed bits")

    module = SimpleNamespace()
    cache_key = (0, "cpu", 2, 1, 2, 2, len(index.pages), dynamic_start, indexed_end, page_size)
    cached_prefix, _ = memory_bounded_vpq_code_error_for(
        module=module,
        cache_key=cache_key,
        values_t=values,
        pack=pack,
        context_len_i=indexed_end,
    )
    cached_full, _ = memory_bounded_vpq_code_error_for(
        module=module,
        cache_key=cache_key,
        values_t=values,
        pack=pack,
        context_len_i=context_len,
    )
    if not torch.equal(cached_prefix, code_error_full[:indexed_end]):
        raise AssertionError("cached code_error prefix changed bits")
    if not torch.equal(cached_full, code_error_full):
        raise AssertionError("exact-suffix code_error append changed bits")
    cache_entry = next(iter(module._pagedpq_joint_memory_bounded_vpq_cache.values()))
    if cache_entry[2].ndim != 1:
        raise AssertionError("memory-bounded cache retained a context-by-dim plane")

    vhat_rebuilt = reconstruct_vpq_values_from_pack_torch(
        values=values,
        pack=pack,
        dynamic_start=dynamic_start,
        context_len=context_len,
    )
    if not torch.equal(vhat_rebuilt, vhat):
        raise AssertionError("transient vhat reconstruction changed bits")
    residual_rebuilt = values.float() - vhat_rebuilt.float()
    if not torch.equal(residual_rebuilt, residual):
        raise AssertionError("transient residual reconstruction changed bits")

    k_count = 2
    heads = 2
    probs = torch.softmax(torch.randn((k_count, heads, context_len)), dim=2)
    base = (probs.reshape(k_count * heads, context_len) @ vhat).reshape(k_count, heads, dim)
    values_lo = _rowwise_int8_qdq(values.float())
    commit_mask = (
        (values.float() - values_lo).pow(2).sum(dim=1, dtype=torch.float64).to(torch.float16)
        < code_error_full.to(torch.float16)
    )
    residual_lo = torch.where(
        commit_mask.reshape(-1, 1),
        values_lo - vhat,
        torch.zeros((), dtype=torch.float32),
    )
    common = dict(
        args=SimpleNamespace(profile_native_ops=False),
        layer_id=0,
        stats={},
        device=torch.device("cpu"),
        wall_profile_enabled=False,
        use_incremental_v_grid=False,
        max_exact_v_count=7,
        context_len=context_len,
        k_count=k_count,
        group_heads=heads,
        head_dim=dim,
        prob_dtype=torch.float32,
        probs_grid=probs,
        base_output_grid=base,
        code_error=code_error_full,
        joint_v_budgets=[3, 7],
        joint_v_budgets_t=torch.tensor([3, 7], dtype=torch.long),
        v_commit_mask=commit_mask,
        v_hi_frac=0.1,
    )
    old = build_joint_vprefix_grid(
        JointVPrefixGridRuntime(
            **common,
            residual=residual,
            residual_lo_commit=residual_lo,
        )
    )

    residual_lo_rebuilt = torch.where(
        commit_mask.reshape(-1, 1),
        values_lo - vhat_rebuilt,
        torch.zeros((), dtype=torch.float32),
    )
    bounded = build_joint_vprefix_grid(
        JointVPrefixGridRuntime(
            **common,
            residual=residual_rebuilt,
            residual_lo_commit=residual_lo_rebuilt,
        )
    )
    if not torch.equal(bounded.grid_outputs, old.grid_outputs):
        raise AssertionError("memory-bounded V-prefix outputs changed bits")
    if not torch.equal(bounded.v_lo_reads_grid, old.v_lo_reads_grid):
        raise AssertionError("memory-bounded V-prefix commit counts changed")
    print("memory-bounded V-PQ CPU parity: PASS")


if __name__ == "__main__":
    main()
