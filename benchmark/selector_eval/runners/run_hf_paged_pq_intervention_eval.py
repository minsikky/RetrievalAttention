#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    _sync_if_cuda,
    parse_csv_ints,
)

from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import (
    log,
    _env_truthy,
    _require_canonical_gpu_frontier,
    cache_sequence_length,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_stats import ApproxStats
from benchmark.selector_eval.runners.hf_paged_pq_intervention_value import (
    parse_head_budget_map,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_trace import (
    make_needle_prompt,
    greedy_dense_trace,
    teacher_forced_trace,
    summarize_logit_trace,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_forward_state import PagedPQForwardState
from benchmark.selector_eval.runners.hf_paged_pq_intervention_index_sidecars import build_decode_index_sidecars
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint import (
    JointKVDecodeContext,
    approximate_joint_kv_all_heads,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_patch_state import PagedPQPatchState
from benchmark.selector_eval.runners.hf_paged_pq_intervention_qkv import project_and_update_kv_cache
from benchmark.selector_eval.runners.hf_paged_pq_intervention_cli import (
    approx_stats_payload,
    build_hf_paged_pq_intervention_arg_parser,
)

@contextlib.contextmanager
def patched_paged_pq_attention(model, layer_ids: list[int], args, stats: dict[int, ApproxStats]):
    originals = {}
    device = next(model.parameters()).device
    _require_canonical_gpu_frontier(args)
    key_bytes = int(args.key_bytes)
    value_bytes = int(args.value_bytes)
    nprobes = parse_csv_ints(args.nprobes)
    budget_by_head = parse_head_budget_map(args.budget_by_head)
    online_confidence_rule = str(getattr(args, "online_confidence_rule", "none"))
    if online_confidence_rule != "joint_kv_stability":
        raise ValueError(
            "HF paged-PQ intervention runner is canonical-only; "
            "set online_confidence_rule=joint_kv_stability"
        )
    if str(args.selector_mode) != "fullscan":
        raise ValueError("HF paged-PQ intervention runner is canonical-only; set selector_mode=fullscan")
    if str(args.selector_backend) not in {"cuda_ext", "auto"}:
        raise ValueError("HF paged-PQ intervention runner requires selector_backend=cuda_ext or auto")
    if str(args.tail_mode) != "vpq_value":
        raise ValueError("HF paged-PQ intervention runner is canonical-only; set tail_mode=vpq_value")
    if str(args.selected_value_mode) != "vpq_value":
        raise ValueError("HF paged-PQ intervention runner is canonical-only; set selected_value_mode=vpq_value")
    if str(args.selected_value_exact_rule) != "global_residual_risk":
        raise ValueError(
            "HF paged-PQ intervention runner is canonical-only; "
            "set selected_value_exact_rule=global_residual_risk"
        )
    if bool(getattr(args, "approx_prefill", False)):
        raise ValueError("HF paged-PQ intervention runner is decode-only; disable approx_prefill")
    patch_state = PagedPQPatchState(
        args=args,
        layer_ids=layer_ids,
        device=device,
        stats=stats,
        key_bytes=key_bytes,
        value_bytes=value_bytes,
        online_confidence_rule=online_confidence_rule,
    )
    dense_decode_key_t_float_cache = patch_state.dense_decode_key_t_float_cache
    warm_dense_prefill_decode_sidecars = patch_state.warm_dense_prefill_decode_sidecars

    def make_forward(layer_id: int, module):
        original_forward = module.forward

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            past_key_values = kwargs.pop("past_key_values", None)
            cache_obj = past_key_value if past_key_value is not None else past_key_values

            def call_original_forward():
                original_kwargs = dict(kwargs)
                if past_key_value is not None:
                    original_kwargs["past_key_value"] = past_key_value
                if past_key_values is not None:
                    original_kwargs["past_key_values"] = past_key_values
                if cache_position is not None:
                    original_kwargs["cache_position"] = cache_position
                return original_forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    **original_kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            query_len = int(input_shape[-1])
            if cache_obj is None:
                stats[layer_id].add_passthrough_attention_call()
                return call_original_forward()
            if query_len != 1:
                stats[layer_id].add_passthrough_attention_call()
                out = call_original_forward()
                warm_dense_prefill_decode_sidecars(int(layer_id), self, cache_obj)
                return out

            if cache_position is not None and torch.numel(cache_position) > 0:
                estimated_context_len = int(cache_position.reshape(-1)[-1].item()) + 1
            else:
                past_len = cache_sequence_length(cache_obj, int(layer_id))
                estimated_context_len = (int(past_len) + query_len) if past_len is not None else query_len
            estimated_dynamic_start = min(max(0, int(args.static_prefix)), estimated_context_len)
            estimated_indexed_end = max(estimated_dynamic_start, estimated_context_len - max(0, int(args.static_suffix)))
            estimated_sealed_end = estimated_dynamic_start + (
                (max(0, estimated_indexed_end - estimated_dynamic_start) // max(1, int(args.page_size)))
                * max(1, int(args.page_size))
            )
            min_budget_est = min(
                int(budget_by_head.get(int(head), int(args.budget)))
                for head in range(int(getattr(self, "num_heads", self.config.num_attention_heads)))
            )
            sealed_indexed_tokens_est = max(0, int(estimated_sealed_end) - int(estimated_dynamic_start))
            estimated_tail_blend = (
                float(args.prefill_tail_blend)
                if query_len > 1 and getattr(args, "prefill_tail_blend", None) is not None
                else (
                    float(args.decode_tail_blend)
                    if query_len == 1 and getattr(args, "decode_tail_blend", None) is not None
                    else float(args.tail_blend)
                )
            )
            dense_equivalent = (
                str(args.selector_mode) == "fullscan"
                and (
                    int(sealed_indexed_tokens_est) <= 0
                    or (
                        int(min_budget_est) >= int(sealed_indexed_tokens_est)
                        and str(args.selected_value_mode) == "exact"
                        and float(estimated_tail_blend) <= 0.0
                    )
                )
            )
            if dense_equivalent:
                num_heads_est = int(getattr(self, "num_heads", self.config.num_attention_heads))
                query_start_est = int(estimated_context_len) - int(query_len)
                for local_qpos_est in range(query_len):
                    query_context_len_est = int(query_start_est + local_qpos_est + 1)
                    stats[layer_id].add_count_repeated(
                        num_heads_est,
                        int(query_context_len_est),
                        0,
                        0.0,
                        int(self.head_dim),
                        key_bytes,
                        value_bytes,
                    )
                stats[layer_id].add_passthrough_attention_call()
                return call_original_forward()

            stats[layer_id].add_approx_attention_call()
            wall_profile_enabled = _env_truthy("SELECTOR_PQ_JOINT_WALL_PROFILE", "0")
            patched_attention_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                patched_attention_t0 = time.perf_counter()
            else:
                patched_attention_t0 = 0.0

            qkv_state = project_and_update_kv_cache(
                module=self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache_obj=cache_obj,
                cache_position=cache_position,
                input_shape=input_shape,
                query_len=query_len,
                layer_stats=stats[int(layer_id)],
                device=device,
                wall_profile_enabled=wall_profile_enabled,
                profile_native_ops=bool(getattr(args, "profile_native_ops", False)),
            )
            keys_all = qkv_state.keys_all
            values_all = qkv_state.values_all
            q_all = qkv_state.q_all
            context_len = qkv_state.context_len
            num_heads = qkv_state.num_heads
            num_kv_heads = qkv_state.num_kv_heads
            group_size = qkv_state.group_size

            index_sidecars = build_decode_index_sidecars(
                args=args,
                module=self,
                layer_stats=stats[int(layer_id)],
                device=device,
                keys_all=keys_all,
                values_all=values_all,
                context_len=context_len,
                query_len=query_len,
                num_kv_heads=num_kv_heads,
                online_confidence_rule=online_confidence_rule,
                key_bytes=key_bytes,
                wall_profile_enabled=wall_profile_enabled,
            )
            index_cache = index_sidecars.index_cache
            prefix_index_cache = index_sidecars.prefix_index_cache
            torch_k_cache = index_sidecars.torch_k_cache
            torch_v_cache = index_sidecars.torch_v_cache

            forward_state = PagedPQForwardState(
                args=args,
                module=self,
                patch_state=patch_state,
                layer_id=int(layer_id),
                stats=stats[int(layer_id)],
                device=device,
                values_all=values_all,
                index_cache=index_cache,
                prefix_index_cache=prefix_index_cache,
                context_len=context_len,
                num_kv_heads=num_kv_heads,
                value_bytes=value_bytes,
            )

            joint_decode_context = JointKVDecodeContext(
                args=args,
                model=model,
                module=self,
                layer_id=int(layer_id),
                stats=stats,
                device=device,
                hidden_states=hidden_states,
                q_all=q_all,
                keys_all=keys_all,
                torch_k_cache=torch_k_cache,
                torch_v_cache=torch_v_cache,
                dense_decode_key_t_float_cache=dense_decode_key_t_float_cache,
                num_heads=int(num_heads),
                num_kv_heads=int(num_kv_heads),
                group_size=int(group_size),
                nprobes=nprobes,
                online_confidence_rule=online_confidence_rule,
                key_bytes=int(key_bytes),
                value_bytes=int(value_bytes),
                wall_profile_enabled=bool(wall_profile_enabled),
                forward_state=forward_state,
            )

            joint_outputs = approximate_joint_kv_all_heads(joint_decode_context, 0, context_len)
            if joint_outputs is None:
                raise RuntimeError(
                    "canonical joint K/V decode path declined the request; "
                    "check selector mode, backend, and required frontier CUDA flags"
                )
            attn_output = joint_outputs.reshape(1, 1, -1).contiguous()

            oproj_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                oproj_t0 = time.perf_counter()
            else:
                oproj_t0 = 0.0
            attn_output = self.o_proj(attn_output)
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                stats[layer_id].add_output_projection_timing(time.perf_counter() - oproj_t0)
            if wall_profile_enabled:
                stats[layer_id].add_wall_output_projection_timing(time.perf_counter() - oproj_wall_t0)
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                stats[layer_id].add_patched_attention_timing(time.perf_counter() - patched_attention_t0)
            if wall_profile_enabled:
                stats[layer_id].add_wall_patched_attention_timing(
                    time.perf_counter() - patched_attention_wall_t0
                )
            return attn_output, None

        return types.MethodType(forward, module)

    try:
        for layer_id in layer_ids:
            module = model.model.layers[int(layer_id)].self_attn
            originals[int(layer_id)] = module.forward
            stats[int(layer_id)] = ApproxStats()
            module.forward = make_forward(int(layer_id), module)
            setattr(
                module,
                "_pagedpq_warm_decode_sidecars",
                lambda cache_obj, lid=int(layer_id), mod=module: warm_dense_prefill_decode_sidecars(
                    int(lid),
                    mod,
                    cache_obj,
                ),
            )
        yield
    finally:
        for layer_id, forward in originals.items():
            module = model.model.layers[int(layer_id)].self_attn
            module.forward = forward
            if hasattr(module, "_pagedpq_warm_decode_sidecars"):
                delattr(module, "_pagedpq_warm_decode_sidecars")
            if hasattr(module, "_pagedpq_joint_vpq_sidecar_cache"):
                delattr(module, "_pagedpq_joint_vpq_sidecar_cache")
            if hasattr(module, "_pagedpq_joint_grouped_vpq_sidecar_cache"):
                delattr(module, "_pagedpq_joint_grouped_vpq_sidecar_cache")
        if hasattr(model, "_pagedpq_joint_score_grid_workspace_cache"):
            delattr(model, "_pagedpq_joint_score_grid_workspace_cache")
        if hasattr(model, "_pagedpq_joint_softmax_base_workspace_cache"):
            delattr(model, "_pagedpq_joint_softmax_base_workspace_cache")
        if hasattr(model, "_pagedpq_joint_grouped_output_workspace_cache"):
            delattr(model, "_pagedpq_joint_grouped_output_workspace_cache")


def run() -> None:
    parser = build_hf_paged_pq_intervention_arg_parser()
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    device = torch.device(args.device)
    log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, local_files_only=bool(args.local_files_only))
    tokenizer.pad_token = tokenizer.eos_token
    log("loading model")
    load_kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": bool(args.local_files_only),
        "torch_dtype": torch.bfloat16,
        "attn_implementation": str(args.attn_implementation),
    }
    if not bool(args.cpu_then_to_device):
        load_kwargs["device_map"] = {"": str(device)}
    model = LlamaForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    if bool(args.cpu_then_to_device):
        log(f"moving model to {device}")
        model = model.to(device)
    model.eval()
    forbidden = {tok for tok in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")] if isinstance(tok, int) and tok >= 0}

    prompt = make_needle_prompt(str(args.target), int(args.filler_repeats))
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    log(f"prompt_tokens={int(input_ids.shape[1])} dense_trace_start")
    start = time.perf_counter()
    dense = greedy_dense_trace(model, input_ids, int(args.max_new_tokens), forbidden)
    dense_time = time.perf_counter() - start
    log(f"dense_trace_done seconds={dense_time:.3f}")
    dense_text = tokenizer.decode(dense["tokens"], skip_special_tokens=True)

    layer_ids = parse_csv_ints(args.layers)
    approx_stats: dict[int, ApproxStats] = {}
    start = time.perf_counter()
    log(f"approx_trace_start layers={layer_ids}")
    with patched_paged_pq_attention(model, layer_ids, args, approx_stats):
        approx_teacher = teacher_forced_trace(model, input_ids, dense["tokens"], forbidden)
        approx_free = greedy_dense_trace(model, input_ids, int(args.max_new_tokens), forbidden)
    approx_time = time.perf_counter() - start
    log(f"approx_trace_done seconds={approx_time:.3f}")
    approx_text = tokenizer.decode(approx_free["tokens"], skip_special_tokens=True)

    comparison = summarize_logit_trace(dense, approx_teacher, tokenizer, ignore_token_ids=forbidden)
    stats_payload = approx_stats_payload(approx_stats)

    task = {
        "target": str(args.target),
        "prompt_tokens": int(input_ids.shape[1]),
        "dense_text": dense_text,
        "approx_free_text": approx_text,
        "dense_contains_target": str(args.target).lower() in dense_text.lower(),
        "approx_free_contains_target": str(args.target).lower() in approx_text.lower(),
        "free_run_exact_text_match": dense_text == approx_text,
        "dense_tokens": [int(x) for x in dense["tokens"]],
        "approx_free_tokens": [int(x) for x in approx_free["tokens"]],
    }
    summary = {
        "algorithm": (
            f"hf_routed_paged_pq_k{int(args.budget)}"
            f"_{args.selector_mode}"
            f"_rerank{int(args.rerank_candidates)}"
            f"+{args.tail_mode}_tail_b{int(args.tail_bands)}_s{int(args.tail_samples)}"
            f"_blend{float(args.tail_blend):g}"
        ),
        "layers": layer_ids,
        "dense_seconds": float(dense_time),
        "approx_seconds": float(approx_time),
        "logit_trace": comparison["summary"],
        "task": task,
        "cost_proxy": stats_payload,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "logit_steps.json").write_text(json.dumps(comparison["steps"], indent=2, sort_keys=True), encoding="utf-8")
    log(f"wrote {out_dir}")


if __name__ == "__main__":
    run()
