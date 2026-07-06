#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.data.trace import load_trace, static_tokens, unique_tokens
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import build_page_pq_gpu, parse_csv_ints, rank_paged_pq
from benchmark.selector_eval.metrics.attention import _output_error_metrics
from benchmark.selector_eval.runners.run_joint_kv_budget_policy_eval import (
    apply_score_proxy_variant,
    choose_action,
    load_safetensor_weight,
    load_weight_index,
    mixed_scores_for_variant,
    parse_csv_floats,
    rel_l2,
)
from benchmark.selector_eval.runners.run_layer_quality_eval import _selected_for_budget, _vpq_values_for_tokens
from benchmark.selector_eval.runners.run_value_exact_strategy_eval import (
    dense_attention_output,
    output_from_exact_mask,
    project_head_subset,
    top_mask,
    value_vpq_code_stat_risk,
)

MB = 1024.0 * 1024.0


def _csv_names(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _float(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _policy_path(
    *,
    outputs: dict[tuple[int, int], np.ndarray],
    dense_output: np.ndarray,
    max_output: np.ndarray,
    k_budgets: list[int],
    v_budgets: list[int],
    policy: str,
    threshold: float,
    k_mb_by_idx: list[float],
    v_mb_by_idx: list[float],
) -> list[dict[str, object]]:
    ki = 0
    vi = 0
    rows: list[dict[str, object]] = []
    for step in range(len(k_budgets) + len(v_budgets) + 4):
        cur = outputs[(ki, vi)]
        k_can = ki + 1 < len(k_budgets)
        v_can = vi + 1 < len(v_budgets)
        k_delta = rel_l2(cur, outputs[(ki + 1, vi)]) if k_can else 0.0
        v_delta = rel_l2(cur, outputs[(ki, vi + 1)]) if v_can else 0.0
        extra_k_mb = float(k_mb_by_idx[ki + 1] - k_mb_by_idx[ki]) if k_can else float("inf")
        extra_v_mb = float(v_mb_by_idx[vi + 1] - v_mb_by_idx[vi]) if v_can else float("inf")
        action = choose_action(
            policy=policy,
            k_delta=k_delta,
            v_delta=v_delta,
            k_can=k_can,
            v_can=v_can,
            threshold=float(threshold),
            turn=step,
            extra_k_mb=extra_k_mb,
            extra_v_mb=extra_v_mb,
        )
        next_ki = ki + 1 if action == "k" and k_can else ki
        next_vi = vi + 1 if action == "v" and v_can else vi
        next_delta = rel_l2(cur, outputs[(next_ki, next_vi)]) if (next_ki, next_vi) != (ki, vi) else 0.0
        rows.append(
            {
                "step": int(step),
                "k_index": int(ki),
                "v_index": int(vi),
                "k_budget": int(k_budgets[ki]),
                "v_budget": int(v_budgets[vi]),
                "step_MB_per_head": float(k_mb_by_idx[ki] + v_mb_by_idx[vi]),
                "relL2_to_dense": float(rel_l2(cur, dense_output)),
                "delta_to_max_budget": float(rel_l2(cur, max_output)),
                "next_k_delta": float(k_delta),
                "next_v_delta": float(v_delta),
                "max_next_delta": float(max(k_delta, v_delta)),
                "chosen_action_delta": float(next_delta),
                "action": str(action),
                "accepted": bool(action == "stop"),
            }
        )
        if action == "stop":
            break
        if (next_ki, next_vi) == (ki, vi):
            break
        ki, vi = next_ki, next_vi
    return rows


def _monotonicity_stats(rows: list[dict[str, object]]) -> dict[str, float]:
    by_key: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_key[(int(row["qidx"]), int(row["head"]), int(row["v_budget"]))].append(row)
    k_total = 0
    k_good = 0
    for vals in by_key.values():
        vals.sort(key=lambda r: int(r["k_budget"]))
        prev = None
        for row in vals:
            cur = _float(row["relL2_to_dense"])
            if prev is not None:
                k_total += 1
                if cur <= prev + 1e-12:
                    k_good += 1
            prev = cur
    by_key.clear()
    for row in rows:
        by_key[(int(row["qidx"]), int(row["head"]), int(row["k_budget"]))].append(row)
    v_total = 0
    v_good = 0
    for vals in by_key.values():
        vals.sort(key=lambda r: int(r["v_budget"]))
        prev = None
        for row in vals:
            cur = _float(row["relL2_to_dense"])
            if prev is not None:
                v_total += 1
                if cur <= prev + 1e-12:
                    v_good += 1
            prev = cur
    accepted = [r for r in rows if bool(r.get("accepted"))]
    return {
        "k_relL2_nonincreasing_fraction": float(k_good / k_total) if k_total else float("nan"),
        "v_relL2_nonincreasing_fraction": float(v_good / v_total) if v_total else float("nan"),
        "accepted_count": float(len(accepted)),
        "accepted_mean_delta_to_max": float(np.mean([_float(r["delta_to_max_budget"]) for r in accepted])) if accepted else float("nan"),
        "accepted_max_delta_to_max": float(np.max([_float(r["delta_to_max_budget"]) for r in accepted])) if accepted else float("nan"),
    }


def run() -> None:
    parser = argparse.ArgumentParser(description="Budget convergence diagnostic for joint K/V frontier confidence.")
    parser.add_argument("--qkv_trace", required=True)
    parser.add_argument("--x_trace", required=True)
    parser.add_argument(
        "--model_snapshot",
        default=".hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_lengths", default="500,1000,2000,4000,8000,16000,32000,64000,128000")
    parser.add_argument("--max_qidx_per_decode", type=int, default=1)
    parser.add_argument("--heads", default="")
    parser.add_argument("--k_budgets", default="1024,2048,3072,4096,6144,8192,12288,14336,16384,24576,32768")
    parser.add_argument("--v_budgets", default="256,512,1024,1536,2048,3072,4096,6144,8192,12288,16384")
    parser.add_argument("--policy_k_budgets", default="4096,8192,14336,32768")
    parser.add_argument("--policy_v_budgets", default="1024,2048,4096,6144,8192,12288,16384")
    parser.add_argument("--policy", default="k_first_alternating")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--score_proxy_variant", default="baseline")
    parser.add_argument("--tail_score_calibration", choices=["none", "affine_selected"], default="none")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=8)
    parser.add_argument("--value_subvecs", type=int, default=1)
    parser.add_argument("--value_subbits", type=int, default=4)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--code_stat_bytes", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    t0 = time.perf_counter()

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    trace = load_trace(args.qkv_trace)
    q_indices = trace.q_indices_for_decodes(parse_csv_ints(args.decode_lengths))
    if int(args.max_qidx_per_decode) > 0:
        limited: list[int] = []
        counts: dict[int, int] = {}
        for qidx in q_indices:
            decode = trace.decode_tokens_for_qidx(int(qidx))
            if counts.get(int(decode), 0) >= int(args.max_qidx_per_decode):
                continue
            limited.append(int(qidx))
            counts[int(decode)] = counts.get(int(decode), 0) + 1
        q_indices = limited
    if not q_indices:
        raise ValueError("no query indices selected")

    heads = parse_csv_ints(args.heads) if str(args.heads).strip() else list(range(int(trace.num_heads)))
    k_budgets = sorted(set(parse_csv_ints(args.k_budgets) + parse_csv_ints(args.policy_k_budgets)))
    v_budgets = sorted(set(parse_csv_ints(args.v_budgets) + parse_csv_ints(args.policy_v_budgets)))
    policy_k_budgets = sorted(set(parse_csv_ints(args.policy_k_budgets)))
    policy_v_budgets = sorted(set(parse_csv_ints(args.policy_v_budgets)))
    policy_ki = [k_budgets.index(k) for k in policy_k_budgets]
    policy_vi = [v_budgets.index(v) for v in policy_v_budgets]

    x_data = np.load(args.x_trace, mmap_mode="r")
    x_meta = json.loads(str(x_data["metadata"].item()))
    layer_idx = int(x_meta["layer_idx"])
    model_dir = PROJECT_ROOT / args.model_snapshot
    weight_map = load_weight_index(model_dir)
    wo = load_safetensor_weight(model_dir, weight_map, f"model.layers.{layer_idx}.self_attn.o_proj.weight", device)

    per_head_grid_rows: list[dict[str, object]] = []
    layer_grid_rows: list[dict[str, object]] = []
    policy_path_rows: list[dict[str, object]] = []

    for qidx in q_indices:
        position = int(trace.positions[int(qidx)])
        decode_tokens = int(trace.decode_tokens_for_qidx(int(qidx)))
        context_len = int(position) + 1
        dynamic_start = min(max(0, int(args.static_prefix)), int(trace.input_len))
        indexed_end = max(
            min(max(0, int(args.static_prefix)), int(trace.input_len)),
            min(context_len - max(0, int(args.static_suffix)), trace.keys.shape[1]),
        )
        needed_kv_heads = sorted({int(trace.kv_head_for(h)) for h in heads})
        index_cache = {}
        for kv_head in needed_kv_heads:
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            index_cache[kv_head] = build_page_pq_gpu(
                keys_np,
                dynamic_start=dynamic_start,
                indexed_end=indexed_end,
                page_size=int(args.page_size),
                subvecs=int(args.subvecs),
                subbits=int(args.subbits),
                kmeans_iters=int(args.kmeans_iters),
                seed=2025 + 2027 * int(kv_head),
                key_bytes=int(args.key_bytes),
                router_enabled=False,
                router_prototypes=16,
                router_merge_rel=0.05,
                router_merge_var=0.0,
                router_max_groups=512,
                device=device,
            )

        dense_heads: dict[int, np.ndarray] = {}
        head_outputs: dict[int, dict[tuple[int, int], np.ndarray]] = {}
        head_k_mb: dict[int, list[float]] = {}
        head_v_mb: dict[int, list[float]] = {}
        head_selected: dict[int, list[int]] = {}

        for head in heads:
            kv_head = int(trace.kv_head_for(int(head)))
            index = index_cache[kv_head]
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            values_np = trace.values[kv_head, :context_len].astype(np.float32, copy=False)
            query_np = trace.queries[int(head), int(qidx)].astype(np.float32, copy=False)
            scores_np, _true_probs, dense_head = dense_attention_output(keys_np, values_np, query_np)
            dense_heads[int(head)] = dense_head

            pending = list(range(max(0, int(index.pending_start)), max(0, min(int(index.indexed_end), context_len))))
            base = unique_tokens(
                static_tokens(position, int(args.static_prefix), int(args.static_suffix)) + pending,
                context_len=context_len,
            )
            query_t = torch.as_tensor(query_np, dtype=torch.float32, device=device)
            ranked_t, ranked_scores_t, _selector_seconds, selector_mb, _chosen_nprobe = rank_paged_pq(
                query_t,
                index,
                mode="fullscan",
                selector_backend="torch",
                nprobes=[512],
                budget=int(max(k_budgets)),
                key_bytes=int(args.key_bytes),
                subbits=int(args.subbits),
            )
            ranked_cpu = ranked_t.detach().cpu().numpy().astype(np.int64, copy=False)
            ranked_scores_cpu = ranked_scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
            ranked_cpu, ranked_scores_cpu, score_proxy_extra_mb, _score_proxy_meta = apply_score_proxy_variant(
                variant=str(args.score_proxy_variant),
                index=index,
                keys_np=keys_np,
                query_np=query_np,
                ranked_cpu=ranked_cpu,
                ranked_scores_cpu=ranked_scores_cpu,
                key_bytes=int(args.key_bytes),
                metadata_bytes=int(args.code_stat_bytes),
                kmeans_iters=int(args.kmeans_iters),
                seed=2025 + 4093 * int(kv_head) + 31 * int(head) + int(context_len),
            )

            all_tokens = np.arange(context_len, dtype=np.int64)
            vhat_all, _compressed_v_mb, _fallback_v_mb = _vpq_values_for_tokens(
                index=index,
                values_np=values_np,
                tokens=all_tokens,
                subbits=int(args.subbits),
                value_subvecs=int(args.value_subvecs),
                value_subbits=int(args.value_subbits),
                value_bytes=int(args.value_bytes),
            )
            residual = values_np.astype(np.float32, copy=False) - vhat_all.astype(np.float32, copy=False)
            code_error = value_vpq_code_stat_risk(
                index=index,
                values_np=values_np,
                residual=residual,
                subbits=int(args.subbits),
                value_subvecs=int(args.value_subvecs),
                value_subbits=int(args.value_subbits),
                sensitivity=None,
            )
            actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
            actual_value_subvecs = int(args.value_subvecs) if int(args.value_subvecs) > 0 else int(args.subvecs)
            code_bytes = 1 if actual_value_subbits <= 8 else 2
            metadata_mb = (
                float(context_len * actual_value_subvecs * code_bytes)
                + float(len(index.pages) * actual_value_subvecs * (1 << actual_value_subbits) * int(args.code_stat_bytes))
            ) / MB
            v_pq_codebook_mb = float(
                len(index.pages)
                * actual_value_subvecs
                * (1 << actual_value_subbits)
                * (int(trace.head_dim) // max(1, actual_value_subvecs))
                * int(args.value_bytes)
            ) / MB
            v_mb_by_idx = []
            for v_budget in v_budgets:
                exact_count = max(0, min(int(v_budget), int(context_len)))
                exact_v_mb = float(exact_count * int(trace.head_dim) * int(args.value_bytes)) / MB
                compressed_v_codes_mb = float(max(0, context_len - exact_count) * actual_value_subvecs * code_bytes) / MB
                v_mb_by_idx.append(exact_v_mb + v_pq_codebook_mb + compressed_v_codes_mb + metadata_mb)

            outputs: dict[tuple[int, int], np.ndarray] = {}
            k_mb_by_idx: list[float] = []
            selected_counts: list[int] = []
            for ki, k_budget in enumerate(k_budgets):
                selected_cpu = _selected_for_budget(
                    base=base,
                    ranked_cpu=ranked_cpu,
                    budget=int(k_budget),
                    context_len=context_len,
                )
                selected_counts.append(int(selected_cpu.size))
                score_vec, _missing, _scale, _bias, calibration_extra_mb, _calibration_probe_count = mixed_scores_for_variant(
                    variant=str(args.score_proxy_variant),
                    context_len=context_len,
                    selected_cpu=selected_cpu,
                    ranked_cpu=ranked_cpu,
                    ranked_scores_cpu=ranked_scores_cpu,
                    exact_scores_np=scores_np,
                    query_dim=int(trace.head_dim),
                    calibrate=str(args.tail_score_calibration) == "affine_selected",
                    key_bytes=int(args.key_bytes),
                )
                probs = np.exp(score_vec - float(np.max(score_vec)))
                probs /= max(float(probs.sum()), 1e-20)
                exact_key_mb = float(selected_cpu.size * int(trace.head_dim) * int(args.key_bytes)) / MB
                k_mb_by_idx.append(float(selector_mb) + float(score_proxy_extra_mb) + exact_key_mb + float(calibration_extra_mb))
                for vi, v_budget in enumerate(v_budgets):
                    exact_count = max(0, min(int(v_budget), int(context_len)))
                    exact_mask = top_mask((probs * probs) * code_error, exact_count)
                    outputs[(ki, vi)] = output_from_exact_mask(
                        probs=probs,
                        vhat_all=vhat_all,
                        residual=residual,
                        exact_mask=exact_mask,
                    )

            max_output = outputs[(len(k_budgets) - 1, len(v_budgets) - 1)]
            head_outputs[int(head)] = outputs
            head_k_mb[int(head)] = k_mb_by_idx
            head_v_mb[int(head)] = v_mb_by_idx
            head_selected[int(head)] = selected_counts

            for ki, k_budget in enumerate(k_budgets):
                for vi, v_budget in enumerate(v_budgets):
                    out = outputs[(ki, vi)]
                    prev_k_delta = rel_l2(out, outputs[(ki - 1, vi)]) if ki > 0 else float("nan")
                    prev_v_delta = rel_l2(out, outputs[(ki, vi - 1)]) if vi > 0 else float("nan")
                    next_k_delta = rel_l2(out, outputs[(ki + 1, vi)]) if ki + 1 < len(k_budgets) else 0.0
                    next_v_delta = rel_l2(out, outputs[(ki, vi + 1)]) if vi + 1 < len(v_budgets) else 0.0
                    per_head_grid_rows.append(
                        {
                            "qidx": int(qidx),
                            "position": int(position),
                            "decode_length": int(decode_tokens),
                            "head": int(head),
                            "kv_head": int(kv_head),
                            "k_index": int(ki),
                            "v_index": int(vi),
                            "k_budget": int(k_budget),
                            "v_budget": int(v_budget),
                            "selected_k_tokens": int(selected_counts[ki]),
                            "step_MB_per_head": float(k_mb_by_idx[ki] + v_mb_by_idx[vi]),
                            "relL2_to_dense": float(rel_l2(out, dense_head)),
                            "delta_to_max_budget": float(rel_l2(out, max_output)),
                            "prev_k_delta": float(prev_k_delta),
                            "prev_v_delta": float(prev_v_delta),
                            "next_k_delta": float(next_k_delta),
                            "next_v_delta": float(next_v_delta),
                            "max_next_delta": float(max(next_k_delta, next_v_delta)),
                        }
                    )

            path_rows = _policy_path(
                outputs={ (a, b): outputs[(policy_ki[a], policy_vi[b])] for a in range(len(policy_ki)) for b in range(len(policy_vi)) },
                dense_output=dense_head,
                max_output=max_output,
                k_budgets=policy_k_budgets,
                v_budgets=policy_v_budgets,
                policy=str(args.policy),
                threshold=float(args.threshold),
                k_mb_by_idx=[k_mb_by_idx[i] for i in policy_ki],
                v_mb_by_idx=[v_mb_by_idx[i] for i in policy_vi],
            )
            for row in path_rows:
                row.update(
                    {
                        "qidx": int(qidx),
                        "position": int(position),
                        "decode_length": int(decode_tokens),
                        "head": int(head),
                        "kv_head": int(kv_head),
                        "policy": str(args.policy),
                        "threshold": float(args.threshold),
                    }
                )
                policy_path_rows.append(row)

        dense_concat = np.concatenate([dense_heads[int(head)] for head in heads], axis=0).astype(np.float32, copy=False)
        dense_proj = project_head_subset(
            concat_subset=dense_concat,
            heads=[int(h) for h in heads],
            num_heads=int(trace.num_heads),
            head_dim=int(trace.head_dim),
            wo=wo,
            device=device,
        )
        layer_outputs: dict[tuple[int, int], np.ndarray] = {}
        for ki, k_budget in enumerate(k_budgets):
            for vi, v_budget in enumerate(v_budgets):
                concat = np.concatenate([head_outputs[int(head)][(ki, vi)] for head in heads], axis=0).astype(np.float32, copy=False)
                proj = project_head_subset(
                    concat_subset=concat,
                    heads=[int(h) for h in heads],
                    num_heads=int(trace.num_heads),
                    head_dim=int(trace.head_dim),
                    wo=wo,
                    device=device,
                )
                layer_outputs[(ki, vi)] = proj
                concat_metric = _output_error_metrics(dense_concat, concat)
                proj_metric = _output_error_metrics(dense_proj, proj)
                mean_step_mb = float(np.mean([head_k_mb[int(head)][ki] + head_v_mb[int(head)][vi] for head in heads]))
                layer_grid_rows.append(
                    {
                        "qidx": int(qidx),
                        "position": int(position),
                        "decode_length": int(decode_tokens),
                        "k_index": int(ki),
                        "v_index": int(vi),
                        "k_budget": int(k_budget),
                        "v_budget": int(v_budget),
                        "mean_selected_k_tokens": float(np.mean([head_selected[int(head)][ki] for head in heads])),
                        "mean_step_MB_per_head": mean_step_mb,
                        "attn_concat_relative_L2": float(concat_metric["output_relative_l2"]),
                        "attn_o_proj_relative_L2": float(proj_metric["output_relative_l2"]),
                        "attn_o_proj_cosine": float(proj_metric["output_cosine"]),
                    }
                )
        max_layer_output = layer_outputs[(len(k_budgets) - 1, len(v_budgets) - 1)]
        for row in layer_grid_rows:
            if int(row["qidx"]) != int(qidx):
                continue
            ki = int(row["k_index"])
            vi = int(row["v_index"])
            out = layer_outputs[(ki, vi)]
            row["delta_to_max_budget"] = float(rel_l2(out, max_layer_output))
            row["prev_k_delta"] = float(rel_l2(out, layer_outputs[(ki - 1, vi)])) if ki > 0 else float("nan")
            row["prev_v_delta"] = float(rel_l2(out, layer_outputs[(ki, vi - 1)])) if vi > 0 else float("nan")
            row["next_k_delta"] = float(rel_l2(out, layer_outputs[(ki + 1, vi)])) if ki + 1 < len(k_budgets) else 0.0
            row["next_v_delta"] = float(rel_l2(out, layer_outputs[(ki, vi + 1)])) if vi + 1 < len(v_budgets) else 0.0
            row["max_next_delta"] = float(max(float(row["next_k_delta"]), float(row["next_v_delta"])))

    _write_csv(out_dir / "per_head_budget_grid.csv", per_head_grid_rows)
    _write_csv(out_dir / "layer_uniform_budget_grid.csv", layer_grid_rows)
    _write_csv(out_dir / "policy_path_per_head.csv", policy_path_rows)

    accepted = [r for r in policy_path_rows if bool(r.get("accepted"))]
    grid_stats = _monotonicity_stats(per_head_grid_rows)
    summary = {
        "elapsed_seconds": float(time.perf_counter() - t0),
        "qkv_trace": str(args.qkv_trace),
        "x_trace": str(args.x_trace),
        "decode_lengths": [int(trace.decode_tokens_for_qidx(int(qidx))) for qidx in q_indices],
        "heads": [int(h) for h in heads],
        "k_budgets": [int(k) for k in k_budgets],
        "v_budgets": [int(v) for v in v_budgets],
        "policy_k_budgets": [int(k) for k in policy_k_budgets],
        "policy_v_budgets": [int(v) for v in policy_v_budgets],
        "policy": str(args.policy),
        "threshold": float(args.threshold),
        "per_head_grid_rows": int(len(per_head_grid_rows)),
        "layer_grid_rows": int(len(layer_grid_rows)),
        "policy_path_rows": int(len(policy_path_rows)),
        "accepted_count": int(len(accepted)),
        "accepted_mean_step_MB_per_head": float(np.mean([_float(r["step_MB_per_head"]) for r in accepted])) if accepted else float("nan"),
        "accepted_mean_relL2_to_dense": float(np.mean([_float(r["relL2_to_dense"]) for r in accepted])) if accepted else float("nan"),
        "accepted_max_relL2_to_dense": float(np.max([_float(r["relL2_to_dense"]) for r in accepted])) if accepted else float("nan"),
        "accepted_mean_delta_to_max_budget": float(np.mean([_float(r["delta_to_max_budget"]) for r in accepted])) if accepted else float("nan"),
        "accepted_max_delta_to_max_budget": float(np.max([_float(r["delta_to_max_budget"]) for r in accepted])) if accepted else float("nan"),
        "k_relL2_nonincreasing_fraction": float(grid_stats["k_relL2_nonincreasing_fraction"]),
        "v_relL2_nonincreasing_fraction": float(grid_stats["v_relL2_nonincreasing_fraction"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    run()
