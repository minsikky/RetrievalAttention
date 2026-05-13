#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.costs.base import CostTrace, kv_read_bytes
from benchmark.selector_eval.data.trace import attention_probs, load_trace, static_tokens
from benchmark.selector_eval.metrics.attention import compute_metrics
from benchmark.selector_eval.metrics.tail_estimators import tail_estimate_from_name, tail_output_metrics
from benchmark.selector_eval.selectors.base import QueryState
from benchmark.selector_eval.selectors.oracle import TopMassOracleSelector, selector_from_name


PRESETS = {
    "gated_paged_pq_2048_g512_t098": {
        "selectors": "gated_paged_pq",
        "targets": "0.98",
        "static_prefix": 128,
        "static_suffix": 128,
        "paged_pq_page_size": 2048,
        "paged_router_max_groups": 512,
        "paged_router_merge_rel": 0.05,
        "paged_nprobes": "1,2,4,8,16,32,64,128,256,512",
    },
    "snapshot_all_t098": {
        "selectors": "top_mass_oracle,retroinfer_snapshot,pqcache_full_scan_snapshot,paged_local_pq_snapshot,gated_paged_pq_snapshot,sparq_r16,magicpig,retrievalattention_graph",
        "targets": "0.98",
        "static_prefix": 128,
        "static_suffix": 128,
        "paged_pq_page_size": 2048,
        "paged_router_max_groups": 512,
        "paged_router_merge_rel": 0.05,
        "paged_nprobes": "1,2,4,8,16,32,64,128,256,512",
    },
    "online_all_t098": {
        "selectors": "retroinfer_online_proxy,pqcache_full_scan_online,gated_paged_pq_online,paged_local_pq_online,ivfpq_periodic_rebuild,sparq_r16",
        "targets": "0.98",
        "static_prefix": 128,
        "static_suffix": 128,
        "paged_pq_page_size": 2048,
        "paged_router_max_groups": 512,
        "paged_router_merge_rel": 0.05,
        "paged_nprobes": "1,2,4,8,16,32,64,128,256,512",
    },
}


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def parse_csv_names(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=sorted(PRESETS), default="")
    preset_args, _unknown = pre.parse_known_args()

    parser = argparse.ArgumentParser(description="Unified selector evaluation runner.", parents=[pre])
    if preset_args.preset:
        parser.set_defaults(**PRESETS[preset_args.preset])
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selectors", default="dense,top_mass_oracle")
    parser.add_argument("--decode_lengths", required=True)
    parser.add_argument("--targets", default="0.95,0.98")
    parser.add_argument(
        "--tail_estimators",
        default="",
        help=(
            "Optional comma-separated output tail estimators. Examples: "
            "uniform_tail_s1024_seed0,oracle_prob_tail_s1024_seed0. "
            "Rows are emitted as '<selector>+<tail_estimator>'."
        ),
    )
    parser.add_argument("--heads", default="", help="Comma-separated heads. Empty means all heads.")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--retroinfer_cluster_size", type=int, default=256)
    parser.add_argument("--paged_pq_page_size", type=int, default=2048)
    parser.add_argument("--paged_pq_subvecs", type=int, default=2)
    parser.add_argument("--paged_pq_subbits", type=int, default=6)
    parser.add_argument("--paged_pq_kmeans_iters", type=int, default=3)
    parser.add_argument("--paged_pq_permutation", default="none")
    parser.add_argument("--value_pq_subvecs", type=int, default=0)
    parser.add_argument("--value_pq_subbits", type=int, default=0)
    parser.add_argument("--paged_router_max_groups", type=int, default=512)
    parser.add_argument("--paged_router_merge_rel", type=float, default=0.05)
    parser.add_argument("--paged_nprobes", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--ivfpq_nprobes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--ivfpq_coarse_clusters", type=int, default=128)
    parser.add_argument("--ivfpq_rebuild_interval", type=int, default=8192)
    parser.add_argument("--sparq_rank", type=int, default=16)
    parser.add_argument("--magicpig_bits", type=int, default=10)
    parser.add_argument("--magicpig_tables", type=int, default=150)
    parser.add_argument("--magicpig_min_collisions", type=int, default=2)
    parser.add_argument("--ra_provenance_topk", type=int, default=64)
    parser.add_argument("--ra_connect_window", type=int, default=8)
    parser.add_argument("--ra_degree", type=int, default=32)
    parser.add_argument("--ra_seed_count", type=int, default=64)
    parser.add_argument("--ra_max_visits", type=int, default=2048)
    parser.add_argument("--ra_min_visits", type=int, default=64)
    args = parser.parse_args()
    if args.preset:
        for key, value in PRESETS[args.preset].items():
            setattr(args, key, value)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    trace = load_trace(args.trace)
    decode_lengths = parse_csv_ints(args.decode_lengths)
    targets = parse_csv_floats(args.targets)
    paged_kwargs = {
        "static_prefix": int(args.static_prefix),
        "static_suffix": int(args.static_suffix),
        "page_size": int(args.paged_pq_page_size),
        "router_max_groups": int(args.paged_router_max_groups),
        "router_merge_rel": float(args.paged_router_merge_rel),
        "nprobes": tuple(parse_csv_ints(args.paged_nprobes)),
        "subvecs": int(args.paged_pq_subvecs),
        "subbits": int(args.paged_pq_subbits),
        "kmeans_iters": int(args.paged_pq_kmeans_iters),
        "pq_permutation": str(args.paged_pq_permutation),
        "value_pq_subvecs": int(args.value_pq_subvecs),
        "value_pq_subbits": int(args.value_pq_subbits),
        "attn_key_bytes": int(args.key_bytes),
        "value_bytes": int(args.value_bytes),
    }
    ivfpq_kwargs = {
        "static_prefix": int(args.static_prefix),
        "static_suffix": int(args.static_suffix),
        "nprobes": tuple(parse_csv_ints(args.ivfpq_nprobes)),
        "coarse_clusters": int(args.ivfpq_coarse_clusters),
        "rebuild_interval": int(args.ivfpq_rebuild_interval),
        "attn_key_bytes": int(args.key_bytes),
        "value_bytes": int(args.value_bytes),
    }
    retroinfer_kwargs = {
        "cluster_size": int(args.retroinfer_cluster_size),
        "static_prefix": int(args.static_prefix),
        "static_suffix": int(args.static_suffix),
        "score_key_bytes": int(args.key_bytes),
        "attn_key_bytes": int(args.key_bytes),
        "value_bytes": int(args.value_bytes),
        "input_len": int(trace.input_len),
    }
    sparq_kwargs = {"rank": int(args.sparq_rank), "score_key_bytes": int(args.key_bytes)}
    magicpig_kwargs = {
        "bits": int(args.magicpig_bits),
        "tables": int(args.magicpig_tables),
        "min_collisions": int(args.magicpig_min_collisions),
        "score_key_bytes": int(args.key_bytes),
    }
    retrievalattention_kwargs = {
        "static_prefix": int(args.static_prefix),
        "static_suffix": int(args.static_suffix),
        "provenance_topk": int(args.ra_provenance_topk),
        "connect_window": int(args.ra_connect_window),
        "degree": int(args.ra_degree),
        "seed_count": int(args.ra_seed_count),
        "max_visits": int(args.ra_max_visits),
        "min_visits": int(args.ra_min_visits),
        "score_key_bytes": int(args.key_bytes),
    }
    selectors = [
        selector_from_name(
            name,
            trace=trace,
            retroinfer_kwargs=retroinfer_kwargs,
            paged_kwargs=paged_kwargs,
            ivfpq_kwargs=ivfpq_kwargs,
            sparq_kwargs=sparq_kwargs,
            magicpig_kwargs=magicpig_kwargs,
            retrievalattention_kwargs=retrievalattention_kwargs,
            key_bytes=int(args.key_bytes),
        )
        for name in parse_csv_names(args.selectors)
    ]
    heads = parse_csv_ints(args.heads) if str(args.heads).strip() else list(range(trace.num_heads))
    tail_estimators = parse_csv_names(args.tail_estimators)
    q_indices = trace.q_indices_for_decodes(decode_lengths)
    oracle = TopMassOracleSelector()

    rows = []
    for qidx in q_indices:
        position = int(trace.positions[int(qidx)])
        decode_tokens = trace.decode_tokens_for_qidx(int(qidx))
        for head in heads:
            kv_head = trace.kv_head_for(int(head))
            keys = trace.keys[kv_head, : position + 1].astype(np.float32, copy=False)
            values = trace.values[kv_head, : position + 1].astype(np.float32, copy=False)
            query = trace.queries[int(head), int(qidx)].astype(np.float32, copy=False)
            scores, probs = attention_probs(keys, query)
            base = static_tokens(position, args.static_prefix, args.static_suffix)
            state = QueryState(
                decode_tokens=decode_tokens,
                position=position,
                qidx=int(qidx),
                head=int(head),
                kv_head=int(kv_head),
                query=query,
                keys=keys,
                values=values,
                scores=scores,
                probs=probs,
                base_tokens=base,
            )

            for target in targets:
                oracle_result = oracle.select(state, target_mass=target)
                oracle_tokens = oracle_result.selected_tokens
                for selector in selectors:
                    result = selector.select(state, target_mass=target)
                    exact_cost = CostTrace()
                    exact_cost.read(
                        "exact_attention",
                        "exact_kv",
                        kv_read_bytes(len(result.selected_tokens), trace.head_dim, args.key_bytes, args.value_bytes),
                    )
                    total_cost = CostTrace()
                    total_cost.extend(result.cost)
                    total_cost.extend(exact_cost)
                    metrics = compute_metrics(state, result, oracle_tokens)
                    metric_aliases = {
                        "FN_mass": metrics["false_negative_mass"],
                        "FP_mass": metrics["false_positive_mass"],
                        "output_relative_L2": metrics["output_relative_l2"],
                        "output_rmsnorm_relative_L2": metrics["output_rmsnorm_relative_l2"],
                        "distribution_JS": metrics["distribution_js"],
                    }
                    has_online_update = result.cost.mb(phase="online_update") > 0.0
                    accounting_mode = result.metadata.get(
                        "accounting_mode",
                        "online_proxy" if has_online_update else "snapshot",
                    )
                    online_update_cumulative_mb = float(
                        result.metadata.get(
                            "online_update_cumulative_MB",
                            result.cost.mb(phase="online_update") if has_online_update else 0.0,
                        )
                    )
                    online_update_indexed_tokens = int(result.metadata.get("online_update_indexed_tokens", 0))
                    online_update_mb_per_token = online_update_cumulative_mb / max(1, int(decode_tokens))
                    selector_mb_per_query = result.cost.mb(phase="selector")
                    exact_kv_mb_per_query = exact_cost.mb(phase="exact_attention")
                    tail_estimator_mb_per_query = 0.0
                    step_mb_per_query = (
                        result.cost.mb(phase="selector")
                        + exact_cost.mb(phase="exact_attention")
                        + online_update_mb_per_token
                        + tail_estimator_mb_per_query
                    )
                    row = {
                        "algorithm": result.algorithm,
                        "accounting_mode": accounting_mode,
                        "online_update_modeled": bool(result.metadata.get("online_update_modeled", has_online_update)),
                        "decode_length": int(decode_tokens),
                        "target_mass": float(target),
                        "qidx": int(qidx),
                        "head": int(head),
                        "kv_head": int(kv_head),
                        "selected_tokens": int(len(result.selected_tokens)),
                        "candidate_tokens": int(len(result.candidate_tokens)),
                        "selector_MB": selector_mb_per_query,
                        "selector_MB_per_query": selector_mb_per_query,
                        "exact_KV_MB": exact_kv_mb_per_query,
                        "exact_KV_MB_per_query": exact_kv_mb_per_query,
                        "tail_estimator": "none",
                        "tail_estimator_MB": tail_estimator_mb_per_query,
                        "tail_estimator_MB_per_query": tail_estimator_mb_per_query,
                        "tail_samples": 0,
                        "tail_population": 0,
                        "tail_estimator_variance": 0.0,
                        "tail_oracle_diagnostic": False,
                        "online_update_MB": online_update_cumulative_mb,
                        "online_update_cumulative_MB": online_update_cumulative_mb,
                        "online_update_indexed_tokens": online_update_indexed_tokens,
                        "online_update_amortized_MB": online_update_mb_per_token,
                        "online_update_MB_per_token": online_update_mb_per_token,
                        "query_MB": selector_mb_per_query + exact_kv_mb_per_query,
                        "query_MB_per_query": selector_mb_per_query + exact_kv_mb_per_query,
                        "step_MB": step_mb_per_query,
                        "step_MB_per_query": step_mb_per_query,
                        "total_MB": step_mb_per_query,
                        "total_MB_per_query": step_mb_per_query,
                        **metrics,
                        **metric_aliases,
                    }
                    rows.append(row)

                    for tail_name in tail_estimators:
                        estimate = tail_estimate_from_name(
                            tail_name,
                            state,
                            result,
                            key_bytes=int(args.key_bytes),
                            value_bytes=int(args.value_bytes),
                        )
                        replaces_exact_attention = bool(estimate.metadata.get("replaces_exact_attention", False))
                        tail_metrics = dict(metrics)
                        tail_metrics.update(tail_output_metrics(state, estimate))
                        tail_metric_aliases = {
                            "FN_mass": tail_metrics["false_negative_mass"],
                            "FP_mass": tail_metrics["false_positive_mass"],
                            "output_relative_L2": tail_metrics["output_relative_l2"],
                            "output_rmsnorm_relative_L2": tail_metrics["output_rmsnorm_relative_l2"],
                            "distribution_JS": tail_metrics["distribution_js"],
                        }
                        tail_exact_kv_mb_per_query = (
                            estimate.cost.mb(phase="exact_attention") if replaces_exact_attention else exact_kv_mb_per_query
                        )
                        tail_mb_per_query = estimate.cost.mb(phase="tail_estimator")
                        tail_step_mb_per_query = (
                            result.cost.mb(phase="selector")
                            + tail_exact_kv_mb_per_query
                            + online_update_mb_per_token
                            + tail_mb_per_query
                        )
                        tail_row = {
                            "algorithm": f"{result.algorithm}+{estimate.name}",
                            "base_algorithm": result.algorithm,
                            "accounting_mode": accounting_mode,
                            "online_update_modeled": bool(result.metadata.get("online_update_modeled", has_online_update)),
                            "decode_length": int(decode_tokens),
                            "target_mass": float(target),
                            "qidx": int(qidx),
                            "head": int(head),
                            "kv_head": int(kv_head),
                            "selected_tokens": int(len(result.selected_tokens)),
                            "candidate_tokens": int(len(result.candidate_tokens)),
                            "selector_MB": selector_mb_per_query,
                            "selector_MB_per_query": selector_mb_per_query,
                            "exact_KV_MB": tail_exact_kv_mb_per_query,
                            "exact_KV_MB_per_query": tail_exact_kv_mb_per_query,
                            "tail_estimator": estimate.name,
                            "tail_estimator_MB": tail_mb_per_query,
                            "tail_estimator_MB_per_query": tail_mb_per_query,
                            "tail_samples": int(estimate.metadata.get("tail_samples", 0)),
                            "tail_population": int(estimate.metadata.get("tail_population", 0)),
                            "tail_estimator_variance": float(estimate.metadata.get("tail_estimator_variance", 0.0)),
                            "tail_oracle_diagnostic": bool(estimate.metadata.get("oracle_diagnostic", False)),
                            "tail_replaces_exact_attention": replaces_exact_attention,
                            "online_update_MB": online_update_cumulative_mb,
                            "online_update_cumulative_MB": online_update_cumulative_mb,
                            "online_update_indexed_tokens": online_update_indexed_tokens,
                            "online_update_amortized_MB": online_update_mb_per_token,
                            "online_update_MB_per_token": online_update_mb_per_token,
                            "query_MB": selector_mb_per_query + tail_exact_kv_mb_per_query + tail_mb_per_query,
                            "query_MB_per_query": selector_mb_per_query + tail_exact_kv_mb_per_query + tail_mb_per_query,
                            "step_MB": tail_step_mb_per_query,
                            "step_MB_per_query": tail_step_mb_per_query,
                            "total_MB": tail_step_mb_per_query,
                            "total_MB_per_query": tail_step_mb_per_query,
                            **tail_metrics,
                            **tail_metric_aliases,
                        }
                        rows.append(tail_row)

    write_csv(out_dir / "samples.csv", rows)
    (out_dir / "samples.json").write_text(json.dumps(rows, indent=2, sort_keys=True))

    grouped: dict[tuple, list[dict]] = {}
    keys = ("algorithm", "accounting_mode", "online_update_modeled", "decode_length", "target_mass")
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    summary = []
    for key, items in sorted(grouped.items(), key=lambda item: item[0]):
        out = {name: value for name, value in zip(keys, key)}
        out["samples"] = len(items)
        for metric in [
            "selected_tokens",
            "candidate_tokens",
            "attention_mass",
            "false_negative_mass",
            "false_positive_mass",
            "FN_mass",
            "FP_mass",
            "output_cosine",
            "output_relative_l2",
            "output_relative_L2",
            "output_rmsnorm_relative_l2",
            "output_rmsnorm_relative_L2",
            "output_centered_cosine",
            "output_mean_abs_relative_error",
            "output_p95_abs_relative_error",
            "output_p99_abs_relative_error",
            "output_max_abs_relative_error",
            "output_linf_relative",
            "distribution_js",
            "distribution_JS",
            "selector_MB",
            "selector_MB_per_query",
            "exact_KV_MB",
            "exact_KV_MB_per_query",
            "tail_estimator_MB",
            "tail_estimator_MB_per_query",
            "tail_samples",
            "tail_population",
            "tail_estimator_variance",
            "online_update_MB",
            "online_update_cumulative_MB",
            "online_update_amortized_MB",
            "online_update_MB_per_token",
            "online_update_indexed_tokens",
            "query_MB",
            "query_MB_per_query",
            "step_MB",
            "step_MB_per_query",
            "total_MB",
            "total_MB_per_query",
        ]:
            out[f"{metric}_mean"] = float(np.mean([float(item[metric]) for item in items]))
        summary.append(out)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[selector_eval] wrote {out_dir}")


if __name__ == "__main__":
    main()
