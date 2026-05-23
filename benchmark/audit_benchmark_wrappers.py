#!/usr/bin/env python3
"""Statically audit benchmark wrapper defaults before Slurm submission."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


EXPORT_RE = re.compile(r'^export\s+([A-Za-z_][A-Za-z0-9_]*)="\$\{\1:-(.*)\}"\s*$')
ASSIGN_DEFAULT_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)="\$\{\1:-(.*)\}"\s*$')
SBATCH_RE = re.compile(r"^#SBATCH\s+(--[A-Za-z0-9_-]+)=(.*)\s*$")


@dataclass(frozen=True)
class WrapperSpec:
    label: str
    path: Path
    expected_exports: dict[str, str]
    require_sbatch: bool = True


SPECS = [
    WrapperSpec(
        label="frontier_ruler",
        path=Path("scripts/run_frontier_ruler_batched_one.sh"),
        expected_exports={
            "MODE": "pagedpq_batched",
            "SELECTOR_MODE": "fullscan",
            "SELECTOR_BACKEND": "cuda_ext",
            "ONLINE_CONFIDENCE_RULE": "joint_kv_stability",
            "TAIL_MODE": "vpq_value",
            "TAIL_BLEND": "1.0",
            "NATIVE_DECODE_TAIL": "1",
            "SELECTED_VALUE_MODE": "vpq_value",
            "SELECTED_VALUE_EXACT_RULE": "global_residual_risk",
            "SELECTED_VALUE_EXACT_TOP": "0",
            "SELECTED_VALUE_MIN_EXACT_TOP": "0",
            "SELECTED_VALUE_MAX_EXACT_TOP": "0",
            "PREFILL_SELECTOR_BACKEND": "native",
            "PREFILL_ATTENTION_BACKEND": "native",
            "INDEX_BUILD_BACKEND": "torch_gpu",
            "SELECTOR_PQ_JOINT_GQA_BATCHED": "1",
            "SELECTOR_PQ_JOINT_VECTOR_POLICY": "1",
            "SELECTOR_PQ_JOINT_REUSE_MAX_TOPK": "1",
            "SELECTOR_PQ_JOINT_GRID_ARTIFACTS": "1",
            "SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_FAST_AFFINE_SELECTED": "0",
            "SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID": "0",
            "SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR": "0",
            "SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX": "1",
            "SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX": "1",
            "SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID": "1",
            "SELECTOR_PQ_JOINT_NATIVE_LAZY_POLICY": "0",
            "SELECTOR_PQ_JOINT_NATIVE_POLICY": "1",
            "SELECTOR_PQ_JOINT_NATIVE_V_PREFIX": "1",
            "SELECTOR_PQ_JOINT_ALLHEAD_PRECOMPUTE": "1",
            "SELECTOR_PQ_JOINT_ALLHEAD_EXACT_PRECOMPUTE": "0",
            "SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX": "0",
            "SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX": "0",
            "SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT": "0",
            "SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID": "1",
            "SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS": "1",
            "SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS": "0",
            "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL": "0",
            "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID": "0",
            "SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE": "0",
            "SELECTOR_PQ_JOINT_FUSED_RISK_POLICY": "0",
            "SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK": "0",
            "SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT": "1",
            "SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE": "0",
            "SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE": "1",
            "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE": "0",
            "SELECTOR_PQ_JOINT_WALL_PROFILE": "0",
            "SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS": "1",
            "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE": "1",
        },
    ),
    WrapperSpec(
        label="frontier_longbench",
        path=Path("scripts/run_frontier_longbench_v2_one.sh"),
        expected_exports={
            "ATTENTION_MODE": "pagedpq",
            "SELECTOR_MODE": "fullscan",
            "SELECTOR_BACKEND": "cuda_ext",
            "ONLINE_CONFIDENCE_RULE": "joint_kv_stability",
            "TAIL_MODE": "vpq_value",
            "TAIL_BLEND": "1.0",
            "NATIVE_DECODE_TAIL": "1",
            "SELECTED_VALUE_MODE": "vpq_value",
            "SELECTED_VALUE_EXACT_RULE": "global_residual_risk",
            "SELECTED_VALUE_EXACT_TOP": "0",
            "SELECTED_VALUE_MIN_EXACT_TOP": "0",
            "SELECTED_VALUE_MAX_EXACT_TOP": "0",
            "PREFILL_SELECTOR_BACKEND": "native",
            "PREFILL_ATTENTION_BACKEND": "native",
            "INDEX_BUILD_BACKEND": "torch_gpu",
            "SELECTOR_PQ_JOINT_GQA_BATCHED": "1",
            "SELECTOR_PQ_JOINT_VECTOR_POLICY": "1",
            "SELECTOR_PQ_JOINT_REUSE_MAX_TOPK": "1",
            "SELECTOR_PQ_JOINT_GRID_ARTIFACTS": "1",
            "SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_FAST_AFFINE_SELECTED": "0",
            "SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID": "0",
            "SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR": "0",
            "SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX": "1",
            "SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX": "1",
            "SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID": "1",
            "SELECTOR_PQ_JOINT_NATIVE_LAZY_POLICY": "0",
            "SELECTOR_PQ_JOINT_NATIVE_POLICY": "1",
            "SELECTOR_PQ_JOINT_NATIVE_V_PREFIX": "1",
            "SELECTOR_PQ_JOINT_ALLHEAD_PRECOMPUTE": "1",
            "SELECTOR_PQ_JOINT_ALLHEAD_EXACT_PRECOMPUTE": "0",
            "SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX": "0",
            "SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX": "0",
            "SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT": "0",
            "SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID": "1",
            "SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS": "1",
            "SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS": "0",
            "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL": "0",
            "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID": "0",
            "SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE": "0",
            "SELECTOR_PQ_JOINT_FUSED_RISK_POLICY": "0",
            "SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK": "0",
            "SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT": "1",
            "SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE": "0",
            "SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE": "1",
            "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE": "0",
            "SELECTOR_PQ_JOINT_WALL_PROFILE": "0",
            "SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS": "1",
            "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE": "1",
        },
    ),
    WrapperSpec(
        label="dense_ruler",
        path=Path("scripts/run_dense_ruler_batched_one.sh"),
        expected_exports={
            "MODE": "dense_batched",
        },
    ),
    WrapperSpec(
        label="dense_longbench",
        path=Path("scripts/run_dense_longbench_v2_one.sh"),
        expected_exports={
            "ATTENTION_MODE": "dense",
        },
    ),
    WrapperSpec(
        label="direct_longbench_hf",
        path=Path("benchmark/run_longbench_v2_hf.sh"),
        expected_exports={
            "ATTENTION_MODE": "dense",
            "SELECTOR_MODE": "fullscan",
            "SELECTOR_BACKEND": "cuda_ext",
            "ONLINE_CONFIDENCE_RULE": "joint_kv_stability",
            "TAIL_MODE": "vpq_value",
            "SELECTED_VALUE_MODE": "vpq_value",
            "SELECTED_VALUE_EXACT_RULE": "global_residual_risk",
            "FRONTIER_CANONICAL_GPU": "1",
            "INDEX_BUILD_BACKEND": "torch_gpu",
            "SELECTOR_PQ_JOINT_GQA_BATCHED": "1",
            "SELECTOR_PQ_JOINT_GRID_ARTIFACTS": "1",
            "SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_FAST_AFFINE_SELECTED": "0",
            "SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID": "0",
            "SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR": "0",
            "SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX": "1",
            "SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX": "1",
            "SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID": "1",
            "SELECTOR_PQ_JOINT_NATIVE_LAZY_POLICY": "0",
            "SELECTOR_PQ_JOINT_NATIVE_POLICY": "1",
            "SELECTOR_PQ_JOINT_NATIVE_V_PREFIX": "1",
            "SELECTOR_PQ_JOINT_ALLHEAD_EXACT_PRECOMPUTE": "0",
            "SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX": "0",
            "SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX": "0",
            "SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT": "0",
            "SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID": "1",
            "SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS": "1",
            "SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS": "0",
            "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL": "0",
            "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID": "0",
            "SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE": "0",
            "SELECTOR_PQ_JOINT_FUSED_RISK_POLICY": "0",
            "SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK": "0",
            "SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT": "1",
            "SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE": "0",
            "SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE": "1",
            "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE": "0",
            "SELECTOR_PQ_JOINT_WALL_PROFILE": "0",
            "SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS": "1",
            "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE": "1",
        },
        require_sbatch=False,
    ),
    WrapperSpec(
        label="direct_public_longdecode_hf",
        path=Path("benchmark/run_public_longdecode_hf.sh"),
        expected_exports={
            "ATTENTION_MODE": "dense",
            "SELECTOR_MODE": "fullscan",
            "SELECTOR_BACKEND": "cuda_ext",
            "ONLINE_CONFIDENCE_RULE": "joint_kv_stability",
            "TAIL_MODE": "vpq_value",
            "SELECTED_VALUE_MODE": "vpq_value",
            "SELECTED_VALUE_EXACT_RULE": "global_residual_risk",
            "FRONTIER_CANONICAL_GPU": "1",
            "INDEX_BUILD_BACKEND": "torch_gpu",
            "SELECTOR_PQ_JOINT_GQA_BATCHED": "1",
            "SELECTOR_PQ_JOINT_GRID_ARTIFACTS": "1",
            "SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_FAST_AFFINE_SELECTED": "0",
            "SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX": "0",
            "SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID": "0",
            "SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR": "0",
            "SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX": "1",
            "SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX": "1",
            "SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID": "1",
            "SELECTOR_PQ_JOINT_NATIVE_LAZY_POLICY": "0",
            "SELECTOR_PQ_JOINT_NATIVE_POLICY": "1",
            "SELECTOR_PQ_JOINT_NATIVE_V_PREFIX": "1",
            "SELECTOR_PQ_JOINT_ALLHEAD_EXACT_PRECOMPUTE": "0",
            "SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX": "0",
            "SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX": "0",
            "SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT": "0",
            "SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID": "1",
            "SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS": "1",
            "SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS": "0",
            "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL": "0",
            "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID": "0",
            "SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE": "0",
            "SELECTOR_PQ_JOINT_FUSED_RISK_POLICY": "0",
            "SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK": "0",
            "SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT": "1",
            "SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE": "0",
            "SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE": "1",
            "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE": "0",
            "SELECTOR_PQ_JOINT_WALL_PROFILE": "0",
            "SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS": "1",
            "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE": "1",
        },
        require_sbatch=False,
    ),
]

REQUIRED_SBATCH = {
    "--account": "zhengya98",
    "--partition": "spgpu",
    "--gpus-per-node": "1",
}


def _parse_exports(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        match = EXPORT_RE.match(stripped) or ASSIGN_DEFAULT_RE.match(stripped)
        if match:
            out[match.group(1)] = match.group(2)
    return out


def _parse_sbatch(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        match = SBATCH_RE.match(line.strip())
        if match:
            out[match.group(1)] = match.group(2)
    return out


def _audit_spec(spec: WrapperSpec) -> tuple[list[str], dict[str, str]]:
    warnings: list[str] = []
    observed: dict[str, str] = {}
    if not spec.path.exists():
        return [f"missing-wrapper:{spec.path}"], observed

    text = spec.path.read_text(encoding="utf-8")
    exports = _parse_exports(text)
    sbatch = _parse_sbatch(text)

    if spec.require_sbatch:
        for key, expected in REQUIRED_SBATCH.items():
            observed[f"SBATCH {key}"] = sbatch.get(key, "missing")
            if sbatch.get(key) != expected:
                warnings.append(f"{key}={sbatch.get(key, 'missing')} expected {expected}")

    for key, expected in spec.expected_exports.items():
        observed[key] = exports.get(key, "missing")
        if exports.get(key) != expected:
            warnings.append(f"{key}={exports.get(key, 'missing')} expected {expected}")

    return warnings, observed


def _markdown(rows: list[tuple[WrapperSpec, list[str], dict[str, str]]]) -> str:
    lines = [
        "| wrapper | path | status | important defaults |",
        "| --- | --- | --- | --- |",
    ]
    for spec, warnings, observed in rows:
        keys = [
            key
            for key in [
                "MODE",
                "ATTENTION_MODE",
                "SELECTOR_BACKEND",
                "ONLINE_CONFIDENCE_RULE",
                "SELECTED_VALUE_MODE",
                "SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX",
                "SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX",
                "SELECTOR_PQ_JOINT_FAST_AFFINE_SELECTED",
                "SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX",
                "SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID",
                "SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR",
                "SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX",
                "SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX",
                "SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID",
                "SELECTOR_PQ_JOINT_NATIVE_LAZY_POLICY",
                "SELECTOR_PQ_JOINT_NATIVE_POLICY",
                "SELECTOR_PQ_JOINT_NATIVE_V_PREFIX",
                "SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX",
                "SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX",
                "SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT",
                "SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID",
                "SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS",
                "SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS",
                "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL",
                "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID",
                "SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE",
                "SELECTOR_PQ_JOINT_FUSED_RISK_POLICY",
                "SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK",
                "SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT",
                "SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE",
                "SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE",
                "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE",
                "SELECTOR_PQ_JOINT_WALL_PROFILE",
                "SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS",
                "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE",
                "PREFILL_SELECTOR_BACKEND",
                "INDEX_BUILD_BACKEND",
                "SBATCH --partition",
                "SBATCH --account",
            ]
            if key in observed
        ]
        defaults = ", ".join(f"{key}={observed[key]}" for key in keys)
        status = "ok" if not warnings else "; ".join(warnings)
        lines.append(f"| {spec.label} | `{spec.path}` | {status} | {defaults} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path")
    args = parser.parse_args()

    rows = [(spec, *_audit_spec(spec)) for spec in SPECS]
    text = _markdown(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")

    bad = [(spec, warnings) for spec, warnings, _ in rows if warnings]
    if bad:
        print("\nWrapper audit warnings:")
        for spec, warnings in bad:
            print(f"- {spec.label}: {'; '.join(warnings)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
