"""Canonical frontier configuration shared by wrappers, audits, and runners."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _truthy(value: object, default: str = "0") -> bool:
    text = str(default if value is None else value).strip().lower()
    return text in {"1", "true", "yes", "on"}


CANONICAL_ARG_DEFAULTS: dict[str, str] = {
    "selector_mode": "fullscan",
    "selector_backend": "cuda_ext",
    "index_build_backend": "torch_gpu",
    "online_confidence_rule": "joint_kv_stability",
    "tail_mode": "vpq_value",
    "tail_score_calibration": "none",
    "selected_value_mode": "vpq_value",
    "selected_value_exact_rule": "global_residual_risk",
    "ranked_confidence_cost_mode": "exact",
    "joint_kv_k_budget_fracs": "0.10,0.30,0.50,0.70,0.90,1.0",
    "joint_kv_v_budget_fracs": "0.05,0.10,0.20,0.40,0.60,0.80,1.0",
    # joint_kv_stability_threshold is intentionally NOT pinned here: tau is a
    # tuning parameter of the canonical algorithm, not an algorithm-semantics
    # switch. It is validated as finite and positive below so threshold sweeps
    # (e.g. the 2026-07-06 tau sweep) run on the canonical GPU path.
    "joint_kv_threshold_mode": "budget_delta_frac",
    "joint_kv_threshold_reference_frac": "0.2",
    "joint_kv_threshold_scale_shape": "sqrt",
    "joint_kv_threshold_min_scale": "0.0",
    "joint_kv_threshold_max_scale": "1.5",
    "joint_kv_start_strategy": "proxy_mass_m0p9",
}


CANONICAL_ENV_DEFAULTS: dict[str, str] = {
    "SELECTOR_PQ_JOINT_VPQ_CACHE": "1",
    "SELECTOR_PQ_JOINT_FP32_PROBS": "1",
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
    "SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS": "0",
    "SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS_BACKEND": "cublas_t",
    "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID": "0",
    "SELECTOR_PQ_JOINT_SPARSE_DIRECT_SCORE_GRID": "0",
    "SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX": "1",
    "SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX": "0",
    "SELECTOR_PQ_JOINT_NATIVE_BUDGET_PREFIX": "0",
    "SELECTOR_PQ_JOINT_RANK_PREFIX_WORKSPACE": "0",
    "SELECTOR_PQ_JOINT_SELECTOR_TOPK_PREFIX": "0",
    "SELECTOR_PQ_JOINT_UNSORTED_K_PREFIX": "0",
    "SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT": "0",
    "SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID": "1",
    "SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS": "1",
    "SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS": "0",
    "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL": "0",
    "SELECTOR_PQ_JOINT_SCORE_GRID_WORKSPACE": "0",
    "SELECTOR_PQ_JOINT_GROUPED_SCORE_WORKSPACE": "0",
    "SELECTOR_PQ_JOINT_NOCALIB_SCORE_GRID_WORKSPACE": "0",
    "SELECTOR_PQ_JOINT_NOCALIB_SCATTER_SCORE_GRID": "0",
    "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID": "0",
    "SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE": "1",
    "SELECTOR_PQ_JOINT_GROUPED_OUTPUT_WORKSPACE": "0",
    "SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE": "0",
    "SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE_GROW_PAD": "1024",
    "SELECTOR_PQ_JOINT_FUSED_RISK_POLICY": "0",
    "SELECTOR_PQ_JOINT_FUSED_MIXED_POLICY": "0",
    "SELECTOR_PQ_JOINT_INTERVAL_RISK_POLICY": "0",
    "SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK": "0",
    "SELECTOR_PQ_JOINT_STAGED_KV_PREFIX": "0",
    "SELECTOR_PQ_JOINT_STAGED_KV_K_STEPS": "2",
    "SELECTOR_PQ_JOINT_STAGED_KV_V_STEPS": "3",
    "SELECTOR_PQ_JOINT_STAGED_RISK_PREFIX": "0",
    "SELECTOR_PQ_JOINT_STAGED_RISK_PREFIX_TOPK": "0",
    "SELECTOR_PQ_JOINT_STAGED_RISK_PREFIX_V_STEPS": "3",
    "SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX": "0",
    "SELECTOR_PQ_JOINT_SCORE_DIRECT_INTERVAL_POLICY": "0",
    "SELECTOR_PQ_JOINT_SCORE_PROB_INTERVAL_POLICY": "0",
    "SELECTOR_PQ_JOINT_SCORE_DIRECT_TOPK_INTERVAL_POLICY": "0",
    "SELECTOR_PQ_JOINT_SCORE_DIRECT_WORKSPACE": "0",
    "SELECTOR_PQ_JOINT_MERGE_RISK_POLICY": "0",
    "SELECTOR_PQ_JOINT_MERGE_RISK_PREFIX": "0",
    "SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE": "0",
    "SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT": "1",
    "SELECTOR_PQ_JOINT_COMPACT_VPQ_RISK_PREFIX": "1",
    "SELECTOR_PQ_JOINT_MEMORY_BOUNDED_VPQ": "1",
    "SELECTOR_PQ_JOINT_MEMORY_TRACE": "0",
    "SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE": "1",
    "SELECTOR_PQ_JOINT_NATIVE_VPQ_APPEND": "1",
    "SELECTOR_PQ_JOINT_NATIVE_VPQ_SIDECAR": "0",
    "SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE": "1",
    "SELECTOR_PQ_JOINT_SOFTMAX_BASE_WORKSPACE": "0",
    "SELECTOR_PQ_JOINT_SOFTMAX_BASE_CUBLAS": "0",
    "SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE": "0",
    "SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS": "0",
    "SELECTOR_PQ_JOINT_NATIVE_PQ_SCALE_IN_KERNEL": "0",
    "SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING": "1",
    "SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING_ACCUMULATE": "0",
    "SELECTOR_PQ_JOINT_FUSED_POLICY_ACCOUNTING": "0",
    "SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING_VERIFY": "0",
    "SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID": "0",
    "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE": "0",
    "SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE": "0",
    "SELECTOR_PQ_JOINT_WALL_PROFILE": "0",
    "SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS": "1",
    "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE": "1",
}


REQUIRED_CANONICAL_ENV_FLAGS: tuple[str, ...] = (
    "SELECTOR_PQ_JOINT_GQA_BATCHED",
    "SELECTOR_PQ_JOINT_VECTOR_POLICY",
    "SELECTOR_PQ_JOINT_REUSE_MAX_TOPK",
    "SELECTOR_PQ_JOINT_GRID_ARTIFACTS",
    "SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX",
    "SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX",
    "SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID",
    "SELECTOR_PQ_JOINT_NATIVE_POLICY",
    "SELECTOR_PQ_JOINT_NATIVE_V_PREFIX",
    "SELECTOR_PQ_JOINT_ALLHEAD_PRECOMPUTE",
    "SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID",
    "SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS",
    "SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE",
    "SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT",
    "SELECTOR_PQ_JOINT_COMPACT_VPQ_RISK_PREFIX",
    "SELECTOR_PQ_JOINT_MEMORY_BOUNDED_VPQ",
    "SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE",
    "SELECTOR_PQ_JOINT_NATIVE_VPQ_APPEND",
    "SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE",
    "SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING",
    "SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS",
    "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE",
)


DISALLOWED_DIAGNOSTIC_ENV_FLAGS: tuple[str, ...] = (
    "SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX",
    "SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX",
    "SELECTOR_PQ_JOINT_FAST_AFFINE_SELECTED",
    "SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX",
    "SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID",
    "SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR",
    "SELECTOR_PQ_JOINT_NATIVE_LAZY_POLICY",
    "SELECTOR_PQ_JOINT_ALLHEAD_EXACT_PRECOMPUTE",
    "SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS",
    "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID",
    "SELECTOR_PQ_JOINT_SPARSE_DIRECT_SCORE_GRID",
    "SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX",
    "SELECTOR_PQ_JOINT_NATIVE_BUDGET_PREFIX",
    "SELECTOR_PQ_JOINT_RANK_PREFIX_WORKSPACE",
    "SELECTOR_PQ_JOINT_SELECTOR_TOPK_PREFIX",
    "SELECTOR_PQ_JOINT_UNSORTED_K_PREFIX",
    "SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT",
    "SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS",
    "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL",
    "SELECTOR_PQ_JOINT_SCORE_GRID_WORKSPACE",
    "SELECTOR_PQ_JOINT_GROUPED_SCORE_WORKSPACE",
    "SELECTOR_PQ_JOINT_NOCALIB_SCORE_GRID_WORKSPACE",
    "SELECTOR_PQ_JOINT_NOCALIB_SCATTER_SCORE_GRID",
    "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID",
    "SELECTOR_PQ_JOINT_GROUPED_OUTPUT_WORKSPACE",
    "SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE",
    "SELECTOR_PQ_JOINT_FUSED_RISK_POLICY",
    "SELECTOR_PQ_JOINT_FUSED_MIXED_POLICY",
    "SELECTOR_PQ_JOINT_INTERVAL_RISK_POLICY",
    "SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK",
    "SELECTOR_PQ_JOINT_STAGED_KV_PREFIX",
    "SELECTOR_PQ_JOINT_STAGED_RISK_PREFIX",
    "SELECTOR_PQ_JOINT_STAGED_RISK_PREFIX_TOPK",
    "SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX",
    "SELECTOR_PQ_JOINT_SCORE_DIRECT_INTERVAL_POLICY",
    "SELECTOR_PQ_JOINT_SCORE_PROB_INTERVAL_POLICY",
    "SELECTOR_PQ_JOINT_SCORE_DIRECT_TOPK_INTERVAL_POLICY",
    "SELECTOR_PQ_JOINT_SCORE_DIRECT_WORKSPACE",
    "SELECTOR_PQ_JOINT_MERGE_RISK_POLICY",
    "SELECTOR_PQ_JOINT_MERGE_RISK_PREFIX",
    "SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE",
    "SELECTOR_PQ_JOINT_NATIVE_VPQ_SIDECAR",
    "SELECTOR_PQ_JOINT_SOFTMAX_BASE_WORKSPACE",
    "SELECTOR_PQ_JOINT_SOFTMAX_BASE_CUBLAS",
    "SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE",
    "SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS",
    "SELECTOR_PQ_JOINT_NATIVE_PQ_SCALE_IN_KERNEL",
    "SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING_VERIFY",
    "SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING_ACCUMULATE",
    "SELECTOR_PQ_JOINT_FUSED_POLICY_ACCOUNTING",
    "SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID",
    "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE",
    "SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE",
)


BASE_FRONTIER_EXPORTS: dict[str, str] = {
    "SELECTOR_MODE": "fullscan",
    "SELECTOR_BACKEND": "cuda_ext",
    "ONLINE_CONFIDENCE_RULE": "joint_kv_stability",
    "TAIL_MODE": "vpq_value",
    "TAIL_SCORE_CALIBRATION": "none",
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
}


DIRECT_RUNTIME_DEFAULTS: dict[str, str] = {
    "SELECTOR_PQ_PRECOMPUTE_RANK_WEIGHTS": "1",
    "SELECTOR_PQ_GEOMETRIC_THREADS": "512",
    "SELECTOR_PQ_SELECTED_CODEWEIGHT_DELTAS": "1",
    "SELECTOR_PQ_SELECTED_EXACT_LISTS": "1",
    "SELECTOR_PQ_THRESHOLD_NATIVE_TOPK": "1",
    "SELECTOR_PQ_THRESHOLD_TOPK": "16384",
    "ENABLE_FUSED_GEOMETRIC_OUTPUT": "1",
    "SELECTOR_PQ_FUSED_DIM_SCAN_OUTPUT": "1",
    "GEOMETRIC_MIN_BUDGET": "4096",
    "GEOMETRIC_MAX_BUDGET": "65536",
    "GEOMETRIC_GROWTH": "1.5",
    "GEOMETRIC_PROBE_SCALE": "1.5",
    "GEOMETRIC_BUDGET_GRANULARITY": "1024",
    "JOINT_KV_POLICY": "k_first_alternating",
    "JOINT_KV_K_BUDGETS": "4096,8192,14336,32768",
    "JOINT_KV_V_BUDGETS": "1024,2048,4096,6144,8192,12288,16384",
    "JOINT_KV_K_BUDGET_FRACS": "0.10,0.30,0.50,0.70,0.90,1.0",
    "JOINT_KV_V_BUDGET_FRACS": "0.05,0.10,0.20,0.40,0.60,0.80,1.0",
    "JOINT_KV_STABILITY_THRESHOLD": "0.002",
    "JOINT_KV_THRESHOLD_MODE": "budget_delta_frac",
    "JOINT_KV_THRESHOLD_REFERENCE_FRAC": "0.2",
    "JOINT_KV_THRESHOLD_SCALE_SHAPE": "sqrt",
    "JOINT_KV_THRESHOLD_MIN_SCALE": "0.0",
    "JOINT_KV_THRESHOLD_MAX_SCALE": "1.5",
    "JOINT_KV_START_STRATEGY": "proxy_mass_m0p9",
    "SELECTED_VALUE_EXACT_MASS": "0.0",
    "SELECTED_VALUE_EXACT_RISK_MASS": "0.0",
    "SELECTED_VALUE_EXACT_ALL_CONTEXT_MAX": "0",
    "SELECTED_VALUE_EXACT_ALL_FRACTION_MIN": "0.0",
    "VALUE_CODE_STAT_BYTES": "2",
    "TAIL_BLEND": "1.0",
    "PREFILL_TAIL_BLEND": "",
    "DECODE_TAIL_BLEND": "",
    "PAGE_SIZE": "5632",
    "PREFILL_CHUNK_SIZE": "0",
    "PREFILL_SELECTOR_PAGE_BLOCK_SIZE": "0",
    "PREFILL_RANK_BUFFER_LIMIT_MB": "4096",
    "PREFILL_TAIL_SCORE_REUSE": "1",
    "SUBVECS": "4",
    "SUBBITS": "8",
    "VALUE_SUBVECS": "1",
    "VALUE_SUBBITS": "4",
    "VALUE_PQ_GROUP_PAGES": "1",
    "KMEANS_ITERS": "3",
    "NPROBES": "16,32,64,128,256,512",
    "DIAGNOSE_DENSE_REFERENCE": "0",
    "PROFILE_NATIVE_OPS": "0",
    "DISABLE_COST_STATS": "0",
    "DISABLE_NATIVE_DECODE_FUSED": "1",
    "ENABLE_NATIVE_DECODE_FUSED": "0",
    "NATIVE_DECODE_SCORELESS_FUSED": "0",
    "NATIVE_DECODE_SCORELESS_FORCE_MODE": "2",
}


@dataclass(frozen=True)
class CanonicalWrapperSpec:
    label: str
    path: Path
    expected_exports: dict[str, str]
    require_sbatch: bool = True


def canonical_frontier_exports(*, mode_key: str, mode_value: str, include_frontier_flag: bool = False) -> dict[str, str]:
    exports = {mode_key: mode_value}
    exports.update(BASE_FRONTIER_EXPORTS)
    if include_frontier_flag:
        exports["FRONTIER_CANONICAL_GPU"] = "1"
    exports.update(CANONICAL_ENV_DEFAULTS)
    return exports


def canonical_wrapper_specs() -> list[CanonicalWrapperSpec]:
    return [
        CanonicalWrapperSpec(
            label="frontier_ruler",
            path=Path("scripts/run_frontier_ruler_batched_one.sh"),
            expected_exports=canonical_frontier_exports(mode_key="MODE", mode_value="pagedpq_batched"),
        ),
        CanonicalWrapperSpec(
            label="frontier_longbench",
            path=Path("scripts/run_frontier_longbench_v2_one.sh"),
            expected_exports=canonical_frontier_exports(mode_key="ATTENTION_MODE", mode_value="pagedpq"),
        ),
        CanonicalWrapperSpec(
            label="dense_ruler",
            path=Path("scripts/run_dense_ruler_batched_one.sh"),
            expected_exports={"MODE": "dense_batched"},
        ),
        CanonicalWrapperSpec(
            label="dense_longbench",
            path=Path("scripts/run_dense_longbench_v2_one.sh"),
            expected_exports={"ATTENTION_MODE": "dense"},
        ),
        CanonicalWrapperSpec(
            label="direct_longbench_hf",
            path=Path("benchmark/run_longbench_v2_hf.sh"),
            expected_exports=canonical_frontier_exports(
                mode_key="ATTENTION_MODE",
                mode_value="dense",
                include_frontier_flag=True,
            ),
            require_sbatch=False,
        ),
        CanonicalWrapperSpec(
            label="direct_public_longdecode_hf",
            path=Path("benchmark/run_public_longdecode_hf.sh"),
            expected_exports=canonical_frontier_exports(
                mode_key="ATTENTION_MODE",
                mode_value="dense",
                include_frontier_flag=True,
            ),
            require_sbatch=False,
        ),
    ]


def canonical_gpu_frontier_mismatches(args: object, env: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    mismatches: list[str] = []
    for attr, expected in CANONICAL_ARG_DEFAULTS.items():
        actual = str(getattr(args, attr, ""))
        if attr == "selector_backend":
            if actual not in {"cuda_ext", "auto"}:
                mismatches.append("selector_backend must be cuda_ext or auto")
            continue
        if actual != expected:
            mismatches.append(f"{attr} must be {expected}")
    if bool(getattr(args, "approx_prefill", False)):
        mismatches.append("approx_prefill must be disabled; canonical frontier is decode-only")

    for name in REQUIRED_CANONICAL_ENV_FLAGS:
        if not _truthy(env.get(name), CANONICAL_ENV_DEFAULTS.get(name, "0")):
            mismatches.append(f"{name} must be enabled for canonical frontier")
    for name in DISALLOWED_DIAGNOSTIC_ENV_FLAGS:
        if _truthy(env.get(name), CANONICAL_ENV_DEFAULTS.get(name, "0")):
            mismatches.append(f"{name} must be disabled for canonical frontier")

    threshold = float(getattr(args, "joint_kv_stability_threshold", float("inf")))
    if not math.isfinite(threshold) or threshold <= 0.0:
        mismatches.append("joint_kv_stability_threshold must be finite and positive")
    return mismatches


def canonical_env_export_lines() -> list[str]:
    return [f'export {key}="${{{key}:-{value}}}"' for key, value in CANONICAL_ENV_DEFAULTS.items()]


def direct_runtime_export_lines() -> list[str]:
    lines = [f'export {key}="${{{key}:-{value}}}"' for key, value in BASE_FRONTIER_EXPORTS.items()]
    lines.extend(f'export {key}="${{{key}:-{value}}}"' for key, value in DIRECT_RUNTIME_DEFAULTS.items())
    lines.extend(
        [
            'if [[ -z "${PREFILL_SELECTOR_TILE_SIZE+x}" ]]; then',
            '  if [[ "${PREFILL_SELECTOR_BACKEND}" == "native" ]]; then',
            "    export PREFILL_SELECTOR_TILE_SIZE=2048",
            "  else",
            "    export PREFILL_SELECTOR_TILE_SIZE=256",
            "  fi",
            "else",
            "  export PREFILL_SELECTOR_TILE_SIZE",
            "fi",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit canonical frontier configuration.")
    parser.add_argument("--emit-shell", action="store_true", help="Print sourceable shell exports")
    parser.add_argument("--emit-direct-runtime-shell", action="store_true", help="Print non-joint HF runtime defaults")
    args = parser.parse_args()
    if args.emit_shell:
        print("\n".join(canonical_env_export_lines()))
        return
    if args.emit_direct_runtime_shell:
        print("\n".join(direct_runtime_export_lines()))
        return
    for key, value in CANONICAL_ENV_DEFAULTS.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
