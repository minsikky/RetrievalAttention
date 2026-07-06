#!/usr/bin/env python3
"""Report CUDA-native coverage for the frontier decode hot path.

This is a static/readiness audit, not a profiler.  It tracks the current
canonical implementation boundary so optimization work can distinguish:

- custom CUDA kernels in ``selector_paged_pq``;
- ATen/PyTorch GPU ops still called from Python/C++;
- CPU/Python orchestration that remains in the decode loop.

The checklist is intentionally conservative: if a component still relies on a
hot-path PyTorch ``topk``, ``matmul``, ``stack``, or per-call tensor
materialization, it is not marked fully CUDA-native even when it executes on the
GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HotPathItem:
    component: str
    status: str
    implementation: str
    evidence: str
    next_kernel_target: str

    @property
    def blocks_full_native(self) -> bool:
        return self.status in {"pytorch-hotpath", "mostly-native", "python-cpu"}


HOTPATH_ITEMS: tuple[HotPathItem, ...] = (
    HotPathItem(
        component="Paged K-PQ score fullscan",
        status="mostly-native",
        implementation=(
            "CUDA extension computes PQ scores; canonical path keeps the dense score rows "
            "for later rank-prefix and score-grid work."
        ),
        evidence="gqa_fullscan_pq_topk_scores / rank_paged_pq_batched_with_scores",
        next_kernel_target=(
            "Fuse score generation with rank-prefix materialization without changing stable/tie-sensitive ordering."
        ),
    ),
    HotPathItem(
        component="Rank-prefix candidate ordering",
        status="pytorch-hotpath",
        implementation=(
            "Canonical path uses torch.topk over PQ score rows for the largest partial K budget.  "
            "Full-budget rows are already handled as exact score-grid rows and do not require a full-context "
            "rank sort.  The current native rank-prefix workspace path preserves parity, but it uses full "
            "segmented radix sort and is slower than canonical PyTorch topk.  The partial budget-prefix "
            "diagnostic also preserves parity, but its 32-pass threshold scan is even slower.  An unsorted "
            "per-budget top-k diagnostic preserves saved-trace parity only when it falls back from the "
            "prefix-based native score-grid, but that makes RULER much slower.  These remain diagnostic."
        ),
        evidence=(
            "torch.topk in approximate_joint_kv_all_heads; "
            "SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX=0; SELECTOR_PQ_JOINT_NATIVE_BUDGET_PREFIX=0"
        ),
        next_kernel_target=(
            "A partial top-k/prefix kernel, or PQ score-generation plus prefix fusion, specialized for heads<=4, "
            "the largest partial budget, and canonical tie/nested-prefix behavior."
        ),
    ),
    HotPathItem(
        component="Exact-K refinement logits",
        status="pytorch-hotpath",
        implementation=(
            "Canonical path computes dense exact logits with PyTorch matmul per KV-head group.  "
            "Native full-logit/cuBLAS/grouped-GQA diagnostic helpers exist but are not promoted.  "
            "A sparse exact-K diagnostic path now uses native arbitrary-token exact logits plus a "
            "native sparse exact-score table for base and ranked-prefix tokens, then feeds the "
            "no-fill tokenfit score-grid path.  It remains guarded off until HF runtime validation "
            "proves it is promotable."
        ),
        evidence=(
            "queries_h @ keys_t.transpose(0, 1); SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS=0; "
            "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID=0; gqa_decode_token_exact_logits_cuda; "
            "joint_sparse_exact_score_table_cuda"
        ),
        next_kernel_target=(
            "Profile and, if positive, promote sparse exact-K refinement; otherwise replace the remaining "
            "dense PyTorch matmul with a better fused all-KV-head sparse exact score-grid kernel."
        ),
    ),
    HotPathItem(
        component="Grouped-GQA exact-logit primitive",
        status="diagnostic-native",
        implementation=(
            "Opt-in CUDA helper computes full exact QK logits with one warp per `(kv_head, token)`, "
            "sharing each K load across local GQA query heads. It passed CUDA unit, long trace parity, "
            "and the short RULER gate, but sustained LongGen accounting regressed, so it remains diagnostic."
        ),
        evidence=(
            "SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS_BACKEND=grouped; "
            "gqa_decode_full_exact_logits_grouped"
        ),
        next_kernel_target=(
            "Do not promote this exact-logit-only shape; sustained runtime is dominated by score/rank/risk/prob work."
        ),
    ),
    HotPathItem(
        component="Mixed exact/PQ score grid",
        status="custom-cuda",
        implementation=(
            "Custom CUDA builds calibrated mixed score rows for every K budget.  Diagnostic model-scoped "
            "single-slot and grouped score-grid workspaces exist, but canonical mode keeps them disabled "
            "until sustained validation proves them."
        ),
        evidence="SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID=1; joint_mixed_score_grid",
        next_kernel_target=(
            "Use grouped device workspaces or a single all-KV-head native entry across layers/decode, and promote "
            "only if LongGen sustained timing improves without logical-stat drift."
        ),
    ),
    HotPathItem(
        component="Mixed softmax + V-PQ base output",
        status="custom-cuda",
        implementation=(
            "Custom CUDA computes softmax probabilities and base output from the mixed score grid.  "
            "A diagnostic model-scoped single-slot workspace exists, but canonical mode keeps it disabled "
            "until parity/profile validation proves it."
        ),
        evidence="SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE=1; joint_softmax_base_outputs",
        next_kernel_target=(
            "Promote softmax/base workspace only if it reduces allocation/runtime without logical-stat drift; "
            "otherwise fuse with score-grid only if it reduces work without accepted-budget drift."
        ),
    ),
    HotPathItem(
        component="V-PQ sidecar and residual/error stats",
        status="mostly-native",
        implementation=(
            "Canonical decode uses persistent grouped sidecar buffers plus native exact-suffix append "
            "for unsealed decode tokens.  An opt-in native sealed-page sidecar constructor exists for "
            "V-PQ reconstruction/residual/error stats, but it is not promoted until HF parity/runtime validation passes."
        ),
        evidence="SELECTOR_PQ_JOINT_NATIVE_VPQ_APPEND=1; joint_vpq_append_exact_suffix_grouped; joint_vpq_sidecars_from_pack",
        next_kernel_target="Validate/promote native sealed-page V-PQ sidecar construction, then remove remaining grouped packing.",
    ),
    HotPathItem(
        component="Residual-risk V-prefix outputs",
        status="custom-cuda",
        implementation="Custom CUDA sorts residual risk and accumulates exact-V residual prefixes.",
        evidence="SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX=1; joint_vprefix_outputs_from_grouped_risk_batched",
        next_kernel_target=(
            "Workspace reuse and fused no-MB policy were not enough; next target is reducing risk-sort/prefix work itself "
            "without on-demand output recomputation, for example policy-aware prefix materialization."
        ),
    ),
    HotPathItem(
        component="Score-direct softmax + residual-risk V-prefix primitive",
        status="diagnostic-native",
        implementation=(
            "Prototype CUDA helper consumes grouped mixed score rows directly and computes V-PQ base plus "
            "residual-risk V-prefix outputs without materializing the full probability grid.  The paired "
            "score-direct interval-policy helper also selects the adaptive K/V output from score-derived "
            "intervals.  Validation preserved quality, but both shapes regressed runtime because the current "
            "wiring still stacks score grids in Python and streams risk-prefix work in a worse shape."
        ),
        evidence="joint_vprefix_outputs_from_grouped_scores_batched; joint_select_policy_from_grouped_scores_intervals_batched_no_mb",
        next_kernel_target=(
            "Do not promote this shape.  The next native target is a single all-KV-head/grouped entry that writes "
            "grouped score/prob/base/risk workspaces directly and avoids Python stacking plus repeated risk-prefix passes."
        ),
    ),
    HotPathItem(
        component="Residual-risk interval policy",
        status="diagnostic-native",
        implementation=(
            "Opt-in CUDA helper reuses residual-risk interval sums and selects the adaptive K/V policy output "
            "without materializing the full V-prefix output grid first. It passed parity but did not improve "
            "RULER runtime, so it remains guarded off."
        ),
        evidence=(
            "SELECTOR_PQ_JOINT_INTERVAL_RISK_POLICY=0; "
            "joint_select_policy_from_grouped_risk_intervals_batched_no_mb"
        ),
        next_kernel_target=(
            "Do not revisit interval-only policy selection unless upstream risk-prefix construction is also fused."
        ),
    ),
    HotPathItem(
        component="Adaptive K/V confidence policy",
        status="custom-cuda",
        implementation="Custom CUDA selects the first stable K/V budget pair for the canonical policy.",
        evidence="SELECTOR_PQ_JOINT_NATIVE_POLICY=1; joint_select_policy_grouped_flat_no_mb",
        next_kernel_target="Keep as native; only revisit after upstream tensors are grouped/persistent.",
    ),
    HotPathItem(
        component="Grouped tensor packing",
        status="pytorch-hotpath",
        implementation=(
            "Python collects per-KV-head records and uses torch.stack/contiguous packing before "
            "grouped residual-risk kernels. A simple native per-group copy helper was validated in an "
            "isolated worktree but regressed runtime, so the remaining target is writing grouped buffers "
            "directly from upstream kernels."
        ),
        evidence="grouped_risk_records, torch.stack(base_output_grid), torch.stack(probs_grid)",
        next_kernel_target="Single all-KV-head native entry point or preallocated grouped output buffers written by earlier kernels.",
    ),
    HotPathItem(
        component="Logical/physical MB accounting",
        status="custom-cuda",
        implementation=(
            "Custom CUDA reduces grouped selected-count, tail-count, selector-MB, exact-KV-MB, "
            "tail-MB, physical-MB, and active-count sums to tiny device tensors.  Compatible grouped "
            "records are batched before accounting, and Python flushes aggregate layer summaries at reporting time."
        ),
        evidence="SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING=1; joint_grouped_accounting_sums_cuda",
        next_kernel_target="Keep canonical; remaining work is reducing upstream risk-prefix/rank-prefix/prob-base runtime.",
    ),
)


def _repo_path(path: str) -> Path:
    return REPO_ROOT / path


def _read(path: str) -> str:
    return _repo_path(path).read_text(encoding="utf-8")


def _read_cuda_extension_text() -> str:
    kernel = _repo_path("benchmark/selector_eval/cuda_ext/paged_pq_kernel.cu")
    parts = _repo_path("benchmark/selector_eval/cuda_ext/paged_pq_kernel_parts")
    chunks = [kernel.read_text(encoding="utf-8")]
    if parts.is_dir():
        for path in sorted(parts.glob("*.cu.inc")):
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def _read_runner_text() -> str:
    runners = _repo_path("benchmark/selector_eval/runners")
    chunks: list[str] = []
    for path in sorted(runners.glob("hf_paged_pq_intervention*.py")):
        chunks.append(path.read_text(encoding="utf-8"))
    chunks.append(_read("benchmark/selector_eval/runners/run_hf_paged_pq_intervention_eval.py"))
    return "\n".join(chunks)


def _check_expected_patterns() -> list[str]:
    """Return missing evidence patterns that would make the audit stale."""

    runner = _read_runner_text()
    ext = _read_cuda_extension_text()
    missing: list[str] = []
    expected = {
        "runner": (
            "SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID",
            "SELECTOR_PQ_JOINT_SCORE_GRID_WORKSPACE",
            "SELECTOR_PQ_JOINT_GROUPED_SCORE_WORKSPACE",
            "_pagedpq_joint_grouped_score_grid_workspace_cache",
            "_pagedpq_joint_score_grid_workspace_cache",
            "SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE",
            "SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE",
            "SELECTOR_PQ_JOINT_SOFTMAX_BASE_WORKSPACE",
            "_pagedpq_joint_softmax_base_workspace_cache",
            "SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX",
            "SELECTOR_PQ_JOINT_NATIVE_BUDGET_PREFIX",
            "SELECTOR_PQ_JOINT_RANK_PREFIX_WORKSPACE",
            "SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS",
            "SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS_BACKEND",
            "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID",
            "SELECTOR_PQ_JOINT_NATIVE_VPQ_APPEND",
            "SELECTOR_PQ_JOINT_NATIVE_VPQ_SIDECAR",
            "SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE",
            "SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX",
            "SELECTOR_PQ_JOINT_SCORE_DIRECT_INTERVAL_POLICY",
            "SELECTOR_PQ_JOINT_SCORE_PROB_INTERVAL_POLICY",
            "SELECTOR_PQ_JOINT_SCORE_DIRECT_TOPK_INTERVAL_POLICY",
            "SELECTOR_PQ_JOINT_MERGE_RISK_POLICY",
            "SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING",
            "joint_vpq_sidecars_for",
            "grouped_risk_records",
            "torch.topk",
            "queries_h @ keys_t.transpose",
            "torch.stack(",
        ),
        "ext": (
            "joint_mixed_score_grid_cuda",
            "joint_softmax_base_outputs_cuda",
            "joint_softmax_base_outputs_grouped_cuda",
            "joint_softmax_base_outputs_workspace_cuda",
            "joint_vprefix_outputs_from_grouped_risk_batched_cuda",
            "joint_vprefix_outputs_from_grouped_scores_batched_cuda",
            "joint_select_policy_from_grouped_scores_intervals_batched_no_mb_cuda",
            "joint_mixed_select_policy_merge_rankpos_no_calib_no_mb_cuda",
            "joint_vprefix_outputs_from_grouped_risk_batched_workspace_cuda",
            "joint_select_policy_from_grouped_risk_intervals_batched_no_mb_cuda",
            "joint_select_policy_grouped_flat_no_mb_cuda",
            "joint_grouped_accounting_sums_cuda",
            "joint_vpq_sidecars_from_pack_cuda",
            "joint_rank_prefix_tokens_workspace_cuda",
            "joint_budget_prefix_tokens_cuda",
            "gqa_decode_full_exact_logits_grouped_cuda",
            "gqa_decode_token_exact_logits_cuda",
            "joint_sparse_exact_score_table_cuda",
        ),
    }
    for pattern in expected["runner"]:
        if pattern not in runner:
            missing.append(f"runner missing pattern: {pattern}")
    for pattern in expected["ext"]:
        if pattern not in ext:
            missing.append(f"cuda extension missing pattern: {pattern}")
    return missing


def render_markdown() -> str:
    missing = _check_expected_patterns()
    blockers = [item for item in HOTPATH_ITEMS if item.blocks_full_native]
    lines: list[str] = [
        "# Frontier CUDA-Native Hot Path Audit",
        "",
        "This audit is static. It records which canonical decode components are truly custom CUDA today and which are still PyTorch/ATen-heavy.",
        "",
        "## Summary",
        "",
    ]
    custom = sum(1 for item in HOTPATH_ITEMS if item.status == "custom-cuda")
    partial = sum(1 for item in HOTPATH_ITEMS if item.status == "mostly-native")
    pytorch = sum(1 for item in HOTPATH_ITEMS if item.status == "pytorch-hotpath")
    cpu = sum(1 for item in HOTPATH_ITEMS if item.status == "python-cpu")
    diagnostic = sum(1 for item in HOTPATH_ITEMS if item.status == "diagnostic-native")
    lines.extend(
        [
            f"- Custom CUDA components: `{custom}`",
            f"- Mostly native but still ATen/PyTorch-dependent: `{partial}`",
            f"- PyTorch/ATen hot-path components: `{pytorch}`",
            f"- CPU/Python accounting-only components: `{cpu}`",
            f"- Diagnostic native components not yet canonical: `{diagnostic}`",
            f"- Full CUDA-native production gate: `{'blocked' if blockers else 'pass'}`",
            f"- Blocking components for full native gate: `{len(blockers)}`",
            "",
            "## Component Map",
            "",
            "| component | status | current implementation | evidence | next native target |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in HOTPATH_ITEMS:
        lines.append(
            "| "
            + " | ".join(
                (
                    item.component,
                    f"`{item.status}`",
                    item.implementation,
                    f"`{item.evidence}`",
                    item.next_kernel_target,
                )
            )
            + " |"
        )
    if blockers:
        lines.extend(
            [
                "",
                "## Full Native Blockers",
                "",
                "These components must become promoted custom CUDA kernels before the backend can be called fully CUDA-native.",
                "",
            ]
        )
        for item in blockers:
            lines.append(f"- `{item.component}`: {item.next_kernel_target}")
    if missing:
        lines.extend(["", "## Staleness Warnings", ""])
        lines.extend(f"- {msg}" for msg in missing)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    parser.add_argument(
        "--fail-on-full-native-blockers",
        action="store_true",
        help="Exit non-zero while any canonical component still blocks the full CUDA-native production gate.",
    )
    args = parser.parse_args()
    text = render_markdown()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if args.fail_on_full_native_blockers:
        blockers = [item for item in HOTPATH_ITEMS if item.blocks_full_native]
        if blockers:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
