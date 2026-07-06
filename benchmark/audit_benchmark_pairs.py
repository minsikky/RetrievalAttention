#!/usr/bin/env python3
"""Pairwise audit for dense-vs-frontier benchmark artifacts.

This is intentionally artifact-only. It does not import model code or run
benchmarks; it reads summaries/predictions and reports whether completed pairs
are comparable and whether the frontier path was actually active.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

MB = 1024 * 1024
DEFAULT_DENSE_HEAD_DIM = 128
DEFAULT_DENSE_BYTES_PER_ELEM = 2


PAIR_PREFIXES = ("dense_", "pagedpq_", "frontier_")
SETTING_KEYS = (
    "model_name",
    "benchmark",
    "dataset_name",
    "split",
    "max_examples",
    "length_filter",
    "difficulty_filter",
    "domain_filter",
    "id_filter",
    "selection",
    "seed",
    "max_input_tokens",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "disable_thinking",
    "use_chat_template",
    "force_max_new_tokens",
    "min_new_tokens",
    "qwen_yarn_factor",
)
ID_KEYS = ("_id", "id", "sample_id", "task_id", "question_id", "problem_id")
PRED_KEYS = ("pred", "prediction", "predicted_answer", "extracted_answer", "choice")
JUDGE_KEYS = ("judge", "correct", "is_correct", "passed")
RESPONSE_KEYS = ("response", "completion", "generated_text", "output")


@dataclass
class ArtifactRun:
    label: str
    path: str
    mode: str
    summary_exists: bool
    predictions_exists: bool
    selected_ids_exists: bool
    quality: float | None = None
    quality_name: str = "quality"
    examples: int | None = None
    sec_per_example: float | None = None
    generated_tokens: float | None = None
    dense_step_mb: float | None = None
    step_mb: float | None = None
    selector_mb: float | None = None
    exact_kv_mb: float | None = None
    tail_mb: float | None = None
    update_mb: float | None = None
    physical_step_mb: float | None = None
    selected_tokens: float | None = None
    active_fraction: float | None = None
    approx_calls: float | None = None
    passthrough_calls: float | None = None
    longgen_completion_pct: float | None = None
    longgen_once_pct: float | None = None
    longgen_range_pct: float | None = None
    longgen_periodic_pct: float | None = None
    longgen_subquestion_pct: float | None = None
    summary: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class PredictionDiff:
    dense_rows: int = 0
    frontier_rows: int = 0
    same_hash: bool | None = None
    same_id_rows: int | None = None
    same_pred_rows: int | None = None
    same_judge_rows: int | None = None
    same_response_rows: int | None = None
    comparable_pred_rows: int = 0
    comparable_judge_rows: int = 0
    comparable_response_rows: int = 0
    first_differences: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PairAudit:
    key: str
    dense: ArtifactRun | None
    frontier: ArtifactRun | None
    quality_delta_pct: float | None
    runtime_ratio: float | None
    logical_savings_pct: float | None
    physical_savings_pct: float | None
    prediction_diff: PredictionDiff | None
    warnings: list[str]


@dataclass
class PendingOrFailed:
    label: str
    output_dir: str
    reason: str
    slurm_out: str = ""
    jobid: str = ""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _truthy(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return None


def _quality(summary: dict[str, Any]) -> tuple[float | None, str]:
    if str(summary.get("benchmark") or "").startswith("longgenbench_sgt_"):
        if _as_float(summary.get("mean_substring_periodic_acc")) is not None:
            return 100.0 * float(summary["mean_substring_periodic_acc"]), "substring_periodic_pct"
        if _as_float(summary.get("mean_substring_range_acc")) is not None:
            return 100.0 * float(summary["mean_substring_range_acc"]), "substring_range_pct"
        if _as_float(summary.get("mean_substring_once_acc")) is not None:
            return 100.0 * float(summary["mean_substring_once_acc"]), "substring_once_pct"
        if _as_float(summary.get("mean_completion_rate")) is not None:
            return 100.0 * float(summary["mean_completion_rate"]), "completion_pct"
    if summary.get("benchmark") == "longgenbench_gsm8k" and _as_float(summary.get("subquestion_accuracy_pct")) is not None:
        return _as_float(summary.get("subquestion_accuracy_pct")), "subquestion_accuracy_pct"
    if _as_float(summary.get("accuracy_pct")) is not None:
        return _as_float(summary.get("accuracy_pct")), "accuracy_pct"
    if _as_float(summary.get("score")) is not None:
        return _as_float(summary.get("score")), "score"
    if _as_float(summary.get("pass_at_1")) is not None:
        return 100.0 * float(summary["pass_at_1"]), "pass@1_pct"
    if _as_float(summary.get("mean_substring_once_acc")) is not None:
        return 100.0 * float(summary["mean_substring_once_acc"]), "substring_once_pct"
    if _as_float(summary.get("mean_completion_rate")) is not None:
        return 100.0 * float(summary["mean_completion_rate"]), "completion_pct"
    return None, "quality"


def _cost(summary: dict[str, Any]) -> dict[str, float]:
    raw = summary.get("cost_proxy_aggregate")
    if isinstance(raw, dict):
        return {
            str(k): float(v)
            for k, v in raw.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
        }
    return {}


def _pct(summary: dict[str, Any], key: str) -> float | None:
    value = _as_float(summary.get(key))
    return None if value is None else 100.0 * value


def _cost_value(cost: dict[str, float], *names: str) -> float | None:
    for name in names:
        if name in cost:
            return cost[name]
    return None


def _avg_context(summary: dict[str, Any]) -> float | None:
    prompt = _as_float(summary.get("avg_used_prompt_tokens"))
    if prompt is None:
        prompt = _as_float(summary.get("avg_prompt_tokens"))
    if prompt is None:
        prompt = _as_float(summary.get("mean_prompt_tokens"))
    if prompt is None:
        prompt = _as_float(summary.get("max_input_tokens"))
    if prompt is None:
        return None
    generated = _as_float(summary.get("avg_generated_tokens"))
    if generated is None:
        generated = _as_float(summary.get("mean_generated_tokens"))
    if generated is None:
        generated = _as_float(summary.get("max_new_tokens")) or 1.0
    return prompt + max(0.0, generated - 1.0) / 2.0


def _dense_step_mb(
    summary: dict[str, Any],
    *,
    head_dim: int = DEFAULT_DENSE_HEAD_DIM,
    bytes_per_elem: int = DEFAULT_DENSE_BYTES_PER_ELEM,
) -> float | None:
    context = _avg_context(summary)
    if context is None:
        return None
    return context * head_dim * bytes_per_elem * 2.0 / MB


def _summary_file(path: Path) -> Path | None:
    direct = path / "summary.json"
    if direct.exists():
        return direct
    summary_dir = path / "summary"
    if summary_dir.is_dir():
        matches = sorted(summary_dir.glob("*.json"))
        if matches:
            return matches[0]
    return None


def _prediction_file(path: Path, summary_path: Path | None = None) -> Path | None:
    direct = path / "predictions.jsonl"
    if direct.exists():
        return direct
    pred_dir = path / "pred"
    if pred_dir.is_dir():
        if summary_path is not None:
            candidate = pred_dir / f"{summary_path.stem}.jsonl"
            if candidate.exists():
                return candidate
        matches = sorted(pred_dir.glob("*.jsonl"))
        if matches:
            return matches[0]
    return None


def _selected_ids_file(path: Path) -> Path | None:
    direct = path / "selected_ids.json"
    return direct if direct.exists() else None


def _mode_from_name_or_summary(path: Path, summary: dict[str, Any] | None) -> str:
    name = path.name
    if name.startswith("dense_"):
        return "dense"
    if name.startswith(("pagedpq_", "frontier_")):
        return "frontier"
    if summary:
        mode = str(summary.get("attention_mode") or summary.get("mode") or "")
        if mode in {"dense", "dense_batched", "dense_stream"}:
            return "dense"
        if mode in {"pagedpq", "pagedpq_batched", "pagedpq_stream"}:
            return "frontier"
    return "unknown"


def _pair_key(path: Path, summary: dict[str, Any] | None) -> str:
    name = path.name
    for prefix in PAIR_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    if summary:
        parts = [
            str(summary.get("benchmark") or summary.get("dataset_name") or path.parent.name),
            str(summary.get("length_filter") or ""),
            str(summary.get("difficulty_filter") or ""),
            str(summary.get("domain_filter") or ""),
            str(summary.get("max_examples") or summary.get("num_examples") or ""),
            str(summary.get("max_input_tokens") or ""),
        ]
        return "_".join(part for part in parts if part)
    return name


def _run_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _summarize_run(path: Path, *, root: Path) -> ArtifactRun:
    summary_path = _summary_file(path)
    predictions_path = _prediction_file(path, summary_path)
    selected_ids_path = _selected_ids_file(path)
    summary = _load_json(summary_path) if summary_path is not None else {}
    cost = _cost(summary)
    quality, quality_name = _quality(summary)
    return ArtifactRun(
        label=_run_label(path, root),
        path=str(path),
        mode=_mode_from_name_or_summary(path, summary),
        summary_exists=summary_path is not None,
        predictions_exists=predictions_path is not None,
        selected_ids_exists=selected_ids_path is not None,
        quality=quality,
        quality_name=quality_name,
        examples=int(summary["num_examples"]) if isinstance(summary.get("num_examples"), int) else None,
        sec_per_example=_as_float(summary.get("avg_generation_sec"))
        or _as_float(summary.get("mean_stream_total_seconds")),
        generated_tokens=_as_float(summary.get("avg_generated_tokens"))
        or _as_float(summary.get("mean_generated_tokens")),
        dense_step_mb=_dense_step_mb(summary),
        step_mb=_cost_value(cost, "mean_step_MB_per_head_query", "mean_logical_frontier_step_MB_per_head_query"),
        selector_mb=_cost_value(cost, "mean_selector_MB_per_head_query", "mean_logical_frontier_selector_MB_per_head_query"),
        exact_kv_mb=_cost_value(cost, "mean_exact_KV_MB_per_head_query", "mean_logical_frontier_exact_KV_MB_per_head_query"),
        tail_mb=_cost_value(cost, "mean_tail_estimator_MB_per_head_query", "mean_logical_frontier_tail_estimator_MB_per_head_query"),
        update_mb=_cost_value(cost, "online_update_MB_per_head_query", "mean_update_MB_per_head_query"),
        physical_step_mb=_cost_value(cost, "mean_physical_gpu_step_MB_per_head_query"),
        selected_tokens=_cost_value(cost, "mean_selected_tokens"),
        active_fraction=_cost_value(cost, "approx_path_active_fraction", "selector_active_fraction"),
        approx_calls=_cost_value(cost, "approx_attention_calls_total", "approx_attention_calls"),
        passthrough_calls=_cost_value(cost, "passthrough_attention_calls_total", "passthrough_attention_calls"),
        longgen_completion_pct=_pct(summary, "mean_completion_rate"),
        longgen_once_pct=_pct(summary, "mean_substring_once_acc"),
        longgen_range_pct=_pct(summary, "mean_substring_range_acc"),
        longgen_periodic_pct=_pct(summary, "mean_substring_periodic_acc"),
        longgen_subquestion_pct=_as_float(summary.get("subquestion_accuracy_pct")),
        summary=summary,
    )


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for marker in ("summary.json", "predictions.jsonl", "selected_ids.json"):
        for file_path in root.rglob(marker):
            run_dir = file_path.parent
            if run_dir not in seen:
                seen.add(run_dir)
                yield run_dir
    for file_path in root.rglob("summary/*.json"):
        run_dir = file_path.parent.parent
        if run_dir not in seen:
            seen.add(run_dir)
            yield run_dir


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_existing(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def _compare_predictions(dense_dir: Path, frontier_dir: Path) -> PredictionDiff | None:
    dense_path = _prediction_file(dense_dir, _summary_file(dense_dir))
    frontier_path = _prediction_file(frontier_dir, _summary_file(frontier_dir))
    if dense_path is None or frontier_path is None:
        return None

    diff = PredictionDiff(same_hash=(_sha256(dense_path) == _sha256(frontier_path)))
    same_id = same_pred = same_judge = same_response = 0
    comparable_pred = comparable_judge = comparable_response = 0

    with dense_path.open("r", encoding="utf-8") as dense_f, frontier_path.open("r", encoding="utf-8") as frontier_f:
        for idx, (dense_line, frontier_line) in enumerate(zip_longest(dense_f, frontier_f)):
            if dense_line is not None:
                diff.dense_rows += 1
            if frontier_line is not None:
                diff.frontier_rows += 1
            if dense_line is None or frontier_line is None:
                continue
            try:
                dense = json.loads(dense_line)
                frontier = json.loads(frontier_line)
            except json.JSONDecodeError:
                continue
            dense_id = _first_existing(dense, ID_KEYS)
            frontier_id = _first_existing(frontier, ID_KEYS)
            if dense_id == frontier_id:
                same_id += 1

            dense_pred = _first_existing(dense, PRED_KEYS)
            frontier_pred = _first_existing(frontier, PRED_KEYS)
            pred_match = None
            if dense_pred is not None and frontier_pred is not None:
                comparable_pred += 1
                pred_match = dense_pred == frontier_pred
                same_pred += int(pred_match)

            dense_judge = _first_existing(dense, JUDGE_KEYS)
            frontier_judge = _first_existing(frontier, JUDGE_KEYS)
            judge_match = None
            if dense_judge is not None and frontier_judge is not None:
                comparable_judge += 1
                judge_match = dense_judge == frontier_judge
                same_judge += int(judge_match)

            dense_response = _first_existing(dense, RESPONSE_KEYS)
            frontier_response = _first_existing(frontier, RESPONSE_KEYS)
            response_match = None
            if dense_response is not None and frontier_response is not None:
                comparable_response += 1
                response_match = dense_response == frontier_response
                same_response += int(response_match)

            if len(diff.first_differences) < 5 and (
                pred_match is False or judge_match is False or response_match is False
            ):
                diff.first_differences.append(
                    {
                        "row": idx,
                        "dense_id": dense_id,
                        "frontier_id": frontier_id,
                        "dense_pred": dense_pred,
                        "frontier_pred": frontier_pred,
                        "dense_judge": dense_judge,
                        "frontier_judge": frontier_judge,
                    }
                )

    diff.same_id_rows = same_id
    diff.same_pred_rows = same_pred if comparable_pred else None
    diff.same_judge_rows = same_judge if comparable_judge else None
    diff.same_response_rows = same_response if comparable_response else None
    diff.comparable_pred_rows = comparable_pred
    diff.comparable_judge_rows = comparable_judge
    diff.comparable_response_rows = comparable_response
    return diff


def _selected_ids_match(dense_dir: Path, frontier_dir: Path) -> bool | None:
    dense_path = _selected_ids_file(dense_dir)
    frontier_path = _selected_ids_file(frontier_dir)
    if dense_path is None or frontier_path is None:
        return None
    return _sha256(dense_path) == _sha256(frontier_path)


def _settings(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: summary.get(key) for key in SETTING_KEYS if key in summary}


def _config(summary: dict[str, Any]) -> dict[str, Any]:
    raw = summary.get("pagedpq_config")
    return raw if isinstance(raw, dict) else {}


def _pair_warnings(dense: ArtifactRun | None, frontier: ArtifactRun | None, pred: PredictionDiff | None) -> list[str]:
    warnings: list[str] = []
    if dense is None:
        return ["missing-dense-run"]
    if frontier is None:
        return ["missing-frontier-run"]

    dense_dir = Path(dense.path)
    frontier_dir = Path(frontier.path)

    if not dense.summary_exists:
        warnings.append("dense-missing-summary")
    if not frontier.summary_exists:
        warnings.append("frontier-missing-summary")
    dense_attention_mode = dense.summary.get("attention_mode") or dense.summary.get("mode")
    frontier_attention_mode = frontier.summary.get("attention_mode") or frontier.summary.get("mode")
    if dense_attention_mode not in {"dense", "dense_batched", "dense_stream"}:
        warnings.append("dense-attention-mode-not-dense")
    if frontier_attention_mode not in {"pagedpq", "pagedpq_batched", "pagedpq_stream"}:
        warnings.append("frontier-attention-mode-not-pagedpq")

    dense_settings = _settings(dense.summary)
    frontier_settings = _settings(frontier.summary)
    for key in sorted(set(dense_settings) & set(frontier_settings)):
        if key == "model_name":
            # Dense/frontier may store equivalent absolute cache paths; still compare exactly.
            pass
        if dense_settings[key] != frontier_settings[key]:
            warnings.append(f"setting-mismatch:{key}")

    selected_match = _selected_ids_match(dense_dir, frontier_dir)
    if selected_match is False:
        warnings.append("selected-ids-mismatch")
    elif selected_match is None and (
        _selected_ids_file(dense_dir) is not None or _selected_ids_file(frontier_dir) is not None
    ):
        warnings.append("selected-ids-missing")

    if pred is None:
        warnings.append("missing-predictions")
    else:
        if pred.dense_rows != pred.frontier_rows:
            warnings.append("prediction-row-count-mismatch")
        if pred.same_id_rows is not None and pred.same_id_rows != min(pred.dense_rows, pred.frontier_rows):
            warnings.append("prediction-id-mismatch")
        if pred.same_hash:
            warnings.append("predictions-byte-identical")

    config = _config(frontier.summary)
    if not config:
        warnings.append("frontier-missing-pagedpq-config")
    else:
        if _truthy(config.get("frontier_canonical_gpu")) is not True:
            warnings.append("frontier-canonical-guard-off")
        if config.get("online_confidence_rule") != "joint_kv_stability":
            warnings.append("frontier-noncanonical-confidence")
        if config.get("selected_value_exact_rule") != "global_residual_risk":
            warnings.append("frontier-noncanonical-v-rule")
        if config.get("tail_score_calibration") != "none":
            warnings.append("frontier-noncanonical-tail-calibration")
        if config.get("selector_mode") != "fullscan":
            warnings.append("frontier-non-fullscan-selector")
        if config.get("selector_backend") != "cuda_ext":
            warnings.append("frontier-non-cuda-selector")
        if _truthy(config.get("approx_prefill")) is True or frontier.summary.get("approx_prefill") is True:
            warnings.append("frontier-approx-prefill")

    if frontier.active_fraction is None:
        warnings.append("frontier-missing-active-fraction")
    elif frontier.active_fraction <= 0.0:
        warnings.append("frontier-approx-path-inactive")
    if frontier.approx_calls is None:
        warnings.append("frontier-missing-approx-calls")
    elif frontier.approx_calls <= 0.0:
        warnings.append("frontier-zero-approx-calls")
    if frontier.step_mb is None or frontier.step_mb <= 0.0:
        warnings.append("frontier-missing-logical-step-mb")
    if frontier.physical_step_mb is None:
        warnings.append("frontier-missing-physical-step-mb")
    dense_mb = dense.dense_step_mb if dense.dense_step_mb is not None else frontier.dense_step_mb
    if dense_mb is not None:
        if frontier.step_mb is not None and frontier.step_mb > dense_mb:
            warnings.append("logical-bandwidth-worse-than-dense")
        if frontier.physical_step_mb is not None and frontier.physical_step_mb > dense_mb:
            warnings.append("physical-bandwidth-worse-than-dense")
    if dense.quality is not None and frontier.quality is not None and frontier.quality < dense.quality:
        warnings.append("frontier-quality-lower")
    if _as_float(frontier.summary.get("max_new_token_hit_count")) not in {None, 0.0}:
        warnings.append("frontier-hit-max-new-tokens")

    return warnings


def _savings(candidate_mb: float | None, dense_mb: float | None) -> float | None:
    if candidate_mb is None or dense_mb is None or dense_mb <= 0.0:
        return None
    return 100.0 * (1.0 - candidate_mb / dense_mb)


def _runtime_ratio(frontier: ArtifactRun | None, dense: ArtifactRun | None) -> float | None:
    if frontier is None or dense is None:
        return None
    if frontier.sec_per_example is None or dense.sec_per_example is None or dense.sec_per_example <= 0.0:
        return None
    return frontier.sec_per_example / dense.sec_per_example


def _audit_pairs(runs: list[ArtifactRun]) -> list[PairAudit]:
    by_key: dict[str, dict[str, ArtifactRun]] = {}
    for run in runs:
        key = _pair_key(Path(run.path), run.summary)
        by_key.setdefault(key, {})[run.mode] = run

    audits: list[PairAudit] = []
    for key in sorted(by_key):
        dense = by_key[key].get("dense")
        frontier = by_key[key].get("frontier")
        pred = _compare_predictions(Path(dense.path), Path(frontier.path)) if dense and frontier else None
        quality_delta = None
        dense_mb = dense.dense_step_mb if dense else None
        if dense and frontier and dense.quality is not None and frontier.quality is not None:
            quality_delta = frontier.quality - dense.quality
        if frontier and dense_mb is None:
            dense_mb = frontier.dense_step_mb
        audits.append(
            PairAudit(
                key=key,
                dense=dense,
                frontier=frontier,
                quality_delta_pct=quality_delta,
                runtime_ratio=_runtime_ratio(frontier, dense),
                logical_savings_pct=_savings(frontier.step_mb if frontier else None, dense_mb),
                physical_savings_pct=_savings(frontier.physical_step_mb if frontier else None, dense_mb),
                prediction_diff=pred,
                warnings=_pair_warnings(dense, frontier, pred),
            )
        )
    return audits


def _run_dirs_from_manifest(path: Path) -> tuple[list[Path], list[PendingOrFailed]]:
    dirs: list[Path] = []
    incomplete: list[PendingOrFailed] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return dirs, incomplete
        for row in reader:
            labels = [part.strip() for part in str(row.get("task_labels", "")).split(",") if part.strip()]
            output_dirs = [part.strip() for part in str(row.get("output_dir", "")).split(",") if part.strip()]
            if not output_dirs:
                continue
            if len(labels) != len(output_dirs):
                labels = [str(row.get("label") or f"run_{idx}") for idx, _ in enumerate(output_dirs)]
            for label, output_dir in zip(labels, output_dirs):
                run_dir = Path(output_dir)
                dirs.append(run_dir)
                if _summary_file(run_dir) is None:
                    incomplete.append(
                        PendingOrFailed(
                            label=label,
                            output_dir=str(run_dir),
                            reason=_classify_missing(row, run_dir),
                            slurm_out=str(row.get("slurm_out", "")),
                            jobid=str(row.get("jobid", "")),
                        )
                    )
    return dirs, incomplete


def _classify_missing(row: dict[str, Any], output_dir: Path) -> str:
    has_partial_predictions = output_dir.exists() and _prediction_file(output_dir, _summary_file(output_dir)) is not None
    slurm_out = Path(str(row.get("slurm_out", "")))
    if not slurm_out.exists():
        if has_partial_predictions:
            return "partial-predictions-no-summary"
        return "pending-or-not-started" if str(row.get("jobid", "")).strip() else "missing-summary-no-slurm-log"
    text = slurm_out.read_text(encoding="utf-8", errors="replace")[-20000:]
    patterns = (
        ("oom", re.compile(r"CUDA out of memory|OutOfMemoryError|oom-kill|out-of-memory", re.I)),
        ("cuda-no-kernel-image", re.compile(r"no kernel image is available for execution", re.I)),
        ("missing-dependency", re.compile(r"ModuleNotFoundError|No module named|ImportError", re.I)),
        ("cuda-extension-import", re.compile(r"undefined symbol|cannot open shared object file", re.I)),
        ("timeout", re.compile(r"TIME LIMIT|DUE TO TIME LIMIT|CANCELLED AT", re.I)),
        ("runtime-error", re.compile(r"Traceback \\(most recent call last\\)|RuntimeError|ValueError", re.I)),
    )
    for label, pattern in patterns:
        if pattern.search(text):
            return f"{label}-partial-predictions-no-summary" if has_partial_predictions else label
    if has_partial_predictions:
        return "partial-predictions-no-summary"
    return "running-or-incomplete-check-slurm"


def _discover(args: argparse.Namespace) -> tuple[list[ArtifactRun], list[PendingOrFailed]]:
    run_dirs: list[tuple[Path, Path]] = []
    incomplete: list[PendingOrFailed] = []

    for root in args.root:
        root = root.resolve()
        for run_dir in _iter_run_dirs(root):
            if _summary_file(run_dir) is None:
                incomplete.append(
                    PendingOrFailed(
                        label=_run_label(run_dir, root),
                        output_dir=str(run_dir),
                        reason="partial-artifact-no-summary",
                    )
                )
                continue
            run_dirs.append((run_dir, root))

    for manifest in args.manifest:
        dirs, missing = _run_dirs_from_manifest(manifest)
        incomplete.extend(missing)
        for run_dir in dirs:
            if _summary_file(run_dir) is not None:
                run_dirs.append((run_dir, manifest.parent.resolve()))

    seen: set[Path] = set()
    runs: list[ArtifactRun] = []
    for run_dir, root in run_dirs:
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        runs.append(_summarize_run(run_dir, root=root))
    return runs, incomplete


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _run_quality(run: ArtifactRun | None) -> str:
    return "n/a" if run is None else _fmt(run.quality, 2)


def _run_sec(run: ArtifactRun | None) -> str:
    return "n/a" if run is None else _fmt(run.sec_per_example, 2)


def _metric(run: ArtifactRun | None) -> str:
    return "n/a" if run is None else run.quality_name


def _pred_field(diff: PredictionDiff | None, same: int | None, total: int) -> str:
    if diff is None or same is None or total <= 0:
        return "n/a"
    return f"{same}/{total}"


def _pair_table(audits: Iterable[PairAudit]) -> str:
    headers = [
        "pair",
        "metric",
        "dense",
        "frontier",
        "delta",
        "frontier s/ex",
        "runtime x",
        "logical MB/hq",
        "physical MB/hq",
        "dense MB/hq",
        "logical save",
        "physical save",
        "selected",
        "active",
        "pred same",
        "judge same",
        "text same",
        "warnings",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for audit in audits:
        frontier = audit.frontier
        dense_mb = audit.dense.dense_step_mb if audit.dense else (frontier.dense_step_mb if frontier else None)
        pred = audit.prediction_diff
        lines.append(
            "| "
            + " | ".join(
                [
                    audit.key,
                    _metric(audit.dense or audit.frontier),
                    _run_quality(audit.dense),
                    _run_quality(frontier),
                    _fmt(audit.quality_delta_pct, 2),
                    _run_sec(frontier),
                    _fmt(audit.runtime_ratio, 2),
                    _fmt(frontier.step_mb if frontier else None),
                    _fmt(frontier.physical_step_mb if frontier else None),
                    _fmt(dense_mb),
                    _fmt(audit.logical_savings_pct, 1),
                    _fmt(audit.physical_savings_pct, 1),
                    _fmt(frontier.selected_tokens if frontier else None, 1),
                    _fmt(frontier.active_fraction if frontier else None),
                    _pred_field(pred, pred.same_pred_rows if pred else None, pred.comparable_pred_rows if pred else 0),
                    _pred_field(pred, pred.same_judge_rows if pred else None, pred.comparable_judge_rows if pred else 0),
                    _pred_field(
                        pred,
                        pred.same_response_rows if pred else None,
                        pred.comparable_response_rows if pred else 0,
                    ),
                    ", ".join(audit.warnings) if audit.warnings else "ok",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _incomplete_table(rows: Iterable[PendingOrFailed]) -> str:
    headers = ["label", "reason", "jobid", "output_dir", "slurm_out"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    any_row = False
    for row in rows:
        any_row = True
        lines.append(
            "| "
            + " | ".join([row.label, row.reason, row.jobid or "n/a", row.output_dir, row.slurm_out or "n/a"])
            + " |"
        )
    return "\n".join(lines) if any_row else "_No incomplete runs detected from the provided roots/manifests._"


def _longgen_table(audits: Iterable[PairAudit]) -> str:
    rows = [
        audit
        for audit in audits
        if (audit.dense and str(audit.dense.summary.get("benchmark") or "").startswith("longgenbench_sgt_"))
        or (audit.frontier and str(audit.frontier.summary.get("benchmark") or "").startswith("longgenbench_sgt_"))
    ]
    if not rows:
        return "_No LongGenBench-SGT pairs detected._"
    headers = [
        "pair",
        "dense completion",
        "frontier completion",
        "dense once",
        "frontier once",
        "dense range",
        "frontier range",
        "dense periodic",
        "frontier periodic",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for audit in rows:
        dense = audit.dense
        frontier = audit.frontier
        lines.append(
            "| "
            + " | ".join(
                [
                    audit.key,
                    _fmt(dense.longgen_completion_pct if dense else None, 2),
                    _fmt(frontier.longgen_completion_pct if frontier else None, 2),
                    _fmt(dense.longgen_once_pct if dense else None, 2),
                    _fmt(frontier.longgen_once_pct if frontier else None, 2),
                    _fmt(dense.longgen_range_pct if dense else None, 2),
                    _fmt(frontier.longgen_range_pct if frontier else None, 2),
                    _fmt(dense.longgen_periodic_pct if dense else None, 2),
                    _fmt(frontier.longgen_periodic_pct if frontier else None, 2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_outputs(args: argparse.Namespace, audits: list[PairAudit], incomplete: list[PendingOrFailed]) -> None:
    md = [
        "# Benchmark Pair Audit",
        "",
        "Dense/frontier pairs are matched by stripping `dense_` / `pagedpq_` prefixes from artifact directories.",
        "",
        "## Completed Pair Table",
        "",
        _pair_table(audits),
        "",
        "## LongGenBench SGT Metrics",
        "",
        "For SGT, `periodic` and `range` are harder smoke checks than `once`. Official SGT paper numbers require an LLM yes/no judge; these substring metrics are artifact-only checks.",
        "",
        _longgen_table(audits),
        "",
        "## Incomplete / Failed Runs",
        "",
        _incomplete_table(incomplete),
        "",
    ]
    text = "\n".join(md)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(text, encoding="utf-8")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pairs": [asdict(audit) for audit in audits],
            "incomplete": [asdict(row) for row in incomplete],
        }
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, default=[], help="Artifact root to scan recursively")
    parser.add_argument("--manifest", action="append", type=Path, default=[], help="Slurm manifest TSV to audit")
    parser.add_argument("--output-md", type=Path, default=None, help="Optional markdown report path")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any pair has warnings")
    args = parser.parse_args()

    if not args.root and not args.manifest:
        raise SystemExit("provide at least one --root or --manifest")

    runs, incomplete = _discover(args)
    audits = _audit_pairs(runs)
    _write_outputs(args, audits, incomplete)

    if args.strict and any(audit.warnings for audit in audits):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
