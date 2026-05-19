#!/usr/bin/env python3
"""Summarize dense/frontier benchmark artifacts and flag readiness issues."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import csv


MEAN_COST_FIELDS = (
    "mean_step_MB_per_head_query",
    "mean_total_MB_per_head_query",
    "mean_selector_MB_per_head_query",
    "mean_exact_KV_MB_per_head_query",
    "mean_tail_estimator_MB_per_head_query",
    "mean_update_MB_per_head_query",
    "online_update_MB_per_head_query",
    "mean_selected_tokens",
)

SUM_COST_FIELDS = (
    "passthrough_attention_calls",
    "passthrough_attention_calls_total",
    "native_selector_seconds",
    "native_selector_seconds_total",
    "native_attention_seconds",
    "native_attention_seconds_total",
    "patched_attention_seconds",
    "patched_attention_seconds_total",
)


@dataclass
class RunSummary:
    label: str
    kind: str
    path: Path
    mode: str
    quality: float | None
    quality_name: str
    examples: int | None
    seconds_per_example: float | None
    step_mb: float | None
    selector_mb: float | None
    exact_kv_mb: float | None
    tail_mb: float | None
    update_mb: float | None
    selected_tokens: float | None
    passthrough: float | None
    warnings: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _run_arg(value: str) -> tuple[str, Path]:
    if ":" not in value:
        path = Path(value)
        return path.name, path
    label, raw_path = value.split(":", 1)
    if not label:
        raise argparse.ArgumentTypeError("run label must be non-empty")
    return label, Path(raw_path)


def _manifest_runs(path: Path) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames and {"label", "output_dir"}.issubset(set(reader.fieldnames)):
            for row in reader:
                label = str(row.get("label", "")).strip()
                output_dir = str(row.get("output_dir", "")).strip()
                if label and output_dir:
                    runs.append((label, Path(output_dir)))
            return runs

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] == "label":
                continue
            if len(row) >= 3:
                runs.append((row[0], Path(row[2])))
    return runs


def _detect_summaries(label: str, path: Path) -> list[tuple[str, str, Path]]:
    if path.is_file():
        kind = "longbench" if path.name == "summary.json" else "ruler"
        return [(label, kind, path)]
    if (path / "summary.json").exists():
        return [(label, "longbench", path)]
    summary_dir = path / "summary"
    if summary_dir.is_dir():
        summaries = sorted(summary_dir.glob("*.json"))
        out: list[tuple[str, str, Path]] = []
        for summary in summaries:
            suffix = summary.stem
            run_label = label if suffix in label else f"{label}/{suffix}"
            out.append((run_label, "ruler", summary))
        return out
    return []


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _examples_from_nulls(value: Any) -> int | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    _, total = value.split("/", 1)
    try:
        return int(total)
    except ValueError:
        return None


def _ruler_eval_csv(path: Path) -> dict[str, str]:
    """Read the post-eval RULER summary CSV next to a pre-eval JSON summary."""
    pred_summary = path.parent.parent / "pred" / f"summary-{path.stem}.csv"
    if not pred_summary.exists():
        return {}
    out: dict[str, str] = {}
    with pred_summary.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0]:
                out[str(row[0])] = str(row[1])
    return out


def _cost_from_aggregate(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("cost_proxy_aggregate")
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _cost_from_layers(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("cost_proxy")
    if not isinstance(raw, dict) or not raw:
        return {}
    layers = [v for v in raw.values() if isinstance(v, dict)]
    if not layers:
        return {}

    out: dict[str, float] = {}
    for field in MEAN_COST_FIELDS:
        vals = [_as_float(layer.get(field)) for layer in layers]
        vals = [v for v in vals if v is not None]
        if vals:
            out[field] = sum(vals) / len(vals)
    for field in SUM_COST_FIELDS:
        vals = [_as_float(layer.get(field)) for layer in layers]
        vals = [v for v in vals if v is not None]
        if vals:
            out[field] = sum(vals)
    return out


def _cost(payload: dict[str, Any]) -> dict[str, float]:
    return _cost_from_aggregate(payload) or _cost_from_layers(payload)


def _passthrough(cost: dict[str, float]) -> float | None:
    if "passthrough_attention_calls_total" in cost:
        return cost["passthrough_attention_calls_total"]
    if "passthrough_attention_calls" in cost:
        return cost["passthrough_attention_calls"]
    return None


def _cost_value(cost: dict[str, float], *names: str) -> float | None:
    for name in names:
        if name in cost:
            return cost[name]
    return None


def _ruler_summary(label: str, path: Path) -> RunSummary:
    payload = _load_json(path)
    eval_csv = _ruler_eval_csv(path)
    cost = _cost(payload)
    mode = str(payload.get("mode", ""))
    warnings = _common_warnings(payload, cost, mode)
    examples = _examples_from_nulls(payload.get("nulls"))
    if examples is None:
        examples = _examples_from_nulls(eval_csv.get("Nulls"))
    if examples is None:
        samples = payload.get("samples")
        examples = len(samples) if isinstance(samples, list) else None
    quality = _as_float(payload.get("score"))
    if quality is None:
        quality = _as_float(eval_csv.get("Score"))
    return RunSummary(
        label=label,
        kind="ruler",
        path=path,
        mode=mode,
        quality=quality,
        quality_name="score",
        examples=examples,
        seconds_per_example=_as_float(payload.get("mean_stream_total_seconds")),
        step_mb=cost.get("mean_step_MB_per_head_query"),
        selector_mb=cost.get("mean_selector_MB_per_head_query"),
        exact_kv_mb=cost.get("mean_exact_KV_MB_per_head_query"),
        tail_mb=cost.get("mean_tail_estimator_MB_per_head_query"),
        update_mb=_cost_value(cost, "mean_update_MB_per_head_query", "online_update_MB_per_head_query"),
        selected_tokens=cost.get("mean_selected_tokens"),
        passthrough=_passthrough(cost),
        warnings=warnings,
    )


def _longbench_summary(label: str, path: Path) -> RunSummary:
    summary_path = path / "summary.json" if path.is_dir() else path
    payload = _load_json(summary_path)
    cost = _cost(payload)
    mode = str(payload.get("attention_mode", ""))
    warnings = _common_warnings(payload, cost, mode)
    return RunSummary(
        label=label,
        kind="longbench-v2",
        path=summary_path,
        mode=mode,
        quality=_as_float(payload.get("accuracy_pct")),
        quality_name="accuracy_pct",
        examples=int(payload["num_examples"]) if isinstance(payload.get("num_examples"), int) else None,
        seconds_per_example=_as_float(payload.get("avg_generation_sec")),
        step_mb=cost.get("mean_step_MB_per_head_query"),
        selector_mb=cost.get("mean_selector_MB_per_head_query"),
        exact_kv_mb=cost.get("mean_exact_KV_MB_per_head_query"),
        tail_mb=cost.get("mean_tail_estimator_MB_per_head_query"),
        update_mb=_cost_value(cost, "mean_update_MB_per_head_query", "online_update_MB_per_head_query"),
        selected_tokens=cost.get("mean_selected_tokens"),
        passthrough=_passthrough(cost),
        warnings=warnings,
    )


def _common_warnings(payload: dict[str, Any], cost: dict[str, float], mode: str) -> list[str]:
    warnings: list[str] = []
    is_frontier = mode in {"pagedpq", "pagedpq_batched", "pagedpq_stream"}
    if mode == "pagedpq_stream":
        warnings.append("streaming-prefill")
    if is_frontier:
        config = payload.get("pagedpq_config") if isinstance(payload.get("pagedpq_config"), dict) else {}
        if payload.get("diagnose_dense_reference") is True:
            warnings.append("diagnostic-dense-reference")
        if not cost:
            warnings.append("missing-cost")
        # Decode-only benchmark mode intentionally uses dense prefill passthrough.
        if _passthrough(cost) not in (None, 0.0) and config.get("approx_prefill") is not False:
            warnings.append("passthrough")
        if not isinstance(payload.get("pagedpq_config"), dict):
            warnings.append("missing-config")
        if config:
            if config.get("selector_backend") != "cuda_ext":
                warnings.append("non-cuda-selector")
            if config.get("index_build_backend") != "torch_gpu":
                warnings.append("non-gpu-index-build")
            prefill_backend = config.get("prefill_selector_backend")
            if prefill_backend not in {None, "torch_matmul", "cuda_ext", "native", "torch_lut", "torch_lut_fp16"}:
                warnings.append("non-gpu-prefill-selector")
            if config.get("selected_value_mode") != "vpq_value":
                warnings.append("selected-v-not-compressed")
            if (_as_float(config.get("selected_value_min_exact_top")) or 0.0) > 0.0:
                warnings.append("selected-v-min-exact-fallback")
            if (_as_float(config.get("selected_value_max_exact_top")) or 0.0) > 0.0:
                warnings.append("selected-v-max-exact-fallback")
            if config.get("online_confidence_rule") in {None, "none"}:
                warnings.append("no-confidence-rule")
            if config.get("ranked_confidence_cost_mode") == "exact":
                warnings.append("sync-cost-accounting")
    return warnings


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _row(run: RunSummary) -> list[str]:
    return [
        run.label,
        run.kind,
        run.mode or "n/a",
        _fmt(run.quality, 2),
        str(run.examples) if run.examples is not None else "n/a",
        _fmt(run.seconds_per_example, 2),
        _fmt(run.step_mb),
        _fmt(run.selector_mb),
        _fmt(run.exact_kv_mb),
        _fmt(run.tail_mb),
        _fmt(run.update_mb, 6),
        _fmt(run.selected_tokens, 1),
        _fmt(run.passthrough, 0),
        ", ".join(run.warnings) if run.warnings else "ok",
    ]


def _markdown_table(runs: Iterable[RunSummary]) -> str:
    headers = [
        "label",
        "kind",
        "mode",
        "quality",
        "n",
        "sec/ex",
        "step MB/hq",
        "selector",
        "exact KV",
        "tail",
        "update",
        "selected",
        "passthrough",
        "readiness",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for run in runs:
        lines.append("| " + " | ".join(_row(run)) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler", action="append", type=_run_arg, default=[], help="LABEL:summary.json")
    parser.add_argument("--longbench", action="append", type=_run_arg, default=[], help="LABEL:run_dir_or_summary.json")
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help="Slurm manifest with label/jobid/output_dir columns; summaries are auto-detected.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any readiness warning is present.")
    args = parser.parse_args()

    runs: list[RunSummary] = []
    for label, path in args.ruler:
        runs.append(_ruler_summary(label, path))
    for label, path in args.longbench:
        runs.append(_longbench_summary(label, path))
    for manifest in args.manifest:
        for label, output_dir in _manifest_runs(manifest):
            detected = _detect_summaries(label, output_dir)
            if not detected:
                runs.append(
                    RunSummary(
                        label=label,
                        kind="missing",
                        path=output_dir,
                        mode="n/a",
                        quality=None,
                        quality_name="n/a",
                        examples=None,
                        seconds_per_example=None,
                        step_mb=None,
                        selector_mb=None,
                        exact_kv_mb=None,
                        tail_mb=None,
                        update_mb=None,
                        selected_tokens=None,
                        passthrough=None,
                        warnings=["missing-summary"],
                    )
                )
                continue
            for run_label, kind, summary_path in detected:
                if kind == "longbench":
                    runs.append(_longbench_summary(run_label, summary_path))
                elif kind == "ruler":
                    runs.append(_ruler_summary(run_label, summary_path))
    if not runs:
        raise SystemExit("provide at least one --ruler or --longbench run")

    text = _markdown_table(runs) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")

    bad = [run for run in runs if run.warnings]
    if bad:
        print("\nReadiness warnings:")
        for run in bad:
            print(f"- {run.label}: {', '.join(run.warnings)}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
