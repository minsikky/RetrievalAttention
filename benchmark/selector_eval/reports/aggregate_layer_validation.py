#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable


FLOAT_FIELDS = [
    "attn_concat_output_relative_l2",
    "attn_concat_output_cosine",
    "attn_o_proj_output_relative_l2",
    "layer_output_output_relative_l2",
    "layer_output_output_cosine",
    "mean_head_attention_mass",
    "min_head_attention_mass",
    "mean_exact_KV_MB_per_head",
    "mean_exact_key_MB_per_head",
    "mean_selected_value_MB_per_head",
    "mean_selected_value_exact_tokens",
    "mean_selected_value_exact_selected_mass",
    "mean_tail_estimator_MB_per_head",
    "mean_confidence_MB_per_head",
    "mean_online_update_MB_per_token_per_head",
    "mean_online_update_cumulative_MB_per_kv_head",
    "mean_step_MB_per_head",
    "max_step_MB_per_head",
    "query_seconds",
]

PER_HEAD_FLOAT_FIELDS = [
    "attention_mass",
    "head_attention_relative_L2",
    "head_attention_cosine",
    "selected_tokens",
    "exact_KV_MB_per_query",
    "selector_MB_per_query",
    "tail_estimator_MB_per_query",
    "confidence_MB_per_query",
    "step_MB_per_query",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        row["_manifest"] = str(path)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def to_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", None}:  # type: ignore[comparison-overlap]
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return "nan"
        if abs(value) >= 100:
            return f"{value:.3f}"
        if abs(value) >= 10:
            return f"{value:.3f}"
        return f"{value:.6f}"
    return str(value)


def normalize_config(label: str) -> str:
    name = str(label)
    name = re.sub(r"^val_q36_", "", name)
    name = re.sub(r"^val_q288c\d+_", "", name)
    name = re.sub(r"^q288c\d+_", "", name)
    name = re.sub(r"^cpu_val_q36_", "", name)
    name = re.sub(r"^val_", "", name)
    name = re.sub(r"_(gpu|cpu)_v\d+$", "", name)
    return name


def slurm_status(jobids: Iterable[str]) -> dict[str, str]:
    ids = [str(j) for j in jobids if str(j)]
    status = {j: "UNKNOWN" for j in ids}
    if not ids:
        return status
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-j", ",".join(ids), "-o", "%i\t%T"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                status[parts[0].strip()] = parts[1].strip()
    except Exception:
        pass
    missing = [j for j, state in status.items() if state == "UNKNOWN"]
    if missing:
        try:
            out = subprocess.check_output(
                [
                    "sacct",
                    "-n",
                    "-P",
                    "-j",
                    ",".join(missing),
                    "--format=JobID,State",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                job, _, state = line.partition("|")
                base = job.split(".", 1)[0].strip()
                if base in status and state.strip():
                    status[base] = state.strip()
        except Exception:
            pass
    return status


def unique_count(rows: list[dict[str, str]], key: str) -> int:
    return len({row.get(key, "") for row in rows if row.get(key, "") != ""})


def max_field(rows: list[dict[str, str]], key: str) -> tuple[float | None, dict[str, str] | None]:
    best_value: float | None = None
    best_row: dict[str, str] | None = None
    for row in rows:
        value = to_float(row, key)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_row = row
    return best_value, best_row


def min_field(rows: list[dict[str, str]], key: str) -> tuple[float | None, dict[str, str] | None]:
    best_value: float | None = None
    best_row: dict[str, str] | None = None
    for row in rows:
        value = to_float(row, key)
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_value = value
            best_row = row
    return best_value, best_row


def mean_field(rows: list[dict[str, str]], key: str) -> float | None:
    vals = [v for row in rows if (v := to_float(row, key)) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize_rows(
    *,
    label: str,
    jobids: list[str],
    output_dirs: list[str],
    tier: str,
    status_values: list[str],
    layer_rows: list[dict[str, str]],
    per_head_rows: list[dict[str, str]],
    summary: dict | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "label": label,
        "config": normalize_config(label),
        "tier": tier,
        "jobids": ",".join(jobids),
        "output_dirs": ",".join(output_dirs),
        "job_status": ",".join(status_values),
        "has_summary": bool(summary),
        "layer_rows": len(layer_rows),
        "per_head_rows": len(per_head_rows),
        "decode_length_count": unique_count(layer_rows, "decode_length"),
        "qidx_count": unique_count(layer_rows, "qidx"),
        "position_count": unique_count(layer_rows, "position"),
    }
    if summary:
        for key in [
            "algorithm",
            "tail_mode",
            "selector_mode",
            "selected_value_mode",
            "selected_value_exact_rule",
            "selected_value_residual_correction",
        ]:
            if key in summary:
                result[key] = summary[key]
    for key in FLOAT_FIELDS:
        result[f"{key}_mean_rows"] = mean_field(layer_rows, key)
        max_value, max_row = max_field(layer_rows, key)
        min_value, min_row = min_field(layer_rows, key)
        result[f"{key}_max_rows"] = max_value
        result[f"{key}_min_rows"] = min_value
        if max_row is not None:
            result[f"{key}_max_decode_length"] = max_row.get("decode_length", "")
            result[f"{key}_max_position"] = max_row.get("position", "")
            result[f"{key}_max_qidx"] = max_row.get("qidx", "")
        if min_row is not None:
            result[f"{key}_min_decode_length"] = min_row.get("decode_length", "")
            result[f"{key}_min_position"] = min_row.get("position", "")
            result[f"{key}_min_qidx"] = min_row.get("qidx", "")
    for key in PER_HEAD_FLOAT_FIELDS:
        result[f"per_head_{key}_mean"] = mean_field(per_head_rows, key)
        max_value, max_row = max_field(per_head_rows, key)
        min_value, min_row = min_field(per_head_rows, key)
        result[f"per_head_{key}_max"] = max_value
        result[f"per_head_{key}_min"] = min_value
        if max_row is not None:
            result[f"per_head_{key}_max_decode_length"] = max_row.get("decode_length", "")
            result[f"per_head_{key}_max_head"] = max_row.get("head", "")
        if min_row is not None:
            result[f"per_head_{key}_min_decode_length"] = min_row.get("decode_length", "")
            result[f"per_head_{key}_min_head"] = min_row.get("head", "")
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    preferred = [
        "label",
        "config",
        "tier",
        "jobids",
        "job_status",
        "has_summary",
        "layer_rows",
        "decode_length_count",
        "qidx_count",
        "position_count",
        "attn_concat_output_relative_l2_max_rows",
        "layer_output_output_relative_l2_max_rows",
        "layer_output_output_cosine_min_rows",
        "mean_step_MB_per_head_max_rows",
        "mean_exact_KV_MB_per_head_max_rows",
        "mean_selected_value_MB_per_head_max_rows",
        "mean_tail_estimator_MB_per_head_max_rows",
        "mean_online_update_MB_per_token_per_head_max_rows",
        "per_head_head_attention_relative_L2_max",
        "per_head_attention_mass_min",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, object]], title: str) -> None:
    columns = [
        "config",
        "tier",
        "job_status",
        "layer_rows",
        "decode_length_count",
        "position_count",
        "attn_concat_output_relative_l2_max_rows",
        "layer_output_output_relative_l2_max_rows",
        "layer_output_output_cosine_min_rows",
        "mean_step_MB_per_head_max_rows",
        "mean_exact_KV_MB_per_head_max_rows",
        "mean_selected_value_MB_per_head_max_rows",
        "mean_tail_estimator_MB_per_head_max_rows",
        "per_head_head_attention_relative_L2_max",
        "per_head_attention_mass_min",
    ]
    present = [col for col in columns if any(col in row for row in rows)]
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(present) + " |\n")
        f.write("| " + " | ".join("---" for _ in present) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(col)) for col in present) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate layer-quality validation outputs referenced by Slurm manifests."
    )
    parser.add_argument("manifests", nargs="+", help="Manifest TSV files with label, jobid, output_dir.")
    parser.add_argument("--out-dir", required=True, help="Directory for aggregate CSV/Markdown reports.")
    parser.add_argument("--title", default="Layer Validation Aggregate")
    args = parser.parse_args()

    manifest_rows: list[dict[str, str]] = []
    for manifest in args.manifests:
        manifest_rows.extend(read_manifest(Path(manifest)))
    job_states = slurm_status([row.get("jobid", "") for row in manifest_rows])

    detail_rows: list[dict[str, object]] = []
    grouped: dict[str, dict[str, object]] = {}
    group_layer_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    group_head_rows: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in manifest_rows:
        label = row.get("label", "")
        output_dir = Path(row.get("output_dir", ""))
        jobid = row.get("jobid", "")
        tier = row.get("tier", "")
        summary = read_json(output_dir / "summary.json")
        layer_rows = read_csv(output_dir / "layer_quality.csv")
        per_head_rows = read_csv(output_dir / "per_head_quality.csv")
        if summary:
            artifact_status = "SUMMARY"
        elif output_dir.exists():
            artifact_status = "OUTPUT_DIR"
        else:
            artifact_status = "NO_OUTPUT"
        status = f"{job_states.get(jobid, 'UNKNOWN')}:{artifact_status}"
        detail_rows.append(
            summarize_rows(
                label=label,
                jobids=[jobid],
                output_dirs=[str(output_dir)],
                tier=tier,
                status_values=[status],
                layer_rows=layer_rows,
                per_head_rows=per_head_rows,
                summary=summary,
            )
        )
        config = normalize_config(label)
        grouped.setdefault(
            config,
            {
                "label": config,
                "jobids": [],
                "output_dirs": [],
                "tiers": [],
                "statuses": [],
                "summary": summary,
            },
        )
        grouped[config]["jobids"].append(jobid)  # type: ignore[index]
        grouped[config]["output_dirs"].append(str(output_dir))  # type: ignore[index]
        grouped[config]["tiers"].append(tier)  # type: ignore[index]
        grouped[config]["statuses"].append(status)  # type: ignore[index]
        if not grouped[config].get("summary") and summary:
            grouped[config]["summary"] = summary
        group_layer_rows[config].extend(layer_rows)
        group_head_rows[config].extend(per_head_rows)

    group_rows: list[dict[str, object]] = []
    for config, meta in sorted(grouped.items()):
        group_rows.append(
            summarize_rows(
                label=str(meta["label"]),
                jobids=list(meta["jobids"]),  # type: ignore[arg-type]
                output_dirs=list(meta["output_dirs"]),  # type: ignore[arg-type]
                tier=",".join(sorted(set(meta["tiers"]))),  # type: ignore[arg-type]
                status_values=list(meta["statuses"]),  # type: ignore[arg-type]
                layer_rows=group_layer_rows[config],
                per_head_rows=group_head_rows[config],
                summary=meta.get("summary") if isinstance(meta.get("summary"), dict) else None,
            )
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "validation_detail.csv", detail_rows)
    write_csv(out_dir / "validation_by_config.csv", group_rows)
    write_markdown(out_dir / "validation_detail.md", detail_rows, f"{args.title}: Per Job")
    write_markdown(out_dir / "validation_by_config.md", group_rows, f"{args.title}: By Config")
    print(f"[aggregate_layer_validation] wrote {out_dir}")


if __name__ == "__main__":
    main()
