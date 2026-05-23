#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames and {"label", "output_dir"}.issubset(set(reader.fieldnames)):
            for row in reader:
                rows.append((str(row["label"]), Path(row["output_dir"])))
            return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0] != "label":
                rows.append((parts[0], Path(parts[2])))
    return rows


def find_summary(output_dir: Path) -> Path | None:
    candidates = [output_dir / "summary.json"]
    summary_dir = output_dir / "summary"
    if summary_dir.exists():
        candidates.extend(sorted(summary_dir.glob("*.json")))
    for path in candidates:
        if path.exists():
            return path
    return None


def profile_from_summary(summary: dict[str, Any]) -> dict[str, float | int | str] | None:
    if isinstance(summary.get("execution_profile"), dict):
        profile = summary["execution_profile"]
        examples = int(profile.get("examples", summary.get("num_examples", 0)) or 0)
        generated_total = float(profile.get("generated_tokens", 0.0) or 0.0)
        return {
            "examples": examples,
            "prompt_tokens": float(summary.get("avg_used_prompt_tokens", summary.get("mean_prompt_tokens", 0.0)) or 0.0),
            "generated_tokens": float(
                summary.get("avg_generated_tokens", summary.get("mean_generated_tokens", generated_total / max(1, examples)))
                or 0.0
            ),
            "prefill_sec": float(profile.get("mean_prefill_forward_seconds", 0.0)),
            "decode_sec": float(profile.get("mean_decode_forward_seconds", 0.0)),
            "total_sec": float(profile.get("mean_generation_seconds", summary.get("avg_generation_sec", 0.0))),
            "prefill_frac": float(profile.get("prefill_fraction_of_generation", 0.0)),
            "decode_frac": float(profile.get("decode_fraction_of_generation", 0.0)),
            "decode_ms_tok": float(profile.get("mean_decode_ms_per_generated_token", 0.0)),
        }

    if "mean_stream_prefill_seconds" in summary and "mean_stream_decode_seconds" in summary:
        total = float(summary.get("mean_stream_total_seconds", 0.0))
        return {
            "examples": int(summary.get("samples", 0) or 0),
            "prompt_tokens": float(summary.get("mean_prompt_tokens", 0.0)),
            "generated_tokens": float(summary.get("mean_generated_tokens", 0.0)),
            "prefill_sec": float(summary.get("mean_stream_prefill_seconds", 0.0)),
            "decode_sec": float(summary.get("mean_stream_decode_seconds", 0.0)),
            "total_sec": total,
            "prefill_frac": float(summary.get("prefill_fraction_of_stream_time", 0.0)),
            "decode_frac": float(summary.get("decode_fraction_of_stream_time", 0.0)),
            "decode_ms_tok": float(summary.get("decode_ms_per_generated_token", 0.0)),
        }
    return None


def fmt(value: float | int | str, digits: int = 3) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report dense prefill/decode execution latency breakdowns.")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    rows = []
    for label, output_dir in load_manifest(args.manifest):
        summary_path = find_summary(output_dir)
        if summary_path is None:
            rows.append({"label": label, "status": "missing"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        profile = profile_from_summary(summary)
        if profile is None:
            rows.append({"label": label, "status": "no_profile"})
            continue
        rows.append({"label": label, "status": "ok", **profile})

    headers = [
        "label",
        "status",
        "examples",
        "prompt_tokens",
        "generated_tokens",
        "prefill_sec/ex",
        "decode_sec/ex",
        "total_sec/ex",
        "prefill_%",
        "decode_%",
        "decode_ms/token",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        if row["status"] != "ok":
            values = [row["label"], row["status"], "", "", "", "", "", "", "", "", ""]
        else:
            values = [
                str(row["label"]),
                "ok",
                fmt(row["examples"]),
                fmt(row["prompt_tokens"], 1),
                fmt(row["generated_tokens"], 1),
                fmt(row["prefill_sec"]),
                fmt(row["decode_sec"]),
                fmt(row["total_sec"]),
                fmt(100.0 * float(row["prefill_frac"]), 1),
                fmt(100.0 * float(row["decode_frac"]), 1),
                fmt(row["decode_ms_tok"], 2),
            ]
        print("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
