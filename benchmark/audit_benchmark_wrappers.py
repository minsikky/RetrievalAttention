#!/usr/bin/env python3
"""Statically audit benchmark wrapper defaults before Slurm submission."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


EXPORT_RE = re.compile(r'^export\s+([A-Za-z_][A-Za-z0-9_]*)="\$\{\1:-(.*)\}"\s*$')
SBATCH_RE = re.compile(r"^#SBATCH\s+(--[A-Za-z0-9_-]+)=(.*)\s*$")


@dataclass(frozen=True)
class WrapperSpec:
    label: str
    path: Path
    expected_exports: dict[str, str]


SPECS = [
    WrapperSpec(
        label="frontier_ruler",
        path=Path("scripts/run_frontier_ruler_batched_one.sh"),
        expected_exports={
            "MODE": "pagedpq_batched",
            "SELECTOR_MODE": "fullscan",
            "SELECTOR_BACKEND": "cuda_ext",
            "ONLINE_CONFIDENCE_RULE": "pq_ranked_mass_budget",
            "TAIL_MODE": "vpq_value",
            "TAIL_BLEND": "1.0",
            "NATIVE_DECODE_TAIL": "1",
            "SELECTED_VALUE_MODE": "vpq_value",
            "SELECTED_VALUE_EXACT_RULE": "selector_rank",
            "SELECTED_VALUE_EXACT_TOP": "256",
            "SELECTED_VALUE_MIN_EXACT_TOP": "0",
            "SELECTED_VALUE_MAX_EXACT_TOP": "0",
            "PREFILL_SELECTOR_BACKEND": "torch_matmul",
            "PREFILL_ATTENTION_BACKEND": "native",
            "INDEX_BUILD_BACKEND": "torch_gpu",
        },
    ),
    WrapperSpec(
        label="frontier_longbench",
        path=Path("scripts/run_frontier_longbench_v2_one.sh"),
        expected_exports={
            "ATTENTION_MODE": "pagedpq",
            "SELECTOR_MODE": "fullscan",
            "SELECTOR_BACKEND": "cuda_ext",
            "ONLINE_CONFIDENCE_RULE": "pq_ranked_mass_budget",
            "TAIL_MODE": "vpq_value",
            "TAIL_BLEND": "1.0",
            "NATIVE_DECODE_TAIL": "1",
            "SELECTED_VALUE_MODE": "vpq_value",
            "SELECTED_VALUE_EXACT_RULE": "selector_rank",
            "SELECTED_VALUE_EXACT_TOP": "256",
            "SELECTED_VALUE_MIN_EXACT_TOP": "0",
            "SELECTED_VALUE_MAX_EXACT_TOP": "0",
            "PREFILL_SELECTOR_BACKEND": "torch_matmul",
            "PREFILL_ATTENTION_BACKEND": "native",
            "INDEX_BUILD_BACKEND": "torch_gpu",
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
]

REQUIRED_SBATCH = {
    "--account": "zhengya98",
    "--partition": "spgpu",
    "--gpus-per-node": "1",
}


def _parse_exports(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        match = EXPORT_RE.match(line.strip())
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
