#!/usr/bin/env python3
"""Statically audit benchmark wrapper defaults before Slurm submission."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.frontier_config import (
    BASE_FRONTIER_EXPORTS,
    CANONICAL_ENV_DEFAULTS,
    DIRECT_RUNTIME_DEFAULTS,
    DISALLOWED_DIAGNOSTIC_ENV_FLAGS,
    REQUIRED_CANONICAL_ENV_FLAGS,
    canonical_env_export_lines,
    canonical_wrapper_specs,
    direct_runtime_export_lines,
)


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
        label=spec.label,
        path=spec.path,
        expected_exports=dict(spec.expected_exports),
        require_sbatch=spec.require_sbatch,
    )
    for spec in canonical_wrapper_specs()
]

REQUIRED_SBATCH = {
    "--account": "zhengya98",
    "--partition": "spgpu",
    "--gpus-per-node": "1",
}


def _parse_exports(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if "frontier_canonical_env.sh" in text or "benchmark.selector_eval.frontier_config --emit-shell" in text:
        out.update(CANONICAL_ENV_DEFAULTS)
    if "frontier_direct_runtime_env.sh" in text or "benchmark.selector_eval.frontier_config --emit-direct-runtime-shell" in text:
        out.update(BASE_FRONTIER_EXPORTS)
        out.update(DIRECT_RUNTIME_DEFAULTS)
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
        "| wrapper | path | status | mode | canonical-on | diagnostic-off |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for spec, warnings, observed in rows:
        mode = observed.get("MODE") or observed.get("ATTENTION_MODE") or "n/a"
        canonical_on = sum(1 for key in REQUIRED_CANONICAL_ENV_FLAGS if observed.get(key) == "1")
        diagnostic_off = sum(1 for key in DISALLOWED_DIAGNOSTIC_ENV_FLAGS if observed.get(key, "0") == "0")
        status = "ok" if not warnings else "; ".join(warnings)
        lines.append(
            f"| {spec.label} | `{spec.path}` | {status} | {mode} | "
            f"{canonical_on}/{len(REQUIRED_CANONICAL_ENV_FLAGS)} | "
            f"{diagnostic_off}/{len(DISALLOWED_DIAGNOSTIC_ENV_FLAGS)} |"
        )
    return "\n".join(lines) + "\n"


def _generated_fragment_warnings() -> list[str]:
    expected = {
        Path("scripts/frontier_canonical_env.sh"): "\n".join(canonical_env_export_lines()) + "\n",
        Path("scripts/frontier_direct_runtime_env.sh"): "\n".join(direct_runtime_export_lines()) + "\n",
    }
    warnings: list[str] = []
    for path, text in expected.items():
        if not path.exists():
            warnings.append(f"missing-generated-fragment:{path}")
        elif path.read_text(encoding="utf-8") != text:
            warnings.append(f"stale-generated-fragment:{path}")
    return warnings


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
    fragment_warnings = _generated_fragment_warnings()
    if bad or fragment_warnings:
        print("\nWrapper audit warnings:")
        for spec, warnings in bad:
            print(f"- {spec.label}: {'; '.join(warnings)}")
        for warning in fragment_warnings:
            print(f"- {warning}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
