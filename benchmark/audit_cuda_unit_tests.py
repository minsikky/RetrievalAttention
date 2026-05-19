#!/usr/bin/env python3
"""Audit CUDA unit-test Slurm artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class UnitRun:
    label: str
    jobid: str
    path: Path
    status: str
    return_code: int | None
    elapsed_seconds: int | None
    tests: int | None
    readiness: str


def _manifest_runs(path: Path) -> list[tuple[str, str, Path]]:
    runs: list[tuple[str, str, Path]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames and {"label", "jobid", "output_dir"}.issubset(set(reader.fieldnames)):
            for row in reader:
                label = str(row.get("label", "")).strip()
                jobid = str(row.get("jobid", "")).strip()
                output_dir = str(row.get("output_dir", "")).strip()
                if label and jobid and output_dir:
                    runs.append((label, jobid, Path(output_dir)))
            return runs
    return runs


def _audit(label: str, jobid: str, output_dir: Path) -> UnitRun:
    summary_path = output_dir / "summary.json" if output_dir.is_dir() else output_dir
    if not summary_path.exists():
        return UnitRun(
            label=label,
            jobid=jobid,
            path=summary_path,
            status="missing",
            return_code=None,
            elapsed_seconds=None,
            tests=None,
            readiness="missing-summary",
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    status = str(payload.get("status", "missing"))
    return_code = payload.get("return_code")
    tests = payload.get("tests")
    observed_jobid = str(payload.get("slurm_job_id", ""))
    if observed_jobid != str(jobid):
        readiness = f"stale-summary:{observed_jobid or 'missing'}"
    else:
        readiness = "ok" if status == "passed" and return_code == 0 else "failed"
    return UnitRun(
        label=label,
        jobid=jobid,
        path=summary_path,
        status=status,
        return_code=int(return_code) if isinstance(return_code, int) else None,
        elapsed_seconds=int(payload["elapsed_seconds"]) if isinstance(payload.get("elapsed_seconds"), int) else None,
        tests=len(tests) if isinstance(tests, list) else None,
        readiness=readiness,
    )


def _fmt(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _markdown(runs: Iterable[UnitRun]) -> str:
    headers = ["label", "jobid", "status", "return code", "elapsed s", "tests", "summary", "readiness"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for run in runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    run.label,
                    run.jobid,
                    run.status,
                    _fmt(run.return_code),
                    _fmt(run.elapsed_seconds),
                    _fmt(run.tests),
                    f"`{run.path}`",
                    run.readiness,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless all runs are ready.")
    args = parser.parse_args()

    runs: list[UnitRun] = []
    for manifest in args.manifest:
        for label, jobid, output_dir in _manifest_runs(manifest):
            runs.append(_audit(label, jobid, output_dir))
    if not runs:
        raise SystemExit("provide at least one manifest with label/output_dir columns")

    text = _markdown(runs)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")

    bad = [run for run in runs if run.readiness != "ok"]
    if bad:
        print("\nCUDA unit readiness warnings:")
        for run in bad:
            print(f"- {run.label}: {run.readiness}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
