#!/usr/bin/env python3
"""Risk-rank key precision study comparator (issue #7).

Pairs per-head-step rows of a full-precision reference eval against
quantized-risk-key variant evals (same inputs, same grid, same policy),
counts controller decision divergences, and classifies each divergence
against the frozen eps_band = 2e-5 equivalence class using the
REFERENCE trace's margins at the first differing decision.

Also supports a golden-gate mode comparing stage-2 golden dumps
(settled budgets + V masks + probe kinds) between a frozen reference
dump dir and per-variant dump dirs — the hard-assert criterion RTL's
validate_stage2 needs.

Outputs a JSON report; prints a human summary.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

_ENTRY_RE = re.compile(
    r"^(?P<action>stop|k|v):k(?P<ki>\d+)/v(?P<vi>\d+):dk=(?P<dk>[^:]+):dv=(?P<dv>[^:]+):"
    r"tk=(?P<tk>[^:]+):tv=(?P<tv>[^:]+)$"
)
_DEESC_RE = re.compile(
    r"^(?P<kind>kd|vd):[kv]\d+->[kv]\d+:d=(?P<d>[^:]+):t=(?P<t>[^:]+)$"
)


def _entry_margin(entry: str) -> float | None:
    """Smallest |delta - threshold| among the comparisons made at this
    trace entry (conservative: the decision could have flipped through
    the tightest comparison)."""
    m = _ENTRY_RE.match(entry)
    if m:
        margins = []
        for d_key, t_key in (("dk", "tk"), ("dv", "tv")):
            try:
                d = float(m.group(d_key))
                t = float(m.group(t_key))
            except ValueError:
                continue
            if math.isfinite(d) and math.isfinite(t):
                margins.append(abs(d - t))
        return min(margins) if margins else None
    m = _DEESC_RE.match(entry)
    if m:
        try:
            return abs(float(m.group("d")) - float(m.group("t")))
        except ValueError:
            return None
    return None


def _entry_decision(entry: str) -> str:
    """The DECISION content of a trace entry: the action and the state it
    was taken from, with the diagnostic delta values stripped. Two runs
    whose keys perturb probe deltas without crossing any threshold print
    different d= values but identical decisions."""
    m = _ENTRY_RE.match(entry)
    if m:
        return f"{m.group('action')}:k{m.group('ki')}/v{m.group('vi')}"
    m = _DEESC_RE.match(entry)
    if m:
        return entry.split(":d=", 1)[0]
    return entry


def _entry_delta_perturbation(ref_entry: str, var_entry: str) -> float:
    """Max |d_ref - d_var| across the comparisons of two same-decision
    entries: the key-induced probe-delta perturbation."""
    worst = 0.0
    ma, mb = _ENTRY_RE.match(ref_entry), _ENTRY_RE.match(var_entry)
    if ma and mb:
        for key in ("dk", "dv"):
            try:
                worst = max(worst, abs(float(ma.group(key)) - float(mb.group(key))))
            except ValueError:
                continue
        return worst
    ma, mb = _DEESC_RE.match(ref_entry), _DEESC_RE.match(var_entry)
    if ma and mb:
        try:
            worst = abs(float(ma.group("d")) - float(mb.group("d")))
        except ValueError:
            pass
    return worst


def _load_rows(run_dir: str) -> dict[tuple, dict]:
    path = os.path.join(run_dir, "per_head_joint_policy.csv")
    rows: dict[tuple, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (
                int(row["qidx"]),
                int(row["position"]),
                int(row["decode_length"]),
                int(row["head"]),
            )
            if key in rows:
                raise ValueError(f"duplicate head-step key {key} in {path}")
            rows[key] = row
    return rows


def compare_population(ref_dir: str, var_dir: str, eps_band: float) -> dict:
    ref_rows = _load_rows(ref_dir)
    var_rows = _load_rows(var_dir)
    shared = sorted(set(ref_rows) & set(var_rows))
    if len(shared) != len(ref_rows) or len(shared) != len(var_rows):
        missing = len(ref_rows) ^ len(var_rows)
        raise ValueError(
            f"row sets differ: ref={len(ref_rows)} var={len(var_rows)} shared={len(shared)}"
        )
    stats = {
        "head_steps": len(shared),
        "trace_divergences": 0,
        "in_band_divergences": 0,
        "out_of_band_divergences": 0,
        "unclassified_divergences": 0,
        "variant_extra_divergences": 0,
        "final_budget_flips": 0,
        "out_of_band_examples": [],
        "max_abs_rel_l2_shift": 0.0,
        "max_v_exact_reads_shift": 0,
        "max_probe_delta_perturbation": 0.0,
    }
    for key in shared:
        r = ref_rows[key]
        v = var_rows[key]
        rel_shift = abs(
            float(r["head_attention_relative_L2"]) - float(v["head_attention_relative_L2"])
        )
        stats["max_abs_rel_l2_shift"] = max(stats["max_abs_rel_l2_shift"], rel_shift)
        stats["max_v_exact_reads_shift"] = max(
            stats["max_v_exact_reads_shift"],
            abs(int(r["v_exact_reads"]) - int(v["v_exact_reads"])),
        )
        if (r["k_budget"], r["v_budget"]) != (v["k_budget"], v["v_budget"]):
            stats["final_budget_flips"] += 1
        ref_trace = [seg.strip() for seg in str(r["policy_trace"]).split("|")]
        var_trace = [seg.strip() for seg in str(v["policy_trace"]).split("|")]
        ref_decisions = [_entry_decision(seg) for seg in ref_trace]
        var_decisions = [_entry_decision(seg) for seg in var_trace]
        # Probe-delta perturbation over the aligned same-decision prefix:
        # the key-induced |d_ref - d_var|, directly comparable to margins
        # and eps_band.
        for i in range(min(len(ref_trace), len(var_trace))):
            if ref_decisions[i] != var_decisions[i]:
                break
            stats["max_probe_delta_perturbation"] = max(
                stats["max_probe_delta_perturbation"],
                _entry_delta_perturbation(ref_trace[i], var_trace[i]),
            )
        if ref_decisions == var_decisions:
            continue
        stats["trace_divergences"] += 1
        div_idx = next(
            (
                i
                for i in range(max(len(ref_decisions), len(var_decisions)))
                if i >= len(ref_decisions)
                or i >= len(var_decisions)
                or ref_decisions[i] != var_decisions[i]
            ),
        )
        if div_idx < len(ref_trace):
            margin = _entry_margin(ref_trace[div_idx])
            source = "reference"
        else:
            # Variant fired a probe past the reference's settled state
            # (reference's failing probe is unrecorded); classify on the
            # variant's own margin and flag the class.
            margin = _entry_margin(var_trace[div_idx])
            source = "variant_extra"
            stats["variant_extra_divergences"] += 1
        if margin is None:
            stats["unclassified_divergences"] += 1
        elif margin <= eps_band:
            stats["in_band_divergences"] += 1
        else:
            stats["out_of_band_divergences"] += 1
            if len(stats["out_of_band_examples"]) < 20:
                stats["out_of_band_examples"].append(
                    {
                        "key": list(key),
                        "margin": margin,
                        "margin_source": source,
                        "ref_entry": ref_trace[div_idx] if div_idx < len(ref_trace) else None,
                        "var_entry": var_trace[div_idx] if div_idx < len(var_trace) else None,
                        "ref_final": [r["k_budget"], r["v_budget"]],
                        "var_final": [v["k_budget"], v["v_budget"]],
                    }
                )
    return stats


def compare_goldens(ref_dump: str, var_dump: str) -> dict:
    results = {"rows": 0, "pass": 0, "fail": 0, "failures": []}
    for ref_path in sorted(glob.glob(os.path.join(ref_dump, "golden2_*.npz"))):
        name = os.path.basename(ref_path)
        var_path = os.path.join(var_dump, name)
        results["rows"] += 1
        if not os.path.exists(var_path):
            results["fail"] += 1
            results["failures"].append({"row": name, "reason": "missing variant dump"})
            continue
        a = np.load(ref_path, allow_pickle=True)
        b = np.load(var_path, allow_pickle=True)
        reasons = []
        for field in ("settled_ki", "settled_vi", "v_exact_count"):
            if int(a[field]) != int(b[field]):
                reasons.append(f"{field}: {int(a[field])} != {int(b[field])}")
        for field in ("v_exact_mask_packed", "v_hi_mask_packed", "v_lo_mask_packed"):
            if not np.array_equal(a[field], b[field]):
                reasons.append(f"{field}: mask mismatch")
        if [str(x) for x in a["probe_kind"]] != [str(x) for x in b["probe_kind"]]:
            reasons.append(
                f"probe_kind: {list(map(str, a['probe_kind']))} != {list(map(str, b['probe_kind']))}"
            )
        if reasons:
            results["fail"] += 1
            results["failures"].append({"row": name, "reason": "; ".join(reasons)})
        else:
            results["pass"] += 1
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="reference run output_dir")
    ap.add_argument(
        "--variant",
        action="append",
        default=[],
        help="TAG=RUN_DIR of a quantized-key run (repeatable)",
    )
    ap.add_argument("--golden_reference", default="", help="frozen stage-2 dump dir")
    ap.add_argument(
        "--golden_variant",
        action="append",
        default=[],
        help="TAG=DUMP_DIR of a quantized-key golden dump (repeatable)",
    )
    ap.add_argument("--eps_band", type=float, default=2e-5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report: dict = {"eps_band": args.eps_band, "population": {}, "goldens": {}}
    for spec in args.variant:
        tag, _, run_dir = spec.partition("=")
        report["population"][tag] = compare_population(args.reference, run_dir, args.eps_band)
    if args.golden_reference:
        for spec in args.golden_variant:
            tag, _, dump_dir = spec.partition("=")
            report["goldens"][tag] = compare_goldens(args.golden_reference, dump_dir)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(f"eps_band={args.eps_band}")
    for tag in sorted(report["population"]):
        s = report["population"][tag]
        print(
            f"[{tag}] steps={s['head_steps']} div={s['trace_divergences']} "
            f"in_band={s['in_band_divergences']} OUT_OF_BAND={s['out_of_band_divergences']} "
            f"var_extra={s['variant_extra_divergences']} unclass={s['unclassified_divergences']} "
            f"final_flips={s['final_budget_flips']} "
            f"max_probe_delta_pert={s['max_probe_delta_perturbation']:.3e} "
            f"max_relL2_shift={s['max_abs_rel_l2_shift']:.3e} "
            f"max_vreads_shift={s['max_v_exact_reads_shift']}"
        )
    for tag in sorted(report["goldens"]):
        g = report["goldens"][tag]
        print(f"[golden {tag}] {g['pass']}/{g['rows']} pass")
        for f in g["failures"]:
            print(f"    FAIL {f['row']}: {f['reason']}")


if __name__ == "__main__":
    main()
