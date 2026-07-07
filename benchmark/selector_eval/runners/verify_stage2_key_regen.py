#!/usr/bin/env python3
"""Verify the key-domain stage-2 golden regeneration (issue #7).

Hard-asserts, per golden2_*.npz row:
  1. Every field of the FROZEN golden is present in the regenerated npz
     and bit-identical (bytes-level, NaN-safe) — the regen must be a pure
     superset.
  2. All key-domain fields (frozen E6M12 contract) are present.

Informational (printed + in the report, never fatal):
  - token-count diffs between fp-domain and key-domain masks;
  - if --study_e6m12 is given, diffs between the regenerated key-domain
    masks (composed quantized-input keys: w17/ce16) and the study's
    gold_dump_e6m12 masks (fp64-input key quantizer) — expected to agree
    except where input quantization moves a boundary token.

Also bit-compares any page_v_*.npz present in both dirs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_KEY_FIELDS = (
    "v_risk_key_exp_bits",
    "v_risk_key_mantissa_bits",
    "v_w17_fp64",
    "v_code_error_fp16",
    "v_risk_key_q_fp64",
    "v_risk_key_cutoff_q",
    "v_exact_mask_key_packed",
    "v_hi_mask_key_packed",
    "v_lo_mask_key_packed",
    "v_dropped_reads_key",
)


def _bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _mask_diff(packed_a: np.ndarray, packed_b: np.ndarray, n: int) -> int:
    ma = np.unpackbits(packed_a)[:n].astype(bool)
    mb = np.unpackbits(packed_b)[:n].astype(bool)
    return int(np.sum(ma != mb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True, help="frozen golden dir (reference fields)")
    ap.add_argument("--regen", required=True, help="regenerated dump dir (superset)")
    ap.add_argument("--study_e6m12", default="", help="study gold_dump_e6m12 dir (informational)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report: dict = {"rows": [], "pages": [], "fatal": 0}
    for ref_path in sorted(glob.glob(os.path.join(args.frozen, "golden2_*.npz"))):
        name = os.path.basename(ref_path)
        row: dict = {"row": name, "frozen_field_mismatches": [], "missing_key_fields": []}
        regen_path = os.path.join(args.regen, name)
        if not os.path.exists(regen_path):
            row["frozen_field_mismatches"].append("MISSING REGEN FILE")
            report["fatal"] += 1
            report["rows"].append(row)
            continue
        a = np.load(ref_path, allow_pickle=True)
        b = np.load(regen_path, allow_pickle=True)
        for field in a.files:
            if field not in b.files:
                row["frozen_field_mismatches"].append(f"{field}: missing")
            elif not _bit_equal(a[field], b[field]):
                row["frozen_field_mismatches"].append(f"{field}: bits differ")
        for field in _KEY_FIELDS:
            if field not in b.files:
                row["missing_key_fields"].append(field)
        if row["frozen_field_mismatches"] or row["missing_key_fields"]:
            report["fatal"] += 1
        if not row["missing_key_fields"]:
            n = int(b["context_len"])
            row["fp_vs_key_mask_token_diffs"] = {
                "exact": _mask_diff(b["v_exact_mask_packed"], b["v_exact_mask_key_packed"], n),
                "hi": _mask_diff(b["v_hi_mask_packed"], b["v_hi_mask_key_packed"], n),
                "lo": _mask_diff(b["v_lo_mask_packed"], b["v_lo_mask_key_packed"], n),
            }
            if args.study_e6m12:
                study_path = os.path.join(args.study_e6m12, name)
                if os.path.exists(study_path):
                    s = np.load(study_path, allow_pickle=True)
                    row["key_vs_study_e6m12_mask_token_diffs"] = {
                        "exact": _mask_diff(s["v_exact_mask_packed"], b["v_exact_mask_key_packed"], n),
                        "hi": _mask_diff(s["v_hi_mask_packed"], b["v_hi_mask_key_packed"], n),
                        "lo": _mask_diff(s["v_lo_mask_packed"], b["v_lo_mask_key_packed"], n),
                    }
        report["rows"].append(row)

    for ref_path in sorted(glob.glob(os.path.join(args.frozen, "page_v_*.npz"))):
        name = os.path.basename(ref_path)
        regen_path = os.path.join(args.regen, name)
        entry = {"page": name, "status": "missing"}
        if os.path.exists(regen_path):
            a = np.load(ref_path, allow_pickle=True)
            b = np.load(regen_path, allow_pickle=True)
            bad = [f for f in a.files if f not in b.files or not _bit_equal(a[f], b[f])]
            entry["status"] = "bit-identical" if not bad else f"DIFFERS: {bad}"
        report["pages"].append(entry)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    for row in report["rows"]:
        flags = []
        if row["frozen_field_mismatches"]:
            flags.append(f"FROZEN-MISMATCH {row['frozen_field_mismatches']}")
        if row["missing_key_fields"]:
            flags.append(f"MISSING-KEY-FIELDS {row['missing_key_fields']}")
        info = row.get("fp_vs_key_mask_token_diffs")
        study = row.get("key_vs_study_e6m12_mask_token_diffs")
        print(
            f"{row['row']}: {'OK' if not flags else '; '.join(flags)}"
            + (f" fp_vs_key={info}" if info else "")
            + (f" key_vs_study={study}" if study else "")
        )
    for entry in report["pages"]:
        print(f"{entry['page']}: {entry['status']}")
    print(f"fatal={report['fatal']}")
    sys.exit(0 if report["fatal"] == 0 else 1)


if __name__ == "__main__":
    main()
