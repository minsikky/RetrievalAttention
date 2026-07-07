#!/usr/bin/env python3
"""Verify the key-domain stage-2 golden regeneration (issue #7).

The next regen legitimately CHANGES the item-6 key-domain outputs (new
E6M12 floor + fp16 commit test) and ADDS the item-7 Vcorr fields. The
hard-assert classes are therefore split:

Hard-asserts (fatal), per golden2_*.npz row:
  1. Every frozen field EXCEPT the item-6 key-domain OUTPUT fields (which
     may move) is present in the regen and bit-identical (bytes-level).
     The item-6 INPUT records (v_w17_fp64, v_code_error_fp16) and the
     constant window params stay in the must-match class.
  2. All key-domain fields and all item-7 fields (Changes 2b, 3a-3d) are
     present in the regen. The 3d kmove fields are required only when the
     frozen row's trace de-escalates on the K axis (probe_kind contains kd).

Informational (printed + in the report, never fatal):
  - per-row token-count diffs (frozen vs regen) for the moved key-domain
    masks, and fp-vs-key mask diffs within the regen;
  - across all rows/records, max |dv_hw - dv_ref| and the per-row worst
    (RTL operand-domain amendment); records with |dv_hw - dv_ref| > 1e-5
    (= eps_band/2) are FLAGGED prominently;
  - if --study_e6m12 is given, key-vs-study mask diffs.

Also bit-compares any page_v_*.npz present in both dirs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# Item-6 key-domain OUTPUT fields the regen is ALLOWED to move (new floor +
# fp16 commit). Everything else frozen must stay bit-identical — including
# the input records v_w17_fp64 / v_code_error_fp16 and the constant
# v_risk_key_exp_bits / v_risk_key_mantissa_bits.
_MOVED_KEY_FIELDS = (
    "v_risk_key_q_fp64",
    "v_risk_key_cutoff_q",
    "v_exact_mask_key_packed",
    "v_hi_mask_key_packed",
    "v_lo_mask_key_packed",
    "v_dropped_reads_key",
)

# Frozen key-domain fields that must still be present (superset guarantee).
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

# Item-7 fields that must be present in every regen row (Changes 2b, 3a-3c).
_ITEM7_FIELDS = (
    "v_commit_mask_packed",
    "v_int8_err_fp16",
    "vcorr_probe_record_kind",
    "vcorr_probe_ki",
    "vcorr_probe_vi_lo",
    "vcorr_probe_vi_hi",
    "vcorr_dv_ref_fp64",
    "vcorr_dv_hw_fp64",
    "vcorr_dv_trace_delta",
    "vcorr_acc_marginal_ref",
    "vcorr_acc_hiboundary_ref",
    "vcorr_acc_marginal_hw",
    "vcorr_acc_hiboundary_hw",
    "vcorr_marginal_tokens",
    "vcorr_marginal_p",
    "vcorr_marginal_offsets",
    "vcorr_hiboundary_tokens",
    "vcorr_hiboundary_p",
    "vcorr_hiboundary_offsets",
    "vcorr_settled_acc_ref",
    "vcorr_settled_acc_hw",
    "vexact_tokens",
    "vexact_v_fp16",
    "vexact_int8_codes",
    "vexact_int8_scale",
    "vexact_int8_scale_fp64",
    "vexact_int8_err_fp16",
    "vexact_commit",
    "vexact_residual_codes",
    "vexact_residual_scale",
    "vexact_residual_scale_fp64",
    "vexact_recon_max_abs_err",
)

# Item-7 kmove fields, required only when the row's trace has a kd entry.
_KMOVE_FIELDS = (
    "kmove_ki_from",
    "kmove_ki_to",
    "kmove_vi",
    "kmove_den_old",
    "kmove_den_new",
    "kmove_crossing_out_tokens",
    "kmove_crossing_in_tokens",
    "kmove_acc_post_ref",
    "kmove_acc_post_hw",
)

# Operand-domain divergence flag threshold (eps_band/2), RTL amendment.
_DV_FLAG = 1e-5


def _bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _mask_diff(packed_a: np.ndarray, packed_b: np.ndarray, n: int) -> int:
    ma = np.unpackbits(packed_a)[:n].astype(bool)
    mb = np.unpackbits(packed_b)[:n].astype(bool)
    return int(np.sum(ma != mb))


def _row_has_kd(a: np.lib.npyio.NpzFile) -> bool:
    if "probe_kind" not in a.files:
        return False
    return any(str(k) == "kd" for k in a["probe_kind"].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True, help="frozen golden dir (reference fields)")
    ap.add_argument("--regen", required=True, help="regenerated dump dir (superset)")
    ap.add_argument("--study_e6m12", default="", help="study gold_dump_e6m12 dir (informational)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    moved = set(_MOVED_KEY_FIELDS)
    report: dict = {
        "rows": [],
        "pages": [],
        "fatal": 0,
        "dv_domain": {"max_abs": 0.0, "worst_row": None, "flagged": []},
    }
    dv_global_max = 0.0
    dv_worst_row = None

    for ref_path in sorted(glob.glob(os.path.join(args.frozen, "golden2_*.npz"))):
        name = os.path.basename(ref_path)
        row: dict = {"row": name, "frozen_field_mismatches": [], "missing_fields": []}
        regen_path = os.path.join(args.regen, name)
        if not os.path.exists(regen_path):
            row["frozen_field_mismatches"].append("MISSING REGEN FILE")
            report["fatal"] += 1
            report["rows"].append(row)
            continue
        a = np.load(ref_path, allow_pickle=True)
        b = np.load(regen_path, allow_pickle=True)

        # 1. must-match class: every frozen field except the moved key
        # outputs (and never a field absent from frozen — the loop is over
        # frozen fields, so new item-7 fields are naturally exempt).
        for field in a.files:
            if field in moved:
                continue
            if field not in b.files:
                row["frozen_field_mismatches"].append(f"{field}: missing")
            elif not _bit_equal(a[field], b[field]):
                row["frozen_field_mismatches"].append(f"{field}: bits differ")

        # 2. presence: key fields + item-7 fields (+ kmove iff kd in trace).
        need = list(_KEY_FIELDS) + list(_ITEM7_FIELDS)
        if _row_has_kd(a):
            need += list(_KMOVE_FIELDS)
        for field in need:
            if field not in b.files:
                row["missing_fields"].append(field)

        if row["frozen_field_mismatches"] or row["missing_fields"]:
            report["fatal"] += 1

        # Informational: moved-mask token diffs frozen->regen, fp-vs-key
        # diffs within regen, and study diffs.
        n = int(b["context_len"]) if "context_len" in b.files else int(a["context_len"])
        row["key_mask_token_diffs_frozen_vs_regen"] = {
            kind: _mask_diff(a[f], b[f], n) if f in b.files else None
            for kind, f in (
                ("exact", "v_exact_mask_key_packed"),
                ("hi", "v_hi_mask_key_packed"),
                ("lo", "v_lo_mask_key_packed"),
            )
        }
        if not any(f not in b.files for f in _KEY_FIELDS):
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

        # RTL amendment: operand-domain dv divergence (nonfatal), per record.
        if "vcorr_dv_hw_fp64" in b.files and "vcorr_dv_ref_fp64" in b.files:
            dv_hw = np.asarray(b["vcorr_dv_hw_fp64"], dtype=np.float64)
            dv_ref = np.asarray(b["vcorr_dv_ref_fp64"], dtype=np.float64)
            if dv_hw.size and dv_hw.shape == dv_ref.shape:
                d = np.abs(dv_hw - dv_ref)
                row_max = float(np.max(d))
                row["dv_hw_minus_ref_max"] = row_max
                if row_max > dv_global_max:
                    dv_global_max = row_max
                    dv_worst_row = name
                for r_i in np.flatnonzero(d > _DV_FLAG).tolist():
                    report["dv_domain"]["flagged"].append(
                        {"row": name, "record": int(r_i), "abs_diff": float(d[r_i])}
                    )
            else:
                row["dv_hw_minus_ref_max"] = 0.0
        if "vcorr_dv_trace_delta" in b.files:
            td = np.asarray(b["vcorr_dv_trace_delta"], dtype=np.float64)
            row["dv_trace_delta_max"] = float(np.max(np.abs(td))) if td.size else 0.0

        report["rows"].append(row)

    report["dv_domain"]["max_abs"] = dv_global_max
    report["dv_domain"]["worst_row"] = dv_worst_row

    for ref_path in sorted(glob.glob(os.path.join(args.frozen, "page_v_*.npz"))):
        name = os.path.basename(ref_path)
        regen_path = os.path.join(args.regen, name)
        entry = {"page": name, "status": "missing"}
        if os.path.exists(regen_path):
            a = np.load(ref_path, allow_pickle=True)
            b = np.load(regen_path, allow_pickle=True)
            bad = [f for f in a.files if f not in b.files or not _bit_equal(a[f], b[f])]
            entry["status"] = "bit-identical" if not bad else f"DIFFERS: {bad}"
            if bad:
                report["fatal"] += 1
        report["pages"].append(entry)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    for row in report["rows"]:
        flags = []
        if row["frozen_field_mismatches"]:
            flags.append(f"FROZEN-MISMATCH {row['frozen_field_mismatches']}")
        if row["missing_fields"]:
            flags.append(f"MISSING-FIELDS {row['missing_fields']}")
        info = row.get("fp_vs_key_mask_token_diffs")
        moved_info = row.get("key_mask_token_diffs_frozen_vs_regen")
        study = row.get("key_vs_study_e6m12_mask_token_diffs")
        dvm = row.get("dv_hw_minus_ref_max")
        print(
            f"{row['row']}: {'OK' if not flags else '; '.join(flags)}"
            + (f" moved_key={moved_info}" if moved_info else "")
            + (f" fp_vs_key={info}" if info else "")
            + (f" key_vs_study={study}" if study else "")
            + (f" dv_hw-ref_max={dvm:.3e}" if dvm is not None else "")
        )
    dv = report["dv_domain"]
    print(
        f"dv operand-domain: max|dv_hw-dv_ref|={dv['max_abs']:.3e} "
        f"(row {dv['worst_row']}); flagged records (>1e-5): {len(dv['flagged'])}"
    )
    for f in dv["flagged"]:
        print(f"  FLAG dv_hw-ref {f['row']} rec{f['record']}: {f['abs_diff']:.3e}")
    for entry in report["pages"]:
        print(f"{entry['page']}: {entry['status']}")
    print(f"fatal={report['fatal']}")
    sys.exit(0 if report["fatal"] == 0 else 1)


if __name__ == "__main__":
    main()
