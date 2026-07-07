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
    # Issue #7 option A (operand-serialization blocker): raw Vcorr operands for
    # the band-member union (settled exact + both probes' marginal/hi-boundary),
    # so G3 rebuilds every acc bit-tight (see _rebuild_vcorr_accs below).
    "vexact_band_tokens",
    "vexact_band_v_fp32",
    "vexact_band_values_lo_fp32",
    "vexact_band_commit",
    "vexact_band_p_settled_fp64",
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


def _quantize_rows_symmetric(x: np.ndarray, bits: int) -> np.ndarray:
    """Per-row symmetric absmax quantization (plane-A / plane-B rule). KEEP IN
    SYNC with run_joint_kv_budget_policy_eval._quantize_rows_symmetric."""
    levels = float((1 << (max(2, int(bits)) - 1)) - 1)
    scale = np.max(np.abs(x), axis=1, keepdims=True) / levels
    scale = np.maximum(scale, 1e-12)
    return (np.round(x / scale) * scale).astype(np.float32, copy=False)


def _reconstruct_vpq_from_pages(page: "np.lib.npyio.NpzFile", tokens: np.ndarray, band_v_fp32: np.ndarray) -> np.ndarray:
    """Rebuild the V-PQ centroid rows (v_pq / vhat) for `tokens` from the
    all-sealed-pages codebook+codes (issue #7 option A, hole (1)). Mirrors
    _vpq_values_for_tokens: v_pq[t] = codebook[p, sub, code[t, sub]] for the
    page p covering token t (searchsorted), with a raw-value fallback for
    non-paged tokens (residual == 0 there, matching the golden vhat := raw
    value rule). Returns float32 rows, bit-identical to the vhat used in the
    accumulators."""
    tokens = np.asarray(tokens, dtype=np.int64)
    hd = int(band_v_fp32.shape[1]) if band_v_fp32.ndim == 2 else 0
    out = np.asarray(band_v_fp32, dtype=np.float32).copy()  # non-paged fallback = raw v
    if tokens.size == 0 or "all_page_starts" not in page.files:
        return out
    starts = np.asarray(page["all_page_starts"], dtype=np.int64)
    sizes = np.asarray(page["all_page_sizes"], dtype=np.int64)
    codebooks = np.asarray(page["all_value_codebooks_fp32"], dtype=np.float32)
    codes = np.asarray(page["all_value_codes_u8"])
    offs = np.asarray(page["all_value_codes_offsets"], dtype=np.int64)
    if starts.size == 0:
        return out
    pid = np.searchsorted(starts, tokens, side="right") - 1
    valid = (pid >= 0) & (pid < starts.size)
    valid &= tokens < (starts[np.maximum(pid, 0)] + sizes[np.maximum(pid, 0)])
    codes2 = codes.reshape(codes.shape[0], -1) if codes.ndim == 1 else codes
    subvecs = int(codebooks.shape[1]) if codebooks.ndim == 4 else 0
    subdim = int(codebooks.shape[3]) if codebooks.ndim == 4 else 0
    for p in np.unique(pid[valid]).tolist():
        sel = np.nonzero(valid & (pid == int(p)))[0]
        local = (tokens[sel] - int(starts[int(p)])).astype(np.int64)
        rows = int(offs[int(p)]) + local
        row = np.zeros((sel.size, subvecs * subdim), dtype=np.float32)
        for sub in range(subvecs):
            row[:, sub * subdim:(sub + 1) * subdim] = codebooks[int(p), sub, codes2[rows, sub].astype(np.int64)]
        out[sel, :subvecs * subdim] = row
    return out


def _rebuild_vcorr_accs(a: "np.lib.npyio.NpzFile", vhat_rows: np.ndarray) -> dict:
    """Issue #7 option-A G3 rebuild in the verifier: reconstruct every Vcorr
    accumulator (marginal/hi-boundary per record + settled total, REF and HW)
    from the serialized raw operands ALONE. KEEP IN SYNC with
    run_joint_kv_budget_policy_eval.rebuild_vcorr_accs_from_operands.

    Returns a dict of rebuilt accs; the caller compares to the stored acc
    fields bit-tight."""
    tok = np.asarray(a["vexact_band_tokens"], dtype=np.int64)
    order = np.argsort(tok, kind="stable")
    tok_sorted = tok[order]
    v = np.asarray(a["vexact_band_v_fp32"], dtype=np.float32)[order]
    vlo = np.asarray(a["vexact_band_values_lo_fp32"], dtype=np.float32)[order]
    vhat64 = np.asarray(vhat_rows, dtype=np.float32)[order].astype(np.float64)
    commit = np.asarray(a["vexact_band_commit"]).astype(bool)[order]
    p_settled = np.asarray(a["vexact_band_p_settled_fp64"], dtype=np.float64)[order]
    hd = int(v.shape[1]) if v.ndim == 2 else 0

    residual_hi = v.astype(np.float64) - vhat64
    residual_lo = vlo.astype(np.float64) - vhat64
    plane_b = _quantize_rows_symmetric((v - vlo), 8)
    recon64 = vlo.astype(np.float64) + plane_b.astype(np.float64)
    diff16_hi = (recon64 - vhat64).astype(np.float16).astype(np.float64)
    diff16_lo = (vlo.astype(np.float64) - vhat64).astype(np.float16).astype(np.float64)
    ops = {"ref": (residual_hi, residual_lo), "hw": (diff16_hi, diff16_lo)}

    def _rows(t: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(tok_sorted, np.asarray(t, dtype=np.int64))
        if pos.size and (
            bool(np.any(pos >= tok_sorted.size))
            or bool(np.any(tok_sorted[np.minimum(pos, tok_sorted.size - 1)] != np.asarray(t, dtype=np.int64)))
        ):
            raise AssertionError("g3 rebuild: band-member token missing from operand union")
        return pos

    marg_off = np.asarray(a["vcorr_marginal_offsets"], dtype=np.int64)
    hib_off = np.asarray(a["vcorr_hiboundary_offsets"], dtype=np.int64)
    R = int(marg_off.size - 1)
    marg_tok = np.asarray(a["vcorr_marginal_tokens"], dtype=np.int64)
    marg_p = np.asarray(a["vcorr_marginal_p"], dtype=np.float64)
    hib_tok = np.asarray(a["vcorr_hiboundary_tokens"], dtype=np.int64)
    hib_p = np.asarray(a["vcorr_hiboundary_p"], dtype=np.float64)
    n = int(a["context_len"])
    hi_s = np.flatnonzero(np.unpackbits(a["v_hi_mask_key_packed"])[:n].astype(bool))
    lo_s = np.flatnonzero(np.unpackbits(a["v_lo_mask_key_packed"])[:n].astype(bool))
    base64 = np.asarray(a["base_output_fp32"], dtype=np.float32).astype(np.float64)

    out = {}
    for dom, (op_hi, op_lo) in ops.items():
        acc_marg = np.zeros((R, hd), dtype=np.float64)
        acc_hib = np.zeros((R, hd), dtype=np.float64)
        for r in range(R):
            m0, m1 = int(marg_off[r]), int(marg_off[r + 1])
            if m1 > m0:
                mr = _rows(marg_tok[m0:m1])
                w = commit[mr]
                if bool(np.any(w)):
                    acc_marg[r] = marg_p[m0:m1][w] @ op_lo[mr][w]
            h0, h1 = int(hib_off[r]), int(hib_off[r + 1])
            if h1 > h0:
                hr = _rows(hib_tok[h0:h1])
                wh = commit[hr]
                xr = op_hi[hr] - (wh.astype(np.float64)[:, None] * op_lo[hr])
                acc_hib[r] = hib_p[h0:h1] @ xr
        out[f"marginal_{dom}"] = acc_marg
        out[f"hiboundary_{dom}"] = acc_hib
        s = base64.copy()
        if hi_s.size:
            hr = _rows(np.sort(hi_s))
            s = s + p_settled[hr] @ op_hi[hr]
        if lo_s.size:
            lr = _rows(np.sort(lo_s))
            s = s + p_settled[lr] @ op_lo[lr]
        out[f"settled_{dom}"] = s - base64
    return out


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
    ap.add_argument(
        "--frozen",
        default="",
        help="frozen golden dir (reference fields). Optional: when omitted the "
        "superset/back-compat comparison is skipped and only the G3 operand "
        "rebuild runs over --regen (used for tiny smoke dumps with no matching "
        "frozen row).",
    )
    ap.add_argument("--regen", required=True, help="regenerated dump dir (superset)")
    ap.add_argument("--study_e6m12", default="", help="study gold_dump_e6m12 dir (informational)")
    ap.add_argument(
        "--g3_tol",
        type=float,
        default=1e-9,
        help="max |rebuilt acc - stored acc| tolerance for the G3 operand "
        "rebuild (fp64; expected bit-tight ~0).",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    moved = set(_MOVED_KEY_FIELDS)
    report: dict = {
        "rows": [],
        "pages": [],
        "fatal": 0,
        "dv_domain": {"max_abs": 0.0, "worst_row": None, "flagged": []},
        "g3": {"rows": [], "max_abs_by_band_domain": {}, "worst": None, "checked": 0},
    }
    dv_global_max = 0.0
    dv_worst_row = None

    for ref_path in sorted(glob.glob(os.path.join(args.frozen, "golden2_*.npz"))) if args.frozen else []:
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

    for ref_path in (sorted(glob.glob(os.path.join(args.frozen, "page_v_*.npz"))) if args.frozen else []):
        name = os.path.basename(ref_path)
        regen_path = os.path.join(args.regen, name)
        entry = {"page": name, "status": "missing"}
        if os.path.exists(regen_path):
            a = np.load(ref_path, allow_pickle=True)
            b = np.load(regen_path, allow_pickle=True)
            # Legacy single-page fields must stay byte-identical (superset). The
            # new all_* multi-page fields are additive and not in the frozen dir.
            bad = [f for f in a.files if f not in b.files or not _bit_equal(a[f], b[f])]
            entry["status"] = "bit-identical" if not bad else f"DIFFERS: {bad}"
            if bad:
                report["fatal"] += 1
        report["pages"].append(entry)

    # G3 gate (issue #7 option A): for EVERY regen row, rebuild every Vcorr
    # accumulator from the serialized operands ALONE -- v_pq reconstructed from
    # the all-sealed-pages codebook+codes, exact-V operands from vexact_band_*,
    # weights from vcorr_*_p / vexact_band_p_settled -- and hard-assert bit-tight
    # against the stored acc. This is the operand-serialization proof RTL's item-7
    # recompute was blocked on.
    band_domains = (
        "marginal_ref", "marginal_hw",
        "hiboundary_ref", "hiboundary_hw",
        "settled_ref", "settled_hw",
    )
    g3_max = {k: 0.0 for k in band_domains}
    g3_worst = None
    g3_worst_val = -1.0
    for regen_path in sorted(glob.glob(os.path.join(args.regen, "golden2_*.npz"))):
        name = os.path.basename(regen_path)
        b = np.load(regen_path, allow_pickle=True)
        if "vexact_band_tokens" not in b.files:
            # lo tier not live for this row (no item-7 operands) -- nothing to
            # rebuild; the presence check above already gates required rows.
            continue
        g3row: dict = {"row": name, "max_abs_by_band_domain": {}, "error": None}
        try:
            ctx = int(b["context_len"])
            kv = int(b["kv_head"])
            page_path = os.path.join(args.regen, f"page_v_ctx{ctx}_kv{kv}.npz")
            if not os.path.exists(page_path):
                raise AssertionError(f"missing page block {os.path.basename(page_path)}")
            page = np.load(page_path, allow_pickle=True)
            tok = np.asarray(b["vexact_band_tokens"], dtype=np.int64)
            vhat_rows = _reconstruct_vpq_from_pages(page, tok, b["vexact_band_v_fp32"])
            rebuilt = _rebuild_vcorr_accs(b, vhat_rows)
            for key in band_domains:
                stored = np.asarray(b["vcorr_settled_acc_" + key.split("_", 1)[1]], dtype=np.float64) \
                    if key.startswith("settled_") else \
                    np.asarray(b["vcorr_acc_" + key], dtype=np.float64)
                got = rebuilt[key]
                err = float(np.max(np.abs(got - stored))) if stored.size else 0.0
                g3row["max_abs_by_band_domain"][key] = err
                if err > g3_max[key]:
                    g3_max[key] = err
                if err > g3_worst_val:
                    g3_worst_val = err
                    g3_worst = f"{name}:{key}"
                if err > float(args.g3_tol):
                    g3row["error"] = (
                        f"{key} operand-rebuild mismatch max_abs={err:.3e} > tol {args.g3_tol:.1e}"
                    )
                    report["fatal"] += 1
        except Exception as exc:  # rebuild failure is fatal
            g3row["error"] = f"{type(exc).__name__}: {exc}"
            report["fatal"] += 1
        report["g3"]["rows"].append(g3row)
        report["g3"]["checked"] += 1
    report["g3"]["max_abs_by_band_domain"] = g3_max
    report["g3"]["worst"] = {"key": g3_worst, "max_abs": (g3_worst_val if g3_worst_val >= 0 else 0.0)}

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
    g3 = report["g3"]
    print(
        f"G3 operand rebuild: checked {g3['checked']} rows; "
        f"max|rebuilt-stored| per band/domain = "
        + ", ".join(f"{k}={v:.2e}" for k, v in g3["max_abs_by_band_domain"].items())
    )
    for g3row in g3["rows"]:
        if g3row.get("error"):
            print(f"  G3 FAIL {g3row['row']}: {g3row['error']}")
    print(f"fatal={report['fatal']}")
    sys.exit(0 if report["fatal"] == 0 else 1)


if __name__ == "__main__":
    main()
