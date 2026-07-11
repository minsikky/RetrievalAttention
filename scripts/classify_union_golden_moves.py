#!/usr/bin/env python3
"""Issue #20 item 3, gate (b): union-aware classification of the stage-2
verifier report against the pre-union blessed goldens (stage2_20260707).

Under the ratified union-commit contract the EXECUTION-domain fields
legitimately move; every WALK-domain field must stay bit-identical (frozen
per-head selection). verify_stage2_key_regen.py enumerates per-row
`frozen_field_mismatches` (excluding its own item-6 moved-key allowlist);
this gate asserts those mismatches are a subset of the documented union
field-change list, that no walk-domain field moved, that all page_v blocks
stayed bit-identical, and that the G3 + two-pass sub-gates are clean.
Exit 1 on any violation.
"""
import argparse
import json
import sys

# The documented union-commit field-change list (see the golden README):
# execution-domain fields recomputed under the union. The verifier's own
# _MOVED_KEY_FIELDS (item-6 key-domain outputs) never appear in
# frozen_field_mismatches and are additionally allowed by construction.
EXPECTED_MOVED = {
    # items 2/3 (band partials, union score domain + extra "union" band)
    "band_labels", "band_count", "band_max", "band_sumexp", "band_acc",
    "combined_output_fp32", "base_output_fp32", "combine_rel_err",
    # item 5 (fp-domain V path over the GIVEN union set, union weights)
    "risk_scores", "v_exact_count", "v_risk_cutoff",
    "v_exact_mask_packed", "v_hi_mask_packed", "v_lo_mask_packed",
    "v_dropped_reads",
    # item 6 input record (w17 is the union softmax weight row)
    "v_w17_fp64",
    # item 7 settled/commit-state fields (3b, 3c, 3e)
    "vcorr_settled_acc_ref", "vcorr_settled_acc_hw",
    "vexact_tokens", "vexact_v_fp16", "vexact_int8_codes",
    "vexact_int8_scale", "vexact_int8_scale_fp64", "vexact_int8_err_fp16",
    "vexact_commit", "vexact_residual_codes", "vexact_residual_scale",
    "vexact_residual_scale_fp64", "vexact_recon_max_abs_err",
    "vexact_band_tokens", "vexact_band_v_fp32", "vexact_band_values_lo_fp32",
    "vexact_band_commit", "vexact_band_p_settled_fp64",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    rep = json.load(open(args.report))

    bad: list[str] = []
    n_rows = 0
    for row in rep.get("rows", []):
        n_rows += 1
        name = row.get("row")
        for miss in row.get("missing_fields", []):
            bad.append(f"{name}: required field missing in regen: {miss}")
        for mm in row.get("frozen_field_mismatches", []):
            field = str(mm).split(":", 1)[0].strip()
            if field == "MISSING REGEN FILE":
                bad.append(f"{name}: {mm}")
            elif field not in EXPECTED_MOVED:
                bad.append(f"{name}: WALK-DOMAIN (or undocumented) field moved: {mm}")
    for page in rep.get("pages", []):
        if page.get("status") not in ("bit-identical",):
            bad.append(f"{page.get('page')}: page block {page.get('status')}")

    g3 = rep.get("g3", {})
    g3_max = max((float(v) for v in g3.get("max_abs_by_band_domain", {}).values()), default=0.0)
    if g3_max > 1e-9:
        bad.append(f"G3 operand rebuild beyond tol: max_abs={g3_max}")
    for r in g3.get("rows", []):
        if r.get("error"):
            bad.append(f"G3 row error {r.get('row')}: {r.get('error')}")
    tp = rep.get("two_pass", {})
    if tp:
        if float(tp.get("max_abs_cutoff", 0.0)) != 0.0 or int(tp.get("max_committed_mismatch", 0)) != 0:
            bad.append(
                f"two-pass rebuild not exact: cutoff={tp.get('max_abs_cutoff')} "
                f"committed_mismatch={tp.get('max_committed_mismatch')}"
            )
        for r in tp.get("rows", []):
            if r.get("error"):
                bad.append(f"two-pass row error {r.get('row')}: {r.get('error')}")

    print(f"[classify] {n_rows} frozen-reference rows checked; "
          f"G3 max_abs={g3_max:.3e}; two_pass checked={tp.get('checked', 0)}")
    if bad:
        print(f"[classify] FAIL: {len(bad)} violation(s)")
        for b in bad:
            print(f"[classify]   {b}")
        return 1
    print("[classify] PASS: walk-domain fields bit-identical; only documented "
          "union execution fields moved; page blocks bit-identical; G3 + two-pass clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
