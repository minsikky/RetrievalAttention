#!/usr/bin/env python3
"""Token-level acceptance analysis for the draft-then-verify GPU experiment.

Compares each draft arm's greedy token stream against the frozen (DRAFT_MODE=off)
arm on identical samples. Greedy decoding is deterministic, so we recover the
token stream by re-tokenizing each arm's stored ``pred`` text with the model
tokenizer; systematic detokenize/retokenize artifacts cancel because both arms
are treated identically.

Metrics per (mode x ctx):
  - per-token agreement rate (over aligned positions from 0)
  - first-divergence position per sample (and whether the answer region passed)
  - agreement run-length distribution
  - acceptance@k for k in {4, 8, 16}: fraction of length-k sliding windows that
    match fully (this is the speculative-decoding acceptance notion)
  - expected accepted-prefix length at k=8: mean over length-8 windows of the
    number of leading tokens that match before the first mismatch (0..8)
  - standalone draft correctness: does the draft pred contain the gold answer
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def load_rows(run_dir: Path, task: str) -> dict[int, dict]:
    path = run_dir / "pred" / f"{task}.jsonl"
    rows = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[int(row["index"])] = row
    return rows


def gold_answers(row: dict) -> list[str]:
    outs = row.get("outputs") or []
    if isinstance(outs, str):
        outs = [outs]
    return [str(o) for o in outs if str(o).strip()]


def answer_token_position(tok_ids: list[int], pred_text: str, golds: list[str], tokenizer) -> int | None:
    """Approx token index at which the (first) gold answer completes in the stream.

    Returns None if no gold string is found in the decoded prefix.
    """
    low = pred_text.lower()
    best = None
    for g in golds:
        gl = g.lower()
        pos = low.find(gl)
        if pos < 0:
            continue
        # character end of the gold occurrence -> approximate token index by
        # tokenizing the text prefix up to that char end.
        char_end = pos + len(gl)
        prefix = pred_text[:char_end]
        n_tok = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        best = n_tok if best is None else min(best, n_tok)
    return best


def contains_gold(pred_text: str, golds: list[str]) -> bool:
    low = pred_text.lower()
    return any(g.lower() in low for g in golds if g.strip())


def run_lengths(matches: list[bool]) -> list[int]:
    runs = []
    cur = 0
    for m in matches:
        if m:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def acceptance_at_k(match_lists: list[list[bool]], k: int) -> float:
    """Fraction of length-k windows that match fully.

    AUDIT FIX (2026-07-12): windows slide WITHIN each sample's stream and
    window counts are pooled across samples. The previous version slid over
    the concatenated pooled list, so windows straddled sample boundaries —
    spurious cross-stream windows that inflated acc@k, worst at thin
    k=16/32 stats."""
    total = 0
    good = 0
    for m in match_lists:
        for i in range(max(0, len(m) - k + 1)):
            total += 1
            if all(m[i: i + k]):
                good += 1
    return good / total if total else float("nan")


def expected_accepted_prefix(match_lists: list[list[bool]], k: int) -> float:
    """Mean accepted-prefix length over per-sample length-k windows.

    Same per-sample windowing as acceptance_at_k (audit fix)."""
    total_w = 0
    total_len = 0
    for m in match_lists:
        for i in range(max(0, len(m) - k + 1)):
            total_w += 1
            for j in range(k):
                if m[i + j]:
                    total_len += 1
                else:
                    break
    return total_len / total_w if total_w else float("nan")


def windows_at_k(match_lists: list[list[bool]], k: int) -> int:
    return sum(max(0, len(m) - k + 1) for m in match_lists)

def renewal_stats(match_lists: list[list[bool]], k: int) -> dict:
    """Exact k-round speculative-decode renewal walk over per-sample bitmaps.

    PRIMARY acceptance metric (2026-07-12 methodology order): from position
    t, accept the run of matches up to k, emit accepted_len, advance
    t += accepted_len + 1 (the verifier-corrected token), repeat; never
    crosses sample boundaries. Exact for the deterministic greedy loop.
    Sliding windows (kept as secondary) average all offsets uniformly, but
    real round starts are renewal-biased toward just-after-a-miss positions.
    """
    rounds: list[int] = []
    for m in match_lists:
        t = 0
        n = len(m)
        while t < n:
            acc = 0
            while acc < k and t + acc < n and m[t + acc]:
                acc += 1
            rounds.append(acc)
            t += acc + 1
    if not rounds:
        return {"k": k, "n_rounds": 0}
    rs = sorted(rounds)

    def q(p: float) -> int:
        return rs[min(len(rs) - 1, max(0, int(round(p * (len(rs) - 1)))))]

    return {
        "k": k,
        "n_rounds": len(rounds),
        "mean_accepted_per_round": sum(rounds) / len(rounds),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "full_rounds_frac": sum(1 for r in rounds if r == k) / len(rounds),
    }


def clopper_pearson_ci(successes: int, n: int, alpha: float = 0.05) -> list[float]:
    """Exact binomial CI for per-position acceptance (scipy if available)."""
    if n <= 0:
        return [float("nan"), float("nan")]
    try:
        from scipy.stats import beta as _beta
        lo = 0.0 if successes <= 0 else float(_beta.ppf(alpha / 2, successes, n - successes + 1))
        hi = 1.0 if successes >= n else float(_beta.ppf(1 - alpha / 2, successes + 1, n - successes))
        return [lo, hi]
    except Exception:
        import math as _math
        p = successes / n
        se = _math.sqrt(max(p * (1 - p), 1e-12) / n)
        return [max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)]



def analyze_pair(frozen_rows, draft_rows, task, tokenizer):
    per_sample = []
    all_matches: list[bool] = []
    match_lists: list[list[bool]] = []
    for idx in sorted(set(frozen_rows) & set(draft_rows)):
        fz = frozen_rows[idx]
        dr = draft_rows[idx]
        fz_ids = tokenizer(fz["pred"], add_special_tokens=False)["input_ids"]
        dr_ids = tokenizer(dr["pred"], add_special_tokens=False)["input_ids"]
        n = min(len(fz_ids), len(dr_ids))
        matches = [fz_ids[i] == dr_ids[i] for i in range(n)]
        first_div = next((i for i, m in enumerate(matches) if not m), None)
        if first_div is None and len(fz_ids) != len(dr_ids):
            first_div = n  # streams identical up to n, then a length divergence
        golds = gold_answers(fz)
        ans_pos = answer_token_position(fz_ids, fz["pred"], golds, tokenizer)
        answer_passed_at_div = (
            None if first_div is None or ans_pos is None else bool(first_div >= ans_pos)
        )
        per_sample.append(
            {
                "index": idx,
                "frozen_len": len(fz_ids),
                "draft_len": len(dr_ids),
                "aligned_len": n,
                "agreement_rate": (sum(matches) / n) if n else float("nan"),
                "first_divergence": first_div,
                "answer_token_pos": ans_pos,
                "answer_region_passed_before_fork": answer_passed_at_div,
                "run_lengths": run_lengths(matches),
                "frozen_correct": contains_gold(fz["pred"], golds),
                "draft_correct": contains_gold(dr["pred"], golds),
                "golds": golds,
                "frozen_pred": fz["pred"],
                "draft_pred": dr["pred"],
            }
        )
        all_matches.extend(matches)
        match_lists.append(matches)

    agg = {
        "n_samples": len(per_sample),
        "pooled_aligned_tokens": len(all_matches),
        "pooled_agreement_rate": (sum(all_matches) / len(all_matches)) if all_matches else float("nan"),
        "n_positions": len(all_matches),
        "per_position_agreement_ci95": clopper_pearson_ci(sum(all_matches), len(all_matches)),
        # PRIMARY: exact renewal-loop statistics (see renewal_stats docstring).
        "renewal_at_k": {str(k): renewal_stats(match_lists, k) for k in ACCEPT_KS},
        # SECONDARY (continuity): sliding-window stats, offset-uniform proxy.
        "acceptance_at_k": {str(k): acceptance_at_k(match_lists, k) for k in ACCEPT_KS},
        "expected_accepted_prefix_at_k": {
            str(k): expected_accepted_prefix(match_lists, k) for k in ACCEPT_KS
        },
        # Window counts alongside so large-k thinness is visible (streams are
        # 64-128 positions; k=32 windows are few). Per-sample windowing.
        "windows_at_k": {
            str(k): windows_at_k(match_lists, k) for k in ACCEPT_KS
        },
        "expected_accepted_prefix_k8": expected_accepted_prefix(match_lists, 8),
        "mean_first_divergence": (
            statistics.mean(
                [s["first_divergence"] for s in per_sample if s["first_divergence"] is not None]
            )
            if any(s["first_divergence"] is not None for s in per_sample)
            else None
        ),
        "n_fully_identical_streams": sum(
            1 for s in per_sample if s["first_divergence"] is None
        ),
        "draft_standalone_correct": sum(1 for s in per_sample if s["draft_correct"]),
        "frozen_correct": sum(1 for s in per_sample if s["frozen_correct"]),
    }
    return {"aggregate": agg, "per_sample": per_sample}


ACCEPT_KS = (4, 8, 16, 32)

MATRIX_ARMS = ["dense", "off", "start1", "pq_only", "middle", "v_exact"]
MATRIX_REFS = ["dense", "off"]
# Free-running gray zone (RTL methodology note on #21): pairs landing here
# should get one teacher-forced confirmation before architecture is bet on
# them, since spec-decode acceptance is formally prefix-conditioned.
GRAY_ZONE = (0.6, 0.85)


def run_bytes_per_token(run_dir: Path, task: str) -> dict | None:
    """Committed-bytes-per-token from a run's summary cost counters.

    Sums over layers: mean_logical_frontier_{component}_MB_per_head_query x
    heads-per-layer (heads = head_query_calls / approx_attention_calls), so
    the result is model-wide MB per generated token, split by component
    (total / exact_kv / selector scan / tail estimator). This is the per-arm
    draft-bytes input to the (tau, k) objective
    (k*draft_bytes + D) / E[prefix]@k. Returns None for runs without cost
    counters (dense_batched arms)."""
    path = run_dir / "summary" / f"{task}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    layers = d.get("cost_proxy")
    if not isinstance(layers, dict) or not layers:
        return None
    comps = {
        "total": "mean_logical_frontier_total_MB_per_head_query",
        "exact_kv": "mean_logical_frontier_exact_KV_MB_per_head_query",
        "selector": "mean_logical_frontier_selector_MB_per_head_query",
        "tail_estimator": "mean_logical_frontier_tail_estimator_MB_per_head_query",
    }
    out = {k: 0.0 for k in comps}
    seen = False
    for layer in layers.values():
        if not isinstance(layer, dict):
            continue
        calls = float(layer.get("approx_attention_calls") or 0)
        hq = float(layer.get("head_query_calls") or 0)
        if calls <= 0 or hq <= 0:
            continue
        heads = hq / calls
        for name, key in comps.items():
            v = layer.get(key)
            if v is not None:
                out[name] += float(v) * heads
                seen = True
    if not seen:
        return None
    out["mean_selected_tokens_l0"] = float(
        layers.get("0", {}).get("mean_selected_tokens", float("nan"))
    )
    return {
        (f"logical_{k}_MB_per_token" if not k.startswith("mean_") else k): v
        for k, v in out.items()
    }


def run_matrix(root: Path, tokenizer, out_json: str) -> None:
    """Full pairwise per-token agreement matrix (issue #21 round 3).

    Every arm is scored against BOTH references (dense stream and off/frozen
    stream); self-pairs (dense_vs_dense, off_vs_off) are kept as harness
    sanity rows and must be identical streams. The headline number is
    off_vs_dense (the dense verify tier's acceptance rate).

    Methodology flag carried in the JSON: free-running alignment is a PROXY
    for prefix-conditioned spec-decode acceptance. Any non-self pair whose
    pooled agreement lands in GRAY_ZONE is listed under
    teacher_forced_confirmation_recommended.
    """
    contexts = [
        ("32k", "qa_1", "qa1_32k_n4"),
        ("128k", "niah_multikey_3", "mk3_128k_n2"),
    ]
    results: dict = {
        "methodology_note": (
            "Free-running alignment: each arm greedy-decodes independently and "
            "streams are aligned position-by-position from the start. This is a "
            "proxy for prefix-conditioned speculative-decode acceptance (which "
            "would force the reference prefix before each draft window). It was "
            "predictive for the start1/start2 arms, but pairs in the "
            f"{GRAY_ZONE} agreement gray zone should get one teacher-forced "
            "confirmation before architecture decisions are made on them."
        ),
        "headline": "off_vs_dense",
        "gray_zone": list(GRAY_ZONE),
        "teacher_forced_confirmation_recommended": [],
        "bytes_per_token": {},
        "pairs": {},
    }
    for ctx, task, prefix in contexts:
        rows_by_arm = {}
        for arm in MATRIX_ARMS:
            run_dir = root / f"{prefix}_{arm}"
            if (run_dir / "pred" / f"{task}.jsonl").exists():
                rows_by_arm[arm] = load_rows(run_dir, task)
                bpt = run_bytes_per_token(run_dir, task)
                if bpt is not None:
                    results["bytes_per_token"][f"{ctx}:{arm}"] = bpt
        for ref in MATRIX_REFS:
            if ref not in rows_by_arm:
                for arm in MATRIX_ARMS:
                    results["pairs"][f"{ctx}:{arm}_vs_{ref}"] = {
                        "skipped": f"missing reference run {prefix}_{ref}"
                    }
                continue
            for arm in MATRIX_ARMS:
                label = f"{ctx}:{arm}_vs_{ref}"
                if arm not in rows_by_arm:
                    results["pairs"][label] = {
                        "skipped": f"missing arm run {prefix}_{arm}"
                    }
                    continue
                pair = analyze_pair(
                    rows_by_arm[ref], rows_by_arm[arm], task, tokenizer
                )
                entry = {
                    "ctx": ctx,
                    "task": task,
                    "reference_run": f"{prefix}_{ref}",
                    "arm_run": f"{prefix}_{arm}",
                    "self_pair": arm == ref,
                    **pair,
                }
                agree = entry["aggregate"]["pooled_agreement_rate"]
                in_gray = (
                    arm != ref
                    and agree == agree  # not NaN
                    and GRAY_ZONE[0] <= agree <= GRAY_ZONE[1]
                )
                entry["teacher_forced_confirmation_recommended"] = bool(in_gray)
                if in_gray:
                    results["teacher_forced_confirmation_recommended"].append(label)
                results["pairs"][label] = entry

    Path(out_json).write_text(json.dumps(results, indent=2))

    print(
        f"{'pair':<26} {'agree':>7} {'acc@4':>7} {'acc@8':>7} {'acc@16':>7} "
        f"{'eap@8':>7} {'armOK':>6} {'gray':>5}"
    )
    for label, r in results["pairs"].items():
        if "aggregate" not in r:
            print(f"{label:<26} {r.get('skipped', '?')}")
            continue
        a = r["aggregate"]
        acc = a["acceptance_at_k"]
        print(
            f"{label:<26} {a['pooled_agreement_rate']:>7.3f} "
            f"{acc['4']:>7.3f} {acc['8']:>7.3f} {acc['16']:>7.3f} "
            f"{a['expected_accepted_prefix_k8']:>7.3f} "
            f"{a['draft_standalone_correct']}/{a['n_samples']:<4} "
            f"{'YES' if r['teacher_forced_confirmation_recommended'] else '':>5}"
        )
    if results["teacher_forced_confirmation_recommended"]:
        print(
            "gray-zone pairs (teacher-forced confirmation recommended): "
            + ", ".join(results["teacher_forced_confirmation_recommended"])
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument(
        "--arm_set",
        choices=["round1", "round2", "matrix"],
        default="round1",
        help=(
            "round1: start1/start2 vs off (20260711 bundle). "
            "round2: pq_only/middle vs off + off-vs-off sanity (20260712 bundle). "
            "matrix: full pairwise matrix over {dense,off,start1,pq_only,middle,"
            "v_exact} scored against BOTH the dense and the off (frozen) "
            "references (20260712_matrix bundles)."
        ),
    )
    args = ap.parse_args()

    root = Path(args.output_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.arm_set == "matrix":
        run_matrix(root, tokenizer, args.out_json)
        return

    # (label, ctx, task, frozen_run, draft_run)
    if args.arm_set == "round2":
        arms = [
            # off-vs-off is the harness sanity: same run compared to itself must
            # be a perfectly identical stream (agree 1.000, eap@8 = 8).
            ("32k_off_vs_off", "32k", "qa_1", "qa1_32k_n4_off", "qa1_32k_n4_off"),
            ("32k_pq_only", "32k", "qa_1", "qa1_32k_n4_off", "qa1_32k_n4_pq_only"),
            ("32k_middle", "32k", "qa_1", "qa1_32k_n4_off", "qa1_32k_n4_middle"),
            ("128k_off_vs_off", "128k", "niah_multikey_3", "mk3_128k_n2_off", "mk3_128k_n2_off"),
            ("128k_pq_only", "128k", "niah_multikey_3", "mk3_128k_n2_off", "mk3_128k_n2_pq_only"),
            ("128k_middle", "128k", "niah_multikey_3", "mk3_128k_n2_off", "mk3_128k_n2_middle"),
        ]
    else:
        arms = [
            ("32k_start1", "32k", "qa_1", "qa1_32k_n4_off", "qa1_32k_n4_start1"),
            ("32k_start2", "32k", "qa_1", "qa1_32k_n4_off", "qa1_32k_n4_start2"),
            ("128k_start1", "128k", "niah_multikey_3", "mk3_128k_n2_off", "mk3_128k_n2_start1"),
            ("128k_start2", "128k", "niah_multikey_3", "mk3_128k_n2_off", "mk3_128k_n2_start2"),
        ]

    results = {}
    for label, ctx, task, frozen_run, draft_run in arms:
        fz_dir = root / frozen_run
        dr_dir = root / draft_run
        if not (fz_dir / "pred" / f"{task}.jsonl").exists():
            results[label] = {"skipped": f"missing frozen {frozen_run}"}
            continue
        if not (dr_dir / "pred" / f"{task}.jsonl").exists():
            results[label] = {"skipped": f"missing draft {draft_run}"}
            continue
        frozen_rows = load_rows(fz_dir, task)
        draft_rows = load_rows(dr_dir, task)
        results[label] = {
            "ctx": ctx,
            "task": task,
            "frozen_run": frozen_run,
            "draft_run": draft_run,
            **analyze_pair(frozen_rows, draft_rows, task, tokenizer),
        }

    Path(args.out_json).write_text(json.dumps(results, indent=2))

    # compact console table
    print(f"{'arm':<14} {'agree':>7} {'acc@4':>7} {'acc@8':>7} {'acc@16':>7} {'eap@8':>7} {'draftOK':>8}")
    for label, r in results.items():
        if "aggregate" not in r:
            print(f"{label:<14} {r.get('skipped','?')}")
            continue
        a = r["aggregate"]
        acc = a["acceptance_at_k"]
        print(
            f"{label:<14} {a['pooled_agreement_rate']:>7.3f} "
            f"{acc['4']:>7.3f} {acc['8']:>7.3f} {acc['16']:>7.3f} "
            f"{a['expected_accepted_prefix_k8']:>7.3f} "
            f"{a['draft_standalone_correct']}/{a['n_samples']:<6}"
        )


if __name__ == "__main__":
    main()
