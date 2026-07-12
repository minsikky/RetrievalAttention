#!/usr/bin/env python3
"""Post-process draft-verify logit traces (issue #21 round-3 addendum).

Each bundle pass (free-running or teacher-forced) dumps a full per-position
logit trace via GREEDY_LOGIT_TRACE_FILE (torch.save list of
{index, prompt_tokens, token_ids, logits[(steps,1,vocab)]}). This script:

1. Converts every trace to the RTL-consumable npz, one per (run, sample),
   written beside the preds as ``logit_topk_sample{index}.npz`` with arrays:
     top_logits   (steps, 32) f32   -- top-32 logits, descending
     top_ids      (steps, 32) i32   -- their token ids
     logsumexp    (steps,)    f32   -- full-vocab logsumexp per position
     argmax       (steps,)    i32   -- the arm's greedy token per position
     token_ids    (steps,)    i32   -- the emitted stream (== forced reference
                                       tokens for teacher-forced passes)
     prompt_tokens ()         i64
   Position semantics: row i is the logit vector that PREDICTS stream position
   i, conditioned on positions 0..i-1 of the emitted (or forced) stream. So
   for a teacher-forced pass, per-position prefix-conditioned greedy
   acceptance is exactly argmax[i] == token_ids[i], and the canonical any-T
   acceptance sum(min(p_T, q_T)) is computable offline from the top-32 +
   logsumexp of the (arm, reference) pair at the same position.

2. For teacher-forced runs (dir name *_tf_{ref}): computes prefix-conditioned
   greedy acceptance (pooled + per sample), acceptance@{4,8,16} over windows
   of the per-position accept flags, E[prefix]@8, and the fork-margin
   distribution at rejection positions: margin = logit[arm argmax] -
   logit[reference token] (>= 0 by construction; near-zero = near-tie fork,
   large = confident disagreement).

Usage:
  python logit_trace_postproc.py --output_root <matrix root> \
      --out_json tf_acceptance.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

TOPK = 32
WINDOW_KS = (4, 8, 16, 32)
PREFIXES = ("qa1_32k_n4", "mk3_128k_n2")


def acceptance_at_k(match_lists: list[list[bool]], k: int) -> float:
    """Fraction of length-k windows fully accepted.

    AUDIT FIX (2026-07-12): windows slide WITHIN each sample's stream and
    counts pool across samples (previously windows straddled sample
    boundaries in the pooled list, inflating acc@k at thin k=16/32)."""
    total = 0
    good = 0
    for m in match_lists:
        for i in range(max(0, len(m) - k + 1)):
            total += 1
            if all(m[i: i + k]):
                good += 1
    return good / total if total else float("nan")


def expected_accepted_prefix(match_lists: list[list[bool]], k: int) -> float:
    """Mean accepted-prefix length over per-sample length-k windows."""
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



def convert_trace(run_dir: Path) -> list[dict]:
    """Write npz files for one run dir; return per-sample summaries."""
    trace_path = run_dir / "logit_trace.pt"
    records = torch.load(trace_path, map_location="cpu", weights_only=False)
    out = []
    for rec in records:
        logits = rec["logits"].reshape(len(rec["token_ids"]), -1).float()
        top = torch.topk(logits, k=min(TOPK, logits.shape[1]), dim=1)
        lse = torch.logsumexp(logits, dim=1)
        argmax = torch.argmax(logits, dim=1)
        token_ids = torch.as_tensor(rec["token_ids"], dtype=torch.int32)
        npz_path = run_dir / f"logit_topk_sample{rec['index']}.npz"
        np.savez_compressed(
            npz_path,
            top_logits=top.values.numpy().astype(np.float32),
            top_ids=top.indices.numpy().astype(np.int32),
            logsumexp=lse.numpy().astype(np.float32),
            argmax=argmax.numpy().astype(np.int32),
            token_ids=token_ids.numpy(),
            prompt_tokens=np.int64(rec["prompt_tokens"]),
        )
        out.append(
            {
                "index": rec["index"],
                "steps": int(logits.shape[0]),
                "npz": str(npz_path),
                "argmax": argmax.tolist(),
                "token_ids": [int(t) for t in rec["token_ids"]],
                # margin vs the emitted/forced token at every position
                "margin_vs_stream": (
                    logits.max(dim=1).values
                    - logits.gather(1, token_ids.long().reshape(-1, 1)).reshape(-1)
                ).tolist(),
            }
        )
    return out


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
        "frac_below_0p5": float((arr < 0.5).mean()),
        "frac_below_1": float((arr < 1.0).mean()),
        "frac_below_2": float((arr < 2.0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    root = Path(args.output_root)

    results: dict = {"converted_runs": [], "teacher_forced": {}}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (run_dir / "logit_trace.pt").exists():
            continue
        samples = convert_trace(run_dir)
        results["converted_runs"].append(
            {"run": run_dir.name, "n_samples": len(samples)}
        )
        if "_tf_" not in run_dir.name:
            continue
        # Teacher-forced pass: prefix-conditioned greedy acceptance + margins.
        all_flags: list[bool] = []
        flag_lists: list[list[bool]] = []
        fork_margins: list[float] = []
        per_sample = []
        for s in samples:
            flags = [a == t for a, t in zip(s["argmax"], s["token_ids"])]
            margins = [
                m for f, m in zip(flags, s["margin_vs_stream"]) if not f
            ]
            first_reject = next(
                (i for i, f in enumerate(flags) if not f), None
            )
            per_sample.append(
                {
                    "index": s["index"],
                    "steps": s["steps"],
                    "acceptance": sum(flags) / len(flags) if flags else float("nan"),
                    "first_rejection": first_reject,
                    "n_rejections": len(margins),
                }
            )
            all_flags.extend(flags)
            flag_lists.append(flags)
            fork_margins.extend(margins)
        results["teacher_forced"][run_dir.name] = {
            "pooled_positions": len(all_flags),
            "tf_greedy_acceptance": (
                sum(all_flags) / len(all_flags) if all_flags else float("nan")
            ),
            "tf_acceptance_ci95": clopper_pearson_ci(sum(all_flags), len(all_flags)),
            # PRIMARY: exact renewal-loop statistics over TF bitmaps.
            "renewal_at_k": {
                str(k): renewal_stats(flag_lists, k) for k in WINDOW_KS
            },
            "acceptance_at_k": {
                str(k): acceptance_at_k(flag_lists, k) for k in WINDOW_KS
            },
            "expected_accepted_prefix_at_k": {
                str(k): expected_accepted_prefix(flag_lists, k) for k in WINDOW_KS
            },
            # window counts alongside so large-k thinness is visible
            # (per-sample windowing, audit fix)
            "windows_at_k": {
                str(k): windows_at_k(flag_lists, k) for k in WINDOW_KS
            },
            "expected_accepted_prefix_k8": expected_accepted_prefix(flag_lists, 8),
            "fork_margin_quantiles": quantiles(fork_margins),
            "fork_margins": fork_margins,
            "per_sample": per_sample,
        }

    Path(args.out_json).write_text(json.dumps(results, indent=2))

    print(f"{'tf run':<34} {'accept':>7} {'acc@8':>7} {'eap@8':>7} {'forks':>6} {'medMargin':>10}")
    for name, r in results["teacher_forced"].items():
        q = r["fork_margin_quantiles"]
        print(
            f"{name:<34} {r['tf_greedy_acceptance']:>7.3f} "
            f"{r['acceptance_at_k']['8']:>7.3f} "
            f"{r['expected_accepted_prefix_k8']:>7.3f} "
            f"{q.get('n', 0):>6} "
            f"{q.get('p50', float('nan')):>10.3f}"
        )


if __name__ == "__main__":
    main()
