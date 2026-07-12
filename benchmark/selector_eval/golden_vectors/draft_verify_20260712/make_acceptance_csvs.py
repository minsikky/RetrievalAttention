#!/usr/bin/env python3
"""Issue #21 golden-4 (acceptance distribution) + golden-2 part-(d)
(accepted-prefix / rollback per round) CSV producer.

Pure reformatting of the draft-then-verify GPU acceptance ground truth
(``acceptance.json``, produced by analyze_draft_verify.py over the arms in
benchmark_suite_result/draft_verify_gpu_20260711/). No model, no GPU: the
per-token agreement structure of the full-model greedy decodes is the ground
truth for accepted prefixes.

Emits, next to this script:
  golden4_acceptance_aggregate.csv    one row per (mode x ctx) arm
  golden4_acceptance_per_sample.csv   one row per (arm, sample)
  golden4_acceptance_runlengths.csv   one row per agreement run (distribution)
  golden2_accepted_prefix_rounds.csv  one row per (arm, sample, k, round):
      the accepted-prefix length and rollback point of each length-k verify
      round over the sample's decode stream -- the draft-then-verify accept/
      rollback ground truth the RTL verify FSM is A/B'd against, at k=4 and k=8.

Column dictionaries are documented in README.md.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "acceptance_source.json"
ROUND_KS = (4, 8)


def reconstruct_matches(sample: dict) -> list[bool]:
    """Rebuild the per-position token-match boolean array for one sample.

    acceptance.json records aligned_len, first_divergence and the agreement
    run-length list. Every sample in this set has a single leading agreement
    run (run_lengths has length <=1), so matches = [True]*run + [False]*rest.
    Guarded: asserts the reconstruction reproduces the recorded run-lengths and
    agreement rate so a multi-run sample can never be silently mis-encoded.
    """
    n = int(sample["aligned_len"])
    fd = sample["first_divergence"]
    runs = list(sample.get("run_lengths") or [])
    if fd is None:
        matches = [True] * n
    else:
        assert len(runs) <= 1, f"multi-run sample not supported: {runs}"
        lead = int(runs[0]) if runs else int(fd)
        matches = [True] * lead + [False] * (n - lead)
    # guards
    got_runs = []
    cur = 0
    for m in matches:
        if m:
            cur += 1
        elif cur:
            got_runs.append(cur)
            cur = 0
    if cur:
        got_runs.append(cur)
    assert got_runs == [r for r in runs if r], (got_runs, runs)
    rate = (sum(matches) / n) if n else 0.0
    assert abs(rate - float(sample["agreement_rate"])) < 1e-9, (rate, sample["agreement_rate"])
    return matches


def accepted_prefix(matches: list[bool], start: int, k: int) -> int:
    c = 0
    for j in range(k):
        if start + j < len(matches) and matches[start + j]:
            c += 1
        else:
            break
    return c


def main() -> None:
    data = json.loads(SRC.read_text())

    # ---- golden4 aggregate ----
    agg_path = HERE / "golden4_acceptance_aggregate.csv"
    with agg_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "arm", "ctx", "task", "mode", "frozen_run", "draft_run",
            "n_samples", "pooled_aligned_tokens", "pooled_agreement_rate",
            "acceptance_at_4", "acceptance_at_8", "acceptance_at_16",
            "expected_accepted_prefix_k8", "mean_first_divergence",
            "n_fully_identical_streams", "draft_standalone_correct", "frozen_correct",
        ])
        for arm, r in data.items():
            if "aggregate" not in r:
                continue
            a = r["aggregate"]
            acc = a["acceptance_at_k"]
            mode = arm.split("_", 1)[1]
            w.writerow([
                arm, r["ctx"], r["task"], mode, r["frozen_run"], r["draft_run"],
                a["n_samples"], a["pooled_aligned_tokens"], f"{a['pooled_agreement_rate']:.6f}",
                f"{acc['4']:.6f}", f"{acc['8']:.6f}", f"{acc['16']:.6f}",
                f"{a['expected_accepted_prefix_k8']:.6f}",
                ("" if a["mean_first_divergence"] is None else a["mean_first_divergence"]),
                a["n_fully_identical_streams"], a["draft_standalone_correct"], a["frozen_correct"],
            ])

    # ---- golden4 per-sample + run-lengths + golden2 rounds ----
    ps_path = HERE / "golden4_acceptance_per_sample.csv"
    rl_path = HERE / "golden4_acceptance_runlengths.csv"
    rd_path = HERE / "golden2_accepted_prefix_rounds.csv"
    ps = ps_path.open("w", newline="")
    rl = rl_path.open("w", newline="")
    rd = rd_path.open("w", newline="")
    ps_w, rl_w, rd_w = csv.writer(ps), csv.writer(rl), csv.writer(rd)
    ps_w.writerow([
        "arm", "ctx", "mode", "sample_index", "frozen_len", "draft_len", "aligned_len",
        "agreement_rate", "first_divergence", "answer_token_pos",
        "answer_region_passed_before_fork", "run_lengths", "n_runs",
        "frozen_correct", "draft_correct",
    ])
    rl_w.writerow(["arm", "ctx", "mode", "sample_index", "run_ordinal", "run_length"])
    rd_w.writerow([
        "arm", "ctx", "mode", "sample_index", "k", "round_index",
        "round_start_pos", "round_end_pos", "aligned_len_available",
        "accepted_prefix_len", "full_accept", "rollback_pos", "n_rejected_in_round",
    ])
    for arm, r in data.items():
        if "per_sample" not in r:
            continue
        ctx, mode = r["ctx"], arm.split("_", 1)[1]
        for s in r["per_sample"]:
            idx = int(s["index"])
            runs = list(s.get("run_lengths") or [])
            ps_w.writerow([
                arm, ctx, mode, idx, s["frozen_len"], s["draft_len"], s["aligned_len"],
                f"{s['agreement_rate']:.6f}",
                ("" if s["first_divergence"] is None else s["first_divergence"]),
                ("" if s["answer_token_pos"] is None else s["answer_token_pos"]),
                ("" if s["answer_region_passed_before_fork"] is None
                 else int(bool(s["answer_region_passed_before_fork"]))),
                ";".join(str(x) for x in runs), len(runs),
                int(bool(s["frozen_correct"])), int(bool(s["draft_correct"])),
            ])
            for j, rlen in enumerate(runs):
                rl_w.writerow([arm, ctx, mode, idx, j, rlen])
            matches = reconstruct_matches(s)
            n = len(matches)
            for k in ROUND_KS:
                n_rounds = n // k
                for ri in range(n_rounds):
                    start = ri * k
                    ap = accepted_prefix(matches, start, k)
                    rd_w.writerow([
                        arm, ctx, mode, idx, k, ri, start, start + k, k,
                        ap, int(ap == k), start + ap, k - ap,
                    ])
    ps.close(); rl.close(); rd.close()

    print("wrote:")
    for p in (agg_path, ps_path, rl_path, rd_path):
        print("  ", p.name)


if __name__ == "__main__":
    main()
