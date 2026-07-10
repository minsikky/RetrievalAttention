#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch


def _load_trace(path: Path) -> list[dict]:
    trace = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(trace, list):
        raise TypeError(f"trace is not a list: {path}")
    return trace


def _load_score(path: Path) -> float:
    payload = json.loads(path.read_text())
    if "score" not in payload:
        raise KeyError(f"summary has no score: {path}")
    return float(payload["score"])


def _peak_mib(path: Path) -> float:
    peaks = [float(value) for value in re.findall(r"peak=([0-9.]+)MiB", path.read_text())]
    if not peaks:
        raise ValueError(f"no memory trace records found: {path}")
    return max(peaks)


def _validate_aligned(stock: list[dict], candidate: list[dict], label: str) -> None:
    if len(stock) != len(candidate):
        raise ValueError(f"{label} sample count mismatch: {len(stock)} != {len(candidate)}")
    for row, (stock_sample, candidate_sample) in enumerate(zip(stock, candidate)):
        if stock_sample.get("index") != candidate_sample.get("index"):
            raise ValueError(
                f"{label} sample index mismatch at row {row}: "
                f"{stock_sample.get('index')} != {candidate_sample.get('index')}"
            )


def _report_greedy_forks(stock: list[dict], offload: list[dict]) -> None:
    for stock_sample, offload_sample in zip(stock, offload):
        stock_tokens = [int(x) for x in stock_sample["token_ids"]]
        offload_tokens = [int(x) for x in offload_sample["token_ids"]]
        mismatch = next(
            (
                step
                for step, (stock_token, offload_token) in enumerate(zip(stock_tokens, offload_tokens))
                if stock_token != offload_token
            ),
            None,
        )
        if mismatch is None and len(stock_tokens) != len(offload_tokens):
            mismatch = min(len(stock_tokens), len(offload_tokens))
        if mismatch is None:
            print(f"greedy sample={stock_sample['index']} trajectory=match steps={len(stock_tokens)}")
            continue

        detail = ""
        stock_logits = stock_sample["logits"]
        offload_logits = offload_sample["logits"]
        if mismatch < min(int(stock_logits.shape[0]), int(offload_logits.shape[0])):
            stock_step = stock_logits[mismatch].float().reshape(-1)
            offload_step = offload_logits[mismatch].float().reshape(-1)
            delta = float((stock_step - offload_step).abs().max().item())
            top2 = torch.topk(stock_step, k=2)
            margin = float((top2.values[0] - top2.values[1]).item())
            detail = f" max_abs_logit_diff={delta:.8f} stock_margin={margin:.8f}"
        print(
            f"greedy sample={stock_sample['index']} first_fork_step={mismatch} "
            f"stock_token={stock_tokens[mismatch] if mismatch < len(stock_tokens) else 'missing'} "
            f"offload_token={offload_tokens[mismatch] if mismatch < len(offload_tokens) else 'missing'}"
            f"{detail}"
        )


def _teacher_metrics(stock: list[dict], teacher: list[dict]) -> tuple[float, float, float, list[str]]:
    global_max = 0.0
    absolute_sum = 0.0
    element_count = 0
    step_max_sum = 0.0
    step_count = 0
    unsafe_disagreements: list[str] = []
    disagreement_count = 0

    for stock_sample, teacher_sample in zip(stock, teacher):
        stock_tokens = [int(x) for x in stock_sample["token_ids"]]
        teacher_tokens = [int(x) for x in teacher_sample["token_ids"]]
        if teacher_tokens != stock_tokens:
            raise ValueError(f"teacher-forced tokens differ from stock for sample {stock_sample['index']}")
        stock_logits = stock_sample["logits"]
        teacher_logits = teacher_sample["logits"]
        if stock_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"teacher logit shape mismatch for sample {stock_sample['index']}: "
                f"{tuple(stock_logits.shape)} != {tuple(teacher_logits.shape)}"
            )

        for step, (stock_step_raw, teacher_step_raw) in enumerate(zip(stock_logits, teacher_logits)):
            stock_step = stock_step_raw.float().reshape(-1)
            teacher_step = teacher_step_raw.float().reshape(-1)
            absolute = (stock_step - teacher_step).abs()
            step_max = float(absolute.max().item())
            global_max = max(global_max, step_max)
            step_max_sum += step_max
            step_count += 1
            absolute_sum += float(absolute.double().sum().item())
            element_count += int(absolute.numel())

            stock_top2 = torch.topk(stock_step, k=2)
            # Use the same reduction for both arms: torch.topk and argmax can
            # choose different indices for an exact tie.
            stock_argmax = int(torch.argmax(stock_step).item())
            teacher_argmax = int(torch.argmax(teacher_step).item())
            if stock_argmax == teacher_argmax:
                continue
            disagreement_count += 1
            stock_margin = float((stock_top2.values[0] - stock_top2.values[1]).item())
            allowed = stock_margin < 2.0 * step_max
            print(
                f"teacher_argmax_disagreement sample={stock_sample['index']} step={step} "
                f"stock_argmax={stock_argmax} offload_argmax={teacher_argmax} "
                f"stock_margin={stock_margin:.8f} max_abs_logit_diff={step_max:.8f} "
                f"allowed_near_tie={str(allowed).lower()}"
            )
            if not allowed:
                unsafe_disagreements.append(
                    f"sample {stock_sample['index']} step {step}: margin {stock_margin:.8f} "
                    f">= 2*delta {2.0 * step_max:.8f}"
                )

    mean_absolute = absolute_sum / max(1, element_count)
    print(
        f"teacher_forced_steps={sum(int(sample['logits'].shape[0]) for sample in teacher)} "
        f"teacher_forced_elements={element_count} argmax_disagreements={disagreement_count}"
    )
    return global_max, mean_absolute, step_max_sum / max(1, step_count), unsafe_disagreements


def run(args: argparse.Namespace) -> int:
    try:
        stock = _load_trace(args.stock_trace)
        offload = _load_trace(args.offload_trace)
        teacher = _load_trace(args.teacher_trace)
        _validate_aligned(stock, offload, "greedy offload")
        _validate_aligned(stock, teacher, "teacher offload")
        _report_greedy_forks(stock, offload)

        stock_score = _load_score(args.stock_summary)
        offload_score = _load_score(args.offload_summary)
        teacher_score = _load_score(args.teacher_summary)
        scores_equal = stock_score == offload_score == teacher_score
        print(
            f"task_scores stock={stock_score:.8f} offload={offload_score:.8f} "
            f"teacher={teacher_score:.8f} equal={str(scores_equal).lower()}"
        )

        teacher_max, teacher_mean, mean_step_max, unsafe = _teacher_metrics(stock, teacher)
        print(
            f"teacher_forced_max_abs_logit_diff={teacher_max:.8f} "
            f"teacher_forced_mean_abs_logit_diff={teacher_mean:.8f} "
            f"teacher_forced_mean_step_max_abs_logit_diff={mean_step_max:.8f} "
            f"threshold={args.max_logit_diff:.8f}"
        )
        for name, path in (
            ("stock", args.stock_console),
            ("offload", args.offload_console),
            ("teacher", args.teacher_console),
        ):
            print(f"{name}_peak_memory_mib={_peak_mib(path):.1f}")

        reasons = []
        if not scores_equal:
            reasons.append("task scores differ")
        if not teacher_max < args.max_logit_diff:
            reasons.append(
                f"teacher max logit diff {teacher_max:.8f} is not below {args.max_logit_diff:.8f}"
            )
        if unsafe:
            reasons.append("unsafe argmax disagreements: " + "; ".join(unsafe))
        if reasons:
            print("KVOFF-AB: FAIL (" + " | ".join(reasons) + ")")
            return 1
        print(
            "KVOFF-AB: PASS (task scores identical; teacher-forced max logit diff below threshold; "
            "all argmax disagreements, if any, are margin-qualified near-ties)"
        )
        return 0
    except Exception as exc:
        print(f"KVOFF-AB: FAIL (comparison error: {exc})")
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stock and CPU-KV-offloaded dense runs")
    parser.add_argument("--stock-trace", type=Path, required=True)
    parser.add_argument("--offload-trace", type=Path, required=True)
    parser.add_argument("--teacher-trace", type=Path, required=True)
    parser.add_argument("--stock-summary", type=Path, required=True)
    parser.add_argument("--offload-summary", type=Path, required=True)
    parser.add_argument("--teacher-summary", type=Path, required=True)
    parser.add_argument("--stock-console", type=Path, required=True)
    parser.add_argument("--offload-console", type=Path, required=True)
    parser.add_argument("--teacher-console", type=Path, required=True)
    parser.add_argument("--max-logit-diff", type=float, default=0.1)
    args = parser.parse_args()
    if args.max_logit_diff <= 0:
        parser.error("--max-logit-diff must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
