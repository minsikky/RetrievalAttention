#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import torch


@dataclass(frozen=True)
class ComparisonMetrics:
    max_absolute: float
    mean_absolute: float
    mean_step_max: float
    unsafe_disagreements: tuple[str, ...]
    disagreement_count: int
    step_count: int
    element_count: int
    max_sample: int | str | None
    max_step: int
    max_token_position: int | None
    max_vocab_id: int


def _load_trace(path: Path) -> list[dict]:
    trace = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(trace, list):
        raise TypeError(f"trace is not a list: {path}")
    return trace


def _load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if "score" not in payload:
        raise KeyError(f"summary has no score: {path}")
    return payload


def _report_arm_timing(name: str, summary: dict) -> float:
    samples = int(summary["samples"])
    elapsed = float(summary["elapsed_seconds"])
    seconds_per_sample = elapsed / max(1, samples)
    print(
        f"arm_wall_clock name={name} elapsed_seconds={elapsed:.6f} samples={samples} "
        f"seconds_per_sample={seconds_per_sample:.6f} "
        f"mean_prefill_seconds={float(summary['mean_stream_prefill_seconds']):.6f} "
        f"mean_decode_seconds={float(summary['mean_stream_decode_seconds']):.6f}"
    )
    return seconds_per_sample


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


def _comparison_metrics(
    stock: list[dict],
    candidate: list[dict],
    *,
    label: str,
) -> ComparisonMetrics:
    global_max = -1.0
    absolute_sum = 0.0
    element_count = 0
    step_max_sum = 0.0
    step_count = 0
    unsafe_disagreements: list[str] = []
    disagreement_count = 0
    max_sample = None
    max_step = -1
    max_token_position = None
    max_vocab_id = -1

    for stock_sample, candidate_sample in zip(stock, candidate):
        stock_tokens = [int(x) for x in stock_sample["token_ids"]]
        candidate_tokens = [int(x) for x in candidate_sample["token_ids"]]
        if candidate_tokens != stock_tokens:
            raise ValueError(f"{label} forced tokens differ from stock for sample {stock_sample['index']}")
        stock_prompt_tokens = stock_sample.get("prompt_tokens")
        candidate_prompt_tokens = candidate_sample.get("prompt_tokens")
        if (
            stock_prompt_tokens is not None
            and candidate_prompt_tokens is not None
            and int(stock_prompt_tokens) != int(candidate_prompt_tokens)
        ):
            raise ValueError(
                f"{label} prompt length differs for sample {stock_sample['index']}: "
                f"{stock_prompt_tokens} != {candidate_prompt_tokens}"
            )
        stock_logits = stock_sample["logits"]
        candidate_logits = candidate_sample["logits"]
        if stock_logits.shape != candidate_logits.shape:
            raise ValueError(
                f"{label} logit shape mismatch for sample {stock_sample['index']}: "
                f"{tuple(stock_logits.shape)} != {tuple(candidate_logits.shape)}"
            )

        for step, (stock_step_raw, candidate_step_raw) in enumerate(zip(stock_logits, candidate_logits)):
            stock_step = stock_step_raw.float().reshape(-1)
            candidate_step = candidate_step_raw.float().reshape(-1)
            absolute = (stock_step - candidate_step).abs()
            step_max = float(absolute.max().item())
            if step_max > global_max:
                global_max = step_max
                max_sample = stock_sample["index"]
                max_step = step
                max_token_position = (
                    int(stock_prompt_tokens) + step if stock_prompt_tokens is not None else None
                )
                max_vocab_id = int(torch.argmax(absolute).item())
            step_max_sum += step_max
            step_count += 1
            absolute_sum += float(absolute.double().sum().item())
            element_count += int(absolute.numel())

            stock_top2 = torch.topk(stock_step, k=2)
            # Use the same reduction for both arms: torch.topk and argmax can
            # choose different indices for an exact tie.
            stock_argmax = int(torch.argmax(stock_step).item())
            candidate_argmax = int(torch.argmax(candidate_step).item())
            if stock_argmax == candidate_argmax:
                continue
            disagreement_count += 1
            stock_margin = float((stock_top2.values[0] - stock_top2.values[1]).item())
            allowed = stock_margin < 2.0 * step_max
            print(
                f"{label}_argmax_disagreement sample={stock_sample['index']} step={step} "
                f"stock_argmax={stock_argmax} candidate_argmax={candidate_argmax} "
                f"stock_margin={stock_margin:.8f} max_abs_logit_diff={step_max:.8f} "
                f"allowed_near_tie={str(allowed).lower()}"
            )
            if not allowed:
                unsafe_disagreements.append(
                    f"sample {stock_sample['index']} step {step}: margin {stock_margin:.8f} "
                    f">= 2*delta {2.0 * step_max:.8f}"
                )

    if step_count == 0:
        raise ValueError(f"{label} trace has no decode steps")
    mean_absolute = absolute_sum / element_count
    print(
        f"{label}_steps={step_count} {label}_elements={element_count} "
        f"{label}_argmax_disagreements={disagreement_count}"
    )
    print(
        f"{label}_max_abs_logit_diff={global_max:.8f} "
        f"{label}_mean_abs_logit_diff={mean_absolute:.8f} "
        f"{label}_mean_step_max_abs_logit_diff={step_max_sum / step_count:.8f}"
    )
    token_position = "unknown" if max_token_position is None else str(max_token_position)
    print(
        f"{label}_global_max_delta sample={max_sample} step={max_step} "
        f"token_position={token_position} vocab_id={max_vocab_id} delta={global_max:.8f}"
    )
    return ComparisonMetrics(
        max_absolute=global_max,
        mean_absolute=mean_absolute,
        mean_step_max=step_max_sum / step_count,
        unsafe_disagreements=tuple(unsafe_disagreements),
        disagreement_count=disagreement_count,
        step_count=step_count,
        element_count=element_count,
        max_sample=max_sample,
        max_step=max_step,
        max_token_position=max_token_position,
        max_vocab_id=max_vocab_id,
    )


def run(args: argparse.Namespace) -> int:
    try:
        stock = _load_trace(args.stock_trace)
        offload = _load_trace(args.offload_trace)
        teacher = _load_trace(args.teacher_trace)
        calibration = _load_trace(args.calibration_trace)
        _validate_aligned(stock, offload, "greedy offload")
        _validate_aligned(stock, teacher, "teacher offload")
        _validate_aligned(stock, calibration, "stock calibration")
        _report_greedy_forks(stock, offload)

        stock_summary = _load_summary(args.stock_summary)
        offload_summary = _load_summary(args.offload_summary)
        teacher_summary = _load_summary(args.teacher_summary)
        calibration_summary = _load_summary(args.calibration_summary)
        stock_score = float(stock_summary["score"])
        offload_score = float(offload_summary["score"])
        teacher_score = float(teacher_summary["score"])
        calibration_score = float(calibration_summary["score"])
        scores_equal = stock_score == offload_score == teacher_score == calibration_score
        print(
            f"task_scores stock={stock_score:.8f} offload={offload_score:.8f} "
            f"teacher={teacher_score:.8f} calibration={calibration_score:.8f} "
            f"equal={str(scores_equal).lower()}"
        )
        stock_seconds = _report_arm_timing("stock", stock_summary)
        offload_seconds = _report_arm_timing("offload", offload_summary)
        _report_arm_timing("teacher", teacher_summary)
        _report_arm_timing("calibration", calibration_summary)
        print(
            "arm_wall_clock_ratio "
            f"offload_over_stock={offload_seconds / max(stock_seconds, 1e-9):.6f}"
        )

        offload_metrics = _comparison_metrics(
            stock,
            teacher,
            label="offload_teacher_forced",
        )
        calibration_metrics = _comparison_metrics(
            stock,
            calibration,
            label="calibration",
        )
        for name, path in (
            ("stock", args.stock_console),
            ("offload", args.offload_console),
            ("teacher", args.teacher_console),
            ("calibration", args.calibration_console),
        ):
            print(f"{name}_peak_memory_mib={_peak_mib(path):.1f}")

        scaled_floor = args.floor_mult * calibration_metrics.max_absolute
        effective_threshold = max(args.max_logit_diff, scaled_floor)
        gate_summary = (
            f"absolute_threshold={args.max_logit_diff:.8f} "
            f"calibration_floor_max_abs={calibration_metrics.max_absolute:.8f} "
            f"floor_mult={args.floor_mult:.8f} scaled_floor={scaled_floor:.8f} "
            f"effective_threshold={effective_threshold:.8f} "
            f"offload_max_abs={offload_metrics.max_absolute:.8f}"
        )
        print(f"noise_floor_gate {gate_summary}")

        reasons = []
        if not scores_equal:
            reasons.append("task scores differ")
        if offload_metrics.max_absolute > effective_threshold:
            reasons.append(
                f"offload max logit diff {offload_metrics.max_absolute:.8f} "
                f"exceeds effective threshold {effective_threshold:.8f}"
            )
        if offload_metrics.unsafe_disagreements:
            reasons.append(
                "unsafe offload argmax disagreements: "
                + "; ".join(offload_metrics.unsafe_disagreements)
            )
        if calibration_metrics.unsafe_disagreements:
            reasons.append(
                "unsafe calibration argmax disagreements: "
                + "; ".join(calibration_metrics.unsafe_disagreements)
            )
        if reasons:
            print("KVOFF-AB: FAIL (" + gate_summary + " | " + " | ".join(reasons) + ")")
            return 1
        print(
            "KVOFF-AB: PASS ("
            + gate_summary
            + " | task scores identical; offload max is within the calibrated threshold; "
            "all offload/calibration argmax disagreements, if any, are margin-qualified near-ties)"
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
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--stock-summary", type=Path, required=True)
    parser.add_argument("--offload-summary", type=Path, required=True)
    parser.add_argument("--teacher-summary", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--stock-console", type=Path, required=True)
    parser.add_argument("--offload-console", type=Path, required=True)
    parser.add_argument("--teacher-console", type=Path, required=True)
    parser.add_argument("--calibration-console", type=Path, required=True)
    parser.add_argument("--max-logit-diff", type=float, default=0.1)
    parser.add_argument("--floor-mult", type=float, default=2.0)
    args = parser.parse_args()
    if args.max_logit_diff <= 0:
        parser.error("--max-logit-diff must be positive")
    if args.floor_mult < 0:
        parser.error("--floor-mult must be non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
