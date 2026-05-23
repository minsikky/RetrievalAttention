from __future__ import annotations

import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch


def _sync_cuda_if_needed(sync_cuda: bool) -> None:
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _query_len(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args and torch.is_tensor(args[0]):
        input_ids = args[0]
    if torch.is_tensor(input_ids) and input_ids.ndim >= 2:
        return int(input_ids.shape[-1])

    inputs_embeds = kwargs.get("inputs_embeds")
    if torch.is_tensor(inputs_embeds) and inputs_embeds.ndim >= 3:
        return int(inputs_embeds.shape[-2])
    return 0


@dataclass
class ForwardPhaseStats:
    calls: int = 0
    seconds: float = 0.0
    input_tokens: int = 0
    max_input_tokens: int = 0

    def add(self, query_tokens: int, seconds: float) -> None:
        self.calls += 1
        self.seconds += float(seconds)
        self.input_tokens += int(max(0, query_tokens))
        self.max_input_tokens = max(self.max_input_tokens, int(max(0, query_tokens)))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "calls": int(self.calls),
            "seconds": float(self.seconds),
            "input_tokens": int(self.input_tokens),
            "max_input_tokens": int(self.max_input_tokens),
            "ms_per_call": float(1000.0 * self.seconds / max(1, self.calls)),
            "ms_per_input_token": float(1000.0 * self.seconds / max(1, self.input_tokens)),
        }


@dataclass
class GenerationExecutionProfile:
    prefill: ForwardPhaseStats = field(default_factory=ForwardPhaseStats)
    decode: ForwardPhaseStats = field(default_factory=ForwardPhaseStats)
    other: ForwardPhaseStats = field(default_factory=ForwardPhaseStats)

    def record(self, query_tokens: int, seconds: float) -> None:
        if int(query_tokens) > 1:
            self.prefill.add(query_tokens, seconds)
        elif int(query_tokens) == 1:
            self.decode.add(query_tokens, seconds)
        else:
            self.other.add(query_tokens, seconds)

    @property
    def total_forward_seconds(self) -> float:
        return float(self.prefill.seconds + self.decode.seconds + self.other.seconds)

    def to_dict(self, *, generation_sec: float | None = None, generated_tokens: int | None = None) -> dict[str, Any]:
        total_forward = self.total_forward_seconds
        generation = float(generation_sec) if generation_sec is not None else total_forward
        generated = int(generated_tokens) if generated_tokens is not None else int(self.decode.calls)
        payload: dict[str, Any] = {
            "prefill": self.prefill.to_dict(),
            "decode": self.decode.to_dict(),
            "other": self.other.to_dict(),
            "total_forward_seconds": float(total_forward),
            "generation_seconds": float(generation),
            "forward_overhead_seconds": float(max(0.0, generation - total_forward)),
            "prefill_fraction_of_generation": float(self.prefill.seconds / max(generation, 1e-9)),
            "decode_fraction_of_generation": float(self.decode.seconds / max(generation, 1e-9)),
            "other_fraction_of_generation": float(self.other.seconds / max(generation, 1e-9)),
            "generated_tokens": int(generated),
            "decode_ms_per_generated_token": float(1000.0 * self.decode.seconds / max(1, generated)),
        }
        return payload


@contextmanager
def profile_model_forward(
    model: Any,
    *,
    enabled: bool = True,
    sync_cuda: bool = True,
) -> Iterator[GenerationExecutionProfile | None]:
    if not enabled:
        yield None
        return

    profile = GenerationExecutionProfile()
    original_forward = model.forward

    def wrapped_forward(_self: Any, *args: Any, **kwargs: Any):
        query_tokens = _query_len(args, kwargs)
        _sync_cuda_if_needed(sync_cuda)
        start = time.perf_counter()
        result = original_forward(*args, **kwargs)
        _sync_cuda_if_needed(sync_cuda)
        profile.record(query_tokens, time.perf_counter() - start)
        return result

    model.forward = types.MethodType(wrapped_forward, model)
    try:
        yield profile
    finally:
        model.forward = original_forward


def aggregate_execution_profiles(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    profiles = [row.get("execution_profile") for row in rows if isinstance(row.get("execution_profile"), dict)]
    if not profiles:
        return {}

    def phase_sum(phase: str, key: str) -> float:
        return float(sum(float(profile.get(phase, {}).get(key, 0.0)) for profile in profiles))

    def root_sum(key: str) -> float:
        return float(sum(float(profile.get(key, 0.0)) for profile in profiles))

    n = max(1, len(profiles))
    total_generation = root_sum("generation_seconds")
    prefill_sec = phase_sum("prefill", "seconds")
    decode_sec = phase_sum("decode", "seconds")
    other_sec = phase_sum("other", "seconds")
    generated_tokens = int(sum(int(profile.get("generated_tokens", 0)) for profile in profiles))
    return {
        "examples": int(len(profiles)),
        "total_generation_seconds": float(total_generation),
        "total_prefill_forward_seconds": float(prefill_sec),
        "total_decode_forward_seconds": float(decode_sec),
        "total_other_forward_seconds": float(other_sec),
        "total_forward_overhead_seconds": float(root_sum("forward_overhead_seconds")),
        "prefill_fraction_of_generation": float(prefill_sec / max(total_generation, 1e-9)),
        "decode_fraction_of_generation": float(decode_sec / max(total_generation, 1e-9)),
        "other_fraction_of_generation": float(other_sec / max(total_generation, 1e-9)),
        "mean_generation_seconds": float(total_generation / n),
        "mean_prefill_forward_seconds": float(prefill_sec / n),
        "mean_decode_forward_seconds": float(decode_sec / n),
        "mean_decode_ms_per_generated_token": float(1000.0 * decode_sec / max(1, generated_tokens)),
        "prefill_forward_calls": int(phase_sum("prefill", "calls")),
        "decode_forward_calls": int(phase_sum("decode", "calls")),
        "other_forward_calls": int(phase_sum("other", "calls")),
        "prefill_input_tokens": int(phase_sum("prefill", "input_tokens")),
        "decode_input_tokens": int(phase_sum("decode", "input_tokens")),
        "generated_tokens": int(generated_tokens),
    }
