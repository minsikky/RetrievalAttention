#!/usr/bin/env python3
"""Attention-output noise injection for relL2 -> task-quality calibration.

Wraps each patched decoder layer's self-attention forward in an otherwise
dense run and adds isotropic Gaussian noise to the post-o_proj attention
output at a controlled relative L2 magnitude. This emulates the per-layer,
per-decode-step o-proj relL2 error that the frontier trace experiments use
as their quality proxy, so task-score degradation vs injected relL2
calibrates whether the tau=0.002 proxy operating point (and observed drift
up to ~0.003) is conservative, right, or loose.

Caveat: real frontier error is structured (correlated with dropped
low-probability tokens), not isotropic. Matched-relL2 Gaussian noise is a
necessary-condition calibration, not a sufficient one.

Activation is env-driven so eval wrappers stay unchanged:
  ATTN_OUTPUT_NOISE_REL_L2  relative L2 of injected noise (0/unset = off)
  ATTN_OUTPUT_NOISE_SCOPE   "decode" (default; query_len==1 only) or "all"
  ATTN_OUTPUT_NOISE_SEED    RNG seed (default 0)
  ATTN_OUTPUT_NOISE_LAYERS  csv layer ids, default all layers
"""
from __future__ import annotations

import contextlib
import os
import types

import torch


def attn_noise_env_config() -> dict | None:
    raw = os.environ.get("ATTN_OUTPUT_NOISE_REL_L2", "").strip()
    if not raw:
        return None
    rel_l2 = float(raw)
    if rel_l2 <= 0.0:
        return None
    scope = os.environ.get("ATTN_OUTPUT_NOISE_SCOPE", "decode").strip().lower()
    if scope not in {"decode", "all"}:
        raise ValueError(f"ATTN_OUTPUT_NOISE_SCOPE must be decode|all, got {scope!r}")
    layers_raw = os.environ.get("ATTN_OUTPUT_NOISE_LAYERS", "").strip()
    layer_ids = [int(x) for x in layers_raw.split(",") if x.strip()] if layers_raw else None
    return {
        "rel_l2": rel_l2,
        "scope": scope,
        "seed": int(os.environ.get("ATTN_OUTPUT_NOISE_SEED", "0")),
        "layer_ids": layer_ids,
    }


@contextlib.contextmanager
def attn_output_noise_patch(
    model,
    rel_l2: float,
    scope: str = "decode",
    seed: int = 0,
    layer_ids: list[int] | None = None,
):
    """Add rel_l2-scaled Gaussian noise to self_attn outputs of `model`."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    if layer_ids is None:
        layer_ids = list(range(len(model.model.layers)))
    originals: dict[int, object] = {}
    counters = {"noised_calls": 0, "passthrough_calls": 0}

    def make_forward(module):
        original_forward = module.forward

        def forward(self, hidden_states, *fwd_args, **fwd_kwargs):
            out = original_forward(hidden_states, *fwd_args, **fwd_kwargs)
            query_len = int(hidden_states.shape[1])
            if scope == "decode" and query_len != 1:
                counters["passthrough_calls"] += 1
                return out
            attn_output = out[0] if isinstance(out, tuple) else out
            attn_f32 = attn_output.float()
            attn_norm = torch.linalg.vector_norm(attn_f32)
            noise = torch.randn(
                attn_f32.shape, generator=generator, device=attn_f32.device, dtype=torch.float32
            )
            noise_norm = torch.linalg.vector_norm(noise)
            scale = float(rel_l2) * attn_norm / torch.clamp(noise_norm, min=1e-20)
            noised = (attn_f32 + noise * scale).to(attn_output.dtype)
            counters["noised_calls"] += 1
            if isinstance(out, tuple):
                return (noised,) + tuple(out[1:])
            return noised

        return types.MethodType(forward, module)

    try:
        for layer_id in layer_ids:
            module = model.model.layers[int(layer_id)].self_attn
            originals[int(layer_id)] = module.forward
            module.forward = make_forward(module)
        yield counters
    finally:
        for layer_id, fwd in originals.items():
            model.model.layers[int(layer_id)].self_attn.forward = fwd


def maybe_attn_output_noise_patch(model):
    """Env-driven context: returns (context_manager, config_dict_or_None)."""
    config = attn_noise_env_config()
    if config is None:
        return contextlib.nullcontext(None), None
    return (
        attn_output_noise_patch(
            model,
            rel_l2=config["rel_l2"],
            scope=config["scope"],
            seed=config["seed"],
            layer_ids=config["layer_ids"],
        ),
        config,
    )
