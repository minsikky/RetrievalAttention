import inspect
import os
from typing import Any, Dict, Tuple

from flash_attn import flash_attn_with_kvcache

try:
    from flash_attn import flash_attn_with_kvcache_retrieval  # type: ignore
except Exception:
    try:
        from flash_attn.flash_attn_interface import (  # type: ignore
            flash_attn_with_kvcache_retrieval,
        )
    except Exception:
        flash_attn_with_kvcache_retrieval = None


def _extract_fused_prefill_outputs(payload: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Normalize possible return formats from fused FlashAttention wrappers:
      1) (attn_out, topk_idx)
      2) (attn_out, topk_idx, profile_dict)
      3) dict with keys for output/index/profile
    """
    if isinstance(payload, tuple):
        if len(payload) == 2:
            return payload[0], payload[1], {}
        if len(payload) >= 3:
            profile = payload[2] if isinstance(payload[2], dict) else {}
            return payload[0], payload[1], profile

    if isinstance(payload, dict):
        out = payload.get("out", payload.get("attn_out"))
        idx = payload.get(
            "retrieval_topk_idx",
            payload.get("topk_idx", payload.get("ra_topk_idx")),
        )
        profile = payload.get("profile", {})
        if out is None or idx is None:
            raise RuntimeError(
                "fused prefill FlashAttention returned dict without required out/topk fields."
            )
        if not isinstance(profile, dict):
            profile = {}
        return out, idx, profile

    raise RuntimeError("Unsupported return format from fused prefill FlashAttention API.")


def _call_flash_attn_fused_prefill(
    query_states,
    key_states,
    value_states,
    causal,
    retrievalattention_cache,
):
    if flash_attn_with_kvcache_retrieval is None:
        raise RuntimeError(
            "RETRIEVALATTN_FA_FUSED_PREFILL=1 requires a flash-attn build "
            "that exports flash_attn_with_kvcache_retrieval."
        )

    base_kwargs = {
        "q": query_states,
        "k_cache": key_states,
        "v_cache": value_states,
        "causal": causal,
    }
    optional_value_by_name = {
        "retrieval_topk": int(retrievalattention_cache.q_knn),
        "k_top": int(retrievalattention_cache.q_knn),
        "retrieval_k": int(retrievalattention_cache.q_knn),
        "ra_topk": int(retrievalattention_cache.q_knn),
        "retrieval_group_size": int(retrievalattention_cache.group_size),
        "group_size": int(retrievalattention_cache.group_size),
        "retrieval_normalize": bool(getattr(retrievalattention_cache, "score_normalize", False)),
        "return_retrieval_idx": True,
        "return_topk_idx": True,
    }

    call_kwargs = dict(base_kwargs)
    try:
        sig = inspect.signature(flash_attn_with_kvcache_retrieval)
        for name, value in optional_value_by_name.items():
            if name in sig.parameters:
                call_kwargs[name] = value
    except Exception:
        # Some extension symbols may not expose an inspectable signature.
        pass

    payload = flash_attn_with_kvcache_retrieval(**call_kwargs)
    return _extract_fused_prefill_outputs(payload)


def retrievalattention_prefill_attn(
    query_states,
    key_states,
    value_states,
    causal,
    layer_idx,
    retrievalattention_cache,
):
    fused_enabled = retrievalattention_cache.uses_flashattn_fused_prefill()

    # Use full attention during prefill to obtain accurate outputs.
    if not fused_enabled:
        attn_out = flash_attn_with_kvcache(
            q=query_states,
            k_cache=key_states,
            v_cache=value_states,
            causal=causal,
        )
        return attn_out

    attn_out, topk_idx, profile = _call_flash_attn_fused_prefill(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        causal=causal,
        retrievalattention_cache=retrievalattention_cache,
    )
    shape = tuple(topk_idx.shape) if hasattr(topk_idx, "shape") else "unknown"
    retrievalattention_cache.register_fused_prefill_knn(
        layer_idx=layer_idx,
        knn_idx=topk_idx,
        profile=profile,
    )
    # Drop per-layer retrieval tensor reference as early as possible.
    del topk_idx
    if os.environ.get("RETRIEVALATTN_PROFILE", "1") == "1":
        print(
            f"[RetrievalAttention] flashattn fused prefill layer={layer_idx} topk_shape={shape}",
            flush=True,
        )
    return attn_out


def retrievalattention_decode_attn(query_states, key_states, value_states, layer_idx, retrievalattention_cache):
    attn_out = retrievalattention_cache.compute(
        query_states.contiguous(), layer_idx
    )
    return attn_out
