import os

from flash_attn import flash_attn_with_kvcache
from flash_attn import flash_attn_with_kvcache_retrieval


def _call_flash_attn_fused_prefill(
    query_states,
    key_states,
    value_states,
    causal,
    retrievalattention_cache,
):
    payload = flash_attn_with_kvcache_retrieval(
        q=query_states,
        k_cache=key_states,
        v_cache=value_states,
        causal=causal,
        retrieval_topk=int(retrievalattention_cache.q_knn),
        retrieval_group_size=int(retrievalattention_cache.group_size),
        retrieval_normalize=bool(getattr(retrievalattention_cache, "score_normalize", False)),
        return_retrieval_idx=True,
    )
    if not isinstance(payload, tuple) or len(payload) < 2:
        raise RuntimeError(
            "flash_attn_with_kvcache_retrieval must return (attn_out, retrieval_topk_idx[, profile])."
        )

    attn_out = payload[0]
    topk_idx = payload[1]
    profile = payload[2] if len(payload) >= 3 and isinstance(payload[2], dict) else {}
    return attn_out, topk_idx, profile


def retrievalattention_prefill_attn(
    query_states,
    key_states,
    value_states,
    causal,
    layer_idx,
    retrievalattention_cache,
):
    # Use full attention during prefill to obtain accurate outputs.
    if not retrievalattention_cache.uses_flashattn_fused_prefill():
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
