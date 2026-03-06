import os

from flash_attn import flash_attn_with_kvcache
from flash_attn import flash_attn_with_kvcache_retrieval
try:
    from flash_attn import flash_attn_with_kvcache_retrieval_graph
except Exception:
    flash_attn_with_kvcache_retrieval_graph = None


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


def _call_flash_attn_fused_prefill_graph(
    query_states,
    key_states,
    value_states,
    causal,
    retrievalattention_cache,
):
    if flash_attn_with_kvcache_retrieval_graph is None:
        raise RuntimeError(
            "flash_attn_with_kvcache_retrieval_graph is unavailable. "
            "Rebuild/install flash-attn fork with graph API support."
        )
    payload = flash_attn_with_kvcache_retrieval_graph(
        q=query_states,
        k_cache=key_states,
        v_cache=value_states,
        causal=causal,
        retrieval_topk=int(retrievalattention_cache.q_knn),
        retrieval_group_size=int(retrievalattention_cache.group_size),
        retrieval_normalize=bool(getattr(retrievalattention_cache, "score_normalize", False)),
        return_retrieval_idx=True,
        graph_nq=int(getattr(retrievalattention_cache, "roar_nq", retrievalattention_cache.q_knn)),
        graph_degree=int(getattr(retrievalattention_cache, "roar_m", retrievalattention_cache.key_degree)),
        graph_dynamic_start=int(getattr(retrievalattention_cache, "dynamic_start", 0)),
        graph_dynamic_end=int(getattr(retrievalattention_cache, "dynamic_end", query_states.shape[1])),
    )
    if not isinstance(payload, tuple) or len(payload) < 4:
        raise RuntimeError(
            "flash_attn_with_kvcache_retrieval_graph must return "
            "(attn_out, retrieval_topk_idx, graph_neighbors[, profile])."
        )
    attn_out = payload[0]
    topk_idx = payload[1]
    graph_neighbors = payload[2]
    profile = payload[3] if len(payload) >= 4 and isinstance(payload[3], dict) else {}
    return attn_out, topk_idx, graph_neighbors, profile


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

    if retrievalattention_cache.uses_flashattn_fused_graph_prefill():
        try:
            attn_out, topk_idx, graph_neighbors, profile = _call_flash_attn_fused_prefill_graph(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                causal=causal,
                retrievalattention_cache=retrievalattention_cache,
            )
            shape = tuple(topk_idx.shape) if hasattr(topk_idx, "shape") else "unknown"
            graph_shape = tuple(graph_neighbors.shape) if hasattr(graph_neighbors, "shape") else "unknown"
            retrievalattention_cache.register_fused_prefill_knn(
                layer_idx=layer_idx,
                knn_idx=topk_idx,
                profile=profile,
                graph_neighbors=graph_neighbors,
                graph_profile=profile,
            )
            del graph_neighbors
        except Exception as exc:
            if retrievalattention_cache.fa_graph_fused_require:
                raise
            print(
                "[RetrievalAttention] WARNING: graph-fused prefill path failed; "
                f"{type(exc).__name__}: {exc}. Falling back to topk-only fused path.",
                flush=True,
            )
            attn_out, topk_idx, profile = _call_flash_attn_fused_prefill(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                causal=causal,
                retrievalattention_cache=retrievalattention_cache,
            )
            shape = tuple(topk_idx.shape) if hasattr(topk_idx, "shape") else "unknown"
            graph_shape = "n/a"
            retrievalattention_cache.register_fused_prefill_knn(
                layer_idx=layer_idx,
                knn_idx=topk_idx,
                profile=profile,
            )
    else:
        attn_out, topk_idx, profile = _call_flash_attn_fused_prefill(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            causal=causal,
            retrievalattention_cache=retrievalattention_cache,
        )
        shape = tuple(topk_idx.shape) if hasattr(topk_idx, "shape") else "unknown"
        graph_shape = "off"
        retrievalattention_cache.register_fused_prefill_knn(
            layer_idx=layer_idx,
            knn_idx=topk_idx,
            profile=profile,
        )
    # Drop per-layer retrieval tensor reference as early as possible.
    del topk_idx
    if os.environ.get("RETRIEVALATTN_PROFILE", "1") == "1":
        print(
            f"[RetrievalAttention] flashattn fused prefill layer={layer_idx} "
            f"topk_shape={shape} graph_shape={graph_shape}",
            flush=True,
        )
    return attn_out


def retrievalattention_decode_attn(query_states, key_states, value_states, layer_idx, retrievalattention_cache):
    attn_out = retrievalattention_cache.compute(
        query_states.contiguous(), layer_idx
    )
    return attn_out
