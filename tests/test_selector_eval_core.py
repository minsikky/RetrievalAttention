from __future__ import annotations

import numpy as np

from benchmark.selector_eval.costs.base import CostTrace, kv_read_bytes
from benchmark.selector_eval.data.trace import attention_probs, static_tokens
from benchmark.selector_eval.data.trace import QKVTrace
from benchmark.selector_eval.metrics.attention import compute_metrics
from benchmark.selector_eval.selectors.base import QueryState
from benchmark.selector_eval.selectors.ivfpq import IVFPQSelector
from benchmark.selector_eval.selectors.hybrid import PageSparQPostingsPQSelector, PageSparQPQSelector, PagedPQSparQRerankSelector
from benchmark.selector_eval.selectors.magicpig import MagicPIGSelector
from benchmark.selector_eval.selectors.oracle import DenseSelector, TopMassOracleSelector, selector_from_name
from benchmark.selector_eval.selectors.paged_pq import PagedPQSelector
from benchmark.selector_eval.selectors.pqcache import PQCacheFullScanSelector
from benchmark.selector_eval.selectors.retroinfer import RetroInferStyleSelector
from benchmark.selector_eval.selectors.retrievalattention import RetrievalAttentionGraphSelector
from benchmark.selector_eval.selectors.sparq import SparQSelector


def _state() -> QueryState:
    keys = np.eye(4, dtype=np.float32)
    values = np.eye(4, dtype=np.float32)
    query = np.asarray([4.0, 1.0, 0.0, 0.0], dtype=np.float32)
    scores, probs = attention_probs(keys, query)
    return QueryState(
        decode_tokens=0,
        position=3,
        qidx=0,
        head=0,
        kv_head=0,
        query=query,
        keys=keys,
        values=values,
        scores=scores,
        probs=probs,
        base_tokens=[],
    )


def test_cost_trace_splits_phases_and_categories() -> None:
    cost = CostTrace()
    cost.read("selector", "pq_codes", 1024)
    cost.read("exact_attention", "exact_kv", 2048)
    cost.write("online_update", "page_codes", 512)

    assert cost.bytes(kind="read") == 3072
    assert cost.bytes(phase="selector") == 1024
    assert cost.bytes(category="exact_kv") == 2048
    assert kv_read_bytes(3, head_dim=4, key_bytes=2, value_bytes=2) == 48


def test_top_mass_oracle_reaches_target_with_prefix() -> None:
    state = _state()
    state.base_tokens.append(3)
    result = TopMassOracleSelector().select(state, target_mass=0.80)
    selected_mass = float(state.probs[np.asarray(result.selected_tokens, dtype=np.int64)].sum())

    assert 3 in result.selected_tokens
    assert selected_mass >= 0.80
    assert len(result.selected_tokens) < state.scores.shape[0]


def test_dense_selector_has_perfect_distribution_and_output_metrics() -> None:
    state = _state()
    dense = DenseSelector().select(state, target_mass=1.0)
    oracle = TopMassOracleSelector().select(state, target_mass=1.0)
    metrics = compute_metrics(state, dense, oracle.selected_tokens)

    assert abs(metrics["attention_mass"] - 1.0) < 1e-6
    assert metrics["false_negative_mass"] == 0.0
    assert metrics["false_positive_mass"] == 0.0
    assert metrics["distribution_js"] < 1e-10
    assert metrics["output_relative_l2"] < 1e-6
    assert metrics["output_cosine"] > 0.999


def test_static_tokens_deduplicates_overlap() -> None:
    assert static_tokens(position=4, static_prefix=3, static_suffix=3) == [0, 1, 2, 3, 4]


def test_pqcache_full_scan_uses_structured_costs_and_reaches_target() -> None:
    state = _state()
    result = PQCacheFullScanSelector(subvecs=2, subbits=1, kmeans_iters=1).select(state, target_mass=0.80)
    selected_mass = float(state.probs[np.asarray(result.selected_tokens, dtype=np.int64)].sum())

    assert selected_mass >= 0.80
    assert len(result.candidate_tokens) == state.scores.shape[0]
    assert result.cost.mb(phase="selector") > 0.0
    assert result.cost.mb(phase="online_update") > 0.0
    assert result.cost.mb(phase="exact_attention") == 0.0


def test_retroinfer_style_scores_centroids_and_reaches_target() -> None:
    state = _state()
    result = RetroInferStyleSelector(cluster_size=2, static_prefix=0, static_suffix=0).select(state, target_mass=0.80)
    selected_mass = float(state.probs[np.asarray(result.selected_tokens, dtype=np.int64)].sum())

    assert result.algorithm == "retroinfer_style"
    assert selected_mass >= 0.80
    assert result.metadata["clusters_scored"] == 2
    assert result.cost.bytes(phase="selector", category="retro_centroids") > 0
    assert result.cost.mb(phase="online_update") == 0.0


def test_retrievalattention_graph_selector_records_graph_costs() -> None:
    keys = np.eye(6, 4, dtype=np.float32).reshape(1, 6, 4)
    values = keys.copy()
    queries = np.asarray([[[4.0, 1.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0]]], dtype=np.float32)
    trace = QKVTrace(keys=keys, values=values, queries=queries, positions=np.asarray([4, 5]), input_len=4, metadata={})
    scores, probs = attention_probs(keys[0], queries[0, 1])
    state = QueryState(
        decode_tokens=2,
        position=5,
        qidx=1,
        head=0,
        kv_head=0,
        query=queries[0, 1],
        keys=keys[0],
        values=values[0],
        scores=scores,
        probs=probs,
        base_tokens=[],
    )
    selector = RetrievalAttentionGraphSelector(
        trace=trace,
        static_prefix=0,
        static_suffix=0,
        provenance_topk=3,
        connect_window=2,
        degree=2,
        seed_count=2,
        min_visits=1,
        max_visits=4,
    )
    result = selector.select(state, target_mass=0.50)

    assert result.algorithm == "retrievalattention_graph"
    assert len(result.candidate_tokens) > 0
    assert result.cost.bytes(phase="selector", category="ra_score_keys") > 0
    assert result.cost.bytes(phase="selector", category="ra_graph_offsets") > 0


def test_paged_local_pq_adapter_uses_selector_interface() -> None:
    keys = np.eye(6, 4, dtype=np.float32).reshape(1, 6, 4)
    values = keys.copy()
    queries = np.asarray([[[4.0, 1.0, 0.0, 0.0]]], dtype=np.float32)
    trace = QKVTrace(keys=keys, values=values, queries=queries, positions=np.asarray([5]), input_len=4, metadata={})
    scores, probs = attention_probs(keys[0], queries[0, 0])
    state = QueryState(
        decode_tokens=2,
        position=5,
        qidx=0,
        head=0,
        kv_head=0,
        query=queries[0, 0],
        keys=keys[0],
        values=values[0],
        scores=scores,
        probs=probs,
        base_tokens=[],
    )
    selector = PagedPQSelector(
        trace=trace,
        static_prefix=0,
        static_suffix=0,
        page_size=2,
        routed=False,
        subvecs=2,
        subbits=1,
        kmeans_iters=1,
    )
    result = selector.select(state, target_mass=0.80)

    assert result.algorithm == "paged_local_pq"
    assert len(result.selected_tokens) > 0
    assert result.cost.mb(phase="selector") > 0.0


def test_gated_paged_pq_registry_alias() -> None:
    keys = np.eye(6, 4, dtype=np.float32).reshape(1, 6, 4)
    values = keys.copy()
    queries = np.asarray([[[4.0, 1.0, 0.0, 0.0]]], dtype=np.float32)
    trace = QKVTrace(keys=keys, values=values, queries=queries, positions=np.asarray([5]), input_len=4, metadata={})
    selector = selector_from_name(
        "gated_paged_pq",
        trace=trace,
        paged_kwargs={
            "static_prefix": 0,
            "static_suffix": 0,
            "page_size": 2,
            "subvecs": 2,
            "subbits": 1,
            "kmeans_iters": 1,
            "router_max_groups": 4,
        },
    )

    assert isinstance(selector, PagedPQSelector)
    assert selector.name == "gated_paged_pq"


def test_paged_local_pq_approx_alias_uses_approx_stop_policy() -> None:
    keys = np.eye(6, 4, dtype=np.float32).reshape(1, 6, 4)
    values = keys.copy()
    queries = np.asarray([[[4.0, 1.0, 0.0, 0.0]]], dtype=np.float32)
    trace = QKVTrace(keys=keys, values=values, queries=queries, positions=np.asarray([5]), input_len=4, metadata={})
    scores, probs = attention_probs(keys[0], queries[0, 0])
    state = QueryState(
        decode_tokens=2,
        position=5,
        qidx=0,
        head=0,
        kv_head=0,
        query=queries[0, 0],
        keys=keys[0],
        values=values[0],
        scores=scores,
        probs=probs,
        base_tokens=[],
    )
    selector = selector_from_name(
        "paged_local_pq_approx_mbp100",
        trace=trace,
        paged_kwargs={
            "static_prefix": 0,
            "static_suffix": 0,
            "page_size": 2,
            "subvecs": 2,
            "subbits": 1,
            "kmeans_iters": 1,
        },
    )
    result = selector.select(state, target_mass=0.50)

    assert isinstance(selector, PagedPQSelector)
    assert selector.stop_policy == "approx_mass"
    assert selector.approx_mass_margin == 0.01
    assert result.algorithm == "paged_local_pq_approx_mbp100"
    assert result.metadata["stop_policy"] == "approx_mass"
    assert result.metadata["selector_approx_mass"] > 0.0


def test_gated_paged_pq_sparq_rerank_registry_alias() -> None:
    keys = np.eye(6, 4, dtype=np.float32).reshape(1, 6, 4)
    values = keys.copy()
    queries = np.asarray([[[4.0, 1.0, 0.0, 0.0]]], dtype=np.float32)
    trace = QKVTrace(keys=keys, values=values, queries=queries, positions=np.asarray([5]), input_len=4, metadata={})
    scores, probs = attention_probs(keys[0], queries[0, 0])
    state = QueryState(
        decode_tokens=2,
        position=5,
        qidx=0,
        head=0,
        kv_head=0,
        query=queries[0, 0],
        keys=keys[0],
        values=values[0],
        scores=scores,
        probs=probs,
        base_tokens=[],
    )
    selector = selector_from_name(
        "gated_paged_pq_sparq_rerank",
        trace=trace,
        paged_kwargs={
            "static_prefix": 0,
            "static_suffix": 0,
            "page_size": 2,
            "subvecs": 2,
            "subbits": 1,
            "kmeans_iters": 1,
            "router_max_groups": 4,
        },
    )
    result = selector.select(state, target_mass=0.50)

    assert isinstance(selector, PagedPQSparQRerankSelector)
    assert result.algorithm == "gated_paged_pq_sparq_rerank"
    assert result.cost.bytes(phase="selector", category="hybrid_sparq_key_channels") > 0
    assert result.metadata["sparq_rank"] > 0


def test_page_sparq_pq_registry_alias() -> None:
    keys = np.eye(6, 4, dtype=np.float32).reshape(1, 6, 4)
    values = keys.copy()
    queries = np.asarray([[[4.0, 1.0, 0.0, 0.0]]], dtype=np.float32)
    trace = QKVTrace(keys=keys, values=values, queries=queries, positions=np.asarray([5]), input_len=4, metadata={})
    scores, probs = attention_probs(keys[0], queries[0, 0])
    state = QueryState(
        decode_tokens=2,
        position=5,
        qidx=0,
        head=0,
        kv_head=0,
        query=queries[0, 0],
        keys=keys[0],
        values=values[0],
        scores=scores,
        probs=probs,
        base_tokens=[],
    )
    selector = selector_from_name(
        "page_sparq_pq",
        trace=trace,
        paged_kwargs={
            "static_prefix": 0,
            "static_suffix": 0,
            "page_size": 2,
            "subvecs": 2,
            "subbits": 1,
            "kmeans_iters": 1,
            "nprobes": (1, 2),
        },
    )
    result = selector.select(state, target_mass=0.50)

    assert isinstance(selector, PageSparQPQSelector)
    assert result.algorithm == "page_sparq_pq"
    assert result.cost.bytes(phase="selector", category="page_sparq_minmax") > 0
    assert result.metadata["online_update_cumulative_MB"] > 0.0


def test_page_sparq_postings_pq_registry_alias() -> None:
    keys = np.eye(6, 4, dtype=np.float32).reshape(1, 6, 4)
    values = keys.copy()
    queries = np.asarray([[[4.0, 1.0, 0.0, 0.0]]], dtype=np.float32)
    trace = QKVTrace(keys=keys, values=values, queries=queries, positions=np.asarray([5]), input_len=4, metadata={})
    scores, probs = attention_probs(keys[0], queries[0, 0])
    state = QueryState(
        decode_tokens=2,
        position=5,
        qidx=0,
        head=0,
        kv_head=0,
        query=queries[0, 0],
        keys=keys[0],
        values=values[0],
        scores=scores,
        probs=probs,
        base_tokens=[],
    )
    selector = selector_from_name(
        "page_sparq_postings_pq",
        trace=trace,
        paged_kwargs={
            "static_prefix": 0,
            "static_suffix": 0,
            "page_size": 2,
            "subvecs": 2,
            "subbits": 1,
            "kmeans_iters": 1,
            "nprobes": (1, 2),
        },
    )
    result = selector.select(state, target_mass=0.50)

    assert isinstance(selector, PageSparQPostingsPQSelector)
    assert result.algorithm == "page_sparq_postings_pq"
    assert result.cost.bytes(phase="selector", category="page_sparq_postings") > 0
    assert result.metadata["postings_per_dim"] > 0

    k_selector = selector_from_name(
        "page_sparq_postings_pq_k128",
        trace=trace,
        paged_kwargs={
            "static_prefix": 0,
            "static_suffix": 0,
            "page_size": 2,
            "subvecs": 2,
            "subbits": 1,
            "kmeans_iters": 1,
            "nprobes": (1,),
        },
    )
    assert isinstance(k_selector, PageSparQPostingsPQSelector)
    assert k_selector.postings_per_dim == 128


def test_paged_pq_snapshot_alias_suppresses_update_cost() -> None:
    keys = np.eye(6, 4, dtype=np.float32).reshape(1, 6, 4)
    values = keys.copy()
    queries = np.asarray([[[4.0, 1.0, 0.0, 0.0]]], dtype=np.float32)
    trace = QKVTrace(keys=keys, values=values, queries=queries, positions=np.asarray([5]), input_len=4, metadata={})
    scores, probs = attention_probs(keys[0], queries[0, 0])
    state = QueryState(
        decode_tokens=2,
        position=5,
        qidx=0,
        head=0,
        kv_head=0,
        query=queries[0, 0],
        keys=keys[0],
        values=values[0],
        scores=scores,
        probs=probs,
        base_tokens=[],
    )
    selector = selector_from_name(
        "gated_paged_pq_snapshot",
        trace=trace,
        paged_kwargs={
            "static_prefix": 0,
            "static_suffix": 0,
            "page_size": 2,
            "subvecs": 2,
            "subbits": 1,
            "kmeans_iters": 1,
            "router_max_groups": 4,
        },
    )
    result = selector.select(state, target_mass=0.50)

    assert result.algorithm == "gated_paged_pq_snapshot"
    assert result.metadata["accounting_mode"] == "snapshot"
    assert result.cost.mb(phase="online_update") == 0.0


def test_pqcache_snapshot_and_online_aliases_differ_in_update_cost() -> None:
    state = _state()
    snapshot = selector_from_name("pqcache_full_scan_snapshot").select(state, target_mass=0.80)
    online = selector_from_name("pqcache_full_scan_online").select(state, target_mass=0.80)

    assert snapshot.algorithm == "pqcache_full_scan_snapshot"
    assert snapshot.metadata["accounting_mode"] == "snapshot"
    assert snapshot.cost.mb(phase="online_update") == 0.0
    assert online.algorithm == "pqcache_full_scan_online_proxy"
    assert online.metadata["accounting_mode"] == "online_proxy"
    assert online.cost.mb(phase="online_update") > 0.0


def test_retroinfer_online_proxy_charges_segment_update() -> None:
    state = _state()
    selector = selector_from_name(
        "retroinfer_online_proxy",
        retroinfer_kwargs={
            "cluster_size": 2,
            "static_prefix": 0,
            "static_suffix": 0,
            "input_len": 2,
        },
    )
    result = selector.select(state, target_mass=0.80)

    assert result.algorithm == "retroinfer_online_proxy"
    assert result.metadata["accounting_mode"] == "online_proxy"
    assert result.cost.bytes(phase="online_update", category="retro_segment_keys") > 0


def test_ivfpq_adapter_uses_selector_interface() -> None:
    keys = np.eye(6, 4, dtype=np.float32).reshape(1, 6, 4)
    values = keys.copy()
    queries = np.asarray([[[4.0, 1.0, 0.0, 0.0]]], dtype=np.float32)
    trace = QKVTrace(keys=keys, values=values, queries=queries, positions=np.asarray([5]), input_len=4, metadata={})
    scores, probs = attention_probs(keys[0], queries[0, 0])
    state = QueryState(
        decode_tokens=2,
        position=5,
        qidx=0,
        head=0,
        kv_head=0,
        query=queries[0, 0],
        keys=keys[0],
        values=values[0],
        scores=scores,
        probs=probs,
        base_tokens=[],
    )
    selector = IVFPQSelector(
        trace=trace,
        policy="frozen_append",
        static_prefix=0,
        static_suffix=0,
        nprobes=(1,),
        coarse_clusters=2,
        subvecs=2,
        subbits=1,
        kmeans_iters=1,
        backend="python",
    )
    result = selector.select(state, target_mass=0.80)

    assert result.algorithm == "ivfpq_frozen_append"
    assert len(result.selected_tokens) > 0
    assert result.cost.mb(phase="selector") > 0.0


def test_sparq_selector_uses_structured_selector_cost() -> None:
    state = _state()
    result = SparQSelector(rank=2).select(state, target_mass=0.80)
    selected_mass = float(state.probs[np.asarray(result.selected_tokens, dtype=np.int64)].sum())

    assert result.algorithm == "sparq_r2"
    assert selected_mass >= 0.80
    assert len(result.candidate_tokens) == state.scores.shape[0]
    assert result.cost.mb(phase="selector") > 0.0


def test_magicpig_selector_uses_hash_sidecar_cost() -> None:
    state = _state()
    result = MagicPIGSelector(bits=2, tables=4, min_collisions=1).select(state, target_mass=0.50)

    assert result.algorithm == "magicpig_k2_l4"
    assert result.cost.bytes(phase="selector", category="hash_codes") > 0
    assert result.cost.bytes(phase="selector", category="hash_key_scan") == 0
    assert result.cost.mb(phase="selector") > 0.0


def test_magicpig_registry_name_with_params() -> None:
    selector = selector_from_name("magicpig_k2_l4", magicpig_kwargs={"min_collisions": 1})

    assert isinstance(selector, MagicPIGSelector)
    assert selector.bits == 2
    assert selector.tables == 4
    assert selector.min_collisions == 1
