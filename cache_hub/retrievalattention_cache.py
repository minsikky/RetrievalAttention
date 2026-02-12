import math
import os
import time
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import torch.nn.functional as F
from .cache import KV_Cache

try:
    import faiss
except Exception:
    faiss = None

try:
    from .fused_qk_topk_kernel import fused_qk_topk_triton
except Exception:
    fused_qk_topk_triton = None


class retrievalattention_cache(KV_Cache):
    """
    Token-level ANN retrieval cache with a projected K–K graph and
    static GPU-resident KV (prefix + suffix window).
    """

    def __init__(
        self,
        valid_start,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        max_new_length: int,
        static_pattern_start: int,
        static_pattern_end: int,
        q_knn: int,
        key_degree: int,
        token_budget: int,
        num_gpus: int,
        model_size: int,
    ) -> None:
        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, num_gpus, model_size)
        self.valid_start = valid_start

        self.static_pattern_start = static_pattern_start
        self.static_pattern_end = static_pattern_end
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

        self.group_size = self.num_heads // self.kv_head
        self.batch_groups = self.batch_size * self.kv_head

        self.max_new_length = max_new_length
        self.input_length = self.max_length - max_new_length
        self.context = 0

        self.q_knn = max(1, int(q_knn))
        self.key_degree = max(1, int(key_degree))
        self.token_budget = max(1, int(token_budget))
        self.debug = os.environ.get("RETRIEVALATTN_DEBUG", "0") == "1"
        self.assert_nonempty = os.environ.get("RETRIEVALATTN_ASSERT_NONEMPTY", "0") == "1"
        self.validate_parity = os.environ.get("RETRIEVALATTN_VALIDATE_PARITY", "0") == "1"
        self.decode_profile = os.environ.get("RETRIEVALATTN_DECODE_PROFILE", "0") == "1"
        self.debug_decode_steps = int(os.environ.get("RETRIEVALATTN_DEBUG_STEPS", "3"))
        self.decode_index_mode = os.environ.get("RETRIEVALATTN_DECODE_INDEX", "faiss")
        self.seed_mode = os.environ.get("RETRIEVALATTN_SEED_MODE", "graph_only").strip().lower()
        if self.seed_mode not in {"graph_only", "faiss"}:
            print(
                f"[RetrievalAttention] WARNING: unknown RETRIEVALATTN_SEED_MODE={self.seed_mode}. "
                "Falling back to graph_only."
            )
            self.seed_mode = "graph_only"
        self.query_mode = os.environ.get("RETRIEVALATTN_QUERY_MODE", "per_head").strip().lower()
        if self.query_mode not in {"per_head", "group_avg"}:
            print(
                f"[RetrievalAttention] WARNING: unknown RETRIEVALATTN_QUERY_MODE={self.query_mode}. "
                "Falling back to per_head."
            )
            self.query_mode = "per_head"
        raw_score_mode = os.environ.get("RETRIEVALATTN_SCORE_MODE", "ip").strip().lower()
        if raw_score_mode in {"cos", "cosine"}:
            self.score_mode = "cosine"
        elif raw_score_mode in {"ip", "inner", "inner_product", "dot"}:
            self.score_mode = "ip"
        else:
            print(
                f"[RetrievalAttention] WARNING: unknown RETRIEVALATTN_SCORE_MODE={raw_score_mode}. "
                "Falling back to ip."
            )
            self.score_mode = "ip"
        self.score_normalize = (self.score_mode == "cosine")
        self.graph_expand = os.environ.get("RETRIEVALATTN_GRAPH_EXPAND", "1") == "1"
        self.graph_weighted = os.environ.get("RETRIEVALATTN_GRAPH_WEIGHTED", "1") == "1"
        self.graph_builder = os.environ.get("RETRIEVALATTN_GRAPH_BUILDER", "legacy").strip().lower()
        if self.graph_builder not in {"legacy", "roar"}:
            print(
                f"[RetrievalAttention] WARNING: unknown RETRIEVALATTN_GRAPH_BUILDER={self.graph_builder}. "
                "Falling back to legacy."
            )
            self.graph_builder = "legacy"
        try:
            self.graph_clique_m = int(os.environ.get("RETRIEVALATTN_GRAPH_CLIQUE_M", "6"))
        except Exception:
            self.graph_clique_m = 6
        self.graph_clique_m = max(0, self.graph_clique_m)
        self.graph_return_weights = os.environ.get("RETRIEVALATTN_GRAPH_RETURN_WEIGHTS", "0") == "1"
        raw_graph_weight_dtype = os.environ.get("RETRIEVALATTN_GRAPH_WEIGHT_DTYPE", "uint16").strip().lower()
        if raw_graph_weight_dtype in {"uint16", "u16"}:
            self.graph_weight_dtype = np.uint16
        elif raw_graph_weight_dtype in {"uint32", "u32"}:
            self.graph_weight_dtype = np.uint32
        else:
            print(
                f"[RetrievalAttention] WARNING: unknown RETRIEVALATTN_GRAPH_WEIGHT_DTYPE={raw_graph_weight_dtype}. "
                "Falling back to uint16."
            )
            self.graph_weight_dtype = np.uint16
        self.graph_weight_max = int(np.iinfo(self.graph_weight_dtype).max)
        self.graph_weight_dtype_name = np.dtype(self.graph_weight_dtype).name
        try:
            self.roar_nq = int(os.environ.get("RETRIEVALATTN_ROAR_NQ", str(self.q_knn)))
        except Exception:
            self.roar_nq = self.q_knn
        self.roar_nq = max(1, self.roar_nq)
        try:
            self.roar_l = int(os.environ.get("RETRIEVALATTN_ROAR_L", "256"))
        except Exception:
            self.roar_l = 256
        self.roar_l = max(1, self.roar_l)
        try:
            self.roar_m = int(os.environ.get("RETRIEVALATTN_ROAR_M", str(self.key_degree)))
        except Exception:
            self.roar_m = self.key_degree
        self.roar_m = max(1, self.roar_m)
        self.roar_enable_enhance = os.environ.get("RETRIEVALATTN_ROAR_ENABLE_ENHANCE", "1") == "1"
        try:
            self.roar_enhance_l = int(os.environ.get("RETRIEVALATTN_ROAR_ENHANCE_L", str(self.roar_l)))
        except Exception:
            self.roar_enhance_l = self.roar_l
        self.roar_enhance_l = max(1, self.roar_enhance_l)
        self.roar_entry = os.environ.get("RETRIEVALATTN_ROAR_ENTRY", "hub").strip().lower()
        if self.roar_entry not in {"hub", "max_degree", "self"}:
            print(
                f"[RetrievalAttention] WARNING: unknown RETRIEVALATTN_ROAR_ENTRY={self.roar_entry}. "
                "Falling back to hub."
            )
            self.roar_entry = "hub"
        try:
            self.roar_max_query_per_pivot = int(os.environ.get("RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT", "0"))
        except Exception:
            self.roar_max_query_per_pivot = 0
        self.roar_max_query_per_pivot = max(0, self.roar_max_query_per_pivot)
        self.roar_log = os.environ.get("RETRIEVALATTN_ROAR_LOG", "1") == "1"
        legacy_graph_hops = os.environ.get("RETRIEVALATTN_GRAPH_HOPS")
        if legacy_graph_hops not in (None, ""):
            print(
                "[RetrievalAttention] WARNING: RETRIEVALATTN_GRAPH_HOPS is deprecated and ignored. "
                "Adaptive best-first graph traversal is used instead."
            )
        try:
            self.expand_width = int(os.environ.get("RETRIEVALATTN_EXPAND_WIDTH", "64"))
        except Exception:
            self.expand_width = 64
        self.expand_width = max(1, self.expand_width)
        try:
            self.min_visits = int(os.environ.get("RETRIEVALATTN_MIN_VISITS", str(self.token_budget)))
        except Exception:
            self.min_visits = self.token_budget
        if self.min_visits <= 0:
            self.min_visits = self.token_budget
        self.min_visits = max(1, self.min_visits)
        try:
            self.max_visits = int(os.environ.get("RETRIEVALATTN_MAX_VISITS", str(self.token_budget * 8)))
        except Exception:
            self.max_visits = self.token_budget * 8
        if self.max_visits <= 0:
            self.max_visits = self.token_budget * 8
        self.max_visits = max(self.min_visits, self.max_visits)
        try:
            self.stop_patience = int(os.environ.get("RETRIEVALATTN_STOP_PATIENCE", "2"))
        except Exception:
            self.stop_patience = 2
        self.stop_patience = max(0, self.stop_patience)
        try:
            self.stop_margin = float(os.environ.get("RETRIEVALATTN_STOP_MARGIN", "0.0"))
        except Exception:
            self.stop_margin = 0.0
        try:
            self.frontier_topn = int(os.environ.get("RETRIEVALATTN_FRONTIER_TOPN", "0"))
        except Exception:
            self.frontier_topn = 0
        self.frontier_topn = max(0, self.frontier_topn)
        self.rerank = os.environ.get("RETRIEVALATTN_RERANK", "1") == "1"
        self.rerank_agg = os.environ.get("RETRIEVALATTN_RERANK_AGG", "max").strip().lower()
        if self.rerank_agg not in {"max", "mean"}:
            print(
                f"[RetrievalAttention] WARNING: unknown RETRIEVALATTN_RERANK_AGG={self.rerank_agg}. "
                "Falling back to max."
            )
            self.rerank_agg = "max"
        try:
            self.seed_ratio = float(os.environ.get("RETRIEVALATTN_SEED_RATIO", "0.7"))
        except Exception:
            self.seed_ratio = 0.7
        self.seed_ratio = max(0.0, min(1.0, self.seed_ratio))
        try:
            self.candidate_multiplier = int(os.environ.get("RETRIEVALATTN_CAND_MULT", "4"))
        except Exception:
            self.candidate_multiplier = 4
        self.candidate_multiplier = max(1, self.candidate_multiplier)
        try:
            self.seed_k_mult = int(os.environ.get("RETRIEVALATTN_SEED_K_MULT", "1"))
        except Exception:
            self.seed_k_mult = 1
        self.seed_k_mult = max(1, self.seed_k_mult)
        try:
            self.seed_prev_k = int(os.environ.get("RETRIEVALATTN_SEED_PREV_K", str(self.token_budget)))
        except Exception:
            self.seed_prev_k = self.token_budget
        self.seed_prev_k = max(1, self.seed_prev_k)
        try:
            self.seed_hub_k = int(os.environ.get("RETRIEVALATTN_SEED_HUB_K", "64"))
        except Exception:
            self.seed_hub_k = 64
        self.seed_hub_k = max(0, self.seed_hub_k)
        try:
            self.seed_tail_k = int(os.environ.get("RETRIEVALATTN_SEED_TAIL_K", "32"))
        except Exception:
            self.seed_tail_k = 32
        self.seed_tail_k = max(0, self.seed_tail_k)
        self.fa_fused_prefill = os.environ.get("RETRIEVALATTN_FA_FUSED_PREFILL", "0") == "1"
        self.fa_shadow_compare = os.environ.get("RETRIEVALATTN_FA_SHADOW_COMPARE", "0") == "1"
        try:
            self.fa_shadow_sample = int(os.environ.get("RETRIEVALATTN_FA_SHADOW_SAMPLE", "256"))
        except Exception:
            self.fa_shadow_sample = 256
        self.fa_shadow_sample = max(1, self.fa_shadow_sample)
        self.fused_prefill_overlap = os.environ.get("RETRIEVALATTN_FUSED_PREFILL_OVERLAP", "1") == "1"
        try:
            self.fused_prefill_overlap_workers = int(
                os.environ.get("RETRIEVALATTN_FUSED_PREFILL_OVERLAP_WORKERS", "1")
            )
        except Exception:
            self.fused_prefill_overlap_workers = 1
        self.fused_prefill_overlap_workers = max(1, self.fused_prefill_overlap_workers)
        if self.fused_prefill_overlap_workers > 1:
            print(
                "[RetrievalAttention] WARNING: RETRIEVALATTN_FUSED_PREFILL_OVERLAP_WORKERS>1 may "
                "oversubscribe CPU with faiss OpenMP threads.",
                flush=True,
            )
        self._fused_async_enabled = (
            self.fa_fused_prefill
            and self.fused_prefill_overlap
            and (not self.fa_shadow_compare)
        )
        if self.fa_fused_prefill and self.fused_prefill_overlap and self.fa_shadow_compare:
            print(
                "[RetrievalAttention] WARNING: fused prefill overlap is disabled because "
                "RETRIEVALATTN_FA_SHADOW_COMPARE=1 requires synchronous shadow check.",
                flush=True,
            )
        self._store_prefill_queries = (
            (not self.fa_fused_prefill)
            or self.fa_shadow_compare
            or self.validate_parity
        )
        self._fallback_seed_warned = False

        self._decode_profile_stats = {
            "calls": 0,
            "heads": 0,
            "compute_total_sec": 0.0,
            "retrieve_total_sec": 0.0,
            "retrieve_seed_sec": 0.0,
            "retrieve_graph_sec": 0.0,
            "retrieve_rerank_sec": 0.0,
            "retrieve_finalize_sec": 0.0,
            "gather_total_sec": 0.0,
            "attn_total_sec": 0.0,
            "visited_total": 0,
            "candidates_total": 0,
        }

        # CPU storage for full K/V (including future decode slots)
        total_len = self.input_length + self.max_new_length
        self.cpu_keys = [
            torch.empty((self.kv_head, total_len, self.head_dim), dtype=self.dtype, pin_memory=True)
            for _ in range(self.layer_num)
        ]
        self.cpu_values = [
            torch.empty((self.kv_head, total_len, self.head_dim), dtype=self.dtype, pin_memory=True)
            for _ in range(self.layer_num)
        ]

        # CPU storage for queries (prefill only; freed after graph build).
        # Fused-prefill mode can skip this unless shadow-compare/parity is requested.
        if self._store_prefill_queries:
            self.cpu_queries = [
                torch.empty((self.kv_head, self.input_length, self.head_dim), dtype=self.dtype, pin_memory=True)
                for _ in range(self.layer_num)
            ]
        else:
            self.cpu_queries = None

        # Static GPU KV (prefix + suffix window)
        self.static_gpu_keys = [
            torch.empty((self.kv_head, self.static_pattern_total, self.head_dim),
                        dtype=self.dtype, device=self.layer_mapping[str(ldx)])
            for ldx in range(self.layer_num)
        ]
        self.static_gpu_values = [
            torch.empty((self.kv_head, self.static_pattern_total, self.head_dim),
                        dtype=self.dtype, device=self.layer_mapping[str(ldx)])
            for ldx in range(self.layer_num)
        ]

        # ANN index and K–K graph per layer/head
        self.indexes = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self.graphs = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self.hub_seeds = [[[] for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self.prev_decode_seeds = [[[] for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self.fused_prefill_knn = [None for _ in range(self.layer_num)]
        self.fused_prefill_profiles = [None for _ in range(self.layer_num)]
        self._fused_prefill_executor = None
        self._fused_prefill_futures = {}
        self._fused_prefill_submitted = [False for _ in range(self.layer_num)]
        self._fused_prefill_done = [False for _ in range(self.layer_num)]
        self._fused_prefill_errors = {}
        self._fused_prefill_submit_count = 0
        self._fused_prefill_done_count = 0
        self._fused_prefill_lock = threading.Lock()
        self._faiss_threads_async_configured = False

        # Track suffix window positions
        self.suffix_start = max(0, self.input_length - self.static_pattern_end)
        self.decode_pos = 0

        prefix_static = min(max(0, self.static_pattern_start), self.input_length)
        suffix_static = min(max(0, self.static_pattern_end), self.input_length)
        self.dynamic_start = prefix_static
        self.dynamic_end = max(self.dynamic_start, self.input_length - suffix_static)
        self.static_index_set = set(range(prefix_static))
        self.static_index_set.update(range(self.dynamic_end, self.input_length))

        self._built = False

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
        return x / denom

    def _score_transform_np(self, x: np.ndarray) -> np.ndarray:
        if self.score_normalize:
            return self._normalize(x)
        return x

    def _score_transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.score_normalize:
            return F.normalize(x, dim=-1)
        return x

    def _knn_recall_at_k(self, a: np.ndarray, b: np.ndarray, k: int) -> float:
        """
        Mean recall@k between two knn index arrays of shape [N, k].
        """
        if a.size == 0 or b.size == 0:
            return 0.0
        n = min(a.shape[0], b.shape[0])
        k = min(k, a.shape[1], b.shape[1])
        if n == 0 or k == 0:
            return 0.0
        recalls = []
        for i in range(n):
            sa = set(int(x) for x in a[i, :k].tolist())
            sb = set(int(x) for x in b[i, :k].tolist())
            recalls.append(len(sa.intersection(sb)) / float(k))
        return float(np.mean(recalls))

    def _empty_graph_csr(self):
        offsets = np.zeros(self.input_length + 1, dtype=np.uint32)
        neighbors = np.empty((0,), dtype=np.int32)
        if self.graph_return_weights:
            weights = np.empty((0,), dtype=self.graph_weight_dtype)
            return offsets, neighbors, weights
        return offsets, neighbors

    def _build_graph_csr_from_knn(self, knn: np.ndarray, keys_cpu: np.ndarray = None):
        """
        Build projected graph from Q->K knn rows.
        Returns:
          graph: CSR graph tuple
          meta: dict with builder name and stage timings/stats
        """
        builder = self.graph_builder
        if builder == "roar":
            return self._build_graph_csr_from_knn_roar(knn, keys_cpu)
        return self._build_graph_csr_from_knn_legacy(knn)

    def _build_graph_csr_from_knn_legacy(self, knn: np.ndarray):
        """
        Build a symmetric projected graph from knn with CSR storage:
          offsets: [num_tokens + 1] uint32
          neighbors: [num_edges] int32
          weights: [num_edges] uint16/uint32 (optional)
        """
        meta = {
            "builder": "legacy",
            "bipartite_sec": 0.0,
            "projection_sec": 0.0,
            "enhance_sec": 0.0,
            "csr_sec": 0.0,
            "total_sec": 0.0,
            "active_queries": 0,
            "active_pivots": 0,
            "projected_nodes": 0,
            "enhanced_nodes": 0,
            "stop_reason": "ok",
        }
        total_start = time.time()
        num_tokens = self.input_length
        empty_graph = self._empty_graph_csr()

        if knn.size == 0 or knn.shape[1] <= 1:
            meta["stop_reason"] = "empty_knn"
            meta["total_sec"] = time.time() - total_start
            return empty_graph, meta

        anchors = knn[:, 0].astype(np.int32, copy=False)
        candidates = knn[:, 1:].astype(np.int32, copy=False)
        if candidates.size == 0:
            meta["stop_reason"] = "empty_candidates"
            meta["total_sec"] = time.time() - total_start
            return empty_graph, meta

        proj_start = time.time()
        if not self.graph_weighted:
            anchor_rep = np.repeat(anchors, candidates.shape[1])
            cand_flat = candidates.reshape(-1)

            # Symmetric edges: anchor->candidate and candidate->anchor.
            src = np.concatenate([anchor_rep, cand_flat], axis=0)
            dst = np.concatenate([cand_flat, anchor_rep], axis=0)

            valid = (
                (src >= 0) & (src < num_tokens)
                & (dst >= 0) & (dst < num_tokens)
                & (src != dst)
            )
            if not np.any(valid):
                meta["stop_reason"] = "no_valid_edges"
                meta["total_sec"] = time.time() - total_start
                return empty_graph, meta
            src = src[valid]
            dst = dst[valid]

            edge_keys = (src.astype(np.int64) << 32) | dst.astype(np.uint32).astype(np.int64)
            edge_keys = np.unique(edge_keys)
            if edge_keys.size == 0:
                meta["stop_reason"] = "empty_unique_edges"
                meta["total_sec"] = time.time() - total_start
                return empty_graph, meta

            rows = (edge_keys >> 32).astype(np.int32, copy=False)
            cols = (edge_keys & np.int64(0xFFFFFFFF)).astype(np.int32, copy=False)
            weights = np.ones((rows.shape[0],), dtype=np.int64)
        else:
            src_chunks = []
            dst_chunks = []

            # Weighted star projection from anchor->candidates.
            anchor_rep = np.repeat(anchors, candidates.shape[1])
            cand_flat = candidates.reshape(-1)
            src_chunks.append(anchor_rep)
            dst_chunks.append(cand_flat)
            src_chunks.append(cand_flat)
            dst_chunks.append(anchor_rep)

            # Clique-lite: add pair edges among top-M candidates in each query row.
            clique_m = min(self.graph_clique_m, int(candidates.shape[1]))
            if clique_m >= 2:
                local = candidates[:, :clique_m]
                for i in range(clique_m - 1):
                    lhs = local[:, i]
                    for j in range(i + 1, clique_m):
                        rhs = local[:, j]
                        src_chunks.append(lhs)
                        dst_chunks.append(rhs)
                        src_chunks.append(rhs)
                        dst_chunks.append(lhs)

            src = np.concatenate(src_chunks, axis=0)
            dst = np.concatenate(dst_chunks, axis=0)
            valid = (
                (src >= 0) & (src < num_tokens)
                & (dst >= 0) & (dst < num_tokens)
                & (src != dst)
            )
            if not np.any(valid):
                meta["stop_reason"] = "no_valid_edges"
                meta["total_sec"] = time.time() - total_start
                return empty_graph, meta
            src = src[valid]
            dst = dst[valid]

            edge_keys = (src.astype(np.int64) << 32) | dst.astype(np.uint32).astype(np.int64)
            edge_keys, edge_counts = np.unique(edge_keys, return_counts=True)
            if edge_keys.size == 0:
                meta["stop_reason"] = "empty_unique_edges"
                meta["total_sec"] = time.time() - total_start
                return empty_graph, meta

            rows = (edge_keys >> 32).astype(np.int32, copy=False)
            cols = (edge_keys & np.int64(0xFFFFFFFF)).astype(np.int32, copy=False)
            weights = edge_counts.astype(np.int64, copy=False)

        if rows.size == 0:
            meta["stop_reason"] = "empty_rows"
            meta["total_sec"] = time.time() - total_start
            return empty_graph, meta

        counts_full = np.bincount(rows, minlength=num_tokens).astype(np.int64, copy=False)
        row_starts = np.empty(num_tokens + 1, dtype=np.int64)
        row_starts[0] = 0
        np.cumsum(counts_full, out=row_starts[1:])
        row_ids = np.nonzero(counts_full)[0]

        kept_cols = []
        kept_weights = []
        row_counts = np.zeros(num_tokens, dtype=np.uint64)
        degree_cap = max(0, int(self.key_degree))

        for row in row_ids:
            start = int(row_starts[row])
            end = int(row_starts[row + 1])
            if end <= start:
                continue
            seg_cols = cols[start:end]
            seg_weights = weights[start:end]

            # Highest weight first; deterministic tie-break by neighbor id.
            order = np.lexsort((seg_cols, -seg_weights))
            if degree_cap > 0 and order.shape[0] > degree_cap:
                order = order[:degree_cap]
            if order.size == 0:
                continue
            kept_cols.append(seg_cols[order].astype(np.int32, copy=False))
            kept_weights.append(seg_weights[order].astype(np.int64, copy=False))
            row_counts[row] = int(order.size)

        if not kept_cols:
            meta["stop_reason"] = "empty_after_degree_cap"
            meta["total_sec"] = time.time() - total_start
            return empty_graph, meta

        neighbors = np.concatenate(kept_cols, axis=0).astype(np.int32, copy=False)
        weights_sel = np.concatenate(kept_weights, axis=0).astype(np.int64, copy=False)

        offsets64 = np.empty(num_tokens + 1, dtype=np.uint64)
        offsets64[0] = 0
        np.cumsum(row_counts, out=offsets64[1:])
        if int(offsets64[-1]) > np.iinfo(np.uint32).max:
            raise RuntimeError(
                f"[RetrievalAttention] CSR offsets exceed uint32 range: edges={int(offsets64[-1])}"
            )
        offsets = offsets64.astype(np.uint32, copy=False)
        meta["projection_sec"] = time.time() - proj_start
        meta["projected_nodes"] = int(np.sum(row_counts > 0))
        meta["active_queries"] = int(knn.shape[0])
        meta["active_pivots"] = int(meta["projected_nodes"])
        meta["total_sec"] = time.time() - total_start

        if self.graph_return_weights:
            weights_out = np.minimum(weights_sel, self.graph_weight_max).astype(self.graph_weight_dtype, copy=False)
            return (offsets, neighbors, weights_out), meta
        return (offsets, neighbors), meta

    def _dedup_dynamic_tokens(self, tokens, exclude: int = -1, max_take: int = 0):
        out = []
        seen = set()
        for tok in tokens:
            tok = int(tok)
            if tok == exclude:
                continue
            if tok < self.dynamic_start or tok >= self.dynamic_end:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if max_take > 0 and len(out) >= max_take:
                break
        return out

    def _acquire_neighbors_roar(self, x: int, candidates, keys_cpu: np.ndarray, degree_cap: int):
        """
        RoarGraph-style AcquireNeighbors:
        - sort candidates by closeness to x
        - diversification gate: sim(x, c) > sim(c, p) for all selected p
        - fill remainder by rank until degree cap
        """
        if degree_cap <= 0:
            return []
        cand = self._dedup_dynamic_tokens(candidates, exclude=int(x), max_take=0)
        if not cand:
            return []

        x_idx = int(x)
        x_vec = keys_cpu[x_idx]
        cand_arr = np.asarray(cand, dtype=np.int32)
        cand_vecs = keys_cpu[cand_arr]
        sim_xc = np.matmul(cand_vecs, x_vec)
        order = np.argsort(-sim_xc)
        sorted_cand = cand_arr[order]
        sorted_sim = sim_xc[order]

        selected = []
        selected_set = set()

        first = int(sorted_cand[0])
        selected.append(first)
        selected_set.add(first)

        for idx in range(1, sorted_cand.shape[0]):
            if len(selected) >= degree_cap:
                break
            c = int(sorted_cand[idx])
            sx = float(sorted_sim[idx])
            if c in selected_set:
                continue
            sel_arr = np.asarray(selected, dtype=np.int32)
            cp_sim = np.matmul(keys_cpu[sel_arr], keys_cpu[c])
            if np.all(sx > cp_sim):
                selected.append(c)
                selected_set.add(c)

        if len(selected) < degree_cap:
            for tok in sorted_cand.tolist():
                tok = int(tok)
                if tok in selected_set:
                    continue
                selected.append(tok)
                selected_set.add(tok)
                if len(selected) >= degree_cap:
                    break

        return selected[:degree_cap]

    def _beam_search_roar(self, source_node: int, adjacency: dict, keys_cpu: np.ndarray, entry_node: int, beam_l: int):
        """
        Small best-first beam over projected graph to gather enhancement candidates.
        """
        if beam_l <= 0:
            return []
        src = int(source_node)
        if src < self.dynamic_start or src >= self.dynamic_end:
            return []

        entry = int(entry_node)
        if entry < self.dynamic_start or entry >= self.dynamic_end:
            entry = src

        src_vec = keys_cpu[src]
        frontier = []
        best_score = {}
        visited = set()
        candidates = []

        def push(node: int):
            node = int(node)
            if node < self.dynamic_start or node >= self.dynamic_end:
                return
            score = float(np.dot(src_vec, keys_cpu[node]))
            prev = best_score.get(node)
            if prev is None or score > prev:
                best_score[node] = score
                heapq.heappush(frontier, (-score, node))

        push(entry)
        if entry != src and src in adjacency:
            push(src)

        while frontier and len(candidates) < beam_l:
            neg_score, node = heapq.heappop(frontier)
            score = -float(neg_score)
            prev = best_score.get(node)
            if prev is None or score < (prev - 1e-8):
                continue
            if node in visited:
                continue
            visited.add(node)
            if node != src:
                candidates.append(int(node))

            for nb in adjacency.get(node, []):
                nb = int(nb)
                if nb in visited:
                    continue
                push(nb)

        return candidates[:beam_l]

    def _adjacency_to_csr(self, adjacency: dict):
        num_tokens = self.input_length
        empty_graph = self._empty_graph_csr()
        if not adjacency:
            return empty_graph

        row_counts = np.zeros(num_tokens, dtype=np.uint64)
        rows = {}
        for row, nbrs in adjacency.items():
            row = int(row)
            if row < self.dynamic_start or row >= self.dynamic_end:
                continue
            clean = self._dedup_dynamic_tokens(nbrs, exclude=row, max_take=self.roar_m)
            if not clean:
                continue
            arr = np.asarray(clean, dtype=np.int32)
            rows[row] = arr
            row_counts[row] = int(arr.shape[0])

        if not rows:
            return empty_graph

        offsets64 = np.empty(num_tokens + 1, dtype=np.uint64)
        offsets64[0] = 0
        np.cumsum(row_counts, out=offsets64[1:])
        if int(offsets64[-1]) > np.iinfo(np.uint32).max:
            raise RuntimeError(
                f"[RetrievalAttention] CSR offsets exceed uint32 range: edges={int(offsets64[-1])}"
            )
        offsets = offsets64.astype(np.uint32, copy=False)
        total_edges = int(offsets[-1])
        neighbors = np.empty((total_edges,), dtype=np.int32)

        for row, arr in rows.items():
            start = int(offsets[row])
            end = start + int(arr.shape[0])
            neighbors[start:end] = arr

        if self.graph_return_weights:
            weights = np.ones((total_edges,), dtype=self.graph_weight_dtype)
            return offsets, neighbors, weights
        return offsets, neighbors

    def _build_graph_csr_from_knn_roar(self, knn: np.ndarray, keys_cpu: np.ndarray):
        """
        RoarGraph-like construction:
          1) query->base links + base->query bridge
          2) neighborhood-aware projection with reverse-edge updates
          3) connectivity enhancement (beam search + reverse-edge updates)
        """
        meta = {
            "builder": "roar",
            "bipartite_sec": 0.0,
            "projection_sec": 0.0,
            "enhance_sec": 0.0,
            "csr_sec": 0.0,
            "total_sec": 0.0,
            "active_queries": 0,
            "active_pivots": 0,
            "projected_nodes": 0,
            "enhanced_nodes": 0,
            "stop_reason": "ok",
        }
        total_start = time.time()
        empty_graph = self._empty_graph_csr()
        if knn.size == 0 or knn.shape[1] == 0:
            meta["stop_reason"] = "empty_knn"
            meta["total_sec"] = time.time() - total_start
            return empty_graph, meta
        if keys_cpu is None:
            raise RuntimeError(
                "Roar graph builder requires keys_cpu but got None."
            )

        num_tokens = self.input_length
        keys = np.asarray(keys_cpu, dtype=np.float32)
        if keys.shape[0] < num_tokens:
            raise RuntimeError(
                f"Roar graph builder keys_cpu too short: got {keys.shape[0]}, need {num_tokens}"
            )

        nq = min(max(1, int(self.roar_nq)), int(knn.shape[1]))
        degree_cap = max(1, int(self.roar_m))
        cand_limit = max(1, int(self.roar_l))
        enhance_limit = max(1, int(self.roar_enhance_l))

        # 1) Construct implicit bipartite graph.
        bip_start = time.time()
        query_count = int(knn.shape[0])
        out_width = max(0, nq - 1)
        pivot_of_query = np.full((query_count,), -1, dtype=np.int32)
        if out_width > 0:
            query_out = np.full((query_count, out_width), -1, dtype=np.int32)
        else:
            query_out = np.empty((query_count, 0), dtype=np.int32)

        for q in range(query_count):
            row = knn[q, :nq]
            filtered = self._dedup_dynamic_tokens(row, exclude=-1, max_take=nq)
            if not filtered:
                continue
            pivot = int(filtered[0])
            pivot_of_query[q] = pivot
            if out_width > 0 and len(filtered) > 1:
                rem = filtered[1 : 1 + out_width]
                query_out[q, : len(rem)] = np.asarray(rem, dtype=np.int32)

        active_q = np.nonzero(pivot_of_query >= 0)[0].astype(np.int32, copy=False)
        meta["active_queries"] = int(active_q.shape[0])
        if active_q.size == 0:
            meta["stop_reason"] = "no_active_queries"
            meta["bipartite_sec"] = time.time() - bip_start
            meta["total_sec"] = time.time() - total_start
            return empty_graph, meta

        pivots = pivot_of_query[active_q]
        order = np.argsort(pivots, kind="stable")
        sorted_pivots = pivots[order]
        sorted_qids = active_q[order]
        pivot_counts = np.bincount(sorted_pivots, minlength=num_tokens).astype(np.int64, copy=False)
        pivot_offsets = np.empty((num_tokens + 1,), dtype=np.int64)
        pivot_offsets[0] = 0
        np.cumsum(pivot_counts, out=pivot_offsets[1:])
        active_pivots = np.nonzero(pivot_counts > 0)[0].astype(np.int32, copy=False)
        meta["active_pivots"] = int(active_pivots.shape[0])
        meta["bipartite_sec"] = time.time() - bip_start

        # 2) Neighborhood-aware projection with reverse-edge updates.
        proj_start = time.time()
        projected = {}
        max_q_per_pivot = int(self.roar_max_query_per_pivot)
        for x in active_pivots.tolist():
            begin = int(pivot_offsets[x])
            end = int(pivot_offsets[x + 1])
            if end <= begin:
                continue
            if max_q_per_pivot > 0:
                end = min(end, begin + max_q_per_pivot)

            candidates = []
            seen = set()
            for pos in range(begin, end):
                qid = int(sorted_qids[pos])
                row = query_out[qid]
                for tok in row.tolist():
                    tok = int(tok)
                    if tok < 0:
                        break
                    if tok == x:
                        continue
                    if tok in seen:
                        continue
                    seen.add(tok)
                    candidates.append(tok)
                    if len(candidates) >= cand_limit:
                        break
                if len(candidates) >= cand_limit:
                    break

            if not candidates:
                continue
            x_neighbors = self._acquire_neighbors_roar(int(x), candidates, keys, degree_cap)
            if not x_neighbors:
                continue
            projected[int(x)] = x_neighbors

            # Reverse-edge maintenance during projection.
            for p in x_neighbors:
                p = int(p)
                p_cands = list(projected.get(p, []))
                if int(x) not in p_cands:
                    p_cands.append(int(x))
                p_neighbors = self._acquire_neighbors_roar(p, p_cands, keys, degree_cap)
                if p_neighbors:
                    projected[p] = p_neighbors
                elif p in projected:
                    del projected[p]

        meta["projection_sec"] = time.time() - proj_start
        meta["projected_nodes"] = int(len(projected))
        if not projected:
            meta["stop_reason"] = "empty_projection"
            meta["total_sec"] = time.time() - total_start
            return empty_graph, meta

        # 3) Connectivity enhancement.
        enh_start = time.time()
        enhanced_nodes = 0
        if self.roar_enable_enhance:
            g_proj = {node: list(nei) for node, nei in projected.items()}
            nprime = {}
            if self.roar_entry in {"hub", "max_degree"}:
                entry_node = max(g_proj.keys(), key=lambda node: len(g_proj.get(node, [])))
            else:
                entry_node = -1

            for x in g_proj.keys():
                x = int(x)
                src_entry = x if self.roar_entry == "self" else int(entry_node)
                beam_candidates = self._beam_search_roar(
                    source_node=x,
                    adjacency=g_proj,
                    keys_cpu=keys,
                    entry_node=src_entry,
                    beam_l=enhance_limit,
                )
                if not beam_candidates:
                    continue

                x_prime = self._acquire_neighbors_roar(x, beam_candidates, keys, degree_cap)
                if not x_prime:
                    continue
                nprime[x] = x_prime
                enhanced_nodes += 1

                for p in x_prime:
                    p = int(p)
                    p_cands = list(nprime.get(p, []))
                    if x not in p_cands:
                        p_cands.append(x)
                    p_prime = self._acquire_neighbors_roar(p, p_cands, keys, degree_cap)
                    if p_prime:
                        nprime[p] = p_prime
                    elif p in nprime:
                        del nprime[p]

            # Merge enhancement into projected graph and re-enforce degree cap.
            for node, nprime_neighbors in nprime.items():
                node = int(node)
                merged = list(projected.get(node, []))
                merged.extend(nprime_neighbors)
                merged_neighbors = self._acquire_neighbors_roar(node, merged, keys, degree_cap)
                if merged_neighbors:
                    projected[node] = merged_neighbors
                elif node in projected:
                    del projected[node]

        meta["enhance_sec"] = time.time() - enh_start
        meta["enhanced_nodes"] = int(enhanced_nodes)

        csr_start = time.time()
        graph = self._adjacency_to_csr(projected)
        meta["csr_sec"] = time.time() - csr_start
        meta["total_sec"] = time.time() - total_start
        return graph, meta

    def _build_hub_seeds_from_graph(self, graph):
        """
        Pick high-degree non-static tokens as reusable decode seed anchors.
        """
        if self.seed_hub_k <= 0:
            return []
        if not (isinstance(graph, tuple) and len(graph) >= 2):
            return []
        offsets = graph[0]
        if offsets is None or offsets.shape[0] <= 1:
            return []
        degree = offsets[1:].astype(np.int64) - offsets[:-1].astype(np.int64)
        if degree.size == 0:
            return []
        if self.static_index_set:
            static_idx = np.fromiter(self.static_index_set, dtype=np.int64)
            static_idx = static_idx[(static_idx >= 0) & (static_idx < degree.shape[0])]
            if static_idx.size > 0:
                degree[static_idx] = -1
        valid_count = int(np.sum(degree >= 0))
        if valid_count <= 0:
            return []
        take = min(self.seed_hub_k, valid_count)
        if take <= 0:
            return []
        part = np.argpartition(degree, degree.shape[0] - take)[-take:]
        part = part[np.argsort(degree[part])[::-1]]
        return [int(x) for x in part.tolist() if int(degree[int(x)]) >= 0]

    def uses_flashattn_fused_prefill(self) -> bool:
        return bool(self.fa_fused_prefill)

    def register_fused_prefill_knn(self, layer_idx: int, knn_idx, profile: dict = None):
        """
        Register fused-prefill top-k indices produced during prefill attention.
        Accepted shapes:
          [1, seq, kv_head, q_knn]
          [seq, kv_head, q_knn]
          [kv_head, seq, q_knn]
        Stored format:
          [seq, kv_head, q_knn] int32 contiguous (on CPU).
        """
        ldx = int(layer_idx)
        if ldx < 0 or ldx >= self.layer_num:
            raise ValueError(f"Invalid layer_idx={layer_idx} for layer_num={self.layer_num}")

        if isinstance(knn_idx, torch.Tensor):
            arr = knn_idx.detach().to("cpu", non_blocking=False).numpy()
        else:
            arr = np.asarray(knn_idx)

        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise RuntimeError(
                    f"fused prefill top-k expects batch dim 1, got shape={tuple(arr.shape)}"
                )
            arr = arr[0]

        if arr.ndim != 3:
            raise RuntimeError(
                f"fused prefill top-k must be rank-3 or rank-4, got shape={tuple(arr.shape)}"
            )

        if arr.shape[0] == self.input_length and arr.shape[1] == self.kv_head:
            norm = arr
        elif arr.shape[0] == self.kv_head and arr.shape[1] == self.input_length:
            norm = np.transpose(arr, (1, 0, 2))
        else:
            raise RuntimeError(
                "fused prefill top-k shape does not match expected layout "
                f"(input_length={self.input_length}, kv_head={self.kv_head}), got {tuple(arr.shape)}"
            )

        if norm.shape[2] < self.q_knn:
            raise RuntimeError(
                f"fused prefill top-k last dim too small: got {norm.shape[2]}, need >= {self.q_knn}"
            )
        if norm.shape[2] > self.q_knn:
            norm = norm[:, :, :self.q_knn]

        norm = np.ascontiguousarray(norm.astype(np.int32, copy=False))
        profile_dict = profile if isinstance(profile, dict) else {}

        if self._fused_async_enabled:
            self._check_fused_score_mode_compat(profile_dict)
            executor = self._ensure_fused_prefill_executor()
            if executor is None:
                raise RuntimeError("fused overlap is enabled but executor is unavailable.")
            with self._fused_prefill_lock:
                if self._fused_prefill_submitted[ldx]:
                    raise RuntimeError(
                        f"Duplicate fused prefill registration for layer={ldx}."
                    )
                self._fused_prefill_submitted[ldx] = True
                self._fused_prefill_submit_count += 1
                self.fused_prefill_profiles[ldx] = profile_dict
            if self._profile_enabled():
                print(
                    f"[RetrievalAttention] fused_overlap submit layer={ldx} "
                    f"workers={self.fused_prefill_overlap_workers}",
                    flush=True,
                )
            try:
                future = executor.submit(self._finalize_fused_layer, ldx, norm, profile_dict)
            except Exception:
                with self._fused_prefill_lock:
                    self._fused_prefill_submitted[ldx] = False
                    self._fused_prefill_submit_count = max(0, self._fused_prefill_submit_count - 1)
                raise
            with self._fused_prefill_lock:
                self._fused_prefill_futures[ldx] = future
            return

        self.fused_prefill_knn[ldx] = norm
        self.fused_prefill_profiles[ldx] = profile_dict

    def _run_fused_shadow_compare(self, layer_idx: int, layer_knn: np.ndarray):
        """
        Sampled shadow parity check between fused-prefill KNN and baseline GPU-topk KNN.
        """
        if not self.fa_shadow_compare:
            return
        if self.cpu_queries is None:
            print(
                "[RetrievalAttention] fused shadow compare skipped: cpu_queries are not stored.",
                flush=True,
            )
            return

        ldx = int(layer_idx)
        hdx = 0
        sample_n = min(self.fa_shadow_sample, self.input_length)
        if sample_n <= 0:
            return
        if layer_knn.shape[0] < sample_n:
            sample_n = layer_knn.shape[0]
        if sample_n <= 0:
            return

        device = self.layer_mapping[str(ldx)]
        baseline_knn, _ = self._gpu_topk_knn(
            keys=self.cpu_keys[ldx][hdx, :self.input_length, :],
            queries=self.cpu_queries[ldx][hdx, :sample_n, :],
            device=device,
            already_normalized=False,
            force_torch_path=True,
        )
        fused_knn = np.ascontiguousarray(layer_knn[:sample_n, hdx, :], dtype=np.int32)
        rec = self._knn_recall_at_k(fused_knn, baseline_knn, self.q_knn)
        print(
            f"[RetrievalAttention] fused_shadow layer={ldx} head={hdx} "
            f"sample={sample_n} recall@{self.q_knn}={rec:.4f}",
            flush=True,
        )

    def _get_allocated_cpu_count(self) -> int:
        """
        Detect effective CPU allocation for this process/job.
        """
        def _parse_int(value):
            if value is None:
                return None
            s = str(value).strip()
            if not s:
                return None
            # e.g. "4(x2)" -> "4", "4,4" -> "4"
            s = s.split("(")[0].split(",")[0].strip()
            try:
                v = int(s)
            except Exception:
                return None
            return v if v > 0 else None

        for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            parsed = _parse_int(os.environ.get(key))
            if parsed is not None:
                return parsed

        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass

        return max(1, int(os.cpu_count() or 1))

    def _resolve_faiss_threads(self, cpu_cap: int, pipeline_enabled: bool):
        """
        Determine a safe FAISS thread count under scheduler CPU allocation.
        Reserve one CPU for the main thread when head pipelining is enabled.
        """
        reserve_for_pipeline = 1 if pipeline_enabled else 0
        usable_cpu_for_faiss = max(1, int(cpu_cap) - reserve_for_pipeline)
        requested = os.environ.get("FAISS_NUM_THREADS")
        if requested is None or requested == "":
            req = usable_cpu_for_faiss
        else:
            try:
                req = int(requested)
            except Exception:
                req = usable_cpu_for_faiss
        req = max(1, req)
        faiss_threads = min(req, usable_cpu_for_faiss)
        return faiss_threads, usable_cpu_for_faiss

    def _profile_enabled(self) -> bool:
        return os.environ.get("RETRIEVALATTN_PROFILE", "1") == "1"

    def _configure_faiss_threads_for_async(self):
        """
        Configure faiss threads before launching async fused-prefill finalize workers.
        """
        if faiss is None or self._faiss_threads_async_configured:
            return
        cpu_cap = self._get_allocated_cpu_count()
        try:
            num_threads, usable_cpu_for_faiss = self._resolve_faiss_threads(
                cpu_cap=cpu_cap,
                pipeline_enabled=False,
            )
            faiss.omp_set_num_threads(num_threads)
            self._faiss_threads_async_configured = True
            if self._profile_enabled():
                print(
                    "[RetrievalAttention] fused_overlap faiss threads "
                    f"set to {num_threads} (cpu_cap={cpu_cap}, faiss_cpu_budget={usable_cpu_for_faiss})",
                    flush=True,
                )
        except Exception as exc:
            print(
                "[RetrievalAttention] WARNING: failed to set faiss threads for fused overlap: "
                f"{exc}",
                flush=True,
            )

    def _ensure_fused_prefill_executor(self):
        if not self._fused_async_enabled:
            return None
        with self._fused_prefill_lock:
            if self._fused_prefill_executor is None:
                self._configure_faiss_threads_for_async()
                self._fused_prefill_executor = ThreadPoolExecutor(
                    max_workers=self.fused_prefill_overlap_workers
                )
        return self._fused_prefill_executor

    def _shutdown_fused_prefill_executor(self):
        with self._fused_prefill_lock:
            executor = self._fused_prefill_executor
            self._fused_prefill_executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def _check_fused_score_mode_compat(self, profile: dict):
        prof = profile if isinstance(profile, dict) else {}
        profile_path = str(prof.get("path", "")).strip().lower()
        if self.score_normalize and profile_path == "native_kernel_fused":
            normalize_applied = bool(prof.get("retrieval_normalize_applied", False))
            if not normalize_applied:
                raise RuntimeError(
                    "RETRIEVALATTN_SCORE_MODE=cosine requested but native fused retrieval path "
                    "did not apply retrieval normalization. Rebuild flash-attn fork with "
                    "retrieval_normalize support, or use RETRIEVALATTN_SCORE_MODE=ip."
                )

    def _finalize_fused_layer(self, ldx: int, layer_knn: np.ndarray, profile: dict = None):
        """
        Finalize one fused-prefill layer on CPU: decode index + graph + hub seeds.
        Runs in overlap worker thread when enabled.
        """
        layer_start = time.time()
        prof = profile if isinstance(profile, dict) else {}
        self._check_fused_score_mode_compat(prof)
        per_head_topk = None
        layer_topk = prof.get("topk_sec", prof.get("fused_sec"))
        if layer_topk is not None:
            try:
                layer_topk = float(layer_topk)
                per_head_topk = layer_topk / float(self.kv_head)
            except Exception:
                per_head_topk = None

        if self._profile_enabled() and prof:
            print(
                f"[RetrievalAttention] fused_overlap profile layer={ldx}: {prof}",
                flush=True,
            )

        try:
            for hdx in range(self.kv_head):
                head_start = time.time()
                knn = np.ascontiguousarray(layer_knn[:, hdx, :], dtype=np.int32)
                head_prof = {"topk_sec": per_head_topk} if per_head_topk is not None else {}
                result = self._finalize_gpu_head_build(ldx, hdx, knn, head_prof, head_start)
                self._commit_head_build_result(result)

            with self._fused_prefill_lock:
                self._fused_prefill_done[ldx] = True
                self._fused_prefill_done_count += 1
                self.fused_prefill_knn[ldx] = None
                self.fused_prefill_profiles[ldx] = None
            if self._profile_enabled():
                print(
                    f"[RetrievalAttention] fused_overlap done layer={ldx} "
                    f"time={time.time() - layer_start:.2f}s",
                    flush=True,
                )
        except Exception as exc:
            with self._fused_prefill_lock:
                self._fused_prefill_errors[ldx] = exc
            raise
        finally:
            layer_knn = None

    def _finalize_gpu_head_build(self, ldx: int, hdx: int, knn: np.ndarray, prof: dict, head_start: float):
        """
        CPU-side postprocess for one GPU-topk head:
        decode seed index + optional parity + CSR graph projection.
        """
        decode_index = None
        keys_cpu = None
        need_graph_keys = (self.graph_builder == "roar")
        need_keys_cpu = (
            need_graph_keys
            or
            self.decode_index_mode == "faiss"
            or (self.validate_parity and ldx == 0 and hdx == 0 and faiss is not None)
        )
        if need_keys_cpu:
            keys_cpu = (
                self.cpu_keys[ldx][hdx, :self.input_length, :]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            keys_cpu = self._score_transform_np(keys_cpu)

        if self.decode_index_mode == "faiss":
            if faiss is None:
                raise RuntimeError(
                    "RETRIEVALATTN_DECODE_INDEX=faiss requires faiss-cpu."
                )
            decode_index = faiss.IndexFlatIP(self.head_dim)
            decode_index.add(keys_cpu)

        parity_msg = None
        if self.validate_parity and ldx == 0 and hdx == 0 and faiss is not None:
            sample_n = min(256, self.input_length)
            queries_cpu = (
                self.cpu_queries[ldx][hdx, :sample_n, :]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            queries_cpu = self._score_transform_np(queries_cpu)
            ref_index = faiss.IndexFlatIP(self.head_dim)
            ref_index.add(keys_cpu)
            _, ref_knn = ref_index.search(queries_cpu, self.q_knn)
            rec = self._knn_recall_at_k(knn[:sample_n], ref_knn, self.q_knn)
            parity_msg = (
                f"[RetrievalAttention] parity layer=0 head=0 sample={sample_n} "
                f"recall@{self.q_knn}={rec:.4f}"
            )

        proj_start = time.time()
        graph, graph_meta = self._build_graph_csr_from_knn(knn, keys_cpu=keys_cpu)
        hub_seeds = self._build_hub_seeds_from_graph(graph)
        graph_edges = 0
        graph_has_weights = False
        if isinstance(graph, tuple) and len(graph) >= 2:
            graph_offsets = graph[0]
            if graph_offsets is not None and graph_offsets.shape[0] > 0:
                graph_edges = int(graph_offsets[-1])
            graph_has_weights = len(graph) >= 3
        proj_elapsed = time.time() - proj_start
        head_elapsed = time.time() - head_start
        topk_elapsed = prof.get("topk_sec", None)

        return {
            "ldx": ldx,
            "hdx": hdx,
            "decode_index": decode_index,
            "graph": graph,
            "hub_seeds": hub_seeds,
            "head_elapsed": head_elapsed,
            "topk_elapsed": topk_elapsed,
            "proj_elapsed": proj_elapsed,
            "graph_edges": graph_edges,
            "graph_has_weights": graph_has_weights,
            "graph_builder": graph_meta.get("builder", self.graph_builder) if isinstance(graph_meta, dict) else self.graph_builder,
            "graph_meta": graph_meta,
            "parity_msg": parity_msg,
        }

    def _commit_head_build_result(self, result: dict):
        """
        Commit one head build result and emit logs in a consistent format.
        """
        ldx = int(result["ldx"])
        hdx = int(result["hdx"])
        self.indexes[ldx][hdx] = result["decode_index"]
        self.graphs[ldx][hdx] = result["graph"]
        self.hub_seeds[ldx][hdx] = list(result.get("hub_seeds", []))

        parity_msg = result.get("parity_msg")
        if parity_msg:
            print(parity_msg)

        topk_elapsed = result.get("topk_elapsed", None)
        topk_str = f"{topk_elapsed:.2f}s" if topk_elapsed is not None else "n/a"
        graph_edges = int(result.get("graph_edges", 0))
        graph_weighted = int(bool(result.get("graph_has_weights", False)))
        graph_builder = str(result.get("graph_builder", self.graph_builder))
        graph_meta = result.get("graph_meta")
        extra = ""
        if isinstance(graph_meta, dict) and (
            (graph_builder == "roar" and self.roar_log)
            or (str(graph_meta.get("stop_reason", "ok")) != "ok")
        ):
            extra = (
                f" builder={graph_builder} "
                f"bip={float(graph_meta.get('bipartite_sec', 0.0)):.2f}s "
                f"enh={float(graph_meta.get('enhance_sec', 0.0)):.2f}s "
                f"csr={float(graph_meta.get('csr_sec', 0.0)):.2f}s "
                f"active_q={int(graph_meta.get('active_queries', 0))} "
                f"active_p={int(graph_meta.get('active_pivots', 0))} "
                f"nodes={int(graph_meta.get('projected_nodes', 0))} "
                f"enh_nodes={int(graph_meta.get('enhanced_nodes', 0))} "
                f"stop={graph_meta.get('stop_reason', 'ok')}"
            )
        print(
            f"[RetrievalAttention] index built layer={ldx} head={hdx} "
            f"time={result['head_elapsed']:.2f}s topk={topk_str} proj={result['proj_elapsed']:.2f}s "
            f"edges={graph_edges} weighted={graph_weighted}{extra}"
        )

    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, batch_idx):
        """
        Store Q/K/V for prefill. Build static GPU KV for prefix+suffix.
        """
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert bsz == 1, "Only batch_size=1 supported for RetrievalAttention prototype."

        valid_start = self.valid_start[batch_idx]

        # Store full keys/values on CPU (per head)
        keys = key_states[0, valid_start:valid_start + self.input_length, :, :].transpose(0, 1).contiguous()
        values = value_states[0, valid_start:valid_start + self.input_length, :, :].transpose(0, 1).contiguous()
        self.cpu_keys[layer_idx][:, :self.input_length, :].copy_(keys, non_blocking=True)
        self.cpu_values[layer_idx][:, :self.input_length, :].copy_(values, non_blocking=True)

        if self.cpu_queries is not None:
            # Store queries for graph build.
            # query_states is [bs, seq, num_heads, head_dim]; we down-project to kv_head by
            # averaging over the group_size heads per kv head.
            queries = query_states[0, valid_start:valid_start + self.input_length, :, :]
            # [seq, num_heads, head_dim] -> [seq, kv_head, group_size, head_dim] -> mean over group
            queries = queries.reshape(self.input_length, self.kv_head, self.group_size, self.head_dim).mean(dim=2)
            # [seq, kv_head, head_dim] -> [kv_head, seq, head_dim]
            queries = queries.transpose(0, 1).contiguous()
            self.cpu_queries[layer_idx][:, :self.input_length, :].copy_(queries, non_blocking=True)

        # Build static GPU KV (prefix + suffix)
        prefix = key_states[0, valid_start:valid_start + self.static_pattern_start, :, :].transpose(0, 1).contiguous()
        suffix = key_states[0, valid_start + self.input_length - self.static_pattern_end:valid_start + self.input_length, :, :].transpose(0, 1).contiguous()
        prefix_v = value_states[0, valid_start:valid_start + self.static_pattern_start, :, :].transpose(0, 1).contiguous()
        suffix_v = value_states[0, valid_start + self.input_length - self.static_pattern_end:valid_start + self.input_length, :, :].transpose(0, 1).contiguous()

        self.static_gpu_keys[layer_idx][:, :self.static_pattern_start, :] = prefix.to(self.layer_mapping[str(layer_idx)])
        self.static_gpu_keys[layer_idx][:, self.static_pattern_start:, :] = suffix.to(self.layer_mapping[str(layer_idx)])
        self.static_gpu_values[layer_idx][:, :self.static_pattern_start, :] = prefix_v.to(self.layer_mapping[str(layer_idx)])
        self.static_gpu_values[layer_idx][:, self.static_pattern_start:, :] = suffix_v.to(self.layer_mapping[str(layer_idx)])

        if (layer_idx == self.layer_num - 1) and (batch_idx + bsz == self.batch_size):
            self.context += seq_len

        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

    def prepare_cache(self):
        """
        Build ANN indexes and projected K–K graph. Free query storage afterwards.
        """
        use_gpu_topk = os.environ.get("RETRIEVALATTN_GPU_TOPK", "0") == "1"
        use_fused_prefill = self.fa_fused_prefill
        if self._built:
            return

        if self.decode_index_mode == "faiss" and faiss is None:
            raise RuntimeError("RETRIEVALATTN_DECODE_INDEX=faiss requires faiss-cpu.")
        if faiss is None and not use_gpu_topk and not use_fused_prefill:
            raise RuntimeError("faiss is not installed. Please install faiss-cpu to use RetrievalAttention.")

        if use_fused_prefill:
            if self._fused_async_enabled:
                missing_layers = [
                    ldx for ldx in range(self.layer_num)
                    if not self._fused_prefill_submitted[ldx]
                ]
            else:
                missing_layers = [ldx for ldx in range(self.layer_num) if self.fused_prefill_knn[ldx] is None]
            if missing_layers:
                raise RuntimeError(
                    "RETRIEVALATTN_FA_FUSED_PREFILL=1 but fused prefill KNN is missing for layers: "
                    f"{missing_layers}. Ensure fused prefill attention path is active."
                )

        start_ts = time.time()
        pipeline_requested = (
            use_gpu_topk
            and (not use_fused_prefill)
            and os.environ.get("RETRIEVALATTN_HEAD_PIPELINE", "1") == "1"
        )
        try:
            pipeline_depth = max(1, int(os.environ.get("RETRIEVALATTN_HEAD_PIPELINE_DEPTH", "1")))
        except Exception:
            pipeline_depth = 1
        try:
            pipeline_min_cpus = max(1, int(os.environ.get("RETRIEVALATTN_HEAD_PIPELINE_MIN_CPUS", "2")))
        except Exception:
            pipeline_min_cpus = 2
        cpu_cap = self._get_allocated_cpu_count()
        pipeline_enabled = pipeline_requested and (cpu_cap >= pipeline_min_cpus)
        profile_enabled = os.environ.get("RETRIEVALATTN_PROFILE", "1") == "1"

        # Enable Faiss threads within scheduler CPU allocation.
        num_threads = None
        usable_cpu_for_faiss = cpu_cap
        if faiss is not None:
            try:
                num_threads, usable_cpu_for_faiss = self._resolve_faiss_threads(
                    cpu_cap=cpu_cap,
                    pipeline_enabled=pipeline_enabled,
                )
                faiss.omp_set_num_threads(num_threads)
            except Exception:
                num_threads = None

        if num_threads is not None:
            thread_msg = (
                f", faiss_threads={num_threads}, cpu_cap={cpu_cap}, "
                f"faiss_cpu_budget={usable_cpu_for_faiss}"
            )
        else:
            thread_msg = f", cpu_cap={cpu_cap}, faiss_cpu_budget={usable_cpu_for_faiss}"
        pipeline_msg = (
            f", pipeline_requested={int(pipeline_requested)}, "
            f"pipeline_enabled={int(pipeline_enabled)}, "
            f"pipeline_depth={pipeline_depth}, "
            f"pipeline_min_cpus={pipeline_min_cpus}"
        )
        fused_overlap_msg = (
            f", fused_overlap_cfg={int(self.fused_prefill_overlap)}, "
            f"fused_overlap_enabled={int(self._fused_async_enabled)}, "
            f"fused_overlap_workers={self.fused_prefill_overlap_workers}"
        )
        if use_fused_prefill:
            mode = "flashattn_fused_prefill"
        else:
            mode = "gpu_topk" if use_gpu_topk else "faiss_cpu"
        print(
            f"[RetrievalAttention] Building ANN indexes (layers={self.layer_num}, kv_heads={self.kv_head}, "
            f"tokens={self.input_length}, mode={mode}, decode_index={self.decode_index_mode}, "
            f"seed_mode={self.seed_mode}, query_mode={self.query_mode}, score_mode={self.score_mode}, "
            f"graph_builder={self.graph_builder}, "
            f"graph_weighted={int(self.graph_weighted)}, clique_m={self.graph_clique_m}, "
            f"graph_return_weights={int(self.graph_return_weights)}, graph_weight_dtype={self.graph_weight_dtype_name}, "
            f"roar_nq={self.roar_nq}, roar_l={self.roar_l}, roar_m={self.roar_m}, "
            f"roar_enhance={int(self.roar_enable_enhance)}, roar_enhance_l={self.roar_enhance_l}, "
            f"roar_entry={self.roar_entry}, roar_max_query_per_pivot={self.roar_max_query_per_pivot}, "
            f"graph_expand={int(self.graph_expand)}, "
            f"expand_width={self.expand_width}, min_visits={self.min_visits}, "
            f"max_visits={self.max_visits}, stop_patience={self.stop_patience}, "
            f"stop_margin={self.stop_margin:.4f}, frontier_topn={self.frontier_topn}, "
            f"rerank={int(self.rerank)}, "
            f"seed_ratio={self.seed_ratio:.2f}, cand_mult={self.candidate_multiplier}, "
            f"seed_k_mult={self.seed_k_mult}, seed_prev_k={self.seed_prev_k}, "
            f"seed_hub_k={self.seed_hub_k}, seed_tail_k={self.seed_tail_k}, "
            f"fused_prefill={int(use_fused_prefill)}, "
            f"fused_shadow={int(self.fa_shadow_compare)}){thread_msg}{pipeline_msg}{fused_overlap_msg}"
        )
        if pipeline_requested and not pipeline_enabled:
            print(
                f"[RetrievalAttention] head pipeline auto-disabled: cpu_cap={cpu_cap} < "
                f"pipeline_min_cpus={pipeline_min_cpus}"
            )

        if use_fused_prefill and self._fused_async_enabled:
            wait_total = 0.0
            wait_count = 0
            first_error_msg = None
            first_error_exc = None
            for ldx in range(self.layer_num):
                with self._fused_prefill_lock:
                    future = self._fused_prefill_futures.get(ldx)
                if future is None:
                    raise RuntimeError(
                        f"Missing fused-overlap future for layer={ldx}. "
                        "Prefill likely did not register all layers."
                    )
                wait_start = time.time()
                try:
                    future.result()
                except Exception as exc:
                    if first_error_msg is None:
                        first_error_msg = (
                            f"Fused overlap finalize failed at layer={ldx}: {exc}"
                        )
                        first_error_exc = exc
                wait_total += (time.time() - wait_start)
                wait_count += 1

            self._shutdown_fused_prefill_executor()
            with self._fused_prefill_lock:
                done_layers = sum(1 for done in self._fused_prefill_done if done)
                submitted_layers = sum(1 for submitted in self._fused_prefill_submitted if submitted)
                if first_error_msg is None and self._fused_prefill_errors:
                    err_ldx = sorted(self._fused_prefill_errors.keys())[0]
                    err = self._fused_prefill_errors[err_ldx]
                    first_error_msg = (
                        f"Fused overlap finalize recorded error at layer={err_ldx}: {err}"
                    )
                    first_error_exc = err
                self._fused_prefill_futures.clear()

            if first_error_msg is not None:
                raise RuntimeError(first_error_msg) from first_error_exc
            if done_layers != self.layer_num:
                raise RuntimeError(
                    "Fused overlap finalize incomplete: "
                    f"done_layers={done_layers}, expected={self.layer_num}, "
                    f"submitted_layers={submitted_layers}"
                )

            if profile_enabled:
                avg_wait = (wait_total / float(wait_count)) if wait_count > 0 else 0.0
                print(
                    "[RetrievalAttention] fused_overlap barrier "
                    f"submits={self._fused_prefill_submit_count} "
                    f"done={self._fused_prefill_done_count} "
                    f"waits={wait_count} wait_total={wait_total:.2f}s wait_avg={avg_wait:.2f}s",
                    flush=True,
                )

            self.cpu_queries = None
            self.prev_decode_seeds = [[[] for _ in range(self.kv_head)] for _ in range(self.layer_num)]
            self._built = True
            return

        for ldx in range(self.layer_num):
            layer_start = time.time()

            if use_fused_prefill:
                layer_knn = self.fused_prefill_knn[ldx]
                if layer_knn is None:
                    raise RuntimeError(f"Missing fused prefill KNN for layer={ldx}")
                profile = self.fused_prefill_profiles[ldx] if isinstance(self.fused_prefill_profiles[ldx], dict) else {}
                self._check_fused_score_mode_compat(profile)
                if self.fa_shadow_compare and ldx == 0:
                    self._run_fused_shadow_compare(ldx, layer_knn)

                layer_topk = profile.get("topk_sec", profile.get("fused_sec"))
                if layer_topk is not None:
                    try:
                        layer_topk = float(layer_topk)
                    except Exception:
                        layer_topk = None
                per_head_topk = (layer_topk / float(self.kv_head)) if layer_topk is not None else None
                if profile_enabled and profile:
                    print(
                        f"[RetrievalAttention] flashattn fused profile layer={ldx}: {profile}",
                        flush=True,
                    )

                for hdx in range(self.kv_head):
                    head_start = time.time()
                    knn = np.ascontiguousarray(layer_knn[:, hdx, :], dtype=np.int32)
                    prof = {"topk_sec": per_head_topk} if per_head_topk is not None else {}
                    result = self._finalize_gpu_head_build(ldx, hdx, knn, prof, head_start)
                    self._commit_head_build_result(result)

                # Free per-layer fused KNN/prof to reduce host memory pressure.
                self.fused_prefill_knn[ldx] = None
                self.fused_prefill_profiles[ldx] = None

                layer_elapsed = time.time() - layer_start
                total_elapsed = time.time() - start_ts
                print(f"[RetrievalAttention] layer {ldx} done in {layer_elapsed:.2f}s (total {total_elapsed:.2f}s)")
                continue

            layer_gpu_cache = os.environ.get("RETRIEVALATTN_LAYER_GPU_CACHE", "1") == "1" and use_gpu_topk
            keys_layer_gpu = None
            queries_layer_gpu = None
            if layer_gpu_cache and self.cpu_queries is not None:
                try:
                    device = self.layer_mapping[str(ldx)]
                    keys_layer_gpu = self.cpu_keys[ldx][:, :self.input_length, :].detach().float().to(device, non_blocking=True)
                    queries_layer_gpu = self.cpu_queries[ldx][:, :self.input_length, :].detach().float().to(device, non_blocking=True)
                    keys_layer_gpu = self._score_transform_torch(keys_layer_gpu)
                    queries_layer_gpu = self._score_transform_torch(queries_layer_gpu)
                    torch.cuda.synchronize(device=device)
                except Exception as exc:
                    print(f"[RetrievalAttention] layer {ldx} GPU cache failed, fallback to streaming: {exc}")
                    keys_layer_gpu = None
                    queries_layer_gpu = None

            use_head_pipeline = pipeline_enabled
            pipeline_executor = ThreadPoolExecutor(max_workers=1) if use_head_pipeline else None
            pending_head_futures = []
            pipeline_submit_count = 0
            pipeline_wait_count = 0
            pipeline_wait_sec_total = 0.0
            try:
                for hdx in range(self.kv_head):
                    head_start = time.time()
                    if use_gpu_topk and profile_enabled:
                        print(
                            f"[RetrievalAttention] gpu_topk begin layer={ldx} head={hdx}",
                            flush=True,
                        )
                    if use_gpu_topk:
                        device = self.layer_mapping[str(ldx)]
                        if keys_layer_gpu is not None and queries_layer_gpu is not None:
                            knn, prof = self._gpu_topk_knn(
                                keys=keys_layer_gpu[hdx],
                                queries=queries_layer_gpu[hdx],
                                device=device,
                                already_normalized=True,
                            )
                        else:
                            if self.cpu_queries is None:
                                raise RuntimeError(
                                    "cpu_queries are unavailable for GPU-topk build. "
                                    "Disable RETRIEVALATTN_FA_FUSED_PREFILL or enable fused registration."
                                )
                            knn, prof = self._gpu_topk_knn(
                                keys=self.cpu_keys[ldx][hdx, :self.input_length, :],
                                queries=self.cpu_queries[ldx][hdx, :self.input_length, :],
                                device=device,
                                already_normalized=False,
                            )

                        if pipeline_executor is not None:
                            pending_head_futures.append(
                                pipeline_executor.submit(
                                    self._finalize_gpu_head_build,
                                    ldx,
                                    hdx,
                                    knn,
                                    prof,
                                    head_start,
                                )
                            )
                            pipeline_submit_count += 1
                            while len(pending_head_futures) > pipeline_depth:
                                wait_start = time.time()
                                done = pending_head_futures.pop(0).result()
                                pipeline_wait_sec_total += time.time() - wait_start
                                pipeline_wait_count += 1
                                self._commit_head_build_result(done)
                        else:
                            result = self._finalize_gpu_head_build(ldx, hdx, knn, prof, head_start)
                            self._commit_head_build_result(result)
                    else:
                        if self.cpu_queries is None:
                            raise RuntimeError(
                                "cpu_queries are unavailable for CPU index build."
                            )
                        keys = (
                            self.cpu_keys[ldx][hdx, :self.input_length, :]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        queries = (
                            self.cpu_queries[ldx][hdx, :self.input_length, :]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )

                        keys = self._score_transform_np(keys)
                        queries = self._score_transform_np(queries)

                        index = faiss.IndexFlatIP(self.head_dim)
                        index.add(keys)
                        self.indexes[ldx][hdx] = index if self.decode_index_mode == "faiss" else None

                        # Q->K KNN (exact on CPU)
                        _, knn = index.search(queries, self.q_knn)
                        prof = {"topk_sec": time.time() - head_start}

                        proj_start = time.time()
                        graph, graph_meta = self._build_graph_csr_from_knn(knn, keys_cpu=keys)
                        self.graphs[ldx][hdx] = graph
                        self.hub_seeds[ldx][hdx] = self._build_hub_seeds_from_graph(graph)
                        graph_edges = 0
                        graph_weighted = 0
                        if isinstance(graph, tuple) and len(graph) >= 2:
                            graph_offsets = graph[0]
                            if graph_offsets is not None and graph_offsets.shape[0] > 0:
                                graph_edges = int(graph_offsets[-1])
                            graph_weighted = int(len(graph) >= 3)

                        head_elapsed = time.time() - head_start
                        proj_elapsed = time.time() - proj_start
                        topk_elapsed = prof.get("topk_sec", None)
                        topk_str = f"{topk_elapsed:.2f}s" if topk_elapsed is not None else "n/a"
                        extra = ""
                        graph_builder = str(graph_meta.get("builder", self.graph_builder)) if isinstance(graph_meta, dict) else self.graph_builder
                        if isinstance(graph_meta, dict) and (
                            (graph_builder == "roar" and self.roar_log)
                            or (str(graph_meta.get("stop_reason", "ok")) != "ok")
                        ):
                            extra = (
                                f" builder={graph_builder} "
                                f"bip={float(graph_meta.get('bipartite_sec', 0.0)):.2f}s "
                                f"enh={float(graph_meta.get('enhance_sec', 0.0)):.2f}s "
                                f"csr={float(graph_meta.get('csr_sec', 0.0)):.2f}s "
                                f"active_q={int(graph_meta.get('active_queries', 0))} "
                                f"active_p={int(graph_meta.get('active_pivots', 0))} "
                                f"nodes={int(graph_meta.get('projected_nodes', 0))} "
                                f"enh_nodes={int(graph_meta.get('enhanced_nodes', 0))} "
                                f"stop={graph_meta.get('stop_reason', 'ok')}"
                            )
                        print(
                            f"[RetrievalAttention] index built layer={ldx} head={hdx} "
                            f"time={head_elapsed:.2f}s topk={topk_str} proj={proj_elapsed:.2f}s "
                            f"edges={graph_edges} weighted={graph_weighted}{extra}"
                        )

                for pending in pending_head_futures:
                    wait_start = time.time()
                    done = pending.result()
                    pipeline_wait_sec_total += time.time() - wait_start
                    pipeline_wait_count += 1
                    self._commit_head_build_result(done)
            finally:
                if pipeline_executor is not None:
                    pipeline_executor.shutdown(wait=True)

            if use_head_pipeline:
                avg_wait = (
                    pipeline_wait_sec_total / float(pipeline_wait_count)
                    if pipeline_wait_count > 0
                    else 0.0
                )
                print(
                    f"[RetrievalAttention] pipeline layer={ldx} submits={pipeline_submit_count} "
                    f"waits={pipeline_wait_count} wait_total={pipeline_wait_sec_total:.2f}s "
                    f"wait_avg={avg_wait:.2f}s depth={pipeline_depth}"
                )

            layer_elapsed = time.time() - layer_start
            total_elapsed = time.time() - start_ts
            print(f"[RetrievalAttention] layer {ldx} done in {layer_elapsed:.2f}s (total {total_elapsed:.2f}s)")

            if keys_layer_gpu is not None or queries_layer_gpu is not None:
                del keys_layer_gpu
                del queries_layer_gpu
                torch.cuda.empty_cache()

        # Free queries to save CPU memory after graph/index finalize.
        self.cpu_queries = None
        self._shutdown_fused_prefill_executor()
        with self._fused_prefill_lock:
            self._fused_prefill_futures.clear()
        self.prev_decode_seeds = [[[] for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._built = True

    def _gpu_topk_knn(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        device: str,
        already_normalized: bool = False,
        force_torch_path: bool = False,
    ):
        """
        Exact blockwise Q·K^T on GPU with running top-k per query.
        Returns knn indices as numpy int32 array [num_queries, q_knn].
        """
        q_block = int(os.environ.get("RETRIEVALATTN_Q_BLOCK", "512"))
        k_block = int(os.environ.get("RETRIEVALATTN_K_BLOCK", "4096"))
        custom_kernel = (not force_torch_path) and (os.environ.get("RETRIEVALATTN_CUSTOM_QK_TOPK", "0") == "1")
        custom_max_block_q = 64
        custom_max_block_k = 256
        custom_max_launch_q_chunk = 1024
        custom_block_q = int(
            os.environ.get(
                "RETRIEVALATTN_CUSTOM_QK_TOPK_BLOCK_Q",
                str(min(max(1, q_block), custom_max_block_q)),
            )
        )
        custom_block_q = max(1, custom_block_q)
        custom_launch_q_chunk = int(
            os.environ.get(
                "RETRIEVALATTN_CUSTOM_QK_TOPK_LAUNCH_Q_CHUNK",
                str(custom_max_launch_q_chunk),
            )
        )
        custom_launch_q_chunk = max(0, custom_launch_q_chunk)
        custom_block_d = int(os.environ.get("RETRIEVALATTN_CUSTOM_QK_TOPK_BLOCK_D", "32"))
        custom_block_k = int(
            os.environ.get(
                "RETRIEVALATTN_CUSTOM_QK_TOPK_BLOCK_K",
                str(min(max(1, k_block), custom_max_block_k)),
            )
        )
        custom_block_k = max(1, custom_block_k)
        profile = os.environ.get("RETRIEVALATTN_PROFILE", "1") == "1"
        overlap = os.environ.get("RETRIEVALATTN_OVERLAP", "1") == "1"

        q = queries.detach().float()
        k = keys.detach().float()
        q_on_gpu = q.is_cuda
        k_on_gpu = k.is_cuda
        overlap = overlap and not (q_on_gpu and k_on_gpu)

        num_q = q.shape[0]
        num_k = k.shape[0]
        k_top = self.q_knn

        req_block_q = custom_block_q
        req_block_k = custom_block_k
        req_launch_q_chunk = custom_launch_q_chunk
        custom_block_q = min(custom_block_q, custom_max_block_q)
        custom_block_k = min(custom_block_k, custom_max_block_k)
        if custom_launch_q_chunk > 0:
            custom_launch_q_chunk = min(custom_launch_q_chunk, custom_max_launch_q_chunk)
        custom_launch_q_chunk = max(custom_block_q, custom_launch_q_chunk) if custom_launch_q_chunk > 0 else 0
        if profile and (
            custom_block_q != req_block_q
            or custom_block_k != req_block_k
            or custom_launch_q_chunk != req_launch_q_chunk
        ):
            print(
                "[RetrievalAttention] custom_fused auto-tune: "
                f"block_q {req_block_q}->{custom_block_q}, "
                f"block_k {req_block_k}->{custom_block_k}, "
                f"launch_q_chunk {req_launch_q_chunk}->{custom_launch_q_chunk}",
                flush=True,
            )

        if custom_kernel:
            if fused_qk_topk_triton is None:
                raise RuntimeError(
                    "[RetrievalAttention] custom qk+topk kernel requested but import failed."
                )

            t_total = time.time()
            t_transfer = 0.0
            with torch.no_grad():
                if not q_on_gpu:
                    t0 = time.time()
                    q = q.to(device, non_blocking=True)
                    t_transfer += time.time() - t0
                if not k_on_gpu:
                    t0 = time.time()
                    k = k.to(device, non_blocking=True)
                    t_transfer += time.time() - t0

                if profile:
                    print(
                        "[RetrievalAttention] gpu_topk(custom_fused) launch: "
                        f"q={num_q} k={num_k} k_top={k_top} "
                        f"block_q={custom_block_q} block_k={custom_block_k} block_d={custom_block_d} "
                        f"launch_q_chunk={custom_launch_q_chunk}",
                        flush=True,
                    )
                try:
                    _, idx = fused_qk_topk_triton(
                        q,
                        k,
                        k_top=k_top,
                        normalize=(self.score_normalize and (not already_normalized)),
                        block_q=custom_block_q,
                        block_k=custom_block_k,
                        block_d=custom_block_d,
                        launch_q_chunk=custom_launch_q_chunk,
                        verbose=profile,
                        return_scores=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "[RetrievalAttention] custom qk+topk kernel failed"
                    ) from exc

                total = time.time() - t_total
                if profile:
                    print(
                        f"[RetrievalAttention] gpu_topk(custom_fused) profile: "
                        f"total={total:.2f}s transfer={t_transfer:.2f}s fused={total - t_transfer:.2f}s",
                        flush=True,
                    )
                knn_out = idx.to(torch.int64).cpu().numpy().astype(np.int32, copy=False)
                return knn_out, {
                    "total_sec": total,
                    "transfer_sec": t_transfer,
                    "matmul_sec": 0.0,
                    "topk_sec": max(0.0, total - t_transfer),
                    "fused_sec": max(0.0, total - t_transfer),
                    "path": "custom_fused",
                }

        knn_out_gpu = torch.empty((num_q, k_top), dtype=torch.int64, device=device)
        t_total = time.time()
        t_transfer = 0.0
        t_matmul = 0.0
        t_topk = 0.0
        with torch.no_grad():
            transfer_stream = None
            if overlap and torch.cuda.is_available():
                transfer_stream = torch.cuda.Stream(device=device)

            for q_start in range(0, num_q, q_block):
                q_end = min(num_q, q_start + q_block)
                if q_on_gpu:
                    q_chunk = q[q_start:q_end]
                else:
                    t0 = time.time()
                    q_chunk = q[q_start:q_end].to(device, non_blocking=True)
                    t_transfer += time.time() - t0
                if self.score_normalize and not already_normalized:
                    q_chunk = F.normalize(q_chunk, dim=-1)

                top_scores = torch.full((q_chunk.shape[0], k_top), -1e9, device=device)
                top_indices = torch.full((q_chunk.shape[0], k_top), -1, device=device, dtype=torch.int64)

                if transfer_stream is None:
                    for k_start in range(0, num_k, k_block):
                        k_end = min(num_k, k_start + k_block)
                        if k_on_gpu:
                            k_chunk = k[k_start:k_end]
                        else:
                            t0 = time.time()
                            k_chunk = k[k_start:k_end].to(device, non_blocking=True)
                            t_transfer += time.time() - t0
                        if self.score_normalize and not already_normalized:
                            k_chunk = F.normalize(k_chunk, dim=-1)

                        t0 = time.time()
                        scores = torch.matmul(q_chunk, k_chunk.transpose(0, 1))  # [Bq, Bk]
                        t_matmul += time.time() - t0
                        t0 = time.time()
                        vals, idx = torch.topk(
                            scores,
                            k=min(k_top, k_chunk.shape[0]),
                            dim=1,
                            sorted=False,
                        )
                        idx = idx + k_start
                        t_topk += time.time() - t0

                        # Merge with running top-k
                        merged_scores = torch.cat([top_scores, vals], dim=1)
                        merged_idx = torch.cat([top_indices, idx], dim=1)
                        t0 = time.time()
                        new_vals, new_pos = torch.topk(merged_scores, k=k_top, dim=1)
                        top_scores = new_vals
                        top_indices = torch.gather(merged_idx, 1, new_pos)
                        t_topk += time.time() - t0
                else:
                    # Prefetch first k-block
                    k_start = 0
                    k_end = min(num_k, k_block)
                    with torch.cuda.stream(transfer_stream):
                        if k_on_gpu:
                            k_chunk = k[k_start:k_end]
                        else:
                            t0 = time.time()
                            k_chunk = k[k_start:k_end].to(device, non_blocking=True)
                            t_transfer += time.time() - t0
                        if self.score_normalize and not already_normalized:
                            k_chunk = F.normalize(k_chunk, dim=-1)

                    while k_start < num_k:
                        # Wait for the current k_chunk to be ready
                        torch.cuda.current_stream(device=device).wait_stream(transfer_stream)

                        # Use current k_chunk for compute
                        t0 = time.time()
                        scores = torch.matmul(q_chunk, k_chunk.transpose(0, 1))  # [Bq, Bk]
                        t_matmul += time.time() - t0
                        t0 = time.time()
                        vals, idx = torch.topk(
                            scores,
                            k=min(k_top, k_chunk.shape[0]),
                            dim=1,
                            sorted=False,
                        )
                        idx = idx + k_start
                        t_topk += time.time() - t0

                        # Merge with running top-k
                        merged_scores = torch.cat([top_scores, vals], dim=1)
                        merged_idx = torch.cat([top_indices, idx], dim=1)
                        t0 = time.time()
                        new_vals, new_pos = torch.topk(merged_scores, k=k_top, dim=1)
                        top_scores = new_vals
                        top_indices = torch.gather(merged_idx, 1, new_pos)
                        t_topk += time.time() - t0

                        # Prefetch next k-block
                        k_start_next = k_start + k_block
                        if k_start_next >= num_k:
                            break
                        k_end_next = min(num_k, k_start_next + k_block)
                        with torch.cuda.stream(transfer_stream):
                            if k_on_gpu:
                                k_chunk = k[k_start_next:k_end_next]
                            else:
                                t0 = time.time()
                                k_chunk = k[k_start_next:k_end_next].to(device, non_blocking=True)
                                t_transfer += time.time() - t0
                            if self.score_normalize and not already_normalized:
                                k_chunk = F.normalize(k_chunk, dim=-1)

                        k_start = k_start_next

                knn_out_gpu[q_start:q_end] = top_indices

        total = time.time() - t_total
        if profile:
            print(
                f"[RetrievalAttention] gpu_topk profile: total={total:.2f}s "
                f"transfer={t_transfer:.2f}s matmul={t_matmul:.2f}s topk={t_topk:.2f}s"
            )
        knn_out = knn_out_gpu.cpu().numpy().astype(np.int32, copy=False)
        return knn_out, {
            "total_sec": total,
            "transfer_sec": t_transfer,
            "matmul_sec": t_matmul,
            "topk_sec": t_topk,
        }

    def sync(self, layer_idx, start_bdx):
        """
        Keep interface compatibility with other KV caches.
        RetrievalAttention does not use async GPU copy events, so this is a no-op.
        """
        return

    def reset_decode_profile(self):
        int_keys = {"calls", "heads", "visited_total", "candidates_total"}
        for key in self._decode_profile_stats:
            if key in int_keys:
                self._decode_profile_stats[key] = 0
            else:
                self._decode_profile_stats[key] = 0.0

    def report_decode_profile(self, reset: bool = False):
        if not self.decode_profile:
            return None
        stats = self._decode_profile_stats
        total = float(stats["compute_total_sec"])
        if total <= 0.0:
            return None

        retrieve = float(stats["retrieve_total_sec"])
        gather = float(stats["gather_total_sec"])
        attn = float(stats["attn_total_sec"])
        other = max(0.0, total - retrieve - gather - attn)

        def pct(v: float) -> float:
            return 100.0 * v / total if total > 0 else 0.0

        msg = (
            "[RetrievalAttention] decode_profile "
            f"calls={int(stats['calls'])} heads={int(stats['heads'])} "
            f"total={total:.3f}s | "
            f"retrieve={retrieve:.3f}s ({pct(retrieve):.1f}%) "
            f"[seed={stats['retrieve_seed_sec']:.3f}s, "
            f"graph={stats['retrieve_graph_sec']:.3f}s, "
            f"rerank={stats['retrieve_rerank_sec']:.3f}s, "
            f"finalize={stats['retrieve_finalize_sec']:.3f}s] | "
            f"gather={gather:.3f}s ({pct(gather):.1f}%) | "
            f"attn={attn:.3f}s ({pct(attn):.1f}%) | "
            f"other={other:.3f}s ({pct(other):.1f}%) | "
            f"visited_total={int(stats['visited_total'])} "
            f"candidates_total={int(stats['candidates_total'])}"
        )
        if reset:
            self.reset_decode_profile()
        return msg

    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """
        Append newly generated token to CPU KV and update static suffix window.
        """
        # key_states/value_states: [bs, 1, kv_head, head_dim]
        pos = self.input_length + self.decode_pos
        if pos < self.cpu_keys[layer_idx].shape[1]:
            self.cpu_keys[layer_idx][:, pos:pos + 1, :].copy_(key_states[0].transpose(0, 1), non_blocking=True)
            self.cpu_values[layer_idx][:, pos:pos + 1, :].copy_(value_states[0].transpose(0, 1), non_blocking=True)

        # Update static suffix window (shift left, append new token)
        if self.static_pattern_end > 0:
            suffix = self.static_gpu_keys[layer_idx][:, self.static_pattern_start:, :]
            suffix_v = self.static_gpu_values[layer_idx][:, self.static_pattern_start:, :]
            suffix = torch.roll(suffix, shifts=-1, dims=1)
            suffix_v = torch.roll(suffix_v, shifts=-1, dims=1)
            suffix[:, -1, :] = key_states[0, 0].to(self.layer_mapping[str(layer_idx)])
            suffix_v[:, -1, :] = value_states[0, 0].to(self.layer_mapping[str(layer_idx)])
            self.static_gpu_keys[layer_idx][:, self.static_pattern_start:, :] = suffix
            self.static_gpu_values[layer_idx][:, self.static_pattern_start:, :] = suffix_v

        if layer_idx == self.layer_num - 1:
            self.decode_pos += 1
            self.context += 1

        return None, None

    def _retrieve_tokens(self, ldx, hdx, query_group):
        """
        Retrieve token indices using seed search + adaptive best-first K-K graph expansion.
        Final candidate list is reranked with the configured retrieval score mode.
        """
        profile = None
        total_start = None
        if self.decode_profile:
            total_start = time.perf_counter()
            profile = {
                "total_sec": 0.0,
                "seed_sec": 0.0,
                "graph_sec": 0.0,
                "rerank_sec": 0.0,
                "finalize_sec": 0.0,
                "visited": 0,
                "candidates": 0,
                "stop_reason": "n/a",
            }

        def finish(tokens, stop_reason: str, visited: int = 0, candidates: int = 0):
            if profile is not None:
                profile["stop_reason"] = stop_reason
                profile["visited"] = int(visited)
                profile["candidates"] = int(candidates)
                profile["total_sec"] = time.perf_counter() - total_start
            return tokens, profile

        index = self.indexes[ldx][hdx]
        graph = self.graphs[ldx][hdx]
        graph_is_csr = isinstance(graph, tuple) and len(graph) >= 2
        if graph_is_csr:
            graph_offsets = graph[0]
            graph_neighbors = graph[1]
            if int(graph_offsets[-1]) != int(graph_neighbors.shape[0]):
                raise RuntimeError(
                    f"[RetrievalAttention] Invalid CSR graph at layer={ldx}, head={hdx}: "
                    f"offsets[-1]={int(graph_offsets[-1])}, neighbors={int(graph_neighbors.shape[0])}"
                )

        q_group = query_group.detach().float()
        if q_group.dim() == 1:
            q_group = q_group.unsqueeze(0)

        if self.query_mode == "group_avg":
            q_seed = q_group.mean(dim=0, keepdim=True)
        else:
            q_seed = q_group

        seed_start = time.perf_counter() if profile is not None else None
        q_seed_cpu = self._score_transform_torch(q_seed.cpu())
        seed_scores = {}
        seed_k = max(self.q_knn, self.q_knn * self.seed_k_mult)
        seed_k = min(self.input_length, seed_k)
        if seed_k <= 0:
            if profile is not None:
                profile["seed_sec"] += time.perf_counter() - seed_start
            return finish([], "seed_k_zero")

        static_indices = self.static_index_set
        if self.seed_mode == "faiss":
            if index is not None:
                q_np = q_seed_cpu.numpy().astype(np.float32)
                sim, idx = index.search(q_np, seed_k)
                for ridx in range(idx.shape[0]):
                    for cidx in range(idx.shape[1]):
                        tok = int(idx[ridx, cidx])
                        score = float(sim[ridx, cidx])
                        prev = seed_scores.get(tok)
                        if prev is None or score > prev:
                            seed_scores[tok] = score
            else:
                # Slow fallback path when no decode index is available.
                if not self._fallback_seed_warned:
                    print(
                        "[RetrievalAttention] WARNING: decode index missing; falling back to brute-force seed search. "
                        "Set RETRIEVALATTN_DECODE_INDEX=faiss for stable quality."
                    )
                    self._fallback_seed_warned = True
                k = self.cpu_keys[ldx][hdx, :self.input_length, :].detach().float().cpu()
                k = self._score_transform_torch(k)
                scores = torch.matmul(q_seed_cpu, k.transpose(0, 1))  # [num_q, num_tokens]
                k_take = min(seed_k, scores.shape[1])
                if k_take <= 0:
                    if profile is not None:
                        profile["seed_sec"] += time.perf_counter() - seed_start
                    return finish([], "seed_k_zero")
                vals, idx = torch.topk(scores, k=k_take, dim=1)
                for ridx in range(idx.shape[0]):
                    row_idx = idx[ridx]
                    row_vals = vals[ridx]
                    for cidx in range(row_idx.shape[0]):
                        tok = int(row_idx[cidx].item())
                        score = float(row_vals[cidx].item())
                        prev = seed_scores.get(tok)
                        if prev is None or score > prev:
                            seed_scores[tok] = score
        else:
            seed_candidates = []
            prev_tokens = self.prev_decode_seeds[ldx][hdx]
            if prev_tokens:
                seed_candidates.extend(prev_tokens[:self.seed_prev_k])
            hub_tokens = self.hub_seeds[ldx][hdx]
            if hub_tokens and self.seed_hub_k > 0:
                seed_candidates.extend(hub_tokens[:self.seed_hub_k])

            if self.seed_tail_k > 0 and self.dynamic_end > self.dynamic_start:
                span = self.dynamic_end - self.dynamic_start
                step = max(1, span // self.seed_tail_k)
                tok = self.dynamic_end - 1
                added = 0
                while tok >= self.dynamic_start and added < self.seed_tail_k:
                    seed_candidates.append(int(tok))
                    tok -= step
                    added += 1

            filtered = []
            filtered_seen = set()
            for tok in seed_candidates:
                tok = int(tok)
                if tok < 0 or tok >= self.input_length:
                    continue
                if tok in static_indices or tok in filtered_seen:
                    continue
                filtered_seen.add(tok)
                filtered.append(tok)

            if not filtered and self.dynamic_end > self.dynamic_start:
                fallback_take = max(8, min(seed_k, 32))
                span = self.dynamic_end - self.dynamic_start
                step = max(1, span // fallback_take)
                tok = self.dynamic_start
                while tok < self.dynamic_end and len(filtered) < fallback_take:
                    filtered.append(int(tok))
                    tok += step

            if filtered:
                idx = torch.tensor(filtered, dtype=torch.long, device="cpu")
                seed_kv = torch.index_select(self.cpu_keys[ldx][hdx], 0, idx).detach().float().cpu()
                seed_kv = self._score_transform_torch(seed_kv)
                seed_sim = torch.matmul(q_seed_cpu, seed_kv.transpose(0, 1))
                if self.rerank_agg == "mean":
                    seed_agg = seed_sim.mean(dim=0)
                else:
                    seed_agg = seed_sim.max(dim=0).values
                take = min(seed_k, int(seed_agg.shape[0]))
                if take > 0:
                    vals, pos = torch.topk(seed_agg, k=take, dim=0)
                    for i in range(take):
                        tok = int(filtered[int(pos[i].item())])
                        seed_scores[tok] = float(vals[i].item())
        if profile is not None:
            profile["seed_sec"] += time.perf_counter() - seed_start

        # Ranked seed list (highest retrieval score first), excluding static tokens.
        seed_ranked = []
        for tok, score in seed_scores.items():
            if tok in static_indices:
                continue
            seed_ranked.append((tok, score))
        seed_ranked.sort(key=lambda x: x[1], reverse=True)
        if not seed_ranked:
            return finish([], "empty_seed_ranked")

        # Reserve a minimum fraction of final budget for seed tokens.
        seed_floor = int(math.ceil(self.token_budget * self.seed_ratio))
        seed_floor = min(self.token_budget, max(0, seed_floor))
        if seed_floor > len(seed_ranked):
            seed_floor = len(seed_ranked)
        selected_seeds = [tok for tok, _ in seed_ranked[:seed_floor]]
        selected_seed_set = set(selected_seeds)

        candidate_target = self.token_budget * self.candidate_multiplier
        candidate_target = max(self.token_budget, candidate_target)
        candidate_target = min(candidate_target, self.input_length)

        candidates = []
        seen = set()
        candidate_scores = {}
        frontier = []
        frontier_best_by_node = {}
        expanded = set()
        visited_count = 0
        stability_steps = 0
        prev_topk_ids = None

        def push_frontier(tok: int, score: float):
            prev = frontier_best_by_node.get(tok)
            if prev is None or score > prev:
                frontier_best_by_node[tok] = score
                heapq.heappush(frontier, (-score, tok))

        def pop_frontier_batch(max_items: int):
            batch = []
            while frontier and len(batch) < max_items:
                neg_score, tok = heapq.heappop(frontier)
                score = -float(neg_score)
                best = frontier_best_by_node.get(tok)
                if best is None:
                    continue
                if score < (best - 1e-8):
                    continue
                if tok in expanded:
                    continue
                batch.append((tok, score))
            return batch

        def frontier_best_score() -> float:
            while frontier:
                neg_score, tok = frontier[0]
                score = -float(neg_score)
                best = frontier_best_by_node.get(tok)
                if best is None or score < (best - 1e-8) or tok in expanded:
                    heapq.heappop(frontier)
                    continue
                return score
            return -float("inf")

        # Prime candidates/frontier with seeds ranked by ANN score.
        for tok, score in seed_ranked:
            if len(candidates) >= candidate_target:
                break
            tok = int(tok)
            if tok not in seen:
                candidates.append(tok)
                seen.add(tok)
                score_f = float(score)
                candidate_scores[tok] = score_f
                push_frontier(tok, score_f)

        stop_reason = "graph_disabled"

        # Adaptive best-first graph expansion.
        graph_start = time.perf_counter() if profile is not None else None
        if self.graph_expand and graph is not None:
            stop_reason = "frontier_exhausted"
            while frontier and len(candidates) < candidate_target and visited_count < self.max_visits:
                batch = pop_frontier_batch(self.expand_width)
                if not batch:
                    break
                expand_nodes = [tok for tok, _score in batch]
                for tok in expand_nodes:
                    expanded.add(tok)
                    visited_count += 1
                    if visited_count >= self.max_visits:
                        break

                new_tokens = []
                new_token_set = set()
                for tok in expand_nodes:
                    if len(candidates) + len(new_tokens) >= candidate_target:
                        break
                    if graph_is_csr:
                        row_start = int(graph_offsets[tok])
                        row_end = int(graph_offsets[tok + 1])
                        nb_iter = graph_neighbors[row_start:row_end]
                    else:
                        nb_iter = graph[tok]
                    for nb in nb_iter:
                        nb = int(nb)
                        if nb in static_indices or nb in seen or nb in new_token_set:
                            continue
                        new_tokens.append(nb)
                        new_token_set.add(nb)
                        if len(candidates) + len(new_tokens) >= candidate_target:
                            break
                if new_tokens:
                    idx = torch.tensor(new_tokens, dtype=torch.long, device="cpu")
                    new_k = torch.index_select(self.cpu_keys[ldx][hdx], 0, idx).detach().float().cpu()
                    new_k = self._score_transform_torch(new_k)
                    new_scores = torch.matmul(q_seed_cpu, new_k.transpose(0, 1))
                    if self.rerank_agg == "mean":
                        agg_new = new_scores.mean(dim=0)
                    else:
                        agg_new = new_scores.max(dim=0).values

                    for i, tok in enumerate(new_tokens):
                        if len(candidates) >= candidate_target:
                            break
                        tok = int(tok)
                        if tok in seen:
                            continue
                        score = float(agg_new[i].item())
                        candidates.append(tok)
                        seen.add(tok)
                        candidate_scores[tok] = score
                        push_frontier(tok, score)

                if self.frontier_topn > 0 and len(frontier) > (self.frontier_topn * 4):
                    trimmed = []
                    for tok, score in frontier_best_by_node.items():
                        if tok in expanded:
                            continue
                        trimmed.append((-score, tok))
                    if len(trimmed) > self.frontier_topn:
                        trimmed.sort()
                        trimmed = trimmed[: self.frontier_topn]
                    frontier = trimmed
                    heapq.heapify(frontier)

                if len(candidates) >= candidate_target:
                    stop_reason = "candidate_cap"
                    break
                if visited_count >= self.max_visits:
                    stop_reason = "max_visits"
                    break

                if visited_count >= self.min_visits:
                    ranked_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
                    topk_items = ranked_items[: min(self.token_budget, len(ranked_items))]
                    current_topk_ids = tuple(tok for tok, _ in topk_items)
                    if prev_topk_ids is not None and current_topk_ids == prev_topk_ids:
                        stability_steps += 1
                    else:
                        stability_steps = 0
                    prev_topk_ids = current_topk_ids

                    if len(topk_items) >= self.token_budget:
                        kth_score = float(topk_items[-1][1])
                    else:
                        kth_score = -float("inf")
                    frontier_best = frontier_best_score()
                    if (
                        stability_steps >= self.stop_patience
                        and frontier_best <= (kth_score - self.stop_margin)
                    ):
                        stop_reason = "stability_gap"
                        break
            if not frontier and stop_reason == "frontier_exhausted":
                stop_reason = "frontier_empty"
        elif self.graph_expand and graph is None:
            stop_reason = "graph_missing"
        if profile is not None:
            profile["graph_sec"] += time.perf_counter() - graph_start

        if not candidates:
            return finish([], stop_reason, visited=visited_count, candidates=len(candidates))

        rerank_start = time.perf_counter() if profile is not None else None
        if self.rerank:
            idx = torch.tensor(candidates, dtype=torch.long, device="cpu")
            cand_k = torch.index_select(self.cpu_keys[ldx][hdx], 0, idx).detach().float().cpu()
            cand_k = self._score_transform_torch(cand_k)
            q_rank = self._score_transform_torch(q_group.detach().float().cpu())
            scores = torch.matmul(q_rank, cand_k.transpose(0, 1))
            if self.rerank_agg == "mean":
                agg_scores = scores.mean(dim=0)
            else:
                agg_scores = scores.max(dim=0).values
            order = torch.argsort(agg_scores, descending=True).cpu().tolist()
            ranked = [candidates[i] for i in order]
        else:
            ranked = sorted(candidates, key=lambda tok: candidate_scores.get(tok, -1e9), reverse=True)
        if profile is not None:
            profile["rerank_sec"] += time.perf_counter() - rerank_start

        # Enforce seed floor in final list.
        finalize_start = time.perf_counter() if profile is not None else None
        final = []
        final_set = set()
        if seed_floor > 0:
            for tok in ranked:
                if tok in selected_seed_set and tok not in final_set:
                    final.append(tok)
                    final_set.add(tok)
                    if len(final) >= seed_floor or len(final) >= self.token_budget:
                        break
        for tok in ranked:
            if tok in final_set:
                continue
            final.append(tok)
            final_set.add(tok)
            if len(final) >= self.token_budget:
                break

        if self.debug and self.decode_pos < self.debug_decode_steps and ldx == 0:
            expanded_cnt = sum(1 for tok in candidates if tok not in selected_seed_set)
            print(
                f"[RetrievalAttention][debug] step={self.decode_pos} layer={ldx} head={hdx} "
                f"seeds={len(seed_ranked)} seed_floor={seed_floor} expanded={expanded_cnt} "
                f"visited={visited_count} candidates={len(candidates)} "
                f"dynamic={len(final)} stop_reason={stop_reason}"
            )
        if final:
            self.prev_decode_seeds[ldx][hdx] = list(final[: self.seed_prev_k])
        else:
            self.prev_decode_seeds[ldx][hdx] = []
        if profile is not None:
            profile["finalize_sec"] += time.perf_counter() - finalize_start

        return finish(final, stop_reason, visited=visited_count, candidates=len(candidates))

    def compute(self, query_states, layer_idx):
        """
        Compute attention for a single decode step using retrieved tokens + static KV.
        query_states: [bs, 1, num_heads, head_dim]
        """
        compute_start = time.perf_counter() if self.decode_profile else None
        if not self._built:
            self.prepare_cache()

        bsz = query_states.shape[0]
        assert bsz == 1, "Only batch_size=1 supported for RetrievalAttention prototype."

        device = self.layer_mapping[str(layer_idx)]
        q = query_states[0, 0]  # [num_heads, head_dim]
        q = q.view(self.kv_head, self.group_size, self.head_dim)

        outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)
        empty_heads = 0
        dynamic_counts = []

        for hdx in range(self.kv_head):
            q_group = q[hdx]  # [group_size, head_dim]
            token_ids, retrieve_profile = self._retrieve_tokens(layer_idx, hdx, q_group)
            if self.decode_profile and retrieve_profile is not None:
                self._decode_profile_stats["retrieve_total_sec"] += float(retrieve_profile["total_sec"])
                self._decode_profile_stats["retrieve_seed_sec"] += float(retrieve_profile["seed_sec"])
                self._decode_profile_stats["retrieve_graph_sec"] += float(retrieve_profile["graph_sec"])
                self._decode_profile_stats["retrieve_rerank_sec"] += float(retrieve_profile["rerank_sec"])
                self._decode_profile_stats["retrieve_finalize_sec"] += float(retrieve_profile["finalize_sec"])
                self._decode_profile_stats["visited_total"] += int(retrieve_profile["visited"])
                self._decode_profile_stats["candidates_total"] += int(retrieve_profile["candidates"])
            if len(token_ids) == 0:
                empty_heads += 1
            dynamic_counts.append(len(token_ids))

            # Gather dynamic tokens from CPU
            gather_start = time.perf_counter() if self.decode_profile else None
            if token_ids:
                idx = torch.tensor(token_ids, dtype=torch.long, device="cpu")
                dyn_k = torch.index_select(self.cpu_keys[layer_idx][hdx], 0, idx).to(device, non_blocking=True)
                dyn_v = torch.index_select(self.cpu_values[layer_idx][hdx], 0, idx).to(device, non_blocking=True)
            else:
                dyn_k = None
                dyn_v = None
            if self.decode_profile:
                self._decode_profile_stats["gather_total_sec"] += (time.perf_counter() - gather_start)

            # Static KV on GPU
            static_k = self.static_gpu_keys[layer_idx][hdx]
            static_v = self.static_gpu_values[layer_idx][hdx]

            if dyn_k is not None:
                k = torch.cat([static_k, dyn_k], dim=0)
                v = torch.cat([static_v, dyn_v], dim=0)
            else:
                k = static_k
                v = static_v

            # Attention: q_group [G, D], k [T, D] -> [G, T]
            attn_start = time.perf_counter() if self.decode_profile else None
            scores = torch.matmul(q_group, k.transpose(0, 1)) * scale
            scores = scores.float()
            attn = torch.softmax(scores, dim=-1).to(v.dtype)
            out = torch.matmul(attn, v)  # [G, D]
            if self.decode_profile:
                self._decode_profile_stats["attn_total_sec"] += (time.perf_counter() - attn_start)
            outputs.append(out)
            if self.decode_profile:
                self._decode_profile_stats["heads"] += 1

        if self.assert_nonempty and empty_heads == self.kv_head:
            raise RuntimeError(
                f"[RetrievalAttention] Empty dynamic retrieval for all heads at decode step={self.decode_pos}, "
                f"layer={layer_idx}. Check decode seed index path."
            )
        if self.debug and self.decode_pos < self.debug_decode_steps and layer_idx == 0:
            avg_dyn = float(sum(dynamic_counts)) / float(len(dynamic_counts)) if dynamic_counts else 0.0
            print(
                f"[RetrievalAttention][debug] step={self.decode_pos} layer={layer_idx} "
                f"empty_heads={empty_heads}/{self.kv_head} avg_dynamic={avg_dyn:.1f}"
            )

        if self.decode_profile:
            self._decode_profile_stats["calls"] += 1
            self._decode_profile_stats["compute_total_sec"] += (time.perf_counter() - compute_start)

        out = torch.cat(outputs, dim=0).view(1, 1, self.num_heads, self.head_dim)
        return out
