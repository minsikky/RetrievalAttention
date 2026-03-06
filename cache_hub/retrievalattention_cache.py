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

from .roargraph_cpp_backend import (
    build_roar_graph_csr_cpp,
    search_roar_graph_csr_cpp,
    roargraph_cpp_available,
    roargraph_cpp_import_error,
)


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
        prefill_bsz: int,
        num_gpus: int,
        model_size: int,
    ) -> None:
        super().__init__(
            layer_num,
            batch_size,
            max_length,
            num_key_value_heads,
            num_heads,
            head_dim,
            dtype,
            layer_mapping,
            prefill_bsz,
            num_gpus,
            model_size,
        )
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
        try:
            self.parity_layers = int(os.environ.get("RETRIEVALATTN_PARITY_LAYERS", "1"))
        except Exception:
            self.parity_layers = 1
        self.parity_layers = max(1, self.parity_layers)
        try:
            self.parity_heads = int(os.environ.get("RETRIEVALATTN_PARITY_HEADS", "1"))
        except Exception:
            self.parity_heads = 1
        self.parity_heads = max(1, self.parity_heads)
        try:
            self.parity_sample = int(os.environ.get("RETRIEVALATTN_PARITY_SAMPLE", "256"))
        except Exception:
            self.parity_sample = 256
        self.parity_sample = max(1, self.parity_sample)
        self.traversal_eval = os.environ.get("RETRIEVALATTN_TRAVERSAL_EVAL", "0") == "1"
        try:
            self.traversal_eval_sample = int(os.environ.get("RETRIEVALATTN_TRAVERSAL_EVAL_SAMPLE", "64"))
        except Exception:
            self.traversal_eval_sample = 64
        self.traversal_eval_sample = max(1, self.traversal_eval_sample)
        try:
            self.graph_train_frac = float(os.environ.get("RETRIEVALATTN_GRAPH_TRAIN_FRAC", "1.0"))
        except Exception:
            self.graph_train_frac = 1.0
        if not np.isfinite(self.graph_train_frac):
            self.graph_train_frac = 1.0
        self.graph_train_frac = max(0.0, min(1.0, self.graph_train_frac))
        self.graph_split_mode = os.environ.get("RETRIEVALATTN_GRAPH_SPLIT", "stratified").strip().lower()
        if self.graph_split_mode not in {"contiguous", "random", "stratified"}:
            print(
                f"[RetrievalAttention] WARNING: unknown RETRIEVALATTN_GRAPH_SPLIT={self.graph_split_mode}. "
                "Falling back to stratified."
            )
            self.graph_split_mode = "stratified"
        try:
            self.graph_split_seed = int(os.environ.get("RETRIEVALATTN_GRAPH_SPLIT_SEED", "1234"))
        except Exception:
            self.graph_split_seed = 1234
        self.parity_holdout_only = os.environ.get("RETRIEVALATTN_PARITY_HOLDOUT_ONLY", "0") == "1"
        self.parity_query_indices = np.empty((0,), dtype=np.int32)
        self.parity_query_count = 0
        self._parity_query_indices_torch = None
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
        self.graph_builder = "roar"
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
        raw_roar_backend = os.environ.get("RETRIEVALATTN_ROAR_BACKEND", "cpp").strip().lower()
        if raw_roar_backend in {"cpp", "roar_cpp"}:
            self.roar_backend = "cpp"
        elif raw_roar_backend in {"python", "py"}:
            self.roar_backend = "python"
        elif raw_roar_backend in {"python_gpu", "py_gpu", "gpu_python"}:
            self.roar_backend = "python_gpu"
        else:
            print(
                f"[RetrievalAttention] WARNING: unknown RETRIEVALATTN_ROAR_BACKEND={raw_roar_backend}. "
                "Falling back to cpp."
            )
            self.roar_backend = "cpp"
        self.roar_python_gpu_enabled = (self.roar_backend == "python_gpu")
        self.roar_python_gpu_device = os.environ.get("RETRIEVALATTN_ROAR_PY_GPU_DEVICE", "cuda").strip()
        try:
            self.roar_python_gpu_batch = int(os.environ.get("RETRIEVALATTN_ROAR_PY_GPU_BATCH", "256"))
        except Exception:
            self.roar_python_gpu_batch = 256
        self.roar_python_gpu_batch = max(1, self.roar_python_gpu_batch)
        try:
            self.roar_cpp_threads = int(os.environ.get("RETRIEVALATTN_ROAR_CPP_THREADS", "0"))
        except Exception:
            self.roar_cpp_threads = 0
        self.roar_cpp_threads = max(0, self.roar_cpp_threads)
        self.decode_backend = "roar_cpp"
        try:
            self.roar_decode_lpq = int(os.environ.get("RETRIEVALATTN_ROAR_DECODE_LPQ", "0"))
        except Exception:
            self.roar_decode_lpq = 0
        self.roar_decode_lpq = max(0, self.roar_decode_lpq)
        try:
            self.roar_decode_init = int(os.environ.get("RETRIEVALATTN_ROAR_DECODE_INIT", "64"))
        except Exception:
            self.roar_decode_init = 64
        self.roar_decode_init = max(1, self.roar_decode_init)
        try:
            self.roar_decode_max_cmps = int(os.environ.get("RETRIEVALATTN_ROAR_DECODE_MAX_CMPS", "0"))
        except Exception:
            self.roar_decode_max_cmps = 0
        self.roar_decode_max_cmps = max(0, self.roar_decode_max_cmps)
        try:
            self.roar_decode_max_hops = int(os.environ.get("RETRIEVALATTN_ROAR_DECODE_MAX_HOPS", "0"))
        except Exception:
            self.roar_decode_max_hops = 0
        self.roar_decode_max_hops = max(0, self.roar_decode_max_hops)
        try:
            self.roar_decode_threads = int(os.environ.get("RETRIEVALATTN_ROAR_DECODE_THREADS", "0"))
        except Exception:
            self.roar_decode_threads = 0
        self.roar_decode_threads = max(0, self.roar_decode_threads)
        self._roar_cpp_available = roargraph_cpp_available()
        if self.roar_backend == "cpp" and not self._roar_cpp_available:
            cpp_err = roargraph_cpp_import_error()
            raise RuntimeError(
                "RoarGraph C++ extension is required but unavailable. "
                "Build it with: "
                "`module load python/3.10.4 && source .venv/bin/activate && "
                "python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`"
                + (f". Import error: {cpp_err}" if cpp_err is not None else "")
            )
        if self.roar_python_gpu_enabled:
            if not torch.cuda.is_available():
                print(
                    "[RetrievalAttention] WARNING: RETRIEVALATTN_ROAR_BACKEND=python_gpu requested but CUDA is unavailable. "
                    "Falling back to python backend."
                )
                self.roar_backend = "python"
                self.roar_python_gpu_enabled = False
            elif not self.roar_python_gpu_device.startswith("cuda"):
                print(
                    f"[RetrievalAttention] WARNING: RETRIEVALATTN_ROAR_PY_GPU_DEVICE={self.roar_python_gpu_device} "
                    "is not a CUDA device. Falling back to python backend."
                )
                self.roar_backend = "python"
                self.roar_python_gpu_enabled = False
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
        self.fa_fused_prefill = True
        raw_fused_prefill = os.environ.get("RETRIEVALATTN_FA_FUSED_PREFILL", "1").strip()
        if raw_fused_prefill not in {"", "1"}:
            raise RuntimeError(
                "Only fused prefill mode is supported. "
                "RETRIEVALATTN_FA_FUSED_PREFILL must be 1."
            )
        self.fa_graph_fused = os.environ.get("RETRIEVALATTN_FA_GRAPH_FUSED", "0") == "1"
        self.fa_graph_fused_require = os.environ.get("RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE", "0") == "1"
        self.fa_graph_fused_check = os.environ.get("RETRIEVALATTN_FA_GRAPH_FUSED_CHECK", "1") == "1"
        self.fa_graph_debug = os.environ.get("RETRIEVALATTN_FA_GRAPH_DEBUG", "0") == "1"
        try:
            self.fa_graph_fused_quality_floor = float(
                os.environ.get("RETRIEVALATTN_FA_GRAPH_FUSED_QUALITY_FLOOR", "0.90")
            )
        except Exception:
            self.fa_graph_fused_quality_floor = 0.90
        self.fa_graph_fused_quality_floor = max(0.0, min(1.0, self.fa_graph_fused_quality_floor))
        raw_shadow_compare = os.environ.get("RETRIEVALATTN_FA_SHADOW_COMPARE", "0").strip()
        if raw_shadow_compare in {"1", "true", "True"}:
            print(
                "[RetrievalAttention] WARNING: RETRIEVALATTN_FA_SHADOW_COMPARE is deprecated "
                "and ignored in fused-only runtime.",
                flush=True,
            )
        self.fa_shadow_compare = False
        self.fa_shadow_sample = 0
        self.retrieval_head_mode = "q_head"
        self.retrieval_heads = self.num_heads
        self.kv_graph_ab = os.environ.get("RETRIEVALATTN_KV_GRAPH_AB", "0") == "1"
        try:
            self.kv_graph_ab_q_block = int(os.environ.get("RETRIEVALATTN_KV_GRAPH_AB_Q_BLOCK", "512"))
        except Exception:
            self.kv_graph_ab_q_block = 512
        self.kv_graph_ab_q_block = max(1, self.kv_graph_ab_q_block)
        try:
            self.kv_graph_ab_k_block = int(os.environ.get("RETRIEVALATTN_KV_GRAPH_AB_K_BLOCK", "4096"))
        except Exception:
            self.kv_graph_ab_k_block = 4096
        self.kv_graph_ab_k_block = max(1, self.kv_graph_ab_k_block)
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
        )
        self._store_prefill_queries = bool(self.validate_parity)
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
            "search_space_total": 0,
            "search_space_heads": 0,
            "visited_ratio_sum": 0.0,
            "visited_ratio_count": 0,
        }
        self._parity_records = []
        self._parity_lock = threading.Lock()
        self.graph_train_query_indices, self.graph_holdout_query_indices = self._build_graph_query_split_indices(
            total=self.input_length,
            train_frac=self.graph_train_frac,
        )

        if self.validate_parity:
            parity_candidates = None
            if self.parity_holdout_only and self.graph_holdout_query_indices.size > 0:
                parity_candidates = self.graph_holdout_query_indices
            self.parity_query_indices = self._build_parity_sample_indices(
                sample_n=self.parity_sample,
                total=self.input_length,
                causal_ref=True,
                candidate_indices=parity_candidates,
            )
            self.parity_query_count = int(self.parity_query_indices.shape[0])
            if self.parity_query_count > 0:
                self._parity_query_indices_torch = torch.from_numpy(
                    self.parity_query_indices.astype(np.int64, copy=False)
                )

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

        # Optional sampled per-Q-head query storage for parity checks.
        self.cpu_queries_qhead_samples = [None for _ in range(self.layer_num)]
        self.cpu_queries_qhead_full = [None for _ in range(self.layer_num)]
        if self.validate_parity and self.parity_query_count > 0:
            parity_layers_take = min(self.layer_num, self.parity_layers)
            for ldx in range(parity_layers_take):
                self.cpu_queries_qhead_samples[ldx] = torch.empty(
                    (self.num_heads, self.parity_query_count, self.head_dim),
                    dtype=self.dtype,
                    pin_memory=True,
                )
                if self.kv_graph_ab:
                    self.cpu_queries_qhead_full[ldx] = torch.empty(
                        (self.num_heads, self.input_length, self.head_dim),
                        dtype=self.dtype,
                        pin_memory=True,
                    )

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

        # Decode seed indexes are keyed by KV-head (shared keys in GQA).
        self.indexes = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        # K-token graphs are shared per KV head (not per q-head).
        self.graphs = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self.hub_seeds = [[[] for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        # Warm-start seeds remain retrieval-head keyed (per q-head in q_head mode).
        self.prev_decode_seeds = [[[] for _ in range(self.retrieval_heads)] for _ in range(self.layer_num)]
        self._decode_key_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cpp_warned = False
        self.fused_prefill_knn = [None for _ in range(self.layer_num)]
        self.fused_prefill_profiles = [None for _ in range(self.layer_num)]
        self.fused_prefill_graph_neighbors = [None for _ in range(self.layer_num)]
        self.fused_prefill_graph_profiles = [None for _ in range(self.layer_num)]
        self._fused_prefill_executor = None
        self._fused_prefill_futures = {}
        self._fused_prefill_submitted = [False for _ in range(self.layer_num)]
        self._fused_prefill_done = [False for _ in range(self.layer_num)]
        self._fused_prefill_errors = {}
        self._fused_prefill_submit_count = 0
        self._fused_prefill_done_count = 0
        self._fused_prefill_lock = threading.Lock()
        self._faiss_threads_async_configured = False
        self._kv_graph_ab_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]

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

    def _retrieval_head_to_kv_head(self, head_idx: int) -> int:
        head_idx = int(head_idx)
        if self.retrieval_head_mode == "q_head":
            if head_idx < 0 or head_idx >= self.num_heads:
                raise IndexError(
                    f"Invalid retrieval q-head index={head_idx} (num_heads={self.num_heads})"
                )
            return int(head_idx // self.group_size)
        if head_idx < 0 or head_idx >= self.kv_head:
            raise IndexError(
                f"Invalid retrieval kv-head index={head_idx} (kv_head={self.kv_head})"
            )
        return head_idx

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

    def _should_run_parity_for(self, ldx: int, hdx: int) -> bool:
        if not self.validate_parity or faiss is None:
            return False
        if ldx < 0 or hdx < 0:
            return False
        if ldx >= self.parity_layers:
            return False
        if hdx >= self.parity_heads:
            return False
        return True

    def _record_parity(
        self,
        layer_idx: int,
        head_idx: int,
        sample_n: int,
        recall: float,
        k: int,
        traversal: dict = None,
        extras: dict = None,
    ):
        record = {
            "layer": int(layer_idx),
            "head": int(head_idx),
            "sample": int(sample_n),
            "k": int(k),
            "recall": float(recall),
        }
        if isinstance(traversal, dict) and traversal:
            record.update({
                "trav_samples": int(traversal.get("samples", 0)),
                "trav_recall": float(traversal.get("recall", 0.0)),
                "trav_recall_cov": float(traversal.get("recall_cov", 0.0)),
                "trav_visited_mean": float(traversal.get("visited_mean", 0.0)),
                "trav_visit_rate": float(traversal.get("visit_rate", 0.0)),
                "trav_prune_rate": float(traversal.get("prune_rate", 0.0)),
                "trav_cand_per_visit": float(traversal.get("cand_per_visit", 0.0)),
            })
        if isinstance(extras, dict) and extras:
            record.update(extras)
        with self._parity_lock:
            self._parity_records.append(record)

    def _evaluate_traversal_efficiency(
        self,
        ldx: int,
        hdx: int,
        kv_hdx: int,
        queries_np: np.ndarray,
        ref_knn_np: np.ndarray,
        graph,
        decode_index,
    ):
        if queries_np is None or ref_knn_np is None:
            return None
        queries = np.ascontiguousarray(queries_np, dtype=np.float32)
        ref_knn = np.ascontiguousarray(ref_knn_np, dtype=np.int32)
        if queries.ndim != 2 or ref_knn.ndim != 2:
            return None
        n = min(int(queries.shape[0]), int(ref_knn.shape[0]))
        if n <= 0:
            return None

        eval_n = min(n, int(self.traversal_eval_sample))
        if eval_n < n:
            take = np.linspace(0, n - 1, num=eval_n, dtype=np.int64)
            queries = np.ascontiguousarray(queries[take, :], dtype=np.float32)
            ref_knn = np.ascontiguousarray(ref_knn[take, :], dtype=np.int32)
        else:
            eval_n = n

        prev_index = self.indexes[ldx][kv_hdx]
        prev_graph = self.graphs[ldx][kv_hdx]
        if decode_index is not None:
            self.indexes[ldx][kv_hdx] = decode_index
        if graph is not None:
            self.graphs[ldx][kv_hdx] = graph

        recalls = []
        recalls_cov = []
        visited_vals = []
        visit_rates = []
        cand_per_visit_vals = []
        k = max(1, int(self.q_knn))
        try:
            for ridx in range(eval_n):
                q_t = torch.from_numpy(queries[ridx])
                tokens, prof = self._retrieve_tokens(
                    ldx,
                    hdx,
                    q_t,
                    update_decode_state=False,
                    enforce_seed_floor=False,
                )
                prof = prof if isinstance(prof, dict) else {}
                visited = max(0, int(prof.get("visited", 0)))
                candidates = max(0, int(prof.get("candidates", 0)))
                search_space = max(0, int(prof.get("search_space", 0)))

                gt = set(int(x) for x in ref_knn[ridx, :k].tolist())
                retrieved_topk = set(int(x) for x in tokens[:k])
                retrieved_cov = set(int(x) for x in tokens)
                rec = (len(gt.intersection(retrieved_topk)) / float(k)) if k > 0 else 0.0
                rec_cov = (len(gt.intersection(retrieved_cov)) / float(k)) if k > 0 else 0.0
                recalls.append(float(rec))
                recalls_cov.append(float(rec_cov))
                visited_vals.append(float(visited))
                if search_space > 0:
                    visit_rates.append(float(visited) / float(search_space))
                if visited > 0:
                    cand_per_visit_vals.append(float(candidates) / float(visited))
        finally:
            self.indexes[ldx][kv_hdx] = prev_index
            self.graphs[ldx][kv_hdx] = prev_graph

        if len(recalls) == 0:
            return None
        visit_rate = float(np.mean(visit_rates)) if visit_rates else 0.0
        return {
            "samples": int(len(recalls)),
            "recall": float(np.mean(recalls)),
            "recall_cov": float(np.mean(recalls_cov)) if recalls_cov else 0.0,
            "visited_mean": float(np.mean(visited_vals)) if visited_vals else 0.0,
            "visit_rate": visit_rate,
            "prune_rate": max(0.0, 1.0 - visit_rate),
            "cand_per_visit": float(np.mean(cand_per_visit_vals)) if cand_per_visit_vals else 0.0,
        }

    def _build_graph_query_split_indices(self, total: int, train_frac: float):
        total = int(total)
        if total <= 0:
            empty = np.empty((0,), dtype=np.int32)
            return empty, empty

        frac = float(train_frac)
        if frac >= 1.0:
            train = np.arange(total, dtype=np.int32)
            holdout = np.empty((0,), dtype=np.int32)
            return train, holdout
        # Keep at least one query row for graph construction to avoid empty-graph runs.
        n_train = int(round(float(total) * max(0.0, frac)))
        n_train = max(1, n_train)
        if total > 1 and frac < 1.0:
            n_train = min(total - 1, n_train)
        else:
            n_train = min(total, n_train)

        if self.graph_split_mode == "contiguous":
            train = np.arange(0, n_train, dtype=np.int32)
            holdout = np.arange(n_train, total, dtype=np.int32)
            return train, holdout

        rng = np.random.default_rng(self.graph_split_seed)
        if self.graph_split_mode == "random":
            perm = rng.permutation(total).astype(np.int32, copy=False)
            train = np.sort(perm[:n_train]).astype(np.int32, copy=False)
            holdout = np.sort(perm[n_train:]).astype(np.int32, copy=False)
            return np.ascontiguousarray(train), np.ascontiguousarray(holdout)

        # Stratified split: sample train rows across the full sequence to avoid
        # contiguous-prefix bias under causal references.
        if n_train >= total:
            train = np.arange(total, dtype=np.int32)
            holdout = np.empty((0,), dtype=np.int32)
            return train, holdout

        bin_edges = np.linspace(0, total, num=n_train + 1, dtype=np.float64)
        picks = []
        for i in range(n_train):
            lo = int(bin_edges[i])
            hi = int(bin_edges[i + 1])
            if hi <= lo:
                hi = min(total, lo + 1)
            if hi <= lo:
                lo = max(0, min(total - 1, lo))
                hi = lo + 1
            if (hi - lo) <= 1:
                picks.append(lo)
            else:
                picks.append(lo + int(rng.integers(0, hi - lo)))

        train = np.unique(np.asarray(picks, dtype=np.int32))
        if train.shape[0] < n_train:
            all_idx = np.arange(total, dtype=np.int32)
            remaining = np.setdiff1d(all_idx, train, assume_unique=True)
            need = int(n_train - train.shape[0])
            if remaining.shape[0] <= need:
                extra = remaining
            else:
                extra = np.sort(rng.choice(remaining, size=need, replace=False).astype(np.int32, copy=False))
            train = np.sort(np.concatenate([train, extra]).astype(np.int32, copy=False))

        all_idx = np.arange(total, dtype=np.int32)
        holdout = np.setdiff1d(all_idx, train, assume_unique=True).astype(np.int32, copy=False)
        return np.ascontiguousarray(train), np.ascontiguousarray(holdout)

    def _build_parity_sample_indices(
        self,
        sample_n: int,
        total: int,
        causal_ref: bool,
        candidate_indices: np.ndarray = None,
    ) -> np.ndarray:
        total = int(total)
        sample_n = int(sample_n)
        if total <= 0 or sample_n <= 0:
            return np.empty((0,), dtype=np.int32)
        start = int(self.q_knn - 1) if causal_ref else 0
        if start >= total:
            start = 0
        if candidate_indices is None:
            candidates = np.arange(start, total, dtype=np.int32)
        else:
            candidates = np.asarray(candidate_indices, dtype=np.int32)
            if candidates.size == 0:
                return np.empty((0,), dtype=np.int32)
            candidates = candidates[(candidates >= start) & (candidates < total)]
            if candidates.size == 0:
                return np.empty((0,), dtype=np.int32)
            candidates = np.unique(candidates)
        if candidates.size <= sample_n:
            return candidates
        take = np.linspace(0, candidates.size - 1, num=sample_n, dtype=np.int64)
        return np.ascontiguousarray(candidates[take], dtype=np.int32)

    def _select_graph_knn_rows(self, knn: np.ndarray) -> np.ndarray:
        """
        Select training query rows used for graph projection.
        Supports both:
          [input_length, q_knn]
          [input_length * qh_merged, q_knn] (merged q-head rows).
        """
        if knn.ndim != 2 or knn.shape[0] <= 0:
            return knn

        train_idx = self.graph_train_query_indices
        if train_idx is None or train_idx.size == 0:
            return knn

        row_count = int(knn.shape[0])
        if row_count == int(self.input_length):
            if train_idx.size == row_count:
                return knn
            return np.ascontiguousarray(knn[train_idx, :], dtype=np.int32)

        if row_count % int(self.input_length) != 0:
            return knn

        qh_merged = row_count // int(self.input_length)
        if train_idx.size == int(self.input_length):
            return knn
        knn_3d = knn.reshape(int(self.input_length), int(qh_merged), int(knn.shape[1]))
        knn_train = knn_3d[train_idx, :, :].reshape(-1, int(knn.shape[1]))
        return np.ascontiguousarray(knn_train, dtype=np.int32)

    def _causal_topk_ref_np(self, queries: np.ndarray, keys: np.ndarray, query_indices: np.ndarray, k: int) -> np.ndarray:
        """
        Exact causal top-k reference for sampled query rows.
        For each sampled query index qi, only keys [0, qi] are valid.
        """
        if queries.size == 0 or keys.size == 0 or query_indices.size == 0:
            return np.empty((0, max(1, int(k))), dtype=np.int32)
        k = max(1, int(k))
        scores = np.matmul(queries.astype(np.float32, copy=False), keys.astype(np.float32, copy=False).T)
        out = np.empty((queries.shape[0], k), dtype=np.int32)
        for ridx in range(queries.shape[0]):
            qidx = int(query_indices[ridx])
            k_limit = min(int(scores.shape[1]), qidx + 1)
            if k_limit <= 0:
                out[ridx, :] = 0
                continue
            row = scores[ridx, :k_limit]
            take_k = min(k, k_limit)
            if take_k == k_limit:
                top_local = np.arange(k_limit, dtype=np.int32)
            else:
                top_local = np.argpartition(row, -take_k)[-take_k:].astype(np.int32, copy=False)
            if take_k < k:
                # Should not happen for our sampling policy; keep shape-stable anyway.
                padded = np.empty((k,), dtype=np.int32)
                padded[:take_k] = top_local
                padded[take_k:] = top_local[-1]
                out[ridx, :] = padded
            else:
                out[ridx, :] = top_local
        return out

    def _decode_dynamic_topk_ref_np(self, queries: np.ndarray, keys: np.ndarray, k: int) -> np.ndarray:
        """
        Exact decode-style top-k reference over the dynamic key range only.
        This matches retrieval candidate space used by `_retrieve_tokens`.
        """
        k = max(1, int(k))
        if queries.size == 0 or keys.size == 0:
            return np.empty((0, k), dtype=np.int32)

        d_start = int(self.dynamic_start)
        d_end = int(self.dynamic_end)
        if d_end <= d_start:
            return np.empty((queries.shape[0], k), dtype=np.int32)

        keys_dyn = np.ascontiguousarray(keys[d_start:d_end, :], dtype=np.float32)
        if keys_dyn.shape[0] <= 0:
            return np.empty((queries.shape[0], k), dtype=np.int32)

        q = np.ascontiguousarray(queries.astype(np.float32, copy=False))
        scores = np.matmul(q, keys_dyn.T)
        take_k = min(k, int(keys_dyn.shape[0]))
        if take_k <= 0:
            return np.empty((queries.shape[0], k), dtype=np.int32)

        idx_local = np.argpartition(scores, -take_k, axis=1)[:, -take_k:]
        idx_abs = idx_local.astype(np.int32, copy=False) + np.int32(d_start)
        if take_k == k:
            return np.ascontiguousarray(idx_abs, dtype=np.int32)

        # Pad when dynamic span is smaller than k.
        out = np.empty((queries.shape[0], k), dtype=np.int32)
        out[:, :take_k] = idx_abs
        pad_val = idx_abs[:, -1:] if take_k > 0 else np.int32(max(0, d_start))
        out[:, take_k:] = pad_val
        return out

    def _exact_causal_topk_torch(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        query_indices: np.ndarray,
        k: int,
        device: str,
    ) -> np.ndarray:
        """
        Exact causal top-k via blocked torch matmul on GPU.
        Used only for offline graph A/B diagnostics.
        """
        if queries.size == 0 or keys.size == 0 or query_indices.size == 0:
            return np.empty((0, max(1, int(k))), dtype=np.int32)

        k = max(1, int(k))
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        keys = np.ascontiguousarray(keys, dtype=np.float32)
        query_indices = np.ascontiguousarray(query_indices, dtype=np.int64)

        q_block = int(self.kv_graph_ab_q_block)
        k_block = int(self.kv_graph_ab_k_block)
        num_q = int(queries.shape[0])
        num_k = int(keys.shape[0])

        q_t = torch.from_numpy(queries).to(device=device, dtype=torch.float32, non_blocking=True)
        k_t = torch.from_numpy(keys).to(device=device, dtype=torch.float32, non_blocking=True)
        qi_t = torch.from_numpy(query_indices).to(device=device, dtype=torch.int64, non_blocking=True)

        out_idx = torch.empty((num_q, k), dtype=torch.int64, device=device)
        neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=device)

        with torch.no_grad():
            for q_start in range(0, num_q, q_block):
                q_end = min(num_q, q_start + q_block)
                q_chunk = q_t[q_start:q_end]
                qi_chunk = qi_t[q_start:q_end]
                top_scores = torch.full((q_chunk.shape[0], k), float("-inf"), dtype=torch.float32, device=device)
                top_indices = torch.full((q_chunk.shape[0], k), -1, dtype=torch.int64, device=device)

                for k_start in range(0, num_k, k_block):
                    k_end = min(num_k, k_start + k_block)
                    k_chunk = k_t[k_start:k_end]
                    scores = torch.matmul(q_chunk, k_chunk.transpose(0, 1))

                    key_idx = torch.arange(k_start, k_end, device=device, dtype=torch.int64)
                    causal_mask = key_idx.unsqueeze(0) <= qi_chunk.unsqueeze(1)
                    scores = torch.where(causal_mask, scores, neg_inf)

                    take_k = min(k, int(k_end - k_start))
                    vals, idx = torch.topk(scores, k=take_k, dim=1, sorted=False)
                    idx = idx + k_start

                    merged_scores = torch.cat([top_scores, vals], dim=1)
                    merged_idx = torch.cat([top_indices, idx], dim=1)
                    new_vals, new_pos = torch.topk(merged_scores, k=k, dim=1, sorted=False)
                    top_scores = new_vals
                    top_indices = torch.gather(merged_idx, 1, new_pos)

                out_idx[q_start:q_end] = top_indices

        return np.ascontiguousarray(out_idx.cpu().numpy().astype(np.int32, copy=False))

    def _get_kv_graph_ab_graph(self, ldx: int, kv_hdx: int, keys_cpu: np.ndarray):
        if not self.kv_graph_ab:
            return None
        cached = self._kv_graph_ab_cache[ldx][kv_hdx]
        if cached is not None:
            return cached

        qfull_buf = None
        if ldx < len(self.cpu_queries_qhead_full):
            qfull_buf = self.cpu_queries_qhead_full[ldx]
        if qfull_buf is None:
            return None

        qh_start = int(kv_hdx * self.group_size)
        qh_end = min(int(qh_start + self.group_size), int(self.num_heads))
        if qh_start >= qh_end:
            return None

        q_group = (
            qfull_buf[qh_start:qh_end, :self.input_length, :]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if q_group.ndim != 3 or q_group.shape[0] <= 0:
            return None

        kv_queries = self._score_transform_np(np.mean(q_group, axis=0))
        query_indices = np.arange(self.input_length, dtype=np.int64)
        knn_full = self._exact_causal_topk_torch(
            queries=kv_queries,
            keys=keys_cpu,
            query_indices=query_indices,
            k=self.q_knn,
            device=self.layer_mapping[str(ldx)],
        )
        knn_graph = self._select_graph_knn_rows(knn_full)
        graph, meta = self._build_graph_csr_from_knn(knn_graph, keys_cpu=keys_cpu)
        cached = (graph, meta)
        self._kv_graph_ab_cache[ldx][kv_hdx] = cached
        if self._profile_enabled():
            print(
                f"[RetrievalAttention] kv_graph_ab built layer={ldx} kv_head={kv_hdx} "
                f"queries={kv_queries.shape[0]} train_queries={knn_graph.shape[0]} "
                f"builder={meta.get('builder', 'n/a')} "
                f"edges={int(graph[0][-1]) if isinstance(graph, tuple) and len(graph) >= 2 and graph[0].shape[0] > 0 else 0}",
                flush=True,
            )
        return cached

    def get_parity_summary(self, reset: bool = False) -> dict:
        with self._parity_lock:
            records = list(self._parity_records)
            if reset:
                self._parity_records.clear()
        if not records:
            return {
                "enabled": bool(self.validate_parity),
                "layers_limit": int(self.parity_layers),
                "heads_limit": int(self.parity_heads),
                "sample_limit": int(self.parity_sample),
                "parity_holdout_only": bool(self.parity_holdout_only),
                "graph_train_frac": float(self.graph_train_frac),
                "graph_split_mode": str(self.graph_split_mode),
                "graph_split_seed": int(self.graph_split_seed),
                "graph_train_queries": int(self.graph_train_query_indices.size),
                "graph_holdout_queries": int(self.graph_holdout_query_indices.size),
                "records": 0,
                "total_sample": 0,
                "recall_mean": None,
                "recall_weighted": None,
                "recall_min": None,
                "recall_max": None,
                "kv_proxy": None,
                "kv_proxy_traversal": None,
                "kv_graph_traversal": None,
                "traversal": None,
                "details": [],
            }

        recalls = [float(x["recall"]) for x in records]
        samples = [max(0, int(x["sample"])) for x in records]
        total_sample = int(sum(samples))
        if total_sample > 0:
            weighted = float(
                sum(float(recalls[i]) * float(samples[i]) for i in range(len(records)))
                / float(total_sample)
            )
        else:
            weighted = float(np.mean(recalls))
        kv_proxy_records = [r for r in records if r.get("kv_proxy_recall") is not None]
        kv_proxy_summary = None
        if kv_proxy_records:
            kv_proxy_samples = [max(0, int(r.get("sample", 0))) for r in kv_proxy_records]
            kv_proxy_total = int(sum(kv_proxy_samples))
            kv_proxy_values = [float(r.get("kv_proxy_recall", 0.0)) for r in kv_proxy_records]
            if kv_proxy_total > 0:
                kv_proxy_weighted = float(
                    sum(
                        float(kv_proxy_values[i]) * float(kv_proxy_samples[i])
                        for i in range(len(kv_proxy_records))
                    )
                    / float(kv_proxy_total)
                )
            else:
                kv_proxy_weighted = float(np.mean(kv_proxy_values))
            kv_proxy_summary = {
                "records": int(len(kv_proxy_records)),
                "total_sample": kv_proxy_total,
                "recall_mean": float(np.mean(kv_proxy_values)),
                "recall_weighted": float(kv_proxy_weighted),
                "recall_min": float(np.min(kv_proxy_values)),
                "recall_max": float(np.max(kv_proxy_values)),
            }
        kv_proxy_trav_records = [r for r in records if int(r.get("kv_proxy_trav_samples", 0)) > 0]
        kv_proxy_trav_summary = None
        if kv_proxy_trav_records:
            kv_proxy_trav_samples = [max(0, int(r.get("kv_proxy_trav_samples", 0))) for r in kv_proxy_trav_records]
            kv_proxy_trav_total = int(sum(kv_proxy_trav_samples))
            if kv_proxy_trav_total > 0:
                def _kv_proxy_trav_wavg(key):
                    return float(
                        sum(
                            float(kv_proxy_trav_records[i].get(key, 0.0)) * float(kv_proxy_trav_samples[i])
                            for i in range(len(kv_proxy_trav_records))
                        )
                        / float(kv_proxy_trav_total)
                    )
                kv_proxy_trav_summary = {
                    "records": int(len(kv_proxy_trav_records)),
                    "total_sample": kv_proxy_trav_total,
                    "recall_mean": _kv_proxy_trav_wavg("kv_proxy_trav_recall"),
                    "recall_cov_mean": _kv_proxy_trav_wavg("kv_proxy_trav_recall_cov"),
                    "visited_mean": _kv_proxy_trav_wavg("kv_proxy_trav_visited_mean"),
                    "visit_rate_mean": _kv_proxy_trav_wavg("kv_proxy_trav_visit_rate"),
                    "prune_rate_mean": _kv_proxy_trav_wavg("kv_proxy_trav_prune_rate"),
                    "cand_per_visit_mean": _kv_proxy_trav_wavg("kv_proxy_trav_cand_per_visit"),
                }
        kv_graph_trav_records = [r for r in records if int(r.get("kv_graph_trav_samples", 0)) > 0]
        kv_graph_trav_summary = None
        if kv_graph_trav_records:
            kv_graph_trav_samples = [max(0, int(r.get("kv_graph_trav_samples", 0))) for r in kv_graph_trav_records]
            kv_graph_trav_total = int(sum(kv_graph_trav_samples))
            if kv_graph_trav_total > 0:
                def _kv_graph_trav_wavg(key):
                    return float(
                        sum(
                            float(kv_graph_trav_records[i].get(key, 0.0)) * float(kv_graph_trav_samples[i])
                            for i in range(len(kv_graph_trav_records))
                        )
                        / float(kv_graph_trav_total)
                    )
                kv_graph_trav_summary = {
                    "records": int(len(kv_graph_trav_records)),
                    "total_sample": kv_graph_trav_total,
                    "recall_mean": _kv_graph_trav_wavg("kv_graph_trav_recall"),
                    "recall_cov_mean": _kv_graph_trav_wavg("kv_graph_trav_recall_cov"),
                    "visited_mean": _kv_graph_trav_wavg("kv_graph_trav_visited_mean"),
                    "visit_rate_mean": _kv_graph_trav_wavg("kv_graph_trav_visit_rate"),
                    "prune_rate_mean": _kv_graph_trav_wavg("kv_graph_trav_prune_rate"),
                    "cand_per_visit_mean": _kv_graph_trav_wavg("kv_graph_trav_cand_per_visit"),
                }
        trav_records = [r for r in records if int(r.get("trav_samples", 0)) > 0]
        traversal_summary = None
        if trav_records:
            trav_samples = [max(0, int(r.get("trav_samples", 0))) for r in trav_records]
            trav_total = int(sum(trav_samples))
            if trav_total > 0:
                def _wavg(key):
                    return float(
                        sum(float(trav_records[i].get(key, 0.0)) * float(trav_samples[i]) for i in range(len(trav_records)))
                        / float(trav_total)
                    )
                traversal_summary = {
                    "records": int(len(trav_records)),
                    "total_sample": trav_total,
                    "recall_mean": _wavg("trav_recall"),
                    "recall_cov_mean": _wavg("trav_recall_cov"),
                    "visited_mean": _wavg("trav_visited_mean"),
                    "visit_rate_mean": _wavg("trav_visit_rate"),
                    "prune_rate_mean": _wavg("trav_prune_rate"),
                    "cand_per_visit_mean": _wavg("trav_cand_per_visit"),
                }
        return {
            "enabled": bool(self.validate_parity),
            "layers_limit": int(self.parity_layers),
            "heads_limit": int(self.parity_heads),
            "sample_limit": int(self.parity_sample),
            "parity_holdout_only": bool(self.parity_holdout_only),
            "graph_train_frac": float(self.graph_train_frac),
            "graph_split_mode": str(self.graph_split_mode),
            "graph_split_seed": int(self.graph_split_seed),
            "graph_train_queries": int(self.graph_train_query_indices.size),
            "graph_holdout_queries": int(self.graph_holdout_query_indices.size),
            "records": int(len(records)),
            "total_sample": int(total_sample),
            "recall_mean": float(np.mean(recalls)),
            "recall_weighted": float(weighted),
            "recall_min": float(np.min(recalls)),
            "recall_max": float(np.max(recalls)),
            "kv_proxy": kv_proxy_summary,
            "kv_proxy_traversal": kv_proxy_trav_summary,
            "kv_graph_traversal": kv_graph_trav_summary,
            "traversal": traversal_summary,
            "details": records,
        }

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
        return self._build_graph_csr_from_knn_roar(knn, keys_cpu)

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

    def _prepare_roar_python_gpu_keys(self, keys_cpu: np.ndarray):
        if not self.roar_python_gpu_enabled:
            return None, 0.0
        if not torch.cuda.is_available():
            return None, 0.0
        start = time.time()
        try:
            keys_gpu = torch.as_tensor(
                np.ascontiguousarray(keys_cpu, dtype=np.float32),
                dtype=torch.float32,
                device=self.roar_python_gpu_device,
            )
            return keys_gpu, float(time.time() - start)
        except Exception as exc:
            if self.roar_log:
                print(
                    "[RetrievalAttention] WARNING: python_gpu Roar key upload failed; "
                    f"{type(exc).__name__}: {exc}. Falling back to python CPU path."
                )
            return None, 0.0

    def _acquire_neighbors_roar_batch(
        self,
        xs,
        candidates_batch,
        keys_cpu: np.ndarray,
        degree_cap: int,
        keys_gpu: torch.Tensor = None,
    ):
        if len(xs) == 0:
            return []
        if keys_gpu is None:
            return [
                self._acquire_neighbors_roar(int(x), candidates_batch[i], keys_cpu, degree_cap, keys_gpu=None)
                for i, x in enumerate(xs)
            ]

        cleaned = []
        max_cand = 0
        for i, x in enumerate(xs):
            cand = self._dedup_dynamic_tokens(candidates_batch[i], exclude=int(x), max_take=0)
            cleaned.append(cand)
            if len(cand) > max_cand:
                max_cand = len(cand)
        if max_cand <= 0:
            return [[] for _ in xs]

        batch_n = len(xs)
        idx = torch.zeros((batch_n, max_cand), dtype=torch.long, device=keys_gpu.device)
        mask = torch.zeros((batch_n, max_cand), dtype=torch.bool, device=keys_gpu.device)
        for i, cand in enumerate(cleaned):
            if not cand:
                continue
            cand_t = torch.as_tensor(cand, dtype=torch.long, device=keys_gpu.device)
            take = int(cand_t.numel())
            idx[i, :take] = cand_t
            mask[i, :take] = True

        cand_vecs = keys_gpu.index_select(0, idx.reshape(-1)).reshape(batch_n, max_cand, -1)
        x_idx = torch.as_tensor([int(x) for x in xs], dtype=torch.long, device=keys_gpu.device)
        x_vecs = keys_gpu.index_select(0, x_idx)
        sim_xc = torch.einsum("bcd,bd->bc", cand_vecs, x_vecs)
        sim_xc = sim_xc.masked_fill(~mask, float("-inf"))

        sim_np = sim_xc.detach().cpu().numpy()

        out = []
        for i, cand in enumerate(cleaned):
            if not cand:
                out.append([])
                continue
            cand_arr = np.asarray(cand, dtype=np.int32)
            csz = cand_arr.shape[0]
            scores = sim_np[i, :csz]
            order = np.argsort(-scores)
            sorted_cand = cand_arr[order]
            sorted_sim = scores[order]

            selected = [int(sorted_cand[0])]
            selected_set = {int(sorted_cand[0])}

            for j in range(1, sorted_cand.shape[0]):
                if len(selected) >= degree_cap:
                    break
                c_tok = int(sorted_cand[j])
                if c_tok in selected_set:
                    continue
                sx = float(sorted_sim[j])
                sel_arr = np.asarray(selected, dtype=np.int32)
                cp_sim = np.matmul(keys_cpu[sel_arr], keys_cpu[c_tok])
                if np.all(sx > cp_sim):
                    selected.append(c_tok)
                    selected_set.add(c_tok)

            if len(selected) < degree_cap:
                for tok in sorted_cand.tolist():
                    tok = int(tok)
                    if tok in selected_set:
                        continue
                    selected.append(tok)
                    selected_set.add(tok)
                    if len(selected) >= degree_cap:
                        break
            out.append(selected[:degree_cap])
        return out

    def _acquire_neighbors_roar(
        self,
        x: int,
        candidates,
        keys_cpu: np.ndarray,
        degree_cap: int,
        keys_gpu: torch.Tensor = None,
    ):
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

        cand_arr = np.asarray(cand, dtype=np.int32)
        if keys_gpu is not None:
            cand_t = torch.as_tensor(cand_arr, dtype=torch.long, device=keys_gpu.device)
            cand_vecs_t = keys_gpu.index_select(0, cand_t)
            x_vec_t = keys_gpu[int(x)]
            sim_xc = torch.matmul(cand_vecs_t, x_vec_t).detach().cpu().numpy()
            gram = torch.matmul(cand_vecs_t, cand_vecs_t.transpose(0, 1)).detach().cpu().numpy()
        else:
            x_idx = int(x)
            x_vec = keys_cpu[x_idx]
            cand_vecs = keys_cpu[cand_arr]
            sim_xc = np.matmul(cand_vecs, x_vec)
            gram = np.matmul(cand_vecs, cand_vecs.T)

        order = np.argsort(-sim_xc)
        sorted_cand = cand_arr[order]
        sorted_sim = sim_xc[order]

        selected = []
        selected_pos = []
        selected_set = set()

        first = int(sorted_cand[0])
        selected.append(first)
        selected_pos.append(int(order[0]))
        selected_set.add(first)

        for idx in range(1, sorted_cand.shape[0]):
            if len(selected) >= degree_cap:
                break
            c = int(sorted_cand[idx])
            sx = float(sorted_sim[idx])
            if c in selected_set:
                continue
            c_pos = int(order[idx])
            cp_sim = gram[c_pos, np.asarray(selected_pos, dtype=np.int32)]
            if np.all(sx > cp_sim):
                selected.append(c)
                selected_set.add(c)
                selected_pos.append(c_pos)

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

    def _beam_search_roar(
        self,
        source_node: int,
        adjacency: dict,
        keys_cpu: np.ndarray,
        entry_node: int,
        beam_l: int,
        keys_gpu: torch.Tensor = None,
    ):
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
        src_vec_gpu = keys_gpu[src] if keys_gpu is not None else None
        frontier = []
        best_score = {}
        visited = set()
        candidates = []

        def push_many(nodes):
            uniq = []
            seen_local = set()
            for node in nodes:
                node = int(node)
                if node in seen_local:
                    continue
                seen_local.add(node)
                if node < self.dynamic_start or node >= self.dynamic_end:
                    continue
                uniq.append(node)
            if not uniq:
                return
            if keys_gpu is None:
                for node in uniq:
                    score = float(np.dot(src_vec, keys_cpu[node]))
                    prev = best_score.get(node)
                    if prev is None or score > prev:
                        best_score[node] = score
                        heapq.heappush(frontier, (-score, node))
                return

            node_t = torch.as_tensor(uniq, dtype=torch.long, device=keys_gpu.device)
            scores = torch.matmul(keys_gpu.index_select(0, node_t), src_vec_gpu).detach().cpu().numpy()
            for i, node in enumerate(uniq):
                score = float(scores[i])
                prev = best_score.get(node)
                if prev is None or score > prev:
                    best_score[node] = score
                    heapq.heappush(frontier, (-score, node))

        def push(node: int):
            push_many([int(node)])

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

            nbrs = []
            for nb in adjacency.get(node, []):
                nb = int(nb)
                if nb in visited:
                    continue
                nbrs.append(nb)
            push_many(nbrs)

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

    def _graph_from_dense_neighbors(self, dense_neighbors: np.ndarray):
        """
        Convert fixed-degree dense neighbor table [seq, m] int32 into CSR graph.
        """
        num_tokens = self.input_length
        empty_graph = self._empty_graph_csr()
        if dense_neighbors is None:
            return empty_graph, {"builder": "flashattn_fused_graph", "stop_reason": "none"}

        arr = np.asarray(dense_neighbors)
        if arr.ndim != 2:
            raise RuntimeError(
                f"fused graph neighbors must be rank-2 [seq,m], got shape={tuple(arr.shape)}"
            )
        if int(arr.shape[0]) != int(num_tokens):
            raise RuntimeError(
                "fused graph neighbors seq mismatch: "
                f"got {int(arr.shape[0])}, expected {int(num_tokens)}"
            )
        degree_cap = max(1, int(self.roar_m))
        ds = int(self.dynamic_start)
        de = int(self.dynamic_end)
        if de <= ds:
            return empty_graph, {"builder": "flashattn_fused_graph", "stop_reason": "empty"}

        # Dynamic region only: [dynamic_tokens, m]
        rows = np.asarray(arr[ds:de], dtype=np.int32)
        if rows.size == 0:
            return empty_graph, {"builder": "flashattn_fused_graph", "stop_reason": "empty"}

        nrows, m = rows.shape
        row_ids = np.arange(ds, de, dtype=np.int32).reshape(-1, 1)

        # Keep only dynamic, non-self neighbors.
        valid = (rows >= ds) & (rows < de) & (rows != row_ids)
        if not np.any(valid):
            return empty_graph, {"builder": "flashattn_fused_graph", "stop_reason": "empty"}

        # Dedup per row while preserving first-seen order across columns.
        keep = valid.copy()
        for c in range(1, m):
            tok_c = rows[:, c]
            dup_prev = np.zeros((nrows,), dtype=bool)
            for p in range(c):
                dup_prev |= valid[:, p] & (tok_c == rows[:, p])
            keep[:, c] &= ~dup_prev

        # Enforce per-row degree cap (first degree_cap kept entries).
        if degree_cap < m:
            rank = np.cumsum(keep, axis=1)
            keep &= rank <= degree_cap

        dyn_counts = keep.sum(axis=1, dtype=np.uint64)
        if np.sum(dyn_counts, dtype=np.uint64) == 0:
            return empty_graph, {"builder": "flashattn_fused_graph", "stop_reason": "empty"}

        row_counts = np.zeros(num_tokens, dtype=np.uint64)
        row_counts[ds:de] = dyn_counts

        offsets64 = np.empty((num_tokens + 1,), dtype=np.uint64)
        offsets64[0] = 0
        np.cumsum(row_counts, out=offsets64[1:])
        total_edges = int(offsets64[-1])
        if total_edges > np.iinfo(np.uint32).max:
            raise RuntimeError(
                f"[RetrievalAttention] CSR offsets exceed uint32 range: edges={total_edges}"
            )
        offsets = offsets64.astype(np.uint32, copy=False)
        neighbors = rows[keep].astype(np.int32, copy=False)
        if int(neighbors.shape[0]) != total_edges:
            raise RuntimeError(
                "fused graph CSR mismatch: "
                f"neighbors={int(neighbors.shape[0])}, offsets[-1]={total_edges}"
            )

        active_rows = int(np.count_nonzero(dyn_counts))

        meta = {
            "builder": "flashattn_fused_graph",
            "backend": "gpu_dense",
            "stop_reason": "ok",
            "projected_nodes": active_rows,
            "active_pivots": active_rows,
            "active_queries": int(self.input_length),
            "enhanced_nodes": 0,
            "bipartite_sec": 0.0,
            "projection_sec": 0.0,
            "enhance_sec": 0.0,
            "csr_sec": 0.0,
            "total_sec": 0.0,
        }
        if self.graph_return_weights:
            weights = np.ones((total_edges,), dtype=self.graph_weight_dtype)
            return (offsets, neighbors, weights), meta
        return (offsets, neighbors), meta

    def _should_use_roar_cpp_backend(self) -> bool:
        available = roargraph_cpp_available()
        self._roar_cpp_available = bool(available)
        if not available:
            cpp_err = roargraph_cpp_import_error()
            raise RuntimeError(
                "RoarGraph C++ extension is required but unavailable. "
                "Build it with: "
                "`module load python/3.10.4 && source .venv/bin/activate && "
                "python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`"
                + (f". Import error: {cpp_err}" if cpp_err is not None else "")
            )
        return True

    def _should_use_roar_cpp_decode_backend(self, graph_is_csr: bool) -> bool:
        if not graph_is_csr:
            return False
        available = roargraph_cpp_available()
        self._roar_cpp_available = bool(available)
        if not available:
            cpp_err = roargraph_cpp_import_error()
            raise RuntimeError(
                "RoarGraph C++ extension is required but unavailable. "
                "Build it with: "
                "`module load python/3.10.4 && source .venv/bin/activate && "
                "python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`"
                + (f". Import error: {cpp_err}" if cpp_err is not None else "")
            )
        return True

    def _get_decode_key_array(self, ldx: int, hdx: int):
        cached = self._decode_key_cache[ldx][hdx]
        if cached is not None:
            return cached["array"], cached["dtype"]

        key_tensor = self.cpu_keys[ldx][hdx, :self.input_length, :]
        if not key_tensor.is_contiguous():
            key_tensor = key_tensor.contiguous()

        key_dtype = "fp32"
        key_view_tensor = None
        if key_tensor.dtype == torch.float32:
            key_array = key_tensor.numpy()
            key_dtype = "fp32"
        elif key_tensor.dtype == torch.float16:
            key_array = key_tensor.numpy()
            key_dtype = "fp16"
        elif key_tensor.dtype == torch.bfloat16:
            key_view_tensor = key_tensor.view(torch.uint16)
            key_array = key_view_tensor.numpy()
            key_dtype = "bf16"
        else:
            key_tensor = key_tensor.float().contiguous()
            key_array = key_tensor.numpy()
            key_dtype = "fp32"

        key_array = np.ascontiguousarray(key_array)
        self._decode_key_cache[ldx][hdx] = {
            "tensor": key_tensor,
            "view_tensor": key_view_tensor,
            "array": key_array,
            "dtype": key_dtype,
        }
        return key_array, key_dtype

    def _build_graph_csr_from_knn_roar_cpp(self, knn: np.ndarray, keys_cpu: np.ndarray):
        meta = {
            "builder": "roar_cpp",
            "backend": "cpp",
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
            raise RuntimeError("Roar graph builder (cpp) requires keys_cpu but got None.")

        num_tokens = self.input_length
        keys = np.ascontiguousarray(keys_cpu, dtype=np.float32)
        if keys.shape[0] < num_tokens:
            raise RuntimeError(
                f"Roar graph builder keys_cpu too short: got {keys.shape[0]}, need {num_tokens}"
            )
        knn_arr = np.ascontiguousarray(knn, dtype=np.int32)

        nq = min(max(1, int(self.roar_nq)), int(knn_arr.shape[1]))
        degree_cap = max(1, int(self.roar_m))
        cand_limit = max(1, int(self.roar_l))
        enhance_limit = max(1, int(self.roar_enhance_l))

        offsets, neighbors, cpp_meta = build_roar_graph_csr_cpp(
            knn_arr,
            keys,
            dynamic_start=self.dynamic_start,
            dynamic_end=self.dynamic_end,
            nq=nq,
            degree_cap=degree_cap,
            cand_limit=cand_limit,
            enable_enhance=bool(self.roar_enable_enhance),
            enhance_limit=enhance_limit,
            entry_mode=self.roar_entry,
            max_query_per_pivot=int(self.roar_max_query_per_pivot),
            num_threads=int(self.roar_cpp_threads),
        )

        offsets = np.ascontiguousarray(offsets, dtype=np.uint32)
        neighbors = np.ascontiguousarray(neighbors, dtype=np.int32)
        if offsets.shape[0] != (num_tokens + 1):
            raise RuntimeError(
                f"Invalid roar cpp offsets length: got {offsets.shape[0]}, expected {num_tokens + 1}"
            )
        if int(offsets[-1]) != int(neighbors.shape[0]):
            raise RuntimeError(
                f"Invalid roar cpp CSR sizes: offsets[-1]={int(offsets[-1])}, neighbors={int(neighbors.shape[0])}"
            )

        if isinstance(cpp_meta, dict):
            meta.update(cpp_meta)
        meta["builder"] = str(meta.get("builder", "roar_cpp"))
        meta["backend"] = "cpp"
        meta["total_sec"] = float(meta.get("total_sec", time.time() - total_start))

        if self.graph_return_weights:
            weights = np.ones((neighbors.shape[0],), dtype=self.graph_weight_dtype)
            return (offsets, neighbors, weights), meta
        return (offsets, neighbors), meta

    def _build_graph_csr_from_knn_roar(self, knn: np.ndarray, keys_cpu: np.ndarray):
        if self.roar_backend == "cpp":
            self._should_use_roar_cpp_backend()
            return self._build_graph_csr_from_knn_roar_cpp(knn, keys_cpu)
        return self._build_graph_csr_from_knn_roar_python(knn, keys_cpu)

    def _build_graph_csr_from_knn_roar_python(self, knn: np.ndarray, keys_cpu: np.ndarray):
        """
        RoarGraph-like construction:
          1) query->base links + base->query bridge
          2) neighborhood-aware projection with reverse-edge updates
          3) connectivity enhancement (beam search + reverse-edge updates)
        """
        meta = {
            "builder": "roar",
            "backend": "python_gpu" if self.roar_python_gpu_enabled else "python",
            "bipartite_sec": 0.0,
            "projection_sec": 0.0,
            "enhance_sec": 0.0,
            "csr_sec": 0.0,
            "total_sec": 0.0,
            "active_queries": 0,
            "active_pivots": 0,
            "projected_nodes": 0,
            "enhanced_nodes": 0,
            "gpu_key_upload_sec": 0.0,
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
        keys_gpu, key_upload_sec = self._prepare_roar_python_gpu_keys(keys)
        meta["gpu_key_upload_sec"] = float(key_upload_sec)
        if keys_gpu is None:
            meta["backend"] = "python"

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
        projection_rows = []
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
            projection_rows.append((int(x), candidates))

        if keys_gpu is not None:
            batch = max(1, int(self.roar_python_gpu_batch))
            for start in range(0, len(projection_rows), batch):
                chunk = projection_rows[start : start + batch]
                xs = [int(row[0]) for row in chunk]
                cand_batch = [row[1] for row in chunk]
                neighbors_batch = self._acquire_neighbors_roar_batch(
                    xs,
                    cand_batch,
                    keys,
                    degree_cap,
                    keys_gpu=keys_gpu,
                )
                for i, x in enumerate(xs):
                    x_neighbors = neighbors_batch[i]
                    if not x_neighbors:
                        continue
                    projected[int(x)] = x_neighbors

                    # Reverse-edge maintenance during projection.
                    for p in x_neighbors:
                        p = int(p)
                        p_cands = list(projected.get(p, []))
                        if int(x) not in p_cands:
                            p_cands.append(int(x))
                        # Reverse-edge maintenance is highly irregular and
                        # candidate sets are tiny. CPU path is faster than
                        # launching/syncing many tiny GPU kernels here.
                        p_neighbors = self._acquire_neighbors_roar(
                            p,
                            p_cands,
                            keys,
                            degree_cap,
                            keys_gpu=None,
                        )
                        if p_neighbors:
                            projected[p] = p_neighbors
                        elif p in projected:
                            del projected[p]
        else:
            for x, candidates in projection_rows:
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
                    keys_gpu=None,
                )
                if not beam_candidates:
                    continue

                x_prime = self._acquire_neighbors_roar(
                    x,
                    beam_candidates,
                    keys,
                    degree_cap,
                    keys_gpu=None,
                )
                if not x_prime:
                    continue
                nprime[x] = x_prime
                enhanced_nodes += 1

                for p in x_prime:
                    p = int(p)
                    p_cands = list(nprime.get(p, []))
                    if x not in p_cands:
                        p_cands.append(x)
                    p_prime = self._acquire_neighbors_roar(
                        p,
                        p_cands,
                        keys,
                        degree_cap,
                        keys_gpu=None,
                    )
                    if p_prime:
                        nprime[p] = p_prime
                    elif p in nprime:
                        del nprime[p]

            # Merge enhancement into projected graph and re-enforce degree cap.
            for node, nprime_neighbors in nprime.items():
                node = int(node)
                merged = list(projected.get(node, []))
                merged.extend(nprime_neighbors)
                merged_neighbors = self._acquire_neighbors_roar(
                    node,
                    merged,
                    keys,
                    degree_cap,
                    keys_gpu=None,
                )
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

    def uses_flashattn_fused_graph_prefill(self) -> bool:
        return bool(self.fa_fused_prefill and self.fa_graph_fused)

    def register_fused_prefill_knn(
        self,
        layer_idx: int,
        knn_idx,
        profile: dict = None,
        graph_neighbors=None,
        graph_profile: dict = None,
    ):
        """
        Register fused-prefill top-k indices produced during prefill attention.
        Accepted shapes:
          [1, seq, retrieval_heads, q_knn]
          [seq, retrieval_heads, q_knn]
          [retrieval_heads, seq, q_knn]
        Stored format:
          [seq, retrieval_heads, q_knn] int32 contiguous (on CPU).

        Optional graph_neighbors accepted shapes:
          [1, kv_head, seq, graph_degree]
          [kv_head, seq, graph_degree]
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

        if arr.shape[0] == self.input_length:
            norm = arr
        elif arr.shape[1] == self.input_length:
            norm = np.transpose(arr, (1, 0, 2))
        else:
            raise RuntimeError(
                "fused prefill top-k shape does not match expected layout "
                f"(input_length={self.input_length}), got {tuple(arr.shape)}"
            )

        in_heads = int(norm.shape[1])
        if in_heads != self.retrieval_heads:
            raise RuntimeError(
                "fused prefill top-k head dimension mismatch: "
                f"input_heads={in_heads}, expected_retrieval_heads={self.retrieval_heads}, "
                f"num_heads={self.num_heads}, kv_head={self.kv_head}, "
                f"head_mode={self.retrieval_head_mode}, shape={tuple(arr.shape)}"
            )

        if norm.shape[2] < self.q_knn:
            raise RuntimeError(
                f"fused prefill top-k last dim too small: got {norm.shape[2]}, need >= {self.q_knn}"
            )
        if norm.shape[2] > self.q_knn:
            norm = norm[:, :, :self.q_knn]

        norm = np.ascontiguousarray(norm.astype(np.int32, copy=False))
        profile_dict = profile if isinstance(profile, dict) else {}
        graph_profile_dict = graph_profile if isinstance(graph_profile, dict) else {}

        graph_norm = None
        if graph_neighbors is not None:
            if isinstance(graph_neighbors, torch.Tensor):
                g_arr = graph_neighbors.detach().to("cpu", non_blocking=False).numpy()
            else:
                g_arr = np.asarray(graph_neighbors)
            if g_arr.ndim == 4:
                if g_arr.shape[0] != 1:
                    raise RuntimeError(
                        f"fused prefill graph neighbors expects batch dim 1, got shape={tuple(g_arr.shape)}"
                    )
                g_arr = g_arr[0]
            if g_arr.ndim != 3:
                raise RuntimeError(
                    "fused prefill graph neighbors must be rank-3 or rank-4, "
                    f"got shape={tuple(g_arr.shape)}"
                )
            if int(g_arr.shape[0]) != int(self.kv_head) or int(g_arr.shape[1]) != int(self.input_length):
                raise RuntimeError(
                    "fused prefill graph neighbors shape mismatch: "
                    f"got {tuple(g_arr.shape)}, expected [kv_head={self.kv_head}, seq={self.input_length}, m]"
                )
            graph_norm = np.ascontiguousarray(g_arr.astype(np.int32, copy=False))

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
                self.fused_prefill_graph_profiles[ldx] = graph_profile_dict
            if self._profile_enabled():
                print(
                    f"[RetrievalAttention] fused_overlap submit layer={ldx} "
                    f"workers={self.fused_prefill_overlap_workers}",
                    flush=True,
                )
            try:
                future = executor.submit(
                    self._finalize_fused_layer,
                    ldx,
                    norm,
                    profile_dict,
                    graph_norm,
                    graph_profile_dict,
                )
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
        self.fused_prefill_graph_neighbors[ldx] = graph_norm
        self.fused_prefill_graph_profiles[ldx] = graph_profile_dict

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

    def _finalize_fused_layer(
        self,
        ldx: int,
        layer_knn: np.ndarray,
        profile: dict = None,
        layer_graph_neighbors: np.ndarray = None,
        graph_profile: dict = None,
    ):
        """
        Finalize one fused-prefill layer on CPU: decode index + graph + hub seeds.
        Runs in overlap worker thread when enabled.
        """
        layer_start = time.time()
        prof = profile if isinstance(profile, dict) else {}
        graph_prof = graph_profile if isinstance(graph_profile, dict) else {}
        self._check_fused_score_mode_compat(prof)
        per_kv_topk = None
        layer_topk = prof.get("topk_sec", prof.get("fused_sec"))
        if layer_topk is not None:
            try:
                layer_topk = float(layer_topk)
                per_kv_topk = layer_topk / float(self.kv_head)
            except Exception:
                per_kv_topk = None

        if self._profile_enabled() and prof:
            print(
                f"[RetrievalAttention] fused_overlap profile layer={ldx}: {prof}",
                flush=True,
            )

        try:
            # Build one shared K-token graph per KV head by merging grouped q-head knn rows.
            for kv_hdx in range(self.kv_head):
                qh_start = int(kv_hdx * self.group_size)
                qh_end = min(int(qh_start + self.group_size), int(self.num_heads))
                if qh_start >= qh_end:
                    continue
                head_start = time.time()
                knn_group = np.ascontiguousarray(layer_knn[:, qh_start:qh_end, :], dtype=np.int32)
                knn = np.ascontiguousarray(knn_group.reshape(-1, knn_group.shape[-1]), dtype=np.int32)
                head_prof = dict(prof) if isinstance(prof, dict) else {}
                if per_kv_topk is not None:
                    head_prof["topk_sec"] = per_kv_topk
                graph_override = None
                graph_override_meta = None
                if layer_graph_neighbors is not None:
                    graph_dense = np.ascontiguousarray(layer_graph_neighbors[kv_hdx], dtype=np.int32)
                    if self.fa_graph_debug:
                        dyn_s = int(self.dynamic_start)
                        dyn_e = int(self.dynamic_end)
                        piv = knn[:, 0] if knn.shape[1] > 0 else np.empty((0,), dtype=np.int32)
                        cand = knn[:, 1:] if knn.shape[1] > 1 else np.empty((knn.shape[0], 0), dtype=np.int32)
                        piv_dyn = float(np.mean((piv >= dyn_s) & (piv < dyn_e))) if piv.size > 0 else 0.0
                        cand_dyn = float(np.mean((cand >= dyn_s) & (cand < dyn_e))) if cand.size > 0 else 0.0
                        g_valid = (graph_dense >= dyn_s) & (graph_dense < dyn_e)
                        g_nonempty_rows = int(np.count_nonzero(np.any(g_valid, axis=1)))
                        g_valid_ratio = float(np.mean(g_valid)) if g_valid.size > 0 else 0.0
                        print(
                            "[RetrievalAttention] graph_debug "
                            f"layer={ldx} kv_head={kv_hdx} "
                            f"knn_pivot_dyn={piv_dyn:.4f} knn_cand_dyn={cand_dyn:.4f} "
                            f"graph_rows_nonempty={g_nonempty_rows} graph_valid_ratio={g_valid_ratio:.4f}",
                            flush=True,
                        )
                    graph_override, graph_override_meta = self._graph_from_dense_neighbors(graph_dense)
                    if isinstance(graph_override_meta, dict):
                        graph_override_meta = dict(graph_override_meta)
                        graph_override_meta.update(
                            {
                                "graph_sec": float(graph_prof.get("graph_sec", 0.0)),
                                "graph_nq": int(graph_prof.get("graph_nq", self.roar_nq)),
                                "graph_degree": int(graph_prof.get("graph_degree", self.roar_m)),
                            }
                        )
                result = self._finalize_gpu_head_build(
                    ldx,
                    kv_hdx,
                    knn,
                    head_prof,
                    head_start,
                    kv_hdx_override=kv_hdx,
                    graph_hdx_override=kv_hdx,
                    parity_hdx_override=qh_start,
                    graph_override=graph_override,
                    graph_override_meta=graph_override_meta,
                )
                self._commit_head_build_result(result)

            with self._fused_prefill_lock:
                self._fused_prefill_done[ldx] = True
                self._fused_prefill_done_count += 1
                self.fused_prefill_knn[ldx] = None
                self.fused_prefill_profiles[ldx] = None
                self.fused_prefill_graph_neighbors[ldx] = None
                self.fused_prefill_graph_profiles[ldx] = None
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
            layer_graph_neighbors = None

    def _finalize_gpu_head_build(
        self,
        ldx: int,
        hdx: int,
        knn: np.ndarray,
        prof: dict,
        head_start: float,
        kv_hdx_override: int = None,
        graph_hdx_override: int = None,
        parity_hdx_override: int = None,
        graph_override=None,
        graph_override_meta: dict = None,
    ):
        """
        CPU-side postprocess for one GPU-topk head:
        decode seed index + optional parity + CSR graph projection.
        """
        kv_hdx = int(kv_hdx_override) if kv_hdx_override is not None else self._retrieval_head_to_kv_head(hdx)
        graph_hdx = int(graph_hdx_override) if graph_hdx_override is not None else kv_hdx
        parity_hdx = int(parity_hdx_override) if parity_hdx_override is not None else hdx
        graph_override_meta = graph_override_meta if isinstance(graph_override_meta, dict) else {}
        knn_for_parity = knn
        if int(knn.shape[0]) != int(self.input_length):
            # Fused q_head graph build may merge multiple q-head rows per KV head.
            # For parity, use the first q-head slice from the merged block.
            if (
                self.retrieval_head_mode == "q_head"
                and int(self.input_length) > 0
                and knn.ndim == 2
                and (int(knn.shape[0]) % int(self.input_length)) == 0
            ):
                qh_merged = int(knn.shape[0]) // int(self.input_length)
                if qh_merged > 0:
                    knn_for_parity = np.ascontiguousarray(
                        knn.reshape(self.input_length, qh_merged, knn.shape[-1])[:, 0, :],
                        dtype=np.int32,
                    )
        decode_index = None
        keys_cpu = None
        need_graph_keys = (self.graph_builder == "roar") and (graph_override is None)
        run_parity = (
            int(knn_for_parity.shape[0]) == int(self.input_length)
            and self._should_run_parity_for(ldx, parity_hdx)
        )
        need_keys_cpu = (
            need_graph_keys
            or
            self.decode_index_mode == "faiss"
            or run_parity
        )
        if need_keys_cpu:
            keys_cpu = (
                self.cpu_keys[ldx][kv_hdx, :self.input_length, :]
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
            if self.indexes[ldx][kv_hdx] is None:
                decode_index = faiss.IndexFlatIP(self.head_dim)
                decode_index.add(keys_cpu)

        parity_msg = None
        parity_record = None
        traversal_queries = None
        traversal_ref_knn = None
        if run_parity:
            qhead_buf_precheck = None
            if ldx < len(self.cpu_queries_qhead_samples):
                qhead_buf_precheck = self.cpu_queries_qhead_samples[ldx]
            if self.cpu_queries is None and qhead_buf_precheck is None:
                raise RuntimeError(
                    "Parity validation requested but prefill queries are unavailable."
                )
            if self.parity_query_indices is not None and self.parity_query_indices.size > 0:
                sample_idx_all = np.ascontiguousarray(self.parity_query_indices, dtype=np.int32)
            else:
                sample_n_req = min(self.parity_sample, self.input_length, int(knn_for_parity.shape[0]))
                sample_idx_all = self._build_parity_sample_indices(
                    sample_n=sample_n_req,
                    total=int(knn_for_parity.shape[0]),
                    causal_ref=True,
                )
            valid_mask = sample_idx_all < int(knn_for_parity.shape[0])
            sample_idx = np.ascontiguousarray(sample_idx_all[valid_mask], dtype=np.int32)
            sample_pos = np.ascontiguousarray(np.nonzero(valid_mask)[0], dtype=np.int32)
            profile_path = str(prof.get("path", "")).strip().lower()
            use_causal_ref = bool(prof.get("retrieval_causal", False))
            if not use_causal_ref:
                use_causal_ref = profile_path.startswith("native_kernel_fused")
            if sample_idx.size > 0:
                sample_n = int(sample_idx.shape[0])
                knn_sample = np.ascontiguousarray(knn_for_parity[sample_idx, :self.q_knn], dtype=np.int32)
                per_q_head_recalls = []
                q_head_refs = []
                kv_proxy_recalls = []
                qhead_buf = None
                if ldx < len(self.cpu_queries_qhead_samples):
                    qhead_buf = self.cpu_queries_qhead_samples[ldx]
                if qhead_buf is not None:
                    if self.retrieval_head_mode == "q_head":
                        qh_start = int(parity_hdx)
                        qh_end = min(int(parity_hdx + 1), int(self.num_heads))
                    else:
                        qh_start = int(kv_hdx * self.group_size)
                        qh_end = min(int(qh_start + self.group_size), int(self.num_heads))
                    q_group = (
                        qhead_buf[qh_start:qh_end, sample_pos, :]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    if q_group.ndim == 3 and q_group.shape[0] > 0:
                        if not use_causal_ref:
                            ref_index = faiss.IndexFlatIP(self.head_dim)
                            ref_index.add(keys_cpu)
                        for qhi in range(q_group.shape[0]):
                            q_queries = self._score_transform_np(q_group[qhi])
                            trav_ref_knn = self._decode_dynamic_topk_ref_np(
                                queries=q_queries,
                                keys=keys_cpu,
                                k=self.q_knn,
                            )
                            if use_causal_ref:
                                ref_knn = self._causal_topk_ref_np(
                                    queries=q_queries,
                                    keys=keys_cpu,
                                    query_indices=sample_idx,
                                    k=self.q_knn,
                                )
                            else:
                                _, ref_knn = ref_index.search(q_queries, self.q_knn)
                            rec_qh = self._knn_recall_at_k(knn_sample, ref_knn, self.q_knn)
                            per_q_head_recalls.append(float(rec_qh))
                            q_head_refs.append(np.ascontiguousarray(ref_knn, dtype=np.int32))
                            if traversal_queries is None and trav_ref_knn.shape[0] == q_queries.shape[0]:
                                traversal_queries = q_queries
                                traversal_ref_knn = trav_ref_knn

                        if self.retrieval_head_mode == "q_head":
                            kv_qh_start = int(kv_hdx * self.group_size)
                            kv_qh_end = min(int(kv_qh_start + self.group_size), int(self.num_heads))
                            kv_group = (
                                qhead_buf[kv_qh_start:kv_qh_end, sample_pos, :]
                                .detach()
                                .float()
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )
                            if kv_group.ndim == 3 and kv_group.shape[0] > 0:
                                kv_queries = self._score_transform_np(np.mean(kv_group, axis=0))
                                if use_causal_ref:
                                    kv_ref_knn = self._causal_topk_ref_np(
                                        queries=kv_queries,
                                        keys=keys_cpu,
                                        query_indices=sample_idx,
                                        k=self.q_knn,
                                    )
                                else:
                                    if 'ref_index' not in locals():
                                        ref_index = faiss.IndexFlatIP(self.head_dim)
                                        ref_index.add(keys_cpu)
                                    _, kv_ref_knn = ref_index.search(kv_queries, self.q_knn)
                                kv_ref_knn = np.ascontiguousarray(kv_ref_knn, dtype=np.int32)
                                for qhi in range(kv_group.shape[0]):
                                    q_queries_full = self._score_transform_np(kv_group[qhi])
                                    if use_causal_ref:
                                        q_ref_knn = self._causal_topk_ref_np(
                                            queries=q_queries_full,
                                            keys=keys_cpu,
                                            query_indices=sample_idx,
                                            k=self.q_knn,
                                        )
                                    else:
                                        _, q_ref_knn = ref_index.search(q_queries_full, self.q_knn)
                                    kv_proxy_recalls.append(
                                        float(self._knn_recall_at_k(kv_ref_knn, q_ref_knn, self.q_knn))
                                    )

                if len(per_q_head_recalls) == 0:
                    # Fallback: legacy grouped-query parity.
                    queries_cpu = (
                        self.cpu_queries[ldx][kv_hdx, sample_idx, :]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    queries_cpu = self._score_transform_np(queries_cpu)
                    trav_ref_knn = self._decode_dynamic_topk_ref_np(
                        queries=queries_cpu,
                        keys=keys_cpu,
                        k=self.q_knn,
                    )
                    if use_causal_ref:
                        ref_knn = self._causal_topk_ref_np(
                            queries=queries_cpu,
                            keys=keys_cpu,
                            query_indices=sample_idx,
                            k=self.q_knn,
                        )
                    else:
                        ref_index = faiss.IndexFlatIP(self.head_dim)
                        ref_index.add(keys_cpu)
                        _, ref_knn = ref_index.search(queries_cpu, self.q_knn)
                    if traversal_queries is None and trav_ref_knn.shape[0] == queries_cpu.shape[0]:
                        traversal_queries = queries_cpu
                        traversal_ref_knn = trav_ref_knn
                    rec = self._knn_recall_at_k(knn_sample, ref_knn, self.q_knn)
                    rec_min = rec
                    rec_max = rec
                    mode_tag = "grouped_fallback"
                    qh_count = 1
                else:
                    rec = float(np.mean(per_q_head_recalls))
                    rec_min = float(np.min(per_q_head_recalls))
                    rec_max = float(np.max(per_q_head_recalls))
                    mode_tag = "per_q_head_mean"
                    qh_count = len(per_q_head_recalls)
                parity_record = {
                    "sample_n": int(sample_n),
                    "recall": float(rec),
                }
                if len(kv_proxy_recalls) > 0:
                    kv_proxy_mean = float(np.mean(kv_proxy_recalls))
                    kv_proxy_min = float(np.min(kv_proxy_recalls))
                    kv_proxy_max = float(np.max(kv_proxy_recalls))
                    parity_record.update(
                        {
                            "kv_proxy_recall": kv_proxy_mean,
                            "kv_proxy_min": kv_proxy_min,
                            "kv_proxy_max": kv_proxy_max,
                            "kv_proxy_qh": int(len(kv_proxy_recalls)),
                        }
                    )
                else:
                    kv_proxy_mean = None
                kv_proxy_traversal_metrics = None
                split_tag = "holdout" if self.parity_holdout_only else "all_queries"
                parity_msg = (
                    f"[RetrievalAttention] parity layer={ldx} head={parity_hdx} sample={sample_n} "
                    f"recall@{self.q_knn}={rec:.4f} "
                    f"range=[{rec_min:.4f},{rec_max:.4f}] qh={qh_count} mode={mode_tag} "
                    f"causal_ref={int(use_causal_ref)} split={split_tag}"
                )
                if kv_proxy_mean is not None:
                    parity_msg = (
                        f"{parity_msg} kv_proxy@{self.q_knn}={kv_proxy_mean:.4f}"
                    )

        proj_start = time.time()
        knn_graph = self._select_graph_knn_rows(knn)
        if graph_override is not None:
            graph = graph_override
            graph_meta = dict(graph_override_meta)
            graph_meta.setdefault("builder", "flashattn_fused_graph")
            graph_meta.setdefault("backend", "gpu_dense")
            graph_meta.setdefault("stop_reason", "ok")
        else:
            graph, graph_meta = self._build_graph_csr_from_knn(knn_graph, keys_cpu=keys_cpu)
        hub_seeds = self._build_hub_seeds_from_graph(graph)
        traversal_metrics = None
        kv_proxy_traversal_metrics = None
        if (
            parity_record is not None
            and self.traversal_eval
            and traversal_queries is not None
            and traversal_ref_knn is not None
        ):
            traversal_metrics = self._evaluate_traversal_efficiency(
                ldx=ldx,
                hdx=parity_hdx,
                kv_hdx=kv_hdx,
                queries_np=traversal_queries,
                ref_knn_np=traversal_ref_knn,
                graph=graph,
                decode_index=decode_index,
            )
            if traversal_metrics is not None and parity_msg is not None:
                parity_msg = (
                    f"{parity_msg} "
                    f"trav_ref=decode_dynamic "
                    f"trav_sample={int(traversal_metrics.get('samples', 0))} "
                    f"trav_recall_strict@{self.q_knn}={float(traversal_metrics.get('recall', 0.0)):.4f} "
                    f"trav_recall_cov@{self.q_knn}={float(traversal_metrics.get('recall_cov', 0.0)):.4f} "
                    f"visited={float(traversal_metrics.get('visited_mean', 0.0)):.1f} "
                    f"visit_rate={100.0 * float(traversal_metrics.get('visit_rate', 0.0)):.2f}% "
                    f"prune_rate={100.0 * float(traversal_metrics.get('prune_rate', 0.0)):.2f}%"
                )
            if (
                graph_override is not None
                and self.fa_graph_fused_check
                and traversal_metrics is not None
                and float(traversal_metrics.get("recall", 0.0)) < self.fa_graph_fused_quality_floor
            ):
                # Safety gate: sampled strict traversal recall fell below configured floor.
                # Fall back to legacy graph construction for this head.
                graph, graph_meta = self._build_graph_csr_from_knn(knn_graph, keys_cpu=keys_cpu)
                hub_seeds = self._build_hub_seeds_from_graph(graph)
                traversal_metrics = self._evaluate_traversal_efficiency(
                    ldx=ldx,
                    hdx=parity_hdx,
                    kv_hdx=kv_hdx,
                    queries_np=traversal_queries,
                    ref_knn_np=traversal_ref_knn,
                    graph=graph,
                    decode_index=decode_index,
                )
                if parity_msg is not None:
                    parity_msg = (
                        f"{parity_msg} "
                        f"graph_fused_fallback=1 "
                        f"floor={self.fa_graph_fused_quality_floor:.3f}"
                    )
        if (
            parity_record is not None
            and self.traversal_eval
            and parity_record.get("kv_proxy_recall") is not None
            and qhead_buf is not None
            and sample_pos is not None
            and keys_cpu is not None
        ):
            kv_qh_start = int(kv_hdx * self.group_size)
            kv_qh_end = min(int(kv_qh_start + self.group_size), int(self.num_heads))
            kv_group = (
                qhead_buf[kv_qh_start:kv_qh_end, sample_pos, :]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            if kv_group.ndim == 3 and kv_group.shape[0] > 0:
                kv_queries = self._score_transform_np(np.mean(kv_group, axis=0))
                kv_trav_ref_knn = self._decode_dynamic_topk_ref_np(
                    queries=kv_queries,
                    keys=keys_cpu,
                    k=self.q_knn,
                )
                kv_proxy_traversal_metrics = self._evaluate_traversal_efficiency(
                    ldx=ldx,
                    hdx=parity_hdx,
                    kv_hdx=kv_hdx,
                    queries_np=kv_queries,
                    ref_knn_np=kv_trav_ref_knn,
                    graph=graph,
                    decode_index=decode_index,
                )
                if isinstance(kv_proxy_traversal_metrics, dict):
                    parity_record.update(
                        {
                            "kv_proxy_trav_samples": int(kv_proxy_traversal_metrics.get("samples", 0)),
                            "kv_proxy_trav_recall": float(kv_proxy_traversal_metrics.get("recall", 0.0)),
                            "kv_proxy_trav_recall_cov": float(kv_proxy_traversal_metrics.get("recall_cov", 0.0)),
                            "kv_proxy_trav_visited_mean": float(kv_proxy_traversal_metrics.get("visited_mean", 0.0)),
                            "kv_proxy_trav_visit_rate": float(kv_proxy_traversal_metrics.get("visit_rate", 0.0)),
                            "kv_proxy_trav_prune_rate": float(kv_proxy_traversal_metrics.get("prune_rate", 0.0)),
                            "kv_proxy_trav_cand_per_visit": float(kv_proxy_traversal_metrics.get("cand_per_visit", 0.0)),
                        }
                    )
                    if parity_msg is not None:
                        parity_msg = (
                            f"{parity_msg} "
                            f"kv_proxy_trav@{self.q_knn}={float(kv_proxy_traversal_metrics.get('recall', 0.0)):.4f}"
                        )
        kv_graph_traversal_metrics = None
        if (
            parity_record is not None
            and self.kv_graph_ab
            and self.traversal_eval
            and traversal_queries is not None
            and traversal_ref_knn is not None
            and keys_cpu is not None
        ):
            kv_graph_payload = self._get_kv_graph_ab_graph(ldx=ldx, kv_hdx=kv_hdx, keys_cpu=keys_cpu)
            if kv_graph_payload is not None:
                kv_graph, kv_graph_meta = kv_graph_payload
                kv_graph_traversal_metrics = self._evaluate_traversal_efficiency(
                    ldx=ldx,
                    hdx=parity_hdx,
                    kv_hdx=kv_hdx,
                    queries_np=traversal_queries,
                    ref_knn_np=traversal_ref_knn,
                    graph=kv_graph,
                    decode_index=decode_index,
                )
                if isinstance(kv_graph_traversal_metrics, dict):
                    parity_record.update(
                        {
                            "kv_graph_trav_samples": int(kv_graph_traversal_metrics.get("samples", 0)),
                            "kv_graph_trav_recall": float(kv_graph_traversal_metrics.get("recall", 0.0)),
                            "kv_graph_trav_recall_cov": float(kv_graph_traversal_metrics.get("recall_cov", 0.0)),
                            "kv_graph_trav_visited_mean": float(kv_graph_traversal_metrics.get("visited_mean", 0.0)),
                            "kv_graph_trav_visit_rate": float(kv_graph_traversal_metrics.get("visit_rate", 0.0)),
                            "kv_graph_trav_prune_rate": float(kv_graph_traversal_metrics.get("prune_rate", 0.0)),
                            "kv_graph_trav_cand_per_visit": float(kv_graph_traversal_metrics.get("cand_per_visit", 0.0)),
                        }
                    )
                    if parity_msg is not None:
                        parity_msg = (
                            f"{parity_msg} "
                            f"kv_graph_trav@{self.q_knn}={float(kv_graph_traversal_metrics.get('recall', 0.0)):.4f}"
                        )
        if parity_record is not None:
            self._record_parity(
                ldx,
                parity_hdx,
                int(parity_record["sample_n"]),
                float(parity_record["recall"]),
                self.q_knn,
                traversal=traversal_metrics,
                extras={k: v for k, v in parity_record.items() if k not in {"sample_n", "recall"}},
            )
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
            "kv_hdx": kv_hdx,
            "graph_hdx": graph_hdx,
            "parity_hdx": parity_hdx,
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
        kv_hdx = int(result.get("kv_hdx", self._retrieval_head_to_kv_head(hdx)))
        graph_hdx = int(result.get("graph_hdx", kv_hdx))
        decode_index = result.get("decode_index")
        if decode_index is not None:
            self.indexes[ldx][kv_hdx] = decode_index
        self.graphs[ldx][graph_hdx] = result["graph"]
        self.hub_seeds[ldx][graph_hdx] = list(result.get("hub_seeds", []))

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
            (graph_builder in {"roar", "roar_cpp", "flashattn_fused_graph"} and self.roar_log)
            or (str(graph_meta.get("stop_reason", "ok")) != "ok")
        ):
            extra = (
                f" builder={graph_builder} "
                f"bip={float(graph_meta.get('bipartite_sec', 0.0)):.2f}s "
                f"enh={float(graph_meta.get('enhance_sec', 0.0)):.2f}s "
                f"csr={float(graph_meta.get('csr_sec', 0.0)):.2f}s "
                f"gsec={float(graph_meta.get('graph_sec', 0.0)):.2f}s "
                f"active_q={int(graph_meta.get('active_queries', 0))} "
                f"active_p={int(graph_meta.get('active_pivots', 0))} "
                f"nodes={int(graph_meta.get('projected_nodes', 0))} "
                f"enh_nodes={int(graph_meta.get('enhanced_nodes', 0))} "
                f"stop={graph_meta.get('stop_reason', 'ok')}"
            )
        print(
            f"[RetrievalAttention] index built layer={ldx} head={graph_hdx} "
            f"kv_head={kv_hdx} "
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

        # Store sampled per-Q-head queries for parity validation.
        qhead_sample_buf = None
        if layer_idx < len(self.cpu_queries_qhead_samples):
            qhead_sample_buf = self.cpu_queries_qhead_samples[layer_idx]
        if qhead_sample_buf is not None and self._parity_query_indices_torch is not None:
            full_q = query_states[0, valid_start:valid_start + self.input_length, :, :]  # [seq, num_heads, dim]
            parity_idx = self._parity_query_indices_torch
            if parity_idx.numel() > 0:
                if parity_idx.device != full_q.device:
                    parity_idx = parity_idx.to(full_q.device, non_blocking=True)
                parity_idx = torch.clamp(parity_idx, min=0, max=max(0, full_q.shape[0] - 1))
                q_samples = torch.index_select(full_q, 0, parity_idx)  # [S, num_heads, dim]
                q_samples = q_samples.transpose(0, 1).contiguous()      # [num_heads, S, dim]
                qhead_sample_buf[:, :q_samples.shape[1], :].copy_(q_samples, non_blocking=True)
        qhead_full_buf = None
        if layer_idx < len(self.cpu_queries_qhead_full):
            qhead_full_buf = self.cpu_queries_qhead_full[layer_idx]
        if qhead_full_buf is not None:
            full_q = query_states[0, valid_start:valid_start + self.input_length, :, :]  # [seq, num_heads, dim]
            q_full = full_q.transpose(0, 1).contiguous()  # [num_heads, seq, dim]
            qhead_full_buf[:, :q_full.shape[1], :].copy_(q_full, non_blocking=True)

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
        if self._built:
            return
        if self.decode_index_mode == "faiss" and faiss is None:
            raise RuntimeError("RETRIEVALATTN_DECODE_INDEX=faiss requires faiss-cpu.")
        if self._fused_async_enabled:
            missing_layers = [
                ldx for ldx in range(self.layer_num)
                if not self._fused_prefill_submitted[ldx]
            ]
        else:
            missing_layers = [
                ldx for ldx in range(self.layer_num)
                if self.fused_prefill_knn[ldx] is None
            ]
        if missing_layers:
            raise RuntimeError(
                "Fused prefill KNN is missing for layers: "
                f"{missing_layers}. Ensure fused prefill attention path is active."
            )

        start_ts = time.time()
        cpu_cap = self._get_allocated_cpu_count()
        profile_enabled = os.environ.get("RETRIEVALATTN_PROFILE", "1") == "1"

        # Enable Faiss threads within scheduler CPU allocation.
        num_threads = None
        usable_cpu_for_faiss = cpu_cap
        if faiss is not None:
            try:
                num_threads, usable_cpu_for_faiss = self._resolve_faiss_threads(
                    cpu_cap=cpu_cap,
                    pipeline_enabled=False,
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
        fused_overlap_msg = (
            f", fused_overlap_cfg={int(self.fused_prefill_overlap)}, "
            f"fused_overlap_enabled={int(self._fused_async_enabled)}, "
            f"fused_overlap_workers={self.fused_prefill_overlap_workers}"
        )
        print(
            f"[RetrievalAttention] Building ANN indexes (layers={self.layer_num}, kv_heads={self.kv_head}, "
            f"retrieval_heads={self.retrieval_heads}, retrieval_head_mode={self.retrieval_head_mode}, "
            f"tokens={self.input_length}, mode=flashattn_fused_prefill, decode_index={self.decode_index_mode}, "
            f"seed_mode={self.seed_mode}, query_mode={self.query_mode}, score_mode={self.score_mode}, "
            f"graph_builder={self.graph_builder}, "
            f"roar_backend={self.roar_backend}, roar_cpp_available={int(self._roar_cpp_available)}, "
            f"roar_cpp_threads={self.roar_cpp_threads}, "
            f"roar_py_gpu={int(self.roar_python_gpu_enabled)}, "
            f"roar_py_gpu_device={self.roar_python_gpu_device}, "
            f"roar_py_gpu_batch={self.roar_python_gpu_batch}, "
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
            f"graph_train_frac={self.graph_train_frac:.3f}, "
            f"graph_split={self.graph_split_mode}, graph_split_seed={self.graph_split_seed}, "
            f"graph_train_queries={int(self.graph_train_query_indices.size)}, "
            f"graph_holdout_queries={int(self.graph_holdout_query_indices.size)}, "
            f"parity_holdout_only={int(self.parity_holdout_only)}, "
            f"traversal_eval={int(self.traversal_eval)}, "
            f"traversal_eval_sample={self.traversal_eval_sample}, "
            f"fused_prefill=1, "
            f"graph_fused_prefill={int(self.fa_graph_fused)}, "
            f"graph_fused_require={int(self.fa_graph_fused_require)}, "
            f"graph_fused_check={int(self.fa_graph_fused_check)}, "
            f"graph_fused_floor={self.fa_graph_fused_quality_floor:.3f}, "
            f"fused_shadow={int(self.fa_shadow_compare)}){thread_msg}{fused_overlap_msg}"
        )

        if self._fused_async_enabled:
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
        else:
            for ldx in range(self.layer_num):
                layer_start = time.time()
                layer_knn = self.fused_prefill_knn[ldx]
                if layer_knn is None:
                    raise RuntimeError(f"Missing fused prefill KNN for layer={ldx}")
                profile = (
                    self.fused_prefill_profiles[ldx]
                    if isinstance(self.fused_prefill_profiles[ldx], dict)
                    else {}
                )
                layer_graph = self.fused_prefill_graph_neighbors[ldx]
                layer_graph_profile = (
                    self.fused_prefill_graph_profiles[ldx]
                    if isinstance(self.fused_prefill_graph_profiles[ldx], dict)
                    else {}
                )
                self._finalize_fused_layer(
                    ldx,
                    layer_knn,
                    profile,
                    layer_graph,
                    layer_graph_profile,
                )
                layer_elapsed = time.time() - layer_start
                total_elapsed = time.time() - start_ts
                print(
                    f"[RetrievalAttention] layer {ldx} done in {layer_elapsed:.2f}s "
                    f"(total {total_elapsed:.2f}s)"
                )

        # Free queries to save CPU memory after graph/index finalize.
        self.cpu_queries = None
        self.cpu_queries_qhead_full = [None for _ in range(self.layer_num)]
        self._shutdown_fused_prefill_executor()
        with self._fused_prefill_lock:
            self._fused_prefill_futures.clear()
        self.prev_decode_seeds = [[[] for _ in range(self.retrieval_heads)] for _ in range(self.layer_num)]
        self._built = True

    def sync(self, layer_idx, start_bdx):
        """
        Keep interface compatibility with other KV caches.
        RetrievalAttention does not use async GPU copy events, so this is a no-op.
        """
        return

    def reset_decode_profile(self):
        int_keys = {
            "calls",
            "heads",
            "visited_total",
            "candidates_total",
            "search_space_total",
            "search_space_heads",
            "visited_ratio_count",
        }
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
        heads = int(stats["heads"])
        visited_total = int(stats["visited_total"])
        candidates_total = int(stats["candidates_total"])
        search_space_total = int(stats["search_space_total"])
        search_space_heads = int(stats["search_space_heads"])
        visited_ratio_sum = float(stats["visited_ratio_sum"])
        visited_ratio_count = int(stats["visited_ratio_count"])

        def pct(v: float) -> float:
            return 100.0 * v / total if total > 0 else 0.0

        visited_per_head = (
            float(visited_total) / float(heads)
            if heads > 0 else 0.0
        )
        search_space_per_head = (
            float(search_space_total) / float(search_space_heads)
            if search_space_heads > 0 else 0.0
        )
        visit_rate_weighted = (
            float(visited_total) / float(search_space_total)
            if search_space_total > 0 else 0.0
        )
        visit_rate_mean = (
            float(visited_ratio_sum) / float(visited_ratio_count)
            if visited_ratio_count > 0 else 0.0
        )
        prune_rate_weighted = max(0.0, 1.0 - visit_rate_weighted)
        cand_per_visit = (
            float(candidates_total) / float(visited_total)
            if visited_total > 0 else 0.0
        )

        msg = (
            "[RetrievalAttention] decode_profile "
            f"calls={int(stats['calls'])} heads={heads} "
            f"total={total:.3f}s | "
            f"retrieve={retrieve:.3f}s ({pct(retrieve):.1f}%) "
            f"[seed={stats['retrieve_seed_sec']:.3f}s, "
            f"graph={stats['retrieve_graph_sec']:.3f}s, "
            f"rerank={stats['retrieve_rerank_sec']:.3f}s, "
            f"finalize={stats['retrieve_finalize_sec']:.3f}s] | "
            f"gather={gather:.3f}s ({pct(gather):.1f}%) | "
            f"attn={attn:.3f}s ({pct(attn):.1f}%) | "
            f"other={other:.3f}s ({pct(other):.1f}%) | "
            f"visited_total={visited_total} "
            f"candidates_total={candidates_total} | "
            f"traversal=[space/head={search_space_per_head:.1f}, "
            f"visited/head={visited_per_head:.1f}, "
            f"visit_rate={100.0 * visit_rate_weighted:.2f}%, "
            f"visit_rate_mean={100.0 * visit_rate_mean:.2f}%, "
            f"prune_rate={100.0 * prune_rate_weighted:.2f}%, "
            f"cand/visit={cand_per_visit:.2f}x]"
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

    def _retrieve_tokens(
        self,
        ldx,
        hdx,
        query_group,
        update_decode_state: bool = True,
        enforce_seed_floor: bool = True,
    ):
        """
        Retrieve token indices using seed search + adaptive best-first K-K graph expansion.
        Final candidate list is reranked with the configured retrieval score mode.
        """
        kv_hdx = self._retrieval_head_to_kv_head(hdx)
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
                "search_space": 0,
                "visited_ratio": 0.0,
                "stop_reason": "n/a",
            }

        def finish(tokens, stop_reason: str, visited: int = 0, candidates: int = 0):
            if profile is not None:
                profile["stop_reason"] = stop_reason
                visited_i = max(0, int(visited))
                candidates_i = max(0, int(candidates))
                search_space = max(0, int(self.dynamic_end - self.dynamic_start))
                profile["visited"] = visited_i
                profile["candidates"] = candidates_i
                profile["search_space"] = search_space
                profile["visited_ratio"] = (
                    float(visited_i) / float(search_space)
                    if search_space > 0 else 0.0
                )
                profile["total_sec"] = time.perf_counter() - total_start
            return tokens, profile

        index = self.indexes[ldx][kv_hdx]
        graph = self.graphs[ldx][kv_hdx]
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
                k = self.cpu_keys[ldx][kv_hdx, :self.input_length, :].detach().float().cpu()
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
            hub_tokens = self.hub_seeds[ldx][kv_hdx]
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
                seed_kv = torch.index_select(self.cpu_keys[ldx][kv_hdx], 0, idx).detach().float().cpu()
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
        visited_count = 0

        # Prime candidates with seeds ranked by ANN score.
        for tok, score in seed_ranked:
            if len(candidates) >= candidate_target:
                break
            tok = int(tok)
            if tok not in seen:
                candidates.append(tok)
                seen.add(tok)
                score_f = float(score)
                candidate_scores[tok] = score_f

        stop_reason = "graph_disabled"

        graph_start = time.perf_counter() if profile is not None else None
        if self.graph_expand and graph is not None:
            use_cpp_decode = self._should_use_roar_cpp_decode_backend(graph_is_csr)
            if use_cpp_decode:
                try:
                    init_take = min(len(seed_ranked), max(1, self.roar_decode_init))
                    init_ids = np.asarray([int(tok) for tok, _ in seed_ranked[:init_take]], dtype=np.int32)
                    init_scores = np.asarray([float(score) for _, score in seed_ranked[:init_take]], dtype=np.float32)
                    lpq = int(self.roar_decode_lpq) if self.roar_decode_lpq > 0 else int(candidate_target)
                    lpq = max(self.token_budget, lpq)
                    lpq = min(self.input_length, lpq)
                    topk_take = min(candidate_target, lpq)
                    max_hops = int(self.roar_decode_max_hops) if self.roar_decode_max_hops > 0 else int(self.max_visits)
                    max_cmps = int(self.roar_decode_max_cmps)
                    key_arr, key_dtype = self._get_decode_key_array(ldx, kv_hdx)
                    q_seed_np = np.ascontiguousarray(q_seed_cpu.numpy(), dtype=np.float32)
                    ids_cpp, scores_cpp, meta_cpp = search_roar_graph_csr_cpp(
                        query=q_seed_np,
                        keys=key_arr,
                        offsets=graph_offsets,
                        neighbors=graph_neighbors,
                        init_ids=init_ids,
                        init_scores=init_scores,
                        topk=topk_take,
                        lpq=lpq,
                        max_cmps=max_cmps,
                        max_hops=max_hops,
                        dynamic_start=self.dynamic_start,
                        dynamic_end=self.dynamic_end,
                        num_threads=int(self.roar_decode_threads),
                        score_agg=self.rerank_agg,
                        key_dtype=key_dtype,
                    )
                    if isinstance(meta_cpp, dict):
                        stop_reason = str(meta_cpp.get("stop_reason", "roar_cpp"))
                        visited_count = int(meta_cpp.get("hops", meta_cpp.get("visited", 0)))
                    else:
                        stop_reason = "roar_cpp"
                        visited_count = 0
                    ids_cpp = np.asarray(ids_cpp, dtype=np.int32)
                    scores_cpp = np.asarray(scores_cpp, dtype=np.float32)
                    for i in range(ids_cpp.shape[0]):
                        if len(candidates) >= candidate_target:
                            stop_reason = "candidate_cap"
                            break
                        tok = int(ids_cpp[i])
                        if tok in static_indices or tok in seen:
                            continue
                        score = float(scores_cpp[i]) if i < scores_cpp.shape[0] else -1e9
                        candidates.append(tok)
                        seen.add(tok)
                        candidate_scores[tok] = score
                    if len(candidates) >= candidate_target and stop_reason == "frontier_empty":
                        stop_reason = "candidate_cap"
                except Exception as exc:
                    if self.decode_backend == "roar_cpp":
                        raise
                    if not self._decode_cpp_warned:
                        print(
                            "[RetrievalAttention] WARNING: decode roar cpp search failed; "
                            f"falling back to python traversal. error={exc}"
                        )
                        self._decode_cpp_warned = True
                    use_cpp_decode = False

            if not use_cpp_decode:
                frontier = []
                frontier_best_by_node = {}
                expanded = set()
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

                for tok, score in candidate_scores.items():
                    push_frontier(int(tok), float(score))

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
                        new_k = torch.index_select(self.cpu_keys[ldx][kv_hdx], 0, idx).detach().float().cpu()
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
            cand_k = torch.index_select(self.cpu_keys[ldx][kv_hdx], 0, idx).detach().float().cpu()
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

        # Enforce seed floor in final list (decode path). Traversal-eval strict
        # top-k quality can disable this to avoid rank distortion from seed forcing.
        finalize_start = time.perf_counter() if profile is not None else None
        final = []
        final_set = set()
        if enforce_seed_floor and seed_floor > 0:
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

        if self.debug and update_decode_state and self.decode_pos < self.debug_decode_steps and ldx == 0:
            expanded_cnt = sum(1 for tok in candidates if tok not in selected_seed_set)
            print(
                f"[RetrievalAttention][debug] step={self.decode_pos} layer={ldx} head={hdx} "
                f"seeds={len(seed_ranked)} seed_floor={seed_floor} expanded={expanded_cnt} "
                f"visited={visited_count} candidates={len(candidates)} "
                f"dynamic={len(final)} stop_reason={stop_reason}"
            )
        if update_decode_state:
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
        q_grouped = q.view(self.kv_head, self.group_size, self.head_dim)

        outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)
        empty_heads = 0
        dynamic_counts = []
        head_count = self.retrieval_heads

        for hdx in range(head_count):
            kv_hdx = self._retrieval_head_to_kv_head(hdx)
            if self.retrieval_head_mode == "q_head":
                q_group = q[hdx]  # [head_dim]
                q_attn = q_group.unsqueeze(0)  # [1, head_dim]
            else:
                q_group = q_grouped[hdx]  # [group_size, head_dim]
                q_attn = q_group

            token_ids, retrieve_profile = self._retrieve_tokens(layer_idx, hdx, q_group)
            if self.decode_profile and retrieve_profile is not None:
                self._decode_profile_stats["retrieve_total_sec"] += float(retrieve_profile["total_sec"])
                self._decode_profile_stats["retrieve_seed_sec"] += float(retrieve_profile["seed_sec"])
                self._decode_profile_stats["retrieve_graph_sec"] += float(retrieve_profile["graph_sec"])
                self._decode_profile_stats["retrieve_rerank_sec"] += float(retrieve_profile["rerank_sec"])
                self._decode_profile_stats["retrieve_finalize_sec"] += float(retrieve_profile["finalize_sec"])
                self._decode_profile_stats["visited_total"] += int(retrieve_profile["visited"])
                self._decode_profile_stats["candidates_total"] += int(retrieve_profile["candidates"])
                search_space = max(0, int(retrieve_profile.get("search_space", 0)))
                self._decode_profile_stats["search_space_total"] += search_space
                if search_space > 0:
                    self._decode_profile_stats["search_space_heads"] += 1
                self._decode_profile_stats["visited_ratio_sum"] += float(
                    retrieve_profile.get("visited_ratio", 0.0)
                )
                self._decode_profile_stats["visited_ratio_count"] += 1
            if len(token_ids) == 0:
                empty_heads += 1
            dynamic_counts.append(len(token_ids))

            # Gather dynamic tokens from CPU
            gather_start = time.perf_counter() if self.decode_profile else None
            if token_ids:
                idx = torch.tensor(token_ids, dtype=torch.long, device="cpu")
                dyn_k = torch.index_select(self.cpu_keys[layer_idx][kv_hdx], 0, idx).to(device, non_blocking=True)
                dyn_v = torch.index_select(self.cpu_values[layer_idx][kv_hdx], 0, idx).to(device, non_blocking=True)
            else:
                dyn_k = None
                dyn_v = None
            if self.decode_profile:
                self._decode_profile_stats["gather_total_sec"] += (time.perf_counter() - gather_start)

            # Static KV on GPU
            static_k = self.static_gpu_keys[layer_idx][kv_hdx]
            static_v = self.static_gpu_values[layer_idx][kv_hdx]

            if dyn_k is not None:
                k = torch.cat([static_k, dyn_k], dim=0)
                v = torch.cat([static_v, dyn_v], dim=0)
            else:
                k = static_k
                v = static_v

            # Attention: q_attn [G, D] (or [1, D] in q_head mode), k [T, D] -> [G, T]
            attn_start = time.perf_counter() if self.decode_profile else None
            scores = torch.matmul(q_attn, k.transpose(0, 1)) * scale
            scores = scores.float()
            attn = torch.softmax(scores, dim=-1).to(v.dtype)
            out = torch.matmul(attn, v)  # [G, D] or [1, D]
            if self.decode_profile:
                self._decode_profile_stats["attn_total_sec"] += (time.perf_counter() - attn_start)
            outputs.append(out)
            if self.decode_profile:
                self._decode_profile_stats["heads"] += 1

        if self.assert_nonempty and empty_heads == head_count:
            raise RuntimeError(
                f"[RetrievalAttention] Empty dynamic retrieval for all heads at decode step={self.decode_pos}, "
                f"layer={layer_idx}. Check decode seed index path."
            )
        if self.debug and self.decode_pos < self.debug_decode_steps and layer_idx == 0:
            avg_dyn = float(sum(dynamic_counts)) / float(len(dynamic_counts)) if dynamic_counts else 0.0
            print(
                f"[RetrievalAttention][debug] step={self.decode_pos} layer={layer_idx} "
                f"empty_heads={empty_heads}/{head_count} avg_dynamic={avg_dyn:.1f}"
            )

        if self.decode_profile:
            self._decode_profile_stats["calls"] += 1
            self._decode_profile_stats["compute_total_sec"] += (time.perf_counter() - compute_start)

        out = torch.cat(outputs, dim=0).view(1, 1, self.num_heads, self.head_dim)
        return out
