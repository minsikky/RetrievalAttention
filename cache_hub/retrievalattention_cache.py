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
    adaptive_budget_select_cuda,
    build_roar_graph_csr_cpp,
    search_roar_graph_csr_cpp,
    search_roar_graph_csr_cuda,
    search_roar_graph_csr_cuda_frontier,
    search_roar_graph_csr_cuda_group_fullgpu,
    search_roar_graph_csr_cuda_group,
    search_roar_graph_csr_cuda_group_kernel,
    search_roar_graph_csr_cuda_group_beam,
    roargraph_cpp_available,
    roargraph_cpp_import_error,
    roargraph_cuda_available,
    roargraph_cuda_import_error,
    roargraph_cuda_kernel_available,
    roargraph_cuda_kernel_import_error,
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
        self.fullgpu_profile_sync = os.environ.get("RETRIEVALATTN_FULLGPU_PROFILE_SYNC", "1") == "1"
        self.fullgpu_ab = os.environ.get("RETRIEVALATTN_FULLGPU_AB", "0") == "1"
        try:
            self.fullgpu_ab_layer = int(os.environ.get("RETRIEVALATTN_FULLGPU_AB_LAYER", "0"))
        except Exception:
            self.fullgpu_ab_layer = 0
        try:
            self.fullgpu_ab_step = int(os.environ.get("RETRIEVALATTN_FULLGPU_AB_STEP", "0"))
        except Exception:
            self.fullgpu_ab_step = 0
        self.fullgpu_kernel_debug = os.environ.get("RETRIEVALATTN_FULLGPU_KERNEL_DEBUG", "0") == "1"
        try:
            self.fullgpu_kernel_debug_layer = int(os.environ.get("RETRIEVALATTN_FULLGPU_KERNEL_DEBUG_LAYER", "-1"))
        except Exception:
            self.fullgpu_kernel_debug_layer = -1
        try:
            self.fullgpu_kernel_debug_step = int(os.environ.get("RETRIEVALATTN_FULLGPU_KERNEL_DEBUG_STEP", "-1"))
        except Exception:
            self.fullgpu_kernel_debug_step = -1
        self._fullgpu_ab_done = False
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
        raw_decode_backend = os.environ.get("RETRIEVALATTN_DECODE_BACKEND", "auto").strip().lower()
        if raw_decode_backend in {"", "auto"}:
            self.decode_backend = "auto"
        elif raw_decode_backend in {"python", "py"}:
            self.decode_backend = "python"
        elif raw_decode_backend in {"roar_cpp", "cpp"}:
            self.decode_backend = "roar_cpp"
        elif raw_decode_backend in {"roar_cuda", "cuda"}:
            self.decode_backend = "roar_cuda"
        elif raw_decode_backend in {"roar_cuda_v2", "cuda_v2"}:
            self.decode_backend = "roar_cuda_v2"
        elif raw_decode_backend in {"roar_cuda_kernel", "cuda_kernel", "kernel"}:
            self.decode_backend = "roar_cuda_kernel"
        elif raw_decode_backend in {"roar_cuda_fullgpu", "cuda_fullgpu", "fullgpu"}:
            self.decode_backend = "roar_cuda_fullgpu"
        elif raw_decode_backend in {"roar_cuda_frontier", "cuda_frontier", "frontier"}:
            self.decode_backend = "roar_cuda_frontier"
        elif raw_decode_backend in {"roar_cuda_beam", "cuda_beam", "beam"}:
            self.decode_backend = "roar_cuda_beam"
        else:
            print(
                f"[RetrievalAttention] WARNING: unknown RETRIEVALATTN_DECODE_BACKEND={raw_decode_backend}. "
                "Falling back to auto."
            )
            self.decode_backend = "auto"
        self.online_dynamic_range = os.environ.get("RETRIEVALATTN_ONLINE_DYNAMIC_RANGE", "0") == "1"
        self.online_graph_enable = os.environ.get("RETRIEVALATTN_ONLINE_GRAPH_ENABLE", "0") == "1"
        self.growing_static_suffix = os.environ.get("RETRIEVALATTN_GROW_STATIC_SUFFIX", "0") == "1"
        if self.growing_static_suffix and (self.online_dynamic_range or self.online_graph_enable):
            raise RuntimeError(
                "RETRIEVALATTN_GROW_STATIC_SUFFIX is a comparison mode and cannot be combined with "
                "RETRIEVALATTN_ONLINE_DYNAMIC_RANGE or RETRIEVALATTN_ONLINE_GRAPH_ENABLE."
            )
        if self.growing_static_suffix and self.decode_backend != "roar_cuda_fullgpu":
            raise RuntimeError(
                "RETRIEVALATTN_GROW_STATIC_SUFFIX currently requires "
                "RETRIEVALATTN_DECODE_BACKEND=roar_cuda_fullgpu."
            )
        self.online_graph_bidirectional = os.environ.get("RETRIEVALATTN_ONLINE_GRAPH_BIDIR", "1") == "1"
        try:
            self.online_graph_insert_k = int(os.environ.get("RETRIEVALATTN_ONLINE_GRAPH_INSERT_K", "8"))
        except Exception:
            self.online_graph_insert_k = 8
        self.online_graph_insert_k = max(1, self.online_graph_insert_k)
        try:
            self.online_graph_neighbor_cap = int(os.environ.get("RETRIEVALATTN_ONLINE_GRAPH_NEIGHBOR_CAP", "16"))
        except Exception:
            self.online_graph_neighbor_cap = 16
        self.online_graph_neighbor_cap = max(self.online_graph_insert_k, self.online_graph_neighbor_cap)
        self.online_graph_signal = os.environ.get("RETRIEVALATTN_ONLINE_GRAPH_SIGNAL", "next").strip().lower()
        if self.online_graph_signal not in {"next", "query_centroid"}:
            print(
                f"[RetrievalAttention] WARNING: unknown RETRIEVALATTN_ONLINE_GRAPH_SIGNAL="
                f"{self.online_graph_signal}; falling back to next."
            )
            self.online_graph_signal = "next"
        try:
            self.online_graph_query_topk = int(os.environ.get("RETRIEVALATTN_ONLINE_GRAPH_QUERY_TOPK", "4"))
        except Exception:
            self.online_graph_query_topk = 4
        self.online_graph_query_topk = max(1, self.online_graph_query_topk)
        try:
            self.online_graph_query_cand_per_step = int(
                os.environ.get("RETRIEVALATTN_ONLINE_GRAPH_QUERY_CAND_PER_STEP", str(self.online_graph_neighbor_cap))
            )
        except Exception:
            self.online_graph_query_cand_per_step = int(self.online_graph_neighbor_cap)
        self.online_graph_query_cand_per_step = max(1, self.online_graph_query_cand_per_step)
        try:
            self.online_graph_defer = int(os.environ.get("RETRIEVALATTN_ONLINE_GRAPH_DEFER", "-1"))
        except Exception:
            self.online_graph_defer = -1
        if self.online_graph_defer < 0:
            self.online_graph_defer = int(self.static_pattern_end)
        self.online_graph_defer = max(0, self.online_graph_defer)
        self.online_graph_log = os.environ.get("RETRIEVALATTN_ONLINE_GRAPH_LOG", "1") == "1"
        self.dynamic_budget_enable = os.environ.get("RETRIEVALATTN_DYNAMIC_BUDGET_ENABLE", "0") == "1"
        try:
            self.dynamic_budget_target_omass = float(
                os.environ.get("RETRIEVALATTN_DYNAMIC_BUDGET_TARGET_OMASS", "0.10")
            )
        except Exception:
            self.dynamic_budget_target_omass = 0.10
        self.dynamic_budget_target_omass = min(max(self.dynamic_budget_target_omass, 0.0), 0.99)
        try:
            self.dynamic_budget_min = int(os.environ.get("RETRIEVALATTN_DYNAMIC_BUDGET_MIN", "16"))
        except Exception:
            self.dynamic_budget_min = 16
        self.dynamic_budget_min = max(1, self.dynamic_budget_min)
        try:
            self.dynamic_budget_max = int(
                os.environ.get(
                    "RETRIEVALATTN_DYNAMIC_BUDGET_MAX",
                    str(max(int(self.token_budget), int(self.token_budget) * 4)),
                )
            )
        except Exception:
            self.dynamic_budget_max = max(int(self.token_budget), int(self.token_budget) * 4)
        self.dynamic_budget_max = max(int(self.dynamic_budget_min), int(self.dynamic_budget_max))
        self.dynamic_budget_mode = os.environ.get("RETRIEVALATTN_DYNAMIC_BUDGET_MODE", "torch").strip().lower()
        if self.dynamic_budget_mode not in {"torch", "cuda", "traversal_cuda"}:
            self.dynamic_budget_mode = "torch"
        self.dynamic_budget_prior = os.environ.get(
            "RETRIEVALATTN_DYNAMIC_BUDGET_PRIOR",
            "global_norm",
        ).strip().lower()
        if self.dynamic_budget_prior not in {"global_norm", "moment_diag"}:
            self.dynamic_budget_prior = "global_norm"
        try:
            self.dynamic_budget_prior_var_scale = float(
                os.environ.get("RETRIEVALATTN_DYNAMIC_BUDGET_PRIOR_VAR_SCALE", "1.0")
            )
        except Exception:
            self.dynamic_budget_prior_var_scale = 1.0
        self.dynamic_budget_prior_var_scale = max(0.0, float(self.dynamic_budget_prior_var_scale))
        self.dynamic_tail_enable = os.environ.get("RETRIEVALATTN_DYNAMIC_TAIL_ENABLE", "0") == "1"
        self.dynamic_tail_mode = os.environ.get(
            "RETRIEVALATTN_DYNAMIC_TAIL_MODE",
            "dynamic_mean",
        ).strip().lower()
        if self.dynamic_tail_mode not in {"dynamic_mean", "zero"}:
            self.dynamic_tail_mode = "dynamic_mean"
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
        self._roar_cuda_available = roargraph_cuda_available()
        self._roar_cuda_kernel_available = roargraph_cuda_kernel_available()
        if self.roar_backend == "cpp" and not self._roar_cpp_available:
            cpp_err = roargraph_cpp_import_error()
            raise RuntimeError(
                "RoarGraph C++ extension is required but unavailable. "
                "Build it with: "
                "`module load python/3.10.4 && source .venv/bin/activate && "
                "python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`"
                + (f". Import error: {cpp_err}" if cpp_err is not None else "")
            )
        if self.decode_backend in {"roar_cuda_kernel", "roar_cuda_fullgpu"} and not self._roar_cuda_kernel_available:
            cuda_err = roargraph_cuda_kernel_import_error()
            raise RuntimeError(
                "RoarGraph torch/CUDA kernel extension is required but unavailable. "
                "Build it with: "
                "`module load python/3.10.4 && source .venv/bin/activate && "
                "python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`"
                + (f". Import error: {cuda_err}" if cuda_err is not None else "")
            )
        if self.decode_backend in {"roar_cuda", "roar_cuda_v2", "roar_cuda_frontier", "roar_cuda_beam"} and not self._roar_cuda_available:
            cuda_err = roargraph_cuda_import_error()
            raise RuntimeError(
                "RoarGraph torch/CUDA extension is required but unavailable. "
                "Build it with: "
                "`module load python/3.10.4 && source .venv/bin/activate && "
                "python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`"
                + (f". Import error: {cuda_err}" if cuda_err is not None else "")
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
            self.roar_cuda_frontier_width = int(
                os.environ.get("RETRIEVALATTN_ROAR_CUDA_FRONTIER_BEAM", "32")
            )
        except Exception:
            self.roar_cuda_frontier_width = 32
        self.roar_cuda_frontier_width = max(1, self.roar_cuda_frontier_width)
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
            "other_setup_sec": 0.0,
            "other_state_prep_sec": 0.0,
            "other_profile_accum_sec": 0.0,
            "other_group_bookkeeping_sec": 0.0,
            "other_output_sec": 0.0,
            "visited_total": 0,
            "candidates_total": 0,
            "final_outputs_total": 0,
            "kernel_round_total": 0,
            "forced_seed_total": 0,
            "stop_frontier_empty": 0,
            "stop_max_visits": 0,
            "stop_candidate_cap": 0,
            "stop_stability_gap": 0,
            "stop_empty_init": 0,
            "search_space_total": 0,
            "search_space_heads": 0,
            "visited_ratio_sum": 0.0,
            "visited_ratio_count": 0,
            "online_update_sec": 0.0,
            "online_provenance_d2h_sec": 0.0,
            "online_overlay_build_cpu_sec": 0.0,
            "online_overlay_h2d_sec": 0.0,
            "online_insert_nodes": 0,
            "online_insert_edges": 0,
            "online_overlay_edges": 0,
            "online_generated_hits": 0,
            "online_generated_head_hits": 0,
            "adaptive_outputs_total": 0,
            "adaptive_total_sec": 0.0,
            "adaptive_upper_bound_sec": 0.0,
            "adaptive_static_logz_sec": 0.0,
            "adaptive_candidate_score_sec": 0.0,
            "adaptive_sort_sec": 0.0,
            "adaptive_select_sec": 0.0,
            "adaptive_reorder_sec": 0.0,
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
        self._decode_cuda_key_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_attn_key_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_attn_key_norm_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_attn_key_prefixmax_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_attn_key_sum_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_attn_key_sumsq_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_value_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_graph_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_graph_device_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_overlay_graph_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_overlay_graph_device_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_graph_degree_cache = [[None for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._decode_cuda_hub_seed_ids = [None for _ in range(self.layer_num)]
        self._decode_cuda_prev_seed_ids = [None for _ in range(self.layer_num)]
        self._decode_cuda_prev_seed_counts = [None for _ in range(self.layer_num)]
        self.oracle_retrieval_enable = False
        self.oracle_debug_enable = False
        self.oracle_answer_start_pos = None
        self.oracle_debug_records = []
        self.oracle_compare_enable = False
        self.oracle_compare_records = []
        self._decode_cpp_warned = False
        self._decode_cuda_warned = False
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
        self._online_graph_pending = [[{} for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._online_graph_pending_order = [[[] for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._online_graph_pending_cursor = [[0 for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._online_graph_overlay = [[{} for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._online_graph_query_sum = [[{} for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._online_graph_query_weight = [[{} for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        self._online_graph_query_candidates = [[{} for _ in range(self.kv_head)] for _ in range(self.layer_num)]
        # Track suffix window positions
        self.suffix_start = max(0, self.input_length - self.static_pattern_end)
        self.decode_pos = 0

        prefix_static = min(max(0, self.static_pattern_start), self.input_length)
        suffix_static = min(max(0, self.static_pattern_end), self.input_length)
        self.dynamic_start = prefix_static
        self.dynamic_end = max(self.dynamic_start, self.input_length - suffix_static)
        self.growing_static_dynamic_end = int(self.dynamic_end)
        self.static_index_set = set(range(prefix_static))
        self.static_index_set.update(range(self.dynamic_end, self.input_length))
        self._online_graph_warned = False

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

    def _decode_token_limit(self) -> int:
        if self.online_dynamic_range or self.online_graph_enable or self.growing_static_suffix:
            return max(int(self.input_length), int(self.context))
        return int(self.input_length)

    def _effective_dynamic_budget_cap(self) -> int:
        if self.dynamic_budget_enable:
            return min(512, int(self.dynamic_budget_max))
        return min(512, int(self.token_budget))

    def _rebuild_static_index_set(self, total_tokens: int):
        total_tokens = max(0, int(total_tokens))
        prefix = set(range(int(self.dynamic_start)))
        suffix_start = max(int(self.dynamic_start), int(self.dynamic_end))
        prefix.update(range(suffix_start, total_tokens))
        self.static_index_set = prefix

    def _refresh_decode_dynamic_state(self):
        if not (self.online_dynamic_range or self.online_graph_enable or self.growing_static_suffix):
            return
        total_tokens = self._decode_token_limit()
        if self.growing_static_suffix:
            self.dynamic_end = min(
                total_tokens,
                max(int(self.dynamic_start), int(self.growing_static_dynamic_end)),
            )
            self._rebuild_static_index_set(total_tokens)
            return
        suffix_keep = max(0, int(self.static_pattern_end))
        if suffix_keep > 0:
            self.dynamic_end = max(int(self.dynamic_start), total_tokens - suffix_keep)
        else:
            self.dynamic_end = max(int(self.dynamic_start), total_tokens)
        self.dynamic_end = min(self.dynamic_end, total_tokens)
        self._rebuild_static_index_set(total_tokens)

    def _online_graph_add_directed_edge(self, layer_idx: int, kv_hdx: int, src: int, dst: int) -> int:
        src = int(src)
        dst = int(dst)
        if src == dst:
            return 0
        if src < int(self.dynamic_start) or dst < int(self.dynamic_start):
            return 0
        per_head = self._online_graph_overlay[layer_idx][kv_hdx]
        cur = per_head.get(src)
        if cur is None:
            per_head[src] = [dst]
            cached = self._decode_cuda_overlay_graph_cache[layer_idx][kv_hdx]
            if cached is not None:
                counts_t, neighbors_t = cached
                if 0 <= src < int(counts_t.shape[0]):
                    counts_t[src] = 1
                    neighbors_t[src, 0] = int(dst)
            return 1
        if dst in cur:
            return 0
        if len(cur) >= int(self.online_graph_neighbor_cap):
            return 0
        cur.append(dst)
        cached = self._decode_cuda_overlay_graph_cache[layer_idx][kv_hdx]
        if cached is not None:
            counts_t, neighbors_t = cached
            count = len(cur)
            if 0 <= src < int(counts_t.shape[0]) and count <= int(neighbors_t.shape[1]):
                counts_t[src] = int(count)
                neighbors_t[src, count - 1] = int(dst)
        return 1

    def _record_online_graph_provenance(self, layer_idx: int, head_idx: int, token_ids):
        if not self.online_graph_enable:
            return
        if not isinstance(token_ids, (list, tuple)) or len(token_ids) == 0:
            return
        token_pos = int(self.input_length + self.decode_pos)
        total_tokens = max(self._decode_token_limit(), token_pos + 1)
        if token_pos < int(self.input_length) or token_pos >= int(self.cpu_keys[layer_idx].shape[1]):
            return
        kv_hdx = self._retrieval_head_to_kv_head(head_idx)
        keep = []
        seen = set()
        for tok in token_ids:
            tok = int(tok)
            if tok < int(self.dynamic_start) or tok >= total_tokens:
                continue
            if tok >= token_pos:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            keep.append(tok)
            if len(keep) >= int(self.online_graph_insert_k):
                break
        if not keep:
            return
        pending = self._online_graph_pending[layer_idx][kv_hdx]
        cur = pending.get(token_pos)
        if cur is None:
            pending[token_pos] = list(keep)
            self._online_graph_pending_order[layer_idx][kv_hdx].append(token_pos)
            return
        for tok in keep:
            if tok in cur:
                continue
            cur.append(tok)
            if len(cur) >= int(self.online_graph_insert_k):
                break

    def _record_online_graph_query_centroid_fullgpu(
        self,
        layer_idx: int,
        kv_hdx: int,
        q_batch_t: torch.Tensor,
        payloads,
        attn: torch.Tensor,
        dyn_ids_t: torch.Tensor = None,
        keep_counts_t: torch.Tensor = None,
    ):
        if not self.online_graph_enable or self.online_graph_signal != "query_centroid":
            return
        total_tokens = int(self._decode_token_limit())
        prefix_len = min(int(self.dynamic_start), total_tokens)
        suffix_start = min(max(int(self.dynamic_start), int(self.dynamic_end)), total_tokens)
        suffix_len = max(0, int(total_tokens - suffix_start))
        if suffix_len <= 0:
            return

        suffix_attn = attn[:, prefix_len:prefix_len + suffix_len].float()
        if suffix_attn.numel() <= 0:
            return
        topk = min(int(self.online_graph_query_topk), int(suffix_attn.shape[1]))
        if topk <= 0:
            return

        top_vals_t, top_idx_t = torch.topk(suffix_attn, k=topk, dim=1)
        top_vals = top_vals_t.detach().cpu()
        top_idx = top_idx_t.detach().cpu()
        q_cpu = q_batch_t.detach().float().cpu()

        for row_idx, payload in enumerate(payloads):
            if not isinstance(payload, dict):
                continue
            cand_tokens = []
            if dyn_ids_t is not None and keep_counts_t is not None:
                take = int(keep_counts_t[row_idx].item())
                if take > 0:
                    cand_tokens = [
                        int(tok) for tok in dyn_ids_t[row_idx, :take].detach().cpu().tolist()
                        if int(tok) >= 0
                    ]
            else:
                cand_tokens = payload.get("cpu_tokens") or []
                if not cand_tokens and "device_ids" in payload:
                    take = max(0, int(payload.get("final_count", 0)))
                    if take > 0:
                        cand_tokens = [
                            int(tok) for tok in payload["device_ids"][:take].detach().cpu().tolist()
                            if int(tok) >= 0
                        ]
            if not cand_tokens:
                continue
            cand_keep = []
            seen_cands = set()
            for tok in cand_tokens:
                tok = int(tok)
                if tok < int(self.dynamic_start) or tok >= int(total_tokens):
                    continue
                if tok in seen_cands:
                    continue
                seen_cands.add(tok)
                cand_keep.append(tok)
                if len(cand_keep) >= int(self.online_graph_query_cand_per_step):
                    break
            if not cand_keep:
                continue

            q_vec = q_cpu[row_idx]
            for j in range(int(top_vals.shape[1])):
                weight = float(top_vals[row_idx, j].item())
                if weight <= 0.0:
                    continue
                tok = int(suffix_start + int(top_idx[row_idx, j].item()))
                if tok < int(self.input_length) or tok >= int(total_tokens):
                    continue
                sum_map = self._online_graph_query_sum[layer_idx][kv_hdx]
                weight_map = self._online_graph_query_weight[layer_idx][kv_hdx]
                cand_map_all = self._online_graph_query_candidates[layer_idx][kv_hdx]
                cur_sum = sum_map.get(tok)
                if cur_sum is None:
                    sum_map[tok] = q_vec.clone().mul_(weight)
                else:
                    cur_sum.add_(q_vec, alpha=weight)
                weight_map[tok] = float(weight_map.get(tok, 0.0)) + weight
                cand_map = cand_map_all.get(tok)
                if cand_map is None:
                    cand_map = {}
                    cand_map_all[tok] = cand_map
                for cand in cand_keep:
                    if cand == tok:
                        continue
                    cand_map[int(cand)] = int(cand_map.get(int(cand), 0)) + 1

    def _select_online_graph_centroid_neighbors(self, layer_idx: int, kv_hdx: int, token_pos: int, neighbors):
        if self.online_graph_signal != "query_centroid":
            return neighbors
        q_sum = self._online_graph_query_sum[layer_idx][kv_hdx].pop(int(token_pos), None)
        weight = float(self._online_graph_query_weight[layer_idx][kv_hdx].pop(int(token_pos), 0.0))
        cand_counts = self._online_graph_query_candidates[layer_idx][kv_hdx].pop(int(token_pos), None)
        if q_sum is None or weight <= 0.0:
            return neighbors

        candidate_ids = []
        seen = set()
        if neighbors:
            for tok in neighbors:
                tok = int(tok)
                if tok in seen:
                    continue
                seen.add(tok)
                candidate_ids.append(tok)
        if cand_counts:
            ranked = sorted(cand_counts.items(), key=lambda item: (-int(item[1]), int(item[0])))
            cap = max(int(self.online_graph_neighbor_cap), int(self.online_graph_insert_k) * 4)
            for tok, _score in ranked:
                tok = int(tok)
                if tok in seen:
                    continue
                seen.add(tok)
                candidate_ids.append(tok)
                if len(candidate_ids) >= cap:
                    break
        if not candidate_ids:
            return neighbors

        key_cpu = self.cpu_keys[layer_idx][kv_hdx]
        cand_idx = torch.as_tensor(candidate_ids, dtype=torch.long, device="cpu")
        cand_keys = torch.index_select(key_cpu, 0, cand_idx).detach().float()
        cand_keys = self._score_transform_torch(cand_keys)
        q_vec = self._score_transform_torch((q_sum / max(weight, 1e-6)).unsqueeze(0).float())
        scores = torch.matmul(q_vec, cand_keys.transpose(0, 1)).squeeze(0)
        take = min(int(self.online_graph_insert_k), int(scores.numel()))
        if take <= 0:
            return neighbors
        vals, pos = torch.topk(scores, k=take, dim=0)
        out = []
        for i in range(int(pos.numel())):
            tok = int(candidate_ids[int(pos[i].item())])
            out.append(tok)
        return out if out else neighbors

    def _fullgpu_payload_token_ids(self, payload):
        if not isinstance(payload, dict) or "device_ids" not in payload:
            return []
        take = max(0, int(payload.get("final_count", 0)))
        if take <= 0:
            return []
        ids_t = payload["device_ids"][:take]
        if ids_t.numel() <= 0:
            return []
        return [int(tok) for tok in ids_t.detach().cpu().tolist() if int(tok) >= 0]

    def _fullgpu_payload_token_ids_group(self, payloads_by_head):
        out = {}
        valid = []
        max_cols = 0
        for hdx, payload in payloads_by_head:
            if not isinstance(payload, dict):
                continue
            if "cpu_tokens" in payload:
                out[int(hdx)] = [int(tok) for tok in payload["cpu_tokens"] if int(tok) >= 0]
                continue
            if "device_ids" not in payload:
                continue
            take = max(0, int(payload.get("final_count", 0)))
            if take <= 0:
                continue
            ids_t = payload["device_ids"]
            if ids_t.numel() <= 0:
                continue
            row = ids_t.reshape(-1)
            max_cols = max(max_cols, int(row.numel()))
            valid.append((int(hdx), row, take))
        if not valid or max_cols <= 0:
            return out

        rows = []
        for _, row, _ in valid:
            if int(row.numel()) < max_cols:
                pad = torch.full(
                    (max_cols - int(row.numel()),),
                    -1,
                    dtype=row.dtype,
                    device=row.device,
                )
                row = torch.cat((row, pad), dim=0)
            elif int(row.numel()) > max_cols:
                row = row[:max_cols]
            rows.append(row)

        stacked_cpu = torch.stack(rows, dim=0).detach().cpu()
        out = {}
        for row_idx, (hdx, _, take) in enumerate(valid):
            take_i = min(int(take), int(stacked_cpu.shape[1]))
            if take_i <= 0:
                continue
            out[int(hdx)] = [
                int(tok) for tok in stacked_cpu[row_idx, :take_i].tolist() if int(tok) >= 0
            ]
        return out

    def _flush_online_graph_pending(self):
        if not self.online_graph_enable:
            return
        total_tokens = self._decode_token_limit()
        if total_tokens <= int(self.input_length):
            return
        eligible_before = max(
            int(self.input_length),
            total_tokens - max(0, int(self.online_graph_defer)),
        )
        if eligible_before <= int(self.input_length):
            return
        start = time.perf_counter() if self.decode_profile else None
        inserted_nodes = 0
        inserted_edges = 0
        dirty_rows = {}
        for layer_idx in range(self.layer_num):
            for kv_hdx in range(self.kv_head):
                queue = self._online_graph_pending_order[layer_idx][kv_hdx]
                cursor = int(self._online_graph_pending_cursor[layer_idx][kv_hdx])
                pending = self._online_graph_pending[layer_idx][kv_hdx]
                while cursor < len(queue):
                    token_pos = int(queue[cursor])
                    if token_pos >= eligible_before:
                        break
                    cursor += 1
                    neighbors = pending.pop(token_pos, None)
                    neighbors = self._select_online_graph_centroid_neighbors(
                        layer_idx,
                        kv_hdx,
                        token_pos,
                        neighbors,
                    )
                    if not neighbors:
                        continue
                    inserted_nodes += 1
                    pair_edges_before = inserted_edges
                    for nb in neighbors:
                        nb = int(nb)
                        if nb < int(self.dynamic_start) or nb >= total_tokens:
                            continue
                        added = self._online_graph_add_directed_edge(layer_idx, kv_hdx, token_pos, nb)
                        inserted_edges += added
                        if added:
                            dirty_rows.setdefault((int(layer_idx), int(kv_hdx)), set()).add(int(token_pos))
                        if self.online_graph_bidirectional:
                            added_rev = self._online_graph_add_directed_edge(layer_idx, kv_hdx, nb, token_pos)
                            inserted_edges += added_rev
                            if added_rev:
                                dirty_rows.setdefault((int(layer_idx), int(kv_hdx)), set()).add(int(nb))
                    if inserted_edges > pair_edges_before:
                        dirty_rows.setdefault((int(layer_idx), int(kv_hdx)), set())
                self._online_graph_pending_cursor[layer_idx][kv_hdx] = cursor
        if self.decode_backend == "roar_cuda_fullgpu":
            for (layer_idx, kv_hdx), rows in dirty_rows.items():
                build_start = time.perf_counter() if self.decode_profile else None
                self._update_decode_overlay_rows_cuda_device(layer_idx, kv_hdx, rows)
                if self.decode_profile:
                    self._decode_profile_stats["online_overlay_build_cpu_sec"] += (
                        time.perf_counter() - build_start
                    )
        if self.decode_profile:
            self._decode_profile_stats["online_update_sec"] += (time.perf_counter() - start)
            self._decode_profile_stats["online_insert_nodes"] += int(inserted_nodes)
            self._decode_profile_stats["online_insert_edges"] += int(inserted_edges)

    def _is_dynamic_generated_token(self, tok: int) -> bool:
        tok = int(tok)
        return tok >= int(self.input_length) and tok < int(self.dynamic_end)

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
        if self.online_graph_enable or self.online_dynamic_range:
            return False
        if self.decode_backend in {
            "python",
            "roar_cuda",
            "roar_cuda_v2",
            "roar_cuda_kernel",
            "roar_cuda_fullgpu",
            "roar_cuda_frontier",
            "roar_cuda_beam",
        }:
            return False
        available = roargraph_cpp_available()
        self._roar_cpp_available = bool(available)
        if not available:
            if self.decode_backend == "auto":
                return False
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

    def _get_decode_key_tensor_cuda(self, ldx: int, hdx: int):
        cached = self._decode_cuda_key_cache[ldx][hdx]
        if cached is not None:
            return cached
        device = self.layer_mapping[str(ldx)]
        total_len = int(self.cpu_keys[ldx].shape[1])
        key_tensor = torch.zeros((total_len, self.head_dim), dtype=torch.float32, device=device)
        prefill_keys = self.cpu_keys[ldx][hdx, :self.input_length, :].detach().to(device, non_blocking=True).float()
        prefill_keys = self._score_transform_torch(prefill_keys)
        key_tensor[:self.input_length, :].copy_(prefill_keys, non_blocking=True)
        self._decode_cuda_key_cache[ldx][hdx] = key_tensor.contiguous()
        return self._decode_cuda_key_cache[ldx][hdx]

    def _get_decode_attn_key_tensor_cuda(self, ldx: int, hdx: int):
        cached = self._decode_cuda_attn_key_cache[ldx][hdx]
        if cached is not None:
            return cached
        device = self.layer_mapping[str(ldx)]
        total_len = int(self.cpu_keys[ldx].shape[1])
        key_tensor = torch.zeros((total_len, self.head_dim), dtype=self.dtype, device=device)
        prefill_keys = self.cpu_keys[ldx][hdx, :self.input_length, :].detach().to(device, non_blocking=True)
        key_tensor[:self.input_length, :].copy_(prefill_keys, non_blocking=True)
        self._decode_cuda_attn_key_cache[ldx][hdx] = key_tensor.contiguous()
        norm_tensor = torch.zeros((total_len,), dtype=torch.float32, device=device)
        prefixmax_tensor = torch.zeros((total_len,), dtype=torch.float32, device=device)
        if int(self.input_length) > 0:
            prefill_norms = torch.linalg.vector_norm(prefill_keys.float(), dim=-1)
            norm_tensor[:self.input_length].copy_(prefill_norms, non_blocking=True)
            if int(self.input_length) > int(self.dynamic_start):
                dyn_prefill = prefill_norms[int(self.dynamic_start):int(self.input_length)]
                if int(dyn_prefill.numel()) > 0:
                    prefixmax_tensor[int(self.dynamic_start):int(self.input_length)] = torch.cummax(
                        dyn_prefill,
                        dim=0,
                    ).values
        dyn_start = int(self.dynamic_start)
        dyn_end = min(int(self.dynamic_end), int(self.input_length))
        sum_tensor = torch.zeros((self.head_dim,), dtype=torch.float32, device=device)
        sumsq_tensor = torch.zeros((self.head_dim,), dtype=torch.float32, device=device)
        if dyn_end > dyn_start:
            dyn_prefill_keys = prefill_keys[dyn_start:dyn_end].float()
            if int(dyn_prefill_keys.numel()) > 0:
                sum_tensor.copy_(dyn_prefill_keys.sum(dim=0), non_blocking=True)
                sumsq_tensor.copy_((dyn_prefill_keys * dyn_prefill_keys).sum(dim=0), non_blocking=True)
        self._decode_cuda_attn_key_norm_cache[ldx][hdx] = norm_tensor
        self._decode_cuda_attn_key_prefixmax_cache[ldx][hdx] = prefixmax_tensor
        self._decode_cuda_attn_key_sum_cache[ldx][hdx] = sum_tensor
        self._decode_cuda_attn_key_sumsq_cache[ldx][hdx] = sumsq_tensor
        return self._decode_cuda_attn_key_cache[ldx][hdx]

    def _get_decode_value_tensor_cuda(self, ldx: int, hdx: int):
        cached = self._decode_cuda_value_cache[ldx][hdx]
        if cached is not None:
            return cached
        device = self.layer_mapping[str(ldx)]
        total_len = int(self.cpu_values[ldx].shape[1])
        value_tensor = torch.zeros((total_len, self.head_dim), dtype=self.dtype, device=device)
        prefill_values = self.cpu_values[ldx][hdx, :self.input_length, :].detach().to(device, non_blocking=True)
        value_tensor[:self.input_length, :].copy_(prefill_values, non_blocking=True)
        self._decode_cuda_value_cache[ldx][hdx] = value_tensor.contiguous()
        return self._decode_cuda_value_cache[ldx][hdx]

    def _get_decode_graph_tensors_cuda(self, ldx: int, hdx: int):
        cached = self._decode_cuda_graph_cache[ldx][hdx]
        if cached is not None:
            return cached
        graph = self.graphs[ldx][hdx]
        if not (isinstance(graph, tuple) and len(graph) >= 2):
            return None
        offsets = np.asarray(graph[0], dtype=np.uint32)
        neighbors = graph[1]
        total_len = int(self.cpu_keys[ldx].shape[1])
        if offsets.shape[0] < (total_len + 1):
            ext_offsets = np.empty((total_len + 1,), dtype=np.uint32)
            ext_offsets[:offsets.shape[0]] = offsets
            last = int(offsets[-1]) if offsets.shape[0] > 0 else 0
            ext_offsets[offsets.shape[0]:] = np.uint32(last)
            offsets = ext_offsets
        offsets_t = torch.as_tensor(np.ascontiguousarray(offsets, dtype=np.uint32), dtype=torch.int64, device="cpu")
        neighbors_t = torch.as_tensor(np.ascontiguousarray(neighbors, dtype=np.int32), dtype=torch.int32, device="cpu")
        self._decode_cuda_graph_cache[ldx][hdx] = (offsets_t, neighbors_t)
        return self._decode_cuda_graph_cache[ldx][hdx]

    def _get_decode_graph_tensors_cuda_device(self, ldx: int, hdx: int):
        cached = self._decode_cuda_graph_device_cache[ldx][hdx]
        if cached is not None:
            return cached
        graph_tensors = self._get_decode_graph_tensors_cuda(ldx, hdx)
        if graph_tensors is None:
            return None
        device = self.layer_mapping[str(ldx)]
        offsets_t, neighbors_t = graph_tensors
        offsets_dev = offsets_t.to(device=device, non_blocking=True)
        neighbors_dev = neighbors_t.to(device=device, non_blocking=True)
        self._decode_cuda_graph_device_cache[ldx][hdx] = (
            offsets_dev.contiguous(),
            neighbors_dev.contiguous(),
        )
        return self._decode_cuda_graph_device_cache[ldx][hdx]

    def _build_decode_overlay_graph_tensors(self, ldx: int, hdx: int):
        total_len = int(self.cpu_keys[ldx].shape[1])
        cap = int(self.online_graph_neighbor_cap)
        counts_t = torch.zeros((total_len,), dtype=torch.int32, device="cpu")
        neighbors_t = torch.full((total_len, cap), -1, dtype=torch.int32, device="cpu")
        overlay = self._online_graph_overlay[ldx][hdx]
        for src, dsts in overlay.items():
            src_i = int(src)
            if src_i < 0 or src_i >= total_len:
                continue
            valid = []
            for dst in dsts:
                dst_i = int(dst)
                if dst_i < 0 or dst_i >= total_len:
                    continue
                valid.append(dst_i)
                if len(valid) >= cap:
                    break
            if not valid:
                continue
            count = min(len(valid), cap)
            counts_t[src_i] = int(count)
            neighbors_t[src_i, :count] = torch.tensor(valid[:count], dtype=torch.int32, device="cpu")
        self._decode_cuda_overlay_graph_cache[ldx][hdx] = (counts_t, neighbors_t)
        self._decode_cuda_overlay_graph_device_cache[ldx][hdx] = None
        return self._decode_cuda_overlay_graph_cache[ldx][hdx]

    def _get_decode_overlay_graph_tensors_cuda(self, ldx: int, hdx: int):
        cached = self._decode_cuda_overlay_graph_cache[ldx][hdx]
        if cached is not None:
            return cached
        return self._build_decode_overlay_graph_tensors(ldx, hdx)

    def _get_decode_overlay_graph_tensors_cuda_device(self, ldx: int, hdx: int):
        cached = self._decode_cuda_overlay_graph_device_cache[ldx][hdx]
        if cached is not None:
            return cached
        counts_t, neighbors_t = self._get_decode_overlay_graph_tensors_cuda(ldx, hdx)
        device = self.layer_mapping[str(ldx)]
        counts_dev = counts_t.to(device=device, non_blocking=True)
        neighbors_dev = neighbors_t.to(device=device, non_blocking=True)
        self._decode_cuda_overlay_graph_device_cache[ldx][hdx] = (
            counts_dev.contiguous(),
            neighbors_dev.contiguous(),
        )
        return self._decode_cuda_overlay_graph_device_cache[ldx][hdx]

    def _update_decode_overlay_rows_cuda_device(self, ldx: int, hdx: int, rows):
        row_set = {int(row) for row in rows if int(row) >= 0}
        if not row_set:
            return
        counts_t, neighbors_t = self._get_decode_overlay_graph_tensors_cuda(ldx, hdx)
        counts_dev, neighbors_dev = self._get_decode_overlay_graph_tensors_cuda_device(ldx, hdx)
        total_len = int(counts_t.shape[0])
        ordered_rows = sorted(row for row in row_set if row < total_len)
        if not ordered_rows:
            return

        device = self.layer_mapping[str(ldx)]
        upload_start = time.perf_counter() if self.decode_profile else None
        row_idx_dev = torch.as_tensor(ordered_rows, dtype=torch.int64, device=device)
        row_counts_dev = counts_t.index_select(0, torch.as_tensor(ordered_rows, dtype=torch.long, device="cpu")).to(device=device, non_blocking=True)
        row_neighbors_dev = neighbors_t.index_select(0, torch.as_tensor(ordered_rows, dtype=torch.long, device="cpu")).to(device=device, non_blocking=True)
        counts_dev.index_copy_(0, row_idx_dev, row_counts_dev)
        neighbors_dev.index_copy_(0, row_idx_dev, row_neighbors_dev)
        if self.decode_profile:
            self._decode_profile_stats["online_overlay_h2d_sec"] += (
                time.perf_counter() - upload_start
            )

    def _get_decode_graph_max_degree(self, ldx: int, hdx: int) -> int:
        cached = self._decode_cuda_graph_degree_cache[ldx][hdx]
        if cached is not None:
            return int(cached)
        graph = self.graphs[ldx][hdx]
        if not (isinstance(graph, tuple) and len(graph) >= 2):
            return 1
        offsets = np.asarray(graph[0], dtype=np.uint32)
        if offsets.shape[0] <= 1:
            self._decode_cuda_graph_degree_cache[ldx][hdx] = 1
            return 1
        degree = offsets[1:].astype(np.int64) - offsets[:-1].astype(np.int64)
        max_degree = int(degree.max()) if degree.size > 0 else 1
        max_degree = max(1, max_degree)
        self._decode_cuda_graph_degree_cache[ldx][hdx] = max_degree
        return max_degree

    def _prepare_decode_cuda_caches(self):
        if self.decode_backend not in {
            "roar_cuda",
            "roar_cuda_v2",
            "roar_cuda_kernel",
            "roar_cuda_fullgpu",
            "roar_cuda_frontier",
            "roar_cuda_beam",
        }:
            return
        for ldx in range(self.layer_num):
            for kv_hdx in range(self.kv_head):
                self._get_decode_key_tensor_cuda(ldx, kv_hdx)
                self._get_decode_graph_tensors_cuda(ldx, kv_hdx)
                if self.decode_backend in {"roar_cuda_beam", "roar_cuda_frontier", "roar_cuda_kernel", "roar_cuda_fullgpu"}:
                    self._get_decode_graph_tensors_cuda_device(ldx, kv_hdx)
                if self.decode_backend == "roar_cuda_fullgpu":
                    self._get_decode_attn_key_tensor_cuda(ldx, kv_hdx)
                    self._get_decode_value_tensor_cuda(ldx, kv_hdx)
                    self._get_decode_overlay_graph_tensors_cuda_device(ldx, kv_hdx)

    def _prepare_decode_cuda_seed_caches(self):
        if self.decode_backend != "roar_cuda_fullgpu":
            return
        hub_cap = max(1, int(self.seed_hub_k))
        prev_cap = max(1, int(self.seed_prev_k))
        for ldx in range(self.layer_num):
            device = self.layer_mapping[str(ldx)]
            hub_ids = torch.full((self.kv_head, hub_cap), -1, dtype=torch.int32, device=device)
            for kv_hdx in range(self.kv_head):
                seeds = self.hub_seeds[ldx][kv_hdx][:hub_cap]
                if seeds:
                    hub_ids[kv_hdx, :len(seeds)] = torch.as_tensor(
                        seeds,
                        dtype=torch.int32,
                        device=device,
                    )
            self._decode_cuda_hub_seed_ids[ldx] = hub_ids
            self._decode_cuda_prev_seed_ids[ldx] = torch.full(
                (self.retrieval_heads, prev_cap),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self._decode_cuda_prev_seed_counts[ldx] = torch.zeros(
                (self.retrieval_heads,),
                dtype=torch.int32,
                device=device,
            )

    def _get_dynamic_attn_max_norm_fullgpu(self, ldx: int, hdx: int):
        prefixmax_t = self._decode_cuda_attn_key_prefixmax_cache[ldx][hdx]
        if prefixmax_t is None:
            self._get_decode_attn_key_tensor_cuda(ldx, hdx)
            prefixmax_t = self._decode_cuda_attn_key_prefixmax_cache[ldx][hdx]
        dyn_end = int(self.dynamic_end)
        if prefixmax_t is None or dyn_end <= int(self.dynamic_start):
            device = self.layer_mapping[str(ldx)]
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        idx = min(max(0, dyn_end - 1), int(prefixmax_t.shape[0]) - 1)
        return prefixmax_t[idx]

    def _get_dynamic_attn_moment_tensors_fullgpu(self, ldx: int, hdx: int):
        sum_t = self._decode_cuda_attn_key_sum_cache[ldx][hdx]
        sumsq_t = self._decode_cuda_attn_key_sumsq_cache[ldx][hdx]
        if sum_t is None or sumsq_t is None:
            self._get_decode_attn_key_tensor_cuda(ldx, hdx)
            sum_t = self._decode_cuda_attn_key_sum_cache[ldx][hdx]
            sumsq_t = self._decode_cuda_attn_key_sumsq_cache[ldx][hdx]
        if sum_t is None or sumsq_t is None:
            device = self.layer_mapping[str(ldx)]
            zero = torch.zeros((self.head_dim,), dtype=torch.float32, device=device)
            return zero, zero
        return sum_t, sumsq_t

    def _get_dynamic_tail_value_fullgpu(self, layer_idx: int, kv_hdx: int):
        value_t = self._get_decode_value_tensor_cuda(layer_idx, kv_hdx)
        if value_t is None:
            device = self.layer_mapping[str(layer_idx)]
            return torch.zeros((self.head_dim,), dtype=torch.float32, device=device)
        dyn_start = int(self.dynamic_start)
        dyn_end = int(self.dynamic_end)
        if dyn_end <= dyn_start:
            return torch.zeros((self.head_dim,), dtype=torch.float32, device=value_t.device)
        dyn_slice = value_t[dyn_start:dyn_end]
        if int(dyn_slice.shape[0]) <= 0:
            return torch.zeros((self.head_dim,), dtype=torch.float32, device=value_t.device)
        return dyn_slice.float().mean(dim=0)

    def _update_dynamic_attn_moments_fullgpu(self, old_dynamic_end: int, new_dynamic_end: int):
        if self.decode_backend != "roar_cuda_fullgpu":
            return
        if self.dynamic_budget_prior != "moment_diag":
            return
        start = max(int(self.dynamic_start), int(old_dynamic_end))
        end = max(start, int(new_dynamic_end))
        if end <= start:
            return
        for ldx in range(self.layer_num):
            for kv_hdx in range(self.kv_head):
                attn_key_cache = self._decode_cuda_attn_key_cache[ldx][kv_hdx]
                sum_cache = self._decode_cuda_attn_key_sum_cache[ldx][kv_hdx]
                sumsq_cache = self._decode_cuda_attn_key_sumsq_cache[ldx][kv_hdx]
                if attn_key_cache is None or sum_cache is None or sumsq_cache is None:
                    continue
                rows = attn_key_cache[start:end].float()
                if int(rows.numel()) <= 0:
                    continue
                sum_cache.add_(rows.sum(dim=0))
                sumsq_cache.add_((rows * rows).sum(dim=0))

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
            self._refresh_decode_dynamic_state()

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
        self._prepare_decode_cuda_caches()
        self._prepare_decode_cuda_seed_caches()
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
            "final_outputs_total",
            "adaptive_outputs_total",
            "kernel_round_total",
            "forced_seed_total",
            "stop_frontier_empty",
            "stop_max_visits",
            "stop_candidate_cap",
            "stop_stability_gap",
            "stop_empty_init",
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
        other_setup = float(stats["other_setup_sec"])
        other_state_prep = float(stats["other_state_prep_sec"])
        other_profile_accum = float(stats["other_profile_accum_sec"])
        other_group_bookkeeping = float(stats["other_group_bookkeeping_sec"])
        other_output = float(stats["other_output_sec"])
        other_accounted = (
            other_setup
            + other_state_prep
            + other_profile_accum
            + other_group_bookkeeping
            + other_output
        )
        other_misc = max(0.0, other - other_accounted)
        heads = int(stats["heads"])
        visited_total = int(stats["visited_total"])
        candidates_total = int(stats["candidates_total"])
        final_outputs_total = int(stats["final_outputs_total"])
        adaptive_outputs_total = int(stats["adaptive_outputs_total"])
        kernel_round_total = int(stats["kernel_round_total"])
        forced_seed_total = int(stats["forced_seed_total"])
        search_space_total = int(stats["search_space_total"])
        search_space_heads = int(stats["search_space_heads"])
        visited_ratio_sum = float(stats["visited_ratio_sum"])
        visited_ratio_count = int(stats["visited_ratio_count"])
        online_update_sec = float(stats["online_update_sec"])
        online_provenance_d2h_sec = float(stats["online_provenance_d2h_sec"])
        online_overlay_build_cpu_sec = float(stats["online_overlay_build_cpu_sec"])
        online_overlay_h2d_sec = float(stats["online_overlay_h2d_sec"])
        online_insert_nodes = int(stats["online_insert_nodes"])
        online_insert_edges = int(stats["online_insert_edges"])
        online_overlay_edges = int(stats["online_overlay_edges"])
        online_generated_hits = int(stats["online_generated_hits"])
        online_generated_head_hits = int(stats["online_generated_head_hits"])
        adaptive_total_sec = float(stats["adaptive_total_sec"])
        adaptive_upper_bound_sec = float(stats["adaptive_upper_bound_sec"])
        adaptive_static_logz_sec = float(stats["adaptive_static_logz_sec"])
        adaptive_candidate_score_sec = float(stats["adaptive_candidate_score_sec"])
        adaptive_sort_sec = float(stats["adaptive_sort_sec"])
        adaptive_select_sec = float(stats["adaptive_select_sec"])
        adaptive_reorder_sec = float(stats["adaptive_reorder_sec"])

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
        outputs_per_head = (
            float(final_outputs_total) / float(heads)
            if heads > 0 else 0.0
        )
        adaptive_outputs_per_head = (
            float(adaptive_outputs_total) / float(heads)
            if heads > 0 else 0.0
        )
        rounds_per_head = (
            float(kernel_round_total) / float(heads)
            if heads > 0 else 0.0
        )
        forced_per_head = (
            float(forced_seed_total) / float(heads)
            if heads > 0 else 0.0
        )
        online_generated_per_head = (
            float(online_generated_hits) / float(heads)
            if heads > 0 else 0.0
        )
        online_generated_head_rate = (
            100.0 * float(online_generated_head_hits) / float(heads)
            if heads > 0 else 0.0
        )
        stop_counts = {
            "frontier_empty": int(stats["stop_frontier_empty"]),
            "max_visits": int(stats["stop_max_visits"]),
            "candidate_cap": int(stats["stop_candidate_cap"]),
            "stability_gap": int(stats["stop_stability_gap"]),
            "empty_init": int(stats["stop_empty_init"]),
        }

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
            f"other={other:.3f}s ({pct(other):.1f}%) "
            f"[setup={other_setup:.3f}s, state={other_state_prep:.3f}s, "
            f"profile={other_profile_accum:.3f}s, group={other_group_bookkeeping:.3f}s, "
            f"output={other_output:.3f}s, misc={other_misc:.3f}s] | "
            f"visited_total={visited_total} "
            f"candidates_total={candidates_total} | "
            f"traversal=[space/head={search_space_per_head:.1f}, "
            f"visited/head={visited_per_head:.1f}, "
            f"visit_rate={100.0 * visit_rate_weighted:.2f}%, "
            f"visit_rate_mean={100.0 * visit_rate_mean:.2f}%, "
            f"prune_rate={100.0 * prune_rate_weighted:.2f}%, "
            f"cand/visit={cand_per_visit:.2f}x, "
            f"out/head={outputs_per_head:.1f}, "
            f"adaptive_out/head={adaptive_outputs_per_head:.1f}, "
            f"rounds/head={rounds_per_head:.1f}, "
            f"forced/head={forced_per_head:.1f}, "
            f"stop={stop_counts}]"
        )
        if self.online_dynamic_range or self.online_graph_enable:
            msg += (
                f" | online=[dynamic_end={int(self.dynamic_end)}, "
                f"update={online_update_sec:.3f}s, "
                f"d2h={online_provenance_d2h_sec:.3f}s, "
                f"build={online_overlay_build_cpu_sec:.3f}s, "
                f"h2d={online_overlay_h2d_sec:.3f}s, "
                f"nodes={online_insert_nodes}, edges={online_insert_edges}, "
                f"overlay_edges={online_overlay_edges}, "
                f"aged_gen/head={online_generated_per_head:.2f}, "
                f"aged_gen_head_rate={online_generated_head_rate:.1f}%]"
            )
        if adaptive_total_sec > 0.0:
            msg += (
                f" | adaptive=[total={adaptive_total_sec:.3f}s, "
                f"upper={adaptive_upper_bound_sec:.3f}s, "
                f"static={adaptive_static_logz_sec:.3f}s, "
                f"score={adaptive_candidate_score_sec:.3f}s, "
                f"sort={adaptive_sort_sec:.3f}s, "
                f"select={adaptive_select_sec:.3f}s, "
                f"reorder={adaptive_reorder_sec:.3f}s]"
            )
        if reset:
            self.reset_decode_profile()
        return msg

    def _accumulate_decode_retrieve_profile(self, retrieve_profile):
        if not self.decode_profile or retrieve_profile is None:
            return
        accum_start = time.perf_counter()
        self._decode_profile_stats["retrieve_total_sec"] += float(retrieve_profile["total_sec"])
        self._decode_profile_stats["retrieve_seed_sec"] += float(retrieve_profile["seed_sec"])
        self._decode_profile_stats["retrieve_graph_sec"] += float(retrieve_profile["graph_sec"])
        self._decode_profile_stats["retrieve_rerank_sec"] += float(retrieve_profile["rerank_sec"])
        self._decode_profile_stats["retrieve_finalize_sec"] += float(retrieve_profile["finalize_sec"])
        self._decode_profile_stats["visited_total"] += int(retrieve_profile["visited"])
        self._decode_profile_stats["candidates_total"] += int(retrieve_profile["candidates"])
        self._decode_profile_stats["final_outputs_total"] += int(retrieve_profile.get("final_outputs", 0))
        self._decode_profile_stats["adaptive_outputs_total"] += int(
            retrieve_profile.get("adaptive_final_outputs", retrieve_profile.get("final_outputs", 0))
        )
        self._decode_profile_stats["kernel_round_total"] += int(retrieve_profile.get("kernel_rounds", 0))
        self._decode_profile_stats["forced_seed_total"] += int(retrieve_profile.get("forced_seeds", 0))
        stop_reason = str(retrieve_profile.get("stop_reason", ""))
        if stop_reason == "frontier_empty":
            self._decode_profile_stats["stop_frontier_empty"] += 1
        elif stop_reason == "max_visits":
            self._decode_profile_stats["stop_max_visits"] += 1
        elif stop_reason == "candidate_cap":
            self._decode_profile_stats["stop_candidate_cap"] += 1
        elif stop_reason == "stability_gap":
            self._decode_profile_stats["stop_stability_gap"] += 1
        elif stop_reason == "empty_init":
            self._decode_profile_stats["stop_empty_init"] += 1
        search_space = max(0, int(retrieve_profile.get("search_space", 0)))
        self._decode_profile_stats["search_space_total"] += search_space
        if search_space > 0:
            self._decode_profile_stats["search_space_heads"] += 1
        self._decode_profile_stats["visited_ratio_sum"] += float(
            retrieve_profile.get("visited_ratio", 0.0)
        )
        self._decode_profile_stats["visited_ratio_count"] += 1
        self._decode_profile_stats["online_overlay_edges"] += int(retrieve_profile.get("online_overlay_edges", 0))
        self._decode_profile_stats["online_generated_hits"] += int(retrieve_profile.get("online_generated_hits", 0))
        self._decode_profile_stats["online_generated_head_hits"] += int(retrieve_profile.get("online_generated_any", 0))
        self._decode_profile_stats["other_profile_accum_sec"] += (time.perf_counter() - accum_start)

    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """
        Append newly generated token to CPU KV and update static suffix window.
        """
        # key_states/value_states: [bs, 1, kv_head, head_dim]
        pos = self.input_length + self.decode_pos
        if pos < self.cpu_keys[layer_idx].shape[1]:
            self.cpu_keys[layer_idx][:, pos:pos + 1, :].copy_(key_states[0].transpose(0, 1), non_blocking=True)
            self.cpu_values[layer_idx][:, pos:pos + 1, :].copy_(value_states[0].transpose(0, 1), non_blocking=True)
            if self.decode_backend == "roar_cuda_fullgpu":
                device = self.layer_mapping[str(layer_idx)]
                key_row = key_states[0, 0].to(device, non_blocking=True)
                value_row = value_states[0, 0].to(device, non_blocking=True)
                score_row = self._score_transform_torch(key_row.float())
                attn_norm_row = torch.linalg.vector_norm(key_row.float(), dim=-1)
                for kv_hdx in range(self.kv_head):
                    key_cache = self._decode_cuda_key_cache[layer_idx][kv_hdx]
                    if key_cache is not None:
                        key_cache[pos:pos + 1, :].copy_(score_row[kv_hdx:kv_hdx + 1], non_blocking=True)
                    attn_key_cache = self._decode_cuda_attn_key_cache[layer_idx][kv_hdx]
                    if attn_key_cache is not None:
                        attn_key_cache[pos:pos + 1, :].copy_(key_row[kv_hdx:kv_hdx + 1], non_blocking=True)
                    norm_cache = self._decode_cuda_attn_key_norm_cache[layer_idx][kv_hdx]
                    prefixmax_cache = self._decode_cuda_attn_key_prefixmax_cache[layer_idx][kv_hdx]
                    if norm_cache is not None:
                        norm_cache[pos] = attn_norm_row[kv_hdx]
                    if prefixmax_cache is not None:
                        if pos < int(self.dynamic_start):
                            prefixmax_cache[pos] = 0.0
                        elif pos == int(self.dynamic_start):
                            prefixmax_cache[pos] = attn_norm_row[kv_hdx]
                        else:
                            prev = prefixmax_cache[pos - 1]
                            prefixmax_cache[pos] = torch.maximum(prev, attn_norm_row[kv_hdx])
                    value_cache = self._decode_cuda_value_cache[layer_idx][kv_hdx]
                    if value_cache is not None:
                        value_cache[pos:pos + 1, :].copy_(value_row[kv_hdx:kv_hdx + 1], non_blocking=True)

        # Update static suffix window (shift left, append new token)
        if self.static_pattern_end > 0 and not self.growing_static_suffix:
            suffix = self.static_gpu_keys[layer_idx][:, self.static_pattern_start:, :]
            suffix_v = self.static_gpu_values[layer_idx][:, self.static_pattern_start:, :]
            suffix = torch.roll(suffix, shifts=-1, dims=1)
            suffix_v = torch.roll(suffix_v, shifts=-1, dims=1)
            suffix[:, -1, :] = key_states[0, 0].to(self.layer_mapping[str(layer_idx)])
            suffix_v[:, -1, :] = value_states[0, 0].to(self.layer_mapping[str(layer_idx)])
            self.static_gpu_keys[layer_idx][:, self.static_pattern_start:, :] = suffix
            self.static_gpu_values[layer_idx][:, self.static_pattern_start:, :] = suffix_v

        if layer_idx == self.layer_num - 1:
            old_dynamic_end = int(self.dynamic_end)
            self.decode_pos += 1
            self.context += 1
            self._refresh_decode_dynamic_state()
            self._update_dynamic_attn_moments_fullgpu(old_dynamic_end, int(self.dynamic_end))
            self._flush_online_graph_pending()

        return None, None

    def _make_decode_retrieve_profile(self):
        if not self.decode_profile:
            return None, None
        total_start = time.perf_counter()
        profile = {
            "total_sec": 0.0,
            "seed_sec": 0.0,
            "graph_sec": 0.0,
            "rerank_sec": 0.0,
            "finalize_sec": 0.0,
            "visited": 0,
            "candidates": 0,
            "final_outputs": 0,
            "kernel_rounds": 0,
            "forced_seeds": 0,
            "frontier_at_stop": 0,
            "search_space": 0,
            "visited_ratio": 0.0,
            "stop_reason": "n/a",
        }
        return total_start, profile

    def _finish_decode_retrieve_profile(self, total_start, profile, stop_reason: str, visited: int, candidates: int):
        if profile is None:
            return None
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
        return profile

    def _decode_stop_reason_from_code(self, code: int) -> str:
        mapping = {
            0: "frontier_empty",
            1: "max_visits",
            2: "candidate_cap",
            3: "stability_gap",
            4: "empty_init",
            5: "adaptive_bound",
        }
        return mapping.get(int(code), f"code_{int(code)}")

    def _maybe_compare_fullgpu_reference(self, layer_idx: int, kv_hdx: int, q, head_count: int, fg_results: dict):
        if (
            not self.fullgpu_ab
            or self._fullgpu_ab_done
            or self.decode_backend != "roar_cuda_fullgpu"
            or layer_idx != self.fullgpu_ab_layer
            or self.decode_pos != self.fullgpu_ab_step
        ):
            return

        ref_states = []
        head_ids = []
        for local_h in range(self.group_size):
            hdx = kv_hdx * self.group_size + local_h
            if hdx >= head_count:
                break
            q_group = q[hdx]
            state = self._prepare_decode_seed_state(
                layer_idx,
                hdx,
                q_group,
                defer_seed_scoring=True,
            )
            ref_states.append(state)
            head_ids.append(hdx)
        if not ref_states:
            return

        ref_results = self._retrieve_tokens_roar_cuda_v2_group(
            layer_idx,
            kv_hdx,
            ref_states,
            update_decode_state=False,
            enforce_seed_floor=True,
        )

        for hdx in head_ids:
            ref_ids, ref_profile = ref_results.get(hdx, ([], None))
            fg_payload, fg_profile = fg_results.get(hdx, ({}, None))
            if isinstance(fg_payload, dict) and "device_ids" in fg_payload:
                fg_mask = fg_payload["device_mask"]
                fg_ids = fg_payload["device_ids"][fg_mask].detach().cpu().tolist()
            else:
                fg_ids = list(fg_payload) if fg_payload else []
            ref_ids = list(ref_ids)
            fg_set = set(int(x) for x in fg_ids)
            ref_set = set(int(x) for x in ref_ids)
            union = len(fg_set | ref_set)
            inter = len(fg_set & ref_set)
            jaccard = (float(inter) / float(union)) if union > 0 else 1.0
            prefix_n = min(16, len(fg_ids), len(ref_ids))
            prefix_same = sum(
                1 for i in range(prefix_n)
                if int(fg_ids[i]) == int(ref_ids[i])
            )
            prefix_rate = (float(prefix_same) / float(prefix_n)) if prefix_n > 0 else 1.0
            print(
                "[RetrievalAttention][fullgpu_ab] "
                f"step={self.decode_pos} layer={layer_idx} kv_head={kv_hdx} head={hdx} "
                f"fg_n={len(fg_ids)} ref_n={len(ref_ids)} "
                f"jaccard={jaccard:.4f} prefix@{prefix_n}={prefix_rate:.4f} "
                f"fg_stop={fg_profile.get('stop_reason', 'n/a') if isinstance(fg_profile, dict) else 'n/a'} "
                f"ref_stop={ref_profile.get('stop_reason', 'n/a') if isinstance(ref_profile, dict) else 'n/a'}"
            )
        self._fullgpu_ab_done = True

    def _should_log_fullgpu_kernel_debug(self, layer_idx: int) -> bool:
        if not self.fullgpu_kernel_debug or self.decode_backend != "roar_cuda_fullgpu":
            return False
        if self.fullgpu_kernel_debug_layer >= 0 and layer_idx != self.fullgpu_kernel_debug_layer:
            return False
        if self.fullgpu_kernel_debug_step >= 0 and self.decode_pos != self.fullgpu_kernel_debug_step:
            return False
        return True

    def _maybe_log_fullgpu_kernel_debug(self, layer_idx: int, kv_hdx: int, out_debug, graph_elapsed: float):
        if not self._should_log_fullgpu_kernel_debug(layer_idx):
            return
        if out_debug is None or getattr(out_debug, "numel", lambda: 0)() <= 0:
            return

        rows = out_debug.tolist()
        if not rows:
            return

        phase_cols = {
            "init": 4,
            "score": 5,
            "merge": 6,
            "expand": 7,
            "finalize": 8,
        }
        work_cols = {
            "scored": 9,
            "edges": 10,
            "accepted": 11,
            "frontier_expanded": 12,
            "find_probes": 13,
            "overlay_edges": 14,
        }

        phase_totals = {name: 0 for name in phase_cols}
        work_totals = {name: 0 for name in work_cols}
        for row in rows:
            for name, col in phase_cols.items():
                if len(row) > col:
                    phase_totals[name] += int(row[col])
            for name, col in work_cols.items():
                if len(row) > col:
                    work_totals[name] += int(row[col])

        q_count = max(1, len(rows))
        total_phase_cycles = sum(phase_totals.values())
        phase_pct = {}
        for name, value in phase_totals.items():
            phase_pct[name] = (
                (100.0 * float(value) / float(total_phase_cycles))
                if total_phase_cycles > 0 else 0.0
            )

        avg_phase = {
            name: (float(value) / float(q_count))
            for name, value in phase_totals.items()
        }
        avg_work = {
            name: (float(value) / float(q_count))
            for name, value in work_totals.items()
        }

        print(
            "[RetrievalAttention][fullgpu_kernel] "
            f"step={self.decode_pos} layer={layer_idx} kv_head={kv_hdx} q={q_count} "
            f"wall_group={graph_elapsed:.4f}s wall_head={graph_elapsed / float(q_count):.4f}s "
            f"avg_cycles="
            f"{{init:{avg_phase['init']:.0f}, score:{avg_phase['score']:.0f}, "
            f"merge:{avg_phase['merge']:.0f}, expand:{avg_phase['expand']:.0f}, "
            f"finalize:{avg_phase['finalize']:.0f}}} "
            f"phase_pct="
            f"{{init:{phase_pct['init']:.1f}, score:{phase_pct['score']:.1f}, "
            f"merge:{phase_pct['merge']:.1f}, expand:{phase_pct['expand']:.1f}, "
            f"finalize:{phase_pct['finalize']:.1f}}} "
            f"avg_work="
            f"{{scored:{avg_work['scored']:.1f}, edges:{avg_work['edges']:.1f}, "
            f"accepted:{avg_work['accepted']:.1f}, expanded:{avg_work['frontier_expanded']:.1f}, "
            f"find_probes:{avg_work['find_probes']:.1f}}}"
        )

    def _prepare_decode_seed_state_fullgpu(self, ldx: int, hdx: int, query_group):
        total_start, profile = self._make_decode_retrieve_profile()
        kv_hdx = self._retrieval_head_to_kv_head(hdx)
        graph = self.graphs[ldx][kv_hdx]
        graph_is_csr = isinstance(graph, tuple) and len(graph) >= 2
        if not graph_is_csr:
            if profile is not None:
                self._finish_decode_retrieve_profile(total_start, profile, "graph_missing", 0, 0)
            return {
                "empty": True,
                "tokens": [],
                "profile": profile,
                "ldx": int(ldx),
                "hdx": int(hdx),
                "kv_hdx": int(kv_hdx),
                "q_group": query_group,
            }

        q_group = query_group.detach().float()
        if q_group.dim() == 1:
            q_group = q_group.unsqueeze(0)
        if self.query_mode == "group_avg":
            q_seed = q_group.mean(dim=0, keepdim=True)
        else:
            q_seed = q_group

        device = self.layer_mapping[str(ldx)]
        q_seed_cuda = self._score_transform_torch(q_seed.to(device, non_blocking=True).float())
        q_rank_cuda = self._score_transform_torch(q_group.to(device, non_blocking=True).float())

        budget_cap = int(self._effective_dynamic_budget_cap())
        candidate_target = budget_cap * self.candidate_multiplier
        candidate_target = max(budget_cap, candidate_target)
        candidate_target = min(candidate_target, self.input_length, 512)
        seed_floor = int(math.ceil(budget_cap * self.seed_ratio))
        seed_floor = min(budget_cap, max(0, seed_floor))
        seed_k = max(self.q_knn, self.q_knn * self.seed_k_mult)
        seed_k = min(self.input_length, seed_k)
        if profile is not None:
            profile["seed_sec"] += 0.0
        return {
            "empty": False,
            "total_start": total_start,
            "profile": profile,
            "ldx": int(ldx),
            "hdx": int(hdx),
            "kv_hdx": int(kv_hdx),
            "q_group": q_group,
            "q_seed_cuda": q_seed_cuda,
            "q_rank_cuda": q_rank_cuda,
            "candidate_target": int(candidate_target),
            "seed_floor": int(seed_floor),
            "seed_k": int(seed_k),
        }

    def _prepare_decode_seed_state(self, ldx, hdx, query_group, defer_seed_scoring: bool = False):
        total_start, profile = self._make_decode_retrieve_profile()
        kv_hdx = self._retrieval_head_to_kv_head(hdx)
        index = self.indexes[ldx][kv_hdx]
        graph = self.graphs[ldx][kv_hdx]
        graph_is_csr = isinstance(graph, tuple) and len(graph) >= 2
        graph_offsets = None
        graph_neighbors = None
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
        device = self.layer_mapping[str(ldx)]
        q_seed_cuda = self._score_transform_torch(q_seed.to(device, non_blocking=True).float())
        q_rank_cuda = self._score_transform_torch(q_group.to(device, non_blocking=True).float())
        seed_scores = {}
        seed_k = max(self.q_knn, self.q_knn * self.seed_k_mult)
        seed_k = min(self.input_length, seed_k)
        if seed_k <= 0:
            if profile is not None:
                profile["seed_sec"] += time.perf_counter() - seed_start
                self._finish_decode_retrieve_profile(total_start, profile, "seed_k_zero", 0, 0)
            return {
                "empty": True,
                "tokens": [],
                "profile": profile,
                "ldx": int(ldx),
                "hdx": int(hdx),
                "kv_hdx": int(kv_hdx),
                "q_group": q_group,
            }

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
                if not self._fallback_seed_warned:
                    print(
                        "[RetrievalAttention] WARNING: decode index missing; falling back to brute-force seed search. "
                        "Set RETRIEVALATTN_DECODE_INDEX=faiss for stable quality."
                    )
                    self._fallback_seed_warned = True
                k = self.cpu_keys[ldx][kv_hdx, :self.input_length, :].detach().float().cpu()
                k = self._score_transform_torch(k)
                scores = torch.matmul(q_seed_cpu, k.transpose(0, 1))
                k_take = min(seed_k, scores.shape[1])
                if k_take <= 0:
                    if profile is not None:
                        profile["seed_sec"] += time.perf_counter() - seed_start
                        self._finish_decode_retrieve_profile(total_start, profile, "seed_k_zero", 0, 0)
                    return {
                        "empty": True,
                        "tokens": [],
                        "profile": profile,
                        "ldx": int(ldx),
                        "hdx": int(hdx),
                        "kv_hdx": int(kv_hdx),
                        "q_group": q_group,
                    }
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
            if filtered and defer_seed_scoring:
                seed_candidate_ids = [int(tok) for tok in filtered]
            elif filtered:
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
            else:
                seed_candidate_ids = []
        if self.seed_mode != "faiss" and defer_seed_scoring:
            if profile is not None:
                profile["seed_sec"] += time.perf_counter() - seed_start
            candidate_target = self.token_budget * self.candidate_multiplier
            candidate_target = max(self.token_budget, candidate_target)
            candidate_target = min(candidate_target, self.input_length)
            if not seed_candidate_ids:
                if profile is not None:
                    self._finish_decode_retrieve_profile(total_start, profile, "empty_seed_ranked", 0, 0)
                return {
                    "empty": True,
                    "tokens": [],
                    "profile": profile,
                    "ldx": int(ldx),
                    "hdx": int(hdx),
                    "kv_hdx": int(kv_hdx),
                    "q_group": q_group,
                }
            return {
                "empty": False,
                "total_start": total_start,
                "profile": profile,
                "ldx": int(ldx),
                "hdx": int(hdx),
                "kv_hdx": int(kv_hdx),
                "graph": graph,
                "graph_is_csr": graph_is_csr,
                "graph_offsets": graph_offsets,
                "graph_neighbors": graph_neighbors,
                "q_group": q_group,
                "q_seed_cuda": q_seed_cuda,
                "q_rank_cuda": q_rank_cuda,
                "candidate_target": int(candidate_target),
                "seed_k": int(seed_k),
                "seed_candidate_ids": seed_candidate_ids,
            }

        if profile is not None:
            profile["seed_sec"] += time.perf_counter() - seed_start

        seed_ranked = []
        for tok, score in seed_scores.items():
            if tok in static_indices:
                continue
            seed_ranked.append((tok, score))
        seed_ranked.sort(key=lambda x: x[1], reverse=True)
        if not seed_ranked:
            if profile is not None:
                self._finish_decode_retrieve_profile(total_start, profile, "empty_seed_ranked", 0, 0)
            return {
                "empty": True,
                "tokens": [],
                "profile": profile,
                "ldx": int(ldx),
                "hdx": int(hdx),
                "kv_hdx": int(kv_hdx),
                "q_group": q_group,
            }

        seed_floor = int(math.ceil(self.token_budget * self.seed_ratio))
        seed_floor = min(self.token_budget, max(0, seed_floor))
        if seed_floor > len(seed_ranked):
            seed_floor = len(seed_ranked)
        selected_seeds = [tok for tok, _ in seed_ranked[:seed_floor]]
        selected_seed_set = set(selected_seeds)

        candidate_target = self.token_budget * self.candidate_multiplier
        candidate_target = max(self.token_budget, candidate_target)
        candidate_target = min(candidate_target, self.input_length)

        init_candidates = []
        init_scores = []
        for tok, score in seed_ranked:
            if len(init_candidates) >= candidate_target:
                break
            tok = int(tok)
            init_candidates.append(tok)
            init_scores.append(float(score))

        return {
            "empty": False,
            "total_start": total_start,
            "profile": profile,
            "ldx": int(ldx),
            "hdx": int(hdx),
            "kv_hdx": int(kv_hdx),
            "graph": graph,
            "graph_is_csr": graph_is_csr,
            "graph_offsets": graph_offsets,
            "graph_neighbors": graph_neighbors,
            "q_group": q_group,
            "q_seed_cuda": q_seed_cuda,
            "q_rank_cuda": q_rank_cuda,
            "seed_floor": int(seed_floor),
            "selected_seed_set": selected_seed_set,
            "candidate_target": int(candidate_target),
            "init_candidates": init_candidates,
            "init_scores": init_scores,
        }

    def _finalize_decode_seed_state(
        self,
        state,
        ranked_tokens,
        stop_reason: str,
        visited_count: int,
        candidate_count: int,
        update_decode_state: bool = True,
        enforce_seed_floor: bool = True,
    ):
        profile = state["profile"]
        finalize_start = time.perf_counter() if profile is not None else None
        final = []
        final_set = set()
        seed_floor = int(state["seed_floor"])
        selected_seed_set = state["selected_seed_set"]
        if enforce_seed_floor and seed_floor > 0:
            for tok in ranked_tokens:
                if tok in selected_seed_set and tok not in final_set:
                    final.append(tok)
                    final_set.add(tok)
                    if len(final) >= seed_floor or len(final) >= self.token_budget:
                        break
        for tok in ranked_tokens:
            tok = int(tok)
            if tok in final_set:
                continue
            final.append(tok)
            final_set.add(tok)
            if len(final) >= self.token_budget:
                break
        if update_decode_state:
            hdx = int(state["hdx"])
            ldx = int(state["ldx"]) if "ldx" in state else None
            if ldx is not None:
                if final:
                    self.prev_decode_seeds[ldx][hdx] = list(final[: self.seed_prev_k])
                else:
                    self.prev_decode_seeds[ldx][hdx] = []
        if profile is not None:
            profile["finalize_sec"] += time.perf_counter() - finalize_start
            self._finish_decode_retrieve_profile(
                state["total_start"],
                profile,
                stop_reason,
                visited_count,
                candidate_count,
            )
        return final, profile

    def _retrieve_tokens_roar_cuda_v2_group(
        self,
        ldx: int,
        kv_hdx: int,
        states,
        update_decode_state: bool = True,
        enforce_seed_floor: bool = True,
    ):
        results = {}
        active_states = []
        for state in states:
            if state.get("empty", False):
                results[int(state["hdx"])] = (list(state.get("tokens", [])), state.get("profile"))
            else:
                active_states.append(state)

        if not active_states:
            return results

        seed_states = [state for state in active_states if "seed_candidate_ids" in state]
        if seed_states:
            seed_q_count = len(seed_states)
            union_tokens = []
            union_pos = {}
            row_members = []
            for state in seed_states:
                row = []
                for tok in state["seed_candidate_ids"]:
                    tok = int(tok)
                    pos = union_pos.get(tok)
                    if pos is None:
                        pos = len(union_tokens)
                        union_pos[tok] = pos
                        union_tokens.append(tok)
                    row.append(int(pos))
                row_members.append(row)

            key_t_seed = self._get_decode_key_tensor_cuda(ldx, kv_hdx)
            if key_t_seed is None:
                raise RuntimeError("roar_cuda_v2 seed key cache unavailable")
            queries_seed = torch.cat([state["q_seed_cuda"] for state in seed_states], dim=0)
            union_ids_t = torch.as_tensor(union_tokens, dtype=torch.long, device=key_t_seed.device)
            seed_keys_t = torch.index_select(key_t_seed, 0, union_ids_t)
            seed_score_start = time.perf_counter()
            seed_scores_t = torch.matmul(queries_seed, seed_keys_t.transpose(0, 1))
            mask = torch.zeros((seed_q_count, len(union_tokens)), dtype=torch.bool, device=key_t_seed.device)
            for i, positions in enumerate(row_members):
                if positions:
                    mask[i, torch.as_tensor(positions, dtype=torch.long, device=key_t_seed.device)] = True
            seed_scores_t = seed_scores_t.masked_fill(~mask, float("-inf"))
            max_seed_k = max(max(1, int(state["seed_k"])) for state in seed_states)
            top_vals_t, top_pos_t = torch.topk(seed_scores_t, k=min(max_seed_k, seed_scores_t.shape[1]), dim=1)
            top_vals = top_vals_t.cpu()
            top_pos = top_pos_t.cpu()
            per_seed_sec = (time.perf_counter() - seed_score_start) / float(max(1, seed_q_count))

            for i, state in enumerate(seed_states):
                seed_ranked = []
                take = min(int(state["seed_k"]), len(state["seed_candidate_ids"]))
                row_vals = top_vals[i]
                row_pos = top_pos[i]
                for j in range(row_pos.shape[0]):
                    score = float(row_vals[j].item())
                    if not math.isfinite(score):
                        continue
                    tok = int(union_tokens[int(row_pos[j].item())])
                    seed_ranked.append((tok, score))
                    if len(seed_ranked) >= take:
                        break
                if not seed_ranked:
                    if state["profile"] is not None:
                        self._finish_decode_retrieve_profile(
                            state["total_start"],
                            state["profile"],
                            "empty_seed_ranked",
                            0,
                            0,
                        )
                    results[int(state["hdx"])] = ([], state["profile"])
                    state["empty"] = True
                    continue
                seed_floor = int(math.ceil(self.token_budget * self.seed_ratio))
                seed_floor = min(self.token_budget, max(0, seed_floor))
                seed_floor = min(seed_floor, len(seed_ranked))
                state["seed_floor"] = int(seed_floor)
                state["selected_seed_set"] = set(tok for tok, _ in seed_ranked[:seed_floor])
                state["init_candidates"] = [int(tok) for tok, _ in seed_ranked]
                state["init_scores"] = [float(score) for _, score in seed_ranked]
                if state["profile"] is not None:
                    state["profile"]["seed_sec"] += per_seed_sec

            active_states = [state for state in active_states if not state.get("empty", False)]
            if not active_states:
                return results

        graph = active_states[0]["graph"]
        if not (isinstance(graph, tuple) and len(graph) >= 2):
            for state in active_states:
                token_ids, retrieve_profile = self._retrieve_tokens(
                    ldx,
                    int(state["hdx"]),
                    state["q_group"],
                    update_decode_state=update_decode_state,
                    enforce_seed_floor=enforce_seed_floor,
                )
                results[int(state["hdx"])] = (token_ids, retrieve_profile)
            return results

        key_t = self._get_decode_key_tensor_cuda(ldx, kv_hdx)
        graph_tensors = self._get_decode_graph_tensors_cuda(ldx, kv_hdx)
        if key_t is None or graph_tensors is None:
            for state in active_states:
                token_ids, retrieve_profile = self._retrieve_tokens(
                    ldx,
                    int(state["hdx"]),
                    state["q_group"],
                    update_decode_state=update_decode_state,
                    enforce_seed_floor=enforce_seed_floor,
                )
                results[int(state["hdx"])] = (token_ids, retrieve_profile)
            return results

        q_count = len(active_states)
        max_init = max(len(state["init_candidates"]) for state in active_states)
        queries_seed = torch.cat([state["q_seed_cuda"] for state in active_states], dim=0)
        queries_rank = torch.cat([state["q_rank_cuda"] for state in active_states], dim=0)
        init_ids_t = torch.full((q_count, max_init), -1, dtype=torch.int32, device="cpu")
        init_scores_t = torch.full((q_count, max_init), -1e30, dtype=torch.float32, device="cpu")
        for i, state in enumerate(active_states):
            n = len(state["init_candidates"])
            if n <= 0:
                continue
            init_ids_t[i, :n] = torch.as_tensor(state["init_candidates"], dtype=torch.int32, device="cpu")
            init_scores_t[i, :n] = torch.as_tensor(state["init_scores"], dtype=torch.float32, device="cpu")

        graph_offsets_t, graph_neighbors_t = graph_tensors
        graph_start = time.perf_counter()
        out_ids_t, _out_scores_t, out_counts_t, out_visited_t, out_stop_t = search_roar_graph_csr_cuda_group(
            queries_seed=queries_seed,
            queries_rank=queries_rank,
            keys=key_t,
            offsets=graph_offsets_t,
            neighbors=graph_neighbors_t,
            init_ids=init_ids_t,
            init_scores=init_scores_t,
            token_budget=int(self._effective_dynamic_budget_cap()),
            candidate_target=int(active_states[0]["candidate_target"]),
            expand_width=int(self.expand_width),
            min_visits=int(self.min_visits),
            max_visits=int(self.max_visits),
            frontier_topn=int(self.frontier_topn),
            stop_patience=int(self.stop_patience),
            stop_margin=float(self.stop_margin),
            dynamic_start=int(self.dynamic_start),
            dynamic_end=int(self.dynamic_end),
            score_agg=self.rerank_agg,
        )
        graph_elapsed = time.perf_counter() - graph_start
        out_ids = out_ids_t.cpu()
        out_counts = out_counts_t.cpu()
        out_visited = out_visited_t.cpu()
        out_stop = out_stop_t.cpu()
        per_graph_sec = graph_elapsed / float(max(1, q_count))

        for i, state in enumerate(active_states):
            keep = int(out_counts[i].item())
            ranked_tokens = []
            if keep > 0:
                row = out_ids[i, :keep]
                ranked_tokens = [int(tok) for tok in row.tolist() if int(tok) >= 0]
            stop_reason = self._decode_stop_reason_from_code(int(out_stop[i].item()))
            visited_count = int(out_visited[i].item())
            candidate_count = len(ranked_tokens)
            profile = state["profile"]
            if profile is not None:
                profile["graph_sec"] += per_graph_sec
                profile["rerank_sec"] += 0.0
            final, retrieve_profile = self._finalize_decode_seed_state(
                state,
                ranked_tokens,
                stop_reason=stop_reason,
                visited_count=visited_count,
                candidate_count=candidate_count,
                update_decode_state=update_decode_state,
                enforce_seed_floor=enforce_seed_floor,
            )
            if retrieve_profile is not None:
                retrieve_profile["total_sec"] = (
                    float(retrieve_profile.get("seed_sec", 0.0))
                    + float(retrieve_profile.get("graph_sec", 0.0))
                    + float(retrieve_profile.get("rerank_sec", 0.0))
                    + float(retrieve_profile.get("finalize_sec", 0.0))
                )
            results[int(state["hdx"])] = (final, retrieve_profile)

        return results

    def _retrieve_tokens_roar_cuda_beam_group(
        self,
        ldx: int,
        kv_hdx: int,
        states,
        update_decode_state: bool = True,
        enforce_seed_floor: bool = True,
    ):
        results = {}
        active_states = []
        for state in states:
            if state.get("empty", False):
                results[int(state["hdx"])] = (list(state.get("tokens", [])), state.get("profile"))
            else:
                active_states.append(state)

        if not active_states:
            return results

        seed_states = [state for state in active_states if "seed_candidate_ids" in state]
        if seed_states:
            seed_q_count = len(seed_states)
            union_tokens = []
            union_pos = {}
            row_members = []
            for state in seed_states:
                row = []
                for tok in state["seed_candidate_ids"]:
                    tok = int(tok)
                    pos = union_pos.get(tok)
                    if pos is None:
                        pos = len(union_tokens)
                        union_pos[tok] = pos
                        union_tokens.append(tok)
                    row.append(int(pos))
                row_members.append(row)

            key_t_seed = self._get_decode_key_tensor_cuda(ldx, kv_hdx)
            if key_t_seed is None:
                raise RuntimeError("roar_cuda_beam seed key cache unavailable")
            queries_seed = torch.cat([state["q_seed_cuda"] for state in seed_states], dim=0)
            union_ids_t = torch.as_tensor(union_tokens, dtype=torch.long, device=key_t_seed.device)
            seed_keys_t = torch.index_select(key_t_seed, 0, union_ids_t)
            seed_score_start = time.perf_counter()
            seed_scores_t = torch.matmul(queries_seed, seed_keys_t.transpose(0, 1))
            mask = torch.zeros((seed_q_count, len(union_tokens)), dtype=torch.bool, device=key_t_seed.device)
            for i, positions in enumerate(row_members):
                if positions:
                    mask[i, torch.as_tensor(positions, dtype=torch.long, device=key_t_seed.device)] = True
            seed_scores_t = seed_scores_t.masked_fill(~mask, float("-inf"))
            max_seed_k = max(max(1, int(state["seed_k"])) for state in seed_states)
            top_vals_t, top_pos_t = torch.topk(seed_scores_t, k=min(max_seed_k, seed_scores_t.shape[1]), dim=1)
            top_vals = top_vals_t.cpu()
            top_pos = top_pos_t.cpu()
            per_seed_sec = (time.perf_counter() - seed_score_start) / float(max(1, seed_q_count))

            for i, state in enumerate(seed_states):
                seed_ranked = []
                take = min(int(state["seed_k"]), len(state["seed_candidate_ids"]))
                row_vals = top_vals[i]
                row_pos = top_pos[i]
                for j in range(row_pos.shape[0]):
                    score = float(row_vals[j].item())
                    if not math.isfinite(score):
                        continue
                    tok = int(union_tokens[int(row_pos[j].item())])
                    seed_ranked.append((tok, score))
                    if len(seed_ranked) >= take:
                        break
                if not seed_ranked:
                    if state["profile"] is not None:
                        self._finish_decode_retrieve_profile(
                            state["total_start"],
                            state["profile"],
                            "empty_seed_ranked",
                            0,
                            0,
                        )
                    results[int(state["hdx"])] = ([], state["profile"])
                    state["empty"] = True
                    continue
                seed_floor = int(math.ceil(self.token_budget * self.seed_ratio))
                seed_floor = min(self.token_budget, max(0, seed_floor))
                seed_floor = min(seed_floor, len(seed_ranked))
                state["seed_floor"] = int(seed_floor)
                state["selected_seed_set"] = set(tok for tok, _ in seed_ranked[:seed_floor])
                state["init_candidates"] = [int(tok) for tok, _ in seed_ranked]
                state["init_scores"] = [float(score) for _, score in seed_ranked]
                if state["profile"] is not None:
                    state["profile"]["seed_sec"] += per_seed_sec

            active_states = [state for state in active_states if not state.get("empty", False)]
            if not active_states:
                return results

        graph = active_states[0]["graph"]
        if not (isinstance(graph, tuple) and len(graph) >= 2):
            for state in active_states:
                token_ids, retrieve_profile = self._retrieve_tokens(
                    ldx,
                    int(state["hdx"]),
                    state["q_group"],
                    update_decode_state=update_decode_state,
                    enforce_seed_floor=enforce_seed_floor,
                )
                results[int(state["hdx"])] = (token_ids, retrieve_profile)
            return results

        key_t = self._get_decode_key_tensor_cuda(ldx, kv_hdx)
        graph_tensors = self._get_decode_graph_tensors_cuda_device(ldx, kv_hdx)
        if key_t is None or graph_tensors is None:
            raise RuntimeError("roar_cuda_beam decode state is unavailable")

        q_count = len(active_states)
        max_init = max(len(state["init_candidates"]) for state in active_states)
        queries_seed = torch.cat([state["q_seed_cuda"] for state in active_states], dim=0)
        queries_rank = torch.cat([state["q_rank_cuda"] for state in active_states], dim=0)
        init_ids_t = torch.full((q_count, max_init), -1, dtype=torch.int32, device="cpu")
        init_scores_t = torch.full((q_count, max_init), -1e30, dtype=torch.float32, device="cpu")
        for i, state in enumerate(active_states):
            n = len(state["init_candidates"])
            if n <= 0:
                continue
            init_ids_t[i, :n] = torch.as_tensor(state["init_candidates"], dtype=torch.int32, device="cpu")
            init_scores_t[i, :n] = torch.as_tensor(state["init_scores"], dtype=torch.float32, device="cpu")

        graph_offsets_t, graph_neighbors_t = graph_tensors
        graph_start = time.perf_counter()
        out_ids_t, _out_scores_t, out_counts_t, out_visited_t, out_stop_t = search_roar_graph_csr_cuda_group_beam(
            queries_seed=queries_seed,
            queries_rank=queries_rank,
            keys=key_t,
            offsets=graph_offsets_t,
            neighbors=graph_neighbors_t,
            init_ids=init_ids_t,
            init_scores=init_scores_t,
            token_budget=int(self.token_budget),
            candidate_target=int(active_states[0]["candidate_target"]),
            beam_width=int(self.expand_width),
            min_visits=int(self.min_visits),
            max_visits=int(self.max_visits),
            stop_patience=int(self.stop_patience),
            stop_margin=float(self.stop_margin),
            dynamic_start=int(self.dynamic_start),
            dynamic_end=int(self.dynamic_end),
            score_agg=self.rerank_agg,
        )
        graph_elapsed = time.perf_counter() - graph_start
        out_ids = out_ids_t.cpu()
        out_counts = out_counts_t.cpu()
        out_visited = out_visited_t.cpu()
        out_stop = out_stop_t.cpu()
        per_graph_sec = graph_elapsed / float(max(1, q_count))

        for i, state in enumerate(active_states):
            keep = int(out_counts[i].item())
            ranked_tokens = []
            if keep > 0:
                row = out_ids[i, :keep]
                ranked_tokens = [int(tok) for tok in row.tolist() if int(tok) >= 0]
            stop_reason = self._decode_stop_reason_from_code(int(out_stop[i].item()))
            visited_count = int(out_visited[i].item())
            candidate_count = len(ranked_tokens)
            profile = state["profile"]
            if profile is not None:
                profile["graph_sec"] += per_graph_sec
                profile["rerank_sec"] += 0.0
            final, retrieve_profile = self._finalize_decode_seed_state(
                state,
                ranked_tokens,
                stop_reason=stop_reason,
                visited_count=visited_count,
                candidate_count=candidate_count,
                update_decode_state=update_decode_state,
                enforce_seed_floor=enforce_seed_floor,
            )
            if retrieve_profile is not None:
                retrieve_profile["total_sec"] = (
                    float(retrieve_profile.get("seed_sec", 0.0))
                    + float(retrieve_profile.get("graph_sec", 0.0))
                    + float(retrieve_profile.get("rerank_sec", 0.0))
                    + float(retrieve_profile.get("finalize_sec", 0.0))
                )
            results[int(state["hdx"])] = (final, retrieve_profile)

        return results

    def _retrieve_tokens_roar_cuda_kernel_group(
        self,
        ldx: int,
        kv_hdx: int,
        states,
        update_decode_state: bool = True,
        enforce_seed_floor: bool = True,
    ):
        results = {}
        active_states = []
        for state in states:
            if state.get("empty", False):
                results[int(state["hdx"])] = (list(state.get("tokens", [])), state.get("profile"))
            else:
                active_states.append(state)

        if not active_states:
            return results

        seed_states = [state for state in active_states if "seed_candidate_ids" in state]
        if seed_states:
            seed_q_count = len(seed_states)
            union_tokens = []
            union_pos = {}
            row_members = []
            for state in seed_states:
                row = []
                for tok in state["seed_candidate_ids"]:
                    tok = int(tok)
                    pos = union_pos.get(tok)
                    if pos is None:
                        pos = len(union_tokens)
                        union_pos[tok] = pos
                        union_tokens.append(tok)
                    row.append(int(pos))
                row_members.append(row)

            key_t_seed = self._get_decode_key_tensor_cuda(ldx, kv_hdx)
            if key_t_seed is None:
                raise RuntimeError("roar_cuda_kernel seed key cache unavailable")
            queries_seed = torch.cat([state["q_seed_cuda"] for state in seed_states], dim=0)
            union_ids_t = torch.as_tensor(union_tokens, dtype=torch.long, device=key_t_seed.device)
            seed_keys_t = torch.index_select(key_t_seed, 0, union_ids_t)
            seed_score_start = time.perf_counter()
            seed_scores_t = torch.matmul(queries_seed, seed_keys_t.transpose(0, 1))
            mask = torch.zeros((seed_q_count, len(union_tokens)), dtype=torch.bool, device=key_t_seed.device)
            for i, positions in enumerate(row_members):
                if positions:
                    mask[i, torch.as_tensor(positions, dtype=torch.long, device=key_t_seed.device)] = True
            seed_scores_t = seed_scores_t.masked_fill(~mask, float("-inf"))
            max_seed_k = max(max(1, int(state["seed_k"])) for state in seed_states)
            top_vals_t, top_pos_t = torch.topk(seed_scores_t, k=min(max_seed_k, seed_scores_t.shape[1]), dim=1)
            top_vals = top_vals_t.cpu()
            top_pos = top_pos_t.cpu()
            per_seed_sec = (time.perf_counter() - seed_score_start) / float(max(1, seed_q_count))

            for i, state in enumerate(seed_states):
                seed_ranked = []
                take = min(int(state["seed_k"]), len(state["seed_candidate_ids"]))
                row_vals = top_vals[i]
                row_pos = top_pos[i]
                for j in range(row_pos.shape[0]):
                    score = float(row_vals[j].item())
                    if not math.isfinite(score):
                        continue
                    tok = int(union_tokens[int(row_pos[j].item())])
                    seed_ranked.append((tok, score))
                    if len(seed_ranked) >= take:
                        break
                if not seed_ranked:
                    if state["profile"] is not None:
                        self._finish_decode_retrieve_profile(
                            state["total_start"],
                            state["profile"],
                            "empty_seed_ranked",
                            0,
                            0,
                        )
                    results[int(state["hdx"])] = ([], state["profile"])
                    state["empty"] = True
                    continue
                seed_floor = int(math.ceil(self.token_budget * self.seed_ratio))
                seed_floor = min(self.token_budget, max(0, seed_floor))
                seed_floor = min(seed_floor, len(seed_ranked))
                state["seed_floor"] = int(seed_floor)
                state["selected_seed_set"] = set(tok for tok, _ in seed_ranked[:seed_floor])
                state["init_candidates"] = [int(tok) for tok, _ in seed_ranked]
                state["init_scores"] = [float(score) for _, score in seed_ranked]
                if state["profile"] is not None:
                    state["profile"]["seed_sec"] += per_seed_sec

            active_states = [state for state in active_states if not state.get("empty", False)]
            if not active_states:
                return results

        graph = active_states[0]["graph"]
        if not (isinstance(graph, tuple) and len(graph) >= 2):
            for state in active_states:
                token_ids, retrieve_profile = self._retrieve_tokens(
                    ldx,
                    int(state["hdx"]),
                    state["q_group"],
                    update_decode_state=update_decode_state,
                    enforce_seed_floor=enforce_seed_floor,
                )
                results[int(state["hdx"])] = (token_ids, retrieve_profile)
            return results

        key_t = self._get_decode_key_tensor_cuda(ldx, kv_hdx)
        graph_tensors = self._get_decode_graph_tensors_cuda_device(ldx, kv_hdx)
        if key_t is None or graph_tensors is None:
            raise RuntimeError("roar_cuda_kernel decode state is unavailable")
        graph_max_degree = int(self._get_decode_graph_max_degree(ldx, kv_hdx))

        q_count = len(active_states)
        max_init = max(len(state["init_candidates"]) for state in active_states)
        queries_seed = torch.cat([state["q_seed_cuda"] for state in active_states], dim=0)
        queries_rank = torch.cat([state["q_rank_cuda"] for state in active_states], dim=0)
        init_ids_t = torch.full((q_count, max_init), -1, dtype=torch.int32, device="cpu")
        init_scores_t = torch.full((q_count, max_init), -1e30, dtype=torch.float32, device="cpu")
        for i, state in enumerate(active_states):
            n = len(state["init_candidates"])
            if n <= 0:
                continue
            init_ids_t[i, :n] = torch.as_tensor(state["init_candidates"], dtype=torch.int32, device="cpu")
            init_scores_t[i, :n] = torch.as_tensor(state["init_scores"], dtype=torch.float32, device="cpu")

        graph_offsets_t, graph_neighbors_t = graph_tensors
        graph_start = time.perf_counter()
        out_ids_t, _out_scores_t, out_counts_t, out_visited_t, out_stop_t = search_roar_graph_csr_cuda_group_kernel(
            queries_seed=queries_seed,
            queries_rank=queries_rank,
            keys=key_t,
            offsets=graph_offsets_t,
            neighbors=graph_neighbors_t,
            init_ids=init_ids_t,
            init_scores=init_scores_t,
            token_budget=int(self.token_budget),
            candidate_target=int(active_states[0]["candidate_target"]),
            beam_width=int(self.expand_width),
            max_degree=int(graph_max_degree),
            min_visits=int(self.min_visits),
            max_visits=int(self.max_visits),
            stop_patience=int(self.stop_patience),
            stop_margin=float(self.stop_margin),
            dynamic_start=int(self.dynamic_start),
            dynamic_end=int(self.dynamic_end),
            score_agg=self.rerank_agg,
        )
        graph_elapsed = time.perf_counter() - graph_start
        out_ids = out_ids_t.cpu()
        out_counts = out_counts_t.cpu()
        out_visited = out_visited_t.cpu()
        out_stop = out_stop_t.cpu()
        per_graph_sec = graph_elapsed / float(max(1, q_count))

        for i, state in enumerate(active_states):
            keep = int(out_counts[i].item())
            ranked_tokens = []
            if keep > 0:
                row = out_ids[i, :keep]
                ranked_tokens = [int(tok) for tok in row.tolist() if int(tok) >= 0]
            stop_reason = self._decode_stop_reason_from_code(int(out_stop[i].item()))
            visited_count = int(out_visited[i].item())
            candidate_count = len(ranked_tokens)
            profile = state["profile"]
            if profile is not None:
                profile["graph_sec"] += per_graph_sec
                profile["rerank_sec"] += 0.0
            final, retrieve_profile = self._finalize_decode_seed_state(
                state,
                ranked_tokens,
                stop_reason=stop_reason,
                visited_count=visited_count,
                candidate_count=candidate_count,
                update_decode_state=update_decode_state,
                enforce_seed_floor=enforce_seed_floor,
            )
            if retrieve_profile is not None:
                retrieve_profile["total_sec"] = (
                    float(retrieve_profile.get("seed_sec", 0.0))
                    + float(retrieve_profile.get("graph_sec", 0.0))
                    + float(retrieve_profile.get("rerank_sec", 0.0))
                    + float(retrieve_profile.get("finalize_sec", 0.0))
                )
            results[int(state["hdx"])] = (final, retrieve_profile)

        return results

    def _retrieve_tokens_roar_cuda_fullgpu_group(
        self,
        ldx: int,
        kv_hdx: int,
        states,
        update_decode_state: bool = True,
        enforce_seed_floor: bool = True,
    ):
        del enforce_seed_floor  # The full-GPU kernel always enforces the seed floor internally.
        results = {}
        active_states = []
        for state in states:
            if state.get("empty", False):
                results[int(state["hdx"])] = (list(state.get("tokens", [])), state.get("profile"))
            else:
                active_states.append(state)

        if not active_states:
            return results

        key_score_t = self._get_decode_key_tensor_cuda(ldx, kv_hdx)
        key_attn_t = self._get_decode_attn_key_tensor_cuda(ldx, kv_hdx)
        value_t = self._get_decode_value_tensor_cuda(ldx, kv_hdx)
        graph_tensors = self._get_decode_graph_tensors_cuda_device(ldx, kv_hdx)
        overlay_tensors = self._get_decode_overlay_graph_tensors_cuda_device(ldx, kv_hdx)
        hub_seed_ids = self._decode_cuda_hub_seed_ids[ldx]
        prev_seed_ids = self._decode_cuda_prev_seed_ids[ldx]
        prev_seed_counts = self._decode_cuda_prev_seed_counts[ldx]
        if (
            key_score_t is None
            or key_attn_t is None
            or value_t is None
            or graph_tensors is None
            or overlay_tensors is None
            or hub_seed_ids is None
            or prev_seed_ids is None
            or prev_seed_counts is None
        ):
            raise RuntimeError("roar_cuda_fullgpu decode state is unavailable")

        graph_offsets_t, graph_neighbors_t = graph_tensors
        overlay_counts_t, overlay_neighbors_t = overlay_tensors
        graph_max_degree = int(self._get_decode_graph_max_degree(ldx, kv_hdx))
        q_count = len(active_states)
        queries_seed = torch.cat([state["q_seed_cuda"] for state in active_states], dim=0)
        queries_rank = torch.cat([state["q_rank_cuda"] for state in active_states], dim=0)
        queries_attn = torch.cat(
            [state["q_group"].to(device=queries_seed.device, non_blocking=True).float() for state in active_states],
            dim=0,
        )
        static_k, _static_v = self._get_fullgpu_static_kv(ldx, kv_hdx)
        scale = 1.0 / math.sqrt(float(self.head_dim))
        if static_k is not None and int(static_k.shape[0]) > 0:
            static_scores_t = torch.matmul(queries_attn, static_k.float().transpose(0, 1)) * scale
            static_logz_t = torch.logsumexp(static_scores_t.float(), dim=-1)
        else:
            static_logz_t = torch.full(
                (q_count,),
                float("-inf"),
                dtype=torch.float32,
                device=queries_seed.device,
            )
        upper_scores_t = torch.zeros((q_count,), dtype=torch.float32, device=queries_seed.device)
        total_score_sum_t = torch.zeros((q_count,), dtype=torch.float32, device=queries_seed.device)
        total_score_sumsq_t = torch.zeros((q_count,), dtype=torch.float32, device=queries_seed.device)
        if self.dynamic_budget_prior == "moment_diag":
            sum_k_t, sumsq_k_t = self._get_dynamic_attn_moment_tensors_fullgpu(ldx, kv_hdx)
            sum_k_t = sum_k_t.to(device=queries_seed.device, dtype=torch.float32)
            sumsq_k_t = sumsq_k_t.to(device=queries_seed.device, dtype=torch.float32)
            total_score_sum_t = torch.matmul(queries_attn.float(), sum_k_t) * float(scale)
            total_score_sumsq_t = torch.matmul(
                queries_attn.float().square(),
                sumsq_k_t,
            ) * float(scale * scale)
        else:
            max_k_norm_t = self._get_dynamic_attn_max_norm_fullgpu(ldx, kv_hdx).to(device=queries_seed.device)
            q_norms_t = torch.linalg.vector_norm(queries_attn.float(), dim=-1)
            upper_scores_t = q_norms_t * max_k_norm_t * float(scale)
        head_indices = [int(state["hdx"]) for state in active_states]
        head_indices_t = torch.as_tensor(
            head_indices,
            dtype=torch.long,
            device=queries_seed.device,
        )
        prev_ids_t = torch.index_select(prev_seed_ids, 0, head_indices_t)
        prev_counts_t = torch.index_select(prev_seed_counts, 0, head_indices_t)
        hub_ids_t = hub_seed_ids[kv_hdx]

        device = queries_seed.device
        kernel_token_budget = int(self.token_budget)
        if self.dynamic_budget_enable and self.dynamic_budget_mode == "traversal_cuda":
            kernel_token_budget = min(128, int(self.dynamic_budget_max))
        if self.decode_profile and self.fullgpu_profile_sync:
            torch.cuda.synchronize(device)
        graph_start = time.perf_counter()
        (
            out_ids_t,
            out_scores_t,
            out_counts_t,
            out_visited_t,
            out_stop_t,
            next_prev_ids_t,
            next_prev_counts_t,
            out_debug_t,
            out_adaptive_keep_t,
            out_adaptive_mass_t,
        ) = search_roar_graph_csr_cuda_group_fullgpu(
            queries_seed=queries_seed,
            queries_rank=queries_rank,
            queries_attn=queries_attn,
            keys=key_score_t,
            attn_keys=key_attn_t,
            static_logz=static_logz_t,
            upper_scores=upper_scores_t,
            total_score_sum=total_score_sum_t,
            total_score_sumsq=total_score_sumsq_t,
            offsets=graph_offsets_t,
            neighbors=graph_neighbors_t,
            overlay_counts=overlay_counts_t,
            overlay_neighbors=overlay_neighbors_t,
            prev_seed_ids=prev_ids_t,
            prev_seed_counts=prev_counts_t,
            hub_seed_ids=hub_ids_t,
            token_budget=int(kernel_token_budget),
            candidate_target=int(active_states[0]["candidate_target"]),
            beam_width=int(self.expand_width),
            max_degree=int(graph_max_degree),
            min_visits=int(self.min_visits),
            max_visits=int(self.max_visits),
            stop_patience=int(self.stop_patience),
            stop_margin=float(self.stop_margin),
            dynamic_start=int(self.dynamic_start),
            dynamic_end=int(self.dynamic_end),
            seed_k=int(active_states[0]["seed_k"]),
            seed_floor=int(active_states[0]["seed_floor"]),
            seed_tail_k=int(self.seed_tail_k),
            seed_prev_k=int(self.seed_prev_k),
            adaptive_enable=bool(self.dynamic_budget_enable and self.dynamic_budget_mode == "traversal_cuda"),
            adaptive_min_keep=int(self.dynamic_budget_min),
            adaptive_target_omass=float(self.dynamic_budget_target_omass),
            adaptive_prior_mode=str(self.dynamic_budget_prior),
            adaptive_prior_var_scale=float(self.dynamic_budget_prior_var_scale),
            score_agg=self.rerank_agg,
        )
        if self.decode_profile and self.fullgpu_profile_sync:
            torch.cuda.synchronize(device)
        graph_elapsed = time.perf_counter() - graph_start
        per_graph_sec = graph_elapsed / float(max(1, q_count))

        if update_decode_state:
            self._decode_cuda_prev_seed_ids[ldx].index_copy_(0, head_indices_t, next_prev_ids_t)
            self._decode_cuda_prev_seed_counts[ldx].index_copy_(0, head_indices_t, next_prev_counts_t)

        out_counts = out_counts_t.detach().cpu()
        out_visited = out_visited_t.detach().cpu()
        out_stop = out_stop_t.detach().cpu()
        out_debug = out_debug_t.detach().cpu()
        out_adaptive_keep = out_adaptive_keep_t.detach().cpu()
        out_adaptive_mass = out_adaptive_mass_t.detach().cpu()
        out_ids_cpu = None
        if self.online_dynamic_range or self.online_graph_enable:
            out_ids_cpu = out_ids_t.detach().cpu()
        out_mask_t = out_ids_t.ge(0)
        self._maybe_log_fullgpu_kernel_debug(
            layer_idx=ldx,
            kv_hdx=kv_hdx,
            out_debug=out_debug,
            graph_elapsed=graph_elapsed,
        )

        for i, state in enumerate(active_states):
            count_i = int(out_counts[i].item())
            visited_i = int(out_visited[i].item())
            stop_reason = self._decode_stop_reason_from_code(int(out_stop[i].item()))
            cand_raw_i = int(out_debug[i, 0].item())
            rounds_i = int(out_debug[i, 1].item())
            frontier_i = int(out_debug[i, 2].item())
            forced_i = int(out_debug[i, 3].item())
            init_cycles_i = int(out_debug[i, 4].item()) if out_debug.shape[1] > 4 else 0
            score_cycles_i = int(out_debug[i, 5].item()) if out_debug.shape[1] > 5 else 0
            merge_cycles_i = int(out_debug[i, 6].item()) if out_debug.shape[1] > 6 else 0
            expand_cycles_i = int(out_debug[i, 7].item()) if out_debug.shape[1] > 7 else 0
            finalize_cycles_i = int(out_debug[i, 8].item()) if out_debug.shape[1] > 8 else 0
            scored_total_i = int(out_debug[i, 9].item()) if out_debug.shape[1] > 9 else 0
            edge_scan_i = int(out_debug[i, 10].item()) if out_debug.shape[1] > 10 else 0
            accepted_i = int(out_debug[i, 11].item()) if out_debug.shape[1] > 11 else 0
            frontier_expanded_i = int(out_debug[i, 12].item()) if out_debug.shape[1] > 12 else 0
            find_probe_i = int(out_debug[i, 13].item()) if out_debug.shape[1] > 13 else 0
            overlay_edges_i = int(out_debug[i, 14].item()) if out_debug.shape[1] > 14 else 0
            adaptive_keep_i = int(out_adaptive_keep[i].item()) if int(out_adaptive_keep.numel()) > 0 else count_i
            adaptive_keep_i = max(0, min(adaptive_keep_i, count_i))
            final_count_i = adaptive_keep_i if (self.dynamic_budget_enable and self.dynamic_budget_mode == "traversal_cuda") else count_i
            adaptive_mass_i = float(out_adaptive_mass[i].item()) if int(out_adaptive_mass.numel()) > 0 else 0.0
            final_tokens_i = []
            if out_ids_cpu is not None and final_count_i > 0:
                final_tokens_i = [
                    int(tok) for tok in out_ids_cpu[i, :final_count_i].tolist()
                    if int(tok) >= 0
                ]
            profile = state["profile"]
            if profile is not None:
                profile["graph_sec"] += per_graph_sec
                self._finish_decode_retrieve_profile(
                    state["total_start"],
                    profile,
                    stop_reason,
                    visited_i,
                    cand_raw_i,
                )
                profile["final_outputs"] = final_count_i
                profile["kernel_rounds"] = rounds_i
                profile["forced_seeds"] = forced_i
                profile["frontier_at_stop"] = frontier_i
                profile["kernel_phase_cycles"] = {
                    "init": init_cycles_i,
                    "score": score_cycles_i,
                    "merge": merge_cycles_i,
                    "expand": expand_cycles_i,
                    "finalize": finalize_cycles_i,
                }
                profile["kernel_work"] = {
                    "scored": scored_total_i,
                    "edges": edge_scan_i,
                    "accepted": accepted_i,
                    "frontier_expanded": frontier_expanded_i,
                    "find_probes": find_probe_i,
                    "overlay_edges": overlay_edges_i,
                }
                if self.dynamic_budget_enable and self.dynamic_budget_mode == "traversal_cuda":
                    profile["adaptive_final_outputs"] = int(final_count_i)
                    profile["adaptive_mass_bound"] = float(adaptive_mass_i)
                    profile["adaptive_upper_score_bound"] = (
                        float(upper_scores_t[i].item())
                        if self.dynamic_budget_prior == "global_norm"
                        else float("nan")
                    )
                profile["online_overlay_edges"] = overlay_edges_i
                if final_tokens_i:
                    online_generated_hits_i = sum(
                        1 for tok in final_tokens_i if self._is_dynamic_generated_token(tok)
                    )
                    profile["online_generated_hits"] = int(online_generated_hits_i)
                    profile["online_generated_any"] = 1 if online_generated_hits_i > 0 else 0
                else:
                    profile["online_generated_hits"] = 0
                    profile["online_generated_any"] = 0
                profile["total_sec"] = (
                    float(profile.get("seed_sec", 0.0))
                    + float(profile.get("graph_sec", 0.0))
                    + float(profile.get("rerank_sec", 0.0))
                    + float(profile.get("finalize_sec", 0.0))
                )
            results[int(state["hdx"])] = (
                {
                    "device_ids": out_ids_t[i],
                    "device_scores": out_scores_t[i],
                    "device_mask": out_ids_t[i].ge(0),
                    "device_attn_keys": key_attn_t,
                    "device_attn_values": value_t,
                    "cpu_tokens": final_tokens_i if out_ids_cpu is not None else None,
                    "final_count": final_count_i,
                    "candidate_count": cand_raw_i,
                    "rounds": rounds_i,
                    "frontier_at_stop": frontier_i,
                    "forced_seeds": forced_i,
                    "adaptive_mass_bound": float(adaptive_mass_i),
                    "adaptive_upper_score_bound": (
                        float(upper_scores_t[i].item())
                        if self.dynamic_budget_prior == "global_norm"
                        else float("nan")
                    ),
                    "adaptive_dynamic_span": int(max(0, int(self.dynamic_end - self.dynamic_start))),
                    "adaptive_candidate_count": int(cand_raw_i),
                    "adaptive_keep_count": int(final_count_i),
                    "kernel_phase_cycles": {
                        "init": init_cycles_i,
                        "score": score_cycles_i,
                        "merge": merge_cycles_i,
                        "expand": expand_cycles_i,
                        "finalize": finalize_cycles_i,
                    },
                    "kernel_work": {
                        "scored": scored_total_i,
                        "edges": edge_scan_i,
                        "accepted": accepted_i,
                        "frontier_expanded": frontier_expanded_i,
                        "find_probes": find_probe_i,
                    },
                },
                profile,
            )

        return results

    def _retrieve_tokens_oracle_fullgpu_group(
        self,
        ldx: int,
        kv_hdx: int,
        states,
    ):
        results = {}
        active_states = []
        for state in states:
            if state.get("empty", False):
                results[int(state["hdx"])] = (list(state.get("tokens", [])), state.get("profile"))
            else:
                active_states.append(state)

        if not active_states:
            return results

        key_score_t = self._get_decode_key_tensor_cuda(ldx, kv_hdx)
        key_attn_t = self._get_decode_attn_key_tensor_cuda(ldx, kv_hdx)
        value_t = self._get_decode_value_tensor_cuda(ldx, kv_hdx)
        if key_score_t is None or key_attn_t is None or value_t is None:
            raise RuntimeError("oracle fullgpu decode state is unavailable")

        dyn_start = int(self.dynamic_start)
        dyn_end = int(self.dynamic_end)
        dyn_len = max(0, dyn_end - dyn_start)
        q_count = len(active_states)
        device = key_score_t.device
        queries_seed = torch.cat([state["q_seed_cuda"] for state in active_states], dim=0)

        if self.decode_profile and self.fullgpu_profile_sync:
            torch.cuda.synchronize(device)
        graph_start = time.perf_counter()
        if dyn_len > 0:
            dyn_keys_t = key_score_t[dyn_start:dyn_end, :]
            topk = min(int(self._effective_dynamic_budget_cap()), int(dyn_keys_t.shape[0]))
            score_t = torch.matmul(queries_seed, dyn_keys_t.transpose(0, 1))
            top_vals_t, top_pos_t = torch.topk(score_t, k=max(1, topk), dim=1)
            top_ids_t = top_pos_t.to(torch.int32) + int(dyn_start)
        else:
            topk = 0
            top_vals_t = torch.empty((q_count, 1), dtype=torch.float32, device=device)
            top_ids_t = torch.full((q_count, 1), -1, dtype=torch.int32, device=device)
        if self.decode_profile and self.fullgpu_profile_sync:
            torch.cuda.synchronize(device)
        graph_elapsed = time.perf_counter() - graph_start
        per_graph_sec = graph_elapsed / float(max(1, q_count))

        out_ids_cpu = None
        if self.online_dynamic_range or self.online_graph_enable:
            out_ids_cpu = top_ids_t.detach().cpu()
        out_mask_t = top_ids_t.ge(0)

        for i, state in enumerate(active_states):
            count_i = int(topk)
            final_tokens_i = []
            if out_ids_cpu is not None and count_i > 0:
                final_tokens_i = [
                    int(tok) for tok in out_ids_cpu[i, :count_i].tolist()
                    if int(tok) >= 0
                ]
            profile = state["profile"]
            if profile is not None:
                profile["graph_sec"] += per_graph_sec
                self._finish_decode_retrieve_profile(
                    state["total_start"],
                    profile,
                    "oracle_topk",
                    count_i,
                    dyn_len,
                )
                profile["final_outputs"] = count_i
                profile["kernel_rounds"] = 0
                profile["forced_seeds"] = 0
                profile["frontier_at_stop"] = 0
                profile["kernel_phase_cycles"] = {
                    "init": 0,
                    "score": 0,
                    "merge": 0,
                    "expand": 0,
                    "finalize": 0,
                }
                profile["kernel_work"] = {
                    "scored": dyn_len,
                    "edges": 0,
                    "accepted": count_i,
                    "frontier_expanded": 0,
                    "find_probes": 0,
                    "overlay_edges": 0,
                }
                profile["online_overlay_edges"] = 0
                if final_tokens_i:
                    online_generated_hits_i = sum(
                        1 for tok in final_tokens_i if self._is_dynamic_generated_token(tok)
                    )
                    profile["online_generated_hits"] = int(online_generated_hits_i)
                    profile["online_generated_any"] = 1 if online_generated_hits_i > 0 else 0
                else:
                    profile["online_generated_hits"] = 0
                    profile["online_generated_any"] = 0
                profile["total_sec"] = (
                    float(profile.get("seed_sec", 0.0))
                    + float(profile.get("graph_sec", 0.0))
                    + float(profile.get("rerank_sec", 0.0))
                    + float(profile.get("finalize_sec", 0.0))
                )
            results[int(state["hdx"])] = (
                {
                    "device_ids": top_ids_t[i],
                    "device_scores": top_vals_t[i].float(),
                    "device_mask": out_mask_t[i],
                    "device_attn_keys": key_attn_t,
                    "device_attn_values": value_t,
                    "cpu_tokens": final_tokens_i if out_ids_cpu is not None else None,
                    "final_count": count_i,
                    "candidate_count": int(dyn_len),
                    "rounds": 0,
                    "frontier_at_stop": 0,
                    "forced_seeds": 0,
                    "kernel_phase_cycles": {
                        "init": 0,
                        "score": 0,
                        "merge": 0,
                        "expand": 0,
                        "finalize": 0,
                    },
                    "kernel_work": {
                        "scored": int(dyn_len),
                        "edges": 0,
                        "accepted": count_i,
                        "frontier_expanded": 0,
                        "find_probes": 0,
                    },
                },
                profile,
            )

        return results

    def _maybe_record_oracle_debug_fullgpu(self, layer_idx: int, kv_hdx: int, head_ids, token_results):
        if not self.oracle_debug_enable:
            return
        if self.oracle_answer_start_pos is None:
            return
        answer_step = int(max(0, int(self.decode_pos) - int(self.oracle_answer_start_pos)))
        for hdx in head_ids:
            payload, _profile = token_results.get(hdx, ({}, None))
            if not isinstance(payload, dict):
                continue
            tokens = payload.get("cpu_tokens")
            if tokens is None and "device_ids" in payload:
                take = max(0, int(payload.get("final_count", 0)))
                if take > 0:
                    tokens = [
                        int(tok) for tok in payload["device_ids"][:take].detach().cpu().tolist()
                        if int(tok) >= 0
                    ]
            self.oracle_debug_records.append(
                {
                    "step": int(answer_step),
                    "decode_pos": int(self.decode_pos),
                    "layer": int(layer_idx),
                    "kv_head": int(kv_hdx),
                    "head": int(hdx),
                    "tokens": [int(tok) for tok in (tokens or [])],
                }
            )

    def _maybe_record_oracle_compare_fullgpu(
        self,
        layer_idx: int,
        kv_hdx: int,
        head_ids,
        q_batch_t: torch.Tensor,
        payloads,
        static_k: torch.Tensor,
        static_v: torch.Tensor,
        dyn_k: torch.Tensor,
        dyn_v: torch.Tensor,
        mask_t: torch.Tensor,
        scores: torch.Tensor,
        attn: torch.Tensor,
        dyn_ids_t: torch.Tensor = None,
        keep_counts_t: torch.Tensor = None,
        adaptive_mass_bounds_t: torch.Tensor = None,
        adaptive_upper_scores_t: torch.Tensor = None,
        adaptive_candidate_counts_t: torch.Tensor = None,
        adaptive_dynamic_span: int = None,
        sparse_out_batch: torch.Tensor = None,
    ):
        if not self.oracle_compare_enable:
            return
        if self.oracle_answer_start_pos is None:
            return
        answer_step = int(max(0, int(self.decode_pos) - int(self.oracle_answer_start_pos)))
        if answer_step > 0:
            return
        device = q_batch_t.device
        total_tokens = int(self._decode_token_limit())
        prefix_len = min(int(self.dynamic_start), int(total_tokens))
        scale = 1.0 / math.sqrt(self.head_dim)
        static_len = int(static_k.shape[0])
        for idx, hdx in enumerate(head_ids):
            dense_scores_static = torch.matmul(q_batch_t[idx:idx + 1], static_k.transpose(0, 1)) * scale
            dense_scores_dyn = torch.matmul(q_batch_t[idx:idx + 1], self._get_decode_attn_key_tensor_cuda(layer_idx, kv_hdx)[self.dynamic_start:self.dynamic_end].transpose(0, 1)) * scale
            dense_scores = torch.cat([dense_scores_static.float(), dense_scores_dyn.float()], dim=-1)
            dense_attn = torch.softmax(dense_scores, dim=-1).squeeze(0)
            sparse_attn = attn[idx]
            sparse_mask = mask_t[idx]
            oracle_dyn_mass = 0.0
            oracle_dense_masses = []
            dyn_ids = []
            if dyn_ids_t is not None and keep_counts_t is not None:
                take = int(keep_counts_t[idx].item())
                if take > 0:
                    dyn_ids_row = dyn_ids_t[idx, :take]
                    valid_mask = (dyn_ids_row >= int(self.dynamic_start)) & (dyn_ids_row < int(self.dynamic_end))
                    valid_ids_t = dyn_ids_row[valid_mask].to(torch.long)
                    if int(valid_ids_t.numel()) > 0:
                        pos_t = (valid_ids_t - int(self.dynamic_start) + static_len).to(torch.long)
                        mass_t = dense_attn.index_select(0, pos_t)
                        oracle_dyn_mass = float(mass_t.sum().item())
                        valid_ids_cpu = valid_ids_t.detach().cpu().tolist()
                        mass_cpu = mass_t.detach().cpu().tolist()
                        dyn_ids = [int(tok) for tok in valid_ids_cpu]
                        oracle_dense_masses = [
                            (int(tok), float(mass))
                            for tok, mass in zip(valid_ids_cpu[:8], mass_cpu[:8])
                        ]
            else:
                dyn_ids = payloads[idx].get("cpu_tokens") or []
                if not dyn_ids and "device_ids" in payloads[idx]:
                    take = max(0, int(payloads[idx].get("final_count", 0)))
                    if take > 0:
                        dyn_ids = [
                            int(tok) for tok in payloads[idx]["device_ids"][:take].detach().cpu().tolist()
                            if int(tok) >= 0
                        ]
                if dyn_ids:
                    for tok in dyn_ids:
                        tok = int(tok)
                        if tok < int(self.dynamic_start) or tok >= int(self.dynamic_end):
                            continue
                        pos = static_len + (tok - int(self.dynamic_start))
                        mass = float(dense_attn[pos].item())
                        oracle_dyn_mass += mass
                        oracle_dense_masses.append((tok, mass))
            dense_v_cat = torch.cat([static_v, self._get_decode_value_tensor_cuda(layer_idx, kv_hdx)[self.dynamic_start:self.dynamic_end]], dim=0)
            sparse_v_cat = torch.cat([static_v, dyn_v[idx]], dim=0)
            dense_out = torch.matmul(dense_attn.unsqueeze(0).to(dense_v_cat.dtype), dense_v_cat).squeeze(0)
            if sparse_out_batch is not None:
                sparse_out = sparse_out_batch[idx].float()
            else:
                sparse_out = torch.matmul(sparse_attn.unsqueeze(0).to(sparse_v_cat.dtype), sparse_v_cat).squeeze(0)
            total_dynamic_mass = float(dense_attn[static_len:].sum().item()) if dense_attn.numel() > static_len else 0.0
            omitted_dynamic_mass = max(0.0, total_dynamic_mass - oracle_dyn_mass)
            adaptive_mass_bound = (
                float(adaptive_mass_bounds_t[idx].item())
                if adaptive_mass_bounds_t is not None
                else float(payloads[idx].get("adaptive_mass_bound", -1.0))
            )
            adaptive_upper_score_bound = (
                float(adaptive_upper_scores_t[idx].item())
                if adaptive_upper_scores_t is not None
                else float(payloads[idx].get("adaptive_upper_score_bound", float("nan")))
            )
            adaptive_candidate_count = (
                int(adaptive_candidate_counts_t[idx].item())
                if adaptive_candidate_counts_t is not None
                else int(payloads[idx].get("adaptive_candidate_count", 0))
            )
            adaptive_keep_count = (
                int(keep_counts_t[idx].item())
                if keep_counts_t is not None
                else int(payloads[idx].get("adaptive_keep_count", len(dyn_ids)))
            )
            adaptive_dynamic_span_i = (
                int(adaptive_dynamic_span)
                if adaptive_dynamic_span is not None
                else int(payloads[idx].get("adaptive_dynamic_span", 0))
            )
            self.oracle_compare_records.append(
                {
                    "step": int(answer_step),
                    "decode_pos": int(self.decode_pos),
                    "layer": int(layer_idx),
                    "kv_head": int(kv_hdx),
                    "head": int(hdx),
                    "oracle_dyn_mass": float(oracle_dyn_mass),
                    "total_dynamic_mass": float(total_dynamic_mass),
                    "omitted_dynamic_mass": float(omitted_dynamic_mass),
                    "oracle_token_count": int(len(dyn_ids)),
                    "sparse_dynamic_count": int(int(sparse_mask.sum().item())),
                    "dense_sparse_out_l2": float(torch.norm(dense_out - sparse_out).item()),
                    "top_oracle_dense_mass": oracle_dense_masses[:8],
                    "adaptive_mass_bound": float(adaptive_mass_bound),
                    "adaptive_upper_score_bound": float(adaptive_upper_score_bound),
                    "adaptive_dynamic_span": int(adaptive_dynamic_span_i),
                    "adaptive_candidate_count": int(adaptive_candidate_count),
                    "adaptive_keep_count": int(adaptive_keep_count),
                }
            )

    def _get_dynamic_attn_score_upper_bound_fullgpu(
        self,
        layer_idx: int,
        kv_hdx: int,
        q_attn: torch.Tensor,
    ) -> float:
        upper_start = time.perf_counter() if self.decode_profile else None
        dyn_start = int(self.dynamic_start)
        dyn_end = int(self.dynamic_end)
        dyn_len = max(0, dyn_end - dyn_start)
        if dyn_len <= 0:
            return float("-inf")
        attn_key_t = self._get_decode_attn_key_tensor_cuda(layer_idx, kv_hdx)
        if attn_key_t is None:
            return float("inf")
        dyn_keys_t = attn_key_t[dyn_start:dyn_end]
        if int(dyn_keys_t.shape[0]) <= 0:
            return float("-inf")
        q_vec = q_attn.squeeze(0).float()
        q_norm = float(torch.linalg.vector_norm(q_vec).item())
        if not math.isfinite(q_norm) or q_norm <= 0.0:
            return float("-inf")
        max_k_norm = float(torch.linalg.vector_norm(dyn_keys_t.float(), dim=-1).max().item())
        if not math.isfinite(max_k_norm) or max_k_norm <= 0.0:
            return float("-inf")
        bound = float((q_norm * max_k_norm) / math.sqrt(float(self.head_dim)))
        if self.decode_profile and upper_start is not None:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(q_attn.device)
            self._decode_profile_stats["adaptive_upper_bound_sec"] += (
                time.perf_counter() - upper_start
            )
        return bound

    def _select_dynamic_budget_count_fullgpu(
        self,
        layer_idx: int,
        kv_hdx: int,
        q_attn: torch.Tensor,
        static_k: torch.Tensor,
        payload,
        dynamic_score_upper: float,
    ):
        count_i = int(payload.get("final_count", 0))
        if not self.dynamic_budget_enable or count_i <= 0:
            return count_i, None, {}

        device = q_attn.device
        ids_t = payload["device_ids"][:count_i].to(torch.long)
        if int(ids_t.numel()) <= 0:
            return count_i, None, {}

        attn_key_t = payload.get("device_attn_keys")
        if attn_key_t is None:
            return count_i, None, {}
        score_start = time.perf_counter() if self.decode_profile else None
        cand_k = torch.index_select(attn_key_t, 0, ids_t)
        q_vec = q_attn.squeeze(0).float()
        scale = 1.0 / math.sqrt(float(self.head_dim))
        dyn_scores = torch.matmul(cand_k.float(), q_vec) * scale
        if self.decode_profile and score_start is not None:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["adaptive_candidate_score_sec"] += (
                time.perf_counter() - score_start
            )
        if dyn_scores.numel() <= 0:
            return count_i, None, {}
        sort_start = time.perf_counter() if self.decode_profile else None
        dyn_scores, sort_idx = torch.sort(dyn_scores.float(), descending=True)
        if self.decode_profile and sort_start is not None:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["adaptive_sort_sec"] += (
                time.perf_counter() - sort_start
            )

        min_keep = min(count_i, max(1, int(self.dynamic_budget_min)))
        search_space = max(0, int(self.dynamic_end - self.dynamic_start))
        unseen_count = max(0, int(search_space) - int(count_i))
        static_start = time.perf_counter() if self.decode_profile else None
        static_scores = (torch.matmul(q_attn, static_k.transpose(0, 1)) * scale).float().squeeze(0)
        static_logz = torch.logsumexp(static_scores, dim=0) if static_scores.numel() > 0 else None
        if self.decode_profile and static_start is not None:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["adaptive_static_logz_sec"] += (
                time.perf_counter() - static_start
            )
        last_bound = 0.0
        select_start = time.perf_counter() if self.decode_profile else None

        for keep in range(int(min_keep), int(count_i) + 1):
            kept = dyn_scores[:keep]
            omitted = dyn_scores[keep:]
            kept_logz = torch.logsumexp(kept, dim=0)
            tail_terms = []
            if omitted.numel() > 0:
                tail_terms.append(torch.logsumexp(omitted, dim=0))
            if unseen_count > 0 and math.isfinite(float(dynamic_score_upper)):
                unseen_log = float(dynamic_score_upper) + math.log(float(unseen_count))
                tail_terms.append(torch.tensor(float(unseen_log), dtype=torch.float32, device=kept.device))
            if not tail_terms:
                stats = {
                    "mass_bound": 0.0,
                    "upper_score_bound": float(dynamic_score_upper),
                    "dynamic_span": int(search_space),
                    "candidate_count": int(count_i),
                }
                return int(keep), sort_idx, stats
            tail_logz = torch.logsumexp(torch.stack(tail_terms), dim=0)
            denom_terms = [kept_logz, tail_logz]
            if static_logz is not None:
                denom_terms.append(static_logz.to(kept.device))
            denom_logz = torch.logsumexp(torch.stack(denom_terms), dim=0)
            omitted_mass_bound = float(torch.exp(tail_logz - denom_logz).item())
            last_bound = omitted_mass_bound
            if omitted_mass_bound <= float(self.dynamic_budget_target_omass):
                stats = {
                    "mass_bound": float(omitted_mass_bound),
                    "upper_score_bound": float(dynamic_score_upper),
                    "dynamic_span": int(search_space),
                    "candidate_count": int(count_i),
                }
                if self.decode_profile and select_start is not None:
                    if self.fullgpu_profile_sync:
                        torch.cuda.synchronize(device)
                    self._decode_profile_stats["adaptive_select_sec"] += (
                        time.perf_counter() - select_start
                    )
                return int(keep), sort_idx, stats

        stats = {
            "mass_bound": float(last_bound),
            "upper_score_bound": float(dynamic_score_upper),
            "dynamic_span": int(search_space),
            "candidate_count": int(count_i),
        }
        if self.decode_profile and select_start is not None:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["adaptive_select_sec"] += (
                time.perf_counter() - select_start
            )
        return int(count_i), sort_idx, stats

    def _apply_dynamic_budget_fullgpu_group(
        self,
        layer_idx: int,
        kv_hdx: int,
        head_ids,
        payloads,
        q_batch_t: torch.Tensor,
        ids_t: torch.Tensor,
        mask_t: torch.Tensor,
        static_scores: torch.Tensor,
        dyn_scores: torch.Tensor,
        dyn_k: torch.Tensor,
        value_t: torch.Tensor,
        scale: float,
    ):
        if not self.dynamic_budget_enable:
            return ids_t, mask_t, dyn_scores, value_t[ids_t.clamp_min(0)], None
        adaptive_start = time.perf_counter() if self.decode_profile else None
        device = q_batch_t.device
        upper_start = time.perf_counter() if self.decode_profile else None
        max_k_norm_t = self._get_dynamic_attn_max_norm_fullgpu(layer_idx, kv_hdx).to(device=device)
        q_norms_t = torch.linalg.vector_norm(q_batch_t.float(), dim=-1)
        upper_scores_t = q_norms_t * max_k_norm_t * float(scale)
        if self.decode_profile and upper_start is not None:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["adaptive_upper_bound_sec"] += (
                time.perf_counter() - upper_start
            )

        static_start = time.perf_counter() if self.decode_profile else None
        static_logz_t = torch.logsumexp(static_scores.float(), dim=-1) if static_scores.numel() > 0 else None
        if self.decode_profile and static_start is not None:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["adaptive_static_logz_sec"] += (
                time.perf_counter() - static_start
            )

        sort_start = time.perf_counter() if self.decode_profile else None
        sortable_scores = dyn_scores.float().masked_fill(~mask_t, float("-inf"))
        sorted_scores_t, sort_idx_t = torch.sort(sortable_scores, dim=1, descending=True)
        sorted_mask_t = torch.gather(mask_t, 1, sort_idx_t)
        sorted_ids_t = torch.gather(ids_t, 1, sort_idx_t)
        candidate_counts_t = torch.as_tensor(
            [
                max(0, min(int(payload.get("final_count", 0)), int(sorted_scores_t.shape[1])))
                if isinstance(payload, dict) else 0
                for payload in payloads
            ],
            dtype=torch.long,
            device=device,
        )
        if self.decode_profile and sort_start is not None:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["adaptive_sort_sec"] += (
                time.perf_counter() - sort_start
            )

        select_start = time.perf_counter() if self.decode_profile else None
        search_space = max(0, int(self.dynamic_end - self.dynamic_start))
        target_omass = float(self.dynamic_budget_target_omass)
        min_keep_global = max(1, int(self.dynamic_budget_min))
        if self.dynamic_budget_mode == "cuda":
            keep_counts_t, mass_bounds_t = adaptive_budget_select_cuda(
                sorted_scores_t,
                sorted_mask_t,
                static_logz_t if static_logz_t is not None else torch.zeros(
                    (sorted_scores_t.shape[0],),
                    dtype=torch.float32,
                    device=device,
                ),
                upper_scores_t,
                min_keep=min_keep_global,
                dynamic_span=int(search_space),
                target_omass=float(target_omass),
            )
            keep_counts_t = torch.minimum(keep_counts_t.to(dtype=torch.long), candidate_counts_t)
            keep_counts_t = keep_counts_t.clamp_min(0)
        else:
            keep_counts = []
            mass_bounds = []
            for row_idx in range(sorted_scores_t.shape[0]):
                old_count = int(candidate_counts_t[row_idx].item())
                if old_count <= 0:
                    keep_counts.append(0)
                    mass_bounds.append(0.0)
                    continue
                min_keep = min(old_count, min_keep_global)
                scores_i = sorted_scores_t[row_idx, :old_count]
                prefix_logz = torch.logcumsumexp(scores_i, dim=0)
                omitted_local = torch.full_like(scores_i, float("-inf"))
                if old_count > 1:
                    suffix_logz = torch.logcumsumexp(scores_i.flip(0), dim=0).flip(0)
                    omitted_local[:-1] = suffix_logz[1:]
                unseen_count = max(0, int(search_space) - int(old_count))
                tail_logz = omitted_local
                upper_i = upper_scores_t[row_idx]
                if unseen_count > 0 and torch.isfinite(upper_i):
                    unseen_log_t = torch.full_like(scores_i, float("-inf"))
                    unseen_log_t[:] = upper_i + math.log(float(unseen_count))
                    tail_logz = torch.logaddexp(tail_logz, unseen_log_t)
                denom_logz = torch.logaddexp(prefix_logz, tail_logz)
                if static_logz_t is not None:
                    denom_logz = torch.logaddexp(denom_logz, static_logz_t[row_idx].expand_as(denom_logz))
                omitted_bounds = torch.exp(tail_logz - denom_logz)
                valid_slice = omitted_bounds[min_keep - 1:]
                keep = int(old_count)
                mass_bound = float(omitted_bounds[-1].item()) if int(omitted_bounds.numel()) > 0 else 0.0
                if int(valid_slice.numel()) > 0:
                    good = torch.nonzero(valid_slice <= target_omass, as_tuple=False)
                    if int(good.numel()) > 0:
                        keep = int(min_keep + int(good[0, 0].item()))
                        mass_bound = float(valid_slice[int(good[0, 0].item())].item())
                keep_counts.append(int(keep))
                mass_bounds.append(float(mass_bound))
            keep_counts_t = torch.as_tensor(keep_counts, dtype=torch.long, device=device)
            mass_bounds_t = torch.as_tensor(mass_bounds, dtype=torch.float32, device=device)
        if self.decode_profile and select_start is not None:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["adaptive_select_sec"] += (
                time.perf_counter() - select_start
            )

        reorder_start = time.perf_counter() if self.decode_profile else None
        max_keep = int(keep_counts_t.max().item()) if int(keep_counts_t.numel()) > 0 else 0
        max_keep = max(0, min(max_keep, int(sorted_scores_t.shape[1])))
        if max_keep > 0:
            sorted_ids_t = sorted_ids_t[:, :max_keep]
            sorted_scores_t = sorted_scores_t[:, :max_keep]
            sorted_mask_t = sorted_mask_t[:, :max_keep]
            col_idx_t = torch.arange(max_keep, dtype=torch.long, device=device).unsqueeze(0)
            new_mask_t = sorted_mask_t & (col_idx_t < keep_counts_t.unsqueeze(1))
            sorted_scores_t = sorted_scores_t.masked_fill(~new_mask_t, float("-inf"))
            sorted_dyn_v = value_t[sorted_ids_t.clamp_min(0)]
        else:
            batch = int(sorted_scores_t.shape[0])
            sorted_ids_t = sorted_ids_t[:, :0]
            sorted_scores_t = sorted_scores_t[:, :0]
            new_mask_t = sorted_mask_t[:, :0]
            sorted_dyn_v = value_t.new_empty((batch, 0, value_t.shape[-1]))
        if self.decode_profile and reorder_start is not None:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["adaptive_reorder_sec"] += (
                time.perf_counter() - reorder_start
            )
        if self.decode_profile and adaptive_start is not None:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["adaptive_total_sec"] += (
                time.perf_counter() - adaptive_start
            )
        budget_meta = {
            "keep_counts_t": keep_counts_t,
            "mass_bounds_t": mass_bounds_t,
            "upper_scores_t": upper_scores_t,
            "candidate_counts_t": candidate_counts_t,
            "dynamic_span": int(search_space),
        }
        return sorted_ids_t, new_mask_t, sorted_scores_t, sorted_dyn_v, budget_meta

    def _get_fullgpu_static_kv(self, layer_idx: int, kv_hdx: int):
        if not self.growing_static_suffix:
            return self.static_gpu_keys[layer_idx][kv_hdx], self.static_gpu_values[layer_idx][kv_hdx]

        total_tokens = self._decode_token_limit()
        prefix_len = min(int(self.dynamic_start), int(total_tokens))
        suffix_start = min(max(int(self.dynamic_start), int(self.dynamic_end)), int(total_tokens))

        prefix_k = self.static_gpu_keys[layer_idx][kv_hdx, :prefix_len, :]
        prefix_v = self.static_gpu_values[layer_idx][kv_hdx, :prefix_len, :]

        attn_key_cache = self._get_decode_attn_key_tensor_cuda(layer_idx, kv_hdx)
        value_cache = self._get_decode_value_tensor_cuda(layer_idx, kv_hdx)
        suffix_k = attn_key_cache[suffix_start:total_tokens, :]
        suffix_v = value_cache[suffix_start:total_tokens, :]

        if prefix_len <= 0:
            return suffix_k, suffix_v
        if suffix_start >= total_tokens:
            return prefix_k, prefix_v
        return torch.cat([prefix_k, suffix_k], dim=0), torch.cat([prefix_v, suffix_v], dim=0)

    def _apply_fullgpu_group_attention(
        self,
        layer_idx: int,
        kv_hdx: int,
        head_ids,
        q_attn_by_head,
        token_results,
        scale: float,
    ):
        device = self.layer_mapping[str(layer_idx)]
        payloads = []
        profiles = []
        q_batch = []
        for hdx in head_ids:
            payload, profile = token_results.get(hdx, ({}, None))
            payloads.append(payload)
            profiles.append(profile)
            q_batch.append(q_attn_by_head[hdx].squeeze(0))

        static_k, static_v = self._get_fullgpu_static_kv(layer_idx, kv_hdx)
        q_batch_t = torch.stack(q_batch, dim=0)  # [q_count, dim]
        gather_start = time.perf_counter() if self.decode_profile else None
        if self.decode_profile and self.fullgpu_profile_sync:
            torch.cuda.synchronize(device)
            gather_start = time.perf_counter()

        key_t = payloads[0]["device_attn_keys"]
        value_t = payloads[0]["device_attn_values"]
        ids_t = torch.stack([p["device_ids"].to(torch.long) for p in payloads], dim=0)
        mask_t = torch.stack([p["device_mask"] for p in payloads], dim=0)
        gather_ids_t = ids_t.clamp_min(0)
        dyn_k = key_t[gather_ids_t]
        dyn_v = None

        if self.decode_profile:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["gather_total_sec"] += (time.perf_counter() - gather_start)

        attn_start = time.perf_counter() if self.decode_profile else None
        if self.decode_profile and self.fullgpu_profile_sync:
            torch.cuda.synchronize(device)
            attn_start = time.perf_counter()

        static_scores = torch.matmul(q_batch_t, static_k.transpose(0, 1)) * scale
        dyn_scores = torch.bmm(dyn_k, q_batch_t.unsqueeze(-1)).squeeze(-1) * scale
        dyn_scores = dyn_scores.float().masked_fill(~mask_t, float("-inf"))
        budget_meta = None
        if self.dynamic_budget_enable and self.dynamic_budget_mode != "traversal_cuda":
            ids_t, mask_t, dyn_scores, dyn_v, budget_meta = self._apply_dynamic_budget_fullgpu_group(
                layer_idx=layer_idx,
                kv_hdx=kv_hdx,
                head_ids=head_ids,
                payloads=payloads,
                q_batch_t=q_batch_t,
                ids_t=ids_t,
                mask_t=mask_t,
                static_scores=static_scores,
                dyn_scores=dyn_scores,
                dyn_k=dyn_k,
                value_t=value_t,
                scale=scale,
            )
        else:
            dyn_v = value_t[gather_ids_t]
        scores = torch.cat([static_scores.float(), dyn_scores], dim=-1)
        static_v_expand = static_v.unsqueeze(0).expand(len(head_ids), -1, -1)
        v_cat = torch.cat([static_v_expand, dyn_v], dim=1)
        attn = torch.softmax(scores, dim=-1).to(v_cat.dtype)
        out_batch = torch.bmm(attn.unsqueeze(1), v_cat).squeeze(1)
        tail_mass_t = None
        if self.dynamic_tail_enable and self.dynamic_budget_enable:
            if budget_meta is not None:
                tail_mass_t = budget_meta["mass_bounds_t"].to(device=device, dtype=out_batch.dtype)
            elif self.dynamic_budget_mode == "traversal_cuda":
                tail_mass_t = torch.as_tensor(
                    [
                        float(p.get("adaptive_mass_bound", 0.0)) if isinstance(p, dict) else 0.0
                        for p in payloads
                    ],
                    dtype=out_batch.dtype,
                    device=device,
                )
            if tail_mass_t is not None:
                tail_mass_t = tail_mass_t.clamp(0.0, 0.99)
                if self.dynamic_tail_mode == "zero":
                    out_batch = (1.0 - tail_mass_t.unsqueeze(1)) * out_batch
                else:
                    tail_v = self._get_dynamic_tail_value_fullgpu(layer_idx, kv_hdx).to(device=device, dtype=out_batch.dtype)
                    if int(tail_v.numel()) > 0:
                        out_batch = (1.0 - tail_mass_t.unsqueeze(1)) * out_batch + tail_mass_t.unsqueeze(1) * tail_v.unsqueeze(0)
        keep_counts_t = (
            budget_meta["keep_counts_t"]
            if budget_meta is not None
            else mask_t.to(dtype=torch.long).sum(dim=1)
        )
        self._record_online_graph_query_centroid_fullgpu(
            layer_idx=layer_idx,
            kv_hdx=kv_hdx,
            q_batch_t=q_batch_t,
            payloads=payloads,
            attn=attn,
            dyn_ids_t=ids_t,
            keep_counts_t=keep_counts_t,
        )
        self._maybe_record_oracle_compare_fullgpu(
            layer_idx=layer_idx,
            kv_hdx=kv_hdx,
            head_ids=head_ids,
            q_batch_t=q_batch_t,
            payloads=payloads,
            static_k=static_k,
            static_v=static_v,
            dyn_k=dyn_k,
            dyn_v=dyn_v,
            mask_t=mask_t,
            scores=scores,
            attn=attn,
            dyn_ids_t=ids_t,
            keep_counts_t=keep_counts_t,
            adaptive_mass_bounds_t=(budget_meta["mass_bounds_t"] if budget_meta is not None else None),
            adaptive_upper_scores_t=(budget_meta["upper_scores_t"] if budget_meta is not None else None),
            adaptive_candidate_counts_t=(budget_meta["candidate_counts_t"] if budget_meta is not None else None),
            adaptive_dynamic_span=(budget_meta["dynamic_span"] if budget_meta is not None else None),
            sparse_out_batch=out_batch,
        )

        if self.decode_profile:
            if self.fullgpu_profile_sync:
                torch.cuda.synchronize(device)
            self._decode_profile_stats["attn_total_sec"] += (time.perf_counter() - attn_start)

        outputs = []
        empty_heads = 0
        dynamic_counts = []
        for idx, hdx in enumerate(head_ids):
            payload = payloads[idx]
            retrieve_profile = profiles[idx]
            dyn_count = int(keep_counts_t[idx].item())
            if self.dynamic_budget_enable and isinstance(payload, dict):
                payload["device_ids"] = ids_t[idx].to(payload["device_ids"].dtype)
                payload["device_mask"] = mask_t[idx]
                payload["device_scores"] = dyn_scores[idx].to(dtype=torch.float32)
                payload["final_count"] = dyn_count
                payload["cpu_tokens"] = None
                if budget_meta is not None:
                    payload["adaptive_mass_bound"] = float(budget_meta["mass_bounds_t"][idx].item())
                    payload["adaptive_upper_score_bound"] = float(
                        budget_meta["upper_scores_t"][idx].item()
                    )
                    payload["adaptive_dynamic_span"] = int(budget_meta["dynamic_span"])
                    payload["adaptive_candidate_count"] = int(
                        budget_meta["candidate_counts_t"][idx].item()
                    )
                    payload["adaptive_keep_count"] = dyn_count
                    if retrieve_profile is not None:
                        retrieve_profile["adaptive_final_outputs"] = int(dyn_count)
                        retrieve_profile["adaptive_mass_bound"] = float(
                            budget_meta["mass_bounds_t"][idx].item()
                        )
                        retrieve_profile["adaptive_upper_score_bound"] = float(
                            budget_meta["upper_scores_t"][idx].item()
                        )
                if tail_mass_t is not None:
                    payload["dynamic_tail_mass"] = float(tail_mass_t[idx].item())
                    if retrieve_profile is not None:
                        retrieve_profile["dynamic_tail_mass"] = float(tail_mass_t[idx].item())
            if dyn_count == 0:
                empty_heads += 1
            dynamic_counts.append(dyn_count)
            outputs.append(out_batch[idx].unsqueeze(0))
        if self.decode_profile:
            self._decode_profile_stats["heads"] += len(head_ids)
        return outputs, empty_heads, dynamic_counts

    def _retrieve_tokens_roar_cuda_frontier_group(
        self,
        ldx: int,
        kv_hdx: int,
        states,
        update_decode_state: bool = True,
        enforce_seed_floor: bool = True,
    ):
        results = {}
        active_states = []
        for state in states:
            if state.get("empty", False):
                results[int(state["hdx"])] = (list(state.get("tokens", [])), state.get("profile"))
            else:
                active_states.append(state)

        if not active_states:
            return results

        seed_states = [state for state in active_states if "seed_candidate_ids" in state]
        if seed_states:
            seed_q_count = len(seed_states)
            union_tokens = []
            union_pos = {}
            row_members = []
            for state in seed_states:
                row = []
                for tok in state["seed_candidate_ids"]:
                    tok = int(tok)
                    pos = union_pos.get(tok)
                    if pos is None:
                        pos = len(union_tokens)
                        union_pos[tok] = pos
                        union_tokens.append(tok)
                    row.append(int(pos))
                row_members.append(row)

            key_t_seed = self._get_decode_key_tensor_cuda(ldx, kv_hdx)
            if key_t_seed is None:
                raise RuntimeError("roar_cuda_frontier seed key cache unavailable")
            queries_seed = torch.cat([state["q_seed_cuda"] for state in seed_states], dim=0)
            union_ids_t = torch.as_tensor(union_tokens, dtype=torch.long, device=key_t_seed.device)
            seed_keys_t = torch.index_select(key_t_seed, 0, union_ids_t)
            seed_score_start = time.perf_counter()
            seed_scores_t = torch.matmul(queries_seed, seed_keys_t.transpose(0, 1))
            mask = torch.zeros((seed_q_count, len(union_tokens)), dtype=torch.bool, device=key_t_seed.device)
            for i, positions in enumerate(row_members):
                if positions:
                    mask[i, torch.as_tensor(positions, dtype=torch.long, device=key_t_seed.device)] = True
            seed_scores_t = seed_scores_t.masked_fill(~mask, float("-inf"))
            max_seed_k = max(max(1, int(state["seed_k"])) for state in seed_states)
            top_vals_t, top_pos_t = torch.topk(seed_scores_t, k=min(max_seed_k, seed_scores_t.shape[1]), dim=1)
            top_vals = top_vals_t.cpu()
            top_pos = top_pos_t.cpu()
            per_seed_sec = (time.perf_counter() - seed_score_start) / float(max(1, seed_q_count))

            for i, state in enumerate(seed_states):
                seed_ranked = []
                take = min(int(state["seed_k"]), len(state["seed_candidate_ids"]))
                row_vals = top_vals[i]
                row_pos = top_pos[i]
                for j in range(row_pos.shape[0]):
                    score = float(row_vals[j].item())
                    if not math.isfinite(score):
                        continue
                    tok = int(union_tokens[int(row_pos[j].item())])
                    seed_ranked.append((tok, score))
                    if len(seed_ranked) >= take:
                        break
                if not seed_ranked:
                    if state["profile"] is not None:
                        self._finish_decode_retrieve_profile(
                            state["total_start"],
                            state["profile"],
                            "empty_seed_ranked",
                            0,
                            0,
                        )
                    results[int(state["hdx"])] = ([], state["profile"])
                    state["empty"] = True
                    continue
                seed_floor = int(math.ceil(self.token_budget * self.seed_ratio))
                seed_floor = min(self.token_budget, max(0, seed_floor))
                seed_floor = min(seed_floor, len(seed_ranked))
                state["seed_floor"] = int(seed_floor)
                state["selected_seed_set"] = set(tok for tok, _ in seed_ranked[:seed_floor])
                state["init_candidates"] = [int(tok) for tok, _ in seed_ranked]
                state["init_scores"] = [float(score) for _, score in seed_ranked]
                if state["profile"] is not None:
                    state["profile"]["seed_sec"] += per_seed_sec

            active_states = [state for state in active_states if not state.get("empty", False)]
            if not active_states:
                return results

        graph = active_states[0]["graph"]
        if not (isinstance(graph, tuple) and len(graph) >= 2):
            for state in active_states:
                token_ids, retrieve_profile = self._retrieve_tokens(
                    ldx,
                    int(state["hdx"]),
                    state["q_group"],
                    update_decode_state=update_decode_state,
                    enforce_seed_floor=enforce_seed_floor,
                )
                results[int(state["hdx"])] = (token_ids, retrieve_profile)
            return results

        key_t = self._get_decode_key_tensor_cuda(ldx, kv_hdx)
        graph_tensors = self._get_decode_graph_tensors_cuda_device(ldx, kv_hdx)
        if key_t is None or graph_tensors is None:
            raise RuntimeError("roar_cuda_frontier decode state is unavailable")

        q_count = len(active_states)
        max_init = max(len(state["init_candidates"]) for state in active_states)
        queries_seed = torch.cat([state["q_seed_cuda"] for state in active_states], dim=0)
        queries_rank = torch.cat([state["q_rank_cuda"] for state in active_states], dim=0)
        init_ids_t = torch.full((q_count, max_init), -1, dtype=torch.int32, device="cpu")
        init_scores_t = torch.full((q_count, max_init), -1e30, dtype=torch.float32, device="cpu")
        for i, state in enumerate(active_states):
            n = len(state["init_candidates"])
            if n <= 0:
                continue
            init_ids_t[i, :n] = torch.as_tensor(state["init_candidates"], dtype=torch.int32, device="cpu")
            init_scores_t[i, :n] = torch.as_tensor(state["init_scores"], dtype=torch.float32, device="cpu")

        graph_offsets_t, graph_neighbors_t = graph_tensors
        graph_start = time.perf_counter()
        out_ids_t, _out_scores_t, out_counts_t, out_visited_t, out_stop_t = search_roar_graph_csr_cuda_frontier(
            queries_seed=queries_seed,
            queries_rank=queries_rank,
            keys=key_t,
            offsets=graph_offsets_t,
            neighbors=graph_neighbors_t,
            init_ids=init_ids_t,
            init_scores=init_scores_t,
            token_budget=int(self.token_budget),
            candidate_target=int(active_states[0]["candidate_target"]),
            frontier_width=int(self.roar_cuda_frontier_width),
            min_visits=int(self.min_visits),
            max_visits=int(self.max_visits),
            stop_patience=int(self.stop_patience),
            stop_margin=float(self.stop_margin),
            dynamic_start=int(self.dynamic_start),
            dynamic_end=int(self.dynamic_end),
            score_agg=self.rerank_agg,
        )
        graph_elapsed = time.perf_counter() - graph_start
        out_ids = out_ids_t.cpu()
        out_counts = out_counts_t.cpu()
        out_visited = out_visited_t.cpu()
        out_stop = out_stop_t.cpu()
        per_graph_sec = graph_elapsed / float(max(1, q_count))

        for i, state in enumerate(active_states):
            keep = int(out_counts[i].item())
            ranked_tokens = []
            if keep > 0:
                row = out_ids[i, :keep]
                ranked_tokens = [int(tok) for tok in row.tolist() if int(tok) >= 0]
            stop_reason = self._decode_stop_reason_from_code(int(out_stop[i].item()))
            visited_count = int(out_visited[i].item())
            candidate_count = len(ranked_tokens)
            profile = state["profile"]
            if profile is not None:
                profile["graph_sec"] += per_graph_sec
                profile["rerank_sec"] += 0.0
            final, retrieve_profile = self._finalize_decode_seed_state(
                state,
                ranked_tokens,
                stop_reason=stop_reason,
                visited_count=visited_count,
                candidate_count=candidate_count,
                update_decode_state=update_decode_state,
                enforce_seed_floor=enforce_seed_floor,
            )
            if retrieve_profile is not None:
                retrieve_profile["total_sec"] = (
                    float(retrieve_profile.get("seed_sec", 0.0))
                    + float(retrieve_profile.get("graph_sec", 0.0))
                    + float(retrieve_profile.get("rerank_sec", 0.0))
                    + float(retrieve_profile.get("finalize_sec", 0.0))
                )
            results[int(state["hdx"])] = (final, retrieve_profile)

        return results

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
                "final_outputs": 0,
                "online_overlay_edges": 0,
                "online_generated_hits": 0,
                "online_generated_any": 0,
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
        q_seed_cuda = None
        q_rank_cuda = None
        token_limit = self._decode_token_limit()
        if self.decode_backend in {
            "roar_cuda",
            "roar_cuda_v2",
            "roar_cuda_kernel",
            "roar_cuda_fullgpu",
            "roar_cuda_frontier",
            "roar_cuda_beam",
        }:
            device = self.layer_mapping[str(ldx)]
            q_seed_cuda = self._score_transform_torch(q_seed.to(device, non_blocking=True).float())
            q_rank_cuda = self._score_transform_torch(q_group.to(device, non_blocking=True).float())
        seed_scores = {}
        seed_k = max(self.q_knn, self.q_knn * self.seed_k_mult)
        seed_k = min(token_limit, seed_k)
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
                k = self.cpu_keys[ldx][kv_hdx, :token_limit, :].detach().float().cpu()
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
                if tok < 0 or tok >= token_limit:
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
        candidate_target = min(candidate_target, token_limit)

        candidates = []
        seen = set()
        candidate_scores = {}
        visited_count = 0
        ranked_override = None

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
                    lpq = min(token_limit, lpq)
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

            use_cuda_decode = (self.decode_backend == "roar_cuda" and graph_is_csr)
            if use_cuda_decode and not use_cpp_decode:
                try:
                    init_ids_t = torch.as_tensor(
                        [int(tok) for tok in candidates],
                        dtype=torch.int32,
                        device="cpu",
                    )
                    init_scores_t = torch.as_tensor(
                        [float(candidate_scores[int(tok)]) for tok in candidates],
                        dtype=torch.float32,
                        device="cpu",
                    )
                    key_t = self._get_decode_key_tensor_cuda(ldx, kv_hdx)
                    graph_tensors = self._get_decode_graph_tensors_cuda(ldx, kv_hdx)
                    if key_t is None or graph_tensors is None or q_seed_cuda is None or q_rank_cuda is None:
                        raise RuntimeError("roar_cuda decode state is unavailable")
                    graph_offsets_t, graph_neighbors_t = graph_tensors
                    ids_cuda, scores_cuda, meta_cuda = search_roar_graph_csr_cuda(
                        query_seed=q_seed_cuda,
                        query_rank=q_rank_cuda,
                        keys=key_t,
                        offsets=graph_offsets_t,
                        neighbors=graph_neighbors_t,
                        init_ids=init_ids_t,
                        init_scores=init_scores_t,
                        token_budget=int(self.token_budget),
                        candidate_target=int(candidate_target),
                        expand_width=int(self.expand_width),
                        min_visits=int(self.min_visits),
                        max_visits=int(self.max_visits),
                        frontier_topn=int(self.frontier_topn),
                        stop_patience=int(self.stop_patience),
                        stop_margin=float(self.stop_margin),
                        dynamic_start=int(self.dynamic_start),
                        dynamic_end=int(self.dynamic_end),
                        score_agg=self.rerank_agg,
                    )
                    if isinstance(meta_cuda, dict):
                        stop_reason = str(meta_cuda.get("stop_reason", "roar_cuda"))
                        visited_count = int(meta_cuda.get("visited", 0))
                    else:
                        stop_reason = "roar_cuda"
                        visited_count = 0
                    ids_list = [int(x) for x in ids_cuda.detach().cpu().tolist()]
                    scores_list = [float(x) for x in scores_cuda.detach().cpu().tolist()]
                    candidates = []
                    seen = set()
                    candidate_scores = {}
                    ranked_override = []
                    for tok, score in zip(ids_list, scores_list):
                        if tok in static_indices or tok in seen:
                            continue
                        seen.add(tok)
                        candidates.append(tok)
                        candidate_scores[tok] = score
                        ranked_override.append(tok)
                        if len(candidates) >= candidate_target:
                            break
                    if len(candidates) >= candidate_target and stop_reason == "frontier_empty":
                        stop_reason = "candidate_cap"
                except Exception as exc:
                    if self.decode_backend == "roar_cuda":
                        raise
                    if not self._decode_cuda_warned:
                        print(
                            "[RetrievalAttention] WARNING: decode roar cuda search failed; "
                            f"falling back to python traversal. error={exc}"
                        )
                        self._decode_cuda_warned = True
                    use_cuda_decode = False

            if not use_cpp_decode and not use_cuda_decode:
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
                            if 0 <= tok and (tok + 1) < int(graph_offsets.shape[0]):
                                row_start = int(graph_offsets[tok])
                                row_end = int(graph_offsets[tok + 1])
                                base_iter = graph_neighbors[row_start:row_end]
                            else:
                                base_iter = ()
                        else:
                            if 0 <= tok < len(graph):
                                base_iter = graph[tok]
                            else:
                                base_iter = ()
                        for nb in base_iter:
                            nb = int(nb)
                            if nb in static_indices or nb in seen or nb in new_token_set:
                                continue
                            new_tokens.append(nb)
                            new_token_set.add(nb)
                            if len(candidates) + len(new_tokens) >= candidate_target:
                                break
                        overlay_iter = ()
                        if self.online_graph_enable:
                            overlay_iter = self._online_graph_overlay[ldx][kv_hdx].get(int(tok), ())
                            if profile is not None:
                                profile["online_overlay_edges"] = int(profile.get("online_overlay_edges", 0)) + len(overlay_iter)
                        for nb in overlay_iter:
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
        if ranked_override is not None:
            ranked = ranked_override
        elif self.rerank:
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
            profile["final_outputs"] = int(len(final))
            online_generated_hits = sum(1 for tok in final if self._is_dynamic_generated_token(tok))
            profile["online_generated_hits"] = int(online_generated_hits)
            profile["online_generated_any"] = 1 if online_generated_hits > 0 else 0
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
        token_results = {}
        profile_results = {}
        q_attn_by_head = {}

        if self.decode_backend == "roar_cuda_fullgpu" and (
            self.retrieval_head_mode != "q_head"
            or self.query_mode != "per_head"
            or self.score_mode != "ip"
        ):
            raise RuntimeError(
                "roar_cuda_fullgpu currently supports only retrieval_head_mode=q_head, "
                "query_mode=per_head, and score_mode=ip."
            )

        use_roar_cuda_v2 = (
            self.decode_backend == "roar_cuda_v2"
            and self.retrieval_head_mode == "q_head"
            and self.query_mode == "per_head"
        )
        use_roar_cuda_kernel = (
            self.decode_backend == "roar_cuda_kernel"
            and self.retrieval_head_mode == "q_head"
            and self.query_mode == "per_head"
        )
        use_roar_cuda_fullgpu = (
            self.decode_backend == "roar_cuda_fullgpu"
            and self.retrieval_head_mode == "q_head"
            and self.query_mode == "per_head"
            and self.score_mode == "ip"
        )
        use_roar_cuda_frontier = (
            self.decode_backend == "roar_cuda_frontier"
            and self.retrieval_head_mode == "q_head"
            and self.query_mode == "per_head"
        )
        use_roar_cuda_beam = (
            self.decode_backend == "roar_cuda_beam"
            and self.retrieval_head_mode == "q_head"
            and self.query_mode == "per_head"
        )
        if (
            self.online_graph_enable
            and self.decode_backend not in {"python", "auto", "roar_cuda_fullgpu"}
            and not self._online_graph_warned
        ):
            print(
                "[RetrievalAttention] WARNING: online decode graph overlay currently only affects the "
                "python/auto decode traversal path."
            )
            self._online_graph_warned = True
        if self.decode_profile:
            self._decode_profile_stats["other_setup_sec"] += (time.perf_counter() - compute_start)

        if use_roar_cuda_v2 or use_roar_cuda_kernel or use_roar_cuda_fullgpu or use_roar_cuda_frontier or use_roar_cuda_beam:
            for kv_hdx in range(self.kv_head):
                states = []
                for local_h in range(self.group_size):
                    hdx = kv_hdx * self.group_size + local_h
                    if hdx >= head_count:
                        break
                    q_group = q[hdx]
                    q_attn_by_head[hdx] = q_group.unsqueeze(0)
                    state_prep_start = time.perf_counter() if self.decode_profile else None
                    if use_roar_cuda_fullgpu:
                        state = self._prepare_decode_seed_state_fullgpu(
                            layer_idx,
                            hdx,
                            q_group,
                        )
                    else:
                        state = self._prepare_decode_seed_state(
                            layer_idx,
                            hdx,
                            q_group,
                            defer_seed_scoring=True,
                        )
                    if self.decode_profile:
                        self._decode_profile_stats["other_state_prep_sec"] += (
                            time.perf_counter() - state_prep_start
                        )
                    states.append(state)
                if use_roar_cuda_kernel:
                    group_results = self._retrieve_tokens_roar_cuda_kernel_group(
                        layer_idx,
                        kv_hdx,
                        states,
                        update_decode_state=True,
                        enforce_seed_floor=True,
                    )
                elif use_roar_cuda_fullgpu:
                    if self.oracle_retrieval_enable:
                        group_results = self._retrieve_tokens_oracle_fullgpu_group(
                            layer_idx,
                            kv_hdx,
                            states,
                        )
                    else:
                        group_results = self._retrieve_tokens_roar_cuda_fullgpu_group(
                            layer_idx,
                            kv_hdx,
                            states,
                            update_decode_state=True,
                            enforce_seed_floor=True,
                        )
                        self._maybe_compare_fullgpu_reference(
                            layer_idx=layer_idx,
                            kv_hdx=kv_hdx,
                            q=q,
                            head_count=head_count,
                            fg_results=group_results,
                        )
                elif use_roar_cuda_frontier:
                    group_results = self._retrieve_tokens_roar_cuda_frontier_group(
                        layer_idx,
                        kv_hdx,
                        states,
                        update_decode_state=True,
                        enforce_seed_floor=True,
                    )
                elif use_roar_cuda_beam:
                    group_results = self._retrieve_tokens_roar_cuda_beam_group(
                        layer_idx,
                        kv_hdx,
                        states,
                        update_decode_state=True,
                        enforce_seed_floor=True,
                    )
                else:
                    group_results = self._retrieve_tokens_roar_cuda_v2_group(
                        layer_idx,
                        kv_hdx,
                        states,
                        update_decode_state=True,
                        enforce_seed_floor=True,
                    )
                bookkeeping_start = time.perf_counter() if self.decode_profile else None
                token_results.update(group_results)
                if self.decode_profile:
                    self._decode_profile_stats["other_group_bookkeeping_sec"] += (
                        time.perf_counter() - bookkeeping_start
                    )
        else:
            for hdx in range(head_count):
                kv_hdx = self._retrieval_head_to_kv_head(hdx)
                if self.retrieval_head_mode == "q_head":
                    q_group = q[hdx]
                    q_attn = q_group.unsqueeze(0)
                else:
                    q_group = q_grouped[hdx]
                    q_attn = q_group
                q_attn_by_head[hdx] = q_attn
                token_ids, retrieve_profile = self._retrieve_tokens(layer_idx, hdx, q_group)
                token_results[hdx] = (token_ids, retrieve_profile)

        if use_roar_cuda_fullgpu:
            for kv_hdx in range(self.kv_head):
                head_ids = []
                for local_h in range(self.group_size):
                    hdx = kv_hdx * self.group_size + local_h
                    if hdx >= head_count:
                        break
                    head_ids.append(hdx)
                if not head_ids:
                    continue
                bookkeeping_start = time.perf_counter() if self.decode_profile else None
                provenance_map = {}
                if self.online_graph_enable:
                    provenance_start = time.perf_counter() if self.decode_profile else None
                    provenance_payloads = []
                    for hdx in head_ids:
                        token_ids, _ = token_results.get(hdx, ({}, None))
                        provenance_payloads.append((hdx, token_ids))
                    provenance_map = self._fullgpu_payload_token_ids_group(provenance_payloads)
                    if self.decode_profile:
                        self._decode_profile_stats["online_provenance_d2h_sec"] += (
                            time.perf_counter() - provenance_start
                        )
                for hdx in head_ids:
                    token_ids, retrieve_profile = token_results.get(hdx, ({}, None))
                    if self.online_graph_enable:
                        self._record_online_graph_provenance(
                            layer_idx,
                            hdx,
                            provenance_map.get(hdx, ()),
                        )
                if self.decode_profile:
                    self._decode_profile_stats["other_group_bookkeeping_sec"] += (
                        time.perf_counter() - bookkeeping_start
                    )
                group_outputs, group_empty, group_dyn_counts = self._apply_fullgpu_group_attention(
                    layer_idx=layer_idx,
                    kv_hdx=kv_hdx,
                    head_ids=head_ids,
                    q_attn_by_head=q_attn_by_head,
                    token_results=token_results,
                    scale=scale,
                )
                if self.oracle_retrieval_enable:
                    self._maybe_record_oracle_debug_fullgpu(
                        layer_idx=layer_idx,
                        kv_hdx=kv_hdx,
                        head_ids=head_ids,
                        token_results=token_results,
                    )
                for hdx in head_ids:
                    _token_ids, retrieve_profile = token_results.get(hdx, ({}, None))
                    self._accumulate_decode_retrieve_profile(retrieve_profile)
                empty_heads += group_empty
                dynamic_counts.extend(group_dyn_counts)
                outputs.extend(group_outputs)

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
            output_start = time.perf_counter() if self.decode_profile else None
            if self.decode_profile:
                self._decode_profile_stats["calls"] += 1
                self._decode_profile_stats["compute_total_sec"] += (time.perf_counter() - compute_start)
            out = torch.cat(outputs, dim=0).view(1, 1, self.num_heads, self.head_dim)
            if self.decode_profile:
                self._decode_profile_stats["other_output_sec"] += (time.perf_counter() - output_start)
            return out

        for hdx in range(head_count):
            kv_hdx = self._retrieval_head_to_kv_head(hdx)
            if hdx not in q_attn_by_head:
                if self.retrieval_head_mode == "q_head":
                    q_attn_by_head[hdx] = q[hdx].unsqueeze(0)
                else:
                    q_attn_by_head[hdx] = q_grouped[hdx]
            q_attn = q_attn_by_head[hdx]
            token_ids, retrieve_profile = token_results.get(hdx, ([], None))
            self._record_online_graph_provenance(layer_idx, hdx, token_ids)
            bookkeeping_start = time.perf_counter() if self.decode_profile else None
            self._accumulate_decode_retrieve_profile(retrieve_profile)
            if self.decode_profile:
                self._decode_profile_stats["other_group_bookkeeping_sec"] += (
                    time.perf_counter() - bookkeeping_start
                )
            gather_start = time.perf_counter() if self.decode_profile else None
            dyn_mask = None
            if isinstance(token_ids, dict) and "device_ids" in token_ids:
                if self.decode_profile and self.fullgpu_profile_sync:
                    torch.cuda.synchronize(device)
                    gather_start = time.perf_counter()
                dyn_mask = token_ids["device_mask"]
                if bool(dyn_mask.any().item()):
                    gather_ids = torch.where(
                        dyn_mask,
                        token_ids["device_ids"].to(torch.long),
                        torch.zeros_like(token_ids["device_ids"], dtype=torch.long),
                    )
                    dyn_k = torch.index_select(token_ids["device_attn_keys"], 0, gather_ids)
                    dyn_v = torch.index_select(token_ids["device_attn_values"], 0, gather_ids)
                    dyn_count = int(dyn_mask.sum().item())
                else:
                    dyn_k = None
                    dyn_v = None
                    dyn_count = 0
            elif token_ids:
                idx = torch.tensor(token_ids, dtype=torch.long, device="cpu")
                dyn_k = torch.index_select(self.cpu_keys[layer_idx][kv_hdx], 0, idx).to(device, non_blocking=True)
                dyn_v = torch.index_select(self.cpu_values[layer_idx][kv_hdx], 0, idx).to(device, non_blocking=True)
                dyn_count = len(token_ids)
            else:
                dyn_k = None
                dyn_v = None
                dyn_count = 0
            if self.decode_profile:
                if isinstance(token_ids, dict) and "device_ids" in token_ids and self.fullgpu_profile_sync:
                    torch.cuda.synchronize(device)
                self._decode_profile_stats["gather_total_sec"] += (time.perf_counter() - gather_start)
            if dyn_count == 0:
                empty_heads += 1
            dynamic_counts.append(dyn_count)

            static_k = self.static_gpu_keys[layer_idx][kv_hdx]
            static_v = self.static_gpu_values[layer_idx][kv_hdx]

            if dyn_k is not None:
                if dyn_mask is not None:
                    attn_start = time.perf_counter() if self.decode_profile else None
                    if self.decode_profile and self.fullgpu_profile_sync:
                        torch.cuda.synchronize(device)
                        attn_start = time.perf_counter()
                    static_scores = torch.matmul(q_attn, static_k.transpose(0, 1)) * scale
                    dyn_scores = torch.matmul(q_attn, dyn_k.transpose(0, 1)) * scale
                    dyn_scores = dyn_scores.float().masked_fill(~dyn_mask.unsqueeze(0), float("-inf"))
                    scores = torch.cat([static_scores.float(), dyn_scores], dim=-1)
                    v = torch.cat([static_v, dyn_v], dim=0)
                    attn = torch.softmax(scores, dim=-1).to(v.dtype)
                    out = torch.matmul(attn, v)
                    if self.decode_profile:
                        if self.fullgpu_profile_sync:
                            torch.cuda.synchronize(device)
                        self._decode_profile_stats["attn_total_sec"] += (time.perf_counter() - attn_start)
                    outputs.append(out)
                    if self.decode_profile:
                        self._decode_profile_stats["heads"] += 1
                    continue
                k = torch.cat([static_k, dyn_k], dim=0)
                v = torch.cat([static_v, dyn_v], dim=0)
            else:
                k = static_k
                v = static_v

            attn_start = time.perf_counter() if self.decode_profile else None
            scores = torch.matmul(q_attn, k.transpose(0, 1)) * scale
            scores = scores.float()
            attn = torch.softmax(scores, dim=-1).to(v.dtype)
            out = torch.matmul(attn, v)
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

        output_start = time.perf_counter() if self.decode_profile else None
        if self.decode_profile:
            self._decode_profile_stats["calls"] += 1
            self._decode_profile_stats["compute_total_sec"] += (time.perf_counter() - compute_start)

        out = torch.cat(outputs, dim=0).view(1, 1, self.num_heads, self.head_dim)
        if self.decode_profile:
            self._decode_profile_stats["other_output_sec"] += (time.perf_counter() - output_start)
        return out
