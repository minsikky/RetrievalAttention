import importlib
import os
import sys
from pathlib import Path

import numpy as np


_ROAR_EXT_MODULE = "roargraph_builder_ext"
_ROAR_EXT_HANDLE = None
_ROAR_EXT_IMPORT_ERROR = None

_ROAR_TORCH_EXT_MODULE = "roargraph_torch_ext"
_ROAR_TORCH_EXT_HANDLE = None
_ROAR_TORCH_EXT_IMPORT_ERROR = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_ext_dir() -> Path:
    return _project_root() / "third_party" / "RoarGraph" / "python_ext"


def _resolve_ext_dir() -> Path:
    override = os.environ.get("RETRIEVALATTN_ROAR_CPP_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _default_ext_dir()


def _try_import() -> bool:
    global _ROAR_EXT_HANDLE
    global _ROAR_EXT_IMPORT_ERROR

    if _ROAR_EXT_HANDLE is not None:
        return True
    if _ROAR_EXT_IMPORT_ERROR is not None:
        return False

    ext_dir = _resolve_ext_dir()
    if ext_dir.exists():
        ext_dir_str = str(ext_dir)
        if ext_dir_str not in sys.path:
            sys.path.insert(0, ext_dir_str)
    try:
        _ROAR_EXT_HANDLE = importlib.import_module(_ROAR_EXT_MODULE)
        return True
    except Exception as exc:  # pragma: no cover - import failure path
        _ROAR_EXT_IMPORT_ERROR = exc
        return False


def _try_import_torch() -> bool:
    global _ROAR_TORCH_EXT_HANDLE
    global _ROAR_TORCH_EXT_IMPORT_ERROR

    if _ROAR_TORCH_EXT_HANDLE is not None:
        return True
    if _ROAR_TORCH_EXT_IMPORT_ERROR is not None:
        return False

    ext_dir = _resolve_ext_dir()
    if ext_dir.exists():
        ext_dir_str = str(ext_dir)
        if ext_dir_str not in sys.path:
            sys.path.insert(0, ext_dir_str)
    try:
        _ROAR_TORCH_EXT_HANDLE = importlib.import_module(_ROAR_TORCH_EXT_MODULE)
        return True
    except Exception as exc:  # pragma: no cover - import failure path
        _ROAR_TORCH_EXT_IMPORT_ERROR = exc
        return False


def roargraph_cpp_available() -> bool:
    return _try_import()


def roargraph_cpp_import_error():
    _try_import()
    return _ROAR_EXT_IMPORT_ERROR


def roargraph_cuda_available() -> bool:
    return _try_import_torch()


def roargraph_cuda_import_error():
    _try_import_torch()
    return _ROAR_TORCH_EXT_IMPORT_ERROR


def build_roar_graph_csr_cpp(
    knn: np.ndarray,
    keys: np.ndarray,
    *,
    dynamic_start: int,
    dynamic_end: int,
    nq: int,
    degree_cap: int,
    cand_limit: int,
    enable_enhance: bool,
    enhance_limit: int,
    entry_mode: str,
    max_query_per_pivot: int,
    num_threads: int,
):
    if not _try_import():
        raise RuntimeError(
            "RoarGraph C++ extension is unavailable. "
            "Build it with: "
            "`module load python/3.10.4 && source .venv/bin/activate && "
            "python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`"
        ) from _ROAR_EXT_IMPORT_ERROR

    knn_arr = np.ascontiguousarray(knn, dtype=np.int32)
    keys_arr = np.ascontiguousarray(keys, dtype=np.float32)

    return _ROAR_EXT_HANDLE.build_graph_csr(
        knn_arr,
        keys_arr,
        int(dynamic_start),
        int(dynamic_end),
        int(nq),
        int(degree_cap),
        int(cand_limit),
        bool(enable_enhance),
        int(enhance_limit),
        str(entry_mode),
        int(max_query_per_pivot),
        int(num_threads),
    )


def search_roar_graph_csr_cpp(
    query: np.ndarray,
    keys: np.ndarray,
    offsets: np.ndarray,
    neighbors: np.ndarray,
    init_ids: np.ndarray,
    init_scores: np.ndarray,
    *,
    topk: int,
    lpq: int,
    max_cmps: int,
    max_hops: int,
    dynamic_start: int,
    dynamic_end: int,
    num_threads: int = 0,
    score_agg: str = "max",
    key_dtype: str = "auto",
):
    if not _try_import():
        raise RuntimeError(
            "RoarGraph C++ extension is unavailable. "
            "Build it with: "
            "`module load python/3.10.4 && source .venv/bin/activate && "
            "python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`"
        ) from _ROAR_EXT_IMPORT_ERROR

    query_arr = np.ascontiguousarray(query, dtype=np.float32)
    if query_arr.ndim == 1:
        query_arr = query_arr[None, :]
    if query_arr.ndim != 2:
        raise ValueError(f"query must be 1D/2D float32, got shape={query_arr.shape}")

    keys_arr = np.ascontiguousarray(keys)
    if key_dtype == "auto":
        if keys_arr.dtype == np.float32:
            key_dtype = "fp32"
        elif keys_arr.dtype == np.float16:
            key_dtype = "fp16"
        elif keys_arr.dtype == np.uint16:
            key_dtype = "bf16"
        else:
            raise ValueError(f"Unsupported keys dtype for Roar decode backend: {keys_arr.dtype}")
    key_dtype = str(key_dtype).strip().lower()

    offsets_arr = np.ascontiguousarray(offsets, dtype=np.uint32)
    neighbors_arr = np.ascontiguousarray(neighbors, dtype=np.int32)
    init_ids_arr = np.ascontiguousarray(init_ids, dtype=np.int32)
    init_scores_arr = np.ascontiguousarray(init_scores, dtype=np.float32)

    return _ROAR_EXT_HANDLE.search_graph_csr(
        query_arr,
        keys_arr,
        offsets_arr,
        neighbors_arr,
        init_ids_arr,
        init_scores_arr,
        int(topk),
        int(lpq),
        int(max_cmps),
        int(max_hops),
        int(dynamic_start),
        int(dynamic_end),
        int(num_threads),
        str(score_agg),
        key_dtype,
    )


def search_roar_graph_csr_cuda(
    query_seed,
    query_rank,
    keys,
    offsets,
    neighbors,
    init_ids,
    init_scores,
    *,
    token_budget: int,
    candidate_target: int,
    expand_width: int,
    min_visits: int,
    max_visits: int,
    frontier_topn: int,
    stop_patience: int,
    stop_margin: float,
    dynamic_start: int,
    dynamic_end: int,
    score_agg: str = "max",
):
    if not _try_import_torch():
        raise RuntimeError(
            "RoarGraph torch/CUDA extension is unavailable. "
            "Build it with: "
            "`module load python/3.10.4 && source .venv/bin/activate && "
            "python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`"
        ) from _ROAR_TORCH_EXT_IMPORT_ERROR

    return _ROAR_TORCH_EXT_HANDLE.search_graph_csr_cuda(
        query_seed,
        query_rank,
        keys,
        offsets,
        neighbors,
        init_ids,
        init_scores,
        int(token_budget),
        int(candidate_target),
        int(expand_width),
        int(min_visits),
        int(max_visits),
        int(frontier_topn),
        int(stop_patience),
        float(stop_margin),
        int(dynamic_start),
        int(dynamic_end),
        str(score_agg),
    )


def search_roar_graph_csr_cuda_group(
    queries_seed,
    queries_rank,
    keys,
    offsets,
    neighbors,
    init_ids,
    init_scores,
    *,
    token_budget: int,
    candidate_target: int,
    expand_width: int,
    min_visits: int,
    max_visits: int,
    frontier_topn: int,
    stop_patience: int,
    stop_margin: float,
    dynamic_start: int,
    dynamic_end: int,
    score_agg: str = "max",
):
    if not _try_import_torch():
        raise RuntimeError(
            "RoarGraph torch/CUDA extension is unavailable. "
            "Build it with: "
            "`module load python/3.10.4 && source .venv/bin/activate && "
            "python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`"
        ) from _ROAR_TORCH_EXT_IMPORT_ERROR

    return _ROAR_TORCH_EXT_HANDLE.search_graph_csr_cuda_group(
        queries_seed,
        queries_rank,
        keys,
        offsets,
        neighbors,
        init_ids,
        init_scores,
        int(token_budget),
        int(candidate_target),
        int(expand_width),
        int(min_visits),
        int(max_visits),
        int(frontier_topn),
        int(stop_patience),
        float(stop_margin),
        int(dynamic_start),
        int(dynamic_end),
        str(score_agg),
    )
