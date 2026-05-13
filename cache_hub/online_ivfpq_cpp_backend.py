import importlib
import os
import sys
from pathlib import Path

import numpy as np


_MODULE = "online_ivfpq_ext"
_HANDLE = None
_IMPORT_ERROR = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_ext_dir() -> Path:
    return _project_root() / "third_party" / "OnlineIVFPQ" / "python_ext"


def _resolve_ext_dir() -> Path:
    override = os.environ.get("RETRIEVALATTN_ONLINE_IVFPQ_CPP_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _default_ext_dir()


def _try_import() -> bool:
    global _HANDLE
    global _IMPORT_ERROR
    if _HANDLE is not None:
        return True
    if _IMPORT_ERROR is not None:
        return False
    ext_dir = _resolve_ext_dir()
    if ext_dir.exists():
        ext_dir_str = str(ext_dir)
        if ext_dir_str not in sys.path:
            sys.path.insert(0, ext_dir_str)
    try:
        _HANDLE = importlib.import_module(_MODULE)
        return True
    except Exception as exc:  # pragma: no cover
        _IMPORT_ERROR = exc
        return False


def online_ivfpq_cpp_available() -> bool:
    return _try_import()


def online_ivfpq_cpp_import_error():
    _try_import()
    return _IMPORT_ERROR


def assign_encode_batch_cpp(
    keys: np.ndarray,
    centroids: np.ndarray,
    codebooks: np.ndarray,
    centroid_sums: np.ndarray,
    counts: np.ndarray,
    *,
    update_centroids: bool,
    num_threads: int = 0,
):
    if not _try_import():
        raise RuntimeError(
            "Online IVF-PQ C++ extension is unavailable. Build it with: "
            "`module load python/3.10.4 && source .venv/bin/activate && "
            "python third_party/OnlineIVFPQ/python_ext/setup.py build_ext --inplace`"
        ) from _IMPORT_ERROR
    return _HANDLE.assign_encode_batch(
        np.ascontiguousarray(keys, dtype=np.float32),
        np.ascontiguousarray(centroids, dtype=np.float32),
        np.ascontiguousarray(codebooks, dtype=np.float32),
        np.ascontiguousarray(centroid_sums, dtype=np.float32),
        np.ascontiguousarray(counts, dtype=np.int64),
        bool(update_centroids),
        int(num_threads),
    )


def rank_nprobes_cpp(
    query: np.ndarray,
    centroids: np.ndarray,
    codebooks: np.ndarray,
    codes: np.ndarray,
    assign: np.ndarray,
    nprobes: np.ndarray,
    *,
    size: int,
    token_start: int,
    num_threads: int = 0,
):
    if not _try_import():
        raise RuntimeError(
            "Online IVF-PQ C++ extension is unavailable. Build it with: "
            "`module load python/3.10.4 && source .venv/bin/activate && "
            "python third_party/OnlineIVFPQ/python_ext/setup.py build_ext --inplace`"
        ) from _IMPORT_ERROR
    return _HANDLE.rank_nprobes(
        np.ascontiguousarray(query, dtype=np.float32),
        np.ascontiguousarray(centroids, dtype=np.float32),
        np.ascontiguousarray(codebooks, dtype=np.float32),
        np.ascontiguousarray(codes, dtype=np.uint16),
        np.ascontiguousarray(assign, dtype=np.int32),
        np.ascontiguousarray(nprobes, dtype=np.int32),
        int(size),
        int(token_start),
        int(num_threads),
    )
