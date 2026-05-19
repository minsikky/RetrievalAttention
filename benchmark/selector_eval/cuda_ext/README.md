# Selector Paged-PQ CUDA Extension

Native CUDA backend for paged-PQ selector primitives used by the selector-eval and HF/RULER intervention paths.

Build from the repository root:

```bash
module load python/3.10.4 cuda/12.8.1
source .venv/bin/activate
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
cd benchmark/selector_eval/cuda_ext
python setup.py build_ext --inplace
```

GPU smoke tests:

```bash
PYTHONPATH=benchmark/selector_eval/cuda_ext .venv/bin/python benchmark/selector_eval/cuda_ext/test_fullscan_pq_topk.py
.venv/bin/python benchmark/selector_eval/cuda_ext/test_gpu_vpq_helpers.py
```

Current scope:

- `fullscan_pq_topk`: batched fullscan PQ scoring/top-k for `[heads, dim]` queries.
- GPU V-PQ helper parity test validates selected-value reconstruction against the CPU/reference implementation.

The extension intentionally covers selector primitives only. HF/RULER integration still owns page-cache policy, selected-token attention, V compression policy, tail policy, and cost accounting.
