# AGENTS.md

## Project context
RetroInfer (Microsoft) repo for long-context attention with CPU–GPU co-execution. Current baseline uses wave index (centroids + clusters) and wave buffer (CPU-managed KV placement). Primary goal: experiment with sparse attention strategies and evaluate on LongBench and RULER, targeting Llama-3.1-8B. Focus is on ANN-based retrieval following RetrievalAttention (arXiv:2409.10516).

## Current decisions
- ANN backend: **faiss-cpu**.
- Graph build: **full prefill queries** (no subsampling).
- Strategy target: follow RetrievalAttention §3.2 (query-guided index; project Q→K links into K–K graph to avoid storing queries).

## Environment constraints
- We are on a **GPU server**. Jobs should be submitted via Slurm using `sbatch`.
- GPU memory is not the main bottleneck; CPU RAM and ANN index size are.

## GPU job workflow (current)
- Use `test.sh` as the entry point for Slurm jobs.
- Submit with `sbatch test.sh` and **do not wait/tail logs** after submission.
- For initial sanity checks, run `simple_test.py` with `RetroInfer` and record runtime + GPU memory.
- Optionally run `Full_Flash_Attn` as a baseline comparison.

## Key code locations
- Retrieval pipeline (decode): `cache_hub/retroinfer_cache.py` → `compute()`
  - Current behavior: query–centroid similarity → `topk` clusters → wave buffer → `weighted_flash_decoding`.
- Prefill index build: `cache_hub/retroinfer_cache.py` → `prefill_update_kv_cache()`
  - Current behavior: segmented k-means + cluster lists (wave index).
- Attention routing: `attn_hub/retroinfer_attn.py`, `model_hub/llama.py` (attention type selection).
- Benchmarks: `benchmark/LongBench`, `benchmark/ruler`.

## Planned work (not yet implemented)
1) Add a new retrieval strategy config (e.g., `retrieval_strategy: attention_aware_graph`) in `config/*.json` and CLI.
2) During prefill:
   - Collect prefill queries per layer/head.
   - Build Q→K KNN (faiss-cpu).
   - Project Q→K links into a bounded-degree **K–K graph** (RoarGraph-style) as per RetrievalAttention.
3) During decode:
   - Use ANN search over the K–K graph (keys only) to retrieve token indices.
   - Route retrieved tokens into attention compute (either via new token-level gather path or via a compatible interface with existing wave buffer).
4) Add knobs for memory/quality:
   - `q_knn`, `key_degree`, possibly `index_dtype`, `graph_build_batch`.
5) Benchmark on Llama-3.1-8B with LongBench and RULER scripts.

## Open questions / design choices to resolve
- Token-level retrieval integration: whether to bypass wave buffer with a new token gather path, or map token hits back into cluster IDs.
- Graph storage format and degree cap to fit CPU RAM.
- Whether to index per layer/head or share across heads (accuracy vs memory).

## Notes from prior discussion
- RetrievalAttention requires prefill to build index.
- Query storage can be avoided by projecting to key–key graph.
- A100 should be fine for GPU memory; CPU RAM is the main risk.

## Current progress (compact)
- GPU workflow stabilized: `test.sh` and `benchmark/ruler/ruler_run_wrapper.sh` use venv, purge module env, and log torch version/path to avoid system PyTorch conflicts.
- Added `LOW_CPU_MEM_USAGE=1` support for streaming model load (llama/qwen).
- Added runtime/memory instrumentation:
  - `benchmark/ruler/data/prepare.py` logs time + RSS/GPU.
  - `benchmark/ruler/pred/call_api.py` logs time + RSS/GPU and model init timing.
  - `model_hub/LLM.py` logs memory before/after KV cache and prefill/decode.
  - `cache_hub/retroinfer_cache.py` prints pinned KV estimate.
  - `benchmark/ruler/ruler_run.sh` logs phase durations (prep/pred/eval).
- Added `REUSE_DATA` (default 1) and `FORCE_PREPARE` to skip/regenerate data. Added `FORCE_PRED` (default 1) to force re-inference by deleting existing pred file.
- Profiling: `ENABLE_PROFILER` default 0; guard to avoid unsafe profiler. `PROFILER_SAFE` gate exists, but multi-thread profiling is not safe. Use logging instead.
- New RetrievalAttention prototype implemented:
  - `cache_hub/retrievalattention_cache.py` (token-level ANN retrieval + static GPU prefix/suffix KV).
  - `attn_hub/retrievalattention_attn.py`.
  - Routing in `model_hub/llama.py` and `model_hub/qwen.py`.
  - Config blocks for `RetrievalAttention` added to all model JSONs.
  - CLI supports `RetrievalAttention` in simple_test, throughput, RULER, LongBench.
  - `benchmark/config.py` computes token budget from RetroInfer budget (token count fairness) and passes `q_knn`, `key_degree`.
  - `requirements.txt` includes `faiss-cpu`.
- RULER runs showed prior errors fixed:
  - OOM due to CPU RAM: need ≥64 GB for 131k context; 32 GB insufficient.
  - Environment leak to system torch fixed by module purge + unset PYTHONPATH/PYTHONHOME.
  - Flashinfer requires torch 2.5.1; ensure venv uses it.

## Known caveats / TODOs
- RetrievalAttention cache currently builds Q→K KNN on CPU (faiss-cpu) and uses a lightweight K–K projection (anchor-based), not full RoarGraph. Per-layer/per-head memory is large.
- RetrievalAttention only supports batch_size=1.
- Generated tokens are not inserted into ANN index (prefill-only index).
- Profiler with multithreaded inference is unsafe (CUPTI error).

## How to run (current)
- RULER wrapper: `sbatch benchmark/ruler/ruler_run_wrapper.sh`
  - Override: `ATTN_TYPE=RetrievalAttention` or `FORCE_PRED=0`, `REUSE_DATA=1`.
- Simple test: `python -u simple_test.py --attn_type RetrievalAttention --model_name meta-llama/Llama-3.1-8B-Instruct`
