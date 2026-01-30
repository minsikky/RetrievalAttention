# AGENTS.md

## Project context
RetroInfer (Microsoft) repo for long-context attention with CPU–GPU co-execution. Current baseline uses wave index (centroids + clusters) and wave buffer (CPU-managed KV placement). Primary goal: experiment with sparse attention strategies and evaluate on LongBench and RULER, targeting Llama-3.1-8B. Focus is on ANN-based retrieval following RetrievalAttention (arXiv:2409.10516).

## Current decisions
- ANN backend: **faiss-cpu**.
- Graph build: **full prefill queries** (no subsampling).
- Strategy target: follow RetrievalAttention §3.2 (query-guided index; project Q→K links into K–K graph to avoid storing queries).

## Environment constraints
- Current machine has **no GPU**; code changes can be made here, but evaluation must be run on a GPU machine (A100 available).
- GPU memory is not the main bottleneck; CPU RAM and ANN index size are.

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

