# AGENTS.md

## Project context
RetroInfer (Microsoft) long-context attention repo with CPU-GPU co-execution.
Current active experiment is RetrievalAttention-style ANN retrieval for `Llama-3.1-8B`.

## Environment constraints
- GPU server with Slurm (`sbatch`).
- CPU RAM and ANN index size are primary bottlenecks, not GPU VRAM.
- Standard job entry for fast iteration: `test.sh`.

## Quick start
- Submit simple test: `sbatch test.sh`
- Submit RULER wrapper: `sbatch benchmark/ruler/ruler_run_wrapper.sh`
- Do not tail/wait after submit; inspect slurm output after completion.

## Current active branch/work
- Branch used for GPU-topk exploration: `gpu_top_k`
- Focus area: `cache_hub/retrievalattention_cache.py`

## Current decisions
- ANN backend: `faiss-cpu` (baseline + decode seed index in GPU-topk mode).
- Graph build: full prefill queries (no subsampling yet).
- Retrieval budget fairness: token budget derived from RetroInfer settings.
- RetrievalAttention static pattern is now paper-style:
  - `static_pattern_start=128`
  - `static_pattern_end=512`
- Runtime token budget is now override-first for fast iteration:
  - `TOKEN_BUDGET_OVERRIDE=100` is default in `test.sh` and `benchmark/ruler/ruler_run_wrapper.sh`.
  - Ratio-based budget path is still available as fallback.
- Roar-style graph builder is implemented and selectable:
  - `RETRIEVALATTN_GRAPH_BUILDER=roar|legacy`
  - default in `test.sh`: `roar`.

## Important caution (quality)
- Decode seeding now defaults to `RETRIEVALATTN_SEED_MODE=graph_only`:
  - warm-start from previous step retrieved tokens,
  - add per-head high-degree hub seeds from the built K-K graph,
  - add dynamic-tail anchors for cold-start robustness.
- `RETRIEVALATTN_SEED_MODE=faiss` remains available for debug/reference and uses full-scan `IndexFlatIP` seed search.
- If decode index is missing while running faiss seed mode, output quality can collapse (e.g., repeated `[INST]` patterns).

## Important caution (latency)
- Decode traversal now uses adaptive best-first expansion (not fixed-hop).
- Default adaptive limits can still be expensive if fanout is high.
- Tune `RETRIEVALATTN_MIN_VISITS` / `RETRIEVALATTN_MAX_VISITS` / `RETRIEVALATTN_EXPAND_WIDTH` first when decode is too slow.
- `test.sh` now uses latency-safe defaults for adaptive decode:
  - `RETRIEVALATTN_EXPAND_WIDTH=48`
  - `RETRIEVALATTN_MIN_VISITS=96`
  - `RETRIEVALATTN_MAX_VISITS=2048`
  - `RETRIEVALATTN_STOP_PATIENCE=1`
  - `RETRIEVALATTN_STOP_MARGIN=0.001`
- Decode critical-path profiling can be enabled with `RETRIEVALATTN_DECODE_PROFILE=1` (enabled by default in `test.sh`).
- Latest decode seed change:
  - default seed path is now `RETRIEVALATTN_SEED_MODE=graph_only` (no full-scan `IndexFlatIP` at decode seed stage).
  - expected effect: reduce `seed` time in decode profile.
  - status: needs fresh run measurement to confirm reduction.
- Current priority after seed-mode change:
  - accelerate graph traversal (`graph` slice in decode profile), since retrieval traversal remains the expected bottleneck.
- Latest measured A/B (same adaptive traversal settings):
  - `slurm-42275514.out` (legacy faiss seed): decode `1268.31 s`; retrieve `[seed=308.814s, graph=896.496s]`.
  - `slurm-42277995.out` (graph_only seed): decode `945.12 s`; retrieve `[seed=10.546s, graph=880.674s]`.
  - interpretation: seed optimization worked (large drop), graph remained dominant.
- Candidate expansion efficiency signal:
  - `visited_total` stayed around `24.87M`,
  - `candidates_total` stayed around `78-80M`,
  - only ~31% of candidate evaluations become visited nodes, so decode bottleneck is now traversal fanout/candidate processing.
- Current priority:
  - update decode traversal from adaptive frontier expansion to Roar/HNSW-style beam-search traversal while preserving retrieval budget fairness.

## Key code locations
- Retrieval prototype cache: `cache_hub/retrievalattention_cache.py`
- Retrieval attention kernels: `attn_hub/retrievalattention_attn.py`
- Model routing: `model_hub/llama.py`, `model_hub/qwen.py`
- Baseline RetroInfer cache: `cache_hub/retroinfer_cache.py`
- Run harness: `test.sh`, `benchmark/ruler/ruler_run_wrapper.sh`
- Decode latency reporting: `model_hub/LLM.py`

## Notes index
- Current status snapshot: `notes/current_status.md`
- Debug and verification playbook: `notes/debug_playbook.md`
- Performance and quality findings log: `notes/findings_log.md`
- Runbook and env flags: `notes/runbook.md`
- Latest compact handoff checkpoint: `notes/context_checkpoint_2026-02-12.md`
- Previous compact checkpoints:
  - `notes/context_checkpoint_2026-02-06.md`
  - `notes/context_checkpoint_2026-02-05.md`

## Open design questions
- Token-level gather vs mapping token hits back to cluster/wave abstractions.
- Graph format and degree cap under CPU RAM constraints.
- Per-layer/per-head indexing vs sharing across heads.
- Decode beam-search traversal design:
  - beam width / ef-search style controls,
  - reverse-edge usage during traversal,
  - stop rule to hit target token budget with bounded latency.
