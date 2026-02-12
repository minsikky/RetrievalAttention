# Runbook

## Standard simple run
```bash
sbatch test.sh
```

## FlashAttention fork build (compute node only)
- Do not run flash-attn build from login node; login node has no CUDA toolkit.
- Submit the build as a Slurm job:
```bash
sbatch install_2.sh
```
- `install_2.sh` now:
  - loads `python/3.10.4` and CUDA module (`cuda/12.8.1` by default),
  - activates `.venv`,
  - resolves `CUDA_HOME` from `which nvcc`,
  - runs `pip install --no-build-isolation -v -e third_party/flash-attn-ra`.

## Interactive Python setup (shell)
```bash
module load python/3.10.4
source .venv/bin/activate
```

## Recommended RetrievalAttention debug run
```bash
RETRIEVALATTN_GPU_TOPK=1 \
RETRIEVALATTN_LAYER_GPU_CACHE=1 \
RETRIEVALATTN_DECODE_INDEX=faiss \
RETRIEVALATTN_SCORE_MODE=ip \
RETRIEVALATTN_DEBUG=1 \
RETRIEVALATTN_ASSERT_NONEMPTY=1 \
RETRIEVALATTN_VALIDATE_PARITY=1 \
sbatch test.sh
```

## Current recommendation on custom fused kernel path
- `RETRIEVALATTN_CUSTOM_QK_TOPK=1` path is currently **frozen** for active iteration.
- Reason:
  - repeated long-context stalls (`chunk 1/...`) and Triton compatibility issues in this cycle.
- Practical guidance:
  - keep custom path disabled for productive runs (`RETRIEVALATTN_CUSTOM_QK_TOPK=0`).
  - use baseline GPU-topk path while moving to FlashAttention-prefill fusion design.

## Recommended latency-safe adaptive run
```bash
RETRIEVALATTN_GPU_TOPK=1 \
RETRIEVALATTN_LAYER_GPU_CACHE=1 \
RETRIEVALATTN_DECODE_INDEX=faiss \
RETRIEVALATTN_SEED_MODE=graph_only \
RETRIEVALATTN_SCORE_MODE=ip \
RETRIEVALATTN_GRAPH_BUILDER=roar \
TOKEN_BUDGET_OVERRIDE=100 \
RETRIEVALATTN_GRAPH_EXPAND=1 \
RETRIEVALATTN_EXPAND_WIDTH=48 \
RETRIEVALATTN_MIN_VISITS=96 \
RETRIEVALATTN_MAX_VISITS=2048 \
RETRIEVALATTN_STOP_PATIENCE=1 \
RETRIEVALATTN_STOP_MARGIN=0.001 \
sbatch test.sh
```

## Graph-builder A/B (legacy vs roar)
```bash
# A: Roar builder (current default in test.sh)
ATTN_TYPE=RetrievalAttention \
RETRIEVALATTN_GRAPH_BUILDER=roar \
sbatch test.sh

# B: Legacy projected graph builder
ATTN_TYPE=RetrievalAttention \
RETRIEVALATTN_GRAPH_BUILDER=legacy \
sbatch test.sh
```

Compare:
```bash
grep -nE "index built layer|builder=|decode_profile|seed=|graph=|Answer:" slurm-*.out
```

## Required A/B after graph-only seed change
```bash
# A: graph-only seed (default)
ATTN_TYPE=RetrievalAttention \
RETRIEVALATTN_SEED_MODE=graph_only \
sbatch test.sh

# B: legacy full-scan seed
ATTN_TYPE=RetrievalAttention \
RETRIEVALATTN_SEED_MODE=faiss \
sbatch test.sh
```

Compare in slurm logs:
```bash
grep -nE "decode_profile|retrieve=|seed=|graph=" slurm-*.out
```
Target:
1. `seed` decreases in A vs B.
2. If `graph` still dominates, traversal becomes the next optimization target.

## Traversal-focused quick sweep (after seed fix)
Use these as controlled sweeps to reduce graph-time overhead while tracking quality:

```bash
# S1: narrower expansion (lower fanout pressure)
RETRIEVALATTN_SEED_MODE=graph_only \
RETRIEVALATTN_EXPAND_WIDTH=24 \
RETRIEVALATTN_MIN_VISITS=192 \
RETRIEVALATTN_MAX_VISITS=1024 \
RETRIEVALATTN_CAND_MULT=3 \
sbatch test.sh

# S2: stronger candidate pruning
RETRIEVALATTN_SEED_MODE=graph_only \
RETRIEVALATTN_FRONTIER_TOPN=256 \
RETRIEVALATTN_CAND_MULT=2 \
RETRIEVALATTN_MIN_VISITS=192 \
RETRIEVALATTN_MAX_VISITS=1024 \
sbatch test.sh

# S3: latency floor reference (aggressive)
RETRIEVALATTN_SEED_MODE=graph_only \
RETRIEVALATTN_EXPAND_WIDTH=16 \
RETRIEVALATTN_MIN_VISITS=128 \
RETRIEVALATTN_MAX_VISITS=768 \
RETRIEVALATTN_CAND_MULT=2 \
sbatch test.sh
```

For each run, compare:
```bash
grep -nE "decode_profile|retrieve=|seed=|graph=|visited_total|candidates_total|Answer:" slurm-*.out
```

Decision rule:
1. Keep settings only if `graph` time drops materially and top-3 quality is not worse than baseline.
2. If quality drops sharply, revert one notch (raise `expand_width` or `max_visits` first).

## Key env flags
- `RETRIEVALATTN_GPU_TOPK`: enable GPU topk build.
- `RETRIEVALATTN_CUSTOM_QK_TOPK`: opt-in Triton custom fused qk+topk kernel path (currently frozen/experimental; not recommended for active runs).
- `RETRIEVALATTN_FA_FUSED_PREFILL`: use FlashAttention fused-prefill retrieval path (expects flash-attn API `flash_attn_with_kvcache_retrieval` from a custom build/fork).
- `RETRIEVALATTN_FA_SHADOW_COMPARE`: in fused-prefill mode, run sampled parity check vs baseline GPU-topk (layer0/head0).
- `RETRIEVALATTN_FA_SHADOW_SAMPLE`: number of sampled queries used in fused shadow compare.
- `RETRIEVALATTN_FUSED_PREFILL_OVERLAP`: overlap CPU finalize (index/graph build) with ongoing fused prefill (`1` default).
- `RETRIEVALATTN_FUSED_PREFILL_OVERLAP_WORKERS`: overlap worker count (`1` recommended to avoid faiss/OpenMP oversubscription).
- `RETRIEVALATTN_CUSTOM_QK_TOPK_BLOCK_Q`: custom kernel Q tile size (default `64`; auto-capped to `<=64` to avoid pathological kernels).
- `RETRIEVALATTN_CUSTOM_QK_TOPK_LAUNCH_Q_CHUNK`: query chunk size per fused kernel launch (default `1024`; auto-capped to `<=1024`; `<=0` runs one monolithic launch).
- `RETRIEVALATTN_CUSTOM_QK_TOPK_BLOCK_D`: custom kernel D tile size (default `32`).
- `RETRIEVALATTN_CUSTOM_QK_TOPK_BLOCK_K`: custom kernel K tile size (default `256`; internally rounded to power-of-two and clamped to `<=256`).
- `RETRIEVALATTN_LAYER_GPU_CACHE`: cache per-layer K/Q on GPU during build.
- `RETRIEVALATTN_OVERLAP`: overlap transfer and compute for block streaming path.
- `RETRIEVALATTN_DECODE_INDEX`: decode seed source (`faiss` recommended).
- `RETRIEVALATTN_SEED_MODE`: decode seed strategy (`graph_only` default; `faiss` for full-scan reference/debug).
- `RETRIEVALATTN_QUERY_MODE`: `per_head` (default) or `group_avg`.
- `RETRIEVALATTN_SCORE_MODE`: retrieval similarity objective (`ip` default, or `cosine` for legacy normalized scoring).
- Caveat: `RETRIEVALATTN_SCORE_MODE=cosine` on native fused prefill requires a rebuilt flash-attn fork that exposes `fwd_kvcache_retrieval(..., retrieval_normalize)`. If the extension is old, runtime raises an explicit rebuild error.
- `RETRIEVALATTN_GRAPH_BUILDER`: graph construction backend (`roar` or `legacy`).
- `RETRIEVALATTN_ROAR_NQ`: Roar query->base neighbor width (uses KNN row prefix).
- `RETRIEVALATTN_ROAR_L`: projection candidate budget per pivot.
- `RETRIEVALATTN_ROAR_M`: degree cap for Roar projection/enhancement.
- `RETRIEVALATTN_ROAR_ENABLE_ENHANCE`: enable connectivity enhancement stage.
- `RETRIEVALATTN_ROAR_ENHANCE_L`: candidate budget used in enhancement beam collection.
- `RETRIEVALATTN_ROAR_ENTRY`: enhancement entry policy (`hub|max_degree|self`).
- `RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT`: optional cap of bridge queries per pivot (`0` disables cap).
- `RETRIEVALATTN_ROAR_LOG`: emit per-stage Roar build metrics in head logs.
- `RETRIEVALATTN_GRAPH_WEIGHTED`: weighted graph projection from prefill top-k (`1` default).
- `RETRIEVALATTN_GRAPH_CLIQUE_M`: clique-lite projection width among top candidates per query row (default `6`).
- `RETRIEVALATTN_GRAPH_RETURN_WEIGHTS`: store CSR edge weights as a third graph tensor (`0` default; decode traversal currently ignores weights).
- `RETRIEVALATTN_GRAPH_WEIGHT_DTYPE`: graph edge weight dtype (`uint16` default, `uint32` optional).
- `RETRIEVALATTN_GRAPH_EXPAND`: enable/disable graph neighbor expansion at decode.
- `RETRIEVALATTN_EXPAND_WIDTH`: number of frontier nodes expanded per adaptive step.
- `RETRIEVALATTN_MIN_VISITS`: minimum graph nodes to expand before adaptive early-stop checks (`<=0` => auto).
- `RETRIEVALATTN_MAX_VISITS`: hard cap on graph nodes expanded per decode/head (`<=0` => auto).
- `RETRIEVALATTN_STOP_PATIENCE`: required number of stable top-k checks before early stop.
- `RETRIEVALATTN_STOP_MARGIN`: stop only when frontier best score is below current kth score by this margin.
- `RETRIEVALATTN_FRONTIER_TOPN`: optional frontier pruning cap (`0` disables pruning).
- `RETRIEVALATTN_RERANK`: rerank candidate tokens using exact query-key dot scores.
- `RETRIEVALATTN_RERANK_AGG`: score aggregation across query heads (`max` default, or `mean`).
- `RETRIEVALATTN_SEED_RATIO`: minimum fraction of final dynamic budget reserved for seed tokens.
- `RETRIEVALATTN_CAND_MULT`: candidate pool size multiplier before final rerank.
- `RETRIEVALATTN_SEED_K_MULT`: multiplier for decode seed search width.
- `RETRIEVALATTN_SEED_PREV_K`: max previous-step retrieved tokens reused as warm-start seeds (graph-only mode).
- `RETRIEVALATTN_SEED_HUB_K`: number of per-head graph hub seeds added at decode (graph-only mode).
- `RETRIEVALATTN_SEED_TAIL_K`: number of dynamic-tail anchor seeds for cold start (graph-only mode).
- `RETRIEVALATTN_DEBUG`: print decode seed/dynamic retrieval diagnostics.
- `RETRIEVALATTN_ASSERT_NONEMPTY`: fail if all heads have empty dynamic retrieval.
- `RETRIEVALATTN_VALIDATE_PARITY`: run sampled parity check vs faiss.
- `RETRIEVALATTN_DECODE_PROFILE`: print end-of-decode critical-path breakdown (`retrieve`, `gather`, `attn`, `other`).
- `RETRIEVALATTN_Q_BLOCK`, `RETRIEVALATTN_K_BLOCK`: block sizes for GPU topk.
- `RETRIEVALATTN_HEAD_PIPELINE`: enable per-head GPU/CPU pipelining for index build.
- `RETRIEVALATTN_HEAD_PIPELINE_DEPTH`: max in-flight heads for pipelined finalize.
- `RETRIEVALATTN_HEAD_PIPELINE_MIN_CPUS`: auto-disable head pipeline if allocated CPUs are below this threshold.
- `FAISS_NUM_THREADS`: optional faiss thread cap (runtime also enforces scheduler-safe CPU budget).

## Metric interpretation
1. `Answer: [...]` in `simple_test.py` output is ground truth.
2. The next printed text line is model-generated output.
3. `[/INST]` and long `...` are generated continuation artifacts and should be treated as a secondary qualitative signal.
4. Primary simple-test metric remains top-3 coded-word correctness.

## Quality sanity checklist
1. Output is not dominated by repeated control tokens.
2. Decode debug logs show non-zero dynamic retrieval for most heads.
3. Parity recall is reasonable for sampled queries.

## If run fails
1. Read slurm log.
2. Verify build banner includes intended env values.
3. Check for empty retrieval assertion.
4. If quality issue persists, follow `notes/debug_playbook.md`.

## Next workstream
- Prioritize decode traversal refactor to beam-search style (paper-aligned):
  - replace/augment adaptive frontier expansion with beam candidate maintenance over built graph,
  - keep token budget fairness fixed (`TOKEN_BUDGET_OVERRIDE=100`) during A/B,
  - compare quality/latency against current adaptive traversal baseline.
