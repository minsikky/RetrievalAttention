# Context Checkpoint — 2026-02-12

## Branch / focus
- Branch: `gpu_top_k`
- Focus file: `cache_hub/retrievalattention_cache.py`
- Objective: improve RetrievalAttention quality/latency balance with paper-aligned ANN behavior.

## What just landed
1. Graph-builder backend selection:
   - `RETRIEVALATTN_GRAPH_BUILDER=legacy|roar`.
2. Roar-style graph build path:
   - query->base bridge extraction from prefill KNN,
   - neighborhood-aware projection (`AcquireNeighbors`-style),
   - reverse-edge updates during projection,
   - connectivity enhancement with beam-style candidate collection,
   - CSR export compatible with current decode.
3. New Roar controls:
   - `RETRIEVALATTN_ROAR_NQ`, `RETRIEVALATTN_ROAR_L`, `RETRIEVALATTN_ROAR_M`,
   - `RETRIEVALATTN_ROAR_ENABLE_ENHANCE`, `RETRIEVALATTN_ROAR_ENHANCE_L`,
   - `RETRIEVALATTN_ROAR_ENTRY`, `RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT`,
   - `RETRIEVALATTN_ROAR_LOG`.
4. Test defaults updated:
   - static window now `128 + 512`,
   - dynamic retrieval budget defaults to `TOKEN_BUDGET_OVERRIDE=100`,
   - adaptive decode floor lowered to `RETRIEVALATTN_MIN_VISITS=96`,
   - `RETRIEVALATTN_GRAPH_BUILDER=roar` in `test.sh`.

## Current behavior boundary
- Prefill graph build is now closer to Roar paper intent.
- Decode traversal is still the existing adaptive best-first frontier expansion.
- Decode seed path remains `graph_only` by default with hub/tail warm starts.

## Next concrete step
- Refactor decode traversal to beam-search style over the built graph:
  - maintain bounded beam/frontier candidates,
  - enforce retrieval budget fairness (`token_budget=100`),
  - preserve fallback path for A/B (`legacy` builder + old traversal).

## Suggested first A/B after resume
1. `RETRIEVALATTN_GRAPH_BUILDER=roar sbatch test.sh`
2. `RETRIEVALATTN_GRAPH_BUILDER=legacy sbatch test.sh`
3. Compare:
   - decode profile (`seed`, `graph`, `rerank`),
   - top-3 coded-word quality,
   - `visited_total` / `candidates_total`.
