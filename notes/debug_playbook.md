# Debug Playbook (RetrievalAttention)

## Goal
Verify correctness first, then tune performance. Avoid optimizing broken behavior.

## Step 1: Guardrail run (must pass)
Use these envs in `test.sh` submission:

```bash
RETRIEVALATTN_GPU_TOPK=1
RETRIEVALATTN_LAYER_GPU_CACHE=1
RETRIEVALATTN_DECODE_INDEX=faiss
RETRIEVALATTN_SEED_MODE=graph_only
RETRIEVALATTN_SCORE_MODE=ip
RETRIEVALATTN_DEBUG=1
RETRIEVALATTN_ASSERT_NONEMPTY=1
RETRIEVALATTN_VALIDATE_PARITY=1
```

Expected signals:
- Build banner contains `mode=gpu_topk` and `decode_index=faiss`.
- Parity log appears (`recall@k=...`).
- Decode debug logs show non-zero dynamic retrieval in most heads.
- No runtime error from `ASSERT_NONEMPTY`.

## Step 2: If quality still degrades
1. Check decode retrieval stats:
   - Are seeds non-zero?
   - Are dynamic retrieved counts mostly non-zero?
2. Compare GPU-topk parity recall:
   - Very low recall means topk mismatch; investigate normalization/blocking.
3. Disable speed optimizations one-by-one:
   - `RETRIEVALATTN_LAYER_GPU_CACHE=0`
   - `RETRIEVALATTN_OVERLAP=0`
   - keep decode index as `faiss`.
4. If decode is too slow with adaptive traversal:
   - set explicit decode traversal caps (avoid auto):
   - `RETRIEVALATTN_MIN_VISITS=96`
   - `RETRIEVALATTN_MAX_VISITS=2048`
   - `RETRIEVALATTN_EXPAND_WIDTH=48`
   - `RETRIEVALATTN_STOP_PATIENCE=1`
   - `RETRIEVALATTN_STOP_MARGIN=0.001`

## Step 2.1: Seed-mode A/B check (required after latest refactor)
1. Run `RETRIEVALATTN_SEED_MODE=graph_only` and record decode profile.
2. Run `RETRIEVALATTN_SEED_MODE=faiss` and record decode profile.
3. Compare:
   - `seed` time should be lower in graph-only mode.
   - if `graph` remains dominant, prioritize traversal optimization next.

## Step 2.5: Frozen custom-kernel path rule
- Current state:
  - Triton custom fused path is experimental and currently frozen for active iteration.
- If testing it anyway, treat as blocked when this pattern appears:
  - log reaches `gpu_topk(custom_fused) chunk 1/...` and no subsequent chunk/profile line appears for several minutes.
- Action when blocked:
1. Stop run and archive the slurm log.
2. Capture last progress line and elapsed wall time.
3. Resume baseline runs with:
   - `RETRIEVALATTN_CUSTOM_QK_TOPK=0`
   - keep `RETRIEVALATTN_DECODE_INDEX=faiss`.
4. Do not continue micro-tuning custom kernel tiles in the same session; continue with FlashAttention-fusion workstream instead.

### Blocked-path evidence checklist
1. Last printed custom line (e.g., `chunk 1/30`).
2. Whether traceback exists or run is silent-stalled.
3. Approximate elapsed time since last custom line.
4. GPU/CPU usage snapshot if available.

## Step 3: Isolate graph vs seed issues
- If seeds are valid but outputs poor, inspect graph projection quality.
- If seeds are empty or near-empty, index/search path is broken.

### Concrete ablations (run in this order)
1. Seed-only mode:
   - use only decode seeds (top-k) without graph expansion.
   - if quality improves, graph projection is the dominant issue.
2. Current graph mode:
   - compare against seed-only at same token budget.
3. Query representation check:
   - compare grouped-query average vs per-head query retrieval.
   - if per-head improves, query averaging is too lossy.

## Step 4: Fair-comparison checks
- Keep token budget constant while testing alternatives.
- Do not change `q_knn`, `key_degree`, and static window simultaneously.

## Step 5: Regression triage order
1. Seed mode correctness (`graph_only` warm-start/hub/tail behavior).
2. Dynamic retrieval non-emptiness.
3. Graph projection distribution.
4. Traversal cost controls (`frontier_topn`, `expand_width`, `max_visits`, stop rules).
5. Runtime optimizations (overlap/caching).
6. Query representation fidelity (group-average vs per-head).
