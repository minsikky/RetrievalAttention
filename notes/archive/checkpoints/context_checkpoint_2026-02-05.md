# Context Checkpoint (2026-02-05)

## Branch / focus
- Branch: `gpu_top_k`
- Main file: `cache_hub/retrievalattention_cache.py`
- Task: make RetrievalAttention both fast and quality-competitive with RetroInfer.

## What is now true
1. GPU-topk build path is working and much faster.
2. Layer-wise GPU caching and overlap reduced transfer cost significantly.
3. Decode crash from bf16 conversion is fixed.
4. Decode retrieval is active (non-empty) in latest debug logs.
5. Quality is still weaker than RetroInfer on `simple_test.py`.
6. Decode traversal is now adaptive best-first (fixed hop is removed).
7. `model_hub/LLM.py` now logs decode total latency in seconds.

## Key evidence
- `slurm-41844983.out`:
  - `recall@32=1.0000` parity on sampled KNN.
  - decode debug shows seeds and non-empty dynamic retrieval.
  - output still under target quality (top-3 coded words partly wrong).
- `slurm-41844519.out`:
  - decode error due to bf16 numpy conversion; fixed.
- `slurm-41847189.out`:
  - quality improved (top-3 line matches RetroInfer in sample),
  - but decode latency is too high (`8663.44 ms/step`),
  - due to permissive auto adaptive limits (`min_visits=2070`, `max_visits=16560`).

## Current hypothesis
- Hard correctness issues are mostly addressed.
- Current bottleneck has shifted to adaptive traversal calibration:
  - quality is better with adaptive best-first search,
  - decode latency is unacceptably high under auto visit limits for long contexts.
- Need explicit visit caps to move to a usable quality/latency operating point.

## Immediate next actions
1. Re-run with explicit adaptive caps:
   - `RETRIEVALATTN_MIN_VISITS=256`, `RETRIEVALATTN_MAX_VISITS=2048`,
   - `RETRIEVALATTN_EXPAND_WIDTH=48`, `RETRIEVALATTN_STOP_PATIENCE=1`, `RETRIEVALATTN_STOP_MARGIN=0.001`.
2. Compare quality/latency against:
   - current adaptive-auto run,
   - RetroInfer baseline.
3. Keep token budget constant for fairness.
4. Only if quality regresses too much, increase `max_visits` gradually.

## Run command template
```bash
RETRIEVALATTN_GPU_TOPK=1 \
RETRIEVALATTN_LAYER_GPU_CACHE=1 \
RETRIEVALATTN_DECODE_INDEX=faiss \
RETRIEVALATTN_DEBUG=1 \
RETRIEVALATTN_ASSERT_NONEMPTY=1 \
RETRIEVALATTN_VALIDATE_PARITY=1 \
sbatch test.sh
```
