# Context Checkpoint — 2026-02-06

## Snapshot
- Branch/workstream: `gpu_top_k`, RetrievalAttention correctness + prefill index-build optimization.
- Decision in this checkpoint:
  - freeze current Triton custom fused qk+topk approach for active iteration,
  - pivot to FlashAttention-prefill fusion approach.

## What was attempted in this cycle
- Added and iterated a Triton custom fused qk+topk kernel path (guarded by `RETRIEVALATTN_CUSTOM_QK_TOPK`).
- Added profiling/progress logs for custom launch/chunk behavior.
- Patched multiple Triton-compatibility issues encountered during compile/runtime.

## Outcome
- Path remains blocked for long-context target workload.
- Characteristic blocked behavior:
  - run reaches `gpu_topk(custom_fused) chunk 1/...` and then does not progress.
- Additional attempts surfaced Triton API/shape constraints (fixed), but did not resolve practical stuck behavior in target run shape.

## Evidence logs
- `slurm-41850330.out`: custom launch entered, stuck at chunk 1.
- `slurm-41850375.out`: compile failure around `tl.cat` reorder assertion.
- `slurm-41850455.out`: compile failure (`tl.cat` rank assertion).
- `slurm-41850475.out`: no traceback; still stuck at chunk 1.

## Current operating guidance
- For productive runs, keep custom path disabled:
  - `RETRIEVALATTN_CUSTOM_QK_TOPK=0`
  - keep `RETRIEVALATTN_GPU_TOPK=1`, `RETRIEVALATTN_DECODE_INDEX=faiss`.

## Next direction (active)
- Move index-building fusion into prefill attention path (FlashAttention-kernel-level approach):
  - piggyback retrieval index signal collection during prefill QK compute,
  - reduce dependence on separate post-prefill topk build pass.

## Guardrails to preserve while pivoting
- Maintain decode seed index quality guardrail:
  - `RETRIEVALATTN_DECODE_INDEX=faiss`.
- Preserve existing parity/debug checks until fused-prefill path reaches stability.

## Immediate next tasks
1. Define fused-prefill data contract (what retrieval artifacts to emit per layer/head).
2. Add minimal instrumentation in prefill attention path for candidate-collection timing.
3. Validate parity/quality against current baseline before replacing baseline build path.
