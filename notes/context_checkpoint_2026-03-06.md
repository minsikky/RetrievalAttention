# Context Checkpoint 2026-03-06

## Runtime families
- `c90fa94` / `8e9cdfc`: old GPU-topk + CPU graph build runtime exists here.
- `ad4d23e` / `bf4ab79`: fused-prefill runtime line.
- current `cpu_graph_builder_opt` worktree: dirty experimental state on top of fused runtime line.

## Important caveat
- `bf4ab79` does **not** define a separate GPU+CPU runtime branch.
- It only adds CPU graph-builder parity harness scripts.
- If a clean old GPU+CPU branch is needed, use `c90fa94`, not `bf4ab79`.

## Best known baselines
- 119k best fused-native run:
  - `slurm-44245076.out`
  - path=`native_kernel_fused_graph`
  - steady-state `native_core_sec ~= 19.9 s/layer`
  - total prefill `670.8404 s`
- 119k regressed fused-native run:
  - `slurm-44370482.out`
  - steady-state `native_core_sec ~= 35.3 s/layer`

## 32k comparison results
- Current tree, fused native GPU top-k + GPU graph:
  - `44431974`
  - `Prefilling latency: 97.0912 s`
- Current tree, native fused top-k + CPU `roar_cpp` graph:
  - `44431973`
  - `Prefilling latency: 143.6257 s`
- Current tree, forced Torch/Python GPU top-k + GPU graph:
  - `44431975`
  - `Prefilling latency: 115.461 s`
- Exported tree from `c90fa94`, old GPU-topk + CPU graph:
  - `44432065`
  - `Prefilling latency: 100.0403 s`
  - caveat: this run uses `retrieval_heads=8`, `retrieval_head_mode=kv_head`

## Experimental status
- `v3_warpk8`:
  - compiles and runs
  - 8k parity ok
  - no 32k speedup
  - `ncu`: registers worsened `190 -> 213`, occupancy unchanged
  - treat as failed experiment
- Forced Python/Torch GPU top-k path:
  - correctness fixes landed:
    - causal masking in blockwise top-k
    - Python GPU graph builder uses first dynamic token as pivot
    - parity uses `retrieval_causal` when available
  - 8k parity restored to `1.0`
  - still slower than native fused at 32k

## Recommendation
- Use current-tree fused native GPU graph path (`44431974`) as the q-head optimization baseline.
- Keep `c90fa94` as the kv-head lower-bound reference for old GPU-topk + CPU graph behavior.
- Do not use current-tree CPU-graph path (`44431973`) as the main baseline.

## Next steps
1. Create clean named copies/branches for:
   - old GPU-topk + CPU graph (`c90fa94`)
   - fused native baseline (`ad4d23e` or `bf4ab79` clean state)
   - experimental native work (current dirty tree + separate flash-attn fork branch)
2. Bisect the fused-native regression between `slurm-44245076.out` and `slurm-44370482.out`.
3. If a fair old-vs-new comparison is needed, compare on the same retrieval objective:
   - either port old path to q-head mode,
   - or compare both in kv-head mode.
