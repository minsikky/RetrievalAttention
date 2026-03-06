# Context Checkpoint 2026-03-06

## Current baseline map
- Stay on current-tree runtime families only:
  - `native_kernel_fused_graph`: native fused GPU top-k + GPU graph
  - `native_kernel_fused`: native fused GPU top-k + current CPU graph finalize
  - `python_retrieval_graph_wrapper`: forced Torch/Python GPU top-k + GPU graph
- Old `c90fa94` runtime is no longer relevant as an optimization baseline.

## Best current baselines
- `32k`
  - native fused GPU graph: `44431974` => `97.0912 s`
  - current GPU+CPU: `44431973` => `143.6257 s`
  - forced Torch/Python: `44431975` => `115.461 s`
- `64k`
  - native fused GPU graph: `44432451` => `369.7804 s`
  - current GPU+CPU: `44432453` => `422.4145 s`
  - forced Torch/Python: `44432452` => `459.9199 s`

## Key conclusions
- Native fused remains the optimization baseline.
- Native fused beats forced Torch/Python at both `32k` and `64k`.
- Current GPU+CPU is slower than native fused, but still better than forced Torch/Python at `64k`.

## kv_head vs q_head conclusion
- Simple grouped-query proxy was misleading.
  - grouped KV-head queries on the existing q-head graph only changed traversal recall slightly.
- True graph A/B was decisive.
  - `8k` base budget:
    - q-head graph traversal: `0.8123779296875`
    - true kv-head graph traversal: `0.209228515625`
  - `8k` high budget:
    - q-head graph traversal: `0.8900146484375`
    - true kv-head graph traversal: `0.2587890625`
- Interpretation:
  - the grouped query is not the main problem
  - the grouped-query-built kv-head graph is the problem
  - more traversal budget does not rescue the kv-head graph
- Decision:
  - keep `q_head` graph construction
  - do not switch graph build to `kv_head`

## Important experiment flags
- `RETRIEVALATTN_KV_GRAPH_AB=1`
  - enables offline true graph A/B inside the current recall harness
  - builds an alternate kv-head graph from exact grouped queries using the same builder
- Related parity outputs:
  - `kv_proxy`: exact grouped-query top-k overlap vs q-head target
  - `kv_proxy_traversal`: grouped-query traversal on the existing q-head graph
  - `kv_graph_traversal`: true traversal recall on the alternate kv-head graph

## Next steps
1. Focus optimization only on `native_kernel_fused_graph`
2. Bisect / explain regression from the older `~20 s/layer` 119k fused-native state to the later `~35 s/layer` state
3. If exploring cheap decode-time approximations, only test grouped queries at traversal time; keep q-head graph build fixed
