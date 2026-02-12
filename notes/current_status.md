# Current Status

## Scope
- Repository: RetrievalAttention / RetroInfer experiments.
- Active branch: `gpu_top_k`.
- Target model for iteration: `meta-llama/Llama-3.1-8B-Instruct`.
- Primary benchmark flow now: `test.sh` first, then RULER subset/full.

## 2026-02-12 latest update (Roar graph-build integration)
- Implemented Roar-style graph construction path in `cache_hub/retrievalattention_cache.py`:
  - builder dispatch: `RETRIEVALATTN_GRAPH_BUILDER=legacy|roar`,
  - Roar pipeline: query-base bipartite bridge, neighborhood-aware projection (`AcquireNeighbors`-style), reverse-edge updates, and connectivity enhancement via beam-style candidate collection.
- Added Roar graph-build controls:
  - `RETRIEVALATTN_ROAR_NQ`,
  - `RETRIEVALATTN_ROAR_L`,
  - `RETRIEVALATTN_ROAR_M`,
  - `RETRIEVALATTN_ROAR_ENABLE_ENHANCE`,
  - `RETRIEVALATTN_ROAR_ENHANCE_L`,
  - `RETRIEVALATTN_ROAR_ENTRY`,
  - `RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT`,
  - `RETRIEVALATTN_ROAR_LOG`.
- Build logs now include graph-builder stage stats (`bip`, `enh`, `csr`, active queries/pivots, projected/enhanced node counts, stop reason).
- Runtime defaults for current fast-iteration path:
  - static pattern moved to `128 + 512`,
  - default dynamic budget override is `100`,
  - adaptive decode defaults now include `RETRIEVALATTN_MIN_VISITS=96`.
- `test.sh` now defaults to `RETRIEVALATTN_GRAPH_BUILDER=roar` and forwards all Roar env flags.
- Runtime observation from current run logs:
  - fused-prefill producer (`flashattn fused prefill layer=...`) advances quickly across layers,
  - CPU finalize (`index built layer=... head=...`) lags significantly with `fused_overlap_workers=1`,
  - interpretation: graph construction/finalize on CPU is the active bottleneck, not GPU top-k generation.

## Immediate next step (locked)
- Decode still uses adaptive best-first frontier traversal (not paper-style beam traversal yet).
- Next implementation target is decode traversal refactor to beam-search style on top of current graph:
  - replace/augment frontier expansion with Roar/HNSW-style beam candidate maintenance,
  - keep retrieval budget fairness (`token_budget=100`) fixed for A/B,
  - compare `legacy` vs `roar` graph builders under the same decode traversal settings first, then switch decode traversal algorithm.
- Parallel track after this:
  - evaluate porting CPU-hot graph/traversal routines to C++ (RoarGraph-inspired kernels) to remove Python-loop bottlenecks.

## 2026-02-10 latest update (decode optimization focus)
- Implemented decode seed refactor to avoid full-scan seed search by default:
  - `RETRIEVALATTN_SEED_MODE=graph_only` is now default.
  - Seeds come from previous-step retrieved tokens + graph hubs + dynamic-tail anchors.
- Measurement status:
  - implementation complete, but no new post-change profile log has been recorded yet in notes.
  - need A/B run (`graph_only` vs `faiss`) to quantify `seed` time drop.
- Current optimization priority:
  - traversal acceleration (`graph` time in decode profile), since retrieval traversal is still expected to dominate decode wall time even after seed improvement.

## 2026-02-10 in-progress update (prefill fusion track)
- Current active optimization track in this thread is **prefill speed** via FlashAttention-kernel-level retrieval fusion.
- Decode speed optimization is intentionally tracked in a separate thread/workstream to avoid mixing regressions.
- FlashAttention fork now includes:
  - Python API export `flash_attn_with_kvcache_retrieval`,
  - native extension symbol `fwd_kvcache_retrieval` (true fused path when compiled).
- Build workflow correction:
  - login node has no CUDA toolchain, so flash-attn build must run as a Slurm compute job.
  - use `sbatch install_2.sh` (compute-node build script).
- Current build run:
  - log: `slurm-42144800.out`
  - status at latest check: metadata stage succeeded, native extension compile started, reached at least `24/85` ninja objects and still running.
  - no fatal build error observed yet in that run.

## 2026-02-10 update (decode seed-mode refactor)
- Decode seeding now supports explicit modes:
  - `RETRIEVALATTN_SEED_MODE=graph_only` (default),
  - `RETRIEVALATTN_SEED_MODE=faiss` (reference/debug).
- `graph_only` mode removes per-step full-scan seed search and seeds traversal from:
  - previous-step retrieved tokens (`RETRIEVALATTN_SEED_PREV_K`),
  - per-head graph hub anchors (`RETRIEVALATTN_SEED_HUB_K`),
  - dynamic-tail anchors (`RETRIEVALATTN_SEED_TAIL_K`).
- Goal: reduce decode seed-stage cost while staying within graph-based ANN retrieval behavior.

## 2026-02-10 update (post-build functional checkpoint)
- FlashAttention fused retrieval build is complete and install succeeded:
  - `slurm-42153021.out` -> `Successfully built flash_attn`, `Successfully installed flash_attn-2.7.3`.
- Native fused API smoke test passed:
  - `slurm-42274991.out`
  - `has fwd_kvcache_retrieval: True`
  - profile shows `path=native_kernel_fused`
  - output/index shapes matched expectations.
- Functional pipeline status in current code:
  - During prefill, GPU runs FlashAttention forward and emits retrieval top-k indices via `flash_attn_with_kvcache_retrieval`.
  - After prefill, `prepare_cache()` consumes fused top-k and CPU performs per-head finalize:
    - build decode seed index (Faiss) when `RETRIEVALATTN_DECODE_INDEX=faiss`,
    - build projected K-K graph (CSR).
  - Decode then uses both:
    - seed search from decode index,
    - adaptive graph expansion from CSR graph.
- Important behavior note:
  - fused FlashAttention retrieval top-k now supports both scoring modes:
    - `ip`: raw signed QK dot-product score (no absolute value),
    - `cosine`: normalized score `qk / (||q|| * ||k|| + eps)`.
  - decode/index/graph paths follow `RETRIEVALATTN_SCORE_MODE` as well, so objective can be consistent end-to-end.

## 2026-02-10 update (retrieval scoring-mode consistency)
- Added end-to-end retrieval scoring control via `RETRIEVALATTN_SCORE_MODE`:
  - `ip` (default): raw inner-product style retrieval scoring.
  - `cosine`: legacy L2-normalized scoring.
- Applied consistently across:
  - prefill index build paths (GPU-topk and CPU faiss build),
  - decode seed search (faiss query + brute-force fallback),
  - graph expansion candidate scoring,
  - final rerank scoring.
- FlashAttention wrapper now forwards `retrieval_normalize` when the interface exposes it, so non-native fallback follows the same score mode.
- Safety guard:
  - if `RETRIEVALATTN_SCORE_MODE=cosine` is requested but native extension does not expose updated `retrieval_normalize` support, code raises an explicit rebuild error to avoid silent mixed-objective behavior.

## 2026-02-10 update (weighted graph projection)
- Upgraded projected graph builder from unweighted star-only dedup to weighted projection:
  - star edges from anchor-candidate pairs are counted by co-occurrence frequency,
  - optional clique-lite edges are added among top-M non-anchor candidates per query row.
- Degree capping now keeps top-weight neighbors per source node (deterministic tie-break by neighbor id), instead of insertion-order truncation.
- Added graph projection controls:
  - `RETRIEVALATTN_GRAPH_WEIGHTED` (default `1`),
  - `RETRIEVALATTN_GRAPH_CLIQUE_M` (default `6`),
  - `RETRIEVALATTN_GRAPH_RETURN_WEIGHTS` (default `0`),
  - `RETRIEVALATTN_GRAPH_WEIGHT_DTYPE` (`uint16` default, `uint32` optional).
- Graph tuple compatibility:
  - decode traversal and hub-seed logic now accept both `(offsets, neighbors)` and `(offsets, neighbors, weights)`.
  - current decode traversal still uses neighbor ids only (edge weights are stored for future traversal weighting / analysis).

## 2026-02-10 update (GPU memory regression observation with fused-prefill run)
- New observation from run comparison:
  - `slurm-42274992.out` (baseline-style run, `RETRIEVALATTN_GPU_TOPK=1`) reported GPU memory around `16.2 GB` (`generate.before_kv_cache`).
  - `slurm-42275514.out` (fused-prefill path, `RETRIEVALATTN_GPU_TOPK=0`) reported GPU memory around `33.2 GB` at the same phase.
- Important detail:
  - PyTorch allocator stats were nearly identical in both logs (`CUDA alloc/res ~= 15.4 GB`),
  - but total GPU memory from process accounting roughly doubled in fused path.
- Interpretation:
  - likely extra non-PyTorch/device-side workspace or retained buffers in the fused path (or adjacent runtime allocations), not normal tensor allocations tracked by `torch.cuda`.
- Decision:
  - do not optimize this yet in current thread; record as the next prefill-fusion follow-up after current integration checkpoints.
- Planned follow-up (next session/workstream):
  - add phase-level memory probes around fused API call boundaries,
  - identify whether extra residency is from flash-attn workspace, persistent buffers, or stream-lifetime retention,
  - reduce peak residency without regressing prefill latency gains.

## 2026-02-10 update (implemented fused-prefill overlap pipeline)
- Implemented layer-level producer/consumer overlap for fused-prefill mode:
  - producer: per-layer `register_fused_prefill_knn(...)` now submits background CPU finalize work,
  - consumer: finalize builds decode index + CSR graph + hub seeds per head/layer,
  - barrier: `prepare_cache()` now waits for outstanding layer futures before decode.
- Added knobs:
  - `RETRIEVALATTN_FUSED_PREFILL_OVERLAP` (default `1`),
  - `RETRIEVALATTN_FUSED_PREFILL_OVERLAP_WORKERS` (default `1`, recommended).
- Added early reference release in fused prefill attention wrapper:
  - drop per-layer `topk_idx` reference right after registration to reduce peak tensor lifetime on GPU side.

## Next prefill optimization target
- Candidate next step: overlap GPU fused-prefill production and CPU head finalize/index-build work.
- Rationale:
  - current path is largely staged (`GPU prefill/top-k` then `CPU finalize/index/graph`),
  - overlap may reduce wall-clock prefill build time.
- Caution:
  - overlap introduces synchronization/queueing complexity and can add overhead if chunking is too fine.
  - should be implemented with coarse producer-consumer batches (e.g., layer/head chunks) and measured.

## Implemented so far
- RetrievalAttention prototype integrated:
  - `cache_hub/retrievalattention_cache.py`
  - `attn_hub/retrievalattention_attn.py`
  - model routing in `model_hub/llama.py` and `model_hub/qwen.py`.
- GPU-topk path for prefill KNN build:
  - blockwise GPU `QK^T` + topk.
  - overlap/pipeline mode and layer-wise GPU caching.
- Instrumentation added:
  - build-time topk/projection timing.
  - memory snapshots in model generation path.
  - RULER prep/pred/eval timing.

## Current quality/perf state
- Major speedup achieved versus CPU-only build path.
- Transfer overhead substantially reduced after layer-wise GPU caching.
- Decode no longer crashes after bf16->numpy fix in decode seed search.
- Quality is improved from worst-case collapse and now closer to RetroInfer on the simple coded-word task.
- Remaining gap is mainly decode latency under adaptive traversal defaults.

## Triton custom fused qk+topk status (frozen)
- Experimental custom kernel path (`RETRIEVALATTN_CUSTOM_QK_TOPK=1`) was iterated with multiple compile-compatibility fixes.
- Current blocker is practical run stability/perf at long context:
  - runs repeatedly stall at first custom chunk (`chunk 1/30`) or fail with Triton compile/runtime constraints.
  - this path is now frozen for active iteration (kept in code behind flag for reference).
- Decision:
  - stop active tuning of this Triton path for now.
  - pivot active work to FlashAttention prefill fusion for index-building signals.

## Latest quality-focused implementation (post-checkpoint)
- Decode retrieval upgraded with explicit quality controls:
  - per-head query mode support for seed search (`RETRIEVALATTN_QUERY_MODE=per_head` default),
  - seed-floor budgeting (`RETRIEVALATTN_SEED_RATIO`, default 0.7),
  - optional graph expansion toggle (`RETRIEVALATTN_GRAPH_EXPAND`),
  - candidate rerank by exact dot product (`RETRIEVALATTN_RERANK=1`, `RETRIEVALATTN_RERANK_AGG=max`),
  - bounded candidate pool via `RETRIEVALATTN_CAND_MULT`.
- `test.sh` now exposes and forwards these new knobs to simple and throughput runs.

## Latest retrieval traversal update
- Decode graph traversal now uses adaptive best-first expansion instead of fixed hop depth.
- Stop conditions are based on:
  - `min_visits`, `max_visits`,
  - top-k stability patience,
  - frontier-best vs current kth-score margin.
- Legacy `RETRIEVALATTN_GRAPH_HOPS` is deprecated and ignored.

## Latest run snapshot (slurm-41847189)
- Config highlights:
  - `RETRIEVALATTN_GRAPH_EXPAND=1`
  - `RETRIEVALATTN_EXPAND_WIDTH=64`
  - auto-resolved `min_visits=2070`, `max_visits=16560` (from budget scaling)
- Observed:
  - Prefill latency: `526.457 s`
  - Decode latency: `8663.44 ms/step` (`0.12 tok/s`)
  - Output top-3 line: `1. wuvfyo 2. vgeqxz 3. rskxnt` (improved quality vs earlier RetrievalAttention runs).
- Interpretation:
  - Quality improved, but decode is too slow with current adaptive limits.
  - Need explicit tighter decode traversal caps for practical throughput.

## Output interpretation note
- In `simple_test.py`, `Answer: [...]` is the ground-truth label.
- The printed decoded string is model output only (not label).
- `[/INST]` and long `...` appearing in output are generated text artifacts; they are useful as a qualitative signal but not the primary metric for coded-word top-3 accuracy.

## Logging update
- `model_hub/LLM.py` now prints decode total latency in seconds in addition to ms/step and tokens/s.
- `cache_hub/retrievalattention_cache.py` now reports decode critical-path breakdown when enabled:
  - retrieval total and subparts (`seed`, `graph`, `rerank`, `finalize`),
  - dynamic KV gather time,
  - attention compute time,
  - residual `other` time.
- `model_hub/LLM.py` prints the decode profile summary at the end of decode if cache supports it.

## Latest default tuning update
- `test.sh` now defaults adaptive traversal to latency-safe values:
  - `RETRIEVALATTN_EXPAND_WIDTH=48`
  - `RETRIEVALATTN_MIN_VISITS=96`
  - `RETRIEVALATTN_MAX_VISITS=2048`
  - `RETRIEVALATTN_STOP_PATIENCE=1`
  - `RETRIEVALATTN_STOP_MARGIN=0.001`
- `RETRIEVALATTN_DECODE_PROFILE=1` is enabled by default in `test.sh`.

## Latest decode profile A/B (seed strategy)
- Compared runs at matched traversal knobs:
  - `slurm-42275514.out` (`RETRIEVALATTN_SEED_MODE=faiss` legacy full-scan seed)
  - `slurm-42277995.out` (`RETRIEVALATTN_SEED_MODE=graph_only`)
- Observed:
  - decode total: `1268.31 s -> 945.12 s` (faster),
  - seed time: `308.814 s -> 10.546 s` (large win),
  - graph time: `896.496 s -> 880.674 s` (still dominant, only small change),
  - retrieve share remained ~98% of decode.
- Candidate processing signal:
  - `visited_total`: `24.87M` in both runs,
  - `candidates_total`: `77.95M -> 79.84M`,
  - visited/candidate ratio ~31%, so traversal fanout/candidate handling is now the main decode bottleneck.
- Quality note from the same A/B:
  - quality did not improve with the faster seed path; top-3 coded-word accuracy remained weak on this sample.
  - conclusion: seed was necessary but not sufficient; next work should target traversal selectivity/quality together.

## Most likely root cause addressed
- GPU-topk mode had decode seed/index mismatch risk.
- Fix added: keep decode seed index (`faiss`) in GPU-topk mode by default.
- Debug/assert tools added to catch empty dynamic retrieval during decode.
- Additional fix added for decode error:
  - cast decode query to float32 before numpy conversion in `_retrieve_tokens`.

## Immediate next validation
1. Keep reliable baseline path for ongoing quality/perf checks:
   - `RETRIEVALATTN_GPU_TOPK=1`, `RETRIEVALATTN_CUSTOM_QK_TOPK=0`.
2. Start FlashAttention-fused index-build design/implementation:
   - collect retrieval top-k candidates during prefill attention compute (instead of separate post-prefill pass).
3. Preserve decode seed quality guardrail:
   - keep `RETRIEVALATTN_DECODE_INDEX=faiss` while fused-prefill path is being validated.
