# Current Status

## Scope
- Repository: RetrievalAttention / RetroInfer experiments.
- Active branch: `gpu_top_k`.
- Target model for iteration: `meta-llama/Llama-3.1-8B-Instruct`.
- Primary benchmark flow now: `test.sh` first, then RULER subset/full.

## 2026-03-06 update (baseline map + current status)
- Decode traversal GPU experiment (controlled ~40k prompt, `GEN_LEN=32`):
  - preservation:
    - experiment preserved at branch `exp/decode-python-gpu`
    - snapshot commit: `efc234f`
    - active runtime no longer carries the `python_gpu` decode backend
  - native follow-up:
    - active branch now includes a new native backend name: `RETRIEVALATTN_DECODE_BACKEND=roar_cuda`
    - implementation:
      - `third_party/RoarGraph/python_ext/roargraph_torch_ext.cpp`
      - built via `third_party/RoarGraph/python_ext/setup.py`
      - keeps Python out of the hot traversal loop
      - uses batched CUDA scoring from a C++ extension
      - current version is not a fully device-resident frontier/visited design yet
  - workload:
    - `DATA_PATH=benchmark/decode_ab_prompt_32k.json`
    - actual prompt length reported by `simple_test.py`: `Input length: 40001`
    - same prefill path for all runs:
      - `RETRIEVALATTN_FA_GRAPH_FUSED=1`
      - `RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1`
  - production CPU decode baseline:
    - `44436176` (`dec32_cpp`)
    - `RETRIEVALATTN_DECODE_BACKEND=roar_cpp`
    - `Prefilling latency: 132.6654 s`
    - `Decoding latency: 55.3619 s`
    - `decode_profile`:
      - `retrieve=42.786 s`
      - `seed=10.611 s`
      - `graph=21.183 s`
      - `rerank=8.338 s`
      - `visited_total=13414695`
      - `candidates_total=12425410`
  - Python CPU control:
    - historical run: `44436598` (`dec32_py`)
    - current matched run: `44443435` (`dec40_py2`)
    - `RETRIEVALATTN_DECODE_BACKEND=python`
    - current matched result:
      - `Prefilling latency: 132.8356 s`
      - `Decoding latency: 158.0929 s`
    - `decode_profile`:
      - `retrieve=144.261 s`
      - `seed=9.834 s`
      - `graph=123.514 s`
      - `rerank=7.971 s`
      - `visited_total=7613077`
      - `candidates_total=9763681`
  - native CUDA-scoring decode path:
    - `44443433` (`dec40_cuda`)
    - `RETRIEVALATTN_DECODE_BACKEND=roar_cuda`
    - `Prefilling latency: 133.6492 s`
    - `Decoding latency: 75.1058 s`
    - `decode_profile`:
      - `retrieve=63.959 s`
      - `seed=9.929 s`
      - `graph=51.631 s`
      - `rerank=0.010 s`
      - `visited_total=7613077`
      - `candidates_total=9763681`
  - interpretation:
    - against the production CPU C++ decode backend, `roar_cuda` is still slower:
      - total decode: `75.1 s` vs `54.0 s`
      - retrieve: `64.0 s` vs `42.1 s`
      - graph stage: `51.6 s` vs `21.2 s`
    - against the same Python traversal policy, `roar_cuda` is a real speedup:
      - total decode: `158.1 / 75.1 ~= 2.10x` faster
      - retrieve: `144.3 / 64.0 ~= 2.25x` faster
      - graph stage: `123.5 / 51.6 ~= 2.39x` faster
    - `python` and `roar_cuda` have identical `visited_total` / `candidates_total`, so the traversal policy stayed aligned.
    - `roar_cuda` nearly eliminates explicit rerank cost because the native backend returns final ranked candidates directly.
  - decision:
    - `roar_cuda` is good enough to keep as an experimental native backend because it clearly beats Python CPU.
    - it is not yet good enough to replace `roar_cpp`.
    - next work should focus on the remaining gap to `roar_cpp`, especially the `graph` slice.
- `roar_cuda_v2` follow-up on current branch:
  - backend:
    - `RETRIEVALATTN_DECODE_BACKEND=roar_cuda_v2`
    - batched per-kv-group native search call
    - current implementation still uses host-managed frontier state inside the extension, but batches all q-heads in a kv-group together
  - 9k smoke on `spgpu` (`44445972`, `slurm-smk-cuda-v2.out`):
    - `Prefilling latency: 13.9278 s`
    - `Decoding latency: 6.5314 s`
    - `graph=2.495 s`
    - comparison on same 9k workload:
      - `roar_cpp`: `4.5536 s` decode, `graph=1.667 s`
      - `roar_cuda`: `8.2327 s` decode, `graph=5.473 s`
    - interpretation:
      - `roar_cuda_v2` beats `roar_cuda`
      - `roar_cuda_v2` still trails `roar_cpp`
  - first 40k run on `spgpu` (`44479317`, `slurm-dec40-cuda-v2.out`):
    - `Prefilling latency: 138.7633 s`
    - `Decoding latency: 59.6887 s`
    - `graph=20.475 s`
    - `seed=21.354 s`
    - `visited_total=7263443`
    - `candidates_total=9763681`
    - comparison on same 40k workload:
      - `roar_cpp`: `54.0159 s` decode, `graph=21.164 s`, `seed=10.088 s`
      - `roar_cuda`: `75.1058 s` decode, `graph=51.631 s`, `seed=9.929 s`
      - `python`: `158.0929 s` decode, `graph=123.514 s`, `seed=9.834 s`
    - interpretation:
      - `roar_cuda_v2` closed most of the gap:
        - decode: `59.69 s` vs `54.02 s` (`~1.10x` slower than `roar_cpp`)
        - graph stage: `20.48 s` vs `21.16 s` (slightly better than `roar_cpp`)
      - the remaining gap was seed stage, not graph stage
      - `roar_cuda_v2` was materially better than `roar_cuda`
  - grouped GPU seed-scoring update on `spgpu` (`44479409`, `slurm-dec40-cuda-v2-s2.out`):
    - `Prefilling latency: 132.8210 s`
    - `Decoding latency: 46.0405 s`
    - `decode_profile`:
      - `seed=7.062 s`
      - `graph=18.868 s`
      - `rerank=0.000 s`
      - `visited_total=7263443`
      - `candidates_total=9763681`
    - comparison on the same 40k workload:
      - `roar_cpp`: decode `54.0159 s`, seed `10.088 s`, graph `21.164 s`
      - `roar_cuda_v2` final: decode `46.0405 s`, seed `7.062 s`, graph `18.868 s`
    - interpretation:
      - `roar_cuda_v2` now beats `roar_cpp` overall on the A40 run
      - decode speedup: `54.0159 / 46.0405 ~= 1.17x`
      - both seed and graph slices are now better than `roar_cpp`
  - longer-decode validation on the same ~40k prompt (`GEN_LEN=100`):
    - `roar_cpp` (`44479444`, `slurm-dec40-cpp-g100.out`):
      - `Decoding latency: 177.2075 s`
      - `seed=32.306 s`
      - `graph=74.807 s`
    - `roar_cuda_v2` (`44479443`, `slurm-dec40-cuda-v2-g100.out`):
      - `Decoding latency: 146.8723 s`
      - `seed=22.533 s`
      - `graph=60.241 s`
    - interpretation:
      - the `roar_cuda_v2` win holds for longer decode
      - decode speedup: `177.2075 / 146.8723 ~= 1.21x`
  - larger-context validation on ~65k prompt (`GEN_LEN=32`):
    - prompt file: `benchmark/decode_ab_prompt_64k.json`
    - actual prompt length reported by `simple_test.py`: `Input length: 65001`
    - `roar_cpp` (`44479446`, `slurm-dec64-cpp-g32.out`):
      - `Prefilling latency: 337.1427 s`
      - `Decoding latency: 56.0003 s`
      - `seed=10.286 s`
      - `graph=23.587 s`
    - `roar_cuda_v2` (`44479447`, `slurm-dec64-cuda-v2-g32.out`):
      - `Prefilling latency: 336.6468 s`
      - `Decoding latency: 46.2675 s`
      - `seed=7.051 s`
      - `graph=19.149 s`
    - interpretation:
      - the `roar_cuda_v2` win holds at larger context size as well
      - decode speedup: `56.0003 / 46.2675 ~= 1.21x`
  - note:
    - grouped profile accounting is now sane enough that `retrieve_total_sec` no longer explodes above total.
    - walltime plus per-slice `seed` / `graph` remain the preferred comparison metrics.
  - next optimization target:
    - reduce the remaining `other` slice in grouped `roar_cuda_v2`
    - then decide whether to make `roar_cuda_v2` the preferred decode backend over `roar_cpp`
- Important branch/runtime caveat:
  - branch names do not currently map cleanly to distinct runtime families.
  - `cpu_graph_builder_opt` commit `bf4ab79` only adds CPU graph-builder parity harness scripts; it does not define a separate GPU+CPU runtime.
  - the old GPU-topk + CPU-graph runtime exists in older commits such as `c90fa94` / `8e9cdfc`.
  - `ad4d23e` / `bf4ab79` are on the fused-prefill runtime line.
  - current worktree on `cpu_graph_builder_opt` is dirty and includes experimental native-kernel work.
- 32k matched runtime comparison on the current worktree:
  - `44431973` (`cmp32_cpugpu`): `RETRIEVALATTN_FA_GRAPH_FUSED=0`, path=`native_kernel_fused`, CPU graph build (`roar_cpp`) => `Prefilling latency: 143.6257 s`.
  - `44431974` (`cmp32_native`): `RETRIEVALATTN_FA_GRAPH_FUSED=1`, path=`native_kernel_fused_graph` => `Prefilling latency: 97.0912 s`.
  - `44431975` (`cmp32_torch`): `RETRIEVALATTN_FA_FORCE_PYTHON_TOPK=1`, path=`python_retrieval_graph_wrapper` => `Prefilling latency: 115.461 s`.
- 64k matched runtime comparison on the current worktree:
  - `44432451` (`cmp64_native`): `RETRIEVALATTN_FA_GRAPH_FUSED=1`, path=`native_kernel_fused_graph` => `Prefilling latency: 369.7804 s`.
    - steady-state per-layer fused retrieval: `native_retrieval_kernel_sec ~= 10.4-10.7 s`, `native_graph_sec ~= 0.06 s`.
  - `44432452` (`cmp64_torch`): `RETRIEVALATTN_FA_FORCE_PYTHON_TOPK=1`, path=`python_retrieval_graph_wrapper` => `Prefilling latency: 459.9199 s`.
    - steady-state per-layer fused retrieval: `topk_sec ~= 13.27-13.28 s`, `graph_sec ~= 0.27 s`.
  - `44432453` (`cmp64_cpugpu`): `RETRIEVALATTN_FA_GRAPH_FUSED=0`, path=`native_kernel_fused` + CPU graph build => `Prefilling latency: 422.4145 s`.
- 64k interpretation:
  - native fused remains fastest.
  - native fused is faster than forced Torch/Python by about `1.24x` end-to-end (`459.9 / 369.8`).
  - current GPU+CPU remains slower than native fused, though still faster than the Torch/Python path at 64k.
- q-head vs kv-head quality signal on current native path:
  - `44432612` (`kvqh_proxy_8k_v2`) added a native recall diagnostic that compares exact grouped KV-head queries against exact q-head top-k on the same sampled holdout rows.
  - result:
    - q-head native parity remained exact (`recall_weighted = 1.0`),
    - kv-head grouped-query proxy recall was only `0.764678955078125`.
  - interpretation:
    - kv-head grouped retrieval would lose substantial recall relative to the q-head objective,
    - so q-head remains the correct retrieval target for quality.
 - q-head vs kv-head traversal signal on current native path:
   - `44432670` (`kvqh_trav_8k`) added a traversal proxy using grouped KV-head queries on the same current native graph.
   - result:
     - q-head traversal recall: `0.77978515625`
     - grouped KV-head traversal proxy: `0.7744140625`
   - interpretation:
     - traversal recall degrades only slightly under grouped KV-head queries on the current graph,
     - much less than the exact top-k degradation.
   - caveat:
     - this is still not a full kv-head graph-build A/B; it is a grouped-query traversal proxy on the current q-head-built graph.
 - true kv-head graph A/B result on current tree:
   - `44432809` (`kvab_8k_base`) and `44432812` (`kvab_8k_high`) compared:
     - current q-head graph traversal,
     - grouped-query traversal on the current q-head graph,
     - true kv-head graph traversal built offline from exact grouped queries with the same `roar_cpp` builder.
   - results:
     - base budget:
       - q-head graph: `0.8123779296875`
       - grouped-query-on-q-graph proxy: `0.8392333984375`
       - true kv-head graph: `0.209228515625`
     - high budget:
       - q-head graph: `0.8900146484375`
       - grouped-query-on-q-graph proxy: `0.9190673828125`
       - true kv-head graph: `0.2587890625`
   - interpretation:
     - the grouped query itself is not what breaks traversal recall.
     - the true kv-head graph is dramatically worse than the q-head graph, and extra traversal budget does not fix it.
     - decision: keep q-head graph construction; do not switch graph build to kv-head.
- Old GPU-topk + CPU-graph path recreated from `c90fa94` on a GPFS export tree:
  - `44432065` (`c90_32_cpugpu`): `RETRIEVALATTN_FA_FUSED_PREFILL=0`, `RETRIEVALATTN_GPU_TOPK=1`, mode=`gpu_topk` => `Prefilling latency: 100.0403 s`.
  - caveat: this run uses `retrieval_heads=8`, `retrieval_head_mode=kv_head`; it is not apples-to-apples against current q-head fused runs.
  - matched speed-only rerun with parity off:
    - `44432245` (`c90_32_kv_cmp`): `retrieval_head_mode=kv_head` => `Prefilling latency: 97.6988 s`.
  - patched experimental q-head run on a separate exported tree (`RetrievalAttention_c90fa94_qhead_tree`):
    - `44432246` (`c90_32_qh_cmp`): `retrieval_head_mode=q_head` => `Prefilling latency: 218.2938 s`.
  - interpretation:
    - old `gpu_topk` only looks competitive in `kv_head` mode.
    - once made apples-to-apples with `q_head`, it is ~`2.23x` slower than old `kv_head`,
      ~`1.50x` slower than current-tree q-head CPU+GPU path (`143.6257 s`),
      and ~`2.25x` slower than current-tree fused native q-head baseline (`97.0912 s`).
- 119k status:
  - best known clean fused-native baseline: `slurm-44245076.out`, path=`native_kernel_fused_graph`, steady-state `native_core_sec ~= 19.9 s/layer`, total prefill `670.8404 s`.
  - current regressed fused-native run: `slurm-44370482.out`, steady-state `native_core_sec ~= 35.3 s/layer`.
- Experimental status:
  - `v3_warpk8` compiles and runs, but failed as a performance experiment:
    - 8k parity ok,
    - 32k no speedup,
    - `ncu` got worse (`registers/thread 190 -> 213`, occupancy unchanged).
  - forced Python/Torch GPU top-k path is now functionally correct after fixes:
    - causal masking added in `_retrieval_group_topk_blockwise`,
    - Python GPU graph builder now uses first dynamic token as pivot,
    - parity uses causal reference when `retrieval_causal=1`,
    - 8k parity restored to `1.0`,
    - but it is still slower at 32k and is not a good baseline.
- Recommendation:
  - current q-head optimization baseline should be `44431974` (`native_kernel_fused_graph`).
  - `44432065` / `44432245` (`c90fa94` old GPU-topk path) are useful kv-head lower-bound speed references, but not fair q-head baselines.
  - patched `c90fa94` q-head old-path run (`44432246`) shows the old runtime family is not a good q-head optimization baseline.
  - going forward, baseline comparisons should stay within the current tree:
    - `native_kernel_fused_graph` (GPU top-k + GPU graph),
    - `native_kernel_fused` + current CPU graph finalize,
    - forced Torch/Python GPU top-k + GPU graph.
  - `44431973` (current CPU-graph path on this branch) should not be used as the main baseline.
- Next actions:
  1. Create clean named branches/worktrees for:
     - old GPU-topk + CPU-graph baseline (`c90fa94`),
     - fused-native baseline (`ad4d23e` or `bf4ab79` clean tree),
     - experimental native path (current dirty worktree plus separate flash-attn fork branch).
  2. Bisect the native fused regression between the `~20 s/layer` state (`slurm-44245076.out`) and the later `~35 s/layer` state (`slurm-44370482.out`).
  3. If a fair old-vs-new comparison is required, either:
     - run both in `kv_head` mode, or
     - port the old GPU-topk path to q-head mode.

## 2026-03-04 update (kernel-mode + splitk runtime groundwork implemented)
- Implemented native retrieval kernel mode plumbing in flash-attn fork:
  - `RETRIEVALATTN_FA_KERNEL_MODE=legacy|v2_local|v2_splitk` (currently `v2_splitk` maps to v2 local kernel path),
  - `RETRIEVALATTN_FA_SPLITK=auto|0|N`.
- Implemented split selection in native retrieval path:
  - legacy mode keeps split=`1`,
  - v2 mode uses heuristic auto split by seqlen_k (`1/2/4/8` buckets) unless overridden.
- Reworked v2 local update path in `flash_fwd_kernel.h`:
  - removed previous quadratic row scan pattern,
  - replaced with single-pass row-slot accumulation + one locked merge per row-slot,
  - overflow falls back to exact legacy insertion for correctness.
- Extended native profile payload:
  - graph-fused timing tensor now additionally reports `retrieval_kernel_mode` and `retrieval_effective_splits`.
- `test.sh` now exposes and forwards:
  - `RETRIEVALATTN_FA_KERNEL_MODE`,
  - `RETRIEVALATTN_FA_SPLITK`.
- Submitted validation jobs:
  - build: `44297529`,
  - A/B runs (dependency on build): `44297531` (`legacy`), `44297532` (`v2_local`).

## 2026-03-04 update (true-v2 single-compile batch landed; build queued)
- Batched all remaining v2 edits before rebuild to avoid multiple full compiles.
- Native retrieval kernel now supports split-k updates in split-KV kernel path:
  - `retrieval_update_fragment_topk(...)` is called in both non-split and split kernel loops.
- Added split-local retrieval buffer mode (`RETRIEVALATTN_FA_KERNEL_MODE=v2_splitk`):
  - when `num_splits > 1`, kernel writes per-split top-k tensors,
  - post-kernel GPU reduction merges `[split, ..., k] -> [..., k]` via `topk + gather`.
- Added split-stride fields in `Flash_fwd_params` to address per-split output slices safely.
- Updated split heuristic:
  - `v2_splitk` mode uses more aggressive auto split selection for long contexts.
- `test.sh` default kernel mode now set to `v2_splitk` (legacy still available via env override).
- Single rebuild submitted after batching edits:
  - build job: `44298163` (later canceled as stale after instrumentation edits).

## 2026-03-04 update (kernel debug/profile instrumentation added)
- Added env-gated kernel instrumentation in flash-attn retrieval path:
  - `RETRIEVALATTN_FA_KERNEL_PROFILE=0|1`
  - `RETRIEVALATTN_FA_KERNEL_DEBUG=0|1`
- Native retrieval now logs phase timings:
  - `native_retrieval_profile: kernel=<...> merge=<...> total=<...>`
  - `merge` isolates split-output reduction (`topk + gather`) overhead.
- Added optional device-side debug counters:
  - candidate total / in-bounds / causal-filtered / norm-filtered /
    locked-call count / local-call count / overflow fallback / merged rows.
  - emitted as `native_retrieval_debug: ...`.
- Timing payloads extended and wired to Python profiles:
  - topk-only path can return optional retrieval timing tensor.
  - graph-fused timing now additionally includes:
    - `retrieval_split_outputs`
    - `native_retrieval_kernel_sec`
    - `native_retrieval_merge_sec`
    - `native_retrieval_total_sec`
- `test.sh` now forwards and logs kernel instrumentation env flags.
- Stale build canceled and resubmitted after these edits:
  - canceled: `44298163`
  - active: `44298703` (compile failed: `cS/tScS` scope issue in `flash_fwd_kernel.h`).
  - patched and resubmitted: `44298749`.
- Added post-build execution helpers:
  - submit A/B matrix: `benchmark/submit_kernel_mode_ab.sh`
  - parse logs: `benchmark/extract_kernel_profiles.sh`
- Queued A/B matrix with build dependency:
  - build job: `44298749`
  - dependent runs: `44299179` (legacy), `44299180` (v2_local), `44299181` (v2_splitk).

## 2026-03-03 update (next-session plan: fused retrieval v2)
- Context:
  - top-k lock-contention fix and A/B runtime switch were added (`RETRIEVALATTN_FA_TOPK_BATCHED=0|1`).
  - the larger bottleneck remains native retrieval core time (`native_core_sec`) on long prefill.
- Immediate first task when resuming:
  - check job outputs from the latest queued runs (`slurm-44245118.out`, `slurm-44245119.out`, `slurm-44245120.out`),
  - confirm build success and compare batched-vs-legacy `native_core_sec` on the same long prompt config.
- Explicit target and harness:
  - target: reduce `native_core_sec` from ~`19.7s/layer` at ~`119k` tokens toward Full-Flash scale (`~5-7s/layer` on A40).
  - keep one fixed benchmark setup for all A/B (`gen_len=1`, fixed prompt length, same model/hardware/flags).
  - required per-layer metrics: `native_core_sec`, `native_graph_sec`, `native_total_sec`.
  - add kernel-internal counters for split count and merge/reduction time in v2 path.
- Kernel architecture plan (v2):
  1. replace global lock-per-score updates with lock-free online top-k in attention CTAs:
     - maintain running per-row top-k in registers/shared memory while sweeping K tiles,
     - do tile-local candidate extraction + small-k merge each tile,
     - final sort once per row at end of K sweep,
     - avoid writing full score matrices and avoid global lock/atomic contention.
  2. reintroduce K-parallelism using deterministic 2-pass split-K:
     - pass A: each K-split CTA emits partial top-k for its K-range,
     - pass B: reduction kernel merges partial lists to final top-k (deterministic ordering),
     - remove current scalability ceiling from single-split sweep on long contexts.
  3. keep graph build lightweight and overlap by Q-chunks (not monolithic in-kernel overlap):
     - stream A: attention+top-k chunk `c+1`,
     - stream B: graph update chunk `c`,
     - use double buffers + CUDA events.
- Runtime safety and rollout:
  - add new runtime path name: `native_kernel_fused_graph_v2`,
  - keep existing path as fallback,
  - expose knobs: split-K enable, chunk size, temporary memory budget.
- Correctness gates before claiming speed:
  1. top-k parity vs causal reference (`causal_ref=1`) across sampled layers/heads.
  2. no empty-graph regressions (`edges=0` only when expected by data).
  3. traversal recall not worse than current fixed-pivot baseline.
  4. deterministic outputs under fixed seed on same hardware.
- Milestones:
  - A: lock-free non-split kernel >`2x` faster than current core at ~`119k`.
  - B: split-K adds >`1.5x` over non-split at ~`119k`.
  - C: fused v2 prefill within `~1.3-1.6x` of Full FlashAttention prefill.

## 2026-03-03 update (graph-fused prototype path landed; native kernel not yet landed)
- Implemented a new **graph-fused prefill runtime path** behind flags:
  - `RETRIEVALATTN_FA_GRAPH_FUSED=1`
  - `RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=0|1`
  - `RETRIEVALATTN_FA_GRAPH_FUSED_CHECK=0|1`
  - `RETRIEVALATTN_FA_GRAPH_FUSED_QUALITY_FLOOR` (default `0.90`)
- RetrievalAttention prefill wrapper now tries:
  - `flash_attn_with_kvcache_retrieval_graph(...)` when graph-fused mode is enabled,
  - falls back to existing `flash_attn_with_kvcache_retrieval(...)` when disabled or on error (unless `REQUIRE=1`).
- Cache build flow now accepts per-layer graph payloads and can skip CPU Roar graph build:
  - `register_fused_prefill_knn(..., graph_neighbors=...)`
  - dense-neighbor `[kv_head, seq, m]` is converted to CSR and committed directly.
- Added quality safety gate:
  - when parity traversal eval is enabled and strict traversal recall falls below floor,
  - graph-fused per-head graph is replaced by legacy CPU Roar build for that head.
- Important clarification:
  - this is currently a **GPU/Torch graph prototype**, not a native C++/CUDA fused kernel in flash-attn.
  - graph construction from top-k is implemented in `flash_attn_interface.py` using GPU tensor ops (torch), not in flash-attn CUDA kernels.
- Extension-side hook status:
  - interface now probes for native symbol `fwd_kvcache_retrieval_graph` and will use it if present,
  - current fork does **not** yet implement that symbol in `flash_api.cpp` / CUDA kernels.
- Next required step for true fusion:
  - implement `fwd_kvcache_retrieval_graph` natively in flash-attn C++/CUDA,
  - move graph build math out of Python/Torch fallback path.

## 2026-02-26 update (runtime cleanup: fused-only path)
- RetrievalAttention runtime is now intentionally **fused-only** for prefill index build:
  - no non-fused CPU/GPU top-k build path in `prepare_cache()`,
  - no Triton custom qk+topk fallback path in cache runtime.
- FlashAttention integration is now strict:
  - `attn_hub/retrievalattention_attn.py` requires `flash_attn_with_kvcache_retrieval` directly from installed flash-attn,
  - wrapper expects tuple return `(attn_out, retrieval_topk_idx[, profile])`.
- Fused registration is now strict q-head contract:
  - accepted fused top-k head dimension must match `retrieval_heads` (`num_heads` in current runtime),
  - previous q-head/kv-head compatibility reshapes were removed.
- Shadow compare path is deprecated/ignored:
  - `RETRIEVALATTN_FA_SHADOW_COMPARE` is ignored with warning.
- Run scripts simplified:
  - removed legacy knobs from `test.sh` (GPU-topk/custom-kernel/graph-builder/backend selectors),
  - `benchmark/ruler/ruler_run_wrapper.sh` now unconditionally validates RoarGraph C++ extension.

## 2026-02-26 update (holdout recall mode)
- Added holdout-aware graph quality evaluation without changing model path:
  - `RETRIEVALATTN_GRAPH_TRAIN_FRAC`: use only a prefix fraction of query rows for graph construction.
  - `RETRIEVALATTN_PARITY_HOLDOUT_ONLY=1`: parity samples only unseen holdout query rows.
- This allows recall checks on queries not used to build the K-graph.

## 2026-02-26 update (fused output-shape + downstream head-structure migration)
- Fused prefill retrieval output is now migrated to **per-q-head** layout:
  - expected shape in cache registration: `[seq, num_heads, q_knn]` (or equivalent transposed forms).
  - this replaces prior per-KV-head fused layout.
- Downstream head structures are now split by role:
  - decode seed index storage (`self.indexes`) remains **KV-head keyed** (shared keys in GQA),
  - graph / hub-seed structures are **KV-head keyed** (shared K-token graph per KV head),
  - previous decode seeds remain **retrieval-head keyed**.
- Added retrieval-head to KV-head mapping in cache:
  - `q_head` mode maps `retrieval_head // group_size -> kv_head`,
  - `kv_head` mode is identity.
- Decode path now supports true per-q-head retrieval:
  - in `q_head` mode, retrieval + graph traversal run per query head,
  - graph/hub lookup uses mapped KV head (shared graph),
  - dynamic/static KV gather also uses mapped KV head.
- Fused prefill graph build behavior in `q_head` mode:
  - grouped q-head top-k rows are merged per KV head before graph projection,
  - one graph is built per KV head from the merged rows.
- `prepare_cache()` logging now reports:
  - `retrieval_heads`, `retrieval_head_mode` alongside `kv_heads`.
- Compatibility note (historical):
  - flash-attn fork native retrieval kernel was updated to emit per-q-head top-k.
  - temporary q-head/kv-head reshape bridges were used during migration and later removed in fused-only cleanup.

## 2026-02-26 update (tiny recall-only harness for fast iteration)
- Added parity summary API in `cache_hub/retrievalattention_cache.py`:
  - `get_parity_summary(reset=False)` returns aggregate recall stats and per-layer/head records.
- Parity sampling scope is now configurable:
  - `RETRIEVALATTN_PARITY_LAYERS`
  - `RETRIEVALATTN_PARITY_HEADS`
  - `RETRIEVALATTN_PARITY_SAMPLE`
- Removed hardcoded parity check on only `layer=0/head=0`; parity now runs for the configured scope.
- Added recall-only mode in `simple_test.py`:
  - flags: `--recall_only --recall_input_tokens --recall_min_recall`.
  - behavior: synthetic token input -> prefill/index build -> parity summary print; skips decode loop.
- Wired `test.sh` for recall-only runs:
  - env flags: `RECALL_ONLY`, `RECALL_INPUT_TOKENS`, `RECALL_MIN_RECALL`.
  - when `RECALL_ONLY=1`, script auto-forces `RETRIEVALATTN_VALIDATE_PARITY=1`.

## 2026-02-13 latest update (decode C++ traversal integration)
- Implemented decode-side RoarGraph C++ search path in extension:
  - new binding: `search_graph_csr(...)` in `third_party/RoarGraph/python_ext/roargraph_builder.cpp`,
  - supports `fp32`, `fp16`, and `bf16` key storage (bf16 via uint16 bit-view to avoid float32 duplication).
- Added Python wrapper:
  - `cache_hub/roargraph_cpp_backend.py::search_roar_graph_csr_cpp(...)`.
- Integrated into decode retrieval flow in `cache_hub/retrievalattention_cache.py`:
  - new selector: `RETRIEVALATTN_DECODE_BACKEND=auto|python|roar_cpp` (default `auto`),
  - C++ path is used for CSR graphs; `auto` mode falls back to Python traversal on runtime error,
  - strict `roar_cpp` mode fails fast if extension is missing/unusable.
- Added decode C++ traversal controls:
  - `RETRIEVALATTN_ROAR_DECODE_INIT` (default `64`),
  - `RETRIEVALATTN_ROAR_DECODE_LPQ` (`0` => candidate target),
  - `RETRIEVALATTN_ROAR_DECODE_MAX_CMPS` (`0` => uncapped),
  - `RETRIEVALATTN_ROAR_DECODE_MAX_HOPS` (`0` => use `RETRIEVALATTN_MAX_VISITS`),
  - `RETRIEVALATTN_ROAR_DECODE_THREADS`.
- Validation completed:
  - extension rebuilt successfully with `module load python/3.10.4 && source .venv/bin/activate && python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`,
  - local smoke tests passed for `fp32` and `bf16` decode search paths,
  - fixed duplicate-ID issue in C++ decode queue insertion.
- Immediate next run task:
  - run `sbatch test.sh` and compare decode profile (`seed`, `graph`, `visited_total`, `candidates_total`) against previous Python-traversal logs.

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

## 2026-02-12 update (RoarGraph C++ backend for graph build)
- Implemented C++ graph-build backend and integrated it into `cache_hub/retrievalattention_cache.py`:
  - runtime selector: `RETRIEVALATTN_ROAR_BACKEND=cpp|python|auto`,
  - cpp path uses module `roargraph_builder_ext` from `third_party/RoarGraph/python_ext`,
  - python Roar builder remains as fallback/debug path.
- Added backend loader: `cache_hub/roargraph_cpp_backend.py`.
- Added extension build artifacts/sources:
  - `third_party/RoarGraph/python_ext/roargraph_builder.cpp`,
  - `third_party/RoarGraph/python_ext/setup.py`.
- Run scripts now default to C++ backend and fail fast if missing:
  - `test.sh`,
  - `benchmark/ruler/ruler_run_wrapper.sh`.
- Build command:
  - `module load python/3.10.4`
  - `source .venv/bin/activate`
  - `python third_party/RoarGraph/python_ext/setup.py build_ext --inplace`
- Local smoke status:
  - extension build succeeded,
  - direct `build_graph_csr(...)` smoke call returned valid CSR and metadata.

## Immediate next step (locked)
- Decode still uses adaptive best-first frontier traversal (not paper-style beam traversal yet).
- Next implementation target is decode traversal refactor to beam-search style on top of current graph:
  - replace/augment frontier expansion with Roar/HNSW-style beam candidate maintenance,
  - keep retrieval budget fairness (`token_budget=100`) fixed for A/B,
  - compare `legacy` vs `roar` graph builders under the same decode traversal settings first, then switch decode traversal algorithm.
- Parallel track after this:
  - evaluate porting decode traversal hot path to C++ as well (graph build path is now C++-backed).

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

## 2026-03-03 native graph-fused flash-attn status
- Added a native C++ entrypoint in the flash-attn fork:
  - `fwd_kvcache_retrieval_graph` (bound in `csrc/flash_attn/flash_api.cpp`).
- Current implementation details:
  - reuses existing native fused prefill top-k path (`mha_fwd_kvcache_retrieval`),
  - builds graph neighbors in C++ on CUDA tensors (no Python graph-build fallback required when symbol is available),
  - returns `(out, softmax_lse, retrieval_indices, graph_neighbors)` for the graph-fused API.
- Status:
  - source patch is in place and rebuilt successfully.
  - validation:
    - build: `slurm-44244822.out` (success),
    - smoke: `slurm-44244823.out` (success, `path=native_kernel_fused_graph`).
  - smoke output confirmed:
    - `idx` shape `(1, 64, 32, 32)` (`int32`),
    - `graph` shape `(1, 8, 64, 16)` (`int32`),
    - native symbol `fwd_kvcache_retrieval_graph` is available and callable.
