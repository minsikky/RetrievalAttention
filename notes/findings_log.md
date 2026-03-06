# Findings Log

## 2026-03-06 update (baseline comparison + branch/runtime map)
- Branch map correction:
  - `cpu_graph_builder_opt` tip commit `bf4ab79` is not a separate GPU+CPU runtime branch; it only adds CPU graph-builder parity harness scripts.
  - the old GPU-topk + CPU-graph runtime exists in older commits such as `c90fa94` / `8e9cdfc`.
  - `gpu_top_k` tip (`ad4d23e`) is already on the fused-prefill runtime line.
  - current working tree on `cpu_graph_builder_opt` contains uncommitted native-kernel experiments and should be treated as a separate experimental state.
- Clean 32k comparison on the current tree:
  - `44431973` (`cmp32_cpugpu`): path=`native_kernel_fused` + CPU `roar_cpp` graph build => `Prefilling latency: 143.6257 s`.
  - `44431974` (`cmp32_native`): path=`native_kernel_fused_graph` => `Prefilling latency: 97.0912 s`.
  - `44431975` (`cmp32_torch`): path=`python_retrieval_graph_wrapper` => `Prefilling latency: 115.461 s`.
  - conclusion: on the current codebase, fused-native GPU graph build is the best 32k baseline.
- Clean 64k comparison on the current tree:
  - `44432451` (`cmp64_native`): path=`native_kernel_fused_graph` => `Prefilling latency: 369.7804 s`.
    - steady-state per-layer fused retrieval: `native_retrieval_kernel_sec ~= 10.4-10.7 s`
    - steady-state per-layer fused graph build: `native_graph_sec ~= 0.06 s`
  - `44432452` (`cmp64_torch`): path=`python_retrieval_graph_wrapper` => `Prefilling latency: 459.9199 s`.
    - steady-state per-layer wrapper top-k: `topk_sec ~= 13.27-13.28 s`
    - steady-state per-layer graph build: `graph_sec ~= 0.27 s`
  - `44432453` (`cmp64_cpugpu`): path=`native_kernel_fused` + CPU `roar_cpp` graph build => `Prefilling latency: 422.4145 s`.
  - conclusion: 64k preserves the same ordering as 32k:
    - native fused GPU graph < current GPU+CPU < forced Torch/Python GPU top-k
    - native fused is ~`1.24x` faster than Torch/Python end-to-end at 64k.
- Native q-head vs kv-head quality proxy (same current native path, 8k holdout recall run):
  - job: `44432612`
  - log: `slurm-kvqh-proxy-8k-v2.out`
  - setup:
    - current native fused graph path,
    - `RECALL_ONLY=1`,
    - `RECALL_INPUT_TOKENS=8192`,
    - holdout queries only (`GRAPH_TRAIN_FRAC=0.9`, `PARITY_HOLDOUT_ONLY=1`),
    - parity over first 2 layers / 8 heads.
  - result:
    - q-head native parity remained exact: `recall_weighted = 1.0`
    - grouped kv-head proxy recall dropped to `kv_proxy.recall_weighted = 0.764678955078125`
  - interpretation:
    - replacing per-q-head retrieval with grouped KV-head-average retrieval loses about `23.5%` of the true q-head top-8 mass on this test.
    - this supports keeping q-head as the retrieval objective.
- Native q-head vs kv-head traversal proxy (same current native path, same 8k holdout setup):
  - job: `44432670`
  - log: `slurm-kvqh-trav-8k.out`
  - result:
    - q-head traversal recall: `traversal.recall_mean = 0.77978515625`
    - grouped kv-head query traversal proxy: `kv_proxy_traversal.recall_mean = 0.7744140625`
  - interpretation:
    - grouped KV-head queries only reduced traversal recall slightly on the current q-head-built graph (`~0.54` percentage points absolute),
    - much smaller than the exact top-k mismatch (`kv_proxy.recall_weighted ~= 0.7647`).
    - likely reason: current traversal recall is already dominated by graph/search limitations, so grouped-query degradation is mostly masked in traversal.
  - caveat:
    - this is not yet a full kv-head graph-build experiment; it uses grouped KV-head queries on the current q-head-built graph.
- True graph A/B on current tree (same builder, same q-head holdout queries, same traversal policy):
  - implementation:
    - `RETRIEVALATTN_KV_GRAPH_AB=1`
    - current q-head graph comes from the normal current-tree path (`native top-k + current CPU graph build`)
    - alternate kv-head graph is built offline from exact grouped KV-head queries using the same `roar_cpp` builder
  - `8k` base-budget run:
    - job: `44432809`
    - log: `slurm-kvab_8k_base.out`
    - summary:
      - q-head graph traversal recall: `0.8123779296875`
      - grouped-query-on-q-graph traversal proxy: `0.8392333984375`
      - true kv-head graph traversal recall: `0.209228515625`
  - `8k` high-budget run:
    - job: `44432812`
    - log: `slurm-kvab_8k_high.out`
    - summary:
      - q-head graph traversal recall: `0.8900146484375`
      - grouped-query-on-q-graph traversal proxy: `0.9190673828125`
      - true kv-head graph traversal recall: `0.2587890625`
  - interpretation:
    - the grouped query itself is not the main problem.
    - the grouped-query-built kv-head graph is the problem.
    - increasing traversal budget does **not** rescue the kv-head graph.
    - therefore kv-head graph construction should be rejected for quality-sensitive use; keep q-head graph construction.
- Old path from `c90fa94` (exported tree on GPFS, job `44432065`):
  - config: `RETRIEVALATTN_FA_FUSED_PREFILL=0`, `RETRIEVALATTN_GPU_TOPK=1`, `RETRIEVALATTN_ROAR_BACKEND=cpp`.
  - log reports `mode=gpu_topk`.
  - result: `Prefilling latency: 100.0403 s`.
  - this is much better than the current branch's CPU-graph path (`143.6 s`) and close to the fused-native baseline (`97.1 s`).
  - caveat: `c90fa94` uses `retrieval_heads=8`, `retrieval_head_mode=kv_head`; this is not apples-to-apples with the current q-head fused baseline.
 - Matched old-path speed-only reruns from separate exported trees:
   - `44432245`: old `gpu_topk` with `retrieval_head_mode=kv_head` => `Prefilling latency: 97.6988 s`.
   - `44432246`: patched old `gpu_topk` with `retrieval_head_mode=q_head` => `Prefilling latency: 218.2938 s`.
   - interpretation:
     - the old runtime family only looks competitive in `kv_head` mode.
     - once made apples-to-apples with `q_head`, it becomes much slower than both the current-tree q-head CPU+GPU path (`143.6257 s`) and the current-tree fused-native q-head path (`97.0912 s`).
     - therefore `c90fa94` should be treated only as a kv-head lower-bound reference, not a q-head optimization baseline.
- Best known 119k fused-native baseline remains:
  - `slurm-44245076.out`: steady-state `native_core_sec ~= 19.9 s/layer`, total prefill `670.8404 s`.
- Current regressed 119k fused-native state:
  - `slurm-44370482.out`: steady-state `native_core_sec ~= 35.3 s/layer`.
- `v3_warpk8` experiment result:
  - compiles and runs.
  - 8k parity ok.
  - 32k A/B: no useful speedup.
  - `ncu` comparison vs v2:
    - `registers/thread`: `190 -> 213`
    - `theoretical occupancy`: unchanged at `16.67%`
    - `eligible warps / scheduler`: slightly worse
    - `kernel duration`: slightly worse
  - conclusion: failed experiment; do not use as default baseline.
- Forced Python/Torch GPU top-k path:
  - added `RETRIEVALATTN_FA_FORCE_PYTHON_TOPK=1`.
  - fixed correctness issues:
    - causal masking in `_retrieval_group_topk_blockwise(...)`,
    - Python GPU graph builder now uses first dynamic token as pivot (matching native path),
    - parity logic now keys off `retrieval_causal` in profile.
  - after fixes:
    - 8k parity restored to `1.0` (`44429181`),
    - 32k still slower than native fused (`44429201`: `117.5961 s`).
  - conclusion: functionally usable, not performance baseline.
- Practical baseline recommendation:
  - q-head target baseline: `44431974` (`native_kernel_fused_graph`).
  - kv-head lower-bound reference: `44432065` (`c90fa94`, `mode=gpu_topk`).
  - for future comparisons, ignore the old runtime family and compare only the current-tree paths:
    - native fused GPU graph,
    - current GPU+CPU finalize path,
    - forced Torch/Python GPU top-k path.

## 2026-03-04 update (implemented kernel optimization checkpoint)
- Implemented first pass of retrieval-kernel optimization hooks and v2 local path:
  - native env controls:
    - `RETRIEVALATTN_FA_KERNEL_MODE=legacy|v2_local|v2_splitk`
    - `RETRIEVALATTN_FA_SPLITK=auto|0|N`
  - legacy keeps split=1, v2 uses split heuristic when enabled.
- Replaced earlier quadratic batched row scan logic in `retrieval_update_fragment_topk`:
  - now single-pass fragment traversal with fixed row slots and one lock-merge per slot,
  - per-score legacy fallback for rare slot overflow.
- Added profile visibility for new mode/split in graph-fused native timing payload:
  - `retrieval_kernel_mode`,
  - `retrieval_effective_splits`.
- Pending verification:
  - build + A/B jobs submitted (`44297529`, `44297531`, `44297532`) to measure whether v2 path reduces `native_core_sec`.

## 2026-03-04 update (single-compile true-v2 batch)
- Added split-k retrieval updates to split-KV flash kernel loops:
  - retrieval top-k now updates during split kernel execution (not only non-split path).
- Added split-local top-k output mode (`RETRIEVALATTN_FA_KERNEL_MODE=v2_splitk`):
  - kernel writes per-split retrieval buffers,
  - final GPU reduction merges per-split candidates using `topk + gather`.
- Added split-stride metadata in `Flash_fwd_params` for retrieval tensors.
- Updated mode semantics:
  - `legacy` -> mode `0`,
  - `v2_local` -> mode `1`,
  - `v2_splitk` -> mode `2`.
- Updated split heuristic for mode `2` to be more aggressive on long contexts.
- `test.sh` default switched to `RETRIEVALATTN_FA_KERNEL_MODE=v2_splitk`.
- Rebuild submitted after batching all edits:
  - `44298163` (later canceled as stale after instrumentation edits).

## 2026-03-04 update (kernel instrumentation for debug/profiling)
- Added kernel-phase profiling knobs:
  - `RETRIEVALATTN_FA_KERNEL_PROFILE=1` enables retrieval-phase timing lines.
  - `RETRIEVALATTN_FA_KERNEL_DEBUG=1` enables retrieval debug counters.
- Native retrieval now emits:
  - `native_retrieval_profile` with `kernel/merge/total` timing breakdown.
  - `native_retrieval_debug` with counter summary:
    - `cand_total`, `in_bounds`, `causal_filtered`, `norm_filtered`,
      `locked_calls`, `local_calls`, `overflow_fallback`, `merged_rows`.
- Added optional timing payload from `fwd_kvcache_retrieval` and wired it through Python profile parsing.
- Extended graph-fused timing payload/profile parsing with:
  - `retrieval_split_outputs`,
  - `native_retrieval_kernel_sec`,
  - `native_retrieval_merge_sec`,
  - `native_retrieval_total_sec`.
- `test.sh` now logs/forwards both kernel instrumentation env flags.
- Build sequence after instrumentation:
  - canceled stale build: `44298163`
  - resubmitted build: `44298703` (failed compile).
- Compile failure detail (`slurm-44298703.out`):
  - duplicate `cS` / `tScS` declaration in non-split kernel scope,
  - missing `tScS` in split-kernel scope where retrieval update is called.
- Fix applied:
  - removed duplicate non-split declaration,
  - added split-scope `cS/tScS` near split-kernel MMA fragment setup.
- Rebuild resubmitted: `44298749`.

## 2026-03-04 update (top-k batched merge A/B regression)
- A/B results on identical long-context config (`~119k` prefill tokens):
  - `slurm-44245119.out` (`RETRIEVALATTN_FA_TOPK_BATCHED=0`): `native_core_sec` mean `~34.10s/layer`.
  - `slurm-44245120.out` (`RETRIEVALATTN_FA_TOPK_BATCHED=1`): `native_core_sec` mean `~71.45s/layer`.
  - previous baseline (`slurm-44245076.out`): `native_core_sec` mean `~19.87s/layer`.
- Conclusion:
  - current batched-merge implementation is a regression and should not be default.
- Root-cause hypothesis from kernel structure:
  - batched path adds quadratic per-fragment work (`seen_row` scan + per-row full `j` rescan in `flash_fwd_kernel.h`),
  - lock reduction did not compensate for the added compute/divergence overhead.
- Immediate safety change:
  - default switched back to legacy path:
    - `flash_api.cpp`: `RETRIEVALATTN_FA_TOPK_BATCHED` now defaults OFF.
    - `test.sh`: `RETRIEVALATTN_FA_TOPK_BATCHED` default set to `0`.
- Next action:
  - replace this batched path with a true lock-free online top-k design (single-pass fragment accumulation, no O(n^2) row scans).

## 2026-03-03 update (graph-fused prefill prototype)
- Added graph-fused prefill runtime integration (flagged) in RetrievalAttention path:
  - prefill wrapper can now call `flash_attn_with_kvcache_retrieval_graph(...)`,
  - cache can consume fused graph payload and bypass CPU Roar graph build.
- Added quality guard for rollout:
  - `RETRIEVALATTN_FA_GRAPH_FUSED_CHECK` + `RETRIEVALATTN_FA_GRAPH_FUSED_QUALITY_FLOOR` fallback to legacy graph build when strict traversal recall is below floor.
- Current limitation:
  - this is **not** yet a native flash-attn fused CUDA graph kernel.
  - graph build currently runs in Python interface via GPU torch ops over top-k output.
- Native kernel status:
  - interface support for `fwd_kvcache_retrieval_graph` probing is added,
  - extension symbol itself is not implemented yet in `flash_api.cpp`/CUDA.

## 2026-03-03 update (decode complexity sweep automation)
- Added sweep submission automation:
  - `benchmark/submit_decode_complexity_sweep.py`
  - submits matrix jobs across `N`, regime families (`linear`, `sqrt`, `log`), and split seeds,
  - emits a TSV manifest (`job_id`, regime params, traversal budget knobs, scaled graph params).
- Added post-run regime summarizer:
  - `benchmark/summarize_decode_complexity.py`
  - ingests `collect_recall_sweep.py` CSV output and computes:
    - per-`(regime, N)` frontier (minimum visit-rate row meeting target recall),
    - per-regime hit-rate and `mean_visit_rate_at_target`,
    - observed scaling exponent (`obs_alpha`) from `trav_visited_mean` vs `N`,
    - configured scaling exponent (`cfg_alpha`) from `max_visits` vs `N`.
- Added runbook section:
  - stage-1 coarse sweep command,
  - collection/summarization commands,
  - stage-2 multi-seed confirmation command.
- Current default protocol in new scripts:
  - holdout split on queries only (`train_frac=0.9`, `split=stratified`, `PARITY_HOLDOUT_ONLY=1`),
  - strict traversal metric target (`trav_recall >= 0.95`),
  - first-pass graph scaling: `ROAR_M` scales with `N` (sqrt law), `ROAR_L/E` fixed.

## 2026-02-27 update (strict traversal recall metric + graph sweep)
- Traversal recall metric was tightened to strict top-k:
  - `trav_recall` now compares `retrieved[:k]` vs exact reference top-k.
  - previous coverage-style metric is retained as `trav_recall_cov`.
- Traversal eval now disables seed-floor forcing during eval-only retrieval:
  - decode runtime behavior unchanged,
  - strict metric now reflects ranking quality instead of seed-floor ordering artifacts.
- Added reusable sweep parser updates (`benchmark/collect_recall_sweep.py`):
  - supports arbitrary TSV columns,
  - extracts graph-build overhead from logs (`graph_build_time/proj/edges` aggregates).
- Graph-construction strict sweep at ~3% traversal budget (`max_visits=256`, fixed traversal knobs):
  - `roar_m/l/enhance_l = 8/4/4`: strict recall `0.8584` @ `2.72%`.
  - `12/8/8`: `0.9082` @ `2.71%`.
  - `16/12/12`: `0.9160` @ `2.70%`.
  - `24/16/16`: `0.9414` @ `2.68%`.
  - `32/20/20`: `0.9541` @ `2.67%` (best in sweep).
- Confirmation runs for best config (`32/20/20`) on different split seeds:
  - seed `4321`: `0.9561` @ `2.67%`,
  - seed `9999`: `0.9678` @ `2.68%`.
- Conclusion:
  - target achieved: strict recall `>=0.95` near `3%` traversal budget.

## 2026-02-27 update (query holdout split semantics fix)
- Confirmed graph build is already query-holdout-only (keys are not held out), but split policy was contiguous:
  - train queries: prefix rows,
  - holdout queries: suffix rows.
- Under causal reference, contiguous-prefix train under-covers late-position keys, which can collapse holdout traversal recall for tail-heavy holdout.
- Implemented configurable query split policy in `cache_hub/retrievalattention_cache.py`:
  - `RETRIEVALATTN_GRAPH_SPLIT=stratified|random|contiguous`,
  - `RETRIEVALATTN_GRAPH_SPLIT_SEED=<int>`,
  - default set to `stratified` for recall experiments.
- `test.sh` now exports these flags and logs them at run start.
- Parity summary now includes `graph_split_mode` and `graph_split_seed` for run traceability.

## 2026-02-27 update (fused top-k parity gap root cause)
- Investigated non-trivial fused top-k parity gap (~7% vs exact causal reference).
- Root causes found in flash-attn retrieval kernel (`third_party/flash-attn-ra/csrc/flash_attn/src/flash_fwd_kernel.h`):
  - retrieval candidates were inserted **before** causal/local masking, so masked future positions could pollute top-k.
  - lock path used bounded spin (`kMaxSpinIters=1024`) and could drop candidates under contention, making top-k best-effort rather than exact.
- Fix applied in kernel source:
  - enforce causal/local window filtering before insertion in `retrieval_update_fragment_topk`.
  - switch to exact lock behavior (no candidate drop on contention) in `retrieval_topk_insert_locked`.
- Operational note:
  - this fix is in the flash-attn fork source and requires rebuilding/installing that fork before new runs.
  - expected tradeoff: better parity correctness, potential top-k runtime increase due stricter locking.

## 2026-02-27 update (traversal recall metric mismatch)
- T1/T2/T3 traversal saturation runs showed:
  - parity recall stayed `1.0`,
  - traversal recall stayed around `0.0117` despite increased visit rate (`~5.4% -> ~21.2%`).
- Root cause in metric definition:
  - traversal eval compared retrieval output (decode path: non-causal, dynamic-key-only search) against parity reference top-k (causal full-key top-k).
  - this is not an apples-to-apples objective, so traversal recall was artificially depressed and flat.
- Fix in `cache_hub/retrievalattention_cache.py`:
  - added `_decode_dynamic_topk_ref_np(...)` for exact dynamic-range decode-style reference,
  - traversal eval now uses this decode-space reference (`trav_ref=decode_dynamic` in parity log),
  - parity recall path remains unchanged (still causal reference for fused-topk correctness).

## 2026-02-26 update (legacy path cleanup)
- Simplified RetrievalAttention runtime to fused-prefill-only index build path.
- Removed cache-runtime custom Triton qk+topk path and dead non-fused prepare-cache branches.
- Tightened FlashAttention wrapper to a single API contract using `flash_attn_with_kvcache_retrieval`.
- Simplified launch scripts to remove inactive legacy toggles and require RoarGraph C++ extension by default.
- Outcome:
  - lower configuration surface for active experiments,
  - less risk of accidental fallback to stale/unsupported build paths.

## 2026-02-26 update (q-head fused retrieval shape + head-mapped downstream structures)
- Implemented fused prefill top-k shape migration from KV-head to q-head:
  - fused registration now expects retrieval-head dimension (`retrieval_heads`), defaulting to `num_heads` in fused mode.
  - temporary compatibility bridges were used during migration:
    - q-head fused output can be collapsed to kv-head mode by selecting one q-head per GQA group,
    - legacy kv-head fused output can be expanded to q-head mode by group repeat.
- Native flash-attn retrieval kernel path updated accordingly:
  - retrieval output buffers now allocate head dimension as `num_heads` (q-head),
  - kernel-side retrieval accumulation now writes per-q-head (no KV-head gate), while KV norm lookup still maps by GQA ratio.
- Cache/index data-structure split is now explicit:
  - decode Faiss index: KV-head keyed,
  - graph + hub seeds: KV-head keyed (shared by grouped q-heads),
  - previous decode seeds: retrieval-head keyed.
- Decode compute path now supports per-q-head retrieval in `RETRIEVALATTN_RETRIEVAL_HEAD_MODE=q_head`:
  - retrieval/traversal/rerank per query head,
  - graph/hub and KV gather both use mapped KV head.
- Fused q-head graph build now merges grouped q-head top-k rows per KV head and builds one shared KV-head graph.
- Expected behavior:
  - better objective alignment with “retrieval should be per q-head” requirement.
  - increased graph build cost proportional to retrieval-head count vs KV-head mode.
- Operational caveat:
  - requires rebuilding flash-attn fork after C++/CUDA changes; old builds will not match new expected fused shape.

## 2026-02-26 update (parity semantics switched to per-q-head)
- Parity validation now supports per-`q`-head comparison for each KV group head:
  - for a validated KV head, collect all grouped query heads (`group_size`),
  - compute per-query-head top-k reference,
  - report mean recall plus min/max range across grouped query heads.
- Causal-aware reference remains enabled for native fused prefill parity (`causal_ref=1`).
- Latest run (`slurm-43878067.out`) with this mode:
  - `parity layer=0 head=0 sample=256 recall@32=0.5788 range=[0.3884,0.8604] qh=4 mode=per_q_head_mean causal_ref=1`.

## 2026-02-26 update (recall-only validation path)
- Implemented a tiny recall-only path for fast algorithm A/B without full decode:
  - `simple_test.py --recall_only --recall_input_tokens <N>`.
  - uses synthetic token IDs, runs prefill + index build, then reports parity summary JSON.
- Added parity aggregation in RetrievalAttention cache:
  - collects per-layer/head sampled recall records,
  - exports `get_parity_summary(reset=False)` with mean/min/max/weighted recall.
- Parity scope is now configurable (`RETRIEVALATTN_PARITY_LAYERS`, `RETRIEVALATTN_PARITY_HEADS`, `RETRIEVALATTN_PARITY_SAMPLE`) instead of fixed `layer=0/head=0`.
- `test.sh` now supports `RECALL_ONLY=1` and can enforce a threshold gate via `RECALL_MIN_RECALL`.

## 2026-02-13 update (decode C++ traversal backend)
- Added decode-side C++ traversal API to the Roar extension:
  - `search_graph_csr(query, keys, offsets, neighbors, init_ids, init_scores, ...)`.
- Added Python wrapper:
  - `search_roar_graph_csr_cpp(...)` in `cache_hub/roargraph_cpp_backend.py`.
- RetrievalAttention decode integration:
  - new env selector: `RETRIEVALATTN_DECODE_BACKEND=auto|python|roar_cpp`,
  - `auto` uses C++ on CSR graphs and falls back to Python traversal if runtime call fails,
  - strict `roar_cpp` mode errors out on missing/broken extension.
- Implemented low-memory key handling:
  - decode C++ path reads existing CPU key storage directly,
  - bf16 path uses `torch.bfloat16 -> torch.uint16 view -> numpy` (no float32 shadow copy).
- Added decode C++ knobs:
  - `RETRIEVALATTN_ROAR_DECODE_INIT`,
  - `RETRIEVALATTN_ROAR_DECODE_LPQ`,
  - `RETRIEVALATTN_ROAR_DECODE_MAX_CMPS`,
  - `RETRIEVALATTN_ROAR_DECODE_MAX_HOPS`,
  - `RETRIEVALATTN_ROAR_DECODE_THREADS`.
- Smoke/validation results:
  - extension rebuild succeeded after changes,
  - fp32 and bf16 search calls returned valid top-k + metadata,
  - fixed duplicate candidate IDs by adding duplicate-handling in C++ queue insert.

## 2026-02-12 update (RoarGraph C++ graph-build backend)
- Implemented a C++ backend for Roar-style graph construction and wired it into RetrievalAttention cache build flow.
- Added extension source/build path:
  - `third_party/RoarGraph/python_ext/roargraph_builder.cpp`
  - `third_party/RoarGraph/python_ext/setup.py`
- Added Python loader/wrapper:
  - `cache_hub/roargraph_cpp_backend.py`
- Builder selection now supports:
  - `RETRIEVALATTN_ROAR_BACKEND=cpp|python|auto`
- Run-script defaults now use C++ backend and fail-fast on missing extension import:
  - `test.sh`
  - `benchmark/ruler/ruler_run_wrapper.sh`
- Smoke result:
  - extension build succeeded,
  - direct extension call returned valid CSR (`offsets`, `neighbors`) and meta (`builder=roar_cpp`, `stop_reason=ok`).
- Expected impact:
  - reduce graph-build wall time by eliminating Python-loop overhead in projection/enhancement stages.
  - next bottleneck remains decode traversal (`graph` stage in decode profile), requiring separate optimization.

## 2026-02-02 to 2026-02-04 summary

### Environment / setup
- Missing Python binary in job script caused early failures.
- Module + venv alignment was required to avoid system torch contamination.
- Flashinfer import required matching torch env (`2.5.1` in venv).

### Memory
- 131k context runs can exceed 32 GB CPU RAM.
- 64 GB+ CPU memory is often needed for stable long-context runs.

### Performance evolution
- CPU-only/faiss topk build was too slow (minutes per head).
- GPU-topk significantly improved prefill index build speed.
- Transfer overhead was dominant before layer-wise GPU caching.
- Layer-wise GPU caching and overlap reduced transfer to near-zero in profile.
- Current observed heavy component per head after transfer fix:
  - projection step still substantial (often ~1.5-2.7s/head).

### Quality observation
- A run showed severe output degradation:
  - repeated `[INST]` pattern instead of expected content.
- Likely cause:
  - decode retrieval path had no reliable seed index in GPU-topk flow.
- Mitigation added:
  - decode seed index retained via `faiss` in GPU-topk mode.
  - debug counters + assertion for empty dynamic retrieval.
- Decode runtime error encountered and fixed:
  - `TypeError: Got unsupported ScalarType BFloat16` in decode seed query conversion.
  - Fix: cast decode query tensor to float before numpy conversion.

### 2026-02-05 update (latest state)
- Log `slurm-41844983.out` shows:
  - parity check `recall@32=1.0000` for sampled GPU-topk vs faiss knn.
  - decode retrieval non-empty (debug: `empty_heads=0/8`, dynamic counts in hundreds).
  - therefore primary catastrophic correctness bug is addressed.
- Remaining issue appears mainly quality-related:
  - output still underperforms RetroInfer on simple coded-word test.
  - model output includes prompt-style continuation (`[/INST]`, many `...`) more than desired.
- Interpretation:
  - likely not a hard retrieval-empty bug anymore.
  - likely due to current graph projection/retrieval design quality (anchor-style projection, grouped-query averaging, effective dynamic coverage).

### 2026-02-05 update (quality-recovery implementation)
- Implemented decode-side quality controls in `cache_hub/retrievalattention_cache.py`:
  - retrieval query mode switch (`per_head` default, optional `group_avg`),
  - seed-floor enforcement via `RETRIEVALATTN_SEED_RATIO` (default 0.7),
  - optional graph expansion toggle (`RETRIEVALATTN_GRAPH_EXPAND`),
  - exact-dot rerank over a bounded candidate pool (`RETRIEVALATTN_RERANK`, `RETRIEVALATTN_CAND_MULT`),
  - rerank score aggregation mode (`RETRIEVALATTN_RERANK_AGG=max|mean`).
- Updated `test.sh` to expose/forward new flags for reproducible A/B runs.
- Next run should compare:
  - `per_head + rerank + graph_expand=1` (new default path),
  - `per_head + rerank + graph_expand=0` (seed-only ablation),
  - with fixed token budget and same prompt/model settings.

### 2026-02-05 update (adaptive traversal)
- Replaced fixed-hop decode graph traversal with adaptive best-first expansion.
- New stopping behavior:
  - requires minimum exploration (`RETRIEVALATTN_MIN_VISITS`),
  - stops on stable top-k + frontier-score gap (`RETRIEVALATTN_STOP_PATIENCE`, `RETRIEVALATTN_STOP_MARGIN`),
  - hard-capped by `RETRIEVALATTN_MAX_VISITS`.
- New traversal controls:
  - `RETRIEVALATTN_EXPAND_WIDTH`,
  - `RETRIEVALATTN_FRONTIER_TOPN`,
  - `RETRIEVALATTN_SEED_K_MULT`.
- `RETRIEVALATTN_GRAPH_HOPS` is now deprecated and ignored.

### 2026-02-05 update (adaptive traversal run result)
- Log: `slurm-41847189.out`
- Configuration used:
  - `RETRIEVALATTN_EXPAND_WIDTH=64`
  - `RETRIEVALATTN_MIN_VISITS=0` (auto -> 2070)
  - `RETRIEVALATTN_MAX_VISITS=0` (auto -> 16560)
  - `RETRIEVALATTN_STOP_PATIENCE=2`
  - `RETRIEVALATTN_STOP_MARGIN=0.0`
- Outcome:
  - quality improved (decoded top-3 line matches RetroInfer line in this sample),
  - decode latency became very high: `8663.44 ms/step` (`0.12 tok/s`),
  - prefill remained large: `526.457 s`.
- Conclusion:
  - adaptive traversal direction is quality-positive,
  - auto visit limits are too permissive for practical decode speed at this context/budget.
  - next iteration should tune decode traversal caps first.

### 2026-02-05 output interpretation clarification
- `simple_test.py` prints:
  - `Answer: ...` => dataset ground truth,
  - decoded string on next line => model output.
- `[/INST]` and long dot sequences are generated continuation artifacts, not a logging bug.
- For this task, prioritize top-3 coded-word accuracy as primary metric; keep continuation artifacts as secondary qualitative signal.

### 2026-02-05 update (decode bottleneck instrumentation + safer defaults)
- Added decode critical-path profiling in `cache_hub/retrievalattention_cache.py`:
  - retrieval sub-timers (`seed`, `graph`, `rerank`, `finalize`),
  - dynamic gather time,
  - attention compute time,
  - full decode compute total.
- `model_hub/LLM.py` now prints retrieval decode profile summary at end of decode (if available).
- Updated `test.sh` defaults to avoid expensive auto adaptive limits:
  - `RETRIEVALATTN_EXPAND_WIDTH=48`
  - `RETRIEVALATTN_MIN_VISITS=256`
  - `RETRIEVALATTN_MAX_VISITS=2048`
  - `RETRIEVALATTN_STOP_PATIENCE=1`
  - `RETRIEVALATTN_STOP_MARGIN=0.001`
  - `RETRIEVALATTN_DECODE_PROFILE=1`

### 2026-02-06 update (custom fused kernel blocked; pivot decision)
- Goal of this cycle:
  - replace torch GPU-topk build path with Triton custom fused qk+topk path.
- Logs and observed outcomes:
  - `slurm-41850330.out`:
    - run reached custom launch and printed `chunk 1/30`.
    - then no additional progress for extended time (appeared stuck).
  - `slurm-41850375.out`:
    - Triton compile failure around `tl.cat` reorder assertion.
  - `slurm-41850455.out`:
    - Triton compile failure (`tl.cat` rank assertion for non-1D tensor).
  - `slurm-41850475.out`:
    - compile errors addressed, but run still remained at `chunk 1/30` with no further progress.
- Interpretation:
  - this is not a decode-seed correctness issue.
  - blocker is custom kernel feasibility/stability/perf in current formulation on long-context workload.
  - likely dominated by Triton codegen/runtime behavior for this fused+sorted kernel shape rather than a simple sync bug.
- Decision:
  - freeze Triton custom fused path for active experimentation (keep code behind flag for reference).
  - move primary optimization effort to FlashAttention-kernel-level prefill fusion (piggyback retrieval index-building signals during prefill attention).

### 2026-02-10 update (FlashAttention prefill-fusion build workflow + current run)
- Build workflow correction:
  - login node does not provide CUDA toolkit (`nvcc` unavailable), so flash-attn editable build must run on compute node through Slurm.
  - `install_2.sh` is now the intended build entry for flash-attn fork (`sbatch install_2.sh`).
- Current build run:
  - log: `slurm-42144800.out`
  - observed healthy startup:
    - `nvcc` resolved on compute node (`/sw/pkgs/arc/cuda/12.8.1/bin/nvcc`),
    - `CUDA_HOME` resolved correctly from `nvcc`,
    - prior metadata-stage failure is no longer present.
  - observed progress:
    - build entered `flash_attn_2_cuda` compile (`running build_ext`, ninja active),
    - reached at least `24/85` object compiles and continued running.
- Warnings seen (non-fatal so far):
  - CUDA minor mismatch warning: toolchain `12.8` vs torch wheel `cu124`,
  - compiler-bounds warning from `torch.utils.cpp_extension`.
- Practical note for future build time:
  - flash-attn setup uses `FLASH_ATTN_CUDA_ARCHS` (not `FLASH_ATTENTION_CUDA_ARCHS`).
  - keeping default multi-arch list makes build much longer.
  - for A100-only environment, `FLASH_ATTN_CUDA_ARCHS=80` reduces compile scope significantly.
  - for mixed A100/A40 environment, use `FLASH_ATTN_CUDA_ARCHS=\"80;86\"`.

### 2026-02-10 update (fused-prefill functional verification)
- Build + install status:
  - `slurm-42153021.out` completed successfully (`Successfully built/installed flash_attn-2.7.3`).
- Native API smoke status:
  - `slurm-42274991.out` passed.
  - `flash_attn_2_cuda` exports `fwd_kvcache_retrieval`.
  - wrapper profile reported `native_kernel_fused`.
- Confirmed functional flow in current implementation:
  - prefill: GPU FlashAttention computes forward + retrieval top-k,
  - post-prefill: CPU still performs per-head finalize work:
    - decode seed index build (Faiss path),
    - projected graph build (CSR).
  - decode uses seed index + graph expansion as before.
- Correctness caveat (ranking semantics):
  - baseline GPU-topk path used explicit L2 normalization before top-k (cosine-style ranking),
  - current fused kernel path uses raw QK scores,
  - these are equivalent only when key norms are effectively constant; otherwise top-k membership/order can change.
- Next optimization direction:
  - evaluate GPU/CPU overlap for prefill finalize stage (producer-consumer style),
  - measure whether synchronization overhead offsets gains before committing.

### 2026-02-10 update (fused-prefill memory utilization regression)
- Log comparison:
  - `slurm-42274992.out` (`RETRIEVALATTN_GPU_TOPK=1`) shows process GPU memory around `16.2 GB` at `generate.before_kv_cache`.
  - `slurm-42275514.out` (fused-prefill path, `RETRIEVALATTN_GPU_TOPK=0`) shows process GPU memory around `33.2 GB` at the same phase.
- Both runs reported similar torch allocator counters:
  - `CUDA(alloc/res) ~= 15378/15380 MB` near start,
  - so the large gap is not explained by normal tracked torch tensor allocations.
- Current interpretation:
  - fused route likely introduces additional non-torch device memory residency (workspace/persistent buffers/runtime allocations), causing much higher nvidia-smi process usage.
- Action status:
  - acknowledged and recorded; no memory optimization patch applied yet in this turn.
  - prefill speed gain remains the primary positive outcome; memory regression is a tracked follow-up item.

### 2026-02-10 update (retrieval scoring objective consistency)
- Added a single retrieval scoring-mode flag:
  - `RETRIEVALATTN_SCORE_MODE=ip|cosine` (default `ip`).
- End-to-end consistency fix:
  - prefill index/graph construction, decode seed scoring, graph expansion scoring, and rerank now share the same objective selection.
- Native fused FlashAttention retrieval update:
  - `fwd_kvcache_retrieval` now accepts `retrieval_normalize` in the updated fork build.
  - Native fused top-k behavior by mode:
    - `ip`: raw signed QK dot-product scores.
    - `cosine`: normalized scores `qk / (||q|| * ||k|| + eps)`.
- Compatibility guard:
  - if `RETRIEVALATTN_SCORE_MODE=cosine` is requested with an older native extension that does not support `retrieval_normalize`, runtime raises an explicit rebuild error instead of silently mixing objectives.
- Practical effect:
  - mixed-objective behavior is removed for both native and fallback fused prefill paths.

### 2026-02-10 update (decode full-scan seed removal path)
- Implemented decode seed strategy switch:
  - `RETRIEVALATTN_SEED_MODE=graph_only` (default),
  - `RETRIEVALATTN_SEED_MODE=faiss` (legacy full-scan reference).
- In `graph_only` mode, `_retrieve_tokens` no longer does global `IndexFlatIP.search` per head/step.
- New graph-only seed sources:
  - previous-step retrieved tokens (`RETRIEVALATTN_SEED_PREV_K`),
  - per-head high-degree graph hubs precomputed at build time (`RETRIEVALATTN_SEED_HUB_K`),
  - dynamic-tail anchor seeds (`RETRIEVALATTN_SEED_TAIL_K`).
- Expected effect:
  - seed-stage decode time should drop significantly relative to full-scan faiss seed mode.

### 2026-02-10 update (status after seed-mode implementation)
- `graph_only` seed path has been implemented and wired through `test.sh` defaults.
- As of this checkpoint, we do **not** yet have a new decode profile log proving the seed-time reduction.
- Next required measurement:
  - run with `RETRIEVALATTN_SEED_MODE=graph_only` (default),
  - run with `RETRIEVALATTN_SEED_MODE=faiss`,
  - compare decode profile `seed` and `graph` slices at matched budget/settings.
- Updated hypothesis:
  - `seed` should decrease,
  - traversal (`graph`) may remain dominant and becomes the next hard target for decode throughput.

### 2026-02-10 update (seed-mode A/B measured; traversal now bottleneck)
- A/B comparison with same adaptive traversal settings:
  - `slurm-42275514.out`: legacy `RETRIEVALATTN_SEED_MODE=faiss`,
  - `slurm-42277995.out`: `RETRIEVALATTN_SEED_MODE=graph_only`.
- Decode profile result:
  - decode total improved `1268.3138s -> 945.1223s`,
  - `seed` improved `308.814s -> 10.546s`,
  - `graph` changed `896.496s -> 880.674s` (still dominant),
  - retrieve remains ~98% of decode.
- Candidate expansion efficiency:
  - `visited_total` nearly unchanged (`~24.87M`),
  - `candidates_total` stayed high (`~78-80M`),
  - only ~31% of scored candidates become visited nodes.
- Practical interpretation:
  - seed full-scan removal worked and removed one major inefficiency,
  - decode bottleneck has shifted to graph expansion/candidate scoring overhead,
  - next optimization target is traversal selectivity (not seed search).
- Quality signal:
  - this speedup did not improve answer quality on the coded-word sample; retrieval quality remains a parallel concern.

### 2026-02-10 update (fused-prefill overlap implementation)
- Implemented overlap path for fused-prefill:
  - layer top-k registration now submits async CPU finalize per layer,
  - finalize builds decode index + projected graph + hub seeds in background,
  - `prepare_cache()` acts as barrier/wait before decode starts.
- Added runtime flags:
  - `RETRIEVALATTN_FUSED_PREFILL_OVERLAP` (`1` default),
  - `RETRIEVALATTN_FUSED_PREFILL_OVERLAP_WORKERS` (`1` default, recommended).
- Added early top-k tensor reference release after fused registration in attention wrapper to shorten GPU-side lifetime.

### 2026-02-10 update (weighted graph projection quality upgrade)
- Replaced unweighted star-only graph projection with weighted co-occurrence projection (default enabled):
  - star edges are now counted by repeated anchor-candidate co-occurrence.
  - added clique-lite projection among top-M candidate neighbors per query row.
- Degree pruning behavior changed:
  - per-node neighbor cap now keeps top-weight edges (deterministic tie-break by neighbor id),
  - avoids prior insertion-order truncation quality loss under tight degree caps.
- Added graph projection flags:
  - `RETRIEVALATTN_GRAPH_WEIGHTED` (`1` default),
  - `RETRIEVALATTN_GRAPH_CLIQUE_M` (`6` default),
  - `RETRIEVALATTN_GRAPH_RETURN_WEIGHTS` (`0` default),
  - `RETRIEVALATTN_GRAPH_WEIGHT_DTYPE` (`uint16` default, `uint32` optional).
- Storage/runtime compatibility:
  - graph tuples can now be `(offsets, neighbors)` or `(offsets, neighbors, weights)`.
  - decode traversal currently consumes neighbors only; stored weights are for future weighted traversal and diagnostics.

### 2026-02-12 update (Roar-style graph build path landed)
- Implemented graph-builder dispatch with `RETRIEVALATTN_GRAPH_BUILDER=legacy|roar`.
- New Roar builder path now performs:
  - query->base bridge extraction from prefill KNN,
  - neighborhood-aware projection using `AcquireNeighbors`-style diversification,
  - reverse-edge updates during projection,
  - connectivity enhancement with beam-style candidate collection + reverse-edge updates,
  - CSR export compatible with existing decode path.
- Added Roar controls:
  - `RETRIEVALATTN_ROAR_NQ`, `RETRIEVALATTN_ROAR_L`, `RETRIEVALATTN_ROAR_M`,
  - `RETRIEVALATTN_ROAR_ENABLE_ENHANCE`, `RETRIEVALATTN_ROAR_ENHANCE_L`,
  - `RETRIEVALATTN_ROAR_ENTRY`, `RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT`,
  - `RETRIEVALATTN_ROAR_LOG`.
- Head-build logs now include optional per-stage stats (`bip`, `enh`, `csr`, active queries/pivots, projected/enhanced nodes, stop reason).
- Test-path defaults changed to current iteration target:
  - static window now `128 + 512`,
  - dynamic budget override default `100`,
  - `RETRIEVALATTN_MIN_VISITS` default reduced to `96`,
  - `RETRIEVALATTN_GRAPH_BUILDER=roar` default in `test.sh`.
- Important status:
  - graph construction is now closer to paper.
  - decode traversal is still adaptive best-first frontier expansion; not yet Roar/HNSW-style beam traversal.
  - next major step is decode traversal refactor to beam search under fixed budget fairness.
- Live-run behavior note:
  - logs may show many `fused_overlap submit layer=...` lines before any `index built layer=... head=...`.
  - with `RETRIEVALATTN_FUSED_PREFILL_OVERLAP_WORKERS=1`, this indicates queue backlog and CPU finalize bottleneck rather than GPU stall.

### 2026-03-03 update (native flash-attn graph-fused symbol added in source)
- Added `fwd_kvcache_retrieval_graph` to `third_party/flash-attn-ra/csrc/flash_attn/flash_api.cpp`.
- Implementation path:
  - call native `mha_fwd_kvcache_retrieval` first (attention + fused top-k),
  - build graph neighbors directly in C++ using CUDA tensor ops (`unique` + stable lexicographic ranking + per-node degree cap),
  - return native outputs with graph neighbors appended.
- Motivation:
  - remove Python-level graph build overhead and make graph-fused prefill available through a native extension symbol.
- Current status:
  - code patch is complete and validated by smoke on CUDA compute.

### 2026-03-03 update (native graph-fused build/smoke fixes)
- First rebuild attempt failed (`slurm-44244667.out`) due C++ tensor API mismatches in new graph helper:
  - unsupported tensor bitshift operators (`<<`, `>>`),
  - template parse issue on `item<bool>()`,
  - bool-mask indexing form rejected by compiler.
- Applied fixes in `flash_api.cpp`:
  - replaced bitshift packing with arithmetic packing (`src * (1<<32) + dst`) and arithmetic unpacking,
  - replaced mask filtering with `at::masked_select`,
  - replaced `item<bool>()` checks with `item().toBool()`.
- Second run built but smoke failed due return-arity mismatch (`too many values to unpack`):
  - `fwd_kvcache_retrieval_graph` returned 5 tensors (included retrieval scores).
  - fixed contract to return 4 tensors expected by Python wrapper:
    - `(out, softmax_lse, retrieval_indices, graph_neighbors)`.
- Final validation:
  - build success: `slurm-44244822.out`,
  - smoke success: `slurm-44244823.out`,
  - smoke profile confirms native path: `path=native_kernel_fused_graph`.

### Reference logs
- Baseline quality reference: `simple_test.out`
- Regression example: `slurm-41814588.out`
- Transfer profiling: `slurm-41810674.out`, `slurm-41811894.out`
- Decode crash fixed: `slurm-41844519.out`
- Latest successful but lower-quality run: `slurm-41844983.out`
- Custom fused blocked/stall evidence: `slurm-41850330.out`, `slurm-41850375.out`, `slurm-41850455.out`, `slurm-41850475.out`
- Fused-prefill build success: `slurm-42153021.out`
- Native fused smoke pass: `slurm-42274991.out`
