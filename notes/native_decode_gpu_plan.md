# Native GPU Decode Traversal Plan

## 2026-03-17 update (online insertion-signal direction pruned)
- Tested whether online edge quality could be improved by adding future-query evidence on top of the current birth-step provenance signal.
- Variants tried:
  - future-window replacement
  - future-window additive hybrid
  - future-window rerank-only
- All three failed as net improvements:
  - replacement reduced retrieval strength too much
  - additive hybrid made the graph denser/slower and failed at `e192`
  - rerank-only preserved quality but increased latency
- Therefore the current source has been reverted to the stable `online_next` insertion path.
- Practical implication for next design work:
  - do not spend more time on simple future-window voting over the same birth-step candidates
  - the next promising change should alter the insertion signal more fundamentally rather than refining this particular support-count scheme

## 2026-03-17 update (query-centroid online signal also pruned)
- Tried a more `Q-K`-aligned online signal:
  - accumulate future query vectors that attend to a token while it remains in the static suffix
  - build a bounded local candidate pool from those future queries
  - choose retirement-time neighbors using a query centroid
- Controlled `e192` result:
  - baseline `online_next`:
    - `45413816`
    - `query_acc=0.667`
    - `avg_decode_sec=872.1 s`
  - `query_centroid`:
    - `45413817`
    - `query_acc=0.667`
    - `avg_decode_sec=1095.4 s`
- Therefore:
  - a more principled local `Q-K` signal still did not improve quality
  - it increased update and attention cost substantially
  - the immediate next step should not be another online insertion-signal tweak inside the same stack

## 2026-03-17 update (oracle retrieval result)
- Implemented an oracle decode benchmark mode that bypasses full-GPU graph retrieval during the answer phase with exact top-`k` dynamic token IDs from dense `Q-K` scores.
- Controlled `e192` result:
  - baseline `online`:
    - `45415995`
    - `query_acc=0.667`
    - `avg_decode_sec=1095.4 s`
  - `online_oracle`:
    - `45415996`
    - `query_acc=0.667`
    - `avg_decode_sec=870.2 s`
- Implication:
  - fixing dynamic retrieval quality alone does not recover the dense-quality gap
  - therefore the next design work should move away from insertion-signal tweaking and focus on the sparse attention composition / token budget / static-dynamic partition itself

## 2026-03-17 update (teacher-forced dense ledger result)
- Added `online_oracle_teacher_dense`:
  - dense ledger generation
  - teacher-forced ledger replay into RetrievalAttention
  - oracle answer-phase retrieval
- Controlled `e192` result:
  - no quality improvement versus `online_oracle`
- Therefore:
  - trajectory drift from ledger generation is not the main limiting factor here
  - the next diagnostic should directly measure the trimmed-vs-dense attention distribution and omitted-tail contribution

## 2026-03-17 update (trimmed-vs-dense compare result)
- Implemented `oracle_compare` on `online_oracle` to measure how much dense dynamic attention mass is retained by the oracle top-`k` token set.
- Controlled `e192` diagnostic:
  - `45470959`
  - average captured dense dynamic mass by oracle set: `~0.045`
  - average omitted dense dynamic mass: `~0.417`
  - average dense-vs-sparse output-vector L2: `~0.0208`
- Practical implication:
  - the current token budget is dropping a large amount of aggregate dynamic mass
  - the sparse distribution is not a small perturbation of dense attention
  - the next experiments should focus on support-context / budget / composition, not on graph-update heuristics

## 2026-03-17 update (adaptive budget approximation v1 is too optimistic)
- Implemented a first runtime adaptive-budget approximation under `online_oracle`.
- Controlled `e192` result:
  - baseline:
    - `45474501`
    - `query_acc=0.667`
    - `avg_decode_sec=860.9 s`
    - `adaptive_out/head=57.8`
  - adaptive:
    - `45477893`
    - `query_acc=0.667`
    - `avg_decode_sec=1685.5 s`
    - `adaptive_out/head=14.1`
- Therefore:
  - the current approximation underestimates omitted-mass risk and trims far too aggressively
  - the general adaptive-budget idea remains viable, but this first formula should not be used as-is

## 2026-03-17 update (adaptive budget moved toward GPU)
- Reworked adaptive-budget selection so it now uses attention-space scores and a conservative unseen-tail upper bound.
- Added `oracle_compare` aggregation to benchmark summaries so each run records:
  - dense dynamic mass captured
  - omitted dense dynamic mass
  - dense-vs-sparse output L2
  - adaptive bound diagnostics
- Short fixed-budget oracle baseline at `e12`:
  - `45486114`
  - `query_acc=1.0`
  - `avg_decode_sec=117.18 s`
  - `avg_oracle_dyn_mass=0.1476`
  - `avg_omitted_dynamic_mass=0.2566`
  - `avg_dense_sparse_out_l2=0.0117`
- First conservative adaptive validation runs at `e48/e24/e12` were too slow and were cancelled:
  - `45482239`
  - `45484436`
  - `45486113`
  - `45487815`
  - conclusion:
    - the conservative bound was no longer underestimating omitted mass
    - but the host-managed adaptive implementation was too expensive
- Implemented four adaptive-path reductions before moving to a custom kernel:
  - cache dynamic max attention-key norms during decode
  - reuse attention-stage score tensors instead of recomputing extra adaptive matmuls
  - use prefix/suffix `logsumexp` instead of repeated keep-sweep reductions
  - remove CPU-side token reorder from the adaptive hot path
- Tiny `e6` adaptive smoke after those four changes:
  - `45491673`
  - `query_acc=1.0`
  - `avg_decode_sec=169.90 s`
  - `avg_omitted_dynamic_mass=0.1091`
  - `avg_dense_sparse_out_l2=0.00158`
  - bound violation rate `0.0`
  - remaining adaptive hotspot:
    - `select=59.269 s`
- Added a new custom CUDA adaptive-select path:
  - env:
    - `RETRIEVALATTN_DYNAMIC_BUDGET_MODE=torch|cuda`
  - this keeps the original adaptive path available while adding a GPU selector for dynamic keep-count selection
- Tiny `e6` A/B:
  - torch mode:
    - `45494367`
    - `avg_decode_sec=168.83 s`
    - `adaptive total=80.025 s`
    - `select=58.057 s`
  - cuda mode:
    - `45494368`
    - `avg_decode_sec=122.04 s`
    - `adaptive total=36.780 s`
    - `select=14.400 s`
  - both preserved:
    - `query_acc=1.0`
    - `avg_omitted_dynamic_mass=0.1091`
    - `avg_dense_sparse_out_l2=0.00158`
- Practical conclusion:
  - moving adaptive keep-count selection into the CUDA extension materially reduced adaptive overhead
  - this is the strongest current evidence that the remaining adaptive bottleneck was caused by GPU-CPU collaboration / Python-managed micro-ops
- Follow-up hot-path refactor attempt:
  - keep adaptive keep-count tensors on GPU until the existing output loop
  - avoid full dynamic `K/V` reorder before keep count is known
  - delay dynamic `V` gather until after keep selection
  - fold payload/profile writes into the existing per-head output loop
- Tiny `e6` rerun after that refactor:
  - `45496127`
  - runtime improved modestly vs the immediate pre-patch run `45495775`
  - but adaptive correctness regressed:
    - `avg_adaptive_keep_count=16.0`
    - `avg_omitted_dynamic_mass=0.4100`
    - `bound_violation_rate=1.0`
    - `avg_adaptive_dynamic_span=0.0`
- Current status of adaptive budget path:
  - the GPU-first direction is still correct for speed
  - but the latest refactor introduced a bookkeeping / selector inconsistency
  - do not use this newest adaptive path for `e192` evaluation until the bound logic is fixed again
- Oracle-compare offset fix:
  - the previously alarming `avg_omitted_dynamic_mass=0.4100` result was a diagnostic bug
  - dense compare had been indexing dynamic tokens with a prefix-only offset even though fullgpu static memory is `prefix + suffix`
  - after fixing the offset:
    - `45498084`
    - `avg_adaptive_keep_count=16.0`
    - `avg_adaptive_mass_bound=0.00425`
    - `avg_omitted_dynamic_mass=0.00425`
    - `bound_violation_rate=0.0`
- Benchmark-specific generated-memory setup:
  - default static split in the wrapper is now `16/32`
  - this avoids the excessive filler padding induced by `128/512`
- Fixed-budget oracle calibration on `16/32`:
  - `e48`: `k≈64` is the first clearly good point
  - `e96`: `k≈128` is clearly better than `k≤64`
  - practical interpretation:
    - adaptive saturation at `400` was overly conservative on this benchmark
    - a useful adaptive policy should land much closer to the `64-128` range here
- Implemented first traversal-time adaptive kernel mode:
  - env:
    - `RETRIEVALATTN_DYNAMIC_BUDGET_MODE=traversal_cuda`
  - fullgpu traversal now receives:
    - attention-space queries
    - attention-space keys
    - static `logZ`
    - unseen-score upper bound
  - kernel returns adaptive keep count / mass bound directly
- First traversal-time validation:
  - `45539711`
  - succeeded end-to-end, but adaptive early stopping is still too weak:
    - `avg_adaptive_keep_count≈73.9`
    - `avg_omitted_dynamic_mass≈0.0945`
    - `avg_adaptive_mass_bound≈0.999`
  - conclusion:
    - traversal-time adaptive plumbing is working
    - the remaining blocker is the looseness of the unseen-score upper bound, not host overhead anymore
- Next experiment:
  - add a moment-prior unseen-tail estimator that uses constant-size attention-key summaries
  - keep the same omitted-mass stopping objective and traversal-time kernel plumbing
  - validate on small compare-on runs before any longer sweep
- First result from that experiment:
  - the moment-prior prototype did reduce candidate/keep counts substantially on `e12`
  - but its omitted-mass estimate was far too optimistic:
    - `avg_adaptive_mass_bound≈0.120`
    - `avg_omitted_dynamic_mass≈0.109`
    - `bound_violation_rate≈65.1%`
  - so this first prior needs a stronger safety correction before it can replace the current worst-case global bound
- Follow-up residual-tail experiment:
  - add one synthetic tail bucket after sparse retrieval:
    - tail mass from the omitted-mass estimate
    - tail value from `mean(V_dynamic)`
  - outcome:
    - slightly better omitted-mass accounting
    - worse dense/sparse output-vector match
    - no accuracy recovery
  - conclusion:
    - probability-mass correction alone is not enough
    - any residual-tail path needs a much better tail-value estimator
- Zero-tail control:
  - use the omitted-mass estimate only as a down-scaling factor on the kept sparse output
  - no synthetic tail value contribution
  - outcome:
    - significantly worse output-vector match than both no-tail and `mean(V_dynamic)` tail
  - conclusion:
    - renormalization error is real, but fixing normalization without modeling the omitted value mixture does not recover quality

## 2026-03-12 update (online decode graph plan extension)
- The custom full-GPU decode backend is now being used for online decode update experiments, not just the original 40k decode AB kernel benchmark.
- Full-GPU runtime changes already implemented:
  - decode device K / attention-K / V caches are full-length and updated online
  - base CSR row pointers are extended to future decode slots
  - an overlay CSR exists for online inserted edges and is read by the full-GPU kernel
- That enables the following experimental mode split on the same `roar_cuda_fullgpu` traversal:
  - `baseline`:
    - no online dynamic range
    - no online graph overlay
  - `dynamic`:
    - dynamic range advances as decode grows
    - generated tokens become retrieval-eligible after they leave the suffix window
    - no new graph edges are inserted
  - `online`:
    - same dynamic range advancement
    - deferred provenance edges inserted into an overlay CSR
- Important conceptual note:
  - `dynamic` is intentionally a seed-exposure baseline, not a fully connected generated-token graph
  - without new edges, aged-out generated tokens are mainly reachable only if seeds land on them
  - this is useful because it separates “mere eligibility” from “added graph connectivity”
- Current short-window validation plan:
  - shrink the suffix window so aged-out generated tokens appear quickly
  - active in-flight jobs:
    - `44964177`
    - `44964183`
    - `44964196`
  - success criteria:
    - `dynamic`: non-zero `aged_gen/head`
    - `online`: non-zero `nodes`, `edges`, `overlay_edges`
    - task quality should remain intact
- Session-steering plan update:
  - for the stock interactive TUI, external safe same-process message injection is not currently available
  - `codex exec resume` is not the right primitive for a live active TUI session
  - app-server protocol does expose the right primitives:
    - `turn/start`
    - `turn/steer`
    - `turn/interrupt`
  - tested successfully:
    - local WebSocket app-server steering
    - Slurm watcher steering an app-managed thread
  - practical recommendation:
    - keep polling for the current stock TUI
    - only pursue same-thread automatic steering if moving to an app-server-managed interactive workflow

## 2026-03-09 status update
- This note is now partially historical.
- A custom CUDA full-GPU decode backend exists and is materially faster than the earlier hybrid/full-ATen traversal attempts on the 40k / 32-step benchmark.
- Latest important measured points:
  - pre-graph-kernel baseline (`44568543`):
    - decode `42.5315 s`
    - retrieve `31.803 s`
  - instrumented stable baseline (`44579772`):
    - decode `43.8746 s`
    - retrieve `31.891 s`
    - kernel hotspot: `merge`, not `score`
  - targeted merge + frontier-slot rewrite (`44580727`):
    - decode `24.9511 s`
    - retrieve `13.318 s`
- The crucial lesson from instrumentation:
  - current full-GPU graph cost is dominated by serial candidate maintenance
  - scoring is not the main bottleneck
  - naive frontier parallelization with atomics was explicitly tested and failed badly on this workload
- Additional 2026-03-11 update:
  - one more kernel-side merge experiment was tried after the successful finalize cleanup:
    - pre-merge threshold filtering against the current kth candidate
    - scratch-buffer candidate merge plus pointer swapping
  - result on `44866961`:
    - decode regressed to `29.7985 s`
    - retrieve stayed flat at `10.573 s`
    - kernel `merge` improved only marginally, while `expand` and `score` got slower
  - conclusion:
    - small pre-merge filtering tricks are not enough
    - the next likely win is to reduce the full-GPU group integration overhead around the stable kernel, not to keep nibbling at the current merge helper

## Updated rule-of-thumb for this backend
- Do:
  - measure with kernel debug on before changing traversal structure
  - optimize `merge/select` first
  - remove serial bookkeeping such as frontier-token -> candidate-slot scans
  - keep the stop rule fixed while doing performance work
- Do not:
  - assume more GPU parallelism is automatically better
  - spend time on scoring kernels when kernel counters say score is only a few percent
  - reintroduce atomics/sync-heavy frontier expansion unless the per-round work size changes materially

## Goal
- Replace the failed Python-controlled `python_gpu` decode experiment with a native GPU decode traversal design that can actually beat CPU `roar_cpp`.

## What Failed
- The preserved `python_gpu` experiment is at branch `exp/decode-python-gpu` (`efc234f`).
- It kept the current Python traversal loop and only moved seed/new-candidate/rerank scoring matmuls to GPU.
- Result on the controlled ~40k prompt:
  - slower than `roar_cpp`
  - slower than the same Python traversal on CPU
- Main reason:
  - too many tiny GPU launches and synchronizations inside the Python frontier loop
  - not enough arithmetic intensity per launch

## Requirements For A Viable GPU Decode Path
- No Python-controlled per-batch frontier loop on the hot path.
- Persistent GPU-resident decode state:
  - query/head-local frontier
  - visited mask or visited stamp array
  - candidate/top-k buffers
  - CSR graph tensors on GPU
  - decode key cache on GPU
- Batched scoring over many nodes at once.
- Batched neighbor expansion over many frontier nodes at once.
- Minimal CPU<->GPU synchronization during a decode step.

## Proposed Architecture
1. Graph residency
- Keep CSR graph on GPU:
  - `offsets` as `uint32`
  - `neighbors` as `int32`
- Upload once after prefill/index build.

2. Key residency
- Keep decode keys on GPU for the full prefill context.
- Only keys are needed for traversal scoring; values can remain on the current path initially.

3. Decode frontier kernel model
- One decode step should batch work across heads.
- Per head:
  - initialize frontier from graph-only seeds
  - expand a bounded number of frontier nodes
  - gather all neighbors for those nodes
  - filter visited / static / out-of-range on GPU
  - score the surviving candidate pool on GPU
  - update frontier and top candidate list on GPU
- Return only the final retrieved token ids needed for attention gather.

4. Stop rule
- Mirror current semantics as closely as possible:
  - `min_visits`
  - `max_visits`
  - `candidate_target`
  - stability-gap style stop
- But compute the stop metrics on GPU and only sync summary state once per decode step.

5. Implementation path
- Phase 1:
  - C++/CUDA extension for GPU decode traversal
  - keep existing `roar_cpp` CPU backend unchanged
  - add new backend name, for example `roar_cuda`
- Phase 2:
  - batched multi-head expansion/scoring kernel
  - optional weighted traversal if edge weights become useful

## First Milestone
- Do not try to fuse traversal into attention first.
- First milestone is a standalone GPU decode traversal backend that beats Python CPU on the same traversal policy.
- Second milestone is to challenge `roar_cpp`.
- Status:
  - achieved
  - current `roar_cuda` backend beats Python CPU on both ~9k and ~40k decode benchmarks
  - remaining milestone is to beat `roar_cpp`

## Suggested A/B Ladder
1. `python` vs native GPU backend on the same traversal policy
- This isolates backend value without algorithm confounds.

2. native GPU backend vs `roar_cpp`
- This is the real production comparison.

3. native GPU backend with larger batched frontier widths
- Check whether more batching is enough to cross the CPU line.

## Success Criteria
- Must beat Python CPU traversal clearly on the same policy.
- Must not degrade retrieval quality relative to current decode-space reference.
- Must show lower `graph` time than `roar_cpp` on the controlled ~40k decode benchmark before scaling up.

## Current Result
- `roar_cuda` already beats Python CPU on the same traversal policy.
- It does not yet beat `roar_cpp`.
- The remaining gap is concentrated in the `graph` slice, not rerank.

## `roar_cuda_v2` Result
- `roar_cuda_v2` changes that conclusion:
  - the graph slice is now effectively solved for the current 40k benchmark
  - `roar_cuda_v2` slightly beats `roar_cpp` on graph stage
  - overall decode still trails `roar_cpp` because seed handling is now the dominant remaining gap

## Grouped GPU Seed Scoring Result
- After moving grouped seed scoring to GPU, `roar_cuda_v2` now beats `roar_cpp` on the controlled 40k A40 benchmark.
- Current best numbers:
  - `roar_cpp`: decode `54.0159 s`, seed `10.088 s`, graph `21.164 s`
  - `roar_cuda_v2`: decode `46.0405 s`, seed `7.062 s`, graph `18.868 s`
- Therefore the current optimization focus changes from “beat `roar_cpp`” to:
  - validate on larger / harder decode settings
  - clean up grouped profile accounting
  - decide whether `roar_cuda_v2` is robust enough to become preferred over `roar_cpp`
- Therefore `v2` should focus next on:
  - grouped seed generation / scoring
  - only after that revisit more frontier-side work

## Validation Status
- Longer decode validation passed:
  - ~40k prompt, `GEN_LEN=100`
  - `roar_cuda_v2` still faster than `roar_cpp`
- Larger-context validation passed:
  - ~65k prompt, `GEN_LEN=32`
  - `roar_cuda_v2` still faster than `roar_cpp`

## Current Decode Backend Map (2026-03-07)
- Best current backend:
  - `roar_cuda_v2`
  - controlled ~40k prompt, `GEN_LEN=32`:
    - `decode=46.0405 s`
    - `seed=7.062 s`
    - `graph=18.868 s`
  - controlled ~40k prompt, `GEN_LEN=100`:
    - `decode=146.8723 s`
  - ~65k prompt, `GEN_LEN=32`:
    - `decode=46.2675 s`
- Competitive experimental backend:
  - hybrid explicit-beam `roar_cuda_beam`
  - ~40k / `GEN_LEN=32`:
    - `decode=46.2885 s`
    - `seed=7.076 s`
    - `graph=19.666 s`
  - ~40k / `GEN_LEN=100`:
    - `decode=148.3664 s`
- Failed decode traversal paths:
  - Python-controlled GPU traversal (`python_gpu`): too many tiny launches / syncs
  - dense score-table beam:
    - ~40k / `GEN_LEN=32`: `decode=178.5231 s`
  - “full GPU” beam with more device-resident state:
    - ~40k / `GEN_LEN=32`: `decode=207.6486 s`
  - `roar_cuda_frontier` small-buffer GPU frontier:
    - ~40k / `GEN_LEN=32`: `decode=232.8459 s`

## What The Recent Failures Actually Mean
- They do **not** prove that GPU traversal is a dead end.
- They do show that expressing irregular graph traversal as a sequence of generic ATen ops is the wrong path.
- The full-GPU attempts were slow because they still spent most of their time in:
  - GPU CSR gather / compaction over ragged rows,
  - repeated `sort` / `masked_select` / `index_select` / `topk`,
  - small-buffer bookkeeping split across many kernels,
  - repeated synchronization back to host for stop logic / counters.
- In other words:
  - the problem is not “frontier on GPU” conceptually,
  - the problem is “frontier on GPU implemented as many generic tensor ops”.

## Updated Next Step
Move to **measured optimizations around the stable custom kernel**, not more speculative merge-side micro-optimizations.

### Required properties
- Keep q-head retrieval objective unchanged.
- Keep current grouped GPU seed scoring path.
- Keep small-buffer traversal only:
  - no dense `[q_count, num_tokens]` score tables.
- No CPU synchronization inside a decode step except final result extraction.
- Preserve the current stable full-GPU kernel unless a new kernel change is justified by measurements.

### Kernel breakdown to implement
1. `expand_frontier_kernel`
- Input:
  - CSR `offsets`, `neighbors`
  - current frontier ids/counts
  - visited bitset
  - dynamic range
- Output:
  - compact neighbor list per query/group
  - updated visited marks
- Must do dedup / visited marking on device.

2. `score_neighbors_kernel` or grouped GEMM + custom packing
- Score compact neighbor buffers against grouped queries.
- Avoid per-round high-level gather/scatter churn.

3. `merge_frontier_candidates_kernel`
- Merge scored neighbors into:
  - next frontier buffer (`frontier_width`)
  - candidate buffer (`candidate_target`)
- This is the key missing piece in current experiments.
- Must not sort full candidate lists on host every round.

4. `stop_metrics_kernel`
- Compute:
  - current top-`token_budget` threshold
  - frontier best
  - stability step updates
- Only return tiny summary tensors to host if absolutely necessary.

5. Final rerank
- Can stay as grouped GPU matmul over final candidate union initially.
- This part is not the bottleneck.

## Backend Recommendation
- Keep `roar_cuda_v2` as the preferred decode backend until a custom-kernel traversal wins.
- Treat `roar_cuda_beam` and `roar_cuda_frontier` as experimental evidence, not promotion candidates.
- If resuming implementation, start from the current grouped seed path and replace only the traversal core.
- For the current full-GPU branch specifically:
  - stable kernel baseline is `44581886`
  - next optimization target is the KV-group integration path:
    - avoid per-head payload dict creation
    - avoid restacking already-batched device IDs and masks
    - keep attention inputs batched per KV-group after search

## 2026-03-07 custom-kernel follow-up
- We tried that next step.
- Two custom-kernel decode iterations were run through a separate split CUDA extension:
  - iteration 1:
    - custom frontier expansion + custom per-neighbor scoring + custom merge
    - ~40k A40 result: `decode=79.7116 s`, `graph=56.450 s`
  - iteration 2:
    - custom frontier expansion + grouped GPU matmul scoring + custom merge
    - ~40k A40 result: `decode=130.0818 s`, `graph=101.482 s`
- matched `roar_cuda_v2` reference on the same setup:
  - `decode=49.3138 s`, `graph=21.514 s`
- What this means:
  - “some custom kernels” are not enough
  - iteration 1 failed because direct scalar per-neighbor scoring was too expensive
  - iteration 2 failed because the ATen-built compact union / row-mask path around scoring was too expensive
- Updated recommendation:
  - stop the current `roar_cuda_kernel` branch
  - keep `roar_cuda_v2` as the preferred decode backend
  - if custom traversal is revisited later, it must natively implement the full round core:
    - frontier expansion
    - dedup / compact union build
    - score staging
    - frontier/candidate merge

## 2026-03-07 custom-kernel result
- Implemented an experimental backend:
  - `RETRIEVALATTN_DECODE_BACKEND=roar_cuda_kernel`
  - separate CUDA extension:
    - `roargraph_cuda_kernel_ext`
- Result:
  - first custom-kernel version:
    - 40k decode `85.6461 s`
    - `graph=58.753 s`
  - second version with frontier-token / warp-level expansion:
    - 40k decode `127.4251 s`
    - `graph=99.428 s`
  - `roar_cuda_v2` reference on same workload:
    - 40k decode `48.5557 s`
    - `graph=21.075 s`
- Conclusion:
  - “custom CUDA traversal” alone is not enough
  - the losing design choice was direct per-neighbor score computation inside the traversal kernel
  - the grouped matmul-based scoring path still dominates for this workload
- If revisiting custom kernels later:
  - keep custom kernels for expansion / seen marking / merge only
  - do **not** keep direct neighbor-by-neighbor scoring in the hot traversal kernel
  - any future kernel path should preserve grouped score evaluation (or something equivalently GEMM-friendly)

## 2026-03-13 online decode graph checkpoint
- Fullgpu online metrics are now trustworthy:
  - `overlay_edges` is counted in the fullgpu kernel/profile path
  - `aged_gen/head` is computed from the final returned token IDs
- Verified on the small-window generated-memory smoke:
  - `dynamic` already retrieves some aged generated tokens:
    - `aged_gen/head=2.98`
  - `online` retrieves many more and actively traverses overlay edges:
    - `overlay_edges=102490740`
    - `aged_gen/head=35.02`
- So the online provenance mechanism is real; the remaining question is cost/amortization, not correctness of the basic path.

## 2026-03-13 new comparison baselines
- Added `growing_static`:
  - RetrievalAttention/fullgpu-only comparison mode
  - generated tokens remain in static attention
  - retrieval dynamic region stays frozen to prompt-only span
- Added `full_dense`:
  - true dense baseline via `Full_Flash_Attn`
- Immediate finding:
  - `growing_static` is not a quality upper bound for this task; it failed the answer-format smoke.
  - dense baseline is both valid and much faster at the tested lengths.

## 2026-03-13 partial length-sweep result
- Sweep root:
  - `generated_memory_eval_result/length_sweep_s16_fullgpu_vs_dense`
- Partial trend with `online e24/e48` still running:
  - `baseline`
    - quality collapses at all tested lengths
  - `dynamic`
    - mild help at the shortest case only
  - `online`
    - strong quality at `e12`
    - large latency cost
  - `full_dense`
    - `e12`: `avg_decode_sec~6.15s`, `query_acc=1.0`
    - `e24`: `avg_decode_sec~8.28s`, `query_acc=1.0`
    - `e48`: `avg_decode_sec~12.98s`, `query_acc=0.667`
- Updated interpretation:
  - at short/medium decode lengths, sparse RetrievalAttention is nowhere near dense attention on latency.
  - if online sparse decode is going to win, it likely has to be at much longer decode lengths than `12/24/48` entries.

## 2026-03-13 online update priority shift
- The completed online sweep makes the bottleneck clear:
  - `online e24`: `update=100.682s`, `graph=31.694s`
  - `online e48`: `update=234.919s`, `graph=55.133s`
- So the next priority is online update implementation efficiency, not more graph-kernel tuning.

## 2026-03-13 persistent overlay result
- Added update sub-buckets:
  - `d2h`
  - `build`
  - `h2d`
- Pre-fix measurement (`45079412`) showed:
  - `update=56.989s`
  - `d2h=7.545s`
  - `build=54.533s`
  - `h2d=1.265s`
- Implemented a persistent row-wise fullgpu overlay:
  - overlay represented as per-row count + fixed-cap row neighbors
  - flush updates only dirty rows on device
  - no full overlay CSR rebuild on every step
- Verification (`45081113`) showed:
  - `update=16.255s`
  - `build=14.716s`
  - `avg_decode_sec=125.531s`
  - retrieval behavior unchanged
- Updated recommendation:
  1. keep the persistent row-wise overlay path
  2. next optimize provenance extraction / Python-side bookkeeping
  3. only then revisit deeper graph-kernel changes

## 2026-03-13 update follow-up status
- Two “safe” follow-up attempts after `45081113` did not survive validation:
  - KV-group provenance merge changed retrieval behavior (`45085490`)
  - staging-buffer reuse + narrower D2H alone preserved behavior but regressed latency (`45090954`)
- So the active baseline should remain the `45081113` version.
- Current safe next-direction ideas should avoid:
  - changing provenance merge semantics
  - extra host/device staging copies that duplicate `index_copy_` work

## 2026-03-15 updated safe baseline
- New safe improvements validated in `45248579`:
  - removed duplicate D2H for provenance by reusing CPU `final_tokens`
  - removed row repacking during flush by using the incrementally maintained CPU overlay row mirror
- This improved the short online case while preserving behavior:
  - `update 15.881s -> 13.201s`
  - `group 9.793s -> 4.553s`
  - `avg_decode_sec 127.324s -> 117.729s`
- Updated recommendation:
  1. treat `45248579` as the current best safe online baseline
  2. rerun focused scaling from this version
  3. only after that, consider deeper constant-factor work in dynamic/fullgpu retrieval

## 2026-03-15 post-rerun conclusion
- Focused `online` vs `full_dense` rerun from the improved baseline still shows no crossover in the tested range:
  - `online e192`: `886.760s`
  - `full_dense e192`: `43.602s`
- So the latest constant-factor wins are real but insufficient.
- Updated implication:
  - the asymptotic story may still be acceptable at the per-head graph/update level;
  - but implementation constants remain far too large for sparse decode to be competitive in the current regime.
- Before attempting extreme lengths (`100k+` decode), the next round of work should focus on more aggressive but still defensible constant-factor reductions, or on a smaller targeted prototype that isolates the online-update machinery without the rest of the benchmark/task noise.
