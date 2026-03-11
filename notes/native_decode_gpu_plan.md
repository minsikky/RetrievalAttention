# Native GPU Decode Traversal Plan

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
