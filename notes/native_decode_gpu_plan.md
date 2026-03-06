# Native GPU Decode Traversal Plan

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
