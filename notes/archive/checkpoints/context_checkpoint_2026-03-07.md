# Context Checkpoint 2026-03-07

## Current Decode Status
- Preferred decode backend remains `roar_cuda_v2`.
- Best measured results:
  - ~40k prompt, `GEN_LEN=32`
    - `roar_cpp`: `54.0159 s`
    - `roar_cuda_v2`: `46.0405 s`
  - ~40k prompt, `GEN_LEN=100`
    - `roar_cpp`: `177.2075 s`
    - `roar_cuda_v2`: `146.8723 s`
  - ~65k prompt, `GEN_LEN=32`
    - `roar_cpp`: `56.0003 s`
    - `roar_cuda_v2`: `46.2675 s`

## Experimental Decode Backends
- Hybrid explicit-beam `roar_cuda_beam`
  - competitive, but still slightly behind `roar_cuda_v2`
  - ~40k / `GEN_LEN=32`: `46.2885 s`
  - ~40k / `GEN_LEN=100`: `148.3664 s`
- Failed variants:
  - dense score-table beam:
    - `slurm-dec40-cuda-beam.out`
    - `178.5231 s`
  - more device-resident full-GPU beam:
    - `slurm-dec40-cuda-beam-gpu.out`
    - `207.6486 s`
- `roar_cuda_frontier` full-GPU frontier:
    - `slurm-dec40-cuda-frontier2.out`
    - `232.8459 s`
- `roar_cuda_kernel` custom CUDA traversal:
  - iteration 1:
    - `slurm-44484124.out`
    - `85.6461 s`
  - iteration 2 (frontier-token / warp-level expansion):
    - `slurm-44484265.out`
    - `127.4251 s`

## Why The Full-GPU Attempts Were Slow
- They were still built out of generic ATen ops over irregular graph traversal.
- Main costs were:
  - ragged CSR gather / compaction on GPU
  - repeated `sort`, `masked_select`, `index_select`, `topk`
  - per-round small-kernel launch chains
  - host synchronization for counts / stop logic in some versions
- Conclusion:
  - “GPU traversal” is not disproven
  - “GPU traversal implemented as many ATen tensor ops” is the bad path

## Next Recommended Path
- Move to custom CUDA kernels for traversal.
- Keep the good parts:
  - grouped GPU seed scoring from `roar_cuda_v2`
  - q-head retrieval objective
  - small frontier/candidate buffers
- Replace only traversal core with custom kernels:
  1. frontier expansion + visited marking
  2. compact neighbor scoring
  3. frontier/candidate merge
  4. stop-metric update
- Do not build another dense full-token-space traversal.

## 2026-03-07 late update
- We tried the custom-kernel traversal path after this checkpoint.
- Result:
  - iteration 1 (`44484147`): `decode=79.7116 s`, `graph=56.450 s`
  - iteration 2 (`44484297`): `decode=130.0818 s`, `graph=101.482 s`
  - matched `roar_cuda_v2` reference (`44484146`): `decode=49.3138 s`, `graph=21.514 s`
- Conclusion:
  - current custom-kernel traversal path is not competitive
  - `roar_cuda_v2` remains the decode backend to keep
  - if custom traversal is attempted again, it must own the full round core natively instead of mixing custom expansion with ATen union/mask/scoring glue

## 2026-03-07 addendum
- We did implement a first custom-kernel traversal path and benchmark it.
- Result:
  - custom kernels with direct per-neighbor scoring are slower than `roar_cuda_v2`
  - the attempted occupancy rewrite made them even slower
- Updated recommendation:
  - if revisiting custom kernels, avoid direct per-neighbor score computation inside traversal
  - preserve grouped score evaluation / GEMM-friendly scoring, and only custom-kernel the expansion / dedup / merge pieces

## Relevant Files
- Parent repo:
  - `cache_hub/retrievalattention_cache.py`
  - `cache_hub/roargraph_cpp_backend.py`
- RoarGraph subrepo:
  - `third_party/RoarGraph/python_ext/roargraph_torch_ext.cpp`
- Notes:
  - `notes/current_status.md`
  - `notes/findings_log.md`
  - `notes/native_decode_gpu_plan.md`

## Current Worktree State
- Parent repo is dirty:
  - `cache_hub/retrievalattention_cache.py`
  - `cache_hub/roargraph_cpp_backend.py`
  - `notes/current_status.md`
  - `notes/findings_log.md`
  - `notes/native_decode_gpu_plan.md`
- RoarGraph subrepo is dirty:
  - `third_party/RoarGraph/python_ext/roargraph_torch_ext.cpp`
  - built `.so` files are untracked
- The active source state still includes:
  - `roar_cuda_beam`
  - `roar_cuda_frontier`
- Neither experimental backend is committed yet in this turn.

## Important Logs
- `slurm-dec40-cuda-v2-s2.out`
- `slurm-dec40-cuda-v2-g100.out`
- `slurm-dec64-cuda-v2-g32.out`
- `slurm-dec40-cuda-beam-v2.out`
- `slurm-dec40-cuda-beam-g100.out`
- `slurm-dec40-cuda-beam-gpu.out`
- `slurm-dec40-cuda-frontier2.out`
