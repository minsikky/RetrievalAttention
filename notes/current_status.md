# Current Status

## Scope

- Repository: RetrievalAttention / RetroInfer long-context attention experiments.
- Current phase: make the frontier algorithm benchmark-ready, not open-ended selector search.
- Target evaluation style: dense prefill plus decode-only approximation.
- Target model family for benchmark work: Qwen3-8B / Llama-3.1-8B scale models that fit on A40-class GPUs.

## Active Algorithm

Canonical frontier path:

- Dense prefill.
- Decode-only paged-PQ selector sidecars.
- Online geometric confidence/budgeting.
- Selected exact K path with selected/mixed V handling.
- V-PQ tail estimation.
- Logical frontier MB accounting separated from physical GPU simulator MB.

`FRONTIER_CANONICAL_GPU=1` is the benchmark-facing guard. Noncanonical fixed-budget or selector-rank fast paths are diagnostic only.

## Latest Useful Results

- Canonical RULER 32k, all layers, 32 decode tokens: job `50481630`, score `100.0`, mean selected `15333`, logical step `11.948 MB/head-query`, decode `69.14s`.
- Canonical RULER 32k, all layers, 128 decode tokens: job `50481532`, score `100.0`, mean selected `15209`, logical step `11.933 MB/head-query`, decode `255.13s`.
- Corrected accounting removed exact-K double counting when exact ranked logits are already computed for confidence/final softmax.
- Fused final-output reuse path is functionally valid but slower than the current canonical path; keep it opt-in only.

## Current Blockers

- Canonical GPU simulator is semantically closer to the CPU-frontier algorithm, but still slow for broad RULER/LongBench validation.
- The main runtime target is exact-logit/confidence/output construction, not more selector algorithm exploration.
- Need Slurm validation for:
  - `scripts/run_frontier_cuda_unit_tests.sh`
  - `scripts/run_exact_logit_backend_bench.sh`
  - profiled RULER smoke with `PROFILE_NATIVE_OPS=1`
  - small LongBench-v2 smoke with `PROFILE_NATIVE_OPS=1`

## Historical Source

The full pre-cleanup append-only status file is preserved at `notes/archive/status_history/current_status_2026-05-20_full.md`.
