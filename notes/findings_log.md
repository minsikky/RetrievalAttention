# Findings Log

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

### Reference logs
- Baseline quality reference: `simple_test.out`
- Regression example: `slurm-41814588.out`
- Transfer profiling: `slurm-41810674.out`, `slurm-41811894.out`
- Decode crash fixed: `slurm-41844519.out`
- Latest successful but lower-quality run: `slurm-41844983.out`
- Custom fused blocked/stall evidence: `slurm-41850330.out`, `slurm-41850375.out`, `slurm-41850455.out`, `slurm-41850475.out`
- Fused-prefill build success: `slurm-42153021.out`
- Native fused smoke pass: `slurm-42274991.out`
