# Parallel GPU Optimization Agents - 2026-05-23

Base commit: `dbeeb96` (`base frontier gpu optimization state`)

All worktrees are under `worktrees/` and are ignored by Git. Each worktree has local `.venv` and `.hf_cache` symlinks to the main checkout.

## Active Workers

| strategy | worktree | branch | agent | status |
| --- | --- | --- | --- | --- |
| no-exact-fill score grid | `worktrees/opt-nofill-score-grid` | `codex/opt-nofill-score-grid` | Curie `019e56da-c399-7421-a6de-ffe1aa944d16` | validation pending; jobs `50736497`-`50736502` |
| specialized top-k rank-prefix kernel | `worktrees/opt-rank-prefix-topk` | `codex/opt-rank-prefix-topk` | Huygens `019e56da-c41a-79c3-9090-b79ce29c7f75` | validation pending; jobs `50736821`, `50736831`, `50736844`, `50736847`, `50736856`, `50736857` |
| fuse PQ scoring + top-k | `worktrees/opt-pq-score-topk-fusion` | `codex/opt-pq-score-topk-fusion` | Poincare `019e56da-c4be-75c2-ad53-ef889d1d2f77` | validation pending; jobs `50736996`, `50737027`, `50737104`, `50737139`, `50737197`, `50737198` |
| fused exact-logit + mixed-score construction | `worktrees/opt-exact-mixed-fusion` | `codex/opt-exact-mixed-fusion` | Ramanujan `019e56da-c7fb-7b32-bbe9-d22fb6a8f089` | validation pending/incomplete; jobs `50737392`, `50737394`, `50737407`, `50737408`; LongGen not submitted |
| persistent V-PQ sealed-page append | `worktrees/opt-vpq-sealed-append` | `codex/opt-vpq-sealed-append` | Avicenna `019e56da-cd9b-7b11-aaae-af09f30cb4c1` | validation pending; jobs `50737140`-`50737145` |
| fused residual-risk + policy selection | `worktrees/opt-risk-policy-fusion` | `codex/opt-risk-policy-fusion` | Copernicus `019e56da-d619-7b70-8b60-53afada273ea` | validation pending; jobs `50736562`, `50736563`, `50736568`, `50736578`, `50736581`, `50736596` |
| native grouped execution across heads/layers | `worktrees/opt-grouped-native-exec` | `codex/opt-grouped-native-exec` | Wegener `019e56eb-5e79-79a0-9433-b3b4cfc6d435` | validation pending; jobs `50737670`, `50737717`, `50737718`, `50737719`, `50737720`, `50737721` |
| allocation/workspace reuse | `worktrees/opt-workspace-reuse` | `codex/opt-workspace-reuse` | Carver `019e56ee-6e7d-7c11-9c6f-c740e83c1c8e` | validation pending; jobs `50737348`, `50737349`, `50737365`, `50737375`, `50737376`, `50737393` |
| custom V-PQ base aggregation by code histograms | `worktrees/opt-vpq-histogram-base` | `codex/opt-vpq-histogram-base` | Aquinas `019e56f8-a114-7c12-9ec4-aaa74f2f1dd2` | validation pending; jobs `50737780`, `50737806`, `50737831`, `50737843`, `50737847`, `50737858`, `50737868` |

## Pending Workers

All requested strategy workers have been launched. Some workers have already handed off with validation pending while their Slurm jobs wait in queue.

## Shared Contract

- Preserve CPU frontier semantics exactly.
- Add diagnostic flags first; do not promote into canonical defaults inside candidate branches.
- Run validation jobs from the assigned worktree, not the main checkout. The repository Slurm scripts hardcode the main path, so each worker must patch/wrap validation scripts locally before submitting.
- Use Slurm outside sandbox with `spgpu` and account `zhengya98` for builds, GPU tests, and benchmark runs.
- Promotion requires CUDA unit tests, long saved-trace parity over `32000,64000,128000` and heads `0,8`, RULER 32k/128 timing and accounting, and sustained LongGen timing if the candidate affects long decode.
- Do not use oracle mass, dense rankings, fixed-budget replacement, selected-mass V, hidden dense reads, or benchmark-specific knobs.

## Monitor Results

Monitor pass completed after submitted Slurm jobs reached terminal state. No candidate is promotable from this batch.

| strategy | CUDA unit | parity | RULER 32k/128 result | LongGen result | decision |
| --- | --- | --- | --- | --- | --- |
| no-exact-fill score grid | pass `50736497`, `5:01` | failed before parity: missing worktree-local trace | timing `44.68s`; accounting `49.24s`, `3.8361` logical MB/hq, `8.9179` physical MB/hq, selected `11730.29` | failed before eval: missing worktree-local LongGenBench dataset | not promotable |
| fused residual-risk + policy selection | failed immediately: `mkdir candidate_eval_result: Permission denied` | failed immediately: missing `.venv/bin/activate` | failed immediately: missing wrapper path | failed immediately: HF cache under Slurm spool | not promotable |
| specialized top-k rank-prefix kernel | pass `50736821`, `4:26` | failed before parity: missing worktree-local trace | timing `44.12s`; accounting `46.30s`, `3.8361` logical MB/hq, `8.9179` physical MB/hq, selected `11730.29` | failed before eval: missing worktree-local LongGenBench dataset | not promotable |
| fuse PQ scoring + top-k | pass `50736996`, `4:29` | failed before parity: missing worktree-local trace | timing `39.15s`; accounting `42.43s`, `3.83597` logical MB/hq, `8.91801` physical MB/hq, selected `11729.49` | failed before eval: missing worktree-local LongGenBench dataset | not promotable |
| fused exact-logit + mixed-score construction | pass `50737392`, `4:18` | failed before parity: missing worktree-local trace | failed in HF path: ranked prefixes do not cover every K take count | not submitted | not promotable |
| persistent V-PQ sealed-page append | pass `50737140`, `4:23` | failed before parity: missing worktree-local trace | no-stats failed import; accounting `49.63s`, `3.8361` logical MB/hq, `8.9179` physical MB/hq, selected `11730.34` | failed before eval: missing worktree-local LongGenBench dataset | not promotable |
| native grouped execution across heads/layers | pass `50737670`, `4:18` | failed before parity: missing worktree-local trace | timing `56.97s`; accounting `67.32s`, `3.8365` logical MB/hq, `8.9179` physical MB/hq, selected `11732.11` | failed before eval: missing worktree-local LongGenBench dataset | not promotable |
| allocation/workspace reuse | failed immediately: `mkdir cuda_unit_result: Permission denied` | failed immediately: missing `.venv/bin/activate` | dependent jobs canceled | dependent jobs canceled | not promotable |
| custom V-PQ base aggregation by code histograms | failed immediately: `mkdir cuda_unit_result: Permission denied` | dependent jobs canceled | dependent jobs canceled | dependent jobs canceled | not promotable |

The repeated failure mode is validation plumbing, not necessarily algorithm semantics: several worktree jobs used worktree-relative trace/dataset/cache/output paths or derived `REPO_ROOT` from Slurm's spool copy of the submitted script. Future worktree validation should either symlink shared traces/datasets/caches into each worktree or pass absolute main-checkout paths explicitly, and wrappers should prefer `SLURM_SUBMIT_DIR`/an explicit `FRONTIER_WORKTREE` over `${BASH_SOURCE[0]}` under `sbatch`.

## Fixed-Path Reruns

Rerun script: `scripts/submit_fixed_candidate_validation_reruns.sh`.

The rerun passes absolute trace, X-trace, HF cache, venv, model snapshot, and LongGenBench paths into the candidate worktrees. This fixes the earlier validation-plumbing failures for the three candidates with plausible short-RULER speedups.

| strategy | fixed jobs | status | current interpretation |
| --- | --- | --- | --- |
| fuse PQ scoring + top-k | parity `50739154`, LongGen 8192 `50739155`, LongGen 16384 `50739156` | parity passed; LongGen 8192 and 16384 completed | Not promotable: semantics are clean, but 8192 accounting is slower than canonical and 16k no-stats is only a marginal/inconsistent `~1%` timing improvement. |
| specialized top-k rank-prefix kernel | parity `50739157`/`50740047`, LongGen 8192 `50739158`, LongGen 16384 `50739159` | parity passed; LongGen 8192 and 16384 completed | Not promotable: semantics are clean, but sustained LongGen is slower than canonical at both 8192 and 16384. |
| no-exact-fill score grid | parity `50740123`, LongGen 8192 `50740139`, LongGen 16384 `50740140` | parity passed; LongGen 8192 and 16384 completed | Not promotable: semantics are clean, but RULER and LongGen are slower than canonical at both sustained lengths. |

Low-memory parity-only duplicates were submitted to reduce queue pressure while leaving the original jobs queued: rank-prefix `50740047`, no-fill `50740048`. These only replace the parity gate if they complete cleanly; they do not replace the LongGen sustained-decode gates. Rank-prefix `50740047` passed. No-fill `50740048` failed due a wrapper variable mismatch (`RETRIEVAL_ATTENTION_WORKTREE` vs `RETRIEVAL_ATTENTION_ROOT`), so the worktree wrapper was patched and resubmitted as no-fill parity `50740123`. The stale original rank/no-fill parity jobs and stale no-fill LongGen jobs were canceled; fixed no-fill LongGen jobs were resubmitted as `50740139` and `50740140`.

Completed fixed evidence:

- PQ-score/top-k parity `50739154`: passed over decode lengths `32000,64000,128000`, heads `0,8`, `rows=12`, max CPU/native attention relL2 `4.25e-09`, max CPU/native o-proj relL2 `1.70e-08`, max Torch/GPU-policy attention relL2 `2.62e-06`, max Torch/GPU-policy o-proj relL2 `2.14e-06`.
- PQ-score/top-k LongGen 8192 accounting `50739155`: generated `8192` tokens in `836.70s`; logical step `1.7620 MB/head-query`; physical step `1.8716 MB/head-query`; selected `4281.05` tokens/head-query; score-grid wall time `74.08s`; patched attention wall time `570.85s`.
- PQ-score/top-k LongGen 16384 timing `50739156`: generated `16384` tokens in `2623.21s` with cost stats disabled. This is only marginally faster than the current canonical 16k no-stats reference (`50723400`, `2650.05s`) and does not offset the slower 8192 accounting result.
- Rank-prefix parity `50740047`: passed over decode lengths `32000,64000,128000`, heads `0,8`, `rows=12`, max CPU/native attention relL2 `4.25e-09`, max CPU/native o-proj relL2 `1.70e-08`, max Torch/GPU-policy attention relL2 `2.62e-06`, max Torch/GPU-policy o-proj relL2 `2.14e-06`.
- Rank-prefix LongGen 8192 accounting `50739158`: generated `8192` tokens in `861.99s`; logical step `1.7620 MB/head-query`; physical step `1.8716 MB/head-query`; selected `4281.09` tokens/head-query; rank-prefix wall time `28.99s`; score-grid wall time `78.46s`; patched attention wall time `596.39s`. This is slower than PQ-score/top-k `50739155` and the current canonical accounting baseline (`50723383`, `842.58s`).
- Rank-prefix LongGen 16384 timing `50739159`: generated `16384` tokens in `2817.99s` with cost stats disabled. This is slower than both PQ-score/top-k `50739156` (`2623.21s`) and the current canonical no-stats reference (`50723400`, `2650.05s`).
- No-fill parity `50740123`: passed over decode lengths `32000,64000,128000`, heads `0,8`, `rows=12`, max CPU/native attention relL2 `4.25e-09`, max CPU/native o-proj relL2 `1.70e-08`, max Torch/GPU-policy attention relL2 `2.62e-06`, max Torch/GPU-policy o-proj relL2 `2.14e-06`.
- No-fill LongGen 8192 accounting `50740139`: generated `8192` tokens in `857.96s`; logical step `1.7620 MB/head-query`; physical step `1.8716 MB/head-query`; selected `4281.02` tokens/head-query; score-grid wall time `73.33s`; patched attention wall time `595.23s`. This is slower than the current canonical accounting baseline (`50723383`, `842.58s`).
- No-fill LongGen 16384 timing `50740140`: generated `16384` tokens in `2908.61s` with cost stats disabled. This is slower than PQ-score/top-k `50739156` (`2623.21s`), rank-prefix `50739159` (`2817.99s`), and the current canonical no-stats reference (`50723400`, `2650.05s`).

Final fixed-path conclusion: none of the three rerun candidates is promotable. The earlier "no candidate promotable" statement was partly caused by validation setup failures, but after fixing the setup the runtime evidence still rejects these candidates. PQ-score/top-k is the closest candidate, with clean parity and a marginal 16k no-stats timing improvement, but it loses on 8192 accounting and the 16k margin is too small/inconsistent to merge.
