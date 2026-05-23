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
