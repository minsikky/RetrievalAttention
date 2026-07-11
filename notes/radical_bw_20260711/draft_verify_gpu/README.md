# Draft-then-verify token acceptance (end-to-end GPU)

Go/no-go gate the CPU study (experiment D) could not provide: does a model
decoding with cheap one-shot attention selection at **all** layers produce the
same greedy tokens as the frozen algorithm, once cross-layer error compounding
and query drift are in play?

## What "draft mode" is

Env flag `SELECTOR_PQ_JOINT_DRAFT_MODE={off,start1,start2}` in the joint-KV
policy controller (`benchmark/selector_eval/runners/hf_paged_pq_intervention_joint_policy.py`,
`select_joint_kv_budgets`).

- `off` (default): unchanged. The controller computes a proxy-mass-derived start
  rung, then runs the stability escalation walk. **Byte-identical** to the frozen
  algorithm (the added code path is never entered).
- `start1` / `start2`: **skip the escalation walk entirely**. Pin each head's
  K-budget rung to `start_rung + 1` (start1) or `start_rung + 2` (start2),
  clamped to the ladder, then apply the frozen `v_target` rule
  (`v = max(v_budgets[0], 0.25*k_target)`) to that rung. Top tokens are selected
  by the existing risk ranking; the fused finalize path reconstructs the output
  for the one-shot budget. Over-provisioned drafts are fine here — we measure
  token acceptance, not bytes.

Prior CPU evidence (layer-16 only, K-recall vs the walk): start+1 ≈ 0.916,
start+2 ≈ 0.996 at ~1.9x bytes. This experiment tests those modes end-to-end.

## Runs (two partition-agnostic BUNDLE jobs)

Token comparisons are contaminated by cross-card FP differences, but the
card-class-consistency constraint only binds WITHIN a comparison. So each
context length runs as one BUNDLE job whose arms execute sequentially inside a
single allocation — every arm shares the exact same GPU by construction
(pattern: `run_fspq_tokpar_rate_probe.sbatch`'s run_arm loop). `nvidia-smi -L`
is echoed at bundle start so the card class is recorded in the log. Arms
compare only within their own bundle; no cached run from any other card class
is used. Each arm greedy-decodes; only the selection differs between arms.

Driver: `scripts/run_draft_verify_bundle.sbatch` (BUNDLE=32k|128k) — the tokpar
identity-gate env (`run_fspq_tokpar_identity_gate.sbatch`) minus the comparison
epilogue, plus `SELECTOR_PQ_JOINT_DRAFT_MODE` per arm. `FUSED_VPREFIX_TOKPAR=0`
(as the gate) so control flows through `select_joint_kv_budgets` (the draft
chokepoint) rather than the tokpar-batched deferral branch. The single-arm
driver `scripts/run_draft_verify_one.sbatch` is retained for reruns.

| bundle | task            | n | max_new | arms (sequential)     | partitions | job |
|--------|-----------------|---|---------|-----------------------|------------|-----|
| 32k    | qa_1            | 4 | default | off → start1 → start2 | gpu_mig40,spgpu,gpu-rtx6000 | 53335714 |
| 128k   | niah_multikey_3 | 2 | 64      | off → start2 → start1 | spgpu,gpu-rtx6000 | 53335715 |

mig40 is excluded from the 128k bundle: mk3@128k dense-prefill SDPA OOMs the
39.25GB slice (suite jobs 53253685/53253684 died there even with
expandable_segments + prefill chunk 8192).

Job history: 6 single-arm mig40 jobs (53333931-936, ~24h queue estimate) then
6 single-arm rtx6000 jobs (53335393-398, never scheduled), both sets cancelled
in favor of these two bundles.
Data: `benchmark_suite_result/ktcache_ab_20260709/.../qa_1/validation.jsonl`
(32k identity-gate data) and
`benchmark_suite_result/frozen_sim_20260707/runs/frozensim_mk3_128k_n16/.../niah_multikey_3/validation.jsonl`.
Results under `benchmark_suite_result/draft_verify_gpu_20260711/` (gitignored).

## Analysis

`analyze_draft_verify.py` re-tokenizes each arm's greedy `pred` text (the pred
rows store text, not token ids; greedy decode is deterministic so this recovers
the stream — detokenize/retokenize artifacts cancel since both arms are treated
identically) and reports, per (mode × ctx):

- per-token agreement rate; first-divergence position vs the answer-region token
  position (early answer-bearing fork vs late post-answer fork);
- agreement run-length distribution;
- acceptance@k for k ∈ {4,8,16} (fraction of length-k windows fully matching);
- expected accepted-prefix length at k=8;
- standalone draft correctness (does the draft stream alone still answer).

Invoke (CPU, in worktree):
```
module load python/3.10.4
export LD_LIBRARY_PATH="$PWD/.venv_cu128/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/.hf_pydeps_cu128:$PYTHONPATH"
.venv_cu128/bin/python notes/radical_bw_20260711/draft_verify_gpu/analyze_draft_verify.py \
  --output_root benchmark_suite_result/draft_verify_gpu_20260711 \
  --tokenizer "$(readlink -f .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659)" \
  --out_json notes/radical_bw_20260711/draft_verify_gpu/acceptance.json
```

## Results

_Pending job completion — filled in by the final commit._
