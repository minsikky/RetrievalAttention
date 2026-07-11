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

## Runs (all `--partition=gpu-rtx6000 --account=zhengya0`)

Token comparisons are contaminated by cross-card FP differences, so ALL arms are
pinned to one single card class (never a multi-partition list). Originally
submitted on gpu_mig40 (53333931-936) to match the cached-baseline card class,
but mig40 was ~24h saturated. Since we run our own frozen arm (`DRAFT_MODE=off`)
with the **identical** env, the comparison is internally consistent on any
single card class and never touches a cached mig40 run; resubmitted with a
command-line `--partition=gpu-rtx6000` override (RTX Pro 6000 Blackwell 96GB —
overrides the mig40 default in the sbatch header). Each arm greedy-decodes;
only the selection differs between arms. First-run-on-Blackwell caveat: if the
first job dies with a kernel/arch error, fall back to the mig40 set.

Driver: `scripts/run_draft_verify_one.sbatch` — the tokpar identity-gate env
(`run_fspq_tokpar_identity_gate.sbatch`) minus the comparison epilogue, plus
`SELECTOR_PQ_JOINT_DRAFT_MODE`. `FUSED_VPREFIX_TOKPAR=0` (as the gate) so control
flows through `select_joint_kv_budgets` (the draft chokepoint) rather than the
tokpar-batched deferral branch.

| run name            | ctx  | task            | n | max_new | mode   | job (rtx6000) |
|---------------------|------|-----------------|---|---------|--------|---------------|
| qa1_32k_n4_off      | 32k  | qa_1            | 4 | default | off    | 53335393 |
| qa1_32k_n4_start1   | 32k  | qa_1            | 4 | default | start1 | 53335394 |
| qa1_32k_n4_start2   | 32k  | qa_1            | 4 | default | start2 | 53335395 |
| mk3_128k_n2_off     | 128k | niah_multikey_3 | 2 | 64      | off    | 53335396 |
| mk3_128k_n2_start1  | 128k | niah_multikey_3 | 2 | 64      | start1 | 53335398 |
| mk3_128k_n2_start2  | 128k | niah_multikey_3 | 2 | 64      | start2 | 53335397 |

Chained to keep ≤2 concurrent: A = 5393→5395→5397, B = 5394→5396→5398.
(The cancelled mig40 set was 53333931-936.)
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
