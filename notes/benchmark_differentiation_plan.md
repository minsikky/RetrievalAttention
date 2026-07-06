# Benchmark Differentiation Plan (paper evaluation strategy) — 2026-07-06

Problem: our task validation so far (niah_single_1, niah_multikey_2, vt at
32k; niah_single_1 at 128k) scored 100.0 under dense, under the frontier at
every tau, and under Gaussian noise to relL2 0.05. That validates tau but
proves these tasks are SATURATED — they cannot support the paper's central
claim, which needs benchmarks where (a) dense is strong but not at ceiling,
(b) prior KV-reduction methods measurably drop, (c) ours stays at dense.

Claim we are building toward: *static-budget KV reduction (eviction, fixed
top-k, uniform quantization) degrades on hard long-context workloads;
an adaptive, stability-certified budget with exact-KV escalation does not,
at comparable bytes/token.* Every phase below either finds the workloads
where (b) happens or produces the matched-budget comparison for (c).

## Phase A — RULER-hard knee finding (RUNNING, jobs 52996323-28 + qa pending)

Tasks chosen for known non-saturation at 128k with Llama-3.1-8B: `qa_2`
(HotpotQA multi-hop in haystack), `qa_1` (SQuAD), `cwe`, `fwe` (aggregation
— attention spread across ALL pages; our own page-pruning negative says
these stress token selection hardest), `niah_multikey_3` (uuid keys).
Arms per task at 131072/n16: dense reference, frontier tau=0.004 (operating
point), frontier tau=0.016 (deliberate stress).
Reading the result:
- dense high + 0.004 == dense + 0.016 < dense -> task is a DIFFERENTIATOR
  (sensitive to KV error, our operating point holds): paper task set.
- 0.016 == dense -> task insensitive; drop it (same lesson as the 32k trio).
- 0.004 < dense -> our operating point breaks: most important possible
  outcome; retune tau on hard tasks (the calibration was easy-task-based —
  the user's original skepticism about the relL2 proxy applies exactly here).
- dense low (<~30) -> floor effects; use 64k instead.

## Phase B — matched-budget baseline matrix (mostly OUR harness, config-only)

Prior-work archetypes reproduced as arms of the trace/GPU runners, all at
MATCHED mean bytes/token to our operating point (2.86 MB/head-query class):
1. **Fixed top-k (Quest-class)**: selector ranking + fixed rung, no ladder
   (predict-only mode already exists; pick rung to match bytes).
2. **Eviction (H2O/SnapKV-class)**: irreversible keep-set — approximate with
   frozen selection (we already measured the failure: relL2 tail 0.20, p99
   0.102 at staleness 2100; on tasks this becomes the eviction column).
3. **Uniform quantization (KIVI-class)**: all-lo tier (k0p0_v0p0 exists;
   4-bit lo4 arm exists, relL2 0.007-0.018 at trace level).
4. **Ours**: deescalate + precision @ tau=0.004.
These four columns on the Phase-A differentiator tasks = the paper's main
table. External codebases (real Quest/H2O/KIVI) only needed for camera-ready
credibility on 1-2 rows; the archetype arms tell us first whether the
differentiation exists. GPU promotion needed for arms 1-3 on task evals:
predict-only and lo-tier are config in the intervention runner; frozen
selection needs a small patch (or is presented trace-level only, honestly).

## Phase C — HELMET (repo ALREADY CLONED: third_party/benchmarks/HELMET,
data.tar.gz present; prep scripts scripts/prepare_helmet_data.sh)

Why: community consensus benchmark exactly because NIAH saturates; the
categories that differentiate KV methods in the HELMET paper are passage
re-ranking, RAG QA, and citation generation — recall alone does not.
Steps: (1) untar/verify data (prep script is idempotent); (2) adapt our
streaming runner: HELMET drives HF generate() much like RULER pred — the
integration point is the same paged-PQ patch used by
call_pagedpq_streaming.py; (3) start with 3 categories x {dense, ours,
fixed-topk} at 64k/128k: rerank (msmarco), rag (nq/hotpotqa kilt), longqa.
Effort: the harness adaptation is the real work (1-2 focused days); do NOT
start it in this window — spec it and run in the next session.

## Phase D — long-generation decode drift (our unique angle)

Ours is decode-only sparsity: per-step error compounds over generated
tokens. Nobody's benchmark measures quality vs GENERATION length at fixed
context; static-budget methods should drift (their error is unconditioned),
ours should not (the ladder re-certifies every step). Infra exists:
`benchmark/public_longdecode_eval.py`, LongGenBench notes in
notes/archive/. This is the evaluation a reviewer cannot say is cherry-
picked from prior work — it is implied by the method's design.
Plan: quality-vs-decode-position curves on public_longdecode + LongGenBench
SGT metrics, ours vs fixed-topk at matched bytes.

## Venue calibration (2026-07-06, after discussion)

Target venue is architecture/circuits (chip tape-out), not NeurIPS-class ML.
Consequences for the evaluation set:
- **Benchmark breadth: RULER-hard + HELMET is sufficient**, plus two
  nearly-free additions: **LongBench-v2** (harness already in repo,
  `longbench_v2_hf_eval.py`; frontier already matched dense at short bins —
  extend to long bins, config-only) and **long-document perplexity**
  (PG19-class slices; smooth metric that renders the tau-knee and baseline
  degradation as continuous curves instead of task-score deltas).
- **Model generality is the real attack surface**: everything is calibrated
  on Llama-3.1-8B. Add ONE second family — Qwen2.5-7B/14B at 128k — and
  re-verify that page size / PQ config / tau / proxy-mass transfer
  untouched. Hardware-relevant twist: Qwen GQA is 7:1 vs Llama 4:1, which
  stresses the kv-lane sharing design; do this BEFORE RTL freezes lane
  arithmetic.
- **One frozen config everywhere**: deesc + precision @ tau=0.004 for every
  accuracy number in the paper; no per-task tuning. Arch reviewers check
  consistency with the hardware-eval config, not benchmark count.
- **Skip** (venue-inappropriate effort): InfiniteBench, BABILong,
  ZeroSCROLLS, LV-Eval; exhaustive author-code baselines beyond Quest +
  KIVI. The load-bearing pillars are matched-bytes iso-accuracy (B),
  decode-drift (D, doubles as the sustained-decode story), and RTL
  bandwidth/energy/area vs a GPU baseline — accuracy section must be
  unimpeachable, not encyclopedic.

## Sequencing / cost

- Tonight (this window): Phase A jobs (8 GPU jobs, ~4-8h each, submitted /
  pending qa data download); analysis when they land.
- Next session: Phase A knee analysis -> pick 3-4 differentiator tasks;
  Phase B GPU arms on those tasks; HELMET harness adaptation (C).
- Paper table: rows = {dense, ours, fixed-topk, eviction, 4-bit}, columns =
  {RULER-hard picks, HELMET rerank/rag/longqa, longdecode drift}, all at
  matched bytes/token, plus the bytes-vs-quality Pareto per task.

## Risks / honest unknowns

- Aggregation tasks (cwe/fwe) may break OURS too (selection must cover
  everything; the ladder would escalate toward dense — quality holds but
  the bytes advantage shrinks; that is a finding about workload taxonomy,
  and the adaptive ladder degrading GRACEFULLY to dense is itself the
  contrast with eviction, which cannot).
- If nothing in RULER-hard differentiates at 128k, the differentiation
  burden moves entirely to HELMET rerank/citation and Phase D drift.
- Baseline archetypes are simulations sharing our selector; reviewers may
  demand author-code baselines — budget for Quest + KIVI real runs later.
