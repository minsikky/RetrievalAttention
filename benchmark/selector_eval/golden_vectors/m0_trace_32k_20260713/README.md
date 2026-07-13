# M0 decode trace — 32k perf-iteration sequence (issue #24, Priority 1)

Model-derived all-head decode trace for the RTL M0 "faithful trace input"
milestone. This is the **32k perf-iteration gate** artifact: 8 consecutive
decode positions, all 32 query heads (grouped by the 8 KV heads), same npz
schema as `epoch_trace_20260710`, produced by the CPU golden model at the
**OP-0.9 fixed parser** operating point.

## Operating point (stated explicitly)
`--start_strategies proxy_mass_m0p9`, run on the **CPU golden model**, which
parses `proxy_mass_m0p9` as **0.9** — this is the post-#23 contract OP
(OP-0.9). No GPU parser is involved; there is no 0.5/0.9 ambiguity here. Every
committed set in this artifact is an OP-0.9 result.

## Context choice — and why
We use the **existing 38,838-token golden context** (decode position 38,837 =
`input_len 6838 + 32000` decode tokens; the same context family as
`epoch_trace_20260710`, which is q287 at 134,838). Rationale: this context is
already a checked-in golden anchor of the same capture
(`real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz`), so (a) it
reuses the identical K/V geometry and pipeline that produced the accepted
128k golden, (b) its newest step (38,837) reuses the golden Q verbatim so it is
directly cross-checkable, and (c) it needs no separate "32k matrix context"
capture. Per-{position,head} npz at 38.8k are ~2–4× smaller than the 134.8k
ones, keeping the 8-position artifact checked-in-able.

## The 8 consecutive positions and the token stream driving them
Positions **38,830 … 38,837** (8 consecutive absolute decode positions;
context lengths **38,831 … 38,838**). Each step k attends to
`keys[:, :position_k+1]`, so the KV cache grows by exactly one real token per
step — a genuine steady-state serial-decode run ending at the golden context.

The token stream is the model's **own captured decode trajectory**: the source
X-trace (`real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz`)
records the layer-16 input hidden state for ALL 137,909 positions of the actual
Llama-3.1-8B generation run (do_sample, temp 0.8, top_p 0.95, stop tokens
masked). Nothing is synthetic. The existing capture subsamples the 131,071
decode positions (no run of 8 consecutive query vectors exists in it), so Q at
the 8 consecutive positions is re-projected on CPU from those real layer inputs
via `scripts/build_consecutive_qkv_trace.py` (rmsnorm → wq → NeoX RoPE — the
identical arithmetic as `scripts/convert_layer_trace_to_qkv_npz.py`). K/V are
sliced verbatim (byte-identical fp16) from the golden qkv trace; the anchor
(38,837) Q is copied verbatim from the golden trace, so a dump at the anchor
reproduces the golden committed sets bit-for-bit (the recompute of that column
agrees with the golden GPU projection to 6.1e-5, one fp16 ULP at that scale).

## Files
- `trace/epoch_q{qidx}_h{head}.npz` — one per (position, head); qidx 0..7 map
  to positions 38,830..38,837 (qidx 7 = the newest/anchor).
- `trace/epoch_trace_index.csv|jsonl`, `trace/README.md` — per-run index +
  auto-generated field README.
- `trace/group_union_q{qidx}_ctx{ctx}.npz` + `group_union_summary.json` —
  per-position committed-set UNION per KV group (issue #24 item 6).
- `gold_run/` — layer/per-head policy CSVs, `gqa_union_stats.csv`, `args.json`.
- `stage2_xcheck/` — head-0 anchor stage-2 item-8 dump used to cross-check the
  pass-1 stream (see below).

## npz schema
Every `epoch_q*_h*.npz` carries the **full `epoch_trace_20260710` field set**
(file meta; per-epoch `epoch_*` columns; region logical bytes; CSR marginal/
boundary token sets; `start_k_tokens`/`committed_k_tokens`/`committed_v_tokens`;
Phase-2 `trace_format_version=2`, `start_v_tokens`, `k_hi_tokens`,
`k_hi_tokens_start`). See `trace/README.md` for the exhaustive per-field list.

### Added at the newest position (38,837) only — pass-1 V-risk stream
Namespaced `pass1_*`, purely additive (committed sets unchanged), emitted by
`--epoch_pass1_stream_position 38837`. This is the **full per-token pass-1
scan-domain V-risk key stream** (today's epoch traces carry only the walk-basis
marginal/boundary tokens). It reproduces the item-8 `two_pass_risk` pass-1
operand EXACTLY (same `mixed_scores` inputs, same signed log-risk
`2*pass1_logit + log(V-error)`):
- `pass1_vrisk_q76` — dense `(context_len,)` fp64 signed log-risk key,
  RNE-quantized on the **Q7.6** fixed-point grid: **int_bits=7, frac_bits=6,
  13-bit total signed, LSB = 2^-6, clamp ±64** (dead on observed data). This
  is the pinned RTL convention (issue #9 / stage2_20260707 item 8). `-inf` is
  the structural sentinel for zero-V-error (never-committable) tokens, passed
  through unquantized.
- `pass1_finite_mask_packed` — `np.packbits` of `isfinite(pass1_vrisk_q76)`;
  recover with `np.unpackbits(...)[:pass1_n_tokens]`. The finite bit is the RTL
  "finite bit stream" (marks committable tokens).
- `pass1_q76_int_bits` (7), `pass1_q76_frac_bits` (6) — the pinned grid.
- `pass1_logrisk_min_fp64`/`pass1_logrisk_max_fp64` — UNquantized finite
  log-risk range (sizes the RTL integer field).
- `pass1_cutoff_rank` (settled exact-V budget, f_mult 1.0),
  `pass1_cutoff_q_fp64` (pass-1 scalar cutoff) — so the pass-1 pick is
  rebuildable from the dense stream.

**Cross-check (issue #24 item 5):** `stage2_xcheck/` re-dumps head 0 at the
anchor via the independent stage-2 item-8 code path
(`--two_pass_cutoff_frac_bits 6 --two_pass_cutoff_int_bits 7`); the P1 job
asserts the finite `pass1_vrisk_q76` values equal that path's
`two_pass_pass1_logrisk_q_fp64` set-for-set. The Q7.6 encoding matches the
S2 golden convention (`golden_vectors/stage2_20260707`, item 8).

### Group union (issue #24 item 6)
`group_union_q{qidx}_ctx{ctx}.npz`: `kv_heads`, CSR
`group_union_k_tokens/_offsets` and `group_union_v_tokens/_offsets` (union of
the committed sets over the 4 query heads sharing each KV head), plus
`sizes_json` with per-group `k_union/k_sum/k_share`, `v_union/v_sum/v_share`
(the GQA-sharing factor RTL A/Bs).

## How produced
- Trace builder: `scripts/build_consecutive_qkv_trace.py` (commit b53d195).
- Run + gates: `scripts/run_m0_trace_p1_32k.sbatch` (CPU,
  `--partition=standard --account=zhengya0`).
- Dumper: `benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py`
  at commit b53d195 (`--epoch_pass1_stream_position`).
- Regression gate: `scripts/run_m0_regression_gate.sbatch` reproduces the
  single-position q287 dump on the ORIGINAL trace with this code (pass-1 off)
  and diffs array-for-array against `golden_vectors/epoch_trace_20260710`
  (70 shared golden keys identical; the 4 Phase-2 fields are a documented
  op-fix-HEAD exception predating the Jul-10 golden, not a task-#40 change).

## Sizes / validation (job 53438985, 8m44s CPU)
- **256** `epoch_q*_h*.npz` (8 positions × 32 heads) + **8** `group_union_*`;
  total **56 MB** (checked in on branch `m0-trace`).
- Gate 2 (walk-MB reconciliation): PASS — `gate2_n_fail=0`, max abs err 0.0 MB.
- Gate 3 (GQA union reproduction vs `gqa_union_stats.csv`): PASS — 64 groups
  checked, 0 fail.
- Pass-1 cross-check (item 5): the anchor head-0 `pass1_vrisk_q76` finite set
  equals the stage-2 item-8 `two_pass_pass1_logrisk_q_fp64` set-for-set
  (33,792 = 33,792, equal=True).
- Regression gate (job 53439512, separate): reproducing the single-position
  q287 dump on the ORIGINAL trace with this code (pass-1 off) is value-
  identical to `golden_vectors/epoch_trace_20260710` on all 70 shared golden
  keys across 32 heads (0 mismatches); only the 4 documented Phase-2 fields are
  extra (op-fix-HEAD, predates the golden; not a task-#40 change).
- Observed GQA committed-set sharing (union/sum) at the 8 positions:
  K 2.20–3.53×, V 1.33–2.79× across the 8 KV groups (matches the 1.99–3.50×
  reported for 128k q287 in the issue).
