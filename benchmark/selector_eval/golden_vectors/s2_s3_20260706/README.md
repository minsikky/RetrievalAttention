# Golden vectors: S2 (PQ scan) + S3 (rank / rung select) — 2026-07-06

Block-level goldens for the first RTL blocks, per issue #5 and
`notes/algorithm_spec_v1.md` §8. One `golden_q<qidx>_h<head>.npz` per trace
row: heads {0, 8, 16, 24}, decode lengths {8000, 32000, 128000} over the
Llama-3.1-8B layer-16 trace (positions 14837 / 38837 / 134837 — context
14.8k / 38.8k / 134.8k). These S2/S3 dumps are taken after the selector and
before any policy simulation, so they are unaffected by the later
de-escalation removal or by precision tiers. Current frozen policy is
escalation-only + precision(0.1,0.1) @ tau=0.004.

## npz contents

| key | meaning |
|---|---|
| `qidx, head, kv_head, position, context_len` | row identity (kv_head = head // 4) |
| `dynamic_start, sealed_end, page_size` | page state: sealed pages cover [dynamic_start, sealed_end) in steps of page_size (5632); tokens outside sealed pages and inside the static prefix (128) / suffix (128) are resident ("base") and DO NOT appear in `ranked_idx` |
| `query_fp32` | the q vector (d=128) for this head/step |
| `ranked_idx` | token indices sorted by PQ logit descending — S3's expected output order (int64) |
| `ranked_scores_raw_fp32` | raw PQ logits in `ranked_idx` order, BEFORE the 1/sqrt(d) scale — S2's expected per-token output (fp32 reference) |
| `ranked_scores_postscale_fp16` | raw / sqrt(128), fp16 — the value the mixed-softmax tail consumes |
| `k_budgets, v_budgets` | rung tables for this context: budget = max(1, min(ctx, **ceil**(frac x ctx))) over the FULL context_len, deduplicated ascending (spec §3) |
| `proxy_mass_start_ki, proxy_mass_start_vi` | proxy_mass_m0p9 start rungs computed from `ranked_scores_raw_fp32` (spec §4 step 1). c = smallest ranked-prefix count with softmax mass >= 0.9 (softmax over indexed tokens only, logits x 1/sqrt(d)); **k_target = max(k_budgets[0], c)** (the crossing count itself, clamped below by the bottom rung); **v_target = max(v_budgets[0], 0.25 x k_target)** — NOTE: 0.25 multiplies the CLAMPED k_target, not raw c (differs when c < k_budgets[0]); ki/vi = first rung with budget >= target (float compare; clamps to the top rung if no rung suffices) |

Rung prefix set at rung ki = base tokens ∪ `ranked_idx[:k_budgets[ki] - |base ∩ selected|]`
— in practice the golden model takes `ranked_idx[:budget]` after excluding
base tokens from the ranking, so the prefix at budget B is exactly
`ranked_idx[:B']` where B' = B minus the base-token count; row-level CSVs
(`selected_k_tokens`) give the settled totals for cross-checking.

## Page blocks for S1/S2 (`page_ctx<ctx>_kv<kv>.npz`)

Codebooks + codes of the LAST sealed page per (context, kv_head) — 12
files covering the same rows as the golden dumps, so S1 (LUT build) and
S2 (scan) can be driven end-to-end without the trace. Fields:
`codebooks_fp32` (subvecs x 256 x subdim, the exact fp32 bits the
reference einsum consumes), `codes_u8` (page_size x subvecs),
`page_start`, `page_size`, build config + seed. Self-checked at dump
time: reference logits recomputed from these blocks match
`ranked_scores_raw_fp32` bit-for-bit on the page's token range
(regenerate/verify with `runners/dump_golden_pq_pages.py`).

## Inputs to drive RTL (pointers, not copies)

- K/V/q trace: `attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz`
  (gitignored artifact; regenerate or copy from the algo machine — see
  `notes/current_status.md` for provenance).
- Page PQ build: `build_page_pq_gpu` in
  `benchmark/selector_eval/runners/run_gpu_paged_pq_eval.py` — page 5632,
  4 subvecs x 8 bits, k-means 3 iters, deterministic seed; codebooks/codes
  for any page can be regenerated bit-exactly from the trace with that
  function (same key prefix + same seed => identical pages; this is the
  same reuse guarantee the sealed-page build cache relies on).
- Regeneration of these dumps:
  `run_joint_kv_budget_policy_eval.py --golden_dump_dir <dir>
  --decode_lengths 8000,32000,128000 --heads 0,8,16,24` + the frozen-config
  flags in `notes/algorithm_spec_v1.md` §8.

## Arithmetic of the reference (verification contract, issue #5)

`ranked_scores_raw_fp32` IS the high-precision fp32-accumulated reference
the tolerance-based S2 compare anchors to. Its exact arithmetic
(`pq_page_scores` in `benchmark/selector_eval/gpu/run_gpu_paged_pq_eval.py`,
torch CPU):

- LUT: `table = einsum("ms,mcs->mc", q_parts, codebooks)` — fp32 in, fp32
  out (each LUT entry is a 32-wide fp32 dot product, torch's default
  reduction order).
- Per-token logit: fp32 accumulator, exactly 4 sequential adds in fixed
  subvector order 0->1->2->3: `scores += table[sub][code[token,sub]]`.
- Ranking: stable descending sort on the fp32 logits (`torch.argsort`,
  `stable=True`); ties keep token order.
- `ranked_scores_postscale_fp16`: raw fp32 **divided by** fp32(sqrt(128))
  (true fp32 division, NOT multiply-by-reciprocal — the two differ by
  1 fp16-ulp on ~34 values across these dumps), then cast to fp16
  round-to-nearest-even.

Per the contract agreed on issue #5: S2 accumulation order is left to
hardware — RTL logits compare against `ranked_scores_raw_fp32` within a
relative tolerance set by the buffer-format quantization step (see issue
#6), NOT bit-exactly. S3 stays exact: given the RTL's own quantized logit
codes, selected sets and prefix counts must match a counting-sort reference
bit-for-bit (256-bin histogram + boundary-bin refine, exact fractional
counts per issue #4).

## Tolerances

- `ranked_idx`: exact match required down to ties; ties in raw fp32 logits
  are broken by scan order (stable). If RTL accumulates LUT adds in a
  different association order, fp16/fp32 rounding can swap true ties only.
- `ranked_scores_raw_fp32`: fp32 reference; RTL comparing at fp16 should
  match `ranked_scores_postscale_fp16` bit-exactly after its own 1/sqrt(d).
- Start rungs: exact integer match.
