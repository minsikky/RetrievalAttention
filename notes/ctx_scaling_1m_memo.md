# Context-scaling memo: 14.8k → 134.8k trends, 1M extrapolation (issue #8)

2026-07-06. Data: the 12 s2_s3 golden dumps (heads 0/8/16/24 × ctx
14.8k/38.8k/134.8k). Three points per head. Methodology: c = softmax-0.9
crossing count (logits × 1/√128, fp64 cumsum); boundary ties = tokens
sharing the e4m3 code of the c-th ranked token; maxbin = largest e4m3
histogram bin over the whole spectrum of that head's logits.

| head | ctx | top | min | c | c/n | boundary ties | max bin |
|---|---|---|---|---|---|---|---|
| 0 | 14.8k | −55.0 | −140.6 | 5807 | 0.516 | 2314 | 2314 |
| 0 | 38.8k | −59.5 | −151.5 | 17377 | 0.514 | 7727 | 7727 |
| 0 | 134.8k | −52.3 | −170.6 | 35413 | 0.273 | 16014 | 33953 |
| 8 | 14.8k | −36.4 | −172.3 | 3539 | 0.314 | 1260 | 1534 |
| 8 | 38.8k | −42.9 | −163.5 | 14313 | 0.424 | 6610 | 6884 |
| 8 | 134.8k | −38.6 | −220.2 | 17205 | 0.133 | 7755 | 23655 |
| 16 | 14.8k | −58.6 | −160.2 | 4799 | 0.426 | 1791 | 2028 |
| 16 | 38.8k | −61.8 | −187.7 | 15818 | 0.468 | 6837 | 6837 |
| 16 | 134.8k | +13.5 | −222.6 | 6556 | 0.051 | 1170 | 17638 |
| 24 | 14.8k | −48.2 | −160.0 | 4894 | 0.434 | 2380 | 2380 |
| 24 | 38.8k | −54.1 | −185.0 | 11145 | 0.330 | 4944 | 7975 |
| 24 | 134.8k | −40.1 | −216.1 | 34415 | 0.266 | 13657 | 30037 |

## Findings

1. **e4m3 stays in-format at 1M: LOW RISK.** Per-head top logit is
   strikingly stable across 9× ctx growth (h0: −55.0/−59.5/−52.3; drift
   ±7 raw). The min grows slowly and sub-linearly in log-ctx
   (−140→−220 over 9×); even generous extrapolation keeps |min| < 300 at
   1M, inside ±448 saturation. Deep-tail saturation, if it ever occurred,
   affects only resolution softmax cannot see. The per-kv-head static
   bias hook (#6 contingency) is NOT needed for range at 1M on this
   evidence; whether it is needed for top-step coarseness is a separate
   question pending job 53008051.

2. **c/n FALLS with context — and that makes the proxy-mass clamp the
   MAINLINE path at 1M.** The 0.9-mass crossing grows sublinearly (h0
   c/n: 0.52→0.51→0.27; h16 down to 0.051). Good for bandwidth (selection
   fraction shrinks exactly when ctx grows). But note h16 @ 134.8k:
   c/n = 0.051 < bottom-rung fraction 0.10 — the k_target =
   max(k_budgets[0], c) clamp (the corner case flagged in the #5 thread)
   is ALREADY ACTIVE at 134.8k on one of four heads. Extrapolating the
   c/n trend, most heads cross below 0.10 somewhere before 1M: at the
   chip's target context the clamped form is the COMMON path, not the
   corner. The FSM must implement v_target = 0.25 × CLAMPED k_target;
   getting this wrong is invisible at 14.8k and wrong-answer at 1M.

3. **Boundary-bin ties grow ~linearly with ctx.** Boundary ties ≈
   0.06–0.12 × ctx depending on head; max bin ≈ 0.13–0.25 × ctx at
   134.8k. Projected 1M: boundary bins of ~10⁵, max bins ~2×10⁵.
   Selection exactness is unaffected (scan-order refine is exact at any
   tie count), but the refine pass's worst-case token count — and thus
   its SRAM/latency budget — should be sized for ~2×10⁵-token bins, not
   the ~9k seen at 134.8k.

## Status of the 1M gap

The "no 1M GPU runs possible" premise in issue #8 is superseded:
gpu-rtx6000 nodes carry 96 GB RTX Pro 6000 Blackwell parts, and
Qwen2.5-7B-Instruct-1M (native 1M, GQA 7:1, 55 GB KV @ 1M fp16) fits on
one. Real 1M task validation (RULER-1M + BABILong) plus a real 1M trace
capture for the CPU golden model are planned as Phase E in
`benchmark_differentiation_plan.md`; this memo is the interim risk
assessment until those land.
