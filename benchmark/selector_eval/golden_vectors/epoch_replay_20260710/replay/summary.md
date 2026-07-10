# Physical-line replay summary (issue #11, Phase 2)

Replay of the realized dependency epoch traces (`../traces/`, 4 positions x 32
heads, frozen operating point, tau=0.004) against the RTL physical contract
(issue #11 "Phase 1 ACCEPTED" comment). Producer:
`scripts/replay_epoch_trace_physical.py` (constants + interpretation notes in
its docstring; `--self_test` covers the K-scale register evict-on-intervening
case, rounding, and LRU monotonicity).

All byte/request counts are integers from ONE realized trace per position --
no pre-averaging anywhere. `requests` = 32 B HBM transactions.

## Positions

Selection mined from `joint_kv_ladder_grid_20260706/deesc_precision_compose`
(the only all-288-position dataset at tau=0.004; 4 heads, near-frozen config);
realized rates from the traces themselves.

| qidx | ctx | role | proxy meanK/step | realized K-esc/step (32 heads, frozen cfg) |
|---|---|---|---|---|
| 137 | 12,000 | high-escalation tail (proxy max 1.25) | 1.25 | 0.4375 |
| 262 | 83,225 | mid-context 80-100k (proxy bucket max) | 0.50 | 0.1250 |
| 283 | 126,580 | near cross-position mean (proxy 0.25 ~= mean 0.244) | 0.25 | 0.0938 |
| 287 | 134,838 | MVD (Phase-1 position, regenerated with v2 fields) | -- | 0.0938 |

Cross-position proxy mean = 0.2439 (matches the contract's ~0.24). Realized
rates under the EXACT frozen config (e4m3 logit buffer + frozen precision
split, absent from the proxy dataset) are systematically lower -- both
features suppress marginal escalations; the tail/mid/mean ORDERING is
preserved (q137 >> q262 > q283 ~= q287), which is what the bracketing needed.

## Headline numbers

Dense per-head-query reference (hw_arch_v0.md section 5 basis): ctx x 128 dim
x 2 B x 2 (K+V) = ctx x 512 B; the section's "67 MB at 128k-class" is exactly
ctx=131,072. All ratios below are position totals over 32 heads.

**Measured physical byte ratio vs dense** (0 B window, head-serial = current
RTL):

| qidx | ctx | physical bytes | dense (32 heads) | ratio | reduction |
|---|---|---|---|---|---|
| 137 | 12,000 | 90,712,352 | 196,608,000 | 0.4614 | 2.17x |
| 262 | 83,225 | 414,168,096 | 1,363,558,400 | 0.3037 | 3.29x |
| 283 | 126,580 | 472,224,768 | 2,073,886,720 | 0.2277 | 4.39x |
| 287 | 134,838 | 513,662,560 | 2,209,185,792 | 0.2325 | **4.30x** |

Oracle (unlimited reuse window, full cross-head dedupe): q287 = 174,052,608 B
= 0.0788 of dense = **12.69x** -- the reuse-window sweep spans the 4.30x
(current RTL) to 12.69x (oracle) range.

**Interleaved-minus-serial byte delta per window** (negative = interleaving
saves; the value of cross-head gather scheduling):

| qidx | 0 B | 64 KiB | 256 KiB | 1 MiB | unlimited |
|---|---|---|---|---|---|
| 137 | +7.48 MB | -26.51 MB | -33.74 MB | -35.23 MB | 0 |
| 262 | +37.89 MB | -70.52 MB | -100.83 MB | -163.12 MB | 0 |
| 283 | +41.49 MB | -129.54 MB | -158.59 MB | -208.00 MB | 0 |
| 287 | +43.97 MB | **-168.59 MB** | **-195.49 MB** | -245.81 MB | 0 |

Two structural findings:

1. **Head-serial order gets almost nothing from a bounded window** (q287:
   513.66 -> 511.44 MB at 64 KiB..1 MiB, -0.4%): one head's full gather
   (~15 MB/head at 134k ctx) streams through before the next head starts, so
   every <=1 MiB window has evicted the shared rows by the time they recur.
   The window only recovers K-scale register evictions (tier flaps on shared
   scale lines).
2. **Interleaving is NEGATIVE without a data cache** (q287 0 B: +43.97 MB,
   +8.6%): token-burst round-robin across 4 heads thrashes the last-owner
   scale/sidecar registers (evict-on-intervening-key is the contract
   semantics). With any window >=64 KiB the GQA row overlap (M2 union factors
   K 0.35-0.44, V 0.49-0.57) lands inside the window: -33.0% at 64 KiB,
   -38.2% at 256 KiB, -48.1% at 1 MiB (q287). Cross-head pipelining therefore
   pays for itself ONLY paired with at least a small shared line buffer.

At the 256 KiB primary bound: interleaved q287 = 315.95 MB = 0.1430 of dense
(**7.0x reduction**), vs head-serial 511.44 MB (4.32x) -- i.e., at 256 KiB the
scheduler choice is worth a factor 1.62x in bytes.

## Reconciliation vs Phase-1 logical bytes (gate)

`physical(0B) / contract-logical` (same token sets and contract widths,
no rounding/dedupe): q137 1.0310, q262 1.0266, q283 1.0202, q287 **1.0190**.
The 1.9-3.1% inflation decomposes into (a) the 32 B rounding of the 4 B/token
K-scale and V-sidecar lines wherever a line's 8 tokens are not all read, and
(b) last-owner register re-fetches on hi/lo tier flaps within shared scale
lines. Row planes are exact multiples of 32 B (no rounding loss).

The Phase-1 npz `epoch_region_logical_bytes` sums (q287: 186,940,300 B) cover
the MARGINAL-BAND gather regions only, by design; the contract-logical basis
(q287: 504,076,112 B) additionally includes (i) the start-rung committed K/V
read sets (charged on the start_eval epoch -- the dominant term, most heads
settle at start), (ii) committed V priced at the contract 260 B/token
(2 planes + sidecars) instead of the Phase-1 128 B lo-plane width, and
(iii) the per-head scan/codebook/metadata streams (ladder accounting, with
the V-PQ metadata stream at the contract ceil(5N/8) rule). Component (i) is
recoverable per head from `start_k_tokens`/`start_v_tokens`/`k_hi_tokens`;
nothing was re-derived outside the trace.

## Gate results

- bytes monotone non-increasing as the window grows, per order: PASS
  (asserted in-tool for all 4 positions x 2 orders).
- unlimited == oracle: PASS by construction (asserted: oracle JSON bytes
  total == unlimited sweep row; both orders converge to the same total).
- reconciliation vs Phase-1 logical: PASS, inflation 1.9-3.1% documented
  above.
- K-scale register semantics unit-tested (`--self_test`): evict-on-
  intervening-key, tier-flap re-fetch, 8-tokens/line coalescing: PASS.
- integers from one realized trace, no pre-averaging: enforced by the JSON
  validator (schema-exact fields, non-negative ints, per-head DAG deps).

## Files

- `replay_sweep.csv` -- position x window x order -> physical bytes, requests,
  per-class byte breakdown (k_rows / k_scale / v_rows / v_sidecar / scan),
  dense + logical references, ratios.
- `epochs_q{137,262,283,287}_oracle.json` -- RTL schema, unlimited window,
  head-serial order.
- `epochs_q{137,262,283,287}_bounded_256KiB.json` -- RTL schema, 256 KiB
  window, head-serial order.

## Interpretation notes flagged for RTL review

1. K-scale register key `{line, slot, tier}`: implemented as (slot = scale
   region per lane, line, tier) -- i.e., an intervening different LINE or TIER
   evicts; `slot` does not change within a lane's K-scale stream. If `slot`
   was meant to be the per-token slot within the line, the register would
   never dedupe adjacent tokens and K-scale bytes rise to 32 B/token.
2. V sidecars (4 B/token, 32 B lines) are given the same last-owner register
   treatment as K-scale (their own slot), consistent with the 260 B/token
   committed-V figure being exact when 8 consecutive tokens share a line.
3. Scan/codebook/metadata streams participate in the reuse window (no
   bypass): in head-serial order they self-evict at bounded windows; in
   interleaved order the 4 heads walk them in lockstep and dedupe. If the
   RTL scan path bypasses the shared buffer, the interleaved scan savings
   (~6.4 MB/lane-scan at 134k) move from the window to the dedicated scan
   sharing already booked in hw_arch section 5.
4. `resource_rates` are copied verbatim from the contract schema example;
   `qk_flops` per gather epoch = 2 x head_dim x (K rows + V rows read by that
   epoch); scan `scan_items` = context_len.
5. Interleaved JSONs are not emitted (contract asks for oracle + bounded);
   the sweep CSV carries the interleaved totals. Both JSONs use head-serial
   order (current RTL).
