# Attention Approximation Progress Report

This report summarizes the experimental path from dense attention to the current low-memory approximate attention direction.

The important evolution is:

```text
dense attention
-> mass-preserving sparse retrieval
-> online paged PQ for long decode
-> attention-output error as the real metric
-> exact head + tail estimation
-> rank-stratified tail estimation
```

## 0. Dense Attention Baseline

At the 128k decode endpoint in the saved trace:

```text
context length = 134,838 tokens
head dim       = 128
K/V precision  = fp16 K + fp16 V = 4 bytes per dimension
```

Dense attention reads every K/V vector for the query:

```text
134,838 tokens * 128 dims * 4 bytes = 65.84 MB/query
```

This is the starting point. Every approximate method below should be understood as reducing this per-query K/V memory traffic, plus whatever selector/index traffic it introduces.

## 1. Mass-Based Target

The first quality metric was attention probability mass:

```text
attention_mass = sum of dense attention probabilities over retrieved tokens
```

This is straightforward because it gives a clean oracle:

```text
sort tokens by true dense attention probability
retrieve the smallest set whose cumulative mass reaches the target
```

For target mass `0.98`, the oracle at 128k is:

| method | attention mass | step MB/query | meaning |
| --- | ---: | ---: | --- |
| dense attention | 1.000 | 65.84 | exact baseline |
| `top_mass_oracle` | 0.980 | 29.96 | unattainable lower bound for mass-based sparse retrieval |

At this stage we only cared about mass. Output error metrics such as relL2 are intentionally not used yet.

## 2. Mass-Based Baseline Survey

We evaluated several sparse/retrieval-style baselines under the same mass target. The question was:

```text
how many MB/query does each method need to reach attention_mass ~= 0.98?
```

Representative 128k results:

| method | attention mass | selector MB/query | exact K/V MB/query | step MB/query | comment |
| --- | ---: | ---: | ---: | ---: | --- |
| `top_mass_oracle` | 0.980 | 0.000 | 29.96 | 29.96 | oracle lower bound |
| `paged_local_pq_snapshot` | 0.980 | 2.285 | 37.63 | 39.91 | custom PQ selector, snapshot |
| `sparq_r16` | 0.980 | 8.214 | 33.61 | 41.82 | strong existing-style selector, high selector traffic |
| `retroinfer_style` | 0.980 | 0.132 | 42.59 | 42.72 | clustering-style proxy |
| `pqcache_full_scan_snapshot` | 0.980 | 0.288 | 43.40 | 43.68 | PQCache-style full scan |
| `retrievalattention_graph` | 0.319 | 1.176 | 1.13 | 2.30 | cheap but fails mass target |
| `magicpig_k10_l150` | 0.305 | 77.91 | 1.15 | 79.06 | hash proxy fails mass target and has high selector traffic |

Takeaway from the mass-only phase:

```text
If we require 0.98 mass, many practical selectors become expensive exact-token retrieval schemes.
```

The oracle itself needs `29.96 MB/query`, and deployable/proxy methods that actually hit the mass target usually exceed that.

## 3. Online Long-Decode Development

The baseline survey included snapshot-style methods, but our target scenario is long decode. We then focused on selectors that can update online as new tokens are generated.

Paged PQ became the strongest path:

```text
new generated tokens stay in a short exact suffix
when a page fills, build a local PQ codebook for that page
query ranks tokens using page-local PQ scores
routed/gated variants reduce selector traffic by scanning fewer groups
```

We tried several online/paged variants and calibration strategies:

| variant family | config | stop rule | result | interpretation |
| --- | --- | --- | ---: | --- |
| early `paged_local_pq_online` | page `512`, PQ `s2b6` | oracle true-mass stop | `42.21 MB/query` at mass 0.98 | high selector traffic and high exact K/V read |
| improved `paged_local_pq_online` | page `3072`, PQ `s4b6` | oracle true-mass stop | `32.95 MB/query` at mass 0.98 | improvement came from larger pages plus better PQ shape, not page size alone |
| margin/schedule calibrated `paged_local_pq_approx_sched_v2` | page `5632`, PQ `s4b6` | approximate PQ-mass stop plus calibrated margin schedule | `32.52 MB/query`, min mass 0.980 | best clean mass-preserving online PQ schedule |
| residual / probe / boundary verification variants | mostly `>32 MB/query` or miss mass | extra selector reads did not pay off |
| routed/gated PQ under mass target | reduced selector traffic but still constrained by mass | useful later for output-estimation phase |

Important distinction:

```text
paged_local_pq_online:
  ranks tokens with page-local PQ
  evaluator uses true dense attention mass to decide when enough tokens were selected
  this is an oracle-frontier diagnostic

paged_local_pq_approx_sched_v2:
  ranks tokens with page-local PQ
  stops using approximate PQ-score mass plus a prechosen safety-margin schedule
  this is deployable in the sense that it does not inspect true achieved mass at runtime
```

`sched_v2` is not a tail estimator. It still retrieves exact K/V tokens and tries to preserve mass `0.98`. Tail estimation starts later with the `uniform_tail_*` and `strat_exp_tail_*` variants.

One caveat: the `sched_v2` margin schedule was calibrated from previous experiments. It avoids direct oracle true-mass stopping, but it is still an empirically tuned stop rule, not a formal guarantee.

Best mass-preserving online result:

| method | minimum mass over decode suite | max step MB/query |
| --- | ---: | ---: |
| `paged_local_pq_approx_sched_v2` | 0.980 | 32.52 |

This was an important practical baseline, but still mass-driven.

## 4. Why We Switched Metrics

After mass-based experiments, we inspected attention-output quality.

Dense attention output is:

```text
y = sum_i softmax(qk_i) v_i
```

The actual computation consumed by the model is `y`, not the attention distribution by itself.

This exposed two issues:

1. Two methods with the same mass can have different output error.
2. A method can have much lower mass but still produce a good output if the omitted contribution is estimated instead of discarded.

Endpoint-only example at 128k, all near mass `0.98`:

| method | attention mass | step MB/query | output relL2 |
| --- | ---: | ---: | ---: |
| `top_mass_oracle` | 0.980 | 29.96 | 0.01465 |
| `sparq_r16` | 0.980 | 41.82 | 0.01588 |
| `paged_local_pq_online` | 0.980 | 39.91 | 0.00649 |
| `retroinfer_style` | 0.980 | 42.72 | 0.00523 |
| `pqcache_full_scan_snapshot` | 0.980 | 43.68 | 0.00640 |

Same mass does not imply same output error. That motivated the metric shift:

```text
old target: retrieve enough tokens to preserve attention mass
new target: minimize attention-output error per MB/query
```

After this point, relL2 becomes the main quality metric, and mass becomes a diagnostic.

Important apples-to-apples rule:

```text
endpoint relL2 at 128k and max relL2 over the full decode suite are different metrics.
```

The rest of this report uses full-suite max relL2 unless explicitly labeled as endpoint-only. The clean comparison below uses the same trace, same decode lengths, same head, and same cost model:

| method | min mass | endpoint relL2 at 128k | full-suite max relL2 | max step MB/query |
| --- | ---: | ---: | ---: | ---: |
| `top_mass_oracle` | 0.980 | 0.01465 | 0.03111 | 29.96 |
| `retroinfer_style` | 0.980 | 0.00522 | 0.02289 | 42.72 |
| `pqcache_full_scan_snapshot` | 0.980 | 0.00641 | 0.02965 | 43.68 |
| `sparq_r16` | 0.980 | 0.01587 | 0.03934 | 41.82 |
| `paged_local_pq_approx_sched_v2` | 0.980 | 0.01489 | 0.03108 | 32.52 |
| `gated_paged_pq_k4096+strat_exp_tail_b8_s4096` | 0.667 | 0.01180 | 0.01445 | 6.85 |
| `gated_paged_pq_k16384+uniform_tail_s2048` | 0.874 | 0.00812 | 0.00812 | 12.14 |
| `gated_paged_pq_k16384+strat_exp_tail_b8_s4096` | 0.874 | 0.00374 | 0.00882 | 13.14 |

So the correct conclusion is not that stratified tail degraded relative to RetroInfer. RetroInfer has a very low 128k endpoint relL2, but its full-suite max relL2 is `0.02289` at `42.72 MB/query`. The low-cost stratified tail point has lower full-suite max relL2, `0.01445`, at `6.85 MB/query`. If we want endpoint relL2 better than RetroInfer, the `k16384+strat_exp` point reaches endpoint relL2 `0.00374` at `13.14 MB/query`.

## 5. Exact Head + Uniform Tail Estimation

The next idea was to stop treating omitted tokens as zero.

Split tokens into:

```text
head = selected high-ranked tokens
tail = unselected tokens
```

Read the head exactly. Estimate the tail by sampling:

```text
tail_num ~= |T| / m * sum_{j in sampled tail} exp(qk_j) v_j
tail_den ~= |T| / m * sum_{j in sampled tail} exp(qk_j)

y_hat = (head_num + tail_num) / (head_den + tail_den)
```

This was the first major output-error frontier push:

| method | min mass | max relL2 over decode suite | max step MB/query | interpretation |
| --- | ---: | ---: | ---: | --- |
| `paged_local_pq_approx_sched_v2` | 0.980 | 0.03108 | 32.52 | mass-preserving online PQ baseline |
| `paged_local_pq_fraction_f40+uniform_tail_s4096` | 0.831 | 0.01010 | 29.55 | lower mass, better output error |
| `paged_local_pq_fraction_f30+uniform_tail_s4096` | 0.755 | 0.02149 | 22.97 | much lower cost, still beats mass baseline |
| `gated_paged_pq_budget_k16384+uniform_tail_s2048` | 0.874 | 0.00812 | 12.14 | routed selector plus tail estimation |

Key takeaway:

```text
We do not need to retrieve 0.98 mass if the omitted tail is estimated.
```

This changed the algorithmic direction from sparse retrieval to sparse retrieval plus statistical estimation.

## 6. Rank-Stratified Tail Estimation

Uniform tail sampling treats the entire unselected tail as one population. But the tail is not uniform: tokens just below the selected head are usually more important than deep-tail tokens.

The current estimator uses the PQ/gated-PQ rank:

```text
top K: exact head
tail band 0: next highest-ranked unselected tokens
tail band 1: lower-ranked tokens
...
tail band 7: lowest-ranked tokens
```

Rank-geometric allocation samples more from higher-ranked tail bands:

```text
band 0: many samples
band 1: fewer
band 2: fewer
...
band 7: fewest
```

Each band is still estimated by scaling its samples to represent the full band. This makes the estimator multi-resolution:

```text
head: 100% exact
near tail: medium sampling resolution
far tail: low sampling resolution
```

Seed-sweep robust results:

| method | mean max relL2 | worst-seed max relL2 | max step MB/query |
| --- | ---: | ---: | ---: |
| `gated_paged_pq_k4096+uniform_tail_s4096` | 0.04020 | 0.06894 | 6.85 |
| `gated_paged_pq_k4096+strat_exp_tail_b8_s4096` | 0.01228 | 0.01445 | 6.85 |
| `gated_paged_pq_k8192+uniform_tail_s2048` | 0.02559 | 0.02766 | 7.94 |
| `gated_paged_pq_k8192+strat_exp_tail_b8_s2048` | 0.01273 | 0.01372 | 7.94 |
| `gated_paged_pq_k16384+uniform_tail_s2048` | 0.00941 | 0.01040 | 12.14 |

Current best low-cost robust point:

```text
gated_paged_pq_budget_k4096 + strat_exp_tail_b8_s4096
worst-seed full-suite relL2 = 0.01445
step MB/query = 6.85
```

This is the best low-cost full-suite frontier so far.

## Frontier Summary

| stage | representative method | main metric at that stage | max step MB/query |
| --- | --- | --- | ---: |
| Dense attention | dense | exact attention, mass 1.0 | 65.84 |
| Mass oracle | `top_mass_oracle` | mass 0.98 oracle | 29.96 |
| Mass-based online PQ | `paged_local_pq_approx_sched_v2` | min mass 0.980 | 32.52 |
| Output metric + uniform tail | `gated_paged_pq_k16384+uniform_tail_s2048` | max relL2 0.00812 | 12.14 |
| Multi-resolution tail | `gated_paged_pq_k4096+strat_exp_tail_b8_s4096` | worst-seed max relL2 0.01445 | 6.85 |

## Current Conclusion

The main finding is:

```text
mass-preserving sparse retrieval is not the right endpoint
output-aware retrieval + tail estimation is much stronger
```

The best current algorithmic shape is:

```text
routed paged PQ selector
+ fixed exact head budget
+ rank-stratified/geometric tail estimator
+ attention-output relL2 as primary quality metric
```

This gives a clearer research story:

- Dense attention costs `65.84 MB/query` at the 128k endpoint.
- Mass oracle reduces this to `29.96 MB/query`, but is not deployable.
- Practical mass-preserving selectors remain around `32+ MB/query`.
- Output-aware tail estimation drops the cost to `~12 MB/query`.
- Rank-stratified tail estimation pushes the low-cost robust point to `6.85 MB/query`.

## Next Step

The next clean improvement is control-variate tail estimation:

```text
cheap PQ estimate for every band
+ sampled exact residual correction
```

Instead of estimating the entire tail contribution from samples, estimate the error between a cheap PQ approximation and exact K/V contribution. If the PQ approximation correlates with the true contribution, this should reduce variance and either lower MB further or reduce relL2 at the same MB.
