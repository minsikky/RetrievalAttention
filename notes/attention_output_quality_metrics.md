# Attention Output Quality Metrics

This note records why `attention_mass` should be treated as a proxy, not the final quality metric, for selector-eval experiments.

## Core Point

Dense attention output is:

```text
softmax(QK^T) V
```

Selecting high-probability tokens approximates only the probability distribution. It does not directly measure whether the resulting value-weighted vector is close to the dense attention output.

Use these local metrics before expensive task-level evaluation:

- `attention_mass`: interpretable selector recall proxy.
- `output_cosine`: direction match of the attention output.
- `output_relative_L2`: magnitude-sensitive output error; use this with cosine because cosine alone can hide scale error.
- `distribution_JS`: distribution-level shift diagnostic, not enough by itself because it ignores `V`.

## Literature Signals

- Value-aware Approximate Attention (`arXiv:2103.09857`): attention approximations should target the true attention sub-layer output and include value-vector effects.
- CAOTE (`arXiv:2504.14051`): token importance for KV eviction should measure attention-output error, not only attention scores.
- Output Perturbation KV selection (`arXiv:2502.03805`): values and pretrained output matrices matter when identifying critical KV entries.
- CurDKV (`arXiv:2509.15038`): attention-score approximation does not guarantee preservation of `softmax(QK^T)V`; value-aware selection can reduce reconstruction loss.
- Delta Attention (`arXiv:2505.11254`): sparse attention can create output distribution shift; correcting the output can recover quality on top of sparse selectors.
- vAttention (`arXiv:2510.05688`): top-k and sampling are complementary; top-k handles peaked distributions, while sampling/estimation handles flatter tails with statistical guarantees.
- Natural sparsity theory (`arXiv:2404.02690`): attention can be sublinear-sparse under assumptions, but stable extremely sparse attention may be impossible; adaptive budgets are more defensible than fixed tiny budgets.

## Experiment Implication

The next selector-eval phase should report mass and output metrics together. A selector that misses `0.98` mass may still be useful if:

- it has high `output_cosine`,
- low `output_relative_L2`,
- low cost,
- and its tail handling is deployable without oracle attention probabilities.

The next concrete algorithm family should be:

```text
exact selected top tokens + cheap tail estimator/correction
```

Candidate tail estimators:

- sample-based tail estimate,
- page/PQ-cluster value summaries,
- low-rank or mean-value residual,
- hybrid top-k plus statistically bounded sampling.

Do not use dense probabilities or achieved mass inside deployable selector logic. Oracle/fixed-fraction variants are diagnostics only.
