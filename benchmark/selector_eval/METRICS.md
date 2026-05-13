# Selector-Eval Metric Stack

Metric hierarchy, from cheapest proxy to most realistic:

1. `attention_mass`: probability mass captured by selected tokens.
2. `FN_mass` / `FP_mass`: oracle target mass missed and extra selected mass.
3. `distribution_JS`: divergence between dense attention distribution and renormalized sparse distribution.
4. `output_cosine`: cosine similarity between dense attention output and sparse attention output.
5. `output_relative_L2`: relative L2 error of attention output.
6. Layer-output / residual-stream error.
7. Logit-level agreement.
8. Task-level benchmarks such as RULER.

Current iteration target:

- Use metrics 1-5 for fast selector iteration on saved real-model Q/K/V traces.
- Promote only promising selectors to layer/logit/task-level experiments.

Interpretation:

- High mass with bad output metrics means selected tokens do not preserve value-weighted output well.
- Good output metrics with lower mass can still be acceptable if missed tokens have cancelling or redundant values.
- FP/FN balance helps distinguish routing recall failure from low-precision over-selection.
