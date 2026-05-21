# Findings Log

This file is now a compact index of high-signal conclusions. The full append-only log is preserved at `notes/archive/status_history/findings_log_2026-05-20_full.md`.

## Current Conclusions

- Mass-only selection was useful for early algorithmic comparison, but it is not sufficient as the final quality target. Attention-output relL2/cosine/logit/hidden drift and task accuracy are required.
- RetrievalAttention-style token graph traversal did not clearly beat compressed/static baselines in the proxy studies. Graph traversal and candidate processing became expensive enough that the original sublinear-efficiency hypothesis was not supported by those runs.
- PQ-style selector logic became the most useful foundation because it gives deployable approximate ranking and supports online long-decode sidecars.
- Tail estimation is the major frontier shift: retrieving less head mass plus estimating the residual tail can reduce output error at much lower logical MB than mass-target-only selection.
- Robustness is head/query dependent. Some heads tolerate aggressive selection and tail estimation; others force larger budgets. This is why online confidence/budgeting is required.
- Dense prefill should remain dense. Sparse/paged selection is targeted at decode, where K/V reuse is low and bandwidth pressure dominates.
- GPU implementation is currently a simulator/benchmark host for a custom-hardware algorithm. Logical frontier MB and physical GPU MB must stay separate.

## Benchmark Findings

- Qwen3.5 generated-memory failures appeared to be model/prompt/benchmark behavior, not sparse replacement degradation, because oracle sparse replacement preserved dense behavior.
- LongBench/RULER benchmark readiness now depends more on canonical frontier runtime and correctness than on finding a new selector family.
- Current canonical 32k RULER smokes can preserve answer quality, but runtime is still not comfortable for broad benchmark sweeps.

## Archived Detailed Logs

- Full historical findings: `notes/archive/status_history/findings_log_2026-05-20_full.md`
- Selector/proxy surveys: `notes/archive/selector_surveys_2026-05/`
- Benchmark audits: `notes/archive/benchmark_audits_2026-05/`
- RA/Roar graph sweeps: `notes/archive/ra_graph_sweeps/`
