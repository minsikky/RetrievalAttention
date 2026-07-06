# LongGenBench SGT Metrics

Figure: `notes/archive/benchmark_audits_2026-05/longgenbench_sgt_metrics_20260529.png`

Caveat: these are substring-smoke metrics from the local SGT scorer, not the official LLM-judge evaluation.

| suite | metric | examples | dense % | frontier % | delta | logical MB savings % | physical MB savings % | active % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGT short | Once | 16 | 0.00 | 0.00 | 0.00 | 44.5 | 33.3 | 68.0 |
| SGT short | Range | 16 | 15.62 | 15.62 | 0.00 | 44.5 | 33.3 | 68.0 |
| SGT short | Periodic | 16 | 1.81 | 1.11 | -0.69 | 44.5 | 33.3 | 68.0 |
| SGT short | Completion | 16 | 100.00 | 99.16 | -0.84 | 44.5 | 33.3 | 68.0 |
| SGT long | Once | 6 | 76.67 | 76.67 | 0.00 | 54.5 | 36.9 | 83.2 |
| SGT long | Range | 6 | 30.28 | 44.44 | 14.17 | 54.5 | 36.9 | 83.2 |
| SGT long | Periodic | 6 | 7.94 | 7.94 | 0.00 | 54.5 | 36.9 | 83.2 |
| SGT long | Completion | 6 | 74.83 | 75.00 | 0.17 | 54.5 | 36.9 | 83.2 |
