# Benchmark Runtime Projection

Scope: current selected matrix from `scripts/submit_frontier_benchmark_matrix.sh`: four RULER ctx8k tasks with 4 samples each, plus LongBench-v2 short/easy n=64 at max input 8192, dense and frontier.

| source | mode | examples used for estimate | sec/example | projected job wall time |
| --- | --- | ---: | ---: | ---: |
| RULER ctx8k mean over four tasks | dense_batched | 16 | 6.62 | 0.4 min/task |
| RULER ctx8k mean over four tasks | pagedpq_batched | 16 | 39.77 | 2.7 min/task |
| LongBench-v2 short/easy | dense | 59 | 7.38 | 7.9 min for n=64 |
| LongBench-v2 short/easy | pagedpq | 59 | 41.21 | 44.0 min for n=64 |

| projection | wall time | interpretation |
| --- | ---: | --- |
| Serial execution of all 10 jobs | 64.2 min | Upper bound if run one after another. |
| Parallel Slurm wave, ignoring queue delay | 44.0 min | Dominated by frontier LongBench; fits the 2h wrapper limit based on current evidence. |
| Longest RULER frontier task job | 2.9 min | Fits the 1h RULER wrapper limit. |

Caveat: this projection is for the selected ctx8k/short-easy validation matrix, not a full LongBench/RULER suite or longer context sweep. Queue wait from account GRES is excluded.
