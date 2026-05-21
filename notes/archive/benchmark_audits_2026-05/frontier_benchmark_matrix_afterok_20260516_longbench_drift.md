# LongBench Drift Report

Compared `longbench_v2_hf_result/frontier_benchmark_matrix_readiness_matrix_20260516_afterok4/dense_lbv2_short_easy_n64_l8192/predictions.jsonl` vs `longbench_v2_hf_result/frontier_benchmark_matrix_readiness_matrix_20260516_afterok4/frontier_lbv2_short_easy_n64_l8192/predictions.jsonl` over `59` common rows.

Status counts in reported set: {"gained_correct": 3}

| id | status | dense | frontier | mean logit relL2 | max logit relL2 | min logit cos | mean hidden relL2 | max hidden relL2 | min hidden cos | KL mean/max | top1 | choice top |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 66fa7f1dbb02136c067c6e6b | gained_correct | None / False | D / True | 0.0670 | 0.1563 | 0.9878 | 0.0726 | 0.1876 | 0.9824 | 0.0069/0.1143 | 0.9844 | 0.9453 |
| 66ec3d1d821e116aacb1c622 | gained_correct | D / False | B / True | 0.0883 | 0.4686 | 0.8985 | 0.1007 | 0.4376 | 0.9047 | 0.0270/1.5467 | 0.9844 | 0.9219 |
| 66f58d6c821e116aacb33e76 | gained_correct | A / False | C / True | 0.0776 | 0.3227 | 0.9487 | 0.0884 | 0.4609 | 0.8936 | 0.0052/0.1215 | 0.9766 | 0.9219 |
