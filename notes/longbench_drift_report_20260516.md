# LongBench Drift Report

Compared `longbench_v2_hf_result/dense_lbv2_short_easy_n64_temp0/predictions.jsonl` vs `longbench_v2_hf_result/frontier_lbv2_short_easy_n64_temp0/predictions.jsonl` over `59` common rows.

Status counts in reported set: {"gained_correct": 4, "lost_correct": 1}

| id | status | dense | frontier | mean logit relL2 | max logit relL2 | min logit cos | mean hidden relL2 | max hidden relL2 | min hidden cos | KL mean/max | top1 | choice top |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 66fa7f1dbb02136c067c6e6b | gained_correct | None / False | D / True | 0.0536 | 0.0877 | 0.9962 | 0.0596 | 0.1098 | 0.9940 | 0.0050/0.0464 | 0.9375 | 0.9375 |
| 66ec3d1d821e116aacb1c622 | gained_correct | D / False | B / True | 0.0663 | 0.1455 | 0.9894 | 0.0804 | 0.2058 | 0.9789 | 0.0091/0.1318 | 0.9375 | 0.8750 |
| 66ebd49a5a08c7b9b35e0550 | lost_correct | B / True | D / False | n/a | n/a | n/a | n/a | n/a | n/a | n/a/n/a | n/a | n/a |
| 66eefa2f821e116aacb2284f | gained_correct | None / False | A / True | 0.0532 | 0.1174 | 0.9934 | 0.0620 | 0.1040 | 0.9946 | 0.0027/0.0136 | 1.0000 | 0.8125 |
| 66f58d6c821e116aacb33e76 | gained_correct | A / False | C / True | n/a | n/a | n/a | n/a | n/a | n/a | n/a/n/a | n/a | n/a |
