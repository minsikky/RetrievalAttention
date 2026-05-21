# LongBench Drift Report

Compared `longbench_v2_hf_result/dense_lbv2_short_easy_n64_temp0/predictions.jsonl` vs `longbench_v2_hf_result/frontier_lbv2_short_easy_n64_temp0/predictions.jsonl` over `59` common rows.

Status counts in reported set: {"gained_correct": 3, "preserved_correct": 1, "preserved_wrong": 1}

| id | status | dense | frontier | mean logit relL2 | max logit relL2 | min logit cos | mean hidden relL2 | max hidden relL2 | min hidden cos | KL mean/max | top1 | choice top |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 66f36490821e116aacb2cc22 | preserved_correct | D / True | D / True | 0.0715 | 0.1391 | 0.9903 | 0.0759 | 0.1274 | 0.9919 | 0.0006/0.0082 | 1.0000 | 1.0000 |
| 66f78ecfbb02136c067c2f12 | preserved_wrong | A / False | A / False | 0.0483 | 0.0859 | 0.9966 | 0.0613 | 0.0968 | 0.9953 | 0.0003/0.0026 | 1.0000 | 1.0000 |
| 66fa7f1dbb02136c067c6e6b | gained_correct | None / False | D / True | 0.0536 | 0.0877 | 0.9962 | 0.0596 | 0.1098 | 0.9940 | 0.0050/0.0464 | 0.9375 | 0.9375 |
| 66ec3d1d821e116aacb1c622 | gained_correct | D / False | B / True | 0.0663 | 0.1455 | 0.9894 | 0.0804 | 0.2058 | 0.9789 | 0.0091/0.1318 | 0.9375 | 0.8750 |
| 66eefa2f821e116aacb2284f | gained_correct | None / False | A / True | 0.0532 | 0.1174 | 0.9934 | 0.0620 | 0.1040 | 0.9946 | 0.0027/0.0136 | 1.0000 | 0.8125 |
