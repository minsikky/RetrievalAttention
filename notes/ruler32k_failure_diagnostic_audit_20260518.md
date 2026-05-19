| label | kind | mode | quality | n | sec/ex | step MB/hq | selector | exact KV | tail | update | selected | passthrough | readiness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frontier_ruler_ctx32768_n4_niah_multikey_2_b2048_tail | ruler | pagedpq_batched | 75.00 | 4 | 46.43 | 3.218 | 1.055 | 1.843 | 0.320 | 0.031528 | 3774.3 | 128 | ok |
| frontier_ruler_ctx32768_n4_fwe_b2048_tail | ruler | pagedpq_batched | 91.67 | 4 | 22.95 | 3.023 | 1.020 | 1.694 | 0.310 | 0.078022 | 3470.0 | 128 | ok |
| frontier_ruler_ctx32768_n4_niah_multikey_2_b768_no_tail | ruler | pagedpq_batched | 50.00 | 4 | 39.86 | 2.273 | 1.055 | 1.218 | 0.000 | 0.016708 | 2494.3 | 128 | selected-v-not-compressed |
| frontier_ruler_ctx32768_n4_fwe_b768_no_tail | ruler | pagedpq_batched | 91.67 | 4 | 43.13 | 2.089 | 1.020 | 1.069 | 0.000 | 0.041348 | 2190.0 | 128 | selected-v-not-compressed |
