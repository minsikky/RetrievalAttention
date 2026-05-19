| label | kind | mode | quality | n | sec/ex | step MB/hq | selector | exact KV | tail | update | selected | passthrough | readiness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_ruler_ctx8192_n4_niah_single_1 | ruler | dense_batched | 100.00 | 4 | 8.60 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ok |
| frontier_ruler_ctx8192_n4_niah_single_1 | ruler | pagedpq_batched | 100.00 | 4 | 37.94 | 2.452 | 2.150 | 0.271 | 0.031 | 0.000206 | 555.1 | 0 | ok |
| dense_ruler_ctx8192_n4_niah_multikey_2 | ruler | dense_batched | 100.00 | 4 | 8.97 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ok |
| frontier_ruler_ctx8192_n4_niah_multikey_2 | ruler | pagedpq_batched | 100.00 | 4 | 40.58 | 2.434 | 2.132 | 0.272 | 0.031 | 0.000207 | 556.5 | 0 | ok |
| dense_ruler_ctx8192_n4_vt | ruler | dense_batched | 100.00 | 4 | 3.59 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ok |
| frontier_ruler_ctx8192_n4_vt | ruler | pagedpq_batched | 100.00 | 4 | 32.08 | 2.479 | 2.177 | 0.272 | 0.030 | 0.000207 | 556.4 | 0 | ok |
| dense_ruler_ctx8192_n4_fwe | ruler | dense_batched | 75.00 | 4 | 4.87 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ok |
| frontier_ruler_ctx8192_n4_fwe | ruler | pagedpq_batched | 75.00 | 4 | 33.20 | 2.474 | 2.173 | 0.271 | 0.030 | 0.000159 | 554.6 | 0 | ok |
| dense_lbv2_short_easy_n64_l8192 | longbench-v2 | dense | 35.59 | 59 | 7.43 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ok |
| frontier_lbv2_short_easy_n64_l8192 | longbench-v2 | pagedpq | 40.68 | 59 | 36.70 | 2.555 | 2.252 | 0.271 | 0.032 | 0.000144 | 554.6 | 0 | ok |
