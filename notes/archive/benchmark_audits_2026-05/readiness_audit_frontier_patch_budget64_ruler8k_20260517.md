| label | kind | mode | quality | n | sec/ex | step MB/hq | selector | exact KV | tail | update | selected | passthrough | readiness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_ruler_ctx8192_n4_niah_single_1 | ruler | dense_batched | 100.00 | 4 | 8.72 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ok |
| frontier_ruler_ctx8192_n4_niah_single_1 | ruler | pagedpq_batched | 100.00 | 4 | 82.71 | 2.399 | 2.097 | 0.271 | 0.031 | 0.000206 | 555.1 | 0 | ok |
| dense_ruler_ctx8192_n4_niah_multikey_2 | ruler | dense_batched | 100.00 | 4 | 8.67 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ok |
| frontier_ruler_ctx8192_n4_niah_multikey_2 | ruler | pagedpq_batched | 100.00 | 4 | 78.16 | 2.382 | 2.079 | 0.272 | 0.031 | 0.000207 | 556.5 | 0 | ok |
| dense_ruler_ctx8192_n4_vt | ruler | dense_batched | 100.00 | 4 | 3.93 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ok |
| frontier_ruler_ctx8192_n4_vt | ruler | pagedpq_batched | 100.00 | 4 | 76.06 | 2.425 | 2.123 | 0.272 | 0.030 | 0.000207 | 556.4 | 0 | ok |
| dense_ruler_ctx8192_n4_fwe | ruler | dense_batched | 75.00 | 4 | 5.11 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ok |
| frontier_ruler_ctx8192_n4_fwe | ruler | pagedpq_batched | 83.33 | 4 | 75.53 | 2.420 | 2.119 | 0.271 | 0.030 | 0.000159 | 554.6 | 0 | ok |
