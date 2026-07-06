# KV-Cache Compression Coverage

Source checklist: October2001/Awesome-KV-Cache-Compression.

Purpose: track which existing-method families are represented in our saved-trace MB-vs-relL2 plot, and which require a different evaluation path.

## Trace-Evaluable Now

These can be evaluated on the saved layer-16 Q/K/V trace because they only require modifying or approximating the current layer's K/V cache before attention.

| family | representative papers / methods | local proxy | status |
| --- | --- | --- | --- |
| Scalar KV quantization | KIVI, KVQuant, SKVQ, QAQ-style | `kivi_b*_g*_w*`, `kvquant_like`, `per_token_kv` | KIVI uses the official quantization layout; others are proxies |
| Vector/PQ quantization | PQCache, CommVQ-style, TurboQuant-style | `pq_like`, `tq`, `tqprod` | completed |
| Pruning / eviction / sparse retention | StreamingLLM, H2O-style retention, Scissorhands-style persistence, SnapKV/PyramidKV, KVzip, RocketKV prompt-retention path | `recent_k*`, `sink_recent_*`, `l2ret_*`, `h2o_k*`, `snapkv_k*`, `kvzip_k*`, `rocket_snap_k*`, `expected_attn_*`, `critical_snap_*`, `chunk_snap_*`, `keydiff_*`, `tova_*`, `cur_*`, `lagkv_*`, `compactor_*` | completed, trace scorer proxies |
| Importance-aware mixed precision | ZipCache / No Token Left Behind-style salient exact + compressed rest | `salient_quant_*` | completed |
| Low-rank compression | Palu, LoRC, OjaKV-style | `lowrank_svd_*` | completed |
| GEAR-style residual compression | GEAR quantization plus low-rank residual plus sparse residual outliers | `gear_like_b*_r*_sp*_w*` | completed, reference-grounded proxy |
| Additive/residual VQ | CommVQ-style residual codebook stages | `commvq_like_m*b*_w*` | completed, locally trained codebooks; official released key codebooks still require all-KV-head integration |
| Sparse dictionary coding | Lexico-style sparse dictionary representation | `lexico_like_d*a*_w*` | completed, greedy matching-pursuit proxy; exact OMP reference path still TODO |
| Progressive mixed precision | PM-KVQ sink/window + lower-bit middle cache | `pmkvq_like_b*_s*_g*_w*` | completed, reference-grounded trace implementation |
| Channel-promoted 2-bit quantization | Kitty dynamic channel-wise precision boost | `kitty_like_k*v*_p*_pb*_buf*_s*` | completed, reference-grounded trace implementation |
| Outlier-immunized/product quantized KV | MILLION DynamicPQCache with PQ codes plus exact residual cache | `million_like_m*b*_w*` | completed, reference-grounded trace implementation using locally trained PQ |
| Weighted KV merging | ZeroMerge/MergeKV top cache + dense merged residual + tail; CaM merge-before-prune | `zeromerge_like_k*_tail*_dense*_obs*_ker*`, `cam_like_k*_merge*_obs*_ker*` | completed, snapshot implementations |
| Mean-centered scalar quantization | TaDA-style mean-centered decoding quantization | `tada_like_b*_g*_w*` | completed, trace proxy |
| Transform coding | KVTC / FreqKV-style channel decorrelation and coefficient coding | `kvtc_like_b*_r*_w*`, `freqkv_like_b*_r*_w*` | completed, PCA/DCT trace proxies |
| Tiered / adaptive precision | MiniKV, KVTuner, TailorKV, adaptive bit-allocation ideas | `tiered_quant_l*_m*_h*_hi*_mid*_w*` | completed, token-score tier proxy |
| Sparse channel coding | LOOKAT-style unstructured per-token sparse channel cache | `lookat_like_p*_b*_mean*_w*` | completed, sparse-channel trace proxy |

## Not Faithfully Covered By Single-Layer Trace

These need a model-level or multi-layer benchmark rather than a single saved layer-16 Q/K/V trace.

| family | why not trace-faithful |
| --- | --- |
| Cross-layer sharing | Needs multiple layer caches and changed layer routing/sharing semantics. A single layer trace cannot measure layer reuse error. |
| Learned/model-transformed methods | Requires changed weights, training/distillation, or modified architecture. A post-hoc Q/K/V trace cannot reproduce it. |
| Prompt compression | Changes the input token sequence before prefill. It is not a decode KV-cache approximation on a fixed trace. |
| Multimodal-specific pruning | Requires vision tokens/modality metadata and VLM tasks. |
| Faithful H2O/SnapKV/AdaKV/DBudgetKV online policies | Need the full sequence of model attention scores across layers/heads/tokens to update cache state. Our proxies only test representative retention heuristics. |
| TriAttention / frequency-stat pruning | Needs calibrated pre-RoPE frequency statistics and model-level cache compaction. A post-RoPE layer trace can only provide a weak proxy. Evaluate through the cloned implementation on task benchmarks if it becomes a serious comparator. |
| Learned pruning / score models | KVzap, DMS, and similar approaches require trained scoring heads or hidden-state features. A single Q/K/V trace does not contain the learned scorer input/output contract. |
| R-KV recurrent redundancy-aware compression | Faithful R-KV updates a bounded cache online using attention importance and key-key redundancy. A one-shot layer snapshot would miss the recurrent compressed-cache state and the O(N²) redundancy step is not feasible as a 128k trace curve. |
| Cross-head policies | DuoAttention, HeadKV, AdaKV/PyramidKV budget allocation, and layer/head routers need all-layer/all-head policy state. Single-head trace points can only approximate their scoring, not the deployed policy. |
| Verification / lossless wrappers | VeriCache-style verification needs model-level speculative/verification control flow and task outputs, not just layer output reconstruction. |
| Shared-cache / serving systems | PolyKV, KVServe, SparKV, and similar systems optimize multi-request sharing, disaggregated transfer, or device/cloud placement. They are serving policies, not per-layer single-query approximation curves. |
| Learned hashing / changed attention paths | DASH-KV-style asymmetric/deep hashing and KV-Direct-style residual-stream alternatives need learned hash functions or changed model execution. A post-hoc Q/K/V trace cannot faithfully reproduce them. |
| Multimodal hybrids | HybridKV, MEDA, and VLM-specific cache methods require modality labels, vision-token structure, and multimodal benchmarks. |

## Reference Repositories Cloned

Local references live under `third_party/baselines/`:

| repo | used for |
| --- | --- |
| `GEAR` | quant + low-rank + sparse residual cost/structure |
| `CommVQ` | residual VQ stages and official released codebook format |
| `lexico` | dictionary atoms, OMP/CSR representation |
| `kvpress` | SnapKV/KVzip/KVzap/DMS implementation details |
| `KVzip` | attention-score based KV eviction |
| `RocketKV` | SnapKV-style prompt compression and decode approximation details |
| `PM-KVQ` | progressive sink/window/middle scalar quantization |
| `Kitty` | channel-wise precision promotion for K and KIVI-style V quantization |
| `MILLION` | PQ code cache plus exact residual cache |
| `ZeroMerge` | weighted merge-cache update and attention weighting |
| `triattention` | frequency-stat pruning; model-level only for faithful eval |
| `KVCache-Factory` | unified pruning/compression reference implementations |
| `R-KV` | redundancy-aware online compression; documented as model-level only for faithful eval |

## Interpretation Rule

The plot should label these as paper-inspired proxies, not faithful reimplementations. The goal is to estimate whether a family can plausibly occupy the `0-5 MB/head-query` frontier before spending time on full implementation or task-level evaluation.
