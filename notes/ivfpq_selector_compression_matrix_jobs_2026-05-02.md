# IVF-PQ Selector/Compression Matrix Jobs

Timestamp: 2026-05-02 20:06 EDT

Trace:

- `attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q36_graphall_s16.npz`

Shared settings:

- Decode cutoffs: `500,1000,2000,4000,8000,16000,32000,64000,128000`
- Mass targets: `0.95,0.98`
- Query positions per cutoff: `4`
- Static prefix/suffix: `128/128`
- IVF fixed nprobes: `16,32,64,128`
- Families: `pqcache,ivfpq`
- Graph methods disabled for this matrix
- PQ-logit rows enabled for fixed-nprobe IVF rows

Jobs:

- `snapshot`: job `49223173`, output `attention_efficiency_result/ivfpq_matrix_g131072_suffix128_snapshot_v1`
- `frozen_append`: job `49223182`, output `attention_efficiency_result/ivfpq_matrix_g131072_suffix128_frozen_append_v1`
- `online_centroid`: job `49223184`, output `attention_efficiency_result/ivfpq_matrix_g131072_suffix128_online_centroid_v1`
- `periodic_rebuild`: job `49223185`, output `attention_efficiency_result/ivfpq_matrix_g131072_suffix128_periodic_rebuild_v1`

Rows to compare:

- Adaptive selector/target mass: `ivfpq_global_pq_oracle_c128*`
- Fixed-nprobe selection with exact K/V: `ivfpq_global_pq_fixed_exactkv_c128_n{16,32,64,128}*`
- Fixed-nprobe selection with PQ logits: `ivfpq_global_pq_fixed_pqlogit_c128_n{16,32,64,128}*`
- Baselines emitted in the same runs: `dense_oracle`, `retroinfer`, `pqcache_m2_b6`

Interpretation target:

- Selection axis: fixed-nprobe exact-K/V rows show oracle mass inside IVF-selected buckets/candidates.
- Compression axis: fixed-nprobe PQ-logit rows reuse the same candidates but replace dynamic-token QK logits with PQ approximate logits, so output cosine degradation is primarily compression/scoring error rather than candidate reachability.
