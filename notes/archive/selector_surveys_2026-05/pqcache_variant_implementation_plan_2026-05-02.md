# PQCache Variant Implementation Plan

Source decks:
- `notes/binary_gated_pqcache_research_proposal_2026-05-02.pptx`
- `notes/dictionary_backed_paged_pqcache_research_proposal_2026-05-02.pptx`

Extracted text:
- `notes/binary_gated_pqcache_research_proposal_2026-05-02.extracted_text.md`
- `notes/dictionary_backed_paged_pqcache_research_proposal_2026-05-02.extracted_text.md`

## Current Baseline

Current proxy PQCache in `benchmark/attention_efficiency_threeway_eval.py`:
- Builds a global PQ codebook over dynamic keys.
- Stores one PQ code per token.
- For each decode query, scores all PQ codes with a query-codebook LUT.
- Ranks all dynamic tokens approximately.
- Exact-reads the prefix needed to reach each target mass.

This is strong in the current results, but still performs a linear full scan over all PQ codes.

## Variant 1: Binary-Gated PQCache

Purpose:
- Reduce full PQ-code scoring by using cheaper binary/hash codes for coarse recall.

Proxy algorithm:
1. Build binary sidecar codes for dynamic keys using random projections or sign bits.
2. Build the same global PQ index as PQCache.
3. For each query, compute query binary code.
4. Full-scan binary sidecar with XNOR/popcount or collision count.
5. Keep top `M` candidates or candidates above a collision threshold.
6. PQ-rerank only those candidates.
7. Exact-read ranked candidates until target mass is reached.

Cost model:
- Binary sidecar scan: `N * binary_bits / 8`.
- PQ candidate scan: `M * pq_code_bytes`.
- PQ LUT score setup: `subvecs * centroids_per_subvec * subdim * score_bytes`.
- Final exact K/V reads: selected tokens only.

Important limitation:
- This does not fully avoid a linear scan. It replaces full PQ scan with a cheaper full binary scan.
- This is still useful because it tests whether binary metadata can preserve enough recall while greatly reducing PQ-code traffic.

Evaluator names:
- `binary_gated_pqcache_b{bits}_m{M}`
- `binary_gated_pqcache_adaptive`

Primary knobs:
- `binary_bits`: 64, 128, 256
- `candidate_budget`: fixed M or adaptive ladder
- `hash_projection`: random sign initially
- `pq_subvecs`, `pq_subbits`, `pq_kmeans_iters`

## Variant 2: IVF-PQ / Hierarchical PQCache

Purpose:
- Actually avoid scanning all token codes by routing to a subset of buckets first.

Proxy algorithm:
1. Build coarse clusters over dynamic keys.
2. Store token IDs and PQ codes inside each inverted list.
3. For each query, score all coarse centroids.
4. Visit top `nprobe` coarse buckets.
5. Scan PQ codes only inside visited buckets.
6. Exact-read ranked candidates until target mass is reached.
7. Adaptive mode increases `nprobe` until target mass is reached or all buckets are visited.

Cost model:
- Coarse centroid scan: `C * head_dim * score_bytes`.
- Selected PQ-code scan: `N_selected * pq_code_bytes`.
- PQ LUT setup: same as PQCache.
- Final exact K/V reads: selected tokens only.
- Optional metadata: list offsets and token IDs for selected buckets.

Why this is the first target:
- It directly answers whether PQCache can move below linear full-code scanning.
- It gives an interpretable curve: scan 5%, 10%, 20%, ... of PQ codes vs recovered mass.
- It is the cleanest comparison against full PQCache, RetroInfer, SparQ, and oracle.

Evaluator names:
- `ivfpq_c{clusters}_p{nprobe}`
- `ivfpq_adaptive_c{clusters}`

Primary knobs:
- `coarse_clusters`: 64, 128, 256, 512
- `nprobe`: fixed ladder or adaptive
- `cluster_training`: simple k-means initially
- `pq_scope`: global PQ first; page/local PQ later

## Variant 3: Dictionary-Backed Paged PQ

Purpose:
- Address long-decode drift and metadata growth, not just per-query search cost.

Proxy algorithm:
1. Divide dynamic tokens into pages, e.g. 128, 256, or 512 tokens.
2. Keep the active page exact while it is filling.
3. When a page seals, train local PQ codebooks per subspace on that page.
4. Intern local centroids into a shared per-subspace dictionary.
5. Store per-page alias tables mapping local code IDs to dictionary IDs.
6. Score tokens through dictionary LUT + alias lookup.
7. Compare against:
   - global PQCache,
   - naive page-local PQ,
   - dictionary-backed page PQ.

Cost model:
- Token code reads: linear in tokens scanned.
- Alias reads: selected/scanned token-subspace alias lookup.
- Dictionary LUT setup: dictionary centroids scored once per query/subspace.
- Metadata accounting:
  - dictionary centroid bytes,
  - page alias bytes,
  - private centroid bytes,
  - outlier tier bytes.

Important limitation:
- By itself, dictionary-backed page PQ may not reduce per-query scans below PQCache.
- Its value is better long-decode adaptation and lower codebook metadata than naive page-local PQ.
- To move closer to oracle in MB-vs-decode, combine it with binary gating or IVF routing.

Evaluator names:
- `page_pq_p{page}_m{subvecs}_b{subbits}`
- `dict_page_pq_p{page}_m{subvecs}_b{subbits}_t{threshold}`
- combined later: `ivf_dict_pq` or `binary_dict_pq`

Primary knobs:
- `page_size`: 128, 256, 512
- `merge_threshold`: centroid distance/SSE threshold
- `alias_bits`: 4, 5, 6, 8
- `private_centroids`: disabled/limited/unlimited

## Recommended Implementation Order

1. Implement `ivfpq_adaptive`.
   - Fastest path to test the main hypothesis: avoid full PQ scan and approach oracle.
   - Add per-query `scanned_code_tokens_mean` and `selected_bucket_count_mean` to raw rows.

2. Implement `binary_gated_pqcache_adaptive`.
   - Reuse existing MagicPIG random projection code plus existing PQCache code.
   - This is a low-effort bridge between hash retrieval and PQ reranking.

3. Implement `page_pq` and `dict_page_pq`.
   - Start without routing; compare quality/metadata versus global PQ.
   - Then combine with IVF/binary routing if it improves quality or drift.

4. Plot against existing baselines.
   - `Oracle`
   - `PQCache full scan`
   - `IVF-PQ adaptive`
   - `Binary-gated PQCache adaptive`
   - `Dictionary-backed paged PQ`
   - `RetroInfer`
   - `SparQ`

## Success Criteria

Algorithmic:
- For target mass 0.95 and 0.98, the new variants should sit between oracle and PQCache full scan in MB-vs-decode.
- The key signal is lower estimated MB than PQCache at matched mass.

Long decode:
- Dictionary-backed page PQ should reduce quality drift versus global PQ or reduce metadata versus naive page-local PQ.
- Dictionary growth should be sublinear or at least much smaller than naive per-page codebooks.

Hardware motivation:
- Surviving variants should expose primitives that GPUs handle poorly:
  - sparse bucket/page selection,
  - top-k/compaction,
  - alias/dictionary lookup,
  - sparse V page fetch,
  - optional binary popcount scan.
