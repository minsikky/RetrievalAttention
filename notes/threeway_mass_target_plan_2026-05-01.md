# Three-Way Mass-Target Proxy Plan

Goal: compare decode-time algorithmic work needed to reach dense-attention mass targets for three methods on real QKV traces.

## Methods

- `retroinfer`: score all visible centroids, rank clusters exactly, then add top clusters until the target mass is represented. This preserves RetroInfer's linear centroid-routing cost. If `--retro_exact_clusters > 0`, clusters after that cap are modeled as estimation-zone clusters using centroid logits plus `value_sum / cluster_size` instead of exact K/V reads.
- `retrievalattention`: build a Roar-style Q-K projected token graph from saved graph queries, then traverse the graph at decode. Per-query cost is scored frontier K vectors plus edge/offset metadata plus final exact K/V reads.
- `hybrid_centroid_graph`: build RetroInfer centroids, then build a Roar-style Q-C projected graph over those centroids. Decode traverses centroids instead of scoring every centroid. This tests whether graph traversal removes RetroInfer's O(number_of_centroids) routing step.

## Cost Model

- Score reads: vectors read to score candidates against the decode query. For RetroInfer/hybrid this means centroids; for RetrievalAttention this means token K vectors.
- Final K/V reads: exact token K/V vectors used in sparse attention.
- Estimation reads: value-sum vectors for represented but non-exact clusters.
- Graph metadata reads: neighbor IDs and row offsets read during traversal.
- Reported `estimated_mb` is the sum of these byte terms. Graph construction is treated as prefill/offline cost and is not charged per decode query.

## Current Fidelity Boundary

- The new evaluator requires `graph_queries` and `graph_positions` in the NPZ so RA and hybrid graph construction is Q-conditioned rather than K-K.
- `graph_query_scope=all` plus causal filtering lets the graph include generated-token query rows up to the evaluated cutoff. This approximates online long-decode graph extension from the real decode trace.
- The online update is still an oracle-style proxy: it uses saved Q-K/Q-C top-k rows, not the exact runtime retrieval provenance path from `cache_hub/retrievalattention_cache.py`.
