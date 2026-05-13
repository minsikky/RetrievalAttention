# Selector-Eval Cost Model

This file defines the logical memory-traffic accounting used by `benchmark/selector_eval`.

## Phases

- `selector`: query-time index/router/scoring reads needed to choose candidate tokens.
- `exact_attention`: exact K/V reads for the final selected token set.
- `online_update`: incremental index maintenance after generated tokens leave the static suffix.

The unit-explicit cost columns are:

```text
selector_MB_per_query          = selector traffic for one query at this decode length
exact_KV_MB_per_query          = exact K/V traffic for one query at this decode length
online_update_cumulative_MB    = cumulative update traffic up to this decode length
online_update_MB_per_token     = online_update_cumulative_MB / decode_length
step_MB_per_query              = selector_MB_per_query + exact_KV_MB_per_query + online_update_MB_per_token
```

Legacy compatibility aliases are still emitted:

```text
selector_MB = selector_MB_per_query
exact_KV_MB = exact_KV_MB_per_query
query_MB = selector_MB_per_query + exact_KV_MB_per_query
online_update_MB = online_update_cumulative_MB
total_MB = step_MB_per_query
```

Do not use legacy aliases in new tables when unit-explicit columns are available.

## Comparison Modes

Maintain two explicit comparison modes:

- Snapshot/query-only comparison: inspect `selector_MB_per_query` and `exact_KV_MB_per_query` separately. This assumes the index already exists and asks how expensive one query is.
- Online/realistic comparison: use `step_MB_per_query`, but only for methods whose online-update model is implemented with comparable assumptions.

Do not mix these interpretations in one ranking. In particular, a method with missing `online_update_cumulative_MB` can look better in `step_MB_per_query` for the wrong reason.

Current online-update status:

- `gated_paged_pq_snapshot`: query-only view with page/PQ maintenance suppressed.
- `gated_paged_pq_online`: modeled with page sealing plus router/PQ maintenance.
- `paged_local_pq_snapshot`: query-only view with page/PQ maintenance suppressed.
- `paged_local_pq_online`: modeled with page sealing plus local PQ maintenance.
- `pqcache_full_scan_snapshot`: query-only view with PQ build traffic suppressed.
- `pqcache_full_scan_online_proxy`: partial framework-port proxy; current-context PQ build is charged, optimized online maintenance is not implemented.
- `ivfpq_periodic_rebuild`: modeled but can be dominated by rebuild traffic.
- `retroinfer_style`: snapshot/query proxy only.
- `retroinfer_online_proxy`: decode-segment clustering/update traffic is charged as a memory proxy, but this is not a full RetroInfer implementation.
- `sparq_r16`: no persistent index update in this selector proxy.
- `magicpig`: hash-sidecar query proxy; hash index update is not yet modeled.
- `retrievalattention_graph`: traversal proxy; production-faithful graph update/build traffic is not yet modeled.

## Paged / Gated Paged PQ

Selector reads:

- `router_groups`: fixed-size coarse routing groups.
- `router_postings`: references from selected router groups to page-local prototypes.
- `page_pq_codebooks`: page-local PQ codebooks for candidate pages.
- `page_pq_codes`: PQ codes for candidate rows.

Exact-attention reads:

- Full-precision K/V for selected tokens only.

Online-update reads/writes:

- `page_build_keys`: generated K vectors needed to seal a page.
- `page_prototypes`, `page_proto_postings`: page-local routing metadata.
- `page_pq_codebooks`, `page_pq_codes`, `page_meta`: page-local PQ index state.
- `router_group`, `router_postings`: global routed-group maintenance.

Excluded from memory MB:

- `page_pq_build_work`
- `page_proto_build_work`

Those are compute-work proxies from the legacy simulator, not memory reads.

## RetroInfer-Style

Selector reads:

- all chunk centroids
- chunk range metadata

Exact-attention reads:

- full K/V for member tokens from selected chunks.

Online-update in `retroinfer_online_proxy`:

- newly visible decode segments are charged for key reads, centroid writes, cluster metadata/posting writes, and value-sum writes.
- this still omits full wave-buffer/cache behavior and should be interpreted as an online memory proxy, not a faithful RetroInfer reproduction.

## PQCache Full Scan

Selector reads:

- PQ codebooks
- all PQ codes in the current context

Online-update:

- `pqcache_full_scan_snapshot` suppresses PQ construction traffic and reports query-only cost.
- `pqcache_full_scan_online_proxy` records PQ construction for the current context as `online_update`; replacing this with a maintained online index should preserve the runner/schema.

## RetrievalAttention Graph

Selector reads:

- key vectors for scored graph nodes
- graph offsets
- graph edge indices

Online-update:

- causal trace replay builds the graph internally, but graph construction traffic is not yet production-cache parity.

## Interpretation Rule

Use separated `selector_MB_per_query` and `exact_KV_MB_per_query` for selector quality/efficiency tables. Use `online_update_cumulative_MB` and `online_update_MB_per_token` to discuss maintenance overhead. Avoid making hardware claims from legacy `total_MB` alone.
