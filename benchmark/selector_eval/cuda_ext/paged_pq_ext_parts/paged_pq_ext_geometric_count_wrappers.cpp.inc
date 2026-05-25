torch::Tensor gqa_decode_geometric_accept_counts(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(dense_pq_scores.is_cuda(), "dense_pq_scores must be CUDA");
  TORCH_CHECK(value_codebooks.is_cuda(), "value_codebooks must be CUDA");
  TORCH_CHECK(value_codes.is_cuda(), "value_codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(is_float_like(values), "values must be float32, float16, or bfloat16");
  TORCH_CHECK(dense_pq_scores.scalar_type() == torch::kFloat32, "dense_pq_scores must be float32");
  TORCH_CHECK(value_codebooks.scalar_type() == torch::kFloat32, "value_codebooks must be float32");
  TORCH_CHECK(
      value_codes.scalar_type() == torch::kLong || value_codes.scalar_type() == torch::kUInt8,
      "value_codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(ranked_scores.scalar_type() == torch::kFloat32, "ranked_scores must be float32");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(dense_pq_scores.dim() == 2, "dense_pq_scores shape must be [heads, pages * page_size]");
  TORCH_CHECK(value_codebooks.dim() == 5, "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, selected]");
  TORCH_CHECK(ranked_scores.dim() == 2, "ranked_scores shape must be [heads, selected]");
  TORCH_CHECK(keys.size(0) == values.size(0), "key/value kv-head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "key/value token count mismatch");
  TORCH_CHECK(keys.size(2) == values.size(2), "key/value dim mismatch");
  TORCH_CHECK(value_codebooks.size(0) == value_codes.size(0), "value kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value/key kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(1) == value_codes.size(1), "value page count mismatch");
  TORCH_CHECK(value_codebooks.size(2) == value_codes.size(3), "value subvec count mismatch");
  TORCH_CHECK(value_codes.size(2) == page_size, "value code page_size mismatch");
  TORCH_CHECK(value_codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(dense_pq_scores.size(0) == queries.size(0), "dense score head count mismatch");
  TORCH_CHECK(dense_pq_scores.size(1) == value_codebooks.size(1) * page_size, "dense score token count mismatch");
  TORCH_CHECK(queries.size(0) == ranked_tokens.size(0), "ranked token head count mismatch");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(queries.size(1) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(queries.size(1) == value_codebooks.size(2) * value_codebooks.size(4), "value codebook dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(granularity > 0, "granularity must be positive");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_geometric_accept_counts_cuda(
      queries.contiguous(),
      keys,
      values,
      dense_pq_scores.contiguous(),
      value_codebooks.contiguous(),
      value_codes.contiguous(),
      page_starts.contiguous(),
      ranked_tokens.contiguous(),
      ranked_scores.contiguous(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
	      scale);
}

torch::Tensor gqa_decode_geometric_accept_counts_vpq(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    int64_t exact_value_top,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(dense_pq_scores.is_cuda(), "dense_pq_scores must be CUDA");
  TORCH_CHECK(value_codebooks.is_cuda(), "value_codebooks must be CUDA");
  TORCH_CHECK(value_codes.is_cuda(), "value_codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(is_float_like(values), "values must be float32, float16, or bfloat16");
  TORCH_CHECK(dense_pq_scores.scalar_type() == torch::kFloat32, "dense_pq_scores must be float32");
  TORCH_CHECK(value_codebooks.scalar_type() == torch::kFloat32, "value_codebooks must be float32");
  TORCH_CHECK(
      value_codes.scalar_type() == torch::kLong || value_codes.scalar_type() == torch::kUInt8,
      "value_codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(ranked_scores.scalar_type() == torch::kFloat32, "ranked_scores must be float32");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(dense_pq_scores.dim() == 2, "dense_pq_scores shape must be [heads, pages * page_size]");
  TORCH_CHECK(value_codebooks.dim() == 5, "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, selected]");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(keys.size(0) == values.size(0), "key/value kv-head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "key/value token count mismatch");
  TORCH_CHECK(keys.size(2) == values.size(2), "key/value dim mismatch");
  TORCH_CHECK(value_codebooks.size(0) == value_codes.size(0), "value kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value/key kv-head count mismatch");
  TORCH_CHECK(value_codes.size(2) == page_size, "value code page_size mismatch");
  TORCH_CHECK(dense_pq_scores.size(0) == queries.size(0), "dense score head count mismatch");
  TORCH_CHECK(dense_pq_scores.size(1) == value_codebooks.size(1) * page_size, "dense score token count mismatch");
  TORCH_CHECK(queries.size(0) == ranked_tokens.size(0), "ranked token head count mismatch");
  TORCH_CHECK(queries.size(1) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(queries.size(1) == value_codebooks.size(2) * value_codebooks.size(4), "value codebook dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(granularity > 0, "granularity must be positive");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_geometric_accept_counts_vpq_cuda(
      queries.contiguous(),
      keys,
      values,
      dense_pq_scores.contiguous(),
      value_codebooks.contiguous(),
      value_codes.contiguous(),
      page_starts.contiguous(),
      ranked_tokens.contiguous(),
      ranked_scores.contiguous(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
      exact_value_top,
      scale);
}

torch::Tensor gqa_decode_geometric_accept_counts_vpq_tail_stability(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    int64_t exact_value_top,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(dense_pq_scores.is_cuda(), "dense_pq_scores must be CUDA");
  TORCH_CHECK(value_codebooks.is_cuda(), "value_codebooks must be CUDA");
  TORCH_CHECK(value_codes.is_cuda(), "value_codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(is_float_like(values), "values must be float32, float16, or bfloat16");
  TORCH_CHECK(dense_pq_scores.scalar_type() == torch::kFloat32, "dense_pq_scores must be float32");
  TORCH_CHECK(value_codebooks.scalar_type() == torch::kFloat32, "value_codebooks must be float32");
  TORCH_CHECK(
      value_codes.scalar_type() == torch::kLong || value_codes.scalar_type() == torch::kUInt8,
      "value_codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(ranked_scores.scalar_type() == torch::kFloat32, "ranked_scores must be float32");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(dense_pq_scores.dim() == 2, "dense_pq_scores shape must be [heads, pages * page_size]");
  TORCH_CHECK(value_codebooks.dim() == 5, "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, selected]");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(keys.size(0) == values.size(0), "key/value kv-head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "key/value token count mismatch");
  TORCH_CHECK(keys.size(2) == values.size(2), "key/value dim mismatch");
  TORCH_CHECK(value_codebooks.size(0) == value_codes.size(0), "value kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value/key kv-head count mismatch");
  TORCH_CHECK(value_codes.size(2) == page_size, "value code page_size mismatch");
  TORCH_CHECK(dense_pq_scores.size(0) == queries.size(0), "dense score head count mismatch");
  TORCH_CHECK(dense_pq_scores.size(1) == value_codebooks.size(1) * page_size, "dense score token count mismatch");
  TORCH_CHECK(queries.size(0) == ranked_tokens.size(0), "ranked token head count mismatch");
  TORCH_CHECK(queries.size(1) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(queries.size(1) == value_codebooks.size(2) * value_codebooks.size(4), "value codebook dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(granularity > 0, "granularity must be positive");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_geometric_accept_counts_vpq_tail_stability_cuda(
      queries.contiguous(),
      keys,
      values,
      dense_pq_scores.contiguous(),
      value_codebooks.contiguous(),
      value_codes.contiguous(),
      page_starts.contiguous(),
      ranked_tokens.contiguous(),
      ranked_scores.contiguous(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
      exact_value_top,
      scale);
}

torch::Tensor gqa_decode_geometric_accept_counts_vpq_proxy(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    int64_t exact_value_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(dense_pq_scores.is_cuda(), "dense_pq_scores must be CUDA");
  TORCH_CHECK(value_codebooks.is_cuda(), "value_codebooks must be CUDA");
  TORCH_CHECK(value_codes.is_cuda(), "value_codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(is_float_like(values), "values must be float32, float16, or bfloat16");
  TORCH_CHECK(dense_pq_scores.scalar_type() == torch::kFloat32, "dense_pq_scores must be float32");
  TORCH_CHECK(value_codebooks.scalar_type() == torch::kFloat32, "value_codebooks must be float32");
  TORCH_CHECK(
      value_codes.scalar_type() == torch::kLong || value_codes.scalar_type() == torch::kUInt8,
      "value_codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(ranked_scores.scalar_type() == torch::kFloat32, "ranked_scores must be float32");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(dense_pq_scores.dim() == 2, "dense_pq_scores shape must be [heads, pages * page_size]");
  TORCH_CHECK(value_codebooks.dim() == 5, "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, selected]");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(keys.size(0) == values.size(0), "key/value kv-head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "key/value token count mismatch");
  TORCH_CHECK(keys.size(2) == values.size(2), "key/value dim mismatch");
  TORCH_CHECK(value_codebooks.size(0) == value_codes.size(0), "value kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value/key kv-head count mismatch");
  TORCH_CHECK(value_codes.size(2) == page_size, "value code page_size mismatch");
  TORCH_CHECK(dense_pq_scores.size(0) == queries.size(0), "dense score head count mismatch");
  TORCH_CHECK(dense_pq_scores.size(1) == value_codebooks.size(1) * page_size, "dense score token count mismatch");
  TORCH_CHECK(queries.size(0) == ranked_tokens.size(0), "ranked token head count mismatch");
  TORCH_CHECK(queries.size(1) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(queries.size(1) == value_codebooks.size(2) * value_codebooks.size(4), "value codebook dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(granularity > 0, "granularity must be positive");
  TORCH_CHECK(proxy_mass_min >= 0.0 && proxy_mass_min <= 1.0, "proxy_mass_min must be in [0, 1]");
  TORCH_CHECK(proxy_tail_mass_max >= 0.0 && proxy_tail_mass_max <= 1.0, "proxy_tail_mass_max must be in [0, 1]");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_geometric_accept_counts_vpq_proxy_cuda(
      queries.contiguous(),
      keys,
      values,
      dense_pq_scores.contiguous(),
      value_codebooks.contiguous(),
      value_codes.contiguous(),
      page_starts.contiguous(),
      ranked_tokens.contiguous(),
      ranked_scores.contiguous(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
      exact_value_top,
      scale,
      proxy_mass_min,
      proxy_tail_mass_max,
      pq_corr_min,
      pq_relrmse_max,
      calibrate_proxy,
      probe_includes_tail);
}

torch::Tensor gqa_decode_ranked_exact_logits(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(ranked_scores.scalar_type() == torch::kFloat32, "ranked_scores must be float32");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, selected]");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(queries.size(0) == ranked_tokens.size(0), "ranked token head count mismatch");
  TORCH_CHECK(queries.size(1) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_ranked_exact_logits_cuda(
      queries.contiguous(),
      keys,
      ranked_tokens.contiguous(),
      ranked_scores.contiguous(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      scale);
}

torch::Tensor gqa_decode_full_exact_logits(
    torch::Tensor queries,
    torch::Tensor keys,
    int64_t group_size,
    int64_t query_context_len,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(queries.size(1) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(query_context_len >= 0, "query_context_len must be non-negative");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_full_exact_logits_cuda(
      queries.contiguous(),
      keys,
      group_size,
      query_context_len,
      scale);
}

torch::Tensor gqa_decode_full_exact_logits_grouped(
    torch::Tensor queries,
    torch::Tensor keys,
    int64_t group_size,
    int64_t query_context_len,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(queries.size(1) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(group_size <= 8, "grouped exact logits currently supports group_size <= 8");
  TORCH_CHECK(query_context_len >= 0, "query_context_len must be non-negative");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_full_exact_logits_grouped_cuda(
      queries.contiguous(),
      keys,
      group_size,
      query_context_len,
      scale);
}

torch::Tensor gqa_decode_full_exact_logits_t_cublas(
    torch::Tensor queries,
    torch::Tensor keys_t,
    int64_t group_size,
    int64_t query_context_len,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys_t.is_cuda(), "keys_t must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(keys_t.scalar_type() == torch::kFloat32, "keys_t must be float32");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys_t.dim() == 3, "keys_t shape must be [kv_heads, dim, tokens]");
  TORCH_CHECK(queries.size(1) == keys_t.size(1), "query/key dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(query_context_len >= 0, "query_context_len must be non-negative");
  TORCH_CHECK(queries.size(0) == keys_t.size(0) * group_size, "heads must equal kv_heads * group_size");
  return gqa_decode_full_exact_logits_t_cublas_cuda(
      queries.contiguous(),
      keys_t,
      group_size,
      query_context_len,
      scale);
}

torch::Tensor gqa_decode_token_exact_logits(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor tokens,
    int64_t group_size,
    int64_t query_context_len,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(tokens.is_cuda(), "tokens must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(tokens.scalar_type() == torch::kLong, "tokens must be int64");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(tokens.dim() == 2, "tokens shape must be [heads, selected]");
  TORCH_CHECK(queries.size(0) == tokens.size(0), "token head count mismatch");
  TORCH_CHECK(queries.size(1) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(query_context_len >= 0, "query_context_len must be non-negative");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_token_exact_logits_cuda(
      queries.contiguous(),
      keys,
      tokens.contiguous(),
      group_size,
      query_context_len,
      scale);
}

std::vector<torch::Tensor> gqa_decode_ranked_exact_logits_with_base_lse(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(ranked_scores.scalar_type() == torch::kFloat32, "ranked_scores must be float32");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, selected]");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(queries.size(0) == ranked_tokens.size(0), "ranked token head count mismatch");
  TORCH_CHECK(queries.size(1) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_ranked_exact_logits_with_base_lse_cuda(
      queries.contiguous(),
      keys,
      ranked_tokens.contiguous(),
      ranked_scores.contiguous(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      scale);
}

