torch::Tensor gqa_decode_vpq_selected_tail_from_scores(
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
    int64_t exact_value_top,
    double scale,
    double tail_blend) {
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
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_vpq_tail_from_scores_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
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
      page_starts.new_empty({0}),
      exact_value_top,
      0.0,
      scale,
      tail_blend);
}

torch::Tensor gqa_decode_vpq_selected_tail_from_scores_counts(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor exact_value_counts,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t exact_value_top,
    double scale,
    double tail_blend) {
  TORCH_CHECK(exact_value_counts.is_cuda(), "exact_value_counts must be CUDA");
  TORCH_CHECK(exact_value_counts.scalar_type() == torch::kLong, "exact_value_counts must be int64");
  TORCH_CHECK(exact_value_counts.dim() == 1, "exact_value_counts shape must be [heads]");
  TORCH_CHECK(exact_value_counts.size(0) == queries.size(0), "exact count head mismatch");
  return gqa_decode_vpq_tail_from_scores_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
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
      exact_value_counts.contiguous(),
      exact_value_top,
      0.0,
      scale,
      tail_blend);
}

torch::Tensor gqa_decode_vpq_selected_tail_agg_from_scores_mass(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    double exact_value_mass,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale,
    double tail_blend) {
  TORCH_CHECK(exact_value_mass >= 0.0 && exact_value_mass <= 1.0, "exact_value_mass must be in [0, 1]");
	  return gqa_decode_vpq_selected_tail_agg_from_scores_cuda(
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
      page_starts.new_empty({0}),
      0,
      exact_value_mass,
      scale,
      tail_blend);
}

torch::Tensor gqa_decode_vpq_selected_tail_agg_from_scores_mass_min(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    double exact_value_mass,
    int64_t exact_value_min_top,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale,
    double tail_blend) {
  TORCH_CHECK(exact_value_mass >= 0.0 && exact_value_mass <= 1.0, "exact_value_mass must be in [0, 1]");
  TORCH_CHECK(exact_value_min_top >= 0, "exact_value_min_top must be non-negative");
	  return gqa_decode_vpq_selected_tail_agg_from_scores_cuda(
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
      page_starts.new_empty({0}),
      exact_value_min_top,
      exact_value_mass,
      scale,
      tail_blend);
}

torch::Tensor gqa_decode_vpq_selected_from_logits_mass_min(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits,
    double exact_value_mass,
    int64_t exact_value_min_top,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(value_codebooks.is_cuda(), "value_codebooks must be CUDA");
  TORCH_CHECK(value_codes.is_cuda(), "value_codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(ranked_logits.is_cuda(), "ranked_logits must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float-like");
  TORCH_CHECK(is_float_like(values), "values must be float-like");
  TORCH_CHECK(value_codebooks.scalar_type() == torch::kFloat32, "value_codebooks must be float32");
  TORCH_CHECK(
      value_codes.scalar_type() == torch::kLong || value_codes.scalar_type() == torch::kUInt8,
      "value_codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(ranked_scores.scalar_type() == torch::kFloat32, "ranked_scores must be float32");
  TORCH_CHECK(ranked_logits.scalar_type() == torch::kFloat32, "ranked_logits must be float32");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, total_tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, total_tokens, dim]");
  TORCH_CHECK(value_codebooks.dim() == 5, "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, ranked]");
  TORCH_CHECK(ranked_scores.dim() == 2, "ranked_scores shape must be [heads, ranked]");
  TORCH_CHECK(ranked_logits.dim() == 2, "ranked_logits shape must be [heads, ranked]");
  TORCH_CHECK(keys.size(0) == values.size(0), "kv head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "token count mismatch");
  TORCH_CHECK(keys.size(2) == queries.size(1), "key dim mismatch");
  TORCH_CHECK(values.size(2) == queries.size(1), "value dim mismatch");
  TORCH_CHECK(ranked_tokens.size(0) == queries.size(0), "ranked token head mismatch");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked score shape mismatch");
  TORCH_CHECK(ranked_logits.sizes() == ranked_tokens.sizes(), "ranked logit shape mismatch");
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value codebook kv head mismatch");
  TORCH_CHECK(value_codes.size(0) == keys.size(0), "value code kv head mismatch");
  TORCH_CHECK(value_codebooks.size(1) == value_codes.size(1), "value page count mismatch");
  TORCH_CHECK(value_codebooks.size(1) == page_starts.size(0), "page starts count mismatch");
  TORCH_CHECK(value_codebooks.size(2) == value_codes.size(3), "value subvec count mismatch");
  TORCH_CHECK(value_codebooks.size(4) * value_codebooks.size(2) == queries.size(1), "value subdim mismatch");
  TORCH_CHECK(exact_value_mass >= 0.0 && exact_value_mass <= 1.0, "exact_value_mass must be in [0, 1]");
  TORCH_CHECK(exact_value_min_top >= 0, "exact_value_min_top must be non-negative");
  return gqa_decode_vpq_selected_from_logits_cuda(
      queries.contiguous(),
      keys,
      values,
      value_codebooks.contiguous(),
      value_codes.contiguous(),
      page_starts.contiguous(),
      ranked_tokens.contiguous(),
      ranked_scores.contiguous(),
      ranked_logits.contiguous(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      page_starts.new_empty({0}),
      exact_value_min_top,
      exact_value_mass,
      scale);
}

torch::Tensor gqa_decode_vpq_selected_tail_agg_from_logits_mass_min(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits,
    double exact_value_mass,
    int64_t exact_value_min_top,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale,
    double tail_blend) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(dense_pq_scores.is_cuda(), "dense_pq_scores must be CUDA");
  TORCH_CHECK(value_codebooks.is_cuda(), "value_codebooks must be CUDA");
  TORCH_CHECK(value_codes.is_cuda(), "value_codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(ranked_logits.is_cuda(), "ranked_logits must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float-like");
  TORCH_CHECK(is_float_like(values), "values must be float-like");
  TORCH_CHECK(dense_pq_scores.scalar_type() == torch::kFloat32, "dense_pq_scores must be float32");
  TORCH_CHECK(value_codebooks.scalar_type() == torch::kFloat32, "value_codebooks must be float32");
  TORCH_CHECK(
      value_codes.scalar_type() == torch::kLong || value_codes.scalar_type() == torch::kUInt8,
      "value_codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(ranked_scores.scalar_type() == torch::kFloat32, "ranked_scores must be float32");
  TORCH_CHECK(ranked_logits.scalar_type() == torch::kFloat32, "ranked_logits must be float32");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, total_tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, total_tokens, dim]");
  TORCH_CHECK(dense_pq_scores.dim() == 2, "dense_pq_scores shape must be [heads, pages * page_size]");
  TORCH_CHECK(value_codebooks.dim() == 5, "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, ranked]");
  TORCH_CHECK(ranked_scores.dim() == 2, "ranked_scores shape must be [heads, ranked]");
  TORCH_CHECK(ranked_logits.dim() == 2, "ranked_logits shape must be [heads, ranked]");
  TORCH_CHECK(keys.size(0) == values.size(0), "kv head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "token count mismatch");
  TORCH_CHECK(keys.size(2) == queries.size(1), "key dim mismatch");
  TORCH_CHECK(values.size(2) == queries.size(1), "value dim mismatch");
  TORCH_CHECK(dense_pq_scores.size(0) == queries.size(0), "dense score head mismatch");
  TORCH_CHECK(ranked_tokens.size(0) == queries.size(0), "ranked token head mismatch");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked score shape mismatch");
  TORCH_CHECK(ranked_logits.sizes() == ranked_tokens.sizes(), "ranked logit shape mismatch");
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value codebook kv head mismatch");
  TORCH_CHECK(value_codes.size(0) == keys.size(0), "value code kv head mismatch");
  TORCH_CHECK(value_codebooks.size(1) == value_codes.size(1), "value page count mismatch");
  TORCH_CHECK(value_codebooks.size(1) == page_starts.size(0), "page starts count mismatch");
  TORCH_CHECK(value_codebooks.size(2) == value_codes.size(3), "value subvec count mismatch");
  TORCH_CHECK(value_codebooks.size(4) * value_codebooks.size(2) == queries.size(1), "value subdim mismatch");
  TORCH_CHECK(dense_pq_scores.size(1) == value_codebooks.size(1) * page_size, "dense score token count mismatch");
  TORCH_CHECK(exact_value_mass >= 0.0 && exact_value_mass <= 1.0, "exact_value_mass must be in [0, 1]");
  TORCH_CHECK(exact_value_min_top >= 0, "exact_value_min_top must be non-negative");
  return gqa_decode_vpq_selected_tail_agg_from_logits_cuda(
      queries.contiguous(),
      keys,
      values,
      dense_pq_scores.contiguous(),
      value_codebooks.contiguous(),
      value_codes.contiguous(),
      page_starts.contiguous(),
      ranked_tokens.contiguous(),
      ranked_scores.contiguous(),
      ranked_logits.contiguous(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      page_starts.new_empty({0}),
      exact_value_min_top,
      exact_value_mass,
      scale,
      tail_blend);
}

torch::Tensor gqa_decode_vpq_selected_tail_agg_from_logits_mass_min_thresholds(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits,
    torch::Tensor exact_value_thresholds,
    torch::Tensor exact_value_threshold_sels,
    double exact_value_mass,
    int64_t exact_value_min_top,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale,
    double tail_blend) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(dense_pq_scores.is_cuda(), "dense_pq_scores must be CUDA");
  TORCH_CHECK(value_codebooks.is_cuda(), "value_codebooks must be CUDA");
  TORCH_CHECK(value_codes.is_cuda(), "value_codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(ranked_logits.is_cuda(), "ranked_logits must be CUDA");
  TORCH_CHECK(exact_value_thresholds.is_cuda(), "exact_value_thresholds must be CUDA");
  TORCH_CHECK(exact_value_threshold_sels.is_cuda(), "exact_value_threshold_sels must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float-like");
  TORCH_CHECK(is_float_like(values), "values must be float-like");
  TORCH_CHECK(dense_pq_scores.scalar_type() == torch::kFloat32, "dense_pq_scores must be float32");
  TORCH_CHECK(value_codebooks.scalar_type() == torch::kFloat32, "value_codebooks must be float32");
  TORCH_CHECK(
      value_codes.scalar_type() == torch::kLong || value_codes.scalar_type() == torch::kUInt8,
      "value_codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(ranked_scores.scalar_type() == torch::kFloat32, "ranked_scores must be float32");
  TORCH_CHECK(ranked_logits.scalar_type() == torch::kFloat32, "ranked_logits must be float32");
  TORCH_CHECK(exact_value_thresholds.scalar_type() == torch::kFloat32, "exact_value_thresholds must be float32");
  TORCH_CHECK(exact_value_threshold_sels.scalar_type() == torch::kLong, "exact_value_threshold_sels must be int64");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, total_tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, total_tokens, dim]");
  TORCH_CHECK(dense_pq_scores.dim() == 2, "dense_pq_scores shape must be [heads, pages * page_size]");
  TORCH_CHECK(value_codebooks.dim() == 5, "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, ranked]");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked score shape mismatch");
  TORCH_CHECK(ranked_logits.sizes() == ranked_tokens.sizes(), "ranked logit shape mismatch");
  TORCH_CHECK(exact_value_thresholds.dim() == 1, "exact_value_thresholds shape must be [heads]");
  TORCH_CHECK(exact_value_threshold_sels.sizes() == exact_value_thresholds.sizes(), "threshold shape mismatch");
  TORCH_CHECK(exact_value_thresholds.size(0) == queries.size(0), "threshold head count mismatch");
  TORCH_CHECK(keys.size(0) == values.size(0), "kv head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "token count mismatch");
  TORCH_CHECK(keys.size(2) == queries.size(1), "key dim mismatch");
  TORCH_CHECK(values.size(2) == queries.size(1), "value dim mismatch");
  TORCH_CHECK(dense_pq_scores.size(0) == queries.size(0), "dense score head mismatch");
  TORCH_CHECK(ranked_tokens.size(0) == queries.size(0), "ranked token head mismatch");
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value codebook kv head mismatch");
  TORCH_CHECK(value_codes.size(0) == keys.size(0), "value code kv head mismatch");
  TORCH_CHECK(value_codebooks.size(1) == value_codes.size(1), "value page count mismatch");
  TORCH_CHECK(value_codebooks.size(1) == page_starts.size(0), "page starts count mismatch");
  TORCH_CHECK(value_codebooks.size(2) == value_codes.size(3), "value subvec count mismatch");
  TORCH_CHECK(value_codebooks.size(4) * value_codebooks.size(2) == queries.size(1), "value subdim mismatch");
  TORCH_CHECK(dense_pq_scores.size(1) == value_codebooks.size(1) * page_size, "dense score token count mismatch");
  TORCH_CHECK(exact_value_mass >= 0.0 && exact_value_mass <= 1.0, "exact_value_mass must be in [0, 1]");
  TORCH_CHECK(exact_value_min_top >= 0, "exact_value_min_top must be non-negative");
  return gqa_decode_vpq_selected_tail_agg_from_logits_cuda(
      queries.contiguous(),
      keys,
      values,
      dense_pq_scores.contiguous(),
      value_codebooks.contiguous(),
      value_codes.contiguous(),
      page_starts.contiguous(),
      ranked_tokens.contiguous(),
      ranked_scores.contiguous(),
      ranked_logits.contiguous(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      page_starts.new_empty({0}),
      exact_value_min_top,
      exact_value_mass,
      scale,
      tail_blend,
      exact_value_thresholds.contiguous(),
      exact_value_threshold_sels.contiguous());
}

torch::Tensor gqa_decode_fullscan_vpq_selected_tail_agg(
    torch::Tensor queries,
    torch::Tensor key_codebooks,
    torch::Tensor key_codes,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t exact_value_top,
    double scale,
    double tail_blend) {
  auto topk = gqa_fullscan_pq_topk_scores(
      queries,
      key_codebooks,
      key_codes,
      page_starts,
      group_size,
      budget);
  return gqa_decode_vpq_selected_tail_agg_from_scores(
      queries,
      keys,
      values,
      topk.at(2),
      value_codebooks,
      value_codes,
      page_starts,
      topk.at(0),
      topk.at(1),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      exact_value_top,
      scale,
      tail_blend);
}

torch::Tensor gqa_decode_fullscan_vpq_selected_tail_agg_mass_min(
    torch::Tensor queries,
    torch::Tensor key_codebooks,
    torch::Tensor key_codes,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    double exact_value_mass,
    int64_t exact_value_min_top,
    int64_t group_size,
    int64_t budget,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale,
    double tail_blend) {
  auto topk = gqa_fullscan_pq_topk_scores(
      queries,
      key_codebooks,
      key_codes,
      page_starts,
      group_size,
      budget);
  return gqa_decode_vpq_selected_tail_agg_from_scores_mass_min(
      queries,
      keys,
      values,
      topk.at(2),
      value_codebooks,
      value_codes,
      page_starts,
      topk.at(0),
      topk.at(1),
      exact_value_mass,
      exact_value_min_top,
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      scale,
      tail_blend);
}

torch::Tensor gqa_decode_scoreless_fullscan_vpq_tail(
    torch::Tensor queries,
    torch::Tensor key_codebooks,
    torch::Tensor key_codes,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t exact_value_top,
    double scale,
    double tail_blend,
    int64_t force_mode) {
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  auto q3 = queries.reshape({1, queries.size(0), queries.size(1)}).contiguous();
  auto topk = gqa_causal_fullscan_pq_topk_fused_force_cuda(
      q3,
      key_codebooks.contiguous(),
      key_codes.contiguous(),
      page_starts.contiguous(),
      group_size,
      budget,
      query_context_len - 1,
      static_prefix,
      static_suffix,
      force_mode);
  auto out3 = gqa_causal_vpq_tail_attention_cuda(
      q3,
      keys.contiguous(),
      values.contiguous(),
      key_codebooks.contiguous(),
      key_codes.contiguous(),
      value_codebooks.contiguous(),
      value_codes.contiguous(),
      page_starts.contiguous(),
      topk.at(0).contiguous(),
      topk.at(1).contiguous(),
      group_size,
      query_context_len - 1,
      static_prefix,
      static_suffix,
      page_size,
      exact_value_top,
      scale,
      tail_blend);
  return out3.reshape({queries.size(0), queries.size(1)}).contiguous();
}

torch::Tensor gqa_decode_vpq_tail_from_scores(
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
    double scale,
    double tail_blend) {
  return gqa_decode_vpq_selected_tail_from_scores(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      ranked_tokens.size(1),
      scale,
      tail_blend);
}

torch::Tensor gqa_decode_vpq_selected_tail_agg_from_scores(
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
    int64_t exact_value_top,
    double scale,
    double tail_blend) {
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
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
	  return gqa_decode_vpq_selected_tail_agg_from_scores_cuda(
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
      page_starts.new_empty({0}),
      exact_value_top,
      0.0,
      scale,
      tail_blend);
}

torch::Tensor gqa_decode_vpq_selected_tail_agg_from_scores_counts(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor exact_value_counts,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t exact_value_top,
    double scale,
    double tail_blend) {
  TORCH_CHECK(exact_value_counts.is_cuda(), "exact_value_counts must be CUDA");
  TORCH_CHECK(exact_value_counts.scalar_type() == torch::kLong, "exact_value_counts must be int64");
  TORCH_CHECK(exact_value_counts.dim() == 1, "exact_value_counts shape must be [heads]");
  TORCH_CHECK(exact_value_counts.size(0) == queries.size(0), "exact count head mismatch");
	  return gqa_decode_vpq_selected_tail_agg_from_scores_cuda(
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
      exact_value_counts.contiguous(),
      exact_value_top,
      0.0,
      scale,
      tail_blend);
}

