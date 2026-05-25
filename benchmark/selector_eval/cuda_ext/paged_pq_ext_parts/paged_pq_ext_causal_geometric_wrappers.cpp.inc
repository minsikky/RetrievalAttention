    const torch::Tensor& exact_scores,
    const torch::Tensor& pq_logits,
    const torch::Tensor& y_indexed,
    const torch::Tensor& indexed_tokens,
    const torch::Tensor& base_tokens,
    const torch::Tensor& ranked_prefix_tokens,
    const torch::Tensor& k_take_counts) {
  TORCH_CHECK(exact_scores.is_cuda(), "exact_scores must be CUDA");
  TORCH_CHECK(pq_logits.is_cuda(), "pq_logits must be CUDA");
  TORCH_CHECK(y_indexed.is_cuda(), "y_indexed must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(base_tokens.is_cuda(), "base_tokens must be CUDA");
  TORCH_CHECK(ranked_prefix_tokens.is_cuda(), "ranked_prefix_tokens must be CUDA");
  TORCH_CHECK(k_take_counts.is_cuda(), "k_take_counts must be CUDA");
  TORCH_CHECK(exact_scores.scalar_type() == torch::kFloat32, "exact_scores must be float32");
  TORCH_CHECK(pq_logits.scalar_type() == torch::kFloat32, "pq_logits must be float32");
  TORCH_CHECK(y_indexed.scalar_type() == torch::kFloat32, "y_indexed must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(base_tokens.scalar_type() == torch::kLong, "base_tokens must be int64");
  TORCH_CHECK(ranked_prefix_tokens.scalar_type() == torch::kLong, "ranked_prefix_tokens must be int64");
  TORCH_CHECK(k_take_counts.scalar_type() == torch::kLong, "k_take_counts must be int64");
  TORCH_CHECK(exact_scores.dim() == 2, "exact_scores shape must be [heads, context]");
  TORCH_CHECK(pq_logits.dim() == 2, "pq_logits shape must be [heads, indexed]");
  TORCH_CHECK(y_indexed.sizes() == pq_logits.sizes(), "y_indexed/pq_logits shape mismatch");
  TORCH_CHECK(indexed_tokens.dim() == 1, "indexed_tokens shape must be [indexed]");
  TORCH_CHECK(base_tokens.dim() == 1, "base_tokens shape must be [base]");
  TORCH_CHECK(ranked_prefix_tokens.dim() == 2, "ranked_prefix_tokens shape must be [heads, max_rank]");
  TORCH_CHECK(k_take_counts.dim() == 1, "k_take_counts shape must be [k]");
  TORCH_CHECK(pq_logits.size(0) == exact_scores.size(0), "pq/exact head count mismatch");
  TORCH_CHECK(y_indexed.size(0) == exact_scores.size(0), "y/exact head count mismatch");
  TORCH_CHECK(indexed_tokens.size(0) == pq_logits.size(1), "indexed token count mismatch");
  TORCH_CHECK(ranked_prefix_tokens.size(0) == exact_scores.size(0), "ranked/exact head count mismatch");
}

torch::Tensor joint_mixed_score_grid_tokenfit_scaled(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate,
    double pq_scale) {
  check_joint_mixed_score_grid_common(
      exact_scores,
      pq_logits,
      y_indexed,
      indexed_tokens,
      base_tokens,
      ranked_prefix_tokens,
      k_take_counts);
  return joint_mixed_score_grid_tokenfit_scaled_cuda(
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      y_indexed.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      calibrate,
      pq_scale);
}

torch::Tensor joint_mixed_score_grid_no_exact_fill_tokenfit_scaled(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate,
    double pq_scale) {
  check_joint_mixed_score_grid_common(
      exact_scores,
      pq_logits,
      y_indexed,
      indexed_tokens,
      base_tokens,
      ranked_prefix_tokens,
      k_take_counts);
  return joint_mixed_score_grid_no_exact_fill_tokenfit_scaled_cuda(
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      y_indexed.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      calibrate,
      pq_scale);
}

torch::Tensor joint_sparse_exact_score_table(
    torch::Tensor base_tokens,
    torch::Tensor base_logits,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_logits,
    int64_t context_len) {
  TORCH_CHECK(base_tokens.is_cuda(), "base_tokens must be CUDA");
  TORCH_CHECK(base_logits.is_cuda(), "base_logits must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_logits.is_cuda(), "ranked_logits must be CUDA");
  TORCH_CHECK(base_tokens.scalar_type() == torch::kLong, "base_tokens must be int64");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(base_logits.scalar_type() == torch::kFloat32, "base_logits must be float32");
  TORCH_CHECK(ranked_logits.scalar_type() == torch::kFloat32, "ranked_logits must be float32");
  TORCH_CHECK(base_tokens.dim() == 1, "base_tokens shape must be [base_count]");
  TORCH_CHECK(base_logits.dim() == 2, "base_logits shape must be [heads, base_count]");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, max_rank]");
  TORCH_CHECK(ranked_logits.dim() == 2, "ranked_logits shape must be [heads, max_rank]");
  TORCH_CHECK(base_logits.size(1) == base_tokens.size(0), "base token/logit count mismatch");
  TORCH_CHECK(ranked_logits.sizes() == ranked_tokens.sizes(), "ranked token/logit shape mismatch");
  TORCH_CHECK(base_logits.size(0) == ranked_tokens.size(0), "head count mismatch");
  TORCH_CHECK(context_len >= 0, "context_len must be non-negative");
  return joint_sparse_exact_score_table_cuda(
      base_tokens.contiguous(),
      base_logits.contiguous(),
      ranked_tokens.contiguous(),
      ranked_logits.contiguous(),
      context_len);
}

torch::Tensor joint_mixed_score_grid_sparse_exact_tokenfit_scaled(
    torch::Tensor pq_logits,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor base_logits,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor ranked_logits,
    torch::Tensor k_take_counts,
    int64_t context_len,
    bool calibrate,
    double pq_scale) {
  TORCH_CHECK(pq_logits.is_cuda(), "pq_logits must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(base_tokens.is_cuda(), "base_tokens must be CUDA");
  TORCH_CHECK(base_logits.is_cuda(), "base_logits must be CUDA");
  TORCH_CHECK(ranked_prefix_tokens.is_cuda(), "ranked_prefix_tokens must be CUDA");
  TORCH_CHECK(ranked_logits.is_cuda(), "ranked_logits must be CUDA");
  TORCH_CHECK(k_take_counts.is_cuda(), "k_take_counts must be CUDA");
  TORCH_CHECK(pq_logits.scalar_type() == torch::kFloat32, "pq_logits must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(base_tokens.scalar_type() == torch::kLong, "base_tokens must be int64");
  TORCH_CHECK(base_logits.scalar_type() == torch::kFloat32, "base_logits must be float32");
  TORCH_CHECK(ranked_prefix_tokens.scalar_type() == torch::kLong, "ranked_prefix_tokens must be int64");
  TORCH_CHECK(ranked_logits.scalar_type() == torch::kFloat32, "ranked_logits must be float32");
  TORCH_CHECK(k_take_counts.scalar_type() == torch::kLong, "k_take_counts must be int64");
  TORCH_CHECK(pq_logits.dim() == 2, "pq_logits shape must be [heads, indexed]");
  TORCH_CHECK(indexed_tokens.dim() == 1, "indexed_tokens shape must be [indexed]");
  TORCH_CHECK(base_tokens.dim() == 1, "base_tokens shape must be [base_count]");
  TORCH_CHECK(base_logits.dim() == 2, "base_logits shape must be [heads, base_count]");
  TORCH_CHECK(ranked_prefix_tokens.dim() == 2, "ranked_prefix_tokens shape must be [heads, max_rank]");
  TORCH_CHECK(ranked_logits.dim() == 2, "ranked_logits shape must be [heads, max_rank]");
  TORCH_CHECK(k_take_counts.dim() == 1, "k_take_counts shape must be [k]");
  TORCH_CHECK(indexed_tokens.size(0) == pq_logits.size(1), "indexed token count mismatch");
  TORCH_CHECK(base_logits.size(0) == pq_logits.size(0), "base/head count mismatch");
  TORCH_CHECK(base_logits.size(1) == base_tokens.size(0), "base token/logit count mismatch");
  TORCH_CHECK(ranked_prefix_tokens.size(0) == pq_logits.size(0), "ranked/head count mismatch");
  TORCH_CHECK(ranked_logits.sizes() == ranked_prefix_tokens.sizes(), "ranked token/logit shape mismatch");
  TORCH_CHECK(context_len >= 0, "context_len must be non-negative");
  return joint_mixed_score_grid_sparse_exact_tokenfit_scaled_cuda(
      pq_logits.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      base_logits.contiguous(),
      ranked_prefix_tokens.contiguous(),
      ranked_logits.contiguous(),
      k_take_counts.contiguous(),
      context_len,
      calibrate,
      pq_scale);
}


torch::Tensor gqa_causal_geometric_accept_counts(
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
    int64_t query_start,
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
  TORCH_CHECK(queries.dim() == 3, "queries shape must be [positions, heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(dense_pq_scores.dim() == 3, "dense_pq_scores shape must be [positions, heads, pages * page_size]");
  TORCH_CHECK(value_codebooks.dim() == 5, "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(ranked_tokens.dim() == 3, "ranked_tokens shape must be [positions, heads, selected]");
  TORCH_CHECK(ranked_scores.dim() == 3, "ranked_scores shape must be [positions, heads, selected]");
  TORCH_CHECK(keys.size(0) == values.size(0), "key/value kv-head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "key/value token count mismatch");
  TORCH_CHECK(keys.size(2) == values.size(2), "key/value dim mismatch");
  TORCH_CHECK(value_codebooks.size(0) == value_codes.size(0), "value kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value/key kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(1) == value_codes.size(1), "value page count mismatch");
  TORCH_CHECK(value_codebooks.size(2) == value_codes.size(3), "value subvec count mismatch");
  TORCH_CHECK(value_codes.size(2) == page_size, "value code page_size mismatch");
  TORCH_CHECK(value_codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(dense_pq_scores.size(0) == queries.size(0), "dense score position count mismatch");
  TORCH_CHECK(dense_pq_scores.size(1) == queries.size(1), "dense score head count mismatch");
  TORCH_CHECK(dense_pq_scores.size(2) == value_codebooks.size(1) * page_size, "dense score token count mismatch");
  TORCH_CHECK(ranked_tokens.size(0) == queries.size(0), "ranked token position count mismatch");
  TORCH_CHECK(ranked_tokens.size(1) == queries.size(1), "ranked token head count mismatch");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(queries.size(2) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(queries.size(2) == value_codebooks.size(2) * value_codebooks.size(4), "value codebook dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(granularity > 0, "granularity must be positive");
  TORCH_CHECK(queries.size(1) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_causal_geometric_accept_counts_cuda(
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
      query_start,
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

torch::Tensor gqa_causal_geometric_accept_counts_vpq(
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
    int64_t query_start,
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
  TORCH_CHECK(queries.dim() == 3, "queries shape must be [positions, heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(dense_pq_scores.dim() == 3, "dense_pq_scores shape must be [positions, heads, pages * page_size]");
  TORCH_CHECK(value_codebooks.dim() == 5, "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(ranked_tokens.dim() == 3, "ranked_tokens shape must be [positions, heads, selected]");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(keys.size(0) == values.size(0), "key/value kv-head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "key/value token count mismatch");
  TORCH_CHECK(keys.size(2) == values.size(2), "key/value dim mismatch");
  TORCH_CHECK(value_codebooks.size(0) == value_codes.size(0), "value kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value/key kv-head count mismatch");
  TORCH_CHECK(value_codes.size(2) == page_size, "value code page_size mismatch");
  TORCH_CHECK(dense_pq_scores.size(0) == queries.size(0), "dense score position count mismatch");
  TORCH_CHECK(dense_pq_scores.size(1) == queries.size(1), "dense score head count mismatch");
  TORCH_CHECK(dense_pq_scores.size(2) == value_codebooks.size(1) * page_size, "dense score token count mismatch");
  TORCH_CHECK(ranked_tokens.size(0) == queries.size(0), "ranked token position count mismatch");
  TORCH_CHECK(ranked_tokens.size(1) == queries.size(1), "ranked token head count mismatch");
  TORCH_CHECK(queries.size(2) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(queries.size(2) == value_codebooks.size(2) * value_codebooks.size(4), "value codebook dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(granularity > 0, "granularity must be positive");
  TORCH_CHECK(queries.size(1) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_causal_geometric_accept_counts_vpq_cuda(
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
      query_start,
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

torch::Tensor gqa_exact_selected_attention(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor tokens,
    int64_t group_size,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(tokens.is_cuda(), "tokens must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(is_float_like(values), "values must be float32, float16, or bfloat16");
  TORCH_CHECK(tokens.scalar_type() == torch::kLong, "tokens must be int64");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(tokens.dim() == 2, "tokens shape must be [heads, selected]");
  TORCH_CHECK(keys.size(0) == values.size(0), "key/value kv-head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "key/value token count mismatch");
  TORCH_CHECK(keys.size(2) == values.size(2), "key/value dim mismatch");
  TORCH_CHECK(queries.size(1) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(tokens.size(0) == queries.size(0), "tokens head count mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_exact_selected_attention_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
      tokens.contiguous(),
      group_size,
      scale);
}

