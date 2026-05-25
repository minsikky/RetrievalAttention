  auto dtype = tensor.scalar_type();
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16;
}

std::vector<torch::Tensor> fullscan_pq_topk(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t budget) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(
      codes.scalar_type() == torch::kLong || codes.scalar_type() == torch::kUInt8,
      "codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(codebooks.dim() == 4, "codebooks shape must be [pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 3, "codes shape must be [pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(codebooks.size(0) == codes.size(0), "page count mismatch");
  TORCH_CHECK(codebooks.size(1) == codes.size(2), "subvec count mismatch");
  TORCH_CHECK(codebooks.size(0) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(queries.size(1) == codebooks.size(1) * codebooks.size(3), "query dim must equal subvecs * subdim");
  TORCH_CHECK(budget >= 0, "budget must be non-negative");
  return fullscan_pq_topk_cuda(
      queries.contiguous(),
      codebooks.contiguous(),
      codes.contiguous(),
      page_starts.contiguous(),
      budget);
}

std::vector<torch::Tensor> fullscan_pq_topk_scores(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t budget) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(
      codes.scalar_type() == torch::kLong || codes.scalar_type() == torch::kUInt8,
      "codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(codebooks.dim() == 4, "codebooks shape must be [pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 3, "codes shape must be [pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(codebooks.size(0) == codes.size(0), "page count mismatch");
  TORCH_CHECK(codebooks.size(1) == codes.size(2), "subvec count mismatch");
  TORCH_CHECK(codebooks.size(0) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(queries.size(1) == codebooks.size(1) * codebooks.size(3), "query dim must equal subvecs * subdim");
  TORCH_CHECK(budget >= 0, "budget must be non-negative");
  return fullscan_pq_topk_scores_cuda(
      queries.contiguous(),
      codebooks.contiguous(),
      codes.contiguous(),
      page_starts.contiguous(),
      budget);
}

std::vector<torch::Tensor> gqa_fullscan_pq_topk(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(
      codes.scalar_type() == torch::kLong || codes.scalar_type() == torch::kUInt8,
      "codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(codebooks.dim() == 5, "codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 4, "codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(codebooks.size(0) == codes.size(0), "kv-head count mismatch");
  TORCH_CHECK(codebooks.size(1) == codes.size(1), "page count mismatch");
  TORCH_CHECK(codebooks.size(2) == codes.size(3), "subvec count mismatch");
  TORCH_CHECK(codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(queries.size(1) == codebooks.size(2) * codebooks.size(4), "query dim must equal subvecs * subdim");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(queries.size(0) <= codebooks.size(0) * group_size, "heads exceed kv_heads * group_size");
  TORCH_CHECK(budget >= 0, "budget must be non-negative");
  return gqa_fullscan_pq_topk_cuda(
      queries.contiguous(),
      codebooks.contiguous(),
      codes.contiguous(),
      page_starts.contiguous(),
      group_size,
      budget);
}

std::vector<torch::Tensor> gqa_fullscan_pq_topk_scores(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(
      codes.scalar_type() == torch::kLong || codes.scalar_type() == torch::kUInt8,
      "codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(codebooks.dim() == 5, "codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 4, "codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(codebooks.size(0) == codes.size(0), "kv-head count mismatch");
  TORCH_CHECK(codebooks.size(1) == codes.size(1), "page count mismatch");
  TORCH_CHECK(codebooks.size(2) == codes.size(3), "subvec count mismatch");
  TORCH_CHECK(codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(queries.size(1) == codebooks.size(2) * codebooks.size(4), "query dim must equal subvecs * subdim");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(queries.size(0) <= codebooks.size(0) * group_size, "heads exceed kv_heads * group_size");
  TORCH_CHECK(budget >= 0, "budget must be non-negative");
  return gqa_fullscan_pq_topk_scores_cuda(
      queries.contiguous(),
      codebooks.contiguous(),
      codes.contiguous(),
      page_starts.contiguous(),
      group_size,
      budget);
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(
      codes.scalar_type() == torch::kLong || codes.scalar_type() == torch::kUInt8,
      "codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(queries.dim() == 3, "queries shape must be [positions, heads, dim]");
  TORCH_CHECK(codebooks.dim() == 5, "codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 4, "codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(codebooks.size(0) == codes.size(0), "kv-head count mismatch");
  TORCH_CHECK(codebooks.size(1) == codes.size(1), "page count mismatch");
  TORCH_CHECK(codebooks.size(2) == codes.size(3), "subvec count mismatch");
  TORCH_CHECK(codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(queries.size(2) == codebooks.size(2) * codebooks.size(4), "query dim must equal subvecs * subdim");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(queries.size(1) <= codebooks.size(0) * group_size, "heads exceed kv_heads * group_size");
  TORCH_CHECK(budget >= 0, "budget must be non-negative");
  return gqa_causal_fullscan_pq_topk_cuda(
      queries.contiguous(),
      codebooks.contiguous(),
      codes.contiguous(),
      page_starts.contiguous(),
      group_size,
      budget,
      query_start,
      static_prefix,
      static_suffix);
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_scores(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(
      codes.scalar_type() == torch::kLong || codes.scalar_type() == torch::kUInt8,
      "codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(queries.dim() == 3, "queries shape must be [positions, heads, dim]");
  TORCH_CHECK(codebooks.dim() == 5, "codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 4, "codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(codebooks.size(0) == codes.size(0), "kv-head count mismatch");
  TORCH_CHECK(codebooks.size(1) == codes.size(1), "page count mismatch");
  TORCH_CHECK(codebooks.size(2) == codes.size(3), "subvec count mismatch");
  TORCH_CHECK(codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(queries.size(2) == codebooks.size(2) * codebooks.size(4), "query dim must equal subvecs * subdim");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(queries.size(1) <= codebooks.size(0) * group_size, "heads exceed kv_heads * group_size");
  TORCH_CHECK(budget >= 0, "budget must be non-negative");
  return gqa_causal_fullscan_pq_topk_scores_cuda(
      queries.contiguous(),
      codebooks.contiguous(),
      codes.contiguous(),
      page_starts.contiguous(),
      group_size,
      budget,
      query_start,
      static_prefix,
      static_suffix);
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_fused(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(
      codes.scalar_type() == torch::kLong || codes.scalar_type() == torch::kUInt8,
      "codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(queries.dim() == 3, "queries shape must be [positions, heads, dim]");
  TORCH_CHECK(codebooks.dim() == 5, "codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 4, "codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(codebooks.size(0) == codes.size(0), "kv-head count mismatch");
  TORCH_CHECK(codebooks.size(1) == codes.size(1), "page count mismatch");
  TORCH_CHECK(codebooks.size(2) == codes.size(3), "subvec count mismatch");
  TORCH_CHECK(codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(queries.size(2) == codebooks.size(2) * codebooks.size(4), "query dim must equal subvecs * subdim");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(queries.size(1) <= codebooks.size(0) * group_size, "heads exceed kv_heads * group_size");
  TORCH_CHECK(budget >= 0, "budget must be non-negative");
  return gqa_causal_fullscan_pq_topk_fused_cuda(
      queries.contiguous(),
      codebooks.contiguous(),
      codes.contiguous(),
      page_starts.contiguous(),
      group_size,
      budget,
      query_start,
      static_prefix,
      static_suffix);
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_fused_force(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t force_mode) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(
      codes.scalar_type() == torch::kLong || codes.scalar_type() == torch::kUInt8,
      "codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(queries.dim() == 3, "queries shape must be [positions, heads, dim]");
  TORCH_CHECK(codebooks.dim() == 5, "codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 4, "codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(codebooks.size(0) == codes.size(0), "kv-head count mismatch");
  TORCH_CHECK(codebooks.size(1) == codes.size(1), "page count mismatch");
  TORCH_CHECK(codebooks.size(2) == codes.size(3), "subvec count mismatch");
  TORCH_CHECK(codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(queries.size(2) == codebooks.size(2) * codebooks.size(4), "query dim must equal subvecs * subdim");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(queries.size(1) <= codebooks.size(0) * group_size, "heads exceed kv_heads * group_size");
  TORCH_CHECK(budget >= 0, "budget must be non-negative");
  TORCH_CHECK(force_mode >= 0 && force_mode <= 2, "force_mode must be 0=auto, 1=smallscan, or 2=localtopk");
  return gqa_causal_fullscan_pq_topk_fused_force_cuda(
      queries.contiguous(),
      codebooks.contiguous(),
      codes.contiguous(),
      page_starts.contiguous(),
      group_size,
      budget,
      query_start,
      static_prefix,
      static_suffix,
      force_mode);
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_top_pages(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t page_budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(
      codes.scalar_type() == torch::kLong || codes.scalar_type() == torch::kUInt8,
      "codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(queries.dim() == 3, "queries shape must be [positions, heads, dim]");
  TORCH_CHECK(codebooks.dim() == 5, "codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 4, "codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(codebooks.size(0) == codes.size(0), "kv-head count mismatch");
  TORCH_CHECK(codebooks.size(1) == codes.size(1), "page count mismatch");
  TORCH_CHECK(codebooks.size(2) == codes.size(3), "subvec count mismatch");
  TORCH_CHECK(codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(queries.size(2) == codebooks.size(2) * codebooks.size(4), "query dim must equal subvecs * subdim");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(queries.size(1) <= codebooks.size(0) * group_size, "heads exceed kv_heads * group_size");
  TORCH_CHECK(page_budget >= 0, "page_budget must be non-negative");
  return gqa_causal_fullscan_pq_top_pages_cuda(
      queries.contiguous(),
      codebooks.contiguous(),
      codes.contiguous(),
      page_starts.contiguous(),
      group_size,
      page_budget,
      query_start,
      static_prefix,
      static_suffix);
}

torch::Tensor exact_selected_attention(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor tokens,
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
  TORCH_CHECK(keys.dim() == 2, "keys shape must be [tokens, dim]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [tokens, dim]");
  TORCH_CHECK(tokens.dim() == 2, "tokens shape must be [heads, selected]");
  TORCH_CHECK(keys.size(0) == values.size(0), "key/value token count mismatch");
  TORCH_CHECK(queries.size(1) == keys.size(1), "query/key dim mismatch");
  TORCH_CHECK(values.size(1) == keys.size(1), "key/value dim mismatch");
  TORCH_CHECK(tokens.size(0) == queries.size(0), "tokens head count mismatch");
  return exact_selected_attention_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
      tokens.contiguous(),
      scale);
}

torch::Tensor gqa_causal_exact_selected_attention(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(is_float_like(values), "values must be float32, float16, or bfloat16");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(ranked_scores.scalar_type() == torch::kFloat32, "ranked_scores must be float32");
  TORCH_CHECK(queries.dim() == 3, "queries shape must be [positions, heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(ranked_tokens.dim() == 3, "ranked_tokens shape must be [positions, heads, selected]");
  TORCH_CHECK(ranked_scores.dim() == 3, "ranked_scores shape must be [positions, heads, selected]");
  TORCH_CHECK(keys.size(0) == values.size(0), "key/value kv-head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "key/value token count mismatch");
  TORCH_CHECK(keys.size(2) == values.size(2), "key/value dim mismatch");
  TORCH_CHECK(queries.size(2) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(ranked_tokens.size(0) == queries.size(0), "ranked token position count mismatch");
  TORCH_CHECK(ranked_tokens.size(1) == queries.size(1), "ranked token head count mismatch");
  TORCH_CHECK(ranked_scores.size(0) == ranked_tokens.size(0), "ranked score position count mismatch");
  TORCH_CHECK(ranked_scores.size(1) == ranked_tokens.size(1), "ranked score head count mismatch");
  TORCH_CHECK(ranked_scores.size(2) == ranked_tokens.size(2), "ranked score selected count mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(queries.size(1) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_causal_exact_selected_attention_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
      ranked_tokens.contiguous(),
      ranked_scores.contiguous(),
      group_size,
      query_start,
      static_prefix,
      static_suffix,
      page_size,
      scale);
}

