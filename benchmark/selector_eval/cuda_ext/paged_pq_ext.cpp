#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> fullscan_pq_topk_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t budget);

std::vector<torch::Tensor> fullscan_pq_topk_scores_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t budget);

std::vector<torch::Tensor> gqa_fullscan_pq_topk_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget);

std::vector<torch::Tensor> gqa_fullscan_pq_topk_scores_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget);

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix);

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_scores_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix);

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_fused_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix);

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_fused_force_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t force_mode);

std::vector<torch::Tensor> gqa_causal_fullscan_pq_top_pages_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t page_budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix);

torch::Tensor exact_selected_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor tokens,
    double scale);

torch::Tensor gqa_causal_exact_selected_attention_cuda(
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
    double scale);

torch::Tensor gqa_causal_vpq_selected_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
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
    double scale);

torch::Tensor gqa_causal_vpq_selected_attention_vpagesize_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t selection_page_size,
    int64_t value_page_size,
    int64_t exact_value_top,
    double scale);

torch::Tensor gqa_causal_vpq_tail_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor key_codebooks,
    torch::Tensor key_codes,
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
    int64_t exact_value_top,
    double scale,
    double tail_blend);

torch::Tensor gqa_causal_vpq_tail_from_scores_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor dense_selected_scores,
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
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale,
    double tail_blend);

torch::Tensor gqa_decode_vpq_tail_from_scores_cuda(
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
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale,
    double tail_blend);

torch::Tensor gqa_decode_vpq_selected_tail_agg_from_scores_cuda(
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
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale,
    double tail_blend);

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
    double tail_blend);

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
    double tail_blend);

torch::Tensor gqa_decode_geometric_accept_counts_cuda(
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
    double scale);

torch::Tensor gqa_decode_geometric_accept_counts_vpq_cuda(
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
    double scale);

torch::Tensor gqa_decode_geometric_accept_counts_vpq_tail_stability_cuda(
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
    double scale);

torch::Tensor gqa_decode_geometric_accept_counts_vpq_proxy_cuda(
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
    bool probe_includes_tail);

torch::Tensor gqa_decode_ranked_exact_logits_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale);

std::vector<torch::Tensor> gqa_decode_ranked_exact_logits_with_base_lse_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale);

torch::Tensor gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_cuda(
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
    double exact_value_mass,
    int64_t exact_value_min_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail);

torch::Tensor gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_cuda(
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
    double exact_value_mass,
    int64_t exact_value_min_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail);

torch::Tensor gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds_cuda(
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
    torch::Tensor approx_exact_thresholds,
    torch::Tensor approx_exact_threshold_sels,
    torch::Tensor probe_exact_thresholds,
    torch::Tensor probe_exact_threshold_sels,
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
    double exact_value_mass,
    int64_t exact_value_min_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail);

std::vector<torch::Tensor> gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds_cuda(
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
    torch::Tensor approx_exact_thresholds,
    torch::Tensor approx_exact_threshold_sels,
    torch::Tensor probe_exact_thresholds,
    torch::Tensor probe_exact_threshold_sels,
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
    double exact_value_mass,
    int64_t exact_value_min_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail,
    double tail_blend);

torch::Tensor gqa_causal_geometric_accept_counts_cuda(
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
    double scale);

torch::Tensor gqa_causal_geometric_accept_counts_vpq_cuda(
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
    double scale);

torch::Tensor gqa_decode_vpq_selected_from_logits_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale);

torch::Tensor gqa_decode_vpq_selected_tail_agg_from_logits_cuda(
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
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale,
    double tail_blend,
    torch::Tensor exact_value_thresholds = torch::Tensor(),
    torch::Tensor exact_value_threshold_sels = torch::Tensor());

torch::Tensor gqa_exact_selected_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor tokens,
    int64_t group_size,
    double scale);

static bool is_float_like(torch::Tensor const& tensor) {
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

torch::Tensor gqa_causal_vpq_selected_attention(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
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
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(value_codebooks.is_cuda(), "value_codebooks must be CUDA");
  TORCH_CHECK(value_codes.is_cuda(), "value_codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(is_float_like(values), "values must be float32, float16, or bfloat16");
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
  TORCH_CHECK(
      value_codebooks.dim() == 5,
      "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
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
  TORCH_CHECK(value_codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(value_codes.size(2) == page_size, "value code page_size mismatch");
  TORCH_CHECK(queries.size(0) == ranked_tokens.size(0), "position count mismatch");
  TORCH_CHECK(queries.size(1) == ranked_tokens.size(1), "head count mismatch");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(queries.size(2) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(
      queries.size(2) == value_codebooks.size(2) * value_codebooks.size(4),
      "value codebook dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(queries.size(1) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_causal_vpq_selected_attention_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
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
      scale);
}

torch::Tensor gqa_causal_vpq_selected_attention_mixed_vpagesize(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t selection_page_size,
    int64_t value_page_size,
    int64_t exact_value_top,
    double scale) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(value_codebooks.is_cuda(), "value_codebooks must be CUDA");
  TORCH_CHECK(value_codes.is_cuda(), "value_codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32, float16, or bfloat16");
  TORCH_CHECK(is_float_like(values), "values must be float32, float16, or bfloat16");
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
  TORCH_CHECK(
      value_codebooks.dim() == 5,
      "value_codebooks shape must be [kv_heads, value_pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, value_pages, value_page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [value_pages]");
  TORCH_CHECK(ranked_tokens.dim() == 3, "ranked_tokens shape must be [positions, heads, selected]");
  TORCH_CHECK(ranked_scores.dim() == 3, "ranked_scores shape must be [positions, heads, selected]");
  TORCH_CHECK(keys.size(0) == values.size(0), "key/value kv-head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "key/value token count mismatch");
  TORCH_CHECK(keys.size(2) == values.size(2), "key/value dim mismatch");
  TORCH_CHECK(value_codebooks.size(0) == value_codes.size(0), "value kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value/key kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(1) == value_codes.size(1), "value page count mismatch");
  TORCH_CHECK(value_codebooks.size(2) == value_codes.size(3), "value subvec count mismatch");
  TORCH_CHECK(value_codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(value_codes.size(2) == value_page_size, "value code page_size mismatch");
  TORCH_CHECK(queries.size(0) == ranked_tokens.size(0), "position count mismatch");
  TORCH_CHECK(queries.size(1) == ranked_tokens.size(1), "head count mismatch");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(queries.size(2) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(
      queries.size(2) == value_codebooks.size(2) * value_codebooks.size(4),
      "value codebook dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(selection_page_size > 0, "selection_page_size must be positive");
  TORCH_CHECK(value_page_size > 0, "value_page_size must be positive");
  TORCH_CHECK(queries.size(1) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_causal_vpq_selected_attention_vpagesize_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
      value_codebooks.contiguous(),
      value_codes.contiguous(),
      page_starts.contiguous(),
      ranked_tokens.contiguous(),
      ranked_scores.contiguous(),
      group_size,
      query_start,
      static_prefix,
      static_suffix,
      selection_page_size,
      value_page_size,
      exact_value_top,
      scale);
}

torch::Tensor gqa_causal_vpq_selected_attention_vpagesize(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t selection_page_size,
    int64_t value_page_size,
    double scale) {
  return gqa_causal_vpq_selected_attention_mixed_vpagesize(
      queries,
      keys,
      values,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      group_size,
      query_start,
      static_prefix,
      static_suffix,
      selection_page_size,
      value_page_size,
      0,
      scale);
}

torch::Tensor gqa_causal_vpq_selected_tail_attention(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor key_codebooks,
    torch::Tensor key_codes,
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
    int64_t exact_value_top,
    double scale,
    double tail_blend) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(key_codebooks.is_cuda(), "key_codebooks must be CUDA");
  TORCH_CHECK(key_codes.is_cuda(), "key_codes must be CUDA");
  TORCH_CHECK(value_codebooks.is_cuda(), "value_codebooks must be CUDA");
  TORCH_CHECK(value_codes.is_cuda(), "value_codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(ranked_tokens.is_cuda(), "ranked_tokens must be CUDA");
  TORCH_CHECK(ranked_scores.is_cuda(), "ranked_scores must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(is_float_like(keys), "keys must be float32/float16/bfloat16");
  TORCH_CHECK(is_float_like(values), "values must be float32/float16/bfloat16");
  TORCH_CHECK(key_codebooks.scalar_type() == torch::kFloat32, "key_codebooks must be float32");
  TORCH_CHECK(value_codebooks.scalar_type() == torch::kFloat32, "value_codebooks must be float32");
  TORCH_CHECK(
      key_codes.scalar_type() == torch::kLong || key_codes.scalar_type() == torch::kUInt8,
      "key_codes must be int64 or uint8");
  TORCH_CHECK(
      value_codes.scalar_type() == torch::kLong || value_codes.scalar_type() == torch::kUInt8,
      "value_codes must be int64 or uint8");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(ranked_tokens.scalar_type() == torch::kLong, "ranked_tokens must be int64");
  TORCH_CHECK(ranked_scores.scalar_type() == torch::kFloat32, "ranked_scores must be float32");
  TORCH_CHECK(queries.dim() == 3, "queries shape must be [positions, heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(key_codebooks.dim() == 5, "key_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(key_codes.dim() == 4, "key_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(value_codebooks.dim() == 5, "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(ranked_tokens.dim() == 3, "ranked_tokens shape must be [positions, heads, selected]");
  TORCH_CHECK(ranked_scores.dim() == 3, "ranked_scores shape must be [positions, heads, selected]");
  TORCH_CHECK(keys.size(0) == values.size(0), "key/value kv-head count mismatch");
  TORCH_CHECK(keys.size(1) == values.size(1), "key/value token count mismatch");
  TORCH_CHECK(keys.size(2) == values.size(2), "key/value dim mismatch");
  TORCH_CHECK(key_codebooks.size(0) == key_codes.size(0), "key kv-head count mismatch");
  TORCH_CHECK(key_codebooks.size(0) == keys.size(0), "key codebook/key kv-head count mismatch");
  TORCH_CHECK(key_codebooks.size(1) == key_codes.size(1), "key page count mismatch");
  TORCH_CHECK(key_codebooks.size(2) == key_codes.size(3), "key subvec count mismatch");
  TORCH_CHECK(value_codebooks.size(0) == value_codes.size(0), "value kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value/key kv-head count mismatch");
  TORCH_CHECK(value_codebooks.size(1) == value_codes.size(1), "value page count mismatch");
  TORCH_CHECK(value_codebooks.size(1) == key_codebooks.size(1), "key/value page count mismatch");
  TORCH_CHECK(value_codebooks.size(2) == value_codes.size(3), "value subvec count mismatch");
  TORCH_CHECK(key_codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(key_codes.size(2) == page_size, "key code page_size mismatch");
  TORCH_CHECK(value_codes.size(2) == page_size, "value code page_size mismatch");
  TORCH_CHECK(queries.size(0) == ranked_tokens.size(0), "position count mismatch");
  TORCH_CHECK(queries.size(1) == ranked_tokens.size(1), "head count mismatch");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(queries.size(2) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(queries.size(2) == key_codebooks.size(2) * key_codebooks.size(4), "key codebook dim mismatch");
  TORCH_CHECK(queries.size(2) == value_codebooks.size(2) * value_codebooks.size(4), "value codebook dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(queries.size(1) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_causal_vpq_tail_attention_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
      key_codebooks.contiguous(),
      key_codes.contiguous(),
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
      exact_value_top,
      scale,
      tail_blend);
}

torch::Tensor gqa_causal_vpq_tail_attention(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor key_codebooks,
    torch::Tensor key_codes,
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
    double scale,
    double tail_blend) {
  return gqa_causal_vpq_selected_tail_attention(
      queries,
      keys,
      values,
      key_codebooks,
      key_codes,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      group_size,
      query_start,
      static_prefix,
      static_suffix,
      page_size,
      ranked_tokens.size(2),
      scale,
      tail_blend);
}

torch::Tensor gqa_causal_vpq_selected_tail_from_scores(
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
  TORCH_CHECK(is_float_like(keys), "keys must be float32/float16/bfloat16");
  TORCH_CHECK(is_float_like(values), "values must be float32/float16/bfloat16");
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
  TORCH_CHECK(value_codebooks.size(0) == keys.size(0), "value/key kv-head mismatch");
  TORCH_CHECK(value_codebooks.size(1) == value_codes.size(1), "value page count mismatch");
  TORCH_CHECK(value_codebooks.size(1) == page_starts.size(0), "page_starts count mismatch");
  TORCH_CHECK(value_codebooks.size(2) == value_codes.size(3), "value subvec count mismatch");
  TORCH_CHECK(value_codes.size(2) == page_size, "value code page_size mismatch");
  TORCH_CHECK(queries.size(0) == ranked_tokens.size(0), "position count mismatch");
  TORCH_CHECK(queries.size(1) == ranked_tokens.size(1), "head count mismatch");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(dense_pq_scores.size(0) == queries.size(0), "dense score position count mismatch");
  TORCH_CHECK(dense_pq_scores.size(1) == queries.size(1), "dense score head count mismatch");
  TORCH_CHECK(dense_pq_scores.size(2) == value_codebooks.size(1) * page_size, "dense score token count mismatch");
  TORCH_CHECK(queries.size(2) == keys.size(2), "query/key dim mismatch");
  TORCH_CHECK(queries.size(2) == value_codebooks.size(2) * value_codebooks.size(4), "value codebook dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(queries.size(1) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_causal_vpq_tail_from_scores_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
      dense_pq_scores.contiguous(),
      dense_pq_scores.new_empty({0}),
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
      page_starts.new_empty({0}),
      exact_value_top,
      0.0,
      scale,
      tail_blend);
}

torch::Tensor gqa_causal_vpq_selected_tail_from_scores_counts(
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
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t exact_value_top,
    double scale,
    double tail_blend) {
  TORCH_CHECK(exact_value_counts.is_cuda(), "exact_value_counts must be CUDA");
  TORCH_CHECK(exact_value_counts.scalar_type() == torch::kLong, "exact_value_counts must be int64");
  TORCH_CHECK(exact_value_counts.dim() == 2, "exact_value_counts shape must be [positions, heads]");
  TORCH_CHECK(exact_value_counts.size(0) == queries.size(0), "exact count position mismatch");
  TORCH_CHECK(exact_value_counts.size(1) == queries.size(1), "exact count head mismatch");
  return gqa_causal_vpq_tail_from_scores_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
      dense_pq_scores.contiguous(),
      dense_pq_scores.new_empty({0}),
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
      exact_value_counts.contiguous(),
      exact_value_top,
      0.0,
      scale,
      tail_blend);
}

torch::Tensor gqa_causal_vpq_selected_tail_from_scores_mass(
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
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale,
    double tail_blend) {
  TORCH_CHECK(exact_value_mass >= 0.0 && exact_value_mass <= 1.0, "exact_value_mass must be in [0, 1]");
  return gqa_causal_vpq_tail_from_scores_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
      dense_pq_scores.contiguous(),
      dense_pq_scores.new_empty({0}),
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
      page_starts.new_empty({0}),
      0,
      exact_value_mass,
      scale,
      tail_blend);
}

torch::Tensor gqa_causal_vpq_selected_tail_from_scores_mass_min(
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
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale,
    double tail_blend) {
  TORCH_CHECK(exact_value_mass >= 0.0 && exact_value_mass <= 1.0, "exact_value_mass must be in [0, 1]");
  TORCH_CHECK(exact_value_min_top >= 0, "exact_value_min_top must be non-negative");
  return gqa_causal_vpq_tail_from_scores_cuda(
      queries.contiguous(),
      keys.contiguous(),
      values.contiguous(),
      dense_pq_scores.contiguous(),
      dense_pq_scores.new_empty({0}),
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
      page_starts.new_empty({0}),
      exact_value_min_top,
      exact_value_mass,
      scale,
      tail_blend);
}

torch::Tensor gqa_causal_vpq_tail_from_scores(
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
    double scale,
    double tail_blend) {
  return gqa_causal_vpq_selected_tail_from_scores(
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
      query_start,
      static_prefix,
      static_suffix,
      page_size,
      ranked_tokens.size(2),
      scale,
      tail_blend);
}

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

torch::Tensor gqa_decode_geometric_accept_counts_vpq_mass_min_proxy(
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
    double exact_value_mass,
    int64_t exact_value_min_top,
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
  TORCH_CHECK(exact_value_mass >= 0.0 && exact_value_mass <= 1.0, "exact_value_mass must be in [0, 1]");
  TORCH_CHECK(exact_value_min_top >= 0, "exact_value_min_top must be non-negative");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_cuda(
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
      exact_value_mass,
      exact_value_min_top,
      scale,
      proxy_mass_min,
      proxy_tail_mass_max,
      pq_corr_min,
      pq_relrmse_max,
      calibrate_proxy,
      probe_includes_tail);
}

torch::Tensor gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits(
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
    double exact_value_mass,
    int64_t exact_value_min_top,
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
  TORCH_CHECK(ranked_logits.is_cuda(), "ranked_logits must be CUDA");
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
  TORCH_CHECK(ranked_logits.scalar_type() == torch::kFloat32, "ranked_logits must be float32");
  TORCH_CHECK(queries.dim() == 2, "queries shape must be [heads, dim]");
  TORCH_CHECK(keys.dim() == 3, "keys shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [kv_heads, tokens, dim]");
  TORCH_CHECK(dense_pq_scores.dim() == 2, "dense_pq_scores shape must be [heads, pages * page_size]");
  TORCH_CHECK(value_codebooks.dim() == 5, "value_codebooks shape must be [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(value_codes.dim() == 4, "value_codes shape must be [kv_heads, pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, selected]");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(ranked_logits.sizes() == ranked_tokens.sizes(), "ranked logits/tokens shape mismatch");
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
  TORCH_CHECK(exact_value_mass >= 0.0 && exact_value_mass <= 1.0, "exact_value_mass must be in [0, 1]");
  TORCH_CHECK(exact_value_min_top >= 0, "exact_value_min_top must be non-negative");
  TORCH_CHECK(queries.size(0) <= keys.size(0) * group_size, "heads exceed kv_heads * group_size");
  return gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_cuda(
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
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
      exact_value_mass,
      exact_value_min_top,
      scale,
      proxy_mass_min,
      proxy_tail_mass_max,
      pq_corr_min,
      pq_relrmse_max,
      calibrate_proxy,
      probe_includes_tail);
}

torch::Tensor gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds(
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
    torch::Tensor approx_exact_thresholds,
    torch::Tensor approx_exact_threshold_sels,
    torch::Tensor probe_exact_thresholds,
    torch::Tensor probe_exact_threshold_sels,
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
    double exact_value_mass,
    int64_t exact_value_min_top,
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
  TORCH_CHECK(ranked_logits.is_cuda(), "ranked_logits must be CUDA");
  TORCH_CHECK(approx_exact_thresholds.is_cuda(), "approx_exact_thresholds must be CUDA");
  TORCH_CHECK(approx_exact_threshold_sels.is_cuda(), "approx_exact_threshold_sels must be CUDA");
  TORCH_CHECK(probe_exact_thresholds.is_cuda(), "probe_exact_thresholds must be CUDA");
  TORCH_CHECK(probe_exact_threshold_sels.is_cuda(), "probe_exact_threshold_sels must be CUDA");
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
  TORCH_CHECK(approx_exact_thresholds.scalar_type() == torch::kFloat32, "approx thresholds must be float32");
  TORCH_CHECK(probe_exact_thresholds.scalar_type() == torch::kFloat32, "probe thresholds must be float32");
  TORCH_CHECK(approx_exact_threshold_sels.scalar_type() == torch::kLong, "approx threshold sels must be int64");
  TORCH_CHECK(probe_exact_threshold_sels.scalar_type() == torch::kLong, "probe threshold sels must be int64");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, selected]");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(ranked_logits.sizes() == ranked_tokens.sizes(), "ranked logits/tokens shape mismatch");
  TORCH_CHECK(approx_exact_thresholds.dim() == 2, "approx thresholds shape must be [heads, steps]");
  TORCH_CHECK(approx_exact_threshold_sels.sizes() == approx_exact_thresholds.sizes(), "approx threshold shape mismatch");
  TORCH_CHECK(probe_exact_thresholds.sizes() == approx_exact_thresholds.sizes(), "probe threshold shape mismatch");
  TORCH_CHECK(probe_exact_threshold_sels.sizes() == approx_exact_thresholds.sizes(), "probe threshold sel shape mismatch");
  TORCH_CHECK(approx_exact_thresholds.size(0) == queries.size(0), "threshold head count mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(granularity > 0, "granularity must be positive");
  return gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds_cuda(
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
      approx_exact_thresholds.contiguous(),
      approx_exact_threshold_sels.contiguous(),
      probe_exact_thresholds.contiguous(),
      probe_exact_threshold_sels.contiguous(),
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
      exact_value_mass,
      exact_value_min_top,
      scale,
      proxy_mass_min,
      proxy_tail_mass_max,
      pq_corr_min,
      pq_relrmse_max,
      calibrate_proxy,
      probe_includes_tail);
}

std::vector<torch::Tensor> gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds(
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
    torch::Tensor approx_exact_thresholds,
    torch::Tensor approx_exact_threshold_sels,
    torch::Tensor probe_exact_thresholds,
    torch::Tensor probe_exact_threshold_sels,
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
    double exact_value_mass,
    int64_t exact_value_min_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail,
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
  TORCH_CHECK(approx_exact_thresholds.is_cuda(), "approx_exact_thresholds must be CUDA");
  TORCH_CHECK(approx_exact_threshold_sels.is_cuda(), "approx_exact_threshold_sels must be CUDA");
  TORCH_CHECK(probe_exact_thresholds.is_cuda(), "probe_exact_thresholds must be CUDA");
  TORCH_CHECK(probe_exact_threshold_sels.is_cuda(), "probe_exact_threshold_sels must be CUDA");
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
  TORCH_CHECK(approx_exact_thresholds.scalar_type() == torch::kFloat32, "approx thresholds must be float32");
  TORCH_CHECK(probe_exact_thresholds.scalar_type() == torch::kFloat32, "probe thresholds must be float32");
  TORCH_CHECK(approx_exact_threshold_sels.scalar_type() == torch::kLong, "approx threshold sels must be int64");
  TORCH_CHECK(probe_exact_threshold_sels.scalar_type() == torch::kLong, "probe threshold sels must be int64");
  TORCH_CHECK(ranked_tokens.dim() == 2, "ranked_tokens shape must be [heads, selected]");
  TORCH_CHECK(ranked_scores.sizes() == ranked_tokens.sizes(), "ranked scores/tokens shape mismatch");
  TORCH_CHECK(ranked_logits.sizes() == ranked_tokens.sizes(), "ranked logits/tokens shape mismatch");
  TORCH_CHECK(approx_exact_thresholds.dim() == 2, "approx thresholds shape must be [heads, steps]");
  TORCH_CHECK(approx_exact_threshold_sels.sizes() == approx_exact_thresholds.sizes(), "approx threshold shape mismatch");
  TORCH_CHECK(probe_exact_thresholds.sizes() == approx_exact_thresholds.sizes(), "probe threshold shape mismatch");
  TORCH_CHECK(probe_exact_threshold_sels.sizes() == approx_exact_thresholds.sizes(), "probe threshold sel shape mismatch");
  TORCH_CHECK(approx_exact_thresholds.size(0) == queries.size(0), "threshold head count mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(granularity > 0, "granularity must be positive");
  return gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds_cuda(
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
      approx_exact_thresholds.contiguous(),
      approx_exact_threshold_sels.contiguous(),
      probe_exact_thresholds.contiguous(),
      probe_exact_threshold_sels.contiguous(),
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
      exact_value_mass,
      exact_value_min_top,
      scale,
      proxy_mass_min,
      proxy_tail_mass_max,
      pq_corr_min,
      pq_relrmse_max,
      calibrate_proxy,
      probe_includes_tail,
      tail_blend);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fullscan_pq_topk", &fullscan_pq_topk, "Fullscan page-PQ top-k selector (CUDA)");
  m.def(
      "fullscan_pq_topk_scores",
      &fullscan_pq_topk_scores,
      "Fullscan page-PQ top-k selector plus dense approximate score matrix (CUDA)");
  m.def(
      "gqa_fullscan_pq_topk",
      &gqa_fullscan_pq_topk,
      "Fullscan page-PQ top-k selector for grouped-query attention (CUDA)");
  m.def(
      "gqa_fullscan_pq_topk_scores",
      &gqa_fullscan_pq_topk_scores,
      "Fullscan page-PQ top-k selector plus dense approximate scores for grouped-query attention (CUDA)");
  m.def(
      "gqa_causal_fullscan_pq_topk",
      &gqa_causal_fullscan_pq_topk,
      "Causal fullscan page-PQ top-k selector for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_fullscan_pq_topk_scores",
      &gqa_causal_fullscan_pq_topk_scores,
      "Causal fullscan page-PQ top-k selector plus dense approximate scores for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_fullscan_pq_topk_fused",
      &gqa_causal_fullscan_pq_topk_fused,
      "Causal fused page-PQ top-k selector for grouped-query prefill, exact through budget 64 (CUDA)");
  m.def(
      "gqa_causal_fullscan_pq_topk_fused_force",
      &gqa_causal_fullscan_pq_topk_fused_force,
      "Causal fused page-PQ top-k selector with force_mode 0=auto, 1=smallscan, 2=localtopk (CUDA)");
  m.def(
      "gqa_causal_fullscan_pq_top_pages",
      &gqa_causal_fullscan_pq_top_pages,
      "Causal fullscan page-PQ top-page selector for grouped-query prefill (CUDA)");
  m.def(
      "exact_selected_attention",
      &exact_selected_attention,
      "Exact selected-token attention for one KV group (CUDA)");
  m.def(
      "gqa_causal_exact_selected_attention",
      &gqa_causal_exact_selected_attention,
      "Causal exact selected-token attention for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_attention",
      &gqa_causal_vpq_selected_attention,
      "Causal selected-token attention with page-local V-PQ values for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_attention_vpagesize",
      &gqa_causal_vpq_selected_attention_vpagesize,
      "Causal selected-token attention with separately paged V-PQ values for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_attention_mixed_vpagesize",
      &gqa_causal_vpq_selected_attention_mixed_vpagesize,
      "Causal selected-token attention with separately paged V-PQ values and fixed exact-value top-k (CUDA)");
  m.def(
      "gqa_causal_vpq_tail_attention",
      &gqa_causal_vpq_tail_attention,
      "Causal selected-token attention with compressed page-PQ tail for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_tail_attention",
      &gqa_causal_vpq_selected_tail_attention,
      "Causal selected-token attention with mixed selected V-PQ values and compressed page-PQ tail (CUDA)");
  m.def(
      "gqa_causal_vpq_tail_from_scores",
      &gqa_causal_vpq_tail_from_scores,
      "Causal selected-token attention with compressed V-PQ tail using precomputed PQ selector scores (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_tail_from_scores",
      &gqa_causal_vpq_selected_tail_from_scores,
      "Causal mixed selected V-PQ attention with compressed V-PQ tail using precomputed PQ selector scores (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_tail_from_scores_counts",
      &gqa_causal_vpq_selected_tail_from_scores_counts,
      "Causal mixed selected V-PQ/tail attention with per-query exact selected-value counts (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_tail_from_scores_mass",
      &gqa_causal_vpq_selected_tail_from_scores_mass,
      "Causal mixed selected V-PQ/tail attention with in-kernel selected-value mass exactness (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_tail_from_scores_mass_min",
      &gqa_causal_vpq_selected_tail_from_scores_mass_min,
      "Causal mixed selected V-PQ/tail attention with in-kernel selected-value mass exactness and min exact selected values (CUDA)");
  m.def(
      "gqa_decode_vpq_tail_from_scores",
      &gqa_decode_vpq_tail_from_scores,
      "Decode selected-token attention with compressed V-PQ tail using precomputed PQ selector scores (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_from_scores",
      &gqa_decode_vpq_selected_tail_from_scores,
      "Decode mixed selected V-PQ attention with compressed V-PQ tail using precomputed PQ selector scores (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_from_scores_counts",
      &gqa_decode_vpq_selected_tail_from_scores_counts,
      "Decode mixed selected V-PQ/tail attention with per-head exact selected-value counts (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_scores",
      &gqa_decode_vpq_selected_tail_agg_from_scores,
      "Decode mixed selected V-PQ attention with code-weight-aggregated compressed V-PQ tail (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_scores_counts",
      &gqa_decode_vpq_selected_tail_agg_from_scores_counts,
      "Decode mixed selected V-PQ attention with code-weight-aggregated tail and per-head exact selected-value counts (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts",
	      &gqa_decode_geometric_accept_counts,
	      "Decode strict-geometric confidence accepted selected-token counts (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq",
	      &gqa_decode_geometric_accept_counts_vpq,
	      "Decode strict-geometric confidence counts with selector-rank V-PQ selected values (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq_tail_stability",
	      &gqa_decode_geometric_accept_counts_vpq_tail_stability,
	      "Decode geometric confidence counts comparing compressed-tail outputs at tail/probe budgets (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq_proxy",
	      &gqa_decode_geometric_accept_counts_vpq_proxy,
	      "Decode geometric confidence counts with in-kernel proxy mass/correlation gating (CUDA)");
	  m.def(
	      "gqa_decode_ranked_exact_logits",
	      &gqa_decode_ranked_exact_logits,
	      "Decode exact QK logits for ranked candidates only (CUDA)");
	  m.def(
	      "gqa_decode_ranked_exact_logits_with_base_lse",
	      &gqa_decode_ranked_exact_logits_with_base_lse,
	      "Decode exact QK logits for ranked candidates and base-window logsumexp (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy",
	      &gqa_decode_geometric_accept_counts_vpq_mass_min_proxy,
	      "Decode geometric confidence counts with selected-mass V-PQ exactness and proxy gating (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits",
	      &gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits,
	      "Decode geometric confidence counts with selected-mass V-PQ exactness, proxy gating, and precomputed exact logits (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds",
	      &gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds,
	      "Decode geometric confidence counts with selected-mass V-PQ exactness, proxy gating, precomputed exact logits, and per-budget thresholds (CUDA)");
	  m.def(
	      "gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds",
	      &gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds,
	      "Decode geometric confidence and final selected/tail output with selected-mass V-PQ exactness, proxy gating, precomputed exact logits, and per-budget thresholds (CUDA)");
	  m.def(
	      "gqa_causal_geometric_accept_counts",
	      &gqa_causal_geometric_accept_counts,
	      "Causal prefill strict-geometric confidence accepted selected-token counts (CUDA)");
	  m.def(
	      "gqa_causal_geometric_accept_counts_vpq",
	      &gqa_causal_geometric_accept_counts_vpq,
	      "Causal prefill strict-geometric confidence counts with selector-rank V-PQ selected values (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_scores_mass",
      &gqa_decode_vpq_selected_tail_agg_from_scores_mass,
      "Decode mixed selected V-PQ attention with code-weight-aggregated tail and in-kernel selected-value mass exactness (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_scores_mass_min",
      &gqa_decode_vpq_selected_tail_agg_from_scores_mass_min,
      "Decode mixed selected V-PQ attention with code-weight-aggregated tail, in-kernel selected-value mass exactness, and min exact selected values (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_from_logits_mass_min",
      &gqa_decode_vpq_selected_from_logits_mass_min,
      "Decode selected-only mixed V-PQ attention using precomputed exact ranked logits and selected-value mass/min exactness (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_logits_mass_min",
      &gqa_decode_vpq_selected_tail_agg_from_logits_mass_min,
      "Decode mixed selected V-PQ attention with code-weight-aggregated tail using precomputed exact ranked logits and selected-value mass/min exactness (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_logits_mass_min_thresholds",
      &gqa_decode_vpq_selected_tail_agg_from_logits_mass_min_thresholds,
      "Decode mixed selected V-PQ attention with precomputed exact ranked logits and selected-value mass/min thresholds (CUDA)");
  m.def(
      "gqa_decode_fullscan_vpq_selected_tail_agg",
      &gqa_decode_fullscan_vpq_selected_tail_agg,
      "Decode fullscan page-PQ selector plus mixed selected V-PQ/tail attention in one native call (CUDA)");
  m.def(
      "gqa_decode_fullscan_vpq_selected_tail_agg_mass_min",
      &gqa_decode_fullscan_vpq_selected_tail_agg_mass_min,
      "Decode fullscan page-PQ selector plus mixed selected V-PQ/tail attention with selected-value mass/min exactness in one native call (CUDA)");
  m.def(
      "gqa_decode_scoreless_fullscan_vpq_tail",
      &gqa_decode_scoreless_fullscan_vpq_tail,
      "Decode fused page-PQ top-k plus scoreless mixed V-PQ/tail attention without materializing dense PQ scores (CUDA)");
  m.def(
      "gqa_exact_selected_attention",
      &gqa_exact_selected_attention,
      "Exact selected-token attention for grouped-query attention (CUDA)");
}
