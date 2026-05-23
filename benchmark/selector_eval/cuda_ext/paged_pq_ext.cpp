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

std::vector<torch::Tensor> selected_mass_thresholds_from_topk_cuda(
    torch::Tensor top_logits,
    torch::Tensor top_order,
    torch::Tensor prefix_lse,
    torch::Tensor prefix_valid_counts,
    torch::Tensor base_lse,
    torch::Tensor budgets,
    double exact_mass,
    int64_t min_top);

torch::Tensor joint_vprefix_outputs_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual,
    torch::Tensor exact_order,
    torch::Tensor v_budgets);

torch::Tensor joint_vprefix_outputs_from_risk_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual,
    torch::Tensor code_error,
    torch::Tensor v_budgets);

torch::Tensor joint_vprefix_outputs_from_grouped_risk_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor row_group_ids,
    torch::Tensor v_budgets);

torch::Tensor joint_vprefix_outputs_from_grouped_risk_batched_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets);

torch::Tensor joint_vprefix_outputs_from_grouped_risk_topk_batched_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    int64_t max_exact);

torch::Tensor joint_vpq_base_outputs_from_probs_cuda(
    torch::Tensor probs,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor fallback_tokens);

std::vector<torch::Tensor> joint_softmax_base_outputs_cuda(
    torch::Tensor score_grid,
    torch::Tensor values);
std::vector<torch::Tensor> joint_mixed_softmax_base_outputs_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    torch::Tensor values,
    bool calibrate);
std::vector<torch::Tensor> joint_mixed_softmax_base_outputs_rankpos_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    torch::Tensor values,
    bool calibrate);

std::vector<torch::Tensor> joint_select_policy_from_grouped_risk_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    torch::Tensor k_mb,
    torch::Tensor v_mb,
    int64_t k_count,
    int64_t heads,
    double threshold,
    int64_t policy_id);

std::vector<torch::Tensor> joint_select_policy_from_grouped_risk_batched_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    torch::Tensor k_mb,
    torch::Tensor v_mb,
    double threshold,
    int64_t policy_id);

torch::Tensor joint_select_policy_cuda(
    torch::Tensor output_grid,
    torch::Tensor k_mb,
    torch::Tensor v_mb,
    double threshold,
    int64_t policy_id);

std::vector<torch::Tensor> joint_select_policy_grouped_flat_cuda(
    torch::Tensor outputs_flat,
    torch::Tensor k_mb,
    torch::Tensor v_mb,
    int64_t k_count,
    int64_t heads,
    double threshold,
    int64_t policy_id);

std::vector<torch::Tensor> joint_select_policy_grouped_flat_no_mb_cuda(
    torch::Tensor outputs_flat,
    int64_t k_count,
    int64_t heads,
    double threshold,
    int64_t policy_id);

torch::Tensor joint_rank_prefix_tokens_cuda(
    torch::Tensor scores,
    torch::Tensor indexed_tokens,
    int64_t max_take);

torch::Tensor joint_mixed_score_grid_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate);

torch::Tensor joint_mixed_score_grid_rankpos_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate);

torch::Tensor joint_mixed_score_grid_no_exact_fill_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate);

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

std::vector<torch::Tensor> selected_mass_thresholds_from_topk(
    torch::Tensor top_logits,
    torch::Tensor top_order,
    torch::Tensor prefix_lse,
    torch::Tensor prefix_valid_counts,
    torch::Tensor base_lse,
    torch::Tensor budgets,
    double exact_mass,
    int64_t min_top) {
  TORCH_CHECK(top_logits.is_cuda(), "top_logits must be CUDA");
  TORCH_CHECK(top_order.is_cuda(), "top_order must be CUDA");
  TORCH_CHECK(prefix_lse.is_cuda(), "prefix_lse must be CUDA");
  TORCH_CHECK(prefix_valid_counts.is_cuda(), "prefix_valid_counts must be CUDA");
  TORCH_CHECK(base_lse.is_cuda(), "base_lse must be CUDA");
  TORCH_CHECK(budgets.is_cuda(), "budgets must be CUDA");
  TORCH_CHECK(top_logits.scalar_type() == torch::kFloat32, "top_logits must be float32");
  TORCH_CHECK(top_order.scalar_type() == torch::kLong, "top_order must be int64");
  TORCH_CHECK(prefix_lse.scalar_type() == torch::kFloat32, "prefix_lse must be float32");
  TORCH_CHECK(prefix_valid_counts.scalar_type() == torch::kLong, "prefix_valid_counts must be int64");
  TORCH_CHECK(base_lse.scalar_type() == torch::kFloat32, "base_lse must be float32");
  TORCH_CHECK(budgets.scalar_type() == torch::kLong, "budgets must be int64");
  TORCH_CHECK(top_logits.dim() == 2, "top_logits shape must be [heads, topk]");
  TORCH_CHECK(top_order.sizes() == top_logits.sizes(), "top_order/top_logits shape mismatch");
  TORCH_CHECK(prefix_lse.dim() == 2, "prefix_lse shape must be [heads, rank]");
  TORCH_CHECK(prefix_valid_counts.sizes() == prefix_lse.sizes(), "prefix valid/lse shape mismatch");
  TORCH_CHECK(base_lse.dim() == 1, "base_lse shape must be [heads]");
  TORCH_CHECK(budgets.dim() == 1, "budgets shape must be [steps]");
  TORCH_CHECK(prefix_lse.size(0) == top_logits.size(0), "prefix/topk head count mismatch");
  TORCH_CHECK(base_lse.size(0) == top_logits.size(0), "base_lse head count mismatch");
  return selected_mass_thresholds_from_topk_cuda(
      top_logits.contiguous(),
      top_order.contiguous(),
      prefix_lse.contiguous(),
      prefix_valid_counts.contiguous(),
      base_lse.contiguous(),
      budgets.contiguous(),
      exact_mass,
      min_top);
}

torch::Tensor joint_vprefix_outputs(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual,
    torch::Tensor exact_order,
    torch::Tensor v_budgets) {
  TORCH_CHECK(base_outputs.is_cuda(), "base_outputs must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(exact_order.is_cuda(), "exact_order must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(base_outputs.scalar_type() == torch::kFloat32, "base_outputs must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(residual.scalar_type() == torch::kFloat32, "residual must be float32");
  TORCH_CHECK(exact_order.scalar_type() == torch::kLong, "exact_order must be int64");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(base_outputs.dim() == 3, "base_outputs shape must be [k, heads, dim]");
  TORCH_CHECK(probs.dim() == 3, "probs shape must be [k, heads, context]");
  TORCH_CHECK(residual.dim() == 2, "residual shape must be [context, dim]");
  TORCH_CHECK(exact_order.dim() == 3, "exact_order shape must be [k, heads, max_exact]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [steps]");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs/base k count mismatch");
  TORCH_CHECK(probs.size(1) == base_outputs.size(1), "probs/base head count mismatch");
  TORCH_CHECK(exact_order.size(0) == base_outputs.size(0), "order/base k count mismatch");
  TORCH_CHECK(exact_order.size(1) == base_outputs.size(1), "order/base head count mismatch");
  TORCH_CHECK(residual.size(0) == probs.size(2), "residual/probs context mismatch");
  TORCH_CHECK(residual.size(1) == base_outputs.size(2), "residual/base dim mismatch");
  return joint_vprefix_outputs_cuda(
      base_outputs.contiguous(),
      probs.contiguous(),
      residual.contiguous(),
      exact_order.contiguous(),
      v_budgets.contiguous());
}

torch::Tensor joint_vprefix_outputs_from_risk(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual,
    torch::Tensor code_error,
    torch::Tensor v_budgets) {
  TORCH_CHECK(base_outputs.is_cuda(), "base_outputs must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(code_error.is_cuda(), "code_error must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(base_outputs.scalar_type() == torch::kFloat32, "base_outputs must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(residual.scalar_type() == torch::kFloat32, "residual must be float32");
  TORCH_CHECK(code_error.scalar_type() == torch::kFloat32, "code_error must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(base_outputs.dim() == 3, "base_outputs shape must be [k, heads, dim]");
  TORCH_CHECK(probs.dim() == 3, "probs shape must be [k, heads, context]");
  TORCH_CHECK(residual.dim() == 2, "residual shape must be [context, dim]");
  TORCH_CHECK(code_error.dim() == 1, "code_error shape must be [context]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [steps]");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs/base k count mismatch");
  TORCH_CHECK(probs.size(1) == base_outputs.size(1), "probs/base head count mismatch");
  TORCH_CHECK(residual.size(0) == probs.size(2), "residual/probs context mismatch");
  TORCH_CHECK(code_error.size(0) == probs.size(2), "code_error/probs context mismatch");
  TORCH_CHECK(residual.size(1) == base_outputs.size(2), "residual/base dim mismatch");
  return joint_vprefix_outputs_from_risk_cuda(
      base_outputs.contiguous(),
      probs.contiguous(),
      residual.contiguous(),
      code_error.contiguous(),
      v_budgets.contiguous());
}

torch::Tensor joint_vprefix_outputs_from_grouped_risk(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor row_group_ids,
    torch::Tensor v_budgets) {
  TORCH_CHECK(base_outputs.is_cuda(), "base_outputs must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(residual_groups.is_cuda(), "residual_groups must be CUDA");
  TORCH_CHECK(code_error_groups.is_cuda(), "code_error_groups must be CUDA");
  TORCH_CHECK(row_group_ids.is_cuda(), "row_group_ids must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(base_outputs.scalar_type() == torch::kFloat32, "base_outputs must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(residual_groups.scalar_type() == torch::kFloat32, "residual_groups must be float32");
  TORCH_CHECK(code_error_groups.scalar_type() == torch::kFloat32, "code_error_groups must be float32");
  TORCH_CHECK(row_group_ids.scalar_type() == torch::kLong, "row_group_ids must be int64");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(base_outputs.dim() == 2, "base_outputs shape must be [rows, dim]");
  TORCH_CHECK(probs.dim() == 2, "probs shape must be [rows, context]");
  TORCH_CHECK(residual_groups.dim() == 3, "residual_groups shape must be [groups, context, dim]");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context]");
  TORCH_CHECK(row_group_ids.dim() == 1, "row_group_ids shape must be [rows]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [steps]");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs/base row count mismatch");
  TORCH_CHECK(row_group_ids.size(0) == base_outputs.size(0), "row_group_ids/base row count mismatch");
  TORCH_CHECK(residual_groups.size(0) == code_error_groups.size(0), "residual/code_error group count mismatch");
  TORCH_CHECK(residual_groups.size(1) == probs.size(1), "residual/probs context mismatch");
  TORCH_CHECK(code_error_groups.size(1) == probs.size(1), "code_error/probs context mismatch");
  TORCH_CHECK(residual_groups.size(2) == base_outputs.size(1), "residual/base dim mismatch");
  return joint_vprefix_outputs_from_grouped_risk_cuda(
      base_outputs.contiguous(),
      probs.contiguous(),
      residual_groups.contiguous(),
      code_error_groups.contiguous(),
      row_group_ids.contiguous(),
      v_budgets.contiguous());
}

torch::Tensor joint_vprefix_outputs_from_grouped_risk_batched(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets) {
  TORCH_CHECK(base_outputs.is_cuda(), "base_outputs must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(residual_groups.is_cuda(), "residual_groups must be CUDA");
  TORCH_CHECK(code_error_groups.is_cuda(), "code_error_groups must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(base_outputs.scalar_type() == torch::kFloat32, "base_outputs must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(residual_groups.scalar_type() == torch::kFloat32, "residual_groups must be float32");
  TORCH_CHECK(code_error_groups.scalar_type() == torch::kFloat32, "code_error_groups must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(base_outputs.dim() == 4, "base_outputs shape must be [groups, k, heads, dim]");
  TORCH_CHECK(probs.dim() == 4, "probs shape must be [groups, k, heads, context]");
  TORCH_CHECK(residual_groups.dim() == 3, "residual_groups shape must be [groups, context, dim]");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [steps]");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs/base group count mismatch");
  TORCH_CHECK(probs.size(1) == base_outputs.size(1), "probs/base k count mismatch");
  TORCH_CHECK(probs.size(2) == base_outputs.size(2), "probs/base head count mismatch");
  TORCH_CHECK(residual_groups.size(0) == base_outputs.size(0), "residual/base group count mismatch");
  TORCH_CHECK(code_error_groups.size(0) == base_outputs.size(0), "code_error/base group count mismatch");
  TORCH_CHECK(residual_groups.size(1) == probs.size(3), "residual/probs context mismatch");
  TORCH_CHECK(code_error_groups.size(1) == probs.size(3), "code_error/probs context mismatch");
  TORCH_CHECK(residual_groups.size(2) == base_outputs.size(3), "residual/base dim mismatch");
  return joint_vprefix_outputs_from_grouped_risk_batched_cuda(
      base_outputs.contiguous(),
      probs.contiguous(),
      residual_groups.contiguous(),
      code_error_groups.contiguous(),
      v_budgets.contiguous());
}

torch::Tensor joint_vprefix_outputs_from_grouped_risk_topk_batched(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    int64_t max_exact) {
  TORCH_CHECK(base_outputs.is_cuda(), "base_outputs must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(residual_groups.is_cuda(), "residual_groups must be CUDA");
  TORCH_CHECK(code_error_groups.is_cuda(), "code_error_groups must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(base_outputs.scalar_type() == torch::kFloat32, "base_outputs must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(residual_groups.scalar_type() == torch::kFloat32, "residual_groups must be float32");
  TORCH_CHECK(code_error_groups.scalar_type() == torch::kFloat32, "code_error_groups must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(base_outputs.dim() == 4, "base_outputs shape must be [groups, k, heads, dim]");
  TORCH_CHECK(probs.dim() == 4, "probs shape must be [groups, k, heads, context]");
  TORCH_CHECK(residual_groups.dim() == 3, "residual_groups shape must be [groups, context, dim]");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [steps]");
  TORCH_CHECK(max_exact >= 0, "max_exact must be non-negative");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs/base group count mismatch");
  TORCH_CHECK(probs.size(1) == base_outputs.size(1), "probs/base k count mismatch");
  TORCH_CHECK(probs.size(2) == base_outputs.size(2), "probs/base head count mismatch");
  TORCH_CHECK(residual_groups.size(0) == base_outputs.size(0), "residual/base group count mismatch");
  TORCH_CHECK(code_error_groups.size(0) == base_outputs.size(0), "code_error/base group count mismatch");
  TORCH_CHECK(residual_groups.size(1) == probs.size(3), "residual/probs context mismatch");
  TORCH_CHECK(code_error_groups.size(1) == probs.size(3), "code_error/probs context mismatch");
  TORCH_CHECK(residual_groups.size(2) == base_outputs.size(3), "residual/base dim mismatch");
  return joint_vprefix_outputs_from_grouped_risk_topk_batched_cuda(
      base_outputs.contiguous(),
      probs.contiguous(),
      residual_groups.contiguous(),
      code_error_groups.contiguous(),
      v_budgets.contiguous(),
      max_exact);
}

torch::Tensor joint_vpq_base_outputs_from_probs(
    torch::Tensor probs,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor fallback_tokens) {
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(value_codebooks.is_cuda(), "value_codebooks must be CUDA");
  TORCH_CHECK(value_codes.is_cuda(), "value_codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(fallback_tokens.is_cuda(), "fallback_tokens must be CUDA");
  return joint_vpq_base_outputs_from_probs_cuda(
      probs.contiguous(),
      values.contiguous(),
      value_codebooks.contiguous(),
      value_codes.contiguous(),
      page_starts.contiguous(),
      fallback_tokens.contiguous());
}

std::vector<torch::Tensor> joint_softmax_base_outputs(
    torch::Tensor score_grid,
    torch::Tensor values) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(score_grid.dim() == 3, "score_grid shape must be [k_count, heads, context_len]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  TORCH_CHECK(values.size(0) >= score_grid.size(2), "values length must cover score_grid context_len");
  return joint_softmax_base_outputs_cuda(score_grid.contiguous(), values.contiguous());
}

std::vector<torch::Tensor> joint_mixed_softmax_base_outputs(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    torch::Tensor values,
    bool calibrate) {
  TORCH_CHECK(exact_scores.is_cuda(), "exact_scores must be CUDA");
  TORCH_CHECK(pq_logits.is_cuda(), "pq_logits must be CUDA");
  TORCH_CHECK(y_indexed.is_cuda(), "y_indexed must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(base_tokens.is_cuda(), "base_tokens must be CUDA");
  TORCH_CHECK(ranked_prefix_tokens.is_cuda(), "ranked_prefix_tokens must be CUDA");
  TORCH_CHECK(k_take_counts.is_cuda(), "k_take_counts must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(exact_scores.scalar_type() == torch::kFloat32, "exact_scores must be float32");
  TORCH_CHECK(pq_logits.scalar_type() == torch::kFloat32, "pq_logits must be float32");
  TORCH_CHECK(y_indexed.scalar_type() == torch::kFloat32, "y_indexed must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(base_tokens.scalar_type() == torch::kLong, "base_tokens must be int64");
  TORCH_CHECK(ranked_prefix_tokens.scalar_type() == torch::kLong, "ranked_prefix_tokens must be int64");
  TORCH_CHECK(k_take_counts.scalar_type() == torch::kLong, "k_take_counts must be int64");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(exact_scores.dim() == 2, "exact_scores shape must be [heads, context_len]");
  TORCH_CHECK(pq_logits.dim() == 2, "pq_logits shape must be [heads, indexed_count]");
  TORCH_CHECK(y_indexed.sizes() == pq_logits.sizes(), "y_indexed must match pq_logits shape");
  TORCH_CHECK(indexed_tokens.dim() == 1, "indexed_tokens shape must be [indexed_count]");
  TORCH_CHECK(base_tokens.dim() == 1, "base_tokens shape must be [base_count]");
  TORCH_CHECK(ranked_prefix_tokens.dim() == 2, "ranked_prefix_tokens shape must be [heads, max_rank_take]");
  TORCH_CHECK(k_take_counts.dim() == 1, "k_take_counts shape must be [k_count]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  TORCH_CHECK(pq_logits.size(0) == exact_scores.size(0), "pq_logits heads must match exact_scores");
  TORCH_CHECK(y_indexed.size(0) == exact_scores.size(0), "y_indexed heads must match exact_scores");
  TORCH_CHECK(indexed_tokens.size(0) == pq_logits.size(1), "indexed_tokens length must match pq_logits indexed_count");
  TORCH_CHECK(ranked_prefix_tokens.size(0) == exact_scores.size(0), "ranked_prefix_tokens heads must match exact_scores");
  TORCH_CHECK(values.size(0) >= exact_scores.size(1), "values length must cover context_len");
  return joint_mixed_softmax_base_outputs_cuda(
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      y_indexed.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      values.contiguous(),
      calibrate);
}

std::vector<torch::Tensor> joint_mixed_softmax_base_outputs_rankpos(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    torch::Tensor values,
    bool calibrate) {
  TORCH_CHECK(exact_scores.is_cuda(), "exact_scores must be CUDA");
  TORCH_CHECK(pq_logits.is_cuda(), "pq_logits must be CUDA");
  TORCH_CHECK(y_indexed.is_cuda(), "y_indexed must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(base_tokens.is_cuda(), "base_tokens must be CUDA");
  TORCH_CHECK(ranked_prefix_tokens.is_cuda(), "ranked_prefix_tokens must be CUDA");
  TORCH_CHECK(k_take_counts.is_cuda(), "k_take_counts must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(exact_scores.scalar_type() == torch::kFloat32, "exact_scores must be float32");
  TORCH_CHECK(pq_logits.scalar_type() == torch::kFloat32, "pq_logits must be float32");
  TORCH_CHECK(y_indexed.scalar_type() == torch::kFloat32, "y_indexed must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(base_tokens.scalar_type() == torch::kLong, "base_tokens must be int64");
  TORCH_CHECK(ranked_prefix_tokens.scalar_type() == torch::kLong, "ranked_prefix_tokens must be int64");
  TORCH_CHECK(k_take_counts.scalar_type() == torch::kLong, "k_take_counts must be int64");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(exact_scores.dim() == 2, "exact_scores shape must be [heads, context_len]");
  TORCH_CHECK(pq_logits.dim() == 2, "pq_logits shape must be [heads, indexed_count]");
  TORCH_CHECK(y_indexed.sizes() == pq_logits.sizes(), "y_indexed must match pq_logits shape");
  TORCH_CHECK(indexed_tokens.dim() == 1, "indexed_tokens shape must be [indexed_count]");
  TORCH_CHECK(base_tokens.dim() == 1, "base_tokens shape must be [base_count]");
  TORCH_CHECK(ranked_prefix_tokens.dim() == 2, "ranked_prefix_tokens shape must be [heads, max_rank_take]");
  TORCH_CHECK(k_take_counts.dim() == 1, "k_take_counts shape must be [k_count]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  TORCH_CHECK(pq_logits.size(0) == exact_scores.size(0), "pq_logits heads must match exact_scores");
  TORCH_CHECK(y_indexed.size(0) == exact_scores.size(0), "y_indexed heads must match exact_scores");
  TORCH_CHECK(indexed_tokens.size(0) == pq_logits.size(1), "indexed_tokens length must match pq_logits indexed_count");
  TORCH_CHECK(ranked_prefix_tokens.size(0) == exact_scores.size(0), "ranked_prefix_tokens heads must match exact_scores");
  TORCH_CHECK(values.size(0) >= exact_scores.size(1), "values length must cover context_len");
  return joint_mixed_softmax_base_outputs_rankpos_cuda(
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      y_indexed.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      values.contiguous(),
      calibrate);
}

torch::Tensor joint_select_policy(
    torch::Tensor output_grid,
    torch::Tensor k_mb,
    torch::Tensor v_mb,
    double threshold,
    int64_t policy_id) {
  TORCH_CHECK(output_grid.is_cuda(), "output_grid must be CUDA");
  TORCH_CHECK(k_mb.is_cuda(), "k_mb must be CUDA");
  TORCH_CHECK(v_mb.is_cuda(), "v_mb must be CUDA");
  TORCH_CHECK(output_grid.scalar_type() == torch::kFloat32, "output_grid must be float32");
  TORCH_CHECK(k_mb.scalar_type() == torch::kFloat32, "k_mb must be float32");
  TORCH_CHECK(v_mb.scalar_type() == torch::kFloat32, "v_mb must be float32");
  TORCH_CHECK(output_grid.dim() == 4, "output_grid shape must be [k, v, heads, dim]");
  TORCH_CHECK(k_mb.dim() == 1, "k_mb shape must be [k]");
  TORCH_CHECK(v_mb.dim() == 1, "v_mb shape must be [v]");
  TORCH_CHECK(k_mb.size(0) == output_grid.size(0), "k_mb/output k count mismatch");
  TORCH_CHECK(v_mb.size(0) == output_grid.size(1), "v_mb/output v count mismatch");
  TORCH_CHECK(policy_id >= 0 && policy_id <= 4, "policy_id must be 0..4");
  return joint_select_policy_cuda(
      output_grid.contiguous(),
      k_mb.contiguous(),
      v_mb.contiguous(),
      threshold,
      policy_id);
}

std::vector<torch::Tensor> joint_select_policy_grouped_flat(
    torch::Tensor outputs_flat,
    torch::Tensor k_mb,
    torch::Tensor v_mb,
    int64_t k_count,
    int64_t heads,
    double threshold,
    int64_t policy_id) {
  TORCH_CHECK(outputs_flat.is_cuda(), "outputs_flat must be CUDA");
  TORCH_CHECK(k_mb.is_cuda(), "k_mb must be CUDA");
  TORCH_CHECK(v_mb.is_cuda(), "v_mb must be CUDA");
  TORCH_CHECK(outputs_flat.scalar_type() == torch::kFloat32, "outputs_flat must be float32");
  TORCH_CHECK(k_mb.scalar_type() == torch::kFloat32, "k_mb must be float32");
  TORCH_CHECK(v_mb.scalar_type() == torch::kFloat32, "v_mb must be float32");
  TORCH_CHECK(outputs_flat.dim() == 3, "outputs_flat shape must be [groups*k*heads, v, dim]");
  TORCH_CHECK(k_mb.dim() == 2, "k_mb shape must be [groups, k]");
  TORCH_CHECK(v_mb.dim() == 2, "v_mb shape must be [groups, v]");
  TORCH_CHECK(k_count > 0, "k_count must be positive");
  TORCH_CHECK(heads > 0, "heads must be positive");
  TORCH_CHECK(k_mb.size(1) == k_count, "k_mb/k_count mismatch");
  TORCH_CHECK(v_mb.size(0) == k_mb.size(0), "v_mb/k_mb group count mismatch");
  TORCH_CHECK(outputs_flat.size(0) == k_mb.size(0) * k_count * heads, "outputs_flat row count mismatch");
  TORCH_CHECK(outputs_flat.size(1) == v_mb.size(1), "outputs_flat/v count mismatch");
  TORCH_CHECK(policy_id >= 0 && policy_id <= 4, "policy_id must be 0..4");
  return joint_select_policy_grouped_flat_cuda(
      outputs_flat.contiguous(),
      k_mb.contiguous(),
      v_mb.contiguous(),
      k_count,
      heads,
      threshold,
      policy_id);
}

std::vector<torch::Tensor> joint_select_policy_grouped_flat_no_mb(
    torch::Tensor outputs_flat,
    int64_t k_count,
    int64_t heads,
    double threshold,
    int64_t policy_id) {
  TORCH_CHECK(outputs_flat.is_cuda(), "outputs_flat must be CUDA");
  TORCH_CHECK(outputs_flat.scalar_type() == torch::kFloat32, "outputs_flat must be float32");
  TORCH_CHECK(outputs_flat.dim() == 3, "outputs_flat shape must be [groups*k*heads, v, dim]");
  TORCH_CHECK(k_count > 0, "k_count must be positive");
  TORCH_CHECK(heads > 0, "heads must be positive");
  TORCH_CHECK(policy_id >= 0 && policy_id <= 3, "policy_id must be a non-MB policy 0..3");
  TORCH_CHECK(outputs_flat.size(0) % (k_count * heads) == 0, "outputs_flat row count mismatch");
  return joint_select_policy_grouped_flat_no_mb_cuda(
      outputs_flat.contiguous(),
      k_count,
      heads,
      threshold,
      policy_id);
}

std::vector<torch::Tensor> joint_select_policy_from_grouped_risk(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    torch::Tensor k_mb,
    torch::Tensor v_mb,
    int64_t k_count,
    int64_t heads,
    double threshold,
    int64_t policy_id) {
  TORCH_CHECK(base_outputs.is_cuda(), "base_outputs must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(residual_groups.is_cuda(), "residual_groups must be CUDA");
  TORCH_CHECK(code_error_groups.is_cuda(), "code_error_groups must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(k_mb.is_cuda(), "k_mb must be CUDA");
  TORCH_CHECK(v_mb.is_cuda(), "v_mb must be CUDA");
  TORCH_CHECK(base_outputs.scalar_type() == torch::kFloat32, "base_outputs must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(residual_groups.scalar_type() == torch::kFloat32, "residual_groups must be float32");
  TORCH_CHECK(code_error_groups.scalar_type() == torch::kFloat32, "code_error_groups must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(k_mb.scalar_type() == torch::kFloat32, "k_mb must be float32");
  TORCH_CHECK(v_mb.scalar_type() == torch::kFloat32, "v_mb must be float32");
  TORCH_CHECK(base_outputs.dim() == 2, "base_outputs shape must be [groups*k*heads, dim]");
  TORCH_CHECK(probs.dim() == 2, "probs shape must be [groups*k*heads, context]");
  TORCH_CHECK(residual_groups.dim() == 3, "residual_groups shape must be [groups, context, dim]");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v]");
  TORCH_CHECK(k_mb.dim() == 2, "k_mb shape must be [groups, k]");
  TORCH_CHECK(v_mb.dim() == 2, "v_mb shape must be [groups, v]");
  TORCH_CHECK(k_count > 0, "k_count must be positive");
  TORCH_CHECK(heads > 0, "heads must be positive");
  TORCH_CHECK(k_mb.size(1) == k_count, "k_mb/k_count mismatch");
  TORCH_CHECK(v_mb.size(0) == k_mb.size(0), "v_mb/k_mb group count mismatch");
  TORCH_CHECK(v_mb.size(1) == v_budgets.size(0), "v_mb/v_budgets count mismatch");
  TORCH_CHECK(base_outputs.size(0) == k_mb.size(0) * k_count * heads, "base_outputs row count mismatch");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs row count mismatch");
  TORCH_CHECK(residual_groups.size(0) == k_mb.size(0), "residual_groups group count mismatch");
  TORCH_CHECK(code_error_groups.size(0) == k_mb.size(0), "code_error_groups group count mismatch");
  TORCH_CHECK(residual_groups.size(1) == probs.size(1), "residual_groups context mismatch");
  TORCH_CHECK(code_error_groups.size(1) == probs.size(1), "code_error_groups context mismatch");
  TORCH_CHECK(residual_groups.size(2) == base_outputs.size(1), "residual_groups dim mismatch");
  TORCH_CHECK(policy_id >= 0 && policy_id <= 4, "policy_id must be 0..4");
  return joint_select_policy_from_grouped_risk_cuda(
      base_outputs.contiguous(),
      probs.contiguous(),
      residual_groups.contiguous(),
      code_error_groups.contiguous(),
      v_budgets.contiguous(),
      k_mb.contiguous(),
      v_mb.contiguous(),
      k_count,
      heads,
      threshold,
      policy_id);
}

std::vector<torch::Tensor> joint_select_policy_from_grouped_risk_batched(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    torch::Tensor k_mb,
    torch::Tensor v_mb,
    double threshold,
    int64_t policy_id) {
  TORCH_CHECK(base_outputs.is_cuda(), "base_outputs must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(residual_groups.is_cuda(), "residual_groups must be CUDA");
  TORCH_CHECK(code_error_groups.is_cuda(), "code_error_groups must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(k_mb.is_cuda(), "k_mb must be CUDA");
  TORCH_CHECK(v_mb.is_cuda(), "v_mb must be CUDA");
  TORCH_CHECK(base_outputs.scalar_type() == torch::kFloat32, "base_outputs must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(residual_groups.scalar_type() == torch::kFloat32, "residual_groups must be float32");
  TORCH_CHECK(code_error_groups.scalar_type() == torch::kFloat32, "code_error_groups must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(k_mb.scalar_type() == torch::kFloat32, "k_mb must be float32");
  TORCH_CHECK(v_mb.scalar_type() == torch::kFloat32, "v_mb must be float32");
  TORCH_CHECK(base_outputs.dim() == 4, "base_outputs shape must be [groups, k, heads, dim]");
  TORCH_CHECK(probs.dim() == 4, "probs shape must be [groups, k, heads, context]");
  TORCH_CHECK(residual_groups.dim() == 3, "residual_groups shape must be [groups, context, dim]");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v]");
  TORCH_CHECK(k_mb.dim() == 2, "k_mb shape must be [groups, k]");
  TORCH_CHECK(v_mb.dim() == 2, "v_mb shape must be [groups, v]");
  TORCH_CHECK(base_outputs.size(0) == k_mb.size(0), "base_outputs/k_mb group mismatch");
  TORCH_CHECK(base_outputs.size(1) == k_mb.size(1), "base_outputs/k_mb k-count mismatch");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs group mismatch");
  TORCH_CHECK(probs.size(1) == base_outputs.size(1), "probs k-count mismatch");
  TORCH_CHECK(probs.size(2) == base_outputs.size(2), "probs head-count mismatch");
  TORCH_CHECK(residual_groups.size(0) == base_outputs.size(0), "residual_groups group mismatch");
  TORCH_CHECK(code_error_groups.size(0) == base_outputs.size(0), "code_error_groups group mismatch");
  TORCH_CHECK(residual_groups.size(1) == probs.size(3), "residual_groups context mismatch");
  TORCH_CHECK(code_error_groups.size(1) == probs.size(3), "code_error_groups context mismatch");
  TORCH_CHECK(residual_groups.size(2) == base_outputs.size(3), "residual_groups dim mismatch");
  TORCH_CHECK(v_mb.size(0) == base_outputs.size(0), "v_mb group mismatch");
  TORCH_CHECK(v_mb.size(1) == v_budgets.size(0), "v_mb/v_budgets count mismatch");
  TORCH_CHECK(policy_id >= 0 && policy_id <= 4, "policy_id must be 0..4");
  return joint_select_policy_from_grouped_risk_batched_cuda(
      base_outputs.contiguous(),
      probs.contiguous(),
      residual_groups.contiguous(),
      code_error_groups.contiguous(),
      v_budgets.contiguous(),
      k_mb.contiguous(),
      v_mb.contiguous(),
      threshold,
      policy_id);
}

torch::Tensor joint_rank_prefix_tokens(
    torch::Tensor scores,
    torch::Tensor indexed_tokens,
    int64_t max_take) {
  TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32, "scores must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(scores.dim() == 2, "scores shape must be [heads, tokens]");
  TORCH_CHECK(indexed_tokens.dim() == 1, "indexed_tokens shape must be [tokens]");
  TORCH_CHECK(scores.size(1) == indexed_tokens.size(0), "scores/indexed_tokens token count mismatch");
  TORCH_CHECK(max_take >= 0, "max_take must be non-negative");
  return joint_rank_prefix_tokens_cuda(scores.contiguous(), indexed_tokens.contiguous(), max_take);
}

torch::Tensor joint_mixed_score_grid(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate) {
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
  return joint_mixed_score_grid_cuda(
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      y_indexed.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      calibrate);
}

torch::Tensor joint_mixed_score_grid_rankpos(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate) {
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
  return joint_mixed_score_grid_rankpos_cuda(
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      y_indexed.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      calibrate);
}

torch::Tensor joint_mixed_score_grid_no_exact_fill(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate) {
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
  return joint_mixed_score_grid_no_exact_fill_cuda(
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      y_indexed.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      calibrate);
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
	      "selected_mass_thresholds_from_topk",
	      &selected_mass_thresholds_from_topk,
	      "Build selected-mass exact-V thresholds from sorted top-k exact logits (CUDA)");
	  m.def(
	      "joint_vprefix_outputs",
	      &joint_vprefix_outputs,
	      "Build residual-risk exact-V prefix output grid from top-risk token order (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_from_risk",
	      &joint_vprefix_outputs_from_risk,
	      "Build residual-risk exact-V output grid by sorting risk scores natively (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_from_grouped_risk",
	      &joint_vprefix_outputs_from_grouped_risk,
	      "Build grouped residual-risk exact-V output grid by sorting risk scores natively (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_from_grouped_risk_batched",
	      &joint_vprefix_outputs_from_grouped_risk_batched,
	      "Build grouped residual-risk exact-V output grid from batched grouped rows (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_from_grouped_risk_topk_batched",
	      &joint_vprefix_outputs_from_grouped_risk_topk_batched,
	      "Build grouped residual-risk exact-V output grid using top-k risk rows (CUDA)");
	  m.def(
	      "joint_vpq_base_outputs_from_probs",
	      &joint_vpq_base_outputs_from_probs,
	      "Build V-PQ reconstructed base outputs from probabilities by page/code aggregation (CUDA)");
	  m.def(
	      "joint_softmax_base_outputs",
	      &joint_softmax_base_outputs,
	      "Compute softmax probabilities and reconstructed base outputs from a joint score grid (CUDA)");
	  m.def(
	      "joint_mixed_softmax_base_outputs",
	      &joint_mixed_softmax_base_outputs,
	      "Build mixed joint score rows implicitly, softmax them, and compute reconstructed base outputs (CUDA)");
	  m.def(
	      "joint_mixed_softmax_base_outputs_rankpos",
	      &joint_mixed_softmax_base_outputs_rankpos,
	      "Build mixed joint score rows with rank-position metadata, softmax them, and compute reconstructed base outputs (CUDA)");
	  m.def(
	      "joint_mixed_score_grid",
	      &joint_mixed_score_grid,
	      "Build joint K/V mixed exact-K plus calibrated K-PQ score grid (CUDA)");
	  m.def(
	      "joint_mixed_score_grid_rankpos",
	      &joint_mixed_score_grid_rankpos,
	      "Build joint K/V mixed score grid using rank-position selected-token metadata (CUDA)");
	  m.def(
	      "joint_mixed_score_grid_no_exact_fill",
	      &joint_mixed_score_grid_no_exact_fill,
	      "Build joint K/V mixed score grid without initial exact-fill; caller must guarantee indexed/base coverage (CUDA)");
	  m.def(
	      "joint_select_policy",
	      &joint_select_policy,
	      "Select final joint K/V budget indices from an output grid using the online stability policy (CUDA)");
  m.def(
      "joint_select_policy_grouped_flat",
      &joint_select_policy_grouped_flat,
      "Select final joint K/V budget indices and outputs from grouped flat output grids (CUDA)");
  m.def(
      "joint_select_policy_grouped_flat_no_mb",
      &joint_select_policy_grouped_flat_no_mb,
      "Select final joint K/V budget indices and outputs from grouped flat output grids for non-MB policies (CUDA)");
  m.def(
      "joint_rank_prefix_tokens",
      &joint_rank_prefix_tokens,
      "Build per-head ranked token prefixes from dense PQ scores using native segmented sort (CUDA)");
  m.def(
      "joint_select_policy_from_grouped_risk",
      &joint_select_policy_from_grouped_risk,
	      "Select final joint K/V budgets and outputs directly from grouped residual-risk rows (CUDA)");
	  m.def(
	      "joint_select_policy_from_grouped_risk_batched",
	      &joint_select_policy_from_grouped_risk_batched,
	      "Select final joint K/V budgets and outputs directly from batched grouped residual-risk rows (CUDA)");
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
