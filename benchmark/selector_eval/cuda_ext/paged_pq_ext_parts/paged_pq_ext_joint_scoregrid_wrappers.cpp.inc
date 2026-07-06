torch::Tensor joint_grouped_accounting_sums(
    torch::Tensor final_indices,
    torch::Tensor selected_counts_by_ki,
    torch::Tensor v_budgets,
    int64_t context_len,
    int64_t head_dim,
    int64_t key_bytes,
    int64_t value_bytes,
    double selector_mb,
    double v_pq_codebook_mb,
    double metadata_mb,
    int64_t actual_value_subvecs,
    int64_t code_bytes) {
  TORCH_CHECK(final_indices.is_cuda(), "final_indices must be CUDA");
  TORCH_CHECK(selected_counts_by_ki.is_cuda(), "selected_counts_by_ki must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(final_indices.scalar_type() == torch::kLong, "final_indices must be int64");
  TORCH_CHECK(selected_counts_by_ki.scalar_type() == torch::kLong, "selected_counts_by_ki must be int64");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(final_indices.dim() == 2 && final_indices.size(1) == 2, "final_indices shape must be [heads, 2]");
  TORCH_CHECK(selected_counts_by_ki.dim() == 1, "selected_counts_by_ki shape must be [k_count]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v_count]");
  return joint_grouped_accounting_sums_cuda(
      final_indices.contiguous(),
      selected_counts_by_ki.contiguous(),
      v_budgets.contiguous(),
      context_len,
      head_dim,
      key_bytes,
      value_bytes,
      selector_mb,
      v_pq_codebook_mb,
      metadata_mb,
      actual_value_subvecs,
      code_bytes);
}

void joint_grouped_accounting_accumulate(
    torch::Tensor accumulator,
    torch::Tensor final_indices,
    torch::Tensor selected_counts_by_ki,
    torch::Tensor v_budgets,
    int64_t context_len,
    int64_t head_dim,
    int64_t key_bytes,
    int64_t value_bytes,
    double selector_mb,
    double v_pq_codebook_mb,
    double metadata_mb,
    int64_t actual_value_subvecs,
    int64_t code_bytes) {
  TORCH_CHECK(accumulator.is_cuda(), "accumulator must be CUDA");
  TORCH_CHECK(final_indices.is_cuda(), "final_indices must be CUDA");
  TORCH_CHECK(selected_counts_by_ki.is_cuda(), "selected_counts_by_ki must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(accumulator.scalar_type() == torch::kFloat64, "accumulator must be float64");
  TORCH_CHECK(final_indices.scalar_type() == torch::kLong, "final_indices must be int64");
  TORCH_CHECK(selected_counts_by_ki.scalar_type() == torch::kLong, "selected_counts_by_ki must be int64");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(accumulator.dim() == 1 && accumulator.size(0) == 11, "accumulator shape must be [11]");
  TORCH_CHECK(final_indices.dim() == 2 && final_indices.size(1) == 2, "final_indices shape must be [heads, 2]");
  TORCH_CHECK(selected_counts_by_ki.dim() == 1, "selected_counts_by_ki shape must be [k_count]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v_count]");
  joint_grouped_accounting_accumulate_cuda(
      accumulator,
      final_indices.contiguous(),
      selected_counts_by_ki.contiguous(),
      v_budgets.contiguous(),
      context_len,
      head_dim,
      key_bytes,
      value_bytes,
      selector_mb,
      v_pq_codebook_mb,
      metadata_mb,
      actual_value_subvecs,
      code_bytes);
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

std::vector<torch::Tensor> joint_select_policy_from_grouped_risk_no_mb(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    int64_t k_count,
    int64_t heads,
    double threshold,
    int64_t policy_id) {
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
  TORCH_CHECK(base_outputs.dim() == 2, "base_outputs shape must be [groups*k*heads, dim]");
  TORCH_CHECK(probs.dim() == 2, "probs shape must be [groups*k*heads, context]");
  TORCH_CHECK(residual_groups.dim() == 3, "residual_groups shape must be [groups, context, dim]");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v]");
  TORCH_CHECK(k_count > 0, "k_count must be positive");
  TORCH_CHECK(heads > 0, "heads must be positive");
  TORCH_CHECK(base_outputs.size(0) % (k_count * heads) == 0, "base_outputs row count mismatch");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs row count mismatch");
  const auto groups = base_outputs.size(0) / (k_count * heads);
  TORCH_CHECK(residual_groups.size(0) == groups, "residual_groups group count mismatch");
  TORCH_CHECK(code_error_groups.size(0) == groups, "code_error_groups group count mismatch");
  TORCH_CHECK(residual_groups.size(1) == probs.size(1), "residual_groups context mismatch");
  TORCH_CHECK(code_error_groups.size(1) == probs.size(1), "code_error_groups context mismatch");
  TORCH_CHECK(residual_groups.size(2) == base_outputs.size(1), "residual_groups dim mismatch");
  TORCH_CHECK(policy_id >= 0 && policy_id <= 3, "no-MB policy_id must be 0..3");
  return joint_select_policy_from_grouped_risk_no_mb_cuda(
      base_outputs.contiguous(),
      probs.contiguous(),
      residual_groups.contiguous(),
      code_error_groups.contiguous(),
      v_budgets.contiguous(),
      k_count,
      heads,
      threshold,
      policy_id);
}

std::vector<torch::Tensor> joint_select_policy_from_grouped_risk_batched_no_mb(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    double threshold,
    int64_t policy_id) {
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
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v]");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs group mismatch");
  TORCH_CHECK(probs.size(1) == base_outputs.size(1), "probs k-count mismatch");
  TORCH_CHECK(probs.size(2) == base_outputs.size(2), "probs head-count mismatch");
  TORCH_CHECK(residual_groups.size(0) == base_outputs.size(0), "residual_groups group mismatch");
  TORCH_CHECK(code_error_groups.size(0) == base_outputs.size(0), "code_error_groups group mismatch");
  TORCH_CHECK(residual_groups.size(1) == probs.size(3), "residual_groups context mismatch");
  TORCH_CHECK(code_error_groups.size(1) == probs.size(3), "code_error_groups context mismatch");
  TORCH_CHECK(residual_groups.size(2) == base_outputs.size(3), "residual_groups dim mismatch");
  TORCH_CHECK(policy_id >= 0 && policy_id <= 3, "no-MB policy_id must be 0..3");
  return joint_select_policy_from_grouped_risk_batched_no_mb_cuda(
      base_outputs.contiguous(),
      probs.contiguous(),
      residual_groups.contiguous(),
      code_error_groups.contiguous(),
      v_budgets.contiguous(),
      threshold,
      policy_id);
}

std::vector<torch::Tensor> joint_select_policy_from_grouped_risk_intervals_batched_no_mb(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    double threshold,
    int64_t policy_id) {
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
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v]");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs group mismatch");
  TORCH_CHECK(probs.size(1) == base_outputs.size(1), "probs k-count mismatch");
  TORCH_CHECK(probs.size(2) == base_outputs.size(2), "probs head-count mismatch");
  TORCH_CHECK(residual_groups.size(0) == base_outputs.size(0), "residual_groups group mismatch");
  TORCH_CHECK(code_error_groups.size(0) == base_outputs.size(0), "code_error_groups group mismatch");
  TORCH_CHECK(residual_groups.size(1) == probs.size(3), "residual_groups context mismatch");
  TORCH_CHECK(code_error_groups.size(1) == probs.size(3), "code_error_groups context mismatch");
  TORCH_CHECK(residual_groups.size(2) == base_outputs.size(3), "residual_groups dim mismatch");
  TORCH_CHECK(policy_id >= 0 && policy_id <= 3, "interval no-MB policy_id must be 0..3");
  return joint_select_policy_from_grouped_risk_intervals_batched_no_mb_cuda(
      base_outputs.contiguous(),
      probs.contiguous(),
      residual_groups.contiguous(),
      code_error_groups.contiguous(),
      v_budgets.contiguous(),
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

int64_t joint_rank_prefix_sort_temp_bytes(int64_t rows, int64_t count) {
  return joint_rank_prefix_sort_temp_bytes_cuda(rows, count);
}

torch::Tensor joint_rank_prefix_tokens_workspace(
    torch::Tensor scores,
    torch::Tensor indexed_tokens,
    int64_t max_take,
    torch::Tensor score_in,
    torch::Tensor score_out,
    torch::Tensor pos_in,
    torch::Tensor pos_out,
    torch::Tensor offsets,
    torch::Tensor temp,
    torch::Tensor out) {
  TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(score_in.is_cuda(), "score_in must be CUDA");
  TORCH_CHECK(score_out.is_cuda(), "score_out must be CUDA");
  TORCH_CHECK(pos_in.is_cuda(), "pos_in must be CUDA");
  TORCH_CHECK(pos_out.is_cuda(), "pos_out must be CUDA");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
  TORCH_CHECK(temp.is_cuda(), "temp must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32, "scores must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(score_in.scalar_type() == torch::kFloat32, "score_in must be float32");
  TORCH_CHECK(score_out.scalar_type() == torch::kFloat32, "score_out must be float32");
  TORCH_CHECK(pos_in.scalar_type() == torch::kInt, "pos_in must be int32");
  TORCH_CHECK(pos_out.scalar_type() == torch::kInt, "pos_out must be int32");
  TORCH_CHECK(offsets.scalar_type() == torch::kLong, "offsets must be int64");
  TORCH_CHECK(temp.scalar_type() == torch::kUInt8, "temp must be uint8");
  TORCH_CHECK(out.scalar_type() == torch::kLong, "out must be int64");
  TORCH_CHECK(scores.dim() == 2, "scores shape must be [heads, tokens]");
  TORCH_CHECK(indexed_tokens.dim() == 1, "indexed_tokens shape must be [tokens]");
  TORCH_CHECK(scores.size(1) == indexed_tokens.size(0), "scores/indexed_tokens token count mismatch");
  TORCH_CHECK(max_take >= 0, "max_take must be non-negative");
  return joint_rank_prefix_tokens_workspace_cuda(
      scores.contiguous(),
      indexed_tokens.contiguous(),
      max_take,
      score_in.contiguous(),
      score_out.contiguous(),
      pos_in.contiguous(),
      pos_out.contiguous(),
      offsets.contiguous(),
      temp.contiguous(),
      out.contiguous());
}

torch::Tensor joint_budget_prefix_tokens(
    torch::Tensor scores,
    torch::Tensor indexed_tokens,
    torch::Tensor take_counts,
    int64_t max_take) {
  TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(take_counts.is_cuda(), "take_counts must be CUDA");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32, "scores must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(take_counts.scalar_type() == torch::kLong, "take_counts must be int64");
  TORCH_CHECK(scores.dim() == 2, "scores shape must be [heads, tokens]");
  TORCH_CHECK(indexed_tokens.dim() == 1, "indexed_tokens shape must be [tokens]");
  TORCH_CHECK(take_counts.dim() == 1, "take_counts shape must be [budgets]");
  TORCH_CHECK(scores.size(1) == indexed_tokens.size(0), "scores/indexed_tokens token count mismatch");
  TORCH_CHECK(max_take >= 0, "max_take must be non-negative");
  return joint_budget_prefix_tokens_cuda(
      scores.contiguous(),
      indexed_tokens.contiguous(),
      take_counts.contiguous(),
      max_take);
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

torch::Tensor joint_mixed_score_grid_workspace(
    torch::Tensor output,
    torch::Tensor selected_mask,
    torch::Tensor fit_scale,
    torch::Tensor fit_bias,
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate) {
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(selected_mask.is_cuda(), "selected_mask must be CUDA");
  TORCH_CHECK(fit_scale.is_cuda(), "fit_scale must be CUDA");
  TORCH_CHECK(fit_bias.is_cuda(), "fit_bias must be CUDA");
  TORCH_CHECK(exact_scores.is_cuda(), "exact_scores must be CUDA");
  TORCH_CHECK(pq_logits.is_cuda(), "pq_logits must be CUDA");
  TORCH_CHECK(y_indexed.is_cuda(), "y_indexed must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(base_tokens.is_cuda(), "base_tokens must be CUDA");
  TORCH_CHECK(ranked_prefix_tokens.is_cuda(), "ranked_prefix_tokens must be CUDA");
  TORCH_CHECK(k_take_counts.is_cuda(), "k_take_counts must be CUDA");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
  TORCH_CHECK(selected_mask.scalar_type() == torch::kUInt8, "selected_mask must be uint8");
  TORCH_CHECK(fit_scale.scalar_type() == torch::kFloat32, "fit_scale must be float32");
  TORCH_CHECK(fit_bias.scalar_type() == torch::kFloat32, "fit_bias must be float32");
  TORCH_CHECK(exact_scores.scalar_type() == torch::kFloat32, "exact_scores must be float32");
  TORCH_CHECK(pq_logits.scalar_type() == torch::kFloat32, "pq_logits must be float32");
  TORCH_CHECK(y_indexed.scalar_type() == torch::kFloat32, "y_indexed must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(base_tokens.scalar_type() == torch::kLong, "base_tokens must be int64");
  TORCH_CHECK(ranked_prefix_tokens.scalar_type() == torch::kLong, "ranked_prefix_tokens must be int64");
  TORCH_CHECK(k_take_counts.scalar_type() == torch::kLong, "k_take_counts must be int64");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(selected_mask.is_contiguous(), "selected_mask must be contiguous");
  TORCH_CHECK(fit_scale.is_contiguous(), "fit_scale must be contiguous");
  TORCH_CHECK(fit_bias.is_contiguous(), "fit_bias must be contiguous");
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
  TORCH_CHECK(output.size(0) == k_take_counts.size(0), "output k dimension mismatch");
  TORCH_CHECK(output.size(1) == exact_scores.size(0), "output head dimension mismatch");
  TORCH_CHECK(output.size(2) == exact_scores.size(1), "output context dimension mismatch");
  TORCH_CHECK(selected_mask.sizes() == output.sizes(), "selected_mask must match output shape");
  TORCH_CHECK(fit_scale.size(0) == output.size(0) && fit_scale.size(1) == output.size(1), "fit_scale shape mismatch");
  TORCH_CHECK(fit_bias.sizes() == fit_scale.sizes(), "fit_bias shape mismatch");
  return joint_mixed_score_grid_workspace_cuda(
      output,
      selected_mask,
      fit_scale,
      fit_bias,
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      y_indexed.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      calibrate);
}

torch::Tensor joint_mixed_score_grid_rankpos_nocalib_workspace(
    torch::Tensor output,
    torch::Tensor token_to_indexed,
    torch::Tensor base_mask,
    torch::Tensor rank_pos,
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    double pq_scale) {
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(token_to_indexed.is_cuda(), "token_to_indexed must be CUDA");
  TORCH_CHECK(base_mask.is_cuda(), "base_mask must be CUDA");
  TORCH_CHECK(rank_pos.is_cuda(), "rank_pos must be CUDA");
  TORCH_CHECK(exact_scores.is_cuda(), "exact_scores must be CUDA");
  TORCH_CHECK(pq_logits.is_cuda(), "pq_logits must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(base_tokens.is_cuda(), "base_tokens must be CUDA");
  TORCH_CHECK(ranked_prefix_tokens.is_cuda(), "ranked_prefix_tokens must be CUDA");
  TORCH_CHECK(k_take_counts.is_cuda(), "k_take_counts must be CUDA");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
  TORCH_CHECK(token_to_indexed.scalar_type() == torch::kInt, "token_to_indexed must be int32");
  TORCH_CHECK(base_mask.scalar_type() == torch::kUInt8, "base_mask must be uint8");
  TORCH_CHECK(rank_pos.scalar_type() == torch::kInt, "rank_pos must be int32");
  TORCH_CHECK(exact_scores.scalar_type() == torch::kFloat32, "exact_scores must be float32");
  TORCH_CHECK(pq_logits.scalar_type() == torch::kFloat32, "pq_logits must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(base_tokens.scalar_type() == torch::kLong, "base_tokens must be int64");
  TORCH_CHECK(ranked_prefix_tokens.scalar_type() == torch::kLong, "ranked_prefix_tokens must be int64");
  TORCH_CHECK(k_take_counts.scalar_type() == torch::kLong, "k_take_counts must be int64");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(token_to_indexed.is_contiguous(), "token_to_indexed must be contiguous");
  TORCH_CHECK(base_mask.is_contiguous(), "base_mask must be contiguous");
  TORCH_CHECK(rank_pos.is_contiguous(), "rank_pos must be contiguous");
  TORCH_CHECK(exact_scores.dim() == 2, "exact_scores shape must be [heads, context]");
  TORCH_CHECK(pq_logits.dim() == 2, "pq_logits shape must be [heads, indexed]");
  TORCH_CHECK(indexed_tokens.dim() == 1, "indexed_tokens shape must be [indexed]");
  TORCH_CHECK(base_tokens.dim() == 1, "base_tokens shape must be [base]");
  TORCH_CHECK(ranked_prefix_tokens.dim() == 2, "ranked_prefix_tokens shape must be [heads, max_rank]");
  TORCH_CHECK(k_take_counts.dim() == 1, "k_take_counts shape must be [k]");
  TORCH_CHECK(output.dim() == 3, "output shape must be [k, heads, context]");
  TORCH_CHECK(token_to_indexed.dim() == 1, "token_to_indexed shape must be [context]");
  TORCH_CHECK(base_mask.dim() == 1, "base_mask shape must be [context]");
  TORCH_CHECK(rank_pos.dim() == 2, "rank_pos shape must be [heads, context]");
  TORCH_CHECK(pq_logits.size(0) == exact_scores.size(0), "pq/exact head count mismatch");
  TORCH_CHECK(indexed_tokens.size(0) == pq_logits.size(1), "indexed token count mismatch");
  TORCH_CHECK(ranked_prefix_tokens.size(0) == exact_scores.size(0), "ranked/exact head count mismatch");
  TORCH_CHECK(output.size(0) == k_take_counts.size(0), "output k dimension mismatch");
  TORCH_CHECK(output.size(1) == exact_scores.size(0), "output head dimension mismatch");
  TORCH_CHECK(output.size(2) == exact_scores.size(1), "output context dimension mismatch");
  TORCH_CHECK(token_to_indexed.size(0) == exact_scores.size(1), "token_to_indexed context mismatch");
  TORCH_CHECK(base_mask.size(0) == exact_scores.size(1), "base_mask context mismatch");
  TORCH_CHECK(rank_pos.size(0) == exact_scores.size(0), "rank_pos head mismatch");
  TORCH_CHECK(rank_pos.size(1) == exact_scores.size(1), "rank_pos context mismatch");
  return joint_mixed_score_grid_rankpos_nocalib_workspace_cuda(
      output,
      token_to_indexed,
      base_mask,
      rank_pos,
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      pq_scale);
}

torch::Tensor joint_mixed_score_grid_nocalib_scatter_workspace(
    torch::Tensor output,
    torch::Tensor token_to_indexed,
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    double pq_scale) {
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(token_to_indexed.is_cuda(), "token_to_indexed must be CUDA");
  TORCH_CHECK(exact_scores.is_cuda(), "exact_scores must be CUDA");
  TORCH_CHECK(pq_logits.is_cuda(), "pq_logits must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(base_tokens.is_cuda(), "base_tokens must be CUDA");
  TORCH_CHECK(ranked_prefix_tokens.is_cuda(), "ranked_prefix_tokens must be CUDA");
  TORCH_CHECK(k_take_counts.is_cuda(), "k_take_counts must be CUDA");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
  TORCH_CHECK(token_to_indexed.scalar_type() == torch::kInt, "token_to_indexed must be int32");
  TORCH_CHECK(exact_scores.scalar_type() == torch::kFloat32, "exact_scores must be float32");
  TORCH_CHECK(pq_logits.scalar_type() == torch::kFloat32, "pq_logits must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(base_tokens.scalar_type() == torch::kLong, "base_tokens must be int64");
  TORCH_CHECK(ranked_prefix_tokens.scalar_type() == torch::kLong, "ranked_prefix_tokens must be int64");
  TORCH_CHECK(k_take_counts.scalar_type() == torch::kLong, "k_take_counts must be int64");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(token_to_indexed.is_contiguous(), "token_to_indexed must be contiguous");
  TORCH_CHECK(exact_scores.dim() == 2, "exact_scores shape must be [heads, context]");
  TORCH_CHECK(pq_logits.dim() == 2, "pq_logits shape must be [heads, indexed]");
  TORCH_CHECK(indexed_tokens.dim() == 1, "indexed_tokens shape must be [indexed]");
  TORCH_CHECK(base_tokens.dim() == 1, "base_tokens shape must be [base]");
  TORCH_CHECK(ranked_prefix_tokens.dim() == 2, "ranked_prefix_tokens shape must be [heads, max_rank]");
  TORCH_CHECK(k_take_counts.dim() == 1, "k_take_counts shape must be [k]");
  TORCH_CHECK(output.dim() == 3, "output shape must be [k, heads, context]");
  TORCH_CHECK(token_to_indexed.dim() == 1, "token_to_indexed shape must be [context]");
  TORCH_CHECK(pq_logits.size(0) == exact_scores.size(0), "pq/exact head count mismatch");
  TORCH_CHECK(indexed_tokens.size(0) == pq_logits.size(1), "indexed token count mismatch");
  TORCH_CHECK(ranked_prefix_tokens.size(0) == exact_scores.size(0), "ranked/exact head count mismatch");
  TORCH_CHECK(output.size(0) == k_take_counts.size(0), "output k dimension mismatch");
  TORCH_CHECK(output.size(1) == exact_scores.size(0), "output head dimension mismatch");
  TORCH_CHECK(output.size(2) == exact_scores.size(1), "output context dimension mismatch");
  TORCH_CHECK(token_to_indexed.size(0) == exact_scores.size(1), "token_to_indexed context mismatch");
  return joint_mixed_score_grid_nocalib_scatter_workspace_cuda(
      output,
      token_to_indexed,
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      pq_scale);
}

torch::Tensor joint_mixed_score_grid_scaled(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate,
    double pq_scale) {
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
  return joint_mixed_score_grid_scaled_cuda(
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

torch::Tensor joint_mixed_score_grid_no_exact_fill_scaled(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate,
    double pq_scale) {
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
  return joint_mixed_score_grid_no_exact_fill_scaled_cuda(
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

static void check_joint_mixed_score_grid_common(
