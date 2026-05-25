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

int64_t joint_grouped_risk_sort_temp_bytes(int64_t rows, int64_t context_len) {
  TORCH_CHECK(rows >= 0, "rows must be nonnegative");
  TORCH_CHECK(context_len >= 0, "context_len must be nonnegative");
  return joint_grouped_risk_sort_temp_bytes_cuda(rows, context_len);
}

torch::Tensor joint_vprefix_outputs_from_grouped_risk_batched_workspace(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    torch::Tensor risk_in,
    torch::Tensor risk_out,
    torch::Tensor ids_in,
    torch::Tensor ids_out,
    torch::Tensor offsets,
    torch::Tensor temp,
    torch::Tensor interval_sums,
    torch::Tensor outputs) {
  TORCH_CHECK(base_outputs.is_cuda(), "base_outputs must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(residual_groups.is_cuda(), "residual_groups must be CUDA");
  TORCH_CHECK(code_error_groups.is_cuda(), "code_error_groups must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(risk_in.is_cuda(), "risk_in must be CUDA");
  TORCH_CHECK(risk_out.is_cuda(), "risk_out must be CUDA");
  TORCH_CHECK(ids_in.is_cuda(), "ids_in must be CUDA");
  TORCH_CHECK(ids_out.is_cuda(), "ids_out must be CUDA");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
  TORCH_CHECK(temp.is_cuda(), "temp must be CUDA");
  TORCH_CHECK(interval_sums.is_cuda(), "interval_sums must be CUDA");
  TORCH_CHECK(outputs.is_cuda(), "outputs must be CUDA");
  TORCH_CHECK(base_outputs.scalar_type() == torch::kFloat32, "base_outputs must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(residual_groups.scalar_type() == torch::kFloat32, "residual_groups must be float32");
  TORCH_CHECK(code_error_groups.scalar_type() == torch::kFloat32, "code_error_groups must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(risk_in.scalar_type() == torch::kFloat32, "risk_in must be float32");
  TORCH_CHECK(risk_out.scalar_type() == torch::kFloat32, "risk_out must be float32");
  TORCH_CHECK(ids_in.scalar_type() == torch::kInt, "ids_in must be int32");
  TORCH_CHECK(ids_out.scalar_type() == torch::kInt, "ids_out must be int32");
  TORCH_CHECK(offsets.scalar_type() == torch::kLong, "offsets must be int64");
  TORCH_CHECK(temp.scalar_type() == torch::kUInt8, "temp must be uint8");
  TORCH_CHECK(interval_sums.scalar_type() == torch::kFloat32, "interval_sums must be float32");
  TORCH_CHECK(outputs.scalar_type() == torch::kFloat32, "outputs must be float32");
  TORCH_CHECK(base_outputs.dim() == 4, "base_outputs shape must be [groups, k, heads, dim]");
  TORCH_CHECK(probs.dim() == 4, "probs shape must be [groups, k, heads, context]");
  TORCH_CHECK(residual_groups.dim() == 3, "residual_groups shape must be [groups, context, dim]");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [steps]");
  TORCH_CHECK(base_outputs.is_contiguous(), "base_outputs must be contiguous");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(residual_groups.is_contiguous(), "residual_groups must be contiguous");
  TORCH_CHECK(code_error_groups.is_contiguous(), "code_error_groups must be contiguous");
  TORCH_CHECK(v_budgets.is_contiguous(), "v_budgets must be contiguous");
  TORCH_CHECK(risk_in.is_contiguous(), "risk_in must be contiguous");
  TORCH_CHECK(risk_out.is_contiguous(), "risk_out must be contiguous");
  TORCH_CHECK(ids_in.is_contiguous(), "ids_in must be contiguous");
  TORCH_CHECK(ids_out.is_contiguous(), "ids_out must be contiguous");
  TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous");
  TORCH_CHECK(temp.is_contiguous(), "temp must be contiguous");
  TORCH_CHECK(interval_sums.is_contiguous(), "interval_sums must be contiguous");
  TORCH_CHECK(outputs.is_contiguous(), "outputs must be contiguous");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs/base group count mismatch");
  TORCH_CHECK(probs.size(1) == base_outputs.size(1), "probs/base k count mismatch");
  TORCH_CHECK(probs.size(2) == base_outputs.size(2), "probs/base head count mismatch");
  TORCH_CHECK(residual_groups.size(0) == base_outputs.size(0), "residual/base group count mismatch");
  TORCH_CHECK(code_error_groups.size(0) == base_outputs.size(0), "code_error/base group count mismatch");
  TORCH_CHECK(residual_groups.size(1) == probs.size(3), "residual/probs context mismatch");
  TORCH_CHECK(code_error_groups.size(1) == probs.size(3), "code_error/probs context mismatch");
  TORCH_CHECK(residual_groups.size(2) == base_outputs.size(3), "residual/base dim mismatch");
  return joint_vprefix_outputs_from_grouped_risk_batched_workspace_cuda(
      base_outputs,
      probs,
      residual_groups,
      code_error_groups,
      v_budgets,
      risk_in,
      risk_out,
      ids_in,
      ids_out,
      offsets,
      temp,
      interval_sums,
	      outputs);
}

torch::Tensor joint_vprefix_outputs_from_grouped_risk_batched_strided_workspace(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    int64_t context_len,
    torch::Tensor risk_in,
    torch::Tensor risk_out,
    torch::Tensor ids_in,
    torch::Tensor ids_out,
    torch::Tensor offsets,
    torch::Tensor temp,
    torch::Tensor interval_sums,
    torch::Tensor outputs) {
  TORCH_CHECK(base_outputs.is_cuda(), "base_outputs must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(residual_groups.is_cuda(), "residual_groups must be CUDA");
  TORCH_CHECK(code_error_groups.is_cuda(), "code_error_groups must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(risk_in.is_cuda(), "risk_in must be CUDA");
  TORCH_CHECK(risk_out.is_cuda(), "risk_out must be CUDA");
  TORCH_CHECK(ids_in.is_cuda(), "ids_in must be CUDA");
  TORCH_CHECK(ids_out.is_cuda(), "ids_out must be CUDA");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
  TORCH_CHECK(temp.is_cuda(), "temp must be CUDA");
  TORCH_CHECK(interval_sums.is_cuda(), "interval_sums must be CUDA");
  TORCH_CHECK(outputs.is_cuda(), "outputs must be CUDA");
  TORCH_CHECK(base_outputs.scalar_type() == torch::kFloat32, "base_outputs must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(residual_groups.scalar_type() == torch::kFloat32, "residual_groups must be float32");
  TORCH_CHECK(code_error_groups.scalar_type() == torch::kFloat32, "code_error_groups must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(risk_in.scalar_type() == torch::kFloat32, "risk_in must be float32");
  TORCH_CHECK(risk_out.scalar_type() == torch::kFloat32, "risk_out must be float32");
  TORCH_CHECK(ids_in.scalar_type() == torch::kInt, "ids_in must be int32");
  TORCH_CHECK(ids_out.scalar_type() == torch::kInt, "ids_out must be int32");
  TORCH_CHECK(offsets.scalar_type() == torch::kLong, "offsets must be int64");
  TORCH_CHECK(temp.scalar_type() == torch::kUInt8, "temp must be uint8");
  TORCH_CHECK(interval_sums.scalar_type() == torch::kFloat32, "interval_sums must be float32");
  TORCH_CHECK(outputs.scalar_type() == torch::kFloat32, "outputs must be float32");
  TORCH_CHECK(base_outputs.dim() == 4, "base_outputs shape must be [groups, k, heads, dim]");
  TORCH_CHECK(probs.dim() == 4, "probs shape must be [groups, k, heads, capacity]");
  TORCH_CHECK(residual_groups.dim() == 3, "residual_groups shape must be [groups, context, dim]");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [steps]");
  TORCH_CHECK(base_outputs.is_contiguous(), "base_outputs must be contiguous");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(residual_groups.is_contiguous(), "residual_groups must be contiguous");
  TORCH_CHECK(code_error_groups.is_contiguous(), "code_error_groups must be contiguous");
  TORCH_CHECK(v_budgets.is_contiguous(), "v_budgets must be contiguous");
  TORCH_CHECK(risk_in.is_contiguous(), "risk_in must be contiguous");
  TORCH_CHECK(risk_out.is_contiguous(), "risk_out must be contiguous");
  TORCH_CHECK(ids_in.is_contiguous(), "ids_in must be contiguous");
  TORCH_CHECK(ids_out.is_contiguous(), "ids_out must be contiguous");
  TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous");
  TORCH_CHECK(temp.is_contiguous(), "temp must be contiguous");
  TORCH_CHECK(interval_sums.is_contiguous(), "interval_sums must be contiguous");
  TORCH_CHECK(outputs.is_contiguous(), "outputs must be contiguous");
  TORCH_CHECK(context_len >= 0, "context_len must be nonnegative");
  TORCH_CHECK(probs.size(0) == base_outputs.size(0), "probs/base group count mismatch");
  TORCH_CHECK(probs.size(1) == base_outputs.size(1), "probs/base k count mismatch");
  TORCH_CHECK(probs.size(2) == base_outputs.size(2), "probs/base head count mismatch");
  TORCH_CHECK(probs.size(3) >= context_len, "probs capacity must cover context_len");
  TORCH_CHECK(residual_groups.size(0) == base_outputs.size(0), "residual/base group count mismatch");
  TORCH_CHECK(code_error_groups.size(0) == base_outputs.size(0), "code_error/base group count mismatch");
  TORCH_CHECK(residual_groups.size(1) >= context_len, "residual/context mismatch");
  TORCH_CHECK(code_error_groups.size(1) >= context_len, "code_error/context mismatch");
  TORCH_CHECK(residual_groups.size(2) == base_outputs.size(3), "residual/base dim mismatch");
  return joint_vprefix_outputs_from_grouped_risk_batched_strided_workspace_cuda(
      base_outputs,
      probs,
      residual_groups,
      code_error_groups,
      v_budgets,
      context_len,
      risk_in,
      risk_out,
      ids_in,
      ids_out,
      offsets,
      temp,
      interval_sums,
      outputs);
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

torch::Tensor joint_vprefix_outputs_from_grouped_scores_batched(
    torch::Tensor score_grid,
    torch::Tensor vhat_groups,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(vhat_groups.is_cuda(), "vhat_groups must be CUDA");
  TORCH_CHECK(residual_groups.is_cuda(), "residual_groups must be CUDA");
  TORCH_CHECK(code_error_groups.is_cuda(), "code_error_groups must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(vhat_groups.scalar_type() == torch::kFloat32, "vhat_groups must be float32");
  TORCH_CHECK(residual_groups.scalar_type() == torch::kFloat32, "residual_groups must be float32");
  TORCH_CHECK(code_error_groups.scalar_type() == torch::kFloat32, "code_error_groups must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(score_grid.dim() == 4, "score_grid shape must be [groups, k_count, heads, context_len]");
  TORCH_CHECK(vhat_groups.dim() == 3, "vhat_groups shape must be [groups, context_len, dim]");
  TORCH_CHECK(residual_groups.sizes() == vhat_groups.sizes(), "residual_groups must match vhat_groups shape");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context_len]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v_steps]");
  TORCH_CHECK(vhat_groups.size(0) == score_grid.size(0), "vhat_groups group count mismatch");
  TORCH_CHECK(vhat_groups.size(1) == score_grid.size(3), "vhat_groups context mismatch");
  TORCH_CHECK(code_error_groups.size(0) == score_grid.size(0), "code_error_groups group count mismatch");
  TORCH_CHECK(code_error_groups.size(1) == score_grid.size(3), "code_error_groups context mismatch");
  return joint_vprefix_outputs_from_grouped_scores_batched_cuda(
      score_grid.contiguous(),
      vhat_groups.contiguous(),
      residual_groups.contiguous(),
      code_error_groups.contiguous(),
      v_budgets.contiguous());
}

torch::Tensor joint_vprefix_outputs_from_grouped_scores_batched_workspace(
    torch::Tensor score_grid,
    torch::Tensor vhat_groups,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    torch::Tensor row_max,
    torch::Tensor row_denom,
    torch::Tensor base_flat,
    torch::Tensor risk_in,
    torch::Tensor risk_out,
    torch::Tensor ids_in,
    torch::Tensor ids_out,
    torch::Tensor offsets,
    torch::Tensor temp,
    torch::Tensor interval_sums,
    torch::Tensor outputs) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(vhat_groups.is_cuda(), "vhat_groups must be CUDA");
  TORCH_CHECK(residual_groups.is_cuda(), "residual_groups must be CUDA");
  TORCH_CHECK(code_error_groups.is_cuda(), "code_error_groups must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(row_max.is_cuda(), "row_max must be CUDA");
  TORCH_CHECK(row_denom.is_cuda(), "row_denom must be CUDA");
  TORCH_CHECK(base_flat.is_cuda(), "base_flat must be CUDA");
  TORCH_CHECK(risk_in.is_cuda(), "risk_in must be CUDA");
  TORCH_CHECK(risk_out.is_cuda(), "risk_out must be CUDA");
  TORCH_CHECK(ids_in.is_cuda(), "ids_in must be CUDA");
  TORCH_CHECK(ids_out.is_cuda(), "ids_out must be CUDA");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
  TORCH_CHECK(temp.is_cuda(), "temp must be CUDA");
  TORCH_CHECK(interval_sums.is_cuda(), "interval_sums must be CUDA");
  TORCH_CHECK(outputs.is_cuda(), "outputs must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(vhat_groups.scalar_type() == torch::kFloat32, "vhat_groups must be float32");
  TORCH_CHECK(residual_groups.scalar_type() == torch::kFloat32, "residual_groups must be float32");
  TORCH_CHECK(code_error_groups.scalar_type() == torch::kFloat32, "code_error_groups must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(row_max.scalar_type() == torch::kFloat32, "row_max must be float32");
  TORCH_CHECK(row_denom.scalar_type() == torch::kFloat32, "row_denom must be float32");
  TORCH_CHECK(base_flat.scalar_type() == torch::kFloat32, "base_flat must be float32");
  TORCH_CHECK(risk_in.scalar_type() == torch::kFloat32, "risk_in must be float32");
  TORCH_CHECK(risk_out.scalar_type() == torch::kFloat32, "risk_out must be float32");
  TORCH_CHECK(ids_in.scalar_type() == torch::kInt, "ids_in must be int32");
  TORCH_CHECK(ids_out.scalar_type() == torch::kInt, "ids_out must be int32");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(temp.scalar_type() == torch::kUInt8, "temp must be uint8");
  TORCH_CHECK(interval_sums.scalar_type() == torch::kFloat32, "interval_sums must be float32");
  TORCH_CHECK(outputs.scalar_type() == torch::kFloat32, "outputs must be float32");
  TORCH_CHECK(score_grid.dim() == 4, "score_grid shape must be [groups, k_count, heads, context_len]");
  TORCH_CHECK(vhat_groups.dim() == 3, "vhat_groups shape must be [groups, context_len, dim]");
  TORCH_CHECK(residual_groups.sizes() == vhat_groups.sizes(), "residual_groups must match vhat_groups shape");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context_len]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v_steps]");
  TORCH_CHECK(vhat_groups.size(0) == score_grid.size(0), "vhat_groups group count mismatch");
  TORCH_CHECK(vhat_groups.size(1) == score_grid.size(3), "vhat_groups context mismatch");
  TORCH_CHECK(code_error_groups.size(0) == score_grid.size(0), "code_error_groups group count mismatch");
  TORCH_CHECK(code_error_groups.size(1) == score_grid.size(3), "code_error_groups context mismatch");
  return joint_vprefix_outputs_from_grouped_scores_batched_workspace_cuda(
      score_grid.contiguous(),
      vhat_groups.contiguous(),
      residual_groups.contiguous(),
      code_error_groups.contiguous(),
      v_budgets.contiguous(),
      row_max.contiguous(),
      row_denom.contiguous(),
      base_flat.contiguous(),
      risk_in.contiguous(),
      risk_out.contiguous(),
      ids_in.contiguous(),
      ids_out.contiguous(),
      offsets.contiguous(),
      temp.contiguous(),
      interval_sums.contiguous(),
      outputs.contiguous());
}

std::vector<torch::Tensor> joint_select_policy_from_grouped_scores_intervals_batched_no_mb(
    torch::Tensor score_grid,
    torch::Tensor vhat_groups,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    double threshold,
    int64_t policy_id) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(vhat_groups.is_cuda(), "vhat_groups must be CUDA");
  TORCH_CHECK(residual_groups.is_cuda(), "residual_groups must be CUDA");
  TORCH_CHECK(code_error_groups.is_cuda(), "code_error_groups must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(vhat_groups.scalar_type() == torch::kFloat32, "vhat_groups must be float32");
  TORCH_CHECK(residual_groups.scalar_type() == torch::kFloat32, "residual_groups must be float32");
  TORCH_CHECK(code_error_groups.scalar_type() == torch::kFloat32, "code_error_groups must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(score_grid.dim() == 4, "score_grid shape must be [groups, k_count, heads, context_len]");
  TORCH_CHECK(vhat_groups.dim() == 3, "vhat_groups shape must be [groups, context_len, dim]");
  TORCH_CHECK(residual_groups.sizes() == vhat_groups.sizes(), "residual_groups must match vhat_groups shape");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context_len]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v_steps]");
  TORCH_CHECK(vhat_groups.size(0) == score_grid.size(0), "vhat_groups group count mismatch");
  TORCH_CHECK(vhat_groups.size(1) == score_grid.size(3), "vhat_groups context mismatch");
  TORCH_CHECK(code_error_groups.size(0) == score_grid.size(0), "code_error_groups group count mismatch");
  TORCH_CHECK(code_error_groups.size(1) == score_grid.size(3), "code_error_groups context mismatch");
  return joint_select_policy_from_grouped_scores_intervals_batched_no_mb_cuda(
      score_grid.contiguous(),
      vhat_groups.contiguous(),
      residual_groups.contiguous(),
      code_error_groups.contiguous(),
      v_budgets.contiguous(),
      threshold,
      policy_id);
}

std::vector<torch::Tensor> joint_mixed_select_policy_intervals_no_mb(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    torch::Tensor vhat,
    torch::Tensor residual,
    torch::Tensor code_error,
    torch::Tensor v_budgets,
    bool calibrate,
    double pq_scale,
    double threshold,
    int64_t policy_id) {
  TORCH_CHECK(policy_id >= 0 && policy_id <= 3, "mixed interval no-MB policy only supports non-MB policies");
  TORCH_CHECK(exact_scores.is_cuda(), "exact_scores must be CUDA");
  TORCH_CHECK(pq_logits.is_cuda(), "pq_logits must be CUDA");
  TORCH_CHECK(y_indexed.is_cuda(), "y_indexed must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(base_tokens.is_cuda(), "base_tokens must be CUDA");
  TORCH_CHECK(ranked_prefix_tokens.is_cuda(), "ranked_prefix_tokens must be CUDA");
  TORCH_CHECK(k_take_counts.is_cuda(), "k_take_counts must be CUDA");
  TORCH_CHECK(vhat.is_cuda(), "vhat must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(code_error.is_cuda(), "code_error must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(exact_scores.scalar_type() == torch::kFloat32, "exact_scores must be float32");
  TORCH_CHECK(pq_logits.scalar_type() == torch::kFloat32, "pq_logits must be float32");
  TORCH_CHECK(y_indexed.scalar_type() == torch::kFloat32, "y_indexed must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(base_tokens.scalar_type() == torch::kLong, "base_tokens must be int64");
  TORCH_CHECK(ranked_prefix_tokens.scalar_type() == torch::kLong, "ranked_prefix_tokens must be int64");
  TORCH_CHECK(k_take_counts.scalar_type() == torch::kLong, "k_take_counts must be int64");
  TORCH_CHECK(vhat.scalar_type() == torch::kFloat32, "vhat must be float32");
  TORCH_CHECK(residual.scalar_type() == torch::kFloat32, "residual must be float32");
  TORCH_CHECK(code_error.scalar_type() == torch::kFloat32, "code_error must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(exact_scores.dim() == 2, "exact_scores shape must be [heads, context_len]");
  TORCH_CHECK(pq_logits.dim() == 2, "pq_logits shape must be [heads, indexed_count]");
  TORCH_CHECK(y_indexed.dim() == 2, "y_indexed shape must be [heads, indexed_count]");
  TORCH_CHECK(indexed_tokens.dim() == 1, "indexed_tokens shape must be [indexed_count]");
  TORCH_CHECK(base_tokens.dim() == 1, "base_tokens shape must be [base_count]");
  TORCH_CHECK(ranked_prefix_tokens.dim() == 2, "ranked_prefix_tokens shape must be [heads, max_rank_take]");
  TORCH_CHECK(k_take_counts.dim() == 1, "k_take_counts shape must be [k_count]");
  TORCH_CHECK(vhat.dim() == 2, "vhat shape must be [context_len, dim]");
  TORCH_CHECK(residual.sizes() == vhat.sizes(), "residual must match vhat shape");
  TORCH_CHECK(code_error.dim() == 1, "code_error shape must be [context_len]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v_steps]");
  TORCH_CHECK(pq_logits.size(0) == exact_scores.size(0), "pq_logits head count mismatch");
  TORCH_CHECK(y_indexed.sizes() == pq_logits.sizes(), "y_indexed must match pq_logits shape");
  TORCH_CHECK(indexed_tokens.size(0) == pq_logits.size(1), "indexed_tokens count mismatch");
  TORCH_CHECK(ranked_prefix_tokens.size(0) == exact_scores.size(0), "ranked_prefix_tokens head count mismatch");
  TORCH_CHECK(vhat.size(0) == exact_scores.size(1), "vhat context mismatch");
  TORCH_CHECK(code_error.size(0) == exact_scores.size(1), "code_error context mismatch");
  return joint_mixed_select_policy_intervals_no_mb_cuda(
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      y_indexed.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      vhat.contiguous(),
      residual.contiguous(),
      code_error.contiguous(),
      v_budgets.contiguous(),
      calibrate,
      pq_scale,
      threshold,
      policy_id);
}

