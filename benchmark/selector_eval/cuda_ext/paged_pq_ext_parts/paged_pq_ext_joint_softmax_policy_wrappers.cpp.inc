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

std::vector<torch::Tensor> joint_softmax_base_outputs_cublas(
    torch::Tensor score_grid,
    torch::Tensor values) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(score_grid.dim() == 3, "score_grid shape must be [k_count, heads, context_len]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  TORCH_CHECK(values.size(0) >= score_grid.size(2), "values length must cover score_grid context_len");
  return joint_softmax_base_outputs_cublas_cuda(score_grid.contiguous(), values.contiguous());
}

std::vector<torch::Tensor> joint_softmax_base_outputs_workspace(
    torch::Tensor probs,
    torch::Tensor outputs,
    torch::Tensor score_grid,
    torch::Tensor values) {
  TORCH_CHECK(probs.is_cuda(), "probs workspace must be CUDA");
  TORCH_CHECK(outputs.is_cuda(), "outputs workspace must be CUDA");
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs workspace must be float32");
  TORCH_CHECK(outputs.scalar_type() == torch::kFloat32, "outputs workspace must be float32");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(probs.dim() == 3, "probs workspace shape must be [k_count, heads, context_len]");
  TORCH_CHECK(outputs.dim() == 3, "outputs workspace shape must be [k_count, heads, dim]");
  TORCH_CHECK(score_grid.dim() == 3, "score_grid shape must be [k_count, heads, context_len]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  TORCH_CHECK(probs.size(0) == score_grid.size(0), "probs/score k count mismatch");
  TORCH_CHECK(probs.size(1) == score_grid.size(1), "probs/score head count mismatch");
  TORCH_CHECK(probs.size(2) == score_grid.size(2), "probs/score context mismatch");
  TORCH_CHECK(outputs.size(0) == score_grid.size(0), "outputs/score k count mismatch");
  TORCH_CHECK(outputs.size(1) == score_grid.size(1), "outputs/score head count mismatch");
  TORCH_CHECK(outputs.size(2) == values.size(1), "outputs/value dim mismatch");
  TORCH_CHECK(values.size(0) >= score_grid.size(2), "values length must cover score_grid context_len");
  TORCH_CHECK(probs.is_contiguous(), "probs workspace must be contiguous");
  TORCH_CHECK(outputs.is_contiguous(), "outputs workspace must be contiguous");
  return joint_softmax_base_outputs_workspace_cuda(
      probs,
      outputs,
	      score_grid.contiguous(),
	      values.contiguous());
}

std::vector<torch::Tensor> joint_softmax_base_outputs_strided_workspace(
    torch::Tensor probs,
    torch::Tensor outputs,
    torch::Tensor score_grid,
    torch::Tensor values) {
  TORCH_CHECK(probs.is_cuda(), "probs workspace must be CUDA");
  TORCH_CHECK(outputs.is_cuda(), "outputs workspace must be CUDA");
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs workspace must be float32");
  TORCH_CHECK(outputs.scalar_type() == torch::kFloat32, "outputs workspace must be float32");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(probs.dim() == 3, "probs workspace shape must be [k_count, heads, capacity]");
  TORCH_CHECK(outputs.dim() == 3, "outputs workspace shape must be [k_count, heads, dim]");
  TORCH_CHECK(score_grid.dim() == 3, "score_grid shape must be [k_count, heads, context_len]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  TORCH_CHECK(probs.size(0) == score_grid.size(0), "probs/score k count mismatch");
  TORCH_CHECK(probs.size(1) == score_grid.size(1), "probs/score head count mismatch");
  TORCH_CHECK(probs.size(2) >= score_grid.size(2), "probs workspace capacity must cover score context");
  TORCH_CHECK(outputs.size(0) == score_grid.size(0), "outputs/score k count mismatch");
  TORCH_CHECK(outputs.size(1) == score_grid.size(1), "outputs/score head count mismatch");
  TORCH_CHECK(outputs.size(2) == values.size(1), "outputs/value dim mismatch");
  TORCH_CHECK(values.size(0) >= score_grid.size(2), "values length must cover score_grid context_len");
  TORCH_CHECK(probs.is_contiguous(), "probs workspace must be contiguous");
  TORCH_CHECK(outputs.is_contiguous(), "outputs workspace must be contiguous");
  return joint_softmax_base_outputs_strided_workspace_cuda(
      probs,
      outputs,
      score_grid.contiguous(),
      values.contiguous());
}

std::vector<torch::Tensor> joint_softmax_base_outputs_workspace_cublas(
    torch::Tensor probs,
    torch::Tensor outputs,
    torch::Tensor score_grid,
    torch::Tensor values) {
  TORCH_CHECK(probs.is_cuda(), "probs workspace must be CUDA");
  TORCH_CHECK(outputs.is_cuda(), "outputs workspace must be CUDA");
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs workspace must be float32");
  TORCH_CHECK(outputs.scalar_type() == torch::kFloat32, "outputs workspace must be float32");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(probs.dim() == 3, "probs workspace shape must be [k_count, heads, context_len]");
  TORCH_CHECK(outputs.dim() == 3, "outputs workspace shape must be [k_count, heads, dim]");
  TORCH_CHECK(score_grid.dim() == 3, "score_grid shape must be [k_count, heads, context_len]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  TORCH_CHECK(probs.size(0) == score_grid.size(0), "probs/score k count mismatch");
  TORCH_CHECK(probs.size(1) == score_grid.size(1), "probs/score head count mismatch");
  TORCH_CHECK(probs.size(2) == score_grid.size(2), "probs/score context mismatch");
  TORCH_CHECK(outputs.size(0) == score_grid.size(0), "outputs/score k count mismatch");
  TORCH_CHECK(outputs.size(1) == score_grid.size(1), "outputs/score head count mismatch");
  TORCH_CHECK(outputs.size(2) == values.size(1), "outputs/value dim mismatch");
  TORCH_CHECK(values.size(0) >= score_grid.size(2), "values length must cover score_grid context_len");
  TORCH_CHECK(probs.is_contiguous(), "probs workspace must be contiguous");
  TORCH_CHECK(outputs.is_contiguous(), "outputs workspace must be contiguous");
  return joint_softmax_base_outputs_workspace_cublas_cuda(
      probs,
      outputs,
      score_grid.contiguous(),
      values.contiguous());
}

std::vector<torch::Tensor> joint_softmax_base_outputs_grouped_cublas(
    torch::Tensor score_grid,
    torch::Tensor values_grouped) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values_grouped.is_cuda(), "values_grouped must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values_grouped.scalar_type() == torch::kFloat32, "values_grouped must be float32");
  TORCH_CHECK(score_grid.dim() == 4, "score_grid shape must be [groups, k_count, heads, context_len]");
  TORCH_CHECK(values_grouped.dim() == 3, "values_grouped shape must be [groups, context_len, dim]");
  TORCH_CHECK(values_grouped.size(0) == score_grid.size(0), "values_grouped group count mismatch");
  TORCH_CHECK(values_grouped.size(1) >= score_grid.size(3), "values_grouped length must cover score_grid context_len");
  return joint_softmax_base_outputs_grouped_cublas_cuda(score_grid.contiguous(), values_grouped.contiguous());
}

void joint_vpq_append_exact_suffix(
    torch::Tensor vhat,
    torch::Tensor residual,
    torch::Tensor code_error,
    torch::Tensor values,
    int64_t start,
    int64_t end) {
  TORCH_CHECK(vhat.is_cuda(), "vhat must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(code_error.is_cuda(), "code_error must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(vhat.is_contiguous(), "vhat must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(code_error.is_contiguous(), "code_error must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  joint_vpq_append_exact_suffix_cuda(vhat, residual, code_error, values, start, end);
}

std::vector<torch::Tensor> joint_vpq_sidecars_from_pack(
    torch::Tensor values,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t context_len) {
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  return joint_vpq_sidecars_from_pack_cuda(
      values.contiguous(),
      codebooks.contiguous(),
      codes.contiguous(),
      page_starts.contiguous(),
      context_len);
}

void joint_vpq_append_exact_suffix_grouped(
    torch::Tensor vhat,
    torch::Tensor residual,
    torch::Tensor code_error,
    torch::Tensor values,
    int64_t start,
    int64_t end) {
  TORCH_CHECK(vhat.is_cuda(), "vhat must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(code_error.is_cuda(), "code_error must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(vhat.is_contiguous(), "vhat must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(code_error.is_contiguous(), "code_error must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  joint_vpq_append_exact_suffix_grouped_cuda(vhat, residual, code_error, values, start, end);
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

std::vector<torch::Tensor> joint_mixed_softmax_base_outputs_tokenfit_scaled(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    torch::Tensor values,
    bool calibrate,
    double pq_scale) {
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
  return joint_mixed_softmax_base_outputs_tokenfit_scaled_cuda(
      exact_scores.contiguous(),
      pq_logits.contiguous(),
      y_indexed.contiguous(),
      indexed_tokens.contiguous(),
      base_tokens.contiguous(),
      ranked_prefix_tokens.contiguous(),
      k_take_counts.contiguous(),
      values.contiguous(),
      calibrate,
      pq_scale);
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

