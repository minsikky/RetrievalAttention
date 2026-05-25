torch::Tensor joint_vprefix_outputs_from_grouped_scores_batched_cuda(
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

  const auto groups = score_grid.size(0);
  const auto k_count = score_grid.size(1);
  const auto heads = score_grid.size(2);
  const auto context_len = score_grid.size(3);
  const auto dim = vhat_groups.size(2);
  const auto v_steps = v_budgets.size(0);
  const auto rows = groups * k_count * heads;
  TORCH_CHECK(vhat_groups.size(0) == groups, "vhat_groups group count mismatch");
  TORCH_CHECK(vhat_groups.size(1) == context_len, "vhat_groups context mismatch");
  TORCH_CHECK(code_error_groups.size(0) == groups, "code_error_groups group count mismatch");
  TORCH_CHECK(code_error_groups.size(1) == context_len, "code_error_groups context mismatch");

  auto outputs = torch::empty({rows, v_steps, dim}, score_grid.options().dtype(torch::kFloat32));
  if (rows == 0 || dim == 0 || v_steps == 0) {
    return outputs;
  }
  auto score_c = score_grid.contiguous();
  auto vhat_c = vhat_groups.contiguous();
  auto residual_c = residual_groups.contiguous();
  auto error_c = code_error_groups.contiguous();
  if (context_len == 0) {
    return outputs.zero_();
  }

  auto flat_opts = score_c.options().dtype(torch::kFloat32);
  auto long_opts = score_c.options().dtype(torch::kLong);
  auto int_opts = score_c.options().dtype(torch::kInt);
  auto byte_opts = score_c.options().dtype(torch::kUInt8);
  auto row_max = torch::empty({rows}, flat_opts);
  auto row_denom = torch::empty({rows}, flat_opts);
  auto base_flat = torch::empty({rows, dim}, flat_opts);
  auto risk_in = torch::empty({rows, context_len}, flat_opts);
  auto risk_out = torch::empty({rows, context_len}, flat_opts);
  auto ids_in = torch::empty({rows, context_len}, int_opts);
  auto ids_out = torch::empty({rows, context_len}, int_opts);
  auto offsets = torch::empty({rows + 1}, long_opts);
  auto interval_sums = torch::empty({rows, v_steps, dim}, flat_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  joint_softmax_row_stats_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      score_c.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  joint_base_outputs_from_grouped_scores_kernel<dims_per_block><<<
      static_cast<unsigned int>(rows * dim_tiles), threads, 0, stream>>>(
      score_c.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      vhat_c.data_ptr<float>(),
      base_flat.data_ptr<float>(),
      groups,
      k_count,
      heads,
      context_len,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t total = rows * context_len;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_grouped_flat_risk_sort_inputs_from_scores_kernel<<<
      static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      score_c.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      error_c.data_ptr<float>(),
      risk_in.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      groups,
      k_count,
      heads,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t offset_blocks = ((rows + 1) + threads - 1) / threads;
  joint_segment_offsets_kernel<<<static_cast<unsigned int>(offset_blocks), threads, 0, stream>>>(
      offsets.data_ptr<int64_t>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t temp_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      temp_bytes,
      risk_in.data_ptr<float>(),
      risk_out.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      ids_out.data_ptr<int32_t>(),
      total,
      rows,
      offsets.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  auto temp = torch::empty({static_cast<int64_t>(std::max<size_t>(temp_bytes, 1))}, byte_opts);
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temp.data_ptr(),
      temp_bytes,
      risk_in.data_ptr<float>(),
      risk_out.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      ids_out.data_ptr<int32_t>(),
      total,
      rows,
      offsets.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t interval_blocks = rows * v_steps * dim_tiles;
  joint_grouped_flat_vprefix_interval_sums_from_scores_kernel<int32_t, dims_per_block><<<
      static_cast<unsigned int>(interval_blocks), threads, 0, stream>>>(
      score_c.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      residual_c.data_ptr<float>(),
      ids_out.data_ptr<int32_t>(),
      v_budgets.data_ptr<int64_t>(),
      interval_sums.data_ptr<float>(),
      groups,
      k_count,
      heads,
      context_len,
      context_len,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t out_total = rows * dim;
  const int64_t out_blocks = (out_total + threads - 1) / threads;
  joint_grouped_flat_vprefix_prefix_intervals_kernel<<<static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
      base_flat.data_ptr<float>(),
      interval_sums.data_ptr<float>(),
      v_budgets.data_ptr<int64_t>(),
      outputs.data_ptr<float>(),
      rows,
      context_len,
      context_len,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor joint_vprefix_outputs_from_grouped_scores_batched_workspace_cuda(
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

  const auto groups = score_grid.size(0);
  const auto k_count = score_grid.size(1);
  const auto heads = score_grid.size(2);
  const auto context_len = score_grid.size(3);
  const auto dim = vhat_groups.size(2);
  const auto v_steps = v_budgets.size(0);
  const auto rows = groups * k_count * heads;
  TORCH_CHECK(vhat_groups.size(0) == groups, "vhat_groups group count mismatch");
  TORCH_CHECK(vhat_groups.size(1) == context_len, "vhat_groups context mismatch");
  TORCH_CHECK(code_error_groups.size(0) == groups, "code_error_groups group count mismatch");
  TORCH_CHECK(code_error_groups.size(1) == context_len, "code_error_groups context mismatch");
  TORCH_CHECK(row_max.numel() >= rows, "row_max workspace too small");
  TORCH_CHECK(row_denom.numel() >= rows, "row_denom workspace too small");
  TORCH_CHECK(base_flat.numel() >= rows * dim, "base_flat workspace too small");
  TORCH_CHECK(risk_in.numel() >= rows * context_len, "risk_in workspace too small");
  TORCH_CHECK(risk_out.numel() >= rows * context_len, "risk_out workspace too small");
  TORCH_CHECK(ids_in.numel() >= rows * context_len, "ids_in workspace too small");
  TORCH_CHECK(ids_out.numel() >= rows * context_len, "ids_out workspace too small");
  TORCH_CHECK(offsets.numel() >= rows + 1, "offsets workspace too small");
  TORCH_CHECK(interval_sums.numel() >= rows * v_steps * dim, "interval_sums workspace too small");
  TORCH_CHECK(outputs.numel() >= rows * v_steps * dim, "outputs workspace too small");

  auto outputs_view = outputs.reshape({rows, v_steps, dim});
  if (rows == 0 || dim == 0 || v_steps == 0) {
    return outputs_view;
  }
  auto score_c = score_grid.contiguous();
  auto vhat_c = vhat_groups.contiguous();
  auto residual_c = residual_groups.contiguous();
  auto error_c = code_error_groups.contiguous();
  if (context_len == 0) {
    return outputs_view.zero_();
  }

  auto row_max_view = row_max.reshape({-1}).slice(0, 0, rows);
  auto row_denom_view = row_denom.reshape({-1}).slice(0, 0, rows);
  auto base_view = base_flat.reshape({-1}).slice(0, 0, rows * dim).reshape({rows, dim});
  auto interval_view = interval_sums.reshape({-1}).slice(0, 0, rows * v_steps * dim).reshape({rows, v_steps, dim});

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  joint_softmax_row_stats_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      score_c.data_ptr<float>(),
      row_max_view.data_ptr<float>(),
      row_denom_view.data_ptr<float>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  joint_base_outputs_from_grouped_scores_kernel<dims_per_block><<<
      static_cast<unsigned int>(rows * dim_tiles), threads, 0, stream>>>(
      score_c.data_ptr<float>(),
      row_max_view.data_ptr<float>(),
      row_denom_view.data_ptr<float>(),
      vhat_c.data_ptr<float>(),
      base_view.data_ptr<float>(),
      groups,
      k_count,
      heads,
      context_len,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t total = rows * context_len;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_grouped_flat_risk_sort_inputs_from_scores_kernel<<<
      static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      score_c.data_ptr<float>(),
      row_max_view.data_ptr<float>(),
      row_denom_view.data_ptr<float>(),
      error_c.data_ptr<float>(),
      risk_in.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      groups,
      k_count,
      heads,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t offset_blocks = ((rows + 1) + threads - 1) / threads;
  joint_segment_offsets_kernel<<<static_cast<unsigned int>(offset_blocks), threads, 0, stream>>>(
      offsets.data_ptr<int64_t>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t temp_bytes = static_cast<size_t>(temp.numel());
  TORCH_CHECK(temp_bytes > 0, "temp workspace too small");
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temp.data_ptr(),
      temp_bytes,
      risk_in.data_ptr<float>(),
      risk_out.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      ids_out.data_ptr<int32_t>(),
      total,
      rows,
      offsets.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t interval_blocks = rows * v_steps * dim_tiles;
  joint_grouped_flat_vprefix_interval_sums_from_scores_kernel<int32_t, dims_per_block><<<
      static_cast<unsigned int>(interval_blocks), threads, 0, stream>>>(
      score_c.data_ptr<float>(),
      row_max_view.data_ptr<float>(),
      row_denom_view.data_ptr<float>(),
      residual_c.data_ptr<float>(),
      ids_out.data_ptr<int32_t>(),
      v_budgets.data_ptr<int64_t>(),
      interval_view.data_ptr<float>(),
      groups,
      k_count,
      heads,
      context_len,
      context_len,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t out_total = rows * dim;
  const int64_t out_blocks = (out_total + threads - 1) / threads;
  joint_grouped_flat_vprefix_prefix_intervals_kernel<<<static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
      base_view.data_ptr<float>(),
      interval_view.data_ptr<float>(),
      v_budgets.data_ptr<int64_t>(),
      outputs_view.data_ptr<float>(),
      rows,
      context_len,
      context_len,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs_view;
}

std::vector<torch::Tensor> joint_select_policy_from_grouped_scores_intervals_batched_no_mb_cuda(
    torch::Tensor score_grid,
    torch::Tensor vhat_groups,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    double threshold,
    int64_t policy_id) {
  TORCH_CHECK(policy_id >= 0 && policy_id <= 3, "score-interval no-MB policy only supports non-MB policies");
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

  const auto groups = score_grid.size(0);
  const auto k_count = score_grid.size(1);
  const auto heads = score_grid.size(2);
  const auto context_len = score_grid.size(3);
  const auto dim = vhat_groups.size(2);
  const auto v_steps = v_budgets.size(0);
  const auto rows = groups * k_count * heads;
  TORCH_CHECK(vhat_groups.size(0) == groups, "vhat_groups group count mismatch");
  TORCH_CHECK(vhat_groups.size(1) == context_len, "vhat_groups context mismatch");
  TORCH_CHECK(code_error_groups.size(0) == groups, "code_error_groups group count mismatch");
  TORCH_CHECK(code_error_groups.size(1) == context_len, "code_error_groups context mismatch");

  auto final_outputs = torch::empty({groups, heads, dim}, score_grid.options().dtype(torch::kFloat32));
  auto final_indices = torch::empty({groups, heads, 2}, score_grid.options().dtype(torch::kLong));
  if (groups == 0 || k_count == 0 || heads == 0 || v_steps == 0 || dim == 0) {
    return {final_outputs, final_indices};
  }
  auto score_c = score_grid.contiguous();
  auto vhat_c = vhat_groups.contiguous();
  auto residual_c = residual_groups.contiguous();
  auto error_c = code_error_groups.contiguous();
  if (context_len == 0) {
    final_outputs.zero_();
    final_indices.zero_();
    return {final_outputs, final_indices};
  }

  auto flat_opts = score_c.options().dtype(torch::kFloat32);
  auto long_opts = score_c.options().dtype(torch::kLong);
  auto int_opts = score_c.options().dtype(torch::kInt);
  auto byte_opts = score_c.options().dtype(torch::kUInt8);
  auto row_max = torch::empty({rows}, flat_opts);
  auto row_denom = torch::empty({rows}, flat_opts);
  auto base_flat = torch::empty({rows, dim}, flat_opts);
  auto risk_in = torch::empty({rows, context_len}, flat_opts);
  auto risk_out = torch::empty({rows, context_len}, flat_opts);
  auto ids_in = torch::empty({rows, context_len}, int_opts);
  auto ids_out = torch::empty({rows, context_len}, int_opts);
  auto offsets = torch::empty({rows + 1}, long_opts);
  auto interval_sums = torch::empty({rows, v_steps, dim}, flat_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  joint_softmax_row_stats_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      score_c.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  joint_base_outputs_from_grouped_scores_kernel<dims_per_block><<<
      static_cast<unsigned int>(rows * dim_tiles), threads, 0, stream>>>(
      score_c.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      vhat_c.data_ptr<float>(),
      base_flat.data_ptr<float>(),
      groups,
      k_count,
      heads,
      context_len,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t total = rows * context_len;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_grouped_flat_risk_sort_inputs_from_scores_kernel<<<
      static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      score_c.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      error_c.data_ptr<float>(),
      risk_in.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      groups,
      k_count,
      heads,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t offset_blocks = ((rows + 1) + threads - 1) / threads;
  joint_segment_offsets_kernel<<<static_cast<unsigned int>(offset_blocks), threads, 0, stream>>>(
      offsets.data_ptr<int64_t>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t temp_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      temp_bytes,
      risk_in.data_ptr<float>(),
      risk_out.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      ids_out.data_ptr<int32_t>(),
      total,
      rows,
      offsets.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  auto temp = torch::empty({static_cast<int64_t>(std::max<size_t>(temp_bytes, 1))}, byte_opts);
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temp.data_ptr(),
      temp_bytes,
      risk_in.data_ptr<float>(),
      risk_out.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      ids_out.data_ptr<int32_t>(),
      total,
      rows,
      offsets.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t interval_blocks = rows * v_steps * dim_tiles;
  joint_grouped_flat_vprefix_interval_sums_from_scores_kernel<int32_t, dims_per_block><<<
      static_cast<unsigned int>(interval_blocks), threads, 0, stream>>>(
      score_c.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      residual_c.data_ptr<float>(),
      ids_out.data_ptr<int32_t>(),
      v_budgets.data_ptr<int64_t>(),
      interval_sums.data_ptr<float>(),
      groups,
      k_count,
      heads,
      context_len,
      context_len,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const size_t shared_bytes = static_cast<size_t>(threads) * 4 * sizeof(double);
  joint_select_policy_grouped_intervals_no_mb_kernel<<<
      static_cast<unsigned int>(groups * heads), threads, shared_bytes, stream>>>(
      base_flat.data_ptr<float>(),
      interval_sums.data_ptr<float>(),
      final_indices.data_ptr<int64_t>(),
      final_outputs.data_ptr<float>(),
      groups,
      k_count,
      v_steps,
      heads,
      dim,
      threshold,
      static_cast<int>(policy_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {final_outputs, final_indices};
}

std::vector<torch::Tensor> joint_mixed_select_policy_intervals_no_mb_cuda(
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

  const auto heads = exact_scores.size(0);
  const auto context_len = exact_scores.size(1);
  const auto indexed_count = indexed_tokens.size(0);
  const auto k_count = k_take_counts.size(0);
  const auto max_rank_take = ranked_prefix_tokens.size(1);
  const auto dim = vhat.size(1);
  const auto v_steps = v_budgets.size(0);
  const auto rows = k_count * heads;
  TORCH_CHECK(pq_logits.size(0) == heads, "pq_logits head count mismatch");
  TORCH_CHECK(pq_logits.size(1) == indexed_count, "pq_logits indexed count mismatch");
  TORCH_CHECK(y_indexed.sizes() == pq_logits.sizes(), "y_indexed must match pq_logits shape");
  TORCH_CHECK(ranked_prefix_tokens.size(0) == heads, "ranked_prefix_tokens head count mismatch");
  TORCH_CHECK(vhat.size(0) == context_len, "vhat context mismatch");
  TORCH_CHECK(code_error.size(0) == context_len, "code_error context mismatch");

  auto final_outputs = torch::empty({heads, dim}, exact_scores.options().dtype(torch::kFloat32));
  auto final_indices = torch::empty({heads, 2}, exact_scores.options().dtype(torch::kLong));
  if (heads == 0 || k_count == 0 || v_steps == 0 || dim == 0) {
    return {final_outputs, final_indices};
  }
  if (context_len == 0) {
    final_outputs.zero_();
    final_indices.zero_();
    return {final_outputs, final_indices};
  }

  auto flat_opts = exact_scores.options().dtype(torch::kFloat32);
  auto long_opts = exact_scores.options().dtype(torch::kLong);
  auto int_opts = exact_scores.options().dtype(torch::kInt);
  auto byte_opts = exact_scores.options().dtype(torch::kUInt8);
  auto selected_mask = torch::empty({k_count, heads, context_len}, byte_opts);
  auto token_to_indexed = torch::empty({context_len}, int_opts);
  auto fit_scale = torch::empty({k_count, heads}, flat_opts);
  auto fit_bias = torch::empty({k_count, heads}, flat_opts);
  auto row_max = torch::empty({rows}, flat_opts);
  auto row_denom = torch::empty({rows}, flat_opts);
  auto probs = torch::empty({rows, context_len}, flat_opts);
  auto base_flat = torch::empty({rows, dim}, flat_opts);
  auto risk_in = torch::empty({rows, context_len}, flat_opts);
  auto risk_out = torch::empty({rows, context_len}, flat_opts);
  auto ids_in = torch::empty({rows, context_len}, int_opts);
  auto ids_out = torch::empty({rows, context_len}, int_opts);
  auto offsets = torch::empty({rows + 1}, long_opts);
  auto interval_sums = torch::empty({rows, v_steps, dim}, flat_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t mask_total = rows * context_len;
  const int64_t mask_blocks = (mask_total + threads - 1) / threads;
  joint_fill_byte_kernel<<<static_cast<unsigned int>(mask_blocks), threads, 0, stream>>>(
      selected_mask.data_ptr<unsigned char>(),
      mask_total,
      static_cast<unsigned char>(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t token_map_blocks = (context_len + threads - 1) / threads;
  joint_fill_int_kernel<<<static_cast<unsigned int>(token_map_blocks), threads, 0, stream>>>(
      token_to_indexed.data_ptr<int>(),
      context_len,
      -1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (indexed_count > 0) {
    const int64_t blocks = (indexed_count + threads - 1) / threads;
    joint_indexed_token_to_pos_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        token_to_indexed.data_ptr<int>(),
        indexed_tokens.data_ptr<int64_t>(),
        indexed_count,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int64_t base_count = base_tokens.size(0);
  if (base_count > 0) {
    const int64_t total = k_count * heads * base_count;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_selected_mask_set_base_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        selected_mask.data_ptr<unsigned char>(),
        base_tokens.data_ptr<int64_t>(),
        k_count,
        heads,
        context_len,
        base_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (max_rank_take > 0) {
    const int64_t total = k_count * heads * max_rank_take;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_selected_mask_set_ranked_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        selected_mask.data_ptr<unsigned char>(),
        ranked_prefix_tokens.data_ptr<int64_t>(),
        k_take_counts.data_ptr<int64_t>(),
        k_count,
        heads,
        max_rank_take,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int64_t fit_blocks = rows;
  const size_t fit_shared_bytes = static_cast<size_t>(threads) * 5 * sizeof(float);
  joint_affine_selected_fit_kernel<<<static_cast<unsigned int>(fit_blocks), threads, fit_shared_bytes, stream>>>(
      pq_logits.data_ptr<float>(),
      y_indexed.data_ptr<float>(),
      indexed_tokens.data_ptr<int64_t>(),
      selected_mask.data_ptr<unsigned char>(),
      fit_scale.data_ptr<float>(),
      fit_bias.data_ptr<float>(),
      k_count,
      heads,
      indexed_count,
      context_len,
      static_cast<float>(pq_scale),
      calibrate);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  joint_mixed_row_stats_from_mask_kernel<<<
      static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      selected_mask.data_ptr<unsigned char>(),
      k_take_counts.data_ptr<int64_t>(),
      fit_scale.data_ptr<float>(),
      fit_bias.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      rows,
      heads,
      indexed_count,
      max_rank_take,
      context_len,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t total = rows * context_len;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_mixed_probs_risk_inputs_from_mask_kernel<<<static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      selected_mask.data_ptr<unsigned char>(),
      k_take_counts.data_ptr<int64_t>(),
      fit_scale.data_ptr<float>(),
      fit_bias.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      code_error.data_ptr<float>(),
      probs.data_ptr<float>(),
      risk_in.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      rows,
      heads,
      indexed_count,
      max_rank_take,
      context_len,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  joint_base_outputs_from_probs_kernel<dims_per_block><<<
      static_cast<unsigned int>(rows * dim_tiles), threads, 0, stream>>>(
      probs.data_ptr<float>(),
      vhat.data_ptr<float>(),
      base_flat.data_ptr<float>(),
      rows,
      context_len,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t offset_blocks = ((rows + 1) + threads - 1) / threads;
  joint_segment_offsets_kernel<<<static_cast<unsigned int>(offset_blocks), threads, 0, stream>>>(
      offsets.data_ptr<int64_t>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t temp_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      temp_bytes,
      risk_in.data_ptr<float>(),
      risk_out.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      ids_out.data_ptr<int32_t>(),
      total,
      rows,
      offsets.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  auto temp = torch::empty({static_cast<int64_t>(std::max<size_t>(temp_bytes, 1))}, byte_opts);
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temp.data_ptr(),
      temp_bytes,
      risk_in.data_ptr<float>(),
      risk_out.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      ids_out.data_ptr<int32_t>(),
      total,
      rows,
      offsets.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t interval_blocks = rows * v_steps * dim_tiles;
  joint_grouped_flat_vprefix_interval_sums_kernel<int32_t, dims_per_block><<<
      static_cast<unsigned int>(interval_blocks), threads, 0, stream>>>(
      probs.data_ptr<float>(),
      residual.data_ptr<float>(),
      ids_out.data_ptr<int32_t>(),
      v_budgets.data_ptr<int64_t>(),
      interval_sums.data_ptr<float>(),
      1,
      k_count,
      heads,
      context_len,
      context_len,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const size_t shared_bytes = static_cast<size_t>(threads) * 4 * sizeof(double);
  joint_select_policy_grouped_intervals_no_mb_kernel<<<
      static_cast<unsigned int>(heads), threads, shared_bytes, stream>>>(
      base_flat.data_ptr<float>(),
      interval_sums.data_ptr<float>(),
      final_indices.data_ptr<int64_t>(),
      final_outputs.data_ptr<float>(),
      1,
      k_count,
      v_steps,
      heads,
      dim,
      threshold,
      static_cast<int>(policy_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {final_outputs, final_indices};
}

