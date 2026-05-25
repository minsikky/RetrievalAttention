torch::Tensor joint_vprefix_outputs_from_risk_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual,
    torch::Tensor code_error,
    torch::Tensor v_budgets) {
  const auto k_count = base_outputs.size(0);
  const auto heads = base_outputs.size(1);
  const auto dim = base_outputs.size(2);
  const auto context_len = probs.size(2);
  const auto v_steps = v_budgets.size(0);
  auto outputs = torch::empty({k_count, v_steps, heads, dim}, base_outputs.options().dtype(torch::kFloat32));
  if (k_count == 0 || heads == 0 || dim == 0 || v_steps == 0) {
    return outputs;
  }
  if (context_len == 0) {
    return outputs.copy_(base_outputs.reshape({k_count, 1, heads, dim}).expand({k_count, v_steps, heads, dim}));
  }

  const int64_t rows = k_count * heads;
  auto flat_opts = base_outputs.options().dtype(torch::kFloat32);
  auto long_opts = base_outputs.options().dtype(torch::kLong);
  auto int_opts = base_outputs.options().dtype(torch::kInt);
  auto byte_opts = base_outputs.options().dtype(torch::kUInt8);
  auto risk_in = torch::empty({rows, context_len}, flat_opts);
  auto risk_out = torch::empty({rows, context_len}, flat_opts);
  auto ids_in = torch::empty({rows, context_len}, int_opts);
  auto ids_out = torch::empty({rows, context_len}, int_opts);
  auto offsets = torch::empty({rows + 1}, long_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = rows * context_len;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_risk_sort_inputs_kernel<<<static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      probs.data_ptr<float>(),
      code_error.data_ptr<float>(),
      risk_in.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      rows,
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
  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, byte_opts);
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

  const int64_t out_total = k_count * heads * dim;
  const int64_t out_blocks = (out_total + threads - 1) / threads;
  joint_vprefix_outputs_kernel<int32_t><<<static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
      base_outputs.data_ptr<float>(),
      probs.data_ptr<float>(),
      residual.data_ptr<float>(),
      ids_out.data_ptr<int32_t>(),
      v_budgets.data_ptr<int64_t>(),
      outputs.data_ptr<float>(),
      k_count,
      heads,
      context_len,
      context_len,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor joint_vprefix_outputs_from_grouped_risk_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor row_group_ids,
    torch::Tensor v_budgets) {
  const auto rows = base_outputs.size(0);
  const auto dim = base_outputs.size(1);
  const auto context_len = probs.size(1);
  const auto v_steps = v_budgets.size(0);
  auto outputs = torch::empty({rows, v_steps, dim}, base_outputs.options().dtype(torch::kFloat32));
  if (rows == 0 || dim == 0 || v_steps == 0) {
    return outputs;
  }
  if (context_len == 0) {
    return outputs.copy_(base_outputs.reshape({rows, 1, dim}).expand({rows, v_steps, dim}));
  }

  auto flat_opts = base_outputs.options().dtype(torch::kFloat32);
  auto long_opts = base_outputs.options().dtype(torch::kLong);
  auto int_opts = base_outputs.options().dtype(torch::kInt);
  auto byte_opts = base_outputs.options().dtype(torch::kUInt8);
  auto risk_in = torch::empty({rows, context_len}, flat_opts);
  auto risk_out = torch::empty({rows, context_len}, flat_opts);
  auto ids_in = torch::empty({rows, context_len}, int_opts);
  auto ids_out = torch::empty({rows, context_len}, int_opts);
  auto offsets = torch::empty({rows + 1}, long_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = rows * context_len;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_grouped_risk_sort_inputs_kernel<<<static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      probs.data_ptr<float>(),
      code_error_groups.data_ptr<float>(),
      row_group_ids.data_ptr<int64_t>(),
      risk_in.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      rows,
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
  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, byte_opts);
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

  const int64_t out_total = rows * dim;
  const int64_t out_blocks = (out_total + threads - 1) / threads;
  joint_grouped_vprefix_outputs_kernel<int32_t><<<static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
      base_outputs.data_ptr<float>(),
      probs.data_ptr<float>(),
      residual_groups.data_ptr<float>(),
      row_group_ids.data_ptr<int64_t>(),
      ids_out.data_ptr<int32_t>(),
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

torch::Tensor joint_vprefix_outputs_from_grouped_risk_batched_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets) {
  const auto groups = base_outputs.size(0);
  const auto k_count = base_outputs.size(1);
  const auto heads = base_outputs.size(2);
  const auto dim = base_outputs.size(3);
  const auto context_len = probs.size(3);
  const auto v_steps = v_budgets.size(0);
  const auto rows = groups * k_count * heads;
  auto outputs = torch::empty({rows, v_steps, dim}, base_outputs.options().dtype(torch::kFloat32));
  if (rows == 0 || dim == 0 || v_steps == 0) {
    return outputs;
  }
  auto base_flat = base_outputs.reshape({rows, dim});
  auto probs_flat = probs.reshape({rows, context_len});
  if (context_len == 0) {
    return outputs.copy_(base_flat.reshape({rows, 1, dim}).expand({rows, v_steps, dim}));
  }

  auto flat_opts = base_outputs.options().dtype(torch::kFloat32);
  auto long_opts = base_outputs.options().dtype(torch::kLong);
  auto int_opts = base_outputs.options().dtype(torch::kInt);
  auto byte_opts = base_outputs.options().dtype(torch::kUInt8);
  auto risk_in = torch::empty({rows, context_len}, flat_opts);
  auto risk_out = torch::empty({rows, context_len}, flat_opts);
  auto ids_in = torch::empty({rows, context_len}, int_opts);
  auto ids_out = torch::empty({rows, context_len}, int_opts);
  auto offsets = torch::empty({rows + 1}, long_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = rows * context_len;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_grouped_flat_risk_sort_inputs_kernel<<<static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      probs_flat.data_ptr<float>(),
      code_error_groups.data_ptr<float>(),
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
  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, byte_opts);
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

  auto interval_sums = torch::empty({rows, v_steps, dim}, flat_opts);
  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t interval_blocks = rows * v_steps * dim_tiles;
  joint_grouped_flat_vprefix_interval_sums_kernel<int32_t, dims_per_block><<<
      static_cast<unsigned int>(interval_blocks), threads, 0, stream>>>(
      probs_flat.data_ptr<float>(),
      residual_groups.data_ptr<float>(),
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

int64_t joint_grouped_risk_sort_temp_bytes_cuda(int64_t rows, int64_t context_len) {
  rows = std::max<int64_t>(0, rows);
  context_len = std::max<int64_t>(0, context_len);
  const int64_t total = rows * context_len;
  if (rows == 0 || context_len == 0 || total == 0) {
    return 1;
  }
  size_t temp_bytes = 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      temp_bytes,
      static_cast<float*>(nullptr),
      static_cast<float*>(nullptr),
      static_cast<int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      total,
      rows,
      static_cast<int64_t*>(nullptr),
      static_cast<int64_t*>(nullptr),
      0,
      sizeof(float) * 8,
      stream);
  return static_cast<int64_t>(std::max<size_t>(temp_bytes, 1));
}

torch::Tensor joint_vprefix_outputs_from_grouped_risk_batched_workspace_cuda(
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
  const auto groups = base_outputs.size(0);
  const auto k_count = base_outputs.size(1);
  const auto heads = base_outputs.size(2);
  const auto dim = base_outputs.size(3);
  const auto context_len = probs.size(3);
  const auto v_steps = v_budgets.size(0);
  const auto rows = groups * k_count * heads;
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
  auto base_flat = base_outputs.reshape({rows, dim});
  auto probs_flat = probs.reshape({rows, context_len});
  if (context_len == 0) {
    return outputs_view.copy_(base_flat.reshape({rows, 1, dim}).expand({rows, v_steps, dim}));
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = rows * context_len;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_grouped_flat_risk_sort_inputs_kernel<<<static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      probs_flat.data_ptr<float>(),
      code_error_groups.data_ptr<float>(),
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

  auto interval_view = interval_sums.reshape({rows, v_steps, dim});
  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t interval_blocks = rows * v_steps * dim_tiles;
  joint_grouped_flat_vprefix_interval_sums_kernel<int32_t, dims_per_block><<<
      static_cast<unsigned int>(interval_blocks), threads, 0, stream>>>(
      probs_flat.data_ptr<float>(),
      residual_groups.data_ptr<float>(),
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
      base_flat.data_ptr<float>(),
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

torch::Tensor joint_vprefix_outputs_from_grouped_risk_batched_strided_workspace_cuda(
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
  const auto groups = base_outputs.size(0);
  const auto k_count = base_outputs.size(1);
  const auto heads = base_outputs.size(2);
  const auto dim = base_outputs.size(3);
  const auto prob_stride = probs.size(3);
  const auto v_steps = v_budgets.size(0);
  const auto rows = groups * k_count * heads;
  context_len = std::max<int64_t>(0, std::min<int64_t>(context_len, prob_stride));
  TORCH_CHECK(probs.dim() == 4, "probs shape must be [groups, k_count, heads, capacity]");
  TORCH_CHECK(probs.size(0) == groups && probs.size(1) == k_count && probs.size(2) == heads,
              "probs grouped shape mismatch");
  TORCH_CHECK(probs.is_contiguous(), "strided probs workspace must be contiguous");
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
  auto base_flat = base_outputs.reshape({rows, dim});
  auto probs_flat = probs.reshape({rows, prob_stride});
  if (context_len == 0) {
    return outputs_view.copy_(base_flat.reshape({rows, 1, dim}).expand({rows, v_steps, dim}));
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = rows * context_len;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_grouped_flat_risk_sort_inputs_strided_kernel<<<static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      probs_flat.data_ptr<float>(),
      code_error_groups.data_ptr<float>(),
      risk_in.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      groups,
      k_count,
      heads,
      context_len,
      prob_stride);
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

  auto interval_view = interval_sums.reshape({rows, v_steps, dim});
  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t interval_blocks = rows * v_steps * dim_tiles;
  joint_grouped_flat_vprefix_interval_sums_strided_kernel<int32_t, dims_per_block><<<
      static_cast<unsigned int>(interval_blocks), threads, 0, stream>>>(
      probs_flat.data_ptr<float>(),
      residual_groups.data_ptr<float>(),
      ids_out.data_ptr<int32_t>(),
      v_budgets.data_ptr<int64_t>(),
      interval_view.data_ptr<float>(),
      groups,
      k_count,
      heads,
      context_len,
      prob_stride,
      context_len,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t out_total = rows * dim;
  const int64_t out_blocks = (out_total + threads - 1) / threads;
  joint_grouped_flat_vprefix_prefix_intervals_kernel<<<static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
      base_flat.data_ptr<float>(),
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

torch::Tensor joint_vprefix_outputs_from_grouped_risk_topk_batched_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    int64_t max_exact) {
  const auto groups = base_outputs.size(0);
  const auto k_count = base_outputs.size(1);
  const auto heads = base_outputs.size(2);
  const auto dim = base_outputs.size(3);
  const auto context_len = probs.size(3);
  const auto v_steps = v_budgets.size(0);
  const auto rows = groups * k_count * heads;
  auto outputs = torch::empty({rows, v_steps, dim}, base_outputs.options().dtype(torch::kFloat32));
  if (rows == 0 || dim == 0 || v_steps == 0) {
    return outputs;
  }
  auto base_flat = base_outputs.reshape({rows, dim});
  auto probs_flat = probs.reshape({rows, context_len});
  if (context_len == 0) {
    return outputs.copy_(base_flat.reshape({rows, 1, dim}).expand({rows, v_steps, dim}));
  }

  max_exact = std::max<int64_t>(0, std::min<int64_t>(max_exact, context_len));
  if (max_exact == 0) {
    return outputs.copy_(base_flat.reshape({rows, 1, dim}).expand({rows, v_steps, dim}));
  }

  auto flat_opts = base_outputs.options().dtype(torch::kFloat32);
  auto risk = torch::empty({rows, context_len}, flat_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = rows * context_len;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_grouped_flat_risk_values_kernel<<<static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      probs_flat.data_ptr<float>(),
      code_error_groups.data_ptr<float>(),
      risk.data_ptr<float>(),
      groups,
      k_count,
      heads,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto topk = at::topk(risk, max_exact, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  auto exact_order = std::get<1>(topk).contiguous();

  auto interval_sums = torch::empty({rows, v_steps, dim}, flat_opts);
  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t interval_blocks = rows * v_steps * dim_tiles;
  joint_grouped_flat_vprefix_interval_sums_kernel<int64_t, dims_per_block><<<
      static_cast<unsigned int>(interval_blocks), threads, 0, stream>>>(
      probs_flat.data_ptr<float>(),
      residual_groups.data_ptr<float>(),
      exact_order.data_ptr<int64_t>(),
      v_budgets.data_ptr<int64_t>(),
      interval_sums.data_ptr<float>(),
      groups,
      k_count,
      heads,
      context_len,
      max_exact,
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
      max_exact,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

