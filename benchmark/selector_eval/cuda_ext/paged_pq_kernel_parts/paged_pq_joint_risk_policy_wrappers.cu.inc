__global__ void joint_grouped_accounting_sums_kernel(
    const int64_t* __restrict__ final_indices,
    const int64_t* __restrict__ selected_counts_by_ki,
    const int64_t* __restrict__ v_budgets,
    double* __restrict__ sums,
    int64_t heads,
    int64_t k_count,
    int64_t v_count,
    int64_t context_len,
    int64_t head_dim,
    int64_t key_bytes,
    int64_t value_bytes,
    double selector_mb,
    double v_pq_codebook_mb,
    double metadata_mb,
    int64_t actual_value_subvecs,
    int64_t code_bytes) {
  extern __shared__ double joint_accounting_shared[];
  for (int i = threadIdx.x; i < 11 * blockDim.x; i += blockDim.x) {
    joint_accounting_shared[i] = 0.0;
  }
  __syncthreads();
  double local[11] = {0.0};
  constexpr double mb = 1024.0 * 1024.0;
  for (int64_t h = threadIdx.x; h < heads; h += blockDim.x) {
    int64_t ki = final_indices[h * 2];
    int64_t vi = final_indices[h * 2 + 1];
    ki = ki < 0 ? 0 : (ki >= k_count ? k_count - 1 : ki);
    vi = vi < 0 ? 0 : (vi >= v_count ? v_count - 1 : vi);
    int64_t selected_count = selected_counts_by_ki[ki] < 0 ? 0 : selected_counts_by_ki[ki];
    int64_t exact_v = v_budgets[vi] < 0 ? 0 : (v_budgets[vi] > context_len ? context_len : v_budgets[vi]);
    int64_t tail_count = context_len > exact_v ? context_len - exact_v : 0;
    double exact_key_mb = static_cast<double>(selected_count * head_dim * key_bytes) / mb;
    double exact_v_mb = static_cast<double>(exact_v * head_dim * value_bytes) / mb;
    double tail_mb =
        v_pq_codebook_mb + static_cast<double>(tail_count * actual_value_subvecs * code_bytes) / mb + metadata_mb;
    double dense_key_mb = static_cast<double>(context_len * head_dim * key_bytes) / mb;
    local[0] += static_cast<double>(selected_count);
    local[1] += static_cast<double>(tail_count);
    local[2] += selector_mb;
    local[3] += exact_key_mb + exact_v_mb;
    local[4] += tail_mb;
    local[6] += dense_key_mb + exact_v_mb;
    local[8] += selector_mb > 0.0 ? 1.0 : 0.0;
    local[9] += tail_mb > 0.0 ? 1.0 : 0.0;
  }
  for (int j = 0; j < 11; ++j) {
    joint_accounting_shared[j * blockDim.x + threadIdx.x] = local[j];
  }
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      for (int j = 0; j < 11; ++j) {
        joint_accounting_shared[j * blockDim.x + threadIdx.x] +=
            joint_accounting_shared[j * blockDim.x + threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int j = 0; j < 11; ++j) {
      sums[j] = joint_accounting_shared[j * blockDim.x];
    }
  }
}

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
    int64_t policy_id) {
  const auto rows = base_outputs.size(0);
  const auto dim = base_outputs.size(1);
  const auto context_len = probs.size(1);
  const auto groups = k_mb.size(0);
  const auto v_count = v_budgets.size(0);
  auto final_outputs = torch::empty({groups, heads, dim}, base_outputs.options().dtype(torch::kFloat32));
  auto final_indices = torch::empty({groups, heads, 2}, base_outputs.options().dtype(torch::kLong));
  if (groups == 0 || heads == 0 || k_count == 0 || v_count == 0 || dim == 0) {
    return {final_outputs, final_indices};
  }

  auto opts = base_outputs.options().dtype(torch::kFloat32);
  auto int_opts = base_outputs.options().dtype(torch::kInt);
  auto long_opts = base_outputs.options().dtype(torch::kLong);
  auto byte_opts = base_outputs.options().dtype(torch::kUInt8);
  auto risk_in = torch::empty({rows, context_len}, opts);
  auto risk_out = torch::empty({rows, context_len}, opts);
  auto ids_in = torch::empty({rows, context_len}, int_opts);
  auto ids_out = torch::empty({rows, context_len}, int_opts);
  auto offsets = torch::empty({rows + 1}, long_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = rows * context_len;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_grouped_flat_risk_sort_inputs_kernel<<<static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      probs.data_ptr<float>(),
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

  const int shared_bytes = threads * 4 * sizeof(double);
  joint_select_policy_grouped_risk_kernel<int32_t>
      <<<static_cast<unsigned int>(groups * heads), threads, shared_bytes, stream>>>(
          base_outputs.data_ptr<float>(),
          probs.data_ptr<float>(),
          residual_groups.data_ptr<float>(),
          ids_out.data_ptr<int32_t>(),
          v_budgets.data_ptr<int64_t>(),
          k_mb.data_ptr<float>(),
          v_mb.data_ptr<float>(),
          final_indices.data_ptr<int64_t>(),
          final_outputs.data_ptr<float>(),
          groups,
          k_count,
          v_count,
          heads,
          dim,
          context_len,
          threshold,
          static_cast<int>(policy_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {final_outputs, final_indices};
}

std::vector<torch::Tensor> joint_select_policy_from_grouped_risk_batched_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    torch::Tensor k_mb,
    torch::Tensor v_mb,
    double threshold,
    int64_t policy_id) {
  const auto groups = base_outputs.size(0);
  const auto k_count = base_outputs.size(1);
  const auto heads = base_outputs.size(2);
  const auto dim = base_outputs.size(3);
  const auto context_len = probs.size(3);
  const auto rows = groups * k_count * heads;
  return joint_select_policy_from_grouped_risk_cuda(
      base_outputs.reshape({rows, dim}),
      probs.reshape({rows, context_len}),
      residual_groups,
      code_error_groups,
      v_budgets,
      k_mb,
      v_mb,
      k_count,
      heads,
      threshold,
      policy_id);
}

std::vector<torch::Tensor> joint_select_policy_from_grouped_risk_no_mb_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    int64_t k_count,
    int64_t heads,
    double threshold,
    int64_t policy_id) {
  const auto groups = base_outputs.size(0) / (k_count * heads);
  auto opts = base_outputs.options().dtype(torch::kFloat32);
  auto k_mb = torch::empty({groups, k_count}, opts);
  auto v_mb = torch::empty({groups, v_budgets.size(0)}, opts);
  return joint_select_policy_from_grouped_risk_cuda(
      base_outputs,
      probs,
      residual_groups,
      code_error_groups,
      v_budgets,
      k_mb,
      v_mb,
      k_count,
      heads,
      threshold,
      policy_id);
}

std::vector<torch::Tensor> joint_select_policy_from_grouped_risk_batched_no_mb_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    double threshold,
    int64_t policy_id) {
  const auto groups = base_outputs.size(0);
  const auto k_count = base_outputs.size(1);
  const auto heads = base_outputs.size(2);
  const auto dim = base_outputs.size(3);
  const auto context_len = probs.size(3);
  return joint_select_policy_from_grouped_risk_no_mb_cuda(
      base_outputs.reshape({groups * k_count * heads, dim}),
      probs.reshape({groups * k_count * heads, context_len}),
      residual_groups,
      code_error_groups,
      v_budgets,
      k_count,
      heads,
      threshold,
      policy_id);
}

std::vector<torch::Tensor> joint_select_policy_from_grouped_risk_intervals_batched_no_mb_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    double threshold,
    int64_t policy_id) {
  const auto groups = base_outputs.size(0);
  const auto k_count = base_outputs.size(1);
  const auto heads = base_outputs.size(2);
  const auto dim = base_outputs.size(3);
  const auto context_len = probs.size(3);
  const auto v_count = v_budgets.size(0);
  const auto rows = groups * k_count * heads;
  auto final_outputs = torch::empty({groups, heads, dim}, base_outputs.options().dtype(torch::kFloat32));
  auto final_indices = torch::empty({groups, heads, 2}, base_outputs.options().dtype(torch::kLong));
  if (rows == 0 || dim == 0 || v_count == 0) {
    return {final_outputs, final_indices};
  }
  auto base_flat = base_outputs.reshape({rows, dim});
  auto probs_flat = probs.reshape({rows, context_len});
  auto opts = base_outputs.options().dtype(torch::kFloat32);
  auto int_opts = base_outputs.options().dtype(torch::kInt);
  auto long_opts = base_outputs.options().dtype(torch::kLong);
  auto byte_opts = base_outputs.options().dtype(torch::kUInt8);
  auto risk_in = torch::empty({rows, context_len}, opts);
  auto risk_out = torch::empty({rows, context_len}, opts);
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
  auto interval_sums = torch::empty({rows, v_count, dim}, opts);
  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t interval_blocks = rows * v_count * dim_tiles;
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
      v_count,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int shared_bytes = threads * 4 * sizeof(double);
  joint_select_policy_grouped_intervals_no_mb_kernel<<<
      static_cast<unsigned int>(groups * heads), threads, shared_bytes, stream>>>(
      base_flat.data_ptr<float>(),
      interval_sums.data_ptr<float>(),
      final_indices.data_ptr<int64_t>(),
      final_outputs.data_ptr<float>(),
      groups,
      k_count,
      v_count,
      heads,
      dim,
      threshold,
      static_cast<int>(policy_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {final_outputs, final_indices};
}

torch::Tensor joint_select_policy_cuda(
    torch::Tensor output_grid,
    torch::Tensor k_mb,
    torch::Tensor v_mb,
    double threshold,
    int64_t policy_id) {
  const auto k_count = output_grid.size(0);
  const auto v_count = output_grid.size(1);
  const auto heads = output_grid.size(2);
  const auto dim = output_grid.size(3);
  auto final_indices = torch::empty({heads, 2}, output_grid.options().dtype(torch::kLong));
  if (heads == 0) {
    return final_indices;
  }
  const int threads = 256;
  const int shared_bytes = threads * 4 * sizeof(double);
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_select_policy_kernel<<<static_cast<unsigned int>(heads), threads, shared_bytes, stream>>>(
      output_grid.data_ptr<float>(),
      k_mb.data_ptr<float>(),
      v_mb.data_ptr<float>(),
      final_indices.data_ptr<int64_t>(),
      k_count,
      v_count,
      heads,
      dim,
      threshold,
      static_cast<int>(policy_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return final_indices;
}

std::vector<torch::Tensor> joint_select_policy_grouped_flat_cuda(
    torch::Tensor outputs_flat,
    torch::Tensor k_mb,
    torch::Tensor v_mb,
    int64_t k_count,
    int64_t heads,
    double threshold,
    int64_t policy_id) {
  const auto rows = outputs_flat.size(0);
  const auto v_count = outputs_flat.size(1);
  const auto dim = outputs_flat.size(2);
  const auto groups = k_mb.size(0);
  auto final_outputs = torch::empty({groups, heads, dim}, outputs_flat.options().dtype(torch::kFloat32));
  auto final_indices = torch::empty({groups, heads, 2}, outputs_flat.options().dtype(torch::kLong));
  if (groups == 0 || heads == 0) {
    return {final_outputs, final_indices};
  }
  const int threads = 256;
  const int shared_bytes = threads * 4 * sizeof(double);
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_select_policy_grouped_flat_kernel<<<static_cast<unsigned int>(groups * heads), threads, shared_bytes, stream>>>(
      outputs_flat.data_ptr<float>(),
      k_mb.data_ptr<float>(),
      v_mb.data_ptr<float>(),
      final_indices.data_ptr<int64_t>(),
      final_outputs.data_ptr<float>(),
      groups,
      k_count,
      v_count,
      heads,
      dim,
      threshold,
      static_cast<int>(policy_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {final_outputs, final_indices};
}

std::vector<torch::Tensor> joint_select_policy_grouped_flat_no_mb_cuda(
    torch::Tensor outputs_flat,
    int64_t k_count,
    int64_t heads,
    double threshold,
    int64_t policy_id) {
  const auto rows = outputs_flat.size(0);
  const auto v_count = outputs_flat.size(1);
  const auto groups = rows / (k_count * heads);
  const auto dim = outputs_flat.size(2);
  auto final_outputs = torch::empty({groups, heads, dim}, outputs_flat.options().dtype(torch::kFloat32));
  auto final_indices = torch::empty({groups, heads, 2}, outputs_flat.options().dtype(torch::kLong));
  if (groups == 0 || v_count == 0 || dim == 0) {
    return {final_outputs, final_indices};
  }
  const int threads = 256;
  const int shared_bytes = threads * 4 * sizeof(double);
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_select_policy_grouped_flat_kernel<<<static_cast<unsigned int>(groups * heads), threads, shared_bytes, stream>>>(
      outputs_flat.data_ptr<float>(),
      nullptr,
      nullptr,
      final_indices.data_ptr<int64_t>(),
      final_outputs.data_ptr<float>(),
      groups,
      k_count,
      v_count,
      heads,
      dim,
      threshold,
      static_cast<int>(policy_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {final_outputs, final_indices};
}
