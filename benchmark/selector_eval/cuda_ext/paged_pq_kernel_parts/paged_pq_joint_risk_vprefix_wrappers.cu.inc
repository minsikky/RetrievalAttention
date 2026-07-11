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

std::vector<torch::Tensor> joint_vprefix_outputs_precision_from_risk_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_hi,
    torch::Tensor residual_lo,
    torch::Tensor commit_mask,
    torch::Tensor code_error,
    torch::Tensor v_budgets,
    torch::Tensor v_hi_counts) {
  const auto k_count = base_outputs.size(0);
  const auto heads = base_outputs.size(1);
  const auto dim = base_outputs.size(2);
  const auto context_len = probs.size(2);
  const auto v_steps = v_budgets.size(0);
  auto outputs = torch::empty(
      {k_count, v_steps, heads, dim},
      base_outputs.options().dtype(torch::kFloat32));
  auto lo_reads = torch::empty(
      {k_count, v_steps, heads},
      base_outputs.options().dtype(torch::kInt));
  if (k_count == 0 || heads == 0 || dim == 0 || v_steps == 0) {
    return {outputs, lo_reads};
  }
  if (context_len == 0) {
    outputs.copy_(base_outputs.reshape({k_count, 1, heads, dim}).expand(
        {k_count, v_steps, heads, dim}));
    lo_reads.zero_();
    return {outputs, lo_reads};
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
  joint_vprefix_precision_outputs_kernel<int32_t><<<
      static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
      base_outputs.data_ptr<float>(),
      probs.data_ptr<float>(),
      residual_hi.data_ptr<float>(),
      residual_lo.data_ptr<float>(),
      commit_mask.data_ptr<bool>(),
      ids_out.data_ptr<int32_t>(),
      v_budgets.data_ptr<int64_t>(),
      v_hi_counts.data_ptr<int64_t>(),
      outputs.data_ptr<float>(),
      lo_reads.data_ptr<int32_t>(),
      k_count,
      heads,
      context_len,
      context_len,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {outputs, lo_reads};
}

std::vector<torch::Tensor> joint_vprefix_outputs_precision_from_risk_tokpar_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_hi,
    torch::Tensor residual_lo,
    torch::Tensor commit_mask,
    torch::Tensor code_error,
    torch::Tensor v_budgets,
    torch::Tensor v_hi_counts) {
  const auto k_count = base_outputs.size(0);
  const auto heads = base_outputs.size(1);
  const auto dim = base_outputs.size(2);
  const auto context_len = probs.size(2);
  const auto v_steps = v_budgets.size(0);
  auto outputs = torch::empty(
      {k_count, v_steps, heads, dim},
      base_outputs.options().dtype(torch::kFloat32));
  auto lo_reads = torch::empty(
      {k_count, v_steps, heads},
      base_outputs.options().dtype(torch::kInt));
  if (k_count == 0 || heads == 0 || dim == 0 || v_steps == 0) {
    return {outputs, lo_reads};
  }
  if (context_len == 0) {
    outputs.copy_(base_outputs.reshape({k_count, 1, heads, dim}).expand(
        {k_count, v_steps, heads, dim}));
    lo_reads.zero_();
    return {outputs, lo_reads};
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

  launch_joint_vprefix_precision_tokpar<int32_t>(
      base_outputs,
      probs,
      residual_hi,
      residual_lo,
      commit_mask,
      ids_out,
      v_budgets,
      v_hi_counts,
      outputs,
      lo_reads);
  return {outputs, lo_reads};
}

std::vector<torch::Tensor> joint_vprefix_outputs_precision_from_risk_tokpar_batched_cuda(
    std::vector<torch::Tensor> base_outputs_groups,
    std::vector<torch::Tensor> probs_groups,
    std::vector<torch::Tensor> residual_hi_groups,
    std::vector<torch::Tensor> residual_lo_groups,
    std::vector<torch::Tensor> commit_mask_groups,
    std::vector<torch::Tensor> code_error_groups,
    torch::Tensor v_budgets,
    torch::Tensor v_hi_counts,
    int64_t transient_budget_bytes,
    bool vectorized_partials) {
  const int64_t groups = static_cast<int64_t>(base_outputs_groups.size());
  TORCH_CHECK(groups > 0, "batched V-prefix requires at least one group");
  const auto k_count = base_outputs_groups[0].size(0);
  const auto heads = base_outputs_groups[0].size(1);
  const auto dim = base_outputs_groups[0].size(2);
  const auto context_len = probs_groups[0].size(2);
  const auto v_steps = v_budgets.size(0);
  const int64_t rows_per_group = k_count * heads;
  const int64_t rows = groups * rows_per_group;
  const int64_t chunks =
      (context_len + kJointVPrefixTokenChunk - 1) / kJointVPrefixTokenChunk;
  auto output_opts = base_outputs_groups[0].options().dtype(torch::kFloat32);
  auto outputs = torch::empty(
      {groups, k_count, v_steps, heads, dim}, output_opts);
  auto lo_reads = torch::empty(
      {groups, k_count, v_steps, heads},
      base_outputs_groups[0].options().dtype(torch::kInt));
  if (rows == 0 || dim == 0 || v_steps == 0) {
    return {outputs, lo_reads};
  }
  if (context_len == 0) {
    for (int64_t group = 0; group < groups; ++group) {
      outputs[group].copy_(base_outputs_groups[group].reshape(
          {k_count, 1, heads, dim}).expand({k_count, v_steps, heads, dim}));
    }
    lo_reads.zero_();
    return {outputs, lo_reads};
  }

  auto flat_opts = output_opts;
  auto long_opts = base_outputs_groups[0].options().dtype(torch::kLong);
  auto int_opts = base_outputs_groups[0].options().dtype(torch::kInt);
  auto byte_opts = base_outputs_groups[0].options().dtype(torch::kUInt8);
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t temp_bytes = 0;
  const int64_t total = rows * context_len;
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
  const int64_t pointer_bytes =
      groups * kJointBatchedVPrefixPointerStride * static_cast<int64_t>(sizeof(int64_t));
  const int64_t sort_bytes =
      total * static_cast<int64_t>(2 * sizeof(float) + 2 * sizeof(int32_t)) +
      (rows + 1) * static_cast<int64_t>(sizeof(int64_t)) +
      static_cast<int64_t>(temp_bytes);
  const int64_t partial_bytes =
      chunks * rows *
      (dim * static_cast<int64_t>(2 * sizeof(float)) +
       static_cast<int64_t>(sizeof(int32_t)));
  const int64_t required_bytes = pointer_bytes + sort_bytes + partial_bytes;
  TORCH_CHECK(
      transient_budget_bytes > 0,
      "batched V-prefix transient budget must be positive");
  TORCH_CHECK(
      required_bytes <= transient_budget_bytes,
      "batched V-prefix requires ", required_bytes,
      " transient bytes (sort=", sort_bytes,
      ", partials=", partial_bytes,
      ", pointers=", pointer_bytes,
      ") but budget is ", transient_budget_bytes);

  auto risk_in = torch::empty({rows, context_len}, flat_opts);
  auto risk_out = torch::empty({rows, context_len}, flat_opts);
  auto ids_in = torch::empty({rows, context_len}, int_opts);
  auto ids_out = torch::empty({rows, context_len}, int_opts);
  auto offsets = torch::empty({rows + 1}, long_opts);
  auto pointer_table = torch::empty(
      {groups, kJointBatchedVPrefixPointerStride}, long_opts);
  std::vector<int64_t> host_pointers(
      static_cast<size_t>(groups * kJointBatchedVPrefixPointerStride));
  for (int64_t group = 0; group < groups; ++group) {
    const int64_t base = group * kJointBatchedVPrefixPointerStride;
    host_pointers[base + kJointBatchedVPrefixBasePointer] =
        reinterpret_cast<int64_t>(base_outputs_groups[group].data_ptr<float>());
    host_pointers[base + kJointBatchedVPrefixProbsPointer] =
        reinterpret_cast<int64_t>(probs_groups[group].data_ptr<float>());
    host_pointers[base + kJointBatchedVPrefixResidualHiPointer] =
        reinterpret_cast<int64_t>(residual_hi_groups[group].data_ptr<float>());
    host_pointers[base + kJointBatchedVPrefixResidualLoPointer] =
        reinterpret_cast<int64_t>(residual_lo_groups[group].data_ptr<float>());
    host_pointers[base + kJointBatchedVPrefixCommitPointer] =
        reinterpret_cast<int64_t>(commit_mask_groups[group].data_ptr<bool>());
    host_pointers[base + kJointBatchedVPrefixCodeErrorPointer] =
        reinterpret_cast<int64_t>(code_error_groups[group].data_ptr<float>());
  }
  C10_CUDA_CHECK(cudaMemcpyAsync(
      pointer_table.data_ptr<int64_t>(),
      host_pointers.data(),
      static_cast<size_t>(pointer_bytes),
      cudaMemcpyHostToDevice,
      stream));

  const int threads = 256;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_batched_risk_sort_inputs_kernel<<<
      static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      pointer_table.data_ptr<int64_t>(),
      risk_in.data_ptr<float>(),
      ids_in.data_ptr<int32_t>(),
      groups,
      rows_per_group,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t offset_blocks = ((rows + 1) + threads - 1) / threads;
  joint_segment_offsets_kernel<<<
      static_cast<unsigned int>(offset_blocks), threads, 0, stream>>>(
      offsets.data_ptr<int64_t>(), rows, context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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

  auto hi_partials = torch::empty({chunks, rows, dim}, flat_opts);
  auto lo_partials = torch::empty({chunks, rows, dim}, flat_opts);
  auto commit_partials = torch::empty({chunks, rows}, int_opts);
  if (vectorized_partials) {
    TORCH_CHECK(dim % 4 == 0, "vectorized token partials require dim divisible by four");
    const int64_t dim_tiles = (dim + 127) / 128;
    const int64_t partial_warps = chunks * rows * dim_tiles;
    const int64_t partial_blocks = (partial_warps + 7) / 8;
    joint_batched_vprefix_precision_token_partials_vec4_kernel<int32_t><<<
        static_cast<unsigned int>(partial_blocks), threads, 0, stream>>>(
        pointer_table.data_ptr<int64_t>(),
        ids_out.data_ptr<int32_t>(),
        hi_partials.data_ptr<float>(),
        lo_partials.data_ptr<float>(),
        commit_partials.data_ptr<int32_t>(),
        groups,
        rows_per_group,
        context_len,
        context_len,
        dim,
        chunks);
  } else {
    const int64_t partial_total = chunks * rows * dim;
    const int64_t partial_blocks = (partial_total + threads - 1) / threads;
    joint_batched_vprefix_precision_token_partials_kernel<int32_t><<<
        static_cast<unsigned int>(partial_blocks), threads, 0, stream>>>(
        pointer_table.data_ptr<int64_t>(),
        ids_out.data_ptr<int32_t>(),
        hi_partials.data_ptr<float>(),
        lo_partials.data_ptr<float>(),
        commit_partials.data_ptr<int32_t>(),
        groups,
        rows_per_group,
        context_len,
        context_len,
        dim,
        chunks);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t output_total = rows * dim;
  const int64_t output_blocks = (output_total + threads - 1) / threads;
  joint_batched_vprefix_precision_outputs_tokpar_kernel<int32_t><<<
      static_cast<unsigned int>(output_blocks), threads, 0, stream>>>(
      pointer_table.data_ptr<int64_t>(),
      ids_out.data_ptr<int32_t>(),
      v_budgets.data_ptr<int64_t>(),
      v_hi_counts.data_ptr<int64_t>(),
      hi_partials.data_ptr<float>(),
      lo_partials.data_ptr<float>(),
      commit_partials.data_ptr<int32_t>(),
      outputs.data_ptr<float>(),
      lo_reads.data_ptr<int32_t>(),
      groups,
      k_count,
      heads,
      context_len,
      context_len,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {outputs, lo_reads};
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

torch::Tensor joint_vprefix_outputs_from_grouped_risk_batched_vpq_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor code_error_groups,
    torch::Tensor v_budgets,
    int64_t page_size) {
  const auto groups = base_outputs.size(0);
  const auto k_count = base_outputs.size(1);
  const auto heads = base_outputs.size(2);
  const auto dim = base_outputs.size(3);
  const auto context_len = probs.size(3);
  const auto v_steps = v_budgets.size(0);
  const auto rows = groups * k_count * heads;
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto code_count = value_codebooks.size(3);
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
#define DISPATCH_JOINT_VPQ_INTERVAL(CODE_T)                                                       \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                                                \
      at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(),                       \
      "joint_grouped_flat_vprefix_interval_sums_vpq", [&] {                                      \
        joint_grouped_flat_vprefix_interval_sums_vpq_kernel<int32_t, scalar_t, CODE_T, dims_per_block><<< \
            static_cast<unsigned int>(interval_blocks), threads, 0, stream>>>(                    \
            probs_flat.data_ptr<float>(),                                                         \
            values.data_ptr<scalar_t>(),                                                          \
            value_codebooks.data_ptr<float>(),                                                     \
            value_codes.data_ptr<CODE_T>(),                                                       \
            page_starts.data_ptr<int64_t>(),                                                       \
            ids_out.data_ptr<int32_t>(),                                                           \
            v_budgets.data_ptr<int64_t>(),                                                         \
            interval_sums.data_ptr<float>(),                                                       \
            groups,                                                                                \
            k_count,                                                                               \
            heads,                                                                                 \
            context_len,                                                                           \
            context_len,                                                                           \
            v_steps,                                                                               \
            dim,                                                                                   \
            pages,                                                                                 \
            page_size,                                                                              \
            value_subvecs,                                                                         \
            code_count);                                                                           \
      })
  if (value_codes.scalar_type() == torch::kUInt8) {
    DISPATCH_JOINT_VPQ_INTERVAL(uint8_t);
  } else if (value_codes.scalar_type() == torch::kInt) {
    DISPATCH_JOINT_VPQ_INTERVAL(int32_t);
  } else if (value_codes.scalar_type() == torch::kLong) {
    DISPATCH_JOINT_VPQ_INTERVAL(int64_t);
  } else {
    TORCH_CHECK(false, "value_codes must be uint8, int32, or int64");
  }
#undef DISPATCH_JOINT_VPQ_INTERVAL
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
