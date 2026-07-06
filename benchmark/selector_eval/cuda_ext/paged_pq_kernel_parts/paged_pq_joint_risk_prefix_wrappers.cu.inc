
torch::Tensor joint_grouped_accounting_sums_cuda(
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
  TORCH_CHECK(context_len >= 0, "context_len must be non-negative");
  TORCH_CHECK(head_dim > 0, "head_dim must be positive");
  TORCH_CHECK(key_bytes > 0, "key_bytes must be positive");
  TORCH_CHECK(value_bytes > 0, "value_bytes must be positive");
  TORCH_CHECK(actual_value_subvecs >= 0, "actual_value_subvecs must be non-negative");
  TORCH_CHECK(code_bytes >= 0, "code_bytes must be non-negative");
  const auto heads = final_indices.size(0);
  const auto k_count = selected_counts_by_ki.size(0);
  const auto v_count = v_budgets.size(0);
  TORCH_CHECK(k_count > 0, "selected_counts_by_ki must be non-empty");
  TORCH_CHECK(v_count > 0, "v_budgets must be non-empty");

  auto final_indices_c = final_indices.contiguous();
  auto selected_counts_c = selected_counts_by_ki.contiguous();
  auto v_budgets_c = v_budgets.contiguous();
  auto sums = torch::empty({11}, final_indices.options().dtype(torch::kFloat64));
  if (heads == 0) {
    sums.zero_();
    return sums;
  }

  const int threads = 128;
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_grouped_accounting_sums_kernel<<<1, threads, threads * 11 * sizeof(double), stream>>>(
      final_indices_c.data_ptr<int64_t>(),
      selected_counts_c.data_ptr<int64_t>(),
      v_budgets_c.data_ptr<int64_t>(),
      sums.data_ptr<double>(),
      heads,
      k_count,
      v_count,
      context_len,
      head_dim,
      key_bytes,
      value_bytes,
      selector_mb,
      v_pq_codebook_mb,
      metadata_mb,
      actual_value_subvecs,
      code_bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return sums;
}

void joint_grouped_accounting_accumulate_cuda(
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
  TORCH_CHECK(context_len >= 0, "context_len must be non-negative");
  TORCH_CHECK(head_dim > 0, "head_dim must be positive");
  TORCH_CHECK(key_bytes > 0, "key_bytes must be positive");
  TORCH_CHECK(value_bytes > 0, "value_bytes must be positive");
  TORCH_CHECK(actual_value_subvecs >= 0, "actual_value_subvecs must be non-negative");
  TORCH_CHECK(code_bytes >= 0, "code_bytes must be non-negative");
  const auto heads = final_indices.size(0);
  const auto k_count = selected_counts_by_ki.size(0);
  const auto v_count = v_budgets.size(0);
  TORCH_CHECK(k_count > 0, "selected_counts_by_ki must be non-empty");
  TORCH_CHECK(v_count > 0, "v_budgets must be non-empty");
  if (heads == 0) {
    return;
  }

  auto final_indices_c = final_indices.contiguous();
  auto selected_counts_c = selected_counts_by_ki.contiguous();
  auto v_budgets_c = v_budgets.contiguous();
  const int threads = 128;
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_grouped_accounting_accumulate_kernel<<<1, threads, threads * 11 * sizeof(double), stream>>>(
      final_indices_c.data_ptr<int64_t>(),
      selected_counts_c.data_ptr<int64_t>(),
      v_budgets_c.data_ptr<int64_t>(),
      accumulator.data_ptr<double>(),
      heads,
      k_count,
      v_count,
      context_len,
      head_dim,
      key_bytes,
      value_bytes,
      selector_mb,
      v_pq_codebook_mb,
      metadata_mb,
      actual_value_subvecs,
      code_bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor joint_rank_prefix_tokens_cuda(
    torch::Tensor scores,
    torch::Tensor indexed_tokens,
    int64_t max_take) {
  const auto rows = scores.size(0);
  const auto count = scores.size(1);
  max_take = std::max<int64_t>(0, std::min<int64_t>(max_take, count));
  auto out = torch::empty({rows, max_take}, indexed_tokens.options().dtype(torch::kLong));
  if (rows == 0 || count == 0 || max_take == 0) {
    return out;
  }

  auto float_opts = scores.options().dtype(torch::kFloat32);
  auto int_opts = scores.options().dtype(torch::kInt);
  auto long_opts = scores.options().dtype(torch::kLong);
  auto byte_opts = scores.options().dtype(torch::kUInt8);
  auto score_in = torch::empty({rows, count}, float_opts);
  auto score_out = torch::empty({rows, count}, float_opts);
  auto pos_in = torch::empty({rows, count}, int_opts);
  auto pos_out = torch::empty({rows, count}, int_opts);
  auto offsets = torch::empty({rows + 1}, long_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = rows * count;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_rank_prefix_sort_inputs_kernel<<<static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      scores.data_ptr<float>(),
      score_in.data_ptr<float>(),
      pos_in.data_ptr<int32_t>(),
      rows,
      count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t offset_blocks = ((rows + 1) + threads - 1) / threads;
  joint_segment_offsets_kernel<<<static_cast<unsigned int>(offset_blocks), threads, 0, stream>>>(
      offsets.data_ptr<int64_t>(),
      rows,
      count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t temp_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      temp_bytes,
      score_in.data_ptr<float>(),
      score_out.data_ptr<float>(),
      pos_in.data_ptr<int32_t>(),
      pos_out.data_ptr<int32_t>(),
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
      score_in.data_ptr<float>(),
      score_out.data_ptr<float>(),
      pos_in.data_ptr<int32_t>(),
      pos_out.data_ptr<int32_t>(),
      total,
      rows,
      offsets.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t gather_total = rows * max_take;
  const int64_t gather_blocks = (gather_total + threads - 1) / threads;
  joint_rank_prefix_gather_tokens_kernel<<<static_cast<unsigned int>(gather_blocks), threads, 0, stream>>>(
      pos_out.data_ptr<int32_t>(),
      indexed_tokens.data_ptr<int64_t>(),
      out.data_ptr<int64_t>(),
      rows,
      count,
      max_take);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

int64_t joint_rank_prefix_sort_temp_bytes_cuda(int64_t rows, int64_t count) {
  rows = std::max<int64_t>(0, rows);
  count = std::max<int64_t>(0, count);
  const int64_t total = rows * count;
  if (rows == 0 || count == 0 || total == 0) {
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

torch::Tensor joint_rank_prefix_tokens_workspace_cuda(
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
  const auto rows = scores.size(0);
  const auto count = scores.size(1);
  max_take = std::max<int64_t>(0, std::min<int64_t>(max_take, count));
  TORCH_CHECK(score_in.numel() >= rows * count, "score_in workspace too small");
  TORCH_CHECK(score_out.numel() >= rows * count, "score_out workspace too small");
  TORCH_CHECK(pos_in.numel() >= rows * count, "pos_in workspace too small");
  TORCH_CHECK(pos_out.numel() >= rows * count, "pos_out workspace too small");
  TORCH_CHECK(offsets.numel() >= rows + 1, "offsets workspace too small");
  TORCH_CHECK(out.numel() >= rows * max_take, "out workspace too small");
  auto out_view = out.reshape({rows, max_take});
  if (rows == 0 || count == 0 || max_take == 0) {
    return out_view;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = rows * count;
  const int64_t fill_blocks = (total + threads - 1) / threads;
  joint_rank_prefix_sort_inputs_kernel<<<static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>(
      scores.data_ptr<float>(),
      score_in.data_ptr<float>(),
      pos_in.data_ptr<int32_t>(),
      rows,
      count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t offset_blocks = ((rows + 1) + threads - 1) / threads;
  joint_segment_offsets_kernel<<<static_cast<unsigned int>(offset_blocks), threads, 0, stream>>>(
      offsets.data_ptr<int64_t>(),
      rows,
      count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t temp_bytes = static_cast<size_t>(temp.numel());
  TORCH_CHECK(temp_bytes > 0, "temp workspace too small");
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temp.data_ptr(),
      temp_bytes,
      score_in.data_ptr<float>(),
      score_out.data_ptr<float>(),
      pos_in.data_ptr<int32_t>(),
      pos_out.data_ptr<int32_t>(),
      total,
      rows,
      offsets.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t gather_total = rows * max_take;
  const int64_t gather_blocks = (gather_total + threads - 1) / threads;
  joint_rank_prefix_gather_tokens_kernel<<<static_cast<unsigned int>(gather_blocks), threads, 0, stream>>>(
      pos_out.data_ptr<int32_t>(),
      indexed_tokens.data_ptr<int64_t>(),
      out_view.data_ptr<int64_t>(),
      rows,
      count,
      max_take);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out_view;
}

torch::Tensor joint_budget_prefix_tokens_cuda(
    torch::Tensor scores,
    torch::Tensor indexed_tokens,
    torch::Tensor take_counts,
    int64_t max_take) {
  const auto rows = scores.size(0);
  const auto count = scores.size(1);
  const auto budget_count = take_counts.size(0);
  max_take = std::max<int64_t>(0, std::min<int64_t>(max_take, count));
  auto out = torch::empty({rows, max_take}, indexed_tokens.options().dtype(torch::kLong));
  if (rows == 0 || count == 0 || budget_count == 0 || max_take == 0) {
    return out;
  }

  auto int_opts = scores.options().dtype(torch::kInt);
  auto thresholds = torch::empty({rows, budget_count}, int_opts);
  auto greater_counts = torch::empty({rows, budget_count}, int_opts);
  auto tie_quotas = torch::empty({rows, budget_count}, int_opts);
  auto band_counters = torch::empty({rows, budget_count}, int_opts);
  auto tie_counters = torch::empty({rows, budget_count}, int_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      out.data_ptr<int64_t>(),
      0xff,
      static_cast<size_t>(out.numel()) * sizeof(int64_t),
      stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      band_counters.data_ptr<int32_t>(),
      0,
      static_cast<size_t>(band_counters.numel()) * sizeof(int32_t),
      stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      tie_counters.data_ptr<int32_t>(),
      0,
      static_cast<size_t>(tie_counters.numel()) * sizeof(int32_t),
      stream));

  const int threshold_threads = 256;
  const int64_t threshold_blocks = rows * budget_count;
  joint_budget_prefix_thresholds_kernel<<<
      static_cast<unsigned int>(threshold_blocks),
      threshold_threads,
      static_cast<size_t>(threshold_threads) * sizeof(int),
      stream>>>(
      scores.data_ptr<float>(),
      take_counts.data_ptr<int64_t>(),
      thresholds.data_ptr<int32_t>(),
      greater_counts.data_ptr<int32_t>(),
      tie_quotas.data_ptr<int32_t>(),
      rows,
      count,
      budget_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int fill_threads = 256;
  const int64_t total = rows * count;
  const int64_t fill_blocks = (total + fill_threads - 1) / fill_threads;
  joint_budget_prefix_fill_kernel<<<static_cast<unsigned int>(fill_blocks), fill_threads, 0, stream>>>(
      scores.data_ptr<float>(),
      indexed_tokens.data_ptr<int64_t>(),
      take_counts.data_ptr<int64_t>(),
      thresholds.data_ptr<int32_t>(),
      tie_quotas.data_ptr<int32_t>(),
      band_counters.data_ptr<int32_t>(),
      tie_counters.data_ptr<int32_t>(),
      out.data_ptr<int64_t>(),
      rows,
      count,
      budget_count,
      max_take);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

static torch::Tensor joint_mixed_score_grid_cuda_impl(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate,
    float pq_scale,
    bool fill_exact_first) {
  const auto heads = exact_scores.size(0);
  const auto context_len = exact_scores.size(1);
  const auto indexed_count = indexed_tokens.size(0);
  const auto k_count = k_take_counts.size(0);
  const auto max_rank_take = ranked_prefix_tokens.size(1);
  auto outputs = torch::empty({k_count, heads, context_len}, exact_scores.options().dtype(torch::kFloat32));
  if (k_count == 0 || heads == 0 || context_len == 0) {
    return outputs;
  }

  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t score_total = k_count * heads * context_len;
  if (fill_exact_first && calibrate) {
    const int64_t score_blocks = (score_total + threads - 1) / threads;
    joint_mixed_score_grid_fill_exact_kernel<<<static_cast<unsigned int>(score_blocks), threads, 0, stream>>>(
        exact_scores.data_ptr<float>(),
        outputs.data_ptr<float>(),
        k_count,
        heads,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int64_t base_count = base_tokens.size(0);
  if (!calibrate) {
    auto byte_opts = exact_scores.options().dtype(torch::kUInt8);
    auto int_opts = exact_scores.options().dtype(torch::kInt);
    auto token_to_indexed = torch::full({context_len}, -1, int_opts);
    auto base_mask = torch::zeros({context_len}, byte_opts);
    auto rank_pos = torch::full({heads, context_len}, -1, int_opts);
    if (indexed_count > 0) {
      const int64_t blocks = (indexed_count + threads - 1) / threads;
      joint_indexed_token_to_pos_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
          token_to_indexed.data_ptr<int>(),
          indexed_tokens.data_ptr<int64_t>(),
          indexed_count,
          context_len);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (base_count > 0) {
      const int64_t blocks = (base_count + threads - 1) / threads;
      joint_base_mask_set_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
          base_mask.data_ptr<unsigned char>(),
          base_tokens.data_ptr<int64_t>(),
          base_count,
          context_len);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (max_rank_take > 0) {
      const int64_t total = heads * max_rank_take;
      const int64_t blocks = (total + threads - 1) / threads;
      joint_rank_position_scatter_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
          rank_pos.data_ptr<int>(),
          ranked_prefix_tokens.data_ptr<int64_t>(),
          heads,
          max_rank_take,
          context_len);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    {
      const int64_t total = k_count * heads * context_len;
      const int64_t blocks = (total + threads - 1) / threads;
      joint_mixed_score_grid_fill_rankpos_nocalib_kernel<<<
          static_cast<unsigned int>(blocks), threads, 0, stream>>>(
          exact_scores.data_ptr<float>(),
          pq_logits.data_ptr<float>(),
          token_to_indexed.data_ptr<int>(),
          base_mask.data_ptr<unsigned char>(),
          rank_pos.data_ptr<int>(),
          k_take_counts.data_ptr<int64_t>(),
          outputs.data_ptr<float>(),
          k_count,
          heads,
          context_len,
          indexed_count,
          max_rank_take,
          pq_scale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return outputs;
  }

  auto byte_opts = exact_scores.options().dtype(torch::kUInt8);
  auto selected_mask = torch::zeros({k_count, heads, context_len}, byte_opts);
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

  auto fit_scale = torch::empty({k_count, heads}, exact_scores.options().dtype(torch::kFloat32));
  auto fit_bias = torch::empty({k_count, heads}, exact_scores.options().dtype(torch::kFloat32));
  const int64_t fit_blocks = k_count * heads;
  const size_t shared_bytes = static_cast<size_t>(threads) * 5 * sizeof(float);
  joint_affine_selected_fit_kernel<<<static_cast<unsigned int>(fit_blocks), threads, shared_bytes, stream>>>(
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
      pq_scale,
      calibrate);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (indexed_count > 0) {
    const int64_t total = k_count * heads * indexed_count;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_mixed_score_grid_apply_indexed_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        exact_scores.data_ptr<float>(),
        pq_logits.data_ptr<float>(),
        indexed_tokens.data_ptr<int64_t>(),
        k_take_counts.data_ptr<int64_t>(),
        fit_scale.data_ptr<float>(),
        fit_bias.data_ptr<float>(),
        outputs.data_ptr<float>(),
        k_count,
        heads,
        indexed_count,
        max_rank_take,
        pq_scale,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (base_count > 0) {
    const int64_t total = k_count * heads * base_count;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_mixed_score_grid_apply_base_exact_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        exact_scores.data_ptr<float>(),
        base_tokens.data_ptr<int64_t>(),
        outputs.data_ptr<float>(),
        k_count,
        heads,
        context_len,
        base_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (max_rank_take > 0) {
    const int64_t total = k_count * heads * max_rank_take;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_mixed_score_grid_apply_ranked_exact_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        exact_scores.data_ptr<float>(),
        ranked_prefix_tokens.data_ptr<int64_t>(),
        k_take_counts.data_ptr<int64_t>(),
        outputs.data_ptr<float>(),
        k_count,
        heads,
        max_rank_take,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return outputs;
}
