torch::Tensor joint_mixed_score_grid_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate) {
  return joint_mixed_score_grid_cuda_impl(
      exact_scores,
      pq_logits,
      y_indexed,
      indexed_tokens,
      base_tokens,
      ranked_prefix_tokens,
      k_take_counts,
      calibrate,
      1.0f,
      true);
}

torch::Tensor joint_mixed_score_grid_workspace_cuda(
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
  const auto heads = exact_scores.size(0);
  const auto context_len = exact_scores.size(1);
  const auto indexed_count = indexed_tokens.size(0);
  const auto k_count = k_take_counts.size(0);
  const auto max_rank_take = ranked_prefix_tokens.size(1);
  TORCH_CHECK(output.dim() == 3, "output shape must be [k_count, heads, context_len]");
  TORCH_CHECK(selected_mask.dim() == 3, "selected_mask shape must be [k_count, heads, context_len]");
  TORCH_CHECK(fit_scale.dim() == 2, "fit_scale shape must be [k_count, heads]");
  TORCH_CHECK(fit_bias.dim() == 2, "fit_bias shape must be [k_count, heads]");
  TORCH_CHECK(output.size(0) == k_count && output.size(1) == heads && output.size(2) == context_len,
              "output shape mismatch");
  TORCH_CHECK(selected_mask.size(0) == k_count && selected_mask.size(1) == heads && selected_mask.size(2) == context_len,
              "selected_mask shape mismatch");
  TORCH_CHECK(fit_scale.size(0) == k_count && fit_scale.size(1) == heads, "fit_scale shape mismatch");
  TORCH_CHECK(fit_bias.size(0) == k_count && fit_bias.size(1) == heads, "fit_bias shape mismatch");
  if (k_count == 0 || heads == 0 || context_len == 0) {
    return output;
  }

  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
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
        output.data_ptr<float>(),
        k_count,
        heads,
        context_len,
        indexed_count,
        max_rank_take,
        1.0f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  const int64_t score_total = k_count * heads * context_len;
  const int64_t score_blocks = (score_total + threads - 1) / threads;
  joint_mixed_score_grid_fill_exact_kernel<<<static_cast<unsigned int>(score_blocks), threads, 0, stream>>>(
      exact_scores.data_ptr<float>(),
      output.data_ptr<float>(),
      k_count,
      heads,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  C10_CUDA_CHECK(cudaMemsetAsync(
      selected_mask.data_ptr<unsigned char>(),
      0,
      static_cast<size_t>(selected_mask.numel()),
      stream));

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
      1.0f,
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
        output.data_ptr<float>(),
        k_count,
        heads,
        indexed_count,
        max_rank_take,
        1.0f,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (base_count > 0) {
    const int64_t total = k_count * heads * base_count;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_mixed_score_grid_apply_base_exact_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        exact_scores.data_ptr<float>(),
        base_tokens.data_ptr<int64_t>(),
        output.data_ptr<float>(),
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
        output.data_ptr<float>(),
        k_count,
        heads,
        max_rank_take,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

torch::Tensor joint_mixed_score_grid_rankpos_nocalib_workspace_cuda(
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
  const auto heads = exact_scores.size(0);
  const auto context_len = exact_scores.size(1);
  const auto indexed_count = indexed_tokens.size(0);
  const auto k_count = k_take_counts.size(0);
  const auto max_rank_take = ranked_prefix_tokens.size(1);
  TORCH_CHECK(output.dim() == 3, "output shape must be [k_count, heads, context_len]");
  TORCH_CHECK(token_to_indexed.dim() == 1, "token_to_indexed shape must be [context_len]");
  TORCH_CHECK(base_mask.dim() == 1, "base_mask shape must be [context_len]");
  TORCH_CHECK(rank_pos.dim() == 2, "rank_pos shape must be [heads, context_len]");
  TORCH_CHECK(output.size(0) == k_count && output.size(1) == heads && output.size(2) == context_len,
              "output shape mismatch");
  TORCH_CHECK(token_to_indexed.size(0) == context_len, "token_to_indexed context mismatch");
  TORCH_CHECK(base_mask.size(0) == context_len, "base_mask context mismatch");
  TORCH_CHECK(rank_pos.size(0) == heads && rank_pos.size(1) == context_len, "rank_pos shape mismatch");
  if (k_count == 0 || heads == 0 || context_len == 0) {
    return output;
  }

  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      token_to_indexed.data_ptr<int>(),
      0xff,
      static_cast<size_t>(token_to_indexed.numel()) * sizeof(int),
      stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      base_mask.data_ptr<unsigned char>(),
      0,
      static_cast<size_t>(base_mask.numel()) * sizeof(unsigned char),
      stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      rank_pos.data_ptr<int>(),
      0xff,
      static_cast<size_t>(rank_pos.numel()) * sizeof(int),
      stream));

  const int64_t base_count = base_tokens.size(0);
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
      output.data_ptr<float>(),
      k_count,
      heads,
      context_len,
      indexed_count,
      max_rank_take,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor joint_mixed_score_grid_nocalib_scatter_workspace_cuda(
    torch::Tensor output,
    torch::Tensor token_to_indexed,
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    double pq_scale) {
  const auto heads = exact_scores.size(0);
  const auto context_len = exact_scores.size(1);
  const auto indexed_count = indexed_tokens.size(0);
  const auto k_count = k_take_counts.size(0);
  const auto max_rank_take = ranked_prefix_tokens.size(1);
  TORCH_CHECK(output.dim() == 3, "output shape must be [k_count, heads, context_len]");
  TORCH_CHECK(token_to_indexed.dim() == 1, "token_to_indexed shape must be [context_len]");
  TORCH_CHECK(output.size(0) == k_count && output.size(1) == heads && output.size(2) == context_len,
              "output shape mismatch");
  TORCH_CHECK(token_to_indexed.size(0) == context_len, "token_to_indexed context mismatch");
  if (k_count == 0 || heads == 0 || context_len == 0) {
    return output;
  }

  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      token_to_indexed.data_ptr<int>(),
      0xff,
      static_cast<size_t>(token_to_indexed.numel()) * sizeof(int),
      stream));

  const int64_t base_count = base_tokens.size(0);
  if (indexed_count > 0) {
    const int64_t blocks = (indexed_count + threads - 1) / threads;
    joint_indexed_token_to_pos_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        token_to_indexed.data_ptr<int>(),
        indexed_tokens.data_ptr<int64_t>(),
        indexed_count,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int64_t total = k_count * heads * context_len;
  const int64_t blocks = (total + threads - 1) / threads;
  joint_mixed_score_grid_fill_pq_or_full_exact_kernel<<<
      static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      k_take_counts.data_ptr<int64_t>(),
      output.data_ptr<float>(),
      k_count,
      heads,
      context_len,
      indexed_count,
      max_rank_take,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (base_count > 0) {
    const int64_t base_total = k_count * heads * base_count;
    const int64_t base_blocks = (base_total + threads - 1) / threads;
    joint_mixed_score_grid_apply_base_exact_kernel<<<static_cast<unsigned int>(base_blocks), threads, 0, stream>>>(
        exact_scores.data_ptr<float>(),
        base_tokens.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        k_count,
        heads,
        context_len,
        base_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (max_rank_take > 0) {
    const int64_t rank_total = k_count * heads * max_rank_take;
    const int64_t rank_blocks = (rank_total + threads - 1) / threads;
    joint_mixed_score_grid_apply_ranked_exact_kernel<<<static_cast<unsigned int>(rank_blocks), threads, 0, stream>>>(
        exact_scores.data_ptr<float>(),
        ranked_prefix_tokens.data_ptr<int64_t>(),
        k_take_counts.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        k_count,
        heads,
        max_rank_take,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

torch::Tensor joint_mixed_score_grid_scaled_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate,
    double pq_scale) {
  return joint_mixed_score_grid_cuda_impl(
      exact_scores,
      pq_logits,
      y_indexed,
      indexed_tokens,
      base_tokens,
      ranked_prefix_tokens,
      k_take_counts,
      calibrate,
      static_cast<float>(pq_scale),
      true);
}

torch::Tensor joint_mixed_score_grid_rankpos_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate) {
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
  const int64_t score_blocks = (score_total + threads - 1) / threads;
  joint_mixed_score_grid_fill_exact_kernel<<<static_cast<unsigned int>(score_blocks), threads, 0, stream>>>(
      exact_scores.data_ptr<float>(),
      outputs.data_ptr<float>(),
      k_count,
      heads,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto byte_opts = exact_scores.options().dtype(torch::kUInt8);
  auto int_opts = exact_scores.options().dtype(torch::kInt);
  auto base_mask = torch::zeros({context_len}, byte_opts);
  auto rank_pos = torch::full({heads, context_len}, -1, int_opts);
  const int64_t base_count = base_tokens.size(0);
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

  auto fit_scale = torch::empty({k_count, heads}, exact_scores.options().dtype(torch::kFloat32));
  auto fit_bias = torch::empty({k_count, heads}, exact_scores.options().dtype(torch::kFloat32));
  const int64_t fit_blocks = k_count * heads;
  const size_t shared_bytes = static_cast<size_t>(threads) * 5 * sizeof(float);
  joint_affine_selected_fit_rankpos_kernel<<<static_cast<unsigned int>(fit_blocks), threads, shared_bytes, stream>>>(
      pq_logits.data_ptr<float>(),
      y_indexed.data_ptr<float>(),
      indexed_tokens.data_ptr<int64_t>(),
      base_mask.data_ptr<unsigned char>(),
      rank_pos.data_ptr<int>(),
      k_take_counts.data_ptr<int64_t>(),
      fit_scale.data_ptr<float>(),
      fit_bias.data_ptr<float>(),
      k_count,
      heads,
      indexed_count,
      context_len,
      1.0f,
      calibrate);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (indexed_count > 0) {
    const int64_t total = k_count * heads * indexed_count;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_mixed_score_grid_apply_indexed_rankpos_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        exact_scores.data_ptr<float>(),
        pq_logits.data_ptr<float>(),
        indexed_tokens.data_ptr<int64_t>(),
        base_mask.data_ptr<unsigned char>(),
        rank_pos.data_ptr<int>(),
        k_take_counts.data_ptr<int64_t>(),
        fit_scale.data_ptr<float>(),
        fit_bias.data_ptr<float>(),
        outputs.data_ptr<float>(),
        k_count,
        heads,
        indexed_count,
        max_rank_take,
        1.0f,
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

torch::Tensor joint_mixed_score_grid_no_exact_fill_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate) {
  return joint_mixed_score_grid_cuda_impl(
      exact_scores,
      pq_logits,
      y_indexed,
      indexed_tokens,
      base_tokens,
      ranked_prefix_tokens,
      k_take_counts,
      calibrate,
      1.0f,
      false);
}

torch::Tensor joint_mixed_score_grid_no_exact_fill_scaled_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate,
    double pq_scale) {
  return joint_mixed_score_grid_cuda_impl(
      exact_scores,
      pq_logits,
      y_indexed,
      indexed_tokens,
      base_tokens,
      ranked_prefix_tokens,
      k_take_counts,
      calibrate,
      static_cast<float>(pq_scale),
      false);
}

static torch::Tensor joint_mixed_score_grid_tokenfit_cuda_impl(
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
  if (fill_exact_first) {
    const int64_t score_total = k_count * heads * context_len;
    const int64_t score_blocks = (score_total + threads - 1) / threads;
    joint_mixed_score_grid_fill_exact_kernel<<<static_cast<unsigned int>(score_blocks), threads, 0, stream>>>(
        exact_scores.data_ptr<float>(),
        outputs.data_ptr<float>(),
        k_count,
        heads,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto int_opts = exact_scores.options().dtype(torch::kInt);
  auto token_to_indexed = torch::full({context_len}, -1, int_opts);
  if (indexed_count > 0) {
    const int64_t blocks = (indexed_count + threads - 1) / threads;
    joint_indexed_token_to_pos_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        token_to_indexed.data_ptr<int>(),
        indexed_tokens.data_ptr<int64_t>(),
        indexed_count,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto fit_scale = torch::empty({k_count, heads}, exact_scores.options().dtype(torch::kFloat32));
  auto fit_bias = torch::empty({k_count, heads}, exact_scores.options().dtype(torch::kFloat32));
  const int64_t fit_blocks = k_count * heads;
  const size_t shared_bytes = static_cast<size_t>(threads) * 5 * sizeof(float);
  joint_affine_selected_fit_tokenlist_kernel<<<static_cast<unsigned int>(fit_blocks), threads, shared_bytes, stream>>>(
      pq_logits.data_ptr<float>(),
      y_indexed.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_tokens.data_ptr<int64_t>(),
      ranked_prefix_tokens.data_ptr<int64_t>(),
      k_take_counts.data_ptr<int64_t>(),
      fit_scale.data_ptr<float>(),
      fit_bias.data_ptr<float>(),
      k_count,
      heads,
      indexed_count,
      context_len,
      base_tokens.size(0),
      max_rank_take,
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

  const int64_t base_count = base_tokens.size(0);
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

torch::Tensor joint_mixed_score_grid_tokenfit_scaled_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate,
    double pq_scale) {
  return joint_mixed_score_grid_tokenfit_cuda_impl(
      exact_scores,
      pq_logits,
      y_indexed,
      indexed_tokens,
      base_tokens,
      ranked_prefix_tokens,
      k_take_counts,
      calibrate,
      static_cast<float>(pq_scale),
      true);
}

torch::Tensor joint_mixed_score_grid_no_exact_fill_tokenfit_scaled_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor y_indexed,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    bool calibrate,
    double pq_scale) {
  return joint_mixed_score_grid_tokenfit_cuda_impl(
      exact_scores,
      pq_logits,
      y_indexed,
      indexed_tokens,
      base_tokens,
      ranked_prefix_tokens,
      k_take_counts,
      calibrate,
      static_cast<float>(pq_scale),
      false);
}

torch::Tensor joint_mixed_score_grid_sparse_exact_tokenfit_scaled_cuda(
    torch::Tensor pq_logits,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor base_logits,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor ranked_logits,
    torch::Tensor k_take_counts,
    int64_t context_len,
    bool calibrate,
    double pq_scale) {
  const auto heads = pq_logits.size(0);
  const auto indexed_count = indexed_tokens.size(0);
  const auto base_count = base_tokens.size(0);
  const auto k_count = k_take_counts.size(0);
  const auto max_rank_take = ranked_prefix_tokens.size(1);
  auto outputs = torch::empty({k_count, heads, context_len}, pq_logits.options().dtype(torch::kFloat32));
  if (k_count == 0 || heads == 0 || context_len == 0) {
    return outputs;
  }

  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto int_opts = pq_logits.options().dtype(torch::kInt);
  auto token_to_indexed = torch::full({context_len}, -1, int_opts);
  if (indexed_count > 0) {
    const int64_t blocks = (indexed_count + threads - 1) / threads;
    joint_indexed_token_to_pos_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        token_to_indexed.data_ptr<int>(),
        indexed_tokens.data_ptr<int64_t>(),
        indexed_count,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto fit_scale = torch::empty({k_count, heads}, pq_logits.options().dtype(torch::kFloat32));
  auto fit_bias = torch::empty_like(fit_scale);
  const int64_t fit_blocks = k_count * heads;
  const size_t shared_bytes = static_cast<size_t>(threads) * 5 * sizeof(float);
  joint_affine_selected_fit_sparse_tokenlist_kernel<<<static_cast<unsigned int>(fit_blocks), threads, shared_bytes, stream>>>(
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_tokens.data_ptr<int64_t>(),
      base_logits.data_ptr<float>(),
      ranked_prefix_tokens.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      k_take_counts.data_ptr<int64_t>(),
      fit_scale.data_ptr<float>(),
      fit_bias.data_ptr<float>(),
      k_count,
      heads,
      indexed_count,
      context_len,
      base_count,
      max_rank_take,
      static_cast<float>(pq_scale),
      calibrate);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (indexed_count > 0) {
    const int64_t total = k_count * heads * indexed_count;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_mixed_score_grid_apply_indexed_calibrated_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        pq_logits.data_ptr<float>(),
        indexed_tokens.data_ptr<int64_t>(),
        fit_scale.data_ptr<float>(),
        fit_bias.data_ptr<float>(),
        outputs.data_ptr<float>(),
        k_count,
        heads,
        indexed_count,
        static_cast<float>(pq_scale),
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (base_count > 0) {
    const int64_t total = k_count * heads * base_count;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_mixed_score_grid_apply_base_logits_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        base_tokens.data_ptr<int64_t>(),
        base_logits.data_ptr<float>(),
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
    joint_mixed_score_grid_apply_ranked_logits_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        ranked_prefix_tokens.data_ptr<int64_t>(),
        ranked_logits.data_ptr<float>(),
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

torch::Tensor joint_sparse_exact_score_table_cuda(
    torch::Tensor base_tokens,
    torch::Tensor base_logits,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_logits,
    int64_t context_len) {
  const auto heads = base_logits.size(0);
  const auto base_count = base_tokens.size(0);
  const auto max_rank_take = ranked_tokens.size(1);
  auto out = torch::empty({heads, context_len}, base_logits.options().dtype(torch::kFloat32));
  if (heads == 0 || context_len == 0) {
    return out;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<float>(), 0, static_cast<size_t>(out.numel()) * sizeof(float), stream));
  const int threads = 256;
  if (base_count > 0) {
    const int64_t total = heads * base_count;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_sparse_exact_score_table_base_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        base_tokens.data_ptr<int64_t>(),
        base_logits.data_ptr<float>(),
        out.data_ptr<float>(),
        heads,
        base_count,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (max_rank_take > 0) {
    const int64_t total = heads * max_rank_take;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_sparse_exact_score_table_ranked_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        ranked_tokens.data_ptr<int64_t>(),
        ranked_logits.data_ptr<float>(),
        out.data_ptr<float>(),
        heads,
        max_rank_take,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
