std::vector<torch::Tensor> joint_mixed_softmax_base_outputs_cuda(
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

  const auto heads = exact_scores.size(0);
  const auto context_len = exact_scores.size(1);
  const auto indexed_count = indexed_tokens.size(0);
  const auto k_count = k_take_counts.size(0);
  const auto max_rank_take = ranked_prefix_tokens.size(1);
  const auto dim = values.size(1);
  auto probs = torch::empty({k_count, heads, context_len}, exact_scores.options().dtype(torch::kFloat32));
  auto outputs = torch::empty({k_count, heads, dim}, exact_scores.options().dtype(torch::kFloat32));
  if (k_count == 0 || heads == 0 || context_len == 0 || dim == 0) {
    if (context_len == 0 && k_count > 0 && heads > 0 && dim > 0) {
      outputs.zero_();
    }
    return {probs, outputs};
  }

  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto byte_opts = exact_scores.options().dtype(torch::kUInt8);
  auto selected_mask = torch::zeros({k_count, heads, context_len}, byte_opts);
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

  auto fit_scale = torch::empty({k_count, heads}, exact_scores.options().dtype(torch::kFloat32));
  auto fit_bias = torch::empty({k_count, heads}, exact_scores.options().dtype(torch::kFloat32));
  const int64_t fit_blocks = k_count * heads;
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
      1.0f,
      calibrate);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto pos_opts = exact_scores.options().dtype(torch::kInt);
  auto token_to_indexed = torch::full({context_len}, -1, pos_opts);
  if (indexed_count > 0) {
    const int64_t blocks = (indexed_count + threads - 1) / threads;
    joint_indexed_token_to_pos_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        token_to_indexed.data_ptr<int>(),
        indexed_tokens.data_ptr<int64_t>(),
        indexed_count,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int64_t rows = k_count * heads;
  joint_mixed_softmax_rows_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      k_take_counts.data_ptr<int64_t>(),
      selected_mask.data_ptr<unsigned char>(),
      fit_scale.data_ptr<float>(),
      fit_bias.data_ptr<float>(),
      probs.data_ptr<float>(),
      rows,
      heads,
      indexed_count,
      max_rank_take,
      1.0f,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t out_blocks = rows * dim_tiles;
  joint_base_outputs_from_probs_kernel<dims_per_block><<<
      static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
      probs.data_ptr<float>(),
      values.data_ptr<float>(),
      outputs.data_ptr<float>(),
      rows,
      context_len,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {probs, outputs};
}

std::vector<torch::Tensor> joint_mixed_softmax_base_outputs_rankpos_cuda(
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

  const auto heads = exact_scores.size(0);
  const auto context_len = exact_scores.size(1);
  const auto indexed_count = indexed_tokens.size(0);
  const auto k_count = k_take_counts.size(0);
  const auto max_rank_take = ranked_prefix_tokens.size(1);
  const auto dim = values.size(1);
  auto probs = torch::empty({k_count, heads, context_len}, exact_scores.options().dtype(torch::kFloat32));
  auto outputs = torch::empty({k_count, heads, dim}, exact_scores.options().dtype(torch::kFloat32));
  if (k_count == 0 || heads == 0 || context_len == 0 || dim == 0) {
    if (context_len == 0 && k_count > 0 && heads > 0 && dim > 0) {
      outputs.zero_();
    }
    return {probs, outputs};
  }

  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
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
  const size_t fit_shared_bytes = static_cast<size_t>(threads) * 5 * sizeof(float);
  joint_affine_selected_fit_rankpos_kernel<<<static_cast<unsigned int>(fit_blocks), threads, fit_shared_bytes, stream>>>(
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

  const int64_t rows = k_count * heads;
  joint_mixed_softmax_rows_rankpos_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_mask.data_ptr<unsigned char>(),
      rank_pos.data_ptr<int>(),
      k_take_counts.data_ptr<int64_t>(),
      fit_scale.data_ptr<float>(),
      fit_bias.data_ptr<float>(),
      probs.data_ptr<float>(),
      rows,
      heads,
      indexed_count,
      max_rank_take,
      1.0f,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t out_blocks = rows * dim_tiles;
  joint_base_outputs_from_probs_kernel<dims_per_block><<<
      static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
      probs.data_ptr<float>(),
      values.data_ptr<float>(),
      outputs.data_ptr<float>(),
      rows,
      context_len,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {probs, outputs};
}

std::vector<torch::Tensor> joint_mixed_softmax_base_outputs_tokenfit_scaled_cuda(
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

  const auto heads = exact_scores.size(0);
  const auto context_len = exact_scores.size(1);
  const auto indexed_count = indexed_tokens.size(0);
  const auto k_count = k_take_counts.size(0);
  const auto max_rank_take = ranked_prefix_tokens.size(1);
  const auto dim = values.size(1);
  auto probs = torch::empty({k_count, heads, context_len}, exact_scores.options().dtype(torch::kFloat32));
  auto outputs = torch::empty({k_count, heads, dim}, exact_scores.options().dtype(torch::kFloat32));
  if (k_count == 0 || heads == 0 || context_len == 0 || dim == 0) {
    if (context_len == 0 && k_count > 0 && heads > 0 && dim > 0) {
      outputs.zero_();
    }
    return {probs, outputs};
  }

  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto byte_opts = exact_scores.options().dtype(torch::kUInt8);
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
  const size_t fit_shared_bytes = static_cast<size_t>(threads) * 5 * sizeof(float);
  joint_affine_selected_fit_tokenlist_kernel<<<static_cast<unsigned int>(fit_blocks), threads, fit_shared_bytes, stream>>>(
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
      base_count,
      max_rank_take,
      static_cast<float>(pq_scale),
      calibrate);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t rows = k_count * heads;
  joint_mixed_softmax_rows_rankpos_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_mask.data_ptr<unsigned char>(),
      rank_pos.data_ptr<int>(),
      k_take_counts.data_ptr<int64_t>(),
      fit_scale.data_ptr<float>(),
      fit_bias.data_ptr<float>(),
      probs.data_ptr<float>(),
      rows,
      heads,
      indexed_count,
      max_rank_take,
      static_cast<float>(pq_scale),
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t out_blocks = rows * dim_tiles;
  joint_base_outputs_from_probs_kernel<dims_per_block><<<
      static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
      probs.data_ptr<float>(),
      values.data_ptr<float>(),
      outputs.data_ptr<float>(),
      rows,
      context_len,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {probs, outputs};
}

