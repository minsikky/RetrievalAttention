template <typename order_t>
void launch_joint_vprefix_precision_tokpar(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_hi,
    torch::Tensor residual_lo,
    torch::Tensor commit_mask,
    torch::Tensor exact_order,
    torch::Tensor v_budgets,
    torch::Tensor v_hi_counts,
    torch::Tensor outputs,
    torch::Tensor lo_reads) {
  const auto k_count = base_outputs.size(0);
  const auto heads = base_outputs.size(1);
  const auto dim = base_outputs.size(2);
  const auto context_len = probs.size(2);
  const auto max_exact = exact_order.numel() / (k_count * heads);
  const auto v_steps = v_budgets.size(0);
  const int64_t rows = k_count * heads;
  const int64_t chunks =
      (max_exact + kJointVPrefixTokenChunk - 1) / kJointVPrefixTokenChunk;
  auto partial_opts = base_outputs.options().dtype(torch::kFloat32);
  auto int_opts = base_outputs.options().dtype(torch::kInt);
  auto hi_partials = torch::empty({chunks, rows, dim}, partial_opts);
  auto lo_partials = torch::empty({chunks, rows, dim}, partial_opts);
  auto commit_partials = torch::empty({chunks, rows}, int_opts);

  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t partial_total = chunks * rows * dim;
  if (partial_total > 0) {
    const int64_t partial_blocks = (partial_total + threads - 1) / threads;
    joint_vprefix_precision_token_partials_kernel<order_t><<<
        static_cast<unsigned int>(partial_blocks), threads, 0, stream>>>(
        probs.data_ptr<float>(),
        residual_hi.data_ptr<float>(),
        residual_lo.data_ptr<float>(),
        commit_mask.data_ptr<bool>(),
        exact_order.data_ptr<order_t>(),
        hi_partials.data_ptr<float>(),
        lo_partials.data_ptr<float>(),
        commit_partials.data_ptr<int32_t>(),
        rows,
        context_len,
        max_exact,
        dim,
        chunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int64_t output_total = rows * dim;
  const int64_t output_blocks = (output_total + threads - 1) / threads;
  joint_vprefix_precision_outputs_tokpar_kernel<order_t><<<
      static_cast<unsigned int>(output_blocks), threads, 0, stream>>>(
      base_outputs.data_ptr<float>(),
      probs.data_ptr<float>(),
      residual_hi.data_ptr<float>(),
      residual_lo.data_ptr<float>(),
      commit_mask.data_ptr<bool>(),
      exact_order.data_ptr<order_t>(),
      v_budgets.data_ptr<int64_t>(),
      v_hi_counts.data_ptr<int64_t>(),
      hi_partials.data_ptr<float>(),
      lo_partials.data_ptr<float>(),
      commit_partials.data_ptr<int32_t>(),
      outputs.data_ptr<float>(),
      lo_reads.data_ptr<int32_t>(),
      k_count,
      heads,
      context_len,
      max_exact,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor joint_vprefix_outputs_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual,
    torch::Tensor exact_order,
    torch::Tensor v_budgets) {
  const auto k_count = base_outputs.size(0);
  const auto heads = base_outputs.size(1);
  const auto dim = base_outputs.size(2);
  const auto context_len = probs.size(2);
  const auto max_exact = exact_order.size(2);
  const auto v_steps = v_budgets.size(0);
  auto outputs = torch::empty({k_count, v_steps, heads, dim}, base_outputs.options().dtype(torch::kFloat32));
  if (k_count == 0 || heads == 0 || dim == 0 || v_steps == 0) {
    return outputs;
  }
  const int threads = 256;
  const int64_t total = k_count * heads * dim;
  const int64_t blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_vprefix_outputs_kernel<int64_t><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      base_outputs.data_ptr<float>(),
      probs.data_ptr<float>(),
      residual.data_ptr<float>(),
      exact_order.data_ptr<int64_t>(),
      v_budgets.data_ptr<int64_t>(),
      outputs.data_ptr<float>(),
      k_count,
      heads,
      context_len,
      max_exact,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

std::vector<torch::Tensor> joint_vprefix_outputs_precision_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_hi,
    torch::Tensor residual_lo,
    torch::Tensor commit_mask,
    torch::Tensor exact_order,
    torch::Tensor v_budgets,
    torch::Tensor v_hi_counts) {
  const auto k_count = base_outputs.size(0);
  const auto heads = base_outputs.size(1);
  const auto dim = base_outputs.size(2);
  const auto context_len = probs.size(2);
  const auto max_exact = exact_order.size(2);
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
  const int threads = 256;
  const int64_t total = k_count * heads * dim;
  const int64_t blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_vprefix_precision_outputs_kernel<int64_t><<<
      static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      base_outputs.data_ptr<float>(),
      probs.data_ptr<float>(),
      residual_hi.data_ptr<float>(),
      residual_lo.data_ptr<float>(),
      commit_mask.data_ptr<bool>(),
      exact_order.data_ptr<int64_t>(),
      v_budgets.data_ptr<int64_t>(),
      v_hi_counts.data_ptr<int64_t>(),
      outputs.data_ptr<float>(),
      lo_reads.data_ptr<int32_t>(),
      k_count,
      heads,
      context_len,
      max_exact,
      v_steps,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {outputs, lo_reads};
}

std::vector<torch::Tensor> joint_vprefix_outputs_precision_tokpar_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_hi,
    torch::Tensor residual_lo,
    torch::Tensor commit_mask,
    torch::Tensor exact_order,
    torch::Tensor v_budgets,
    torch::Tensor v_hi_counts) {
  const auto k_count = base_outputs.size(0);
  const auto heads = base_outputs.size(1);
  const auto dim = base_outputs.size(2);
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
  launch_joint_vprefix_precision_tokpar<int64_t>(
      base_outputs,
      probs,
      residual_hi,
      residual_lo,
      commit_mask,
      exact_order,
      v_budgets,
      v_hi_counts,
      outputs,
      lo_reads);
  return {outputs, lo_reads};
}

torch::Tensor joint_vpq_base_outputs_from_probs_cuda(
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
  TORCH_CHECK(probs.dim() == 3, "probs must have shape [k_count, heads, context_len]");
  TORCH_CHECK(values.dim() == 2, "values must have shape [context_len, dim]");
  TORCH_CHECK(value_codebooks.dim() == 4, "value_codebooks must have shape [pages, subvecs, codes, subdim]");
  TORCH_CHECK(value_codes.dim() == 3, "value_codes must have shape [pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts must be 1-D");
  TORCH_CHECK(fallback_tokens.dim() == 1, "fallback_tokens must be 1-D");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(value_codebooks.scalar_type() == torch::kFloat32, "value_codebooks must be float32");
  TORCH_CHECK(page_starts.scalar_type() == torch::kLong, "page_starts must be int64");
  TORCH_CHECK(fallback_tokens.scalar_type() == torch::kLong, "fallback_tokens must be int64");

  const auto k_count = probs.size(0);
  const auto heads = probs.size(1);
  const auto context_len = probs.size(2);
  const auto dim = values.size(1);
  const auto pages = value_codebooks.size(0);
  const auto value_subvecs = value_codebooks.size(1);
  const auto code_count = value_codebooks.size(2);
  const auto subdim = value_codebooks.size(3);
  const auto page_size = value_codes.size(1);
  const auto code_subvecs = value_codes.size(2);
  const auto fallback_count = fallback_tokens.size(0);
  TORCH_CHECK(values.size(0) >= context_len, "values length must cover probs context_len");
  TORCH_CHECK(value_subvecs == 1, "joint_vpq_base_outputs_from_probs currently supports VALUE_SUBVECS=1 only");
  TORCH_CHECK(code_subvecs >= 1, "value_codes must have at least one subvector");
  TORCH_CHECK(subdim == dim, "single-subvector V-PQ codebook dim must match value dim");
  TORCH_CHECK(value_codes.size(0) == pages, "value_codes pages must match value_codebooks pages");
  TORCH_CHECK(page_starts.size(0) == pages, "page_starts pages must match value_codebooks pages");

  auto probs_c = probs.contiguous();
  auto values_c = values.contiguous();
  auto codebooks_c = value_codebooks.contiguous();
  auto codes_c = value_codes.contiguous();
  auto page_starts_c = page_starts.contiguous();
  auto fallback_tokens_c = fallback_tokens.contiguous();
  auto outputs = torch::empty({k_count, heads, dim}, probs.options().dtype(torch::kFloat32));
  if (k_count == 0 || heads == 0 || dim == 0) {
    return outputs;
  }

  const int64_t rows = k_count * heads;
  auto weights = torch::zeros({rows, pages, code_count}, probs.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = rows * context_len;
  const int64_t weight_blocks = (total + threads - 1) / threads;
  if (total > 0 && pages > 0 && code_count > 0) {
    if (value_codes.scalar_type() == torch::kUInt8) {
      joint_vpq_code_weight_kernel<uint8_t><<<static_cast<unsigned int>(weight_blocks), threads, 0, stream>>>(
          probs_c.data_ptr<float>(),
          codes_c.data_ptr<uint8_t>(),
          page_starts_c.data_ptr<int64_t>(),
          weights.data_ptr<float>(),
          rows,
          context_len,
          pages,
          page_size,
          code_subvecs,
          code_count);
    } else if (value_codes.scalar_type() == torch::kLong) {
      joint_vpq_code_weight_kernel<int64_t><<<static_cast<unsigned int>(weight_blocks), threads, 0, stream>>>(
          probs_c.data_ptr<float>(),
          codes_c.data_ptr<int64_t>(),
          page_starts_c.data_ptr<int64_t>(),
          weights.data_ptr<float>(),
          rows,
          context_len,
          pages,
          page_size,
          code_subvecs,
          code_count);
    } else if (value_codes.scalar_type() == torch::kInt) {
      joint_vpq_code_weight_kernel<int32_t><<<static_cast<unsigned int>(weight_blocks), threads, 0, stream>>>(
          probs_c.data_ptr<float>(),
          codes_c.data_ptr<int32_t>(),
          page_starts_c.data_ptr<int64_t>(),
          weights.data_ptr<float>(),
          rows,
          context_len,
          pages,
          page_size,
          code_subvecs,
          code_count);
    } else {
      TORCH_CHECK(false, "value_codes must be uint8, int32, or int64");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int64_t out_total = rows * dim;
  const int64_t out_blocks = (out_total + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "joint_vpq_base_values", [&] {
    joint_vpq_base_from_code_weights_kernel<scalar_t><<<static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
        probs_c.data_ptr<float>(),
        values_c.data_ptr<scalar_t>(),
        codebooks_c.data_ptr<float>(),
        weights.data_ptr<float>(),
        fallback_tokens_c.data_ptr<int64_t>(),
        outputs.data_ptr<float>(),
        rows,
        context_len,
        dim,
        pages,
        code_count,
        fallback_count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

std::vector<torch::Tensor> joint_softmax_base_outputs_cuda(
    torch::Tensor score_grid,
    torch::Tensor values) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(score_grid.dim() == 3, "score_grid shape must be [k_count, heads, context_len]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  const auto k_count = score_grid.size(0);
  const auto heads = score_grid.size(1);
  const auto context_len = score_grid.size(2);
  const auto dim = values.size(1);
  TORCH_CHECK(values.size(0) >= context_len, "values length must cover score_grid context_len");

  auto scores_c = score_grid.contiguous();
  auto values_c = values.contiguous();
  auto probs = torch::empty_like(scores_c, scores_c.options().dtype(torch::kFloat32));
  auto outputs = torch::empty({k_count, heads, dim}, scores_c.options().dtype(torch::kFloat32));
  if (k_count == 0 || heads == 0 || context_len == 0 || dim == 0) {
    if (context_len == 0 && k_count > 0 && heads > 0 && dim > 0) {
      outputs.zero_();
    }
    return {probs, outputs};
  }

  const int threads = 256;
  const int64_t rows = k_count * heads;
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_softmax_rows_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      scores_c.data_ptr<float>(),
      probs.data_ptr<float>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t out_blocks = rows * dim_tiles;
  joint_base_outputs_from_probs_kernel<dims_per_block><<<
      static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
      probs.data_ptr<float>(),
      values_c.data_ptr<float>(),
      outputs.data_ptr<float>(),
      rows,
      context_len,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {probs, outputs};
}

std::vector<torch::Tensor> joint_softmax_base_outputs_workspace_cuda(
    torch::Tensor probs,
    torch::Tensor outputs,
    torch::Tensor score_grid,
    torch::Tensor values) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs workspace must be CUDA");
  TORCH_CHECK(outputs.is_cuda(), "outputs workspace must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs workspace must be float32");
  TORCH_CHECK(outputs.scalar_type() == torch::kFloat32, "outputs workspace must be float32");
  TORCH_CHECK(score_grid.dim() == 3, "score_grid shape must be [k_count, heads, context_len]");
  TORCH_CHECK(probs.dim() == 3, "probs workspace shape must be [k_count, heads, context_len]");
  TORCH_CHECK(outputs.dim() == 3, "outputs workspace shape must be [k_count, heads, dim]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  const auto k_count = score_grid.size(0);
  const auto heads = score_grid.size(1);
  const auto context_len = score_grid.size(2);
  const auto dim = values.size(1);
  TORCH_CHECK(probs.size(0) == k_count && probs.size(1) == heads && probs.size(2) == context_len,
              "probs workspace shape mismatch");
  TORCH_CHECK(outputs.size(0) == k_count && outputs.size(1) == heads && outputs.size(2) == dim,
              "outputs workspace shape mismatch");
  TORCH_CHECK(values.size(0) >= context_len, "values length must cover score_grid context_len");
  TORCH_CHECK(score_grid.is_contiguous(), "score_grid must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(probs.is_contiguous(), "probs workspace must be contiguous");
  TORCH_CHECK(outputs.is_contiguous(), "outputs workspace must be contiguous");
  if (k_count == 0 || heads == 0 || context_len == 0 || dim == 0) {
    if (context_len == 0 && k_count > 0 && heads > 0 && dim > 0) {
      outputs.zero_();
    }
    return {probs, outputs};
  }

  const int threads = 256;
  const int64_t rows = k_count * heads;
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_softmax_rows_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      score_grid.data_ptr<float>(),
      probs.data_ptr<float>(),
      rows,
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

std::vector<torch::Tensor> joint_softmax_base_outputs_strided_workspace_cuda(
    torch::Tensor probs,
    torch::Tensor outputs,
    torch::Tensor score_grid,
    torch::Tensor values) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs workspace must be CUDA");
  TORCH_CHECK(outputs.is_cuda(), "outputs workspace must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs workspace must be float32");
  TORCH_CHECK(outputs.scalar_type() == torch::kFloat32, "outputs workspace must be float32");
  TORCH_CHECK(score_grid.dim() == 3, "score_grid shape must be [k_count, heads, context_len]");
  TORCH_CHECK(probs.dim() == 3, "probs workspace shape must be [k_count, heads, capacity]");
  TORCH_CHECK(outputs.dim() == 3, "outputs workspace shape must be [k_count, heads, dim]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  const auto k_count = score_grid.size(0);
  const auto heads = score_grid.size(1);
  const auto context_len = score_grid.size(2);
  const auto prob_stride = probs.size(2);
  const auto dim = values.size(1);
  TORCH_CHECK(probs.size(0) == k_count && probs.size(1) == heads, "probs workspace shape mismatch");
  TORCH_CHECK(probs.size(2) >= context_len, "probs workspace capacity must cover score_grid context_len");
  TORCH_CHECK(outputs.size(0) == k_count && outputs.size(1) == heads && outputs.size(2) == dim,
              "outputs workspace shape mismatch");
  TORCH_CHECK(values.size(0) >= context_len, "values length must cover score_grid context_len");
  TORCH_CHECK(score_grid.is_contiguous(), "score_grid must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(probs.is_contiguous(), "probs workspace must be contiguous");
  TORCH_CHECK(outputs.is_contiguous(), "outputs workspace must be contiguous");
  auto probs_view = probs.narrow(2, 0, context_len);
  if (k_count == 0 || heads == 0 || context_len == 0 || dim == 0) {
    if (context_len == 0 && k_count > 0 && heads > 0 && dim > 0) {
      outputs.zero_();
    }
    return {probs_view, outputs};
  }

  const int threads = 256;
  const int64_t rows = k_count * heads;
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_softmax_rows_strided_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      score_grid.data_ptr<float>(),
      probs.data_ptr<float>(),
      rows,
      context_len,
      prob_stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t out_blocks = rows * dim_tiles;
  joint_base_outputs_from_probs_strided_kernel<dims_per_block><<<
      static_cast<unsigned int>(out_blocks), threads, 0, stream>>>(
      probs.data_ptr<float>(),
      values.data_ptr<float>(),
      outputs.data_ptr<float>(),
      rows,
      context_len,
      prob_stride,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {probs_view, outputs};
}

template <int DIMS_PER_BLOCK>
__global__ void joint_grouped_base_outputs_from_probs_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ values_grouped,
    float* __restrict__ outputs,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t value_stride_tokens,
    int64_t dim) {
  constexpr int THREADS = 256;
  constexpr int TOKEN_LANES = THREADS / DIMS_PER_BLOCK;
  __shared__ float partial[DIMS_PER_BLOCK][TOKEN_LANES];

  const int64_t rows = groups * k_count * heads;
  const int64_t dim_tiles = (dim + DIMS_PER_BLOCK - 1) / DIMS_PER_BLOCK;
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  const int64_t dim_tile = block % dim_tiles;
  const int64_t row = block / dim_tiles;
  if (row >= rows) {
    return;
  }

  const int lane_dim = static_cast<int>(threadIdx.x % DIMS_PER_BLOCK);
  const int lane_token = static_cast<int>(threadIdx.x / DIMS_PER_BLOCK);
  const int64_t d = dim_tile * DIMS_PER_BLOCK + lane_dim;
  const int64_t group = row / (k_count * heads);
  float acc = 0.0f;
  if (d < dim) {
    const int64_t prob_offset = row * context_len;
    const int64_t value_offset = group * value_stride_tokens * dim;
    for (int64_t token = lane_token; token < context_len; token += TOKEN_LANES) {
      acc += probs[prob_offset + token] * values_grouped[value_offset + token * dim + d];
    }
  }

  partial[lane_dim][lane_token] = acc;
  __syncthreads();
  for (int stride = TOKEN_LANES >> 1; stride > 0; stride >>= 1) {
    if (lane_token < stride) {
      partial[lane_dim][lane_token] += partial[lane_dim][lane_token + stride];
    }
    __syncthreads();
  }
  if (lane_token == 0 && d < dim) {
    outputs[row * dim + d] = partial[lane_dim][0];
  }
}

std::vector<torch::Tensor> joint_softmax_base_outputs_grouped_cuda(
    torch::Tensor score_grid,
    torch::Tensor values_grouped) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values_grouped.is_cuda(), "values_grouped must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values_grouped.scalar_type() == torch::kFloat32, "values_grouped must be float32");
  TORCH_CHECK(score_grid.dim() == 4, "score_grid shape must be [groups, k_count, heads, context_len]");
  TORCH_CHECK(values_grouped.dim() == 3, "values_grouped shape must be [groups, context_len, dim]");
  const auto groups = score_grid.size(0);
  const auto k_count = score_grid.size(1);
  const auto heads = score_grid.size(2);
  const auto context_len = score_grid.size(3);
  const auto value_stride_tokens = values_grouped.size(1);
  const auto dim = values_grouped.size(2);
  TORCH_CHECK(values_grouped.size(0) == groups, "values_grouped group count mismatch");
  TORCH_CHECK(values_grouped.size(1) >= context_len, "values_grouped length must cover score_grid context_len");

  auto scores_c = score_grid.contiguous();
  auto values_c = values_grouped.contiguous();
  auto probs = torch::empty_like(scores_c, scores_c.options().dtype(torch::kFloat32));
  auto outputs = torch::empty({groups, k_count, heads, dim}, scores_c.options().dtype(torch::kFloat32));
  if (groups == 0 || k_count == 0 || heads == 0 || context_len == 0 || dim == 0) {
    if (context_len == 0 && groups > 0 && k_count > 0 && heads > 0 && dim > 0) {
      outputs.zero_();
    }
    return {probs, outputs};
  }

  const int threads = 256;
  const int64_t rows = groups * k_count * heads;
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_softmax_rows_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      scores_c.data_ptr<float>(),
      probs.data_ptr<float>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  joint_grouped_base_outputs_from_probs_kernel<dims_per_block><<<
      static_cast<unsigned int>(rows * dim_tiles), threads, 0, stream>>>(
      probs.data_ptr<float>(),
      values_c.data_ptr<float>(),
      outputs.data_ptr<float>(),
      groups,
      k_count,
      heads,
      context_len,
      value_stride_tokens,
      dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {probs, outputs};
}

std::vector<torch::Tensor> joint_softmax_base_outputs_cublas_cuda(
    torch::Tensor score_grid,
    torch::Tensor values) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(score_grid.dim() == 3, "score_grid shape must be [k_count, heads, context_len]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  const auto k_count = score_grid.size(0);
  const auto heads = score_grid.size(1);
  const auto context_len = score_grid.size(2);
  const auto dim = values.size(1);
  TORCH_CHECK(values.size(0) >= context_len, "values length must cover score_grid context_len");

  auto scores_c = score_grid.contiguous();
  auto values_c = values.contiguous();
  auto probs = torch::empty_like(scores_c, scores_c.options().dtype(torch::kFloat32));
  auto outputs = torch::empty({k_count, heads, dim}, scores_c.options().dtype(torch::kFloat32));
  if (k_count == 0 || heads == 0 || context_len == 0 || dim == 0) {
    if (context_len == 0 && k_count > 0 && heads > 0 && dim > 0) {
      outputs.zero_();
    }
    return {probs, outputs};
  }

  const int threads = 256;
  const int64_t rows = k_count * heads;
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_softmax_rows_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      scores_c.data_ptr<float>(),
      probs.data_ptr<float>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasSetStream(handle, stream.stream()), "cublasSetStream");
  cublasMath_t requested_math = CUBLAS_DEFAULT_MATH;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
  if (env_enabled_default("SELECTOR_PQ_JOINT_SOFTMAX_BASE_CUBLAS_TF32", false)) {
    requested_math = CUBLAS_TF32_TENSOR_OP_MATH;
  }
#endif
  ScopedCublasMathMode math_mode(handle, requested_math);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Row-major C = P @ V is column-major C^T = V^T @ P^T.
  check_cublas(
      cublasSgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          static_cast<int>(dim),
          static_cast<int>(rows),
          static_cast<int>(context_len),
          &alpha,
          values_c.data_ptr<float>(),
          static_cast<int>(dim),
          probs.data_ptr<float>(),
          static_cast<int>(context_len),
          &beta,
          outputs.data_ptr<float>(),
          static_cast<int>(dim)),
      "cublasSgemm joint_softmax_base_outputs_cublas");
  return {probs, outputs};
}

std::vector<torch::Tensor> joint_softmax_base_outputs_workspace_cublas_cuda(
    torch::Tensor probs,
    torch::Tensor outputs,
    torch::Tensor score_grid,
    torch::Tensor values) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs workspace must be CUDA");
  TORCH_CHECK(outputs.is_cuda(), "outputs workspace must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs workspace must be float32");
  TORCH_CHECK(outputs.scalar_type() == torch::kFloat32, "outputs workspace must be float32");
  TORCH_CHECK(score_grid.dim() == 3, "score_grid shape must be [k_count, heads, context_len]");
  TORCH_CHECK(probs.dim() == 3, "probs workspace shape must be [k_count, heads, context_len]");
  TORCH_CHECK(outputs.dim() == 3, "outputs workspace shape must be [k_count, heads, dim]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  const auto k_count = score_grid.size(0);
  const auto heads = score_grid.size(1);
  const auto context_len = score_grid.size(2);
  const auto dim = values.size(1);
  TORCH_CHECK(probs.size(0) == k_count && probs.size(1) == heads && probs.size(2) == context_len,
              "probs workspace shape mismatch");
  TORCH_CHECK(outputs.size(0) == k_count && outputs.size(1) == heads && outputs.size(2) == dim,
              "outputs workspace shape mismatch");
  TORCH_CHECK(values.size(0) >= context_len, "values length must cover score_grid context_len");
  TORCH_CHECK(score_grid.is_contiguous(), "score_grid must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(probs.is_contiguous(), "probs workspace must be contiguous");
  TORCH_CHECK(outputs.is_contiguous(), "outputs workspace must be contiguous");
  if (k_count == 0 || heads == 0 || context_len == 0 || dim == 0) {
    if (context_len == 0 && k_count > 0 && heads > 0 && dim > 0) {
      outputs.zero_();
    }
    return {probs, outputs};
  }

  const int threads = 256;
  const int64_t rows = k_count * heads;
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_softmax_rows_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      score_grid.data_ptr<float>(),
      probs.data_ptr<float>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasSetStream(handle, stream.stream()), "cublasSetStream");
  cublasMath_t requested_math = CUBLAS_DEFAULT_MATH;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
  if (env_enabled_default("SELECTOR_PQ_JOINT_SOFTMAX_BASE_CUBLAS_TF32", false)) {
    requested_math = CUBLAS_TF32_TENSOR_OP_MATH;
  }
#endif
  ScopedCublasMathMode math_mode(handle, requested_math);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
      cublasSgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          static_cast<int>(dim),
          static_cast<int>(rows),
          static_cast<int>(context_len),
          &alpha,
          values.data_ptr<float>(),
          static_cast<int>(dim),
          probs.data_ptr<float>(),
          static_cast<int>(context_len),
          &beta,
          outputs.data_ptr<float>(),
          static_cast<int>(dim)),
      "cublasSgemm joint_softmax_base_outputs_workspace_cublas");
  return {probs, outputs};
}

std::vector<torch::Tensor> joint_softmax_base_outputs_grouped_cublas_cuda(
    torch::Tensor score_grid,
    torch::Tensor values_grouped) {
  TORCH_CHECK(score_grid.is_cuda(), "score_grid must be CUDA");
  TORCH_CHECK(values_grouped.is_cuda(), "values_grouped must be CUDA");
  TORCH_CHECK(score_grid.scalar_type() == torch::kFloat32, "score_grid must be float32");
  TORCH_CHECK(values_grouped.scalar_type() == torch::kFloat32, "values_grouped must be float32");
  TORCH_CHECK(score_grid.dim() == 4, "score_grid shape must be [groups, k_count, heads, context_len]");
  TORCH_CHECK(values_grouped.dim() == 3, "values_grouped shape must be [groups, context_len, dim]");
  const auto groups = score_grid.size(0);
  const auto k_count = score_grid.size(1);
  const auto heads = score_grid.size(2);
  const auto context_len = score_grid.size(3);
  const auto value_context_stride = values_grouped.size(1);
  const auto dim = values_grouped.size(2);
  TORCH_CHECK(values_grouped.size(0) == groups, "values_grouped group count mismatch");
  TORCH_CHECK(values_grouped.size(1) >= context_len, "values_grouped length must cover score_grid context_len");

  auto scores_c = score_grid.contiguous();
  auto values_c = values_grouped.contiguous();
  auto probs = torch::empty_like(scores_c, scores_c.options().dtype(torch::kFloat32));
  auto outputs = torch::empty({groups, k_count, heads, dim}, scores_c.options().dtype(torch::kFloat32));
  if (groups == 0 || k_count == 0 || heads == 0 || context_len == 0 || dim == 0) {
    if (context_len == 0 && groups > 0 && k_count > 0 && heads > 0 && dim > 0) {
      outputs.zero_();
    }
    return {probs, outputs};
  }

  const int threads = 256;
  const int64_t rows_per_group = k_count * heads;
  const int64_t rows = groups * rows_per_group;
  auto stream = at::cuda::getCurrentCUDAStream();
  joint_softmax_rows_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      scores_c.data_ptr<float>(),
      probs.data_ptr<float>(),
      rows,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasSetStream(handle, stream.stream()), "cublasSetStream");
  cublasMath_t requested_math = CUBLAS_DEFAULT_MATH;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
  if (env_enabled_default("SELECTOR_PQ_JOINT_SOFTMAX_BASE_CUBLAS_TF32", false)) {
    requested_math = CUBLAS_TF32_TENSOR_OP_MATH;
  }
#endif
  ScopedCublasMathMode math_mode(handle, requested_math);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Per group, row-major C = P @ V is column-major C^T = V^T @ P^T.
  check_cublas(
      cublasSgemmStridedBatched(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          static_cast<int>(dim),
          static_cast<int>(rows_per_group),
          static_cast<int>(context_len),
          &alpha,
          values_c.data_ptr<float>(),
          static_cast<int>(dim),
          static_cast<long long>(value_context_stride * dim),
          probs.data_ptr<float>(),
          static_cast<int>(context_len),
          static_cast<long long>(rows_per_group * context_len),
          &beta,
          outputs.data_ptr<float>(),
          static_cast<int>(dim),
          static_cast<long long>(rows_per_group * dim),
          static_cast<int>(groups)),
      "cublasSgemmStridedBatched joint_softmax_base_outputs_grouped_cublas");
  return {probs, outputs};
}
