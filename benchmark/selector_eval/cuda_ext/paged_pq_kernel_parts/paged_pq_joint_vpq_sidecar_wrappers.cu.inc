template <typename value_t, typename err_t>
__global__ void joint_vpq_append_exact_suffix_kernel(
    const value_t* __restrict__ values,
    float* __restrict__ vhat,
    float* __restrict__ residual,
    err_t* __restrict__ code_error,
    int64_t start,
    int64_t end,
    int64_t dim) {
  const int64_t token_count = end > start ? end - start : 0;
  const int64_t total = token_count * dim;
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear < total) {
    const int64_t local_token = linear / dim;
    const int64_t d = linear - local_token * dim;
    const int64_t token = start + local_token;
    const int64_t offset = token * dim + d;
    vhat[offset] = static_cast<float>(values[offset]);
    residual[offset] = 0.0f;
  }
  const int64_t err_linear = linear - total;
  if (err_linear >= 0 && err_linear < token_count) {
    code_error[start + err_linear] = static_cast<err_t>(0);
  }
}

template <typename value_t, typename err_t>
__global__ void joint_vpq_append_exact_suffix_grouped_kernel(
    const value_t* __restrict__ values,
    float* __restrict__ vhat,
    float* __restrict__ residual,
    err_t* __restrict__ code_error,
    int64_t groups,
    int64_t start,
    int64_t end,
    int64_t dim,
    int64_t value_context_len,
    int64_t cache_capacity) {
  const int64_t token_count = end > start ? end - start : 0;
  const int64_t value_total = groups * token_count * dim;
  const int64_t error_total = groups * token_count;
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear < value_total) {
    const int64_t d = linear % dim;
    const int64_t token_linear = linear / dim;
    const int64_t local_token = token_linear % token_count;
    const int64_t group = token_linear / token_count;
    const int64_t token = start + local_token;
    const int64_t value_offset = (group * value_context_len + token) * dim + d;
    const int64_t cache_offset = (group * cache_capacity + token) * dim + d;
    vhat[cache_offset] = static_cast<float>(values[value_offset]);
    residual[cache_offset] = 0.0f;
  }
  const int64_t err_linear = linear - value_total;
  if (err_linear >= 0 && err_linear < error_total) {
    const int64_t local_token = err_linear % token_count;
    const int64_t group = err_linear / token_count;
    code_error[group * cache_capacity + start + local_token] = static_cast<err_t>(0);
  }
}

template <typename value_t, typename code_t>
__global__ void joint_vpq_sidecars_fill_kernel(
    const value_t* __restrict__ values,
    const float* __restrict__ codebooks,
    const code_t* __restrict__ codes,
    const int64_t* __restrict__ page_starts,
    float* __restrict__ vhat,
    float* __restrict__ residual,
    double* __restrict__ sum_sq,
    int* __restrict__ counts,
    int64_t context_len,
    int64_t dim,
    int64_t pages,
    int64_t page_size,
    int64_t subvecs,
    int64_t centroids,
    int64_t subdim) {
  const int64_t total = context_len * subvecs;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t token = linear / subvecs;
  const int64_t sub = linear - token * subvecs;
  const int64_t first_start = pages > 0 ? page_starts[0] : 0;
  int64_t page = (token - first_start) / page_size;
  if (token < first_start) {
    page = -1;
  }
  bool valid = page >= 0 && page < pages;
  int64_t row = 0;
  if (valid) {
    row = token - page_starts[page];
    valid = row >= 0 && row < page_size;
  }

  int64_t code = 0;
  if (valid) {
    code = static_cast<int64_t>(codes[(page * page_size + row) * subvecs + sub]);
    code = max(static_cast<int64_t>(0), min(code, centroids - 1));
  }

  double sq = 0.0;
  for (int64_t d = 0; d < subdim; ++d) {
    const int64_t out_d = sub * subdim + d;
    if (out_d >= dim) {
      continue;
    }
    const int64_t out_offset = token * dim + out_d;
    const float value = static_cast<float>(values[out_offset]);
    if (valid) {
      const float approx = codebooks[((page * subvecs + sub) * centroids + code) * subdim + d];
      const float diff = value - approx;
      vhat[out_offset] = approx;
      residual[out_offset] = diff;
      sq += static_cast<double>(diff) * static_cast<double>(diff);
    } else {
      vhat[out_offset] = value;
      residual[out_offset] = 0.0f;
    }
  }
  if (valid) {
    const int64_t bucket = (page * subvecs + sub) * centroids + code;
    atomicAdd(sum_sq + bucket, sq);
    atomicAdd(counts + bucket, 1);
  }
}

template <typename code_t>
__global__ void joint_vpq_sidecars_error_kernel(
    const code_t* __restrict__ codes,
    const int64_t* __restrict__ page_starts,
    const double* __restrict__ sum_sq,
    const int* __restrict__ counts,
    double* __restrict__ code_error,
    int64_t context_len,
    int64_t pages,
    int64_t page_size,
    int64_t subvecs,
    int64_t centroids) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= context_len) {
    return;
  }
  const int64_t first_start = pages > 0 ? page_starts[0] : 0;
  int64_t page = (token - first_start) / page_size;
  if (token < first_start) {
    page = -1;
  }
  bool valid = page >= 0 && page < pages;
  int64_t row = 0;
  if (valid) {
    row = token - page_starts[page];
    valid = row >= 0 && row < page_size;
  }
  if (!valid) {
    code_error[token] = 0.0;
    return;
  }
  double risk = 0.0;
  for (int64_t sub = 0; sub < subvecs; ++sub) {
    int64_t code = static_cast<int64_t>(codes[(page * page_size + row) * subvecs + sub]);
    code = max(static_cast<int64_t>(0), min(code, centroids - 1));
    const int64_t bucket = (page * subvecs + sub) * centroids + code;
    const int count = counts[bucket];
    if (count > 0) {
      risk += sum_sq[bucket] / static_cast<double>(count);
    }
  }
  code_error[token] = risk;
}

void joint_vpq_append_exact_suffix_cuda(
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
  TORCH_CHECK(vhat.scalar_type() == torch::kFloat32, "vhat must be float32");
  TORCH_CHECK(residual.scalar_type() == torch::kFloat32, "residual must be float32");
  TORCH_CHECK(vhat.is_contiguous(), "vhat must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(code_error.is_contiguous(), "code_error must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(vhat.dim() == 2, "vhat shape must be [capacity, dim]");
  TORCH_CHECK(residual.sizes() == vhat.sizes(), "residual must match vhat shape");
  TORCH_CHECK(code_error.dim() == 1, "code_error shape must be [capacity]");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  TORCH_CHECK(values.size(1) == vhat.size(1), "values dim must match vhat dim");
  const int64_t capacity = vhat.size(0);
  const int64_t context_len = values.size(0);
  const int64_t dim = vhat.size(1);
  TORCH_CHECK(code_error.size(0) >= capacity, "code_error capacity must cover vhat capacity");
  start = std::max<int64_t>(0, std::min<int64_t>(start, std::min(capacity, context_len)));
  end = std::max<int64_t>(start, std::min<int64_t>(end, std::min(capacity, context_len)));
  const int64_t token_count = end - start;
  if (token_count <= 0 || dim <= 0) {
    return;
  }
  const int threads = 256;
  const int64_t total = token_count * dim + token_count;
  const int64_t blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (code_error.scalar_type() == torch::kFloat64) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "joint_vpq_append_exact_suffix", [&] {
      joint_vpq_append_exact_suffix_kernel<scalar_t, double><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
          values.data_ptr<scalar_t>(),
          vhat.data_ptr<float>(),
          residual.data_ptr<float>(),
          code_error.data_ptr<double>(),
          start,
          end,
          dim);
    });
  } else {
    TORCH_CHECK(code_error.scalar_type() == torch::kFloat32, "code_error must be float32 or float64");
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "joint_vpq_append_exact_suffix", [&] {
      joint_vpq_append_exact_suffix_kernel<scalar_t, float><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
          values.data_ptr<scalar_t>(),
          vhat.data_ptr<float>(),
          residual.data_ptr<float>(),
          code_error.data_ptr<float>(),
          start,
          end,
          dim);
    });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> joint_vpq_sidecars_from_pack_cuda(
    torch::Tensor values,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t context_len) {
  TORCH_CHECK(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(page_starts.scalar_type() == torch::kInt64, "page_starts must be int64");
  TORCH_CHECK(values.dim() == 2, "values shape must be [context_len, dim]");
  TORCH_CHECK(codebooks.dim() == 4, "codebooks shape must be [pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 3, "codes shape must be [pages, page_size, subvecs]");
  TORCH_CHECK(page_starts.dim() == 1, "page_starts shape must be [pages]");
  TORCH_CHECK(codebooks.size(0) == codes.size(0), "codebooks/codes page count mismatch");
  TORCH_CHECK(codebooks.size(1) == codes.size(2), "codebooks/codes subvec count mismatch");
  TORCH_CHECK(page_starts.size(0) == codebooks.size(0), "page_starts page count mismatch");
  const int64_t value_len = values.size(0);
  const int64_t dim = values.size(1);
  const int64_t pages = codebooks.size(0);
  const int64_t subvecs = codebooks.size(1);
  const int64_t centroids = codebooks.size(2);
  const int64_t subdim = codebooks.size(3);
  const int64_t page_size = codes.size(1);
  context_len = std::max<int64_t>(0, std::min<int64_t>(context_len, value_len));
  TORCH_CHECK(pages > 0, "joint_vpq_sidecars_from_pack requires at least one page");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(subvecs > 0, "subvecs must be positive");
  TORCH_CHECK(centroids > 0, "centroids must be positive");
  TORCH_CHECK(subdim > 0, "subdim must be positive");
  TORCH_CHECK(subvecs * subdim == dim, "subvecs * subdim must equal value dim");

  auto values_c = values.contiguous();
  auto codebooks_c = codebooks.contiguous();
  auto codes_c = codes.contiguous();
  auto page_starts_c = page_starts.contiguous();
  auto float_opts = values.options().dtype(torch::kFloat32);
  auto double_opts = values.options().dtype(torch::kFloat64);
  auto int_opts = values.options().dtype(torch::kInt32);
  auto vhat = torch::empty({context_len, dim}, float_opts);
  auto residual = torch::empty({context_len, dim}, float_opts);
  auto code_error = torch::empty({context_len}, double_opts);
  auto sum_sq = torch::zeros({pages * subvecs * centroids}, double_opts);
  auto counts = torch::zeros({pages * subvecs * centroids}, int_opts);
  if (context_len <= 0) {
    return {vhat, residual, code_error};
  }

  const int threads = 256;
  const int64_t fill_total = context_len * subvecs;
  const int64_t fill_blocks = (fill_total + threads - 1) / threads;
  const int64_t err_blocks = (context_len + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();

#define DISPATCH_VPQ_SIDECAR_CODE(CODE_TYPE, CODE_PTR_EXPR)                                    \
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values_c.scalar_type(), "joint_vpq_sidecars_from_pack_values", [&] { \
    joint_vpq_sidecars_fill_kernel<scalar_t, CODE_TYPE><<<static_cast<unsigned int>(fill_blocks), threads, 0, stream>>>( \
        values_c.data_ptr<scalar_t>(),                                                         \
        codebooks_c.data_ptr<float>(),                                                         \
        CODE_PTR_EXPR,                                                                         \
        page_starts_c.data_ptr<int64_t>(),                                                     \
        vhat.data_ptr<float>(),                                                                \
        residual.data_ptr<float>(),                                                            \
        sum_sq.data_ptr<double>(),                                                             \
        counts.data_ptr<int>(),                                                                \
        context_len,                                                                           \
        dim,                                                                                   \
        pages,                                                                                 \
        page_size,                                                                             \
        subvecs,                                                                               \
        centroids,                                                                             \
        subdim);                                                                               \
  });                                                                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();                                                              \
  joint_vpq_sidecars_error_kernel<CODE_TYPE><<<static_cast<unsigned int>(err_blocks), threads, 0, stream>>>( \
      CODE_PTR_EXPR,                                                                           \
      page_starts_c.data_ptr<int64_t>(),                                                       \
      sum_sq.data_ptr<double>(),                                                               \
      counts.data_ptr<int>(),                                                                  \
      code_error.data_ptr<double>(),                                                           \
      context_len,                                                                             \
      pages,                                                                                   \
      page_size,                                                                               \
      subvecs,                                                                                 \
      centroids);                                                                              \
  C10_CUDA_KERNEL_LAUNCH_CHECK()

  if (codes_c.scalar_type() == torch::kUInt8) {
    DISPATCH_VPQ_SIDECAR_CODE(uint8_t, codes_c.data_ptr<uint8_t>());
  } else if (codes_c.scalar_type() == torch::kInt32) {
    DISPATCH_VPQ_SIDECAR_CODE(int32_t, codes_c.data_ptr<int32_t>());
  } else {
    TORCH_CHECK(codes_c.scalar_type() == torch::kInt64, "codes must be uint8, int32, or int64");
    DISPATCH_VPQ_SIDECAR_CODE(int64_t, codes_c.data_ptr<int64_t>());
  }
#undef DISPATCH_VPQ_SIDECAR_CODE

  return {vhat, residual, code_error};
}

void joint_vpq_append_exact_suffix_grouped_cuda(
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
  TORCH_CHECK(vhat.scalar_type() == torch::kFloat32, "vhat must be float32");
  TORCH_CHECK(residual.scalar_type() == torch::kFloat32, "residual must be float32");
  TORCH_CHECK(vhat.is_contiguous(), "vhat must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(code_error.is_contiguous(), "code_error must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(vhat.dim() == 3, "vhat shape must be [groups, capacity, dim]");
  TORCH_CHECK(residual.sizes() == vhat.sizes(), "residual must match vhat shape");
  TORCH_CHECK(code_error.dim() == 2, "code_error shape must be [groups, capacity]");
  TORCH_CHECK(values.dim() == 3, "values shape must be [groups, context_len, dim]");
  TORCH_CHECK(values.size(0) == vhat.size(0), "values groups must match vhat groups");
  TORCH_CHECK(values.size(2) == vhat.size(2), "values dim must match vhat dim");
  TORCH_CHECK(code_error.size(0) == vhat.size(0), "code_error groups must match vhat groups");
  TORCH_CHECK(code_error.size(1) >= vhat.size(1), "code_error capacity must cover vhat capacity");
  const int64_t groups = vhat.size(0);
  const int64_t capacity = vhat.size(1);
  const int64_t context_len = values.size(1);
  const int64_t dim = vhat.size(2);
  start = std::max<int64_t>(0, std::min<int64_t>(start, std::min(capacity, context_len)));
  end = std::max<int64_t>(start, std::min<int64_t>(end, std::min(capacity, context_len)));
  const int64_t token_count = end - start;
  if (groups <= 0 || token_count <= 0 || dim <= 0) {
    return;
  }
  const int threads = 256;
  const int64_t total = groups * token_count * dim + groups * token_count;
  const int64_t blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (code_error.scalar_type() == torch::kFloat64) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "joint_vpq_append_exact_suffix_grouped", [&] {
      joint_vpq_append_exact_suffix_grouped_kernel<scalar_t, double><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
          values.data_ptr<scalar_t>(),
          vhat.data_ptr<float>(),
          residual.data_ptr<float>(),
          code_error.data_ptr<double>(),
          groups,
          start,
          end,
          dim,
          context_len,
          capacity);
    });
  } else {
    TORCH_CHECK(code_error.scalar_type() == torch::kFloat32, "code_error must be float32 or float64");
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "joint_vpq_append_exact_suffix_grouped", [&] {
      joint_vpq_append_exact_suffix_grouped_kernel<scalar_t, float><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
          values.data_ptr<scalar_t>(),
          vhat.data_ptr<float>(),
          residual.data_ptr<float>(),
          code_error.data_ptr<float>(),
          groups,
          start,
          end,
          dim,
          context_len,
          capacity);
    });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

