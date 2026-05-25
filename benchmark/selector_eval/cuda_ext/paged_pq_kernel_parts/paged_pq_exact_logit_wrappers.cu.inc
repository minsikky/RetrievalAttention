template <typename key_scalar_t>
__global__ void gqa_decode_full_exact_logits_kernel(
    const float* __restrict__ queries,
    const key_scalar_t* __restrict__ keys,
    float* __restrict__ logits,
    int64_t heads,
    int64_t kv_heads,
    int64_t context_len,
    int64_t dim,
    int64_t key_stride0,
    int64_t key_stride1,
    int64_t key_stride2,
    int64_t group_size,
    float scale) {
  constexpr int warps_per_block = 8;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
  const int64_t total = heads * context_len;
  if (warp_id >= warps_per_block || linear >= total) {
    return;
  }

  const int64_t head = linear / context_len;
  const int64_t token = linear - head * context_len;
  const int64_t kv_head = min(kv_heads - 1, max(static_cast<int64_t>(0), head / group_size));
  const float* __restrict__ query = queries + head * dim;

  float acc = 0.0f;
  for (int64_t d = lane; d < dim; d += 32) {
    const float q = query[d];
    const float k = load_as_float(keys, kv_head * key_stride0 + token * key_stride1 + d * key_stride2);
    acc += q * k;
  }
  acc = warp_reduce_sum(acc);
  if (lane == 0) {
    logits[linear] = acc * scale;
  }
}

template <typename key_scalar_t, int MAX_GROUP_HEADS>
__global__ void gqa_decode_full_exact_logits_grouped_kernel(
    const float* __restrict__ queries,
    const key_scalar_t* __restrict__ keys,
    float* __restrict__ logits,
    int64_t heads,
    int64_t kv_heads,
    int64_t context_len,
    int64_t dim,
    int64_t key_stride0,
    int64_t key_stride1,
    int64_t key_stride2,
    int64_t group_size,
    float scale) {
  constexpr int warps_per_block = 8;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
  const int64_t total = kv_heads * context_len;
  if (warp_id >= warps_per_block || linear >= total) {
    return;
  }

  const int64_t kv_head = linear / context_len;
  const int64_t token = linear - kv_head * context_len;
  const int64_t head_start = kv_head * group_size;
  int64_t local_heads = heads - head_start;
  if (local_heads > group_size) {
    local_heads = group_size;
  }
  if (local_heads <= 0) {
    return;
  }

  float acc[MAX_GROUP_HEADS];
#pragma unroll
  for (int h = 0; h < MAX_GROUP_HEADS; ++h) {
    acc[h] = 0.0f;
  }

  for (int64_t d = lane; d < dim; d += 32) {
    const float k = load_as_float(keys, kv_head * key_stride0 + token * key_stride1 + d * key_stride2);
#pragma unroll
    for (int h = 0; h < MAX_GROUP_HEADS; ++h) {
      if (h < local_heads) {
        acc[h] += queries[(head_start + h) * dim + d] * k;
      }
    }
  }

#pragma unroll
  for (int h = 0; h < MAX_GROUP_HEADS; ++h) {
    if (h < local_heads) {
      const float dot = warp_reduce_sum(acc[h]);
      if (lane == 0) {
        logits[(head_start + h) * context_len + token] = dot * scale;
      }
    }
  }
}

torch::Tensor gqa_decode_full_exact_logits_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    int64_t group_size,
    int64_t query_context_len,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const int64_t context_len = std::min<int64_t>(std::max<int64_t>(0, query_context_len), total_tokens);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto logits = torch::empty({heads, context_len}, opts);
  if (heads == 0 || context_len == 0) {
    return logits;
  }
  if (dim == 0 || kv_heads == 0) {
    logits.fill_(-std::numeric_limits<float>::infinity());
    return logits;
  }

  constexpr int warps_per_block = 8;
  constexpr int threads = warps_per_block * 32;
  const int64_t total = heads * context_len;
  const int64_t blocks = (total + warps_per_block - 1) / warps_per_block;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_decode_full_exact_logits_keys",
      [&] {
        gqa_decode_full_exact_logits_kernel<scalar_t>
            <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                queries.data_ptr<float>(),
                keys.data_ptr<scalar_t>(),
                logits.data_ptr<float>(),
                heads,
                kv_heads,
                context_len,
                dim,
                keys.stride(0),
                keys.stride(1),
                keys.stride(2),
                group_size,
                static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return logits;
}

torch::Tensor gqa_decode_full_exact_logits_grouped_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    int64_t group_size,
    int64_t query_context_len,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const int64_t context_len = std::min<int64_t>(std::max<int64_t>(0, query_context_len), total_tokens);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto logits = torch::empty({heads, context_len}, opts);
  if (heads == 0 || context_len == 0) {
    return logits;
  }
  if (dim == 0 || kv_heads == 0) {
    logits.fill_(-std::numeric_limits<float>::infinity());
    return logits;
  }
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(group_size <= 8, "grouped exact logits currently supports group_size <= 8");

  constexpr int warps_per_block = 8;
  constexpr int threads = warps_per_block * 32;
  const int64_t total = kv_heads * context_len;
  const int64_t blocks = (total + warps_per_block - 1) / warps_per_block;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_decode_full_exact_logits_grouped_keys",
      [&] {
        gqa_decode_full_exact_logits_grouped_kernel<scalar_t, 8>
            <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                queries.data_ptr<float>(),
                keys.data_ptr<scalar_t>(),
                logits.data_ptr<float>(),
                heads,
                kv_heads,
                context_len,
                dim,
                keys.stride(0),
                keys.stride(1),
                keys.stride(2),
                group_size,
                static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return logits;
}

torch::Tensor gqa_decode_full_exact_logits_t_cublas_cuda(
    torch::Tensor queries,
    torch::Tensor keys_t,
    int64_t group_size,
    int64_t query_context_len,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys_t.size(0);
  const auto key_dim = keys_t.size(1);
  const auto total_tokens = keys_t.size(2);
  const int64_t context_len = std::min<int64_t>(std::max<int64_t>(0, query_context_len), total_tokens);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto logits = torch::empty({heads, context_len}, opts);
  if (heads == 0 || context_len == 0) {
    return logits;
  }
  TORCH_CHECK(dim == key_dim, "query/key dim mismatch");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(heads == kv_heads * group_size, "cuBLAS exact logits require heads == kv_heads * group_size");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(keys_t.scalar_type() == torch::kFloat32, "keys_t must be float32");
  TORCH_CHECK(queries.stride(1) == 1, "queries must be row-contiguous in the head dimension");
  TORCH_CHECK(keys_t.stride(2) == 1, "keys_t must be contiguous along the token dimension");
  TORCH_CHECK(keys_t.stride(1) >= context_len, "keys_t dim stride must cover query_context_len");
  if (dim == 0 || kv_heads == 0) {
    logits.fill_(-std::numeric_limits<float>::infinity());
    return logits;
  }
  TORCH_CHECK(context_len <= std::numeric_limits<int>::max(), "context_len exceeds cuBLAS int range");
  TORCH_CHECK(group_size <= std::numeric_limits<int>::max(), "group_size exceeds cuBLAS int range");
  TORCH_CHECK(dim <= std::numeric_limits<int>::max(), "dim exceeds cuBLAS int range");
  TORCH_CHECK(kv_heads <= std::numeric_limits<int>::max(), "kv_heads exceeds cuBLAS int range");
  TORCH_CHECK(keys_t.stride(1) <= std::numeric_limits<int>::max(), "keys_t leading dimension exceeds cuBLAS int range");
  TORCH_CHECK(queries.stride(0) <= std::numeric_limits<int>::max(), "query leading dimension exceeds cuBLAS int range");
  TORCH_CHECK(logits.stride(0) <= std::numeric_limits<int>::max(), "logit leading dimension exceeds cuBLAS int range");

  auto stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasSetStream(handle, stream.stream()), "cublasSetStream");
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
  const bool use_tf32 = env_enabled_default("SELECTOR_PQ_NATIVE_EXACT_LOGITS_TF32", true);
  ScopedCublasMathMode math_mode(
      handle,
      use_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
#endif
  const float alpha = static_cast<float>(scale);
  const float beta = 0.0f;
  const int m = static_cast<int>(context_len);
  const int n = static_cast<int>(group_size);
  const int k = static_cast<int>(dim);
  const int lda = static_cast<int>(keys_t.stride(1));
  const int ldb = static_cast<int>(queries.stride(0));
  const int ldc = static_cast<int>(logits.stride(0));
  const long long stride_a = static_cast<long long>(keys_t.stride(0));
  const long long stride_b = static_cast<long long>(group_size * queries.stride(0));
  const long long stride_c = static_cast<long long>(group_size * logits.stride(0));
  check_cublas(
      cublasSgemmStridedBatched(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          m,
          n,
          k,
          &alpha,
          keys_t.data_ptr<float>(),
          lda,
          stride_a,
          queries.data_ptr<float>(),
          ldb,
          stride_b,
          &beta,
          logits.data_ptr<float>(),
          ldc,
          stride_c,
          static_cast<int>(kv_heads)),
      "cublasSgemmStridedBatched");
  return logits;
}

template <typename key_scalar_t>
__global__ void gqa_decode_token_exact_logits_kernel(
    const float* __restrict__ queries,
    const key_scalar_t* __restrict__ keys,
    const int64_t* __restrict__ tokens,
    float* __restrict__ logits,
    int64_t heads,
    int64_t kv_heads,
    int64_t selected,
    int64_t context_len,
    int64_t dim,
    int64_t key_stride0,
    int64_t key_stride1,
    int64_t key_stride2,
    int64_t group_size,
    float scale) {
  constexpr int warps_per_block = 8;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
  const int64_t total = heads * selected;
  if (warp_id >= warps_per_block || linear >= total) {
    return;
  }

  const int64_t head = linear / selected;
  const int64_t sel = linear - head * selected;
  const int64_t token = tokens[linear];
  float out = -INFINITY;
  if (token >= 0 && token < context_len && kv_heads > 0 && dim > 0) {
    const int64_t kv_head = min(kv_heads - 1, max(static_cast<int64_t>(0), head / group_size));
    const float* __restrict__ query = queries + head * dim;
    float acc = 0.0f;
    for (int64_t d = lane; d < dim; d += 32) {
      const float q = query[d];
      const float k = load_as_float(keys, kv_head * key_stride0 + token * key_stride1 + d * key_stride2);
      acc += q * k;
    }
    const float dot = warp_reduce_sum(acc);
    if (lane == 0) {
      out = dot * scale;
    }
  }
  if (lane == 0) {
    logits[linear] = out;
  }
}

torch::Tensor gqa_decode_token_exact_logits_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor tokens,
    int64_t group_size,
    int64_t query_context_len,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto selected = tokens.size(1);
  const int64_t context_len = std::min<int64_t>(std::max<int64_t>(0, query_context_len), total_tokens);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto logits = torch::empty({heads, selected}, opts);
  if (heads == 0 || selected == 0) {
    return logits;
  }
  if (dim == 0 || kv_heads == 0 || context_len == 0) {
    logits.fill_(-std::numeric_limits<float>::infinity());
    return logits;
  }
  TORCH_CHECK(group_size > 0, "group_size must be positive");

  constexpr int warps_per_block = 8;
  constexpr int threads = warps_per_block * 32;
  const int64_t total = heads * selected;
  const int64_t blocks = (total + warps_per_block - 1) / warps_per_block;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_decode_token_exact_logits_keys",
      [&] {
        gqa_decode_token_exact_logits_kernel<scalar_t>
            <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                queries.data_ptr<float>(),
                keys.data_ptr<scalar_t>(),
                tokens.data_ptr<int64_t>(),
                logits.data_ptr<float>(),
                heads,
                kv_heads,
                selected,
                context_len,
                dim,
                keys.stride(0),
                keys.stride(1),
                keys.stride(2),
                group_size,
                static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return logits;
}

torch::Tensor gqa_decode_ranked_exact_logits_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(1);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto ranked_logits = torch::empty({heads, ranked}, opts);
  if (heads == 0 || ranked == 0) {
    return ranked_logits;
  }
  if (dim == 0 || kv_heads == 0 || total_tokens == 0) {
    ranked_logits.fill_(-std::numeric_limits<float>::infinity());
    return ranked_logits;
  }
  const int threads = 256;
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  auto long_opts = queries.options().dtype(torch::kLong);
  auto base_tokens = torch::empty({heads, max_base}, long_opts);
  auto base_logits = torch::empty({heads, max_base}, opts);
  auto base_counts = torch::zeros({heads}, long_opts);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_decode_ranked_exact_logits_keys",
      [&] {
        gqa_decode_base_ranked_logits_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
            queries.data_ptr<float>(),
            keys.data_ptr<scalar_t>(),
            ranked_tokens.data_ptr<int64_t>(),
            ranked_scores.data_ptr<float>(),
            base_tokens.data_ptr<int64_t>(),
            base_logits.data_ptr<float>(),
            base_counts.data_ptr<int64_t>(),
            ranked_logits.data_ptr<float>(),
            heads,
            kv_heads,
            ranked,
            dim,
            total_tokens,
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            page_size,
            max_base,
            group_size,
            query_context_len,
            static_prefix,
            static_suffix,
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ranked_logits;
}

__global__ void gqa_decode_base_lse_from_logits_kernel(
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    float* __restrict__ base_lse,
    int64_t heads,
    int64_t max_base) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }
  extern __shared__ float shared[];
  const int tid = threadIdx.x;
  int64_t count = base_counts[head];
  if (count < 0) {
    count = 0;
  }
  if (count > max_base) {
    count = max_base;
  }
  const float* row = base_logits + head * max_base;
  float local_max = -INFINITY;
  for (int64_t idx = tid; idx < count; idx += blockDim.x) {
    local_max = fmaxf(local_max, row[idx]);
  }
  shared[tid] = local_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  float max_logit = shared[0];
  float local_sum = 0.0f;
  if (isfinite(max_logit)) {
    for (int64_t idx = tid; idx < count; idx += blockDim.x) {
      float logit = row[idx];
      if (isfinite(logit)) {
        local_sum += expf(logit - max_logit);
      }
    }
  }
  shared[tid] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    base_lse[head] = (isfinite(max_logit) && shared[0] > 0.0f) ? max_logit + logf(shared[0]) : -INFINITY;
  }
}

std::vector<torch::Tensor> gqa_decode_ranked_exact_logits_with_base_lse_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(1);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto ranked_logits = torch::empty({heads, ranked}, opts);
  auto base_lse = torch::empty({heads}, opts);
  if (heads == 0) {
    return {ranked_logits, base_lse};
  }
  if (dim == 0 || kv_heads == 0 || total_tokens == 0) {
    ranked_logits.fill_(-std::numeric_limits<float>::infinity());
    base_lse.fill_(-std::numeric_limits<float>::infinity());
    return {ranked_logits, base_lse};
  }
  const int threads = 256;
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  auto long_opts = queries.options().dtype(torch::kLong);
  auto base_tokens = torch::empty({heads, max_base}, long_opts);
  auto base_logits = torch::empty({heads, max_base}, opts);
  auto base_counts = torch::zeros({heads}, long_opts);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_decode_ranked_exact_logits_with_base_lse_keys",
      [&] {
        gqa_decode_base_ranked_logits_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
            queries.data_ptr<float>(),
            keys.data_ptr<scalar_t>(),
            ranked_tokens.data_ptr<int64_t>(),
            ranked_scores.data_ptr<float>(),
            base_tokens.data_ptr<int64_t>(),
            base_logits.data_ptr<float>(),
            base_counts.data_ptr<int64_t>(),
            ranked_logits.data_ptr<float>(),
            heads,
            kv_heads,
            ranked,
            dim,
            total_tokens,
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            page_size,
            max_base,
            group_size,
            query_context_len,
            static_prefix,
            static_suffix,
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gqa_decode_base_lse_from_logits_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      base_lse.data_ptr<float>(),
      heads,
      max_base);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {ranked_logits, base_lse};
}

