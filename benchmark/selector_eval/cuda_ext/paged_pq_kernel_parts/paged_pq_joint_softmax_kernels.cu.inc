__global__ void joint_softmax_rows_kernel(
    const float* __restrict__ scores,
    float* __restrict__ probs,
    int64_t rows,
    int64_t context_len) {
  int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows || context_len <= 0) {
    return;
  }

  extern __shared__ float shared[];
  float local_max = -INFINITY;
  const int64_t row_offset = row * context_len;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    local_max = fmaxf(local_max, scores[row_offset + token]);
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_v = shared[0];

  float local_sum = 0.0f;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    float p = expf(scores[row_offset + token] - max_v);
    probs[row_offset + token] = p;
    local_sum += p;
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float denom = fmaxf(shared[0], 1.0e-20f);
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    probs[row_offset + token] /= denom;
  }
}

__global__ void joint_softmax_rows_strided_kernel(
    const float* __restrict__ scores,
    float* __restrict__ probs,
    int64_t rows,
    int64_t context_len,
    int64_t prob_stride) {
  int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows || context_len <= 0) {
    return;
  }

  extern __shared__ float shared[];
  float local_max = -INFINITY;
  const int64_t score_offset = row * context_len;
  const int64_t prob_offset = row * prob_stride;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    local_max = fmaxf(local_max, scores[score_offset + token]);
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_v = shared[0];

  float local_sum = 0.0f;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    float p = expf(scores[score_offset + token] - max_v);
    probs[prob_offset + token] = p;
    local_sum += p;
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float denom = fmaxf(shared[0], 1.0e-20f);
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    probs[prob_offset + token] /= denom;
  }
}

template <int DIMS_PER_BLOCK>
__global__ void joint_base_outputs_from_probs_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ values,
    float* __restrict__ outputs,
    int64_t rows,
    int64_t context_len,
    int64_t dim) {
  constexpr int THREADS = 256;
  constexpr int TOKEN_LANES = THREADS / DIMS_PER_BLOCK;
  __shared__ float partial[DIMS_PER_BLOCK][TOKEN_LANES];

  int64_t dim_tiles = (dim + DIMS_PER_BLOCK - 1) / DIMS_PER_BLOCK;
  int64_t block = static_cast<int64_t>(blockIdx.x);
  int64_t dim_tile = block % dim_tiles;
  int64_t row = block / dim_tiles;
  if (row >= rows) {
    return;
  }

  int lane_dim = static_cast<int>(threadIdx.x % DIMS_PER_BLOCK);
  int lane_token = static_cast<int>(threadIdx.x / DIMS_PER_BLOCK);
  int64_t d = dim_tile * DIMS_PER_BLOCK + lane_dim;
  float acc = 0.0f;
  if (d < dim) {
    const int64_t prob_offset = row * context_len;
    for (int64_t token = lane_token; token < context_len; token += TOKEN_LANES) {
      acc += probs[prob_offset + token] * values[token * dim + d];
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

template <int DIMS_PER_BLOCK>
__global__ void joint_base_outputs_from_probs_strided_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ values,
    float* __restrict__ outputs,
    int64_t rows,
    int64_t context_len,
    int64_t prob_stride,
    int64_t dim) {
  constexpr int THREADS = 256;
  constexpr int TOKEN_LANES = THREADS / DIMS_PER_BLOCK;
  __shared__ float partial[DIMS_PER_BLOCK][TOKEN_LANES];

  int64_t dim_tiles = (dim + DIMS_PER_BLOCK - 1) / DIMS_PER_BLOCK;
  int64_t block = static_cast<int64_t>(blockIdx.x);
  int64_t dim_tile = block % dim_tiles;
  int64_t row = block / dim_tiles;
  if (row >= rows) {
    return;
  }

  int lane_dim = static_cast<int>(threadIdx.x % DIMS_PER_BLOCK);
  int lane_token = static_cast<int>(threadIdx.x / DIMS_PER_BLOCK);
  int64_t d = dim_tile * DIMS_PER_BLOCK + lane_dim;
  float acc = 0.0f;
  if (d < dim) {
    const int64_t prob_offset = row * prob_stride;
    for (int64_t token = lane_token; token < context_len; token += TOKEN_LANES) {
      acc += probs[prob_offset + token] * values[token * dim + d];
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

__global__ void joint_grouped_flat_risk_values_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ code_error_groups,
    float* __restrict__ risk,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t rows = groups * k_count * heads;
  int64_t total = rows * context_len;
  if (linear >= total) {
    return;
  }
  int64_t token = linear % context_len;
  int64_t row = linear / context_len;
  int64_t group = row / (k_count * heads);
  float p = probs[linear];
  float e = code_error_groups[group * context_len + token];
  risk[linear] = p * p * e;
}

__global__ void joint_grouped_flat_risk_sort_inputs_strided_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ code_error_groups,
    float* __restrict__ risk,
    int32_t* __restrict__ ids,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t prob_stride) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t rows = groups * k_count * heads;
  int64_t total = rows * context_len;
  if (linear >= total) {
    return;
  }
  int64_t token = linear % context_len;
  int64_t row = linear / context_len;
  int64_t group = row / (k_count * heads);
  float p = probs[row * prob_stride + token];
  float e = code_error_groups[group * context_len + token];
  risk[linear] = p * p * e;
  ids[linear] = static_cast<int32_t>(token);
}

__global__ void joint_rank_prefix_sort_inputs_kernel(
    const float* __restrict__ scores,
    float* __restrict__ score_in,
    int32_t* __restrict__ pos_in,
    int64_t rows,
    int64_t count) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * count;
  if (linear >= total) {
    return;
  }
  int64_t pos = linear % count;
  score_in[linear] = scores[linear];
  pos_in[linear] = static_cast<int32_t>(pos);
}

__global__ void joint_rank_prefix_gather_tokens_kernel(
    const int32_t* __restrict__ sorted_pos,
    const int64_t* __restrict__ indexed_tokens,
    int64_t* __restrict__ ranked_tokens,
    int64_t rows,
    int64_t count,
    int64_t max_take) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * max_take;
  if (linear >= total) {
    return;
  }
  int64_t take = linear % max_take;
  int64_t row = linear / max_take;
  int64_t pos = static_cast<int64_t>(sorted_pos[row * count + take]);
  ranked_tokens[linear] = (pos >= 0 && pos < count) ? indexed_tokens[pos] : static_cast<int64_t>(-1);
}

__device__ __forceinline__ uint32_t ordered_float_key_desc(float x) {
  uint32_t bits = __float_as_uint(x);
  // Monotonic float-to-uint transform: larger float -> larger key.
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__global__ void joint_budget_prefix_thresholds_kernel(
    const float* __restrict__ scores,
    const int64_t* __restrict__ take_counts,
    int32_t* __restrict__ thresholds,
    int32_t* __restrict__ greater_counts,
    int32_t* __restrict__ tie_quotas,
    int64_t rows,
    int64_t count,
    int64_t budget_count) {
  int64_t rb = static_cast<int64_t>(blockIdx.x);
  int64_t row = rb / budget_count;
  int64_t bi = rb - row * budget_count;
  if (row >= rows || bi >= budget_count) {
    return;
  }
  extern __shared__ int prefix_counts[];
  int64_t take64 = take_counts[bi];
  take64 = max(static_cast<int64_t>(0), min(take64, count));
  if (take64 <= 0 || count <= 0) {
    if (threadIdx.x == 0) {
      thresholds[rb] = 0;
      greater_counts[rb] = 0;
      tie_quotas[rb] = 0;
    }
    return;
  }
  __shared__ uint32_t prefix_shared;
  __shared__ int kth_remaining_shared;
  if (threadIdx.x == 0) {
    prefix_shared = 0u;
    kth_remaining_shared = static_cast<int>(take64);
  }
  __syncthreads();
  uint32_t mask = 0u;
  const float* row_scores = scores + row * count;
  for (int bit = 31; bit >= 0; --bit) {
    mask |= (1u << bit);
    uint32_t candidate = prefix_shared | (1u << bit);
    int local_count = 0;
    for (int64_t j = threadIdx.x; j < count; j += blockDim.x) {
      uint32_t key = ordered_float_key_desc(row_scores[j]);
      if ((key & mask) == candidate) {
        local_count += 1;
      }
    }
    prefix_counts[threadIdx.x] = local_count;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        prefix_counts[threadIdx.x] += prefix_counts[threadIdx.x + stride];
      }
      __syncthreads();
    }
    int count_with_one = prefix_counts[0];
    if (threadIdx.x == 0) {
      if (count_with_one >= kth_remaining_shared) {
        prefix_shared = candidate;
      } else {
        kth_remaining_shared -= count_with_one;
      }
    }
    __syncthreads();
  }
  uint32_t prefix = prefix_shared;

  int local_greater = 0;
  int local_equal = 0;
  for (int64_t j = threadIdx.x; j < count; j += blockDim.x) {
    uint32_t key = ordered_float_key_desc(row_scores[j]);
    if (key > prefix) {
      local_greater += 1;
    } else if (key == prefix) {
      local_equal += 1;
    }
  }
  prefix_counts[threadIdx.x] = local_greater;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      prefix_counts[threadIdx.x] += prefix_counts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  int greater = prefix_counts[0];
  prefix_counts[threadIdx.x] = local_equal;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      prefix_counts[threadIdx.x] += prefix_counts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  int equal = max(1, prefix_counts[0]);
  if (threadIdx.x == 0) {
    int quota = max(0, min(static_cast<int>(take64) - greater, equal));
    thresholds[rb] = static_cast<int32_t>(prefix);
    greater_counts[rb] = greater;
    tie_quotas[rb] = quota;
  }
}

__global__ void joint_budget_prefix_fill_kernel(
    const float* __restrict__ scores,
    const int64_t* __restrict__ indexed_tokens,
    const int64_t* __restrict__ take_counts,
    const int32_t* __restrict__ thresholds,
    const int32_t* __restrict__ tie_quotas,
    int32_t* __restrict__ band_counters,
    int32_t* __restrict__ tie_counters,
    int64_t* __restrict__ out,
    int64_t rows,
    int64_t count,
    int64_t budget_count,
    int64_t max_take) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * count;
  if (linear >= total || max_take <= 0) {
    return;
  }
  int64_t pos = linear % count;
  int64_t row = linear / count;
  uint32_t key = ordered_float_key_desc(scores[linear]);
  int64_t category = -1;
  for (int64_t bi = 0; bi < budget_count; ++bi) {
    int64_t take = max(static_cast<int64_t>(0), min(take_counts[bi], max_take));
    int64_t prev_take = (bi == 0) ? 0 : max(static_cast<int64_t>(0), min(take_counts[bi - 1], max_take));
    if (take <= prev_take) {
      continue;
    }
    int64_t rb = row * budget_count + bi;
    uint32_t threshold = static_cast<uint32_t>(thresholds[rb]);
    bool selected = key > threshold;
    if (!selected && key == threshold) {
      int slot = atomicAdd(tie_counters + rb, 1);
      selected = slot < tie_quotas[rb];
    }
    if (selected) {
      category = bi;
      break;
    }
  }
  if (category < 0) {
    return;
  }
  int64_t start = (category == 0)
      ? 0
      : max(static_cast<int64_t>(0), min(take_counts[category - 1], max_take));
  int64_t end = max(static_cast<int64_t>(0), min(take_counts[category], max_take));
  int slot = atomicAdd(band_counters + row * budget_count + category, 1);
  int64_t out_pos = start + static_cast<int64_t>(slot);
  if (out_pos >= start && out_pos < end) {
    out[row * max_take + out_pos] = indexed_tokens[pos];
  }
}

