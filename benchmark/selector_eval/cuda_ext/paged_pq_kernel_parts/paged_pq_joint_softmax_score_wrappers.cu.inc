__global__ void joint_softmax_row_stats_kernel(
    const float* __restrict__ scores,
    float* __restrict__ row_max,
    float* __restrict__ row_denom,
    int64_t rows,
    int64_t context_len) {
  int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows || context_len <= 0) {
    return;
  }

  extern __shared__ float shared[];
  const int64_t row_offset = row * context_len;
  float local_max = -INFINITY;
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
    local_sum += expf(scores[row_offset + token] - max_v);
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_max[row] = max_v;
    row_denom[row] = fmaxf(shared[0], 1.0e-20f);
  }
}

template <int DIMS_PER_BLOCK>
__global__ void joint_base_outputs_from_grouped_scores_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ vhat_groups,
    float* __restrict__ outputs,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
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
    const int64_t score_offset = row * context_len;
    const float max_v = row_max[row];
    const float denom = row_denom[row];
    for (int64_t token = lane_token; token < context_len; token += TOKEN_LANES) {
      const float p = expf(scores[score_offset + token] - max_v) / denom;
      acc += p * vhat_groups[(group * context_len + token) * dim + d];
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

__global__ void joint_grouped_flat_risk_sort_inputs_from_scores_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ code_error_groups,
    float* __restrict__ risk_in,
    int32_t* __restrict__ ids_in,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len) {
  const int64_t rows = groups * k_count * heads;
  const int64_t total = rows * context_len;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t token = linear % context_len;
  const int64_t row = linear / context_len;
  const int64_t group = row / (k_count * heads);
  const float p = expf(scores[linear] - row_max[row]) / row_denom[row];
  const float e = code_error_groups[group * context_len + token];
  risk_in[linear] = p * p * e;
  ids_in[linear] = static_cast<int32_t>(token);
}

__global__ void joint_grouped_flat_risk_values_from_scores_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ code_error_groups,
    float* __restrict__ risk,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len) {
  const int64_t rows = groups * k_count * heads;
  const int64_t total = rows * context_len;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t token = linear % context_len;
  const int64_t row = linear / context_len;
  const int64_t group = row / (k_count * heads);
  const float p = expf(scores[linear] - row_max[row]) / row_denom[row];
  const float e = code_error_groups[group * context_len + token];
  risk[linear] = p * p * e;
}

template <typename index_t, int DIMS_PER_BLOCK>
__global__ void joint_grouped_flat_vprefix_interval_sums_from_scores_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ residual_groups,
    const index_t* __restrict__ sorted_ids,
    const int64_t* __restrict__ v_budgets,
    float* __restrict__ interval_sums,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t sorted_count,
    int64_t v_steps,
    int64_t dim) {
  constexpr int THREADS = 256;
  constexpr int TOKEN_LANES = THREADS / DIMS_PER_BLOCK;
  __shared__ float partial[DIMS_PER_BLOCK][TOKEN_LANES];

  const int64_t rows = groups * k_count * heads;
  const int64_t dim_tiles = (dim + DIMS_PER_BLOCK - 1) / DIMS_PER_BLOCK;
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  const int64_t dim_tile = block % dim_tiles;
  const int64_t tmp = block / dim_tiles;
  const int64_t v_step = tmp % v_steps;
  const int64_t row = tmp / v_steps;
  if (row >= rows) {
    return;
  }

  const int lane_dim = static_cast<int>(threadIdx.x % DIMS_PER_BLOCK);
  const int lane_pos = static_cast<int>(threadIdx.x / DIMS_PER_BLOCK);
  const int64_t d = dim_tile * DIMS_PER_BLOCK + lane_dim;
  const int64_t group = row / (k_count * heads);
  const int64_t budget = v_budgets[v_step];
  const int64_t exact_count = budget <= 0 ? 0 : (budget < sorted_count ? budget : sorted_count);
  int64_t prev_count = 0;
  if (v_step > 0) {
    const int64_t prev_budget = v_budgets[v_step - 1];
    prev_count = prev_budget <= 0 ? 0 : (prev_budget < sorted_count ? prev_budget : sorted_count);
  }
  const int64_t start_count = exact_count < prev_count ? 0 : prev_count;
  float acc = 0.0f;
  if (d < dim) {
    const int64_t score_offset = row * context_len;
    const float max_v = row_max[row];
    const float denom = row_denom[row];
    for (int64_t pos = start_count + lane_pos; pos < exact_count; pos += TOKEN_LANES) {
      const int64_t token = static_cast<int64_t>(sorted_ids[row * sorted_count + pos]);
      const float p = expf(scores[score_offset + token] - max_v) / denom;
      acc += p * residual_groups[(group * context_len + token) * dim + d];
    }
  }

  partial[lane_dim][lane_pos] = acc;
  __syncthreads();
  for (int stride = TOKEN_LANES >> 1; stride > 0; stride >>= 1) {
    if (lane_pos < stride) {
      partial[lane_dim][lane_pos] += partial[lane_dim][lane_pos + stride];
    }
    __syncthreads();
  }
  if (lane_pos == 0 && d < dim) {
    interval_sums[(row * v_steps + v_step) * dim + d] = partial[lane_dim][0];
  }
}

__device__ __forceinline__ float joint_mixed_score_from_mask(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ selected_mask,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    int64_t row,
    int64_t head,
    int64_t ki,
    int64_t token,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    float pq_scale) {
  const int pos = token_to_indexed[token];
  const bool full_exact = k_take_counts[ki] > max_rank_take;
  if (pos >= 0 && !full_exact && selected_mask[row * context_len + token] == static_cast<unsigned char>(0)) {
    return fit_scale[row] * (pq_logits[head * indexed_count + static_cast<int64_t>(pos)] * pq_scale) + fit_bias[row];
  }
  return exact_scores[head * context_len + token];
}

__global__ void joint_mixed_row_stats_from_mask_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ selected_mask,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    float* __restrict__ row_max,
    float* __restrict__ row_denom,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    float pq_scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows || context_len <= 0) {
    return;
  }
  const int64_t head = row % heads;
  const int64_t ki = row / heads;
  extern __shared__ float shared[];
  float local_max = -INFINITY;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    const float score = joint_mixed_score_from_mask(
        exact_scores,
        pq_logits,
        token_to_indexed,
        selected_mask,
        k_take_counts,
        fit_scale,
        fit_bias,
        row,
        head,
        ki,
        token,
        heads,
        indexed_count,
        max_rank_take,
        context_len,
        pq_scale);
    local_max = fmaxf(local_max, score);
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
    const float score = joint_mixed_score_from_mask(
        exact_scores,
        pq_logits,
        token_to_indexed,
        selected_mask,
        k_take_counts,
        fit_scale,
        fit_bias,
        row,
        head,
        ki,
        token,
        heads,
        indexed_count,
        max_rank_take,
        context_len,
        pq_scale);
    local_sum += expf(score - max_v);
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_max[row] = max_v;
    row_denom[row] = fmaxf(shared[0], 1.0e-20f);
  }
}

template <int DIMS_PER_BLOCK>
__global__ void joint_base_outputs_from_mixed_mask_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ selected_mask,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ vhat,
    float* __restrict__ outputs,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    int64_t dim,
    float pq_scale) {
  constexpr int THREADS = 256;
  constexpr int TOKEN_LANES = THREADS / DIMS_PER_BLOCK;
  __shared__ float partial[DIMS_PER_BLOCK][TOKEN_LANES];

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
  const int64_t head = row % heads;
  const int64_t ki = row / heads;
  float acc = 0.0f;
  if (d < dim) {
    const float max_v = row_max[row];
    const float denom = row_denom[row];
    for (int64_t token = lane_token; token < context_len; token += TOKEN_LANES) {
      const float score = joint_mixed_score_from_mask(
          exact_scores,
          pq_logits,
          token_to_indexed,
          selected_mask,
          k_take_counts,
          fit_scale,
          fit_bias,
          row,
          head,
          ki,
          token,
          heads,
          indexed_count,
          max_rank_take,
          context_len,
          pq_scale);
      const float p = expf(score - max_v) / denom;
      acc += p * vhat[token * dim + d];
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

__global__ void joint_risk_sort_inputs_from_mixed_mask_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ selected_mask,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ code_error,
    float* __restrict__ risk_in,
    int32_t* __restrict__ ids_in,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    float pq_scale) {
  const int64_t total = rows * context_len;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t token = linear % context_len;
  const int64_t row = linear / context_len;
  const int64_t head = row % heads;
  const int64_t ki = row / heads;
  const float score = joint_mixed_score_from_mask(
      exact_scores,
      pq_logits,
      token_to_indexed,
      selected_mask,
      k_take_counts,
      fit_scale,
      fit_bias,
      row,
      head,
      ki,
      token,
      heads,
      indexed_count,
      max_rank_take,
      context_len,
      pq_scale);
  const float p = expf(score - row_max[row]) / row_denom[row];
  risk_in[linear] = p * p * code_error[token];
  ids_in[linear] = static_cast<int32_t>(token);
}

__global__ void joint_mixed_probs_risk_inputs_from_mask_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ selected_mask,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ code_error,
    float* __restrict__ probs,
    float* __restrict__ risk_in,
    int32_t* __restrict__ ids_in,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    float pq_scale) {
  const int64_t total = rows * context_len;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t token = linear % context_len;
  const int64_t row = linear / context_len;
  const int64_t head = row % heads;
  const int64_t ki = row / heads;
  const float score = joint_mixed_score_from_mask(
      exact_scores,
      pq_logits,
      token_to_indexed,
      selected_mask,
      k_take_counts,
      fit_scale,
      fit_bias,
      row,
      head,
      ki,
      token,
      heads,
      indexed_count,
      max_rank_take,
      context_len,
      pq_scale);
  const float p = expf(score - row_max[row]) / row_denom[row];
  probs[linear] = p;
  risk_in[linear] = p * p * code_error[token];
  ids_in[linear] = static_cast<int32_t>(token);
}

__device__ __forceinline__ float joint_mixed_score_from_rankpos_nocalib(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    int64_t row,
    int64_t head,
    int64_t ki,
    int64_t token,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    float pq_scale) {
  const int pos = token_to_indexed[token];
  const int64_t take = k_take_counts[ki];
  const bool full_exact = take > max_rank_take;
  if (pos >= 0 && !full_exact) {
    bool selected = base_mask[token] != static_cast<unsigned char>(0);
    const int rank_i = rank_pos[head * context_len + token];
    selected = selected || (rank_i >= 0 && static_cast<int64_t>(rank_i) < take);
    if (!selected) {
      return pq_logits[head * indexed_count + static_cast<int64_t>(pos)] * pq_scale;
    }
  }
  return exact_scores[head * context_len + token];
}

__global__ void joint_mixed_softmax_rows_rankpos_nocalib_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    float* __restrict__ probs,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    float pq_scale,
    int64_t context_len) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows || context_len <= 0) {
    return;
  }
  const int64_t head = row % heads;
  const int64_t ki = row / heads;
  const int64_t row_offset = row * context_len;

  extern __shared__ float shared[];
  float local_max = -INFINITY;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    const float score = joint_mixed_score_from_rankpos_nocalib(
        exact_scores,
        pq_logits,
        token_to_indexed,
        base_mask,
        rank_pos,
        k_take_counts,
        row,
        head,
        ki,
        token,
        heads,
        indexed_count,
        max_rank_take,
        context_len,
        pq_scale);
    local_max = fmaxf(local_max, score);
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
    const float score = joint_mixed_score_from_rankpos_nocalib(
        exact_scores,
        pq_logits,
        token_to_indexed,
        base_mask,
        rank_pos,
        k_take_counts,
        row,
        head,
        ki,
        token,
        heads,
        indexed_count,
        max_rank_take,
        context_len,
        pq_scale);
    const float p = expf(score - max_v);
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

__global__ void joint_mixed_row_stats_rankpos_nocalib_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    float* __restrict__ row_max,
    float* __restrict__ row_denom,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    float pq_scale,
    int64_t context_len) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows || context_len <= 0) {
    return;
  }
  const int64_t head = row % heads;
  const int64_t ki = row / heads;

  extern __shared__ float shared[];
  float local_max = -INFINITY;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    const float score = joint_mixed_score_from_rankpos_nocalib(
        exact_scores,
        pq_logits,
        token_to_indexed,
        base_mask,
        rank_pos,
        k_take_counts,
        row,
        head,
        ki,
        token,
        heads,
        indexed_count,
        max_rank_take,
        context_len,
        pq_scale);
    local_max = fmaxf(local_max, score);
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
    const float score = joint_mixed_score_from_rankpos_nocalib(
        exact_scores,
        pq_logits,
        token_to_indexed,
        base_mask,
        rank_pos,
        k_take_counts,
        row,
        head,
        ki,
        token,
        heads,
        indexed_count,
        max_rank_take,
        context_len,
        pq_scale);
    local_sum += expf(score - max_v);
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_max[row] = max_v;
    row_denom[row] = fmaxf(shared[0], 1.0e-20f);
  }
}

template <int DIMS_PER_BLOCK>
__global__ void joint_base_outputs_from_mixed_rankpos_nocalib_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ vhat,
    float* __restrict__ outputs,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    int64_t dim,
    float pq_scale) {
  constexpr int THREADS = 256;
  constexpr int TOKEN_LANES = THREADS / DIMS_PER_BLOCK;
  __shared__ float partial[DIMS_PER_BLOCK][TOKEN_LANES];

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
  const int64_t head = row % heads;
  const int64_t ki = row / heads;
  float acc = 0.0f;
  if (d < dim) {
    const float max_v = row_max[row];
    const float denom = row_denom[row];
    for (int64_t token = lane_token; token < context_len; token += TOKEN_LANES) {
      const float score = joint_mixed_score_from_rankpos_nocalib(
          exact_scores,
          pq_logits,
          token_to_indexed,
          base_mask,
          rank_pos,
          k_take_counts,
          row,
          head,
          ki,
          token,
          heads,
          indexed_count,
          max_rank_take,
          context_len,
          pq_scale);
      const float p = expf(score - max_v) / denom;
      acc += p * vhat[token * dim + d];
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

__global__ void joint_risk_sort_inputs_from_mixed_rankpos_nocalib_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ code_error,
    float* __restrict__ risk_in,
    int32_t* __restrict__ ids_in,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    float pq_scale) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * context_len;
  if (linear >= total) {
    return;
  }
  const int64_t token = linear % context_len;
  const int64_t row = linear / context_len;
  const int64_t head = row % heads;
  const int64_t ki = row / heads;
  const float score = joint_mixed_score_from_rankpos_nocalib(
      exact_scores,
      pq_logits,
      token_to_indexed,
      base_mask,
      rank_pos,
      k_take_counts,
      row,
      head,
      ki,
      token,
      heads,
      indexed_count,
      max_rank_take,
      context_len,
      pq_scale);
  const float p = expf(score - row_max[row]) / row_denom[row];
  risk_in[linear] = p * p * code_error[token];
  ids_in[linear] = static_cast<int32_t>(token);
}

template <typename index_t, int DIMS_PER_BLOCK>
__global__ void joint_vprefix_interval_sums_from_mixed_rankpos_nocalib_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ residual,
    const index_t* __restrict__ sorted_ids,
    const int64_t* __restrict__ v_budgets,
    float* __restrict__ interval_sums,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    int64_t sorted_count,
    int64_t v_steps,
    int64_t dim,
    float pq_scale) {
  constexpr int THREADS = 256;
  constexpr int TOKEN_LANES = THREADS / DIMS_PER_BLOCK;
  __shared__ float partial[DIMS_PER_BLOCK][TOKEN_LANES];

  const int64_t dim_tiles = (dim + DIMS_PER_BLOCK - 1) / DIMS_PER_BLOCK;
  int64_t block = static_cast<int64_t>(blockIdx.x);
  const int64_t dim_tile = block % dim_tiles;
  block /= dim_tiles;
  const int64_t vi = block % v_steps;
  const int64_t row = block / v_steps;
  if (row >= rows || vi >= v_steps) {
    return;
  }

  const int lane_dim = static_cast<int>(threadIdx.x % DIMS_PER_BLOCK);
  const int lane_pos = static_cast<int>(threadIdx.x / DIMS_PER_BLOCK);
  const int64_t d = dim_tile * DIMS_PER_BLOCK + lane_dim;
  const int64_t head = row % heads;
  const int64_t ki = row / heads;
  int64_t exact_count = v_budgets[vi];
  exact_count = max(static_cast<int64_t>(0), min(exact_count, context_len));
  exact_count = min(exact_count, sorted_count);
  int64_t prev_count = 0;
  if (vi > 0) {
    prev_count = v_budgets[vi - 1];
    prev_count = max(static_cast<int64_t>(0), min(prev_count, context_len));
    prev_count = min(prev_count, sorted_count);
  }
  const int64_t start_count = exact_count < prev_count ? 0 : prev_count;
  float acc = 0.0f;
  if (d < dim) {
    const float max_v = row_max[row];
    const float denom = row_denom[row];
    for (int64_t j = start_count + lane_pos; j < exact_count; j += TOKEN_LANES) {
      const int64_t token = static_cast<int64_t>(sorted_ids[row * sorted_count + j]);
      if (token >= 0 && token < context_len) {
        const float score = joint_mixed_score_from_rankpos_nocalib(
            exact_scores,
            pq_logits,
            token_to_indexed,
            base_mask,
            rank_pos,
            k_take_counts,
            row,
            head,
            ki,
            token,
            heads,
            indexed_count,
            max_rank_take,
            context_len,
            pq_scale);
        const float p = expf(score - max_v) / denom;
        acc += p * residual[token * dim + d];
      }
    }
  }

  partial[lane_dim][lane_pos] = acc;
  __syncthreads();
  for (int stride = TOKEN_LANES >> 1; stride > 0; stride >>= 1) {
    if (lane_pos < stride) {
      partial[lane_dim][lane_pos] += partial[lane_dim][lane_pos + stride];
    }
    __syncthreads();
  }
  if (lane_pos == 0 && d < dim) {
    interval_sums[(row * v_steps + vi) * dim + d] = partial[lane_dim][0];
  }
}

__global__ void joint_risk_sort_inputs_from_probs_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ code_error,
    float* __restrict__ risk_in,
    int32_t* __restrict__ ids_in,
    int64_t rows,
    int64_t context_len) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * context_len;
  if (linear >= total) {
    return;
  }
  const int64_t token = linear % context_len;
  const float p = probs[linear];
  risk_in[linear] = p * p * code_error[token];
  ids_in[linear] = static_cast<int32_t>(token);
}

template <typename index_t, int DIMS_PER_BLOCK>
__global__ void joint_vprefix_interval_sums_from_mixed_mask_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ selected_mask,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    const float* __restrict__ row_max,
    const float* __restrict__ row_denom,
    const float* __restrict__ residual,
    const index_t* __restrict__ sorted_ids,
    const int64_t* __restrict__ v_budgets,
    float* __restrict__ interval_sums,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    int64_t sorted_count,
    int64_t v_steps,
    int64_t dim,
    float pq_scale) {
  constexpr int THREADS = 256;
  constexpr int TOKEN_LANES = THREADS / DIMS_PER_BLOCK;
  __shared__ float partial[DIMS_PER_BLOCK][TOKEN_LANES];

  const int64_t dim_tiles = (dim + DIMS_PER_BLOCK - 1) / DIMS_PER_BLOCK;
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  const int64_t dim_tile = block % dim_tiles;
  const int64_t tmp = block / dim_tiles;
  const int64_t v_step = tmp % v_steps;
  const int64_t row = tmp / v_steps;
  if (row >= rows) {
    return;
  }
  const int lane_dim = static_cast<int>(threadIdx.x % DIMS_PER_BLOCK);
  const int lane_pos = static_cast<int>(threadIdx.x / DIMS_PER_BLOCK);
  const int64_t d = dim_tile * DIMS_PER_BLOCK + lane_dim;
  const int64_t head = row % heads;
  const int64_t ki = row / heads;
  const int64_t budget = v_budgets[v_step];
  const int64_t exact_count = budget <= 0 ? 0 : (budget < sorted_count ? budget : sorted_count);
  int64_t prev_count = 0;
  if (v_step > 0) {
    const int64_t prev_budget = v_budgets[v_step - 1];
    prev_count = prev_budget <= 0 ? 0 : (prev_budget < sorted_count ? prev_budget : sorted_count);
  }
  const int64_t start_count = exact_count < prev_count ? 0 : prev_count;
  float acc = 0.0f;
  if (d < dim) {
    const float max_v = row_max[row];
    const float denom = row_denom[row];
    for (int64_t pos = start_count + lane_pos; pos < exact_count; pos += TOKEN_LANES) {
      const int64_t token = static_cast<int64_t>(sorted_ids[row * sorted_count + pos]);
      const float score = joint_mixed_score_from_mask(
          exact_scores,
          pq_logits,
          token_to_indexed,
          selected_mask,
          k_take_counts,
          fit_scale,
          fit_bias,
          row,
          head,
          ki,
          token,
          heads,
          indexed_count,
          max_rank_take,
          context_len,
          pq_scale);
      const float p = expf(score - max_v) / denom;
      acc += p * residual[token * dim + d];
    }
  }
  partial[lane_dim][lane_pos] = acc;
  __syncthreads();
  for (int stride = TOKEN_LANES >> 1; stride > 0; stride >>= 1) {
    if (lane_pos < stride) {
      partial[lane_dim][lane_pos] += partial[lane_dim][lane_pos + stride];
    }
    __syncthreads();
  }
  if (lane_pos == 0 && d < dim) {
    interval_sums[(row * v_steps + v_step) * dim + d] = partial[lane_dim][0];
  }
}
