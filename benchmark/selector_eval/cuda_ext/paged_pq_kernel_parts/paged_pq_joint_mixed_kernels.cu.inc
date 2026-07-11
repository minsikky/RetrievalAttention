__global__ void joint_mixed_softmax_rows_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const int64_t* __restrict__ k_take_counts,
    const unsigned char* __restrict__ selected_mask,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    float* __restrict__ probs,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    float pq_scale,
    int64_t context_len) {
  int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows || context_len <= 0) {
    return;
  }

  int64_t head = row % heads;
  int64_t ki = row / heads;
  bool full_exact = k_take_counts[ki] > max_rank_take;
  const int64_t exact_offset = head * context_len;
  const int64_t pq_offset = head * indexed_count;
  const int64_t row_offset = row * context_len;
  const float scale = fit_scale[row];
  const float bias = fit_bias[row];

  extern __shared__ float shared[];
  float local_max = -INFINITY;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    float score = exact_scores[exact_offset + token];
    int pos = token_to_indexed[token];
    if (pos >= 0 && !full_exact && selected_mask[row_offset + token] == static_cast<unsigned char>(0)) {
      score = scale * (pq_logits[pq_offset + static_cast<int64_t>(pos)] * pq_scale) + bias;
    }
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
    float score = exact_scores[exact_offset + token];
    int pos = token_to_indexed[token];
    if (pos >= 0 && !full_exact && selected_mask[row_offset + token] == static_cast<unsigned char>(0)) {
      score = scale * (pq_logits[pq_offset + static_cast<int64_t>(pos)] * pq_scale) + bias;
    }
    float p = expf(score - max_v);
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

__global__ void joint_mixed_softmax_rows_rankpos_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    float* __restrict__ probs,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    float pq_scale,
    int64_t context_len) {
  int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows || context_len <= 0) {
    return;
  }

  int64_t head = row % heads;
  int64_t ki = row / heads;
  int64_t take = k_take_counts[ki];
  bool full_exact = take > max_rank_take;
  const int64_t exact_offset = head * context_len;
  const int64_t pq_offset = head * indexed_count;
  const int64_t rank_offset = head * context_len;
  const int64_t row_offset = row * context_len;
  const float scale = fit_scale[row];
  const float bias = fit_bias[row];

  extern __shared__ float shared[];
  float local_max = -INFINITY;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    float score = exact_scores[exact_offset + token];
    int pos = token_to_indexed[token];
    if (pos >= 0 && !full_exact) {
      bool selected = base_mask[token] != static_cast<unsigned char>(0);
      int rank_i = rank_pos[rank_offset + token];
      selected = selected || (rank_i >= 0 && static_cast<int64_t>(rank_i) < take);
      if (!selected) {
        score = scale * (pq_logits[pq_offset + static_cast<int64_t>(pos)] * pq_scale) + bias;
      }
    }
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
    float score = exact_scores[exact_offset + token];
    int pos = token_to_indexed[token];
    if (pos >= 0 && !full_exact) {
      bool selected = base_mask[token] != static_cast<unsigned char>(0);
      int rank_i = rank_pos[rank_offset + token];
      selected = selected || (rank_i >= 0 && static_cast<int64_t>(rank_i) < take);
      if (!selected) {
        score = scale * (pq_logits[pq_offset + static_cast<int64_t>(pos)] * pq_scale) + bias;
      }
    }
    float p = expf(score - max_v);
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

__global__ void joint_mixed_score_grid_apply_indexed_rankpos_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int64_t* __restrict__ indexed_tokens,
    const unsigned char* __restrict__ base_mask,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    float* __restrict__ score_grid,
    int64_t k_count,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    float pq_scale,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * indexed_count;
  if (linear >= total) {
    return;
  }
  int64_t j = linear % indexed_count;
  int64_t tmp = linear / indexed_count;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t token = indexed_tokens[j];
  if (token < 0 || token >= context_len) {
    return;
  }
  int64_t kh = ki * heads + head;
  int64_t take = k_take_counts[ki];
  bool full_exact = take > max_rank_take;
  bool selected = base_mask[token] != static_cast<unsigned char>(0);
  int pos = rank_pos[head * context_len + token];
  selected = selected || (pos >= 0 && static_cast<int64_t>(pos) < take);
  if (full_exact || selected) {
    score_grid[kh * context_len + token] = exact_scores[head * context_len + token];
  } else {
    float calibrated = fit_scale[kh] * (pq_logits[head * indexed_count + j] * pq_scale) + fit_bias[kh];
    score_grid[kh * context_len + token] = calibrated;
  }
}

__global__ void joint_mixed_score_grid_fill_rankpos_nocalib_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    float* __restrict__ score_grid,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t indexed_count,
    int64_t max_rank_take,
    float pq_scale) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * context_len;
  if (linear >= total) {
    return;
  }
  int64_t token = linear % context_len;
  int64_t tmp = linear / context_len;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t take = k_take_counts[ki];
  bool full_exact = take > max_rank_take;
  int pos = token_to_indexed[token];
  bool selected = full_exact || pos < 0 || base_mask[token] != static_cast<unsigned char>(0);
  int rank_i = rank_pos[head * context_len + token];
  selected = selected || (rank_i >= 0 && static_cast<int64_t>(rank_i) < take);
  if (selected) {
    score_grid[linear] = exact_scores[head * context_len + token];
  } else {
    score_grid[linear] = pq_logits[head * indexed_count + static_cast<int64_t>(pos)] * pq_scale;
  }
}

__global__ void joint_precision_score_maps_init_kernel(
    int32_t* __restrict__ token_to_indexed,
    unsigned char* __restrict__ base_mask,
    int32_t* __restrict__ rank_pos,
    int64_t heads,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * context_len;
  if (linear >= total) {
    return;
  }
  rank_pos[linear] = -1;
  if (linear < context_len) {
    token_to_indexed[linear] = -1;
    base_mask[linear] = 0;
  }
}

__global__ void joint_precision_score_maps_scatter_kernel(
    int32_t* __restrict__ token_to_indexed,
    unsigned char* __restrict__ base_mask,
    int32_t* __restrict__ rank_pos,
    const int64_t* __restrict__ indexed_tokens,
    const int64_t* __restrict__ base_tokens,
    const int64_t* __restrict__ ranked_prefix_tokens,
    int64_t indexed_count,
    int64_t base_count,
    int64_t heads,
    int64_t rank_count,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t rank_total = heads * rank_count;
  int64_t total = indexed_count + base_count + rank_total;
  if (linear >= total) {
    return;
  }
  if (linear < indexed_count) {
    int64_t token = indexed_tokens[linear];
    if (token >= 0 && token < context_len) {
      token_to_indexed[token] = static_cast<int32_t>(linear);
    }
    return;
  }
  linear -= indexed_count;
  if (linear < base_count) {
    int64_t token = base_tokens[linear];
    if (token >= 0 && token < context_len) {
      base_mask[token] = 1;
    }
    return;
  }
  linear -= base_count;
  int64_t head = linear / rank_count;
  int64_t pos = linear - head * rank_count;
  int64_t token = ranked_prefix_tokens[head * rank_count + pos];
  if (token >= 0 && token < context_len) {
    rank_pos[head * context_len + token] = static_cast<int32_t>(pos);
  }
}

__global__ void joint_precision_score_grid_fill_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const float* __restrict__ scores_lo,
    const int32_t* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int32_t* __restrict__ rank_pos,
    const int64_t* __restrict__ take_counts,
    const int64_t* __restrict__ hi_counts,
    float* __restrict__ output,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t indexed_count) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * context_len;
  if (linear >= total) {
    return;
  }
  int64_t token = linear % context_len;
  int64_t kh = linear / context_len;
  int64_t head = kh % heads;
  int64_t ki = kh / heads;
  float exact = exact_scores[head * context_len + token];
  int32_t indexed_pos = token_to_indexed[token];
  float value = indexed_pos >= 0 && indexed_pos < indexed_count
      ? pq_logits[head * indexed_count + indexed_pos]
      : exact;
  if (base_mask[token]) {
    value = exact;
  }
  int32_t pos = rank_pos[head * context_len + token];
  if (pos >= 0) {
    if (pos < hi_counts[ki]) {
      value = exact;
    } else if (pos < take_counts[ki]) {
      value = scores_lo[head * context_len + token];
    }
  }
  output[linear] = value;
}

__global__ void joint_mixed_score_grid_fill_pq_or_full_exact_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const int64_t* __restrict__ k_take_counts,
    float* __restrict__ score_grid,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t indexed_count,
    int64_t max_rank_take,
    float pq_scale) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * context_len;
  if (linear >= total) {
    return;
  }
  int64_t token = linear % context_len;
  int64_t tmp = linear / context_len;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t take = k_take_counts[ki];
  bool full_exact = take > max_rank_take;
  int pos = token_to_indexed[token];
  if (full_exact || pos < 0) {
    score_grid[linear] = exact_scores[head * context_len + token];
  } else {
    score_grid[linear] = pq_logits[head * indexed_count + static_cast<int64_t>(pos)] * pq_scale;
  }
}

__global__ void joint_mixed_score_grid_fill_exact_kernel(
    const float* __restrict__ exact_scores,
    float* __restrict__ score_grid,
    int64_t k_count,
    int64_t heads,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * context_len;
  if (linear >= total) {
    return;
  }
  int64_t token = linear % context_len;
  int64_t head = (linear / context_len) % heads;
  score_grid[linear] = exact_scores[head * context_len + token];
}

__global__ void joint_mixed_score_grid_apply_indexed_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int64_t* __restrict__ indexed_tokens,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    float* __restrict__ score_grid,
    int64_t k_count,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    float pq_scale,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * indexed_count;
  if (linear >= total) {
    return;
  }
  int64_t j = linear % indexed_count;
  int64_t tmp = linear / indexed_count;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t token = indexed_tokens[j];
  if (token >= 0 && token < context_len) {
    int64_t kh = ki * heads + head;
    if (k_take_counts[ki] > max_rank_take) {
      score_grid[kh * context_len + token] = exact_scores[head * context_len + token];
    } else {
      float calibrated = fit_scale[kh] * (pq_logits[head * indexed_count + j] * pq_scale) + fit_bias[kh];
      score_grid[kh * context_len + token] = calibrated;
    }
  }
}

__global__ void joint_mixed_score_grid_apply_indexed_calibrated_kernel(
    const float* __restrict__ pq_logits,
    const int64_t* __restrict__ indexed_tokens,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    float* __restrict__ score_grid,
    int64_t k_count,
    int64_t heads,
    int64_t indexed_count,
    float pq_scale,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * indexed_count;
  if (linear >= total) {
    return;
  }
  int64_t j = linear % indexed_count;
  int64_t tmp = linear / indexed_count;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t token = indexed_tokens[j];
  if (token >= 0 && token < context_len) {
    int64_t kh = ki * heads + head;
    score_grid[kh * context_len + token] =
        fit_scale[kh] * (pq_logits[head * indexed_count + j] * pq_scale) + fit_bias[kh];
  }
}

__global__ void joint_mixed_score_grid_apply_base_exact_kernel(
    const float* __restrict__ exact_scores,
    const int64_t* __restrict__ base_tokens,
    float* __restrict__ score_grid,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t base_count) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * base_count;
  if (linear >= total) {
    return;
  }
  int64_t base_i = linear % base_count;
  int64_t tmp = linear / base_count;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t token = base_tokens[base_i];
  if (token >= 0 && token < context_len) {
    score_grid[(ki * heads + head) * context_len + token] = exact_scores[head * context_len + token];
  }
}

__global__ void joint_mixed_score_grid_apply_base_logits_kernel(
    const int64_t* __restrict__ base_tokens,
    const float* __restrict__ base_logits,
    float* __restrict__ score_grid,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t base_count) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * base_count;
  if (linear >= total) {
    return;
  }
  int64_t base_i = linear % base_count;
  int64_t tmp = linear / base_count;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t token = base_tokens[base_i];
  if (token >= 0 && token < context_len) {
    score_grid[(ki * heads + head) * context_len + token] = base_logits[head * base_count + base_i];
  }
}

__global__ void joint_mixed_score_grid_apply_ranked_exact_kernel(
    const float* __restrict__ exact_scores,
    const int64_t* __restrict__ ranked_prefix_tokens,
    const int64_t* __restrict__ k_take_counts,
    float* __restrict__ score_grid,
    int64_t k_count,
    int64_t heads,
    int64_t max_rank_take,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * max_rank_take;
  if (linear >= total) {
    return;
  }
  int64_t rank_i = linear % max_rank_take;
  int64_t tmp = linear / max_rank_take;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t take = k_take_counts[ki];
  if (rank_i >= take) {
    return;
  }
  int64_t token = ranked_prefix_tokens[head * max_rank_take + rank_i];
  if (token >= 0 && token < context_len) {
    score_grid[(ki * heads + head) * context_len + token] = exact_scores[head * context_len + token];
  }
}

__global__ void joint_mixed_score_grid_apply_ranked_logits_kernel(
    const int64_t* __restrict__ ranked_prefix_tokens,
    const float* __restrict__ ranked_logits,
    const int64_t* __restrict__ k_take_counts,
    float* __restrict__ score_grid,
    int64_t k_count,
    int64_t heads,
    int64_t max_rank_take,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * max_rank_take;
  if (linear >= total) {
    return;
  }
  int64_t rank_i = linear % max_rank_take;
  int64_t tmp = linear / max_rank_take;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t take = k_take_counts[ki];
  if (rank_i >= take) {
    return;
  }
  int64_t token = ranked_prefix_tokens[head * max_rank_take + rank_i];
  if (token >= 0 && token < context_len) {
    score_grid[(ki * heads + head) * context_len + token] =
        ranked_logits[head * max_rank_take + rank_i];
  }
}

__device__ __forceinline__ float joint_mixed_score_from_sparse_rankpos(
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const int* __restrict__ base_pos,
    const float* __restrict__ base_logits,
    const int* __restrict__ rank_pos,
    const float* __restrict__ ranked_logits,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    int64_t row,
    int64_t head,
    int64_t ki,
    int64_t token,
    int64_t indexed_count,
    int64_t base_count,
    int64_t max_rank_take,
    int64_t context_len,
    float pq_scale) {
  const int base_i = base_pos[token];
  if (base_i >= 0) {
    return base_logits[head * base_count + static_cast<int64_t>(base_i)];
  }
  const int rank_i = rank_pos[head * context_len + token];
  if (rank_i >= 0 && static_cast<int64_t>(rank_i) < k_take_counts[ki]) {
    return ranked_logits[head * max_rank_take + static_cast<int64_t>(rank_i)];
  }
  const int pos = token_to_indexed[token];
  if (pos >= 0) {
    return fit_scale[row] * (pq_logits[head * indexed_count + static_cast<int64_t>(pos)] * pq_scale) + fit_bias[row];
  }
  return 0.0f;
}

__global__ void joint_mixed_sparse_softmax_rows_kernel(
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const int* __restrict__ base_pos,
    const float* __restrict__ base_logits,
    const int* __restrict__ rank_pos,
    const float* __restrict__ ranked_logits,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ fit_scale,
    const float* __restrict__ fit_bias,
    float* __restrict__ probs,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t base_count,
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
    const float score = joint_mixed_score_from_sparse_rankpos(
        pq_logits,
        token_to_indexed,
        base_pos,
        base_logits,
        rank_pos,
        ranked_logits,
        k_take_counts,
        fit_scale,
        fit_bias,
        row,
        head,
        ki,
        token,
        indexed_count,
        base_count,
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
    const float score = joint_mixed_score_from_sparse_rankpos(
        pq_logits,
        token_to_indexed,
        base_pos,
        base_logits,
        rank_pos,
        ranked_logits,
        k_take_counts,
        fit_scale,
        fit_bias,
        row,
        head,
        ki,
        token,
        indexed_count,
        base_count,
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

__global__ void joint_sparse_exact_score_table_base_kernel(
    const int64_t* __restrict__ base_tokens,
    const float* __restrict__ base_logits,
    float* __restrict__ exact_scores,
    int64_t heads,
    int64_t base_count,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * base_count;
  if (linear >= total) {
    return;
  }
  int64_t base_i = linear % base_count;
  int64_t head = linear / base_count;
  int64_t token = base_tokens[base_i];
  if (token >= 0 && token < context_len) {
    exact_scores[head * context_len + token] = base_logits[linear];
  }
}

__global__ void joint_sparse_exact_score_table_ranked_kernel(
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_logits,
    float* __restrict__ exact_scores,
    int64_t heads,
    int64_t max_rank_take,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * max_rank_take;
  if (linear >= total) {
    return;
  }
  int64_t rank_i = linear % max_rank_take;
  int64_t head = linear / max_rank_take;
  int64_t token = ranked_tokens[head * max_rank_take + rank_i];
  if (token >= 0 && token < context_len) {
    exact_scores[head * context_len + token] = ranked_logits[linear];
  }
}

}  // namespace
