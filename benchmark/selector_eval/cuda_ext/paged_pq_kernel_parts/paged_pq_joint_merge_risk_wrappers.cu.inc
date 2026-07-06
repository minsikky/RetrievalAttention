__device__ __forceinline__ float joint_merge_base_score_nocalib(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    int64_t head,
    int64_t token,
    int64_t indexed_count,
    int64_t context_len,
    float pq_scale) {
  const int pos = token_to_indexed[token];
  if (pos < 0 || base_mask[token] != static_cast<unsigned char>(0)) {
    return exact_scores[head * context_len + token];
  }
  return pq_logits[head * indexed_count + static_cast<int64_t>(pos)] * pq_scale;
}

__global__ void joint_merge_ref_stats_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int64_t* __restrict__ ranked_prefix_tokens,
    float* __restrict__ head_ref,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    float pq_scale) {
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads || context_len <= 0) {
    return;
  }
  extern __shared__ float shared[];
  float local_max = -INFINITY;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    local_max = fmaxf(
        local_max,
        joint_merge_base_score_nocalib(
            exact_scores,
            pq_logits,
            token_to_indexed,
            base_mask,
            head,
            token,
            indexed_count,
            context_len,
            pq_scale));
  }
  for (int64_t j = threadIdx.x; j < max_rank_take; j += blockDim.x) {
    const int64_t token = ranked_prefix_tokens[head * max_rank_take + j];
    if (token >= 0 && token < context_len) {
      local_max = fmaxf(local_max, exact_scores[head * context_len + token]);
    }
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    head_ref[head] = shared[0];
  }
}

__global__ void joint_merge_base_sum_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const float* __restrict__ head_ref,
    float* __restrict__ base_sum,
    int64_t heads,
    int64_t indexed_count,
    int64_t context_len,
    float pq_scale) {
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads || context_len <= 0) {
    return;
  }
  extern __shared__ float shared[];
  const float ref = head_ref[head];
  float local_sum = 0.0f;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    const float score = joint_merge_base_score_nocalib(
        exact_scores,
        pq_logits,
        token_to_indexed,
        base_mask,
        head,
        token,
        indexed_count,
        context_len,
        pq_scale);
    local_sum += expf(score - ref);
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
    base_sum[head] = fmaxf(shared[0], 1.0e-20f);
  }
}

template <int DIMS_PER_BLOCK>
__global__ void joint_merge_base_num_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const float* __restrict__ head_ref,
    const float* __restrict__ vhat,
    float* __restrict__ base_num,
    int64_t heads,
    int64_t indexed_count,
    int64_t context_len,
    int64_t dim,
    float pq_scale) {
  constexpr int THREADS = 256;
  constexpr int TOKEN_LANES = THREADS / DIMS_PER_BLOCK;
  __shared__ float partial[DIMS_PER_BLOCK][TOKEN_LANES];

  const int64_t dim_tiles = (dim + DIMS_PER_BLOCK - 1) / DIMS_PER_BLOCK;
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  const int64_t dim_tile = block % dim_tiles;
  const int64_t head = block / dim_tiles;
  if (head >= heads) {
    return;
  }
  const int lane_dim = static_cast<int>(threadIdx.x % DIMS_PER_BLOCK);
  const int lane_token = static_cast<int>(threadIdx.x / DIMS_PER_BLOCK);
  const int64_t d = dim_tile * DIMS_PER_BLOCK + lane_dim;
  const float ref = head_ref[head];
  float acc = 0.0f;
  if (d < dim) {
    for (int64_t token = lane_token; token < context_len; token += TOKEN_LANES) {
      const float score = joint_merge_base_score_nocalib(
          exact_scores,
          pq_logits,
          token_to_indexed,
          base_mask,
          head,
          token,
          indexed_count,
          context_len,
          pq_scale);
      acc += expf(score - ref) * vhat[token * dim + d];
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
    base_num[head * dim + d] = partial[lane_dim][0];
  }
}

__global__ void joint_merge_base_risk_inputs_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const float* __restrict__ code_error,
    const float* __restrict__ head_ref,
    float* __restrict__ risk_in,
    int32_t* __restrict__ ids_in,
    int64_t heads,
    int64_t indexed_count,
    int64_t context_len,
    float pq_scale) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = heads * context_len;
  if (linear >= total) {
    return;
  }
  const int64_t token = linear % context_len;
  const int64_t head = linear / context_len;
  const float score = joint_merge_base_score_nocalib(
      exact_scores,
      pq_logits,
      token_to_indexed,
      base_mask,
      head,
      token,
      indexed_count,
      context_len,
      pq_scale);
  const float w = expf(2.0f * (score - head_ref[head]));
  risk_in[linear] = w * code_error[token];
  ids_in[linear] = static_cast<int32_t>(token);
}

__global__ void joint_merge_exact_risk_inputs_kernel(
    const float* __restrict__ exact_scores,
    const int64_t* __restrict__ ranked_prefix_tokens,
    const float* __restrict__ code_error,
    const float* __restrict__ head_ref,
    float* __restrict__ risk_in,
    int32_t* __restrict__ ids_in,
    int64_t heads,
    int64_t max_rank_take,
    int64_t context_len) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = heads * max_rank_take;
  if (linear >= total) {
    return;
  }
  const int64_t j = linear % max_rank_take;
  const int64_t head = linear / max_rank_take;
  const int64_t token = ranked_prefix_tokens[head * max_rank_take + j];
  if (token >= 0 && token < context_len) {
    const float score = exact_scores[head * context_len + token];
    const float w = expf(2.0f * (score - head_ref[head]));
    risk_in[linear] = w * code_error[token];
    ids_in[linear] = static_cast<int32_t>(token);
  } else {
    risk_in[linear] = -INFINITY;
    ids_in[linear] = 0;
  }
}

__global__ void joint_merge_row_denoms_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int64_t* __restrict__ ranked_prefix_tokens,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ head_ref,
    const float* __restrict__ base_sum,
    float* __restrict__ row_max,
    float* __restrict__ row_denom,
    int64_t rows,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    float pq_scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t head = row % heads;
  const int64_t ki = row / heads;
  int64_t take = k_take_counts[ki];
  take = max(static_cast<int64_t>(0), min(take, max_rank_take));
  float denom = base_sum[head];
  const float ref = head_ref[head];
  for (int64_t j = threadIdx.x; j < take; j += blockDim.x) {
    const int64_t token = ranked_prefix_tokens[head * max_rank_take + j];
    if (token >= 0 && token < context_len) {
      const float exact_score = exact_scores[head * context_len + token];
      const float base_score = joint_merge_base_score_nocalib(
          exact_scores,
          pq_logits,
          token_to_indexed,
          base_mask,
          head,
          token,
          indexed_count,
          context_len,
          pq_scale);
      denom += expf(exact_score - ref) - expf(base_score - ref);
    }
  }
  extern __shared__ float shared[];
  shared[threadIdx.x] = denom - base_sum[head];
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_max[row] = ref;
    row_denom[row] = fmaxf(base_sum[head] + shared[0], 1.0e-20f);
  }
}

template <int DIMS_PER_BLOCK>
__global__ void joint_merge_base_outputs_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int64_t* __restrict__ ranked_prefix_tokens,
    const int64_t* __restrict__ k_take_counts,
    const float* __restrict__ head_ref,
    const float* __restrict__ base_num,
    const float* __restrict__ row_denom,
    const float* __restrict__ vhat,
    float* __restrict__ base_outputs,
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
  int64_t take = k_take_counts[ki];
  take = max(static_cast<int64_t>(0), min(take, max_rank_take));
  const float ref = head_ref[head];
  float acc = 0.0f;
  if (d < dim) {
    for (int64_t j = lane_token; j < take; j += TOKEN_LANES) {
      const int64_t token = ranked_prefix_tokens[head * max_rank_take + j];
      if (token >= 0 && token < context_len) {
        const float exact_score = exact_scores[head * context_len + token];
        const float base_score = joint_merge_base_score_nocalib(
            exact_scores,
            pq_logits,
            token_to_indexed,
            base_mask,
            head,
            token,
            indexed_count,
            context_len,
            pq_scale);
        const float delta = expf(exact_score - ref) - expf(base_score - ref);
        acc += delta * vhat[token * dim + d];
      }
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
    base_outputs[row * dim + d] =
        (base_num[head * dim + d] + partial[lane_dim][0]) / row_denom[row];
  }
}

__global__ void joint_merge_risk_order_kernel(
    const float* __restrict__ base_risk_sorted,
    const int32_t* __restrict__ base_ids_sorted,
    const float* __restrict__ exact_risk_sorted,
    const int32_t* __restrict__ exact_ids_sorted,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    int32_t* __restrict__ merged_ids,
    int64_t rows,
    int64_t heads,
    int64_t context_len,
    int64_t max_rank_take,
    int64_t max_exact) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows || max_exact <= 0) {
    return;
  }
  const int64_t head = row % heads;
  const int64_t ki = row / heads;
  int64_t take = k_take_counts[ki];
  take = max(static_cast<int64_t>(0), min(take, max_rank_take));
  int64_t base_pos = 0;
  int64_t exact_pos = 0;
  int64_t out_pos = 0;
  while (out_pos < max_exact && (base_pos < context_len || exact_pos < max_rank_take)) {
    while (base_pos < context_len) {
      const int64_t token = static_cast<int64_t>(base_ids_sorted[head * context_len + base_pos]);
      const int rank = (token >= 0 && token < context_len) ? rank_pos[head * context_len + token] : -1;
      if (!(rank >= 0 && static_cast<int64_t>(rank) < take)) {
        break;
      }
      ++base_pos;
    }
    while (exact_pos < max_rank_take) {
      const int64_t token = static_cast<int64_t>(exact_ids_sorted[head * max_rank_take + exact_pos]);
      const int rank = (token >= 0 && token < context_len) ? rank_pos[head * context_len + token] : -1;
      if (rank >= 0 && static_cast<int64_t>(rank) < take) {
        break;
      }
      ++exact_pos;
    }
    if (base_pos >= context_len && exact_pos >= max_rank_take) {
      break;
    }
    const float base_score =
        (base_pos < context_len) ? base_risk_sorted[head * context_len + base_pos] : -INFINITY;
    const float exact_score =
        (exact_pos < max_rank_take) ? exact_risk_sorted[head * max_rank_take + exact_pos] : -INFINITY;
    if (exact_score > base_score) {
      merged_ids[row * max_exact + out_pos] = exact_ids_sorted[head * max_rank_take + exact_pos];
      ++exact_pos;
    } else {
      merged_ids[row * max_exact + out_pos] = base_ids_sorted[head * context_len + base_pos];
      ++base_pos;
    }
    ++out_pos;
  }
  while (out_pos < max_exact) {
    merged_ids[row * max_exact + out_pos] = 0;
    ++out_pos;
  }
}

__global__ void joint_merge_grouped_indexed_token_to_pos_kernel(
    int* __restrict__ token_to_indexed,
    const int64_t* __restrict__ indexed_tokens,
    int64_t groups,
    int64_t indexed_count,
    int64_t context_len) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = groups * indexed_count;
  if (linear >= total) {
    return;
  }
  const int64_t group = linear / indexed_count;
  const int64_t pos = linear % indexed_count;
  const int64_t token = indexed_tokens[group * indexed_count + pos];
  if (token >= 0 && token < context_len) {
    token_to_indexed[group * context_len + token] = static_cast<int>(pos);
  }
}

__global__ void joint_merge_grouped_base_mask_set_kernel(
    unsigned char* __restrict__ base_mask,
    const int64_t* __restrict__ base_tokens,
    int64_t groups,
    int64_t base_count,
    int64_t context_len) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = groups * base_count;
  if (linear >= total) {
    return;
  }
  const int64_t group = linear / base_count;
  const int64_t pos = linear % base_count;
  const int64_t token = base_tokens[group * base_count + pos];
  if (token >= 0 && token < context_len) {
    base_mask[group * context_len + token] = static_cast<unsigned char>(1);
  }
}

__global__ void joint_merge_grouped_rank_position_scatter_kernel(
    int* __restrict__ rank_pos,
    const int64_t* __restrict__ ranked_prefix_tokens,
    int64_t groups,
    int64_t heads,
    int64_t max_rank_take,
    int64_t context_len) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = groups * heads * max_rank_take;
  if (linear >= total) {
    return;
  }
  const int64_t j = linear % max_rank_take;
  const int64_t head = (linear / max_rank_take) % heads;
  const int64_t group = linear / (heads * max_rank_take);
  const int64_t token = ranked_prefix_tokens[(group * heads + head) * max_rank_take + j];
  if (token >= 0 && token < context_len) {
    rank_pos[(group * heads + head) * context_len + token] = static_cast<int>(j);
  }
}

__device__ __forceinline__ float joint_merge_grouped_base_score_nocalib(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    int64_t group,
    int64_t head,
    int64_t token,
    int64_t heads,
    int64_t indexed_count,
    int64_t context_len,
    float pq_scale) {
  const int pos = token_to_indexed[group * context_len + token];
  if (pos < 0 || base_mask[group * context_len + token] != static_cast<unsigned char>(0)) {
    return exact_scores[(group * heads + head) * context_len + token];
  }
  return pq_logits[(group * heads + head) * indexed_count + static_cast<int64_t>(pos)] * pq_scale;
}

__global__ void joint_merge_grouped_ref_stats_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const int64_t* __restrict__ ranked_prefix_tokens,
    float* __restrict__ head_ref,
    int64_t groups,
    int64_t heads,
    int64_t indexed_count,
    int64_t max_rank_take,
    int64_t context_len,
    float pq_scale) {
  const int64_t segment = static_cast<int64_t>(blockIdx.x);
  const int64_t group = segment / heads;
  const int64_t head = segment % heads;
  if (group >= groups || context_len <= 0) {
    return;
  }
  extern __shared__ float shared[];
  float local_max = -INFINITY;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    local_max = fmaxf(
        local_max,
        joint_merge_grouped_base_score_nocalib(
            exact_scores,
            pq_logits,
            token_to_indexed,
            base_mask,
            group,
            head,
            token,
            heads,
            indexed_count,
            context_len,
            pq_scale));
  }
  for (int64_t j = threadIdx.x; j < max_rank_take; j += blockDim.x) {
    const int64_t token = ranked_prefix_tokens[(group * heads + head) * max_rank_take + j];
    if (token >= 0 && token < context_len) {
      local_max = fmaxf(local_max, exact_scores[(group * heads + head) * context_len + token]);
    }
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    head_ref[segment] = shared[0];
  }
}

__global__ void joint_merge_grouped_base_risk_inputs_kernel(
    const float* __restrict__ exact_scores,
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const unsigned char* __restrict__ base_mask,
    const float* __restrict__ code_error_groups,
    const float* __restrict__ head_ref,
    float* __restrict__ risk_in,
    int32_t* __restrict__ ids_in,
    int64_t groups,
    int64_t heads,
    int64_t indexed_count,
    int64_t context_len,
    float pq_scale) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = groups * heads * context_len;
  if (linear >= total) {
    return;
  }
  const int64_t token = linear % context_len;
  const int64_t head = (linear / context_len) % heads;
  const int64_t group = linear / (heads * context_len);
  const int64_t segment = group * heads + head;
  const float score = joint_merge_grouped_base_score_nocalib(
      exact_scores,
      pq_logits,
      token_to_indexed,
      base_mask,
      group,
      head,
      token,
      heads,
      indexed_count,
      context_len,
      pq_scale);
  const float w = expf(2.0f * (score - head_ref[segment]));
  risk_in[linear] = w * code_error_groups[group * context_len + token];
  ids_in[linear] = static_cast<int32_t>(token);
}

__global__ void joint_merge_grouped_exact_risk_inputs_kernel(
    const float* __restrict__ exact_scores,
    const int64_t* __restrict__ ranked_prefix_tokens,
    const float* __restrict__ code_error_groups,
    const float* __restrict__ head_ref,
    float* __restrict__ risk_in,
    int32_t* __restrict__ ids_in,
    int64_t groups,
    int64_t heads,
    int64_t max_rank_take,
    int64_t context_len) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = groups * heads * max_rank_take;
  if (linear >= total) {
    return;
  }
  const int64_t j = linear % max_rank_take;
  const int64_t head = (linear / max_rank_take) % heads;
  const int64_t group = linear / (heads * max_rank_take);
  const int64_t segment = group * heads + head;
  const int64_t token = ranked_prefix_tokens[(group * heads + head) * max_rank_take + j];
  if (token >= 0 && token < context_len) {
    const float score = exact_scores[(group * heads + head) * context_len + token];
    const float w = expf(2.0f * (score - head_ref[segment]));
    risk_in[linear] = w * code_error_groups[group * context_len + token];
    ids_in[linear] = static_cast<int32_t>(token);
  } else {
    risk_in[linear] = -INFINITY;
    ids_in[linear] = 0;
  }
}

__global__ void joint_merge_grouped_risk_order_kernel(
    const float* __restrict__ base_risk_sorted,
    const int32_t* __restrict__ base_ids_sorted,
    const float* __restrict__ exact_risk_sorted,
    const int32_t* __restrict__ exact_ids_sorted,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    int32_t* __restrict__ merged_ids,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t max_rank_take,
    int64_t max_exact) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t rows = groups * k_count * heads;
  if (row >= rows || max_exact <= 0) {
    return;
  }
  const int64_t head = row % heads;
  const int64_t ki = (row / heads) % k_count;
  const int64_t group = row / (k_count * heads);
  const int64_t segment = group * heads + head;
  int64_t take = k_take_counts[ki];
  take = max(static_cast<int64_t>(0), min(take, max_rank_take));
  int64_t base_pos = 0;
  int64_t exact_pos = 0;
  int64_t out_pos = 0;
  while (out_pos < max_exact && (base_pos < context_len || exact_pos < max_rank_take)) {
    while (base_pos < context_len) {
      const int64_t token = static_cast<int64_t>(base_ids_sorted[segment * context_len + base_pos]);
      const int rank =
          (token >= 0 && token < context_len) ? rank_pos[segment * context_len + token] : -1;
      if (!(rank >= 0 && static_cast<int64_t>(rank) < take)) {
        break;
      }
      ++base_pos;
    }
    while (exact_pos < max_rank_take) {
      const int64_t token = static_cast<int64_t>(exact_ids_sorted[segment * max_rank_take + exact_pos]);
      const int rank =
          (token >= 0 && token < context_len) ? rank_pos[segment * context_len + token] : -1;
      if (rank >= 0 && static_cast<int64_t>(rank) < take) {
        break;
      }
      ++exact_pos;
    }
    if (base_pos >= context_len && exact_pos >= max_rank_take) {
      break;
    }
    const float base_score =
        (base_pos < context_len) ? base_risk_sorted[segment * context_len + base_pos] : -INFINITY;
    const float exact_score =
        (exact_pos < max_rank_take) ? exact_risk_sorted[segment * max_rank_take + exact_pos] : -INFINITY;
    if (exact_score > base_score) {
      merged_ids[row * max_exact + out_pos] = exact_ids_sorted[segment * max_rank_take + exact_pos];
      ++exact_pos;
    } else {
      merged_ids[row * max_exact + out_pos] = base_ids_sorted[segment * context_len + base_pos];
      ++base_pos;
    }
    ++out_pos;
  }
  while (out_pos < max_exact) {
    merged_ids[row * max_exact + out_pos] = 0;
    ++out_pos;
  }
}

torch::Tensor joint_vprefix_outputs_from_grouped_merge_risk_batched_cuda(
    torch::Tensor base_outputs,
    torch::Tensor probs,
    torch::Tensor residual_groups,
    torch::Tensor code_error_groups,
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    torch::Tensor v_budgets,
    double pq_scale) {
  TORCH_CHECK(base_outputs.is_cuda(), "base_outputs must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(residual_groups.is_cuda(), "residual_groups must be CUDA");
  TORCH_CHECK(code_error_groups.is_cuda(), "code_error_groups must be CUDA");
  TORCH_CHECK(exact_scores.is_cuda(), "exact_scores must be CUDA");
  TORCH_CHECK(pq_logits.is_cuda(), "pq_logits must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(base_tokens.is_cuda(), "base_tokens must be CUDA");
  TORCH_CHECK(ranked_prefix_tokens.is_cuda(), "ranked_prefix_tokens must be CUDA");
  TORCH_CHECK(k_take_counts.is_cuda(), "k_take_counts must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(base_outputs.scalar_type() == torch::kFloat32, "base_outputs must be float32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(residual_groups.scalar_type() == torch::kFloat32, "residual_groups must be float32");
  TORCH_CHECK(code_error_groups.scalar_type() == torch::kFloat32, "code_error_groups must be float32");
  TORCH_CHECK(exact_scores.scalar_type() == torch::kFloat32, "exact_scores must be float32");
  TORCH_CHECK(pq_logits.scalar_type() == torch::kFloat32, "pq_logits must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(base_tokens.scalar_type() == torch::kLong, "base_tokens must be int64");
  TORCH_CHECK(ranked_prefix_tokens.scalar_type() == torch::kLong, "ranked_prefix_tokens must be int64");
  TORCH_CHECK(k_take_counts.scalar_type() == torch::kLong, "k_take_counts must be int64");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(base_outputs.dim() == 4, "base_outputs shape must be [groups, k, heads, dim]");
  TORCH_CHECK(probs.dim() == 4, "probs shape must be [groups, k, heads, context]");
  TORCH_CHECK(residual_groups.dim() == 3, "residual_groups shape must be [groups, context, dim]");
  TORCH_CHECK(code_error_groups.dim() == 2, "code_error_groups shape must be [groups, context]");
  TORCH_CHECK(exact_scores.dim() == 3, "exact_scores shape must be [groups, heads, context]");
  TORCH_CHECK(pq_logits.dim() == 3, "pq_logits shape must be [groups, heads, indexed]");
  TORCH_CHECK(indexed_tokens.dim() == 2, "indexed_tokens shape must be [groups, indexed]");
  TORCH_CHECK(base_tokens.dim() == 2, "base_tokens shape must be [groups, base_count]");
  TORCH_CHECK(ranked_prefix_tokens.dim() == 3, "ranked_prefix_tokens shape must be [groups, heads, max_rank_take]");
  TORCH_CHECK(k_take_counts.dim() == 1, "k_take_counts shape must be [k_count]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v_steps]");

  const auto groups = base_outputs.size(0);
  const auto k_count = base_outputs.size(1);
  const auto heads = base_outputs.size(2);
  const auto dim = base_outputs.size(3);
  const auto context_len = probs.size(3);
  const auto indexed_count = indexed_tokens.size(1);
  const auto base_count = base_tokens.size(1);
  const auto max_rank_take = ranked_prefix_tokens.size(2);
  const auto v_steps = v_budgets.size(0);
  const auto rows = groups * k_count * heads;
  TORCH_CHECK(probs.size(0) == groups && probs.size(1) == k_count && probs.size(2) == heads,
              "probs/base shape mismatch");
  TORCH_CHECK(residual_groups.size(0) == groups && residual_groups.size(1) == context_len &&
              residual_groups.size(2) == dim,
              "residual_groups shape mismatch");
  TORCH_CHECK(code_error_groups.size(0) == groups && code_error_groups.size(1) == context_len,
              "code_error_groups shape mismatch");
  TORCH_CHECK(exact_scores.size(0) == groups && exact_scores.size(1) == heads &&
              exact_scores.size(2) == context_len,
              "exact_scores shape mismatch");
  TORCH_CHECK(pq_logits.size(0) == groups && pq_logits.size(1) == heads &&
              pq_logits.size(2) == indexed_count,
              "pq_logits shape mismatch");
  TORCH_CHECK(indexed_tokens.size(0) == groups, "indexed_tokens group mismatch");
  TORCH_CHECK(base_tokens.size(0) == groups, "base_tokens group mismatch");
  TORCH_CHECK(ranked_prefix_tokens.size(0) == groups && ranked_prefix_tokens.size(1) == heads,
              "ranked_prefix_tokens shape mismatch");
  TORCH_CHECK(k_take_counts.size(0) == k_count, "k_take_counts length mismatch");

  auto outputs = torch::empty({rows, v_steps, dim}, base_outputs.options().dtype(torch::kFloat32));
  if (rows == 0 || dim == 0 || v_steps == 0) {
    return outputs;
  }
  auto base_flat = base_outputs.reshape({rows, dim});
  auto probs_flat = probs.reshape({rows, context_len});
  if (context_len == 0) {
    return outputs.copy_(base_flat.reshape({rows, 1, dim}).expand({rows, v_steps, dim}));
  }

  int64_t max_exact = 0;
  auto v_cpu = v_budgets.detach().cpu();
  const auto* v_ptr = v_cpu.data_ptr<int64_t>();
  for (int64_t i = 0; i < v_steps; ++i) {
    max_exact = std::max<int64_t>(max_exact, v_ptr[i]);
  }
  max_exact = std::max<int64_t>(0, std::min<int64_t>(max_exact, context_len));
  if (max_exact == 0) {
    return outputs.copy_(base_flat.reshape({rows, 1, dim}).expand({rows, v_steps, dim}));
  }

  auto flat_opts = base_outputs.options().dtype(torch::kFloat32);
  auto long_opts = base_outputs.options().dtype(torch::kLong);
  auto int_opts = base_outputs.options().dtype(torch::kInt);
  auto byte_opts = base_outputs.options().dtype(torch::kUInt8);
  auto token_to_indexed = torch::empty({groups, context_len}, int_opts);
  auto base_mask = torch::empty({groups, context_len}, byte_opts);
  auto rank_pos = torch::empty({groups, heads, context_len}, int_opts);
  auto head_ref = torch::empty({groups, heads}, flat_opts);
  const int64_t segments = groups * heads;
  auto base_risk_in = torch::empty({segments, context_len}, flat_opts);
  auto base_risk_out = torch::empty({segments, context_len}, flat_opts);
  auto base_ids_in = torch::empty({segments, context_len}, int_opts);
  auto base_ids_out = torch::empty({segments, context_len}, int_opts);
  auto base_offsets = torch::empty({segments + 1}, long_opts);
  auto exact_risk_in = torch::empty({segments, std::max<int64_t>(max_rank_take, 1)}, flat_opts);
  auto exact_risk_out = torch::empty({segments, std::max<int64_t>(max_rank_take, 1)}, flat_opts);
  auto exact_ids_in = torch::empty({segments, std::max<int64_t>(max_rank_take, 1)}, int_opts);
  auto exact_ids_out = torch::empty({segments, std::max<int64_t>(max_rank_take, 1)}, int_opts);
  auto exact_offsets = torch::empty({segments + 1}, long_opts);
  auto merged_ids = torch::empty({rows, max_exact}, int_opts);
  auto interval_sums = torch::empty({rows, v_steps, dim}, flat_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t token_map_total = groups * context_len;
  const int64_t token_map_blocks = (token_map_total + threads - 1) / threads;
  joint_fill_int_kernel<<<static_cast<unsigned int>(token_map_blocks), threads, 0, stream>>>(
      token_to_indexed.data_ptr<int>(),
      token_map_total,
      -1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  joint_fill_byte_kernel<<<static_cast<unsigned int>(token_map_blocks), threads, 0, stream>>>(
      base_mask.data_ptr<unsigned char>(),
      token_map_total,
      static_cast<unsigned char>(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t rank_pos_total = groups * heads * context_len;
  const int64_t rank_pos_blocks = (rank_pos_total + threads - 1) / threads;
  joint_fill_int_kernel<<<static_cast<unsigned int>(rank_pos_blocks), threads, 0, stream>>>(
      rank_pos.data_ptr<int>(),
      rank_pos_total,
      -1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (indexed_count > 0) {
    const int64_t total = groups * indexed_count;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_merge_grouped_indexed_token_to_pos_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        token_to_indexed.data_ptr<int>(),
        indexed_tokens.data_ptr<int64_t>(),
        groups,
        indexed_count,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (base_count > 0) {
    const int64_t total = groups * base_count;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_merge_grouped_base_mask_set_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        base_mask.data_ptr<unsigned char>(),
        base_tokens.data_ptr<int64_t>(),
        groups,
        base_count,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (max_rank_take > 0) {
    const int64_t total = groups * heads * max_rank_take;
    const int64_t blocks = (total + threads - 1) / threads;
    joint_merge_grouped_rank_position_scatter_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        rank_pos.data_ptr<int>(),
        ranked_prefix_tokens.data_ptr<int64_t>(),
        groups,
        heads,
        max_rank_take,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  joint_merge_grouped_ref_stats_kernel<<<
      static_cast<unsigned int>(segments), threads, threads * sizeof(float), stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_mask.data_ptr<unsigned char>(),
      ranked_prefix_tokens.data_ptr<int64_t>(),
      head_ref.data_ptr<float>(),
      groups,
      heads,
      indexed_count,
      max_rank_take,
      context_len,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t base_total = segments * context_len;
  const int64_t base_blocks = (base_total + threads - 1) / threads;
  joint_merge_grouped_base_risk_inputs_kernel<<<static_cast<unsigned int>(base_blocks), threads, 0, stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_mask.data_ptr<unsigned char>(),
      code_error_groups.data_ptr<float>(),
      head_ref.data_ptr<float>(),
      base_risk_in.data_ptr<float>(),
      base_ids_in.data_ptr<int32_t>(),
      groups,
      heads,
      indexed_count,
      context_len,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t offset_blocks = ((segments + 1) + threads - 1) / threads;
  joint_segment_offsets_kernel<<<static_cast<unsigned int>(offset_blocks), threads, 0, stream>>>(
      base_offsets.data_ptr<int64_t>(),
      segments,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  size_t base_temp_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      base_temp_bytes,
      base_risk_in.data_ptr<float>(),
      base_risk_out.data_ptr<float>(),
      base_ids_in.data_ptr<int32_t>(),
      base_ids_out.data_ptr<int32_t>(),
      base_total,
      segments,
      base_offsets.data_ptr<int64_t>(),
      base_offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  auto base_temp = torch::empty({static_cast<int64_t>(std::max<size_t>(base_temp_bytes, 1))}, byte_opts);
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      base_temp.data_ptr(),
      base_temp_bytes,
      base_risk_in.data_ptr<float>(),
      base_risk_out.data_ptr<float>(),
      base_ids_in.data_ptr<int32_t>(),
      base_ids_out.data_ptr<int32_t>(),
      base_total,
      segments,
      base_offsets.data_ptr<int64_t>(),
      base_offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (max_rank_take > 0) {
    const int64_t exact_total = segments * max_rank_take;
    const int64_t exact_blocks = (exact_total + threads - 1) / threads;
    joint_merge_grouped_exact_risk_inputs_kernel<<<static_cast<unsigned int>(exact_blocks), threads, 0, stream>>>(
        exact_scores.data_ptr<float>(),
        ranked_prefix_tokens.data_ptr<int64_t>(),
        code_error_groups.data_ptr<float>(),
        head_ref.data_ptr<float>(),
        exact_risk_in.data_ptr<float>(),
        exact_ids_in.data_ptr<int32_t>(),
        groups,
        heads,
        max_rank_take,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    joint_segment_offsets_kernel<<<static_cast<unsigned int>(offset_blocks), threads, 0, stream>>>(
        exact_offsets.data_ptr<int64_t>(),
        segments,
        max_rank_take);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    size_t exact_temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr,
        exact_temp_bytes,
        exact_risk_in.data_ptr<float>(),
        exact_risk_out.data_ptr<float>(),
        exact_ids_in.data_ptr<int32_t>(),
        exact_ids_out.data_ptr<int32_t>(),
        exact_total,
        segments,
        exact_offsets.data_ptr<int64_t>(),
        exact_offsets.data_ptr<int64_t>() + 1,
        0,
        sizeof(float) * 8,
        stream);
    auto exact_temp = torch::empty({static_cast<int64_t>(std::max<size_t>(exact_temp_bytes, 1))}, byte_opts);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        exact_temp.data_ptr(),
        exact_temp_bytes,
        exact_risk_in.data_ptr<float>(),
        exact_risk_out.data_ptr<float>(),
        exact_ids_in.data_ptr<int32_t>(),
        exact_ids_out.data_ptr<int32_t>(),
        exact_total,
        segments,
        exact_offsets.data_ptr<int64_t>(),
        exact_offsets.data_ptr<int64_t>() + 1,
        0,
        sizeof(float) * 8,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    exact_risk_out.zero_();
    exact_ids_out.zero_();
  }

  joint_merge_grouped_risk_order_kernel<<<static_cast<unsigned int>(rows), 1, 0, stream>>>(
      base_risk_out.data_ptr<float>(),
      base_ids_out.data_ptr<int32_t>(),
      exact_risk_out.data_ptr<float>(),
      exact_ids_out.data_ptr<int32_t>(),
      rank_pos.data_ptr<int>(),
      k_take_counts.data_ptr<int64_t>(),
      merged_ids.data_ptr<int32_t>(),
      groups,
      k_count,
      heads,
      context_len,
      max_rank_take,
      max_exact);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  const int64_t interval_blocks = rows * v_steps * dim_tiles;
  joint_grouped_flat_vprefix_interval_sums_kernel<int32_t, dims_per_block><<<
      static_cast<unsigned int>(interval_blocks), threads, 0, stream>>>(
      probs_flat.data_ptr<float>(),
      residual_groups.data_ptr<float>(),
      merged_ids.data_ptr<int32_t>(),
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

std::vector<torch::Tensor> joint_mixed_select_policy_merge_rankpos_no_calib_no_mb_cuda(
    torch::Tensor exact_scores,
    torch::Tensor pq_logits,
    torch::Tensor indexed_tokens,
    torch::Tensor base_tokens,
    torch::Tensor ranked_prefix_tokens,
    torch::Tensor k_take_counts,
    torch::Tensor vhat,
    torch::Tensor residual,
    torch::Tensor code_error,
    torch::Tensor v_budgets,
    double pq_scale,
    double threshold,
    int64_t policy_id) {
  TORCH_CHECK(policy_id >= 0 && policy_id <= 3, "merge-risk policy only supports non-MB policies");
  TORCH_CHECK(exact_scores.is_cuda(), "exact_scores must be CUDA");
  TORCH_CHECK(pq_logits.is_cuda(), "pq_logits must be CUDA");
  TORCH_CHECK(indexed_tokens.is_cuda(), "indexed_tokens must be CUDA");
  TORCH_CHECK(base_tokens.is_cuda(), "base_tokens must be CUDA");
  TORCH_CHECK(ranked_prefix_tokens.is_cuda(), "ranked_prefix_tokens must be CUDA");
  TORCH_CHECK(k_take_counts.is_cuda(), "k_take_counts must be CUDA");
  TORCH_CHECK(vhat.is_cuda(), "vhat must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(code_error.is_cuda(), "code_error must be CUDA");
  TORCH_CHECK(v_budgets.is_cuda(), "v_budgets must be CUDA");
  TORCH_CHECK(exact_scores.scalar_type() == torch::kFloat32, "exact_scores must be float32");
  TORCH_CHECK(pq_logits.scalar_type() == torch::kFloat32, "pq_logits must be float32");
  TORCH_CHECK(indexed_tokens.scalar_type() == torch::kLong, "indexed_tokens must be int64");
  TORCH_CHECK(base_tokens.scalar_type() == torch::kLong, "base_tokens must be int64");
  TORCH_CHECK(ranked_prefix_tokens.scalar_type() == torch::kLong, "ranked_prefix_tokens must be int64");
  TORCH_CHECK(k_take_counts.scalar_type() == torch::kLong, "k_take_counts must be int64");
  TORCH_CHECK(vhat.scalar_type() == torch::kFloat32, "vhat must be float32");
  TORCH_CHECK(residual.scalar_type() == torch::kFloat32, "residual must be float32");
  TORCH_CHECK(code_error.scalar_type() == torch::kFloat32, "code_error must be float32");
  TORCH_CHECK(v_budgets.scalar_type() == torch::kLong, "v_budgets must be int64");
  TORCH_CHECK(exact_scores.dim() == 2, "exact_scores shape must be [heads, context_len]");
  TORCH_CHECK(pq_logits.dim() == 2, "pq_logits shape must be [heads, indexed_count]");
  TORCH_CHECK(indexed_tokens.dim() == 1, "indexed_tokens shape must be [indexed_count]");
  TORCH_CHECK(base_tokens.dim() == 1, "base_tokens shape must be [base_count]");
  TORCH_CHECK(ranked_prefix_tokens.dim() == 2, "ranked_prefix_tokens shape must be [heads, max_rank_take]");
  TORCH_CHECK(k_take_counts.dim() == 1, "k_take_counts shape must be [k_count]");
  TORCH_CHECK(vhat.dim() == 2, "vhat shape must be [context_len, dim]");
  TORCH_CHECK(residual.sizes() == vhat.sizes(), "residual must match vhat shape");
  TORCH_CHECK(code_error.dim() == 1, "code_error shape must be [context_len]");
  TORCH_CHECK(v_budgets.dim() == 1, "v_budgets shape must be [v_steps]");

  const auto heads = exact_scores.size(0);
  const auto context_len = exact_scores.size(1);
  const auto indexed_count = indexed_tokens.size(0);
  const auto k_count = k_take_counts.size(0);
  const auto max_rank_take = ranked_prefix_tokens.size(1);
  const auto dim = vhat.size(1);
  const auto v_steps = v_budgets.size(0);
  const auto rows = k_count * heads;
  TORCH_CHECK(pq_logits.size(0) == heads, "pq_logits head count mismatch");
  TORCH_CHECK(pq_logits.size(1) == indexed_count, "pq_logits indexed count mismatch");
  TORCH_CHECK(ranked_prefix_tokens.size(0) == heads, "ranked_prefix_tokens head count mismatch");
  TORCH_CHECK(vhat.size(0) == context_len, "vhat context mismatch");
  TORCH_CHECK(code_error.size(0) == context_len, "code_error context mismatch");

  auto final_outputs = torch::empty({heads, dim}, exact_scores.options().dtype(torch::kFloat32));
  auto final_indices = torch::empty({heads, 2}, exact_scores.options().dtype(torch::kLong));
  if (heads == 0 || k_count == 0 || v_steps == 0 || dim == 0) {
    return {final_outputs, final_indices};
  }
  if (context_len == 0) {
    final_outputs.zero_();
    final_indices.zero_();
    return {final_outputs, final_indices};
  }

  int64_t max_exact = 0;
  auto v_cpu = v_budgets.detach().cpu();
  const auto* v_ptr = v_cpu.data_ptr<int64_t>();
  for (int64_t i = 0; i < v_steps; ++i) {
    max_exact = std::max<int64_t>(max_exact, v_ptr[i]);
  }
  max_exact = std::max<int64_t>(0, std::min<int64_t>(max_exact, context_len));

  auto flat_opts = exact_scores.options().dtype(torch::kFloat32);
  auto long_opts = exact_scores.options().dtype(torch::kLong);
  auto int_opts = exact_scores.options().dtype(torch::kInt);
  auto byte_opts = exact_scores.options().dtype(torch::kUInt8);
  auto token_to_indexed = torch::empty({context_len}, int_opts);
  auto base_mask = torch::empty({context_len}, byte_opts);
  auto rank_pos = torch::empty({heads, context_len}, int_opts);
  auto head_ref = torch::empty({heads}, flat_opts);
  auto base_sum = torch::empty({heads}, flat_opts);
  auto base_num = torch::empty({heads, dim}, flat_opts);
  auto row_max = torch::empty({rows}, flat_opts);
  auto row_denom = torch::empty({rows}, flat_opts);
  auto base_flat = torch::empty({rows, dim}, flat_opts);
  auto interval_sums = torch::empty({rows, v_steps, dim}, flat_opts);

  auto base_risk_in = torch::empty({heads, context_len}, flat_opts);
  auto base_risk_out = torch::empty({heads, context_len}, flat_opts);
  auto base_ids_in = torch::empty({heads, context_len}, int_opts);
  auto base_ids_out = torch::empty({heads, context_len}, int_opts);
  auto base_offsets = torch::empty({heads + 1}, long_opts);
  torch::Tensor exact_risk_in;
  torch::Tensor exact_risk_out;
  torch::Tensor exact_ids_in;
  torch::Tensor exact_ids_out;
  torch::Tensor exact_offsets;
  if (max_rank_take > 0) {
    exact_risk_in = torch::empty({heads, max_rank_take}, flat_opts);
    exact_risk_out = torch::empty({heads, max_rank_take}, flat_opts);
    exact_ids_in = torch::empty({heads, max_rank_take}, int_opts);
    exact_ids_out = torch::empty({heads, max_rank_take}, int_opts);
    exact_offsets = torch::empty({heads + 1}, long_opts);
  } else {
    exact_risk_in = torch::empty({heads, 1}, flat_opts);
    exact_risk_out = torch::empty({heads, 1}, flat_opts);
    exact_ids_in = torch::empty({heads, 1}, int_opts);
    exact_ids_out = torch::empty({heads, 1}, int_opts);
    exact_offsets = torch::empty({heads + 1}, long_opts);
  }
  auto merged_ids = torch::empty({rows, std::max<int64_t>(max_exact, 1)}, int_opts);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t token_map_blocks = (context_len + threads - 1) / threads;
  joint_fill_int_kernel<<<static_cast<unsigned int>(token_map_blocks), threads, 0, stream>>>(
      token_to_indexed.data_ptr<int>(),
      context_len,
      -1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  joint_fill_byte_kernel<<<static_cast<unsigned int>(token_map_blocks), threads, 0, stream>>>(
      base_mask.data_ptr<unsigned char>(),
      context_len,
      static_cast<unsigned char>(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t rank_pos_total = heads * context_len;
  const int64_t rank_pos_blocks = (rank_pos_total + threads - 1) / threads;
  joint_fill_int_kernel<<<static_cast<unsigned int>(rank_pos_blocks), threads, 0, stream>>>(
      rank_pos.data_ptr<int>(),
      rank_pos_total,
      -1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (indexed_count > 0) {
    const int64_t blocks = (indexed_count + threads - 1) / threads;
    joint_indexed_token_to_pos_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        token_to_indexed.data_ptr<int>(),
        indexed_tokens.data_ptr<int64_t>(),
        indexed_count,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
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

  joint_merge_ref_stats_kernel<<<static_cast<unsigned int>(heads), threads, threads * sizeof(float), stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_mask.data_ptr<unsigned char>(),
      ranked_prefix_tokens.data_ptr<int64_t>(),
      head_ref.data_ptr<float>(),
      heads,
      indexed_count,
      max_rank_take,
      context_len,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  joint_merge_base_sum_kernel<<<static_cast<unsigned int>(heads), threads, threads * sizeof(float), stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_mask.data_ptr<unsigned char>(),
      head_ref.data_ptr<float>(),
      base_sum.data_ptr<float>(),
      heads,
      indexed_count,
      context_len,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int dims_per_block = 8;
  const int64_t dim_tiles = (dim + dims_per_block - 1) / dims_per_block;
  joint_merge_base_num_kernel<dims_per_block><<<
      static_cast<unsigned int>(heads * dim_tiles), threads, 0, stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_mask.data_ptr<unsigned char>(),
      head_ref.data_ptr<float>(),
      vhat.data_ptr<float>(),
      base_num.data_ptr<float>(),
      heads,
      indexed_count,
      context_len,
      dim,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t base_total = heads * context_len;
  const int64_t base_blocks = (base_total + threads - 1) / threads;
  joint_merge_base_risk_inputs_kernel<<<static_cast<unsigned int>(base_blocks), threads, 0, stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_mask.data_ptr<unsigned char>(),
      code_error.data_ptr<float>(),
      head_ref.data_ptr<float>(),
      base_risk_in.data_ptr<float>(),
      base_ids_in.data_ptr<int32_t>(),
      heads,
      indexed_count,
      context_len,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t offset_blocks = ((heads + 1) + threads - 1) / threads;
  joint_segment_offsets_kernel<<<static_cast<unsigned int>(offset_blocks), threads, 0, stream>>>(
      base_offsets.data_ptr<int64_t>(),
      heads,
      context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  size_t base_temp_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      base_temp_bytes,
      base_risk_in.data_ptr<float>(),
      base_risk_out.data_ptr<float>(),
      base_ids_in.data_ptr<int32_t>(),
      base_ids_out.data_ptr<int32_t>(),
      base_total,
      heads,
      base_offsets.data_ptr<int64_t>(),
      base_offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  auto base_temp = torch::empty({static_cast<int64_t>(std::max<size_t>(base_temp_bytes, 1))}, byte_opts);
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      base_temp.data_ptr(),
      base_temp_bytes,
      base_risk_in.data_ptr<float>(),
      base_risk_out.data_ptr<float>(),
      base_ids_in.data_ptr<int32_t>(),
      base_ids_out.data_ptr<int32_t>(),
      base_total,
      heads,
      base_offsets.data_ptr<int64_t>(),
      base_offsets.data_ptr<int64_t>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (max_rank_take > 0) {
    const int64_t exact_total = heads * max_rank_take;
    const int64_t exact_blocks = (exact_total + threads - 1) / threads;
    joint_merge_exact_risk_inputs_kernel<<<static_cast<unsigned int>(exact_blocks), threads, 0, stream>>>(
        exact_scores.data_ptr<float>(),
        ranked_prefix_tokens.data_ptr<int64_t>(),
        code_error.data_ptr<float>(),
        head_ref.data_ptr<float>(),
        exact_risk_in.data_ptr<float>(),
        exact_ids_in.data_ptr<int32_t>(),
        heads,
        max_rank_take,
        context_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    joint_segment_offsets_kernel<<<static_cast<unsigned int>(offset_blocks), threads, 0, stream>>>(
        exact_offsets.data_ptr<int64_t>(),
        heads,
        max_rank_take);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    size_t exact_temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr,
        exact_temp_bytes,
        exact_risk_in.data_ptr<float>(),
        exact_risk_out.data_ptr<float>(),
        exact_ids_in.data_ptr<int32_t>(),
        exact_ids_out.data_ptr<int32_t>(),
        exact_total,
        heads,
        exact_offsets.data_ptr<int64_t>(),
        exact_offsets.data_ptr<int64_t>() + 1,
        0,
        sizeof(float) * 8,
        stream);
    auto exact_temp = torch::empty({static_cast<int64_t>(std::max<size_t>(exact_temp_bytes, 1))}, byte_opts);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        exact_temp.data_ptr(),
        exact_temp_bytes,
        exact_risk_in.data_ptr<float>(),
        exact_risk_out.data_ptr<float>(),
        exact_ids_in.data_ptr<int32_t>(),
        exact_ids_out.data_ptr<int32_t>(),
        exact_total,
        heads,
        exact_offsets.data_ptr<int64_t>(),
        exact_offsets.data_ptr<int64_t>() + 1,
        0,
        sizeof(float) * 8,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    exact_risk_out.zero_();
    exact_ids_out.zero_();
  }

  joint_merge_row_denoms_kernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_mask.data_ptr<unsigned char>(),
      ranked_prefix_tokens.data_ptr<int64_t>(),
      k_take_counts.data_ptr<int64_t>(),
      head_ref.data_ptr<float>(),
      base_sum.data_ptr<float>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      rows,
      heads,
      indexed_count,
      max_rank_take,
      context_len,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  joint_merge_base_outputs_kernel<dims_per_block><<<
      static_cast<unsigned int>(rows * dim_tiles), threads, 0, stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_mask.data_ptr<unsigned char>(),
      ranked_prefix_tokens.data_ptr<int64_t>(),
      k_take_counts.data_ptr<int64_t>(),
      head_ref.data_ptr<float>(),
      base_num.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      vhat.data_ptr<float>(),
      base_flat.data_ptr<float>(),
      rows,
      heads,
      indexed_count,
      max_rank_take,
      context_len,
      dim,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (max_exact > 0) {
    joint_merge_risk_order_kernel<<<static_cast<unsigned int>(rows), 1, 0, stream>>>(
        base_risk_out.data_ptr<float>(),
        base_ids_out.data_ptr<int32_t>(),
        exact_risk_out.data_ptr<float>(),
        exact_ids_out.data_ptr<int32_t>(),
        rank_pos.data_ptr<int>(),
        k_take_counts.data_ptr<int64_t>(),
        merged_ids.data_ptr<int32_t>(),
        rows,
        heads,
        context_len,
        max_rank_take,
        max_exact);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    merged_ids.zero_();
  }

  const int64_t interval_blocks = rows * v_steps * dim_tiles;
  joint_vprefix_interval_sums_from_mixed_rankpos_nocalib_kernel<int32_t, dims_per_block><<<
      static_cast<unsigned int>(interval_blocks), threads, 0, stream>>>(
      exact_scores.data_ptr<float>(),
      pq_logits.data_ptr<float>(),
      token_to_indexed.data_ptr<int>(),
      base_mask.data_ptr<unsigned char>(),
      rank_pos.data_ptr<int>(),
      k_take_counts.data_ptr<int64_t>(),
      row_max.data_ptr<float>(),
      row_denom.data_ptr<float>(),
      residual.data_ptr<float>(),
      merged_ids.data_ptr<int32_t>(),
      v_budgets.data_ptr<int64_t>(),
      interval_sums.data_ptr<float>(),
      rows,
      heads,
      indexed_count,
      max_rank_take,
      context_len,
      std::max<int64_t>(max_exact, 1),
      v_steps,
      dim,
      static_cast<float>(pq_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const size_t shared_bytes = static_cast<size_t>(threads) * 4 * sizeof(double);
  joint_select_policy_grouped_intervals_no_mb_kernel<<<
      static_cast<unsigned int>(heads), threads, shared_bytes, stream>>>(
      base_flat.data_ptr<float>(),
      interval_sums.data_ptr<float>(),
      final_indices.data_ptr<int64_t>(),
      final_outputs.data_ptr<float>(),
      1,
      k_count,
      v_steps,
      heads,
      dim,
      threshold,
      static_cast<int>(policy_id));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {final_outputs, final_indices};
}
