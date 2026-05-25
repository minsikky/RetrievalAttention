template <typename vcode_t>
__global__ void gqa_decode_tail_sum_codeweights_kernel(
    const float* __restrict__ dense_pq_scores,
    const float* __restrict__ max_logits,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const unsigned char* __restrict__ tail_mask,
    float* __restrict__ partial_sum,
    float* __restrict__ code_weight_sums,
    int64_t heads,
    int64_t kv_heads,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t tail_blocks,
    int64_t pages_per_block,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    float scale) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  int64_t block = static_cast<int64_t>(blockIdx.y);
  if (head >= heads || block >= tail_blocks) {
    return;
  }
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t prefix_end = 0;
  int64_t base_tail_start = 0;
  token_in_base_window(
      0,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      &prefix_end,
      &base_tail_start);
  int64_t page_begin = block * pages_per_block;
  int64_t page_end = min(pages, page_begin + pages_per_block);
  const float* score_row = dense_pq_scores + head * (pages * page_size);
  const unsigned char* mask_row = tail_mask + head * (pages * page_size);
  float max_logit = max_logits[head];
  float local_sum = 0.0f;
  int64_t total_rows = max(static_cast<int64_t>(0), page_end - page_begin) * page_size;
  if (isfinite(max_logit)) {
    for (int64_t local = threadIdx.x; local < total_rows; local += blockDim.x) {
      int64_t page = page_begin + local / page_size;
      int64_t row = local - (page - page_begin) * page_size;
      int64_t page_start = page_starts[page];
      bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
      if (!valid_page) {
        continue;
      }
      int64_t token = page_start + row;
      if (token < 0 || token >= query_context_len) {
        continue;
      }
      int64_t ordinal = page * page_size + row;
      if (mask_row[ordinal]) {
        continue;
      }
      float weight = expf(score_row[ordinal] * scale - max_logit);
      local_sum += weight;
      for (int64_t sub = 0; sub < value_subvecs; ++sub) {
        int64_t code = static_cast<int64_t>(
            value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
        code = max(static_cast<int64_t>(0), min(code, value_centroids - 1));
        atomicAdd(
            code_weight_sums + (((head * pages + page) * value_subvecs + sub) * value_centroids + code),
            weight);
      }
    }
  }
  extern __shared__ float shared[];
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_sum[head * tail_blocks + block] = shared[0];
  }
}

template <typename vcode_t>
__global__ void gqa_decode_tail_sum_codeweights_nomask_kernel(
    const float* __restrict__ dense_pq_scores,
    const float* __restrict__ max_logits,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    float* __restrict__ partial_sum,
    float* __restrict__ code_weight_sums,
    int64_t heads,
    int64_t kv_heads,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t tail_blocks,
    int64_t pages_per_block,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    float scale) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  int64_t block = static_cast<int64_t>(blockIdx.y);
  if (head >= heads || block >= tail_blocks) {
    return;
  }
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t prefix_end = 0;
  int64_t base_tail_start = 0;
  token_in_base_window(
      0,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      &prefix_end,
      &base_tail_start);
  int64_t page_begin = block * pages_per_block;
  int64_t page_end = min(pages, page_begin + pages_per_block);
  const float* score_row = dense_pq_scores + head * (pages * page_size);
  float max_logit = max_logits[head];
  float local_sum = 0.0f;
  int64_t total_rows = max(static_cast<int64_t>(0), page_end - page_begin) * page_size;
  if (isfinite(max_logit)) {
    for (int64_t local = threadIdx.x; local < total_rows; local += blockDim.x) {
      int64_t page = page_begin + local / page_size;
      int64_t row = local - (page - page_begin) * page_size;
      int64_t page_start = page_starts[page];
      bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
      if (!valid_page) {
        continue;
      }
      int64_t token = page_start + row;
      if (token < 0 || token >= query_context_len) {
        continue;
      }
      float weight = expf(score_row[page * page_size + row] * scale - max_logit);
      local_sum += weight;
      for (int64_t sub = 0; sub < value_subvecs; ++sub) {
        int64_t code = static_cast<int64_t>(
            value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
        code = max(static_cast<int64_t>(0), min(code, value_centroids - 1));
        atomicAdd(
            code_weight_sums + (((head * pages + page) * value_subvecs + sub) * value_centroids + code),
            weight);
      }
    }
  }
  extern __shared__ float shared[];
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_sum[head * tail_blocks + block] = shared[0];
  }
}

__global__ void gqa_decode_tail_denom_from_partials_kernel(
    const float* __restrict__ partial_sum,
    float* __restrict__ tail_denoms,
    int64_t heads,
    int64_t tail_blocks) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }
  float local_sum = 0.0f;
  for (int64_t idx = threadIdx.x; idx < tail_blocks; idx += blockDim.x) {
    local_sum += partial_sum[head * tail_blocks + idx];
  }
  extern __shared__ float shared[];
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    tail_denoms[head] = shared[0];
  }
}

__global__ void gqa_decode_final_denom_kernel(
    const float* __restrict__ partial_sum,
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    const float* __restrict__ ranked_logits,
    const float* __restrict__ max_logits,
    float* __restrict__ denoms,
    float* __restrict__ selected_denoms,
    int64_t heads,
    int64_t ranked,
    int64_t max_base,
    int64_t tail_blocks) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }
  float local_sum = 0.0f;
  for (int64_t idx = threadIdx.x; idx < tail_blocks; idx += blockDim.x) {
    local_sum += partial_sum[head * tail_blocks + idx];
  }
  extern __shared__ float shared[];
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float max_logit = max_logits[head];
    float selected_sum = 0.0f;
    if (isfinite(max_logit)) {
      const float* base_logits_row = base_logits + head * max_base;
      int64_t base_count = base_counts[head];
      for (int64_t idx = 0; idx < base_count; ++idx) {
        selected_sum += expf(base_logits_row[idx] - max_logit);
      }
      const float* ranked_logits_row = ranked_logits + head * ranked;
      for (int64_t sel = 0; sel < ranked; ++sel) {
        float logit = ranked_logits_row[sel];
        if (isfinite(logit)) {
          selected_sum += expf(logit - max_logit);
        }
      }
    }
    selected_denoms[head] = fmaxf(selected_sum, 1.0e-20f);
    denoms[head] = fmaxf(selected_sum + shared[0], 1.0e-20f);
  }
}

