__global__ void gqa_decode_tail_partial_max_kernel(
    const float* __restrict__ dense_pq_scores,
    const int64_t* __restrict__ page_starts,
    const unsigned char* __restrict__ tail_mask,
    float* __restrict__ partial_max,
    int64_t heads,
    int64_t pages,
    int64_t page_size,
    int64_t tail_blocks,
    int64_t pages_per_block,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    float scale) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  int64_t block = static_cast<int64_t>(blockIdx.y);
  if (head >= heads || block >= tail_blocks) {
    return;
  }
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
  float local_max = -INFINITY;
  int64_t total_rows = max(static_cast<int64_t>(0), page_end - page_begin) * page_size;
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
    local_max = fmaxf(local_max, score_row[ordinal] * scale);
  }
  extern __shared__ float shared[];
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_max[head * tail_blocks + block] = shared[0];
  }
}

__global__ void gqa_decode_tail_partial_max_nomask_kernel(
    const float* __restrict__ dense_pq_scores,
    const int64_t* __restrict__ page_starts,
    float* __restrict__ partial_max,
    int64_t heads,
    int64_t pages,
    int64_t page_size,
    int64_t tail_blocks,
    int64_t pages_per_block,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    float scale) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  int64_t block = static_cast<int64_t>(blockIdx.y);
  if (head >= heads || block >= tail_blocks) {
    return;
  }
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
  float local_max = -INFINITY;
  int64_t total_rows = max(static_cast<int64_t>(0), page_end - page_begin) * page_size;
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
    local_max = fmaxf(local_max, score_row[page * page_size + row] * scale);
  }
  extern __shared__ float shared[];
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_max[head * tail_blocks + block] = shared[0];
  }
}

__global__ void gqa_decode_final_max_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    const float* __restrict__ ranked_logits,
    float* __restrict__ max_logits,
    int64_t heads,
    int64_t ranked,
    int64_t max_base,
    int64_t tail_blocks) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }
  float local_max = -INFINITY;
  for (int64_t idx = threadIdx.x; idx < tail_blocks; idx += blockDim.x) {
    local_max = fmaxf(local_max, partial_max[head * tail_blocks + idx]);
  }
  extern __shared__ float shared[];
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float max_value = shared[0];
    const float* base_logits_row = base_logits + head * max_base;
    int64_t base_count = base_counts[head];
    for (int64_t idx = 0; idx < base_count; ++idx) {
      max_value = fmaxf(max_value, base_logits_row[idx]);
    }
    const float* ranked_logits_row = ranked_logits + head * ranked;
    for (int64_t sel = 0; sel < ranked; ++sel) {
      max_value = fmaxf(max_value, ranked_logits_row[sel]);
    }
    max_logits[head] = max_value;
  }
}

