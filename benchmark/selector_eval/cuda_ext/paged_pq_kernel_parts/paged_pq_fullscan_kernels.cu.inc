template <typename code_t>
__global__ void pq_scores_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ codebooks,
    const code_t* __restrict__ codes,
    float* __restrict__ scores,
    int64_t heads,
    int64_t dim,
    int64_t pages,
    int64_t page_size,
    int64_t subvecs,
    int64_t centroids,
    int64_t subdim,
    int64_t total_tokens) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * total_tokens;
  if (linear >= total) {
    return;
  }

  int64_t head = linear / total_tokens;
  int64_t token_ordinal = linear - head * total_tokens;
  int64_t page = token_ordinal / page_size;
  int64_t row = token_ordinal - page * page_size;

  const float* q = queries + head * dim;
  const code_t* code_row = codes + (page * page_size + row) * subvecs;

  float score = 0.0f;
  for (int64_t sub = 0; sub < subvecs; ++sub) {
    int64_t code = static_cast<int64_t>(code_row[sub]);
    code = max(static_cast<int64_t>(0), min(code, centroids - 1));
    const float* cb = codebooks + (((page * subvecs + sub) * centroids + code) * subdim);
    const float* q_sub = q + sub * subdim;
    float dot = 0.0f;
    for (int64_t d = 0; d < subdim; ++d) {
      dot += q_sub[d] * cb[d];
    }
    score += dot;
  }
  scores[linear] = score;
}

__global__ void map_topk_tokens_kernel(
    const int64_t* __restrict__ top_indices,
    const int64_t* __restrict__ page_starts,
    int64_t* __restrict__ top_tokens,
    int64_t heads,
    int64_t k,
    int64_t page_size) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * k;
  if (linear >= total) {
    return;
  }
  int64_t idx = top_indices[linear];
  int64_t page = idx / page_size;
  int64_t row = idx - page * page_size;
  top_tokens[linear] = page_starts[page] + row;
}

template <typename code_t>
__global__ void gqa_pq_scores_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ codebooks,
    const code_t* __restrict__ codes,
    float* __restrict__ scores,
    int64_t heads,
    int64_t kv_heads,
    int64_t dim,
    int64_t pages,
    int64_t page_size,
    int64_t subvecs,
    int64_t centroids,
    int64_t subdim,
    int64_t total_tokens,
    int64_t group_size) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * total_tokens;
  if (linear >= total) {
    return;
  }

  int64_t head = linear / total_tokens;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t token_ordinal = linear - head * total_tokens;
  int64_t page = token_ordinal / page_size;
  int64_t row = token_ordinal - page * page_size;

  const float* q = queries + head * dim;
  const code_t* code_row = codes + ((kv_head * pages + page) * page_size + row) * subvecs;

  float score = 0.0f;
  for (int64_t sub = 0; sub < subvecs; ++sub) {
    int64_t code = static_cast<int64_t>(code_row[sub]);
    code = max(static_cast<int64_t>(0), min(code, centroids - 1));
    const float* cb = codebooks + ((((kv_head * pages + page) * subvecs + sub) * centroids + code) * subdim);
    const float* q_sub = q + sub * subdim;
    float dot = 0.0f;
    for (int64_t d = 0; d < subdim; ++d) {
      dot += q_sub[d] * cb[d];
    }
    score += dot;
  }
  scores[linear] = score;
}

template <typename code_t>
__global__ void gqa_pq_scores_lut_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ codebooks,
    const code_t* __restrict__ codes,
    float* __restrict__ scores,
    int64_t heads,
    int64_t kv_heads,
    int64_t dim,
    int64_t pages,
    int64_t page_size,
    int64_t subvecs,
    int64_t centroids,
    int64_t subdim,
    int64_t total_tokens,
    int64_t group_size) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  int64_t page = static_cast<int64_t>(blockIdx.y);
  if (head >= heads || page >= pages) {
    return;
  }

  int64_t kv_head = min(head / group_size, kv_heads - 1);
  extern __shared__ float lut[];
  const float* q = queries + head * dim;
  int64_t lut_entries = subvecs * centroids;
  for (int64_t entry = threadIdx.x; entry < lut_entries; entry += blockDim.x) {
    int64_t sub = entry / centroids;
    int64_t centroid = entry - sub * centroids;
    const float* cb = codebooks + ((((kv_head * pages + page) * subvecs + sub) * centroids + centroid) * subdim);
    const float* q_sub = q + sub * subdim;
    float dot = 0.0f;
    for (int64_t d = 0; d < subdim; ++d) {
      dot += q_sub[d] * cb[d];
    }
    lut[entry] = dot;
  }
  __syncthreads();

  float* page_scores = scores + head * total_tokens + page * page_size;
  for (int64_t row = threadIdx.x; row < page_size; row += blockDim.x) {
    const code_t* code_row = codes + ((kv_head * pages + page) * page_size + row) * subvecs;
    float score = 0.0f;
    for (int64_t sub = 0; sub < subvecs; ++sub) {
      int64_t code = static_cast<int64_t>(code_row[sub]);
      code = max(static_cast<int64_t>(0), min(code, centroids - 1));
      score += lut[sub * centroids + code];
    }
    page_scores[row] = score;
  }
}

template <typename code_t>
__global__ void gqa_causal_pq_scores_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ codebooks,
    const code_t* __restrict__ codes,
    const int64_t* __restrict__ page_starts,
    float* __restrict__ scores,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t dim,
    int64_t pages,
    int64_t page_size,
    int64_t subvecs,
    int64_t centroids,
    int64_t subdim,
    int64_t total_tokens,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = positions * heads * total_tokens;
  if (linear >= total) {
    return;
  }

  int64_t token_ordinal = linear % total_tokens;
  int64_t tmp = linear / total_tokens;
  int64_t head = tmp % heads;
  int64_t pos = tmp / heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t page = token_ordinal / page_size;
  int64_t row = token_ordinal - page * page_size;

  int64_t query_context_len = query_start + pos + 1;
  int64_t dyn_start = min(max(static_cast<int64_t>(0), static_prefix), query_context_len);
  int64_t indexed_end = max(dyn_start, query_context_len - max(static_cast<int64_t>(0), static_suffix));
  int64_t sealed_end =
      dyn_start + (max(static_cast<int64_t>(0), indexed_end - dyn_start) / page_size) * page_size;
  int64_t page_start = page_starts[page];
  bool valid_page = page_start >= dyn_start && page_start + page_size <= sealed_end;
  if (!valid_page) {
    scores[linear] = -INFINITY;
    return;
  }

  const float* q = queries + (pos * heads + head) * dim;
  const code_t* code_row = codes + ((kv_head * pages + page) * page_size + row) * subvecs;

  float score = 0.0f;
  for (int64_t sub = 0; sub < subvecs; ++sub) {
    int64_t code = static_cast<int64_t>(code_row[sub]);
    code = max(static_cast<int64_t>(0), min(code, centroids - 1));
    const float* cb = codebooks + ((((kv_head * pages + page) * subvecs + sub) * centroids + code) * subdim);
    const float* q_sub = q + sub * subdim;
    float dot = 0.0f;
    for (int64_t d = 0; d < subdim; ++d) {
      dot += q_sub[d] * cb[d];
    }
    score += dot;
  }
  scores[linear] = score;
}

template <typename code_t>
__global__ void gqa_causal_pq_scores_lut_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ codebooks,
    const code_t* __restrict__ codes,
    const int64_t* __restrict__ page_starts,
    float* __restrict__ scores,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t dim,
    int64_t pages,
    int64_t page_size,
    int64_t subvecs,
    int64_t centroids,
    int64_t subdim,
    int64_t total_tokens,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  int64_t row_id = static_cast<int64_t>(blockIdx.x);
  int64_t page = static_cast<int64_t>(blockIdx.y);
  int64_t rows_total = positions * heads;
  if (row_id >= rows_total || page >= pages) {
    return;
  }

  int64_t pos = row_id / heads;
  int64_t head = row_id - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t query_context_len = query_start + pos + 1;
  int64_t dyn_start = min(max(static_cast<int64_t>(0), static_prefix), query_context_len);
  int64_t indexed_end = max(dyn_start, query_context_len - max(static_cast<int64_t>(0), static_suffix));
  int64_t sealed_end =
      dyn_start + (max(static_cast<int64_t>(0), indexed_end - dyn_start) / page_size) * page_size;
  int64_t page_start = page_starts[page];
  bool valid_page = page_start >= dyn_start && page_start + page_size <= sealed_end;
  float* page_scores = scores + row_id * total_tokens + page * page_size;
  if (!valid_page) {
    for (int64_t row = threadIdx.x; row < page_size; row += blockDim.x) {
      page_scores[row] = -INFINITY;
    }
    return;
  }

  extern __shared__ float lut[];
  const float* q = queries + (pos * heads + head) * dim;
  int64_t lut_entries = subvecs * centroids;
  for (int64_t entry = threadIdx.x; entry < lut_entries; entry += blockDim.x) {
    int64_t sub = entry / centroids;
    int64_t centroid = entry - sub * centroids;
    const float* cb = codebooks + ((((kv_head * pages + page) * subvecs + sub) * centroids + centroid) * subdim);
    const float* q_sub = q + sub * subdim;
    float dot = 0.0f;
    for (int64_t d = 0; d < subdim; ++d) {
      dot += q_sub[d] * cb[d];
    }
    lut[entry] = dot;
  }
  __syncthreads();

  for (int64_t row = threadIdx.x; row < page_size; row += blockDim.x) {
    const code_t* code_row = codes + ((kv_head * pages + page) * page_size + row) * subvecs;
    float score = 0.0f;
    for (int64_t sub = 0; sub < subvecs; ++sub) {
      int64_t code = static_cast<int64_t>(code_row[sub]);
      code = max(static_cast<int64_t>(0), min(code, centroids - 1));
      score += lut[sub * centroids + code];
    }
    page_scores[row] = score;
  }
}

template <int KMAX, typename index_t>
__device__ __forceinline__ void insert_topk(float score, index_t index, float (&scores)[KMAX], index_t (&indices)[KMAX]) {
  if (!isfinite(score) || index < 0 || score <= scores[KMAX - 1]) {
    return;
  }
  int slot = KMAX - 1;
  while (slot > 0 && score > scores[slot - 1]) {
    scores[slot] = scores[slot - 1];
    indices[slot] = indices[slot - 1];
    --slot;
  }
  scores[slot] = score;
  indices[slot] = index;
}

__device__ __forceinline__ bool topk_candidate_better(
    float lhs_score,
    int64_t lhs_index,
    float rhs_score,
    int64_t rhs_index) {
  if (!isfinite(lhs_score) || lhs_index < 0) {
    return false;
  }
  if (!isfinite(rhs_score) || rhs_index < 0) {
    return true;
  }
  return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_index < rhs_index);
}

__device__ __forceinline__ bool topk_candidate_after(
    float score,
    int64_t index,
    float prev_score,
    int64_t prev_index) {
  if (!isfinite(score) || index < 0) {
    return false;
  }
  if (!isfinite(prev_score) || prev_index < 0) {
    return false;
  }
  return score < prev_score || (score == prev_score && index > prev_index);
}

template <typename code_t, int KMAX>
__global__ void gqa_causal_pq_topk_fused_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ codebooks,
    const code_t* __restrict__ codes,
    const int64_t* __restrict__ page_starts,
    int64_t* __restrict__ top_tokens,
    float* __restrict__ top_scores,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t dim,
    int64_t pages,
    int64_t page_size,
    int64_t subvecs,
    int64_t centroids,
    int64_t subdim,
    int64_t group_size,
    int64_t k_out,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  int64_t row_id = static_cast<int64_t>(blockIdx.x);
  int64_t rows_total = positions * heads;
  if (row_id >= rows_total) {
    return;
  }
  int64_t pos = row_id / heads;
  int64_t head = row_id - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t query_context_len = query_start + pos + 1;
  int64_t dyn_start = min(max(static_cast<int64_t>(0), static_prefix), query_context_len);
  int64_t indexed_end = max(dyn_start, query_context_len - max(static_cast<int64_t>(0), static_suffix));
  int64_t sealed_end =
      dyn_start + (max(static_cast<int64_t>(0), indexed_end - dyn_start) / page_size) * page_size;

  extern __shared__ unsigned char smem[];
  float* lut = reinterpret_cast<float*>(smem);
  float* candidate_scores = lut + subvecs * centroids;
  int32_t* candidate_indices = reinterpret_cast<int32_t*>(candidate_scores + blockDim.x * KMAX);
  float* reduce_scores = reinterpret_cast<float*>(candidate_indices + blockDim.x * KMAX);
  int32_t* reduce_indices = reinterpret_cast<int32_t*>(reduce_scores + blockDim.x);

  float local_scores[KMAX];
  int32_t local_indices[KMAX];
  #pragma unroll
  for (int i = 0; i < KMAX; ++i) {
    local_scores[i] = -INFINITY;
    local_indices[i] = -1;
  }

  const float* q = queries + (pos * heads + head) * dim;
  for (int64_t page = 0; page < pages; ++page) {
    int64_t page_start = page_starts[page];
    bool valid_page = page_start >= dyn_start && page_start + page_size <= sealed_end;
    if (!valid_page) {
      continue;
    }

    int64_t lut_entries = subvecs * centroids;
    for (int64_t entry = threadIdx.x; entry < lut_entries; entry += blockDim.x) {
      int64_t sub = entry / centroids;
      int64_t centroid = entry - sub * centroids;
      const float* cb = codebooks + ((((kv_head * pages + page) * subvecs + sub) * centroids + centroid) * subdim);
      const float* q_sub = q + sub * subdim;
      float dot = 0.0f;
      for (int64_t d = 0; d < subdim; ++d) {
        dot += q_sub[d] * cb[d];
      }
      lut[entry] = dot;
    }
    __syncthreads();

    for (int64_t page_row = threadIdx.x; page_row < page_size; page_row += blockDim.x) {
      const code_t* code_row = codes + ((kv_head * pages + page) * page_size + page_row) * subvecs;
      float score = 0.0f;
      for (int64_t sub = 0; sub < subvecs; ++sub) {
        int64_t code = static_cast<int64_t>(code_row[sub]);
        code = max(static_cast<int64_t>(0), min(code, centroids - 1));
        score += lut[sub * centroids + code];
      }
      insert_topk<KMAX>(
          score,
          static_cast<int32_t>(page * page_size + page_row),
          local_scores,
          local_indices);
    }
    __syncthreads();
  }

  int64_t base = static_cast<int64_t>(threadIdx.x) * KMAX;
  #pragma unroll
  for (int i = 0; i < KMAX; ++i) {
    candidate_scores[base + i] = local_scores[i];
    candidate_indices[base + i] = local_indices[i];
  }
  __syncthreads();

  float prev_score = INFINITY;
  int32_t prev_index = -1;
  const int64_t candidate_count = static_cast<int64_t>(blockDim.x) * KMAX;
  for (int64_t rank = 0; rank < k_out; ++rank) {
    float best_score = -INFINITY;
    int32_t best_index = -1;
    for (int64_t i = threadIdx.x; i < candidate_count; i += blockDim.x) {
      float score = candidate_scores[i];
      int32_t index = candidate_indices[i];
      if (rank > 0 && !topk_candidate_after(score, index, prev_score, prev_index)) {
        continue;
      }
      if (topk_candidate_better(score, index, best_score, best_index)) {
        best_score = score;
        best_index = index;
      }
    }
    reduce_scores[threadIdx.x] = best_score;
    reduce_indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        float other_score = reduce_scores[threadIdx.x + stride];
        int32_t other_index = reduce_indices[threadIdx.x + stride];
        if (topk_candidate_better(other_score, other_index, reduce_scores[threadIdx.x], reduce_indices[threadIdx.x])) {
          reduce_scores[threadIdx.x] = other_score;
          reduce_indices[threadIdx.x] = other_index;
        }
      }
      __syncthreads();
    }

    prev_score = reduce_scores[0];
    prev_index = reduce_indices[0];
    if (threadIdx.x == 0) {
      int64_t token = 0;
      if (prev_index >= 0) {
        int64_t page = prev_index / page_size;
        int64_t page_row = prev_index - page * page_size;
        token = page_starts[page] + page_row;
      }
      top_tokens[(row_id * k_out) + rank] = token;
      top_scores[(row_id * k_out) + rank] = prev_score;
    }
    __syncthreads();
  }
}

template <typename code_t>
__global__ void gqa_causal_pq_topk_fused_smallscan_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ codebooks,
    const code_t* __restrict__ codes,
    const int64_t* __restrict__ page_starts,
    int64_t* __restrict__ top_tokens,
    float* __restrict__ top_scores,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t dim,
    int64_t pages,
    int64_t page_size,
    int64_t subvecs,
    int64_t centroids,
    int64_t subdim,
    int64_t group_size,
    int64_t k_out,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  int64_t row_id = static_cast<int64_t>(blockIdx.x);
  int64_t rows_total = positions * heads;
  if (row_id >= rows_total) {
    return;
  }
  int64_t pos = row_id / heads;
  int64_t head = row_id - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t query_context_len = query_start + pos + 1;
  int64_t dyn_start = min(max(static_cast<int64_t>(0), static_prefix), query_context_len);
  int64_t indexed_end = max(dyn_start, query_context_len - max(static_cast<int64_t>(0), static_suffix));
  int64_t sealed_end =
      dyn_start + (max(static_cast<int64_t>(0), indexed_end - dyn_start) / page_size) * page_size;
  int64_t total_tokens = pages * page_size;

  extern __shared__ unsigned char smem[];
  float* lut = reinterpret_cast<float*>(smem);
  float* candidate_scores = lut + subvecs * centroids;
  int32_t* candidate_indices = reinterpret_cast<int32_t*>(candidate_scores + total_tokens);
  float* reduce_scores = reinterpret_cast<float*>(candidate_indices + total_tokens);
  int32_t* reduce_indices = reinterpret_cast<int32_t*>(reduce_scores + blockDim.x);

  for (int64_t token = threadIdx.x; token < total_tokens; token += blockDim.x) {
    candidate_scores[token] = -INFINITY;
    candidate_indices[token] = -1;
  }
  __syncthreads();

  const float* q = queries + (pos * heads + head) * dim;
  for (int64_t page = 0; page < pages; ++page) {
    int64_t page_start = page_starts[page];
    bool valid_page = page_start >= dyn_start && page_start + page_size <= sealed_end;
    if (!valid_page) {
      continue;
    }

    int64_t lut_entries = subvecs * centroids;
    for (int64_t entry = threadIdx.x; entry < lut_entries; entry += blockDim.x) {
      int64_t sub = entry / centroids;
      int64_t centroid = entry - sub * centroids;
      const float* cb = codebooks + ((((kv_head * pages + page) * subvecs + sub) * centroids + centroid) * subdim);
      const float* q_sub = q + sub * subdim;
      float dot = 0.0f;
      for (int64_t d = 0; d < subdim; ++d) {
        dot += q_sub[d] * cb[d];
      }
      lut[entry] = dot;
    }
    __syncthreads();

    for (int64_t page_row = threadIdx.x; page_row < page_size; page_row += blockDim.x) {
      const code_t* code_row = codes + ((kv_head * pages + page) * page_size + page_row) * subvecs;
      float score = 0.0f;
      for (int64_t sub = 0; sub < subvecs; ++sub) {
        int64_t code = static_cast<int64_t>(code_row[sub]);
        code = max(static_cast<int64_t>(0), min(code, centroids - 1));
        score += lut[sub * centroids + code];
      }
      int64_t token_ordinal = page * page_size + page_row;
      candidate_scores[token_ordinal] = score;
      candidate_indices[token_ordinal] = static_cast<int32_t>(token_ordinal);
    }
    __syncthreads();
  }

  float prev_score = INFINITY;
  int32_t prev_index = -1;
  for (int64_t rank = 0; rank < k_out; ++rank) {
    float best_score = -INFINITY;
    int32_t best_index = -1;
    for (int64_t i = threadIdx.x; i < total_tokens; i += blockDim.x) {
      float score = candidate_scores[i];
      int32_t index = candidate_indices[i];
      if (rank > 0 && !topk_candidate_after(score, index, prev_score, prev_index)) {
        continue;
      }
      if (topk_candidate_better(score, index, best_score, best_index)) {
        best_score = score;
        best_index = index;
      }
    }
    reduce_scores[threadIdx.x] = best_score;
    reduce_indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        float other_score = reduce_scores[threadIdx.x + stride];
        int32_t other_index = reduce_indices[threadIdx.x + stride];
        if (topk_candidate_better(other_score, other_index, reduce_scores[threadIdx.x], reduce_indices[threadIdx.x])) {
          reduce_scores[threadIdx.x] = other_score;
          reduce_indices[threadIdx.x] = other_index;
        }
      }
      __syncthreads();
    }

    prev_score = reduce_scores[0];
    prev_index = reduce_indices[0];
    if (threadIdx.x == 0) {
      int64_t token = 0;
      if (prev_index >= 0) {
        int64_t page = prev_index / page_size;
        int64_t page_row = prev_index - page * page_size;
        token = page_starts[page] + page_row;
      }
      top_tokens[(row_id * k_out) + rank] = token;
      top_scores[(row_id * k_out) + rank] = prev_score;
    }
    __syncthreads();
  }
}

template <typename code_t>
__global__ void gqa_causal_pq_top_pages_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ codebooks,
    const code_t* __restrict__ codes,
    const int64_t* __restrict__ page_starts,
    int64_t* __restrict__ top_pages,
    float* __restrict__ top_scores,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t dim,
    int64_t pages,
    int64_t page_size,
    int64_t subvecs,
    int64_t centroids,
    int64_t subdim,
    int64_t group_size,
    int64_t page_budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  int64_t row_id = static_cast<int64_t>(blockIdx.x);
  int64_t rows_total = positions * heads;
  if (row_id >= rows_total) {
    return;
  }
  int64_t pos = row_id / heads;
  int64_t head = row_id - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t query_context_len = query_start + pos + 1;
  int64_t dyn_start = min(max(static_cast<int64_t>(0), static_prefix), query_context_len);
  int64_t indexed_end = max(dyn_start, query_context_len - max(static_cast<int64_t>(0), static_suffix));
  int64_t sealed_end =
      dyn_start + (max(static_cast<int64_t>(0), indexed_end - dyn_start) / page_size) * page_size;

  extern __shared__ unsigned char smem[];
  float* lut = reinterpret_cast<float*>(smem);
  float* page_scores = lut + subvecs * centroids;
  int32_t* page_ids = reinterpret_cast<int32_t*>(page_scores + pages);
  float* reduce_scores = reinterpret_cast<float*>(page_ids + pages);
  int32_t* reduce_ids = reinterpret_cast<int32_t*>(reduce_scores + blockDim.x);
  int32_t* reduce_rows = reduce_ids + blockDim.x;

  for (int64_t page = threadIdx.x; page < pages; page += blockDim.x) {
    page_scores[page] = -INFINITY;
    page_ids[page] = -1;
  }
  __syncthreads();

  const float* q = queries + (pos * heads + head) * dim;
  for (int64_t page = 0; page < pages; ++page) {
    int64_t page_start = page_starts[page];
    bool valid_page = page_start >= dyn_start && page_start + page_size <= sealed_end;
    if (!valid_page) {
      continue;
    }

    int64_t lut_entries = subvecs * centroids;
    for (int64_t entry = threadIdx.x; entry < lut_entries; entry += blockDim.x) {
      int64_t sub = entry / centroids;
      int64_t centroid = entry - sub * centroids;
      const float* cb = codebooks + ((((kv_head * pages + page) * subvecs + sub) * centroids + centroid) * subdim);
      const float* q_sub = q + sub * subdim;
      float dot = 0.0f;
      for (int64_t d = 0; d < subdim; ++d) {
        dot += q_sub[d] * cb[d];
      }
      lut[entry] = dot;
    }
    __syncthreads();

    float local_best = -INFINITY;
    int32_t local_row = -1;
    for (int64_t page_row = threadIdx.x; page_row < page_size; page_row += blockDim.x) {
      const code_t* code_row = codes + ((kv_head * pages + page) * page_size + page_row) * subvecs;
      float score = 0.0f;
      for (int64_t sub = 0; sub < subvecs; ++sub) {
        int64_t code = static_cast<int64_t>(code_row[sub]);
        code = max(static_cast<int64_t>(0), min(code, centroids - 1));
        score += lut[sub * centroids + code];
      }
      if (score > local_best || (score == local_best && (local_row < 0 || page_row < local_row))) {
        local_best = score;
        local_row = static_cast<int32_t>(page_row);
      }
    }
    reduce_scores[threadIdx.x] = local_best;
    reduce_rows[threadIdx.x] = local_row;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        float other_score = reduce_scores[threadIdx.x + stride];
        int32_t other_row = reduce_rows[threadIdx.x + stride];
        if (other_score > reduce_scores[threadIdx.x] ||
            (other_score == reduce_scores[threadIdx.x] &&
             other_row >= 0 &&
             (reduce_rows[threadIdx.x] < 0 || other_row < reduce_rows[threadIdx.x]))) {
          reduce_scores[threadIdx.x] = other_score;
          reduce_rows[threadIdx.x] = other_row;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && isfinite(reduce_scores[0]) && reduce_rows[0] >= 0) {
      int64_t token = page_start + static_cast<int64_t>(reduce_rows[0]);
      page_scores[page] = reduce_scores[0];
      page_ids[page] = static_cast<int32_t>(token / max(static_cast<int64_t>(1), page_size));
    }
    __syncthreads();
  }

  for (int64_t rank = 0; rank < page_budget; ++rank) {
    float best_score = -INFINITY;
    int32_t best_page = -1;
    for (int64_t page = threadIdx.x; page < pages; page += blockDim.x) {
      float score = page_scores[page];
      int32_t page_id = page_ids[page];
      if (!isfinite(score) || page_id < 0) {
        continue;
      }
      bool duplicate = false;
      for (int64_t prev = 0; prev < rank; ++prev) {
        if (top_pages[row_id * page_budget + prev] == static_cast<int64_t>(page_id)) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) {
        continue;
      }
      if (score > best_score || (score == best_score && (best_page < 0 || page_id < best_page))) {
        best_score = score;
        best_page = page_id;
      }
    }
    reduce_scores[threadIdx.x] = best_score;
    reduce_ids[threadIdx.x] = best_page;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        float other_score = reduce_scores[threadIdx.x + stride];
        int32_t other_page = reduce_ids[threadIdx.x + stride];
        if (other_score > reduce_scores[threadIdx.x] ||
            (other_score == reduce_scores[threadIdx.x] &&
             other_page >= 0 &&
             (reduce_ids[threadIdx.x] < 0 || other_page < reduce_ids[threadIdx.x]))) {
          reduce_scores[threadIdx.x] = other_score;
          reduce_ids[threadIdx.x] = other_page;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      top_pages[row_id * page_budget + rank] = static_cast<int64_t>(reduce_ids[0]);
      top_scores[row_id * page_budget + rank] = reduce_scores[0];
    }
    __syncthreads();
  }
}
