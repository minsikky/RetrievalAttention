#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

int64_t decode_tail_pages_per_block() {
  const char* value = std::getenv("SELECTOR_PQ_DECODE_TAIL_PAGES_PER_BLOCK");
  if (value == nullptr || value[0] == '\0') {
    return 1;
  }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0) {
    return 1;
  }
  return std::min<int64_t>(64, std::max<int64_t>(1, static_cast<int64_t>(parsed)));
}

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* __restrict__ ptr, int64_t index) {
  return static_cast<float>(ptr[index]);
}

template <typename scalar_t>
__device__ __forceinline__ float load_strided3_as_float(
    const scalar_t* __restrict__ ptr,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2) {
  return load_as_float(ptr, i0 * stride0 + i1 * stride1 + i2 * stride2);
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float logaddexp_device(float a, float b) {
  if (!isfinite(a)) {
    return b;
  }
  if (!isfinite(b)) {
    return a;
  }
  float m = fmaxf(a, b);
  return m + logf(expf(a - m) + expf(b - m));
}

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

__global__ void exact_selected_logits_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    const int64_t* __restrict__ tokens,
    float* __restrict__ logits,
    int64_t heads,
    int64_t selected,
    int64_t dim,
    int64_t total_tokens,
    float scale) {
  int64_t pair = static_cast<int64_t>(blockIdx.x);
  int64_t total_pairs = heads * selected;
  if (pair >= total_pairs) {
    return;
  }
  int64_t head = pair / selected;
  int64_t token = tokens[pair];
  token = max(static_cast<int64_t>(0), min(token, total_tokens - 1));

  const float* q = queries + head * dim;
  const float* k = keys + token * dim;
  float partial = 0.0f;
  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    partial += q[d] * k[d];
  }

  extern __shared__ float reduce_buf[];
  reduce_buf[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    logits[pair] = reduce_buf[0] * scale;
  }
}

__global__ void exact_selected_output_kernel(
    const float* __restrict__ values,
    const int64_t* __restrict__ tokens,
    const float* __restrict__ logits,
    float* __restrict__ outputs,
    int64_t heads,
    int64_t selected,
    int64_t dim,
    int64_t total_tokens) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }

  __shared__ float max_logit;
  __shared__ float denom;
  if (threadIdx.x == 0) {
    float local_max = -INFINITY;
    const float* head_logits = logits + head * selected;
    for (int64_t sel = 0; sel < selected; ++sel) {
      local_max = fmaxf(local_max, head_logits[sel]);
    }
    float local_denom = 0.0f;
    for (int64_t sel = 0; sel < selected; ++sel) {
      local_denom += expf(head_logits[sel] - local_max);
    }
    max_logit = local_max;
    denom = fmaxf(local_denom, 1.0e-20f);
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float accum = 0.0f;
    const float* head_logits = logits + head * selected;
    const int64_t* head_tokens = tokens + head * selected;
    for (int64_t sel = 0; sel < selected; ++sel) {
      int64_t token = head_tokens[sel];
      token = max(static_cast<int64_t>(0), min(token, total_tokens - 1));
      float weight = expf(head_logits[sel] - max_logit) / denom;
      accum += weight * values[token * dim + d];
    }
    outputs[head * dim + d] = accum;
  }
}

template <typename key_t>
__global__ void gqa_exact_selected_logits_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    const int64_t* __restrict__ tokens,
    float* __restrict__ logits,
    int64_t heads,
    int64_t kv_heads,
    int64_t selected,
    int64_t dim,
    int64_t total_tokens,
    int64_t group_size,
    float scale) {
  int64_t pair = static_cast<int64_t>(blockIdx.x);
  int64_t total_pairs = heads * selected;
  if (pair >= total_pairs) {
    return;
  }
  int64_t head = pair / selected;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t token = tokens[pair];
  token = max(static_cast<int64_t>(0), min(token, total_tokens - 1));

  const float* q = queries + head * dim;
  const key_t* k = keys + (kv_head * total_tokens + token) * dim;
  float partial = 0.0f;
  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    partial += q[d] * load_as_float(k, d);
  }

  extern __shared__ float reduce_buf[];
  reduce_buf[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    logits[pair] = reduce_buf[0] * scale;
  }
}

template <typename value_t>
__global__ void gqa_exact_selected_output_kernel(
    const value_t* __restrict__ values,
    const int64_t* __restrict__ tokens,
    const float* __restrict__ logits,
    float* __restrict__ outputs,
    int64_t heads,
    int64_t kv_heads,
    int64_t selected,
    int64_t dim,
    int64_t total_tokens,
    int64_t group_size) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }
  int64_t kv_head = min(head / group_size, kv_heads - 1);

  __shared__ float max_logit;
  __shared__ float denom;
  if (threadIdx.x == 0) {
    float local_max = -INFINITY;
    const float* head_logits = logits + head * selected;
    for (int64_t sel = 0; sel < selected; ++sel) {
      local_max = fmaxf(local_max, head_logits[sel]);
    }
    float local_denom = 0.0f;
    for (int64_t sel = 0; sel < selected; ++sel) {
      local_denom += expf(head_logits[sel] - local_max);
    }
    max_logit = local_max;
    denom = fmaxf(local_denom, 1.0e-20f);
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float accum = 0.0f;
    const float* head_logits = logits + head * selected;
    const int64_t* head_tokens = tokens + head * selected;
    for (int64_t sel = 0; sel < selected; ++sel) {
      int64_t token = head_tokens[sel];
      token = max(static_cast<int64_t>(0), min(token, total_tokens - 1));
      float weight = expf(head_logits[sel] - max_logit) / denom;
      accum += weight * load_as_float(values, (kv_head * total_tokens + token) * dim + d);
    }
    outputs[head * dim + d] = accum;
  }
}

__global__ void gqa_causal_map_topk_tokens_kernel(
    const int64_t* __restrict__ top_indices,
    const int64_t* __restrict__ page_starts,
    int64_t* __restrict__ top_tokens,
    int64_t positions,
    int64_t heads,
    int64_t k,
    int64_t page_size) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = positions * heads * k;
  if (linear >= total) {
    return;
  }
  int64_t idx = top_indices[linear];
  int64_t page = idx / page_size;
  int64_t row = idx - page * page_size;
  top_tokens[linear] = page_starts[page] + row;
}

__device__ inline bool token_in_base_window(
    int64_t token,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t* prefix_end_out,
    int64_t* base_tail_start_out) {
  int64_t prefix_end = min(max(static_cast<int64_t>(0), static_prefix), query_context_len);
  int64_t dyn_start = prefix_end;
  int64_t indexed_end = max(dyn_start, query_context_len - max(static_cast<int64_t>(0), static_suffix));
  int64_t sealed_end =
      dyn_start + (max(static_cast<int64_t>(0), indexed_end - dyn_start) / page_size) * page_size;
  int64_t base_tail_start = max(sealed_end, prefix_end);
  if (prefix_end_out != nullptr) {
    *prefix_end_out = prefix_end;
  }
  if (base_tail_start_out != nullptr) {
    *base_tail_start_out = base_tail_start;
  }
  return (token >= 0 && token < prefix_end) || (token >= base_tail_start && token < query_context_len);
}

__global__ void gqa_causal_exact_selected_attention_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    const float* __restrict__ values,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_logits,
    float* __restrict__ outputs,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t selected,
    int64_t dim,
    int64_t total_tokens,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    float scale) {
  int64_t pair = static_cast<int64_t>(blockIdx.x);
  int64_t total_pairs = positions * heads;
  if (pair >= total_pairs) {
    return;
  }
  int64_t pos = pair / heads;
  int64_t head = pair - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t query_context_len = min(query_start + pos + 1, total_tokens);

  const float* q = queries + (pos * heads + head) * dim;
  const int64_t* head_tokens = ranked_tokens + (pos * heads + head) * selected;
  const float* head_scores = ranked_logits + (pos * heads + head) * selected;

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

  __shared__ float shared_max_logit;
  __shared__ float shared_denom;
  if (threadIdx.x == 0) {
    float max_logit = -INFINITY;
    for (int64_t token = 0; token < prefix_end; ++token) {
      const float* k = keys + (kv_head * total_tokens + token) * dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        dot += q[d] * k[d];
      }
      max_logit = fmaxf(max_logit, dot * scale);
    }
    for (int64_t token = base_tail_start; token < query_context_len; ++token) {
      const float* k = keys + (kv_head * total_tokens + token) * dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        dot += q[d] * k[d];
      }
      max_logit = fmaxf(max_logit, dot * scale);
    }
    for (int64_t sel = 0; sel < selected; ++sel) {
      float selector_score = head_scores[sel];
      int64_t token = head_tokens[sel];
      if (!isfinite(selector_score) || token < 0 || token >= query_context_len) {
        continue;
      }
      if (token_in_base_window(
              token,
              query_context_len,
              static_prefix,
              static_suffix,
              page_size,
              nullptr,
              nullptr)) {
        continue;
      }
      const float* k = keys + (kv_head * total_tokens + token) * dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        dot += q[d] * k[d];
      }
      max_logit = fmaxf(max_logit, dot * scale);
    }

    float denom = 0.0f;
    if (isfinite(max_logit)) {
      for (int64_t token = 0; token < prefix_end; ++token) {
        const float* k = keys + (kv_head * total_tokens + token) * dim;
        float dot = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
          dot += q[d] * k[d];
        }
        denom += expf(dot * scale - max_logit);
      }
      for (int64_t token = base_tail_start; token < query_context_len; ++token) {
        const float* k = keys + (kv_head * total_tokens + token) * dim;
        float dot = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
          dot += q[d] * k[d];
        }
        denom += expf(dot * scale - max_logit);
      }
      for (int64_t sel = 0; sel < selected; ++sel) {
        float selector_score = head_scores[sel];
        int64_t token = head_tokens[sel];
        if (!isfinite(selector_score) || token < 0 || token >= query_context_len) {
          continue;
        }
        if (token_in_base_window(
                token,
                query_context_len,
                static_prefix,
                static_suffix,
                page_size,
                nullptr,
                nullptr)) {
          continue;
        }
        const float* k = keys + (kv_head * total_tokens + token) * dim;
        float dot = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
          dot += q[d] * k[d];
        }
        denom += expf(dot * scale - max_logit);
      }
    }
    shared_max_logit = max_logit;
    shared_denom = fmaxf(denom, 1.0e-20f);
  }
  __syncthreads();

  float max_logit = shared_max_logit;
  float denom = shared_denom;
  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float accum = 0.0f;
    if (isfinite(max_logit)) {
      for (int64_t token = 0; token < prefix_end; ++token) {
        const float* k = keys + (kv_head * total_tokens + token) * dim;
        float dot = 0.0f;
        for (int64_t kd = 0; kd < dim; ++kd) {
          dot += q[kd] * k[kd];
        }
        float weight = expf(dot * scale - max_logit) / denom;
        accum += weight * values[(kv_head * total_tokens + token) * dim + d];
      }
      for (int64_t token = base_tail_start; token < query_context_len; ++token) {
        const float* k = keys + (kv_head * total_tokens + token) * dim;
        float dot = 0.0f;
        for (int64_t kd = 0; kd < dim; ++kd) {
          dot += q[kd] * k[kd];
        }
        float weight = expf(dot * scale - max_logit) / denom;
        accum += weight * values[(kv_head * total_tokens + token) * dim + d];
      }
      for (int64_t sel = 0; sel < selected; ++sel) {
        float selector_score = head_scores[sel];
        int64_t token = head_tokens[sel];
        if (!isfinite(selector_score) || token < 0 || token >= query_context_len) {
          continue;
        }
        if (token_in_base_window(
                token,
                query_context_len,
                static_prefix,
                static_suffix,
                page_size,
                nullptr,
                nullptr)) {
          continue;
        }
        const float* k = keys + (kv_head * total_tokens + token) * dim;
        float dot = 0.0f;
        for (int64_t kd = 0; kd < dim; ++kd) {
          dot += q[kd] * k[kd];
        }
        float weight = expf(dot * scale - max_logit) / denom;
        accum += weight * values[(kv_head * total_tokens + token) * dim + d];
      }
    }
    outputs[pair * dim + d] = accum;
  }
}

__global__ void gqa_causal_build_selected_tokens_kernel(
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    int64_t* __restrict__ selected_tokens,
    int64_t* __restrict__ selected_counts,
    int64_t positions,
    int64_t heads,
    int64_t ranked,
    int64_t max_selected,
    int64_t total_tokens,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size) {
  int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total_pairs = positions * heads;
  if (pair >= total_pairs) {
    return;
  }
  int64_t pos = pair / heads;
  int64_t query_context_len = min(query_start + pos + 1, total_tokens);
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

  int64_t out_offset = pair * max_selected;
  int64_t count = 0;
  for (int64_t token = 0; token < prefix_end && count < max_selected; ++token) {
    selected_tokens[out_offset + count] = token;
    ++count;
  }
  for (int64_t token = base_tail_start; token < query_context_len && count < max_selected; ++token) {
    selected_tokens[out_offset + count] = token;
    ++count;
  }
  const int64_t* ranked_row = ranked_tokens + pair * ranked;
  const float* score_row = ranked_scores + pair * ranked;
  for (int64_t sel = 0; sel < ranked && count < max_selected; ++sel) {
    float selector_score = score_row[sel];
    int64_t token = ranked_row[sel];
    if (!isfinite(selector_score) || token < 0 || token >= query_context_len) {
      continue;
    }
    if (token_in_base_window(
            token,
            query_context_len,
            static_prefix,
            static_suffix,
            page_size,
            nullptr,
            nullptr)) {
      continue;
    }
    selected_tokens[out_offset + count] = token;
    ++count;
  }
  selected_counts[pair] = count;
  for (int64_t i = count; i < max_selected; ++i) {
    selected_tokens[out_offset + i] = -1;
  }
}

__global__ void gqa_causal_selected_logits_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    const int64_t* __restrict__ selected_tokens,
    const int64_t* __restrict__ selected_counts,
    float* __restrict__ logits,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t max_selected,
    int64_t dim,
    int64_t total_tokens,
    int64_t group_size,
    float scale) {
  int64_t pair = static_cast<int64_t>(blockIdx.x);
  int64_t total_pairs = positions * heads * max_selected;
  if (pair >= total_pairs) {
    return;
  }
  int64_t sel = pair % max_selected;
  int64_t qh = pair / max_selected;
  int64_t head = qh % heads;
  int64_t pos = qh / heads;
  int64_t count = selected_counts[qh];
  int64_t token = selected_tokens[pair];
  if (sel >= count || token < 0 || token >= total_tokens) {
    if (threadIdx.x == 0) {
      logits[pair] = -INFINITY;
    }
    return;
  }

  int64_t kv_head = min(head / group_size, kv_heads - 1);
  const float* q = queries + (pos * heads + head) * dim;
  const float* k = keys + (kv_head * total_tokens + token) * dim;
  float partial = 0.0f;
  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    partial += q[d] * k[d];
  }

  extern __shared__ float reduce_buf[];
  reduce_buf[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    logits[pair] = reduce_buf[0] * scale;
  }
}

__global__ void gqa_causal_selected_output_kernel(
    const float* __restrict__ values,
    const int64_t* __restrict__ selected_tokens,
    const int64_t* __restrict__ selected_counts,
    const float* __restrict__ logits,
    float* __restrict__ outputs,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t max_selected,
    int64_t dim,
    int64_t total_tokens,
    int64_t group_size) {
  int64_t qh = static_cast<int64_t>(blockIdx.x);
  int64_t total_qh = positions * heads;
  if (qh >= total_qh) {
    return;
  }
  int64_t head = qh % heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t count = selected_counts[qh];

  __shared__ float max_logit;
  __shared__ float denom;
  if (threadIdx.x == 0) {
    float local_max = -INFINITY;
    const float* row_logits = logits + qh * max_selected;
    for (int64_t sel = 0; sel < count; ++sel) {
      local_max = fmaxf(local_max, row_logits[sel]);
    }
    float local_denom = 0.0f;
    if (isfinite(local_max)) {
      for (int64_t sel = 0; sel < count; ++sel) {
        local_denom += expf(row_logits[sel] - local_max);
      }
    }
    max_logit = local_max;
    denom = fmaxf(local_denom, 1.0e-20f);
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float accum = 0.0f;
    if (isfinite(max_logit)) {
      const float* row_logits = logits + qh * max_selected;
      const int64_t* row_tokens = selected_tokens + qh * max_selected;
      for (int64_t sel = 0; sel < count; ++sel) {
        int64_t token = row_tokens[sel];
        if (token < 0 || token >= total_tokens) {
          continue;
        }
        float weight = expf(row_logits[sel] - max_logit) / denom;
        accum += weight * values[(kv_head * total_tokens + token) * dim + d];
      }
    }
    outputs[qh * dim + d] = accum;
  }
}

template <typename key_t, typename value_t>
__global__ void gqa_causal_warp_tiled_selected_attention_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    const value_t* __restrict__ values,
    const int64_t* __restrict__ selected_tokens,
    const int64_t* __restrict__ selected_counts,
    float* __restrict__ outputs,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t max_selected,
    int64_t dim,
    int64_t total_tokens,
    int64_t group_size,
    float scale) {
  constexpr int warps_per_block = 8;
  int64_t qh = static_cast<int64_t>(blockIdx.x);
  int64_t total_qh = positions * heads;
  if (qh >= total_qh) {
    return;
  }
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int64_t pos = qh / heads;
  int64_t head = qh - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t count = selected_counts[qh];

  extern __shared__ float shared_logits[];
  const float* q = queries + (pos * heads + head) * dim;
  const int64_t* row_tokens = selected_tokens + qh * max_selected;

  for (int64_t base = 0; base < count; base += warps_per_block) {
    int64_t sel = base + warp;
    if (warp < warps_per_block && sel < count) {
      int64_t token = row_tokens[sel];
      float partial = 0.0f;
      if (token >= 0 && token < total_tokens) {
        const key_t* k = keys + (kv_head * total_tokens + token) * dim;
        for (int64_t d = lane; d < dim; d += 32) {
          partial += q[d] * load_as_float(k, d);
        }
      }
      float dot = warp_reduce_sum(partial);
      if (lane == 0) {
        shared_logits[sel] = (token >= 0 && token < total_tokens) ? dot * scale : -INFINITY;
      }
    }
  }
  __syncthreads();

  __shared__ float max_logit;
  __shared__ float denom;
  if (threadIdx.x == 0) {
    float local_max = -INFINITY;
    for (int64_t sel = 0; sel < count; ++sel) {
      local_max = fmaxf(local_max, shared_logits[sel]);
    }
    float local_denom = 0.0f;
    if (isfinite(local_max)) {
      for (int64_t sel = 0; sel < count; ++sel) {
        local_denom += expf(shared_logits[sel] - local_max);
      }
    }
    max_logit = local_max;
    denom = fmaxf(local_denom, 1.0e-20f);
    if (isfinite(local_max)) {
      float inv_denom = 1.0f / denom;
      for (int64_t sel = 0; sel < count; ++sel) {
        shared_logits[sel] = expf(shared_logits[sel] - local_max) * inv_denom;
      }
    }
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float accum = 0.0f;
    if (isfinite(max_logit)) {
      for (int64_t sel = 0; sel < count; ++sel) {
        int64_t token = row_tokens[sel];
        if (token < 0 || token >= total_tokens) {
          continue;
        }
        float weight = shared_logits[sel];
        accum += weight * load_as_float(values, (kv_head * total_tokens + token) * dim + d);
      }
    }
    outputs[qh * dim + d] = accum;
  }
}

template <typename key_t, typename value_t, typename vcode_t>
__global__ void gqa_causal_warp_tiled_vpq_selected_attention_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    const value_t* __restrict__ values,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ selected_tokens,
    const int64_t* __restrict__ selected_counts,
    float* __restrict__ outputs,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t max_selected,
    int64_t dim,
    int64_t total_tokens,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t exact_value_top,
    float scale) {
  constexpr int warps_per_block = 8;
  int64_t qh = static_cast<int64_t>(blockIdx.x);
  int64_t total_qh = positions * heads;
  if (qh >= total_qh) {
    return;
  }
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int64_t pos = qh / heads;
  int64_t head = qh - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t count = selected_counts[qh];
  const bool exact_values_by_selector_rank = exact_value_top < 0;
  const int64_t exact_value_limit = exact_values_by_selector_rank ? -exact_value_top : exact_value_top;

  extern __shared__ float shared_logits[];
  unsigned char* shared_exact = reinterpret_cast<unsigned char*>(shared_logits + max_selected);
  const float* q = queries + (pos * heads + head) * dim;
  const int64_t* row_tokens = selected_tokens + qh * max_selected;

  for (int64_t base = 0; base < count; base += warps_per_block) {
    int64_t sel = base + warp;
    if (warp < warps_per_block && sel < count) {
      int64_t token = row_tokens[sel];
      float partial = 0.0f;
      if (token >= 0 && token < total_tokens) {
        const key_t* k = keys + (kv_head * total_tokens + token) * dim;
        for (int64_t d = lane; d < dim; d += 32) {
          partial += q[d] * load_as_float(k, d);
        }
      }
      float dot = warp_reduce_sum(partial);
      if (lane == 0) {
        shared_logits[sel] = (token >= 0 && token < total_tokens) ? dot * scale : -INFINITY;
      }
    }
  }
  __syncthreads();

  __shared__ float max_logit;
  __shared__ float denom;
  if (threadIdx.x == 0) {
    float local_max = -INFINITY;
    for (int64_t sel = 0; sel < count; ++sel) {
      local_max = fmaxf(local_max, shared_logits[sel]);
    }
    float local_denom = 0.0f;
    if (isfinite(local_max)) {
      for (int64_t sel = 0; sel < count; ++sel) {
        local_denom += expf(shared_logits[sel] - local_max);
      }
    }
    max_logit = local_max;
    denom = fmaxf(local_denom, 1.0e-20f);
    if (exact_value_limit > 0) {
      if (exact_value_limit >= count) {
        for (int64_t sel = 0; sel < count; ++sel) {
          shared_exact[sel] = isfinite(shared_logits[sel]) ? 1 : 0;
        }
      } else if (exact_values_by_selector_rank) {
        for (int64_t sel = 0; sel < count; ++sel) {
          shared_exact[sel] = (sel < exact_value_limit && isfinite(shared_logits[sel])) ? 1 : 0;
        }
      } else {
        for (int64_t sel = 0; sel < count; ++sel) {
          shared_exact[sel] = 0;
        }
        int64_t exact_count = min(count, exact_value_limit);
        for (int64_t rank = 0; rank < exact_count; ++rank) {
          int64_t best_sel = -1;
          float best_logit = -INFINITY;
          for (int64_t sel = 0; sel < count; ++sel) {
            if (shared_exact[sel]) {
              continue;
            }
            float logit = shared_logits[sel];
            if (logit > best_logit || (logit == best_logit && (best_sel < 0 || sel < best_sel))) {
              best_logit = logit;
              best_sel = sel;
            }
          }
          if (best_sel >= 0 && isfinite(best_logit)) {
            shared_exact[best_sel] = 1;
          }
        }
      }
    }
    if (isfinite(local_max)) {
      float inv_denom = 1.0f / denom;
      for (int64_t sel = 0; sel < count; ++sel) {
        shared_logits[sel] = expf(shared_logits[sel] - local_max) * inv_denom;
      }
    }
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float accum = 0.0f;
    if (isfinite(max_logit)) {
      for (int64_t sel = 0; sel < count; ++sel) {
        int64_t token = row_tokens[sel];
        if (token < 0 || token >= total_tokens) {
          continue;
        }
        float value = 0.0f;
        bool value_loaded = false;
        bool exact_value = exact_value_limit > 0 && shared_exact[sel] != 0;
        if (!exact_value && pages > 0 && value_subvecs > 0 && value_subdim > 0) {
          int64_t first_start = page_starts[0];
          int64_t page = (token - first_start) / page_size;
          if (token >= first_start && page >= 0 && page < pages) {
            int64_t row = token - page_starts[page];
            if (row >= 0 && row < page_size) {
              int64_t sub = d / value_subdim;
              int64_t sub_d = d - sub * value_subdim;
              if (sub >= 0 && sub < value_subvecs) {
                int64_t code = static_cast<int64_t>(
                    value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
                code = max(static_cast<int64_t>(0), min(code, value_centroids - 1));
                value = value_codebooks
                    [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
                     sub_d];
                value_loaded = true;
              }
            }
          }
        }
        if (exact_value || !value_loaded) {
          value = load_as_float(values, (kv_head * total_tokens + token) * dim + d);
        }
        float weight = shared_logits[sel];
        accum += weight * value;
      }
    }
    outputs[qh * dim + d] = accum;
  }
}

template <typename key_t, typename value_t>
__global__ void gqa_causal_fused_selected_attention_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    const value_t* __restrict__ values,
    const int64_t* __restrict__ selected_tokens,
    const int64_t* __restrict__ selected_counts,
    float* __restrict__ outputs,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t max_selected,
    int64_t dim,
    int64_t total_tokens,
    int64_t group_size,
    float scale) {
  int64_t qh = static_cast<int64_t>(blockIdx.x);
  int64_t total_qh = positions * heads;
  if (qh >= total_qh) {
    return;
  }
  int64_t pos = qh / heads;
  int64_t head = qh - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t count = selected_counts[qh];

  extern __shared__ float shared[];
  float* shared_logits = shared;
  float* reduce_buf = shared + max_selected;
  const float* q = queries + (pos * heads + head) * dim;
  const int64_t* row_tokens = selected_tokens + qh * max_selected;

  for (int64_t sel = 0; sel < count; ++sel) {
    int64_t token = row_tokens[sel];
    float partial = 0.0f;
    if (token >= 0 && token < total_tokens) {
      const key_t* k = keys + (kv_head * total_tokens + token) * dim;
      for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
        partial += q[d] * load_as_float(k, d);
      }
    }
    reduce_buf[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      shared_logits[sel] = (token >= 0 && token < total_tokens) ? reduce_buf[0] * scale : -INFINITY;
    }
    __syncthreads();
  }

  __shared__ float max_logit;
  __shared__ float denom;
  if (threadIdx.x == 0) {
    float local_max = -INFINITY;
    for (int64_t sel = 0; sel < count; ++sel) {
      local_max = fmaxf(local_max, shared_logits[sel]);
    }
    float local_denom = 0.0f;
    if (isfinite(local_max)) {
      for (int64_t sel = 0; sel < count; ++sel) {
        local_denom += expf(shared_logits[sel] - local_max);
      }
    }
    max_logit = local_max;
    denom = fmaxf(local_denom, 1.0e-20f);
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float accum = 0.0f;
    if (isfinite(max_logit)) {
      for (int64_t sel = 0; sel < count; ++sel) {
        int64_t token = row_tokens[sel];
        if (token < 0 || token >= total_tokens) {
          continue;
        }
        float weight = expf(shared_logits[sel] - max_logit) / denom;
        accum += weight * load_as_float(values, (kv_head * total_tokens + token) * dim + d);
      }
    }
    outputs[qh * dim + d] = accum;
  }
}

template <typename key_t, typename value_t, typename vcode_t>
__global__ void gqa_causal_fused_vpq_selected_attention_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    const value_t* __restrict__ values,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ selected_tokens,
    const int64_t* __restrict__ selected_counts,
    float* __restrict__ outputs,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t max_selected,
    int64_t dim,
    int64_t total_tokens,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    float scale) {
  int64_t qh = static_cast<int64_t>(blockIdx.x);
  int64_t total_qh = positions * heads;
  if (qh >= total_qh) {
    return;
  }
  int64_t pos = qh / heads;
  int64_t head = qh - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t count = selected_counts[qh];

  extern __shared__ float shared[];
  float* shared_logits = shared;
  float* reduce_buf = shared + max_selected;
  const float* q = queries + (pos * heads + head) * dim;
  const int64_t* row_tokens = selected_tokens + qh * max_selected;

  for (int64_t sel = 0; sel < count; ++sel) {
    int64_t token = row_tokens[sel];
    float partial = 0.0f;
    if (token >= 0 && token < total_tokens) {
      const int64_t key_base = (kv_head * total_tokens + token) * dim;
      for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
        partial += q[d] * load_as_float(keys, key_base + d);
      }
    }
    reduce_buf[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      shared_logits[sel] = (token >= 0 && token < total_tokens) ? reduce_buf[0] * scale : -INFINITY;
    }
    __syncthreads();
  }

  __shared__ float max_logit;
  __shared__ float denom;
  if (threadIdx.x == 0) {
    float local_max = -INFINITY;
    for (int64_t sel = 0; sel < count; ++sel) {
      local_max = fmaxf(local_max, shared_logits[sel]);
    }
    float local_denom = 0.0f;
    if (isfinite(local_max)) {
      for (int64_t sel = 0; sel < count; ++sel) {
        local_denom += expf(shared_logits[sel] - local_max);
      }
    }
    max_logit = local_max;
    denom = fmaxf(local_denom, 1.0e-20f);
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float accum = 0.0f;
    if (isfinite(max_logit)) {
      for (int64_t sel = 0; sel < count; ++sel) {
        int64_t token = row_tokens[sel];
        if (token < 0 || token >= total_tokens) {
          continue;
        }
        float value = 0.0f;
        bool value_loaded = false;
        if (pages > 0 && value_subvecs > 0 && value_subdim > 0) {
          int64_t first_start = page_starts[0];
          int64_t page = (token - first_start) / page_size;
          if (token >= first_start && page >= 0 && page < pages) {
            int64_t row = token - page_starts[page];
            if (row >= 0 && row < page_size) {
              int64_t sub = d / value_subdim;
              int64_t sub_d = d - sub * value_subdim;
              if (sub >= 0 && sub < value_subvecs) {
                int64_t code = static_cast<int64_t>(
                    value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
                code = max(static_cast<int64_t>(0), min(code, value_centroids - 1));
                value = value_codebooks
                    [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
                     sub_d];
                value_loaded = true;
              }
            }
          }
        }
        if (!value_loaded) {
          value = load_as_float(values, (kv_head * total_tokens + token) * dim + d);
        }
        float weight = expf(shared_logits[sel] - max_logit) / denom;
        accum += weight * value;
      }
    }
    outputs[qh * dim + d] = accum;
  }
}

template <typename key_t, typename value_t, typename kcode_t, typename vcode_t>
__global__ void gqa_causal_fused_vpq_tail_attention_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    const value_t* __restrict__ values,
    const float* __restrict__ key_codebooks,
    const kcode_t* __restrict__ key_codes,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    float* __restrict__ outputs,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t pages,
    int64_t page_size,
    int64_t key_subvecs,
    int64_t key_centroids,
    int64_t key_subdim,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t exact_value_top,
    float scale,
    float tail_blend) {
  int64_t qh = static_cast<int64_t>(blockIdx.x);
  int64_t total_qh = positions * heads;
  if (qh >= total_qh) {
    return;
  }
  int64_t pos = qh / heads;
  int64_t head = qh - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t query_context_len = min(query_start + pos + 1, total_tokens);
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

  const float* q = queries + (pos * heads + head) * dim;
  const int64_t* ranked_row = ranked_tokens + qh * ranked;
  const float* ranked_score_row = ranked_scores + qh * ranked;
  const bool exact_values_by_selector_rank = exact_value_top < 0;
  const int64_t exact_value_limit = exact_values_by_selector_rank ? -exact_value_top : exact_value_top;
  const bool exact_all_ranked_values = exact_value_limit > 0 && exact_value_limit >= ranked;
  const bool mixed_ranked_values = exact_value_limit > 0 && exact_value_limit < ranked;
  const bool aggregate_selected_vpq = tail_blend >= 0.999999f;
  const int64_t max_selected =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + ranked + 4;
  const int64_t code_weight_count = pages * value_subvecs * value_centroids;

  extern __shared__ unsigned char shared_raw[];
  float* selected_logits = reinterpret_cast<float*>(shared_raw);
  float* code_weight_sums = selected_logits + max_selected;
  float* reduce_buf = code_weight_sums + code_weight_count;
  int32_t* selected_tokens = reinterpret_cast<int32_t*>(reduce_buf + blockDim.x);
  unsigned char* selected_exact = reinterpret_cast<unsigned char*>(selected_tokens + max_selected);

  __shared__ float max_logit;
  __shared__ float denom;
  __shared__ float selected_denom;

  __shared__ int32_t selected_count_shared;
  __shared__ int32_t selected_ranked_start_shared;
  if (threadIdx.x == 0) {
    int64_t count = 0;
    for (int64_t token = 0; token < prefix_end && count < max_selected; ++token) {
      selected_tokens[count] = static_cast<int32_t>(token);
      selected_logits[count] = -INFINITY;
      selected_exact[count] = 1;
      ++count;
    }
    for (int64_t token = base_tail_start; token < query_context_len && count < max_selected; ++token) {
      selected_tokens[count] = static_cast<int32_t>(token);
      selected_logits[count] = -INFINITY;
      selected_exact[count] = 1;
      ++count;
    }
    selected_ranked_start_shared = static_cast<int32_t>(count);
    for (int64_t sel = 0; sel < ranked && count < max_selected; ++sel) {
      int64_t token = ranked_row[sel];
      float selector_score = ranked_score_row[sel];
      bool valid = isfinite(selector_score) && token >= 0 && token < query_context_len &&
          !token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr);
      if (!valid) {
        continue;
      }
      selected_tokens[count] = static_cast<int32_t>(token);
      selected_logits[count] = -INFINITY;
      selected_exact[count] =
          (exact_all_ranked_values || (exact_values_by_selector_rank && sel < exact_value_limit)) ? 1 : 0;
      ++count;
    }
    selected_count_shared = static_cast<int32_t>(count);
  }
  __syncthreads();

  int64_t selected_count = static_cast<int64_t>(selected_count_shared);
  int64_t selected_ranked_start = static_cast<int64_t>(selected_ranked_start_shared);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  constexpr int warps_per_block = 4;
  for (int64_t base = 0; base < selected_count; base += warps_per_block) {
    int64_t idx = base + warp;
    if (warp < warps_per_block && idx < selected_count) {
      int64_t token = static_cast<int64_t>(selected_tokens[idx]);
      float partial = 0.0f;
      if (token >= 0 && token < total_tokens) {
        const key_t* k = keys + (kv_head * total_tokens + token) * dim;
        for (int64_t d = lane; d < dim; d += 32) {
          partial += q[d] * load_as_float(k, d);
        }
      }
      float dot = warp_reduce_sum(partial);
      if (lane == 0) {
        selected_logits[idx] = (token >= 0 && token < total_tokens) ? dot * scale : -INFINITY;
      }
    }
  }
  __syncthreads();

  float selected_max = -INFINITY;
  for (int64_t idx = threadIdx.x; idx < selected_count; idx += blockDim.x) {
    selected_max = fmaxf(selected_max, selected_logits[idx]);
  }
  reduce_buf[threadIdx.x] = selected_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] = fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  selected_max = reduce_buf[0];
  if (mixed_ranked_values && !exact_values_by_selector_rank && threadIdx.x == 0) {
    for (int64_t rank = 0; rank < exact_value_limit; ++rank) {
      int64_t best_idx = -1;
      float best_logit = -INFINITY;
      for (int64_t idx = 0; idx < selected_count; ++idx) {
        if (selected_exact[idx]) {
          continue;
        }
        float logit = selected_logits[idx];
        if (logit > best_logit || (logit == best_logit && (best_idx < 0 || idx < best_idx))) {
          best_logit = logit;
          best_idx = idx;
        }
      }
      if (best_idx >= 0 && isfinite(best_logit)) {
        selected_exact[best_idx] = 1;
      }
    }
  }
  __syncthreads();

  float tail_local_max = -INFINITY;
  if (tail_blend > 0.0f) {
    int64_t total_rows = pages * page_size;
    for (int64_t ordinal = threadIdx.x; ordinal < total_rows; ordinal += blockDim.x) {
      int64_t page = ordinal / page_size;
      int64_t row = ordinal - page * page_size;
      int64_t page_start = page_starts[page];
      bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
      if (!valid_page) {
        continue;
      }
      int64_t token = page_start + row;
      if (token < 0 || token >= query_context_len) {
        continue;
      }
      bool is_ranked = false;
      for (int64_t sel = 0; sel < ranked; ++sel) {
        if (isfinite(ranked_score_row[sel]) && ranked_row[sel] == token) {
          is_ranked = true;
          break;
        }
      }
      if (is_ranked) {
        continue;
      }
      float score = 0.0f;
      for (int64_t sub = 0; sub < key_subvecs; ++sub) {
        int64_t code = static_cast<int64_t>(
            key_codes[((kv_head * pages + page) * page_size + row) * key_subvecs + sub]);
        code = max(static_cast<int64_t>(0), min(code, key_centroids - 1));
        const float* cb =
            key_codebooks + ((((kv_head * pages + page) * key_subvecs + sub) * key_centroids + code) * key_subdim);
        const float* q_sub = q + sub * key_subdim;
        for (int64_t kd = 0; kd < key_subdim; ++kd) {
          score += q_sub[kd] * cb[kd];
        }
      }
      tail_local_max = fmaxf(tail_local_max, score * scale);
    }
  }
  reduce_buf[threadIdx.x] = tail_local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] = fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    max_logit = fmaxf(selected_max, reduce_buf[0]);
  }
  for (int64_t idx = threadIdx.x; idx < code_weight_count; idx += blockDim.x) {
    code_weight_sums[idx] = 0.0f;
  }
  __syncthreads();

  float selected_sum_local = 0.0f;
  if (isfinite(max_logit)) {
    for (int64_t idx = threadIdx.x; idx < selected_count; idx += blockDim.x) {
      float weight = expf(selected_logits[idx] - max_logit);
      selected_sum_local += weight;
      if (aggregate_selected_vpq && selected_exact[idx] == 0 && pages > 0 && value_subvecs > 0) {
        int64_t token = static_cast<int64_t>(selected_tokens[idx]);
        int64_t first_start = page_starts[0];
        int64_t page = (token - first_start) / page_size;
        bool aggregated = false;
        if (token >= first_start && page >= 0 && page < pages) {
          int64_t row = token - page_starts[page];
          if (row >= 0 && row < page_size) {
            aggregated = true;
            for (int64_t sub = 0; sub < value_subvecs; ++sub) {
              int64_t vcode = static_cast<int64_t>(
                  value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
              vcode = max(static_cast<int64_t>(0), min(vcode, value_centroids - 1));
              atomicAdd(
                  code_weight_sums + ((page * value_subvecs + sub) * value_centroids + vcode),
                  weight);
            }
          }
        }
        if (!aggregated) {
          selected_exact[idx] = 1;
        }
      }
    }
  }
  reduce_buf[threadIdx.x] = selected_sum_local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    selected_denom = fmaxf(reduce_buf[0], 1.0e-20f);
  }
  __syncthreads();

  float tail_sum_local = 0.0f;
  if (tail_blend > 0.0f && isfinite(max_logit)) {
    int64_t total_rows = pages * page_size;
    for (int64_t ordinal = threadIdx.x; ordinal < total_rows; ordinal += blockDim.x) {
      int64_t page = ordinal / page_size;
      int64_t row = ordinal - page * page_size;
      int64_t page_start = page_starts[page];
      bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
      if (!valid_page) {
        continue;
      }
      int64_t token = page_start + row;
      if (token < 0 || token >= query_context_len) {
        continue;
      }
      bool is_ranked = false;
      for (int64_t sel = 0; sel < ranked; ++sel) {
        if (isfinite(ranked_score_row[sel]) && ranked_row[sel] == token) {
          is_ranked = true;
          break;
        }
      }
      if (is_ranked) {
        continue;
      }
      float score = 0.0f;
      for (int64_t sub = 0; sub < key_subvecs; ++sub) {
        int64_t code = static_cast<int64_t>(
            key_codes[((kv_head * pages + page) * page_size + row) * key_subvecs + sub]);
        code = max(static_cast<int64_t>(0), min(code, key_centroids - 1));
        const float* cb =
            key_codebooks + ((((kv_head * pages + page) * key_subvecs + sub) * key_centroids + code) * key_subdim);
        const float* q_sub = q + sub * key_subdim;
        for (int64_t kd = 0; kd < key_subdim; ++kd) {
          score += q_sub[kd] * cb[kd];
        }
      }
      float weight = expf(score * scale - max_logit);
      tail_sum_local += weight;
      for (int64_t sub = 0; sub < value_subvecs; ++sub) {
        int64_t vcode = static_cast<int64_t>(
            value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
        vcode = max(static_cast<int64_t>(0), min(vcode, value_centroids - 1));
        atomicAdd(
            code_weight_sums + ((page * value_subvecs + sub) * value_centroids + vcode),
            weight);
      }
    }
  }
  reduce_buf[threadIdx.x] = tail_sum_local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    denom = fmaxf(selected_denom + reduce_buf[0], 1.0e-20f);
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float selected_accum = 0.0f;
    float tail_accum = 0.0f;
    if (isfinite(max_logit)) {
      for (int64_t idx = 0; idx < selected_count; ++idx) {
        bool exact_value = selected_exact[idx] != 0;
        if (aggregate_selected_vpq && !exact_value) {
          continue;
        }
        int64_t token = static_cast<int64_t>(selected_tokens[idx]);
        float weight = expf(selected_logits[idx] - max_logit);
        float value = 0.0f;
        bool value_loaded = false;
        if (!exact_value && pages > 0 && value_subvecs > 0 && value_subdim > 0) {
          int64_t first_start = page_starts[0];
          int64_t page = (token - first_start) / page_size;
          if (token >= first_start && page >= 0 && page < pages) {
            int64_t row = token - page_starts[page];
            if (row >= 0 && row < page_size) {
              int64_t sub = d / value_subdim;
              int64_t sub_d = d - sub * value_subdim;
              if (sub >= 0 && sub < value_subvecs) {
                int64_t vcode = static_cast<int64_t>(
                    value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
                vcode = max(static_cast<int64_t>(0), min(vcode, value_centroids - 1));
                value = value_codebooks
                    [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + vcode) * value_subdim) +
                     sub_d];
                value_loaded = true;
              }
            }
          }
        }
        if (exact_value || !value_loaded) {
          value = load_as_float(values, (kv_head * total_tokens + token) * dim + d);
        }
        selected_accum += weight * value;
      }
      if (tail_blend > 0.0f && value_subdim > 0) {
        int64_t sub = d / value_subdim;
        int64_t sub_d = d - sub * value_subdim;
        if (sub >= 0 && sub < value_subvecs) {
          for (int64_t page = 0; page < pages; ++page) {
            for (int64_t code = 0; code < value_centroids; ++code) {
              float weight = code_weight_sums[(page * value_subvecs + sub) * value_centroids + code];
              if (weight == 0.0f) {
                continue;
              }
              float value = value_codebooks
                  [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
                   sub_d];
              tail_accum += weight * value;
            }
          }
        }
      }
    }
    float full = (selected_accum + tail_accum) / denom;
    if (tail_blend > 0.0f && tail_blend < 1.0f) {
      float selected_only = selected_accum / selected_denom;
      outputs[qh * dim + d] = selected_only + tail_blend * (full - selected_only);
    } else {
      outputs[qh * dim + d] = full;
    }
  }
}

template <typename key_t, typename value_t, typename vcode_t>
__global__ void gqa_causal_fused_vpq_tail_from_scores_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    const value_t* __restrict__ values,
    const float* __restrict__ dense_pq_scores,
    const float* __restrict__ dense_selected_scores,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    float* __restrict__ outputs,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
	    int64_t ranked,
	    int64_t dim,
	    int64_t total_tokens,
	    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    const int64_t* __restrict__ exact_value_counts,
    int64_t exact_value_top,
    float exact_value_mass,
    float scale,
    float tail_blend) {
  int64_t qh = static_cast<int64_t>(blockIdx.x);
  int64_t total_qh = positions * heads;
  if (qh >= total_qh) {
    return;
  }
  int64_t pos = qh / heads;
  int64_t head = qh - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t query_context_len = min(query_start + pos + 1, total_tokens);
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

  const float* q = queries + (pos * heads + head) * dim;
  const float* score_row = dense_pq_scores + qh * (pages * page_size);
  const int64_t* ranked_row = ranked_tokens + qh * ranked;
  const float* ranked_score_row = ranked_scores + qh * ranked;
  const int64_t exact_value_top_row = exact_value_counts != nullptr ? exact_value_counts[qh] : exact_value_top;
  const bool exact_values_by_mass = exact_value_mass > 0.0f;
  const bool exact_values_by_selector_rank = exact_value_top_row < 0;
  const int64_t exact_value_limit = exact_values_by_selector_rank ? -exact_value_top_row : exact_value_top_row;
  const bool exact_all_ranked_values = !exact_values_by_mass && exact_value_limit > 0 && exact_value_limit >= ranked;
  const bool mixed_ranked_values = exact_values_by_mass || (exact_value_limit > 0 && exact_value_limit < ranked);
  const bool aggregate_selected_vpq = tail_blend >= 0.999999f;
  const int64_t max_selected =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + ranked + 4;
  const int64_t code_weight_count = pages * value_subvecs * value_centroids;
  const int64_t total_rows = pages * page_size;
  const int64_t ranked_bitmap_words = (total_rows + 31) / 32;
  int64_t visible_rows = total_rows;
  if (pages > 0 && page_size > 0) {
    int64_t visible_tokens = base_tail_start - page_starts[0];
    visible_rows =
        min(total_rows, (max(static_cast<int64_t>(0), visible_tokens) / page_size) * page_size);
  }
  const int64_t active_pages = page_size > 0 ? (visible_rows / page_size) : pages;
  const int64_t active_code_weight_count = active_pages * value_subvecs * value_centroids;

  extern __shared__ unsigned char shared_raw[];
  float* selected_logits = reinterpret_cast<float*>(shared_raw);
  float* code_weight_sums = selected_logits + max_selected;
  float* reduce_buf = code_weight_sums + code_weight_count;
  int32_t* selected_tokens = reinterpret_cast<int32_t*>(reduce_buf + blockDim.x);
  unsigned char* selected_exact = reinterpret_cast<unsigned char*>(selected_tokens + max_selected);
  uintptr_t bitmap_addr = reinterpret_cast<uintptr_t>(selected_exact + max_selected);
  bitmap_addr = (bitmap_addr + alignof(unsigned int) - 1) & ~(uintptr_t(alignof(unsigned int) - 1));
  unsigned int* ranked_bitmap = reinterpret_cast<unsigned int*>(bitmap_addr);

  for (int64_t word = threadIdx.x; word < ranked_bitmap_words; word += blockDim.x) {
    ranked_bitmap[word] = 0u;
  }
  __syncthreads();
  if (pages > 0 && page_size > 0) {
    int64_t first_page_start = page_starts[0];
    for (int64_t sel = threadIdx.x; sel < ranked; sel += blockDim.x) {
      int64_t token = ranked_row[sel];
      float selector_score = ranked_score_row[sel];
      if (!isfinite(selector_score) || token < 0 || token >= query_context_len) {
        continue;
      }
      int64_t rel = token - first_page_start;
      if (rel < 0) {
        continue;
      }
      int64_t page = rel / page_size;
      int64_t row = rel - page * page_size;
      if (page < 0 || page >= pages || row < 0 || row >= page_size) {
        continue;
      }
      if (page_starts[page] + row != token) {
        continue;
      }
      int64_t ordinal = page * page_size + row;
      atomicOr(ranked_bitmap + (ordinal >> 5), 1u << (ordinal & 31));
    }
  }
  __syncthreads();

  __shared__ float max_logit;
  __shared__ float denom;
  __shared__ float selected_denom;

  __shared__ int32_t selected_count_shared;
  __shared__ int32_t selected_ranked_start_shared;
  if (threadIdx.x == 0) {
    int64_t count = 0;
    for (int64_t token = 0; token < prefix_end && count < max_selected; ++token) {
      selected_tokens[count] = static_cast<int32_t>(token);
      selected_logits[count] = -INFINITY;
      selected_exact[count] = 1;
      ++count;
    }
    for (int64_t token = base_tail_start; token < query_context_len && count < max_selected; ++token) {
      selected_tokens[count] = static_cast<int32_t>(token);
      selected_logits[count] = -INFINITY;
      selected_exact[count] = 1;
      ++count;
    }
    selected_ranked_start_shared = static_cast<int32_t>(count);
    for (int64_t sel = 0; sel < ranked && count < max_selected; ++sel) {
      int64_t token = ranked_row[sel];
      float selector_score = ranked_score_row[sel];
      bool valid = isfinite(selector_score) && token >= 0 && token < query_context_len &&
          !token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr);
      if (!valid) {
        continue;
      }
      selected_tokens[count] = static_cast<int32_t>(token);
      selected_logits[count] = -INFINITY;
      selected_exact[count] =
          (exact_all_ranked_values || (exact_values_by_selector_rank && sel < exact_value_limit)) ? 1 : 0;
      ++count;
    }
    selected_count_shared = static_cast<int32_t>(count);
  }
  __syncthreads();

  int64_t selected_count = static_cast<int64_t>(selected_count_shared);
  int64_t selected_ranked_start = static_cast<int64_t>(selected_ranked_start_shared);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  constexpr int warps_per_block = 4;
  for (int64_t base = 0; base < selected_count; base += warps_per_block) {
    int64_t idx = base + warp;
    if (warp < warps_per_block && idx < selected_count) {
      int64_t token = static_cast<int64_t>(selected_tokens[idx]);
      float selected_logit = -INFINITY;
      bool loaded_from_dense = false;
      if (dense_selected_scores != nullptr && pages > 0 && page_size > 0 && token >= 0 && token < total_tokens) {
        int64_t first_start = page_starts[0];
        int64_t rel = token - first_start;
        if (rel >= 0) {
          int64_t page = rel / page_size;
          int64_t row = rel - page * page_size;
          if (page >= 0 && page < pages && row >= 0 && row < page_size && page_starts[page] + row == token) {
            float score = dense_selected_scores[qh * total_rows + page * page_size + row];
            if (isfinite(score)) {
              selected_logit = score * scale;
              loaded_from_dense = true;
            }
          }
        }
      }
      float partial = 0.0f;
      if (!loaded_from_dense && token >= 0 && token < total_tokens) {
        const key_t* k = keys + (kv_head * total_tokens + token) * dim;
        for (int64_t d = lane; d < dim; d += 32) {
          partial += q[d] * load_as_float(k, d);
        }
      }
      float dot = warp_reduce_sum(partial);
      if (lane == 0) {
        selected_logits[idx] =
            loaded_from_dense ? selected_logit : ((token >= 0 && token < total_tokens) ? dot * scale : -INFINITY);
      }
    }
  }
  __syncthreads();

  float selected_max = -INFINITY;
  for (int64_t idx = threadIdx.x; idx < selected_count; idx += blockDim.x) {
    selected_max = fmaxf(selected_max, selected_logits[idx]);
  }
  reduce_buf[threadIdx.x] = selected_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] = fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  selected_max = reduce_buf[0];
  if (exact_values_by_mass && threadIdx.x == 0 && isfinite(selected_max)) {
    const float target = fminf(fmaxf(exact_value_mass, 0.0f), 1.0f);
    float total = 0.0f;
    float exact_sum = 0.0f;
    int64_t exact_count = 0;
    for (int64_t idx = 0; idx < selected_count; ++idx) {
      float logit = selected_logits[idx];
      if (!isfinite(logit)) {
        continue;
      }
      float weight = expf(logit - selected_max);
      total += weight;
      if (selected_exact[idx]) {
        exact_sum += weight;
        if (idx >= selected_ranked_start) {
          ++exact_count;
        }
      }
    }
    while (total > 1.0e-20f && exact_sum / total < target) {
      int64_t best_idx = -1;
      float best_logit = -INFINITY;
      for (int64_t idx = 0; idx < selected_count; ++idx) {
        if (selected_exact[idx]) {
          continue;
        }
        float logit = selected_logits[idx];
        if (logit > best_logit || (logit == best_logit && (best_idx < 0 || idx < best_idx))) {
          best_logit = logit;
          best_idx = idx;
        }
      }
      if (best_idx < 0 || !isfinite(best_logit)) {
        break;
      }
      selected_exact[best_idx] = 1;
      ++exact_count;
      exact_sum += expf(best_logit - selected_max);
    }
    while (exact_value_limit > 0 && exact_count < exact_value_limit) {
      int64_t best_idx = -1;
      float best_logit = -INFINITY;
      for (int64_t idx = 0; idx < selected_count; ++idx) {
        if (selected_exact[idx]) {
          continue;
        }
        float logit = selected_logits[idx];
        if (logit > best_logit || (logit == best_logit && (best_idx < 0 || idx < best_idx))) {
          best_logit = logit;
          best_idx = idx;
        }
      }
      if (best_idx < 0 || !isfinite(best_logit)) {
        break;
      }
      selected_exact[best_idx] = 1;
      ++exact_count;
    }
  } else if (mixed_ranked_values && !exact_values_by_selector_rank && threadIdx.x == 0) {
    for (int64_t rank = 0; rank < exact_value_limit; ++rank) {
      int64_t best_idx = -1;
      float best_logit = -INFINITY;
      for (int64_t idx = 0; idx < selected_count; ++idx) {
        if (selected_exact[idx]) {
          continue;
        }
        float logit = selected_logits[idx];
        if (logit > best_logit || (logit == best_logit && (best_idx < 0 || idx < best_idx))) {
          best_logit = logit;
          best_idx = idx;
        }
      }
      if (best_idx >= 0 && isfinite(best_logit)) {
        selected_exact[best_idx] = 1;
      }
    }
  }
  __syncthreads();

  float tail_local_max = -INFINITY;
  if (tail_blend > 0.0f) {
    for (int64_t ordinal = threadIdx.x; ordinal < visible_rows; ordinal += blockDim.x) {
      int64_t page = ordinal / page_size;
      int64_t row = ordinal - page * page_size;
      int64_t page_start = page_starts[page];
      bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
      if (!valid_page) {
        continue;
      }
      int64_t token = page_start + row;
      if (token < 0 || token >= query_context_len) {
        continue;
      }
      bool is_ranked = (ranked_bitmap[ordinal >> 5] & (1u << (ordinal & 31))) != 0u;
      if (is_ranked) {
        continue;
      }
      float score = score_row[ordinal];
      if (isfinite(score)) {
        tail_local_max = fmaxf(tail_local_max, score * scale);
      }
    }
  }
  reduce_buf[threadIdx.x] = tail_local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] = fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    max_logit = fmaxf(selected_max, reduce_buf[0]);
  }
  for (int64_t idx = threadIdx.x; idx < active_code_weight_count; idx += blockDim.x) {
    code_weight_sums[idx] = 0.0f;
  }
  __syncthreads();

  float selected_sum_local = 0.0f;
  if (isfinite(max_logit)) {
    for (int64_t idx = threadIdx.x; idx < selected_count; idx += blockDim.x) {
      float weight = expf(selected_logits[idx] - max_logit);
      selected_sum_local += weight;
      if (aggregate_selected_vpq && selected_exact[idx] == 0 && pages > 0 && value_subvecs > 0) {
        int64_t token = static_cast<int64_t>(selected_tokens[idx]);
        int64_t first_start = page_starts[0];
        int64_t page = (token - first_start) / page_size;
        bool aggregated = false;
        if (token >= first_start && page >= 0 && page < active_pages) {
          int64_t row = token - page_starts[page];
          if (row >= 0 && row < page_size) {
            aggregated = true;
            for (int64_t sub = 0; sub < value_subvecs; ++sub) {
              int64_t vcode = static_cast<int64_t>(
                  value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
              vcode = max(static_cast<int64_t>(0), min(vcode, value_centroids - 1));
              atomicAdd(
                  code_weight_sums + ((page * value_subvecs + sub) * value_centroids + vcode),
                  weight);
            }
          }
        }
        if (!aggregated) {
          selected_exact[idx] = 1;
        }
      }
    }
  }
  reduce_buf[threadIdx.x] = selected_sum_local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    selected_denom = fmaxf(reduce_buf[0], 1.0e-20f);
  }
  __syncthreads();

  float tail_sum_local = 0.0f;
  if (tail_blend > 0.0f && isfinite(max_logit)) {
    for (int64_t ordinal = threadIdx.x; ordinal < visible_rows; ordinal += blockDim.x) {
      int64_t page = ordinal / page_size;
      int64_t row = ordinal - page * page_size;
      int64_t page_start = page_starts[page];
      bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
      if (!valid_page) {
        continue;
      }
      int64_t token = page_start + row;
      if (token < 0 || token >= query_context_len) {
        continue;
      }
      bool is_ranked = (ranked_bitmap[ordinal >> 5] & (1u << (ordinal & 31))) != 0u;
      if (is_ranked) {
        continue;
      }
      float score = score_row[ordinal];
      if (!isfinite(score)) {
        continue;
      }
      float weight = expf(score * scale - max_logit);
      tail_sum_local += weight;
      for (int64_t sub = 0; sub < value_subvecs; ++sub) {
        int64_t vcode = static_cast<int64_t>(
            value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
        vcode = max(static_cast<int64_t>(0), min(vcode, value_centroids - 1));
        atomicAdd(
            code_weight_sums + ((page * value_subvecs + sub) * value_centroids + vcode),
            weight);
      }
    }
  }
  reduce_buf[threadIdx.x] = tail_sum_local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    denom = fmaxf(selected_denom + reduce_buf[0], 1.0e-20f);
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float selected_accum = 0.0f;
    float tail_accum = 0.0f;
    if (isfinite(max_logit)) {
      for (int64_t idx = 0; idx < selected_count; ++idx) {
        bool exact_value = selected_exact[idx] != 0;
        if (aggregate_selected_vpq && !exact_value) {
          continue;
        }
        int64_t token = static_cast<int64_t>(selected_tokens[idx]);
        float weight = expf(selected_logits[idx] - max_logit);
        float value = 0.0f;
        bool value_loaded = false;
        if (!exact_value && pages > 0 && value_subvecs > 0 && value_subdim > 0) {
          int64_t first_start = page_starts[0];
          int64_t page = (token - first_start) / page_size;
          if (token >= first_start && page >= 0 && page < pages) {
            int64_t row = token - page_starts[page];
            if (row >= 0 && row < page_size) {
              int64_t sub = d / value_subdim;
              int64_t sub_d = d - sub * value_subdim;
              if (sub >= 0 && sub < value_subvecs) {
                int64_t vcode = static_cast<int64_t>(
                    value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
                vcode = max(static_cast<int64_t>(0), min(vcode, value_centroids - 1));
                value = value_codebooks
                    [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + vcode) * value_subdim) +
                     sub_d];
                value_loaded = true;
              }
            }
          }
        }
        if (exact_value || !value_loaded) {
          value = load_as_float(values, (kv_head * total_tokens + token) * dim + d);
        }
        selected_accum += weight * value;
      }
      if (tail_blend > 0.0f && value_subdim > 0) {
        int64_t sub = d / value_subdim;
        int64_t sub_d = d - sub * value_subdim;
        if (sub >= 0 && sub < value_subvecs) {
          for (int64_t page = 0; page < active_pages; ++page) {
            for (int64_t code = 0; code < value_centroids; ++code) {
              float weight = code_weight_sums[(page * value_subvecs + sub) * value_centroids + code];
              if (weight == 0.0f) {
                continue;
              }
              float value = value_codebooks
                  [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
                   sub_d];
              tail_accum += weight * value;
            }
          }
        }
      }
    }
    float full = (selected_accum + tail_accum) / denom;
    if (tail_blend > 0.0f && tail_blend < 1.0f) {
      float selected_only = selected_accum / selected_denom;
      outputs[qh * dim + d] = selected_only + tail_blend * (full - selected_only);
    } else {
      outputs[qh * dim + d] = full;
    }
  }
}

template <typename vcode_t>
__global__ void gqa_decode_vpq_tail_from_scores_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    const float* __restrict__ values,
    const float* __restrict__ dense_pq_scores,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    float* __restrict__ outputs,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    const int64_t* __restrict__ exact_value_counts,
    int64_t exact_value_top,
    float scale,
    float tail_blend) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  const int64_t exact_value_top_row = exact_value_counts != nullptr ? exact_value_counts[head] : exact_value_top;
  const bool exact_values_by_selector_rank = exact_value_top_row < 0;
  const int64_t exact_value_limit = exact_values_by_selector_rank ? -exact_value_top_row : exact_value_top_row;
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

  extern __shared__ unsigned char shared_raw[];
  float* selected_logits = reinterpret_cast<float*>(shared_raw);
  const int64_t max_selected =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + ranked + 4;
  const bool exact_all_ranked_values = exact_value_limit > 0 && exact_value_limit >= ranked;
  const bool mixed_ranked_values = exact_value_limit > 0 && exact_value_limit < ranked;
  float* ranked_logits = selected_logits + max_selected;
  unsigned char* ranked_exact = reinterpret_cast<unsigned char*>(ranked_logits + ranked);
  const float* q = queries + head * dim;
  const int64_t* ranked_row = ranked_tokens + head * ranked;
  const float* ranked_score_row = ranked_scores + head * ranked;
  const float* score_row = dense_pq_scores + head * (pages * page_size);

  __shared__ int64_t selected_count;
  __shared__ float max_logit;
  __shared__ float denom;

  if (threadIdx.x == 0) {
    int64_t count = 0;
    float local_max = -INFINITY;
    if (mixed_ranked_values) {
      for (int64_t sel = 0; sel < ranked; ++sel) {
        ranked_logits[sel] = -INFINITY;
        ranked_exact[sel] = 0;
      }
    }
    for (int64_t token = 0; token < prefix_end; ++token) {
      const float* k = keys + (kv_head * total_tokens + token) * dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        dot += q[d] * k[d];
      }
      float logit = dot * scale;
      selected_logits[count++] = logit;
      local_max = fmaxf(local_max, logit);
    }
    for (int64_t token = base_tail_start; token < query_context_len; ++token) {
      const float* k = keys + (kv_head * total_tokens + token) * dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        dot += q[d] * k[d];
      }
      float logit = dot * scale;
      selected_logits[count++] = logit;
      local_max = fmaxf(local_max, logit);
    }
    for (int64_t sel = 0; sel < ranked; ++sel) {
      int64_t token = ranked_row[sel];
      float selector_score = ranked_score_row[sel];
      if (!isfinite(selector_score) || token < 0 || token >= query_context_len) {
        continue;
      }
      if (token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr)) {
        continue;
      }
      const float* k = keys + (kv_head * total_tokens + token) * dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        dot += q[d] * k[d];
      }
      float logit = dot * scale;
      selected_logits[count++] = logit;
      if (mixed_ranked_values) {
        ranked_logits[sel] = logit;
        if (exact_values_by_selector_rank && sel < exact_value_limit) {
          ranked_exact[sel] = 1;
        }
      }
      local_max = fmaxf(local_max, logit);
    }
    if (mixed_ranked_values && !exact_values_by_selector_rank) {
      for (int64_t rank = 0; rank < exact_value_limit; ++rank) {
        int64_t best_sel = -1;
        float best_logit = -INFINITY;
        for (int64_t sel = 0; sel < ranked; ++sel) {
          if (ranked_exact[sel]) {
            continue;
          }
          float logit = ranked_logits[sel];
          if (logit > best_logit || (logit == best_logit && (best_sel < 0 || sel < best_sel))) {
            best_logit = logit;
            best_sel = sel;
          }
        }
        if (best_sel >= 0 && isfinite(best_logit)) {
          ranked_exact[best_sel] = 1;
        }
      }
    }
    if (tail_blend > 0.0f) {
      for (int64_t page = 0; page < pages; ++page) {
        int64_t page_start = page_starts[page];
        bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
        if (!valid_page) {
          continue;
        }
        for (int64_t row = 0; row < page_size; ++row) {
          int64_t token = page_start + row;
          if (token < 0 || token >= query_context_len) {
            continue;
          }
          bool is_ranked = false;
          for (int64_t sel = 0; sel < ranked; ++sel) {
            if (isfinite(ranked_score_row[sel]) && ranked_row[sel] == token) {
              is_ranked = true;
              break;
            }
          }
          if (!is_ranked) {
            local_max = fmaxf(local_max, score_row[page * page_size + row] * scale);
          }
        }
      }
    }

    float local_denom = 0.0f;
    if (isfinite(local_max)) {
      for (int64_t idx = 0; idx < count; ++idx) {
        local_denom += expf(selected_logits[idx] - local_max);
      }
      if (tail_blend > 0.0f) {
        for (int64_t page = 0; page < pages; ++page) {
          int64_t page_start = page_starts[page];
          bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
          if (!valid_page) {
            continue;
          }
          for (int64_t row = 0; row < page_size; ++row) {
            int64_t token = page_start + row;
            if (token < 0 || token >= query_context_len) {
              continue;
            }
            bool is_ranked = false;
            for (int64_t sel = 0; sel < ranked; ++sel) {
              if (isfinite(ranked_score_row[sel]) && ranked_row[sel] == token) {
                is_ranked = true;
                break;
              }
            }
            if (!is_ranked) {
              local_denom += expf(score_row[page * page_size + row] * scale - local_max);
            }
          }
        }
      }
    }
    selected_count = count;
    max_logit = local_max;
    denom = fmaxf(local_denom, 1.0e-20f);
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float accum = 0.0f;
    int64_t idx = 0;
    if (isfinite(max_logit)) {
      for (int64_t token = 0; token < prefix_end; ++token) {
        float weight = expf(selected_logits[idx++] - max_logit) / denom;
        accum += weight * values[(kv_head * total_tokens + token) * dim + d];
      }
      for (int64_t token = base_tail_start; token < query_context_len; ++token) {
        float weight = expf(selected_logits[idx++] - max_logit) / denom;
        accum += weight * values[(kv_head * total_tokens + token) * dim + d];
      }
      for (int64_t sel = 0; sel < ranked; ++sel) {
        int64_t token = ranked_row[sel];
        float selector_score = ranked_score_row[sel];
        if (!isfinite(selector_score) || token < 0 || token >= query_context_len) {
          continue;
        }
        if (token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr)) {
          continue;
        }
        float weight = expf(selected_logits[idx++] - max_logit) / denom;
        bool exact_value = exact_all_ranked_values || (mixed_ranked_values && ranked_exact[sel] != 0);
        float value = 0.0f;
        bool value_loaded = false;
        if (!exact_value && pages > 0 && value_subvecs > 0 && value_subdim > 0) {
          int64_t first_start = page_starts[0];
          int64_t page = (token - first_start) / page_size;
          if (token >= first_start && page >= 0 && page < pages) {
            int64_t row = token - page_starts[page];
            if (row >= 0 && row < page_size) {
              int64_t sub = d / value_subdim;
              int64_t sub_d = d - sub * value_subdim;
              if (sub >= 0 && sub < value_subvecs) {
                int64_t vcode = static_cast<int64_t>(
                    value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
                vcode = max(static_cast<int64_t>(0), min(vcode, value_centroids - 1));
                value = value_codebooks
                    [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + vcode) * value_subdim) +
                     sub_d];
                value_loaded = true;
              }
            }
          }
        }
        if (exact_value || !value_loaded) {
          value = values[(kv_head * total_tokens + token) * dim + d];
        }
        accum += weight * value;
      }
      if (tail_blend > 0.0f) {
        for (int64_t page = 0; page < pages; ++page) {
          int64_t page_start = page_starts[page];
          bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
          if (!valid_page) {
            continue;
          }
          for (int64_t row = 0; row < page_size; ++row) {
            int64_t token = page_start + row;
            if (token < 0 || token >= query_context_len) {
              continue;
            }
            bool is_ranked = false;
            for (int64_t sel = 0; sel < ranked; ++sel) {
              if (isfinite(ranked_score_row[sel]) && ranked_row[sel] == token) {
                is_ranked = true;
                break;
              }
            }
            if (is_ranked) {
              continue;
            }
            int64_t sub = d / value_subdim;
            int64_t sub_d = d - sub * value_subdim;
            float value = 0.0f;
            if (sub >= 0 && sub < value_subvecs) {
              int64_t vcode = static_cast<int64_t>(
                  value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
              vcode = max(static_cast<int64_t>(0), min(vcode, value_centroids - 1));
              value = value_codebooks
                  [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + vcode) * value_subdim) +
                   sub_d];
            }
            float weight = expf(score_row[page * page_size + row] * scale - max_logit) / denom;
            accum += weight * value;
          }
        }
      }
    }
    outputs[head * dim + d] = accum;
  }
}

template <typename key_t>
__global__ void gqa_decode_base_logits_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    int64_t* __restrict__ base_tokens,
    float* __restrict__ base_logits,
    int64_t* __restrict__ base_counts,
    int64_t heads,
    int64_t kv_heads,
    int64_t dim,
    int64_t total_tokens,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t key_stride_dim,
    int64_t page_size,
    int64_t max_base,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    float scale) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  int warps = blockDim.x >> 5;
  if (warps <= 0) {
    warps = 1;
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
  const float* q = queries + head * dim;
  int64_t* base_tokens_row = base_tokens + head * max_base;
  float* base_logits_row = base_logits + head * max_base;

  __shared__ int64_t base_count_shared;
  if (threadIdx.x == 0) {
    int64_t base_count = 0;
    for (int64_t token = 0; token < prefix_end && base_count < max_base; ++token) {
      base_tokens_row[base_count] = token;
      ++base_count;
    }
    for (int64_t token = base_tail_start; token < query_context_len && base_count < max_base; ++token) {
      base_tokens_row[base_count] = token;
      ++base_count;
    }
    base_counts[head] = base_count;
    base_count_shared = base_count;
  }
  __syncthreads();

  int64_t base_count = base_count_shared;
  for (int64_t idx = warp_id; idx < base_count; idx += warps) {
    int64_t token = base_tokens_row[idx];
    float partial = 0.0f;
    for (int64_t d = lane; d < dim; d += 32) {
      partial += q[d] *
          load_strided3_as_float(keys, kv_head, token, d, key_stride_head, key_stride_token, key_stride_dim);
    }
    float dot = warp_reduce_sum(partial);
    if (lane == 0) {
      base_logits_row[idx] = dot * scale;
    }
  }
}

template <typename key_t>
__global__ void gqa_decode_base_ranked_logits_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    int64_t* __restrict__ base_tokens,
    float* __restrict__ base_logits,
    int64_t* __restrict__ base_counts,
    float* __restrict__ ranked_logits,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t key_stride_dim,
    int64_t page_size,
    int64_t max_base,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    float scale) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  int warps = blockDim.x >> 5;
  if (warps <= 0) {
    warps = 1;
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
  const float* q = queries + head * dim;
  int64_t* base_tokens_row = base_tokens + head * max_base;
  float* base_logits_row = base_logits + head * max_base;
  const int64_t* ranked_row = ranked_tokens + head * ranked;
  const float* ranked_score_row = ranked_scores + head * ranked;
  float* ranked_logit_row = ranked_logits + head * ranked;

  __shared__ int64_t base_count_shared;
  if (threadIdx.x == 0) {
    int64_t base_count = 0;
    for (int64_t token = 0; token < prefix_end && base_count < max_base; ++token) {
      base_tokens_row[base_count] = token;
      ++base_count;
    }
    for (int64_t token = base_tail_start; token < query_context_len && base_count < max_base; ++token) {
      base_tokens_row[base_count] = token;
      ++base_count;
    }
    base_counts[head] = base_count;
    base_count_shared = base_count;
  }
  for (int64_t sel = threadIdx.x; sel < ranked; sel += blockDim.x) {
    ranked_logit_row[sel] = -INFINITY;
  }
  __syncthreads();

  int64_t base_count = base_count_shared;
  for (int64_t idx = warp_id; idx < base_count; idx += warps) {
    int64_t token = base_tokens_row[idx];
    float partial = 0.0f;
    for (int64_t d = lane; d < dim; d += 32) {
      partial += q[d] *
          load_strided3_as_float(keys, kv_head, token, d, key_stride_head, key_stride_token, key_stride_dim);
    }
    float dot = warp_reduce_sum(partial);
    if (lane == 0) {
      base_logits_row[idx] = dot * scale;
    }
  }
  for (int64_t sel = warp_id; sel < ranked; sel += warps) {
    int64_t token = ranked_row[sel];
    float selector_score = ranked_score_row[sel];
    if (!isfinite(selector_score) || token < 0 || token >= query_context_len ||
        token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr)) {
      continue;
    }
    float partial = 0.0f;
    for (int64_t d = lane; d < dim; d += 32) {
      partial += q[d] *
          load_strided3_as_float(keys, kv_head, token, d, key_stride_head, key_stride_token, key_stride_dim);
    }
    float dot = warp_reduce_sum(partial);
    if (lane == 0) {
      ranked_logit_row[sel] = dot * scale;
    }
  }
}

__global__ void gqa_decode_filter_ranked_logits_input_kernel(
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    const float* __restrict__ ranked_logits_input,
    float* __restrict__ ranked_logits,
    int64_t heads,
    int64_t ranked,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * ranked;
  if (linear >= total) {
    return;
  }
  int64_t head = linear / ranked;
  int64_t sel = linear - head * ranked;
  int64_t token = ranked_tokens[linear];
  float selector_score = ranked_scores[linear];
  float logit = ranked_logits_input[linear];
  if (!isfinite(selector_score) || !isfinite(logit) || token < 0 || token >= query_context_len ||
      token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr)) {
    ranked_logits[linear] = -INFINITY;
    return;
  }
  ranked_logits[linear] = logit;
}

template <typename key_t>
__global__ void gqa_decode_ranked_logits_mask_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    unsigned char* __restrict__ tail_mask,
    int64_t* __restrict__ base_tokens,
    float* __restrict__ base_logits,
    int64_t* __restrict__ base_counts,
    float* __restrict__ ranked_logits,
    unsigned char* __restrict__ ranked_exact,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t key_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t max_base,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    const int64_t* __restrict__ exact_value_counts,
    const float* __restrict__ exact_value_thresholds,
    const int64_t* __restrict__ exact_value_threshold_sels,
    int64_t exact_value_top,
    float exact_value_mass,
    float scale) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  int warps = blockDim.x >> 5;
  if (warps <= 0) {
    warps = 1;
  }
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  const int64_t exact_value_top_row = exact_value_counts != nullptr ? exact_value_counts[head] : exact_value_top;
  const bool exact_values_by_mass = exact_value_mass > 0.0f;
  const bool exact_values_by_threshold =
      exact_values_by_mass && exact_value_thresholds != nullptr && exact_value_threshold_sels != nullptr;
  const bool exact_values_by_selector_rank = exact_value_top_row < 0;
  const int64_t exact_value_limit = exact_values_by_selector_rank ? -exact_value_top_row : exact_value_top_row;
  const float exact_value_threshold =
      exact_values_by_threshold ? exact_value_thresholds[head] : INFINITY;
  const int64_t exact_value_threshold_sel =
      exact_values_by_threshold ? exact_value_threshold_sels[head] : -1;
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
  const float* q = queries + head * dim;
  int64_t* base_tokens_row = base_tokens + head * max_base;
  float* base_logits_row = base_logits + head * max_base;
  const int64_t* ranked_row = ranked_tokens + head * ranked;
  const float* ranked_score_row = ranked_scores + head * ranked;
  float* ranked_logits_row = ranked_logits + head * ranked;
  unsigned char* ranked_exact_row = ranked_exact + head * ranked;
  unsigned char* tail_mask_row = tail_mask + head * (pages * page_size);

  __shared__ int64_t base_count_shared;
  if (threadIdx.x == 0) {
    int64_t base_count = 0;
    for (int64_t token = 0; token < prefix_end && base_count < max_base; ++token) {
      base_tokens_row[base_count] = token;
      ++base_count;
    }
    for (int64_t token = base_tail_start; token < query_context_len && base_count < max_base; ++token) {
      base_tokens_row[base_count] = token;
      ++base_count;
    }
    base_counts[head] = base_count;
    base_count_shared = base_count;
  }
  for (int64_t sel = threadIdx.x; sel < ranked; sel += blockDim.x) {
    ranked_logits_row[sel] = -INFINITY;
    ranked_exact_row[sel] = 0;
  }
  __syncthreads();

  int64_t base_count = base_count_shared;
	  for (int64_t idx = warp_id; idx < base_count; idx += warps) {
	    int64_t token = base_tokens_row[idx];
	    float partial = 0.0f;
	    for (int64_t d = lane; d < dim; d += 32) {
	      partial += q[d] *
	          load_strided3_as_float(keys, kv_head, token, d, key_stride_head, key_stride_token, key_stride_dim);
	    }
    float dot = warp_reduce_sum(partial);
    if (lane == 0) {
      base_logits_row[idx] = dot * scale;
    }
  }

  for (int64_t sel = warp_id; sel < ranked; sel += warps) {
    int64_t token = ranked_row[sel];
    float selector_score = ranked_score_row[sel];
    if (!isfinite(selector_score) || token < 0 || token >= query_context_len) {
      continue;
    }
    if (token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr)) {
      continue;
    }
	    float partial = 0.0f;
	    for (int64_t d = lane; d < dim; d += 32) {
	      partial += q[d] *
	          load_strided3_as_float(keys, kv_head, token, d, key_stride_head, key_stride_token, key_stride_dim);
	    }
    float dot = warp_reduce_sum(partial);
    if (lane == 0) {
      ranked_logits_row[sel] = dot * scale;
      if (exact_values_by_selector_rank && sel < exact_value_limit) {
        ranked_exact_row[sel] = 1;
      }
      if (pages > 0) {
        int64_t first_start = page_starts[0];
        int64_t page = (token - first_start) / page_size;
        if (token >= first_start && page >= 0 && page < pages) {
          int64_t row = token - page_starts[page];
          bool valid_page = page_starts[page] >= prefix_end && page_starts[page] + page_size <= base_tail_start;
          if (valid_page && row >= 0 && row < page_size) {
            tail_mask_row[page * page_size + row] = 1;
          }
        }
      }
    }
  }
  __syncthreads();
  if (exact_values_by_mass) {
    if (threadIdx.x != 0) {
      return;
    }
    const float target = fminf(fmaxf(exact_value_mass, 0.0f), 1.0f);
    float max_logit = -INFINITY;
    for (int64_t idx = 0; idx < base_count; ++idx) {
      max_logit = fmaxf(max_logit, base_logits_row[idx]);
    }
    for (int64_t sel = 0; sel < ranked; ++sel) {
      max_logit = fmaxf(max_logit, ranked_logits_row[sel]);
    }
    if (!isfinite(max_logit)) {
      return;
    }
    float total = 0.0f;
    float exact_sum = 0.0f;
    int64_t exact_count = 0;
    for (int64_t idx = 0; idx < base_count; ++idx) {
      float logit = base_logits_row[idx];
      if (isfinite(logit)) {
        float weight = expf(logit - max_logit);
        total += weight;
        exact_sum += weight;
      }
    }
    for (int64_t sel = 0; sel < ranked; ++sel) {
      float logit = ranked_logits_row[sel];
      if (isfinite(logit)) {
        total += expf(logit - max_logit);
      }
    }
    while (total > 1.0e-20f && exact_sum / total < target) {
      int64_t best_sel = -1;
      float best_logit = -INFINITY;
      for (int64_t sel = 0; sel < ranked; ++sel) {
        if (ranked_exact_row[sel]) {
          continue;
        }
        float logit = ranked_logits_row[sel];
        if (logit > best_logit || (logit == best_logit && (best_sel < 0 || sel < best_sel))) {
          best_logit = logit;
          best_sel = sel;
        }
      }
      if (best_sel < 0 || !isfinite(best_logit)) {
        break;
      }
      ranked_exact_row[best_sel] = 1;
      ++exact_count;
      exact_sum += expf(best_logit - max_logit);
    }
    while (exact_value_limit > 0 && exact_count < exact_value_limit) {
      int64_t best_sel = -1;
      float best_logit = -INFINITY;
      for (int64_t sel = 0; sel < ranked; ++sel) {
        if (ranked_exact_row[sel]) {
          continue;
        }
        float logit = ranked_logits_row[sel];
        if (logit > best_logit || (logit == best_logit && (best_sel < 0 || sel < best_sel))) {
          best_logit = logit;
          best_sel = sel;
        }
      }
      if (best_sel < 0 || !isfinite(best_logit)) {
        break;
      }
      ranked_exact_row[best_sel] = 1;
      ++exact_count;
    }
    return;
  }
  if (exact_value_limit <= 0) {
    return;
  }
  if (exact_values_by_selector_rank) {
    return;
  }
  if (threadIdx.x != 0) {
    return;
  }
  if (exact_value_limit >= ranked) {
    for (int64_t sel = 0; sel < ranked; ++sel) {
      if (isfinite(ranked_logits_row[sel])) {
        ranked_exact_row[sel] = 1;
      }
    }
    return;
  }
  for (int64_t rank = 0; rank < exact_value_limit; ++rank) {
    int64_t best_sel = -1;
    float best_logit = -INFINITY;
    for (int64_t sel = 0; sel < ranked; ++sel) {
      if (ranked_exact_row[sel]) {
        continue;
      }
      float logit = ranked_logits_row[sel];
      if (logit > best_logit || (logit == best_logit && (best_sel < 0 || sel < best_sel))) {
        best_logit = logit;
        best_sel = sel;
      }
    }
    if (best_sel >= 0 && isfinite(best_logit)) {
      ranked_exact_row[best_sel] = 1;
    }
  }
}

__device__ __forceinline__ bool selected_mass_exact_value_for_rank(
    float ranked_logit,
    int64_t sel,
    float threshold_logit,
    int64_t threshold_sel);

template <typename key_t>
__global__ void gqa_decode_ranked_logits_mask_from_logits_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    const float* __restrict__ ranked_logits_input,
    unsigned char* __restrict__ tail_mask,
    int64_t* __restrict__ base_tokens,
    float* __restrict__ base_logits,
    int64_t* __restrict__ base_counts,
    float* __restrict__ ranked_logits,
    unsigned char* __restrict__ ranked_exact,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t key_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t max_base,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    const int64_t* __restrict__ exact_value_counts,
    const float* __restrict__ exact_value_thresholds,
    const int64_t* __restrict__ exact_value_threshold_sels,
    int64_t exact_value_top,
    float exact_value_mass,
    float scale) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  int warps = blockDim.x >> 5;
  if (warps <= 0) {
    warps = 1;
  }
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  const int64_t exact_value_top_row = exact_value_counts != nullptr ? exact_value_counts[head] : exact_value_top;
  const bool exact_values_by_mass = exact_value_mass > 0.0f;
  const bool exact_values_by_threshold =
      exact_values_by_mass && exact_value_thresholds != nullptr && exact_value_threshold_sels != nullptr;
  const bool exact_values_by_selector_rank = exact_value_top_row < 0;
  const int64_t exact_value_limit = exact_values_by_selector_rank ? -exact_value_top_row : exact_value_top_row;
  const float exact_value_threshold =
      exact_values_by_threshold ? exact_value_thresholds[head] : INFINITY;
  const int64_t exact_value_threshold_sel =
      exact_values_by_threshold ? exact_value_threshold_sels[head] : -1;
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
  const float* q = queries + head * dim;
  int64_t* base_tokens_row = base_tokens + head * max_base;
  float* base_logits_row = base_logits + head * max_base;
  const int64_t* ranked_row = ranked_tokens + head * ranked;
  const float* ranked_score_row = ranked_scores + head * ranked;
  const float* ranked_logits_input_row = ranked_logits_input + head * ranked;
  float* ranked_logits_row = ranked_logits + head * ranked;
  unsigned char* ranked_exact_row = ranked_exact + head * ranked;
  unsigned char* tail_mask_row = tail_mask + head * (pages * page_size);

  __shared__ int64_t base_count_shared;
  if (threadIdx.x == 0) {
    int64_t base_count = 0;
    for (int64_t token = 0; token < prefix_end && base_count < max_base; ++token) {
      base_tokens_row[base_count] = token;
      ++base_count;
    }
    for (int64_t token = base_tail_start; token < query_context_len && base_count < max_base; ++token) {
      base_tokens_row[base_count] = token;
      ++base_count;
    }
    base_counts[head] = base_count;
    base_count_shared = base_count;
  }
  for (int64_t sel = threadIdx.x; sel < ranked; sel += blockDim.x) {
    ranked_logits_row[sel] = -INFINITY;
    ranked_exact_row[sel] = 0;
  }
  __syncthreads();

  int64_t base_count = base_count_shared;
  for (int64_t idx = warp_id; idx < base_count; idx += warps) {
    int64_t token = base_tokens_row[idx];
    float partial = 0.0f;
    for (int64_t d = lane; d < dim; d += 32) {
      partial += q[d] *
          load_strided3_as_float(keys, kv_head, token, d, key_stride_head, key_stride_token, key_stride_dim);
    }
    float dot = warp_reduce_sum(partial);
    if (lane == 0) {
      base_logits_row[idx] = dot * scale;
    }
  }

  for (int64_t sel = threadIdx.x; sel < ranked; sel += blockDim.x) {
    int64_t token = ranked_row[sel];
    float selector_score = ranked_score_row[sel];
    float logit = ranked_logits_input_row[sel];
    if (!isfinite(selector_score) || !isfinite(logit) || token < 0 || token >= query_context_len) {
      continue;
    }
    if (token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr)) {
      continue;
    }
    ranked_logits_row[sel] = logit;
    if (
        exact_values_by_threshold &&
        selected_mass_exact_value_for_rank(logit, sel, exact_value_threshold, exact_value_threshold_sel)) {
      ranked_exact_row[sel] = 1;
    }
    if (exact_values_by_selector_rank && sel < exact_value_limit) {
      ranked_exact_row[sel] = 1;
    }
    if (pages > 0) {
      int64_t first_start = page_starts[0];
      int64_t page = (token - first_start) / page_size;
      if (token >= first_start && page >= 0 && page < pages) {
        int64_t row = token - page_starts[page];
        bool valid_page = page_starts[page] >= prefix_end && page_starts[page] + page_size <= base_tail_start;
        if (valid_page && row >= 0 && row < page_size) {
          tail_mask_row[page * page_size + row] = 1;
        }
      }
    }
  }
  __syncthreads();
  if (exact_values_by_mass) {
    if (exact_values_by_threshold) {
      return;
    }
    if (threadIdx.x != 0) {
      return;
    }
    const float target = fminf(fmaxf(exact_value_mass, 0.0f), 1.0f);
    float max_logit = -INFINITY;
    for (int64_t idx = 0; idx < base_count; ++idx) {
      max_logit = fmaxf(max_logit, base_logits_row[idx]);
    }
    for (int64_t sel = 0; sel < ranked; ++sel) {
      max_logit = fmaxf(max_logit, ranked_logits_row[sel]);
    }
    if (!isfinite(max_logit)) {
      return;
    }
    float total = 0.0f;
    float exact_sum = 0.0f;
    int64_t exact_count = 0;
    for (int64_t idx = 0; idx < base_count; ++idx) {
      float logit = base_logits_row[idx];
      if (isfinite(logit)) {
        float weight = expf(logit - max_logit);
        total += weight;
        exact_sum += weight;
      }
    }
    for (int64_t sel = 0; sel < ranked; ++sel) {
      float logit = ranked_logits_row[sel];
      if (isfinite(logit)) {
        total += expf(logit - max_logit);
      }
    }
    while (total > 1.0e-20f && exact_sum / total < target) {
      int64_t best_sel = -1;
      float best_logit = -INFINITY;
      for (int64_t sel = 0; sel < ranked; ++sel) {
        if (ranked_exact_row[sel]) {
          continue;
        }
        float logit = ranked_logits_row[sel];
        if (logit > best_logit || (logit == best_logit && (best_sel < 0 || sel < best_sel))) {
          best_logit = logit;
          best_sel = sel;
        }
      }
      if (best_sel < 0 || !isfinite(best_logit)) {
        break;
      }
      ranked_exact_row[best_sel] = 1;
      ++exact_count;
      exact_sum += expf(best_logit - max_logit);
    }
    while (exact_value_limit > 0 && exact_count < exact_value_limit) {
      int64_t best_sel = -1;
      float best_logit = -INFINITY;
      for (int64_t sel = 0; sel < ranked; ++sel) {
        if (ranked_exact_row[sel]) {
          continue;
        }
        float logit = ranked_logits_row[sel];
        if (logit > best_logit || (logit == best_logit && (best_sel < 0 || sel < best_sel))) {
          best_logit = logit;
          best_sel = sel;
        }
      }
      if (best_sel < 0 || !isfinite(best_logit)) {
        break;
      }
      ranked_exact_row[best_sel] = 1;
      ++exact_count;
    }
    return;
  }
  if (exact_value_limit <= 0) {
    return;
  }
  if (exact_values_by_selector_rank) {
    return;
  }
  if (threadIdx.x != 0) {
    return;
  }
  if (exact_value_limit >= ranked) {
    for (int64_t sel = 0; sel < ranked; ++sel) {
      if (isfinite(ranked_logits_row[sel])) {
        ranked_exact_row[sel] = 1;
      }
    }
    return;
  }
  for (int64_t rank = 0; rank < exact_value_limit; ++rank) {
    int64_t best_sel = -1;
    float best_logit = -INFINITY;
    for (int64_t sel = 0; sel < ranked; ++sel) {
      if (ranked_exact_row[sel]) {
        continue;
      }
      float logit = ranked_logits_row[sel];
      if (logit > best_logit || (logit == best_logit && (best_sel < 0 || sel < best_sel))) {
        best_logit = logit;
        best_sel = sel;
      }
    }
    if (best_sel >= 0 && isfinite(best_logit)) {
      ranked_exact_row[best_sel] = 1;
    }
  }
}

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

template <typename value_t, typename vcode_t>
__global__ void gqa_decode_tail_agg_output_kernel(
    const value_t* __restrict__ values,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ base_tokens,
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_logits,
    const unsigned char* __restrict__ ranked_exact,
    const float* __restrict__ max_logits,
    const float* __restrict__ denoms,
    const float* __restrict__ selected_denoms,
    const float* __restrict__ code_weight_sums,
    float* __restrict__ outputs,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t value_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t max_base,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    float scale,
    float tail_blend) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * dim;
  if (linear >= total) {
    return;
  }
  int64_t head = linear / dim;
  int64_t d = linear - head * dim;
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
  const int64_t* base_tokens_row = base_tokens + head * max_base;
  const float* base_logits_row = base_logits + head * max_base;
  int64_t base_count = base_counts[head];
  const int64_t* ranked_row = ranked_tokens + head * ranked;
  const float* ranked_logits_row = ranked_logits + head * ranked;
  const unsigned char* ranked_exact_row = ranked_exact + head * ranked;
  float max_logit = max_logits[head];
  float denom = denoms[head];
  float selected_denom = selected_denoms[head];
  float selected_accum = 0.0f;
  float tail_accum = 0.0f;

  if (isfinite(max_logit)) {
    for (int64_t idx = 0; idx < base_count; ++idx) {
      int64_t token = base_tokens_row[idx];
      float weight = expf(base_logits_row[idx] - max_logit);
      selected_accum += weight *
          load_strided3_as_float(values, kv_head, token, d, value_stride_head, value_stride_token, value_stride_dim);
    }
    for (int64_t sel = 0; sel < ranked; ++sel) {
      float logit = ranked_logits_row[sel];
      if (!isfinite(logit)) {
        continue;
      }
      int64_t token = ranked_row[sel];
      float weight = expf(logit - max_logit);
      bool exact_value = ranked_exact_row[sel] != 0;
      float value = 0.0f;
      bool value_loaded = false;
      if (!exact_value && pages > 0 && value_subvecs > 0 && value_subdim > 0) {
        int64_t first_start = page_starts[0];
        int64_t page = (token - first_start) / page_size;
        if (token >= first_start && page >= 0 && page < pages) {
          int64_t row = token - page_starts[page];
          if (row >= 0 && row < page_size) {
            int64_t sub = d / value_subdim;
            int64_t sub_d = d - sub * value_subdim;
            if (sub >= 0 && sub < value_subvecs) {
              int64_t vcode = static_cast<int64_t>(
                  value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
              vcode = max(static_cast<int64_t>(0), min(vcode, value_centroids - 1));
              value = value_codebooks
                  [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + vcode) * value_subdim) +
                   sub_d];
              value_loaded = true;
            }
          }
        }
      }
      if (exact_value || !value_loaded) {
        value =
            load_strided3_as_float(values, kv_head, token, d, value_stride_head, value_stride_token, value_stride_dim);
      }
      selected_accum += weight * value;
    }
    if (tail_blend > 0.0f && pages > 0 && value_subvecs > 0 && value_subdim > 0) {
      int64_t sub = d / value_subdim;
      int64_t sub_d = d - sub * value_subdim;
      if (sub >= 0 && sub < value_subvecs) {
        for (int64_t page = 0; page < pages; ++page) {
          int64_t page_start = page_starts[page];
          bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
          if (!valid_page) {
            continue;
          }
          for (int64_t code = 0; code < value_centroids; ++code) {
            float weight_sum =
                code_weight_sums[(((head * pages + page) * value_subvecs + sub) * value_centroids + code)];
            float value = value_codebooks
                [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
                 sub_d];
            tail_accum += weight_sum * value;
          }
        }
      }
    }
  }
  float full = (selected_accum + tail_accum) / denom;
  if (tail_blend > 0.0f && tail_blend < 1.0f) {
    float selected_only = selected_accum / selected_denom;
    outputs[head * dim + d] = selected_only + tail_blend * (full - selected_only);
  } else {
    outputs[head * dim + d] = full;
  }
}

__device__ __forceinline__ int64_t round_budget_up_device(
    int64_t budget,
    int64_t granularity,
    int64_t max_budget) {
  if (granularity <= 1) {
    return budget < max_budget ? budget : max_budget;
  }
  int64_t rounded = ((budget + granularity - 1) / granularity) * granularity;
  if (rounded > max_budget) {
    rounded = max_budget;
  }
  return rounded;
}

__device__ __forceinline__ bool complete_page_for_token(
    const int64_t* __restrict__ page_starts,
    int64_t pages,
    int64_t page_size,
    int64_t token,
    int64_t query_context_len,
    int64_t prefix_end,
    int64_t base_tail_start,
    int64_t* __restrict__ page_out,
    int64_t* __restrict__ row_out) {
  if (pages <= 0 || page_size <= 0 || token < 0 || token >= query_context_len) {
    return false;
  }
  int64_t first_start = page_starts[0];
  if (token < first_start) {
    return false;
  }
  int64_t page = (token - first_start) / page_size;
  if (page < 0 || page >= pages) {
    return false;
  }
  int64_t page_start = page_starts[page];
  bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
  int64_t row = token - page_start;
  if (!valid_page || row < 0 || row >= page_size) {
    return false;
  }
  *page_out = page;
  *row_out = row;
  return true;
}

__device__ __forceinline__ bool exact_value_for_selector_rank(
    int64_t sel,
    int64_t exact_value_top,
    int64_t max_budget) {
  if (exact_value_top < 0) {
    return sel < -exact_value_top;
  }
  if (exact_value_top >= max_budget) {
    return true;
  }
  if (exact_value_top <= 0) {
    return false;
  }
  // Positive sub-max exact_value_top normally means "top by exact logit".
  // The incremental confidence path is only routed here for exact-all,
  // selector-rank, or no-exact cases; keep a conservative selector-rank
  // interpretation for accidental direct calls.
  return sel < exact_value_top;
}

template <typename vcode_t>
__device__ __forceinline__ float load_vpq_value_for_dim(
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    int64_t kv_head,
    int64_t page,
    int64_t row,
    int64_t dim_idx,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim) {
  if (value_subvecs <= 0 || value_centroids <= 0 || value_subdim <= 0) {
    return 0.0f;
  }
  int64_t sub = dim_idx / value_subdim;
  int64_t sub_d = dim_idx - sub * value_subdim;
  if (sub < 0 || sub >= value_subvecs) {
    return 0.0f;
  }
  int64_t code = static_cast<int64_t>(
      value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
  code = max(static_cast<int64_t>(0), min(code, value_centroids - 1));
  return value_codebooks
      [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
	       sub_d];
}

template <typename value_t, typename vcode_t>
__device__ __forceinline__ float selected_value_for_rank_dim_explicit(
    const value_t* __restrict__ values,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    int64_t token,
    bool exact_value,
    int64_t kv_head,
    int64_t dim_idx,
    int64_t total_tokens,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t value_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim);

template <typename value_t, typename vcode_t>
__device__ __forceinline__ float selected_value_for_rank_dim(
    const value_t* __restrict__ values,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    int64_t token,
    int64_t sel,
    int64_t exact_value_top,
    int64_t max_budget,
    int64_t kv_head,
    int64_t dim_idx,
    int64_t total_tokens,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t value_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim) {
  bool exact_value = exact_value_for_selector_rank(sel, exact_value_top, max_budget);
  return selected_value_for_rank_dim_explicit(
      values,
      value_codebooks,
      value_codes,
      page_starts,
      token,
      exact_value,
      kv_head,
      dim_idx,
      total_tokens,
      value_stride_head,
      value_stride_token,
      value_stride_dim,
      pages,
      page_size,
      value_subvecs,
      value_centroids,
      value_subdim);
}

template <typename value_t, typename vcode_t>
__device__ __forceinline__ float selected_value_for_rank_dim_explicit(
    const value_t* __restrict__ values,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    int64_t token,
    bool exact_value,
    int64_t kv_head,
    int64_t dim_idx,
    int64_t total_tokens,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t value_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim) {
  int64_t page = -1;
  int64_t row = -1;
  if (!exact_value && pages > 0 && page_size > 0) {
    int64_t first_start = page_starts[0];
    if (token >= first_start) {
      page = (token - first_start) / page_size;
      if (page >= 0 && page < pages) {
        row = token - page_starts[page];
      }
    }
    if (page >= 0 && page < pages && row >= 0 && row < page_size) {
      return load_vpq_value_for_dim(
          value_codebooks,
          value_codes,
          kv_head,
          page,
          row,
          dim_idx,
          pages,
          page_size,
          value_subvecs,
          value_centroids,
          value_subdim);
    }
  }
  if (token < 0 || token >= total_tokens) {
    return 0.0f;
  }
  return load_strided3_as_float(
      values,
      kv_head,
      token,
      dim_idx,
      value_stride_head,
      value_stride_token,
      value_stride_dim);
}

__device__ __forceinline__ bool selected_mass_exact_value_for_rank(
    float ranked_logit,
    int64_t sel,
    float threshold_logit,
    int64_t threshold_sel) {
  if (threshold_sel < 0 || !isfinite(ranked_logit) || !isfinite(threshold_logit)) {
    return false;
  }
  return ranked_logit > threshold_logit || (ranked_logit == threshold_logit && sel <= threshold_sel);
}

__device__ __forceinline__ void selected_mass_exact_threshold_device(
    const float* __restrict__ base_logits,
    int64_t base_count,
    const float* __restrict__ ranked_logits,
    int64_t budget,
    float exact_value_mass,
    int64_t exact_value_min_top,
    float* __restrict__ threshold_logit_out,
    int64_t* __restrict__ threshold_sel_out) {
  *threshold_logit_out = INFINITY;
  *threshold_sel_out = -1;
  if (budget <= 0 || (exact_value_mass <= 0.0f && exact_value_min_top <= 0)) {
    return;
  }
  float target = fminf(fmaxf(exact_value_mass, 0.0f), 1.0f);
  int64_t min_top = max(static_cast<int64_t>(0), exact_value_min_top);
  float max_logit = -INFINITY;
  for (int64_t idx = 0; idx < base_count; ++idx) {
    max_logit = fmaxf(max_logit, base_logits[idx]);
  }
  for (int64_t sel = 0; sel < budget; ++sel) {
    max_logit = fmaxf(max_logit, ranked_logits[sel]);
  }
  if (!isfinite(max_logit)) {
    return;
  }
  float total = 0.0f;
  float exact_sum = 0.0f;
  for (int64_t idx = 0; idx < base_count; ++idx) {
    float logit = base_logits[idx];
    if (isfinite(logit)) {
      float weight = expf(logit - max_logit);
      total += weight;
      exact_sum += weight;
    }
  }
  for (int64_t sel = 0; sel < budget; ++sel) {
    float logit = ranked_logits[sel];
    if (isfinite(logit)) {
      total += expf(logit - max_logit);
    }
  }
  if (total <= 1.0e-20f) {
    return;
  }

  int64_t exact_count = 0;
  float prev_logit = INFINITY;
  int64_t prev_sel = -1;
  while ((target > 0.0f && exact_sum / total < target) || exact_count < min_top) {
    int64_t best_sel = -1;
    float best_logit = -INFINITY;
    for (int64_t sel = 0; sel < budget; ++sel) {
      float logit = ranked_logits[sel];
      if (!isfinite(logit)) {
        continue;
      }
      if (exact_count > 0 && !(logit < prev_logit || (logit == prev_logit && sel > prev_sel))) {
        continue;
      }
      if (logit > best_logit || (logit == best_logit && (best_sel < 0 || sel < best_sel))) {
        best_logit = logit;
        best_sel = sel;
      }
    }
    if (best_sel < 0 || !isfinite(best_logit)) {
      break;
    }
    exact_sum += expf(best_logit - max_logit);
    ++exact_count;
    prev_logit = best_logit;
    prev_sel = best_sel;
  }
  if (exact_count > 0) {
    *threshold_logit_out = prev_logit;
    *threshold_sel_out = prev_sel;
  }
}

__global__ void gqa_decode_geometric_selected_rank_kernel(
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    int32_t* __restrict__ selected_ranks,
    int64_t heads,
    int64_t ranked,
    int64_t pages,
    int64_t page_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * ranked;
  if (linear >= total || pages <= 0 || page_size <= 0) {
    return;
  }
  int64_t head = linear / ranked;
  int64_t sel = linear - head * ranked;
  float selector_score = ranked_scores[linear];
  int64_t token = ranked_tokens[linear];
  if (!isfinite(selector_score) || token < 0 || token >= query_context_len) {
    return;
  }
  if (token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr)) {
    return;
  }
  int64_t first_start = page_starts[0];
  if (token < first_start) {
    return;
  }
  int64_t page = (token - first_start) / page_size;
  if (page < 0 || page >= pages) {
    return;
  }
  int64_t row = token - page_starts[page];
  if (row < 0 || row >= page_size) {
    return;
  }
  selected_ranks[head * (pages * page_size) + page * page_size + row] = static_cast<int32_t>(sel);
}

template <typename value_t, typename vcode_t>
__device__ float gqa_decode_geometric_output_dim(
    const value_t* __restrict__ values,
    const float* __restrict__ dense_pq_scores,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ base_tokens,
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_logits,
    const int32_t* __restrict__ selected_ranks,
    int64_t head,
    int64_t dim_idx,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t value_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t max_base,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t keep,
    float scale,
    bool include_tail) {
  (void)dim;
  int64_t kv_head = head / group_size;
  if (kv_head >= kv_heads) {
    kv_head = kv_heads - 1;
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
  const int64_t* base_tokens_row = base_tokens + head * max_base;
  const float* base_logits_row = base_logits + head * max_base;
  int64_t base_count = base_counts[head];
  const int64_t* ranked_row = ranked_tokens + head * ranked;
  const float* ranked_logit_row = ranked_logits + head * ranked;
  const float* dense_row = dense_pq_scores + head * (pages * page_size);

  float max_logit = -INFINITY;
  for (int64_t idx = 0; idx < base_count; ++idx) {
    max_logit = fmaxf(max_logit, base_logits_row[idx]);
  }
  int64_t keep_clamped = keep;
  if (keep_clamped < 0) {
    keep_clamped = 0;
  }
  if (keep_clamped > ranked) {
    keep_clamped = ranked;
  }
  for (int64_t sel = 0; sel < keep_clamped; ++sel) {
    max_logit = fmaxf(max_logit, ranked_logit_row[sel]);
  }
  if (include_tail && pages > 0) {
    for (int64_t page = 0; page < pages; ++page) {
      int64_t page_start = page_starts[page];
      bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
      if (!valid_page) {
        continue;
      }
      for (int64_t row = 0; row < page_size; ++row) {
        int64_t token = page_start + row;
        if (token < 0 || token >= query_context_len || token >= total_tokens) {
          continue;
        }
        int32_t selected_rank = selected_ranks[head * (pages * page_size) + page * page_size + row];
        if (selected_rank >= 0 && selected_rank < keep_clamped) {
          continue;
        }
        max_logit = fmaxf(max_logit, dense_row[page * page_size + row] * scale);
      }
    }
  }
  if (!isfinite(max_logit)) {
    return 0.0f;
  }

  float denom = 0.0f;
  float accum = 0.0f;
  for (int64_t idx = 0; idx < base_count; ++idx) {
    int64_t token = base_tokens_row[idx];
    if (token < 0 || token >= total_tokens) {
      continue;
    }
    float weight = expf(base_logits_row[idx] - max_logit);
    denom += weight;
    accum += weight * load_strided3_as_float(values, kv_head, token, dim_idx, value_stride_head, value_stride_token, value_stride_dim);
  }
  for (int64_t sel = 0; sel < keep_clamped; ++sel) {
    int64_t token = ranked_row[sel];
    float logit = ranked_logit_row[sel];
    if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
      continue;
    }
    float weight = expf(logit - max_logit);
    denom += weight;
    accum += weight * load_strided3_as_float(values, kv_head, token, dim_idx, value_stride_head, value_stride_token, value_stride_dim);
  }
  if (include_tail && pages > 0 && value_subvecs > 0 && value_subdim > 0) {
    int64_t sub = dim_idx / value_subdim;
    int64_t sub_d = dim_idx - sub * value_subdim;
    if (sub >= 0 && sub < value_subvecs) {
      for (int64_t page = 0; page < pages; ++page) {
        int64_t page_start = page_starts[page];
        bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
        if (!valid_page) {
          continue;
        }
        for (int64_t row = 0; row < page_size; ++row) {
          int64_t token = page_start + row;
          if (token < 0 || token >= query_context_len || token >= total_tokens) {
            continue;
          }
          int32_t selected_rank = selected_ranks[head * (pages * page_size) + page * page_size + row];
          if (selected_rank >= 0 && selected_rank < keep_clamped) {
            continue;
          }
          int64_t vcode = static_cast<int64_t>(
              value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
          if (vcode < 0) {
            vcode = 0;
          }
          if (vcode >= value_centroids) {
            vcode = value_centroids - 1;
          }
          float value = value_codebooks
              [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + vcode) * value_subdim) +
               sub_d];
          float weight = expf(dense_row[page * page_size + row] * scale - max_logit);
          denom += weight;
          accum += weight * value;
        }
      }
    }
  }
  if (denom <= 0.0f || !isfinite(denom)) {
    return 0.0f;
  }
  return accum / denom;
}

template <typename value_t, typename vcode_t>
__global__ void gqa_decode_geometric_accept_counts_kernel(
    const value_t* __restrict__ values,
    const float* __restrict__ dense_pq_scores,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ base_tokens,
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    int64_t* __restrict__ accepted_counts,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t value_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t max_base,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t min_budget,
    int64_t max_budget_arg,
    int64_t granularity,
	    float growth,
	    float probe_scale,
	    float rel_l2_max,
	    int64_t exact_value_top,
	    float exact_value_mass,
	    int64_t exact_value_min_top,
	    float scale,
	    bool probe_includes_tail) {
  int64_t head = blockIdx.x;
  if (head >= heads) {
    return;
  }
  extern __shared__ float shared[];
  float* diff_sums = shared;
	  float* probe_sums = shared + blockDim.x;
	  __shared__ int64_t accepted_shared;
	  __shared__ int done_shared;
	  __shared__ float approx_exact_threshold_shared;
	  __shared__ float probe_exact_threshold_shared;
	  __shared__ int64_t approx_exact_threshold_sel_shared;
	  __shared__ int64_t probe_exact_threshold_sel_shared;
  __shared__ float global_max_shared;

  int64_t max_budget = max_budget_arg;
  if (max_budget <= 0 || max_budget > ranked) {
    max_budget = ranked;
  }
  int64_t start_budget = min_budget;
  if (start_budget < 0) {
    start_budget = 0;
  }
  if (start_budget > max_budget) {
    start_budget = max_budget;
  }
  start_budget = round_budget_up_device(start_budget, granularity, max_budget);

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
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  const int64_t* base_tokens_row = base_tokens + head * max_base;
  const float* base_logits_row = base_logits + head * max_base;
  int64_t base_count = base_counts[head];
  const int64_t* ranked_row = ranked_tokens + head * ranked;
  const float* ranked_score_row = ranked_scores + head * ranked;
  const float* dense_row = dense_pq_scores + head * (pages * page_size);

  float local_max = -INFINITY;
  for (int64_t idx = threadIdx.x; idx < base_count; idx += blockDim.x) {
    local_max = fmaxf(local_max, base_logits_row[idx]);
  }
  for (int64_t sel = threadIdx.x; sel < max_budget; sel += blockDim.x) {
    float selector_score = ranked_score_row[sel];
    if (isfinite(selector_score)) {
      local_max = fmaxf(local_max, selector_score * scale);
    }
  }
  int64_t flat_tokens = pages * page_size;
  for (int64_t ordinal = threadIdx.x; ordinal < flat_tokens; ordinal += blockDim.x) {
    int64_t page = ordinal / page_size;
    int64_t row = ordinal - page * page_size;
    int64_t page_start = page_starts[page];
    bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
    int64_t token = page_start + row;
    if (valid_page && token >= 0 && token < query_context_len && token < total_tokens) {
      local_max = fmaxf(local_max, dense_row[ordinal] * scale);
    }
  }
  diff_sums[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      diff_sums[threadIdx.x] = fmaxf(diff_sums[threadIdx.x], diff_sums[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    accepted_shared = max_budget;
    done_shared = 0;
    global_max_shared = diff_sums[0];
  }
  __syncthreads();

  float global_max = global_max_shared;
  if (!isfinite(global_max)) {
    if (threadIdx.x == 0) {
      accepted_counts[head] = max_budget;
    }
    return;
  }

  int64_t d = threadIdx.x;
  bool active_dim = d < dim;
  float base_den = 0.0f;
  float base_num = 0.0f;
  float tail_den = 0.0f;
  float tail_num = 0.0f;
  if (active_dim) {
    for (int64_t idx = 0; idx < base_count; ++idx) {
      int64_t token = base_tokens_row[idx];
      if (token < 0 || token >= total_tokens) {
        continue;
      }
      float logit = base_logits_row[idx];
      if (!isfinite(logit)) {
        continue;
      }
      float w = expf(logit - global_max);
      base_den += w;
      base_num += w * load_strided3_as_float(
                          values,
                          kv_head,
                          token,
                          d,
                          value_stride_head,
                          value_stride_token,
                          value_stride_dim);
    }
    for (int64_t ordinal = 0; ordinal < flat_tokens; ++ordinal) {
      int64_t page = ordinal / page_size;
      int64_t row = ordinal - page * page_size;
      int64_t page_start = page_starts[page];
      bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
      int64_t token = page_start + row;
      if (!valid_page || token < 0 || token >= query_context_len || token >= total_tokens) {
        continue;
      }
      float w = expf(dense_row[ordinal] * scale - global_max);
      tail_den += w;
      tail_num += w * load_vpq_value_for_dim(
                        value_codebooks,
                        value_codes,
                        kv_head,
                        page,
                        row,
                        d,
                        pages,
                        page_size,
                        value_subvecs,
                        value_centroids,
                        value_subdim);
    }
  }

  float approx_den = base_den + tail_den;
  float approx_num = base_num + tail_num;
  float probe_den = probe_includes_tail ? (base_den + tail_den) : base_den;
  float probe_num = probe_includes_tail ? (base_num + tail_num) : base_num;
	  int64_t approx_keep = 0;
	  int64_t probe_keep = 0;
	  int64_t tail_budget = start_budget;
	  // This legacy selector-score kernel is kept for older entry points. The
	  // canonical selected-mass path uses the codeweights kernel with exact QK
	  // logits, so do not apply selected-mass V routing here from PQ scores.
	  bool selected_mass_exact = false;
	  while (tail_budget < max_budget) {
    float target = fmaxf(static_cast<float>(tail_budget + granularity), probe_scale * static_cast<float>(tail_budget));
    int64_t probe_budget = round_budget_up_device(static_cast<int64_t>(ceilf(target)), granularity, max_budget);
	    if (probe_budget < tail_budget) {
	      probe_budget = tail_budget;
	    }
	    if (threadIdx.x == 0) {
	      selected_mass_exact_threshold_device(
	          base_logits_row,
	          base_count,
	          ranked_score_row,
	          tail_budget,
	          exact_value_mass,
	          exact_value_min_top,
	          &approx_exact_threshold_shared,
	          &approx_exact_threshold_sel_shared);
	      selected_mass_exact_threshold_device(
	          base_logits_row,
	          base_count,
	          ranked_score_row,
	          probe_budget,
	          exact_value_mass,
	          exact_value_min_top,
	          &probe_exact_threshold_shared,
	          &probe_exact_threshold_sel_shared);
	    }
	    __syncthreads();

	    float local_diff = 0.0f;
	    float local_probe = 0.0f;
    if (active_dim) {
      for (int64_t sel = approx_keep; sel < tail_budget; ++sel) {
        float selector_score = ranked_score_row[sel];
        float logit = selector_score * scale;
        int64_t token = ranked_row[sel];
        if (!isfinite(selector_score) || token < 0 || token >= query_context_len || token >= total_tokens) {
          continue;
        }
	        float selected_w = expf(logit - global_max);
	        float selected_v = 0.0f;
	        if (selected_mass_exact) {
	          bool exact_value = selected_mass_exact_value_for_rank(
	              logit, sel, approx_exact_threshold_shared, approx_exact_threshold_sel_shared);
	          selected_v = selected_value_for_rank_dim_explicit(
	              values,
	              value_codebooks,
	              value_codes,
	              page_starts,
	              token,
	              exact_value,
	              kv_head,
	              d,
	              total_tokens,
	              value_stride_head,
	              value_stride_token,
	              value_stride_dim,
	              pages,
	              page_size,
	              value_subvecs,
	              value_centroids,
	              value_subdim);
	        } else {
	          selected_v = selected_value_for_rank_dim(
	              values,
	              value_codebooks,
	              value_codes,
	              page_starts,
	              token,
	              sel,
	              exact_value_top,
	              max_budget,
	              kv_head,
	              d,
	              total_tokens,
	              value_stride_head,
	              value_stride_token,
	              value_stride_dim,
	              pages,
	              page_size,
	              value_subvecs,
	              value_centroids,
	              value_subdim);
	        }
        approx_den += selected_w;
        approx_num += selected_w * selected_v;
        int64_t page = -1;
        int64_t row = -1;
        if (complete_page_for_token(
                page_starts,
                pages,
                page_size,
                token,
                query_context_len,
                prefix_end,
                base_tail_start,
                &page,
                &row)) {
          float pq_w = expf(dense_row[page * page_size + row] * scale - global_max);
          float pq_v = load_vpq_value_for_dim(
              value_codebooks,
              value_codes,
              kv_head,
              page,
              row,
              d,
              pages,
              page_size,
              value_subvecs,
              value_centroids,
              value_subdim);
          approx_den -= pq_w;
          approx_num -= pq_w * pq_v;
        }
      }
      approx_keep = tail_budget;
      for (int64_t sel = probe_keep; sel < probe_budget; ++sel) {
        float selector_score = ranked_score_row[sel];
        float logit = selector_score * scale;
        int64_t token = ranked_row[sel];
        if (!isfinite(selector_score) || token < 0 || token >= query_context_len || token >= total_tokens) {
          continue;
        }
	        float selected_w = expf(logit - global_max);
	        float selected_v = 0.0f;
	        if (selected_mass_exact) {
	          bool exact_value = selected_mass_exact_value_for_rank(
	              logit, sel, probe_exact_threshold_shared, probe_exact_threshold_sel_shared);
	          selected_v = selected_value_for_rank_dim_explicit(
	              values,
	              value_codebooks,
	              value_codes,
	              page_starts,
	              token,
	              exact_value,
	              kv_head,
	              d,
	              total_tokens,
	              value_stride_head,
	              value_stride_token,
	              value_stride_dim,
	              pages,
	              page_size,
	              value_subvecs,
	              value_centroids,
	              value_subdim);
	        } else {
	          selected_v = selected_value_for_rank_dim(
	              values,
	              value_codebooks,
	              value_codes,
	              page_starts,
	              token,
	              sel,
	              exact_value_top,
	              max_budget,
	              kv_head,
	              d,
	              total_tokens,
	              value_stride_head,
	              value_stride_token,
	              value_stride_dim,
	              pages,
	              page_size,
	              value_subvecs,
	              value_centroids,
	              value_subdim);
	        }
        probe_den += selected_w;
        probe_num += selected_w * selected_v;
        if (probe_includes_tail) {
          int64_t page = -1;
          int64_t row = -1;
          if (complete_page_for_token(
                  page_starts,
                  pages,
                  page_size,
                  token,
                  query_context_len,
                  prefix_end,
                  base_tail_start,
                  &page,
                  &row)) {
            float pq_w = expf(dense_row[page * page_size + row] * scale - global_max);
            float pq_v = load_vpq_value_for_dim(
                value_codebooks,
                value_codes,
                kv_head,
                page,
                row,
                d,
                pages,
                page_size,
                value_subvecs,
                value_centroids,
                value_subdim);
            probe_den -= pq_w;
            probe_num -= pq_w * pq_v;
          }
        }
      }
      probe_keep = probe_budget;
      float approx_tail = approx_num / fmaxf(approx_den, 1.0e-20f);
      float probe_only = probe_num / fmaxf(probe_den, 1.0e-20f);
      float delta = approx_tail - probe_only;
      local_diff += delta * delta;
      local_probe += probe_only * probe_only;
    }
    diff_sums[threadIdx.x] = local_diff;
    probe_sums[threadIdx.x] = local_probe;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        diff_sums[threadIdx.x] += diff_sums[threadIdx.x + stride];
        probe_sums[threadIdx.x] += probe_sums[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      float denom = sqrtf(fmaxf(probe_sums[0], 1.0e-40f));
      float rel = sqrtf(fmaxf(diff_sums[0], 0.0f)) / denom;
      if (rel <= rel_l2_max) {
        accepted_shared = probe_budget;
        done_shared = 1;
      }
    }
    __syncthreads();
    if (done_shared != 0 || probe_budget >= max_budget) {
      break;
    }
    float next_target = fmaxf(static_cast<float>(probe_budget + granularity), growth * static_cast<float>(probe_budget));
    int64_t next_budget = round_budget_up_device(static_cast<int64_t>(ceilf(next_target)), granularity, max_budget);
    if (next_budget <= probe_budget) {
      break;
    }
    tail_budget = next_budget;
  }
  if (threadIdx.x == 0) {
    accepted_counts[head] = accepted_shared;
  }
}

template <typename value_t, typename vcode_t>
__global__ void gqa_decode_geometric_accept_counts_codeweights_kernel(
    const value_t* __restrict__ values,
    const float* __restrict__ dense_pq_scores,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ base_tokens,
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_logits,
    const float* __restrict__ max_logits,
    const float* __restrict__ tail_denoms,
    const float* __restrict__ code_weight_sums,
    const float* __restrict__ approx_exact_thresholds,
    const int64_t* __restrict__ approx_exact_threshold_sels,
    const float* __restrict__ probe_exact_thresholds,
    const int64_t* __restrict__ probe_exact_threshold_sels,
    int64_t* __restrict__ accepted_counts,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t value_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t max_base,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t min_budget,
    int64_t max_budget_arg,
    int64_t granularity,
    float growth,
	    float probe_scale,
	    float rel_l2_max,
	    int64_t exact_value_top,
	    float exact_value_mass,
	    int64_t exact_value_min_top,
	    float scale,
	    bool probe_includes_tail,
	    int64_t threshold_steps) {
  int64_t head = blockIdx.x;
  if (head >= heads) {
    return;
  }
  extern __shared__ float shared[];
  float* diff_sums = shared;
	  float* probe_sums = shared + blockDim.x;
	  __shared__ int64_t accepted_shared;
	  __shared__ int done_shared;
	  __shared__ float approx_exact_threshold_shared;
	  __shared__ float probe_exact_threshold_shared;
	  __shared__ int64_t approx_exact_threshold_sel_shared;
	  __shared__ int64_t probe_exact_threshold_sel_shared;

  int64_t max_budget = max_budget_arg;
  if (max_budget <= 0 || max_budget > ranked) {
    max_budget = ranked;
  }
  int64_t start_budget = min_budget;
  if (start_budget < 0) {
    start_budget = 0;
  }
  if (start_budget > max_budget) {
    start_budget = max_budget;
  }
  start_budget = round_budget_up_device(start_budget, granularity, max_budget);

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
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  const int64_t* base_tokens_row = base_tokens + head * max_base;
  const float* base_logits_row = base_logits + head * max_base;
  int64_t base_count = base_counts[head];
  const int64_t* ranked_row = ranked_tokens + head * ranked;
  const float* ranked_logit_row = ranked_logits + head * ranked;
  const float* dense_row = dense_pq_scores + head * (pages * page_size);
  float global_max = max_logits[head];

  if (threadIdx.x == 0) {
    accepted_shared = max_budget;
    done_shared = 0;
  }
  __syncthreads();
  if (!isfinite(global_max)) {
    if (threadIdx.x == 0) {
      accepted_counts[head] = max_budget;
    }
    return;
  }

  int64_t d = threadIdx.x;
  bool active_dim = d < dim;
  float base_den = 0.0f;
  float base_num = 0.0f;
  float tail_den = tail_denoms[head];
  float tail_num = 0.0f;
  if (active_dim) {
    for (int64_t idx = 0; idx < base_count; ++idx) {
      int64_t token = base_tokens_row[idx];
      if (token < 0 || token >= total_tokens) {
        continue;
      }
      float logit = base_logits_row[idx];
      if (!isfinite(logit)) {
        continue;
      }
      float w = expf(logit - global_max);
      base_den += w;
      base_num += w * load_strided3_as_float(
                          values,
                          kv_head,
                          token,
                          d,
                          value_stride_head,
                          value_stride_token,
                          value_stride_dim);
    }
    if (value_subdim > 0) {
      int64_t sub = d / value_subdim;
      int64_t sub_d = d - sub * value_subdim;
      if (sub >= 0 && sub < value_subvecs) {
        for (int64_t page = 0; page < pages; ++page) {
          for (int64_t code = 0; code < value_centroids; ++code) {
            float w = code_weight_sums[(((head * pages + page) * value_subvecs + sub) * value_centroids + code)];
            if (w == 0.0f) {
              continue;
            }
            float value = value_codebooks
                [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
                 sub_d];
            tail_num += w * value;
          }
        }
      }
    }
  }

  float approx_den = base_den + tail_den;
  float approx_num = base_num + tail_num;
  float probe_den = probe_includes_tail ? (base_den + tail_den) : base_den;
  float probe_num = probe_includes_tail ? (base_num + tail_num) : base_num;
	  int64_t approx_keep = 0;
	  int64_t probe_keep = 0;
	  int64_t tail_budget = start_budget;
	  bool selected_mass_exact = exact_value_mass > 0.0f;
	  int64_t budget_step = 0;
	  while (tail_budget < max_budget) {
	    float target = fmaxf(static_cast<float>(tail_budget + granularity), probe_scale * static_cast<float>(tail_budget));
	    int64_t probe_budget = round_budget_up_device(static_cast<int64_t>(ceilf(target)), granularity, max_budget);
	    if (probe_budget < tail_budget) {
	      probe_budget = tail_budget;
	    }
	    if (threadIdx.x == 0) {
	      if (selected_mass_exact && threshold_steps > 0 && budget_step < threshold_steps &&
	          approx_exact_thresholds != nullptr && approx_exact_threshold_sels != nullptr &&
	          probe_exact_thresholds != nullptr && probe_exact_threshold_sels != nullptr) {
	        int64_t thresh_idx = head * threshold_steps + budget_step;
	        approx_exact_threshold_shared = approx_exact_thresholds[thresh_idx];
	        approx_exact_threshold_sel_shared = approx_exact_threshold_sels[thresh_idx];
	        probe_exact_threshold_shared = probe_exact_thresholds[thresh_idx];
	        probe_exact_threshold_sel_shared = probe_exact_threshold_sels[thresh_idx];
	      } else {
	        selected_mass_exact_threshold_device(
	            base_logits_row,
	            base_count,
	            ranked_logit_row,
	            tail_budget,
	            exact_value_mass,
	            exact_value_min_top,
	            &approx_exact_threshold_shared,
	            &approx_exact_threshold_sel_shared);
	        selected_mass_exact_threshold_device(
	            base_logits_row,
	            base_count,
	            ranked_logit_row,
	            probe_budget,
	            exact_value_mass,
	            exact_value_min_top,
	            &probe_exact_threshold_shared,
	            &probe_exact_threshold_sel_shared);
	      }
	    }
	    __syncthreads();

	    float local_diff = 0.0f;
    float local_probe = 0.0f;
    if (active_dim) {
      for (int64_t sel = approx_keep; sel < tail_budget; ++sel) {
        float logit = ranked_logit_row[sel];
        int64_t token = ranked_row[sel];
        if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
          continue;
	        }
	        float selected_w = expf(logit - global_max);
	        float selected_v = 0.0f;
	        if (selected_mass_exact) {
	          bool exact_value = selected_mass_exact_value_for_rank(
	              logit, sel, approx_exact_threshold_shared, approx_exact_threshold_sel_shared);
	          selected_v = selected_value_for_rank_dim_explicit(
	              values,
	              value_codebooks,
	              value_codes,
	              page_starts,
	              token,
	              exact_value,
	              kv_head,
	              d,
	              total_tokens,
	              value_stride_head,
	              value_stride_token,
	              value_stride_dim,
	              pages,
	              page_size,
	              value_subvecs,
	              value_centroids,
	              value_subdim);
	        } else {
	          selected_v = selected_value_for_rank_dim(
	              values,
	              value_codebooks,
	              value_codes,
	              page_starts,
	              token,
	              sel,
	              exact_value_top,
	              max_budget,
	              kv_head,
	              d,
	              total_tokens,
	              value_stride_head,
	              value_stride_token,
	              value_stride_dim,
	              pages,
	              page_size,
	              value_subvecs,
	              value_centroids,
	              value_subdim);
	        }
        approx_den += selected_w;
        approx_num += selected_w * selected_v;
        int64_t page = -1;
        int64_t row = -1;
        if (complete_page_for_token(
                page_starts,
                pages,
                page_size,
                token,
                query_context_len,
                prefix_end,
                base_tail_start,
                &page,
                &row)) {
          float pq_w = expf(dense_row[page * page_size + row] * scale - global_max);
          float pq_v = load_vpq_value_for_dim(
              value_codebooks,
              value_codes,
              kv_head,
              page,
              row,
              d,
              pages,
              page_size,
              value_subvecs,
              value_centroids,
              value_subdim);
          approx_den -= pq_w;
          approx_num -= pq_w * pq_v;
        }
      }
      approx_keep = tail_budget;
      for (int64_t sel = probe_keep; sel < probe_budget; ++sel) {
        float logit = ranked_logit_row[sel];
        int64_t token = ranked_row[sel];
        if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
          continue;
	        }
	        float selected_w = expf(logit - global_max);
	        float selected_v = 0.0f;
	        if (selected_mass_exact) {
	          bool exact_value = selected_mass_exact_value_for_rank(
	              logit, sel, probe_exact_threshold_shared, probe_exact_threshold_sel_shared);
	          selected_v = selected_value_for_rank_dim_explicit(
	              values,
	              value_codebooks,
	              value_codes,
	              page_starts,
	              token,
	              exact_value,
	              kv_head,
	              d,
	              total_tokens,
	              value_stride_head,
	              value_stride_token,
	              value_stride_dim,
	              pages,
	              page_size,
	              value_subvecs,
	              value_centroids,
	              value_subdim);
	        } else {
	          selected_v = selected_value_for_rank_dim(
	              values,
	              value_codebooks,
	              value_codes,
	              page_starts,
	              token,
	              sel,
	              exact_value_top,
	              max_budget,
	              kv_head,
	              d,
	              total_tokens,
	              value_stride_head,
	              value_stride_token,
	              value_stride_dim,
	              pages,
	              page_size,
	              value_subvecs,
	              value_centroids,
	              value_subdim);
	        }
        probe_den += selected_w;
        probe_num += selected_w * selected_v;
        if (probe_includes_tail) {
          int64_t page = -1;
          int64_t row = -1;
          if (complete_page_for_token(
                  page_starts,
                  pages,
                  page_size,
                  token,
                  query_context_len,
                  prefix_end,
                  base_tail_start,
                  &page,
                  &row)) {
            float pq_w = expf(dense_row[page * page_size + row] * scale - global_max);
            float pq_v = load_vpq_value_for_dim(
                value_codebooks,
                value_codes,
                kv_head,
                page,
                row,
                d,
                pages,
                page_size,
                value_subvecs,
                value_centroids,
                value_subdim);
            probe_den -= pq_w;
            probe_num -= pq_w * pq_v;
          }
        }
      }
      probe_keep = probe_budget;
      float approx_tail = approx_num / fmaxf(approx_den, 1.0e-20f);
      float probe_only = probe_num / fmaxf(probe_den, 1.0e-20f);
      float delta = approx_tail - probe_only;
      local_diff += delta * delta;
      local_probe += probe_only * probe_only;
    }
    diff_sums[threadIdx.x] = local_diff;
    probe_sums[threadIdx.x] = local_probe;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        diff_sums[threadIdx.x] += diff_sums[threadIdx.x + stride];
        probe_sums[threadIdx.x] += probe_sums[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      float denom = sqrtf(fmaxf(probe_sums[0], 1.0e-40f));
      float rel = sqrtf(fmaxf(diff_sums[0], 0.0f)) / denom;
      if (rel <= rel_l2_max) {
        accepted_shared = probe_budget;
        done_shared = 1;
      }
    }
    __syncthreads();
    if (done_shared != 0 || probe_budget >= max_budget) {
      break;
    }
    float next_target = fmaxf(static_cast<float>(probe_budget + granularity), growth * static_cast<float>(probe_budget));
    int64_t next_budget = round_budget_up_device(static_cast<int64_t>(ceilf(next_target)), granularity, max_budget);
    if (next_budget <= probe_budget) {
      break;
    }
    tail_budget = next_budget;
    ++budget_step;
  }
  if (threadIdx.x == 0) {
    accepted_counts[head] = accepted_shared;
  }
}

template <typename value_t, typename vcode_t>
__global__ void gqa_decode_geometric_final_output_codeweights_kernel(
    const value_t* __restrict__ values,
    const float* __restrict__ dense_pq_scores,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ base_tokens,
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_logits,
    const float* __restrict__ tail_partial_max,
    const float* __restrict__ confidence_max_logits,
    const float* __restrict__ tail_denoms,
    const float* __restrict__ code_weight_sums,
    const int64_t* __restrict__ accepted_counts,
    float* __restrict__ outputs,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t value_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t max_base,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t tail_blocks,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    float exact_value_mass,
    int64_t exact_value_min_top,
    float scale,
    float tail_blend) {
  int64_t head = blockIdx.x;
  if (head >= heads) {
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
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  const int64_t* base_tokens_row = base_tokens + head * max_base;
  const float* base_logits_row = base_logits + head * max_base;
  int64_t base_count = base_counts[head];
  const int64_t* ranked_row = ranked_tokens + head * ranked;
  const float* ranked_logit_row = ranked_logits + head * ranked;
  const float* dense_row = dense_pq_scores + head * (pages * page_size);

  __shared__ int64_t accepted_shared;
  __shared__ float final_max_shared;
  __shared__ float threshold_logit_shared;
  __shared__ int64_t threshold_sel_shared;
  if (threadIdx.x == 0) {
    int64_t accepted = accepted_counts[head];
    if (accepted < 0) {
      accepted = 0;
    }
    if (accepted > ranked) {
      accepted = ranked;
    }
    accepted_shared = accepted;
    float final_max = -INFINITY;
    for (int64_t idx = 0; idx < base_count; ++idx) {
      final_max = fmaxf(final_max, base_logits_row[idx]);
    }
    for (int64_t sel = 0; sel < accepted; ++sel) {
      final_max = fmaxf(final_max, ranked_logit_row[sel]);
    }
    for (int64_t block = 0; block < tail_blocks; ++block) {
      final_max = fmaxf(final_max, tail_partial_max[head * tail_blocks + block]);
    }
    final_max_shared = final_max;
    selected_mass_exact_threshold_device(
        base_logits_row,
        base_count,
        ranked_logit_row,
        accepted,
        exact_value_mass,
        exact_value_min_top,
        &threshold_logit_shared,
        &threshold_sel_shared);
  }
  __syncthreads();

  int64_t d = threadIdx.x;
  if (d >= dim) {
    return;
  }
  int64_t accepted = accepted_shared;
  float final_max = final_max_shared;
  if (!isfinite(final_max)) {
    outputs[head * dim + d] = 0.0f;
    return;
  }
  float threshold_logit = threshold_logit_shared;
  int64_t threshold_sel = threshold_sel_shared;

  float denom = 0.0f;
  float selected_denom = 0.0f;
  float numer = 0.0f;
  float selected_numer = 0.0f;
  for (int64_t idx = 0; idx < base_count; ++idx) {
    int64_t token = base_tokens_row[idx];
    float logit = base_logits_row[idx];
    if (token < 0 || token >= total_tokens || !isfinite(logit)) {
      continue;
    }
    float w = expf(logit - final_max);
    float v = load_strided3_as_float(
        values,
        kv_head,
        token,
        d,
        value_stride_head,
        value_stride_token,
        value_stride_dim);
    denom += w;
    selected_denom += w;
    numer += w * v;
    selected_numer += w * v;
  }

  float tail_scale = expf(confidence_max_logits[head] - final_max);
  denom += tail_denoms[head] * tail_scale;
  if (value_subdim > 0) {
    int64_t sub = d / value_subdim;
    int64_t sub_d = d - sub * value_subdim;
    if (sub >= 0 && sub < value_subvecs) {
      for (int64_t page = 0; page < pages; ++page) {
        for (int64_t code = 0; code < value_centroids; ++code) {
          float w = code_weight_sums[(((head * pages + page) * value_subvecs + sub) * value_centroids + code)];
          if (w == 0.0f) {
            continue;
          }
          float value = value_codebooks
              [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
               sub_d];
          numer += (w * tail_scale) * value;
        }
      }
    }
  }

  for (int64_t sel = 0; sel < accepted; ++sel) {
    float logit = ranked_logit_row[sel];
    int64_t token = ranked_row[sel];
    if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
      continue;
    }
    float w = expf(logit - final_max);
    bool exact_value = selected_mass_exact_value_for_rank(logit, sel, threshold_logit, threshold_sel);
    float selected_v = selected_value_for_rank_dim_explicit(
        values,
        value_codebooks,
        value_codes,
        page_starts,
        token,
        exact_value,
        kv_head,
        d,
        total_tokens,
        value_stride_head,
        value_stride_token,
        value_stride_dim,
        pages,
        page_size,
        value_subvecs,
        value_centroids,
        value_subdim);
    denom += w;
    selected_denom += w;
    numer += w * selected_v;
    selected_numer += w * selected_v;

    int64_t page = -1;
    int64_t row = -1;
    if (complete_page_for_token(
            page_starts,
            pages,
            page_size,
            token,
            query_context_len,
            prefix_end,
            base_tail_start,
            &page,
            &row)) {
      float pq_w = expf(dense_row[page * page_size + row] * scale - final_max);
      float pq_v = load_vpq_value_for_dim(
          value_codebooks,
          value_codes,
          kv_head,
          page,
          row,
          d,
          pages,
          page_size,
          value_subvecs,
          value_centroids,
          value_subdim);
      denom -= pq_w;
      numer -= pq_w * pq_v;
    }
  }

  float full = numer / fmaxf(denom, 1.0e-20f);
  if (tail_blend > 0.0f && tail_blend < 1.0f) {
    float selected_only = selected_numer / fmaxf(selected_denom, 1.0e-20f);
    outputs[head * dim + d] = selected_only + tail_blend * (full - selected_only);
  } else {
    outputs[head * dim + d] = full;
  }
}

__global__ void gqa_decode_proxy_gate_counts_kernel(
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    const float* __restrict__ ranked_logits,
    const float* __restrict__ ranked_scores,
    int64_t* __restrict__ accepted_counts,
    int64_t heads,
    int64_t ranked,
    int64_t max_base,
    int64_t max_budget_arg,
    float scale,
    float proxy_mass_min,
    float proxy_tail_mass_max,
    float pq_corr_min,
    float pq_relrmse_max,
    bool calibrate) {
  int64_t head = blockIdx.x;
  if (head >= heads) {
    return;
  }
  extern __shared__ float shared[];
  const int64_t max_budget = max_budget_arg <= 0 ? ranked : min(max_budget_arg, ranked);
  if (max_budget <= 0) {
    if (threadIdx.x == 0) {
      accepted_counts[head] = 0;
    }
    return;
  }
  int64_t keep = accepted_counts[head];
  keep = max(static_cast<int64_t>(0), min(keep, max_budget));
  const float* base_row = base_logits + head * max_base;
  const int64_t base_count = max(static_cast<int64_t>(0), base_counts[head]);
  const float* exact_row = ranked_logits + head * ranked;
  const float* score_row = ranked_scores + head * ranked;

  float local_x = 0.0f;
  float local_y = 0.0f;
  float local_x2 = 0.0f;
  float local_y2 = 0.0f;
  float local_xy = 0.0f;
  float local_count = 0.0f;
  float local_base_max = -INFINITY;
  float local_selected_max = -INFINITY;
  for (int64_t idx = threadIdx.x; idx < base_count; idx += blockDim.x) {
    local_base_max = fmaxf(local_base_max, base_row[idx]);
  }
  for (int64_t sel = threadIdx.x; sel < max_budget; sel += blockDim.x) {
    float exact = exact_row[sel];
    float score = score_row[sel];
    if (sel < keep && isfinite(exact) && isfinite(score)) {
      float x = score * scale;
      float y = exact;
      local_x += x;
      local_y += y;
      local_x2 += x * x;
      local_y2 += y * y;
      local_xy += x * y;
      local_count += 1.0f;
      local_selected_max = fmaxf(local_selected_max, y);
    }
  }

  shared[threadIdx.x] = local_x;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  __shared__ float sx, sy, sx2, sy2, sxy, scount, base_max, selected_max;
  if (threadIdx.x == 0) sx = shared[0];
  __syncthreads();

  shared[threadIdx.x] = local_y;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) sy = shared[0];
  __syncthreads();

  shared[threadIdx.x] = local_x2;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) sx2 = shared[0];
  __syncthreads();

  shared[threadIdx.x] = local_y2;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) sy2 = shared[0];
  __syncthreads();

  shared[threadIdx.x] = local_xy;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) sxy = shared[0];
  __syncthreads();

  shared[threadIdx.x] = local_count;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) scount = shared[0];
  __syncthreads();

  shared[threadIdx.x] = local_base_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) base_max = shared[0];
  __syncthreads();

  shared[threadIdx.x] = local_selected_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) selected_max = shared[0];
  __syncthreads();

  __shared__ float fit_scale_shared, fit_bias_shared, corr_shared, relrmse_shared;
  if (threadIdx.x == 0) {
    float count = scount;
    float fit_scale = 1.0f;
    float fit_bias = 0.0f;
    float corr = 0.0f;
    float relrmse = INFINITY;
    if (calibrate && count >= 2.0f) {
      float inv_count = 1.0f / count;
      float mean_x = sx * inv_count;
      float mean_y = sy * inv_count;
      float var_x = fmaxf(sx2 * inv_count - mean_x * mean_x, 0.0f);
      float var_y = fmaxf(sy2 * inv_count - mean_y * mean_y, 0.0f);
      float cov = sxy * inv_count - mean_x * mean_y;
      bool flat_case = var_x <= 1.0e-20f;
      fit_scale = flat_case ? 0.0f : cov / fmaxf(var_x, 1.0e-20f);
      fit_bias = mean_y - fit_scale * mean_x;
      if (!flat_case && (!isfinite(fit_scale) || fit_scale <= 0.0f)) {
        fit_scale = 1.0f;
        fit_bias = 0.0f;
      }
      if (var_x > 1.0e-20f && var_y > 1.0e-20f) {
        corr = cov / sqrtf(var_x * var_y);
        if (!isfinite(corr)) corr = 0.0f;
      }
    }
    fit_scale_shared = fit_scale;
    fit_bias_shared = fit_bias;
    corr_shared = corr;
    relrmse_shared = relrmse;
  }
  __syncthreads();

  float fit_scale = fit_scale_shared;
  float fit_bias = fit_bias_shared;
  local_x = 0.0f;
  local_base_max = base_max;
  float local_base_sum = 0.0f;
  float local_selected_sum = 0.0f;
  float local_tail_max = -INFINITY;
  for (int64_t idx = threadIdx.x; idx < base_count; idx += blockDim.x) {
    float logit = base_row[idx];
    if (isfinite(logit) && isfinite(local_base_max)) {
      local_base_sum += expf(logit - local_base_max);
    }
  }
  for (int64_t sel = threadIdx.x; sel < max_budget; sel += blockDim.x) {
    float exact = exact_row[sel];
    float score = score_row[sel];
    if (!isfinite(score)) {
      continue;
    }
    if (sel < keep && isfinite(exact) && isfinite(selected_max)) {
      local_selected_sum += expf(exact - selected_max);
      float pred = fit_scale * (score * scale) + fit_bias;
      float err = pred - exact;
      local_x += err * err;
    } else if (sel >= keep) {
      local_tail_max = fmaxf(local_tail_max, fit_scale * (score * scale) + fit_bias);
    }
  }

  shared[threadIdx.x] = local_x;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0 && calibrate && scount >= 2.0f) {
    float mean_y = sy / fmaxf(scount, 1.0f);
    float var_y = fmaxf(sy2 / fmaxf(scount, 1.0f) - mean_y * mean_y, 0.0f);
    float rmse = sqrtf(shared[0] / fmaxf(scount, 1.0f));
    relrmse_shared = rmse / fmaxf(sqrtf(var_y), 1.0e-6f);
  }
  __syncthreads();

  shared[threadIdx.x] = local_base_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  __shared__ float base_lse, selected_lse_part, tail_max;
  if (threadIdx.x == 0) base_lse = (isfinite(base_max) && shared[0] > 0.0f) ? base_max + logf(shared[0]) : -INFINITY;
  __syncthreads();

  shared[threadIdx.x] = local_selected_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    selected_lse_part = (isfinite(selected_max) && shared[0] > 0.0f) ? selected_max + logf(shared[0]) : -INFINITY;
  }
  __syncthreads();

  shared[threadIdx.x] = local_tail_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) tail_max = shared[0];
  __syncthreads();

  float local_tail_sum = 0.0f;
  for (int64_t sel = threadIdx.x; sel < max_budget; sel += blockDim.x) {
    float score = score_row[sel];
    if (sel >= keep && isfinite(score) && isfinite(tail_max)) {
      float pred = fit_scale * (score * scale) + fit_bias;
      local_tail_sum += expf(pred - tail_max);
    }
  }
  shared[threadIdx.x] = local_tail_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float selected_lse = logaddexp_device(base_lse, selected_lse_part);
    float tail_lse = (isfinite(tail_max) && shared[0] > 0.0f) ? tail_max + logf(shared[0]) : -INFINITY;
    float total_lse = logaddexp_device(selected_lse, tail_lse);
    float selected_mass = isfinite(total_lse) ? expf(selected_lse - total_lse) : 0.0f;
    float tail_mass = isfinite(total_lse) ? expf(tail_lse - total_lse) : 0.0f;
    bool gate =
        selected_mass >= proxy_mass_min &&
        tail_mass <= proxy_tail_mass_max &&
        corr_shared >= pq_corr_min &&
        relrmse_shared <= pq_relrmse_max;
    if (!gate) {
      accepted_counts[head] = max_budget;
    }
  }
}

template <typename value_t, typename vcode_t>
__device__ float gqa_causal_geometric_output_dim_bitmap(
    const value_t* __restrict__ values,
    const float* __restrict__ dense_row,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int32_t* __restrict__ base_tokens,
    const float* __restrict__ base_logits,
    int64_t base_count,
    const int64_t* __restrict__ ranked_tokens_row,
    const float* __restrict__ ranked_logits,
    const unsigned int* __restrict__ selected_bitmap,
    int64_t head,
    int64_t dim_idx,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t value_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t keep,
    float scale,
    bool include_tail) {
  (void)dim;
  int64_t kv_head = head / group_size;
  if (kv_head >= kv_heads) {
    kv_head = kv_heads - 1;
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

  int64_t keep_clamped = keep;
  if (keep_clamped < 0) {
    keep_clamped = 0;
  }
  if (keep_clamped > ranked) {
    keep_clamped = ranked;
  }

  float max_logit = -INFINITY;
  for (int64_t idx = 0; idx < base_count; ++idx) {
    max_logit = fmaxf(max_logit, base_logits[idx]);
  }
  for (int64_t sel = 0; sel < keep_clamped; ++sel) {
    max_logit = fmaxf(max_logit, ranked_logits[sel]);
  }
  if (include_tail && pages > 0) {
    for (int64_t page = 0; page < pages; ++page) {
      int64_t page_start = page_starts[page];
      bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
      if (!valid_page) {
        continue;
      }
      for (int64_t row = 0; row < page_size; ++row) {
        int64_t token = page_start + row;
        if (token < 0 || token >= query_context_len || token >= total_tokens) {
          continue;
        }
        int64_t ordinal = page * page_size + row;
        bool selected = (selected_bitmap[ordinal >> 5] & (1u << (ordinal & 31))) != 0u;
        if (selected) {
          continue;
        }
        max_logit = fmaxf(max_logit, dense_row[ordinal] * scale);
      }
    }
  }
  if (!isfinite(max_logit)) {
    return 0.0f;
  }

  float denom = 0.0f;
  float accum = 0.0f;
  for (int64_t idx = 0; idx < base_count; ++idx) {
    int64_t token = static_cast<int64_t>(base_tokens[idx]);
    if (token < 0 || token >= total_tokens) {
      continue;
    }
    float weight = expf(base_logits[idx] - max_logit);
    denom += weight;
    accum += weight * load_strided3_as_float(
                          values,
                          kv_head,
                          token,
                          dim_idx,
                          value_stride_head,
                          value_stride_token,
                          value_stride_dim);
  }
  for (int64_t sel = 0; sel < keep_clamped; ++sel) {
    int64_t token = ranked_tokens_row[sel];
    float logit = ranked_logits[sel];
    if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
      continue;
    }
    float weight = expf(logit - max_logit);
    denom += weight;
    accum += weight * load_strided3_as_float(
                          values,
                          kv_head,
                          token,
                          dim_idx,
                          value_stride_head,
                          value_stride_token,
                          value_stride_dim);
  }
  if (include_tail && pages > 0 && value_subvecs > 0 && value_subdim > 0) {
    int64_t sub = dim_idx / value_subdim;
    int64_t sub_d = dim_idx - sub * value_subdim;
    if (sub >= 0 && sub < value_subvecs) {
      for (int64_t page = 0; page < pages; ++page) {
        int64_t page_start = page_starts[page];
        bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
        if (!valid_page) {
          continue;
        }
        for (int64_t row = 0; row < page_size; ++row) {
          int64_t token = page_start + row;
          if (token < 0 || token >= query_context_len || token >= total_tokens) {
            continue;
          }
          int64_t ordinal = page * page_size + row;
          bool selected = (selected_bitmap[ordinal >> 5] & (1u << (ordinal & 31))) != 0u;
          if (selected) {
            continue;
          }
          int64_t vcode = static_cast<int64_t>(
              value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
          if (vcode < 0) {
            vcode = 0;
          }
          if (vcode >= value_centroids) {
            vcode = value_centroids - 1;
          }
          float value = value_codebooks
              [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + vcode) * value_subdim) +
               sub_d];
          float weight = expf(dense_row[ordinal] * scale - max_logit);
          denom += weight;
          accum += weight * value;
        }
      }
    }
  }
  if (denom <= 0.0f || !isfinite(denom)) {
    return 0.0f;
  }
  return accum / denom;
}

template <typename key_t, typename value_t, typename vcode_t>
__global__ void gqa_causal_geometric_accept_counts_kernel(
    const float* __restrict__ queries,
    const key_t* __restrict__ keys,
    const value_t* __restrict__ values,
    const float* __restrict__ dense_pq_scores,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    int64_t* __restrict__ accepted_counts,
    int64_t positions,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t key_stride_head,
    int64_t key_stride_token,
    int64_t key_stride_dim,
    int64_t value_stride_head,
    int64_t value_stride_token,
    int64_t value_stride_dim,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t max_base,
    int64_t min_budget,
    int64_t max_budget_arg,
    int64_t granularity,
    float growth,
    float probe_scale,
    float rel_l2_max,
    int64_t exact_value_top,
    float scale) {
  int64_t qh = static_cast<int64_t>(blockIdx.x);
  int64_t total_qh = positions * heads;
  if (qh >= total_qh) {
    return;
  }
  int64_t pos = qh / heads;
  int64_t head = qh - pos * heads;
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  int64_t query_context_len = min(query_start + pos + 1, total_tokens);
  const float* q = queries + qh * dim;
  const float* dense_row = dense_pq_scores + qh * (pages * page_size);
  const int64_t* ranked_row = ranked_tokens + qh * ranked;
  const float* ranked_score_row = ranked_scores + qh * ranked;

  const int64_t flat_tokens = pages * page_size;
  extern __shared__ unsigned char shared_raw[];
  float* diff_sums = reinterpret_cast<float*>(shared_raw);
  float* probe_sums = diff_sums + blockDim.x;
  float* base_logits = probe_sums + blockDim.x;
  float* ranked_logits = base_logits + max_base;
  int32_t* base_tokens = reinterpret_cast<int32_t*>(ranked_logits + ranked);

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

  __shared__ int32_t base_count_shared;
  if (threadIdx.x == 0) {
    int32_t count = 0;
    for (int64_t token = 0; token < prefix_end && count < max_base; ++token) {
      base_tokens[count++] = static_cast<int32_t>(token);
    }
    for (int64_t token = base_tail_start; token < query_context_len && count < max_base; ++token) {
      base_tokens[count++] = static_cast<int32_t>(token);
    }
    base_count_shared = count;
  }
  for (int64_t idx = threadIdx.x; idx < max_base; idx += blockDim.x) {
    base_logits[idx] = -INFINITY;
  }
  for (int64_t sel = threadIdx.x; sel < ranked; sel += blockDim.x) {
    ranked_logits[sel] = -INFINITY;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  int warps = blockDim.x >> 5;
  if (warps <= 0) {
    warps = 1;
  }
  int64_t base_count = static_cast<int64_t>(base_count_shared);
  for (int64_t idx = warp_id; idx < base_count; idx += warps) {
    int64_t token = static_cast<int64_t>(base_tokens[idx]);
    float partial = 0.0f;
    for (int64_t d = lane; d < dim; d += 32) {
      partial += q[d] *
          load_strided3_as_float(keys, kv_head, token, d, key_stride_head, key_stride_token, key_stride_dim);
    }
    float dot = warp_reduce_sum(partial);
    if (lane == 0) {
      base_logits[idx] = dot * scale;
    }
  }
  for (int64_t sel = warp_id; sel < ranked; sel += warps) {
    int64_t token = ranked_row[sel];
    float selector_score = ranked_score_row[sel];
    if (!isfinite(selector_score) || token < 0 || token >= query_context_len ||
        token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr)) {
      continue;
    }
    float partial = 0.0f;
    for (int64_t d = lane; d < dim; d += 32) {
      partial += q[d] *
          load_strided3_as_float(keys, kv_head, token, d, key_stride_head, key_stride_token, key_stride_dim);
    }
    float dot = warp_reduce_sum(partial);
    if (lane == 0) {
      ranked_logits[sel] = dot * scale;
    }
  }
  __syncthreads();

  int64_t max_budget = max_budget_arg;
  if (max_budget <= 0 || max_budget > ranked) {
    max_budget = ranked;
  }
  int64_t start_budget = min_budget;
  if (start_budget < 0) {
    start_budget = 0;
  }
  if (start_budget > max_budget) {
    start_budget = max_budget;
  }
  start_budget = round_budget_up_device(start_budget, granularity, max_budget);
  __shared__ int64_t accepted_shared;
  __shared__ int done_shared;
  __shared__ float global_max_shared;
  if (threadIdx.x == 0) {
    accepted_shared = max_budget;
    done_shared = 0;
  }
  __syncthreads();

  float local_global_max = -INFINITY;
  for (int64_t idx = threadIdx.x; idx < base_count; idx += blockDim.x) {
    local_global_max = fmaxf(local_global_max, base_logits[idx]);
  }
  for (int64_t sel = threadIdx.x; sel < max_budget; sel += blockDim.x) {
    local_global_max = fmaxf(local_global_max, ranked_logits[sel]);
  }
  for (int64_t ordinal = threadIdx.x; ordinal < flat_tokens; ordinal += blockDim.x) {
    int64_t page = ordinal / page_size;
    int64_t row = ordinal - page * page_size;
    int64_t page_start = page_starts[page];
    bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
    int64_t token = page_start + row;
    if (valid_page && token >= 0 && token < query_context_len && token < total_tokens) {
      local_global_max = fmaxf(local_global_max, dense_row[ordinal] * scale);
    }
  }
  diff_sums[threadIdx.x] = local_global_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      diff_sums[threadIdx.x] = fmaxf(diff_sums[threadIdx.x], diff_sums[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    global_max_shared = diff_sums[0];
  }
  __syncthreads();

  float global_max = global_max_shared;
  if (!isfinite(global_max)) {
    if (threadIdx.x == 0) {
      accepted_counts[qh] = max_budget;
    }
    return;
  }

  int64_t d = threadIdx.x;
  bool active_dim = d < dim;
  float base_den = 0.0f;
  float base_num = 0.0f;
  float tail_den = 0.0f;
  float tail_num = 0.0f;
  if (active_dim) {
    for (int64_t idx = 0; idx < base_count; ++idx) {
      int64_t token = static_cast<int64_t>(base_tokens[idx]);
      if (token < 0 || token >= total_tokens) {
        continue;
      }
      float logit = base_logits[idx];
      if (!isfinite(logit)) {
        continue;
      }
      float w = expf(logit - global_max);
      base_den += w;
      base_num += w * load_strided3_as_float(
                          values,
                          kv_head,
                          token,
                          d,
                          value_stride_head,
                          value_stride_token,
                          value_stride_dim);
    }
    for (int64_t ordinal = 0; ordinal < flat_tokens; ++ordinal) {
      int64_t page = ordinal / page_size;
      int64_t row = ordinal - page * page_size;
      int64_t page_start = page_starts[page];
      bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
      int64_t token = page_start + row;
      if (!valid_page || token < 0 || token >= query_context_len || token >= total_tokens) {
        continue;
      }
      float w = expf(dense_row[ordinal] * scale - global_max);
      tail_den += w;
      tail_num += w * load_vpq_value_for_dim(
                        value_codebooks,
                        value_codes,
                        kv_head,
                        page,
                        row,
                        d,
                        pages,
                        page_size,
                        value_subvecs,
                        value_centroids,
                        value_subdim);
    }
  }

  float approx_den = base_den + tail_den;
  float approx_num = base_num + tail_num;
  float probe_den = base_den;
  float probe_num = base_num;
  int64_t approx_keep = 0;
  int64_t probe_keep = 0;
  int64_t tail_budget = start_budget;
  while (tail_budget < max_budget) {
    float target = fmaxf(static_cast<float>(tail_budget + granularity), probe_scale * static_cast<float>(tail_budget));
    int64_t probe_budget = round_budget_up_device(static_cast<int64_t>(ceilf(target)), granularity, max_budget);
    if (probe_budget < tail_budget) {
      probe_budget = tail_budget;
    }

    float local_diff = 0.0f;
    float local_probe = 0.0f;
    if (active_dim) {
      for (int64_t sel = approx_keep; sel < tail_budget; ++sel) {
        float logit = ranked_logits[sel];
        int64_t token = ranked_row[sel];
        if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
          continue;
        }
        float selected_w = expf(logit - global_max);
        float selected_v = selected_value_for_rank_dim(
            values,
            value_codebooks,
            value_codes,
            page_starts,
            token,
            sel,
            exact_value_top,
            max_budget,
            kv_head,
            d,
            total_tokens,
            value_stride_head,
            value_stride_token,
            value_stride_dim,
            pages,
            page_size,
            value_subvecs,
            value_centroids,
            value_subdim);
        approx_den += selected_w;
        approx_num += selected_w * selected_v;
        int64_t page = -1;
        int64_t row = -1;
        if (complete_page_for_token(
                page_starts,
                pages,
                page_size,
                token,
                query_context_len,
                prefix_end,
                base_tail_start,
                &page,
                &row)) {
          float pq_w = expf(dense_row[page * page_size + row] * scale - global_max);
          float pq_v = load_vpq_value_for_dim(
              value_codebooks,
              value_codes,
              kv_head,
              page,
              row,
              d,
              pages,
              page_size,
              value_subvecs,
              value_centroids,
              value_subdim);
          approx_den -= pq_w;
          approx_num -= pq_w * pq_v;
        }
      }
      approx_keep = tail_budget;
      for (int64_t sel = probe_keep; sel < probe_budget; ++sel) {
        float logit = ranked_logits[sel];
        int64_t token = ranked_row[sel];
        if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
          continue;
        }
        float selected_w = expf(logit - global_max);
        float selected_v = selected_value_for_rank_dim(
            values,
            value_codebooks,
            value_codes,
            page_starts,
            token,
            sel,
            exact_value_top,
            max_budget,
            kv_head,
            d,
            total_tokens,
            value_stride_head,
            value_stride_token,
            value_stride_dim,
            pages,
            page_size,
            value_subvecs,
            value_centroids,
            value_subdim);
        probe_den += selected_w;
        probe_num += selected_w * selected_v;
      }
      probe_keep = probe_budget;
      float approx_tail = approx_num / fmaxf(approx_den, 1.0e-20f);
      float probe_only = probe_num / fmaxf(probe_den, 1.0e-20f);
      float delta = approx_tail - probe_only;
      local_diff += delta * delta;
      local_probe += probe_only * probe_only;
    }
    diff_sums[threadIdx.x] = local_diff;
    probe_sums[threadIdx.x] = local_probe;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        diff_sums[threadIdx.x] += diff_sums[threadIdx.x + stride];
        probe_sums[threadIdx.x] += probe_sums[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      float denom = sqrtf(fmaxf(probe_sums[0], 1.0e-40f));
      float rel = sqrtf(fmaxf(diff_sums[0], 0.0f)) / denom;
      if (rel <= rel_l2_max) {
        accepted_shared = probe_budget;
        done_shared = 1;
      }
    }
    __syncthreads();
    if (done_shared != 0 || probe_budget >= max_budget) {
      break;
    }
    float next_target = fmaxf(static_cast<float>(probe_budget + granularity), growth * static_cast<float>(probe_budget));
    int64_t next_budget = round_budget_up_device(static_cast<int64_t>(ceilf(next_target)), granularity, max_budget);
    if (next_budget <= probe_budget) {
      break;
    }
    tail_budget = next_budget;
  }
  if (threadIdx.x == 0) {
    accepted_counts[qh] = accepted_shared;
  }
}

}  // namespace

std::vector<torch::Tensor> fullscan_pq_topk_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t budget) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto pages = codebooks.size(0);
  const auto subvecs = codebooks.size(1);
  const auto centroids = codebooks.size(2);
  const auto subdim = codebooks.size(3);
  const auto page_size = codes.size(1);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({heads, k}, long_opts);
  auto top_scores = torch::empty({heads, k}, score_opts);
  if (heads == 0 || pages == 0 || k == 0) {
    return {top_tokens, top_scores};
  }

  auto scores = torch::empty({heads, total_tokens}, score_opts);
  const int threads = 128;
  const int64_t score_elems = heads * total_tokens;
  const int blocks = static_cast<int>((score_elems + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (codes.scalar_type() == torch::kUInt8) {
    pq_scores_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        scores.data_ptr<float>(),
        heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens);
  } else {
    pq_scores_kernel<int64_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  top_scores = std::get<0>(topk);
  auto top_indices = std::get<1>(topk).contiguous();

  const int64_t map_elems = heads * k;
  const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
  map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
      top_indices.data_ptr<int64_t>(),
      page_starts.data_ptr<int64_t>(),
      top_tokens.data_ptr<int64_t>(),
      heads,
      k,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {top_tokens, top_scores};
}

std::vector<torch::Tensor> fullscan_pq_topk_scores_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t budget) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto pages = codebooks.size(0);
  const auto subvecs = codebooks.size(1);
  const auto centroids = codebooks.size(2);
  const auto subdim = codebooks.size(3);
  const auto page_size = codes.size(1);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({heads, k}, long_opts);
  auto top_scores = torch::empty({heads, k}, score_opts);
  auto scores = torch::empty({heads, total_tokens}, score_opts);
  if (heads == 0 || pages == 0) {
    return {top_tokens, top_scores, scores};
  }

  const int threads = 256;
  const int64_t score_elems = heads * total_tokens;
  const int blocks = static_cast<int>((score_elems + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (codes.scalar_type() == torch::kUInt8) {
    pq_scores_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        scores.data_ptr<float>(),
        heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens);
  } else {
    pq_scores_kernel<int64_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (k == 0) {
    return {top_tokens, top_scores, scores};
  }

  auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  top_scores = std::get<0>(topk);
  auto top_indices = std::get<1>(topk).contiguous();

  const int64_t map_elems = heads * k;
  const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
  map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
      top_indices.data_ptr<int64_t>(),
      page_starts.data_ptr<int64_t>(),
      top_tokens.data_ptr<int64_t>(),
      heads,
      k,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {top_tokens, top_scores, scores};
}

std::vector<torch::Tensor> gqa_fullscan_pq_topk_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({heads, k}, long_opts);
  auto top_scores = torch::empty({heads, k}, score_opts);
  if (heads == 0 || pages == 0 || k == 0) {
    return {top_tokens, top_scores};
  }

  auto scores = torch::empty({heads, total_tokens}, score_opts);
  const int threads = 256;
  const int64_t score_elems = heads * total_tokens;
  const int blocks = static_cast<int>((score_elems + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (codes.scalar_type() == torch::kUInt8) {
    gqa_pq_scores_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        scores.data_ptr<float>(),
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size);
  } else {
    gqa_pq_scores_kernel<int64_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  top_scores = std::get<0>(topk);
  auto top_indices = std::get<1>(topk).contiguous();

  const int64_t map_elems = heads * k;
  const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
  map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
      top_indices.data_ptr<int64_t>(),
      page_starts.data_ptr<int64_t>(),
      top_tokens.data_ptr<int64_t>(),
      heads,
      k,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {top_tokens, top_scores};
}

std::vector<torch::Tensor> gqa_fullscan_pq_topk_scores_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({heads, k}, long_opts);
  auto top_scores = torch::empty({heads, k}, score_opts);
  auto scores = torch::empty({heads, total_tokens}, score_opts);
  if (heads == 0 || pages == 0) {
    return {top_tokens, top_scores, scores};
  }

  const int threads = 256;
  const int64_t score_elems = heads * total_tokens;
  const int blocks = static_cast<int>((score_elems + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (codes.scalar_type() == torch::kUInt8) {
    gqa_pq_scores_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        scores.data_ptr<float>(),
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size);
  } else {
    gqa_pq_scores_kernel<int64_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (k == 0) {
    return {top_tokens, top_scores, scores};
  }

  auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  top_scores = std::get<0>(topk);
  auto top_indices = std::get<1>(topk).contiguous();

  const int64_t map_elems = heads * k;
  const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
  map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
      top_indices.data_ptr<int64_t>(),
      page_starts.data_ptr<int64_t>(),
      top_tokens.data_ptr<int64_t>(),
      heads,
      k,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {top_tokens, top_scores, scores};
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({positions, heads, k}, long_opts);
  auto top_scores = torch::empty({positions, heads, k}, score_opts);
  if (positions == 0 || heads == 0 || pages == 0 || k == 0) {
    return {top_tokens, top_scores};
  }

  auto scores = torch::empty({positions * heads, total_tokens}, score_opts);
  const int threads = 256;
  const dim3 blocks(static_cast<unsigned int>(positions * heads), static_cast<unsigned int>(pages), 1);
  const size_t lut_smem = static_cast<size_t>(subvecs * centroids) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (codes.scalar_type() == torch::kUInt8) {
    gqa_causal_pq_scores_lut_kernel<uint8_t><<<blocks, threads, lut_smem, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        page_starts.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size,
        query_start,
        static_prefix,
        static_suffix);
  } else {
    gqa_causal_pq_scores_lut_kernel<int64_t><<<blocks, threads, lut_smem, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size,
        query_start,
        static_prefix,
        static_suffix);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  top_scores = std::get<0>(topk).reshape({positions, heads, k}).contiguous();
  auto top_indices = std::get<1>(topk).contiguous();

  const int64_t map_elems = positions * heads * k;
  const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
  gqa_causal_map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
      top_indices.data_ptr<int64_t>(),
      page_starts.data_ptr<int64_t>(),
      top_tokens.data_ptr<int64_t>(),
      positions,
      heads,
      k,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {top_tokens, top_scores};
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_scores_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({positions, heads, k}, long_opts);
  auto top_scores = torch::empty({positions, heads, k}, score_opts);
  auto scores = torch::empty({positions * heads, total_tokens}, score_opts);
  if (positions == 0 || heads == 0 || pages == 0) {
    return {top_tokens, top_scores, scores.reshape({positions, heads, total_tokens}).contiguous()};
  }

  const int threads = 256;
  const dim3 blocks(static_cast<unsigned int>(positions * heads), static_cast<unsigned int>(pages), 1);
  const size_t lut_smem = static_cast<size_t>(subvecs * centroids) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (codes.scalar_type() == torch::kUInt8) {
    gqa_causal_pq_scores_lut_kernel<uint8_t><<<blocks, threads, lut_smem, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        page_starts.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size,
        query_start,
        static_prefix,
        static_suffix);
  } else {
    gqa_causal_pq_scores_lut_kernel<int64_t><<<blocks, threads, lut_smem, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size,
        query_start,
        static_prefix,
        static_suffix);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (k > 0) {
    auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
    top_scores = std::get<0>(topk).reshape({positions, heads, k}).contiguous();
    auto top_indices = std::get<1>(topk).contiguous();

    const int64_t map_elems = positions * heads * k;
    const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
    gqa_causal_map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
        top_indices.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        top_tokens.data_ptr<int64_t>(),
        positions,
        heads,
        k,
        page_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {top_tokens, top_scores, scores.reshape({positions, heads, total_tokens}).contiguous()};
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_fused_force_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t force_mode) {
  TORCH_CHECK(force_mode >= 0 && force_mode <= 2, "force_mode must be 0=auto, 1=smallscan, or 2=localtopk");
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({positions, heads, k}, long_opts);
  auto top_scores = torch::empty({positions, heads, k}, score_opts);
  if (positions == 0 || heads == 0 || pages == 0 || k == 0) {
    return {top_tokens, top_scores};
  }
  if (k > 64 || total_tokens > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return gqa_causal_fullscan_pq_topk_cuda(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        budget,
        query_start,
        static_prefix,
        static_suffix);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t rows = positions * heads;
  const int blocks = static_cast<int>(rows);
  if (total_tokens <= 4096 && (force_mode == 1 || (force_mode == 0 && page_size >= 1024))) {
    const int threads = 256;
    size_t shared_bytes =
        static_cast<size_t>(subvecs * centroids) * sizeof(float)
        + static_cast<size_t>(total_tokens) * sizeof(float)
        + static_cast<size_t>(total_tokens) * sizeof(int32_t)
        + static_cast<size_t>(threads) * sizeof(float)
        + static_cast<size_t>(threads) * sizeof(int32_t);
    if (codes.scalar_type() == torch::kUInt8) {
      gqa_causal_pq_topk_fused_smallscan_kernel<uint8_t><<<blocks, threads, shared_bytes, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<uint8_t>(),
          page_starts.data_ptr<int64_t>(),
          top_tokens.data_ptr<int64_t>(),
          top_scores.data_ptr<float>(),
          positions,
          heads,
          kv_heads,
          dim,
          pages,
          page_size,
          subvecs,
          centroids,
          subdim,
          group_size,
          k,
          query_start,
          static_prefix,
          static_suffix);
    } else {
      gqa_causal_pq_topk_fused_smallscan_kernel<int64_t><<<blocks, threads, shared_bytes, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<int64_t>(),
          page_starts.data_ptr<int64_t>(),
          top_tokens.data_ptr<int64_t>(),
          top_scores.data_ptr<float>(),
          positions,
          heads,
          kv_heads,
          dim,
          pages,
          page_size,
          subvecs,
          centroids,
          subdim,
          group_size,
          k,
          query_start,
          static_prefix,
          static_suffix);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {top_tokens, top_scores};
  }
  if (force_mode == 0 && k > 16) {
    return gqa_causal_fullscan_pq_topk_cuda(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        budget,
        query_start,
        static_prefix,
        static_suffix);
  }
  if (force_mode == 1) {
    return gqa_causal_fullscan_pq_topk_cuda(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        budget,
        query_start,
        static_prefix,
        static_suffix);
  }

  auto launch = [&](auto code_ptr, auto k_tag, int threads) {
    constexpr int KMAX = decltype(k_tag)::value;
    size_t shared_bytes =
        static_cast<size_t>(subvecs * centroids) * sizeof(float)
        + static_cast<size_t>(threads * KMAX) * sizeof(float)
        + static_cast<size_t>(threads * KMAX) * sizeof(int32_t)
        + static_cast<size_t>(threads) * sizeof(float)
        + static_cast<size_t>(threads) * sizeof(int32_t);
    gqa_causal_pq_topk_fused_kernel<std::remove_pointer_t<decltype(code_ptr)>, KMAX><<<
        blocks,
        threads,
        shared_bytes,
        stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        code_ptr,
        page_starts.data_ptr<int64_t>(),
        top_tokens.data_ptr<int64_t>(),
        top_scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        group_size,
        k,
        query_start,
        static_prefix,
        static_suffix);
  };

  if (codes.scalar_type() == torch::kUInt8) {
    if (k <= 8) {
      launch(codes.data_ptr<uint8_t>(), std::integral_constant<int, 8>{}, 256);
    } else if (k <= 16) {
      launch(codes.data_ptr<uint8_t>(), std::integral_constant<int, 16>{}, 256);
    } else if (k <= 32) {
      launch(codes.data_ptr<uint8_t>(), std::integral_constant<int, 32>{}, 128);
    } else {
      launch(codes.data_ptr<uint8_t>(), std::integral_constant<int, 64>{}, 64);
    }
  } else {
    if (k <= 8) {
      launch(codes.data_ptr<int64_t>(), std::integral_constant<int, 8>{}, 256);
    } else if (k <= 16) {
      launch(codes.data_ptr<int64_t>(), std::integral_constant<int, 16>{}, 256);
    } else if (k <= 32) {
      launch(codes.data_ptr<int64_t>(), std::integral_constant<int, 32>{}, 128);
    } else {
      launch(codes.data_ptr<int64_t>(), std::integral_constant<int, 64>{}, 64);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {top_tokens, top_scores};
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_fused_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  return gqa_causal_fullscan_pq_topk_fused_force_cuda(
      queries,
      codebooks,
      codes,
      page_starts,
      group_size,
      budget,
      query_start,
      static_prefix,
      static_suffix,
      0);
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_top_pages_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t page_budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(queries.dim() == 3, "queries must have shape [positions, heads, dim]");
  TORCH_CHECK(codebooks.dim() == 5, "codebooks must have shape [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 4, "codes must have shape [kv_heads, pages, page_size, subvecs]");
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  TORCH_CHECK(codes.size(0) == kv_heads, "codes kv_heads mismatch");
  TORCH_CHECK(codes.size(1) == pages, "codes pages mismatch");
  TORCH_CHECK(codes.size(3) == subvecs, "codes subvecs mismatch");
  TORCH_CHECK(page_starts.numel() == pages, "page_starts pages mismatch");
  TORCH_CHECK(subvecs * subdim == dim, "subvecs * subdim must equal dim");
  TORCH_CHECK(group_size > 0, "group_size must be positive");

  int64_t k = std::max<int64_t>(0, std::min<int64_t>(page_budget, pages));
  auto long_opts = queries.options().dtype(torch::kLong);
  auto float_opts = queries.options().dtype(torch::kFloat32);
  auto top_pages = torch::empty({positions, heads, k}, long_opts);
  auto top_scores = torch::empty({positions, heads, k}, float_opts);
  if (positions == 0 || heads == 0 || k == 0 || pages == 0) {
    return {top_pages, top_scores};
  }

  const int threads = 256;
  const int64_t blocks = positions * heads;
  const size_t shared_bytes =
      static_cast<size_t>(subvecs * centroids) * sizeof(float) +
      static_cast<size_t>(pages) * sizeof(float) +
      static_cast<size_t>(pages) * sizeof(int32_t) +
      static_cast<size_t>(threads) * sizeof(float) +
      static_cast<size_t>(threads) * sizeof(int32_t) * 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (codes.scalar_type() == torch::kUInt8) {
    gqa_causal_pq_top_pages_kernel<uint8_t><<<static_cast<int>(blocks), threads, shared_bytes, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        page_starts.data_ptr<int64_t>(),
        top_pages.data_ptr<int64_t>(),
        top_scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        group_size,
        k,
        query_start,
        static_prefix,
        static_suffix);
  } else {
    TORCH_CHECK(codes.scalar_type() == torch::kLong, "codes must be uint8 or int64");
    gqa_causal_pq_top_pages_kernel<int64_t><<<static_cast<int>(blocks), threads, shared_bytes, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        top_pages.data_ptr<int64_t>(),
        top_scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        group_size,
        k,
        query_start,
        static_prefix,
        static_suffix);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {top_pages, top_scores};
}

torch::Tensor exact_selected_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor tokens,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto total_tokens = keys.size(0);
  const auto selected = tokens.size(1);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({heads, dim}, opts);
  if (heads == 0 || dim == 0) {
    return outputs;
  }
  if (selected == 0 || total_tokens == 0) {
    outputs.zero_();
    return outputs;
  }

  auto logits = torch::empty({heads, selected}, opts);
  const int threads = 128;
  const int64_t pairs = heads * selected;
  auto stream = at::cuda::getCurrentCUDAStream();
  exact_selected_logits_kernel<<<static_cast<int>(pairs), threads, threads * sizeof(float), stream>>>(
      queries.data_ptr<float>(),
      keys.data_ptr<float>(),
      tokens.data_ptr<int64_t>(),
      logits.data_ptr<float>(),
      heads,
      selected,
      dim,
      total_tokens,
      static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  exact_selected_output_kernel<<<static_cast<int>(heads), threads, 0, stream>>>(
      values.data_ptr<float>(),
      tokens.data_ptr<int64_t>(),
      logits.data_ptr<float>(),
      outputs.data_ptr<float>(),
      heads,
      selected,
      dim,
      total_tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_causal_exact_selected_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto selected = ranked_tokens.size(2);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({positions, heads, dim}, opts);
  if (positions == 0 || heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }
  const int64_t max_selected = std::min<int64_t>(
      total_tokens,
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
          std::max<int64_t>(1, page_size) + selected);
  if (max_selected == 0) {
    outputs.zero_();
    return outputs;
  }

  auto long_opts = queries.options().dtype(torch::kLong);
  auto selected_tokens = torch::empty({positions, heads, max_selected}, long_opts);
  auto selected_counts = torch::empty({positions, heads}, long_opts);
  const int build_threads = 128;
  const int64_t qh_pairs = positions * heads;
  const int build_blocks = static_cast<int>((qh_pairs + build_threads - 1) / build_threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gqa_causal_build_selected_tokens_kernel<<<build_blocks, build_threads, 0, stream>>>(
      ranked_tokens.data_ptr<int64_t>(),
      ranked_scores.data_ptr<float>(),
      selected_tokens.data_ptr<int64_t>(),
      selected_counts.data_ptr<int64_t>(),
      positions,
      heads,
      selected,
      max_selected,
      total_tokens,
      query_start,
      static_prefix,
      static_suffix,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int warp_tiled_threads = 256;
  const size_t shared_bytes = static_cast<size_t>(max_selected) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, keys.scalar_type(), "gqa_causal_selected_keys", [&] {
    using key_scalar_t = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "gqa_causal_selected_values", [&] {
      gqa_causal_warp_tiled_selected_attention_kernel<key_scalar_t, scalar_t>
          <<<static_cast<int>(qh_pairs), warp_tiled_threads, shared_bytes, stream>>>(
              queries.data_ptr<float>(),
              keys.data_ptr<key_scalar_t>(),
              values.data_ptr<scalar_t>(),
              selected_tokens.data_ptr<int64_t>(),
              selected_counts.data_ptr<int64_t>(),
              outputs.data_ptr<float>(),
              positions,
              heads,
              kv_heads,
              max_selected,
              dim,
              total_tokens,
              group_size,
              static_cast<float>(scale));
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_causal_vpq_selected_attention_vpagesize_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t selection_page_size,
    int64_t value_page_size,
    int64_t exact_value_top,
    double scale) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto selected = ranked_tokens.size(2);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({positions, heads, dim}, opts);
  if (positions == 0 || heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }
  const int64_t max_selected = std::min<int64_t>(
      total_tokens,
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
          std::max<int64_t>(1, selection_page_size) + selected);
  if (max_selected == 0) {
    outputs.zero_();
    return outputs;
  }

  auto long_opts = queries.options().dtype(torch::kLong);
  auto selected_tokens = torch::empty({positions, heads, max_selected}, long_opts);
  auto selected_counts = torch::empty({positions, heads}, long_opts);
  const int build_threads = 128;
  const int64_t qh_pairs = positions * heads;
  const int build_blocks = static_cast<int>((qh_pairs + build_threads - 1) / build_threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gqa_causal_build_selected_tokens_kernel<<<build_blocks, build_threads, 0, stream>>>(
      ranked_tokens.data_ptr<int64_t>(),
      ranked_scores.data_ptr<float>(),
      selected_tokens.data_ptr<int64_t>(),
      selected_counts.data_ptr<int64_t>(),
      positions,
      heads,
      selected,
      max_selected,
      total_tokens,
      query_start,
      static_prefix,
      static_suffix,
      selection_page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int warp_tiled_threads = 256;
  const size_t shared_bytes =
      static_cast<size_t>(max_selected) * sizeof(float) +
      (exact_value_top > 0 ? static_cast<size_t>(max_selected) * sizeof(unsigned char) : 0);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, keys.scalar_type(), "gqa_causal_vpq_selected_keys", [&] {
    using key_scalar_t = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "gqa_causal_vpq_selected_values", [&] {
      using value_scalar_t = scalar_t;
      if (value_codes.scalar_type() == torch::kUInt8) {
        gqa_causal_warp_tiled_vpq_selected_attention_kernel<key_scalar_t, value_scalar_t, uint8_t>
            <<<static_cast<int>(qh_pairs), warp_tiled_threads, shared_bytes, stream>>>(
                queries.data_ptr<float>(),
                keys.data_ptr<key_scalar_t>(),
                values.data_ptr<value_scalar_t>(),
                value_codebooks.data_ptr<float>(),
                value_codes.data_ptr<uint8_t>(),
                page_starts.data_ptr<int64_t>(),
                selected_tokens.data_ptr<int64_t>(),
                selected_counts.data_ptr<int64_t>(),
                outputs.data_ptr<float>(),
                positions,
                heads,
                kv_heads,
                max_selected,
                dim,
                total_tokens,
                pages,
                value_page_size,
                value_subvecs,
                value_centroids,
                value_subdim,
                group_size,
                exact_value_top,
                static_cast<float>(scale));
      } else {
        gqa_causal_warp_tiled_vpq_selected_attention_kernel<key_scalar_t, value_scalar_t, int64_t>
            <<<static_cast<int>(qh_pairs), warp_tiled_threads, shared_bytes, stream>>>(
                queries.data_ptr<float>(),
                keys.data_ptr<key_scalar_t>(),
                values.data_ptr<value_scalar_t>(),
                value_codebooks.data_ptr<float>(),
                value_codes.data_ptr<int64_t>(),
                page_starts.data_ptr<int64_t>(),
                selected_tokens.data_ptr<int64_t>(),
                selected_counts.data_ptr<int64_t>(),
                outputs.data_ptr<float>(),
                positions,
                heads,
                kv_heads,
                max_selected,
                dim,
                total_tokens,
                pages,
                value_page_size,
                value_subvecs,
                value_centroids,
                value_subdim,
                group_size,
                exact_value_top,
                static_cast<float>(scale));
      }
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_causal_vpq_selected_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale) {
  return gqa_causal_vpq_selected_attention_vpagesize_cuda(
      queries,
      keys,
      values,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      group_size,
      query_start,
      static_prefix,
      static_suffix,
      page_size,
      page_size,
      0,
      scale);
}

torch::Tensor gqa_causal_vpq_tail_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor key_codebooks,
    torch::Tensor key_codes,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t exact_value_top,
    double scale,
    double tail_blend) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(2);
  const auto pages = key_codebooks.size(1);
  const auto key_subvecs = key_codebooks.size(2);
  const auto key_centroids = key_codebooks.size(3);
  const auto key_subdim = key_codebooks.size(4);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({positions, heads, dim}, opts);
  if (positions == 0 || heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }

  const int threads = 128;
  const int64_t qh_pairs = positions * heads;
  const int64_t max_selected =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + ranked + 4;
  const int64_t code_weight_count = pages * value_subvecs * value_centroids;
  const int64_t total_rows = pages * page_size;
  const int64_t ranked_bitmap_words = (total_rows + 31) / 32;
  const size_t shared_bytes =
      static_cast<size_t>(max_selected + code_weight_count + threads) * sizeof(float) +
      static_cast<size_t>(max_selected) * (sizeof(int32_t) + sizeof(unsigned char)) +
      static_cast<size_t>(ranked_bitmap_words) * sizeof(unsigned int) +
      sizeof(unsigned int);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_causal_vpq_tail_keys",
      [&] {
        using key_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            values.scalar_type(),
            "gqa_causal_vpq_tail_values",
            [&] {
              using value_scalar_t = scalar_t;
              if (key_codes.scalar_type() == torch::kUInt8 && value_codes.scalar_type() == torch::kUInt8) {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, uint8_t, uint8_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, uint8_t, uint8_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        key_codebooks.data_ptr<float>(),
                        key_codes.data_ptr<uint8_t>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<uint8_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
	                        positions,
	                        heads,
	                        kv_heads,
	                        ranked,
	                        dim,
	                        total_tokens,
	                        pages,
	                        page_size,
                        key_subvecs,
                        key_centroids,
                        key_subdim,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_top,
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              } else if (key_codes.scalar_type() == torch::kUInt8) {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, uint8_t, int64_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, uint8_t, int64_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        key_codebooks.data_ptr<float>(),
                        key_codes.data_ptr<uint8_t>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<int64_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        pages,
                        page_size,
                        key_subvecs,
                        key_centroids,
                        key_subdim,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_top,
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              } else if (value_codes.scalar_type() == torch::kUInt8) {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, int64_t, uint8_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, int64_t, uint8_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        key_codebooks.data_ptr<float>(),
                        key_codes.data_ptr<int64_t>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<uint8_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        pages,
                        page_size,
                        key_subvecs,
                        key_centroids,
                        key_subdim,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_top,
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              } else {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, int64_t, int64_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, int64_t, int64_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        key_codebooks.data_ptr<float>(),
                        key_codes.data_ptr<int64_t>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<int64_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        pages,
                        page_size,
                        key_subvecs,
                        key_centroids,
                        key_subdim,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_top,
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              }
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_causal_vpq_tail_from_scores_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor dense_selected_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale,
    double tail_blend) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(2);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({positions, heads, dim}, opts);
  if (positions == 0 || heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }

  const int threads = 128;
  const int64_t qh_pairs = positions * heads;
  const int64_t max_selected =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + ranked + 4;
  const int64_t code_weight_count = pages * value_subvecs * value_centroids;
  const int64_t ranked_bitmap_words = ((pages * page_size) + 31) / 32;
  const bool has_dense_selected_scores = dense_selected_scores.numel() > 0;
  if (has_dense_selected_scores) {
    TORCH_CHECK(dense_selected_scores.is_cuda(), "dense_selected_scores must be CUDA");
    TORCH_CHECK(dense_selected_scores.scalar_type() == torch::kFloat32, "dense_selected_scores must be float32");
    TORCH_CHECK(dense_selected_scores.sizes() == dense_pq_scores.sizes(), "dense_selected_scores shape mismatch");
  }
  const int64_t* exact_value_counts_ptr =
      exact_value_counts.numel() > 0 ? exact_value_counts.data_ptr<int64_t>() : nullptr;
  const float* dense_selected_ptr =
      has_dense_selected_scores ? dense_selected_scores.data_ptr<float>() : nullptr;
  const size_t shared_bytes =
      static_cast<size_t>(max_selected + code_weight_count + threads) * sizeof(float) +
      static_cast<size_t>(max_selected) * (sizeof(int32_t) + sizeof(unsigned char)) +
      static_cast<size_t>(ranked_bitmap_words) * sizeof(unsigned int) +
      alignof(unsigned int);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_causal_vpq_tail_from_scores_keys",
      [&] {
        using key_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            values.scalar_type(),
            "gqa_causal_vpq_tail_from_scores_values",
            [&] {
              using value_scalar_t = scalar_t;
              if (value_codes.scalar_type() == torch::kUInt8) {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_from_scores_kernel<key_scalar_t, value_scalar_t, uint8_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_from_scores_kernel<key_scalar_t, value_scalar_t, uint8_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        dense_pq_scores.data_ptr<float>(),
                        dense_selected_ptr,
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<uint8_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        pages,
                        page_size,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_counts_ptr,
                        exact_value_top,
                        static_cast<float>(exact_value_mass),
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              } else {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_from_scores_kernel<key_scalar_t, value_scalar_t, int64_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_from_scores_kernel<key_scalar_t, value_scalar_t, int64_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        dense_pq_scores.data_ptr<float>(),
                        dense_selected_ptr,
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<int64_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        pages,
                        page_size,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_counts_ptr,
                        exact_value_top,
                        static_cast<float>(exact_value_mass),
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              }
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_decode_vpq_tail_from_scores_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale,
    double tail_blend) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(1);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({heads, dim}, opts);
  if (heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }
  const int threads = 128;
  const int64_t max_selected =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + ranked + 4;
  const bool needs_mixed_exact_buffers =
      exact_value_counts.numel() > 0 || (exact_value_top > 0 && exact_value_top < ranked);
  const size_t mixed_bytes =
      needs_mixed_exact_buffers
          ? (static_cast<size_t>(ranked) * sizeof(float) + static_cast<size_t>(ranked) * sizeof(unsigned char))
          : 0;
  const size_t shared_bytes = static_cast<size_t>(max_selected) * sizeof(float) + mixed_bytes;
  const int64_t* exact_value_counts_ptr =
      exact_value_counts.numel() > 0 ? exact_value_counts.data_ptr<int64_t>() : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (value_codes.scalar_type() == torch::kUInt8) {
    gqa_decode_vpq_tail_from_scores_kernel<uint8_t><<<static_cast<int>(heads), threads, shared_bytes, stream>>>(
        queries.data_ptr<float>(),
        keys.data_ptr<float>(),
        values.data_ptr<float>(),
        dense_pq_scores.data_ptr<float>(),
        value_codebooks.data_ptr<float>(),
        value_codes.data_ptr<uint8_t>(),
        page_starts.data_ptr<int64_t>(),
        ranked_tokens.data_ptr<int64_t>(),
        ranked_scores.data_ptr<float>(),
        outputs.data_ptr<float>(),
        heads,
        kv_heads,
        ranked,
        dim,
        total_tokens,
        pages,
        page_size,
        value_subvecs,
        value_centroids,
        value_subdim,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        exact_value_counts_ptr,
        exact_value_top,
        static_cast<float>(scale),
        static_cast<float>(tail_blend));
  } else {
    gqa_decode_vpq_tail_from_scores_kernel<int64_t><<<static_cast<int>(heads), threads, shared_bytes, stream>>>(
        queries.data_ptr<float>(),
        keys.data_ptr<float>(),
        values.data_ptr<float>(),
        dense_pq_scores.data_ptr<float>(),
        value_codebooks.data_ptr<float>(),
        value_codes.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        ranked_tokens.data_ptr<int64_t>(),
        ranked_scores.data_ptr<float>(),
        outputs.data_ptr<float>(),
        heads,
        kv_heads,
        ranked,
        dim,
        total_tokens,
        pages,
        page_size,
        value_subvecs,
        value_centroids,
        value_subdim,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        exact_value_counts_ptr,
        exact_value_top,
        static_cast<float>(scale),
        static_cast<float>(tail_blend));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_decode_vpq_selected_tail_agg_from_scores_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale,
    double tail_blend) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(1);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({heads, dim}, opts);
  if (heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }
  if (pages == 0 || page_size <= 0) {
    outputs.zero_();
    return outputs;
  }

  const int threads = 256;
  const bool include_tail = tail_blend > 0.0;
  const int64_t flat_tokens = pages * page_size;
  const int64_t pages_per_block = decode_tail_pages_per_block();
  const int64_t tail_blocks = std::max<int64_t>(1, (pages + pages_per_block - 1) / pages_per_block);
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  auto byte_opts = queries.options().dtype(torch::kUInt8);
  auto tail_mask = torch::zeros({heads, flat_tokens}, byte_opts);
  auto long_opts = queries.options().dtype(torch::kLong);
  auto base_tokens = torch::empty({heads, max_base}, long_opts);
  auto base_logits = torch::empty({heads, max_base}, opts);
  auto base_counts = torch::zeros({heads}, long_opts);
  auto ranked_logits = torch::empty({heads, ranked}, opts);
  auto ranked_exact = torch::zeros({heads, ranked}, byte_opts);
  auto partial_max = torch::empty({heads, tail_blocks}, opts);
  auto max_logits = torch::empty({heads}, opts);
  auto partial_sum = torch::empty({heads, tail_blocks}, opts);
  auto denoms = torch::empty({heads}, opts);
  auto selected_denoms = torch::empty({heads}, opts);
  auto code_weight_sums = torch::zeros({heads, pages, value_subvecs, value_centroids}, opts);
  const int64_t* exact_value_counts_ptr =
      exact_value_counts.numel() > 0 ? exact_value_counts.data_ptr<int64_t>() : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_decode_ranked_logits_mask_keys",
      [&] {
        gqa_decode_ranked_logits_mask_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
            queries.data_ptr<float>(),
            keys.data_ptr<scalar_t>(),
            page_starts.data_ptr<int64_t>(),
            ranked_tokens.data_ptr<int64_t>(),
            ranked_scores.data_ptr<float>(),
            tail_mask.data_ptr<unsigned char>(),
            base_tokens.data_ptr<int64_t>(),
            base_logits.data_ptr<float>(),
            base_counts.data_ptr<int64_t>(),
            ranked_logits.data_ptr<float>(),
            ranked_exact.data_ptr<unsigned char>(),
            heads,
            kv_heads,
            ranked,
            dim,
            total_tokens,
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            pages,
            page_size,
            max_base,
            group_size,
            query_context_len,
            static_prefix,
            static_suffix,
            exact_value_counts_ptr,
            nullptr,
            nullptr,
            exact_value_top,
            static_cast<float>(exact_value_mass),
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 tail_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(tail_blocks));
  if (include_tail) {
    gqa_decode_tail_partial_max_kernel<<<tail_grid, threads, threads * sizeof(float), stream>>>(
        dense_pq_scores.data_ptr<float>(),
        page_starts.data_ptr<int64_t>(),
        tail_mask.data_ptr<unsigned char>(),
        partial_max.data_ptr<float>(),
        heads,
        pages,
        page_size,
        tail_blocks,
        pages_per_block,
        query_context_len,
        static_prefix,
        static_suffix,
        static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    partial_max.fill_(-std::numeric_limits<float>::infinity());
    partial_sum.zero_();
  }

  gqa_decode_final_max_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_max.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (include_tail) {
    if (value_codes.scalar_type() == torch::kUInt8) {
      gqa_decode_tail_sum_codeweights_kernel<uint8_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
          dense_pq_scores.data_ptr<float>(),
          max_logits.data_ptr<float>(),
          value_codebooks.data_ptr<float>(),
          value_codes.data_ptr<uint8_t>(),
          page_starts.data_ptr<int64_t>(),
          tail_mask.data_ptr<unsigned char>(),
          partial_sum.data_ptr<float>(),
          code_weight_sums.data_ptr<float>(),
          heads,
          kv_heads,
          pages,
          page_size,
          value_subvecs,
          value_centroids,
          value_subdim,
          tail_blocks,
          pages_per_block,
          group_size,
          query_context_len,
          static_prefix,
          static_suffix,
          static_cast<float>(scale));
    } else {
      gqa_decode_tail_sum_codeweights_kernel<int64_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
          dense_pq_scores.data_ptr<float>(),
          max_logits.data_ptr<float>(),
          value_codebooks.data_ptr<float>(),
          value_codes.data_ptr<int64_t>(),
          page_starts.data_ptr<int64_t>(),
          tail_mask.data_ptr<unsigned char>(),
          partial_sum.data_ptr<float>(),
          code_weight_sums.data_ptr<float>(),
          heads,
          kv_heads,
          pages,
          page_size,
          value_subvecs,
          value_centroids,
          value_subdim,
          tail_blocks,
          pages_per_block,
          group_size,
          query_context_len,
          static_prefix,
          static_suffix,
          static_cast<float>(scale));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  gqa_decode_final_denom_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_sum.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      denoms.data_ptr<float>(),
      selected_denoms.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int output_threads = 256;
  const int64_t output_total = heads * dim;
  const int output_blocks = static_cast<int>((output_total + output_threads - 1) / output_threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "gqa_decode_tail_agg_output_values",
      [&] {
        using value_scalar_t = scalar_t;
        if (value_codes.scalar_type() == torch::kUInt8) {
          gqa_decode_tail_agg_output_kernel<value_scalar_t, uint8_t>
              <<<output_blocks, output_threads, 0, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<uint8_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  ranked_exact.data_ptr<unsigned char>(),
                  max_logits.data_ptr<float>(),
                  denoms.data_ptr<float>(),
                  selected_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  outputs.data_ptr<float>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  static_cast<float>(tail_blend));
        } else {
          gqa_decode_tail_agg_output_kernel<value_scalar_t, int64_t>
              <<<output_blocks, output_threads, 0, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<int64_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  ranked_exact.data_ptr<unsigned char>(),
                  max_logits.data_ptr<float>(),
                  denoms.data_ptr<float>(),
                  selected_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  outputs.data_ptr<float>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  static_cast<float>(tail_blend));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_decode_geometric_accept_counts_cuda_impl(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits_input,
    bool use_ranked_logits_input,
    torch::Tensor approx_exact_thresholds,
    torch::Tensor approx_exact_threshold_sels,
    torch::Tensor probe_exact_thresholds,
    torch::Tensor probe_exact_threshold_sels,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
	    double rel_l2_max,
	    int64_t exact_value_top,
	    double exact_value_mass,
	    int64_t exact_value_min_top,
	    double scale,
	    bool probe_includes_tail,
    bool apply_proxy_gate,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(1);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto counts = torch::empty({heads}, queries.options().dtype(torch::kLong));
  if (heads == 0) {
    return counts;
  }
  if (dim == 0 || kv_heads == 0 || total_tokens == 0 || pages == 0 || page_size <= 0) {
    counts.zero_();
    return counts;
  }

  const int threads = 256;
  const int shared_bytes = threads * 2 * sizeof(float);
  const int64_t pages_per_block = decode_tail_pages_per_block();
  const int64_t tail_blocks = std::max<int64_t>(1, (pages + pages_per_block - 1) / pages_per_block);
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  auto opts = queries.options().dtype(torch::kFloat32);
  auto long_opts = queries.options().dtype(torch::kLong);
  auto base_tokens = torch::empty({heads, max_base}, long_opts);
  auto base_logits = torch::empty({heads, max_base}, opts);
  auto base_counts = torch::zeros({heads}, long_opts);
  auto ranked_logits = torch::empty({heads, ranked}, opts);
  auto partial_max = torch::empty({heads, tail_blocks}, opts);
  auto max_logits = torch::empty({heads}, opts);
  auto partial_sum = torch::empty({heads, tail_blocks}, opts);
  auto tail_denoms = torch::empty({heads}, opts);
  auto code_weight_sums = torch::zeros({heads, pages, value_subvecs, value_centroids}, opts);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (use_ranked_logits_input) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        keys.scalar_type(),
        "gqa_decode_geometric_accept_counts_base_keys",
        [&] {
          gqa_decode_base_logits_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
              queries.data_ptr<float>(),
              keys.data_ptr<scalar_t>(),
              base_tokens.data_ptr<int64_t>(),
              base_logits.data_ptr<float>(),
              base_counts.data_ptr<int64_t>(),
              heads,
              kv_heads,
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
    const int filter_threads = 256;
    const int64_t filter_total = heads * ranked;
    const int filter_blocks = static_cast<int>((filter_total + filter_threads - 1) / filter_threads);
    gqa_decode_filter_ranked_logits_input_kernel<<<filter_blocks, filter_threads, 0, stream>>>(
        ranked_tokens.data_ptr<int64_t>(),
        ranked_scores.data_ptr<float>(),
        ranked_logits_input.data_ptr<float>(),
        ranked_logits.data_ptr<float>(),
        heads,
        ranked,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        keys.scalar_type(),
        "gqa_decode_geometric_accept_counts_base_ranked_keys",
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
  }

  dim3 tail_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(tail_blocks));
  gqa_decode_tail_partial_max_nomask_kernel<<<tail_grid, threads, threads * sizeof(float), stream>>>(
      dense_pq_scores.data_ptr<float>(),
      page_starts.data_ptr<int64_t>(),
      partial_max.data_ptr<float>(),
      heads,
      pages,
      page_size,
      tail_blocks,
      pages_per_block,
      query_context_len,
      static_prefix,
      static_suffix,
      static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_final_max_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_max.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (value_codes.scalar_type() == torch::kUInt8) {
    gqa_decode_tail_sum_codeweights_nomask_kernel<uint8_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
        dense_pq_scores.data_ptr<float>(),
        max_logits.data_ptr<float>(),
        value_codes.data_ptr<uint8_t>(),
        page_starts.data_ptr<int64_t>(),
        partial_sum.data_ptr<float>(),
        code_weight_sums.data_ptr<float>(),
        heads,
        kv_heads,
        pages,
        page_size,
        value_subvecs,
        value_centroids,
        tail_blocks,
        pages_per_block,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        static_cast<float>(scale));
  } else {
    gqa_decode_tail_sum_codeweights_nomask_kernel<int64_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
        dense_pq_scores.data_ptr<float>(),
        max_logits.data_ptr<float>(),
        value_codes.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        partial_sum.data_ptr<float>(),
        code_weight_sums.data_ptr<float>(),
        heads,
        kv_heads,
        pages,
        page_size,
        value_subvecs,
        value_centroids,
        tail_blocks,
        pages_per_block,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        static_cast<float>(scale));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_tail_denom_from_partials_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_sum.data_ptr<float>(),
      tail_denoms.data_ptr<float>(),
      heads,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t threshold_steps =
      (approx_exact_thresholds.defined() && approx_exact_thresholds.numel() > 0) ? approx_exact_thresholds.size(1) : 0;
  const float* approx_exact_thresholds_ptr =
      threshold_steps > 0 ? approx_exact_thresholds.data_ptr<float>() : nullptr;
  const int64_t* approx_exact_threshold_sels_ptr =
      threshold_steps > 0 ? approx_exact_threshold_sels.data_ptr<int64_t>() : nullptr;
  const float* probe_exact_thresholds_ptr =
      threshold_steps > 0 ? probe_exact_thresholds.data_ptr<float>() : nullptr;
  const int64_t* probe_exact_threshold_sels_ptr =
      threshold_steps > 0 ? probe_exact_threshold_sels.data_ptr<int64_t>() : nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "gqa_decode_geometric_accept_counts_values",
      [&] {
        using value_scalar_t = scalar_t;
        if (value_codes.scalar_type() == torch::kUInt8) {
          gqa_decode_geometric_accept_counts_codeweights_kernel<value_scalar_t, uint8_t>
              <<<static_cast<int>(heads), threads, shared_bytes, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  dense_pq_scores.data_ptr<float>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<uint8_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  max_logits.data_ptr<float>(),
                  tail_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  approx_exact_thresholds_ptr,
                  approx_exact_threshold_sels_ptr,
                  probe_exact_thresholds_ptr,
                  probe_exact_threshold_sels_ptr,
                  counts.data_ptr<int64_t>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  min_budget,
                  max_budget,
                  granularity,
                  static_cast<float>(growth),
                  static_cast<float>(probe_scale),
	                  static_cast<float>(rel_l2_max),
	                  exact_value_top,
	                  static_cast<float>(exact_value_mass),
	                  exact_value_min_top,
	                  static_cast<float>(scale),
	                  probe_includes_tail,
	                  threshold_steps);
        } else {
          gqa_decode_geometric_accept_counts_codeweights_kernel<value_scalar_t, int64_t>
              <<<static_cast<int>(heads), threads, shared_bytes, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  dense_pq_scores.data_ptr<float>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<int64_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  max_logits.data_ptr<float>(),
                  tail_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  approx_exact_thresholds_ptr,
                  approx_exact_threshold_sels_ptr,
                  probe_exact_thresholds_ptr,
                  probe_exact_threshold_sels_ptr,
                  counts.data_ptr<int64_t>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  min_budget,
                  max_budget,
                  granularity,
                  static_cast<float>(growth),
                  static_cast<float>(probe_scale),
	                  static_cast<float>(rel_l2_max),
	                  exact_value_top,
	                  static_cast<float>(exact_value_mass),
	                  exact_value_min_top,
	                  static_cast<float>(scale),
	                  probe_includes_tail,
	                  threshold_steps);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (apply_proxy_gate) {
    gqa_decode_proxy_gate_counts_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
        base_logits.data_ptr<float>(),
        base_counts.data_ptr<int64_t>(),
        ranked_logits.data_ptr<float>(),
        ranked_scores.data_ptr<float>(),
        counts.data_ptr<int64_t>(),
        heads,
        ranked,
        max_base,
        max_budget,
        static_cast<float>(scale),
        static_cast<float>(proxy_mass_min),
        static_cast<float>(proxy_tail_mass_max),
        static_cast<float>(pq_corr_min),
        static_cast<float>(pq_relrmse_max),
        calibrate_proxy);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return counts;
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
  if (heads == 0 || ranked == 0) {
    base_lse.fill_(-std::numeric_limits<float>::infinity());
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

torch::Tensor gqa_decode_geometric_accept_counts_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    double scale) {
  return gqa_decode_geometric_accept_counts_cuda_impl(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      torch::Tensor(),
      false,
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
	      rel_l2_max,
	      max_budget,
	      0.0,
	      0,
	      scale,
      false,
      false,
      0.0,
      1.0,
      -1.0,
      std::numeric_limits<double>::infinity(),
      false);
}

torch::Tensor gqa_decode_geometric_accept_counts_vpq_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    int64_t exact_value_top,
    double scale) {
  return gqa_decode_geometric_accept_counts_cuda_impl(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      torch::Tensor(),
      false,
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
	      rel_l2_max,
	      exact_value_top,
	      0.0,
	      0,
	      scale,
      false,
      false,
      0.0,
      1.0,
      -1.0,
      std::numeric_limits<double>::infinity(),
      false);
}

torch::Tensor gqa_decode_geometric_accept_counts_vpq_tail_stability_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    int64_t exact_value_top,
    double scale) {
  return gqa_decode_geometric_accept_counts_cuda_impl(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      torch::Tensor(),
      false,
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
	      rel_l2_max,
	      exact_value_top,
	      0.0,
	      0,
	      scale,
      true,
      false,
      0.0,
      1.0,
      -1.0,
      std::numeric_limits<double>::infinity(),
      false);
}

torch::Tensor gqa_decode_geometric_accept_counts_vpq_proxy_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    int64_t exact_value_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail) {
  return gqa_decode_geometric_accept_counts_cuda_impl(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      torch::Tensor(),
      false,
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
	      rel_l2_max,
	      exact_value_top,
	      0.0,
	      0,
	      scale,
      probe_includes_tail,
      true,
      proxy_mass_min,
      proxy_tail_mass_max,
      pq_corr_min,
      pq_relrmse_max,
      calibrate_proxy);
}

torch::Tensor gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    double exact_value_mass,
    int64_t exact_value_min_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail) {
  return gqa_decode_geometric_accept_counts_cuda_impl(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      torch::Tensor(),
      false,
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
      0,
      exact_value_mass,
      exact_value_min_top,
      scale,
      probe_includes_tail,
      true,
      proxy_mass_min,
      proxy_tail_mass_max,
      pq_corr_min,
      pq_relrmse_max,
      calibrate_proxy);
}

torch::Tensor gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    double exact_value_mass,
    int64_t exact_value_min_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail) {
  return gqa_decode_geometric_accept_counts_cuda_impl(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      ranked_logits,
      true,
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
      0,
      exact_value_mass,
      exact_value_min_top,
      scale,
      probe_includes_tail,
      true,
      proxy_mass_min,
      proxy_tail_mass_max,
      pq_corr_min,
      pq_relrmse_max,
      calibrate_proxy);
}

torch::Tensor gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits,
    torch::Tensor approx_exact_thresholds,
    torch::Tensor approx_exact_threshold_sels,
    torch::Tensor probe_exact_thresholds,
    torch::Tensor probe_exact_threshold_sels,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    double exact_value_mass,
    int64_t exact_value_min_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail) {
  return gqa_decode_geometric_accept_counts_cuda_impl(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      ranked_logits,
      true,
      approx_exact_thresholds,
      approx_exact_threshold_sels,
      probe_exact_thresholds,
      probe_exact_threshold_sels,
      group_size,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
      0,
      exact_value_mass,
      exact_value_min_top,
      scale,
      probe_includes_tail,
      true,
      proxy_mass_min,
      proxy_tail_mass_max,
      pq_corr_min,
      pq_relrmse_max,
      calibrate_proxy);
}

std::vector<torch::Tensor> gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits,
    torch::Tensor approx_exact_thresholds,
    torch::Tensor approx_exact_threshold_sels,
    torch::Tensor probe_exact_thresholds,
    torch::Tensor probe_exact_threshold_sels,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    double exact_value_mass,
    int64_t exact_value_min_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail,
    double tail_blend) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(1);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto counts = torch::empty({heads}, queries.options().dtype(torch::kLong));
  auto outputs = torch::empty({heads, dim}, queries.options().dtype(torch::kFloat32));
  if (heads == 0) {
    return {counts, outputs};
  }
  if (dim == 0 || kv_heads == 0 || total_tokens == 0 || pages == 0 || page_size <= 0) {
    counts.zero_();
    outputs.zero_();
    return {counts, outputs};
  }

  const int threads = 256;
  const int shared_bytes = threads * 2 * sizeof(float);
  const int64_t pages_per_block = decode_tail_pages_per_block();
  const int64_t tail_blocks = std::max<int64_t>(1, (pages + pages_per_block - 1) / pages_per_block);
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  auto opts = queries.options().dtype(torch::kFloat32);
  auto long_opts = queries.options().dtype(torch::kLong);
  auto base_tokens = torch::empty({heads, max_base}, long_opts);
  auto base_logits = torch::empty({heads, max_base}, opts);
  auto base_counts = torch::zeros({heads}, long_opts);
  auto filtered_ranked_logits = torch::empty({heads, ranked}, opts);
  auto partial_max = torch::empty({heads, tail_blocks}, opts);
  auto max_logits = torch::empty({heads}, opts);
  auto partial_sum = torch::empty({heads, tail_blocks}, opts);
  auto tail_denoms = torch::empty({heads}, opts);
  auto code_weight_sums = torch::zeros({heads, pages, value_subvecs, value_centroids}, opts);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_decode_geometric_output_base_keys",
      [&] {
        gqa_decode_base_logits_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
            queries.data_ptr<float>(),
            keys.data_ptr<scalar_t>(),
            base_tokens.data_ptr<int64_t>(),
            base_logits.data_ptr<float>(),
            base_counts.data_ptr<int64_t>(),
            heads,
            kv_heads,
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

  const int filter_threads = 256;
  const int64_t filter_total = heads * ranked;
  const int filter_blocks = static_cast<int>((filter_total + filter_threads - 1) / filter_threads);
  gqa_decode_filter_ranked_logits_input_kernel<<<filter_blocks, filter_threads, 0, stream>>>(
      ranked_tokens.data_ptr<int64_t>(),
      ranked_scores.data_ptr<float>(),
      ranked_logits.data_ptr<float>(),
      filtered_ranked_logits.data_ptr<float>(),
      heads,
      ranked,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 tail_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(tail_blocks));
  gqa_decode_tail_partial_max_nomask_kernel<<<tail_grid, threads, threads * sizeof(float), stream>>>(
      dense_pq_scores.data_ptr<float>(),
      page_starts.data_ptr<int64_t>(),
      partial_max.data_ptr<float>(),
      heads,
      pages,
      page_size,
      tail_blocks,
      pages_per_block,
      query_context_len,
      static_prefix,
      static_suffix,
      static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_final_max_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_max.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      filtered_ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (value_codes.scalar_type() == torch::kUInt8) {
    gqa_decode_tail_sum_codeweights_nomask_kernel<uint8_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
        dense_pq_scores.data_ptr<float>(),
        max_logits.data_ptr<float>(),
        value_codes.data_ptr<uint8_t>(),
        page_starts.data_ptr<int64_t>(),
        partial_sum.data_ptr<float>(),
        code_weight_sums.data_ptr<float>(),
        heads,
        kv_heads,
        pages,
        page_size,
        value_subvecs,
        value_centroids,
        tail_blocks,
        pages_per_block,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        static_cast<float>(scale));
  } else {
    gqa_decode_tail_sum_codeweights_nomask_kernel<int64_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
        dense_pq_scores.data_ptr<float>(),
        max_logits.data_ptr<float>(),
        value_codes.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        partial_sum.data_ptr<float>(),
        code_weight_sums.data_ptr<float>(),
        heads,
        kv_heads,
        pages,
        page_size,
        value_subvecs,
        value_centroids,
        tail_blocks,
        pages_per_block,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        static_cast<float>(scale));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_tail_denom_from_partials_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_sum.data_ptr<float>(),
      tail_denoms.data_ptr<float>(),
      heads,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t threshold_steps =
      (approx_exact_thresholds.defined() && approx_exact_thresholds.numel() > 0) ? approx_exact_thresholds.size(1) : 0;
  const float* approx_exact_thresholds_ptr =
      threshold_steps > 0 ? approx_exact_thresholds.data_ptr<float>() : nullptr;
  const int64_t* approx_exact_threshold_sels_ptr =
      threshold_steps > 0 ? approx_exact_threshold_sels.data_ptr<int64_t>() : nullptr;
  const float* probe_exact_thresholds_ptr =
      threshold_steps > 0 ? probe_exact_thresholds.data_ptr<float>() : nullptr;
  const int64_t* probe_exact_threshold_sels_ptr =
      threshold_steps > 0 ? probe_exact_threshold_sels.data_ptr<int64_t>() : nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "gqa_decode_geometric_output_counts_values",
      [&] {
        using value_scalar_t = scalar_t;
        if (value_codes.scalar_type() == torch::kUInt8) {
          gqa_decode_geometric_accept_counts_codeweights_kernel<value_scalar_t, uint8_t>
              <<<static_cast<int>(heads), threads, shared_bytes, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  dense_pq_scores.data_ptr<float>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<uint8_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  filtered_ranked_logits.data_ptr<float>(),
                  max_logits.data_ptr<float>(),
                  tail_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  approx_exact_thresholds_ptr,
                  approx_exact_threshold_sels_ptr,
                  probe_exact_thresholds_ptr,
                  probe_exact_threshold_sels_ptr,
                  counts.data_ptr<int64_t>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  min_budget,
                  max_budget,
                  granularity,
                  static_cast<float>(growth),
                  static_cast<float>(probe_scale),
                  static_cast<float>(rel_l2_max),
                  0,
                  static_cast<float>(exact_value_mass),
                  exact_value_min_top,
                  static_cast<float>(scale),
                  probe_includes_tail,
                  threshold_steps);
        } else {
          gqa_decode_geometric_accept_counts_codeweights_kernel<value_scalar_t, int64_t>
              <<<static_cast<int>(heads), threads, shared_bytes, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  dense_pq_scores.data_ptr<float>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<int64_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  filtered_ranked_logits.data_ptr<float>(),
                  max_logits.data_ptr<float>(),
                  tail_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  approx_exact_thresholds_ptr,
                  approx_exact_threshold_sels_ptr,
                  probe_exact_thresholds_ptr,
                  probe_exact_threshold_sels_ptr,
                  counts.data_ptr<int64_t>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  min_budget,
                  max_budget,
                  granularity,
                  static_cast<float>(growth),
                  static_cast<float>(probe_scale),
                  static_cast<float>(rel_l2_max),
                  0,
                  static_cast<float>(exact_value_mass),
                  exact_value_min_top,
                  static_cast<float>(scale),
                  probe_includes_tail,
                  threshold_steps);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_proxy_gate_counts_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      filtered_ranked_logits.data_ptr<float>(),
      ranked_scores.data_ptr<float>(),
      counts.data_ptr<int64_t>(),
      heads,
      ranked,
      max_base,
      max_budget,
      static_cast<float>(scale),
      static_cast<float>(proxy_mass_min),
      static_cast<float>(proxy_tail_mass_max),
      static_cast<float>(pq_corr_min),
      static_cast<float>(pq_relrmse_max),
      calibrate_proxy);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "gqa_decode_geometric_output_final_values",
      [&] {
        using value_scalar_t = scalar_t;
        if (value_codes.scalar_type() == torch::kUInt8) {
          gqa_decode_geometric_final_output_codeweights_kernel<value_scalar_t, uint8_t>
              <<<static_cast<int>(heads), threads, 0, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  dense_pq_scores.data_ptr<float>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<uint8_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  filtered_ranked_logits.data_ptr<float>(),
                  partial_max.data_ptr<float>(),
                  max_logits.data_ptr<float>(),
                  tail_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  counts.data_ptr<int64_t>(),
                  outputs.data_ptr<float>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  tail_blocks,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(exact_value_mass),
                  exact_value_min_top,
                  static_cast<float>(scale),
                  static_cast<float>(tail_blend));
        } else {
          gqa_decode_geometric_final_output_codeweights_kernel<value_scalar_t, int64_t>
              <<<static_cast<int>(heads), threads, 0, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  dense_pq_scores.data_ptr<float>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<int64_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  filtered_ranked_logits.data_ptr<float>(),
                  partial_max.data_ptr<float>(),
                  max_logits.data_ptr<float>(),
                  tail_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  counts.data_ptr<int64_t>(),
                  outputs.data_ptr<float>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  tail_blocks,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(exact_value_mass),
                  exact_value_min_top,
                  static_cast<float>(scale),
                  static_cast<float>(tail_blend));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {counts, outputs};
}

torch::Tensor gqa_causal_geometric_accept_counts_cuda_impl(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    int64_t exact_value_top,
    double scale) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(2);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto counts = torch::empty({positions, heads}, queries.options().dtype(torch::kLong));
  if (positions == 0 || heads == 0) {
    return counts;
  }
  if (dim == 0 || kv_heads == 0 || total_tokens == 0 || pages == 0 || page_size <= 0) {
    counts.zero_();
    return counts;
  }

  const int threads = 256;
  const int64_t flat_tokens = pages * page_size;
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  const size_t shared_bytes =
      static_cast<size_t>(threads * 2 + max_base + ranked) * sizeof(float) +
      static_cast<size_t>(max_base) * sizeof(int32_t);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t qh_pairs = positions * heads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_causal_geometric_accept_counts_keys",
      [&] {
        using key_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            values.scalar_type(),
            "gqa_causal_geometric_accept_counts_values",
            [&] {
              using value_scalar_t = scalar_t;
              if (value_codes.scalar_type() == torch::kUInt8) {
                cudaFuncSetAttribute(
                    gqa_causal_geometric_accept_counts_kernel<key_scalar_t, value_scalar_t, uint8_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_geometric_accept_counts_kernel<key_scalar_t, value_scalar_t, uint8_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        dense_pq_scores.data_ptr<float>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<uint8_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        counts.data_ptr<int64_t>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        keys.stride(0),
                        keys.stride(1),
                        keys.stride(2),
                        values.stride(0),
                        values.stride(1),
                        values.stride(2),
                        pages,
                        page_size,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        max_base,
                        min_budget,
                        max_budget,
                        granularity,
                        static_cast<float>(growth),
                        static_cast<float>(probe_scale),
                        static_cast<float>(rel_l2_max),
                        exact_value_top,
                        static_cast<float>(scale));
              } else {
                cudaFuncSetAttribute(
                    gqa_causal_geometric_accept_counts_kernel<key_scalar_t, value_scalar_t, int64_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_geometric_accept_counts_kernel<key_scalar_t, value_scalar_t, int64_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        dense_pq_scores.data_ptr<float>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<int64_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        counts.data_ptr<int64_t>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        keys.stride(0),
                        keys.stride(1),
                        keys.stride(2),
                        values.stride(0),
                        values.stride(1),
                        values.stride(2),
                        pages,
                        page_size,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        max_base,
                        min_budget,
                        max_budget,
                        granularity,
                        static_cast<float>(growth),
                        static_cast<float>(probe_scale),
                        static_cast<float>(rel_l2_max),
                        exact_value_top,
                        static_cast<float>(scale));
              }
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return counts;
}

torch::Tensor gqa_causal_geometric_accept_counts_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    double scale) {
  return gqa_causal_geometric_accept_counts_cuda_impl(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      group_size,
      query_start,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
      max_budget,
      scale);
}

torch::Tensor gqa_causal_geometric_accept_counts_vpq_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    int64_t exact_value_top,
    double scale) {
  return gqa_causal_geometric_accept_counts_cuda_impl(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      group_size,
      query_start,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
      exact_value_top,
      scale);
}

torch::Tensor gqa_decode_vpq_selected_from_logits_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits_input,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(1);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({heads, dim}, opts);
  if (heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }
  if (pages == 0 || page_size <= 0) {
    outputs.zero_();
    return outputs;
  }

  const int threads = 256;
  const int64_t flat_tokens = pages * page_size;
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  auto byte_opts = queries.options().dtype(torch::kUInt8);
  auto tail_mask = torch::zeros({heads, flat_tokens}, byte_opts);
  auto long_opts = queries.options().dtype(torch::kLong);
  auto base_tokens = torch::empty({heads, max_base}, long_opts);
  auto base_logits = torch::empty({heads, max_base}, opts);
  auto base_counts = torch::zeros({heads}, long_opts);
  auto ranked_logits = torch::empty({heads, ranked}, opts);
  auto ranked_exact = torch::zeros({heads, ranked}, byte_opts);
  auto partial_max = torch::full({heads, 1}, -std::numeric_limits<float>::infinity(), opts);
  auto max_logits = torch::empty({heads}, opts);
  auto partial_sum = torch::zeros({heads, 1}, opts);
  auto denoms = torch::empty({heads}, opts);
  auto selected_denoms = torch::empty({heads}, opts);
  auto code_weight_sums = torch::empty({1}, opts);
  const int64_t* exact_value_counts_ptr =
      exact_value_counts.numel() > 0 ? exact_value_counts.data_ptr<int64_t>() : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream();
	  AT_DISPATCH_FLOATING_TYPES_AND2(
	      at::ScalarType::Half,
	      at::ScalarType::BFloat16,
	      keys.scalar_type(),
	      "gqa_decode_ranked_logits_mask_from_logits_keys",
      [&] {
        gqa_decode_ranked_logits_mask_from_logits_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
            queries.data_ptr<float>(),
            keys.data_ptr<scalar_t>(),
            page_starts.data_ptr<int64_t>(),
            ranked_tokens.data_ptr<int64_t>(),
            ranked_scores.data_ptr<float>(),
            ranked_logits_input.data_ptr<float>(),
            tail_mask.data_ptr<unsigned char>(),
            base_tokens.data_ptr<int64_t>(),
            base_logits.data_ptr<float>(),
            base_counts.data_ptr<int64_t>(),
            ranked_logits.data_ptr<float>(),
            ranked_exact.data_ptr<unsigned char>(),
            heads,
            kv_heads,
            ranked,
            dim,
            total_tokens,
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            pages,
            page_size,
            max_base,
            group_size,
            query_context_len,
            static_prefix,
            static_suffix,
            exact_value_counts_ptr,
            nullptr,
            nullptr,
            exact_value_top,
            static_cast<float>(exact_value_mass),
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_final_max_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_max.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_final_denom_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_sum.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      denoms.data_ptr<float>(),
      selected_denoms.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int output_threads = 256;
  const int64_t output_total = heads * dim;
  const int output_blocks = static_cast<int>((output_total + output_threads - 1) / output_threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "gqa_decode_selected_from_logits_output_values",
      [&] {
        using value_scalar_t = scalar_t;
        if (value_codes.scalar_type() == torch::kUInt8) {
          gqa_decode_tail_agg_output_kernel<value_scalar_t, uint8_t>
              <<<output_blocks, output_threads, 0, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<uint8_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  ranked_exact.data_ptr<unsigned char>(),
                  max_logits.data_ptr<float>(),
                  denoms.data_ptr<float>(),
                  selected_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  outputs.data_ptr<float>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  0.0f);
        } else {
          gqa_decode_tail_agg_output_kernel<value_scalar_t, int64_t>
              <<<output_blocks, output_threads, 0, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<int64_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  ranked_exact.data_ptr<unsigned char>(),
                  max_logits.data_ptr<float>(),
                  denoms.data_ptr<float>(),
                  selected_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  outputs.data_ptr<float>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  0.0f);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_decode_vpq_selected_tail_agg_from_logits_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits_input,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale,
    double tail_blend,
    torch::Tensor exact_value_thresholds = torch::Tensor(),
    torch::Tensor exact_value_threshold_sels = torch::Tensor()) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(1);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({heads, dim}, opts);
  if (heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }
  if (pages == 0 || page_size <= 0) {
    outputs.zero_();
    return outputs;
  }

  const int threads = 256;
  const bool include_tail = tail_blend > 0.0;
  const int64_t flat_tokens = pages * page_size;
  const int64_t pages_per_block = decode_tail_pages_per_block();
  const int64_t tail_blocks = std::max<int64_t>(1, (pages + pages_per_block - 1) / pages_per_block);
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  auto byte_opts = queries.options().dtype(torch::kUInt8);
  auto tail_mask = torch::zeros({heads, flat_tokens}, byte_opts);
  auto long_opts = queries.options().dtype(torch::kLong);
  auto base_tokens = torch::empty({heads, max_base}, long_opts);
  auto base_logits = torch::empty({heads, max_base}, opts);
  auto base_counts = torch::zeros({heads}, long_opts);
  auto ranked_logits = torch::empty({heads, ranked}, opts);
  auto ranked_exact = torch::zeros({heads, ranked}, byte_opts);
  auto partial_max = torch::empty({heads, tail_blocks}, opts);
  auto max_logits = torch::empty({heads}, opts);
  auto partial_sum = torch::empty({heads, tail_blocks}, opts);
  auto denoms = torch::empty({heads}, opts);
  auto selected_denoms = torch::empty({heads}, opts);
  auto code_weight_sums = torch::zeros({heads, pages, value_subvecs, value_centroids}, opts);
  const int64_t* exact_value_counts_ptr =
      exact_value_counts.numel() > 0 ? exact_value_counts.data_ptr<int64_t>() : nullptr;
  const bool has_exact_value_thresholds =
      exact_value_thresholds.defined() && exact_value_threshold_sels.defined() &&
      exact_value_thresholds.numel() > 0 && exact_value_threshold_sels.numel() > 0;
  const float* exact_value_thresholds_ptr =
      has_exact_value_thresholds ? exact_value_thresholds.data_ptr<float>() : nullptr;
  const int64_t* exact_value_threshold_sels_ptr =
      has_exact_value_thresholds ? exact_value_threshold_sels.data_ptr<int64_t>() : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_decode_ranked_logits_mask_from_logits_keys",
      [&] {
        gqa_decode_ranked_logits_mask_from_logits_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
            queries.data_ptr<float>(),
            keys.data_ptr<scalar_t>(),
            page_starts.data_ptr<int64_t>(),
            ranked_tokens.data_ptr<int64_t>(),
            ranked_scores.data_ptr<float>(),
            ranked_logits_input.data_ptr<float>(),
            tail_mask.data_ptr<unsigned char>(),
            base_tokens.data_ptr<int64_t>(),
            base_logits.data_ptr<float>(),
            base_counts.data_ptr<int64_t>(),
            ranked_logits.data_ptr<float>(),
            ranked_exact.data_ptr<unsigned char>(),
            heads,
            kv_heads,
            ranked,
            dim,
            total_tokens,
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            pages,
            page_size,
            max_base,
            group_size,
            query_context_len,
            static_prefix,
            static_suffix,
            exact_value_counts_ptr,
            exact_value_thresholds_ptr,
            exact_value_threshold_sels_ptr,
            exact_value_top,
            static_cast<float>(exact_value_mass),
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 tail_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(tail_blocks));
  if (include_tail) {
    gqa_decode_tail_partial_max_kernel<<<tail_grid, threads, threads * sizeof(float), stream>>>(
        dense_pq_scores.data_ptr<float>(),
        page_starts.data_ptr<int64_t>(),
        tail_mask.data_ptr<unsigned char>(),
        partial_max.data_ptr<float>(),
        heads,
        pages,
        page_size,
        tail_blocks,
        pages_per_block,
        query_context_len,
        static_prefix,
        static_suffix,
        static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    partial_max.fill_(-std::numeric_limits<float>::infinity());
    partial_sum.zero_();
  }

  gqa_decode_final_max_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_max.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (include_tail) {
    if (value_codes.scalar_type() == torch::kUInt8) {
      gqa_decode_tail_sum_codeweights_kernel<uint8_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
          dense_pq_scores.data_ptr<float>(),
          max_logits.data_ptr<float>(),
          value_codebooks.data_ptr<float>(),
          value_codes.data_ptr<uint8_t>(),
          page_starts.data_ptr<int64_t>(),
          tail_mask.data_ptr<unsigned char>(),
          partial_sum.data_ptr<float>(),
          code_weight_sums.data_ptr<float>(),
          heads,
          kv_heads,
          pages,
          page_size,
          value_subvecs,
          value_centroids,
          value_subdim,
          tail_blocks,
          pages_per_block,
          group_size,
          query_context_len,
          static_prefix,
          static_suffix,
          static_cast<float>(scale));
    } else {
      gqa_decode_tail_sum_codeweights_kernel<int64_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
          dense_pq_scores.data_ptr<float>(),
          max_logits.data_ptr<float>(),
          value_codebooks.data_ptr<float>(),
          value_codes.data_ptr<int64_t>(),
          page_starts.data_ptr<int64_t>(),
          tail_mask.data_ptr<unsigned char>(),
          partial_sum.data_ptr<float>(),
          code_weight_sums.data_ptr<float>(),
          heads,
          kv_heads,
          pages,
          page_size,
          value_subvecs,
          value_centroids,
          value_subdim,
          tail_blocks,
          pages_per_block,
          group_size,
          query_context_len,
          static_prefix,
          static_suffix,
          static_cast<float>(scale));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  gqa_decode_final_denom_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_sum.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      denoms.data_ptr<float>(),
      selected_denoms.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int output_threads = 256;
  const int64_t output_total = heads * dim;
  const int output_blocks = static_cast<int>((output_total + output_threads - 1) / output_threads);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "gqa_decode_tail_agg_from_logits_output_values",
      [&] {
        using value_scalar_t = scalar_t;
        if (value_codes.scalar_type() == torch::kUInt8) {
          gqa_decode_tail_agg_output_kernel<value_scalar_t, uint8_t>
              <<<output_blocks, output_threads, 0, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<uint8_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  ranked_exact.data_ptr<unsigned char>(),
                  max_logits.data_ptr<float>(),
                  denoms.data_ptr<float>(),
                  selected_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  outputs.data_ptr<float>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  static_cast<float>(tail_blend));
        } else {
          gqa_decode_tail_agg_output_kernel<value_scalar_t, int64_t>
              <<<output_blocks, output_threads, 0, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<int64_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  ranked_exact.data_ptr<unsigned char>(),
                  max_logits.data_ptr<float>(),
                  denoms.data_ptr<float>(),
                  selected_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  outputs.data_ptr<float>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  static_cast<float>(tail_blend));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_exact_selected_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor tokens,
    int64_t group_size,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto selected = tokens.size(1);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({heads, dim}, opts);
  if (heads == 0 || dim == 0) {
    return outputs;
  }
  if (selected == 0 || total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }

  auto logits = torch::empty({heads, selected}, opts);
  const int threads = 128;
  const int64_t pairs = heads * selected;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, keys.scalar_type(), "gqa_exact_selected_keys", [&] {
    gqa_exact_selected_logits_kernel<scalar_t><<<static_cast<int>(pairs), threads, threads * sizeof(float), stream>>>(
        queries.data_ptr<float>(),
        keys.data_ptr<scalar_t>(),
        tokens.data_ptr<int64_t>(),
        logits.data_ptr<float>(),
        heads,
        kv_heads,
        selected,
        dim,
        total_tokens,
        group_size,
        static_cast<float>(scale));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "gqa_exact_selected_values", [&] {
    gqa_exact_selected_output_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
        values.data_ptr<scalar_t>(),
        tokens.data_ptr<int64_t>(),
        logits.data_ptr<float>(),
        outputs.data_ptr<float>(),
        heads,
        kv_heads,
        selected,
        dim,
        total_tokens,
        group_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}
