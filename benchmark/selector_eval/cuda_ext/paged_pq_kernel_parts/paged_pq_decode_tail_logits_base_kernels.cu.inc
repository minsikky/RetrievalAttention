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

