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

