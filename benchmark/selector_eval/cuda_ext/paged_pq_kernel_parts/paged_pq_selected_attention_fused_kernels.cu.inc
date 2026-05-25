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

