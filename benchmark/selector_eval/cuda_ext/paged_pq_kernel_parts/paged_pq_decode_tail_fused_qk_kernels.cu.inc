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

