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

template <typename order_t>
__global__ void joint_vprefix_outputs_kernel(
    const float* __restrict__ base_outputs,
    const float* __restrict__ probs,
    const float* __restrict__ residual,
    const order_t* __restrict__ exact_order,
    const int64_t* __restrict__ v_budgets,
    float* __restrict__ outputs,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t max_exact,
    int64_t v_steps,
    int64_t dim) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * dim;
  if (linear >= total) {
    return;
  }

  int64_t d = linear % dim;
  int64_t tmp = linear / dim;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t kh = ki * heads + head;

  float acc = base_outputs[kh * dim + d];
  int64_t prev_count = 0;
  for (int64_t vi = 0; vi < v_steps; ++vi) {
    int64_t exact_count = v_budgets[vi];
    exact_count = max(static_cast<int64_t>(0), min(exact_count, context_len));
    exact_count = min(exact_count, max_exact);
    if (exact_count < prev_count) {
      // The Python budget parser sorts canonical schedules, but keep this helper
      // well-defined for diagnostic calls with unsorted budgets.
      acc = base_outputs[kh * dim + d];
      prev_count = 0;
    }
    for (int64_t j = prev_count; j < exact_count; ++j) {
      int64_t token = static_cast<int64_t>(exact_order[(kh * max_exact) + j]);
      if (token >= 0 && token < context_len) {
        acc += probs[kh * context_len + token] * residual[token * dim + d];
      }
    }
    outputs[((ki * v_steps + vi) * heads + head) * dim + d] = acc;
    prev_count = exact_count;
  }
}

__global__ void joint_risk_sort_inputs_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ code_error,
    float* __restrict__ risk,
    int32_t* __restrict__ token_ids,
    int64_t rows,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * context_len;
  if (linear >= total) {
    return;
  }
  int64_t token = linear % context_len;
  float p = probs[linear];
  float e = code_error[token];
  risk[linear] = p * p * e;
  token_ids[linear] = static_cast<int32_t>(token);
}

__global__ void joint_grouped_risk_sort_inputs_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ code_error_groups,
    const int64_t* __restrict__ row_group_ids,
    float* __restrict__ risk,
    int32_t* __restrict__ token_ids,
    int64_t rows,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * context_len;
  if (linear >= total) {
    return;
  }
  int64_t token = linear % context_len;
  int64_t row = linear / context_len;
  int64_t group = row_group_ids[row];
  float p = probs[linear];
  float e = code_error_groups[group * context_len + token];
  risk[linear] = p * p * e;
  token_ids[linear] = static_cast<int32_t>(token);
}

__global__ void joint_grouped_flat_risk_sort_inputs_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ code_error_groups,
    float* __restrict__ risk,
    int32_t* __restrict__ token_ids,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t rows = groups * k_count * heads;
  int64_t total = rows * context_len;
  if (linear >= total) {
    return;
  }
  int64_t token = linear % context_len;
  int64_t row = linear / context_len;
  int64_t group = row / (k_count * heads);
  float p = probs[linear];
  float e = code_error_groups[group * context_len + token];
  risk[linear] = p * p * e;
  token_ids[linear] = static_cast<int32_t>(token);
}

template <typename code_t>
__global__ void joint_vpq_code_weight_kernel(
    const float* __restrict__ probs,
    const code_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    float* __restrict__ code_weights,
    int64_t rows,
    int64_t context_len,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t code_count) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * context_len;
  if (linear >= total || pages <= 0 || page_size <= 0 || value_subvecs <= 0 || code_count <= 0) {
    return;
  }

  int64_t token = linear % context_len;
  int64_t row = linear / context_len;
  int64_t first_start = page_starts[0];
  int64_t page = (token - first_start) / page_size;
  if (token < first_start || page < 0 || page >= pages) {
    return;
  }
  int64_t page_start = page_starts[page];
  int64_t row_in_page = token - page_start;
  if (row_in_page < 0 || row_in_page >= page_size) {
    return;
  }

  int64_t code = static_cast<int64_t>(value_codes[(page * page_size + row_in_page) * value_subvecs]);
  code = max(static_cast<int64_t>(0), min(code, code_count - 1));
  atomicAdd(&code_weights[(row * pages + page) * code_count + code], probs[linear]);
}

template <typename value_t>
__global__ void joint_vpq_base_from_code_weights_kernel(
    const float* __restrict__ probs,
    const value_t* __restrict__ values,
    const float* __restrict__ value_codebooks,
    const float* __restrict__ code_weights,
    const int64_t* __restrict__ fallback_tokens,
    float* __restrict__ outputs,
    int64_t rows,
    int64_t context_len,
    int64_t dim,
    int64_t pages,
    int64_t code_count,
    int64_t fallback_count) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * dim;
  if (linear >= total) {
    return;
  }

  int64_t d = linear % dim;
  int64_t row = linear / dim;
  float acc = 0.0f;

  for (int64_t page = 0; page < pages; ++page) {
    const float* weights = code_weights + (row * pages + page) * code_count;
    const float* codebook = value_codebooks + page * code_count * dim;
    for (int64_t code = 0; code < code_count; ++code) {
      acc += weights[code] * codebook[code * dim + d];
    }
  }

  for (int64_t i = 0; i < fallback_count; ++i) {
    int64_t token = fallback_tokens[i];
    if (token >= 0 && token < context_len) {
      acc += probs[row * context_len + token] * load_as_float(values, token * dim + d);
    }
  }

  outputs[row * dim + d] = acc;
}

