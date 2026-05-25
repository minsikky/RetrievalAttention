__global__ void gqa_decode_geometric_pick_counts_from_step_diffs_kernel(
    const float* __restrict__ diff_squares,
    const float* __restrict__ probe_squares,
    const float* __restrict__ max_logits,
    int64_t* __restrict__ accepted_counts,
    int64_t heads,
    int64_t threshold_steps,
    int64_t dim,
    int64_t min_budget,
    int64_t max_budget_arg,
    int64_t ranked,
    int64_t granularity,
    float growth,
    float probe_scale,
    float rel_l2_max) {
  int64_t head = blockIdx.x;
  if (head >= heads) {
    return;
  }
  extern __shared__ float shared[];
  float* diff_shared = shared;
  float* probe_shared = shared + blockDim.x;
  int64_t max_budget = max_budget_arg;
  if (max_budget <= 0 || max_budget > ranked) {
    max_budget = ranked;
  }
  if (!isfinite(max_logits[head])) {
    if (threadIdx.x == 0) {
      accepted_counts[head] = max_budget;
    }
    return;
  }
  __shared__ int done_shared;
  __shared__ int64_t accepted_shared;
  if (threadIdx.x == 0) {
    done_shared = 0;
    accepted_shared = max_budget;
  }
  __syncthreads();

  for (int64_t step = 0; step < threshold_steps; ++step) {
    float local_diff = 0.0f;
    float local_probe = 0.0f;
    int64_t row_offset = (head * threshold_steps + step) * dim;
    for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
      local_diff += diff_squares[row_offset + d];
      local_probe += probe_squares[row_offset + d];
    }
    diff_shared[threadIdx.x] = local_diff;
    probe_shared[threadIdx.x] = local_probe;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        diff_shared[threadIdx.x] += diff_shared[threadIdx.x + stride];
        probe_shared[threadIdx.x] += probe_shared[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      int64_t tail_budget = 0;
      int64_t probe_budget = 0;
      geometric_budget_pair_for_step_device(
          step,
          min_budget,
          max_budget,
          granularity,
          growth,
          probe_scale,
          &tail_budget,
          &probe_budget);
      float denom = sqrtf(fmaxf(probe_shared[0], 1.0e-40f));
      float rel = sqrtf(fmaxf(diff_shared[0], 0.0f)) / denom;
      if (rel <= rel_l2_max) {
        accepted_shared = probe_budget;
        done_shared = 1;
      }
    }
    __syncthreads();
    if (done_shared != 0) {
      break;
    }
  }
  if (threadIdx.x == 0) {
    accepted_counts[head] = accepted_shared;
  }
}

__global__ void gqa_decode_geometric_pick_candidate_outputs_kernel(
    const float* __restrict__ candidate_outputs,
    const int64_t* __restrict__ accepted_counts,
    float* __restrict__ outputs,
    int64_t heads,
    int64_t threshold_steps,
    int64_t dim,
    int64_t min_budget,
    int64_t max_budget_arg,
    int64_t ranked,
    int64_t granularity,
    float growth,
    float probe_scale) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * dim;
  if (linear >= total || threshold_steps <= 0) {
    return;
  }
  int64_t head = linear / dim;
  int64_t d = linear - head * dim;
  int64_t max_budget = max_budget_arg;
  if (max_budget <= 0 || max_budget > ranked) {
    max_budget = ranked;
  }
  int64_t accepted = accepted_counts[head];
  accepted = max(static_cast<int64_t>(0), min(accepted, max_budget));
  int64_t chosen_step = threshold_steps - 1;
  for (int64_t step = 0; step < threshold_steps; ++step) {
    int64_t tail_budget = 0;
    int64_t probe_budget = 0;
    geometric_budget_pair_for_step_device(
        step,
        min_budget,
        max_budget,
        granularity,
        growth,
        probe_scale,
        &tail_budget,
        &probe_budget);
    if (probe_budget >= accepted) {
      chosen_step = step;
      break;
    }
  }
  outputs[head * dim + d] = candidate_outputs[(head * threshold_steps + chosen_step) * dim + d];
}

template <typename value_t>
__global__ void gqa_decode_geometric_pick_output_from_deltas_kernel(
    const value_t* __restrict__ values,
    const float* __restrict__ value_codebooks,
    const int64_t* __restrict__ base_tokens,
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    const float* __restrict__ final_den_incs,
    const float* __restrict__ final_code_incs,
    const int32_t* __restrict__ probe_exact_counts,
    const int32_t* __restrict__ probe_exact_tokens,
    const float* __restrict__ probe_exact_weights,
    const float* __restrict__ max_logits,
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
    int64_t max_base,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t min_budget,
    int64_t max_budget_arg,
    int64_t granularity,
    float growth,
    float probe_scale,
    int64_t exact_list_capacity,
    int64_t threshold_steps) {
  int64_t head = blockIdx.x;
  int64_t d = blockIdx.y;
  if (head >= heads || d >= dim || threshold_steps <= 0) {
    return;
  }
  extern __shared__ float shared[];
  float* den_parts = shared;
  float* num_parts = den_parts + blockDim.x;

  int64_t max_budget = max_budget_arg;
  if (max_budget <= 0 || max_budget > ranked) {
    max_budget = ranked;
  }
  int64_t accepted = accepted_counts[head];
  accepted = max(static_cast<int64_t>(0), min(accepted, max_budget));
  int64_t chosen_step = threshold_steps - 1;
  for (int64_t step = 0; step < threshold_steps; ++step) {
    int64_t tail_budget = 0;
    int64_t probe_budget = 0;
    geometric_budget_pair_for_step_device(
        step,
        min_budget,
        max_budget,
        granularity,
        growth,
        probe_scale,
        &tail_budget,
        &probe_budget);
    if (probe_budget >= accepted) {
      chosen_step = step;
      break;
    }
  }

  float global_max = max_logits[head];
  if (!isfinite(global_max)) {
    if (threadIdx.x == 0) {
      outputs[head * dim + d] = 0.0f;
    }
    return;
  }
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  const int64_t* base_tokens_row = base_tokens + head * max_base;
  const float* base_logits_row = base_logits + head * max_base;
  int64_t base_count = max(static_cast<int64_t>(0), min(base_counts[head], max_base));

  float den_local = 0.0f;
  float num_local = 0.0f;
  for (int64_t idx = threadIdx.x; idx < base_count; idx += blockDim.x) {
    int64_t token = base_tokens_row[idx];
    float logit = base_logits_row[idx];
    if (token < 0 || token >= total_tokens || !isfinite(logit)) {
      continue;
    }
    float w = expf(logit - global_max);
    den_local += w;
    num_local += w * load_strided3_as_float(
                            values,
                            kv_head,
                            token,
                            d,
                            value_stride_head,
                            value_stride_token,
                            value_stride_dim);
  }

  if (threadIdx.x == 0) {
    den_local += tail_denoms[head];
  }
  if (value_subdim > 0) {
    int64_t sub = d / value_subdim;
    int64_t sub_d = d - sub * value_subdim;
    if (sub >= 0 && sub < value_subvecs) {
      int64_t total_codes = pages * value_centroids;
      for (int64_t idx = threadIdx.x; idx < total_codes; idx += blockDim.x) {
        int64_t page = idx / value_centroids;
        int64_t code = idx - page * value_centroids;
        float w = code_weight_sums[(((head * pages + page) * value_subvecs + sub) * value_centroids + code)];
        float value = value_codebooks
            [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
             sub_d];
        num_local += w * value;
      }
    }
  }

  for (int64_t step = threadIdx.x; step <= chosen_step; step += blockDim.x) {
    den_local += final_den_incs[head * threshold_steps + step];
  }
  if (value_subdim > 0) {
    int64_t sub = d / value_subdim;
    int64_t sub_d = d - sub * value_subdim;
    if (sub >= 0 && sub < value_subvecs) {
      int64_t total_codes = pages * value_centroids;
      for (int64_t step = 0; step <= chosen_step; ++step) {
        int64_t code_base = (((head * threshold_steps + step) * pages) * value_subvecs + sub) * value_centroids;
        for (int64_t idx = threadIdx.x; idx < total_codes; idx += blockDim.x) {
          int64_t page = idx / value_centroids;
          int64_t code = idx - page * value_centroids;
          float value = value_codebooks
              [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
               sub_d];
          num_local += final_code_incs[code_base + page * value_subvecs * value_centroids + code] * value;
        }
      }
    }
  }

  int64_t capacity = exact_list_capacity;
  if (capacity <= 0 || capacity > max_budget) {
    capacity = max_budget;
  }
  for (int64_t step = 0; step <= chosen_step; ++step) {
    int64_t count_idx = head * threshold_steps + step;
    int64_t exact_count = static_cast<int64_t>(probe_exact_counts[count_idx]);
    exact_count = max(static_cast<int64_t>(0), min(exact_count, capacity));
    int64_t list_base = (head * threshold_steps + step) * capacity;
    for (int64_t idx = threadIdx.x; idx < exact_count; idx += blockDim.x) {
      int64_t token = static_cast<int64_t>(probe_exact_tokens[list_base + idx]);
      float w = probe_exact_weights[list_base + idx];
      if (w == 0.0f || token < 0 || token >= total_tokens) {
        continue;
      }
      num_local += w * load_strided3_as_float(
                         values,
                         kv_head,
                         token,
                         d,
                         value_stride_head,
                         value_stride_token,
                         value_stride_dim);
    }
  }

  den_parts[threadIdx.x] = den_local;
  num_parts[threadIdx.x] = num_local;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      den_parts[threadIdx.x] += den_parts[threadIdx.x + stride];
      num_parts[threadIdx.x] += num_parts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    outputs[head * dim + d] = num_parts[0] / fmaxf(den_parts[0], 1.0e-20f);
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

__global__ void gqa_decode_geometric_final_head_stats_kernel(
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    const float* __restrict__ ranked_logits,
    const float* __restrict__ tail_partial_max,
    const float* __restrict__ probe_exact_thresholds,
    const int64_t* __restrict__ probe_exact_threshold_sels,
    const int64_t* __restrict__ accepted_counts,
    float* __restrict__ final_max_logits,
    float* __restrict__ exact_value_thresholds,
    int64_t* __restrict__ exact_value_threshold_sels,
    int64_t heads,
    int64_t ranked,
    int64_t max_base,
    int64_t tail_blocks,
    int64_t threshold_steps,
    int64_t min_budget,
    int64_t max_budget_arg,
    int64_t granularity,
    float growth,
    float probe_scale) {
  int64_t head = blockIdx.x;
  if (head >= heads) {
    return;
  }
  extern __shared__ float shared[];
  int64_t max_budget = max_budget_arg;
  if (max_budget <= 0 || max_budget > ranked) {
    max_budget = ranked;
  }
  int64_t accepted = accepted_counts[head];
  accepted = max(static_cast<int64_t>(0), min(accepted, max_budget));

  const float* base_row = base_logits + head * max_base;
  const float* ranked_row = ranked_logits + head * ranked;
  int64_t base_count = max(static_cast<int64_t>(0), min(base_counts[head], max_base));

  float local_max = -INFINITY;
  for (int64_t idx = threadIdx.x; idx < base_count; idx += blockDim.x) {
    local_max = fmaxf(local_max, base_row[idx]);
  }
  for (int64_t sel = threadIdx.x; sel < accepted; sel += blockDim.x) {
    local_max = fmaxf(local_max, ranked_row[sel]);
  }
  for (int64_t block = threadIdx.x; block < tail_blocks; block += blockDim.x) {
    local_max = fmaxf(local_max, tail_partial_max[head * tail_blocks + block]);
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
    final_max_logits[head] = shared[0];
    float threshold = INFINITY;
    int64_t threshold_sel = -1;
    if (threshold_steps > 0 && probe_exact_thresholds != nullptr && probe_exact_threshold_sels != nullptr) {
      int64_t chosen_step = threshold_steps - 1;
      for (int64_t step = 0; step < threshold_steps; ++step) {
        int64_t tail_budget = 0;
        int64_t probe_budget = 0;
        geometric_budget_pair_for_step_device(
            step,
            min_budget,
            max_budget,
            granularity,
            growth,
            probe_scale,
            &tail_budget,
            &probe_budget);
        if (probe_budget >= accepted) {
          chosen_step = step;
          break;
        }
      }
      int64_t offset = head * threshold_steps + chosen_step;
      threshold = probe_exact_thresholds[offset];
      threshold_sel = probe_exact_threshold_sels[offset];
    }
    exact_value_thresholds[head] = threshold;
    exact_value_threshold_sels[head] = threshold_sel;
  }
}

