template <typename value_t, typename vcode_t>
__global__ void gqa_decode_geometric_step_diff_codeweights_kernel(
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
    float* __restrict__ diff_squares,
    float* __restrict__ probe_squares,
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
    float scale,
    bool probe_includes_tail,
    int64_t threshold_steps) {
  int64_t head = blockIdx.x;
  int64_t step = blockIdx.y;
  int64_t d = blockIdx.z;
  if (head >= heads || step >= threshold_steps || d >= dim) {
    return;
  }
  extern __shared__ float shared[];
  float* approx_den_parts = shared;
  float* approx_num_parts = approx_den_parts + blockDim.x;
  float* probe_den_parts = approx_num_parts + blockDim.x;
  float* probe_num_parts = probe_den_parts + blockDim.x;
  __shared__ float base_den_shared;
  __shared__ float base_num_shared;
  __shared__ float tail_den_shared;
  __shared__ float tail_num_shared;

  int64_t max_budget = max_budget_arg;
  if (max_budget <= 0 || max_budget > ranked) {
    max_budget = ranked;
  }
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
  int64_t threshold_idx = head * threshold_steps + step;
  float approx_threshold = approx_exact_thresholds[threshold_idx];
  int64_t approx_threshold_sel = approx_exact_threshold_sels[threshold_idx];
  float probe_threshold = probe_exact_thresholds[threshold_idx];
  int64_t probe_threshold_sel = probe_exact_threshold_sels[threshold_idx];

  if (threadIdx.x == 0) {
    float base_den = 0.0f;
    float base_num = 0.0f;
    if (isfinite(global_max)) {
      for (int64_t idx = 0; idx < base_count; ++idx) {
        int64_t token = base_tokens_row[idx];
        float logit = base_logits_row[idx];
        if (token < 0 || token >= total_tokens || !isfinite(logit)) {
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
    }
    float tail_num = 0.0f;
    if (isfinite(global_max) && value_subdim > 0) {
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
    base_den_shared = base_den;
    base_num_shared = base_num;
    tail_den_shared = isfinite(global_max) ? tail_denoms[head] : 0.0f;
    tail_num_shared = tail_num;
  }
  __syncthreads();

  float approx_den_local = 0.0f;
  float approx_num_local = 0.0f;
  float probe_den_local = 0.0f;
  float probe_num_local = 0.0f;
  if (isfinite(global_max)) {
    for (int64_t sel = threadIdx.x; sel < tail_budget; sel += blockDim.x) {
      float logit = ranked_logit_row[sel];
      int64_t token = ranked_row[sel];
      if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
        continue;
      }
      float selected_w = expf(logit - global_max);
      bool exact_value = selected_mass_exact_value_for_rank(logit, sel, approx_threshold, approx_threshold_sel);
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
      approx_den_local += selected_w;
      approx_num_local += selected_w * selected_v;
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
        approx_den_local -= pq_w;
        approx_num_local -= pq_w * pq_v;
      }
    }
    for (int64_t sel = threadIdx.x; sel < probe_budget; sel += blockDim.x) {
      float logit = ranked_logit_row[sel];
      int64_t token = ranked_row[sel];
      if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
        continue;
      }
      float selected_w = expf(logit - global_max);
      bool exact_value = selected_mass_exact_value_for_rank(logit, sel, probe_threshold, probe_threshold_sel);
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
      probe_den_local += selected_w;
      probe_num_local += selected_w * selected_v;
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
          probe_den_local -= pq_w;
          probe_num_local -= pq_w * pq_v;
        }
      }
    }
  }

  approx_den_parts[threadIdx.x] = approx_den_local;
  approx_num_parts[threadIdx.x] = approx_num_local;
  probe_den_parts[threadIdx.x] = probe_den_local;
  probe_num_parts[threadIdx.x] = probe_num_local;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      approx_den_parts[threadIdx.x] += approx_den_parts[threadIdx.x + stride];
      approx_num_parts[threadIdx.x] += approx_num_parts[threadIdx.x + stride];
      probe_den_parts[threadIdx.x] += probe_den_parts[threadIdx.x + stride];
      probe_num_parts[threadIdx.x] += probe_num_parts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float approx_den = base_den_shared + tail_den_shared + approx_den_parts[0];
    float approx_num = base_num_shared + tail_num_shared + approx_num_parts[0];
    float probe_den = base_den_shared + (probe_includes_tail ? tail_den_shared : 0.0f) + probe_den_parts[0];
    float probe_num = base_num_shared + (probe_includes_tail ? tail_num_shared : 0.0f) + probe_num_parts[0];
    float approx_tail = approx_num / fmaxf(approx_den, 1.0e-20f);
    float probe_only = probe_num / fmaxf(probe_den, 1.0e-20f);
    float delta = approx_tail - probe_only;
    int64_t out_idx = (head * threshold_steps + step) * dim + d;
    diff_squares[out_idx] = delta * delta;
    probe_squares[out_idx] = probe_only * probe_only;
  }
}

template <typename value_t, typename vcode_t>
__global__ void gqa_decode_geometric_dim_scan_codeweights_kernel(
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
    const float* __restrict__ ranked_weights,
    const float* __restrict__ ranked_pq_weights,
    const int32_t* __restrict__ ranked_pages,
    const int32_t* __restrict__ ranked_rows,
    const float* __restrict__ approx_den_incs,
    const float* __restrict__ probe_den_incs,
    const float* __restrict__ final_den_incs,
    const float* __restrict__ approx_code_incs,
    const float* __restrict__ probe_code_incs,
    const float* __restrict__ final_code_incs,
    const float* __restrict__ approx_exact_num_incs,
    const float* __restrict__ probe_exact_num_incs,
    const float* __restrict__ final_exact_num_incs,
    const int32_t* __restrict__ approx_exact_counts,
    const int32_t* __restrict__ probe_exact_counts,
    const int32_t* __restrict__ approx_exact_tokens,
    const int32_t* __restrict__ probe_exact_tokens,
    const float* __restrict__ approx_exact_weights,
    const float* __restrict__ probe_exact_weights,
    const float* __restrict__ max_logits,
    const float* __restrict__ tail_denoms,
    const float* __restrict__ code_weight_sums,
    const float* __restrict__ approx_exact_thresholds,
    const int64_t* __restrict__ approx_exact_threshold_sels,
    const float* __restrict__ probe_exact_thresholds,
    const int64_t* __restrict__ probe_exact_threshold_sels,
    float* __restrict__ diff_squares,
    float* __restrict__ probe_squares,
    float* __restrict__ candidate_outputs,
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
    float scale,
    bool probe_includes_tail,
    bool write_candidate_outputs,
    bool use_selected_code_incs,
    bool use_selected_exact_num_incs,
    bool use_selected_exact_lists,
    int64_t exact_list_capacity,
    int64_t threshold_steps) {
  int64_t head = blockIdx.x;
  int64_t d = blockIdx.y;
  if (head >= heads || d >= dim || threshold_steps <= 0) {
    return;
  }
  extern __shared__ float shared[];
  float* den0 = shared;
  float* num0 = den0 + blockDim.x;
  float* den1 = num0 + blockDim.x;
  float* num1 = den1 + blockDim.x;
  float* den2 = num1 + blockDim.x;
  float* num2 = den2 + blockDim.x;

  int64_t max_budget = max_budget_arg;
  if (max_budget <= 0 || max_budget > ranked) {
    max_budget = ranked;
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
  int64_t base_count = max(static_cast<int64_t>(0), min(base_counts[head], max_base));
  const int64_t* ranked_row = ranked_tokens + head * ranked;
  const float* ranked_logit_row = ranked_logits + head * ranked;
  const float* ranked_weight_row = ranked_weights != nullptr ? ranked_weights + head * ranked : nullptr;
  const float* ranked_pq_weight_row = ranked_pq_weights != nullptr ? ranked_pq_weights + head * ranked : nullptr;
  const int32_t* ranked_page_row = ranked_pages != nullptr ? ranked_pages + head * ranked : nullptr;
  const int32_t* ranked_row_row = ranked_rows != nullptr ? ranked_rows + head * ranked : nullptr;
  const float* dense_row = dense_pq_scores + head * (pages * page_size);
  bool use_code_incs =
      use_selected_code_incs && approx_den_incs != nullptr && probe_den_incs != nullptr &&
      final_den_incs != nullptr && approx_code_incs != nullptr && probe_code_incs != nullptr &&
      final_code_incs != nullptr && ranked_weight_row != nullptr && ranked_pq_weight_row != nullptr &&
      ranked_page_row != nullptr && ranked_row_row != nullptr && value_subdim > 0 && value_subvecs > 0 &&
      value_centroids > 0;
  bool use_exact_num_incs =
      use_code_incs && use_selected_exact_num_incs && approx_exact_num_incs != nullptr &&
      probe_exact_num_incs != nullptr && final_exact_num_incs != nullptr;
  bool use_exact_lists =
      use_code_incs && use_selected_exact_lists && approx_exact_counts != nullptr &&
      probe_exact_counts != nullptr && approx_exact_tokens != nullptr && probe_exact_tokens != nullptr &&
      approx_exact_weights != nullptr && probe_exact_weights != nullptr;
  int64_t list_capacity = exact_list_capacity;
  if (list_capacity <= 0 || list_capacity > max_budget) {
    list_capacity = max_budget;
  }
  float global_max = max_logits[head];
  if (!isfinite(global_max)) {
    return;
  }

  float local_base_den = 0.0f;
  float local_base_num = 0.0f;
  for (int64_t idx = threadIdx.x; idx < base_count; idx += blockDim.x) {
    int64_t token = base_tokens_row[idx];
    float logit = base_logits_row[idx];
    if (token < 0 || token >= total_tokens || !isfinite(logit)) {
      continue;
    }
    float w = expf(logit - global_max);
    local_base_den += w;
    local_base_num += w * load_strided3_as_float(
                            values,
                            kv_head,
                            token,
                            d,
                            value_stride_head,
                            value_stride_token,
                            value_stride_dim);
  }

  float local_tail_num = 0.0f;
  if (value_subdim > 0) {
    int64_t sub = d / value_subdim;
    int64_t sub_d = d - sub * value_subdim;
    if (sub >= 0 && sub < value_subvecs) {
      int64_t total_codes = pages * value_centroids;
      for (int64_t idx = threadIdx.x; idx < total_codes; idx += blockDim.x) {
        int64_t page = idx / value_centroids;
        int64_t code = idx - page * value_centroids;
        float w = code_weight_sums[(((head * pages + page) * value_subvecs + sub) * value_centroids + code)];
        if (w == 0.0f) {
          continue;
        }
        float value = value_codebooks
            [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
             sub_d];
        local_tail_num += w * value;
      }
    }
  }

  den0[threadIdx.x] = local_base_den;
  num0[threadIdx.x] = local_base_num;
  den1[threadIdx.x] = local_tail_num;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      den0[threadIdx.x] += den0[threadIdx.x + stride];
      num0[threadIdx.x] += num0[threadIdx.x + stride];
      den1[threadIdx.x] += den1[threadIdx.x + stride];
    }
    __syncthreads();
  }

  __shared__ float approx_den_shared;
  __shared__ float approx_num_shared;
  __shared__ float probe_den_shared;
  __shared__ float probe_num_shared;
  __shared__ float final_den_shared;
  __shared__ float final_num_shared;
  if (threadIdx.x == 0) {
    float base_den = den0[0];
    float base_num = num0[0];
    float tail_den = tail_denoms[head];
    float tail_num = den1[0];
    approx_den_shared = base_den + tail_den;
    approx_num_shared = base_num + tail_num;
    probe_den_shared = probe_includes_tail ? (base_den + tail_den) : base_den;
    probe_num_shared = probe_includes_tail ? (base_num + tail_num) : base_num;
    final_den_shared = base_den + tail_den;
    final_num_shared = base_num + tail_num;
  }
  __syncthreads();

  int64_t approx_keep = 0;
  int64_t probe_keep = 0;
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
    int64_t thresh_idx = head * threshold_steps + step;
    float approx_threshold = approx_exact_thresholds[thresh_idx];
    int64_t approx_threshold_sel = approx_exact_threshold_sels[thresh_idx];
    float probe_threshold = probe_exact_thresholds[thresh_idx];
    int64_t probe_threshold_sel = probe_exact_threshold_sels[thresh_idx];
    float local_approx_code_num = 0.0f;
    float local_probe_code_num = 0.0f;
    float local_final_code_num = 0.0f;
    if (use_code_incs && value_subdim > 0) {
      int64_t sub = d / value_subdim;
      int64_t sub_d = d - sub * value_subdim;
      if (sub >= 0 && sub < value_subvecs) {
        int64_t total_codes = pages * value_centroids;
        int64_t code_base = (((head * threshold_steps + step) * pages) * value_subvecs + sub) * value_centroids;
        for (int64_t idx = threadIdx.x; idx < total_codes; idx += blockDim.x) {
          int64_t page = idx / value_centroids;
          int64_t code = idx - page * value_centroids;
          float value = value_codebooks
              [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
               sub_d];
          int64_t weight_idx = code_base + page * value_subvecs * value_centroids + code;
          local_approx_code_num += approx_code_incs[weight_idx] * value;
          local_probe_code_num += probe_code_incs[weight_idx] * value;
          if (write_candidate_outputs) {
            local_final_code_num += final_code_incs[weight_idx] * value;
          }
        }
      }
    }

    float local_approx_den = 0.0f;
    float local_approx_num = local_approx_code_num;
    int64_t approx_exact_count = use_exact_lists ? static_cast<int64_t>(approx_exact_counts[thresh_idx]) : 0;
    bool use_approx_exact_list = use_exact_lists && approx_exact_count <= list_capacity;
    if (use_approx_exact_list) {
      int64_t list_base = (head * threshold_steps + step) * list_capacity;
      int64_t exact_count = min(approx_exact_count, list_capacity);
      for (int64_t idx = threadIdx.x; idx < exact_count; idx += blockDim.x) {
        int64_t token = approx_exact_tokens[list_base + idx];
        float selected_w = approx_exact_weights[list_base + idx];
        if (selected_w == 0.0f || token < 0 || token >= total_tokens) {
          continue;
        }
        local_approx_num += selected_w * load_strided3_as_float(
                                          values,
                                          kv_head,
                                          token,
                                          d,
                                          value_stride_head,
                                          value_stride_token,
                                          value_stride_dim);
      }
    } else if (!use_exact_num_incs) {
    for (int64_t sel = approx_keep + threadIdx.x; sel < tail_budget; sel += blockDim.x) {
      float logit = ranked_logit_row[sel];
      int64_t token = ranked_row[sel];
      if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
        continue;
	      }
	      float selected_w = ranked_weight_row != nullptr ? ranked_weight_row[sel] : expf(logit - global_max);
	      bool exact_value = selected_mass_exact_value_for_rank(logit, sel, approx_threshold, approx_threshold_sel);
	      int64_t page = -1;
	      int64_t row = -1;
	      bool has_tail_page = false;
	      if (ranked_page_row != nullptr && ranked_row_row != nullptr) {
	        page = static_cast<int64_t>(ranked_page_row[sel]);
	        row = static_cast<int64_t>(ranked_row_row[sel]);
	        has_tail_page = page >= 0 && row >= 0;
	      } else {
	        has_tail_page = complete_page_for_token(
	            page_starts,
	            pages,
	            page_size,
	            token,
	            query_context_len,
	            prefix_end,
	            base_tail_start,
	            &page,
	            &row);
	      }
	      if (use_code_incs) {
	        if (exact_value || !has_tail_page) {
	          float selected_v = exact_value
	              ? load_strided3_as_float(
	                    values,
	                    kv_head,
	                    token,
	                    d,
	                    value_stride_head,
	                    value_stride_token,
	                    value_stride_dim)
	              : selected_value_for_rank_dim_explicit(
	                    values,
	                    value_codebooks,
	                    value_codes,
	                    page_starts,
	                    token,
	                    false,
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
	          local_approx_num += selected_w * selected_v;
	        }
	        continue;
	      }
	      float pq_w = 0.0f;
	      float pq_v = 0.0f;
	      if (has_tail_page) {
	        pq_w = ranked_pq_weight_row != nullptr
	            ? ranked_pq_weight_row[sel]
	            : expf(dense_row[page * page_size + row] * scale - global_max);
	        pq_v = load_vpq_value_for_dim(
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
	      float selected_v = 0.0f;
	      if (exact_value) {
	        selected_v = load_strided3_as_float(
	            values,
	            kv_head,
	            token,
	            d,
	            value_stride_head,
	            value_stride_token,
	            value_stride_dim);
	      } else if (has_tail_page) {
	        selected_v = pq_v;
	      } else {
	        selected_v = selected_value_for_rank_dim_explicit(
	            values,
	            value_codebooks,
	            value_codes,
	            page_starts,
	            token,
	            false,
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
	      local_approx_den += selected_w;
	      local_approx_num += selected_w * selected_v;
	      if (has_tail_page) {
	        local_approx_den -= pq_w;
	        local_approx_num -= pq_w * pq_v;
	      }
    }
    }

    float local_probe_den = 0.0f;
    float local_probe_num = local_probe_code_num;
    float local_final_den = 0.0f;
    float local_final_num = local_final_code_num;
    int64_t probe_exact_count = use_exact_lists ? static_cast<int64_t>(probe_exact_counts[thresh_idx]) : 0;
    bool use_probe_exact_list = use_exact_lists && probe_exact_count <= list_capacity;
    if (use_probe_exact_list) {
      int64_t list_base = (head * threshold_steps + step) * list_capacity;
      int64_t exact_count = min(probe_exact_count, list_capacity);
      for (int64_t idx = threadIdx.x; idx < exact_count; idx += blockDim.x) {
        int64_t token = probe_exact_tokens[list_base + idx];
        float selected_w = probe_exact_weights[list_base + idx];
        if (selected_w == 0.0f || token < 0 || token >= total_tokens) {
          continue;
        }
        float selected_v = load_strided3_as_float(
            values,
            kv_head,
            token,
            d,
            value_stride_head,
            value_stride_token,
            value_stride_dim);
        local_probe_num += selected_w * selected_v;
        if (write_candidate_outputs) {
          local_final_num += selected_w * selected_v;
        }
      }
    } else if (!use_exact_num_incs) {
    for (int64_t sel = probe_keep + threadIdx.x; sel < probe_budget; sel += blockDim.x) {
      float logit = ranked_logit_row[sel];
      int64_t token = ranked_row[sel];
      if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
        continue;
	      }
	      float selected_w = ranked_weight_row != nullptr ? ranked_weight_row[sel] : expf(logit - global_max);
	      bool exact_value = selected_mass_exact_value_for_rank(logit, sel, probe_threshold, probe_threshold_sel);
	      int64_t page = -1;
	      int64_t row = -1;
	      bool has_tail_page = false;
	      if (ranked_page_row != nullptr && ranked_row_row != nullptr) {
	        page = static_cast<int64_t>(ranked_page_row[sel]);
	        row = static_cast<int64_t>(ranked_row_row[sel]);
	        has_tail_page = page >= 0 && row >= 0;
	      } else {
	        has_tail_page = complete_page_for_token(
	            page_starts,
	            pages,
	            page_size,
	            token,
	            query_context_len,
	            prefix_end,
	            base_tail_start,
	            &page,
	            &row);
	      }
	      if (use_code_incs) {
	        if (exact_value || !has_tail_page) {
	          float selected_v = exact_value
	              ? load_strided3_as_float(
	                    values,
	                    kv_head,
	                    token,
	                    d,
	                    value_stride_head,
	                    value_stride_token,
	                    value_stride_dim)
	              : selected_value_for_rank_dim_explicit(
	                    values,
	                    value_codebooks,
	                    value_codes,
	                    page_starts,
	                    token,
	                    false,
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
	          local_probe_num += selected_w * selected_v;
	          if (write_candidate_outputs) {
	            local_final_num += selected_w * selected_v;
	          }
	        }
	        continue;
	      }
	      float pq_w = 0.0f;
	      float pq_v = 0.0f;
	      if (has_tail_page) {
	        pq_w = ranked_pq_weight_row != nullptr
	            ? ranked_pq_weight_row[sel]
	            : expf(dense_row[page * page_size + row] * scale - global_max);
	        pq_v = load_vpq_value_for_dim(
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
	      float selected_v = 0.0f;
	      if (exact_value) {
	        selected_v = load_strided3_as_float(
	            values,
	            kv_head,
	            token,
	            d,
	            value_stride_head,
	            value_stride_token,
	            value_stride_dim);
	      } else if (has_tail_page) {
	        selected_v = pq_v;
	      } else {
	        selected_v = selected_value_for_rank_dim_explicit(
	            values,
	            value_codebooks,
	            value_codes,
	            page_starts,
	            token,
	            false,
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
	      local_probe_den += selected_w;
	      local_probe_num += selected_w * selected_v;
	      if (write_candidate_outputs) {
	        local_final_den += selected_w;
	        local_final_num += selected_w * selected_v;
	      }
	      if (probe_includes_tail && has_tail_page) {
	          local_probe_den -= pq_w;
	          local_probe_num -= pq_w * pq_v;
	      }
	      if (write_candidate_outputs && !probe_includes_tail && has_tail_page) {
	          local_final_den -= pq_w;
	          local_final_num -= pq_w * pq_v;
	      }
	    }
    }

    den0[threadIdx.x] = local_approx_den;
    num0[threadIdx.x] = local_approx_num;
    den1[threadIdx.x] = local_probe_den;
    num1[threadIdx.x] = local_probe_num;
    den2[threadIdx.x] = local_final_den;
    num2[threadIdx.x] = local_final_num;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        den0[threadIdx.x] += den0[threadIdx.x + stride];
        num0[threadIdx.x] += num0[threadIdx.x + stride];
        den1[threadIdx.x] += den1[threadIdx.x + stride];
        num1[threadIdx.x] += num1[threadIdx.x + stride];
        den2[threadIdx.x] += den2[threadIdx.x + stride];
        num2[threadIdx.x] += num2[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      if (use_code_incs) {
        int64_t inc_idx = head * threshold_steps + step;
        approx_den_shared += approx_den_incs[inc_idx];
        probe_den_shared += probe_den_incs[inc_idx];
        if (write_candidate_outputs) {
          final_den_shared += final_den_incs[inc_idx];
        }
      } else {
        approx_den_shared += den0[0];
        probe_den_shared += den1[0];
        if (write_candidate_outputs) {
          final_den_shared += probe_includes_tail ? den1[0] : den2[0];
        }
      }
      float approx_num_add = num0[0];
      float probe_num_add = num1[0];
      float final_num_add = write_candidate_outputs ? (probe_includes_tail ? num1[0] : num2[0]) : 0.0f;
      if (use_exact_num_incs) {
        int64_t num_idx = (head * threshold_steps + step) * dim + d;
        approx_num_add += approx_exact_num_incs[num_idx];
        probe_num_add += probe_exact_num_incs[num_idx];
        if (write_candidate_outputs) {
          final_num_add += final_exact_num_incs[num_idx];
        }
      }
      approx_num_shared += approx_num_add;
      probe_num_shared += probe_num_add;
      if (write_candidate_outputs) {
        final_num_shared += final_num_add;
      }
      float approx_tail = approx_num_shared / fmaxf(approx_den_shared, 1.0e-20f);
      float probe_only = probe_num_shared / fmaxf(probe_den_shared, 1.0e-20f);
      float delta = approx_tail - probe_only;
      int64_t out_idx = (head * threshold_steps + step) * dim + d;
      diff_squares[out_idx] = delta * delta;
      probe_squares[out_idx] = probe_only * probe_only;
      if (write_candidate_outputs && candidate_outputs != nullptr) {
        candidate_outputs[out_idx] = final_num_shared / fmaxf(final_den_shared, 1.0e-20f);
      }
    }
    __syncthreads();
    approx_keep = tail_budget;
    probe_keep = probe_budget;
  }
}

