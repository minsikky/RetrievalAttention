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

__global__ void gqa_decode_rank_weight_metadata_kernel(
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_logits,
    const float* __restrict__ dense_pq_scores,
    const int64_t* __restrict__ page_starts,
    const float* __restrict__ max_logits,
    float* __restrict__ ranked_weights,
    float* __restrict__ ranked_pq_weights,
    int32_t* __restrict__ ranked_pages,
    int32_t* __restrict__ ranked_rows,
    int64_t heads,
    int64_t ranked,
    int64_t pages,
    int64_t page_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    float scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * ranked;
  if (idx >= total) {
    return;
  }
  int64_t head = idx / ranked;
  int64_t sel = idx - head * ranked;
  int64_t token = ranked_tokens[idx];
  float logit = ranked_logits[idx];
  float global_max = max_logits[head];
  float selected_w = 0.0f;
  float pq_w = 0.0f;
  int32_t page_i32 = -1;
  int32_t row_i32 = -1;
  if (isfinite(global_max) && isfinite(logit) && token >= 0 && token < query_context_len) {
    selected_w = expf(logit - global_max);
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
      pq_w = expf(dense_pq_scores[head * (pages * page_size) + page * page_size + row] * scale - global_max);
      page_i32 = static_cast<int32_t>(page);
      row_i32 = static_cast<int32_t>(row);
    }
  }
  ranked_weights[idx] = selected_w;
  ranked_pq_weights[idx] = pq_w;
  ranked_pages[idx] = page_i32;
  ranked_rows[idx] = row_i32;
}

__device__ __forceinline__ void geometric_budget_pair_for_step_device(
    int64_t step,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    float growth,
    float probe_scale,
    int64_t* __restrict__ tail_budget_out,
    int64_t* __restrict__ probe_budget_out);

template <typename vcode_t>
__global__ void gqa_decode_geometric_selected_code_delta_kernel(
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_logits,
    const float* __restrict__ ranked_weights,
    const float* __restrict__ ranked_pq_weights,
    const int32_t* __restrict__ ranked_pages,
    const int32_t* __restrict__ ranked_rows,
    const vcode_t* __restrict__ value_codes,
    const float* __restrict__ approx_exact_thresholds,
    const int64_t* __restrict__ approx_exact_threshold_sels,
    const float* __restrict__ probe_exact_thresholds,
    const int64_t* __restrict__ probe_exact_threshold_sels,
    float* __restrict__ approx_den_incs,
    float* __restrict__ probe_den_incs,
    float* __restrict__ final_den_incs,
    float* __restrict__ approx_code_incs,
    float* __restrict__ probe_code_incs,
    float* __restrict__ final_code_incs,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t group_size,
    int64_t min_budget,
    int64_t max_budget_arg,
    int64_t granularity,
    float growth,
    float probe_scale,
    bool probe_includes_tail,
    int64_t threshold_steps) {
  int64_t head = blockIdx.x;
  int64_t step = blockIdx.y;
  if (head >= heads || step >= threshold_steps || value_subvecs <= 0 || value_centroids <= 0) {
    return;
  }
  extern __shared__ float shared[];
  float* approx_den_parts = shared;
  float* probe_den_parts = approx_den_parts + blockDim.x;
  float* final_den_parts = probe_den_parts + blockDim.x;

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
  int64_t approx_keep = 0;
  int64_t probe_keep = 0;
  if (step > 0) {
    int64_t prev_tail = 0;
    int64_t prev_probe = 0;
    geometric_budget_pair_for_step_device(
        step - 1,
        min_budget,
        max_budget,
        granularity,
        growth,
        probe_scale,
        &prev_tail,
        &prev_probe);
    approx_keep = prev_tail;
    probe_keep = prev_probe;
  }

  int64_t threshold_idx = head * threshold_steps + step;
  float approx_threshold = approx_exact_thresholds[threshold_idx];
  int64_t approx_threshold_sel = approx_exact_threshold_sels[threshold_idx];
  float probe_threshold = probe_exact_thresholds[threshold_idx];
  int64_t probe_threshold_sel = probe_exact_threshold_sels[threshold_idx];
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  const int64_t* ranked_row_tokens = ranked_tokens + head * ranked;
  const float* ranked_row_logits = ranked_logits + head * ranked;
  const float* ranked_row_weights = ranked_weights + head * ranked;
  const float* ranked_row_pq_weights = ranked_pq_weights + head * ranked;
  const int32_t* ranked_row_pages = ranked_pages + head * ranked;
  const int32_t* ranked_row_rows = ranked_rows + head * ranked;

  float approx_den_local = 0.0f;
  float probe_den_local = 0.0f;
  float final_den_local = 0.0f;
  for (int64_t sel = approx_keep + threadIdx.x; sel < tail_budget; sel += blockDim.x) {
    float selected_w = ranked_row_weights[sel];
    if (selected_w == 0.0f) {
      continue;
    }
    int64_t page = static_cast<int64_t>(ranked_row_pages[sel]);
    int64_t row = static_cast<int64_t>(ranked_row_rows[sel]);
    bool has_tail_page = page >= 0 && row >= 0 && page < pages && row < page_size;
    float pq_w = has_tail_page ? ranked_row_pq_weights[sel] : 0.0f;
    approx_den_local += selected_w - pq_w;
    if (!has_tail_page) {
      continue;
    }
    float logit = ranked_row_logits[sel];
    bool exact_value = selected_mass_exact_value_for_rank(logit, sel, approx_threshold, approx_threshold_sel);
    float selected_delta = exact_value ? 0.0f : selected_w;
    float code_delta = selected_delta - pq_w;
    if (code_delta == 0.0f) {
      continue;
    }
    for (int64_t sub = 0; sub < value_subvecs; ++sub) {
      int64_t code = static_cast<int64_t>(
          value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
      code = max(static_cast<int64_t>(0), min(code, value_centroids - 1));
      int64_t out_idx = (((head * threshold_steps + step) * pages + page) * value_subvecs + sub) *
                            value_centroids +
                        code;
      atomicAdd(approx_code_incs + out_idx, code_delta);
    }
  }

  for (int64_t sel = probe_keep + threadIdx.x; sel < probe_budget; sel += blockDim.x) {
    float selected_w = ranked_row_weights[sel];
    if (selected_w == 0.0f) {
      continue;
    }
    int64_t page = static_cast<int64_t>(ranked_row_pages[sel]);
    int64_t row = static_cast<int64_t>(ranked_row_rows[sel]);
    bool has_tail_page = page >= 0 && row >= 0 && page < pages && row < page_size;
    float pq_w = has_tail_page ? ranked_row_pq_weights[sel] : 0.0f;
    probe_den_local += selected_w - (probe_includes_tail ? pq_w : 0.0f);
    final_den_local += selected_w - (!probe_includes_tail ? pq_w : 0.0f);
    if (!has_tail_page) {
      continue;
    }
    float logit = ranked_row_logits[sel];
    bool exact_value = selected_mass_exact_value_for_rank(logit, sel, probe_threshold, probe_threshold_sel);
    float selected_delta = exact_value ? 0.0f : selected_w;
    float probe_code_delta = selected_delta - (probe_includes_tail ? pq_w : 0.0f);
    float final_code_delta = selected_delta - (!probe_includes_tail ? pq_w : 0.0f);
    for (int64_t sub = 0; sub < value_subvecs; ++sub) {
      int64_t code = static_cast<int64_t>(
          value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
      code = max(static_cast<int64_t>(0), min(code, value_centroids - 1));
      int64_t out_idx = (((head * threshold_steps + step) * pages + page) * value_subvecs + sub) *
                            value_centroids +
                        code;
      if (probe_code_delta != 0.0f) {
        atomicAdd(probe_code_incs + out_idx, probe_code_delta);
      }
      if (final_code_delta != 0.0f) {
        atomicAdd(final_code_incs + out_idx, final_code_delta);
      }
    }
  }

  approx_den_parts[threadIdx.x] = approx_den_local;
  probe_den_parts[threadIdx.x] = probe_den_local;
  final_den_parts[threadIdx.x] = final_den_local;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      approx_den_parts[threadIdx.x] += approx_den_parts[threadIdx.x + stride];
      probe_den_parts[threadIdx.x] += probe_den_parts[threadIdx.x + stride];
      final_den_parts[threadIdx.x] += final_den_parts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    int64_t idx = head * threshold_steps + step;
    approx_den_incs[idx] = approx_den_parts[0];
    probe_den_incs[idx] = probe_den_parts[0];
    final_den_incs[idx] = final_den_parts[0];
  }
}

