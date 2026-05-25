__global__ void selected_mass_thresholds_from_topk_kernel(
    const float* __restrict__ top_logits,
    const int64_t* __restrict__ top_order,
    const float* __restrict__ prefix_lse,
    const int64_t* __restrict__ prefix_valid_counts,
    const float* __restrict__ base_lse,
    const int64_t* __restrict__ budgets,
    float* __restrict__ thresholds,
    int64_t* __restrict__ threshold_sels,
    unsigned char* __restrict__ sufficient,
    int64_t heads,
    int64_t k_top,
    int64_t rank,
    int64_t steps,
    float exact_mass,
    int64_t min_top) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  int64_t step = static_cast<int64_t>(blockIdx.y);
  if (head >= heads || step >= steps) {
    return;
  }
  constexpr int kThreads = 256;
  __shared__ float scan_weights[kThreads];
  __shared__ int scan_counts[kThreads];
  __shared__ int hit_pos;

  int64_t budget = budgets[step];
  budget = max(static_cast<int64_t>(0), min(rank, budget));
  int64_t out_idx = head * steps + step;
  if (threadIdx.x == 0) {
    thresholds[out_idx] = INFINITY;
    threshold_sels[out_idx] = -1;
    sufficient[out_idx] = 1;
  }
  __syncthreads();
  if (budget <= 0 || rank <= 0 || k_top <= 0 || (exact_mass <= 0.0f && min_top <= 0)) {
    return;
  }

  int64_t lse_idx = head * rank + budget - 1;
  float ranked_lse = prefix_lse[lse_idx];
  int64_t valid_count = prefix_valid_counts[lse_idx];
  valid_count = max(static_cast<int64_t>(0), min(budget, valid_count));
  int64_t min_count = min(budget, max(static_cast<int64_t>(0), min_top));
  if (valid_count <= 0 && min_count <= 0) {
    return;
  }
  float total_lse = logaddexp_device(base_lse[head], ranked_lse);
  if (!isfinite(total_lse)) {
    return;
  }
  float target = fminf(fmaxf(exact_mass, 0.0f), 1.0f - 1.0e-7f);
  float base_mass = isfinite(base_lse[head]) ? expf(base_lse[head] - total_lse) : 0.0f;
  bool mass_already_met = target <= 0.0f || base_mass >= target;

  float running_weight = 0.0f;
  int running_count = 0;
  bool found = false;
  int64_t top_base = head * k_top;
  for (int64_t chunk = 0; chunk < k_top; chunk += kThreads) {
    int64_t j = chunk + threadIdx.x;
    float weight = 0.0f;
    int count = 0;
    if (j < k_top) {
      int64_t sel = top_order[top_base + j];
      float logit = top_logits[top_base + j];
      if (sel >= 0 && sel < budget && isfinite(logit)) {
        weight = expf(logit - total_lse);
        count = 1;
      }
    }
    scan_weights[threadIdx.x] = weight;
    scan_counts[threadIdx.x] = count;
    __syncthreads();
    for (int offset = 1; offset < kThreads; offset <<= 1) {
      float add_weight = 0.0f;
      int add_count = 0;
      if (threadIdx.x >= offset) {
        add_weight = scan_weights[threadIdx.x - offset];
        add_count = scan_counts[threadIdx.x - offset];
      }
      __syncthreads();
      scan_weights[threadIdx.x] += add_weight;
      scan_counts[threadIdx.x] += add_count;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      hit_pos = kThreads;
    }
    __syncthreads();
    if (count > 0) {
      int cumulative_count = running_count + scan_counts[threadIdx.x];
      float cumulative_mass = base_mass + running_weight + scan_weights[threadIdx.x];
      bool enough_by_mass = mass_already_met || cumulative_mass >= target;
      bool enough_by_count = cumulative_count >= min_count;
      bool last_valid_fallback = valid_count > 0 && cumulative_count >= valid_count;
      if ((enough_by_mass && enough_by_count) || last_valid_fallback) {
        atomicMin(&hit_pos, static_cast<int>(threadIdx.x));
      }
    }
    __syncthreads();
    if (hit_pos < kThreads) {
      if (threadIdx.x == 0) {
        int64_t hit_j = chunk + static_cast<int64_t>(hit_pos);
        thresholds[out_idx] = top_logits[top_base + hit_j];
        threshold_sels[out_idx] = top_order[top_base + hit_j];
      }
      found = true;
    }
    float chunk_weight = scan_weights[kThreads - 1];
    int chunk_count = scan_counts[kThreads - 1];
    running_weight += chunk_weight;
    running_count += chunk_count;
    __syncthreads();
    if (found) {
      break;
    }
  }
  if (!found && threadIdx.x == 0) {
    sufficient[out_idx] = 0;
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

