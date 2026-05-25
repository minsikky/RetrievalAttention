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
__global__ void gqa_decode_tail_agg_output_parallel_kernel(
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
    float tail_blend) {
  int64_t head = blockIdx.x;
  int64_t d = blockIdx.y;
  if (head >= heads || d >= dim) {
    return;
  }
  extern __shared__ float shared[];
  float* selected_parts = shared;
  float* tail_parts = selected_parts + blockDim.x;

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

  float selected_local = 0.0f;
  float tail_local = 0.0f;
  if (isfinite(max_logit)) {
    for (int64_t idx = threadIdx.x; idx < base_count; idx += blockDim.x) {
      int64_t token = base_tokens_row[idx];
      if (token < 0 || token >= total_tokens) {
        continue;
      }
      float logit = base_logits_row[idx];
      if (!isfinite(logit)) {
        continue;
      }
      float weight = expf(logit - max_logit);
      selected_local += weight *
          load_strided3_as_float(values, kv_head, token, d, value_stride_head, value_stride_token, value_stride_dim);
    }
    for (int64_t sel = threadIdx.x; sel < ranked; sel += blockDim.x) {
      float logit = ranked_logits_row[sel];
      if (!isfinite(logit)) {
        continue;
      }
      int64_t token = ranked_row[sel];
      if (token < 0 || token >= total_tokens) {
        continue;
      }
      float weight = expf(logit - max_logit);
      bool exact_value = ranked_exact_row[sel] != 0;
      float value = selected_value_for_rank_dim_explicit(
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
      selected_local += weight * value;
    }
    if (tail_blend > 0.0f && pages > 0 && value_subvecs > 0 && value_subdim > 0) {
      int64_t sub = d / value_subdim;
      int64_t sub_d = d - sub * value_subdim;
      if (sub >= 0 && sub < value_subvecs) {
        int64_t total_codes = pages * value_centroids;
        for (int64_t idx = threadIdx.x; idx < total_codes; idx += blockDim.x) {
          int64_t page = idx / value_centroids;
          int64_t code = idx - page * value_centroids;
          int64_t page_start = page_starts[page];
          bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
          if (!valid_page) {
            continue;
          }
          float weight_sum =
              code_weight_sums[(((head * pages + page) * value_subvecs + sub) * value_centroids + code)];
          if (weight_sum == 0.0f) {
            continue;
          }
          float value = value_codebooks
              [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
               sub_d];
          tail_local += weight_sum * value;
        }
      }
    }
  }
  selected_parts[threadIdx.x] = selected_local;
  tail_parts[threadIdx.x] = tail_local;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      selected_parts[threadIdx.x] += selected_parts[threadIdx.x + stride];
      tail_parts[threadIdx.x] += tail_parts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float full = (selected_parts[0] + tail_parts[0]) / denoms[head];
    if (tail_blend > 0.0f && tail_blend < 1.0f) {
      float selected_only = selected_parts[0] / selected_denoms[head];
      outputs[head * dim + d] = selected_only + tail_blend * (full - selected_only);
    } else {
      outputs[head * dim + d] = full;
    }
  }
}

template <typename value_t, typename vcode_t>
void launch_gqa_decode_tail_agg_output(
    const value_t* values,
    const float* value_codebooks,
    const vcode_t* value_codes,
    const int64_t* page_starts,
    const int64_t* base_tokens,
    const float* base_logits,
    const int64_t* base_counts,
    const int64_t* ranked_tokens,
    const float* ranked_logits,
    const unsigned char* ranked_exact,
    const float* max_logits,
    const float* denoms,
    const float* selected_denoms,
    const float* code_weight_sums,
    float* outputs,
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
    float tail_blend,
    cudaStream_t stream) {
  const int threads = 256;
  const char* parallel_env = std::getenv("SELECTOR_PQ_TAIL_OUTPUT_PARALLEL");
  const bool use_parallel =
      parallel_env != nullptr && parallel_env[0] != '\0' && parallel_env[0] != '0';
  if (use_parallel) {
    dim3 output_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(dim));
    gqa_decode_tail_agg_output_parallel_kernel<value_t, vcode_t>
        <<<output_grid, threads, threads * 2 * sizeof(float), stream>>>(
            values,
            value_codebooks,
            value_codes,
            page_starts,
            base_tokens,
            base_logits,
            base_counts,
            ranked_tokens,
            ranked_logits,
            ranked_exact,
            max_logits,
            denoms,
            selected_denoms,
            code_weight_sums,
            outputs,
            heads,
            kv_heads,
            ranked,
            dim,
            total_tokens,
            value_stride_head,
            value_stride_token,
            value_stride_dim,
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
            tail_blend);
    return;
  }
  const int64_t output_total = heads * dim;
  const int output_blocks = static_cast<int>((output_total + threads - 1) / threads);
  gqa_decode_tail_agg_output_kernel<value_t, vcode_t>
      <<<output_blocks, threads, 0, stream>>>(
          values,
          value_codebooks,
          value_codes,
          page_starts,
          base_tokens,
          base_logits,
          base_counts,
          ranked_tokens,
          ranked_logits,
          ranked_exact,
          max_logits,
          denoms,
          selected_denoms,
          code_weight_sums,
          outputs,
          heads,
          kv_heads,
          ranked,
          dim,
          total_tokens,
          value_stride_head,
          value_stride_token,
          value_stride_dim,
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
          scale,
          tail_blend);
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

