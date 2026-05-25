template <typename value_t, typename vcode_t>
__device__ float gqa_causal_geometric_output_dim_bitmap(
    const value_t* __restrict__ values,
    const float* __restrict__ dense_row,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int32_t* __restrict__ base_tokens,
    const float* __restrict__ base_logits,
    int64_t base_count,
    const int64_t* __restrict__ ranked_tokens_row,
    const float* __restrict__ ranked_logits,
    const unsigned int* __restrict__ selected_bitmap,
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

  int64_t keep_clamped = keep;
  if (keep_clamped < 0) {
    keep_clamped = 0;
  }
  if (keep_clamped > ranked) {
    keep_clamped = ranked;
  }

  float max_logit = -INFINITY;
  for (int64_t idx = 0; idx < base_count; ++idx) {
    max_logit = fmaxf(max_logit, base_logits[idx]);
  }
  for (int64_t sel = 0; sel < keep_clamped; ++sel) {
    max_logit = fmaxf(max_logit, ranked_logits[sel]);
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
        int64_t ordinal = page * page_size + row;
        bool selected = (selected_bitmap[ordinal >> 5] & (1u << (ordinal & 31))) != 0u;
        if (selected) {
          continue;
        }
        max_logit = fmaxf(max_logit, dense_row[ordinal] * scale);
      }
    }
  }
  if (!isfinite(max_logit)) {
    return 0.0f;
  }

  float denom = 0.0f;
  float accum = 0.0f;
  for (int64_t idx = 0; idx < base_count; ++idx) {
    int64_t token = static_cast<int64_t>(base_tokens[idx]);
    if (token < 0 || token >= total_tokens) {
      continue;
    }
    float weight = expf(base_logits[idx] - max_logit);
    denom += weight;
    accum += weight * load_strided3_as_float(
                          values,
                          kv_head,
                          token,
                          dim_idx,
                          value_stride_head,
                          value_stride_token,
                          value_stride_dim);
  }
  for (int64_t sel = 0; sel < keep_clamped; ++sel) {
    int64_t token = ranked_tokens_row[sel];
    float logit = ranked_logits[sel];
    if (!isfinite(logit) || token < 0 || token >= query_context_len || token >= total_tokens) {
      continue;
    }
    float weight = expf(logit - max_logit);
    denom += weight;
    accum += weight * load_strided3_as_float(
                          values,
                          kv_head,
                          token,
                          dim_idx,
                          value_stride_head,
                          value_stride_token,
                          value_stride_dim);
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
          int64_t ordinal = page * page_size + row;
          bool selected = (selected_bitmap[ordinal >> 5] & (1u << (ordinal & 31))) != 0u;
          if (selected) {
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
          float weight = expf(dense_row[ordinal] * scale - max_logit);
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

