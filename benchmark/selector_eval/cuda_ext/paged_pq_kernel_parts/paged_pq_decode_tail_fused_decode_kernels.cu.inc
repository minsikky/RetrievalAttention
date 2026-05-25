template <typename vcode_t>
__global__ void gqa_decode_vpq_tail_from_scores_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    const float* __restrict__ values,
    const float* __restrict__ dense_pq_scores,
    const float* __restrict__ value_codebooks,
    const vcode_t* __restrict__ value_codes,
    const int64_t* __restrict__ page_starts,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_scores,
    float* __restrict__ outputs,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t pages,
    int64_t page_size,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    const int64_t* __restrict__ exact_value_counts,
    int64_t exact_value_top,
    float scale,
    float tail_blend) {
  int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= heads) {
    return;
  }
  int64_t kv_head = min(head / group_size, kv_heads - 1);
  const int64_t exact_value_top_row = exact_value_counts != nullptr ? exact_value_counts[head] : exact_value_top;
  const bool exact_values_by_selector_rank = exact_value_top_row < 0;
  const int64_t exact_value_limit = exact_values_by_selector_rank ? -exact_value_top_row : exact_value_top_row;
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

  extern __shared__ unsigned char shared_raw[];
  float* selected_logits = reinterpret_cast<float*>(shared_raw);
  const int64_t max_selected =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + ranked + 4;
  const bool exact_all_ranked_values = exact_value_limit > 0 && exact_value_limit >= ranked;
  const bool mixed_ranked_values = exact_value_limit > 0 && exact_value_limit < ranked;
  float* ranked_logits = selected_logits + max_selected;
  unsigned char* ranked_exact = reinterpret_cast<unsigned char*>(ranked_logits + ranked);
  const float* q = queries + head * dim;
  const int64_t* ranked_row = ranked_tokens + head * ranked;
  const float* ranked_score_row = ranked_scores + head * ranked;
  const float* score_row = dense_pq_scores + head * (pages * page_size);

  __shared__ int64_t selected_count;
  __shared__ float max_logit;
  __shared__ float denom;

  if (threadIdx.x == 0) {
    int64_t count = 0;
    float local_max = -INFINITY;
    if (mixed_ranked_values) {
      for (int64_t sel = 0; sel < ranked; ++sel) {
        ranked_logits[sel] = -INFINITY;
        ranked_exact[sel] = 0;
      }
    }
    for (int64_t token = 0; token < prefix_end; ++token) {
      const float* k = keys + (kv_head * total_tokens + token) * dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        dot += q[d] * k[d];
      }
      float logit = dot * scale;
      selected_logits[count++] = logit;
      local_max = fmaxf(local_max, logit);
    }
    for (int64_t token = base_tail_start; token < query_context_len; ++token) {
      const float* k = keys + (kv_head * total_tokens + token) * dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        dot += q[d] * k[d];
      }
      float logit = dot * scale;
      selected_logits[count++] = logit;
      local_max = fmaxf(local_max, logit);
    }
    for (int64_t sel = 0; sel < ranked; ++sel) {
      int64_t token = ranked_row[sel];
      float selector_score = ranked_score_row[sel];
      if (!isfinite(selector_score) || token < 0 || token >= query_context_len) {
        continue;
      }
      if (token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr)) {
        continue;
      }
      const float* k = keys + (kv_head * total_tokens + token) * dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        dot += q[d] * k[d];
      }
      float logit = dot * scale;
      selected_logits[count++] = logit;
      if (mixed_ranked_values) {
        ranked_logits[sel] = logit;
        if (exact_values_by_selector_rank && sel < exact_value_limit) {
          ranked_exact[sel] = 1;
        }
      }
      local_max = fmaxf(local_max, logit);
    }
    if (mixed_ranked_values && !exact_values_by_selector_rank) {
      for (int64_t rank = 0; rank < exact_value_limit; ++rank) {
        int64_t best_sel = -1;
        float best_logit = -INFINITY;
        for (int64_t sel = 0; sel < ranked; ++sel) {
          if (ranked_exact[sel]) {
            continue;
          }
          float logit = ranked_logits[sel];
          if (logit > best_logit || (logit == best_logit && (best_sel < 0 || sel < best_sel))) {
            best_logit = logit;
            best_sel = sel;
          }
        }
        if (best_sel >= 0 && isfinite(best_logit)) {
          ranked_exact[best_sel] = 1;
        }
      }
    }
    if (tail_blend > 0.0f) {
      for (int64_t page = 0; page < pages; ++page) {
        int64_t page_start = page_starts[page];
        bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
        if (!valid_page) {
          continue;
        }
        for (int64_t row = 0; row < page_size; ++row) {
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
          if (!is_ranked) {
            local_max = fmaxf(local_max, score_row[page * page_size + row] * scale);
          }
        }
      }
    }

    float local_denom = 0.0f;
    if (isfinite(local_max)) {
      for (int64_t idx = 0; idx < count; ++idx) {
        local_denom += expf(selected_logits[idx] - local_max);
      }
      if (tail_blend > 0.0f) {
        for (int64_t page = 0; page < pages; ++page) {
          int64_t page_start = page_starts[page];
          bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
          if (!valid_page) {
            continue;
          }
          for (int64_t row = 0; row < page_size; ++row) {
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
            if (!is_ranked) {
              local_denom += expf(score_row[page * page_size + row] * scale - local_max);
            }
          }
        }
      }
    }
    selected_count = count;
    max_logit = local_max;
    denom = fmaxf(local_denom, 1.0e-20f);
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float accum = 0.0f;
    int64_t idx = 0;
    if (isfinite(max_logit)) {
      for (int64_t token = 0; token < prefix_end; ++token) {
        float weight = expf(selected_logits[idx++] - max_logit) / denom;
        accum += weight * values[(kv_head * total_tokens + token) * dim + d];
      }
      for (int64_t token = base_tail_start; token < query_context_len; ++token) {
        float weight = expf(selected_logits[idx++] - max_logit) / denom;
        accum += weight * values[(kv_head * total_tokens + token) * dim + d];
      }
      for (int64_t sel = 0; sel < ranked; ++sel) {
        int64_t token = ranked_row[sel];
        float selector_score = ranked_score_row[sel];
        if (!isfinite(selector_score) || token < 0 || token >= query_context_len) {
          continue;
        }
        if (token_in_base_window(token, query_context_len, static_prefix, static_suffix, page_size, nullptr, nullptr)) {
          continue;
        }
        float weight = expf(selected_logits[idx++] - max_logit) / denom;
        bool exact_value = exact_all_ranked_values || (mixed_ranked_values && ranked_exact[sel] != 0);
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
          value = values[(kv_head * total_tokens + token) * dim + d];
        }
        accum += weight * value;
      }
      if (tail_blend > 0.0f) {
        for (int64_t page = 0; page < pages; ++page) {
          int64_t page_start = page_starts[page];
          bool valid_page = page_start >= prefix_end && page_start + page_size <= base_tail_start;
          if (!valid_page) {
            continue;
          }
          for (int64_t row = 0; row < page_size; ++row) {
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
            int64_t sub = d / value_subdim;
            int64_t sub_d = d - sub * value_subdim;
            float value = 0.0f;
            if (sub >= 0 && sub < value_subvecs) {
              int64_t vcode = static_cast<int64_t>(
                  value_codes[((kv_head * pages + page) * page_size + row) * value_subvecs + sub]);
              vcode = max(static_cast<int64_t>(0), min(vcode, value_centroids - 1));
              value = value_codebooks
                  [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + vcode) * value_subdim) +
                   sub_d];
            }
            float weight = expf(score_row[page * page_size + row] * scale - max_logit) / denom;
            accum += weight * value;
          }
        }
      }
    }
    outputs[head * dim + d] = accum;
  }
}

