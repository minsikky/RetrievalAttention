template <typename value_t, typename vcode_t>
__global__ void gqa_decode_geometric_final_output_codeweights_parallel_kernel(
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
    const float* __restrict__ final_max_logits,
    const float* __restrict__ confidence_max_logits,
    const float* __restrict__ tail_denoms,
    const float* __restrict__ code_weight_sums,
    const float* __restrict__ exact_value_thresholds,
    const int64_t* __restrict__ exact_value_threshold_sels,
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
    float scale,
    float tail_blend) {
  int64_t head = blockIdx.x;
  int64_t d = blockIdx.y;
  if (head >= heads || d >= dim) {
    return;
  }
  extern __shared__ float shared[];
  float* denom_parts = shared;
  float* selected_denom_parts = denom_parts + blockDim.x;
  float* numer_parts = selected_denom_parts + blockDim.x;
  float* selected_numer_parts = numer_parts + blockDim.x;

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
    threshold_logit_shared = exact_value_thresholds != nullptr ? exact_value_thresholds[head] : INFINITY;
    threshold_sel_shared = exact_value_threshold_sels != nullptr ? exact_value_threshold_sels[head] : -1;
  }
  __syncthreads();

  int64_t accepted = accepted_shared;
  float final_max = final_max_logits[head];
  float threshold_logit = threshold_logit_shared;
  int64_t threshold_sel = threshold_sel_shared;

  float denom_local = 0.0f;
  float selected_denom_local = 0.0f;
  float numer_local = 0.0f;
  float selected_numer_local = 0.0f;
  if (isfinite(final_max)) {
    for (int64_t idx = threadIdx.x; idx < base_count; idx += blockDim.x) {
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
      denom_local += w;
      selected_denom_local += w;
      numer_local += w * v;
      selected_numer_local += w * v;
    }

    float tail_scale = expf(confidence_max_logits[head] - final_max);
    if (threadIdx.x == 0) {
      denom_local += tail_denoms[head] * tail_scale;
    }
    if (value_subdim > 0) {
      int64_t sub = d / value_subdim;
      int64_t sub_d = d - sub * value_subdim;
      if (sub >= 0 && sub < value_subvecs) {
        int64_t total_codes = pages * value_centroids;
        for (int64_t idx = threadIdx.x; idx < total_codes; idx += blockDim.x) {
          int64_t page = idx / value_centroids;
          int64_t code = idx - page * value_centroids;
          float weight_sum =
              code_weight_sums[(((head * pages + page) * value_subvecs + sub) * value_centroids + code)];
          if (weight_sum == 0.0f) {
            continue;
          }
          float value = value_codebooks
              [((((kv_head * pages + page) * value_subvecs + sub) * value_centroids + code) * value_subdim) +
               sub_d];
          numer_local += (weight_sum * tail_scale) * value;
        }
      }
    }

    for (int64_t sel = threadIdx.x; sel < accepted; sel += blockDim.x) {
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
      denom_local += w;
      selected_denom_local += w;
      numer_local += w * selected_v;
      selected_numer_local += w * selected_v;

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
        denom_local -= pq_w;
        numer_local -= pq_w * pq_v;
      }
    }
  }

  denom_parts[threadIdx.x] = denom_local;
  selected_denom_parts[threadIdx.x] = selected_denom_local;
  numer_parts[threadIdx.x] = numer_local;
  selected_numer_parts[threadIdx.x] = selected_numer_local;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      denom_parts[threadIdx.x] += denom_parts[threadIdx.x + stride];
      selected_denom_parts[threadIdx.x] += selected_denom_parts[threadIdx.x + stride];
      numer_parts[threadIdx.x] += numer_parts[threadIdx.x + stride];
      selected_numer_parts[threadIdx.x] += selected_numer_parts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (!isfinite(final_max)) {
      outputs[head * dim + d] = 0.0f;
      return;
    }
    float full = numer_parts[0] / fmaxf(denom_parts[0], 1.0e-20f);
    if (tail_blend > 0.0f && tail_blend < 1.0f) {
      float selected_only = selected_numer_parts[0] / fmaxf(selected_denom_parts[0], 1.0e-20f);
      outputs[head * dim + d] = selected_only + tail_blend * (full - selected_only);
    } else {
      outputs[head * dim + d] = full;
    }
  }
}

__global__ void gqa_decode_proxy_gate_counts_kernel(
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ base_counts,
    const float* __restrict__ ranked_logits,
    const float* __restrict__ ranked_scores,
    int64_t* __restrict__ accepted_counts,
    int64_t heads,
    int64_t ranked,
    int64_t max_base,
    int64_t max_budget_arg,
    float scale,
    float proxy_mass_min,
    float proxy_tail_mass_max,
    float pq_corr_min,
    float pq_relrmse_max,
    bool calibrate) {
  int64_t head = blockIdx.x;
  if (head >= heads) {
    return;
  }
  extern __shared__ float shared[];
  float* red0 = shared;
  float* red1 = red0 + blockDim.x;
  float* red2 = red1 + blockDim.x;
  float* red3 = red2 + blockDim.x;
  float* red4 = red3 + blockDim.x;
  float* red5 = red4 + blockDim.x;
  float* red6 = red5 + blockDim.x;
  float* red7 = red6 + blockDim.x;
  const int64_t max_budget = max_budget_arg <= 0 ? ranked : min(max_budget_arg, ranked);
  if (max_budget <= 0) {
    if (threadIdx.x == 0) {
      accepted_counts[head] = 0;
    }
    return;
  }
  int64_t keep = accepted_counts[head];
  keep = max(static_cast<int64_t>(0), min(keep, max_budget));
  const float* base_row = base_logits + head * max_base;
  const int64_t base_count = max(static_cast<int64_t>(0), base_counts[head]);
  const float* exact_row = ranked_logits + head * ranked;
  const float* score_row = ranked_scores + head * ranked;

  float local_x = 0.0f;
  float local_y = 0.0f;
  float local_x2 = 0.0f;
  float local_y2 = 0.0f;
  float local_xy = 0.0f;
  float local_count = 0.0f;
  float local_base_max = -INFINITY;
  float local_selected_max = -INFINITY;
  for (int64_t idx = threadIdx.x; idx < base_count; idx += blockDim.x) {
    local_base_max = fmaxf(local_base_max, base_row[idx]);
  }
  for (int64_t sel = threadIdx.x; sel < max_budget; sel += blockDim.x) {
    float exact = exact_row[sel];
    float score = score_row[sel];
    if (sel < keep && isfinite(exact) && isfinite(score)) {
      float x = score * scale;
      float y = exact;
      local_x += x;
      local_y += y;
      local_x2 += x * x;
      local_y2 += y * y;
      local_xy += x * y;
      local_count += 1.0f;
      local_selected_max = fmaxf(local_selected_max, y);
    }
  }

  __shared__ float sx, sy, sx2, sy2, sxy, scount, base_max, selected_max;
  red0[threadIdx.x] = local_x;
  red1[threadIdx.x] = local_y;
  red2[threadIdx.x] = local_x2;
  red3[threadIdx.x] = local_y2;
  red4[threadIdx.x] = local_xy;
  red5[threadIdx.x] = local_count;
  red6[threadIdx.x] = local_base_max;
  red7[threadIdx.x] = local_selected_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      red0[threadIdx.x] += red0[threadIdx.x + stride];
      red1[threadIdx.x] += red1[threadIdx.x + stride];
      red2[threadIdx.x] += red2[threadIdx.x + stride];
      red3[threadIdx.x] += red3[threadIdx.x + stride];
      red4[threadIdx.x] += red4[threadIdx.x + stride];
      red5[threadIdx.x] += red5[threadIdx.x + stride];
      red6[threadIdx.x] = fmaxf(red6[threadIdx.x], red6[threadIdx.x + stride]);
      red7[threadIdx.x] = fmaxf(red7[threadIdx.x], red7[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sx = red0[0];
    sy = red1[0];
    sx2 = red2[0];
    sy2 = red3[0];
    sxy = red4[0];
    scount = red5[0];
    base_max = red6[0];
    selected_max = red7[0];
  }
  __syncthreads();

  __shared__ float fit_scale_shared, fit_bias_shared, corr_shared, relrmse_shared;
  if (threadIdx.x == 0) {
    float count = scount;
    float fit_scale = 1.0f;
    float fit_bias = 0.0f;
    float corr = 0.0f;
    float relrmse = INFINITY;
    if (calibrate && count >= 2.0f) {
      float inv_count = 1.0f / count;
      float mean_x = sx * inv_count;
      float mean_y = sy * inv_count;
      float var_x = fmaxf(sx2 * inv_count - mean_x * mean_x, 0.0f);
      float var_y = fmaxf(sy2 * inv_count - mean_y * mean_y, 0.0f);
      float cov = sxy * inv_count - mean_x * mean_y;
      bool flat_case = var_x <= 1.0e-20f;
      fit_scale = flat_case ? 0.0f : cov / fmaxf(var_x, 1.0e-20f);
      fit_bias = mean_y - fit_scale * mean_x;
      if (!flat_case && (!isfinite(fit_scale) || fit_scale <= 0.0f)) {
        fit_scale = 1.0f;
        fit_bias = 0.0f;
      }
      if (var_x > 1.0e-20f && var_y > 1.0e-20f) {
        corr = cov / sqrtf(var_x * var_y);
        if (!isfinite(corr)) corr = 0.0f;
      }
    }
    fit_scale_shared = fit_scale;
    fit_bias_shared = fit_bias;
    corr_shared = corr;
    relrmse_shared = relrmse;
  }
  __syncthreads();

  float fit_scale = fit_scale_shared;
  float fit_bias = fit_bias_shared;
  local_x = 0.0f;
  local_base_max = base_max;
  float local_base_sum = 0.0f;
  float local_selected_sum = 0.0f;
  float local_tail_max = -INFINITY;
  for (int64_t idx = threadIdx.x; idx < base_count; idx += blockDim.x) {
    float logit = base_row[idx];
    if (isfinite(logit) && isfinite(local_base_max)) {
      local_base_sum += expf(logit - local_base_max);
    }
  }
  for (int64_t sel = threadIdx.x; sel < max_budget; sel += blockDim.x) {
    float exact = exact_row[sel];
    float score = score_row[sel];
    if (!isfinite(score)) {
      continue;
    }
    if (sel < keep && isfinite(exact) && isfinite(selected_max)) {
      local_selected_sum += expf(exact - selected_max);
      float pred = fit_scale * (score * scale) + fit_bias;
      float err = pred - exact;
      local_x += err * err;
    } else if (sel >= keep) {
      local_tail_max = fmaxf(local_tail_max, fit_scale * (score * scale) + fit_bias);
    }
  }

  red0[threadIdx.x] = local_x;
  red1[threadIdx.x] = local_base_sum;
  red2[threadIdx.x] = local_selected_sum;
  red3[threadIdx.x] = local_tail_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      red0[threadIdx.x] += red0[threadIdx.x + stride];
      red1[threadIdx.x] += red1[threadIdx.x + stride];
      red2[threadIdx.x] += red2[threadIdx.x + stride];
      red3[threadIdx.x] = fmaxf(red3[threadIdx.x], red3[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && calibrate && scount >= 2.0f) {
    float mean_y = sy / fmaxf(scount, 1.0f);
    float var_y = fmaxf(sy2 / fmaxf(scount, 1.0f) - mean_y * mean_y, 0.0f);
    float rmse = sqrtf(red0[0] / fmaxf(scount, 1.0f));
    relrmse_shared = rmse / fmaxf(sqrtf(var_y), 1.0e-6f);
  }
  __syncthreads();

  __shared__ float base_lse, selected_lse_part, tail_max;
  if (threadIdx.x == 0) {
    base_lse = (isfinite(base_max) && red1[0] > 0.0f) ? base_max + logf(red1[0]) : -INFINITY;
    selected_lse_part = (isfinite(selected_max) && red2[0] > 0.0f) ? selected_max + logf(red2[0]) : -INFINITY;
    tail_max = red3[0];
  }
  __syncthreads();

  float local_tail_sum = 0.0f;
  for (int64_t sel = threadIdx.x; sel < max_budget; sel += blockDim.x) {
    float score = score_row[sel];
    if (sel >= keep && isfinite(score) && isfinite(tail_max)) {
      float pred = fit_scale * (score * scale) + fit_bias;
      local_tail_sum += expf(pred - tail_max);
    }
  }
  red0[threadIdx.x] = local_tail_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) red0[threadIdx.x] += red0[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float selected_lse = logaddexp_device(base_lse, selected_lse_part);
    float tail_lse = (isfinite(tail_max) && red0[0] > 0.0f) ? tail_max + logf(red0[0]) : -INFINITY;
    float total_lse = logaddexp_device(selected_lse, tail_lse);
    float selected_mass = isfinite(total_lse) ? expf(selected_lse - total_lse) : 0.0f;
    float tail_mass = isfinite(total_lse) ? expf(tail_lse - total_lse) : 0.0f;
    bool gate =
        selected_mass >= proxy_mass_min &&
        tail_mass <= proxy_tail_mass_max &&
        corr_shared >= pq_corr_min &&
        relrmse_shared <= pq_relrmse_max;
    if (!gate) {
      accepted_counts[head] = max_budget;
    }
  }
}

