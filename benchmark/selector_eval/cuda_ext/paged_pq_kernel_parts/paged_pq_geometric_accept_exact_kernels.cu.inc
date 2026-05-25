template <typename value_t>
__global__ void gqa_decode_geometric_selected_exact_delta_kernel(
    const value_t* __restrict__ values,
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_logits,
    const float* __restrict__ ranked_weights,
    const int32_t* __restrict__ ranked_pages,
    const int32_t* __restrict__ ranked_rows,
    const float* __restrict__ approx_exact_thresholds,
    const int64_t* __restrict__ approx_exact_threshold_sels,
    const float* __restrict__ probe_exact_thresholds,
    const int64_t* __restrict__ probe_exact_threshold_sels,
    float* __restrict__ approx_exact_num_incs,
    float* __restrict__ probe_exact_num_incs,
    float* __restrict__ final_exact_num_incs,
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
    int64_t group_size,
    int64_t min_budget,
    int64_t max_budget_arg,
    int64_t granularity,
    float growth,
    float probe_scale,
    int64_t threshold_steps) {
  int64_t head = blockIdx.x;
  int64_t step = blockIdx.y;
  if (head >= heads || step >= threshold_steps || dim <= 0) {
    return;
  }
  extern __shared__ float shared[];
  float* approx_num = shared;
  float* probe_num = approx_num + dim;
  float* final_num = probe_num + dim;
  for (int64_t d = threadIdx.x; d < 3 * dim; d += blockDim.x) {
    shared[d] = 0.0f;
  }
  __syncthreads();

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
  const int32_t* ranked_row_pages = ranked_pages + head * ranked;
  const int32_t* ranked_row_rows = ranked_rows + head * ranked;

  for (int64_t sel = approx_keep + threadIdx.x; sel < tail_budget; sel += blockDim.x) {
    float selected_w = ranked_row_weights[sel];
    int64_t token = ranked_row_tokens[sel];
    if (selected_w == 0.0f || token < 0 || token >= total_tokens) {
      continue;
    }
    int64_t page = static_cast<int64_t>(ranked_row_pages[sel]);
    int64_t row = static_cast<int64_t>(ranked_row_rows[sel]);
    bool has_tail_page = page >= 0 && row >= 0 && page < pages && row < page_size;
    float logit = ranked_row_logits[sel];
    bool exact_value =
        selected_mass_exact_value_for_rank(logit, sel, approx_threshold, approx_threshold_sel);
    if (!exact_value && has_tail_page) {
      continue;
    }
    for (int64_t d = 0; d < dim; ++d) {
      float value = load_strided3_as_float(
          values,
          kv_head,
          token,
          d,
          value_stride_head,
          value_stride_token,
          value_stride_dim);
      atomicAdd(approx_num + d, selected_w * value);
    }
  }

  for (int64_t sel = probe_keep + threadIdx.x; sel < probe_budget; sel += blockDim.x) {
    float selected_w = ranked_row_weights[sel];
    int64_t token = ranked_row_tokens[sel];
    if (selected_w == 0.0f || token < 0 || token >= total_tokens) {
      continue;
    }
    int64_t page = static_cast<int64_t>(ranked_row_pages[sel]);
    int64_t row = static_cast<int64_t>(ranked_row_rows[sel]);
    bool has_tail_page = page >= 0 && row >= 0 && page < pages && row < page_size;
    float logit = ranked_row_logits[sel];
    bool exact_value =
        selected_mass_exact_value_for_rank(logit, sel, probe_threshold, probe_threshold_sel);
    if (!exact_value && has_tail_page) {
      continue;
    }
    for (int64_t d = 0; d < dim; ++d) {
      float value = load_strided3_as_float(
          values,
          kv_head,
          token,
          d,
          value_stride_head,
          value_stride_token,
          value_stride_dim);
      float weighted = selected_w * value;
      atomicAdd(probe_num + d, weighted);
      atomicAdd(final_num + d, weighted);
    }
  }
  __syncthreads();

  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    int64_t out_idx = (head * threshold_steps + step) * dim + d;
    approx_exact_num_incs[out_idx] = approx_num[d];
    probe_exact_num_incs[out_idx] = probe_num[d];
    final_exact_num_incs[out_idx] = final_num[d];
  }
}

__global__ void gqa_decode_geometric_selected_exact_list_kernel(
    const int64_t* __restrict__ ranked_tokens,
    const float* __restrict__ ranked_logits,
    const float* __restrict__ ranked_weights,
    const int32_t* __restrict__ ranked_pages,
    const int32_t* __restrict__ ranked_rows,
    const float* __restrict__ approx_exact_thresholds,
    const int64_t* __restrict__ approx_exact_threshold_sels,
    const float* __restrict__ probe_exact_thresholds,
    const int64_t* __restrict__ probe_exact_threshold_sels,
    int32_t* __restrict__ approx_exact_counts,
    int32_t* __restrict__ probe_exact_counts,
    int32_t* __restrict__ approx_exact_tokens,
    int32_t* __restrict__ probe_exact_tokens,
    float* __restrict__ approx_exact_weights,
    float* __restrict__ probe_exact_weights,
    int64_t heads,
    int64_t ranked,
    int64_t total_tokens,
    int64_t pages,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget_arg,
    int64_t granularity,
    float growth,
    float probe_scale,
    int64_t exact_list_capacity,
    int64_t threshold_steps) {
  int64_t head = blockIdx.x;
  int64_t step = blockIdx.y;
  if (head >= heads || step >= threshold_steps) {
    return;
  }
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

  int64_t count_idx = head * threshold_steps + step;
  if (threadIdx.x == 0) {
    approx_exact_counts[count_idx] = 0;
    probe_exact_counts[count_idx] = 0;
  }
  __syncthreads();

  float approx_threshold = approx_exact_thresholds[count_idx];
  int64_t approx_threshold_sel = approx_exact_threshold_sels[count_idx];
  float probe_threshold = probe_exact_thresholds[count_idx];
  int64_t probe_threshold_sel = probe_exact_threshold_sels[count_idx];
  const int64_t* ranked_row_tokens = ranked_tokens + head * ranked;
  const float* ranked_row_logits = ranked_logits + head * ranked;
  const float* ranked_row_weights = ranked_weights + head * ranked;
  const int32_t* ranked_row_pages = ranked_pages + head * ranked;
  const int32_t* ranked_row_rows = ranked_rows + head * ranked;
  int64_t capacity = exact_list_capacity;
  if (capacity <= 0 || capacity > max_budget) {
    capacity = max_budget;
  }
  int64_t list_base = (head * threshold_steps + step) * capacity;

  for (int64_t sel = approx_keep + threadIdx.x; sel < tail_budget; sel += blockDim.x) {
    float selected_w = ranked_row_weights[sel];
    int64_t token = ranked_row_tokens[sel];
    if (selected_w == 0.0f || token < 0 || token >= total_tokens) {
      continue;
    }
    int64_t page = static_cast<int64_t>(ranked_row_pages[sel]);
    int64_t row = static_cast<int64_t>(ranked_row_rows[sel]);
    bool has_tail_page = page >= 0 && row >= 0 && page < pages && row < page_size;
    float logit = ranked_row_logits[sel];
    bool exact_value =
        selected_mass_exact_value_for_rank(logit, sel, approx_threshold, approx_threshold_sel);
    if (!exact_value && has_tail_page) {
      continue;
    }
    int64_t out = static_cast<int64_t>(atomicAdd(approx_exact_counts + count_idx, 1));
    if (out < capacity) {
      approx_exact_tokens[list_base + out] = static_cast<int32_t>(token);
      approx_exact_weights[list_base + out] = selected_w;
    }
  }

  for (int64_t sel = probe_keep + threadIdx.x; sel < probe_budget; sel += blockDim.x) {
    float selected_w = ranked_row_weights[sel];
    int64_t token = ranked_row_tokens[sel];
    if (selected_w == 0.0f || token < 0 || token >= total_tokens) {
      continue;
    }
    int64_t page = static_cast<int64_t>(ranked_row_pages[sel]);
    int64_t row = static_cast<int64_t>(ranked_row_rows[sel]);
    bool has_tail_page = page >= 0 && row >= 0 && page < pages && row < page_size;
    float logit = ranked_row_logits[sel];
    bool exact_value =
        selected_mass_exact_value_for_rank(logit, sel, probe_threshold, probe_threshold_sel);
    if (!exact_value && has_tail_page) {
      continue;
    }
    int64_t out = static_cast<int64_t>(atomicAdd(probe_exact_counts + count_idx, 1));
    if (out < capacity) {
      probe_exact_tokens[list_base + out] = static_cast<int32_t>(token);
      probe_exact_weights[list_base + out] = selected_w;
    }
  }
}

__device__ __forceinline__ void geometric_budget_pair_for_step_device(
    int64_t step,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    float growth,
    float probe_scale,
    int64_t* __restrict__ tail_budget_out,
    int64_t* __restrict__ probe_budget_out) {
  int64_t tail_budget = min_budget;
  if (tail_budget < 0) {
    tail_budget = 0;
  }
  if (tail_budget > max_budget) {
    tail_budget = max_budget;
  }
  tail_budget = round_budget_up_device(tail_budget, granularity, max_budget);
  int64_t probe_budget = tail_budget;
  for (int64_t idx = 0; idx <= step; ++idx) {
    float target = fmaxf(static_cast<float>(tail_budget + granularity), probe_scale * static_cast<float>(tail_budget));
    probe_budget = round_budget_up_device(static_cast<int64_t>(ceilf(target)), granularity, max_budget);
    if (probe_budget < tail_budget) {
      probe_budget = tail_budget;
    }
    if (idx == step || probe_budget >= max_budget) {
      break;
    }
    float next_target = fmaxf(static_cast<float>(probe_budget + granularity), growth * static_cast<float>(probe_budget));
    int64_t next_budget = round_budget_up_device(static_cast<int64_t>(ceilf(next_target)), granularity, max_budget);
    if (next_budget <= probe_budget) {
      tail_budget = probe_budget;
      break;
    }
    tail_budget = next_budget;
  }
  *tail_budget_out = tail_budget;
  *probe_budget_out = probe_budget;
}

