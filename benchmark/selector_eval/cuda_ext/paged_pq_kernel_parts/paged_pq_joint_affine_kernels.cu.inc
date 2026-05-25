__global__ void joint_segment_offsets_kernel(
    int64_t* __restrict__ offsets,
    int64_t rows,
    int64_t context_len) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx > rows) {
    return;
  }
  offsets[idx] = idx * context_len;
}

__global__ void joint_fill_int_kernel(
    int* __restrict__ data,
    int64_t count,
    int value) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = value;
  }
}

__global__ void joint_fill_byte_kernel(
    unsigned char* __restrict__ data,
    int64_t count,
    unsigned char value) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = value;
  }
}

__global__ void joint_selected_mask_set_base_kernel(
    unsigned char* __restrict__ selected_mask,
    const int64_t* __restrict__ base_tokens,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t base_count) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * base_count;
  if (linear >= total) {
    return;
  }
  int64_t base_i = linear % base_count;
  int64_t tmp = linear / base_count;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t token = base_tokens[base_i];
  if (token >= 0 && token < context_len) {
    selected_mask[(ki * heads + head) * context_len + token] = static_cast<unsigned char>(1);
  }
}

__global__ void joint_selected_mask_set_ranked_kernel(
    unsigned char* __restrict__ selected_mask,
    const int64_t* __restrict__ ranked_prefix_tokens,
    const int64_t* __restrict__ k_take_counts,
    int64_t k_count,
    int64_t heads,
    int64_t max_rank_take,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = k_count * heads * max_rank_take;
  if (linear >= total) {
    return;
  }
  int64_t rank_i = linear % max_rank_take;
  int64_t tmp = linear / max_rank_take;
  int64_t head = tmp % heads;
  int64_t ki = tmp / heads;
  int64_t take = k_take_counts[ki];
  if (rank_i >= take) {
    return;
  }
  int64_t token = ranked_prefix_tokens[head * max_rank_take + rank_i];
  if (token >= 0 && token < context_len) {
    selected_mask[(ki * heads + head) * context_len + token] = static_cast<unsigned char>(1);
  }
}

__global__ void joint_indexed_token_to_pos_kernel(
    int* __restrict__ token_to_indexed,
    const int64_t* __restrict__ indexed_tokens,
    int64_t indexed_count,
    int64_t context_len) {
  int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= indexed_count) {
    return;
  }
  int64_t token = indexed_tokens[j];
  if (token >= 0 && token < context_len) {
    token_to_indexed[token] = static_cast<int>(j);
  }
}

__global__ void joint_base_mask_set_kernel(
    unsigned char* __restrict__ base_mask,
    const int64_t* __restrict__ base_tokens,
    int64_t base_count,
    int64_t context_len) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= base_count) {
    return;
  }
  int64_t token = base_tokens[idx];
  if (token >= 0 && token < context_len) {
    base_mask[token] = static_cast<unsigned char>(1);
  }
}

__global__ void joint_rank_position_scatter_kernel(
    int* __restrict__ rank_pos,
    const int64_t* __restrict__ ranked_prefix_tokens,
    int64_t heads,
    int64_t max_rank_take,
    int64_t context_len) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = heads * max_rank_take;
  if (linear >= total) {
    return;
  }
  int64_t rank_i = linear % max_rank_take;
  int64_t head = linear / max_rank_take;
  int64_t token = ranked_prefix_tokens[head * max_rank_take + rank_i];
  if (token >= 0 && token < context_len) {
    rank_pos[head * context_len + token] = static_cast<int>(rank_i);
  }
}

__global__ void joint_affine_selected_fit_kernel(
    const float* __restrict__ pq_logits,
    const float* __restrict__ y_indexed,
    const int64_t* __restrict__ indexed_tokens,
    const unsigned char* __restrict__ selected_mask,
    float* __restrict__ fit_scale,
    float* __restrict__ fit_bias,
    int64_t k_count,
    int64_t heads,
    int64_t indexed_count,
    int64_t context_len,
    float pq_scale,
    bool calibrate) {
  int64_t kh = blockIdx.x;
  int64_t rows = k_count * heads;
  if (kh >= rows) {
    return;
  }
  extern __shared__ float shared[];
  float* sx_parts = shared;
  float* sy_parts = sx_parts + blockDim.x;
  float* sx2_parts = sy_parts + blockDim.x;
  float* sxy_parts = sx2_parts + blockDim.x;
  float* count_parts = sxy_parts + blockDim.x;

  int64_t head = kh % heads;
  float sx = 0.0f;
  float sy = 0.0f;
  float sx2 = 0.0f;
  float sxy = 0.0f;
  float count = 0.0f;
  if (calibrate) {
    for (int64_t j = threadIdx.x; j < indexed_count; j += blockDim.x) {
      int64_t token = indexed_tokens[j];
      if (token < 0 || token >= context_len) {
        continue;
      }
      if (selected_mask[kh * context_len + token] == static_cast<unsigned char>(0)) {
        continue;
      }
      float x = pq_logits[head * indexed_count + j] * pq_scale;
      float y = y_indexed[head * indexed_count + j];
      sx += x;
      sy += y;
      sx2 += x * x;
      sxy += x * y;
      count += 1.0f;
    }
  }
  sx_parts[threadIdx.x] = sx;
  sy_parts[threadIdx.x] = sy;
  sx2_parts[threadIdx.x] = sx2;
  sxy_parts[threadIdx.x] = sxy;
  count_parts[threadIdx.x] = count;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sx_parts[threadIdx.x] += sx_parts[threadIdx.x + stride];
      sy_parts[threadIdx.x] += sy_parts[threadIdx.x + stride];
      sx2_parts[threadIdx.x] += sx2_parts[threadIdx.x + stride];
      sxy_parts[threadIdx.x] += sxy_parts[threadIdx.x + stride];
      count_parts[threadIdx.x] += count_parts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float scale = 1.0f;
    float bias = 0.0f;
    float n = count_parts[0];
    if (calibrate && n >= 2.0f) {
      float inv_n = 1.0f / n;
      float mean_x = sx_parts[0] * inv_n;
      float mean_y = sy_parts[0] * inv_n;
      float var_x = fmaxf(sx2_parts[0] * inv_n - mean_x * mean_x, 0.0f);
      float cov = sxy_parts[0] * inv_n - mean_x * mean_y;
      if (var_x <= 1.0e-20f) {
        scale = 0.0f;
        bias = mean_y;
      } else {
        float fitted_scale = cov / fmaxf(var_x, 1.0e-20f);
        float fitted_bias = mean_y - fitted_scale * mean_x;
        if (isfinite(fitted_scale) && fitted_scale > 0.0f) {
          scale = fitted_scale;
          bias = fitted_bias;
        }
      }
    }
    fit_scale[kh] = scale;
    fit_bias[kh] = bias;
  }
}

__global__ void joint_affine_selected_fit_rankpos_kernel(
    const float* __restrict__ pq_logits,
    const float* __restrict__ y_indexed,
    const int64_t* __restrict__ indexed_tokens,
    const unsigned char* __restrict__ base_mask,
    const int* __restrict__ rank_pos,
    const int64_t* __restrict__ k_take_counts,
    float* __restrict__ fit_scale,
    float* __restrict__ fit_bias,
    int64_t k_count,
    int64_t heads,
    int64_t indexed_count,
    int64_t context_len,
    float pq_scale,
    bool calibrate) {
  int64_t kh = blockIdx.x;
  int64_t rows = k_count * heads;
  if (kh >= rows) {
    return;
  }
  extern __shared__ float shared[];
  float* sx_parts = shared;
  float* sy_parts = sx_parts + blockDim.x;
  float* sx2_parts = sy_parts + blockDim.x;
  float* sxy_parts = sx2_parts + blockDim.x;
  float* count_parts = sxy_parts + blockDim.x;

  int64_t head = kh % heads;
  int64_t ki = kh / heads;
  int64_t take = k_take_counts[ki];
  float sx = 0.0f;
  float sy = 0.0f;
  float sx2 = 0.0f;
  float sxy = 0.0f;
  float count = 0.0f;
  if (calibrate) {
    for (int64_t j = threadIdx.x; j < indexed_count; j += blockDim.x) {
      int64_t token = indexed_tokens[j];
      if (token < 0 || token >= context_len) {
        continue;
      }
      bool selected = base_mask[token] != static_cast<unsigned char>(0);
      int pos = rank_pos[head * context_len + token];
      selected = selected || (pos >= 0 && static_cast<int64_t>(pos) < take);
      if (!selected) {
        continue;
      }
      float x = pq_logits[head * indexed_count + j] * pq_scale;
      float y = y_indexed[head * indexed_count + j];
      sx += x;
      sy += y;
      sx2 += x * x;
      sxy += x * y;
      count += 1.0f;
    }
  }
  sx_parts[threadIdx.x] = sx;
  sy_parts[threadIdx.x] = sy;
  sx2_parts[threadIdx.x] = sx2;
  sxy_parts[threadIdx.x] = sxy;
  count_parts[threadIdx.x] = count;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sx_parts[threadIdx.x] += sx_parts[threadIdx.x + stride];
      sy_parts[threadIdx.x] += sy_parts[threadIdx.x + stride];
      sx2_parts[threadIdx.x] += sx2_parts[threadIdx.x + stride];
      sxy_parts[threadIdx.x] += sxy_parts[threadIdx.x + stride];
      count_parts[threadIdx.x] += count_parts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float scale = 1.0f;
    float bias = 0.0f;
    float n = count_parts[0];
    if (calibrate && n >= 2.0f) {
      float inv_n = 1.0f / n;
      float mean_x = sx_parts[0] * inv_n;
      float mean_y = sy_parts[0] * inv_n;
      float var_x = fmaxf(sx2_parts[0] * inv_n - mean_x * mean_x, 0.0f);
      float cov = sxy_parts[0] * inv_n - mean_x * mean_y;
      if (var_x <= 1.0e-20f) {
        scale = 0.0f;
        bias = mean_y;
      } else {
        float fitted_scale = cov / fmaxf(var_x, 1.0e-20f);
        float fitted_bias = mean_y - fitted_scale * mean_x;
        if (isfinite(fitted_scale) && fitted_scale > 0.0f) {
          scale = fitted_scale;
          bias = fitted_bias;
        }
      }
    }
    fit_scale[kh] = scale;
    fit_bias[kh] = bias;
  }
}

__global__ void joint_affine_selected_fit_tokenlist_kernel(
    const float* __restrict__ pq_logits,
    const float* __restrict__ y_indexed,
    const int* __restrict__ token_to_indexed,
    const int64_t* __restrict__ base_tokens,
    const int64_t* __restrict__ ranked_prefix_tokens,
    const int64_t* __restrict__ k_take_counts,
    float* __restrict__ fit_scale,
    float* __restrict__ fit_bias,
    int64_t k_count,
    int64_t heads,
    int64_t indexed_count,
    int64_t context_len,
    int64_t base_count,
    int64_t max_rank_take,
    float pq_scale,
    bool calibrate) {
  int64_t kh = blockIdx.x;
  int64_t rows = k_count * heads;
  if (kh >= rows) {
    return;
  }
  extern __shared__ float shared[];
  float* sx_parts = shared;
  float* sy_parts = sx_parts + blockDim.x;
  float* sx2_parts = sy_parts + blockDim.x;
  float* sxy_parts = sx2_parts + blockDim.x;
  float* count_parts = sxy_parts + blockDim.x;

  int64_t head = kh % heads;
  int64_t ki = kh / heads;
  int64_t take = k_take_counts[ki];
  take = max(static_cast<int64_t>(0), min(take, max_rank_take));
  float sx = 0.0f;
  float sy = 0.0f;
  float sx2 = 0.0f;
  float sxy = 0.0f;
  float count = 0.0f;
  if (calibrate) {
    const int64_t selected_count = base_count + take;
    for (int64_t idx = threadIdx.x; idx < selected_count; idx += blockDim.x) {
      int64_t token = -1;
      if (idx < base_count) {
        token = base_tokens[idx];
      } else {
        int64_t rank_i = idx - base_count;
        token = ranked_prefix_tokens[head * max_rank_take + rank_i];
      }
      if (token < 0 || token >= context_len) {
        continue;
      }
      int pos = token_to_indexed[token];
      if (pos < 0 || static_cast<int64_t>(pos) >= indexed_count) {
        continue;
      }
      float x = pq_logits[head * indexed_count + static_cast<int64_t>(pos)] * pq_scale;
      float y = y_indexed[head * indexed_count + static_cast<int64_t>(pos)];
      sx += x;
      sy += y;
      sx2 += x * x;
      sxy += x * y;
      count += 1.0f;
    }
  }
  sx_parts[threadIdx.x] = sx;
  sy_parts[threadIdx.x] = sy;
  sx2_parts[threadIdx.x] = sx2;
  sxy_parts[threadIdx.x] = sxy;
  count_parts[threadIdx.x] = count;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sx_parts[threadIdx.x] += sx_parts[threadIdx.x + stride];
      sy_parts[threadIdx.x] += sy_parts[threadIdx.x + stride];
      sx2_parts[threadIdx.x] += sx2_parts[threadIdx.x + stride];
      sxy_parts[threadIdx.x] += sxy_parts[threadIdx.x + stride];
      count_parts[threadIdx.x] += count_parts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float scale = 1.0f;
    float bias = 0.0f;
    float n = count_parts[0];
    if (calibrate && n >= 2.0f) {
      float inv_n = 1.0f / n;
      float mean_x = sx_parts[0] * inv_n;
      float mean_y = sy_parts[0] * inv_n;
      float var_x = fmaxf(sx2_parts[0] * inv_n - mean_x * mean_x, 0.0f);
      float cov = sxy_parts[0] * inv_n - mean_x * mean_y;
      if (var_x <= 1.0e-20f) {
        scale = 0.0f;
        bias = mean_y;
      } else {
        float fitted_scale = cov / fmaxf(var_x, 1.0e-20f);
        float fitted_bias = mean_y - fitted_scale * mean_x;
        if (isfinite(fitted_scale) && fitted_scale > 0.0f) {
          scale = fitted_scale;
          bias = fitted_bias;
        }
      }
    }
    fit_scale[kh] = scale;
    fit_bias[kh] = bias;
  }
}

__global__ void joint_affine_selected_fit_sparse_tokenlist_kernel(
    const float* __restrict__ pq_logits,
    const int* __restrict__ token_to_indexed,
    const int64_t* __restrict__ base_tokens,
    const float* __restrict__ base_logits,
    const int64_t* __restrict__ ranked_prefix_tokens,
    const float* __restrict__ ranked_logits,
    const int64_t* __restrict__ k_take_counts,
    float* __restrict__ fit_scale,
    float* __restrict__ fit_bias,
    int64_t k_count,
    int64_t heads,
    int64_t indexed_count,
    int64_t context_len,
    int64_t base_count,
    int64_t max_rank_take,
    float pq_scale,
    bool calibrate) {
  int64_t kh = blockIdx.x;
  int64_t rows = k_count * heads;
  if (kh >= rows) {
    return;
  }
  extern __shared__ float shared[];
  float* sx_parts = shared;
  float* sy_parts = sx_parts + blockDim.x;
  float* sx2_parts = sy_parts + blockDim.x;
  float* sxy_parts = sx2_parts + blockDim.x;
  float* count_parts = sxy_parts + blockDim.x;

  int64_t head = kh % heads;
  int64_t ki = kh / heads;
  int64_t take = k_take_counts[ki];
  take = max(static_cast<int64_t>(0), min(take, max_rank_take));
  float sx = 0.0f;
  float sy = 0.0f;
  float sx2 = 0.0f;
  float sxy = 0.0f;
  float count = 0.0f;
  if (calibrate) {
    const int64_t selected_count = base_count + take;
    for (int64_t idx = threadIdx.x; idx < selected_count; idx += blockDim.x) {
      int64_t token = -1;
      float y = 0.0f;
      if (idx < base_count) {
        token = base_tokens[idx];
        y = base_logits[head * base_count + idx];
      } else {
        int64_t rank_i = idx - base_count;
        token = ranked_prefix_tokens[head * max_rank_take + rank_i];
        y = ranked_logits[head * max_rank_take + rank_i];
      }
      if (token < 0 || token >= context_len || !isfinite(y)) {
        continue;
      }
      int pos = token_to_indexed[token];
      if (pos < 0 || static_cast<int64_t>(pos) >= indexed_count) {
        continue;
      }
      float x = pq_logits[head * indexed_count + static_cast<int64_t>(pos)] * pq_scale;
      sx += x;
      sy += y;
      sx2 += x * x;
      sxy += x * y;
      count += 1.0f;
    }
  }
  sx_parts[threadIdx.x] = sx;
  sy_parts[threadIdx.x] = sy;
  sx2_parts[threadIdx.x] = sx2;
  sxy_parts[threadIdx.x] = sxy;
  count_parts[threadIdx.x] = count;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sx_parts[threadIdx.x] += sx_parts[threadIdx.x + stride];
      sy_parts[threadIdx.x] += sy_parts[threadIdx.x + stride];
      sx2_parts[threadIdx.x] += sx2_parts[threadIdx.x + stride];
      sxy_parts[threadIdx.x] += sxy_parts[threadIdx.x + stride];
      count_parts[threadIdx.x] += count_parts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float scale = 1.0f;
    float bias = 0.0f;
    float n = count_parts[0];
    if (calibrate && n >= 2.0f) {
      float inv_n = 1.0f / n;
      float mean_x = sx_parts[0] * inv_n;
      float mean_y = sy_parts[0] * inv_n;
      float var_x = fmaxf(sx2_parts[0] * inv_n - mean_x * mean_x, 0.0f);
      float cov = sxy_parts[0] * inv_n - mean_x * mean_y;
      if (var_x <= 1.0e-20f) {
        scale = 0.0f;
        bias = mean_y;
      } else {
        float fitted_scale = cov / fmaxf(var_x, 1.0e-20f);
        float fitted_bias = mean_y - fitted_scale * mean_x;
        if (isfinite(fitted_scale) && fitted_scale > 0.0f) {
          scale = fitted_scale;
          bias = fitted_bias;
        }
      }
    }
    fit_scale[kh] = scale;
    fit_bias[kh] = bias;
  }
}

