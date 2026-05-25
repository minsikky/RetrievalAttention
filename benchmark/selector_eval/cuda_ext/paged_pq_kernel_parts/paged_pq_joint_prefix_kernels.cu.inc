template <typename order_t>
__global__ void joint_grouped_vprefix_outputs_kernel(
    const float* __restrict__ base_outputs,
    const float* __restrict__ probs,
    const float* __restrict__ residual_groups,
    const int64_t* __restrict__ row_group_ids,
    const order_t* __restrict__ exact_order,
    const int64_t* __restrict__ v_budgets,
    float* __restrict__ outputs,
    int64_t rows,
    int64_t context_len,
    int64_t max_exact,
    int64_t v_steps,
    int64_t dim) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * dim;
  if (linear >= total) {
    return;
  }

  int64_t d = linear % dim;
  int64_t row = linear / dim;
  int64_t group = row_group_ids[row];

  float acc = base_outputs[row * dim + d];
  int64_t prev_count = 0;
  for (int64_t vi = 0; vi < v_steps; ++vi) {
    int64_t exact_count = v_budgets[vi];
    exact_count = max(static_cast<int64_t>(0), min(exact_count, context_len));
    exact_count = min(exact_count, max_exact);
    if (exact_count < prev_count) {
      acc = base_outputs[row * dim + d];
      prev_count = 0;
    }
    for (int64_t j = prev_count; j < exact_count; ++j) {
      int64_t token = static_cast<int64_t>(exact_order[row * max_exact + j]);
      if (token >= 0 && token < context_len) {
        acc += probs[row * context_len + token] * residual_groups[(group * context_len + token) * dim + d];
      }
    }
    outputs[(row * v_steps + vi) * dim + d] = acc;
    prev_count = exact_count;
  }
}

template <typename order_t>
__global__ void joint_grouped_flat_vprefix_outputs_kernel(
    const float* __restrict__ base_outputs,
    const float* __restrict__ probs,
    const float* __restrict__ residual_groups,
    const order_t* __restrict__ exact_order,
    const int64_t* __restrict__ v_budgets,
    float* __restrict__ outputs,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t max_exact,
    int64_t v_steps,
    int64_t dim) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t rows = groups * k_count * heads;
  int64_t total = rows * dim;
  if (linear >= total) {
    return;
  }

  int64_t d = linear % dim;
  int64_t row = linear / dim;
  int64_t group = row / (k_count * heads);

  float acc = base_outputs[row * dim + d];
  int64_t prev_count = 0;
  for (int64_t vi = 0; vi < v_steps; ++vi) {
    int64_t exact_count = v_budgets[vi];
    exact_count = max(static_cast<int64_t>(0), min(exact_count, context_len));
    exact_count = min(exact_count, max_exact);
    if (exact_count < prev_count) {
      acc = base_outputs[row * dim + d];
      prev_count = 0;
    }
    for (int64_t j = prev_count; j < exact_count; ++j) {
      int64_t token = static_cast<int64_t>(exact_order[row * max_exact + j]);
      if (token >= 0 && token < context_len) {
        acc += probs[row * context_len + token] * residual_groups[(group * context_len + token) * dim + d];
      }
    }
    outputs[(row * v_steps + vi) * dim + d] = acc;
    prev_count = exact_count;
  }
}

template <typename order_t, int DIMS_PER_BLOCK>
__global__ void joint_grouped_flat_vprefix_interval_sums_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ residual_groups,
    const order_t* __restrict__ exact_order,
    const int64_t* __restrict__ v_budgets,
    float* __restrict__ interval_sums,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t max_exact,
    int64_t v_steps,
    int64_t dim) {
  constexpr int THREADS = 256;
  constexpr int TOKEN_LANES = THREADS / DIMS_PER_BLOCK;
  __shared__ float partial[DIMS_PER_BLOCK][TOKEN_LANES];

  int64_t rows = groups * k_count * heads;
  int64_t dim_tiles = (dim + DIMS_PER_BLOCK - 1) / DIMS_PER_BLOCK;
  int64_t block = static_cast<int64_t>(blockIdx.x);
  int64_t dim_tile = block % dim_tiles;
  block /= dim_tiles;
  int64_t vi = block % v_steps;
  int64_t row = block / v_steps;
  if (row >= rows || vi >= v_steps) {
    return;
  }

  int lane_dim = static_cast<int>(threadIdx.x % DIMS_PER_BLOCK);
  int lane_token = static_cast<int>(threadIdx.x / DIMS_PER_BLOCK);
  int64_t d = dim_tile * DIMS_PER_BLOCK + lane_dim;
  float acc = 0.0f;

  int64_t exact_count = v_budgets[vi];
  exact_count = max(static_cast<int64_t>(0), min(exact_count, context_len));
  exact_count = min(exact_count, max_exact);
  int64_t prev_count = 0;
  if (vi > 0) {
    prev_count = v_budgets[vi - 1];
    prev_count = max(static_cast<int64_t>(0), min(prev_count, context_len));
    prev_count = min(prev_count, max_exact);
  }
  int64_t start_count = exact_count < prev_count ? 0 : prev_count;

  if (d < dim) {
    int64_t group = row / (k_count * heads);
    for (int64_t j = start_count + lane_token; j < exact_count; j += TOKEN_LANES) {
      int64_t token = static_cast<int64_t>(exact_order[row * max_exact + j]);
      if (token >= 0 && token < context_len) {
        acc += probs[row * context_len + token] * residual_groups[(group * context_len + token) * dim + d];
      }
    }
  }

  partial[lane_dim][lane_token] = acc;
  __syncthreads();
  for (int stride = TOKEN_LANES >> 1; stride > 0; stride >>= 1) {
    if (lane_token < stride) {
      partial[lane_dim][lane_token] += partial[lane_dim][lane_token + stride];
    }
    __syncthreads();
  }
  if (lane_token == 0 && d < dim) {
    interval_sums[(row * v_steps + vi) * dim + d] = partial[lane_dim][0];
  }
}

template <typename order_t, int DIMS_PER_BLOCK>
__global__ void joint_grouped_flat_vprefix_interval_sums_strided_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ residual_groups,
    const order_t* __restrict__ exact_order,
    const int64_t* __restrict__ v_budgets,
    float* __restrict__ interval_sums,
    int64_t groups,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t prob_stride,
    int64_t max_exact,
    int64_t v_steps,
    int64_t dim) {
  constexpr int THREADS = 256;
  constexpr int TOKEN_LANES = THREADS / DIMS_PER_BLOCK;
  __shared__ float partial[DIMS_PER_BLOCK][TOKEN_LANES];

  int64_t rows = groups * k_count * heads;
  int64_t dim_tiles = (dim + DIMS_PER_BLOCK - 1) / DIMS_PER_BLOCK;
  int64_t block = static_cast<int64_t>(blockIdx.x);
  int64_t dim_tile = block % dim_tiles;
  block /= dim_tiles;
  int64_t vi = block % v_steps;
  int64_t row = block / v_steps;
  if (row >= rows || vi >= v_steps) {
    return;
  }

  int lane_dim = static_cast<int>(threadIdx.x % DIMS_PER_BLOCK);
  int lane_token = static_cast<int>(threadIdx.x / DIMS_PER_BLOCK);
  int64_t d = dim_tile * DIMS_PER_BLOCK + lane_dim;
  float acc = 0.0f;

  int64_t exact_count = v_budgets[vi];
  exact_count = max(static_cast<int64_t>(0), min(exact_count, context_len));
  exact_count = min(exact_count, max_exact);
  int64_t prev_count = 0;
  if (vi > 0) {
    prev_count = v_budgets[vi - 1];
    prev_count = max(static_cast<int64_t>(0), min(prev_count, context_len));
    prev_count = min(prev_count, max_exact);
  }
  int64_t start_count = exact_count < prev_count ? 0 : prev_count;

  if (d < dim) {
    int64_t group = row / (k_count * heads);
    for (int64_t j = start_count + lane_token; j < exact_count; j += TOKEN_LANES) {
      int64_t token = static_cast<int64_t>(exact_order[row * max_exact + j]);
      if (token >= 0 && token < context_len) {
        acc += probs[row * prob_stride + token] * residual_groups[(group * context_len + token) * dim + d];
      }
    }
  }

  partial[lane_dim][lane_token] = acc;
  __syncthreads();
  for (int stride = TOKEN_LANES >> 1; stride > 0; stride >>= 1) {
    if (lane_token < stride) {
      partial[lane_dim][lane_token] += partial[lane_dim][lane_token + stride];
    }
    __syncthreads();
  }
  if (lane_token == 0 && d < dim) {
    interval_sums[(row * v_steps + vi) * dim + d] = partial[lane_dim][0];
  }
}

__global__ void joint_grouped_flat_vprefix_prefix_intervals_kernel(
    const float* __restrict__ base_outputs,
    const float* __restrict__ interval_sums,
    const int64_t* __restrict__ v_budgets,
    float* __restrict__ outputs,
    int64_t rows,
    int64_t context_len,
    int64_t max_exact,
    int64_t v_steps,
    int64_t dim) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = rows * dim;
  if (linear >= total) {
    return;
  }

  int64_t d = linear % dim;
  int64_t row = linear / dim;
  float acc = base_outputs[row * dim + d];
  int64_t prev_count = 0;
  for (int64_t vi = 0; vi < v_steps; ++vi) {
    int64_t exact_count = v_budgets[vi];
    exact_count = max(static_cast<int64_t>(0), min(exact_count, context_len));
    exact_count = min(exact_count, max_exact);
    if (exact_count < prev_count) {
      acc = base_outputs[row * dim + d];
    }
    acc += interval_sums[(row * v_steps + vi) * dim + d];
    outputs[(row * v_steps + vi) * dim + d] = acc;
    prev_count = exact_count;
  }
}

template <typename order_t>
__device__ float joint_grouped_flat_output_value(
    const float* __restrict__ base_outputs,
    const float* __restrict__ probs,
    const float* __restrict__ residual_groups,
    const order_t* __restrict__ exact_order,
    const int64_t* __restrict__ v_budgets,
    int64_t group,
    int64_t ki,
    int64_t vi,
    int64_t head,
    int64_t k_count,
    int64_t heads,
    int64_t context_len,
    int64_t v_count,
    int64_t dim,
    int64_t d) {
  int64_t row = group * (k_count * heads) + ki * heads + head;
  float acc = base_outputs[row * dim + d];
  int64_t exact_count = v_budgets[vi];
  exact_count = max(static_cast<int64_t>(0), min(exact_count, context_len));
  for (int64_t j = 0; j < exact_count; ++j) {
    int64_t token = static_cast<int64_t>(exact_order[row * context_len + j]);
    if (token >= 0 && token < context_len) {
      acc += probs[row * context_len + token] * residual_groups[(group * context_len + token) * dim + d];
    }
  }
  return acc;
}

__device__ float joint_grouped_flat_interval_output_value(
    const float* __restrict__ base_outputs,
    const float* __restrict__ interval_sums,
    int64_t group,
    int64_t ki,
    int64_t vi,
    int64_t head,
    int64_t k_count,
    int64_t heads,
    int64_t v_count,
    int64_t dim,
    int64_t d) {
  int64_t row = group * (k_count * heads) + ki * heads + head;
  float acc = base_outputs[row * dim + d];
  for (int64_t vj = 0; vj <= vi && vj < v_count; ++vj) {
    acc += interval_sums[(row * v_count + vj) * dim + d];
  }
  return acc;
}

__global__ void joint_select_policy_kernel(
    const float* __restrict__ outputs,
    const float* __restrict__ k_mb,
    const float* __restrict__ v_mb,
    int64_t* __restrict__ final_indices,
    int64_t k_count,
    int64_t v_count,
    int64_t heads,
    int64_t dim,
    double threshold,
    int policy_id) {
  int64_t head = blockIdx.x;
  if (head >= heads || k_count <= 0 || v_count <= 0 || dim <= 0) {
    return;
  }
  extern __shared__ double joint_policy_shared[];
  double* k_diff_parts = joint_policy_shared;
  double* k_ref_parts = k_diff_parts + blockDim.x;
  double* v_diff_parts = k_ref_parts + blockDim.x;
  double* v_ref_parts = v_diff_parts + blockDim.x;
  __shared__ int64_t ki_shared;
  __shared__ int64_t vi_shared;
  __shared__ int stop_shared;
  if (threadIdx.x == 0) {
    ki_shared = 0;
    vi_shared = 0;
    stop_shared = 0;
  }
  __syncthreads();

  int64_t max_steps = k_count + v_count + 4;
  for (int64_t step = 0; step < max_steps; ++step) {
    int64_t ki = ki_shared;
    int64_t vi = vi_shared;
    bool k_can = (ki + 1) < k_count;
    bool v_can = (vi + 1) < v_count;
    double k_diff = 0.0;
    double k_ref = 0.0;
    double v_diff = 0.0;
    double v_ref = 0.0;
    for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
      int64_t cur_idx = (((ki * v_count + vi) * heads + head) * dim + d);
      double cur = static_cast<double>(outputs[cur_idx]);
      if (k_can) {
        int64_t next_idx = ((((ki + 1) * v_count + vi) * heads + head) * dim + d);
        double nxt = static_cast<double>(outputs[next_idx]);
        double delta = cur - nxt;
        k_diff += delta * delta;
        k_ref += nxt * nxt;
      }
      if (v_can) {
        int64_t next_idx = (((ki * v_count + (vi + 1)) * heads + head) * dim + d);
        double nxt = static_cast<double>(outputs[next_idx]);
        double delta = cur - nxt;
        v_diff += delta * delta;
        v_ref += nxt * nxt;
      }
    }
    k_diff_parts[threadIdx.x] = k_diff;
    k_ref_parts[threadIdx.x] = k_ref;
    v_diff_parts[threadIdx.x] = v_diff;
    v_ref_parts[threadIdx.x] = v_ref;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        k_diff_parts[threadIdx.x] += k_diff_parts[threadIdx.x + stride];
        k_ref_parts[threadIdx.x] += k_ref_parts[threadIdx.x + stride];
        v_diff_parts[threadIdx.x] += v_diff_parts[threadIdx.x + stride];
        v_ref_parts[threadIdx.x] += v_ref_parts[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      double k_delta = k_can ? sqrt(fmax(k_diff_parts[0], 0.0)) / fmax(sqrt(fmax(k_ref_parts[0], 0.0)), 1.0e-20) : 0.0;
      double v_delta = v_can ? sqrt(fmax(v_diff_parts[0], 0.0)) / fmax(sqrt(fmax(v_ref_parts[0], 0.0)), 1.0e-20) : 0.0;
      bool k_bad = k_can && k_delta > threshold;
      bool v_bad = v_can && v_delta > threshold;
      if (!k_bad && !v_bad) {
        stop_shared = 1;
      } else {
        int action = 0;  // 0 = K, 1 = V
        if (policy_id == 0) {
          action = k_bad ? 0 : 1;
        } else if (policy_id == 1) {
          action = v_bad ? 1 : 0;
        } else if (policy_id == 2) {
          int preferred = (static_cast<int>(step) % 2 == 0) ? 0 : 1;
          if (preferred == 0 && k_bad) {
            action = 0;
          } else if (preferred == 1 && v_bad) {
            action = 1;
          } else {
            action = v_bad ? 1 : 0;
          }
        } else if (policy_id == 3) {
          int preferred = (static_cast<int>(step) % 2 == 0) ? 1 : 0;
          if (preferred == 1 && v_bad) {
            action = 1;
          } else if (preferred == 0 && k_bad) {
            action = 0;
          } else {
            action = k_bad ? 0 : 1;
          }
        } else {
          double extra_k = k_can ? static_cast<double>(k_mb[ki + 1] - k_mb[ki]) : INFINITY;
          double extra_v = v_can ? static_cast<double>(v_mb[vi + 1] - v_mb[vi]) : INFINITY;
          double k_gain = k_bad ? k_delta / fmax(extra_k, 1.0e-9) : -1.0;
          double v_gain = v_bad ? v_delta / fmax(extra_v, 1.0e-9) : -1.0;
          action = (k_gain >= v_gain) ? 0 : 1;
        }
        if (action == 0 && k_can) {
          ki_shared = ki + 1;
        } else if (action == 1 && v_can) {
          vi_shared = vi + 1;
        } else {
          stop_shared = 1;
        }
      }
    }
    __syncthreads();
    if (stop_shared != 0) {
      break;
    }
  }
  if (threadIdx.x == 0) {
    final_indices[head * 2] = ki_shared;
    final_indices[head * 2 + 1] = vi_shared;
  }
}

