template <typename order_t>
__global__ void joint_select_policy_grouped_risk_kernel(
    const float* __restrict__ base_outputs,
    const float* __restrict__ probs,
    const float* __restrict__ residual_groups,
    const order_t* __restrict__ exact_order,
    const int64_t* __restrict__ v_budgets,
    const float* __restrict__ k_mb,
    const float* __restrict__ v_mb,
    int64_t* __restrict__ final_indices,
    float* __restrict__ final_outputs,
    int64_t groups,
    int64_t k_count,
    int64_t v_count,
    int64_t heads,
    int64_t dim,
    int64_t context_len,
    double threshold,
    int policy_id) {
  int64_t gh = blockIdx.x;
  int64_t group = gh / heads;
  int64_t head = gh - group * heads;
  if (group >= groups || head >= heads || k_count <= 0 || v_count <= 0 || dim <= 0) {
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
      float cur_f = joint_grouped_flat_output_value(
          base_outputs,
          probs,
          residual_groups,
          exact_order,
          v_budgets,
          group,
          ki,
          vi,
          head,
          k_count,
          heads,
          context_len,
          v_count,
          dim,
          d);
      double cur = static_cast<double>(cur_f);
      if (k_can) {
        double nxt = static_cast<double>(joint_grouped_flat_output_value(
            base_outputs,
            probs,
            residual_groups,
            exact_order,
            v_budgets,
            group,
            ki + 1,
            vi,
            head,
            k_count,
            heads,
            context_len,
            v_count,
            dim,
            d));
        double delta = cur - nxt;
        k_diff += delta * delta;
        k_ref += nxt * nxt;
      }
      if (v_can) {
        double nxt = static_cast<double>(joint_grouped_flat_output_value(
            base_outputs,
            probs,
            residual_groups,
            exact_order,
            v_budgets,
            group,
            ki,
            vi + 1,
            head,
            k_count,
            heads,
            context_len,
            v_count,
            dim,
            d));
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
        int action = 0;
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
          double extra_k = k_can ? static_cast<double>(k_mb[group * k_count + ki + 1] - k_mb[group * k_count + ki]) : INFINITY;
          double extra_v = v_can ? static_cast<double>(v_mb[group * v_count + vi + 1] - v_mb[group * v_count + vi]) : INFINITY;
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

  int64_t final_ki = ki_shared;
  int64_t final_vi = vi_shared;
  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    final_outputs[(gh * dim) + d] = joint_grouped_flat_output_value(
        base_outputs,
        probs,
        residual_groups,
        exact_order,
        v_budgets,
        group,
        final_ki,
        final_vi,
        head,
        k_count,
        heads,
        context_len,
        v_count,
        dim,
        d);
  }
  if (threadIdx.x == 0) {
    final_indices[gh * 2] = final_ki;
    final_indices[gh * 2 + 1] = final_vi;
  }
}

__global__ void joint_select_policy_grouped_flat_kernel(
    const float* __restrict__ outputs,
    const float* __restrict__ k_mb,
    const float* __restrict__ v_mb,
    int64_t* __restrict__ final_indices,
    float* __restrict__ final_outputs,
    bool* __restrict__ boundary_groups,
    double* __restrict__ accounting_sums,
    const int64_t* __restrict__ selected_counts_by_ki,
    const int64_t* __restrict__ v_budgets,
    int64_t groups,
    int64_t k_count,
    int64_t v_count,
    int64_t heads,
    int64_t dim,
    int64_t context_len,
    int64_t head_dim,
    int64_t key_bytes,
    int64_t value_bytes,
    double selector_mb,
    double v_pq_codebook_mb,
    double metadata_mb,
    int64_t actual_value_subvecs,
    int64_t code_bytes,
    double threshold,
    int policy_id) {
  int64_t gh = blockIdx.x;
  int64_t group = gh / heads;
  int64_t head = gh - group * heads;
  if (group >= groups || head >= heads || k_count <= 0 || v_count <= 0 || dim <= 0) {
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

  const int64_t rows_per_group = k_count * heads;
  const int64_t group_row_base = group * rows_per_group;
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
    int64_t cur_row = group_row_base + ki * heads + head;
    for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
      double cur = static_cast<double>(outputs[(cur_row * v_count + vi) * dim + d]);
      if (k_can) {
        int64_t next_row = group_row_base + (ki + 1) * heads + head;
        double nxt = static_cast<double>(outputs[(next_row * v_count + vi) * dim + d]);
        double delta = cur - nxt;
        k_diff += delta * delta;
        k_ref += nxt * nxt;
      }
      if (v_can) {
        double nxt = static_cast<double>(outputs[(cur_row * v_count + (vi + 1)) * dim + d]);
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
        int action = 0;
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
          double extra_k = k_can ? static_cast<double>(k_mb[group * k_count + ki + 1] - k_mb[group * k_count + ki]) : INFINITY;
          double extra_v = v_can ? static_cast<double>(v_mb[group * v_count + vi + 1] - v_mb[group * v_count + vi]) : INFINITY;
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

  int64_t final_ki = ki_shared;
  int64_t final_vi = vi_shared;
  int64_t final_row = group_row_base + final_ki * heads + head;
  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    final_outputs[(gh * dim) + d] = outputs[(final_row * v_count + final_vi) * dim + d];
  }
  if (threadIdx.x == 0) {
    final_indices[gh * 2] = final_ki;
    final_indices[gh * 2 + 1] = final_vi;
    if (boundary_groups != nullptr && (final_ki >= k_count - 1 || final_vi >= v_count - 1)) {
      boundary_groups[group] = true;
    }
    if (accounting_sums != nullptr && selected_counts_by_ki != nullptr && v_budgets != nullptr) {
      constexpr double mb = 1024.0 * 1024.0;
      int64_t selected_count = selected_counts_by_ki[final_ki] < 0 ? 0 : selected_counts_by_ki[final_ki];
      int64_t exact_v = v_budgets[final_vi] < 0 ? 0 : (v_budgets[final_vi] > context_len ? context_len : v_budgets[final_vi]);
      int64_t tail_count = context_len > exact_v ? context_len - exact_v : 0;
      double exact_key_mb = static_cast<double>(selected_count * head_dim * key_bytes) / mb;
      double exact_v_mb = static_cast<double>(exact_v * head_dim * value_bytes) / mb;
      double tail_mb =
          v_pq_codebook_mb + static_cast<double>(tail_count * actual_value_subvecs * code_bytes) / mb + metadata_mb;
      double dense_key_mb = static_cast<double>(context_len * head_dim * key_bytes) / mb;
      atomicAdd(&accounting_sums[0], static_cast<double>(selected_count));
      atomicAdd(&accounting_sums[1], static_cast<double>(tail_count));
      atomicAdd(&accounting_sums[2], selector_mb);
      atomicAdd(&accounting_sums[3], exact_key_mb + exact_v_mb);
      atomicAdd(&accounting_sums[4], tail_mb);
      atomicAdd(&accounting_sums[6], dense_key_mb + exact_v_mb);
      atomicAdd(&accounting_sums[8], selector_mb > 0.0 ? 1.0 : 0.0);
      atomicAdd(&accounting_sums[9], tail_mb > 0.0 ? 1.0 : 0.0);
    }
  }
}

__global__ void joint_select_policy_grouped_intervals_no_mb_kernel(
    const float* __restrict__ base_outputs,
    const float* __restrict__ interval_sums,
    int64_t* __restrict__ final_indices,
    float* __restrict__ final_outputs,
    int64_t groups,
    int64_t k_count,
    int64_t v_count,
    int64_t heads,
    int64_t dim,
    double threshold,
    int policy_id) {
  int64_t gh = blockIdx.x;
  int64_t group = gh / heads;
  int64_t head = gh - group * heads;
  if (group >= groups || head >= heads || k_count <= 0 || v_count <= 0 || dim <= 0) {
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
      double cur = static_cast<double>(joint_grouped_flat_interval_output_value(
          base_outputs,
          interval_sums,
          group,
          ki,
          vi,
          head,
          k_count,
          heads,
          v_count,
          dim,
          d));
      if (k_can) {
        double nxt = static_cast<double>(joint_grouped_flat_interval_output_value(
            base_outputs,
            interval_sums,
            group,
            ki + 1,
            vi,
            head,
            k_count,
            heads,
            v_count,
            dim,
            d));
        double delta = cur - nxt;
        k_diff += delta * delta;
        k_ref += nxt * nxt;
      }
      if (v_can) {
        double nxt = static_cast<double>(joint_grouped_flat_interval_output_value(
            base_outputs,
            interval_sums,
            group,
            ki,
            vi + 1,
            head,
            k_count,
            heads,
            v_count,
            dim,
            d));
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
        int action = 0;
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
        } else {
          int preferred = (static_cast<int>(step) % 2 == 0) ? 1 : 0;
          if (preferred == 1 && v_bad) {
            action = 1;
          } else if (preferred == 0 && k_bad) {
            action = 0;
          } else {
            action = k_bad ? 0 : 1;
          }
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

  int64_t final_ki = ki_shared;
  int64_t final_vi = vi_shared;
  for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
    final_outputs[(gh * dim) + d] = joint_grouped_flat_interval_output_value(
        base_outputs,
        interval_sums,
        group,
        final_ki,
        final_vi,
        head,
        k_count,
        heads,
        v_count,
        dim,
        d);
  }
  if (threadIdx.x == 0) {
    final_indices[gh * 2] = final_ki;
    final_indices[gh * 2 + 1] = final_vi;
  }
}
