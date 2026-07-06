#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

int64_t decode_tail_pages_per_block() {
  const char* value = std::getenv("SELECTOR_PQ_DECODE_TAIL_PAGES_PER_BLOCK");
  if (value == nullptr || value[0] == '\0') {
    return 1;
  }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0) {
    return 1;
  }
  return std::min<int64_t>(64, std::max<int64_t>(1, static_cast<int64_t>(parsed)));
}

int decode_geometric_threads() {
  const char* value = std::getenv("SELECTOR_PQ_GEOMETRIC_THREADS");
  if (value == nullptr || value[0] == '\0') {
    return 256;
  }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value) {
    return 256;
  }
  if (parsed <= 64) {
    return 64;
  }
  if (parsed <= 128) {
    return 128;
  }
  if (parsed <= 256) {
    return 256;
  }
  return 512;
}

void check_cublas(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cuBLAS status ", static_cast<int>(status));
}

bool env_enabled_default(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  if (value[0] == '0' || value[0] == 'f' || value[0] == 'F' || value[0] == 'n' || value[0] == 'N') {
    return false;
  }
  return true;
}

class ScopedCublasMathMode {
 public:
  ScopedCublasMathMode(cublasHandle_t handle, cublasMath_t requested)
      : handle_(handle), active_(false), previous_(CUBLAS_DEFAULT_MATH) {
    check_cublas(cublasGetMathMode(handle_, &previous_), "cublasGetMathMode");
    if (previous_ != requested) {
      check_cublas(cublasSetMathMode(handle_, requested), "cublasSetMathMode");
      active_ = true;
    }
  }

  ~ScopedCublasMathMode() {
    if (active_) {
      // Destructors cannot safely throw through Python extension boundaries.
      (void)cublasSetMathMode(handle_, previous_);
    }
  }

  ScopedCublasMathMode(const ScopedCublasMathMode&) = delete;
  ScopedCublasMathMode& operator=(const ScopedCublasMathMode&) = delete;

 private:
  cublasHandle_t handle_;
  bool active_;
  cublasMath_t previous_;
};

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* __restrict__ ptr, int64_t index) {
  return static_cast<float>(ptr[index]);
}

template <typename scalar_t>
__device__ __forceinline__ float load_strided3_as_float(
    const scalar_t* __restrict__ ptr,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2) {
  return load_as_float(ptr, i0 * stride0 + i1 * stride1 + i2 * stride2);
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float logaddexp_device(float a, float b) {
  if (!isfinite(a)) {
    return b;
  }
  if (!isfinite(b)) {
    return a;
  }
  float m = fmaxf(a, b);
  return m + logf(expf(a - m) + expf(b - m));
}


// Keep this translation unit ordered: later fragments depend on helpers and kernels above.
// See paged_pq_kernel_parts/README.md before adding or moving fragments.
#include "paged_pq_kernel_parts/paged_pq_fullscan_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_selected_attention_basic_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_selected_attention_fused_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_decode_tail_fused_qk_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_decode_tail_fused_score_tail_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_decode_tail_fused_decode_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_decode_tail_logits_base_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_decode_tail_logits_mask_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_decode_tail_logits_reduce_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_decode_tail_logits_codeweight_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_decode_tail_agg_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_geometric_accept_basic_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_geometric_accept_codeweight_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_geometric_accept_exact_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_geometric_output_stepdiff_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_geometric_output_pick_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_geometric_output_final_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_geometric_output_proxy_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_geometric_output_dimscan_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_accept_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_softmax_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_prefix_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_policy_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_affine_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_mixed_kernels.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_fullscan_topk_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_fullscan_selected_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_fullscan_tail_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_fullscan_tail_agg_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_fullscan_geometric_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_exact_logit_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_geometric_count_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_geometric_output_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_causal_geometric_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_softmax_base_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_softmax_score_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_score_direct_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_merge_risk_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_vpq_sidecar_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_mixed_output_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_risk_vprefix_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_risk_policy_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_risk_prefix_wrappers.cu.inc"
#include "paged_pq_kernel_parts/paged_pq_joint_score_grid_wrappers.cu.inc"
