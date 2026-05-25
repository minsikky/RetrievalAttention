#include <torch/extension.h>

#include <vector>

// Keep extension declarations, wrappers, and bindings ordered.
// See paged_pq_ext_parts/README.md before moving fragments.
#include "paged_pq_ext_parts/paged_pq_ext_cuda_decls_core.cpp.inc"
#include "paged_pq_ext_parts/paged_pq_ext_cuda_decls_joint.cpp.inc"
#include "paged_pq_ext_parts/paged_pq_ext_fullscan_wrappers.cpp.inc"
#include "paged_pq_ext_parts/paged_pq_ext_causal_vpq_wrappers.cpp.inc"
#include "paged_pq_ext_parts/paged_pq_ext_decode_tail_wrappers.cpp.inc"
#include "paged_pq_ext_parts/paged_pq_ext_geometric_count_wrappers.cpp.inc"
#include "paged_pq_ext_parts/paged_pq_ext_geometric_output_wrappers.cpp.inc"
#include "paged_pq_ext_parts/paged_pq_ext_joint_vprefix_wrappers.cpp.inc"
#include "paged_pq_ext_parts/paged_pq_ext_joint_softmax_policy_wrappers.cpp.inc"
#include "paged_pq_ext_parts/paged_pq_ext_joint_scoregrid_wrappers.cpp.inc"
#include "paged_pq_ext_parts/paged_pq_ext_causal_geometric_wrappers.cpp.inc"
#include "paged_pq_ext_parts/paged_pq_ext_bindings.cpp.inc"
