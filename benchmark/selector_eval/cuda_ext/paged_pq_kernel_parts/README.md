# Paged-PQ CUDA Fragments

These `.cu.inc` files are implementation fragments for `../paged_pq_kernel.cu`.
They are not standalone translation units and should not be compiled directly.

The include order in `paged_pq_kernel.cu` is part of the contract:

1. shared kernels and helper kernels
2. geometric/output kernels
3. joint kernels
4. public CUDA wrapper functions

Later fragments may depend on kernels or helpers declared by earlier fragments.
When adding code, put it in the narrowest existing fragment and keep each
fragment focused enough to review. If a fragment starts approaching 1000 lines,
split it at kernel or wrapper boundaries before adding more logic.

Validate structural changes with:

```bash
OUTPUT_DIR=cuda_unit_result/<name> CUDA_UNIT_TEST_SET=all sbatch scripts/run_frontier_cuda_unit_tests.sh
```
