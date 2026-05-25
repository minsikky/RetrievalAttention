# Paged-PQ Pybind Fragments

These `.cpp.inc` files are implementation fragments included by
`../paged_pq_ext.cpp`. They are not standalone translation units.

The include order is part of the extension contract:

1. CUDA implementation forward declarations
2. Python-visible wrapper functions with validation and contiguous conversion
3. the `PYBIND11_MODULE` binding table

Keep exported wrapper bodies in the narrowest matching fragment. If a fragment
approaches 1000 lines, split it at top-level wrapper boundaries before adding
more bindings or validation logic.

Validate structural changes with the CUDA unit Slurm wrapper:

```bash
OUTPUT_DIR=cuda_unit_result/<name> CUDA_UNIT_TEST_SET=all sbatch scripts/run_frontier_cuda_unit_tests.sh
```
