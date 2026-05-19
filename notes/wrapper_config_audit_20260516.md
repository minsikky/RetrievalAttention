| wrapper | path | status | important defaults |
| --- | --- | --- | --- |
| frontier_ruler | `scripts/run_frontier_ruler_batched_one.sh` | ok | MODE=pagedpq_batched, SELECTOR_BACKEND=cuda_ext, ONLINE_CONFIDENCE_RULE=pq_ranked_mass_budget, SELECTED_VALUE_MODE=vpq_value, PREFILL_SELECTOR_BACKEND=torch_matmul, INDEX_BUILD_BACKEND=torch_gpu, SBATCH --partition=spgpu, SBATCH --account=zhengya98 |
| frontier_longbench | `scripts/run_frontier_longbench_v2_one.sh` | ok | ATTENTION_MODE=pagedpq, SELECTOR_BACKEND=cuda_ext, ONLINE_CONFIDENCE_RULE=pq_ranked_mass_budget, SELECTED_VALUE_MODE=vpq_value, PREFILL_SELECTOR_BACKEND=torch_matmul, INDEX_BUILD_BACKEND=torch_gpu, SBATCH --partition=spgpu, SBATCH --account=zhengya98 |
| dense_ruler | `scripts/run_dense_ruler_batched_one.sh` | ok | MODE=dense_batched, SBATCH --partition=spgpu, SBATCH --account=zhengya98 |
| dense_longbench | `scripts/run_dense_longbench_v2_one.sh` | ok | ATTENTION_MODE=dense, SBATCH --partition=spgpu, SBATCH --account=zhengya98 |
