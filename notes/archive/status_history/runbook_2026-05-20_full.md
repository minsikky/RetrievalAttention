# Runbook

## Frontier Benchmark Readiness Presets (2026-05-16)

Use these wrappers for real dense-vs-frontier benchmark validation. They include `spgpu` / `zhengya98` Slurm headers. The HF LongBench-v2 and public long-decode wrappers accept `HF_MODEL_PRESET=qwen3_8b|llama31_8b|qwen3_5_9b`; default is `qwen3_8b`. Older RULER streaming wrappers are still Llama-specific unless explicitly updated.

Frontier RULER, one task:
```bash
TASK_NAME=niah_single_1 CONTEXT_LEN=8192 NUM_SAMPLES=4 \
OUTPUT_ROOT=ruler_eval_result/frontier_batched \
sbatch scripts/run_frontier_ruler_batched_one.sh
```

Dense RULER, matching task:
```bash
TASK_NAME=niah_single_1 CONTEXT_LEN=8192 NUM_SAMPLES=4 \
OUTPUT_ROOT=ruler_eval_result/dense_batched \
sbatch scripts/run_dense_ruler_batched_one.sh
```

Frontier LongBench-v2, one slice:
```bash
MAX_EXAMPLES=64 LENGTH_FILTER=short DIFFICULTY_FILTER=easy MAX_INPUT_TOKENS=8192 \
OUTPUT_DIR=longbench_v2_hf_result/frontier_batched_lbv2 \
sbatch scripts/run_frontier_longbench_v2_one.sh
```

Dense LongBench-v2, matching slice:
```bash
MAX_EXAMPLES=64 LENGTH_FILTER=short DIFFICULTY_FILTER=easy MAX_INPUT_TOKENS=8192 \
OUTPUT_DIR=longbench_v2_hf_result/dense_lbv2 \
sbatch scripts/run_dense_longbench_v2_one.sh
```

Current frontier preset:

- `MODE=pagedpq_batched` for RULER, `ATTENTION_MODE=pagedpq` for LongBench-v2.
- `pq_ranked_mass_budget` confidence with conservative upper-bound selector-cost accounting.
- `cuda_ext` decode selector, `torch_matmul` prefill selector, native selected/tail attention, GPU index build.
- Selected V uses `vpq_value` with `selector_rank` exact top `256`; tail uses V-PQ blend `1.0`.
- Profiling is off by default. Enable `PROFILE_NATIVE_OPS=1` only for diagnostic timing runs.

Current pending validation manifests:

- `notes/benchmark_readiness_checklist.md`
- `notes/wrapper_config_audit_20260516.md`
- `notes/benchmark_runtime_projection_20260516.md`
- `notes/slurm_manifests/frontier_cuda_unit_tests_20260516.tsv`
- `notes/cuda_unit_audit_20260516.md`
- `notes/slurm_manifests/ruler_frontier_wrapper_smoke_20260516.tsv`
- `notes/slurm_manifests/longbench_frontier_wrapper_smoke_20260516.tsv`
- `notes/slurm_manifests/dense_wrapper_smoke_20260516.tsv`
- `notes/slurm_manifests/frontier_benchmark_matrix_afterok_20260516.tsv`

Audit wrapper defaults before submitting benchmark batches:
```bash
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
.venv/bin/python benchmark/audit_benchmark_wrappers.py \
  --output notes/wrapper_config_audit_20260516.md
```

Poll with:
```bash
scripts/poll_frontier_readiness_smokes.sh
```

Generate a compact readiness table from completed artifacts:
```bash
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
.venv/bin/python benchmark/audit_benchmark_readiness.py \
  --manifest notes/slurm_manifests/ruler_ctx8192_batched_success_20260516.tsv \
  --manifest notes/slurm_manifests/longbench_v2_short_easy_n64_batched_20260516.tsv \
  --output notes/readiness_audit_20260516.md
```

After wrapper smokes pass, submit a paired benchmark matrix:
```bash
TAG=ctx8k_short_easy_$(date +%Y%m%d_%H%M%S) \
RULER_TASKS=niah_single_1,niah_multikey_2,vt,fwe \
RULER_CONTEXT_LEN=8192 \
RULER_NUM_SAMPLES=4 \
LONGBENCH_MAX_EXAMPLES=64 \
LONGBENCH_LENGTH_FILTER=short \
LONGBENCH_DIFFICULTY_FILTER=easy \
LONGBENCH_MAX_INPUT_TOKENS=8192 \
scripts/submit_frontier_benchmark_matrix.sh
```

To queue the matrix behind smoke/unit gates, set `SBATCH_DEPENDENCY=afterok:<jobids>`.

Use `DRY_RUN=1` first if changing task lists or output roots.

Generate LongBench relL2/cosine-to-accuracy drift report from paired predictions plus diagnostic rows:
```bash
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
.venv/bin/python benchmark/report_longbench_drift.py \
  --dense longbench_v2_hf_result/dense_lbv2_short_easy_n64_temp0 \
  --frontier longbench_v2_hf_result/frontier_lbv2_short_easy_n64_temp0 \
  --diag-glob 'longbench_v2_hf_result/frontier_readiness_20260516_diag_fulltail64_temp0_*/summary.json' \
  --changed-only \
  --output notes/longbench_drift_report_20260516.md
```

Finalize the queued benchmark matrix after it completes:
```bash
bash scripts/finalize_frontier_benchmark_matrix.sh
```

Default outputs:

- `notes/frontier_benchmark_matrix_afterok_20260516_audit.md`
- `notes/frontier_benchmark_matrix_afterok_20260516_longbench_compare.txt`
- `notes/frontier_benchmark_matrix_afterok_20260516_longbench_drift.md`

If the final drift report has changed rows with missing diagnostics, submit row-level dense-reference diagnostics:
```bash
bash scripts/submit_longbench_changed_row_diagnostics.sh
```

Then rerun `bash scripts/finalize_frontier_benchmark_matrix.sh`.

Run the strict completion gate before claiming benchmark readiness:
```bash
bash scripts/check_frontier_benchmark_readiness.sh
```

This gate reruns wrapper/default audits, CUDA unit-test artifact audit, wrapper-smoke artifact audit, benchmark-matrix artifact audit, and checks that the LongBench drift report is not a placeholder.

## Attention-Efficiency Proxy Sweep (2026-04-29)
- Purpose: compare dense oracle, static/chunk, RetroInfer-style, and RetrievalAttention-style token selection by algorithmic efficiency, not wall-clock latency.
- Main outputs:
  - `attention_efficiency_samples.jsonl`
  - `summary.csv`
  - `summary.json`
  - optional `attention_efficiency_n*.png`
- Local CPU smoke:
```bash
LD_LIBRARY_PATH=/sw/pkgs/arc/python/3.10.4/lib .venv/bin/python benchmark/attention_efficiency_eval.py \
  --output_dir /tmp/attention_eff_smoke \
  --context_lengths 1024 \
  --budgets 32,64 \
  --num_queries 8 \
  --num_heads 4 \
  --num_kv_heads 2 \
  --head_dim 32 \
  --static_prefix 16 \
  --static_suffix 32 \
  --budget_mode dynamic \
  --device cpu \
  --plot
```
- Slurm sweep:
```bash
sbatch benchmark/run_attention_efficiency_eval.sh
```
- Useful overrides:
```bash
sbatch --export=ALL,OUTPUT_DIR=attention_efficiency_result/proxy_v2,CONTEXT_LENGTHS=32768,65536,131072,BUDGETS=64,128,256,512,1024,2048,RA_VISIT_BUDGET=2048 benchmark/run_attention_efficiency_eval.sh
```
- `BUDGET_MODE=dynamic` means the budget is extra retrieved tokens beyond static prefix/suffix; static tokens are still counted in token-read ratio.

## Generic HuggingFace Backend Smoke Commands (2026-04-24)
- Config-only check against cached Llama:
```bash
module load python/3.10.4
.venv/bin/python benchmark/generated_memory_hf_eval.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --config_only \
  --local_files_only \
  --output_dir /tmp/generated_memory_hf_config_smoke_llama31
```
- Slurm inventory check:
```bash
sbatch --parsable --time=60:00 \
  --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct,OUTPUT_DIR=generated_memory_hf_eval_result/inventory_llama31,INVENTORY_ONLY=1,LOCAL_FILES_ONLY=1 \
  benchmark/run_generated_memory_hf.sh
```
- Tiny native HF generated-memory smoke:
```bash
sbatch --parsable --time=90:00 \
  --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct,OUTPUT_DIR=generated_memory_hf_eval_result/smoke_llama31_native,LOCAL_FILES_ONLY=1,NUM_SAMPLES=1,NUM_ENTRIES=6,NUM_QUERIES=2,HF_ATTENTION_MODE=native \
  benchmark/run_generated_memory_hf.sh
```
- Tiny oracle-topk sparse decode smoke:
```bash
sbatch --parsable --time=90:00 \
  --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct,OUTPUT_DIR=generated_memory_hf_eval_result/smoke_llama31_oracle_topk,LOCAL_FILES_ONLY=1,NUM_SAMPLES=1,NUM_ENTRIES=6,NUM_QUERIES=2,HF_ATTENTION_MODE=oracle_topk,HF_SPARSE_TOPK=16,HF_SPARSE_STATIC_PREFIX=16,HF_SPARSE_STATIC_SUFFIX=16 \
  benchmark/run_generated_memory_hf.sh
```
- Tiny RoarGraph-backed HF sparse decode smoke:
```bash
sbatch --parsable --time=60:00 \
  --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct,LOCAL_FILES_ONLY=1,OUTPUT_DIR=generated_memory_hf_eval_result/smoke_llama31_graph_topk_roar,NUM_SAMPLES=1,NUM_ENTRIES=6,NUM_QUERIES=2,HF_ATTENTION_MODE=graph_topk_roar,HF_GRAPH_SEARCH_BACKEND=cuda_group,HF_SPARSE_TOPK=16,HF_SPARSE_STATIC_PREFIX=16,HF_SPARSE_STATIC_SUFFIX=16,HF_GRAPH_DEGREE=8,HF_GRAPH_VISIT_BUDGET=64,HF_GRAPH_SEED_COUNT=16,HF_GRAPH_ONLINE_EDGES=8,HF_GRAPH_CANDIDATE_TARGET=32,HF_GRAPH_EXPAND_WIDTH=16,HF_GRAPH_MIN_VISITS=8,HF_GRAPH_FRONTIER_TOPN=32 \
  benchmark/run_generated_memory_hf.sh
```
- `graph_topk_roar` keeps the generic HF replacement boundary but uses the RoarGraph wrapper path instead of the Python heap traversal:
  - graph build: `build_roar_graph_csr_cpp`
  - search backend: `HF_GRAPH_SEARCH_BACKEND=cpp|cuda_group|cuda_fullgpu`
  - `cuda_group` uses native CUDA scoring with CPU frontier bookkeeping
  - `cuda_fullgpu` calls the RoarGraph full-GPU traversal kernel; this is the closer analogue to the optimized RetrievalAttention decode backend
  - online birth-time edges are merged into CSR before search
- For Qwen3.5 / newer HF models:
  - use the overlay installed by `scripts/setup_hf_pydeps.sh`:
```bash
scripts/setup_hf_pydeps.sh
```
  - then pass `HF_EXTRA_PYTHONPATH=.hf_pydeps`
  - this overlays Transformers main while keeping `.venv` torch active
  - keep `HF_ATTENTION_MODE=native` for first load/inventory
  - only target `full_attention` layers for sparse/RA replacement; leave `linear_attention` layers native
  - use `HF_LANGUAGE_MODEL_ONLY=1` for text-only generated-memory runs
  - do not use `TRUST_REMOTE_CODE=1` for config/tokenizer checks unless explicitly required and approved
- Qwen3.5 config/tokenizer checks:
```bash
module load python/3.10.4
PYTHONPATH=.hf_pydeps .venv/bin/python benchmark/generated_memory_hf_eval.py \
  --model_name Qwen/Qwen3.5-9B \
  --config_only \
  --output_dir /tmp/generated_memory_hf_config_qwen35_9b

PYTHONPATH=.hf_pydeps .venv/bin/python benchmark/generated_memory_hf_eval.py \
  --model_name Qwen/Qwen3.5-9B \
  --tokenizer_only \
  --output_dir /tmp/generated_memory_hf_tokenizer_qwen35_9b
```
- Qwen3.5 expected attention plan:
  - full-attention layers: `3,7,11,15,19,23,27,31`
  - linear-attention layers: all other layers
  - RA/sparse replacement should only target those 8 full-attention layers
- CPU-only sparse patcher unit test:
```bash
module load python/3.10.4
.venv/bin/python scripts/test_hf_sparse_patchers.py
```

## Current runtime contract (2026-02-26)
- RetrievalAttention prefill/index build is fused-only.
- Required components:
  - flash-attn build exporting `flash_attn_with_kvcache_retrieval`,
  - RoarGraph C++ extension (`third_party/RoarGraph/python_ext`).
- Deprecated runtime toggles are no longer active in `test.sh`:
  - `RETRIEVALATTN_GPU_TOPK`,
  - `RETRIEVALATTN_CUSTOM_QK_TOPK*`,
  - `RETRIEVALATTN_GRAPH_BUILDER`,
  - `RETRIEVALATTN_Q_BLOCK`, `RETRIEVALATTN_K_BLOCK`,
  - `RETRIEVALATTN_OVERLAP`, `RETRIEVALATTN_LAYER_GPU_CACHE`.
- `RETRIEVALATTN_FA_SHADOW_COMPARE` is deprecated and ignored in fused-only runtime.
- Holdout recall controls:
  - `RETRIEVALATTN_GRAPH_TRAIN_FRAC` (`0.0~1.0`): fraction of query rows used to build K-graph.
  - `RETRIEVALATTN_GRAPH_SPLIT` (`stratified|random|contiguous`): train/holdout query split policy when `GRAPH_TRAIN_FRAC < 1.0`.
  - `RETRIEVALATTN_GRAPH_SPLIT_SEED` (int): RNG seed used by non-contiguous split modes.
  - `RETRIEVALATTN_PARITY_HOLDOUT_ONLY` (`0|1`): when enabled, parity/recall samples only from holdout rows.
  - `RETRIEVALATTN_TRAVERSAL_EVAL` (`0|1`): run traversal-efficiency eval during parity (forced to `1` in `RECALL_ONLY=1` mode).
  - `RETRIEVALATTN_TRAVERSAL_EVAL_SAMPLE`: max sampled queries for traversal-efficiency eval.

## Branch/runtime baseline commands (2026-03-06)
- Decode traversal backend A/B on controlled ~40k prompt:
- Historical experiment note:
  - these `python_gpu` commands are for the preserved experiment branch only.
  - replay from branch `exp/decode-python-gpu` at commit `efc234f`.
```bash
# Production CPU decode baseline
sbatch --job-name=dec32_cpp \
  --output=slurm-dec32-cpp.out --error=slurm-dec32-cpp.out \
  --export=ALL,DATA_PATH=benchmark/decode_ab_prompt_32k.json,GEN_LEN=32, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_DECODE_BACKEND=roar_cpp,RETRIEVALATTN_DECODE_GPU_KEYS=0, \
RETRIEVALATTN_ROAR_BACKEND=cpp,RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0 \
  test.sh

# Python CPU traversal control
sbatch --job-name=dec32_py \
  --output=slurm-dec32-py.out --error=slurm-dec32-py.out \
  --export=ALL,DATA_PATH=benchmark/decode_ab_prompt_32k.json,GEN_LEN=32, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_DECODE_BACKEND=python,RETRIEVALATTN_DECODE_GPU_KEYS=0, \
RETRIEVALATTN_ROAR_BACKEND=cpp,RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0 \
  test.sh

# Experimental GPU decode path
sbatch --job-name=dec32_gpu \
  --output=slurm-dec32-gpu.out --error=slurm-dec32-gpu.out \
  --export=ALL,DATA_PATH=benchmark/decode_ab_prompt_32k.json,GEN_LEN=32, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_DECODE_BACKEND=python_gpu,RETRIEVALATTN_DECODE_GPU_KEYS=1, \
RETRIEVALATTN_ROAR_BACKEND=cpp,RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0 \
  test.sh
```
- Current conclusion:
  - `roar_cpp` remains the only practical decode traversal backend here.
  - `python_gpu` is slower than both `roar_cpp` and the same Python traversal on CPU.
  - treat `python_gpu` as a failed experiment unless the traversal loop itself is moved out of Python.
- Active native decode backend A/B:
```bash
# Production baseline
sbatch --job-name=dec40_cpp2 \
  --output=slurm-dec40-cpp2.out --error=slurm-dec40-cpp2.out \
  --export=ALL,DATA_PATH=benchmark/decode_ab_prompt_32k.json,GEN_LEN=32, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_DECODE_BACKEND=roar_cpp,RETRIEVALATTN_ROAR_BACKEND=cpp, \
RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0 \
  test.sh

# Python control
sbatch --job-name=dec40_py2 \
  --output=slurm-dec40-py2.out --error=slurm-dec40-py2.out \
  --export=ALL,DATA_PATH=benchmark/decode_ab_prompt_32k.json,GEN_LEN=32, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_DECODE_BACKEND=python,RETRIEVALATTN_ROAR_BACKEND=cpp, \
RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0 \
  test.sh

# Native CUDA-scoring backend
sbatch --job-name=dec40_cuda \
  --output=slurm-dec40-cuda.out --error=slurm-dec40-cuda.out \
  --export=ALL,DATA_PATH=benchmark/decode_ab_prompt_32k.json,GEN_LEN=32, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_DECODE_BACKEND=roar_cuda,RETRIEVALATTN_ROAR_BACKEND=cpp, \
RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0 \
  test.sh
```
- Current `roar_cuda` conclusion:
  - clearly beats `python`
  - does not yet beat `roar_cpp`
  - keep it as experimental native backend, not default
- `roar_cuda_v2` validation commands:
```bash
# 40k prompt, longer decode
sbatch --job-name=dec40_cpp_g100 \
  --output=slurm-dec40-cpp-g100.out --error=slurm-dec40-cpp-g100.out \
  --export=ALL,DATA_PATH=benchmark/decode_ab_prompt_32k.json,GEN_LEN=100, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_DECODE_BACKEND=roar_cpp,RETRIEVALATTN_ROAR_BACKEND=cpp, \
RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0 \
  test.sh

sbatch --job-name=dec40_cuda_v2_g100 \
  --output=slurm-dec40-cuda-v2-g100.out --error=slurm-dec40-cuda-v2-g100.out \
  --export=ALL,DATA_PATH=benchmark/decode_ab_prompt_32k.json,GEN_LEN=100, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_DECODE_BACKEND=roar_cuda_v2,RETRIEVALATTN_ROAR_BACKEND=cpp, \
RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0 \
  test.sh

# ~65k prompt, 32-step decode
sbatch --job-name=dec64_cpp_g32 \
  --output=slurm-dec64-cpp-g32.out --error=slurm-dec64-cpp-g32.out \
  --export=ALL,DATA_PATH=benchmark/decode_ab_prompt_64k.json,GEN_LEN=32, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_DECODE_BACKEND=roar_cpp,RETRIEVALATTN_ROAR_BACKEND=cpp, \
RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0 \
  test.sh

sbatch --job-name=dec64_cuda_v2_g32 \
  --output=slurm-dec64-cuda-v2-g32.out --error=slurm-dec64-cuda-v2-g32.out \
  --export=ALL,DATA_PATH=benchmark/decode_ab_prompt_64k.json,GEN_LEN=32, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_DECODE_BACKEND=roar_cuda_v2,RETRIEVALATTN_ROAR_BACKEND=cpp, \
RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0 \
  test.sh
```
- Current `roar_cuda_v2` conclusion:
  - beats `roar_cpp` on the A40 controlled benchmark family
  - validated on:
    - ~40k prompt, `GEN_LEN=32`
    - ~40k prompt, `GEN_LEN=100`
    - ~65k prompt, `GEN_LEN=32`
  - next step is not “make it faster than `roar_cpp`” anymore; it is “decide whether to promote it and clean up profiling/overhead”
- Current tree, 32k, native fused GPU graph baseline:
```bash
sbatch --job-name=cmp32_native \
  --export=ALL,RECALL_ONLY=1,RECALL_INPUT_TOKENS=32768,GEN_LEN=1, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0, \
RETRIEVALATTN_FA_KERNEL_PROFILE=1,RETRIEVALATTN_FA_GRAPH_PROFILE=1, \
RETRIEVALATTN_FA_KERNEL_MODE=v2_splitk \
  test.sh
```
- Current tree, 32k, native fused top-k + CPU graph:
```bash
sbatch --job-name=cmp32_cpugpu \
  --export=ALL,RECALL_ONLY=1,RECALL_INPUT_TOKENS=32768,GEN_LEN=1, \
RETRIEVALATTN_FA_GRAPH_FUSED=0,RETRIEVALATTN_ROAR_BACKEND=cpp, \
RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0, \
RETRIEVALATTN_FA_KERNEL_PROFILE=1 \
  test.sh
```
- Current tree, 32k, forced Torch/Python GPU top-k + GPU graph:
```bash
sbatch --job-name=cmp32_torch \
  --export=ALL,RECALL_ONLY=1,RECALL_INPUT_TOKENS=32768,GEN_LEN=1, \
RETRIEVALATTN_FA_GRAPH_FUSED=1,RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1, \
RETRIEVALATTN_FA_FORCE_PYTHON_TOPK=1, \
RETRIEVALATTN_VALIDATE_PARITY=0,RETRIEVALATTN_TRAVERSAL_EVAL=0 \
  test.sh
```
- Old GPU-topk + CPU-graph path from `c90fa94`:
  - use an exported tree on GPFS, not `/tmp`, because compute nodes cannot see login-node `/tmp`.
```bash
mkdir -p /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention_c90fa94_tree
git archive c90fa94 | tar -x -C /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention_c90fa94_tree
ln -sf /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/.venv \
  /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention_c90fa94_tree/.venv

sbatch -D /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention_c90fa94_tree \
  --job-name=c90_32_cpugpu \
  --partition=spgpu --account=zhengya98 \
  --output=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/slurm-c90-32k.out \
  --error=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/slurm-c90-32k.out \
  --export=ALL,RECALL_ONLY=1,RECALL_INPUT_TOKENS=32768,GEN_LEN=1, \
RETRIEVALATTN_FA_FUSED_PREFILL=0,RETRIEVALATTN_GPU_TOPK=1, \
RETRIEVALATTN_ROAR_BACKEND=cpp, \
RETRIEVALATTN_ROAR_CPP_PATH=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/third_party/RoarGraph/python_ext, \
RETRIEVALATTN_PARITY_SAMPLE=64 \
  /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention_c90fa94_tree/test.sh
```
- Baseline results already observed:
  - `cmp32_native` (`44431974`): `97.0912 s`
  - `cmp32_cpugpu` (`44431973`): `143.6257 s`
  - `cmp32_torch` (`44431975`): `115.461 s`
  - `c90_32_cpugpu` (`44432065`): `100.0403 s`
- Caveat:
  - `c90fa94` old path uses `retrieval_head_mode=kv_head`; it is a lower-bound speed reference, not a fair q-head baseline.

## Next-session fused-kernel v2 checklist (2026-03-03)
1. Read pending job outputs first:
```bash
grep -E "native_core_sec|native_graph_sec|native_total_sec|ERROR|Traceback" slurm-44245118.out slurm-44245119.out slurm-44245120.out
```
2. Use fixed harness config for all A/B (do not change prompt/model/hardware between runs):
```bash
GEN_LEN=1 \
RETRIEVALATTN_FA_GRAPH_FUSED=1 \
RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=1 \
RETRIEVALATTN_FA_GRAPH_PROFILE=1 \
RETRIEVALATTN_FA_KERNEL_PROFILE=1 \
sbatch test.sh
```
Kernel-instrumentation knobs (default `0`):
- `RETRIEVALATTN_FA_KERNEL_PROFILE=1`: logs `native_retrieval_profile` and exposes native retrieval phase timings in profile payload.
- `RETRIEVALATTN_FA_KERNEL_DEBUG=1`: logs `native_retrieval_debug` counter summary (higher overhead; use only for debug).
3. A/B top-k merge mode only (everything else fixed):
```bash
RETRIEVALATTN_FA_TOPK_BATCHED=1 sbatch test.sh
RETRIEVALATTN_FA_TOPK_BATCHED=0 sbatch test.sh
```
4. Kernel-mode A/B helper (submit once, three runs):
```bash
# After build job succeeds (example: 44298749)
./benchmark/submit_kernel_mode_ab.sh 44298749
```
5. Fast profile extraction after runs:
```bash
./benchmark/extract_kernel_profiles.sh slurm-ra-kab-*.out
grep -nE "native_retrieval_profile|native_retrieval_debug|native_graph_profile|fused_overlap profile|index built layer" slurm-ra-kab-*.out
```
6. v2 implementation order:
   - lock-free online top-k inside attention CTA,
   - deterministic split-K 2-pass partial-topk reduction,
   - chunked overlap (attention/top-k stream + graph stream) with double buffers.
7. Required correctness gates for each performance claim:
   - top-k parity (causal reference),
   - no unexpected `edges=0`,
   - traversal recall non-regression,
   - deterministic output on fixed seed/hardware.
8. Milestones:
   - M-A: non-split lock-free core >2x faster,
   - M-B: split-K adds >1.5x over non-split,
   - M-C: fused v2 prefill within ~1.3-1.6x of Full FlashAttention.

## Standard simple run
```bash
sbatch test.sh
```

## Tiny recall-only run (prefill/index only)
Use this to iterate on graph/index algorithms without full decode latency.
```bash
RECALL_ONLY=1 \
RECALL_INPUT_TOKENS=8192 \
RETRIEVALATTN_VALIDATE_PARITY=1 \
RETRIEVALATTN_PARITY_LAYERS=1 \
RETRIEVALATTN_PARITY_HEADS=1 \
RETRIEVALATTN_PARITY_SAMPLE=256 \
sbatch test.sh
```
Holdout variant (recommended for graph quality):
```bash
RECALL_ONLY=1 \
RECALL_INPUT_TOKENS=8192 \
RETRIEVALATTN_VALIDATE_PARITY=1 \
RETRIEVALATTN_GRAPH_TRAIN_FRAC=0.9 \
RETRIEVALATTN_GRAPH_SPLIT=stratified \
RETRIEVALATTN_GRAPH_SPLIT_SEED=1234 \
RETRIEVALATTN_PARITY_HOLDOUT_ONLY=1 \
RETRIEVALATTN_PARITY_LAYERS=1 \
RETRIEVALATTN_PARITY_HEADS=1 \
RETRIEVALATTN_PARITY_SAMPLE=256 \
sbatch test.sh
```
Strict-metric target config (achieved >=0.95 near 3% traversal):
```bash
RECALL_ONLY=1 \
RECALL_INPUT_TOKENS=8192 \
RETRIEVALATTN_VALIDATE_PARITY=1 \
RETRIEVALATTN_GRAPH_TRAIN_FRAC=0.9 \
RETRIEVALATTN_GRAPH_SPLIT=stratified \
RETRIEVALATTN_GRAPH_SPLIT_SEED=1234 \
RETRIEVALATTN_PARITY_HOLDOUT_ONLY=1 \
RETRIEVALATTN_TRAVERSAL_EVAL=1 \
RETRIEVALATTN_TRAVERSAL_EVAL_SAMPLE=128 \
RETRIEVALATTN_EXPAND_WIDTH=24 \
RETRIEVALATTN_MIN_VISITS=32 \
RETRIEVALATTN_MAX_VISITS=256 \
RETRIEVALATTN_CAND_MULT=2 \
RETRIEVALATTN_ROAR_M=32 \
RETRIEVALATTN_ROAR_L=20 \
RETRIEVALATTN_ROAR_ENHANCE_L=20 \
RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT=0 \
RETRIEVALATTN_SEED_HUB_K=256 \
RETRIEVALATTN_SEED_TAIL_K=128 \
sbatch test.sh
```
Traversal saturation diagnostic (same holdout split):
```bash
# T1: baseline traversal
RECALL_ONLY=1 \
RECALL_INPUT_TOKENS=8192 \
RETRIEVALATTN_GRAPH_TRAIN_FRAC=0.9 \
RETRIEVALATTN_PARITY_HOLDOUT_ONLY=1 \
RETRIEVALATTN_PARITY_LAYERS=1 \
RETRIEVALATTN_PARITY_HEADS=1 \
RETRIEVALATTN_PARITY_SAMPLE=256 \
RETRIEVALATTN_TRAVERSAL_EVAL=1 \
RETRIEVALATTN_TRAVERSAL_EVAL_SAMPLE=128 \
RETRIEVALATTN_EXPAND_WIDTH=48 \
RETRIEVALATTN_MIN_VISITS=96 \
RETRIEVALATTN_MAX_VISITS=2048 \
sbatch test.sh

# T2: larger traversal budget
RECALL_ONLY=1 \
RECALL_INPUT_TOKENS=8192 \
RETRIEVALATTN_GRAPH_TRAIN_FRAC=0.9 \
RETRIEVALATTN_PARITY_HOLDOUT_ONLY=1 \
RETRIEVALATTN_PARITY_LAYERS=1 \
RETRIEVALATTN_PARITY_HEADS=1 \
RETRIEVALATTN_PARITY_SAMPLE=256 \
RETRIEVALATTN_TRAVERSAL_EVAL=1 \
RETRIEVALATTN_TRAVERSAL_EVAL_SAMPLE=128 \
RETRIEVALATTN_EXPAND_WIDTH=96 \
RETRIEVALATTN_MIN_VISITS=1024 \
RETRIEVALATTN_MAX_VISITS=8192 \
RETRIEVALATTN_CAND_MULT=8 \
sbatch test.sh

# T3: near-exhaustive traversal stress
RECALL_ONLY=1 \
RECALL_INPUT_TOKENS=8192 \
RETRIEVALATTN_GRAPH_TRAIN_FRAC=0.9 \
RETRIEVALATTN_PARITY_HOLDOUT_ONLY=1 \
RETRIEVALATTN_PARITY_LAYERS=1 \
RETRIEVALATTN_PARITY_HEADS=1 \
RETRIEVALATTN_PARITY_SAMPLE=256 \
RETRIEVALATTN_TRAVERSAL_EVAL=1 \
RETRIEVALATTN_TRAVERSAL_EVAL_SAMPLE=128 \
RETRIEVALATTN_EXPAND_WIDTH=128 \
RETRIEVALATTN_MIN_VISITS=4096 \
RETRIEVALATTN_MAX_VISITS=32768 \
RETRIEVALATTN_CAND_MULT=16 \
sbatch test.sh
```
Interpretation rule:
1. If `trav_recall` reaches >0.9 with modest `trav_visit_rate`, graph is usable and decode policy/seeding is the main bottleneck.
2. If `trav_recall` stays low even when `trav_visit_rate` is very high (approaching full search), graph construction is the primary issue.

Optional gate:
```bash
RECALL_ONLY=1 \
RECALL_MIN_RECALL=0.95 \
sbatch test.sh
```

## Decode complexity-regime sweep (N up to 64k)
Purpose:
- compare `O(N)`, `O(sqrt(N))`, and `O(log N)` traversal-budget families,
- primary metric: minimum `trav_visit_rate` that reaches strict `trav_recall >= 0.95`,
- first-pass prefill scaling: scale `ROAR_M` with `N` only (`L/E` fixed).

Stage 1 (coarse sweep, single split seed):
```bash
python benchmark/submit_decode_complexity_sweep.py \
  --prefix dcs_stage1 \
  --out_tsv notes/decode_complexity_stage1_$(date +%F).tsv \
  --sizes 8192,16384,32768,65536 \
  --families linear,sqrt,log \
  --linear_rates 0.01,0.02,0.03,0.05 \
  --sqrt_coeffs 1.0,2.0,3.0,4.0 \
  --log_coeffs 8,12,16,24 \
  --base_roar_m 32 \
  --base_roar_l 20 \
  --base_roar_enhance_l 16 \
  --m_scale_ref_tokens 8192 \
  --m_scale_exponent 0.5 \
  --split_seeds 1234 \
  --partition spgpu
```

After runs finish:
```bash
python benchmark/collect_recall_sweep.py \
  --jobs_tsv notes/decode_complexity_stage1_$(date +%F).tsv \
  --out_csv notes/decode_complexity_stage1_$(date +%F).csv \
  --log_pattern "slurm-{name}-{job_id}.out"
```

```bash
python benchmark/summarize_decode_complexity.py \
  --in_csv notes/decode_complexity_stage1_$(date +%F).csv \
  --target_recall 0.95 \
  --out_frontier_csv notes/decode_complexity_stage1_frontier_$(date +%F).csv \
  --out_regime_csv notes/decode_complexity_stage1_regime_$(date +%F).csv \
  --out_json notes/decode_complexity_stage1_report_$(date +%F).json
```

Stage 2 (confirm finalists with multiple split seeds):
```bash
python benchmark/submit_decode_complexity_sweep.py \
  --prefix dcs_stage2 \
  --out_tsv notes/decode_complexity_stage2_$(date +%F).tsv \
  --sizes 8192,16384,32768,65536 \
  --families linear,sqrt,log \
  --linear_rates 0.02,0.03 \
  --sqrt_coeffs 2.0,3.0 \
  --log_coeffs 12,16 \
  --base_roar_m 32 \
  --base_roar_l 20 \
  --base_roar_enhance_l 16 \
  --m_scale_ref_tokens 8192 \
  --m_scale_exponent 0.5 \
  --split_seeds 1234,4321,9999 \
  --partition spgpu
```

Interpretation:
1. `mean_visit_rate_at_target` is the main comparison key across regimes.
2. `obs_alpha` from `summarize_decode_complexity.py` estimates observed decode scaling using `trav_visited_mean`.
3. If `obs_alpha` is low but hit-rate is poor, the regime is too aggressive and fails quality targets at larger `N`.

## FlashAttention fork build (compute node only)
- Do not run flash-attn build from login node; login node has no CUDA toolkit.
- Submit the build as a Slurm job:
```bash
sbatch install_2.sh
```
- `install_2.sh` now:
  - loads `python/3.10.4` and CUDA module (`cuda/12.8.1` by default),
  - activates `.venv`,
  - resolves `CUDA_HOME` from `which nvcc`,
  - runs `pip install --no-build-isolation -v -e third_party/flash-attn-ra`.
- After changing retrieval kernel C++/CUDA sources, rebuild is required before runs.

## Interactive Python setup (shell)
```bash
module load python/3.10.4
source .venv/bin/activate
```

## RoarGraph C++ graph-builder extension
- Source path: `third_party/RoarGraph/python_ext/roargraph_builder.cpp`
- Build once per environment:
```bash
module load python/3.10.4
source .venv/bin/activate
python third_party/RoarGraph/python_ext/setup.py build_ext --inplace
```
- Default run scripts (`test.sh`, `benchmark/ruler/ruler_run_wrapper.sh`) now use:
  - `RETRIEVALATTN_ROAR_BACKEND=cpp`
  - and fail early if the extension import is missing.
- Force Python fallback for debug:
```bash
RETRIEVALATTN_ROAR_BACKEND=python sbatch test.sh
```
- Experimental GPU-assisted Python graph build path:
```bash
RETRIEVALATTN_ROAR_BACKEND=python_gpu \
RETRIEVALATTN_ROAR_PY_GPU_DEVICE=cuda \
RETRIEVALATTN_ROAR_PY_GPU_BATCH=256 \
sbatch test.sh
```

## spgpu A/B script (cpp vs python_gpu)
- Script: `benchmark/run_roar_backend_ab_spgpu.sh`
- Runs two back-to-back recall-only jobs in one Slurm allocation with identical settings:
  - `RETRIEVALATTN_ROAR_BACKEND=cpp`
  - `RETRIEVALATTN_ROAR_BACKEND=python_gpu`
- Submit:
```bash
sbatch benchmark/run_roar_backend_ab_spgpu.sh
```

## Recommended RetrievalAttention debug run
```bash
RETRIEVALATTN_GPU_TOPK=1 \
RETRIEVALATTN_LAYER_GPU_CACHE=1 \
RETRIEVALATTN_DECODE_INDEX=faiss \
RETRIEVALATTN_SCORE_MODE=ip \
RETRIEVALATTN_DEBUG=1 \
RETRIEVALATTN_ASSERT_NONEMPTY=1 \
RETRIEVALATTN_VALIDATE_PARITY=1 \
sbatch test.sh
```

## Current recommendation on custom fused kernel path
- `RETRIEVALATTN_CUSTOM_QK_TOPK=1` path is currently **frozen** for active iteration.
- Reason:
  - repeated long-context stalls (`chunk 1/...`) and Triton compatibility issues in this cycle.
- Practical guidance:
  - keep custom path disabled for productive runs (`RETRIEVALATTN_CUSTOM_QK_TOPK=0`).
  - use baseline GPU-topk path while moving to FlashAttention-prefill fusion design.

## Recommended latency-safe adaptive run
```bash
RETRIEVALATTN_GPU_TOPK=1 \
RETRIEVALATTN_LAYER_GPU_CACHE=1 \
RETRIEVALATTN_DECODE_INDEX=faiss \
RETRIEVALATTN_DECODE_BACKEND=auto \
RETRIEVALATTN_SEED_MODE=graph_only \
RETRIEVALATTN_SCORE_MODE=ip \
RETRIEVALATTN_GRAPH_BUILDER=roar \
TOKEN_BUDGET_OVERRIDE=100 \
RETRIEVALATTN_GRAPH_EXPAND=1 \
RETRIEVALATTN_EXPAND_WIDTH=48 \
RETRIEVALATTN_MIN_VISITS=96 \
RETRIEVALATTN_MAX_VISITS=2048 \
RETRIEVALATTN_STOP_PATIENCE=1 \
RETRIEVALATTN_STOP_MARGIN=0.001 \
sbatch test.sh
```

## Graph-builder A/B (legacy vs roar)
```bash
# A: Roar builder (current default in test.sh)
ATTN_TYPE=RetrievalAttention \
RETRIEVALATTN_GRAPH_BUILDER=roar \
sbatch test.sh

# B: Legacy projected graph builder
ATTN_TYPE=RetrievalAttention \
RETRIEVALATTN_GRAPH_BUILDER=legacy \
sbatch test.sh
```

Compare:
```bash
grep -nE "index built layer|builder=|decode_profile|seed=|graph=|Answer:" slurm-*.out
```

## Required A/B after graph-only seed change
```bash
# A: graph-only seed (default)
ATTN_TYPE=RetrievalAttention \
RETRIEVALATTN_SEED_MODE=graph_only \
sbatch test.sh

# B: legacy full-scan seed
ATTN_TYPE=RetrievalAttention \
RETRIEVALATTN_SEED_MODE=faiss \
sbatch test.sh
```

Compare in slurm logs:
```bash
grep -nE "decode_profile|retrieve=|seed=|graph=" slurm-*.out
```
Target:
1. `seed` decreases in A vs B.
2. If `graph` still dominates, traversal becomes the next optimization target.

## Traversal-focused quick sweep (after seed fix)
Use these as controlled sweeps to reduce graph-time overhead while tracking quality:

```bash
# S1: narrower expansion (lower fanout pressure)
RETRIEVALATTN_SEED_MODE=graph_only \
RETRIEVALATTN_EXPAND_WIDTH=24 \
RETRIEVALATTN_MIN_VISITS=192 \
RETRIEVALATTN_MAX_VISITS=1024 \
RETRIEVALATTN_CAND_MULT=3 \
sbatch test.sh

# S2: stronger candidate pruning
RETRIEVALATTN_SEED_MODE=graph_only \
RETRIEVALATTN_FRONTIER_TOPN=256 \
RETRIEVALATTN_CAND_MULT=2 \
RETRIEVALATTN_MIN_VISITS=192 \
RETRIEVALATTN_MAX_VISITS=1024 \
sbatch test.sh

# S3: latency floor reference (aggressive)
RETRIEVALATTN_SEED_MODE=graph_only \
RETRIEVALATTN_EXPAND_WIDTH=16 \
RETRIEVALATTN_MIN_VISITS=128 \
RETRIEVALATTN_MAX_VISITS=768 \
RETRIEVALATTN_CAND_MULT=2 \
sbatch test.sh
```

For each run, compare:
```bash
grep -nE "decode_profile|retrieve=|seed=|graph=|visited_total|candidates_total|Answer:" slurm-*.out
```

Decision rule:
1. Keep settings only if `graph` time drops materially and top-3 quality is not worse than baseline.
2. If quality drops sharply, revert one notch (raise `expand_width` or `max_visits` first).

## Key env flags
- `RETRIEVALATTN_GPU_TOPK`: enable GPU topk build.
- `RETRIEVALATTN_CUSTOM_QK_TOPK`: opt-in Triton custom fused qk+topk kernel path (currently frozen/experimental; not recommended for active runs).
- `RETRIEVALATTN_FA_FUSED_PREFILL`: use FlashAttention fused-prefill retrieval path (expects flash-attn API `flash_attn_with_kvcache_retrieval` from a custom build/fork).
- `RETRIEVALATTN_FA_GRAPH_FUSED`: enable graph-fused prefill path (prototype).
- `RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE`: fail-fast if graph-fused call fails (no fallback).
- `RETRIEVALATTN_FA_GRAPH_FUSED_CHECK`: run quality-floor check and fallback per-head when below floor.
- `RETRIEVALATTN_FA_GRAPH_FUSED_QUALITY_FLOOR`: strict traversal recall floor used by graph-fused fallback gate (default `0.90`).
- `RETRIEVALATTN_RETRIEVAL_HEAD_MODE`: retrieval indexing mode (`q_head` or `kv_head`; fused path default is `q_head`).
  - `q_head`: fused top-k is expected as `[seq, num_heads, q_knn]`.
  - `kv_head`: legacy behavior keyed by KV heads.
  - current graph design is shared per KV head in both modes; `q_head` mainly controls per-head retrieval/seed/rerank behavior.
- `RETRIEVALATTN_KV_GRAPH_AB`: offline recall-harness A/B for true kv-head graph quality on the current tree.
  - `0` default.
  - `1`: build an alternate kv-head graph from exact grouped queries using the same builder, and report:
    - `kv_proxy`: grouped-query exact top-k overlap vs q-head target
    - `kv_proxy_traversal`: grouped-query traversal on the current q-head graph
    - `kv_graph_traversal`: true traversal recall on the alternate kv-head graph
- `RETRIEVALATTN_KV_GRAPH_AB_Q_BLOCK`: GPU q-block size for the offline exact grouped-query top-k helper (`512` default).
- `RETRIEVALATTN_KV_GRAPH_AB_K_BLOCK`: GPU k-block size for the offline exact grouped-query top-k helper (`4096` default).
- `RETRIEVALATTN_FA_SHADOW_COMPARE`: in fused-prefill mode, run sampled parity check vs baseline GPU-topk (layer0/head0).
- `RETRIEVALATTN_FA_SHADOW_SAMPLE`: number of sampled queries used in fused shadow compare.
- Note: native graph-fused symbol (`fwd_kvcache_retrieval_graph`) is now patched in flash-attn source; until rebuilt on compute, runtime may still use Python graph-fused fallback.
- `RETRIEVALATTN_FUSED_PREFILL_OVERLAP`: overlap CPU finalize (index/graph build) with ongoing fused prefill (`1` default).
- `RETRIEVALATTN_FUSED_PREFILL_OVERLAP_WORKERS`: overlap worker count (`1` recommended to avoid faiss/OpenMP oversubscription).
- `RETRIEVALATTN_CUSTOM_QK_TOPK_BLOCK_Q`: custom kernel Q tile size (default `64`; auto-capped to `<=64` to avoid pathological kernels).
- `RETRIEVALATTN_CUSTOM_QK_TOPK_LAUNCH_Q_CHUNK`: query chunk size per fused kernel launch (default `1024`; auto-capped to `<=1024`; `<=0` runs one monolithic launch).
- `RETRIEVALATTN_CUSTOM_QK_TOPK_BLOCK_D`: custom kernel D tile size (default `32`).
- `RETRIEVALATTN_CUSTOM_QK_TOPK_BLOCK_K`: custom kernel K tile size (default `256`; internally rounded to power-of-two and clamped to `<=256`).
- `RETRIEVALATTN_LAYER_GPU_CACHE`: cache per-layer K/Q on GPU during build.
- `RETRIEVALATTN_OVERLAP`: overlap transfer and compute for block streaming path.
- `RETRIEVALATTN_DECODE_INDEX`: decode seed source (`faiss` recommended).
- `RETRIEVALATTN_SEED_MODE`: decode seed strategy (`graph_only` default; `faiss` for full-scan reference/debug).
- `RETRIEVALATTN_QUERY_MODE`: `per_head` (default) or `group_avg`.
- `RETRIEVALATTN_SCORE_MODE`: retrieval similarity objective (`ip` default, or `cosine` for legacy normalized scoring).
- Caveat: `RETRIEVALATTN_SCORE_MODE=cosine` on native fused prefill requires a rebuilt flash-attn fork that exposes `fwd_kvcache_retrieval(..., retrieval_normalize)`. If the extension is old, runtime raises an explicit rebuild error.
- `RETRIEVALATTN_GRAPH_BUILDER`: graph construction backend (`roar` or `legacy`).
- `RETRIEVALATTN_ROAR_NQ`: Roar query->base neighbor width (uses KNN row prefix).
- `RETRIEVALATTN_ROAR_L`: projection candidate budget per pivot.
- `RETRIEVALATTN_ROAR_M`: degree cap for Roar projection/enhancement.
- `RETRIEVALATTN_ROAR_ENABLE_ENHANCE`: enable connectivity enhancement stage.
- `RETRIEVALATTN_ROAR_ENHANCE_L`: candidate budget used in enhancement beam collection.
- `RETRIEVALATTN_ROAR_ENTRY`: enhancement entry policy (`hub|max_degree|self`).
- `RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT`: optional cap of bridge queries per pivot (`0` disables cap).
- `RETRIEVALATTN_ROAR_LOG`: emit per-stage Roar build metrics in head logs.
- `RETRIEVALATTN_ROAR_BACKEND`: Roar builder backend selector (`cpp` recommended, `python`, `python_gpu` experimental).
- `RETRIEVALATTN_ROAR_PY_GPU_DEVICE`: CUDA device string for `python_gpu` backend (`cuda` default).
- `RETRIEVALATTN_ROAR_PY_GPU_BATCH`: batch size for projection-stage GPU scoring in `python_gpu`.
- `RETRIEVALATTN_ROAR_CPP_THREADS`: thread override for C++ Roar builder (`0` lets OpenMP decide).
- `RETRIEVALATTN_DECODE_BACKEND`: decode traversal backend (`auto` default, `python`, `roar_cpp`).
- `RETRIEVALATTN_ROAR_DECODE_INIT`: number of seed tokens sent to C++ decode queue.
- `RETRIEVALATTN_ROAR_DECODE_LPQ`: C++ decode queue capacity (`0` => candidate target).
- `RETRIEVALATTN_ROAR_DECODE_MAX_CMPS`: max neighbor-score evaluations in C++ decode (`0` => uncapped).
- `RETRIEVALATTN_ROAR_DECODE_MAX_HOPS`: max expanded nodes in C++ decode (`0` => follow `RETRIEVALATTN_MAX_VISITS`).
- `RETRIEVALATTN_ROAR_DECODE_THREADS`: OpenMP thread override for C++ decode call.
- `RETRIEVALATTN_GRAPH_WEIGHTED`: weighted graph projection from prefill top-k (`1` default).
- `RETRIEVALATTN_GRAPH_CLIQUE_M`: clique-lite projection width among top candidates per query row (default `6`).
- `RETRIEVALATTN_GRAPH_RETURN_WEIGHTS`: store CSR edge weights as a third graph tensor (`0` default; decode traversal currently ignores weights).
- `RETRIEVALATTN_GRAPH_WEIGHT_DTYPE`: graph edge weight dtype (`uint16` default, `uint32` optional).
- `RETRIEVALATTN_GRAPH_EXPAND`: enable/disable graph neighbor expansion at decode.
- `RETRIEVALATTN_EXPAND_WIDTH`: number of frontier nodes expanded per adaptive step.
- `RETRIEVALATTN_MIN_VISITS`: minimum graph nodes to expand before adaptive early-stop checks (`<=0` => auto).
- `RETRIEVALATTN_MAX_VISITS`: hard cap on graph nodes expanded per decode/head (`<=0` => auto).
- `RETRIEVALATTN_STOP_PATIENCE`: required number of stable top-k checks before early stop.
- `RETRIEVALATTN_STOP_MARGIN`: stop only when frontier best score is below current kth score by this margin.
- `RETRIEVALATTN_FRONTIER_TOPN`: optional frontier pruning cap (`0` disables pruning).
- `RETRIEVALATTN_RERANK`: rerank candidate tokens using exact query-key dot scores.
- `RETRIEVALATTN_RERANK_AGG`: score aggregation across query heads (`max` default, or `mean`).
- `RETRIEVALATTN_SEED_RATIO`: minimum fraction of final dynamic budget reserved for seed tokens.
- `RETRIEVALATTN_CAND_MULT`: candidate pool size multiplier before final rerank.
- `RETRIEVALATTN_SEED_K_MULT`: multiplier for decode seed search width.
- `RETRIEVALATTN_SEED_PREV_K`: max previous-step retrieved tokens reused as warm-start seeds (graph-only mode).
- `RETRIEVALATTN_SEED_HUB_K`: number of per-head graph hub seeds added at decode (graph-only mode).
- `RETRIEVALATTN_SEED_TAIL_K`: number of dynamic-tail anchor seeds for cold start (graph-only mode).
- `RETRIEVALATTN_DEBUG`: print decode seed/dynamic retrieval diagnostics.
- `RETRIEVALATTN_ASSERT_NONEMPTY`: fail if all heads have empty dynamic retrieval.
- `RETRIEVALATTN_VALIDATE_PARITY`: run sampled parity check vs faiss.
- `RETRIEVALATTN_PARITY_LAYERS`: number of starting layers included in parity sampling.
- `RETRIEVALATTN_PARITY_HEADS`: number of starting KV-heads per layer included in parity sampling.
- `RETRIEVALATTN_PARITY_SAMPLE`: per-head sampled query count for parity.
- `RETRIEVALATTN_DECODE_PROFILE`: print end-of-decode critical-path breakdown (`retrieve`, `gather`, `attn`, `other`).
  - includes traversal efficiency fields: `space/head`, `visited/head`, `visit_rate`, `prune_rate`, `cand/visit`.
- `RECALL_ONLY`: run `simple_test.py` in prefill/index-only recall mode (`gen_len` forced to 1; no decode loop).
- `RECALL_INPUT_TOKENS`: synthetic input length used when `RECALL_ONLY=1`.
- `RECALL_MIN_RECALL`: optional weighted recall threshold; run fails if below target.
- `RETRIEVALATTN_Q_BLOCK`, `RETRIEVALATTN_K_BLOCK`: block sizes for GPU topk.
- `RETRIEVALATTN_HEAD_PIPELINE`: enable per-head GPU/CPU pipelining for index build.
- `RETRIEVALATTN_HEAD_PIPELINE_DEPTH`: max in-flight heads for pipelined finalize.
- `RETRIEVALATTN_HEAD_PIPELINE_MIN_CPUS`: auto-disable head pipeline if allocated CPUs are below this threshold.
- `FAISS_NUM_THREADS`: optional faiss thread cap (runtime also enforces scheduler-safe CPU budget).

## Metric interpretation
1. `Answer: [...]` in `simple_test.py` output is ground truth.
2. The next printed text line is model-generated output.
3. `[/INST]` and long `...` are generated continuation artifacts and should be treated as a secondary qualitative signal.
4. Primary simple-test metric remains top-3 coded-word correctness.

## Quality sanity checklist
1. Output is not dominated by repeated control tokens.
2. Decode debug logs show non-zero dynamic retrieval for most heads.
3. Parity recall is reasonable for sampled queries.

## If run fails
1. Read slurm log.
2. Verify build banner includes intended env values.
3. Check for empty retrieval assertion.
4. If quality issue persists, follow `notes/debug_playbook.md`.

## Native graph-fused smoke
- Build flash-attn fork (compute node): `sbatch install_2.sh`
- Run native graph-fused smoke after build:
  - `sbatch --dependency=afterok:<build_jobid> smoke_flashattn_fused_graph.sh`
- Expected smoke signals:
  - extension has symbol `fwd_kvcache_retrieval_graph`,
  - profile path is `native_kernel_fused_graph`,
  - output includes `[OK] flash_attn_with_kvcache_retrieval_graph smoke test passed.`

## True kv-head graph A/B
- Purpose:
  - compare q-head graph traversal vs a true grouped-query-built kv-head graph on the same current-tree harness.
- Recommended small decisive run:
```bash
RECALL_ONLY=1 \
RECALL_INPUT_TOKENS=8192 \
RETRIEVALATTN_FA_GRAPH_FUSED=0 \
RETRIEVALATTN_ROAR_BACKEND=cpp \
RETRIEVALATTN_VALIDATE_PARITY=1 \
RETRIEVALATTN_KV_GRAPH_AB=1 \
RETRIEVALATTN_PARITY_LAYERS=2 \
RETRIEVALATTN_PARITY_HEADS=32 \
RETRIEVALATTN_PARITY_SAMPLE=256 \
RETRIEVALATTN_GRAPH_TRAIN_FRAC=0.9 \
RETRIEVALATTN_GRAPH_SPLIT=stratified \
RETRIEVALATTN_GRAPH_SPLIT_SEED=1234 \
RETRIEVALATTN_PARITY_HOLDOUT_ONLY=1 \
RETRIEVALATTN_TRAVERSAL_EVAL=1 \
RETRIEVALATTN_TRAVERSAL_EVAL_SAMPLE=64 \
sbatch test.sh
```
- Higher-budget follow-up:
```bash
RECALL_ONLY=1 \
RECALL_INPUT_TOKENS=8192 \
RETRIEVALATTN_FA_GRAPH_FUSED=0 \
RETRIEVALATTN_ROAR_BACKEND=cpp \
RETRIEVALATTN_VALIDATE_PARITY=1 \
RETRIEVALATTN_KV_GRAPH_AB=1 \
RETRIEVALATTN_PARITY_LAYERS=2 \
RETRIEVALATTN_PARITY_HEADS=32 \
RETRIEVALATTN_PARITY_SAMPLE=256 \
RETRIEVALATTN_GRAPH_TRAIN_FRAC=0.9 \
RETRIEVALATTN_GRAPH_SPLIT=stratified \
RETRIEVALATTN_GRAPH_SPLIT_SEED=1234 \
RETRIEVALATTN_PARITY_HOLDOUT_ONLY=1 \
RETRIEVALATTN_TRAVERSAL_EVAL=1 \
RETRIEVALATTN_TRAVERSAL_EVAL_SAMPLE=64 \
RETRIEVALATTN_EXPAND_WIDTH=96 \
RETRIEVALATTN_MIN_VISITS=1024 \
RETRIEVALATTN_MAX_VISITS=8192 \
RETRIEVALATTN_CAND_MULT=8 \
sbatch test.sh
```
- Interpretation:
  - compare `traversal.recall_mean` vs `kv_graph_traversal.recall_mean`
  - if `kv_graph_traversal` collapses while `kv_proxy_traversal` stays close, the grouped-query-built graph is the failure mode
  - current result already showed that; keep q-head graph construction

## Next workstream
- Prioritize decode traversal refactor to beam-search style (paper-aligned):
  - replace/augment adaptive frontier expansion with beam candidate maintenance over built graph,
  - keep token budget fairness fixed (`TOKEN_BUDGET_OVERRIDE=100`) during A/B,
  - compare quality/latency against current adaptive traversal baseline.

## HF RoarGraph FullGPU Smokes
- FullGPU backend selection:
```bash
HF_ATTENTION_MODE=graph_topk_roar \
HF_GRAPH_SEARCH_BACKEND=cuda_fullgpu \
sbatch benchmark/run_generated_memory_hf.sh
```
- Latency-only fixed-decode mode:
```bash
FORCE_MAX_DECODE_STEPS=1 \
HF_ATTENTION_MODE=graph_topk_roar \
HF_GRAPH_SEARCH_BACKEND=cuda_fullgpu \
sbatch benchmark/run_generated_memory_hf.sh
```
- Extension rebuild used for the current fullgpu path:
```bash
module load cuda/12.6.3 python/3.10.4
source .venv/bin/activate
python third_party/RoarGraph/python_ext/setup.py build_ext --inplace
```
- Current fullgpu constraints:
  - kernel supports `head_dim <= 256`
  - `beam_width <= 64`
  - `max_degree <= 16`
  - hub seeds `<= 128`
  - previous seeds `<= 512`
- Key summary fields:
  - `cuda_fallbacks`: should be `0` for a valid fullgpu run
  - `fullgpu_fallback_reasons`: should be empty unless a model violates a fullgpu contract
  - `csr_cuda_uploads`: should be `0` on the overlay-cache fullgpu path
  - `base_csr_cuda_uploads` and `overlay_cuda_uploads`: measure remaining graph-transfer overhead

## HF Candidate Model Inventory
- Supported presets in `benchmark/run_generated_memory_hf.sh`:
```bash
HF_MODEL_PRESET=qwen3_8b
HF_MODEL_PRESET=mistral_nemo_12b
HF_MODEL_PRESET=glm4_9b
```
- Preset model IDs:
  - `qwen3_8b`: `Qwen/Qwen3-8B`
  - `mistral_nemo_12b`: `mistralai/Mistral-Nemo-Instruct-2407`
  - `glm4_9b`: `zai-org/glm-4-9b-chat-hf`
- Submit all three inventory-only smokes:
```bash
bash benchmark/submit_hf_candidate_inventory.sh
```
- Minimal single-model inventory smoke:
```bash
HF_MODEL_PRESET=qwen3_8b \
INVENTORY_ONLY=1 \
HF_ATTENTION_MODE=native \
sbatch benchmark/run_generated_memory_hf.sh
```
- Inspect after completion:
  - `config_summary.json`
  - `tokenizer_summary.json`
  - `attention_inventory.json`
  - `counts_by_kind`
  - `replaceable_full_attention_count`

## HuggingFace Cache Location
- HF scripts default caches to the workspace, not home:
```bash
HF_CACHE_DIR=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/.hf_cache
HF_HOME=${HF_CACHE_DIR}
HF_HUB_CACHE=${HF_CACHE_DIR}/hub
HF_DATASETS_CACHE=${HF_CACHE_DIR}/datasets
TRANSFORMERS_CACHE=${HF_CACHE_DIR}/transformers
```
- `benchmark/run_generated_memory_hf.sh`, `benchmark/run_longbench_v2_hf.sh`, and direct Python entrypoints set these defaults unless already overridden.
- This avoids writing model/dataset caches to `~/.cache/huggingface` on quota-limited home storage.

## Slurm Partition Racing
- Submit the same job to multiple GPU partitions and keep the first viable winner:
```bash
scripts/slurm_race_submit.sh \
  --partitions spgpu,gpu_mig40,gpu-rtx6000 \
  --safe spgpu,gpu_mig40 \
  --risky gpu-rtx6000 \
  --output-base generated_memory_hf_eval_result/race_smoke \
  --job-name hf-race \
  --export 'MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct,LOCAL_FILES_ONLY=1,NUM_SAMPLES=1,NUM_ENTRIES=6,NUM_QUERIES=2,HF_ATTENTION_MODE=graph_topk_roar,HF_GRAPH_SEARCH_BACKEND=cuda_fullgpu' \
  --watch \
  -- benchmark/run_generated_memory_hf.sh
```
- Watcher policy:
  - `spgpu` and `gpu_mig40` are safe; first `RUNNING` job wins immediately.
  - `gpu-rtx6000` is risky; it wins only after `OUTPUT_DIR/hf_job_ready` exists.
  - once a winner is selected, other non-terminal jobs are canceled.
- Output files:
  - manifest: `/tmp/slurm_race_<tag>.tsv`
  - winner: `/tmp/slurm_race_<tag>.winner`
  - watcher log: `.codex/slurm/race-watch-<tag>.log`
- HF jobs write `hf_job_ready` after model load, tokenizer/config load, and attention inventory/patcher install.
