# Selector Evaluation Framework

This package is the common path for comparing sparse-attention selector algorithms on saved Q/K/V traces.

## Current Contract

Every algorithm should be implemented as a selector:

```python
result = selector.select(query_state, target_mass=0.98)
```

The selector returns:

- `selected_tokens`: tokens whose exact K/V will be read for attention.
- `candidate_tokens`: tokens scored or considered by the selector.
- `cost`: structured selector/update memory events.
- `metadata`: selector-specific settings such as `nprobe`, page size, group count, or budget.

Metrics are computed outside the selector. Exact K/V read cost is also added outside the selector so all algorithms use the same accounting.

Initial prefill/index construction is intentionally not included in per-query `total_MB`. Selector adapters reset build-time counters after construction and record only online extension/update traffic plus query-time selector traffic.

## Standard Output Schema

The runner writes `samples.csv`, `samples.json`, `summary.csv`, and `summary.json`.

Core columns:

- `algorithm`
- `accounting_mode`
- `online_update_modeled`
- `decode_length`
- `target_mass`
- `selected_tokens`
- `candidate_tokens`
- `attention_mass`
- `false_negative_mass`
- `false_positive_mass`
- `FN_mass`
- `FP_mass`
- `output_cosine`
- `output_relative_l2`
- `output_relative_L2`
- `distribution_js`
- `distribution_JS`
- `selector_MB`
- `selector_MB_per_query`
- `exact_KV_MB`
- `exact_KV_MB_per_query`
- `query_MB`
- `query_MB_per_query`
- `online_update_MB`
- `online_update_cumulative_MB`
- `online_update_MB_per_token`
- `step_MB_per_query`
- `total_MB`
- `total_MB_per_query`

Tables should show unit-explicit cost columns:

- `selector_MB_per_query`: selector/index/router/scoring traffic for one query at this decode length.
- `exact_KV_MB_per_query`: exact K/V traffic for one query at this decode length.
- `online_update_cumulative_MB`: cumulative online maintenance traffic up to this decode length.
- `online_update_MB_per_token`: `online_update_cumulative_MB / decode_length`.
- `step_MB_per_query`: `selector_MB_per_query + exact_KV_MB_per_query + online_update_MB_per_token`.
- `walk_step_MB_per_head` (adaptive-ladder runner): faithful walk traffic — selector + the DEEPEST band read on each axis during the walk + online update. The plain `step_MB_*` fields charge the settled state only, which understates real traffic (escalation probes read lookahead bands; de-escalation refunds nothing). Use the walk field for DRAM/bandwidth claims; see COST_MODEL.md.

Legacy aliases such as `selector_MB`, `exact_KV_MB`, `query_MB`, `online_update_MB`, and `total_MB` are still emitted for compatibility. Do not use them in new tables when the unit-explicit columns are available.

## Quick Smoke

```bash
TRACE=attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
OUTPUT_DIR=attention_efficiency_result/selector_eval_smoke \
SELECTORS=dense,top_mass_oracle \
DECODE_LENGTHS=32000 \
TARGETS=0.97,0.98 \
HEADS=0 \
bash benchmark/selector_eval/runners/run_selector_eval.sh
```

Current gated paged PQ preset:

```bash
TRACE=attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
OUTPUT_DIR=attention_efficiency_result/selector_eval_gated_paged_pq_outputs_t098 \
PRESET=gated_paged_pq_2048_g512_t098 \
DECODE_LENGTHS=500,1000,2000,4000,8000,16000,32000,64000,128000 \
HEADS=0 \
bash benchmark/selector_eval/runners/run_selector_eval.sh
```

## Plot Summary

```bash
LD_LIBRARY_PATH=/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-} \
.venv/bin/python benchmark/selector_eval/reports/plot_summary.py \
  --summary_csv attention_efficiency_result/selector_eval_smoke/summary.csv \
  --output_dir attention_efficiency_result/selector_eval_smoke/plots
```

Compact metrics table:

```bash
LD_LIBRARY_PATH=/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-} \
.venv/bin/python benchmark/selector_eval/reports/print_metrics_table.py \
  --summary_csv attention_efficiency_result/selector_eval_smoke/summary.csv \
  --target 0.98
```

## Slurm Sweep

```bash
TRACE=attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
OUTPUT_DIR=attention_efficiency_result/selector_eval_full_v1 \
SELECTORS=top_mass_oracle,retroinfer_style,pqcache_full_scan,paged_local_pq,paged_routed_pq,ivfpq_periodic_rebuild,sparq_r16,magicpig,retrievalattention_graph \
DECODE_LENGTHS=500,1000,2000,4000,8000,16000,32000,64000,128000 \
TARGETS=0.95,0.97,0.98 \
sbatch -A zhengya98 benchmark/selector_eval/runners/run_selector_eval_slurm.sh
```

Explicit snapshot/online comparison:

```bash
TRACE=attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
OUTPUT_DIR=attention_efficiency_result/selector_eval_snapshot_online_variants_t098_h0 \
SELECTORS=top_mass_oracle,retroinfer_snapshot,retroinfer_online_proxy,pqcache_full_scan_snapshot,pqcache_full_scan_online,gated_paged_pq_snapshot,gated_paged_pq_online,paged_local_pq_snapshot,paged_local_pq_online,sparq_r16,ivfpq_periodic_rebuild,magicpig,retrievalattention_graph \
DECODE_LENGTHS=500,1000,2000,4000,8000,16000,32000,64000,128000 \
TARGETS=0.98 \
HEADS=0 \
sbatch -A zhengya98 benchmark/selector_eval/runners/run_selector_eval_slurm.sh
```

The snapshot variants suppress index-maintenance traffic and should be compared by the separated `selector_MB_per_query` and `exact_KV_MB_per_query` columns.
The online variants/proxies include modeled maintenance traffic and should be compared with `step_MB_per_query` only when their online assumptions are comparable.

## Porting Order

1. `dense` and `top_mass_oracle`: implemented.
2. `retroinfer_style`: implemented as contiguous chunk centroid routing with exact reads for selected cluster members.
3. `pqcache_full_scan`: implemented as a framework-port baseline. It rebuilds PQ for the current context and records build memory as `online_update`; replacing it with a maintained online index should not change the runner or metrics.
4. `paged_local_pq` and `gated_paged_pq`: implemented as adapters over `benchmark/online_ivfpq_simulator.py::PagedLocalPQIndex`. `paged_routed_pq` remains accepted as a compatibility alias.
5. `ivfpq_frozen_append`, `ivfpq_online_centroid`, and `ivfpq_periodic_rebuild`: implemented as adapters over `benchmark/online_ivfpq_simulator.py::OnlineIVFPQIndex`.
6. `sparq` / `sparq_r<N>`: implemented.
7. `magicpig` / `magicpig_k<N>_l<N>`: implemented as a hash-sidecar selector baseline.
8. `retrievalattention_graph` / `ra_graph`: implemented as a causal Q-K provenance graph replay baseline with bounded traversal.
9. Remaining: production-cache parity for `retrievalattention`.

## Additional Docs

- Cost accounting: `benchmark/selector_eval/COST_MODEL.md`
- Metric hierarchy: `benchmark/selector_eval/METRICS.md`
- Latest compact results page: `notes/selector_eval_latest_results.md`

## Verification

Core tests currently live in:

```text
tests/test_selector_eval_core.py
```

The repo venv does not currently include `pytest`; until that is added, run:

```bash
LD_LIBRARY_PATH=/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-} .venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path
p = Path("tests/test_selector_eval_core.py")
spec = importlib.util.spec_from_file_location("test_selector_eval_core", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in sorted(dir(mod)):
    if name.startswith("test_"):
        getattr(mod, name)()
        print("PASS", name)
PY
```
