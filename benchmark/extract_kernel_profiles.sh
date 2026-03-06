#!/bin/bash
set -euo pipefail

# Extract compact kernel profiling summaries from slurm logs.
# Usage:
#   ./benchmark/extract_kernel_profiles.sh slurm-ra-kab-*.out
#   ./benchmark/extract_kernel_profiles.sh slurm-44244990.out slurm-44245076.out

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <slurm_log1> [slurm_log2 ...]" >&2
  exit 2
fi

echo "file,ret_prof_n,ret_kernel_mean_s,ret_merge_mean_s,ret_total_mean_s,graph_prof_n,graph_core_mean_s,graph_graph_mean_s,graph_total_mean_s,dbg_n,dbg_cand_mean,dbg_in_bounds_mean,dbg_causal_filtered_mean,dbg_norm_filtered_mean,dbg_locked_calls_mean,dbg_local_calls_mean,dbg_overflow_mean,dbg_merged_rows_mean"

for f in "$@"; do
  if [ ! -f "$f" ]; then
    echo "${f},MISSING" >&2
    continue
  fi

  awk -v file="$f" '
    BEGIN {
      r_n=0; r_k=0; r_m=0; r_t=0;
      g_n=0; g_c=0; g_g=0; g_t=0;
      d_n=0; d_cand=0; d_inb=0; d_causal=0; d_norm=0; d_locked=0; d_local=0; d_over=0; d_merge=0;
    }
    /native_retrieval_profile:/ {
      if (match($0, /kernel=([0-9.]+)s/, a)) { r_k += a[1]; }
      if (match($0, /merge=([0-9.]+)s/, a)) { r_m += a[1]; }
      if (match($0, /total=([0-9.]+)s/, a)) { r_t += a[1]; }
      r_n += 1;
    }
    /native_graph_profile:/ {
      if (match($0, /core=([0-9.]+)s/, a)) { g_c += a[1]; }
      if (match($0, /graph=([0-9.]+)s/, a)) { g_g += a[1]; }
      if (match($0, /total=([0-9.]+)s/, a)) { g_t += a[1]; }
      g_n += 1;
    }
    /native_retrieval_debug:/ {
      if (match($0, /cand_total=([0-9]+)/, a)) { d_cand += a[1]; }
      if (match($0, /in_bounds=([0-9]+)/, a)) { d_inb += a[1]; }
      if (match($0, /causal_filtered=([0-9]+)/, a)) { d_causal += a[1]; }
      if (match($0, /norm_filtered=([0-9]+)/, a)) { d_norm += a[1]; }
      if (match($0, /locked_calls=([0-9]+)/, a)) { d_locked += a[1]; }
      if (match($0, /local_calls=([0-9]+)/, a)) { d_local += a[1]; }
      if (match($0, /overflow_fallback=([0-9]+)/, a)) { d_over += a[1]; }
      if (match($0, /merged_rows=([0-9]+)/, a)) { d_merge += a[1]; }
      d_n += 1;
    }
    END {
      rk = (r_n > 0 ? r_k / r_n : -1);
      rm = (r_n > 0 ? r_m / r_n : -1);
      rt = (r_n > 0 ? r_t / r_n : -1);
      gc = (g_n > 0 ? g_c / g_n : -1);
      gg = (g_n > 0 ? g_g / g_n : -1);
      gt = (g_n > 0 ? g_t / g_n : -1);
      dc = (d_n > 0 ? d_cand / d_n : -1);
      di = (d_n > 0 ? d_inb / d_n : -1);
      df = (d_n > 0 ? d_causal / d_n : -1);
      dn = (d_n > 0 ? d_norm / d_n : -1);
      dl = (d_n > 0 ? d_locked / d_n : -1);
      dj = (d_n > 0 ? d_local / d_n : -1);
      dof = (d_n > 0 ? d_over / d_n : -1);
      dm = (d_n > 0 ? d_merge / d_n : -1);
      printf("%s,%d,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
             file, r_n, rk, rm, rt, g_n, gc, gg, gt, d_n, dc, di, df, dn, dl, dj, dof, dm);
    }
  ' "$f"
done

