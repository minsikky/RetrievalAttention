PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fullscan_pq_topk", &fullscan_pq_topk, "Fullscan page-PQ top-k selector (CUDA)");
  m.def(
      "fullscan_pq_topk_scores",
      &fullscan_pq_topk_scores,
      "Fullscan page-PQ top-k selector plus dense approximate score matrix (CUDA)");
  m.def(
      "gqa_fullscan_pq_topk",
      &gqa_fullscan_pq_topk,
      "Fullscan page-PQ top-k selector for grouped-query attention (CUDA)");
  m.def(
      "gqa_fullscan_pq_topk_scores",
      &gqa_fullscan_pq_topk_scores,
      "Fullscan page-PQ top-k selector plus dense approximate scores for grouped-query attention (CUDA)");
  m.def(
      "gqa_causal_fullscan_pq_topk",
      &gqa_causal_fullscan_pq_topk,
      "Causal fullscan page-PQ top-k selector for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_fullscan_pq_topk_scores",
      &gqa_causal_fullscan_pq_topk_scores,
      "Causal fullscan page-PQ top-k selector plus dense approximate scores for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_fullscan_pq_topk_fused",
      &gqa_causal_fullscan_pq_topk_fused,
      "Causal fused page-PQ top-k selector for grouped-query prefill, exact through budget 64 (CUDA)");
  m.def(
      "gqa_causal_fullscan_pq_topk_fused_force",
      &gqa_causal_fullscan_pq_topk_fused_force,
      "Causal fused page-PQ top-k selector with force_mode 0=auto, 1=smallscan, 2=localtopk (CUDA)");
  m.def(
      "gqa_causal_fullscan_pq_top_pages",
      &gqa_causal_fullscan_pq_top_pages,
      "Causal fullscan page-PQ top-page selector for grouped-query prefill (CUDA)");
  m.def(
      "exact_selected_attention",
      &exact_selected_attention,
      "Exact selected-token attention for one KV group (CUDA)");
  m.def(
      "gqa_causal_exact_selected_attention",
      &gqa_causal_exact_selected_attention,
      "Causal exact selected-token attention for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_attention",
      &gqa_causal_vpq_selected_attention,
      "Causal selected-token attention with page-local V-PQ values for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_attention_vpagesize",
      &gqa_causal_vpq_selected_attention_vpagesize,
      "Causal selected-token attention with separately paged V-PQ values for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_attention_mixed_vpagesize",
      &gqa_causal_vpq_selected_attention_mixed_vpagesize,
      "Causal selected-token attention with separately paged V-PQ values and fixed exact-value top-k (CUDA)");
  m.def(
      "gqa_causal_vpq_tail_attention",
      &gqa_causal_vpq_tail_attention,
      "Causal selected-token attention with compressed page-PQ tail for grouped-query prefill (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_tail_attention",
      &gqa_causal_vpq_selected_tail_attention,
      "Causal selected-token attention with mixed selected V-PQ values and compressed page-PQ tail (CUDA)");
  m.def(
      "gqa_causal_vpq_tail_from_scores",
      &gqa_causal_vpq_tail_from_scores,
      "Causal selected-token attention with compressed V-PQ tail using precomputed PQ selector scores (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_tail_from_scores",
      &gqa_causal_vpq_selected_tail_from_scores,
      "Causal mixed selected V-PQ attention with compressed V-PQ tail using precomputed PQ selector scores (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_tail_from_scores_counts",
      &gqa_causal_vpq_selected_tail_from_scores_counts,
      "Causal mixed selected V-PQ/tail attention with per-query exact selected-value counts (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_tail_from_scores_mass",
      &gqa_causal_vpq_selected_tail_from_scores_mass,
      "Causal mixed selected V-PQ/tail attention with in-kernel selected-value mass exactness (CUDA)");
  m.def(
      "gqa_causal_vpq_selected_tail_from_scores_mass_min",
      &gqa_causal_vpq_selected_tail_from_scores_mass_min,
      "Causal mixed selected V-PQ/tail attention with in-kernel selected-value mass exactness and min exact selected values (CUDA)");
  m.def(
      "gqa_decode_vpq_tail_from_scores",
      &gqa_decode_vpq_tail_from_scores,
      "Decode selected-token attention with compressed V-PQ tail using precomputed PQ selector scores (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_from_scores",
      &gqa_decode_vpq_selected_tail_from_scores,
      "Decode mixed selected V-PQ attention with compressed V-PQ tail using precomputed PQ selector scores (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_from_scores_counts",
      &gqa_decode_vpq_selected_tail_from_scores_counts,
      "Decode mixed selected V-PQ/tail attention with per-head exact selected-value counts (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_scores",
      &gqa_decode_vpq_selected_tail_agg_from_scores,
      "Decode mixed selected V-PQ attention with code-weight-aggregated compressed V-PQ tail (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_scores_counts",
      &gqa_decode_vpq_selected_tail_agg_from_scores_counts,
      "Decode mixed selected V-PQ attention with code-weight-aggregated tail and per-head exact selected-value counts (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts",
	      &gqa_decode_geometric_accept_counts,
	      "Decode strict-geometric confidence accepted selected-token counts (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq",
	      &gqa_decode_geometric_accept_counts_vpq,
	      "Decode strict-geometric confidence counts with selector-rank V-PQ selected values (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq_tail_stability",
	      &gqa_decode_geometric_accept_counts_vpq_tail_stability,
	      "Decode geometric confidence counts comparing compressed-tail outputs at tail/probe budgets (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq_proxy",
	      &gqa_decode_geometric_accept_counts_vpq_proxy,
	      "Decode geometric confidence counts with in-kernel proxy mass/correlation gating (CUDA)");
	  m.def(
	      "gqa_decode_ranked_exact_logits",
	      &gqa_decode_ranked_exact_logits,
	      "Decode exact QK logits for ranked candidates only (CUDA)");
	  m.def(
	      "gqa_decode_full_exact_logits",
	      &gqa_decode_full_exact_logits,
	      "Decode full exact QK logits for all context tokens (CUDA)");
	  m.def(
	      "gqa_decode_full_exact_logits_grouped",
	      &gqa_decode_full_exact_logits_grouped,
	      "Decode full exact QK logits while sharing K loads across GQA heads in each KV group (CUDA)");
	  m.def(
	      "gqa_decode_full_exact_logits_t_cublas",
	      &gqa_decode_full_exact_logits_t_cublas,
	      "Decode full exact QK logits from cached transposed K using cuBLAS strided-batched GEMM (CUDA)");
	  m.def(
	      "gqa_decode_token_exact_logits",
	      &gqa_decode_token_exact_logits,
	      "Decode exact QK logits for arbitrary per-head token lists without base-window filtering (CUDA)");
	  m.def(
	      "gqa_decode_ranked_exact_logits_with_base_lse",
	      &gqa_decode_ranked_exact_logits_with_base_lse,
	      "Decode exact QK logits for ranked candidates and base-window logsumexp (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy",
	      &gqa_decode_geometric_accept_counts_vpq_mass_min_proxy,
	      "Decode geometric confidence counts with selected-mass V-PQ exactness and proxy gating (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits",
	      &gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits,
	      "Decode geometric confidence counts with selected-mass V-PQ exactness, proxy gating, and precomputed exact logits (CUDA)");
	  m.def(
	      "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds",
	      &gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds,
	      "Decode geometric confidence counts with selected-mass V-PQ exactness, proxy gating, precomputed exact logits, and per-budget thresholds (CUDA)");
	  m.def(
	      "gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds",
	      &gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds,
	      "Decode geometric confidence and final selected/tail output with selected-mass V-PQ exactness, proxy gating, precomputed exact logits, and per-budget thresholds (CUDA)");
	  m.def(
	      "selected_mass_thresholds_from_topk",
	      &selected_mass_thresholds_from_topk,
	      "Build selected-mass exact-V thresholds from sorted top-k exact logits (CUDA)");
	  m.def(
	      "joint_vprefix_outputs",
	      &joint_vprefix_outputs,
	      "Build residual-risk exact-V prefix output grid from top-risk token order (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_precision",
	      &joint_vprefix_outputs_precision,
	      "Build fused progressive-precision V-prefix output grids (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_precision_tokpar",
	      &joint_vprefix_outputs_precision_tokpar,
	      "Build token-parallel progressive-precision V-prefix output grids (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_from_risk",
	      &joint_vprefix_outputs_from_risk,
	      "Build residual-risk exact-V output grid by sorting risk scores natively (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_precision_from_risk",
	      &joint_vprefix_outputs_precision_from_risk,
	      "Build fused progressive-precision V-prefix grids with native risk sorting (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_precision_from_risk_tokpar",
	      &joint_vprefix_outputs_precision_from_risk_tokpar,
	      "Build token-parallel progressive-precision V-prefix grids with native risk sorting (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_precision_from_risk_tokpar_batched",
	      &joint_vprefix_outputs_precision_from_risk_tokpar_batched,
	      "Build KV-head-batched token-parallel progressive-precision V-prefix grids with one segmented sort (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_from_grouped_risk",
	      &joint_vprefix_outputs_from_grouped_risk,
	      "Build grouped residual-risk exact-V output grid by sorting risk scores natively (CUDA)");
		  m.def(
		      "joint_vprefix_outputs_from_grouped_risk_batched",
		      &joint_vprefix_outputs_from_grouped_risk_batched,
		      "Build grouped residual-risk exact-V output grid from batched grouped rows (CUDA)");
		  m.def(
		      "joint_vprefix_outputs_from_grouped_risk_batched_vpq",
		      &joint_vprefix_outputs_from_grouped_risk_batched_vpq,
		      "Build grouped residual-risk exact-V output grid with compact V-PQ residual reconstruction (CUDA)");
		  m.def(
		      "joint_grouped_risk_sort_temp_bytes",
	      &joint_grouped_risk_sort_temp_bytes,
	      "Return CUB temp-storage bytes required for grouped residual-risk segmented sort (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_from_grouped_risk_batched_workspace",
	      &joint_vprefix_outputs_from_grouped_risk_batched_workspace,
	      "Build grouped residual-risk exact-V output grid using caller-provided CUDA workspaces");
	  m.def(
	      "joint_vprefix_outputs_from_grouped_risk_batched_strided_workspace",
	      &joint_vprefix_outputs_from_grouped_risk_batched_strided_workspace,
	      "Build grouped residual-risk exact-V output grid from padded probability workspaces");
	  m.def(
	      "joint_vprefix_outputs_from_grouped_risk_topk_batched",
	      &joint_vprefix_outputs_from_grouped_risk_topk_batched,
	      "Build grouped residual-risk exact-V output grid using top-k risk rows (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_from_grouped_scores_batched",
	      &joint_vprefix_outputs_from_grouped_scores_batched,
	      "Build grouped residual-risk V-prefix outputs directly from mixed score rows without materializing probabilities (CUDA)");
	  m.def(
	      "joint_vprefix_outputs_from_grouped_scores_batched_workspace",
	      &joint_vprefix_outputs_from_grouped_scores_batched_workspace,
	      "Build grouped residual-risk V-prefix outputs directly from mixed score rows using caller-provided CUDA workspaces");
	  m.def(
	      "joint_vpq_base_outputs_from_probs",
	      &joint_vpq_base_outputs_from_probs,
	      "Build V-PQ reconstructed base outputs from probabilities by page/code aggregation (CUDA)");
	  m.def(
	      "joint_softmax_base_outputs",
	      &joint_softmax_base_outputs,
	      "Compute softmax probabilities and reconstructed base outputs from a joint score grid (CUDA)");
	  m.def(
	      "joint_softmax_base_outputs_cublas",
	      &joint_softmax_base_outputs_cublas,
	      "Compute softmax probabilities and reconstructed base outputs with cuBLAS base GEMM (CUDA)");
	  m.def(
	      "joint_softmax_base_outputs_workspace",
	      &joint_softmax_base_outputs_workspace,
	      "Compute softmax probabilities and reconstructed base outputs into caller-provided workspaces (CUDA)");
	  m.def(
	      "joint_softmax_base_outputs_strided_workspace",
	      &joint_softmax_base_outputs_strided_workspace,
	      "Compute softmax/base into padded caller-provided probability workspace (CUDA)");
	  m.def(
	      "joint_softmax_base_outputs_workspace_cublas",
	      &joint_softmax_base_outputs_workspace_cublas,
	      "Compute softmax probabilities and reconstructed base outputs with cuBLAS base GEMM into caller-provided workspaces (CUDA)");
	  m.def(
	      "joint_softmax_base_outputs_grouped_cublas",
	      &joint_softmax_base_outputs_grouped_cublas,
	      "Compute grouped softmax probabilities and reconstructed base outputs with strided batched cuBLAS base GEMM (CUDA)");
	  m.def(
	      "joint_vpq_append_exact_suffix",
	      &joint_vpq_append_exact_suffix,
	      "Append exact unsealed-suffix V rows into persistent V-PQ sidecars and zero residual/error stats (CUDA)");
	  m.def(
	      "joint_vpq_sidecars_from_pack",
	      &joint_vpq_sidecars_from_pack,
	      "Construct V-PQ reconstructed values, residuals, and residual-risk stats from packed page codebooks/codes (CUDA)");
	  m.def(
	      "joint_vpq_append_exact_suffix_grouped",
	      &joint_vpq_append_exact_suffix_grouped,
	      "Append exact unsealed-suffix V rows into grouped persistent V-PQ sidecars and zero residual/error stats (CUDA)");
	  m.def(
	      "joint_mixed_softmax_base_outputs",
	      &joint_mixed_softmax_base_outputs,
	      "Build mixed joint score rows implicitly, softmax them, and compute reconstructed base outputs (CUDA)");
	  m.def(
	      "joint_mixed_softmax_base_outputs_rankpos",
	      &joint_mixed_softmax_base_outputs_rankpos,
	      "Build mixed joint score rows with rank-position metadata, softmax them, and compute reconstructed base outputs (CUDA)");
	  m.def(
	      "joint_mixed_softmax_base_outputs_tokenfit_scaled",
	      &joint_mixed_softmax_base_outputs_tokenfit_scaled,
	      "Build mixed joint score rows with token-list affine fit and rank-position membership, softmax them, and compute reconstructed base outputs (CUDA)");
	  m.def(
	      "joint_mixed_softmax_base_outputs_sparse_exact_tokenfit_scaled",
	      &joint_mixed_softmax_base_outputs_sparse_exact_tokenfit_scaled,
	      "Build sparse-exact mixed score rows without a dense exact table, softmax them, and compute reconstructed base outputs (CUDA)");
	  m.def(
	      "joint_mixed_score_grid",
	      &joint_mixed_score_grid,
	      "Build joint K/V mixed exact-K plus calibrated K-PQ score grid (CUDA)");
	  m.def(
	      "joint_mixed_score_grid_workspace",
	      &joint_mixed_score_grid_workspace,
	      "Build canonical joint K/V mixed score grid into caller-provided CUDA workspaces");
	  m.def(
	      "joint_mixed_score_grid_rankpos_nocalib_workspace",
	      &joint_mixed_score_grid_rankpos_nocalib_workspace,
	      "Build raw K-PQ mixed score grid into caller-provided output and rank-position workspaces");
	  m.def(
	      "joint_mixed_score_grid_nocalib_scatter_workspace",
	      &joint_mixed_score_grid_nocalib_scatter_workspace,
	      "Build raw K-PQ mixed score grid by filling PQ rows then scattering exact selected logits");
	  m.def(
	      "joint_mixed_score_grid_scaled",
	      &joint_mixed_score_grid_scaled,
	      "Build joint K/V mixed score grid while applying a scalar K-PQ logit scale in CUDA");
	  m.def(
	      "joint_mixed_score_grid_rankpos",
	      &joint_mixed_score_grid_rankpos,
	      "Build joint K/V mixed score grid using rank-position selected-token metadata (CUDA)");
	  m.def(
	      "joint_mixed_score_grid_no_exact_fill",
	      &joint_mixed_score_grid_no_exact_fill,
	      "Build joint K/V mixed score grid without initial exact-fill; caller must guarantee indexed/base coverage (CUDA)");
	  m.def(
	      "joint_mixed_score_grid_no_exact_fill_scaled",
	      &joint_mixed_score_grid_no_exact_fill_scaled,
	      "Build no-fill joint K/V mixed score grid while applying a scalar K-PQ logit scale in CUDA");
	  m.def(
	      "joint_mixed_score_grid_tokenfit_scaled",
	      &joint_mixed_score_grid_tokenfit_scaled,
	      "Build joint K/V mixed score grid with mask-free selected-token affine fit and in-kernel PQ scaling");
	  m.def(
	      "joint_mixed_score_grid_no_exact_fill_tokenfit_scaled",
	      &joint_mixed_score_grid_no_exact_fill_tokenfit_scaled,
	      "Build no-fill joint K/V mixed score grid with mask-free selected-token affine fit and in-kernel PQ scaling");
	  m.def(
	      "joint_mixed_score_grid_sparse_exact_tokenfit_scaled",
	      &joint_mixed_score_grid_sparse_exact_tokenfit_scaled,
	      "Build no-fill joint K/V mixed score grid from sparse base/ranked exact logits without a full exact table");
	  m.def(
	      "joint_sparse_exact_score_table",
	      &joint_sparse_exact_score_table,
	      "Build a sparse exact-logit table from base and ranked token exact-logit lists (CUDA)");
	  m.def(
	      "joint_select_policy",
	      &joint_select_policy,
	      "Select final joint K/V budget indices from an output grid using the online stability policy (CUDA)");
  m.def(
      "joint_select_policy_grouped_flat",
      &joint_select_policy_grouped_flat,
      "Select final joint K/V budget indices and outputs from grouped flat output grids (CUDA)");
  m.def(
      "joint_select_policy_grouped_flat_no_mb",
      &joint_select_policy_grouped_flat_no_mb,
      "Select final joint K/V budget indices and outputs from grouped flat output grids for non-MB policies (CUDA)");
  m.def(
      "joint_select_policy_grouped_flat_no_mb_accounting",
      &joint_select_policy_grouped_flat_no_mb_accounting,
      "Select final joint K/V budget indices, outputs, and fused accounting sums for non-MB policies (CUDA)");
  m.def(
      "joint_select_policy_grouped_flat_staged_no_mb",
      &joint_select_policy_grouped_flat_staged_no_mb,
      "Select staged joint K/V budget indices, outputs, and per-group boundary flags for non-MB policies (CUDA)");
  m.def(
      "joint_grouped_accounting_sums",
      &joint_grouped_accounting_sums,
      "Aggregate grouped joint K/V logical and physical MB accounting from selected policy indices (CUDA)");
  m.def(
      "joint_grouped_accounting_accumulate",
      &joint_grouped_accounting_accumulate,
      "Accumulate grouped joint K/V logical and physical MB accounting into a caller-owned CUDA tensor");
  m.def(
      "joint_rank_prefix_tokens",
      &joint_rank_prefix_tokens,
      "Build per-head ranked token prefixes from dense PQ scores using native segmented sort (CUDA)");
  m.def(
      "joint_rank_prefix_sort_temp_bytes",
      &joint_rank_prefix_sort_temp_bytes,
      "Temporary storage size for rank-prefix segmented sort workspace (CUDA)");
  m.def(
      "joint_rank_prefix_tokens_workspace",
      &joint_rank_prefix_tokens_workspace,
      "Build per-head ranked token prefixes using caller-provided native segmented-sort workspace (CUDA)");
  m.def(
      "joint_budget_prefix_tokens",
      &joint_budget_prefix_tokens,
      "Build nested budget-prefix token sets from dense PQ scores without full sorted order (CUDA diagnostic)");
  m.def(
      "joint_select_policy_from_grouped_risk",
      &joint_select_policy_from_grouped_risk,
	      "Select final joint K/V budgets and outputs directly from grouped residual-risk rows (CUDA)");
  m.def(
      "joint_select_policy_from_grouped_risk_batched",
      &joint_select_policy_from_grouped_risk_batched,
      "Select final joint K/V budgets and outputs directly from batched grouped residual-risk rows (CUDA)");
  m.def(
      "joint_select_policy_from_grouped_risk_no_mb",
      &joint_select_policy_from_grouped_risk_no_mb,
      "Select final joint K/V budgets and outputs from grouped residual-risk rows for non-MB policies (CUDA)");
  m.def(
      "joint_select_policy_from_grouped_risk_batched_no_mb",
      &joint_select_policy_from_grouped_risk_batched_no_mb,
      "Select final joint K/V budgets and outputs from batched grouped residual-risk rows for non-MB policies (CUDA)");
  m.def(
      "joint_select_policy_from_grouped_risk_intervals_batched_no_mb",
      &joint_select_policy_from_grouped_risk_intervals_batched_no_mb,
      "Select final joint K/V budgets from batched grouped residual-risk interval sums for non-MB policies (CUDA)");
  m.def(
      "joint_vprefix_outputs_from_grouped_merge_risk_batched",
      &joint_vprefix_outputs_from_grouped_merge_risk_batched,
      "Build grouped V-prefix outputs by merging base PQ-risk and exact selected-risk orders (CUDA diagnostic)");
  m.def(
      "joint_select_policy_from_grouped_scores_intervals_batched_no_mb",
      &joint_select_policy_from_grouped_scores_intervals_batched_no_mb,
      "Select final joint K/V budgets directly from grouped score-grid residual-risk interval sums for non-MB policies (CUDA)");
  m.def(
      "joint_select_policy_from_grouped_scores_probs_intervals_batched_no_mb",
      &joint_select_policy_from_grouped_scores_probs_intervals_batched_no_mb,
      "Select final joint K/V budgets from grouped score-grid probabilities reused across base/risk intervals for non-MB policies (CUDA)");
  m.def(
      "joint_select_policy_from_grouped_scores_topk_intervals_batched_no_mb",
      &joint_select_policy_from_grouped_scores_topk_intervals_batched_no_mb,
      "Select final joint K/V budgets from grouped score-grid exact top-k residual-risk intervals for non-MB policies (CUDA)");
  m.def(
      "joint_softmax_base_outputs_grouped",
      &joint_softmax_base_outputs_grouped,
      "Compute grouped softmax probabilities and V-PQ base outputs from grouped score grids (CUDA)");
  m.def(
      "joint_mixed_select_policy_intervals_no_mb",
      &joint_mixed_select_policy_intervals_no_mb,
      "Select final joint K/V budgets directly from mixed exact-K/K-PQ score ingredients for non-MB policies (CUDA)");
  m.def(
      "joint_mixed_select_policy_intervals_rankpos_no_calib_no_mb",
      &joint_mixed_select_policy_intervals_rankpos_no_calib_no_mb,
      "Select final joint K/V budgets from no-calibration rank-position mixed score ingredients for non-MB policies (CUDA)");
  m.def(
      "joint_mixed_select_policy_merge_rankpos_no_calib_no_mb",
      &joint_mixed_select_policy_merge_rankpos_no_calib_no_mb,
      "Select final joint K/V budgets by merging base PQ-risk and exact selected-risk orders for non-MB policies (CUDA diagnostic)");
	  m.def(
	      "gqa_causal_geometric_accept_counts",
	      &gqa_causal_geometric_accept_counts,
	      "Causal prefill strict-geometric confidence accepted selected-token counts (CUDA)");
	  m.def(
	      "gqa_causal_geometric_accept_counts_vpq",
	      &gqa_causal_geometric_accept_counts_vpq,
	      "Causal prefill strict-geometric confidence counts with selector-rank V-PQ selected values (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_scores_mass",
      &gqa_decode_vpq_selected_tail_agg_from_scores_mass,
      "Decode mixed selected V-PQ attention with code-weight-aggregated tail and in-kernel selected-value mass exactness (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_scores_mass_min",
      &gqa_decode_vpq_selected_tail_agg_from_scores_mass_min,
      "Decode mixed selected V-PQ attention with code-weight-aggregated tail, in-kernel selected-value mass exactness, and min exact selected values (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_from_logits_mass_min",
      &gqa_decode_vpq_selected_from_logits_mass_min,
      "Decode selected-only mixed V-PQ attention using precomputed exact ranked logits and selected-value mass/min exactness (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_logits_mass_min",
      &gqa_decode_vpq_selected_tail_agg_from_logits_mass_min,
      "Decode mixed selected V-PQ attention with code-weight-aggregated tail using precomputed exact ranked logits and selected-value mass/min exactness (CUDA)");
  m.def(
      "gqa_decode_vpq_selected_tail_agg_from_logits_mass_min_thresholds",
      &gqa_decode_vpq_selected_tail_agg_from_logits_mass_min_thresholds,
      "Decode mixed selected V-PQ attention with precomputed exact ranked logits and selected-value mass/min thresholds (CUDA)");
  m.def(
      "gqa_decode_fullscan_vpq_selected_tail_agg",
      &gqa_decode_fullscan_vpq_selected_tail_agg,
      "Decode fullscan page-PQ selector plus mixed selected V-PQ/tail attention in one native call (CUDA)");
  m.def(
      "gqa_decode_fullscan_vpq_selected_tail_agg_mass_min",
      &gqa_decode_fullscan_vpq_selected_tail_agg_mass_min,
      "Decode fullscan page-PQ selector plus mixed selected V-PQ/tail attention with selected-value mass/min exactness in one native call (CUDA)");
  m.def(
      "gqa_decode_scoreless_fullscan_vpq_tail",
      &gqa_decode_scoreless_fullscan_vpq_tail,
      "Decode fused page-PQ top-k plus scoreless mixed V-PQ/tail attention without materializing dense PQ scores (CUDA)");
  m.def(
      "gqa_exact_selected_attention",
      &gqa_exact_selected_attention,
      "Exact selected-token attention for grouped-query attention (CUDA)");
}
