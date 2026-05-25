std::vector<torch::Tensor> gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits,
    torch::Tensor approx_exact_thresholds,
    torch::Tensor approx_exact_threshold_sels,
    torch::Tensor probe_exact_thresholds,
    torch::Tensor probe_exact_threshold_sels,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    double exact_value_mass,
    int64_t exact_value_min_top,
    double scale,
    double proxy_mass_min,
    double proxy_tail_mass_max,
    double pq_corr_min,
    double pq_relrmse_max,
    bool calibrate_proxy,
    bool probe_includes_tail,
    double tail_blend) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(1);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto counts = torch::empty({heads}, queries.options().dtype(torch::kLong));
  auto outputs = torch::empty({heads, dim}, queries.options().dtype(torch::kFloat32));
  if (heads == 0) {
    return {counts, outputs};
  }
  if (dim == 0 || kv_heads == 0 || total_tokens == 0 || pages == 0 || page_size <= 0) {
    counts.zero_();
    outputs.zero_();
    return {counts, outputs};
  }

  const int threads = decode_geometric_threads();
  const int shared_bytes = threads * 2 * sizeof(float);
  const int64_t pages_per_block = decode_tail_pages_per_block();
  const int64_t tail_blocks = std::max<int64_t>(1, (pages + pages_per_block - 1) / pages_per_block);
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  auto opts = queries.options().dtype(torch::kFloat32);
  auto long_opts = queries.options().dtype(torch::kLong);
  auto base_tokens = torch::empty({heads, max_base}, long_opts);
  auto base_logits = torch::empty({heads, max_base}, opts);
  auto base_counts = torch::zeros({heads}, long_opts);
  auto filtered_ranked_logits = torch::empty({heads, ranked}, opts);
  auto partial_max = torch::empty({heads, tail_blocks}, opts);
  auto max_logits = torch::empty({heads}, opts);
  auto partial_sum = torch::empty({heads, tail_blocks}, opts);
  auto tail_denoms = torch::empty({heads}, opts);
  auto code_weight_sums = torch::zeros({heads, pages, value_subvecs, value_centroids}, opts);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_decode_geometric_output_base_keys",
      [&] {
        gqa_decode_base_logits_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
            queries.data_ptr<float>(),
            keys.data_ptr<scalar_t>(),
            base_tokens.data_ptr<int64_t>(),
            base_logits.data_ptr<float>(),
            base_counts.data_ptr<int64_t>(),
            heads,
            kv_heads,
            dim,
            total_tokens,
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            page_size,
            max_base,
            group_size,
            query_context_len,
            static_prefix,
            static_suffix,
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int filter_threads = 256;
  const int64_t filter_total = heads * ranked;
  const int filter_blocks = static_cast<int>((filter_total + filter_threads - 1) / filter_threads);
  gqa_decode_filter_ranked_logits_input_kernel<<<filter_blocks, filter_threads, 0, stream>>>(
      ranked_tokens.data_ptr<int64_t>(),
      ranked_scores.data_ptr<float>(),
      ranked_logits.data_ptr<float>(),
      filtered_ranked_logits.data_ptr<float>(),
      heads,
      ranked,
      query_context_len,
      static_prefix,
      static_suffix,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 tail_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(tail_blocks));
  gqa_decode_tail_partial_max_nomask_kernel<<<tail_grid, threads, threads * sizeof(float), stream>>>(
      dense_pq_scores.data_ptr<float>(),
      page_starts.data_ptr<int64_t>(),
      partial_max.data_ptr<float>(),
      heads,
      pages,
      page_size,
      tail_blocks,
      pages_per_block,
      query_context_len,
      static_prefix,
      static_suffix,
      static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_final_max_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_max.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      filtered_ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (value_codes.scalar_type() == torch::kUInt8) {
    gqa_decode_tail_sum_codeweights_nomask_kernel<uint8_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
        dense_pq_scores.data_ptr<float>(),
        max_logits.data_ptr<float>(),
        value_codes.data_ptr<uint8_t>(),
        page_starts.data_ptr<int64_t>(),
        partial_sum.data_ptr<float>(),
        code_weight_sums.data_ptr<float>(),
        heads,
        kv_heads,
        pages,
        page_size,
        value_subvecs,
        value_centroids,
        tail_blocks,
        pages_per_block,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        static_cast<float>(scale));
  } else {
    gqa_decode_tail_sum_codeweights_nomask_kernel<int64_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
        dense_pq_scores.data_ptr<float>(),
        max_logits.data_ptr<float>(),
        value_codes.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        partial_sum.data_ptr<float>(),
        code_weight_sums.data_ptr<float>(),
        heads,
        kv_heads,
        pages,
        page_size,
        value_subvecs,
        value_centroids,
        tail_blocks,
        pages_per_block,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        static_cast<float>(scale));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_tail_denom_from_partials_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_sum.data_ptr<float>(),
      tail_denoms.data_ptr<float>(),
      heads,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t threshold_steps =
      (approx_exact_thresholds.defined() && approx_exact_thresholds.numel() > 0) ? approx_exact_thresholds.size(1) : 0;
  const float* approx_exact_thresholds_ptr =
      threshold_steps > 0 ? approx_exact_thresholds.data_ptr<float>() : nullptr;
  const int64_t* approx_exact_threshold_sels_ptr =
      threshold_steps > 0 ? approx_exact_threshold_sels.data_ptr<int64_t>() : nullptr;
  const float* probe_exact_thresholds_ptr =
      threshold_steps > 0 ? probe_exact_thresholds.data_ptr<float>() : nullptr;
  const int64_t* probe_exact_threshold_sels_ptr =
      threshold_steps > 0 ? probe_exact_threshold_sels.data_ptr<int64_t>() : nullptr;

  const char* dim_scan_output_env = std::getenv("SELECTOR_PQ_FUSED_DIM_SCAN_OUTPUT");
  const bool use_dim_scan_output =
      dim_scan_output_env != nullptr && dim_scan_output_env[0] != '\0' && dim_scan_output_env[0] != '0' &&
      threshold_steps > 0 && exact_value_mass > 0.0 && tail_blend == 1.0;
  if (use_dim_scan_output) {
    return gqa_decode_geometric_dim_scan_output_vpq_mass_min_proxy_from_logits_thresholds_cuda(
        values,
        dense_pq_scores,
        value_codebooks,
        value_codes,
        page_starts,
        base_tokens,
        base_logits,
        base_counts,
        ranked_tokens,
        ranked_scores,
        filtered_ranked_logits,
        max_logits,
        tail_denoms,
        code_weight_sums,
        counts,
        outputs,
        approx_exact_thresholds_ptr,
        approx_exact_threshold_sels_ptr,
        probe_exact_thresholds_ptr,
        probe_exact_threshold_sels_ptr,
        threads,
        heads,
        kv_heads,
        ranked,
        dim,
        total_tokens,
        pages,
        page_size,
        max_base,
        value_subvecs,
        value_centroids,
        value_subdim,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        min_budget,
        max_budget,
        granularity,
        growth,
        probe_scale,
        rel_l2_max,
        exact_value_mass,
        exact_value_min_top,
        scale,
        proxy_mass_min,
        proxy_tail_mass_max,
        pq_corr_min,
        pq_relrmse_max,
        calibrate_proxy,
        probe_includes_tail,
        threshold_steps,
        opts);
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "gqa_decode_geometric_output_counts_values",
      [&] {
        using value_scalar_t = scalar_t;
        if (value_codes.scalar_type() == torch::kUInt8) {
          gqa_decode_geometric_accept_counts_codeweights_kernel<value_scalar_t, uint8_t>
              <<<static_cast<int>(heads), threads, shared_bytes, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  dense_pq_scores.data_ptr<float>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<uint8_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  filtered_ranked_logits.data_ptr<float>(),
                  max_logits.data_ptr<float>(),
                  tail_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  approx_exact_thresholds_ptr,
                  approx_exact_threshold_sels_ptr,
                  probe_exact_thresholds_ptr,
                  probe_exact_threshold_sels_ptr,
                  counts.data_ptr<int64_t>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  min_budget,
                  max_budget,
                  granularity,
                  static_cast<float>(growth),
                  static_cast<float>(probe_scale),
                  static_cast<float>(rel_l2_max),
                  0,
                  static_cast<float>(exact_value_mass),
                  exact_value_min_top,
                  static_cast<float>(scale),
                  probe_includes_tail,
                  threshold_steps);
        } else {
          gqa_decode_geometric_accept_counts_codeweights_kernel<value_scalar_t, int64_t>
              <<<static_cast<int>(heads), threads, shared_bytes, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  dense_pq_scores.data_ptr<float>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<int64_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  filtered_ranked_logits.data_ptr<float>(),
                  max_logits.data_ptr<float>(),
                  tail_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  approx_exact_thresholds_ptr,
                  approx_exact_threshold_sels_ptr,
                  probe_exact_thresholds_ptr,
                  probe_exact_threshold_sels_ptr,
                  counts.data_ptr<int64_t>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  min_budget,
                  max_budget,
                  granularity,
                  static_cast<float>(growth),
                  static_cast<float>(probe_scale),
                  static_cast<float>(rel_l2_max),
                  0,
                  static_cast<float>(exact_value_mass),
                  exact_value_min_top,
                  static_cast<float>(scale),
                  probe_includes_tail,
                  threshold_steps);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_proxy_gate_counts_kernel<<<static_cast<int>(heads), threads, threads * 8 * sizeof(float), stream>>>(
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      filtered_ranked_logits.data_ptr<float>(),
      ranked_scores.data_ptr<float>(),
      counts.data_ptr<int64_t>(),
      heads,
      ranked,
      max_base,
      max_budget,
      static_cast<float>(scale),
      static_cast<float>(proxy_mass_min),
      static_cast<float>(proxy_tail_mass_max),
      static_cast<float>(pq_corr_min),
      static_cast<float>(pq_relrmse_max),
      calibrate_proxy);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const char* final_parallel_env = std::getenv("SELECTOR_PQ_FUSED_GEOMETRIC_OUTPUT_PARALLEL");
  const bool use_final_parallel =
      final_parallel_env != nullptr && final_parallel_env[0] != '\0' && final_parallel_env[0] != '0';
  torch::Tensor final_output_max_logits;
  torch::Tensor final_exact_thresholds;
  torch::Tensor final_exact_threshold_sels;
  if (use_final_parallel) {
    final_output_max_logits = torch::empty({heads}, opts);
    final_exact_thresholds = torch::empty({heads}, opts);
    final_exact_threshold_sels = torch::empty({heads}, long_opts);
    gqa_decode_geometric_final_head_stats_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
        base_logits.data_ptr<float>(),
        base_counts.data_ptr<int64_t>(),
        filtered_ranked_logits.data_ptr<float>(),
        partial_max.data_ptr<float>(),
        probe_exact_thresholds_ptr,
        probe_exact_threshold_sels_ptr,
        counts.data_ptr<int64_t>(),
        final_output_max_logits.data_ptr<float>(),
        final_exact_thresholds.data_ptr<float>(),
        final_exact_threshold_sels.data_ptr<int64_t>(),
        heads,
        ranked,
        max_base,
        tail_blocks,
        threshold_steps,
        min_budget,
        max_budget,
        granularity,
        static_cast<float>(growth),
        static_cast<float>(probe_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "gqa_decode_geometric_output_final_values",
      [&] {
        using value_scalar_t = scalar_t;
        dim3 parallel_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(dim));
        const int parallel_shared_bytes = threads * 4 * sizeof(float);
        if (value_codes.scalar_type() == torch::kUInt8) {
          if (use_final_parallel) {
            gqa_decode_geometric_final_output_codeweights_parallel_kernel<value_scalar_t, uint8_t>
                <<<parallel_grid, threads, parallel_shared_bytes, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  dense_pq_scores.data_ptr<float>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<uint8_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  filtered_ranked_logits.data_ptr<float>(),
                  partial_max.data_ptr<float>(),
                  final_output_max_logits.data_ptr<float>(),
                  max_logits.data_ptr<float>(),
                  tail_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  final_exact_thresholds.data_ptr<float>(),
                  final_exact_threshold_sels.data_ptr<int64_t>(),
                  counts.data_ptr<int64_t>(),
                  outputs.data_ptr<float>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  tail_blocks,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  static_cast<float>(tail_blend));
          } else {
            gqa_decode_geometric_final_output_codeweights_kernel<value_scalar_t, uint8_t>
                <<<static_cast<int>(heads), threads, 0, stream>>>(
                    values.data_ptr<value_scalar_t>(),
                    dense_pq_scores.data_ptr<float>(),
                    value_codebooks.data_ptr<float>(),
                    value_codes.data_ptr<uint8_t>(),
                    page_starts.data_ptr<int64_t>(),
                    base_tokens.data_ptr<int64_t>(),
                    base_logits.data_ptr<float>(),
                    base_counts.data_ptr<int64_t>(),
                    ranked_tokens.data_ptr<int64_t>(),
                    filtered_ranked_logits.data_ptr<float>(),
                    partial_max.data_ptr<float>(),
                    max_logits.data_ptr<float>(),
                    tail_denoms.data_ptr<float>(),
                    code_weight_sums.data_ptr<float>(),
                    counts.data_ptr<int64_t>(),
                    outputs.data_ptr<float>(),
                    heads,
                    kv_heads,
                    ranked,
                    dim,
                    total_tokens,
                    values.stride(0),
                    values.stride(1),
                    values.stride(2),
                    pages,
                    page_size,
                    max_base,
                    value_subvecs,
                    value_centroids,
                    value_subdim,
                    tail_blocks,
                    group_size,
                    query_context_len,
                    static_prefix,
                    static_suffix,
                    static_cast<float>(exact_value_mass),
                    exact_value_min_top,
                    static_cast<float>(scale),
                    static_cast<float>(tail_blend));
          }
        } else {
          if (use_final_parallel) {
            gqa_decode_geometric_final_output_codeweights_parallel_kernel<value_scalar_t, int64_t>
                <<<parallel_grid, threads, parallel_shared_bytes, stream>>>(
                  values.data_ptr<value_scalar_t>(),
                  dense_pq_scores.data_ptr<float>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<int64_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  filtered_ranked_logits.data_ptr<float>(),
                  partial_max.data_ptr<float>(),
                  final_output_max_logits.data_ptr<float>(),
                  max_logits.data_ptr<float>(),
                  tail_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
                  final_exact_thresholds.data_ptr<float>(),
                  final_exact_threshold_sels.data_ptr<int64_t>(),
                  counts.data_ptr<int64_t>(),
                  outputs.data_ptr<float>(),
                  heads,
                  kv_heads,
                  ranked,
                  dim,
                  total_tokens,
                  values.stride(0),
                  values.stride(1),
                  values.stride(2),
                  pages,
                  page_size,
                  max_base,
                  value_subvecs,
                  value_centroids,
                  value_subdim,
                  tail_blocks,
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  static_cast<float>(tail_blend));
          } else {
            gqa_decode_geometric_final_output_codeweights_kernel<value_scalar_t, int64_t>
                <<<static_cast<int>(heads), threads, 0, stream>>>(
                    values.data_ptr<value_scalar_t>(),
                    dense_pq_scores.data_ptr<float>(),
                    value_codebooks.data_ptr<float>(),
                    value_codes.data_ptr<int64_t>(),
                    page_starts.data_ptr<int64_t>(),
                    base_tokens.data_ptr<int64_t>(),
                    base_logits.data_ptr<float>(),
                    base_counts.data_ptr<int64_t>(),
                    ranked_tokens.data_ptr<int64_t>(),
                    filtered_ranked_logits.data_ptr<float>(),
                    partial_max.data_ptr<float>(),
                    max_logits.data_ptr<float>(),
                    tail_denoms.data_ptr<float>(),
                    code_weight_sums.data_ptr<float>(),
                    counts.data_ptr<int64_t>(),
                    outputs.data_ptr<float>(),
                    heads,
                    kv_heads,
                    ranked,
                    dim,
                    total_tokens,
                    values.stride(0),
                    values.stride(1),
                    values.stride(2),
                    pages,
                    page_size,
                    max_base,
                    value_subvecs,
                    value_centroids,
                    value_subdim,
                    tail_blocks,
                    group_size,
                    query_context_len,
                    static_prefix,
                    static_suffix,
                    static_cast<float>(exact_value_mass),
                    exact_value_min_top,
                    static_cast<float>(scale),
                    static_cast<float>(tail_blend));
          }
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {counts, outputs};
}
