std::vector<torch::Tensor> gqa_decode_geometric_dim_scan_output_vpq_mass_min_proxy_from_logits_thresholds_cuda(
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor base_tokens,
    torch::Tensor base_logits,
    torch::Tensor base_counts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor filtered_ranked_logits,
    torch::Tensor max_logits,
    torch::Tensor tail_denoms,
    torch::Tensor code_weight_sums,
    torch::Tensor counts,
    torch::Tensor outputs,
    const float* approx_exact_thresholds_ptr,
    const int64_t* approx_exact_threshold_sels_ptr,
    const float* probe_exact_thresholds_ptr,
    const int64_t* probe_exact_threshold_sels_ptr,
    int threads,
    int64_t heads,
    int64_t kv_heads,
    int64_t ranked,
    int64_t dim,
    int64_t total_tokens,
    int64_t pages,
    int64_t page_size,
    int64_t max_base,
    int64_t value_subvecs,
    int64_t value_centroids,
    int64_t value_subdim,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
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
    int64_t threshold_steps,
    torch::TensorOptions opts
) {
  auto stream = at::cuda::getCurrentCUDAStream();
    auto diff_squares = torch::empty({heads, threshold_steps, dim}, opts);
    auto probe_squares = torch::empty({heads, threshold_steps, dim}, opts);
    const char* two_pass_output_env = std::getenv("SELECTOR_PQ_GEOMETRIC_TWO_PASS_OUTPUT");
    const bool request_two_pass_output =
        two_pass_output_env != nullptr && two_pass_output_env[0] != '\0' && two_pass_output_env[0] != '0';
    torch::Tensor candidate_outputs;
    torch::Tensor ranked_weights;
    torch::Tensor ranked_pq_weights;
    torch::Tensor ranked_pages;
    torch::Tensor ranked_rows;
    const char* precompute_weights_env = std::getenv("SELECTOR_PQ_PRECOMPUTE_RANK_WEIGHTS");
    const bool precompute_rank_weights =
        precompute_weights_env != nullptr && precompute_weights_env[0] != '\0' && precompute_weights_env[0] != '0';
    if (precompute_rank_weights) {
      ranked_weights = torch::empty({heads, ranked}, opts);
      ranked_pq_weights = torch::empty({heads, ranked}, opts);
      ranked_pages = torch::empty({heads, ranked}, opts.dtype(torch::kInt32));
      ranked_rows = torch::empty({heads, ranked}, opts.dtype(torch::kInt32));
      const int weight_threads = 256;
      const int64_t weight_total = heads * ranked;
      const int weight_blocks = static_cast<int>((weight_total + weight_threads - 1) / weight_threads);
      gqa_decode_rank_weight_metadata_kernel<<<weight_blocks, weight_threads, 0, stream>>>(
          ranked_tokens.data_ptr<int64_t>(),
          filtered_ranked_logits.data_ptr<float>(),
          dense_pq_scores.data_ptr<float>(),
          page_starts.data_ptr<int64_t>(),
          max_logits.data_ptr<float>(),
          ranked_weights.data_ptr<float>(),
          ranked_pq_weights.data_ptr<float>(),
          ranked_pages.data_ptr<int32_t>(),
          ranked_rows.data_ptr<int32_t>(),
          heads,
          ranked,
          pages,
          page_size,
          query_context_len,
          static_prefix,
          static_suffix,
          static_cast<float>(scale));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const float* ranked_weights_ptr = precompute_rank_weights ? ranked_weights.data_ptr<float>() : nullptr;
    const float* ranked_pq_weights_ptr = precompute_rank_weights ? ranked_pq_weights.data_ptr<float>() : nullptr;
    const int32_t* ranked_pages_ptr = precompute_rank_weights ? ranked_pages.data_ptr<int32_t>() : nullptr;
    const int32_t* ranked_rows_ptr = precompute_rank_weights ? ranked_rows.data_ptr<int32_t>() : nullptr;
    torch::Tensor approx_den_incs;
    torch::Tensor probe_den_incs;
    torch::Tensor final_den_incs;
    torch::Tensor approx_code_incs;
    torch::Tensor probe_code_incs;
    torch::Tensor final_code_incs;
    torch::Tensor approx_exact_num_incs;
    torch::Tensor probe_exact_num_incs;
    torch::Tensor final_exact_num_incs;
    torch::Tensor approx_exact_counts;
    torch::Tensor probe_exact_counts;
    torch::Tensor approx_exact_tokens;
    torch::Tensor probe_exact_tokens;
    torch::Tensor approx_exact_weights;
    torch::Tensor probe_exact_weights;
    const char* selected_code_delta_env = std::getenv("SELECTOR_PQ_SELECTED_CODEWEIGHT_DELTAS");
    const bool use_selected_code_deltas =
        precompute_rank_weights && selected_code_delta_env != nullptr && selected_code_delta_env[0] != '\0' &&
        selected_code_delta_env[0] != '0' && value_subvecs > 0 && value_centroids > 0;
    if (use_selected_code_deltas) {
      approx_den_incs = torch::empty({heads, threshold_steps}, opts);
      probe_den_incs = torch::empty({heads, threshold_steps}, opts);
      final_den_incs = torch::empty({heads, threshold_steps}, opts);
      approx_code_incs = torch::zeros({heads, threshold_steps, pages, value_subvecs, value_centroids}, opts);
      probe_code_incs = torch::zeros({heads, threshold_steps, pages, value_subvecs, value_centroids}, opts);
      final_code_incs = torch::zeros({heads, threshold_steps, pages, value_subvecs, value_centroids}, opts);
      dim3 delta_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(threshold_steps));
      const int delta_shared_bytes = threads * 3 * sizeof(float);
      if (value_codes.scalar_type() == torch::kUInt8) {
        gqa_decode_geometric_selected_code_delta_kernel<uint8_t>
            <<<delta_grid, threads, delta_shared_bytes, stream>>>(
                ranked_tokens.data_ptr<int64_t>(),
                filtered_ranked_logits.data_ptr<float>(),
                ranked_weights.data_ptr<float>(),
                ranked_pq_weights.data_ptr<float>(),
                ranked_pages.data_ptr<int32_t>(),
                ranked_rows.data_ptr<int32_t>(),
                value_codes.data_ptr<uint8_t>(),
                approx_exact_thresholds_ptr,
                approx_exact_threshold_sels_ptr,
                probe_exact_thresholds_ptr,
                probe_exact_threshold_sels_ptr,
                approx_den_incs.data_ptr<float>(),
                probe_den_incs.data_ptr<float>(),
                final_den_incs.data_ptr<float>(),
                approx_code_incs.data_ptr<float>(),
                probe_code_incs.data_ptr<float>(),
                final_code_incs.data_ptr<float>(),
                heads,
                kv_heads,
                ranked,
                pages,
                page_size,
                value_subvecs,
                value_centroids,
                group_size,
                min_budget,
                max_budget,
                granularity,
                static_cast<float>(growth),
                static_cast<float>(probe_scale),
                probe_includes_tail,
                threshold_steps);
      } else {
        gqa_decode_geometric_selected_code_delta_kernel<int64_t>
            <<<delta_grid, threads, delta_shared_bytes, stream>>>(
                ranked_tokens.data_ptr<int64_t>(),
                filtered_ranked_logits.data_ptr<float>(),
                ranked_weights.data_ptr<float>(),
                ranked_pq_weights.data_ptr<float>(),
                ranked_pages.data_ptr<int32_t>(),
                ranked_rows.data_ptr<int32_t>(),
                value_codes.data_ptr<int64_t>(),
                approx_exact_thresholds_ptr,
                approx_exact_threshold_sels_ptr,
                probe_exact_thresholds_ptr,
                probe_exact_threshold_sels_ptr,
                approx_den_incs.data_ptr<float>(),
                probe_den_incs.data_ptr<float>(),
                final_den_incs.data_ptr<float>(),
                approx_code_incs.data_ptr<float>(),
                probe_code_incs.data_ptr<float>(),
                final_code_incs.data_ptr<float>(),
                heads,
                kv_heads,
                ranked,
                pages,
                page_size,
                value_subvecs,
                value_centroids,
                group_size,
                min_budget,
                max_budget,
                granularity,
                static_cast<float>(growth),
                static_cast<float>(probe_scale),
                probe_includes_tail,
                threshold_steps);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const char* selected_exact_delta_env = std::getenv("SELECTOR_PQ_SELECTED_EXACT_DELTAS");
    const bool use_selected_exact_deltas =
        use_selected_code_deltas && selected_exact_delta_env != nullptr && selected_exact_delta_env[0] != '\0' &&
        selected_exact_delta_env[0] != '0';
    if (use_selected_exact_deltas) {
      approx_exact_num_incs = torch::empty({heads, threshold_steps, dim}, opts);
      probe_exact_num_incs = torch::empty({heads, threshold_steps, dim}, opts);
      final_exact_num_incs = torch::empty({heads, threshold_steps, dim}, opts);
      dim3 exact_delta_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(threshold_steps));
      const int exact_delta_shared_bytes = static_cast<int>(3 * dim * sizeof(float));
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          values.scalar_type(),
          "gqa_decode_geometric_selected_exact_delta_values",
          [&] {
            gqa_decode_geometric_selected_exact_delta_kernel<scalar_t>
                <<<exact_delta_grid, threads, exact_delta_shared_bytes, stream>>>(
                    values.data_ptr<scalar_t>(),
                    ranked_tokens.data_ptr<int64_t>(),
                    filtered_ranked_logits.data_ptr<float>(),
                    ranked_weights.data_ptr<float>(),
                    ranked_pages.data_ptr<int32_t>(),
                    ranked_rows.data_ptr<int32_t>(),
                    approx_exact_thresholds_ptr,
                    approx_exact_threshold_sels_ptr,
                    probe_exact_thresholds_ptr,
                    probe_exact_threshold_sels_ptr,
                    approx_exact_num_incs.data_ptr<float>(),
                    probe_exact_num_incs.data_ptr<float>(),
                    final_exact_num_incs.data_ptr<float>(),
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
                    group_size,
                    min_budget,
                    max_budget,
                    granularity,
                    static_cast<float>(growth),
                    static_cast<float>(probe_scale),
                    threshold_steps);
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const char* selected_exact_lists_env = std::getenv("SELECTOR_PQ_SELECTED_EXACT_LISTS");
    const bool use_selected_exact_lists =
        use_selected_code_deltas && selected_exact_lists_env != nullptr && selected_exact_lists_env[0] != '\0' &&
        selected_exact_lists_env[0] != '0';
    int64_t exact_list_capacity = max_budget;
    if (use_selected_exact_lists) {
      const char* exact_list_cap_env = std::getenv("SELECTOR_PQ_SELECTED_EXACT_LIST_CAP");
      if (exact_list_cap_env != nullptr && exact_list_cap_env[0] != '\0') {
        char* end = nullptr;
        long parsed = std::strtol(exact_list_cap_env, &end, 10);
        if (end != exact_list_cap_env && parsed > 0) {
          exact_list_capacity = std::min<int64_t>(max_budget, static_cast<int64_t>(parsed));
        }
      }
      exact_list_capacity = std::max<int64_t>(1, std::min<int64_t>(max_budget, exact_list_capacity));
    }
    const bool use_two_pass_output =
        request_two_pass_output && use_selected_code_deltas && use_selected_exact_lists &&
        exact_list_capacity >= std::min<int64_t>(max_budget, ranked);
    if (!use_two_pass_output) {
      candidate_outputs = torch::empty({heads, threshold_steps, dim}, opts);
    }
    if (use_selected_exact_lists) {
      approx_exact_counts = torch::empty({heads, threshold_steps}, opts.dtype(torch::kInt32));
      probe_exact_counts = torch::empty({heads, threshold_steps}, opts.dtype(torch::kInt32));
      approx_exact_tokens =
          torch::empty({heads, threshold_steps, exact_list_capacity}, opts.dtype(torch::kInt32));
      probe_exact_tokens =
          torch::empty({heads, threshold_steps, exact_list_capacity}, opts.dtype(torch::kInt32));
      approx_exact_weights = torch::empty({heads, threshold_steps, exact_list_capacity}, opts);
      probe_exact_weights = torch::empty({heads, threshold_steps, exact_list_capacity}, opts);
      dim3 exact_list_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(threshold_steps));
      gqa_decode_geometric_selected_exact_list_kernel<<<exact_list_grid, threads, 0, stream>>>(
          ranked_tokens.data_ptr<int64_t>(),
          filtered_ranked_logits.data_ptr<float>(),
          ranked_weights.data_ptr<float>(),
          ranked_pages.data_ptr<int32_t>(),
          ranked_rows.data_ptr<int32_t>(),
          approx_exact_thresholds_ptr,
          approx_exact_threshold_sels_ptr,
          probe_exact_thresholds_ptr,
          probe_exact_threshold_sels_ptr,
          approx_exact_counts.data_ptr<int32_t>(),
          probe_exact_counts.data_ptr<int32_t>(),
          approx_exact_tokens.data_ptr<int32_t>(),
          probe_exact_tokens.data_ptr<int32_t>(),
          approx_exact_weights.data_ptr<float>(),
          probe_exact_weights.data_ptr<float>(),
          heads,
          ranked,
          total_tokens,
          pages,
          page_size,
          min_budget,
          max_budget,
          granularity,
          static_cast<float>(growth),
          static_cast<float>(probe_scale),
          exact_list_capacity,
          threshold_steps);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const float* approx_den_incs_ptr = use_selected_code_deltas ? approx_den_incs.data_ptr<float>() : nullptr;
    const float* probe_den_incs_ptr = use_selected_code_deltas ? probe_den_incs.data_ptr<float>() : nullptr;
    const float* final_den_incs_ptr = use_selected_code_deltas ? final_den_incs.data_ptr<float>() : nullptr;
    const float* approx_code_incs_ptr = use_selected_code_deltas ? approx_code_incs.data_ptr<float>() : nullptr;
    const float* probe_code_incs_ptr = use_selected_code_deltas ? probe_code_incs.data_ptr<float>() : nullptr;
    const float* final_code_incs_ptr = use_selected_code_deltas ? final_code_incs.data_ptr<float>() : nullptr;
    const float* approx_exact_num_incs_ptr =
        use_selected_exact_deltas ? approx_exact_num_incs.data_ptr<float>() : nullptr;
    const float* probe_exact_num_incs_ptr =
        use_selected_exact_deltas ? probe_exact_num_incs.data_ptr<float>() : nullptr;
    const float* final_exact_num_incs_ptr =
        use_selected_exact_deltas ? final_exact_num_incs.data_ptr<float>() : nullptr;
    const int32_t* approx_exact_counts_ptr =
        use_selected_exact_lists ? approx_exact_counts.data_ptr<int32_t>() : nullptr;
    const int32_t* probe_exact_counts_ptr =
        use_selected_exact_lists ? probe_exact_counts.data_ptr<int32_t>() : nullptr;
    const int32_t* approx_exact_tokens_ptr =
        use_selected_exact_lists ? approx_exact_tokens.data_ptr<int32_t>() : nullptr;
    const int32_t* probe_exact_tokens_ptr =
        use_selected_exact_lists ? probe_exact_tokens.data_ptr<int32_t>() : nullptr;
    const float* approx_exact_weights_ptr =
        use_selected_exact_lists ? approx_exact_weights.data_ptr<float>() : nullptr;
    const float* probe_exact_weights_ptr =
        use_selected_exact_lists ? probe_exact_weights.data_ptr<float>() : nullptr;
    const int dim_scan_shared_bytes = threads * 6 * sizeof(float);
    dim3 dim_scan_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(dim));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        values.scalar_type(),
        "gqa_decode_geometric_output_dim_scan_values",
        [&] {
          using value_scalar_t = scalar_t;
          if (value_codes.scalar_type() == torch::kUInt8) {
            gqa_decode_geometric_dim_scan_codeweights_kernel<value_scalar_t, uint8_t>
                <<<dim_scan_grid, threads, dim_scan_shared_bytes, stream>>>(
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
                    ranked_weights_ptr,
                    ranked_pq_weights_ptr,
                    ranked_pages_ptr,
                    ranked_rows_ptr,
                    approx_den_incs_ptr,
                    probe_den_incs_ptr,
                    final_den_incs_ptr,
                    approx_code_incs_ptr,
                    probe_code_incs_ptr,
                    final_code_incs_ptr,
                    approx_exact_num_incs_ptr,
                    probe_exact_num_incs_ptr,
                    final_exact_num_incs_ptr,
                    approx_exact_counts_ptr,
                    probe_exact_counts_ptr,
                    approx_exact_tokens_ptr,
                    probe_exact_tokens_ptr,
                    approx_exact_weights_ptr,
                    probe_exact_weights_ptr,
                    max_logits.data_ptr<float>(),
                    tail_denoms.data_ptr<float>(),
                    code_weight_sums.data_ptr<float>(),
                    approx_exact_thresholds_ptr,
                    approx_exact_threshold_sels_ptr,
                    probe_exact_thresholds_ptr,
                    probe_exact_threshold_sels_ptr,
                    diff_squares.data_ptr<float>(),
                    probe_squares.data_ptr<float>(),
                    use_two_pass_output ? nullptr : candidate_outputs.data_ptr<float>(),
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
                    static_cast<float>(scale),
                    probe_includes_tail,
                    !use_two_pass_output,
                    use_selected_code_deltas,
                    use_selected_exact_deltas,
                    use_selected_exact_lists,
                    exact_list_capacity,
                    threshold_steps);
          } else {
            gqa_decode_geometric_dim_scan_codeweights_kernel<value_scalar_t, int64_t>
                <<<dim_scan_grid, threads, dim_scan_shared_bytes, stream>>>(
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
                    ranked_weights_ptr,
                    ranked_pq_weights_ptr,
                    ranked_pages_ptr,
                    ranked_rows_ptr,
                    approx_den_incs_ptr,
                    probe_den_incs_ptr,
                    final_den_incs_ptr,
                    approx_code_incs_ptr,
                    probe_code_incs_ptr,
                    final_code_incs_ptr,
                    approx_exact_num_incs_ptr,
                    probe_exact_num_incs_ptr,
                    final_exact_num_incs_ptr,
                    approx_exact_counts_ptr,
                    probe_exact_counts_ptr,
                    approx_exact_tokens_ptr,
                    probe_exact_tokens_ptr,
                    approx_exact_weights_ptr,
                    probe_exact_weights_ptr,
                    max_logits.data_ptr<float>(),
                    tail_denoms.data_ptr<float>(),
                    code_weight_sums.data_ptr<float>(),
                    approx_exact_thresholds_ptr,
                    approx_exact_threshold_sels_ptr,
                    probe_exact_thresholds_ptr,
                    probe_exact_threshold_sels_ptr,
                    diff_squares.data_ptr<float>(),
                    probe_squares.data_ptr<float>(),
                    use_two_pass_output ? nullptr : candidate_outputs.data_ptr<float>(),
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
                    static_cast<float>(scale),
                    probe_includes_tail,
                    !use_two_pass_output,
                    use_selected_code_deltas,
                    use_selected_exact_deltas,
                    use_selected_exact_lists,
                    exact_list_capacity,
                    threshold_steps);
          }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gqa_decode_geometric_pick_counts_from_step_diffs_kernel<<<
        static_cast<int>(heads),
        threads,
        threads * 2 * sizeof(float),
        stream>>>(
        diff_squares.data_ptr<float>(),
        probe_squares.data_ptr<float>(),
        max_logits.data_ptr<float>(),
        counts.data_ptr<int64_t>(),
        heads,
        threshold_steps,
        dim,
        min_budget,
        max_budget,
        ranked,
        granularity,
        static_cast<float>(growth),
        static_cast<float>(probe_scale),
        static_cast<float>(rel_l2_max));
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
    if (use_two_pass_output) {
      const int output_shared_bytes = threads * 2 * sizeof(float);
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          values.scalar_type(),
          "gqa_decode_geometric_two_pass_output_values",
          [&] {
            gqa_decode_geometric_pick_output_from_deltas_kernel<scalar_t>
                <<<dim_scan_grid, threads, output_shared_bytes, stream>>>(
                    values.data_ptr<scalar_t>(),
                    value_codebooks.data_ptr<float>(),
                    base_tokens.data_ptr<int64_t>(),
                    base_logits.data_ptr<float>(),
                    base_counts.data_ptr<int64_t>(),
                    final_den_incs.data_ptr<float>(),
                    final_code_incs.data_ptr<float>(),
                    probe_exact_counts.data_ptr<int32_t>(),
                    probe_exact_tokens.data_ptr<int32_t>(),
                    probe_exact_weights.data_ptr<float>(),
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
                    max_base,
                    value_subvecs,
                    value_centroids,
                    value_subdim,
                    group_size,
                    min_budget,
                    max_budget,
                    granularity,
                    static_cast<float>(growth),
                    static_cast<float>(probe_scale),
                    exact_list_capacity,
                    threshold_steps);
          });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      const int pick_threads = 256;
      const int64_t pick_total = heads * dim;
      const int pick_blocks = static_cast<int>((pick_total + pick_threads - 1) / pick_threads);
      gqa_decode_geometric_pick_candidate_outputs_kernel<<<pick_blocks, pick_threads, 0, stream>>>(
          candidate_outputs.data_ptr<float>(),
          counts.data_ptr<int64_t>(),
          outputs.data_ptr<float>(),
          heads,
          threshold_steps,
          dim,
          min_budget,
          max_budget,
          ranked,
          granularity,
          static_cast<float>(growth),
          static_cast<float>(probe_scale));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {counts, outputs};
}
