torch::Tensor gqa_causal_geometric_accept_counts_cuda_impl(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    int64_t exact_value_top,
    double scale) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(2);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto counts = torch::empty({positions, heads}, queries.options().dtype(torch::kLong));
  if (positions == 0 || heads == 0) {
    return counts;
  }
  if (dim == 0 || kv_heads == 0 || total_tokens == 0 || pages == 0 || page_size <= 0) {
    counts.zero_();
    return counts;
  }

  const int threads = 256;
  const int64_t flat_tokens = pages * page_size;
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  const size_t shared_bytes =
      static_cast<size_t>(threads * 2 + max_base + ranked) * sizeof(float) +
      static_cast<size_t>(max_base) * sizeof(int32_t);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t qh_pairs = positions * heads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_causal_geometric_accept_counts_keys",
      [&] {
        using key_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            values.scalar_type(),
            "gqa_causal_geometric_accept_counts_values",
            [&] {
              using value_scalar_t = scalar_t;
              if (value_codes.scalar_type() == torch::kUInt8) {
                cudaFuncSetAttribute(
                    gqa_causal_geometric_accept_counts_kernel<key_scalar_t, value_scalar_t, uint8_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_geometric_accept_counts_kernel<key_scalar_t, value_scalar_t, uint8_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        dense_pq_scores.data_ptr<float>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<uint8_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        counts.data_ptr<int64_t>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        keys.stride(0),
                        keys.stride(1),
                        keys.stride(2),
                        values.stride(0),
                        values.stride(1),
                        values.stride(2),
                        pages,
                        page_size,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        max_base,
                        min_budget,
                        max_budget,
                        granularity,
                        static_cast<float>(growth),
                        static_cast<float>(probe_scale),
                        static_cast<float>(rel_l2_max),
                        exact_value_top,
                        static_cast<float>(scale));
              } else {
                cudaFuncSetAttribute(
                    gqa_causal_geometric_accept_counts_kernel<key_scalar_t, value_scalar_t, int64_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_geometric_accept_counts_kernel<key_scalar_t, value_scalar_t, int64_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        dense_pq_scores.data_ptr<float>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<int64_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        counts.data_ptr<int64_t>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        keys.stride(0),
                        keys.stride(1),
                        keys.stride(2),
                        values.stride(0),
                        values.stride(1),
                        values.stride(2),
                        pages,
                        page_size,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        max_base,
                        min_budget,
                        max_budget,
                        granularity,
                        static_cast<float>(growth),
                        static_cast<float>(probe_scale),
                        static_cast<float>(rel_l2_max),
                        exact_value_top,
                        static_cast<float>(scale));
              }
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return counts;
}

torch::Tensor gqa_causal_geometric_accept_counts_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    double scale) {
  return gqa_causal_geometric_accept_counts_cuda_impl(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      group_size,
      query_start,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
      max_budget,
      scale);
}

torch::Tensor gqa_causal_geometric_accept_counts_vpq_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    int64_t min_budget,
    int64_t max_budget,
    int64_t granularity,
    double growth,
    double probe_scale,
    double rel_l2_max,
    int64_t exact_value_top,
    double scale) {
  return gqa_causal_geometric_accept_counts_cuda_impl(
      queries,
      keys,
      values,
      dense_pq_scores,
      value_codebooks,
      value_codes,
      page_starts,
      ranked_tokens,
      ranked_scores,
      group_size,
      query_start,
      static_prefix,
      static_suffix,
      page_size,
      min_budget,
      max_budget,
      granularity,
      growth,
      probe_scale,
      rel_l2_max,
      exact_value_top,
      scale);
}

torch::Tensor gqa_decode_vpq_selected_from_logits_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits_input,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(1);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({heads, dim}, opts);
  if (heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }
  if (pages == 0 || page_size <= 0) {
    outputs.zero_();
    return outputs;
  }

  const int threads = 256;
  const int64_t flat_tokens = pages * page_size;
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  auto byte_opts = queries.options().dtype(torch::kUInt8);
  auto tail_mask = torch::zeros({heads, flat_tokens}, byte_opts);
  auto long_opts = queries.options().dtype(torch::kLong);
  auto base_tokens = torch::empty({heads, max_base}, long_opts);
  auto base_logits = torch::empty({heads, max_base}, opts);
  auto base_counts = torch::zeros({heads}, long_opts);
  auto ranked_logits = torch::empty({heads, ranked}, opts);
  auto ranked_exact = torch::zeros({heads, ranked}, byte_opts);
  auto partial_max = torch::full({heads, 1}, -std::numeric_limits<float>::infinity(), opts);
  auto max_logits = torch::empty({heads}, opts);
  auto partial_sum = torch::zeros({heads, 1}, opts);
  auto denoms = torch::empty({heads}, opts);
  auto selected_denoms = torch::empty({heads}, opts);
  auto code_weight_sums = torch::empty({1}, opts);
  const int64_t* exact_value_counts_ptr =
      exact_value_counts.numel() > 0 ? exact_value_counts.data_ptr<int64_t>() : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream();
	  AT_DISPATCH_FLOATING_TYPES_AND2(
	      at::ScalarType::Half,
	      at::ScalarType::BFloat16,
	      keys.scalar_type(),
	      "gqa_decode_ranked_logits_mask_from_logits_keys",
      [&] {
        gqa_decode_ranked_logits_mask_from_logits_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
            queries.data_ptr<float>(),
            keys.data_ptr<scalar_t>(),
            page_starts.data_ptr<int64_t>(),
            ranked_tokens.data_ptr<int64_t>(),
            ranked_scores.data_ptr<float>(),
            ranked_logits_input.data_ptr<float>(),
            tail_mask.data_ptr<unsigned char>(),
            base_tokens.data_ptr<int64_t>(),
            base_logits.data_ptr<float>(),
            base_counts.data_ptr<int64_t>(),
            ranked_logits.data_ptr<float>(),
            ranked_exact.data_ptr<unsigned char>(),
            heads,
            kv_heads,
            ranked,
            dim,
            total_tokens,
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            pages,
            page_size,
            max_base,
            group_size,
            query_context_len,
            static_prefix,
            static_suffix,
            exact_value_counts_ptr,
            nullptr,
            nullptr,
            exact_value_top,
            static_cast<float>(exact_value_mass),
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_final_max_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_max.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gqa_decode_final_denom_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_sum.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      denoms.data_ptr<float>(),
      selected_denoms.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "gqa_decode_selected_from_logits_output_values",
      [&] {
        using value_scalar_t = scalar_t;
        if (value_codes.scalar_type() == torch::kUInt8) {
          launch_gqa_decode_tail_agg_output<value_scalar_t, uint8_t>(
                  values.data_ptr<value_scalar_t>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<uint8_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  ranked_exact.data_ptr<unsigned char>(),
                  max_logits.data_ptr<float>(),
                  denoms.data_ptr<float>(),
                  selected_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
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
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  0.0f,
                  stream);
        } else {
          launch_gqa_decode_tail_agg_output<value_scalar_t, int64_t>(
                  values.data_ptr<value_scalar_t>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<int64_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  ranked_exact.data_ptr<unsigned char>(),
                  max_logits.data_ptr<float>(),
                  denoms.data_ptr<float>(),
                  selected_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
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
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  0.0f,
                  stream);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_decode_vpq_selected_tail_agg_from_logits_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    torch::Tensor ranked_logits_input,
    int64_t group_size,
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale,
    double tail_blend,
    torch::Tensor exact_value_thresholds = torch::Tensor(),
    torch::Tensor exact_value_threshold_sels = torch::Tensor()) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(1);
  const auto pages = value_codebooks.size(1);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({heads, dim}, opts);
  if (heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }
  if (pages == 0 || page_size <= 0) {
    outputs.zero_();
    return outputs;
  }

  const int threads = 256;
  const bool include_tail = tail_blend > 0.0;
  const int64_t flat_tokens = pages * page_size;
  const int64_t pages_per_block = decode_tail_pages_per_block();
  const int64_t tail_blocks = std::max<int64_t>(1, (pages + pages_per_block - 1) / pages_per_block);
  const int64_t max_base =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + 4;
  auto byte_opts = queries.options().dtype(torch::kUInt8);
  auto tail_mask = torch::zeros({heads, flat_tokens}, byte_opts);
  auto long_opts = queries.options().dtype(torch::kLong);
  auto base_tokens = torch::empty({heads, max_base}, long_opts);
  auto base_logits = torch::empty({heads, max_base}, opts);
  auto base_counts = torch::zeros({heads}, long_opts);
  auto ranked_logits = torch::empty({heads, ranked}, opts);
  auto ranked_exact = torch::zeros({heads, ranked}, byte_opts);
  auto partial_max = torch::empty({heads, tail_blocks}, opts);
  auto max_logits = torch::empty({heads}, opts);
  auto partial_sum = torch::empty({heads, tail_blocks}, opts);
  auto denoms = torch::empty({heads}, opts);
  auto selected_denoms = torch::empty({heads}, opts);
  auto code_weight_sums = torch::zeros({heads, pages, value_subvecs, value_centroids}, opts);
  const int64_t* exact_value_counts_ptr =
      exact_value_counts.numel() > 0 ? exact_value_counts.data_ptr<int64_t>() : nullptr;
  const bool has_exact_value_thresholds =
      exact_value_thresholds.defined() && exact_value_threshold_sels.defined() &&
      exact_value_thresholds.numel() > 0 && exact_value_threshold_sels.numel() > 0;
  const float* exact_value_thresholds_ptr =
      has_exact_value_thresholds ? exact_value_thresholds.data_ptr<float>() : nullptr;
  const int64_t* exact_value_threshold_sels_ptr =
      has_exact_value_thresholds ? exact_value_threshold_sels.data_ptr<int64_t>() : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_decode_ranked_logits_mask_from_logits_keys",
      [&] {
        gqa_decode_ranked_logits_mask_from_logits_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
            queries.data_ptr<float>(),
            keys.data_ptr<scalar_t>(),
            page_starts.data_ptr<int64_t>(),
            ranked_tokens.data_ptr<int64_t>(),
            ranked_scores.data_ptr<float>(),
            ranked_logits_input.data_ptr<float>(),
            tail_mask.data_ptr<unsigned char>(),
            base_tokens.data_ptr<int64_t>(),
            base_logits.data_ptr<float>(),
            base_counts.data_ptr<int64_t>(),
            ranked_logits.data_ptr<float>(),
            ranked_exact.data_ptr<unsigned char>(),
            heads,
            kv_heads,
            ranked,
            dim,
            total_tokens,
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            pages,
            page_size,
            max_base,
            group_size,
            query_context_len,
            static_prefix,
            static_suffix,
            exact_value_counts_ptr,
            exact_value_thresholds_ptr,
            exact_value_threshold_sels_ptr,
            exact_value_top,
            static_cast<float>(exact_value_mass),
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 tail_grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(tail_blocks));
  if (include_tail) {
    gqa_decode_tail_partial_max_kernel<<<tail_grid, threads, threads * sizeof(float), stream>>>(
        dense_pq_scores.data_ptr<float>(),
        page_starts.data_ptr<int64_t>(),
        tail_mask.data_ptr<unsigned char>(),
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
  } else {
    partial_max.fill_(-std::numeric_limits<float>::infinity());
    partial_sum.zero_();
  }

  gqa_decode_final_max_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_max.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (include_tail) {
    if (value_codes.scalar_type() == torch::kUInt8) {
      gqa_decode_tail_sum_codeweights_kernel<uint8_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
          dense_pq_scores.data_ptr<float>(),
          max_logits.data_ptr<float>(),
          value_codebooks.data_ptr<float>(),
          value_codes.data_ptr<uint8_t>(),
          page_starts.data_ptr<int64_t>(),
          tail_mask.data_ptr<unsigned char>(),
          partial_sum.data_ptr<float>(),
          code_weight_sums.data_ptr<float>(),
          heads,
          kv_heads,
          pages,
          page_size,
          value_subvecs,
          value_centroids,
          value_subdim,
          tail_blocks,
          pages_per_block,
          group_size,
          query_context_len,
          static_prefix,
          static_suffix,
          static_cast<float>(scale));
    } else {
      gqa_decode_tail_sum_codeweights_kernel<int64_t><<<tail_grid, threads, threads * sizeof(float), stream>>>(
          dense_pq_scores.data_ptr<float>(),
          max_logits.data_ptr<float>(),
          value_codebooks.data_ptr<float>(),
          value_codes.data_ptr<int64_t>(),
          page_starts.data_ptr<int64_t>(),
          tail_mask.data_ptr<unsigned char>(),
          partial_sum.data_ptr<float>(),
          code_weight_sums.data_ptr<float>(),
          heads,
          kv_heads,
          pages,
          page_size,
          value_subvecs,
          value_centroids,
          value_subdim,
          tail_blocks,
          pages_per_block,
          group_size,
          query_context_len,
          static_prefix,
          static_suffix,
          static_cast<float>(scale));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  gqa_decode_final_denom_kernel<<<static_cast<int>(heads), threads, threads * sizeof(float), stream>>>(
      partial_sum.data_ptr<float>(),
      base_logits.data_ptr<float>(),
      base_counts.data_ptr<int64_t>(),
      ranked_logits.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      denoms.data_ptr<float>(),
      selected_denoms.data_ptr<float>(),
      heads,
      ranked,
      max_base,
      tail_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "gqa_decode_tail_agg_from_logits_output_values",
      [&] {
        using value_scalar_t = scalar_t;
        if (value_codes.scalar_type() == torch::kUInt8) {
          launch_gqa_decode_tail_agg_output<value_scalar_t, uint8_t>(
                  values.data_ptr<value_scalar_t>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<uint8_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  ranked_exact.data_ptr<unsigned char>(),
                  max_logits.data_ptr<float>(),
                  denoms.data_ptr<float>(),
                  selected_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
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
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  static_cast<float>(tail_blend),
                  stream);
        } else {
          launch_gqa_decode_tail_agg_output<value_scalar_t, int64_t>(
                  values.data_ptr<value_scalar_t>(),
                  value_codebooks.data_ptr<float>(),
                  value_codes.data_ptr<int64_t>(),
                  page_starts.data_ptr<int64_t>(),
                  base_tokens.data_ptr<int64_t>(),
                  base_logits.data_ptr<float>(),
                  base_counts.data_ptr<int64_t>(),
                  ranked_tokens.data_ptr<int64_t>(),
                  ranked_logits.data_ptr<float>(),
                  ranked_exact.data_ptr<unsigned char>(),
                  max_logits.data_ptr<float>(),
                  denoms.data_ptr<float>(),
                  selected_denoms.data_ptr<float>(),
                  code_weight_sums.data_ptr<float>(),
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
                  group_size,
                  query_context_len,
                  static_prefix,
                  static_suffix,
                  static_cast<float>(scale),
                  static_cast<float>(tail_blend),
                  stream);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_exact_selected_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor tokens,
    int64_t group_size,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto selected = tokens.size(1);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({heads, dim}, opts);
  if (heads == 0 || dim == 0) {
    return outputs;
  }
  if (selected == 0 || total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }

  auto logits = torch::empty({heads, selected}, opts);
  const int threads = 128;
  const int64_t pairs = heads * selected;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, keys.scalar_type(), "gqa_exact_selected_keys", [&] {
    gqa_exact_selected_logits_kernel<scalar_t><<<static_cast<int>(pairs), threads, threads * sizeof(float), stream>>>(
        queries.data_ptr<float>(),
        keys.data_ptr<scalar_t>(),
        tokens.data_ptr<int64_t>(),
        logits.data_ptr<float>(),
        heads,
        kv_heads,
        selected,
        dim,
        total_tokens,
        group_size,
        static_cast<float>(scale));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "gqa_exact_selected_values", [&] {
    gqa_exact_selected_output_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
        values.data_ptr<scalar_t>(),
        tokens.data_ptr<int64_t>(),
        logits.data_ptr<float>(),
        outputs.data_ptr<float>(),
        heads,
        kv_heads,
        selected,
        dim,
        total_tokens,
        group_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

std::vector<torch::Tensor> selected_mass_thresholds_from_topk_cuda(
    torch::Tensor top_logits,
    torch::Tensor top_order,
    torch::Tensor prefix_lse,
    torch::Tensor prefix_valid_counts,
    torch::Tensor base_lse,
    torch::Tensor budgets,
    double exact_mass,
    int64_t min_top) {
  const auto heads = top_logits.size(0);
  const auto k_top = top_logits.size(1);
  const auto rank = prefix_lse.size(1);
  const auto steps = budgets.size(0);
  auto float_opts = top_logits.options().dtype(torch::kFloat32);
  auto long_opts = top_order.options().dtype(torch::kLong);
  auto byte_opts = top_order.options().dtype(torch::kUInt8);
  auto thresholds = torch::empty({heads, steps}, float_opts);
  auto threshold_sels = torch::empty({heads, steps}, long_opts);
  auto sufficient = torch::empty({heads, steps}, byte_opts);
  if (heads == 0 || steps == 0) {
    return {thresholds, threshold_sels, sufficient};
  }
  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned int>(heads), static_cast<unsigned int>(steps));
  selected_mass_thresholds_from_topk_kernel<<<grid, threads, 0, stream>>>(
      top_logits.data_ptr<float>(),
      top_order.data_ptr<int64_t>(),
      prefix_lse.data_ptr<float>(),
      prefix_valid_counts.data_ptr<int64_t>(),
      base_lse.data_ptr<float>(),
      budgets.data_ptr<int64_t>(),
      thresholds.data_ptr<float>(),
      threshold_sels.data_ptr<int64_t>(),
      sufficient.data_ptr<unsigned char>(),
      heads,
      k_top,
      rank,
      steps,
      static_cast<float>(exact_mass),
      min_top);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {thresholds, threshold_sels, sufficient};
}

