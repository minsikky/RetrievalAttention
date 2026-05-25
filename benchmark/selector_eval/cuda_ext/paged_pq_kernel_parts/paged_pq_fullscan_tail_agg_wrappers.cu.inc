torch::Tensor gqa_decode_vpq_selected_tail_agg_from_scores_cuda(
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
    int64_t query_context_len,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale,
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

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_decode_ranked_logits_mask_keys",
      [&] {
        gqa_decode_ranked_logits_mask_kernel<scalar_t><<<static_cast<int>(heads), threads, 0, stream>>>(
            queries.data_ptr<float>(),
            keys.data_ptr<scalar_t>(),
            page_starts.data_ptr<int64_t>(),
            ranked_tokens.data_ptr<int64_t>(),
            ranked_scores.data_ptr<float>(),
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
      "gqa_decode_tail_agg_output_values",
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

