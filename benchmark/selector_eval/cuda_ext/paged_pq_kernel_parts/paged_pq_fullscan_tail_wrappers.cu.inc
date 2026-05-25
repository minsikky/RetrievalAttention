torch::Tensor gqa_causal_vpq_tail_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor key_codebooks,
    torch::Tensor key_codes,
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
    int64_t exact_value_top,
    double scale,
    double tail_blend) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto ranked = ranked_tokens.size(2);
  const auto pages = key_codebooks.size(1);
  const auto key_subvecs = key_codebooks.size(2);
  const auto key_centroids = key_codebooks.size(3);
  const auto key_subdim = key_codebooks.size(4);
  const auto value_subvecs = value_codebooks.size(2);
  const auto value_centroids = value_codebooks.size(3);
  const auto value_subdim = value_codebooks.size(4);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({positions, heads, dim}, opts);
  if (positions == 0 || heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }

  const int threads = 128;
  const int64_t qh_pairs = positions * heads;
  const int64_t max_selected =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + ranked + 4;
  const int64_t code_weight_count = pages * value_subvecs * value_centroids;
  const int64_t total_rows = pages * page_size;
  const int64_t ranked_bitmap_words = (total_rows + 31) / 32;
  const size_t shared_bytes =
      static_cast<size_t>(max_selected + code_weight_count + threads) * sizeof(float) +
      static_cast<size_t>(max_selected) * (sizeof(int32_t) + sizeof(unsigned char)) +
      static_cast<size_t>(ranked_bitmap_words) * sizeof(unsigned int) +
      sizeof(unsigned int);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_causal_vpq_tail_keys",
      [&] {
        using key_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            values.scalar_type(),
            "gqa_causal_vpq_tail_values",
            [&] {
              using value_scalar_t = scalar_t;
              if (key_codes.scalar_type() == torch::kUInt8 && value_codes.scalar_type() == torch::kUInt8) {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, uint8_t, uint8_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, uint8_t, uint8_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        key_codebooks.data_ptr<float>(),
                        key_codes.data_ptr<uint8_t>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<uint8_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
	                        positions,
	                        heads,
	                        kv_heads,
	                        ranked,
	                        dim,
	                        total_tokens,
	                        pages,
	                        page_size,
                        key_subvecs,
                        key_centroids,
                        key_subdim,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_top,
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              } else if (key_codes.scalar_type() == torch::kUInt8) {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, uint8_t, int64_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, uint8_t, int64_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        key_codebooks.data_ptr<float>(),
                        key_codes.data_ptr<uint8_t>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<int64_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        pages,
                        page_size,
                        key_subvecs,
                        key_centroids,
                        key_subdim,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_top,
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              } else if (value_codes.scalar_type() == torch::kUInt8) {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, int64_t, uint8_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, int64_t, uint8_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        key_codebooks.data_ptr<float>(),
                        key_codes.data_ptr<int64_t>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<uint8_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        pages,
                        page_size,
                        key_subvecs,
                        key_centroids,
                        key_subdim,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_top,
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              } else {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, int64_t, int64_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_attention_kernel<key_scalar_t, value_scalar_t, int64_t, int64_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        key_codebooks.data_ptr<float>(),
                        key_codes.data_ptr<int64_t>(),
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<int64_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        pages,
                        page_size,
                        key_subvecs,
                        key_centroids,
                        key_subdim,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_top,
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              }
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_causal_vpq_tail_from_scores_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor dense_pq_scores,
    torch::Tensor dense_selected_scores,
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
    torch::Tensor exact_value_counts,
    int64_t exact_value_top,
    double exact_value_mass,
    double scale,
    double tail_blend) {
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
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({positions, heads, dim}, opts);
  if (positions == 0 || heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }

  const int threads = 128;
  const int64_t qh_pairs = positions * heads;
  const int64_t max_selected =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + ranked + 4;
  const int64_t code_weight_count = pages * value_subvecs * value_centroids;
  const int64_t ranked_bitmap_words = ((pages * page_size) + 31) / 32;
  const bool has_dense_selected_scores = dense_selected_scores.numel() > 0;
  if (has_dense_selected_scores) {
    TORCH_CHECK(dense_selected_scores.is_cuda(), "dense_selected_scores must be CUDA");
    TORCH_CHECK(dense_selected_scores.scalar_type() == torch::kFloat32, "dense_selected_scores must be float32");
    TORCH_CHECK(dense_selected_scores.sizes() == dense_pq_scores.sizes(), "dense_selected_scores shape mismatch");
  }
  const int64_t* exact_value_counts_ptr =
      exact_value_counts.numel() > 0 ? exact_value_counts.data_ptr<int64_t>() : nullptr;
  const float* dense_selected_ptr =
      has_dense_selected_scores ? dense_selected_scores.data_ptr<float>() : nullptr;
  const size_t shared_bytes =
      static_cast<size_t>(max_selected + code_weight_count + threads) * sizeof(float) +
      static_cast<size_t>(max_selected) * (sizeof(int32_t) + sizeof(unsigned char)) +
      static_cast<size_t>(ranked_bitmap_words) * sizeof(unsigned int) +
      alignof(unsigned int);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      keys.scalar_type(),
      "gqa_causal_vpq_tail_from_scores_keys",
      [&] {
        using key_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            values.scalar_type(),
            "gqa_causal_vpq_tail_from_scores_values",
            [&] {
              using value_scalar_t = scalar_t;
              if (value_codes.scalar_type() == torch::kUInt8) {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_from_scores_kernel<key_scalar_t, value_scalar_t, uint8_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_from_scores_kernel<key_scalar_t, value_scalar_t, uint8_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        dense_pq_scores.data_ptr<float>(),
                        dense_selected_ptr,
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<uint8_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        pages,
                        page_size,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_counts_ptr,
                        exact_value_top,
                        static_cast<float>(exact_value_mass),
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              } else {
                cudaFuncSetAttribute(
                    gqa_causal_fused_vpq_tail_from_scores_kernel<key_scalar_t, value_scalar_t, int64_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes));
                gqa_causal_fused_vpq_tail_from_scores_kernel<key_scalar_t, value_scalar_t, int64_t>
                    <<<static_cast<int>(qh_pairs), threads, shared_bytes, stream>>>(
                        queries.data_ptr<float>(),
                        keys.data_ptr<key_scalar_t>(),
                        values.data_ptr<value_scalar_t>(),
                        dense_pq_scores.data_ptr<float>(),
                        dense_selected_ptr,
                        value_codebooks.data_ptr<float>(),
                        value_codes.data_ptr<int64_t>(),
                        page_starts.data_ptr<int64_t>(),
                        ranked_tokens.data_ptr<int64_t>(),
                        ranked_scores.data_ptr<float>(),
                        outputs.data_ptr<float>(),
                        positions,
                        heads,
                        kv_heads,
                        ranked,
                        dim,
                        total_tokens,
                        pages,
                        page_size,
                        value_subvecs,
                        value_centroids,
                        value_subdim,
                        group_size,
                        query_start,
                        static_prefix,
                        static_suffix,
                        exact_value_counts_ptr,
                        exact_value_top,
                        static_cast<float>(exact_value_mass),
                        static_cast<float>(scale),
                        static_cast<float>(tail_blend));
              }
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_decode_vpq_tail_from_scores_cuda(
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
  const int threads = 128;
  const int64_t max_selected =
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
      std::max<int64_t>(1, page_size) + ranked + 4;
  const bool needs_mixed_exact_buffers =
      exact_value_counts.numel() > 0 || (exact_value_top > 0 && exact_value_top < ranked);
  const size_t mixed_bytes =
      needs_mixed_exact_buffers
          ? (static_cast<size_t>(ranked) * sizeof(float) + static_cast<size_t>(ranked) * sizeof(unsigned char))
          : 0;
  const size_t shared_bytes = static_cast<size_t>(max_selected) * sizeof(float) + mixed_bytes;
  const int64_t* exact_value_counts_ptr =
      exact_value_counts.numel() > 0 ? exact_value_counts.data_ptr<int64_t>() : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (value_codes.scalar_type() == torch::kUInt8) {
    gqa_decode_vpq_tail_from_scores_kernel<uint8_t><<<static_cast<int>(heads), threads, shared_bytes, stream>>>(
        queries.data_ptr<float>(),
        keys.data_ptr<float>(),
        values.data_ptr<float>(),
        dense_pq_scores.data_ptr<float>(),
        value_codebooks.data_ptr<float>(),
        value_codes.data_ptr<uint8_t>(),
        page_starts.data_ptr<int64_t>(),
        ranked_tokens.data_ptr<int64_t>(),
        ranked_scores.data_ptr<float>(),
        outputs.data_ptr<float>(),
        heads,
        kv_heads,
        ranked,
        dim,
        total_tokens,
        pages,
        page_size,
        value_subvecs,
        value_centroids,
        value_subdim,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        exact_value_counts_ptr,
        exact_value_top,
        static_cast<float>(scale),
        static_cast<float>(tail_blend));
  } else {
    gqa_decode_vpq_tail_from_scores_kernel<int64_t><<<static_cast<int>(heads), threads, shared_bytes, stream>>>(
        queries.data_ptr<float>(),
        keys.data_ptr<float>(),
        values.data_ptr<float>(),
        dense_pq_scores.data_ptr<float>(),
        value_codebooks.data_ptr<float>(),
        value_codes.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        ranked_tokens.data_ptr<int64_t>(),
        ranked_scores.data_ptr<float>(),
        outputs.data_ptr<float>(),
        heads,
        kv_heads,
        ranked,
        dim,
        total_tokens,
        pages,
        page_size,
        value_subvecs,
        value_centroids,
        value_subdim,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        exact_value_counts_ptr,
        exact_value_top,
        static_cast<float>(scale),
        static_cast<float>(tail_blend));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

