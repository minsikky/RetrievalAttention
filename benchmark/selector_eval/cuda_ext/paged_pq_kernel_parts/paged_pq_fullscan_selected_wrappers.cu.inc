std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_fused_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  return gqa_causal_fullscan_pq_topk_fused_force_cuda(
      queries,
      codebooks,
      codes,
      page_starts,
      group_size,
      budget,
      query_start,
      static_prefix,
      static_suffix,
      0);
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_top_pages_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t page_budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
  TORCH_CHECK(codebooks.is_cuda(), "codebooks must be CUDA");
  TORCH_CHECK(codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(page_starts.is_cuda(), "page_starts must be CUDA");
  TORCH_CHECK(queries.scalar_type() == torch::kFloat32, "queries must be float32");
  TORCH_CHECK(codebooks.scalar_type() == torch::kFloat32, "codebooks must be float32");
  TORCH_CHECK(queries.dim() == 3, "queries must have shape [positions, heads, dim]");
  TORCH_CHECK(codebooks.dim() == 5, "codebooks must have shape [kv_heads, pages, subvecs, centroids, subdim]");
  TORCH_CHECK(codes.dim() == 4, "codes must have shape [kv_heads, pages, page_size, subvecs]");
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  TORCH_CHECK(codes.size(0) == kv_heads, "codes kv_heads mismatch");
  TORCH_CHECK(codes.size(1) == pages, "codes pages mismatch");
  TORCH_CHECK(codes.size(3) == subvecs, "codes subvecs mismatch");
  TORCH_CHECK(page_starts.numel() == pages, "page_starts pages mismatch");
  TORCH_CHECK(subvecs * subdim == dim, "subvecs * subdim must equal dim");
  TORCH_CHECK(group_size > 0, "group_size must be positive");

  int64_t k = std::max<int64_t>(0, std::min<int64_t>(page_budget, pages));
  auto long_opts = queries.options().dtype(torch::kLong);
  auto float_opts = queries.options().dtype(torch::kFloat32);
  auto top_pages = torch::empty({positions, heads, k}, long_opts);
  auto top_scores = torch::empty({positions, heads, k}, float_opts);
  if (positions == 0 || heads == 0 || k == 0 || pages == 0) {
    return {top_pages, top_scores};
  }

  const int threads = 256;
  const int64_t blocks = positions * heads;
  const size_t shared_bytes =
      static_cast<size_t>(subvecs * centroids) * sizeof(float) +
      static_cast<size_t>(pages) * sizeof(float) +
      static_cast<size_t>(pages) * sizeof(int32_t) +
      static_cast<size_t>(threads) * sizeof(float) +
      static_cast<size_t>(threads) * sizeof(int32_t) * 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (codes.scalar_type() == torch::kUInt8) {
    gqa_causal_pq_top_pages_kernel<uint8_t><<<static_cast<int>(blocks), threads, shared_bytes, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        page_starts.data_ptr<int64_t>(),
        top_pages.data_ptr<int64_t>(),
        top_scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        group_size,
        k,
        query_start,
        static_prefix,
        static_suffix);
  } else {
    TORCH_CHECK(codes.scalar_type() == torch::kLong, "codes must be uint8 or int64");
    gqa_causal_pq_top_pages_kernel<int64_t><<<static_cast<int>(blocks), threads, shared_bytes, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        top_pages.data_ptr<int64_t>(),
        top_scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        group_size,
        k,
        query_start,
        static_prefix,
        static_suffix);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {top_pages, top_scores};
}

torch::Tensor exact_selected_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor tokens,
    double scale) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto total_tokens = keys.size(0);
  const auto selected = tokens.size(1);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({heads, dim}, opts);
  if (heads == 0 || dim == 0) {
    return outputs;
  }
  if (selected == 0 || total_tokens == 0) {
    outputs.zero_();
    return outputs;
  }

  auto logits = torch::empty({heads, selected}, opts);
  const int threads = 128;
  const int64_t pairs = heads * selected;
  auto stream = at::cuda::getCurrentCUDAStream();
  exact_selected_logits_kernel<<<static_cast<int>(pairs), threads, threads * sizeof(float), stream>>>(
      queries.data_ptr<float>(),
      keys.data_ptr<float>(),
      tokens.data_ptr<int64_t>(),
      logits.data_ptr<float>(),
      heads,
      selected,
      dim,
      total_tokens,
      static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  exact_selected_output_kernel<<<static_cast<int>(heads), threads, 0, stream>>>(
      values.data_ptr<float>(),
      tokens.data_ptr<int64_t>(),
      logits.data_ptr<float>(),
      outputs.data_ptr<float>(),
      heads,
      selected,
      dim,
      total_tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_causal_exact_selected_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t page_size,
    double scale) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto selected = ranked_tokens.size(2);
  auto opts = queries.options().dtype(torch::kFloat32);
  auto outputs = torch::empty({positions, heads, dim}, opts);
  if (positions == 0 || heads == 0 || dim == 0) {
    return outputs;
  }
  if (total_tokens == 0 || kv_heads == 0) {
    outputs.zero_();
    return outputs;
  }
  const int64_t max_selected = std::min<int64_t>(
      total_tokens,
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
          std::max<int64_t>(1, page_size) + selected);
  if (max_selected == 0) {
    outputs.zero_();
    return outputs;
  }

  auto long_opts = queries.options().dtype(torch::kLong);
  auto selected_tokens = torch::empty({positions, heads, max_selected}, long_opts);
  auto selected_counts = torch::empty({positions, heads}, long_opts);
  const int build_threads = 128;
  const int64_t qh_pairs = positions * heads;
  const int build_blocks = static_cast<int>((qh_pairs + build_threads - 1) / build_threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gqa_causal_build_selected_tokens_kernel<<<build_blocks, build_threads, 0, stream>>>(
      ranked_tokens.data_ptr<int64_t>(),
      ranked_scores.data_ptr<float>(),
      selected_tokens.data_ptr<int64_t>(),
      selected_counts.data_ptr<int64_t>(),
      positions,
      heads,
      selected,
      max_selected,
      total_tokens,
      query_start,
      static_prefix,
      static_suffix,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int warp_tiled_threads = 256;
  const size_t shared_bytes = static_cast<size_t>(max_selected) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, keys.scalar_type(), "gqa_causal_selected_keys", [&] {
    using key_scalar_t = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "gqa_causal_selected_values", [&] {
      gqa_causal_warp_tiled_selected_attention_kernel<key_scalar_t, scalar_t>
          <<<static_cast<int>(qh_pairs), warp_tiled_threads, shared_bytes, stream>>>(
              queries.data_ptr<float>(),
              keys.data_ptr<key_scalar_t>(),
              values.data_ptr<scalar_t>(),
              selected_tokens.data_ptr<int64_t>(),
              selected_counts.data_ptr<int64_t>(),
              outputs.data_ptr<float>(),
              positions,
              heads,
              kv_heads,
              max_selected,
              dim,
              total_tokens,
              group_size,
              static_cast<float>(scale));
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_causal_vpq_selected_attention_vpagesize_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor value_codebooks,
    torch::Tensor value_codes,
    torch::Tensor page_starts,
    torch::Tensor ranked_tokens,
    torch::Tensor ranked_scores,
    int64_t group_size,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t selection_page_size,
    int64_t value_page_size,
    int64_t exact_value_top,
    double scale) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = keys.size(0);
  const auto total_tokens = keys.size(1);
  const auto selected = ranked_tokens.size(2);
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
  const int64_t max_selected = std::min<int64_t>(
      total_tokens,
      std::max<int64_t>(0, static_prefix) + std::max<int64_t>(0, static_suffix) +
          std::max<int64_t>(1, selection_page_size) + selected);
  if (max_selected == 0) {
    outputs.zero_();
    return outputs;
  }

  auto long_opts = queries.options().dtype(torch::kLong);
  auto selected_tokens = torch::empty({positions, heads, max_selected}, long_opts);
  auto selected_counts = torch::empty({positions, heads}, long_opts);
  const int build_threads = 128;
  const int64_t qh_pairs = positions * heads;
  const int build_blocks = static_cast<int>((qh_pairs + build_threads - 1) / build_threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gqa_causal_build_selected_tokens_kernel<<<build_blocks, build_threads, 0, stream>>>(
      ranked_tokens.data_ptr<int64_t>(),
      ranked_scores.data_ptr<float>(),
      selected_tokens.data_ptr<int64_t>(),
      selected_counts.data_ptr<int64_t>(),
      positions,
      heads,
      selected,
      max_selected,
      total_tokens,
      query_start,
      static_prefix,
      static_suffix,
      selection_page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int warp_tiled_threads = 256;
  const size_t shared_bytes =
      static_cast<size_t>(max_selected) * sizeof(float) +
      (exact_value_top > 0 ? static_cast<size_t>(max_selected) * sizeof(unsigned char) : 0);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, keys.scalar_type(), "gqa_causal_vpq_selected_keys", [&] {
    using key_scalar_t = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "gqa_causal_vpq_selected_values", [&] {
      using value_scalar_t = scalar_t;
      if (value_codes.scalar_type() == torch::kUInt8) {
        gqa_causal_warp_tiled_vpq_selected_attention_kernel<key_scalar_t, value_scalar_t, uint8_t>
            <<<static_cast<int>(qh_pairs), warp_tiled_threads, shared_bytes, stream>>>(
                queries.data_ptr<float>(),
                keys.data_ptr<key_scalar_t>(),
                values.data_ptr<value_scalar_t>(),
                value_codebooks.data_ptr<float>(),
                value_codes.data_ptr<uint8_t>(),
                page_starts.data_ptr<int64_t>(),
                selected_tokens.data_ptr<int64_t>(),
                selected_counts.data_ptr<int64_t>(),
                outputs.data_ptr<float>(),
                positions,
                heads,
                kv_heads,
                max_selected,
                dim,
                total_tokens,
                pages,
                value_page_size,
                value_subvecs,
                value_centroids,
                value_subdim,
                group_size,
                exact_value_top,
                static_cast<float>(scale));
      } else {
        gqa_causal_warp_tiled_vpq_selected_attention_kernel<key_scalar_t, value_scalar_t, int64_t>
            <<<static_cast<int>(qh_pairs), warp_tiled_threads, shared_bytes, stream>>>(
                queries.data_ptr<float>(),
                keys.data_ptr<key_scalar_t>(),
                values.data_ptr<value_scalar_t>(),
                value_codebooks.data_ptr<float>(),
                value_codes.data_ptr<int64_t>(),
                page_starts.data_ptr<int64_t>(),
                selected_tokens.data_ptr<int64_t>(),
                selected_counts.data_ptr<int64_t>(),
                outputs.data_ptr<float>(),
                positions,
                heads,
                kv_heads,
                max_selected,
                dim,
                total_tokens,
                pages,
                value_page_size,
                value_subvecs,
                value_centroids,
                value_subdim,
                group_size,
                exact_value_top,
                static_cast<float>(scale));
      }
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}

torch::Tensor gqa_causal_vpq_selected_attention_cuda(
    torch::Tensor queries,
    torch::Tensor keys,
    torch::Tensor values,
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
    double scale) {
  return gqa_causal_vpq_selected_attention_vpagesize_cuda(
      queries,
      keys,
      values,
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
      page_size,
      0,
      scale);
}

