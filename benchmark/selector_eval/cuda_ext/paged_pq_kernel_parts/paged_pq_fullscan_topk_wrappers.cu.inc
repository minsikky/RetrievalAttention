std::vector<torch::Tensor> fullscan_pq_topk_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t budget) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto pages = codebooks.size(0);
  const auto subvecs = codebooks.size(1);
  const auto centroids = codebooks.size(2);
  const auto subdim = codebooks.size(3);
  const auto page_size = codes.size(1);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({heads, k}, long_opts);
  auto top_scores = torch::empty({heads, k}, score_opts);
  if (heads == 0 || pages == 0 || k == 0) {
    return {top_tokens, top_scores};
  }

  auto scores = torch::empty({heads, total_tokens}, score_opts);
  const int threads = 128;
  const int64_t score_elems = heads * total_tokens;
  const int blocks = static_cast<int>((score_elems + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (codes.scalar_type() == torch::kUInt8) {
    pq_scores_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        scores.data_ptr<float>(),
        heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens);
  } else {
    pq_scores_kernel<int64_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  top_scores = std::get<0>(topk);
  auto top_indices = std::get<1>(topk).contiguous();

  const int64_t map_elems = heads * k;
  const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
  map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
      top_indices.data_ptr<int64_t>(),
      page_starts.data_ptr<int64_t>(),
      top_tokens.data_ptr<int64_t>(),
      heads,
      k,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {top_tokens, top_scores};
}

std::vector<torch::Tensor> fullscan_pq_topk_scores_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t budget) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto pages = codebooks.size(0);
  const auto subvecs = codebooks.size(1);
  const auto centroids = codebooks.size(2);
  const auto subdim = codebooks.size(3);
  const auto page_size = codes.size(1);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({heads, k}, long_opts);
  auto top_scores = torch::empty({heads, k}, score_opts);
  auto scores = torch::empty({heads, total_tokens}, score_opts);
  if (heads == 0 || pages == 0) {
    return {top_tokens, top_scores, scores};
  }

  const int threads = 256;
  const int64_t score_elems = heads * total_tokens;
  const int blocks = static_cast<int>((score_elems + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (codes.scalar_type() == torch::kUInt8) {
    pq_scores_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        scores.data_ptr<float>(),
        heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens);
  } else {
    pq_scores_kernel<int64_t><<<blocks, threads, 0, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (k == 0) {
    return {top_tokens, top_scores, scores};
  }

  auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  top_scores = std::get<0>(topk);
  auto top_indices = std::get<1>(topk).contiguous();

  const int64_t map_elems = heads * k;
  const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
  map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
      top_indices.data_ptr<int64_t>(),
      page_starts.data_ptr<int64_t>(),
      top_tokens.data_ptr<int64_t>(),
      heads,
      k,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {top_tokens, top_scores, scores};
}

std::vector<torch::Tensor> gqa_fullscan_pq_topk_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({heads, k}, long_opts);
  auto top_scores = torch::empty({heads, k}, score_opts);
  if (heads == 0 || pages == 0 || k == 0) {
    return {top_tokens, top_scores};
  }

  auto scores = torch::empty({heads, total_tokens}, score_opts);
  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool use_lut_scores = env_enabled_default("SELECTOR_PQ_FULLSCAN_LUT_SCORES", false);

  if (use_lut_scores) {
    const dim3 score_blocks(static_cast<unsigned int>(heads), static_cast<unsigned int>(pages), 1);
    const size_t lut_smem = static_cast<size_t>(subvecs * centroids) * sizeof(float);
    if (codes.scalar_type() == torch::kUInt8) {
      gqa_pq_scores_lut_kernel<uint8_t><<<score_blocks, threads, lut_smem, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<uint8_t>(),
          scores.data_ptr<float>(),
          heads,
          kv_heads,
          dim,
          pages,
          page_size,
          subvecs,
          centroids,
          subdim,
          total_tokens,
          group_size);
    } else {
      gqa_pq_scores_lut_kernel<int64_t><<<score_blocks, threads, lut_smem, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<int64_t>(),
          scores.data_ptr<float>(),
          heads,
          kv_heads,
          dim,
          pages,
          page_size,
          subvecs,
          centroids,
          subdim,
          total_tokens,
          group_size);
    }
  } else {
    const int64_t score_elems = heads * total_tokens;
    const int blocks = static_cast<int>((score_elems + threads - 1) / threads);
    if (codes.scalar_type() == torch::kUInt8) {
      gqa_pq_scores_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<uint8_t>(),
          scores.data_ptr<float>(),
          heads,
          kv_heads,
          dim,
          pages,
          page_size,
          subvecs,
          centroids,
          subdim,
          total_tokens,
          group_size);
    } else {
      gqa_pq_scores_kernel<int64_t><<<blocks, threads, 0, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<int64_t>(),
          scores.data_ptr<float>(),
          heads,
          kv_heads,
          dim,
          pages,
          page_size,
          subvecs,
          centroids,
          subdim,
          total_tokens,
          group_size);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  top_scores = std::get<0>(topk);
  auto top_indices = std::get<1>(topk).contiguous();

  const int64_t map_elems = heads * k;
  const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
  map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
      top_indices.data_ptr<int64_t>(),
      page_starts.data_ptr<int64_t>(),
      top_tokens.data_ptr<int64_t>(),
      heads,
      k,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {top_tokens, top_scores};
}

std::vector<torch::Tensor> gqa_fullscan_pq_topk_scores_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget) {
  const auto heads = queries.size(0);
  const auto dim = queries.size(1);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({heads, k}, long_opts);
  auto top_scores = torch::empty({heads, k}, score_opts);
  auto scores = torch::empty({heads, total_tokens}, score_opts);
  if (heads == 0 || pages == 0) {
    return {top_tokens, top_scores, scores};
  }

  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool use_lut_scores = env_enabled_default("SELECTOR_PQ_FULLSCAN_LUT_SCORES", false);

  if (use_lut_scores) {
    const dim3 score_blocks(static_cast<unsigned int>(heads), static_cast<unsigned int>(pages), 1);
    const size_t lut_smem = static_cast<size_t>(subvecs * centroids) * sizeof(float);
    if (codes.scalar_type() == torch::kUInt8) {
      gqa_pq_scores_lut_kernel<uint8_t><<<score_blocks, threads, lut_smem, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<uint8_t>(),
          scores.data_ptr<float>(),
          heads,
          kv_heads,
          dim,
          pages,
          page_size,
          subvecs,
          centroids,
          subdim,
          total_tokens,
          group_size);
    } else {
      gqa_pq_scores_lut_kernel<int64_t><<<score_blocks, threads, lut_smem, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<int64_t>(),
          scores.data_ptr<float>(),
          heads,
          kv_heads,
          dim,
          pages,
          page_size,
          subvecs,
          centroids,
          subdim,
          total_tokens,
          group_size);
    }
  } else {
    const int64_t score_elems = heads * total_tokens;
    const int blocks = static_cast<int>((score_elems + threads - 1) / threads);
    if (codes.scalar_type() == torch::kUInt8) {
      gqa_pq_scores_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<uint8_t>(),
          scores.data_ptr<float>(),
          heads,
          kv_heads,
          dim,
          pages,
          page_size,
          subvecs,
          centroids,
          subdim,
          total_tokens,
          group_size);
    } else {
      gqa_pq_scores_kernel<int64_t><<<blocks, threads, 0, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<int64_t>(),
          scores.data_ptr<float>(),
          heads,
          kv_heads,
          dim,
          pages,
          page_size,
          subvecs,
          centroids,
          subdim,
          total_tokens,
          group_size);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (k == 0) {
    return {top_tokens, top_scores, scores};
  }

  auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  top_scores = std::get<0>(topk);
  auto top_indices = std::get<1>(topk).contiguous();

  const int64_t map_elems = heads * k;
  const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
  map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
      top_indices.data_ptr<int64_t>(),
      page_starts.data_ptr<int64_t>(),
      top_tokens.data_ptr<int64_t>(),
      heads,
      k,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {top_tokens, top_scores, scores};
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({positions, heads, k}, long_opts);
  auto top_scores = torch::empty({positions, heads, k}, score_opts);
  if (positions == 0 || heads == 0 || pages == 0 || k == 0) {
    return {top_tokens, top_scores};
  }

  auto scores = torch::empty({positions * heads, total_tokens}, score_opts);
  const int threads = 256;
  const dim3 blocks(static_cast<unsigned int>(positions * heads), static_cast<unsigned int>(pages), 1);
  const size_t lut_smem = static_cast<size_t>(subvecs * centroids) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (codes.scalar_type() == torch::kUInt8) {
    gqa_causal_pq_scores_lut_kernel<uint8_t><<<blocks, threads, lut_smem, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        page_starts.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size,
        query_start,
        static_prefix,
        static_suffix);
  } else {
    gqa_causal_pq_scores_lut_kernel<int64_t><<<blocks, threads, lut_smem, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size,
        query_start,
        static_prefix,
        static_suffix);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  top_scores = std::get<0>(topk).reshape({positions, heads, k}).contiguous();
  auto top_indices = std::get<1>(topk).contiguous();

  const int64_t map_elems = positions * heads * k;
  const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
  gqa_causal_map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
      top_indices.data_ptr<int64_t>(),
      page_starts.data_ptr<int64_t>(),
      top_tokens.data_ptr<int64_t>(),
      positions,
      heads,
      k,
      page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {top_tokens, top_scores};
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_scores_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix) {
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({positions, heads, k}, long_opts);
  auto top_scores = torch::empty({positions, heads, k}, score_opts);
  auto scores = torch::empty({positions * heads, total_tokens}, score_opts);
  if (positions == 0 || heads == 0 || pages == 0) {
    return {top_tokens, top_scores, scores.reshape({positions, heads, total_tokens}).contiguous()};
  }

  const int threads = 256;
  const dim3 blocks(static_cast<unsigned int>(positions * heads), static_cast<unsigned int>(pages), 1);
  const size_t lut_smem = static_cast<size_t>(subvecs * centroids) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (codes.scalar_type() == torch::kUInt8) {
    gqa_causal_pq_scores_lut_kernel<uint8_t><<<blocks, threads, lut_smem, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<uint8_t>(),
        page_starts.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size,
        query_start,
        static_prefix,
        static_suffix);
  } else {
    gqa_causal_pq_scores_lut_kernel<int64_t><<<blocks, threads, lut_smem, stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        codes.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        positions,
        heads,
        kv_heads,
        dim,
        pages,
        page_size,
        subvecs,
        centroids,
        subdim,
        total_tokens,
        group_size,
        query_start,
        static_prefix,
        static_suffix);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (k > 0) {
    auto topk = at::topk(scores, k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
    top_scores = std::get<0>(topk).reshape({positions, heads, k}).contiguous();
    auto top_indices = std::get<1>(topk).contiguous();

    const int64_t map_elems = positions * heads * k;
    const int map_blocks = static_cast<int>((map_elems + threads - 1) / threads);
    gqa_causal_map_topk_tokens_kernel<<<map_blocks, threads, 0, stream>>>(
        top_indices.data_ptr<int64_t>(),
        page_starts.data_ptr<int64_t>(),
        top_tokens.data_ptr<int64_t>(),
        positions,
        heads,
        k,
        page_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {top_tokens, top_scores, scores.reshape({positions, heads, total_tokens}).contiguous()};
}

std::vector<torch::Tensor> gqa_causal_fullscan_pq_topk_fused_force_cuda(
    torch::Tensor queries,
    torch::Tensor codebooks,
    torch::Tensor codes,
    torch::Tensor page_starts,
    int64_t group_size,
    int64_t budget,
    int64_t query_start,
    int64_t static_prefix,
    int64_t static_suffix,
    int64_t force_mode) {
  TORCH_CHECK(force_mode >= 0 && force_mode <= 2, "force_mode must be 0=auto, 1=smallscan, or 2=localtopk");
  const auto positions = queries.size(0);
  const auto heads = queries.size(1);
  const auto dim = queries.size(2);
  const auto kv_heads = codebooks.size(0);
  const auto pages = codebooks.size(1);
  const auto subvecs = codebooks.size(2);
  const auto centroids = codebooks.size(3);
  const auto subdim = codebooks.size(4);
  const auto page_size = codes.size(2);
  const auto total_tokens = pages * page_size;
  const auto k = std::min<int64_t>(std::max<int64_t>(budget, 0), total_tokens);

  auto long_opts = queries.options().dtype(torch::kLong);
  auto score_opts = queries.options().dtype(torch::kFloat32);
  auto top_tokens = torch::empty({positions, heads, k}, long_opts);
  auto top_scores = torch::empty({positions, heads, k}, score_opts);
  if (positions == 0 || heads == 0 || pages == 0 || k == 0) {
    return {top_tokens, top_scores};
  }
  if (k > 64 || total_tokens > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return gqa_causal_fullscan_pq_topk_cuda(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        budget,
        query_start,
        static_prefix,
        static_suffix);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t rows = positions * heads;
  const int blocks = static_cast<int>(rows);
  if (total_tokens <= 4096 && (force_mode == 1 || (force_mode == 0 && page_size >= 1024))) {
    const int threads = 256;
    size_t shared_bytes =
        static_cast<size_t>(subvecs * centroids) * sizeof(float)
        + static_cast<size_t>(total_tokens) * sizeof(float)
        + static_cast<size_t>(total_tokens) * sizeof(int32_t)
        + static_cast<size_t>(threads) * sizeof(float)
        + static_cast<size_t>(threads) * sizeof(int32_t);
    if (codes.scalar_type() == torch::kUInt8) {
      gqa_causal_pq_topk_fused_smallscan_kernel<uint8_t><<<blocks, threads, shared_bytes, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<uint8_t>(),
          page_starts.data_ptr<int64_t>(),
          top_tokens.data_ptr<int64_t>(),
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
      gqa_causal_pq_topk_fused_smallscan_kernel<int64_t><<<blocks, threads, shared_bytes, stream>>>(
          queries.data_ptr<float>(),
          codebooks.data_ptr<float>(),
          codes.data_ptr<int64_t>(),
          page_starts.data_ptr<int64_t>(),
          top_tokens.data_ptr<int64_t>(),
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
    return {top_tokens, top_scores};
  }
  if (force_mode == 0 && k > 16) {
    return gqa_causal_fullscan_pq_topk_cuda(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        budget,
        query_start,
        static_prefix,
        static_suffix);
  }
  if (force_mode == 1) {
    return gqa_causal_fullscan_pq_topk_cuda(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        budget,
        query_start,
        static_prefix,
        static_suffix);
  }

  auto launch = [&](auto code_ptr, auto k_tag, int threads) {
    constexpr int KMAX = decltype(k_tag)::value;
    size_t shared_bytes =
        static_cast<size_t>(subvecs * centroids) * sizeof(float)
        + static_cast<size_t>(threads * KMAX) * sizeof(float)
        + static_cast<size_t>(threads * KMAX) * sizeof(int32_t)
        + static_cast<size_t>(threads) * sizeof(float)
        + static_cast<size_t>(threads) * sizeof(int32_t);
    gqa_causal_pq_topk_fused_kernel<std::remove_pointer_t<decltype(code_ptr)>, KMAX><<<
        blocks,
        threads,
        shared_bytes,
        stream>>>(
        queries.data_ptr<float>(),
        codebooks.data_ptr<float>(),
        code_ptr,
        page_starts.data_ptr<int64_t>(),
        top_tokens.data_ptr<int64_t>(),
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
  };

  if (codes.scalar_type() == torch::kUInt8) {
    if (k <= 8) {
      launch(codes.data_ptr<uint8_t>(), std::integral_constant<int, 8>{}, 256);
    } else if (k <= 16) {
      launch(codes.data_ptr<uint8_t>(), std::integral_constant<int, 16>{}, 256);
    } else if (k <= 32) {
      launch(codes.data_ptr<uint8_t>(), std::integral_constant<int, 32>{}, 128);
    } else {
      launch(codes.data_ptr<uint8_t>(), std::integral_constant<int, 64>{}, 64);
    }
  } else {
    if (k <= 8) {
      launch(codes.data_ptr<int64_t>(), std::integral_constant<int, 8>{}, 256);
    } else if (k <= 16) {
      launch(codes.data_ptr<int64_t>(), std::integral_constant<int, 16>{}, 256);
    } else if (k <= 32) {
      launch(codes.data_ptr<int64_t>(), std::integral_constant<int, 32>{}, 128);
    } else {
      launch(codes.data_ptr<int64_t>(), std::integral_constant<int, 64>{}, 64);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {top_tokens, top_scores};
}
