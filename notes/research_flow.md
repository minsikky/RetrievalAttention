# High-Level Research Flow

## Goal
Study long-context attention methods as a path toward algorithms and hardware for efficient long prefill and long decode.

The core hypothesis is that more dynamic and unstructured sparsity can provide better algorithmic efficiency than regular static sparsity, even when it does not immediately produce wall-clock speedups on current GPUs.

## Framing
Long-context attention methods sit on a spectrum.

- Structured/static sparsity: fixed windows, chunk patterns, block patterns, sink tokens, static global tokens.
- Structured plus dynamic sparsity: systems such as RetroInfer and chunk-based retrieval/attention methods that combine regular sparse structure with data-dependent token selection.
- More unstructured/dynamic sparsity: methods such as RetrievalAttention, where graph traversal or ANN-like search chooses a small content-dependent token set.

The important distinction is algorithmic efficiency versus realized hardware efficiency.

- Algorithmic efficiency asks how many tokens/edges/bytes must be examined to preserve dense-attention quality.
- Hardware efficiency asks whether that irregular access pattern maps well to the available machine.

Current GPUs strongly favor regular dense or block-regular parallel work. Therefore a method can be algorithmically superior but still slower in practice because of gather/scatter overhead, poor memory coalescing, synchronization, and low arithmetic intensity.

## Research Direction
The project should first establish the algorithmic case, then use that evidence to motivate hardware support.

1. Compare sparse attention families on long prefill.
   - Measure quality versus algorithmic cost, not only latency.
   - Useful cost metrics include retrieved-token count, visited-node count, memory bytes touched, KV reads, graph edges traversed, and effective attention mass covered.
   - The target claim is not initially "RetrievalAttention is faster on GPU"; the target claim is "more dynamic/unstructured retrieval can preserve accuracy with less attention work."

2. Use RetrievalAttention-style graph retrieval as the aggressive dynamic-sparsity representative.
   - Prefill builds a graph from dense/full-query information.
   - Decode uses graph traversal to retrieve a small content-dependent subset.
   - Static-window handling should be treated as a baseline compatibility mechanism, not the final answer for long decode.

3. Extend the method from long prefill to long decode.
   - Original RetrievalAttention assumes a mostly fixed corpus/ledger and can leave newly decoded tokens in a static window.
   - Long decode needs a continuously evolving memory.
   - The proposed path is online graph update: insert newly generated tokens into the graph and attach them to relevant existing tokens.

4. Evaluate online graph update by algorithmic efficiency and quality.
   - The target is to reduce decode attention complexity from O(n) toward sublinear behavior, ideally close to O(log n) traversal/search cost.
   - The main quality constraint is that sparse attention outputs should remain close enough to dense attention to avoid hidden-state/KV drift and answer degradation.
   - Fixed retrieval budgets are insufficient in some settings, so adaptive budgets based on estimated omitted mass are a key direction.

5. Treat current-GPU latency as a separate, secondary axis.
   - If graph retrieval is slower than dense FlashAttention on GPUs, that does not invalidate the algorithmic result.
   - It identifies a hardware/software mismatch: irregular but lower-work attention is not accelerated by dense-GPU execution models.

6. Use the algorithmic result to motivate hardware.
   - A future accelerator should target graph traversal, sparse KV reads, candidate frontier management, and irregular memory access.
   - The hardware pitch becomes stronger if experiments show that dynamic sparse retrieval touches far less memory than dense attention at similar quality, while GPUs fail to convert that reduced work into proportional latency savings.

## Current Thesis
More aggressive dynamic sparse attention may be the right algorithm for long-context inference, but the wrong match for commodity GPU execution. The research path is to prove the algorithmic efficiency first, extend it to online long decode, then propose hardware that makes the irregular sparse work efficient in practice.

## Immediate Evidence Plan
Use `benchmark/attention_efficiency_eval.py` as the first controlled proxy before spending cluster time on full model sweeps.

- Dense oracle: exact dense top-k upper bound for a fixed token budget.
- Static/chunk baseline: prefix plus local suffix plus regular chunk reads.
- RetroInfer-style baseline: regular contiguous clusters scored by centroids, then expanded to tokens.
- RetrievalAttention-style baseline: content-dependent graph traversal with bounded visited nodes.

Primary plots should show dense mass covered, recall at budget, and sparse-output L2 versus token read ratio. The proxy also reports metadata work, graph nodes visited, graph edges touched, and clusters scored so we can separate "fewer KV reads" from "more index/search work."

The main claim this experiment can support is algorithmic: dynamic/unstructured retrieval can recover dense-attention behavior with fewer token reads. It should not be used as a GPU latency claim.
