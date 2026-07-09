from __future__ import annotations

import contextlib
import types
from dataclasses import dataclass
from typing import Iterator

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


@dataclass
class _LayerKV:
    key: torch.Tensor
    value: torch.Tensor
    length: int = 0


class DenseKVOffloadCache(Cache):
    """Pre-sized CPU KV cache with a shared bounded GPU staging pool."""

    def __init__(
        self,
        *,
        num_layers: int,
        max_cache_len: int,
        kv_block_tokens: int,
        staging_buffers: int,
        query_block_tokens: int = 2048,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if num_layers <= 0 or max_cache_len <= 0:
            raise ValueError("num_layers and max_cache_len must be positive")
        if kv_block_tokens <= 0 or query_block_tokens <= 0:
            raise ValueError("KV and query block sizes must be positive")
        if staging_buffers < 2:
            raise ValueError("staging_buffers must be at least 2 for H2D/compute overlap")
        self.num_layers = int(num_layers)
        self.max_cache_len = int(max_cache_len)
        self.kv_block_tokens = int(kv_block_tokens)
        self.query_block_tokens = int(query_block_tokens)
        self.staging_buffers = int(staging_buffers)
        self.device = torch.device(device)
        self._layers: list[_LayerKV | None] = [None] * self.num_layers
        self._seen_tokens = 0
        self.h2d_bytes = 0

        self._copy_stream: torch.cuda.Stream | None = None
        self._stage_keys: list[torch.Tensor] = []
        self._stage_values: list[torch.Tensor] = []
        self._ready_events: list[torch.cuda.Event] = []
        self._consumed_events: list[torch.cuda.Event] = []
        self._slot_used: list[bool] = []

    def __len__(self) -> int:
        return self.num_layers

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        idx = 0 if layer_idx is None else int(layer_idx)
        layer = self._layers[idx]
        return 0 if layer is None else int(layer.length)

    def get_max_cache_shape(self) -> None:
        # This is a capacity limit for the CPU allocation, not a StaticCache
        # attention length: unused capacity must never be visible to attention.
        return None

    def _allocate_layer(self, key_states: torch.Tensor, value_states: torch.Tensor) -> _LayerKV:
        if key_states.shape != value_states.shape:
            raise ValueError("key and value states must have identical shapes")
        if key_states.ndim != 4 or int(key_states.shape[0]) != 1:
            raise ValueError("dense KV offload currently requires batch size 1 KV tensors")
        shape = (
            int(key_states.shape[0]),
            int(key_states.shape[1]),
            self.max_cache_len,
            int(key_states.shape[3]),
        )
        pin_memory = self.device.type == "cuda"
        return _LayerKV(
            key=torch.empty(shape, dtype=key_states.dtype, device="cpu", pin_memory=pin_memory),
            value=torch.empty(shape, dtype=value_states.dtype, device="cpu", pin_memory=pin_memory),
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        idx = int(layer_idx)
        if not 0 <= idx < self.num_layers:
            raise IndexError(f"layer index {idx} is outside a {self.num_layers}-layer cache")
        layer = self._layers[idx]
        if layer is None:
            layer = self._allocate_layer(key_states, value_states)
            self._layers[idx] = layer
        if layer.key.dtype != key_states.dtype or layer.value.dtype != value_states.dtype:
            raise TypeError("KV dtype changed after cache allocation")

        append_len = int(key_states.shape[-2])
        start = int(layer.length)
        stop = start + append_len
        if stop > self.max_cache_len:
            raise RuntimeError(
                f"KV cache capacity exceeded at layer {idx}: {stop} > {self.max_cache_len}"
            )
        # A blocking D2H append makes the pinned source ready before the
        # independent H2D stream starts reading it. Each token is appended
        # once; the much larger repeated prefill traffic is H2D.
        layer.key[:, :, start:stop, :].copy_(key_states.detach(), non_blocking=False)
        layer.value[:, :, start:stop, :].copy_(value_states.detach(), non_blocking=False)
        layer.length = stop
        if idx == 0:
            self._seen_tokens = stop

        # Qwen2Attention normally expects update() to return the full cache.
        # The patched attention consumes the CPU store directly, so returning
        # only the new tensors avoids accidentally materializing GPU history.
        return key_states, value_states

    def _ensure_staging(self, layer: _LayerKV) -> None:
        if self.device.type != "cuda" or self._stage_keys:
            return
        shape = (
            int(layer.key.shape[0]),
            int(layer.key.shape[1]),
            self.kv_block_tokens,
            int(layer.key.shape[3]),
        )
        self._copy_stream = torch.cuda.Stream(device=self.device)
        for _ in range(self.staging_buffers):
            self._stage_keys.append(torch.empty(shape, dtype=layer.key.dtype, device=self.device))
            self._stage_values.append(torch.empty(shape, dtype=layer.value.dtype, device=self.device))
            self._ready_events.append(torch.cuda.Event())
            self._consumed_events.append(torch.cuda.Event())
            self._slot_used.append(False)

    def _enqueue_h2d(self, layer: _LayerKV, start: int, stop: int, slot: int) -> None:
        if self._copy_stream is None:
            raise RuntimeError("CUDA staging was not initialized")
        count = int(stop - start)
        with torch.cuda.stream(self._copy_stream):
            if self._slot_used[slot]:
                self._copy_stream.wait_event(self._consumed_events[slot])
            self._stage_keys[slot][:, :, :count, :].copy_(
                layer.key[:, :, start:stop, :], non_blocking=True
            )
            self._stage_values[slot][:, :, :count, :].copy_(
                layer.value[:, :, start:stop, :], non_blocking=True
            )
            self._ready_events[slot].record(self._copy_stream)
        self._slot_used[slot] = True
        self.h2d_bytes += count * (
            int(layer.key.shape[0])
            * int(layer.key.shape[1])
            * int(layer.key.shape[3])
            * (layer.key.element_size() + layer.value.element_size())
        )

    def _cpu_blocks(self, layer: _LayerKV) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        for start in range(0, int(layer.length), self.kv_block_tokens):
            stop = min(int(layer.length), start + self.kv_block_tokens)
            yield start, layer.key[:, :, start:stop, :], layer.value[:, :, start:stop, :]

    def _cuda_blocks(self, layer: _LayerKV) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        self._ensure_staging(layer)
        current_stream = torch.cuda.current_stream(self.device)
        blocks = [
            (start, min(int(layer.length), start + self.kv_block_tokens))
            for start in range(0, int(layer.length), self.kv_block_tokens)
        ]
        initial = min(self.staging_buffers, len(blocks))
        for block_idx in range(initial):
            start, stop = blocks[block_idx]
            self._enqueue_h2d(layer, start, stop, block_idx)

        for block_idx, (start, stop) in enumerate(blocks):
            slot = block_idx % self.staging_buffers
            current_stream.wait_event(self._ready_events[slot])
            count = int(stop - start)
            yield (
                start,
                self._stage_keys[slot][:, :, :count, :],
                self._stage_values[slot][:, :, :count, :],
            )
            self._consumed_events[slot].record(current_stream)
            next_idx = block_idx + self.staging_buffers
            if next_idx < len(blocks):
                next_start, next_stop = blocks[next_idx]
                self._enqueue_h2d(layer, next_start, next_stop, slot)

    def iter_layer_blocks(self, layer_idx: int) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        layer = self._layers[int(layer_idx)]
        if layer is None or layer.length == 0:
            raise RuntimeError(f"layer {layer_idx} has no cached KV")
        if self.device.type == "cuda":
            yield from self._cuda_blocks(layer)
        else:
            yield from self._cpu_blocks(layer)


def streamed_exact_attention(
    query_states: torch.Tensor,
    cache: DenseKVOffloadCache,
    *,
    layer_idx: int,
    query_start: int,
    scaling: float,
) -> torch.Tensor:
    """Exact GQA attention merged blockwise with fp32 softmax state."""
    if query_states.ndim != 4 or int(query_states.shape[0]) != 1:
        raise ValueError("dense KV offload currently supports batch size 1")
    batch, query_heads, query_len, head_dim = map(int, query_states.shape)
    layer = cache._layers[int(layer_idx)]
    if layer is None:
        raise RuntimeError(f"layer {layer_idx} was not appended before attention")
    kv_heads = int(layer.key.shape[1])
    if query_heads % kv_heads:
        raise ValueError(f"{query_heads} query heads are not divisible by {kv_heads} KV heads")
    groups = query_heads // kv_heads
    grouped_query = query_states.reshape(batch, kv_heads, groups, query_len, head_dim)

    row_max = torch.full(
        (batch, kv_heads, groups, query_len),
        -torch.inf,
        dtype=torch.float32,
        device=query_states.device,
    )
    row_sum = torch.zeros_like(row_max)
    numerator = torch.zeros(
        (batch, kv_heads, groups, query_len, head_dim),
        dtype=torch.float32,
        device=query_states.device,
    )

    for key_start, key_block, value_block in cache.iter_layer_blocks(layer_idx):
        key_len = int(key_block.shape[-2])
        key_stop = key_start + key_len
        key_t = key_block.unsqueeze(2).transpose(-1, -2)
        value_f32 = value_block.unsqueeze(2).float()
        for query_offset in range(0, query_len, cache.query_block_tokens):
            query_stop = min(query_len, query_offset + cache.query_block_tokens)
            query_block = grouped_query[:, :, :, query_offset:query_stop, :]
            # The dot product follows the model dtype. Softmax statistics and
            # the weighted-V numerator are explicitly accumulated in fp32.
            scores = torch.matmul(query_block, key_t).float()
            scores.mul_(float(scaling))

            first_query_position = query_start + query_offset
            last_query_position = query_start + query_stop - 1
            if key_stop - 1 > first_query_position:
                query_positions = torch.arange(
                    first_query_position,
                    last_query_position + 1,
                    device=query_states.device,
                )
                key_positions = torch.arange(key_start, key_stop, device=query_states.device)
                causal = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
                scores.masked_fill_(causal.view(1, 1, 1, query_stop - query_offset, key_len), -torch.inf)

            old_max = row_max[:, :, :, query_offset:query_stop]
            old_sum = row_sum[:, :, :, query_offset:query_stop]
            old_numerator = numerator[:, :, :, query_offset:query_stop, :]
            block_max = scores.amax(dim=-1)
            merged_max = torch.maximum(old_max, block_max)
            old_scale = torch.exp(old_max - merged_max)
            block_exp = torch.exp(scores - merged_max.unsqueeze(-1))

            old_numerator.mul_(old_scale.unsqueeze(-1))
            old_numerator.add_(torch.matmul(block_exp, value_f32))
            old_sum.mul_(old_scale)
            old_sum.add_(block_exp.sum(dim=-1))
            old_max.copy_(merged_max)

    output = numerator / row_sum.unsqueeze(-1)
    return output.to(query_states.dtype).permute(0, 3, 1, 2, 4).reshape(
        batch, query_len, query_heads, head_dim
    )


def _qwen2_offload_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    del attention_mask, cache_position
    if self.training:
        raise ValueError("dense KV offload is an inference-only attention path")
    if kwargs.get("output_attentions", False):
        raise ValueError("dense KV offload does not support output_attentions=True")
    if not isinstance(past_key_value, DenseKVOffloadCache):
        raise TypeError("patched Qwen2 attention requires DenseKVOffloadCache")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_start = past_key_value.get_seq_length(self.layer_idx)
    past_key_value.update(key_states, value_states, self.layer_idx)
    attn_output = streamed_exact_attention(
        query_states,
        past_key_value,
        layer_idx=self.layer_idx,
        query_start=query_start,
        scaling=self.scaling,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


def _no_causal_mask(self, *args, **kwargs):
    del self, args, kwargs
    # The patched attention enforces causality inside each bounded score tile.
    return None


@contextlib.contextmanager
def patched_qwen2_dense_kv_offload(model):
    """Patch only Qwen2 attention and mask construction for this context."""
    if getattr(model.config, "model_type", None) != "qwen2":
        raise TypeError("dense KV offload currently supports Qwen2/Qwen2.5 models only")
    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if base_model is None or layers is None:
        raise TypeError("model does not expose the expected Qwen2 decoder layout")

    original_mask = base_model._update_causal_mask
    original_forwards = [layer.self_attn.forward for layer in layers]
    base_model._update_causal_mask = types.MethodType(_no_causal_mask, base_model)
    for layer in layers:
        layer.self_attn.forward = types.MethodType(_qwen2_offload_attention_forward, layer.self_attn)
    try:
        yield
    finally:
        base_model._update_causal_mask = original_mask
        for layer, original_forward in zip(layers, original_forwards):
            layer.self_attn.forward = original_forward
