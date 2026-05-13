from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QKVTrace:
    keys: np.ndarray
    values: np.ndarray
    queries: np.ndarray
    positions: np.ndarray
    input_len: int
    metadata: dict

    @property
    def num_heads(self) -> int:
        return int(self.queries.shape[0])

    @property
    def kv_heads(self) -> int:
        return int(self.keys.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.queries.shape[-1])

    def kv_head_for(self, head: int) -> int:
        return min(self.kv_heads - 1, int(head) * self.kv_heads // max(1, self.num_heads))

    def decode_tokens_for_qidx(self, qidx: int) -> int:
        return max(0, int(self.positions[int(qidx)]) - int(self.input_len) + 1)

    def q_indices_for_decodes(self, decode_lengths: list[int]) -> list[int]:
        wanted = {int(x) for x in decode_lengths}
        return [idx for idx in range(self.positions.shape[0]) if self.decode_tokens_for_qidx(idx) in wanted]


def load_trace(path: str | Path) -> QKVTrace:
    data = np.load(Path(path))
    keys = np.asarray(data["keys"], dtype=np.float32)
    values = np.asarray(data["values"], dtype=np.float32)
    queries = np.asarray(data["queries"], dtype=np.float32)
    positions = np.asarray(data["positions"], dtype=np.int64)
    metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
    input_len = int(metadata.get("input_len", int(positions.min()) + 1))
    return QKVTrace(keys=keys, values=values, queries=queries, positions=positions, input_len=input_len, metadata=metadata)


def static_tokens(position: int, static_prefix: int, static_suffix: int) -> list[int]:
    end = int(position) + 1
    prefix = list(range(0, min(int(static_prefix), end)))
    suffix_start = max(0, end - max(0, int(static_suffix)))
    suffix = list(range(suffix_start, end))
    return sorted(set(prefix + suffix))


def unique_tokens(tokens: list[int], *, context_len: int) -> list[int]:
    seen = set()
    out = []
    for tok in tokens:
        tok = int(tok)
        if tok < 0 or tok >= int(context_len) or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def attention_probs(keys: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = 1.0 / np.sqrt(float(query.shape[-1]))
    scores = (keys.astype(np.float32, copy=False) @ query.astype(np.float32, copy=False)) * scale
    logits = scores - np.max(scores)
    probs = np.exp(logits).astype(np.float32)
    probs /= max(float(probs.sum()), 1e-20)
    return scores.astype(np.float32, copy=False), probs

