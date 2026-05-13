from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from benchmark.selector_eval.costs.base import CostTrace


@dataclass
class QueryState:
    decode_tokens: int
    position: int
    qidx: int
    head: int
    kv_head: int
    query: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    scores: np.ndarray
    probs: np.ndarray
    base_tokens: list[int]


@dataclass
class SelectionResult:
    algorithm: str
    selected_tokens: list[int]
    candidate_tokens: list[int] = field(default_factory=list)
    cost: CostTrace = field(default_factory=CostTrace)
    metadata: dict = field(default_factory=dict)


class Selector(Protocol):
    name: str

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        ...

