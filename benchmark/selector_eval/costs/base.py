from __future__ import annotations

from dataclasses import dataclass, field


MB = 1024.0 * 1024.0


@dataclass(frozen=True)
class CostEvent:
    phase: str
    kind: str
    category: str
    bytes: float


@dataclass
class CostTrace:
    """Structured logical memory traffic for one selector/query decision."""

    events: list[CostEvent] = field(default_factory=list)

    def add(self, *, phase: str, kind: str, category: str, bytes_: float) -> None:
        self.events.append(
            CostEvent(
                phase=str(phase),
                kind=str(kind),
                category=str(category),
                bytes=float(bytes_),
            )
        )

    def read(self, phase: str, category: str, bytes_: float) -> None:
        self.add(phase=phase, kind="read", category=category, bytes_=bytes_)

    def write(self, phase: str, category: str, bytes_: float) -> None:
        self.add(phase=phase, kind="write", category=category, bytes_=bytes_)

    def extend(self, other: "CostTrace") -> None:
        self.events.extend(other.events)

    def bytes(self, *, phase: str | None = None, kind: str | None = None, category: str | None = None) -> float:
        total = 0.0
        for event in self.events:
            if phase is not None and event.phase != phase:
                continue
            if kind is not None and event.kind != kind:
                continue
            if category is not None and event.category != category:
                continue
            total += float(event.bytes)
        return total

    def mb(self, *, phase: str | None = None, kind: str | None = None, category: str | None = None) -> float:
        return self.bytes(phase=phase, kind=kind, category=category) / MB

    def flat_mb(self, prefix: str = "cost") -> dict[str, float]:
        out = {
            f"{prefix}_read_mb": self.mb(kind="read"),
            f"{prefix}_write_mb": self.mb(kind="write"),
            f"{prefix}_total_mb": self.mb(),
        }
        for phase in sorted({event.phase for event in self.events}):
            out[f"{prefix}_{phase}_mb"] = self.mb(phase=phase)
        for category in sorted({event.category for event in self.events}):
            out[f"{prefix}_{category}_mb"] = self.mb(category=category)
        return out


def kv_read_bytes(token_count: int, head_dim: int, key_bytes: int = 2, value_bytes: int = 2) -> int:
    return int(token_count) * int(head_dim) * (int(key_bytes) + int(value_bytes))

