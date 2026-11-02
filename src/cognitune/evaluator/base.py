from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class EvaluationResult:
    raw_metric: float
    utility: float
    status: str = "ok"
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    restart_seconds: float = 0.0
    warmup_seconds: float = 0.0
    benchmark_seconds: float = 0.0
    system_metrics: dict[str, Any] | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, config: dict, trial: int) -> EvaluationResult:
        raise NotImplementedError

    def prepare(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def restore_default(self) -> None:
        pass

    def health_check(self) -> bool:
        return True

    def close(self) -> None:
        pass


def metric_utility(metric: float, default_metric: float, objective: str) -> float:
    if metric <= 0 or default_metric <= 0:
        raise ValueError("metrics must be positive")
    sign = 1.0 if objective == "throughput" else -1.0
    return sign * __import__("math").log(metric / default_metric)
