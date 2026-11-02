from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    request_id: str = ""
    latency_seconds: float = 0.0
    normalization_repairs: tuple[str, ...] = ()


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, *, expert: str, template: str) -> LLMResult:
        raise NotImplementedError
