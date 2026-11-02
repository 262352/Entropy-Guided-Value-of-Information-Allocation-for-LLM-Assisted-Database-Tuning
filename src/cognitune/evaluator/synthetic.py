from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from cognitune.search_space.space import SearchSpace
from .base import EvaluationResult, Evaluator, metric_utility


class SyntheticEvaluator(Evaluator):
    """A deterministic noisy landscape with nonlinear, conditional and joint effects."""

    def __init__(
        self,
        space: SearchSpace,
        seed: int,
        objective: str = "throughput",
        default_metric: float = 1000.0,
        noise_std: float = 0.01,
    ):
        self.space = space
        self.seed = seed
        self.objective = objective
        self.default_metric = default_metric
        self.noise_std = noise_std
        self.default_latent = self._latent(space.get_default())

    def evaluate(self, config: dict, trial: int) -> EvaluationResult:
        started = time.monotonic()
        config = self.space.repair(config)
        if not self.space.validate(config):
            return EvaluationResult(0.0, -1.0, "INVALID", metadata={"trial": trial}, config=config)
        latent = self._latent(config) - self.default_latent
        noise = self._noise(config)
        signed = latent + noise
        metric = self.default_metric * math.exp(signed if self.objective == "throughput" else -signed)
        utility = metric_utility(metric, self.default_metric, self.objective)
        return EvaluationResult(
            raw_metric=metric, utility=utility, status="SUCCESS", elapsed_seconds=time.monotonic() - started,
            metadata={"latent": latent, "noise": noise, "trial": trial}, config=config,
        )

    def _latent(self, config: dict) -> float:
        vector = self.space.encode(config)
        core = vector[: min(6, len(vector))]
        targets = np.asarray([0.75, 0.25, 0.68, 0.82, 0.35, 0.6][:len(core)])
        main = -0.35 * float(np.sum((core - targets) ** 2))
        nonlinear = 0.12 * math.sin(4 * math.pi * float(core[0])) if len(core) else 0.0
        interaction = 0.0
        if len(core) >= 2:
            interaction += 0.22 * math.exp(-((core[0] - 0.72) ** 2 + (core[1] - 0.28) ** 2) / 0.018)
        if len(core) >= 4:
            # Neither member is useful alone near the default: this is the joint-only test feature.
            interaction += 0.28 * math.exp(-((core[2] - 0.78) ** 2 + (core[3] - 0.78) ** 2) / 0.012)
            interaction -= 0.04 * (abs(core[2] - 0.78) + abs(core[3] - 0.78))
        active_bonus = 0.015 * len(self.space.get_active_knobs(config))
        category = sum((index + 1) * 0.004 for index, value in enumerate(vector) if value > 0.5)
        return main + nonlinear + interaction + active_bonus + category

    def _noise(self, config: dict) -> float:
        payload = f"{self.seed}:{self.space.config_key(config)}".encode()
        derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        return float(np.random.default_rng(derived).normal(0.0, self.noise_std))
