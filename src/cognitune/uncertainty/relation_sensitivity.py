from __future__ import annotations

from collections import defaultdict

import numpy as np

from cognitune.config.schema import KnobSpec
from .value_entropy import _entropy, recommendation_distribution, value_grid


def relation_sensitivity(relation, target: KnobSpec, alpha: float = 0.1) -> float | None:
    if len(relation.branches) < 2:
        return None
    grid = value_grid(target)
    distributions = []
    for branch in relation.branches:
        proxy = type("BranchRecommendation", (), {
            "recommended_values": branch.target_values,
            "recommended_interval": branch.target_interval,
        })()
        distributions.append(recommendation_distribution(target, proxy, grid, alpha))
    average = np.mean(distributions, axis=0)
    return max(0.0, _entropy(average) - float(np.mean([_entropy(item) for item in distributions])))


def relation_statistics(responses: list, knobs: dict[str, KnobSpec], kappa: float = 1.0) -> list[dict]:
    grouped = defaultdict(list)
    support = defaultdict(int)
    for response in responses:
        for relation in response.conditional_relations:
            value = relation_sensitivity(relation, knobs[relation.target])
            if value is not None:
                grouped[(relation.source, relation.target)].append(value)
                support[(relation.source, relation.target)] += 1
    result = []
    for (source, target), values in sorted(grouped.items()):
        mean, std = float(np.mean(values)), float(np.std(values, ddof=0))
        result.append({
            "source": source, "target": target, "weight": mean - kappa * std,
            "support_count": support[(source, target)], "mean": mean, "std": std,
            "lower_bound": mean - kappa * std, "status": "unverified",
        })
    return result

