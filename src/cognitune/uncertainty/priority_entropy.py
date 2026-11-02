from __future__ import annotations

import math


def priority_statistics(responses: list, knob_names: list[str], alpha: float = 0.5) -> dict[str, dict[str, float]]:
    if not responses:
        raise ValueError("at least one cognitive sample is required")
    result = {}
    for knob in knob_names:
        counts = {"high": 0, "medium": 0, "low": 0}
        for response in responses:
            for level in counts:
                counts[level] += int(knob in response.priority[level])
        total = len(responses) + 3 * alpha
        probabilities = {level: (count + alpha) / total for level, count in counts.items()}
        entropy = -sum(p * math.log(p) for p in probabilities.values()) / math.log(3)
        score = probabilities["high"] + 0.5 * probabilities["medium"]
        result[knob] = {"selection_entropy": entropy, "priority_score": score, **{f"p_{k}": v for k, v in probabilities.items()}}
    return result

