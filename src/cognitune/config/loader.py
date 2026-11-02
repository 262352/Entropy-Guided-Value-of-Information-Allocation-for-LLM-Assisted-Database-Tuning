from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import ExperimentConfig, KnobSpec


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping in {path}")
    return data


def load_search_space(path: str | Path):
    from cognitune.search_space.space import SearchSpace

    raw = load_yaml(path)
    knobs = [KnobSpec.from_dict(item) for item in raw["knobs"]]
    return SearchSpace(
        knobs=knobs,
        forbidden=list(raw.get("forbidden_combinations", [])),
        hard_constraints=list(raw.get("hard_constraints", [])),
    )


def load_experiment(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(load_yaml(path))

