from __future__ import annotations

import json
from pathlib import Path

from .space import SearchSpace


def build_initial_design(space: SearchSpace, seed: int, n_init: int) -> dict:
    default = space.get_default()
    random_configs = space.sample_iid(seed, max(n_init * 2, n_init - 1))
    configs = [default]
    for config in random_configs:
        if space.config_key(config) != space.config_key(default):
            configs.append(config)
        if len(configs) == n_init:
            break
    if len(configs) != n_init:
        raise RuntimeError("could not construct a unique shared initial design")
    return {
        "metadata": {
            "initial_design": "iid_random", "lhs_used": False,
            "sobol_used": False, "halton_used": False,
            "n_init": n_init, "seed": seed, "space_fingerprint": space.fingerprint(),
        },
        "configs": configs,
    }


def save_initial_design(path: str | Path, design: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def load_initial_design(path: str | Path, space: SearchSpace, n_init: int, seed: int) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = data["metadata"]
    expected = {"initial_design": "iid_random", "lhs_used": False, "sobol_used": False, "halton_used": False, "n_init": n_init, "seed": seed, "space_fingerprint": space.fingerprint()}
    if metadata != expected:
        raise ValueError("initial design metadata does not match this run")
    configs = [space.repair(config) for config in data["configs"]]
    if len(configs) != n_init or len({space.config_key(c) for c in configs}) != n_init:
        raise ValueError("initial design must contain n_init unique configurations")
    if configs[0] != space.get_default() or not all(space.validate(c) for c in configs):
        raise ValueError("initial design is invalid or does not begin with defaults")
    return configs

