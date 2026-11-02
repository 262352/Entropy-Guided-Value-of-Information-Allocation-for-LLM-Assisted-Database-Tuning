from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


NUMERIC_TYPES = {"integer", "float"}
SUPPORTED_TYPES = NUMERIC_TYPES | {"categorical", "ordinal", "bool"}


@dataclass(frozen=True)
class ActivationCondition:
    parent: str
    values: tuple[Any, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ActivationCondition":
        values = raw.get("in")
        if values is None:
            values = [raw["equals"]]
        return cls(parent=str(raw["parent"]), values=tuple(values))


@dataclass(frozen=True)
class KnobSpec:
    name: str
    kind: str
    default: Any
    lower: float | int | None = None
    upper: float | int | None = None
    choices: tuple[Any, ...] = ()
    log: bool = False
    unit: str | None = None
    category: str = "other"
    restart_required: bool = False
    condition: ActivationCondition | None = None
    description: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KnobSpec":
        kind = str(raw["type"])
        if kind not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported knob type {kind!r}")
        condition = raw.get("condition")
        item = cls(
            name=str(raw["name"]), kind=kind, default=raw["default"],
            lower=raw.get("lower"), upper=raw.get("upper"),
            choices=tuple(raw.get("choices", ())), log=bool(raw.get("log", False)),
            unit=raw.get("unit"), category=str(raw.get("category", "other")),
            restart_required=bool(raw.get("restart_required", False)),
            condition=ActivationCondition.from_dict(condition) if condition else None,
            description=str(raw.get("description", "")),
        )
        item.validate_definition()
        return item

    def validate_definition(self) -> None:
        if self.kind in NUMERIC_TYPES:
            if self.lower is None or self.upper is None or self.lower >= self.upper:
                raise ValueError(f"{self.name}: numeric bounds must be increasing")
            if self.log and self.lower <= 0:
                raise ValueError(f"{self.name}: logarithmic bounds must be positive")
            if not self.lower <= self.default <= self.upper:
                raise ValueError(f"{self.name}: default is outside bounds")
        elif self.kind in {"categorical", "ordinal"}:
            if len(self.choices) < 2 or self.default not in self.choices:
                raise ValueError(f"{self.name}: invalid choices/default")
        elif self.kind == "bool" and not isinstance(self.default, bool):
            raise ValueError(f"{self.name}: bool default must be boolean")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dbms: str
    workload: str
    objective: str
    budget: int = 50
    n_init: int = 5
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    methods: tuple[str, ...] = ("cognitune", "smac", "gp_bo")
    candidate_pool_size: int = 10_000
    final_validation_repeats: int = 3
    failure_penalty: float = 1.0
    initialization: dict[str, Any] = field(default_factory=dict)
    cognitune: dict[str, Any] = field(default_factory=dict)
    evaluator: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        data = raw.get("experiment", raw)
        config = cls(
            name=str(data["name"]), dbms=str(data["dbms"]),
            workload=str(data["workload"]), objective=str(data["objective"]),
            budget=int(data.get("budget", 50)), n_init=int(data.get("n_init", 5)),
            seeds=tuple(int(seed) for seed in data.get("seeds", (0, 1, 2, 3, 4))),
            methods=tuple(data.get("methods", ("cognitune", "smac", "gp_bo"))),
            candidate_pool_size=int(data.get("candidate_pool_size", 10_000)),
            final_validation_repeats=int(data.get("final_validation_repeats", 3)),
            failure_penalty=float(data.get("failure_penalty", 1.0)),
            initialization=dict(data.get("initialization", {})),
            cognitune=dict(data.get("cognitune", {})),
            evaluator=dict(data.get("evaluator", {})),
            llm=dict(data.get("llm", {})),
        )
        if config.objective not in {"throughput", "latency"}:
            raise ValueError("objective must be throughput or latency")
        if config.n_init < 1 or config.budget < config.n_init:
            raise ValueError("budget must be at least n_init >= 1")
        if set(config.methods) - {"cognitune", "cognitune_v2", "smac", "gp_bo", "ottertune", "cdbtune", "dbtune", "restune", "mctuner"}:
            raise ValueError("unsupported tuning method")
        return config
