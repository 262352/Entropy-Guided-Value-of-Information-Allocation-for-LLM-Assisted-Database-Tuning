from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from cognitune.config.schema import KnobSpec


@dataclass(frozen=True)
class EncodedNode:
    knob: KnobSpec
    value_start: int
    value_stop: int
    mask_index: int


class SpaceEncoder:
    def __init__(self, knobs: list[KnobSpec]):
        self.nodes: list[EncodedNode] = []
        cursor = 0
        for knob in knobs:
            width = len(knob.choices) if knob.kind == "categorical" else 1
            if knob.kind == "bool":
                width = 2
            self.nodes.append(EncodedNode(knob, cursor, cursor + width, cursor + width))
            cursor += width + 1
        self.dimension = cursor

    @staticmethod
    def _numeric_encode(knob: KnobSpec, value: float) -> float:
        lower, upper = float(knob.lower), float(knob.upper)
        if knob.log:
            return (math.log(float(value)) - math.log(lower)) / (math.log(upper) - math.log(lower))
        return (float(value) - lower) / (upper - lower)

    @staticmethod
    def _numeric_decode(knob: KnobSpec, value: float) -> float | int:
        clipped = float(np.clip(value, 0.0, 1.0))
        lower, upper = float(knob.lower), float(knob.upper)
        raw = math.exp(math.log(lower) + clipped * (math.log(upper) - math.log(lower))) if knob.log else lower + clipped * (upper - lower)
        return int(np.rint(raw)) if knob.kind == "integer" else float(raw)

    def encode(self, config: dict[str, Any], active: set[str]) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float64)
        for node in self.nodes:
            knob = node.knob
            is_active = knob.name in active
            vector[node.mask_index] = 1.0 if is_active else 0.0
            if not is_active:
                continue
            value = config[knob.name]
            if knob.kind in {"integer", "float"}:
                vector[node.value_start] = self._numeric_encode(knob, value)
            elif knob.kind == "ordinal":
                vector[node.value_start] = knob.choices.index(value) / (len(knob.choices) - 1)
            else:
                choices = (False, True) if knob.kind == "bool" else knob.choices
                vector[node.value_start + choices.index(value)] = 1.0
        return vector

    def decode(self, vector: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for node in self.nodes:
            knob = node.knob
            if vector[node.mask_index] < 0.5:
                continue
            values = vector[node.value_start:node.value_stop]
            if knob.kind in {"integer", "float"}:
                result[knob.name] = self._numeric_decode(knob, values[0])
            elif knob.kind == "ordinal":
                index = int(np.rint(np.clip(values[0], 0, 1) * (len(knob.choices) - 1)))
                result[knob.name] = knob.choices[index]
            else:
                choices = (False, True) if knob.kind == "bool" else knob.choices
                result[knob.name] = choices[int(np.argmax(values))]
        return result

