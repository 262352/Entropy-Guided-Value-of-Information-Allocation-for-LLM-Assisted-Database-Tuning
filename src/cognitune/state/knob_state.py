from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class KnobState(str, Enum):
    ACTIVE = "ACTIVE"
    WATCH = "WATCH"
    DORMANT = "DORMANT"


@dataclass
class KnobStatus:
    state: KnobState
    reason: str
    last_changed_trial: int
    protected: bool = False
    challenge_due: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload

