from __future__ import annotations

from .knob_state import KnobState, KnobStatus


class StateManager:
    def __init__(
        self,
        statuses: dict[str, KnobStatus],
        challenge_interval: int = 5,
        active_cap: int | None = None,
        challenge_reserve: set[str] | None = None,
        reserve_until: int = 0,
    ):
        self.statuses = statuses
        self.challenge_interval = challenge_interval
        self.active_cap = active_cap
        self.challenge_reserve = challenge_reserve or set()
        self.reserve_until = reserve_until

    @classmethod
    def initialize(
        cls,
        priority_stats: dict[str, dict[str, float]],
        protected: set[str] | None = None,
        active_fraction: float = 0.4,
        exploit_budget: int | None = None,
        challenge_budget: int = 0,
        challenge_interval: int = 5,
    ) -> "StateManager":
        protected = protected or set()
        ranked = sorted(priority_stats, key=lambda name: (-priority_stats[name]["priority_score"], name))
        active_count = max(1, round(len(ranked) * active_fraction)) if exploit_budget is None else min(len(ranked), exploit_budget)
        selected = set(ranked[:active_count])
        challenge_ranked = sorted(
            (name for name in ranked if name not in selected),
            key=lambda name: (-priority_stats[name].get("selection_entropy", 0.0), name),
        )
        selected.update(challenge_ranked[:challenge_budget])
        statuses = {}
        for name in ranked:
            is_protected = name in protected
            state = KnobState.ACTIVE if name in selected or is_protected else KnobState.WATCH
            statuses[name] = KnobStatus(state, "initial_priority", 0, is_protected)
        active_cap = None
        if exploit_budget is not None:
            active_cap = int(exploit_budget) + int(challenge_budget)
        reserve = set(challenge_ranked[:challenge_budget])
        return cls(statuses, challenge_interval, active_cap, reserve, 3 * challenge_interval)

    def update(
        self,
        trial: int,
        condition_improvement: dict[str, float],
        uncertainty: dict[str, float],
        observation_counts: dict[str, int],
        priority_scores: dict[str, float] | None = None,
        promote_threshold: float = 0.02,
        demote_threshold: float = 0.002,
        min_observations: int = 3,
        promote_uncertainty_threshold: float = 0.85,
        demote_uncertainty_threshold: float = 0.45,
        priority_floor: float = 0.25,
    ) -> list[dict]:
        transitions = []
        for name, status in self.statuses.items():
            previous = status.state
            gain = condition_improvement.get(name, 0.0)
            count = observation_counts.get(name, 0)
            priority = priority_scores.get(name) if priority_scores else None
            if status.protected:
                state = KnobState.ACTIVE
                reason = "protected"
            elif gain >= promote_threshold and (priority is None or priority >= priority_floor):
                state = KnobState.ACTIVE
                reason = "promising_conditional_gain"
            elif uncertainty.get(name, 0.0) >= promote_uncertainty_threshold:
                # High cognitive disagreement is a reason to keep watching a
                # knob, not a reason to put every knob into the main space.
                state = KnobState.WATCH if previous == KnobState.DORMANT else previous
                reason = "uncertain_watch"
            elif count >= min_observations and gain <= demote_threshold:
                if uncertainty.get(name, 0.0) >= demote_uncertainty_threshold:
                    state = previous
                    reason = "retain_uncertain"
                else:
                    state = KnobState.WATCH if previous == KnobState.ACTIVE else KnobState.DORMANT
                    reason = "low_conditional_improvement"
            else:
                state = previous
                reason = "retain"
            if state != previous:
                status.last_changed_trial = trial
                status.reason = reason
                transitions.append({"knob": name, "from": previous.value, "to": state.value, "reason": reason})
            status.state = state

        if trial <= self.reserve_until:
            for name in self.challenge_reserve:
                status = self.statuses.get(name)
                if status is not None and not status.protected and status.state != KnobState.ACTIVE:
                    transitions.append({
                        "knob": name, "from": status.state.value, "to": KnobState.ACTIVE.value,
                        "reason": "challenge_reserve",
                    })
                    status.state = KnobState.ACTIVE
                    status.reason = "challenge_reserve"
                    status.last_changed_trial = trial

        if self.active_cap is not None:
            protected_count = sum(1 for status in self.statuses.values() if status.protected)
            active = [name for name, status in self.statuses.items() if status.state == KnobState.ACTIVE and not status.protected]
            reserve_active = {
                name for name in active
                if name in self.challenge_reserve and trial <= self.reserve_until
            }
            overflow = len(active) - reserve_active.__len__() - max(0, self.active_cap - protected_count)
            if overflow > 0:
                ranking = sorted(
                    (name for name in active if name not in reserve_active),
                    key=lambda name: ((priority_scores or {}).get(name, 0.0), name),
                )
                for name in ranking[:overflow]:
                    status = self.statuses[name]
                    transitions.append({"knob": name, "from": status.state.value, "to": KnobState.WATCH.value, "reason": "active_budget_cap"})
                    status.state = KnobState.WATCH
                    status.reason = "active_budget_cap"
                    status.last_changed_trial = trial

        for status in self.statuses.values():
            status.challenge_due = (
                status.state != KnobState.ACTIVE
                and trial > 0
                and trial % self.challenge_interval == 0
            )
        return transitions

    def names(self, *states: KnobState) -> list[str]:
        accepted = set(states)
        return sorted(name for name, status in self.statuses.items() if status.state in accepted)

    def challenge_names(self) -> list[str]:
        return sorted(name for name, status in self.statuses.items() if status.challenge_due)

    def as_dict(self) -> dict[str, dict]:
        return {name: status.as_dict() for name, status in self.statuses.items()}
