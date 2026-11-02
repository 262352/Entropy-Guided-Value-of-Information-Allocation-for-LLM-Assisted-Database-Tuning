from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cognitune.evaluator import Evaluator
from cognitune.experiment.recorder import RunRecorder
from cognitune.search_space.space import SearchSpace
from .common import evaluate_and_record


class _Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class _Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


def run_cdbtune(
    space: SearchSpace,
    evaluator: Evaluator,
    initial_configs: list[dict],
    budget: int,
    seed: int,
    candidate_pool_size: int,
    run_dir: str | Path,
    duplicate_new_best: bool = True,
    warmup_trials: int = 5,
) -> dict:
    """CDBTune-style DDPG tuner adapted to the shared 60-knob protocol.

    The actor maps an encoded configuration plus normalized observed
    throughput to a continuous action in [-1, 1]^knobs, which decodes back to
    a legal configuration. The critic is trained from the online transition
    stream. The first five shared-design points and five random warm-up points
    provide the replay buffer, mirroring CDBTune's random exploration phase.
    """
    recorder = RunRecorder(run_dir)
    configs, utilities = [], []
    best_utility = -float("inf")
    best_raw = 0.0
    default_raw = None
    for trial, config in enumerate(initial_configs):
        best_utility, result = evaluate_and_record(
            evaluator, recorder, "cdbtune", seed, trial, config, "shared_initial", best_utility,
            budget=budget,
        )
        configs.append(config)
        utilities.append(result.utility)
        if result.status == "SUCCESS":
            best_raw = max(best_raw, result.raw_metric)
            default_raw = result.raw_metric if default_raw is None else default_raw

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed * 7919 + 17)
    state_dim = space.encoder.dimension + 2
    action_dim = len(space.knobs)
    actor = _Actor(state_dim, action_dim)
    critic = _Critic(state_dim, action_dim)
    actor_target = _Actor(state_dim, action_dim)
    critic_target = _Critic(state_dim, action_dim)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    memory: list[tuple] = []
    for index in range(1, len(recorder.records)):
        previous, current = recorder.records[index - 1], recorder.records[index]
        if previous.status != "SUCCESS" or current.status != "SUCCESS":
            continue
        memory.append((
            _state(space, previous.config, previous.raw_metric, previous.raw_metric, default_raw),
            _action_from_config(space, current.config),
            float(current.utility - previous.utility),
            _state(space, current.config, current.raw_metric, current.raw_metric, default_raw),
        ))
    gamma, tau = 0.9, 0.005
    batch_size = 8
    updates_per_trial = 4

    previous_state = _state(space, configs[-1], best_raw, best_raw, default_raw)
    previous_raw = best_raw
    while len(configs) < budget:
        trial = len(configs)
        if trial < len(initial_configs) + warmup_trials:
            selected = space.repair(space.sample_iid(seed * 100_003 + trial, 1)[0])
            action = _action_from_config(space, selected)
            source = "cdbtune_random_warmup"
        else:
            sigma = 0.3 * (0.05 / 0.3) ** ((trial - len(initial_configs) - warmup_trials) / max(1, budget - len(initial_configs) - warmup_trials))
            with torch.no_grad():
                action = actor(torch.as_tensor(previous_state, dtype=torch.float32)).numpy()
            action = np.clip(action + rng.normal(0.0, sigma, size=action_dim), -1.0, 1.0)
            selected = _config_from_action(space, action)
            if not space.validate(selected) or space.config_key(selected) in {space.config_key(item) for item in configs}:
                selected = space.repair(space.sample_iid(seed * 100_003 + trial, 1)[0])
                action = _action_from_config(space, selected)
            source = "cdbtune_ddpg"

        best_utility, result = evaluate_and_record(
            evaluator, recorder, "cdbtune", seed, trial, selected, source, best_utility,
            budget=budget, duplicate_new_best=duplicate_new_best,
        )
        previous_raw = max(previous_raw, result.raw_metric) if result.status == "SUCCESS" else previous_raw
        next_state = _state(space, selected, result.raw_metric, previous_raw, default_raw)
        reward = float(result.utility - utilities[-1]) if result.status == "SUCCESS" else -1.0
        memory.append((previous_state, np.asarray(action, dtype=np.float32), reward, next_state))
        if len(memory) > 5000:
            memory.pop(0)
        if len(memory) >= batch_size:
            for _ in range(updates_per_trial):
                _train_step(actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, memory, rng, gamma, tau, batch_size)
        previous_state = next_state
        configs.append(selected)
        utilities.append(result.utility)
    return recorder.finalize()


def _train_step(actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, memory, rng, gamma, tau, batch_size):
    indices = rng.integers(0, len(memory), size=batch_size)
    batch = [memory[index] for index in indices]
    states = torch.as_tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
    actions = torch.as_tensor(np.stack([item[1] for item in batch]), dtype=torch.float32)
    rewards = torch.as_tensor([[item[2]] for item in batch], dtype=torch.float32)
    next_states = torch.as_tensor(np.stack([item[3] for item in batch]), dtype=torch.float32)
    with torch.no_grad():
        target_q = rewards + gamma * critic_target(next_states, actor_target(next_states))
    critic_loss = nn.functional.mse_loss(critic(states, actions), target_q)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    actor_loss = -critic(states, actor(states)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    with torch.no_grad():
        for target, source in ((actor_target, actor), (critic_target, critic)):
            for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                target_parameter.mul_(1.0 - tau).add_(tau * source_parameter)


def _state(space: SearchSpace, config: dict, raw: float, best_raw: float, default_raw: float | None) -> np.ndarray:
    scale = max(default_raw or 1.0, 1.0)
    return np.concatenate([
        space.encode(config),
        np.asarray([raw / scale, best_raw / scale], dtype=np.float64),
    ])


def _action_from_config(space: SearchSpace, config: dict) -> np.ndarray:
    action = []
    for knob in space.knobs:
        value = config.get(knob.name, knob.default)
        if knob.kind in {"integer", "float"}:
            lower, upper = float(knob.lower), float(knob.upper)
            if knob.log:
                position = (math.log(float(value)) - math.log(lower)) / (math.log(upper) - math.log(lower))
            else:
                position = (float(value) - lower) / (upper - lower)
        elif knob.kind == "ordinal":
            position = knob.choices.index(value) / max(1, len(knob.choices) - 1)
        elif knob.kind == "bool":
            position = 1.0 if value else 0.0
        else:
            position = knob.choices.index(value) / max(1, len(knob.choices) - 1)
        action.append(2.0 * float(np.clip(position, 0.0, 1.0)) - 1.0)
    return np.asarray(action, dtype=np.float32)


def _config_from_action(space: SearchSpace, action: np.ndarray) -> dict:
    positions = (np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) + 1.0) / 2.0
    config = {}
    for knob, position in zip(space.knobs, positions):
        if knob.kind in {"integer", "float"}:
            lower, upper = float(knob.lower), float(knob.upper)
            raw = math.exp(math.log(lower) + position * (math.log(upper) - math.log(lower))) if knob.log else lower + position * (upper - lower)
            config[knob.name] = int(round(raw)) if knob.kind == "integer" else float(raw)
        elif knob.kind == "ordinal":
            index = int(round(position * (len(knob.choices) - 1)))
            config[knob.name] = knob.choices[index]
        elif knob.kind == "bool":
            config[knob.name] = bool(position >= 0.5)
        else:
            index = int(round(position * (len(knob.choices) - 1)))
            config[knob.name] = knob.choices[index]
    return space.repair(config)
