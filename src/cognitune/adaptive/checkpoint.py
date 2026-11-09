"""Controller state recovery; database observations are never re-evaluated."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cognitune.agents.ownership import EXPERTS
from .decision import allocate


def save_checkpoint(directory, records, parameters, regions, age, quotas, residuals, groups, rng, extra_state=None):
    path = Path(directory) / 'controller_checkpoint.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'version':1, 'completed':len(records), 'parameters':parameters,
        'regions':regions, 'age':age, 'quotas':quotas, 'residuals':residuals, 'groups':groups,
        'rng_state':rng.bit_generator.state, 'extra_state':extra_state}, sort_keys=True, indent=2))
    temporary.replace(path)


def recover_controller(directory, records, parameters, space, initial_configs):
    if len(records) > parameters['budget']:
        raise ValueError('saved trial count exceeds budget')
    for record in records:
        if record.method != 'cognitune_v3' or record.seed != parameters['seed'] or not space.validate(record.config):
            raise ValueError('saved trace has incompatible method, seed or config')
        if record.trial < len(initial_configs) and record.config != initial_configs[record.trial]:
            raise ValueError('saved trace differs from shared initial design')
    path = Path(directory) / 'controller_checkpoint.json'
    if path.exists():
        state = json.loads(path.read_text())
        if state['version'] != 1 or state['parameters'] != parameters:
            raise ValueError('checkpoint protocol differs; use a separate experiment')
        if state['completed'] > len(records):
            raise ValueError('checkpoint is ahead of trial trace')
        if state['completed'] == len(records):
            rng = np.random.default_rng()
            rng.bit_generator.state = state['rng_state']
            return (state['regions'], state['age'], state['quotas'], state['residuals'],
                    [tuple(g) for g in state['groups']], rng, 'checkpoint_exact_rng')

    # Legacy traces predate checkpoints. Reconstruct measured controller state
    # from complete diagnostics. The past RNG stream cannot be reconstructed;
    # start and disclose a deterministic recovery stream instead of pretending
    # that this is a bit-for-bit replay of the original run.
    regions = {e:{} for e in EXPERTS}
    age = {k.name:0 for k in space.knobs}
    quotas, _ = allocate({}, EXPERTS, parameters['candidate_pool_size'] - 20)
    residuals, groups = [], []
    events_path = Path(directory) / 'round_diagnostics.jsonl'
    proposals, outcomes, joint = {}, {}, {}
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            row = json.loads(line)
            trial = row.get('trial', -1)
            if trial < len(initial_configs) or trial >= len(records):
                continue
            if row['event'] == 'proposal':
                proposals[trial] = row
            elif row['event'] == 'outcome':
                outcomes[trial] = row
            elif row['event'] == 'joint_review' and row.get('agreed'):
                joint.setdefault(trial, []).append(tuple(sorted(row['hypothesis']['patch'])))
    for record in records[len(initial_configs):]:
        trial = record.trial
        if trial not in proposals or trial not in outcomes:
            raise ValueError(f'cannot recover trial {trial}: complete proposal/outcome diagnostics required')
        proposal, outcome = proposals[trial], outcomes[trial]
        for expert, changes in proposal['region_changes'].items():
            for name, change in changes.items():
                regions[expert][name] = change['after']
        for name in age:
            age[name] = 0 if name in proposal['active'] else age[name] + 1
        quotas = outcome['quota_after']
        groups = joint.get(trial, [])
        if record.status == 'SUCCESS' and record.posterior_std:
            residuals.append(abs(record.utility - record.posterior_mean) / record.posterior_std)
    rng = np.random.default_rng(np.random.SeedSequence([parameters['seed'], len(records), 0x4354]))
    return regions, age, quotas, residuals, groups, rng, 'legacy_diagnostics_new_rng_stream'
