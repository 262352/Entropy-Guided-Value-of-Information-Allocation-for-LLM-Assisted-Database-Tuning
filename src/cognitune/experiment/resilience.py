"""Durable evaluation replay and bounded, evidence-preserving recovery."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import uuid
from cognitune.evaluator.base import Evaluator, EvaluationResult


class InfrastructureUnavailable(RuntimeError):
    pass


class ReplayDivergence(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


class DurableEvaluator(Evaluator):
    """Optimizers deterministically replay a prefix, never redo a cached trial.

    All real attempts and diagnostic controls remain separately visible. A
    healthy default plus reproducible candidate failure is a failed trial;
    failed default recovery suspends the worker without poisoning the model.
    """
    def __init__(self, backend, directory, *, initial=None, candidate_attempts=2):
        self.backend, self.directory = backend, Path(directory)
        self.initial = initial or {}
        self.candidate_attempts = candidate_attempts
        self.directory.mkdir(parents=True, exist_ok=True)

    def event(self, **item):
        def clean(value):
            if isinstance(value,float) and not math.isfinite(value): return str(value)
            if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
            if isinstance(value,list): return [clean(v) for v in value]
            return value
        with (self.directory/'recovery_events.jsonl').open('a') as f:
            f.write(json.dumps(clean(item), sort_keys=True, allow_nan=False) + '\n')
            f.flush()

    def evaluate(self, config, trial):
        request = self.directory/f'trial_{trial:04d}'/'request.json'
        saved = request.with_name('result.json')
        if request.exists():
            if json.loads(request.read_text())['config'] != config:
                raise ReplayDivergence(f'optimizer proposal differs at trial {trial}; original evidence retained')
        else:
            atomic_json(request, {'trial':trial, 'config':config})
        if saved.exists():
            value = EvaluationResult(**json.loads(saved.read_text()))
            if value.config != config:
                raise ReplayDivergence('cached result/config mismatch')
            if value.status=='SUCCESS' and (not math.isfinite(value.raw_metric) or value.raw_metric<=0 or not math.isfinite(value.utility)):
                raise ReplayDivergence('cached successful metric is invalid; original retained')
            self.event(trial=trial,kind='cached_trial_replay')
            return value
        if trial in self.initial:
            value = EvaluationResult(**self.initial[trial])
            if value.config != config:
                raise ReplayDivergence('shared initial config mismatch')
            value.metadata = dict(value.metadata or {}, shared_initial_reused=True)
            atomic_json(saved, value.as_dict())
            return value
        last = None
        for attempt in range(self.candidate_attempts):
            token = f'trial_{trial:04d}/attempt_{uuid.uuid4().hex}'
            try:
                last = self.backend.measure(config, token)
            except Exception as error:
                last = EvaluationResult(0., -10., 'INFRA_ERROR', config=config,
                    metadata={'error':str(error), 'exception':type(error).__name__})
            self.event(trial=trial, attempt=attempt, kind='candidate_attempt', result=last.as_dict())
            if last.status == 'SUCCESS' and math.isfinite(last.raw_metric) and last.raw_metric > 0 and math.isfinite(last.utility):
                atomic_json(saved, last.as_dict())
                return last
            # A default control must actually execute this workload, not just SELECT 1.
            try:
                control = self.backend.recover(f'controls/{uuid.uuid4().hex}')
            except Exception as error:
                self.event(trial=trial, kind='recovery_failed', error=str(error))
                raise InfrastructureUnavailable(f'default control failed after trial {trial}: {error}') from error
            self.event(trial=trial, kind='default_control', result=control.as_dict())
            if control.status != 'SUCCESS' or not math.isfinite(control.raw_metric) or control.raw_metric <= 0:
                raise InfrastructureUnavailable(f'nonpositive/failed default control after trial {trial}')
        last.status = 'CRASH'
        last.utility = -10.
        last.metadata = dict(last.metadata or {}, classification='reproduced_candidate_failure_with_healthy_default')
        atomic_json(saved, last.as_dict())
        return last
