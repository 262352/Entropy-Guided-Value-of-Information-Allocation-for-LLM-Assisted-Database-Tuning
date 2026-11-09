import json
from pathlib import Path

import pytest

from cognitune.adaptive.proposals import StructuredMockProposer
from cognitune.evaluator import SyntheticEvaluator
from cognitune.experiment.recorder import RunRecorder
from cognitune.optimizers.cognitune_v3 import run_cognitune_v3
from cognitune.search_space.initial_design import build_initial_design


def run(space, directory, proposer, **kwargs):
    return run_cognitune_v3(space,SyntheticEvaluator(space,0,'throughput',1000,0),
        build_initial_design(space,0,3)['configs'],7,0,40,directory,proposer=proposer,
        posterior_samples=64,fantasies=4,audit_size=2,active_target=4,**kwargs)


class Interrupt:
    def __init__(self, trial=4):
        self.trial=trial
    def __call__(self, expert, context):
        if context['trial'] == self.trial:
            raise RuntimeError('simulated expert interruption')
        return StructuredMockProposer()(expert,context)


def rows(directory):
    return [json.loads(line) for line in (directory/'trials.jsonl').read_text().splitlines()]


def test_resume_restores_rng_and_does_not_repeat_database_observations(space,tmp_path):
    baseline=tmp_path/'baseline'
    resumed=tmp_path/'resumed'
    run(space,baseline,StructuredMockProposer())
    with pytest.raises(RuntimeError,match='expert interruption'):
        run(space,resumed,Interrupt())
    before=(resumed/'trials.jsonl').read_bytes()
    assert len(rows(resumed)) == 4
    checkpoint=json.loads((resumed/'controller_checkpoint.json').read_text())
    assert checkpoint['completed'] == 4
    run(space,resumed,StructuredMockProposer(),resume=True)
    assert (resumed/'trials.jsonl').read_bytes().startswith(before)
    actual,expected=rows(resumed),rows(baseline)
    assert [r['config'] for r in actual] == [r['config'] for r in expected]
    assert [r['raw_metric'] for r in actual] == [r['raw_metric'] for r in expected]
    assert [r['trial'] for r in actual] == list(range(7))


def test_legacy_resume_recovers_diagnostics_and_declares_new_random_stream(space,tmp_path):
    with pytest.raises(RuntimeError,match='expert interruption'):
        run(space,tmp_path,Interrupt())
    (tmp_path/'controller_checkpoint.json').unlink()
    before=(tmp_path/'trials.jsonl').read_bytes()
    run(space,tmp_path,StructuredMockProposer(),resume=True)
    assert (tmp_path/'trials.jsonl').read_bytes().startswith(before)
    events=[json.loads(line) for line in (tmp_path/'round_diagnostics.jsonl').read_text().splitlines()]
    assert any(r.get('recovery_mode') == 'legacy_diagnostics_new_rng_stream' for r in events)
    assert len(rows(tmp_path)) == 7


def test_resume_refuses_corrupt_controller_or_inconsistent_observations(space,tmp_path):
    with pytest.raises(RuntimeError):
        run(space,tmp_path,Interrupt())
    checkpoint=tmp_path/'controller_checkpoint.json'
    data=json.loads(checkpoint.read_text());data['parameters']['eta']=.1
    checkpoint.write_text(json.dumps(data))
    before=(tmp_path/'trials.jsonl').read_bytes()
    with pytest.raises(ValueError,match='protocol differs'):
        run(space,tmp_path,StructuredMockProposer(),resume=True)
    assert (tmp_path/'trials.jsonl').read_bytes() == before
    knobs=tmp_path/'knobs.jsonl'
    lines=knobs.read_text().splitlines();lines.pop();knobs.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='traces disagree'):
        RunRecorder(tmp_path,resume=True)
    assert (tmp_path/'trials.jsonl').read_bytes() == before


def test_resume_never_deletes_an_orphan_knob_trace(tmp_path):
    trace=tmp_path/'knobs.jsonl'
    trace.write_text('{"trial":0}\n')
    before=trace.read_bytes()
    with pytest.raises(ValueError,match='without trial trace'):
        RunRecorder(tmp_path,resume=True)
    assert trace.read_bytes() == before
