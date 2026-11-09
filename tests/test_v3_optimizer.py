import json
import numpy as np
import pytest

from cognitune.evaluator import SyntheticEvaluator
from cognitune.optimizers.cognitune_v3 import bounded_patch, run_cognitune_v3
from cognitune.search_space.initial_design import build_initial_design
from cognitune.surrogate.gp import GPPerformanceSurrogate


class FeedbackMock:
    def __init__(self):
        self.calls = []

    def __call__(self, expert, context):
        self.calls.append((expert, context))
        return {'regions': {k['name']: {'center': context['anchor'].get(k['name'], k['default']),
            'halfwidth': .1, 'action': 'hold'} for k in context['allowed_knobs']},
            'patches': [], 'joint_hypotheses': []}


def test_v3_real_gp_budget_feedback_audit_and_last_trial(space, tmp_path):
    initial = build_initial_design(space, 0, 3)['configs']
    expert = FeedbackMock()
    summary = run_cognitune_v3(space, SyntheticEvaluator(space, 0, 'throughput', 1000, .01),
        initial, 6, 0, 40, tmp_path/'v3', proposer=expert,
        audit_size=2, posterior_samples=64, fantasies=4, active_target=4, probe_interval=1)
    assert summary['trials'] == 6
    assert len(expert.calls) == 4 * 3
    assert {c['trial'] for _, c in expert.calls} == {3, 4, 5}
    events = [json.loads(line) for line in (tmp_path/'v3/round_diagnostics.jsonl').read_text().splitlines()]
    proposals = [r for r in events if r['event'] == 'proposal']
    assert proposals[-1]['selection_reason'] == 'qLogNEI'
    assert any(r['selection_reason'] == 'reserved_reentry' for r in proposals[:-1])
    assert all(sum(r['candidate_quota'].values()) == 20 for r in proposals)
    for row in proposals:
        assert len({len(a) for a in row['audit_configs'].values()}) == 1
        assert sum(row['label_probabilities'].values()) == pytest.approx(1)
    with pytest.raises(FileExistsError):
        run_cognitune_v3(space, None, initial, 6, 0, 40, tmp_path/'v3', proposer=expert)


def test_gp_joint_covariance_noise_and_qlognei(space):
    configs = space.sample_iid(10, 5)
    gp = GPPerformanceSurrogate(space)
    gp.fit(configs, [0, .2, -.1, .5, .1])
    mean, cov, noise = gp.joint_posterior(configs[:2])
    assert mean.shape == (2,) and cov.shape == (2, 2) and noise > 0
    assert abs(cov[0, 1]) > 0
    assert np.linalg.eigvalsh(cov).min() >= -1e-8
    scores = gp.log_noisy_expected_improvement(configs[2:], samples=32)
    assert scores.shape == (3,) and np.isfinite(scores).all()
    np.testing.assert_allclose(scores, gp.log_noisy_expected_improvement(configs[2:], samples=32))


def test_sparse_patch_bounds_and_bool_validation(space):
    anchor = space.get_default()
    assert bounded_patch(space, anchor, {'synchronous_writes':'off'}, space.by_name) is None
    assert bounded_patch(space, anchor, {'cache_mb':8192}, space.by_name, distance=.01) is None
    assert bounded_patch(space, anchor, {'alien':1}, space.by_name) is None


def test_valid_boolean_flip_is_reachable_in_local_channel(space):
    anchor = space.get_default()
    candidate = bounded_patch(space, anchor, {'synchronous_writes':not anchor['synchronous_writes']}, space.by_name)
    assert candidate is not None and candidate['synchronous_writes'] != anchor['synchronous_writes']
