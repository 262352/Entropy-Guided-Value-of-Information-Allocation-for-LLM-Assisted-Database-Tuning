import numpy as np
import pytest

from cognitune.adaptive.decision import allocate, condition, entropy, information_gains, label_probabilities


def test_subset_labels_are_exclusive_and_shared_source_credit_is_split():
    draws = np.array([[0, 2, 1], [0, -1, -2], [0, 1, 3], [0, 2, 2]])
    sources = [set(), {'a', 'b'}, {'c'}]
    p = label_probabilities(draws, sources, delta=.1)
    assert p == {('a', 'b'): .5, (): .25, ('c',): .25}
    assert sum(p.values()) == 1
    quotas, credit = allocate(p, ['a', 'b', 'c'], 100)
    assert credit == {'a': .25, 'b': .25, 'c': .25}
    assert sum(quotas.values()) == 100 and min(quotas.values()) >= 4


def test_no_gain_is_uniform_and_integer_allocation_is_exact():
    q, _ = allocate({(): 1}, ['a', 'b', 'c', 'd'], 80)
    assert q == dict.fromkeys(['a','b','c','d'], 20)
    q, _ = allocate({('a',): .99, (): .01}, ['a','b','c','d'], 81)
    assert sum(q.values()) == 81
    assert q['a'] > q['b'] >= 4
    with pytest.raises(ValueError):
        allocate({}, ['a', 'b'], 3)


def test_noisy_conditioning_uses_noise_and_preserves_psd():
    mean = np.zeros(2)
    cov = np.array([[1, .8], [.8, 1]])
    m, c = condition(mean, cov, 0, 2, 3)
    np.testing.assert_allclose(m, [.5, .4])
    np.testing.assert_allclose(c, cov - np.outer(cov[:, 0], cov[:, 0]) / 4)
    assert np.linalg.eigvalsh(c).min() >= 0


def test_information_uses_all_audits_and_independent_probe_has_zero_information():
    # Last coordinate is an execution candidate, independent of the complete
    # anchor + two-source audit. It must not acquire information about labels.
    mean = np.zeros(4)
    cov = np.eye(4)
    gains, errors, prior = information_gains(mean, cov, .1, 3,
        [set(), {'a'}, {'b'}], [3, 1], np.random.default_rng(11),
        fantasies=64, samples=2048)
    assert gains[0] == pytest.approx(0, abs=1e-12)
    assert errors[0] == pytest.approx(0, abs=1e-12)
    assert gains[1] > .1
    assert ('b',) in prior and entropy(prior) > 0


def test_correlated_audit_draws_do_not_turn_identical_latent_values_into_votes():
    z = np.random.default_rng(0).normal(size=1000)
    draws = np.stack([np.zeros(1000), z, z], axis=1)
    p = label_probabilities(draws, [set(), {'a'}, {'b'}])
    assert ('b',) not in p  # fixed tie break
