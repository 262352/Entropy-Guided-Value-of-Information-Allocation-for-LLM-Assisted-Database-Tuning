"""Finite-audit decision entropy. No scores are interpreted as true rewards."""
from __future__ import annotations

import numpy as np


def factor(covariance):
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    return vectors * np.sqrt(np.maximum(values, 0))


def label_probabilities(draws, sources, delta=0.01):
    """Column zero is the anchor; sources are sets for deduplicated configs.

    A tie is resolved by column order, which the caller fixes by config key.
    A shared proposal has ONE subset label, rather than several winner votes.
    """
    winner = np.argmax(draws[:, 1:], axis=1) + 1
    improved = draws[np.arange(len(draws)), winner] - draws[:, 0] > delta
    labels = [tuple(sorted(sources[i])) if ok else () for i, ok in zip(winner, improved)]
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {key: count / len(draws) for key, count in counts.items()}


def entropy(probabilities):
    values = np.asarray(list(probabilities.values()))
    return float(-np.sum(values * np.log(np.maximum(values, 1e-300))))


def condition(mean, covariance, index, observation, noise):
    variance = max(float(covariance[index, index] + noise), 1e-12)
    column = covariance[:, index]
    return (mean + column / variance * (observation - mean[index]),
            covariance - np.outer(column, column) / variance)


def information_gains(mean, covariance, noise, audit_count, sources, indices,
                      rng, delta=0.01, fantasies=16, samples=256):
    """Frozen audit, anchor, hyperparameters; noisy fantasies, latent labels.

    All audit points remain in the reference even if not in the shortlist.
    Common random numbers reduce variance of candidate comparisons. Returned
    standard errors measure outer Monte Carlo error, not GP calibration.
    """
    reference_mean = mean[:audit_count]
    reference_cov = covariance[:audit_count, :audit_count]
    z = rng.standard_normal((samples, audit_count))
    prior = label_probabilities(reference_mean + z @ factor(reference_cov).T, sources, delta)
    initial_entropy = entropy(prior)
    fantasy_z = rng.standard_normal(fantasies)
    estimates, errors = [], []
    for index in indices:
        variance = max(float(covariance[index, index] + noise), 1e-12)
        column = covariance[:audit_count, index]
        conditional_cov = reference_cov - np.outer(column, column) / variance
        residual = z @ factor(conditional_cov).T
        outcomes = []
        for normal in fantasy_z:
            conditional_mean = reference_mean + column / np.sqrt(variance) * normal
            probabilities = label_probabilities(conditional_mean + residual, sources, delta)
            outcomes.append(initial_entropy - entropy(probabilities))
        estimates.append(float(np.mean(outcomes)))
        errors.append(float(np.std(outcomes, ddof=1) / np.sqrt(fantasies)))
    return np.asarray(estimates), np.asarray(errors), prior


def allocate(probabilities, experts, total, minimum=4, temperature=1.0):
    if total < minimum * len(experts) or temperature <= 0:
        raise ValueError('infeasible candidate quota')
    credit = {expert: 0.0 for expert in experts}
    for subset, probability in probabilities.items():
        # Reference/joint labels do not masquerade as individual expert wins.
        for expert in subset:
            if expert in credit:
                credit[expert] += probability / len(subset)
    weights = np.array([(0.02 + credit[e]) ** (1 / temperature) for e in experts])
    weights /= weights.sum()
    remaining = total - minimum * len(experts)
    fractional = remaining * weights
    quota = np.floor(fractional).astype(int) + minimum
    for index in np.argsort(-(fractional - np.floor(fractional)), kind='stable')[:total - quota.sum()]:
        quota[index] += 1
    return dict(zip(experts, map(int, quota))), credit
