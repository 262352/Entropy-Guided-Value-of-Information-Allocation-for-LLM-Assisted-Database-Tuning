import numpy as np

from cognitune.surrogate import GPPerformanceSurrogate


def test_gp_fits_and_returns_expected_shapes(space):
    configs = space.sample_iid(4, 7)
    utilities = [float(np.sin(space.encode(item).sum())) for item in configs]
    model = GPPerformanceSurrogate(space)
    model.fit(configs, utilities)
    prediction = model.posterior(configs[:3])
    assert prediction.mean.shape == (3,)
    assert prediction.std.shape == (3,)
    assert model.log_ei(configs[:3]).shape == (3,)


def test_both_optimizers_import_the_same_gp_class():
    from cognitune.optimizers import cognitune, gp_bo
    assert cognitune.GPPerformanceSurrogate is gp_bo.GPPerformanceSurrogate

