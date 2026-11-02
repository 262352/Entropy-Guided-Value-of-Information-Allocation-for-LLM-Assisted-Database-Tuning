import numpy as np


def test_encode_decode_round_trip_and_one_hot(space):
    config = space.sample_iid(11, 1)[0]
    vector = space.encode(config)
    decoded = space.decode(vector)
    assert decoded == config
    assert vector.shape == (space.encoder.dimension,)
    assert set(np.unique(vector)).issuperset({0.0, 1.0})


def test_log_sampling_integer_and_repair(space):
    configs = space.sample_iid(2, 40)
    assert all(isinstance(item["cache_mb"], int) for item in configs)
    repaired = space.repair({**space.get_default(), "cache_mb": 999999, "parallel_degree": 4})
    assert repaired["cache_mb"] == 8192
    assert space.validate(repaired)


def test_conditional_and_forbidden(space):
    disabled = space.repair({**space.get_default(), "parallel_enabled": False, "parallel_degree": 12})
    assert "parallel_degree" not in disabled
    forbidden = {**space.get_default(), "synchronous_writes": False, "io_depth": 1}
    repaired = space.repair(forbidden)
    assert space.validate(repaired)
    assert repaired != forbidden

