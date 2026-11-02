from pathlib import Path

import pytest

from cognitune.config import load_search_space


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def space():
    return load_search_space(ROOT / "configs/knobs/synthetic.yaml")

