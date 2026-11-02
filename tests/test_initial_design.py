from cognitune.search_space.initial_design import build_initial_design, load_initial_design, save_initial_design


def test_persisted_initial_design_is_unique_and_reproducible(space, tmp_path):
    design = build_initial_design(space, 7, 5)
    path = tmp_path / "design.json"
    save_initial_design(path, design)
    loaded = load_initial_design(path, space, 5, 7)
    assert loaded[0] == space.get_default()
    assert loaded == build_initial_design(space, 7, 5)["configs"]
    assert len({space.config_key(item) for item in loaded}) == 5

