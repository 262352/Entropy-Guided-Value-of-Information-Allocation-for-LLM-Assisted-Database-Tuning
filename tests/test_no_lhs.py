from pathlib import Path


def test_optimizer_sources_do_not_use_prohibited_initializers():
    root = Path(__file__).resolve().parents[1]
    prohibited = ("LatinHyper" + "cube", "Sobol" + "Engine", "Sobol" + "InitialDesign", "Hal" + "ton", "Ham" + "mersley")
    for path in list((root / "src/cognitune").rglob("*.py")) + list((root / "configs").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in prohibited), path
        if path.name != "initial_design.py":
            assert "l" + "hs" not in text.lower(), path

