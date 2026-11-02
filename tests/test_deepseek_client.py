import json

import pytest

from cognitune.agents import AgentCoordinator
from cognitune.llm.base import LLMResult
from cognitune.llm.deepseek import DeepSeekLLMClient
from cognitune.llm.mock import MockLLMClient


COMMON = {"context": {"hardware": "fixed"}, "knobs": [{"name": "x", "type": "integer", "lower": 0, "upper": 10, "default": 5, "choices": []}]}
VALID_PAYLOAD = {
    "priority_by_knob": {"x": "high"},
    "recommendations": {"x": {"recommended_values": [5], "recommended_interval": [1, 9], "confidence": 0.8, "reason": "test"}},
    "conditional_relations": [],
}


def _complete_with(content: str, prompt: dict | None = None) -> LLMResult:
    client = DeepSeekLLMClient(api_key="test")
    client._post = lambda payload: {
        "id": "r", "model": "deepseek-flash",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "choices": [{"message": {"content": content}}],
    }
    return client.complete(json.dumps(prompt or COMMON), expert="memory_expert", template="mechanism_template")


def test_deepseek_keeps_common_prefix_stable_and_normalizes_output():
    captured = []
    client = DeepSeekLLMClient(api_key="test")

    def fake_post(payload):
        captured.append(payload)
        suffix = json.loads(payload["messages"][1]["content"])
        content = {
            "priority_by_knob": {"x": "high"},
            "recommendations": {"x": {"recommended_values": [99], "recommended_interval": [-1, 99], "confidence": 1, "reason": "test"}},
            "conditional_relations": [],
        }
        return {"id": "r", "model": "deepseek-flash", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "choices": [{"message": {"content": json.dumps(content)}}]}

    client._post = fake_post
    common = {"context": {"hardware": "fixed"}, "knobs": [{"name": "x", "type": "integer", "lower": 0, "upper": 10, "default": 5, "choices": []}]}
    prompt = json.dumps(common)
    first = client.complete(prompt, expert="memory_expert", template="mechanism_template")
    second = client.complete(prompt, expert="optimizer_expert", template="resource_template")
    assert captured[0]["messages"][0] == captured[1]["messages"][0]
    payload = json.loads(first.text)
    assert payload["priority"] == {"high": ["x"], "medium": [], "low": []}
    assert payload["recommendations"]["x"]["recommended_values"] == [10]
    assert first.normalization_repairs
    assert json.loads(second.text)["expert"] == "optimizer_expert"


def test_duplicate_json_objects_are_tolerated():
    result = _complete_with(json.dumps(VALID_PAYLOAD) + json.dumps(VALID_PAYLOAD))
    payload = json.loads(result.text)
    assert payload["priority"] == {"high": ["x"], "medium": [], "low": []}
    assert any("concatenated JSON objects" in repair for repair in result.normalization_repairs)


def test_trailing_characters_after_json_are_discarded():
    result = _complete_with(json.dumps(VALID_PAYLOAD) + '}\n{"note": "stop"')
    payload = json.loads(result.text)
    assert payload["recommendations"]["x"]["recommended_values"] == [5]
    assert any("trailing characters" in repair for repair in result.normalization_repairs)


def test_markdown_fence_and_leading_prose_are_stripped():
    content = "Here is the JSON:\n```json\n" + json.dumps(VALID_PAYLOAD) + "\n```"
    result = _complete_with(content)
    assert json.loads(result.text)["priority"]["high"] == ["x"]
    repairs = " ".join(result.normalization_repairs)
    assert "code fence" in repairs and "leading characters" in repairs


def test_most_complete_object_wins_when_objects_are_concatenated():
    partial = {"expert": "memory_expert", "recommendations": {}}
    result = _complete_with(json.dumps(partial) + json.dumps(VALID_PAYLOAD))
    payload = json.loads(result.text)
    assert payload["priority"]["high"] == ["x"]
    assert payload["recommendations"]["x"]["recommended_values"] == [5]


def test_unparseable_response_raises_retryable_value_error():
    with pytest.raises(ValueError, match="not valid JSON"):
        _complete_with('{"priority_by_knob": ')


def test_coordinator_retries_when_client_rejects_malformed_content(space, tmp_path):
    class FlakyClient(MockLLMClient):
        def __init__(self, inner_space):
            super().__init__(inner_space)
            self.calls = 0

        def complete(self, prompt, *, expert, template):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("DeepSeek response is not valid JSON: Extra data")
            return super().complete(prompt, expert=expert, template=template)

    client = FlakyClient(space)
    coordinator = AgentCoordinator(space, client, tmp_path)
    responses = coordinator.collect({"seed": 0})
    assert len(responses) == 15
    assert client.calls == 16
    invalid = list(tmp_path.glob("*_invalid_1.json"))
    assert len(invalid) == 1
    assert "Extra data" in json.loads(invalid[0].read_text())["error"]
