from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .base import LLMClient, LLMResult


class DeepSeekLLMClient(LLMClient):
    """OpenAI-compatible DeepSeek client with a stable cache-friendly prefix."""

    def __init__(self, api_key: str | None = None, model: str = "deepseek-flash", base_url: str = "https://api.deepseek.com", timeout_seconds: int = 300, max_tokens: int = 16_000):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for the DeepSeek provider")
        self.model = model
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def complete(self, prompt: str, *, expert: str, template: str) -> LLMResult:
        parsed = json.loads(prompt)
        common = {"context": parsed["context"], "knobs": parsed["knobs"]}
        suffix = {
            "expert": expert, "template": template,
            "focus": _expert_focus(expert), "viewpoint": _template_focus(template),
        }
        if parsed.get("validation_feedback"):
            suffix["validation_feedback"] = parsed["validation_feedback"]
            suffix["correction_instruction"] = "Regenerate the complete JSON object and correct every listed validation error."
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _system_prefix(common)},
                {"role": "user", "content": json.dumps(suffix, sort_keys=True, separators=(",", ":"))},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": self.max_tokens,
        }
        started = time.monotonic()
        response = self._post(payload)
        content = response["choices"][0]["message"].get("content") or ""
        if not content.strip():
            raise RuntimeError("DeepSeek returned empty JSON content")
        content, repairs = _normalize_response(content, common["knobs"], expert, template)
        usage = response.get("usage", {})
        details = usage.get("prompt_tokens_details") or {}
        return LLMResult(
            text=content, model=str(response.get("model", self.model)),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            cache_hit_tokens=int(usage.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0))),
            cache_miss_tokens=int(usage.get("prompt_cache_miss_tokens", 0)),
            request_id=str(response.get("id", "")), latency_seconds=time.monotonic() - started,
            normalization_repairs=tuple(repairs),
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "User-Agent": "CogniTune/0.1"},
            method="POST",
        )
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:1000]
                last_error = RuntimeError(f"DeepSeek HTTP {error.code}: {detail}")
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    raise last_error from None
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = RuntimeError(f"DeepSeek request failed: {error}")
            time.sleep(2 ** attempt)
        raise last_error or RuntimeError("DeepSeek request failed")


def _system_prefix(common: dict[str, Any]) -> str:
    names = [item["name"] for item in common["knobs"]]
    schema_example = {
        "expert": "memory_expert", "template": "mechanism_template",
        "priority_by_knob": {"knob_a": "high", "knob_b": "medium", "knob_c": "low"},
        "recommendations": {"knob_a": {"recommended_values": [1], "recommended_interval": [1, 2], "confidence": 0.8, "reason": "mechanism and workload rationale"}},
        "conditional_relations": [{"source": "knob_a", "target": "knob_b", "reason": "resource interaction", "branches": [{"source_interval": [1, 2], "target_values": [3]}, {"source_interval": [3, 4], "target_values": [2]}]}],
    }
    return (
        "You are one member of a database tuning expert ensemble. Return exactly one JSON object and no markdown. "
        "Use only supplied facts; distinguish measured facts from protocol assumptions. Every knob must occur exactly once across high/medium/low and must have one recommendation. "
        "Numeric values and interval endpoints must be inside bounds; categorical and boolean values must be legal. Use at least two executable branches for a conditional relation or omit it. "
        "Scores must reflect this host and workload, not generic importance. "
        f"MANDATORY COMPLETENESS: there are exactly {len(names)} knobs. priority_by_knob must contain exactly {len(names)} keys, recommendations must contain exactly {len(names)} keys, and both key sets must equal EXPECTED_KNOB_NAMES. "
        "Every priority_by_knob value must be exactly high, medium, or low. Never return priority arrays, never return only important knobs, never abbreviate with ellipses, and include expert and template exactly as requested in the user message. recommended_interval is allowed only for numeric knobs. "
        "Before responding, silently verify all names and counts. EXPECTED_KNOB_NAMES="
        + json.dumps(names, separators=(",", ":"))
        + ". Required JSON shape: "
        + json.dumps(schema_example, sort_keys=True, separators=(",", ":"))
        + "\nIMMUTABLE_DATABASE_CONTEXT_JSON:\n"
        + json.dumps(common, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _expert_focus(expert: str) -> str:
    return {
        "memory_expert": "memory budgets, cache residency, per-session amplification, NUMA",
        "concurrency_expert": "connections, locks, serialization failures, workers, contention",
        "optimizer_expert": "cost model, cardinality, access paths, join planning, JIT",
        "execution_expert": "parallel execution, CPU scheduling, temporary work, operator behavior",
        "storage_wal_expert": "WAL, checkpoints, durability, background writing, physical IO",
        "concurrency_execution_expert": "connections, locks, serialization, workers, parallelism, contention",
        "optimizer_workload_expert": "cost model, cardinality, access paths, join planning, statistics, workload mix",
        "workload_expert": "measured workload mix, transaction weights, concurrency and database statistics",
        "resource_expert": "hard host limits, memory envelope, storage topology and resource competition",
    }[expert]


def _template_focus(template: str) -> str:
    return {
        "mechanism_template": "reason from PostgreSQL 14.9 mechanisms and parameter interactions",
        "workload_template": "reason from measured workload mix, scale, concurrency and database statistics",
        "resource_template": "reason from hard host limits, storage topology and resource competition",
        "role_template": "reason from database mechanisms, the measured workload and hard resource limits",
        "interaction_template": "focus on conditional interactions and joint effects between knobs",
    }[template]


def _parse_json_object(content: str) -> tuple[dict[str, Any], list[str]]:
    """Extract the best JSON object from a model response.

    Models occasionally append a duplicate object, a closing brace, markdown
    fences or prose after an otherwise complete object. ``json.loads`` rejects
    such responses with "Extra data", so decode every concatenated value and
    keep the most complete object instead of failing the whole run.
    """
    repairs: list[str] = []
    text = content.lstrip("\ufeff").strip()
    if text.startswith("```"):
        text = _strip_fence(text)
        repairs.append("stripped markdown code fence")
    elif text.endswith("```"):
        text = text[:-3].rstrip()
        repairs.append("stripped trailing markdown code fence")
    start = text.find("{")
    if start < 0:
        raise ValueError(f"DeepSeek response contained no JSON object (preview={_preview(text)})")
    if start:
        repairs.append(f"discarded {start} leading characters before the JSON object")
        text = text[start:]

    decoder = json.JSONDecoder(strict=False)
    objects: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            remainder = text[position:]
            if not objects:
                raise ValueError(
                    f"DeepSeek response is not valid JSON at char {position + error.pos}: "
                    f"{error.msg} (preview={_preview(remainder)})"
                ) from error
            repairs.append(f"discarded {len(remainder)} trailing characters after the JSON object")
            break
        if isinstance(value, dict):
            objects.append(value)
        else:
            repairs.append(f"discarded trailing JSON {type(value).__name__} value")
        position = end
    if not objects:
        raise ValueError(f"DeepSeek response contained no JSON object (preview={_preview(text)})")
    if len(objects) == 1:
        return objects[0], repairs
    repairs.append(f"kept the most complete of {len(objects)} concatenated JSON objects")
    return max(objects, key=_completeness), repairs


def _strip_fence(text: str) -> str:
    lines = text.splitlines()[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


def _preview(text: str, limit: int = 160) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def _completeness(value: dict[str, Any]) -> tuple[int, int, int]:
    recommendations = value.get("recommendations")
    priority = value.get("priority_by_knob") or value.get("priority")
    return (
        len(recommendations) if isinstance(recommendations, dict) else 0,
        len(priority) if isinstance(priority, dict) else 0,
        len(value),
    )


def _normalize_response(content: str, knobs: list[dict[str, Any]], expert: str, template: str) -> tuple[str, list[str]]:
    payload, repairs = _parse_json_object(content)
    mapping = payload.pop("priority_by_knob", None)
    if mapping is not None:
        priority = {"high": [], "medium": [], "low": []}
        for name, level in mapping.items():
            if level not in priority:
                raise ValueError(f"DeepSeek returned invalid priority {level!r} for {name}")
            priority[level].append(name)
        payload["priority"] = priority
    if payload.get("expert") != expert:
        repairs.append(f"expert identity set to requested {expert}")
        payload["expert"] = expert
    if payload.get("template") != template:
        repairs.append(f"template identity set to requested {template}")
        payload["template"] = template
    by_name = {item["name"]: item for item in knobs}
    for name, recommendation in payload.get("recommendations", {}).items():
        if name in by_name:
            _repair_recommendation(name, recommendation, by_name[name], repairs)
    valid_relations = []
    for relation in payload.get("conditional_relations", []):
        source, target = relation.get("source"), relation.get("target")
        if source not in by_name or target not in by_name or source == target:
            repairs.append(f"dropped relation with invalid endpoints {source}->{target}")
            continue
        valid = True
        for branch in relation.get("branches") or []:
            try:
                _repair_branch_value(source, branch, "source_values", "source_interval", by_name[source], repairs)
                _repair_branch_value(target, branch, "target_values", "target_interval", by_name[target], repairs)
            except ValueError as error:
                repairs.append(f"dropped relation {source}->{target}: {error}")
                valid = False
                break
        if valid:
            valid_relations.append(relation)
    payload["conditional_relations"] = valid_relations
    return json.dumps(payload, ensure_ascii=False, sort_keys=True), repairs


def _repair_recommendation(name: str, item: dict[str, Any], knob: dict[str, Any], repairs: list[str]) -> None:
    item["recommended_values"] = [_legal_value(name, value, knob, repairs) for value in item.get("recommended_values", [])]
    interval = item.get("recommended_interval")
    if knob["type"] not in {"integer", "float"}:
        if interval is not None:
            repairs.append(f"removed nonnumeric interval for {name}")
            item.pop("recommended_interval", None)
    elif interval is not None:
        values = sorted(_legal_value(name, value, knob, repairs) for value in interval[:2])
        item["recommended_interval"] = values


def _repair_branch_value(name: str, branch: dict[str, Any], values_key: str, interval_key: str, knob: dict[str, Any], repairs: list[str]) -> None:
    if values_key in branch:
        branch[values_key] = [_legal_value(name, value, knob, repairs) for value in branch[values_key]]
    interval = branch.get(interval_key)
    if interval is not None:
        if knob["type"] not in {"integer", "float"}:
            raise ValueError(f"nonnumeric interval for {name}")
        branch[interval_key] = sorted(_legal_value(name, value, knob, repairs) for value in interval[:2])


def _legal_value(name: str, value: Any, knob: dict[str, Any], repairs: list[str]) -> Any:
    kind = knob["type"]
    if kind in {"integer", "float"}:
        legal = min(max(float(value), float(knob["lower"])), float(knob["upper"]))
        legal = int(round(legal)) if kind == "integer" else legal
        if legal != value:
            repairs.append(f"clamped {name} from {value!r} to {legal!r}")
        return legal
    choices = [False, True] if kind == "bool" else knob["choices"]
    if value not in choices:
        repairs.append(f"replaced illegal {name} value {value!r} with default {knob['default']!r}")
        return knob["default"]
    return value
