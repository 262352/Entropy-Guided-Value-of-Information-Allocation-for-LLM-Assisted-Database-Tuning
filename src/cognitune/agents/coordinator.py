from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from cognitune.llm.base import LLMClient
from cognitune.llm.parser import StructuredResponseParser
from cognitune.search_space.space import SearchSpace
from .ensemble import build_prompt_ensemble


class AgentCoordinator:
    def __init__(self, space: SearchSpace, client: LLMClient, artifact_dir: str | Path):
        self.space = space
        self.client = client
        self.parser = StructuredResponseParser(space)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def collect(self, context: dict, experts=None, templates=None, tag: str = "") -> list:
        responses = []
        items = build_prompt_ensemble(self.space, context, experts=experts, templates=templates)
        for number, item in enumerate(items):
            prefix = f"{tag}_" if tag else ""
            prompt = item.prompt
            errors = []
            for attempt in range(1, 4):
                result = None
                try:
                    result = self.client.complete(prompt, expert=item.expert, template=item.template)
                    parsed = self.parser.parse(result.text)
                    if parsed.expert != item.expert or parsed.template != item.template:
                        raise ValueError("expert/template identity does not match the requested sample")
                    break
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    errors.append(str(error))
                    (self.artifact_dir / f"{prefix}{number:02d}_{item.expert}_{item.template}_invalid_{attempt}.json").write_text(
                        json.dumps({"error": str(error), "raw_response": result.text if result is not None else ""}, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    if attempt == 3:
                        raise ValueError(f"invalid LLM response after 3 attempts: {errors}") from error
                    payload = json.loads(item.prompt)
                    payload["validation_feedback"] = str(error)
                    prompt = json.dumps(payload, sort_keys=True)
            record = {
                "expert": item.expert, "template": item.template,
                "prompt": item.prompt, "raw_response": result.text,
                "parsed": json.loads(result.text), "model": result.model,
                "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                "cache_hit_tokens": result.cache_hit_tokens,
                "cache_miss_tokens": result.cache_miss_tokens,
                "request_id": result.request_id,
                "latency_seconds": result.latency_seconds,
                "normalization_repairs": list(result.normalization_repairs),
                "validation_attempts": attempt,
                "validation_errors": errors,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            (self.artifact_dir / f"{prefix}{number:02d}_{item.expert}_{item.template}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            responses.append(parsed)
        return responses
