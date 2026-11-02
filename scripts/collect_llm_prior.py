#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cognitune.agents import AgentCoordinator
from cognitune.config import load_search_space
from cognitune.llm import DeepSeekLLMClient
from cognitune.llm import StructuredResponseParser
from cognitune.uncertainty import priority_statistics, relation_statistics, value_statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="configs/knobs/postgresql_60_live.yaml")
    parser.add_argument("--profile", default="artifacts/profiles/postgresql_tpcc.json")
    parser.add_argument("--output", default="artifacts/llm/deepseek_tpcc")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    args = parser.parse_args()
    space = load_search_space(args.space)
    context = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    output = Path(args.output)
    response_dir = output / "responses"
    valid_paths = sorted(path for path in response_dir.glob("*.json") if "_invalid_" not in path.name)
    if len(valid_paths) == 15:
        parser_instance = StructuredResponseParser(space)
        responses = [parser_instance.parse(json.loads(path.read_text())["raw_response"]) for path in valid_paths]
    else:
        responses = AgentCoordinator(
            space, DeepSeekLLMClient(model=args.model, base_url=args.base_url), response_dir,
        ).collect(context)
    priority = priority_statistics(responses, list(space.by_name))
    values = value_statistics(responses, space.knobs)
    relations = relation_statistics(responses, space.by_name)
    records = [json.loads(path.read_text()) for path in valid_paths or response_dir.glob("*.json") if "_invalid_" not in path.name]
    report = {
        "model": args.model,
        "samples": len(responses),
        "priority": priority,
        "value": values,
        "relations": relations,
        "usage": {
            "input_tokens": sum(item["input_tokens"] for item in records),
            "output_tokens": sum(item["output_tokens"] for item in records),
            "cache_hit_tokens": sum(item.get("cache_hit_tokens", 0) for item in records),
            "cache_miss_tokens": sum(item.get("cache_miss_tokens", 0) for item in records),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "prior_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["usage"], sort_keys=True))


if __name__ == "__main__":
    main()
