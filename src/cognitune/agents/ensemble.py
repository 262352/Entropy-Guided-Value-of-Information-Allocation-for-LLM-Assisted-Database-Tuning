from __future__ import annotations

import json

from cognitune.search_space.space import SearchSpace
from .expert import EXPERTS, TEMPLATES, ExpertPrompt
from .ownership import build_ownership


def build_prompt_ensemble(
    space: SearchSpace,
    context: dict,
    experts: tuple[str, ...] | None = None,
    templates: tuple[str, ...] | None = None,
) -> list[ExpertPrompt]:
    experts = experts or EXPERTS
    templates = templates or TEMPLATES
    ownership = build_ownership(space)
    knob_context = [
        {"name": k.name, "type": k.kind, "lower": k.lower, "upper": k.upper,
         "choices": list(k.choices), "default": k.default, "category": k.category,
         "description": k.description}
        for k in space.knobs
    ]
    prompts = []
    for expert in experts:
        assigned = sorted(name for name, weights in ownership.items() if weights.get(expert, 0.0) > 0)
        for template in templates:
            prompt = json.dumps({
                "instruction": "Return one strict JSON object covering every knob exactly once in priority. "
                               "Focus on your assigned knobs but you may comment on cross-group relations.",
                "expert": expert, "template": template,
                "assigned_knobs": assigned, "context": context,
                "knobs": knob_context,
            }, sort_keys=True)
            prompts.append(ExpertPrompt(expert, template, prompt))
    return prompts


def build_expert_prompt(space: SearchSpace, context: dict, expert: str, template: str) -> ExpertPrompt:
    return build_prompt_ensemble(space, context, (expert,), (template,))[0]
