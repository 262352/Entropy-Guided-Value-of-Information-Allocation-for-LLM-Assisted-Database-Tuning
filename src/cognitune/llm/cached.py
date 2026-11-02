from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .base import LLMClient, LLMResult


class CachedLLMClient(LLMClient):
    def __init__(self, inner: LLMClient, cache_dir: str | Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def complete(self, prompt: str, *, expert: str, template: str) -> LLMResult:
        key = hashlib.sha256(json.dumps({
            "client": type(self.inner).__name__, "model": getattr(self.inner, "model", ""),
            "prompt": prompt, "expert": expert, "template": template,
        }, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return LLMResult(**json.loads(path.read_text(encoding="utf-8")))
        result = self.inner.complete(prompt, expert=expert, template=template)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result.__dict__, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return result
