"""Cached JSON-object transport for upstream adapters.

Formatting retries never substitute a recommendation or modify expert ranking.
Every received response is saved before parsing, including truncated responses.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import time
import uuid

from .deepseek import LLMTransportError, _parse_json_object


def request_json_object(transport, prompt: str, cache_dir, max_retries: int = 3) -> dict:
    if max_retries < 1:
        raise ValueError('max_retries must be positive')
    model = transport.model
    # Preserve the existing MCTuner response cache keys when resuming.
    fingerprint = hashlib.sha256((model + prompt).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (fingerprint + '.json')
    if path.exists():
        answer = json.loads(path.read_text())['answer']
        if not isinstance(answer, dict) or not answer:
            raise ValueError('Invalid cached JSON object: ' + str(path))
        return answer

    audit_dir = cache_dir / 'attempts' / fingerprint
    audit_dir.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    attempts = []
    feedback = None
    started = time.monotonic()
    for attempt in range(max_retries):
        max_tokens = min(16384, 4096 * 2 ** min(attempt, 2))
        messages = [
            {'role': 'system', 'content': 'You are a database expert. Return one strict JSON object.'},
            {'role': 'user', 'content': prompt},
        ]
        if feedback:
            messages.append({'role': 'user', 'content': (
                'Your previous response was not a complete valid JSON object. '
                'Regenerate the entire object for the original task. Keep reasoning concise, '
                'close every string and brace, and do not add markdown or extra objects. '
                'Formatting error: ' + feedback
            )})
        payload = {'model': model, 'messages': messages,
                   'response_format': {'type': 'json_object'}, 'thinking': {'type': 'disabled'},
                   'max_tokens': max_tokens, 'temperature': 0.2}
        attempt_path = audit_dir / f'{request_id}_{attempt + 1:02d}.json'
        try:
            response = transport._post(payload)  # Bounded HTTP retries, shared by both methods.
        except LLMTransportError as error:
            # Do not wrap another retry loop around the exhausted transport.
            # Authentication/permission errors must also stop immediately.
            attempt_path.write_text(json.dumps({'prompt':prompt, 'transport_error':str(error),
                'retryable':error.retryable, 'status_code':error.status_code,
                'transport_attempts':error.attempts}, ensure_ascii=False, indent=2))
            logging.error('MCTuner transport failed; diagnostics saved at %s; completed caches preserved', attempt_path)
            raise
        entry = {'attempt': attempt + 1, 'max_tokens': max_tokens,
                 'prompt': prompt, 'response': response}
        # Persist first: if parsing fails we still have finish_reason, tokens,
        # request ID and exact content needed to diagnose the service response.
        attempt_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2))
        attempts.append(str(attempt_path))
        try:
            choice = response['choices'][0]
            if choice.get('finish_reason') == 'length':
                raise ValueError('finish_reason=length: response hit the output token limit')
            content = choice['message'].get('content') or ''
            if not isinstance(content, str):
                raise ValueError('response content is not text')
            answer, repairs = _parse_json_object(content)
            if not answer:
                raise ValueError('empty JSON object')
        except (ValueError, TypeError, KeyError, IndexError) as error:
            # Keep provider content in the audit file, not in terminal errors.
            feedback = ('finish_reason=length' if 'finish_reason=length' in str(error)
                        else 'response was empty, malformed, or not a JSON object')
            entry['parse_error'] = str(error)
            attempt_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2))
            logging.warning('MCTuner JSON response invalid (%s/%s, max_tokens=%s); saved at %s',
                            attempt + 1, max_retries, max_tokens, attempt_path)
            continue
        record = {'answer': answer, 'model': response.get('model'), 'usage': response.get('usage'),
                  'elapsed_seconds': time.monotonic() - started, 'prompt': prompt,
                  'normalization_repairs': repairs, 'attempts': attempts}
        # Only complete successful responses enter the reusable cache.
        temporary = path.with_suffix('.' + request_id + '.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        temporary.replace(path)
        return answer
    raise RuntimeError(f'MCTuner returned invalid JSON after {max_retries} responses; '
                       f'raw responses saved in {audit_dir}. Completed caches are preserved; '
                       'fix the service response or rerun the same command to resume preparation.')
