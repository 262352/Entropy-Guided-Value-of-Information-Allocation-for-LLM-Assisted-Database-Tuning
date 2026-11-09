import hashlib
import json
from types import SimpleNamespace

import pytest

from cognitune.llm.json_request import request_json_object


def response(content, finish='stop'):
    return {'model':'test', 'choices':[{'message':{'content':content},'finish_reason':finish}],
            'usage':{'completion_tokens':12}, 'id':'test-response'}


def fake_transport(responses):
    iterator = iter(responses)
    calls = []
    def post(payload):
        calls.append(payload)
        return next(iterator)
    return SimpleNamespace(model='test', _post=post), calls


def test_unterminated_string_retries_records_raw_usage_and_reuses_success(tmp_path):
    broken = '{"reason":"unfinished'
    transport, calls = fake_transport([response(broken), response('{"score":75,"reason":"valid"}')])
    answer = request_json_object(transport, 'original expert prompt', tmp_path)
    assert answer['score'] == 75
    assert len(calls) == 2
    assert [c['max_tokens'] for c in calls] == [4096,8192]
    assert all(c['messages'][1]['content'] == 'original expert prompt' for c in calls)
    assert len(calls[1]['messages']) == 3
    attempts = [json.loads(p.read_text()) for p in (tmp_path/'attempts').glob('*/*.json')]
    assert len(attempts) == 2
    bad = next(r for r in attempts if 'parse_error' in r)
    assert bad['response']['choices'][0]['message']['content'] == broken
    assert bad['response']['usage']['completion_tokens'] == 12
    assert request_json_object(transport, 'original expert prompt', tmp_path) == answer
    assert len(calls) == 2


def test_length_finish_reason_is_retried_even_if_prefix_is_parseable(tmp_path):
    transport, calls = fake_transport([response('{"score":99}', 'length'), response('{"score":50}')])
    assert request_json_object(transport, 'p', tmp_path) == {'score':50}
    assert len(calls) == 2


def test_retry_limit_preserves_failures_and_never_caches_fabricated_recommendation(tmp_path):
    transport, calls = fake_transport([response(''), response('{"'), response('[]')])
    with pytest.raises(RuntimeError, match='invalid JSON after 3'):
        request_json_object(transport, 'p', tmp_path)
    assert len(calls) == 3
    assert [c['max_tokens'] for c in calls] == [4096,8192,16384]
    assert not list(tmp_path.glob('*.json'))
    assert len(list((tmp_path/'attempts').glob('*/*.json'))) == 3


def test_existing_cache_key_is_unchanged_and_no_api_request_is_made(tmp_path):
    prompt='existing prompt'
    key=hashlib.sha256(('test'+prompt).encode()).hexdigest()
    (tmp_path/(key+'.json')).write_text(json.dumps({'answer':{'score':42}}))
    transport, calls=fake_transport([])
    assert request_json_object(transport,prompt,tmp_path) == {'score':42}
    assert calls == []


def test_fenced_object_normalization_is_logged(tmp_path):
    transport, calls=fake_transport([response('```json\n{"score":62}\n```')])
    assert request_json_object(transport, 'p', tmp_path)['score'] == 62
    cached=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert cached['normalization_repairs'] == ['stripped markdown code fence']
