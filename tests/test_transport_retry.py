import io
import json
import ssl
import urllib.error
from types import SimpleNamespace

import pytest

from cognitune.llm.deepseek import DeepSeekLLMClient, LLMTransportError
from cognitune.llm.json_request import request_json_object


class Response:
    def __init__(self, data=b'{"ok":true}'):
        self.data=data
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self):
        return self.data


def install(monkeypatch, outcomes):
    calls, delays = [], []
    iterator=iter(outcomes)
    def urlopen(*args, **kwargs):
        calls.append(1)
        outcome=next(iterator)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr('cognitune.llm.deepseek.urllib.request.urlopen', urlopen)
    monkeypatch.setattr('cognitune.llm.deepseek.time.sleep', delays.append)
    return calls, delays


def test_wrapped_ssl_eof_recovers_with_backoff(monkeypatch):
    error=urllib.error.URLError(ssl.SSLEOFError('UNEXPECTED_EOF_WHILE_READING'))
    calls, delays=install(monkeypatch, [error, error, Response()])
    client=DeepSeekLLMClient(api_key='test-only')
    assert client._post({}) == {'ok':True}
    assert len(calls)==3 and delays==[10,20]


def test_busy_service_retries_then_succeeds(monkeypatch):
    error=urllib.error.HTTPError('https://example.invalid',503,'busy',{},io.BytesIO(b'busy'))
    calls, delays=install(monkeypatch,[error,Response()])
    assert DeepSeekLLMClient(api_key='test-only')._post({}) == {'ok':True}
    assert len(calls)==2 and delays==[10]


def test_bad_credentials_fail_immediately(monkeypatch):
    error=urllib.error.HTTPError('https://example.invalid',401,'unauthorized',{},io.BytesIO(b'invalid key'))
    calls, delays=install(monkeypatch,[error])
    with pytest.raises(LLMTransportError) as raised:
        DeepSeekLLMClient(api_key='test-only')._post({})
    assert raised.value.status_code==401 and not raised.value.retryable
    assert len(calls)==1 and delays==[]


def test_exhaustion_is_bounded_and_diagnostics_preserve_cache(tmp_path,monkeypatch):
    error=urllib.error.URLError('EOF')
    calls, delays=install(monkeypatch,[error]*6)
    client=DeepSeekLLMClient(api_key='test-only')
    (tmp_path/'existing.json').write_text('{"answer":{"score":5}}')
    with pytest.raises(LLMTransportError) as raised:
        request_json_object(client,'prompt',tmp_path)
    assert len(calls)==6 and delays==[10,20,40,60,60]
    assert raised.value.attempts==6
    diagnostic=json.loads(next((tmp_path/'attempts').glob('*/*.json')).read_text())
    assert diagnostic['retryable'] and diagnostic['transport_attempts']==6
    assert (tmp_path/'existing.json').read_text()=='{"answer":{"score":5}}'
    assert len(list(tmp_path.glob('*.json')))==1


def test_truncated_http_envelope_is_transport_failure_not_expert_json(monkeypatch):
    calls, delays=install(monkeypatch,[Response(b'{"choices":'), Response()])
    assert DeepSeekLLMClient(api_key='test-only')._post({}) == {'ok':True}
    assert len(calls)==2 and delays==[10]
