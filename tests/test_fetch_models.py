import pytest
from datetime import datetime, timezone
import ask_llm.model_manager as mm
from types import SimpleNamespace

class DummyResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data
    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception(f"Status {self.status_code}")
    def json(self):
        return self._json

def test_fetch_ollama_api_models_success(monkeypatch):
    iso = '2020-01-01T12:00:00Z'
    monkeypatch.setattr(mm.requests, 'get', lambda url, timeout: DummyResponse(200, {'models':[{'name':'x','modified_at':iso}]}))
    ok, models = mm.fetch_ollama_api_models('http://url')
    assert ok
    assert models[0]['id'] == 'x'
    assert isinstance(models[0]['modified_at'], datetime)

def test_fetch_ollama_api_models_http_error(monkeypatch):
    monkeypatch.setattr(mm.requests, 'get', lambda url, timeout: (_ for _ in ()).throw(Exception('fail')))
    ok, models = mm.fetch_ollama_api_models('http://url')
    assert not ok
    assert models == []

def test_fetch_openai_api_models_init_error(monkeypatch):
    monkeypatch.setattr(mm, 'OpenAI', lambda *args, **kwargs: (_ for _ in ()).throw(Exception('init fail')))
    ok, models = mm.fetch_openai_api_models()
    assert not ok and models == []

class DummyModel:
    def __init__(self, id, owned_by, created):
        self.id = id
        self.owned_by = owned_by
        self.created = created

class DummyClient:
    def __init__(self):
        self.models = self
    def list(self, *args, **kwargs):
        if 'limit' in kwargs or (args and args[0] == 1):
            return None
        return SimpleNamespace(data=[DummyModel('gpt-x','other',1600000000), DummyModel('other','openai',1600000000)])

def test_fetch_openai_api_models_success(monkeypatch):
    monkeypatch.setattr(mm, 'OpenAI', lambda *args, **kwargs: DummyClient())
    ok, models = mm.fetch_openai_api_models()
    assert ok
    ids = [m['id'] for m in models]
    assert 'gpt-x' in ids and 'other' in ids

def test_fetch_openai_api_models_fetch_error(monkeypatch):
    class ClientErr:
        def __init__(self): self.models = self
        def list(self, *args, **kwargs):
            if 'limit' in kwargs or (args and args[0] == 1): return None
            raise Exception('fetch fail')
    monkeypatch.setattr(mm, 'OpenAI', lambda *args, **kwargs: ClientErr())
    ok, models = mm.fetch_openai_api_models()
    assert not ok
    assert models == []