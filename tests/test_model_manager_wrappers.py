import pytest
from types import SimpleNamespace
import ask_llm.model_manager as mm

def test_list_models_wrapper(monkeypatch):
    called = {}
    def fake_list(self): called['list'] = True
    monkeypatch.setattr(mm.ModelManager, 'list_available_models', fake_list)
    config = SimpleNamespace()
    mm.list_models(config)
    assert called.get('list')

def test_delete_model_wrapper(monkeypatch):
    called = {}
    fake_manager = SimpleNamespace(delete_model_alias=lambda alias: True)
    monkeypatch.setattr(mm, 'ModelManager', lambda config: fake_manager)
    config = SimpleNamespace(_load_models_config=lambda: called.setdefault('load', True), force_ollama_check=lambda: called.setdefault('force', True))
    res = mm.delete_model('alias', config)
    assert res is True
    assert called.get('load') and called.get('force')

def test_update_models_interactive_wrapper(monkeypatch):
    called = {}
    fake_mgr = SimpleNamespace(update_models=lambda p: called.setdefault('upd', p))
    monkeypatch.setattr(mm, 'ModelManager', lambda config: fake_mgr)
    config = SimpleNamespace()
    res = mm.update_models_interactive(config, provider='openai')
    assert res == 'openai'
    called.clear()
    res2 = mm.update_models_interactive(config)
    assert res2 is None