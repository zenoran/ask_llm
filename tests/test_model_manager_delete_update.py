import pytest
from types import SimpleNamespace
import ask_llm.model_manager as mm

@pytest.fixture(autouse=True)
def patch_console(monkeypatch):
    printed = []
    class FakeConsole:
        def print(self, *args, **kwargs): printed.append(str(args[0]) if args else '')
    monkeypatch.setattr(mm, 'console', FakeConsole())
    return printed

@pytest.fixture
def manager(tmp_path):
    config = SimpleNamespace(MODELS_CONFIG_PATH=str(tmp_path/'x'), _load_models_config=lambda: None, force_ollama_check=lambda: None)
    m = mm.ModelManager.__new__(mm.ModelManager)
    m.config = config
    m.config_path = tmp_path/'x'
    m.models_data = {'models': {'a':{'type':'openai'}}}
    return m

def test_delete_model_alias_not_found(manager, patch_console):
    ok = manager.delete_model_alias('b')
    assert not ok
    assert any("Alias 'b' not found" in msg for msg in patch_console)

def test_delete_model_alias_cancel(manager, patch_console, monkeypatch):
    monkeypatch.setattr(mm.Confirm, 'ask', lambda *args, **kwargs: False)
    ok = manager.delete_model_alias('a')
    assert ok
    assert any('Deletion cancelled' in msg for msg in patch_console)
    assert 'a' in manager.models_data['models']

def test_delete_model_alias_confirm(manager, patch_console, monkeypatch):
    monkeypatch.setattr(mm.Confirm, 'ask', lambda *args, **kwargs: True)
    called = {}
    def fake_save(added=0, updated=0, deleted=0):
        called['deleted'] = deleted
        return True
    manager.save_config = fake_save
    ok = manager.delete_model_alias('a')
    assert ok
    assert called.get('deleted') == 1
    assert 'a' not in manager.models_data['models']

def test_update_models_success(monkeypatch, manager):
    calls = []
    monkeypatch.setattr(manager, '_update_provider_models', lambda prov: calls.append(prov) or True)
    ok = manager.update_models()
    assert ok
    assert 'openai' in calls and 'ollama' in calls

def test_update_models_provider_exception(monkeypatch, manager, patch_console):
    def fail(prov):
        if prov == 'openai': raise RuntimeError('err')
        return True
    monkeypatch.setattr(manager, '_update_provider_models', fail)
    ok = manager.update_models()
    assert not ok
    assert any('Error updating openai' in msg for msg in patch_console)

def test_update_models_single_provider(monkeypatch, manager):
    calls = []
    monkeypatch.setattr(manager, '_update_provider_models', lambda prov: calls.append(prov) or True)
    ok = manager.update_models(provider_type='openai')
    assert ok
    assert calls == ['openai']