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
def config():
    return SimpleNamespace(
        MODELS_CONFIG_PATH='x',
        MODEL_OPTIONS=['a','b','c'],
        DEFAULT_MODEL_ALIAS='b'
    )

@pytest.fixture
def manager(config):
    m = mm.ModelManager.__new__(mm.ModelManager)
    m.config = config
    m.config_path = None
    m.models_data = {'models': {}}
    return m

def test_list_available_models_empty(manager, patch_console):
    manager.models_data = {'models': {}}
    manager.list_available_models()
    assert any('No models defined' in msg for msg in patch_console)

def test_list_available_models_with_models(manager, patch_console):
    manager.models_data = {'models': {
        'a': {'type':'openai','model_id':'id1','description':'desc1'},
        'b': {'type':'ollama','model_id':'id2'},
        'd': {'type':'huggingface','model_id':'id3'}
    }}
    manager.config.MODEL_OPTIONS = ['a','d']
    manager.config.DEFAULT_MODEL_ALIAS = 'a'
    manager.list_available_models()
    out = ''.join(patch_console)
    assert 'OPENAI Models' in out
    assert 'OLLAMA Models' in out
    assert 'HUGGINGFACE Models' in out
    assert '✓' in out
    assert '✗' in out
    assert 'Default alias' in out

@pytest.mark.parametrize('default_in_options', [True, False])
def test_resolve_model_alias_none(manager, config, patch_console, default_in_options):
    config.DEFAULT_MODEL_ALIAS = 'b'
    config.MODEL_OPTIONS = ['b'] if default_in_options else []
    res = manager.resolve_model_alias(None)
    if default_in_options:
        assert res == 'b'
    else:
        assert res is None
        assert any('No model specified' in msg for msg in patch_console)

def test_resolve_model_alias_exact(manager, config):
    config.MODEL_OPTIONS = ['xyz']
    manager.models_data = {'models': {'xyz':{}}}
    assert manager.resolve_model_alias('xyz') == 'xyz'

def test_resolve_model_alias_defined_but_unavailable(manager, config, patch_console):
    config.MODEL_OPTIONS = []
    manager.models_data = {'models': {'x':{'type':'openai'}}}
    res = manager.resolve_model_alias('x')
    assert res is None
    assert any('defined but unavailable' in msg for msg in patch_console)

def test_resolve_model_alias_partial_single(manager, config):
    config.MODEL_OPTIONS = ['alpha','beta']
    manager.models_data = {'models': {}}
    assert manager.resolve_model_alias('alp') == 'alpha'

def test_resolve_model_alias_partial_multiple(manager, config, patch_console, monkeypatch):
    config.MODEL_OPTIONS = ['apple','apricot','banana']
    manager.models_data = {'models': {}}
    monkeypatch.setattr(mm.Prompt, 'ask', lambda prompt, **kw: '2')
    res = manager.resolve_model_alias('ap')
    assert res == 'apricot'
    assert any('Multiple matches' in msg for msg in patch_console)

def test_resolve_model_alias_not_found(manager, config, patch_console):
    config.MODEL_OPTIONS = ['a']
    manager.models_data = {'models': {}}
    res = manager.resolve_model_alias('zzz')
    assert res is None
    assert any("Alias 'zzz' not found" in msg for msg in patch_console)