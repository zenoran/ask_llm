import yaml
import pytest
from pathlib import Path
from types import SimpleNamespace

import ask_llm.model_manager as mm

@pytest.fixture(autouse=True)
def patch_console(monkeypatch):
    printed = []
    class FakeConsole:
        def print(self, *args, **kwargs):
            printed.append(args[0] if args else '')
    monkeypatch.setattr(mm, 'console', FakeConsole())
    return printed

def test_load_config_nonexistent(tmp_path, patch_console):
    config = SimpleNamespace(MODELS_CONFIG_PATH=str(tmp_path/'models.yaml'))
    manager = mm.ModelManager(config)
    assert manager.models_data == {'models': {}}
    assert any('Config file' in msg for msg in patch_console)

def test_load_config_invalid_yaml(tmp_path, patch_console):
    cfg_path = tmp_path/'models.yaml'
    cfg_path.write_text(':::')
    config = SimpleNamespace(MODELS_CONFIG_PATH=str(cfg_path))
    manager = mm.ModelManager(config)
    assert manager.models_data == {'models': {}}
    assert any('Error loading config' in msg for msg in patch_console)

def test_load_config_missing_models_key(tmp_path, patch_console):
    cfg_path = tmp_path/'models.yaml'
    cfg_path.write_text('foo: bar')
    config = SimpleNamespace(MODELS_CONFIG_PATH=str(cfg_path))
    manager = mm.ModelManager(config)
    assert 'models' in manager.models_data
    assert manager.models_data.get('models') == {}
    assert any('Invalid format' in msg for msg in patch_console)

def test_load_config_valid(tmp_path, patch_console):
    model_def = {'models': {'a': {'type': 'openai', 'model_id': 'mid'}}}
    cfg_path = tmp_path/'models.yaml'
    cfg_path.write_text(yaml.dump(model_def))
    config = SimpleNamespace(MODELS_CONFIG_PATH=str(cfg_path))
    manager = mm.ModelManager(config)
    assert manager.models_data == model_def
    assert not any('Warning' in msg for msg in patch_console)

def test_save_config_and_order(tmp_path, patch_console):
    cfg_path = tmp_path/'models.yaml'
    config = SimpleNamespace(MODELS_CONFIG_PATH=str(cfg_path), _load_models_config=lambda: None)
    manager = mm.ModelManager(config)
    manager.models_data = {'models': {
        'b': {'type': 'openai'},
        'a': {'type': 'ollama'}
    }}
    ok = manager.save_config(added=1, updated=2, deleted=3)
    assert ok
    loaded = yaml.safe_load(cfg_path.read_text())
    assert list(loaded['models'].keys()) == ['a', 'b']
    assert any('1 added' in msg for msg in patch_console)
    assert any('2 updated' in msg for msg in patch_console)
    assert any('3 deleted' in msg for msg in patch_console)

def test_save_config_failure(tmp_path, patch_console, monkeypatch):
    cfg_path = tmp_path/'models.yaml'
    config = SimpleNamespace(MODELS_CONFIG_PATH=str(cfg_path))
    manager = mm.ModelManager(config)
    manager.models_data = {'models': {}}
    monkeypatch.setattr(manager.config_path.parent, 'mkdir', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('fail')))
    ok = manager.save_config()
    assert not ok
    assert any('Error saving configuration' in msg for msg in patch_console)