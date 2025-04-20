import pytest
import pathlib
import traceback as tb_module
import ask_llm.core as core_mod
from types import SimpleNamespace
import sys
from ask_llm.utils.config import is_llama_cpp_available

@pytest.fixture
def tmp_model_cache(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        defined_models={},
        available_ollama_models=[],
        OLLAMA_URL='http://localhost',
        MODEL_CACHE_DIR=str(tmp_path),
        MODELS_CONFIG_PATH='models.yaml',
        VERBOSE=False,
        allow_duplicate_response=False
    )
    return cfg, tmp_path

def test_no_model_definition(tmp_model_cache):
    config, _ = tmp_model_cache
    with pytest.raises(ValueError) as exc:
        core_mod.AskLLM('missing', config)
    assert "Could not find model definition" in str(exc.value)

def test_unsupported_model_type(tmp_model_cache):
    config, _ = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'unknown'}}}
    with pytest.raises(ValueError) as exc:
        core_mod.AskLLM('alias', config)
    assert "Unsupported model type 'unknown'" in str(exc.value)

def test_openai_initialization(tmp_model_cache, monkeypatch):
    config, _ = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'openai', 'model_id': 'model1'}}}
    class DummyClient:
        def __init__(self, model, config):
            self.model = model
            self.config = config
    monkeypatch.setattr(core_mod, 'OpenAIClient', DummyClient)
    created = {}
    class DummyHistoryManager:
        def __init__(self, client, config):
            created['client'] = client
            created['config'] = config
        def load_history(self):
            created['loaded'] = True
    monkeypatch.setattr(core_mod, 'HistoryManager', DummyHistoryManager)
    inst = core_mod.AskLLM('alias', config)
    assert isinstance(inst.client, DummyClient)
    assert created['client'] is inst.client
    assert created.get('loaded', False) is True

def test_huggingface_initialization_missing_dep(tmp_model_cache, monkeypatch):
    config, _ = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'huggingface', 'model_id': 'id1'}}}
    monkeypatch.setattr(core_mod, 'HuggingFaceClient', None)
    with pytest.raises(ImportError):
        core_mod.AskLLM('alias', config)

def test_huggingface_initialization_missing_model_id(tmp_model_cache, monkeypatch):
    config, _ = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'huggingface'}}}
    monkeypatch.setattr(core_mod, 'HuggingFaceClient', lambda model_id, config: None)
    with pytest.raises(ValueError) as exc:
        core_mod.AskLLM('alias', config)
    assert "Missing 'model_id'" in str(exc.value)

def test_huggingface_success(tmp_model_cache, monkeypatch):
    config, _ = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'huggingface', 'model_id': 'id1'}}}
    class DummyHF:
        def __init__(self, model_id, config):
            self.model_id = model_id
    monkeypatch.setattr(core_mod, 'HuggingFaceClient', DummyHF)
    monkeypatch.setattr(core_mod, 'HistoryManager', lambda client, config: SimpleNamespace(load_history=lambda: None))
    inst = core_mod.AskLLM('alias', config)
    assert isinstance(inst.client, DummyHF)

def test_ollama_initialization(tmp_model_cache, monkeypatch):
    config, _ = tmp_model_cache
    config.available_ollama_models = []
    config.defined_models = {'models': {'alias': {'type': 'ollama', 'model_id': 'm1'}}}
    class DummyOllama:
        def __init__(self, model, config):
            self.model = model
    printed = []
    monkeypatch.setattr(core_mod, 'OllamaClient', DummyOllama)
    monkeypatch.setattr(core_mod, 'HistoryManager', lambda client, config: SimpleNamespace(load_history=lambda: None))
    monkeypatch.setattr(core_mod, 'console', SimpleNamespace(print=lambda msg: printed.append(msg)))
    inst = core_mod.AskLLM('alias', config)
    assert any('not found on server' in msg for msg in printed)
    assert isinstance(inst.client, DummyOllama)
    printed.clear()
    config.available_ollama_models = ['m1']
    inst2 = core_mod.AskLLM('alias', config)
    assert printed == []

def test_openai_initialization_missing_model_id(tmp_model_cache, monkeypatch):
    config, _ = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'openai'}}}
    monkeypatch.setattr(core_mod, 'OpenAIClient', lambda model_id, config: None)
    with pytest.raises(ValueError) as exc:
        core_mod.AskLLM('alias', config)
    assert "Missing 'model_id'" in str(exc.value)

def test_initialize_client_gguf_missing_hf(monkeypatch, tmp_model_cache):
    config, _ = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'gguf'}}}
    monkeypatch.setattr(core_mod, 'hf_hub_available', False)
    with pytest.raises(ImportError):
        core_mod.AskLLM('alias', config)
 
def test_initialize_client_gguf_missing_llama(monkeypatch, tmp_model_cache):
    config, _ = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'gguf', 'repo_id': 'r', 'filename': 'f'}}}
    monkeypatch.setattr(core_mod, 'hf_hub_available', True)
    monkeypatch.setattr(core_mod, 'LlamaCppClient', None)
    monkeypatch.setattr(core_mod, 'console', SimpleNamespace(print=lambda *args, **kwargs: None))
    with pytest.raises(ImportError):
        core_mod.AskLLM('alias', config)

def test_initialize_client_gguf_missing_fields(monkeypatch, tmp_model_cache):
    config, _ = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'gguf', 'repo_id': None, 'filename': None}}}
    monkeypatch.setattr(core_mod, 'hf_hub_available', True)
    monkeypatch.setattr(core_mod, 'LlamaCppClient', lambda model_path, config: None)
    monkeypatch.setattr(core_mod, 'console', SimpleNamespace(print=lambda *args, **kwargs: None))
    with pytest.raises(ValueError) as exc:
        core_mod.AskLLM('alias', config)
    assert "missing 'repo_id' or 'filename'" in str(exc.value)

def test_initialize_client_gguf_cached(monkeypatch, tmp_model_cache):
    config, tmp_path = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'gguf', 'repo_id': 'repo', 'filename': 'mod.bin'}}}
    monkeypatch.setattr(core_mod, 'hf_hub_available', True)
    class DummyLlama:
        def __init__(self, model_path, config):
            self.model_path = model_path
            self.config = config
    monkeypatch.setattr(core_mod, 'LlamaCppClient', DummyLlama)
    monkeypatch.setattr(core_mod, 'console', SimpleNamespace(print=lambda *args, **kwargs: None))
    monkeypatch.setattr(core_mod, 'HistoryManager', lambda client, config: SimpleNamespace(load_history=lambda: None))
    monkeypatch.setattr(pathlib.Path, 'is_file', lambda self: True)
    inst = core_mod.AskLLM('alias', config)
    expected = str(pathlib.Path(config.MODEL_CACHE_DIR) / 'repo' / 'mod.bin')
    assert isinstance(inst.client, DummyLlama)
    assert inst.client.model_path == expected

def test_initialize_client_gguf_download(monkeypatch, tmp_model_cache):
    config, tmp_path = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'gguf', 'repo_id': 'repo', 'filename': 'mod.bin'}}}
    monkeypatch.setattr(core_mod, 'hf_hub_available', True)
    monkeypatch.setattr(pathlib.Path, 'is_file', lambda self: False)
    monkeypatch.setattr(core_mod, 'hf_hub_download', lambda **kwargs: '/other/path')
    class DummyLlama2:
        def __init__(self, model_path, config):
            self.model_path = model_path
    monkeypatch.setattr(core_mod, 'LlamaCppClient', DummyLlama2)
    printed = []
    monkeypatch.setattr(core_mod, 'console', SimpleNamespace(print=lambda msg: printed.append(msg)))
    monkeypatch.setattr(core_mod, 'HistoryManager', lambda client, config: SimpleNamespace(load_history=lambda: None))
    inst = core_mod.AskLLM('alias', config)
    assert any('differs from expected cache path' in msg for msg in printed)
    assert inst.client.model_path == '/other/path'

@pytest.mark.skipif(not is_llama_cpp_available(), reason="Requires llama-cpp-python")
def test_initialize_client_gguf_download_success(monkeypatch, tmp_model_cache):
    config, tmp_path = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'gguf', 'repo_id': 'repo', 'filename': 'mod.bin'}}}
    monkeypatch.setattr(core_mod, 'hf_hub_available', True)
    monkeypatch.setattr(pathlib.Path, 'is_file', lambda self: False)
    monkeypatch.setattr(core_mod, 'hf_hub_download', lambda **kwargs: '/other/path')
    class DummyLlama2:
        def __init__(self, model_path, config):
            self.model_path = model_path
    monkeypatch.setattr(core_mod, 'LlamaCppClient', DummyLlama2)
    printed = []
    monkeypatch.setattr(core_mod, 'console', SimpleNamespace(print=lambda msg: printed.append(msg)))
    monkeypatch.setattr(core_mod, 'HistoryManager', lambda client, config: SimpleNamespace(load_history=lambda: None))
    inst = core_mod.AskLLM('alias', config)
    assert any('differs from expected cache path' in msg for msg in printed)
    assert inst.client.model_path == '/other/path'

@pytest.mark.skipif(not is_llama_cpp_available(), reason="Requires llama-cpp-python")
def test_initialize_client_gguf_download_http_error(monkeypatch, tmp_model_cache):
    config, tmp_path = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'gguf', 'repo_id': 'repo', 'filename': 'mod.bin'}}}
    monkeypatch.setattr(core_mod, 'hf_hub_available', True)
    monkeypatch.setattr(pathlib.Path, 'is_file', lambda self: False)
    monkeypatch.setattr(core_mod, 'hf_hub_download', lambda **kwargs: (_ for _ in ()).throw(core_mod.HfHubHTTPError('err')))
    monkeypatch.setattr(core_mod, 'console', SimpleNamespace(print=lambda msg: None))
    monkeypatch.setattr(core_mod, 'HistoryManager', lambda client, config: SimpleNamespace(load_history=lambda: None))
    with pytest.raises(core_mod.HfHubHTTPError):
        core_mod.AskLLM('alias', config)

@pytest.mark.skipif(not is_llama_cpp_available(), reason="Requires llama-cpp-python")
def test_initialize_client_gguf_download_generic_error(monkeypatch, tmp_model_cache):
    config, tmp_path = tmp_model_cache
    config.defined_models = {'models': {'alias': {'type': 'gguf', 'repo_id': 'repo', 'filename': 'mod.bin'}}}
    monkeypatch.setattr(core_mod, 'hf_hub_available', True)
    monkeypatch.setattr(pathlib.Path, 'is_file', lambda self: False)
    monkeypatch.setattr(core_mod, 'hf_hub_download', lambda **kwargs: (_ for _ in ()).throw(RuntimeError('err')))
    printed = []
    monkeypatch.setattr(core_mod, 'console', SimpleNamespace(print=lambda msg: printed.append(msg)))
    config.VERBOSE = True
    monkeypatch.setattr(tb_module, 'print_exc', lambda: printed.append('tb'))
    monkeypatch.setattr(core_mod, 'HistoryManager', lambda client, config: SimpleNamespace(load_history=lambda: None))
    with pytest.raises(RuntimeError):
        core_mod.AskLLM('alias', config)
    assert any("Error downloading file 'mod.bin'" in m for m in printed)
    assert 'tb' in printed
