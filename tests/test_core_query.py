import pytest
import ask_llm.core as core_mod
from types import SimpleNamespace

class DummyHistory:
    def __init__(self, initial_assistant=None):
        self.messages = []
        self.initial_assistant = initial_assistant
        self.removed = False
    def load_history(self):
        pass
    def add_message(self, role, text):
        self.messages.append((role, text))
    def get_context_messages(self):
        return self.messages.copy()
    def get_last_assistant_message(self):
        if self.initial_assistant is not None and not any(r=='assistant' for r,_ in self.messages):
            return self.initial_assistant
        for role, text in reversed(self.messages):
            if role == 'assistant':
                return text
        return None
    def remove_last_message_if_partial(self, role):
        self.removed = True

class DummyClient:
    def __init__(self, responses):
        self._iter = iter(responses)
    def query(self, messages=None, plaintext_output=False, stream=True):
        return next(self._iter)

@pytest.fixture
def base_config(monkeypatch):
    cfg = SimpleNamespace(
        defined_models={'models': {'alias': {'type': 'huggingface', 'model_id': 'id1'}}},
        available_ollama_models=[],
        OLLAMA_URL='', MODEL_CACHE_DIR='',
        VERBOSE=False, allow_duplicate_response=False
    )
    class DummyHF:
        def __init__(self, model_id, config): pass
    monkeypatch.setattr(core_mod, 'HuggingFaceClient', DummyHF)
    monkeypatch.setattr(core_mod, 'HistoryManager', lambda client, config: DummyHistory())
    return cfg

def test_query_success(base_config, monkeypatch):
    cfg = base_config
    def dummy_init(self, config): return DummyClient(['resp'])
    monkeypatch.setattr(core_mod.AskLLM, 'initialize_client', dummy_init)
    inst = core_mod.AskLLM('alias', cfg)
    hist = DummyHistory()
    inst.history_manager = hist
    result = inst.query('prompt', plaintext_output=True, stream=False)
    assert result == 'resp'
    assert hist.messages[0][0] == 'user'
    assert hist.messages[1][0] == 'assistant'

def test_query_duplicate_detection(base_config, monkeypatch):
    cfg = base_config
    initial = 'dup'
    hist = DummyHistory(initial_assistant=initial)
    client = DummyClient(['dup', 'new'])
    monkeypatch.setattr(core_mod.AskLLM, 'initialize_client', lambda self, config: client)
    inst = core_mod.AskLLM('alias', cfg)
    inst.history_manager = hist
    printed = []
    monkeypatch.setattr(core_mod, 'console', SimpleNamespace(print=lambda msg: printed.append(msg)))
    result = inst.query('p', plaintext_output=False, stream=True)
    assert any('duplicate response' in msg.lower() for msg in printed)
    assert result == 'new'

def test_query_keyboard_interrupt(base_config, monkeypatch):
    cfg = base_config
    class ClientKI:
        def query(self, **kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(core_mod.AskLLM, 'initialize_client', lambda self, config: ClientKI())
    inst = core_mod.AskLLM('alias', cfg)
    hist = DummyHistory()
    inst.history_manager = hist
    printed = []
    monkeypatch.setattr(core_mod, 'console', SimpleNamespace(print=lambda msg: printed.append(msg)))
    res = inst.query('p', plaintext_output=False, stream=True)
    assert res == ''
    assert hist.removed is True
    assert any('query interrupted' in msg.lower() for msg in printed)

def test_query_generic_exception(base_config, monkeypatch):
    cfg = base_config
    class ClientErr:
        def query(self, **kwargs): raise RuntimeError('err')
    monkeypatch.setattr(core_mod.AskLLM, 'initialize_client', lambda self, config: ClientErr())
    inst = core_mod.AskLLM('alias', cfg)
    hist = DummyHistory()
    inst.history_manager = hist
    printed = []
    monkeypatch.setattr(core_mod, 'console', SimpleNamespace(print=lambda msg: printed.append(msg)))
    res = inst.query('p', plaintext_output=False, stream=True)
    assert any('error during query' in msg.lower() for msg in printed)
    assert res == ""