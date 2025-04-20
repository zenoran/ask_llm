import subprocess
import pytest
from types import SimpleNamespace
import ask_llm.cli as cli_mod

class DummyHistory:
    def __init__(self):
        self.cleared = False
        self.printed = None
    def clear_history(self):
        self.cleared = True
    def print_history(self, n):
        self.printed = n

class DummyAskLLM:
    def __init__(self):
        self.history_manager = DummyHistory()
        self.client = SimpleNamespace()
        self.queries = []
    def query(self, text, plaintext_output=False, stream=True):
        self.queries.append((text, plaintext_output, stream))

class DummyInputHandler:
    def __init__(self, console=None):
        self.called = False
    def get_input(self, prompt):
        return None, False

@pytest.fixture(autouse=True)
def patch_console(monkeypatch):
    printed = []
    class FakeConsole:
        def print(self, *args, **kwargs):
            printed.append(args[0] if args else '')
    fake_console = FakeConsole()
    monkeypatch.setattr(cli_mod, 'console', fake_console)
    return printed

@pytest.fixture
def config():
    return SimpleNamespace(VERBOSE=False, PLAIN_OUTPUT=False, NO_STREAM=False)

def test_run_app_init_failure(patch_console, config, monkeypatch):
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: (_ for _ in ()).throw(ValueError('init error')))
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=[], command=None)
    with pytest.raises(SystemExit) as exc:
        cli_mod.run_app(args, config, 'alias')
    assert exc.value.code == 1
    assert any('Failed to initialize LLM client for' in msg for msg in patch_console)

def test_run_app_print_and_delete_history(patch_console, config, monkeypatch):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    args = SimpleNamespace(delete_history=True, print_history=5,
                          question=[], command=None)
    cli_mod.run_app(args, config, 'alias')
    assert dummy.history_manager.cleared is True
    assert dummy.history_manager.printed == 5
    assert '' in patch_console
    assert dummy.queries == []

def test_run_app_question_path(patch_console, config, monkeypatch):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=['Hello', 'world'], command=None)
    cli_mod.run_app(args, config, 'alias')
    assert len(dummy.queries) == 1
    text, plaintext, stream = dummy.queries[0]
    assert text == 'Hello world'
    assert plaintext is False
    assert stream is True

def test_run_app_command_success(patch_console, config, monkeypatch):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    fake_result = SimpleNamespace(stdout='out', stderr='', returncode=0)
    monkeypatch.setattr(subprocess, 'run', lambda cmd, shell, capture_output, text, check: fake_result)
    config.VERBOSE = True
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=[], command='echo hi')
    cli_mod.run_app(args, config, 'alias')
    assert any('Executing command' in msg for msg in patch_console)
    assert len(dummy.queries) == 1
    text, _, _ = dummy.queries[0]
    assert 'out' in text

def test_run_app_command_failure(patch_console, config, monkeypatch):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    fake_result = SimpleNamespace(stdout='', stderr='err', returncode=2)
    monkeypatch.setattr(subprocess, 'run', lambda cmd, shell, capture_output, text, check: fake_result)
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=[], command='badcmd')
    cli_mod.run_app(args, config, 'alias')
    assert any('Warning' in msg for msg in patch_console)
    assert dummy.queries[0][0].startswith('Command')

def test_run_app_interactive_mode(patch_console, config, monkeypatch):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    monkeypatch.setattr(cli_mod, 'MultilineInputHandler', DummyInputHandler)
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=[], command=None)
    cli_mod.run_app(args, config, 'alias')
    assert any('Entering interactive mode' in msg for msg in patch_console)
    assert any('Exiting interactive mode' in msg for msg in patch_console)
    assert dummy.queries == []
   
def test_run_app_print_history_with_question(patch_console, config, monkeypatch):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    args = SimpleNamespace(delete_history=False, print_history=3,
                          question=['Hi'], command=None)
    cli_mod.run_app(args, config, 'alias')
    assert dummy.history_manager.printed == 3
    assert len(dummy.queries) == 1

def test_run_app_interactive_empty_input(patch_console, config, monkeypatch):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    class EH:
        def __init__(self, console=None): self.count = 0
        def get_input(self, prompt):
            self.count += 1
            if self.count == 1:
                return '', False
            return None, False
    monkeypatch.setattr(cli_mod, 'MultilineInputHandler', EH)
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=[], command=None)
    cli_mod.run_app(args, config, 'alias')
    assert any('empty input received' in msg.lower() for msg in patch_console)
    assert any('Exiting interactive mode' in msg for msg in patch_console)

def test_run_app_interactive_eof(monkeypatch, patch_console, config):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    class EH2:
        def __init__(self, console=None): pass
        def get_input(self, prompt):
            raise EOFError()
    monkeypatch.setattr(cli_mod, 'MultilineInputHandler', EH2)
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=[], command=None)
    cli_mod.run_app(args, config, 'alias')
    assert any('Exiting interactive mode' in msg for msg in patch_console)