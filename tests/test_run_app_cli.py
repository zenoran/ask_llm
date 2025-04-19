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
        # client with no console attribute to force use of cli console
        self.client = SimpleNamespace()
        self.queries = []
    def query(self, text, plaintext_output=False, stream=True):
        self.queries.append((text, plaintext_output, stream))

class DummyInputHandler:
    def __init__(self, console=None):
        self.called = False
    def get_input(self, prompt):
        # Simulate user immediately quitting
        return None, False

@pytest.fixture(autouse=True)
def patch_console(monkeypatch):
    # Capture console.print outputs
    printed = []
    class FakeConsole:
        def print(self, *args, **kwargs):
            # Record first positional argument or empty string
            printed.append(args[0] if args else '')
    fake_console = FakeConsole()
    monkeypatch.setattr(cli_mod, 'console', fake_console)
    return printed

@pytest.fixture
def config():
    # Default config for run_app
    return SimpleNamespace(VERBOSE=False, PLAIN_OUTPUT=False, NO_STREAM=False)

def test_run_app_init_failure(patch_console, config, monkeypatch):
    # Simulate AskLLM constructor failure
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: (_ for _ in ()).throw(ValueError('init error')))
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=[], command=None)
    with pytest.raises(SystemExit) as exc:
        cli_mod.run_app(args, config, 'alias')
    # Expect exit code 1
    assert exc.value.code == 1
    # Error message printed
    assert any('Failed to initialize LLM client for' in msg for msg in patch_console)

def test_run_app_print_and_delete_history(patch_console, config, monkeypatch):
    # Patch AskLLM to our dummy
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    # Prepare args: delete_history and print_history with no question/command
    args = SimpleNamespace(delete_history=True, print_history=5,
                          question=[], command=None)
    # Call run_app
    cli_mod.run_app(args, config, 'alias')
    # History cleared and printed
    assert dummy.history_manager.cleared is True
    assert dummy.history_manager.printed == 5
    # Console printed a blank line on print_history return
    assert '' in patch_console
    # No queries were made
    assert dummy.queries == []

def test_run_app_question_path(patch_console, config, monkeypatch):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    # Provide a question list
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=['Hello', 'world'], command=None)
    cli_mod.run_app(args, config, 'alias')
    # Should have one query with concatenated text
    assert len(dummy.queries) == 1
    text, plaintext, stream = dummy.queries[0]
    assert text == 'Hello world'
    assert plaintext is False
    assert stream is True

def test_run_app_command_success(patch_console, config, monkeypatch):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    # Simulate subprocess.run returning stdout and no error, returncode 0
    fake_result = SimpleNamespace(stdout='out', stderr='', returncode=0)
    monkeypatch.setattr(subprocess, 'run', lambda cmd, shell, capture_output, text, check: fake_result)
    # Use verbose to hit that branch
    config.VERBOSE = True
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=[], command='echo hi')
    cli_mod.run_app(args, config, 'alias')
    # Should print executing command line
    assert any('Executing command' in msg for msg in patch_console)
    # One query made with output included
    assert len(dummy.queries) == 1
    text, _, _ = dummy.queries[0]
    assert 'out' in text

def test_run_app_command_failure(patch_console, config, monkeypatch):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    # Simulate subprocess.run returning stderr and non-zero returncode
    fake_result = SimpleNamespace(stdout='', stderr='err', returncode=2)
    monkeypatch.setattr(subprocess, 'run', lambda cmd, shell, capture_output, text, check: fake_result)
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=[], command='badcmd')
    cli_mod.run_app(args, config, 'alias')
    # Should print warning about exit status
    assert any('Warning' in msg for msg in patch_console)
    # Query includes error prefix
    assert dummy.queries[0][0].startswith('Command')

def test_run_app_interactive_mode(patch_console, config, monkeypatch):
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    # Patch input handler to immediate exit
    monkeypatch.setattr(cli_mod, 'MultilineInputHandler', DummyInputHandler)
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=[], command=None)
    # Run interactive mode
    cli_mod.run_app(args, config, 'alias')
    # Should print entering and exiting messages
    assert any('Entering interactive mode' in msg for msg in patch_console)
    assert any('Exiting interactive mode' in msg for msg in patch_console)
    # No queries made
    assert dummy.queries == []
   
def test_run_app_print_history_with_question(patch_console, config, monkeypatch):
    # Test print_history with a question continues to query
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    # Provide print_history and a question
    args = SimpleNamespace(delete_history=False, print_history=3,
                          question=['Hi'], command=None)
    cli_mod.run_app(args, config, 'alias')
    # History printed and then query called
    assert dummy.history_manager.printed == 3
    assert len(dummy.queries) == 1

def test_run_app_interactive_empty_input(patch_console, config, monkeypatch):
    # Test interactive mode empty input branch
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    # Handler returns empty string then None
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
    # Should print empty input message and exit
    assert any('empty input received' in msg.lower() for msg in patch_console)
    assert any('Exiting interactive mode' in msg for msg in patch_console)

def test_run_app_interactive_eof(monkeypatch, patch_console, config):
    # Test interactive mode EOFError handling
    dummy = DummyAskLLM()
    monkeypatch.setattr(cli_mod, 'AskLLM', lambda resolved_model_alias, config: dummy)
    # Handler raises EOFError
    class EH2:
        def __init__(self, console=None): pass
        def get_input(self, prompt):
            raise EOFError()
    monkeypatch.setattr(cli_mod, 'MultilineInputHandler', EH2)
    args = SimpleNamespace(delete_history=False, print_history=None,
                          question=[], command=None)
    cli_mod.run_app(args, config, 'alias')
    # Should print exiting message
    assert any('Exiting interactive mode' in msg for msg in patch_console)