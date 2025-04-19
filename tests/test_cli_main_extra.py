import pytest
import traceback
from types import SimpleNamespace
import ask_llm.cli as cli_mod

@pytest.mark.parametrize('success', [False])
def test_main_add_gguf_returns_false(monkeypatch, success):
    """Test --add-gguf when handle_add_gguf returns False."""
    printed = []
    monkeypatch.setattr(cli_mod, 'console', type('C', (), {'print': lambda *args, **kwargs: printed.append(args[0])}))
    # Config instance
    config = SimpleNamespace(VERBOSE=False)
    monkeypatch.setattr(cli_mod, 'Config', lambda *args, **kwargs: config)
    # parse_arguments returns args with add_gguf
    args = SimpleNamespace(list_models=False, add_gguf='repo', delete_model=None,
                          refresh_models=None, question=[], command=None,
                          verbose=False, plain=False, no_stream=False,
                          delete_history=False, print_history=None, model='alias')
    monkeypatch.setattr(cli_mod, 'parse_arguments', lambda cfg: args)
    # handle_add_gguf returns False
    monkeypatch.setattr(cli_mod, 'handle_add_gguf', lambda repo, cfg: False)
    # Patch sys.exit
    exit_calls = []
    monkeypatch.setattr(cli_mod.sys, 'exit', lambda code=0: exit_calls.append(code))
    # Invoke main
    cli_mod.main()
    # Expect failure message and exit(1)
    assert any('failed to process gguf model request' in msg.lower() for msg in printed)
    assert exit_calls == [1]

@pytest.mark.parametrize('provider', ['openai', 'ollama'])
def test_main_refresh_models_returns_false(monkeypatch, provider):
    """Test --refresh-models when update_models_interactive returns False."""
    printed = []
    monkeypatch.setattr(cli_mod, 'console', type('C', (), {'print': lambda *args, **kwargs: printed.append(args[0])}))
    # Config instance
    config = SimpleNamespace(VERBOSE=False)
    monkeypatch.setattr(cli_mod, 'Config', lambda *args, **kwargs: config)
    # parse_arguments yields provider
    args = SimpleNamespace(list_models=False, add_gguf=None, delete_model=None,
                          refresh_models=provider, question=[], command=None,
                          verbose=False, plain=False, no_stream=False,
                          delete_history=False, print_history=None, model='alias')
    monkeypatch.setattr(cli_mod, 'parse_arguments', lambda cfg: args)
    # update_models_interactive returns False
    monkeypatch.setattr(cli_mod, 'update_models_interactive', lambda cfg, provider=None: False)
    # Patch sys.exit
    exit_calls = []
    monkeypatch.setattr(cli_mod.sys, 'exit', lambda code=0: exit_calls.append(code))
    # Invoke main
    cli_mod.main()
    # Expect failure message and exit(1)
    assert any('model list refresh for' in msg.lower() and 'failed' in msg.lower() for msg in printed)
    assert exit_calls == [1]

def test_main_refresh_models_exception(monkeypatch):
    """Test --refresh-models when update_models_interactive raises Exception."""
    printed = []
    monkeypatch.setattr(cli_mod, 'console', type('C', (), {'print': lambda *args, **kwargs: printed.append(args[0])}))
    # Config instance with verbose True to trigger traceback
    config = SimpleNamespace(VERBOSE=True)
    monkeypatch.setattr(cli_mod, 'Config', lambda *args, **kwargs: config)
    # parse_arguments yields refresh_models
    args = SimpleNamespace(list_models=False, add_gguf=None, delete_model=None,
                          refresh_models='openai', question=[], command=None,
                          verbose=True, plain=False, no_stream=False,
                          delete_history=False, print_history=None, model='alias')
    monkeypatch.setattr(cli_mod, 'parse_arguments', lambda cfg: args)
    # update_models_interactive raises
    monkeypatch.setattr(cli_mod, 'update_models_interactive', lambda cfg, provider=None: (_ for _ in ()).throw(RuntimeError('oops')))
    # Patch traceback.print_exc
    tb = []
    monkeypatch.setattr(traceback, 'print_exc', lambda: tb.append('tb'))
    # Patch sys.exit
    exit_calls = []
    monkeypatch.setattr(cli_mod.sys, 'exit', lambda code=0: exit_calls.append(code))
    # Invoke main
    cli_mod.main()
    # Error message printed and traceback
    assert any('error during model refresh' in msg.lower() for msg in printed)
    assert 'tb' in tb
    assert exit_calls == [1]

def test_main_delete_model_returns_false(monkeypatch):
    """Test --delete-model when delete_model returns False."""
    printed = []
    monkeypatch.setattr(cli_mod, 'console', type('C', (), {'print': lambda *args, **kwargs: printed.append(args[0])}))
    # Config instance
    config = SimpleNamespace(VERBOSE=False)
    monkeypatch.setattr(cli_mod, 'Config', lambda *args, **kwargs: config)
    # parse_arguments yields delete_model
    args = SimpleNamespace(list_models=False, add_gguf=None, delete_model='alias',
                          refresh_models=None, question=[], command=None,
                          verbose=False, plain=False, no_stream=False,
                          delete_history=False, print_history=None, model='alias')
    monkeypatch.setattr(cli_mod, 'parse_arguments', lambda cfg: args)
    # delete_model returns False
    monkeypatch.setattr(cli_mod, 'delete_model', lambda alias, cfg: False)
    # Patch sys.exit
    exit_calls = []
    monkeypatch.setattr(cli_mod.sys, 'exit', lambda code=0: exit_calls.append(code))
    # Invoke main
    cli_mod.main()
    # Should exit with code 1 and no success message
    assert exit_calls == [1]
    assert not any('deleted successfully' in msg.lower() for msg in printed)