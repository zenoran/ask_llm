import pytest
import traceback
from argparse import Namespace
import ask_llm.cli as cli_mod

@pytest.mark.parametrize('first_call_exception', [FileNotFoundError])
def test_config_file_not_found(monkeypatch, first_call_exception):
    """Test that missing config file falls back and issues a warning."""
    printed = []
    monkeypatch.setattr(cli_mod, 'console', type('C', (), {'print': lambda *args, **kwargs: printed.append(args[0])}))
    from types import SimpleNamespace
    fake_config = SimpleNamespace()
    call_count = {'n': 0}
    def fake_Config(*args, **kwargs):
        if call_count['n'] == 0:
            call_count['n'] += 1
            raise first_call_exception()
        return fake_config
    monkeypatch.setattr(cli_mod, 'Config', fake_Config)
    args = Namespace(list_models=True, add_gguf=None, delete_model=None,
                     refresh_models=None, question=[], command=None,
                     verbose=False, plain=False, no_stream=False,
                     delete_history=False, print_history=None, model='alias')
    monkeypatch.setattr(cli_mod, 'parse_arguments', lambda cfg: args)
    calls = []
    monkeypatch.setattr(cli_mod, 'list_models', lambda cfg: calls.append(('list_models', cfg)))
    monkeypatch.setattr(cli_mod.sys, 'exit', lambda code=0: calls.append(('exit', code)))
    cli_mod.main()
    assert any('warning' in msg.lower() for msg in printed)
    assert calls == [('list_models', fake_config), ('exit', 0)]

def test_add_gguf_error(monkeypatch):
    """Test that add_gguf exception is handled with error print and exit(1)."""
    printed = []
    monkeypatch.setattr(cli_mod, 'console', type('C', (), {'print': lambda *args, **kwargs: printed.append(args[0])}))
    config = type('Cfg', (), {'VERBOSE': True})()
    monkeypatch.setattr(cli_mod, 'Config', lambda *args, **kwargs: config)
    args = Namespace(list_models=False, add_gguf='repo', delete_model=None,
                     refresh_models=None, question=[], command=None,
                     verbose=True, plain=False, no_stream=False,
                     delete_history=False, print_history=None, model='alias')
    monkeypatch.setattr(cli_mod, 'parse_arguments', lambda cfg: args)
    def bad_add(repo, cfg): raise RuntimeError('oops')
    monkeypatch.setattr(cli_mod, 'handle_add_gguf', bad_add)
    monkeypatch.setattr(traceback, 'print_exc', lambda: printed.append('traceback'))
    monkeypatch.setattr(cli_mod.sys, 'exit', lambda code=1: printed.append(f'exit_{code}'))
    cli_mod.main()
    assert any('error during gguf add operation' in msg.lower() for msg in printed)
    assert 'traceback' in printed
    assert 'exit_1' in printed

@pytest.mark.parametrize('provider', ['openai', 'ollama'])
def test_refresh_models_keyboard_interrupt(monkeypatch, provider):
    """Test that KeyboardInterrupt during refresh_models is handled gracefully."""
    printed = []
    monkeypatch.setattr(cli_mod, 'console', type('C', (), {'print': lambda *args, **kwargs: printed.append(args[0])}))
    config = type('Cfg', (), {'VERBOSE': False})()
    monkeypatch.setattr(cli_mod, 'Config', lambda *args, **kwargs: config)
    args = Namespace(list_models=False, add_gguf=None, delete_model=None,
                     refresh_models=provider, question=[], command=None,
                     verbose=False, plain=False, no_stream=False,
                     delete_history=False, print_history=None, model='alias')
    monkeypatch.setattr(cli_mod, 'parse_arguments', lambda cfg: args)
    monkeypatch.setattr(cli_mod, 'update_models_interactive', lambda cfg, provider=None: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(cli_mod.sys, 'exit', lambda code=1: printed.append(f'exit_{code}'))
    cli_mod.main()
    assert any('cancelled' in msg.lower() for msg in printed)
    assert 'exit_1' in printed

def test_delete_model_error(monkeypatch):
    """Test that delete_model exception is handled with error print and exit(1)."""
    printed = []
    monkeypatch.setattr(cli_mod, 'console', type('C', (), {'print': lambda *args, **kwargs: printed.append(args[0])}))
    config = type('Cfg', (), {'VERBOSE': True})()
    monkeypatch.setattr(cli_mod, 'Config', lambda *args, **kwargs: config)
    args = Namespace(list_models=False, add_gguf=None, delete_model='alias',
                     refresh_models=None, question=[], command=None,
                     verbose=True, plain=False, no_stream=False,
                     delete_history=False, print_history=None, model='alias')
    monkeypatch.setattr(cli_mod, 'parse_arguments', lambda cfg: args)
    def bad_delete(alias, cfg): raise ValueError('del error')
    monkeypatch.setattr(cli_mod, 'delete_model', bad_delete)
    monkeypatch.setattr(traceback, 'print_exc', lambda: printed.append('traceback'))
    monkeypatch.setattr(cli_mod.sys, 'exit', lambda code=1: printed.append(f'exit_{code}'))
    cli_mod.main()
    assert any('error during delete model operation' in msg.lower() for msg in printed)
    assert 'traceback' in printed
    assert 'exit_1' in printed