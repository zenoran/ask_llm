import pytest
import ask_llm.main as main_mod

def test_main_success(monkeypatch):
    """Test that main() calls cli_main and exits with code 0 on success."""
    calls = []
    monkeypatch.setattr(main_mod, 'cli_main', lambda: calls.append('cli_main_called'))
    monkeypatch.setattr(main_mod.sys, 'exit', lambda code=0: calls.append(f'exit_{code}'))
    monkeypatch.setattr(main_mod, 'console', type('C', (), {'print': lambda *args, **kwargs: calls.append(('print', args))}))
    main_mod.main()
    assert calls == ['cli_main_called', 'exit_0']

def test_main_exception(monkeypatch, capsys):
    """Test that main() handles exceptions, prints error, and exits with code 1."""
    def bad_main():
        raise RuntimeError('failure in cli')
    monkeypatch.setattr(main_mod, 'cli_main', bad_main)
    printed = []
    def fake_print(msg): printed.append(msg)
    monkeypatch.setattr(main_mod, 'console', type('C', (), {'print': fake_print}))
    monkeypatch.setattr(main_mod.sys, 'exit', lambda code=1: printed.append(f'exit_{code}'))
    main_mod.main()
    assert any('unexpected error occurred' in str(m).lower() for m in printed)
    assert any('failure in cli' in str(m) for m in printed)
    assert 'exit_1' in printed