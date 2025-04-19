import pytest
import ask_llm.main as main_mod

def test_main_success(monkeypatch):
    """Test that main() calls cli_main and exits with code 0 on success."""
    calls = []
    # Patch cli_main to record invocation
    monkeypatch.setattr(main_mod, 'cli_main', lambda: calls.append('cli_main_called'))
    # Patch sys.exit to record exit code
    monkeypatch.setattr(main_mod.sys, 'exit', lambda code=0: calls.append(f'exit_{code}'))
    # Patch console.print (should not be called)
    monkeypatch.setattr(main_mod, 'console', type('C', (), {'print': lambda *args, **kwargs: calls.append(('print', args))}))
    # Call main
    main_mod.main()
    # Verify cli_main then sys.exit(0)
    assert calls == ['cli_main_called', 'exit_0']

def test_main_exception(monkeypatch, capsys):
    """Test that main() handles exceptions, prints error, and exits with code 1."""
    # Make cli_main raise
    def bad_main():
        raise RuntimeError('failure in cli')
    monkeypatch.setattr(main_mod, 'cli_main', bad_main)
    # Patch console.print to capture printed messages
    printed = []
    def fake_print(msg): printed.append(msg)
    monkeypatch.setattr(main_mod, 'console', type('C', (), {'print': fake_print}))
    # Patch sys.exit to record exit code without raising
    monkeypatch.setattr(main_mod.sys, 'exit', lambda code=1: printed.append(f'exit_{code}'))
    # Call main
    main_mod.main()
    # The first printed message should mention the unexpected error and include exception
    assert any('unexpected error occurred' in str(m).lower() for m in printed)
    assert any('failure in cli' in str(m) for m in printed)
    # Finally, exit with code 1
    assert 'exit_1' in printed