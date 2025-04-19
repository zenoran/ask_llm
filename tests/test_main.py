import pytest
from argparse import Namespace, ArgumentTypeError
from unittest.mock import patch, MagicMock, call
import subprocess # Import subprocess
import shutil
from pathlib import Path

# Import the refactored functions
from src.ask_llm.cli import parse_arguments # Moved from main
from src.ask_llm.model_manager import ModelManager # Replaced resolve_model_alias
from src.ask_llm.core import AskLLM # Removed DEFAULT_SYSTEM_PROMPT
from src.ask_llm.utils.input_handler import MultilineInputHandler # Moved from main
from src.ask_llm.utils.config import Config
from src.ask_llm.utils.history import HistoryManager # Renamed from ChatHistoryManager

# Import client classes for mocking
from src.ask_llm.clients import OpenAIClient, OllamaClient

# Try to import HuggingFaceClient, but mock it if unavailable
try:
    from src.ask_llm.clients import HuggingFaceClient
except ImportError:
    HuggingFaceClient = MagicMock()  # Mock the HuggingFaceClient for tests

# No longer need to import the global config object here

# --- Define Mock Models Globally for Reuse --- #
MOCK_OLLAMA = ['llama3:latest', 'llama3:instruct', 'gemma:7b']
MOCK_OPENAI = ['gpt-4o', 'chatgpt-4o-latest']
MOCK_HF = ['mock-hf-model/pygmalion-3-12b']
ALL_MOCK_MODELS = MOCK_OPENAI + MOCK_OLLAMA + MOCK_HF

# Helper to create a mock config object for tests
def create_mock_config(default_alias='gpt4o', models_path='/fake/models.yaml'):
    """Creates a MagicMock config object with necessary attributes for tests."""
    mock_config = MagicMock()
    # Match actual Config properties used in parsing/resolution
    mock_config.DEFAULT_MODEL_ALIAS = default_alias
    mock_config.MODELS_CONFIG_PATH = models_path 
    mock_config.MODEL_OPTIONS = ALL_MOCK_MODELS # Used by resolve_model_alias
    # Add other attributes if needed by tested functions
    mock_config.VERBOSE = False 
    # Mock defined_models structure if resolve_model_alias needs it for error messages
    mock_config.defined_models = {
        'models': {
            'gpt4o': {'type': 'openai', 'model_id': 'gpt-4o'},
            'llama3instruct': {'type': 'ollama', 'model_id': 'llama3:instruct'},
            'gemma7b': {'type': 'ollama', 'model_id': 'gemma:7b'},
            'mockhf': {'type': 'huggingface', 'model_id': 'mock-hf-model/pygmalion-3-12b'}
        }
    }
    return mock_config

# --- Tests for parse_arguments --- #
# Focus on argument parsing itself, not model resolution which happens later.

@pytest.fixture
def mock_config_for_parse():
    # Use a simpler config focused just on what parse_arguments needs
    return create_mock_config(default_alias='gpt4o')

def test_parse_arguments_defaults(mock_config_for_parse):
    """Test default argument values using injected mock config."""
    with patch('sys.argv', ['script_name']):
        args = parse_arguments(config_obj=mock_config_for_parse)
    assert args.question == []
    assert args.verbose is False
    # Check against the DEFAULT_MODEL_ALIAS used in the fixture
    assert args.model == mock_config_for_parse.DEFAULT_MODEL_ALIAS
    assert args.delete_history is False
    assert args.print_history is None
    assert args.command is None
    assert args.plain is False
    # Note: The old boolean --refresh-models is gone, the new one takes args
    # Let's test the new --refresh-models argument structure
    assert args.refresh_models is None # Default should be None
    assert args.delete_model is None # Test new flag

def test_parse_arguments_simple_question(mock_config_for_parse):
    """Test parsing a simple question."""
    question = "What is the weather?"
    with patch('sys.argv', ['script_name', question]):
        args = parse_arguments(config_obj=mock_config_for_parse)
    assert args.question == [question]

def test_parse_arguments_flags(mock_config_for_parse):
    """Test parsing boolean flags."""
    with patch('sys.argv', ['script_name', '--verbose', '--plain']):
        args = parse_arguments(config_obj=mock_config_for_parse)
    assert args.verbose is True
    assert args.plain is True

def test_parse_arguments_history_flags(mock_config_for_parse):
    """Test parsing history-related flags."""
    with patch('sys.argv', ['script_name', '--delete-history']):
        args = parse_arguments(config_obj=mock_config_for_parse)
    assert args.delete_history is True
    assert args.print_history is None

    with patch('sys.argv', ['script_name', '--print-history']):
        args = parse_arguments(config_obj=mock_config_for_parse)
    assert args.delete_history is False
    assert args.print_history == -1

    with patch('sys.argv', ['script_name', '--print-history', '5']):
        args = parse_arguments(config_obj=mock_config_for_parse)
    assert args.delete_history is False
    assert args.print_history == 5

def test_parse_arguments_command(mock_config_for_parse):
    """Test parsing the command argument."""
    command = "ls -l"
    with patch('sys.argv', ['script_name', '-c', command]):
        args = parse_arguments(config_obj=mock_config_for_parse)
    assert args.command == command

def test_parse_arguments_refresh_models_choices(mock_config_for_parse):
    """Test parsing the --refresh-models argument with choices."""
    with patch('sys.argv', ['script_name', '--refresh-models', 'openai']):
        args = parse_arguments(config_obj=mock_config_for_parse)
    assert args.refresh_models == 'openai'

    with patch('sys.argv', ['script_name', '--refresh-models', 'ollama']):
        args = parse_arguments(config_obj=mock_config_for_parse)
    assert args.refresh_models == 'ollama'

    # Test invalid choice (argparse handles this)
    with patch('sys.argv', ['script_name', '--refresh-models', 'invalid']):
        with pytest.raises(SystemExit):
            parse_arguments(config_obj=mock_config_for_parse)

def test_parse_arguments_delete_model(mock_config_for_parse):
    """Test parsing the --delete-model flag."""
    alias_to_delete = "some_alias"
    with patch('sys.argv', ['script_name', '--delete-model', alias_to_delete]):
        args = parse_arguments(config_obj=mock_config_for_parse)
    assert args.delete_model == alias_to_delete

# === Tests for main function execution paths ===

# Import main and necessary functions/classes for patching
from ask_llm.cli import main # Import main from cli instead of main.py
from argparse import Namespace

@pytest.fixture
def mock_ask_llm_instance():
    """Provides a mock AskLLM instance for main function tests."""
    mock_instance = MagicMock(name="AskLLM_Instance")
    # Mock nested attributes/methods as needed by main
    mock_instance.history_manager = MagicMock(name="HistoryManager_Instance")
    mock_instance.client = MagicMock(name="Client_Instance")
    mock_instance.client.console = MagicMock(name="Console_Instance")
    return mock_instance

@pytest.fixture
def patch_main_dependencies(mocker, mock_ask_llm_instance):
    """Patches dependencies used directly by the main function."""
    # Patch parse_arguments (now imported from cli)
    mock_parse = mocker.patch('ask_llm.cli.parse_arguments') # Target cli
    
    # Mock config (patch global import in cli if needed, or pass instance)    
    mock_config = create_mock_config()
    mock_config.update_from_args = MagicMock(return_value=mock_config)
    mocker.patch('ask_llm.main.global_config', mock_config) # Removed global patch (use cli.Config)

    # Patch functions called for action flags (target cli module)
    mock_list_models = mocker.patch('ask_llm.cli.list_models')
    mock_add_gguf = mocker.patch('ask_llm.cli.handle_add_gguf')
    # The old update_openai function seems replaced by update_models_interactive
    mock_update_models = mocker.patch('ask_llm.cli.update_models_interactive') 
    mock_delete_model = mocker.patch('ask_llm.cli.delete_model')

    # Patch resolve_model_alias placeholder (will use ModelManager instance mock in tests)
    mock_resolve = MagicMock()

    # Patch run_app (target cli module)
    mock_run_app = mocker.patch('ask_llm.cli.run_app') 
    
    # Patch sys.exit in cli module so calls to sys.exit() in main() are intercepted
    mock_exit = mocker.patch('ask_llm.cli.sys.exit')

    return {
        'parse_arguments': mock_parse,
        'config': mock_config,
        'list_available_models': mock_list_models, # Key name kept for compatibility
        'handle_add_gguf': mock_add_gguf,
        'update_models_interactive': mock_update_models, # New key name
        'delete_model_from_config': mock_delete_model, # Key name kept
        'resolve_model_alias': mock_resolve,
        'run_app': mock_run_app,
        'sys_exit': mock_exit
    }

# Helper function to create a default args Namespace for main tests
def create_default_mock_args(**kwargs):
    """Creates a Namespace with defaults for all args, allowing overrides."""
    defaults = {
        'question': [],
        'model': 'gpt4o', # Default for testing, override as needed
        'list_models': False,
        'add_gguf': None,
        'refresh_models': None,
        'delete_model': None,
        'verbose': False,
        'delete_history': False,
        'print_history': None,
        'command': None,
        'plain': False,
        'no_stream': False
    }
    defaults.update(kwargs)
    return Namespace(**defaults)

def test_main_simple_question(patch_main_dependencies):
    """Test main pathway for a simple question."""
    mock_args = create_default_mock_args(question=["Hello"], model='gpt4o')
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    patch_main_dependencies['resolve_model_alias'].return_value = 'gpt4o'
    
    # Call main (adjust if main's structure changed)
    # We might need to mock Config instantiation if main creates it
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']) as mock_config_init, \
         patch('ask_llm.cli.ModelManager') as mock_model_manager_class: # Patch ModelManager
        
        # Configure the mock ModelManager instance
        mock_manager_instance = mock_model_manager_class.return_value
        mock_manager_instance.resolve_model_alias.return_value = 'gpt4o'
        # Re-patch the specific mock instance might be more robust here
        patch_main_dependencies['resolve_model_alias'] = mock_manager_instance.resolve_model_alias

        main() # Call the cli.main directly

    patch_main_dependencies['parse_arguments'].assert_called_once()
    # Check that resolve_model_alias on the model_manager instance was called
    mock_manager_instance.resolve_model_alias.assert_called_once_with(mock_args.model)
    # Assert run_app was called with positional args
    patch_main_dependencies['run_app'].assert_called_once_with(mock_args, patch_main_dependencies['config'], 'gpt4o')
    patch_main_dependencies['sys_exit'].assert_called_once_with(0)

def test_main_list_models(patch_main_dependencies):
    # Use helper to create args, override list_models
    mock_args = create_default_mock_args(list_models=True)
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    
    # Reset sys_exit mock to clear any previous calls
    patch_main_dependencies['sys_exit'].reset_mock()
    
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']): # Mock Config instantiation
        main() # Call the cli.main directly
    patch_main_dependencies['list_available_models'].assert_called_once_with(patch_main_dependencies['config'])
    
    # Assert sys_exit was called with 0
    patch_main_dependencies['sys_exit'].assert_called_with(0)

def test_main_add_gguf(patch_main_dependencies):
    repo_id = "some/repo"
    # Use helper to create args, override add_gguf
    mock_args = create_default_mock_args(add_gguf=repo_id)
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    patch_main_dependencies['handle_add_gguf'].return_value = True # Assume success
    
    # Reset sys_exit mock to clear any previous calls
    patch_main_dependencies['sys_exit'].reset_mock()
    
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']): # Mock Config instantiation
        main() # Call the cli.main directly
    patch_main_dependencies['handle_add_gguf'].assert_called_once_with(repo_id, patch_main_dependencies['config'])
    
    # Assert sys_exit was called with 0
    patch_main_dependencies['sys_exit'].assert_called_with(0)

def test_main_refresh_openai(patch_main_dependencies):
    # Use helper, override refresh_models
    mock_args = create_default_mock_args(refresh_models='openai')
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    patch_main_dependencies['update_models_interactive'].return_value = True
    
    # Reset sys_exit mock to clear any previous calls
    patch_main_dependencies['sys_exit'].reset_mock()
    
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']): # Mock Config instantiation
        main() # Call the cli.main directly
    patch_main_dependencies['update_models_interactive'].assert_called_once_with(patch_main_dependencies['config'], provider='openai')
    
    # Assert sys_exit was called with 0
    patch_main_dependencies['sys_exit'].assert_called_with(0)

def test_main_refresh_ollama(patch_main_dependencies):
    # Use helper, override refresh_models
    mock_args = create_default_mock_args(refresh_models='ollama')
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    patch_main_dependencies['update_models_interactive'].return_value = True # Assume success
    
    # Reset sys_exit mock to clear any previous calls
    patch_main_dependencies['sys_exit'].reset_mock()
    
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']): # Mock Config instantiation
        main() # Call the cli.main directly
    patch_main_dependencies['update_models_interactive'].assert_called_once_with(patch_main_dependencies['config'], provider='ollama')
    
    # Assert sys_exit was called with 0
    patch_main_dependencies['sys_exit'].assert_called_with(0)

def test_main_delete_model(patch_main_dependencies):
    alias = "model_to_delete"
    # Use helper, override delete_model
    mock_args = create_default_mock_args(delete_model=alias)
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    patch_main_dependencies['delete_model_from_config'].return_value = True
    
    # Reset sys_exit mock to clear any previous calls
    patch_main_dependencies['sys_exit'].reset_mock()
    
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']): # Mock Config instantiation
        main() # Call the cli.main directly
    patch_main_dependencies['delete_model_from_config'].assert_called_once_with(alias, patch_main_dependencies['config'])
    
    # Assert sys_exit was called with 0
    patch_main_dependencies['sys_exit'].assert_called_with(0)

def test_main_resolve_alias_fails(patch_main_dependencies):
    """Test main pathway when resolve_model_alias returns None."""
    mock_args = create_default_mock_args(model='bad-alias')
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    
    # Patch ModelManager constructor to return our specific mock instance
    mock_manager_instance = MagicMock()
    mock_manager_instance.resolve_model_alias.return_value = None
    
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']), \
         patch('ask_llm.cli.ModelManager', return_value=mock_manager_instance) as mock_model_manager_ctor: # Patch constructor
        
        main() # Call the cli.main directly
        
    # Assertions *after* main() is called
    mock_model_manager_ctor.assert_called_once_with(patch_main_dependencies['config']) # Check manager init
    mock_manager_instance.resolve_model_alias.assert_called_once_with('bad-alias') # Check resolve call
    patch_main_dependencies['run_app'].assert_not_called() # Should pass now
    patch_main_dependencies['sys_exit'].assert_called_once_with(1)

# --- Tests for other main paths (delete_history, print_history, command, interactive) ---
# These tests primarily assert that run_app is called correctly, so the patching 
# of run_app should be sufficient if main() logic correctly passes args/config.
# We might need to add the Config instantiation mock as done in other tests.

def test_main_delete_history_path(patch_main_dependencies):
    """Test main pathway when --delete-history is used."""
    mock_args = create_default_mock_args(delete_history=True)
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    # patch_main_dependencies['resolve_model_alias\'].return_value = \'gpt4o\' # Don't preset globally
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']), \
         patch('ask_llm.cli.ModelManager') as mock_model_manager_class:
        mock_manager_instance = mock_model_manager_class.return_value
        mock_manager_instance.resolve_model_alias.return_value = 'gpt4o' # Mock resolve for this path
        # patch_main_dependencies['resolve_model_alias\'] = mock_manager_instance.resolve_model_alias # Not needed
        main() # Call the cli.main directly
    mock_manager_instance.resolve_model_alias.assert_called_once_with(mock_args.model)
    # Assert run_app was called with positional args
    patch_main_dependencies['run_app'].assert_called_once_with(mock_args, patch_main_dependencies['config'], 'gpt4o')
    patch_main_dependencies['sys_exit'].assert_called_once_with(0)

def test_main_print_history_path(patch_main_dependencies):
    """Test main pathway when --print-history is used."""
    mock_args = create_default_mock_args(print_history=5)
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    # patch_main_dependencies['resolve_model_alias\'].return_value = \'gpt4o\'
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']), \
         patch('ask_llm.cli.ModelManager') as mock_model_manager_class:
        mock_manager_instance = mock_model_manager_class.return_value
        mock_manager_instance.resolve_model_alias.return_value = 'gpt4o'
        # patch_main_dependencies['resolve_model_alias\'] = mock_manager_instance.resolve_model_alias
        main() # Call the cli.main directly
    mock_manager_instance.resolve_model_alias.assert_called_once_with(mock_args.model)
    # Assert run_app was called with positional args
    patch_main_dependencies['run_app'].assert_called_once_with(mock_args, patch_main_dependencies['config'], 'gpt4o')
    patch_main_dependencies['sys_exit'].assert_called_once_with(0)

def test_main_command_path(patch_main_dependencies):
    """Test main pathway when -c command is used."""
    mock_args = create_default_mock_args(command="echo hello")
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    # patch_main_dependencies['resolve_model_alias\'].return_value = \'gpt4o\'
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']), \
         patch('ask_llm.cli.ModelManager') as mock_model_manager_class:
        mock_manager_instance = mock_model_manager_class.return_value
        mock_manager_instance.resolve_model_alias.return_value = 'gpt4o'
        # patch_main_dependencies['resolve_model_alias\'] = mock_manager_instance.resolve_model_alias
        main() # Call the cli.main directly
    mock_manager_instance.resolve_model_alias.assert_called_once_with(mock_args.model)
    # Assert run_app was called with positional args
    patch_main_dependencies['run_app'].assert_called_once_with(mock_args, patch_main_dependencies['config'], 'gpt4o')
    patch_main_dependencies['sys_exit'].assert_called_once_with(0)

def test_main_interactive_path(patch_main_dependencies):
    """Test main pathway when no question or command leads to interactive."""
    mock_args = create_default_mock_args()
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    # patch_main_dependencies['resolve_model_alias\'].return_value = \'gpt4o\'
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']), \
         patch('ask_llm.cli.ModelManager') as mock_model_manager_class:
        mock_manager_instance = mock_model_manager_class.return_value
        mock_manager_instance.resolve_model_alias.return_value = 'gpt4o'
        # patch_main_dependencies['resolve_model_alias\'] = mock_manager_instance.resolve_model_alias
        main() # Call the cli.main directly
    mock_manager_instance.resolve_model_alias.assert_called_once_with(mock_args.model)
    # Assert run_app was called with positional args
    patch_main_dependencies['run_app'].assert_called_once_with(mock_args, patch_main_dependencies['config'], 'gpt4o')
    patch_main_dependencies['sys_exit'].assert_called_once_with(0)

def test_main_run_app_exception(patch_main_dependencies):
    """Test main handling when run_app raises an exception."""
    mock_args = create_default_mock_args(question=["Hi"])
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    patch_main_dependencies['resolve_model_alias'].return_value = 'gpt4o'
    patch_main_dependencies['run_app'].side_effect = Exception("Run app failed")
    # Mock config VERBOSE for the try/except block in main
    patch_main_dependencies['config'].VERBOSE = False 
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']), \
         patch('ask_llm.cli.ModelManager') as mock_model_manager_class, \
         patch('traceback.print_exc') as mock_traceback:
        mock_manager_instance = mock_model_manager_class.return_value
        mock_manager_instance.resolve_model_alias.return_value = 'gpt4o'
        patch_main_dependencies['resolve_model_alias'] = mock_manager_instance.resolve_model_alias
        main() # Call the cli.main directly
    patch_main_dependencies['run_app'].assert_called_once()
    patch_main_dependencies['sys_exit'].assert_called_once_with(1)
    mock_traceback.assert_not_called()

def test_main_run_app_exception_verbose(patch_main_dependencies):
    """Test main handling when run_app raises an exception in verbose mode."""
    mock_args = create_default_mock_args(question=["Hi"], verbose=True)
    patch_main_dependencies['parse_arguments'].return_value = mock_args
    patch_main_dependencies['resolve_model_alias'].return_value = 'gpt4o'
    patch_main_dependencies['run_app'].side_effect = Exception("Run app failed")
    patch_main_dependencies['config'].VERBOSE = True # Ensure config mock reflects verbose
    with patch('ask_llm.cli.Config', return_value=patch_main_dependencies['config']), \
         patch('ask_llm.cli.ModelManager') as mock_model_manager_class, \
         patch('traceback.print_exc') as mock_traceback:
        mock_manager_instance = mock_model_manager_class.return_value
        mock_manager_instance.resolve_model_alias.return_value = 'gpt4o'
        patch_main_dependencies['resolve_model_alias'] = mock_manager_instance.resolve_model_alias
        main() # Call the cli.main directly
    patch_main_dependencies['run_app'].assert_called_once()
    patch_main_dependencies['sys_exit'].assert_called_once_with(1)
    mock_traceback.assert_called_once() 