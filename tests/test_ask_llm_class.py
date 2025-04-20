import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import contextlib

from ask_llm.core import AskLLM
from ask_llm.models.message import Message
from ask_llm.utils.config import Config
from ask_llm.clients import OpenAIClient, OllamaClient

# Check if Hugging Face dependencies are available
# We need the base import to check availability
try:
    from ask_llm.clients import HuggingFaceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HuggingFaceClient = MagicMock() # Mock the base class if import fails
    HUGGINGFACE_AVAILABLE = False

# Define ActualHFClient as MagicMock if dependencies are missing
# The tests will use this mock object
if not HUGGINGFACE_AVAILABLE:
    ActualHFClient = MagicMock()
else:
    # Only import the implementation if needed and available (less crucial now)
    # This might still fail if HuggingFaceClient import itself failed above
    try:
        from ask_llm.clients.huggingface_client import HuggingFaceClient as ActualHFClient
    except ImportError:
        # Fallback if the implementation import fails for some reason
        ActualHFClient = MagicMock()

@pytest.fixture
def mock_clients():
    patches = [
        patch('ask_llm.core.OpenAIClient', autospec=True),
        patch('ask_llm.core.OllamaClient', autospec=True),
        patch('ask_llm.core.is_huggingface_available', return_value=HUGGINGFACE_AVAILABLE)
    ]
    # Only patch HuggingFaceClient if dependencies might be available
    if HUGGINGFACE_AVAILABLE:
        patches.append(patch('ask_llm.core.HuggingFaceClient', autospec=True))

    with contextlib.ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in patches]
        mock_map = {
            "openai": mocks[0],
            "ollama": mocks[1],
            # Use MagicMock directly if HF not available, otherwise use the patched mock
            "huggingface": mocks[3] if HUGGINGFACE_AVAILABLE else MagicMock(spec=HuggingFaceClient if HUGGINGFACE_AVAILABLE else MagicMock())
        }
        yield mock_map

@pytest.fixture
def mock_history_manager():
    with patch('ask_llm.core.HistoryManager', autospec=True) as mock_hm:
        mock_instance = mock_hm.return_value
        mock_instance.remove_last_message_if_partial = MagicMock()
        yield mock_instance # Yield the mock instance for assertions

@pytest.fixture
def mock_global_config():
    mock_config = MagicMock(spec=Config)
    mock_config.DEFAULT_MODEL_ALIAS = "mock-default-openai" # Needs to match a key in defined_models
    mock_config.TEMPERATURE = 0.8 
    mock_config.HISTORY_FILE = '/tmp/fake_history.jsonl' # Needed by HistoryManager
    mock_config.HISTORY_DURATION = 600
    mock_config.MAX_TOKENS = 1024
    mock_config.allow_duplicate_response = False
    mock_config.VERBOSE = False
    mock_config.available_ollama_models = ["mock-ollama"]  # Add available Ollama models
    mock_config.OLLAMA_URL = "http://localhost:11434"  # Add OLLAMA_URL
    mock_config.defined_models = {
        'models': {
            'mock-default-openai': {'type': 'openai', 'model_id': 'mock-default-openai'},
            'gpt-4o': {'type': 'openai', 'model_id': 'gpt-4o'},
            'mock-ollama': {'type': 'ollama', 'model_id': 'mock-ollama'},
            'mock-hf': {'type': 'huggingface', 'model_id': 'mock-hf'},
        }
    }
    
    return mock_config



def test_ask_llm_init(mock_global_config, mock_clients, mock_history_manager):
    """Test AskLLM initialization."""
    with patch.object(AskLLM, 'initialize_client', return_value=MagicMock()) as mock_init_client, \
         patch.object(AskLLM, 'load_history') as mock_load_hist, \
         patch('ask_llm.core.HistoryManager') as mock_hm_class:
        mock_hm_class.return_value = mock_history_manager

        instance = AskLLM(resolved_model_alias=mock_global_config.DEFAULT_MODEL_ALIAS, config=mock_global_config)
        mock_init_client.assert_called_once()
        mock_load_hist.assert_called_once()
        mock_hm_class.assert_called_once_with(client=mock_init_client.return_value, config=mock_global_config)
        assert instance.resolved_model_alias == mock_global_config.DEFAULT_MODEL_ALIAS


@pytest.mark.parametrize(
    "model_id, expected_client_mock_key, expected_client_class",
    [
        ("gpt-4o", "openai", OpenAIClient),
        ("mock-ollama", "ollama", OllamaClient),
        # Parametrize HF test conditionally
        pytest.param(
            "mock-hf", "huggingface", HuggingFaceClient,
            marks=pytest.mark.skipif(not HUGGINGFACE_AVAILABLE, reason="requires huggingface dependencies")
        ),
    ]
)
def test_ask_llm_initialize_client(
    mock_global_config, mock_clients, mock_history_manager, # Need history manager fixture to run init
    model_id, expected_client_mock_key, expected_client_class
):
    """Test that the correct client is initialized based on model_id."""
    resolved_alias = model_id # Using the param alias as the resolved one
    mock_global_config.DEFAULT_MODEL_ALIAS = resolved_alias # Ensure config default matches
    mock_global_config.model_definition = mock_global_config.defined_models['models'][resolved_alias]
    if expected_client_mock_key == "ollama":
        with patch('ask_llm.core.OllamaClient') as patched_ollama:
            with patch.object(AskLLM, 'load_history'): # Still mock load_history
                instance = AskLLM(resolved_model_alias=resolved_alias, config=mock_global_config)
            patched_ollama.assert_called_once_with(model=model_id, config=mock_global_config)
            instance.client = patched_ollama.return_value
    else:
        with patch.object(AskLLM, 'load_history'): # Still mock load_history
            instance = AskLLM(resolved_model_alias=resolved_alias, config=mock_global_config)
        mock_client_class = mock_clients[expected_client_mock_key]
        if expected_client_mock_key == "openai":
            mock_client_class.assert_called_once_with(model_id, config=mock_global_config)
        elif expected_client_mock_key == "huggingface":
            mock_client_class.assert_called_once_with(model_id=model_id, config=mock_global_config)
    if expected_client_mock_key != "ollama":
        assert instance.client == mock_clients[expected_client_mock_key].return_value


def test_ask_llm_initialize_client_unknown(mock_global_config, mock_clients, mock_history_manager):
    """Test initialization with an unknown model raises ValueError."""
    unknown_alias = "unknown-model"
    if unknown_alias in mock_global_config.defined_models['models']:
        del mock_global_config.defined_models['models'][unknown_alias]

    with pytest.raises(ValueError, match="Could not find model definition for resolved alias"):
        with patch.object(AskLLM, 'load_history'):
             instance = AskLLM(resolved_model_alias=unknown_alias, config=mock_global_config)

@pytest.mark.skipif(not HUGGINGFACE_AVAILABLE, reason="requires huggingface dependencies")
def test_ask_llm_query_simple(
    mock_global_config, mock_clients, mock_history_manager
):
    """Test a simple query flow."""
    prompt = "Hello, world!"
    expected_response = "Hi there!"
    initial_context = [{"role": "user", "content": "previous prompt"}]
    complete_context_messages = initial_context + [{"role": "user", "content": prompt}]
    mock_global_config.DEFAULT_MODEL_ALIAS = "mock-hf"
    mock_hf_client_class = mock_clients["huggingface"]
    mock_client_instance = mock_hf_client_class.return_value
    mock_client_instance.query.return_value = expected_response
    mock_client_instance.console = MagicMock()
    mock_history_manager.get_context_messages.return_value = complete_context_messages
    mock_history_manager.get_last_assistant_message.return_value = None
    with patch.object(AskLLM, 'load_history'):
        instance = AskLLM(resolved_model_alias="mock-hf", config=mock_global_config)
        instance.history_manager = mock_history_manager
    def mock_isinstance_fixed(obj, classinfo):
        """A fixed mock for isinstance that handles tuples and mock objects correctly."""
        if isinstance(classinfo, tuple):
            return False
        import builtins
        return builtins.isinstance(obj, classinfo)

    with patch('ask_llm.core.isinstance', side_effect=mock_isinstance_fixed):
        response = instance.query(prompt)
    assert response == expected_response
    mock_hf_client_class.assert_called_once_with(model_id="mock-hf", config=mock_global_config)
    assert instance.client == mock_client_instance
    mock_history_manager.add_message.assert_any_call("user", prompt)
    mock_history_manager.get_context_messages.assert_called_once()
    mock_history_manager.get_last_assistant_message.assert_called_once()
    mock_history_manager.add_message.assert_any_call("assistant", expected_response)
    assert mock_history_manager.add_message.call_count == 2
    mock_client_instance.query.assert_called_once_with(
        messages=complete_context_messages,
        plaintext_output=False
    )

@pytest.mark.skipif(not HUGGINGFACE_AVAILABLE, reason="requires huggingface dependencies")
def test_ask_llm_query_duplicate_response(
    mock_global_config, mock_clients, mock_history_manager
):
    """Test query flow when a duplicate response is initially detected."""
    prompt = "Tell me a joke."
    duplicate_response = "Why did the chicken cross the road?"
    final_response = "To get to the other side!"
    complete_context_messages = [{"role": "user", "content": prompt}]
    mock_global_config.DEFAULT_MODEL_ALIAS = "mock-hf"
    mock_global_config.allow_duplicate_response = False
    mock_hf_client_class = mock_clients["huggingface"]
    mock_client_instance = mock_hf_client_class.return_value
    mock_client_instance.query.side_effect = [duplicate_response, final_response]
    console_mock = MagicMock()
    with patch('ask_llm.core.console', console_mock):
        mock_history_manager.get_context_messages.return_value = complete_context_messages
        mock_history_manager.get_last_assistant_message.return_value = duplicate_response
        with patch.object(AskLLM, 'load_history'):
            instance = AskLLM(resolved_model_alias="mock-hf", config=mock_global_config)
            instance.history_manager = mock_history_manager
        def mock_isinstance_fixed(obj, classinfo):
            """A fixed mock for isinstance that handles tuples and mock objects correctly."""
            if isinstance(classinfo, tuple):
                return False
            import builtins
            return builtins.isinstance(obj, classinfo)
    
        with patch('ask_llm.core.isinstance', side_effect=mock_isinstance_fixed):
            response = instance.query(prompt)
    assert response == final_response
    mock_hf_client_class.assert_called_once_with(model_id="mock-hf", config=mock_global_config)
    assert instance.client == mock_client_instance
    mock_history_manager.add_message.assert_any_call("user", prompt)
    mock_history_manager.get_context_messages.assert_called()
    mock_history_manager.get_last_assistant_message.assert_called_once()
    mock_history_manager.add_message.assert_any_call("assistant", final_response)
    assert mock_history_manager.add_message.call_count == 2 # User prompt, final assistant response
    assert mock_client_instance.query.call_count == 2
    console_mock.print.assert_called_once_with(
        "[yellow]Detected duplicate response. Regenerating with higher temperature...[/yellow]"
    )

@pytest.mark.skipif(not HUGGINGFACE_AVAILABLE, reason="requires huggingface dependencies")
def test_ask_llm_query_keyboard_interrupt(
    mock_global_config, mock_clients, mock_history_manager
):
    """Test that KeyboardInterrupt during query is handled."""
    prompt = "Infinite loop prompt"
    complete_context_messages = [{"role": "user", "content": prompt}]
    mock_global_config.DEFAULT_MODEL_ALIAS = "mock-hf"
    mock_hf_client_class = mock_clients["huggingface"]
    mock_client_instance = mock_hf_client_class.return_value
    mock_client_instance.query.side_effect = KeyboardInterrupt
    console_mock = MagicMock()
    with patch('ask_llm.core.console', console_mock):
        mock_history_manager.get_context_messages.return_value = complete_context_messages
        with patch.object(AskLLM, 'load_history'):
            instance = AskLLM(resolved_model_alias="mock-hf", config=mock_global_config)
            instance.history_manager = mock_history_manager
        def mock_isinstance_fixed(obj, classinfo):
            """A fixed mock for isinstance that handles tuples and mock objects correctly."""
            if isinstance(classinfo, tuple):
                return False
            import builtins
            return builtins.isinstance(obj, classinfo)
    
        with patch('ask_llm.core.isinstance', side_effect=mock_isinstance_fixed):
            response = instance.query(prompt)
    assert response == "" # Query should return empty string on interrupt
    mock_hf_client_class.assert_called_once_with(model_id="mock-hf", config=mock_global_config)
    assert instance.client == mock_client_instance
    mock_history_manager.add_message.assert_called_once_with("user", prompt)
    mock_history_manager.get_context_messages.assert_called_once()
    mock_history_manager.remove_last_message_if_partial.assert_called_once_with("assistant")
    mock_client_instance.query.assert_called_once_with(
        messages=complete_context_messages,
        plaintext_output=False
    )
    console_mock.print.assert_called_once_with(
        "[bold red]Query interrupted.[/bold red]"
    ) 