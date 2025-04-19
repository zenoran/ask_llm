import pytest
from unittest.mock import patch, MagicMock, call
from rich.console import Console

# Import AskLLM from the correct location
from src.ask_llm.core import AskLLM

# Import things that might be needed for mocking config/clients
from src.ask_llm.utils.config import Config
from src.ask_llm.clients import OpenAIClient, OllamaClient

# Try to import HuggingFaceClient, but mock it if unavailable
try:
    from src.ask_llm.clients import HuggingFaceClient
except ImportError:
    HuggingFaceClient = MagicMock()  # Mock the HuggingFaceClient for tests

# Import the actual client classes to fix isinstance checks
from src.ask_llm.clients.huggingface_client import HuggingFaceClient as ActualHFClient


# Mock client classes to prevent actual client initialization
@pytest.fixture
def mock_clients():
    # Patch targets should be where AskLLM (in core.py) looks them up
    with patch('src.ask_llm.core.OpenAIClient', autospec=True) as mock_openai, \
         patch('src.ask_llm.core.OllamaClient', autospec=True) as mock_ollama, \
         patch('src.ask_llm.core.is_huggingface_available', return_value=True), \
         patch('src.ask_llm.core.HuggingFaceClient', autospec=True) as mock_hf:
        yield {
            "openai": mock_openai,
            "ollama": mock_ollama,
            "huggingface": mock_hf,
        }

# Mock HistoryManager
@pytest.fixture
def mock_history_manager():
    # Patch where AskLLM (in core.py) looks it up
    with patch('src.ask_llm.core.HistoryManager', autospec=True) as mock_hm:
        # Mock the instance returned by the constructor
        mock_instance = mock_hm.return_value
        # Add the remove_last_message_if_partial method
        mock_instance.remove_last_message_if_partial = MagicMock()
        yield mock_instance # Yield the mock instance for assertions

# Mock the Config class used by AskLLM
@pytest.fixture
def mock_global_config():
    # Create a mock config instance for AskLLM to use
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
    
    # Add mock defined_models structure needed by initialize_client
    mock_config.defined_models = {
        'models': {
            'mock-default-openai': {'type': 'openai', 'model_id': 'mock-default-openai'},
            'gpt-4o': {'type': 'openai', 'model_id': 'gpt-4o'},
            'mock-ollama': {'type': 'ollama', 'model_id': 'mock-ollama'},
            'mock-hf': {'type': 'huggingface', 'model_id': 'mock-hf'},
        }
    }
    
    return mock_config


# === Tests for AskLLM ===

def test_ask_llm_init(mock_global_config, mock_clients, mock_history_manager):
    """Test AskLLM initialization."""
    # Patch initialize_client and load_history within the test scope
    with patch.object(AskLLM, 'initialize_client', return_value=MagicMock()) as mock_init_client, \
         patch.object(AskLLM, 'load_history') as mock_load_hist, \
         patch('src.ask_llm.core.HistoryManager') as mock_hm_class:
        
        # Setup the mock_hm_class to return mock_history_manager
        mock_hm_class.return_value = mock_history_manager

        instance = AskLLM(resolved_model_alias=mock_global_config.DEFAULT_MODEL_ALIAS, config=mock_global_config)

        # Check methods were called
        mock_init_client.assert_called_once()
        # Check load_history was called on the instance
        mock_load_hist.assert_called_once()
        # Check HistoryManager was called with the client from initialize_client
        mock_hm_class.assert_called_once_with(client=mock_init_client.return_value)


@pytest.mark.parametrize(
    "model_id, expected_client_mock_key, expected_client_class",
    [
        ("gpt-4o", "openai", OpenAIClient),
        ("mock-ollama", "ollama", OllamaClient),
        ("mock-hf", "huggingface", HuggingFaceClient),
    ]
)
def test_ask_llm_initialize_client(
    mock_global_config, mock_clients, mock_history_manager, # Need history manager fixture to run init
    model_id, expected_client_mock_key, expected_client_class
):
    """Test that the correct client is initialized based on model_id."""
    # Set the resolved model alias directly on the instance for this test
    # as initialize_client takes it as an argument now
    resolved_alias = model_id # Using the param alias as the resolved one
    mock_global_config.DEFAULT_MODEL_ALIAS = resolved_alias # Ensure config default matches
    
    # Override the model definition for access in initialize_client
    mock_global_config.model_definition = mock_global_config.defined_models['models'][resolved_alias]

    # Special handling for OllamaClient which has a different constructor
    if expected_client_mock_key == "ollama":
        # Patch the OllamaClient constructor directly
        with patch('src.ask_llm.core.OllamaClient') as patched_ollama:
            # Instantiate AskLLM - this calls initialize_client internally during __init__
            with patch.object(AskLLM, 'load_history'): # Still mock load_history
                # Pass the resolved alias to init
                instance = AskLLM(resolved_model_alias=resolved_alias, config=mock_global_config)
            
            # Check if OllamaClient was initialized correctly
            patched_ollama.assert_called_once_with(model_name=model_id, config=mock_global_config)
            # Set the client to the mock return value for the assertion below
            instance.client = patched_ollama.return_value
    else:
        # For other client types
        # Instantiate AskLLM - this calls initialize_client internally during __init__
        with patch.object(AskLLM, 'load_history'): # Still mock load_history
            # Pass the resolved alias to init
            instance = AskLLM(resolved_model_alias=resolved_alias, config=mock_global_config)

        # Get the mock for the expected client class
        mock_client_class = mock_clients[expected_client_mock_key]

        # Assert that the expected client class was instantiated 
        # Check for the specific client type's initialization logic
        if expected_client_mock_key == "openai":
            # OpenAIClient(model_id)
            mock_client_class.assert_called_once_with(model_id)
        elif expected_client_mock_key == "huggingface":
            # HuggingFaceClient(model_id=model_id)
            mock_client_class.assert_called_once_with(model_id=model_id)

    # Assert that the instance's client attribute is the instance of the correct client
    if expected_client_mock_key != "ollama":
        assert instance.client == mock_clients[expected_client_mock_key].return_value


def test_ask_llm_initialize_client_unknown(mock_global_config, mock_clients, mock_history_manager):
    """Test initialization with an unknown model raises ValueError."""
    unknown_alias = "unknown-model"
    # Ensure the alias is not in defined_models
    if unknown_alias in mock_global_config.defined_models['models']:
        del mock_global_config.defined_models['models'][unknown_alias]

    with pytest.raises(ValueError, match="Could not find model definition for resolved alias"):
        with patch.object(AskLLM, 'load_history'):
             # Pass the unknown alias
             instance = AskLLM(resolved_model_alias=unknown_alias, config=mock_global_config)

def test_ask_llm_query_simple(
    mock_global_config, mock_clients, mock_history_manager
):
    """Test a simple query flow."""
    prompt = "Hello, world!"
    expected_response = "Hi there!"
    # Mock context messages from history manager
    initial_context = [{"role": "user", "content": "previous prompt"}]
    # Expected messages passed to client.query (includes the new prompt)
    complete_context_messages = initial_context + [{"role": "user", "content": prompt}]

    # --- Setup Mocks ---
    # Ensure AskLLM initializes with a HuggingFace model ID
    mock_global_config.DEFAULT_MODEL_ALIAS = "mock-hf"
    # Get the mock HF Client *class* and its *return_value* (the instance)
    mock_hf_client_class = mock_clients["huggingface"]
    mock_client_instance = mock_hf_client_class.return_value
    # Configure the mock instance *after* AskLLM initializes it
    mock_client_instance.query.return_value = expected_response
    mock_client_instance.console = MagicMock()

    # Mock the history manager methods
    mock_history_manager.get_context_messages.return_value = complete_context_messages
    mock_history_manager.get_last_assistant_message.return_value = None

    # --- Test --- #
    with patch.object(AskLLM, 'load_history'):
        instance = AskLLM(resolved_model_alias="mock-hf", config=mock_global_config)
        instance.history_manager = mock_history_manager

    # For the query tests, update the mock_isinstance function
    def mock_isinstance_fixed(obj, classinfo):
        """A fixed mock for isinstance that handles tuples and mock objects correctly."""
        # Always return False for streaming_clients check to avoid stream parameter
        if isinstance(classinfo, tuple):
            return False
        # For other isinstance checks, use the real isinstance
        import builtins
        return builtins.isinstance(obj, classinfo)

    with patch('src.ask_llm.core.isinstance', side_effect=mock_isinstance_fixed):
        response = instance.query(prompt)

    # --- Assertions --- #
    assert response == expected_response
    # Verify AskLLM initialized the correct (mocked) client class
    mock_hf_client_class.assert_called_once_with(model_id="mock-hf")
    # Verify the instance created is the one we configured
    assert instance.client == mock_client_instance

    # Check history manager calls
    mock_history_manager.add_message.assert_any_call("user", prompt)
    mock_history_manager.get_context_messages.assert_called_once()
    mock_history_manager.get_last_assistant_message.assert_called_once()
    mock_history_manager.add_message.assert_any_call("assistant", expected_response)
    assert mock_history_manager.add_message.call_count == 2

    # Check client query call - now without stream parameter
    mock_client_instance.query.assert_called_once_with(
        messages=complete_context_messages,
        plaintext_output=False
    )

def test_ask_llm_query_duplicate_response(
    mock_global_config, mock_clients, mock_history_manager
):
    """Test query flow when a duplicate response is initially detected."""
    prompt = "Tell me a joke."
    duplicate_response = "Why did the chicken cross the road?"
    final_response = "To get to the other side!"
    complete_context_messages = [{"role": "user", "content": prompt}]

    # --- Setup Mocks ---
    # Ensure AskLLM initializes with a HuggingFace model ID
    mock_global_config.DEFAULT_MODEL_ALIAS = "mock-hf"
    mock_global_config.allow_duplicate_response = False
    # Get the mock HF Client *class* and its *return_value* (the instance)
    mock_hf_client_class = mock_clients["huggingface"]
    mock_client_instance = mock_hf_client_class.return_value
    # Configure the mock instance *after* AskLLM initializes it
    mock_client_instance.query.side_effect = [duplicate_response, final_response]
    
    # Set up console directly on the instance (the core uses the global console)
    console_mock = MagicMock()
    with patch('src.ask_llm.core.console', console_mock):
        mock_history_manager.get_context_messages.return_value = complete_context_messages
        mock_history_manager.get_last_assistant_message.return_value = duplicate_response
    
        # --- Test --- #
        with patch.object(AskLLM, 'load_history'):
            instance = AskLLM(resolved_model_alias="mock-hf", config=mock_global_config)
            instance.history_manager = mock_history_manager
    
        # For the query tests, update the mock_isinstance function
        def mock_isinstance_fixed(obj, classinfo):
            """A fixed mock for isinstance that handles tuples and mock objects correctly."""
            # Always return False for streaming_clients check
            if isinstance(classinfo, tuple):
                return False
            # For other isinstance checks, use the real isinstance
            import builtins
            return builtins.isinstance(obj, classinfo)
    
        with patch('src.ask_llm.core.isinstance', side_effect=mock_isinstance_fixed):
            response = instance.query(prompt)
    
    # --- Assertions --- #
    assert response == final_response
    # Verify AskLLM initialized the correct (mocked) client class
    mock_hf_client_class.assert_called_once_with(model_id="mock-hf")
    # Verify the instance created is the one we configured
    assert instance.client == mock_client_instance

    # Check history manager calls
    mock_history_manager.add_message.assert_any_call("user", prompt)
    mock_history_manager.get_context_messages.assert_called()
    mock_history_manager.get_last_assistant_message.assert_called_once()
    mock_history_manager.add_message.assert_any_call("assistant", final_response)
    assert mock_history_manager.add_message.call_count == 2 # User prompt, final assistant response

    # Check client query calls
    assert mock_client_instance.query.call_count == 2
    # Check the console call for the duplicate response
    console_mock.print.assert_called_once_with(
        "[yellow]Detected duplicate response. Regenerating with higher temperature...[/yellow]"
    )

def test_ask_llm_query_keyboard_interrupt(
    mock_global_config, mock_clients, mock_history_manager
):
    """Test that KeyboardInterrupt during query is handled."""
    prompt = "Infinite loop prompt"
    complete_context_messages = [{"role": "user", "content": prompt}]

    # --- Setup Mocks ---
    # Ensure AskLLM initializes with a HuggingFace model ID
    mock_global_config.DEFAULT_MODEL_ALIAS = "mock-hf"
    # Get the mock HF Client *class* and its *return_value* (the instance)
    mock_hf_client_class = mock_clients["huggingface"]
    mock_client_instance = mock_hf_client_class.return_value
    # Configure the mock instance *after* AskLLM initializes it
    mock_client_instance.query.side_effect = KeyboardInterrupt
    
    # Set up console directly on the instance (the core uses the global console)
    console_mock = MagicMock()
    with patch('src.ask_llm.core.console', console_mock):
        mock_history_manager.get_context_messages.return_value = complete_context_messages
    
        # --- Test --- #
        with patch.object(AskLLM, 'load_history'):
            instance = AskLLM(resolved_model_alias="mock-hf", config=mock_global_config)
            instance.history_manager = mock_history_manager
    
        # For the query tests, update the mock_isinstance function
        def mock_isinstance_fixed(obj, classinfo):
            """A fixed mock for isinstance that handles tuples and mock objects correctly."""
            # Always return False for streaming_clients check
            if isinstance(classinfo, tuple):
                return False
            # For other isinstance checks, use the real isinstance
            import builtins
            return builtins.isinstance(obj, classinfo)
    
        with patch('src.ask_llm.core.isinstance', side_effect=mock_isinstance_fixed):
            response = instance.query(prompt)
    
    # --- Assertions --- #
    assert response == "" # Query should return empty string on interrupt
    # Verify AskLLM initialized the correct (mocked) client class
    mock_hf_client_class.assert_called_once_with(model_id="mock-hf")
    # Verify the instance created is the one we configured
    assert instance.client == mock_client_instance

    # Check history manager calls
    mock_history_manager.add_message.assert_called_once_with("user", prompt)
    mock_history_manager.get_context_messages.assert_called_once()
    mock_history_manager.remove_last_message_if_partial.assert_called_once_with("assistant")

    # Check client query call
    mock_client_instance.query.assert_called_once_with(
        messages=complete_context_messages,
        plaintext_output=False
    )
    # Check console message through mocked console - without the newline
    console_mock.print.assert_called_once_with(
        "[bold red]Query interrupted.[/bold red]"
    ) 