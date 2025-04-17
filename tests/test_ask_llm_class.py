import pytest
from unittest.mock import patch, MagicMock, call

# Import the class to test
from src.ask_llm.main import AskLLM

# Import things that might be needed for mocking config/clients
from src.ask_llm.utils.config import Config
from src.ask_llm.clients import OpenAIClient, OllamaClient

# Try to import HuggingFaceClient, but mock it if unavailable
try:
    from src.ask_llm.clients import HuggingFaceClient
except ImportError:
    HuggingFaceClient = MagicMock()  # Mock the HuggingFaceClient for tests


# Mock client classes to prevent actual client initialization
@pytest.fixture
def mock_clients():
    # Revert target back to where AskLLM looks up the names
    with patch('src.ask_llm.main.OpenAIClient', autospec=True) as mock_openai, \
         patch('src.ask_llm.main.OllamaClient', autospec=True) as mock_ollama, \
         patch('src.ask_llm.main.is_huggingface_available', return_value=True), \
         patch('src.ask_llm.main.HuggingFaceClient', autospec=True) as mock_hf:
        yield {
            "openai": mock_openai,
            "ollama": mock_ollama,
            "huggingface": mock_hf,
        }

# Mock HistoryManager
@pytest.fixture
def mock_history_manager():
    with patch('src.ask_llm.main.HistoryManager', autospec=True) as mock_hm:
        # Mock the instance returned by the constructor
        mock_instance = mock_hm.return_value
        yield mock_instance # Yield the mock instance for assertions

# Mock global_config used by AskLLM
@pytest.fixture
def mock_global_config(monkeypatch):
    # Create a mock config instance for AskLLM to use
    mock_config = MagicMock(spec=Config)
    mock_config.DEFAULT_MODEL = "mock-default-openai"
    mock_config.OPENAPI_MODELS = ["mock-default-openai", "gpt-4o"]
    mock_config.OLLAMA_MODELS = ["mock-ollama"]
    mock_config.HUGGINGFACE_MODELS = ["mock-hf"]
    mock_config.TEMPERATURE = 0.8 # Example attribute needed later

    # Patch the global_config where AskLLM imports it
    monkeypatch.setattr('src.ask_llm.main.global_config', mock_config)
    return mock_config


# === Tests for AskLLM ===

# Use fixtures to provide mocks
def test_ask_llm_init(mock_global_config, mock_clients, mock_history_manager):
    """Test AskLLM initialization."""
    # Patch initialize_client and load_history within the test scope
    with patch.object(AskLLM, 'initialize_client', return_value=MagicMock()) as mock_init_client, \
         patch.object(AskLLM, 'load_history') as mock_load_hist:

        instance = AskLLM()

        # Check attributes are set (using the mock config default)
        assert instance.model_id == mock_global_config.DEFAULT_MODEL

        # Check methods were called
        mock_init_client.assert_called_once()
        # Check HistoryManager was instantiated (implicitly by mock_history_manager fixture)
        # Check load_history was called on the instance
        mock_load_hist.assert_called_once()
        # Check HistoryManager was called with the client from initialize_client
        from src.ask_llm.main import HistoryManager # Need class for assertion
        HistoryManager.assert_called_once_with(client=mock_init_client.return_value)


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
    # Set the default model in the mock config for this specific test
    mock_global_config.DEFAULT_MODEL = model_id

    # Instantiate AskLLM - this calls initialize_client internally during __init__
    # We don't need to mock initialize_client here because we *want* to test it.
    with patch.object(AskLLM, 'load_history'): # Still mock load_history
        instance = AskLLM()

    # Get the mock for the expected client class
    mock_client_class = mock_clients[expected_client_mock_key]

    # Assert that the expected client class was instantiated with only the model_id
    mock_client_class.assert_called_once_with(model_id)

    # Assert that the instance's client attribute is the instance of the mocked class
    assert instance.client == mock_client_class.return_value


def test_ask_llm_initialize_client_unknown(mock_global_config, mock_clients, mock_history_manager):
    """Test initialization with an unknown model raises ValueError."""
    mock_global_config.DEFAULT_MODEL = "unknown-model"
    mock_global_config.OPENAPI_MODELS = ["gpt-4o"] # Ensure it's not in any list
    mock_global_config.OLLAMA_MODELS = ["llama3"]
    mock_global_config.HUGGINGFACE_MODELS = ["hf-model"]

    with pytest.raises(ValueError, match="Unknown model specified"):
        with patch.object(AskLLM, 'load_history'):
             instance = AskLLM() # Initialization should fail here 

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
    mock_global_config.DEFAULT_MODEL = "mock-hf"
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
    # Instantiate AskLLM *without* patching initialize_client
    # It should use the mocked HuggingFaceClient from the fixture
    with patch.object(AskLLM, 'load_history'):
        instance = AskLLM()
        # Ensure the history manager mock is used
        instance.history_manager = mock_history_manager

    # Call the method under test, patching isinstance within main's scope
    def mock_isinstance(obj, classinfo):
        if classinfo is mock_hf_client_class:
            return True
        import builtins
        return builtins.isinstance(obj, classinfo)

    with patch('src.ask_llm.main.isinstance', side_effect=mock_isinstance):
        response = instance.query(prompt)

    # --- Assertions --- #
    assert response == expected_response
    # Verify AskLLM initialized the correct (mocked) client class
    mock_hf_client_class.assert_called_once_with("mock-hf")
    # Verify the instance created is the one we configured
    assert instance.client == mock_client_instance

    # Check history manager calls (remain the same)
    mock_history_manager.add_message.assert_any_call("user", prompt)
    mock_history_manager.get_context_messages.assert_called_once()
    mock_history_manager.get_last_assistant_message.assert_called_once()
    mock_history_manager.add_message.assert_any_call("assistant", expected_response)
    assert mock_history_manager.add_message.call_count == 2

    # Check client query call (remains the same)
    mock_client_instance.query.assert_called_once_with(
        messages=complete_context_messages,
        plaintext_output=False,
        stream=True
    )

def test_ask_llm_query_duplicate_response(
    mock_global_config, mock_clients, mock_history_manager
):
    """Test query flow when a duplicate response is initially detected."""
    prompt = "Tell me a joke."
    duplicate_response = "Why did the chicken cross the road?"
    final_response = "To get to the other side!"
    complete_context_messages = [{"role": "user", "content": prompt}]
    initial_temp = mock_global_config.TEMPERATURE

    # --- Setup Mocks ---
    # Ensure AskLLM initializes with a HuggingFace model ID
    mock_global_config.DEFAULT_MODEL = "mock-hf"
    # Get the mock HF Client *class* and its *return_value* (the instance)
    mock_hf_client_class = mock_clients["huggingface"]
    mock_client_instance = mock_hf_client_class.return_value
    # Configure the mock instance *after* AskLLM initializes it
    mock_client_instance.query.side_effect = [duplicate_response, final_response]
    mock_client_instance.console = MagicMock()

    mock_history_manager.get_context_messages.return_value = complete_context_messages
    mock_history_manager.get_last_assistant_message.return_value = duplicate_response

    # --- Test --- #
    # Instantiate AskLLM *without* patching initialize_client
    with patch.object(AskLLM, 'load_history'):
        instance = AskLLM()
        # Ensure the history manager mock is used
        instance.history_manager = mock_history_manager

    # Call the method under test, patching isinstance within main's scope
    def mock_isinstance(obj, classinfo):
        if classinfo is mock_hf_client_class:
            return True
        import builtins
        return builtins.isinstance(obj, classinfo)

    with patch('src.ask_llm.main.isinstance', side_effect=mock_isinstance):
        response = instance.query(prompt)

    # --- Assertions --- #
    assert response == final_response
    assert mock_global_config.TEMPERATURE == initial_temp # Temperature should be reset
    # Verify AskLLM initialized the correct (mocked) client class
    mock_hf_client_class.assert_called_once_with("mock-hf")
    # Verify the instance created is the one we configured
    assert instance.client == mock_client_instance

    # Check history manager calls (remain the same)
    mock_history_manager.add_message.assert_any_call("user", prompt)
    mock_history_manager.get_context_messages.assert_called()
    mock_history_manager.get_last_assistant_message.assert_called_once()
    mock_history_manager.add_message.assert_any_call("assistant", final_response)
    assert mock_history_manager.add_message.call_count == 2 # User prompt, final assistant response

    # Check client query calls (remain the same)
    assert mock_client_instance.query.call_count == 2
    expected_call_args = {
        "messages": complete_context_messages,
        "plaintext_output": False,
        "stream": True
    }
    mock_client_instance.query.assert_has_calls([
        call(**expected_call_args),
        call(**expected_call_args) 
    ])
    mock_client_instance.console.print.assert_called_once_with(
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
    mock_global_config.DEFAULT_MODEL = "mock-hf"
    # Get the mock HF Client *class* and its *return_value* (the instance)
    mock_hf_client_class = mock_clients["huggingface"]
    mock_client_instance = mock_hf_client_class.return_value
    # Configure the mock instance *after* AskLLM initializes it
    mock_client_instance.query.side_effect = KeyboardInterrupt
    mock_client_instance.console = MagicMock()

    mock_history_manager.get_context_messages.return_value = complete_context_messages

    # --- Test --- #
    # Instantiate AskLLM *without* patching initialize_client
    with patch.object(AskLLM, 'load_history'):
        instance = AskLLM()
        # Ensure the history manager mock is used
        instance.history_manager = mock_history_manager

    # Call the method under test, patching isinstance within main's scope
    def mock_isinstance(obj, classinfo):
        if classinfo is mock_hf_client_class:
            return True
        import builtins
        return builtins.isinstance(obj, classinfo)

    with patch('src.ask_llm.main.isinstance', side_effect=mock_isinstance):
        response = instance.query(prompt)

    # --- Assertions --- #
    assert response is None # Query should return None on interrupt
    # Verify AskLLM initialized the correct (mocked) client class
    mock_hf_client_class.assert_called_once_with("mock-hf")
    # Verify the instance created is the one we configured
    assert instance.client == mock_client_instance

    # Check history manager calls (remain the same)
    mock_history_manager.add_message.assert_called_once_with("user", prompt)
    mock_history_manager.get_context_messages.assert_called_once()

    # Check client query call (remains the same)
    mock_client_instance.query.assert_called_once_with(
        messages=complete_context_messages,
        plaintext_output=False,
        stream=True
    )
    # Check console message
    mock_client_instance.console.print.assert_called_once_with(
        "\n[bold red]Query interrupted.[/bold red]"
    ) 