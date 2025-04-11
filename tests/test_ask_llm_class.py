import pytest
from unittest.mock import patch, MagicMock

# Import the class to test
from src.ask_llm.main import AskLLM

# Import things that might be needed for mocking config/clients
from src.ask_llm.utils.config import Config
from src.ask_llm.clients import OpenAIClient, OllamaClient, HuggingFaceClient


# Mock client classes to prevent actual client initialization
@pytest.fixture
def mock_clients():
    with patch('src.ask_llm.main.OpenAIClient', autospec=True) as mock_openai, \
         patch('src.ask_llm.main.OllamaClient', autospec=True) as mock_ollama, \
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
    context_messages = [{"role": "user", "content": "previous prompt"}]

    # --- Setup Mocks ---
    # Mock the client instance that gets created
    mock_client_instance = MagicMock()
    mock_client_instance.query.return_value = expected_response

    # Mock the history manager methods
    mock_history_manager.get_context_messages_excluding_last.return_value = context_messages
    # Mock get_last_assistant_message to return None initially (no duplicate)
    mock_history_manager.get_last_assistant_message.return_value = None

    # --- Test --- #
    # Instantiate AskLLM, patching initialize_client to return our mock client instance
    with patch.object(AskLLM, 'initialize_client', return_value=mock_client_instance), \
         patch.object(AskLLM, 'load_history'):
        instance = AskLLM()
        # Set the mock client directly on the instance for clarity, though patch does it
        instance.client = mock_client_instance
        # Set the mock history manager on the instance
        instance.history_manager = mock_history_manager

        # Call the method under test
        response = instance.query(prompt)

    # --- Assertions --- #
    assert response == expected_response

    # Check history manager calls
    mock_history_manager.add_message.assert_any_call("user", prompt)
    mock_history_manager.get_context_messages_excluding_last.assert_called_once()
    mock_history_manager.get_last_assistant_message.assert_called_once()
    mock_history_manager.add_message.assert_any_call("assistant", expected_response)
    assert mock_history_manager.add_message.call_count == 2

    # Check client query call
    mock_client_instance.query.assert_called_once_with(
        context_messages, prompt, plaintext_output=False
    )

def test_ask_llm_query_duplicate_response(
    mock_global_config, mock_clients, mock_history_manager
):
    """Test query flow when a duplicate response is initially detected."""
    prompt = "Tell me a joke."
    duplicate_response = "Why did the chicken cross the road?"
    final_response = "To get to the other side!"
    context_messages = []
    initial_temp = mock_global_config.TEMPERATURE

    # --- Setup Mocks ---
    mock_client_instance = MagicMock()
    # Client returns duplicate first, then final response
    mock_client_instance.query.side_effect = [duplicate_response, final_response]
    # Mock console print to avoid output during test
    mock_client_instance.console = MagicMock()

    mock_history_manager.get_context_messages_excluding_last.return_value = context_messages
    # History manager returns the duplicate response when checked
    mock_history_manager.get_last_assistant_message.return_value = duplicate_response

    # --- Test --- #
    with patch.object(AskLLM, 'initialize_client', return_value=mock_client_instance), \
         patch.object(AskLLM, 'load_history'):
        instance = AskLLM()
        instance.client = mock_client_instance
        instance.history_manager = mock_history_manager

        response = instance.query(prompt)

    # --- Assertions --- #
    assert response == final_response
    assert mock_global_config.TEMPERATURE == initial_temp # Temperature should be reset

    # Check history manager calls
    mock_history_manager.add_message.assert_any_call("user", prompt)
    mock_history_manager.get_context_messages_excluding_last.assert_called_once()
    mock_history_manager.get_last_assistant_message.assert_called_once()
    # Assistant message should be the *final* response
    mock_history_manager.add_message.assert_any_call("assistant", final_response)
    assert mock_history_manager.add_message.call_count == 2 # User prompt, final assistant response

    # Check client query calls (called twice)
    assert mock_client_instance.query.call_count == 2
    # First call
    mock_client_instance.query.assert_any_call(
        context_messages, prompt, plaintext_output=False
    )
    # Second call (after temp change)
    mock_client_instance.query.assert_any_call(
        context_messages, prompt, plaintext_output=False
    )
    # Check console warning was printed
    mock_client_instance.console.print.assert_called_once_with(
        "[yellow]Detected duplicate response. Regenerating with higher temperature...[/yellow]"
    )

def test_ask_llm_query_keyboard_interrupt(
    mock_global_config, mock_clients, mock_history_manager
):
    """Test that KeyboardInterrupt during query is handled."""
    prompt = "Infinite loop prompt"

    # --- Setup Mocks ---
    mock_client_instance = MagicMock()
    # Simulate KeyboardInterrupt when client.query is called
    mock_client_instance.query.side_effect = KeyboardInterrupt
    # Mock console
    mock_client_instance.console = MagicMock()

    mock_history_manager.get_context_messages_excluding_last.return_value = []

    # --- Test --- #
    with patch.object(AskLLM, 'initialize_client', return_value=mock_client_instance), \
         patch.object(AskLLM, 'load_history'):
        instance = AskLLM()
        instance.client = mock_client_instance
        instance.history_manager = mock_history_manager

        response = instance.query(prompt)

    # --- Assertions --- #
    assert response is None # Query should return None on interrupt

    # Check history manager calls (only user message should be added)
    mock_history_manager.add_message.assert_called_once_with("user", prompt)
    mock_history_manager.get_context_messages_excluding_last.assert_called_once()

    # Check client query call
    mock_client_instance.query.assert_called_once()
    # Check console message
    mock_client_instance.console.print.assert_called_once_with(
        "\n[bold red]Query interrupted.[/bold red]"
    ) 