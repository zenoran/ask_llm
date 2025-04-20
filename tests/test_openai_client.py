import pytest
from unittest.mock import patch, MagicMock
import os

from ask_llm.clients.openai_client import OpenAIClient
from ask_llm.models.message import Message
from ask_llm.utils.config import Config
from ask_llm.clients.base import LLMClient
from openai import OpenAIError

class TestOpenAIClient:
    """Test suite for OpenAIClient class."""
    
    @pytest.fixture
    def mock_config_obj(self):
        """Provides a standard mock Config object for OpenAI tests."""
        mock_cfg = MagicMock(spec=Config) # Use spec=Config for better mocking
        mock_cfg.VERBOSE = False
        mock_cfg.PLAIN_OUTPUT = False
        mock_cfg.NO_STREAM = False
        mock_cfg.MAX_TOKENS = 100
        mock_cfg.TEMPERATURE = 0.7
        mock_cfg.TOP_P = 0.9
        mock_cfg.SYSTEM_MESSAGE = "Test System Prompt"
        return mock_cfg

    @pytest.fixture
    def mock_openai_api(self):
        """Mocks the openai.OpenAI client constructor."""
        with patch('ask_llm.clients.openai_client.OpenAI') as mock_openai_constructor:
            yield mock_openai_constructor

    @pytest.fixture
    def client(self, mock_config_obj, mock_openai_api):
        """Provides an initialized OpenAIClient instance with mocked dependencies."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            instance = OpenAIClient(model="gpt-4o", config=mock_config_obj) 
            instance.client = mock_openai_api.return_value # Ensure instance uses mocked OpenAI client
            return instance

    def test_init(self, mock_config_obj, mock_openai_api):
        """Test client initialization and API key retrieval."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            client = OpenAIClient("gpt-4o", config=mock_config_obj)
            assert client.model == "gpt-4o"
            assert client.config == mock_config_obj
            assert client.api_key == "test-api-key"
            mock_openai_api.assert_called_once_with(api_key="test-api-key")

    def test_init_missing_api_key(self, mock_config_obj):
        """Test exception when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                OpenAIClient("gpt-4o", config=mock_config_obj)
    
    def test_prepare_api_messages(self, client: OpenAIClient, mock_config_obj):
        """Test that messages are formatted correctly for the API."""
        messages = [
            Message("user", "Hello"),
            Message("assistant", "Hi there"),
            Message("user", "How are you?")
        ]
        mock_config_obj.SYSTEM_MESSAGE = "Be helpful."
        api_messages = client._prepare_api_messages(messages)
        
        assert api_messages[0] == {"role": "system", "content": "Be helpful."}
        assert api_messages[1:] == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]

    def test_prepare_api_messages_with_existing_system(self, client: OpenAIClient, mock_config_obj):
        """Test that an existing system message isn't overridden by default."""
        messages = [
            Message("system", "Initial system prompt"),
            Message("user", "Hello")
        ]
        mock_config_obj.SYSTEM_MESSAGE = "This should not be used"

        api_messages = client._prepare_api_messages(messages)
        
        assert len(api_messages) == 2
        assert api_messages[0] == {"role": "system", "content": "Initial system prompt"}
        assert api_messages[1] == {"role": "user", "content": "Hello"}

    def test_prepare_api_messages_no_default_system(self, client: OpenAIClient, mock_config_obj):
        """Test that no system message is added if default is empty."""
        messages = [Message("user", "Hello")]
        mock_config_obj.SYSTEM_MESSAGE = None # Set system message to None/empty
        
        api_messages = client._prepare_api_messages(messages)
        
        assert len(api_messages) == 1
        assert api_messages[0] == {"role": "user", "content": "Hello"}

    @patch('ask_llm.clients.openai_client.OpenAIClient._handle_streaming_output')
    def test_query(self, mock_handle_streaming, client: OpenAIClient, mock_openai_api):
        """Test the main query method (streaming case)."""
        messages = [Message("user", "Test prompt")]
        api_messages = client._prepare_api_messages(messages) # Prepare messages to check payload
        expected_payload = {
            "model": client.model,
            "messages": api_messages,
            "max_tokens": client.config.MAX_TOKENS,
            "temperature": client.config.TEMPERATURE,
            "top_p": client.config.TOP_P,
            "stream": True
        }
        client.query(messages)
        
        mock_create = client.client.chat.completions.create
        mock_create.assert_called_once_with(**expected_payload)
        
        mock_handle_streaming.assert_called_once()
        stream_arg = mock_handle_streaming.call_args[1]['stream_iterator']
        assert hasattr(stream_arg, '__next__')

    @patch('ask_llm.clients.base.LLMClient._print_assistant_message') # Mock the base print method
    def test_query_non_streaming(self, mock_print_assistant, client: OpenAIClient, mock_openai_api):
        """Test the main query method (non-streaming case)."""
        messages = [Message("user", "Test prompt")]
        api_messages = client._prepare_api_messages(messages) # Prepare messages for checking payload
        client.config.NO_STREAM = True 
        client.config.VERBOSE = True # To check usage print
        
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Full response"
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        mock_completion.usage.total_tokens = 15
        
        client.client.chat.completions.create.return_value = mock_completion
        with patch.object(client.console, 'print') as mock_console_print:
            response = client.query(messages, stream=False)
        
        mock_create = client.client.chat.completions.create
        expected_payload = {
            "model": client.model,
            "messages": api_messages,
            "max_tokens": client.config.MAX_TOKENS,
            "temperature": client.config.TEMPERATURE,
            "top_p": client.config.TOP_P,
            "stream": False
        }
        mock_create.assert_called_once_with(**expected_payload)
        assert response == "Full response"
        mock_print_assistant.assert_called_once_with("Full response", second_part=None)
        mock_console_print.assert_any_call("[dim]OpenAI Tokens: Prompt=10, Completion=5, Total=15[/dim]")

    def test_stream_response_success(self, client: OpenAIClient):
        """Test successful streaming response handling."""
        mock_stream_obj = MagicMock()
        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]), 
        ]
        mock_stream_obj.__iter__.return_value = iter(chunks)
        client.client.chat.completions.create.return_value = mock_stream_obj
        payload = {"model": client.model, "messages": [], "stream": True}
        with patch.object(LLMClient, '_handle_streaming_output', return_value="Hello world!") as mock_base_handler:
            response = client._stream_response([], False, payload) 
            assert response == "Hello world!"
            passed_iterator = mock_base_handler.call_args[1]['stream_iterator']
            content_list = list(passed_iterator)
            assert content_list == ["Hello", " world", "!"]
            client.client.chat.completions.create.assert_called_once_with(**payload)
    def test_stream_response_openai_error(self, client: OpenAIClient):
        """Test OpenAIError handling in stream response."""
        client.client.chat.completions.create.side_effect = OpenAIError("API error")
        payload = {"model": client.model, "messages": [], "stream": True}
        response = client._stream_response([], False, payload)
        assert response == "ERROR: OpenAI API Error - API error"
    def test_stream_response_generic_error(self, client: OpenAIClient):
        """Test generic Exception handling in stream response."""
        client.client.chat.completions.create.side_effect = Exception("Unexpected error")
        payload = {"model": client.model, "messages": [], "stream": True}
        response = client._stream_response([], False, payload)
        assert response == "ERROR: Unexpected error - Unexpected error"
    def test_get_full_response_openai_error(self, client: OpenAIClient):
        """Test OpenAIError handling in full response."""
        client.client.chat.completions.create.side_effect = OpenAIError("Full API error")
        payload = {"model": client.model, "messages": [], "stream": False}
        response = client._get_full_response([], False, payload)
        assert response == "ERROR: OpenAI API Error - Full API error"

    def test_get_full_response_generic_error(self, client: OpenAIClient):
        """Test generic Exception handling in full response."""
        client.client.chat.completions.create.side_effect = Exception("Full Unexpected error")
        payload = {"model": client.model, "messages": [], "stream": False}
        response = client._get_full_response([], False, payload)
        assert response == "ERROR: Unexpected error - Full Unexpected error"
    def test_get_styling(self, client: OpenAIClient):
        """Test the get_styling method returns correct values."""
        title, border_style = client.get_styling()
        assert border_style == "green"
        assert title == "[bold green]gpt-4o[/bold green]" 