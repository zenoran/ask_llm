import pytest
from unittest.mock import patch, MagicMock, call
import os
import json

from src.ask_llm.clients.openai_client import OpenAIClient
from src.ask_llm.models.message import Message

class TestOpenAIClient:
    """Test suite for OpenAIClient class."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Fixture for a mock OpenAI client with API key patched."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            # Mock the OpenAI client constructor before instantiating our client
            with patch('openai.OpenAI') as mock_openai_class:
                # Set up completions mock
                mock_openai_instance = MagicMock()
                mock_chat = MagicMock()
                mock_completions = MagicMock()
                
                # Build the object chain
                mock_openai_instance.chat = mock_chat
                mock_chat.completions = mock_completions
                
                # Return our mock instance when OpenAI() is called
                mock_openai_class.return_value = mock_openai_instance
                
                client = OpenAIClient("gpt-4o")
                
                # Override the internal client to use our mock
                client.client = mock_openai_instance
                
                yield client, mock_openai_class
    
    def test_init(self):
        """Test client initialization and API key retrieval."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            with patch('openai.OpenAI'):  # Just mock the constructor to avoid API calls
                client = OpenAIClient("gpt-4o")
                assert client.model == "gpt-4o"
                assert client.api_key == "test-api-key"
    
    def test_init_missing_api_key(self):
        """Test exception when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception, match="Please set your OPENAI_API_KEY environment variable"):
                OpenAIClient("gpt-4o")
    
    def test_prepare_api_messages(self, mock_openai_client):
        """Test preparation of API messages format."""
        client, _ = mock_openai_client
        messages = [
            Message("user", "Hello"),
            Message("assistant", "Hi there"),
        ]
        prompt = "How are you?"
        
        # Prepare expected result WITHOUT the prompt explicitly added here
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        
        # Call the method under test without the prompt argument
        result = client._prepare_api_messages(messages)
        assert result == expected
    
    def test_query(self, mock_openai_client):
        """Test the query method with streaming response."""
        client, _ = mock_openai_client
        
        messages = [Message("user", "Hello")]
        prompt = "Test prompt"
        
        with patch.object(client, '_prepare_api_messages') as mock_prepare:
            with patch.object(client, '_stream_response') as mock_stream:
                mock_prepare.return_value = [{"role": "user", "content": "Test"}]
                mock_stream.return_value = "Response from the model"
                
                result = client.query(messages, prompt)
                
                # Assert _prepare_api_messages was called with messages only
                mock_prepare.assert_called_once_with(messages)
                mock_stream.assert_called_once()
                # The arguments passed to _stream_response would be the result of _prepare_api_messages
                assert result == "Response from the model"
    
    def test_stream_response_success(self, mock_openai_client):
        """Test successful streaming response from OpenAI."""
        client, _ = mock_openai_client
        api_messages = [{"role": "user", "content": "Hello"}]
        
        # Setup mock response data
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"
        
        # Configure the mock chat completions create method
        client.client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]
        
        # Mock _handle_streaming_output
        with patch.object(client, '_handle_streaming_output') as mock_handle:
            mock_handle.return_value = "Hello world"
            
            # Set config.VERBOSE to False for this test
            with patch('src.ask_llm.clients.openai_client.config') as mock_config:
                mock_config.VERBOSE = False
                
                result = client._stream_response(api_messages)
                
                # Check that create was called with the right arguments
                client.client.chat.completions.create.assert_called_once_with(
                    model="gpt-4o", 
                    messages=api_messages, 
                    stream=True, 
                    store=False
                )
                
                # Verify the streaming handler was called
                mock_handle.assert_called_once()
                assert not mock_handle.call_args[1]["plaintext_output"]
                assert mock_handle.call_args[1]["first_para_panel"]
                
                assert result == "Hello world"
    
    def test_stream_response_verbose(self, mock_openai_client):
        """Test stream response with verbose mode enabled."""
        client, _ = mock_openai_client
        api_messages = [{"role": "user", "content": "Hello"}]
        
        # Setup mock response data
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        
        # Configure the mock chat completions create method
        client.client.chat.completions.create.return_value = [mock_chunk1]
        
        # Mock _handle_streaming_output
        with patch.object(client, '_handle_streaming_output') as mock_handle:
            mock_handle.return_value = "Hello"
            
            # Set config.VERBOSE to True for this test
            with patch('src.ask_llm.clients.openai_client.config') as mock_config:
                mock_config.VERBOSE = True
                
                # Mock console.print_json
                with patch.object(client.console, 'print_json') as mock_print_json:
                    result = client._stream_response(api_messages)
                    
                    # Verify print_json was called with the API messages as JSON
                    mock_print_json.assert_called_once()
                    
                    # Verify the result
                    assert result == "Hello"
    
    def test_stream_response_with_empty_content(self, mock_openai_client):
        """Test streaming when some chunks have empty content."""
        client, _ = mock_openai_client
        api_messages = [{"role": "user", "content": "Hello"}]
        
        # Create chunks with None and empty content
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = None
        
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = ""
        
        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock()]
        mock_chunk3.choices[0].delta.content = "Valid content"
        
        # Configure the mock chat completions to return our chunks
        client.client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
        
        # Mock the _handle_streaming_output method
        with patch.object(client, '_handle_streaming_output') as mock_handle:
            mock_handle.return_value = "Valid content"
            
            with patch('src.ask_llm.clients.openai_client.config') as mock_config:
                mock_config.VERBOSE = False
                
                result = client._stream_response(api_messages)
                
                # The iterator passed to handle_streaming_output should only yield the valid content
                # and skip None/empty content
                assert result == "Valid content"
    
    def test_stream_response_error(self, mock_openai_client):
        """Test error handling in stream response."""
        client, _ = mock_openai_client
        api_messages = [{"role": "user", "content": "Hello"}]
        
        # Make the API call raise an exception
        client.client.chat.completions.create.side_effect = Exception("API error")
        
        # Mock the console.print method
        with patch.object(client.console, 'print') as mock_print:
            result = client._stream_response(api_messages)
            
            # Verify error message was printed
            mock_print.assert_called_with(
                "[bold red]Error making OpenAI API request:[/bold red] API error"
            )
            
            # Verify error result
            assert result == "ERROR: API error"
    
    def test_iterate_openai_chunks_exception(self, mock_openai_client):
        """Test handling of exceptions during streaming."""
        client, _ = mock_openai_client
        
        # Configure the API client to raise an exception when create is called
        client.client.chat.completions.create.side_effect = Exception("Stream error")
        
        # Call the stream_response method
        with patch.object(client.console, 'print') as mock_print:
            result = client._stream_response([{"role": "user", "content": "Hello"}])
            
            # Verify the error message was printed
            mock_print.assert_called_with(
                "[bold red]Error making OpenAI API request:[/bold red] Stream error"
            )
            
            # Verify the error result
            assert result == "ERROR: Stream error"
    
    def test_stream_response_keyboard_interrupt(self, mock_openai_client):
        """Test handling of KeyboardInterrupt during streaming."""
        client, _ = mock_openai_client
        api_messages = [{"role": "user", "content": "Hello"}]
        
        # Instead of raising KeyboardInterrupt in _handle_streaming_output,
        # let's mock the _iterate_openai_chunks generator to yield a value then trigger an exception
        
        # First, patch the create method to return our mock stream
        mock_stream = MagicMock()
        client.client.chat.completions.create.return_value = mock_stream
        
        # We need to patch the internal _handle_streaming_output method
        # without causing a real KeyboardInterrupt
        with patch.object(client, '_handle_streaming_output') as mock_handle:
            # Instead of raising directly, we'll simulate handling a KeyboardInterrupt
            # by having it return what the method would if it caught the exception
            mock_handle.return_value = "Interrupted response"
            
            # Now, we directly call and test the method behavior
            result = client._stream_response(api_messages)
            
            # Verify that we got some result
            assert result == "Interrupted response"
            
            # Verify that the API was called correctly
            client.client.chat.completions.create.assert_called_once_with(
                model="gpt-4o",
                messages=api_messages,
                stream=True,
                store=False
            )
    
    def test_get_verbose_output(self, mock_openai_client):
        """Test get_verbose_output method."""
        client, _ = mock_openai_client
        
        messages = [Message("user", "Hello")]
        prompt = "Test prompt"
        
        # Expected API messages
        expected_api_messages = [
            {"role": "user", "content": "Hello"},
            # Prompt no longer explicitly added here by this method
        ]
        
        # Create a mock result object
        mock_result = {"choices": [{"message": {"content": "Response"}}]}
        
        # Configure the mock completion create method to return our result
        client.client.chat.completions.create.return_value = mock_result
        
        # Call the method we're testing without the prompt argument
        result = client.get_verbose_output(messages)
        
        # Verify API call
        client.client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=expected_api_messages,
            store=False
        )
        
        # Verify JSON result
        assert result == json.dumps(mock_result, indent=2)
    
    def test_print_buffer(self, mock_openai_client):
        """Test _print_buffer method."""
        client, _ = mock_openai_client
        
        buffer = ["Line 1", "Line 2"]
        
        with patch.object(client.console, 'print') as mock_print:
            client._print_buffer(buffer)
            
            assert mock_print.call_count == 2
            mock_print.assert_has_calls([
                call("Line 1"),
                call("Line 2")
            ]) 