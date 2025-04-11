import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_config():
    """Mock the global config to avoid validation errors during tests."""
    mock_config = MagicMock()
    mock_config.VERBOSE = False
    mock_config.DEFAULT_MODEL = "gpt-4o"
    mock_config.MODEL_OPTIONS = ["gpt-4o", "llama3:latest"]
    mock_config.OLLAMA_MODELS = ["llama3:latest"]
    mock_config.OPENAI_MODELS = ["gpt-4o"]
    mock_config.HUGGINGFACE_MODELS = []
    
    # Patch both potential import paths
    with patch('src.ask_llm.utils.config.config', mock_config), \
         patch('ask_llm.utils.config.config', mock_config):
        yield mock_config 