import pytest
from ask_llm.cli import parse_arguments
from ask_llm.utils.config import Config
from unittest.mock import patch

class DummyConfig:
    DEFAULT_MODEL_ALIAS = 'test'
    MODELS_CONFIG_PATH = '/path/models.yaml'

@pytest.fixture
def config_obj():
    return DummyConfig()

def test_parse_no_stream_flag(config_obj):
    """Test that --no-stream flag is parsed correctly."""
    with patch('sys.argv', ['script', '--no-stream']):
        args = parse_arguments(config_obj)
    assert args.no_stream is True

def test_parse_plain_and_no_stream(config_obj):
    """Test parsing both --plain and --no-stream together."""
    with patch('sys.argv', ['script', '--plain', '--no-stream']):
        args = parse_arguments(config_obj)
    assert args.plain is True
    assert args.no_stream is True

# Test default model alias handling
def test_default_model_alias_provided():
    pass # Add pass to fix syntax error

# Mock configuration for tests
