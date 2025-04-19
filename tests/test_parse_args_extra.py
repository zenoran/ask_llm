import pytest
from unittest.mock import patch
from src.ask_llm.cli import parse_arguments

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