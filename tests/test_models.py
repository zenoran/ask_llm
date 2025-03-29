import pytest
from ask_llm.models.message import Message

def test_message_initialization():
    msg = Message(role="user", content="Hello, world!")
    assert msg.role == "user"
    assert msg.content == "Hello, world!"
    assert msg.timestamp is not None

def test_message_from_dict():
    data = {"role": "assistant", "content": "Hello, user!", "timestamp": 1234567890}
    msg = Message.from_dict(data)
    assert msg.role == "assistant"
    assert msg.content == "Hello, user!"
    assert msg.timestamp == 1234567890

def test_message_to_dict():
    msg = Message(role="user", content="Test message")
    msg_dict = msg.to_dict()
    assert msg_dict["role"] == "user"
    assert msg_dict["content"] == "Test message"
    assert "timestamp" in msg_dict

def test_message_to_api_format():
    msg = Message(role="assistant", content="API response")
    api_format = msg.to_api_format()
    assert api_format["role"] == "assistant"
    assert api_format["content"] == "API response"