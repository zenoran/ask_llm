import unittest
from ask_llm.clients.openai_client import OpenAIClient
from ask_llm.clients.ollama_client import OllamaClient
from ask_llm.models.message import Message

class TestOpenAIClient(unittest.TestCase):
    def setUp(self):
        self.client = OpenAIClient(model="gpt-4o")

    def test_query(self):
        prompt = "What is the capital of France?"
        messages = [Message(role="user", content=prompt)]
        response = self.client.query(messages, prompt)
        self.assertIn("Paris", response)

    def test_format_response(self):
        response_text = "The capital of France is Paris."
        formatted_response = self.client.format_response(response_text, None)
        self.assertIsNotNone(formatted_response)

class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient(model="deepseek-r1")

    def test_query(self):
        prompt = "What is the capital of France?"
        messages = [Message(role="user", content=prompt)]
        response = self.client.query(messages, prompt)
        self.assertIn("Paris", response)

    def test_format_response(self):
        response_text = "The capital of France is Paris."
        formatted_response = self.client.format_response(response_text, None)
        self.assertIsNotNone(formatted_response)

if __name__ == "__main__":
    unittest.main()