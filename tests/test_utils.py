# Test utilities for the ask_llm project

import unittest
from ask_llm.utils.history import HistoryManager
from ask_llm.models.message import Message

class TestHistoryManager(unittest.TestCase):
    def setUp(self):
        self.history_manager = HistoryManager()
        self.test_message = Message(role="user", content="Test message")

    def test_load_history(self):
        # Assuming the history file is empty for this test
        history = self.history_manager.load_history()
        self.assertEqual(history, [])

    def test_save_history(self):
        self.history_manager.save_history([self.test_message])
        loaded_history = self.history_manager.load_history()
        self.assertEqual(len(loaded_history), 1)
        self.assertEqual(loaded_history[0]['content'], "Test message")

    def test_add_message(self):
        self.history_manager.add_message(self.test_message)
        loaded_history = self.history_manager.load_history()
        self.assertIn(self.test_message.content, [msg['content'] for msg in loaded_history])

if __name__ == '__main__':
    unittest.main()