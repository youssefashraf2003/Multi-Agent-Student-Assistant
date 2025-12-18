import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock
from agents.search_agent.core import SearchAgent

class MockSearchAgent(SearchAgent):
    def __init__(self):
        # Skip real init which loads tools
        self.llm = MagicMock()
        self.tools = {}
        self.tools["Search"] = MagicMock()
        self.tools["Search"].run.return_value = "Mock Search Result"

class TestSearchAgent(unittest.TestCase):
    def setUp(self):
        self.agent = MockSearchAgent()

    def test_run_returns_sources(self):
        # Mock LLM response sequence
        # 1. Action: Search
        # 2. Final Answer
        self.agent.llm.invoke.side_effect = [
            MagicMock(content="Thought: Need to search.\nAction: Search\nAction Input: test query"),
            MagicMock(content="Thought: Done.\nFinal Answer: The answer is 42.")
        ]

        result = self.agent.run("test query")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["response"], "The answer is 42.")
        self.assertIn("Search: test query", result["sources"])

if __name__ == '__main__':
    unittest.main()
