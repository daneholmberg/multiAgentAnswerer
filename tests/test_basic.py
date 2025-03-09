#!/usr/bin/env python3
"""
Basic tests for the Multi-Agent AI Collaboration Tool.
"""
import unittest
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multiagent.models.base import Answer, Evaluation, ImprovedAnswer


class TestModels(unittest.TestCase):
    """Test the data models."""

    def test_answer_model(self):
        """Test the Answer model."""
        answer = Answer(content="This is a test answer.", agent_id="test_agent")
        self.assertEqual(answer.content, "This is a test answer.")
        self.assertEqual(answer.agent_id, "test_agent")
        self.assertEqual(str(answer), "This is a test answer.")

    def test_improved_answer_model(self):
        """Test the ImprovedAnswer model."""
        improved = ImprovedAnswer(
            original_answer_id="test_answer",
            content="This is an improved answer.",
            agent_id="test_agent",
            improvements="Added more details.",
        )
        self.assertEqual(improved.content, "This is an improved answer.")
        self.assertEqual(improved.agent_id, "test_agent")
        self.assertEqual(improved.original_answer_id, "test_answer")
        self.assertEqual(improved.improvements, "Added more details.")
        self.assertEqual(str(improved), "This is an improved answer.")


if __name__ == "__main__":
    unittest.main()
