#!/usr/bin/env python3
"""
Tests for the model selection functionality.
"""
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multiagent.config.model_selection import filter_agents, AVAILABLE_MODELS


class TestModelSelection(unittest.TestCase):
    """Test the model selection functionality."""

    def test_filter_agents(self):
        """Test the filter_agents function."""
        # Create a mock agents dictionary
        mock_agents = {
            "agent1": "agent1_object",
            "agent2": "agent2_object",
            "agent3": "agent3_object",
        }

        # Test filtering with a subset of agents
        selected_models = {"agent1", "agent3"}
        filtered_agents = filter_agents(mock_agents, selected_models)

        self.assertEqual(len(filtered_agents), 2)
        self.assertIn("agent1", filtered_agents)
        self.assertIn("agent3", filtered_agents)
        self.assertNotIn("agent2", filtered_agents)

        # Test filtering with all agents
        selected_models = {"agent1", "agent2", "agent3"}
        filtered_agents = filter_agents(mock_agents, selected_models)

        self.assertEqual(len(filtered_agents), 3)
        self.assertIn("agent1", filtered_agents)
        self.assertIn("agent2", filtered_agents)
        self.assertIn("agent3", filtered_agents)

        # Test filtering with no agents
        selected_models = set()
        filtered_agents = filter_agents(mock_agents, selected_models)

        self.assertEqual(len(filtered_agents), 0)

    def test_available_models(self):
        """Test that all expected models are available."""
        expected_models = {"deepseek", "openai_o3", "openai_o1", "claude_37", "claude_36"}
        self.assertEqual(set(AVAILABLE_MODELS.keys()), expected_models)


if __name__ == "__main__":
    unittest.main()
