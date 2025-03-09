#!/usr/bin/env python3
"""
Basic usage example for the Multi-Agent AI Collaboration Tool.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import process_question
from src.multiagent.config.model_selection import get_model_selection


async def main():
    """Run a simple example of the Multi-Agent AI Collaboration Tool."""
    question = "What are the ethical implications of using AI in healthcare?"

    print(f"Processing question: {question}")

    # Example 1: Interactive model selection
    print("\nExample 1: Interactive model selection")
    selected_models = get_model_selection(use_all_models=False)
    result1 = await process_question(question, selected_models)

    print("\nFinal Answer (Interactive selection):")
    print(result1)

    # Example 2: Use all models
    print("\nExample 2: Using all models")
    result2 = await process_question(question, get_model_selection(use_all_models=True))

    print("\nFinal Answer (All models):")
    print(result2)


if __name__ == "__main__":
    asyncio.run(main())
