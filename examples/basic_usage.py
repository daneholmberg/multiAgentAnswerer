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


async def main():
    """Run a simple example of the Multi-Agent AI Collaboration Tool."""
    question = "What are the ethical implications of using AI in healthcare?"

    print(f"Processing question: {question}")
    result = await process_question(question)

    print("\nFinal Answer:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
