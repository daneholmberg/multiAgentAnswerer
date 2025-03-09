# Multi-Agent AI Collaboration Tool

This project implements a collaborative AI system using CrewAI to orchestrate multiple state-of-the-art language models. The system enables models to independently answer questions, anonymously evaluate each other's responses, and iteratively improve answers based on structured feedback.

## Features

- Orchestrates multiple AI models (DeepSeek R1, OpenAI O3 Mini High, OpenAI O1, Claude 3.7 Thinking, and Claude 3.6)
- Anonymous evaluation of responses
- Weighted scoring system based on custom criteria
- Iterative improvement of the best answers
- Final judgment round to select the optimal response

## Requirements

- Python 3.8+
- API keys for various AI models (OpenAI, Anthropic, DeepSeek)

## Installation

1. Clone this repository
2. Install the package: `pip install -e .`
3. Create a `.env` file with your API keys (see `src/multiagent/config/.env.example`)

## Usage

```bash
# Run as a module
python -m multiagent "Your question here"

# Or use the installed command
multiagent "Your question here"
```

## Project Structure

```
multiagent/
├── docs/                    # Documentation
│   └── design/              # Design documents
├── src/                     # Source code
│   ├── main.py              # Application entry point
│   └── multiagent/          # Main package
│       ├── agents/          # AI agent definitions
│       ├── config/          # Configuration files
│       ├── models/          # Data models
│       ├── tasks/           # Task definitions
│       └── utils/           # Utility functions
├── tests/                   # Test suite
├── examples/                # Example usage
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
└── README.md                # This file
```

## Development

To set up a development environment:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
``` 