#!/usr/bin/env python3
"""
Tests for the multi-agent workflow components.
"""
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call
from typing import Dict, List
from crewai import Agent

# Add the src directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multiagent.models.base import (
    Answer,
    Evaluation,
    ImprovedAnswer,
    AnswerScore,
    EvaluationCriterion,
)
from src.main import improve_answers, get_initial_answers, evaluate_answers, final_judgment
from src.multiagent.utils.run_logger import RunLogger


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self, role: str = "mock_agent"):
        super().__init__(
            role=role,
            goal="Mock goal",
            backstory="Mock backstory",
            allow_delegation=False,
            llm=MagicMock(model="mock-model"),
        )


class MockCrewOutput:
    """Mock for CrewAI's output"""

    def __init__(self, content):
        self.content = content

    def __str__(self):
        return self.content


@pytest.fixture
def test_data():
    """Fixture providing test data."""
    question = "What is the meaning of life?"

    # Mock answers
    answers = {
        "openai_o3": Answer(
            content="The meaning of life is to maximize happiness.", agent_id="openai_o3"
        ),
        "claude_36": Answer(
            content="The meaning of life is to seek understanding.", agent_id="claude_36"
        ),
    }

    # Mock evaluations
    evaluations = [
        Evaluation(
            evaluator_id="evaluator1",
            question=question,
            criteria=[
                EvaluationCriterion(
                    name="clarity",
                    description="How clear and understandable is the answer",
                    weight=0.5,
                ),
                EvaluationCriterion(
                    name="depth",
                    description="How deep and thorough is the answer",
                    weight=0.5,
                ),
            ],
            scores=[
                AnswerScore(
                    answer_id="Answer 1",
                    criteria_scores={"clarity": 8, "depth": 7},
                    total_score=7.5,
                    reasoning="Clear but could be deeper",
                )
            ],
        )
    ]

    # Mock improved answers
    improved_answers = {
        "openai_o3_improver1": ImprovedAnswer(
            content="The meaning of life is to maximize happiness and understanding.",
            agent_id="improver1",
            original_answer_id="openai_o3",
            improvements="Added understanding aspect",
        )
    }

    # Mock run logger
    run_logger = AsyncMock(spec=RunLogger)

    return {
        "question": question,
        "answers": answers,
        "evaluations": evaluations,
        "improved_answers": improved_answers,
        "run_logger": run_logger,
    }


def test_crew_output_handling():
    """Test handling of different CrewOutput formats."""
    # Test single output
    single_output = MockCrewOutput("test")
    assert str(single_output) == "test"

    # Test list output
    list_output = [MockCrewOutput("test1"), MockCrewOutput("test2")]
    assert len(list_output) == 2
    assert str(list_output[0]) == "test1"
    assert str(list_output[1]) == "test2"


@patch("src.main.get_answering_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_get_initial_answers_success(mock_crew, mock_get_agents, test_data):
    """Test successful retrieval of initial answers."""
    # Setup mocks
    mock_agents = {
        "openai_o3": MockAgent(role="openai_o3"),
        "claude_36": MockAgent(role="claude_36"),
    }
    mock_get_agents.return_value = mock_agents

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = [
        MockCrewOutput("The meaning of life is to maximize happiness."),
        MockCrewOutput("The meaning of life is to seek understanding."),
    ]
    mock_crew.return_value = mock_crew_instance

    # Run test
    answers, _ = await get_initial_answers(test_data["question"])

    # Assertions
    assert len(answers) == 2
    assert "openai_o3" in answers
    assert "claude_36" in answers
    mock_crew.assert_called_once()
    assert mock_crew_instance.kickoff.await_count == 1


@patch("src.main.get_answering_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_get_initial_answers_error(mock_crew, mock_get_agents, test_data):
    """Test error handling in getting initial answers."""
    # Setup error condition
    mock_agents = {
        "openai_o3": MockAgent(role="openai_o3"),
        "claude_36": MockAgent(role="claude_36"),
    }
    mock_get_agents.return_value = mock_agents

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.side_effect = RuntimeError("Network error")
    mock_crew.return_value = mock_crew_instance

    # Test error handling
    with pytest.raises(RuntimeError):
        await get_initial_answers(test_data["question"])
    assert mock_crew_instance.kickoff.await_count == 1


@pytest.mark.parametrize(
    "agent_responses,expected_count",
    [
        ([], 0),
        ([MockCrewOutput("single answer")], 1),
        ([MockCrewOutput("answer1"), MockCrewOutput("answer2")], 2),
    ],
)
@patch("src.main.get_answering_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_get_initial_answers_response_counts(
    mock_crew, mock_get_agents, agent_responses, expected_count, test_data
):
    """Test getting initial answers with different response counts."""
    # Setup mocks with matching number of agents
    mock_agents = {f"agent{i}": MockAgent(role=f"agent{i}") for i in range(len(agent_responses))}
    mock_get_agents.return_value = mock_agents

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = agent_responses
    mock_crew.return_value = mock_crew_instance

    answers, _ = await get_initial_answers(test_data["question"])
    assert len(answers) == expected_count
    assert mock_crew_instance.kickoff.await_count == 1


@patch("src.main.get_evaluator_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_evaluate_answers_success(mock_crew, mock_get_evaluators, test_data):
    """Test successful evaluation of answers."""
    # Setup mocks
    mock_evaluators = {
        "evaluator1": MockAgent(role="evaluator1"),
        "evaluator2": MockAgent(role="evaluator2"),
    }
    mock_get_evaluators.return_value = mock_evaluators

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = [
        MockCrewOutput(
            json.dumps(
                {
                    "criteria": [
                        {"name": "clarity", "weight": 0.5, "description": "Test clarity"},
                        {"name": "depth", "weight": 0.5, "description": "Test depth"},
                    ],
                    "scores": [
                        {
                            "answer_id": "Answer 1",
                            "criteria_scores": {"clarity": 8, "depth": 7},
                            "reasoning": "Clear but could be deeper",
                        }
                    ],
                }
            )
        )
    ]
    mock_crew.return_value = mock_crew_instance

    # Run test
    evaluations = await evaluate_answers(
        test_data["question"], test_data["answers"], test_data["run_logger"]
    )

    # Assertions
    assert len(evaluations) > 0
    mock_crew.assert_called_once()
    assert mock_crew_instance.kickoff.await_count == 1


@patch("src.main.get_evaluator_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_evaluate_answers_invalid_json(mock_crew, mock_get_evaluators, test_data):
    """Test handling of invalid JSON in evaluation responses."""
    mock_evaluators = {
        "evaluator1": MockAgent(role="evaluator1"),
    }
    mock_get_evaluators.return_value = mock_evaluators

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = [MockCrewOutput("invalid json")]
    mock_crew.return_value = mock_crew_instance

    # The evaluate_answers function should handle the JSON decode error and return an empty list
    evaluations = await evaluate_answers(
        test_data["question"], test_data["answers"], test_data["run_logger"]
    )
    assert len(evaluations) == 0  # No valid evaluations due to JSON error
    assert mock_crew_instance.kickoff.await_count == 1


@patch("src.main.get_improver_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_improve_answers_success(mock_crew, mock_get_improvers, test_data):
    """Test successful improvement of answers."""
    # Setup mocks
    mock_improvers = {
        "improver1": MockAgent(role="improver1"),
        "improver2": MockAgent(role="improver2"),
    }
    mock_get_improvers.return_value = mock_improvers

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = [
        MockCrewOutput(
            json.dumps(
                {
                    "improved_answer": "The meaning of life is to maximize happiness and understanding.",
                    "improvements": "Added understanding aspect",
                }
            )
        )
    ]
    mock_crew.return_value = mock_crew_instance

    # Run test
    improved = await improve_answers(
        test_data["question"],
        ["openai_o3"],
        test_data["answers"],
        test_data["evaluations"],
        test_data["run_logger"],
    )

    # Assertions
    assert len(improved) > 0
    mock_crew.assert_called_once()
    assert mock_crew_instance.kickoff.await_count == 1


@patch("src.main.get_improver_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_improve_answers_empty_feedback(mock_crew, mock_get_improvers, test_data):
    """Test improving answers with empty feedback."""
    mock_improvers = {
        "improver1": MockAgent(role="improver1"),
    }
    mock_get_improvers.return_value = mock_improvers

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = [
        MockCrewOutput(
            json.dumps(
                {
                    "improved_answer": "The meaning of life is to maximize happiness.",
                    "improvements": "No feedback to incorporate",
                }
            )
        )
    ]
    mock_crew.return_value = mock_crew_instance

    empty_evaluations = []
    improved = await improve_answers(
        test_data["question"],
        ["openai_o3"],
        test_data["answers"],
        empty_evaluations,
        test_data["run_logger"],
    )
    assert improved is not None
    assert mock_crew_instance.kickoff.await_count == 1


@patch("src.main.get_evaluator_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_final_judgment_success(mock_crew, mock_get_evaluators, test_data):
    """Test successful final judgment."""
    mock_evaluators = {
        "claude_37": MockAgent(role="claude_37"),
    }
    mock_get_evaluators.return_value = mock_evaluators

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = [
        MockCrewOutput(
            json.dumps(
                {
                    "best_answer_id": "Improved Answer 1",
                    "reasoning": "Most comprehensive answer",
                    "final_score": 9.5,
                }
            )
        )
    ]
    mock_crew.return_value = mock_crew_instance

    best_answer, score = await final_judgment(
        test_data["question"], test_data["improved_answers"], test_data["run_logger"]
    )
    assert best_answer is not None
    assert score == 9.5
    assert mock_crew_instance.kickoff.await_count == 1


@pytest.mark.integration
@pytest.mark.asyncio
@patch("src.main.get_answering_agents")
@patch("src.main.get_evaluator_agents")
@patch("src.main.get_improver_agents")
@patch("src.main.Crew")
async def test_full_workflow_integration(
    mock_crew, mock_get_improvers, mock_get_evaluators, mock_get_agents, test_data
):
    """Test the full workflow from question to final answer."""
    # Setup mocks for each stage
    mock_agents = {
        "openai_o3": MockAgent(role="openai_o3"),
        "claude_36": MockAgent(role="claude_36"),
    }
    mock_get_agents.return_value = mock_agents

    mock_evaluators = {
        "evaluator1": MockAgent(role="evaluator1"),
        "evaluator2": MockAgent(role="evaluator2"),
    }
    mock_get_evaluators.return_value = mock_evaluators

    mock_improvers = {
        "improver1": MockAgent(role="improver1"),
        "improver2": MockAgent(role="improver2"),
    }
    mock_get_improvers.return_value = mock_improvers

    # Setup mock crew instance with different responses for each stage
    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.side_effect = [
        # Initial answers stage
        [
            MockCrewOutput("The meaning of life is to maximize happiness."),
            MockCrewOutput("The meaning of life is to seek understanding."),
        ],
        # Evaluation stage
        [
            MockCrewOutput(
                json.dumps(
                    {
                        "criteria": [
                            {"name": "clarity", "weight": 0.5, "description": "Test clarity"},
                            {"name": "depth", "weight": 0.5, "description": "Test depth"},
                        ],
                        "scores": [
                            {
                                "answer_id": "Answer 1",
                                "criteria_scores": {"clarity": 8, "depth": 7},
                                "reasoning": "Clear but could be deeper",
                            }
                        ],
                    }
                )
            )
        ],
        # Improvement stage
        [
            MockCrewOutput(
                json.dumps(
                    {
                        "improved_answer": "The meaning of life is to maximize happiness and understanding.",
                        "improvements": "Added understanding aspect",
                    }
                )
            )
        ],
        # Final judgment stage
        [
            MockCrewOutput(
                json.dumps(
                    {
                        "best_answer_id": "Improved Answer 1",
                        "reasoning": "Most comprehensive answer",
                        "final_score": 9.5,
                    }
                )
            )
        ],
    ]
    mock_crew.return_value = mock_crew_instance

    # Step 1: Get initial answers
    answers, run_logger = await get_initial_answers(test_data["question"])
    assert len(answers) == 2

    # Step 2: Evaluate answers
    evaluations = await evaluate_answers(test_data["question"], answers, run_logger)
    assert len(evaluations) > 0

    # Step 3: Improve best answers
    best_agent_ids = ["openai_o3"]  # Simulating selection of best answer
    improved_answers = await improve_answers(
        test_data["question"], best_agent_ids, answers, evaluations, run_logger
    )
    assert len(improved_answers) > 0

    # Step 4: Final judgment
    best_answer, score = await final_judgment(test_data["question"], improved_answers, run_logger)
    assert best_answer is not None
    assert score == 9.5

    # Verify all stages were called
    assert mock_crew_instance.kickoff.await_count == 4
    assert mock_crew_instance.kickoff.call_count == 4


@pytest.mark.slow
@pytest.mark.asyncio
@patch("src.main.get_improver_agents")
@patch("src.main.Crew")
async def test_improve_answers_performance(mock_crew, mock_get_improvers, test_data):
    """Test the performance of improving answers with a larger dataset."""
    # Setup mocks
    mock_improvers = {
        "improver1": MockAgent(role="improver1"),
        "improver2": MockAgent(role="improver2"),
    }
    mock_get_improvers.return_value = mock_improvers

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = [
        MockCrewOutput(
            json.dumps(
                {
                    "improved_answer": f"Improved answer {i}",
                    "improvements": f"Improvements for answer {i}",
                }
            )
        )
        for i in range(5)  # Simulate 5 improvers working in parallel
    ]
    mock_crew.return_value = mock_crew_instance

    # Create a larger dataset
    best_agent_ids = ["agent1", "agent2", "agent3"]
    answers = {
        f"agent{i}": Answer(content=f"Original answer {i}", agent_id=f"agent{i}")
        for i in range(1, 4)
    }
    evaluations = [test_data["evaluations"][0] for _ in range(3)]  # Duplicate evaluations

    # Measure performance
    import time

    start_time = time.time()

    improved_answers = await improve_answers(
        test_data["question"],
        best_agent_ids,
        answers,
        evaluations,
        test_data["run_logger"],
    )

    end_time = time.time()
    duration = end_time - start_time

    # Assertions
    assert len(improved_answers) > 0
    assert duration < 5.0  # Should complete in under 5 seconds
    mock_crew.assert_called()
    assert mock_crew_instance.kickoff.await_count == 3  # One call per agent
