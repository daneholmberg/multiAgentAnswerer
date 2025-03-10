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

    # Run test with selected models
    selected_models = {"openai_o3", "claude_36"}
    answers, _ = await get_initial_answers(test_data["question"], selected_models)

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
    """Test error handling in get_initial_answers."""
    # Setup mocks
    mock_agents = {
        "openai_o3": MockAgent(role="openai_o3"),
        "claude_36": MockAgent(role="claude_36"),
    }
    mock_get_agents.return_value = mock_agents

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.side_effect = Exception("Test exception")
    mock_crew.return_value = mock_crew_instance

    # Run test with selected models
    selected_models = {"openai_o3", "claude_36"}
    with pytest.raises(Exception):
        await get_initial_answers(test_data["question"], selected_models)
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
    """Test that the response counts are correct for various inputs."""
    # Setup mocks
    mock_agents = {
        "openai_o3": MockAgent(role="openai_o3"),
        "claude_36": MockAgent(role="claude_36"),
    }
    mock_get_agents.return_value = mock_agents

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = agent_responses
    mock_crew.return_value = mock_crew_instance

    # Run test with selected models
    selected_models = {"openai_o3", "claude_36"}
    answers, _ = await get_initial_answers(test_data["question"], selected_models)

    # Assertions
    assert len(answers) == expected_count
    assert mock_crew_instance.kickoff.await_count == 1


@patch("src.main.get_evaluator_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_evaluate_answers_success(mock_crew, mock_get_evaluators, test_data):
    """Test successful evaluation of answers."""
    # Setup mocks
    mock_evaluator_agents = {
        "evaluator1": MockAgent(role="evaluator1"),
        "evaluator2": MockAgent(role="evaluator2"),
    }
    mock_get_evaluators.return_value = mock_evaluator_agents

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

    # Create test answers
    answers = {
        "agent1": Answer(content="Test answer 1", agent_id="agent1", latency=0.5),
    }
    run_logger = AsyncMock()

    # Run test with selected models
    selected_models = {"evaluator1", "evaluator2"}
    evaluations = await evaluate_answers(
        test_data["question"], answers, run_logger, selected_models
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
    # Setup mocks with response that's not valid JSON
    mock_evaluator_agents = {"evaluator1": MockAgent(role="evaluator1")}
    mock_get_evaluators.return_value = mock_evaluator_agents

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = [MockCrewOutput("Not a valid JSON response")]
    mock_crew.return_value = mock_crew_instance

    # Create test answers
    answers = {
        "agent1": Answer(content="Test answer 1", agent_id="agent1", latency=0.5),
    }
    run_logger = AsyncMock()

    # Run test with selected models
    selected_models = {"evaluator1"}
    evaluations = await evaluate_answers(
        test_data["question"], answers, run_logger, selected_models
    )

    # Assertions
    assert len(evaluations) == 0  # No valid evaluations due to JSON error
    assert mock_crew_instance.kickoff.await_count == 1


@patch("src.main.get_improver_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_improve_answers_success(mock_crew, mock_get_improvers, test_data):
    """Test successful improvement of answers."""
    # Setup mocks
    mock_improver_agents = {
        "improver1": MockAgent(role="improver1"),
    }
    mock_get_improvers.return_value = mock_improver_agents

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = [
        MockCrewOutput(
            json.dumps(
                {
                    "improved_answer": "This is an improved test answer",
                    "improvements": "Added more details",
                }
            )
        )
    ]
    mock_crew.return_value = mock_crew_instance

    # Create test data
    best_agent_ids = ["agent1"]
    answers = {
        "agent1": Answer(content="Test answer 1", agent_id="agent1", latency=0.5),
    }
    evaluations = [
        Evaluation(
            criteria=[
                {"name": "clarity", "weight": 0.5, "description": "Test clarity"},
                {"name": "depth", "weight": 0.5, "description": "Test depth"},
            ],
            scores=[
                {
                    "answer_id": "Answer 1",
                    "criteria_scores": {"clarity": 8, "depth": 7},
                    "reasoning": "Clear but could be deeper",
                }
            ],
            evaluator_id="evaluator1",
        )
    ]
    run_logger = AsyncMock()

    # Run test with selected models
    selected_models = {"improver1"}
    improved_answers = await improve_answers(
        test_data["question"], best_agent_ids, answers, evaluations, run_logger, selected_models
    )

    # Assertions
    assert len(improved_answers) > 0
    mock_crew.assert_called_once()
    assert mock_crew_instance.kickoff.await_count == 1


@patch("src.main.get_improver_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_improve_answers_empty_feedback(mock_crew, mock_get_improvers, test_data):
    """Test improvement of answers with empty feedback."""
    # Setup mocks
    mock_improver_agents = {
        "improver1": MockAgent(role="improver1"),
    }
    mock_get_improvers.return_value = mock_improver_agents

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = []  # Empty response
    mock_crew.return_value = mock_crew_instance

    # Create test data
    best_agent_ids = ["agent1"]
    answers = {
        "agent1": Answer(content="Test answer 1", agent_id="agent1", latency=0.5),
    }
    evaluations = [
        Evaluation(
            criteria=[
                {"name": "clarity", "weight": 0.5, "description": "Test clarity"},
                {"name": "depth", "weight": 0.5, "description": "Test depth"},
            ],
            scores=[
                {
                    "answer_id": "Answer 1",
                    "criteria_scores": {"clarity": 8, "depth": 7},
                    "reasoning": "Clear but could be deeper",
                }
            ],
            evaluator_id="evaluator1",
        )
    ]
    run_logger = AsyncMock()

    # Run test with selected models
    selected_models = {"improver1"}
    improved_answers = await improve_answers(
        test_data["question"], best_agent_ids, answers, evaluations, run_logger, selected_models
    )

    # Assertions
    assert improved_answers is not None
    assert mock_crew_instance.kickoff.await_count == 1


@patch("src.main.get_evaluator_agents")
@patch("src.main.Crew")
@pytest.mark.asyncio
async def test_final_judgment_success(mock_crew, mock_get_evaluators, test_data):
    """Test successful final judgment."""
    # Setup mocks
    mock_evaluator_agents = {
        "evaluator1": MockAgent(role="evaluator1"),
    }
    mock_get_evaluators.return_value = mock_evaluator_agents

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

    # Create test improved answers
    improved_answers = {
        "agent1": ImprovedAnswer(
            content="Improved test answer 1",
            agent_id="agent1",
            improvements="Added details",
            original_answer="Original test answer 1",
        ),
    }
    run_logger = AsyncMock()

    # Run test with selected models
    selected_models = {"evaluator1"}
    best_answer, score = await final_judgment(
        test_data["question"], improved_answers, run_logger, selected_models
    )

    # Assertions
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
    """Test the full workflow integration."""
    # Setup mocks for answering agents
    mock_answering_agents = {
        "openai_o3": MockAgent(role="openai_o3"),
        "claude_36": MockAgent(role="claude_36"),
    }
    mock_get_agents.return_value = mock_answering_agents

    # Setup mocks for evaluator agents
    mock_evaluator_agents = {
        "evaluator1": MockAgent(role="evaluator1"),
        "evaluator2": MockAgent(role="evaluator2"),
    }
    mock_get_evaluators.return_value = mock_evaluator_agents

    # Setup mocks for improver agents
    mock_improver_agents = {
        "improver1": MockAgent(role="improver1"),
        "improver2": MockAgent(role="improver2"),
    }
    mock_get_improvers.return_value = mock_improver_agents

    # Setup mock crew instance
    mock_crew_instance = AsyncMock()

    # Configure kickoff for different stages
    mock_crew_instance.kickoff.side_effect = [
        # Initial answers
        [MockCrewOutput("Answer from agent 1"), MockCrewOutput("Answer from agent 2")],
        # Evaluations
        [
            MockCrewOutput(
                json.dumps(
                    {
                        "agent_id": "openai_o3",
                        "score": 8.5,
                        "feedback": "Good answer but could be more detailed",
                    }
                )
            ),
            MockCrewOutput(
                json.dumps(
                    {
                        "agent_id": "claude_36",
                        "score": 7.2,
                        "feedback": "Decent answer but lacks examples",
                    }
                )
            ),
        ],
        # Improved answers
        [
            MockCrewOutput("Improved answer from agent 1 with more details"),
            MockCrewOutput("Improved answer from agent 2 with examples"),
        ],
        # Final judgment
        [
            MockCrewOutput(
                json.dumps(
                    {
                        "best_answer": "Improved answer from agent 1 with more details",
                        "score": 9.2,
                        "reasoning": "This answer is more comprehensive",
                    }
                )
            )
        ],
    ]

    mock_crew.return_value = mock_crew_instance

    # Test the full workflow
    selected_models = {
        "openai_o3",
        "claude_36",
        "evaluator1",
        "evaluator2",
        "improver1",
        "improver2",
    }
    answers, run_logger = await get_initial_answers(test_data["question"], selected_models)
    assert len(answers) == 2

    evaluations = await evaluate_answers(
        test_data["question"], answers, run_logger, selected_models
    )
    assert len(evaluations) > 0

    best_agent_ids = select_best_answers(evaluations, answers)
    assert len(best_agent_ids) > 0

    improved_answers = await improve_answers(
        test_data["question"], best_agent_ids, answers, evaluations, run_logger, selected_models
    )
    assert len(improved_answers) > 0

    best_answer, score = await final_judgment(
        test_data["question"], improved_answers, run_logger, selected_models
    )
    assert best_answer is not None
    assert score == 9.2

    # Verify all stages were called
    assert mock_crew_instance.kickoff.await_count == 4
    assert mock_crew_instance.kickoff.call_count == 4


@pytest.mark.slow
@pytest.mark.asyncio
@patch("src.main.get_improver_agents")
@patch("src.main.Crew")
async def test_improve_answers_performance(mock_crew, mock_get_improvers, test_data):
    """Test performance of improve_answers with many agents."""
    # Create a large number of mock agents
    num_agents = 5
    mock_improver_agents = {
        f"improver{i}": MockAgent(role=f"improver{i}") for i in range(num_agents)
    }
    mock_get_improvers.return_value = mock_improver_agents

    # Create a mock response for each agent
    agent_responses = [
        MockCrewOutput(
            json.dumps(
                {
                    "improved_answer": f"Improved answer {i}",
                    "improvements": f"Improvements {i}",
                }
            )
        )
        for i in range(num_agents)
    ]

    mock_crew_instance = AsyncMock()
    mock_crew_instance.kickoff.return_value = agent_responses
    mock_crew.return_value = mock_crew_instance

    # Create test data
    best_agent_ids = ["agent1", "agent2"]
    answers = {
        f"agent{i}": Answer(content=f"Test answer {i}", agent_id=f"agent{i}", latency=0.5)
        for i in range(1, 3)
    }
    evaluations = [
        Evaluation(
            criteria=[
                {"name": "clarity", "weight": 0.5, "description": "Test clarity"},
                {"name": "depth", "weight": 0.5, "description": "Test depth"},
            ],
            scores=[
                {
                    "answer_id": "Answer 1",
                    "criteria_scores": {"clarity": 8, "depth": 7},
                    "reasoning": "Clear but could be deeper",
                }
            ],
            evaluator_id="evaluator1",
        )
    ]
    run_logger = AsyncMock()

    # Run test with selected models
    selected_models = {f"improver{i}" for i in range(num_agents)}
    start_time = time.time()
    improved_answers = await improve_answers(
        test_data["question"], best_agent_ids, answers, evaluations, run_logger, selected_models
    )
    end_time = time.time()

    # Assertions
    assert len(improved_answers) > 0
    assert mock_crew_instance.kickoff.await_count == 1

    # Check performance - should be much faster than sequential execution
    execution_time = end_time - start_time
    logger.info(f"Parallel execution time for {num_agents} agents: {execution_time:.2f}s")
