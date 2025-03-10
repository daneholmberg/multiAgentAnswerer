#!/usr/bin/env python3
import json
import logging
import re
from typing import Dict, Any, List

from multiagent.models.base import Answer, Evaluation, ImprovedAnswer
from multiagent.utils.agent_task_manager import AgentTaskManager
from multiagent.utils.run_logger import RunLogger

logger = logging.getLogger(__name__)


async def process_answer_result(
    agent_id: str, agent: Any, raw_result: str, run_logger: RunLogger
) -> Answer:
    """
    Process raw result from an answering agent.

    Args:
        agent_id: ID of the agent
        agent: Agent object
        raw_result: Raw result from the agent
        run_logger: RunLogger instance

    Returns:
        Answer object
    """
    # Extract the final answer from the raw result
    final_answer_match = re.search(r"FINAL_ANSWER:\s*(.*?)(?:\n|$)", raw_result, re.DOTALL)
    if final_answer_match:
        answer_content = final_answer_match.group(1).strip()
    else:
        # If no FINAL_ANSWER found, use the entire content
        answer_content = raw_result.strip()

    # Log the answer
    await run_logger.log_initial_answer(agent, answer_content)

    return Answer(content=answer_content, agent_id=agent_id)


async def process_evaluation_result(
    agent_id: str, agent: Any, raw_result: str, run_logger: RunLogger
) -> Evaluation:
    """
    Process raw result from an evaluator agent.

    Args:
        agent_id: ID of the agent
        agent: Agent object
        raw_result: Raw result from the agent
        run_logger: RunLogger instance

    Returns:
        Evaluation object
    """
    # Extract JSON from the result
    try:
        parsed_result = AgentTaskManager.extract_json_from_result(raw_result)

        if not parsed_result:
            raise ValueError("Could not extract valid JSON from evaluation result")

        # Validate and normalize the reasoning format for each score
        for score in parsed_result.get("scores", []):
            if "reasoning" not in score:
                raise ValueError(f"Missing 'reasoning' field in score: {score}")

            reasoning = score["reasoning"]
            if not isinstance(reasoning, dict):
                raise ValueError(f"Invalid reasoning format (must be dict): {reasoning}")

            # Ensure all required fields are present
            required_fields = {"strengths", "weaknesses", "improvement_suggestions"}
            missing_fields = required_fields - set(reasoning.keys())
            if missing_fields:
                raise ValueError(f"Missing required reasoning fields: {missing_fields}")

        # Create Evaluation object
        evaluation = Evaluation.from_dict(
            parsed_result, evaluator_id=agent_id, question=run_logger.question
        )

        # Log the evaluation
        await run_logger.log_evaluation(agent, parsed_result)

        return evaluation
    except Exception as e:
        error_msg = (
            f"Failed to process evaluation from {agent_id}: {str(e)}\nRaw result: {raw_result}"
        )
        logger.error(error_msg)
        await run_logger.log_event("error", "process_evaluation", {"error": error_msg})
        raise ValueError(error_msg)


async def process_improvement_result(
    agent_id: str,
    agent: Any,
    raw_result: str,
    run_logger: RunLogger,
    best_agent_id: str = None,
    original_answer: str = None,
) -> Dict[str, Any]:
    """
    Process raw result from an improver agent.

    Args:
        agent_id: ID of the agent
        agent: Agent object
        raw_result: Raw result from the agent
        run_logger: RunLogger instance
        best_agent_id: ID of the agent whose answer is being improved
        original_answer: Original answer content

    Returns:
        Dictionary with improvement data
    """
    # Extract JSON from the result
    default_structure = {
        "improved_answer": raw_result,
        "improvements": "Extracted from unstructured response",
    }

    parsed_result = AgentTaskManager.extract_json_from_result(raw_result, default_structure)

    # Calculate improvement score if original answer is provided
    if original_answer:
        improvement_score = len(parsed_result.get("improved_answer", "")) / max(
            len(original_answer), 1
        )
        parsed_result["improvement_score"] = improvement_score

    # Log the improvement
    if best_agent_id:
        await run_logger.log_improvement(agent, best_agent_id, parsed_result)

    return parsed_result


async def process_judgment_result(
    agent_id: str, agent: Any, raw_result: str, run_logger: RunLogger
) -> Dict[str, Any]:
    """
    Process raw result from a final judgment agent.

    Args:
        agent_id: ID of the agent
        agent: Agent object
        raw_result: Raw result from the agent
        run_logger: RunLogger instance

    Returns:
        Dictionary with final judgment and confidence score
    """
    final_response = raw_result

    # Try to extract a confidence score - default to 0.0 if not found
    confidence_score = 0.0
    try:
        # Look for a score in the response (flexible matching)
        score_match = re.search(
            r"(?:score|rating|confidence)\"?\s*:\s*(\d+\.?\d*)", final_response, re.IGNORECASE
        )
        if score_match:
            confidence_score = float(score_match.group(1))
    except (ValueError, AttributeError):
        pass

    # Log the judgment
    await run_logger.log_final_judgment(agent, final_response)

    return {"final_response": final_response, "confidence_score": confidence_score}
