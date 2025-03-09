import json
import os
import logging
from typing import Dict, List, Tuple, Any

from multiagent.models.base import (
    Answer,
    Evaluation,
    EvaluationCriterion,
    AnswerScore,
    ImprovedAnswer,
)

# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def anonymize_answers(answers: Dict[str, Answer]) -> Dict[str, str]:
    """
    Anonymize answers by replacing agent IDs with generic identifiers.

    Args:
        answers: Dictionary mapping agent IDs to Answer objects

    Returns:
        Dictionary mapping anonymized IDs to answer content
    """
    return {f"Answer {i+1}": str(answer) for i, answer in enumerate(answers.values())}


def deanonymize_answers(
    anonymized_ids: List[str], original_answers: Dict[str, Answer]
) -> List[str]:
    """
    Convert anonymized answer IDs back to original agent IDs.

    Args:
        anonymized_ids: List of anonymized answer IDs (e.g., "Answer 1")
        original_answers: Dictionary mapping agent IDs to Answer objects

    Returns:
        List of original agent IDs
    """
    # Create a mapping from anonymized ID to original agent ID
    anonymized_map = {
        f"Answer {i+1}": agent_id for i, agent_id in enumerate(original_answers.keys())
    }

    return [anonymized_map.get(anon_id) for anon_id in anonymized_ids if anon_id in anonymized_map]


def parse_evaluation_result(evaluation_text: str, question: str, evaluator_id: str) -> Evaluation:
    """
    Parse the evaluation text from an agent into structured evaluation data.

    This is complex and might require some prompt engineering to get the agent
    to return data in a specific format.

    Args:
        evaluation_text: Raw evaluation text from an agent
        question: The original question
        evaluator_id: ID of the evaluating agent

    Returns:
        Structured Evaluation object

    Raises:
        json.JSONDecodeError: If the evaluation text cannot be parsed as JSON
        ValueError: If the parsed JSON does not have the expected structure
    """
    # Try parsing as JSON first
    try:
        if "```json" in evaluation_text:
            json_start = evaluation_text.find("```json") + 7
            json_end = evaluation_text.find("```", json_start)
            data = json.loads(evaluation_text[json_start:json_end].strip())
        else:
            data = json.loads(evaluation_text)

        criteria = [
            EvaluationCriterion(name=c["name"], description=c["description"], weight=c["weight"])
            for c in data.get("criteria", [])
        ]

        weights = {c.name: c.weight for c in criteria}

        scores = []
        for score_data in data.get("scores", []):
            criteria_scores = score_data.get("criteria_scores", {})
            total_score = AnswerScore.calculate_total_score(criteria_scores, weights)
            scores.append(
                AnswerScore(
                    answer_id=score_data["answer_id"],
                    criteria_scores=criteria_scores,
                    total_score=total_score,
                    reasoning=score_data.get("reasoning", ""),
                )
            )

        return Evaluation(
            criteria=criteria, scores=scores, evaluator_id=evaluator_id, question=question
        )
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing evaluation result: {e}")
        logger.debug(f"Raw evaluation text: {evaluation_text}")
        raise  # Re-raise the JSON decode error
    except Exception as e:
        logger.error(f"Error parsing evaluation result: {e}")
        logger.debug(f"Raw evaluation text: {evaluation_text}")
        raise ValueError(f"Invalid evaluation format: {e}") from e


def parse_improvement_result(
    improvement_text: str, original_answer_id: str, agent_id: str
) -> ImprovedAnswer:
    """
    Parse the improvement text from an agent into an ImprovedAnswer object.

    Args:
        improvement_text: Raw improvement text from an agent
        original_answer_id: ID of the original answer being improved
        agent_id: ID of the agent providing the improvement

    Returns:
        ImprovedAnswer object
    """
    # In a real implementation, you might need more robust parsing
    try:
        if "```json" in improvement_text:
            json_start = improvement_text.find("```json") + 7
            json_end = improvement_text.find("```", json_start)
            data = json.loads(improvement_text[json_start:json_end].strip())

            return ImprovedAnswer(
                original_answer_id=original_answer_id,
                content=data.get("improved_answer", improvement_text),
                agent_id=agent_id,
                improvements=data.get("improvements", ""),
            )
        else:
            # Simpler fallback - assume the entire text is the improved answer
            return ImprovedAnswer(
                original_answer_id=original_answer_id,
                content=improvement_text,
                agent_id=agent_id,
                improvements="",
            )
    except Exception as e:
        logger.error(f"Error parsing improvement result: {e}")
        logger.debug(f"Raw improvement text: {improvement_text}")

        # Fallback - return the original text
        return ImprovedAnswer(
            original_answer_id=original_answer_id,
            content=improvement_text,
            agent_id=agent_id,
            improvements="",
        )


def combine_evaluations(evaluations: List[Evaluation]) -> Dict[str, float]:
    """
    Combine multiple evaluations into a single set of average scores.

    Args:
        evaluations: List of Evaluation objects from different evaluators

    Returns:
        Dictionary mapping answer IDs to their average scores
    """
    if not evaluations:
        return {}

    # Combine all scores for each answer
    all_scores = {}
    for eval in evaluations:
        for score in eval.scores:
            if score.answer_id not in all_scores:
                all_scores[score.answer_id] = []
            all_scores[score.answer_id].append(score.total_score)

    # Calculate average scores
    average_scores = {
        answer_id: sum(scores) / len(scores) for answer_id, scores in all_scores.items()
    }

    return average_scores
