from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class Answer(BaseModel):
    """Model representing an answer from an agent."""

    content: str
    agent_id: str

    def __str__(self) -> str:
        return self.content


class EvaluationCriterion(BaseModel):
    """Model representing a criterion for evaluating answers."""

    name: str
    description: str
    weight: float = Field(ge=0.0, le=1.0)


class AnswerScore(BaseModel):
    """Model representing a score for an answer."""

    answer_id: str
    criteria_scores: Dict[str, float]  # criterion_name -> score
    total_score: float
    reasoning: str

    @classmethod
    def calculate_total_score(
        cls, criteria_scores: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Calculate the weighted total score."""
        total = 0.0
        for criterion, score in criteria_scores.items():
            total += score * weights.get(criterion, 1.0)
        return total


class Evaluation(BaseModel):
    """Model representing a complete evaluation."""

    criteria: List[EvaluationCriterion]
    scores: List[AnswerScore]
    evaluator_id: str
    question: str

    def get_best_answers(self, top_n: int = 2, threshold: float = 0.1) -> List[str]:
        """
        Get the top N answers plus any that are within threshold of the Nth best.

        Args:
            top_n: Number of top answers to select
            threshold: Score difference threshold for including additional answers

        Returns:
            List of answer IDs for the best answers
        """
        if not self.scores:
            return []

        # Sort scores in descending order
        sorted_scores = sorted(self.scores, key=lambda x: x.total_score, reverse=True)

        # Always include the top N answers
        result = [score.answer_id for score in sorted_scores[: min(top_n, len(sorted_scores))]]

        # If we have more than N answers, check if any are within threshold
        if len(sorted_scores) > top_n:
            threshold_value = sorted_scores[top_n - 1].total_score - threshold
            for score in sorted_scores[top_n:]:
                if score.total_score >= threshold_value:
                    result.append(score.answer_id)
                else:
                    # Since the list is sorted, we can break once we find a score below threshold
                    break

        return result


class ImprovedAnswer(BaseModel):
    """Model representing an improved answer."""

    original_answer_id: str
    content: str
    agent_id: str
    improvements: str

    def __str__(self) -> str:
        return self.content
