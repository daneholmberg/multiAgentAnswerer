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
    reasoning: Dict[
        str, str
    ]  # Must contain 'strengths', 'weaknesses', and 'improvement_suggestions'

    @property
    def formatted_reasoning(self) -> str:
        """Format the reasoning into a readable string."""
        if isinstance(self.reasoning, dict):
            # Convert structured feedback into readable format
            formatted_parts = []
            if "strengths" in self.reasoning:
                formatted_parts.append(f"Strengths: {self.reasoning['strengths']}")
            if "weaknesses" in self.reasoning:
                formatted_parts.append(f"Weaknesses: {self.reasoning['weaknesses']}")
            if "improvement_suggestions" in self.reasoning:
                formatted_parts.append(
                    f"Improvement Suggestions: {self.reasoning['improvement_suggestions']}"
                )
            return "\n".join(formatted_parts)
        return str(self.reasoning)

    def __str__(self) -> str:
        return self.formatted_reasoning

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

    @classmethod
    def from_dict(cls, data: dict, evaluator_id: str, question: str = "") -> "Evaluation":
        """
        Create an Evaluation object from a dictionary representing a judge's output.

        Args:
            data: Dictionary containing criteria and scores from judge
            evaluator_id: ID of the evaluator agent
            question: The original question being evaluated

        Returns:
            An Evaluation object

        Raises:
            ValueError: If the data is malformed or missing required fields
        """
        try:
            # Extract criteria from the data
            if "criteria" not in data:
                raise ValueError("Missing 'criteria' field in evaluation data")

            criteria = []
            for criterion_data in data.get("criteria", []):
                if not isinstance(criterion_data, dict):
                    raise ValueError(f"Invalid criterion data format: {criterion_data}")
                if "name" not in criterion_data:
                    raise ValueError(f"Missing 'name' field in criterion: {criterion_data}")

                criteria.append(
                    EvaluationCriterion(
                        name=criterion_data["name"],
                        description=criterion_data.get("description", ""),
                        weight=criterion_data.get("weight", 1.0),
                    )
                )

            # Create a weights dictionary for total score calculation
            weights = {c.name: c.weight for c in criteria}

            # Extract scores from the data
            if "scores" not in data:
                raise ValueError("Missing 'scores' field in evaluation data")

            scores = []
            for score_data in data.get("scores", []):
                if not isinstance(score_data, dict):
                    raise ValueError(f"Invalid score data format: {score_data}")
                if "answer_id" not in score_data:
                    raise ValueError(f"Missing 'answer_id' field in score: {score_data}")

                # Get criteria scores, handling different possible formats
                if "criteria_scores" in score_data:
                    criteria_scores = score_data["criteria_scores"]
                    if not isinstance(criteria_scores, dict):
                        raise ValueError(f"Invalid criteria_scores format: {criteria_scores}")
                else:
                    # Try to construct criteria scores from individual criterion fields
                    criteria_scores = {}
                    for criterion in criteria:
                        if criterion.name not in score_data:
                            raise ValueError(
                                f"Missing score for criterion '{criterion.name}' in: {score_data}"
                            )
                        criteria_scores[criterion.name] = score_data.get(criterion.name, 0)

                # Get or calculate total score
                if "total_score" in score_data:
                    total_score = score_data["total_score"]
                elif "weighted_total" in score_data:
                    total_score = score_data["weighted_total"]
                else:
                    total_score = AnswerScore.calculate_total_score(criteria_scores, weights)

                # Extract and validate reasoning
                reasoning = score_data.get("reasoning", "")
                if not isinstance(reasoning, str):
                    if isinstance(reasoning, dict):
                        # Try to convert dict to string if possible
                        try:
                            import json

                            reasoning = json.dumps(reasoning)
                        except Exception as e:
                            raise ValueError(
                                f"Invalid reasoning format (dict conversion failed): {reasoning}"
                            )
                    else:
                        raise ValueError(f"Invalid reasoning format (must be string): {reasoning}")

                scores.append(
                    AnswerScore(
                        answer_id=score_data["answer_id"],
                        criteria_scores=criteria_scores,
                        total_score=total_score,
                        reasoning=reasoning,
                    )
                )

            return cls(
                criteria=criteria, scores=scores, evaluator_id=evaluator_id, question=question
            )

        except Exception as e:
            # Add context to the error message
            raise ValueError(f"Error processing evaluation data: {str(e)}\nRaw data: {data}") from e

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

    content: str
    original_agent_id: str
    improver_agent_id: Optional[str] = None
    improvements: str = "No improvements specified"

    def __str__(self) -> str:
        return self.content
