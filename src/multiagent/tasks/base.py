from typing import Dict, List, Any, Optional
from uuid import uuid4
from crewai import Task, Agent
from multiagent.models.base import Answer, ImprovedAnswer


def create_answer_task(question: str, agent: Agent, async_execution: bool = False) -> Task:
    """
    Create a task to generate an answer to the question.

    Args:
        question: The question to answer
        agent: The agent to assign the task to
        async_execution: Whether to execute the task asynchronously

    Returns:
        Task to generate an answer
    """
    task_description = f"""
    Your task is to provide the best possible answer to the following question:
    
    "{question}"
    
    Guidelines:
    1. Be thorough yet concise.
    2. Consider multiple perspectives if relevant.
    3. Structure your answer clearly.
    4. Use examples or analogies if they help clarify concepts.
    5. Be honest about uncertainty rather than making claims you cannot support.
    
    Your response will be anonymized and evaluated by other AI agents, so focus on quality rather than personal style.
    
    You can use whatever reasoning tokens you want, but the final answer should be at the end of the response like so:
    FINAL_ANSWER: <your answer here>
    
    THERE SHOULD BE NOTHING ELSE AFTER THE FINAL ANSWER.
    """

    return Task(
        description=task_description,
        agent=agent,
        expected_output="A comprehensive, clear, and accurate answer to the question.",
        async_execution=async_execution,
    )


def create_evaluation_task(
    question: str, anonymized_answers: Dict[str, str], agent: Agent, async_execution: bool = False
) -> Task:
    """
    Create a task to evaluate anonymized answers.

    Args:
        question: The original question
        anonymized_answers: Dictionary mapping anonymized IDs to answer content
        agent: The agent to assign the task to
        async_execution: Whether to execute the task asynchronously

    Returns:
        Task to evaluate the answers
    """
    # Convert answers dict to formatted string
    answers_text = "\n\n".join(
        [
            f"{answer_id}:\n{answer_content}"
            for answer_id, answer_content in anonymized_answers.items()
        ]
    )

    task_description = f"""
    Your task is to objectively evaluate the following anonymized answers to this question:
    
    Question: "{question}"
    
    Answers:
    {answers_text}
    
    Follow these steps:
    
    1. Define 3-5 evaluation criteria relevant to this specific question. Examples MIGHT include: 
       - Accuracy/Correctness
       - Comprehensiveness
       - Clarity/Communication
       - Use of evidence/examples
       - Practical applicability
       - Logical reasoning
       
       But remember that you are the one who is defining the criteria, so you can define whatever criteria fits the question and the answers.
       
    2. Assign each criterion a weight (0.0-1.0) based on its importance to this question, with weights summing to 1.0.
    
    3. For each answer, assign a score (0-10) for each criterion, explaining your reasoning.
    
    4. Calculate a weighted total score for each answer.
    
    5. Provide specific feedback on each answer's strengths and weaknesses.
    
    You can use reasoning before you build the json object, but the json object HAS TO BE AT THE END OF YOUR RESPONSE.
    YOU SHOULD NOT HAVE ANYTHING BUT VALID JSON AFTER YOU START BUILDING THE JSON OBJECT.
    
    json structure:
    ```json
    {{
      "criteria": [
        {{ "name": "criterion1", "description": "description1", "weight": 0.X }},
        {{ "name": "criterion2", "description": "description2", "weight": 0.Y }}
        // ... other criteria
      ],
      "scores": [
        {{
          "reasoning": {{
            "strengths": "Clear explanation of key concepts...",
            "weaknesses": "Could provide more examples...",
            "improvement_suggestions": "Consider adding..."
          }},
          "answer_id": "Answer X",
          "criteria_scores": {{ "criterion1": 8, "criterion2": 7 }}
        }},
        // ... scores for other answers IN THE EXACT SAME FORMAT
      ]
    }}
    ```
    
    REASONING HAS TO GO BEFORE ANY EVALUATION SCORES.
    
    Be objective and fair in your evaluation, using the same standards across all answers.
    """

    return Task(
        description=task_description,
        agent=agent,
        expected_output="A JSON object with evaluation criteria, weights, and scores for each answer.",
        async_execution=async_execution,
    )


def create_improvement_task(
    question: str,
    original_answer: str,
    evaluation_feedback: str,
    agent: Agent,
    async_execution: bool = False,
) -> Task:
    """
    Create a task to improve an answer based on evaluation feedback.

    Args:
        question: The original question
        original_answer: The answer to improve
        evaluation_feedback: Feedback from the evaluation
        agent: The agent to assign the task to
        async_execution: Whether to execute the task asynchronously

    Returns:
        Task to improve the answer
    """
    task_description = f"""
    Your task is to improve the following answer based on evaluation feedback:
    
    Question: "{question}"
    
    Original Answer:
    {original_answer}
    
    Evaluation Feedback:
    {evaluation_feedback}
    
    Guidelines:
    1. Address specific weaknesses mentioned in the feedback.
    2. Preserve the strengths of the original answer.
    3. Maintain or improve clarity and conciseness.
    4. Add more evidence, examples, or reasoning where needed.
    5. Correct any inaccuracies identified in the feedback.
    
    Format your response as a JSON object with the following structure:
    ```json
    {{
      "improved_answer": "Your complete improved answer here",
      "improvements": "A brief summary of the main improvements you made"
    }}
    ```
    
    Focus on creating a substantively better answer, not just superficial changes.
    """

    return Task(
        description=task_description,
        agent=agent,
        expected_output="A JSON object with the improved answer and a summary of improvements.",
        async_execution=async_execution,
    )


def create_final_judgment_task(
    question: str, improved_answers: Dict[str, str], agent: Agent, async_execution: bool = False
) -> Task:
    """
    Create a task for final judgment of improved answers.

    Args:
        question: The original question
        improved_answers: Dictionary mapping anonymized IDs to improved answer content
        agent: The agent to assign the task to
        async_execution: Whether to execute the task asynchronously

    Returns:
        Task for final judgment
    """
    # Convert answers dict to formatted string
    answers_text = "\n\n".join(
        [
            f"{answer_id}:\n{answer_content}"
            for answer_id, answer_content in improved_answers.items()
        ]
    )

    task_description = f"""
    Your task is to analyze the improved answers and provide a final answer that can either be exactly the same as one of the improved answers or a combination of the improved answers.
    
    Question: "{question}"
    
    Improved Answers:
    {answers_text}
    
    Please provide:
    1. A thorough analysis of each answer's strengths and weaknesses
    2. The final answer that you believe is the best solution to the question
    3. A confidence score (0-10) indicating how strong you believe the final solution is
    
    Focus on providing valuable insights that will help users understand the best way to solve their problem.
    You may combine ideas from multiple answers if that produces the best solution.
    
    Be thorough in your explanation but also clear and practical in your recommendations.
    """

    return Task(
        description=task_description,
        agent=agent,
        expected_output="A detailed analysis and recommendation for the best solution.",
        async_execution=async_execution,
    )
