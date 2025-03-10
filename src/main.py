#!/usr/bin/env python3
import os
import json
import asyncio
import argparse
import logging
import sys
import re
import traceback
from typing import Dict, List, Any, Set, Tuple
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from rich.logging import RichHandler
from rich.theme import Theme
from rich.style import Style
from crewai import Crew, Agent, Task
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from multiagent.agents.base import (
    get_answering_agents,
    get_evaluator_agents,
    get_improver_agents,
)
from multiagent.tasks.base import (
    create_answer_task,
    create_evaluation_task,
    create_improvement_task,
    create_final_judgment_task,
)
from multiagent.models.base import Answer, Evaluation, ImprovedAnswer
from multiagent.utils.base import (
    anonymize_answers,
    deanonymize_answers,
    parse_evaluation_result,
    parse_improvement_result,
    combine_evaluations,
)
from multiagent.config.model_selection import (
    get_model_selection,
    filter_agents,
    get_final_judge_selection,
    AVAILABLE_MODELS,
)
from multiagent.utils.executor import ThreadedTaskExecutor
from multiagent.utils.run_logger import RunLogger

# Load environment variables
load_dotenv()

# Set up console for rich output with custom theme
custom_theme = Theme(
    {
        "info": Style(color="cyan"),
        "warning": Style(color="yellow"),
        "error": Style(color="red", bold=True),
        "debug": Style(color="grey50"),
        "step.start": Style(color="magenta", bold=True),
        "step.complete": Style(color="green", bold=True),
        "agent.action": Style(color="blue"),
        "progress": Style(color="yellow"),
    }
)
console = Console(theme=custom_theme)

# Set up logging with rich handler
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            enable_link_path=False,
        )
    ],
)
logger = logging.getLogger(__name__)


def format_log_message(message: str, style: str = None) -> str:
    """Format a log message with optional styling."""
    return f"[{style}]{message}[/{style}]" if style else message


def log_step_start(step_name: str) -> None:
    """Log the start of a major processing step."""
    message = format_log_message(f"▶ Starting: {step_name}", "step.start")
    logger.info(message)
    console.print(
        Panel(format_log_message(f"Starting {step_name}...", "step.start"), border_style="magenta")
    )


def log_step_complete(step_name: str) -> None:
    """Log the completion of a major processing step."""
    message = format_log_message(f"✓ Completed: {step_name}", "step.complete")
    logger.info(message)
    console.print(
        Panel(format_log_message(f"Completed {step_name}", "step.complete"), border_style="green")
    )


def log_agent_action(agent_role: str, action: str) -> None:
    """Log an agent's action with nice formatting."""
    message = format_log_message(f"🤖 {agent_role}: {action}", "agent.action")
    logger.info(message)


def log_progress(message: str) -> None:
    """Log a progress message with nice formatting."""
    formatted = format_log_message(f"⏳ {message}", "progress")
    logger.info(formatted)


async def get_initial_answers(
    question: str, selected_models: Set[str]
) -> Tuple[Dict[str, Answer], RunLogger]:
    """
    Get initial answers from selected agents by executing tasks in parallel.

    Args:
        question: The question to answer
        selected_models: Set of model IDs to use

    Returns:
        Dictionary mapping agent IDs to their Answer objects and RunLogger

    Raises:
        TaskExecutionError: If any agent fails to provide an answer
        ValueError: If no agents are available after filtering
    """
    log_step_start("Getting initial answers from selected agents")
    log_progress(f"Processing question: {question}")

    # Initialize the run logger asynchronously
    run_logger = await RunLogger.create(question)

    try:
        console.print(
            Panel(f"[bold cyan]Question:[/bold cyan] {question}", title="Initial Question")
        )

        agents = get_answering_agents()
        # Filter agents based on selected models
        agents = filter_agents(agents, selected_models)

        if not agents:
            raise ValueError("No agents available after filtering with selected models")

        log_progress(f"Using {len(agents)} answering agents")

        # Create answer tasks
        answer_tasks = {}
        answers = {}

        # Create tasks for each agent
        for agent_id, agent in agents.items():
            task = create_answer_task(question, agent, async_execution=False)
            answer_tasks[agent_id] = task

        # Execute all tasks in parallel using ThreadedTaskExecutor
        log_progress("Executing answer tasks in parallel...")
        results = ThreadedTaskExecutor.execute_tasks_by_agent(agents, answer_tasks)

        # Process results
        for agent_id, result in results.items():
            answer_content = str(result)
            answers[agent_id] = Answer(content=answer_content, agent_id=agent_id)
            log_agent_action(agents[agent_id].role, "provided an answer")
            console.print(
                f"[bold green]{agents[agent_id].role}[/bold green] has provided an answer."
            )

            # Log the answer
            await run_logger.log_initial_answer(agents[agent_id], answer_content)

        log_step_complete("Getting initial answers from selected agents")
        return answers, run_logger

    except Exception as e:
        error_msg = f"Failed to get initial answers: {str(e)}"
        logger.error(error_msg)
        # Instead of calling log_error, log it as a general event
        await run_logger.log_event("error", "initial_answers", {"error": error_msg})
        raise


async def evaluate_answers(
    question: str, answers: Dict[str, Answer], run_logger: RunLogger, selected_models: Set[str]
) -> List[Evaluation]:
    """
    Have agents evaluate all answers by executing tasks in parallel.

    Args:
        question: The original question
        answers: Dictionary mapping agent IDs to their Answer objects
        run_logger: RunLogger instance
        selected_models: Set of model IDs to use

    Returns:
        List of Evaluation objects from each evaluator

    Raises:
        ValueError: If no evaluators are available or if evaluation data is invalid
        TaskExecutionError: If any evaluator fails to complete evaluation
    """
    log_step_start("Evaluating answers")
    log_progress(f"Evaluating {len(answers)} answers")

    console.print("\n[bold]Step 2: Evaluating answers...[/bold]")

    try:
        # Anonymize answers
        anonymized_answers = anonymize_answers(answers)
        logger.debug(format_log_message("Answers anonymized successfully", "debug"))

        # Display anonymized answers
        for anon_id, content in anonymized_answers.items():
            console.print(Panel(content, title=f"[bold cyan]{anon_id}[/bold cyan]"))

        # Get evaluator agents and filter based on selected models
        evaluator_agents = get_evaluator_agents()
        evaluator_agents = filter_agents(evaluator_agents, selected_models)

        if not evaluator_agents:
            raise ValueError("No evaluator agents available after filtering with selected models")

        log_progress(f"Using {len(evaluator_agents)} evaluator agents")

        # Create evaluation tasks
        evaluation_tasks = {}

        # Create tasks for each evaluator
        for evaluator_id, agent in evaluator_agents.items():
            task = create_evaluation_task(
                question,
                anonymized_answers,
                agent,
                async_execution=False,
            )
            evaluation_tasks[evaluator_id] = task

        # Execute all tasks in parallel using ThreadedTaskExecutor
        log_progress("Executing evaluation tasks in parallel...")
        results = ThreadedTaskExecutor.execute_tasks_by_agent(evaluator_agents, evaluation_tasks)

        # Process results
        evaluations = []
        for evaluator_id, result in results.items():
            # Parse the evaluation result
            try:
                # Try to extract JSON from the text if needed
                raw_result = str(result)
                try:
                    parsed_result = json.loads(raw_result)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown blocks
                    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_result, re.DOTALL)
                    if json_match:
                        try:
                            parsed_result = json.loads(json_match.group(1))
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON in markdown block: {e}")
                    else:
                        raise ValueError("Could not find JSON in response")

                logger.debug(f"Raw evaluation result from {evaluator_id}: {parsed_result}")

                # Create Evaluation object with enhanced error context
                try:
                    evaluation = Evaluation.from_dict(
                        parsed_result, evaluator_id=evaluator_id, question=question
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Invalid evaluation data from {evaluator_id}: {str(e)}\n"
                        f"Raw result: {raw_result}"
                    )

                evaluations.append(evaluation)
                log_agent_action(evaluator_agents[evaluator_id].role, "provided an evaluation")
                console.print(
                    f"[bold green]{evaluator_agents[evaluator_id].role}[/bold green] has provided an evaluation."
                )

                # Log the evaluation
                await run_logger.log_evaluation(evaluator_agents[evaluator_id], parsed_result)

            except Exception as e:
                error_msg = (
                    f"Failed to process evaluation from {evaluator_id}: {str(e)}\n"
                    f"Raw result: {raw_result}"
                )
                logger.error(error_msg)
                await run_logger.log_event("error", "process_evaluation", {"error": error_msg})
                raise ValueError(error_msg)

        if not evaluations:
            raise ValueError("No valid evaluations were produced")

        log_step_complete("Evaluating answers")
        return evaluations

    except Exception as e:
        error_msg = f"Failed to evaluate answers: {str(e)}"
        logger.error(error_msg)
        await run_logger.log_event("error", "evaluate_answers", {"error": error_msg})
        raise


def select_best_answers(evaluations: List[Evaluation], answers: Dict[str, Answer]) -> List[str]:
    """
    Select the best answers based on evaluations.

    Args:
        evaluations: List of Evaluation objects
        answers: Dictionary mapping agent IDs to their Answer objects

    Returns:
        List of agent IDs for the best answers
    """
    log_step_start("Selecting best answers")
    logger.info(f"Processing {len(evaluations)} evaluations")

    # Display the results table whether we have evaluations or not
    table = Table(title="Answer Scores")
    table.add_column("Answer ID", style="cyan")
    table.add_column("Average Score", style="green")

    # If we have evaluations, combine scores from all evaluators
    if evaluations:
        # Combine evaluations from all evaluators
        average_scores = combine_evaluations(evaluations)
        logger.debug(f"Combined scores: {average_scores}")

        # Sort answers by score
        sorted_answers = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Sorted answers by score: {sorted_answers}")

        # Display scores
        for answer_id, score in sorted_answers:
            table.add_row(answer_id, f"{score:.2f}")

        console.print(table)

        # Get best answers
        if evaluations:
            # Use the first evaluation's selection logic
            # (they should all use similar criteria)
            best_answer_ids = evaluations[0].get_best_answers(top_n=2, threshold=0.1)
            logger.info(f"Selected best answers: {best_answer_ids}")
        else:
            # Fallback if no evaluations
            logger.warning("No evaluations available, falling back to top 2 by score")
            best_answer_ids = [answer_id for answer_id, _ in sorted_answers[:2]]
    else:
        # If we have no evaluations at all, but we have answers, use them directly
        logger.warning("No evaluations available, using all available answers")
        # Get answer IDs that aren't just numerical indexes
        best_answer_ids = list(answers.keys())

        # Display placeholder scores
        for answer_id in best_answer_ids:
            table.add_row(answer_id, "N/A")

        console.print(table)

    # Convert anonymized IDs back to agent IDs if necessary
    if evaluations:
        best_agent_ids = deanonymize_answers(best_answer_ids, answers)
    else:
        best_agent_ids = best_answer_ids

    logger.info(f"Best agent IDs: {best_agent_ids}")

    if best_answer_ids:
        console.print(f"[bold]Selected best answers:[/bold] {', '.join(best_answer_ids)}")
    else:
        console.print("[bold yellow]No answers selected[/bold yellow]")

    log_step_complete("Selecting best answers")
    return best_agent_ids


async def improve_answers(
    question: str,
    best_agent_ids: List[str],
    answers: Dict[str, Answer],
    evaluations: List[Evaluation],
    run_logger: RunLogger,
    selected_models: Set[str],
) -> Dict[str, ImprovedAnswer]:
    """
    Improve the best answers based on evaluation feedback by executing tasks in parallel.

    Args:
        question: The original question
        best_agent_ids: List of agent IDs for the best answers
        answers: Dictionary mapping agent IDs to their Answer objects
        evaluations: List of Evaluation objects
        run_logger: RunLogger instance
        selected_models: Set of model IDs to use

    Returns:
        Dictionary mapping agent IDs to their ImprovedAnswer objects

    Raises:
        TaskExecutionError: If any improver fails to complete their task
        ValueError: If no improvers are available after filtering or if no best answers to improve
    """
    log_step_start("Improving best answers")
    logger.info(f"Improving answers from {len(best_agent_ids)} agents")

    try:
        # Early validation
        if not best_agent_ids:
            raise ValueError("No best answers to improve")

        console.print("\n[bold]Step 4: Improving best answers...[/bold]")

        # Get improver agents and filter based on selected models
        improver_agents = get_improver_agents()
        improver_agents = filter_agents(improver_agents, selected_models)

        if not improver_agents:
            raise ValueError("No improver agents available after filtering with selected models")

        logger.info(f"Using {len(improver_agents)} improver agents")

        improved_answers = {}

        # For each best answer
        for best_agent_id in best_agent_ids:
            logger.info(f"Processing improvements for agent: {best_agent_id}")
            original_answer = answers[best_agent_id]

            # Create anonymized ID
            anonymized_id = None
            for anon_id, answer in zip(
                [f"Answer {i+1}" for i in range(len(answers))], answers.values()
            ):
                if answer.agent_id == best_agent_id:
                    anonymized_id = anon_id
                    break

            if not anonymized_id:
                raise ValueError(f"Could not find anonymized ID for agent {best_agent_id}")

            # Collect feedback for this answer
            feedback = []
            for evaluation in evaluations:
                for score in evaluation.scores:
                    if score.answer_id == anonymized_id:
                        feedback.append(
                            f"Evaluator: {evaluation.evaluator_id}\n"
                            f"Score: {score.total_score:.2f}\n"
                            f"Reasoning: {score.reasoning}"
                        )

            if not feedback:
                raise ValueError(f"No feedback found for answer from agent {best_agent_id}")

            feedback_text = "\n\n".join(feedback)

            # Create improvement tasks
            improvement_tasks = {}

            # Create tasks for each improver
            for improver_id, agent in improver_agents.items():
                logger.debug(f"Creating improvement task for improver: {agent.role}")
                task = create_improvement_task(
                    question,
                    str(original_answer),
                    feedback_text,
                    agent,
                    async_execution=False,
                )
                improvement_tasks[improver_id] = task

            # Execute tasks in parallel
            console.print(f"Improving answer from [bold green]{best_agent_id}[/bold green]...")
            results = ThreadedTaskExecutor.execute_tasks_by_agent(
                improver_agents, improvement_tasks
            )

            # Process results and find the best improvement
            best_improvement = None
            best_improver_id = None
            best_improvement_score = -1

            for improver_id, result in results.items():
                try:
                    # Try different JSON extraction methods
                    parsed_result = None
                    raw_result = str(result)

                    # First try direct JSON parsing
                    try:
                        parsed_result = json.loads(raw_result)
                    except json.JSONDecodeError:
                        # Try to find JSON in markdown blocks
                        json_matches = re.finditer(
                            r"```(?:json)?\s*(.*?)\s*```", raw_result, re.DOTALL
                        )
                        for match in json_matches:
                            try:
                                parsed_result = json.loads(match.group(1))
                                if parsed_result:
                                    break
                            except json.JSONDecodeError:
                                continue

                        # If no valid JSON found in markdown blocks, try to extract a JSON-like structure
                        if not parsed_result:
                            # Look for content between curly braces
                            json_like = re.search(r"\{.*\}", raw_result, re.DOTALL)
                            if json_like:
                                try:
                                    parsed_result = json.loads(json_like.group(0))
                                except json.JSONDecodeError:
                                    pass

                    # If we still don't have valid JSON, create a basic structure from the raw text
                    if not parsed_result:
                        parsed_result = {
                            "improved_answer": raw_result,
                            "improvements": "Extracted from unstructured response",
                        }

                    # Calculate improvement score
                    improvement_score = len(parsed_result.get("improved_answer", "")) / max(
                        len(str(original_answer)), 1
                    )

                    # Check if this is the best improvement so far
                    if improvement_score > best_improvement_score:
                        best_improvement = parsed_result
                        best_improver_id = improver_id
                        best_improvement_score = improvement_score

                    log_agent_action(improver_agents[improver_id].role, "provided an improvement")
                    console.print(
                        f"[bold green]{improver_agents[improver_id].role}[/bold green] has provided an improvement."
                    )

                    # Log the improvement
                    await run_logger.log_improvement(
                        improver_agents[improver_id], best_agent_id, parsed_result
                    )

                except Exception as e:
                    logger.warning(f"Failed to process improvement from {improver_id}: {str(e)}")
                    continue

            if not best_improvement:
                raise ValueError(f"No valid improvements generated for answer from {best_agent_id}")

            # Add the best improvement to the results
            improved_answers[best_agent_id] = ImprovedAnswer(
                content=best_improvement.get("improved_answer"),
                original_agent_id=best_agent_id,
                improver_agent_id=best_improver_id,
                improvements=best_improvement.get("improvements", "No improvements specified"),
            )

        log_step_complete("Improving best answers")
        return improved_answers

    except Exception as e:
        error_msg = f"Failed to improve answers: {str(e)}"
        logger.error(error_msg)
        # Instead of calling log_error, log it as a general event
        await run_logger.log_event("error", "improve_answers", {"error": error_msg})
        raise


async def final_judgment(
    question: str,
    improved_answers: Dict[str, ImprovedAnswer],
    run_logger: RunLogger,
    final_judge: str,
) -> Tuple[str, float]:
    """
    Make a final judgment on the improved answers.

    Args:
        question: The original question
        improved_answers: Dictionary mapping keys to ImprovedAnswer objects
        run_logger: RunLogger instance
        final_judge: The model ID of the final judge to aggregate answers

    Returns:
        The final response from the judge and a confidence score
    """
    log_step_start("Making final judgment")
    logger.info(
        f"Making final judgment on {len(improved_answers)} improved answers using {final_judge} as judge"
    )

    try:
        console.print("\n[bold]Step 5: Making final judgment...[/bold]")
        console.print(
            f"[bold cyan]Using {AVAILABLE_MODELS[final_judge]} as the final judge[/bold cyan]"
        )

        # Validate inputs
        if not improved_answers:
            raise ValueError("No improved answers available for final judgment")

        # Anonymize improved answers
        anonymized_improved = {
            f"Improved Answer {i+1}": str(answer)
            for i, answer in enumerate(improved_answers.values())
        }
        logger.debug("Improved answers anonymized successfully")

        # Display anonymized improved answers
        for anon_id, content in anonymized_improved.items():
            console.print(
                Panel(
                    content[:500] + "..." if len(content) > 500 else content,
                    title=f"[bold cyan]{anon_id}[/bold cyan]",
                )
            )

        # Get evaluator agent for final judgment (single judge)
        evaluator_agents = get_evaluator_agents()
        evaluator_agents = filter_agents(evaluator_agents, {final_judge})
        if not evaluator_agents:
            raise ValueError(f"Final judge model {final_judge} not found in evaluator agents")

        logger.info(f"Using {final_judge} as the final judge model")

        # Get the single judge agent
        judge_agent = next(iter(evaluator_agents.values()))

        # Create judgment task for the single judge
        task = create_final_judgment_task(
            question,
            anonymized_improved,
            judge_agent,
            async_execution=False,
        )

        # Execute the task
        logger.info(f"Executing final judgment task with {judge_agent.role}...")
        result = ThreadedTaskExecutor.execute_task(judge_agent, task)

        # Get the raw response
        final_response = str(result)

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
        await run_logger.log_final_judgment(judge_agent, final_response)

        console.print("\n[bold]Final Judgment:[/bold]")
        console.print(
            Panel(
                final_response,
                title=f"[bold green]Final Judgment (Confidence: {confidence_score:.2f})[/bold green]",
                border_style="green",
            )
        )

        log_step_complete("Making final judgment")
        return final_response, confidence_score

    except Exception as e:
        error_msg = f"Failed to make final judgment: {str(e)}"
        logger.error(error_msg)
        await run_logger.log_event("error", "final_judgment", {"error": error_msg})
        raise


async def process_question(
    question: str, selected_models: Set[str], final_judge: str
) -> Tuple[str, float]:
    """
    Process a question through the multi-agent workflow.

    Args:
        question: The question to process
        selected_models: Set of model IDs to use
        final_judge: The model ID of the final judge to aggregate answers

    Returns:
        The final best answer and its final score

    Raises:
        TaskExecutionError: If any stage of the process fails
        ValueError: If invalid inputs are provided
    """
    logger.info(f"Starting to process question: {question}")
    run_logger = None

    try:
        # Step 1: Get initial answers
        answers, run_logger = await get_initial_answers(question, selected_models)

        # Step 2: Evaluate answers
        evaluations = await evaluate_answers(question, answers, run_logger, selected_models)

        # Step 3: Select best answers
        best_agent_ids = select_best_answers(evaluations, answers)

        # Step 4: Improve answers
        improved_answers = await improve_answers(
            question, best_agent_ids, answers, evaluations, run_logger, selected_models
        )

        # Step 5: Final judgment
        final_response, confidence_score = await final_judgment(
            question, improved_answers, run_logger, final_judge
        )

        # Log final summary
        await run_logger.log_summary(final_response)

        logger.info("Question processing completed successfully")
        return final_response, confidence_score

    except Exception as e:
        error_msg = f"Failed to process question: {str(e)}"
        logger.error(error_msg)
        if run_logger:
            # Instead of calling log_error, log it as a general event
            await run_logger.log_event("error", "process_question", {"error": error_msg})
        raise


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Multi-Agent AI Collaboration Tool")
    parser.add_argument("question", nargs="?", default=None, help="The question to process")
    parser.add_argument(
        "--all-models", action="store_true", help="Use all available models without prompting"
    )
    args = parser.parse_args()

    try:
        if args.question:
            question = args.question
        else:
            # Interactive mode
            console.print(
                Panel(
                    "Welcome to the Multi-Agent AI Collaboration Tool!\n"
                    "This tool uses multiple AI models to collaboratively answer your questions.",
                    title="[bold cyan]Multi-Agent AI Collaboration[/bold cyan]",
                )
            )
            question = console.input("[bold yellow]Please enter your question:[/bold yellow] ")

        # Get model selection
        selected_models = get_model_selection(use_all_models=args.all_models)

        # Get final judge selection
        console.print("\n[bold]Selecting a single model as the final judge...[/bold]")
        final_judge = get_final_judge_selection()

        # Process the question
        loop = asyncio.get_event_loop()
        final_response, confidence_score = loop.run_until_complete(
            process_question(question, selected_models, final_judge)
        )

        # Print the final answer with proper formatting
        console.print("\n[bold]Final Answer:[/bold]")
        console.print(
            Panel(
                final_response,
                title=f"[bold green]Final Answer (Confidence: {confidence_score:.2f})[/bold green]",
                border_style="green",
            )
        )

    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        console.print(
            Panel(
                f"[bold red]Error:[/bold red] {str(e)}\n\n"
                f"[bold yellow]Stack trace:[/bold yellow]\n{traceback.format_exc()}",
                title="[bold red]Error Details[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
