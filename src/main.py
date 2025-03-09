#!/usr/bin/env python3
import os
import json
import asyncio
import argparse
import logging
from typing import Dict, List, Any, Set, Tuple
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from rich.logging import RichHandler
from rich.theme import Theme
from rich.style import Style
from crewai import Crew, Agent

from src.multiagent.agents.base import (
    get_answering_agents,
    get_evaluator_agents,
    get_improver_agents,
)
from src.multiagent.tasks.base import (
    create_answer_task,
    create_evaluation_task,
    create_improvement_task,
    create_final_judgment_task,
)
from src.multiagent.models.base import Answer, Evaluation, ImprovedAnswer
from src.multiagent.utils.base import (
    anonymize_answers,
    deanonymize_answers,
    parse_evaluation_result,
    parse_improvement_result,
    combine_evaluations,
)
from src.multiagent.config.model_selection import get_model_selection, filter_agents
from src.multiagent.utils.run_logger import RunLogger

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
    question: str, selected_models: Set[str] = None
) -> Tuple[Dict[str, Answer], RunLogger]:
    """
    Get initial answers from selected agents using a single crew with delegated tasks.

    Args:
        question: The question to answer
        selected_models: Set of model IDs to use (if None, use all models)

    Returns:
        Dictionary mapping agent IDs to their Answer objects and RunLogger
    """
    log_step_start("Getting initial answers from selected agents")
    log_progress(f"Processing question: {question}")

    # Initialize the run logger asynchronously
    run_logger = await RunLogger.create(question)

    console.print(Panel(f"[bold cyan]Question:[/bold cyan] {question}", title="Initial Question"))

    try:
        # Get all answering agents
        agents = get_answering_agents()

        # Filter agents based on selected models
        if selected_models:
            agents = filter_agents(agents, selected_models)

        log_progress(f"Using {len(agents)} answering agents")

        # Create answer tasks
        answers = {}
        agents_list = list(agents.values())

        # Create tasks for all agents except the last one
        tasks = []
        for agent in agents_list[:-1]:
            task = create_answer_task(question, agent, async_execution=False)
            tasks.append(task)

        # Make the last task async to handle coordination
        if agents_list:
            last_task = create_answer_task(question, agents_list[-1], async_execution=True)
            tasks.append(last_task)

        # Create a single crew with all agents and tasks
        crew = Crew(
            agents=agents_list,
            tasks=tasks,
            verbose=1,
        )

        # Execute all tasks
        log_progress("Executing answer tasks...")
        results = await crew.kickoff()

        # Handle results - ensure we get a list even if there's only one result
        if not isinstance(results, (list, tuple)):
            results = [results]

        log_progress(f"Received {len(results)} answers")

        # Process results
        for i, (agent_id, agent) in enumerate(agents.items()):
            try:
                if i < len(results):
                    answer_content = str(results[i])
                    answers[agent_id] = Answer(content=answer_content, agent_id=agent_id)
                    log_agent_action(agent.role, "provided an answer")
                    console.print(f"[bold green]{agent.role}[/bold green] has provided an answer.")

                    # Log the answer
                    run_logger.log_initial_answer(agent, answer_content)
                else:
                    logger.warning(f"No result received for agent {agent_id}")
                    answers[agent_id] = Answer(content="No response received", agent_id=agent_id)
            except Exception as e:
                logger.error(
                    format_log_message(f"Error processing answer from {agent.role}: {e}", "error")
                )
                answers[agent_id] = Answer(
                    content=f"Error processing answer: {str(e)}", agent_id=agent_id
                )

        log_step_complete("Getting initial answers")
        return answers, run_logger

    except Exception as e:
        error_msg = f"Error getting initial answers: {str(e)}"
        logger.error(format_log_message(error_msg, "error"))
        raise RuntimeError(error_msg) from e


async def evaluate_answers(
    question: str, answers: Dict[str, Answer], run_logger: RunLogger
) -> List[Evaluation]:
    """
    Have agents evaluate all answers using a single crew with delegated tasks.

    Args:
        question: The original question
        answers: Dictionary mapping agent IDs to their Answer objects
        run_logger: RunLogger instance

    Returns:
        List of Evaluation objects from each evaluator
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

        # Get all evaluator agents
        evaluator_agents = get_evaluator_agents()
        log_progress(f"Initialized {len(evaluator_agents)} evaluator agents")

        # Create tasks for all evaluators
        evaluator_list = list(evaluator_agents.values())
        tasks = []

        # Create tasks for all evaluators except the last one
        for agent in evaluator_list[:-1]:
            task = create_evaluation_task(
                question,
                anonymized_answers,
                agent,
                async_execution=False,
            )
            tasks.append(task)

        # Make the last task async to handle coordination
        if evaluator_list:
            last_task = create_evaluation_task(
                question,
                anonymized_answers,
                evaluator_list[-1],
                async_execution=True,
            )
            tasks.append(last_task)

        # Create a single crew with all evaluators and tasks
        crew = Crew(
            agents=evaluator_list,
            tasks=tasks,
            verbose=1,
        )

        # Execute all tasks
        log_progress("Executing evaluation tasks...")
        results = await crew.kickoff()

        # Handle results - ensure we get a list even if there's only one result
        if not isinstance(results, (list, tuple)):
            results = [results]

        log_progress(f"Received {len(results)} evaluations")

        # Process results
        evaluations = []
        for i, (evaluator_id, agent) in enumerate(evaluator_agents.items()):
            try:
                if i < len(results):
                    evaluation_text = str(results[i])
                    evaluation = parse_evaluation_result(evaluation_text, question, evaluator_id)
                    evaluations.append(evaluation)
                    log_agent_action(agent.role, "evaluated the answers")
                    console.print(
                        f"[bold green]{agent.role}[/bold green] has evaluated the answers."
                    )

                    # Log the evaluation
                    evaluation_data = json.loads(evaluation_text)
                    run_logger.log_evaluation(agent, evaluation_data)
                else:
                    logger.warning(f"No evaluation result received from {evaluator_id}")
            except json.JSONDecodeError as e:
                logger.error(
                    format_log_message(f"Error parsing evaluation from {agent.role}: {e}", "error")
                )
                logger.debug(
                    format_log_message(
                        f"Raw evaluation text: {evaluation_text if 'evaluation_text' in locals() else 'N/A'}",
                        "debug",
                    )
                )
                # Skip creating an evaluation object for invalid JSON
                continue
            except Exception as e:
                logger.error(
                    format_log_message(
                        f"Error processing evaluation from {agent.role}: {e}", "error"
                    )
                )
                logger.debug(
                    format_log_message(
                        f"Raw evaluation text: {evaluation_text if 'evaluation_text' in locals() else 'N/A'}",
                        "debug",
                    )
                )
                # Skip creating an evaluation object for other errors
                continue

        log_step_complete("Evaluating answers")
        return evaluations

    except Exception as e:
        error_msg = f"Error evaluating answers: {str(e)}"
        logger.error(format_log_message(error_msg, "error"))
        raise RuntimeError(error_msg) from e


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

    # Combine evaluations from all evaluators
    average_scores = combine_evaluations(evaluations)
    logger.debug(f"Combined scores: {average_scores}")

    # Sort answers by score
    sorted_answers = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"Sorted answers by score: {sorted_answers}")

    # Display scores
    table = Table(title="Answer Scores")
    table.add_column("Answer ID", style="cyan")
    table.add_column("Average Score", style="green")

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

    # Convert anonymized IDs back to agent IDs
    best_agent_ids = deanonymize_answers(best_answer_ids, answers)
    logger.info(f"Best agent IDs: {best_agent_ids}")

    console.print(f"[bold]Selected best answers:[/bold] {', '.join(best_answer_ids)}")

    log_step_complete("Selecting best answers")
    return best_agent_ids


async def improve_answers(
    question: str,
    best_agent_ids: List[str],
    answers: Dict[str, Answer],
    evaluations: List[Evaluation],
    run_logger: RunLogger,
) -> Dict[str, ImprovedAnswer]:
    """
    Improve the best answers based on evaluation feedback.

    Args:
        question: The original question
        best_agent_ids: List of agent IDs for the best answers
        answers: Dictionary mapping agent IDs to their Answer objects
        evaluations: List of Evaluation objects
        run_logger: RunLogger instance

    Returns:
        Dictionary mapping agent IDs to their ImprovedAnswer objects
    """
    log_step_start("Improving best answers")
    logger.info(f"Improving answers from {len(best_agent_ids)} agents")

    console.print("\n[bold]Step 4: Improving best answers...[/bold]")

    # Get improver agents
    improver_agents = get_improver_agents()
    logger.info(f"Initialized {len(improver_agents)} improver agents")

    # Create improvement tasks
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
            logger.error(f"Could not find anonymized ID for agent {best_agent_id}")
            continue

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
        logger.debug(f"Collected {len(feedback)} feedback items for {best_agent_id}")

        feedback_text = "\n\n".join(feedback)

        # Create a task for each improver agent
        improver_list = list(improver_agents.values())
        tasks = []
        for i, agent in enumerate(improver_list):
            logger.debug(f"Creating improvement task for improver: {agent.role}")
            tasks.append(
                create_improvement_task(
                    question,
                    str(original_answer),
                    feedback_text,
                    agent,
                    async_execution=(i == len(improver_list) - 1),
                )
            )

        # Create a Crew for improvement
        crew = Crew(
            agents=improver_list,
            tasks=tasks,
            verbose=1,
        )
        logger.info(f"Created improvement crew for {best_agent_id}")

        # Execute tasks
        console.print(f"Improving answer from [bold green]{best_agent_id}[/bold green]...")
        logger.info(f"Executing improvement tasks for {best_agent_id}...")
        results = await crew.kickoff()
        logger.info("Received improvement results")

        # Process results - handle both single result and list of results
        if not isinstance(results, (list, tuple)):
            results = [results]

        # Process results
        for i, (improver_id, agent) in enumerate(improver_agents.items()):
            try:
                if i < len(results):
                    improvement_text = str(results[i])
                    improved_answer = parse_improvement_result(
                        improvement_text, best_agent_id, improver_id
                    )
                    key = f"{best_agent_id}_{improver_id}"
                    improved_answers[key] = improved_answer
                    logger.info(f"Processed improvement from {agent.role}")
                    console.print(
                        f"[bold green]{agent.role}[/bold green] has improved the answer from [bold blue]{best_agent_id}[/bold blue]."
                    )

                    # Log the improvement
                    improved_answer_data = json.loads(improvement_text)
                    run_logger.log_improvement(agent, best_agent_id, improved_answer_data)
                else:
                    logger.warning(f"No improvement result received from {improver_id}")
            except Exception as e:
                logger.error(f"Error parsing improvement from {agent.role}: {e}")
                logger.debug(
                    f"Raw improvement text: {str(results[i]) if i < len(results) else 'N/A'}"
                )

    log_step_complete("Improving answers")
    return improved_answers


async def final_judgment(
    question: str, improved_answers: Dict[str, ImprovedAnswer], run_logger: RunLogger
) -> Tuple[str, float]:
    """
    Make a final judgment on the improved answers.

    Args:
        question: The original question
        improved_answers: Dictionary mapping keys to ImprovedAnswer objects
        run_logger: RunLogger instance

    Returns:
        The content of the best answer and its final score
    """
    log_step_start("Making final judgment")
    logger.info(f"Making final judgment on {len(improved_answers)} improved answers")

    console.print("\n[bold]Step 5: Making final judgment...[/bold]")

    # Anonymize improved answers
    anonymized_improved = {
        f"Improved Answer {i+1}": str(answer) for i, answer in enumerate(improved_answers.values())
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

    # Get a judge agent (using claude for final judgment)
    evaluator_agents = get_evaluator_agents()
    judge_agent = evaluator_agents.get("claude_37", list(evaluator_agents.values())[0])

    # Create judgment task
    task = create_final_judgment_task(
        question,
        anonymized_improved,
        judge_agent,
        async_execution=True,  # Only one task, so it can be async
    )

    # Create a Crew for judgment
    crew = Crew(
        agents=[judge_agent],
        tasks=[task],
        verbose=1,
    )

    # Execute task
    logger.info("Executing final judgment task...")
    results = await crew.kickoff()

    # Parse result
    judgment_text = str(results[0]) if results else "{}"
    logger.debug(f"Raw judgment text: {judgment_text}")

    try:
        # Try parsing as JSON
        if "```json" in judgment_text:
            json_start = judgment_text.find("```json") + 7
            json_end = judgment_text.find("```", json_start)
            data = json.loads(judgment_text[json_start:json_end].strip())
        else:
            data = json.loads(judgment_text)

        best_answer_id = data.get("best_answer_id", "")
        reasoning = data.get("reasoning", "")
        final_score = data.get("final_score", 0)

        # Map anonymized ID back to actual answer
        answer_index = int(best_answer_id.split()[-1]) - 1
        best_answer_key = list(improved_answers.keys())[answer_index]
        best_answer = improved_answers[best_answer_key]

        # Display results
        console.print(
            Panel(
                f"[bold cyan]Best Answer:[/bold cyan] {best_answer_id}\n"
                f"[bold cyan]Score:[/bold cyan] {final_score}\n"
                f"[bold cyan]Reasoning:[/bold cyan] {reasoning}",
                title="Final Judgment",
            )
        )

        console.print(Panel(str(best_answer), title="[bold green]Selected Answer[/bold green]"))

        # Log final judgment
        judgment_data = json.loads(judgment_text)
        run_logger.log_final_judgment(judge_agent, judgment_data)

        log_step_complete("Making final judgment")
        return str(best_answer), final_score

    except Exception as e:
        logger.error(f"Error parsing judgment result: {e}")
        logger.debug(f"Raw judgment text: {judgment_text}")

        # Fallback
        console.print(
            "[bold red]Error in final judgment. Returning first improved answer.[/bold red]"
        )
        return str(list(improved_answers.values())[0]), 0


async def process_question(question: str, selected_models: Set[str] = None) -> Tuple[str, float]:
    """
    Process a question through the multi-agent workflow.

    Args:
        question: The question to process
        selected_models: Set of model IDs to use (if None, use all models)

    Returns:
        The final best answer and its final score
    """
    logger.info(f"Starting to process question: {question}")
    try:
        # Step 1: Get initial answers
        answers, run_logger = await get_initial_answers(question, selected_models)

        # Step 2: Evaluate answers
        evaluations = await evaluate_answers(question, answers, run_logger)

        # Step 3: Select best answers
        best_agent_ids = select_best_answers(evaluations, answers)

        # Step 4: Improve answers
        improved_answers = await improve_answers(
            question, best_agent_ids, answers, evaluations, run_logger
        )

        # Step 5: Final judgment
        best_answer, final_score = await final_judgment(question, improved_answers, run_logger)

        # Log final summary
        run_logger.log_summary(best_answer)

        logger.info("Question processing completed successfully")
        return best_answer, final_score

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Multi-Agent AI Collaboration Tool")
    parser.add_argument("question", nargs="?", default=None, help="The question to process")
    parser.add_argument(
        "--all-models", action="store_true", help="Use all available models without prompting"
    )
    args = parser.parse_args()

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

    # Process the question
    loop = asyncio.get_event_loop()
    final_answer, final_score = loop.run_until_complete(process_question(question, selected_models))

    # Save result to file
    with open("last_answer.txt", "w") as f:
        f.write(f"{final_answer}\n{final_score}")

    console.print("\n[bold green]Final answer saved to last_answer.txt[/bold green]")


if __name__ == "__main__":
    main()
