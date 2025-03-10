#!/usr/bin/env python3
import os
import json
import asyncio
import argparse
import logging
import sys
import re
import traceback
import builtins
from typing import Dict, List, Any, Set, Tuple
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# Apply aggressive output suppression before importing any modules
# ===============================================================

# 1. Patch print function to silence unwanted output
original_print = builtins.print


def silent_print(*args, **kwargs):
    # Silence detailed outputs but allow concise status messages
    # Check if this is a large output we want to suppress
    if args:
        # Skip large outputs likely to be task content or answers
        large_content = [arg for arg in args if isinstance(arg, str) and len(arg) > 300]
        if large_content:
            return

        # Skip specific patterns related to detailed task execution
        patterns_to_skip = ["FINAL_ANSWER:", "```json", "```", '{"concept":', "Task Execution"]

        if any(
            isinstance(arg, str) and any(pattern in arg for pattern in patterns_to_skip)
            for arg in args
        ):
            return

    # Check caller - only filter specific modules
    caller_frames = traceback.extract_stack()
    if len(caller_frames) >= 2:
        caller = caller_frames[-2]
        caller_file = caller.filename

        # Always filter verbose LLM/API-related modules
        if any(
            module in caller_file.lower() for module in ["litellm", "openai", "anthropic", "http"]
        ):
            return

    # Allow brief status message prints - they're informative
    return original_print(*args, **kwargs)


builtins.print = silent_print

# 2. Patch rich.console.Console to prevent agent output
try:
    # This needs to happen BEFORE any other imports that might use rich
    from rich.console import Console as RichConsole

    # Store the original Console class
    OriginalConsole = RichConsole

    # Create a patched version that filters agent-related output
    class FilteredConsole(OriginalConsole):
        def print(self, *args, **kwargs):
            # Get caller stack frame
            caller_frames = traceback.extract_stack()
            if len(caller_frames) >= 2:
                caller = caller_frames[-2]
                caller_file = caller.filename

                # Skip printing only from certain CrewAI modules that show full task content
                if any(
                    module in caller_file.lower()
                    for module in ["llm", "task_runner", "task_output"]
                ):
                    return

                # Only skip specific agent content - task details, but keep status updates
                if args:
                    # Check for detailed agent content we want to skip
                    content_to_skip = ["Final Answer:"] + [
                        content
                        for content in args
                        if isinstance(content, str) and len(content) > 500
                    ]

                    # Check for answer blocks in any format (JSON, markdown, etc.)
                    code_block_patterns = [
                        "```json",
                        "```python",
                        "```",
                        '{"concept":',
                        "FINAL_ANSWER:",
                    ]

                    if any(
                        isinstance(arg, str)
                        and any(pattern in arg for pattern in code_block_patterns)
                        for arg in args
                    ):
                        return

                    # Skip if there's any long content
                    if content_to_skip:
                        return

            # Call original print for our own messages and allowed agent status
            return super().print(*args, **kwargs)

    # Replace the original Console with our filtered version
    import rich.console

    rich.console.Console = FilteredConsole

except ImportError:
    # Rich library not installed, no need to patch
    pass

# Now import the rest of the modules
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
from multiagent.utils.console_logger import ConsoleLogger
from multiagent.utils.agent_task_manager import AgentTaskManager
from multiagent.utils.result_processors import (
    process_answer_result,
    process_evaluation_result,
    process_improvement_result,
    process_judgment_result,
)

# Load environment variables
load_dotenv()

# Configure logging for third-party libraries to be completely silent
THIRD_PARTY_LOGGERS = [
    "httpx",
    "litellm",
    "urllib3",
    "openai",
    "anthropic",
    "crewai",
    "asyncio",
    "aiohttp",
    "requests",
    "urllib3.connectionpool",
    "openrouter",
    "LiteLLM",  # Also catch uppercase variants
    "openai._base_client",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "crewai.agent",  # CrewAI agent execution logs
    "crewai.task",  # CrewAI task execution logs
    "crewai.crew",  # CrewAI crew logs
    "crewai.llm",  # CrewAI LLM logs
    "crewai.utilities",  # CrewAI utility logs
]

# Additional pattern-based logging silencers
LOGGER_PATTERNS = [
    "crewai",
    "litellm",
    "llm",
    "openai",
    "anthropic",
    "http",
    "urllib",
    "requests",
]

# Completely silence all third-party loggers
for logger_name in THIRD_PARTY_LOGGERS:
    log = logging.getLogger(logger_name)
    log.setLevel(logging.CRITICAL)  # Only show critical errors
    log.propagate = False  # Don't propagate to root logger

    # Remove any existing handlers
    for handler in log.handlers[:]:
        log.removeHandler(handler)

    # Add a null handler to prevent propagation
    log.addHandler(logging.NullHandler())

# Also silence any logger that contains any of the patterns (case insensitive)
for name in logging.root.manager.loggerDict:
    if any(pattern.lower() in name.lower() for pattern in LOGGER_PATTERNS):
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        logger.addHandler(logging.NullHandler())

# Initialize console logger
console = ConsoleLogger()


def setup_run_logging(run_dir: Path) -> logging.FileHandler:
    """
    Set up logging configuration for a specific run.

    Args:
        run_dir: Path to the run directory

    Returns:
        The configured file handler
    """
    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create and configure the file handler for detailed logging
    log_path = run_dir / "run.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Capture all logs in the file

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Allow all logs to be potentially captured

    # Remove any existing handlers from the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add our file handler for comprehensive logging to file
    root_logger.addHandler(file_handler)

    # Set up a filtered handler for console output
    class FilteredStreamHandler(logging.StreamHandler):
        def emit(self, record):
            # Messages to show in console:
            # 1. multiagent.utils.agent_task_manager INFO messages
            # 2. __main__ INFO messages that are informative status updates
            # 3. Error messages from any source

            # Always show errors from any source
            if record.levelno >= logging.ERROR:
                super().emit(record)
                return

            # Show agent task manager informational logs
            if (
                record.name == "multiagent.utils.agent_task_manager"
                and record.levelno == logging.INFO
            ):
                # These are useful status updates
                super().emit(record)
                return

            # For __main__ logs, be selective
            if record.name == "__main__" and record.levelno >= logging.INFO:
                # Skip message patterns containing detailed content
                skip_patterns = [
                    # Skip full answers and other detailed content
                    "answer:\n",
                    "answers for evaluation",
                    "improved answer",
                    "Original answer",
                    "{",
                    "}",  # Likely part of JSON content
                ]

                # Skip logs with large messages
                if len(record.getMessage()) > 100:  # Message too long, probably contains an answer
                    return

                # Skip messages containing any of the skip patterns
                if any(pattern.lower() in record.getMessage().lower() for pattern in skip_patterns):
                    return

                # This is a brief status update, show it
                super().emit(record)
                return

    # Add our filtered stream handler
    stream_handler = FilteredStreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)

    return file_handler


# Get our logger
logger = logging.getLogger(__name__)


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
    console.start_step("Getting initial answers")
    console.display_question(question)

    # Initialize the run logger asynchronously
    run_logger = await RunLogger.create(question)

    # Set up logging for this run
    file_handler = setup_run_logging(run_logger.run_dir)

    try:
        # Log start of new question processing
        logger.info("=" * 80)
        logger.info(f"Starting to process new question: {question}")
        logger.info("=" * 80)

        # Create task manager for answers
        task_manager = AgentTaskManager(
            step_name="Getting initial answers",
            run_logger=run_logger,
            process_result_func=lambda agent_id, agent, raw_result, run_logger: asyncio.create_task(
                process_answer_result(agent_id, agent, raw_result, run_logger)
            ),
            log_action_msg="provided an answer",
        )

        # Execute tasks and get results
        answers = await task_manager.execute(
            agent_getter_func=get_answering_agents,
            task_creator_func=lambda question, agent, async_execution: create_answer_task(
                question, agent, async_execution
            ),
            task_input=question,
            selected_models=selected_models,
            async_execution=False,
        )

        console.complete_step("Getting initial answers")
        return answers, run_logger

    except Exception as e:
        error_msg = f"Failed to get initial answers: {str(e)}"
        logger.error(error_msg)
        console.error(error_msg)
        await run_logger.log_event("error", "initial_answers", {"error": error_msg})
        raise
    finally:
        # Clean up logging
        root_logger = logging.getLogger()
        root_logger.removeHandler(file_handler)
        file_handler.close()


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
    console.start_step("Evaluating answers")

    # Set up logging for this run
    file_handler = setup_run_logging(run_logger.run_dir)

    try:
        # Log anonymized answers to file
        anonymized_answers = anonymize_answers(answers)
        logger.info("Anonymized answers for evaluation:")
        for anon_id, content in anonymized_answers.items():
            logger.info(f"{anon_id}:\n{content}\n")

        # Create task manager for evaluations
        task_manager = AgentTaskManager(
            step_name="Evaluating answers",
            run_logger=run_logger,
            process_result_func=lambda agent_id, agent, raw_result, run_logger: asyncio.create_task(
                process_evaluation_result(agent_id, agent, raw_result, run_logger)
            ),
            log_action_msg="provided an evaluation",
        )

        # Execute tasks and get results
        evaluation_results = await task_manager.execute(
            agent_getter_func=get_evaluator_agents,
            task_creator_func=lambda task_input, agent, async_execution: create_evaluation_task(
                question, anonymize_answers(answers), agent, async_execution
            ),
            task_input=question,
            selected_models=selected_models,
            async_execution=False,
        )

        # Convert results to a list of evaluations
        evaluations = list(evaluation_results.values())

        if not evaluations:
            raise ValueError("No valid evaluations were produced")

        console.complete_step("Evaluating answers")
        return evaluations

    except Exception as e:
        error_msg = f"Failed to evaluate answers: {str(e)}"
        logger.error(error_msg)
        console.error(error_msg)
        await run_logger.log_event("error", "evaluate_answers", {"error": error_msg})
        raise
    finally:
        # Clean up logging
        root_logger = logging.getLogger()
        root_logger.removeHandler(file_handler)
        file_handler.close()


def select_best_answers(
    evaluations: List[Evaluation], answers: Dict[str, Answer], run_logger: RunLogger
) -> List[str]:
    """
    Select the best answers based on evaluations.

    Args:
        evaluations: List of Evaluation objects
        answers: Dictionary mapping agent IDs to their Answer objects
        run_logger: RunLogger instance for logging

    Returns:
        List of agent IDs for the best answers
    """
    console.start_step("Selecting best answers")

    # Set up logging for this run
    file_handler = setup_run_logging(run_logger.run_dir)

    try:
        # If we have evaluations, combine scores from all evaluators
        if evaluations:
            # Combine evaluations from all evaluators
            average_scores = combine_evaluations(evaluations)
            logger.debug(f"Combined scores: {average_scores}")

            # Sort answers by score for logging
            sorted_scores = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)
            logger.info("Sorted answer scores:")
            for answer_id, score in sorted_scores:
                logger.info(f"{answer_id}: {score:.2f}")

            # Get best answers
            best_answer_ids = evaluations[0].get_best_answers(top_n=2, threshold=0.1)
            logger.info(f"Selected best answers: {best_answer_ids}")
        else:
            # Fallback if no evaluations
            logger.warning("No evaluations available, using all available answers")
            best_answer_ids = list(answers.keys())

        # Convert anonymized IDs back to agent IDs if necessary
        if evaluations:
            best_agent_ids = deanonymize_answers(best_answer_ids, answers)
        else:
            best_agent_ids = best_answer_ids

        console.info(f"Selected {len(best_agent_ids)} best answers")
        console.complete_step("Selecting best answers")
        return best_agent_ids

    except Exception as e:
        error_msg = f"Failed to select best answers: {str(e)}"
        logger.error(error_msg)
        console.error(error_msg)
        raise
    finally:
        # Clean up logging
        root_logger = logging.getLogger()
        root_logger.removeHandler(file_handler)
        file_handler.close()


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
    console.start_step("Improving best answers")

    # Set up logging for this run
    file_handler = setup_run_logging(run_logger.run_dir)

    try:
        improved_answers = {}
        for best_agent_id in best_agent_ids:
            console.info(f"Improving answer from {answers[best_agent_id].agent_id}")
            logger.info(f"Processing improvements for answer from {best_agent_id}")
            logger.info(f"Original answer:\n{str(answers[best_agent_id])}\n")

            # Collect and log feedback for this answer
            feedback = []
            for evaluation in evaluations:
                for score in evaluation.scores:
                    if score.answer_id == f"Answer {list(answers.keys()).index(best_agent_id) + 1}":
                        feedback_text = (
                            f"Evaluator: {evaluation.evaluator_id}\n"
                            f"Score: {score.total_score:.2f}\n"
                            f"Reasoning: {score.reasoning}"
                        )
                        feedback.append(feedback_text)
                        logger.info(f"Evaluation feedback:\n{feedback_text}\n")

            # Create task manager for this specific answer improvement
            task_manager = AgentTaskManager(
                step_name=f"Improving answer from {best_agent_id}",
                run_logger=run_logger,
                process_result_func=lambda agent_id, agent, raw_result, run_logger: asyncio.create_task(
                    process_improvement_result(
                        agent_id,
                        agent,
                        raw_result,
                        run_logger,
                        best_agent_id=best_agent_id,
                        original_answer=str(answers[best_agent_id]),
                    )
                ),
                log_action_msg="provided an improvement",
            )

            # Execute tasks for this specific answer
            improvement_results = await task_manager.execute(
                agent_getter_func=get_improver_agents,
                task_creator_func=lambda task_input, agent, async_execution: create_improvement_task(
                    question,
                    str(answers[best_agent_id]),
                    "\n\n".join(feedback) if feedback else "No specific feedback available",
                    agent,
                    async_execution=async_execution,
                ),
                task_input=question,
                selected_models=selected_models,
                async_execution=False,
            )

            # Find the best improvement
            if improvement_results:
                best_improvement = next(iter(improvement_results.values()))
                improved_answers[best_agent_id] = best_improvement
                logger.info(f"Improved answer from {best_agent_id}:\n{str(best_improvement)}\n")

        console.complete_step("Improving best answers")
        return improved_answers

    except Exception as e:
        error_msg = f"Failed to improve answers: {str(e)}"
        logger.error(error_msg)
        console.error(error_msg)
        await run_logger.log_event("error", "improve_answers", {"error": error_msg})
        raise
    finally:
        # Clean up logging
        root_logger = logging.getLogger()
        root_logger.removeHandler(file_handler)
        file_handler.close()


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
    console.start_step("Making final judgment")
    console.info(f"Using {AVAILABLE_MODELS[final_judge]} as final judge")

    # Set up logging for this run
    file_handler = setup_run_logging(run_logger.run_dir)

    try:
        # Log improved answers
        logger.info("Improved answers for final judgment:")
        for i, (agent_id, answer) in enumerate(improved_answers.items(), 1):
            logger.info(f"Improved Answer {i} (from {agent_id}):\n{str(answer)}\n")

        # Create task manager for final judgment
        task_manager = AgentTaskManager(
            step_name="Making final judgment",
            run_logger=run_logger,
            process_result_func=lambda agent_id, agent, raw_result, run_logger: asyncio.create_task(
                process_judgment_result(agent_id, agent, raw_result, run_logger)
            ),
            log_action_msg="provided a final judgment",
        )

        # Execute judgment task
        judgment_results = await task_manager.execute(
            agent_getter_func=lambda: filter_agents(get_evaluator_agents(), {final_judge}),
            task_creator_func=lambda task_input, agent, async_execution: create_final_judgment_task(
                question,
                {
                    f"Improved Answer {i+1}": str(answer)
                    for i, answer in enumerate(improved_answers.values())
                },
                agent,
                async_execution,
            ),
            task_input=question,
            selected_models={final_judge},
            async_execution=False,
        )

        # Get the result
        if not judgment_results:
            raise ValueError(f"No judgment results received from {final_judge}")

        judgment_data = next(iter(judgment_results.values()))
        final_response = judgment_data["final_response"]
        confidence_score = judgment_data["confidence_score"]

        # Log final judgment details
        logger.info(f"Final judgment from {final_judge}:")
        logger.info(f"Response: {final_response}")
        logger.info(f"Confidence score: {confidence_score}")

        console.complete_step("Making final judgment")
        return final_response, confidence_score

    except Exception as e:
        error_msg = f"Failed to make final judgment: {str(e)}"
        logger.error(error_msg)
        console.error(error_msg)
        await run_logger.log_event("error", "final_judgment", {"error": error_msg})
        raise
    finally:
        # Clean up logging
        root_logger = logging.getLogger()
        root_logger.removeHandler(file_handler)
        file_handler.close()


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
        best_agent_ids = select_best_answers(evaluations, answers, run_logger)

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
        console.info(f"Final answer ready (confidence: {confidence_score:.2f})")

        return final_response, confidence_score

    except Exception as e:
        error_msg = f"Failed to process question: {str(e)}"
        logger.error(error_msg)
        console.error(error_msg)
        if run_logger:
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
            console.info("Welcome to the Multi-Agent AI Collaboration Tool!")
            question = input("Please enter your question: ")

        # Get model selection
        selected_models = get_model_selection(use_all_models=args.all_models)

        # Get final judge selection
        console.info("Selecting final judge model...")
        final_judge = get_final_judge_selection()

        # Process the question
        loop = asyncio.get_event_loop()
        final_response, confidence_score = loop.run_until_complete(
            process_question(question, selected_models, final_judge)
        )

        # Display final answer
        console.info("\nFinal Answer:")
        console.display_question(final_response)

    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        console.error(f"Critical error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
