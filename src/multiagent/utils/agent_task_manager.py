#!/usr/bin/env python3
import json
import logging
import re
from typing import Dict, List, Any, Callable, TypeVar, Generic, Optional, Set, Tuple

from rich.console import Console
from rich.panel import Panel
from multiagent.utils.run_logger import RunLogger
from multiagent.utils.console_logger import ConsoleLogger
from multiagent.config.model_selection import filter_agents
from multiagent.utils.executor import ThreadedTaskExecutor

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar("T")  # Type for agent result
R = TypeVar("R")  # Type for processed result


class AgentTaskManager(Generic[T, R]):
    """
    A manager class that handles the common patterns for agent task execution and result processing.

    This class centralizes the logic for:
    - Getting and filtering agents
    - Creating and executing tasks
    - Processing results
    - Handling errors consistently
    - Logging agent actions and events
    """

    def __init__(
        self,
        step_name: str,
        run_logger: RunLogger,
        process_result_func: Callable[[str, Any, Any, RunLogger], T],
        log_action_msg: str = "provided a result",
    ):
        """
        Initialize the AgentTaskManager.

        Args:
            step_name: Name of the processing step
            run_logger: RunLogger instance for logging
            process_result_func: Function to process the result from each agent
            log_action_msg: Message to log when an agent provides a result
        """
        self.step_name = step_name
        self.run_logger = run_logger
        self.process_result_func = process_result_func
        self.log_action_msg = log_action_msg
        self.console_logger = ConsoleLogger()

    async def execute(
        self,
        agent_getter_func: Callable[[], Dict[str, Any]],
        task_creator_func: Callable[[str, Any, bool], Any],
        task_input: Any,
        selected_models: Set[str],
        async_execution: bool = False,
    ) -> Dict[str, T]:
        """
        Execute tasks for all selected agents in parallel.

        Args:
            agent_getter_func: Function to get available agents
            task_creator_func: Function to create tasks for agents
            task_input: Input data for the tasks
            selected_models: Set of model IDs to use
            async_execution: Whether to execute tasks asynchronously

        Returns:
            Dictionary mapping agent IDs to their processed results

        Raises:
            ValueError: If no agents are available after filtering
            TaskExecutionError: If task execution fails
        """
        try:
            # Get and filter agents
            agents = agent_getter_func()
            agents = filter_agents(agents, selected_models)

            if not agents:
                raise ValueError(
                    f"No agents available for {self.step_name} after filtering with selected models"
                )

            logger.info(f"Using {len(agents)} agents for {self.step_name}")
            self.console_logger.info(f"Using {len(agents)} models for {self.step_name}")

            # Create tasks
            tasks = {}
            for agent_id, agent in agents.items():
                task = task_creator_func(task_input, agent, async_execution)
                tasks[agent_id] = task
                logger.info(f"Created task for {agent.role}")

            # Execute tasks
            logger.info(f"Executing {self.step_name} tasks in parallel...")
            results = ThreadedTaskExecutor.execute_tasks_by_agent(agents, tasks)

            # Process results
            processed_results = {}
            for agent_id, result in results.items():
                try:
                    raw_result = str(result)
                    # Await the coroutine result from process_result_func
                    processed_result = await self.process_result_func(
                        agent_id, agents[agent_id], raw_result, self.run_logger
                    )
                    processed_results[agent_id] = processed_result

                    # Log agent action
                    logger.info(f"Agent {agents[agent_id].role} has {self.log_action_msg}")
                except Exception as e:
                    logger.warning(f"Failed to process result from {agent_id}: {str(e)}")
                    self.console_logger.warning(
                        f"Failed to process result from {agents[agent_id].role}"
                    )
                    continue

            return processed_results

        except Exception as e:
            error_msg = f"Failed to execute {self.step_name} tasks: {str(e)}"
            logger.error(error_msg)
            self.console_logger.error(error_msg)
            raise

    @staticmethod
    def extract_json_from_result(raw_result: str, default_structure: Optional[Dict] = None) -> Dict:
        """
        Extract structured JSON from raw result text.

        Args:
            raw_result: Raw text result from agent
            default_structure: Default structure to use if JSON extraction fails

        Returns:
            Extracted JSON as a dictionary
        """
        # Try direct JSON parsing
        try:
            return json.loads(raw_result)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown blocks
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_result, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Try to find JSON-like structure
            json_like = re.search(r"\{.*\}", raw_result, re.DOTALL)
            if json_like:
                try:
                    return json.loads(json_like.group(0))
                except json.JSONDecodeError:
                    pass

        # Return default structure or empty dict if all extraction methods fail
        return default_structure if default_structure is not None else {}
