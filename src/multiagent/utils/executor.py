import threading
import concurrent.futures
import traceback
import sys
import io
import contextlib
from typing import List, Dict, Any, TypeVar, Generic, Callable
from crewai import Agent, Task
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Define a generic type for task results
T = TypeVar("T")


class TaskExecutionError(Exception):
    """Custom exception for task execution failures that includes agent context."""

    def __init__(self, message: str, agent_role: str = None, original_error: Exception = None):
        self.agent_role = agent_role
        self.original_error = original_error
        super().__init__(
            f"Task execution failed for agent {agent_role}: {message}\n"
            + (f"Original error: {str(original_error)}\n" if original_error else "")
            + (f"Stack trace:\n{traceback.format_exc()}" if original_error else "")
        )


@contextlib.contextmanager
def redirect_stdout_stderr():
    """Context manager to redirect stdout and stderr to StringIO."""
    # Create StringIO objects to capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Redirect stdout/stderr to our capture objects
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        yield
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Get the captured output (for debugging if needed)
        stdout_value = stdout_capture.getvalue()
        stderr_value = stderr_capture.getvalue()

        # Log the captured output at debug level for troubleshooting
        if stdout_value.strip():
            logger.debug(f"Captured stdout: {stdout_value}")
        if stderr_value.strip():
            logger.debug(f"Captured stderr: {stderr_value}")


class ThreadedTaskExecutor:
    """
    Utility class for executing tasks in a multithreaded manner.
    Particularly useful for I/O-bound operations like LLM API calls.
    """

    @staticmethod
    def execute_task(agent: Agent, task: Task) -> Any:
        """
        Execute a single task with the given agent.

        Args:
            agent: The agent to execute the task
            task: The task to execute

        Returns:
            The result of the task execution

        Raises:
            TaskExecutionError: If task execution fails
        """
        try:
            # Use context manager to redirect stdout/stderr during task execution
            with redirect_stdout_stderr():
                result = agent.execute_task(task)

            if result is None:
                raise TaskExecutionError("Task returned None result", agent_role=agent.role)
            return result
        except Exception as e:
            # Log the error and raise a TaskExecutionError with context
            logger.error(f"Error executing task with agent {agent.role}: {str(e)}")
            raise TaskExecutionError(str(e), agent_role=agent.role, original_error=e)

    @staticmethod
    def execute_tasks_in_parallel(
        agent_task_pairs: List[tuple[Agent, Task]], max_workers: int = None
    ) -> List[Any]:
        """
        Execute multiple tasks in parallel using a thread pool.

        Args:
            agent_task_pairs: List of (agent, task) tuples to execute
            max_workers: Maximum number of worker threads (default: None, which uses ThreadPoolExecutor default)

        Returns:
            List of task execution results in the same order as the input agent_task_pairs

        Raises:
            TaskExecutionError: If any task execution fails
        """
        results = [None] * len(agent_task_pairs)  # Pre-allocate result list
        errors = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each agent-task pair
            future_to_idx = {
                executor.submit(ThreadedTaskExecutor.execute_task, agent, task): i
                for i, (agent, task) in enumerate(agent_task_pairs)
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Collect all errors that occur
                    errors.append(e)
                    agent_role = agent_task_pairs[idx][0].role
                    logger.error(f"Task execution failed for agent {agent_role}: {str(e)}")

        # If any errors occurred, raise an exception with all error details
        if errors:
            error_messages = "\n".join([str(e) for e in errors])
            raise TaskExecutionError(
                f"Multiple task executions failed:\n{error_messages}", agent_role="multiple"
            )

        return results

    @staticmethod
    def execute_tasks_by_agent(agents: Dict[str, Agent], tasks: Dict[str, Task]) -> Dict[str, Any]:
        """
        Execute tasks for corresponding agents and return results in a dictionary.

        Args:
            agents: Dictionary mapping agent_id to Agent
            tasks: Dictionary mapping agent_id to Task

        Returns:
            Dictionary mapping agent_id to task execution result

        Raises:
            ValueError: If agent IDs don't match between agents and tasks
            TaskExecutionError: If any task execution fails
        """
        # Validate that all task keys exist in agents
        missing_agents = set(tasks.keys()) - set(agents.keys())
        if missing_agents:
            raise ValueError(
                f"Agent IDs {missing_agents} from tasks not found in agents dictionary"
            )

        # Create agent-task pairs
        agent_task_pairs = [(agents[agent_id], task) for agent_id, task in tasks.items()]
        agent_ids = list(tasks.keys())

        # Execute tasks in parallel
        results = ThreadedTaskExecutor.execute_tasks_in_parallel(agent_task_pairs)

        # Map results back to agent IDs
        return {agent_id: result for agent_id, result in zip(agent_ids, results)}

    @staticmethod
    def execute_task_with_each_agent(
        task_creator: Callable[[Agent], Task], agents: Dict[str, Agent]
    ) -> Dict[str, Any]:
        """
        Create and execute a task for each agent in parallel.

        Args:
            task_creator: Function that creates a task for a given agent
            agents: Dictionary mapping agent_id to Agent

        Returns:
            Dictionary mapping agent_id to task execution result

        Raises:
            TaskExecutionError: If any task creation or execution fails
        """
        try:
            # Create a task for each agent
            tasks = {agent_id: task_creator(agent) for agent_id, agent in agents.items()}
        except Exception as e:
            raise TaskExecutionError("Failed to create tasks for agents", original_error=e)

        # Execute tasks in parallel
        return ThreadedTaskExecutor.execute_tasks_by_agent(agents, tasks)
