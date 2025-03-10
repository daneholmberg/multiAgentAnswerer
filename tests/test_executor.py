#!/usr/bin/env python3
import logging
import sys
import time
from typing import Dict

from crewai import Agent, Task

from multiagent.agents.base import get_answering_agents
from multiagent.tasks.base import create_answer_task
from multiagent.utils.executor import ThreadedTaskExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def test_simple_execution():
    """Test the execution of a single task."""
    logger.info("Testing simple task execution...")

    # Get a sample agent
    agents = get_answering_agents()
    agent = next(iter(agents.values()))

    # Create a task
    task = create_answer_task("What is the capital of France?", agent, async_execution=False)

    # Execute the task
    start_time = time.time()
    result = ThreadedTaskExecutor.execute_task(agent, task)
    end_time = time.time()

    logger.info(f"Task executed in {end_time - start_time:.2f} seconds")
    logger.info(f"Result: {result[:100]}...")

    return result


def test_parallel_execution():
    """Test the execution of multiple tasks in parallel."""
    logger.info("Testing parallel task execution...")

    # Get sample agents
    agents = get_answering_agents()
    logger.info(f"Using {len(agents)} agents")

    # Create tasks
    questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "What are the main differences between Python and JavaScript?",
        "How does photosynthesis work?",
        "What is the theory of relativity?",
    ]

    agent_task_pairs = []
    for i, (agent_id, agent) in enumerate(agents.items()):
        if i < len(questions):
            task = create_answer_task(questions[i], agent, async_execution=False)
            agent_task_pairs.append((agent, task))

    # Execute tasks in parallel
    start_time = time.time()
    results = ThreadedTaskExecutor.execute_tasks_in_parallel(agent_task_pairs)
    end_time = time.time()

    logger.info(f"All tasks executed in {end_time - start_time:.2f} seconds")

    for i, result in enumerate(results):
        if result:
            logger.info(f"Result {i+1}: {str(result)[:50]}...")
        else:
            logger.warning(f"Result {i+1} is None")

    return results


def test_tasks_by_agent():
    """Test the execution of tasks by agent."""
    logger.info("Testing tasks by agent execution...")

    # Get sample agents
    agents = get_answering_agents()
    logger.info(f"Using {len(agents)} agents")

    # Create tasks
    tasks = {}
    for agent_id, agent in agents.items():
        task = create_answer_task(
            f"Tell me about your role as {agent.role}", agent, async_execution=False
        )
        tasks[agent_id] = task

    # Execute tasks by agent
    start_time = time.time()
    results = ThreadedTaskExecutor.execute_tasks_by_agent(agents, tasks)
    end_time = time.time()

    logger.info(f"All tasks executed in {end_time - start_time:.2f} seconds")

    for agent_id, result in results.items():
        if result:
            logger.info(f"Agent {agent_id}: {str(result)[:50]}...")
        else:
            logger.warning(f"Agent {agent_id} returned None")

    return results


def test_task_with_each_agent():
    """Test the execution of a task with each agent."""
    logger.info("Testing task with each agent execution...")

    # Get sample agents
    agents = get_answering_agents()
    logger.info(f"Using {len(agents)} agents")

    # Define task creator
    def task_creator(agent: Agent) -> Task:
        return create_answer_task(
            f"What makes your approach as {agent.role} unique?", agent, async_execution=False
        )

    # Execute task with each agent
    start_time = time.time()
    results = ThreadedTaskExecutor.execute_task_with_each_agent(task_creator, agents)
    end_time = time.time()

    logger.info(f"All tasks executed in {end_time - start_time:.2f} seconds")

    for agent_id, result in results.items():
        if result:
            logger.info(f"Agent {agent_id}: {str(result)[:50]}...")
        else:
            logger.warning(f"Agent {agent_id} returned None")

    return results


if __name__ == "__main__":
    logger.info("Starting executor tests...")

    try:
        # Uncomment to test each function separately
        # test_simple_execution()
        # test_parallel_execution()
        # test_tasks_by_agent()
        # test_task_with_each_agent()

        # Or run all tests
        logger.info("Running all tests...")

        result1 = test_simple_execution()
        logger.info("-" * 80)

        result2 = test_parallel_execution()
        logger.info("-" * 80)

        result3 = test_tasks_by_agent()
        logger.info("-" * 80)

        result4 = test_task_with_each_agent()

        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Error during tests: {e}")
        raise
