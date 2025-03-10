import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal
from crewai import Agent
import aiohttp
import asyncio
import aiofiles


class RunLogger:
    def __init__(self, question: str):
        self.question = question
        self.run_dir = None  # Will be set during async initialization

    @classmethod
    async def create(cls, question: str) -> "RunLogger":
        """Create a new RunLogger instance asynchronously"""
        instance = cls(question)
        await instance._initialize()
        return instance

    async def _initialize(self):
        """Initialize the RunLogger asynchronously"""
        self.run_dir = await self._create_run_directory()
        await self._write_question()

    async def _get_run_name(self) -> str:
        """Get a summarized name for the run using Claude 3.6"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                }
                payload = {
                    "model": "openrouter/anthropic/claude-3.5-sonnet",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Please provide a short (5-7 words) title summarizing this question. Make it descriptive but concise, using lowercase with hyphens between words: {self.question}",
                        }
                    ],
                }
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"].strip()
                    else:
                        return "untitled-run"
        except Exception as e:
            print(f"Error getting run name: {e}")
            return "untitled-run"

    async def _create_run_directory(self) -> Path:
        """Create a new run directory with incrementing number prefix"""
        base_dir = Path("docs/runs")
        base_dir.mkdir(parents=True, exist_ok=True)

        # Get existing run directories
        existing_runs = [d for d in base_dir.iterdir() if d.is_dir()]
        next_run_num = (
            1
            if not existing_runs
            else max(
                int(d.name.split("-")[0]) for d in existing_runs if d.name.split("-")[0].isdigit()
            )
            + 1
        )

        # Get run name asynchronously
        run_name = await self._get_run_name()

        # Create timestamped directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"{next_run_num:04d}-{timestamp}-{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    async def _write_question(self):
        """Write the original question to question.md"""
        async with aiofiles.open(self.run_dir / "question.md", "w") as f:
            await f.write(f"# Original Question\n\n{self.question}\n")

    async def log_event(
        self,
        event_type: Union[Literal["error"], Literal["warning"], Literal["info"]],
        source: str,
        data: Dict[str, Any],
    ):
        """Log an event to the run directory.

        Args:
            event_type: Type of event ("error", "warning", "info")
            source: Source/component that generated the event
            data: Event data to log
        """
        events_dir = self.run_dir / "events"
        events_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        event_file = events_dir / f"{timestamp}_{event_type}_{source}.json"

        event_data = {"timestamp": timestamp, "type": event_type, "source": source, **data}

        async with aiofiles.open(event_file, "w") as f:
            await f.write(json.dumps(event_data, indent=2))

    async def log_initial_answer(self, agent: Agent, answer: str):
        """Log an initial answer from an agent"""
        agent_dir = self.run_dir / "1_initial_answers"
        agent_dir.mkdir(exist_ok=True)

        async with aiofiles.open(agent_dir / f"{agent.role}.md", "w") as f:
            content = (
                f"# Initial Answer from {agent.role}\n\n"
                f"## Agent Details\n"
                f"- Role: {agent.role}\n"
                f"- Model: {agent.llm.model}\n\n"
                f"## Answer\n\n{answer}\n"
            )
            await f.write(content)

    async def log_evaluation(self, evaluator: Agent, evaluation_data: Dict[str, Any]):
        """Log an evaluation from an agent"""
        eval_dir = self.run_dir / "2_evaluations"
        eval_dir.mkdir(exist_ok=True)

        content = []
        content.append(f"# Evaluation by {evaluator.role}\n\n")
        content.append(f"## Evaluator Details\n")
        content.append(f"- Role: {evaluator.role}\n")
        content.append(f"- Model: {evaluator.llm.model}\n\n")
        content.append("## Evaluation Criteria\n\n")

        for criterion in evaluation_data.get("criteria", []):
            content.append(f"### {criterion['name']} (Weight: {criterion['weight']})\n")
            content.append(f"{criterion['description']}\n\n")

        content.append("## Scores\n\n")
        for score in evaluation_data.get("scores", []):
            content.append(f"### {score['answer_id']}\n")
            content.append("#### Criteria Scores\n")
            for criterion, score_value in score["criteria_scores"].items():
                content.append(f"- {criterion}: {score_value}\n")
            content.append(f"\n#### Reasoning\n{score['reasoning']}\n\n")

        async with aiofiles.open(eval_dir / f"{evaluator.role}.md", "w") as f:
            await f.write("".join(content))

    async def log_improvement(
        self, improver: Agent, original_answer_id: str, improved_answer: Dict[str, Any]
    ):
        """Log an improvement from an agent"""
        improve_dir = self.run_dir / "3_improvements"
        improve_dir.mkdir(exist_ok=True)

        content = (
            f"# Improvement by {improver.role}\n\n"
            f"## Improver Details\n"
            f"- Role: {improver.role}\n"
            f"- Model: {improver.llm.model}\n"
            f"- Original Answer ID: {original_answer_id}\n\n"
            f"## Improvements Made\n{improved_answer['improvements']}\n\n"
            f"## Improved Answer\n\n{improved_answer['improved_answer']}\n"
        )

        async with aiofiles.open(
            improve_dir / f"{improver.role}_{original_answer_id}.md", "w"
        ) as f:
            await f.write(content)

    async def log_final_judgment(self, judge: Agent, judgment_data: Dict[str, Any]):
        """Log the final judgment"""
        content = (
            f"# Final Judgment by {judge.role}\n\n"
            f"## Judge Details\n"
            f"- Role: {judge.role}\n"
            f"- Model: {judge.llm.model}\n\n"
            f"## Selected Answer\n"
            f"Best Answer ID: {judgment_data['best_answer_id']}\n\n"
            f"## Reasoning\n{judgment_data['reasoning']}\n\n"
            f"## Final Score: {judgment_data['final_score']}\n"
        )

        async with aiofiles.open(self.run_dir / "4_final_judgment.md", "w") as f:
            await f.write(content)

    async def log_summary(self, final_answer: str):
        """Write a summary of the entire run"""
        content = (
            f"# Run Summary\n\n"
            f"## Question\n{self.question}\n\n"
            f"## Final Answer\n{final_answer}\n\n"
            f"\nRun completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        async with aiofiles.open(self.run_dir / "summary.md", "w") as f:
            await f.write(content)
