import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from crewai import Agent
import aiohttp
import asyncio


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
        self._write_question()

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

    def _write_question(self):
        """Write the original question to question.md"""
        with open(self.run_dir / "question.md", "w") as f:
            f.write(f"# Original Question\n\n{self.question}\n")

    def log_initial_answer(self, agent: Agent, answer: str):
        """Log an initial answer from an agent"""
        agent_dir = self.run_dir / "1_initial_answers"
        agent_dir.mkdir(exist_ok=True)

        with open(agent_dir / f"{agent.role}.md", "w") as f:
            f.write(f"# Initial Answer from {agent.role}\n\n")
            f.write(f"## Agent Details\n")
            f.write(f"- Role: {agent.role}\n")
            f.write(f"- Model: {agent.llm.model}\n\n")
            f.write(f"## Answer\n\n{answer}\n")

    def log_evaluation(self, evaluator: Agent, evaluation_data: Dict[str, Any]):
        """Log an evaluation from an agent"""
        eval_dir = self.run_dir / "2_evaluations"
        eval_dir.mkdir(exist_ok=True)

        with open(eval_dir / f"{evaluator.role}.md", "w") as f:
            f.write(f"# Evaluation by {evaluator.role}\n\n")
            f.write(f"## Evaluator Details\n")
            f.write(f"- Role: {evaluator.role}\n")
            f.write(f"- Model: {evaluator.llm.model}\n\n")
            f.write("## Evaluation Criteria\n\n")

            for criterion in evaluation_data.get("criteria", []):
                f.write(f"### {criterion['name']} (Weight: {criterion['weight']})\n")
                f.write(f"{criterion['description']}\n\n")

            f.write("## Scores\n\n")
            for score in evaluation_data.get("scores", []):
                f.write(f"### {score['answer_id']}\n")
                f.write("#### Criteria Scores\n")
                for criterion, score_value in score["criteria_scores"].items():
                    f.write(f"- {criterion}: {score_value}\n")
                f.write(f"\n#### Reasoning\n{score['reasoning']}\n\n")

    def log_improvement(
        self, improver: Agent, original_answer_id: str, improved_answer: Dict[str, Any]
    ):
        """Log an improvement from an agent"""
        improve_dir = self.run_dir / "3_improvements"
        improve_dir.mkdir(exist_ok=True)

        with open(improve_dir / f"{improver.role}_{original_answer_id}.md", "w") as f:
            f.write(f"# Improvement by {improver.role}\n\n")
            f.write(f"## Improver Details\n")
            f.write(f"- Role: {improver.role}\n")
            f.write(f"- Model: {improver.llm.model}\n")
            f.write(f"- Original Answer ID: {original_answer_id}\n\n")
            f.write(f"## Improvements Made\n{improved_answer['improvements']}\n\n")
            f.write(f"## Improved Answer\n\n{improved_answer['improved_answer']}\n")

    def log_final_judgment(self, judge: Agent, judgment_data: Dict[str, Any]):
        """Log the final judgment"""
        with open(self.run_dir / "4_final_judgment.md", "w") as f:
            f.write(f"# Final Judgment by {judge.role}\n\n")
            f.write(f"## Judge Details\n")
            f.write(f"- Role: {judge.role}\n")
            f.write(f"- Model: {judge.llm.model}\n\n")
            f.write(f"## Selected Answer\n")
            f.write(f"Best Answer ID: {judgment_data['best_answer_id']}\n\n")
            f.write(f"## Reasoning\n{judgment_data['reasoning']}\n\n")
            f.write(f"## Final Score: {judgment_data['final_score']}\n")

    def log_summary(self, final_answer: str):
        """Write a summary of the entire run"""
        with open(self.run_dir / "summary.md", "w") as f:
            f.write(f"# Run Summary\n\n")
            f.write(f"## Question\n{self.question}\n\n")
            f.write(f"## Final Answer\n{final_answer}\n\n")
            f.write(
                f"\nRun completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
