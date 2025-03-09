# Comprehensive Guide and Prompt for Implementing a Multi-Agent AI Collaboration Tool using CrewAI

## Objective:

Build a robust yet simple Python-based personal project leveraging CrewAI to orchestrate multiple state-of-the-art AI models: DeepSeek R1, OpenAI O3 Mini High, OpenAI O1, Claude 3.7 Thinking, and Claude 3.6 (Claude 3.5 new). These models will independently answer a given question or task, anonymously judge each other's responses, and iteratively improve the answers based on structured feedback.

## Project Workflow

### Step 1: Initial Independent Answering

- Prompt each model independently with the user's query.
- Collect individual answers without revealing the origin of each response.

### Step 2: Anonymous Evaluation

- Each model will anonymously evaluate all collected answers (including its own).
- Models must:
  - Define a set of evaluation criteria relevant to the question/task.
  - Assign each criterion a weight based on importance.
  - Score each answer based on these criteria.
- Clearly output criteria, weights, scores, and reasoning.

### Step 2: Identify Best Answers

- Compute weighted scores from each model's evaluation.
- Select the best 2 answers. If the next best answers have scores close enough to the 2nd best, include them as well.

### Step 3: Iterative Improvement

- Feed the best answer(s) back into each model with the evaluation feedback.
- Prompt models to suggest specific improvements for the selected answers.
- Generate improved versions of these answers.

### Step 4: Final Judgement Round

- Repeat the evaluation process on the refined answers to choose the final best answer.

## Implementation Using CrewAI

### Define CrewAI Agents

- Create CrewAI agents corresponding to each AI model (Deepseek R1, OpenAI O3 mini-high, O1, Claude 3.7 Thinking, Claude 3.5 new).
- Clearly define the role, persona, and backend (LLM API) for each agent.

### Define Tasks

- Clearly separate tasks: Answer generation, criteria creation and scoring (Judge), and iterative improvement.

#### Example Agent Definitions:

```python
from crewai import Agent, Task, Crew

# Define Answering Agents
agent_deepseek = Agent(role="DeepseekExpert", backstory="You excel at deep analytical responses.", goal="Provide comprehensive answers.", llm="Deepseek R1 API")
agent_o3 = Agent(role="OpenAIMiniHigh", backstory="Specialist in concise, accurate responses.", goal="Provide clear and succinct answers.", llm="OpenAI O3 Mini High API")
agent_o1 = Agent(role="OpenAIStandard", backstory="You balance clarity and depth.", goal="Provide accurate, balanced answers.", llm="OpenAI O1 API")
agent_claude37 = Agent(role="Claude37Thinker", backstory="You provide thoughtful, reflective answers.", goal="Offer insightful responses.", llm="Claude 3.7 API")
agent_claude35new = Agent(role="Claude35Concise", backstory="You provide concise yet informative answers.", goal="Answer clearly and succinctly.", llm="Claude 3.5 (Claude 3.6) API")

# Define Evaluator Agent
judge_agent = Agent(role="Evaluator", backstory="You objectively evaluate answers.", goal="Create criteria, assign weights, evaluate and score each answer.", llm="Claude 3.7 API")

### Execution of Tasks

# User question
user_question = "<Insert user's query here>"

# Agents independently answer
answers = {}
for agent in [agent_deepseek, agent_o3, agent_o1, agent_claude37]:
    answer = agent.run(user_question)
    answers[agent.role] = answer

# Evaluation Task
judge_task = Task(
    description=f"Evaluate anonymously these answers to: '{user_question}'. Create weighted criteria, score each answer, and provide detailed reasoning.",
    agent=agent_claude37
)
evaluation_result = judge_agent.run(f"Question: {user_question}\nAnswers: {answers}")

# Parse evaluation_result to select best answers and perform refinement if necessary

### Best Practices

- Ensure anonymity of responses by using neutral identifiers (Answer 1, Answer 2, etc.) during evaluations.
- Keep iterative improvements limited to 2-3 rounds for clarity.
- Use asynchronous or parallel API calls to enhance performance and reduce latency.
- Maintain comprehensive logs to monitor each iteration for debugging and analysis purposes.

## Additional Helpful Libraries
- `requests` or SDKs provided by AI models to handle API interactions.
- `asyncio` or `concurrent.futures` for parallelizing API calls.
- `json` or `pydantic` to structure and parse agent responses and evaluation outputs.

By following this guide, you will efficiently construct a robust and simple-to-use multi-agent AI system capable of collaborative, iterative answer generation and evaluation.

```
