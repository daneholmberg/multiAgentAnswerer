import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from crewai import Agent, LLM

# Load environment variables
load_dotenv()

# Set up API keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Provider-specific configurations
PROVIDER_CONFIGS = {
    "openai": {
        "prefix": "openrouter/openai",
        "temperature": 1.0,
    },
    "anthropic": {
        "prefix": "openrouter/anthropic",
        "temperature": 0.5,
    },
    "deepseek": {
        "prefix": "openrouter/deepseek",
        "temperature": 0.6,
    },
}


def create_llm(provider: str, model: str, reasoning_effort: Optional[str] = None) -> LLM:
    """Create an LLM configuration via OpenRouter.

    Args:
        provider: The provider name ('openai', 'anthropic', or 'deepseek')
        model: The specific model name
        reasoning_effort: Optional reasoning effort parameter for OpenAI models
    """
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unsupported provider: {provider}")

    config = PROVIDER_CONFIGS[provider]
    return LLM(
        model=f"{config['prefix']}/{model}",
        base_url="https://openrouter.ai/api/v1",
        temperature=config["temperature"],
        max_tokens=4000,
        reasoning_effort=reasoning_effort if provider == "openai" and "o1" in model else None,
        api_key=OPENROUTER_API_KEY,
    )


def create_answering_agent(
    role: str,
    backstory: str,
    llm: LLM,
    agent_id: str,
) -> Agent:
    """Create an agent specialized in answering questions."""
    return Agent(
        role=role,
        backstory=backstory,
        goal=f"Provide the best possible answer to the user's question as {role}.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        tools=[],
    )


def create_evaluator_agent(
    role: str,
    backstory: str,
    llm: LLM,
    agent_id: str,
) -> Agent:
    """Create an agent specialized in evaluating answers."""
    return Agent(
        role=role,
        backstory=backstory,
        goal="Objectively evaluate answers based on relevant criteria.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        tools=[],
    )


def create_improver_agent(
    role: str,
    backstory: str,
    llm: LLM,
    agent_id: str,
) -> Agent:
    """Create an agent specialized in improving answers."""
    return Agent(
        role=role,
        backstory=backstory,
        goal="Improve the given answer based on evaluation feedback.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        tools=[],
    )


# Create the LLM configurations
deepseek_r1_config = create_llm("deepseek", "deepseek-r1")
openai_o3_mini_high_config = create_llm("openai", "o3-mini-high", reasoning_effort="high")
openai_o1_config = create_llm("openai", "o1")
claude_37_thinking_config = create_llm("anthropic", "claude-3.7-sonnet:thinking")
claude_36_config = create_llm("anthropic", "claude-3.5-sonnet")


# Create the agents
def get_answering_agents() -> Dict[str, Agent]:
    """Get the answering agents."""
    return {
        "deepseek": create_answering_agent(
            role="DeepSeek Expert",
            backstory="You are an expert in deep analytical responses with the Deepseek R1 model. Your strength is providing comprehensive, well-reasoned answers that consider multiple perspectives.",
            llm=deepseek_r1_config,
            agent_id="deepseek",
        ),
        "openai_o3": create_answering_agent(
            role="OpenAI O3 Expert",
            backstory="You are a specialist in concise, accurate responses using the OpenAI O3 Mini High model. Your strength is clarity and efficiency in your answers.",
            llm=openai_o3_mini_high_config,
            agent_id="openai_o3",
        ),
        "openai_o1": create_answering_agent(
            role="OpenAI O1 Expert",
            backstory="You balance clarity and depth using the OpenAI O1 model. Your strength is providing accurate, well-balanced answers that are both informative and accessible.",
            llm=openai_o1_config,
            agent_id="openai_o1",
        ),
        "claude_37": create_answering_agent(
            role="Claude Thinking Expert",
            backstory="You provide thoughtful, reflective answers using the Claude 3.7 Thinking model. Your strength is insight and nuance in your responses.",
            llm=claude_37_thinking_config,
            agent_id="claude_37",
        ),
        "claude_36": create_answering_agent(
            role="Claude 3.6 Expert",
            backstory="You provide concise yet informative answers using the Claude 3.6 model. Your strength is balancing brevity with depth.",
            llm=claude_36_config,
            agent_id="claude_36",
        ),
    }


def get_evaluator_agents() -> Dict[str, Agent]:
    """Get the evaluator agents."""
    return {
        "deepseek": create_evaluator_agent(
            role="DeepSeek Judge",
            backstory="You are an objective evaluator using the Deepseek R1 model. You assess answers based on their merits without bias.",
            llm=deepseek_r1_config,
            agent_id="deepseek",
        ),
        "openai_o3": create_evaluator_agent(
            role="OpenAI O3 Judge",
            backstory="You are a fair and balanced evaluator using the OpenAI O3 Mini High model. You focus on practicality and clarity in your assessments.",
            llm=openai_o3_mini_high_config,
            agent_id="openai_o3",
        ),
        "openai_o1": create_evaluator_agent(
            role="OpenAI O1 Judge",
            backstory="You are a thorough evaluator using the OpenAI O1 model. Your evaluations are detailed and consider multiple dimensions of quality.",
            llm=openai_o1_config,
            agent_id="openai_o1",
        ),
        "claude_37": create_evaluator_agent(
            role="Claude Thinking Judge",
            backstory="You are a thoughtful evaluator using the Claude 3.7 Thinking model. You consider both the logical and creative aspects of answers.",
            llm=claude_37_thinking_config,
            agent_id="claude_37",
        ),
        "claude_36": create_evaluator_agent(
            role="Claude 3.6 Judge",
            backstory="You are a balanced evaluator using the Claude 3.6 model. You assess answers based on their accuracy, clarity, and relevance.",
            llm=claude_36_config,
            agent_id="claude_36",
        ),
    }


def get_improver_agents() -> Dict[str, Agent]:
    """Get the improver agents."""
    return {
        "deepseek": create_improver_agent(
            role="DeepSeek Improver",
            backstory="You are an expert at refining answers using the Deepseek R1 model. You identify weaknesses and enhance strengths to create better responses.",
            llm=deepseek_r1_config,
            agent_id="deepseek",
        ),
        "openai_o3": create_improver_agent(
            role="OpenAI O3 Improver",
            backstory="You are skilled at enhancing answers for clarity using the OpenAI O3 Mini High model. You make answers more concise and accessible.",
            llm=openai_o3_mini_high_config,
            agent_id="openai_o3",
        ),
        "openai_o1": create_improver_agent(
            role="OpenAI O1 Improver",
            backstory="You excel at balanced improvements using the OpenAI O1 model. You enhance both substance and style in answers.",
            llm=openai_o1_config,
            agent_id="openai_o1",
        ),
        "claude_37": create_improver_agent(
            role="Claude Thinking Improver",
            backstory="You are adept at thoughtful refinements using the Claude 3.7 Thinking model. You add depth and insight to answers.",
            llm=claude_37_thinking_config,
            agent_id="claude_37",
        ),
        "claude_36": create_improver_agent(
            role="Claude 3.6 Improver",
            backstory="You are skilled at enhancing answers using the Claude 3.6 model. You improve both content and presentation for maximum impact.",
            llm=claude_36_config,
            agent_id="claude_36",
        ),
    }
