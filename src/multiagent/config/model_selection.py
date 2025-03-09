"""
Model selection utilities for the Multi-Agent AI Collaboration Tool.
"""

from typing import Dict, List, Set
from rich.console import Console
import curses
import sys

# Available models
AVAILABLE_MODELS = {
    "deepseek": "DeepSeek R1",
    "openai_o3": "OpenAI O3 Mini High",
    "openai_o1": "OpenAI O1",
    "claude_37": "Claude 3.7 Thinking",
    "claude_36": "Claude 3.6",
}


def checkbox_menu(stdscr, items: List[Dict]) -> List[Dict]:
    """
    Display an interactive checkbox menu using curses.

    Args:
        stdscr: curses window object
        items: List of items with 'id', 'label', and 'checked' keys

    Returns:
        Updated list of items with their checked status
    """
    curses.curs_set(0)  # Hide cursor
    current_row = 0

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        # Print instructions
        title = "Model Selection Menu"
        instructions = [
            "Use ↑/↓ arrow keys to navigate",
            "Press SPACE to toggle selection",
            "Press ENTER to confirm",
            "Press Q to quit and use all models",
            "",  # Empty line for spacing
        ]

        # Center the title
        start_x = max(0, (w - len(title)) // 2)
        stdscr.addstr(0, start_x, title)

        # Print instructions
        for idx, line in enumerate(instructions, 1):
            if idx >= h:  # Don't print if we're out of screen space
                break
            stdscr.addstr(idx, 2, line)

        # Print items
        start_row = len(instructions) + 1
        for idx, item in enumerate(items):
            if start_row + idx >= h:  # Leave room for title and status
                break

            # Highlight current row
            if idx == current_row:
                attr = curses.A_REVERSE
            else:
                attr = curses.A_NORMAL

            # Format item text
            checkbox = "[x]" if item["checked"] else "[ ]"
            text = f" {checkbox} {item['label']} ({item['id']})"
            text = text[: w - 1]  # Truncate if too long

            stdscr.addstr(start_row + idx, 2, text, attr)

        # Handle key input
        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(items) - 1:
            current_row += 1
        elif key == ord(" "):  # Space to toggle
            items[current_row]["checked"] = not items[current_row]["checked"]
        elif key == ord("\n"):  # Enter to confirm
            break
        elif key == ord("q"):  # Q to quit and use all
            return None

    return items


def get_model_selection(use_all_models: bool = False) -> Set[str]:
    """
    Get the user's model selection using a curses-based checkbox interface.

    Args:
        use_all_models: If True, skip the selection and use all models

    Returns:
        Set of model IDs to use
    """
    console = Console()

    if use_all_models:
        console.print("[bold green]Using all available models[/bold green]")
        return set(AVAILABLE_MODELS.keys())

    # Initialize checkbox data
    checkboxes = [
        {"id": model_id, "label": model_name, "checked": True}
        for model_id, model_name in AVAILABLE_MODELS.items()
    ]

    try:
        # Run the curses interface
        updated_checkboxes = curses.wrapper(checkbox_menu, checkboxes)

        if updated_checkboxes is None:  # User quit
            console.print("[yellow]Selection cancelled. Using all models by default.[/yellow]")
            return set(AVAILABLE_MODELS.keys())

        # Get selected models
        selected_models = {item["id"] for item in updated_checkboxes if item["checked"]}

        if not selected_models:
            console.print(
                "[bold yellow]No models selected. Using all models by default.[/bold yellow]"
            )
            return set(AVAILABLE_MODELS.keys())

        console.print(
            f"[bold green]Selected models: {', '.join(AVAILABLE_MODELS[model_id] for model_id in selected_models)}[/bold green]"
        )
        return selected_models

    except Exception as e:
        console.print(f"[red]Error during model selection: {e}. Using all models.[/red]")
        return set(AVAILABLE_MODELS.keys())


def filter_agents(agents: Dict, selected_models: Set[str]) -> Dict:
    """
    Filter agents based on selected models.

    Args:
        agents: Dictionary of agents
        selected_models: Set of model IDs to use

    Returns:
        Filtered dictionary of agents
    """
    return {agent_id: agent for agent_id, agent in agents.items() if agent_id in selected_models}
