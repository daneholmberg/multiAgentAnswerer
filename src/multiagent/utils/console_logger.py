from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
from rich.style import Style

# Set up console theme
custom_theme = Theme(
    {
        "info": Style(color="cyan"),
        "warning": Style(color="yellow"),
        "error": Style(color="red", bold=True),
        "step": Style(color="magenta", bold=True),
        "success": Style(color="green", bold=True),
        "model": Style(color="blue", bold=True),
        "timestamp": Style(color="grey50"),
    }
)


class ConsoleLogger:
    def __init__(self):
        self.console = Console(theme=custom_theme)

    def _get_timestamp(self) -> str:
        """Get current timestamp in HH:MM:SS format"""
        return f"[timestamp]{datetime.now().strftime('%H:%M:%S')}[/timestamp]"

    def start_step(self, step_name: str):
        """Log the start of a major step"""
        self.console.print(f"{self._get_timestamp()} [step]▶ Starting: {step_name}[/step]")

    def complete_step(self, step_name: str):
        """Log the completion of a major step"""
        self.console.print(f"{self._get_timestamp()} [success]✓ Completed: {step_name}[/success]")

    def model_start(self, model_name: str, task: str):
        """Log when a model starts processing"""
        self.console.print(f"{self._get_timestamp()} [model]{model_name}[/model] starting {task}")

    def model_complete(self, model_name: str, task: str):
        """Log when a model completes processing"""
        self.console.print(f"{self._get_timestamp()} [model]{model_name}[/model] completed {task}")

    def info(self, message: str):
        """Log an informational message"""
        self.console.print(f"{self._get_timestamp()} [info]{message}[/info]")

    def warning(self, message: str):
        """Log a warning message"""
        self.console.print(f"{self._get_timestamp()} [warning]⚠ {message}[/warning]")

    def error(self, message: str):
        """Log an error message"""
        self.console.print(f"{self._get_timestamp()} [error]❌ {message}[/error]")

    def display_question(self, question: str):
        """Display the initial question in a panel"""
        self.console.print(Panel(f"[info]{question}[/info]", title="Question"))
