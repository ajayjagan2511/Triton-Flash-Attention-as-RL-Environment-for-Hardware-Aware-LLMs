import json
import sys
import warnings
from collections.abc import AsyncIterator
from typing import Any

from rich import print
from rich.markup import escape

from pm_env.schemas.transcript import (
    AnswersSubmittedEvent,
    ErrorEvent,
    Event,
    MessageAddedEvent,
    MessageChunkEvent,
    ScoringEvent,
    StepCompletedEvent,
    StepStartedEvent,
    TaskCompletedEvent,
    TaskPreHookCompletedEvent,
    TaskStartedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)


async def stream_transcript_to_stdout(event_stream: AsyncIterator[Event]):
    # https://github.com/fastapi/sqlmodel/discussions/1369
    warnings.filterwarnings(
        "ignore",
        message=".*Accessing the 'model_fields' attribute on the instance is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*The `dict` method is deprecated; use `model_dump` instead.*",
    )

    new_message = True

    async for event in event_stream:
        match event:
            case TaskStartedEvent():
                print(f"[blue]{'━' * 80}[/blue]")
                print(f"[blue]Task: {escape(event.task_id)}[/blue]")
                print(f"[blue]{'━' * 80}[/blue]")

            case TaskPreHookCompletedEvent():
                print("Executed task pre hook")
                print("Pre hook metadata: ")
                for key, value in event.metadata.items():
                    _print_kv(key, value)

            case StepStartedEvent():
                print(f"\n\n Starting step {event.step + 1}")

            case MessageChunkEvent():
                if event.delta.content:
                    if new_message:
                        _ = print("\n👤 Assistant:")
                        new_message = False
                    sys.stdout.write(event.delta.content)
                    sys.stdout.flush()

            case MessageAddedEvent():
                new_message = True
                if event.message.role == "user" and event.message.content:
                    _ = print(f"\n👤 User: {escape(str(event.message.content))}\n")

            case ToolCallStartedEvent():
                match event.tool_call.function.name:
                    case "bash":
                        print("\n\n🔧 Calling Bash:")
                        try:
                            arguments = json.loads(event.tool_call.function.arguments)
                            for line in str(arguments["command"]).splitlines():
                                print(escape(line))
                        except (json.JSONDecodeError, KeyError):
                            print(
                                f"Invalid arguments: {escape(repr(event.tool_call.function.arguments))}"
                            )

                        print()
                    case _:
                        print(
                            f"\n\n🔧 Calling tool: {escape(event.tool_call.function.name or '')} {escape(event.tool_call.function.arguments)}"
                        )

            case ToolCallCompletedEvent():
                print("✅ Tool call completed:")
                if event.result.structuredContent:
                    for key, value in event.result.structuredContent.items():
                        print(f"{escape(str(key))}:\n{escape(str(value))}")

            case AnswersSubmittedEvent():
                print("\n\n✅ Answers submitted:")
                for key, value in event.answers.items():
                    _print_kv(key, value)

            case ScoringEvent():
                scoring = event.scoring
                print("\n\n✅ Scoring completed:")
                print("[bold]Score:[/bold]", scoring.score)

                print(
                    f"\n[bold]Continue task? -> {'Yes' if scoring.continue_task else 'No'}[/bold]"
                )

                print("\n[bold]Metadata:[/bold]")
                for key, value in scoring.metadata.items():
                    _print_kv(key, value)

                print("\n\n")

            case ErrorEvent():
                print("\n\n[bold red]💥 Error occurred:[/bold red]")
                print(f"[red]Exception Type: {escape(event.exception_type)}[/red]")
                print(f"[red]Message: {escape(event.message)}[/red]")
                if event.traceback:
                    print("\n[red]Full Traceback:[/red]")
                    print(f"[red]{escape(event.traceback)}[/red]\n")

            case StepCompletedEvent():
                pass
            case TaskCompletedEvent():
                pass
            case TokenUsageEvent():
                pass
            case _:
                print(f"[bold red]Unknown event type: {type(event).__name__}[bold red]")


def _print_kv(key: Any, value: Any) -> None:
    """Print a key-value pair with Rich markup escaping."""
    print(f"{escape(str(key))}: {escape(str(value))}")
