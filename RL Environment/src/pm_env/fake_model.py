import json
from collections.abc import AsyncGenerator

from pm_env.evaluation_runner import EvaluationRunner
from pm_env.get_data_dir import get_env_data_dir
from pm_env.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from pm_env.schemas.transcript import Event, MessageAddedEvent

# Add any messages you want the model to produce.
# The evaluation runner assumes that the model keeps working on a task step
# until a message does not contain a tool call, so make sure to always add a
# message without tool calls to the end of each step.
messages: list[Message] = [
    # Step 1
    Message(
        role="assistant",
        content="Here is the path to my Python executable.",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="tool_call_1",
                type="function",
                function=Function(
                    name="submit_answers",
                    arguments=json.dumps({"path": "/workdir/.venv/bin/python"}),
                ),
            )
        ],
    ),
    Message(role="assistant", content=""),
    # Step 2
    Message(
        role="assistant",
        content="And the Python version.",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="tool_call_1",
                type="function",
                function=Function(
                    name="bash",
                    arguments=json.dumps(
                        {
                            "command": f"echo '3.12.11' > {get_env_data_dir()}/python_version.txt"
                        }
                    ),
                ),
            )
        ],
    ),
    Message(role="assistant", content=""),
]


def setup_fake_model(evaluation_runner: EvaluationRunner):
    evaluation_runner._collect_model_response = (
        send_messages()
    )  # pyright: ignore[reportPrivateUsage]


def send_messages():
    i_message = 0

    async def send_message() -> AsyncGenerator[Event]:
        nonlocal i_message
        if i_message >= len(messages):
            raise RuntimeError("Not enough messages defined for fake model.")

        yield MessageAddedEvent(message=messages[i_message])
        i_message += 1

    return send_message
