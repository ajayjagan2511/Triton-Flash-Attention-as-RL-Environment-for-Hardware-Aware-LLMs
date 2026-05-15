import json
import traceback
import warnings
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

import litellm
from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    Choices,
    CustomStreamWrapper,
)
from litellm.types.utils import Message as LiteLlmMessage
from litellm.types.utils import ModelResponse, ModelResponseStream, StreamingChoices
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from pm_env.load_task import load_task
from pm_env.schemas import RunStatus
from pm_env.schemas.chat import ChatCompletionMessageToolCall, Delta, Message
from pm_env.schemas.evaluation_run_config import EvaluationRunConfig
from pm_env.schemas.run_state import RunState
from pm_env.schemas.transcript import (
    AnswersSubmittedEvent,
    ErrorEvent,
    Event,
    MessageAddedEvent,
    MessageChunkEvent,
    ResourceMetrics,
    ScoringEvent,
    StepCompletedEvent,
    StepStartedEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    Transcript,
)
from pm_env.system_prompts import get_system_message
from pm_env.task import Step, Task


class EvaluationRunner:
    def __init__(self, config: EvaluationRunConfig):
        self.config: Final[EvaluationRunConfig] = config
        self.task: Task
        self.transcript: Transcript
        self.run_state: RunState
        self.tools: list[ChatCompletionToolParam]

    async def run(self) -> AsyncGenerator[Event]:
        # Can be removed when https://github.com/BerriAI/litellm/issues/10202 gets fixed
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
        # Can be removed when https://github.com/BerriAI/litellm/issues/14230 gets fixed
        warnings.filterwarnings(
            "ignore", message="Use 'content=<...>' to upload raw bytes/text content."
        )

        try:
            async with self._connect_to_mcp_server() as mcp_client:
                self.task = load_task(self.config)
                await mcp_client.call_tool("register_tools", {"tools": self.task.tools})
                async for event in self._run(mcp_client):
                    yield event
        except (Exception, KeyboardInterrupt) as e:
            error_event = ErrorEvent(
                exception_type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_exc(),
            )
            completed_event = TaskCompletedEvent(status="error")
            yield self._process_event(error_event)
            yield self._process_event(completed_event)
        finally:
            self._maybe_save_transcript_to_file()

    async def _run(
        self, mcp_client: Client[StreamableHttpTransport]
    ) -> AsyncGenerator[Event]:
        self.tools = await self._load_tools(mcp_client)

        self._delete_transcript()
        self.transcript = Transcript(run_id=self.config.run_id)

        yield self._process_event(
            TaskStartedEvent(
                run_id=self.config.run_id,
                task_id=self.task.id,
                model=self.config.model,
                n_steps=len(self.task.steps),
            )
        )

        system_message = get_system_message(self.config.model)
        if system_message.content:
            yield self._process_event(MessageAddedEvent(message=system_message))
        else:
            print("ALERT system message is empty, skipping!")

        run_status: RunStatus = "passed"

        for i_step, step in enumerate(self.task.steps):
            yield self._process_event(StepStartedEvent(step=i_step))
            event = None
            async for event in self._execute_step(step, mcp_client):
                yield event

            assert isinstance(event, ScoringEvent)
            yield self._process_event(StepCompletedEvent(step=i_step))

            if not event.scoring.continue_task:
                run_status = "failed"
                break

        yield self._process_event(TaskCompletedEvent(status=run_status))

    async def _execute_step(
        self, step: Step, mcp_client: Client[StreamableHttpTransport]
    ) -> AsyncGenerator[Event]:
        instructions = LiteLlmMessage(role="user", content=step.instructions)
        yield self._process_event(
            MessageAddedEvent(message=Message(**instructions.model_dump()))
        )

        while True:
            message_event = None
            async for event in self._collect_model_response():
                yield event
                if isinstance(event, MessageAddedEvent):
                    message_event = event

            assert message_event is not None
            message = message_event.message

            if message.tool_calls is None:
                # Step is done when the model doesn't want to make any more tool calls.
                break

            assert message.tool_calls is not None
            async for event in self._execute_tool_calls(message.tool_calls, mcp_client):
                yield event

        scoring = step.judge.evaluate(self.transcript)
        yield self._process_event(ScoringEvent(scoring=scoring))

    @asynccontextmanager
    async def _connect_to_mcp_server(self):
        url = f"http://{self.config.mcp_server_config.host}:{self.config.mcp_server_config.port}/mcp"
        transport = StreamableHttpTransport(url)
        async with Client[StreamableHttpTransport](transport) as client:
            yield client

    async def _load_tools(
        self, client: Client[StreamableHttpTransport]
    ) -> list[ChatCompletionToolParam]:
        """Retrieves the tools available for this task using the LiteLLM/OpenAI tool schema."""
        return [
            self._convert_mcp_tool_to_litellm_tool(tool)
            for tool in await client.list_tools()
        ]

    def _convert_mcp_tool_to_litellm_tool(self, tool: Tool) -> ChatCompletionToolParam:
        """The evaluation runner needs to use the MCP tool schema when talking to the MCP server
        but needs to expose tools to the model via the LiteLLM/OpenAI schema."""
        assert tool.description is not None
        return ChatCompletionToolParam(
            type="function",
            function=ChatCompletionToolParamFunctionChunk(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
            ),
        )

    async def _collect_model_response(self) -> AsyncGenerator[Event]:
        chunks = []

        response = await litellm.acompletion(
            **self._get_completion_params()
        )  # pyright: ignore[reportArgumentType]

        assert isinstance(response, CustomStreamWrapper)

        async for chunk in response:
            assert isinstance(chunk, ModelResponseStream)
            assert isinstance(chunk.choices[0], StreamingChoices)

            chunks.append(chunk)

            delta = chunk.choices[0].delta

            yield MessageChunkEvent(delta=Delta(**delta.model_dump()))

        full_response = litellm.stream_chunk_builder(chunks)

        assert isinstance(full_response, ModelResponse)
        assert isinstance(full_response.choices[0], Choices)

        yield self._process_event(
            MessageAddedEvent(
                message=Message(**full_response.choices[0].message.model_dump())
            )
        )

        # Emit token usage if available
        usage = getattr(full_response, "usage", None)
        if usage:
            yield self._process_event(
                TokenUsageEvent(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    cache_read_tokens=getattr(usage, "cache_read_input_tokens", None),
                    cache_write_tokens=getattr(
                        usage, "cache_creation_input_tokens", None
                    ),
                )
            )

    async def _execute_tool_calls(
        self,
        tool_calls: Sequence[ChatCompletionMessageToolCall],
        mcp_client: Client[StreamableHttpTransport],
    ) -> AsyncGenerator[Event]:
        for tool_call in tool_calls:
            yield self._process_event(ToolCallStartedEvent(tool_call=tool_call))

            assert tool_call.function.name

            answers = None

            try:
                if tool_call.function.name == "submit_answers":
                    result, answers = _submit_answers(tool_call)
                else:
                    result = await mcp_client.call_tool_mcp(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments or "{}"),
                    )
            except json.JSONDecodeError:
                result = _build_call_tool_result(
                    result=f"Error: Invalid arguments {tool_call.function.arguments!r}",
                    is_error=True,
                )

            # Extract resource metrics from structuredContent if present
            resource_metrics: ResourceMetrics | None = None
            if (
                result.structuredContent
                and "_pm_env_resource_metrics" in result.structuredContent
            ):
                resource_metrics = ResourceMetrics(
                    **result.structuredContent["_pm_env_resource_metrics"]
                )

            yield self._process_event(
                ToolCallCompletedEvent(
                    tool_call_id=tool_call.id,
                    result=result,
                    resource_metrics=resource_metrics,
                )
            )

            if answers:
                yield self._process_event(AnswersSubmittedEvent(answers=answers))

            result_dict = {}
            result_dict["role"] = "tool"
            result_dict["tool_call_id"] = tool_call.id

            if len(result.content) == 0:
                result_dict["content"] = []

            elif type(result.content[0]) is ImageContent:
                image_url = (
                    f"data:{result.content[0].mimeType};base64,{result.content[0].data}"
                )

                result_dict["content"] = [
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            else:
                result_dict["content"] = [result.content[0].model_dump()]

            yield self._process_event(MessageAddedEvent(message=Message(**result_dict)))

    def _get_completion_params(self):
        completion_params = {
            "stream": True,
            "stream_options": {"include_usage": True},
            "model": self.config.model,
            "messages": self._prepare_messages(),
            "tools": self.tools,
            "tool_choice": "auto" if self.tools else None,
            "max_tokens": 10000,
            "num_retries": 10,
            "timeout": 300.0,
            # this is needed for together ai and other such providers, doesn't seem to break
            # claude api calls, so doing it for everyone
            "allowed_openai_params": ["tools", "tool_choice"],
        }

        completion_params["api_key"] = self.config.model_api_key

        return completion_params

    def _prepare_messages(self) -> list[dict[str, Any]]:
        """Serializes messages and adds cache control.

        See:
            https://platform.claude.com/docs/en/build-with-claude/prompt-caching
        """
        messages: list[dict[str, Any]] = [
            message.model_dump() for message in self.transcript.messages
        ]

        # Add cache_control to last message with content
        for message in reversed(messages):
            if isinstance(message["content"], str):
                # Convert to list of dictionaries so we can add cache control
                message["content"] = [{"type": "text", "text": message["content"]}]
            if isinstance(message["content"], list) and message["content"]:
                message["content"][-1]["cache_control"] = (
                    {  # pyright: ignore[reportArgumentType]
                        "type": "ephemeral",
                        "ttl": "1h",
                    }
                )
                break

        return messages

    def _delete_transcript(self):
        if self.config.transcript_file:
            try:
                Path(self.config.transcript_file).unlink()
            except FileNotFoundError:
                pass

    def _process_event(self, event: Event):
        if hasattr(self, "transcript"):
            self.transcript.events.append(event)

        if hasattr(self, "run_state"):
            self.run_state.apply(event)
        elif isinstance(event, TaskStartedEvent):
            self.run_state = RunState(event)

        return event

    def _maybe_save_transcript_to_file(self):
        if hasattr(self, "transcript") and self.config.transcript_file:
            Path(self.config.transcript_file).parent.mkdir(parents=True, exist_ok=True)
            Path(self.config.transcript_file).write_text(
                self.transcript.model_dump_json(indent=2)
            )
            _ = print(f"\n📁 Transcript saved to: {self.config.transcript_file}")


def _submit_answers(
    tool_call: ChatCompletionMessageToolCall,
) -> tuple[CallToolResult, dict[str, str]]:
    arguments = json.loads(tool_call.function.arguments or "{}")

    if not isinstance(arguments, dict):
        return (
            _build_call_tool_result(
                result=f"Error: Invalid arguments {arguments!r}",
                is_error=True,
            ),
            {},
        )

    if len(arguments) == 1 and "answers" in arguments:
        answers = arguments["answers"]
    else:
        answers = arguments

    # Convert values to strings
    answers = {key: str(value) for key, value in answers.items()}

    return _build_call_tool_result(result="Answers submitted", is_error=False), answers


def _build_call_tool_result(result: str, is_error: bool) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=result)],
        structuredContent={"result": result},
        isError=is_error,
    )
