import datetime as dt
from typing import Any, Literal

from mcp.types import CallToolResult
from pydantic import BaseModel, Field, SerializeAsAny

from . import RunStatus
from .chat import ChatCompletionMessageToolCall, Delta, Message
from .scoring import Scoring


class BaseEvent(BaseModel):
    """Base class for all events that happen during an evaluation run."""

    timestamp: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))


class MetadataEvent(BaseEvent):
    """For events that add metadata to the run."""

    metadata: dict[str, Any]


class TaskStartedEvent(BaseEvent):
    type: Literal["task_started"] = "task_started"
    run_id: str
    task_id: str
    model: str | None = None
    n_steps: int


class TaskPreHookCompletedEvent(MetadataEvent):
    type: Literal["task_pre_hook_completed"] = "task_pre_hook_completed"


class StepStartedEvent(BaseEvent):
    type: Literal["step_started"] = "step_started"
    step: int
    """0-indexed step number."""


class MessageChunkEvent(BaseEvent):
    type: Literal["message_chunk"] = "message_chunk"
    delta: Delta


class MessageAddedEvent(BaseEvent):
    type: Literal["message_added"] = "message_added"
    message: Message


class ToolCallStartedEvent(BaseEvent):
    type: Literal["tool_call_started"] = "tool_call_started"
    tool_call: ChatCompletionMessageToolCall


class ResourceSample(BaseModel):
    """A single resource sample taken during tool execution."""

    timestamp_ms: int
    """Milliseconds since tool call started."""
    cpu_percent: float
    """CPU utilization percentage (0-100+, can exceed 100 on multi-core)."""
    memory_mb: float
    """Memory usage in megabytes."""


class ResourceMetrics(BaseModel):
    """Aggregated resource metrics for a tool call."""

    samples: list[ResourceSample] = []
    """Time-series samples."""
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0


class ToolCallCompletedEvent(BaseEvent):
    type: Literal["tool_call_completed"] = "tool_call_completed"
    tool_call_id: str
    result: CallToolResult
    resource_metrics: ResourceMetrics | None = None
    """Resource usage during tool execution (CPU, memory)."""


class AnswersSubmittedEvent(BaseEvent):
    type: Literal["answers_submitted"] = "answers_submitted"
    answers: dict[str, str]


class ScoringEvent(BaseEvent):
    type: Literal["scoring"] = "scoring"
    scoring: Scoring


class StepCompletedEvent(BaseEvent):
    type: Literal["step_completed"] = "step_completed"
    step: int
    """0-indexed step number."""


class ErrorEvent(BaseEvent):
    type: Literal["error"] = "error"
    exception_type: str
    message: str
    traceback: str | None = None


class TaskCompletedEvent(BaseEvent):
    type: Literal["task_completed"] = "task_completed"
    status: RunStatus


class TokenUsageEvent(BaseEvent):
    type: Literal["token_usage"] = "token_usage"
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


Event = (
    TaskStartedEvent
    | StepStartedEvent
    | StepCompletedEvent
    | TaskCompletedEvent
    | MessageChunkEvent
    | MessageAddedEvent
    | ToolCallStartedEvent
    | ToolCallCompletedEvent
    | AnswersSubmittedEvent
    | ScoringEvent
    | ErrorEvent
    | TaskCompletedEvent
    | MetadataEvent
    | TokenUsageEvent
)


class Transcript(BaseModel):
    """Contains the events that occurred during a run.

    Use `RunState` object and pass the events to it to retrieve the state
    of the run after a certain event."""

    run_id: str
    events: list[SerializeAsAny[Event]] = []

    @property
    def messages(self) -> list[Message]:
        return [
            event.message
            for event in self.events
            if isinstance(event, MessageAddedEvent)
        ]

    @property
    def answers(self) -> dict[str, str]:
        answers_: dict[str, str] = {}
        for event in self.events:
            if isinstance(event, AnswersSubmittedEvent):
                answers_.update(event.answers)
        return answers_
