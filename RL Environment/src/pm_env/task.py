from dataclasses import dataclass

from pm_env.judges.judge import Judge
from pm_env.schemas.required_hardware import RequiredHardware


@dataclass(kw_only=True)
class Task:
    id: str
    steps: list["Step"]
    tools: list[str] | None
    """Set to None to use all available tools."""

    required_hardware: RequiredHardware = "cpu"
    """Hardware requirement for this task. Defaults to 2 vcpu 8GB cpu instance."""


@dataclass(frozen=True, kw_only=True)
class Step:
    instructions: str
    judge: Judge
