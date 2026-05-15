import importlib
import inspect
import pkgutil
from types import ModuleType

from fastmcp.tools.tool import ToolResult


def discover_tools() -> list[str]:
    """Goes through all modules in the `pm_env.tools` package and returns a list of valid tool names.

    A tool:
        - Is a function or class with the same name as the module it is defined in.
        - Is a function with a return type of `ToolResult`.
        - Is a class with a __call__ method that returns a `ToolResult`.
    """
    import pm_env.tools

    tools: list[str] = []

    for module_info in pkgutil.iter_modules(pm_env.tools.__path__):
        module_name = module_info.name
        try:
            module = importlib.import_module(f"pm_env.tools.{module_name}")
            if _module_contains_tool(module=module, tool_name=module_name):
                tools.append(module_name)

        except ModuleNotFoundError:
            pass

    return tools


def _module_contains_tool(module: ModuleType, tool_name: str) -> bool:
    if not hasattr(module, tool_name):
        return False

    candidate = getattr(module, tool_name)

    if not callable(candidate):
        return False

    if inspect.isclass(candidate):
        candidate = candidate.__call__

    sig = inspect.signature(candidate)

    if not issubclass(sig.return_annotation, ToolResult):
        return False

    untyped = [
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.annotation is inspect.Parameter.empty
    ]
    if untyped:
        raise TypeError(
            f"Tool {tool_name!r} has untyped parameters: {', '.join(untyped)}"
        )

    return True
