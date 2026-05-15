import importlib
import inspect
from typing import Final

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult

from pm_env.mcp_servers.discover_tools import discover_tools


class McpServer:
    def __init__(self, name: str):
        self.server: Final = FastMCP(name)
        self.registered_tools: set[str] = set()

        # Cannot directly decorate the method.
        # See https://gofastmcp.com/patterns/decorating-methods
        self.server.tool()(self.register_tools)

    def register_tools(self, tools: list[str] | None = None):
        """Registers `tools` with `server`.

        Can only be called once.

        Args:
            tools: A list of tool names to register. If None, all available tools will be registered.
        """
        self._register_tools(tools)
        self.server.remove_tool("register_tools")

    def run(self) -> None:
        raise NotImplementedError("Sub-classes must implement this")

    def _register_tools(self, tools: list[str] | None):
        if tools is None:
            tools_to_register = discover_tools()
        else:
            tools_to_register = tools

        for tool in tools_to_register:
            try:
                module = importlib.import_module(f"pm_env.tools.{tool}")
            except ModuleNotFoundError:
                raise RuntimeError(f"Tool {tool!r} not found")

            candidate = getattr(module, tool)

            if inspect.isclass(candidate):
                # Only instance methods can be registered as tools
                # https://gofastmcp.com/patterns/decorating-methods
                candidate = candidate()
                candidate = getattr(candidate, "__call__")

            function = inspect.signature(candidate)

            if not issubclass(function.return_annotation, ToolResult):
                raise RuntimeError(f"Tool {tool!r} does not return ToolResult")

            self.server.tool(candidate, name=tool)
            self.registered_tools.add(tool)
