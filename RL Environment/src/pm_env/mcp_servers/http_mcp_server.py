import os
import socket
import subprocess
import sys
import time
from contextlib import closing, contextmanager
from typing import Final, override

from pm_env.mcp_servers.mcp_server import McpServer
from pm_env.schemas.http_mcp_server_config import HttpMcpServerConfig


class HttpMcpServer(McpServer):
    def __init__(self, config: HttpMcpServerConfig):
        super().__init__("PM MCP Server")
        self.config: Final = config

    @override
    def run(self):
        self.server.run(
            transport="streamable-http",
            host=self.config.host,
            port=self.config.port,
            uvicorn_config={"timeout_graceful_shutdown": None},
        )


@contextmanager
def run_server(
    config: HttpMcpServerConfig, task_id: str, suppress_output: bool = False
):
    server = HttpMcpServer(config)

    # Create a subprocess to run the server
    cmd = [
        sys.executable,
        "-m",
        "pm_env.mcp_servers.http_mcp_server",
        config.host,
        str(config.port),
        str(suppress_output),
    ]

    stdout = subprocess.DEVNULL if suppress_output else None
    stderr = subprocess.DEVNULL if suppress_output else None

    process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)

    start_time = time.time()
    timeout = 30  # Increased from 10 to 30 seconds

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(
                f"MCP server failed to start within {timeout} seconds. "
                + f"Check if port {config.port} is already in use or if there are startup issues."
            )

        # Check if process has died
        if process.poll() is not None:
            raise RuntimeError(
                f"MCP server process exited unexpectedly with code {process.returncode}"
            )

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(1.0)  # Set socket timeout to avoid hanging
            result = s.connect_ex((config.host, config.port))
            if result == 0:
                # Port is open, give it a bit more time to fully initialize
                time.sleep(0.5)
                break

        time.sleep(0.2)  # Increased from 0.1 for less aggressive polling

    try:
        yield server
    finally:
        process.terminate()
        process.wait(timeout=10)


def _subprocess_entrypoint():
    """Entrypoint for subprocess to run the MCP server."""
    import sys

    # Read config from command line arguments
    host = sys.argv[1]
    port = int(sys.argv[2])
    suppress_output = sys.argv[3] == "True"

    config = HttpMcpServerConfig(host=host, port=port)
    server = HttpMcpServer(config)

    if suppress_output:
        _run_server_silently(server)
    else:
        server.run()


def _run_server_silently(server: HttpMcpServer):
    """Run an MCP server with suppressed output.

    Used for example in `pm_dev check` to not clutter the output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            server.run()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    _subprocess_entrypoint()
