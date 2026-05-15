import asyncio
import os
import shlex
import signal
from typing import Annotated, Any, Final

from fastmcp.tools.tool import ToolResult


class BashToolResult(ToolResult):
    """Result from a Bash command with stdout, stderr, and additional remarks by the system."""

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        system: str = "",
        restarted: bool = False,
    ):
        self.restarted: bool = restarted
        self.content_dict: dict[str, Any] = {"stdout": stdout}
        if stderr:
            self.content_dict["stderr"] = stderr
        if system:
            self.content_dict["system"] = system
        super().__init__(structured_content=self.content_dict)


class BashSession:
    """A Bash shell session."""

    _command: Final[list[str]] = ["/usr/bin/env", "bash"]
    _output_delay_s: float = 0.1
    _exit_marker: str = "<<exit>>"
    _command_kill_timeout_s: float = 5.0  # Time to wait after process termination

    def __init__(self, max_output_length: int = 10000):
        self._process: asyncio.subprocess.Process | None = None
        self._stdout_buffer: str = ""
        self._stderr_buffer: str = ""
        self.max_output_length: int = max_output_length

    @property
    def _started(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self):
        if self._process and self._process.returncode is None:
            # Already running
            return

        def demote():
            """Demote the process if `PM_DEMOTE_ID` environmnet variable is set."""

            if "PM_DEMOTE_ID" not in os.environ:
                return

            uid_gid = int(os.environ["PM_DEMOTE_ID"])

            os.setsid()
            os.setgid(uid_gid)
            os.setuid(uid_gid)

        self._process = await asyncio.create_subprocess_shell(
            shlex.join(self._command),
            preexec_fn=demote,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._stdout_buffer = ""
        self._stderr_buffer = ""

    async def stop(self):
        if not self._started:
            return

        assert self._process
        pid = self._process.pid

        await self._close_stdin()
        await self._terminate_process(pid)

        try:
            await self._wait_for_process_to_die()
        except TimeoutError:
            await self._kill_process(pid)

        try:
            await self._wait_for_process_to_die()
        except TimeoutError:
            # Process is really stuck. No idea what to do.
            pass

        await self._close_stdout_and_stderr()

        self._process = None

    async def _close_stdin(self):
        assert self._process
        assert self._process.stdin
        try:
            self._process.stdin.close()
            await self._process.stdin.wait_closed()
        except Exception:
            pass

    async def _terminate_process(self, pid: int):
        assert self._process
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                self._process.terminate()
            except (ProcessLookupError, OSError):
                pass

    async def _kill_process(self, pid: int):
        assert self._process
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                self._process.kill()
            except (ProcessLookupError, OSError):
                pass

    async def _wait_for_process_to_die(self):
        assert self._process
        await asyncio.wait_for(
            self._process.wait(), timeout=self._command_kill_timeout_s
        )

    async def _close_stdout_and_stderr(self):
        assert self._process
        assert self._process.stdout
        assert self._process.stderr
        try:
            self._process.stdout.feed_eof()
            self._process.stderr.feed_eof()
        except Exception:
            pass

    async def _ensure_process_alive(self) -> bool:
        """Ensure the process is alive, auto-restart if needed.

        Returns True if the process needed to be restarted."""
        if self._started:
            return False

        await self.start()
        return True

    async def _read_stream_data(self, stream: asyncio.StreamReader) -> bytes:
        """Read available data from a stream without blocking."""
        data = b""
        while True:
            try:
                chunk = await asyncio.wait_for(stream.read(4096), timeout=0.01)
                if not chunk:
                    break
                data += chunk
            except TimeoutError:
                break
        return data

    async def _read_available_output(self):
        """Read any available output from stdout/stderr without blocking."""
        assert self._process
        assert self._process.stdout
        assert self._process.stderr

        try:
            stdout_data = await self._read_stream_data(self._process.stdout)
            if stdout_data:
                self._stdout_buffer += stdout_data.decode("utf-8", errors="replace")
        except Exception:
            pass

        try:
            stderr_data = await self._read_stream_data(self._process.stderr)
            if stderr_data:
                self._stderr_buffer += stderr_data.decode("utf-8", errors="replace")
        except Exception:
            pass

    async def _clear_buffers(self):
        self._stdout_buffer = ""
        self._stderr_buffer = ""

    async def _read_until_exit_marker(self, timeout: float) -> tuple[str, str, bool]:
        assert self._process
        assert self._process.stdout
        assert self._process.stderr

        restarted = False

        try:
            async with asyncio.timeout(timeout):
                while True:
                    await asyncio.sleep(self._output_delay_s)
                    await self._read_available_output()
                    if self._exit_marker in self._stdout_buffer:
                        stdout = self._stdout_buffer[
                            : self._stdout_buffer.index(self._exit_marker)
                        ]
                        break

                stderr = self._stderr_buffer

        except TimeoutError:
            await self.stop()
            await self.start()
            stdout = self._stdout_buffer
            stderr = self._stderr_buffer
            if self._exit_marker in stdout:
                stdout = stdout[: stdout.index(self._exit_marker)]

            if stderr:
                stderr += "\n"

            stderr += f"[Interrupted due to timeout after {timeout}s]"

            restarted = True

        await self._clear_buffers()

        return stdout, stderr, restarted

    async def run(self, command: str, timeout_s: float) -> BashToolResult:
        """Execute a command in the bash shell.

        Args:
            command: The bash command to execute.
            timeout_s: Maximum time to wait for command completion.
        """
        restarted = await self._ensure_process_alive()
        if restarted:
            # Let user know we auto-restarted
            system_msg = "Session was automatically restarted."
        else:
            system_msg = ""

        assert self._process
        assert self._process.stdin

        command = command.rstrip()

        if command.endswith("&"):
            command = f"({command})"

        # Prepare command with exit marker
        full_command = f"{command}\necho '{self._exit_marker}'\n"

        self._process.stdin.write(full_command.encode())
        await self._process.stdin.drain()

        # Read output
        try:
            stdout, stderr, restarted = await self._read_until_exit_marker(timeout_s)

            if restarted:
                system_msg = "Session was automatically restarted."

            if len(stdout) > self.max_output_length:
                stdout = stdout[: self.max_output_length] + "..."
                system_msg += (
                    f"stdout was truncated to {self.max_output_length} characters."
                )

            if len(stderr) > self.max_output_length:
                stderr = stderr[: self.max_output_length] + "..."
                system_msg += (
                    f"stderr was truncated to {self.max_output_length} characters."
                )

            result = BashToolResult(
                stdout=stdout,
                stderr=stderr,
                system=system_msg,
                restarted=restarted,
            )
            return result

        except Exception as e:
            await self.start()
            raise Exception(
                f"Unexpected error in bash session: {e}. Session restarted."
            ) from e


class bash_base:
    """A tool that allows the model to run bash commands."""

    session: BashSession | None
    lock: asyncio.Lock

    def __init__(self, max_output_length: int = 10000):
        self.session = None
        self.lock = asyncio.Lock()
        self.max_output_length: int = max_output_length

    async def __call__(
        self,
        *,
        command: Annotated[
            str | None,
            "The bash command to run. Required unless the tool is being restarted.",
        ] = None,
        restart: Annotated[
            bool,
            "Specifying true will restart this tool. Otherwise, leave this unspecified.",
        ] = False,
        timeout_s: float = 300,
    ) -> ToolResult:
        """THIS DOCSTRING IS SET DYNAMICALLY BELOW.
        f-strings dont work in docstrings, so we need to set it dynamically on module load.
        """

        async with self.lock:
            if restart:
                if self.session:
                    try:
                        await self.session.stop()
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        pass  # Ignore errors during shutdown
                self.session = BashSession(max_output_length=self.max_output_length)
                await self.session.start()
                return BashToolResult(
                    system="Tool has been manually restarted.", restarted=True
                )

            if self.session is None:
                self.session = BashSession(max_output_length=self.max_output_length)
                await self.session.start()

            if command:
                result = await self.session.run(command, timeout_s)

                return result

            raise ValueError("No command provided.")

    async def dispose(self):
        """Dispose of the Bash session."""
        if self.session:
            try:
                await self.session.stop()
            except Exception:
                pass
            finally:
                self.session = None


# We dynamically set the docstring here so that it can access
# configuration variables. f-strings are not allowed in docstrings
#
# This is done because models read tool docstrings to learn
# how they are used.
bash_base.__call__.__doc__ = f"""Execute a Bash `command`.

        The Bash shell session persists between calls.
        Set `restart` to True to restart the session.

        `timeout_s` is the maximum time in seconds to wait for the command to complete.
        The timeout can not be extended beyond the default ({300}s).
        Also note that the maximum output length of a bash tool in characters is {10000}.
        """
