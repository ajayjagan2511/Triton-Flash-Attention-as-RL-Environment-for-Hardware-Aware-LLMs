import os
from typing import Annotated, Final

from fastmcp.tools.tool import ToolResult

from pm_env.tools.bash_base import bash_base


def _on_cluster() -> bool:
    return (
        os.path.exists("/var/run/secrets/kubernetes.io")
        or os.getenv("KUBERNETES_SERVICE_HOST") is not None
    )


def _artifact_cache_url() -> str:
    # First check if we are on tailscale
    if _on_cluster():
        return "http://artifact-cache-pypi-head.default.svc.cluster.local/simple/"
    else:
        return "https://artifact-cache-pypi-clust1.tail645fbe.ts.net/simple/"


class bash:
    def __init__(self, max_output_length: int = 1000):
        self.bash_base: Final = bash_base(max_output_length=max_output_length)

    async def __call__(
        self,
        command: Annotated[
            str | None,
            "The bash command to run. Required unless the tool is being restarted.",
        ] = None,
        restart: Annotated[
            bool,
            "Specifying true will restart this tool. Otherwise, leave this unspecified.",
        ] = False,
    ) -> ToolResult:
        return await self.bash_base.__call__(command=command, restart=restart)

    async def dispose(self):
        """Dispose of the Bash session."""
        await self.bash_base.dispose()


# We dynamically set the docstring here so that it can access
# configuration variables. f-strings are not allowed in docstrings
#
# This is done because models read tool docstrings to learn
# how they are used.
bash.__call__.__doc__ = f"""
Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You don't have access to the internet via this tool.
* Commands that take more than {300} seconds will time out and the tool will need to be restarted.
* Commands that produce a stdout + stderr with more than {1000} characters will be truncated.
* State is persistent across command calls and discussions with the user.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.
"""

if __name__ == "__main__":
    print(bash.__call__.__doc__)
