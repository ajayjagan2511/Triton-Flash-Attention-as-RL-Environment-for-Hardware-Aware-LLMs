import os
from pathlib import Path
from typing import Annotated

import anyio
import rich
import typer

from pm_env.check_permissions import check_scoring_data_permissions
from pm_env.run_helpers import (
    Runtime,
    build_configs,
    build_container,
    clean_up_old_containers,
    parse_config,
    run_containerized,
)
from pm_env.schemas.evaluation_run_config import EvaluationRunConfig
from pm_env.schemas.http_mcp_server_config import HttpMcpServerConfig
from pm_env.tasks import get_tasks

app = typer.Typer(add_completion=False)

_sample_config = EvaluationRunConfig(
    run_id="",
    task_id="",
    model="",
    model_api_key="",
)


@app.command()
def check() -> None:
    """Check that the environment is properly set up."""
    from pm_env.tasks import get_tasks

    tasks = get_tasks(config=_sample_config)

    if len(tasks) == 0:
        rich.print("[bold red]No tasks found in environment.[/bold red]")
        raise typer.Exit(1)

    rich.print(f"[bold blue]Available tasks: {len(tasks)}[/bold blue]")

    from pm_env.mcp_servers.mcp_server import McpServer

    mcp_server = McpServer("")
    mcp_server.register_tools(None)

    rich.print(
        f"[bold blue]Available MCP tools:\n{'\n'.join(f'  - {t}' for t in sorted(mcp_server.registered_tools))}[/bold blue]"
    )

    if os.environ.get("PM_CONTAINERIZED"):
        check_scoring_data_permissions()
        rich.print("[bold blue]Scoring data permissions checked.[/bold blue]")


@app.command()
def create_run_config(
    config_path: Annotated[
        str,
        typer.Argument(help="Path where the config file should be created."),
    ] = "run_config.json",
    model: Annotated[
        str, typer.Option(help="Model name.")
    ] = "claude-haiku-4-5-20251001",
    model_api_key: Annotated[
        str | None,
        typer.Option(
            help="Model API key. Falls back to `ANTHROPIC_API_KEY` environment variable. Leaves the `model_api_key` field empty if that's also not provided.",
        ),
    ] = None,
) -> None:
    """Create a default evaluation run configuration."""
    from pm_env.schemas.evaluation_run_config import EvaluationRunConfig

    model_api_key = model_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    task_id = get_tasks(config=_sample_config)[0].id

    config = EvaluationRunConfig(
        run_id="",
        task_id=task_id,
        model=model,
        model_api_key=model_api_key,
        mcp_server_config=HttpMcpServerConfig(),
        transcript_file="out/transcript.json",
    )

    config_path_ = Path(config_path)
    config_path_.write_text(config.model_dump_json(indent=2))

    rich.print(f"[bold blue]Run config written to {config_path_}[/bold blue]")


@app.command()
def list_tasks(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output tasks as JSON"),
    ] = False,
) -> None:
    """List all available tasks."""
    import json as json_lib

    from pm_env.tasks import get_tasks

    tasks = get_tasks(config=_sample_config)

    if json_output:
        task_list = [
            {
                "id": task.id,
                "tools": task.tools if task.tools is not None else [],
            }
            for task in tasks
        ]
        # We are using normal print here because rich.print can introduce newlines which breaks json parsing
        print(json_lib.dumps(task_list, indent=2))
    else:
        rich.print("Available tasks:")
        for task in tasks:
            rich.print(f"  - {task.id!r}")


@app.command()
def run(
    config: Annotated[
        str,
        typer.Option(
            "--config",
            "-c",
            help="JSON-serialized `EvaluationRunConfig` or path to a file containing one.",
        ),
    ],
    containerized: Annotated[
        bool,
        typer.Option(
            help="Whether to execute the run inside a podman/docker container."
        ),
    ] = True,
    runtime: Annotated[
        str,
        typer.Option(help="Runtime to use for containerization (podman or docker)."),
    ] = "podman",
    build_context: Annotated[
        str, typer.Option(help="Path to the build context for the container.")
    ] = ".",
    n_parallel: Annotated[
        int,
        typer.Option(
            "--n-parallel",
            "-n",
            help="Number of parallel runs. Can only be >1 when running containerized.",
        ),
    ] = 1,
    dev: Annotated[
        bool,
        typer.Option(
            help="Skips container building and directly mounts `src/pm_env` into the container."
            + "This allows iterating quickly on an environment. "
            + "Can only be used when running containerized. "
            + "Still requires rebuilding the container if dependencies or env/scoring data change."
        ),
    ] = False,
    keep_containers: Annotated[
        bool,
        typer.Option(
            help="Persist containers after runs instead of removing them. "
            + "Containers are named pm_env_run_<run_id>. "
            + "Existing containers with the same names get removed before the runs start. "
            + "After runs finish, you can copy the data from a container to your host machine. "
            + "Example: `podman cp pm_env_run_<run_id>:/workdir/ ./out/`",
        ),
    ] = False,
) -> None:
    """Execute an evaluation run."""
    from typing import cast

    from pm_env.mcp_servers.http_mcp_server import run_server as run_mcp_server

    if n_parallel > 1 and not containerized:
        _print_and_abort("Cannot run multiple runs without containerization.")

    if dev and not containerized:
        _print_and_abort("Cannot use the `--dev` option without containerization.")

    if keep_containers and not containerized:
        _print_and_abort(
            "Cannot use the `--keep-containers` option without containerization."
        )

    # If we are already inside a container or should run uncontainerized, don't launch the TUI.
    # Just execute the run and stream to websocket
    if "PM_CONTAINERIZED" in os.environ or containerized is False:
        run_config = parse_config(config)
        with run_mcp_server(run_config.mcp_server_config, run_config.task_id):
            from pm_env.run_helpers import run_non_containerized

            anyio.run(run_non_containerized, run_config)
        return

    run_configs = build_configs(parse_config(config), n_parallel)
    runtime_ = cast(Runtime, runtime)

    # Run without UI - build container and run with stdout output
    _run_without_ui(run_configs, runtime_, dev, build_context, keep_containers)
    return


def _run_without_ui(
    run_configs: list[EvaluationRunConfig],
    runtime: Runtime,
    dev: bool,
    build_context: str,
    keep_containers: bool,
) -> None:
    """Run containerized evaluations without the TUI, outputting directly to stdout."""
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    # Build container if needed
    if not dev:
        build_container(runtime, build_context)

    clean_up_old_containers(runtime)

    # Log container names for easy reference
    typer.secho("\nStarting containers:", fg=typer.colors.BLUE)
    for config in run_configs:
        typer.secho(f"  pm_env_run_{config.run_id}", fg=typer.colors.BLUE)
    typer.echo()

    # Run containers in parallel using threads
    run_fn = partial(
        run_containerized,
        runtime=runtime,
        dev=dev,
        log_file=None,
        keep_container=keep_containers,
    )

    with ThreadPoolExecutor(max_workers=len(run_configs)) as executor:
        list(executor.map(run_fn, run_configs))

    # Log completion message with copy example if containers are preserved
    if keep_containers:
        typer.secho(
            f"\nContainers preserved. To copy data:\n  {runtime} cp pm_env_run_{run_configs[0].run_id}:/workdir/ ./out/",
            fg=typer.colors.BLUE,
        )


def _print_and_abort(message: str) -> None:
    typer.secho(
        message,
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Abort()


if __name__ == "__main__":
    app()
