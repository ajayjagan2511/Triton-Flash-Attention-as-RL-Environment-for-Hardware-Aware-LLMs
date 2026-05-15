"""Shared helpers for running containerized evaluations."""

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Literal

import anyio
import pydantic
import typer

from pm_env.load_task import load_task
from pm_env.schemas.evaluation_run_config import EvaluationRunConfig
from pm_env.transcript_streaming.stream_transcript_to_stdout import (
    stream_transcript_to_stdout,
)
from pm_env.transcript_streaming.stream_transcript_to_websocket import (
    stream_transcript_to_websocket,
)

Runtime = Literal["podman", "docker"]


def parse_config(config: str) -> EvaluationRunConfig:
    """Parse an EvaluationRunConfig from a JSON string or file path."""
    try:
        return EvaluationRunConfig.model_validate_json(config)
    except pydantic.ValidationError:
        pass

    if not Path(config).is_file():
        _print_and_abort(
            "Failed to load evaluation run configuration.  "
            + f"{config!r} is not a valid configuration and does not point to an existing file."
        )

    try:
        return EvaluationRunConfig.model_validate_json(Path(config).read_text())
    except pydantic.ValidationError:
        _print_and_abort(
            "Failed to load evaluation run configuration. "
            + f"{config!r} does not contain a valid configuration."
        )

    raise AssertionError("unreachable")


def build_configs(run_config: EvaluationRunConfig, n: int) -> list[EvaluationRunConfig]:
    """Create n variations of a run configuration for parallel runs."""
    if n == 1:
        return [run_config]

    # Create n variations of the run configuration.
    # If the original websocket port is x, the first
    # run will use x, the second one x+1, and so on.
    return [
        run_config.model_copy(
            update={
                "run_id": f"{run_config.run_id}-{i}",
                "websocket_config": run_config.websocket_config.model_copy(
                    update={
                        "port": run_config.websocket_config.port + i,
                    }
                ),
                "transcript_file": (
                    f"{run_config.transcript_file.rstrip('.json')}_{i}.json"
                    if run_config.transcript_file
                    else None
                ),
            }
        )
        for i in range(n)
    ]


def build_container(runtime: Runtime, build_context: str) -> None:
    """Build the container image."""
    build_command = get_container_build_command(runtime, build_context)

    typer.secho(
        f"Building container image with command: {shlex.join(build_command)!r}",
        fg=typer.colors.BLUE,
    )

    build_result = subprocess.run(build_command, check=False)

    if build_result.returncode != 0:
        _print_and_abort(
            "Failed to build environment container image. "
            + "Make sure to pass the correct build context via the '--build-context' option."
        )

    typer.secho("Environment container image built successfully.", fg=typer.colors.BLUE)


def get_container_build_command(runtime: Runtime, build_context: str) -> list[str]:
    """Build the command to construct a container image."""
    command: list[str] = []

    # Our current CI setup requires sudo
    if os.environ.get("CI"):
        command.append("sudo")

    command.extend([runtime, "build"])

    if runtime == "docker":
        # Docker defaults to `Dockerfile`
        command.extend(["--file", "Containerfile"])

    command.extend(["--tag", "pm_env", build_context])
    return command


def run_containerized(
    run_config: EvaluationRunConfig,
    runtime: Runtime,
    dev: bool,
    log_file: Path | None = None,
    keep_container: bool = False,
) -> None:
    """Run a single containerized evaluation.

    Args:
        run_config: The evaluation run configuration.
        runtime: Container runtime to use (podman or docker).
        dev: Whether to mount source code for development.
        log_file: If provided, redirect output to this file. Otherwise output to stdout.
        keep_container: If True, keep the container after run completes.
    """
    run_command, _ = get_container_run_command(run_config, runtime, dev, keep_container)

    # Only attach TTY when outputting to stdout (not when redirecting to log file)
    # This avoids TTY issues when running in the TUI with multiprocessing
    # Insert TTY flags right after "run" subcommand
    if log_file is None and sys.stdin.isatty():
        run_index = run_command.index("run") + 1
        run_command.insert(run_index, "--interactive")
        run_command.insert(run_index, "--tty")

    if log_file:
        with open(log_file, "w") as f:
            subprocess.run(run_command, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(run_command, check=True)


def get_container_run_command(
    run_config: EvaluationRunConfig,
    runtime: Runtime,
    dev: bool,
    keep_container: bool,
) -> tuple[list[str], EvaluationRunConfig]:
    """Build the command to run a container.

    Returns:
        A tuple of (command, updated_config). The config may be modified
        if transcript_file path needs to be transformed for container mount.
    """
    task = load_task(run_config)

    command: list[str] = []

    # Our current CI setup requires sudo
    if os.environ.get("CI"):
        command.append("sudo")

    command.extend(
        [
            runtime,
            "run",
            "--cap-add=NET_ADMIN",
            "--name",
            f"pm_env_run_{run_config.run_id}",
            "--publish",
            f"{run_config.websocket_config.port}:{run_config.websocket_config.port}",
        ]
    )

    if not keep_container:
        command.append("--rm")

    if dev:
        command.extend(
            [
                "--volume",
                f"{Path(__file__).parent.absolute()}:{'/pm_env/.venv/lib/python3.12/site-packages/pm_env/'}",
            ]
        )

    # Mount NVIDIA GPU devices using CDI (Container Device Interface)
    # Only mount if task requires GPU hardware
    if task.required_hardware.startswith("h100-") and Path("/dev/nvidia0").exists():
        command.extend(["--device", "nvidia.com/gpu=all"])

    # Mount TPU devices which are exposed as /dev/vfio/[0-9]
    # Only mount if task requires TPU hardware
    if task.required_hardware.startswith("tpu-"):
        if Path("/dev/vfio/vfio").exists():
            command.extend(["--device=/dev/vfio/vfio:/dev/vfio/vfio"])

        vfio_path = Path("/dev/vfio")
        if vfio_path.exists():
            for device in sorted(vfio_path.glob("[0-9]*")):
                command.extend([f"--device={device}:{device}"])

    if run_config.transcript_file:
        file_path = Path(run_config.transcript_file).absolute()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(
            [
                "--mount",
                f"type=bind,source={file_path.parent},target=/out",
            ]
        )
        run_config = run_config.model_copy(
            update={"transcript_file": f"/out/{file_path.name}"}
        )

    command.append("localhost/pm_env" if runtime == "podman" else "pm_env")

    command.extend(
        [
            "/pm_env/.venv/bin/pm_env",
            "run",
            "--no-containerized",
            "--config",
            run_config.model_dump_json(),
        ]
    )

    return command, run_config


async def run_non_containerized(run_config: EvaluationRunConfig):
    from pm_env.evaluation_runner import EvaluationRunner
    from pm_env.schemas.transcript import Event as TaskEvent

    # _maybe_block_internet()

    runner = EvaluationRunner(run_config)

    async with anyio.create_task_group() as tg:
        stdout_send, stdout_recv = anyio.create_memory_object_stream[TaskEvent]()
        websocket_send, websocket_recv = anyio.create_memory_object_stream[TaskEvent]()

        tg.start_soon(
            stream_transcript_to_websocket, run_config.websocket_config, websocket_recv
        )
        tg.start_soon(stream_transcript_to_stdout, stdout_recv)

        # Stream events to all outputs
        async for event in runner.run():
            await stdout_send.send(event)
            await websocket_send.send(event)

        await stdout_send.aclose()
        await websocket_send.aclose()


def clean_up_old_containers(runtime: Runtime) -> None:
    """Stop and remove old containers with pm_env_run_* names.

    Args:
        runtime: Container runtime to use (podman or docker).
    """
    from loguru import logger

    result = subprocess.run(
        [runtime, "ps", "-a", "--filter", "name=pm_env_run_", "-q"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.warning(
            f"Failed to list containers: {result.stderr.strip() or 'unknown error'}"
        )
        return

    container_ids = result.stdout.strip().split()

    if container_ids:
        rm_result = subprocess.run(
            [runtime, "rm", "--force", *container_ids], capture_output=True, text=True
        )
        if rm_result.returncode != 0:
            logger.warning(
                f"Failed to remove containers: {rm_result.stderr.strip() or 'unknown error'}"
            )


def _maybe_block_internet():
    if "PM_CONTAINERIZED" in os.environ:
        # Setup firewall so model user can't access external internet.
        # Allow access to metadata server and internal networks for TPU/GCP services.
        # Can't do this in dockerfile because network capabilities are not
        # enabled during container image creation.

        # IPv4 rules: allow localhost, metadata servers, private networks, and Tailscale
        firewall_rules = [
            # Allow localhost
            "iptables -A OUTPUT -m owner --uid-owner model -d 127.0.0.0/8 -j ACCEPT",
            # Allow GCP/AWS metadata server (both use 169.254.169.254)
            "iptables -A OUTPUT -m owner --uid-owner model -d 169.254.169.254 -j ACCEPT",
            # Allow link-local addresses (169.254.0.0/16) for metadata and other services
            "iptables -A OUTPUT -m owner --uid-owner model -d 169.254.0.0/16 -j ACCEPT",
            # Allow private networks (RFC 1918)
            "iptables -A OUTPUT -m owner --uid-owner model -d 10.0.0.0/8 -j ACCEPT",
            "iptables -A OUTPUT -m owner --uid-owner model -d 172.16.0.0/12 -j ACCEPT",
            "iptables -A OUTPUT -m owner --uid-owner model -d 192.168.0.0/16 -j ACCEPT",
            # Allow Tailscale network (100.64.0.0/10) for artifact cache access
            "iptables -A OUTPUT -m owner --uid-owner model -d 100.64.0.0/10 -j ACCEPT",
            # Reject everything else
            "iptables -A OUTPUT -m owner --uid-owner model -j REJECT",
        ]

        # IPv6 rules: allow localhost, link-local, and unique local addresses
        firewall_rules_v6 = [
            # Allow localhost
            "ip6tables -A OUTPUT -m owner --uid-owner model -d ::1 -j ACCEPT",
            # Allow AWS IPv6 metadata server (IMDSv2)
            "ip6tables -A OUTPUT -m owner --uid-owner model -d fd00:ec2::254 -j ACCEPT",
            # Allow link-local (fe80::/10)
            "ip6tables -A OUTPUT -m owner --uid-owner model -d fe80::/10 -j ACCEPT",
            # Allow unique local (fd00::/8) - includes AWS metadata
            "ip6tables -A OUTPUT -m owner --uid-owner model -d fd00::/8 -j ACCEPT",
            # Reject everything else
            "ip6tables -A OUTPUT -m owner --uid-owner model -j REJECT",
        ]

        # Apply all firewall rules
        for rule in firewall_rules + firewall_rules_v6:
            result = subprocess.run(
                rule,
                capture_output=True,
                shell=True,
                text=True,
            )
            assert (
                not result.stderr
            ), f"failed to apply firewall rule '{rule}': {result.stderr}"


def _print_and_abort(message: str) -> None:
    typer.secho(
        message,
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Abort()
