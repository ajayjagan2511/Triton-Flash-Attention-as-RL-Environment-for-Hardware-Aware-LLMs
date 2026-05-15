from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_SCRIPT = REPO_ROOT / "src" / "pm_env" / "score_flash_attention.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sft_harness_out" / "rollouts"
DEFAULT_SUCCESS_JSONL = DEFAULT_OUTPUT_DIR / "flash_attention_success.jsonl"
DEFAULT_NEGATIVE_JSONL = DEFAULT_OUTPUT_DIR / "flash_attention_negative.jsonl"

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

DEFAULT_PROMPT_TEMPLATE = """Write a complete triton_attention.py implementing Flash Attention v2.

Constraints:
- Input tensors q, k, v are CUDA float16 with shape [batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}].
- Implement class FlashAttention(torch.autograd.Function) with forward(q, k, v) and backward(do) -> (dq, dk, dv).
- Use a Triton kernel (must include @triton.jit or tl.dot). Do not use torch.nn.MultiheadAttention or scaled_dot_product_attention.
- Assume a memory budget around {memory_budget_mb} MB and favor SRAM tiling.

Return ONLY the full python file content for triton_attention.py.
"""

DEFAULT_SYSTEM_PROMPT = ""


@dataclass(frozen=True)
# class to encapsulate all configuration parameters for the rollout harness
class HarnessConfig:
    model: str
    api_key: str | None
    max_turns: int
    temperature: float
    success_threshold: float
    max_tokens: int
    num_rollouts: int
    score_script: Path
    output_dir: Path
    success_jsonl: Path
    negative_jsonl: Path | None
    save_negative: bool
    prompt_file: Path | None
    system_prompt: str
    seed: int | None
    judge_timeout: int


def parse_args() -> HarnessConfig:
    parser = argparse.ArgumentParser(
        description="Generate self-correction rollouts for Flash Attention SFT."
    )
    parser.add_argument("--model", required=True, help="LiteLLM model name.")
    parser.add_argument("--api-key", default=None, help="API key for the model.")
    parser.add_argument(
        "--api-key-env",
        default="MODEL_API_KEY",
        help="Environment variable to read the API key from.",
    )
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--success-threshold", type=float, default=3.0) # Target Speedup
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--num-rollouts", type=int, default=2)
    parser.add_argument("--score-script", type=Path, default=DEFAULT_SCORE_SCRIPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--success-jsonl", type=Path, default=DEFAULT_SUCCESS_JSONL)
    parser.add_argument("--negative-jsonl", type=Path, default=DEFAULT_NEGATIVE_JSONL)
    parser.add_argument(
        "--save-negative",
        action="store_true",
        help="Save failed rollouts to negative_jsonl.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional prompt template file (uses {batch_size}, {seq_len}).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Optional system prompt for the model.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--judge-timeout", type=int, default=600)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get(args.api_key_env)

    return HarnessConfig(
        model=args.model,
        api_key=api_key,
        max_turns=args.max_turns,
        temperature=args.temperature,
        success_threshold=args.success_threshold,
        max_tokens=args.max_tokens,
        num_rollouts=args.num_rollouts,
        score_script=args.score_script,
        output_dir=args.output_dir,
        success_jsonl=args.success_jsonl,
        negative_jsonl=args.negative_jsonl,
        save_negative=args.save_negative,
        prompt_file=args.prompt_file,
        system_prompt=args.system_prompt,
        seed=args.seed,
        judge_timeout=args.judge_timeout,
    )


def load_prompt(template_path: Path | None) -> str:
    """
    Load the prompt template from a file, or return the default if no file is provided.
    """
    if template_path is None:
        return DEFAULT_PROMPT_TEMPLATE
    return template_path.read_text()


def build_prompt(template: str, rng: random.Random) -> tuple[str, dict[str, int]]:
    """
    Build the prompt by filling in the constraints with random values.
    """
    constraints = {
        "batch_size": rng.choice([1, 2, 4]),
        "seq_len": rng.choice([512, 768, 1024, 1536, 2048]),
        "num_heads": 4,
        "head_dim": 64,
        "memory_budget_mb": rng.choice([64, 96, 128]),
    }
    return template.format(**constraints), constraints


def call_model(config: HarnessConfig, messages: list[dict[str, str]]) -> str:
    """
    Calls the LiteLLM API with exponential backoff for network/rate-limit resilience.
    """
    max_retries = 3
    base_delay = 5

    for attempt in range(max_retries):
        try:
            # Get response from model
            response = litellm.completion(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key,
            )

            # Extraction of the content
            choice = response.choices[0]
            message = getattr(choice, "message", None)

            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)

            if not content:
                raise RuntimeError("Model response missing content")

            return content

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Fatal API Error after {max_retries} attempts: {e}")
                raise e

            # Exponential backoff: 5s, then 10s
            delay = base_delay * (2**attempt)
            print(
                f"API Error (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)


def extract_code(text: str) -> str:
    """Extract the last Python code block from the model's response."""
    matches = CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def _has_unclosed_code_block(text: str) -> bool:
    return "```" in text and CODE_BLOCK_RE.search(text) is None


def wrap_code_block(code: str) -> str:
    return f"```python\n{code}\n```"


def run_judge(
    config: HarnessConfig, run_dir: Path, code: str, turn: int
) -> dict[str, Any]:
    """
    Run the judge script on the provided code and return the parsed result.
    """
    target_path = run_dir / f"turn_{turn:02d}.py"
    output_path = run_dir / f"result_{turn:02d}.json"
    target_path.write_text(code)

    # Run the judge script as a subprocess, passing the target code path and expected output path.
    # Subprocess because the judge may involve GPU execution and we want to isolate it from the main process, as well as capture stdout/stderr for debugging.
    result = subprocess.run(
        [sys.executable, str(config.score_script), str(target_path), str(output_path)],
        capture_output=True,
        text=True,
        timeout=config.judge_timeout,
    )

    (run_dir / f"judge_{turn:02d}.stdout").write_text(result.stdout)
    (run_dir / f"judge_{turn:02d}.stderr").write_text(result.stderr)

    raw_error = result.stderr.strip() or result.stdout.strip()
    # Keep the last 3000 characters to prevent context bloat
    truncated_error = raw_error[-3000:] if len(raw_error) > 3000 else raw_error

    if result.returncode != 0:
        return {
            "score": 0.0,
            "metadata": {
                "error": "judge_failed",
                "detail": truncated_error,
            },
        }
    if not output_path.exists():
        return {
            "score": 0.0,
            "metadata": {"error": "missing_output", "detail": str(output_path)},
        }
    try:
        return json.loads(output_path.read_text())
    except json.JSONDecodeError as exc:
        return {
            "score": 0.0,
            "metadata": {"error": "invalid_output", "detail": str(exc)},
        }


def format_execution_result(
    score: float, metadata: dict[str, Any], success_threshold: float
) -> str:
    """
    Format the execution result to provide feedback to the model for the next turn.
    This includes the score, any relevant metadata, and guidance on next steps based on whether the score meets the success threshold or if there were errors.
    """
    lines = ["<execution_result>"]
    if score >= success_threshold:
        lines.append(f"Score: {score:.3f}. Speedup achieved.")
    elif score == 0.0:
        error = (
            metadata.get("traceback")
            or metadata.get("detail")
            or metadata.get("error")
            or "Unknown error"
        )
        lines.append(f"Error: {error}")
    else:
        lines.append(f"Score: {score:.3f}")
        if "speedup" in metadata:
            lines.append(f"Speedup: {metadata['speedup']}")
        if "agent_ms" in metadata:
            lines.append(f"Agent ms: {metadata['agent_ms']}")
        if "baseline_ms" in metadata:
            lines.append(f"Baseline ms: {metadata['baseline_ms']}")
        if "mem_factor" in metadata:
            lines.append(f"Mem factor: {metadata['mem_factor']}")
        lines.append(
            f"Target is > {success_threshold:.1f}. Re-evaluate SRAM tiling and block sizes."
        )
    lines.append("</execution_result>")

    if score == 0.0:
        lines.append("Analyze the error and generate a corrected kernel.")
    elif score < success_threshold:
        lines.append(
            "Mathematical constraints met, but performance is sub-optimal. Improve the kernel."
        )

    return "\n".join(lines)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def make_run_dir(base_dir: Path, rollout_idx: int) -> Path:
    """
    Create a unique directory (with timestamp) for this rollout to store code, judge outputs, and results.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"{timestamp}-{rollout_idx:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_rollout(config: HarnessConfig, rollout_idx: int) -> bool:
    """
    Run a single rollout of the self-correction loop.
    """
    rng = random.Random(None if config.seed is None else config.seed + rollout_idx)
    template = load_prompt(config.prompt_file)
    prompt, constraints = build_prompt(template, rng)

    run_dir = make_run_dir(config.output_dir, rollout_idx)
    # The JSONL records are stored for each rollout, but we also keep a summary in the main success/negative JSONL files for easy aggregation and analysis.
    messages: list[dict[str, str]] = []
    # We include the full prompt in the first user message to ensure the model has all the context it needs, and then we append feedback and code iteratively in subsequent turns.
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})

    final_score = 0.0
    final_metadata: dict[str, Any] = {}

    for turn in range(1, config.max_turns + 1):
        response_text = call_model(config, messages)
        # We append the full response (including any explanatory text) to the messages to maintain the conversation history, but we only extract and run the code block for judging.
        messages.append({"role": "assistant", "content": response_text})

        if not CODE_BLOCK_RE.search(response_text):
            if _has_unclosed_code_block(response_text):
                messages.append(
                    {
                        "role": "user",
                        "content": "Output looks truncated (opened a code block but did not close it). Please resend the FULL triton_attention.py inside a single ```python``` block.",
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Formatting Error: No ```python code block found. Please output ONLY the python code inside markdown backticks.",
                    }
                )
            # Skip judge execution and prompt the model to fix the formatting in the next turn.
            continue

        code = extract_code(response_text)

        result = run_judge(config, run_dir, code, turn)
        final_score = float(result.get("score", 0.0))
        final_metadata = result.get("metadata", {})

        if final_metadata.get("error") == "cuda_unavailable":
            raise SystemExit("CUDA unavailable. Run rollouts on a GPU host.")

        feedback = format_execution_result(
            final_score, final_metadata, config.success_threshold
        )
        # feedback to model for next turn.
        messages.append({"role": "user", "content": feedback})

        # early exit if successful to avoid unnecessary iterations and to save compute resources.
        if final_score >= config.success_threshold:
            record = {
                "messages": messages,
                "metadata": {
                    "success": True,
                    "turns": turn,
                    "score": final_score,
                    "constraints": constraints,
                    "judge": final_metadata,
                },
            }
            append_jsonl(config.success_jsonl, record)
            return True

    # save a negative example if we exhaust all turns without success, which can be useful for analysis and future training.
    if config.save_negative and config.negative_jsonl is not None:
        record = {
            "messages": messages,
            "metadata": {
                "success": False,
                "turns": config.max_turns,
                "score": final_score,
                "constraints": constraints,
                "judge": final_metadata,
            },
        }
        append_jsonl(config.negative_jsonl, record)

    return False


def main() -> None:
    config = parse_args()

    # Judging metrics
    if not config.score_script.exists():
        raise SystemExit(f"Score script not found: {config.score_script}")

    success_count = 0
    for rollout_idx in range(config.num_rollouts):
        success = run_rollout(config, rollout_idx)
        if success:
            success_count += 1

    print(f"Completed {config.num_rollouts} rollouts with {success_count} successes")


if __name__ == "__main__":
    main()
