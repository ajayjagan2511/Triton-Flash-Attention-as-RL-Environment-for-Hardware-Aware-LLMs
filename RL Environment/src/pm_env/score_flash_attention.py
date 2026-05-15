import importlib.util
import json
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

# For reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _write_result(output_path: str, score: float, metadata: dict[str, Any]) -> None:
    Path(output_path).write_text(
        json.dumps({"score": float(score), "metadata": metadata})
    )


def _import_flash_attention(target_path: str) -> type[torch.autograd.Function]:
    spec = importlib.util.spec_from_file_location("triton_attention", target_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {target_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["triton_attention"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "FlashAttention"):
        raise AttributeError("FlashAttention not found in target module")
    return module.FlashAttention


def _make_identity_mha(
    embed_dim: int, num_heads: int, device: torch.device
) -> torch.nn.MultiheadAttention:
    """
    Create a MultiheadAttention module with weights initialized to identity matrices.
    This allows us to focus on the attention computation, we dont want to worry about the full MultiheadAttention here.
    """
    mha = torch.nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        bias=False,
        device=device,
        dtype=torch.float16,
    )
    mha.eval()
    with torch.no_grad():
        eye = torch.eye(embed_dim, device=device, dtype=torch.float16)
        # Set in_proj_weight to have identity matrices for q, k, v projections
        mha.in_proj_weight.copy_(torch.cat([eye, eye, eye], dim=0))
        # Set out_proj to identity as well.
        mha.out_proj.weight.copy_(eye)
    return mha


def _time_ms(fn: Callable[[], None], iters: int = 100, warmup: int = 10) -> float:
    """
    Time a function in milliseconds, with CUDA synchronization. Runs a few warmup iterations before timing.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _peak_memory_bytes(fn: Callable[[], None]) -> int:
    """
    Measure peak memory usage of a function in bytes, with CUDA synchronization.
    First, resets peak memory stats before running.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def _naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / (q.size(-1) ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def _static_check_source(target_path: str) -> tuple[bool, dict[str, Any]]:
    """
    Perform static checks on the source code to enforce Triton usage and reject forbidden APIs.
    Checks for:
    - Presence of `triton` import and usage of `@triton.jit` or `tl.dot`.
    - Absence of `scaled_dot_product_attention` or `torch.nn.MultiheadAttention`.
    """
    try:
        source = Path(target_path).read_text()
    except Exception as exc:
        return False, {"error": "source_read_failed", "detail": str(exc)}

    has_triton_import = re.search(
        r"^\s*(import\s+triton\b|from\s+triton\b)", source, re.M
    )
    has_jit = re.search(r"@\s*triton\.jit\b", source)
    has_tl_dot = re.search(r"\btl\.dot\b", source)

    if not has_triton_import:
        return False, {"error": "triton_missing"}
    if not (has_jit or has_tl_dot):
        return False, {"error": "triton_kernel_missing"}

    if re.search(r"\bscaled_dot_product_attention\b", source):
        return False, {
            "error": "forbidden_api",
            "detail": "scaled_dot_product_attention",
        }

    if re.search(r"\btorch\.nn\.MultiheadAttention\b", source) or re.search(
        r"\bnn\.MultiheadAttention\b", source
    ):
        return False, {"error": "forbidden_api", "detail": "MultiheadAttention"}

    return True, {}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: score_flash_attention.py <target_path> <output_path>")

    # Get the target file path and output path from command line arguments
    target_path = sys.argv[1]
    output_path = sys.argv[2]
    # Initialize score and metadata
    score = 0.0
    metadata: dict[str, Any] = {"stages": {}}

    static_ok, static_meta = _static_check_source(target_path)
    if not static_ok:
        # Static checks failed, write score as 0.0 and exit early
        _write_result(output_path, 0.0, static_meta)
        return
    # Log static check success and continue with dynamic checks
    metadata["stages"]["static"] = True

    try:
        # Dynamically import the FlashAttention class from the target file
        FlashAttention = _import_flash_attention(target_path)
    except Exception as exc:
        _write_result(
            output_path,
            0.0,
            {
                "error": "import_failed",
                "detail": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return
    # We can assign a small score for successful import before proceeding to runtime checks.
    score += 0.1
    metadata["stages"]["import"] = True

    # Check for CUDA availability
    if not torch.cuda.is_available():
        _write_result(output_path, 0.0, {"error": "cuda_unavailable"})
        return

    # Set up standard dimensions and random inputs for testing
    device = torch.device("cuda")
    batch_size = 2
    num_heads = 4
    head_dim = 64
    embed_dim = num_heads * head_dim
    # Have non-power-of-2 sequence lengths to encourage more general solutions and avoid optimizations that only work for fixed shapes.
    seq_len = random.choice([512, 768, 1024, 1536, 2048])
    metadata["seq_len"] = seq_len

    try:
        # Create a reference MultiheadAttention with identity weights to serve as a baseline for correctness and timing.
        mha = _make_identity_mha(embed_dim, num_heads, device)
        q = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        k = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        v = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        # Gradient of output for backward pass
        do = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.float16,
        )

        # Clone to ensure original tensors are not modified during testing, and set requires_grad for autograd checks.
        q_base = q.detach().clone().requires_grad_(True)
        k_base = k.detach().clone().requires_grad_(True)
        v_base = v.detach().clone().requires_grad_(True)

        # Reshape inputs to (seq_len, batch_size, embed_dim) for PyTorch MHA
        q_mha = (
            q_base.permute(2, 0, 1, 3)
            .contiguous()
            .reshape(seq_len, batch_size, embed_dim)
        )
        k_mha = (
            k_base.permute(2, 0, 1, 3)
            .contiguous()
            .reshape(seq_len, batch_size, embed_dim)
        )
        v_mha = (
            v_base.permute(2, 0, 1, 3)
            .contiguous()
            .reshape(seq_len, batch_size, embed_dim)
        )
        do_mha = (
            do.permute(2, 0, 1, 3).contiguous().reshape(seq_len, batch_size, embed_dim)
        )

        out_base, _ = mha(q_mha, k_mha, v_mha, need_weights=False)
        out_base.backward(do_mha)

        out_base = (
            out_base.reshape(seq_len, batch_size, num_heads, head_dim)
            .permute(1, 2, 0, 3)
            .contiguous()
        )
        dq_base = q_base.grad
        dk_base = k_base.grad
        dv_base = v_base.grad

        # Memory Cleanup before testing agent implementation to reduce risk of OOM during timing.
        del q_mha, k_mha, v_mha, do_mha, q_base, k_base, v_base
        torch.cuda.empty_cache()

    # If any of the above steps fail (e.g. due to OOM or other issues)
    except Exception as exc:
        _write_result(
            output_path,
            score,
            {"error": "baseline_failed", "detail": str(exc), "seq_len": seq_len},
        )
        return

    q_agent = q.detach().clone().requires_grad_(True)
    k_agent = k.detach().clone().requires_grad_(True)
    v_agent = v.detach().clone().requires_grad_(True)
    out_agent = FlashAttention.apply(q_agent, k_agent, v_agent)
    # Mathematical correctness
    try:
        torch.testing.assert_close(out_agent, out_base, atol=1e-3, rtol=1e-3)
    except Exception as exc:
        _write_result(
            output_path,
            score,
            {"error": "forward_failed", "detail": str(exc), **metadata},
        )
        return

    # If we reach this point, the forward pass is correct, we can assign a partial score before testing backward and timing.
    score += 0.3
    metadata["stages"]["forward"] = True

    try:
        out_agent.backward(do)
        dq_agent = q_agent.grad
        dk_agent = k_agent.grad
        dv_agent = v_agent.grad

        torch.testing.assert_close(dq_agent, dq_base, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(dk_agent, dk_base, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(dv_agent, dv_base, atol=1e-3, rtol=1e-3)
    except Exception as exc:  # noqa: BLE001
        _write_result(
            output_path,
            score,
            {"error": "backward_failed", "detail": str(exc), **metadata},
        )
        return

    # If we reach this point, both forward and backward are correct. We can assign the remaining score before timing.
    score += 0.6
    metadata["stages"]["backward"] = True

    # Once correctness is verified, we can proceed to timing. We will time both the baseline and the agent implementation, and calculate a speedup factor to adjust the score.
    try:
        torch.cuda.empty_cache()
        q_time = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.float16,
        )
        k_time = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.float16,
        )
        v_time = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.float16,
        )

        q_time_mha = (
            q_time.permute(2, 0, 1, 3)
            .contiguous()
            .reshape(seq_len, batch_size, embed_dim)
        )
        k_time_mha = (
            k_time.permute(2, 0, 1, 3)
            .contiguous()
            .reshape(seq_len, batch_size, embed_dim)
        )
        v_time_mha = (
            v_time.permute(2, 0, 1, 3)
            .contiguous()
            .reshape(seq_len, batch_size, embed_dim)
        )

        # Use naive attention as the timing baseline, with MHA as an upper-bound reference.
        def baseline_fn():
            with torch.no_grad():
                _naive_attention(q_time, k_time, v_time)

        def mha_fn():
            with torch.no_grad():
                mha(q_time_mha, k_time_mha, v_time_mha, need_weights=False)

        def agent_fn():
            with torch.no_grad():
                FlashAttention.apply(q_time, k_time, v_time)

        baseline_mem = _peak_memory_bytes(baseline_fn)
        baseline_time = _time_ms(baseline_fn)
        mha_time = _time_ms(mha_fn)

        del q_time_mha, k_time_mha, v_time_mha
        torch.cuda.empty_cache()

        agent_mem = _peak_memory_bytes(agent_fn)
        agent_time = _time_ms(agent_fn)
    except Exception as exc:
        _write_result(
            output_path,
            score,
            {"error": "timing_failed", "detail": str(exc), **metadata},
        )
        return

    if agent_time <= 0:
        _write_result(
            output_path,
            score,
            {"error": "invalid_timing", "baseline_ms": baseline_time, **metadata},
        )
        return

    speedup = baseline_time / agent_time
    upper_bound = baseline_time / mha_time if mha_time > 0 else speedup
    # Cap speedup using MHA as an upper-bound reference.
    speedup = min(speedup, upper_bound)
    # If the agent uses significantly more memory than the baseline, we can penalize the score.
    # But we want to avoid reducing the score to 0 if the agent is slower but still correct and makes some progress on optimization.
    if agent_mem > 0:
        mem_ratio = baseline_mem / agent_mem
    else:
        mem_ratio = 1.0
    # We can use a simple linear scaling for memory usage, with a floor to prevent it from reducing the score too much if the agent is slower but still correct.
    mem_factor = min(max(mem_ratio, 0.25), 2.0)
    # prevent collating the score to 0 if the agent is slower than the baseline, but still give some credit for correctness.
    final_score = max(0.1, score * speedup * mem_factor)
    metadata.update(
        {
            "baseline_ms": baseline_time,
            "mha_ms": mha_time,
            "agent_ms": agent_time,
            "speedup": speedup,
            "speedup_cap": upper_bound,
            "baseline_mem_bytes": baseline_mem,
            "agent_mem_bytes": agent_mem,
            "mem_ratio": mem_ratio,
            "mem_factor": mem_factor,
        }
    )

    _write_result(output_path, final_score, metadata)


if __name__ == "__main__":
    main()
