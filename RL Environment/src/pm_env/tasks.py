import sys
from pathlib import Path
from textwrap import dedent

from pm_env.get_data_dir import get_env_data_dir
from pm_env.judges.executable_judge import ExecutableJudge
from pm_env.schemas.evaluation_run_config import EvaluationRunConfig
from pm_env.task import Step, Task


def get_tasks(config: EvaluationRunConfig) -> list[Task]:
    """Create tasks for this environment."""
    # Evaluation script is stored here.
    score_script = (Path(__file__).parent / "score_flash_attention.py").as_posix()
    BASE_DIR = "/workspace/ajay_jagannath_pm_env_slim"

    return [
        Task(
            id="flash-attention-triton",
            tools=["bash"],
            steps=[
                # We go for a single-step task to give the agent maximum freedom to iterate and optimize.
                Step(
                    instructions=dedent(
                        f"""
                        Your task is to write optimized Flash Attention v2 forward-pass and backward-pass kernels using OpenAI Triton. You must minimize High Bandwidth Memory (HBM) overhead by fusing computations and tiling intermediate states directly in SRAM.

                        You have access to a mathematically correct, unoptimized PyTorch implementation in {BASE_DIR}/env_data/naive_attention.py to serve as a mathematical guide for the logic.

                        ALGORITHMIC REQUIREMENTS:
                        To achieve the target speedup, your implementation must specifically utilize the Flash Attention V2 algorithm design. Standard attention or V1 implementations will fail the performance benchmarks.

                        Your Triton kernel must strictly adhere to the following architectural constraints:

                        Algorithmic Loop Ordering (V2 Specific): You must parallelize over the sequence length of the Query (Q) dimension. Your outer loop must load blocks of Q into SRAM. Your inner loop must iterate over blocks of Keys (K) and Values (V). (Note: This is the reverse of V1, which looped over K and V on the outside).

                        Memory Fusing: You are strictly forbidden from materializing the full (seq_len, seq_len) attention matrix (S) or the softmax matrix (P) in High Bandwidth Memory (HBM). All intermediate QK^T operations must remain in SRAM.

                        Online Softmax: You must compute the softmax incrementally. Inside the inner loop, you must maintain and update a running maximum (m_i) and a running denominator sum (l_i) for each row, rescaling the accumulator (O_i) at each step to prevent numerical overflow before finally multiplying by the V block.

                        Triton-Specific Optimizations: You should utilize tl.dot for matrix multiplications, and ensure your memory pointers (ptrs) are dynamically masked to handle sequence lengths that are not perfectly divisible by your chosen block sizes.

                        When you are done, you must create a file {BASE_DIR}/out/triton_attention.py containing the following:
                        1. flash_attention_forward(q, k, v): Must accept Query, Key, and Value tensors of shape [batch_size, num_heads, seq_len, head_dim]. It must return the computed attention output and the intermediate log-sum-exp (m, l) statistics needed for the backward pass.
                        2. flash_attention_backward(do, q, k, v, out, m, l): Must accept the upstream gradient (do), the original inputs, the forward output, and the intermediate statistics. It must return the gradients dq, dk, and dv.
                        3. FlashAttention: A class inheriting from torch.autograd.Function that properly implements the forward and backward static methods, wrapping your Triton kernels into a single differentiable PyTorch operation.

                        FlashAttention.apply(q, k, v) must return only the output tensor. Store m and l internally for backward.

                        Your wrapped implementation must mathematically match the autograd-derived backward gradients of standard PyTorch attention within a 1e-3 tolerance across various standard sequence lengths. Your final solution will be evaluated on an A100 GPU for both strict mathematical correctness and combined execution latency.
                        SECURITY DIRECTIVE: You are strictly quarantined to the {BASE_DIR}/out and {BASE_DIR}/env_data directories. You are explicitly forbidden from reading the scoring scripts or navigating outside these folders. Any attempt to cat files outside these directories will result in an immediate failure.
                        """
                    ),
                    judge=ExecutableJudge(
                        [
                            sys.executable,
                            score_script,
                            f"{BASE_DIR}/out/triton_attention.py",
                            f"{BASE_DIR}/out/triton_results.txt",
                        ]
                    ),
                ),
            ],
        ),
    ]
