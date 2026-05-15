# Flash Attention v2: RL Environment & Data Scaffold

This repository contains a Reinforcement Learning (RL) environment and synthetic data scaffold designed to evaluate and train LLMs in **hardware-aware ML systems programming**. The task challenges agents to bridge the gap between mathematical definitions and hardware-aware execution by writing a custom, memory-fused Flash Attention v2 kernel using OpenAI Triton.

This infrastructure is split to support two critical phases:
1. **Offline SFT Generation:** A multi-turn harness that acts as a verifiable trajectory generator for Supervised Fine-Tuning (SFT) datasets.
2. **Online RL Evaluation:** A zero-shot environment that acts as a continuous, Rule-Based Reward Model for algorithms like PPO or GRPO.

---

## 1. Motivation & Background

I designed this environment because it mirrors the real-world challenge of pairing representation learning with systems-aware execution. Attention is a major cost in modern AI, meaning that memory-efficient kernels are critical. I previously built a custom Flash Attention v2 Triton kernel that achieved a ~4x inference speedup over naive baselines on an A100 GPU via SRAM tiling and memory-bound optimizations. This environment encodes that exact trial-and-error process into a rigorous benchmark.

**Alignment with Preference Model's Mission**
I am deeply interested in lightweight RL policies that steer frozen base models. In previous research, I explored policy-guided control for masked diffusion language models (LLaDA), replacing heuristic remasking schedules with an RL policy trained via REINFORCE to achieve measurable gains on structured generation. This environment brings that same philosophy to ML infrastructure: training agents to optimize system execution without altering the underlying math.

---

## 2. Environment Architecture & Design Choices

The key metric for this task is non-differentiable execution speed. Pure SFT can teach logical correctness but struggles to optimize abstract performance objectives and Out-of-domain generalization. This environment provides a continuous reward signal that forces the agent to explore block sizes, synchronization choices, and High Bandwidth Memory (HBM) traffic.

### The Reward Model (The Judge)
The environment grading is constructed in such a way to provide a dense, continuous signal while preventing reward hacking:
* **Reward Shaping:** Partial credit is awarded for incremental progress: Import success (+0.1), Forward-pass correctness (+0.3), Backward-pass correctness (+0.6).
* **Latency Multiplier:** A speedup multiplier is applied *only* when both the forward and backward passes achieve strict mathematical equivalence to PyTorch baselines. 
* **Dynamic Constraints:** Sequence lengths are randomized per run—specifically including non-power-of-2 dimensions—to prevent the agent from overfitting to fixed tensor shapes and force correct dynamic masking.
* **Peak Memory Tracking:** Peak HBM usage is measured alongside execution speed to align the reward with the prompt's minimization requirements. This is ensure that gains are through methods intended in the prompt.

### Design Choice: Isolating the Kernel vs. Full MHA
The Judge puts identity matrices onto a reference PyTorch `MultiheadAttention` module. This intentionally isolates the memory-bound attention operation from `cuBLAS` GEMM projections (like $X*W_q$). This isolation avoids distracting attention performance with linear projection overheads, providing a un-noisy gradient for the RL optimizer.

### Why SFT Bootstrapping is Required
For complex systems tasks, pure RL often fails to reach the first success, starving the reward signal. This repository includes a **Rollout Harness** that bootstraps RL via a multi-turn self-correction loop (`Prompt -> Bad Code -> Compiler Error -> Reflection -> Fixed Code`). It serializes successful trajectories into JSONL. This can be used later (not implemented here) to fine-tune a base model to prevent 'cold start' failures in the RL phase.

### Models Used
I have used two models Anthropic-claude-haiku-4-5 and Anthropic-claude-sonnet-4-6. Both are frontier models in line with this task. They can be switched out with any other model.

### Implementation 
In the run for Anthropic-claude-sonnet-4-6 (seen in out_sonnet_4-6), the judge failed in the inital agent run, so I ran it manually afterward (after fix), so the results stored in out_sonnet_4.6/triton_results.txt. The code passed the judge's tests and got a **non-zero score**. 
The output from Anthropic-claude-haiku-4-5 (out_haiku_4-5) **failed the judge's static test**, earning score of 0.0. 
This supports the fact that Sonnet is a stronger model than Haiku. These scores and metadata can be used as reward in our RL training to steer the trajectory.

---
## 3. Agentic Behviour and Challenges

### What the task teaches (Code-to-Reasoning Transfer)
* **Constraint Satisfaction:** Kernel tiling on modern GPUs is a multi-dimensional packing problem (thread blocks, SRAM limits, warp sync). Training on this builds the model's ability to juggle strict parameters simultaneously.
* **Long-Horizon Planning:** To achieve meaningful speedups, the model must plan memory layouts early.
* **Numerical Stability:** Implementing online softmax with correct running statistics inside an SRAM loop.
THe goal is to make the model hardware-aware by making it focus of the hardware constraints while RL-finetuning.

### Some Agent Struggles
* Getting the backward pass gradients correct under the reversed loop ordering specific to Flash Attention v2.
* Matching PyTorch gradients within tight `1e-3` tolerances in `fp16`.
* Handling edge-case indexing bugs when masking non-divisible sequence lengths.
* In the first pass, models wrote tests in the self-relection process, but Haiku_4-5 never considered execution speed as a metric to optimize on.

### Reward Hacking
* *Zero-Compute Hacking.* The agent returns `torch.zeros_like(q)` to win on latency. **Mitigation:** Strict `torch.allclose` mathematical checks against randomized PyTorch baselines.
* *The Wrapper Cheat.* The agent imports built-in `scaled_dot_product_attention` or the `flash_attn` library. **Mitigation:** Static AST checks enforce the usage of `@triton.jit` and `tl.dot` while banning shortcut APIs.
* *Hardcoding.* The agent hardcodes block sizes that only work for a specific sequence length (say `seq_len=1024`). **Mitigation:** Randomized sequence dimensions during evaluation.

---

## 4. Future Work

* **Curriculum Learning Pipeline:** If full implementation is too steep a reward cliff, we need to construct a curriculum: Level 1 (Triton batched matmul), Level 2 (Flash Attention Forward only), Level 3 (Forward + Backward). Moving from easier sub-tasks to the full challenge can help the agent bootstrap skills incrementally.
* **FP8 / Quantized Kernels:** Future iterations should evaluate FP8 block-wise quantized attention to support multi-episode training, adjusting tolerance checks for quantization noise.
* **Trace-Based Observation Spaces:** Utilizing the raw `ptxas` compiler traces currently captured in the metadata to train agents via RL on reasoning traces, rather than purely scalar rewards.

---

## 5. Execution Guide

### Prerequisites & Hardware
* This environment requires a CUDA-capable GPU. The scoring script natively targets A100-class GPUs and uses `fp16` tensors. 
* Install `uv`, and ensure Podman or Docker is installed with GPU passthrough enabled.

### Phase 1: Installation & Setup
1. Sync the host dependencies:
   ```bash
   uv sync
   ```
2. Generate the mathematical reference data (`env_data/naive_attention.py`):
   ```bash
   uv run setup_data.py
   ```

### Phase 2: Verifying the Reward Model (Unit Tests)
In RL, a reward bug ruins the training run. Validate that the Judge strictly enforces mathematical correctness and properly rejects cheating APIs before executing GPU rollouts.
```bash
uv run pytest --junitxml test_out/pytest-results.xml
```
The results of the test will be stored in test_out/pytest-results.xml

### Phase 3: The SFT Data Pipeline (Offline Harness)
Generate multi-turn self-correction trajectories for SFT using the standalone harness. *Requires a provider API key.*
```bash
export MODEL_API_KEY="your_api_key_here"
uv run python sft_harness_scripts/rollout_harness.py     --model anthropic/claude-haiku-4-5-20251001     --num-rollouts 3     --temperature 0.6     --max-turns 4     --success-jsonl sft_harness_out/rollouts/flash_sft_success.jsonl     --negative-jsonl sft_harness_out/rollouts/flash_sft_negative.jsonl     --save-negative
```
The logs of User and Agent per turn for each role, the code generated (if it passes static checks) per turn and the judge output/error can be found at sft_harness_scripts/rollouts
Note: Rate Limiting is a problem to consider if large number of rollouts and turns are used.

### Phase 4: The RL Environment (Online Judge)
Execute the environment in standard zero-shot mode.

1. Generate a run config for `flash-attention-triton`:
   ```bash
   export ANTHROPIC_API_KEY="your_api_key_here"
   uv run pm_env create-run-config --model claude-haiku-4-5-20251001 --model-api-key $ANTHROPIC_API_KEY
   ```
2. Execute the environment container:
   ```bash
   uv run pm_env run --config <your-config>
   ```
   *(If using Docker instead of Podman, append `--runtime docker`).*

**Optional: Manual Judge Execution**
To bypass the agent and manually evaluate a candidate solution file:
```bash
uv run python src/pm_env/score_flash_attention.py <solution_path> /tmp/triton_results.txt
```

---

## 6. Infrastructure & Environment Adaptation

**Context:** Deployment and execution of the pm\_env agentic evaluation harness for validating Flash Attention v2 Triton kernels on A100 GPUs.

I had encountered some problems when adapting the default Docker-based evaluation harness to restricted High-Performance Computing (HPC) clusters (TAMU Grace) and cloud-tenant environments (RunPod). This portion outlines the engineering solutions implemented to ensure a successful, mathematically validated agentic run.

_**Note:** On a standard AWS/GCP instance with **sudo privileges and a standard Docker daemon**, the framework's default containerization likely functions flawlessly. The adaptations below were necessary and intentional specifically for running agentic evaluations within the strict security boundaries of SLURM-scheduled HPC hardware and restricted cloud pods._

### 1. Containerization in HPC: Rootless Execution & Toolchain Parity


**The Problem:** The pm\_env framework defaults to Docker/Podman containerization for agent sandboxing. While standard practice, it presents two critical failures on enterprise HPC clusters:

1.  **Privilege Demotion Crashes:** HPCs utilize rootless container engines (like Singularity/Apptainer). The framework utilizes preexec\_fn (os.setuid/setgid) to drop root privileges before opening a bash shell for the agent. In a rootless HPC namespace, the user is already non-root, and attempting to modify user IDs triggers an immediate kernel-level security block (Exception occurred in preexec\_fn).
    
2.  **Shallow Runtime Images:** The provided evaluation container image was a minimal runtime. It lacked the necessary JIT-compilation toolchain (binutils like as and ld, and glibc-devel headers like stdlib.h) required by Triton to dynamically compile the PTX binaries for the A100 GPU.
    
I disabled the framework's containerization flags and initialized a native virtual environment directly on the A100 host. I also added a Security Directive in the prompt to prevent access to folders other than env_data/ and out/.
```
export PM_CONTAINERIZED=0
```

    
### 2. Firewall Restrictions (iptables)


**The Problem:** To safely sandbox the agent and prevent unauthorized web requests, the evaluation harness attempts to dynamically apply iptables firewall rules (\_maybe\_block\_internet). Attempting to execute iptables results in a fatal AssertionError: failed to apply firewall rule.

**The Solution:** Because the evaluation was safely taking place in an isolated cloud pod, the software-level firewall was redundant. I bypassed the restriction by not calling this function.


### 3. Port Collisions


**The Problem:** The framework configures a WebSocket broadcaster on 0.0.0.0:8001. On managed cloud platforms (like RunPod), the 8000-8005 port range is heavily reserved by the host OS for internal web-terminal proxies, JupyterLab endpoints, and health probes. This causes immediate \[Errno 98\] Address already in use crashes upon startup.

**The Solution:** Instead of attempting to kill locked system proxy processes, I shifted the framework's communication ports to an unreserved range via run\_config.json:

```
"websocket\_config": {"host": "0.0.0.0","port": 9042}
```

### Conclusion


By mapping the native JIT toolchains, evading the internal firewall, and evading host port collisions, the pm\_env evaluation harness successfully executed. The Agent was able to autonomously use bash, read the reference implementation, and compile its Triton kernel on the A100 hardware for mathematical validation and judging.

---

## AI Usage Disclosure
I utilized GitHub Copilot to assist with debugging/quickly coding the Triton syntax, modularizing Python functions, and formatting documentation. I used Gemini Pro to ideate on environment completeness and edge-case coverage. However, the core architectural design, the reward shaping logic, the prompt constraints, and the mathematical verification strategies are my own original work.
