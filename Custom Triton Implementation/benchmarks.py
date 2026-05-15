import math
import time
import torch
import matplotlib.pyplot as plt

from helpers import (
    mha_torch_wrapper, 
    mha_naive_wrapper, 
    mha_triton_wrapper, 
    time_ms0, 
    report
)

def _dtype_nbytes(dtype: torch.dtype) -> int:
    # torch.finfo exists for float types, but simplest:
    if dtype == torch.float16: return 2
    if dtype == torch.bfloat16: return 2
    if dtype == torch.float32: return 4
    if dtype == torch.float64: return 8
    # fallback
    return torch.tensor([], dtype=dtype).element_size()

def _throughput_tokens_per_s(N: int, ms: float) -> float:
    # "Tokens" = sequence positions processed per forward call
    # tokens/s = N / (ms/1000)
    return (N * 1000.0) / ms if ms > 0 else float("inf")

def _throughput_bytes_per_s(N: int, D: int, dtype: torch.dtype, ms: float, rw_factor: float = 3.0) -> float:
    """
    Rough bandwidth estimate: reads X and writes O; rw_factor is a heuristic.
    For attention, real traffic is higher (QKV, weights, softmax, etc).
    Use as a consistent relative metric, not absolute truth.
    """
    nbytes = _dtype_nbytes(dtype)
    bytes_moved = rw_factor * (N * D * nbytes)
    return (bytes_moved * 1000.0) / ms if ms > 0 else float("inf")

def _throughput_tflops(N: int, D: int, ms: float) -> float:
    """
    FLOPs for attention = 4 * N^2 * D_head * H
    Returns TFLOPs/s.
    """
    flops = 4.0 * N * N * D
    tflops = flops / 1e12
    return tflops / (ms / 1000.0) if ms > 0 else float("inf")

def _set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@torch.no_grad()
def correctness_check(X, mha, atol=None, rtol=None, verbose=True):
    """
    Prints max/mean abs error vs torch reference for naive and triton.
    If atol/rtol provided, asserts allclose.
    """
    ref = mha_torch_wrapper(X, mha)
    out_naive = mha_naive_wrapper(X, mha)
    out_triton = mha_triton_wrapper(X, mha)

    max_abs_naive = (out_naive - ref).abs().max().item()
    mean_abs_naive = (out_naive - ref).abs().mean().item()
    max_abs_triton = (out_triton - ref).abs().max().item()
    mean_abs_triton = (out_triton - ref).abs().mean().item()

    if verbose:
        print("naive max abs err:", max_abs_naive)
        print("naive mean abs err:", mean_abs_naive)
        print("triton max abs err:", max_abs_triton)
        print("triton mean abs err:", mean_abs_triton)
        print("No. of zeros (triton):", (out_triton == 0).sum().item())

    if (atol is not None) or (rtol is not None):
        if atol is None: atol = 0.0
        if rtol is None: rtol = 0.0
        assert torch.allclose(out_naive, ref, atol=atol, rtol=rtol), "naive failed allclose"
        assert torch.allclose(out_triton, ref, atol=atol, rtol=rtol), "triton failed allclose"

    return {
        "naive_max_abs_err": max_abs_naive,
        "naive_mean_abs_err": mean_abs_naive,
        "triton_max_abs_err": max_abs_triton,
        "triton_mean_abs_err": mean_abs_triton,
    }

def bench_one(N: int, D: int, H: int,
              device="cuda", dtype=torch.float16,
              iters_torch=100, warmup_torch=25,
              iters_triton=100, warmup_triton=25,
              iters_naive=20, warmup_naive=5,
              do_correctness=False,
              seed=42):
    """
    Returns a dict with timings + throughput for torch/naive/triton at (N, D, H).
    Uses the same timing function time_ms0(...) you already have.
    """

    _set_seed(seed)
    X = torch.randn((N, D), device=device, dtype=dtype)
    mha = torch.nn.MultiheadAttention(embed_dim=D, num_heads=H, device=device, dtype=dtype)
    mha.eval()

    errs = None
    if do_correctness:
        errs = correctness_check(X, mha, verbose=True)

    # Ensure any lazy init is done before timing (optional)
    _sync()
    with torch.no_grad():
        _ = mha_torch_wrapper(X, mha)
        _ = mha_naive_wrapper(X, mha)
        _ = mha_triton_wrapper(X, mha)
    _sync()

    # timing: keep your function
    torch_ms  = time_ms0(lambda: mha_torch_wrapper(X, mha),  iters=iters_torch,  warmup=warmup_torch)
    naive_ms  = time_ms0(lambda: mha_naive_wrapper(X, mha),  iters=iters_naive,  warmup=warmup_naive)
    triton_ms = time_ms0(lambda: mha_triton_wrapper(X, mha), iters=iters_triton, warmup=warmup_triton)

    # optional report calls (your helper)
    try:
        report("torch_mha",  torch_ms,  N, D)
        report("naive_mha",  naive_ms,  N, D)
        report("triton_mha", triton_ms, N, D)
    except NameError:
        pass

    res = {
        "N": N, "D": D, "H": H, "dtype": str(dtype).replace("torch.", ""),
        "torch_ms": float(torch_ms),
        "naive_ms": float(naive_ms),
        "triton_ms": float(triton_ms),
        "torch_toks_s": _throughput_tokens_per_s(N, torch_ms),
        "naive_toks_s": _throughput_tokens_per_s(N, naive_ms),
        "triton_toks_s": _throughput_tokens_per_s(N, triton_ms),
        "torch_Bs": _throughput_bytes_per_s(N, D, dtype, torch_ms),
        "naive_Bs": _throughput_bytes_per_s(N, D, dtype, naive_ms),
        "triton_Bs": _throughput_bytes_per_s(N, D, dtype, triton_ms),
        "torch_tflops": _throughput_tflops(N, D, torch_ms),
        "naive_tflops": _throughput_tflops(N, D, naive_ms),
        "triton_tflops": _throughput_tflops(N, D, triton_ms),
    }
    if errs is not None:
        res.update(errs)
    return res

def sweep_bench(
    Ns,
    D=128,
    H=2,
    device="cuda",
    dtype=torch.float16,
    do_correctness_first=True,
    seed=42,
    # per-impl timing params
    iters_torch=100, warmup_torch=25,
    iters_triton=100, warmup_triton=25,
    iters_naive=20, warmup_naive=5,
):
    """
    Sweeps over Ns. Returns list[dict] results.
    """
    results = []
    Ns = list(Ns)

    # One correctness check at the first N (optional)
    do_corr = bool(do_correctness_first)
    for i, N in enumerate(Ns):
        print(f"\n=== N={N}, D={D}, H={H}, dtype={dtype} ===")
        r = bench_one(
            N=N, D=D, H=H, device=device, dtype=dtype,
            iters_torch=iters_torch, warmup_torch=warmup_torch,
            iters_triton=iters_triton, warmup_triton=warmup_triton,
            iters_naive=iters_naive, warmup_naive=warmup_naive,
            do_correctness=do_corr,
            seed=seed,
        )
        do_corr = False  # only for first point
        results.append(r)
    return results

def plot_latency(results, x_key="N", logx=True, logy=False, title="MHA latency"):
    xs = [r[x_key] for r in results]
    torch_ms  = [r["torch_ms"] for r in results]
    naive_ms  = [r["naive_ms"] for r in results]
    triton_ms = [r["triton_ms"] for r in results]

    plt.figure()
    plt.plot(xs, torch_ms, marker="o", label="torch")
    plt.plot(xs, naive_ms, marker="o", label="naive")
    plt.plot(xs, triton_ms, marker="o", label="triton")
    plt.xlabel(x_key)
    plt.ylabel("time (ms)")
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    if logx: plt.xscale("log", base=2)
    if logy: plt.yscale("log")
    plt.show()

def plot_throughput(results, x_key="N", which="toks_s", logx=True, logy=False, title="MHA throughput"):
    """
    which: "toks_s" (tokens/s) or "Bs" (bytes/s)
    """
    xs = [r[x_key] for r in results]
    torch_tp  = [r[f"torch_{which}"] for r in results]
    naive_tp  = [r[f"naive_{which}"] for r in results]
    triton_tp = [r[f"triton_{which}"] for r in results]

    plt.figure()
    plt.plot(xs, torch_tp, marker="o", label="torch")
    plt.plot(xs, naive_tp, marker="o", label="naive")
    plt.plot(xs, triton_tp, marker="o", label="triton")
    plt.xlabel(x_key)
    plt.ylabel("tokens/s" if which == "toks_s" else "bytes/s (heuristic)")
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    if logx: plt.xscale("log", base=2)
    if logy: plt.yscale("log")
    plt.show()

def plot_tflops(results, logx=True, logy=False, title="MHA TFLOPs/s"):
    Ns = [r["N"] for r in results]
    plt.figure(figsize=(10, 6))
    plt.plot(Ns, [r["torch_tflops"] for r in results], "o-", label="torch")
    plt.plot(Ns, [r["naive_tflops"] for r in results], "s-", label="naive")
    plt.plot(Ns, [r["triton_tflops"] for r in results], "^-", label="triton")
    if logx: plt.xscale("log", base=2)
    if logy: plt.yscale("log")
    plt.xlabel("Sequence length N")
    plt.ylabel("TFLOPs/s")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_table(results):
    # Minimal, dependency-free table print
    cols = ["N","D","H","torch_ms","naive_ms","triton_ms",
            "torch_toks_s","naive_toks_s","triton_toks_s",
            "torch_tflops", "naive_tflops", "triton_tflops"]
    header = " | ".join([f"{c:>12}" for c in cols])
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                row.append(f"{v:12.4f}")
            else:
                row.append(f"{str(v):>12}")
        print(" | ".join(row))