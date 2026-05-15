import torch
import triton
from helpers import (
    mha_torch_wrapper, 
    mha_naive_wrapper, 
    mha_triton_wrapper, 
    time_ms0, 
    report
)
from benchmarks import (
    sweep_bench, 
    print_table, 
    plot_latency, 
    plot_throughput, 
    plot_tflops
)

if __name__ == "__main__":
    print("Triton version:", triton.__version__)

    # -----------------------------
    # Static Benchmark
    # -----------------------------
    torch.manual_seed(42)
    device="cuda"
    N, D, H = 8192, 512, 8

    X = torch.randn((N, D), device=device, dtype=torch.float16)

    mha = torch.nn.MultiheadAttention(embed_dim=D, num_heads=H, device=device, dtype=torch.float16)
    mha.eval()

    # correctness check first
    with torch.no_grad():
        ref = mha_torch_wrapper(X, mha)
        out = mha_naive_wrapper(X, mha)
        out_triton = mha_triton_wrapper(X, mha)
        print("max abs err:", (out - ref).abs().max().item())
        print("mean abs err:", (out - ref).abs().mean().item())
        print("triton max abs err:", (out_triton - ref).abs().max().item())
        print("triton mean abs err:", (out_triton - ref).abs().mean().item())

    # timing
    torch_ms = time_ms0(lambda: mha_torch_wrapper(X, mha), iters=100, warmup=25)
    naive_ms = time_ms0(lambda: mha_naive_wrapper(X, mha), iters=20, warmup=5)  # naive is O(N^2); use fewer iters
    triton_ms = time_ms0(lambda: mha_triton_wrapper(X, mha), iters=100, warmup=25)

    report("torch_mha", torch_ms, N, D)
    report("naive_mha", naive_ms, N, D)
    report("triton_mha", triton_ms, N, D)


    # -----------------------------
    # Benchmarking Execution
    # -----------------------------
    torch.manual_seed(42)
    device = "cuda"
    D, H = 2048, 32

    Ns = [256, 512, 1024, 1536, 2048, 3148, 4096, 5674, 8192, 11468, 16384, 24568, 32768, 45764, 65536] 

    results = sweep_bench(
        Ns=Ns, D=D, H=H, device=device, dtype=torch.float16,
        do_correctness_first=True,
        iters_torch=100, warmup_torch=25,
        iters_triton=100, warmup_triton=25,
        iters_naive=20, warmup_naive=5,  
    )

    print_table(results)

    plot_latency(results, logx=True, logy=False, title=f"MHA latency (D={D}, H={H})")
    plot_throughput(results, which="toks_s", logx=True, logy=False, title=f"MHA throughput (tokens/s) (D={D}, H={H})")
    plot_tflops(results, logx=True, logy=False, title=f"MHA TFLOPs/s (D={D}, H={H})")