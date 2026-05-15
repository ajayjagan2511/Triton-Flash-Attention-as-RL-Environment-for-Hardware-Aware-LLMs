import torch

from naive_attention import multiheaded_attention_naive
from flash_attention_v2 import multiheaded_attention_triton 

@torch.no_grad()
def time_ms0(fn, iters=100, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def mha_naive_wrapper(X, mha: torch.nn.MultiheadAttention):
    num_heads = mha.num_heads
    in_proj_w = mha.in_proj_weight
    in_proj_b = mha.in_proj_bias
    out_proj = mha.out_proj
    device = in_proj_w.device

    attention_naive_out = multiheaded_attention_naive(
        X,
        in_proj_w,
        out_proj.weight,
        in_proj_b,
        out_proj.bias,
        num_heads,
        device
    )

    return attention_naive_out

def mha_triton_wrapper(X, mha: torch.nn.MultiheadAttention):
    num_heads = mha.num_heads
    in_proj_w = mha.in_proj_weight
    in_proj_b = mha.in_proj_bias
    out_proj = mha.out_proj
    device = in_proj_w.device

    attention_triton_out = multiheaded_attention_triton(
        X, X, X,
        in_proj_w,
        out_proj.weight,
        in_proj_b,
        out_proj.bias,
        num_heads,
        device
    )

    return attention_triton_out

def mha_torch_wrapper(X, mha: torch.nn.MultiheadAttention):
    # For (N, D) input, PyTorch interprets as (L, E) when unbatched.
    # This returns (L, E). Use need_weights=False to time only output.
    out, _ = mha(X, X, X, need_weights=False)
    return out

def report(name, ms, N, D):
    toks_per_s = N / (ms / 1e3)
    print(f"{name:>14}: {ms:8.3f} ms | {toks_per_s:10.1f} tokens/s")