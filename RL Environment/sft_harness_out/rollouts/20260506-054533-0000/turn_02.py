import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_forward_kernel(
    Q, K, V, Out, L, M,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_mb, stride_mh, stride_ms,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Flash Attention v2 forward kernel.
    Computes attention with online softmax and recomputation-friendly design.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_m = tl.program_id(2)
    
    # Precompute scaling factor
    scale = 1.0 / tl.sqrt(BLOCK_D * 1.0)
    
    # Offsets for query block
    q_offset = batch_idx * stride_qb + head_idx * stride_qh + block_m * BLOCK_M * stride_qs
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_offset = batch_idx * stride_ob + head_idx * stride_oh + block_m * BLOCK_M * stride_os
    l_offset = batch_idx * stride_lb + head_idx * stride_lh + block_m * BLOCK_M * stride_ls
    m_offset = batch_idx * stride_mb + head_idx * stride_mh + block_m * BLOCK_M * stride_ms
    
    # Initialize accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Load Q block
    q_ptrs = Q + q_offset + tl.arange(0, BLOCK_M)[:, None] * stride_qs + tl.arange(0, BLOCK_D)[None, :] * stride_qd
    q_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    # Iterate over K,V blocks
    for block_n in range(tl.cdiv(N_CTX, BLOCK_N)):
        k_ptrs = K + k_offset + block_n * BLOCK_N * stride_ks + tl.arange(0, BLOCK_N)[None, :] * stride_ks + tl.arange(0, BLOCK_D)[:, None] * stride_kd
        v_ptrs = V + v_offset + block_n * BLOCK_N * stride_vs + tl.arange(0, BLOCK_N)[None, :] * stride_vs + tl.arange(0, BLOCK_D)[:, None] * stride_vd
        
        k_mask = (block_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]) < N_CTX
        v_mask = (block_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]) < N_CTX
        
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        
        # Compute attention scores: Q @ K^T / sqrt(d)
        s = tl.dot(q, k)
        s = s * scale
        
        # Causal mask: prevent attending to future tokens
        causal_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) >= (block_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
        s = tl.where(causal_mask, s, float("-inf"))
        
        # Online softmax: m_ij = max(m_i, max(s))
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Compute exp(s - m_new)
        p = tl.exp(s - m_new[:, None])
        
        # Update l_i: l_i = exp(m_i - m_new) * l_i + sum(exp(s - m_new))
        l_ij = tl.sum(p, axis=1)
        l_i = tl.exp(m_i - m_new) * l_i + l_ij
        
        # Update accumulator: acc = diag(exp(m_i - m_new)) @ acc + p @ V
        acc = tl.exp(m_i - m_new)[:, None] * acc + tl.dot(p, v)
        
        # Update m_i
        m_i = m_new
    
    # Normalize output: O = acc / l_i
    out = acc / l_i[:, None]
    
    # Store outputs
    o_ptrs = Out + o_offset + tl.arange(0, BLOCK_M)[:, None] * stride_os + tl.arange(0, BLOCK_D)[None, :] * stride_od
    o_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) < N_CTX
    tl.store(o_ptrs, out.to(Out.dtype.element_type), mask=o_mask)
    
    # Store l_i and m_i for backward
    l_ptrs = L + l_offset + tl.arange(0, BLOCK_M) * stride_ls
    m_ptrs = M + m_offset + tl.arange(0, BLOCK_M) * stride_ms
    tl.store(l_ptrs, l_i, mask=(block_m * BLOCK_M + tl.arange(0, BLOCK_M)) < N_CTX)
    tl.store(m_ptrs, m_i, mask=(block_m * BLOCK_M + tl.arange(0, BLOCK_M)) < N_CTX)


@triton.jit
def flash_attention_backward_kernel(
    Q, K, V, Out, DO, DQ, DK, DV, L, M,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    stride_mb, stride_mh, stride_ms,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Flash Attention v2 backward kernel.
    Computes gradients for Q, K, V.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_m = tl.program_id(2)
    
    # Precompute scaling factor
    scale = 1.0 / tl.sqrt(BLOCK_D * 1.0)
    
    # Offsets
    q_offset = batch_idx * stride_qb + head_idx * stride_qh + block_m * BLOCK_M * stride_qs
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_offset = batch_idx * stride_ob + head_idx * stride_oh + block_m * BLOCK_M * stride_os
    do_offset = batch_idx * stride_dob + head_idx * stride_doh + block_m * BLOCK_M * stride_dos
    dq_offset = batch_idx * stride_dqb + head_idx * stride_dqh + block_m * BLOCK_M * stride_dqs
    l_offset = batch_idx * stride_lb + head_idx * stride_lh + block_m * BLOCK_M * stride_ls
    m_offset = batch_idx * stride_mb + head_idx * stride_mh + block_m * BLOCK_M * stride_ms
    
    # Load Q, L, M
    q_ptrs = Q + q_offset + tl.arange(0, BLOCK_M)[:, None] * stride_qs + tl.arange(0, BLOCK_D)[None, :] * stride_qd
    q_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    l_ptrs = L + l_offset + tl.arange(0, BLOCK_M) * stride_ls
    l = tl.load(l_ptrs, mask=(block_m * BLOCK_M + tl.arange(0, BLOCK_M)) < N_CTX, other=0.0).to(tl.float32)
    
    m_ptrs = M + m_offset + tl.arange(0, BLOCK_M) * stride_ms
    m = tl.load(m_ptrs, mask=(block_m * BLOCK_M + tl.arange(0, BLOCK_M)) < N_CTX, other=0.0).to(tl.float32)
    
    # Load DO
    do_ptrs = DO + do_offset + tl.arange(0, BLOCK_M)[:, None] * stride_dos + tl.arange(0, BLOCK_D)[None, :] * stride_dod
    do = tl.load(do_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    # Load O
    o_ptrs = Out + o_offset + tl.arange(0, BLOCK_M)[:, None] * stride_os + tl.arange(0, BLOCK_D)[None, :] * stride_od
    o = tl.load(o_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    # Compute dS scaling factor: sum(DO * O) for each query
    ds_scale = tl.sum(do * o, axis=1)
    
    # Initialize DQ accumulator
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Iterate over K,V blocks
    for block_n in range(tl.cdiv(N_CTX, BLOCK_N)):
        k_ptrs = K + k_offset + block_n * BLOCK_N * stride_ks + tl.arange(0, BLOCK_N)[None, :] * stride_ks + tl.arange(0, BLOCK_D)[:, None] * stride_kd
        v_ptrs = V + v_offset + block_n * BLOCK_N * stride_vs + tl.arange(0, BLOCK_N)[None, :] * stride_vs + tl.arange(0, BLOCK_D)[:, None] * stride_vd
        
        k_mask = (block_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]) < N_CTX
        v_mask = (block_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]) < N_CTX
        
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        
        # Recompute attention scores and softmax
        s = tl.dot(q, k)
        s = s * scale
        
        # Causal mask
        causal_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) >= (block_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
        s = tl.where(causal_mask, s, float("-inf"))
        
        # Softmax
        p = tl.exp(s - m[:, None])
        p = p / l[:, None]
        
        # Compute dV: dV += P^T @ DO
        dv_block = tl.dot(p.T, do)
        dv_ptrs = DV + batch_idx * stride_dvb + head_idx * stride_dvh + block_n * BLOCK_N * stride_dvs + tl.arange(0, BLOCK_N)[:, None] * stride_dvs + tl.arange(0, BLOCK_D)[None, :] * stride_dvd
        tl.store(dv_ptrs, dv_block.to(DV.dtype.element_type), mask=v_mask)
        
        # Compute dS: dS = DO @ V^T
        ds = tl.dot(do, v.T)
        
        # Compute dP: dP = dS * P
        dp = ds * p
        
        # Subtract ds_scale term: dP -= P * ds_scale
        dp = dp - p * ds_scale[:, None]
        
        # Compute dK: dK += P^T @ dP
        dk_block = tl.dot(p.T, dp)
        dk_ptrs = DK + batch_idx * stride_dkb + head_idx * stride_dkh + block_n * BLOCK_N * stride_dks + tl.arange(0, BLOCK_N)[:, None] * stride_dks + tl.arange(0, BLOCK_D)[None, :] * stride_dkd
        tl.store(dk_ptrs, dk_block.to(DK.dtype.element_type), mask=k_mask)
        
        # Compute dQ: dQ += dP @ K
        dq = dq + tl.dot(dp, k)
    
    # Scale dQ by 1/sqrt(d)
    dq = dq * scale
    
    # Store dQ
    dq_ptrs = DQ + dq_offset + tl.arange(0, BLOCK_M)[:, None] * stride_dqs + tl.arange(0, BLOCK_D)[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(DQ.dtype.element_type), mask=q_mask)


class FlashAttention(torch.autograd.Function):
    """
    Flash Attention v2 implementation using Triton kernels.
    """
    
    @staticmethod
    def forward(ctx, q, k, v):
        """
        Forward pass of Flash Attention.
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
        
        Returns:
            out: Output tensor [batch, heads, seq_len, head_dim]
        """
        batch, heads, seq_len, head_dim = q.shape
        assert q.dtype == torch.float16
        assert k.dtype == torch.float16
        assert v.dtype == torch.float16
        assert head_dim == 64
        
        # Allocate output and auxiliary tensors
        out = torch.empty_like(q)
        l = torch.zeros((batch, heads, seq_len), dtype=torch.float32, device=q.device)
        m = torch.full((batch, heads, seq_len), float("-inf"), dtype=torch.float32, device=q.device)
        
        # Kernel parameters
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_D = 64
        
        # Grid dimensions
        grid = (batch, heads, (seq_len + BLOCK_M - 1) // BLOCK_M)
        
        # Launch forward kernel
        flash_attention_forward_kernel[grid](
            q, k, v, out, l, m,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            l.stride(0), l.stride(1), l.stride(2),
            m.stride(0), m.stride(1), m.stride(2),
            N_CTX=seq_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        
        ctx.save_for_backward(q, k, v, out, l, m)
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_D = BLOCK_D
        
        return out
    
    @staticmethod
    def backward(ctx, do):
        """
        Backward pass of Flash Attention.
        
        Args:
            do: Gradient of output [batch, heads, seq_len, head_dim]
        
        Returns:
            dq, dk, dv: Gradients for q, k, v
        """
        q, k, v, out, l, m = ctx.saved_tensors
        batch, heads, seq_len, head_dim = q.shape
        
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N
        BLOCK_D = ctx.BLOCK_D
        
        # Allocate gradient tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        # Grid dimensions
        grid = (batch, heads, (seq_len + BLOCK_M - 1) // BLOCK_M)
        
        # Launch backward kernel
        flash_attention_backward_kernel[grid](
            q, k, v, out, do, dq, dk, dv, l, m,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            l.stride(0), l.stride(1), l.stride(2),
            m.stride(0), m.stride(1), m.stride(2),
            N_CTX=seq_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        
        return dq, dk, dv


def flash_attention(q, k, v):
    """
    Wrapper function for Flash Attention v2.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
    
    Returns:
        out: Output tensor [batch, heads, seq_len, head_dim]
    """
    return FlashAttention.apply(q, k, v)


if __name__ == "__main__":
    # Test script
    torch.manual_seed(42)
    
    batch, heads, seq_len, head_dim = 1, 4, 512, 64
    
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    
    # Forward pass
    out = flash_attention(q, k, v)
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    
    # Backward pass
    loss = out.sum()
    loss.backward()
    
    print("Forward and backward passes completed successfully!")
    print(f"Q grad shape: {q.grad.shape}")
    print(f"K grad shape: {k.grad.shape}")
    print(f"V grad shape: {v.grad.shape}")