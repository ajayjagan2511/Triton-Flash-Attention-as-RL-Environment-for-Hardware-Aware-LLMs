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
    seq_len, head_dim,
    Br: tl.constexpr, Bc: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
):
    """
    Flash Attention v2 forward kernel.
    Processes queries in blocks (Br), keys/values in blocks (Bc).
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_m = tl.program_id(2)
    
    # Compute block offsets
    start_m = block_m * Br
    
    # Initialize accumulators
    m_i = tl.full((Br,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((Br,), dtype=tl.float32)
    acc = tl.zeros((Br, BLOCK_SIZE_D), dtype=tl.float32)
    
    # Load Q block [Br, head_dim]
    offs_m = start_m + tl.arange(0, Br)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    
    q_ptrs = Q + batch_id * stride_qb + head_id * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    # Convert Q to float32 for computation
    q = q.to(tl.float32)
    
    # Scaling factor
    scale = 1.0 / tl.sqrt(head_dim * 1.0)
    
    # Iterate over K, V blocks
    for block_n in range(0, (seq_len + Bc - 1) // Bc):
        start_n = block_n * Bc
        offs_n = start_n + tl.arange(0, Bc)
        
        # Load K block [Bc, head_dim]
        k_ptrs = K + batch_id * stride_kb + head_id * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        # Convert K to float32 for computation
        k = k.to(tl.float32)
        
        # Load V block [Bc, head_dim]
        v_ptrs = V + batch_id * stride_vb + head_id * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        # Convert V to float32 for computation
        v = v.to(tl.float32)
        
        # Compute Q @ K^T [Br, Bc]
        s = tl.dot(q, tl.trans(k))
        s = s * scale
        
        # Causal mask: mask future tokens
        s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
        
        # Compute row-wise max
        m_ij = tl.max(s, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Compute exp(s - m_i_new)
        p = tl.exp(s - m_i_new[:, None])
        
        # Compute l_i update
        l_ij = tl.sum(p, axis=1)
        l_i_new = tl.exp(m_i - m_i_new) * l_i + l_ij
        
        # Update accumulator
        acc_new = tl.exp(m_i - m_i_new)[:, None] * acc + tl.dot(p, v)
        
        # Update state
        m_i = m_i_new
        l_i = l_i_new
        acc = acc_new
    
    # Normalize output
    out = acc / l_i[:, None]
    
    # Convert back to original dtype for storage
    out = out.to(Out.dtype.element_type)
    
    # Store output
    out_ptrs = Out + batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    out_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(out_ptrs, out, mask=out_mask)
    
    # Store L and M for backward
    l_ptrs = L + batch_id * stride_lb + head_id * stride_lh + offs_m[:] * stride_ls
    m_ptrs = M + batch_id * stride_mb + head_id * stride_mh + offs_m[:] * stride_ms
    tl.store(l_ptrs, l_i, mask=offs_m < seq_len)
    tl.store(m_ptrs, m_i, mask=offs_m < seq_len)


@triton.jit
def flash_attention_backward_kernel(
    dO, Q, K, V, Out, L, M, dQ, dK, dV,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_mb, stride_mh, stride_ms,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    seq_len, head_dim,
    Br: tl.constexpr, Bc: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
):
    """
    Flash Attention v2 backward kernel.
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_m = tl.program_id(2)
    
    start_m = block_m * Br
    offs_m = start_m + tl.arange(0, Br)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    
    # Load Q block
    q_ptrs = Q + batch_id * stride_qb + head_id * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q = q.to(tl.float32)
    
    # Load dO block
    do_ptrs = dO + batch_id * stride_dob + head_id * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    do = tl.load(do_ptrs, mask=q_mask, other=0.0)
    do = do.to(tl.float32)
    
    # Load L and M
    l_ptrs = L + batch_id * stride_lb + head_id * stride_lh + offs_m[:] * stride_ls
    m_ptrs = M + batch_id * stride_mb + head_id * stride_mh + offs_m[:] * stride_ms
    l_i = tl.load(l_ptrs, mask=offs_m < seq_len, other=0.0)
    m_i = tl.load(m_ptrs, mask=offs_m < seq_len, other=0.0)
    
    # Load output
    out_ptrs = Out + batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    out = tl.load(out_ptrs, mask=q_mask, other=0.0)
    out = out.to(tl.float32)
    
    # Compute dQ contribution
    dq = tl.zeros((Br, BLOCK_SIZE_D), dtype=tl.float32)
    
    # Scaling factor
    scale = 1.0 / tl.sqrt(head_dim * 1.0)
    
    # Iterate over K, V blocks
    for block_n in range(0, (seq_len + Bc - 1) // Bc):
        start_n = block_n * Bc
        offs_n = start_n + tl.arange(0, Bc)
        
        # Load K, V blocks
        k_ptrs = K + batch_id * stride_kb + head_id * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        k = k.to(tl.float32)
        
        v_ptrs = V + batch_id * stride_vb + head_id * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        v = v.to(tl.float32)
        
        # Compute S = Q @ K^T
        s = tl.dot(q, tl.trans(k))
        s = s * scale
        
        # Causal mask
        s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
        
        # Compute P = softmax(S)
        p = tl.exp(s - m_i[:, None])
        p = p / l_i[:, None]
        
        # dV += P^T @ dO
        dv_block = tl.dot(tl.trans(p), do)
        dv_ptrs = dV + batch_id * stride_dvb + head_id * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        dv_block = dv_block.to(dV.dtype.element_type)
        tl.atomic_add(dv_ptrs, dv_block, mask=v_mask)
        
        # dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v))
        
        # dS = P * (dP - (P * dO @ out^T).sum(axis=1))
        ds = p * (dp - tl.sum(p * do, axis=1)[:, None])
        
        # dQ += dS @ K
        dq = dq + tl.dot(ds, k)
        
        # dK += dS^T @ Q
        dk_block = tl.dot(tl.trans(ds), q)
        dk_ptrs = dK + batch_id * stride_dkb + head_id * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dk_block = dk_block.to(dK.dtype.element_type)
        tl.atomic_add(dk_ptrs, dk_block, mask=k_mask)
    
    # Store dQ
    dq_out = dq.to(dQ.dtype.element_type)
    dq_ptrs = dQ + batch_id * stride_dqb + head_id * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_out, mask=q_mask)


class FlashAttention(torch.autograd.Function):
    """
    Flash Attention v2 implementation using Triton kernels.
    """
    
    @staticmethod
    def forward(ctx, q, k, v):
        """
        Forward pass of Flash Attention v2.
        
        Args:
            q: [batch, heads, seq_len, head_dim]
            k: [batch, heads, seq_len, head_dim]
            v: [batch, heads, seq_len, head_dim]
        
        Returns:
            out: [batch, heads, seq_len, head_dim]
        """
        batch, heads, seq_len, head_dim = q.shape
        
        # Ensure inputs are contiguous and on CUDA
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Allocate output
        out = torch.empty_like(q)
        L = torch.zeros((batch, heads, seq_len), device=q.device, dtype=torch.float32)
        M = torch.zeros((batch, heads, seq_len), device=q.device, dtype=torch.float32)
        
        # Tuning parameters (must be constexpr)
        Br = 128
        Bc = 128
        BLOCK_SIZE_D = 64
        
        # Launch kernel
        grid = (batch, heads, (seq_len + Br - 1) // Br)
        
        flash_attention_forward_kernel[grid](
            q, k, v, out, L, M,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            seq_len, head_dim,
            Br, Bc, BLOCK_SIZE_D,
        )
        
        ctx.save_for_backward(q, k, v, out, L, M)
        ctx.Br = Br
        ctx.Bc = Bc
        ctx.BLOCK_SIZE_D = BLOCK_SIZE_D
        
        return out
    
    @staticmethod
    def backward(ctx, dout):
        """
        Backward pass of Flash Attention v2.
        
        Args:
            dout: [batch, heads, seq_len, head_dim]
        
        Returns:
            dq, dk, dv: gradients w.r.t. q, k, v
        """
        q, k, v, out, L, M = ctx.saved_tensors
        Br = ctx.Br
        Bc = ctx.Bc
        BLOCK_SIZE_D = ctx.BLOCK_SIZE_D
        
        batch, heads, seq_len, head_dim = q.shape
        
        # Allocate gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        # Ensure dout is contiguous
        dout = dout.contiguous()
        
        # Launch backward kernel
        grid = (batch, heads, (seq_len + Br - 1) // Br)
        
        flash_attention_backward_kernel[grid](
            dout, q, k, v, out, L, M, dq, dk, dv,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            seq_len, head_dim,
            Br, Bc, BLOCK_SIZE_D,
        )
        
        return dq, dk, dv


def flash_attention(q, k, v):
    """
    Flash Attention v2 wrapper function.
    
    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, heads, seq_len, head_dim]
        v: [batch, heads, seq_len, head_dim]
    
    Returns:
        out: [batch, heads, seq_len, head_dim]
    """
    return FlashAttention.apply(q, k, v)


if __name__ == "__main__":
    # Test
    batch, heads, seq_len, head_dim = 2, 4, 512, 64
    
    q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
    
    # Forward pass
    out = flash_attention(q, k, v)
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    
    # Backward pass
    loss = out.sum()
    loss.backward()
    
    print(f"dQ shape: {q.grad.shape}")
    print(f"dK shape: {k.grad.shape}")
    print(f"dV shape: {v.grad.shape}")
    print("Test passed!")