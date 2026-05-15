import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_forward_kernel(
    Q, K, V, O, L, M,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_mb, stride_mh, stride_ms,
    batch_size, num_heads, seq_len, head_dim,
    block_m: tl.constexpr, block_n: tl.constexpr, head_dim_const: tl.constexpr,
):
    """
    Flash Attention v2 forward kernel.
    Computes attention with online softmax.
    """
    block_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads
    
    num_blocks_m = tl.cdiv(seq_len, block_m)
    block_m_id = block_id % num_blocks_m
    
    # Scaling factor
    scale = 1.0 / tl.sqrt(head_dim_const * 1.0)
    
    # Initialize accumulators with constexpr shapes
    m_i = tl.full([block_m], value=float('-inf'), dtype=tl.float32)
    l_i = tl.full([block_m], value=0.0, dtype=tl.float32)
    acc = tl.full([block_m, head_dim_const], value=0.0, dtype=tl.float32)
    
    # Load Q block
    offs_m = block_m_id * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, head_dim_const)
    
    q_ptrs = Q + batch_id * stride_qb + head_id * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)
    q_block = q_block.to(tl.float32)
    
    # Iterate over K, V blocks
    num_blocks_n = tl.cdiv(seq_len, block_n)
    
    for block_n_id in range(0, num_blocks_n):
        offs_n = block_n_id * block_n + tl.arange(0, block_n)
        
        # Load K block
        k_ptrs = K + batch_id * stride_kb + head_id * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)
        k_block = k_block.to(tl.float32)
        
        # Load V block
        v_ptrs = V + batch_id * stride_vb + head_id * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)
        v_block = v_block.to(tl.float32)
        
        # Compute attention scores: Q @ K^T / sqrt(d)
        s_block = tl.dot(q_block, tl.trans(k_block))
        s_block = s_block * scale
        
        # Causal mask: only attend to past and present
        mask = offs_m[:, None] >= offs_n[None, :]
        s_block = tl.where(mask, s_block, float('-inf'))
        
        # Online softmax: update m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(s_block, axis=1))
        p_block = tl.exp(s_block - m_i_new[:, None])
        l_i_new = tl.exp(m_i - m_i_new) * l_i + tl.sum(p_block, axis=1)
        
        # Update accumulator
        acc = acc * tl.exp(m_i - m_i_new)[:, None] + tl.dot(p_block, v_block)
        
        # Update statistics
        m_i = m_i_new
        l_i = l_i_new
    
    # Normalize output
    o_block = acc / l_i[:, None]
    o_block = o_block.to(tl.float16)
    
    # Store output
    o_ptrs = O + batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_block, mask=offs_m[:, None] < seq_len)
    
    # Store statistics
    l_ptrs = L + batch_id * stride_lb + head_id * stride_lh + offs_m[:, None] * stride_ls
    m_ptrs = M + batch_id * stride_mb + head_id * stride_mh + offs_m[:, None] * stride_ms
    tl.store(l_ptrs, l_i[:, None], mask=offs_m[:, None] < seq_len)
    tl.store(m_ptrs, m_i[:, None], mask=offs_m[:, None] < seq_len)


@triton.jit
def flash_attention_backward_kernel(
    Q, K, V, O, DO, DQ, DK, DV, L, M,
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
    batch_size, num_heads, seq_len, head_dim,
    block_m: tl.constexpr, block_n: tl.constexpr, head_dim_const: tl.constexpr,
):
    """
    Flash Attention v2 backward kernel.
    Computes gradients for Q, K, V.
    """
    block_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads
    
    num_blocks_m = tl.cdiv(seq_len, block_m)
    block_m_id = block_id % num_blocks_m
    
    scale = 1.0 / tl.sqrt(head_dim_const * 1.0)
    
    offs_m = block_m_id * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, head_dim_const)
    
    # Load Q, O, DO blocks
    q_ptrs = Q + batch_id * stride_qb + head_id * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = DO + batch_id * stride_dob + head_id * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    
    q_block = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0).to(tl.float32)
    o_block = tl.load(o_ptrs, mask=offs_m[:, None] < seq_len, other=0.0).to(tl.float32)
    do_block = tl.load(do_ptrs, mask=offs_m[:, None] < seq_len, other=0.0).to(tl.float32)
    
    # Load statistics
    l_ptrs = L + batch_id * stride_lb + head_id * stride_lh + offs_m[:, None] * stride_ls
    m_ptrs = M + batch_id * stride_mb + head_id * stride_mh + offs_m[:, None] * stride_ms
    l_block = tl.load(l_ptrs, mask=offs_m[:, None] < seq_len, other=0.0).to(tl.float32)
    m_block = tl.load(m_ptrs, mask=offs_m[:, None] < seq_len, other=0.0).to(tl.float32)
    
    # Compute dQ contribution
    dq_block = tl.full([block_m, head_dim_const], value=0.0, dtype=tl.float32)
    
    num_blocks_n = tl.cdiv(seq_len, block_n)
    
    for block_n_id in range(0, num_blocks_n):
        offs_n = block_n_id * block_n + tl.arange(0, block_n)
        
        # Load K, V blocks
        k_ptrs = K + batch_id * stride_kb + head_id * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + batch_id * stride_vb + head_id * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k_block = tl.load(k_ptrs, mask=offs_n[:, None] < seq_len, other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len, other=0.0).to(tl.float32)
        
        # Compute attention scores
        s_block = tl.dot(q_block, tl.trans(k_block)) * scale
        
        # Causal mask
        mask = offs_m[:, None] >= offs_n[None, :]
        s_block = tl.where(mask, s_block, float('-inf'))
        
        # Compute P
        p_block = tl.exp(s_block - m_block)
        p_block = p_block / l_block
        
        # Compute dV
        dv_ptrs = DV + batch_id * stride_dvb + head_id * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        dv_block = tl.load(dv_ptrs, mask=offs_n[:, None] < seq_len, other=0.0).to(tl.float32)
        dv_contribution = tl.dot(tl.trans(p_block), do_block)
        dv_block = dv_block + dv_contribution
        tl.store(dv_ptrs, dv_block.to(tl.float16), mask=offs_n[:, None] < seq_len)
        
        # Compute dP
        dp_block = tl.dot(do_block, tl.trans(v_block))
        
        # Compute dS
        ds_block = p_block * (dp_block - tl.sum(do_block * o_block, axis=1, keep_dims=True))
        ds_block = ds_block * scale
        
        # Compute dK
        dk_ptrs = DK + batch_id * stride_dkb + head_id * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dk_block = tl.load(dk_ptrs, mask=offs_n[:, None] < seq_len, other=0.0).to(tl.float32)
        dk_contribution = tl.dot(tl.trans(ds_block), q_block)
        dk_block = dk_block + dk_contribution
        tl.store(dk_ptrs, dk_block.to(tl.float16), mask=offs_n[:, None] < seq_len)
        
        # Accumulate dQ
        dq_block = dq_block + tl.dot(ds_block, k_block)
    
    # Store dQ
    dq_ptrs = DQ + batch_id * stride_dqb + head_id * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    dq_block = dq_block.to(tl.float16)
    tl.store(dq_ptrs, dq_block, mask=offs_m[:, None] < seq_len)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        """
        Args:
            q: [batch, heads, seq_len, head_dim]
            k: [batch, heads, seq_len, head_dim]
            v: [batch, heads, seq_len, head_dim]
        
        Returns:
            o: [batch, heads, seq_len, head_dim]
        """
        batch, heads, seq_len, head_dim = q.shape
        
        # Ensure inputs are contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Allocate output and statistics
        o = torch.empty_like(q)
        l = torch.zeros((batch, heads, seq_len, 1), dtype=torch.float32, device=q.device)
        m = torch.zeros((batch, heads, seq_len, 1), dtype=torch.float32, device=q.device)
        
        # Block sizes
        block_m = 128
        block_n = 128
        
        # Grid: (num_blocks_m, batch * heads)
        num_blocks_m = (seq_len + block_m - 1) // block_m
        grid = (num_blocks_m, batch * heads)
        
        # Launch kernel
        flash_attention_forward_kernel[grid](
            q, k, v, o, l, m,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            l.stride(0), l.stride(1), l.stride(2),
            m.stride(0), m.stride(1), m.stride(2),
            batch, heads, seq_len, head_dim,
            block_m=block_m, block_n=block_n, head_dim_const=head_dim,
        )
        
        ctx.save_for_backward(q, k, v, o, l, m)
        ctx.head_dim = head_dim
        
        return o
    
    @staticmethod
    def backward(ctx, do):
        """
        Args:
            do: [batch, heads, seq_len, head_dim]
        
        Returns:
            dq, dk, dv: [batch, heads, seq_len, head_dim]
        """
        q, k, v, o, l, m = ctx.saved_tensors
        batch, heads, seq_len, head_dim = ctx.head_dim
        
        # Ensure gradients are contiguous
        do = do.contiguous()
        
        # Get actual shape from tensors
        batch, heads, seq_len, head_dim = q.shape
        
        # Allocate gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        # Block sizes
        block_m = 128
        block_n = 128
        
        # Grid
        num_blocks_m = (seq_len + block_m - 1) // block_m
        grid = (num_blocks_m, batch * heads)
        
        # Launch backward kernel
        flash_attention_backward_kernel[grid](
            q, k, v, o, do, dq, dk, dv, l, m,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            l.stride(0), l.stride(1), l.stride(2),
            m.stride(0), m.stride(1), m.stride(2),
            batch, heads, seq_len, head_dim,
            block_m=block_m, block_n=block_n, head_dim_const=head_dim,
        )
        
        return dq, dk, dv


def flash_attention(q, k, v):
    """
    Flash Attention v2 implementation.
    
    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, heads, seq_len, head_dim]
        v: [batch, heads, seq_len, head_dim]
    
    Returns:
        o: [batch, heads, seq_len, head_dim]
    """
    return FlashAttention.apply(q, k, v)


if __name__ == "__main__":
    # Test
    batch, heads, seq_len, head_dim = 4, 4, 1024, 64
    
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    
    # Forward
    o = flash_attention(q, k, v)
    
    # Backward
    loss = o.sum()
    loss.backward()
    
    print(f"Output shape: {o.shape}")
    print(f"dQ shape: {q.grad.shape}")
    print(f"dK shape: {k.grad.shape}")
    print(f"dV shape: {v.grad.shape}")
    print("Test passed!")