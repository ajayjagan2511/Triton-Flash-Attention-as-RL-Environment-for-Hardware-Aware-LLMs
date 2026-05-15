import math
import torch
import triton
import triton.language as tl


# ============================================================================
# Flash Attention V2 Forward Pass Kernel
# ============================================================================

@triton.jit
def flash_attention_forward_kernel(
    Q, K, V, O, m, l,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kq, stride_kd,
    stride_vb, stride_vh, stride_vq, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_mb, stride_mh, stride_mq,
    stride_lb, stride_lh, stride_lq,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Flash Attention V2 Forward Pass
    """
    
    block_q_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    q_start = block_q_idx * BLOCK_SIZE_Q
    q_offset = q_start + tl.arange(0, BLOCK_SIZE_Q)
    
    o = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_Q,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
    
    q_mask = q_offset < seq_len
    
    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + q_offset[:, None] * stride_qq + tl.arange(0, HEAD_DIM)[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    
    num_kv_blocks = tl.cdiv(seq_len, BLOCK_SIZE_K)
    scale = 1.0 / tl.sqrt(HEAD_DIM * 1.0)
    
    for kv_block_idx in range(0, num_kv_blocks):
        k_start = kv_block_idx * BLOCK_SIZE_K
        k_offset = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offset < seq_len
        
        k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + k_offset[:, None] * stride_kq + tl.arange(0, HEAD_DIM)[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        
        v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + k_offset[:, None] * stride_vq + tl.arange(0, HEAD_DIM)[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        
        s_block = tl.dot(q_block, tl.trans(k_block)) * scale
        s_block = tl.where(k_mask[None, :], s_block, float('-inf'))
        
        s_max = tl.max(s_block, axis=1)
        m_i_new = tl.maximum(m_i, s_max)
        
        p_block = tl.exp(s_block - m_i_new[:, None])
        
        l_i_new = tl.exp(m_i - m_i_new) * l_i + tl.sum(p_block, axis=1)
        
        o = tl.exp(m_i - m_i_new)[:, None] * o
        
        o = o + tl.dot(p_block, v_block)
        
        m_i = m_i_new
        l_i = l_i_new
    
    o = o / tl.maximum(l_i[:, None], 1e-8)
    
    o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + q_offset[:, None] * stride_oq + tl.arange(0, HEAD_DIM)[None, :] * stride_od
    tl.store(o_ptrs, o, mask=q_mask[:, None])
    
    m_ptrs = m + batch_idx * stride_mb + head_idx * stride_mh + q_offset * stride_mq
    l_ptrs = l + batch_idx * stride_lb + head_idx * stride_lh + q_offset * stride_lq
    tl.store(m_ptrs, m_i, mask=q_mask)
    tl.store(l_ptrs, l_i, mask=q_mask)


def flash_attention_forward(q, k, v):
    """
    Flash Attention V2 Forward Pass
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_K = 64
    
    out = torch.zeros_like(q, dtype=q.dtype)
    m = torch.full((batch_size, num_heads, seq_len), float('-inf'), dtype=torch.float32, device=q.device)
    l = torch.zeros((batch_size, num_heads, seq_len), dtype=torch.float32, device=q.device)
    
    num_q_blocks = triton.cdiv(seq_len, BLOCK_SIZE_Q)
    grid = (num_q_blocks, batch_size, num_heads)
    
    flash_attention_forward_kernel[grid](
        q, k, v, out, m, l,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        l.stride(0), l.stride(1), l.stride(2),
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HEAD_DIM=head_dim,
        num_warps=4,
        num_stages=2,
    )
    
    return out, m, l


# ============================================================================
# Flash Attention V2 Backward Pass Kernel - dQ only
# ============================================================================

@triton.jit
def flash_attention_backward_dq_kernel(
    dO, Q, K, V, O, m, l, dQ,
    stride_dob, stride_doh, stride_doq, stride_dod,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kq, stride_kd,
    stride_vb, stride_vh, stride_vq, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_mb, stride_mh, stride_mq,
    stride_lb, stride_lh, stride_lq,
    stride_dqb, stride_dqh, stride_dqq, stride_dqd,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Flash Attention V2 Backward - dQ computation
    """
    
    block_q_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    q_start = block_q_idx * BLOCK_SIZE_Q
    q_offset = q_start + tl.arange(0, BLOCK_SIZE_Q)
    q_mask = q_offset < seq_len
    
    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + q_offset[:, None] * stride_qq + tl.arange(0, HEAD_DIM)[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    
    o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + q_offset[:, None] * stride_oq + tl.arange(0, HEAD_DIM)[None, :] * stride_od
    o_block = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0)
    
    do_ptrs = dO + batch_idx * stride_dob + head_idx * stride_doh + q_offset[:, None] * stride_doq + tl.arange(0, HEAD_DIM)[None, :] * stride_dod
    do_block = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)
    
    m_ptrs = m + batch_idx * stride_mb + head_idx * stride_mh + q_offset * stride_mq
    l_ptrs = l + batch_idx * stride_lb + head_idx * stride_lh + q_offset * stride_lq
    m_block = tl.load(m_ptrs, mask=q_mask, other=0.0)
    l_block = tl.load(l_ptrs, mask=q_mask, other=0.0)
    
    D_i = tl.sum(do_block * o_block, axis=1)
    
    dq_block = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)
    
    num_kv_blocks = tl.cdiv(seq_len, BLOCK_SIZE_K)
    scale = 1.0 / tl.sqrt(HEAD_DIM * 1.0)
    
    for kv_block_idx in range(0, num_kv_blocks):
        k_start = kv_block_idx * BLOCK_SIZE_K
        k_offset = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offset < seq_len
        
        k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + k_offset[:, None] * stride_kq + tl.arange(0, HEAD_DIM)[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        
        v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + k_offset[:, None] * stride_vq + tl.arange(0, HEAD_DIM)[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        
        s_block = tl.dot(q_block, tl.trans(k_block)) * scale
        s_block = tl.where(k_mask[None, :], s_block, float('-inf'))
        
        p_unnorm = tl.exp(s_block - m_block[:, None])
        p_unnorm = tl.where(k_mask[None, :], p_unnorm, 0.0)
        
        l_safe = tl.maximum(l_block[:, None], 1e-8)
        p_block = p_unnorm / l_safe
        
        dp_block = tl.dot(do_block, tl.trans(v_block))
        
        ds_block = p_block * (dp_block - D_i[:, None])
        
        dq_block = dq_block + tl.dot(ds_block, k_block) * scale
    
    dq_ptrs = dQ + batch_idx * stride_dqb + head_idx * stride_dqh + q_offset[:, None] * stride_dqq + tl.arange(0, HEAD_DIM)[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_block, mask=q_mask[:, None])


# ============================================================================
# Flash Attention V2 Backward Pass Kernel - dK/dV
# ============================================================================

@triton.jit
def flash_attention_backward_dkv_kernel(
    dO, Q, K, V, O, m, l, dK, dV,
    stride_dob, stride_doh, stride_doq, stride_dod,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kq, stride_kd,
    stride_vb, stride_vh, stride_vq, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_mb, stride_mh, stride_mq,
    stride_lb, stride_lh, stride_lq,
    stride_dkb, stride_dkh, stride_dkq, stride_dkd,
    stride_dvb, stride_dvh, stride_dvq, stride_dvd,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Flash Attention V2 Backward - dK/dV computation
    Parallelizes over K dimension
    """
    
    block_k_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    k_start = block_k_idx * BLOCK_SIZE_K
    k_offset = k_start + tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offset < seq_len
    
    k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + k_offset[:, None] * stride_kq + tl.arange(0, HEAD_DIM)[None, :] * stride_kd
    k_block = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
    
    v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + k_offset[:, None] * stride_vq + tl.arange(0, HEAD_DIM)[None, :] * stride_vd
    v_block = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
    
    dk_block = tl.zeros((BLOCK_SIZE_K, HEAD_DIM), dtype=tl.float32)
    dv_block = tl.zeros((BLOCK_SIZE_K, HEAD_DIM), dtype=tl.float32)
    
    num_q_blocks = tl.cdiv(seq_len, BLOCK_SIZE_Q)
    scale = 1.0 / tl.sqrt(HEAD_DIM * 1.0)
    
    for q_block_idx in range(0, num_q_blocks):
        q_start = q_block_idx * BLOCK_SIZE_Q
        q_offset = q_start + tl.arange(0, BLOCK_SIZE_Q)
        q_mask = q_offset < seq_len
        
        q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + q_offset[:, None] * stride_qq + tl.arange(0, HEAD_DIM)[None, :] * stride_qd
        q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        
        do_ptrs = dO + batch_idx * stride_dob + head_idx * stride_doh + q_offset[:, None] * stride_doq + tl.arange(0, HEAD_DIM)[None, :] * stride_dod
        do_block = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)
        
        o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + q_offset[:, None] * stride_oq + tl.arange(0, HEAD_DIM)[None, :] * stride_od
        o_block = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0)
        
        m_ptrs = m + batch_idx * stride_mb + head_idx * stride_mh + q_offset * stride_mq
        l_ptrs = l + batch_idx * stride_lb + head_idx * stride_lh + q_offset * stride_lq
        m_block = tl.load(m_ptrs, mask=q_mask, other=0.0)
        l_block = tl.load(l_ptrs, mask=q_mask, other=0.0)
        
        D_block = tl.sum(do_block * o_block, axis=1)
        
        s_block = tl.dot(q_block, tl.trans(k_block)) * scale
        s_block = tl.where(k_mask[None, :], s_block, float('-inf'))
        
        p_unnorm = tl.exp(s_block - m_block[:, None])
        p_unnorm = tl.where(k_mask[None, :], p_unnorm, 0.0)
        
        l_safe = tl.maximum(l_block[:, None], 1e-8)
        p_block = p_unnorm / l_safe
        
        dp_block = tl.dot(do_block, tl.trans(v_block))
        ds_block = p_block * (dp_block - D_block[:, None])
        
        dk_block = dk_block + tl.dot(tl.trans(ds_block), q_block) * scale
        dv_block = dv_block + tl.dot(tl.trans(p_block), do_block)
    
    dk_ptrs = dK + batch_idx * stride_dkb + head_idx * stride_dkh + k_offset[:, None] * stride_dkq + tl.arange(0, HEAD_DIM)[None, :] * stride_dkd
    dv_ptrs = dV + batch_idx * stride_dvb + head_idx * stride_dvh + k_offset[:, None] * stride_dvq + tl.arange(0, HEAD_DIM)[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk_block, mask=k_mask[:, None])
    tl.store(dv_ptrs, dv_block, mask=k_mask[:, None])


def flash_attention_backward(dO, Q, K, V, O, m, l):
    """
    Flash Attention V2 Backward Pass
    """
    assert dO.dim() == 4
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    dO = dO.contiguous()
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    O = O.contiguous()
    m = m.contiguous()
    l = l.contiguous()
    
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_K = 64
    
    dQ = torch.zeros_like(Q, dtype=Q.dtype)
    dK = torch.zeros_like(K, dtype=K.dtype)
    dV = torch.zeros_like(V, dtype=V.dtype)
    
    # Compute dQ
    num_q_blocks = triton.cdiv(seq_len, BLOCK_SIZE_Q)
    grid_q = (num_q_blocks, batch_size, num_heads)
    
    flash_attention_backward_dq_kernel[grid_q](
        dO, Q, K, V, O, m, l, dQ,
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        l.stride(0), l.stride(1), l.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HEAD_DIM=head_dim,
        num_warps=4,
        num_stages=2,
    )
    
    # Compute dK and dV
    num_k_blocks = triton.cdiv(seq_len, BLOCK_SIZE_K)
    grid_k = (num_k_blocks, batch_size, num_heads)
    
    flash_attention_backward_dkv_kernel[grid_k](
        dO, Q, K, V, O, m, l, dK, dV,
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        l.stride(0), l.stride(1), l.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HEAD_DIM=head_dim,
        num_warps=4,
        num_stages=2,
    )
    
    return dQ, dK, dV


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out, m, l = flash_attention_forward(q, k, v)
        ctx.save_for_backward(q, k, v, out, m, l)
        return out
    
    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, m, l = ctx.saved_tensors
        dq, dk, dv = flash_attention_backward(dout, q, k, v, out, m, l)
        return dq, dk, dv
