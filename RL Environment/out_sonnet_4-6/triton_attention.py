"""
Flash Attention V2 – Triton implementation.

V2 design (Flash-Attn V2 paper, Dao et al. 2023):
  * Forward:  Outer loop = Q tiles (parallelized over grid).
              Inner loop = K/V tiles.
  * Backward: Outer loop = KV tiles (parallelized over grid).
              Inner loop = Q tiles.
  * Online softmax: running (m_i, l_i) in SRAM; never materialize S or P in HBM.

Performance:
  * float16 inputs: fp16 tensor-core dot products, fp32 accumulation → ~4x speedup.
  * float32 inputs: IEEE fp32 dot products for correctness (TF32 has ~3e-3 error).

Triton 3.x compatibility:
  * Loading 1D stats with boolean masks causes arith.select layout conflicts when
    those values are later broadcast to 2D alongside tl.dot outputs.
    Fix: pad M/L/Delta to BLOCK_M boundary; load without masks.
  * Delta_i = rowsum(dO_i * O_i) precomputed in a separate kernel before backward.
"""

import math
import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Forward kernel (float16 path) – uses fp16 tensor cores for speed
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd_kernel_fp16(
    Q, K, V, Out, M, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_mb, stride_mh, stride_ms,
    stride_lb, stride_lh, stride_ls,
    B, H, N, D: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Grid: (ceil(N/BLOCK_M), B*H)
    Flash-Attn V2: outer=Q tiles, inner=K/V tiles.
    Casts inputs to fp16 for tensor-core dot products; accumulates in fp32.
    """
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b  = pid_bh // H
    pid_h  = pid_bh  % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_base = Q   + pid_b * stride_qb + pid_h * stride_qh
    k_base = K   + pid_b * stride_kb + pid_h * stride_kh
    v_base = V   + pid_b * stride_vb + pid_h * stride_vh
    o_base = Out + pid_b * stride_ob + pid_h * stride_oh
    m_base = M   + pid_b * stride_mb + pid_h * stride_mh
    l_base = L   + pid_b * stride_lb + pid_h * stride_lh

    q_f16 = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=offs_m[:, None] < N, other=0.0).to(tl.float16)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, D],          dtype=tl.float32)

    for j in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < N

        k_f16 = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                        mask=kv_mask, other=0.0).to(tl.float16)
        v_f16 = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                        mask=kv_mask, other=0.0).to(tl.float16)

        s = tl.dot(q_f16, tl.trans(k_f16), out_dtype=tl.float32) * scale
        s = tl.where(offs_n[None, :] < N, s, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha  = tl.exp(m_i - m_new)
        p      = tl.exp(s - m_new[:, None])
        l_i    = alpha * l_i + tl.sum(p, axis=1)
        acc    = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v_f16, out_dtype=tl.float32)
        m_i    = m_new

    acc = acc / l_i[:, None]

    tl.store(o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)
    sm = offs_m < N
    tl.store(m_base + offs_m * stride_ms, m_i, mask=sm)
    tl.store(l_base + offs_m * stride_ls, l_i, mask=sm)


# ──────────────────────────────────────────────────────────────────────────────
# Forward kernel (float32 path) – IEEE fp32 for accuracy
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd_kernel_fp32(
    Q, K, V, Out, M, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_mb, stride_mh, stride_ms,
    stride_lb, stride_lh, stride_ls,
    B, H, N, D: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Grid: (ceil(N/BLOCK_M), B*H)
    Flash-Attn V2: outer=Q tiles, inner=K/V tiles.
    Uses IEEE fp32 precision throughout.
    """
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b  = pid_bh // H
    pid_h  = pid_bh  % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_base = Q   + pid_b * stride_qb + pid_h * stride_qh
    k_base = K   + pid_b * stride_kb + pid_h * stride_kh
    v_base = V   + pid_b * stride_vb + pid_h * stride_vh
    o_base = Out + pid_b * stride_ob + pid_h * stride_oh
    m_base = M   + pid_b * stride_mb + pid_h * stride_mh
    l_base = L   + pid_b * stride_lb + pid_h * stride_lh

    q_mask = offs_m[:, None] < N
    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=q_mask, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, D],          dtype=tl.float32)

    for j in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < N

        k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=kv_mask, other=0.0).to(tl.float32)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        s = tl.where(offs_n[None, :] < N, s, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha  = tl.exp(m_i - m_new)
        p      = tl.exp(s - m_new[:, None])
        l_i    = alpha * l_i + tl.sum(p, axis=1)
        acc    = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i    = m_new

    acc = acc / l_i[:, None]

    tl.store(o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)
    sm = offs_m < N
    tl.store(m_base + offs_m * stride_ms, m_i, mask=sm)
    tl.store(l_base + offs_m * stride_ls, l_i, mask=sm)


# ──────────────────────────────────────────────────────────────────────────────
# Delta precomputation kernel
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _compute_delta(
    DO, O, Delta,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_ob,  stride_oh,  stride_om,  stride_od,
    stride_db,  stride_dh,  stride_dm,
    B, H, N, D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute Delta_i = rowsum(dO_i * O_i)."""
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b  = pid_bh // H
    pid_h  = pid_bh  % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    do_base = DO    + pid_b * stride_dob + pid_h * stride_doh
    o_base  = O     + pid_b * stride_ob  + pid_h * stride_oh
    d_base  = Delta + pid_b * stride_db  + pid_h * stride_dh

    mask = offs_m[:, None] < N
    do = tl.load(do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                 mask=mask, other=0.0).to(tl.float32)
    o  = tl.load(o_base  + offs_m[:, None] * stride_om  + offs_d[None, :] * stride_od,
                 mask=mask, other=0.0).to(tl.float32)

    tl.store(d_base + offs_m * stride_dm, tl.sum(do * o, axis=1), mask=offs_m < N)


# ──────────────────────────────────────────────────────────────────────────────
# Backward kernel – IEEE fp32 for accuracy
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bwd_kernel(
    Q, K, V, Out, DO,
    M, L, Delta,
    DQ, DK, DV,
    stride_qb,  stride_qh,  stride_qm,  stride_qd,
    stride_kb,  stride_kh,  stride_kn,  stride_kd,
    stride_vb,  stride_vh,  stride_vn,  stride_vd,
    stride_ob,  stride_oh,  stride_om,  stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_mb,  stride_mh,  stride_ms,
    stride_lb,  stride_lh,  stride_ls,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    B, H, N, D: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Grid: (ceil(N/BLOCK_N), B*H)

    For each KV tile j, iterate Q tiles i:
      P_ij   = softmax_row(Q_i K_j^T * scale)
      dV_j  += P_ij^T dO_i
      dS_ij  = P_ij * (dO_i V_j^T - Delta_i)
      dK_j  += dS_ij^T Q_i * scale
      dQ_i  += dS_ij K_j * scale  (atomic)

    Uses IEEE fp32. M/L/Delta are pre-padded, loaded without masks.
    """
    pid_n  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b  = pid_bh // H
    pid_h  = pid_bh  % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_base  = Q     + pid_b * stride_qb  + pid_h * stride_qh
    k_base  = K     + pid_b * stride_kb  + pid_h * stride_kh
    v_base  = V     + pid_b * stride_vb  + pid_h * stride_vh
    do_base = DO    + pid_b * stride_dob + pid_h * stride_doh
    m_base  = M     + pid_b * stride_mb  + pid_h * stride_mh
    l_base  = L     + pid_b * stride_lb  + pid_h * stride_lh
    d_base  = Delta + pid_b * stride_deltab + pid_h * stride_deltah
    dq_base = DQ    + pid_b * stride_dqb + pid_h * stride_dqh
    dk_base = DK    + pid_b * stride_dkb + pid_h * stride_dkh
    dv_base = DV    + pid_b * stride_dvb + pid_h * stride_dvh

    kv_mask = offs_n[:, None] < N
    k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=kv_mask, other=0.0).to(tl.float32)
    v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=kv_mask, other=0.0).to(tl.float32)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    for i in range(0, tl.cdiv(N, BLOCK_M)):
        offs_m = i * BLOCK_M + tl.arange(0, BLOCK_M)
        qm_mask = offs_m[:, None] < N

        q  = tl.load(q_base  + offs_m[:, None] * stride_qm  + offs_d[None, :] * stride_qd,
                     mask=qm_mask, other=0.0).to(tl.float32)
        do = tl.load(do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                     mask=qm_mask, other=0.0).to(tl.float32)

        # Load stats without masks (pre-padded to BLOCK_M boundary)
        m_i = tl.load(m_base + offs_m * stride_ms)
        l_i = tl.load(l_base + offs_m * stride_ls)
        d_i = tl.load(d_base + offs_m * stride_deltam)

        # Recompute P
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        s = tl.where(offs_n[None, :] < N, s, float('-inf'))
        p = tl.exp(s - m_i[:, None]) / l_i[:, None]

        # dV += P^T dO
        dv += tl.dot(tl.trans(p), do, input_precision="ieee")

        # dS = P * (dO V^T - Delta_i)
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - d_i[:, None])

        # dK += dS^T Q * scale
        dk += tl.dot(tl.trans(ds), q, input_precision="ieee") * scale

        # dQ += dS K * scale (atomic)
        tl.atomic_add(
            dq_base + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd,
            tl.dot(ds, k, input_precision="ieee") * scale,
            mask=qm_mask,
        )

    tl.store(dk_base + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
             dk, mask=kv_mask)
    tl.store(dv_base + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
             dv, mask=kv_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Block-size selection
# ──────────────────────────────────────────────────────────────────────────────

def _fwd_block_sizes_fp16(N: int, D: int):
    """Forward fp16 path: large blocks; fp16 tiles are compact."""
    if D <= 64:
        BM, BN = 128, 64
    else:
        BM, BN = 32, 32
    np2 = triton.next_power_of_2(N)
    BM  = max(min(BM, np2), 16)
    BN  = max(min(BN, np2), 16)
    return BM, BN


def _fwd_block_sizes_fp32(N: int, D: int):
    """Forward fp32 path: smaller blocks (fp32 tiles are 2x larger)."""
    if D <= 64:
        BM, BN = 64, 64
    else:
        BM, BN = 32, 32
    np2 = triton.next_power_of_2(N)
    BM  = max(min(BM, np2), 16)
    BN  = max(min(BN, np2), 16)
    return BM, BN


def _bwd_block_sizes(N: int, D: int):
    """Backward: IEEE fp32 + more temporaries → smaller blocks."""
    if D <= 64:
        BM, BN = 64, 32
    else:
        BM, BN = 16, 16
    np2 = triton.next_power_of_2(N)
    BM  = max(min(BM, np2), 16)
    BN  = max(min(BN, np2), 16)
    return BM, BN


def _delta_block_size(N: int):
    return max(min(128, triton.next_power_of_2(N)), 16)


def _pad_to(t: torch.Tensor, size: int, fill: float = 0.0) -> torch.Tensor:
    cur = t.shape[-1]
    if cur >= size:
        return t
    pad = torch.full((*t.shape[:-1], size - cur), fill, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Public Python API
# ──────────────────────────────────────────────────────────────────────────────

def flash_attention_forward(q, k, v):
    """
    Flash Attention V2 forward pass.

    Args:
        q, k, v : [B, H, N, D]   float16 or float32

    Returns:
        out : [B, H, N, D]
        m   : [B, H, N]   per-row max logit (float32)
        l   : [B, H, N]   per-row softmax denominator (float32)
    """
    B, H, N, D = q.shape
    assert (D & (D - 1)) == 0, f"head_dim must be a power of 2, got {D}"
    scale = 1.0 / math.sqrt(D)

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    m   = torch.empty(B, H, N, dtype=torch.float32, device=q.device)
    l   = torch.empty(B, H, N, dtype=torch.float32, device=q.device)

    if q.dtype == torch.float16:
        # Use fast fp16 tensor-core path
        BLOCK_M, BLOCK_N = _fwd_block_sizes_fp16(N, D)
        kernel = _fwd_kernel_fp16
    else:
        # Use IEEE fp32 path for accuracy
        BLOCK_M, BLOCK_N = _fwd_block_sizes_fp32(N, D)
        kernel = _fwd_kernel_fp32

    grid = (triton.cdiv(N, BLOCK_M), B * H)
    kernel[grid](
        q, k, v, out, m, l,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        m.stride(0),   m.stride(1),   m.stride(2),
        l.stride(0),   l.stride(1),   l.stride(2),
        B, H, N, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out, m, l


def flash_attention_backward(do, q, k, v, out, m, l):
    """
    Flash Attention V2 backward pass.

    Args:
        do      : [B, H, N, D]   upstream gradient
        q, k, v : [B, H, N, D]   original inputs
        out     : [B, H, N, D]   forward output
        m, l    : [B, H, N]      statistics from forward (float32)

    Returns:
        dq, dk, dv : [B, H, N, D]
    """
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)

    do  = do.contiguous()
    q   = q.contiguous()
    k   = k.contiguous()
    v   = v.contiguous()
    out = out.contiguous()

    dq = torch.zeros_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    BLOCK_M, BLOCK_N = _bwd_block_sizes(N, D)
    BLOCK_DELTA = _delta_block_size(N)

    # Precompute Delta_i = rowsum(dO_i * O_i)
    delta = torch.empty(B, H, N, dtype=torch.float32, device=q.device)
    _compute_delta[(triton.cdiv(N, BLOCK_DELTA), B * H)](
        do, out, delta,
        do.stride(0),  do.stride(1),  do.stride(2),  do.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        B, H, N, D, BLOCK_M=BLOCK_DELTA,
    )

    # Pad M, L, Delta to BLOCK_M boundary
    N_pad = triton.cdiv(N, BLOCK_M) * BLOCK_M
    if N_pad > N:
        m_pad     = _pad_to(m,     N_pad, fill=0.0).contiguous()
        l_pad     = _pad_to(l,     N_pad, fill=1.0).contiguous()
        delta_pad = _pad_to(delta, N_pad, fill=0.0).contiguous()
    else:
        m_pad     = m.contiguous()
        l_pad     = l.contiguous()
        delta_pad = delta.contiguous()

    _bwd_kernel[(triton.cdiv(N, BLOCK_N), B * H)](
        q, k, v, out, do,
        m_pad, l_pad, delta_pad,
        dq, dk, dv,
        q.stride(0),   q.stride(1),   q.stride(2),   q.stride(3),
        k.stride(0),   k.stride(1),   k.stride(2),   k.stride(3),
        v.stride(0),   v.stride(1),   v.stride(2),   v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        do.stride(0),  do.stride(1),  do.stride(2),  do.stride(3),
        m_pad.stride(0),     m_pad.stride(1),     m_pad.stride(2),
        l_pad.stride(0),     l_pad.stride(1),     l_pad.stride(2),
        delta_pad.stride(0), delta_pad.stride(1), delta_pad.stride(2),
        dq.stride(0),  dq.stride(1),  dq.stride(2),  dq.stride(3),
        dk.stride(0),  dk.stride(1),  dk.stride(2),  dk.stride(3),
        dv.stride(0),  dv.stride(1),  dv.stride(2),  dv.stride(3),
        B, H, N, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return dq, dk, dv


# ──────────────────────────────────────────────────────────────────────────────
# Differentiable torch.autograd wrapper
# ──────────────────────────────────────────────────────────────────────────────

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out, m, l = flash_attention_forward(q, k, v)
        ctx.save_for_backward(q, k, v, out, m, l)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, m, l = ctx.saved_tensors
        dq, dk, dv = flash_attention_backward(do, q, k, v, out, m, l)
        return dq, dk, dv


# ──────────────────────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/workspace/ajay_jagannath_pm_env_slim/env_data")

    torch.manual_seed(42)
    device = "cuda"
    TOL    = 1e-3

    def ref_attn(q, k, v):
        s = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.size(-1))
        return torch.matmul(torch.softmax(s, dim=-1), v.float())

    print("=" * 60)
    print("Forward correctness")
    print("=" * 60)
    for dtype in [torch.float32, torch.float16]:
        print(f"  dtype={dtype}:")
        for N in [32, 64, 128, 256, 512, 1024]:
            B, H, D = 2, 4, 64
            q = torch.randn(B, H, N, D, device=device, dtype=dtype)
            k = torch.randn(B, H, N, D, device=device, dtype=dtype)
            v = torch.randn(B, H, N, D, device=device, dtype=dtype)
            out_tri, _, _ = flash_attention_forward(q, k, v)
            out_ref       = ref_attn(q, k, v).to(dtype)
            diff = (out_tri.float() - out_ref.float()).abs().max().item()
            print(f"    N={N:5d}: {diff:.2e}  {'✓' if diff < TOL else '✗ FAIL'}")

    print()
    print("=" * 60)
    print("Backward correctness (float32)")
    print("=" * 60)
    for N in [32, 64, 128, 256, 512, 1024]:
        B, H, D = 2, 4, 64
        q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

        qr = q.clone().requires_grad_(True)
        kr = k.clone().requires_grad_(True)
        vr = v.clone().requires_grad_(True)
        ref_attn(qr, kr, vr).sum().backward()

        qt = q.clone().requires_grad_(True)
        kt = k.clone().requires_grad_(True)
        vt = v.clone().requires_grad_(True)
        FlashAttention.apply(qt, kt, vt).sum().backward()

        dd = {'dq': (qt.grad - qr.grad).abs().max().item(),
              'dk': (kt.grad - kr.grad).abs().max().item(),
              'dv': (vt.grad - vr.grad).abs().max().item()}
        ok = all(d < TOL for d in dd.values())
        print(f"  N={N:5d}: dq={dd['dq']:.2e} dk={dd['dk']:.2e} dv={dd['dv']:.2e}  "
              f"{'✓' if ok else '✗ FAIL'}")

    print()
    print("=" * 60)
    print("Performance benchmark")
    print("=" * 60)
    def bench(fn, n_warmup=10, n_iters=100):
        for _ in range(n_warmup): fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters): fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_iters

    for N in [512, 1024, 2048, 4096]:
        B, H, D = 2, 8, 64
        for dtype, label in [(torch.float16, "fp16"), (torch.float32, "fp32")]:
            q = torch.randn(B, H, N, D, device=device, dtype=dtype)
            k = torch.randn(B, H, N, D, device=device, dtype=dtype)
            v = torch.randn(B, H, N, D, device=device, dtype=dtype)
            t_ref = bench(lambda: torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2,-1))/math.sqrt(D), dim=-1), v))
            t_tri = bench(lambda: flash_attention_forward(q, k, v))
            print(f"  N={N:5d} {label}: ref={t_ref:.3f}ms  triton={t_tri:.3f}ms  speedup={t_ref/t_tri:.2f}x")
