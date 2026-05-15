import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER = REPO_ROOT / "src" / "pm_env" / "score_flash_attention.py"


def _run_score(tmp_path: Path, source: str) -> dict:
    """
    Helper to run the judge script on a given source code string, writing to a temporary file and returning the parsed JSON results.
    """
    target_path = tmp_path / "triton_attention.py"
    output_path = tmp_path / "judge_testing.json"
    target_path.write_text(source)

    subprocess.run(
        [sys.executable, str(SCORER), str(target_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output_path.read_text())


def _has_cuda() -> bool:
    """
    Checking for CUDA availability. Essential for test 2/4.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _has_triton() -> bool:
    """
    Checking for Triton availability. Essential for test 2/4.
    """
    try:
        import triton

        return True
    except Exception:
        return False


def test_cheater_returns_zero(tmp_path: Path):
    """
    A test to verify that a trivial implementation that doesn't do any computation gets a score of 0.
    This checks that the judge is not giving false positives.
    """
    source = """
import torch
import triton
import triton.language as tl

# Dummy kernel
@triton.jit
def _dummy(x_ptr):
    return

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    @staticmethod
    def backward(ctx, do):
        return None, None, None
"""
    results = _run_score(tmp_path, source)
    assert results["score"] == 0.0


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for judge timing")
@pytest.mark.skipif(not _has_triton(), reason="Triton required for static checks")
def test_reference_attention_scores(tmp_path: Path):
    """
    A correct but unoptimized (naive) implementation should score in and around 0.0 - 1.0,
    because the autograd.Function wrapper adds overhead versus the baseline.
    """
    source = """
import torch
import triton
import triton.language as tl

@triton.jit
def _dummy(x_ptr):
    return


def _attention(q, k, v):
    scale = 1.0 / (q.size(-1) ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        ctx.save_for_backward(q, k, v)
        return _attention(q, k, v)

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        q2 = q.detach().requires_grad_(True)
        k2 = k.detach().requires_grad_(True)
        v2 = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _attention(q2, k2, v2)
        dq, dk, dv = torch.autograd.grad(out, (q2, k2, v2), do)
        return dq, dk, dv
"""
    results = _run_score(tmp_path, source)
    assert 0.3 <= results["score"] <= 1.3


def test_multihead_attention_is_rejected(tmp_path: Path):
    """
    MultiheadAttention is a forbidden shortcut and should be rejected by static checks.
    Since it uses forbidden APIs, it should never run, so it does not need CUDA or Triton.
    """
    source = """
import torch
import triton
import triton.language as tl

@triton.jit
def _dummy(x_ptr):
    return


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        mha = torch.nn.MultiheadAttention(
            embed_dim=q.size(-1) * q.size(1),
            num_heads=q.size(1),
            bias=False,
            device=q.device,
            dtype=q.dtype,
        )
        q_mha = q.permute(2, 0, 1, 3).contiguous().reshape(q.size(2), q.size(0), -1)
        k_mha = k.permute(2, 0, 1, 3).contiguous().reshape(k.size(2), k.size(0), -1)
        v_mha = v.permute(2, 0, 1, 3).contiguous().reshape(v.size(2), v.size(0), -1)
        out, _ = mha(q_mha, k_mha, v_mha, need_weights=False)
        return out.reshape(q.size(2), q.size(0), q.size(1), q.size(3)).permute(1, 2, 0, 3)

    @staticmethod
    def backward(ctx, do):
        return None, None, None
"""
    results = _run_score(tmp_path, source)
    assert results["score"] == 0.0


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for judge timing")
@pytest.mark.skipif(not _has_triton(), reason="Triton required for kernel execution")
def test_triton_flash_attention_kernel_scores_nonzero(tmp_path: Path):
    """
    The Triton FlashAttention kernel implementation should pass correctness checks and earn a non-zero score.
    """
    source = """
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attention(
    Q, K, V,
    O, l, m,
    stride_qh, stride_qn,
    stride_kh, stride_kn,
    stride_vh, stride_vn,
    stride_oh, stride_on,
    stride_lh,
    stride_mh,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_q: tl.constexpr,
    N_v: tl.constexpr,
    D_head: tl.constexpr,
    B_c: tl.constexpr,
    B_r: tl.constexpr,
    T_r: tl.constexpr,
    T_c: tl.constexpr,
    sm_scale,
):
    head_idx = tl.program_id(0)
    tr_idx = tl.program_id(1)

    Q_ptr = Q + head_idx * stride_qh
    K_ptr = K + head_idx * stride_kh
    V_ptr = V + head_idx * stride_vh
    O_ptr = O + head_idx * stride_oh
    l_ptr = l + head_idx * stride_lh
    m_ptr = m + head_idx * stride_mh

    offs_q = tr_idx * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, BLOCK_D)

    qo_mask = (offs_q[:, None] < N_q) & (offs_d[None, :] < D_head)
    q_ptrs_2d = Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :]
    q = tl.load(q_ptrs_2d, mask=qo_mask, other=0.0)

    m_i = tl.full([BLOCK_R], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_R], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_R, BLOCK_D], dtype=tl.float32)

    for tc in range(T_c):
        offs_kv = tc * BLOCK_C + tl.arange(0, BLOCK_C)
        kv_mask = (offs_kv[:, None] < N_v) & (offs_d[None, :] < D_head)

        k_ptrs_2d = K_ptr + offs_kv[:, None] * stride_kn + offs_d[None, :]
        k = tl.load(k_ptrs_2d, mask=kv_mask, other=0.0)

        v_ptrs_2d = V_ptr + offs_kv[:, None] * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs_2d, mask=kv_mask, other=0.0)

        s = tl.dot(q, k.T) * sm_scale

        s_mask = (offs_q[:, None] < N_q) & (offs_kv[None, :] < N_v)
        s = tl.where(s_mask, s, float("-inf"))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_i = o_i * alpha[:, None]
        o_i += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    o_i = o_i / l_i[:, None]

    o_ptrs_2d = O_ptr + offs_q[:, None] * stride_on + offs_d[None, :]
    tl.store(o_ptrs_2d, o_i, mask=qo_mask)


def _naive_attention(q, k, v):
    scale = 1.0 / (q.size(-1) ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def multiheaded_attention_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    W_qkv: torch.Tensor,
    W_out: torch.Tensor,
    b_qkv: torch.Tensor,
    b_out: torch.Tensor,
    num_heads: int = 1,
    device: str = "cuda",
    block_size: int = 64,
) -> torch.Tensor:
    N_q, D = query.shape
    N_v, _ = key.shape
    dtype = W_qkv.dtype
    D_head = D // num_heads

    Q = torch.matmul(query, W_qkv[0:D, :].T) + b_qkv[0:D][None, :]
    K = torch.matmul(key, W_qkv[D : 2 * D, :].T) + b_qkv[D : 2 * D][None, :]
    V = torch.matmul(value, W_qkv[2 * D : 3 * D, :].T) + b_qkv[2 * D : 3 * D][None, :]

    Q = Q.reshape(N_q, num_heads, D_head).permute(1, 0, 2).contiguous()
    K = K.reshape(N_v, num_heads, D_head).permute(1, 0, 2).contiguous()
    V = V.reshape(N_v, num_heads, D_head).permute(1, 0, 2).contiguous()

    BLOCK_D = triton.next_power_of_2(D_head)
    B_c = min(triton.next_power_of_2(N_v), block_size)
    B_r = min(triton.next_power_of_2(N_q), block_size)
    BLOCK_C = B_c
    BLOCK_R = B_r

    T_c = triton.cdiv(N_v, B_c)
    T_r = triton.cdiv(N_q, B_r)

    O = torch.zeros_like(Q)
    l = torch.zeros((num_heads, N_q), device=device, dtype=torch.float32)
    m = torch.full((num_heads, N_q), fill_value=float("-inf"), device=device, dtype=torch.float32)

    grid = (num_heads, T_r)
    num_warps = 4 if D_head <= block_size else 8
    sm_scale = 1.0 / math.sqrt(D_head)

    _attention[grid](
        Q,
        K,
        V,
        O,
        l,
        m,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        V.stride(0),
        V.stride(1),
        O.stride(0),
        O.stride(1),
        l.stride(0),
        m.stride(0),
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        BLOCK_D=BLOCK_D,
        N_q=N_q,
        N_v=N_v,
        D_head=D_head,
        B_c=B_c,
        B_r=B_r,
        T_r=T_r,
        T_c=T_c,
        sm_scale=sm_scale,
        num_warps=num_warps,
    )

    O = O.permute(1, 0, 2).contiguous().reshape(N_q, D)
    out = torch.matmul(O, W_out.T) + b_out[None, :]
    return out


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        ctx.save_for_backward(q, k, v)
        batch, heads, seq_len, head_dim = q.shape
        embed_dim = heads * head_dim
        device = q.device
        dtype = q.dtype

        eye = torch.eye(embed_dim, device=device, dtype=dtype)
        W_qkv = eye.repeat(3, 1)
        b_qkv = torch.zeros(3 * embed_dim, device=device, dtype=dtype)
        W_out = eye
        b_out = torch.zeros(embed_dim, device=device, dtype=dtype)

        outputs = []
        for b in range(batch):
            q_2d = q[b].permute(1, 0, 2).contiguous().reshape(seq_len, embed_dim)
            k_2d = k[b].permute(1, 0, 2).contiguous().reshape(seq_len, embed_dim)
            v_2d = v[b].permute(1, 0, 2).contiguous().reshape(seq_len, embed_dim)

            out_2d = multiheaded_attention_triton(
                q_2d,
                k_2d,
                v_2d,
                W_qkv,
                W_out,
                b_qkv,
                b_out,
                num_heads=heads,
                device=device,
                block_size=64,
            )
            out = out_2d.reshape(seq_len, heads, head_dim).permute(1, 0, 2).contiguous()
            outputs.append(out)

        return torch.stack(outputs, dim=0)

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        q2 = q.detach().requires_grad_(True)
        k2 = k.detach().requires_grad_(True)
        v2 = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _naive_attention(q2, k2, v2)
        dq, dk, dv = torch.autograd.grad(out, (q2, k2, v2), do)
        return dq, dk, dv
"""
    results = _run_score(tmp_path, source)
    assert results["score"] >= 0.1
    assert results["metadata"]["stages"]["backward"] is True
