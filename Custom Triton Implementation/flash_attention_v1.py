import math
import torch
import triton
import triton.language as tl

@triton.jit
def _attention(
        Q, K, V,
        O, l, m,
        BLOCK_R : tl.constexpr,
        BLOCK_C : tl.constexpr,
        BLOCK_D : tl.constexpr,
        N_q : tl.constexpr, N_v : tl.constexpr,
        D_head : tl.constexpr, B_c : tl.constexpr, B_r : tl.constexpr,
        T_r : tl.constexpr, T_c : tl.constexpr,
        ):

    head_idx = tl.program_id(0)
    # transpose everything when passing args, so that moving across is linear.
    Q_ptr = Q + (head_idx * N_q * D_head)
    K_ptr = K + (head_idx * N_v * D_head)
    V_ptr = V + (head_idx * N_v * D_head)
    O_ptr = O + (head_idx * N_q * D_head)
    l_ptr = l + (head_idx * N_q)
    m_ptr = m + (head_idx * N_q)

    # Get ptrs for loading K,V,Q,O into SRAM
    D_indices = tl.arange(0, BLOCK_D)[:, None]          # (D_head, 1)
    C_indices = tl.arange(0, BLOCK_C)[None, :]          # (1, B_c)
    R_indices = tl.arange(0, BLOCK_R)[None, :]          # (1, B_r)
    # Moving the start of each row by N_v steps
    # then get B_c size slices from each row
    K_row_start_ptrs = K_ptr + (D_indices * N_v)            # (D_head, 1)
    K_j_ptrs = K_row_start_ptrs + C_indices                 # (D_head, B_c)
    V_row_start_ptrs = V_ptr + (D_indices * N_v)            # (D_head, 1)
    V_j_ptrs = V_row_start_ptrs + C_indices                 # (D_head, B_c)
    mask_KV = (D_indices < D_head) & (C_indices < B_c)      # (BLOCK_D, BLOCK_C)

    # Same for Q and O
    Q_row_start_ptrs = Q_ptr + (D_indices * N_q)            # (D_head, 1)
    Q_i_ptrs = Q_row_start_ptrs + R_indices                 # (D_head, B_r)
    O_row_start_ptrs = O_ptr + (D_indices * N_q)            # (D_head, 1)
    O_i_ptrs = O_row_start_ptrs + R_indices                 # (D_head, B_r)
    mask_QO = (D_indices < D_head) & (R_indices < B_r)      # (BLOCK_D, BLOCK_R)

    # Get the pointers for l and m
    l_i_ptrs = l_ptr + R_indices                            # (BLOCK_R,)
    m_i_ptrs = m_ptr + R_indices                            # (BLOCK_R,)
    mask_lm = R_indices < B_r                               # (BLOCK_R,)

    for j in range(T_c):
        K_j = tl.load(K_j_ptrs, mask=mask_KV, other=0.0)    # (D_head, B_c)
        V_j = tl.load(V_j_ptrs, mask=mask_KV, other=0.0).to(tl.float32)    # (D_head, B_c)

        for i in range(T_r):
            Q_i = tl.load(Q_i_ptrs, mask=mask_QO, other=0.0)                    # (D_head, B_r)
            O_i = tl.load(O_i_ptrs, mask=mask_QO, other=0.0).to(tl.float32)     # (D_head, B_r)
            l_i = tl.load(l_i_ptrs, mask=mask_lm, other=0.0).to(tl.float32)
            m_i = tl.load(m_i_ptrs, mask=mask_lm, other=float('-inf')).to(tl.float32)

            # Compute S_ij = Q_i x K_j^T
            scale = 1.0 / tl.sqrt(tl.full([], D_head, dtype=tl.float32))  # scalar
            S_ij = tl.dot(K_j.T, Q_i).to(tl.float32) * scale               # (B_c, B_r)

            # get the rowmax for S_ij
            m_ij = tl.max(S_ij, axis=0)                     # (B_r,)

            # compute exponents for softmax after subtracting max
            # element for numerical stability
            P_ij = tl.exp(S_ij - m_ij)                      # (B_c, B_r)

            # Compute the row sums for softmax
            l_ij = tl.sum(P_ij, axis=0)                     # (B_r,)

            # Update the global max m_i till this point
            # tl.cat threw an error when given dim=0, weird but okay because it defaults to 0
            # m_i_new = tl.max(tl.cat(m_ij, m_i[None, :]), axis=0, keep_dims=True)     # (1, B_r)
            m_i_new = tl.maximum(m_ij, m_i)                 # (B_r,)
            alpha = tl.exp(m_i - m_i_new)                   # (B_r,)
            beta  = tl.exp(m_ij - m_i_new)                  # (B_r,)
            l_i_new = alpha * l_i + beta * l_ij               # (B_r,)

            # Update the output
            # Broadcasting works here as the last dim is matched (D_head, Br) and (Br)
            PV = tl.dot(V_j, P_ij)                          # (D_head, B_r)
            O_i = ((l_i * alpha) * O_i + beta * PV) / l_i_new       # (D_head, B_r)

            # Store the computed O_i, l_i and m_i values
            tl.store(O_i_ptrs, O_i, mask=mask_QO)
            tl.store(l_i_ptrs, l_i_new, mask=mask_lm)
            tl.store(m_i_ptrs, m_i_new, mask=mask_lm)

            Q_i_ptrs += B_r
            O_i_ptrs += B_r
            l_i_ptrs += B_r
            m_i_ptrs += B_r

        K_j_ptrs += B_c    # shift ptrs for next iter
        V_j_ptrs += B_c
        Q_i_ptrs -= N_q
        O_i_ptrs -= N_q
        l_i_ptrs -= N_q
        m_i_ptrs -= N_q


def multiheaded_attention_triton(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        W_qkv, W_out,
        b_qkv, b_out,
        num_heads=1,
        device="cuda") -> torch.Tensor:

    N_q, D = query.shape
    N_v, _ = value.shape
    dtype = W_qkv.dtype

    M = 16*1024*4     # HARD CODED SRAM

    # Q = torch.matmul(query, W_qkv[0:D, :].transpose(0,1)) + b_qkv[0:D][None, :]
    # K = torch.matmul(query, W_qkv[D:2*D, :].transpose(0,1)) + b_qkv[D:2*D][None, :]
    # V = torch.matmul(query, W_qkv[2*D:3*D, :].transpose(0,1)) + b_qkv[2*D:3*D][None, :]

    # buffers
    # Q = torch.zeros((N_q, D), device=device, dtype=dtype)
    # K = torch.zeros((N_v, D), device=device, dtype=dtype)
    # V = torch.zeros((N_v, D), device=device, dtype=dtype)
    D_head = math.ceil(D / num_heads)

    # for head in range(num_heads):
    #     hs = head * D_head
    #     he = min(D, (head + 1) * D_head)

    #     Wq = W_qkv[0:D, :][hs:he, :]
    #     Wk = W_qkv[D:2*D, :][hs:he, :]
    #     Wv = W_qkv[2*D:3*D, :][hs:he, :]

    #     bq = b_qkv[hs:he]
    #     bk = b_qkv[D + hs:D + he]
    #     bv = b_qkv[2*D + hs:2*D + he]

    #     Q[:, hs:he] = (query @ Wq.T) + bq[None, :]
    #     K[:, hs:he] = (key   @ Wk.T) + bk[None, :]
    #     V[:, hs:he] = (value @ Wv.T) + bv[None, :]

    # --- QKV Projections (full) ---
    Q = torch.matmul(query, W_qkv[0:D, :].T) + b_qkv[0:D][None, :]       # (N_q, D)
    K = torch.matmul(key, W_qkv[D:2*D, :].T) + b_qkv[D:2*D][None, :]     # (N_v, D)
    V = torch.matmul(value, W_qkv[2*D:3*D, :].T) + b_qkv[2*D:3*D][None, :] # (N_v, D)

    # multiheads are cascaded column wise, by taking the transpose, we can access each head in continuous space
    Q = Q.T.contiguous()
    K = K.T.contiguous()
    V = V.T.contiguous()

    B_c = math.ceil(M / (4*D))
    B_r = min(math.ceil(M / (4*D)), D)

    O = torch.zeros((D, N_q), device=device, dtype=torch.float32)    # transposed to match Q,K,V

    l = torch.zeros((num_heads, N_q), device=device, dtype=torch.float32)
    m = torch.full((num_heads, N_q), float('-inf'), device=device, dtype=torch.float32)

    T_r = math.ceil(N_q / B_r)
    T_c = math.ceil(N_v / B_c)

    BLOCK_R = triton.next_power_of_2(B_r)
    BLOCK_C = triton.next_power_of_2(B_c)
    BLOCK_D = triton.next_power_of_2(D_head)

    num_warps = 8
    grid = (num_heads, )

    _attention[grid](
        Q, K, V,
        O, l, m,
        BLOCK_R, BLOCK_C, BLOCK_D,
        N_q, N_v,
        D_head, B_c, B_r,
        T_r, T_c,
        num_warps=num_warps
    )

    #print(O.dtype, W_out.dtype)

    O = O.to(torch.float16)
    attention = torch.matmul(O.T, W_out.transpose(0,1)) + b_out[None, :]

    return attention