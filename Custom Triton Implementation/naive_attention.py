import math
import torch

def attention_naive(X: torch.Tensor, W_q, W_k, W_v,
                    b_q, b_k, b_v, device) -> torch.Tensor:
    """
        X :     NxD (N: number of tokens, D: Dimension of latent space tokenizer)
        W_*:    D_headxD (D_head: Model space dimension / num heads)
    """

    # check if X is NxD
    assert(X.dim()==2)

    Q = torch.matmul(X, W_q.transpose(0,1)) + b_q[None, :]
    K = torch.matmul(X, W_k.transpose(0,1)) + b_k[None, :]
    V = torch.matmul(X, W_v.transpose(0,1)) + b_v[None, :]
    D_V = torch.tensor(V.shape[1], device=device)

    KQ_normalised = torch.matmul(Q, K.transpose(0,1)) / torch.sqrt(D_V)
    KQ_softmax = torch.softmax(KQ_normalised, dim=1)

    attention = torch.matmul(KQ_softmax, V)

    return attention

# @torch.compile(fullgraph=True)
def multiheaded_attention_naive(X: torch.Tensor, W_qkv, W_out,
                    b_qkv, b_out, num_heads=1, device="cuda") -> torch.Tensor:
    """
    W_qkv: 3DxD
    W_out: DxD
    b_qkv: 3D
    b_out: D
    """
    # check if X is NxD
    assert(X.dim()==2)

    N, D = X.shape
    D_head = math.ceil(D / num_heads)
    attention = torch.empty((N, D), device=device, dtype=torch.float16)

    for head in range(num_heads):
        head_start = head*D_head
        head_end = min(D, (head+1)*D_head)
        attention[:,head_start:head_end] = attention_naive(
            X,
            W_qkv[0:D, :][head_start:head_end, :],
            W_qkv[D:2*D, :][head_start:head_end, :],
            W_qkv[2*D:3*D, :][head_start:head_end, :],
            b_qkv[0+head_start:0+head_end],
            b_qkv[D+head_start:D+head_end],
            b_qkv[2*D+head_start:2*D+head_end],
            device
        )

    attention = torch.matmul(attention, W_out.transpose(0,1)) + b_out[None, :]

    return attention