"""Path A forward pass: real softmax attention with a normalizer state node.

Path B avoided softmax's global denominator by swapping the kernel. Path A keeps
the real softmax and instead promotes just the denominator to a state variable.
Per query i:

    s_ij   = (q_i . k_j) / sqrt(d_k)
    z*_i   = sum_j softmax_j(s_i.) v_j             (the usual attention output)

The one genuinely global thing is the log-normalizer c_i = log sum_j exp(s_ij).
Path A treats that scalar as the only non-local signal each token needs -- the
"divisive normalization pool" of the spec. Everything else stays local.

This module gives the plain forward and the autograd reference (true softmax
loss). The state-node PC version lives in pc_updates_A.py. Both share this forward
so the gradient check compares like with like.

Numerics: exp overflows easily, so we always work relative to the row max /
log-sum-exp. The softmax weights a_ij = exp(s_ij - c_i) are what we actually carry,
never the raw exp(s_ij).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

Tensor = torch.Tensor


@dataclass
class SoftmaxForward:
    X: Tensor          # (B, N, d)     input tokens
    Q: Tensor          # (B, N, d_k)
    K: Tensor          # (B, N, d_k)
    V: Tensor          # (B, N, d_v)
    scores: Tensor     # (B, N, N)     s_ij (query i, key j), scaled
    c: Tensor          # (B, N)        c_i = log sum_j exp(s_ij)   (true log-normalizer)
    A: Tensor          # (B, N, N)     a_ij = softmax weights
    z: Tensor          # (B, N, d_v)   z*_i = sum_j a_ij v_j
    scale: float       # 1 / sqrt(d_k)


def softmax_attention_forward(
    X: Tensor, W_Q: Tensor, W_K: Tensor, W_V: Tensor,
) -> SoftmaxForward:
    """Standard softmax attention. X:(B,N,d); W_Q,W_K:(d,d_k); W_V:(d,d_v)."""
    d_k = W_Q.shape[1]
    scale = 1.0 / (d_k ** 0.5)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = torch.einsum("bik,bjk->bij", Q, K) * scale      # (B, N, N)
    c = torch.logsumexp(scores, dim=-1)                       # (B, N)
    A = torch.exp(scores - c.unsqueeze(-1))                   # (B, N, N)
    z = torch.einsum("bij,bjd->bid", A, V)                    # (B, N, d_v)

    return SoftmaxForward(X=X, Q=Q, K=K, V=V, scores=scores, c=c, A=A, z=z, scale=scale)


def softmax_pred_loss(z: Tensor, x_out: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """F_pred = 1/2 sum ||x_out - z||^2 (mask optional, as in Path B)."""
    resid = x_out - z
    if mask is not None:
        resid = resid * mask.unsqueeze(-1)
    return 0.5 * resid.pow(2).sum()


def softmax_backprop_gradients(
    X: Tensor, W_Q: Tensor, W_K: Tensor, W_V: Tensor,
    x_out: Tensor, mask: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Autograd gradients of the true softmax loss, pooled over the batch."""
    W_Q = W_Q.detach().clone().requires_grad_(True)
    W_K = W_K.detach().clone().requires_grad_(True)
    W_V = W_V.detach().clone().requires_grad_(True)
    fwd = softmax_attention_forward(X, W_Q, W_K, W_V)
    loss = softmax_pred_loss(fwd.z, x_out, mask=mask)
    loss.backward()
    return {
        "W_Q": W_Q.grad.detach().clone(),
        "W_K": W_K.grad.detach().clone(),
        "W_V": W_V.grad.detach().clone(),
        "loss": loss.detach().clone(),
    }


def softmax_backprop_per_example(
    X: Tensor, W_Q: Tensor, W_K: Tensor, W_V: Tensor,
    x_out: Tensor, mask: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Per-sequence softmax gradients, (B, *weight_shape), for the delta metric."""
    B = X.shape[0]
    grads = {"W_Q": [], "W_K": [], "W_V": []}
    for b in range(B):
        g = softmax_backprop_gradients(
            X[b : b + 1], W_Q, W_K, W_V, x_out[b : b + 1],
            mask=None if mask is None else mask[b : b + 1],
        )
        for k in grads:
            grads[k].append(g[k])
    return {k: torch.stack(v, dim=0) for k, v in grads.items()}
