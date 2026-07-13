"""Causal (autoregressive) linear attention + its local PC gradients (M3).

A char-level LM has to be causal: position i may only look at positions j <= i. For
linear attention that turns the two shared accumulators into *running* (prefix)
sums:

    S_i = sum_{j<=i} phi(k_j) v_j^T ,   u_i = sum_{j<=i} phi(k_j)
    z_i = S_i^T phi(q_i) / (u_i^T phi(q_i))

So the forward is a forward scan. The gradient is the mirror image: a key j feeds
every query i >= j, so the error signal each key receives is a *suffix* sum
(reverse scan) of the per-position query errors. This is the same forward-scan /
backward-scan pair you get differentiating any recurrence, and it stays local:
one forward accumulation, one backward accumulation.

The weight gradients are the M1 rules with the global accumulator error G_S / H
replaced by their suffix sums GG_S[j] = sum_{i>=j} G_{S_i}, GG_H[j] = sum_{i>=j} H_i.
Given here in the closed form (beta -> inf); checked against autograd in
tests/test_gradients_causal.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .linear_attn import get_feature_map

Tensor = torch.Tensor


@dataclass
class CausalForward:
    X: Tensor          # (B, N, d)
    Q: Tensor; K: Tensor; V: Tensor
    phi_q: Tensor; phi_k: Tensor
    phi_q_prime: Tensor; phi_k_prime: Tensor
    S_cum: Tensor      # (B, N, m, d_v)  prefix sum of phi(k) v^T
    u_cum: Tensor      # (B, N, m)       prefix sum of phi(k)
    denom: Tensor      # (B, N)          n_i = u_i . phi(q_i) + eps
    z: Tensor          # (B, N, d_v)
    eps_guard: float


def causal_attention_forward(
    X: Tensor, W_Q: Tensor, W_K: Tensor, W_V: Tensor,
    feature_map: str = "elu+1", eps_guard: float = 1e-6,
) -> CausalForward:
    fm = get_feature_map(feature_map)
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    phi_q, phi_k = fm.phi(Q), fm.phi(K)
    phi_q_prime, phi_k_prime = fm.phi_prime(Q), fm.phi_prime(K)

    outer = torch.einsum("bnm,bnd->bnmd", phi_k, V)     # (B, N, m, d_v)
    S_cum = outer.cumsum(dim=1)                          # prefix sum over positions
    u_cum = phi_k.cumsum(dim=1)                          # (B, N, m)

    denom = torch.einsum("bnm,bnm->bn", u_cum, phi_q) + eps_guard
    num = torch.einsum("bnmd,bnm->bnd", S_cum, phi_q)
    z = num / denom.unsqueeze(-1)

    return CausalForward(
        X=X, Q=Q, K=K, V=V, phi_q=phi_q, phi_k=phi_k,
        phi_q_prime=phi_q_prime, phi_k_prime=phi_k_prime,
        S_cum=S_cum, u_cum=u_cum, denom=denom, z=z, eps_guard=eps_guard,
    )


def _suffix_sum(t: Tensor) -> Tensor:
    """sum over positions i >= j, along dim 1 (a reverse cumulative sum)."""
    return t.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])


def causal_pc_gradients(
    fwd: CausalForward, x_out: Tensor, mask: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Closed-form (beta -> inf) local PC gradients, per example.

    ``dz_out`` may be a plain regression error (x_out - z) or, for the LM, an
    upstream gradient injected as x_out = z - g so that (x_out - z) = -g. Either
    way the code below only uses eps = x_out - z.
    """
    z, n = fwd.z, fwd.denom
    eps = x_out - z
    if mask is not None:
        eps = eps * mask.unsqueeze(-1)

    # per-position accumulator error signals (as in M1, but kept per position)
    G_S = torch.einsum("bnm,bnd->bnmd", fwd.phi_q, eps) / n[:, :, None, None]   # (B,N,m,d_v)
    zeps = (z * eps).sum(dim=-1) / n                                            # (B,N)
    H = zeps.unsqueeze(-1) * fwd.phi_q                                          # (B,N,m)

    # a key at position j is read by every query i >= j -> suffix sums
    GG_S = _suffix_sum(G_S)                                                     # (B,N,m,d_v)
    GG_H = _suffix_sum(H)                                                       # (B,N,m)

    # W_Q: per position, through z_i's dependence on q_i (uses the prefix S_i,u_i)
    S_eps = torch.einsum("bnmd,bnd->bnm", fwd.S_cum, eps)
    z_dot_eps = (z * eps).sum(dim=-1)
    bracket = S_eps - fwd.u_cum * z_dot_eps.unsqueeze(-1)
    dFdq = -(fwd.phi_q_prime * bracket) / n.unsqueeze(-1)
    gWQ = torch.einsum("bnd,bnm->bdm", fwd.X, dFdq)

    # W_V: dF/dv_j = -GG_S[j]^T phi(k_j)
    dFdv = -torch.einsum("bnmd,bnm->bnd", GG_S, fwd.phi_k)
    gWV = torch.einsum("bnd,bne->bde", fwd.X, dFdv)

    # W_K: dF/dphi(k_j) = -GG_S[j] v_j + GG_H[j]
    dFdphik = -torch.einsum("bnmd,bnd->bnm", GG_S, fwd.V) + GG_H
    dFdk = fwd.phi_k_prime * dFdphik
    gWK = torch.einsum("bnd,bnm->bdm", fwd.X, dFdk)

    return {"W_Q": gWQ, "W_K": gWK, "W_V": gWV}
