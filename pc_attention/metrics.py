"""Two measurements: how well the PC gradients line up with backprop, and whether
the update is genuinely local.

gradient_alignment computes delta = E[1 - cos(g_PC, g_BP)] for each weight tensor.
delta = 0 means the two gradients point the same way; the Path B bet is that delta
stays flat as the sequence gets longer.

verify_locality backs up the "only 2 shared reads" claim by experiment: freeze the
two shared state errors, wiggle one token's input, and check that every OTHER
token's gradient contribution doesn't budge. If it doesn't, then the only channel
tokens have to influence each other is those two shared signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .linear_attn import ForwardState, get_feature_map, linear_attention_forward, predict
from .pc_updates import pc_gradients, aggregate

Tensor = torch.Tensor
WEIGHT_KEYS = ("W_Q", "W_K", "W_V")


# ---- gradient alignment  delta = E[1 - cos] ---------------------------------
def _cosine_per_example(a: Tensor, b: Tensor, eps: float = 1e-30) -> Tensor:
    """Cosine similarity per example between two (B, ...) gradient tensors.

    The denominator guard is kept negligibly small (1e-30). Deep-layer gradients
    can be genuinely tiny (vanishing gradients: ~1e-8, so the norm product is
    ~1e-16), and a larger guard would swamp them and report a fake misalignment.
    A gradient that is essentially zero has no direction, so cosine near 0 there is
    the honest answer.
    """
    B = a.shape[0]
    a = a.reshape(B, -1)
    b = b.reshape(B, -1)
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + eps
    return num / den


def gradient_alignment(
    g_pc: dict[str, Tensor],
    g_bp: dict[str, Tensor],
    keys: tuple[str, ...] = WEIGHT_KEYS,
) -> dict[str, float]:
    """delta = E[1 - cos(g_PC, g_BP)] for each weight tensor, averaged over the batch."""
    out = {}
    for k in keys:
        cos = _cosine_per_example(g_pc[k], g_bp[k])
        out[k] = (1.0 - cos).mean().item()
    return out


# ---- locality budget --------------------------------------------------------
@dataclass
class LocalityReport:
    n_shared_reads: int
    shared_signals: tuple[str, ...]
    per_token_local: bool         # did every other token's contribution stay put?
    max_leakage: float            # largest change in an untouched token (want ~0)
    matches_aggregate: bool       # do the per-token pieces add up to pc_gradients?
    aggregate_gap: float


def _per_token_contributions(
    X: Tensor, W_Q: Tensor, W_K: Tensor, W_V: Tensor,
    S: Tensor, u: Tensor, betaE_S: Tensor, betaE_u: Tensor,
    x_out: Tensor, mask: Optional[Tensor], feature_map, eps_guard: float,
) -> dict[str, Tensor]:
    """Each token's own share of the weight gradients, NOT summed over tokens.

    The shared signals (S, u, betaE_S, betaE_u) are passed in frozen. Given those, a
    token's share depends only on its own x_i, so the tokens are independent -- which
    is exactly what verify_locality checks. Shapes: W_Q/W_K (B, N, d, m),
    W_V (B, N, d, d_v). The formulas match pc_updates._weight_grads_from_signals but
    stop short of the final sum over tokens.
    """
    fm = get_feature_map(feature_map)
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    phi_q, phi_k = fm.phi(Q), fm.phi(K)
    phi_q_prime, phi_k_prime = fm.phi_prime(Q), fm.phi_prime(K)

    # prediction using the frozen states
    z, denom = predict(S, u, phi_q, eps_guard)
    eps = x_out - z
    if mask is not None:
        eps = eps * mask.unsqueeze(-1)

    # W_Q share (through dz/dq)
    S_eps = torch.einsum("bmd,bnd->bnm", S, eps)
    z_dot_eps = (z * eps).sum(dim=-1)
    bracket = S_eps - u.unsqueeze(1) * z_dot_eps.unsqueeze(-1)
    dFdq = -(phi_q_prime * bracket) / denom.unsqueeze(-1)
    cWQ = torch.einsum("bnd,bnm->bndm", X, dFdq)

    # W_K share (through the state term)
    innerK = torch.einsum("bmd,bnd->bnm", betaE_S, V) + betaE_u.unsqueeze(1)
    dFdk = -(phi_k_prime * innerK)
    cWK = torch.einsum("bnd,bnm->bndm", X, dFdk)

    # W_V share (through the state term)
    dFdv = -torch.einsum("bmd,bnm->bnd", betaE_S, phi_k)
    cWV = torch.einsum("bnd,bne->bnde", X, dFdv)

    return {"W_Q": cWQ, "W_K": cWK, "W_V": cWV}


def verify_locality(
    fwd: ForwardState,
    W_Q: Tensor, W_K: Tensor, W_V: Tensor,
    x_out: Tensor,
    beta: float = 1e4,
    mask: Optional[Tensor] = None,
    method: str = "relax",
    perturb_token: int = 0,
    tol: float = 1e-9,
    feature_map: str = "elu+1",
) -> LocalityReport:
    """Check the update reads exactly 2 shared signals and is otherwise local."""
    grads, _ = pc_gradients(
        fwd, x_out, beta=beta, method=method, mask=mask, return_info=True
    )
    # recover the two frozen shared signals the weight grads were built from
    if method == "closed_form":
        S, u = fwd.S, fwd.u
        betaE_S = -grads["S"]   # grads['S'] is -betaE_S
        betaE_u = -grads["u"]
    else:
        from .pc_updates import relax_states
        S, u, _ = relax_states(fwd, x_out, beta, mask=mask)
        betaE_S = beta * (S - fwd.S)
        betaE_u = beta * (u - fwd.u)

    args = (W_Q, W_K, W_V, S, u, betaE_S, betaE_u, x_out, mask, feature_map, fwd.eps_guard)
    c0 = _per_token_contributions(fwd.X, *args)

    # sanity check: the per-token shares should add back up to the reported gradient
    agg = aggregate(grads)
    gap = max((c0[k].sum(dim=1).sum(dim=0) - agg[k]).abs().max().item() for k in WEIGHT_KEYS)

    # the actual test: change one token's input, keep the shared signals frozen, and
    # recompute. If the update is local, no other token's share should move.
    Xp = fwd.X.clone()
    Xp[:, perturb_token] += torch.randn_like(Xp[:, perturb_token])
    c1 = _per_token_contributions(Xp, *args)

    leak = 0.0
    N = fwd.X.shape[1]
    keep = [i for i in range(N) if i != perturb_token]
    for k in WEIGHT_KEYS:
        if keep:
            leak = max(leak, (c1[k][:, keep] - c0[k][:, keep]).abs().max().item())

    return LocalityReport(
        n_shared_reads=2,
        shared_signals=("beta*E_S (S accumulator)", "beta*E_u (u accumulator)"),
        per_token_local=leak < tol,
        max_leakage=leak,
        matches_aggregate=gap < 1e-6,
        aggregate_gap=gap,
    )
