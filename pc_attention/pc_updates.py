"""Predictive-coding update rules for the Path B layer.

Predictive coding (PC) trains a network in two alternating phases, using only
signals available locally at each weight (no stored backward pass through the whole
graph):

  1. inference / relaxation -- hold the weights fixed and let the "state" variables
     settle until predictions and their targets stop pulling against each other;
  2. learning -- with the states settled, nudge each weight using only quantities
     that live right where that weight does.

Here the two accumulators S and u are promoted to state variables. The energy the
whole thing rolls downhill is

    F = (1/2) sum_i ||x_i^out - z_i||^2                        (prediction error)
      + (beta/2) ||S - S*||^2 + (beta/2) ||u - u*||^2          (keep S, u near truth)

S* and u* are the accumulators the forward pass actually produced, and beta says
how hard we pin S, u to them. Take beta large and the states barely move from
S*, u*, at which point these local learning updates equal ordinary backprop
gradients (worked out in derivations.md, checked in tests/test_gradients.py).

The only thing a weight update reads beyond its own token is the pair of state
errors beta*(S - S*) and beta*(u - u*). Two shared reads and nothing else: that is
the entire locality claim.

Two entry points give the same gradients:
  - method="relax": actually run phase 1 (gradient descent on S, u) and read the
    residuals off the settled states. This is the faithful PC procedure.
  - method="closed_form": skip the loop and use the beta -> infinity values
    directly. Faster, and a useful reference the relaxation should converge to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .linear_attn import ForwardState, predict

Tensor = torch.Tensor


# ---- phase 1: state gradients (inference) -----------------------------------
def state_gradients(
    S: Tensor, u: Tensor,
    S_star: Tensor, u_star: Tensor,
    phi_q: Tensor, x_out: Tensor,
    beta: float, eps_guard: float,
    mask: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Gradients of F with respect to the states S and u (full details in
    derivations.md section 3).

    The du term is the one that's easy to get wrong: its prediction-error part
    comes out with a PLUS sign, opposite to the minus in dF/dS, so it is written
    out explicitly below. Also returns z, denom and eps computed along the way,
    since the callers want them. mask (B, N) restricts supervision to some tokens
    (the recall task only scores the query position).
    """
    z, denom = predict(S, u, phi_q, eps_guard)
    eps = x_out - z                                    # (B, N, d_v)
    if mask is not None:
        eps = eps * mask.unsqueeze(-1)

    # dF/dS = - sum_i phi(q_i) eps_i^T / n_i + beta (S - S*)
    phi_q_over_n = phi_q / denom.unsqueeze(-1)         # (B, N, m)
    dFdS = -torch.einsum("bnm,bnd->bmd", phi_q_over_n, eps) + beta * (S - S_star)

    # dF/du = + sum_i (eps_i^T z_i / n_i) phi(q_i) + beta (u - u*)   <- note the +
    s_scalar = (eps * z).sum(dim=-1) / denom          # (B, N)  = eps_i^T z_i / n_i
    dFdu = torch.einsum("bn,bnm->bm", s_scalar, phi_q) + beta * (u - u_star)

    return dFdS, dFdu, z, denom, eps


@dataclass
class RelaxInfo:
    steps: int
    res_S: float          # final max |dF/dS|  (how close to settled)
    res_u: float          # final max |dF/du|
    converged: bool
    history: list = field(default_factory=list)  # (res_S, res_u) per step, if asked


def relax_states(
    fwd: ForwardState,
    x_out: Tensor,
    beta: float,
    lr: Optional[float] = None,
    n_steps: int = 200,
    tol: float = 1e-9,
    record_history: bool = False,
    mask: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, RelaxInfo]:
    """Phase 1: let S and u settle to where dF/dS = dF/du = 0, weights held fixed.

    We start S, u at the forward values S*, u* and take gradient-descent steps. The
    default step size lr = 1/beta is deliberate: near the solution the S, u dynamics
    are linear with rate beta, and starting from S* we are already about one step of
    size 1/beta away, so it lands in a handful of iterations. A larger step
    overshoots (lr*beta > 2 diverges).
    """
    if lr is None:
        lr = 1.0 / beta

    S = fwd.S.clone()
    u = fwd.u.clone()
    S_star, u_star = fwd.S, fwd.u

    info = RelaxInfo(steps=0, res_S=float("nan"), res_u=float("nan"), converged=False)
    for t in range(n_steps):
        dFdS, dFdu, _, _, _ = state_gradients(
            S, u, S_star, u_star, fwd.phi_q, x_out, beta, fwd.eps_guard, mask=mask
        )
        res_S = dFdS.abs().max().item()
        res_u = dFdu.abs().max().item()
        if record_history:
            info.history.append((res_S, res_u))
        info.steps = t + 1
        info.res_S, info.res_u = res_S, res_u
        if max(res_S, res_u) < tol:      # settled
            info.converged = True
            break
        S = S - lr * dFdS
        u = u - lr * dFdu

    return S, u, info


# ---- phase 2: weight gradients ----------------------------------------------
def _weight_grads_from_signals(
    fwd: ForwardState,
    S: Tensor, u: Tensor,
    z: Tensor, denom: Tensor, eps: Tensor,
    betaE_S: Tensor, betaE_u: Tensor,
) -> dict[str, Tensor]:
    """The weight gradients, one per example (batch axis kept).

    Each block is an outer product summed over tokens -- Hebbian and local. The
    only cross-token information is betaE_S = beta*(S - S*) and betaE_u = beta*(u -
    u*), the two shared state errors. Derivation in derivations.md section 5.
    """
    X, V = fwd.X, fwd.V

    # W_Q comes from the prediction-error term, through how z depends on q
    # (the dz/dq Jacobian, derivations.md sections 4 and 5.1):
    #   dF/dq_i = -(1/n_i) phi'(q_i) * [ S eps_i - u (z_i . eps_i) ]
    S_eps = torch.einsum("bmd,bnd->bnm", S, eps)                 # (B, N, m)
    z_dot_eps = (z * eps).sum(dim=-1)                            # (B, N)
    bracket = S_eps - u.unsqueeze(1) * z_dot_eps.unsqueeze(-1)   # (B, N, m)
    dFdq = -(fwd.phi_q_prime * bracket) / denom.unsqueeze(-1)    # (B, N, m)
    gWQ = torch.einsum("bnd,bnm->bdm", X, dFdq)                  # (B, d, m)

    # W_K comes from the state term:
    #   dF/dk_j = -phi'(k_j) * ( betaE_S . v_j + betaE_u )
    innerK = torch.einsum("bmd,bnd->bnm", betaE_S, V) + betaE_u.unsqueeze(1)  # (B, N, m)
    dFdk = -(fwd.phi_k_prime * innerK)                          # (B, N, m)
    gWK = torch.einsum("bnd,bnm->bdm", X, dFdk)                 # (B, d, m)

    # W_V also comes from the state term:
    #   dF/dv_j = - betaE_S^T phi(k_j)
    dFdv = -torch.einsum("bmd,bnm->bnd", betaE_S, fwd.phi_k)    # (B, N, d_v)
    gWV = torch.einsum("bnd,bne->bde", X, dFdv)                 # (B, d, d_v)

    # Also hand back what backprop's gradients of the accumulators S*, u* would be,
    # so the unit test can compare. The signs flip: dL/dS* = -betaE_S,
    # dL/du* = -betaE_u (derivations.md section 6).
    return {
        "W_Q": gWQ, "W_K": gWK, "W_V": gWV,
        "S": -betaE_S, "u": -betaE_u,
    }


def pc_gradients(
    fwd: ForwardState,
    x_out: Tensor,
    beta: float = 1e4,
    method: str = "relax",
    lr: Optional[float] = None,
    n_steps: int = 200,
    tol: float = 1e-9,
    return_info: bool = False,
    mask: Optional[Tensor] = None,
):
    """The PC weight gradients (plus the S, u error signals).

    Gradients come back per example (batch axis kept) so metrics.py can take a
    cosine per example and average; call aggregate() to sum them into the pooled
    gradient used for training. mask (B, N) supervises only some tokens.
    """
    if method == "closed_form":
        # beta -> infinity shortcut: at that limit the two state errors have closed
        # forms (G_S and -H below) and the states just sit at S*, u*.
        z, denom, eps = fwd.z, fwd.denom, x_out - fwd.z
        if mask is not None:
            eps = eps * mask.unsqueeze(-1)
        phi_q_over_n = fwd.phi_q / denom.unsqueeze(-1)
        G_S = torch.einsum("bnm,bnd->bmd", phi_q_over_n, eps)          # = betaE_S
        s_scalar = (eps * z).sum(dim=-1) / denom
        H = torch.einsum("bn,bnm->bm", s_scalar, fwd.phi_q)
        betaE_S, betaE_u = G_S, -H
        grads = _weight_grads_from_signals(
            fwd, fwd.S, fwd.u, z, denom, eps, betaE_S, betaE_u
        )
        info = RelaxInfo(steps=0, res_S=0.0, res_u=0.0, converged=True)
    elif method == "relax":
        # run the inference loop, then read the state errors off the settled states
        S, u, info = relax_states(
            fwd, x_out, beta, lr=lr, n_steps=n_steps, tol=tol, mask=mask
        )
        z, denom = predict(S, u, fwd.phi_q, fwd.eps_guard)
        eps = x_out - z
        if mask is not None:
            eps = eps * mask.unsqueeze(-1)
        betaE_S = beta * (S - fwd.S)     # equals G_S once settled
        betaE_u = beta * (u - fwd.u)     # equals -H once settled
        grads = _weight_grads_from_signals(fwd, S, u, z, denom, eps, betaE_S, betaE_u)
    else:
        raise ValueError(f"unknown method {method!r} (use 'relax' or 'closed_form')")

    if return_info:
        return grads, info
    return grads


def aggregate(grads: dict[str, Tensor]) -> dict[str, Tensor]:
    """Sum the per-example gradients into one shared-weight gradient."""
    return {k: v.sum(dim=0) for k, v in grads.items()}
