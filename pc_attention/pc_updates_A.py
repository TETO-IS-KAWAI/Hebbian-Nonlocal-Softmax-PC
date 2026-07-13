"""Path A free energy and its local update rules (derivations.md PART A).

The softmax denominator is promoted to a per-token state node, stored in log space
as l_i (= log zeta_i). The energy is

    F = (1/2) sum_i ||x_i^out - z_i||^2  +  (gamma/2) sum_i (l_i - c_i)^2

with z_i = sum_j exp(s_ij - l_i) v_j and c_i = log sum_j exp(s_ij) the true
log-normalizer. The second term pins l_i to c_i with precision gamma; as gamma
grows the state sits on the true normalizer and the weight updates become exact
softmax-attention backprop.

The single non-local read per token is that scalar c_i (the divisive-normalization
pool) -- locality budget 1, versus Path B's 2 shared states.

Same two entry points as Path B:
  * method="relax"       -- relax the log-normalizer l_i, then read the weights.
  * method="closed_form" -- the gamma -> inf limit (l_i = c_i, so b_ij = a_ij).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .softmax_attn import SoftmaxForward

Tensor = torch.Tensor


def _state_grad(
    l: Tensor, scores: Tensor, c: Tensor, V: Tensor, x_out: Tensor,
    gamma: float, mask: Optional[Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """dF/dl_i at the current log-normalizer. Returns (dF/dl, b, z, eps)."""
    b = torch.exp(scores - l.unsqueeze(-1))            # (B, N, N)  b_ij = exp(s_ij - l_i)
    z = torch.einsum("bij,bjd->bid", b, V)             # (B, N, d_v)  z_i = sum_j b_ij v_j
    eps = x_out - z
    if mask is not None:
        eps = eps * mask.unsqueeze(-1)
    # dF/dl_i = eps_i . z_i + gamma (l_i - c_i)
    dFdl = (eps * z).sum(dim=-1) + gamma * (l - c)      # (B, N)
    return dFdl, b, z, eps


@dataclass
class RelaxInfoA:
    steps: int
    res: float
    converged: bool


def relax_normalizer(
    fwd: SoftmaxForward, x_out: Tensor, gamma: float,
    lr: Optional[float] = None, n_steps: int = 200, tol: float = 1e-9,
    mask: Optional[Tensor] = None,
) -> tuple[Tensor, RelaxInfoA]:
    """Relax the log-normalizer l_i to dF/dl_i = 0, weights fixed.

    Starts at l_i = c_i (the true value) and steps with lr ~ 1/gamma, so it lands
    in a few iterations -- same fixed-point geometry as Path B.
    """
    if lr is None:
        lr = 1.0 / gamma
    l = fwd.c.clone()
    info = RelaxInfoA(steps=0, res=float("nan"), converged=False)
    for t in range(n_steps):
        dFdl, _, _, _ = _state_grad(l, fwd.scores, fwd.c, fwd.V, x_out, gamma, mask)
        res = dFdl.abs().max().item()
        info.steps, info.res = t + 1, res
        if res < tol:
            info.converged = True
            break
        l = l - lr * dFdl
    return l, info


def _weight_grads(fwd: SoftmaxForward, b: Tensor, eps: Tensor, r: Tensor) -> dict[str, Tensor]:
    """Local weight gradients (derivations.md A.4), per example.

    b_ij carries the state normalizer, fwd.A the true softmax weights, r_i the
    per-token normalizer error -- the one shared scalar each token reads.
    """
    V, Q, K, X, scale = fwd.V, fwd.Q, fwd.K, fwd.X, fwd.scale

    eps_dot_v = torch.einsum("bid,bjd->bij", eps, V)                 # (B, N, N)
    # dF/ds_ij = -b_ij (eps_i . v_j) - r_i a_ij
    w = -b * eps_dot_v - r.unsqueeze(-1) * fwd.A                     # (B, N, N)

    dFdq = scale * torch.einsum("bij,bjk->bik", w, K)               # (B, N, d_k)
    dFdk = scale * torch.einsum("bij,bik->bjk", w, Q)               # (B, N, d_k)
    dFdv = -torch.einsum("bij,bid->bjd", b, eps)                    # (B, N, d_v)

    gWQ = torch.einsum("bid,bik->bdk", X, dFdq)                     # (B, d, d_k)
    gWK = torch.einsum("bjd,bjk->bdk", X, dFdk)                     # (B, d, d_k)
    gWV = torch.einsum("bjd,bje->bde", X, dFdv)                     # (B, d, d_v)
    return {"W_Q": gWQ, "W_K": gWK, "W_V": gWV}


def pc_gradients_A(
    fwd: SoftmaxForward, x_out: Tensor,
    gamma: float = 1e4, method: str = "relax",
    lr: Optional[float] = None, n_steps: int = 200, tol: float = 1e-9,
    mask: Optional[Tensor] = None, return_info: bool = False,
    naive: bool = False, alpha: float = 1.0,
):
    """Path A local PC gradients for W_Q, W_K, W_V (per example).

    alpha scales the global redistribution term (the normalizer error r_i):
    alpha=1 is the exact rule, alpha=0 drops it entirely (the naive, strictly-local
    rule that no longer tracks backprop). naive=True is shorthand for alpha=0.
    The in-between values trace the locality spectrum. Only meaningful with
    method="closed_form".
    """
    if method == "closed_form":
        # gamma -> inf: l_i = c_i, so b_ij = a_ij and z_i = z*_i (the true output).
        z = fwd.z
        eps = x_out - z
        if mask is not None:
            eps = eps * mask.unsqueeze(-1)
        r = -(eps * z).sum(dim=-1)                    # r_i = -(eps_i . z_i)
        r = (0.0 if naive else alpha) * r             # keep a fraction of the redistribution
        grads = _weight_grads(fwd, fwd.A, eps, r)
        info = RelaxInfoA(steps=0, res=0.0, converged=True)
    elif method == "relax":
        l, info = relax_normalizer(fwd, x_out, gamma, lr=lr, n_steps=n_steps, tol=tol, mask=mask)
        b = torch.exp(fwd.scores - l.unsqueeze(-1))
        z = torch.einsum("bij,bjd->bid", b, fwd.V)
        eps = x_out - z
        if mask is not None:
            eps = eps * mask.unsqueeze(-1)
        r = gamma * (l - fwd.c)                        # normalizer error; -> -(eps.z) at fixed point
        grads = _weight_grads(fwd, b, eps, r)
    else:
        raise ValueError(f"unknown method {method!r}")

    if return_info:
        return grads, info
    return grads


def aggregate(grads: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v.sum(dim=0) for k, v in grads.items()}
