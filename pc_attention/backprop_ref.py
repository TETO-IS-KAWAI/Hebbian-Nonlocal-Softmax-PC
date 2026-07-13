"""Ground-truth gradients from PyTorch autograd, used to check pc_updates.py.

Runs the very same forward (imported from linear_attn, not rewritten) and lets
autograd backprop the prediction energy

    L = (1/2) sum_i || x_i^out - z_i ||^2.

Whatever comes out here is the target the analytic local rules have to reproduce.
If the two ever disagree, this side is assumed correct.
"""

from __future__ import annotations

from typing import Optional

import torch

from .linear_attn import ForwardState, linear_attention_forward

Tensor = torch.Tensor


def pred_loss(z: Tensor, x_out: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """The prediction energy, summed over batch and tokens.

    mask (B, N) weights each token's error, e.g. to score only the query position
    of the recall task.
    """
    resid = x_out - z
    if mask is not None:
        resid = resid * mask.unsqueeze(-1)
    return 0.5 * resid.pow(2).sum()


def backprop_gradients(
    X: Tensor,
    W_Q: Tensor,
    W_K: Tensor,
    W_V: Tensor,
    x_out: Tensor,
    feature_map: str = "elu+1",
    eps_guard: float = 1e-6,
    mask: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Autograd gradients of the prediction energy, pooled over the batch.

    Also returns gradients of the accumulators S*, u* (grabbed with retain_grad),
    which the unit test lines up against the PC state-error signals. The weight
    grads have the weight's shape; the S, u grads keep the batch dimension.
    """
    W_Q = W_Q.detach().clone().requires_grad_(True)
    W_K = W_K.detach().clone().requires_grad_(True)
    W_V = W_V.detach().clone().requires_grad_(True)

    fwd: ForwardState = linear_attention_forward(
        X, W_Q, W_K, W_V, feature_map=feature_map, eps_guard=eps_guard
    )
    # S*, u* are computed values, not leaves, so ask autograd to keep their grads.
    fwd.S.retain_grad()
    fwd.u.retain_grad()

    loss = pred_loss(fwd.z, x_out, mask=mask)
    loss.backward()

    return {
        "W_Q": W_Q.grad.detach().clone(),
        "W_K": W_K.grad.detach().clone(),
        "W_V": W_V.grad.detach().clone(),
        "S": fwd.S.grad.detach().clone(),   # dL/dS*  (B, m, d_v)
        "u": fwd.u.grad.detach().clone(),   # dL/du*  (B, m)
        "loss": loss.detach().clone(),
    }


def backprop_gradients_per_example(
    X: Tensor,
    W_Q: Tensor,
    W_K: Tensor,
    W_V: Tensor,
    x_out: Tensor,
    feature_map: str = "elu+1",
    eps_guard: float = 1e-6,
    mask: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """One weight gradient per sequence, stacked as (B, *weight_shape).

    The delta = E[1 - cos] metric needs a separate gradient for each example to
    average the cosine over the batch, which the pooled version above can't give.
    So we just run the batch one sequence at a time.
    """
    B = X.shape[0]
    grads = {"W_Q": [], "W_K": [], "W_V": []}
    for b in range(B):
        g = backprop_gradients(
            X[b : b + 1], W_Q, W_K, W_V, x_out[b : b + 1],
            feature_map=feature_map, eps_guard=eps_guard,
            mask=None if mask is None else mask[b : b + 1],
        )
        for key in grads:
            grads[key].append(g[key])
    return {k: torch.stack(v, dim=0) for k, v in grads.items()}
