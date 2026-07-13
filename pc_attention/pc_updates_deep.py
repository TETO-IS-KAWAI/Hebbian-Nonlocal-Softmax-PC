"""Predictive coding across a stack of linear-attention layers (M2).

Each pair of layers gets an activity (belief) node a^l in between. The input
a^0 = X and the output a^L = target are clamped; the in-between activities settle
during an inference phase, then the weights get their local update.

We use the "fixed-prediction" form of PC (Song et al., NeurIPS 2020, "Can the
brain do backprop?"). During inference the top-down predictions mu^l and the
layer Jacobians are held at their feedforward values instead of being recomputed
from the moving activities. That one change is what makes PC line up with
backprop: run the inference to equilibrium and the settled errors e^l become
exactly the backprop error signals, so the weight updates match backprop even at
depth. (The naive version that recomputes predictions does NOT match backprop --
the free activities soak up part of the error -- which is why we don't use it.)

The interesting regime for Path C is a *truncated* inference budget. The output
error has to travel down layer by layer, so with only a few inference steps the
lower layers (farthest from the clamped output) barely feel it: their gradients
come out attenuated and delta grows with depth. Run inference longer and the gap
closes. So depth cost here is really a cost of how long you let inference run.

    dF/da^l = e^l - (d mu^{l+1}/da^l)^T e^{l+1},     e^l = a^l - mu^l

The downward term is one layer's transpose applied to the error above; we get it
from autograd on that single layer, linearized at the fixed feedforward point.
"""

from __future__ import annotations

from typing import Optional

import torch

from .linear_attn import linear_attention_forward
from .pc_updates import pc_gradients

Tensor = torch.Tensor
LayerWeights = tuple[Tensor, Tensor, Tensor]


def layer_input_vjp(
    a_lin: Tensor, W_Q: Tensor, W_K: Tensor, W_V: Tensor,
    e_out: Tensor, feature_map: str, eps_guard: float,
) -> Tensor:
    """(d LinAttn(a)/d a)^T e_out, linearized at a = a_lin.

    A one-layer transpose (autograd on a single layer), i.e. the local feedback
    connection -- not a backward pass through the whole stack. a_lin is the fixed
    feedforward activity, so this Jacobian stays constant across inference steps.
    """
    a = a_lin.detach().clone().requires_grad_(True)
    fwd = linear_attention_forward(a, W_Q, W_K, W_V, feature_map, eps_guard)
    fwd.z.backward(e_out)
    return a.grad.detach()


def relax_deep(
    X: Tensor,
    weights: list[LayerWeights],
    target: Tensor,
    feature_map: str = "elu+1",
    eps_guard: float = 1e-6,
    lr: float = 0.5,
    n_steps: int = 200,
    tol: float = 1e-9,
) -> tuple[list[Tensor], list[Tensor], int, float]:
    """Settle the in-between activities with predictions fixed at feedforward.

    Returns (feedforward activities a_ff, settled errors e, steps taken, residual).
    e[l] = a^l - mu^l at the end of inference; e[0] is unused.
    """
    from .deep_linear_attn import deep_forward
    a_ff, _ = deep_forward(X, weights, feature_map, eps_guard)   # mu^l = a_ff[l]
    L = len(weights)

    a = [act.clone() for act in a_ff]
    a[L] = target                      # clamp the output to the label; a[0]=X already

    steps, res = 0, float("nan")
    for t in range(n_steps):
        e = [None] * (L + 1)
        for l in range(1, L + 1):
            e[l] = a[l] - a_ff[l]      # error against the FIXED feedforward prediction

        max_res = 0.0
        new_a = list(a)
        for l in range(1, L):          # free activities only
            # feedback from the layer above, linearized at the fixed feedforward point
            vjp = layer_input_vjp(a_ff[l], *weights[l], e[l + 1], feature_map, eps_guard)
            grad = e[l] - vjp
            max_res = max(max_res, grad.abs().max().item())
            new_a[l] = a[l] - lr * grad
        a = new_a

        steps, res = t + 1, max_res
        if max_res < tol:
            break

    e = [None] + [a[l] - a_ff[l] for l in range(1, L + 1)]
    return a_ff, e, steps, res


def deep_pc_grads_per_example(
    X: Tensor,
    weights: list[LayerWeights],
    target: Tensor,
    feature_map: str = "elu+1",
    eps_guard: float = 1e-6,
    lr: float = 0.5,
    n_steps: int = 200,
    tol: float = 1e-9,
) -> tuple[list[dict[str, Tensor]], int, float]:
    """PC weight gradients for every layer, one per example.

    Returns (list of per-layer grad dicts, inference steps used, final residual).
    Each layer reuses the single-layer rule at its feedforward operating point,
    with the settled error e^l standing in for the prediction error.
    """
    a_ff, e, steps, res = relax_deep(
        X, weights, target, feature_map, eps_guard, lr=lr, n_steps=n_steps, tol=tol
    )
    L = len(weights)

    grads = []
    for l in range(1, L + 1):
        fwd_l = linear_attention_forward(a_ff[l - 1], *weights[l - 1], feature_map, eps_guard)
        # fwd_l.z == mu^l == a_ff[l], so passing x_out = a_ff[l] + e[l] makes the
        # internal (x_out - z) equal the settled error e^l exactly.
        x_out = a_ff[l] + e[l]
        g = pc_gradients(fwd_l, x_out, method="closed_form")   # per-example, accumulators exact
        grads.append({"W_Q": g["W_Q"], "W_K": g["W_K"], "W_V": g["W_V"]})

    return grads, steps, res
