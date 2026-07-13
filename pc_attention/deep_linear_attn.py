"""Stack of linear-attention layers (M2), plus its autograd reference.

M1 was a single layer, where predictive coding is provably exact. Depth is where
PC is known to struggle: error signals attenuate as they travel down through
layers. M2 stacks L of the Path B layers and lets us watch how the PC-vs-backprop
gap grows with depth L (and sequence length N) -- the delta(N, L) surface that
Path C then tries to fit.

Layout is a plain feedforward stack, no residual connection:

    a^0 = X (input)
    a^l = LinAttn_l(a^{l-1})     for l = 1..L
    loss = (1/2) || target - a^L ||^2

We use a dense target on every token (not the masked recall target): M2 measures
gradient alignment, not task accuracy, so a generic regression target keeps the
depth effect clean and easy to read.
"""

from __future__ import annotations

from typing import Optional

import torch

from .linear_attn import ForwardState, linear_attention_forward

Tensor = torch.Tensor
LayerWeights = tuple[Tensor, Tensor, Tensor]   # (W_Q, W_K, W_V)


def deep_forward(
    X: Tensor,
    weights: list[LayerWeights],
    feature_map: str = "elu+1",
    eps_guard: float = 1e-6,
) -> tuple[list[Tensor], list[ForwardState]]:
    """Run the stack. Returns the activities [a^0, a^1, ..., a^L] and the per-layer
    ForwardState objects (so the PC code can reuse the intermediate quantities)."""
    activities = [X]
    fwds: list[ForwardState] = []
    a = X
    for (W_Q, W_K, W_V) in weights:
        fwd = linear_attention_forward(a, W_Q, W_K, W_V, feature_map, eps_guard)
        fwds.append(fwd)
        activities.append(fwd.z)
        a = fwd.z
    return activities, fwds


def deep_backprop_grads_per_example(
    X: Tensor,
    weights: list[LayerWeights],
    target: Tensor,
    feature_map: str = "elu+1",
    eps_guard: float = 1e-6,
) -> list[dict[str, Tensor]]:
    """True gradients through the whole stack, one per example.

    Returns a list (one dict per layer) of per-example weight grads shaped
    (B, *weight_shape). Like the M1 reference, we run the batch one sequence at a
    time so delta = E[1 - cos] can average the cosine over the batch.
    """
    B = X.shape[0]
    L = len(weights)
    per_layer = [{"W_Q": [], "W_K": [], "W_V": []} for _ in range(L)]

    for b in range(B):
        # weights are shared across the batch, so just make grad-tracking copies
        ws = [tuple(w.detach().clone().requires_grad_(True) for w in layer)
              for layer in weights]
        a = X[b : b + 1]
        for (W_Q, W_K, W_V) in ws:
            fwd = linear_attention_forward(a, W_Q, W_K, W_V, feature_map, eps_guard)
            a = fwd.z
        loss = 0.5 * (target[b : b + 1] - a).pow(2).sum()
        loss.backward()
        for l, (W_Q, W_K, W_V) in enumerate(ws):
            per_layer[l]["W_Q"].append(W_Q.grad.detach().clone())
            per_layer[l]["W_K"].append(W_K.grad.detach().clone())
            per_layer[l]["W_V"].append(W_V.grad.detach().clone())

    return [{k: torch.stack(v, dim=0) for k, v in layer.items()} for layer in per_layer]
