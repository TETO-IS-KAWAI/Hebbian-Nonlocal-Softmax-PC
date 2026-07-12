"""A tiny causal char-level language model, trainable by PC or by backprop (M3).

Architecture (one causal linear-attention layer):

    ids -> Emb[ids] + positional encoding = X
        -> Z = CausalLinAttn(X)
        -> logits = Z @ W_out
        -> softmax cross-entropy against the next character

M3's question is whether the *local* PC updates train this end-to-end model as
well as backprop. Because every piece has an exact local rule -- the attention
layer's causal PC gradient (causal_linear_attn.py), a Hebbian outer product for
the output head, and a one-layer feedback (transpose) for the embedding -- PC here
reproduces the backprop gradient, so the two training runs should track each other.

The cross-entropy output fits PC cleanly: the error handed back at the logits is
just p - y (softmax minus one-hot), the standard prediction-error form.

`pc_grads` uses one-layer autograd only to apply the attention layer's input
transpose (dL/dX, the local feedback connection) -- never a backward pass through
the whole model.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .causal_linear_attn import causal_attention_forward, causal_pc_gradients

Tensor = torch.Tensor
PARAM_KEYS = ("Emb", "W_Q", "W_K", "W_V", "W_out")


def sinusoidal_pe(N: int, d: int, dtype=torch.float64) -> Tensor:
    pe = torch.zeros(N, d, dtype=dtype)
    pos = torch.arange(N, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=dtype) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def init_params(vocab: int, d: int, m: int, seed: int = 0, dtype=torch.float64) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "Emb": torch.randn(vocab, d, generator=g, dtype=dtype) * 0.1,
        "W_Q": torch.randn(d, m, generator=g, dtype=dtype) / d**0.5,
        "W_K": torch.randn(d, m, generator=g, dtype=dtype) / d**0.5,
        "W_V": torch.randn(d, d, generator=g, dtype=dtype) / d**0.5,
        "W_out": torch.randn(d, vocab, generator=g, dtype=dtype) / d**0.5,
    }


def ce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    logp = logits.log_softmax(dim=-1)
    return -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()


def accuracy(logits: Tensor, targets: Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).double().mean().item()


def backprop_grads(params, ids, targets, pe, feature_map="elu+1"):
    """Gradients of the CE loss through the whole model by autograd."""
    p = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    X = p["Emb"][ids] + pe
    fwd = causal_attention_forward(X, p["W_Q"], p["W_K"], p["W_V"], feature_map)
    logits = fwd.z @ p["W_out"]
    loss = ce_loss(logits, targets)
    loss.backward()
    return {k: p[k].grad.detach().clone() for k in PARAM_KEYS}, loss.item(), accuracy(logits, targets)


def pc_grads(params, ids, targets, pe, feature_map="elu+1"):
    """The same gradients from purely local PC rules."""
    B, N = ids.shape
    vocab = params["W_out"].shape[1]

    X = params["Emb"][ids] + pe
    fwd = causal_attention_forward(X, params["W_Q"], params["W_K"], params["W_V"], feature_map)
    Z = fwd.z
    logits = Z @ params["W_out"]
    loss = ce_loss(logits, targets)

    # output prediction error: p - y, scaled to match the mean-CE loss
    p = logits.softmax(dim=-1)
    y = torch.zeros_like(p)
    y.scatter_(-1, targets.unsqueeze(-1), 1.0)
    e_logits = (p - y) / (B * N)                                   # (B, N, vocab)

    # output head: Hebbian outer product
    gW_out = torch.einsum("bnd,bnv->dv", Z, e_logits)

    # error handed back to the attention output (one-layer feedback)
    dLdZ = torch.einsum("bnv,dv->bnd", e_logits, params["W_out"])  # (B, N, d)

    # attention weights: causal PC rule with eps = -dLdZ  (x_out = Z - dLdZ)
    g_attn = causal_pc_gradients(fwd, Z - dLdZ)
    gW_Q = g_attn["W_Q"].sum(dim=0)
    gW_K = g_attn["W_K"].sum(dim=0)
    gW_V = g_attn["W_V"].sum(dim=0)

    # error back to the embedding: attention layer's input transpose (local feedback)
    Xr = X.detach().requires_grad_(True)
    fr = causal_attention_forward(Xr, params["W_Q"], params["W_K"], params["W_V"], feature_map)
    fr.z.backward(dLdZ)
    dLdX = Xr.grad                                                 # (B, N, d)

    # scatter the input error onto the embedding rows that produced it
    gEmb = torch.zeros_like(params["Emb"])
    gEmb.index_add_(0, ids.reshape(-1), dLdX.reshape(-1, dLdX.shape[-1]))

    grads = {"Emb": gEmb, "W_Q": gW_Q, "W_K": gW_K, "W_V": gW_V, "W_out": gW_out}
    return grads, loss.item(), accuracy(logits, targets)


@torch.no_grad()
def generate(params, prime_ids, pe, n_new, feature_map="elu+1", temperature=1.0):
    """Greedy/temperature sampling for a qualitative check. prime_ids: (1, N0)."""
    ids = prime_ids.clone()
    N_max = pe.shape[0]
    for _ in range(n_new):
        ctx = ids[:, -N_max:]
        X = params["Emb"][ctx] + pe[: ctx.shape[1]]
        fwd = causal_attention_forward(X, params["W_Q"], params["W_K"], params["W_V"], feature_map)
        logits = (fwd.z @ params["W_out"])[:, -1] / temperature
        nxt = torch.multinomial(logits.softmax(-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids
