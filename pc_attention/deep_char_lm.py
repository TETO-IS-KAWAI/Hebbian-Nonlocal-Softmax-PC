"""Deep causal char-LM (residual stack) trained by finite-budget PC (extension 1).

M3's LM was a single causal layer, where inference is trivial. To study what a
*bounded inference budget* costs, we need depth. This stacks L causal
linear-attention layers with residual connections:

    a^0 = Emb[ids] + pos
    a^l = a^{l-1} + CausalLinAttn_l(a^{l-1})       l = 1..L
    logits = a^L @ W_out ,  softmax cross-entropy

PC training reuses the M2 fixed-prediction scheme: relax the in-between activities
for T steps, then apply each layer's local weight rule (causal PC for attention,
Hebbian for the head, one-layer transpose for the embedding). The cross-entropy
head injects its error `p - y` at the top activity instead of a clamped target.

The knob is T (inference steps). At large T this equals backprop (checked in
run.py's gate); at small T the error hasn't reached the lower layers, so their
updates are wrong -- the realistic, biologically-plausible regime.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .causal_linear_attn import causal_attention_forward, causal_pc_gradients
from .char_lm import sinusoidal_pe, ce_loss, accuracy

Tensor = torch.Tensor


def init_params(vocab, d, m, L, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return {
        "Emb": torch.randn(vocab, d, generator=g, dtype=dtype) * 0.1,
        "attn": [
            (torch.randn(d, m, generator=g, dtype=dtype) / d**0.5,
             torch.randn(d, m, generator=g, dtype=dtype) / d**0.5,
             torch.randn(d, d, generator=g, dtype=dtype) / d**0.5)
            for _ in range(L)
        ],
        "W_out": torch.randn(d, vocab, generator=g, dtype=dtype) / d**0.5,
    }


def forward_lm(params, ids, pe, fm="elu+1"):
    """Returns activities a[0..L], per-layer attention outputs z[1..L], logits."""
    a = [params["Emb"][ids] + pe]
    z = [None]
    for (WQ, WK, WV) in params["attn"]:
        fwd = causal_attention_forward(a[-1], WQ, WK, WV, fm)
        z.append(fwd.z)
        a.append(a[-1] + fwd.z)         # residual
    logits = a[-1] @ params["W_out"]
    return a, z, logits


def backprop_grads(params, ids, targets, pe, fm="elu+1"):
    Emb = params["Emb"].detach().clone().requires_grad_(True)
    attn = [tuple(w.detach().clone().requires_grad_(True) for w in layer) for layer in params["attn"]]
    Wout = params["W_out"].detach().clone().requires_grad_(True)
    a = Emb[ids] + pe
    for (WQ, WK, WV) in attn:
        a = a + causal_attention_forward(a, WQ, WK, WV, fm).z
    logits = a @ Wout
    loss = ce_loss(logits, targets)
    loss.backward()
    grads = {"Emb": Emb.grad.detach().clone(),
             "W_out": Wout.grad.detach().clone(),
             "attn": [tuple(w.grad.detach().clone() for w in layer) for layer in attn]}
    return grads, loss.item(), accuracy(logits, targets)


def _layer_vjp(a_in, WQ, WK, WV, e_out, fm):
    """(d attn(a)/da)^T e_out, linearized at a_in (one-layer transpose)."""
    a = a_in.detach().clone().requires_grad_(True)
    fwd = causal_attention_forward(a, WQ, WK, WV, fm)
    fwd.z.backward(e_out)
    return a.grad.detach()


def pc_grads(params, ids, targets, pe, fm="elu+1", T=8, lr=0.3, feedback_mats=None):
    """Finite-budget fixed-prediction PC gradients. T = inference steps.

    feedback_mats: if None, the top-down feedback uses each layer's exact transpose
    (weight transport). If a list of L fixed (d,d) matrices is given, feedback uses
    those random matrices instead -- feedback alignment, i.e. PC *without* weight
    transport (extension 4). feedback_mats[l] carries the error into activity a[l].
    """
    B, N = ids.shape
    L = len(params["attn"])
    a_ff, z_ff, logits = forward_lm(params, ids, pe, fm)
    loss = ce_loss(logits, targets)

    def feedback(target_l, e_above, a_lin, layer_w):
        # I (residual) + [ exact transpose | random ] applied to the error above
        if feedback_mats is None:
            return e_above + _layer_vjp(a_lin, *layer_w, e_above, fm)
        return e_above + e_above @ feedback_mats[target_l]

    # output error injected at the top activity (fixed at feedforward)
    p = logits.softmax(dim=-1)
    y = torch.zeros_like(p); y.scatter_(-1, targets.unsqueeze(-1), 1.0)
    e_out = (p - y) / (B * N)                                   # (B,N,vocab)
    g_head = e_out @ params["W_out"].T                         # dL/da[L], fixed

    # relax the free activities a[1..L] for T steps
    a = [act.clone() for act in a_ff]
    for _ in range(T):
        e = [None] + [a[l] - a_ff[l] for l in range(1, L + 1)]
        new_a = list(a)
        for l in range(1, L + 1):
            if l < L:
                dFda = e[l] - feedback(l, e[l + 1], a_ff[l], params["attn"][l])
            else:
                dFda = e[l] + g_head
            new_a[l] = a[l] - lr * dFda
        a = new_a
    e = [None] + [a[l] - a_ff[l] for l in range(1, L + 1)]

    # weight grads from the settled errors
    attn_grads = []
    for l in range(1, L + 1):
        fwd_l = causal_attention_forward(a_ff[l - 1], *params["attn"][l - 1], fm)
        g = causal_pc_gradients(fwd_l, z_ff[l] + e[l])         # eps = e[l]
        attn_grads.append((g["W_Q"].sum(0), g["W_K"].sum(0), g["W_V"].sum(0)))

    gW_out = torch.einsum("bnd,bnv->dv", a_ff[L], e_out)

    # embedding error: feed dL/da[1] = -e[1] back through layer 1
    dLda0 = feedback(0, -e[1], a_ff[0], params["attn"][0])
    gEmb = torch.zeros_like(params["Emb"])
    gEmb.index_add_(0, ids.reshape(-1), dLda0.reshape(-1, dLda0.shape[-1]))

    grads = {"Emb": gEmb, "W_out": gW_out, "attn": attn_grads}
    return grads, loss.item(), accuracy(logits, targets)


def flat_dev(g1, g2):
    """Worst abs deviation between two grad dicts (for the gate)."""
    d = max((g1["Emb"] - g2["Emb"]).abs().max().item(),
            (g1["W_out"] - g2["W_out"]).abs().max().item())
    for la, lb in zip(g1["attn"], g2["attn"]):
        for wa, wb in zip(la, lb):
            d = max(d, (wa - wb).abs().max().item())
    return d
