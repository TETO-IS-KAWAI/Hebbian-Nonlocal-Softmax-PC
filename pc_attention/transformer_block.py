"""A real pre-norm transformer block trained by predictive coding (extension 2).

Everything so far was a bare attention layer. A real block adds three things PC has
never been shown to handle together: multi-head attention, a feed-forward (MLP)
sub-block, and LayerNorm -- another global operation, this time over the feature
dimension.

The key idea that makes this tractable: put the PC activity (error) nodes on the
**residual stream**, between whole sub-layers, not inside them. Each sub-layer

    a^{k} = a^{k-1} + sublayer_k(a^{k-1})            (pre-norm: sublayer = f(LN(x)))

is then a self-contained deterministic map. Fixed-prediction PC needs only two
local operations per sub-layer: its weight gradient given the error at its output,
and its input transpose for feedback -- both computed by autograd on that *single*
sub-layer (never the whole stack). LayerNorm and the MLP nonlinearity just ride
along inside a sub-layer; they need no bespoke PC rule, because the error node sits
after the sub-layer, on the residual stream. That is the whole point.

At converged inference this equals backprop (checked in experiments/02/run.py); the
value of extension 2 is showing the framework scales to the full block unchanged.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

from .char_lm import ce_loss, accuracy

Tensor = torch.Tensor


# ---- building blocks --------------------------------------------------------
def layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * gamma + beta


def causal_linear_mha(x, W_Q, W_K, W_V, W_O, H):
    """Multi-head causal linear attention (phi = elu+1), heads run in parallel."""
    B, N, d = x.shape
    dh = d // H
    phi = lambda t: F.elu(t) + 1.0
    Q = (x @ W_Q).view(B, N, H, dh)
    K = (x @ W_K).view(B, N, H, dh)
    V = (x @ W_V).view(B, N, H, dh)
    pq, pk = phi(Q), phi(K)
    S = torch.einsum("bnhm,bnhd->bnhmd", pk, V).cumsum(dim=1)     # prefix sum per head
    u = pk.cumsum(dim=1)
    num = torch.einsum("bnhmd,bnhm->bnhd", S, pq)
    den = torch.einsum("bnhm,bnhm->bnh", u, pq).unsqueeze(-1) + 1e-6
    z = (num / den).reshape(B, N, d)
    return z @ W_O


def ffn(x, W1, b1, W2, b2):
    return F.gelu(x @ W1 + b1) @ W2 + b2


# ---- model as (embedding, residual sub-layers, head) ------------------------
def init_params(vocab, d, H, d_ff, L, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    def r(*shape, s=1.0):
        return torch.randn(*shape, generator=g, dtype=dtype) * s
    p = {"Emb": r(vocab, d, s=0.1), "W_out": r(d, vocab, s=1 / d**0.5),
         "lnf_g": torch.ones(d, dtype=dtype), "lnf_b": torch.zeros(d, dtype=dtype)}
    for l in range(L):
        p[f"ln1_g{l}"] = torch.ones(d, dtype=dtype); p[f"ln1_b{l}"] = torch.zeros(d, dtype=dtype)
        p[f"WQ{l}"] = r(d, d, s=1 / d**0.5); p[f"WK{l}"] = r(d, d, s=1 / d**0.5)
        p[f"WV{l}"] = r(d, d, s=1 / d**0.5); p[f"WO{l}"] = r(d, d, s=1 / d**0.5)
        p[f"ln2_g{l}"] = torch.ones(d, dtype=dtype); p[f"ln2_b{l}"] = torch.zeros(d, dtype=dtype)
        p[f"W1{l}"] = r(d, d_ff, s=1 / d**0.5); p[f"b1{l}"] = torch.zeros(d_ff, dtype=dtype)
        p[f"W2{l}"] = r(d_ff, d, s=1 / d_ff**0.5); p[f"b2{l}"] = torch.zeros(d, dtype=dtype)
    return p


def sublayers(L, H):
    """List of (name, param_keys, fn) residual sub-layers, in order."""
    subs = []
    for l in range(L):
        subs.append((
            f"attn{l}",
            [f"ln1_g{l}", f"ln1_b{l}", f"WQ{l}", f"WK{l}", f"WV{l}", f"WO{l}"],
            (lambda p, x, l=l: causal_linear_mha(
                layernorm(x, p[f"ln1_g{l}"], p[f"ln1_b{l}"]),
                p[f"WQ{l}"], p[f"WK{l}"], p[f"WV{l}"], p[f"WO{l}"], H)),
        ))
        subs.append((
            f"ffn{l}",
            [f"ln2_g{l}", f"ln2_b{l}", f"W1{l}", f"b1{l}", f"W2{l}", f"b2{l}"],
            (lambda p, x, l=l: ffn(
                layernorm(x, p[f"ln2_g{l}"], p[f"ln2_b{l}"]),
                p[f"W1{l}"], p[f"b1{l}"], p[f"W2{l}"], p[f"b2{l}"])),
        ))
    return subs


def forward(params, ids, pe, L, H):
    a = [params["Emb"][ids] + pe]
    for _, _, fn in sublayers(L, H):
        a.append(a[-1] + fn(params, a[-1]))
    logits = layernorm(a[-1], params["lnf_g"], params["lnf_b"]) @ params["W_out"]
    return a, logits


def backprop_grads(params, ids, targets, pe, L, H):
    p = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    _, logits = forward(p, ids, pe, L, H)
    loss = ce_loss(logits, targets)
    loss.backward()
    return {k: p[k].grad.detach().clone() for k in params}, loss.item(), accuracy(logits, targets)


def _sublayer_backward(params, keys, fn, x_in, out_grad):
    """One-layer autograd: param grads and input grad for a single sub-layer,
    given the gradient at its output. This is the local weight rule + feedback."""
    p = {k: (params[k].detach().clone().requires_grad_(True) if k in keys else params[k]) for k in params}
    xin = x_in.detach().clone().requires_grad_(True)
    out = fn(p, xin)
    out.backward(out_grad)
    pg = {k: p[k].grad.detach() for k in keys}
    return pg, xin.grad.detach()


def pc_grads(params, ids, targets, pe, L, H, T=64, lr=0.3):
    """Fixed-prediction PC over the residual-stream activities. T = inference steps."""
    subs = sublayers(L, H)
    K = len(subs)
    a_ff, logits = forward(params, ids, pe, L, H)
    loss = ce_loss(logits, targets)

    # head gradient dL/da[K], from one-layer autograd through (final LN, W_out)
    aK = a_ff[K].detach().clone().requires_grad_(True)
    gf = params["lnf_g"].detach().clone().requires_grad_(True)
    bf = params["lnf_b"].detach().clone().requires_grad_(True)
    Wo = params["W_out"].detach().clone().requires_grad_(True)
    lg = ce_loss(layernorm(aK, gf, bf) @ Wo, targets)
    lg.backward()
    g_head = aK.grad.detach()
    head_grads = {"lnf_g": gf.grad.detach(), "lnf_b": bf.grad.detach(), "W_out": Wo.grad.detach()}

    # relax activities a[1..K]
    a = [act.clone() for act in a_ff]
    for _ in range(T):
        e = [None] + [a[k] - a_ff[k] for k in range(1, K + 1)]
        new_a = list(a)
        for k in range(1, K + 1):
            if k < K:
                _, vjp = _sublayer_backward(params, subs[k][1], subs[k][2], a_ff[k], e[k + 1])
                dFda = e[k] - (e[k + 1] + vjp)            # residual feedback
            else:
                dFda = e[k] + g_head
            new_a[k] = a[k] - lr * dFda
        a = new_a
    e = [None] + [a[k] - a_ff[k] for k in range(1, K + 1)]

    # weight grads from settled errors: each sub-layer's local rule (dL/dout = -e[k])
    grads = {k: torch.zeros_like(v) for k, v in params.items()}
    for k in range(1, K + 1):
        pg, _ = _sublayer_backward(params, subs[k - 1][1], subs[k - 1][2], a_ff[k - 1], -e[k])
        for name, val in pg.items():
            grads[name] = val
    for name, val in head_grads.items():
        grads[name] = val

    # embedding: transpose of dL/da[1] = -e[1] through sub-layer 1, plus the residual identity
    _, vjp1 = _sublayer_backward(params, subs[0][1], subs[0][2], a_ff[0], -e[1])
    dLda0 = -e[1] + vjp1
    gEmb = torch.zeros_like(params["Emb"])
    gEmb.index_add_(0, ids.reshape(-1), dLda0.reshape(-1, dLda0.shape[-1]))
    grads["Emb"] = gEmb

    return grads, loss.item(), accuracy(logits, targets)


def worst_dev(g1, g2):
    return max((g1[k] - g2[k]).abs().max().item() for k in g1)
