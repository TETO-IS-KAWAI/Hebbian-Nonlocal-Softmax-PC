"""Correctness gate for causal linear attention (M3 foundation).

The forward is a prefix scan, the gradient a suffix scan. Check the closed-form
local PC gradients against autograd on the causal forward.

  python tests/test_gradients_causal.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pc_attention.causal_linear_attn import causal_attention_forward, causal_pc_gradients

torch.set_default_dtype(torch.float64)
ATOL, RTOL = 1e-3, 1e-2
KEYS = ("W_Q", "W_K", "W_V")


def make(d=16, m=32, B=4, N=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, N, d, generator=g),
        torch.randn(d, m, generator=g) / d**0.5,
        torch.randn(d, m, generator=g) / d**0.5,
        torch.randn(d, d, generator=g) / d**0.5,
        torch.randn(B, N, d, generator=g),
    )


def _bp(X, WQ, WK, WV, xo):
    WQ, WK, WV = (w.clone().requires_grad_(True) for w in (WQ, WK, WV))
    fwd = causal_attention_forward(X, WQ, WK, WV)
    (0.5 * (xo - fwd.z).pow(2).sum()).backward()
    return {"W_Q": WQ.grad, "W_K": WK.grad, "W_V": WV.grad}


def _pc(X, WQ, WK, WV, xo):
    fwd = causal_attention_forward(X, WQ, WK, WV)
    g = causal_pc_gradients(fwd, xo)
    return {k: g[k].sum(dim=0) for k in KEYS}


def test_causal_pc_matches_autograd():
    X, WQ, WK, WV, xo = make()
    g_bp, g_pc = _bp(X, WQ, WK, WV, xo), _pc(X, WQ, WK, WV, xo)
    for k in KEYS:
        assert torch.allclose(g_pc[k], g_bp[k], atol=ATOL, rtol=RTOL), (
            f"{k}: max dev {(g_pc[k] - g_bp[k]).abs().max().item():.2e}"
        )


def test_causality():
    """A future token must not affect a past position's output."""
    X, WQ, WK, WV, _ = make(N=8, seed=1)
    fwd = causal_attention_forward(X, WQ, WK, WV)
    z0 = fwd.z.clone()
    X2 = X.clone()
    X2[:, 5:] += torch.randn_like(X2[:, 5:])       # perturb positions 5..7
    z2 = causal_attention_forward(X2, WQ, WK, WV).z
    assert torch.allclose(z0[:, :5], z2[:, :5], atol=1e-10), "future leaked into the past"


def main():
    print("=" * 64)
    print("CAUSAL LINEAR ATTENTION GATE (d=16, m=32, batch=4, N=8, float64)")
    print("=" * 64)
    X, WQ, WK, WV, xo = make()
    g_bp, g_pc = _bp(X, WQ, WK, WV, xo), _pc(X, WQ, WK, WV, xo)
    ok = True
    for k in KEYS:
        dev = (g_pc[k] - g_bp[k]).abs().max().item()
        passed = torch.allclose(g_pc[k], g_bp[k], atol=ATOL, rtol=RTOL)
        ok = ok and passed
        print(f"   {k}: max abs dev = {dev:.3e}   {'PASS' if passed else 'FAIL'}")
    # causality
    try:
        test_causality(); print("   causality: PASS (future does not affect the past)")
    except AssertionError as e:
        ok = False; print(f"   causality: FAIL ({e})")
    print("=" * 64)
    print(f"RESULT: {'ALL PASS' if ok else 'FAILURE'}")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
