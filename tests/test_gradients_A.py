"""Path A correctness gate.

Same idea as the Path B gate, for real softmax attention: check that the analytic
local PC gradients (pc_updates_A.py) match softmax autograd (softmax_attn.py) for
W_Q, W_K, W_V, as gamma -> large and after the normalizer relaxes.

  python tests/test_gradients_A.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pc_attention.softmax_attn import softmax_attention_forward, softmax_backprop_per_example
from pc_attention.pc_updates_A import pc_gradients_A

torch.set_default_dtype(torch.float64)

ATOL, RTOL = 1e-3, 1e-2
KEYS = ("W_Q", "W_K", "W_V")


def make_problem(d=16, d_k=16, d_v=16, B=4, N=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, N, d, generator=g)
    W_Q = torch.randn(d, d_k, generator=g) / d**0.5
    W_K = torch.randn(d, d_k, generator=g) / d**0.5
    W_V = torch.randn(d, d_v, generator=g) / d**0.5
    x_out = torch.randn(B, N, d_v, generator=g)
    return dict(X=X, W_Q=W_Q, W_K=W_K, W_V=W_V, x_out=x_out)


def _bp(p):
    return softmax_backprop_per_example(p["X"], p["W_Q"], p["W_K"], p["W_V"], p["x_out"])


def _pc(p, method, gamma, n_steps=300):
    fwd = softmax_attention_forward(p["X"], p["W_Q"], p["W_K"], p["W_V"])
    return pc_gradients_A(fwd, p["x_out"], gamma=gamma, method=method,
                          n_steps=n_steps, return_info=True)


def max_dev(a, b):
    return (a - b).abs().max().item()


def _report(g_pc, g_bp, label):
    print(f"\n[{label}] max |g_PC - g_BP| per tensor (atol={ATOL}, rtol={RTOL}):")
    ok = True
    for k in KEYS:
        md = max_dev(g_pc[k], g_bp[k])
        rd = md / (g_bp[k].abs().max().item() + 1e-30)
        passed = torch.allclose(g_pc[k], g_bp[k], atol=ATOL, rtol=RTOL)
        ok = ok and passed
        print(f"   {k:>3}: max abs dev = {md:.3e}   rel dev = {rd:.3e}   {'PASS' if passed else 'FAIL'}")
    return ok


def test_closed_form_matches_softmax_autograd():
    p = make_problem()
    g_bp = _bp(p)
    g_pc, _ = _pc(p, method="closed_form", gamma=float("inf"))
    for k in KEYS:
        assert torch.allclose(g_pc[k], g_bp[k], atol=ATOL, rtol=RTOL), (
            f"{k}: max dev {max_dev(g_pc[k], g_bp[k]):.3e}"
        )


def test_relaxation_matches_softmax_autograd_large_gamma():
    p = make_problem()
    g_bp = _bp(p)
    g_pc, info = _pc(p, method="relax", gamma=1e5, n_steps=500)
    assert info.converged, f"normalizer relaxation did not converge: res={info.res:.2e}"
    for k in KEYS:
        assert torch.allclose(g_pc[k], g_bp[k], atol=ATOL, rtol=RTOL), (
            f"{k}: max dev {max_dev(g_pc[k], g_bp[k]):.3e}"
        )


def main():
    print("=" * 70)
    print("PATH A CORRECTNESS GATE  (d=16, d_k=16, batch=4, N=8, float64)")
    print("=" * 70)
    p = make_problem()
    g_bp = _bp(p)

    g_cf, _ = _pc(p, method="closed_form", gamma=float("inf"))
    ok_cf = _report(g_cf, g_bp, "closed form (gamma -> inf)")

    g_rx, info = _pc(p, method="relax", gamma=1e5, n_steps=500)
    print(f"\n   normalizer relaxation: converged={info.converged} in {info.steps} steps (res={info.res:.2e})")
    ok_rx = _report(g_rx, g_bp, "relaxation (gamma=1e5)")

    print("\n" + "-" * 70)
    print("SENSITIVITY: worst weight-grad deviation vs gamma (relax, 500 steps)")
    print("-" * 70)
    for gamma in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
        g, info = _pc(p, method="relax", gamma=gamma, n_steps=500)
        worst = max(max_dev(g[k], g_bp[k]) for k in KEYS)
        print(f"   gamma={gamma:>7.0e}: worst dev = {worst:.3e}  (converged={info.converged}, steps={info.steps})")

    print("\n" + "=" * 70)
    passed = ok_cf and ok_rx
    print(f"RESULT: {'ALL PASS' if passed else 'FAILURE'}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
