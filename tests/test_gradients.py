"""The correctness gate.

This is the checkpoint the whole milestone hinges on. On a small random layer
(d=16, m=32, batch=4, N=8) it checks that the hand-derived PC gradients in
pc_updates.py agree with PyTorch autograd (backprop_ref.py) for W_Q, W_K, W_V and
the two accumulator error signals S, u. If these don't match, the derivation is
wrong and nothing downstream is trustworthy.

Two limits are checked: the closed-form beta -> infinity gradients (should match to
machine precision), and the gradients read off the actual relaxation loop at a
large but finite beta (should match up to a small, well-understood residual).

Run it directly for a full report of max deviations plus beta / step-count
sensitivity:  python tests/test_gradients.py
It also exposes pytest-style test_* functions if you have pytest installed.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pc_attention.linear_attn import linear_attention_forward
from pc_attention.backprop_ref import (
    backprop_gradients,
    backprop_gradients_per_example,
)
from pc_attention.pc_updates import pc_gradients

# float64 so the tolerances test the maths, not float32 rounding.
torch.set_default_dtype(torch.float64)

ATOL, RTOL = 1e-3, 1e-2
WEIGHT_KEYS = ("W_Q", "W_K", "W_V")
ALL_KEYS = ("W_Q", "W_K", "W_V", "S", "u")


def make_problem(d=16, m=32, B=4, N=8, seed=0, feature_map="elu+1"):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, N, d, generator=g)
    W_Q = torch.randn(d, m, generator=g) * (1.0 / d**0.5)
    W_K = torch.randn(d, m, generator=g) * (1.0 / d**0.5)
    W_V = torch.randn(d, d, generator=g) * (1.0 / d**0.5)
    x_out = torch.randn(B, N, d, generator=g)
    return dict(X=X, W_Q=W_Q, W_K=W_K, W_V=W_V, x_out=x_out, feature_map=feature_map)


def _bp_per_example(p):
    """Autograd gradients for all five tensors, one per example.

    Weight grads come from the per-example helper; the S, u grads come from the
    pooled call (its per-example batch slices are already what we want).
    """
    w = backprop_gradients_per_example(
        p["X"], p["W_Q"], p["W_K"], p["W_V"], p["x_out"], p["feature_map"]
    )
    full = backprop_gradients(
        p["X"], p["W_Q"], p["W_K"], p["W_V"], p["x_out"], p["feature_map"]
    )
    return {"W_Q": w["W_Q"], "W_K": w["W_K"], "W_V": w["W_V"],
            "S": full["S"], "u": full["u"]}


def _pc_per_example(p, method, beta, n_steps=200):
    fwd = linear_attention_forward(
        p["X"], p["W_Q"], p["W_K"], p["W_V"], feature_map=p["feature_map"]
    )
    return pc_gradients(fwd, p["x_out"], beta=beta, method=method,
                        n_steps=n_steps, return_info=True)


def max_dev(a, b):
    return (a - b).abs().max().item()


def rel_dev(a, b):
    denom = b.abs().max().item()
    return max_dev(a, b) / denom if denom > 0 else max_dev(a, b)


def _report(g_pc, g_bp, label):
    print(f"\n[{label}] max |g_PC - g_BP| per tensor (atol={ATOL}, rtol={RTOL}):")
    ok = True
    for k in ALL_KEYS:
        md, rd = max_dev(g_pc[k], g_bp[k]), rel_dev(g_pc[k], g_bp[k])
        passed = torch.allclose(g_pc[k], g_bp[k], atol=ATOL, rtol=RTOL)
        ok = ok and passed
        print(f"   {k:>3}: max abs dev = {md:.3e}   rel dev = {rd:.3e}   "
              f"{'PASS' if passed else 'FAIL'}")
    return ok


# ---- pytest entry points ----------------------------------------------------
def test_closed_form_matches_autograd():
    p = make_problem()
    g_bp = _bp_per_example(p)
    g_pc, _ = _pc_per_example(p, method="closed_form", beta=float("inf"))
    for k in ALL_KEYS:
        assert torch.allclose(g_pc[k], g_bp[k], atol=ATOL, rtol=RTOL), (
            f"{k}: max dev {max_dev(g_pc[k], g_bp[k]):.3e}"
        )


def test_relaxation_matches_autograd_large_beta():
    p = make_problem()
    g_bp = _bp_per_example(p)
    g_pc, info = _pc_per_example(p, method="relax", beta=1e4, n_steps=300)
    assert info.converged, f"relaxation did not converge: res_S={info.res_S:.2e}"
    for k in ALL_KEYS:
        assert torch.allclose(g_pc[k], g_bp[k], atol=ATOL, rtol=RTOL), (
            f"{k}: max dev {max_dev(g_pc[k], g_bp[k]):.3e}"
        )


def test_relu_feature_map():
    p = make_problem(feature_map="relu", seed=3)
    g_bp = _bp_per_example(p)
    g_pc, _ = _pc_per_example(p, method="closed_form", beta=float("inf"))
    for k in WEIGHT_KEYS:
        assert torch.allclose(g_pc[k], g_bp[k], atol=ATOL, rtol=RTOL), (
            f"{k}: max dev {max_dev(g_pc[k], g_bp[k]):.3e}"
        )


# ---- standalone report ------------------------------------------------------
def main():
    print("=" * 70)
    print("CORRECTNESS GATE  (d=16, m=32, batch=4, N=8, float64)")
    print("=" * 70)
    p = make_problem()
    g_bp = _bp_per_example(p)

    g_cf, _ = _pc_per_example(p, method="closed_form", beta=float("inf"))
    ok_cf = _report(g_cf, g_bp, "closed form (beta -> inf)")

    g_rx, info = _pc_per_example(p, method="relax", beta=1e4, n_steps=300)
    print(f"\n   relaxation: converged={info.converged} in {info.steps} steps "
          f"(res_S={info.res_S:.2e}, res_u={info.res_u:.2e})")
    ok_rx = _report(g_rx, g_bp, "relaxation (beta=1e4)")

    # How much the finite-beta relaxation is off by, as beta grows: the residual
    # should shrink like 1/beta.
    print("\n" + "-" * 70)
    print("SENSITIVITY: max weight-grad deviation vs beta (relax, 300 steps)")
    print("-" * 70)
    for beta in (1e1, 1e2, 1e3, 1e4, 1e6):
        g, info = _pc_per_example(p, method="relax", beta=beta, n_steps=300)
        worst = max(max_dev(g[k], g_bp[k]) for k in WEIGHT_KEYS)
        print(f"   beta={beta:>7.0e}: worst weight dev = {worst:.3e}   "
              f"(converged={info.converged}, steps={info.steps})")

    # How few inference steps we can get away with (it converges very fast).
    print("\n" + "-" * 70)
    print("SENSITIVITY: max weight-grad deviation vs #inference steps (beta=1e4)")
    print("-" * 70)
    for ns in (1, 2, 5, 10, 50, 300):
        g, info = _pc_per_example(p, method="relax", beta=1e4, n_steps=ns)
        worst = max(max_dev(g[k], g_bp[k]) for k in WEIGHT_KEYS)
        print(f"   steps={ns:>4}: worst weight dev = {worst:.3e}   "
              f"(res_S={info.res_S:.2e})")

    print("\n" + "=" * 70)
    passed = ok_cf and ok_rx
    print(f"RESULT: {'ALL PASS' if passed else 'FAILURE — see FAIL rows above'}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
