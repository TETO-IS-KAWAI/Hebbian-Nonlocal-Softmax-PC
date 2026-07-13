"""M2 experiment: the delta(N, L) surface for a stack of linear-attention layers.

Two things come out of this:

  1. With inference run to convergence, fixed-prediction PC matches backprop at
     every depth and every length -- delta collapses to ~0. So there is no intrinsic
     depth or length penalty for Path B once inference has settled.

  2. Under a *bounded* inference budget, a depth penalty appears, because the output
     error only travels about one layer per inference step. With too few steps the
     lower layers never hear about the error and their gradients are undefined, so
     delta grows with depth L. It stays flat in N, confirming Path B's "b ~ 0"
     prediction (the sequence-length coefficient of the error is zero).

Outputs:
  results/m2_delta_NL.csv        raw delta(N, L) at the fixed budget
  results/m2_delta_heatmap.png   delta(N, L) heatmap (grows down L, flat across N)
  results/m2_convergence.png     delta vs inference steps for a few depths (-> 0)

float64 throughout, same reasoning as M1. Use --quick for a small sweep.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pc_attention.deep_linear_attn import deep_backprop_grads_per_example
from pc_attention.pc_updates_deep import deep_pc_grads_per_example
from pc_attention.metrics import gradient_alignment

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
WEIGHT_KEYS = ("W_Q", "W_K", "W_V")


def make_problem(d, m, B, N, L, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, N, d, generator=g, dtype=dtype)
    W = [(torch.randn(d, m, generator=g, dtype=dtype) / d**0.5,
          torch.randn(d, m, generator=g, dtype=dtype) / d**0.5,
          torch.randn(d, d, generator=g, dtype=dtype) / d**0.5) for _ in range(L)]
    T = torch.randn(B, N, d, generator=g, dtype=dtype)   # dense regression target
    return X, W, T


def mean_delta(X, W, T, n_steps, lr, tol):
    """One scalar delta: averaged over the three weight tensors and all layers."""
    g_bp = deep_backprop_grads_per_example(X, W, T)
    g_pc, steps, res = deep_pc_grads_per_example(X, W, T, lr=lr, n_steps=n_steps, tol=tol)
    vals = [gradient_alignment(g_pc[l], g_bp[l])[k] for l in range(len(W)) for k in WEIGHT_KEYS]
    return sum(vals) / len(vals), steps, res


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--Ns", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--Ls", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--budget", type=int, default=4, help="fixed inference steps for the surface")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--reps", type=int, default=3)
    cfg = ap.parse_args()
    cfg.dtype = torch.float64
    if cfg.quick:
        cfg.Ns = [4, 16, 64]
        cfg.Ls = [1, 2, 4, 8]
        cfg.reps = 2
    return cfg


def make_plots(cfg, surface, conv_L, conv_steps, conv_curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # heatmap: rows = L, cols = N
    fig, ax = plt.subplots(figsize=(7.5, 5))
    M = np.array([[surface[(N, L)] for N in cfg.Ns] for L in cfg.Ls])
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cfg.Ns))); ax.set_xticklabels(cfg.Ns)
    ax.set_yticks(range(len(cfg.Ls))); ax.set_yticklabels(cfg.Ls)
    ax.set_xlabel("sequence length N"); ax.set_ylabel("depth L")
    ax.set_title(f"delta(N, L) at fixed inference budget = {cfg.budget} steps\n"
                 f"(grows with depth, flat across N)")
    for i in range(len(cfg.Ls)):
        for j in range(len(cfg.Ns)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="delta = E[1 - cos(g_PC, g_BP)]")
    fig.tight_layout()
    p1 = os.path.join(RESULTS_DIR, "m2_delta_heatmap.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    # convergence: delta vs inference steps for a few L (fixed N)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for L, curve in zip(conv_L, conv_curves):
        ax.plot(conv_steps, [max(c, 1e-16) for c in curve], marker="o", label=f"L={L}")
    ax.set_yscale("log"); ax.set_xscale("log", base=2)
    ax.set_xlabel("inference steps"); ax.set_ylabel("mean delta over layers")
    ax.set_title("Inference budget vs alignment (deeper needs more steps, then delta -> 0)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    p2 = os.path.join(RESULTS_DIR, "m2_convergence.png")
    fig.savefig(p2, dpi=130); plt.close(fig)
    return p1, p2


def main():
    cfg = parse_args()
    torch.set_default_dtype(cfg.dtype)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"config: Ns={cfg.Ns} Ls={cfg.Ls} d={cfg.d} m={cfg.m} batch={cfg.batch} "
          f"budget={cfg.budget} lr={cfg.lr} reps={cfg.reps}")

    # --- delta(N, L) surface at the fixed budget, plus full-relaxation check ---
    surface, full_relax = {}, {}
    print("\ndelta(N, L)  [fixed budget / full relaxation]")
    print("   L \\ N  " + "".join(f"{N:>16}" for N in cfg.Ns))
    for L in cfg.Ls:
        cells = []
        for N in cfg.Ns:
            dsum, dfull = 0.0, 0.0
            for r in range(cfg.reps):
                X, W, T = make_problem(cfg.d, cfg.m, cfg.batch, N, L, seed=700 + r, dtype=cfg.dtype)
                d_fixed, _, _ = mean_delta(X, W, T, n_steps=cfg.budget, lr=cfg.lr, tol=0.0)
                d_full, _, _ = mean_delta(X, W, T, n_steps=2000, lr=cfg.lr, tol=1e-11)
                dsum += d_fixed / cfg.reps
                dfull += d_full / cfg.reps
            surface[(N, L)] = dsum
            full_relax[(N, L)] = dfull
            cells.append(f"{dsum:.2f}/{dfull:.0e}")
        print(f"   L={L:<4} " + "".join(f"{c:>16}" for c in cells))

    worst_full = max(full_relax.values())
    print(f"\nfull-relaxation worst delta over the grid: {worst_full:.2e}  "
          f"(fixed-prediction PC == backprop at all N, L once inference converges)")

    # --- convergence curves: delta vs inference steps at fixed N ---
    conv_N = cfg.Ns[len(cfg.Ns) // 2]
    conv_steps = [1, 2, 4, 8, 16, 32, 64]
    conv_L = [L for L in cfg.Ls if L >= 2]
    conv_curves = []
    print(f"\nconvergence at N={conv_N}: mean delta vs inference steps")
    for L in conv_L:
        curve = []
        for ns in conv_steps:
            dsum = 0.0
            for r in range(cfg.reps):
                X, W, T = make_problem(cfg.d, cfg.m, cfg.batch, conv_N, L, seed=700 + r, dtype=cfg.dtype)
                dv, _, _ = mean_delta(X, W, T, n_steps=ns, lr=cfg.lr, tol=0.0)
                dsum += dv / cfg.reps
            curve.append(dsum)
        conv_curves.append(curve)
        print(f"   L={L:<3} " + " ".join(f"{c:.2f}" for c in curve))

    # --- CSV ---
    csv_path = os.path.join(RESULTS_DIR, "m2_delta_NL.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "L", "delta_fixed_budget", "delta_full_relax", "budget"])
        for L in cfg.Ls:
            for N in cfg.Ns:
                w.writerow([N, L, surface[(N, L)], full_relax[(N, L)], cfg.budget])

    p1, p2 = make_plots(cfg, surface, conv_L, conv_steps, conv_curves)
    print(f"\nwrote {csv_path}\nwrote {p1}\nwrote {p2}")

    # N-flatness summary
    print("\nN-flatness (Path B b~0): std of delta across N, per depth L (fixed budget):")
    import statistics
    for L in cfg.Ls:
        vals = [surface[(N, L)] for N in cfg.Ns]
        sd = statistics.pstdev(vals)
        print(f"   L={L:<3} delta across N = {[round(v,3) for v in vals]}  std={sd:.2e}")


if __name__ == "__main__":
    main()
