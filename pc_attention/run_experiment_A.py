"""Path A experiment: softmax attention trained by PC with a normalizer state node.

Three things:

  1. delta(N) for Path A, both in the gamma -> inf limit (closed form) and at a
     finite gamma. Does keeping the real softmax cost anything in N?
  2. the gamma-scan: delta as a function of the normalizer precision gamma, at a
     few N. Path C predicts larger gamma shrinks delta.
  3. a side-by-side with Path B on the *same* inputs and weights (d_k = m so the
     weight tensors line up), i.e. real softmax vs the linear surrogate.

delta = E[1 - cos(g_PC, g_BP)] per weight tensor, averaged over reps and tensors.
float64 throughout. Use --quick for a small sweep.

Outputs: results/pathA_delta_vs_N.png, results/pathA_gamma_scan.png,
         results/pathA_delta.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pc_attention.softmax_attn import softmax_attention_forward, softmax_backprop_per_example
from pc_attention.pc_updates_A import pc_gradients_A
from pc_attention.linear_attn import linear_attention_forward
from pc_attention.backprop_ref import backprop_gradients_per_example
from pc_attention.pc_updates import pc_gradients
from pc_attention.metrics import gradient_alignment

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
KEYS = ("W_Q", "W_K", "W_V")


def make_weights(d, d_k, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    W_Q = torch.randn(d, d_k, generator=g, dtype=dtype) / d**0.5
    W_K = torch.randn(d, d_k, generator=g, dtype=dtype) / d**0.5
    W_V = torch.randn(d, d_k, generator=g, dtype=dtype) / d**0.5
    return W_Q, W_K, W_V


def mean_over(dct):
    return sum(dct[k] for k in KEYS) / len(KEYS)


def pathA_delta(X, W, x_out, gamma, method, n_steps=200):
    WQ, WK, WV = W
    fwd = softmax_attention_forward(X, WQ, WK, WV)
    g_bp = softmax_backprop_per_example(X, WQ, WK, WV, x_out)
    g_pc = pc_gradients_A(fwd, x_out, gamma=gamma, method=method, n_steps=n_steps)
    return mean_over(gradient_alignment(g_pc, g_bp))


def pathB_delta(X, W, x_out):
    WQ, WK, WV = W
    fwd = linear_attention_forward(X, WQ, WK, WV)
    g_bp = backprop_gradients_per_example(X, WQ, WK, WV, x_out)
    g_pc = pc_gradients(fwd, x_out, method="closed_form")
    return mean_over(gradient_alignment(g_pc, g_bp))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--Ns", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--d-k", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=1e4, help="finite gamma for the delta(N) curve")
    ap.add_argument("--gammas", type=float, nargs="+", default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--gamma-scan-Ns", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--reps", type=int, default=4)
    cfg = ap.parse_args()
    cfg.dtype = torch.float64
    if cfg.quick:
        cfg.Ns = [4, 16, 64]
        cfg.gamma_scan_Ns = [8, 64]
        cfg.reps = 2
    return cfg


def make_plots(cfg, rows, scan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = [r["N"] for r in rows]
    floor = 1e-18

    # delta(N): Path A closed form, Path A finite gamma, Path B closed form
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(Ns, [max(r["A_closed"], floor) for r in rows], marker="o", label="Path A softmax (gamma->inf)")
    ax.plot(Ns, [max(r["A_finite"], floor) for r in rows], marker="s",
            label=f"Path A softmax (gamma={cfg.gamma:g})")
    ax.plot(Ns, [max(r["B_closed"], floor) for r in rows], marker="^", label="Path B linear (gamma->inf)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("sequence length N"); ax.set_ylabel("delta = E[1 - cos(g_PC, g_BP)]")
    ax.set_title("delta(N): PC recovers backprop for softmax (Path A) and linear (Path B)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    p1 = os.path.join(RESULTS_DIR, "pathA_delta_vs_N.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    # gamma-scan
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for N in cfg.gamma_scan_Ns:
        ax.plot(cfg.gammas, [max(scan[(N, gm)], floor) for gm in cfg.gammas], marker="o", label=f"N={N}")
    # reference slope ~ 1/gamma^2  (delta = 1-cos is second order in the O(1/gamma)
    # gradient error, so it falls twice as fast as the error itself)
    g0 = cfg.gammas[0]
    ref = [scan[(cfg.gamma_scan_Ns[0], g0)] * (g0 / gm) ** 2 for gm in cfg.gammas]
    ax.plot(cfg.gammas, [max(r, floor) for r in ref], "k--", alpha=0.5, label="~ 1/gamma^2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("normalizer precision gamma"); ax.set_ylabel("delta")
    ax.set_title("Path A gamma-scan: delta shrinks as 1/gamma^2, flat in N (Path C prediction)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    p2 = os.path.join(RESULTS_DIR, "pathA_gamma_scan.png")
    fig.savefig(p2, dpi=130); plt.close(fig)
    return p1, p2


def main():
    cfg = parse_args()
    torch.set_default_dtype(cfg.dtype)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"config: Ns={cfg.Ns} d={cfg.d} d_k={cfg.d_k} batch={cfg.batch} "
          f"gamma={cfg.gamma:g} reps={cfg.reps}")

    # --- delta(N) ---
    print("\n   N | Path A (gamma->inf) | Path A (finite) | Path B (gamma->inf)")
    rows = []
    for N in cfg.Ns:
        A_c = A_f = B_c = 0.0
        for r in range(cfg.reps):
            g = torch.Generator().manual_seed(4000 + r)
            X = torch.randn(cfg.batch, N, cfg.d, generator=g, dtype=cfg.dtype)
            x_out = torch.randn(cfg.batch, N, cfg.d_k, generator=g, dtype=cfg.dtype)
            W = make_weights(cfg.d, cfg.d_k, seed=5000 + r, dtype=cfg.dtype)
            A_c += pathA_delta(X, W, x_out, gamma=float("inf"), method="closed_form") / cfg.reps
            A_f += pathA_delta(X, W, x_out, gamma=cfg.gamma, method="relax", n_steps=200) / cfg.reps
            B_c += pathB_delta(X, W, x_out) / cfg.reps
        rows.append({"N": N, "A_closed": A_c, "A_finite": A_f, "B_closed": B_c})
        print(f"  {N:>4} | {A_c:>18.2e} | {A_f:>15.2e} | {B_c:>18.2e}")

    # --- gamma-scan ---
    print("\ngamma-scan: delta vs gamma (finite-gamma relaxation)")
    scan = {}
    for N in cfg.gamma_scan_Ns:
        line = []
        for gm in cfg.gammas:
            dsum = 0.0
            for r in range(cfg.reps):
                g = torch.Generator().manual_seed(4000 + r)
                X = torch.randn(cfg.batch, N, cfg.d, generator=g, dtype=cfg.dtype)
                x_out = torch.randn(cfg.batch, N, cfg.d_k, generator=g, dtype=cfg.dtype)
                W = make_weights(cfg.d, cfg.d_k, seed=5000 + r, dtype=cfg.dtype)
                dsum += pathA_delta(X, W, x_out, gamma=gm, method="relax", n_steps=500) / cfg.reps
            scan[(N, gm)] = dsum
            line.append(f"{dsum:.1e}")
        print(f"   N={N:>4}: " + "  ".join(f"g={gm:.0e}:{v}" for gm, v in zip(cfg.gammas, line)))

    # --- CSV ---
    csv_path = os.path.join(RESULTS_DIR, "pathA_delta.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "A_closed", "A_finite_gamma", "gamma_finite", "B_closed"])
        for row in rows:
            w.writerow([row["N"], row["A_closed"], row["A_finite"], cfg.gamma, row["B_closed"]])

    p1, p2 = make_plots(cfg, rows, scan)
    print(f"\nwrote {csv_path}\nwrote {p1}\nwrote {p2}")

    # summary
    worstA = max(r["A_closed"] for r in rows)
    print(f"\nPath A closed-form worst delta over N: {worstA:.2e}  "
          f"(softmax PC == softmax backprop, flat in N)")
    print("Locality budget: Path A = 1 shared per-token scalar (c_i, the divisive-"
          "normalization pool); Path B = 2 shared states (S, u).")


if __name__ == "__main__":
    main()
