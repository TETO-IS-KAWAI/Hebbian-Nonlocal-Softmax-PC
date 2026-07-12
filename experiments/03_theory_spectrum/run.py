"""Extension 3 -- the continuous locality spectrum.

THEORY.md states the discrete rows: exact softmax PC keeps the global
redistribution term (delta = 0), the naive rule drops it (delta ~ 0.1-0.6). This
sweeps the fraction alpha of that term that is kept, from 0 (naive) to 1 (exact),
and measures delta at two attention-sharpness levels. It shows delta rising
smoothly from machine-zero as you remove global information -- the price of
locality, made continuous.

Outputs: results/ext3_spectrum.png, results/ext3_spectrum.csv
"""

from __future__ import annotations

import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")

from pc_attention.softmax_attn import softmax_attention_forward, softmax_backprop_per_example
from pc_attention.pc_updates_A import pc_gradients_A
from pc_attention.metrics import gradient_alignment

KEYS = ("W_Q", "W_K", "W_V")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
N, B, d, REPS = 32, 16, 16, 4


def mean_over(dct):
    return sum(dct[k] for k in KEYS) / len(KEYS)


def delta_at(alpha, wscale):
    acc = 0.0
    for r in range(REPS):
        g = torch.Generator().manual_seed(4000 + r)
        X = torch.randn(B, N, d, generator=g)
        x_out = torch.randn(B, N, d, generator=g)
        gg = torch.Generator().manual_seed(5000 + r)
        W = [wscale * torch.randn(d, d, generator=gg) / d**0.5 for _ in range(3)]
        fwd = softmax_attention_forward(X, *W)
        g_bp = softmax_backprop_per_example(X, *W, x_out)
        g_pc = pc_gradients_A(fwd, x_out, method="closed_form", alpha=alpha)
        acc += mean_over(gradient_alignment(g_pc, g_bp)) / REPS
    return acc


def main():
    torch.set_default_dtype(torch.float64)
    os.makedirs(RESULTS, exist_ok=True)
    print("alpha-scan: delta vs fraction of redistribution kept (0=naive, 1=exact)\n")
    print(f"{'alpha':>7}" + "".join(f"{a:>9}" for a in ALPHAS))
    curves = {}
    for wscale, label in [(1.0, "diffuse (x1)"), (12.0, "sharp (x12)")]:
        row = [delta_at(a, wscale) for a in ALPHAS]
        curves[label] = row
        print(f"{label:>16}" + "".join(f"{v:>9.1e}" if v < 1e-3 else f"{v:>9.3f}" for v in row))

    with open(os.path.join(RESULTS, "ext3_spectrum.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["alpha"] + list(curves.keys()))
        for i, a in enumerate(ALPHAS):
            w.writerow([a] + [curves[k][i] for k in curves])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, row in curves.items():
        ax.plot(ALPHAS, [max(v, 1e-18) for v in row], marker="o", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("fraction of global redistribution kept  (0 = naive local, 1 = exact)")
    ax.set_ylabel("delta = E[1 - cos(g_PC, g_BP)]")
    ax.set_title("Extension 3: the locality spectrum\n"
                 "delta rises smoothly from machine-zero as global info is removed")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(RESULTS, "ext3_spectrum.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
