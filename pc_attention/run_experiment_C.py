"""Path C: quantify the cost of forcing locality.

Path C doesn't build a new trainer; it measures how the PC-vs-backprop gap
delta(N, L) scales. A natural guess for how it worsens with depth and length is
    delta(N, L) ~ 1 - (1 - a)^L (1 - b log N)
with a depth term (a) and a sequence-length term (b log N).

We pull the three axes together:

  * length axis b  -- delta(N) for the exact methods (Path B, Path A) and for a
    deliberately-too-local rule (Path A with the global redistribution term
    dropped). The exact methods should give b = 0; the naive one is where any
    length cost shows up.
  * depth axis a   -- from M2 (run_experiment_m2.py): at converged inference the
    exact stack has a = 0; the depth cost there is inference latency, not an
    irreducible error.
  * normalizer precision gamma -- from Path A (run_experiment_A.py): delta ~ 1/gamma^2.

The interesting finding is that the naive rule's length dependence is not a fixed
b log N: it flips sign with attention sharpness (diffuse attention -> cost shrinks
with N; peaked attention -> cost grows with N), refining the guess above.

Outputs: results/pathC_delta_vs_N.png, results/pathC_summary.csv
"""

from __future__ import annotations

import csv
import math
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
Ns = [4, 8, 16, 32, 64, 128, 256]
REPS = 4
B, d, d_k = 16, 16, 16


def mean_over(dct):
    return sum(dct[k] for k in KEYS) / len(KEYS)


def sample(N, r, wscale):
    g = torch.Generator().manual_seed(4000 + r)
    X = torch.randn(B, N, d, generator=g)
    x_out = torch.randn(B, N, d_k, generator=g)
    gg = torch.Generator().manual_seed(5000 + r)
    W = [wscale * torch.randn(d, d_k, generator=gg) / d**0.5 for _ in range(3)]
    return X, x_out, W


def delta_curve(kind, wscale=1.0):
    """delta(N) for one method. kind in {B, A, naive}."""
    out = []
    for N in Ns:
        acc = 0.0
        for r in range(REPS):
            X, x_out, W = sample(N, r, wscale)
            if kind == "B":
                fwd = linear_attention_forward(X, *W)
                g_bp = backprop_gradients_per_example(X, *W, x_out)
                g_pc = pc_gradients(fwd, x_out, method="closed_form")
            else:
                fwd = softmax_attention_forward(X, *W)
                g_bp = softmax_backprop_per_example(X, *W, x_out)
                g_pc = pc_gradients_A(fwd, x_out, method="closed_form", naive=(kind == "naive"))
            acc += mean_over(gradient_alignment(g_pc, g_bp)) / REPS
        out.append(acc)
    return out


def fit_b(delta):
    """Least-squares slope of delta vs log2(N): the length coefficient b."""
    xs = [math.log2(N) for N in Ns]
    ys = delta
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


def main():
    torch.set_default_dtype(torch.float64)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Path C: delta(N) for exact vs naive-local rules, Ns={Ns}, reps={REPS}\n")

    curves = {
        "Path B exact (linear)": delta_curve("B"),
        "Path A exact (softmax)": delta_curve("A"),
        "naive-local, diffuse attn (x1)": delta_curve("naive", wscale=1.0),
        "naive-local, sharp attn (x12)": delta_curve("naive", wscale=12.0),
    }

    print(f"{'method':<34}" + "".join(f"{N:>9}" for N in Ns) + f"{'  b(slope/log2N)':>16}")
    summary = []
    for name, cur in curves.items():
        b = fit_b(cur)
        summary.append((name, cur, b))
        print(f"{name:<34}" + "".join(f"{v:>9.1e}" if abs(v) < 1e-3 else f"{v:>9.3f}" for v in cur)
              + f"{b:>16.4f}")

    print("\naxes of the conjecture delta(N,L) ~ 1 - (1-a)^L (1 - b log N):")
    print("  length b : exact paths ~ 0 (global info preserved); naive rule nonzero,")
    print("             and its sign flips with attention sharpness (see rows above).")
    print("  depth  a : from M2 -> 0 at converged inference; depth cost is inference")
    print("             latency (error propagates ~1 layer/step), not irreducible error.")
    print("  gamma    : from Path A -> delta ~ 1/gamma^2 (normalizer precision).")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "pathC_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"N={N}" for N in Ns] + ["b_slope_per_log2N"])
        for name, cur, b in summary:
            w.writerow([name] + [f"{v:.6e}" for v in cur] + [f"{b:.6f}"])

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"Path B exact (linear)": ("^", "-"), "Path A exact (softmax)": ("o", "-"),
              "naive-local, diffuse attn (x1)": ("s", "--"), "naive-local, sharp attn (x12)": ("D", "--")}
    for name, cur, b in summary:
        mk, ls = styles[name]
        ax.plot(Ns, [max(abs(v), 1e-18) for v in cur], marker=mk, ls=ls, label=f"{name}  (b={b:+.3f})")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("sequence length N"); ax.set_ylabel("delta = E[1 - cos(g_PC, g_BP)]")
    ax.set_title("Path C: forcing locality is free when global info is kept (exact),\n"
                 "costly when it is dropped (naive) -- and the length trend depends on sharpness")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, "pathC_delta_vs_N.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nwrote {csv_path}\nwrote {p}")


if __name__ == "__main__":
    main()
