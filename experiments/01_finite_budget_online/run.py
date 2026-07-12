"""Extension 1 -- does PC still train an attention stack under a bounded inference
budget? Every earlier result relaxed inference to convergence, which is expensive
and biologically implausible. Here we cap inference at T steps and watch training.

We train a 3-layer causal char-LM with backprop and with PC at T in
{1, 2, 4, 8, 32}, from the same init on the same data, and compare loss curves.
T -> large should match backprop; small T is the realistic regime where the error
never reaches the lower layers.

Outputs: results/ext1_finite_budget.png, results/ext1_finite_budget.csv
"""

from __future__ import annotations

import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")

from pc_attention.deep_char_lm import init_params, forward_lm, backprop_grads, pc_grads, flat_dev
from pc_attention.char_lm import sinusoidal_pe

TEXT = "the quick brown fox jumps over the lazy dog. " * 24
CFG = {"d": 64, "m": 64, "N": 32, "B": 16, "L": 3, "lr": 2e-3, "steps": 300, "eval_every": 15}
T_LIST = [1, 2, 4, 8, 32]


def encode(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return ids, stoi, {i: c for c, i in stoi.items()}


def batch(data, B, N, gen):
    idx = torch.randint(0, len(data) - N - 1, (B,), generator=gen)
    x = torch.stack([data[i:i + N] for i in idx])
    y = torch.stack([data[i + 1:i + N + 1] for i in idx])
    return x, y


def train(mode, params0, data, pe, T=None):
    params = {"Emb": params0["Emb"].clone(),
              "W_out": params0["W_out"].clone(),
              "attn": [tuple(w.clone() for w in layer) for layer in params0["attn"]]}
    flat = [params["Emb"], params["W_out"]] + [w for layer in params["attn"] for w in layer]
    for t in flat:
        t.requires_grad_(True)
    opt = torch.optim.Adam(flat, lr=CFG["lr"])
    gen = torch.Generator().manual_seed(1234)

    steps, losses = [], []
    for t in range(CFG["steps"]):
        ids, tg = batch(data, CFG["B"], CFG["N"], gen)
        plain = {"Emb": params["Emb"].detach(), "W_out": params["W_out"].detach(),
                 "attn": [tuple(w.detach() for w in layer) for layer in params["attn"]]}
        if mode == "bp":
            grads, loss, _ = backprop_grads(plain, ids, tg, pe)
        else:
            grads, loss, _ = pc_grads(plain, ids, tg, pe, T=T, lr=0.3)
        params["Emb"].grad = grads["Emb"].clone()
        params["W_out"].grad = grads["W_out"].clone()
        for layer, glayer in zip(params["attn"], grads["attn"]):
            for w, gw in zip(layer, glayer):
                w.grad = gw.clone()
        opt.step(); opt.zero_grad(set_to_none=False)
        if t % CFG["eval_every"] == 0 or t == CFG["steps"] - 1:
            steps.append(t); losses.append(loss)
    return steps, losses


def main():
    torch.set_default_dtype(torch.float64)
    os.makedirs(RESULTS, exist_ok=True)
    data, stoi, itos = encode(TEXT)
    vocab = len(stoi)
    pe = sinusoidal_pe(CFG["N"], CFG["d"])
    params0 = init_params(vocab, CFG["d"], CFG["m"], CFG["L"], seed=0)
    print(f"3-layer causal char-LM, vocab {vocab}, {CFG}")

    # gradient-vs-budget gate (one batch)
    g = torch.Generator().manual_seed(7)
    ids, tg = batch(data, CFG["B"], CFG["N"], g)
    g_bp, _, _ = backprop_grads(params0, ids, tg, pe)
    print("\nworst |g_PC - g_BP| vs inference budget T (one batch):")
    for T in T_LIST + [200]:
        g_pc, _, _ = pc_grads(params0, ids, tg, pe, T=T, lr=0.3)
        print(f"   T={T:>3}: {flat_dev(g_pc, g_bp):.2e}")

    # training runs
    print("\ntraining ...")
    curves = {}
    s_bp, l_bp = train("bp", params0, data, pe)
    curves["backprop"] = (s_bp, l_bp)
    print(f"   backprop      final loss {l_bp[-1]:.4f}")
    for T in T_LIST:
        s, l = train("pc", params0, data, pe, T=T)
        curves[f"PC T={T}"] = (s, l)
        print(f"   PC T={T:<3}      final loss {l[-1]:.4f}")

    # CSV
    csv_path = os.path.join(RESULTS, "ext1_finite_budget.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + list(curves.keys()))
        for i, st in enumerate(curves["backprop"][0]):
            w.writerow([st] + [f"{curves[k][1][i]:.5f}" for k in curves])

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(*curves["backprop"], lw=3.5, alpha=0.4, color="black", label="backprop")
    for k in curves:
        if k == "backprop":
            continue
        s, l = curves[k]
        ax.plot(s, l, marker=".", label=k)
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Extension 1: PC training under a bounded inference budget T\n"
                 "(deeper error needs more T to reach the lower layers)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(RESULTS, "ext1_finite_budget.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nwrote {csv_path}\nwrote {p}")


if __name__ == "__main__":
    main()
