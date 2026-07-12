"""Extension 2 -- does PC scale to a real transformer block?

Trains a pre-norm transformer block (multi-head causal linear attention + FFN +
LayerNorm + residual) as a char-LM, with backprop and with fixed-prediction PC.
The PC activity/error nodes live on the residual stream between sub-layers, so
LayerNorm, the MLP nonlinearity, and the multiple heads all ride inside a
sub-layer and need no special PC rule.

Outputs: results/ext2_transformer.png, results/ext2_transformer.csv
"""

from __future__ import annotations

import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")

from pc_attention.transformer_block import (
    init_params, backprop_grads, pc_grads, worst_dev,
)
from pc_attention.char_lm import sinusoidal_pe

TEXT = "the quick brown fox jumps over the lazy dog. " * 24
CFG = {"d": 64, "H": 4, "d_ff": 128, "L": 1, "N": 32, "B": 16,
       "lr": 2e-3, "steps": 200, "eval_every": 10, "T": 32}


def encode(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return ids, stoi, {i: c for c, i in stoi.items()}


def batch(data, B, N, gen):
    idx = torch.randint(0, len(data) - N - 1, (B,), generator=gen)
    return (torch.stack([data[i:i + N] for i in idx]),
            torch.stack([data[i + 1:i + N + 1] for i in idx]))


def train(mode, params0, data, pe, cfg):
    params = {k: v.clone().requires_grad_(True) for k, v in params0.items()}
    opt = torch.optim.Adam(list(params.values()), lr=cfg["lr"])
    gen = torch.Generator().manual_seed(1234)
    steps, losses = [], []
    for t in range(cfg["steps"]):
        ids, tg = batch(data, cfg["B"], cfg["N"], gen)
        plain = {k: v.detach() for k, v in params.items()}
        if mode == "bp":
            grads, loss, _ = backprop_grads(plain, ids, tg, pe, cfg["L"], cfg["H"])
        else:
            grads, loss, _ = pc_grads(plain, ids, tg, pe, cfg["L"], cfg["H"], T=cfg["T"], lr=0.3)
        for k in params:
            params[k].grad = grads[k].clone()
        opt.step(); opt.zero_grad(set_to_none=False)
        if t % cfg["eval_every"] == 0 or t == cfg["steps"] - 1:
            steps.append(t); losses.append(loss)
    return steps, losses


def main():
    torch.set_default_dtype(torch.float64)
    os.makedirs(RESULTS, exist_ok=True)
    data, stoi, itos = encode(TEXT)
    vocab = len(stoi)
    pe = sinusoidal_pe(CFG["N"], CFG["d"])
    params0 = init_params(vocab, CFG["d"], CFG["H"], CFG["d_ff"], CFG["L"], seed=0)
    print(f"pre-norm transformer block: {CFG}, vocab {vocab}, "
          f"params {sum(v.numel() for v in params0.values())}")

    # gate: full-relaxation PC gradient == backprop, across ALL params (incl LN, FFN)
    g = torch.Generator().manual_seed(7)
    ids, tg = batch(data, CFG["B"], CFG["N"], g)
    g_bp, _, _ = backprop_grads(params0, ids, tg, pe, CFG["L"], CFG["H"])
    print("\nworst |g_PC - g_BP| over ALL params vs inference budget T:")
    for T in (1, 8, 32, 200):
        g_pc, _, _ = pc_grads(params0, ids, tg, pe, CFG["L"], CFG["H"], T=T, lr=0.3)
        print(f"   T={T:>3}: {worst_dev(g_pc, g_bp):.2e}")

    print("\ntraining ...")
    s_bp, l_bp = train("bp", params0, data, pe, CFG)
    s_pc, l_pc = train("pc", params0, data, pe, CFG)
    max_gap = max(abs(a - b) for a, b in zip(l_bp, l_pc))
    print(f"   backprop final loss {l_bp[-1]:.4f}")
    print(f"   PC (T={CFG['T']}) final loss {l_pc[-1]:.4f}")
    print(f"   max |loss_pc - loss_bp| = {max_gap:.2e}")

    with open(os.path.join(RESULTS, "ext2_transformer.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "bp_loss", "pc_loss"])
        for i in range(len(s_bp)):
            w.writerow([s_bp[i], l_bp[i], l_pc[i]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_bp, l_bp, lw=3.5, alpha=0.4, color="black", label="backprop")
    ax.plot(s_pc, l_pc, lw=1.3, ls="--", color="crimson", label=f"PC (local, T={CFG['T']})")
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Extension 2: PC trains a full transformer block (MHA + FFN + LayerNorm)\n"
                 f"identically to backprop (max gap {max_gap:.1e})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(RESULTS, "ext2_transformer.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
