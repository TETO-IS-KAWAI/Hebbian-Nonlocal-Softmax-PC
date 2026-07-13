"""Extension 4 -- is PC ever *better* than backprop? The weight-transport test.

Backprop's backward pass multiplies errors by the exact transpose of the forward
weights. That "weight transport" is the classic biological objection to backprop:
a neuron would need to know the downstream synapse weights exactly. Our PC so far
also used the exact transpose (via one-layer autograd) for feedback.

Here we cut it: feedback uses fixed *random* matrices instead of the transpose
(feedback alignment). Backprop cannot do this -- its gradient is defined by the
transpose. If PC still trains the char-LM with random feedback, then PC-attention
removes weight transport, something backprop structurally cannot.

We compare three runs on the same deep causal char-LM: backprop, PC with exact
feedback, and PC with random feedback (no weight transport).

Outputs: results/ext4_feedback_alignment.png, results/ext4_feedback_alignment.csv
"""

from __future__ import annotations

import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")

from pc_attention.deep_char_lm import init_params, backprop_grads, pc_grads
from pc_attention.char_lm import sinusoidal_pe

TEXT = "the quick brown fox jumps over the lazy dog. " * 24
CFG = {"d": 64, "m": 64, "N": 32, "B": 16, "L": 2, "lr": 2e-3, "steps": 400, "eval_every": 20, "T": 12}


def encode(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return torch.tensor([stoi[c] for c in text], dtype=torch.long), stoi


def batch(data, B, N, gen):
    idx = torch.randint(0, len(data) - N - 1, (B,), generator=gen)
    return (torch.stack([data[i:i + N] for i in idx]),
            torch.stack([data[i + 1:i + N + 1] for i in idx]))


def train(mode, params0, data, pe, feedback_mats=None):
    params = {"Emb": params0["Emb"].clone(), "W_out": params0["W_out"].clone(),
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
            grads, loss, _ = pc_grads(plain, ids, tg, pe, T=CFG["T"], lr=0.3, feedback_mats=feedback_mats)
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
    data, stoi = encode(TEXT)
    vocab = len(stoi)
    pe = sinusoidal_pe(CFG["N"], CFG["d"])
    params0 = init_params(vocab, CFG["d"], CFG["m"], CFG["L"], seed=0)

    # fixed random feedback matrices (persist across training), one per layer
    gf = torch.Generator().manual_seed(99)
    B_rand = [torch.randn(CFG["d"], CFG["d"], generator=gf) / CFG["d"]**0.5 for _ in range(CFG["L"])]

    print(f"{CFG['L']}-layer causal char-LM, vocab {vocab}, {CFG}")
    print("training three ways ...")
    curves = {}
    curves["backprop"] = train("bp", params0, data, pe)
    print(f"   backprop               final loss {curves['backprop'][1][-1]:.4f}")
    curves["PC exact feedback"] = train("pc", params0, data, pe, feedback_mats=None)
    print(f"   PC (exact transpose)   final loss {curves['PC exact feedback'][1][-1]:.4f}")
    curves["PC random feedback"] = train("pc", params0, data, pe, feedback_mats=B_rand)
    print(f"   PC (random feedback)   final loss {curves['PC random feedback'][1][-1]:.4f}")

    with open(os.path.join(RESULTS, "ext4_feedback_alignment.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step"] + list(curves.keys()))
        for i, st in enumerate(curves["backprop"][0]):
            w.writerow([st] + [f"{curves[k][1][i]:.5f}" for k in curves])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(*curves["backprop"], lw=3.5, alpha=0.4, color="black", label="backprop (needs weight transport)")
    ax.plot(*curves["PC exact feedback"], lw=1.4, ls="--", color="steelblue", label="PC, exact feedback")
    ax.plot(*curves["PC random feedback"], lw=1.8, color="crimson", label="PC, random feedback (no transport)")
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Extension 4: PC trains the attention LM with random feedback\n"
                 "(no weight transport) -- something backprop cannot do")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(RESULTS, "ext4_feedback_alignment.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
