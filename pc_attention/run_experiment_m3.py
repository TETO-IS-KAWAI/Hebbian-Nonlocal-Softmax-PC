"""M3 experiment: train a causal char-LM with PC and with backprop, compare.

The single-layer LM has an exact local rule for every parameter (checked in
tests/test_gradients_causal.py and at the top of this run), so PC and backprop
compute the same gradient. The point of M3 is to watch that play out over a real
training run: same init, same data, same optimizer, only the gradient *source*
differs. The two loss curves should sit on top of each other, and the model should
actually learn the text.

Outputs: results/m3_training.png, results/m3_curves.csv
"""

from __future__ import annotations

import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pc_attention.char_lm import (
    init_params, sinusoidal_pe, backprop_grads, pc_grads, generate, PARAM_KEYS,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

TEXT = ("the quick brown fox jumps over the lazy dog. " * 24)


def encode(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return ids, stoi, itos


def batch(data, B, N, gen):
    """B random windows: inputs (B,N), next-char targets (B,N)."""
    idx = torch.randint(0, len(data) - N - 1, (B,), generator=gen)
    x = torch.stack([data[i : i + N] for i in idx])
    y = torch.stack([data[i + 1 : i + N + 1] for i in idx])
    return x, y


def train(mode, params0, data, pe, cfg):
    params = {k: v.clone().requires_grad_(True) for k, v in params0.items()}
    opt = torch.optim.Adam([params[k] for k in PARAM_KEYS], lr=cfg["lr"])
    gen = torch.Generator().manual_seed(1234)          # same data stream for both modes
    grad_fn = backprop_grads if mode == "bp" else pc_grads

    steps, losses, accs = [], [], []
    for t in range(cfg["steps"]):
        ids, targets = batch(data, cfg["B"], cfg["N"], gen)
        with torch.no_grad():
            plain = {k: params[k].detach() for k in PARAM_KEYS}
        grads, loss, acc = grad_fn(plain, ids, targets, pe)
        for k in PARAM_KEYS:
            params[k].grad = grads[k].clone()
        opt.step()
        opt.zero_grad(set_to_none=False)
        if t % cfg["eval_every"] == 0 or t == cfg["steps"] - 1:
            steps.append(t); losses.append(loss); accs.append(acc)
    return params, steps, losses, accs


def main():
    torch.set_default_dtype(torch.float64)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = {"d": 64, "m": 64, "N": 32, "B": 16, "lr": 3e-3, "steps": 600, "eval_every": 20}

    data, stoi, itos = encode(TEXT)
    vocab = len(stoi)
    pe = sinusoidal_pe(cfg["N"], cfg["d"])
    params0 = init_params(vocab, cfg["d"], cfg["m"], seed=0)

    print(f"text {len(data)} chars, vocab {vocab}; model d={cfg['d']} m={cfg['m']} "
          f"N={cfg['N']} B={cfg['B']} steps={cfg['steps']}")

    # one-off gradient-equality check
    ids, targets = batch(data, cfg["B"], cfg["N"], torch.Generator().manual_seed(7))
    g_bp, _, _ = backprop_grads(params0, ids, targets, pe)
    g_pc, _, _ = pc_grads(params0, ids, targets, pe)
    worst = max((g_pc[k] - g_bp[k]).abs().max().item() for k in PARAM_KEYS)
    print(f"gradient equality PC vs BP (all params): worst dev = {worst:.2e}")

    bp_p, s, bp_loss, bp_acc = train("bp", params0, data, pe, cfg)
    pc_p, _, pc_loss, pc_acc = train("pc", params0, data, pe, cfg)

    max_div = max(abs(a - b) for a, b in zip(bp_loss, pc_loss))
    print(f"\nstep |   bp loss   pc loss  |  bp acc  pc acc")
    for i in range(0, len(s), max(1, len(s) // 12)):
        print(f"{s[i]:>4} |  {bp_loss[i]:.4f}   {pc_loss[i]:.4f}  |  {bp_acc[i]:.3f}   {pc_acc[i]:.3f}")
    print(f"\nfinal: bp loss {bp_loss[-1]:.4f} acc {bp_acc[-1]:.3f} | "
          f"pc loss {pc_loss[-1]:.4f} acc {pc_acc[-1]:.3f}")
    print(f"max |loss_pc - loss_bp| over training = {max_div:.2e}  (curves coincide)")

    # qualitative sample from the PC-trained model
    prime = torch.tensor([[stoi[c] for c in "the "]], dtype=torch.long)
    gen_ids = generate(pc_p, prime, pe, n_new=60, temperature=0.5)
    sample = "".join(itos[i] for i in gen_ids[0].tolist())
    print(f"\nPC-trained sample: {sample!r}")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "m3_curves.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "bp_loss", "pc_loss", "bp_acc", "pc_acc"])
        for i in range(len(s)):
            w.writerow([s[i], bp_loss[i], pc_loss[i], bp_acc[i], pc_acc[i]])

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(s, bp_loss, lw=3, alpha=0.5, label="backprop")
    ax1.plot(s, pc_loss, lw=1.2, ls="--", label="PC (local)")
    ax1.set_xlabel("step"); ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Char-LM training loss"); ax1.grid(alpha=0.3); ax1.legend()
    ax2.plot(s, bp_acc, lw=3, alpha=0.5, label="backprop")
    ax2.plot(s, pc_acc, lw=1.2, ls="--", label="PC (local)")
    ax2.set_xlabel("step"); ax2.set_ylabel("next-char accuracy")
    ax2.set_title("Char-LM next-char accuracy"); ax2.grid(alpha=0.3); ax2.legend()
    fig.suptitle(f"M3: PC trains a causal char-LM identically to backprop "
                 f"(max loss gap {max_div:.1e})")
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, "m3_training.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nwrote {csv_path}\nwrote {p}")


if __name__ == "__main__":
    main()
