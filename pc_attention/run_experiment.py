"""The M1 experiment. For each sequence length N in the sweep it does three things:

  1. measures delta(N) = E[1 - cos(g_PC, g_BP)] for W_Q, W_K, W_V, averaged over a
     few random starts. Two versions: "masked" scores only the query (the real
     task), "dense" scores every token against a random target (a harder stress
     test, since now all N tokens feed into the gradient sums).
  2. checks that the locality budget is 2.
  3. trains the layer twice from the same start, once with backprop and once with
     PC, and reports final recall accuracy, to see whether PC keeps up.

It writes results/delta_vs_N.{png,csv} and results/accuracy_vs_N.png.

Everything runs in float64 on purpose: delta lands around 1e-8, and float32 rounding
noise would sit right on top of the number we're trying to read. Pass --quick for a
fast sanity sweep.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pc_attention.linear_attn import linear_attention_forward
from pc_attention.backprop_ref import backprop_gradients_per_example, pred_loss
from pc_attention.pc_updates import pc_gradients, aggregate
from pc_attention.metrics import gradient_alignment, verify_locality
from pc_attention.tasks import RecallTask

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
WEIGHT_KEYS = ("W_Q", "W_K", "W_V")


# ---- helpers ----------------------------------------------------------------
def init_weights(d, m, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    # 1/sqrt(d) scaling keeps q, k, v in a reasonable range at init.
    WQ = torch.randn(d, m, generator=g, dtype=dtype) / d**0.5
    WK = torch.randn(d, m, generator=g, dtype=dtype) / d**0.5
    WV = torch.randn(d, d, generator=g, dtype=dtype) / d**0.5
    return WQ, WK, WV


def dense_target(batch_X, seed, dtype):
    # a random regression target, one per token, for the "dense" delta condition
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*batch_X.shape, generator=g, dtype=dtype)


# ---- delta(N) ---------------------------------------------------------------
def measure_delta(task, N, cfg):
    """Average delta over a few random (weights, data) draws for one N."""
    d, m = cfg.d, cfg.m
    acc = {cond: {k: 0.0 for k in WEIGHT_KEYS} for cond in ("masked", "dense")}
    acc_cf = {cond: {k: 0.0 for k in WEIGHT_KEYS} for cond in ("masked", "dense")}
    relax_steps_tot, relax_res_max = 0, 0.0

    for r in range(cfg.delta_reps):
        WQ, WK, WV = init_weights(d, m, seed=1000 + r, dtype=cfg.dtype)
        batch = task.generate(cfg.batch, N, seed=2000 + r)
        fwd = linear_attention_forward(batch.X, WQ, WK, WV, cfg.feature_map, cfg.eps_guard)

        for cond in ("masked", "dense"):
            if cond == "masked":            # on-task: only the query is supervised
                x_out, mask = batch.x_out, batch.mask
            else:                           # stress test: every token supervised
                x_out, mask = dense_target(batch.X, 3000 + r, cfg.dtype), None

            g_bp = backprop_gradients_per_example(
                batch.X, WQ, WK, WV, x_out, cfg.feature_map, cfg.eps_guard, mask=mask
            )
            # PC gradients from the actual inference loop, plus the closed-form value
            # recorded alongside as the best-case floor.
            g_rx, info = pc_gradients(
                fwd, x_out, beta=cfg.beta, method="relax",
                n_steps=cfg.relax_steps, mask=mask, return_info=True,
            )
            g_cf = pc_gradients(fwd, x_out, method="closed_form", mask=mask)

            d_rx = gradient_alignment(g_rx, g_bp)
            d_cf = gradient_alignment(g_cf, g_bp)
            for k in WEIGHT_KEYS:
                acc[cond][k] += d_rx[k] / cfg.delta_reps
                acc_cf[cond][k] += d_cf[k] / cfg.delta_reps
            if cond == "masked":
                relax_steps_tot += info.steps
                relax_res_max = max(relax_res_max, info.res_S, info.res_u)

    return acc, acc_cf, relax_steps_tot / cfg.delta_reps, relax_res_max


# ---- training (PC vs backprop, matched setup) -------------------------------
def evaluate(task, N, WQ, WK, WV, cfg):
    batch = task.generate(cfg.eval_batch, N, seed=9999)
    with torch.no_grad():
        fwd = linear_attention_forward(batch.X, WQ, WK, WV, cfg.feature_map, cfg.eps_guard)
        acc = task.accuracy(fwd.z, batch)
        loss = pred_loss(fwd.z, batch.x_out, mask=batch.mask).item() / cfg.eval_batch
    return acc, loss


def train(task, N, cfg, mode):
    """Train the layer. mode is 'bp' or 'pc'; returns (final_acc, final_loss).

    Both modes share the same init, the same data order, and the same Adam
    optimiser. The only difference is where the gradient comes from, so any gap in
    the final accuracy is really PC vs backprop and nothing else.
    """
    WQ, WK, WV = init_weights(cfg.d, cfg.m, seed=42, dtype=cfg.dtype)
    WQ, WK, WV = (w.clone().requires_grad_(True) for w in (WQ, WK, WV))
    opt = torch.optim.Adam([WQ, WK, WV], lr=cfg.lr)

    for t in range(cfg.train_steps):
        batch = task.generate(cfg.train_batch, N, seed=50_000 + t)  # same stream for bp & pc
        if mode == "bp":
            fwd = linear_attention_forward(batch.X, WQ, WK, WV, cfg.feature_map, cfg.eps_guard)
            loss = pred_loss(fwd.z, batch.x_out, mask=batch.mask)
            opt.zero_grad(); loss.backward(); opt.step()
        else:  # pc: compute the local gradients by hand and hand them to Adam
            with torch.no_grad():
                fwd = linear_attention_forward(
                    batch.X, WQ.detach(), WK.detach(), WV.detach(),
                    cfg.feature_map, cfg.eps_guard,
                )
                g = aggregate(pc_gradients(
                    fwd, batch.x_out, beta=cfg.beta, method=cfg.train_method,
                    n_steps=cfg.relax_steps, mask=batch.mask,
                ))
            WQ.grad = g["W_Q"].clone(); WK.grad = g["W_K"].clone(); WV.grad = g["W_V"].clone()
            opt.step()

    return evaluate(task, N, WQ, WK, WV, cfg)


# ---- plotting ---------------------------------------------------------------
def make_plots(rows, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = [r["N"] for r in rows]
    floor = 1e-17   # a log axis can't show an exact zero, so clamp to a tiny floor

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, cond in zip(axes, ("masked", "dense")):
        for k in WEIGHT_KEYS:
            y = [max(r[f"delta_{cond}_{k}"], floor) for r in rows]
            ax.plot(Ns, y, marker="o", label=k)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length N")
        ax.set_title(f"delta(N)  [{cond} targets]  beta={cfg.beta:g}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("delta = E[1 - cos(g_PC, g_BP)]")
    fig.suptitle("Path B: PC-vs-backprop gradient alignment (flat => claim holds)")
    fig.tight_layout()
    p1 = os.path.join(RESULTS_DIR, "delta_vs_N.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ns, [r["acc_bp"] for r in rows], marker="o", label="backprop")
    ax.plot(Ns, [r["acc_pc"] for r in rows], marker="s", label="PC (local)")
    ax.axhline(1.0 / cfg.n_values, ls="--", color="gray", label="chance (1/vocab)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length N"); ax.set_ylabel("recall accuracy")
    ax.set_title(f"Recall accuracy after {cfg.train_steps} steps (d={cfg.d}, m={cfg.m})")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    p2 = os.path.join(RESULTS_DIR, "accuracy_vs_N.png")
    fig.savefig(p2, dpi=130); plt.close(fig)
    return p1, p2


# ---- main -------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke sweep")
    ap.add_argument("--Ns", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1e4)
    ap.add_argument("--feature-map", default="elu+1")
    ap.add_argument("--delta-reps", type=int, default=5)
    ap.add_argument("--relax-steps", type=int, default=30)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--train-batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--train-method", default="relax", choices=["relax", "closed_form"])
    ap.add_argument("--n-values", type=int, default=16)
    ap.add_argument("--n-keys", type=int, default=512)
    ap.add_argument("--eps-guard", type=float, default=1e-6)
    cfg = ap.parse_args()
    cfg.dtype = torch.float64
    if cfg.quick:
        cfg.Ns = [4, 16, 64]
        cfg.delta_reps = 2
        cfg.train_steps = 60
    return cfg


def main():
    cfg = parse_args()
    torch.set_default_dtype(cfg.dtype)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    task = RecallTask(d=cfg.d, n_values=cfg.n_values, n_keys=cfg.n_keys,
                      seed=0, dtype=cfg.dtype)

    print(f"config: Ns={cfg.Ns} d={cfg.d} m={cfg.m} beta={cfg.beta:g} "
          f"phi={cfg.feature_map} delta_reps={cfg.delta_reps} "
          f"train_steps={cfg.train_steps} method={cfg.train_method}")
    print("-" * 92)
    header = (f"{'N':>4} | {'delta_masked (WQ/WK/WV)':>34} | {'delta_dense (WQ/WK/WV)':>34} | "
              f"{'relax':>10} | {'acc bp/pc':>13} | loc")
    print(header)
    print("-" * len(header))

    rows = []
    for N in cfg.Ns:
        delta, delta_cf, relax_steps, relax_res = measure_delta(task, N, cfg)

        # locality budget: one check per N is plenty
        WQ, WK, WV = init_weights(cfg.d, cfg.m, seed=7, dtype=cfg.dtype)
        b = task.generate(cfg.batch, N, seed=7)
        fwd = linear_attention_forward(b.X, WQ, WK, WV, cfg.feature_map, cfg.eps_guard)
        rep = verify_locality(fwd, WQ, WK, WV, b.x_out, beta=cfg.beta,
                              mask=b.mask, method="relax", feature_map=cfg.feature_map)

        acc_bp, loss_bp = train(task, N, cfg, "bp")
        acc_pc, loss_pc = train(task, N, cfg, "pc")

        row = {"N": N, "relax_steps": relax_steps, "relax_res": relax_res,
               "acc_bp": acc_bp, "acc_pc": acc_pc, "loss_bp": loss_bp, "loss_pc": loss_pc,
               "locality_budget": rep.n_shared_reads, "locality_ok": int(rep.per_token_local)}
        for cond in ("masked", "dense"):
            for k in WEIGHT_KEYS:
                row[f"delta_{cond}_{k}"] = delta[cond][k]
                row[f"delta_cf_{cond}_{k}"] = delta_cf[cond][k]
        rows.append(row)

        dm = "/".join(f"{delta['masked'][k]:.1e}" for k in WEIGHT_KEYS)
        dd = "/".join(f"{delta['dense'][k]:.1e}" for k in WEIGHT_KEYS)
        print(f"{N:>4} | {dm:>34} | {dd:>34} | {relax_steps:>4.0f}/{relax_res:.0e} | "
              f"{acc_bp:>5.3f}/{acc_pc:<5.3f} | {rep.n_shared_reads}{'ok' if rep.per_token_local else 'XX'}")

    # raw numbers to CSV
    csv_path = os.path.join(RESULTS_DIR, "delta_vs_N.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    p1, p2 = make_plots(rows, cfg)
    print("-" * len(header))
    print(f"wrote {csv_path}")
    print(f"wrote {p1}")
    print(f"wrote {p2}")

    # one-line takeaway: the largest delta anywhere in the sweep
    worst_masked = max(r[f"delta_masked_{k}"] for r in rows for k in WEIGHT_KEYS)
    worst_dense = max(r[f"delta_dense_{k}"] for r in rows for k in WEIGHT_KEYS)
    print(f"\nheadline: worst delta over all N/tensors  masked={worst_masked:.2e}  dense={worst_dense:.2e}")
    print("Path B prediction is delta(N) ~ flat; see README for interpretation.")


if __name__ == "__main__":
    main()
