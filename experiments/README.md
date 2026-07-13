# Extensions — beyond the M1–M3 ladder

The milestone ladder (in the repo root) established that predictive coding's local
updates reproduce backprop for attention **whenever the layer's global information
is preserved and inference is run to convergence**. Every result there is
machine-precision exact, which raises the obvious question: what happens when those
two idealizations are removed? These four folders each drop one assumption.

| # | folder | drops which idealization | status |
|---|---|---|---|
| ① | `01_finite_budget_online/` | converged inference | done. PC trains fine with a budget ~ depth; unbounded relaxation isn't needed |
| ② | `02_transformer_block/`    | attention-only, single head | done. PC = backprop on a full block (MHA + FFN + LayerNorm + residual) |
| ③ | `03_theory_spectrum/`      | consolidation + continuous spectrum | done. See `THEORY.md`; `delta` hits zero only when all global info is kept |
| ④ | `04_pc_advantages/`        | backprop's weight transport | done. PC-attention trains with random feedback and no weight transport |

Each folder has its own `README.md` and `run.py`, and reuses the core library in
`../../pc_attention/`. Run from the repo root with the project venv, e.g.
`./.venv/Scripts/python.exe experiments/01_finite_budget_online/run.py`.
