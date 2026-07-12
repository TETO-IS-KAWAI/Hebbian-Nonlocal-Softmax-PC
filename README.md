# Predictive Coding on Linear Self-Attention — Path B / Milestone M1

Can **predictive coding** (PC) — biologically-plausible *local* learning — train a
self-attention layer? Softmax attention breaks PC's locality via its global
normalizer (see `SPEC_full.md` PART III). **Path B** sidesteps this: replace the
softmax kernel with a decomposable feature map `phi(q)^T phi(k)`, which compresses
the global sum into two shared accumulator states `(S, u)`. This milestone (M1)
implements one linear-attention layer, derives its PC local update rules, and
tests — against autograd — the falsifiable claim that PC gradients stay aligned
with backprop gradients as sequence length `N` grows.

**Headline results (all reproduced by the commands below):**

| Check | Result |
|---|---|
| Analytic PC gradients vs autograd (correctness gate) | **match to ~1e-16** (closed form); ~1e-5 via relaxation at β=1e4 |
| `delta(N) = E[1 - cos(g_PC, g_BP)]` over N = 4…256 | **flat at the ~1e-8 floor** — Path B claim holds |
| Locality budget | **2 shared reads (S, u)**, empirically verified (zero cross-token leakage) |
| PC-trained vs backprop-trained recall accuracy | **identical at every N** (curves overlap exactly) |

---

## Install & run

Everything uses only `torch`, `numpy`, `matplotlib`.

```bash
python -m venv --system-site-packages .venv     # already created here
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # if starting fresh

# 1) Correctness gate (the key checkpoint) — prints max deviations + sensitivity
./.venv/Scripts/python.exe tests/test_gradients.py
#    (also has pytest-style test_* fns if you `pip install pytest`; core deps stay torch/numpy/matplotlib)

# 2) Full experiment: delta(N) sweep + PC-vs-backprop training
./.venv/Scripts/python.exe pc_attention/run_experiment.py            # ~40 s
./.venv/Scripts/python.exe pc_attention/run_experiment.py --quick    # ~8 s smoke
```

Outputs land in `results/`: `delta_vs_N.png`, `accuracy_vs_N.png`,
`delta_vs_N.csv`, `run_log.txt`.

### Module map (`SPEC_full.md` PART VII)

| file | role |
|---|---|
| `linear_attn.py` | Path B forward: `Q,K,V`, swappable `phi`, accumulators `S,u`, output `z_i` |
| `pc_updates.py`  | `F_ext` + analytic **local** update rules; inference relaxation loop |
| `backprop_ref.py`| same forward via autograd → reference gradients |
| `tasks.py`       | associative-recall / copy task generator |
| `metrics.py`     | `delta = E[1-cos]`; locality-budget verifier |
| `run_experiment.py` | `N` sweep → `delta(N)` curve, plots, CSV, PC/BP training |
| `derivations.md` | the gradients worked out by hand (read this) |

---

## 1. Correctness gate (kickoff step 5) — **PASSED**

`tests/test_gradients.py`, random single layer, `d=16, m=32, batch=4, N=8`,
float64, tolerance `atol=1e-3, rtol=1e-2`. Actual max deviations:

```
closed form (beta -> inf):   W_Q 4e-16  W_K 6e-16  W_V 2e-15   S 1e-17  u 6e-17   (machine precision)
relaxation  (beta = 1e4):    W_Q 1e-6   W_K 2e-6   W_V 3e-5    S 2e-7   u 1e-6    (converges in 3 steps)
```

The closed-form PC rule reproduces autograd **to machine precision** — the
derivation is exact, not approximate. The iterative relaxation reaches the same
answer up to an `O(1/beta)` residual bias, confirmed by the β-scan:

```
beta   1e1     1e2     1e3     1e4     1e6
dev    3.0e-2  3.0e-3  3.0e-4  3.0e-5  2.8e-7      # worst weight-grad deviation ~ 1/beta
```

**Sensitivity to inference steps:** with `lr = 1/beta` the state relaxation starts
one step from the fixed point and converges in **2–3 iterations** (the dynamics
near the fixed point are linear with rate β); ≥2 steps already saturates accuracy.

Both are gated in `pytest`. The gate passing was the precondition for everything
below.

## 2. `delta(N)` — **flat → Path B claim holds** (kickoff step 8, SPEC IX.9.2)

`delta(N) = E[1 - cos(g_PC, g_BP)]` per weight tensor, averaged over 5 random
inits, `d=16, m=32, batch=32, beta=1e4`. Two conditions: **masked** (on-task, only
the query position supervised) and **dense** (all N tokens supervised — stresses
N-dependence since then every token contributes to the accumulator sums).

![delta vs N](results/delta_vs_N.png)

```
N        4       8      16      32      64     128     256
masked W_Q  2.3e-8  2.3e-8  2.9e-8  3.1e-8  3.5e-8  3.9e-8  3.8e-8
masked W_K  1.7e-9  1.7e-9  2.2e-9  2.3e-9  2.6e-9  2.9e-9  2.9e-9
masked W_V  1.6e-12 3.4e-12 5.7e-12 8.0e-12 1.0e-11 1.2e-11 1.3e-11
dense  W_Q  1.9e-10 1.1e-10 7.2e-11 4.2e-11 2.2e-11 1.2e-11 6.1e-12
```

**Interpretation (honest).** `delta` sits at the **~1e-8 numerical floor across the
entire N = 4…256 range** — eight orders of magnitude below the `O(1)`
misalignment that would falsify Path B. The masked curves show a *mild* <2×
upward drift (2.3e-8 → 3.8e-8 for W_Q over a 64× change in N); the dense curves
drift *downward*. These sub-2× wiggles are floating-point accumulation artifacts
(the denominator `u^T phi(q)` and the Hebbian sums grow with N), **not** a real
approximation error — they vanish in the closed-form limit and are dwarfed by the
scale that matters. This is the predicted outcome: because `(S, u)` preserve the
global information *exactly*, PC on linear attention is not an approximation of
backprop — it **is** backprop, re-expressed with local updates. `delta(N)` is
flat because there is no locality cost to grow.

> This is the "b ≈ 0" prediction of `SPEC_full.md` PART VI (Path C ties): for
> linear attention the sequence-length coefficient of the alignment error is zero.

## 3. Locality budget = 2 (kickoff step 7, SPEC IX.9.3) — **verified**

`metrics.verify_locality` freezes the two shared signals (`beta*E_S`, `beta*E_u`
— the residuals of the `S` and `u` accumulators) and perturbs a single token's
input. Every *other* token's gradient contribution stays **bit-for-bit identical**
(max leakage `0.0`), and the per-token contributions sum to the aggregate gradient
to 1e-16. So each weight update is a per-token sum reading only token-local
quantities plus exactly **2** broadcast shared states — the Path B target. Holds
at every N in the sweep.

## 4. PC vs backprop accuracy (kickoff step 8, SPEC IX.9.4) — **identical**

Single linear-attention layer trained on associative recall (Adam, identical
init/data-stream/optimizer; only the gradient *source* differs — autograd vs the
PC local rules).

![accuracy vs N](results/accuracy_vs_N.png)

The two curves **coincide exactly at every N** (e.g. N=4: 0.398/0.398; N=64:
0.086/0.086) — PC reproduces backprop's training trajectory, as the gate
predicts.

**On the absolute level:** at the spec's tiny `d=16`, accuracy declines from 0.40
(N=4) toward chance (1/16) by N≈128. This is a **capacity** limit, not a PC
failure — 256 distinct keys cannot be stored in a 16-dim space. Raising capacity
confirms the task is genuinely learnable *and* that parity persists:

```
d=64, m=64:   N=4  0.838/0.838     N=8  0.654/0.654     N=16  0.375/0.375   (bp/pc)
```

### Task design note (flagged)
The "small vocab (16)" is the **value/output** vocabulary (16 classes). Keys are
drawn *distinct* from a larger alphabet (default 512) so a unique binding exists
and `N` can reach 256 — 256 distinct keys is impossible from a size-16 alphabet.
Targets are value *embeddings* and decoding is nearest-embedding, so the whole
task lives inside the Path-B regression free energy `F_pred` (no extra
cross-entropy head), making the PC/BP comparison exact. This is the standard
(M)QAR setup; see `tasks.py`.

## 5. Corrections to the spec's math (kickoff ground rule)

The derivation **result** needed no correction (it matches autograd to 1e-16), but
several equations in `SPEC_full.md` PART IV.B.4 are written loosely. Full worked
derivation and these flags are in [`derivations.md`](derivations.md):

1. **`dz_i/dq_i`** is written with `phi'(q_i)` slotted directly into matrix
   products; correctly it needs `diag(phi'(q_i))` (a Jacobian). The spec's form
   reduces to the right thing once `diag(·)` is inserted — notation, not a math
   error.
2. **`dF/du`** (left as "derive carefully") is
   `+ sum_i (eps_i^T z_i / n_i) phi(q_i) + beta*E_u` — note the **+** sign,
   opposite to the F_pred part of `dF/dS`.
3. **`dF/dW_{Q,K,V}`** are only dimensionally consistent read as **outer products**
   `x_i ⊗ (...)`, not literal matrix products.
4. The weight updates are local **because** the global backprop signal is carried
   by the two shared residuals `beta*E_S, beta*E_u` — which *is* the locality
   budget of 2. The spec leaves this implicit; it is made explicit here.

---

## Summary for the M1 decision

- **(a) Unit test:** passes — closed form to ~1e-16, relaxation to ~1e-5 at
  β=1e4, tolerance `atol=1e-3, rtol=1e-2`.
- **(b) `delta(N)`:** flat at the ~1e-8 fp floor across N=4…256 (mild <2×
  log-scale drift is fp noise). Path B claim (approximately flat in N) **holds**.
- **(c) Accuracy:** PC = backprop exactly at every N; absolute level is
  capacity-limited at d=16 and rises to 0.84 (N=4) at d=64 with parity intact.
- **(d) Derivation:** no result-level correction; four notational/sign issues in
  the spec's written equations flagged in `derivations.md`.

**Caveat carried forward:** M1 is a *single linear* layer, where PC is provably
exact — the honest test of the central research question is depth (M2) and real
softmax (Path A, M4), where a genuine `delta(N,L)` cost is expected to appear.
