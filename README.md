# Predictive Coding on Attention

Can predictive coding (PC), a biologically-plausible local-update alternative to
backprop, train a self-attention layer? Softmax attention breaks PC's usual
locality assumption because its normalizer sums over every token at once. This
repo works through two ways around that (Path B: swap in a decomposable feature
map; Path A: keep real softmax but promote the normalizer to a state node), then
tests the result at depth, in a causal language model, and in a full transformer
block.

Every claim below is backed by a script you can run yourself. Dependencies stay
minimal: `torch`, `numpy`, `matplotlib`.

```bash
python -m venv --system-site-packages .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # if starting fresh

./.venv/Scripts/python.exe tests/test_gradients.py               # correctness gate
./.venv/Scripts/python.exe pc_attention/run_experiment.py        # full M1 sweep, ~40s
```

**Headline results:**

| Check | Result |
|---|---|
| Analytic PC gradients vs. autograd | match to ~1e-16 in closed form; ~1e-5 via relaxation at β=1e4 |
| `delta(N) = E[1 - cos(g_PC, g_BP)]` over N = 4…256 | flat at the ~1e-12 floor, no growth with sequence length |
| Locality budget | 2 shared reads (Path B) or 1 shared scalar (Path A), both empirically verified |
| PC-trained vs. backprop-trained accuracy | identical at every N; curves overlap |

### Layout

| file | role |
|---|---|
| `linear_attn.py` | Path B forward: projections, feature map, accumulators `S, u`, output |
| `pc_updates.py` | extended free energy + analytic local update rules; inference relaxation |
| `backprop_ref.py` | same forward via autograd, the reference gradient |
| `tasks.py` | associative-recall / copy task generator |
| `metrics.py` | `delta = E[1-cos]`; locality-budget verifier |
| `run_experiment.py` | sweeps N, produces the `delta(N)` curve, plots, CSV |
| `derivations.md` | the gradients worked out by hand |

---

## M1 — a single linear-attention layer

Linear attention replaces the softmax kernel with a decomposable feature map
`phi(q)^T phi(k)`, which compresses the sum over all tokens into two shared
accumulator states `(S, u)`. Promote those states to PC state nodes and every
weight update becomes local: a per-token sum plus a read of the two shared
residuals. Full derivation in [derivations.md](derivations.md).

### 1. Correctness gate

`tests/test_gradients.py`, a random single layer with `d=16, m=32, batch=4, N=8`,
float64, tolerance `atol=1e-3, rtol=1e-2`. Actual max deviations:

```
closed form (beta -> inf):   W_Q 4e-16  W_K 6e-16  W_V 2e-15   S 1e-17  u 6e-17   (machine precision)
relaxation  (beta = 1e4):    W_Q 1e-6   W_K 2e-6   W_V 3e-5    S 2e-7   u 1e-6    (converges in 3 steps)
```

The closed-form rule reproduces autograd to machine precision, so the derivation
itself is exact, not an approximation. The iterative relaxation reaches the same
answer up to an `O(1/beta)` bias, confirmed by a beta scan:

```
beta   1e1     1e2     1e3     1e4     1e6
dev    3.0e-2  3.0e-3  3.0e-4  3.0e-5  2.8e-7      # worst weight-grad deviation ~ 1/beta
```

With `lr = 1/beta`, the state relaxation starts one step from the fixed point and
converges in 2-3 iterations (the dynamics near the fixed point are linear with
rate beta). Two steps already saturates accuracy.

### 2. `delta(N)`

`delta(N) = E[1 - cos(g_PC, g_BP)]` per weight tensor, averaged over 5 random
inits (`d=16, m=32, batch=32, beta=1e4`). Two conditions: masked, where only the
query position is supervised (the real task), and dense, where all N tokens are
supervised with random targets, which stresses any N-dependence since every token
then feeds the accumulator sums.

![delta vs N](results/delta_vs_N.png)

```
N        4       8      16      32      64     128     256
masked W_Q  2.0e-12 8.4e-14 5.8e-15 3.7e-16 1.9e-17 2.4e-17 3.5e-17
masked W_K  6.2e-13 9.0e-15 1.4e-16 1.5e-15 1.9e-14 2.9e-13 4.2e-12
masked W_V  5.6e-16 1.2e-17 3.1e-17 2.0e-17 8.3e-18 1.1e-16 1.6e-15
dense  W_Q  1.9e-11 1.9e-12 2.3e-13 3.1e-14 4.1e-15 5.5e-16 6.5e-17
```

`delta` sits at the ~1e-12 machine-precision floor across the whole N=4…256 range,
about ten orders of magnitude below the O(1) misalignment that would signal a real
problem. The small back-and-forth movement (W_K wanders between 1e-16 and 1e-12) is
float64 rounding jitter with no trend in N, not a real approximation error. Because
`(S, u)` preserve the global information exactly, PC on linear attention is not an
approximation of backprop; it computes the same gradient, re-expressed as local
updates. That is why `delta(N)` stays flat: there is no locality cost to grow.

*(An earlier version of this table read ~1e-8 with a slight upward drift, caused by
too large an epsilon in the cosine-similarity guard swamping these tiny gradients.
Fixing the guard dropped the floor to its true ~1e-12 and also affected the M2
numbers below, where the same swamping hit deep-layer gradients harder.)*

### 3. Locality budget

`metrics.verify_locality` freezes the two shared signals (`beta*E_S`, `beta*E_u`,
the residuals of the S and u accumulators) and perturbs a single token's input.
Every other token's gradient contribution stays bit-for-bit identical (max leakage
`0.0`), and the per-token contributions sum to the aggregate gradient to 1e-16. So
each weight update is a per-token sum reading only token-local quantities plus
exactly two broadcast shared states, at every N in the sweep.

### 4. PC vs. backprop accuracy

A single linear-attention layer trained on associative recall with Adam, identical
init/data-stream/optimizer between the two runs; only the gradient source differs.

![accuracy vs N](results/accuracy_vs_N.png)

The two curves coincide exactly at every N (N=4: 0.398/0.398; N=64: 0.086/0.086).
At the small `d=16` used here, accuracy declines from 0.40 (N=4) toward chance
(1/16) by N≈128 — a capacity limit, not a PC failure, since 256 distinct keys
cannot be stored in a 16-dimensional space. Raising capacity confirms the task is
learnable and that parity holds:

```
d=64, m=64:   N=4  0.838/0.838     N=8  0.654/0.654     N=16  0.375/0.375   (bp/pc)
```

The recall task itself: the vocabulary of 16 is the value/output vocabulary, and
keys are drawn distinct from a larger 512-key alphabet so a unique binding exists
even at N=256. Targets are value embeddings and decoding is nearest-embedding, so
the task fits inside the same regression free energy the layer already minimizes,
which keeps the PC/BP comparison exact. See `tasks.py`.

### Notes on the derivation

The closed-form and relaxed gradients match autograd exactly, so no result needed
correcting. A few places in the derivation are easy to get wrong on paper, worth
flagging (worked out fully in [derivations.md](derivations.md)):

- `dz_i/dq_i` needs the full Jacobian `diag(phi'(q_i))`, not the elementwise
  derivative slotted directly into a matrix product.
- `dF/du` carries a **plus** sign on its prediction-error term, opposite to the
  minus in `dF/dS` — easy to flip by habit.
- The weight gradients `dF/dW_{Q,K,V}` are outer products of a per-token quantity
  with the input, not literal matrix products.
- Locality here means exactly two shared reads per update, the residuals
  `beta*E_S` and `beta*E_u` — everything else is genuinely per-token.

### M1 summary

- Unit test passes: closed form to ~1e-16, relaxation to ~1e-5 at β=1e4.
- `delta(N)` is flat at the ~1e-12 floor across N=4…256.
- PC and backprop reach identical accuracy at every N; the absolute level is
  capacity-limited at small d and rises with it, with parity intact.
- The derivation needed no correction, only clearer notation at a few steps.

A single linear layer is where PC is easiest: the harder tests are depth and real
softmax, next.

---

## M2 — depth

Stack `L` linear-attention layers (`a^l = LinAttn_l(a^{l-1})`, plain feedforward,
dense regression target) and measure how PC-vs-backprop alignment behaves with
depth. Code: [deep_linear_attn.py](pc_attention/deep_linear_attn.py),
[pc_updates_deep.py](pc_attention/pc_updates_deep.py),
[run_experiment_m2.py](pc_attention/run_experiment_m2.py).

```bash
./.venv/Scripts/python.exe pc_attention/run_experiment_m2.py            # ~100s
./.venv/Scripts/python.exe pc_attention/run_experiment_m2.py --quick
```

**Which PC.** The naive version of deep PC, recomputing predictions as the
activities move and relaxing to equilibrium, does not equal backprop: the free
activities soak up part of the error, so delta stays large even for a vanishingly
small target nudge (checked directly: ~0.75 at L=2). Instead this uses
fixed-prediction PC (Song et al., NeurIPS 2020): predictions and layer Jacobians
are held at their feedforward values during inference. The downward error at each
activity, `(d pred^{l+1}/da^l)^T e^{l+1}`, is a one-layer transpose computed by
autograd on that single layer, never a backward pass through the whole stack.

**Converged inference recovers backprop at every depth.** Run the activity
relaxation to convergence and delta collapses to the numerical floor across the
whole grid: worst delta over N∈{4..64} × L∈{1,2,4,8,16}, full relaxation, is
**1.7e-8**. A deep stack of linear attention trained by fixed-prediction PC
computes the exact backprop gradient; there is no intrinsic depth or length
penalty.

**Under a bounded inference budget, depth costs, length doesn't.** The output
error travels roughly one layer per inference step, so with a fixed budget the
lower layers haven't heard the error yet. `delta(N, L)` at a fixed budget of 4
steps:

![delta(N,L) heatmap](results/m2_delta_heatmap.png)

The bands are perfectly horizontal: delta grows with depth L (0 → 0.38 → 0.69 as L
goes 4 → 8 → 16) and stays flat in N to ~1e-8 (std across N per row ≤ 1.6e-8).

**The depth cost is inference latency, not an irreducible error.** Give inference
more steps and each depth's delta falls off a cliff once the budget reaches
roughly L:

![delta vs inference budget](results/m2_convergence.png)

Depth L wants on the order of L inference steps; sequence length N wants none.

A natural conjecture is that the alignment error grows with depth and length as
`delta(N,L) ~ 1 - (1-a)^L (1 - b log N)`. For linear attention with
fixed-prediction PC the cleaner description is `b = 0` exactly, and the depth term
is not a smooth exponential decay of magnitude but a hard propagation front in
direction, set by the inference budget T: `delta` is roughly
`max(0, (L - c*T)/L)`. (Gradient magnitudes do vanish with depth — deep layers sit
at ~1e-8 — but their direction is exact once the front reaches them; that
magnitude decay is the network's own vanishing-gradient behavior, present in
backprop too, not something PC adds.)

### M2 summary

- Fixed-prediction PC equals backprop at all tested (N, L) once inference
  converges, worst delta 1.7e-8.
- With a fixed inference budget, delta grows with L purely because the error
  needs roughly L steps to reach the bottom layer.
- delta is flat in N at every depth, extending the M1 result to the deep case.
- Fix found along the way: the delta metric's cosine guard (`eps=1e-12`) was
  large enough to swamp tiny deep-layer gradients and fake a misalignment;
  tightened to `1e-30`. This also lowered the reported M1 floor from ~1e-8 to
  its true ~1e-12.

---

## Path A — real softmax with a normalizer state node

Path B drops softmax. Path A keeps it and promotes only the denominator to a
per-token state node stored in log space, so the one global operation left is the
scalar `c_i = log sum_j exp(s_ij)`, the divisive-normalization pool. Derivation in
[derivations.md](derivations.md) part A; code in
[softmax_attn.py](pc_attention/softmax_attn.py),
[pc_updates_A.py](pc_attention/pc_updates_A.py).

```bash
./.venv/Scripts/python.exe tests/test_gradients_A.py
./.venv/Scripts/python.exe pc_attention/run_experiment_A.py
```

**Correctness gate.** Analytic Path A gradients vs. softmax autograd, d=16, N=8:

```
closed form (gamma -> inf):  W_Q 4e-15  W_K 4e-15  W_V 3e-15   (machine precision)
relaxation  (gamma = 1e5):   W_Q 9e-4   W_K 8e-4   W_V 1e-3    (converges in 4 steps)
gamma scan (worst weight dev):  1e2->1.1   1e3->0.13   1e4->0.014   1e5->1.4e-3   ~ 1/gamma
```

The local rules recover real softmax-attention backprop exactly at
`gamma -> inf`. The one non-local read per token is the scalar `c_i`: a locality
budget of 1, versus Path B's 2.

**delta(N), flat like Path B.**

![Path A delta(N)](results/pathA_delta_vs_N.png)

Closed-form softmax PC sits at the ~1e-17 floor across N, overlapping Path B;
finite `gamma=1e4` sits at ~1e-7 and is flat to slightly decreasing in N. One
might expect Path A to degrade with N as the softmax pool grows, but it doesn't:
`c_i` is the exact log-sum-exp, so nothing global is being approximated away.

**gamma-scan: delta ~ 1/gamma², flat in N.**

![Path A gamma-scan](results/pathA_gamma_scan.png)

`delta` falls two decades per decade of gamma (it is `1 - cos`, second-order in
the `O(1/gamma)` gradient error), and the N=8/32/128 curves lie on top of each
other: larger normalizer precision shrinks the gap, with no N dependence.

---

## Path C — the cost of forcing locality

How does the PC-vs-backprop gap actually scale? Code:
[run_experiment_C.py](pc_attention/run_experiment_C.py).

```bash
./.venv/Scripts/python.exe pc_attention/run_experiment_C.py
```

This ties together the three axes measured across the project and adds a
deliberately naive local rule for contrast: Path A with the global redistribution
term (`-z_i` in the softmax Jacobian) dropped entirely, to see where a length cost
actually shows up.

![Path C](results/pathC_delta_vs_N.png)

```
method                             delta(N=4 .. 256)             b (slope per log2 N)
Path B exact (linear)              ~1e-17 (flat)                 0.0000
Path A exact (softmax)             ~1e-17 (flat)                -0.0000
naive-local, diffuse attn (x1)     0.17 -> 0.045 (shrinks)      -0.0173
naive-local, sharp attn (x12)      0.56 -> 0.61 (grows)         +0.0106
```

**Length axis.** The exact methods give a length coefficient of 0: Path B keeps
the global sum in `(S, u)`, Path A keeps it in `c_i`, so nothing is approximated
away and there is no length cost. The naive rule is the only one that pays a
length cost, and its sign depends on attention sharpness — with diffuse attention
the dropped term vanishes as N grows so the cost shrinks, with peaked attention it
grows. A single `b log N` term doesn't capture that; the cost is real (around 0.5)
but its N-trend is regime-dependent.

**Depth axis.** From M2, at converged inference the depth coefficient is 0
(fixed-prediction PC equals backprop at every depth). The depth cost that does
exist is inference latency, not an irreducible approximation error.

**Normalizer precision.** From Path A, `delta ~ 1/gamma^2`. Larger precision
shrinks the gap.

**Bottom line.** For attention layers whose global information is preserved
exactly, Path B's `(S, u)` or Path A's `c_i`, forcing locality is free: delta sits
at the numerical floor, flat in both N and L at convergence. A cost of order
0.1-0.6 appears only when the global term is thrown away, and even then its
dependence on N is set by how peaked the attention is, not by a clean `log N`.

---

## M3 — a causal char-level LM trained by PC

The final ladder step: a real autoregressive model. One causal linear-attention
layer (position i sees only j ≤ i), an embedding, sinusoidal positions, and a
softmax cross-entropy head. Code:
[causal_linear_attn.py](pc_attention/causal_linear_attn.py),
[char_lm.py](pc_attention/char_lm.py),
[run_experiment_m3.py](pc_attention/run_experiment_m3.py).

```bash
./.venv/Scripts/python.exe tests/test_gradients_causal.py
./.venv/Scripts/python.exe pc_attention/run_experiment_m3.py     # ~20s
```

Causal linear attention makes the two accumulators running (prefix) sums; the
gradient is the mirror image, a suffix sum (reverse scan), since a key at position
j feeds every query i ≥ j. The local rule matches autograd to ~1e-15
(`tests/test_gradients_causal.py`), and causality is checked directly: perturbing
a future token leaves past outputs untouched.

Every parameter has an exact local rule: the attention weights from the causal PC
gradient, the output head from a Hebbian outer product, the embedding from the
attention layer's one-layer input transpose, and the softmax cross-entropy error
is just `p - y`. The whole model's PC gradient equals its backprop gradient, worst
deviation 2.9e-15 across all five parameter tensors.

![M3 training](results/m3_training.png)

Trained from the same init on the same data with the same optimizer, PC and
backprop produce the same curve: max loss gap over 600 steps is 3.1e-15. The LM
learns (loss 3.48 → 0.03, next-char accuracy 0.02 → 0.99), and the PC-trained model
reproduces the text:

```
the quick brown fox jumps over the lazy dog. the quick brown fox
```

---

## Overall status

| Milestone | What | Result |
|---|---|---|
| M1 | single linear-attention layer, Path B | PC = backprop to ~1e-16; delta(N) flat at floor; locality budget 2 |
| M2 | depth L stack, fixed-prediction PC | PC = backprop at all (N,L) once inference converges; depth cost is latency |
| Path A | real softmax + normalizer state node | PC = softmax backprop to ~4e-15; locality budget 1 scalar; delta ~1/gamma² |
| Path C | cost of forcing locality | exact paths have zero length cost; naive local rule costs ~0.5 |
| M3 | causal char-LM, PC vs. backprop | PC = backprop to ~3e-15 across all params; LM learns to 0.99 accuracy |

Gates: `tests/test_gradients{,_A,_causal}.py`. Experiments:
`pc_attention/run_experiment{,_m2,_A,_C,_m3}.py`. Derivations for Path B and Path
A are in [derivations.md](derivations.md). Further extensions (finite inference
budgets, a full transformer block, feedback alignment) are in
[experiments/](experiments/).

Across linear and softmax attention, single-layer and deep, non-causal and
causal, up to a real char-LM, PC's local updates reproduce backprop exactly
whenever the attention layer's global information is preserved (Path B's `(S,u)`,
Path A's `c_i`) and inference runs to convergence. The costs that remain are a
state-relaxation bias of `O(1/beta)` or `O(1/gamma^2)`, an inference-latency
penalty with depth, and — if the global term is deliberately dropped — an order-1
misalignment. Sequence length never hurts on its own.
