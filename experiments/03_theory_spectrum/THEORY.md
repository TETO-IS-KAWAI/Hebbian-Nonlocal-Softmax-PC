# The PC / attention locality spectrum — consolidated statements

This folder collects the project's results as clean claims, and traces the
**locality spectrum**: how much non-local signal an attention layer's local
learning rule needs, and what it costs when you provide less. Everything here is
proved constructively in `../../derivations.md` and checked numerically by the
gates and experiments cited.

Notation: `g_PC` = the analytic local update, `g_BP` = backprop. `delta = E[1 -
cos(g_PC, g_BP)]`. "Local" = each weight update is a per-token (or per-position)
sum reading only that site's own quantities plus a small, named set of shared
signals.

---

## Claim 1 (Path B, linear attention)

Give the two feature-map accumulators `S = sum_j phi(k_j) v_j^T` and
`u = sum_j phi(k_j)` the status of state nodes, with consistency precision `beta`.
Then gradient descent on the extended free energy has **all updates local**, using
exactly **two** shared reads (the residuals `beta*(S-S*)`, `beta*(u-u*)`), and as
`beta -> inf` at the inference fixed point,

> `g_PC(W_Q,W_K,W_V) = g_BP` exactly; the finite-`beta` bias is `O(1/beta)`.

*Proof:* derivations.md §3–6. *Check:* `tests/test_gradients.py` (closed form
~1e-16, relaxation ~1e-5 at beta=1e4, bias ∝ 1/beta). *Length:* `delta(N)` flat at
the ~1e-12 floor for N up to 256 (`run_experiment.py`).

## Claim 2 (Path A, softmax attention)

Keep the real softmax; promote only the log-normalizer `l_i` (= log of the softmax
denominator) to a per-token state node with precision `gamma` and target the true
log-sum-exp `c_i`. Then every vector/matrix update is local, using exactly **one**
shared per-token scalar (`c_i`, the divisive-normalization pool), and as
`gamma -> inf`,

> `g_PC(W_Q,W_K,W_V) = g_BP` for real softmax attention, exactly; the finite-`gamma`
> bias is `O(1/gamma)`, so `delta ~ 1/gamma^2`.

*Proof:* derivations.md PART A. *Check:* `tests/test_gradients_A.py` (closed form
~4e-15, bias ∝ 1/gamma); `run_experiment_A.py` (delta ∝ 1/gamma^2, flat in N).

## Claim 3 (depth)

Stack the layers and put activity nodes between them (fixed-prediction PC). At
converged inference,

> `g_PC = g_BP` at every depth `L` and length `N` (worst delta 1.7e-8 over the
> grid). The depth cost under a *bounded* inference budget `T` is pure latency: the
> error advances ~one layer per step, so a budget `T ~ L` suffices and `delta` is
> independent of `N`.

*Check:* `run_experiment_m2.py`; `experiments/01_finite_budget_online/` shows
training needs only `T ~ depth`, not converged inference.

## Claim 4 (causal / autoregressive)

For causal linear attention the accumulators become prefix (forward-scan) sums and
the error signal a suffix (backward-scan) sum. The local rule is unchanged in
spirit and still equals backprop.

*Check:* `tests/test_gradients_causal.py` (~1e-15 + a causality test);
`run_experiment_m3.py` (a char-LM trains identically to backprop, ~3e-15 gap).

## Claim 5 (the cost of *less* locality)

If the global term is thrown away (the naive rule that drops the softmax
redistribution `-z_i`, i.e. Path A with the normalizer error `r_i = 0`), the
updates become strictly local (zero shared reads) but no longer track backprop.
`delta` is order 0.1-0.6, and its sign in `N` depends on attention sharpness:
it shrinks for diffuse attention and grows for peaked attention. A single
`b log N` length penalty doesn't capture that; a length cost only shows up once
the normalizer is approximated, and even then its direction is regime-dependent.

*Check:* `run_experiment_C.py`; the `alpha`-scan in this folder's `run.py`
interpolates continuously from naive (`alpha=0`) to exact (`alpha=1`).

---

## The spectrum

| method | softmax? | shared non-local reads per update | `g_PC = g_BP`? |
|---|---|---|---|
| plain backprop | — | the whole stored forward pass | (by definition) |
| **Path B** (linear) | no | **2** shared states `(S, u)` | yes, `beta -> inf` |
| **Path A** (softmax) | yes | **1** per-token scalar `c_i` | yes, `gamma -> inf` |
| naive local | yes | **0** | no (delta ~ 0.1–0.6) |

Reading the table top to bottom trades global information for locality. The two
middle rows are the useful discovery: you can pay just one scalar (Path A) or two
shared states (Path B) and still get backprop *exactly*. Pay nothing (bottom row)
and you lose it. There is no free lunch below one scalar per token, but one scalar
is enough — and it is exactly the divisive-normalization pool cortex is already
believed to compute.

`run.py` in this folder produces the continuous version of the last two rows: the
`alpha`-scan sweeping the fraction of the redistribution term that is kept,
tracing `delta` from the naive floor up to machine-zero.
