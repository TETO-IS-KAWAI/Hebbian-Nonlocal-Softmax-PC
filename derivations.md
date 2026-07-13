# Derivations — Path B (linear attention) local update rules

This file works through the analytic gradients of the extended free energy
`F_ext` for Path B by hand, before they are implemented in
`pc_attention/pc_updates.py`. `dF/dS` and `dF/dW_Q` fall out fairly directly;
`dF/du` and the `dz_i/dq_i` Jacobian take more care and are easy to get
dimensionally wrong on a first pass, so both are worked out fully here, with the
easy-to-miss steps flagged in section 7.

Everything below is verified numerically against autograd in
`tests/test_gradients.py`.

---

## 1. Notation and dimensions

Per sequence (batch handled by summing independent sequences):

| symbol | shape | meaning |
|---|---|---|
| `x_i`            | `R^d`      | input token `i` (`i = 1..N`) |
| `W_Q, W_K`       | `R^{d×m}`  | query/key projections |
| `W_V`            | `R^{d×d_v}`| value projection (`d_v = d` here) |
| `q_i = W_Q^T x_i`| `R^m`      | query |
| `k_j = W_K^T x_j`| `R^m`      | key |
| `v_j = W_V^T x_j`| `R^{d_v}`  | value |
| `phi(·)`         | `R^m→R^m`  | feature map, elementwise, `phi(x)=elu(x)+1` |
| `S`              | `R^{m×d_v}`| accumulator state node |
| `u`              | `R^m`      | accumulator state node |
| `x_i^out`        | `R^{d_v}`  | target / output belief |

Feature-map target values (what the states are pinned toward):

```
S* = sum_j phi(k_j) v_j^T        in R^{m×d_v}
u* = sum_n phi(k_n)              in R^m
```

Per-token scalars/vectors used repeatedly:

```
n_i   = u^T phi(q_i)              (denominator, scalar; guard with +eps)
z_i   = S^T phi(q_i) / n_i        in R^{d_v}    (prediction)
eps_i = x_i^out - z_i             in R^{d_v}    (prediction error)
```

Note `z_i` uses the **state** `S,u`, while `S*,u*` are the **targets** built
from the keys/values. In the forward pass `S=S*, u=u*`; they separate during
inference relaxation.

---

## 2. Extended free energy

```
F_ext   = F_pred + F_state
F_pred  = (1/2) sum_i || x_i^out - z_i ||^2
F_state = (beta/2) || S - S* ||_F^2 + (beta/2) || u - u* ||^2
```

`beta` = precision of state-consistency. `dF_pred/dz_i = -eps_i`.

Define the two residuals (these carry the learning signal):

```
E_S = S - S*     in R^{m×d_v}
E_u = u - u*     in R^m
```

---

## 3. State-node gradients (inference / relaxation)

### 3.1 dF/dS

`z_i = (1/n_i) S^T phi(q_i)`, and `n_i` does **not** depend on `S`. Component form
`z_i[a] = (1/n_i) sum_b S[b,a] phi(q_i)[b]`, so
`dz_i[a]/dS[b,c] = (1/n_i) phi(q_i)[b] delta_{a,c}`. Contracting with
`dF_pred/dz_i = -eps_i`:

```
dF_pred/dS = - sum_i phi(q_i) eps_i^T / n_i          in R^{m×d_v}
dF/dS      = - sum_i phi(q_i) eps_i^T / n_i  +  beta E_S
```

Each term is a per-token outer product summed into a shared matrix, so this is local.

### 3.2 dF/du (the term that's easy to get wrong)

Now `u` enters only through `n_i = u^T phi(q_i)`. Write `z_i = S^T phi(q_i) · n_i^{-1}`:

```
dz_i[a]/du[b] = (S^T phi(q_i))[a] · (-1) n_i^{-2} · dn_i/du[b]
              = -(S^T phi(q_i))[a] phi(q_i)[b] / n_i^2
```

Using `S^T phi(q_i) = n_i z_i`:

```
dz_i[a]/du[b] = - z_i[a] phi(q_i)[b] / n_i
```

Contract with `dF_pred/dz_i = -eps_i`:

```
dF_pred/du[b] = sum_i sum_a (-eps_i[a]) · (- z_i[a] phi(q_i)[b] / n_i)
              = sum_i (eps_i^T z_i) phi(q_i)[b] / n_i
```

Therefore

```
dF/du = + sum_i ( eps_i^T z_i / n_i ) phi(q_i)  +  beta E_u        in R^m
```

**Sign note:** the `F_pred` part of `dF/du` is **+** (a positive multiple of
`phi(q_i)`), in contrast to `dF/dS` which is **−**. The scalar
`(eps_i^T z_i)/n_i` is per-token and only reads token-local quantities plus the
shared `S,u`, so it stays local. Define `H := sum_i (eps_i^T z_i / n_i) phi(q_i)`.

### 3.3 Inference fixed point (the key identity)

At the relaxation fixed point `dF/dS = 0` and `dF/du = 0`, so the residuals,
though `O(1/beta)` small, satisfy

```
beta E_S = sum_i phi(q_i) eps_i^T / n_i   =:  G_S      (finite, beta-independent)
beta E_u = - sum_i (eps_i^T z_i / n_i) phi(q_i) = -H   (finite, beta-independent)
```

This is why `beta -> inf` recovers backprop: the *product* `beta·E` stays finite
and equals the backprop gradient of the accumulator (see §6). Initialising the
relaxation at `S=S*, u=u*` and stepping with `lr ≈ 1/beta` reaches this fixed
point in a handful of steps because near it the dynamics are essentially linear.

---

## 4. Prediction Jacobian dz_i/dq_i (needed for W_Q)

`phi` is elementwise, so `d phi(q_i)/d q_i = diag(phi'(q_i))`. With
`num_i = S^T phi(q_i)` and `n_i = u^T phi(q_i)`:

```
d num_i / d q_i = S^T diag(phi'(q_i))                  in R^{d_v×m}
d n_i  / d q_i  = phi'(q_i) ⊙ u                        in R^m   (as a row: u^T diag(phi'(q_i)))
```

Quotient rule, using `num_i = n_i z_i`:

```
dz_i/dq_i = (1/n_i) [ S^T diag(phi'(q_i)) - z_i (phi'(q_i) ⊙ u)^T ]    in R^{d_v×m}
```

We never form this matrix explicitly; we only need `(dz_i/dq_i)^T eps_i`:

```
(dz_i/dq_i)^T eps_i = (1/n_i) phi'(q_i) ⊙ [ S eps_i - u (z_i^T eps_i) ]   in R^m
```

---

## 5. Weight gradients (learning)

`W_Q` enters only `F_pred` (through `q_i`); `W_K, W_V` enter only `F_state`
(through the targets `S*, u*`). With `q_i = W_Q^T x_i` we have
`dF/dW = sum x_· (dF/d·)^T` (outer products).

### 5.1 W_Q

```
dF/dq_i = (dz_i/dq_i)^T (-eps_i) = -(1/n_i) phi'(q_i) ⊙ [ S eps_i - u (z_i^T eps_i) ]
dF/dW_Q = sum_i x_i (dF/dq_i)^T                                      in R^{d×m}
```

Written as an outer product, `dF/dW_Q = - sum_i x_i ⊗ [(dz_i/dq_i)^T eps_i]`, using
only `x_i, phi(q_i), phi'(q_i), S, u, eps_i`.

### 5.2 W_K

`W_K` enters `S*` (via `phi(k_j) v_j^T`) and `u*` (via `phi(k_j)`). From
`F_state`:

```
dF/dphi(k_j) = -beta E_S v_j  -  beta E_u          (via S* and u* respectively)
dF/dk_j      = phi'(k_j) ⊙ ( -beta E_S v_j - beta E_u )
dF/dW_K      = sum_j x_j (dF/dk_j)^T
             = - sum_j x_j [ phi'(k_j) ⊙ ( beta E_S · v_j + beta E_u ) ]^T
```

At the fixed point substitute `beta E_S = G_S`, `beta E_u = -H`:

```
dF/dW_K = - sum_j x_j [ phi'(k_j) ⊙ ( G_S v_j - H ) ]^T
```

`v_j = W_V^T x_j` is token-local, and `G_S, H` are just shared-state reads.

### 5.3 W_V

`W_V` enters `S*` only:

```
dF/dv_j = -beta E_S^T phi(k_j)
dF/dW_V = sum_j x_j (dF/dv_j)^T = - sum_j x_j ( beta E_S^T phi(k_j) )^T
```

At the fixed point `beta E_S = G_S`:

```
dF/dW_V = - sum_j x_j ( G_S^T phi(k_j) )^T                          in R^{d×d_v}
```

**Local.**

---

## 6. Equivalence to backprop (why the unit test must pass)

Let `L = (1/2) sum_i || x_i^out - z_i ||^2` be the plain forward loss with
`z_i` built directly from `S*, u*` (this is `backprop_ref.py`). Standard
backprop gives:

```
dL/dS* = - sum_i phi(q_i) eps_i^T / n_i = -G_S
dL/du* = + sum_i (eps_i^T z_i / n_i) phi(q_i) = +H
```

Compare with §3.3: `beta E_S = G_S = -dL/dS*` and `beta E_u = -H = -dL/du*`.
So the PC state residuals reconstruct the backprop accumulator gradients up to
sign: **`beta E_S = -dL/dS*`, `beta E_u = -dL/du*`.** Substituting these into
§5.2 / §5.3 turns the PC weight gradients into exactly the backprop chain-rule
expressions:

```
dL/dv_j     = dL/dS*^T phi(k_j) = -G_S^T phi(k_j) = dF/dv_j            ⇒  dF/dW_V = dL/dW_V
dL/dphi(k_j)= dL/dS* v_j + dL/du* = -(G_S v_j - H)                     ⇒  dF/dW_K = dL/dW_K
```

and `dF/dW_Q = dL/dW_Q` directly (`W_Q` path is identical in both). Hence, at
the inference fixed point with `beta -> inf`:

```
g_PC(W_Q) = g_BP(W_Q),  g_PC(W_K) = g_BP(W_K),  g_PC(W_V) = g_BP(W_V)
```

exactly (to `O(1/beta)`). For the accumulator gradients we report the signed
match `-beta E_S ?= dL/dS*` and `-beta E_u ?= dL/du*`.

---

## 7. Notes on notation (the steps most likely to trip you up)

1. **`dz_i/dq_i` needs a full Jacobian, not the elementwise derivative dropped
   straight into a matrix product.** It's tempting to write something like
   `S^T phi'(q_i)`, treating `phi'(q_i)` as if it could sit inside the matrix
   product on its own. Since `phi` is elementwise, its Jacobian is the diagonal
   matrix `diag(phi'(q_i))`, so the correct object is (§4):
   `(1/n_i)[ S^T diag(phi'(q_i)) - z_i (phi'(q_i)⊙u)^T ]`. Any looser shorthand
   that omits the `diag(·)` will look plausible but produce the wrong shape.

2. **`dF/du` carries a plus sign** on its prediction-error term:
   `+ sum_i (eps_i^T z_i / n_i) phi(q_i) + beta E_u` (§3.2), opposite to the
   minus on the equivalent term in `dF/dS`. Easy to get backwards by pattern
   matching against `dF/dS`.

3. **`dF/dW_Q` (and W_K, W_V) are outer products**, not literal matrix products:
   `x_i ⊗ [(dz_i/dq_i)^T eps_i]` (§5.1), where `x_i ∈ R^d` and the bracketed term
   is in `R^m`. Writing it as a plain product mixes the shapes.

4. **The locality claim rests entirely on `beta E_S` and `beta E_u`.** Every
   per-token weight update is local *because* the only cross-token information
   it needs is carried by these two shared residuals — that pair is what
   README.md calls the "locality budget = 2."

---

# PART A — Softmax attention with a normalizer state node (Path A)

This is the derivation behind `pc_updates_A.py`. Path A keeps the *real* softmax
(no feature-map surrogate) and promotes only the per-token normalizer to a state
node, so the single global operation left is one scalar per token.

## A.1 Forward and notation

Per query `i`, key `j` (`d_k` = query/key dim, scale `tau = 1/sqrt(d_k)`):

```
s_ij   = tau * q_i . k_j
E_ij   = exp(s_ij)
zeta*_i = sum_j E_ij              (true normalizer)   ,   c_i = log zeta*_i  (true LSE)
z_tilde_i = sum_j E_ij v_j        (un-normalized output)
```

Promote the normalizer to a state node. We store it in **log space** as
`l_i` (= "log zeta_i"), which is where softmax's dynamic range lives:

```
z_i   = z_tilde_i * exp(-l_i) = sum_j b_ij v_j ,   b_ij := exp(s_ij - l_i)
eps_i = x_i^out - z_i
a_ij  := exp(s_ij - c_i)          (true softmax weights, sum_j a_ij = 1)
```

`b_ij` uses the **state** normalizer `l_i`; `a_ij` uses the **true** one `c_i`.
They coincide when the state has relaxed to the truth (`l_i = c_i`).

## A.2 Extended free energy

```
F_ext_A = F_pred + F_norm
F_pred  = (1/2) sum_i || x_i^out - z_i ||^2
F_norm  = (gamma/2) sum_i ( l_i - c_i )^2
```

`gamma` = precision pinning the state normalizer to the true LSE `c_i`. This is
the divisive-normalization "pool" of PART V: the only non-local read is the scalar
`c_i` per token.

## A.3 State-node update (the log-normalizer)

`z_i = z_tilde_i exp(-l_i)`, so `dz_i/dl_i = -z_i`.

```
dF_pred/dl_i = (-eps_i) . (-z_i) = eps_i . z_i
dF_norm/dl_i = gamma ( l_i - c_i )
dF/dl_i      = eps_i . z_i + gamma ( l_i - c_i )
```

Define the **normalizer error** `r_i := gamma ( l_i - c_i )`. At the inference
fixed point `dF/dl_i = 0`:

```
r_i = - eps_i . z_i          (finite, gamma-independent -- like beta*E in Path B)
```

Relaxing `l_i` (init at `c_i`, step `lr ~ 1/gamma`) reaches this in a few steps,
same reasoning as Path B §3.3.

## A.4 Error at the logits, then the weights

Everything factors through `s_ij`. With `l_i` held as a state (const w.r.t. `s`):

```
dz_i/ds_ij   = b_ij v_j                       => dF_pred/ds_ij = - b_ij (eps_i . v_j)
dc_i/ds_ij   = a_ij                           => dF_norm/ds_ij = - r_i a_ij
dF/ds_ij     = - b_ij (eps_i . v_j) - r_i a_ij
```

Chain `s_ij = tau q_i . k_j` and `v_j = W_V^T x_j` to the weights (outer products,
`dF/dW = sum x (dF/d*)^T`):

```
dF/dq_i = tau * sum_j [ -b_ij (eps_i . v_j) - r_i a_ij ] k_j
dF/dk_j = tau * sum_i [ -b_ij (eps_i . v_j) - r_i a_ij ] q_i
dF/dv_j = - sum_i b_ij eps_i
dF/dW_Q = sum_i x_i (dF/dq_i)^T,  dF/dW_K = sum_j x_j (dF/dk_j)^T,  dF/dW_V = sum_j x_j (dF/dv_j)^T
```

Reads per update: token-local `x, q, k, v, eps`, plus the one shared per-token
scalar the normalizer carries (`c_i` via `a_ij`, or equivalently `r_i`). That is a
locality budget of 1, versus Path B's 2.

## A.5 Equivalence to softmax backprop (the gate)

Let `L = (1/2) sum_i ||x_i^out - z*_i||^2` with the *true* softmax output
`z*_i = sum_j a_ij v_j` (this is `backprop_ref` for Path A). As `gamma -> inf` the
state relaxes to `l_i = c_i`, so `b_ij -> a_ij`, `z_i -> z*_i`, and
`r_i -> -(eps_i . z*_i)`. Substituting `r_i` into `dF/dq_i`:

```
dF/dq_i = -tau sum_j a_ij ( eps_i . (v_j - z*_i) ) k_j
```

which is exactly the softmax-attention backprop `dL/dq_i` (the softmax Jacobian is
`dz*_i/ds_ij = a_ij (v_j - z*_i)`). `dF/dW_K`, `dF/dW_V` match the same way. So at
the fixed point with `gamma -> inf`:

```
g_PC_A(W_Q) = g_BP(W_Q),  g_PC_A(W_K) = g_BP(W_K),  g_PC_A(W_V) = g_BP(W_V)
```

exactly (to `O(1/gamma)`). This is the Path A correctness gate.

## A.6 What Path A does and does not buy (honest)

Computing `c_i = log sum_n exp(s_in)` is still a sum over all keys -- Path A does
**not** remove globality. It concentrates it into one scalar per token (the
divisive-normalization pool), leaving every vector/matrix update Hebbian and
local. Contrast the budgets: Path B = 2 shared *states* (S, u) but no softmax;
Path A = 1 shared *scalar* per token (c_i) with the real softmax kept.
