"""Forward pass for the linear-attention layer (Path B).

Ordinary attention normalises with a softmax whose denominator sums over every
token at once. Linear attention swaps the exp(q.k) kernel for phi(q).phi(k) with a
non-negative feature map phi. The payoff is that the big sum over tokens collapses
into two running totals:

    S = sum_j phi(k_j) v_j^T        (m x d_v matrix)
    u = sum_j phi(k_j)              (length-m vector)

and the output for token i is just

    z_i = (S^T phi(q_i)) / (u^T phi(q_i)).

So token i only needs its own q_i plus the two shared totals S and u; it never
looks at the other tokens one by one. That locality is the whole reason predictive
coding can train this layer (see pc_updates.py).

Both gradient implementations -- the analytic PC rules (pc_updates.py) and the
autograd reference (backprop_ref.py) -- call this same function, so a gradient
mismatch means a real disagreement rather than two subtly different forwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

Tensor = torch.Tensor


# ---- feature maps -----------------------------------------------------------
# phi is applied element-wise, so its Jacobian is the diagonal matrix
# diag(phi'(x)). We keep phi and its element-wise derivative phi' together because
# the gradient code needs both; swapping the feature map (elu+1 -> relu -> ...) is
# then just swapping this pair.
@dataclass(frozen=True)
class FeatureMap:
    name: str
    phi: Callable[[Tensor], Tensor]
    phi_prime: Callable[[Tensor], Tensor]

    def __call__(self, x: Tensor) -> Tensor:  # convenience
        return self.phi(x)


def _elu_plus_one(x: Tensor) -> Tensor:
    return torch.nn.functional.elu(x) + 1.0


def _elu_plus_one_prime(x: Tensor) -> Tensor:
    # phi(x) = elu(x) + 1, which is x+1 for x>0 and exp(x) for x<=0.
    # Its derivative is 1 for x>0 and exp(x) for x<=0.
    return torch.where(x > 0, torch.ones_like(x), torch.exp(x))


def _relu(x: Tensor) -> Tensor:
    return torch.relu(x)


def _relu_prime(x: Tensor) -> Tensor:
    return (x > 0).to(x.dtype)


FEATURE_MAPS: dict[str, FeatureMap] = {
    "elu+1": FeatureMap("elu+1", _elu_plus_one, _elu_plus_one_prime),
    "relu": FeatureMap("relu", _relu, _relu_prime),
}


def get_feature_map(fm: "str | FeatureMap") -> FeatureMap:
    if isinstance(fm, FeatureMap):
        return fm
    if fm not in FEATURE_MAPS:
        raise KeyError(f"unknown feature map {fm!r}; have {list(FEATURE_MAPS)}")
    return FEATURE_MAPS[fm]


# ---- forward ----------------------------------------------------------------
@dataclass
class ForwardState:
    """Everything the forward pass computed, bundled so the gradient code can
    reuse it instead of recomputing. All tensors carry a leading batch axis B."""

    X: Tensor          # (B, N, d)   input tokens
    Q: Tensor          # (B, N, m)
    K: Tensor          # (B, N, m)
    V: Tensor          # (B, N, d_v)
    phi_q: Tensor      # (B, N, m)
    phi_k: Tensor      # (B, N, m)
    phi_q_prime: Tensor  # (B, N, m)
    phi_k_prime: Tensor  # (B, N, m)
    S: Tensor          # (B, m, d_v)  accumulator  S = sum_j phi(k_j) v_j^T
    u: Tensor          # (B, m)       accumulator  u = sum_j phi(k_j)
    denom: Tensor      # (B, N)       n_i = u^T phi(q_i) + eps  (guarded)
    z: Tensor          # (B, N, d_v)  output
    eps_guard: float

    @property
    def batch(self) -> int:
        return self.X.shape[0]

    @property
    def seq_len(self) -> int:
        return self.X.shape[1]


def predict(S: Tensor, u: Tensor, phi_q: Tensor, eps_guard: float) -> tuple[Tensor, Tensor]:
    """Compute z_i = S^T phi(q_i) / (u^T phi(q_i) + eps) for every token.

    Pulled out on its own because the inference loop in pc_updates.py re-evaluates
    z as it nudges S and u around, not just at their original values. The eps in
    the denominator guards against a divide-by-zero if u^T phi(q_i) is ever tiny.
    Returns z and the (guarded) denominator, since callers want the denominator too.
    """
    num = torch.einsum("bmd,bnm->bnd", S, phi_q)   # (B, N, d_v)
    denom = torch.einsum("bm,bnm->bn", u, phi_q) + eps_guard  # (B, N)
    z = num / denom.unsqueeze(-1)
    return z, denom


def linear_attention_forward(
    X: Tensor,
    W_Q: Tensor,
    W_K: Tensor,
    W_V: Tensor,
    feature_map: "str | FeatureMap" = "elu+1",
    eps_guard: float = 1e-6,
) -> ForwardState:
    """Run the layer. Shapes: X (B, N, d); W_Q, W_K (d, m); W_V (d, d_v).

    B = batch, N = sequence length, d = model dim, m = feature-map dim.
    """
    fm = get_feature_map(feature_map)

    Q = X @ W_Q          # (B, N, m)
    K = X @ W_K          # (B, N, m)
    V = X @ W_V          # (B, N, d_v)

    phi_q = fm.phi(Q)
    phi_k = fm.phi(K)
    phi_q_prime = fm.phi_prime(Q)
    phi_k_prime = fm.phi_prime(K)

    # The two running totals. Summing over the token axis here is where the whole
    # sequence gets folded down into a fixed-size state.
    S = torch.einsum("bnm,bnd->bmd", phi_k, V)   # (B, m, d_v)
    u = phi_k.sum(dim=1)                          # (B, m)

    z, denom = predict(S, u, phi_q, eps_guard)

    return ForwardState(
        X=X, Q=Q, K=K, V=V,
        phi_q=phi_q, phi_k=phi_k,
        phi_q_prime=phi_q_prime, phi_k_prime=phi_k_prime,
        S=S, u=u, denom=denom, z=z, eps_guard=eps_guard,
    )
