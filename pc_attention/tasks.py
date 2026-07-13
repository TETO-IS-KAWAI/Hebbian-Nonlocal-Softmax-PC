"""Associative recall: the toy problem the layer is trained on.

Each example is a list of key-value pairs followed by one query key, and the answer
is the value that was paired with that key. You can't solve it from position alone,
so it genuinely needs attention (a plain MLP can't do it).

How a pair is handed to a single attention layer: bundle a key and its value into
one token by adding their embeddings.

    memory token :  x = KeyEmb[key] + ValEmb[value]
    query  token :  x = KeyEmb[query_key] + QueryFlag

Now W_K can read the key part, W_V the value part, and W_Q the query's key; attention
lines the query up with the matching memory token and copies out its value. The
training target at the query position is ValEmb[answer], and we read the model's
answer back by finding the value embedding nearest to its output.

Two choices worth knowing about:
  * "vocab = 16" is the number of possible VALUES (the 16 answer classes). Keys are
    drawn distinct from a much larger pool (512 by default) so every query has one
    unambiguous answer and N can go up to 256 -- you simply cannot have 256 distinct
    keys out of an alphabet of 16. This is the usual associative-recall setup.
  * Because the target is a value embedding and we score by nearest neighbour, the
    task fits inside the same squared-error energy the PC layer already minimises.
    No separate classification head, which keeps the PC-vs-backprop comparison clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

Tensor = torch.Tensor


@dataclass
class RecallBatch:
    X: Tensor          # (B, N+1, d)   input tokens (N memory pairs, then the query)
    x_out: Tensor      # (B, N+1, d)   target, nonzero only at the query position
    mask: Tensor       # (B, N+1)      1 at the query position, else 0
    labels: Tensor     # (B,)          correct value index for each query
    query_pos: int     # index of the query token (= N, the last position)


class RecallTask:
    def __init__(
        self,
        d: int = 16,
        n_values: int = 16,
        n_keys: int = 512,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        self.d = d
        self.n_values = n_values
        self.n_keys = n_keys
        self.dtype = dtype
        self.device = device
        g = torch.Generator(device=device).manual_seed(seed)

        # Embeddings are random unit vectors, fixed for the life of the task.
        def unit(rows: int) -> Tensor:
            e = torch.randn(rows, d, generator=g, dtype=dtype, device=device)
            return e / e.norm(dim=-1, keepdim=True)

        self.KeyEmb = unit(n_keys)          # (n_keys, d)
        self.ValEmb = unit(n_values)        # (n_values, d)
        self.QueryFlag = unit(1)[0]         # (d,)  marks the query token

    def generate(self, batch: int, N: int, seed: Optional[int] = None) -> RecallBatch:
        if N > self.n_keys:
            raise ValueError(f"N={N} exceeds key alphabet {self.n_keys}")
        g = torch.Generator(device=self.device)
        g.manual_seed(seed if seed is not None else torch.seed())

        T = N + 1
        X = torch.zeros(batch, T, self.d, dtype=self.dtype, device=self.device)
        x_out = torch.zeros_like(X)
        mask = torch.zeros(batch, T, dtype=self.dtype, device=self.device)
        labels = torch.zeros(batch, dtype=torch.long, device=self.device)

        # Build each sequence in the batch independently.
        for b in range(batch):
            keys = torch.randperm(self.n_keys, generator=g, device=self.device)[:N]  # distinct
            vals = torch.randint(self.n_values, (N,), generator=g, device=self.device)  # may repeat
            X[b, :N] = self.KeyEmb[keys] + self.ValEmb[vals]

            qi = torch.randint(N, (1,), generator=g, device=self.device).item()  # which key to ask
            X[b, N] = self.KeyEmb[keys[qi]] + self.QueryFlag
            x_out[b, N] = self.ValEmb[vals[qi]]
            mask[b, N] = 1.0
            labels[b] = vals[qi]

        return RecallBatch(X=X, x_out=x_out, mask=mask, labels=labels, query_pos=N)

    def decode(self, z_query: Tensor) -> Tensor:
        """Turn an output vector into a value guess: the nearest value embedding.
        (B, d) -> predicted value indices (B,)."""
        d2 = torch.cdist(z_query, self.ValEmb)          # distance to each value, (B, n_values)
        return d2.argmin(dim=-1)

    def accuracy(self, z: Tensor, batch: RecallBatch) -> float:
        """Fraction of queries whose decoded value is correct."""
        pred = self.decode(z[:, batch.query_pos])
        return (pred == batch.labels).double().mean().item()
