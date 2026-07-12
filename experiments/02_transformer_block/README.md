# Extension 2 — a real transformer block

**Question.** Everything so far was a bare attention layer. Does PC scale to a real
pre-norm transformer block: multi-head causal attention + feed-forward MLP +
**LayerNorm** + residual connections?

**The idea.** Put the PC error nodes on the **residual stream**, between whole
sub-layers, not inside them. Each sub-layer `a^k = a^{k-1} + sublayer_k(a^{k-1})`
is then a self-contained map, and fixed-prediction PC needs only two local
operations per sub-layer — its weight gradient given the output error, and its
input transpose for feedback — both from autograd on that single sub-layer.
LayerNorm, the MLP nonlinearity, and the multiple heads all live *inside* a
sub-layer, so they need no bespoke PC rule; the global-over-features LayerNorm is
absorbed because the error node sits after the sub-layer. Code:
[`../../pc_attention/transformer_block.py`](../../pc_attention/transformer_block.py),
[`run.py`](run.py).

**Result.**

![transformer block](../../results/ext2_transformer.png)

```
gate (worst |g_PC - g_BP| over ALL params, incl. LayerNorm γ/β and FFN):
   T=1  1.1e-1     T=8  2.8e-2     T=32  1.8e-5     T=200  3.2e-13
training (1 block, d=64, 4 heads, d_ff=128):
   backprop  final loss 0.0410
   PC T=32   final loss 0.0410     max curve gap 1.4e-9
```

At converged inference the PC gradient equals backprop across **every** parameter
of the block, LayerNorm and FFN included (worst 3e-13), and PC training tracks
backprop exactly. So the fixed-prediction residual-stream formulation carries the
whole block unchanged — LayerNorm is not a special case, it is just another
within-sub-layer op with the error node on the stream after it.
