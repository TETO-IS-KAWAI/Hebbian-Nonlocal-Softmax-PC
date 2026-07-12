# Extension 4 — is PC ever *better*? Dropping weight transport

**Question.** Backprop's backward pass multiplies errors by the exact transpose of
the forward weights. That "weight transport" is the classic biological objection:
a neuron would have to know its downstream synapses exactly. Our PC so far used the
same exact transpose for feedback. Can PC drop it?

**Setup.** Replace the feedback transpose with fixed **random** matrices (feedback
alignment) and train a 2-layer causal char-LM three ways: backprop, PC with exact
feedback, PC with random feedback. Backprop *cannot* run this experiment — its
gradient is defined by the transpose. Code:
[`run.py`](run.py); the random-feedback option is `feedback_mats=` in
[`../../pc_attention/deep_char_lm.py`](../../pc_attention/deep_char_lm.py).

Why it can work: with random feedback the resulting gradient is no longer equal to
backprop, but it is still positively aligned with it (measured cosine: +1.0 at the
top layer, decaying to ~+0.35 at the bottom), and feedback alignment says the
forward weights adapt over training to make that random feedback increasingly
useful.

**Result.**

![feedback alignment](../../results/ext4_feedback_alignment.png)

```
final cross-entropy loss (2-layer causal char-LM):
   backprop (weight transport)     0.0242
   PC, exact transpose feedback    0.0242   (identical to backprop)
   PC, random feedback             0.0572   (no weight transport)
```

Random-feedback PC **learns the LM** (loss 4.0 → 0.057), tracking backprop closely
and lagging only modestly, while exact-transpose PC is indistinguishable from
backprop. So PC-attention trains **without weight transport** — the forward weights
align to fixed random feedback over training. Backprop cannot do this at all; its
gradient is the transpose by definition.

**Honest read.** This is not "PC beats backprop on loss" — exact PC ties backprop,
random feedback costs a little. The advantage is structural: PC drops backprop's
biologically-implausible weight-transport requirement and still trains attention.
It's a *capability* backprop lacks, not a lower number. (Random feedback is
Lillicrap et al.'s feedback alignment; the contribution here is that it carries
through PC on attention, not just plain MLPs.)
