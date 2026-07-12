"""pc_attention — Predictive Coding on linear self-attention (Path B / M1)."""

from .linear_attn import (
    FeatureMap,
    FEATURE_MAPS,
    ForwardState,
    linear_attention_forward,
    predict,
)

__all__ = [
    "FeatureMap",
    "FEATURE_MAPS",
    "ForwardState",
    "linear_attention_forward",
    "predict",
]
