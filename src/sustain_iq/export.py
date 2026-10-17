"""Export a quantised model and test vectors for the Rust inference engine.

The golden file holds the model's integers (weights, biases, requantisation
multipliers), standardisation constants, and a set of quantised inputs with the logits
and class the Python reference produces. The Rust test rebuilds the model from the
integers and must reproduce every logit exactly.
"""

from __future__ import annotations

import numpy as np

from .mlp import Standardizer
from .quant import QModel


def edge_case_inputs(n_features: int) -> np.ndarray:
    """Inputs that stress saturation: all +127, all -127, zeros, alternating extremes."""
    alt = np.where(np.arange(n_features) % 2 == 0, 127, -127)
    return np.stack(
        [np.full(n_features, 127), np.full(n_features, -127), np.zeros(n_features), alt, -alt]
    ).astype(np.int8)


def golden_dict(q: QModel, std: Standardizer, X_std: np.ndarray, n_samples: int = 300, seed: int = 0) -> dict:
    """Model integers plus ``n_samples`` real inputs and the edge cases, with expected outputs."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(X_std), size=min(n_samples, len(X_std)), replace=False)
    xq = np.concatenate([q.quantize_input(X_std[pick]), edge_case_inputs(X_std.shape[1])])
    logits = q.logits_from_q(xq)
    return {
        "weight_bits": q.weight_bits,
        "input_scale": q.input_scale,
        "standardizer": {"mean": std.mean.tolist(), "std": std.std.tolist()},
        "layers": [
            {
                "in": int(l.w.shape[1]),
                "out": int(l.w.shape[0]),
                "w": l.w.astype(int).ravel().tolist(),  # row-major: out x in
                "b": l.b.astype(int).tolist(),
                "m0": int(l.m0),
                "shift": int(l.shift),
            }
            for l in q.layers
        ],
        "inputs": xq.astype(int).tolist(),
        "logits": logits.astype(int).tolist(),
        "pred": logits.argmax(axis=1).astype(int).tolist(),
    }
