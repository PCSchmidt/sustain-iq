"""Post-training integer quantisation of an MLP, and the integer inference that mirrors it.

Scheme (the one the Rust ``tinyinfer`` crate implements, bit for bit):

* inputs: symmetric int8, one scale for the whole input vector
* weights: symmetric per-tensor integers in [-qmax, qmax] (qmax 127 for int8, 7 for int4),
  stored one per int8 here (int4 is packed only in the size accounting)
* biases: int32 at scale ``s_weight * s_input``
* accumulators: int64 here, int32 on the device (a layer's sum never exceeds 2^31 for these sizes)
* requantisation of hidden layers: ``y = (acc * m0 + (1 << (shift-1))) >> shift``, then clamp to
  [0, 127] (that clamp is the ReLU). ``m0`` is a 31-bit integer and ``shift`` a right shift,
  chosen so ``m0 / 2^shift`` approximates ``s_weight * s_input / s_output``
* output layer: the int32 accumulator is the logit; the class is its argmax

Only integer operations are used in inference, so any correct implementation gives
identical results; that is what the cross-language test checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mlp import MLP


def fixed_point(m: float) -> tuple[int, int]:
    """(m0, shift) with ``m0`` in [2^30, 2^31) and ``m0 / 2^shift`` approximating ``m``."""
    if not m > 0.0:
        raise ValueError("multiplier must be positive")
    shift = 30 - int(np.floor(np.log2(m)))
    m0 = int(round(m * 2.0**shift))
    if m0 == 1 << 31:  # rounding carried into the next power of two
        m0 >>= 1
        shift -= 1
    if shift < 1:
        raise ValueError("multiplier too large for this fixed-point form")
    return m0, shift


@dataclass
class QLayer:
    w: np.ndarray  # (out, in) int8
    b: np.ndarray  # (out,) int32
    m0: int  # 0 for the output layer
    shift: int


@dataclass
class QModel:
    input_scale: float
    layers: list[QLayer]
    weight_bits: int

    def quantize_input(self, X_std: np.ndarray) -> np.ndarray:
        return np.clip(np.round(X_std / self.input_scale), -127, 127).astype(np.int8)

    def logits_from_q(self, xq: np.ndarray) -> np.ndarray:
        h = xq.astype(np.int64)
        for i, layer in enumerate(self.layers):
            acc = h @ layer.w.T.astype(np.int64) + layer.b.astype(np.int64)
            if i < len(self.layers) - 1:
                h = np.clip((acc * layer.m0 + (1 << (layer.shift - 1))) >> layer.shift, 0, 127)
            else:
                return acc
        raise AssertionError("model has no layers")

    def logits(self, X_std: np.ndarray) -> np.ndarray:
        return self.logits_from_q(self.quantize_input(X_std))

    def predict(self, X_std: np.ndarray) -> np.ndarray:
        return self.logits(X_std).argmax(axis=1)


def quantize(mlp: MLP, calib_std: np.ndarray, weight_bits: int = 8) -> QModel:
    """Quantise ``mlp``. ``calib_std`` is standardised calibration data (the training set)."""
    qmax = 2 ** (weight_bits - 1) - 1
    _, acts = mlp.forward(calib_std, keep=True)
    s_in = float(np.abs(calib_std).max()) / 127.0
    input_scale = s_in
    layers: list[QLayer] = []
    last = len(mlp.W) - 1
    for i, (W, b) in enumerate(zip(mlp.W, mlp.b)):
        s_w = float(np.abs(W).max()) / qmax
        wq = np.clip(np.round(W / s_w), -qmax, qmax).astype(np.int8)
        bq = np.round(b / (s_w * s_in))
        if np.abs(bq).max() >= 2**31:
            raise ValueError("bias does not fit in int32; recalibrate")
        if i < last:
            s_out = max(float(acts[i + 1].max()), 1e-9) / 127.0
            m0, shift = fixed_point(s_w * s_in / s_out)
            s_in = s_out
        else:
            m0, shift = 0, 0
        layers.append(QLayer(wq, bq.astype(np.int32), m0, shift))
    return QModel(input_scale, layers, weight_bits)


def footprint(q: QModel) -> dict:
    """Parameters, multiply-accumulates, and flash/RAM bytes an on-device copy would need.

    Flash: packed weights + 4-byte biases + 8 bytes (m0, shift) per hidden layer.
    RAM: an int8 input buffer plus two int8 activation buffers sized to the widest layer.
    These are accounting figures, not measurements.
    """
    n_w = sum(l.w.size for l in q.layers)
    n_b = sum(l.b.size for l in q.layers)
    weight_bytes = n_w if q.weight_bits == 8 else (n_w * q.weight_bits + 7) // 8
    widths = [q.layers[0].w.shape[1]] + [l.w.shape[0] for l in q.layers]
    return {
        "params": n_w + n_b,
        "macs": int(n_w),
        "flash_bytes": int(weight_bytes + 4 * n_b + 8 * (len(q.layers) - 1)),
        "ram_bytes": int(widths[0] + 2 * max(widths[1:])),
    }
