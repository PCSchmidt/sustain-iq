"""A small multilayer perceptron in plain numpy: ReLU hidden layers, softmax cross-entropy, Adam.

Written out rather than imported so the layer arithmetic is exactly what the
quantiser and the Rust inference engine reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        return cls(X.mean(axis=0), X.std(axis=0) + 1e-9)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [rng.normal(0.0, np.sqrt(2.0 / sizes[i]), size=(sizes[i + 1], sizes[i])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    # ---------------------------------------------------------------- inference
    def forward(self, X: np.ndarray, keep: bool = False):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            acts.append(h)
        return (h, acts) if keep else h

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)

    def n_params(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    # ----------------------------------------------------------------- training
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 60,
        lr: float = 3e-3,
        batch: int = 128,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> "MLP":
        """Adam on class-balanced softmax cross-entropy. Returns self."""
        rng = np.random.default_rng(seed)
        n_classes = self.sizes[-1]
        counts = np.bincount(y, minlength=n_classes).astype(float)
        class_w = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                idx = order[s : s + batch]
                xb, yb = X[idx], y[idx]
                logits, acts = self.forward(xb, keep=True)
                p = np.exp(logits - logits.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                grad = p
                grad[np.arange(len(yb)), yb] -= 1.0
                grad *= (class_w[yb] / len(yb))[:, None]
                gW, gb = [], []
                for i in range(len(self.W) - 1, -1, -1):
                    gW.insert(0, grad.T @ acts[i] + weight_decay * self.W[i])
                    gb.insert(0, grad.sum(axis=0))
                    if i > 0:
                        grad = (grad @ self.W[i]) * (acts[i] > 0)
                t += 1
                for k, (p_, g) in enumerate(zip(params, gW + gb)):
                    m[k] = 0.9 * m[k] + 0.1 * g
                    v[k] = 0.999 * v[k] + 0.001 * g * g
                    mh = m[k] / (1 - 0.9**t)
                    vh = v[k] / (1 - 0.999**t)
                    p_ -= lr * mh / (np.sqrt(vh) + 1e-8)
        return self
