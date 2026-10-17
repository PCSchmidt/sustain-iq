"""Windowing, decimation and log band-energy features for vibration signals.

The feature extractor is deliberately small enough to run on a microcontroller:
one Hann-windowed real FFT of a fixed 1024-sample window, power summed into 32
bands that are log-spaced in *bin index*, then log10. Because the bands are defined
in bin space, decimating the signal (to model a slower, cheaper sensor) changes the
absolute frequency of every band but leaves the compute cost and feature count the
same.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

WINDOW = 1024
STRIDE = 512
N_BANDS = 32


def decimate(x: np.ndarray, factor: int) -> np.ndarray:
    """Low-pass and keep every ``factor``-th sample (zero-phase FIR)."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return x
    return sps.decimate(x, factor, ftype="fir", zero_phase=True).astype(x.dtype)


def windows(x: np.ndarray, size: int = WINDOW, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping windows as a (n, size) view-free array, and each window's start sample."""
    if len(x) < size:
        return np.empty((0, size), dtype=x.dtype), np.empty(0, dtype=np.int64)
    starts = np.arange(0, len(x) - size + 1, stride, dtype=np.int64)
    idx = starts[:, None] + np.arange(size)[None, :]
    return x[idx], starts


def band_edges(n_bins: int, n_bands: int = N_BANDS) -> np.ndarray:
    """Strictly increasing bin-index edges, log-spaced, from bin 1 (DC excluded) to ``n_bins``.

    Early bands are one bin wide (log spacing is finer than the FFT resolution
    there), so the edges are forced to advance by at least one bin.
    """
    geom = np.geomspace(1, n_bins, n_bands + 1)
    edges = [1]
    for g in geom[1:-1]:
        edges.append(max(int(round(g)), edges[-1] + 1))
    edges.append(n_bins)
    edges = np.array(edges)
    if np.any(np.diff(edges) < 1):
        raise ValueError("too many bands for this number of bins")
    return edges


def band_features(win: np.ndarray, n_bands: int = N_BANDS) -> np.ndarray:
    """log10 band power of each window (n, n_bands). Removes each window's mean first."""
    n = win.shape[1]
    x = win - win.mean(axis=1, keepdims=True)
    x = x * np.hanning(n)[None, :]
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2  # (n, n//2 + 1)
    edges = band_edges(power.shape[1], n_bands)
    out = np.empty((win.shape[0], n_bands), dtype=np.float64)
    for b in range(n_bands):
        out[:, b] = power[:, edges[b] : edges[b + 1]].sum(axis=1)
    return np.log10(out + 1e-12)


def fft_butterflies(n: int = WINDOW) -> int:
    """Complex butterflies in a radix-2 FFT of size ``n`` (a cost estimate, not a measurement)."""
    return (n // 2) * int(np.log2(n))


def shape_features(win: np.ndarray, n_bands: int = N_BANDS) -> np.ndarray:
    """Band features with each window's mean log power removed: spectral *shape* only.

    Multiplying a window by a constant adds the same amount to every log band, so this
    is exactly invariant to overall amplitude (and costs one subtraction per band).
    """
    f = band_features(win, n_bands)
    return f - f.mean(axis=1, keepdims=True)


def highpass(x: np.ndarray, fs: float, cutoff_hz: float = 1000.0, order: int = 4) -> np.ndarray:
    """Causal Butterworth high-pass (the form a microcontroller would run as biquads)."""
    sos = sps.butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return sps.sosfilt(sos, x).astype(x.dtype)


def envelope_band_features(win_hp: np.ndarray, n_bands: int = N_BANDS, lo_bin: int = 2, hi_bin: int = 66) -> np.ndarray:
    """Envelope-spectrum features of already high-passed windows.

    The classic bearing-diagnosis route: a defect excites a structural resonance once per
    impact, so the *envelope* (here: rectified, mean removed) of the high-passed signal
    carries the defect's repetition frequency. Bins ``lo_bin..hi_bin`` (about 23-770 Hz at
    12 kHz and 1024 samples) are summed in equal-width groups, log10'd, and the window's
    mean removed (shape only).
    """
    env = np.abs(win_hp)
    env = env - env.mean(axis=1, keepdims=True)
    env = env * np.hanning(env.shape[1])[None, :]
    power = np.abs(np.fft.rfft(env, axis=1)) ** 2
    width = (hi_bin - lo_bin) // n_bands
    if width < 1:
        raise ValueError("not enough bins for this many bands")
    out = np.empty((power.shape[0], n_bands))
    for b in range(n_bands):
        out[:, b] = power[:, lo_bin + b * width : lo_bin + (b + 1) * width].sum(axis=1)
    f = np.log10(out + 1e-12)
    return f - f.mean(axis=1, keepdims=True)
