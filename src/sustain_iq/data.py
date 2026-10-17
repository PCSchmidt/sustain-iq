"""Loading CWRU recordings, building a windowed dataset, and leak-aware splits.

Why the splits matter: windows cut from one recording are near-duplicates of each
other. A random split of pooled windows puts neighbouring windows of the same
recording on both sides, so a classifier can score near-perfectly by recognising
the recording, not the fault. The honest protocols hold out whole recordings:
leave-one-load-out (same fault sizes, unseen motor load) and leave-one-size-out
(unseen fault diameter). Both are provided, along with the leaky split as a
deliberate contrast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .cwru_files import FILES, LOCKED_FILES
from .signals import STRIDE, WINDOW, band_features, decimate, envelope_band_features, highpass, shape_features, windows

FEATURE_SETS = ("abs", "shape", "envelope")  # absolute log bands; amplitude-free shape; envelope spectrum

CLASSES = ["N", "IR", "B", "OR"]  # normal, inner race, ball, outer race
CWRU_FS = 12_000  # Hz, 12k drive-end data


@dataclass(frozen=True)
class Recording:
    label: str
    fault: str
    diameter: float  # inches; 0 for normal
    load: int  # hp
    signal: np.ndarray  # drive-end acceleration, float32
    note: str = ""  # for locked recordings: why they are out of distribution


@dataclass
class Dataset:
    X: np.ndarray  # (n, n_bands) log band power
    y: np.ndarray  # (n,) class index
    rec: np.ndarray  # (n,) recording index
    load: np.ndarray  # (n,) motor load hp
    diameter: np.ndarray  # (n,) fault diameter in inches (0 = normal)
    start: np.ndarray  # (n,) window start sample within its (possibly decimated) recording
    rec_len: np.ndarray  # (n,) length in samples of the window's recording

    def subset(self, idx: np.ndarray) -> "Dataset":
        return Dataset(*(getattr(self, f)[idx] for f in ("X", "y", "rec", "load", "diameter", "start", "rec_len")))


def load_de(path: Path, file_no: int | None = None) -> np.ndarray:
    """The drive-end time series from a CWRU .mat file.

    Each file normally holds one ``X<nnn>_DE_time`` variable. ``Normal_2.mat`` (file 99)
    also carries an exact copy of file 98's recording, so when ``file_no`` is given the
    variable named for that file is chosen, never just the first match: taking the first
    would silently duplicate one load's data as another's.
    """
    mat = loadmat(str(path))
    keys = sorted(k for k in mat if k.endswith("_DE_time"))
    if file_no is not None and f"X{file_no:03d}_DE_time" in keys:
        return np.asarray(mat[f"X{file_no:03d}_DE_time"], dtype=np.float32).ravel()
    if len(keys) != 1:
        raise ValueError(f"cannot pick a drive-end variable in {path.name}: found {keys}, file_no={file_no}")
    return np.asarray(mat[keys[0]], dtype=np.float32).ravel()


def load_recordings(data_dir: Path, locked: bool = False) -> list[Recording]:
    """The 40 training/evaluation recordings, or (``locked=True``) the 24 locked ones."""
    recs = []
    for fault, diameter, load, n, label, *note in LOCKED_FILES if locked else FILES:
        recs.append(Recording(label, fault, diameter, load, load_de(data_dir / f"{label}.mat", n), note[0] if note else ""))
    return recs


def build_dataset(
    recs: list[Recording],
    factor: int = 1,
    win: int = WINDOW,
    stride: int = STRIDE,
    n_bands: int = 32,
    features: str = "abs",
) -> Dataset:
    """Windows and features for every recording; ``factor`` decimates first (a slower sensor).

    ``features``: ``abs`` log band power, ``shape`` the same minus each window's mean
    (amplitude-free), ``envelope`` the envelope spectrum of the high-passed signal
    (needs the full 12 kHz rate, so ``factor`` must be 1).
    """
    if features not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {features!r}")
    if features == "envelope" and factor != 1:
        raise ValueError("envelope features need the full sample rate")
    parts = {k: [] for k in ("X", "y", "rec", "load", "diameter", "start", "rec_len")}
    for i, r in enumerate(recs):
        x = decimate(r.signal, factor)
        if features == "envelope":
            x = highpass(x, CWRU_FS / factor)
        w, starts = windows(x, win, stride)
        if len(w) == 0:
            continue
        n = len(w)
        if features == "abs":
            parts["X"].append(band_features(w, n_bands))
        elif features == "shape":
            parts["X"].append(shape_features(w, n_bands))
        else:
            parts["X"].append(envelope_band_features(w, n_bands))
        parts["y"].append(np.full(n, CLASSES.index(r.fault)))
        parts["rec"].append(np.full(n, i))
        parts["load"].append(np.full(n, r.load))
        parts["diameter"].append(np.full(n, r.diameter))
        parts["start"].append(starts)
        parts["rec_len"].append(np.full(n, len(x)))
    return Dataset(**{k: np.concatenate(v) for k, v in parts.items()})


# ----------------------------------------------------------------------------- splits
def _normal_time_split(ds: Dataset, test_frac: float = 0.25, gap: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """For the Normal recordings (one per load, no fault size): first part train, last part test, with a gap."""
    is_n = ds.y == CLASSES.index("N")
    cut = (ds.rec_len * (1.0 - test_frac)).astype(np.int64)
    train = is_n & (ds.start + WINDOW <= cut - gap)
    test = is_n & (ds.start >= cut)
    return np.flatnonzero(train), np.flatnonzero(test)


def split_random(ds: Dataset, test_frac: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """LEAKY: random windows pooled across all recordings. For contrast only."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds.y))
    n_test = int(len(perm) * test_frac)
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def split_leave_one_load_out(ds: Dataset, load: int) -> tuple[np.ndarray, np.ndarray]:
    """Train on three motor loads, test on the fourth (whole recordings held out)."""
    return np.flatnonzero(ds.load != load), np.flatnonzero(ds.load == load)


def split_leave_one_size_out(ds: Dataset, diameter: float) -> tuple[np.ndarray, np.ndarray]:
    """Train on two fault diameters, test on the third. Normal recordings are split by time."""
    fault = ds.diameter > 0
    train = np.flatnonzero(fault & (ds.diameter != diameter))
    test = np.flatnonzero(fault & (ds.diameter == diameter))
    n_train, n_test = _normal_time_split(ds)
    return np.concatenate([train, n_train]), np.concatenate([test, n_test])


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = len(CLASSES)) -> float:
    """Mean per-class recall over the classes present in ``y_true``."""
    recalls = [np.mean(y_pred[y_true == c] == c) for c in range(n_classes) if np.any(y_true == c)]
    return float(np.mean(recalls))
