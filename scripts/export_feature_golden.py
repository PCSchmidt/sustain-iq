"""Export raw windows and the Python features, int8 inputs and classes the Rust extractor must reproduce.

    python scripts/export_feature_golden.py

Writes firmware/tinyfeat/tests/features_golden.json (or --out): 12 real CWRU windows per class (N, IR, B, OR),
plus a few synthetic edge windows (all zeros, a constant offset, a very large amplitude, a tiny amplitude),
with, for each window, the float64 features `band_features` gives, the standardised features, the int8 input
and the class predicted by the deployed int8 model. The model is the one in tinyinfer's golden.json: this
script retrains it exactly as export_golden.py does and refuses to continue if its logits differ, so the
two golden files describe the same network. A larger file for a local check can be made with
`--per-class 500 --out <path>` (about 30 MB, not committed; point TINYFEAT_GOLDEN at it). These are training windows: the file tests numerical
agreement of an implementation, not generalisation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sustain_iq.data import CLASSES, build_dataset, load_recordings  # noqa: E402
from sustain_iq.export import golden_dict  # noqa: E402
from sustain_iq.mlp import MLP, Standardizer  # noqa: E402
from sustain_iq.quant import quantize  # noqa: E402
from sustain_iq.signals import STRIDE, WINDOW, band_features  # noqa: E402

SIZES = [32, 32, 16, 4]
PER_CLASS = 12


def f32_text(v: np.float32) -> str:
    return np.format_float_positional(v, unique=True, trim="-")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=PER_CLASS, help="real windows per class (default: the small committed file)")
    ap.add_argument("--out", type=Path, default=ROOT / "firmware" / "tinyfeat" / "tests" / "features_golden.json")
    args = ap.parse_args()
    recs = load_recordings(ROOT / "data" / "cwru")
    ds = build_dataset(recs)
    std = Standardizer.fit(ds.X)
    Xs = std(ds.X)
    mlp = MLP(SIZES, seed=0).fit(Xs, ds.y, seed=0)
    q = quantize(mlp, Xs, weight_bits=8)

    golden_path = ROOT / "firmware" / "tinyinfer" / "tests" / "golden.json"
    old = json.loads(golden_path.read_text(encoding="utf-8"))
    new = golden_dict(q, std, Xs)
    if new["logits"] != old["logits"] or new["layers"] != old["layers"]:
        raise SystemExit("retrained model differs from tinyinfer/tests/golden.json; rerun scripts/export_golden.py first")

    rng = np.random.default_rng(1)
    windows, label, source = [], [], []
    for c, name in enumerate(CLASSES):
        for i in rng.choice(np.flatnonzero(ds.y == c), size=args.per_class, replace=False):
            r = recs[ds.rec[i]]
            start = int(ds.start[i])
            windows.append(r.signal[start : start + WINDOW].astype(np.float32))
            label.append(c)
            source.append(f"{r.label}@{start}")
    base = windows[0]
    edge = {
        "zeros": np.zeros(WINDOW, np.float32),
        "constant_offset": np.full(WINDOW, 0.37, np.float32),
        "large_amplitude": (base * 40).astype(np.float32),
        "tiny_amplitude": (base * 1e-3).astype(np.float32),
    }
    for name, w in edge.items():
        windows.append(w)
        label.append(-1)
        source.append(f"edge:{name}")

    W = np.stack(windows)
    feats = band_features(W)
    xs = std(feats)
    xq = q.quantize_input(xs)
    pred = q.predict(xs)
    out = {
        "window": WINDOW,
        "stride": STRIDE,
        "source": source,
        "label": label,
        "windows": [[f32_text(v) for v in w] for w in W],
        "features": feats.tolist(),
        "x_std": xs.tolist(),
        "xq": xq.astype(int).tolist(),
        "pred": pred.astype(int).tolist(),
    }
    dest = args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    # the shortest round-trip decimal of each float32, written as a JSON number (not a string) for the Rust loader
    dest.write_text(json.dumps({**out, "windows": [[float(s) for s in row] for row in out["windows"]]}), encoding="utf-8")
    n_real = len(CLASSES) * args.per_class
    acc = float((pred[:n_real] == np.array(label[:n_real])).mean())
    print(f"wrote {dest} ({dest.stat().st_size} bytes); {len(W)} windows; deployed-model accuracy on the real ones {acc:.3f} (training windows, not an evaluation)")


if __name__ == "__main__":
    main()
