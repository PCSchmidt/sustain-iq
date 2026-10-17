"""Train the deployment model on all 40 CWRU recordings and export golden vectors for Rust.

    python scripts/export_golden.py

Writes firmware/tinyinfer/tests/golden.json. The model is the 32-32-16-4 int8 MLP on
absolute log band features. It is trained on every training recording, so it is the
model a device would run, not one used for evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sustain_iq.data import build_dataset, load_recordings  # noqa: E402
from sustain_iq.export import golden_dict  # noqa: E402
from sustain_iq.mlp import MLP, Standardizer  # noqa: E402
from sustain_iq.quant import quantize  # noqa: E402

SIZES = [32, 32, 16, 4]


def main() -> None:
    ds = build_dataset(load_recordings(ROOT / "data" / "cwru"))
    std = Standardizer.fit(ds.X)
    Xs = std(ds.X)
    mlp = MLP(SIZES, seed=0).fit(Xs, ds.y, seed=0)
    q = quantize(mlp, Xs, weight_bits=8)
    out = golden_dict(q, std, Xs)
    dest = ROOT / "firmware" / "tinyinfer" / "tests" / "golden.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out), encoding="utf-8")
    acc = float((q.predict(Xs) == ds.y).mean())
    print(f"wrote {dest} ({dest.stat().st_size} bytes); {len(out['inputs'])} vectors; training-set accuracy {acc:.3f} (not an evaluation)")


if __name__ == "__main__":
    main()
