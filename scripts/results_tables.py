"""Print the phase 1 result tables (markdown) from results/phase1_*.json.

    python scripts/results_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def ms(x, digits=3):
    return f"{x[0]:.{digits}f} ± {x[1]:.{digits}f}"


def leak() -> str:
    d = load("phase1_leak.json")
    lines = ["| How the held-out data is chosen | MLP (float) | MLP (int8) | Logistic regression |", "| --- | --- | --- | --- |"]
    for r in d["rows"]:
        i8 = ms(r["mlp_int8"]) if "mlp_int8" in r else "n/a"
        lines.append(f"| {r['protocol']} | {ms(r['mlp_float'])} | {i8} | {ms(r['logreg'])} |")
    return "\n".join(lines)


def confusion() -> str:
    d = load("phase1_leak.json")
    out = []
    for proto, c in d["confusion"].items():
        lines = [f"Confusion matrix, {proto}, summed over folds (rows true, columns predicted):", "", "| | " + " | ".join(d["classes"]) + " | recall |", "| --- | " + " | ".join(["---"] * (len(d["classes"]) + 1)) + " |"]
        for i, k in enumerate(d["classes"]):
            tot = sum(c[i])
            lines.append(f"| {k} | " + " | ".join(str(v) for v in c[i]) + f" | {c[i][i] / tot:.2f} |")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def features() -> str:
    d = load("phase1_features.json")
    lines = ["| Feature set | Unseen load: MLP / int8 / logistic | Unseen fault size: MLP / int8 / logistic |", "| --- | --- | --- |"]
    names = {"abs": "Absolute log band power", "shape": "Band shape (amplitude removed)", "envelope": "Envelope spectrum"}
    for k, v in d.items():
        a, b = v["leave-one-load-out"], v["leave-one-size-out"]
        lines.append(
            f"| {names[k]} | {a['float'][0]:.3f} / {a['int8'][0]:.3f} / {a['logreg'][0]:.3f} | {b['float'][0]:.3f} / {b['int8'][0]:.3f} / {b['logreg'][0]:.3f} |"
        )
    return "\n".join(lines)


def lock() -> str:
    d = load("phase1_lock.json")
    groups = sorted(next(iter(d["abs"].values()))["recall_by_group"])
    lines = ["| Feature set | Model | Balanced accuracy | Fault detected | " + " | ".join(f"Recall {g}" for g in groups) + " |", "| --- | --- | --- | --- | " + " | ".join(["---"] * len(groups)) + " |"]
    names = {"abs": "Absolute", "shape": "Shape", "envelope": "Envelope"}
    mn = {"mlp_float": "MLP", "mlp_int8": "MLP int8", "logreg": "Logistic"}
    for feat, rows in d.items():
        if feat == "n_locked_windows":
            continue
        for m, r in rows.items():
            lines.append(
                f"| {names[feat]} | {mn[m]} | {r['balanced_accuracy']:.3f} | {r['fault_detected']:.3f} | "
                + " | ".join(f"{r['recall_by_group'][g]:.2f}" for g in groups)
                + " |"
            )
    return "\n".join(lines)


def ladder() -> str:
    d = load("phase1_ladder.json")["rows"]
    lines = [
        "| Hidden layers | Parameters | Flash int8 / int4 (bytes) | RAM (bytes) | MACs | Diagnosis, unseen load: float / int8 / int4 | Detection, unseen load: float / int8 / int4 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in d:
        lo = "leave-one-load-out"
        lines.append(
            f"| {r['hidden'][0]}, {r['hidden'][1]} | {r['params']} | {r['int8']['flash_bytes']} / {r['int4']['flash_bytes']} | {r['int8']['ram_bytes']} | {r['int8']['macs']} | "
            f"{r[lo + ':float'][0]:.3f} / {r[lo + ':int8'][0]:.3f} / {r[lo + ':int4'][0]:.3f} | "
            f"{r[lo + ':detect_float'][0]:.3f} / {r[lo + ':detect_int8'][0]:.3f} / {r[lo + ':detect_int4'][0]:.3f} |"
        )
    return "\n".join(lines)


def bandwidth() -> str:
    d = load("phase1_bandwidth.json")["rows"]
    lines = [
        "| Sample rate (Hz) | Nyquist (Hz) | Window (ms) | Diagnosis, unseen load (MLP) | Detection, unseen load: MLP | Detection, unseen load: RMS threshold | Detection, unseen size: MLP | Detection, unseen size: RMS threshold |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(d, key=lambda r: -r["sample_rate_hz"]):
        lo, so = "leave-one-load-out", "leave-one-size-out"
        lines.append(
            f"| {r['sample_rate_hz']:.0f} | {r['nyquist_hz']:.0f} | {r['window_s'] * 1000:.0f} | {r[lo + ':float'][0]:.3f} | {r[lo + ':detect_float'][0]:.3f} | "
            f"{r[lo + ':detect_rms_threshold'][0]:.3f} | {r[so + ':detect_float'][0]:.3f} | {r[so + ':detect_rms_threshold'][0]:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    for title, fn in (("LEAK", leak), ("CONFUSION", confusion), ("FEATURES", features), ("LOCK", lock), ("LADDER", ladder), ("BANDWIDTH", bandwidth)):
        print(f"## {title}\n")
        print(fn())
        print()
