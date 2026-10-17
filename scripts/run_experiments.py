"""Phase 1 experiments on the CWRU bearing data.

    python scripts/run_experiments.py leak        # how much does a leaky split inflate the score?
    python scripts/run_experiments.py ladder      # model size x weight precision under honest splits
    python scripts/run_experiments.py bandwidth   # what if the accelerometer is slower than the lab's?
    python scripts/run_experiments.py features    # three feature sets, same honest folds
    python scripts/run_experiments.py lock        # ONE evaluation of every feature set on the locked recordings
    python scripts/run_experiments.py all

Every protocol that is not named "random (leaky)" holds out whole recordings:
leave-one-load-out (4 folds) and leave-one-size-out (3 folds). Scores are balanced
accuracy on the held-out recordings, averaged over folds and 3 training seeds. All
features are standardised with statistics from the training windows only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sustain_iq.data import (  # noqa: E402
    CLASSES,
    CWRU_FS,
    FEATURE_SETS,
    balanced_accuracy,
    build_dataset,
    load_recordings,
    split_leave_one_load_out,
    split_leave_one_size_out,
    split_random,
)
from sustain_iq.mlp import MLP, Standardizer  # noqa: E402
from sustain_iq.quant import footprint, quantize  # noqa: E402

RESULTS = ROOT / "results"
SEEDS = (0, 1, 2)
LOADS = (0, 1, 2, 3)
SIZES = (0.007, 0.014, 0.021)
MAIN_SIZES = [32, 32, 16, len(CLASSES)]


def honest_folds(ds):
    for L in LOADS:
        yield "leave-one-load-out", f"load {L}", *split_leave_one_load_out(ds, L)
    for d in SIZES:
        yield "leave-one-size-out", f"size {d:.3f}", *split_leave_one_size_out(ds, d)


def evaluate(ds, tr, te, sizes, bits=(None, 8, 4), seeds=SEEDS, baseline=False):
    """Balanced accuracy per precision (None = float32), per seed, on one train/test split."""
    std = Standardizer.fit(ds.X[tr])
    Xs, Xt = std(ds.X[tr]), std(ds.X[te])
    ytr, yte = ds.y[tr], ds.y[te]
    out = {b: [] for b in bits}
    det = {b: [] for b in bits}  # healthy vs faulty, from the same model (predicted class != Normal)
    foot = {}
    preds = {}
    for seed in seeds:
        m = MLP(sizes, seed).fit(Xs, ytr, seed=seed)
        for b in bits:
            if b is None:
                p = m.predict(Xt)
            else:
                q = quantize(m, Xs, weight_bits=b)
                p = q.predict(Xt)
                foot[b] = footprint(q)
            out[b].append(balanced_accuracy(yte, p))
            det[b].append(balanced_accuracy((yte > 0).astype(int), (p > 0).astype(int), n_classes=2))
            if seed == seeds[0]:
                preds[b] = p
    res = {"acc": out, "det": det, "footprint": foot, "n_train": int(len(tr)), "n_test": int(len(te)), "preds0": preds, "y": yte}
    if baseline:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xs, ytr)
        res["logreg"] = balanced_accuracy(yte, lr.predict(Xt))
    return res


def mean_std(xs):
    return float(np.mean(xs)), float(np.std(xs))


def confusion(y, p):
    c = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for a, b in zip(y, p):
        c[a, b] += 1
    return c


# ---------------------------------------------------------------------------- studies
def study_leak(recs):
    ds = build_dataset(recs)
    rows = []
    # leaky: three random splits
    accs, lrs = [], []
    for s in SEEDS:
        tr, te = split_random(ds, 0.3, seed=s)
        r = evaluate(ds, tr, te, MAIN_SIZES, bits=(None,), seeds=(s,), baseline=True)
        accs += r["acc"][None]
        lrs.append(r["logreg"])
    rows.append({"protocol": "random windows (leaky)", "mlp_float": mean_std(accs), "logreg": mean_std(lrs), "folds": 3})
    # honest protocols
    by = {}
    conf = {}
    for proto, name, tr, te in honest_folds(ds):
        r = evaluate(ds, tr, te, MAIN_SIZES, bits=(None, 8), baseline=True)
        d = by.setdefault(proto, {"float": [], "int8": [], "logreg": []})
        d["float"].append(float(np.mean(r["acc"][None])))
        d["int8"].append(float(np.mean(r["acc"][8])))
        d["logreg"].append(r["logreg"])
        d.setdefault("folds", []).append({"fold": name, "float": float(np.mean(r["acc"][None])), "int8": float(np.mean(r["acc"][8])), "logreg": r["logreg"]})
        conf[proto] = conf.get(proto, 0) + confusion(r["y"], r["preds0"][None])
    for proto, d in by.items():
        rows.append({"protocol": proto, "mlp_float": mean_std(d["float"]), "mlp_int8": mean_std(d["int8"]), "logreg": mean_std(d["logreg"]), "folds": d["folds"]})
    out = {"model": MAIN_SIZES, "rows": rows, "confusion": {k: v.tolist() for k, v in conf.items()}, "classes": CLASSES}
    (RESULTS / "phase1_leak.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n== leakage vs honest protocols (balanced accuracy, mean +/- std over folds) ==")
    for r in rows:
        extra = f"  int8 {r['mlp_int8'][0]:.3f}" if "mlp_int8" in r else ""
        print(f"{r['protocol']:24s} MLP {r['mlp_float'][0]:.3f} +/- {r['mlp_float'][1]:.3f}{extra}   logistic regression {r['logreg'][0]:.3f}")


def study_ladder(recs):
    ds = build_dataset(recs)
    configs = [(4, 4), (8, 8), (16, 8), (32, 16), (64, 32)]
    rows = []
    for h1, h2 in configs:
        sizes = [32, h1, h2, len(CLASSES)]
        agg = {proto: {b: [] for b in (None, 8, 4)} for proto in ("leave-one-load-out", "leave-one-size-out")}
        dagg = {proto: {b: [] for b in (None, 8, 4)} for proto in agg}
        foot = {}
        for proto, name, tr, te in honest_folds(ds):
            r = evaluate(ds, tr, te, sizes)
            for b in (None, 8, 4):
                agg[proto][b].append(float(np.mean(r["acc"][b])))
                dagg[proto][b].append(float(np.mean(r["det"][b])))
            foot = r["footprint"]
        row = {"hidden": [h1, h2], "params": foot[8]["params"], "int8": foot[8], "int4": foot[4]}
        for proto in agg:
            for b, key in ((None, "float"), (8, "int8"), (4, "int4")):
                row[f"{proto}:{key}"] = mean_std(agg[proto][b])
                row[f"{proto}:detect_{key}"] = mean_std(dagg[proto][b])
        rows.append(row)
        print(f"hidden ({h1:>2},{h2:>2}) params {row['params']:5d} | diagnosis, load-out f/8/4 "
              f"{row['leave-one-load-out:float'][0]:.3f}/{row['leave-one-load-out:int8'][0]:.3f}/{row['leave-one-load-out:int4'][0]:.3f} | "
              f"detection, size-out f/8/4 "
              f"{row['leave-one-size-out:detect_float'][0]:.3f}/{row['leave-one-size-out:detect_int8'][0]:.3f}/{row['leave-one-size-out:detect_int4'][0]:.3f}", flush=True)
    (RESULTS / "phase1_ladder.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def window_rms(recs, factor):
    """Root-mean-square of every window, in the same order as ``build_dataset``'s windows."""
    from sustain_iq.signals import decimate, windows

    parts = []
    for r in recs:
        w, _ = windows(decimate(r.signal, factor))
        if len(w):
            parts.append(np.sqrt((w.astype(np.float64) ** 2).mean(axis=1)))
    return np.concatenate(parts)


def rms_threshold_detection(rms, faulty, tr, te):
    """Healthy-vs-faulty by one threshold on window RMS, chosen on the training windows only."""
    cands = np.quantile(rms[tr], np.linspace(0.01, 0.99, 197))
    best = max(cands, key=lambda t: balanced_accuracy(faulty[tr], (rms[tr] > t).astype(int), 2))
    return balanced_accuracy(faulty[te], (rms[te] > best).astype(int), 2)


def study_bandwidth(recs):
    rows = []
    for factor in (1, 2, 4, 8, 16):
        ds = build_dataset(recs, factor=factor)
        fs = CWRU_FS / factor
        rms = window_rms(recs, factor)
        faulty = (ds.y > 0).astype(int)
        agg = {proto: {b: [] for b in (None, 8)} for proto in ("leave-one-load-out", "leave-one-size-out")}
        dagg = {proto: {b: [] for b in (None, 8)} for proto in agg}
        ragg = {proto: [] for proto in agg}
        for proto, name, tr, te in honest_folds(ds):
            r = evaluate(ds, tr, te, MAIN_SIZES, bits=(None, 8))
            ragg[proto].append(rms_threshold_detection(rms, faulty, tr, te))
            for b in (None, 8):
                agg[proto][b].append(float(np.mean(r["acc"][b])))
                dagg[proto][b].append(float(np.mean(r["det"][b])))
        row = {"factor": factor, "sample_rate_hz": fs, "nyquist_hz": fs / 2, "window_s": 1024 / fs, "windows": int(len(ds.y))}
        for proto in agg:
            row[f"{proto}:float"] = mean_std(agg[proto][None])
            row[f"{proto}:int8"] = mean_std(agg[proto][8])
            row[f"{proto}:detect_float"] = mean_std(dagg[proto][None])
            row[f"{proto}:detect_int8"] = mean_std(dagg[proto][8])
            row[f"{proto}:detect_rms_threshold"] = mean_std(ragg[proto])
        rows.append(row)
        print(f"fs {fs:7.0f} Hz (Nyquist {fs/2:6.0f}, window {1024/fs*1000:5.0f} ms) | diagnosis load-out {row['leave-one-load-out:float'][0]:.3f} "
              f"| detection load-out {row['leave-one-load-out:detect_float'][0]:.3f} size-out {row['leave-one-size-out:detect_float'][0]:.3f} "
              f"| RMS-threshold load-out {row['leave-one-load-out:detect_rms_threshold'][0]:.3f} size-out {row['leave-one-size-out:detect_rms_threshold'][0]:.3f}", flush=True)
    (RESULTS / "phase1_bandwidth.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def study_features(recs):
    out = {}
    print("\n== feature sets under the honest protocols (balanced accuracy, mean over folds) ==")
    for feat in FEATURE_SETS:
        ds = build_dataset(recs, features=feat)
        agg = {}
        for proto, name, tr, te in honest_folds(ds):
            r = evaluate(ds, tr, te, MAIN_SIZES, bits=(None, 8), baseline=True)
            a = agg.setdefault(proto, {"float": [], "int8": [], "logreg": []})
            a["float"].append(float(np.mean(r["acc"][None])))
            a["int8"].append(float(np.mean(r["acc"][8])))
            a["logreg"].append(r["logreg"])
        out[feat] = {proto: {k: mean_std(v) for k, v in a.items()} for proto, a in agg.items()}
        for proto, a in out[feat].items():
            print(f"{feat:9s} {proto:20s} MLP {a['float'][0]:.3f}  int8 {a['int8'][0]:.3f}  logreg {a['logreg'][0]:.3f}", flush=True)
    (RESULTS / "phase1_features.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def lock_group(label: str) -> str:
    m = re.match(r"(IR|B)028_", label)
    if m:
        return f"{m.group(1)} 0.028 in"
    m = re.match(r"OR\d{3}@(3|12)_", label)
    return f"OR @{m.group(1)}:00" if m else label


def study_lock(train_recs, lock_recs):
    """Train on all 40 recordings; evaluate once on the 24 locked ones. Nothing here is used to choose anything."""
    out = {}
    print("\n== locked confirmatory set (recall per group; balanced accuracy over IR/B/OR) ==")
    for feat in FEATURE_SETS:
        ds = build_dataset(train_recs, features=feat)
        lk = build_dataset(lock_recs, features=feat)
        std = Standardizer.fit(ds.X)
        Xs, Xl = std(ds.X), std(lk.X)
        groups = np.array([lock_group(lock_recs[i].label) for i in lk.rec])
        from sklearn.linear_model import LogisticRegression

        preds = {"float": [], "int8": []}
        for seed in SEEDS:
            m = MLP(MAIN_SIZES, seed).fit(Xs, ds.y, seed=seed)
            preds["float"].append(m.predict(Xl))
            preds["int8"].append(quantize(m, Xs, 8).predict(Xl))
        lr_pred = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xs, ds.y).predict(Xl)
        row = {}
        for name, plist in (("mlp_float", preds["float"]), ("mlp_int8", preds["int8"]), ("logreg", [lr_pred])):
            per_group = {g: float(np.mean([np.mean(p[groups == g] == lk.y[groups == g]) for p in plist])) for g in sorted(set(groups))}
            row[name] = {
                "balanced_accuracy": float(np.mean([balanced_accuracy(lk.y, p) for p in plist])),
                "fault_detected": float(np.mean([np.mean(p != 0) for p in plist])),  # every locked window is faulty
                "recall_by_group": per_group,
            }
        out[feat] = row
        for name, r in row.items():
            g = "  ".join(f"{k}: {v:.2f}" for k, v in r["recall_by_group"].items())
            print(f"{feat:9s} {name:10s} bal.acc {r['balanced_accuracy']:.3f}  fault detected {r['fault_detected']:.3f} | {g}", flush=True)
    out["n_locked_windows"] = int(len(lk.y))
    (RESULTS / "phase1_lock.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["leak", "ladder", "bandwidth", "features", "lock", "all"])
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    recs = load_recordings(ROOT / "data" / "cwru")
    studies = {"leak": study_leak, "ladder": study_ladder, "bandwidth": study_bandwidth, "features": study_features}
    for name in (studies if args.study == "all" else [args.study] if args.study != "lock" else []):
        studies[name](recs)
    if args.study in ("lock", "all"):
        study_lock(recs, load_recordings(ROOT / "data" / "cwru", locked=True))


if __name__ == "__main__":
    main()
