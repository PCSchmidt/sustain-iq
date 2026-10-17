"""Draw the phase 1 findings as one static SVG (docs/phase1-findings.svg).

    python scripts/make_figure.py

Left: the same four-class classifier scored four ways (a leaky random split, an
unseen motor load, an unseen fault size, and the locked recordings). Right: what
survives a slower accelerometer, for fault detection by the MLP, fault diagnosis by
the MLP, and fault detection by a single RMS threshold.

All numbers are read from ``results/phase1_*.json``. Colours are validated categorical
slots 1-3 with light and dark steps selected rather than inverted; marker shape repeats
colour so identity never rests on colour alone. The aqua slot is below 3:1 contrast on
the light surface, so every series is direct-labelled and the README carries the
numbers as a table.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "phase1-findings.svg"
W, H = 940, 480


def load(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def main() -> None:
    leak = {r["protocol"]: r["mlp_float"][0] for r in load("phase1_leak.json")["rows"]}
    lock = load("phase1_lock.json")["abs"]["mlp_float"]["balanced_accuracy"]
    bars = [
        ("Random windows", "(leaky)", leak["random windows (leaky)"]),
        ("Unseen motor load", "", leak["leave-one-load-out"]),
        ("Unseen fault size", "", leak["leave-one-size-out"]),
        ("Locked recordings", "(new sizes, positions)", lock),
    ]
    bw = load("phase1_bandwidth.json")["rows"]
    rows = sorted(bw, key=lambda r: -r["sample_rate_hz"])  # fast to slow, left to right
    series = [
        ("Detection, MLP", "s1", "circle", [r["leave-one-load-out:detect_float"][0] for r in rows]),
        ("Diagnosis, MLP", "s2", "square", [r["leave-one-load-out:float"][0] for r in rows]),
        ("Detection, RMS threshold only", "s3", "diamond", [r["leave-one-load-out:detect_rms_threshold"][0] for r in rows]),
    ]

    o: list[str] = []
    add = o.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">')
    add('<title id="t">Bearing-fault classification: how the score depends on how you test, and what a slower sensor costs</title>')
    add('<desc id="d">Left: a four-class bearing-fault classifier scores 99.9 percent on a leaky random split, 98.7 percent on an '
        'unseen motor load, 51.9 percent on an unseen fault size and 43.5 percent on locked recordings. Right: as the accelerometer '
        'sample rate falls from 12,000 to 750 hertz, fault detection by the classifier stays at 100 percent, four-class diagnosis '
        'falls from 98.7 to about 86 to 92 percent, and detection by a single vibration-level threshold collapses.</desc>')
    add("""<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}
@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.bg{fill:var(--surface)} .h1{font-size:16px;font-weight:600;fill:var(--ink)} .h2{font-size:12px;fill:var(--ink2)}
.panel{font-size:13px;font-weight:600;fill:var(--ink)} .tick{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}
.axis-title{font-size:12px;fill:var(--ink2)} .lab{font-size:12px;fill:var(--ink2)} .val{font-size:12px;font-weight:600;fill:var(--ink)} .note{font-size:11px;fill:var(--muted)}
.grid{stroke:var(--grid);stroke-width:1} .base{stroke:var(--axis);stroke-width:1}
.b{fill:var(--s1)} .l1{stroke:var(--s1)} .l2{stroke:var(--s2)} .l3{stroke:var(--s3)}
.f1{fill:var(--s1)} .f2{fill:var(--s2)} .f3{fill:var(--s3)} .ring{stroke:var(--surface);stroke-width:2}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
</style>""")
    add(f'<rect class="bg" width="{W}" height="{H}"/>')
    add('<text class="h1" x="24" y="30">Bearing-fault classification on the CWRU data: it depends on how you test it</text>')
    add('<text class="h2" x="24" y="50">Balanced accuracy of a 32-32-16-4 MLP on 32 log band-energy features (1,024-sample windows). Chance is 25% for four classes.</text>')

    # ---------------------------------------------------------------- left: bars
    lx, bar_x0, bar_w = 24, 262, 170
    add(f'<text class="panel" x="{lx}" y="100">Four-class diagnosis, scored four ways</text>')
    top, row_h = 125, 62
    for frac in (0.0, 0.5, 1.0):
        x = bar_x0 + frac * bar_w
        add(f'<line class="{"base" if frac == 0 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 8}" y2="{top + 4 * row_h - 18}"/>')
        add(f'<text class="tick" x="{x:.1f}" y="{top + 4 * row_h - 2}" text-anchor="middle">{int(frac * 100)}%</text>')
    for i, (name, sub, v) in enumerate(bars):
        cy = top + i * row_h + 16
        w = v * bar_w
        add(f'<text class="lab" x="{bar_x0 - 12}" y="{cy - (4 if sub else -4)}" text-anchor="end">{escape(name)}</text>')
        if sub:
            add(f'<text class="note" x="{bar_x0 - 12}" y="{cy + 12}" text-anchor="end">{escape(sub)}</text>')
        # 4 px rounded data end, square at the baseline
        r = 4
        add(f'<path class="b" d="M{bar_x0},{cy - 12} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{24 - 2 * r} a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.1f} z">'
            f'<title>{escape(name)} {escape(sub)}: {100 * v:.1f}%</title></path>')
        add(f'<text class="val" x="{bar_x0 + w + 8:.1f}" y="{cy + 4}">{100 * v:.1f}%</text>')

    # --------------------------------------------------------- right: lines
    rx0, rx1, ytop, ybot = 560, 760, 150, 360
    add(f'<text class="panel" x="520" y="100">What a slower accelerometer costs</text>')
    # legend (three series)
    lx0 = 520
    for k, (name, cls, shape, _) in enumerate(series):
        gx = lx0 + k * 0  # stacked legend, one per row
        gy = 118 + k * 0
    legend_y = 120
    xs = [520, 640, 760]
    for (name, cls, shape, _), x in zip(series, xs):
        pass
    for k, (name, cls, shape, _) in enumerate(series):
        x = 520 + [0, 112, 224][k]
        add(_marker(shape, cls, x + 6, legend_y, 5))
        add(f'<text class="lab" x="{x + 16}" y="{legend_y + 4}">{escape(["Detection (MLP)", "Diagnosis (MLP)", "Detection (RMS only)"][k])}</text>')

    def px(i):
        return rx0 + 14 + i / (len(rows) - 1) * (rx1 - rx0 - 28)

    def py(v):
        return ybot - v * (ybot - ytop)

    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        add(f'<line class="{"base" if v == 0 else "grid"}" x1="{rx0}" x2="{rx1}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>')
        add(f'<text class="tick" x="{rx0 - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>')
    for i, r in enumerate(rows):
        f = r["sample_rate_hz"]
        label = f"{f / 1000:g}k" if f >= 1000 else f"{f:g}"
        add(f'<text class="tick" x="{px(i):.1f}" y="{ybot + 18}" text-anchor="middle">{label}</text>')
    add(f'<text class="axis-title" x="{(rx0 + rx1) / 2:.0f}" y="{ybot + 40}" text-anchor="middle">Accelerometer sample rate (Hz), slower to the right</text>')
    end_labels = []
    for name, cls, shape, vals in series:
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        add(f'<polyline class="line l{cls[1]}" points="{pts}"/>')
        for i, v in enumerate(vals):
            tip = f"{name}: {100 * v:.1f}% at {rows[i]['sample_rate_hz']:g} Hz"
            add(_marker(shape, cls, px(i), py(v), 5, title=tip))
        end_labels.append((py(vals[-1]), name))
    # direct labels at the right end (the lines have separated by the slowest sensor)
    for y, name in end_labels:
        add(f'<text class="lab" x="{rx1 + 14}" y="{y + 4:.1f}">{escape(name)}</text>')

    add(f'<text class="note" x="24" y="{H - 34}">Left: random split 3 seeds; unseen load 4 folds; unseen size 3 folds; locked set evaluated after all choices (3 training seeds). '
        f'Right: leave-one-load-out, 3 training seeds.</text>')
    add(f'<text class="note" x="24" y="{H - 20}">The locked set is 24 recordings never used for training or choices (0.028 in. faults; outer race at 3:00 and 12:00). '
        f'Lab data with large seeded faults.</text>')
    add(f'<text class="note" x="24" y="{H - 6}">Data: Case Western Reserve University Bearing Data Center. Values: docs/results.md.</text>')
    add("</svg>")
    OUT.write_text("\n".join(o) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


def _marker(shape: str, cls: str, cx: float, cy: float, r: float, title: str = "") -> str:
    t = f"<title>{escape(title)}</title>" if title else ""
    if shape == "circle":
        return f'<circle class="{"f" + cls[1]} ring" cx="{cx:.1f}" cy="{cy:.1f}" r="{r}">{t}</circle>'
    if shape == "square":
        return f'<rect class="{"f" + cls[1]} ring" x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1.5">{t}</rect>'
    pts = f"{cx:.1f},{cy - r * 1.3:.1f} {cx + r * 1.3:.1f},{cy:.1f} {cx:.1f},{cy + r * 1.3:.1f} {cx - r * 1.3:.1f},{cy:.1f}"
    return f'<polygon class="{"f" + cls[1]} ring" points="{pts}" stroke-linejoin="round">{t}</polygon>'


if __name__ == "__main__":
    main()
