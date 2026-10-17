# Sustain-IQ

[![CI](https://github.com/PCSchmidt/sustain-iq/actions/workflows/ci.yml/badge.svg)](https://github.com/PCSchmidt/sustain-iq/actions/workflows/ci.yml)

A small vibration classifier for machine health, taken from public bearing data down to something that fits in a microcontroller: features a microcontroller can compute, an integer-only model of a couple of kilobytes, and an inference engine in Rust that reproduces the Python reference bit for bit. The question behind it is the same one the sibling projects ask at other scales (a detector, a language model): what does each reduction in size, precision and sensor quality cost, and what does an honest test actually show?

**Status: phase 1 (host side) complete; nothing has run on a microcontroller yet.** Everything below was measured on a PC with the public CWRU bearing dataset. Phase 2 puts the same model on a microcontroller with a real accelerometer; no hardware has been bought or tested, and no on-device latency, memory or energy numbers exist. See the [roadmap](docs/roadmap.md).

Part of [physical-ai-portfolio](https://github.com/PCSchmidt/physical-ai-portfolio). It applies the measurement habits from [Sentinel Node](https://github.com/PCSchmidt/sentinel-node) and [TriageBox](https://github.com/PCSchmidt/triagebox).

![Left: one four-class classifier scores 99.9% on a leaky random split, 98.7% on an unseen motor load, 51.9% on an unseen fault size and 43.5% on locked recordings. Right: as the accelerometer sample rate falls from 12 kHz to 750 Hz, fault detection by the model stays at 100%, diagnosis falls to 86-92%, and detection by a single vibration-level threshold collapses.](docs/phase1-findings.svg)

## What the experiments show

Data: 40 recordings from the [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter) (healthy, and inner-race, ball and outer-race faults at three sizes and four motor loads), cut into 1,024-sample windows; plus 24 further recordings held back as a **locked test set** that was never used for training, tuning, or choosing between options. Model: a 32-32-16-4 MLP on 32 log band energies. Scores are balanced accuracy; full tables in [docs/results.md](docs/results.md).

1. **Telling faults from healthy is easy and robust; telling which fault it is, is not.** Fault *detection* scored 100% under every honest protocol and 97-100% on the locked recordings. Four-class *diagnosis* held up across unseen motor loads (98.7%) but fell to 52% for an unseen fault size and 39-56% on the locked recordings (chance is 25-33%). The model recognises the fault sizes it saw, not the fault types.
2. **The usual shortcut hides this.** Scoring on random windows pooled across recordings gives 99.9%. Most of the damage in this dataset comes not from that leak (about 1 point) but from never testing on a condition the model has not seen.
3. **Three feature sets did not fix it.** Amplitude-free band shape and the classic envelope spectrum were tried because the baseline failed; neither generalised across fault size, and the envelope features were worse across loads. They were evaluated on the same folds that exposed the problem, then on the locked set, and all results are reported.
4. **A tiny model is enough.** A 172-parameter model (224 bytes of flash at int8, including the requantisation constants) reaches 97.6% diagnosis across loads and 100% detection; 700 parameters reach 99.1%. int8 costs at most 0.3 points; int4 weights cost 0.1 to 3 points (most at the smallest sizes).
5. **A slow, cheap accelerometer still detects faults.** With the signal low-passed and decimated to 750 Hz (Nyquist 375 Hz), the model still detected faults 100% of the time across loads and sizes, while a single threshold on vibration level fell to 38-48%. Diagnosis degrades steadily as the rate falls (98.7% at 12 kHz, 93% at 3 kHz, 86% at 1.5 kHz, 92% at 750 Hz; the last is not monotonic and is within the noise of the others).
6. **The inference engine matches the reference exactly.** The Rust `tinyinfer` crate (integer-only, `no_std`, no allocation) reproduces all 305 logits and classes exported from Python, including saturation edge cases. The deployed 32-32-16-4 model needs about 1.8 KB of flash and 96 bytes of RAM for activations, and 1,600 multiply-accumulates per window (accounting, not measurement).

## What these results do not show

- **CWRU is lab data with large, deliberately seeded faults on one test rig.** Detection near 100% reflects that: at the full sample rate a single vibration-level threshold already scores 95-99.6%. Real, incipient faults on a real machine will be harder, and none of this says how the model behaves on a different machine.
- **Fault *detection* here is mostly "is something clearly wrong".** The healthy side of the cross-fault-size test is split by time within the same recordings, so its false-alarm figure is optimistic; the cross-load detection figures have no such overlap. The locked set contains no healthy data, so it says nothing about false alarms.
- **Three training seeds and a handful of folds.** The standard deviations describe spread across folds and seeds; they are not confidence intervals.
- **No on-device measurement.** Flash, RAM and multiply counts are arithmetic on the model. Feature extraction (a 1,024-point FFT per window) will likely cost more than the network and is not counted in the MACs; it is estimated only as a butterfly count.

## Layout

| Path | What it is |
| --- | --- |
| `src/sustain_iq/` | Windowing, feature extraction, leak-aware splits, a numpy MLP, integer quantisation, export |
| `scripts/` | Download the data, run the studies, build the tables and figure |
| `firmware/tinyinfer/` | Integer-only Rust inference engine and the bit-exactness test |
| `results/` | Saved results (JSON) behind every number in the docs |
| `docs/` | [Results](docs/results.md), [design](docs/design.md), [roadmap](docs/roadmap.md) |

## Reproduce

```sh
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python scripts/download_cwru.py                   # 64 files, about 190 MB, from Case Western's site
python scripts/run_experiments.py all
python scripts/export_golden.py                   # writes the Rust test vectors
cd firmware/tinyinfer && cargo test
```

## Data and licence

The CWRU data belong to Case Western Reserve University's Bearing Data Center; the scripts download them and the repository does not redistribute them. Please cite the Bearing Data Center when using results derived from them. Code: MIT, see [LICENSE](LICENSE).

## Scope

Condition monitoring of machinery. No weapons or targeting functionality, and no controlled or employer information.
