# Phase 1 results

All numbers are balanced accuracy (mean recall over the classes present) on held-out recordings unless stated, from `results/phase1_*.json`, produced by `scripts/run_experiments.py`. "Mean ± std" is over folds and 3 training seeds; the standard deviation describes spread, not a confidence interval.

## Setup

- **Data.** 40 CWRU 12 kHz drive-end recordings: normal baseline (4 loads) and inner-race (IR), ball (B) and outer-race (OR, centred at 6:00) faults at 0.007, 0.014 and 0.021 inch diameters, each at motor loads 0-3 hp. Cut into 1,024-sample windows with a 512-sample stride (85 ms at 12 kHz).
- **Locked set.** 24 further recordings that were never used for training, tuning, feature choice or model choice: 0.028 inch inner-race and ball faults (larger than any training fault) and outer-race faults at the 3:00 and 12:00 positions (the training outer-race faults are all at 6:00). Evaluated after all the studies below, for every variant and nothing selected from it. The evaluation was run twice with identical numbers (the second run only added the "fault detected" column). It contains no healthy recordings.
- **Features.** A Hann-windowed real FFT of the window, power summed into 32 bands that are log-spaced in bin index, then log10. Standardised with training-window statistics only.
- **Model.** MLP 32-32-16-4 (1,652 parameters), ReLU, class-balanced cross-entropy, Adam, 60 epochs, 3 training seeds. Post-training int8 quantisation described in [design.md](design.md).
- **Honest protocols hold out whole recordings.** *Leave-one-load-out*: train on three motor loads, test on the fourth (4 folds). *Leave-one-size-out*: train on two fault diameters, test on the third (3 folds); the normal recordings, which have no diameter, are split by time (the first 75% train, the last 25% test, with a gap), so the healthy side of this protocol is optimistic.
- **A data quirk handled.** `Normal_2.mat` (file 99) contains an exact copy of file 98's recording as well as its own. The loader picks the variable named for the file; taking the first match would have silently used one load's data as another's.

## 1. How the test changes the score

| How the held-out data is chosen | MLP (float) | MLP (int8) | Logistic regression |
| --- | --- | --- | --- |
| random windows (leaky) | 0.999 ± 0.000 | n/a | 0.995 ± 0.001 |
| leave-one-load-out | 0.987 ± 0.012 | 0.987 ± 0.012 | 0.983 ± 0.008 |
| leave-one-size-out | 0.519 ± 0.030 | 0.519 ± 0.030 | 0.614 ± 0.119 |

- A random split of pooled windows lets near-duplicate neighbours of one recording sit on both sides. Here that inflates the score by only about a point, because the classes are so separable across loads.
- The large drop appears when a fault *size* is held out, and the plain logistic regression generalises better there (0.614) than the MLP (0.519), with large fold-to-fold variation (±0.119). More model did not help.

Where the cross-size failure comes from:

Confusion matrix, leave-one-load-out, summed over folds (rows true, columns predicted):

| | N | IR | B | OR | recall |
| --- | --- | --- | --- | --- | --- |
| N | 3312 | 0 | 0 | 0 | 1.00 |
| IR | 0 | 2758 | 66 | 14 | 0.97 |
| B | 0 | 82 | 2755 | 3 | 0.97 |
| OR | 0 | 1 | 4 | 2839 | 1.00 |

Confusion matrix, leave-one-size-out, summed over folds (rows true, columns predicted):

| | N | IR | B | OR | recall |
| --- | --- | --- | --- | --- | --- |
| N | 2466 | 0 | 0 | 0 | 1.00 |
| IR | 0 | 812 | 1483 | 543 | 0.29 |
| B | 0 | 1063 | 1647 | 130 | 0.58 |
| OR | 0 | 1045 | 958 | 841 | 0.30 |

Healthy windows are never confused with faulty ones (recall 1.00, subject to the optimistic healthy-side split above), but across an unseen fault size the three fault types are confused with each other at close to chance (recall 0.29, 0.58, 0.30 against a chance level of about 0.33).

## 2. Three feature sets, same folds

Tried after the baseline failed across fault size, in this order: (a) absolute log band power (the baseline), (b) the same with each window's mean removed (exactly amplitude-invariant), (c) the envelope spectrum of the 1 kHz high-passed signal (the classic bearing-diagnosis route). These variants were designed after seeing the baseline's cross-size failure and evaluated on the same folds, so they are exploratory; the locked set below is the confirmation.

| Feature set | Unseen load: MLP / int8 / logistic | Unseen fault size: MLP / int8 / logistic |
| --- | --- | --- |
| Absolute log band power | 0.987 / 0.987 / 0.983 | 0.519 / 0.519 / 0.614 |
| Band shape (amplitude removed) | 0.985 / 0.986 / 0.982 | 0.535 / 0.534 / 0.586 |
| Envelope spectrum | 0.839 / 0.839 / 0.806 | 0.524 / 0.523 / 0.600 |

None generalises across fault size, so the failure is not simply that amplitude scales with severity. The envelope features are clearly worse across loads.

## 3. The locked recordings (every variant, nothing selected)

Each model is trained on all 40 recordings. "Fault detected" is the fraction of locked windows classified as anything other than healthy (every locked window is faulty). Recall is the fraction of windows classified as the correct fault type.

| Feature set | Model | Balanced accuracy | Fault detected | Recall B 0.028 in | Recall IR 0.028 in | Recall OR @12:00 | Recall OR @3:00 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Absolute | MLP | 0.435 | 1.000 | 0.00 | 1.00 | 0.18 | 0.42 |
| Absolute | MLP int8 | 0.435 | 1.000 | 0.00 | 1.00 | 0.19 | 0.42 |
| Absolute | Logistic | 0.453 | 1.000 | 0.00 | 0.99 | 0.25 | 0.50 |
| Shape | MLP | 0.388 | 1.000 | 0.00 | 1.00 | 0.15 | 0.18 |
| Shape | MLP int8 | 0.387 | 1.000 | 0.00 | 1.00 | 0.15 | 0.17 |
| Shape | Logistic | 0.446 | 1.000 | 0.00 | 1.00 | 0.18 | 0.49 |
| Envelope | MLP | 0.473 | 0.973 | 0.85 | 0.01 | 0.63 | 0.47 |
| Envelope | MLP int8 | 0.473 | 0.973 | 0.85 | 0.02 | 0.63 | 0.47 |
| Envelope | Logistic | 0.562 | 0.976 | 0.93 | 0.03 | 0.82 | 0.63 |

Balanced accuracy is over the three fault classes present (chance 0.33). No variant is reliably good, and the failure modes differ by feature set: the absolute and shape features call every larger inner-race fault correctly and miss every larger ball fault, while the envelope features do the opposite. All variants flag the locked windows as faulty 97-100% of the time. Because the failure pattern depends on the feature set, no single variant should be chosen from this table; nothing was.

## 4. Model size and weight precision

Absolute features, 3 training seeds. "Diagnosis" is four-class balanced accuracy and "detection" is healthy-versus-faulty balanced accuracy from the same model, both under leave-one-load-out. Flash is packed weights plus 4-byte biases plus 8 bytes of requantisation constants per hidden layer; RAM is an int8 input buffer plus two int8 activation buffers; MACs are multiply-accumulates per window. These are accounting figures, not measurements.

| Hidden layers | Parameters | Flash int8 / int4 (bytes) | RAM (bytes) | MACs | Diagnosis, unseen load: float / int8 / int4 | Detection, unseen load: float / int8 / int4 |
| --- | --- | --- | --- | --- | --- | --- |
| 4, 4 | 172 | 224 / 144 | 40 | 160 | 0.976 / 0.973 / 0.947 | 1.000 / 1.000 / 0.992 |
| 8, 8 | 372 | 448 / 272 | 48 | 352 | 0.983 / 0.983 / 0.974 | 1.000 / 1.000 / 1.000 |
| 16, 8 | 700 | 800 / 464 | 64 | 672 | 0.991 / 0.990 / 0.986 | 1.000 / 1.000 / 1.000 |
| 32, 16 | 1652 | 1824 / 1024 | 96 | 1600 | 0.987 / 0.987 / 0.983 | 1.000 / 1.000 / 1.000 |
| 64, 32 | 4324 | 4640 / 2528 | 160 | 4224 | 0.990 / 0.990 / 0.989 | 1.000 / 1.000 / 1.000 |

Beyond about 700 parameters there is no gain on this task. int8 costs at most 0.3 points; int4 costs 0.1 to 3 points, most at the smallest sizes. The int4 results are simulated (weights restricted to 15 levels and stored as int8); the Rust engine implements int8 only. The feature extractor, not the network, will dominate the cost on a device.

## 5. A slower, cheaper accelerometer

The 12 kHz recordings are low-pass filtered and decimated to emulate slower sensors. The window stays 1,024 samples (so the compute is constant and the window gets longer in time). The comparison baseline is the simplest possible detector: one threshold on the window's RMS, chosen on the training windows.

| Sample rate (Hz) | Nyquist (Hz) | Window (ms) | Diagnosis, unseen load (MLP) | Detection, unseen load: MLP | Detection, unseen load: RMS threshold | Detection, unseen size: MLP | Detection, unseen size: RMS threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12000 | 6000 | 85 | 0.987 | 1.000 | 0.996 | 1.000 | 0.948 |
| 6000 | 3000 | 171 | 0.973 | 1.000 | 0.709 | 1.000 | 0.798 |
| 3000 | 1500 | 341 | 0.933 | 1.000 | 0.549 | 1.000 | 0.623 |
| 1500 | 750 | 683 | 0.859 | 1.000 | 0.675 | 1.000 | 0.688 |
| 750 | 375 | 1365 | 0.917 | 1.000 | 0.381 | 1.000 | 0.482 |

- At the full rate the RMS threshold already detects faults 95-99.6% of the time, so detection on CWRU is close to trivial at 12 kHz.
- Once the high frequencies are filtered away, the overall level stops separating healthy from faulty (38-71%), but the band features still do (100%): the information is in the shape of the low-frequency spectrum, not its total.
- Diagnosis degrades as the rate falls. The 750 Hz figure (0.917) is higher than the 1.5 kHz figure (0.859); with this few folds that is noise, not a trend.
- The healthy side of the unseen-size detection column is split by time within the same recordings, so only the unseen-load columns are free of that overlap.
- Implication for the hardware phase: a modest accelerometer sampled at a few kHz is enough to detect a fault with these features, while fine diagnosis wants more bandwidth and, more importantly, training data that covers the conditions it will see.

## 6. The inference engine

`firmware/tinyinfer` is integer-only, `no_std` and allocation-free. `scripts/export_golden.py` trains the deployment model on all 40 recordings, exports its integers and 305 quantised inputs (300 real windows and 5 saturation edge cases), and the Rust test rebuilds the model and requires every logit and class to match exactly: 305 of 305 do. The same integers are also recomputed with plain Python integers in the Python tests. The deployed 32-32-16-4 int8 model: 1,652 parameters, 1,824 bytes of flash, 96 bytes of RAM for activations, 1,600 MACs per window.

## Limitations

- CWRU is a single lab rig with large, deliberately seeded single-point faults; near-perfect detection is expected there and says little about incipient faults or other machines.
- Fault diagnosis does not generalise to unseen severities or fault positions with these small models and features; the 52% and 39-56% figures are the honest ones.
- Seeds and folds are few; spreads are not confidence intervals. The locked set has no healthy data.
- Two choices were made after seeing results on the training-and-evaluation folds (the added feature sets, and adding detection as a metric). The locked set was evaluated only after those choices (twice, with identical numbers, the second time to add a column) and was not used to pick anything.
- Everything is on a PC. No latency, memory, or energy was measured on a microcontroller.
