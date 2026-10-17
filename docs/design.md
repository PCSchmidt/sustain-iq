# Design

## Data flow

```text
accelerometer ─▶ 1,024-sample window ─▶ Hann + real FFT ─▶ 32 log band energies ─▶ standardise ─▶ int8 ─▶ MLP (int8) ─▶ class
   (device)            (device)              (device)             (device)            (device)    (device)  (tinyinfer)
```

Everything after the sensor is meant to run on the microcontroller. In phase 1 the same steps run on a PC (numpy for the features, a Python integer reference for the network) so the numbers can be checked before any hardware exists.

## Decisions made

- **Features a microcontroller can compute.** One fixed-size FFT per window, band sums, a log. The 32 bands are log-spaced in *bin index*, so decimating the signal (a slower sensor) changes the frequency of every band but not the cost or the feature count. The first few bands are one bin wide because log spacing is finer than the FFT resolution there.
- **A small MLP, not a CNN.** 32-32-16-4 is 1,652 parameters. The model-size study shows nothing is gained beyond about 700 parameters on this task.
- **Integer quantisation written out by hand.** Symmetric per-tensor int8 weights and inputs, int32 biases and accumulators, and a TFLite-style fixed-point requantisation: `y = (acc * m0 + (1 << (shift - 1))) >> shift`, clamped to 0..127 (the clamp is the ReLU), with `m0` a 31-bit integer. Only integer operations, so any correct implementation gives identical results. Writing it out, rather than calling a framework, is what makes the bit-exact cross-language test possible.
- **A bit-exact reference.** The Python quantiser exports its integers and 305 test inputs; the Rust crate must reproduce every logit exactly.
- **Whole-recording splits.** Windows from one recording are near-duplicates, so every reported number holds out entire recordings (a motor load, or a fault size), and a locked set of never-used recordings is kept for confirmation. A leaky random split is reported only as a contrast.
- **Report detection and diagnosis separately.** The honest results say a tiny model reliably detects a fault but cannot reliably name it under unseen conditions. Treating them as one "accuracy" would hide that.
- **Always compare against the simplest baseline.** A one-threshold RMS detector and a logistic regression are reported beside the MLP. They showed that detection on this dataset is nearly trivial at the full rate, and that a plain linear model can beat the MLP across fault sizes.

## Compute budget (arithmetic, not measurement)

| Item | Deployed model (32-32-16-4, int8) |
| --- | --- |
| Parameters | 1,652 |
| Flash | 1,824 bytes (weights, int32 biases, requantisation constants) |
| RAM for activations | 96 bytes (int8 input plus two int8 buffers) |
| MACs per window | 1,600 |
| FFT per window | 5,120 complex radix-2 butterflies for 1,024 points |

The network is tiny next to the feature extraction: one 1,024-point FFT is about 5,000 butterflies of roughly ten floating-point operations each, against 1,600 integer multiply-accumulates for the whole network, and the window itself (1,024 samples at 2-4 bytes each) dwarfs the 96 bytes of activations as the dominant RAM cost. Neither is measured yet; the operation counts are rough estimates. These are the figures phase 2 exists to replace with measurements.

## Phase 2 plan (not started)

Run the same model on a real microcontroller with a real accelerometer.

- **Board.** A Raspberry Pi Pico 2 (RP2350) is the leading candidate: about $5 for the board (list price per Raspberry Pi's March 2026 product brief), a dual-core Arm Cortex-M33 part with TrustZone, signed boot and one-time-programmable key storage, which also makes it a platform for a later secure-firmware project. Embassy builds for it (checked, see `firmware/pico2-bringup`, not yet run on a board). Whether its floating-point and DSP support suit an FFT is not verified.
- **Sensor.** An SPI or I2C MEMS accelerometer breakout. Its maximum output data rate and bandwidth must be read from the datasheet; the bandwidth study says a few kHz is enough to detect faults and that diagnosis wants more.
- **Measurements to make:** the achieved sample rate and its jitter; cycles per window split into FFT, features and network; flash and RAM from the linker map; energy per window with a current-sense breakout; and numeric agreement between the device's features and the Python features (feature mismatch is the likeliest way a good host model fails on a device).
- **A real rig.** A box fan with weights added, to produce imbalance and misalignment signals that CWRU does not contain, collected under several mountings and speeds. The honest lesson from phase 1 applies: train and test on different conditions.

## Open decisions

- Whether the deployed model should be trained for detection only (healthy vs faulty), given that diagnosis does not generalise. Phase 1 supports detection as the dependable function.
- Whether to add an anomaly-detection formulation (model of "normal" only), which would not need fault examples at all, as a comparison.
- Whether the on-device FFT uses a library or a hand-written fixed-point version, and how that changes the features.
- The accelerometer, once its datasheet bandwidth is checked.

## Known limitations

- Phase 1 is host-only and uses a single lab dataset.
- int4 weights are simulated; the Rust engine implements int8 only.
- Key management, secure update and any security review are out of scope until the secure-firmware project.
