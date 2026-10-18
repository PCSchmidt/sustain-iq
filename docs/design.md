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

## Host-side preparation for phase 2

Done before any hardware exists, so that bring-up is a matter of running and measuring ([bring-up checklist](bringup-checklist.md)). None of this has run on a board.

**Features in Rust (`firmware/tinyfeat`).** The same computation as `band_features`: mean removal, the symmetric Hann window, a 1,024-point FFT, 513 power bins summed into the 32 log-spaced bands, `log10(sum + 1e-12)`. Twiddles, the window and the band edges are generated from the Python definitions by `scripts/gen_tinyfeat_tables.py`, so they cannot drift apart. The FFT is a plain radix-2 complex FFT in `f32` of the real signal (about twice the work of a packed real FFT; that optimisation waits for a cycle measurement that says it is needed).

- *Agreement with Python.* On 52 windows (12 real windows per class plus edge cases) committed with the repository: worst feature error 9.3e-5 log10 units on real windows, int8 inputs identical on all 1,632 values compared, predicted class identical for all 52. On a larger local run of 2,004 windows (2,000 real windows from the 40 training recordings plus 4 edge windows; `--per-class 500`, 29 MB, not committed): worst feature error 7.1e-4 on one real window (mean 7.9e-7), **64,096 of 64,096 int8 inputs identical, predicted class identical for 2,004 of 2,004**. These are training windows, so this tests that the implementation reproduces Python's numbers, not that the model generalises.
- *Cost, from the linked RP2350 image (`firmware/pico2-bringup`, binary `feature_probe`).* About 7.6 KB of flash (10,304 bytes against the 2,728-byte blink image; 6.2 KB of it is the tables) and about 12.4 KB of RAM (`.bss` 12,744 against 328 bytes), of which 8 KB is the extractor's scratch and 4 KB is the probe's test window. Cycles and time are not measured.
- *A stuck sensor is a numerical hazard.* A constant window has no AC content, so every band should sit at the floor (-12). A plain `f32` mean left a rounding residue that produced arithmetic-noise features, so the code uses a two-pass mean and returns exactly -12. The Python pipeline has the same flaw in a milder form (its float32 mean of a constant 0.37 is off by 5.5e-8, which puts band 0 at -9.6); the Rust test checks the exact answer and does not require agreement with that artefact.
- *Decision made:* `f32`, hand-written, not fixed-point and not a library. The RP2350's Cortex-M33 cores should have a single-precision FPU (unconfirmed until the first run), the features then match Python to the figures above, and a fixed-point version would add a second source of feature drift. If the measured cycle cost is too high, fixed-point is the fallback.

**ADXL345 driver (`firmware/tinyadxl`).** A small `no_std` driver for `embedded-hal` 1.0 `SpiDevice`: ID check, configuration (rate, range, full resolution, stream FIFO, watermark interrupt), FIFO level, sample and FIFO reads. It is tested against a simulated chip that models the register file, the 32-entry FIFO and `INT_SOURCE`, rejects framing the chip would not accept, and enforces the datasheet's FIFO timing rule. Writing the test against the rule found a bug in my first version (the 5 microsecond gap was inside the transaction with chip select still low; the driver now releases chip select and waits, via a delay provider). The simulator and the driver share my reading of the chip, so agreement shows consistency only; the register map, bit meanings and timing rule were cross-checked against Linux's `adxl34x` driver and Adafruit's library, but the datasheet PDF itself could not be fetched, and the SPI mode and the 3.9 mg scale remain unverified.

## Phase 2 plan (host-side preparation done; nothing run on a board)

Run the same model on a real microcontroller with a real accelerometer.

- **Board.** A Raspberry Pi Pico 2 (RP2350) is the leading candidate: about $5 for the board (list price per Raspberry Pi's March 2026 product brief), a dual-core Arm Cortex-M33 part with TrustZone, signed boot and one-time-programmable key storage, which also makes it a platform for a later secure-firmware project. Embassy builds for it (checked, see `firmware/pico2-bringup`, not yet run on a board). Whether its floating-point and DSP support suit an FFT is not verified.
- **Sensor.** An SPI or I2C MEMS accelerometer breakout. Its maximum output data rate and bandwidth must be read from the datasheet; the bandwidth study says a few kHz is enough to detect faults and that diagnosis wants more.
- **Measurements to make:** the achieved sample rate and its jitter; cycles per window split into FFT, features and network; flash and RAM from the linker map; energy per window with a current-sense breakout; and numeric agreement between the device's features and the Python features (feature mismatch is the likeliest way a good host model fails on a device).
- **A real rig.** A box fan with weights added, to produce imbalance and misalignment signals that CWRU does not contain, collected under several mountings and speeds. The honest lesson from phase 1 applies: train and test on different conditions.

## Open decisions

- Whether the deployed model should be trained for detection only (healthy vs faulty), given that diagnosis does not generalise. Phase 1 supports detection as the dependable function.
- Whether to add an anomaly-detection formulation (model of "normal" only), which would not need fault examples at all, as a comparison.
- ~~Whether the on-device FFT uses a library or a hand-written fixed-point version~~ Decided: hand-written `f32` (see above); revisit only if the measured cycles are too high.
- The accelerometer: an ADXL345 is on order. At 3,200 Hz it samples near the 3 kHz point of the bandwidth study (detection 100%, diagnosis 93%), but that study used an ideal low-pass filter and the sensor's own analog bandwidth is half its output rate, so the study should be repeated with a closer model of the sensor before trusting the result.

## Known limitations

- Phase 1 is host-only and uses a single lab dataset.
- int4 weights are simulated; the Rust engine implements int8 only.
- Key management, secure update and any security review are out of scope until the secure-firmware project.
