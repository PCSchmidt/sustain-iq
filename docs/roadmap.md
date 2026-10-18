# Roadmap

| Milestone | Deliverable | Needs | Status |
| --- | --- | --- | --- |
| 0.1 | Host pipeline: CWRU download, windowing, features, leak-aware splits, numpy MLP, integer quantisation | PC | Done |
| 0.2 | Studies: leakage, feature sets, locked-set confirmation, size and precision, sensor bandwidth, with simple baselines | PC | Done |
| 0.3 | Rust integer inference engine, bit-exact with the Python reference (305 of 305 vectors) | PC | Done |
| 0.4 | Device bring-up: board and accelerometer sampling at a fixed rate; measured rate and jitter | Microcontroller board, accelerometer | Planned. Prepared without hardware: toolchain checked, ADXL345 driver (`tinyadxl`) written and tested against a simulated chip, and a [bring-up checklist](bringup-checklist.md). Nothing has run on a board |
| 0.5 | On-device features: FFT and band energies in Rust, matched to the Python features | Board | Host side done: `tinyfeat` reproduces the Python features and int8 inputs on 2,004 windows (64,096 of 64,096 int8 values identical); links for the RP2350 at about 7.6 KB flash. Not run on a board; cycles unmeasured |
| 0.6 | On-device inference with `tinyinfer`: cycles per window, flash, RAM, and energy per window measured | Board, current-sense breakout | Planned |
| 0.7 | Own rig data (fan with added weights): imbalance and misalignment classes, collected across mountings and speeds; honest cross-condition evaluation | Board, rig | Planned |
| 0.8 | Detection-only and anomaly-detection variants compared with the classifier | PC or board | Planned |
| Optional | Gateway and cloud path (MQTT, device certificates, remaining-useful-life model), as in the original project outline | Gateway, cloud account | Planned |

A milestone is done only when its README section leads with measured results.
