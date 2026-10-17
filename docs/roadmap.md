# Roadmap

| Milestone | Deliverable | Needs | Status |
| --- | --- | --- | --- |
| 0.1 | Host pipeline: CWRU download, windowing, features, leak-aware splits, numpy MLP, integer quantisation | PC | Done |
| 0.2 | Studies: leakage, feature sets, locked-set confirmation, size and precision, sensor bandwidth, with simple baselines | PC | Done |
| 0.3 | Rust integer inference engine, bit-exact with the Python reference (305 of 305 vectors) | PC | Done |
| 0.4 | Device bring-up: board and accelerometer sampling at a fixed rate; measured rate and jitter | Microcontroller board, accelerometer | Planned. Toolchain checked without hardware: `tinyinfer` cross-compiles for Cortex-M33 and a minimal Embassy image builds (`firmware/pico2-bringup`); nothing has run on a board yet |
| 0.5 | On-device features: FFT and band energies in Rust, matched to the Python features | Board | Planned |
| 0.6 | On-device inference with `tinyinfer`: cycles per window, flash, RAM, and energy per window measured | Board, current-sense breakout | Planned |
| 0.7 | Own rig data (fan with added weights): imbalance and misalignment classes, collected across mountings and speeds; honest cross-condition evaluation | Board, rig | Planned |
| 0.8 | Detection-only and anomaly-detection variants compared with the classifier | PC or board | Planned |
| Optional | Gateway and cloud path (MQTT, device certificates, remaining-useful-life model), as in the original project outline | Gateway, cloud account | Planned |

A milestone is done only when its README section leads with measured results.
