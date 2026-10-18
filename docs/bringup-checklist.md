# Bring-up checklist (for when the board and sensor arrive)

Nothing below has been done on hardware. The host-side code that this checklist exercises is described in [design.md](design.md#host-side-preparation-for-phase-2). Each step has a pass condition, so a result can be recorded as measured, not remembered.

## What is and is not known before starting

| Item | State |
| --- | --- |
| Rust builds for the RP2350 (Cortex-M33, `thumbv8m.main-none-eabihf`) and an Embassy blink image links | Done on the host; **not run on a board** |
| `tinyfeat` features agree with the Python features | Done on the host (see design.md) |
| `tinyadxl` register map, command bits, ODR codes, 5 MHz limit, FIFO timing rule | Cross-checked against Linux's `adxl34x` driver and Adafruit's `Adafruit_ADXL345`; the datasheet PDF could not be fetched |
| ADXL345 needs SPI mode 3 | **Unverified** (believed from the datasheet; neither source above states it) |
| Scale of 3.9 mg per count in full-resolution mode | **Unverified** (Adafruit's library rounds to 4 mg) |
| The Cortex-M33 in the RP2350 has a single-precision FPU, so the `hard-float` target is valid | **Unverified**; the first run of `feature_probe` will tell (a missing FPU faults) |
| Anything about timing, jitter, cycles, energy | **Unmeasured** |

## Steps

1. **Flash the blink image** (`firmware/pico2-bringup`, `cargo build --release`). Pass: the on-board LED blinks at 2 Hz. This also settles how images get onto the board (UF2 copy or a debug probe) and whether the boot ROM accepts the image-definition block.
2. **Wire the ADXL345 for SPI** (chip select, clock, MOSI, MISO, 3.3 V, ground, and INT1 to a free GPIO). Check the breakout's own documentation for how it selects SPI instead of I2C; I have not confirmed how SparkFun's and Adafruit's boards do it. Record the pins used.
3. **Read DEVID.** Open the bus at 1 MHz, mode 3, read register 0x00. Pass: 0xE5. If it fails, try mode 0 and record which one works, because that settles the mode question.
4. **Run `Adxl345::init`** at 3,200 Hz, +-16 g, watermark 16, and read the FIFO level and interrupt source. Pass: no error, and the level rises.
5. **Achieved sample rate.** Drain the FIFO for 10 seconds and count samples. Pass: close to 32,000. Record the exact count.
6. **Jitter and loss.** Timestamp each watermark interrupt (16 samples is 5.0 ms at 3,200 Hz) with the RP2350 timer. Record min, max and standard deviation of the interval, and the number of overruns reported by `interrupt_source`. Pass: zero overruns over 10 minutes.
7. **FIFO timing rule.** At 5 MHz with `read_fifo` (which waits 5 microseconds after each sample), confirm clean data. Then, as an experiment only, remove the wait and look for corrupted or repeated samples. This confirms that the rule matters on this chip.
8. **Scale and orientation.** Lay the board flat. Pass: one axis reads about +-256 counts (1 g at 3.9 mg per count), the others near zero. Record the real value, then decide whether 3.9 mg per count is right.
9. **Run `feature_probe`.** Pass: the LED lights as designed (a missing FPU would fault here). Then time `Extractor::extract` with the Cortex-M33's cycle counter on a real 1,024-sample window, and record cycles and the share of the sample period that is.
10. **Compare features with the host.** Stream one captured window out over serial, compute the Python features for the same samples, and compare with the device's. Pass: agreement of the order seen on the host (worst 1e-3 in log10 units). A disagreement here is the likeliest way a good model fails on a device.
11. **Then the model:** generate Rust constants from `tinyinfer/tests/golden.json`, run the whole chain on device, and compare the class with the host's for 100 windows. Record cycles, flash and RAM from the linker map.
12. **Energy:** add the INA219 in line with the supply and record energy per window (milestone 0.6).

## Things to watch for

- The ADXL345's full-resolution mode reports 13-bit values; check they sign-extend correctly at both ends of the range.
- A stuck or saturated sensor produces a constant window. `tinyfeat` returns the exact answer (all bands at the 1e-12 floor, i.e. -12) for that, which the standardiser then maps far from any training value; decide what the firmware should do (flag a sensor fault), and test it.
- The bandwidth study decimated CWRU with an ideal low-pass filter. The real sensor's analog bandwidth is half its output rate (per Adafruit's library comments), which is a different, softer filter: re-check detection on a decimated-then-quantised replay before trusting the 3,200 Hz result.
