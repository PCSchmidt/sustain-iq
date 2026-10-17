# pico2-bringup

Minimal [Embassy](https://embassy.dev) firmware for the Raspberry Pi Pico 2 (RP2350, Cortex-M33). It blinks the on-board LED at 2 Hz. Its only job is to prove the toolchain: that Rust builds a valid RP2350 image, before the first sensor code is written.

## Build

```sh
rustup target add thumbv8m.main-none-eabihf
cargo build --release
```

## Status

- Builds and links. Release image: 2,728 bytes text, 328 bytes bss, with the RP2350 image-definition block placed after the vector table.
- **Not yet run on a board.** Whether the boot ROM accepts the image, and the flashing route (UF2 drag-and-drop or a debug probe), are confirmed only once the hardware arrives.
- Only the Cortex-M33 side is targeted. The RP2350's RISC-V cores are not used.

`memory.x` carries the standard RP2350 boot-block sections (`.start_block`, `.bi_entries`, `.end_block`); without them the link fails.
