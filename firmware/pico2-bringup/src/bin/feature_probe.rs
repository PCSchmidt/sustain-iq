//! Link check for the feature extractor on the RP2350: build a synthetic 1,024-sample window, run
//! `tinyfeat`, and light the LED when the first band is above a threshold.
//!
//! The window is a triangle wave plus pseudo-random noise, so this proves only that the extractor
//! compiles, links and fits on the chip, and gives its flash and RAM cost from the image. It has not
//! run on a board and its timing is unknown.
#![no_std]
#![no_main]

use core::hint::black_box;

use embassy_executor::Spawner;
use embassy_rp::block::ImageDef;
use embassy_rp::gpio::{Level, Output};
use embassy_time::Timer;
use panic_halt as _;
use tinyfeat::{Extractor, N_BANDS, WINDOW};

#[link_section = ".start_block"]
#[used]
pub static IMAGE_DEF: ImageDef = ImageDef::secure_exe();

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    let mut led = Output::new(p.PIN_25, Level::Low);

    let mut window = [0.0f32; WINDOW];
    let mut seed = 1u32;
    for (i, w) in window.iter_mut().enumerate() {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (seed >> 16) as f32 / 65_536.0 - 0.5;
        *w = ((i % 64) as f32 - 32.0) / 32.0 + 0.1 * noise;
    }

    let mut extractor = Extractor::new();
    let mut features = [0.0f32; N_BANDS];
    loop {
        extractor.extract(black_box(&window), &mut features);
        led.set_level(if black_box(features[0]) > 0.0 { Level::High } else { Level::Low });
        Timer::after_millis(500).await;
    }
}
