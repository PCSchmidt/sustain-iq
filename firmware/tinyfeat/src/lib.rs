//! Log band-energy features for a 1,024-sample vibration window, `no_std` and allocation-free.
//!
//! The same computation as `sustain_iq.signals.band_features`:
//!
//! 1. remove the window's mean
//! 2. multiply by the symmetric Hann window `np.hanning(1024)`
//! 3. FFT (here a radix-2 decimation-in-time complex FFT on `f32`, imaginary part zero)
//! 4. power `re^2 + im^2` for bins 0..=512
//! 5. sum the power in 32 bands that are log-spaced in bin index (edges in `tables::BAND_EDGES`)
//! 6. `log10(sum + 1e-12)`
//!
//! Then [`standardize_quantize`] turns the 32 features into the `i8` input of the `tinyinfer` network.
//!
//! The tables (twiddles, window, band edges) are generated from the Python definitions by
//! `scripts/gen_tinyfeat_tables.py`; `tests/golden.rs` checks the result against Python on real CWRU
//! windows. The code is a straightforward complex FFT of the real signal: it does about twice the work
//! of a packed real FFT. Nothing here is timed; cycle counts are for the on-device phase.

#![no_std]

#[rustfmt::skip]
pub mod tables;

use tables::{BAND_EDGES, HANN_HALF, TWIDDLE_COS, TWIDDLE_SIN_NEG};

pub const WINDOW: usize = 1024;
pub const N_BANDS: usize = 32;
const LOG2_WINDOW: u32 = 10;

/// Scratch space for one extraction: 8 KiB (two `f32` arrays of 1,024).
pub struct Extractor {
    re: [f32; WINDOW],
    im: [f32; WINDOW],
}

impl Default for Extractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Extractor {
    pub const fn new() -> Self {
        Self {
            re: [0.0; WINDOW],
            im: [0.0; WINDOW],
        }
    }

    /// Features of one window of samples, written to `out`.
    pub fn extract(&mut self, samples: &[f32; WINDOW], out: &mut [f32; N_BANDS]) {
        // Two-pass mean: a plain f32 sum of 1,024 values loses enough precision that a constant window
        // (a stuck or saturated sensor) would keep a residue of rounding error after mean removal, and the
        // features of that window would be arithmetic noise. The second pass averages the residuals.
        let mut sum = 0.0f32;
        for &s in samples.iter() {
            sum += s;
        }
        let rough = sum / WINDOW as f32;
        let mut resid = 0.0f32;
        for &s in samples.iter() {
            resid += s - rough;
        }
        let mean = rough + resid / WINDOW as f32;

        // mean removal and Hann window, written in bit-reversed order for the in-place FFT
        for (k, &s) in samples.iter().enumerate() {
            let w = if k < WINDOW / 2 {
                HANN_HALF[k]
            } else {
                HANN_HALF[WINDOW - 1 - k]
            };
            let dst = (k as u32).reverse_bits() >> (32 - LOG2_WINDOW);
            self.re[dst as usize] = (s - mean) * w;
            self.im[dst as usize] = 0.0;
        }
        self.fft_bit_reversed();

        for (b, slot) in out.iter_mut().enumerate() {
            let mut power = 0.0f32;
            for k in BAND_EDGES[b] as usize..BAND_EDGES[b + 1] as usize {
                power += self.re[k] * self.re[k] + self.im[k] * self.im[k];
            }
            *slot = libm::log10f(power + 1e-12);
        }
    }

    /// Iterative radix-2 decimation-in-time FFT; input already in bit-reversed order.
    fn fft_bit_reversed(&mut self) {
        let mut len = 2;
        while len <= WINDOW {
            let half = len / 2;
            let step = WINDOW / len;
            let mut i = 0;
            while i < WINDOW {
                for j in 0..half {
                    let (wr, wi) = (TWIDDLE_COS[j * step], TWIDDLE_SIN_NEG[j * step]);
                    let (a, b) = (i + j, i + j + half);
                    let (vr, vi) = (
                        self.re[b] * wr - self.im[b] * wi,
                        self.re[b] * wi + self.im[b] * wr,
                    );
                    let (ur, ui) = (self.re[a], self.im[a]);
                    self.re[a] = ur + vr;
                    self.im[a] = ui + vi;
                    self.re[b] = ur - vr;
                    self.im[b] = ui - vi;
                }
                i += len;
            }
            len *= 2;
        }
    }
}

/// Standardise (`(f - mean) / std`), scale by the model's input scale, round half to even (as numpy does)
/// and clamp to `-127..=127`: the `i8` input vector of the `tinyinfer` network.
pub fn standardize_quantize(
    features: &[f32; N_BANDS],
    mean: &[f32; N_BANDS],
    std: &[f32; N_BANDS],
    input_scale: f32,
    out: &mut [i8; N_BANDS],
) {
    for i in 0..N_BANDS {
        let v = libm::rintf((features[i] - mean[i]) / std[i] / input_scale);
        out[i] = v.clamp(-127.0, 127.0) as i8;
    }
}
