//! Integer-only MLP inference, `no_std` and allocation-free, for microcontrollers.
//!
//! This is a direct transcription of the Python reference in `sustain_iq.quant`:
//!
//! * inputs and hidden activations are `i8`; weights are `i8`; biases are `i32`
//! * a layer computes `acc = bias + sum(weight * input)` in `i32`
//! * a hidden layer requantises with a fixed-point multiplier and clamps to `0..=127`
//!   (that clamp is the ReLU): `y = clamp((acc * m0 + (1 << (shift - 1))) >> shift, 0, 127)`
//! * the output layer returns `acc` itself; the class is the index of its largest value
//!   (the lowest index wins a tie, as in numpy's `argmax`)
//!
//! Only integer operations are used, so any correct implementation produces identical
//! results; `tests/golden.rs` checks that against vectors exported from Python.
//!
//! Memory use is two caller-provided `i8` buffers (as wide as the widest layer) and one
//! `i32` slice for the logits. Nothing is measured here: cycle counts and energy are for
//! the on-device phase.

#![no_std]

/// One dense layer. `weights` is row-major, `out_dim` rows of `in_dim`.
#[derive(Clone, Copy)]
pub struct Layer<'a> {
    pub weights: &'a [i8],
    pub bias: &'a [i32],
    pub in_dim: usize,
    pub out_dim: usize,
    /// Requantisation multiplier (a 31-bit integer) and right shift; ignored for the output layer.
    pub m0: i32,
    pub shift: u32,
}

/// Requantise an accumulator to `0..=127`: round-half-up fixed-point multiply, then clamp.
#[inline]
pub fn requantize(acc: i32, m0: i32, shift: u32) -> i8 {
    let v = ((acc as i64 * m0 as i64) + (1i64 << (shift - 1))) >> shift;
    v.clamp(0, 127) as i8
}

impl<'a> Layer<'a> {
    #[inline]
    fn accumulate(&self, x: &[i8], o: usize) -> i32 {
        let row = &self.weights[o * self.in_dim..(o + 1) * self.in_dim];
        let mut acc = self.bias[o];
        for (w, v) in row.iter().zip(x.iter()) {
            acc += (*w as i32) * (*v as i32);
        }
        acc
    }

    /// Hidden layer: writes `out_dim` values to `out`.
    pub fn forward_hidden(&self, x: &[i8], out: &mut [i8]) {
        for (o, slot) in out.iter_mut().take(self.out_dim).enumerate() {
            *slot = requantize(self.accumulate(x, o), self.m0, self.shift);
        }
    }

    /// Output layer: writes the raw `i32` logits to `out`.
    pub fn forward_output(&self, x: &[i8], out: &mut [i32]) {
        for (o, slot) in out.iter_mut().take(self.out_dim).enumerate() {
            *slot = self.accumulate(x, o);
        }
    }
}

/// Index of the largest value; the first one wins a tie.
pub fn argmax(v: &[i32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// A stack of layers: hidden layers followed by one output layer.
pub struct Mlp<'a> {
    pub layers: &'a [Layer<'a>],
}

impl<'a> Mlp<'a> {
    /// Classify one quantised input. `buf_a` and `buf_b` must each hold the widest
    /// layer (including the input); `logits` must hold the output width. Returns the class
    /// and leaves the logits in `logits`.
    pub fn predict(
        &self,
        input: &[i8],
        buf_a: &mut [i8],
        buf_b: &mut [i8],
        logits: &mut [i32],
    ) -> usize {
        let last = self.layers.len() - 1;
        buf_a[..input.len()].copy_from_slice(input);
        let mut in_a = true; // the current activations live in buf_a
        for (i, layer) in self.layers.iter().enumerate() {
            let (src, dst): (&[i8], &mut [i8]) = if in_a {
                (&*buf_a, &mut *buf_b)
            } else {
                (&*buf_b, &mut *buf_a)
            };
            let src = &src[..layer.in_dim];
            if i == last {
                layer.forward_output(src, logits);
            } else {
                layer.forward_hidden(src, dst);
                in_a = !in_a;
            }
        }
        argmax(&logits[..self.layers[last].out_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requantize_rounds_half_up_and_clamps() {
        // m0 / 2^shift = 0.5 : (1 << 30) / 2^31
        let (m0, shift) = (1 << 30, 31);
        assert_eq!(requantize(0, m0, shift), 0);
        assert_eq!(requantize(1, m0, shift), 1); // 0.5 rounds up
        assert_eq!(requantize(3, m0, shift), 2); // 1.5 rounds up
        assert_eq!(requantize(10, m0, shift), 5);
        assert_eq!(requantize(-10, m0, shift), 0); // negative: ReLU clamp
        assert_eq!(requantize(1000, m0, shift), 127); // saturates
        assert_eq!(requantize(i32::MAX, m0, shift), 127);
        assert_eq!(requantize(i32::MIN, m0, shift), 0);
    }

    #[test]
    fn argmax_takes_the_first_of_equal_maxima() {
        assert_eq!(argmax(&[1, 5, 5, 2]), 1);
        assert_eq!(argmax(&[-3, -3]), 0);
        assert_eq!(argmax(&[7]), 0);
    }

    #[test]
    fn a_two_layer_network_computes_what_it_should() {
        // hidden: 2 -> 2, identity-ish weights, scale 1:  y = relu(x) after (acc * 2^30 + 2^30) >> 30 = acc
        let w1 = [1i8, 0, 0, 1];
        let b1 = [0i32, 0];
        let w2 = [1i8, -1, -1, 1]; // output: [x0 - x1, x1 - x0]
        let b2 = [0i32, 0];
        let layers = [
            Layer {
                weights: &w1,
                bias: &b1,
                in_dim: 2,
                out_dim: 2,
                m0: 1 << 30,
                shift: 30,
            },
            Layer {
                weights: &w2,
                bias: &b2,
                in_dim: 2,
                out_dim: 2,
                m0: 0,
                shift: 0,
            },
        ];
        let net = Mlp { layers: &layers };
        let (mut a, mut b, mut logits) = ([0i8; 2], [0i8; 2], [0i32; 2]);
        assert_eq!(net.predict(&[10, 3], &mut a, &mut b, &mut logits), 0);
        assert_eq!(logits, [7, -7]);
        assert_eq!(net.predict(&[-5, 4], &mut a, &mut b, &mut logits), 1); // -5 is clamped to 0 by the hidden ReLU
        assert_eq!(logits, [-4, 4]);
    }
}
