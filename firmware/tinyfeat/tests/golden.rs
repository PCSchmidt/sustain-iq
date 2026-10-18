//! Agreement with the Python features on real CWRU windows, and end to end through `tinyinfer`.
//!
//! `features_golden.json` (written by `scripts/export_feature_golden.py`) holds raw windows and, for each,
//! the float64 features, int8 input and class that the Python pipeline produces for the deployed model in
//! `tinyinfer/tests/golden.json`. The Rust code works in `f32`, so exact equality is not expected for the
//! features or (rarely) the int8 inputs: what matters, and what these tests bound, is how far they
//! drift and whether the predicted class changes.

use serde::Deserialize;
use tinyfeat::{standardize_quantize, Extractor, N_BANDS, WINDOW};
use tinyinfer::{Layer, Mlp};

const CONSTANT_WINDOW: &str = "edge:constant_offset";

#[derive(Deserialize)]
struct Golden {
    windows: Vec<Vec<f32>>,
    features: Vec<Vec<f64>>,
    xq: Vec<Vec<i8>>,
    pred: Vec<usize>,
    source: Vec<String>,
}

#[derive(Deserialize)]
struct Standardizer {
    mean: Vec<f32>,
    std: Vec<f32>,
}

#[derive(Deserialize)]
struct LayerJson {
    #[serde(rename = "in")]
    in_dim: usize,
    out: usize,
    w: Vec<i8>,
    b: Vec<i32>,
    m0: i32,
    shift: u32,
}

#[derive(Deserialize)]
struct Model {
    input_scale: f32,
    standardizer: Standardizer,
    layers: Vec<LayerJson>,
}

/// The window golden file: `TINYFEAT_GOLDEN` if set (a larger local file), else the small committed one.
fn window_golden() -> Golden {
    match std::env::var("TINYFEAT_GOLDEN") {
        Ok(p) => {
            serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|_| panic!("{p}")))
                .unwrap()
        }
        Err(_) => read("tests/features_golden.json"),
    }
}

fn read<T: for<'de> Deserialize<'de>>(rel: &str) -> T {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), rel);
    serde_json::from_str(
        &std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("{path}: run the export scripts")),
    )
    .unwrap()
}

fn extract_all(g: &Golden) -> Vec<[f32; N_BANDS]> {
    let mut ex = Extractor::new();
    g.windows
        .iter()
        .map(|w| {
            let samples: [f32; WINDOW] = w.as_slice().try_into().expect("1024 samples");
            let mut out = [0.0f32; N_BANDS];
            ex.extract(&samples, &mut out);
            out
        })
        .collect()
}

#[test]
fn features_match_python_within_a_small_bound() {
    let g: Golden = window_golden();
    assert!(g.windows.len() >= 50);
    // (max error, where, sum, count) for real windows and for the synthetic edge windows
    let mut real = (0.0f64, String::new(), 0.0f64, 0usize);
    let mut edge = (0.0f64, String::new(), 0.0f64, 0usize);
    for (i, f) in extract_all(&g).iter().enumerate() {
        if g.source[i] == CONSTANT_WINDOW {
            // The true features of a constant window are all -12 (no AC power, only the 1e-12 floor). Python's
            // float32 mean is off by 5.5e-8 for this window, leaving a DC residue that shows up as -9.6 in band 0,
            // so Python is the one that is wrong here; check Rust against the mathematically correct value.
            let worst = f
                .iter()
                .map(|v| (*v as f64 + 12.0).abs())
                .fold(0.0, f64::max);
            println!("constant window: Rust is within {worst:.2e} of the exact -12 in every band (Python band 0 is {:.3})", g.features[i][0]);
            assert!(worst < 1e-3, "constant window features are not -12");
            continue;
        }
        let slot = if g.source[i].starts_with("edge:") {
            &mut edge
        } else {
            &mut real
        };
        for (b, (&got, &want)) in f.iter().zip(g.features[i].iter()).enumerate() {
            let d = (got as f64 - want).abs();
            slot.2 += d;
            slot.3 += 1;
            if d > slot.0 {
                slot.0 = d;
                slot.1 = format!("{} band {b}", g.source[i]);
            }
        }
    }
    println!(
        "real windows: feature error vs Python (log10 units) max {:.2e} at {}, mean {:.2e}",
        real.0,
        real.1,
        real.2 / real.3 as f64
    );
    println!(
        "edge windows: max {:.2e} at {}, mean {:.2e}",
        edge.0,
        edge.1,
        edge.2 / edge.3 as f64
    );
    assert!(
        real.0 < 2e-3,
        "max feature error on real windows {:.3e} at {}",
        real.0,
        real.1
    );
    assert!(
        edge.0 < 1e-3,
        "max feature error on edge windows {:.3e} at {}",
        edge.0,
        edge.1
    );
}

#[test]
fn int8_inputs_and_predicted_classes_agree_with_python() {
    let g: Golden = window_golden();
    let m: Model = read("../tinyinfer/tests/golden.json");
    let mean: [f32; N_BANDS] = m.standardizer.mean.as_slice().try_into().unwrap();
    let std: [f32; N_BANDS] = m.standardizer.std.as_slice().try_into().unwrap();

    let layers: Vec<Layer> = m
        .layers
        .iter()
        .map(|l| Layer {
            weights: &l.w,
            bias: &l.b,
            in_dim: l.in_dim,
            out_dim: l.out,
            m0: l.m0,
            shift: l.shift,
        })
        .collect();
    let net = Mlp { layers: &layers };
    let width = m.layers.iter().map(|l| l.out.max(l.in_dim)).max().unwrap();
    let (mut a, mut b, mut logits) = (
        vec![0i8; width],
        vec![0i8; width],
        vec![0i32; m.layers.last().unwrap().out],
    );

    let (mut exact, mut off_by_one, mut worse, mut class_same) = (0usize, 0usize, 0usize, 0usize);
    let feats = extract_all(&g);
    for (i, f) in feats.iter().enumerate() {
        let mut xq = [0i8; N_BANDS];
        standardize_quantize(f, &mean, &std, m.input_scale, &mut xq);
        for (&got, &want) in xq.iter().zip(g.xq[i].iter()) {
            if g.source[i] == CONSTANT_WINDOW {
                continue; // Python's own features are arithmetic noise here; see the feature test
            }
            match (got as i32 - want as i32).abs() {
                0 => exact += 1,
                1 => off_by_one += 1,
                _ => worse += 1,
            }
        }
        let class = net.predict(&xq, &mut a, &mut b, &mut logits);
        if class == g.pred[i] {
            class_same += 1;
        } else {
            println!(
                "class differs for {}: rust {class}, python {}",
                g.source[i], g.pred[i]
            );
        }
    }
    let total = (feats.len() - 1) * N_BANDS; // the constant window's inputs are not compared
    println!(
        "int8 inputs: {exact}/{total} identical, {off_by_one} off by one, {worse} off by more; classes: {class_same}/{} agree",
        feats.len()
    );
    assert_eq!(worse, 0, "an int8 input differs by more than one step");
    assert!(
        off_by_one * 100 <= total,
        "more than 1% of int8 inputs are off by one"
    );
    assert_eq!(class_same, feats.len(), "a predicted class changed");
}
