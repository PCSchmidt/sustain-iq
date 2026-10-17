//! Bit-exact agreement with the Python reference.
//!
//! `golden.json` is written by `scripts/export_golden.py`: a trained int8 model's integers,
//! plus real and edge-case inputs with the logits and class that `sustain_iq.quant`
//! produces. Every logit must match exactly, not approximately.

use serde::Deserialize;
use tinyinfer::{Layer, Mlp};

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
struct Golden {
    layers: Vec<LayerJson>,
    inputs: Vec<Vec<i8>>,
    logits: Vec<Vec<i32>>,
    pred: Vec<usize>,
}

fn load() -> Golden {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden.json");
    serde_json::from_str(
        &std::fs::read_to_string(path).expect("golden.json (run scripts/export_golden.py)"),
    )
    .unwrap()
}

#[test]
fn every_logit_and_class_matches_the_python_reference() {
    let g = load();
    let layers: Vec<Layer> = g
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
    let width = g.layers.iter().map(|l| l.out.max(l.in_dim)).max().unwrap();
    let n_out = g.layers.last().unwrap().out;
    let (mut a, mut b, mut logits) = (vec![0i8; width], vec![0i8; width], vec![0i32; n_out]);

    assert!(
        g.inputs.len() > 100,
        "expected a few hundred vectors, got {}",
        g.inputs.len()
    );
    for (k, input) in g.inputs.iter().enumerate() {
        let class = net.predict(input, &mut a, &mut b, &mut logits);
        assert_eq!(logits, g.logits[k], "logits differ for vector {k}");
        assert_eq!(class, g.pred[k], "class differs for vector {k}");
    }
}

#[test]
fn the_exported_model_has_the_expected_shape() {
    let g = load();
    let dims: Vec<(usize, usize)> = g.layers.iter().map(|l| (l.in_dim, l.out)).collect();
    assert_eq!(dims, vec![(32, 32), (32, 16), (16, 4)]);
    for l in &g.layers {
        assert_eq!(l.w.len(), l.in_dim * l.out);
        assert_eq!(l.b.len(), l.out);
    }
    // Hidden layers carry a 31-bit multiplier; the output layer has none.
    assert!(g.layers[..2]
        .iter()
        .all(|l| l.m0 >= 1 << 30 && l.shift >= 1));
    assert_eq!((g.layers[2].m0, g.layers[2].shift), (0, 0));
}

#[test]
fn flash_and_ram_budget_of_the_exported_model() {
    let g = load();
    let weights: usize = g.layers.iter().map(|l| l.w.len()).sum();
    let biases: usize = g.layers.iter().map(|l| l.b.len()).sum();
    let flash = weights + 4 * biases + 8 * (g.layers.len() - 1);
    let widest = g.layers.iter().map(|l| l.out.max(l.in_dim)).max().unwrap();
    let ram = g.layers[0].in_dim + 2 * widest;
    // Accounting only (not measured): the whole model is a couple of kilobytes.
    assert_eq!((weights, biases), (32 * 32 + 32 * 16 + 16 * 4, 32 + 16 + 4));
    assert!(flash < 2_500, "flash {flash} bytes");
    assert!(ram < 128, "ram {ram} bytes");
}
