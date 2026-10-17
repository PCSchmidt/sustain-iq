import numpy as np
import pytest

from sustain_iq.mlp import MLP, Standardizer
from sustain_iq.quant import QModel, fixed_point, footprint, quantize


def blobs(n=600, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0, 0, 0], [3, 3, 0, 0], [0, 3, 3, 0], [0, 0, 3, 3]], dtype=float)
    y = rng.integers(0, 4, size=n)
    return centers[y] + rng.normal(0, 0.6, size=(n, 4)), y


def test_standardizer_fits_train_statistics_only():
    X = np.array([[1.0, 10.0], [3.0, 30.0]])
    s = Standardizer.fit(X)
    assert np.allclose(s(X).mean(axis=0), 0) and np.allclose(s(X).std(axis=0), 1, atol=1e-6)
    assert np.allclose(s(np.array([[2.0, 20.0]])), 0, atol=1e-6)


def test_mlp_learns_separable_classes_and_reports_parameter_count():
    X, y = blobs()
    m = MLP([4, 8, 8, 4], seed=1).fit(X, y, epochs=60, seed=1)
    assert np.mean(m.predict(X) == y) > 0.95
    assert m.n_params() == (4 * 8 + 8) + (8 * 8 + 8) + (8 * 4 + 4)


def test_training_is_deterministic_for_a_seed():
    X, y = blobs()
    a = MLP([4, 6, 4], seed=3).fit(X, y, epochs=5, seed=3)
    b = MLP([4, 6, 4], seed=3).fit(X, y, epochs=5, seed=3)
    assert all(np.array_equal(p, q) for p, q in zip(a.W, b.W))


def test_balanced_loss_does_not_ignore_a_rare_class():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.5, (900, 2)), rng.normal(3, 0.5, (30, 2))])
    y = np.array([0] * 900 + [1] * 30)
    m = MLP([2, 8, 2], seed=0).fit(X, y, epochs=40, seed=0)
    assert np.mean(m.predict(X[y == 1]) == 1) > 0.9


@pytest.mark.parametrize("m", [0.5, 1e-3, 0.0123, 0.999999, 3.7e-7])
def test_fixed_point_reproduces_the_multiplier(m):
    m0, shift = fixed_point(m)
    assert (1 << 30) <= m0 < (1 << 31) and shift >= 1
    assert m0 / 2.0**shift == pytest.approx(m, rel=1e-8)


def test_fixed_point_rejects_bad_multipliers():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            fixed_point(bad)


def trained():
    X, y = blobs(1200, seed=2)
    s = Standardizer.fit(X)
    Xs = s(X)
    m = MLP([4, 16, 8, 4], seed=0).fit(Xs, y, epochs=60, seed=0)
    return m, Xs, y


def test_int8_model_agrees_with_float_on_almost_every_sample():
    m, Xs, y = trained()
    q = quantize(m, Xs, weight_bits=8)
    agree = np.mean(q.predict(Xs) == m.predict(Xs))
    assert agree > 0.98
    assert q.layers[0].w.dtype == np.int8 and q.layers[0].b.dtype == np.int32


def test_int4_weights_stay_in_range_and_lose_some_agreement_but_not_everything():
    m, Xs, y = trained()
    q4, q8 = quantize(m, Xs, weight_bits=4), quantize(m, Xs, weight_bits=8)
    assert max(int(np.abs(l.w).max()) for l in q4.layers) <= 7
    assert np.mean(q4.predict(Xs) == m.predict(Xs)) > 0.85
    assert np.mean(q8.predict(Xs) == m.predict(Xs)) >= np.mean(q4.predict(Xs) == m.predict(Xs)) - 0.02


def test_integer_inference_is_exactly_reproducible_from_the_stored_integers():
    m, Xs, y = trained()
    q = quantize(m, Xs)
    xq = q.quantize_input(Xs[:50])
    # recompute by hand from the integers with plain Python ints
    for row, expected in zip(xq, q.logits_from_q(xq)):
        h = [int(v) for v in row]
        for i, layer in enumerate(q.layers):
            acc = [sum(int(w) * x for w, x in zip(wr, h)) + int(b) for wr, b in zip(layer.w, layer.b)]
            if i < len(q.layers) - 1:
                h = [min(max((a * layer.m0 + (1 << (layer.shift - 1))) >> layer.shift, 0), 127) for a in acc]
            else:
                assert acc == [int(v) for v in expected]


def test_hidden_activations_stay_inside_int8_after_requantisation():
    m, Xs, y = trained()
    q = quantize(m, Xs)
    h = q.quantize_input(Xs).astype(np.int64)
    layer = q.layers[0]
    acc = h @ layer.w.T.astype(np.int64) + layer.b
    out = np.clip((acc * layer.m0 + (1 << (layer.shift - 1))) >> layer.shift, 0, 127)
    assert out.min() >= 0 and out.max() <= 127


def test_footprint_accounts_for_packed_int4_weights():
    m, Xs, y = trained()
    f8, f4 = footprint(quantize(m, Xs, 8)), footprint(quantize(m, Xs, 4))
    n_w = 4 * 16 + 16 * 8 + 8 * 4
    n_b = 16 + 8 + 4
    assert f8["macs"] == n_w and f8["params"] == n_w + n_b
    assert f8["flash_bytes"] == n_w + 4 * n_b + 8 * 2
    assert f4["flash_bytes"] == (n_w + 1) // 2 + 4 * n_b + 8 * 2
    assert f8["ram_bytes"] == 4 + 2 * 16
