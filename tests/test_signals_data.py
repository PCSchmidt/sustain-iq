import numpy as np
import pytest
from scipy.io import savemat

from sustain_iq.data import (
    CLASSES,
    Dataset,
    balanced_accuracy,
    load_de,
    split_leave_one_load_out,
    split_leave_one_size_out,
    split_random,
)
from sustain_iq.signals import band_edges, band_features, decimate, fft_butterflies, windows


def test_windows_cover_the_signal_with_the_requested_stride():
    x = np.arange(3000, dtype=np.float32)
    w, starts = windows(x, size=1024, stride=512)
    assert w.shape == (4, 1024) and list(starts) == [0, 512, 1024, 1536]
    assert np.array_equal(w[1], x[512:1536])
    empty, s = windows(x[:100])
    assert empty.shape[0] == 0 and s.size == 0


def test_band_edges_are_strictly_increasing_and_span_the_spectrum():
    e = band_edges(513, 32)
    assert len(e) == 33 and e[0] == 1 and e[-1] == 513 and np.all(np.diff(e) >= 1)
    assert np.diff(e)[0] == 1 and np.diff(e)[-1] > 50  # one bin wide at the bottom, wide at the top
    with pytest.raises(ValueError):
        band_edges(10, 32)


def test_a_tone_lights_up_the_band_that_contains_its_frequency():
    fs, n = 12_000, 1024
    t = np.arange(n) / fs
    lo = band_features(np.sin(2 * np.pi * 300 * t)[None, :])[0]
    hi = band_features(np.sin(2 * np.pi * 3000 * t)[None, :])[0]
    assert lo.argmax() < hi.argmax()
    assert lo.max() - np.median(lo) > 3  # clearly above the other bands (log10 units)


def test_features_ignore_dc_offset_and_scale_additively_in_log_power():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(5, 1024))
    f0 = band_features(w)
    assert np.allclose(band_features(w + 7.0), f0, atol=1e-6)
    assert np.allclose(band_features(w * 10.0), f0 + 2.0, atol=1e-6)  # x10 amplitude = +100x power = +2 in log10


def test_decimate_reduces_length_and_removes_content_above_the_new_nyquist():
    fs = 12_000
    t = np.arange(24_000) / fs
    x = (np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 5000 * t)).astype(np.float32)
    y = decimate(x, 4)  # new fs 3000, Nyquist 1500: the 5 kHz tone must go
    assert len(y) == 6000
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 4 / fs)
    assert freqs[spec.argmax()] == pytest.approx(200, abs=1.0)
    assert spec[freqs > 1000].max() < 0.01 * spec.max()
    assert decimate(x, 1) is x
    with pytest.raises(ValueError):
        decimate(x, 0)


def test_fft_butterflies():
    assert fft_butterflies(1024) == 512 * 10


def test_load_de_picks_the_variable_for_the_file_and_refuses_to_guess(tmp_path):
    a, b = np.arange(5.0), np.arange(5.0) + 100
    savemat(tmp_path / "99.mat", {"X098_DE_time": a, "X099_DE_time": b, "X099_FE_time": b})
    assert np.array_equal(load_de(tmp_path / "99.mat", 99), b.astype(np.float32))
    with pytest.raises(ValueError):  # two candidates and no file number: do not guess
        load_de(tmp_path / "99.mat")
    savemat(tmp_path / "105.mat", {"X105_DE_time": a})
    assert np.array_equal(load_de(tmp_path / "105.mat"), a.astype(np.float32))


def fake_dataset():
    rows = []
    for rec_id, (label, y, load, diam, length) in enumerate([
        ("N0", 0, 0, 0.0, 6000), ("N1", 0, 1, 0.0, 6000),
        ("IR_a0", 1, 0, 0.007, 3000), ("IR_a1", 1, 1, 0.007, 3000),
        ("IR_b0", 1, 0, 0.014, 3000), ("IR_b1", 1, 1, 0.014, 3000),
    ]):
        starts = np.arange(0, length - 1024 + 1, 512)
        for s in starts:
            rows.append((y, rec_id, load, diam, s, length))
    y, rec, load, diam, start, length = map(np.array, zip(*rows))
    return Dataset(np.zeros((len(y), 4)), y, rec, load, diam, start, length)


def test_leave_one_load_out_keeps_whole_recordings_apart():
    ds = fake_dataset()
    tr, te = split_leave_one_load_out(ds, load=1)
    assert set(ds.load[tr]) == {0} and set(ds.load[te]) == {1}
    assert not set(tr) & set(te)


def test_leave_one_size_out_holds_out_a_diameter_and_time_splits_normal_with_a_gap():
    ds = fake_dataset()
    tr, te = split_leave_one_size_out(ds, diameter=0.014)
    fault_tr = tr[ds.y[tr] != 0]
    fault_te = te[ds.y[te] != 0]
    assert set(ds.diameter[fault_tr]) == {0.007} and set(ds.diameter[fault_te]) == {0.014}
    n_tr, n_te = tr[ds.y[tr] == 0], te[ds.y[te] == 0]
    assert len(n_tr) > 0 and len(n_te) > 0 and not set(n_tr) & set(n_te)
    # every train window ends well before any test window of the same recording starts
    assert (ds.start[n_tr] + 1024).max() < ds.start[n_te].min()


def test_random_split_is_leaky_by_construction():
    ds = fake_dataset()
    tr, te = split_random(ds, test_frac=0.3, seed=1)
    assert not set(tr) & set(te) and len(tr) + len(te) == len(ds.y)
    # the leak: every recording has windows on both sides, so neighbouring near-duplicates
    # of the same recording are in train and test
    both = [r for r in np.unique(ds.rec) if np.any(ds.rec[tr] == r) and np.any(ds.rec[te] == r)]
    assert len(both) >= 4


def test_balanced_accuracy_averages_recall_over_present_classes():
    y = np.array([0, 0, 0, 0, 1, 1, 2])
    p = np.array([0, 0, 0, 0, 1, 0, 2])
    assert balanced_accuracy(y, p) == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert len(CLASSES) == 4


def test_shape_features_are_exactly_invariant_to_amplitude():
    from sustain_iq.signals import shape_features

    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 1024))
    assert np.allclose(shape_features(w), shape_features(w * 37.0), atol=1e-9)
    assert np.allclose(shape_features(w).mean(axis=1), 0, atol=1e-9)


def test_envelope_features_find_the_modulation_frequency_not_the_carrier():
    from sustain_iq.signals import envelope_band_features, highpass

    fs, n = 12_000, 4096
    t = np.arange(n) / fs
    carrier = np.sin(2 * np.pi * 3000 * t)
    bursts = (1 + 0.9 * np.sin(2 * np.pi * 107 * t)) * carrier  # 3 kHz resonance modulated at ~107 Hz
    steady = carrier
    hp = lambda x: highpass(x.astype(np.float32), fs)[None, 1500:2524]  # skip the filter transient
    mod, flat = envelope_band_features(hp(bursts))[0], envelope_band_features(hp(steady))[0]
    # 107 Hz sits near bin 9 at 11.7 Hz/bin: group (9-2)//2 = 3
    assert mod.argmax() in (2, 3, 4)
    assert mod.max() - np.median(mod) > 1.0  # a clear envelope line
    assert flat.max() - np.median(flat) < mod.max() - np.median(mod)


def test_feature_set_validation_and_envelope_needs_full_rate():
    from sustain_iq.data import Recording, build_dataset

    rec = Recording("x", "IR", 0.007, 0, np.random.default_rng(0).normal(size=12_000).astype(np.float32))
    with pytest.raises(ValueError):
        build_dataset([rec], features="mfcc")
    with pytest.raises(ValueError):
        build_dataset([rec], factor=2, features="envelope")
    for f in ("abs", "shape", "envelope"):
        ds = build_dataset([rec], features=f)
        assert ds.X.shape[1] == 32 and ds.X.shape[0] == (12_000 - 1024) // 512 + 1
