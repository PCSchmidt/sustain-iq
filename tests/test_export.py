import json

import numpy as np

from sustain_iq.export import edge_case_inputs, golden_dict
from sustain_iq.mlp import MLP, Standardizer
from sustain_iq.quant import quantize


def small_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=400)
    X = rng.normal(size=(400, 6)) + np.eye(4)[y] @ rng.normal(size=(4, 6)) * 2
    std = Standardizer.fit(X)
    Xs = std(X)
    m = MLP([6, 8, 8, 4], seed=0).fit(Xs, y, epochs=30, seed=0)
    return quantize(m, Xs), std, Xs


def test_edge_cases_cover_extremes():
    e = edge_case_inputs(6)
    assert e.shape == (5, 6) and e.dtype == np.int8
    assert (e[0] == 127).all() and (e[1] == -127).all() and (e[2] == 0).all()
    assert (e[3] == -e[4]).all()


def test_golden_file_round_trips_and_is_reproducible_from_plain_integers():
    q, std, Xs = small_model()
    g = json.loads(json.dumps(golden_dict(q, std, Xs, n_samples=40)))
    assert len(g["inputs"]) == 40 + 5 and len(g["logits"]) == len(g["pred"]) == 45
    layers = g["layers"]
    for x, expected, pred in zip(g["inputs"], g["logits"], g["pred"]):
        h = list(x)
        for i, L in enumerate(layers):
            w = np.array(L["w"]).reshape(L["out"], L["in"])
            acc = [int(sum(int(a) * int(b) for a, b in zip(row, h)) + L["b"][o]) for o, row in enumerate(w)]
            if i < len(layers) - 1:
                h = [min(max((a * L["m0"] + (1 << (L["shift"] - 1))) >> L["shift"], 0), 127) for a in acc]
            else:
                assert acc == expected
                assert int(np.argmax(acc)) == pred


def test_standardizer_constants_are_exported():
    q, std, Xs = small_model()
    g = golden_dict(q, std, Xs, n_samples=5)
    assert np.allclose(g["standardizer"]["mean"], std.mean) and len(g["standardizer"]["std"]) == 6
