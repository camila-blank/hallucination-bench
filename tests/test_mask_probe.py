"""Plan 0017 rev 4: the two-class deception probe on synthetic data."""

import numpy as np

from hallucination_bench.evals.mask import probe as PR


def _data(seed=0, n=40, d=16, shift=3.0):
    rng = np.random.default_rng(seed)
    w, y, f = [], [], []
    for i in range(n):
        c = i % 2
        k = rng.integers(2, 5)
        x = rng.normal(size=(k, d))
        x[:, 0] += shift * c
        w.append(x.astype(np.float32))
        y.append(c)
        f.append((i // 2) % 4)  # both members of a pair share the fold
    return w, y, f


def test_probe_separates_and_scores():
    w, y, f = _data()
    p = PR.fit(w, y, f, lams=(0.1, 1.0))
    s = p.summary()
    assert s["auroc_text"] > 0.95 and s["auroc_pos"] > 0.9
    assert s["n_texts"] == 40 and len(p.oof_z) == s["n_pos"]
    hi = np.zeros((3, 16), np.float32)
    hi[:, 0] = 3.0
    lo = np.zeros((3, 16), np.float32)
    assert (p.score(hi) > p.score(lo)).all()
    assert np.isfinite(s["tau_pos"]) and np.isfinite(s["tau_text"])


def test_probe_null_on_noise():
    w, y, f = _data(seed=1, shift=0.0)
    s = PR.fit(w, y, f, lams=(1.0,)).summary()
    assert 0.2 < s["auroc_text"] < 0.8
