"""Plan 0017 rev 4: two-class deception probes on the sibling-probe machinery (imported, not
copied). Pure numpy: this module is shipped into the Modal job with the package source.

A probe is plan 0003's recipe V2 (shrinkage-whitened DiffMean, ``probes.cross_fit_whitened``,
lambda by inner CV) with 2 concepts (0 = honest, 1 = deceptive / lie; the score is concept
1's), every position pooled under one centring key (ending = offset = 0), and plan 0006's null z
(``um_probe.null_stats``: per fold model, z over that fold's OOF honest positions). Eval
positions are scored by every fold model and the z averaged over models.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hallucination_bench.evals.sibling_probe import probes as P
from hallucination_bench.evals.sibling_probe import um_probe as U

N_FOLDS = 4
VALIDITY_MIN = 0.90  # pre-registered: OOF text-level AUROC on the probe's own data
DECEPTIVE = 1


@dataclass
class Probe:
    ts: P.TextSet
    models: list[P.FoldModel]
    null: U.NullStats
    lams: list[float]
    oof_z: np.ndarray  # [rows] concept-1 z, each row by its own fold's model and null

    def text_mean_z(self) -> np.ndarray:
        T = len(self.ts.ids)
        s = np.bincount(self.ts.text, weights=np.nan_to_num(self.oof_z), minlength=T)
        n = np.bincount(self.ts.text, weights=np.isfinite(self.oof_z).astype(float), minlength=T)
        return s / np.maximum(n, 1)

    def summary(self) -> dict[str, Any]:
        y = self.ts.concept
        tm = self.text_mean_z()
        rc = self.ts.row_concept
        z = self.oof_z
        return {
            "auroc_text": P.auroc(tm[y == DECEPTIVE], tm[y == 0]),
            "auroc_pos": P.auroc(z[rc == DECEPTIVE], z[rc == 0]),
            "tau_pos": P.q95(z[rc == 0]),
            "tau_text": P.q95(tm[y == 0]),
            "lams": self.lams,
            "n_texts": len(y),
            "n_pos": len(z),
        }

    def score(self, H: np.ndarray) -> np.ndarray:
        """[P] fold-averaged concept-1 z of new positions."""
        if not len(H):
            return np.zeros(0)
        e = np.zeros(len(H), dtype=np.int64)
        zs = [self.null.z(k, m.score(H, e, e), e)[:, DECEPTIVE] for k, m in enumerate(self.models)]
        return np.nanmean(zs, axis=0)


def fit(
    windows: Sequence[np.ndarray],
    labels: Sequence[int],
    folds: Sequence[int],
    ids: Sequence[str] | None = None,
    n_folds: int = N_FOLDS,
    lams: Sequence[float] = P.LAMS_W,
) -> Probe:
    """Cross-fit a two-class probe on texts ``windows[i]`` [n_i, d] with class ``labels[i]``
    (0 honest, 1 deceptive) and fold ``folds[i]``."""
    n = len(windows)
    ts = P.TextSet.build(
        windows,
        ids=list(ids) if ids is not None else [str(i) for i in range(n)],
        concept=list(labels),
        kind=["implied"] * n,
        ending=[0] * n,
        fold=list(folds),
        offsets=[[0] * int(w.shape[0]) for w in windows],
    )
    cf, lam, _ = P.cross_fit_whitened(ts, 2, n_folds, lams=lams)
    ns = U.null_stats(ts, cf.oof, 2, n_folds)
    z = U.oof_z(ts, cf.oof, ns, n_folds)[:, DECEPTIVE]
    return Probe(ts=ts, models=cf.models, null=ns, lams=lam, oof_z=z)
