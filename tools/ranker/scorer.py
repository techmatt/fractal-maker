#!/usr/bin/env python
"""Location preference ranker — scorer entry point.

Deployed head: **pref_loc_v1** (v7+colored:logi, refit on run2+dive+campaign1 = 379 labels,
3-batch LOBO meanSp +0.436, certified). v1 has the
same feature blocks (v7+colored) and affine shape as v0, so every consumer is unchanged by the
flip. pref_loc_v0/model.npz remains on disk for rollback.

Loads the deployed linear head (`data/ranker/pref_loc_v1/model.npz`) and maps a joined frozen
feature record -> a scalar rank score (higher == more likely human-good). The head is a plain
affine map on standardized features, so scoring is dependency-light (no torch): standardize with
the stored (mean, scale), dot with W, add b.

    >>> from tools.ranker.scorer import RankerScorer
    >>> s = RankerScorer.load()
    >>> s.score_matrix({"morph": M, "v7": V, "colored": C})   # each (N, dim)

============================ HARD SCOPE — READ BEFORE WIRING ============================
This head ranks the NOT-BAD; it must NEVER steer discovery. Do NOT wire RankerScorer into
frontier priority (steered_frontier.py), dive-start selection (--dive selection), production
seeding, or any generation-side decision. A model that both selects and ranks degrades on its own
selections — that is exactly how canonical p_good became a badness filter rather than a goodness
ranker (docs/design/aesthetic_scoring.md §2). Legitimate consumers ONLY: keeper
ranking, emission feed ordering, and dive-result sorting — all of which rank an already-produced
set without feeding back into what gets produced.
========================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "data" / "ranker" / "pref_loc_v1" / "model.npz"

# =========================================================================== #
# FROZEN FEATURE EXTRACTOR — PINNED, deliberately NOT the active checkpoint.
#
# `RankerScorer.load()` itself resolves nothing but the affine head in the .npz — it never
# touches a checkpoint. The exposure was one layer out, in the two places that BUILD the `v7`
# feature block: `build_features.py --scorer` defaulted to `production_seeder.SCORER_PATH`, and
# `score_locations._ensure_stack` constructed `Scorer(ps.SCORER_PATH)`. Both resolve to
# `active_ckpt.ACTIVE_CKPT`, so flipping the discovery gate silently repointed the ranker's
# frozen features at the new head.
#
# That failure would have been SILENT, which is what makes it worth a pin rather than a comment.
# pref_loc_v1's W was fit on v7-penultimate features; v8 shares the backbone
# (mobilenetv4_conv_medium), so a v8 penultimate is also 1280-D and every shape check, every
# standardize, every dot product would succeed. The head would just be applying v7-fit weights
# to a differently-organized feature space and returning confident nonsense — and its
# certification (3-batch LOBO meanSp +0.436) would no longer describe the deployed object at
# all, while still being quoted as if it did.
#
# So it is pinned to the v7 penultimate, with NO refit. A refit is a separate, deliberate act
# that must be re-certified; it is not something a discovery-gate flip is allowed to perform as
# a side effect. When a refit does happen, move this pin and re-certify in the same change.
PENULTIMATE_CKPT = ROOT / "data" / "classifier" / "v7" / "model_best.pt"
PENULTIMATE_VERSION = PENULTIMATE_CKPT.parent.name          # "v7"
# The feature-block name in `sets` this checkpoint produces. Kept equal to the version token so
# a future refit cannot move one without the other going visibly inconsistent.
PENULTIMATE_BLOCK = PENULTIMATE_VERSION


def penultimate_scorer():
    """The PINNED frozen feature extractor for the deployed ranker head.

    Every ranker-side caller must build its `Scorer` through here rather than through
    `production_seeder.SCORER_PATH` / `active_ckpt.ACTIVE_CKPT`. Raises rather than falling
    back if the pinned weight is missing — a ranker silently scoring on a different head is
    worse than a ranker that does not run."""
    import sys as _sys
    if str(ROOT / "tools" / "mining") not in _sys.path:
        _sys.path.insert(0, str(ROOT / "tools" / "mining"))
    from score_lib import Scorer
    if not PENULTIMATE_CKPT.exists():
        raise SystemExit(
            f"ranker feature extractor missing: {PENULTIMATE_CKPT}. pref_loc_v1's features are "
            f"PINNED to the {PENULTIMATE_VERSION} penultimate (see PENULTIMATE_CKPT) and must "
            f"not fall back to the active checkpoint — that would silently repoint the frozen "
            f"features and void the head's certification.")
    return Scorer(str(PENULTIMATE_CKPT))


class RankerScorer:
    def __init__(self, mean, scale, W, b, sets, head, reg, use_prior):
        self.mean = np.asarray(mean, np.float64)
        self.scale = np.asarray(scale, np.float64)
        self.W = np.asarray(W, np.float64)
        self.b = float(b)
        self.sets = [str(s) for s in sets]      # feature blocks, in concat order
        self.head = str(head)
        self.reg = float(reg)
        self.use_prior = bool(use_prior)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL):
        z = np.load(path, allow_pickle=True)
        return cls(z["mean"], z["scale"], z["W"], z["b"], z["sets"], z["head"],
                   z["reg"], bool(z["use_prior"]))

    def _stack(self, blocks: dict) -> np.ndarray:
        return np.concatenate([np.atleast_2d(blocks[b]).astype(np.float64) for b in self.sets],
                              axis=1)

    def score_matrix(self, blocks: dict) -> np.ndarray:
        """blocks: {'morph': (N,768), 'v7': (N,1280), 'colored': (N,768)} (only the deployed
        blocks are required). Returns (N,) rank scores."""
        X = self._stack(blocks)
        Xs = (X - self.mean) / self.scale
        return Xs @ self.W + self.b


def _cli():
    """Score the deployed head's features.npz and print id, score, human (if labeled)."""
    feat = ROOT / "data/ranker/pref_loc_v1/features.npz"
    z = np.load(feat, allow_pickle=True)
    s = RankerScorer.load()
    blocks = {b: z[b] for b in s.sets}
    sc = s.score_matrix(blocks)
    order = np.argsort(-sc)
    print(f"# ranker  head={s.head} sets={s.sets} prior={s.use_prior}")
    for i in order:
        hs = int(z["score"][i])
        print(f"{sc[i]:+.4f}  {z['ids'][i]:44s} {z['family'][i]:16s} "
              f"human={hs if hs else '-'}")


if __name__ == "__main__":
    _cli()
