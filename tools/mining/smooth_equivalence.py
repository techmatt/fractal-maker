r"""smooth_equivalence.py — how far a STRANGE-mode render is from its own location's
SMOOTH render, on the colored-CLIP substrate.

WHY. Matt's mid-labeling verdict on the (27) mining sheet was that many strange modes are
smooth-equivalent at many locations — the same picture arriving under a different mode name.
That is a duplicate of something already judged, it wastes a sheet row, and at release time
it makes "% smooth vs % unique-mode" an unmeasurable dial. Nothing in the pipeline measured
it, because every existing dedup instrument is deliberately blind to exactly this axis:

  * the LIBRARY / emission near-dup (`descriptor.NEAR_DUP_THRESHOLD`, 0.974) runs on the
    `morph_gray` grayscale morphology embedding of a location's SMOOTH field. Mode and
    palette are invisible to it BY DESIGN — it answers "same place", and two modes at one
    location are the same place whatever they look like. Its cosine is identically 1 here.
  * `library_dedup` is coordinates.
  * the mining head answers "is this good", not "is this new".

So the substrate is `colored_clip` (palette-ON CLIP, `vit_base_patch16_clip_224.openai`,
the same model and timm eval transform as the grayscale morphology canon — the two land in
ONE embedding space, which is the whole reason that producer exists). The pair compared is:

    cos( CLIP(mode render) , CLIP(smooth render) )

with the SAME location, SAME palette, SAME colour params, SAME geometry, SAME render tail.
The ONLY difference between the two frames is the mode, so the cosine is a measure of the
mode and of nothing else. The smooth twin is rendered through the SAME
`build_mining_sheet.render_one` the strange row went through, at `MR.SMOOTH_MODE` — not a
second render path, and not a cached field from another geometry.

WHAT THE CUT MEANS, AND WHAT IT DOES NOT (read before quoting a share)
---------------------------------------------------------------------
`STRICT_CUT` is `descriptor.NEAR_DUP_THRESHOLD` (0.974) and `IDENTITY_INTERLEAVE` is Matt's
own same-vs-distinct interleave zone [0.934, 0.986] from the precanon calibration
(`data/atlas/precanon_calibration/adoption.json`). BOTH were derived on the GRAYSCALE morph
embedding, and the tree already records that the grayscale anchors "do not transfer"
(`steered_frontier`, `morph_anchor_calibrate`). Applied to colored CLIP they are a BORROWED
yardstick, not a calibrated one. They are used anyway, and reported as such, because they
are the project's one human-anchored near-dup scale and a fresh calibration is a labeling
sitting nobody has bought. Two things keep the number honest:

  * `unrelated_reference()` — the cosine distribution between renders of DIFFERENT locations
    on this same substrate. It is the floor the scale actually has here, and without it a
    "0.97" is unreadable: colored CLIP over one palette family is a much narrower band than
    grayscale CLIP over arbitrary morphology.
  * the full quantiles are reported beside every share, so a reader can move the cut.

    from tools.mining.smooth_equivalence import Embedder, cos_to_smooth, band_of
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "queries",
           ROOT / "tools" / "emission"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission.descriptor import NEAR_DUP_THRESHOLD    # noqa: E402  THE near-dup owner

# The strict near-dup cut, imported not re-typed.
STRICT_CUT = NEAR_DUP_THRESHOLD

# Matt's same-vs-distinct interleave zone, recorded verbatim from the precanon calibration's
# adoption record. A pair inside it is one HE would have called either way; a per-mode share
# that lands here is a statement that the mode is sometimes a duplicate, not that it is one.
IDENTITY_INTERLEAVE = (0.934, 0.986)
INTERLEAVE_SOURCE = "data/atlas/precanon_calibration/adoption.json (morph_gray-derived)"


def band_of(cos: float) -> str:
    """`near_dup` (>= strict) / `interleave` (inside Matt's zone, below strict) /
    `distinct` (below the zone). The bands are ORDERED and the strict cut sits INSIDE the
    interleave zone, so `near_dup` is checked first and the three are disjoint."""
    if cos >= STRICT_CUT:
        return "near_dup"
    if cos >= IDENTITY_INTERLEAVE[0]:
        return "interleave"
    return "distinct"


# --------------------------------------------------------------------------- #
# The substrate.
# --------------------------------------------------------------------------- #
class Embedder:
    """colored-CLIP over image FILES, unit-normalized, batched, cached by path.

    Wraps `tools.curation.colored_clip.load_clip/embed_clip` — the producer that defines the
    substrate — so there is exactly one place where the model and transform are chosen.
    Constructed lazily: importing this module must not cost a GPU stack."""

    def __init__(self):
        from tools.curation import colored_clip as cc
        self._cc = cc
        self.model, self.tf = cc.load_clip()
        self.device = cc.DEV
        self.model_name = cc.CLIP_MODEL
        self._cache: dict[str, np.ndarray] = {}

    def embed_paths(self, paths, batch: int = 32) -> np.ndarray:
        """`(N, D)` unit-normalized rows, in the order given. Missing files raise — a
        silently-skipped crop would shift every downstream row's pairing by one."""
        from PIL import Image
        paths = [str(p) for p in paths]
        todo = [p for p in dict.fromkeys(paths) if p not in self._cache]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            imgs = []
            for p in chunk:
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB").copy())
            vecs = self._cc.embed_clip(self.model, self.tf, imgs)
            for p, v in zip(chunk, vecs):
                v = np.asarray(v, dtype=np.float32)
                self._cache[p] = v / (np.linalg.norm(v) + 1e-9)
        return np.stack([self._cache[p] for p in paths])


def cos_to_smooth(mode_vecs: np.ndarray, smooth_vecs: np.ndarray) -> np.ndarray:
    """Row-wise cosine of two unit-normalized stacks (a mode render and ITS OWN location's
    smooth twin, paired by index)."""
    return np.einsum("ij,ij->i", mode_vecs, smooth_vecs).astype(np.float64)


def unrelated_reference(vecs: np.ndarray, seed: int = 0, n_pairs: int = 20000) -> dict:
    """The cosine distribution over pairs of DIFFERENT rows on this substrate — the scale's
    floor, without which a near-dup share is unreadable.

    Deliberately NOT the full pairwise matrix: a seeded sample of index pairs is enough for
    quantiles and does not grow with N**2. Pairs are drawn i != j."""
    rng = np.random.default_rng(seed)
    n = len(vecs)
    if n < 2:
        return {"n_pairs": 0}
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n - 1, size=n_pairs)
    j = np.where(j >= i, j + 1, j)                      # i != j, uniform over the rest
    c = np.einsum("ij,ij->i", vecs[i], vecs[j]).astype(np.float64)
    return {"n_pairs": int(n_pairs), **quantiles(c),
            "share_ge_strict": float((c >= STRICT_CUT).mean())}


def quantiles(c) -> dict:
    c = np.asarray(c, dtype=np.float64)
    if c.size == 0:
        return {"n": 0}
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    return {"n": int(c.size), "mean": float(c.mean()),
            "q": {f"p{int(q*100)}": float(np.quantile(c, q)) for q in qs},
            "min": float(c.min()), "max": float(c.max())}


def per_group_table(groups, cosines) -> dict:
    """`{group: {n, share_near_dup, share_interleave, share_distinct, median}}`.

    Shares are OF THE GROUP, so a mode with 12 rows and a mode with 200 are both readable;
    `n` is carried on every row because a share without its population is the error this
    project keeps finding."""
    out: dict[str, dict] = {}
    idx: dict[str, list] = {}
    for k, c in zip(groups, cosines):
        idx.setdefault(k, []).append(float(c))
    for k in sorted(idx):
        c = np.asarray(idx[k], dtype=np.float64)
        bands = [band_of(x) for x in c]
        out[k] = {
            "n": int(c.size),
            "share_near_dup": float(sum(b == "near_dup" for b in bands) / c.size),
            "share_interleave": float(sum(b == "interleave" for b in bands) / c.size),
            "share_distinct": float(sum(b == "distinct" for b in bands) / c.size),
            "median_cos": float(np.median(c)),
            "p05_cos": float(np.quantile(c, 0.05)),
            "p95_cos": float(np.quantile(c, 0.95)),
        }
    return out


def yardstick_block() -> dict:
    """The provenance every consumer stamps beside a share computed here."""
    return {
        "substrate": "colored_clip — palette-ON CLIP vit_base_patch16_clip_224.openai, "
                     "timm eval transform (tools/curation/colored_clip.py)",
        "pair": "the mode render vs a SMOOTH render of the same location at the same "
                "palette, colour params, geometry and render tail (mining_roster.SMOOTH_MODE "
                "through build_mining_sheet.render_one) — the mode is the only difference",
        "strict_cut": STRICT_CUT,
        "strict_cut_owner": "tools/emission/descriptor.NEAR_DUP_THRESHOLD",
        "identity_interleave": list(IDENTITY_INTERLEAVE),
        "identity_interleave_source": INTERLEAVE_SOURCE,
        "caveat": "BOTH anchors were derived on the GRAYSCALE morph embedding and the tree "
                  "records that grayscale anchors do not transfer. On colored CLIP they are "
                  "a borrowed yardstick, not a calibrated one — read every share next to "
                  "`unrelated_reference`, which is the scale's actual floor here.",
    }
