"""heads.py — THE four live heads, their pins, their eval material and their scorers.

One place so the quantization tooling stays head-agnostic: `quantize_head.py` and
`eval_quant.py` both take a `HeadSpec` and never learn which head they hold.

**Every checkpoint here is RESOLVED from that head's own pin module**, never restated:
`production_pins.ACTIVE_CKPT`, `wallpaper_pins.HEAD_CKPT_REL`,
`mining_pins.ACTIVE_MINING_CKPT`, `scorer.data.ACTIVE_SCORER_DIR`. A pin flip moves this
file's inputs with it and a stale literal here would quantize a head that is no longer
serving. Nothing is written to any of them — this whole tree is read-only on the pins.

THE THREE CORN HEADS GO THROUGH ONE LOADER, `tools/scoring/eval_model.load_model`, which
builds the backbone, K, geometry, interpolation and mean/std off the checkpoint's OWN config.
What differs between them and is NOT in the config is the marginal convention their own
battery reads:

  * location  — `expit(logits)` IS the cumulative rank probability (tools/backbone_search/
    eval_arms.py, tools/v11's battery).
  * wallpaper, mining — `cumprod(sigmoid(logits))`, the conditional-to-marginal chain
    (report_v4_eval.load_head, mining_gate.MiningScorer).

Both are monotone in the same logits, and the difference matters here only because the
agreement numbers must be on the scale each head's own gate cuts. It is carried as
`HeadSpec.marginal` rather than inferred.

The pref head is a different animal — a single-tower RANKING-margin head (one scalar per
frame, comparable only WITHIN a location) on `mobilenetv4_conv_small`, with its own deploy
transform (squash 224). It gets its own scorer and its own metrics.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "scoring",
           ROOT / "tools" / "queries", ROOT / "tools" / "queries" / "scorer"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# The location head's PRIMARY population: eval rows that touch neither training nor a
# checkpoint pick. Same two sources the backbone study excluded, named there as
# SELECTION_SOURCES — restated here rather than imported so this module does not drag the
# study's scipy/matplotlib tail in for two strings.
SELECTION_SOURCES = ("prospect_census", "loose0_v3_floor")


@dataclass(frozen=True)
class QItem:
    """One scoreable unit. `path` for the three crop-backed heads, `img` for pref's
    in-memory recolors — a candidate frame has no file, it is a recolor of a cached field."""
    key: str
    group: str
    label: int | None = None
    path: str | None = None
    img: object = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class HeadSpec:
    key: str
    label: str
    pin: str                      # the module attribute the ckpt is resolved from
    ckpt_rel: str                 # resolved AT IMPORT from that pin
    kind: str                     # "corn" | "ranker"
    marginal: str                 # "sigmoid" | "cumprod" | "scalar"
    material: str                 # what population this head is measured on
    gate_threshold: float | None  # the deployed cut on p_ge3, where one exists
    gate_pin: str | None

    def items(self, limit: int | None = None):
        return _MATERIAL[self.key](limit)

    def sample(self, n: int):
        """A small slice of the SAME material, for the round-trip re-score. Small on
        purpose: the round-trip proves the file, the acceptance run proves the rung."""
        return _MATERIAL[self.key](n)


def _location_ckpt() -> str:
    import production_pins as PP
    return PP.ACTIVE_CKPT


def _wallpaper_ckpt() -> str:
    from tools.wallpaper import wallpaper_pins as WP
    return WP.HEAD_CKPT_REL


def _wallpaper_gate() -> float:
    from tools.wallpaper import wallpaper_pins as WP
    return WP.GATE_THRESHOLD


def _mining_ckpt() -> str:
    from tools.mining import mining_pins as MP
    return MP.ACTIVE_MINING_CKPT


def _mining_gate() -> float:
    from tools.mining import mining_pins as MP
    return MP.MINING_GATE_THRESHOLD


def _pref_ckpt() -> str:
    from tools.queries.scorer import data as SD
    return (Path(SD.ACTIVE_SCORER_DIR).relative_to(ROOT) / "model_best.pt").as_posix()


# --------------------------------------------------------------------------- #
# material
# --------------------------------------------------------------------------- #
def _location_material(limit=None):
    """The 2,190 PRIMARY v11 eval locations at the deploy-canonical render."""
    import partitions as P
    from classifier.data_v11 import load_locations_v11

    locs = load_locations_v11(verify_paths=False)
    ev = sorted((l for l in locs if l.split == "eval"), key=lambda l: l.location_id)
    prim = [l for l in ev if l.source not in SELECTION_SOURCES]
    if limit:
        prim = prim[:limit]
    return [QItem(key=str(l.location_id),
                  # the cluster the paired bootstrap resamples over; a location outside any
                  # leakage group is its own cluster, negated so it cannot collide with a
                  # real group id (the convention eval_arms.py uses)
                  group=str(l.split_group if l.split_group is not None else -l.location_id),
                  label=l.label, path=str(l.canonical().path),
                  extra={"partition": P.partition_of(l.fractal_type), "source": l.source,
                         "fractal_type": l.fractal_type})
            for l in prim]


def _wallpaper_material(limit=None):
    """Sheet D — the blind minibrot rows. READ-ONLY use of an eval instrument."""
    from tools.wallpaper.sheet_d_reverdict import load_rows

    rows, _meta = load_rows()
    if limit:
        rows = rows[:limit]
    return [QItem(key=r.image_id, group=r.image_id, label=r.label, path=str(r.jpg),
                  extra={"vein": r.vein, "partition": r.partition, "family": r.family})
            for r in rows]


def _mining_material(limit=None):
    """Sheet E — the blind render-mode rows. READ-ONLY, same standing as sheet D."""
    from tools.mining.sheet_e_reverdict import load_rows

    rows, _meta = load_rows()
    if limit:
        rows = rows[:limit]
    return [QItem(key=r.image_id, group=r.loc, label=r.label, path=str(r.jpg),
                  extra={"mode": r.mode, "kind": r.kind, "partition": r.partition,
                         "family": r.family})
            for r in rows]


def _pref_material(limit=None):
    """The pref head's REAL candidate sets, recolored off each location's cached field.

    There is no committed eval instrument for this head: the query records and images its
    1,100 labeled passes key into are gone from disk (`tools/audit/durability_map.py`, "the
    PREF-HEAD JOIN" — the labels survive and their join went). What DOES survive is the
    production side of the same question: `data/library/library_records.jsonl` holds 47
    locations x 12 `palette_candidates[]`, each with the durable coloring recipe it was
    ranked under and the `pref_score`/`pref_rank` the deployed head gave it. So the draw is
    the candidate sets the head actually served, re-rendered through the committed recipe —
    `colored_clip.render_candidates`, the live recolor path, not a reimplementation.

    `limit` is a LOCATION limit, not a frame limit: a candidate set is the unit here, and
    half a location's candidates cannot answer a within-location ranking question."""
    import json

    from PIL import Image  # noqa: F401  (render_candidates returns PIL)

    from tools import colormap as cm
    from tools.curation import colored_clip as cc

    recs = [json.loads(l) for l in
            (ROOT / "data/library/library_records.jsonl").read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    recs.sort(key=lambda r: r["location_id"])
    if limit:
        recs = recs[:limit]
    lib = cm.PaletteLibrary(str(cc.POOL_COLORMAPS), str(cc.FEATURES))
    items = []
    for rec in recs:
        keys, imgs = cc.render_candidates(rec, lib)
        by_var = {c["variant_id"]: c for c in rec["palette_candidates"]}
        for k, im in zip(keys, imgs):
            var = k.split("/", 1)[1]
            cand = by_var[var]
            items.append(QItem(key=k, group=rec["location_id"], label=None,
                               img=np.asarray(im.convert("RGB"), dtype=np.uint8),
                               extra={"variant_id": var,
                                      "palette": cand["palette_ref"]["name"],
                                      "pref_rank": cand.get("pref_rank"),
                                      "pref_score_recorded": cand.get("pref_score")}))
    return items


_MATERIAL = {
    "location": _location_material,
    "wallpaper": _wallpaper_material,
    "mining": _mining_material,
    "pref": _pref_material,
}


# --------------------------------------------------------------------------- #
# scorers
# --------------------------------------------------------------------------- #
class _PathSet:
    """JPGs on disk -> deploy-transformed tensors, index-carrying so a shuffled loader could
    not silently misalign rows. Mirrors `classifier.train_v4._RenderSet`; not imported from
    there because that module pulls the whole v4 train harness in for six lines."""

    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            im.load()
            return self.transform(im.convert("RGB")), i


class CornScorer:
    """A CORN ordinal head + its deploy transform, scored deterministically in fp32.

    NO autocast, on purpose: `MiningScorer` documents the same choice, and a half-precision
    forward pass would put a second quantization inside the arm that is measuring the first."""

    def __init__(self, spec: HeadSpec, device=None):
        import torch
        from eval_model import load_model

        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tf, self.K, self.cfg = load_model(ROOT / spec.ckpt_rel, self.device)
        self.model.eval()

    def score(self, items, batch_size: int = 64, num_workers: int | None = None):
        """`(P, rank)` — P is (N, K-1) cumulative rank probabilities on this head's OWN
        marginal convention; rank is the monotone sum-of-sigmoids score.

        Decode runs in DataLoader workers above a few hundred rows and in-process below it:
        worker startup is seconds on Windows and the round-trip samples are ~50 rows, where
        that is most of the cost."""
        import torch
        from torch.utils.data import DataLoader

        n = len(items)
        if num_workers is None:
            num_workers = 4 if n >= 256 else 0
        logits = np.zeros((n, self.K - 1), dtype=np.float64)
        loader = DataLoader(_PathSet([it.path for it in items], self.tf),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=(self.device == "cuda"))
        with torch.no_grad():
            for x, idx in loader:
                logits[idx.numpy()] = self.model(
                    x.to(self.device, non_blocking=True)).float().cpu().numpy()
        del loader
        cond = 1.0 / (1.0 + np.exp(-logits))
        P = np.cumprod(cond, axis=1) if self.spec.marginal == "cumprod" else cond
        return P, cond.sum(axis=1)


class PrefScorer:
    """The pref-v3-gvo single-tower ranking head. One scalar per frame; comparable only
    within a location, which is why every pref metric downstream is within-`group`."""

    def __init__(self, spec: HeadSpec, device=None):
        import torch

        from tools.queries.scorer import data as SD
        from tools.queries.scorer import train as ST

        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ST.build_model().to(self.device)
        ck = torch.load(ROOT / spec.ckpt_rel, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.tf = SD.build_transform(train=False)
        self.K = 2                       # one scalar -> one "probability-shaped" column
        self.cfg = ck.get("config", {})

    def score(self, items, batch_size: int = 64):
        """`(P, rank)` with P = (N,1) holding the raw utility, so the generic delta code
        upstream reads the same shape for every head. There is no probability here to
        threshold — the head is a ranker."""
        import torch
        from PIL import Image

        out = np.zeros(len(items), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(items), batch_size):
                chunk = items[i:i + batch_size]
                x = torch.stack([self.tf(Image.fromarray(it.img).convert("RGB"))
                                 for it in chunk]).to(self.device)
                out[i:i + len(chunk)] = self.model(x).view(-1).float().cpu().numpy()
        return out.reshape(-1, 1), out


def load_scorer(spec: HeadSpec, device=None):
    return PrefScorer(spec, device) if spec.kind == "ranker" else CornScorer(spec, device)


HEADS = {
    "location": HeadSpec(
        key="location", label="v11 location head (mnv4_conv_medium, CORN K=4)",
        pin="tools/scoring/production_pins.ACTIVE_CKPT", ckpt_rel=_location_ckpt(),
        kind="corn", marginal="sigmoid",
        material="PRIMARY: 2,190 v11 eval locations at the deploy-canonical render",
        gate_threshold=None, gate_pin=None),
    "wallpaper": HeadSpec(
        key="wallpaper", label="wallpaper v4b seed-1 (mnv4_conv_medium, CORN K=4)",
        pin="tools/wallpaper/wallpaper_pins.HEAD_CKPT_REL", ckpt_rel=_wallpaper_ckpt(),
        kind="corn", marginal="cumprod",
        material="sheet D — 2026-08-11_wallpaper_blind_minibrot_v1, blind, eval-only",
        gate_threshold=_wallpaper_gate(),
        gate_pin="tools/wallpaper/wallpaper_pins.GATE_THRESHOLD"),
    "mining": HeadSpec(
        key="mining", label="mining v3 render-mode head (mnv4_conv_small, CORN K=3)",
        pin="tools/mining/mining_pins.ACTIVE_MINING_CKPT", ckpt_rel=_mining_ckpt(),
        kind="corn", marginal="cumprod",
        material="sheet E — 2026-08-11_render_mode_blind_v1, blind, eval-only",
        gate_threshold=_mining_gate(),
        gate_pin="tools/mining/mining_pins.MINING_GATE_THRESHOLD"),
    "pref": HeadSpec(
        key="pref", label="pref-v3-gvo palette ranker (mnv4_conv_small, single tower)",
        pin="tools/queries/scorer/data.ACTIVE_SCORER_DIR", ckpt_rel=_pref_ckpt(),
        kind="ranker", marginal="scalar",
        material="47 library locations x 12 production palette_candidates (564 frames)",
        gate_threshold=None, gate_pin=None),
}
