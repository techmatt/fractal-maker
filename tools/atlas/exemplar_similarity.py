#!/usr/bin/env python
r"""exemplar_similarity.py — "does it look like the ones he liked?", as a RECORDED feature.

THE HYPOTHESIS THIS EXISTS TO TEST, and the reason it is not allowed to steer. Matt's
proposal is that closeness to a handful of exemplars predicts his own verdict. That is a
claim about labels, so the only thing that can settle it is labels: this module computes the
similarity, writes it on every candidate, and **selects on it exactly once** — the
~60-row exemplar mini-chunk of the label batch, which is registered as its own biased draw
so the answer can be read against the stratified chunks rather than out of them. It is not
in the crawl's sort key, not in the prior, not in the quota, and not in the stratification.

ONE SUBSTRATE, AND WHY THIS ONE. Mixing embedding spaces is the failure mode here: a
steering-JPG CLIP embedding and a `morph_clip` embedding are different spaces and a cosine
across them is a number with no referent. Every embedding on both sides of this comparison
therefore comes from ONE function — `embed_fields` — applied to the 64x36 escape-time field
each row already has, colour-mapped deterministically and stretched (never cropped) to the
model's input.

The field was chosen over a render because it is the only substrate that EXISTS for the
whole population. The mini-chunk is drawn from all recorded candidates, pushed and passed
over, of which there are tens of thousands and none has a JPG — rendering them to get a
prettier embedding would cost more than the crawl that produced them. The cost of the choice
is real and is stated rather than buried: 64x36 is a coarse look at a picture, so this
measures COMPOSITION and BANDING STRUCTURE and is close to blind to the palette-level
qualities a full render would carry. A null result is therefore evidence about the
hypothesis AT THIS FIDELITY and not about the hypothesis in general.

WHAT IS IN THE EXEMPLAR SET, in three legs, each naming where its verdict is recorded:

  1. the gate's named FAVOURITE (`q4_neig_089`) and the two calibration references;
  2. SHEET-PASSED tiles — the calibration sheet rows that cleared the interior level Matt
     tolerated, read out of `view_screen_gate.json` `calibration_set.passed_tiles`;
  3. label-corpus rows scored 3 or 4 by hand.

THE SECOND LEG IS THE WEAKEST AND SAYS SO. "Passed" there is a DERIVED set, not an
enumeration of verdicts: the gate names the tiles that failed and identifies the rest by
being at or under `PASSED_MAX_INTERIOR`. They are tiles Matt looked at and did not name as
failing, on a sheet he ruled on, and the project's own gate requires them to stay in the top
quintile (G5) — which is a code-enforced positive set, not a per-tile opinion.

Recovering them at all took a correction. They are NOT recoverable by regenerating the sheet:
regenerating it from `scratch/view_rescreen/scores.jsonl` at the recorded seed recovered **0
of the 6 named tiles**, the row-order dependence §11.7 records as `--sheet-order`. The gate
does the regeneration correctly and cross-checks the named tiles against it, so the fix was to
have the gate WRITE the passed rows' keys into its record rather than to re-derive them here.
`[measured: 2026-08-01; view_screen_gate.passed_tiles]`

  uv run python tools/atlas/exemplar_similarity.py --run-dir data/discovery/<run>
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "tools" / "orbital", ROOT / "tools" / "corpus", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import field_metrics as fm            # noqa: E402
import maneuver_screen as ms          # noqa: E402
import view_field_cache as vfc        # noqa: E402
import view_screen as vs              # noqa: E402

SUBSTRATE = "field64x36_twilight_shifted_clip_vitb16"
CLIP_MODEL = "vit_base_patch16_clip_224.openai"      # the library morph_clip backbone
COLORMAP = "twilight_shifted"                        # the cheap-presentation palette


# --------------------------------------------------------------------------- #
# the exemplar set
# --------------------------------------------------------------------------- #
# Each entry names WHERE its positive verdict is recorded. That column is the point: an
# exemplar set assembled from memory is a set of preferences the author has, not the one the
# record supports.
EXEMPLARS = (
    dict(key="q4_neig_089", label="the favourite (neighborhood_expand k16 d2 p43)",
         cx="-0.6565829797781106821644", cy="0.3719359208903791863335",
         fw=6.3994681768e-08, family="mandelbrot",
         verdict="data/atlas/view_screen_gate.json v4.anchors.favourite"),
    dict(key="mb19_p35_16x", label="mb19_p35 at 16x",
         cx="-0.74977483272365342795786040375088960",
         cy="0.10761724352653678278696798751738616",
         fw=16 * 2.0174060071e-10, family="mandelbrot",
         verdict="view_screen.REF_VIEWS (the screen's own calibration reference)"),
    dict(key="minibroteye", label="the minibrot eye at 4x",
         cx="-0.746339", cy="0.112242", fw=4 * 1.4575000000e-04, family="mandelbrot",
         verdict="view_screen.REF_VIEWS (the screen's own calibration reference)"),
)

# ...plus the label corpus's own top human scores, resolved at run time rather than pasted
# in, so re-running after new labels land picks them up instead of freezing 2026-08-01's
# view of the corpus.
#
# TWO FILTERS, EACH WITH A REASON.
# (a) C-PLANE ONLY. The crawl's population is entirely parameter-plane views of minibrot
#     neighbourhoods; a julia or phoenix row is a z-plane viewport, a different object whose
#     composition vocabulary does not transfer. Including them would put distance-to-a-julia
#     into a feature claiming to measure distance-to-what-he-liked-about-these.
# (b) ONE PER (score, family). Otherwise the set is whatever the corpus happens to hold most
#     of: the two score-4 rows are both multibrot5 at the SAME fw in the same roster batch,
#     so admitting both would weight the `mean` similarity toward one near-duplicate pair
#     while pretending to be two independent verdicts. Within a cell the pick is a SEEDED
#     shuffle, not the author's choice, because "which of 318 good rows" is exactly where an
#     exemplar set gets quietly fitted to the answer it is supposed to test.
CORPUS_SCORES = (4, 3)
CORPUS_PER_SCORE_FAMILY = 1
CORPUS_FAMILIES = ("mandelbrot", "multibrot3", "multibrot4", "multibrot5")
CORPUS_DRAW_SEED = 20260801
N_SHEET, N_CORPUS = 3, 2          # the two derived legs, capped so the set stays 5-8
GATE_REL = "data/atlas/view_screen_gate.json"


def sheet_exemplars(*, n: int = N_SHEET, seed: int = CORPUS_DRAW_SEED) -> list[dict]:
    """The calibration sheet's passed tiles, read from the gate record. Never raises.

    One per OPERATOR before any operator gets two, then a seeded shuffle inside the cell —
    the same discipline every draw in this tree uses, and for the same reason: "which three
    of twelve" is otherwise a choice the author makes, and this set is supposed to test a
    hypothesis rather than agree with one.
    """
    import random
    from pathlib import Path as _P
    import paths
    p = _P(paths.durable(GATE_REL))
    if not p.exists():
        return []
    g = json.loads(p.read_text(encoding="utf-8"))
    tiles = (g.get("v4") or g.get("v3") or {}).get("calibration_set", {}).get(
        "passed_tiles") or []
    cells: dict = {}
    for t in tiles:
        cells.setdefault(t["tile"].split("|", 1)[0], []).append(t)
    rng = random.Random(seed)
    for c in cells.values():
        c.sort(key=lambda t: t["key"])
        rng.shuffle(c)
    out, i = [], 0
    while len(out) < n:
        avail = [k for k in sorted(cells) if len(cells[k]) > i]
        if not avail:
            break
        for k in avail:
            if len(out) >= n:
                break
            t = cells[k][i]
            out.append(dict(key=f"sheet_{t['key']}", label=f"v2 Q5 sheet: {t['tile']}",
                            cx=t["cx"], cy=t["cy"], fw=float(t["fw"]),
                            family=t.get("partition") or "mandelbrot",
                            verdict=(f"{GATE_REL} calibration_set.passed_tiles "
                                     f"(interior {t['interior_fraction']} <= the level Matt "
                                     f"tolerated; a DERIVED positive, not a named verdict)")))
        i += 1
    return out


def corpus_exemplars(*, seed: int = CORPUS_DRAW_SEED) -> list[dict]:
    """Human-labelled positives from the label corpus, as exemplar rows. Never raises."""
    import random
    root = ROOT / "data" / "label_corpus" / "batches"
    cells: dict = {}
    for p in sorted(root.glob("*/images.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            s = (r.get("label") or {}).get("score")
            if s not in CORPUS_SCORES:
                continue
            rd = r["render"]
            fam = rd.get("fractal_type") or "mandelbrot"
            if fam not in CORPUS_FAMILIES:
                continue
            cells.setdefault((s, fam), []).append(dict(
                key=f"corpus_{r['image_id']}", label=f"corpus label {s}: {r['image_id']}",
                cx=rd["cx"], cy=rd["cy"], fw=float(rd["fw"]), family=fam,
                verdict=f"{p.parent.name}/images.jsonl label.score={s}"))
    rng = random.Random(seed)
    out, seen_fam = [], set()
    for s in CORPUS_SCORES:                      # 4 before 3: the stronger verdict wins the
        for fam in CORPUS_FAMILIES:              # family slot, and 3 fills what is left
            pool = sorted(cells.get((s, fam), []), key=lambda r: r["key"])
            if not pool or fam in seen_fam:
                continue
            rng.shuffle(pool)
            out.extend(pool[:CORPUS_PER_SCORE_FAMILY])
            seen_fam.add(fam)
    return out[:N_CORPUS]


def exemplar_set() -> list[dict]:
    """The three legs, strongest verdict first. Capped at 8 — a bigger set does not make the
    hypothesis easier to test, it makes `mean` similarity converge on "generically fractal"."""
    return list(EXEMPLARS) + sheet_exemplars() + corpus_exemplars()


# --------------------------------------------------------------------------- #
# the substrate
# --------------------------------------------------------------------------- #
def field_to_rgb(field: np.ndarray) -> np.ndarray:
    """One 64x36 escape-time field -> uint8 RGB, deterministically.

    `field * DENSITY` mod 1 is the render path's own colour coordinate — one cycle is
    `1/DENSITY = 40` iterations (`field_metrics.DENSITY`, from `src/cli.rs`) — so the bands
    this shows are the bands the render shows, at this resolution. Non-escaping pixels are
    BLACK, matching `interior_mode=black`, and not the colormap's zero: an interior that
    took a palette colour would make a black disc look like structure to the embedder.

    Cyclic and phase-free rather than percentile-stretched, which is the choice worth
    naming: a stretch is relative to the frame's own value range, so two frames with
    identical structure at different depths would embed differently for a reason that is
    about depth and not about the picture.
    """
    from matplotlib import colormaps
    t = np.mod(np.asarray(field, dtype=np.float64) * fm.DENSITY, 1.0)
    fin = np.isfinite(field)
    rgb = colormaps[COLORMAP](np.where(fin, t, 0.0))[..., :3]
    rgb = np.where(fin[..., None], rgb, 0.0)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_input_image(field: np.ndarray, size) -> "object":
    """One field -> the model's input image, STRETCHED to `size = (H, W)`. Never cropped.

    Its own function so the no-crop claim is testable without loading a 300 MB backbone.
    The failure it exists to prevent is silent: timm's default eval transform resizes to
    `input/crop_pct` and centre-crops, so a 16:9 field would reach the model with its left
    and right thirds gone — an embedding of a picture nobody screened.
    """
    from PIL import Image
    return Image.fromarray(field_to_rgb(field)).resize((size[1], size[0]), Image.BICUBIC)


class Embedder:
    """CLIP over `field_to_rgb`, STRETCHED to the model input — never centre-cropped.

    timm's default eval transform resizes to `input/crop_pct` and centre-crops, which on a
    16:9 field would throw the left and right thirds away — i.e. it would embed a different
    picture from the one being screened. So the transform here is the same deterministic
    stretch the classifier's deploy transform uses (`classifier.data.Transform`): resize the
    whole frame to the square input, normalize, done.
    """

    def __init__(self, model_name: str = CLIP_MODEL):
        import timm
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(model_name, pretrained=True,
                                       num_classes=0).eval().to(self.dev)
        cfg = timm.data.resolve_model_data_config(self.model)
        self.size = tuple(cfg["input_size"][1:])
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(self.dev)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(self.dev)

    def embed_fields(self, fields, bs: int = 128) -> np.ndarray:
        """`(N, D)` L2-normalized embeddings for a sequence of 64x36 fields."""
        torch = self.torch
        out = []
        buf = list(fields)
        for i in range(0, len(buf), bs):
            ims = [to_input_image(f, self.size) for f in buf[i:i + bs]]
            x = torch.from_numpy(
                np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in ims])
            ).permute(0, 3, 1, 2).to(self.dev)
            x = (x - self.mean) / self.std
            with torch.no_grad():
                e = self.model(x).float().cpu().numpy()
            out.append(e)
        e = np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)
        n = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.maximum(n, 1e-12)


def exemplar_fields(rows: list[dict], *, threads: int = 2) -> tuple[list, list]:
    """Dump each exemplar's 64x36 field AT ITS OWN FRAME under the screen's cap policy.

    Identical to how the candidates' fields were produced — same geometry, same
    `screen_maxiter`, same policy token — because "embedded identically" is the whole
    contract. A dump that fails drops the exemplar from the set with a named reason rather
    than substituting anything.
    """
    fields, kept = [], []
    for r in rows:
        meta = vs.view_frame_policy(r["fw"])
        if not meta["screened"]:
            r["skipped"] = meta["screen_reason"]
            continue
        try:
            with tempfile.TemporaryDirectory() as td:
                f, _ = fm.dump_field(r["cx"], r["cy"], float(r["fw"]),
                                     meta["view_maxiter"], Path(td) / "f.bin",
                                     width=fm.SCREEN_W, height=fm.SCREEN_H,
                                     ss=fm.SCREEN_SS, family=r.get("family", "mandelbrot"),
                                     threads=threads, timeout=fm.FIELD_TIMEOUT_S)
        except Exception as e:
            r["skipped"] = f"dump_field:{str(e)[:120]}"
            continue
        fields.append(f)
        kept.append(dict(r, view_maxiter=meta["view_maxiter"],
                         **{fm.POLICY_KEY: meta[fm.POLICY_KEY]}))
    return kept, fields


def similarities(cand: np.ndarray, ex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(max, mean)` cosine of every candidate against the exemplar set.

    BOTH, because they are different questions and the prompt asks for both: `max` is "does
    it look like ONE of the things he liked", `mean` is "does it look like the KIND of thing
    he liked". A set holding two near-duplicate exemplars pulls `mean` toward that pair
    while leaving `max` alone, which is exactly the disagreement worth being able to see.
    """
    if ex.size == 0 or cand.size == 0:
        z = np.zeros(len(cand), dtype=np.float32)
        return z, z
    s = cand @ ex.T
    return s.max(axis=1), s.mean(axis=1)


# --------------------------------------------------------------------------- #
def run(run_dir: Path, out: Path | None = None, *, limit: int | None = None) -> dict:
    cache = vfc.RunFieldCache(run_dir / "view_fields", mode="r")
    rows = cache.rows[:limit] if limit else cache.rows
    ex_rows, ex_fields = exemplar_fields(exemplar_set())
    emb = Embedder()
    ex_emb = emb.embed_fields(ex_fields)

    keys = [r["key"] for r in rows]
    cand_emb = emb.embed_fields([cache.get(k) for k in keys])
    smax, smean = similarities(cand_emb, ex_emb)

    out = out or (run_dir / "exemplar_sim.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for k, a, b in zip(keys, smax, smean):
            f.write(json.dumps(dict(key=k, exemplar_sim_max=round(float(a), 6),
                                    exemplar_sim_mean=round(float(b), 6),
                                    substrate=SUBSTRATE)) + "\n")
    rep = dict(
        n=len(keys), substrate=SUBSTRATE, clip_model=CLIP_MODEL, colormap=COLORMAP,
        geometry=[fm.SCREEN_W, fm.SCREEN_H, fm.SCREEN_SS],
        policy=ms.screen_policy_token(),
        exemplars=[{k: v for k, v in r.items() if k != "skipped"} for r in ex_rows],
        exemplars_skipped=[dict(key=r["key"], reason=r["skipped"])
                           for r in exemplar_set() if r.get("skipped")],
        sim_max=dict(min=float(smax.min()) if len(smax) else None,
                     median=float(np.median(smax)) if len(smax) else None,
                     max=float(smax.max()) if len(smax) else None),
        NOTE=("RECORDED, never the crawl's ordering. Selects exactly once: the exemplar "
              "mini-chunk of the label batch, registered as its own biased draw."),
        out=str(out))
    (run_dir / "exemplar_sim_report.json").write_text(json.dumps(rep, indent=2) + "\n",
                                                      encoding="utf-8")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    rep = run(a.run_dir, a.out, limit=a.limit)
    print(json.dumps({k: v for k, v in rep.items() if k != "exemplars"}, indent=2))
    for r in rep["exemplars"]:
        print(f"  exemplar {r['key']:34s} fw={float(r['fw']):.4g}  {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
