r"""levels_reference.py — THE deriver for the committed levels reference record.

WHAT THE RECORD IS. `data/palettes/levels_reference.json`: the per-image tone statistics of
Matt's own `LevelsCheck` wallpapers — a folder of pictures he has already judged well-leveled
— and the [P10, P90] BAND of each statistic across them. That band is the target the
production auto-level operator (`tools/palettes/autolevel.py`) projects onto, and it is the
whole reason the operator has no invented constants: it aims at a range read off the
reference set, so an in-range render is the identity.

READ-ONLY SOURCE, OUTSIDE THE REPO. `C:\Users\techm\Desktop\GreatWallpapers\LevelsCheck` is
never written and never copied in (48 third-party/personal wallpapers). That makes the record
UNREGENERABLE from committed inputs — if the folder changes, nothing in the tree rebuilds the
old band — which is why it is a tracked canary (`tests/test_tracked_artifacts.py`) and why the
verification below has THREE outcomes rather than two.

  P10-P90 is stated, not derived: narrower (an IQR) calls one reference image in four out of
  range, wider (min-max) is a single image's opinion at each edge. The sensitivity to that
  choice ships in the record (`iqr`, `minmax`) so a reader can price it.

THE THREE OUTCOMES (the "a verification tool that cannot reach its authority reports UNKNOWN"
rule, CLAUDE.md). A default run VERIFIES; `--write` is what writes.
  OK (0)       the source folder was read and the re-derivation is byte-identical.
  DRIFT (1)    the derivation and the committed record disagree — either the folder changed
               (re-derive with --write and say so in the commit) or the record was hand-edited.
               Also raised when the record's own bands do not follow from its own per-image
               stats, which is the hand-edit signature.
  UNKNOWN (2)  the folder is unreachable. The record's SELF-consistency is still checked (the
               bands are recomputed from the per-image block it carries), and that check is
               reported as what it is: partial. "Cannot reach the source" is never "OK", and
               it is never "DRIFT" either.

    uv run python tools/palettes/levels_reference.py                  # verify (default)
    uv run python tools/palettes/levels_reference.py --write          # freeze
    uv run python tools/palettes/levels_reference.py --against <old>  # band-vs-band readout
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools import paths as P                                   # noqa: E402  storage class
from tools.palettes import autolevel as AL                     # noqa: E402  THE measurement

# --------------------------------------------------------------------------- #
SOURCE_DIR = Path(r"C:\Users\techm\Desktop\GreatWallpapers\LevelsCheck")   # READ-ONLY
RECORD_PATH = AL.RECORD_PATH                  # the operator names its own record; no copy here
SCHEMA = "levels_reference/v1"
VERSION = "levels_v1"
BAND_LO_PCT, BAND_HI_PCT = 10.0, 90.0
BOOTSTRAP_N = 2000                            # edge-SE resamples; seeded, so re-runs match
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class ReferenceDerivationError(RuntimeError):
    """The reference cannot be derived from the source as asked."""


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# 1. Measure the source (the only step that needs the folder).
# --------------------------------------------------------------------------- #
def source_files(src: Path = SOURCE_DIR) -> list:
    return sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def measure_source(src: Path = SOURCE_DIR, verbose: bool = True) -> list:
    """Per-image tone statistics over the read-only reference folder, in file-name order.

    The statistics come from `autolevel.tone_stats` — the SAME function the operator measures
    a production render with. A reference measured by a second implementation would be a band
    the operator is not actually aiming at."""
    if not src.exists():
        raise ReferenceDerivationError(f"reference source folder is unreachable: {src}")
    per = []
    for p in source_files(src):
        with Image.open(p) as im:
            size = list(im.size)
            st = AL.tone_stats(np.asarray(im.convert("RGB")))
        st["file"] = p.name
        st["size"] = size
        per.append(st)
        if verbose:
            log(f"  {p.name:62s} blk {str(st['black_pt'])[:5]:>5s} "
                f"(all {st['black_pt_all']:.3f})  wht {st['white_pt']:.3f}  "
                f"mid {st['mid']:.3f}")
    if not per:
        raise ReferenceDerivationError(f"no readable images in {src}")
    return per


# --------------------------------------------------------------------------- #
# 2. Bands — a PURE function of the per-image block, so the record verifies against
#    itself even when the folder cannot be reached.
# --------------------------------------------------------------------------- #
def _edge_se(v: np.ndarray) -> list:
    """Bootstrap SE of each band edge over the reference IMAGES — the number that says
    whether N images is enough to PLACE an edge, rather than whether the band is wide.
    Seeded (`default_rng(0)`), so a re-derivation is byte-identical."""
    rng = np.random.default_rng(0)
    lo, hi = [], []
    for _ in range(BOOTSTRAP_N):
        s = v[rng.integers(0, v.size, v.size)]
        lo.append(np.percentile(s, BAND_LO_PCT))
        hi.append(np.percentile(s, BAND_HI_PCT))
    return [float(np.std(lo, ddof=1)), float(np.std(hi, ddof=1))]


def band_of(values) -> dict | None:
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size),
            "band": [float(np.percentile(v, BAND_LO_PCT)),
                     float(np.percentile(v, BAND_HI_PCT))],
            "median": float(np.median(v)),
            "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
            "minmax": [float(v.min()), float(v.max())],
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "edge_se": _edge_se(v)}


def bands_from_per_image(per: list) -> dict:
    """{statistic: band block} from the per-image stats alone. THE regeneration check that
    needs no images: the committed record's `bands` must be exactly this over its own
    `per_image`."""
    return {k: band_of([r[k] for r in per]) for k in AL.STAT_KEYS}


# --------------------------------------------------------------------------- #
# 3. The record.
# --------------------------------------------------------------------------- #
def build(per: list, *, src: Path = SOURCE_DIR) -> dict:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "what": ("Tone-statistic BAND read off Matt's own LevelsCheck wallpapers — the target "
                 "the production band auto-level projects onto. A render whose three "
                 "statistics all sit inside these bands is left EXACTLY untouched."),
        "source": str(src),
        "source_is_read_only": True,
        "n_images": len(per),
        "band_pct": [BAND_LO_PCT, BAND_HI_PCT],
        "band_choice": ("[P10, P90] across images keeps the middle 80% of a set already judged "
                        "good. Narrower (the `iqr` beside it) calls one reference image in four "
                        "out of range; wider (`minmax`) is one image's opinion at each edge. "
                        "Stated, not derived — the sensitivity is in the record."),
        "definitions": {
            "black_pt": (f"P{AL.CLIP_LO} of OKLab L over NEUTRAL pixels (chroma <= "
                         f"{AL.CHROMA_NEUTRAL}); null when the chroma guard declares it "
                         f"unmeasurable (neutral share < {AL.NEUTRAL_FRAC_MIN}, or a neutral "
                         f"black more than {AL.DARK_MARGIN} above the all-pixel black)"),
            "white_pt": f"P{AL.CLIP_HI} of OKLab L, all pixels",
            "mid": f"median OKLab L over the structure mask (L > {AL.MASK_L})"},
        "measured_by": "tools/palettes/autolevel.tone_stats (the operator's own measurement)",
        "bands": bands_from_per_image(per),
        "black_unmeasurable": [r["file"] for r in per if r["black_pt"] is None],
        "derived": time.strftime("%Y-%m-%d"),
        "command": "uv run python tools/palettes/levels_reference.py --write",
        "per_image": per,
    }


def serialize(doc: dict) -> str:
    return json.dumps(doc, indent=1) + "\n"


def committed(path: str | Path | None = None) -> dict | None:
    p = Path(path) if path else (ROOT / RECORD_PATH)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 4. Verify / write.
# --------------------------------------------------------------------------- #
OK, DRIFT, UNKNOWN = 0, 1, 2


def self_consistent(doc: dict) -> tuple:
    """(ok, differing statistic names). The bands a hand-edit would have desynced from the
    per-image block they claim to summarize."""
    want = bands_from_per_image(doc["per_image"])
    bad = [k for k in AL.STAT_KEYS if json.dumps(want.get(k)) != json.dumps(doc["bands"].get(k))]
    return (not bad), bad


def verify(src: Path = SOURCE_DIR, path: str | Path | None = None) -> int:
    """Three-outcome verification (see the module docstring). Never claims OK without having
    read the source."""
    doc = committed(path)
    if doc is None:
        log(f"[levels] MISSING: {RECORD_PATH} — run with --write to freeze it.")
        return DRIFT
    ok, bad = self_consistent(doc)
    if not ok:
        log(f"[levels] DRIFT: the committed bands do not follow from the record's own "
            f"per-image stats ({bad}). That is the hand-edit signature — re-derive with "
            f"--write.")
        return DRIFT
    if not src.exists():
        log(f"[levels] UNKNOWN — the reference source {src} is unreachable, so the record "
            f"could not be checked against what it claims to summarize. What WAS checked: "
            f"its {len(doc['per_image'])} per-image stats reproduce its bands exactly. "
            f"That is a partial verification and is not an OK.")
        return UNKNOWN
    fresh = build(measure_source(src, verbose=False), src=src)
    fresh["derived"] = doc.get("derived")          # the date records WHEN, not what
    if serialize(fresh) != serialize(doc):
        keys = [k for k in fresh if json.dumps(fresh[k]) != json.dumps(doc.get(k))]
        log(f"[levels] DRIFT: re-deriving from {src} does not reproduce {RECORD_PATH} "
            f"(differing keys: {keys}). Either the folder changed (re-derive with --write and "
            f"say so in the commit) or the record was hand-edited.")
        return DRIFT
    log(f"[levels] OK — {RECORD_PATH} re-derives byte-identically from {doc['n_images']} "
        f"images at {src}.")
    return OK


def write(src: Path = SOURCE_DIR) -> dict:
    per = measure_source(src)
    doc = build(per, src=src)
    out = P.durable(RECORD_PATH, mkparents=True)
    out.write_text(serialize(doc), encoding="utf-8")
    log(f"\n[levels] froze {RECORD_PATH}: n={doc['n_images']}, "
        f"{len(doc['black_unmeasurable'])} with no measurable black point")
    for k in AL.STAT_KEYS:
        b = doc["bands"][k]
        log(f"  {k:9s} band [{b['band'][0]:.4f}, {b['band'][1]:.4f}] (n={b['n']}) "
            f"median {b['median']:.4f}  edge SE {b['edge_se'][0]:.4f}/{b['edge_se'][1]:.4f}")
    return doc


# --------------------------------------------------------------------------- #
# 5. Band-vs-band readout (what a top-up did to the edges).
# --------------------------------------------------------------------------- #
def compare(new: dict, old: dict) -> list:
    """One row per statistic: both bands, both edge SEs, and the movement of each edge.

    The edge SE is the number that answers "is N images enough to PLACE this edge", so a
    top-up is judged by whether the SEs fell, not by whether the band moved."""
    rows = []
    for k in AL.STAT_KEYS:
        a, b = new["bands"].get(k), old["bands"].get(k)
        if not a or not b:
            continue
        rows.append({
            "stat": k,
            "n_new": a["n"], "n_old": b["n"],
            "band_new": a["band"], "band_old": b["band"],
            "d_lo": a["band"][0] - b["band"][0], "d_hi": a["band"][1] - b["band"][1],
            "se_new": a["edge_se"], "se_old": b["edge_se"],
            "d_se_lo": a["edge_se"][0] - b["edge_se"][0],
            "d_se_hi": a["edge_se"][1] - b["edge_se"][1],
            "width_new": a["band"][1] - a["band"][0],
            "width_old": b["band"][1] - b["band"][0],
        })
    return rows


def print_compare(rows: list) -> None:
    log("\n| statistic | n | band (new) | band (old) | Δlo | Δhi | edge SE new | edge SE old |")
    log("|---|--:|---|---|--:|--:|---|---|")
    for r in rows:
        log(f"| {r['stat']} | {r['n_old']}→{r['n_new']} | "
            f"[{r['band_new'][0]:.3f}, {r['band_new'][1]:.3f}] | "
            f"[{r['band_old'][0]:.3f}, {r['band_old'][1]:.3f}] | "
            f"{r['d_lo']:+.3f} | {r['d_hi']:+.3f} | "
            f"{r['se_new'][0]:.3f}/{r['se_new'][1]:.3f} | "
            f"{r['se_old'][0]:.3f}/{r['se_old'][1]:.3f} |")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="freeze the record (default: verify; a durable record of a past "
                         "state is not rewritten by a default run)")
    ap.add_argument("--source", type=Path, default=SOURCE_DIR,
                    help="the READ-ONLY reference folder")
    ap.add_argument("--against", type=Path, default=None,
                    help="a previous levels_reference.json to print a band-vs-band readout "
                         "against (no default: an earlier record may live anywhere)")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:                                          # noqa: BLE001
            pass

    doc = write(args.source) if args.write else None
    rc = OK if args.write else verify(args.source)
    if args.against:
        if not Path(args.against).exists():
            log(f"[levels] --against {args.against} does not exist; no comparison printed.")
        else:
            new = doc or committed()
            print_compare(compare(new, json.loads(Path(args.against)
                                                  .read_text(encoding="utf-8"))))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
