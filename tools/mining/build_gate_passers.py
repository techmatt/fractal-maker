r"""build_gate_passers.py — regenerate the wallpaper-v3 gate-passer set, DURABLY this time.

`scratchpad/gate_passers_v3.json` was the population both July render-mode samplers keyed
off, and `scratchpad/` is the disposable tree, so it went with everything else in the wipe
(`scratch/stage2_label_audit/report.md`: "the plan is also not re-derivable"). That verdict
was too pessimistic by one file. The audit's recovery path was "regenerate from v3 + the
surviving `data/library/library_records.jsonl`" — a DIFFERENT, 47-location population that
would have given a different draw. The set the samplers actually keyed off is derivable
EXACTLY, because the batch it was cut from survives:

    data/wallpaper_corpus/batches/2026-07-09_wallpaper_headbatch_dramatic_v1/
        images.jsonl (1000 rows, tracked)  +  crops/ (1000 JPGs, on disk, rebuildable)

Score those 1000 crops with the pinned wallpaper head v3 through the deploy transform and
keep `p_ge3 > 0.90` (`wallpaper_pins.GATE_THRESHOLD`), and the result is **401 rows over 112
distinct locations** — the two counts both July samplers print in their own headers
(`build_sample`: "gate-passers (401 rows)"; `build_scale_sample`: "the 112 v3-gate-passer
locations"). That agreement is the verification: the regeneration is checked against a census
recorded while the old path still worked, not against itself. `--expect-rows/--expect-locs`
make it a hard failure, and `main` passes them by default.

WHY THE PROBE IS THE STORED CROP, not a re-render. The gate was applied to these exact JPGs;
re-rendering would re-derive the population under today's colour tail and quietly move the
boundary. The crops are already at the label-crop pins (1280x720 ss2 lanczos3), which is the
resolution v3's deploy transform expects.

OUTPUT — `data/render_mode_corpus/gate_passers_<head version>.json`, `paths.durable()`,
tracked, ONE FILE PER HEAD (2026-08-11). Self describing: the `meta` block records the source
batch, head pin, threshold, transform, the expected/realized counts and the command, so
"record how it was built" lives IN the artifact rather than beside it where the two can drift
apart.

WHY THE NAME CARRIES THE HEAD AND NOT AN ARTIFACT VERSION. The gate is a cut on a
train-prior-calibrated CORN marginal, so the passing set is a property of (head, threshold)
and of nothing else — v4b's volume-matched 0.6052 keeps a DIFFERENT set than v3's 0.90, and
overwriting one with the other would silently move the universe under three frozen mining
corpora that record which population they were cut from. The old file therefore stays where
it is; a flip ADDS a file. The census check is keyed the same way (`CENSUS`).

    uv run python tools/mining/build_gate_passers.py            # write (verifies counts)
    uv run python tools/mining/build_gate_passers.py --report   # recount from disk, no write
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import location as loc_mod                              # noqa: E402
from tools import paths                                 # noqa: E402
from tools.wallpaper import wallpaper_pins as WP        # noqa: E402

SOURCE_BATCH = "2026-07-09_wallpaper_headbatch_dramatic_v1"
SOURCE_DIR = ROOT / "data" / "wallpaper_corpus" / "batches" / SOURCE_BATCH

# THE OUTPUT PATH IS KEYED ON THE HEAD VERSION, DERIVED FROM THE LIVE PIN (2026-08-11). The
# "v3" in the original name was never the artifact's version — it is the HEAD's, and writing
# a v4b population into a file called `gate_passers_v3.json` would be a record that lies about
# which head drew it. One file per head, so the frozen mining corpora keep naming the exact
# population they were cut from.
OUT_REL_FMT = "data/render_mode_corpus/gate_passers_{version}.json"


def out_rel(version: str | None = None) -> str:
    return OUT_REL_FMT.format(version=version or WP.HEAD_VERSION)


OUT_REL = out_rel()

# The census recorded while the old path still worked — the two counts the July samplers
# print about the file they read. This is the whole verification (verification_practice.md
# §3, "assert the COUNT against an independently recorded census").
#
# IT IS A FACT ABOUT ONE HEAD, so it is keyed on one. v3's counts verify a v3 REGENERATION and
# say nothing about any other head's population: the gate is a cut on a train-prior-calibrated
# CORN marginal, so v4b's volume-matched 0.6052 is a different cut on a different scale and the
# set it keeps has no recorded census to check against. A head with no census entry is built
# with the counts REPORTED and unchecked (`census_source: null` in the artifact) rather than
# checked against a number nobody measured — and never by re-baselining v3's, which is the one
# move `build`'s error message forbids.
CENSUS = {
    "v3": (401, 112,
           "tools/render_mode_pilot/build_sample.py header ('gate-passers (401 rows)') "
           "and build_scale_sample.py docstring ('the 112 v3-gate-passer locations')"),
}
EXPECT_ROWS, EXPECT_LOCS, CENSUS_SOURCE = CENSUS.get(
    WP.HEAD_VERSION, (0, 0, None))


def log(msg):
    print(msg, flush=True)


def _source_rows() -> list:
    p = SOURCE_DIR / "images.jsonl"
    if not p.exists():
        raise SystemExit(
            f"[gate-passers] source batch manifest absent: {p}\n"
            f"This is the population, not a cache — there is nothing to fall back to. "
            f"It is a TRACKED file; restore it with `git checkout -- {p.relative_to(ROOT)}`.")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_marginals(rows, ckpt) -> np.ndarray:
    """(N, K-1) CORN MARGINALS off the stored crops — cumprod of the conditional sigmoids.

    Marginal, never the raw conditional: `cond[:,1]` is P(tier>=3 | tier>=2), and gating on
    it would admit rows the head believes are probably bad.

    fp32, NO AUTOCAST — and this one is measured, not stylistic. Under `torch.autocast` the
    same 1000 crops give **403** passers instead of 401: two rows sit close enough to
    `p_ge3 == 0.90` that fp16 accumulation moves them across it. A gate boundary is exactly
    the place where a statistic quantized below the difference it must resolve reports an
    agreement it never checked (`verification_practice.md` §1.9), so the population is
    derived at full precision. The autocast path is fine where the score only RANKS (the
    fresh-sheet screen); it is not fine where the score CUTS."""
    import torch
    from PIL import Image
    from classifier.inference import load_scorer

    scorer = load_scorer(str(ckpt))
    log(f"[gate-passers] head {WP.HEAD_VERSION} on {scorer.device} "
        f"(K={scorer.config.get('num_classes')})")
    out, buf = [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            logits = scorer.model(torch.stack(buf).to(scorer.device)).float()
        out.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
        buf.clear()

    missing = []
    for r in rows:
        p = SOURCE_DIR / "crops" / f"{r['image_id']}.jpg"
        if not p.exists():
            missing.append(r["image_id"])
            continue
        with Image.open(p) as im:
            im.load()
            buf.append(scorer.transform(im.convert("RGB")))
        if len(buf) == 32:
            flush()
    flush()
    if missing:
        # A crop that is absent is a row that CANNOT be gated, and silently gating the rest
        # would hand back a short population that reads as complete. The crops are
        # gitignored-but-rebuildable, so the message names the rebuild rather than the loss.
        raise SystemExit(
            f"[gate-passers] {len(missing)} of {len(rows)} source crops are missing "
            f"(e.g. {missing[:5]}). The gate was applied to these exact JPGs; a partial "
            f"scan would silently shrink the population. Rebuild them from the tracked "
            f"render blocks (tools/wallpaper/label_crop.render_label_crop) and re-run.")
    return np.cumprod(np.concatenate(out, axis=0), axis=1)


def build(expect_rows: int, expect_locs: int, threshold: float) -> dict:
    rows = _source_rows()
    marg = score_marginals(rows, WP.HEAD_CKPT)
    assert marg.shape[0] == len(rows), (marg.shape, len(rows))

    passers = []
    for i, r in enumerate(rows):
        p_ge3 = float(marg[i, 1])
        if not (p_ge3 > threshold):
            continue
        loc = loc_mod.from_render_block(r["render"])
        prov = r.get("provenance", {})
        passers.append({
            "image_id": r["image_id"],
            "location_key": loc.key(),
            "family": prov.get("family") or r["render"].get("fractal_type"),
            "palette": r["render"]["palette"],
            "p_ge2": float(marg[i, 0]),
            "p_ge3": p_ge3,
            "p_ge4": float(marg[i, 2]) if marg.shape[1] > 2 else None,
            "pred": 1.0 + float(marg[i].sum()),
            # the colormap recipe the render-mode batches INHERIT verbatim
            "params": prov.get("params", {}),
            # the version-invariant render block — the only thing a re-render needs
            "render": r["render"],
        })

    locs = {p["location_key"] for p in passers}
    if not passers:
        # A derived set can pass by evaluating EMPTY (verification_practice.md §5). With no
        # recorded census for this head the count check is off, so this is the only thing
        # standing between an unreadable pin and a zero-row "population".
        raise SystemExit(
            f"[gate-passers] ZERO rows pass p_ge3 > {threshold} under head "
            f"{WP.HEAD_VERSION} over {len(rows)} scanned crops. That is not a population; "
            f"check the pin and the threshold before writing anything.")
    ok_rows = (expect_rows in (0, len(passers)))
    ok_locs = (expect_locs in (0, len(locs)))
    if not (ok_rows and ok_locs):
        raise SystemExit(
            f"[gate-passers] REGENERATION DOES NOT MATCH THE RECORDED CENSUS.\n"
            f"    got    {len(passers)} rows / {len(locs)} locations\n"
            f"    expect {expect_rows} rows / {expect_locs} locations\n"
            f"    census {CENSUS_SOURCE}\n"
            f"A mismatch means the head pin, the threshold, the deploy transform or the "
            f"source crops have moved — the set would NOT be the one the July samplers "
            f"drew from, and a corpus built on it is not comparable. Do not re-baseline "
            f"the expectation; find what moved.")

    return {
        "meta": {
            "what": f"wallpaper-head-{WP.HEAD_VERSION} gate-passer set — the population the "
                    "render-mode (mining) corpus samplers draw from: one row per (location, "
                    "palette) whose stored 1280x720 crop the deployed wallpaper head scores "
                    "above the emission gate.",
            "artifact": out_rel(),
            "regenerated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "replaces": "scratchpad/gate_passers_v3.json (disposable tree; lost in the "
                        "derived-artifact wipe — scratch/stage2_label_audit/report.md)",
            "built_by": "tools/mining/build_gate_passers.py",
            "command": "uv run python tools/mining/build_gate_passers.py",
            "source_batch": SOURCE_BATCH,
            "source_manifest": f"data/wallpaper_corpus/batches/{SOURCE_BATCH}/images.jsonl",
            "source_probe": "the batch's OWN stored crops/<image_id>.jpg (1280x720 ss2 "
                            "lanczos3) — NOT a re-render: the July gate was applied to "
                            "these exact JPGs, and re-rendering would move the boundary "
                            "under today's colour tail",
            "source_rows_scanned": len(rows),
            "head": {"ckpt": WP.HEAD_CKPT_REL, "version": WP.HEAD_VERSION,
                     "pin": "tools/wallpaper/wallpaper_pins.HEAD_CKPT_REL"},
            "gate": {"statistic": "marginal p_ge3 = cumprod(sigmoid(logits))[1] "
                                  "(NEVER the CORN conditional)",
                     "rule": f"p_ge3 > {threshold}",
                     "threshold": threshold,
                     "pin": "tools/wallpaper/wallpaper_pins.GATE_THRESHOLD"},
            "deploy_transform": "classifier.data.Transform(train=False) — 1280x720 -> "
                                "384x224 bicubic stretch + normalize (present.rs's JPG path)",
            "precision": "fp32, no autocast. MEASURED under v3 @ 0.90: the same 1000 crops "
                         "under torch.autocast give 403 passers, not 401 — two rows sit "
                         "close enough to the gate that fp16 accumulation moves them across "
                         "it. A cut needs full precision; a ranking does not.",
            "verified_against_census": {
                "expected_rows": expect_rows, "expected_locations": expect_locs,
                "realized_rows": len(passers), "realized_locations": len(locs),
                "census_source": CENSUS_SOURCE,
                "checked": bool(expect_rows or expect_locs),
                "note": "the counts a census recorded for THIS head version "
                        "(`CENSUS`), or — when that head has no entry — the realized "
                        "counts reported and unchecked. A census is a fact about one head's "
                        "cut on one scale and is never re-baselined onto another.",
            },
            "library_records_not_used": {
                "path": "data/library/library_records.jsonl",
                "why": "the audit's suggested fallback, and a DIFFERENT population — 47 "
                       "curated fresh-discovery locations from a later era, whose 12 "
                       "palette candidates carry no v3 gate score. Unioning it in would "
                       "have broadened the draw across two eras; the July design is one "
                       "population, and this source reproduces it exactly. Decision "
                       "2026-08-06, Matt.",
            },
        },
        "rows": passers,
    }


def report(path: Path) -> dict:
    """Recount from the artifact on disk — trusts nothing that ran earlier."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc["rows"]
    by_loc = Counter(r["location_key"] for r in rows)
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "rows": len(rows), "locations": len(by_loc),
        "by_family_rows": dict(Counter(r["family"] for r in rows).most_common()),
        "by_family_locations": dict(Counter(
            {r["location_key"]: r["family"] for r in rows}.values()).most_common()),
        "palettes": len({r["palette"] for r in rows}),
        "rows_per_location": {"min": min(by_loc.values()), "max": max(by_loc.values()),
                              "mean": round(len(rows) / len(by_loc), 2)},
        "p_ge3": {"min": round(min(r["p_ge3"] for r in rows), 4),
                  "max": round(max(r["p_ge3"] for r in rows), 4)},
        "head": doc["meta"]["head"], "gate": doc["meta"]["gate"],
    }


def main():
    ap = argparse.ArgumentParser(
        description=f"Regenerate the wallpaper-{WP.HEAD_VERSION} gate-passer set.")
    ap.add_argument("--report", action="store_true", help="recount the artifact on disk, no write")
    ap.add_argument("--threshold", type=float, default=WP.GATE_THRESHOLD)
    ap.add_argument("--expect-rows", type=int, default=EXPECT_ROWS, help="0 disables the check")
    ap.add_argument("--expect-locs", type=int, default=EXPECT_LOCS, help="0 disables the check")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                            # noqa: BLE001
        pass

    out = paths.durable(OUT_REL, mkparents=True)
    if args.report:
        if not out.exists():
            raise SystemExit(f"[gate-passers] nothing to report: {out} does not exist")
        print(json.dumps(report(out), indent=2))
        return

    doc = build(args.expect_rows, args.expect_locs, args.threshold)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    rep = report(out)
    log("=" * 74)
    log(f"GATE-PASSER SET REGENERATED -> {rep['path']}  ({out.stat().st_size/1e3:.0f} kB)")
    log("=" * 74)
    census = (f"census {args.expect_rows}/{args.expect_locs} — MATCH"
              if (args.expect_rows or args.expect_locs)
              else f"NO recorded census for head {WP.HEAD_VERSION} — counts reported, "
                   f"not checked")
    log(f"rows {rep['rows']} / locations {rep['locations']}  ({census})")
    log(f"families (locations): {rep['by_family_locations']}")
    log(f"palettes {rep['palettes']}  ·  rows/location {rep['rows_per_location']}  ·  "
        f"p_ge3 {rep['p_ge3']}")


if __name__ == "__main__":
    main()
