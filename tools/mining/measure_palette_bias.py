r"""measure_palette_bias.py — hue/flavor-family distribution of what a sheet SERVED, against
what it was OFFERED and against the whole pool.

The question is Matt's, mid-labeling: "palettes over-concentrate in a universally-good
subset (few greens; purple/fire/ice heavy)". Three columns are needed to say whose fault
that is, and the tree has all three:

  POOL      `data/palettes/pool_colormaps.json` — 987 palettes, what exists.
  PROPOSED  the sheet's own screen log, where one exists: every palette CANDIDATE the
            screen scored per location. Present for the wallpaper sitting
            (`scratch/wallpaper_sitting/v2/screen.jsonl`); the mining sheets inherit an
            already-chosen palette per row and have no proposal stage, which is itself the
            finding.
  SERVED    the batch's `images.jsonl`.

POOL vs PROPOSED isolates the DRAW; PROPOSED vs SERVED isolates the PICK; and a family that
is absent from POOL-for-this-sheet is a POPULATION failure no pick could have fixed. Sheet B
is the third case: its 38 palettes come frozen out of `gate_passers_v3.json`.

    uv run python tools/mining/measure_palette_bias.py
    uv run python tools/mining/measure_palette_bias.py --json scratch/palette_bias.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.palettes import hue_families as HF              # noqa: E402

# (label, corpus dir, batch id, optional proposal screen log). The screen log is SCRATCH and
# may be gone — its column is reported as absent rather than as zeros.
SHEETS = (
    ("A wallpaper", "wallpaper_corpus", "2026-08-10_wallpaper_correction_v2",
     "scratch/wallpaper_sitting/v2/screen.jsonl"),
    ("B mining", "render_mode_corpus", "2026-08-10_render_mode_correction_v2", None),
    ("B v1 mining", "render_mode_corpus", "2026-08-06_render_mode_fresh_sheet_v1", None),
)
GATE_PASSERS = ROOT / "data" / "render_mode_corpus" / "gate_passers_v3.json"


def served_palettes(corpus: str, batch_id: str) -> list:
    p = ROOT / "data" / corpus / "batches" / batch_id / "images.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line)["render"]["palette"])
    return out


def proposed_palettes(rel: str | None) -> list | None:
    """Every palette CANDIDATE the sheet's screen scored. `None` when the sheet has no
    proposal stage OR when its screen log has been wiped — the two are distinguished by the
    caller's `rel` being None vs the file being absent, and both are reported, because a
    silently empty proposal column reads as "nothing was proposed"."""
    if rel is None:
        return None
    p = ROOT / rel
    if not p.exists():
        return None
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for c in json.loads(line).get("candidates") or []:
            out.append(c["palette"])
    return out


def measure() -> dict:
    verdicts = HF.families_over_pool()
    doc = {
        "families": list(HF.FAMILIES),
        "family_rule": "tools/palettes/hue_families — 6 hue groups over the "
                       "palette_deficit 12-bin chroma-weighted hue histogram, plus the "
                       "neutral/spectral prepulls taken from palette_categories.json",
        "pool": {"n": len(verdicts),
                 "artifact": "data/palettes/pool_colormaps.json",
                 "shares": HF.share_table(list(verdicts), verdicts)},
        "sheets": {},
    }
    gp = json.loads(GATE_PASSERS.read_text(encoding="utf-8")) if GATE_PASSERS.exists() else None
    if gp:
        names = [r["palette"] for r in gp["rows"]]
        doc["gate_passers_v3"] = {
            "n_rows": len(names), "n_distinct": len(set(names)),
            "source_batch": gp["meta"]["source_batch"],
            "shares": HF.share_table(names, verdicts),
            "why_it_matters": "sheet B's ENTIRE palette universe. A family with 0 here is "
                              "unreachable by any draw the mining sheets could make.",
        }
    for label, corpus, batch, screen in SHEETS:
        served = served_palettes(corpus, batch)
        if not served:
            continue
        prop = proposed_palettes(screen)
        doc["sheets"][label] = {
            "batch_id": batch,
            "n_served": len(served), "n_distinct_served": len(set(served)),
            "served_shares": HF.share_table(served, verdicts),
            "n_proposed": (len(prop) if prop is not None else None),
            "proposed_shares": (HF.share_table(prop, verdicts) if prop is not None else None),
            "proposal_source": screen,
            "proposal_note": ("no proposal stage — this sheet inherits an already-chosen "
                              "palette per row" if screen is None else
                              ("the screen log is present" if prop is not None else
                               "the screen log is GONE (scratch) — proposal column unavailable, "
                               "which is not the same as empty")),
        }
    return doc


def print_report(doc):
    cols = [("pool", doc["pool"]["shares"], doc["pool"]["n"])]
    if "gate_passers_v3" in doc:
        cols.append(("gate-pass", doc["gate_passers_v3"]["shares"],
                     doc["gate_passers_v3"]["n_rows"]))
    for label, s in doc["sheets"].items():
        if s["proposed_shares"]:
            cols.append((f"{label} prop", s["proposed_shares"], s["n_proposed"]))
        cols.append((f"{label} served", s["served_shares"], s["n_served"]))
    print(f"{'family':<10}" + "".join(f"{c[0]:>18}" for c in cols))
    print(f"{'(n)':<10}" + "".join(f"{c[2]:>18}" for c in cols))
    for f in HF.FAMILIES:
        row = f"{f:<10}"
        for _lbl, sh, _n in cols:
            v = sh.get(f, {"n": 0, "share": 0.0})
            row += f"{v['n']:>10}{v['share']*100:>7.1f}%"
        print(row)
    print()
    for label, s in doc["sheets"].items():
        print(f"{label}: {s['n_distinct_served']} distinct palettes over {s['n_served']} rows"
              f"   ({s['proposal_note']})")
    if "gate_passers_v3" in doc:
        g = doc["gate_passers_v3"]
        absent = [f for f in HF.FAMILIES if g["shares"][f]["n"] == 0]
        print(f"gate_passers_v3: {g['n_distinct']} distinct palettes over {g['n_rows']} rows "
              f"from {g['source_batch']}; families with ZERO supply: {absent}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None, help="also write the full measurement here")
    args = ap.parse_args(argv)
    doc = measure()
    print_report(doc)
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
