#!/usr/bin/env python
r"""Recut the keeper cuts against the durable v9 eval slice — STAGED, not adopted.

The keeper cut is an F0.5 threshold on a SPECIFIC head's `P(>=3)`. v9 reads raised-cap
renders, so its probability scale is not v8's, and the committed cuts in
`data/atlas/keeper_cuts.json` describe v8's gate only.

The derivation is `tools/atlas/keeper_cut.derive` itself — imported, run against the v9
slice with the v9 column prefix. Nothing about the objective, grid, tie-break, LOO-OOF,
sufficiency floor or the census-only rule for `julia:multibrot*` changes; if any of that
moved, the v8 and v9 tables would stop being comparable and the recut would be measuring
its own edit.

  keeper positive = `label >= 3`   (a class-4 location is emphatically a keeper)
  objective       = F0.5, precision-weighted, tie-break toward HIGHER t

WRITTEN TO A STAGED PATH. `data/atlas/keeper_cuts_v9.json`, NOT `keeper_cuts.json`. The
live file must keep naming the ACTIVE checkpoint — `tools/atlas/test_steered_frontier.py`
holds its provenance stamp to `active_ckpt.ACTIVE_VERSION`, which is the guard that makes a
rollback-that-forgets go red rather than silent. Overwriting it now would either break that
guard or quietly deploy a v9 threshold onto a v8 gate. Build is not flip: the swap happens
with the ACTIVE_CKPT flip, in its own pass, conditional on the pre-registered bar.

  uv run python tools/v9/keeper_cut_v9.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

import keeper_cut as kc   # noqa: E402  the derivation, imported not copied
import paths              # noqa: E402

VERSION = "v9"
EVAL = ROOT / "data" / "v9" / "eval_scores_v9.jsonl"
OUT_REL = "data/atlas/keeper_cuts_v9.json"
LIVE = ROOT / "data" / "atlas" / "keeper_cuts.json"


def main() -> int:
    if not EVAL.exists():
        sys.exit(f"missing {EVAL} — run tools/v9/eval_v9.py first (freezes the eval scores)")
    cuts = kc.derive(EVAL, VERSION)

    print("=" * 86)
    print(f"KEEPER cut recut on the {VERSION} slice (F0.5 / precision-weighted) — STAGED")
    print("=" * 86)
    print(f"{'partition':20s} {'n':>4s} {'pos':>4s} {'t_keep':>7s} {'F0.5':>6s} "
          f"{'oof':>6s} {'P':>5s} {'R':>5s}  {'v8 t':>6s}  status")
    v8_cuts = json.loads(LIVE.read_text(encoding="utf-8"))["cuts"] if LIVE.exists() else {}
    moved = {}
    for part in sorted(cuts):
        d = cuts[part]
        t8 = v8_cuts.get(part, {}).get("t")
        t8s = f"{t8:.2f}" if t8 is not None else "-"
        if d["calibrated"]:
            print(f"{part:20s} {d['n']:4d} {d['pos']:4d} {d['t']:7.2f} {d['f']:6.3f} "
                  f"{d['oof_f']:6.3f} {d['prec']:5.2f} {d['rec']:5.2f}  {t8s:>6s}  calibrated")
        else:
            print(f"{part:20s} {d['n']:4d} {d['pos']:4d} {d['t']:7.2f} {'--':>6s} "
                  f"{'--':>6s} {'--':>5s} {'--':>5s}  {t8s:>6s}  UNCALIBRATED -> baseline")
        if t8 is not None:
            moved[part] = {"v8": t8, "v9": d["t"], "delta": round(d["t"] - t8, 4),
                           "v8_calibrated": bool(v8_cuts.get(part, {}).get("calibrated")),
                           "v9_calibrated": bool(d["calibrated"])}

    out = paths.durable(OUT_REL, mkparents=True)
    kc.write(cuts, out, eval_path=EVAL, version=VERSION)
    # annotate the staged file: it must never be mistaken for the live one
    doc = json.loads(out.read_text(encoding="utf-8"))
    doc["status"] = "STAGED — NOT the live keeper cut"
    doc["keeper_predicate"] = "label >= 3"
    doc["vs_v8"] = moved
    doc["adoption"] = (
        "Adopted by the ACTIVE_CKPT flip pass, not here. Until then data/atlas/"
        "keeper_cuts.json (model=v8) stays live: tools/atlas/test_steered_frontier.py holds "
        "its provenance stamp to active_ckpt.ACTIVE_VERSION, and a v9 threshold on a v8 "
        "gate is a number about nothing.")
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT_REL} (durable, STAGED)")
    print(f"live  {LIVE.relative_to(ROOT).as_posix()} left UNTOUCHED "
          f"(model={json.loads(LIVE.read_text(encoding='utf-8'))['provenance']['model']})"
          if LIVE.exists() else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
