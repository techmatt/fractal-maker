#!/usr/bin/env python
r"""Rebuild a discovery run's `outcome_feats.npz` from its `outcome_ledger.jsonl`.

WHY THIS EXISTS. The feature store was demoted from committed to `bulk()` on 2026-08-08
(`artifacts._is_discovery_feats`): it is the derived sidecar of the ledger — one 1280-D
penultimate vector per admitted q3 — and at 3.23 of steady_state_v2's 10.77 MB it was 30%
of a modern run's committed tree bytes for a store `production_seeder.py` describes as
"logged, never gates". A `bulk()` artifact is only honestly bulk if something rebuilds it,
and `discovery_sinks._require_feats` names THIS command. It is the other half of the
demotion, not a convenience.

WHAT IT REPRODUCES, AND WHAT IT DOES NOT. The recipe is the run's own: `prescreen._render`
at v5 search fidelity (640x360 ss2, `active_ckpt.PALETTE`, `auto_maxiter(fw)`) on the row's
outcome frame and its own plane, then `prescreen.embed_paths` -> the penultimate hook. Same
two functions `production_seeder.outcome_feature` calls, reached by import so there is no
second copy of the recipe.

  **It is NOT a byte-restore.** Each banked vector was pulled through the head that was
  ACTIVE when its run walked — every ledger row records that in `scorer_version` — and
  those weights are de-tracked under ACTIVE+PREVIOUS retention. By default this embeds
  through `production_pins.ACTIVE_CKPT`, i.e. TODAY's head, and stamps which one into the
  npz under the reserved `__meta__` key. Pass `--model` to embed through a specific
  checkpoint if you still have it. A run whose rows name several `scorer_version`s never
  had a single-head store to begin with; `--check` prints the tally.

WHICH ROWS. Exactly the rows with `distinct: true` — that is the population the seeder
stores a feature for, and it is derived from the ledger rather than restated (verified on
two banked runs: steady_state_v2 2,244 npz keys == 2,244 distinct of 2,718 rows;
campaign2/breadth 311 == 311 of 326).

THE PLANE IS PART OF THE IDENTITY. A julia row carries `julia_c_*`, a phoenix row
`phoenix_{c,p,zm1}_*`, and rendering either without them produces a real-looking image of a
DIFFERENT fractal at the right coordinates — the failure `build_q4_harvest_batches`
already refuses. This refuses it the same way, per row, rather than embedding the default
plane.

  uv run python tools/atlas/recompute_outcome_feats.py --run data/discovery/<run>
  uv run python tools/atlas/recompute_outcome_feats.py --run <dir> --check      # no render
  uv run python tools/atlas/recompute_outcome_feats.py --run <dir> --limit 8    # bounded
      end-to-end: renders and embeds 8 rows, writes NOTHING (a bounded run that wrote a
      real store would leave a store nobody can tell from a complete one)

Writes: the run's `outcome_feats.npz` through `discovery_sinks.feats_path` -> `bulk()`,
out-of-tree. Render scratch goes to `scratch/recompute_outcome_feats/<run>/`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "corpus",
           ROOT / "tools" / "mining", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import discovery_sinks as ds   # noqa: E402
import paths                   # noqa: E402
import prescreen               # noqa: E402


def _rows(run: Path) -> list[dict]:
    led = run / "outcome_ledger.jsonl"
    if not led.exists():
        raise SystemExit(f"no outcome_ledger.jsonl in {run} — nothing to rebuild from")
    return [json.loads(l) for l in led.open(encoding="utf-8") if l.strip()]


def render_family_of(partition: str) -> str:
    """Ledger PARTITION -> render `fractal_type`. Namespaced on the ledger side
    (`julia:multibrot3`), not on the render side (`julia_multibrot3`); a derived partition
    renders as its base (`phoenix:classic` -> `phoenix`). Same mapping as
    `steered_frontier.render_family_of`, restated here for the reason
    `build_q4_harvest_batches` restates it: that module imports torch, and a rebuild tool
    should not pay a classifier import to translate a string."""
    base = partition.split("+", 1)[0]
    if base.startswith("phoenix"):
        return "phoenix"
    if base.startswith("julia:"):
        tail = base.split(":", 1)[1]
        return "julia" if tail == "mandelbrot" else f"julia_{tail}"
    return base


def plane_of(r: dict) -> tuple[str, tuple | None, dict]:
    """(family, c, family_params) for one ledger row — or a refusal.

    Fails loud on a missing plane parameter rather than letting the engine substitute its
    default: a phoenix frame rendered without `p` is a different fractal wearing the right
    coordinates, and it looks fine."""
    fam = render_family_of(r.get("family") or r.get("partition") or "mandelbrot")
    if fam == "mandelbrot" or fam.startswith("multibrot"):
        return fam, None, {}

    c_re = r.get("c_re") or r.get("julia_c_re") or r.get("phoenix_c_re")
    c_im = r.get("c_im") or r.get("julia_c_im") or r.get("phoenix_c_im")
    if c_re is None or c_im is None:
        raise SystemExit(
            f"row {r.get('id')!r} is family {fam!r} but carries no `c` in any of "
            f"c_re / julia_c_re / phoenix_c_re. Refusing: without `c` the engine renders "
            f"its default plane at these coordinates, i.e. a different fractal.")
    if fam != "phoenix":
        return fam, (c_re, c_im), {}

    keys = ("p_re", "p_im", "zm1_re", "zm1_im")
    fp = {k: r.get(k) if r.get(k) is not None else r.get(f"phoenix_{k}") for k in keys}
    if any(v is None for v in fp.values()):
        raise SystemExit(
            f"phoenix row {r.get('id')!r} has no (p, z_-1) — missing "
            f"{[k for k in keys if fp[k] is None]}. Refusing for the same reason as `c`.")
    return fam, (c_re, c_im), fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="a discovery run dir holding outcome_ledger.jsonl")
    ap.add_argument("--model", default=None,
                    help="checkpoint to embed through (default: production_pins.ACTIVE_CKPT)")
    ap.add_argument("--check", action="store_true",
                    help="report the population and the scorer_version tally; render nothing")
    ap.add_argument("--limit", type=int, default=0,
                    help="bounded end-to-end over N rows. Renders and embeds for real and "
                         "writes NOTHING — a partial store is indistinguishable from a "
                         "complete one once it is on disk.")
    a = ap.parse_args()

    run = a.run if a.run.is_absolute() else (ROOT / a.run)
    rows = _rows(run)
    want = [r for r in rows if r.get("distinct")]
    out = ds.feats_path(run)
    vers = Counter(r.get("scorer_version") or "<unstamped>" for r in want)

    print(f"run            : {run}")
    print(f"ledger rows    : {len(rows)}   distinct (the stored population): {len(want)}")
    print(f"scorer_version : {dict(vers)}"
          + ("   <-- MIXED: the banked store was never one head's output"
             if len(vers) > 1 else ""))
    print(f"feats out      : {out}   (bulk, {'present' if out.exists() else 'absent'})")
    if a.check:
        return 0
    if not want:
        raise SystemExit("no distinct rows — nothing to embed")

    from production_pins import ACTIVE_CKPT           # noqa: PLC0415  (torch import)
    from score_lib import Scorer                      # noqa: PLC0415
    model = a.model or ACTIVE_CKPT
    scorer = Scorer(str(model))
    print(f"embedding head : {model}  (NOT necessarily the head the rows were scored by)")

    todo = want[:a.limit] if a.limit else want
    tiles = paths.scratch("recompute_outcome_feats", run.name)
    tiles.mkdir(parents=True, exist_ok=True)
    feats: dict[str, np.ndarray] = {}
    try:
        for i, r in enumerate(todo):
            fam, c, fp = plane_of(r)
            tile = tiles / f"{r['id']}.jpg"
            ok, err = prescreen._render(r["outcome_cx"], r["outcome_cy"], r["outcome_fw"],
                                        tile, family=fam, c=c, family_params=fp)
            if not ok:
                raise SystemExit(f"outcome tile render failed [{r['id']}]: {err}")
            feats[r["id"]] = prescreen.embed_paths(scorer, [tile])[0].astype(np.float32)
            tile.unlink(missing_ok=True)
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)}", flush=True)
    finally:
        shutil.rmtree(tiles, ignore_errors=True)

    if a.limit:
        print(f"\n--limit {a.limit}: {len(feats)} vectors embedded, NOTHING WRITTEN. "
              f"A bounded run must not leave a store that reads as complete.")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / (out.stem + "_tmp.npz")
    np.savez_compressed(tmp, __meta__=np.array(
        json.dumps({"rebuilt_by": "tools/atlas/recompute_outcome_feats.py",
                    "embedding_head": str(model),
                    "ledger_scorer_versions": dict(vers),
                    "n": len(feats),
                    "note": ("NOT a byte-restore of the original store — those vectors "
                            "came from the head each row's scorer_version names.")})),
                        **feats)
    tmp.replace(out)
    print(f"\nWROTE {out}   ({len(feats)} vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
