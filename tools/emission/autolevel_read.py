#!/usr/bin/env python
"""autolevel_read.py — the auto-level readout off the release record's `autolevel` stamp.

Written for run 26 under `scratch/production_run26/`; promoted here for run 27 because it is
the standing post-run read of a switch that is now ON in production, and a reader that lives
only in a `scratch/` tree is one wipe from not existing (`CLAUDE.md`, "neither scratch tree is
a dependency tier"). It reads the DURABLE record (`data/emission/release_records/`) and one
run's scratch pool; it writes one JSON and changes nothing.

The numbers:
  * IDENTITY SHARE, overall, by render style, and ON THE ROWS THAT SHIPPED. "Identity" is the
    stamp's own `curve.identity` — the operator ran, measured the render, and found it already
    inside the committed band, so the map it applied is the identity. `acted` is its
    complement and both are read, because a disagreement between them is a bug in the stamp,
    not a statistic. The RELEASED split is separate because the two populations answered
    differently in run 26 (58% identity over all 48 scored, 75% over the 12 released): the
    selector prefers already-in-band renders, so quoting the scored share as "what ships" is
    wrong in a knowable direction.
  * CHROMA-CAP PREVALENCE, over the rows the operator ACTED on (a cap on an identity row is
    meaningless), as both the share of rows capped at all and the share of stops capped.
  * RENDER-COST IMPACT. The operator's cost is one EXTRA render per acting row: the BEFORE
    render is what it measures, the AFTER render is what ships. So the expected multiplier on
    the colorize leg is 1 + acted_share, and the realized one is measured within-run off the
    attempts' own JPG mtimes.

Population: `stage == "release"` rows for this run — one row per SCORED candidate. The `gate`
rows mirror the same candidates with the same stamp, so counting both double-counts.

  uv run python tools/emission/autolevel_read.py --run-id prod27 \
      --out scratch/production_run27/autolevel_read.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import paths  # noqa: E402

DEFAULT_REC = ROOT / "data" / "emission" / "release_records" / "emission_diversity_v1.jsonl"


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.0f}%)" if d else f"{n}/0 (n/a)"


def read_report(record: Path, run_id: str, pool: Path) -> dict:
    rows = [json.loads(l) for l in record.read_text(encoding="utf-8").splitlines() if l.strip()]
    mine = [r for r in rows if r.get("run_id") == run_id and r.get("stage") == "release"]
    if not mine:
        raise SystemExit(f"no stage=release rows with run_id={run_id!r} in {record} "
                         f"(run_ids present: {sorted({r.get('run_id') for r in rows})})")

    unstamped = [r for r in mine if not isinstance(r.get("autolevel"), dict)]
    stamped = [r for r in mine if isinstance(r.get("autolevel"), dict)]

    def split(rs):
        smooth = [r for r in rs if r.get("render_style") == "smooth"]
        strange = [r for r in rs if r.get("render_style") != "smooth"]
        return smooth, strange

    rep = {
        "run_id": run_id,
        "record": str(record),
        "n_scored_rows": len(mine),
        "n_unstamped": len(unstamped),
        "schema_versions": dict(Counter(r.get("schema_version") for r in mine)),
        "switch": dict(Counter(r["autolevel"].get("switch") for r in stamped)),
        "operator": dict(Counter(r["autolevel"].get("operator") for r in stamped)),
        "reference_sha": dict(Counter((r["autolevel"].get("reference") or {}).get("sha256") for r in stamped)),
        "decisions": dict(Counter(r.get("decision") for r in mine)),
        "slot_sources": dict(Counter(r.get("slot_source") for r in mine if r.get("slot_source"))),
    }

    # --- identity / acted, overall and by mode -------------------------------------------
    disagree = [r for r in stamped
                if bool(r["autolevel"].get("acted")) == bool(r["autolevel"]["curve"].get("identity"))]
    rep["acted_identity_disagreements"] = len(disagree)

    by_mode = defaultdict(lambda: {"n": 0, "identity": 0})
    for r in stamped:
        ident = bool(r["autolevel"]["curve"].get("identity"))
        m = by_mode[r.get("render_style")]
        m["n"] += 1
        m["identity"] += int(ident)
    rep["by_render_style"] = {k: {**v, "identity_share": round(v["identity"] / v["n"], 4)}
                              for k, v in sorted(by_mode.items())}

    released = [r for r in stamped if r.get("decision") == "selected"]
    for tag, rs in (("overall", stamped), ("smooth", split(stamped)[0]),
                    ("strange", split(stamped)[1]), ("released", released)):
        ident = sum(1 for r in rs if r["autolevel"]["curve"].get("identity"))
        rep[f"identity_{tag}"] = {"n": len(rs), "identity": ident,
                                  "share": round(ident / len(rs), 4) if rs else None}

    # --- why a non-identity row acted: which end moved ------------------------------------
    # `sides` is a SIGN per axis, not a flag: -1 = the render's statistic sits below the
    # committed band, +1 = above it, 0 = inside. A row can move on more than one axis, so
    # these counts are per AXIS and deliberately do not sum to the acting-row count.
    sides = Counter()
    axes_per_row = Counter()
    for r in stamped:
        c = r["autolevel"]["curve"]
        if c.get("identity"):
            continue
        s = c.get("sides") or {}
        moved = 0
        for axis, sign in s.items():
            if sign:
                sides[f"{axis}{'_below' if sign < 0 else '_above'}"] += 1
                moved += 1
        axes_per_row[moved] += 1
    rep["acting_rows_by_side"] = dict(sides)
    rep["acting_rows_axes_moved"] = dict(sorted(axes_per_row.items()))

    # --- chroma cap, over ACTING rows -----------------------------------------------------
    acting = [r for r in stamped if not r["autolevel"]["curve"].get("identity")]
    capped = [r for r in acting if (r["autolevel"].get("chroma_cap") or {}).get("n_capped", 0) > 0]
    frac = [(r["autolevel"]["chroma_cap"]["n_capped"], r["autolevel"]["chroma_cap"].get("n_stops") or 0)
            for r in capped]
    rep["chroma_cap"] = {
        "n_acting": len(acting),
        "n_rows_capped": len(capped),
        "row_share_of_acting": round(len(capped) / len(acting), 4) if acting else None,
        "retain": sorted({(r["autolevel"]["chroma_cap"] or {}).get("retain") for r in stamped}),
        "capped_stop_shares": [round(n / d, 4) for n, d in frac if d],
        "max_capped_stops": max((n for n, _ in frac), default=0),
    }

    # --- render cost ----------------------------------------------------------------------
    # NOTHING in the pool or colorize log carries a duration, and run 25's emission scratch
    # (the switch-OFF comparison) was wiped with the rest of scratch/ — so the cost is
    # measured WITHIN this run instead, as an A/B on its own attempts: each pooled row's JPG
    # mtime ordered ascending gives the per-attempt wall delta, grouped by whether the
    # operator acted. Same box, same geometry, same minute — a better control than run 25.
    cost = {"note": "one EXTRA render per ACTING row (the BEFORE render is what is measured)",
            "expected_multiplier_on_colorize": round(1 + (len(acting) / len(stamped)), 4) if stamped else None}
    if pool.exists():
        prows = [json.loads(l) for l in pool.read_text(encoding="utf-8").splitlines() if l.strip()]
        stamps = []
        for p in prows:
            jpg = p.get("jpg")
            al = p.get("autolevel") if isinstance(p.get("autolevel"), dict) else None
            if not jpg or not al:
                continue
            f = Path(jpg)
            if not f.is_absolute():
                f = ROOT / f
            if f.exists():
                stamps.append((f.stat().st_mtime, bool((al.get("curve") or {}).get("identity"))))
        stamps.sort()
        deltas = {"identity": [], "acting": []}
        for (t0, _), (t1, ident) in zip(stamps, stamps[1:]):
            deltas["identity" if ident else "acting"].append(t1 - t0)
        cost["pool_rows"] = len(prows)
        cost["timed_attempts"] = len(stamps)
        for k, v in deltas.items():
            cost[f"n_{k}"] = len(v)
            cost[f"median_s_{k}"] = round(sorted(v)[len(v) // 2], 2) if v else None
            cost[f"mean_s_{k}"] = round(sum(v) / len(v), 2) if v else None
        mi, ma = cost.get("median_s_identity"), cost.get("median_s_acting")
        cost["measured_multiplier"] = round(ma / mi, 3) if (mi and ma) else None
        cost["caveat"] = ("mtime deltas attribute the gap BEFORE a render to that render, so "
                          "the first attempt has no delta and any stall between attempts lands "
                          "on whichever row followed it")
    else:
        cost["timed_attempts"] = f"pool absent at {pool}"
    rep["render_cost"] = cost
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=str(DEFAULT_REC))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--pool", default=None,
                    help="default: the emission run's own scratch pool_log.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pool = Path(args.pool) if args.pool else paths.scratch("emission", args.run_id, "pool_log.jsonl")
    out = Path(args.out) if args.out else paths.scratch("autolevel_read", f"{args.run_id}.json")
    rep = read_report(Path(args.record), args.run_id, pool)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")

    print(f"run {args.run_id}: {rep['n_scored_rows']} scored rows ({rep['n_unstamped']} unstamped), "
          f"switch={rep['switch']}, disagreements={rep['acted_identity_disagreements']}")
    for tag in ("overall", "smooth", "strange", "released"):
        d = rep[f"identity_{tag}"]
        print(f"  identity {tag:<8} {pct(d['identity'], d['n'])}")
    print("  acting by side:", rep["acting_rows_by_side"],
          "| axes moved:", rep["acting_rows_axes_moved"])
    c = rep["chroma_cap"]
    print(f"  chroma cap: {pct(c['n_rows_capped'], c['n_acting'])} of acting rows capped, "
          f"max {c['max_capped_stops']} stops, retain={c['retain']}")
    print(f"  expected colorize render multiplier: "
          f"{rep['render_cost']['expected_multiplier_on_colorize']}")
    print("  by style:", {k: f"{v['identity']}/{v['n']}" for k, v in rep["by_render_style"].items()})
    print("->", out)


if __name__ == "__main__":
    main()
