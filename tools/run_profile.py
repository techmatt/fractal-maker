#!/usr/bin/env python
r"""run_profile.py — READ a finished run's stage timings: where the wall went, per stage, with
the outliers visible.

    uv run python tools/run_profile.py data/discovery/prod26_20260812 \
                                       data/discovery/prod26_20260812_dive \
                                       scratch/emission/prod26
    uv run python tools/run_profile.py <dir> --json --top 10

THE EMISSION LEG'S ROWS ARE NOT IN ITS `--out` DIR (since 2026-08-13): they land in the durable
`data/emission/run_telemetry/<run>/`, while `summary.json` stays in the scratch out dir. So an
emission out dir is given here exactly as before and this reader FOLLOWS it — first to the path
the run recorded for its own stream (`summary.stage_times_path`, which is also how an ephemeral
run's `--record-root` is found), then to the production home for a run of that name — and prints
the redirect, because a table drawn from a different directory than the one asked for must say
so. The telemetry dir may also be named directly; it simply has no summary to check against.

REPORT-SIDE ONLY, AND THAT IS THE DESIGN. Nothing here runs during a run, nothing here alerts,
nothing in the run reads it back. An in-run outlier detector would need a threshold that is
correct before the run has a median to compare against, and would then be a live cutoff on a
statistic computed from three units — `measurement_practice.md`'s small-sample rule, aimed at
the run's own telemetry. Outliers are a thing you SEE AFTERWARDS, ranked, with the population
they were ranked against printed beside them.

WHAT IT READS. `stage_times.jsonl` (tools/stage_times.py) for the per-unit rows, and
`summary.json` for the aggregates the run already published. Both, on purpose: the summary is
the run's own statement of `active_min`/`wall_min`, and quoting it beside the per-unit total is
what makes an UNCHARGED unit visible — if `frontier_batch.total_s` and `active_min * 60`
disagree, a batch ran that the quota never billed. That check is `--check`'s whole job and it
is relational (`verification_practice.md` §5) rather than a frozen literal.

ABSENCE IS REPORTED, NOT SKIPPED. A run dir with no `stage_times.jsonl` gets a loud
`PER-UNIT TIMING UNAVAILABLE` line naming the stream and the reason it can be legitimately
absent (the run predates it, 2026-08-12), and the summary-derived stage totals are still
printed — labelled `[summary]` so nobody reads an aggregate as a distribution. Every run before
run 27 is in that state, so a reader that silently printed an empty table would describe the
entire history of this pipeline as instantaneous (`verification_practice.md` §2).

THE OUTLIER RULE, and why it is this one. A unit is flagged when
`dur_s > OUTLIER_K * median(its stage)`, K = 3. Median, not mean, because the thing being
detected is exactly what drags a mean; a ratio, not an absolute, because the stages here differ
by three orders of magnitude (a colorize attempt is seconds, a release render is minutes) and
one threshold cannot serve both. A stage with fewer than `MIN_N_FOR_OUTLIERS` units gets NO
flags and says so: a median over 3 units is not a population to be an outlier from, and a
detector that fires on it reports the run's first slow unit every time.

UNACCOUNTED WALL. Rows carry `t_end`, so the span from the first unit's start to the last
unit's end is recoverable, and `span - sum(dur_s)` is the wall no stage claims — model loads,
checkpoint writes, teardown, the reaper's dead time. It is printed because a stage table that
sums to 60% of the wall clock and does not say so is the more misleading of the two numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Both roots: `tools.` is how every caller imports this package's modules, and `tools/` is on
# the path because that is how this tree's sys.path convention works (tools/README.md).
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools import stage_times as stimes           # noqa: E402
# The emission leg's run-keyed telemetry home is `emission_sinks`' to name, not this reader's
# to restate. It is pathlib-only, so importing it here costs nothing a reader would notice.
from tools.emission import emission_sinks as esinks  # noqa: E402

# A unit is an outlier at more than this multiple of its stage's median.
OUTLIER_K = 3.0
# Below this many units a stage's median is not a population; no flags, and the table says so.
MIN_N_FOR_OUTLIERS = 5

# Summary keys that are already a stage total, as `(summary key, stage, seconds-per-unit)`.
# Read so a run predating `stage_times.jsonl` still profiles at stage granularity, and so the
# cross-checks have something to hold the per-unit total against.
#
# KEYED BY MODE, because `active_min` names a DIFFERENT stage in each: in `steered` it is the
# batch loop, in `dive` it is the dive loop. One flat table would both fail to check the dive
# total and report a permanent unknown ("frontier_batch vs active_min") on every dive dir —
# a red lane that protects nothing (`verification_practice.md` §4).
SUMMARY_STAGE_KEYS = {
    "steered": (("pre_loop_draw_min", "root_draw", 60.0),
                ("root_draw_min", "root_refill", 60.0),
                ("active_min", "frontier_batch", 60.0)),
    "dive": (("active_min", "dive", 60.0),),
    # An emission out dir has no `mode` and publishes no stage aggregate to check against;
    # its stage table stands on the per-unit rows alone.
    None: (),
}


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def _quantile(xs: list[float], q: float) -> float:
    """Nearest-rank quantile on a sorted list. Nearest-rank, not interpolated: every number
    this prints is then a duration some unit ACTUALLY took, which is what makes a p90 next to
    a top-N list checkable by eye."""
    if not xs:
        return 0.0
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def _round_slack_s(rows: list[dict], stage: str) -> float:
    """Seconds of DISAGREEMENT the summary check must forgive for `stage`, from rounding alone.

    In-run rows are full-precision and get 0. A backfilled row is a printed duration, so it
    carries at most half its print quantum of error, and the stage total carries the sum —
    which for run 26's single 30 s refill is 0.5 s against a 29.4 s aggregate, i.e. 1.7%, and
    tripped a flat 2% tolerance on a stream that was in fact exact. A tolerance that fires on
    the arithmetic of its own inputs is `verification_practice.md` §4: it goes red during
    ordinary use and gets trained out."""
    tot = 0.0
    for r in rows:
        if str(r.get("stage")) != stage or not r.get("source"):
            continue
        tot += float((r.get("meta") or {}).get("quant_s", 1.0)) / 2.0
    return tot


def stage_stats(rows: list[dict]) -> dict:
    """Per-stage distribution over already-read rows. Stages in `stage_times.STAGES` order,
    then any unknown stage alphabetically — an unknown stage is a caller this module has not
    been taught about, and it is shown rather than dropped."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r.get("stage")), []).append(r)
    order = [s for s in stimes.STAGES if s in by] + sorted(set(by) - set(stimes.STAGES))
    out = {}
    for s in order:
        ds = sorted(float(r.get("dur_s") or 0.0) for r in by[s])
        med = _quantile(ds, 0.5)
        out[s] = {
            "n": len(ds), "total_s": round(sum(ds), 2),
            "min_s": round(ds[0], 2), "p50_s": round(med, 2),
            "p90_s": round(_quantile(ds, 0.9), 2), "max_s": round(ds[-1], 2),
            "mean_s": round(sum(ds) / len(ds), 2),
            "outlier_threshold_s": (round(OUTLIER_K * med, 2)
                                    if len(ds) >= MIN_N_FOR_OUTLIERS else None),
        }
    return out


def outliers(rows: list[dict], stats: dict, top: int) -> list[dict]:
    """The slowest units overall, each with its ratio to its own stage's median and whether
    that clears `OUTLIER_K`. Sorted by RATIO, not by raw seconds: sorting by seconds returns
    the release renders every time and buries the colorize attempt that took 30x its peers."""
    out = []
    for r in rows:
        s = str(r.get("stage"))
        st = stats.get(s)
        if not st:
            continue
        med = st["p50_s"]
        dur = float(r.get("dur_s") or 0.0)
        ratio = (dur / med) if med > 0 else None
        eligible = st["n"] >= MIN_N_FOR_OUTLIERS
        out.append({
            "stage": s, "unit": r.get("unit"), "dur_s": round(dur, 2),
            "stage_p50_s": med, "ratio": (round(ratio, 2) if ratio is not None else None),
            "flagged": bool(eligible and ratio is not None and ratio > OUTLIER_K),
            # Why it was slow is usually in the meta the writer attached; carried verbatim.
            "meta": r.get("meta") or {},
            "stage_n": st["n"],
        })
    out.sort(key=lambda d: (-(d["ratio"] or 0.0), -d["dur_s"]))
    return out[:top]


def span_of(rows: list[dict]) -> dict:
    """Wall span covered by the rows and how much of it no stage claims.

    `t_end` is an end time, so the span starts at the first unit's IMPLIED start
    (`t_end - dur_s`) — using the first `t_end` would silently drop the first unit's own
    duration out of the span and inflate the unaccounted share."""
    ts = [(float(r["t_end"]) - float(r.get("dur_s") or 0.0), float(r["t_end"]))
          for r in rows if r.get("t_end") is not None]
    if not ts:
        return {"span_s": None, "accounted_s": None, "unaccounted_s": None,
                "accounted_frac": None}
    lo = min(a for a, _ in ts)
    hi = max(b for _, b in ts)
    acc = sum(float(r.get("dur_s") or 0.0) for r in rows)
    span = hi - lo
    return {"span_s": round(span, 1), "accounted_s": round(acc, 1),
            "unaccounted_s": round(span - acc, 1),
            "accounted_frac": (round(acc / span, 4) if span > 0 else None)}


# --------------------------------------------------------------------------- #
# one run dir
# --------------------------------------------------------------------------- #
def stream_dir_for(run_dir, summary: dict | None = None) -> Path:
    """Where this run dir's per-unit rows actually are, in decreasing order of authority.

    The dir itself if it holds the stream; else the path the run RECORDED for its own stream
    (`summary.stage_times_path`, written at the write site from the resolved path, so it is
    right for an ephemeral run's `--record-root` too — derived state, not a guess); else the
    emission leg's production run-keyed home for a run of that name. A recorded path that no
    longer exists falls through rather than deciding the answer: the record is of where the run
    wrote, and the file can be gone."""
    d = Path(run_dir)
    if (d / stimes.STREAM).exists():
        return d
    recorded = (summary or {}).get("stage_times_path")
    if recorded and Path(recorded).exists():
        return Path(recorded).parent
    alt = esinks.run_telemetry_dir(esinks.default_record_root(ROOT), d.name)
    return alt if (alt / stimes.STREAM).exists() else d


def profile_dir(run_dir, top: int = 8) -> dict:
    d = Path(run_dir)
    summary = {}
    sp = d / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))
    sd = stream_dir_for(d, summary)
    rows, diag = stimes.read(sd)
    stats = stage_stats(rows)
    prof = {
        "run_dir": str(d), "present": diag["present"], "stream_diag": diag,
        # Only when it is NOT the dir asked for — a null here means "read where you pointed".
        "stream_dir": (str(sd) if sd != d else None),
        "mode": summary.get("mode"),
        "stages": stats,
        "span": span_of(rows),
        "top": outliers(rows, stats, top),
        # The run's OWN aggregates, quoted rather than recomputed. `--check` holds the
        # per-unit totals against these; a reader without the stream gets only these.
        "summary_aggregates": {k: summary[k] for k in
                               ("active_min", "wall_min", "pre_loop_draw_min", "root_draw_min",
                                "batches", "n_dives_done", "wall_over_active", "attempts",
                                "release_rendered")
                               if k in summary},
        "summary_stage_times": summary.get("stage_times"),
        # A backfilled stream is not an in-run measurement and must never be read as one.
        "sources": sorted({str(r["source"]) for r in rows if r.get("source")}),
    }
    prof["checks"] = _checks(prof, rows, summary)
    return prof


def _checks(prof: dict, rows: list[dict], summary: dict) -> list[dict]:
    """Relational checks between the per-unit stream and the run's own published aggregates.

    Each returns a verdict of `ok` / `off` / `unknown`. **`unknown` is a distinct outcome from
    `ok`** — an aggregate this run never published cannot confirm anything, and reporting that
    as a pass is `verification_practice.md` §2 exactly."""
    out = []
    stats = prof["stages"]
    mode = summary.get("mode")
    # An unrecognised mode gets the steered table rather than nothing: a mode this reader has
    # not been taught about should be checked with the closest thing it has and be visibly
    # wrong, not sail through with zero checks.
    keys = SUMMARY_STAGE_KEYS.get(mode, SUMMARY_STAGE_KEYS["steered"] if mode else ())
    for key, stage, unit_s in keys:
        have = stats.get(stage)
        agg = summary.get(key)
        if not have and agg is None:
            # Neither side exists — a stage this mode never runs. Nothing is being checked
            # and nothing is being hidden, so it is not a finding. (Contrast the branch
            # below, where ONE side exists: that IS the finding.)
            continue
        if not have or agg is None:
            out.append({"check": f"{stage} vs summary.{key}", "verdict": "unknown",
                        "why": ("no per-unit rows for this stage" if not have
                                else f"summary has no `{key}`")})
            continue
        want = float(agg) * unit_s
        got = have["total_s"]
        # The summary's own value is rounded too (`round(x/60, 2)` minutes = ±0.3 s), on top of
        # whatever the rows cost. Both are arithmetic, not disagreement.
        slack = _round_slack_s(rows, stage) + 0.3
        diff = abs(got - want)
        rel = diff / want if want > 0 else None
        out.append({"check": f"{stage} vs summary.{key}",
                    "verdict": ("ok" if (diff <= slack or (rel is not None and rel <= 0.02))
                                else "off"),
                    "stream_s": got, "summary_s": round(want, 1),
                    "rel_diff": (round(rel, 4) if rel is not None else None),
                    "slack_s": round(slack, 1)})
    # Row COUNT against the run's own unit count, per mode. Cheaper than the seconds check and
    # catches a different failure: a unit that ran and wrote no row at all.
    for key, stage in (("batches", "frontier_batch"), ("n_dives_done", "dive")):
        n = summary.get(key)
        if n is None or stage not in stats:
            continue
        got = stats[stage]["n"]
        out.append({"check": f"{stage} rows vs summary.{key}", "stream_n": got,
                    "summary_n": n, "verdict": ("ok" if got == n else "off")})
    return out


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt(prof: dict, top: int) -> str:
    L = []
    L.append(f"=== {prof['run_dir']}  (mode={prof.get('mode')})")
    d = prof["stream_diag"]
    if not prof["present"]:
        L.append(f"  !! PER-UNIT TIMING UNAVAILABLE - no `{stimes.STREAM}` in this run dir, "
                 f"nor in the durable emission telemetry home for a run of this name.")
        L.append("     Legitimate for any run started before 2026-08-12 (the stream did not "
                 "exist); for a later run it means the record is gone.")
    else:
        if prof.get("stream_dir"):
            L.append(f"  rows read from {prof['stream_dir']} (the run-keyed telemetry home; "
                     f"this dir holds the summary)")
        extra = []
        if d["n_torn"]:
            extra.append(f"{d['n_torn']} torn line(s) (kill mid-append)")
        if d["n_dup"]:
            extra.append(f"{d['n_dup']} duplicate unit(s) dropped, last kept (resume replay)")
        L.append(f"  stream: {d['n_raw']} raw row(s)" + ("; " + "; ".join(extra) if extra else ""))
    if prof["sources"]:
        L.append(f"  !! source={','.join(prof['sources'])} - these rows were NOT measured "
                 f"in-run; read their resolution accordingly.")
    if prof["summary_aggregates"]:
        L.append("  summary: " + "  ".join(f"{k}={v}" for k, v in
                                           prof["summary_aggregates"].items()))
    if prof["stages"]:
        tot = sum(s["total_s"] for s in prof["stages"].values()) or 1.0
        L.append(f"  {'stage':<16}{'n':>5}{'total_m':>9}{'share':>7}{'min':>8}{'p50':>8}"
                 f"{'p90':>8}{'max':>9}   outlier>")
        for name, s in prof["stages"].items():
            thr = (f"{s['outlier_threshold_s']:.1f}s" if s["outlier_threshold_s"] is not None
                   else f"n<{MIN_N_FOR_OUTLIERS}")
            L.append(f"  {name:<16}{s['n']:>5}{s['total_s']/60:>9.1f}"
                     f"{s['total_s']/tot:>7.1%}{s['min_s']:>8.1f}{s['p50_s']:>8.1f}"
                     f"{s['p90_s']:>8.1f}{s['max_s']:>9.1f}   {thr:>8}")
        sp = prof["span"]
        if sp["span_s"]:
            L.append(f"  span {sp['span_s']/60:.1f}m wall, {sp['accounted_s']/60:.1f}m in "
                     f"stages ({sp['accounted_frac']:.1%}); "
                     f"{sp['unaccounted_s']/60:.1f}m unaccounted (loads, checkpoints, "
                     f"teardown, dead time)")
    elif prof["summary_stage_times"]:
        L.append("  [summary] stage totals only (no per-unit rows): "
                 + json.dumps(prof["summary_stage_times"]))
    for c in prof["checks"]:
        mark = {"ok": "ok ", "off": "OFF", "unknown": "?  "}[c["verdict"]]
        L.append(f"  [{mark}] {c['check']}: " +
                 ", ".join(f"{k}={v}" for k, v in c.items()
                           if k not in ("check", "verdict")))
    if prof["top"]:
        nfl = sum(1 for t in prof["top"] if t["flagged"])
        L.append(f"  top {len(prof['top'])} by ratio-to-stage-median "
                 f"({nfl} over {OUTLIER_K}x):")
        for t in prof["top"]:
            m = " ".join(f"{k}={v}" for k, v in list(t["meta"].items())[:4])
            L.append(f"    {'FLAG' if t['flagged'] else '    '} {t['stage']:<16}"
                     f"{str(t['unit']):<24}{t['dur_s']:>8.1f}s "
                     f"{(str(t['ratio']) + 'x') if t['ratio'] is not None else '   -':>7} "
                     f"of p50 {t['stage_p50_s']:.1f}s (n={t['stage_n']})  {m}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+",
                    help="one or more run dirs (breadth, dive, emission out) — profiled "
                         "independently and printed in the order given")
    ap.add_argument("--top", type=int, default=8,
                    help="units to list per run dir, ranked by ratio to their stage median")
    ap.add_argument("--json", action="store_true", help="machine-readable profile to stdout")
    args = ap.parse_args()
    profs = [profile_dir(d, top=args.top) for d in args.run_dir]
    if args.json:
        print(json.dumps(profs, indent=2))
    else:
        print("\n\n".join(_fmt(p, args.top) for p in profs))


if __name__ == "__main__":
    main()
