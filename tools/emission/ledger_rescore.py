#!/usr/bin/env python
r"""ledger_rescore.py — bring the stage-2 intake ledgers CURRENT under the active head.

WHY. A ledger's decode block (`decoded_class`, `p_notbad`, `p_good`, `t_good`) is one head's
verdict, and `corpus_common.is_current_decoded` correctly refuses to consume it under another.
At the v10 flip that took the whole emission intake from ~1.4k admissible locations to **16**
— every non-classic ledger is still v7-stamped, so `descriptor.load_admitted` rejects it. The
locations did not stop existing; only their verdicts went stale. This re-derives the verdicts.

WHAT IT DOES NOT DO. It never touches an original ledger. A discovery ledger is that run's
record of what it found AND what the head of the day said about it, so overwriting the decode
block erases the only evidence of the previous operating point. The output is a SIBLING record
named for the ledger stem and the head version, `<stem>.rescored_<version>.jsonl`, which
`descriptor.resolve_rows` overlays at read time (see `descriptor.RESCORE_SUFFIX_FMT` for why
the version is in the name). Re-running after the NEXT flip writes a new sibling; the old one
stays as the record of what v10 said.

THE HEAD IS NEVER PINNED HERE — scorer, stamp and threshold all resolve from the live pins
(`production_seeder.SCORER_PATH`/`SCORER_VERSION`, `t_good_for`,
`production_pins.auto_maxiter`), so this is the same "re-mint, don't reproduce" shape
`classic_phoenix_supply` follows. Resume is by id WITHIN the version-named file, which cannot
carry another head's rows, so the resume-key bug that made a flip read as "already done" is
structurally absent rather than purged.

PRESENTATION. Each row's canonical `Location` (`descriptor.location_of` — julia twins resolve
through the asserted schema tag, phoenix carries its full parameter point) is re-rendered at
the deploy presentation, 640x360 ss2 (`reframe.RENDER_*`, the mirror of the classifier's
384x224 stretch), under the LIVE `auto_maxiter` policy — which is not the policy the original
rows were rendered under (base 500 -> 4000 on 2026-07-31), so every row carries
`maxiter_policy_token` and a re-score can never be pooled with a legacy-policy score.

GUARD. Escape-time families dump the co-located guard field and score through
`guard.make_guarded_scorer`, exactly as `q4_harvest_ledger` does. Phoenix has no escape-time
backend, so no field is dumped and the guarded scorer passes the tile through unguarded —
which is what `classic_phoenix_supply` already does for the same reason. `guard_source` records
which of the two happened, so a phoenix `guard_pass` is never mistaken for a re-checked one.

WHAT IS CARRIED, NOT RE-DERIVED. `distinct`/`dup_of` are the run's own morphology dedup against
its own cloud — a property of the population, not of the head — so they come across from the
original row untouched. Re-deriving them here would require the run's cloud, which no longer
exists, and would silently redefine the admitted set on a dimension this pass has no business
touching.

  uv run python tools/emission/ledger_rescore.py --limit 3           # smoke
  uv run python tools/emission/ledger_rescore.py --only q4_harvest   # one ledger
  uv run python tools/emission/ledger_rescore.py > scratch/ledger_rescore.log 2>&1  &
  uv run python tools/emission/ledger_rescore.py status              # read-only census
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "reframe",
           ROOT / "tools" / "scoring", ROOT / "tools" / "mining", ROOT / "tools" / "corpus",
           ROOT / "tools" / "orbital", ROOT / "tools" / "explorer"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import corpus_common as cc                      # noqa: E402
import guard                                    # noqa: E402
import reframe                                  # noqa: E402
import production_seeder as ps                  # noqa: E402
import location as loc_mod                      # noqa: E402
import field_metrics as fm                      # noqa: E402  (POLICY_KEY — the axis's owner)
import paths as _paths                          # noqa: E402
from score_lib import corn_decode               # noqa: E402
import partitions as P                          # noqa: E402  (THE partition map + the split)
import release_mix as RM                        # noqa: E402  (THE release-mix ratio table)
import supply_routing as srt                    # noqa: E402  (THE channel table; pure data)
from tools.emission import descriptor as D      # noqa: E402

# ---------------------------------------------------------------------------- #
# THE stage-2 intake population. These seven are what `stage_first_release`'s six library
# ledgers plus the q4_harvest supply resolve to on disk; the survey's decode-currency census
# is taken over exactly this list. `classic_phoenix` is here so the pass VERIFIES it is
# already current rather than a reader having to remember that it is.
# ---------------------------------------------------------------------------- #
LEDGERS = (
    ("c1_breadth",      "data/discovery/campaign1/breadth/outcome_ledger.jsonl"),
    ("c1_dive",         "data/discovery/campaign1/dive/outcome_ledger.jsonl"),
    ("c2_breadth",      "data/discovery/campaign2/breadth/outcome_ledger.jsonl"),
    ("c2_dive",         "data/discovery/campaign2/dive/outcome_ledger.jsonl"),
    ("phoenix_grid",    "data/discovery/phoenix_grid/grid/outcome_ledger_v7_t45.jsonl"),
    ("classic_phoenix", "data/discovery/classic_phoenix/outcome_ledger.jsonl"),
    ("q4_harvest",      "data/emission/q4_harvest/outcome_ledger.jsonl"),
)

SCRATCH = ROOT / "scratch" / "emission" / "ledger_rescore"
RENDER_W, RENDER_H, RENDER_SS = reframe.RENDER_W, reframe.RENDER_H, reframe.RENDER_SS
# Families with a `--dump-field-source f64` escape-time backend. Phoenix has none (see GUARD
# in the module docstring), so it is the sole exclusion and it is named, not inferred.
NO_GUARD_FIELD_FAMILIES = frozenset({"phoenix"})


def log(msg: str):
    print(msg, flush=True)


def ledger_path(rel: str) -> Path:
    return ROOT / rel


def _rel(p: Path) -> str:
    """Repo-relative when in-tree, absolute otherwise — a `relative_to` that raises on an
    out-of-tree ledger would make the whole pass untestable against a fixture."""
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------- #
# One row.
# ---------------------------------------------------------------------------- #
def _score_row(scorer, row: dict, tile: Path) -> dict:
    """Render + guarded-score + decode ONE ledger row at the deploy presentation.

    Returns the decode block only; the caller merges it onto the original row."""
    loc = D.location_of(row)                    # canonical: julia/phoenix/multibrot resolved
    cand = {"cx": str(loc.cx), "cy": str(loc.cy), "fw": float(loc.fw),
            "maxiter": int(loc.maxiter)}        # loc.maxiter IS the live auto_maxiter policy
    want_field = loc_mod.family_of(loc) not in NO_GUARD_FIELD_FAMILIES
    prev, reframe.DUMP_GUARD_FIELD = reframe.DUMP_GUARD_FIELD, want_field
    try:
        ok, err = reframe._render(loc, cand, tile, RENDER_W, RENDER_H, RENDER_SS)
    finally:
        reframe.DUMP_GUARD_FIELD = prev
    if not ok:
        raise RuntimeError(f"render failed: {err}")

    triple = scorer.score_paths_k([tile])[0]    # K-aware: (score, p_ge2, p_ge3[, p_ge4])
    score, notbad, good = float(triple[0]), float(triple[1]), float(triple[2])
    great = float(triple[3]) if len(triple) > 3 else None
    guard_pass = score > guard.GUARD_SENTINEL + 1e-6
    if not guard_pass:
        notbad = good = 0.0
        great = None if great is None else 0.0

    reason = None
    if want_field:
        # The reason (not just the verdict) for the reject autopsy, off the SAME field the
        # guarded scorer gated on — so the two can never disagree silently.
        from colormap import load_field
        stats = guard.field_measures(load_field(guard.field_sidecar_for(tile)).values)
        reason = guard.guard_fail(stats.interior_frac, stats.field_std)
        assert guard_pass == (reason is None), \
            f"guard disagreement: sentinel_pass={guard_pass} field_reason={reason}"

    # THE PARTITION, RESOLVED FROM THE ROW — not `row["family"]`, which is the ledger's
    # partition for the nine BASE partitions and wrong for exactly one: a classic-phoenix row
    # says `phoenix`. This read `t_good_for(row["family"])` until 2026-08-08 and the bug was
    # invisible for as long as it mattered least — phoenix and phoenix:classic were BOTH
    # UNCALIBRATED at 0.50, so the wrong key returned the right number. The v11 flip
    # calibrated phoenix at 0.77 and left phoenix:classic at the baseline, at which point the
    # first re-score minted all 24 classic rows against 0.77 (max p_good 0.639), decoded every
    # one to class 1, and took the partition's entire admitted supply to zero — which the
    # liveness census caught, and which reads as "v11 lost classic phoenix" rather than as a
    # key resolution bug. `classic_phoenix_servable` below already used the row resolver;
    # these two sites now agree.
    t = ps.t_good_for(P.partition_of_row(row, row.get("family")))
    decoded = corn_decode(notbad, good, t, great) if guard_pass else None
    return {
        "decoded_class": decoded, "p_notbad": notbad, "p_good": good, "p_ge4": great,
        "t_good": t, "canon_pgood": good,
        "guard_pass": bool(guard_pass), "guard_fail": reason,
        "guard_source": "field" if want_field else "none (no escape-time backend)",
        "scorer_version": ps.SCORER_VERSION,
        fm.POLICY_KEY: fm.policy_token(),       # live policy, under the axis's own key
        "rescore_maxiter": int(loc.maxiter),
        "rescore_presentation": f"{RENDER_W}x{RENDER_H}ss{RENDER_SS}",
    }


# ---------------------------------------------------------------------------- #
# One ledger.
# ---------------------------------------------------------------------------- #
def rescore_ledger(tag: str, rel: str, scorer, *, limit: int | None = None) -> dict:
    src = ledger_path(rel)
    out = D.rescore_path(src)
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    already = [r for r in rows if cc.is_current_decoded(r)]
    if len(already) == len(rows):
        # VERIFY, don't redo. A ledger the active head already wrote is current by
        # construction; re-rendering it would burn the budget to reproduce its own numbers.
        log(f"[{tag}] {len(rows)} rows ALREADY current ({cc.active_scorer_version()}) — "
            f"verified, not re-scored")
        return dict(tag=tag, ledger=rel, n_rows=len(rows), n_rescored=0, n_failed=0,
                    already_current=True, sibling=None)

    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    todo = [r for r in rows if r["id"] not in done]
    if limit:
        todo = todo[:limit]
    log(f"[{tag}] {len(rows)} rows ({len(already)} already current); "
        f"{len(done)} in sibling, {len(todo)} to score -> {_rel(out)}")

    # durable(): this records a scoring pass over a population, at a cost of hours; it must
    # survive `rm -r scratch/*`. Asserted not-gitignored at the write site. An out-of-tree
    # ledger (a fixture) has no class to declare — `durable()` would raise on a path it
    # cannot relate to the repo, and a fixture is not making a durability claim.
    rel_out = _rel(out)
    if not Path(rel_out).is_absolute():
        _paths.durable(rel_out, mkparents=True)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "a", encoding="utf-8")
    t0 = time.time()
    failed = Counter()
    n_ok = 0
    try:
        for k, row in enumerate(todo):
            work = SCRATCH / tag / row["id"]
            try:
                block = _score_row(scorer, row, work / "tile.jpg")
            except Exception as e:                              # noqa: BLE001
                failed[f"{type(e).__name__}: {str(e)[:100]}"] += 1
                log(f"  WARN {tag}/{row['id']} {type(e).__name__}: {str(e)[:140]}")
                shutil.rmtree(work, ignore_errors=True)
                continue
            fh.write(json.dumps({**row, **block, "rescored_from": rel}) + "\n")
            fh.flush()
            n_ok += 1
            shutil.rmtree(work, ignore_errors=True)
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                rate = (k + 1) / el if el else 0.0
                log(f"[{tag}] {k+1}/{len(todo)}  ({rate:.2f} row/s, "
                    f"ETA {(len(todo)-k-1)/rate/60:.1f}m)" if rate else f"[{tag}] {k+1}/{len(todo)}")
    finally:
        fh.close()
    return dict(tag=tag, ledger=rel, n_rows=len(rows), n_rescored=n_ok,
                n_failed=int(sum(failed.values())), fail_reasons=dict(failed.most_common(5)),
                already_current=False, sibling=_rel(out))


# ---------------------------------------------------------------------------- #
# Census — what the intake sees, before and after.
# ---------------------------------------------------------------------------- #
def census() -> dict:
    """Per-ledger: rows, rows the ACTIVE head has a verdict for (after overlay), and rows
    `descriptor.load_admitted` admits. Read-only; renders nothing."""
    per, tot_rows, tot_cur, tot_adm = [], 0, 0, 0
    for tag, rel in LEDGERS:
        src = ledger_path(rel)
        raw = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        resolved = D.resolve_rows(src)
        cur = sum(1 for r in resolved if cc.is_current_decoded(r))
        adm = D.load_admitted(src)
        classes = Counter(r.get("decoded_class") for r in resolved if cc.is_current_decoded(r))
        per.append(dict(tag=tag, ledger=rel, n_rows=len(raw), n_current=cur,
                        n_admitted=len(adm),
                        sibling_rows=len(D.load_rescored(src)),
                        decoded_class_hist={str(k): v for k, v in sorted(
                            classes.items(), key=lambda kv: (kv[0] is None, kv[0]))},
                        floor_admit=D.source_tag_of(raw[0]) in D.FLOOR_ADMIT_SOURCES if raw else None))
        tot_rows += len(raw)
        tot_cur += cur
        tot_adm += len(adm)
    return dict(active_version=cc.active_scorer_version(),
                maxiter_policy_token=loc_mod.maxiter_policy_token(),
                per_ledger=per, total_rows=tot_rows, total_current=tot_cur,
                total_admitted=tot_adm)


def intake_union() -> dict:
    """What stage 2 ACTUALLY intakes: `descriptor.load_union_admitted` over the seven ledgers.

    THE union reader, not a mirror of it — the driver calls the same function, so "what the
    census says stage 2 would intake" and "what stage 2 intakes" cannot be two numbers.

    Row identity is namespaced by ledger there, so the run-scoped id collisions between
    campaign1 and campaign2 (`st_<fam>_<arm>_<seq>` reused for DIFFERENT locations) no longer
    alias and no longer abort the driver; they are still COUNTED here, because the count is
    what says the namespacing is doing work. Deduplication is by LOCATION identity."""
    _rows, diag = D.load_union_admitted([ledger_path(rel) for _t, rel in LEDGERS])
    return dict(n_union=diag["n_union"], n_collisions=diag["n_id_collisions"],
                n_benign_overlaps=diag["n_location_overlaps"],
                collision_sample=diag["collision_sample"],
                overlap_sample=diag["overlap_sample"], driver_reachable=True)


# ---------------------------------------------------------------------------- #
# The externally-supplied supply check. Where the visibility that came OUT of the crawl
# census (`steered_frontier.deferred_partitions` SKIP SITE 1) went back in.
# ---------------------------------------------------------------------------- #
def classic_supply_note(ledger: Path | None = None, n_union: int | None = None) -> dict:
    """Servable `phoenix:classic` count at intake, and whether it is BELOW what the release
    mix asks of it.

    `phoenix:classic` is EXTERNALLY SUPPLIED (`supply_routing`): no crawl produces it, so the
    only place its supply can be noticed is here, where the intake population is known. The
    crawl used to report it as starved every batch, which is a permanent false alarm; this is
    the same fact stated once, at the moment somebody could act on it.

    SERVABLE = admitted through its own ledger (`descriptor.load_admitted`: current-decode ∧
    guard ∧ distinct ∧ q3) AND still >= 3 when re-decoded at THIS partition's own threshold,
    `t_good_for("phoenix:classic")`. The re-decode is not redundant: a ledger row carries the
    `t_good` it was minted under (0.5 on every classic row today), and a later per-partition
    derivation moves the cut without rewriting a single row — reading the stamped
    `decoded_class` would report a count against a threshold nobody is using any more.

    THE LOW-WATER IS DERIVED, NOT DECLARED. It is what the committed release mix asks for at
    this intake's size: `release_mix.shares()["phoenix:classic"] * n_union`, rounded up. A
    hardcoded "top up below 10" would be a number nobody decided that goes stale the moment
    the ratio table or the library size moves; this one moves with both. The hint names the
    command from the routing table (`supply_routing.supply_command`), so there is no second
    copy of what `classic_plane_descent` actually is.

    NO AUTOMATED TOP-UP. Printed count + manual run is the intended first version: the descent
    is a GPU job of its own and a census command that silently launched one would be a very
    surprising thing for `ledger_rescore status` to do."""
    part = P.CLASSIC_PHOENIX
    if ledger is None:
        ledger = ledger_path(dict((t, r) for t, r in LEDGERS)["classic_phoenix"])
    t_good = ps.t_good_for(part)
    n_admitted = n_servable = 0
    for row in D.load_admitted(Path(ledger)):
        if P.partition_of_row(row, row.get("family")) != part:
            continue
        n_admitted += 1
        if corn_decode(row["p_notbad"], row["p_good"], t_good, row.get("p_ge4")) >= 3:
            n_servable += 1
    if n_union is None:
        n_union = intake_union()["n_union"]
    share = RM.shares()[part]
    wanted = int(math.ceil(share * float(n_union)))
    return dict(partition=part, externally_supplied=srt.is_externally_supplied(part),
                ledger=_rel(Path(ledger)), t_good=t_good, t_good_status=ps.t_good_status(part),
                n_admitted=n_admitted, n_servable=n_servable,
                release_share=share, n_union=n_union, wanted=wanted,
                low=bool(n_servable < wanted), command=srt.supply_command(part))


def print_classic_supply_note(note: dict):
    st = "LOW" if note["low"] else "ok"
    log(f"  {note['partition']} servable: {note['n_servable']}/{note['n_admitted']} admitted "
        f"at t_good {note['t_good']:.2f} ({note['t_good_status']}) — "
        f"release mix asks ~{note['wanted']} of a {note['n_union']}-row intake "
        f"({note['release_share']:.2%}) — {st}")
    if note["low"] and note["externally_supplied"]:
        log(f"    EXTERNALLY SUPPLIED — no crawl makes classic. Top up manually: "
            f"{note['command']}")


def print_census(c: dict):
    log(f"\n=== INTAKE CENSUS (head {c['active_version']}, "
        f"maxiter policy {c['maxiter_policy_token']}) ===")
    log(f"  {'ledger':18s} {'rows':>5s} {'current':>8s} {'admitted':>9s}  decoded_class")
    for p in c["per_ledger"]:
        log(f"  {p['tag']:18s} {p['n_rows']:5d} {p['n_current']:8d} {p['n_admitted']:9d}"
            f"  {p['decoded_class_hist']}")
    log(f"  {'TOTAL':18s} {c['total_rows']:5d} {c['total_current']:8d} {c['total_admitted']:9d}")
    u = intake_union()
    log(f"  union stage 2 intakes: {u['n_union']} "
        f"({u['n_benign_overlaps']} cross-ledger same-location overlaps dropped)")
    if u["n_collisions"]:
        log(f"  {u['n_collisions']} run-scoped id collision(s), namespaced apart by ledger and "
            f"admitted as the distinct locations they are (a bare id-keyed union would have "
            f"dropped them). e.g. {u['collision_sample'][:3]}")
    print_classic_supply_note(classic_supply_note(n_union=u["n_union"]))


# ---------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=("run", "status"))
    ap.add_argument("--limit", type=int, default=None, help="cap rows PER LEDGER (smoke)")
    ap.add_argument("--only", nargs="+", default=None, help="ledger tags to process")
    ap.add_argument("--threads", type=int, default=cc.DEFAULT_ENGINE_THREADS,
                    help="RAYON_NUM_THREADS for the engine child (one process at a time)")
    args = ap.parse_args(argv)

    if args.cmd == "status":
        c = census()
        print_census(c)
        return 0

    # One engine process at a time, 7 threads, BELOW_NORMAL — the committed single-engine
    # default. `reframe._render` takes neither env nor creationflags, so the pair is applied
    # to THIS process and inherited by every child it spawns.
    import os
    os.environ["RAYON_NUM_THREADS"] = str(int(args.threads))
    log(f"[priority] {cc.set_below_normal_priority()}  RAYON_NUM_THREADS={args.threads}")

    assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX
    scorer = guard.make_guarded_scorer(ps.SCORER_PATH)
    log(f"=== ledger re-score: GUARDED CORN ({ps.SCORER_PATH}, {ps.SCORER_VERSION}), "
        f"maxiter policy {loc_mod.maxiter_policy_token()} ===")
    SCRATCH.mkdir(parents=True, exist_ok=True)

    todo = [(t, r) for t, r in LEDGERS if not args.only or t in set(args.only)]
    results = []
    t0 = time.time()
    for tag, rel in todo:
        results.append(rescore_ledger(tag, rel, scorer, limit=args.limit))
    log(f"\n=== re-scored {sum(r['n_rescored'] for r in results)} rows in "
        f"{(time.time()-t0)/60:.1f}m ({sum(r['n_failed'] for r in results)} failed) ===")
    for r in results:
        log(f"  {r['tag']:18s} rescored={r['n_rescored']:4d} failed={r['n_failed']:3d}"
            + (f"  {r.get('fail_reasons')}" if r.get("fail_reasons") else ""))
    print_census(census())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
