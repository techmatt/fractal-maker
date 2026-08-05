#!/usr/bin/env python
r"""precanon_minfw_replay.py — replay one run's admission path under both dedup radius scales.

THE QUESTION (asked while production was still `1.5 x max(fw)`; ADOPTED `0.25 x min(fw)`
2026-08-04 partly on this measurement). `production_seeder.is_distinct` scaled its dedup disc
by `1.5 * max(fw)`.
The fate sheet established that in 28/28 top-tier pairs the displacer is the WIDER frame
(median 17.3x), so `max(fw)` is set by the displacer every time and what the cut deletes is a
deep zoom inside a wide outcome's disc. `min(fw)` — the scale-aware house pattern — is staged
as `ps.DEDUP_SCALE`. This tool MEASURES the flip on a whole run's population. It decides
nothing and writes nothing under `data/`.

THE POPULATION is every harvest check of `harvest_v2_proving_20260803`: 2,767 rows, of which
2,531 died as `precanon_dup`. Two records are joined 1:1 on (batch, node_id, cx, cy, fw):
`q4_candidates.jsonl` (identity, coords, fate, canonical decode, reframed outcome) and
`harvest_log.jsonl` (the `precanon_dup` id — the only place the displacer is written down).

THE REPLAY REPRODUCES THE RUN'S ITERATION ORDER AND CLOUD STATE, and refuses to report
anything until it has re-fired the recorded run exactly (`--scale max` self-check: every
`precanon_dup` id, every `q3_dup`, every ledger `distinct`/`dup_of`, every admission). A
replay that cannot reproduce the original is not evidence. The run makes this possible:
`freshness_prior` was OFF (`summary.prior_rows == 0`), so the cloud starts EMPTY and its
entire trajectory is this run's own admissions in admission order.

  batch b: (1) the pre-canonical filter runs over ALL of b's checks against a FROZEN cloud;
           (2) surviving checks are canonically rendered and processed in order, and an
               admission joins the cloud mid-batch, visible to later checks in b.

WHAT A REPLAY CANNOT KNOW, AND HOW THAT IS HANDLED. Under `min(fw)` some rows reach a render
the run never performed, so their fate is undecidable without re-rendering — which this
prompt forbids. Rather than guess, the undecidable rows are counted by class and the cloud is
modelled at BOTH ends:

  M0 (determinate)  the cloud grows only where the record settles it: recorded admissions,
                    plus rows whose admission becomes certain under min because their
                    REFRAMED OUTCOME is already in the ledger (an admit-stage q3_dup that is
                    distinct at the smaller radius). Lower bound on cloud growth =>
                    UPPER bound on how many rejects survive.
  M1 (saturated)    additionally every undecidable row is assumed to become an admission, at
                    its CANDIDATE geometry (no reframe exists for it; reframe nudges the
                    centre <=0.25*fw and fw <=1.41x, so this is close but not exact), entering
                    the cloud at the END of its batch and deduped against it under min. Upper
                    bound on cloud growth => LOWER bound on survivors, and the upper bound on
                    originally-admitted rows displaced by the new arrivals.
  M2 (interpolation) the same, at the run's OWN observed admission rate among the checks it
                    did render (67/236). NOT a measurement — it guesses which rows land, and
                    is reported only to say where inside a wide bracket the answer likely
                    sits. M0 and M1 are the numbers; M2 is a reading of them.

THE BRACKET IS WIDE AND THAT IS THE RESULT, not a defect of the tool: min(fw) admits more,
and the extra admissions are themselves cloud members that displace rows max(fw) never
touched. The rule is not monotone at the run level.

THE CONFOUND THE BRACKET DOES NOT COVER: an admission fires the julia hook and the triggered
maneuvers, so a real `min(fw)` run would source a DIFFERENT candidate stream. This replay
holds the candidate stream fixed and measures the rule; it is not a simulation of the run.

  uv run python tools/atlas/precanon_minfw_replay.py                     # self-check + all models
  uv run python tools/atlas/precanon_minfw_replay.py --run-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                                    # noqa: E402
import production_seeder as ps                  # noqa: E402

RUN_DIR = ROOT / "data" / "discovery" / "harvest_v2_proving_20260803"
OUT = Path(paths.scratch("precanon_minfw"))

# THE K THIS REPLAY RUNS AT IS THE RUN'S OWN K, NOT THE LIVE ONE. The run executed under
# 1.5 x max(fw); the self-check below refuses to report anything until it re-fires that
# rule exactly, and the whole comparison is "same K, change the scale". Production moved to
# 0.25 x min(fw) on 2026-08-04 (data/atlas/precanon_calibration/adoption.json) — reading
# the LIVE ps.DEDUP_K here would break the self-check and silently redefine what was
# measured.
REPLAY_K = ps.RETIRED_DEDUP_K

# fates a harvest CHECK can end in (the record's other two fates never reach the filter:
# `below_tau_h` is never rendered and `interior_gt_30` is removed at sourcing).
CHECK_FATES = frozenset({"precanon_dup", "canon_not_q3", "q3_dup", "admitted",
                         "reframe_not_q3", "guarded", "render_failed"})


# =========================================================================== #
# records
# =========================================================================== #
def _jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _key(r: dict) -> tuple:
    """1:1 join key between the two records. `repr` on the floats, not a rounded format —
    both files serialize the same Python float, so the round trip is exact and a formatted
    key could merge two genuinely distinct candidates."""
    return (r["batch"], r["node_id"], repr(r["cx"]), repr(r["cy"]), repr(r["fw"]))


def cand_ident(r: dict):
    """The candidate's dup-identity vector, same rule as `steered_frontier.ident_c`: the
    phoenix (c,p,z_{-1}) 6-vector, the julia seed c 2-vector, or None on a c-plane row.

    Phoenix reads the six `phoenix_*` columns rather than `julia_c_re/im` — those hold only
    `c`, and a 2-vector identity would declare two phoenixes sharing `c` but differing in `p`
    or `z_{-1}` to be the same point (the bug `_q4_record`'s phoenix block exists to fix)."""
    if r["partition"] == "phoenix":
        v = [r.get(k) for k in ("phoenix_c_re", "phoenix_c_im", "phoenix_p_re",
                                "phoenix_p_im", "phoenix_zm1_re", "phoenix_zm1_im")]
        if any(x is None for x in v):
            raise SystemExit(f"phoenix row {_key(r)} has no 6-vector identity in the record")
        return tuple(float(x) for x in v)
    cre = r.get("julia_c_re")
    return None if cre is None else (float(cre), float(r["julia_c_im"]))


def load_population(run_dir: Path) -> tuple[list[dict], dict]:
    """The run's harvest checks in run order, each carrying its recorded fate, its recorded
    displacer id, and (where it has one) its ledger row. Fails loud on any join defect."""
    q4 = [r for r in _jl(run_dir / "q4_candidates.jsonl") if r["fate"] in CHECK_FATES]
    hlog = _jl(run_dir / "harvest_log.jsonl")
    ledger = {r["id"]: r for r in _jl(run_dir / "outcome_ledger.jsonl")}

    hby = {}
    for r in hlog:
        k = _key(r)
        if k in hby:
            raise SystemExit(f"harvest_log key is not unique: {k}")
        hby[k] = r
    if len(q4) != len(hby):
        raise SystemExit(f"{len(q4)} q4 checks vs {len(hby)} harvest_log rows — not 1:1")

    rows = []
    for r in q4:
        k = _key(r)
        h = hby.get(k)
        if h is None:
            raise SystemExit(f"q4 check {k} has no harvest_log row")
        rec_dup = h.get("precanon_dup")
        if (rec_dup is not None) != (r["fate"] == "precanon_dup"):
            raise SystemExit(f"{k}: fate {r['fate']} disagrees with precanon_dup {rec_dup}")
        led = ledger.get(r.get("outcome_id")) if r.get("outcome_id") else None
        if r.get("outcome_id") and led is None:
            raise SystemExit(f"{k}: outcome_id {r['outcome_id']} not in the ledger")
        rows.append(dict(key=k, batch=r["batch"], partition=r["partition"],
                         cx=float(r["cx"]), cy=float(r["cy"]), fw=float(r["fw"]),
                         ident=cand_ident(r), fate=r["fate"], rec_dup=rec_dup,
                         canon_decoded=r.get("canon_decoded"), rank_tier=r.get("rank_tier"),
                         mix_source=r.get("mix_source"), ledger=led,
                         node_id=r["node_id"], depth=r["depth"],
                         cheap_pgood=r.get("cheap_pgood"),
                         julia_c_re=r.get("julia_c_re"), julia_c_im=r.get("julia_c_im"),
                         phoenix={k2: r.get(k2) for k2 in
                                  ("phoenix_c_re", "phoenix_c_im", "phoenix_p_re",
                                   "phoenix_p_im", "phoenix_zm1_re", "phoenix_zm1_im")
                                  if r.get(k2) is not None} or None))
    # rows are already in run order (append order of the store); assert the batch index is
    # monotone rather than trusting it — a resume that re-ordered the file would silently
    # replay a different run.
    bs = [r["batch"] for r in rows]
    if bs != sorted(bs):
        raise SystemExit("q4 store is not in batch order — the replay cannot reproduce it")
    return rows, ledger


# =========================================================================== #
# the replay
# =========================================================================== #
def synth_cloud_row(r: dict, oid: str) -> dict:
    """A cloud member standing in for an admission the run never made. Carries the identity
    fields `ps.row_ident` reads, so the identity clause behaves exactly as it would for a
    real ledger row; the geometry is the CANDIDATE's (see the module note)."""
    row = dict(id=oid, family=r["partition"], decoded_class=3, guard_pass=True,
               outcome_cx=r["cx"], outcome_cy=r["cy"], outcome_fw=r["fw"], synthetic=True)
    if r["partition"] == "phoenix":
        row.update({k: float(v) for k, v in (r["phoenix"] or {}).items()})
    elif r["julia_c_re"] is not None:
        row["julia_c_re"], row["julia_c_im"] = float(r["julia_c_re"]), float(r["julia_c_im"])
    return row


def replay(rows: list[dict], scale: str, *, admit_frac: float, strict: bool) -> dict:
    """Walk the run's batches under one radius scale.

    `strict` (self-check mode) asserts every replayed decision against the record and exits
    loud on the first disagreement.

    `admit_frac` is what FRACTION of the undecidable rows is assumed to become an admission
    and enter the cloud: 0.0 = M0 (none), 1.0 = M1 (all). A fraction in between selects a
    deterministic stride in RUN ORDER (a Bresenham accumulator, no RNG) — position-unbiased,
    but WHICH rows are picked is a guess, so an intermediate model is an interpolation
    between two measured bounds and is labelled as one, never as a measurement.
    """
    clouds: dict[str, list] = defaultdict(list)
    by_batch: dict[int, list] = defaultdict(list)
    for r in rows:
        by_batch[r["batch"]].append(r)

    out = dict(scale=scale, admit_frac=admit_frac,
               precanon_dup=0, survived_precanon=0,
               canon_not_q3=0, q3_dup_prereframe=0, q3_dup_admit=0,
               admitted=0, reframe_not_q3=0, guarded=0,
               undecidable_no_canon=0, undecidable_no_reframe=0,
               synth_admitted=0, synth_deduped=0,
               displaced=[], newly_survived=[], still_rejected=[],
               newly_distinct_at_admit=[], newly_distinct_prereframe=[])
    seq = 0
    acc = [0.0]

    def _take() -> bool:
        """Deterministic stride over the undecidable rows in run order (see `admit_frac`)."""
        if admit_frac <= 0.0:
            return False
        if admit_frac >= 1.0:
            return True
        acc[0] += admit_frac
        if acc[0] >= 1.0:
            acc[0] -= 1.0
            return True
        return False

    for b in sorted(by_batch):
        batch = by_batch[b]
        # ---- stage 1: pre-canonical filter, cloud frozen for the whole batch ----------
        for r in batch:
            distinct, dup_of = ps.is_distinct(r["cx"], r["cy"], r["fw"],
                                              clouds[r["partition"]], REPLAY_K,
                                              c=r["ident"], scale=scale)
            r["rep_dup"] = None if distinct else dup_of
        if strict:
            for r in batch:
                if r["rep_dup"] != r["rec_dup"]:
                    raise SystemExit(f"[self-check] {r['key']} precanon: replay "
                                     f"{r['rep_dup']!r} != recorded {r['rec_dup']!r}")

        survivors = []
        for r in batch:
            if r["rep_dup"] is not None:
                out["precanon_dup"] += 1
                if r["fate"] != "precanon_dup":
                    out["displaced"].append(dict(key=r["key"], partition=r["partition"],
                                                     was=r["fate"], stage="precanon",
                                                     dup_of=r["rep_dup"]))
                continue
            out["survived_precanon"] += 1
            if r["fate"] == "precanon_dup":
                # SNAPSHOT, not the live dict: the three replays share one row list and each
                # overwrites `rep_dup`, so a reference here would report the last mode's
                # verdict on the first mode's population.
                out["newly_survived"].append(dict(r))
            survivors.append(r)
        for r in batch:
            if r["rep_dup"] is not None and r["fate"] == "precanon_dup":
                out["still_rejected"].append(dict(r))

        # ---- stage 2: canonical decode -> pre-reframe dedup -> admit ------------------
        pending_synth = []
        for r in survivors:
            if r["canon_decoded"] is None:
                # never rendered by the run: it was killed by the max(fw) filter. Its
                # canonical decode does not exist and this prompt does not render it.
                out["undecidable_no_canon"] += 1
                if _take():
                    pending_synth.append(r)
                continue
            if r["canon_decoded"] < 3:
                out["canon_not_q3"] += 1
                if strict and r["fate"] != "canon_not_q3":
                    raise SystemExit(f"[self-check] {r['key']} canon_not_q3 vs {r['fate']}")
                continue
            pre_distinct, _ = ps.is_distinct(r["cx"], r["cy"], r["fw"],
                                             clouds[r["partition"]], REPLAY_K,
                                             c=r["ident"], scale=scale)
            if not pre_distinct:
                out["q3_dup_prereframe"] += 1
                if strict and not (r["fate"] == "q3_dup" and r["ledger"] is None):
                    raise SystemExit(f"[self-check] {r['key']} pre-reframe q3_dup vs "
                                     f"{r['fate']} (ledger={bool(r['ledger'])})")
                if r["fate"] != "q3_dup" or r["ledger"] is not None:
                    out["displaced"].append(dict(key=r["key"], partition=r["partition"],
                                                     was=r["fate"], stage="pre_reframe"))
                continue
            led = r["ledger"]
            if led is None:
                # was a pre-reframe q3_dup under max; under min it reaches reframe, which
                # the run never ran for it.
                out["undecidable_no_reframe"] += 1
                out["newly_distinct_prereframe"].append(r)
                if _take():
                    pending_synth.append(r)
                continue
            # admit(): guard + reframed decode are recorded; only the radius rule moves.
            guard_pass = bool(led.get("guard_pass", True))
            is_q3 = guard_pass and (led.get("decoded_class") or 0) >= 3
            distinct, dup_of = (False, None)
            if is_q3:
                distinct, dup_of = ps.is_distinct(led["outcome_cx"], led["outcome_cy"],
                                                  led["outcome_fw"], clouds[r["partition"]],
                                                  REPLAY_K, c=r["ident"], scale=scale)
            if strict and (distinct != bool(led.get("distinct")) or dup_of != led.get("dup_of")):
                raise SystemExit(f"[self-check] {r['key']} admit: replay "
                                 f"({distinct}, {dup_of!r}) != ledger "
                                 f"({led.get('distinct')}, {led.get('dup_of')!r})")
            if is_q3 and distinct:
                out["admitted"] += 1
                clouds[r["partition"]].append(led)
                if r["fate"] != "admitted":
                    out["newly_distinct_at_admit"].append(
                        dict(key=r["key"], partition=r["partition"], was=r["fate"],
                             oid=led["id"]))
            elif is_q3:
                out["q3_dup_admit"] += 1
                if r["fate"] == "admitted":
                    out["displaced"].append(dict(key=r["key"], partition=r["partition"],
                                                     was="admitted", stage="admit",
                                                     dup_of=dup_of))
            elif not guard_pass:
                out["guarded"] += 1
            else:
                out["reframe_not_q3"] += 1
                if strict and r["fate"] != "reframe_not_q3":
                    raise SystemExit(f"[self-check] {r['key']} reframe_not_q3 vs {r['fate']}")

        # M1: the batch's undecidable rows become cloud members AFTER stage 2 — stage 1 for
        # this batch already ran, so they can only bite from the next batch on. Each is still
        # deduped against the growing cloud under the same rule, so a cluster of new arrivals
        # collapses the way the live path would collapse it.
        for r in pending_synth:
            oid = f"synth_{seq:06d}"
            seq += 1
            distinct, _ = ps.is_distinct(r["cx"], r["cy"], r["fw"], clouds[r["partition"]],
                                         REPLAY_K, c=r["ident"], scale=scale)
            if distinct:
                clouds[r["partition"]].append(synth_cloud_row(r, oid))
                out["synth_admitted"] += 1
            else:
                out["synth_deduped"] += 1

    out["cloud_sizes"] = {p: len(v) for p, v in sorted(clouds.items()) if v}
    out["cloud_total"] = sum(len(v) for v in clouds.values())
    return out


# =========================================================================== #
# distributions
# =========================================================================== #
def _q(xs, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    def at(f):
        i = f * (len(s) - 1)
        lo, hi = math.floor(i), math.ceil(i)
        return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (i - lo)
    return {f"p{int(round(q*100))}": at(q) for q in qs}


def pair_geometry(r: dict, ledger: dict) -> dict | None:
    """The arithmetic of the cut that fired on this row, from the record."""
    d = ledger.get(r["rec_dup"])
    if d is None:
        return None
    dist = math.hypot(r["cx"] - float(d["outcome_cx"]), r["cy"] - float(d["outcome_cy"]))
    a, b = r["fw"], float(d["outcome_fw"])
    return dict(dist=dist, fw_cand=a, fw_disp=b, fw_min=min(a, b), fw_max=max(a, b),
                d_over_min=dist / min(a, b), d_over_max=dist / max(a, b),
                fw_ratio=b / a, partition=r["partition"], dup_of=r["rec_dup"],
                key=r["key"], cheap_pgood=r.get("cheap_pgood"))


def distributions(newly, rejected, ledger) -> dict:
    def block(rows, name):
        g = [x for x in (pair_geometry(r, ledger) for r in rows) if x is not None]
        return dict(
            set=name, n=len(rows), n_with_displacer=len(g),
            d_over_min=_q([x["d_over_min"] for x in g]),
            d_over_max=_q([x["d_over_max"] for x in g]),
            fw_ratio=_q([x["fw_ratio"] for x in g]),
            dist_abs=_q([x["dist"] for x in g]),
            fw_min_abs=_q([x["fw_min"] for x in g]),
            displacer_is_wider=sum(1 for x in g if x["fw_ratio"] > 1.0),
            by_partition=dict(Counter(x["partition"] for x in g)))
    return dict(newly_surviving=block(newly, "newly_surviving"),
                still_rejected=block(rejected, "still_rejected"))


def flat_floor_sweep(newly, ledger, floors=(1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                                            1e-2, 0.1, 0.3)) -> list[dict]:
    """How many newly-surviving rows a FLAT absolute-distance floor would take back, i.e.
    radius = max(REPLAY_K*min(fw), floor). A count per floor, not a proposal."""
    g = [x for x in (pair_geometry(r, ledger) for r in newly) if x is not None]
    return [dict(floor=f, recaught=sum(1 for x in g if x["dist"] < f),
                 frac=(sum(1 for x in g if x["dist"] < f) / len(g)) if g else None)
            for f in floors]


# =========================================================================== #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--out", type=Path, default=OUT / "replay.json")
    a = ap.parse_args()

    rows, ledger = load_population(a.run_dir)
    rec = Counter(r["fate"] for r in rows)
    print(f"population: {len(rows)} harvest checks, {len(ledger)} ledger rows, "
          f"{len(set(r['batch'] for r in rows))} batches")
    print(f"recorded fates: {json.dumps(dict(sorted(rec.items())))}")

    # ---- self-check: the replay must re-fire the recorded run exactly ---------------
    base = replay(rows, "max", admit_frac=0.0, strict=True)
    ok = (base["precanon_dup"] == rec["precanon_dup"]
          and base["admitted"] == rec["admitted"]
          and base["canon_not_q3"] == rec["canon_not_q3"]
          and base["q3_dup_prereframe"] + base["q3_dup_admit"] == rec["q3_dup"]
          and base["reframe_not_q3"] == rec["reframe_not_q3"]
          and not base["displaced"])
    print(f"\nSELF-CHECK max(fw): precanon_dup {base['precanon_dup']}/{rec['precanon_dup']}  "
          f"admitted {base['admitted']}/{rec['admitted']}  "
          f"q3_dup {base['q3_dup_prereframe']}+{base['q3_dup_admit']}/{rec['q3_dup']}  "
          f"canon_not_q3 {base['canon_not_q3']}/{rec['canon_not_q3']}  "
          f"cloud {base['cloud_total']}  -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("replay does not reproduce the run — stopping (a replay that cannot "
              "reproduce the original is not evidence)")
        return 1

    # ---- the counterfactual, bracketed --------------------------------------------
    m0 = replay(rows, "min", admit_frac=0.0, strict=False)
    m1 = replay(rows, "min", admit_frac=1.0, strict=False)
    # The interpolation: the run's OWN observed admission rate among rendered checks
    # (67/236). Not a measurement of the counterfactual — a reading of where inside the
    # M0..M1 bracket the answer sits if the newly-surviving population behaves like the
    # population the run did render. It does not have to.
    obs_rate = rec["admitted"] / base["survived_precanon"]
    m2 = replay(rows, "min", admit_frac=obs_rate, strict=False)

    res = dict(
        run=a.run_dir.name, dedup_k=REPLAY_K, recorded_fates=dict(sorted(rec.items())),
        n_checks=len(rows), n_batches=len(set(r["batch"] for r in rows)),
        self_check=dict(reproduced=ok, **{k: base[k] for k in
                        ("precanon_dup", "admitted", "canon_not_q3", "q3_dup_prereframe",
                         "q3_dup_admit", "reframe_not_q3", "cloud_total", "cloud_sizes")}))

    res["observed_admit_rate_of_rendered"] = obs_rate
    for tag, m in (("M0_determinate", m0), ("M1_saturated", m1),
                   ("M2_observed_rate", m2)):
        newly, rej = m["newly_survived"], m["still_rejected"]
        res[tag] = dict(
            {k: m[k] for k in ("precanon_dup", "survived_precanon", "canon_not_q3",
                               "q3_dup_prereframe", "q3_dup_admit", "admitted",
                               "reframe_not_q3", "guarded", "undecidable_no_canon",
                               "undecidable_no_reframe", "synth_admitted", "synth_deduped",
                               "cloud_total", "cloud_sizes")},
            newly_surviving=len(newly),
            newly_surviving_by_partition=dict(Counter(r["partition"] for r in newly)),
            still_rejected=len(rej),
            still_rejected_by_partition=dict(Counter(r["partition"] for r in rej)),
            displaced=m["displaced"],
            displaced_was_admitted=[x for x in m["displaced"] if x["was"] == "admitted"],
            newly_distinct_at_admit=m["newly_distinct_at_admit"],
            newly_distinct_prereframe=len(m["newly_distinct_prereframe"]),
            # A still-rejected row's geometry is stated against the row that displaced it
            # UNDER max(fw) — the like-for-like pairing that splits the 2,531 into survivors
            # and non-survivors. Under M1 some rows are instead killed by a SYNTHETIC cloud
            # member, so the recorded pair is no longer the firing one; that count is
            # reported rather than folded into the quantiles.
            rejected_by_synthetic=sum(1 for r in rej
                                      if str(r.get("rep_dup", "")).startswith("synth_")),
            distributions=distributions(newly, rej, ledger),
            flat_floor=flat_floor_sweep(newly, ledger))
        print(f"\n=== min(fw), {tag} ===")
        print(f"  precanon survivors {m['survived_precanon']} (was {base['survived_precanon']});"
              f" newly surviving {len(newly)} of {rec['precanon_dup']} "
              f"({100*len(newly)/rec['precanon_dup']:.1f}%)")
        print(f"  by partition: {json.dumps(dict(Counter(r['partition'] for r in newly)))}")
        print(f"  undecidable: {m['undecidable_no_canon']} no canonical decode, "
              f"{m['undecidable_no_reframe']} no reframe")
        n_lost = sum(1 for x in m["displaced"] if x["was"] == "admitted")
        print(f"  determinate admissions {m['admitted']} (recorded {rec['admitted']}); "
              f"newly distinct at admit {len(m['newly_distinct_at_admit'])}; "
              f"originally-admitted now colliding {n_lost} "
              f"(all fates displaced by new cloud members: {len(m['displaced'])})")
        print(f"  cloud {m['cloud_total']} (synthetic admitted {m['synth_admitted']}, "
              f"deduped away {m['synth_deduped']})")

    # ---- item 4: downstream volume, as counts ---------------------------------------
    # The ranked-harvest batch and the sitting both draw from `q4_candidates.jsonl` with a
    # TIERED sort (tier 2 = has a canonical decode, tier 1 = cheap score only; never pooled).
    # A newly-surviving row is rendered canonically, so it arrives as a TIER-2 row instead of
    # the tier-1 row it is today. That is the whole downstream effect at the queue level.
    store = _jl(a.run_dir / "q4_candidates.jsonl")
    tiers = Counter(int(r.get("rank_tier") or 0) for r in store)
    res["downstream"] = dict(
        queue_rows_total=len(store),
        queue_by_tier={str(k): v for k, v in sorted(tiers.items())},
        added_canonical_renders=dict(M0=m0["undecidable_no_canon"],
                                     M1=m1["undecidable_no_canon"],
                                     M2=m2["undecidable_no_canon"]),
        tier1_to_tier2=dict(
            M0=dict(Counter(r["partition"] for r in m0["newly_survived"])),
            M1=dict(Counter(r["partition"] for r in m1["newly_survived"])),
            M2=dict(Counter(r["partition"] for r in m2["newly_survived"]))),
        run_rendered_checks=base["survived_precanon"],
        run_canonical_q3_of_rendered=rec["admitted"] + rec["q3_dup"] + rec["reframe_not_q3"],
        run_admitted_of_rendered=rec["admitted"])
    print(f"\n=== downstream volume ===")
    print(f"  queue {len(store)} rows, tiers {dict(sorted(tiers.items()))}; the run rendered "
          f"{base['survived_precanon']} checks")
    print(f"  added canonical confirmation renders: M0 {m0['undecidable_no_canon']}, "
          f"M2 {m2['undecidable_no_canon']}, M1 {m1['undecidable_no_canon']} "
          f"(each a 640x360 ss2 render)")
    print(f"  those rows move tier 1 -> tier 2 in the ranked/sitting draw")

    # the pair rows themselves, for the sheet
    res["newly_surviving_pairs"] = [pair_geometry(r, ledger) for r in m0["newly_survived"]]
    res["still_rejected_pairs"] = [pair_geometry(r, ledger) for r in m0["still_rejected"]]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
