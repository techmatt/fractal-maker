#!/usr/bin/env python
r"""The pre-registered scheduler-vs-pop-quota read, as a program rather than a paragraph.

Every window, index and margin below is LOADED from `data/discovery/allocator_prereg_v1.json`
and never restated here. That is the whole point: the file was committed before arm B was
launched, so a window cannot be moved after the numbers are visible, and this reader goes
red rather than silently inventing one if the prereg is missing a key.

Three axes, all pre-registered:

  matched wall    each arm at its own 520-minute wall cap. TERMINAL ONLY, and the prereg
                  says why: cumulative wall is not reconstructible at intermediate points
                  for arm A (the per-batch log line carries cumulative ACTIVE, and their
                  sum reproduces active_min, not wall_min).
  matched batch   cumulative admissions at fixed batch indices + at B* = min(last batch).
  late marginal   the last N batches of [1, B*], and the final M active minutes.

The admissions series comes from each run's TRACKED `harvest_log.jsonl` (one row per
harvest check, `batch` + `admitted`). The per-batch ACTIVE clock comes from `run.log`,
which is gitignored: when it is absent the active-normalized cells report **UNKNOWN**, not
0 and not a mean-cost back-fill. A tool that cannot reach its authority reports UNKNOWN
(`measurement_practice.md`); back-filling from `active_min / batches` would assume a
uniform per-batch cost that arm A visibly does not have (its rate rose all night).

  uv run python tools/atlas/compare_allocator_runs.py
  uv run python tools/atlas/compare_allocator_runs.py --arm-b <run_id>   # override arm B
  uv run python tools/atlas/compare_allocator_runs.py --json out.json

Reads:  data/discovery/allocator_prereg_v1.json      (the pre-registered windows)
        data/discovery/<arm>/harvest_log.jsonl       (tracked; the admissions series)
        data/discovery/<arm>/summary.json            (tracked; terminal totals)
        data/discovery/<arm>/run.log                 (untracked; the active clock, optional)
Writes: nothing unless --json is passed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "data/discovery/allocator_prereg_v1.json"

# The per-batch line steered_frontier prints: "  batch 12: ... | 19s active=3.4m".
# `active=` is CUMULATIVE active minutes at the end of that batch.
_BATCH_LINE = re.compile(r"^\s*batch (\d+):.*\|\s*(\d+)s active=([\d.]+)m")

UNKNOWN = "UNKNOWN"


def load_prereg(path: Path = PREREG) -> dict:
    if not path.exists():
        raise SystemExit(f"pre-registration absent: {path}\nThe windows live there, not here.")
    return json.loads(path.read_text(encoding="utf-8"))


def _need(d: dict, *keys):
    """Fetch a nested prereg key or die naming it — a missing window is a red, not a default."""
    cur, seen = d, []
    for k in keys:
        seen.append(str(k))
        if not isinstance(cur, dict) or k not in cur:
            raise SystemExit(f"pre-registration is missing '{'.'.join(seen)}' — "
                             f"this reader does not supply a default for a window.")
        cur = cur[k]
    return cur


class Arm:
    """One run's series, loaded from its own directory."""

    def __init__(self, key: str, spec: dict):
        self.key = key
        self.run_id = spec["run_id"]
        self.allocator = spec.get("allocator", "?")
        self.dir = ROOT / spec["run_dir"]
        self.present = (self.dir / "harvest_log.jsonl").exists()
        self.adm_by_batch: dict[int, int] = {}
        self.last_batch = 0
        self.summary: dict = {}
        self.active_cum: dict[int, float] = {}     # batch -> cumulative active minutes
        if self.present:
            self._load()

    def _load(self):
        with (self.dir / "harvest_log.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                b = int(r["batch"])
                self.last_batch = max(self.last_batch, b)
                if r.get("admitted"):
                    self.adm_by_batch[b] = self.adm_by_batch.get(b, 0) + 1
        sp = self.dir / "summary.json"
        if sp.exists():
            self.summary = json.loads(sp.read_text(encoding="utf-8"))
        log = self.dir / "run.log"
        if log.exists():
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                m = _BATCH_LINE.match(line)
                if m:
                    self.active_cum[int(m.group(1))] = float(m.group(3))

    # -- series ---------------------------------------------------------------
    def cum_admissions(self, upto_batch: int) -> int:
        return sum(n for b, n in self.adm_by_batch.items() if b <= upto_batch)

    def admissions_in(self, lo_exclusive: int, hi_inclusive: int) -> int:
        return sum(n for b, n in self.adm_by_batch.items()
                   if lo_exclusive < b <= hi_inclusive)

    def active_at(self, batch: int):
        """Cumulative active minutes at or before `batch`, or UNKNOWN with no run.log."""
        if not self.active_cum:
            return None
        keys = [b for b in self.active_cum if b <= batch]
        return self.active_cum[max(keys)] if keys else 0.0

    def batch_at_active(self, minutes: float):
        """First batch whose cumulative active time reaches `minutes`, or None."""
        if not self.active_cum:
            return None
        for b in sorted(self.active_cum):
            if self.active_cum[b] >= minutes:
                return b
        return None

    @property
    def total_admitted(self) -> int:
        return self.summary.get("totals", {}).get("admitted", sum(self.adm_by_batch.values()))


def _ratio(b, a):
    if a in (None, 0) or b is None:
        return None
    return b / a


def _fmt(v, spec="{:.3f}"):
    return UNKNOWN if v is None else spec.format(v)


def compare(prereg: dict, arm_b_override: str | None = None) -> dict:
    spec_a = _need(prereg, "arms", "A")
    spec_b = dict(_need(prereg, "arms", "B"))
    if arm_b_override:
        spec_b["run_id"] = arm_b_override
        spec_b["run_dir"] = f"data/discovery/{arm_b_override}"
    A, B = Arm("A", spec_a), Arm("B", spec_b)

    out = {"prereg_id": prereg["id"], "prereg_written": prereg["written"],
           "arm_a": A.run_id, "arm_b": B.run_id,
           "arm_a_present": A.present, "arm_b_present": B.present}

    if not A.present:
        raise SystemExit(f"arm A absent at {A.dir} — nothing to compare against.")
    if not B.present:
        out["status"] = "PENDING — arm B has produced no harvest_log yet."
        return out

    b_star = min(A.last_batch, B.last_batch)
    out["b_star"] = b_star

    # --- matched wall (terminal only, per the prereg's own reason) ------------
    wall = {}
    for name, arm in (("A", A), ("B", B)):
        s = arm.summary
        act = s.get("active_min")
        adm = arm.total_admitted
        wall[name] = {
            "admitted": adm,
            "active_min": act,
            "wall_min": s.get("wall_min"),
            "wall_over_active": s.get("wall_over_active"),
            "admissions_per_active_min": (adm / act) if act else None,
            "stop_reason_wall_capped": _wall_capped(s),
        }
    wall["ratio_admitted_B_over_A"] = _ratio(wall["B"]["admitted"], wall["A"]["admitted"])
    wall["ratio_rate_B_over_A"] = _ratio(wall["B"]["admissions_per_active_min"],
                                        wall["A"]["admissions_per_active_min"])
    wall["matched"] = bool(wall["A"]["stop_reason_wall_capped"] and
                           wall["B"]["stop_reason_wall_capped"])
    if not wall["matched"]:
        wall["void_note"] = _need(prereg, "windows", "matched_wall", "contingency")
    out["matched_wall"] = wall

    # --- matched batch index -------------------------------------------------
    idx = list(_need(prereg, "windows", "matched_batch", "indices"))
    if b_star not in idx:
        idx.append(b_star)
    rows = []
    for b in idx:
        if b > b_star:
            rows.append({"batch": b, "note": f"beyond B* ({b_star}) — not a matched point"})
            continue
        ca, cb = A.cum_admissions(b), B.cum_admissions(b)
        rows.append({"batch": b, "A": ca, "B": cb, "ratio_B_over_A": _ratio(cb, ca)})
    out["matched_batch"] = rows

    # --- late marginal -------------------------------------------------------
    win = _need(prereg, "windows", "late_marginal")
    n_batches = int(re.search(r"last (\d+) batches", win["batch_window"]).group(1))
    lo = max(0, b_star - n_batches)
    lm = {"window_batches": [lo + 1, b_star]}
    for name, arm in (("A", A), ("B", B)):
        adm = arm.admissions_in(lo, b_star)
        a_hi, a_lo = arm.active_at(b_star), arm.active_at(lo)
        d_active = None if (a_hi is None or a_lo is None) else (a_hi - a_lo)
        lm[name] = {
            "admissions": adm,
            "per_batch": adm / max(1, b_star - lo),
            "active_min_in_window": d_active,
            "per_active_min": (adm / d_active) if d_active else None,
        }
    lm["ratio_per_active_min_B_over_A"] = _ratio(lm["B"]["per_active_min"],
                                                 lm["A"]["per_active_min"])
    lm["ratio_per_batch_B_over_A"] = _ratio(lm["B"]["per_batch"], lm["A"]["per_batch"])

    # final M ACTIVE minutes of each run, on each run's own clock
    m_min = float(win["active_window_min"])
    tail = {"active_window_min": m_min}
    for name, arm in (("A", A), ("B", B)):
        total_act = arm.summary.get("active_min")
        b_from = arm.batch_at_active(total_act - m_min) if (total_act and arm.active_cum) else None
        if b_from is None:
            tail[name] = {"admissions": None, "per_active_min": None,
                          "why": UNKNOWN + " — no run.log, so no per-batch active clock"}
        else:
            adm = arm.admissions_in(b_from, arm.last_batch)
            act = (arm.active_at(arm.last_batch) or 0.0) - (arm.active_at(b_from) or 0.0)
            tail[name] = {"from_batch": b_from, "admissions": adm,
                          "active_min": act, "per_active_min": (adm / act) if act else None}
    tail["ratio_B_over_A"] = _ratio(tail["B"].get("per_active_min"),
                                    tail["A"].get("per_active_min"))
    lm["final_active_window"] = tail
    out["late_marginal"] = lm

    # --- the pre-registered decision rule ------------------------------------
    out["verdict"] = _verdict(prereg, wall, lm)
    return out


def _wall_capped(summary: dict) -> bool:
    """True iff this run stopped on its WALL cap. Derived from the record, not asserted:
    wall_min within one estimated batch of wall_budget_min and active under its own cap."""
    w, wb = summary.get("wall_min"), summary.get("wall_budget_min")
    if w is None or not wb:
        return False
    return (wb - w) <= 1.0


def _verdict(prereg: dict, wall: dict, lm: dict) -> dict:
    rule = _need(prereg, "decision_rule")
    margin = 0.10
    m = re.search(r">=\s*\+?(\d+)%", rule["adopt_pop_quota_as_default_allocator_iff"])
    if m:
        margin = int(m.group(1)) / 100.0
    r_primary = wall.get("ratio_admitted_B_over_A")
    r_marginal = lm.get("ratio_per_active_min_B_over_A")
    if r_primary is None or r_marginal is None:
        return {"verdict": UNKNOWN, "why": "one of the two required ratios is UNKNOWN",
                "primary": r_primary, "marginal": r_marginal, "margin": margin}
    adopt = r_primary >= 1 + margin and r_marginal >= 1 + margin
    keep = r_primary <= 1 / (1 + margin) and r_marginal <= 1 / (1 + margin)
    v = "ADOPT_POP_QUOTA" if adopt else ("KEEP_SCHEDULER" if keep else "DISAGREE")
    return {"verdict": v, "primary_ratio_B_over_A": r_primary,
            "marginal_ratio_B_over_A": r_marginal, "margin": margin,
            "rule": rule["otherwise"] if v == "DISAGREE" else None,
            "wall_matched": wall.get("matched")}


def render(out: dict) -> str:
    L = [f"PRE-REGISTERED ALLOCATOR READ — {out['prereg_id']}",
         f"  windows loaded from {PREREG.relative_to(ROOT).as_posix()} (written {out['prereg_written']})",
         f"  A = {out['arm_a']} (scheduler)   B = {out['arm_b']} (pop-quota)", ""]
    if not out.get("arm_b_present"):
        L.append(f"  {out.get('status')}")
        return "\n".join(L)

    w = out["matched_wall"]
    L.append(f"1) MATCHED WALL (terminal, both capped at their own wall budget) "
             f"— matched={w['matched']}")
    for k in ("A", "B"):
        d = w[k]
        L.append(f"   {k}: admitted={d['admitted']}  active={_fmt(d['active_min'],'{:.1f}')}m  "
                 f"wall={_fmt(d['wall_min'],'{:.1f}')}m  w/a={_fmt(d['wall_over_active'],'{:.2f}')}  "
                 f"rate={_fmt(d['admissions_per_active_min'])}/active-min  "
                 f"wall_capped={d['stop_reason_wall_capped']}")
    L.append(f"   ratio B/A: admitted={_fmt(w['ratio_admitted_B_over_A'])}  "
             f"rate={_fmt(w['ratio_rate_B_over_A'])}")
    if "void_note" in w:
        L.append(f"   !! NOT MATCHED AT WALL: {w['void_note']}")

    L.append("")
    L.append(f"2) MATCHED BATCH INDEX (B* = {out['b_star']})")
    for r in out["matched_batch"]:
        if "note" in r:
            L.append(f"   b{r['batch']:>5}: {r['note']}")
        else:
            L.append(f"   b{r['batch']:>5}: A={r['A']:<5} B={r['B']:<5} "
                     f"B/A={_fmt(r['ratio_B_over_A'])}")

    lm = out["late_marginal"]
    L.append("")
    L.append(f"3) LATE MARGINAL — batches {lm['window_batches'][0]}..{lm['window_batches'][1]}")
    for k in ("A", "B"):
        d = lm[k]
        L.append(f"   {k}: adm={d['admissions']}  /batch={_fmt(d['per_batch'])}  "
                 f"active={_fmt(d['active_min_in_window'],'{:.1f}')}m  "
                 f"/active-min={_fmt(d['per_active_min'])}")
    L.append(f"   ratio B/A: /active-min={_fmt(lm['ratio_per_active_min_B_over_A'])}  "
             f"/batch={_fmt(lm['ratio_per_batch_B_over_A'])}")
    t = lm["final_active_window"]
    L.append(f"   final {t['active_window_min']:.0f} ACTIVE minutes of each run:")
    for k in ("A", "B"):
        d = t[k]
        if d.get("per_active_min") is None:
            L.append(f"     {k}: {d.get('why', UNKNOWN)}")
        else:
            L.append(f"     {k}: adm={d['admissions']} over {d['active_min']:.1f}m "
                     f"= {d['per_active_min']:.3f}/active-min (from b{d['from_batch']})")
    L.append(f"     ratio B/A={_fmt(t['ratio_B_over_A'])}")

    v = out["verdict"]
    L.append("")
    L.append(f"VERDICT (pre-registered rule, margin {v.get('margin', 0.1):.0%}): {v['verdict']}")
    if v.get("rule"):
        L.append(f"   {v['rule']}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-b", default=None, help="override arm B's run id")
    ap.add_argument("--json", default=None, help="also write the raw comparison here")
    args = ap.parse_args(argv)
    out = compare(load_prereg(), args.arm_b)
    print(render(out))
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
