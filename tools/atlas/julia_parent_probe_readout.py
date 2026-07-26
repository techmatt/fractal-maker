#!/usr/bin/env python
r"""julia parent-sourcing probe — readout.

One question (prompts/prompt_julia_parent_sourcing_probe.md): does sourcing julia roots from
the c-diverse near-∂M sampler reduce the rate at which julia candidates die as near-duplicates
(`precanon_dup`) before they are ever rendered?

Primary metric = julia `precanon_dup` rate, PER PARTITION. A harvest_log row is a pre-canonical
dup iff `precanon_dup is not None` (it carries the id of the admitted q3 it collided with; a
rendered check has `precanon_dup is None`). This is the SAME detection tau_h_retained_readout.py
uses (`rendered = precanon_dup is None`) — one function `precanon_admit()` is applied identically
to the committed campaign baseline logs and to the probe log, so any arm difference is a
difference in the runs, not in how two people counted.

Secondary (§2 — a falling dup rate is blind to admissions collapsing):
  * admission rate among RENDERED candidates, per partition and per root supply (mix_source),
  * distinct looks admitted + active-min per distinct look (scheduler tally + price EMA),
straight from each run's summary.json (scheduler ON in both arms).

Guards (§3): per-unit reconciliation (harvest_log ties to summary totals) and the julia
birth-stamp assertion (every julia ledger row born `julia_schema=campaign`).

Run: uv run python tools/atlas/julia_parent_probe_readout.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows cp1252 console chokes on ✓/→
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
DISC = ROOT / "data/discovery"

JPARTS = ["julia:mandelbrot", "julia:multibrot3", "julia:multibrot4", "julia:multibrot5"]

# campaign 2 breadth flipped julia hook spacing 0.2 -> 0.1 at the batch-1211 resume; the current
# (probe-matching) era is seg-B. Mirrors tau_h_retained_readout.SEG_BOUNDARY exactly.
SEG_BOUNDARY = {"campaign2/breadth": 1211}


def load(run_or_path):
    p = Path(run_or_path)
    if not p.is_absolute():
        p = DISC / run_or_path / "harvest_log.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_summary(run_or_path):
    p = Path(run_or_path)
    p = p if p.is_absolute() else DISC / run_or_path
    return json.loads((p / "summary.json").read_text(encoding="utf-8"))


def seg_rows(rows, run):
    b = SEG_BOUNDARY.get(run)
    if b is None:
        return [("all", rows)]
    return [(f"seg-A(<{b})", [r for r in rows if r["batch"] < b]),
            (f"seg-B(>={b})", [r for r in rows if r["batch"] >= b])]


def precanon_admit(rows):
    """The one metric function, applied identically to every arm. A precanon dup is
    `precanon_dup is not None`; a rendered check is its complement; an admission is
    `admitted == True` (a distinct-q3 reframe survivor)."""
    n = len(rows)
    n_dup = sum(1 for r in rows if r.get("precanon_dup") is not None)
    n_render = n - n_dup
    n_admit = sum(1 for r in rows if r.get("admitted"))
    return dict(
        n_checks=n, n_precanon_dup=n_dup,
        precanon_rate=(n_dup / n if n else float("nan")),
        n_render=n_render, n_admit=n_admit,
        admit_rate_rendered=(n_admit / n_render if n_render else float("nan")),
    )


def by_source_group(mix):
    if mix == "sampler":
        return "sampler"
    if isinstance(mix, str) and mix.startswith("julia_hook"):
        return "hook"
    return str(mix)


def reconcile(rows, summary, unit):
    """§3 per-unit reconciliation: found (harvest_log rows) == written (rendered incl. admitted)
    + dropped (precanon_dup). And the harvest-log admitted/precanon counts tie to summary totals."""
    st = precanon_admit(rows)
    t = summary.get("totals", {})
    problems = []
    if st["n_checks"] != st["n_render"] + st["n_precanon_dup"]:
        problems.append("found != render + precanon_dup (arithmetic)")
    if t.get("harvest_checks") is not None and t["harvest_checks"] != st["n_checks"]:
        problems.append(f"summary.harvest_checks={t['harvest_checks']} != log rows {st['n_checks']}")
    if t.get("precanon_dup") is not None and t["precanon_dup"] != st["n_precanon_dup"]:
        problems.append(f"summary.precanon_dup={t['precanon_dup']} != log dups {st['n_precanon_dup']}")
    # ledger admissions (distinct q3) tie to harvest_log admitted rows.
    led = ROOT_led(unit)
    if led is not None:
        adm_led = sum(1 for r in led if r.get("distinct") and r.get("decoded_class") == 3)
        if adm_led != st["n_admit"]:
            problems.append(f"ledger admits {adm_led} != harvest_log admitted {st['n_admit']}")
    if problems:
        raise SystemExit(f"[reconcile] {unit}: " + "; ".join(problems))
    return st


def ROOT_led(unit):
    p = (Path(unit) if Path(unit).is_absolute() else DISC / unit) / "outcome_ledger.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def assert_birth_stamp(unit):
    """§1: a julia row that lands untagged must fail loudly, not resolve by guess."""
    led = ROOT_led(unit)
    if led is None:
        return 0
    n = 0
    for r in led:
        fam = r.get("family", "")
        if fam.startswith("julia:"):
            tag = r.get("julia_schema")
            if tag != "campaign":
                raise SystemExit(f"[birth-stamp] {unit}: julia row {r.get('id')} tag={tag!r} "
                                 f"(expected 'campaign') — untagged/mis-tagged, failing loud.")
            n += 1
    return n


def fmt(st):
    return (f"precanon_dup {st['n_precanon_dup']:>5}/{st['n_checks']:<5} = "
            f"{100*st['precanon_rate']:5.1f}%   rendered {st['n_render']:>4}  "
            f"admit(among rendered) {st['n_admit']:>3}/{st['n_render']:<4} = "
            f"{100*st['admit_rate_rendered']:4.1f}%" if st['n_render'] else
            f"precanon_dup {st['n_precanon_dup']:>5}/{st['n_checks']:<5} = "
            f"{100*st['precanon_rate']:5.1f}%   rendered 0 (no admits possible)")


def main():
    out = {"baseline": {}, "probe": {}}

    print("=" * 96)
    print("BASELINE — committed campaign harvest logs (hook-sourced julia; SAME metric fn)")
    print("=" * 96)
    for run in ["campaign2/breadth", "campaign2/dive"]:
        rows = load(run)
        summ = load_summary(run)
        reconcile(rows, summ, run)
        nstamp = assert_birth_stamp(run)
        print(f"\n{run}  (reconciled ✓, {nstamp} julia rows birth-stamp ✓)")
        sc = summ.get("scheduler", {})
        for seg, srows in seg_rows(rows, run):
            for jp in JPARTS:
                st = precanon_admit([r for r in srows if r["partition"] == jp])
                if st["n_checks"] == 0:
                    continue
                looks = sc.get("looks", {}).get(jp)
                price = sc.get("prices", {}).get(jp)
                extra = (f"   looks={looks} price={price} active-min/look" if looks is not None else "")
                print(f"  [{seg:>11}] {jp:18} {fmt(st)}{extra if seg.startswith('seg-B') or seg=='all' else ''}")
                out["baseline"].setdefault(run, {}).setdefault(seg, {})[jp] = st

    print("\n" + "=" * 96)
    print("PROBE — sampler-sourced julia roots (data/discovery/julia_parent_probe/breadth)")
    print("=" * 96)
    unit = "julia_parent_probe/breadth"
    rows = load(unit)
    summ = load_summary(unit)
    reconcile(rows, summ, unit)
    nstamp = assert_birth_stamp(unit)
    sc = summ.get("scheduler", {})
    print(f"\nreconciled ✓, {nstamp} julia rows birth-stamp ✓, "
          f"active_min={summ.get('active_min')}, batches={summ.get('batches')}")

    for jp in JPARTS:
        jrows = [r for r in rows if r["partition"] == jp]
        if not jrows:
            continue
        st = precanon_admit(jrows)
        looks = sc.get("looks", {}).get(jp)
        price = sc.get("prices", {}).get(jp)
        print(f"\n  {jp}  (ALL supplies)  {fmt(st)}")
        if looks is not None:
            print(f"    distinct looks={looks}   price={price} active-min/look")
        out["probe"].setdefault(jp, {})["all"] = st
        # split by root supply
        srcs = Counter(by_source_group(r.get("mix_source")) for r in jrows)
        for grp in sorted(srcs):
            gst = precanon_admit([r for r in jrows if by_source_group(r.get("mix_source")) == grp])
            print(f"      · {grp:8} {fmt(gst)}")
            out["probe"][jp][grp] = gst

    outp = ROOT / "scratch" / "julia_parent_probe" / "readout.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
