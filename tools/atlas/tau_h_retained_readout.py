#!/usr/bin/env python
"""tau_h retroactive read — what the RETAINED store (ledgers) can still say.

The full (partition, cheap_pgood, canonical_fate) join per harvest check lived in
each run's harvest_log.jsonl, which is gitignored (.gitignore:172) and was not kept
through the campaign1->campaign2->fractal-maker migration. Only outcome_ledger.jsonl
survives = ADMISSIONS ONLY (distinct q3), each carrying cheap_pgood.

So the COST axis of the tau_h curve (renders saved = f(tau_h), needs reject cheap
scores) is unrecoverable. The BENEFIT axis (admissions retained = f(tau_h), for
tau_h >= current) IS recoverable from the retained admissions' cheap_pgood, and is
reported here. It is a conservative LOWER bound: raising tau_h only shrinks the
greedy dedup cloud, which can only promote later q3_dups to distinct -> true
retention >= this count.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEDGERS = {
    "campaign2/breadth": ROOT / "data/discovery/campaign2/breadth",
    "campaign2/dive":    ROOT / "data/discovery/campaign2/dive",
    "campaign1/breadth": ROOT / "data/discovery/campaign1/breadth",
    "campaign1/dive":    ROOT / "data/discovery/campaign1/dive",
}
# tau_h is a fixed per-partition constant (derive_tau_h, keep=0.90 of true-q3 by
# cheap p_good in the offline fidelity study) — identical across runs. Read it from
# a summary that carries it rather than re-deriving.
TAU_H = json.load(open(LEDGERS["campaign2/breadth"] / "summary.json"))["tau_h"]

PARTS = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5",
         "julia:mandelbrot", "julia:multibrot3", "julia:multibrot4", "julia:multibrot5"]


def load(run_dir):
    p = run_dir / "outcome_ledger.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# pool admitted (distinct q3) cheap_pgood per partition across all retained ledgers
by_part = {p: [] for p in PARTS}
per_run_admits = {}
for name, d in LEDGERS.items():
    rows = load(d)
    adm = [r for r in rows if r.get("distinct") and r.get("decoded_class") == 3]
    per_run_admits[name] = len(adm)
    for r in adm:
        cp = r.get("cheap_pgood")
        fam = r.get("family")
        if cp is not None and fam in by_part:
            by_part[fam].append(float(cp))

print("=" * 96)
print("RETAINED admitted-check cheap_pgood distribution, per partition (all 4 ledgers pooled)")
print("current tau_h = fixed derive_tau_h constant; headroom = min_admit_cheap - tau_h")
print("=" * 96)
print(f"{'partition':20s} {'n_adm':>5s} {'tau_h':>7s} {'min':>7s} {'p5':>7s} {'p10':>7s} "
      f"{'p25':>7s} {'med':>7s} {'headroom':>8s} {'<tau_h?':>7s}")
rows_out = []
for p in PARTS:
    v = np.array(by_part[p])
    t = TAU_H[p]
    if len(v) == 0:
        print(f"{p:20s} {'0':>5s}  (no retained admissions)")
        continue
    mn, p5, p10, p25, med = (float(np.min(v)), float(np.percentile(v, 5)),
                             float(np.percentile(v, 10)), float(np.percentile(v, 25)),
                             float(np.median(v)))
    n_below = int((v < t).sum())
    hr = mn - t
    print(f"{p:20s} {len(v):5d} {t:7.4f} {mn:7.4f} {p5:7.4f} {p10:7.4f} {p25:7.4f} "
          f"{med:7.4f} {hr:+8.4f} {n_below:7d}")
    rows_out.append(dict(partition=p, n_admitted=len(v), tau_h=t, min_cheap=mn,
                         p5=p5, p10=p10, p25=p25, median=med,
                         zero_loss_headroom=hr, n_admits_below_tau_h=n_below))

# admissions-retained curve: for a swept tau_h', count retained admissions (cheap>=tau_h').
# reported as fraction of actual, per partition. Only tau_h' >= current is meaningful
# (below current we have no reject data and admissions there don't exist to lose).
print()
print("=" * 96)
print("Admissions-retained vs candidate tau_h' (fraction of actual admissions; LOWER bound)")
print("=" * 96)
sweep = [round(x, 2) for x in np.arange(0.20, 0.96, 0.05)]
hdr = "partition".ljust(20) + "cur_t " + " ".join(f"{s:>5.2f}" for s in sweep)
print(hdr)
curve_out = {}
for p in PARTS:
    v = np.array(by_part[p])
    if len(v) == 0:
        continue
    t = TAU_H[p]
    fracs = [float((v >= s).mean()) for s in sweep]
    curve_out[p] = dict(tau_h=t, sweep=sweep, retained_frac=fracs, n=len(v))
    print(p.ljust(20) + f"{t:5.2f} " + " ".join(f"{f:5.2f}" for f in fracs))

out = dict(
    note=("harvest_log.jsonl (per-check cheap_pgood x canonical fate) is gitignored and "
          "was not retained; reject cheap-scores are unrecoverable. This file reports only "
          "what the admissions ledger retains: the admitted-check cheap_pgood distribution "
          "and the admissions-retained (benefit) axis. The renders-saved (cost) axis is "
          "unrecoverable retroactively."),
    tau_h=TAU_H, per_run_admits=per_run_admits,
    per_partition=rows_out, retained_curve=curve_out,
)
outp = ROOT / "out/tau_h/retained_readout.json"
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\nwrote {outp}")
print(f"per-run admits: {per_run_admits}")
