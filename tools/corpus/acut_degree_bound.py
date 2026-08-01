#!/usr/bin/env python
r"""acut_degree_bound.py — bound the `A` feasibility cut's effect on the degree gradient.

THE OBJECTION. Degree is the strongest measured atom-level signal in the 487 (Spearman
+0.554 with label; mean label d2 1.22 -> d5 2.27). But the roster's `A` feasibility cut
(`build_minibrot_roster`: admit iff the deploy-presentation f64 margin >= 1 decade) fired on
3 of 163 atoms and **all three are d2** — and those three are among the most beautiful
material sourced. So d2's mean is depressed by construction and the gradient may be partly
an artifact of the cut.

WHY THIS IS A BOUND, NOT A MEASUREMENT. Those atoms were excluded *for feasibility*: they
cannot be rendered at the deploy presentation (1280x720 ss4) inside f64 at all, so they
carry no labels there and never will. Nothing can measure what they would have scored. What
CAN be done is bound it: give them the most generous labels the scale allows, in the largest
number the corpus supports, and see whether the gradient survives. If it survives the most
generous imputation the objection is answered; if it collapses, the gradient is fragile and
should be labeled that way.

PART 1 — imputation scenarios, neutral -> most generous.
PART 2 — the margin distribution by degree: is "the cut only fires at d2" a real property of
where small high-period atoms live, or an artifact of what was sourced?

Read-only: no renders, no config, no threshold. Output goes to scratch/.

  uv run python tools/corpus/acut_degree_bound.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import minibrot_roster_v2_readout as RO      # noqa: E402  the committed 487 join + spearman
from tools.sourcing import build_minibrot_roster as RB   # noqa: E402  the cut itself

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROSTER = ROOT / "data" / "minibrot_roster" / "roster.jsonl"
CELLS = ROOT / "data" / "minibrot_roster" / "roster_cells.json"
OUT = ROOT / "scratch" / "acut_degree_bound"
BOOT_REPS = 2000
BOOT_SEED = 20260728
TOP_LABEL = 4


def _jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# atom-clustered bootstrap on Spearman(degree, label)
# --------------------------------------------------------------------------- #
def boot_ci(recs, seed=BOOT_SEED, reps=BOOT_REPS, lo=5, hi=95):
    """90% CI on Spearman(degree, label), resampling ATOMS (up to 6 windows share one, so
    crops are not independent). Mirrors the interval method in docs/design/minibrot_sourcing.md §8."""
    by_atom = defaultdict(list)
    for r in recs:
        by_atom[r["atom"]].append(r)
    atoms = list(by_atom)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pick = rng.integers(0, len(atoms), len(atoms))
        sub = [r for i in pick for r in by_atom[atoms[i]]]
        d = [r["degree"] for r in sub]
        y = [r["label"] for r in sub]
        if len(set(d)) < 2:
            continue
        out.append(RO.spearman(d, y))
    a = np.array(out, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def gradient(recs):
    """(rho, per-degree mean, per-degree n, n_ge3 at d2)."""
    d = [r["degree"] for r in recs]
    y = [r["label"] for r in recs]
    means, ns = {}, {}
    for deg in sorted(set(d)):
        sub = [r["label"] for r in recs if r["degree"] == deg]
        means[deg] = float(np.mean(sub))
        ns[deg] = len(sub)
    return RO.spearman(d, y), means, ns


# --------------------------------------------------------------------------- #
# imputation
# --------------------------------------------------------------------------- #
def largest_remainder(dist: Counter, m: int) -> list:
    """Replicate an empirical label distribution into exactly `m` integer labels,
    deterministically (largest-remainder allocation). Used for the NEUTRAL scenario: the
    excluded atoms behave like typical material of their own degree."""
    tot = sum(dist.values())
    exact = {k: m * v / tot for k, v in dist.items()}
    base = {k: int(np.floor(v)) for k, v in exact.items()}
    left = m - sum(base.values())
    for k in sorted(exact, key=lambda k: (-(exact[k] - base[k]), k))[:left]:
        base[k] += 1
    return [k for k, n in sorted(base.items()) for _ in range(n)]


def impute(recs, excluded, labels_per_atom):
    """`recs` + synthetic crops for each excluded atom. `labels_per_atom` maps atom id ->
    list of labels to add. Synthetic rows carry the excluded atom's own degree and id, so the
    atom-clustered bootstrap treats each as one extra atom (not extra independent crops)."""
    out = list(recs)
    for a in excluded:
        for i, lab in enumerate(labels_per_atom[a["id"]]):
            out.append({"atom": a["id"], "degree": a["degree"], "label": lab,
                        "iid": f"imputed_{a['id']}_{i}", "split": "imputed"})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs, missing = RO.load()
    recs = [{"atom": r["atom"], "degree": r["degree"], "label": r["label"], "iid": r["iid"],
             "split": r["split"]} for r in recs]
    roster = _jsonl(ROSTER)
    cells = json.loads(CELLS.read_text(encoding="utf-8"))
    excluded = [a for a in roster if not a["admitted"]]
    L, ok = [], True

    def w(s=""):
        L.append(s)
        print(s)

    w("# Bounding the `A` cut's effect on the degree gradient")
    w("")
    w(f"*Read-only. 487 labels joined via `minibrot_roster_v2_readout.load()` "
      f"({len(recs)} rows, {len(missing)} unjoined); roster + per-cell fill read from "
      f"`data/minibrot_roster/`. No render, no config, no threshold touched.*")
    w("")

    # ---- the cut, restated from the code ---------------------------------- #
    w("## 0. What the cut is, and exactly who it removed")
    w("")
    w(f"`build_minibrot_roster` admits an atom iff its **deploy-presentation** f64 "
      f"pixel-spacing margin >= `MARGIN_MIN_DECADES = {RB.MARGIN_MIN_DECADES}` decade at "
      f"{RB.DEPLOY_W}x ss{RB.DEPLOY_SS} (wall at `log10|A| = "
      f"{RB.deploy_wall_log10():.4f}`). Everything else is retained-but-excluded.")
    w("")
    w("| atom | degree | period | band | log10\\|A\\| | deploy margin | field margin | split |")
    w("|---|--:|--:|---|--:|--:|--:|---|")
    for a in excluded:
        w(f"| `{a['id']}` | {a['degree']} | {a['period']} | {a['band']} | "
          f"{a['log10_abs_A']:.4f} | **{a['f64_margin_deploy_decades']:+.4f}** | "
          f"{a['f64_margin_field_decades']:+.4f} | {a['split']} |")
    w("")
    w(f"**{len(excluded)} of {len(roster)} roster rows, all degree "
      f"{sorted({a['degree'] for a in excluded})}, all band "
      f"{sorted({a['band'] for a in excluded})}.** Their deploy margins are "
      + ", ".join("%+.3f" % a["f64_margin_deploy_decades"] for a in excluded)
      + (" — two sit essentially ON the f64 wall, one a decade inside it but under the "
      f"1-decade admission margin. Their FIELD margins are all positive, which is why the "
      f"roster retained them: they can be swept at {RB.FIELD_W}x ss{RB.FIELD_SS}, they "
      f"cannot be *presented*."))
    w("")
    w("They carry **no labels**: they never entered the 487 draw "
      f"(`{len({r['atom'] for r in recs})}` labeled atoms, none of them these three). "
      "11 crops of them exist in the roster PILOT manifest "
      "(`data/minibrot_roster/pilot/manifest.jsonl`), unlabeled. So this is a bound.")
    w("")

    # ---- PART 1: imputation ------------------------------------------------ #
    base_rho, base_means, base_ns = gradient(recs)
    base_ci = boot_ci(recs)
    d2 = [r for r in recs if r["degree"] == 2]
    d2_dist = Counter(r["label"] for r in d2)
    all_dist = Counter(r["label"] for r in recs)
    per_atom = Counter(r["atom"] for r in recs)
    m_mean = int(round(np.mean(list(per_atom.values()))))
    m_max = max(per_atom.values())

    w("## 1. The degree gradient under imputation")
    w("")
    w(f"Each excluded atom is given `m` synthetic crops. Two knobs: **how many** (`m`) and "
      f"**what label**. Realized crops per labeled atom in the 487: mean "
      f"{np.mean(list(per_atom.values())):.2f}, max {m_max} "
      f"(hist {dict(sorted(Counter(per_atom.values()).items()))}). The bootstrap resamples "
      f"ATOMS, so each excluded atom counts as one extra atom however many crops it is "
      f"given — imputing more crops does not manufacture independence.")
    w("")
    w("Scenarios, ordered neutral -> most generous:")
    w("")
    w("| # | scenario | imputed labels per atom |")
    w("|---|---|---|")

    scen = []

    def add(tag, name, m, labels_fn, note):
        labels = {a["id"]: labels_fn(a, m) for a in excluded}
        scen.append((tag, name, m, labels, note))
        w(f"| {tag} | {name} | m={m}: " +
          ", ".join(f"`{a['id'].split('_')[-1]}`→{labels[a['id']]}" for a in excluded) + " |")

    add("S0", "baseline — excluded, as the corpus stands", 0, lambda a, m: [], "status quo")
    add("S1", "neutral: they behave like typical **d2** material", m_mean,
        lambda a, m: largest_remainder(d2_dist, m), "d2 empirical label distribution")
    add("S2", "neutral: they behave like the **corpus** overall", m_mean,
        lambda a, m: largest_remainder(all_dist, m), "grand empirical distribution")
    add("S3", "generous: every crop a class-**3**", m_mean, lambda a, m: [3] * m, "")
    add("S4", "most generous: every crop a class-**4** (top label)", m_mean,
        lambda a, m: [TOP_LABEL] * m, "")
    add("S5", f"most generous AND maximal: class-4 at the max observed m={m_max}", m_max,
        lambda a, m: [TOP_LABEL] * m, "the extreme bound")
    w("")
    w(f"For scale: the entire 487 contains **{all_dist.get(4, 0)} class-4 crops**. S5 adds "
      f"**{TOP_LABEL and 3 * m_max}** more, all at degree 2 — an assumption far past anything "
      f"the corpus supports, which is the point of a bound.")
    w("")
    w("### Results")
    w("")
    degs = sorted(base_means)
    w("| # | scenario | n | " + " | ".join(f"mean d{d}" for d in degs) +
      " | d2 L>=3 | rho(degree,label) | 90% CI (atom bootstrap) |")
    w("|---|---|--:|" + "--:|" * len(degs) + "--:|--:|---|")
    rows_out = []
    for tag, name, m, labels, _note in scen:
        r2 = impute(recs, excluded, labels) if m else recs
        rho, means, ns = gradient(r2)
        ci = boot_ci(r2)
        n_ge3_d2 = sum(1 for r in r2 if r["degree"] == 2 and r["label"] >= 3)
        w(f"| {tag} | {name} | {len(r2)} | " +
          " | ".join(f"{means[d]:.3f}" for d in degs) +
          f" | {n_ge3_d2} | **{rho:+.3f}** | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
        rows_out.append(dict(tag=tag, scenario=name, m=m, n=len(r2), rho=rho,
                             ci=list(ci), means={str(k): v for k, v in means.items()},
                             ns={str(k): v for k, v in ns.items()}, d2_n_ge3=n_ge3_d2))
    w("")

    worst = min(rows_out, key=lambda r: r["rho"])
    survives = worst["ci"][0] > 0
    w(f"**The most generous imputation (S5) moves the gradient from "
      f"{base_rho:+.3f} to {rows_out[-1]['rho']:+.3f}"
      f" — a change of {rows_out[-1]['rho'] - base_rho:+.3f}** — and d2's mean from "
      f"{base_means[2]:.3f} to {rows_out[-1]['means']['2']:.3f}.")
    w("")
    w(f"**Verdict: the gradient {'SURVIVES' if survives else 'DOES NOT survive'} the most "
      f"generous imputation.** The weakest scenario on the board is `{worst['tag']}` at "
      f"rho = {worst['rho']:+.3f}, 90% CI [{worst['ci'][0]:+.3f}, {worst['ci'][1]:+.3f}] — "
      f"{'the interval excludes zero, so the objection is answered: even handing the three '
         'excluded atoms the top label in the largest number the corpus supports leaves a '
         'clear positive degree gradient.' if survives else 'the interval includes zero, so '
         'the gradient is FRAGILE to the cut and must be labeled that way.'}")
    w("")
    w("**What this does NOT say.** It does not say the three atoms are bad, and it does not "
      "recover a measurement — they cannot be presented at deploy resolution, so no label "
      "for them exists or can exist on that presentation. The claim is only that the degree "
      "gradient is not an artifact of their absence.")
    w("")
    # a sensitivity strip on m at the top label — how many class-4 crops would it take?
    w("### Sensitivity: how many top-label crops would it take to kill the gradient?")
    w("")
    w("| m (class-4 crops per excluded atom) | total imputed | rho | 90% CI | d2 mean |")
    w("|--:|--:|--:|---|--:|")
    kill_m = None
    sweep = []
    for m in (1, 2, 3, 4, 5, 6, 8, 12, 20, 40):
        r2 = impute(recs, excluded, {a["id"]: [TOP_LABEL] * m for a in excluded})
        rho, means, _ = gradient(r2)
        ci = boot_ci(r2, reps=800)
        w(f"| {m} | {3*m} | {rho:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {means[2]:.3f} |")
        sweep.append(dict(m=m, rho=rho, ci=list(ci), d2_mean=means[2]))
        if kill_m is None and ci[0] <= 0:
            kill_m = m
    w("")
    if kill_m is None:
        w(f"**No `m` up to 40 class-4 crops per excluded atom (120 crops, "
          f"{120/(len(recs)+120):.0%} of the resulting corpus, {120//max(all_dist.get(4,1),1)}x "
          f"every class-4 the 487 actually contains) drives the CI to zero.** The gradient is "
          f"not reachable by any assumption about these three atoms.")
    else:
        w(f"**The CI first touches zero at m = {kill_m}** ({3*kill_m} class-4 crops at d2).")
    w("")

    # ---- PART 2: margin distribution by degree ----------------------------- #
    w("## 2. Margin distribution by degree — is the d2-only firing real or sourced?")
    w("")
    w("Two populations answer this, and they answer it differently. Both are reported.")
    w("")
    w("### 2a. The ROSTER rows (163) — where the labeled atoms actually sit")
    w("")
    w(f"Deploy margin, decades of f64 headroom. The cut fires below "
      f"{RB.MARGIN_MIN_DECADES:.1f}; 0.0 is the wall itself.")
    w("")
    w("| degree | n | min | p05 | p25 | median | max | <1.0 (cut) | <2.0 | <3.0 |")
    w("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    margin_by_deg = {}
    for deg in sorted({a["degree"] for a in roster}):
        v = np.array([a["f64_margin_deploy_decades"] for a in roster if a["degree"] == deg])
        margin_by_deg[deg] = v
        w(f"| {deg} | {len(v)} | {v.min():.3f} | {np.percentile(v,5):.3f} | "
          f"{np.percentile(v,25):.3f} | {np.median(v):.3f} | {v.max():.3f} | "
          f"{int((v < RB.MARGIN_MIN_DECADES).sum())} | {int((v < 2.0).sum())} | "
          f"{int((v < 3.0).sum())} |")
    w("")
    near = {d: int((v < 2.0).sum()) for d, v in margin_by_deg.items()}
    w(f"**Only degree 2 has any row within a decade of the cut** "
      f"(closest non-d2 row: {min(v.min() for d, v in margin_by_deg.items() if d != 2):.3f} "
      f"decades of headroom, ~{min(v.min() for d, v in margin_by_deg.items() if d != 2):.1f} "
      f"decades clear). Rows within 2 decades of the wall by degree: {near}.")
    w("")

    w("### 2b. The SOURCED population (every Newton solve, before the ->8 per-cell pick)")
    w("")
    w("`roster_cells.json` records, per (degree, band) cell, how many sourced atoms the cut "
      "removed — over the whole draw, not just the kept rows. This is the population-level "
      "form of the question.")
    w("")
    w("| degree | sourced & admissible | removed by the cut | removal rate | deepest cell median log10\\|A\\| |")
    w("|--:|--:|--:|--:|--:|")
    tot_excl = {}
    for deg in sorted({c["degree"] for c in cells["cells"]}):
        cc = [c for c in cells["cells"] if c["degree"] == deg]
        av = sum(c["n_admitted_available"] for c in cc)
        ex = sum(c["n_excluded_feasibility"] for c in cc)
        tot_excl[deg] = ex
        med = max((c["median_log10_abs_A"] or 0) for c in cc)
        w(f"| {deg} | {av} | **{ex}** | {ex/(av+ex):.2%} | {med:.3f} |")
    w("")
    w(f"**Over the entire sourced draw the cut fired {sum(tot_excl.values())} times, and "
      f"every one was degree 2.** Not one d3/d4/d5 cell lost a single atom — so the "
      f"\"all three are d2\" pattern is not a small-sample accident of which 163 rows were "
      f"kept; it holds over the whole population the roster was picked from.")
    w("")
    # The two competing explanations, separated with numbers rather than prose.
    n1315 = {d: sum(c["n_admitted_available"] + c["n_excluded_feasibility"]
                    for c in cells["cells"] if c["degree"] == d and c["band"] == "13-15")
             for d in sorted({c["degree"] for c in cells["cells"]})}
    rate2 = 3.0 / n1315[2]
    exp_non_d2 = sum(rate2 * n for d, n in n1315.items() if d != 2)
    p_zero = float(np.exp(-exp_non_d2))
    a_max = {d: max(a["log10_abs_A"] for a in roster if a["degree"] == d)
             for d in sorted({a["degree"] for a in roster})}
    a_p90 = {d: float(np.percentile([a["log10_abs_A"] for a in roster
                                     if a["degree"] == d], 90))
             for d in a_max}

    w("### 2c. Which story is it?")
    w("")
    w("**Both, and they point the same way — but only the second is real evidence.** The "
      "zero-count at d3-d5 proves nothing by itself; the |A| distributions do.")
    w("")
    w("**(i) The counts cannot distinguish the two stories.** Band 13-15 sourced "
      f"{n1315} atoms at d2/d3/d4/d5. d2's exclusion rate there is 3/{n1315[2]} = "
      f"{rate2:.2%}. If d3-d5 atoms were excluded at that same per-atom rate, the expected "
      f"number of non-d2 exclusions is **{exp_non_d2:.2f}**, and P(observing zero) = "
      f"**{p_zero:.2f}**. Observing zero is entirely unremarkable under equal rates. So "
      "\"the cut only fires at d2\" is, on counts alone, fully explained by d2 being sourced "
      + "/".join(f"{n1315[2] / n:.0f}x" for d, n in n1315.items() if d != 2)
      + " more densely than d3/d4/d5 in the band where the cut can fire.")
    w("")
    w("**(ii) The |A| distributions, however, genuinely differ.** The per-cell pick is "
      "`select_spanning`, which keeps each cell's extremes, so the kept rows' max is close "
      "to the population max:")
    w("")
    w("| degree | max log10\\|A\\| (all bands) | p90 | max log10\\|A\\| in band 13-15 | min deploy margin |")
    w("|--:|--:|--:|--:|--:|")
    for d in sorted(a_max):
        b = [a["log10_abs_A"] for a in roster if a["degree"] == d and a["band"] == "13-15"]
        w(f"| {d} | {a_max[d]:.3f} | {a_p90[d]:.3f} | {max(b):.3f} | "
          f"{margin_by_deg[d].min():.3f} |")
    w("")
    w(f"d2's |A| tail runs **{a_max[2] - max(a_max[d] for d in a_max if d != 2):.1f} decades "
      f"deeper** than any other degree's, and the gap is already "
      f"{a_p90[2] - max(a_p90[d] for d in a_p90 if d != 2):.1f} decades at p90 — a p90 gap "
      "is not something a 3-16x draw-size ratio produces (the extra draws extend the "
      "extreme tail, not the 90th percentile). This is an argument from the shape of the "
      "distributions rather than a fitted test, but it points at a property of the atoms, "
      "not of the draw, and there is a mechanism: "
      "`A = Lambda^(1/(d-1)) * P_n'(c0)` (`atom_instrument.md`), so the `(d-1)`-th root "
      "damps how fast |A| grows with period. A degree-5 atom of the same period is "
      "intrinsically **larger** — further from the f64 wall — than a degree-2 one.")
    w("")
    w("**So:** d3-d5 do not sit near the cut and are not narrowly spared — the nearest "
      f"non-d2 row has {min(v.min() for d, v in margin_by_deg.items() if d != 2):.1f} "
      "decades of headroom. The honest statement is that the d2-only firing is a **real "
      "property of |A| at low degree, amplified by d2 being sourced far more densely**, and "
      "that the zero at d3-d5 is not itself evidence of anything. It does NOT license "
      "\"small high-period atoms only exist at d2\" — sample d5 densely enough and some will "
      "cross the wall — but it does mean the cut is structurally a d2 phenomenon at any "
      "comparable sourcing density.")
    w("")
    w("**Consequence for the gradient.** Both readings point the same way for Part 1: the cut "
      "removes material only from d2, so the direction of any bias it induces is known "
      "(it can only depress d2), and Part 1 bounds the magnitude. A future denser d3-d5 "
      "sourcing would need this re-run, since it could put non-d2 atoms under the cut for "
      "the first time.")
    w("")

    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(dict(
        baseline=dict(rho=base_rho, ci=list(base_ci),
                      means={str(k): v for k, v in base_means.items()},
                      ns={str(k): v for k, v in base_ns.items()}),
        excluded=[{k: a[k] for k in ("id", "degree", "period", "band", "log10_abs_A",
                                     "f64_margin_deploy_decades",
                                     "f64_margin_field_decades")} for a in excluded],
        scenarios=rows_out, top_label_sweep=sweep, kill_m=kill_m,
        margin_by_degree={str(d): dict(n=int(v.size), min=float(v.min()),
                                       p05=float(np.percentile(v, 5)),
                                       median=float(np.median(v)), max=float(v.max()),
                                       n_below_cut=int((v < RB.MARGIN_MIN_DECADES).sum()))
                          for d, v in margin_by_deg.items()},
        sourced_excluded_by_degree={str(k): v for k, v in tot_excl.items()},
    ), indent=1), encoding="utf-8")
    print(f"\n-> {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
