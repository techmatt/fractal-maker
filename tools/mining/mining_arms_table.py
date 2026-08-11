r"""mining_arms_table.py — the (28) render-mode arms, side by side, one table.

`prompts/retrains_28_addendum_augarm.md` §3 and `retrains_28_addendum_uniform.md` §7 both ask
for the same object: every arm's per-seed band, its staged pick, its winner-rule verdict
against the incumbent, and — the part that makes the table an explanation rather than a
scoreboard — the PER-KIND TRAINING-WEIGHT SHARES beside the outcome, so the mechanism story
each arm supports is legible without opening four run dirs.

THE ARMS DISCIPLINE, and it is the reason this file lists them rather than globbing:
each arm changes exactly ONE thing against `dedup_weighted`. There is no combined
aug+uniform arm — that needs Matt's explicit word (addendum 3 §6) and does not have it.

    arm             geometry                                weights        objective
    dedup_weighted  border 0.05/edge + flips  (v1's recipe) 1/group_size   AP>=3
    aug_gentle      axis 0.03/axis + flips                  1/group_size   AP>=3
    aug_strong      border 0.10/edge + axis 0.03 + flips    1/group_size   AP>=3
    uniform         border 0.05/edge + flips  (v1's recipe) UNIFORM        AP>=3
    ap2_selected    border 0.05/edge + flips  (v1's recipe) 1/group_size   AP>=2

THE FIFTH ARM IS THE ONE THAT SETTLES THE OTHER FOUR (`prompts/settlement_28b.md` §1).
Arms 1-4 moved geometry across a 4x range and lifted the near-dup weighting entirely, and
the failure set never moved: the same five cells fail in all four, every arm wins the >=3
boundary and loses the >=2 boundary. That is what "v1 was SELECTED on AP>=2 and every v3 arm
was selected on AP>=3" predicts, and it was the only surviving explanation because nothing
had tested it. `ap2_selected` is that test: v1's own objective on an otherwise identical arm,
declared before the run in `classifier.train_mining_head_v3.SELECTION_METRICS`.

READ IT AS A SETTLING TEST, NOT A CANDIDATE. The arm is selected on the SAME eval side its
>=2 cells are then read on, so a >=2 gain here is partly the selection finding it; that is
inherent to the question and is why the arm answers "is the objective the mechanism", not
"should the objective change". Which objective the head's role needs is Matt's decision.

`dedup_weighted`'s geometry is the row every reading of these arms turns on, and it is NOT
"no augmentation": v1's recipe already crops U(0,5%) off each of the four edges and flips on
both axes. Only COLOUR aug is off, and deliberately — palette is the label. The addendum's
0-3%/axis arm is therefore GENTLER geometry than the arm it was proposed as an addition to,
which is why `aug_strong` exists: the two bracket the measured arm on both sides.

Reads only. Every input is a committed run record; nothing here trains, scores or moves a pin.

    uv run python tools/mining/mining_arms_table.py
    uv run python tools/mining/mining_arms_table.py --out scratch/retrains_28_arms.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HEADS = ROOT / "data" / "render_mode_head"


@dataclass(frozen=True)
class Arm:
    key: str
    dirname: str
    geometry: str
    weights: str
    objective: str
    why: str


ARMS = (
    Arm("dedup_weighted", "v3", "border 0.05/edge + flips (v1 recipe)", "1/group_size",
        "ap_ge3", "the declared retrain"),
    Arm("aug_gentle", "v3_aug", "axis 0.03/axis + flips", "1/group_size",
        "ap_ge3", "addendum 2 as written — GENTLER geometry than the declared arm"),
    Arm("aug_strong", "v3_augx", "border 0.10/edge + axis 0.03/axis + flips", "1/group_size",
        "ap_ge3", "the other bracket: stronger geometry than the declared arm"),
    Arm("uniform", "v3_uniform", "border 0.05/edge + flips (v1 recipe)", "UNIFORM",
        "ap_ge3", "addendum 3 — the settling arm for the near-dup weighting"),
    Arm("ap2_selected", "v3_ap2", "border 0.05/edge + flips (v1 recipe)", "1/group_size",
        "ap_ge2", "(28b) §1 — the settling arm for the SELECTION OBJECTIVE, v1's own"),
)

METRIC_KEYS = ("auc_ge3", "ap_ge3", "auc_ge2", "ap_ge2")


def load(arm: Arm) -> dict | None:
    d = HEADS / arm.dirname
    m, r = d / "metrics.json", d / "report.json"
    if not m.exists():
        return None
    out = {"arm": arm, "dir": d, "metrics": json.loads(m.read_text(encoding="utf-8"))}
    out["report"] = json.loads(r.read_text(encoding="utf-8")) if r.exists() else None
    return out


def derive_dose(uniform: bool) -> dict:
    """Per-kind training weight removed, computed from the corpus rather than read.

    `dedup_weighted` was trained before the trainer stamped `train_weight_by_kind`, so its
    record has no dose. The dose is a PURE FUNCTION of the pooled corpus and the weighting
    mode, so it is derived here for every arm and cross-checked against the stamp where one
    exists — deriving it is better than back-editing a committed run record, and the
    cross-check is what stops the derivation and the stamp drifting apart."""
    from tools.mining.mining_corpus import load_corpus
    tr = load_corpus(require_crops=False).train
    out = {}
    for k in sorted({r.kind for r in tr}):
        ws = [1.0 if uniform else r.weight for r in tr if r.kind == k]
        out[k] = {"train_rows": len(ws), "effective_weight": round(sum(ws), 2),
                  "down_weighted_pct": round(100 * (1 - sum(ws) / len(ws)), 2)}
    return out


def dose_for(r: dict) -> tuple[dict, str]:
    """`(dose, note)` — the stamped dose if the record has one, else the derived one."""
    stamped = r["metrics"].get("train_weight_by_kind")
    derived = derive_dose(r["arm"].weights == "UNIFORM")
    if not stamped:
        return derived, "derived"
    for k, v in derived.items():
        if abs(v["down_weighted_pct"] - stamped[k]["down_weighted_pct"]) > 0.01:
            raise AssertionError(
                f"{r['arm'].key}: stamped dose for {k} "
                f"({stamped[k]['down_weighted_pct']}%) disagrees with the corpus "
                f"({v['down_weighted_pct']}%) — one of them describes another run")
    return stamped, "stamped"


def _pct(share: dict | None, kind: str) -> str:
    if not share or kind not in share:
        return "—"
    return f"{share[kind]['down_weighted_pct']:.1f}%"


def build(rows: list[dict]) -> str:
    L, A = [], None
    A = L.append
    A("## Render-mode arms — one corpus, one split, one objective, one variable each\n")
    A("Eval slice is identical across arms: 827 rows (973 before near-dup dedup) over 136 "
      "locations, 214 good. Baseline is mining **v1** re-scored on the same crops through "
      "the same scorer in every row.\n")

    A("| arm | geometry | weights | objective | staged AP≥3 | 5-seed AP≥3 | 5-seed AUC≥3 "
      "| 5-seed AUC≥2 | verdict |")
    A("|---|---|---|---|---:|---|---|---|---|")
    for r in rows:
        a, m = r["arm"], r["metrics"]
        rep = r["report"]
        # `staged.ap_good` is AP>=3 AT THE STAGED EPOCH on every arm, including the one whose
        # OBJECTIVE is AP>=2 — so this column compares the same quantity across all five.
        # (`staged.selection_value` is the objective's own number; the older four arms have
        # no such key because their objective and this column were the same thing.)
        staged = m.get("staged", {}).get("ap_good")
        obj = (m.get("selection") or {}).get("metric") or a.objective
        if obj != a.objective:
            raise AssertionError(
                f"{a.key}: run record says objective {obj!r}, this table says "
                f"{a.objective!r} — one of them describes another run")
        band = (rep or {}).get("v3_seed_band", {}).get("mean_sd", {})
        def bd(k):
            b = band.get(k)
            return "—" if not b else f"{b['mean']:.3f} ± {b['sd']:.3f}"
        wr = (rep or {}).get("winner_rule", {})
        v = "—"
        if wr:
            v = ("**candidate**" if wr["winner"] == "v3" else "v1 keeps") + \
                f" (a {'✓' if wr['clause_a']['pass'] else '✗'}/" \
                f"b {'✓' if wr['clause_b']['pass'] else '✗'})"
        A(f"| `{a.key}` | {a.geometry} | {a.weights} | `{a.objective}` | "
          f"{'—' if staged is None else f'{staged:.3f}'} | {bd('ap_ge3')} | "
          f"{bd('auc_ge3')} | {bd('auc_ge2')} | {v} |")

    # The cell the fifth arm was run to settle, pulled out of the per-arm reports so the
    # comparison is one table rather than five files. `pooled.auc_ge2` fails in all four
    # AP>=3-selected arms; whether it still fails under v1's own objective is the answer.
    A("\n### The pooled ≥2 boundary — the cell the objective story predicts\n")
    A("| arm | objective | pooled AUC≥2 Δ (95% CI) | pooled AUC≥3 Δ (95% CI) |")
    A("|---|---|---|---|")
    for r in rows:
        a = r["arm"]
        nw = ((r["report"] or {}).get("no_worse") or {}).get("pooled")
        def ci(key):
            c = (nw or {}).get("delta_ci", {}).get(key)
            if not c or c.get("n_draws", 0) == 0:
                return "—"
            tag = " **worse**" if c["significantly_worse"] else (
                " **better**" if c["significantly_better"] else "")
            return f"{c['median']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]{tag}"
        A(f"| `{a.key}` | `{a.objective}` | {ci('auc_ge2')} | {ci('auc_ge3')} |")

    A("\n### Mechanism — training weight removed per kind (the dose, beside the outcome)\n")
    A("| arm | direct | composite | pure | train rows | effective weight | source |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        a = r["arm"]
        share, src = dose_for(r)
        tot_rows = sum(v["train_rows"] for v in share.values())
        tot_w = sum(v["effective_weight"] for v in share.values())
        A(f"| `{a.key}` | {_pct(share, 'direct')} | {_pct(share, 'composite')} | "
          f"{_pct(share, 'pure')} | {tot_rows} | {tot_w:.1f} | {src} |")

    A("\n### Where best-epoch landed (the early-plateau caveat's test)\n")
    A("| arm | best epoch per seed | median | early (≤10 of 40)? |")
    A("|---|---|---:|---|")
    for r in rows:
        eps = [s.get("best_epoch") for s in r["metrics"].get("per_seed", [])]
        eps = [e for e in eps if e is not None]
        if not eps:
            A(f"| `{r['arm'].key}` | — | — | — |")
            continue
        med = sorted(eps)[len(eps) // 2]
        A(f"| `{r['arm'].key}` | {' '.join(str(e) for e in eps)} | {med} | "
          f"{'YES' if med <= 10 else 'no'} |")

    passing = [r for r in rows
               if (r["report"] or {}).get("winner_rule", {}).get("winner") == "v3"]
    A("")
    if len(passing) > 1:
        A(f"**{len(passing)} arms pass the winner rule — NO auto-pick.** Each supports a "
          f"different mechanism story:")
        for r in passing:
            A(f"- `{r['arm'].key}` — {r['arm'].why}")
        A("Matt chooses.")
    elif len(passing) == 1:
        A(f"**One arm passes: `{passing[0]['arm'].key}`** — {passing[0]['arm'].why}. "
          f"Still STAGED; adoption is a separate prompt.")
    else:
        A("**No arm passes the winner rule** — v1 keeps candidacy on every one of them.")

    missing = [a.key for a in ARMS if not any(r["arm"].key == a.key for r in rows)]
    if missing:
        A(f"\nNOT RUN: {', '.join(missing)}.")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "scratch" / "retrains_28_arms.md")
    a = ap.parse_args(argv)
    rows = [r for r in (load(arm) for arm in ARMS) if r]
    if not rows:
        raise SystemExit("[arms] no arm has a metrics.json yet")
    md = build(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, encoding="utf-8")
    # The FILE is the deliverable and it is written first. The console echo is a
    # convenience and must never be able to fail the run: a redirected stdout on Windows
    # is cp1252, and the table's "≥" killed the first pass AFTER the file was already on
    # disk — an exit code that reported failure for work that had succeeded.
    sys.stdout.reconfigure(errors="replace")
    print(md)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
