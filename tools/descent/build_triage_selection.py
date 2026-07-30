#!/usr/bin/env python
"""Turn the triage wall's **accepted** atoms into a descent-harness selection set.

The wall's whole purpose: `selection.json` (40 atoms drawn per-degree from the
roster) is an unfiltered sample, so the descent tool spends Matt's time on atoms his
eye would have thrown away in a second. The accepted set is that same time spent once,
up front, at ~1s per atom.

Writes `data/descent_harness/selection_triage.json` in the exact schema
`store.load_selection()` reads. **`selection.json` is left in place** — this is an
additional selection file, not a replacement, so the original 40-atom study is still
reproducible. Point the descent harness at it with:

    DESCENT_SELECTION=data/descent_harness/selection_triage.json \
        uv run python tools/descent/app.py
    # or:  uv run python tools/descent/app.py --selection <path>

`split` is assigned here (the triage pool has none): a deterministic atom-level 70/30
train/eval draw keyed on a hash of the atom id, so it is **stable as the pool grows** —
adding atoms never reshuffles the atoms already assigned (which a permutation-based
draw would). Atom-level == minibrot-disjoint by construction, as in the roster.

Run:  uv run python tools/descent/build_triage_selection.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import triage_store as ts     # noqa: E402

OUT = ts.TRIAGE_DIR.parent / "selection_triage.json"
TRAIN_FRAC = 0.70
SPLIT_SALT = "triage-split-v1"


def split_for(atom_id: str, train_frac: float = TRAIN_FRAC) -> str:
    """Deterministic, growth-stable 70/30 assignment from the atom id alone."""
    h = hashlib.sha256(f"{SPLIT_SALT}|{atom_id}".encode()).hexdigest()
    u = int(h[:12], 16) / float(1 << 48)
    return "train" if u < train_frac else "eval"


def build(train_frac: float = TRAIN_FRAC) -> dict:
    pool = {a["id"]: a for a in ts.load_pool()}
    verdicts = ts.load_verdicts()
    accepted = [pool[i] for i in pool if verdicts.get(i) == "accept"]
    accepted.sort(key=lambda a: a["id"])
    atoms = [{
        "id": a["id"], "degree": a["degree"], "period": a["period"],
        "split": split_for(a["id"], train_frac), "family": a["family"],
        "cx": a["cx"], "cy": a["cy"], "fw": a["fw"],
        "f64_margin_deploy_decades": a["f64_margin_deploy_decades"],
        "f64_margin_field_decades": a["f64_margin_field_decades"],
    } for a in accepted]
    n_rej = sum(1 for v in verdicts.values() if v == "reject")
    return {
        "source_pool": ts.rel(ts.POOL),
        "source_verdicts": ts.rel(ts.VERDICTS),
        "selection_rule": "triage verdict == accept",
        "train_frac": train_frac,
        "split_salt": SPLIT_SALT,
        "pool_size": len(pool),
        "n_accepted": len(atoms),
        "n_rejected": n_rej,
        "n_untriaged": len(pool) - len(verdicts),
        "note": ("Accepted set from the minibrot triage wall (tools/descent/triage_app.py). "
                 "Additional to data/descent_harness/selection.json, which is left in place. "
                 "The rejected atoms are the negative class and live in the verdict log, not "
                 "here. No model output belongs in this file."),
        "atoms": atoms,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--allow-empty", action="store_true",
                    help="write even when nothing has been accepted yet")
    args = ap.parse_args(argv)

    doc = build(args.train_frac)
    if not doc["atoms"] and not args.allow_empty:
        print(f"nothing accepted yet ({doc['n_untriaged']}/{doc['pool_size']} un-triaged) — "
              f"run the wall first, or pass --allow-empty", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    by_deg = Counter(a["degree"] for a in doc["atoms"])
    by_split = Counter(a["split"] for a in doc["atoms"])
    print(f"wrote {ts.rel(args.out)}: {len(doc['atoms'])} accepted "
          f"of {doc['pool_size']} pool ({doc['n_rejected']} rejected, "
          f"{doc['n_untriaged']} un-triaged)")
    print(f"  per-degree: {dict(sorted(by_deg.items()))}")
    print(f"  per-split:  {dict(sorted(by_split.items()))}")
    print(f"  use with:   DESCENT_SELECTION={ts.rel(args.out)} "
          f"uv run python tools/descent/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
