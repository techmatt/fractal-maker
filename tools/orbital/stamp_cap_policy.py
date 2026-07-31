#!/usr/bin/env python
"""One-shot, idempotent, re-checkable: stamp every COMMITTED orbital score record with
the iteration-cap policy it was actually computed under.

Why this exists, and what a cross-policy comparison does:
`docs/design/orbital_field_metrics.md` §7 (cap policy itself: `auto_maxiter.md`).

Every existing record is stamped LEGACY, which is what it is. `field_metrics` stamps new
records with the live policy at write time, and every comparison/aggregation refuses to
mix two policies (`fm.require_one_policy`).

What gets stamped, and what deliberately does not
-------------------------------------------------
STAMPED — score records and the reports derived from them. Each is a measure over a
RENDERED FIELD, so the cap is an input to its value:
    data/orbital/measures.jsonl        945 rows   (measure_atoms)
    data/orbital/screen_scores.jsonl  3759 rows   (screen_pool phase 2)
    data/orbital/validation.json                  reference-vs-triage verdict
    data/orbital/maxiter_stability.json           drift ratios across multipliers
    data/orbital/screen_report.json               distribution + implied floor

NOT STAMPED — `data/orbital/screen_pool.jsonl`. It is the ENUMERATION, not a score:
Newton nuclei from `atom_lib.solve_nucleus` (mpmath at NEWTON_STEPS), whose fields
(cx/cy/window_scale/period/log10_abs_A/f64_margin_deploy_decades) are analytic
properties of the atom. No field is rendered and no iteration cap is consulted anywhere
on that path, so a cap token there would assert a dependence that does not exist — a
false provenance claim is worse than none. Pinned both ways by
`test_orbital.py::test_the_enumeration_is_not_stamped_with_a_cap_policy`.

The JSONL stamp is the empty-string legacy token, which is a no-op for readers by the
same invariant the field-cache stems use — the point is that the key is PRESENT, so the
file states its own provenance instead of relying on a reader's default. The JSON
reports additionally carry a human-readable `maxiter_policy` line.

Run:   uv run python tools/orbital/stamp_cap_policy.py           # stamp (idempotent)
       uv run python tools/orbital/stamp_cap_policy.py --check   # assert fully stamped
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (HERE, REPO_ROOT / "tools" / "corpus", REPO_ROOT / "tools" / "explorer",
          REPO_ROOT / "tools"):
    sys.path.insert(0, str(p))

import field_metrics as fm      # noqa: E402
import paths                    # noqa: E402

# The policy every committed orbital record was computed under. Not "whatever is live"
# — a fixed historical claim, which is the whole point of writing it down.
STAMP_POLICY = "legacy"

SCORE_JSONL = ("data/orbital/measures.jsonl", "data/orbital/screen_scores.jsonl")
SCORE_REPORTS = ("data/orbital/validation.json", "data/orbital/maxiter_stability.json",
                 "data/orbital/screen_report.json")
NOT_A_SCORE = ("data/orbital/screen_pool.jsonl",)   # enumeration — see module docstring


def _token() -> str:
    """The token for the legacy policy — resolved through the shared definition rather
    than hardcoded, so it cannot drift from `loc_mod.LEGACY_MAXITER_POLICY`."""
    import location as loc_mod
    return fm.policy_token(loc_mod.LEGACY_MAXITER_POLICY)


def stamp_jsonl(rel: str, token: str, *, check: bool) -> tuple[int, int]:
    """Returns (rows, rows_missing_the_key). Rewrites in place unless `check`."""
    p = paths.durable(rel)
    if not p.exists():
        return 0, 0
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    missing = [r for r in rows if fm.POLICY_KEY not in r]
    if missing and not check:
        for r in missing:
            r[fm.POLICY_KEY] = token
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return len(rows), len(missing)


def stamp_report(rel: str, token: str, *, check: bool) -> bool:
    """Returns True iff the file was (or would be) modified."""
    p = paths.durable(rel)
    if not p.exists():
        return False
    doc = json.loads(p.read_text(encoding="utf-8"))
    if fm.POLICY_KEY in doc:
        return False
    if not check:
        doc[fm.POLICY_KEY] = token
        doc["maxiter_policy"] = fm.describe_policy(token)
        p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return True


def audit(*, check: bool) -> dict:
    token = _token()
    out = {"token": token, "policy": fm.describe_policy(token),
           "jsonl": {}, "reports": {}, "unstamped_by_design": list(NOT_A_SCORE)}
    for rel in SCORE_JSONL:
        n, miss = stamp_jsonl(rel, token, check=check)
        out["jsonl"][rel] = {"rows": n, "needed_stamp": miss}
    for rel in SCORE_REPORTS:
        out["reports"][rel] = stamp_report(rel, token, check=check)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report only; exit 1 if anything is unstamped")
    args = ap.parse_args(argv)

    r = audit(check=args.check)
    print(f"cap-policy stamp: {r['policy']}  (token {r['token']!r})")
    pending = 0
    for rel, s in r["jsonl"].items():
        print(f"  {rel:40s} {s['rows']:5d} rows, {s['needed_stamp']:5d} "
              f"{'unstamped' if args.check else 'stamped'}")
        pending += s["needed_stamp"]
    for rel, changed in r["reports"].items():
        print(f"  {rel:40s} {'unstamped' if args.check and changed else ('stamped' if changed else 'already stamped')}")
        pending += int(bool(changed))
    for rel in r["unstamped_by_design"]:
        print(f"  {rel:40s} NOT a score record — enumeration, no cap dependence")
    if args.check and pending:
        print(f"\n{pending} unstamped item(s) — run without --check", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
