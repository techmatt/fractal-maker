"""One-time (idempotent) back-stamp of `julia_schema` onto every existing discovery ledger.

Both writing arms now stamp the tag on new rows; this stamps the rows that predate the
tag. For each julia row we DETECT the era from field presence (`detect_schema`), but we do
not trust the detection blindly: before writing the tag we VERIFY the row round-trips — the
location `descriptor.location_of` resolves it to AFTER stamping must equal the location an
independent field-presence resolver (`_reference_location`, the de-facto reader every
consumer used before the tag existed) produces. A mismatch, a contradictory row (both
layouts), or an undetectable row aborts the file without writing.

Native and phoenix rows carry no schema ambiguity and are passed through untouched.

    uv run python tools/corpus/backstamp_julia_schema.py            # stamp in place
    uv run python tools/corpus/backstamp_julia_schema.py --dry-run  # report only, write nothing
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import julia_ledger_schema as jls  # noqa: E402
from tools.emission import descriptor as D  # noqa: E402  (location_of — the round-trip target)

# Every discovery ledger that can hold julia rows. Glob is broad; non-julia rows are
# skipped, so phoenix-only / native-only ledgers pass through as no-ops.
LEDGER_GLOBS = (
    "data/discovery/**/outcome_ledger*.jsonl",
    "data/discovery/**/all_outcomes.jsonl",
)


def _reference_location(row: dict):
    """The location `row` resolves to TODAY, via the pre-tag field-presence reader (walk:
    viewport = julia_z_*, c = outcome_*; campaign: viewport = outcome_*, c = julia_c_*).
    Independent of the asserted resolver so the round-trip check is a genuine cross-check."""
    fam = D.render_family_of(row["family"])
    if jls._has(row, "julia_z_cx", "julia_z_cy", "julia_z_fw"):      # walk shape
        cx, cy, fw_v, c_re, c_im = (row["julia_z_cx"], row["julia_z_cy"], row["julia_z_fw"],
                                    row["outcome_cx"], row["outcome_cy"])
    elif jls._has(row, "julia_c_re", "julia_c_im"):                  # campaign shape
        cx, cy, fw_v, c_re, c_im = (row["outcome_cx"], row["outcome_cy"], row["outcome_fw"],
                                    row["julia_c_re"], row["julia_c_im"])
    else:
        raise ValueError(f"row {row.get('id')!r}: no julia coordinate layout present")
    fw = float(fw_v)
    from tools.corpus import location as loc_mod
    return loc_mod.Location(family=fam, cx=str(cx), cy=str(cy), fw=str(fw),
                            maxiter=D.auto_maxiter(fw), c_re=str(c_re), c_im=str(c_im))


def _ledger_files() -> list[str]:
    seen: list[str] = []
    for g in LEDGER_GLOBS:
        for f in glob.glob(os.path.join(ROOT, g), recursive=True):
            if f not in seen:
                seen.append(f)
    return sorted(seen)


def stamp_file(path: str, dry_run: bool) -> dict:
    """Stamp one ledger. Returns per-file counts. Raises before writing on any verification
    failure (round-trip mismatch / contradiction / undetectable), so a bad file is never
    half-written."""
    rows = []
    n_julia = n_stamped = n_already = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if jls.is_julia_row(row):
                n_julia += 1
                if jls.SCHEMA_KEY in row:
                    n_already += 1
                else:
                    ref = _reference_location(row)          # today's resolution (pre-tag)
                    jls.stamp(row)                           # detect era + set tag
                    got = D.location_of(row)                 # asserted resolution (post-tag)
                    if got.key() != ref.key():
                        raise AssertionError(
                            f"{path}: row {row.get('id')!r} does NOT round-trip after stamping "
                            f"({jls.schema_of(row)}):\n  ref = {ref.key()}\n  got = {got.key()}")
                    n_stamped += 1
            rows.append(row)
    if n_stamped and not dry_run:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp, path)
    return {"path": path, "julia": n_julia, "stamped": n_stamped, "already": n_already}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="verify + report, write nothing")
    args = ap.parse_args()

    files = _ledger_files()
    print(f"scanning {len(files)} ledger file(s){' (dry-run)' if args.dry_run else ''}\n")
    tot_julia = tot_stamped = tot_already = 0
    for path in files:
        r = stamp_file(path, args.dry_run)
        if r["julia"]:
            rel = os.path.relpath(path, ROOT)
            print(f"  {rel}: {r['julia']} julia rows, "
                  f"stamped {r['stamped']}, already-tagged {r['already']}")
        tot_julia += r["julia"]; tot_stamped += r["stamped"]; tot_already += r["already"]
    print(f"\n{'would stamp' if args.dry_run else 'stamped'} {tot_stamped} of {tot_julia} julia "
          f"rows ({tot_already} already tagged) across {len(files)} files — all round-trip verified")


if __name__ == "__main__":
    main()
