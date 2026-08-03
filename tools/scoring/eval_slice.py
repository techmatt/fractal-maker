"""THE frozen eval slice: where a version's is, and what its score columns are called.

`tools/<v>/eval_<v>.py` freezes one JSONL row per eval location — the human `label`, the
`fractal_type`, the `source` instrument, the `location_id`, and that head's cumulative
probabilities under a **version-prefixed** column name. Two conventions, both of which were
re-derived by hand at every read site:

    path     data/<version>/eval_scores_<version>.jsonl
    columns  <version>_p_ge2   P(label>=2) = P(not bad)
             <version>_p_ge3   P(label>=3) = the column t_good and the keeper cut are cut on
             <version>_p_ge4   P(label>=4) — K=4 heads only (v8 onward); absent on v5..v7
             <version>_score   the monotone rank score, sum of the cutpoint sigmoids

The prefix is version-scoped ON PURPOSE: a slice re-scored under a new head gains columns
rather than overwriting them, so a row can be re-decoded under either. That is also why
"read the eval slice" is never a bare filename — every reader needs the pin to name a column,
and each one used to spell out both conventions inline. This module owns them.

    import eval_slice
    rows = eval_slice.load()                      # the ACTIVE version's slice
    p2, p3, p4 = eval_slice.probs(rows[0])        # p4 is None on a K=3 slice

`version` defaults to `production_pins.ACTIVE_VERSION` everywhere, so a reader follows the
pin instead of racing it — the failure this repo has hit twice is a threshold quietly
re-derived off the PREVIOUS head's columns because a literal "v8" outlived the flip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools" / "scoring") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "scoring"))

from production_pins import ACTIVE_VERSION  # noqa: E402

#: Cutpoint column suffixes, in rank order. Index k is P(label >= k+2).
CUTPOINTS = ("p_ge2", "p_ge3", "p_ge4")


def _v(version: str | None) -> str:
    return version or ACTIVE_VERSION


def rel_for(version: str | None = None) -> str:
    """Repo-relative path of a version's frozen eval slice."""
    v = _v(version)
    return f"data/{v}/eval_scores_{v}.jsonl"


def path_for(version: str | None = None) -> Path:
    """Absolute path of a version's frozen eval slice."""
    return ROOT / rel_for(version)


def column(name: str, version: str | None = None) -> str:
    """A version-prefixed column name: `column("p_ge3")` -> `"v10_p_ge3"`."""
    return f"{_v(version)}_{name}"


def load(version: str | None = None, path=None) -> list[dict]:
    """Rows of a version's frozen eval slice. `path` overrides the resolved location (a
    caller re-deriving a slice that is not the pinned one still gets the column helpers)."""
    p = Path(path) if path is not None else path_for(version)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def probs(row: dict, version: str | None = None) -> tuple[float, float, float | None]:
    """`(P(>=2), P(>=3), P(>=4))` for one row; the third is **None on a K=3 slice**.

    None is not a missing value to paper over — it is the K=3 decode, and
    `derive_t_good.keeper_pred` / `score_lib.corn_decode` both take it as such. Silently
    substituting 0.0 would turn a rank-consistent 3-class decode into a 4-class one that
    can never promote."""
    v = _v(version)
    return (row[f"{v}_p_ge2"], row[f"{v}_p_ge3"], row.get(f"{v}_p_ge4"))


def has_class4(rows, version: str | None = None) -> bool:
    """True iff the slice carries the third cutpoint on EVERY row (K=4, v8 onward).

    All-or-none is enforced by the caller that sweeps (`derive_t_good.great_column` raises):
    a slice where only some rows carry the column would sweep two different predicates over
    one population."""
    key = column("p_ge4", version)
    return bool(rows) and all(key in r for r in rows)
