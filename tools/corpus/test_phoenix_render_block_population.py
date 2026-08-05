"""The phoenix `c` invariant, checked over the PERSISTED corpus rather than at one builder.

`build_q4_harvest_batches._render_block` sourced a phoenix row's `c` from `c_re`/`julia_c_re`
only. A q4 CHECK writes it there; an OUTCOME-LEDGER row writes it as `phoenix_c_re` and leaves
`julia_c_re` null — so a ledger-sourced phoenix row rendered the engine's DEFAULT phoenix plane
at the right coordinates: a real-looking image of a DIFFERENT fractal. Fixed at that authority,
with its own loud-raise and unit tests (`tools/atlas/test_precanon_calibration_sheet.py`).

THIS FILE IS THE OTHER HALF, and it is not redundant with those. The unit tests prove one
builder handles one row correctly; this one asks whether any crop the corpus SHIPPED was ever
built from a `c`-less phoenix block — the question a builder-level test cannot answer, because
the corpus is written by several builders and two of them null `c` unconditionally.

  THE SIGNATURE. `fractal_type == "phoenix"` with `c` absent and `p` PRESENT. Both-absent is a
  different thing and is CORRECT: the engine defaults for c AND p are the fixed Ushiki plane,
  which is exactly what a pre-parameterisation `phoenix:classic` row means (see
  `production_seeder.PHOENIX_C_DEFAULT`). Mixed — someone's p on the default c — is the fractal
  nobody asked for.

  THE BLIND SPOT, pinned by the second test. A builder that nulls `c` while never emitting `p`
  produces a row indistinguishable from the legacy classic convention, so the signature above
  cannot see it. Two live builders do exactly that (`build_supply_crawl_batches`,
  `build_label_seeded_batches`). They are fail-closed here instead: while their `_render_block`
  hardcodes `c_re = None`, their batches must contain no phoenix row at all.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "atlas")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import corpus_common as cc  # noqa: E402

# through the module that owns the path, not a hand-joined one (a batch family can relocate).
BATCHES = Path(cc.batch_dir("_")).parent
MANIFESTS = ("images.jsonl", "blind.jsonl", "sheet.jsonl")

# generator_version -> the module whose `_render_block` nulls `c` unconditionally
C_NULLING_BUILDERS = {
    "supply_crawl_v1": "tools/atlas/build_supply_crawl_batches.py",
    "label_seeded_v2": "tools/atlas/build_label_seeded_batches.py",
}


def misrendered(render: dict) -> bool:
    """True iff this render block is the bug's signature: a phoenix plane with somebody's `p`
    and no `c`, which the engine renders as `p` over the DEFAULT `c`."""
    ft = render.get("fractal_type") or render.get("family") or ""
    if "phoenix" not in str(ft):
        return False
    has_c = render.get("c_re") is not None and render.get("c_im") is not None
    has_p = render.get("p_re") is not None and render.get("p_im") is not None
    return has_p and not has_c


def _manifest_rows():
    """(batch_id, manifest, image_id, render) over every persisted corpus manifest."""
    for batch in sorted(os.listdir(BATCHES)):
        for name in MANIFESTS:
            f = BATCHES / batch / name
            if not f.exists():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                yield batch, name, r.get("image_id"), (r.get("render") or r)


def test_no_shipped_phoenix_row_lost_its_c():
    bad = [(b, m, i) for b, m, i, rn in _manifest_rows() if misrendered(rn)]
    assert not bad, (
        "phoenix render blocks with `p` but no `c` — every crop built from one is a "
        "DIFFERENT fractal at the right coordinates:\n  "
        + "\n  ".join(f"{b}/{m} {i}" for b, m, i in bad[:20]))


def test_the_signature_check_is_red_on_an_injected_row():
    """Proved red by injection: without this the test above passes on an empty read."""
    ok = dict(fractal_type="phoenix", c_re="0.32", c_im="0.12", p_re="-0.11", p_im="0.44")
    assert not misrendered(ok)
    assert misrendered({k: v for k, v in ok.items() if not k.startswith("c_")})
    # classic: BOTH absent -> the fixed Ushiki plane, which is what such a row means
    assert not misrendered(dict(fractal_type="phoenix"))
    # and a non-phoenix row is never the signature, whatever its columns say
    assert not misrendered(dict(fractal_type="julia:multibrot5", p_re="-0.11", p_im="0.44"))


def test_phoenix_rows_saw_a_builder_that_could_carry_their_c():
    """Fail-closed for the blind spot: a builder that nulls `c` and never writes `p` emits a
    phoenix row that LOOKS classic, so no signature can catch it. While such a builder exists,
    its batches must hold no phoenix row."""
    still_nulling = {
        gv: src for gv, src in C_NULLING_BUILDERS.items()
        if re.search(r'render\["c_re"\]\s*=\s*None', (ROOT / src).read_text(encoding="utf-8"))
    }
    if not still_nulling:
        pytest.skip("both builders now source `c`; the signature test covers them")
    owners = {}
    for batch in sorted(os.listdir(BATCHES)):
        bj = BATCHES / batch / "batch.json"
        if bj.exists():
            gv = json.loads(bj.read_text(encoding="utf-8")).get("generator_version")
            if gv in still_nulling:
                owners[batch] = gv
    offenders = [
        (b, i) for b, _m, i, rn in _manifest_rows() if b in owners
        and "phoenix" in str(rn.get("fractal_type") or rn.get("family") or "")
    ]
    assert not offenders, (
        "a phoenix row shipped through a builder that hardcodes `c_re = None` "
        f"({sorted(set(owners[b] for b, _ in offenders))} -> "
        f"{sorted(set(still_nulling[owners[b]] for b, _ in offenders))}): its crop is the "
        "DEFAULT phoenix plane and nothing in the record says so.\n  "
        + "\n  ".join(f"{b} {i}" for b, i in offenders[:20]))
