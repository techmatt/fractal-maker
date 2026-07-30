"""Production-block equality for the descent harness's canonical crop.

Round-trip identity (test_descent_smoke) proves the stored block is *self*-consistent
with its crop — it re-renders from the same block, so it cannot catch a block that is
systematically WRONG against the rest of the label corpus. This test is that missing
check: it asserts the harness's canonical render block equals the **production** block —
the 640×360 ss2 twilight_shifted native label crop built by
`tools/corpus/build_native_multibrot_band.py`, which derives `maxiter`/`palette` from
`tools/scoring/active_ckpt.py` — **field-for-field apart from the coordinates and family**.

Both the harness (`store.canonical_render_block`) and the reference below import the SAME
production primitives (`active_ckpt.auto_maxiter`, `active_ckpt.PALETTE`), so they move in
lockstep: if production's maxiter derivation or palette changes, the harness follows and
this test still passes; but a harness that reverts to a hand-copied constant — or to the
explorer's navigation `auto_maxiter` once that diverges from production — breaks it,
rather than silently drifting off-distribution.

Run:  uv run python -m pytest tools/descent/test_canonical_block.py -q
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "scoring"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "mining"))

import store            # noqa: E402
import corpus_common as cc   # noqa: E402
import active_ckpt as prod   # noqa: E402

# Corpus-canonical geometry for the twilight_shifted native label crop (matches
# build_native_multibrot_band). Only the DERIVED fields (maxiter/palette) come from prod.
GEO = dict(composition="center", width=640, height=360, ss=2,
           filter="lanczos3", interior_mode="black")


def production_ref_block(cx, cy, fw, family):
    """The authoritative production canonical block, built straight from the same
    primitives build_native_multibrot_band uses (active_ckpt + corpus_common)."""
    blk = cc.render_block(cx=cc.hp_str(cx), cy=cc.hp_str(cy), fw=cc.hp_str(fw),
                          maxiter=prod.auto_maxiter(float(fw)),
                          palette=prod.PALETTE, **GEO)
    blk["fractal_type"] = family
    blk["c_re"] = None
    blk["c_im"] = None
    return blk


def _cases():
    atoms = store.load_selection()
    picked = {}
    for a in atoms:                      # one atom per degree (distinct families)
        picked.setdefault(a["degree"], a)
    return list(picked.values())


def test_canonical_block_equals_production_field_for_field():
    for a in _cases():
        got = store.canonical_render_block(a["cx"], a["cy"], a["fw"], a["family"])
        ref = production_ref_block(a["cx"], a["cy"], a["fw"], a["family"])
        assert got == ref, f"{a['id']}: {got} != {ref}"


def test_maxiter_derives_from_production_not_fixed_not_nav():
    # maxiter DERIVES (varies with fw) via active_ckpt — not a fixed constant.
    a = _cases()[0]
    shallow = store.canonical_render_block(a["cx"], a["cy"], "1e-2", a["family"])["maxiter"]
    deep = store.canonical_render_block(a["cx"], a["cy"], "1e-6", a["family"])["maxiter"]
    assert shallow != deep, "maxiter should derive from fw, not be fixed"
    # and it is production's derivation exactly
    assert shallow == prod.auto_maxiter(1e-2)
    assert deep == prod.auto_maxiter(1e-6)


def test_vivid_differs_only_in_palette():
    a = _cases()[0]
    canon = store.canonical_render_block(a["cx"], a["cy"], a["fw"], a["family"])
    vivid = store.vivid_render_block(a["cx"], a["cy"], a["fw"], a["family"])
    assert vivid["palette"] == store.VIVID_PALETTE and canon["palette"] == prod.PALETTE
    assert {k: v for k, v in canon.items() if k != "palette"} == \
           {k: v for k, v in vivid.items() if k != "palette"}


if __name__ == "__main__":
    test_canonical_block_equals_production_field_for_field()
    test_maxiter_derives_from_production_not_fixed_not_nav()
    test_vivid_differs_only_in_palette()
    print("canonical block == production (field-for-field): PASS")
