"""The iteration-cap policy has two load-bearing copies. Pin them to agree.

`tools/scoring/active_ckpt.auto_maxiter` is PRODUCTION (label crops, corpus crops,
discovery renders, the descent harness's canonical block). `tools/explorer/render_core
.auto_maxiter` is the shared explorer + descent NAVIGATION cap, and `tools/orbital/`
measures through it. They are two independent transcriptions of the same four constants
and the same closed form — there is no shared source to import, because one is a `float`
implementation and the other a `Decimal` one.

That is a real duplication, so it gets a real check rather than a `# keep in sync`
comment: `# keep in sync` is not a mechanism (the v8 render-worker count had already gone
stale against exactly such a comment). If the raise lands in one copy and not the other,
production and navigation silently disagree about how deep to iterate and every
navigation-sourced measurement drifts off the production distribution.

Also pins the ADOPTED values themselves, so the raise cannot be quietly reverted, and the
non-binding-clamp claim in docs/design/auto_maxiter.md, so a future manifest that pushes
the deep tail past the clamp is a red test rather than a silent truncation.

Run:  uv run python -m pytest tools/scoring/test_maxiter_policy.py -q
"""
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "scoring"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "explorer"))

import active_ckpt as prod     # noqa: E402
import render_core as nav      # noqa: E402

# The adopted policy (docs/design/auto_maxiter.md). base 500 -> 4000 (x8, the MEDIAN of
# the measured convergent multiple), clamp 8000 -> 67000, k and min unchanged.
ADOPTED = {"base": 4000, "k": 0.30, "min": 200, "max": 67000, "fw_home": 3.0}


def test_production_constants_are_the_adopted_policy():
    assert prod.MAXITER_BASE == ADOPTED["base"]
    assert prod.MAXITER_K == ADOPTED["k"]
    assert prod.MAXITER_MIN == ADOPTED["min"]
    assert prod.MAXITER_MAX == ADOPTED["max"]
    assert float(prod.FW_HOME) == ADOPTED["fw_home"]


def test_navigation_constants_equal_production():
    assert nav.MAXITER_BASE == prod.MAXITER_BASE
    assert nav.MAXITER_K == prod.MAXITER_K
    assert nav.MAXITER_MIN == prod.MAXITER_MIN
    assert nav.MAXITER_MAX == prod.MAXITER_MAX
    assert float(nav.FW_HOME) == float(prod.FW_HOME)


def test_the_two_implementations_agree_pointwise():
    """Constants agreeing is not the same as the FUNCTIONS agreeing — one is f64 and
    one is Decimal, and the clamp/int-truncation could diverge at the edges. Sweep the
    whole live range plus both clamp shoulders."""
    fws = [3.0 * (2.0 ** -e) for e in range(0, 64)]          # fw_home down to ~1.6e-19
    fws += [4.242640687119286, 3.92635175e-10, 3.0, 1.0, 1e-3, 1e-6, 1e-13]
    fws += [1e30, 1e-300]                                     # both clamp shoulders
    for fw in fws:
        assert prod.auto_maxiter(fw) == nav.auto_maxiter(fw), fw


def test_clamps_are_reachable_and_correct():
    # an absurdly wide frame floors at MIN; an absurdly deep one caps at MAX
    assert prod.auto_maxiter(1e30) == ADOPTED["min"]
    assert prod.auto_maxiter(1e-300) == ADOPTED["max"]
    # ...and the closed form is the stated one in between
    fw = 1e-6
    want = int(ADOPTED["base"] * (1.0 + ADOPTED["k"] * math.log2(3.0 / fw)))
    assert prod.auto_maxiter(fw) == want


def test_raise_is_x8_over_the_superseded_policy():
    """The un-clamped raise is exactly x8 of the old base at every depth (k unchanged),
    which is the whole claim: the SHAPE was right, the BASE was 8x too low."""
    def old(fw):
        val = 500 * (1.0 + 0.30 * math.log2(3.0 / fw))
        return max(200, min(8000, val))
    # equality up to the shared int() truncation (both sides floor, one ulp apart)
    for fw in (3.0, 1.0, 1e-2, 1e-6, 3.92635175e-10):
        assert abs(prod.auto_maxiter(fw) - int(8.0 * old(fw))) <= 1, fw


def test_clamp_is_non_binding_over_the_v8_manifest():
    """docs/design/auto_maxiter.md claims 67000 is non-binding over the corpus as it
    stands (measured max 43,397). If a manifest ever pushes the deep tail into the
    clamp, the deep end starts truncating and the cap decision has to be revisited —
    that must be a red test, not a silent change of behaviour."""
    manifest = REPO_ROOT / "data" / "v8" / "manifest.jsonl"
    if not manifest.exists():                       # bulk/relocated checkout
        return
    caps = []
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                # the aug-cache geometry jitters fw by up to +-10%, so check the
                # widest AND deepest slot each location can produce, not just fw.
                fw = float(json.loads(line)["fw"])
                caps += [prod.auto_maxiter(fw * 0.90), prod.auto_maxiter(fw * 1.10)]
    assert caps
    assert max(caps) < ADOPTED["max"], \
        f"clamp {ADOPTED['max']} is BINDING (max cap {max(caps)}) — re-read auto_maxiter.md"
    assert max(caps) < 64000, f"deep tail at {max(caps)}, past the stated 64000 headroom"
    assert min(caps) > ADOPTED["min"], f"floor {ADOPTED['min']} is binding (min {min(caps)})"


if __name__ == "__main__":
    for t in (test_production_constants_are_the_adopted_policy,
              test_navigation_constants_equal_production,
              test_the_two_implementations_agree_pointwise,
              test_clamps_are_reachable_and_correct,
              test_raise_is_x8_over_the_superseded_policy,
              test_clamp_is_non_binding_over_the_v8_manifest):
        t()
        print(f"PASS  {t.__name__}")
