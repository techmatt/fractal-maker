#!/usr/bin/env python
"""Guard 81-set re-render tripwire (SLOW / opt-in).

The guard's real regression exposure is the **live f64 fast smooth-field path**
(`render-one --dump-field --dump-field-source f64`, 640x360 ss2) crossed with the
two pinned gates — not `guard.py`'s arithmetic on a frozen field. So this pins only
the 81 canonical `(cx,cy,fw)` coords + their expected keep/drop verdicts
(`data/atlas/guard_tripwire.json`, ~16 KB, tracked) and regresses them by rendering
each field through the real path every run.

Per-outcome verdict match (not just "20 drop") localizes a flip and cannot be fooled
by two offsetting flips. The `0.25` / `6.0` thresholds are FIXED — this is a
tripwire, not tuning.

Only the RENDER pass is marked slow (81 tiles, ~46s) — deliberately not the module, so
the cheap fixture-integrity check below stays in the default lane. A corrupt fixture is
exactly what would make the render pass vacuous, and it costs nothing to keep checking.
The fast `test_guard.py` gate stays pure/GPU-free. Run the render pass explicitly:

  uv run pytest -m slow                             # the whole opt-in lane
  uv run pytest tools/atlas/test_guard_tripwire.py -m slow

The trailing `-m slow` is not optional: `addopts` in pyproject filters `-m "not slow"`
by default, and naming the file on the command line does NOT override a marker filter —
without it the path resolves and then deselects to zero tests, which reads as a pass.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import guard  # noqa: E402

FIXTURE = ROOT / "data" / "atlas" / "guard_tripwire.json"
FIELD_DIR = ROOT / "scratch" / "atlas" / "guard_tripwire" / "fields"   # ephemeral render scratch
WORKERS = 4   # project hard cap: never exceed 4 concurrent workers. Measured optimal:
              # 4x(default threads) = 46.5s, vs 50.0s at 4x3, 64.6s at 4x2, 51.6s at
              # 2x6, 53.3s at 1x12. Do not "tune" the thread count — it is already there.


def _load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _verdict_of(cx, cy, fw, oid, family):
    """Render the field through the live f64 guard path + apply the pinned gates."""
    st = guard.measure_location(cx, cy, fw, FIELD_DIR / f"{oid}.bin", family=family)
    reason = guard.guard_fail(st.interior_frac, st.field_std)
    return (oid, "keep" if reason is None else reason, st.interior_frac, st.field_std)


def test_fixture_is_the_canonical_81_20_set():
    """Cheap guard on the pinned fixture itself (no render): 81 outcomes, 20 drops,
    per-gate 13/9/2, thresholds unchanged. Catches fixture corruption before the
    expensive render pass runs."""
    fx = _load_fixture()
    assert fx["thresholds"] == {"interior_cap": guard.INTERIOR_CAP,
                                "field_std_floor": guard.FIELD_STD_FLOOR}
    outs = fx["outcomes"]
    assert len(outs) == 81
    from collections import Counter
    vc = Counter(o["verdict"] for o in outs)
    assert vc["keep"] == 61
    assert vc["interior"] == 11 and vc["flat"] == 7 and vc["both"] == 2
    drops = vc["interior"] + vc["flat"] + vc["both"]
    assert drops == 20
    # per-gate attribution the drop manifest documented (interior_gate 13, flat 9, both 2).
    assert vc["interior"] + vc["both"] == 13
    assert vc["flat"] + vc["both"] == 9
    assert all(o["family"] == "mandelbrot" for o in outs)


@pytest.mark.slow
def test_live_f64_path_reproduces_every_verdict():
    """Render all 81 fields through the real `render-one --dump-field-source f64` path
    at GUARD_STAT_RES, measure via the real `guard.py`, and assert each per-outcome
    verdict matches the pinned fixture. A single flip fails with a localized diff."""
    if not Path(guard.BIN).exists():
        pytest.skip(f"release binary not built: {guard.BIN}")
    fx = _load_fixture()
    outs = fx["outcomes"]
    FIELD_DIR.mkdir(parents=True, exist_ok=True)

    got = {}
    stats = {}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_verdict_of, o["cx"], o["cy"], o["fw"], o["id"], o["family"])
                for o in outs]
        for f in cf.as_completed(futs):
            oid, verdict, ifrac, fstd = f.result()
            got[oid] = verdict
            stats[oid] = (ifrac, fstd)

    mism = []
    for o in outs:
        exp, g = o["verdict"], got[o["id"]]
        if exp != g:
            ifrac, fstd = stats[o["id"]]
            mism.append(f"{o['id'][-6:]} exp={exp} got={g} "
                        f"(if={ifrac:.4f} fs={fstd:.3f})")
    assert not mism, "live f64 guard path flipped verdicts:\n  " + "\n  ".join(mism)
