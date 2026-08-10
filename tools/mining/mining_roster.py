r"""mining_roster.py — THE strange-mode roster and its per-mode render recipe, once.

The mining head judges a render's MODE/palette/colour choice at a fixed location, so the
roster is not a detail of one builder — it is the corpus's class vocabulary. It lived in
four places (`build_sample.MODES`, `build_scale_sample.MODES`, `render_batch.PURE_FIELD_SPEC
+ SPEC_FILE`, `render_scale_batch.PURE_FIELD_SPEC + SPEC_FILE`) in two dialects, and the two
dialects had already diverged: the samplers carried 15 and 13 modes respectively while the
renderers carried 15 and 13 recipes, and nothing tied a mode to the recipe that renders it.
A mode present in one and absent in the other is a silent hole in a corpus, so both halves
are here and `test_mining_sheet.py` fails on a roster entry with no recipe.

THE ROSTER IS 15 — the pilot's full set. `build_scale_sample` dropped `trap_circle` (dead
solo) and `exp_smoothing` (near-smooth) at SAMPLE time, and `train_mining_head` then dropped
`trap_circle`, `exp_smoothing` and `direct_trap_screen` again at TRAIN time. Those are two
different decisions and only the second one belongs downstream: a corpus that drops a mode
cannot be used to re-examine the drop, while a trainer that drops one can be re-run. So the
corpus carries every registered non-`smooth` mode and the trainer keeps its own drop list.
(Decision 2026-08-06, Matt; the prompt's "keep the full mode roster in the corpus — mode
drops were trainer-side decisions and stay there".)

THREE RENDER KINDS, and the kind decides colour fidelity:

  pure      `render-one --dump-field <field>` -> `colormap.render_candidate` with the FULL
            approved colour params. Bit-faithful, `transfer=grad` included.
  composite `render-one --coloring <specs/*.json>`. The Rust coloring path cannot express
  direct   `transfer=grad`, so that one knob is DROPPED and the row stamps
            `transfer_dropped: true` rather than pretending it was honoured.

`direct_*` additionally sweeps a 3x3 opacity x threshold grid (the thresholds span the
measured p75..p95 across trap shapes), and the family is palette-INDIFFERENT by construction
— which is why the samplers dedupe it to one palette per location and spread the grid
instead.

ROLLOFF is gated to `direct_trap_screen` alone (soft_knee @ 0.35, the screen-family blowout
recovery). Every other mode renders rolloff-off, and `Rolloff::None` is exact identity, so
those renders are byte-identical to the pre-rolloff path.

    from tools.mining.mining_roster import ROSTER, MODES, MODE_KIND, spec_for, DIRECT_GRID
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPECS_DIR = ROOT / "specs"

# --- the direct-trap sweep --------------------------------------------------- #
DIRECT_OPACITY = (0.15, 0.30, 0.45)
DIRECT_THRESHOLD = (0.05, 0.08, 0.12)          # spans measured p75..p95 across shapes
DIRECT_GRID = tuple((op, th) for op in DIRECT_OPACITY for th in DIRECT_THRESHOLD)  # 9 cells

# --- the SMOOTH baseline ------------------------------------------------------- #
# `smooth` is the base carrier every composite builds on (`specs/smooth.json`,
# tier=promoted) and is deliberately NOT on the roster: the corpus's class vocabulary is the
# 15 STRANGE modes, and a corpus that quietly held smooth rows under the same mode axis would
# make "the head judges the mode choice" false.
#
# It is nameable here anyway, because two live callers need to render the smooth twin of a
# strange row through the SAME path and at the SAME geometry:
#   * the smooth-EQUIVALENCE measure (`smooth_equivalence.py`) — a mode render that is
#     indistinguishable from its own location's smooth render is a duplicate of something
#     Matt has already judged, and the distance is only meaningful if the only difference
#     between the two frames is the mode;
#   * a sheet's small smooth-for-comparison slice.
# `spec_for(SMOOTH_MODE)` therefore answers a real field spec, `SMOOTH_MODE not in MODES`
# stays true, and `missing_recipes()` is untouched (it walks MODES and the two recipe tables).
SMOOTH_MODE = "smooth"
SMOOTH_KIND = "pure"
SMOOTH_FIELD_SPEC = {"field": "smooth", "transform": "linear"}

# --- pure modes: the `--coloring` field spec handed to `--dump-field` ---------- #
PURE_FIELD_SPEC = {
    "tia": {"field": "tia", "skip": 1},
    "stripe": {"field": "stripe", "stripe_density": 6},
    "exp_smoothing": {"field": "exp_smoothing"},
    "gaussian_int": {"field": "gaussian_int"},
    "trap_circle": {"field": "trap_circle"},
    "curv_linear": {"field": "curvature"},
}

# --- composite + direct modes: the committed spec file under specs/ ------------ #
SPEC_FILE = {
    "smooth_mean_angle": "smooth_mean_angle",
    "smooth_angle_min": "smooth_angle_min",
    "composite_c7_smooth_trap_circle": "composite_c7_smooth_trap_circle",
    "composite_c13_smooth_stripe": "composite_c13_smooth_stripe",
    "composite_c17_smooth_curvature": "composite_c17_smooth_curvature",
    "direct_trap_ring": "direct_trap_ring",
    "direct_trap_screen": "direct_trap_screen",
    "direct_trap_multiply": "direct_trap_multiply",
    "direct_trap_lines": "direct_trap_lines",
}

# The roster, in a fixed order (mode, kind). Order is load-bearing only for reporting —
# every draw sorts explicitly — but it is frozen so two runs print the same table.
ROSTER = (
    # pure-field (dump-field + python recolor; transfer=grad faithful)
    ("tia", "pure"),
    ("stripe", "pure"),
    ("exp_smoothing", "pure"),
    ("gaussian_int", "pure"),
    ("trap_circle", "pure"),
    ("curv_linear", "pure"),
    # composite (Rust --coloring)
    ("smooth_mean_angle", "composite"),
    ("smooth_angle_min", "composite"),
    ("composite_c7_smooth_trap_circle", "composite"),
    ("composite_c13_smooth_stripe", "composite"),
    ("composite_c17_smooth_curvature", "composite"),
    # direct-trap family (Rust --coloring; opacity x threshold sweep, palette-deduped)
    ("direct_trap_ring", "direct"),
    ("direct_trap_screen", "direct"),
    ("direct_trap_multiply", "direct"),
    ("direct_trap_lines", "direct"),
)
MODES = tuple(m for m, _k in ROSTER)
MODE_KIND = {m: k for m, k in ROSTER}
DIRECT_MODES = tuple(m for m, k in ROSTER if k == "direct")

# Modes the sampler dropped in July (`build_scale_sample.DROPPED`) and the modes the trainer
# dropped (`train_mining_head`). Recorded as DATA so the difference between "not in this
# corpus" and "not in that training run" is inspectable rather than remembered.
SCALE_SAMPLER_DROPPED_2026_07 = ("trap_circle", "exp_smoothing")
TRAINER_DROPPED_V1 = ("trap_circle", "exp_smoothing", "direct_trap_screen")

# The one adopted highlight rolloff, gated to the screen family's blowout.
ROLLOFF = {"direct_trap_screen": ("soft_knee", 0.35)}


def kind_of(mode: str) -> str:
    """The render KIND of a mode, including the off-roster smooth baseline. Raises on an
    unknown mode for the same reason `spec_for` does."""
    if mode == SMOOTH_MODE:
        return SMOOTH_KIND
    kind = MODE_KIND.get(mode)
    if kind is None:
        raise KeyError(f"{mode!r} is not on the mining roster ({len(MODES)} modes): {MODES}")
    return kind


def rolloff_for(mode: str) -> tuple:
    """`(name, strength)`; `("none", 1.0)` for every mode but `direct_trap_screen`."""
    return ROLLOFF.get(mode, ("none", 1.0))


def rolloff_token(mode: str) -> str:
    """The stamped string form — `"none"` or `"soft_knee@0.35"`."""
    name, strength = rolloff_for(mode)
    return "none" if name == "none" else f"{name}@{strength}"


def spec_for(mode: str, mode_params: dict | None = None) -> dict:
    """The mode's canonical coloring spec, BEFORE the row's colour params are layered on.

    `pure` -> the field-dump spec; `composite`/`direct` -> the committed `specs/<file>.json`
    with `tier` stripped (a curation label, not a render knob) and `mode_params` applied.
    Raises on an unknown mode rather than returning a default: a typo'd mode name that
    silently rendered `smooth` would put a mislabeled class in the corpus."""
    import json

    if mode == SMOOTH_MODE:
        return dict(SMOOTH_FIELD_SPEC)
    kind = MODE_KIND.get(mode)
    if kind is None:
        raise KeyError(f"{mode!r} is not on the mining roster ({len(MODES)} modes): {MODES}")
    if kind == "pure":
        return dict(PURE_FIELD_SPEC[mode])
    spec = json.loads((SPECS_DIR / f"{SPEC_FILE[mode]}.json").read_text(encoding="utf-8"))
    spec.pop("tier", None)
    spec.update(mode_params or {})
    return spec


def missing_recipes() -> list:
    """Roster modes with no render recipe, and recipes with no roster entry — BOTH ways.

    A one-way check passes on the failure that matters (a mode added to the roster and not
    to the recipe table renders nothing), so the guard walks both sets."""
    have = set(PURE_FIELD_SPEC) | set(SPEC_FILE)
    out = [f"roster mode with no recipe: {m}" for m in MODES if m not in have]
    out += [f"recipe with no roster entry: {m}" for m in sorted(have - set(MODES))]
    out += [f"spec file absent: specs/{SPEC_FILE[m]}.json" for m in SPEC_FILE
            if not (SPECS_DIR / f"{SPEC_FILE[m]}.json").exists()]
    return out
