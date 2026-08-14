"""Render-mode candidate RENDERING — the roster, the recipes, and the three render paths.

WHAT THIS IS NOW (2026-08-12). The deploy-tail DRIVER was retired on this date
(prompts/deploy_tail_recon.md, Matt): `main()`, the `--ephemeral` lane, the alternates state
and the diversity allocation behind it (`tail_alloc.py`) are gone. What is left is the LIBRARY
HALF — six live importers take the roster, the canonical inherited coloring and the render
dispatch from here, so the module stays exactly as `emit_v1` stayed when its own driver went:
delete the dead `main()`, keep what has readers.

WHY THE DRIVER WENT, in one paragraph, so this is not rediscovered. It curated the ACCUMULATED
emission corpus of `emit_v1`: for each already-emitted smooth wallpaper it rendered the promoted
strange modes over that location's INHERITED approved palette, scored them through the locked
mining gate, and kept a diversity-allocated ~25% as alternate wallpapers alongside the smooth
ones. Two facts ended it. Its input had no producer — `emit_v1.main` was deleted 2026-08-11 and
`scratch/wallpaper/emit_v1/manifest.jsonl` has been unwritable since. And the run path already
MAKES its product: `attempt_budget.plan` walks each partition's rank-ordered supply from index 0
ONCE PER HEAD, so the same top location is planned for a smooth attempt and a strange attempt in
one run, and both are released at the same 2560x1440 ss4 canon. The one capability that went
with it, recorded rather than left to be found: this pass varied ONLY the mode over a palette a
human had already approved (a controlled A/B on a known-good coloring), where the run path picks
(flavor, style) jointly by deficit and never re-uses a released row's palette.

THE COST THE RETIREMENT BOOKED. `alloc_input = [c for c in cands if scorer.gate(...)]` lived
here and was the ONLY caller of `mining_gate.MiningScorer.gate` in the tree. With it gone,
`MINING_GATE_THRESHOLD` (0.0949) has no acting site anywhere: `floors.MINING_RELEASE` is a
`Floor` and cannot remove a row, `floors.MINING_POOL` is 0.0, and `selection.rank_select` holds
no floor. The gate is ANNOTATION EVERYWHERE as of this date. The 0.0949 lock and its
superseded-by chain (`data/render_mode_head/v3/mining_gate_lock_2026-08-11.*`) are untouched as
provenance — a frozen measurement record keeps what was true when it was written.

WHAT IS STILL HERE, and who reads it:
  * `ROSTER` / `load_promoted_roster` — the candidate roster DERIVED from the mode registry
    (`specs/modes_registry.json`, `tier == "promoted"`), minus `smooth` (the base carrier).
    Promote or demote a mode there and every reader follows with no edit here. A promoted mode
    with no render recipe below is RETURNED as `unmapped` and skipped, never guessed.
    Read by `emission/build_emission_diversity_v1.render_styles` (the live emission driver).
  * `_color_params` — THE canonical inherited coloring (transfer=pct, gamma 1, no
    reverse/phase/cycles at its `{}` default). Four sheet builders call it for exactly that:
    it is the recipe that applies to a palette nothing has ever fitted a head to.
  * `render_candidate` / `render_pure` / `render_rust` — the three MODE RENDER PATHS, which
    differ and must (== `render_mode_pilot/render_scale_batch.py`, the dataset_v1 recipe the
    mining head learned):
      - pure  (tia, stripe): render-one --dump-field <mode field> -> colormap.render_candidate
        with the FULL inherited param set. Bit-faithful. The field dump is keyed with the
        render-mode token (`loc_mod.field_mode_token`) so it can NEVER collide with the cached
        smooth field.
      - composite (C13, C17): render-one --coloring @spec --palette (Rust, grad-less).
      - direct  (direct_trap_multiply, direct_trap_screen): render-one --coloring @spec,
        palette-indifferent (ONE candidate per direct-trap mode, no palette axis).
        direct_trap_screen at its sweet spot (opacity 0.15 / threshold 0.08) + the dataset_v1
        soft_knee@0.35 highlight rolloff.
    `normal_map` is OFF for all modes (no spec enables it; `shade:none` composites).
  * The BAND AUTO-LEVEL seam (`_level_python`, `_info`, and `level=(kind != "direct")`). The
    operator is reached ONLY through `autolevel.maybe_level` — one switch, not two — and the
    direct-trap family is excluded where the KIND is known rather than by a flag inside the
    render, because a palette-indifferent mode has no LUT for it to act on. `stamp_log=False`
    suppresses only the STAMP WRITE (never the levelling), for a render that runs in a worker
    process whose parent writes the log — see `emission/release_pass.py`.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "corpus"))
sys.path.insert(0, str(REPO / "tools" / "queries"))

import colormap as cm                     # noqa: E402
import location as loc_mod                 # noqa: E402
import query_sampler as qs                 # noqa: E402
from tools.palettes import autolevel as AL  # noqa: E402  THE band auto-level (switch: ON, 2026-08-11)

EXE = str(REPO / "target/release/fractal-generator.exe")
POOL_CMAPS = str(REPO / "data/palettes/pool_colormaps.json")
MODES_REGISTRY = REPO / "specs" / "modes_registry.json"   # SOURCE OF TRUTH for mode promotion

OUT_DIR = REPO / "scratch/mining/deploy_tail"   # DISPOSABLE scratch (this module's own tree)
FIELD_TMP = OUT_DIR / "_fields"             # disposable field dumps (module-owned, token'd)

JPG_Q = 95

# ---- candidate roster (DERIVED from the mode registry, not hardcoded) ------ #
# kind: pure -> dump field + colormap tail; composite/direct -> Rust --coloring.
# The recipe tables below are the dataset_v1 render recipes (== render_scale_batch.py, the
# recipe the mining_v1 head learned); the ACTIVE roster is the registry-PROMOTED subset of
# them (load_promoted_roster). Recipes are kept even for non-promoted modes so a future
# promotion in modes_registry.json needs no code change here.
PURE_FIELD_SPEC = {
    "tia": {"field": "tia", "skip": 1},
    "stripe": {"field": "stripe", "stripe_density": 6},
}
SPEC_FILE = {
    "smooth_mean_angle": "smooth_mean_angle",
    "smooth_angle_min": "smooth_angle_min",
    "composite_c7_smooth_trap_circle": "composite_c7_smooth_trap_circle",
    "composite_c13_smooth_stripe": "composite_c13_smooth_stripe",
    "composite_c17_smooth_curvature": "composite_c17_smooth_curvature",
    "direct_trap_multiply": "direct_trap_multiply",
    "direct_trap_screen": "direct_trap_screen",
}
# highlight rolloff / mode params: INERT unless the keyed mode is promoted into the roster
# (the direct_trap family is currently `niche`); kept == dataset_v1 for recipe fidelity so a
# future promotion is a no-code-change flip. direct_trap sweet spot: opacity 0.15 / thr 0.08.
ROLLOFF = {"direct_trap_screen": ("soft_knee", 0.35)}
MODE_PARAMS = {"direct_trap_screen": {"direct_threshold": 0.08, "direct_opacity": 0.15}}


def load_promoted_roster():
    """Candidate roster = the modes promoted in the SOURCE OF TRUTH (modes_registry.json,
    `tier == "promoted"`), NOT a hardcoded list. `smooth` is the base carrier the emission path
    already ships, so it is excluded (a strange candidate that re-rendered smooth would dupe
    it). Each promoted mode maps to its dataset_v1 render recipe (pure field vs composite/direct
    spec); a promoted mode with NO recipe here is RETURNED as unmapped and skipped, never
    guessed. Returns (roster, unmapped)."""
    reg = json.loads(MODES_REGISTRY.read_text(encoding="utf-8"))
    promoted = [e["spec"] for e in reg if e.get("tier") == "promoted" and e["spec"] != "smooth"]
    roster, unmapped = [], []
    for m in promoted:
        if m in PURE_FIELD_SPEC:
            roster.append((m, "pure"))
        elif m in SPEC_FILE:
            roster.append((m, "direct" if m.startswith("direct_trap") else "composite"))
        else:
            unmapped.append(m)
    return roster, unmapped


# (mode, kind), display/report order = registry order. Derived at import; the registry is the
# only knob — promote/demote a mode there and this roster follows with no edit here.
ROSTER, UNMAPPED_PROMOTED = load_promoted_roster()

_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = qs.load_pool_library()
    return _LIB


def _locflags(loc):
    return loc_mod.render_one_flags(loc) + ["--cx", loc.cx, "--cy", loc.cy,
                                            "--fw", loc.fw, "--maxiter", str(loc.maxiter)]


def _run(cmd, retries=1):
    """render-one shell-out with one retry (renders occasionally fail transiently
    under resource contention; the recipe is deterministic so a retry recovers).

    Launch defaults come from `corpus_common` (`DEFAULT_ENGINE_THREADS` at BELOW_NORMAL, the
    committed pairing for ONE `fractal-generator.exe`), never restated here — this launcher
    used to inherit whatever priority its parent happened to have, so an interactive session
    driving a long emission run competed with the desktop for no reason. One engine at a time
    is what this path does, so the per-process 7 is the right number."""
    import corpus_common as _cc                # noqa: PLC0415  (tools/corpus already on path)
    for attempt in range(retries + 1):
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                           env=_cc.default_engine_env(),
                           creationflags=_cc.default_creationflags())
        if r.returncode == 0:
            return
        if attempt == retries:
            raise RuntimeError(" ".join(map(str, cmd[:3])) + " ... :\n" + r.stderr[-800:])


def _color_params(emit_params: dict) -> dict:
    """THE canonical inherited coloring, defaulted to what the emission path emits
    (transfer=pct). `_color_params({})` is the recipe four sheet builders reach for."""
    return {
        "reverse": bool(emit_params.get("reverse", False)),
        "log_premap": emit_params.get("log_premap", "none"),
        "gamma": float(emit_params.get("gamma", 1.0)),
        "phase": float(emit_params.get("phase", 0.0)),
        "n_cycles": int(emit_params.get("n_cycles", 1)),
        "transfer": emit_params.get("transfer", "pct"),
        "transfer_gamma": float(emit_params.get("transfer_gamma", 0.0)),
    }


def _field_stem(loc, mode, w, h, ss, maxiter_policy=None):
    """Field-dump stem carrying the render-mode token (loc_mod.field_mode_token) so a
    strange pure-field dump can never collide with the cached SMOOTH field, and the
    iteration-CAP policy token (loc_mod.maxiter_policy_token) so a field dumped under
    the old cap can never be served under the new one (docs/design/auto_maxiter.md).
    Both are empty for the legacy value, so existing stems are byte-identical."""
    tok = loc_mod.field_mode_token(mode)
    suffix = f"|{tok}" if tok else ""
    ptok = loc_mod.maxiter_policy_token(maxiter_policy)
    suffix += f"|{ptok}" if ptok else ""
    h16 = hashlib.sha1(f"{loc.key()}|{w}x{h}ss{ss}|{loc.maxiter}{suffix}".encode()).hexdigest()[:16]
    return f"{loc.family}_{h16}_{w}x{h}ss{ss}__{mode}"


# --------------------------------------------------------------------------- #
# Render one candidate at (w,h,ss,filt) -> out_path (jpg for scoring, png for a keeper).
# --------------------------------------------------------------------------- #
def field_tmp_token() -> str:
    """Per-PROCESS token appended to a disposable field dump's name.

    The dump is written and unlinked inside one call, so within a process the name only has to
    be unique against itself. Across processes it does not: two concurrent release workers
    rendering the same (location, mode, geometry) — the same location released under two
    palettes — would otherwise write and `finally`-unlink ONE file, and the loser reads a
    truncated or already-deleted field. `render_rust` solved this for its own temps by keying
    on the output stem; the field stem cannot take that (its whole job is to identify the
    FIELD), so the process id rides alongside it instead. Nothing caches at these names."""
    return f"p{os.getpid()}"


def render_pure(loc, mode, palette, cp, out_path, w, h, ss, filt, *, stamp_log=True):
    spec = dict(PURE_FIELD_SPEC[mode])
    FIELD_TMP.mkdir(parents=True, exist_ok=True)
    binp = FIELD_TMP / f"{_field_stem(loc, mode, w, h, ss)}__{field_tmp_token()}.bin"
    lev = None
    try:
        _run([EXE, "render-one"] + _locflags(loc) + ["--width", str(w), "--height", str(h),
             "--supersample", str(ss), "--coloring", json.dumps(spec), "--dump-field", str(binp)])
        fld = cm.load_field(str(binp))
        ow, oh = fld.out_size
        ptype = lib().palette_type(palette)
        phase = cp["phase"] if ptype == "cyclic" else 0.0
        ncyc = cp["n_cycles"] if ptype == "cyclic" else 1
        cfg = cm.CandidateConfig(palette=palette, location=fld.location, eval_width=ow,
            eval_height=oh, reverse=cp["reverse"], log_premap=cp["log_premap"],
            gamma=cp["gamma"], phase=phase, n_cycles=ncyc, transfer=cp["transfer"],
            transfer_gamma=cp["transfer_gamma"], filter=filt)
        prep = cm.stretch_field(fld)
        prof = cm.gradient_transfer_profile(fld, prep) if cp["transfer"] == "grad" else None
        img = cm.render_candidate(fld, cfg, lib(), prep=prep, profile=prof)
        # BAND AUTO-LEVEL (switch default OFF). The re-render is another LUT over the SAME
        # cached field — no re-iteration, no second engine call — and it only happens when
        # the curve actually acts; an in-band render comes back as its own bytes.
        lev = _level_python(img, palette, out_path,
                            lambda ovr: cm.render_candidate(fld, cfg, ovr, prep=prep,
                                                            profile=prof),
                            stamp_log=stamp_log)
        _save(lev.img, out_path)
    finally:
        binp.unlink(missing_ok=True)
        binp.with_suffix(".json").unlink(missing_ok=True)
    return _info({"transfer_dropped": False}, lev)


def _level_python(img, palette, out_path, recolor, *, stamp_log=True):
    """The band auto-level over the PYTHON coloring tail: the leveled stops go through
    `autolevel.OverrideLibrary`, which bakes with `colormap.build_lut` — the same bake, the
    same mirror flag — so the Rust<->Python LUT seam is untouched and only the stop COLOURS
    differ. `recolor(library) -> image` is the call site's own recolor, given a library.

    `stamp_log=False` suppresses ONLY the stamp write, never the levelling: the stamp still
    comes back on the returned `Leveled` and in this render's info block, so a caller that
    renders in a worker process can have its PARENT write the row (`autolevel.append_stamp`).
    That is the whole reason the flag exists — a shared append-only log with N writers has no
    order, and the release record's stamp file must be identical to the serial one."""
    entry = lib().colormaps[palette]
    mirror = bool(entry.get("mirror_needed"))
    out_path = Path(out_path)
    return AL.maybe_level(
        img, entry,
        lambda stops: recolor(AL.OverrideLibrary(lib(), palette, stops, mirror)),
        key=out_path.name, log_dir=out_path.parent if stamp_log else None)


def _info(info: dict, lev) -> dict:
    """Attach the auto-level stamp to a render's info block — and ONLY when there is one.

    Presence of `autolevel` is exactly "this render was produced with the operator on" — and
    since the operator returns the base render's own bytes on an in-band image, an ON row with
    `acted: false` is a row the band accepted unchanged. With the switch OFF (the contract the
    flip kept, `FRACTAL_AUTOLEVEL=0`) there is no stamp and no key at all, so such a record is
    byte-identical to what this path wrote before the operator existed."""
    if lev is not None and lev.stamp is not None:
        info = dict(info, autolevel=lev.stamp)
    return info


def render_rust(loc, mode, palette, cp, out_path, w, h, ss, filt, *, level=True,
                stamp_log=True):
    """`level=False` is the DIRECT-trap family: palette-indifferent by construction (one
    candidate per mode, no palette axis), so there is no LUT for the operator to act on and
    the auto-level is unreachable there rather than merely disabled."""
    spec = json.loads((REPO / "specs" / f"{SPEC_FILE[mode]}.json").read_text())
    spec.pop("tier", None)
    spec.pop("note", None)
    spec.update(MODE_PARAMS.get(mode, {}))
    ptype = lib().palette_type(palette)
    spec["transform"] = "log" if cp["log_premap"] == "log" else "linear"
    spec["gamma"] = cp["gamma"]
    spec["reverse"] = cp["reverse"]
    if ptype == "cyclic":
        spec["palette_cycles"] = float(cp["n_cycles"])
        spec["palette_offset"] = float(cp["phase"])
    rolloff = ROLLOFF.get(mode, ("none", 1.0))
    if rolloff[0] != "none":
        spec["rolloff"] = rolloff[0]
        spec["rolloff_strength"] = rolloff[1]
    transfer_dropped = cp["transfer"] == "grad"
    FIELD_TMP.mkdir(parents=True, exist_ok=True)
    # Keyed on the OUTPUT stem, not just (family, mode, geometry): the auto-level's second
    # engine pass reuses this temp, and two renders of the same location at the same geometry
    # (before/after arms in a verification sheet, two concurrent workers) would otherwise
    # write and delete one file.
    stem = f"{Path(out_path).stem}__{loc.family}_{mode}_{w}x{h}"
    tmp_png = FIELD_TMP / f"{stem}.png"
    tmp_cmaps = FIELD_TMP / f"{stem}__autolevel.json"
    lev = None

    def _engine(cmaps_path):
        _run([EXE, "render-one"] + _locflags(loc) + ["--width", str(w), "--height", str(h),
             "--supersample", str(ss), "--filter", filt, "--palette", palette,
             "--colormaps", str(cmaps_path), "--coloring", json.dumps(spec),
             "--out", str(tmp_png)])
        with Image.open(tmp_png) as im:
            return np.asarray(im.convert("RGB"))

    try:
        img = _engine(POOL_CMAPS)
        if level:
            # BAND AUTO-LEVEL over the RUST path (switch default OFF). The surgery stays
            # Python-side and ends in a one-entry colormap JSON under the SAME palette name,
            # so the engine's bake, mirror flag and spec are bit-identical to the call above
            # and only the stop colours differ. A curve that acts costs a SECOND engine
            # render; an in-band one costs nothing.
            entry = lib().colormaps[palette]
            lev = AL.maybe_level(
                img, entry,
                lambda stops: _engine(AL.one_entry_colormaps(entry, stops, tmp_cmaps)),
                key=Path(out_path).name,
                log_dir=Path(out_path).parent if stamp_log else None)
            img = lev.img
        _save(img, out_path)
    finally:
        tmp_png.unlink(missing_ok=True)
        tmp_cmaps.unlink(missing_ok=True)
    return _info({"transfer_dropped": transfer_dropped, "rolloff": rolloff, "spec": spec}, lev)


def _save(img_arr, out_path):
    im = Image.fromarray(np.asarray(img_arr))
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        im.save(out_path, quality=JPG_Q)
    else:
        im.save(out_path)


def render_candidate(loc, mode, kind, palette, cp, out_path, w, h, ss, filt, *, stamp_log=True):
    if kind == "pure":
        return render_pure(loc, mode, palette, cp, out_path, w, h, ss, filt,
                           stamp_log=stamp_log)
    # The auto-level is a map on the PALETTE, so it reaches every palette-mapped kind and
    # exactly none of the direct-trap family (palette-indifferent, `kind == "direct"`).
    return render_rust(loc, mode, palette, cp, out_path, w, h, ss, filt,
                       level=(kind != "direct"), stamp_log=stamp_log)
