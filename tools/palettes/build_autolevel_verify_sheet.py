r"""build_autolevel_verify_sheet.py — the band auto-level, through the PRODUCTION path.

WHAT THIS IS FOR. This sheet is the evidence the 2026-08-11 flip was adopted on, and it stays
the way the wiring is re-proved: the unit tests prove the rule; this proves the WIRING — that
the operator does what the study
measured when it runs inside the real render functions (`deploy_tail.render_candidate` for
the strange modes, `build_emission_diversity_v1.render_smooth` for the base carrier) rather
than inside a study driver that merely resembles them.

THE THREE CLAIMS IT CHECKS, at production fidelity and on production code paths:
  1. IDENTITY IS FREE. An in-band render comes back as the base render's own bytes — max |Δ|
     is exactly 0, not "small". The operator short-circuits before the re-render, so this is
     structural; the sheet is where it is confirmed against real renders.
  2. THE STAMP REPLAYS. For acting examples, the stop list is rebuilt from the stamp ALONE
     (`autolevel.stops_from_stamp`) and re-rendered; the bytes must match the band arm.
     Bounded by `SheetSpec.replay_n` (a render each) and the bound is REPORTED, never silent.
  3. THE POPULATION SPANS WHAT WAS ASKED FOR — smooth + the two modes the v2 sheets were
     scored on, including at least one in-band identity and one white-below-band correction.
     Coverage is checked and reported; it is not assumed.

SINK ISOLATION. This is a throwaway run and it asserts that before its first write: the
emission record sinks are resolved under `scratch/` via `emission_sinks`, so nothing here can
reach `data/emission/`. It writes no durable artifact at all — the sheet, the arms and the
verdict all land in `scratch/autolevel_verify/`.

THE SWITCH is set PER JOB inside the worker (`FRACTAL_AUTOLEVEL`), never by editing the
default: the `before` arm runs the same code with the operator off, which is what makes the
two arms a controlled comparison rather than two programs.

    uv run python -u tools/palettes/build_autolevel_verify_sheet.py --limit 2   # bounded e2e
    uv run python -u tools/palettes/build_autolevel_verify_sheet.py            # the sheet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "queries"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.palettes import autolevel as AL                        # noqa: E402


# =========================================================================== #
# 1. The sheet instance (one today; a second is an entry, not a refactor).
# =========================================================================== #
@dataclass(frozen=True)
class SheetSpec:
    name: str
    out_dir: Path
    # the two v2-sheet modes + the base carrier. `kind` is deploy_tail's render dispatch.
    strange_modes: tuple = ("composite_c7_smooth_trap_circle", "smooth_angle_min")
    n_per_strange: int = 20           # pre-screened candidates per strange mode
    n_smooth: int = 10                # pre-screened smooth candidates
    n_sheet: int = 12                 # rows on the sheet (both arms rendered for these only)
    geom: tuple = (1280, 720, 2)      # the mining head's label geometry
    filt: str = "lanczos3"
    workers: int = 3                  # concurrent engine PROCESSES (<= 4, CLAUDE.md)
    engine_threads: int = 4
    replay_n: int = 3                 # acting examples re-rendered from their stamp alone
    arms: tuple = ("before", "band")
    must_cover: tuple = ("identity", "white_below")


SPECS = {
    "stage_v1": SheetSpec(name="stage_v1", out_dir=ROOT / "scratch" / "autolevel_verify"),
}


# =========================================================================== #
# 2. Population — the STUDY's own selection rules, imported rather than re-picked.
#    The study is where "which examples price this lever" was decided; copying that
#    here would be a second population with the same name.
# =========================================================================== #
def population(sspec: SheetSpec) -> list:
    from tools.studies import palette_autolevel as PA
    from tools.studies import palette_autolevel_band as PB

    out = []
    v2 = PA.SPECS["exposure_verify_2modes_v2"]
    pop = PA.select_population(v2)
    for mode in sspec.strange_modes:
        rows = [p for p in pop if p["mode"] == mode][:sspec.n_per_strange]
        for r in rows:
            out.append({"example_id": r["example_id"], "mode": mode, "kind": "composite",
                        "label": r["label"], "fractal_type": r.get("fractal_type"),
                        "palette": r["palette"], "render": r["render"],
                        "color_params": r["color_params"],
                        "corpus_crop": r.get("corpus_crop")})
    band = PB.SPECS["band_v1"]
    for r in PB._smooth_population(band)[:sspec.n_smooth]:
        out.append({"example_id": r["example_id"], "mode": "smooth", "kind": "smooth",
                    "label": r["label"], "fractal_type": r.get("fractal_type"),
                    "palette": r["palette"], "render": r["render"],
                    "color_params": r["color_params"],
                    "corpus_crop": r.get("corpus_crop")})
    return out


def prescreen(pop: list) -> list:
    """Classify every candidate off its COMMITTED CORPUS CROP — same recipe, same geometry,
    already on disk — so the sheet's coverage is CHOSEN rather than hoped for.

    This is selection only. The class that goes in the verdict is measured on the production
    render below; the pre-screen's job is to stop a 12-row sheet from being 12 identities
    because the required corrections happened to sit at position 13. Both are reported, so a
    disagreement between crop and production render is visible rather than absorbed."""
    ref = AL.load_reference()
    band = AL.bands(ref)
    for ex in pop:
        p = ROOT / ex["corpus_crop"] if ex.get("corpus_crop") else None
        if p is None or not p.exists():
            ex["prescreen"] = "no_crop"
            continue
        st = AL.tone_stats(np.asarray(Image.open(p).convert("RGB")))
        ex["prescreen"] = classify(AL.make_stamp(ref, AL.derive_band_curve(st, band), st,
                                                 n_capped=0, n_stops=0, acted=False))
    return pop


# =========================================================================== #
# 3. One arm, through the PRODUCTION render entry points.
# =========================================================================== #
def _one_arm(ex, arm, sd, out_path, stops, mods) -> dict:
    """One production render. The switch is set per ARM, so `before` and `band` are the same
    production code with one boolean different.

    `stops` is the REPLAY arm: the stop list rebuilt from the stamp is pushed straight into
    the render through a one-entry colormap library/file, with the switch OFF — a replay has
    to reproduce the render from the RECORD, not by re-deriving the curve from the image."""
    loc_mod, cm, dt, BED = mods
    os.environ[AL.SWITCH_ENV] = "1" if arm == "band" else "0"
    w, h, ss = sd["geom"]
    filt = sd["filt"]
    loc = loc_mod.from_render_block(ex["render"])
    cp = dt._color_params(ex["color_params"])
    palette = ex["palette"]
    lib = dt.lib()
    entry = lib.colormaps[palette]
    t0 = time.time()

    if stops is not None and ex["kind"] == "smooth":
        replay_lib = AL.OverrideLibrary(lib, palette, stops, bool(entry.get("mirror_needed")))
        _orig = dt.lib
        dt.lib = lambda: replay_lib                          # noqa: E731  scoped to this call
        try:
            info = BED.render_smooth(dt, cm, loc, palette, cp, out_path, w, h, ss, filt)
        finally:
            dt.lib = _orig
    elif stops is not None:
        cmaps = out_path.parent / f"{out_path.stem}__replay_cmaps.json"
        AL.one_entry_colormaps(entry, stops, cmaps)
        _orig_pool = dt.POOL_CMAPS
        dt.POOL_CMAPS = str(cmaps)
        try:
            info = dt.render_candidate(loc, ex["mode"], ex["kind"], palette, cp,
                                       out_path, w, h, ss, filt)
        finally:
            dt.POOL_CMAPS = _orig_pool
            cmaps.unlink(missing_ok=True)
    elif ex["kind"] == "smooth":
        info = BED.render_smooth(dt, cm, loc, palette, cp, out_path, w, h, ss, filt)
    else:
        info = dt.render_candidate(loc, ex["mode"], ex["kind"], palette, cp,
                                   out_path, w, h, ss, filt)

    img = np.asarray(Image.open(out_path).convert("RGB"))
    return {"arm": arm, "path": str(out_path), "stamp": (info or {}).get("autolevel"),
            "stats": AL.tone_stats(img), "secs": time.time() - t0}


def _render_example(job: tuple) -> dict:
    """ONE LOCATION, ALL ITS ARMS, in one worker — before, band, and (when asked, and when
    the curve actually acted) the replay.

    The unit of work is the EXAMPLE and not the ARM on purpose: two arms of one example share
    a location, a geometry and therefore the render path's cached field / engine temp, so
    running them in different workers is two processes writing one file. Serializing them
    inside a worker costs nothing (the pool still runs `workers` locations at once) and
    removes the race rather than defending against it."""
    ex, sd, arms_dir_s, want_replay = job
    arms_dir = Path(arms_dir_s)
    os.environ.setdefault("RAYON_NUM_THREADS", str(sd["engine_threads"]))
    import location as loc_mod
    from tools import colormap as cm
    from tools.mining import deploy_tail as dt
    from tools.emission import build_emission_diversity_v1 as BED
    mods = (loc_mod, cm, dt, BED)

    eid = ex["example_id"]
    out = {"example_id": eid, "arms": {}}
    t0 = time.time()
    for arm in sd["arms"]:
        out["arms"][arm] = _one_arm(ex, arm, sd, arms_dir / f"{eid}__{arm}.jpg", None, mods)
    stamp = out["arms"]["band"]["stamp"]
    if want_replay and stamp and stamp.get("acted"):
        entry = dt.lib().colormaps[ex["palette"]]
        stops = AL.stops_from_stamp(stamp, entry)
        out["arms"]["replay"] = _one_arm(ex, "replay", sd, arms_dir / f"{eid}__replay.jpg",
                                         stops, mods)
    out["secs"] = time.time() - t0
    return out


def _sd(sspec: SheetSpec) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in sspec.__dict__.items()}


def _run_jobs(jobs: list, sspec: SheetSpec, tag: str) -> list:
    recs, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=sspec.workers) as px:
        futs = {px.submit(_render_example, j): j[0]["example_id"] for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            eid = futs[fu]
            try:
                recs.append(fu.result())
            except Exception as e:                                   # noqa: BLE001
                print(f"[fail] {tag} {eid}: {type(e).__name__}: {e}", flush=True)
                continue
            print(f"[{tag} {i}/{len(jobs)}] {eid} "
                  f"{'+'.join(recs[-1]['arms'])} {recs[-1]['secs']:.1f}s "
                  f"(elapsed {time.time() - t0:.0f}s)", flush=True)
    return recs


# =========================================================================== #
# 4. Classification + the coverage requirement.
# =========================================================================== #
def classify(stamp: dict | None) -> str:
    """What the operator did to this example, as one token for the coverage check."""
    if stamp is None:
        return "switch_off"
    c = stamp["curve"]
    if not c.get("applies"):
        return "not_applied"
    if c.get("identity"):
        return "identity"
    s = c["sides"]
    if s["white_pt"] == -1:
        return "white_below"
    if s["white_pt"] == +1:
        return "white_above"
    return "mid_below" if s["mid"] == -1 else ("mid_above" if s["mid"] == +1 else "black_only")


def choose_rows(recs: list, sspec: SheetSpec, klass_key: str = "prescreen") -> tuple:
    """`n_sheet` rows spanning the three modes, with the required classes present.

    Deterministic: the required classes are seated first (one example each, in id order),
    then the rest fill round-robin across modes so no mode is systematically dropped."""
    by_class = {}
    for r in recs:
        by_class.setdefault(r[klass_key], []).append(r)
    picked, seen = [], set()
    missing = []
    for want in sspec.must_cover:
        cands = sorted(by_class.get(want, []), key=lambda r: r["example_id"])
        if not cands:
            missing.append(want)
            continue
        picked.append(cands[0])
        seen.add(cands[0]["example_id"])
    by_mode = {}
    for r in sorted(recs, key=lambda r: r["example_id"]):
        if r["example_id"] not in seen:
            by_mode.setdefault(r["mode"], []).append(r)
    modes = sorted(by_mode)
    k = 0
    while len(picked) < sspec.n_sheet and any(by_mode.values()):
        m = modes[k % len(modes)]
        k += 1
        if by_mode[m]:
            picked.append(by_mode[m].pop(0))
    picked.sort(key=lambda r: (r["mode"], r["example_id"]))
    return picked, missing


# =========================================================================== #
# 5. The sheet.
# =========================================================================== #
TILE_W, TILE_H, CAP_H, PAD = 560, 315, 118, 8


def _font(sz):
    for name in ("consola.ttf", "arial.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def _caption(r: dict, arm: str) -> str:
    st = r[f"{arm}_stats"]
    lines = [arm.upper()]
    blk = "n/a(guard)" if st["black_pt"] is None else f"{st['black_pt']:.3f}"
    lines.append(f"L med(mask) {st['mid']:.3f}  blk {blk} (all {st['black_pt_all']:.3f}) "
                 f"wht {st['white_pt']:.3f}")
    lines.append(f"in-mask chroma {st['in_mask_chroma']:.4f}  dark chroma "
                 f"{st['dark_chroma']:.4f}")
    if arm == "before":
        lines.append("switch OFF — the pre-operator render (forced, not the default)")
        return "\n".join(lines)
    stamp = r.get("stamp")
    if stamp is None:
        lines.append("NO STAMP (operator did not run)")
        return "\n".join(lines)
    c, cap = stamp["curve"], stamp["chroma_cap"]
    ref = stamp["reference"]
    if not c.get("applies"):
        lines.append(f"NOT APPLIED — {c.get('reason')}")
    elif c.get("identity"):
        lines.append("IDENTITY — all three statistics in band")
    else:
        s = c["sides"]
        lines.append(f"curve [{c['black_pt']:.3f},{c['white_pt']:.3f}]->"
                     f"[{c['out_ends'][0]:.3f},{c['out_ends'][1]:.3f}] p={c['exponent']:.3f}"
                     f"{'  CLAMPED' if c.get('clamped') else ''}")
        lines.append(f"sides blk{s['black_pt']:+d} wht{s['white_pt']:+d} mid{s['mid']:+d}"
                     f"{'  BLACK GUARDED' if c.get('black_guarded') else ''}"
                     f"  chroma-capped {cap['n_capped']}/{cap['n_stops']}")
    lines.append(f"stamp {stamp['operator']} · ref {ref['version']} n={ref['n_images']} "
                 f"sha {ref['sha256'][:8]} · Δmax {r.get('delta_max', '?')}"
                 f" · replay {r.get('replay', '—')}")
    return "\n".join(lines)


def build_sheet(rows: list, title: str, out_png: Path, sspec: SheetSpec):
    f, fh = _font(15), _font(19)
    head = 58
    W = PAD + len(sspec.arms) * (TILE_W + PAD)
    H = head + len(rows) * (TILE_H + CAP_H + PAD) + PAD
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 12), title, font=fh, fill=(235, 235, 235))
    y = head
    for r in rows:
        for i, arm in enumerate(sspec.arms):
            x = PAD + i * (TILE_W + PAD)
            p = r.get(f"{arm}_path")
            if p and Path(p).exists():
                with Image.open(p) as im:
                    sheet.paste(im.convert("RGB").resize((TILE_W, TILE_H), Image.LANCZOS),
                                (x, y))
            else:
                d.rectangle([x, y, x + TILE_W, y + TILE_H], fill=(40, 30, 30))
                d.text((x + 8, y + 8), "(no render)", font=f, fill=(200, 120, 120))
            d.multiline_text((x + 4, y + TILE_H + 4), _caption(r, arm), font=f,
                             fill=(210, 210, 215), spacing=3)
        d.text((PAD, y + TILE_H + CAP_H - 18),
               f"{r['example_id']}  {r['mode']}  {r.get('fractal_type')}  "
               f"palette {r['palette']}  [{r['klass']}]", font=f, fill=(150, 150, 160))
        y += TILE_H + CAP_H + PAD
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


# =========================================================================== #
# 6. Driver.
# =========================================================================== #
def _max_abs(a: Path, b: Path) -> int:
    x = np.asarray(Image.open(a).convert("RGB")).astype(np.int16)
    y = np.asarray(Image.open(b).convert("RGB")).astype(np.int16)
    if x.shape != y.shape:
        return -1
    return int(np.abs(x - y).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", default="stage_v1")
    ap.add_argument("--limit", type=int, default=0,
                    help="bounded end-to-end: N examples per mode (the sheet is stamped "
                         "INCOMPLETE and is not the verification)")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:                                            # noqa: BLE001
            pass
    sspec = SPECS[args.spec]

    # SINK ISOLATION, asserted before the first write. This run must not be able to name a
    # path under data/emission/ — it is a throwaway that drives the production render code.
    from tools.emission import emission_sinks as ESINKS
    root = ESINKS.resolve_record_root(ROOT, smoke=True, explicit=None, run_id=sspec.name)
    ESINKS.use(root)
    sinks = ESINKS.assert_isolated(ROOT, root, "autolevel_verify")
    print(f"[sinks] EPHEMERAL — record sinks isolated under {root}; "
          f"nothing writes data/emission/. " + " ".join(p.name for p in sinks), flush=True)

    out = sspec.out_dir
    arms = out / "arms"
    arms.mkdir(parents=True, exist_ok=True)
    print(f"[ref] {AL.load_reference()['_path']} · "
          + " · ".join(f"{k} [{v[0]:.3f},{v[1]:.3f}]"
                       for k, v in AL.bands(AL.load_reference()).items()), flush=True)

    pop = prescreen(population(sspec))
    pre_counts = {k: sum(1 for p in pop if p["prescreen"] == k)
                  for k in sorted({p["prescreen"] for p in pop})}
    print(f"[pop] {len(pop)} candidates pre-screened on their committed corpus crops: "
          f"{pre_counts}", flush=True)

    chosen, missing = choose_rows(pop, sspec, "prescreen")
    if args.limit:
        keep, seen = [], {}
        for p in chosen:
            if seen.get(p["mode"], 0) < args.limit:
                seen[p["mode"]] = seen.get(p["mode"], 0) + 1
                keep.append(p)
        chosen = keep
    by_id = {p["example_id"]: p for p in chosen}
    print(f"[pick] {len(chosen)} sheet rows: "
          + ", ".join(f"{m}×{sum(1 for p in chosen if p['mode'] == m)}"
                      for m in sorted({p['mode'] for p in chosen}))
          + (f"  MISSING {missing}" if missing else ""), flush=True)

    # -- every arm of a row in ONE worker (see `_render_example`). The class in the verdict is
    #    measured HERE, on the production render, never taken from the pre-screen that
    #    selected the row. The replay is asked for on the first `replay_n` rows the pre-screen
    #    says will act; whether it happened is reported, so the cap is never silent.
    replay_ask = [e["example_id"] for e in chosen
                  if e["prescreen"] not in ("identity", "no_crop")][:sspec.replay_n]
    n_replay_asked = len(replay_ask)
    jobs = [(ex, _sd(sspec), str(arms), ex["example_id"] in replay_ask) for ex in chosen]
    recs = _run_jobs(jobs, sspec, "render")

    merged = {}
    for r in recs:
        e = merged.setdefault(r["example_id"], dict(by_id[r["example_id"]]))
        for arm, a in r["arms"].items():
            e[f"{arm}_path"] = a["path"]
            e[f"{arm}_stats"] = a["stats"]
        e["stamp"] = r["arms"]["band"]["stamp"]
    rows = [e for e in merged.values() if "before_path" in e and "band_path" in e]
    for e in rows:
        e["klass"] = classify(e.get("stamp"))
        e["delta_max"] = _max_abs(Path(e["before_path"]), Path(e["band_path"]))
        e["replay"] = (_max_abs(Path(e["band_path"]), Path(e["replay_path"]))
                       if e.get("replay_path") else "—")
    picked = sorted(rows, key=lambda r: (r["mode"], r["example_id"]))
    covered = {e["klass"] for e in picked}
    missing = sorted(set(sspec.must_cover) - covered)
    acting = [e for e in picked if (e.get("stamp") or {}).get("acted")]
    n_skipped_replay = len(acting) - sum(1 for e in picked if isinstance(e["replay"], int))

    # -- verdict.
    idents = [e for e in rows if e["klass"] == "identity"]
    bad_ident = [e["example_id"] for e in idents if e["delta_max"] != 0]
    replayed = [e for e in picked if isinstance(e.get("replay"), int)]
    verdict = {
        "spec": sspec.name,
        "incomplete": bool(args.limit),
        "reference": {k: v for k, v in AL.load_reference().items()
                      if k in ("version", "n_images", "derived", "_sha256")},
        "n_prescreened": len(pop), "n_sheet_rows": len(picked),
        "prescreen_by_class": pre_counts,
        "by_class": {k: sum(1 for e in rows if e["klass"] == k)
                     for k in sorted({e["klass"] for e in rows})},
        "coverage_required": list(sspec.must_cover), "coverage_missing": missing,
        "prescreen_vs_render_disagreements": [
            {"example_id": e["example_id"], "prescreen": e["prescreen"], "render": e["klass"]}
            for e in picked if e.get("prescreen") != e["klass"]],
        "identity": {"n": len(idents), "max_abs_delta": [e["delta_max"] for e in idents],
                     "byte_identical": not bad_ident, "offenders": bad_ident},
        "replay": {"n_checked": len(replayed), "n_acting": len(acting),
                   "n_asked": n_replay_asked, "n_not_checked": n_skipped_replay,
                   "max_abs_delta": [e["replay"] for e in replayed],
                   "exact": all(e["replay"] == 0 for e in replayed) if replayed else None},
        "rows": [{k: e.get(k) for k in
                  ("example_id", "mode", "palette", "prescreen", "klass", "delta_max",
                   "replay", "stamp")}
                 for e in picked],
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")

    title = (f"band auto-level — PRODUCTION path, switch OFF | switch ON   "
             f"({len(picked)} examples, {sspec.geom[0]}x{sspec.geom[1]} ss{sspec.geom[2]}, "
             f"literal render bytes)" + ("   [INCOMPLETE --limit run]" if args.limit else ""))
    build_sheet(picked, title, out / "sheet_autolevel_production.png", sspec)

    print(f"\n[verdict] pre-screen {pre_counts} -> sheet classes {verdict['by_class']}")
    if verdict["prescreen_vs_render_disagreements"]:
        print(f"[verdict] pre-screen vs production render disagreed on "
              f"{verdict['prescreen_vs_render_disagreements']}")
    print(f"[verdict] identity n={len(idents)} byte-identical="
          f"{verdict['identity']['byte_identical']} (max|Δ| {verdict['identity']['max_abs_delta']})")
    print(f"[verdict] replay {len(replayed)}/{len(acting)} acting checked "
          f"(max|Δ| {verdict['replay']['max_abs_delta']}); "
          f"{n_skipped_replay} not checked (replay_n cap)")
    if missing:
        print(f"[verdict][WARN] coverage MISSING: {missing}")
    print(f"[out] {out/'sheet_autolevel_production.png'} + verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
