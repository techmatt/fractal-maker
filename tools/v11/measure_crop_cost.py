#!/usr/bin/env python
r"""COST + DISK: `crop-batch` (one extended field per location) vs the legacy per-tile recipe.

Sizing input for the v11 rebuild. Two arms over the SAME locations and the SAME 24 tiles:

  A  crop-batch      one extended-field iteration pass per location, 24 tiles derived
  B  v4-render-batch the v8b/v9/v10 recipe — 24 independent iterate+shade renders

Arm B's plan is built FROM arm A's emitted manifest, so the two render the identical 24
(viewport, palette, AA) slots rather than two independently-drawn fan-outs. Only the
execution differs, which is the whole question.

SAMPLED IN RUN ORDER (CLAUDE.md's projection rule): contiguous blocks, one per region of
the location file, not a stratified or random draw. `data/v10/manifest.jsonl` is emitted in
FAMILY order with the expensive material contiguous, and a sample unbiased for mean
per-tile cost is not unbiased for the wall clock of a run whose expensive work is
contiguous — that is the 1.65x miss the v9 cache render took. Contiguous blocks preserve
the ordering the run will actually experience.

SCALING TO 32 / 48 TILES IS FITTED, NOT ASSUMED. Arm A is run at both 24 and 48 tiles per
location, so its per-location cost decomposes into a measured fixed part (the one field)
and a measured marginal part (per tile); the 32-tile point is interpolated on that fit,
never on a "tiles are free" claim. Arm B is linear in tiles by construction (each row is an
independent render) and is scaled as such.

DISK is measured from the real emitted JPGs at both the per-tile quality DRAW and a flat
q85 (the v4..v10 cache's actual setting), so the cost of adding the quality axis is a
number rather than a compression-ratio guess.

  uv run python tools/v11/measure_crop_cost.py [--blocks 6] [--per-block 6]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "tools/corpus", "tools/scoring"):
    sys.path.insert(0, str(ROOT / sub))

import corpus_common as cc            # noqa: E402
import paths                          # noqa: E402
import production_pins as pins        # noqa: E402
from location import maxiter_policy_token  # noqa: E402

BIN = ROOT / "target" / "release" / "fractal-generator.exe"
MANIFEST = ROOT / "data" / "v10" / "manifest.jsonl"
COLORMAPS = ROOT / "data" / "v10" / "colormaps.json"

# The v8b palette axis: two pinned + two drawn per location.
PINNED = ["twilight_shifted", "blue_orange"]
AA = ["aliased:point", "antialiased:lanczos3"]
CACHE_JPGQ = 85                       # v4-render-batch's locked cache quality
Q_LO, Q_HI = 85, 95                   # the proposed per-tile draw (train-time jitter band)

# The v11 rebuild's projected population (the prompt's figure; v10's corpus is 8,382, so
# this is 8,382 + ~5.9k of new material, not a count of anything on disk today).
V11_LOCATIONS = 14_300
TILE_POINTS = (24, 32, 48)

# Measured 2026-08-07 by walking both trees:
#   uv run python -c "... os.walk(paths.bulk('data/v9/aug_cache')) ..."
V10_CACHE = {"tiles": 201_216, "gib": 14.30, "mean_kib": 74.5,
             "note": "data/v9/aug_cache (170,808 @ 12.09 GiB) + data/v10/aug_cache "
                     "(30,408 @ 2.21 GiB), measured 2026-08-07"}


def pick_blocks(n_blocks: int, per_block: int) -> list:
    """`n_blocks` CONTIGUOUS runs of `per_block` locations, evenly spaced through the file."""
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    out, N = [], len(rows)
    for b in range(n_blocks):
        start = min(N - per_block, int(b * (N - per_block) / max(1, n_blocks - 1)))
        out.extend(rows[start:start + per_block])
    return out, N


def engine(argv: list, log: Path) -> float:
    env = cc.default_engine_env()
    t0 = time.time()
    with log.open("ab") as f:
        rc = subprocess.run([str(BIN), *argv], cwd=str(ROOT), env=env,
                            stdout=f, stderr=subprocess.STDOUT,
                            creationflags=cc.default_creationflags()).returncode
    if rc != 0:
        sys.exit(f"engine failed (rc={rc}): {argv[0]} {argv[1]}\n  see {log}")
    return time.time() - t0


def tree_bytes(root: Path) -> tuple[int, int]:
    n = b = 0
    for dp, _d, fs in os.walk(root):
        for f in fs:
            if f.endswith(".jpg"):
                n += 1
                b += os.path.getsize(os.path.join(dp, f))
    return n, b


def legacy_plan_from(manifest: Path, out_root: Path, jpgq_note: str) -> tuple[Path, int]:
    """Arm B's plan: the SAME 24 slots arm A emitted, one `v4-render-batch` row each.

    The AA level maps back to the legacy pair it stands for (`aliased` -> ss1 box,
    `antialiased` -> ss2 lanczos3), and the viewport is reconstructed from the crop's
    recorded scale/shift — so a slot is the same window in both arms.

    The cap is `auto_maxiter(fw_slot)`, which is what v9/v10 ACTUALLY pay, not the
    canonical-frame cap v11 uses. Using v11's rule here would credit arm B with v11's cap
    policy and measure the executor change against a recipe it never ran; the two differ by
    ~3% of the cap over a [0.90, 1.10] scale draw."""
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        fw0 = float(r["render"]["fw"])
        sc = r["crop"]["scale"]
        fw = sc * fw0
        cx = float(r["render"]["cx"]) + r["crop"]["shift_dx"] * fw0
        cy = float(r["render"]["cy"]) + r["crop"]["shift_dy"] * fw0
        ss, filt = (1, "box") if r["aa"]["level"] == "aliased" else (2, "lanczos3")
        row = {"cx": repr(cx), "cy": repr(cy), "fw": repr(fw),
               "fractal_type": r["render"]["fractal_type"],
               "palette": r["palette"], "ss": ss, "filter": filt,
               "maxiter": int(pins.auto_maxiter(fw)),
               "out": (out_root / str(r["loc_id"]) /
                       Path(r["out"]).name).as_posix()}
        for k in ("c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im"):
            if r["render"].get(k) is not None:
                row[k] = r["render"][k]
        rows.append(row)
    p = out_root.parent / f"legacy_plan_{jpgq_note}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--per-block", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else paths.scratch("v11_cost")
    out.mkdir(parents=True, exist_ok=True)
    if not BIN.exists():
        sys.exit(f"release binary missing: {BIN} (cargo build --release)")
    log = out / "engine.log"

    locs, n_corpus = pick_blocks(a.blocks, a.per_block)
    fams = Counter(r.get("fractal_type", "mandelbrot") for r in locs)
    print(f"population: {len(locs)} locations in {a.blocks} contiguous blocks of "
          f"{a.per_block} over {n_corpus} rows (RUN ORDER)\n  families {dict(sorted(fams.items()))}",
          flush=True)

    token = maxiter_policy_token()
    loc_rows = []
    for r in locs:
        base = {k: r[k] for k in ("cx", "cy", "fw", "fractal_type") if r.get(k) is not None}
        for k in ("c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im"):
            if r.get(k) is not None:
                base[k] = r[k]
        loc_rows.append({"loc_id": r["loc_id"], **base,
                         "maxiter": int(pins.auto_maxiter(float(r["fw"]))),
                         "maxiter_policy": token})
    lp = out / "locations.jsonl"
    lp.write_text("\n".join(json.dumps(r) for r in loc_rows) + "\n", encoding="utf-8")

    res = {"population": {"locations": len(locs), "blocks": a.blocks,
                          "per_block": a.per_block, "corpus_rows": n_corpus,
                          "order": "contiguous blocks in file (run) order",
                          "families": dict(sorted(fams.items()))},
           "engine": {"threads": cc.DEFAULT_ENGINE_THREADS, "priority": "BELOW_NORMAL",
                      "processes": 1},
           "arms": {}}

    # --- Arm A at 24 and 48 tiles/location (the fit) ---
    for n_tiles, geoms, draw in ((24, 3, 2), (48, 6, 2)):
        root = out / f"cropA_{n_tiles}"
        mf = out / f"cropA_{n_tiles}.jsonl"
        wall = engine(["crop-batch", "--locations", str(lp), "--colormaps", str(COLORMAPS),
                       "--out-root", str(root), "--manifest", str(mf),
                       "--geoms", str(geoms), "--aa", " ".join(AA),
                       "--palettes", " ".join(PINNED),
                       "--palette-pool", "magma inferno plasma viridis cividis turbo "
                                         "twilight cmr.amber cmr.copper cmr.ember",
                       "--draw-palettes", str(draw),
                       "--jpg-quality-lo", str(Q_LO), "--jpg-quality-hi", str(Q_HI),
                       "--no-resume", "--log-every", "10"], log)
        n, b = tree_bytes(root)
        res["arms"][f"crop_batch_{n_tiles}"] = {
            "tiles_per_location": n_tiles, "tiles": n, "wall_s": round(wall, 2),
            "s_per_location": round(wall / len(locs), 4),
            "s_per_tile": round(wall / max(n, 1), 5),
            "bytes": b, "mean_kib": round(b / max(n, 1) / 1024, 2),
            "jpg_quality": f"draw U[{Q_LO},{Q_HI}]",
        }
        print(f"  crop-batch @{n_tiles:>2} tiles/loc: {wall:7.1f}s  "
              f"{wall/len(locs):.3f} s/loc  {n} tiles  {b/max(n,1)/1024:.1f} KiB/tile",
              flush=True)

    # --- Arm B: the same 24 slots, one legacy render each ---
    lroot = out / "legacyB_24"
    plan, n_rows = legacy_plan_from(out / "cropA_24.jsonl", lroot, "q85")
    wall = engine(["v4-render-batch", "--plan", str(plan), "--colormaps", str(COLORMAPS),
                   "--jpg-quality", str(CACHE_JPGQ), "--log-every", "5000"], log)
    n, b = tree_bytes(lroot)
    res["arms"]["v4_render_batch_24"] = {
        "tiles_per_location": 24, "tiles": n, "wall_s": round(wall, 2),
        "s_per_location": round(wall / len(locs), 4),
        "s_per_tile": round(wall / max(n, 1), 5),
        "bytes": b, "mean_kib": round(b / max(n, 1) / 1024, 2),
        "jpg_quality": f"flat q{CACHE_JPGQ}",
        "plan_rows": n_rows,
    }
    print(f"  v4-render-batch @24 tiles/loc: {wall:7.1f}s  {wall/len(locs):.3f} s/loc  "
          f"{n} tiles  {b/max(n,1)/1024:.1f} KiB/tile", flush=True)

    # --- fit + projections ---
    A24 = res["arms"]["crop_batch_24"]["s_per_location"]
    A48 = res["arms"]["crop_batch_48"]["s_per_location"]
    marginal = (A48 - A24) / 24.0                    # s per extra tile
    fixed = A24 - 24 * marginal                      # s for the one field
    B_per_tile = res["arms"]["v4_render_batch_24"]["s_per_location"] / 24.0
    kib_draw = res["arms"]["crop_batch_24"]["mean_kib"]
    kib_flat = res["arms"]["v4_render_batch_24"]["mean_kib"]

    res["fit"] = {"crop_batch_fixed_s_per_location": round(fixed, 4),
                  "crop_batch_marginal_s_per_tile": round(marginal, 5),
                  "legacy_s_per_tile": round(B_per_tile, 5),
                  "basis": "two measured points (24, 48) on the same locations; legacy is "
                           "linear in tiles by construction"}
    res["projection"] = {"locations": V11_LOCATIONS, "points": {}}
    print(f"\n  fit: crop-batch = {fixed:.3f} s/location (one field) + "
          f"{marginal:.4f} s/tile   |   legacy = {B_per_tile:.4f} s/tile")
    print(f"\n=== PROJECTION at {V11_LOCATIONS:,} locations (1 process, "
          f"{cc.DEFAULT_ENGINE_THREADS} threads, BelowNormal) ===")
    print(f"  {'tiles/loc':>9} {'crop-batch':>12} {'legacy':>12} {'speedup':>8} "
          f"{'GB (draw)':>11} {'GB (q85)':>10}")
    for t in TILE_POINTS:
        a_h = V11_LOCATIONS * (fixed + t * marginal) / 3600
        b_h = V11_LOCATIONS * t * B_per_tile / 3600
        gb_d = V11_LOCATIONS * t * kib_draw / 1024 / 1024
        gb_f = V11_LOCATIONS * t * kib_flat / 1024 / 1024
        res["projection"]["points"][t] = {
            "crop_batch_hours": round(a_h, 2), "legacy_hours": round(b_h, 2),
            "speedup": round(b_h / a_h, 2) if a_h else None,
            "gib_quality_draw": round(gb_d, 1), "gib_flat_q85": round(gb_f, 1),
            "tiles": V11_LOCATIONS * t,
        }
        print(f"  {t:>9} {a_h:>10.2f} h {b_h:>10.2f} h {b_h/a_h:>7.1f}x "
              f"{gb_d:>10.1f} {gb_f:>9.1f}")

    res["disk"] = {
        "mean_kib_per_tile": {"quality_draw": kib_draw, "flat_q85": kib_flat,
                              "delta_pct": round(100 * (kib_draw / kib_flat - 1), 1)},
        "v10_cache_today": V10_CACHE,
        "peak_transient": {
            "note": "the extended field is a Vec<f32> in RAM, never written — transient "
                    "DISK is one JPG per in-flight tile",
            "field_ram_mib_per_location_ss2": round(1230 * 782 * 4 / 2**20, 2),
            "transient_disk_mib": 0.0,
        },
    }
    (out / "cost.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n  disk/tile: {kib_draw:.1f} KiB at the q{Q_LO}..{Q_HI} draw vs "
          f"{kib_flat:.1f} KiB flat q{CACHE_JPGQ} "
          f"({100*(kib_draw/kib_flat-1):+.1f}%);  v10 cache today: "
          f"{V10_CACHE['tiles']:,} tiles / {V10_CACHE['gib']} GiB")
    print(f"  peak transient disk: ~one JPG per in-flight tile; the field "
          f"({1230*782*4/2**20:.1f} MiB at ss2) never touches disk")
    print(f"\nwrote {out / 'cost.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
