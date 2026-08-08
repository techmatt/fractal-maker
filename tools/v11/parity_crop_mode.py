#!/usr/bin/env python
r"""PARITY: `crop-batch` (extended-field crop) vs the legacy `v4-render-batch` recipe.

The bar is `verification_practice.md` §8 — **functional parity on the output decision, not
byte identity**. The two paths legitimately differ: the legacy tile iterates its own frame
at `ss`, the new one resamples an extended field at a non-integer ratio from a fractional
offset, and the `aliased` level is a nearest-neighbour point sample rather than an exact
pixel-centre one (no even `field_ss` contains the ss1 sample points; see
`src/crop_batch.rs`). So this regresses what the deployed classifier DECIDES, and reports
the score-delta distribution around it.

  * IDENTITY geometry only (`--geoms 1`, scale 1.0, no shift) — the one framing where the
    two paths address the same window, so a delta is the resampling chain and nothing else.
  * BOTH AA levels, each against its legacy twin: `aliased:point` vs `ss1 box`,
    `antialiased:lanczos3` vs `ss2 lanczos3`.
  * Locations span every family in the corpus, apportioned by `tools/apportion.py`
    (`deal_round_robin` with a 1-per-family preseed, so a 111-row family is present without
    the 4,388-row one swamping the sample).
  * Both at `--field-ss 2` (the cheap default) AND at each `--also-field-ss` value. The
    sweep is not decoration: an EVEN `field_ss` places its sub-cell centres at 0.25/0.75 of
    a pixel and so contains no exact ss1 sample point, while an ODD one puts a sample at
    exactly 0.5 — so at the identity crop `field_ss 3` reproduces the legacy point sample
    exactly and `field_ss 2` cannot. That is the whole aliased-arm question, and it is a
    measurement rather than an argument.

SCORING IS fp32, NOT autocast. `score_lib.Scorer` runs the head under `torch.autocast` on
CUDA, which `verification_practice.md` §1.9 records as fine where a score RANKS and wrong
where it CUTS: fp16 accumulation moved two rows across a 0.90 cutpoint in the wallpaper-v3
population. This measurement is a difference of two scores near a threshold, which is the
cutting case, so the head is run at fp32 here and the deltas are the model's, not the
accumulator's.

  uv run python tools/v11/parity_crop_mode.py [--n 30] [--out scratch/v11_parity]

Writes the run tree + `parity.json` under `--out` (scratch-class: this is a measurement of
a transient state, per CLAUDE.md's analysis-text rule; what survives goes in the report).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "tools/corpus", "tools/scoring", "tools/mining", "tools/atlas"):
    sys.path.insert(0, str(ROOT / sub))

import apportion                      # noqa: E402
import corpus_common as cc            # noqa: E402
import paths                          # noqa: E402
import production_pins as pins        # noqa: E402
from location import maxiter_policy_token  # noqa: E402

BIN = ROOT / "target" / "release" / "fractal-generator.exe"
MANIFEST = ROOT / "data" / "v10" / "manifest.jsonl"
COLORMAPS = ROOT / "data" / "v10" / "colormaps.json"
PALETTE = "twilight_shifted"          # the deploy-matched scoring instrument
CACHE_JPGQ = 85                       # the v4..v10 cache's actual quality (v4_cache.rs)
SEED = 20260807

# (crop-batch aa spec, legacy ss, legacy filter, manifest aa label)
ARMS = [("aliased:point", 1, "box", "aliased"),
        ("antialiased:lanczos3", 2, "lanczos3", "antialiased")]


def pick_locations(n: int) -> list:
    """`n` locations spanning every family, apportioned then seeded-shuffled within family.

    `preseed={fam: 1}` makes one-per-family a FLOOR rather than a bonus (apportion.py's own
    distinction) — without it the three smallest families vanish from a 30-row sample."""
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r.get("fractal_type", "mandelbrot")].append(r)
    sizes = {k: len(v) for k, v in by_fam.items()}
    take = apportion.deal_round_robin(sizes, n, preseed={k: 1 for k in sizes})
    import random
    out = []
    for fam in sorted(sizes):
        pool = sorted(by_fam[fam], key=lambda r: r["loc_id"])
        random.Random(f"{SEED}|{fam}").shuffle(pool)
        out.extend(pool[: take.get(fam, 0)])
    return sorted(out, key=lambda r: r["loc_id"])


def write_inputs(locs: list, out: Path) -> tuple[Path, Path, dict]:
    """The location JSONL (crop-batch) and the matched legacy plan (v4-render-batch).

    Both carry the SAME per-row cap: `auto_maxiter(fw)` at the CANONICAL frame. That is
    also the prompt's rule for the extended field — the cap is NOT re-derived from the
    extended `fw`, so a wider field does not silently buy a higher cap."""
    token = maxiter_policy_token()
    loc_rows, plan_rows, caps = [], [], {}
    for r in locs:
        fw = float(r["fw"])
        mi = int(pins.auto_maxiter(fw))
        caps[r["loc_id"]] = mi
        base = {k: r[k] for k in ("cx", "cy", "fw", "fractal_type") if r.get(k) is not None}
        for k in ("c_re", "c_im", "p_re", "p_im", "zm1_re", "zm1_im"):
            if r.get(k) is not None:
                base[k] = r[k]
        loc_rows.append({"loc_id": r["loc_id"], **base,
                         "maxiter": mi, "maxiter_policy": token})
        for _spec, ss, filt, label in ARMS:
            plan_rows.append({**base, "palette": PALETTE, "ss": ss, "filter": filt,
                              "maxiter": mi,
                              "out": (out / "legacy" / str(r["loc_id"]) / f"{label}.jpg").as_posix()})
    lp = out / "locations.jsonl"
    pp = out / "legacy_plan.jsonl"
    lp.write_text("\n".join(json.dumps(r) for r in loc_rows) + "\n", encoding="utf-8")
    pp.write_text("\n".join(json.dumps(r) for r in plan_rows) + "\n", encoding="utf-8")
    return lp, pp, caps


def engine(argv: list, log: Path) -> float:
    """Run the engine once through the COMMITTED launch defaults (threads + priority),
    never a restated pair. One process at a time — `verification_practice.md` §3."""
    env = cc.default_engine_env()
    t0 = time.time()
    with log.open("ab") as f:
        rc = subprocess.run([str(BIN), *argv], cwd=str(ROOT), env=env,
                            stdout=f, stderr=subprocess.STDOUT,
                            creationflags=cc.default_creationflags()).returncode
    if rc != 0:
        sys.exit(f"engine failed (rc={rc}): {' '.join(argv[:3])}...\n  see {log}")
    return time.time() - t0


def score_fp32(scorer, paths_):
    """Cumulative rank probabilities at fp32 — the Scorer's transform and head, without
    its `torch.autocast` wrapper (see the module docstring)."""
    import torch
    from PIL import Image
    out = []
    for i in range(0, len(paths_), 32):
        batch = []
        for p in paths_[i:i + 32]:
            with Image.open(p) as im:
                im.load()
                batch.append(im.convert("RGB"))
        x = torch.stack([scorer.transform(im) for im in batch]).to(scorer.device)
        with torch.no_grad():
            logits = scorer.model(x).float().cpu()
        P = torch.sigmoid(logits).numpy()
        out.extend([tuple(float(v) for v in row) for row in P])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--also-field-ss", type=int, nargs="*", default=[3],
                    help="extra field_ss values to run alongside the default 2")
    a = ap.parse_args()
    out = Path(a.out) if a.out else paths.scratch("v11_parity")
    out.mkdir(parents=True, exist_ok=True)
    if not BIN.exists():
        sys.exit(f"release binary missing: {BIN} (cargo build --release)")

    locs = pick_locations(a.n)
    fams = Counter(r.get("fractal_type", "mandelbrot") for r in locs)
    print(f"parity population: {len(locs)} locations, families {dict(sorted(fams.items()))}",
          flush=True)
    lp, pp, caps = write_inputs(locs, out)
    log = out / "engine.log"

    fsss = [2] + [s for s in (a.also_field_ss or []) if s != 2]
    t_new = {}
    for fss in fsss:
        t_new[fss] = engine(
            ["crop-batch", "--locations", str(lp), "--colormaps", str(COLORMAPS),
             "--out-root", str(out / f"new_ss{fss}"),
             "--manifest", str(out / f"tiles_ss{fss}.jsonl"),
             "--field-ss", str(fss),
             "--geoms", "1", "--aa", " ".join(s for s, *_ in ARMS),
             "--palettes", PALETTE, "--draw-palettes", "0",
             "--jpg-quality-lo", str(CACHE_JPGQ), "--jpg-quality-hi", str(CACHE_JPGQ),
             "--no-resume", "--log-every", "10"], log)
    t_old = engine(["v4-render-batch", "--plan", str(pp), "--colormaps", str(COLORMAPS),
                    "--jpg-quality", str(CACHE_JPGQ), "--log-every", "1000"], log)
    print("render: " + "  ".join(f"new@ss{k} {v:.1f}s" for k, v in t_new.items())
          + f"  legacy {t_old:.1f}s   ({len(locs)} locations x {len(ARMS)} arms)", flush=True)

    # Pair by (loc_id, aa label) off the EMITTED manifest, not by reconstructing filenames.
    pairs = []
    for fss in fsss:
        new_by = {}
        for line in (out / f"tiles_ss{fss}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                new_by[(r["loc_id"], r["aa"]["level"])] = r["out"]
        for r in locs:
            for _spec, _ss, _f, label in ARMS:
                n_p = new_by.get((r["loc_id"], label))
                o_p = (out / "legacy" / str(r["loc_id"]) / f"{label}.jpg").as_posix()
                if n_p and Path(n_p).exists() and Path(o_p).exists():
                    pairs.append((r, f"{label}@ss{fss}", n_p, o_p))
    missing = len(locs) * len(ARMS) * len(fsss) - len(pairs)
    if missing:
        print(f"  WARNING: {missing} pair(s) missing a rendered tile", flush=True)
    if not pairs:
        sys.exit("no pairs to compare")

    from score_lib import Scorer, corn_decode
    scorer = Scorer(model_path=pins.ACTIVE_CKPT)
    print(f"scoring {2*len(pairs)} tiles on {pins.ACTIVE_CKPT} (K={scorer.k}, fp32)", flush=True)
    Pn = score_fp32(scorer, [p[2] for p in pairs])
    Po = score_fp32(scorer, [p[3] for p in pairs])

    recs, flips = [], []
    for (r, label, n_p, o_p), pn, po in zip(pairs, Pn, Po):
        fam = r.get("fractal_type", "mandelbrot")
        # Decode through score_lib's SINGLE-SOURCE rank decode at the production t_good for
        # this partition; never re-implement the >= counting inline.
        import partitions as part
        from production_seeder import t_good_for  # noqa: E402  (the live per-partition cut)
        tg = t_good_for(part.partition_of(fam, default=fam))
        cn = corn_decode(pn[0], pn[1], p_great=pn[2] if len(pn) > 2 else None, t_good=tg)
        co = corn_decode(po[0], po[1], p_great=po[2] if len(po) > 2 else None, t_good=tg)
        rec = {"loc_id": r["loc_id"], "family": fam, "aa": label, "t_good": tg,
               "maxiter": caps[r["loc_id"]],
               "score_new": sum(pn), "score_old": sum(po),
               "p_new": list(pn), "p_old": list(po),
               "class_new": cn, "class_old": co,
               "new": n_p, "old": o_p}
        recs.append(rec)
        if cn != co:
            flips.append(rec)

    def q(vals, p):
        v = sorted(vals)
        return v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]

    summary = {"n_pairs": len(recs), "n_locations": len(locs), "ckpt": pins.ACTIVE_CKPT,
               "k": scorer.k, "precision": "fp32 (autocast disabled)",
               "families": dict(sorted(fams.items())),
               "wall_s": {**{f"crop_batch_ss{k}": round(v, 2) for k, v in t_new.items()},
                          "v4_render_batch": round(t_old, 2)},
               "by_arm": {}}
    for label in sorted({r["aa"] for r in recs}):
        sub = [r for r in recs if r["aa"] == label]
        if not sub:
            continue
        d = [r["score_new"] - r["score_old"] for r in sub]
        dn = [abs(r["p_new"][0] - r["p_old"][0]) for r in sub]
        summary["by_arm"][label] = {
            "n": len(sub),
            "d_score": {"mean": sum(d) / len(d), "median": q(d, 0.5),
                        "p05": q(d, 0.05), "p95": q(d, 0.95),
                        "min": min(d), "max": max(d),
                        "max_abs": max(abs(x) for x in d)},
            "abs_d_p_notbad": {"median": q(dn, 0.5), "p95": q(dn, 0.95), "max": max(dn)},
            "flips": sum(1 for r in sub if r["class_new"] != r["class_old"]),
        }
    summary["flips"] = flips
    (out / "parity.json").write_text(
        json.dumps({"summary": summary, "records": recs}, indent=1), encoding="utf-8")

    print("\n=== PARITY (functional, on the deployed decision) ===")
    for label, s in summary["by_arm"].items():
        ds = s["d_score"]
        print(f"  {label:<12} n={s['n']:<3} dscore med {ds['median']:+.4f}  "
              f"[p05 {ds['p05']:+.4f}, p95 {ds['p95']:+.4f}]  max|d| {ds['max_abs']:.4f}  "
              f"|dP(notbad)| med {s['abs_d_p_notbad']['median']:.4f} "
              f"max {s['abs_d_p_notbad']['max']:.4f}  FLIPS {s['flips']}/{s['n']}")
    if flips:
        print("\n  decision flips (side-by-side images written):")
        for f in flips:
            print(f"    loc {f['loc_id']:<6} {f['family']:<18} {f['aa']:<12} "
                  f"class {f['class_old']} -> {f['class_new']}  "
                  f"(score {f['score_old']:.3f} -> {f['score_new']:.3f}, t_good {f['t_good']})")
            print(f"      old {f['old']}\n      new {f['new']}")
        _flip_sheet(flips, out / "flips.png")
    print(f"\nwrote {out / 'parity.json'}")
    return 0


def _flip_sheet(flips: list, dest: Path) -> None:
    """One row per flip: legacy tile | new tile. A decision flip must be LOOKED AT."""
    from PIL import Image, ImageDraw
    rows = [(Image.open(f["old"]).convert("RGB"), Image.open(f["new"]).convert("RGB"), f)
            for f in flips]
    w, h = rows[0][0].size
    sheet = Image.new("RGB", (2 * w + 12, len(rows) * (h + 20)), (16, 16, 16))
    d = ImageDraw.Draw(sheet)
    for i, (a, b, f) in enumerate(rows):
        y = i * (h + 20)
        sheet.paste(a, (0, y + 20))
        sheet.paste(b, (w + 12, y + 20))
        d.text((2, y + 5), f"loc {f['loc_id']} {f['family']} {f['aa']}  "
                           f"LEGACY class {f['class_old']} | NEW class {f['class_new']}",
               fill=(230, 230, 230))
    sheet.save(dest)
    print(f"  wrote {dest}")


if __name__ == "__main__":
    raise SystemExit(main())
