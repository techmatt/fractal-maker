#!/usr/bin/env python
r"""q4_harvest_ledger.py — mint a CURRENT-decoded `q4_harvest` supply ledger for emission.

The q4 tight harvest (`tools/studies/q4_harvest_tight.py`) selects palette-free minibrot
framings by the q4 GOODNESS field (G >= a label-derived cutoff). That signal is ORTHOGONAL
to v7 (v7 is blind to q4 quality and to the window labels), so these locations must NOT be
gated on v7's own q3 verdict (`decoded_class==3`) — v7 would silently veto good q4 framings.

This tool renders + guarded-v7-decodes each harvest candidate at reframe/deploy fidelity
(640x360 ss2, mandelbrot, no reframe search — the harvest already fixed the framing) and
writes a source-tagged (`mix_source="q4_harvest"`) intake-ready ledger. It is the mandelbrot
analogue of `tools/phoenix/classic_phoenix_supply.py`; it reuses that path's exact
primitives (reframe guard-field render, `guard.make_guarded_scorer`, `score_lib.corn_decode`).

v7 is a FLOOR here, not the gate. `decoded_class` is COMPUTED and stored (for the readout),
but admission is deferred to the emission driver's source-aware `descriptor.load_admitted`,
which admits a `q4_harvest` row on the v7 badness floor (`p_notbad>=0.5`) ∧ guard_pass ∧
distinct — see docs/design/q4_harvest_emission.md.

`distinct` is set True for every guard-passing row: the harvest already elliptical-NMS-deduped
the framings, and the REAL morphology dedup ("incremental medoid within type") is the emission
driver's intake clustering (`descriptor.assign_morph_clusters`, cos 0.974) — this ledger seeds
that, it does not pre-empt it.

Durable outputs under data/emission/q4_harvest/ (survive `rm -r out/*`):
  rescored.jsonl        per-candidate v7 rescore result (guard-pass AND guard-fail) — resume key
  outcome_ledger.jsonl  guard-passing rows, distinct=True (intake-ready; floor applied at intake)
  stats.json           per-condition counts (rendered / guard / floor) + decoded_class hist

Decode tiles + guard fields are transient (out/emission/q4_harvest_decode/, per-candidate wiped).

  uv run python tools/emission/q4_harvest_ledger.py --limit 5    # smoke
  uv run python tools/emission/q4_harvest_ledger.py              # full (background)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "reframe",
          ROOT / "tools" / "scoring", ROOT / "tools" / "mining", ROOT / "tools" / "corpus"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import guard                                    # noqa: E402
import reframe                                  # noqa: E402
import production_seeder as ps                  # noqa: E402
from score_lib import corn_decode               # noqa: E402
from colormap import load_field                 # noqa: E402

CANDIDATES = ROOT / "out" / "q4_stage1" / "harvest_tight" / "candidates.json"
RUN_DIR = ROOT / "data" / "emission" / "q4_harvest"
SCRATCH = ROOT / "out" / "emission" / "q4_harvest_decode"
SOURCE_TAG = "q4_harvest"
FAMILY = "mandelbrot"
RENDER_W, RENDER_H, RENDER_SS = reframe.RENDER_W, reframe.RENDER_H, reframe.RENDER_SS


def log(msg: str):
    print(msg, flush=True)


def _cand_id(i: int) -> str:
    return f"q4h_{i:04d}"


def _decode_candidate(scorer, c: dict, tile: Path):
    """Render the candidate framing at 640x360 ss2 with the co-located guard field, read
    the guard verdict/stats off that field, and guarded-v7-score. Returns
    (guard_pass, guard_fail_reason, p_notbad, p_good, interior_frac, field_std)."""
    loc = reframe.Location(family=FAMILY, c_re=None, c_im=None,
                           cx=str(c["cx_win"]), cy=str(c["cy_win"]),
                           fw=float(c["fw_win"]), family_params={})
    cand = {"cx": str(c["cx_win"]), "cy": str(c["cy_win"]),
            "fw": float(c["fw_win"]), "maxiter": int(c["maxiter"])}
    ok, err = reframe._render(loc, cand, tile, RENDER_W, RENDER_H, RENDER_SS)
    if not ok:
        raise RuntimeError(f"render failed: {err}")
    # guard stats + reason from the co-located field sidecar (same field the guarded
    # scorer gates on; read here to record the reason for the reject autopsy).
    sidecar = guard.field_sidecar_for(tile)
    stats = guard.field_measures(load_field(sidecar).values)
    reason = guard.guard_fail(stats.interior_frac, stats.field_std)
    score, notbad, good = scorer.score_paths([tile])[0]
    guard_pass = float(score) > guard.GUARD_SENTINEL + 1e-6
    # the guarded scorer's sentinel and the field-derived reason must agree (same field).
    assert guard_pass == (reason is None), \
        f"guard disagreement: sentinel_pass={guard_pass} field_reason={reason}"
    nb = float(notbad) if guard_pass else 0.0
    g = float(good) if guard_pass else 0.0
    return guard_pass, reason, nb, g, float(stats.interior_frac), float(stats.field_std)


def _row(i: int, c: dict, gp, reason, nb, g, t, interior_frac, field_std) -> dict:
    decoded = corn_decode(nb, g, t) if gp else None
    return {
        "id": _cand_id(i), "family": FAMILY,
        "outcome_cx": str(c["cx_win"]), "outcome_cy": str(c["cy_win"]),
        "outcome_fw": str(c["fw_win"]), "maxiter": int(c["maxiter"]),
        "reached_depth": 0,
        "decoded_class": decoded, "p_notbad": nb, "p_good": g, "t_good": t,
        "canon_pgood": g, "guard_pass": bool(gp), "guard_fail": reason,
        # distinct: True for every guard survivor — morphology dedup is the emission
        # driver's intake clustering (incremental medoid within type), not this ledger.
        "distinct": bool(gp), "dup_of": None,
        "mix_source": SOURCE_TAG, "scorer_version": ps.SCORER_VERSION,
        "interior_frac": interior_frac, "field_std": field_std,
        # q4 provenance (blind to v7): the goodness field value + framing identity.
        "q4_minibrot_id": c["minibrot_id"], "q4_scale": c["scale"], "q4_G": c["G"],
        "q4_box": c.get("box"),
    }


def rescore(cands, scorer, t, rescored_path: Path, limit=None):
    done = set()
    if rescored_path.exists():
        for line in rescored_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [(i, c) for i, c in enumerate(cands) if _cand_id(i) not in done]
    if limit:
        todo = todo[:limit]
    log(f"[rescore] {len(cands)} candidates; {len(done)} done, {len(todo)} to score (limit={limit})")
    fh = open(rescored_path, "a", encoding="utf-8")
    t0 = time.time()
    for k, (i, c) in enumerate(todo):
        cid = _cand_id(i)
        dwork = SCRATCH / cid
        try:
            gp, reason, nb, g, ifrac, fstd = _decode_candidate(scorer, c, dwork / "tile.jpg")
        except Exception as e:
            log(f"  WARN {cid} render/score failed: {type(e).__name__}: {str(e)[:140]}")
            shutil.rmtree(dwork, ignore_errors=True)
            continue
        row = _row(i, c, gp, reason, nb, g, t, ifrac, fstd)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        shutil.rmtree(dwork, ignore_errors=True)
        if (k + 1) % 10 == 0 or k + 1 == len(todo):
            el = time.time() - t0
            rate = (k + 1) / el if el else 0
            eta = (len(todo) - k - 1) / rate / 60 if rate else 0
            log(f"[rescore] {len(done)+k+1}/{len(cands)}  ({rate:.2f} cand/s, ETA {eta:.1f}m)")
    fh.close()


def finalize(rescored_path: Path, man: dict, t: float):
    rows = [json.loads(l) for l in rescored_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    guard_pass = [r for r in rows if r.get("guard_pass")]
    floor_admit = [r for r in guard_pass if (r.get("p_notbad") or 0.0) >= 0.5]
    with open(RUN_DIR / "outcome_ledger.jsonl", "w", encoding="utf-8") as f:
        for r in guard_pass:                       # intake-ready; floor applied at intake
            f.write(json.dumps(r) + "\n")
    guard_fail_reasons = Counter(r.get("guard_fail") for r in rows if not r.get("guard_pass"))
    decoded_hist = Counter(r.get("decoded_class") for r in guard_pass)
    stats = {
        "source_tag": SOURCE_TAG, "family": FAMILY, "scorer_version": ps.SCORER_VERSION,
        "t_good": t, "floor_pnotbad": 0.5,
        "harvest_gate": man.get("gate_used"), "harvest_cutoff":
            man.get(man.get("gate_used", "tight"), {}).get("cutoff"),
        "n_candidates": len(man["candidates"]),
        "n_rendered": len(rows),
        "n_render_failed": len(man["candidates"]) - len(rows),
        "n_guard_pass": len(guard_pass),
        "n_guard_fail": len(rows) - len(guard_pass),
        "guard_fail_reasons": dict(guard_fail_reasons),
        "n_floor_admitted": len(floor_admit),
        "n_below_floor": len(guard_pass) - len(floor_admit),
        "decoded_class_hist_among_guardpass": {str(k): v for k, v in sorted(
            decoded_hist.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "n_decoded_class_3": decoded_hist.get(3, 0),
        "minibrots_covered": len({r["q4_minibrot_id"] for r in floor_admit}),
    }
    (RUN_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="cap candidates scored (smoke)")
    ap.add_argument("--candidates", default=str(CANDIDATES), help="harvest candidates.json")
    args = ap.parse_args(argv)

    man = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    cands = man["candidates"]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t = ps.t_good_for(FAMILY)

    log(f"=== q4 harvest → v7-decoded ledger  (source={SOURCE_TAG}, t_good={t}) ===")
    log(f"[collect] {len(cands)} harvest candidates over "
        f"{man.get('n_minibrots_covered')} minibrots ({man.get('gate_used')} gate)")

    reframe.DUMP_GUARD_FIELD = True
    assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX
    scorer = guard.make_guarded_scorer(ps.SCORER_PATH)
    rescored_path = RUN_DIR / "rescored.jsonl"

    rescore(cands, scorer, t, rescored_path, limit=args.limit)
    stats = finalize(rescored_path, man, t)

    log("\n=== Q4 HARVEST LEDGER ===")
    log(f"  candidates {stats['n_candidates']} → rendered {stats['n_rendered']} "
        f"(render-fail {stats['n_render_failed']})")
    log(f"  guard: pass {stats['n_guard_pass']} / fail {stats['n_guard_fail']} "
        f"{stats['guard_fail_reasons']}")
    log(f"  v7 FLOOR (p_notbad>=0.5): admitted {stats['n_floor_admitted']} "
        f"/ below-floor {stats['n_below_floor']}  over {stats['minibrots_covered']} minibrots")
    log(f"  (for reference — v7 q3 would keep only {stats['n_decoded_class_3']}: "
        f"decoded_class hist {stats['decoded_class_hist_among_guardpass']})")
    log(f"  durable → {RUN_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
