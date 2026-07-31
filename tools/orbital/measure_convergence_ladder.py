#!/usr/bin/env python
"""§1 — find where the ring score converges, as a function of fw.

PROMOTED from `scratch/rescore/converge.py` on 2026-07-31, together with its output.
This is the producer of `data/orbital/maxiter_convergence_ladder.json`: the 32-atom
cap ladder that the base 500 -> 4000 (x8) raise rests on
(`docs/design/auto_maxiter.md`). It had been sitting in the disposable tree, which
made the only evidence for a production constant one `rm -r scratch/*` from gone.

!! THE MEASUREMENT IS RELATIVE TO A CAP POLICY, so the policy is a PARAMETER here, not
a live read. Every `conv_mult` in the artifact is a multiple of the production cap at
the time of measurement, and the production cap was raised on 2026-07-31 (500 -> 4000).
Reading the live constants (`rc.auto_maxiter`) meant a fresh run measured convergence
against the NEW policy and reported ratios near 1 — a different quantity wearing the
same producer, and no way to repeat the legacy measurement at all.

So `--policy-base` (and its three companions) default to the LIVE policy and can be set
explicitly; `--legacy-policy` selects `(500, 0.30, 200, 8000)`, the policy the committed
artifact was measured under, which makes that measurement repeatable. Output is stamped
with `maxiter_policy_token` either way, and `--out` defaults to a token-named file under
`scratch/orbital/` so a run cannot silently overwrite the committed legacy artifact with
a new-policy one under the same filename. The committed artifact is unchanged.

Sample atoms spanning the full fw range in the pool; for each, raise the cap on
a multiplier ladder (of the production auto_maxiter) until `radial_rings` stops
moving; record the convergent absolute cap. Fit convergent_cap vs fw two ways
(a constant multiple of the production policy, and the production log-form with a
free coefficient) and write the chosen scoring-cap policy to scoring_cap.json.

Measured at 320x180 (the validation fidelity where the references were
calibrated). The cap is an escape-time quantity, independent of spatial res, so
the same policy governs the 64x36 screen.

Run:  uv run python tools/orbital/measure_convergence_ladder.py [--legacy-policy]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
for p in (REPO_ROOT / "tools" / "orbital", REPO_ROOT / "tools" / "explorer",
          REPO_ROOT / "tools" / "descent", REPO_ROOT / "tools" / "corpus",
          REPO_ROOT / "tools"):
    sys.path.insert(0, str(p))

import rescore_lib as rl        # noqa: E402
import field_metrics as fm      # noqa: E402
import render_core as rc        # noqa: E402
import triage_store as ts       # noqa: E402
import location as loc_mod      # noqa: E402  (policy token + LEGACY_MAXITER_POLICY)

POOL = REPO_ROOT / "data" / "orbital" / "screen_pool.jsonl"
SCORES = REPO_ROOT / "data" / "orbital" / "screen_scores.jsonl"
# The committed artifact — a LEGACY-policy measurement. Reachable only via an explicit
# `--out`, never by default, so a new-policy run cannot land on top of it. See the
# module docstring.
COMMITTED_OUT = REPO_ROOT / "data" / "orbital" / "maxiter_convergence_ladder.json"
SCRATCH_DIR = REPO_ROOT / "scratch" / "orbital"

# The cap policy is a PARAMETER, as (base, k, clamp_min, clamp_max).
#   LIVE   — whatever tools/explorer/render_core.py says today (the default).
#   LEGACY — (500, 0.30, 200, 8000): the policy the committed ladder was measured under,
#            named once in tools/corpus/location.py so it cannot drift from the token.
POLICY_LIVE = (rc.MAXITER_BASE, rc.MAXITER_K, rc.MAXITER_MIN, rc.MAXITER_MAX)
POLICY_LEGACY = loc_mod.LEGACY_MAXITER_POLICY

WORKERS = 4
THREADS = 3
FW_MIN_320 = 1e-10          # below this, fw/320 enters f64's quantization regime
LADDER = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]
N_PER_BIN = 3
N_BINS = 10


def policy_maxiter(fw, policy=POLICY_LIVE) -> int:
    """The production-form cap `base * (1 + k*log2(3/fw))`, clamped — under an EXPLICIT
    policy rather than whatever `render_core` says today.

    `policy=POLICY_LIVE` reproduces `rc.auto_maxiter(fw)` exactly (asserted by
    tools/orbital/test_rescore_lib.py). Passing `POLICY_LEGACY` is what makes the
    committed ladder's measurement repeatable after the base 500 -> 4000 raise."""
    base, k, lo, hi = policy
    fwf = float(fw)
    ratio = 3.0 / fwf if fwf > 0 else 1.0
    lz = math.log2(ratio) if ratio > 0 else 0.0
    return int(max(lo, min(hi, base * (1.0 + k * lz))))


def policy_label(policy) -> str:
    """`(token, human)` collapsed to the human half — legacy renders as its four
    constants because its token is the empty string."""
    return fm.describe_policy(loc_mod.maxiter_policy_token(policy))


def load_reachable_pool() -> list[dict]:
    """screen_pool atoms that were actually screenable (present in screen_scores),
    carrying coords + fw. These are the ones a field dump won't reject."""
    coords = {}
    for line in POOL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            coords[r["id"]] = r
    reachable = []
    for line in SCORES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            s = json.loads(line)
            c = coords.get(s["id"])
            if c is not None and float(c["window_scale"]) * 4 >= FW_MIN_320:
                reachable.append(c)
    return reachable


def stratified_sample(rows, n_bins=N_BINS, per_bin=N_PER_BIN, seed=20260730) -> list[dict]:
    fw = np.array([float(r["window_scale"]) * 4 for r in rows])
    lo, hi = np.log10(fw.min()), np.log10(fw.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(np.log10(fw), edges) - 1, 0, n_bins - 1)
    rng = np.random.default_rng(seed)
    picks = []
    for b in range(n_bins):
        members = [rows[i] for i in np.nonzero(idx == b)[0]]
        if not members:
            continue
        take = rng.choice(len(members), size=min(per_bin, len(members)), replace=False)
        picks.extend(members[t] for t in take)
    return picks


def ladder_for_atom(a: dict, policy=POLICY_LIVE) -> dict:
    """Climb the cap ladder for one atom; early-stop after two stable steps.

    Every multiplier is relative to `policy`'s cap at this fw, so the record is only
    meaningful alongside the policy it was measured under — which is why the run stamps
    it (see `main`)."""
    fw = float(a["window_scale"]) * 4.0
    prod = policy_maxiter(fw, policy)
    pts = []
    prev = None
    stable = 0
    for mult in LADDER:
        maxiter = max(64, int(round(mult * prod)))
        try:
            m = rl.measure_both(a["cx"], a["cy"], fw, maxiter, family=a["family"],
                                width=fm.MEASURE_W, height=fm.MEASURE_H,
                                ss=fm.MEASURE_SS, threads=THREADS)
            rings = m["radial_rings"]
        except Exception as e:
            pts.append({"mult": mult, "maxiter": maxiter, "error": str(e)[:120]})
            break
        pts.append({"mult": mult, "maxiter": maxiter, "rings": rings,
                    "cycles_spanned": round(m["cycles_spanned"], 3),
                    "escaped_px": m["escaped_px"]})
        if prev is not None:
            tol = max(1.0, 0.02 * max(rings, prev))
            if abs(rings - prev) <= tol:
                stable += 1
            else:
                stable = 0
        prev = rings
        if stable >= 2:            # two consecutive stable steps → converged
            break
    return {"id": a["id"], "fw": fw, "prod_cap": prod, "period": a.get("period"),
            "log10_abs_A": a.get("log10_abs_A"), "points": pts}


def analyze_ladder(rec: dict) -> dict:
    good = [p for p in rec["points"] if "rings" in p]
    if len(good) < 2:
        rec.update({"converged": False, "conv_mult": None, "conv_maxiter": None,
                    "asymptote": None, "note": "too few points"})
        return rec
    asym = good[-1]["rings"]
    tol = max(1.0, 0.03 * asym)
    conv = next((p for p in good if abs(p["rings"] - asym) <= tol), good[-1])
    last_two = [p["rings"] for p in good[-2:]]
    converged = abs(last_two[-1] - last_two[-2]) <= max(1.0, 0.03 * asym)
    rec.update({
        "converged": bool(converged),
        "asymptote": asym,
        "conv_mult": conv["mult"],
        "conv_maxiter": conv["maxiter"],
        "rings_at_prod": good[0]["rings"],
        "clip_ratio": round(asym / good[0]["rings"], 3) if good[0]["rings"] else None,
        "top_mult_reached": good[-1]["mult"],
    })
    return rec


def fit_and_choose(recs, log, policy=POLICY_LIVE):
    conv = [r for r in recs if r.get("conv_maxiter")]
    fw = np.array([r["fw"] for r in conv])
    cm = np.array([r["conv_maxiter"], ], dtype=float) if False else \
        np.array([r["conv_maxiter"] for r in conv], dtype=float)
    prod = np.array([r["prod_cap"] for r in conv], dtype=float)
    ratio = cm / prod
    x = np.log2(3.0 / fw)                       # production's log-depth variable
    # model 1: constant multiple of production
    mult_mean, mult_med, mult_max = float(ratio.mean()), float(np.median(ratio)), float(ratio.max())
    # model 2: log-form  conv = a + b*x  (a=base, k=b/a)
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, cm, rcond=None)
    pred = A @ np.array([a, b])
    ss_res = float(((cm - pred) ** 2).sum())
    ss_tot = float(((cm - cm.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    k = float(b / a) if a else None

    log(f"\n  convergent-cap fit over {len(conv)} atoms")
    log(f"    conv/prod ratio: mean {mult_mean:.2f}  median {mult_med:.2f}  max {mult_max:.2f}")
    log(f"    log-form conv = {a:.0f} * (1 + {k:.3f}*log2(3/fw))   R^2={r2:.3f}")
    # Report the policy the ratios are RELATIVE TO — reading it off the parameter rather
    # than restating "base=500 k=0.300 (clamp 8000)", which is how this line went stale
    # the moment the production base moved to 4000.
    pb, pk, plo, phi = policy
    log(f"    measured against base={pb} k={pk:.3f} (clamp {plo}-{phi})")

    # chosen policy: an ENVELOPE that does not clip any sampled atom. Use the
    # constant-multiple model at ceil(max ratio) — simplest form that dominates
    # every convergent point — unless the log-form envelope is materially tighter.
    env_mult = float(np.ceil(mult_max))
    clamp_max = int(np.ceil(cm.max() / 1000.0) * 1000) + 2000
    policy = {"policy": "mult_of_prod", "mult_of_prod": env_mult,
              "clamp_max": clamp_max,
              "note": "scoring-only envelope: ceil(max conv/prod ratio) x production cap; "
                      "dominates every sampled atom's convergent cap. Production untouched.",
              "fit": {"ratio_mean": round(mult_mean, 3), "ratio_median": round(mult_med, 3),
                      "ratio_max": round(mult_max, 3),
                      "logform_base": round(float(a), 1), "logform_k": round(k, 4),
                      "logform_r2": round(r2, 4),
                      "n_atoms": len(conv),
                      "n_not_converged": sum(1 for r in conv if not r["converged"])}}
    return policy


def _log(*a):
    print(*a, flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    b, k, lo, hi = POLICY_LIVE
    ap.add_argument("--policy-base", type=int, default=b,
                    help=f"cap policy base (default: LIVE = {b})")
    ap.add_argument("--policy-k", type=float, default=k,
                    help=f"cap policy log-depth coefficient (default: LIVE = {k})")
    ap.add_argument("--policy-clamp-min", type=int, default=lo,
                    help=f"cap policy lower clamp (default: LIVE = {lo})")
    ap.add_argument("--policy-clamp-max", type=int, default=hi,
                    help=f"cap policy upper clamp (default: LIVE = {hi})")
    ap.add_argument("--legacy-policy", action="store_true",
                    help=f"measure against {POLICY_LEGACY} — the policy the committed "
                         f"ladder was measured under; makes that measurement repeatable")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: scratch/orbital/, named for the policy "
                         "token). The committed data/orbital/ artifact is reachable only "
                         "by naming it here.")
    args = ap.parse_args(argv)
    args.policy = (POLICY_LEGACY if args.legacy_policy else
                   (args.policy_base, args.policy_k, args.policy_clamp_min,
                    args.policy_clamp_max))
    if args.out is None:
        token = loc_mod.maxiter_policy_token(args.policy) or "legacy"
        args.out = SCRATCH_DIR / f"maxiter_convergence_ladder__{token}.json"
    return args


def main(argv=None):
    args = parse_args(argv)
    policy = args.policy
    token = loc_mod.maxiter_policy_token(policy)
    _log(f"cap policy: {policy_label(policy)}  token={token!r}")
    if args.out.resolve() == COMMITTED_OUT.resolve() and policy != POLICY_LEGACY:
        raise SystemExit(
            f"refusing to overwrite {COMMITTED_OUT.relative_to(REPO_ROOT).as_posix()} with a "
            f"{policy_label(policy)} measurement: it holds the LEGACY-policy ladder the "
            f"base 500 -> 4000 raise rests on, and the two are different quantities, not "
            f"versions of one. Write the new measurement to its own path.")

    rows = load_reachable_pool()
    # anchor references explicitly (eye + mb19) so the calibration atoms are in-sample
    refs = ts.load_references()
    ref_rows = [{"id": r["id"], "cx": r["cx"], "cy": r["cy"],
                 "window_scale": r["base_scale"], "family": r["family"],
                 "period": r.get("period"), "log10_abs_A": r.get("log10_abs_A"),
                 "label": r["label"]} for r in refs]
    sample = stratified_sample(rows)
    # dedup refs into the sample by id
    have = {a["id"] for a in sample}
    sample += [r for r in ref_rows if r["id"] not in have]
    _log(f"convergence sample: {len(sample)} atoms "
         f"(fw {min(float(a['window_scale'])*4 for a in sample):.2e} .. "
         f"{max(float(a['window_scale'])*4 for a in sample):.2e})")

    t0 = time.time()
    recs = []
    done = [0]

    def work(a):
        r = analyze_ladder(ladder_for_atom(a, policy))
        recs.append(r)
        done[0] += 1
        lbl = a.get("label", a["id"][:10])
        _log(f"  [{done[0]:2d}/{len(sample)}] {lbl:14s} fw={r['fw']:.2e} "
             f"prod={r['prod_cap']:5d} rings {r.get('rings_at_prod')}->{r.get('asymptote')} "
             f"conv_mult={r.get('conv_mult')} conv_cap={r.get('conv_maxiter')} "
             f"{'CONV' if r.get('converged') else 'NOT-CONV'}")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, sample))

    recs.sort(key=lambda r: r["fw"])
    envelope = fit_and_choose(recs, _log, policy)
    # The output carries the policy it was measured under, DERIVED from the parameter —
    # the artifact is meaningless without it, and a hardcoded description is how a
    # metadata file outlives the thing it records.
    doc = {
        "what": "maxiter CONVERGENCE LADDER: per atom, radial_rings measured up a "
                "multiplier ladder of the cap policy below until it stopped moving.",
        "producer": "tools/orbital/measure_convergence_ladder.py",
        "geometry": [fm.MEASURE_W, fm.MEASURE_H, fm.MEASURE_SS],
        fm.POLICY_KEY: token,
        "maxiter_policy": policy_label(policy),
        "not_reproducible_under_current_policy": policy != POLICY_LIVE,
        "ladder": LADDER, "records": recs, "policy": envelope,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    _log(f"\n  fitted envelope: {envelope['policy']} "
         f"mult={envelope['mult_of_prod']} clamp_max={envelope['clamp_max']} "
         f"(reported only — nothing reads it; see docs/design/auto_maxiter.md)")
    _log(f"  wall {time.time()-t0:.0f}s  -> {args.out}")
    # show conv cap vs fw table
    _log("\n  per-atom convergent cap vs fw:")
    _log(f"    {'fw':>11}{'prod':>7}{'rings@prod':>11}{'asym':>8}{'clip×':>7}{'conv_mult':>10}{'conv_cap':>10}  conv?")
    for r in recs:
        if r.get("conv_maxiter"):
            _log(f"    {r['fw']:>11.2e}{r['prod_cap']:>7d}{r['rings_at_prod']:>11.1f}"
                 f"{r['asymptote']:>8.1f}{(r.get('clip_ratio') or 0):>7.2f}"
                 f"{r['conv_mult']:>10.0f}{r['conv_maxiter']:>10d}  {r['converged']}")


if __name__ == "__main__":
    main()
