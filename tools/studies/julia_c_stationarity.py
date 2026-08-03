#!/usr/bin/env python
r"""Julia look-stationarity under a c-perturbation: morph similarity as a FUNCTION of |dc|.

ANALYSIS ONLY. Nothing here writes a sampler, pool or config value; the deliverable is the
curve and a PROPOSED c-spacing floor. Adoption is dictated separately.

WHAT THIS REPLACES. `supply_routing.CSPACING_FLOOR = 1e-2` rests on one bucketed table
(`julia_c_sourcing.md`) computed over a labelled ladder's own pairs: 5 decade buckets, and
the floor read off as "the coarsest bucket boundary where the near-dup rate reaches the
different-atom baseline". Two things that table cannot answer, and this pass is built to:

  1. It is a POINT, not a curve. Half-decade resolution and a viability-screened population
     say whether the knee is at 1e-2 or somewhere between 3e-3 and 3e-2.
  2. Its pairs are whatever the ladder happened to contain — a population selected by a
     labelling batch, with |dc| entangled with atom identity and with each member rendered
     at its OWN viewport. Here every pair is rendered at the SAME z-viewport, so a cosine
     difference is a difference in the JULIA SET, never in the framing.

POPULATION — pairs of VIABLE c at controlled |dc|, from two cohorts.
The production pool is already thinned at 1e-2, so it contains NO sub-floor pairs and cannot
by itself be the population. So:

  region  14 REGIONS (8 spread across the v2 supply pool, 6 centred on near-minibrot atom c's
          so `atom_size` has support) x satellites drawn isotropically inside 10 half-decade
          annuli from 1e-5 to 3e-1. This cohort CARRIES THE CURVE: it is the only source of
          sub-1e-2 pairs at all.
  pool    the whole 539-c v2 supply pool, paired against itself. Every member is production-
          accepted, so this is the unmanufactured cross-check — but it is thinned at 1e-2 and
          therefore has support only AT AND ABOVE the current floor. Where the two cohorts
          overlap they are compared rather than merged.

Cross-region pairs are the "different region, any distance" reference — the analogue of the
different-atom baseline the 1e-2 floor was read against.

THE DEGENERATE-PAIR CONFOUND, and why the screen is not optional. Two solid-black or two
dust frames are near-dup at cosine ~1 for a reason that has nothing to do with c-stationarity,
and they concentrate at SMALL |dc| (a whole annulus lands in the same lake). Reading the
sub-floor bins off unscreened pairs therefore measures the draw, not the look. The primary
curve is over pairs where BOTH members pass the stage-2 viability screen; the unscreened
curve is reported beside it so the size of that effect is visible rather than assumed.

Viability is defined ONLY by that render screen. The eps=0.02 membership-flip test is the
stage-1 screen of one channel and is scale-blind — see the DRAW block below for why using it
as a filter emptied two regions outright. It is kept per c as a covariate.

SUBSTRATE. `library_annotate.{ensure_field, morph_gray_image}` + `colored_clip.{load_clip,
embed_clip}` — imported, never reimplemented (morphology_dedup.md's load-bearing-code
warning). That is the canonical morph_clip substrate the cos >= 0.974 near-dup yardstick
belongs to; the cheap steering-JPG substrate is a different space and its ordering does not
transfer. Cheap render geometry is fine because only ordering and the cut matter, and the
canonical geometry (640x360 ss2) IS the cheap one.

    uv run python -u tools/studies/julia_c_stationarity.py draw
    uv run python -u tools/studies/julia_c_stationarity.py measure   # renders + CLIP (GPU)
    uv run python -u tools/studies/julia_c_stationarity.py analyze

Everything lands under scratch/julia_cstat/ (disposable); field bins are purged per-unit.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "sourcing"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.corpus import location as loc_mod                       # noqa: E402
from tools.studies.q4_neighborhood_sweep import auto_maxiter       # noqa: E402
from tools.studies.q4_c_perturbation import compute_metrics        # noqa: E402
from tools.studies.q4_dM_property import mandel_inside             # noqa: E402
from tools.wallpaper import library_annotate as la                 # noqa: E402

OUT = ROOT / "scratch" / "julia_cstat"
TMP = OUT / "fields"

# --- the near-dup yardstick. NOT re-decided here (morphology_dedup.md §2). --------------
NEAR_DUP = 0.974

# --- draw geometry ---------------------------------------------------------------------
SEED = 20260803
N_POOL_REGIONS = 8            # region centres drawn spread across the v2 supply pool
N_ATOM_REGIONS = 6            # region centres on near-minibrot atom c's (atom_size covariate)
REGION_MIN_SEP = 0.15         # min |dc| between region centres -> regions are independent
SHELL_LOG10 = [-5.0 + 0.5 * k for k in range(10)]   # 1e-5 ... 10^-0.5 = 3.16e-1
PER_SHELL = 5                 # satellites per region per half-decade annulus ...
PER_SHELL_FAR = 8             # ... raised from shell 1e-2 up, where viability thins out
SHELL_EPS = 0.02              # the q4 boundary sampler's stage-1 shell — a COVARIATE here
SHELL_RING_N = 16             # ring samples for the membership-flip test
SCREEN_MAXITER = 2000

# NO draw-time boundary filter, and that is a correction to this pass's own first design.
# The eps=0.02 membership-flip screen is the stage-1 screen of ONE channel (the q4 boundary
# sampler) and it is scale-blind: around a near-minibrot c the 0.02 ring is enormous relative
# to the atom, lands in uniform territory, and the test rejects EVERY draw. Two of eight
# pool-sourced regions came back empty in every sub-1e-1 shell for exactly that reason — and
# near_minibrot is the very channel CSPACING_FLOOR governs. Viability is therefore defined
# only by the stage-2 render screen below, which is scale-free and channel-agnostic; the
# eps-shell verdict is recorded per c as a covariate instead of used as a filter.

# --- the SHARED z-viewports. Both members of every pair render at all three. ------------
# The class favours wide whole-julia framings (julia_c_sourcing.md §Framing), so `wide` is
# the canonical read; `mid` and `off` exist so a knee cannot be a single-framing artifact.
VIEWPORTS = [
    dict(vid="wide", cx="0.0", cy="0.0", fw="1.3"),
    dict(vid="mid", cx="0.0", cy="0.0", fw="0.55"),
    dict(vid="off", cx="0.28", cy="0.20", fw="0.55"),
]

# --- stage-2 viability (julia_c_sourcing.md §screen step 2), applied post-render --------
# NOTE the calibration geometry: these thresholds were set on 768x432 ss1 fields and are
# applied here to the morph field box-downsampled to 640x360. interior_frac is
# resolution-free; mid/occupancy shift mildly. Reported effect: the screened and unscreened
# curves are both in the output, so the screen's contribution is visible, not assumed.
VIAB_INTERIOR_MAX = 0.85
VIAB_DUST_MID = 0.04
VIAB_DUST_OCC = 0.06
VIAB_VIEWPORT = "wide"        # viability is a property of c, read at the canonical view

# --- render fan-out. 3 processes x 3 rayon threads on a 12-core box (CLAUDE.md). --------
WORKERS = 3
ENGINE_THREADS = 3
BATCH = 96                    # renders per CLIP batch / checkpoint

# --- analysis bins ---------------------------------------------------------------------
BIN_EDGES_LOG10 = [-5.5 + 0.5 * k for k in range(12)]   # ... -0.5, 0.0


# --------------------------------------------------------------------------- #
# ∂M covariates                                                                #
# --------------------------------------------------------------------------- #
def mandel_de(cre: float, cim: float, maxiter: int = 4000, bail: float = 1e6) -> float:
    """Exterior distance estimate |z|·ln|z|/|z'| at c; nan if c does not escape.

    Same closed form as `q4_dM_property.mandel_de`, re-stated at this pass's maxiter only
    because that module's is bound to its own PROBE constants. It is the FINE covariate:
    the ring-probe in `signed_dist_dM` floors at radius 5e-4, which cannot resolve the
    sub-floor end of this study at all."""
    c = complex(cre, cim)
    z = 0j
    dz = 0j
    for _ in range(maxiter):
        dz = 2.0 * z * dz + 1.0
        z = z * z + c
        if abs(z) > bail:
            az = abs(z)
            return float(az * math.log(az) / (abs(dz) + 1e-300))
    return float("nan")


def boundary_ok(cre: float, cim: float, eps: float = SHELL_EPS) -> bool:
    """The sampler's stage-1 screen: membership is NON-CONSTANT over {c} ∪ ring(eps).

    Reuses `q4_dM_property.mandel_inside` — a second copy of "is c in M" is exactly the
    harness-constant divergence verification_practice.md §1.8 names."""
    ang = np.linspace(0.0, 2.0 * math.pi, SHELL_RING_N, endpoint=False)
    re = np.concatenate(([cre], cre + eps * np.cos(ang)))
    im = np.concatenate(([cim], cim + eps * np.sin(ang)))
    m = mandel_inside(re, im, SCREEN_MAXITER)
    return bool(m.any() and not m.all())


# --------------------------------------------------------------------------- #
# Stage: draw                                                                  #
# --------------------------------------------------------------------------- #
def _load_pool() -> list:
    p = ROOT / "data/atlas/julia_supply_pool_v2.json"
    if not p.exists():
        raise SystemExit(f"MISSING {p} — rebuild with tools/atlas/build_julia_supply_pool_v2.py")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_atoms() -> list:
    import near_minibrot_julia as nmj
    nuclei, _rep = nmj.load_nuclei()
    if not nuclei:
        raise SystemExit("MISSING near-minibrot nuclei — no roster/harvest candidates on disk")
    return nuclei


def _spread_pick(cands, n, min_sep, rng):
    """Farthest-point-first pick of n candidates at least `min_sep` apart, seeded start."""
    out = [cands[int(rng.integers(len(cands)))]]
    while len(out) < n:
        best, bestd = None, -1.0
        for c in cands:
            d = min(math.hypot(c[0] - o[0], c[1] - o[1]) for o in out)
            if d > bestd:
                best, bestd = c, d
        if best is None or bestd < min_sep:
            break
        out.append(best)
    return out


def stage_draw():
    rng = np.random.default_rng(SEED)
    regions = []

    pool = _load_pool()
    pc = [(float(r["c_re"]), float(r["c_im"])) for r in pool]
    for i, (a, b) in enumerate(_spread_pick(pc, N_POOL_REGIONS, REGION_MIN_SEP, rng)):
        regions.append(dict(region=f"pool{i}", src="supply_pool_v2", c_re=a, c_im=b,
                            atom_size=None, atom_period=None))

    atoms = _load_atoms()
    atoms.sort(key=lambda a: a["size"])
    # log-spaced quantiles of atom size, so the atom-size covariate has real spread.
    qs = np.linspace(0.05, 0.95, N_ATOM_REGIONS)
    for i, q in enumerate(qs):
        a = atoms[int(round(q * (len(atoms) - 1)))]
        th = float(rng.uniform(0.0, 2.0 * math.pi))          # rung 1.0, per the live channel
        regions.append(dict(region=f"atom{i}", src="near_minibrot",
                            c_re=a["cx"] + a["size"] * math.cos(th),
                            c_im=a["cy"] + a["size"] * math.sin(th),
                            atom_size=float(a["size"]), atom_period=int(a["period"])))

    rows = []
    for reg in regions:
        c0 = (reg["c_re"], reg["c_im"])
        rows.append(dict(cid=f"{reg['region']}_c", cohort="region", shell_k=None,
                         shell_log10=None, dc_from_centre=0.0, **reg))
        for k, lg in enumerate(SHELL_LOG10):
            n = PER_SHELL if lg < -2.0 else PER_SHELL_FAR
            for s in range(n):
                # log-uniform radius inside the half-decade annulus, uniform angle
                r = float(10.0 ** rng.uniform(lg - 0.5, lg))
                th = float(rng.uniform(0.0, 2.0 * math.pi))
                rows.append(dict(cid=f"{reg['region']}_s{k}_{s}", cohort="region",
                                 shell_k=k, shell_log10=lg, dc_from_centre=r,
                                 region=reg["region"], src=reg["src"],
                                 c_re=c0[0] + r * math.cos(th),
                                 c_im=c0[1] + r * math.sin(th),
                                 atom_size=reg["atom_size"], atom_period=reg["atom_period"]))

    for i, r in enumerate(pool):
        rows.append(dict(cid=f"pool_{i:04d}", cohort="pool", region="pool", src="supply_pool_v2",
                         shell_k=None, shell_log10=None, dc_from_centre=None,
                         c_re=float(r["c_re"]), c_im=float(r["c_im"]),
                         atom_size=r.get("atom_size"), atom_period=None))

    print(f"covariates: ∂M distance estimate for {len(rows)} c ...")
    for r in rows:
        de = mandel_de(r["c_re"], r["c_im"])
        r["de"] = None if not math.isfinite(de) else de
        r["inside_M"] = bool(mandel_inside(np.array([r["c_re"]]), np.array([r["c_im"]]),
                                           SCREEN_MAXITER)[0])
        r["boundary_ok"] = boundary_ok(r["c_re"], r["c_im"])

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "c_pool.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_reg = sum(1 for r in rows if r["cohort"] == "region")
    n_bd = sum(1 for r in rows if r["boundary_ok"])
    print(f"draw: {len(regions)} regions, {n_reg} region c, {len(rows) - n_reg} pool c; "
          f"{n_bd}/{len(rows)} inside the eps={SHELL_EPS} shell (covariate only) -> {p}")
    print(f"renders to come: {len(rows)} x {len(VIEWPORTS)} = {len(rows) * len(VIEWPORTS)}")


# --------------------------------------------------------------------------- #
# Stage: measure — canonical morph field per (c, viewport) -> CLIP embedding    #
# --------------------------------------------------------------------------- #
def _loc(row, vp) -> loc_mod.Location:
    return loc_mod.Location(family="julia", cx=vp["cx"], cy=vp["cy"], fw=vp["fw"],
                            maxiter=auto_maxiter(float(vp["fw"])),
                            c_re=repr(row["c_re"]), c_im=repr(row["c_im"]))


def _box2(g: np.ndarray, ss: int) -> np.ndarray:
    h, w = g.shape
    oh, ow = h // ss, w // ss
    return g[:oh * ss, :ow * ss].reshape(oh, ss, ow, ss).mean(axis=(1, 3))


def _one(row, vp):
    """(key, morph_gray PIL image, viability metrics|None) for one (c, viewport)."""
    loc = _loc(row, vp)
    field = la.ensure_field(loc, retain=False, tmp_dir=TMP, cache_root=TMP)
    img = la.morph_gray_image(field)
    met = None
    if vp["vid"] == VIAB_VIEWPORT:
        met = compute_metrics(_box2(field.values, field.supersample))
        met = {k: met[k] for k in ("interior_frac", "mid_detail_frac", "occupancy",
                                   "flat_frac", "busy_frac")}
    return f"{row['cid']}|{vp['vid']}", img, met


def stage_measure():
    import os
    os.environ["RAYON_NUM_THREADS"] = str(ENGINE_THREADS)
    from tools.curation.colored_clip import load_clip, embed_clip

    rows = [json.loads(l) for l in (OUT / "c_pool.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    units = [(r, vp) for r in rows for vp in VIEWPORTS]

    emb_path = OUT / "emb.npz"
    done_keys, done_vecs, viab = [], [], {}
    if emb_path.exists():
        z = np.load(emb_path, allow_pickle=False)
        done_keys = list(z["keys"])
        done_vecs = [z["emb"][i] for i in range(z["emb"].shape[0])]
        viab = json.loads((OUT / "viability.json").read_text(encoding="utf-8"))
    have = set(done_keys)
    todo = [u for u in units if f"{u[0]['cid']}|{u[1]['vid']}" not in have]
    print(f"measure: {len(units)} units, {len(have)} done, {len(todo)} to go "
          f"({WORKERS} procs x {ENGINE_THREADS} threads)")
    if not todo:
        return

    TMP.mkdir(parents=True, exist_ok=True)
    # Warm the live maxiter policy on the MAIN thread. `location._active_ckpt()` registers
    # the module in sys.modules BEFORE exec_module, so two worker threads reaching it first
    # get a half-built module and one of them dies on a missing constant. Resolving it once
    # here removes the race from this caller; the loader itself is untouched.
    loc_mod.current_maxiter_policy()
    model, tf = load_clip()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for b0 in range(0, len(todo), BATCH):
            chunk = todo[b0:b0 + BATCH]
            res = list(ex.map(lambda u: _one(*u), chunk))
            vecs = embed_clip(model, tf, [r[1] for r in res])
            for (key, _img, met), v in zip(res, vecs):
                done_keys.append(key)
                done_vecs.append(np.asarray(v, dtype=np.float32))
                if met is not None:
                    viab[key.split("|")[0]] = met
            np.savez(emb_path, keys=np.array(done_keys), emb=np.stack(done_vecs))
            (OUT / "viability.json").write_text(json.dumps(viab), encoding="utf-8")
            n = b0 + len(chunk)
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el:.0f}s  eta {el / max(n, 1) * (len(todo) - n):.0f}s",
                  flush=True)
    print(f"measure done: {len(done_keys)} embeddings -> {emb_path}")


# --------------------------------------------------------------------------- #
# Stage: analyze                                                               #
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _bin_of(dc: float) -> int | None:
    if dc <= 0:
        return None
    lg = math.log10(dc)
    for i in range(len(BIN_EDGES_LOG10) - 1):
        if BIN_EDGES_LOG10[i] <= lg < BIN_EDGES_LOG10[i + 1]:
            return i
    return None


def _curve(pairs, cosv, sel=None):
    """pairs: (i,j,dc,bin). cosv: dict vid -> (npairs,) cosine. -> per-bin stats."""
    out = []
    for b in range(len(BIN_EDGES_LOG10) - 1):
        idx = [t for t, p in enumerate(pairs)
               if p[3] == b and (sel is None or sel[t])]
        if not idx:
            continue
        row = dict(bin=b, lo=10.0 ** BIN_EDGES_LOG10[b], hi=10.0 ** BIN_EDGES_LOG10[b + 1],
                   n=len(idx))
        strict = np.ones(len(idx), dtype=bool)
        for vid, cv in cosv.items():
            c = cv[idx]
            nd = c >= NEAR_DUP
            strict &= nd
            lo, hi = wilson(int(nd.sum()), len(idx))
            row[f"med_{vid}"] = float(np.median(c))
            row[f"nd_{vid}"] = float(nd.mean())
            row[f"nd_{vid}_ci"] = [round(lo, 4), round(hi, 4)]
        lo, hi = wilson(int(strict.sum()), len(idx))
        row["nd_all"] = float(strict.mean())
        row["nd_all_ci"] = [round(lo, 4), round(hi, 4)]
        out.append(row)
    return out


def stage_analyze():
    rows = [json.loads(l) for l in (OUT / "c_pool.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    viab = json.loads((OUT / "viability.json").read_text(encoding="utf-8"))
    z = np.load(OUT / "emb.npz", allow_pickle=False)
    keys = list(z["keys"])
    E = z["emb"].astype(np.float64)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    kidx = {k: i for i, k in enumerate(keys)}
    vids = [v["vid"] for v in VIEWPORTS]

    # only c with a complete viewport set AND a viability read
    live = [r for r in rows
            if all(f"{r['cid']}|{v}" in kidx for v in vids) and r["cid"] in viab]
    print(f"{len(live)}/{len(rows)} c with a complete {len(vids)}-viewport embedding set")

    for r in live:
        m = viab[r["cid"]]
        r["viable"] = bool(m["interior_frac"] <= VIAB_INTERIOR_MAX
                           and not (m["mid_detail_frac"] < VIAB_DUST_MID
                                    and m["occupancy"] < VIAB_DUST_OCC))
    n_viab = sum(1 for r in live if r["viable"])
    print(f"stage-2 viable: {n_viab}/{len(live)} ({n_viab / max(len(live), 1):.1%})")

    V = {v: E[[kidx[f"{r['cid']}|{v}"] for r in live]] for v in vids}

    # ---- pair construction -------------------------------------------------
    # region: within-region pairs, the only source of sub-1e-2 |dc| — carries the curve.
    # pool:   the production pool against itself, subsampled — the >=1e-2 cross-check.
    # cross:  different regions, any distance — the baseline.
    by_reg, pool_idx = {}, []
    for i, r in enumerate(live):
        if r["cohort"] == "pool":
            pool_idx.append(i)
        else:
            by_reg.setdefault(r["region"], []).append(i)

    rng = np.random.default_rng(SEED)

    def _mk(i, j):
        dc = math.hypot(live[i]["c_re"] - live[j]["c_re"], live[i]["c_im"] - live[j]["c_im"])
        return (i, j, dc, _bin_of(dc))

    pairs = []
    for idxs in by_reg.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                p = _mk(idxs[a], idxs[b])
                if p[3] is not None:
                    pairs.append(p)

    pool_pairs = []
    for a in range(len(pool_idx)):
        for b in range(a + 1, len(pool_idx)):
            p = _mk(pool_idx[a], pool_idx[b])
            if p[3] is not None:
                pool_pairs.append(p)
    if len(pool_pairs) > 40000:
        keep = rng.choice(len(pool_pairs), 40000, replace=False)
        pool_pairs = [pool_pairs[t] for t in keep]

    regs = sorted(by_reg)
    cross = []
    for a in range(len(regs)):
        for b in range(a + 1, len(regs)):
            ia, ib = by_reg[regs[a]], by_reg[regs[b]]
            for _ in range(min(400, len(ia) * len(ib))):
                p = _mk(ia[int(rng.integers(len(ia)))], ib[int(rng.integers(len(ib)))])
                cross.append(p if p[3] is not None else (p[0], p[1], p[2],
                                                         len(BIN_EDGES_LOG10) - 2))

    def cosines(pl):
        I = np.array([p[0] for p in pl], dtype=np.intp)
        J = np.array([p[1] for p in pl], dtype=np.intp)
        return {v: np.einsum("ij,ij->i", V[v][I], V[v][J]) for v in vids}

    def viab_mask(pl):
        return np.array([live[i]["viable"] and live[j]["viable"] for i, j, _, _ in pl],
                        dtype=bool)

    cw, cp, cc = cosines(pairs), cosines(pool_pairs), cosines(cross)
    viable_w, viable_p, viable_c = viab_mask(pairs), viab_mask(pool_pairs), viab_mask(cross)

    curve_all = _curve(pairs, cw)
    curve_via = _curve(pairs, cw, sel=viable_w)
    curve_pool = _curve(pool_pairs, cp, sel=viable_p)

    def _ref(pl, cv, sel):
        idx = np.where(sel)[0]
        d = dict(n=int(len(idx)))
        strict = np.ones(len(idx), dtype=bool)
        for v in vids:
            c = cv[v][idx]
            nd = c >= NEAR_DUP
            strict &= nd
            d[f"med_{v}"] = float(np.median(c))
            d[f"nd_{v}"] = float(nd.mean())
        d["nd_all"] = float(strict.mean())
        return d

    ref = dict(
        cross_region_all=_ref(cross, cc, np.ones(len(cross), bool)),
        cross_region_viable=_ref(cross, cc, viable_c),
    )

    # ---- the floor read ----------------------------------------------------
    base = ref["cross_region_viable"][f"nd_{VIAB_VIEWPORT}"]
    key = f"nd_{VIAB_VIEWPORT}"

    def _first(pred):
        for row in curve_via:
            if pred(row):
                return row
        return None

    # The SAME rule the 1e-2 floor was read under: first bin whose point estimate reaches
    # the baseline. `indistinct` is the weaker read — the first bin that cannot be told
    # from baseline at 95% — and it fires one or more bins FINER, so the two bracket the knee.
    strict_floor = _first(lambda r: r[key] <= base)
    indistinct = _first(lambda r: r[key + "_ci"][0] <= base)

    # ---- what a floor COSTS, measured on the pool the floor governs --------
    # Greedy first-wins thinning of the committed v2 supply pool at each candidate floor,
    # in the pool's own file order — the same rule `supply_routing.cspacing_ok` applies.
    # A near-dup rate alone cannot choose a floor: every value of it is bought with pool
    # size, and the coarse end of this curve empties the channel.
    pool_rows = [r for r in rows if r["cohort"] == "pool"]
    cost = []
    for b in range(len(BIN_EDGES_LOG10) - 1):
        f = 10.0 ** BIN_EDGES_LOG10[b]
        acc = []
        for r in pool_rows:
            if all(math.hypot(r["c_re"] - a[0], r["c_im"] - a[1]) >= f for a in acc):
                acc.append((r["c_re"], r["c_im"]))
        row = dict(floor=f, pool_survivors=len(acc), pool_n=len(pool_rows), n_pairs=0)
        # ...against the near-dup rate in the bin that STARTS at that floor — the CLOSEST
        # pairs the floor still admits, which is the pairs its whole job is to make distinct.
        # Deliberately not an "all pairs >= f" aggregate: this draw is uniform in log|dc| by
        # construction, so any aggregate over bins reports the draw's shape, not a pool's.
        for c in curve_via:
            if c["bin"] == b:
                row.update({f"nd_{VIAB_VIEWPORT}_at_floor": c[f"nd_{VIAB_VIEWPORT}"],
                            "nd_all_at_floor": c["nd_all"], "n_pairs": c["n"]})
        cost.append(row)

    # ---- covariate read (ONE read, not a model) ----------------------------
    def covariate_split(key, label):
        vals = [r[key] for r in live if r.get(key) is not None]
        if len(vals) < 20:
            return None
        med = float(np.median(vals))
        out = {"key": key, "label": label, "median": med, "bins": []}
        for b in range(len(BIN_EDGES_LOG10) - 1):
            sel_lo, sel_hi = [], []
            for t, (i, j, dc, bn) in enumerate(pairs):
                if bn != b or not viable_w[t]:
                    continue
                a_, b_ = live[i].get(key), live[j].get(key)
                if a_ is None or b_ is None:
                    continue
                (sel_lo if 0.5 * (a_ + b_) < med else sel_hi).append(t)
            if len(sel_lo) < 15 or len(sel_hi) < 15:
                continue
            out["bins"].append(dict(
                bin=b, lo=10.0 ** BIN_EDGES_LOG10[b], hi=10.0 ** BIN_EDGES_LOG10[b + 1],
                n_low=len(sel_lo), n_high=len(sel_hi),
                nd_low=float((cw[VIAB_VIEWPORT][sel_lo] >= NEAR_DUP).mean()),
                nd_high=float((cw[VIAB_VIEWPORT][sel_hi] >= NEAR_DUP).mean()),
            ))
        return out

    covs = [c for c in (covariate_split("de", "∂M distance estimate (exterior c)"),
                        covariate_split("atom_size", "atom size (near-minibrot regions)"))
            if c]

    result = dict(
        measured_at="2026-08-03",
        substrate=dict(producer=la.MORPH_PRODUCER, geometry=f"{la.W}x{la.H}ss{la.SS}",
                       clip="vit_base_patch16_clip_224.openai", near_dup_cos=NEAR_DUP),
        viewports=VIEWPORTS,
        population=dict(n_c=len(live), n_viable=n_viab, n_regions=len(by_reg),
                        n_pairs_region=len(pairs), n_pairs_pool=len(pool_pairs),
                        n_pairs_cross=len(cross),
                        screen=dict(stage1=f"membership non-constant over ring(eps={SHELL_EPS})",
                                    stage2=f"interior<={VIAB_INTERIOR_MAX} and not "
                                           f"(mid<{VIAB_DUST_MID} and occ<{VIAB_DUST_OCC})")),
        curve_viable_pairs=curve_via,
        curve_all_pairs=curve_all,
        curve_pool_cohort=curve_pool,
        reference=ref,
        floor_read=dict(
            rule="first half-decade bin (fine->coarse) whose near-dup rate at the canonical "
                 "viewport reaches the cross-region baseline",
            baseline=base,
            reaches_baseline_at=None if strict_floor is None else strict_floor["lo"],
            reaches_baseline_bin=None if strict_floor is None else
            [strict_floor["lo"], strict_floor["hi"], strict_floor[key], strict_floor["n"]],
            indistinct_from_baseline_at=None if indistinct is None else indistinct["lo"],
        ),
        floor_cost=cost,
        covariates=covs,
    )
    (OUT / "stationarity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ---- print -------------------------------------------------------------
    print(f"\n{'=' * 92}\njulia look-stationarity vs |dc| — VIABLE pairs "
          f"(both members pass the stage-2 screen)\n{'=' * 92}")
    hdr = f"{'|dc| bin':>18} {'n':>6}"
    for v in vids:
        hdr += f"  {'med_' + v:>9} {'nd_' + v:>7}"
    print(hdr + f"  {'nd_all':>7}")
    for r in curve_via:
        line = f"{r['lo']:>8.1e}-{r['hi']:<8.1e} {r['n']:>6}"
        for v in vids:
            line += f"  {r['med_' + v]:>9.4f} {r['nd_' + v]:>7.3f}"
        print(line + f"  {r['nd_all']:>7.3f}")
    rv = ref["cross_region_viable"]
    line = f"{'cross-region ref':>18} {rv['n']:>6}"
    for v in vids:
        line += f"  {rv['med_' + v]:>9.4f} {rv['nd_' + v]:>7.3f}"
    print(line + f"  {rv['nd_all']:>7.3f}")
    if curve_pool:
        print(f"\n-- v2 supply-pool cohort (production-accepted c, thinned at 1e-2) --")
        for r in curve_pool:
            line = f"{r['lo']:>8.1e}-{r['hi']:<8.1e} {r['n']:>6}"
            for v in vids:
                line += f"  {r['med_' + v]:>9.4f} {r['nd_' + v]:>7.3f}"
            print(line + f"  {r['nd_all']:>7.3f}")
    print(f"\nbaseline (cross-region, viable, {VIAB_VIEWPORT}) = {base:.4f}")
    print(f"  rate reaches baseline first at |dc| >= "
          f"{result['floor_read']['reaches_baseline_at']}")
    print(f"  indistinct from baseline (95% CI) from |dc| >= "
          f"{result['floor_read']['indistinct_from_baseline_at']}")
    print(f"\n-- what each candidate floor buys, and what it costs the pool --")
    print(f"{'floor':>9} {'nd_wide@f':>10} {'nd_all@f':>9} {'pairs':>7} "
          f"{'pool kept':>10}  (of {len(pool_rows)} committed v2 c)")
    for r in cost:
        nw = r.get(f"nd_{VIAB_VIEWPORT}_at_floor")
        na = r.get("nd_all_at_floor")
        print(f"{r['floor']:>9.1e} {('-' if nw is None else f'{nw:.3f}'):>10} "
              f"{('-' if na is None else f'{na:.3f}'):>9} "
              f"{r['n_pairs']:>7} {r['pool_survivors']:>10}")
    for c in covs:
        print(f"\ncovariate {c['label']} (split at median {c['median']:.3g}):")
        for b in c["bins"]:
            print(f"  {b['lo']:.1e}-{b['hi']:.1e}  low n={b['n_low']:>4} nd={b['nd_low']:.3f}"
                  f"   high n={b['n_high']:>4} nd={b['nd_high']:.3f}")
    print(f"\nwrote {OUT / 'stationarity.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["draw", "measure", "analyze", "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("draw", "all"):
        stage_draw()
    if a.stage in ("measure", "all"):
        stage_measure()
    if a.stage in ("analyze", "all"):
        stage_analyze()
