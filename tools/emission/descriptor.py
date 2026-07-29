"""descriptor.py — location intake: admitted rows → Location, morph embedding, and
incremental morph-cluster assignment.

The admitted-location loader enforces the current-decode predicate
(`corpus_common.is_current_decoded`) — a v6/v5/unstamped row is never consumed as a
current verdict. The canonical morph embedding is the LIBRARY recipe verbatim (a 640×360
ss2 smooth field → `library_annotate.morph_gray_image` robust-z tanh gray →
`colored_clip` CLIP `vit_base_patch16_clip_224.openai`). Clustering is incremental and
WITHIN fractal type (matching the established within-family CLIP dedup convention): a
location joins an existing cluster iff its cosine to the cluster medoid exceeds the strict
near-dup threshold (0.974), else it founds a new cluster.

LIBRARY SEEDING (why `assign_morph_clusters` takes a `library`). The clustering used to
start with an EMPTY medoid list on every call, so an intake batch was deduplicated only
against ITSELF and never against the released library. Every intake therefore adds a seam
across which near-duplicates are never merged, and the error is proportional to the number
of seams — a campaign adds seams. The library's own per-type medoids are now seeded in
before a new batch is clustered, mirroring the discovery side (`deficit_scheduler`'s
`seed_from_library` / `load_library_seed_embeddings`, same 0.974 metric, same CLIP recipe,
so the two seeds are metric-consistent).

The guard: **an existing library row is never re-assigned.** A new row may join a seeded
cluster or found a new one; nothing already in the library moves. That is enforced three
ways: (a) seeded clusters keep their library cluster INDEX, so a join reproduces the
library's own `<type>#<k>` tag; (b) seeded medoids are FROZEN for the pass — a joining row
never displaces or updates one; (c) `verify_library_unmoved` re-checks the produced tags
against the library's own assignment map and raises on any move. (b) is not a new rule: the
incremental medoid has always been the founding member's embedding and has never been
updated by later joins, so freezing a seeded medoid IS the existing semantics applied to a
cluster whose founder happens to predate this batch.

The Location construction + admitted filter + clustering are pure (numpy only); the CLIP
model + `library_annotate` are imported lazily inside `embed_locations` so this module
loads without torch for unit tests.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "corpus"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import corpus_common as cc            # noqa: E402  is_current_decoded / require_current
from tools.corpus import location as loc_mod  # noqa: E402
from tools.corpus import julia_ledger_schema as jls  # noqa: E402  asserted julia (viewport,c) resolve

# Strict near-dup cosine threshold — the established within-family morph-CLIP dedup knee
# (tools/studies/morphology_dedup.py DEFAULT_THRESHOLD). Join a cluster iff cos > this.
NEAR_DUP_THRESHOLD = 0.974

# --------------------------------------------------------------------------- #
# Source-aware admission quality predicate.
#
# The default emission source is a DISCOVERY ledger whose locations were found BY v7
# (the guided-descend reward is v7's q3 verdict), so `decoded_class==3` is both the
# selection signal and the admission gate — self-consistent. A FLOOR source is
# different: its locations were selected by a quality signal ORTHOGONAL to v7 (e.g. the
# q4 goodness field, which is blind to v7 and to the window labels). Gating those on
# v7's own `decoded_class==3` would let v7 silently veto locations it never chose — the
# exact wrong thing. So a floor source is admitted on a v7 BADNESS FLOOR (reject clear
# junk, `p_notbad >= FLOOR_PNOTBAD`) and the human does the quality pick downstream.
# Guard + distinct + current-decode still apply to EVERY source. See
# docs/design/q4_harvest_emission.md.
FLOOR_ADMIT_SOURCES = frozenset({"q4_harvest"})
FLOOR_PNOTBAD = 0.5   # v7 floor: p_notbad = sigma(l0) = P(class>=2); reject clear badness


def source_tag_of(row: dict) -> str | None:
    """Durable per-row source tag: `mix_source` (newer supply producers) else the older
    `_source_tag` intake convention. None when untagged."""
    return row.get("mix_source") or row.get("_source_tag")


def admit_quality(row: dict) -> bool:
    """Source-aware quality predicate. A FLOOR_ADMIT_SOURCES row admits on the v7 badness
    floor (`p_notbad >= FLOOR_PNOTBAD`); every other source on the q3 gate
    (`decoded_class == 3`). Guard/distinct/current are checked by the caller."""
    if source_tag_of(row) in FLOOR_ADMIT_SOURCES:
        return (row.get("p_notbad") or 0.0) >= FLOOR_PNOTBAD
    return row.get("decoded_class") == 3

# auto_maxiter policy (mirror tools/scoring/active_ckpt.py — replicated here to keep this
# module torch-free; it is a pure function of fw).
_FW_HOME = 3.0
_MAXITER_BASE, _MAXITER_K, _MAXITER_MIN, _MAXITER_MAX = 500, 0.30, 200, 8000


def auto_maxiter(fw: float) -> int:
    ratio = _FW_HOME / fw if fw > 0 else 1.0
    lz = math.log2(ratio) if ratio > 0 else 0.0
    val = _MAXITER_BASE * (1.0 + _MAXITER_K * lz)
    return int(max(_MAXITER_MIN, min(_MAXITER_MAX, val)))


# --------------------------------------------------------------------------- #
# Partition (ledger `family`) → render family (mirror steered_frontier.render_family_of).
# --------------------------------------------------------------------------- #
# Phoenix identity resolves ABSENT axes to the classic Ushiki plane (z_{-1}=0), so a legacy
# pre-axis phoenix row keys byte-for-byte as explicit-Ushiki — mirrors
# production_seeder.PHOENIX_*_DEFAULT / row_phoenix_key.
_PHOENIX_C_DEFAULT = (0.5667, 0.0)
_PHOENIX_P_DEFAULT = (-0.5, 0.0)
_PHOENIX_ZM1_DEFAULT = (0.0, 0.0)


def render_family_of(partition: str) -> str:
    if partition == "mandelbrot" or partition in ("multibrot3", "multibrot4", "multibrot5"):
        return partition
    if partition == "phoenix":
        return "phoenix"
    if partition == "julia:mandelbrot":
        return "julia"
    if partition.startswith("julia:multibrot"):
        return "julia_" + partition.split(":", 1)[1]
    raise ValueError(f"unknown partition {partition!r}")


def _phoenix_family_params(row: dict) -> dict:
    """(p, z_{-1}) family_params for a phoenix row, absent axes → Ushiki defaults."""
    def g(kre, kim, default):
        vre, vim = row.get(kre), row.get(kim)
        return (float(vre) if vre is not None else default[0],
                float(vim) if vim is not None else default[1])
    p = g("phoenix_p_re", "phoenix_p_im", _PHOENIX_P_DEFAULT)
    z = g("phoenix_zm1_re", "phoenix_zm1_im", _PHOENIX_ZM1_DEFAULT)
    return {"p_re": repr(p[0]), "p_im": repr(p[1]),
            "zm1_re": repr(z[0]), "zm1_im": repr(z[1])}


def location_of(row: dict) -> loc_mod.Location:
    """Ledger row → canonical Location. Native/phoenix coords are the reframed OUTCOME
    viewport; phoenix rows carry the full (c, p, z_{-1}) parameter point (absent axes →
    Ushiki defaults). Julia twins resolve through the ASSERTED schema tag
    (`julia_ledger_schema.viewport_and_c`): a CAMPAIGN row reads the viewport from
    `outcome_*` and c from `julia_c_*`; a WALK row reads the viewport from `julia_z_*` and c
    from `outcome_*`. An untagged/unknown-tagged julia row raises — no shape inference."""
    fam = render_family_of(row["family"])
    if jls.is_julia_row(row):
        cx, cy, fw_v, c_re, c_im = jls.viewport_and_c(row)   # asserts julia_schema
        fw = float(fw_v)
        return loc_mod.Location(family=fam, cx=str(cx), cy=str(cy), fw=str(fw),
                                maxiter=auto_maxiter(fw), c_re=str(c_re), c_im=str(c_im))
    fw = float(row["outcome_fw"])
    kw = dict(family=fam, cx=str(row["outcome_cx"]), cy=str(row["outcome_cy"]),
              fw=str(fw), maxiter=auto_maxiter(fw))
    if row["family"] == "phoenix":
        cre, cim = row.get("phoenix_c_re"), row.get("phoenix_c_im")
        kw["c_re"] = repr(float(cre)) if cre is not None else repr(_PHOENIX_C_DEFAULT[0])
        kw["c_im"] = repr(float(cim)) if cim is not None else repr(_PHOENIX_C_DEFAULT[1])
        kw["family_params"] = _phoenix_family_params(row)
    return loc_mod.Location(**kw)


# --------------------------------------------------------------------------- #
# Admitted-location loader (current-decode ENFORCED).
# --------------------------------------------------------------------------- #
def load_admitted(ledger_path: Path, require_current: bool = False) -> list:
    """Yield admitted rows from a run-scoped ledger: current-decode ∧ <quality> ∧
    guard_pass ∧ distinct, where <quality> is source-aware (`admit_quality`): the q3 gate
    `decoded_class==3` for a normal discovery source, or the v7 badness floor
    `p_notbad>=FLOOR_PNOTBAD` for a FLOOR_ADMIT_SOURCES row (e.g. `q4_harvest`). With
    `require_current=True` a stale-decoded row RAISES (`cc.StaleDecodeError`) instead of
    being skipped — the strict verdict-trust form used to prove old-ledger rows are
    rejected."""
    rows = []
    for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if require_current:
            cc.require_current(row)       # raises on stale decode
        elif not cc.is_current_decoded(row):
            continue
        if not row.get("guard_pass") or not row.get("distinct"):
            continue
        if not admit_quality(row):        # source-aware: q3 gate OR v7 floor
            continue
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Canonical morph embedding (library recipe).
# --------------------------------------------------------------------------- #
def embed_locations(rows: list, field_cache: Path, embs_path: Path) -> dict:
    """location_id → (L2-normalized morph-CLIP embedding, retained field bin/json paths).

    Renders each location's 640×360 ss2 smooth field once (retained under `field_cache`
    for reuse by the pref palette ranker), grays it via the library robust-z tanh transfer,
    and CLIP-embeds. Persists the embeddings atomically to `embs_path` (npz keyed by id)."""
    import torch  # noqa: F401  (ensures the CUDA context is up before the CLIP load)
    from tools.wallpaper import library_annotate as la
    from tools.curation.colored_clip import load_clip, embed_clip
    from tools import colormap as cm

    field_cache.mkdir(parents=True, exist_ok=True)
    model, tf = load_clip()
    out = {}
    fields = {}
    for row in rows:
        loc = location_of(row)
        field = la.ensure_field(loc, retain=True, tmp_dir=field_cache, cache_root=field_cache)
        gray = la.morph_gray_image(field)
        emb = embed_clip(model, tf, [gray])[0].astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)
        out[row["id"]] = emb
        # remember the retained field path (deterministic stem) for the palette ranker.
        from tools.wallpaper import library_store as store
        stem = store.field_stem(loc, "smooth", la.W, la.H, la.SS)
        fields[row["id"]] = (str(field_cache / f"{stem}.bin"), str(field_cache / f"{stem}.json"))
    _save_embs(out, embs_path)
    return out, fields


def _save_embs(embs: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(embs.keys())
    mat = np.stack([embs[i] for i in ids]) if ids else np.zeros((0, 768), np.float32)
    # tmp MUST end in .npz — np.savez_compressed appends .npz to any other suffix, which
    # would leave os.replace looking for a file numpy never wrote.
    tmp = path.parent / (path.stem + "_tmp.npz")
    np.savez_compressed(tmp, ids=np.array(ids, dtype=object), emb=mat.astype(np.float32))
    import os
    os.replace(tmp, path)


def load_embs(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    z = np.load(path, allow_pickle=True)
    return {str(i): e.astype(np.float32) for i, e in zip(z["ids"], z["emb"])}


# --------------------------------------------------------------------------- #
# Incremental medoid clustering (within type, at the strict near-dup threshold).
# --------------------------------------------------------------------------- #
def _cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cluster_incremental(items: list, threshold: float = NEAR_DUP_THRESHOLD,
                        seed_medoids: list | None = None) -> dict:
    """items: list of (id, emb) IN A STABLE ORDER. Returns id → cluster_key.

    Incremental: for each item, join the existing cluster whose medoid it is most similar
    to IF that cosine exceeds `threshold`, else found a new cluster. The medoid is the
    founding member's embedding (a deterministic incremental medoid) and is NEVER updated
    by a later join.

    `seed_medoids` is a list of `(cluster_key, embedding)` for clusters that ALREADY EXIST
    (the library's per-type medoids). They are pre-loaded ahead of every item, so a new item
    that near-duplicates a library look joins the LIBRARY's cluster and reports the library's
    own key instead of founding a fresh one. Because the medoid is the founder's embedding
    and is never updated, a seeded medoid is frozen by the same rule that already governs an
    in-batch one — seeding introduces no new medoid semantics. New clusters are keyed with
    consecutive integers starting past `max(seed keys)`, so a library key space with gaps or
    an offset (the campaign1/library_intake_2 union offsets one pass past the other) is
    preserved verbatim."""
    seeds = list(seed_medoids or [])
    medoids: list = [e for _k, e in seeds]        # index → medoid embedding (frozen for seeds)
    keys: list = [int(k) for k, _e in seeds]      # index → cluster key
    next_key = (max(keys) + 1) if keys else 0
    assign: dict = {}
    for cid, emb in items:
        best_i, best_cos = -1, -1.0
        for i, med in enumerate(medoids):
            c = _cos(emb, med)
            if c > best_cos:
                best_cos, best_i = c, i
        if best_i >= 0 and best_cos > threshold:
            assign[cid] = keys[best_i]
        else:
            medoids.append(emb)
            keys.append(next_key)
            assign[cid] = next_key
            next_key += 1
    return assign


def assign_morph_clusters(rows: list, embs: dict,
                          threshold: float = NEAR_DUP_THRESHOLD,
                          library: dict | None = None) -> dict:
    """location_id → morph cluster tag `<type>#<k>`, clustering WITHIN each fractal type
    (the within-family dedup convention). Ledger order is the stable incremental order.

    `library` is the existing library's per-type medoids, `{fractal_type: [(k, emb), ...]}`
    as returned by `library_medoids`. When given, each type's clustering is SEEDED with those
    medoids, so a new row that near-duplicates a library look joins the library's cluster
    `<type>#<k>` instead of founding a parallel one across the intake seam. When None (the
    pre-fix behaviour) the batch is deduplicated only against itself.

    Nothing already in the library moves: seeded clusters keep their key, seeded medoids are
    frozen, and only ids in `rows` appear in the returned map. `verify_library_unmoved` is the
    mechanical re-check the callers run."""
    lib = library or {}
    by_type: dict = {}
    for row in rows:
        by_type.setdefault(row["family"], []).append(row["id"])
    tags = {}
    for ftype, ids in by_type.items():
        items = [(i, embs[i]) for i in ids if i in embs]
        assign = cluster_incremental(items, threshold, seed_medoids=lib.get(ftype))
        for i, k in assign.items():
            tags[i] = f"{ftype}#{k}"
    return tags


# --------------------------------------------------------------------------- #
# Library seed: the existing library's per-type medoids, and the never-moved guard.
# --------------------------------------------------------------------------- #
# The library is an emission-driver intake SNAPSHOT: `intake.json` carrying
# `cluster_tags` ({location_id: "<type>#<k>"}, in the stable union order the pass clustered
# in) beside a `morph_embs.npz` in `_save_embs` format. `stage_first_release.py` writes that
# pair for the unioned library, and the driver's own fresh intake writes the identical
# shapes. The medoid of a cluster is its FOUNDING member — the first id, in snapshot order,
# carrying that tag — which is the same definition `campaign1_intake.cluster` and
# `deficit_scheduler.load_library_seed_embeddings` recover.
def library_medoids(intake_path, embs_path) -> dict:
    """`{fractal_type: [(k, medoid_emb), ...]}`, one medoid per existing library cluster,
    ordered by cluster key. Returns {} if either artifact is absent — the caller decides
    whether that is fatal (it MUST be loud: a silently-empty seed is exactly the un-deduped
    seam this seeding exists to close)."""
    ip, ep = Path(intake_path), Path(embs_path)
    if not ip.exists() or not ep.exists():
        return {}
    meta = json.loads(ip.read_text(encoding="utf-8"))
    tags = meta.get("cluster_tags") or {}
    embs = load_embs(ep)
    founder: dict = {}                      # tag → founding location id (snapshot order)
    for loc_id, tag in tags.items():
        if tag not in founder and loc_id in embs:
            founder[tag] = loc_id
    by_type: dict = {}
    for tag, loc_id in founder.items():
        ftype, _, k = tag.rpartition("#")
        e = np.asarray(embs[loc_id], np.float32).reshape(-1)
        by_type.setdefault(ftype, []).append((int(k), e / (np.linalg.norm(e) + 1e-9)))
    return {f: sorted(v, key=lambda kv: kv[0]) for f, v in by_type.items()}


def library_assignments(intake_path) -> dict:
    """`{location_id: "<type>#<k>"}` — the library's own cluster assignment, for the
    never-moved guard. {} if the snapshot is absent."""
    ip = Path(intake_path)
    if not ip.exists():
        return {}
    return dict((json.loads(ip.read_text(encoding="utf-8")).get("cluster_tags") or {}))


class LibraryRowMoved(RuntimeError):
    """An intake pass re-assigned a location that the library had already clustered. That
    rewrites committed library state (reachability, per-cell deficits, the release record's
    morph_cluster column), so it is refused rather than reported."""


# The released library the forward fix seeds against: the emission-driver intake snapshot
# `stage_first_release.py` assembles by unioning the two committed intake passes (campaign1
# offset past library_intake_2, 1387 locations / 1268 clusters). Same two file names the
# driver's own fresh intake writes, so any driver `--out` dir is a valid `--library` too.
DEFAULT_LIBRARY_DIR = ROOT / "scratch" / "first_release"
LIBRARY_INTAKE_NAME = "intake.json"
LIBRARY_EMBS_NAME = "morph_embs.npz"


def load_library_seed(library_dir) -> tuple:
    """`(medoids, prior_assignments, note)` for a library snapshot dir.

    Returns empty maps and an explaining note when the snapshot is absent — the caller MUST
    surface that note. Silence is the failure mode this whole change exists to remove: the
    discovery side's `deficit_scheduler.load_library_seed_embeddings` returns {} on a missing
    artifact and its caller no-ops quietly, so once `scratch/emission/` was wiped its
    library-wide deficit seeding degraded to run-local scarcity with nothing said."""
    d = Path(library_dir)
    ip, ep = d / LIBRARY_INTAKE_NAME, d / LIBRARY_EMBS_NAME
    med = library_medoids(ip, ep)
    if not med:
        missing = [p.name for p in (ip, ep) if not p.exists()]
        why = f"missing {', '.join(missing)}" if missing else "snapshot holds no clusters"
        return {}, {}, (f"LIBRARY SEED ABSENT ({d}: {why}) — this batch is deduplicated "
                        f"against ITSELF ONLY and adds an un-deduped intake seam.")
    prior = library_assignments(ip)
    n = sum(len(v) for v in med.values())
    return med, prior, (f"library seed: {n} medoids over {len(med)} types from {d}")


def verify_library_unmoved(prior: dict, tags: dict) -> None:
    """Raise `LibraryRowMoved` if any location present in BOTH the library assignment
    (`prior`) and this pass's `tags` changed cluster. Cheap, exact, and the only mechanical
    statement of "nothing already in the library moves" — run it on every seeded intake."""
    moved = {i: (prior[i], tags[i]) for i in tags if i in prior and prior[i] != tags[i]}
    if moved:
        sample = ", ".join(f"{i}: {a} -> {b}" for i, (a, b) in list(moved.items())[:5])
        raise LibraryRowMoved(
            f"{len(moved)} library location(s) re-assigned by this intake ({sample}"
            f"{', ...' if len(moved) > 5 else ''}). Existing library rows must never move — "
            f"refusing rather than rewriting committed library state.")
