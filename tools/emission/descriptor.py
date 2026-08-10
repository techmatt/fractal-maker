"""descriptor.py — location intake: admitted rows → Location, morph embedding, and
incremental morph-cluster assignment.

The admitted-location loader cuts on the row's RAW P(>=3) against `floors.GOOD_FLOOR`,
read at read time (it used to enforce a frozen `decoded_class` plus a decode-VERSION
predicate; both retired 2026-08-09). The canonical morph embedding is the LIBRARY recipe verbatim (a 640×360
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
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "corpus", ROOT / "tools" / "scoring"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from partitions import base_partition, partition_of_row  # noqa: E402  THE partition resolver
from tools.corpus import location as loc_mod  # noqa: E402
from tools.corpus import julia_ledger_schema as jls  # noqa: E402  asserted julia (viewport,c) resolve
from tools.emission import floors as F  # noqa: E402  THE cut owner (GOOD_FLOOR)

# Strict near-dup cosine threshold — the established within-family morph-CLIP dedup knee
# (tools/studies/morphology_dedup.py DEFAULT_THRESHOLD). Join a cluster iff cos > this.
NEAR_DUP_THRESHOLD = 0.974

# --------------------------------------------------------------------------- #
# THE FLOOR-ADMIT BYPASS.
#
# The default emission source is a DISCOVERY ledger whose locations were found BY the active
# scorer, so applying that scorer's own `floors.GOOD_FLOOR` to them is self-consistent. A
# FLOOR-ADMIT source is different: its locations were selected by a quality signal ORTHOGONAL
# to the scorer — a HUMAN label (`human_q3plus`), or the q4 goodness field (`q4_harvest`), both
# blind to the scorer and to the window labels. Gating those on the scorer's own verdict would
# let it silently veto locations it never chose. Guard + distinct still apply to EVERY source.
# See docs/design/q4_harvest_emission.md.
#
# THE QUALITY CUT ITSELF MOVED OUT (2026-08-09, prompts/selection_restructure_3.md). It was
# `admit_quality`, a source-aware predicate over the STORED `decoded_class >= 3` — a class
# frozen into the row at harvest time against that day's per-partition `t_good`. Both halves
# of that are gone: there is no per-partition table, and no reader consumes a frozen verdict.
# `load_admitted` applies `floors.passes_good_floor` to the raw `p_good` directly, with the
# bypass below, and the read-time ranked intake supplies its own predicate through the same
# reader (`ranked_intake.admit`, at the junk floor). One comparison, two heights, no verdicts.
#
# THE BADNESS FLOOR IS GONE (2026-08-04). A floor-admit source used to still face a machine
# BADNESS floor — `p_notbad >= FLOOR_PNOTBAD` (0.5) — on the reading that "reject clear junk"
# was a weaker claim than "judge quality" and so was safe to keep. It is not a weaker claim,
# it is the same claim at a lower threshold, and it is made by the same head the source was
# chosen independently of. Two things made keeping it indefensible:
#
#   * It vetoed the authority. `human_q3plus` rows carry a human 3 or 4. A machine
#     `p_notbad < 0.5` on such a row is a disagreement between the head and Matt, and the
#     floor resolved it for the head — silently, at intake, on material selected precisely
#     because the head had never judged it.
#   * The number never survived its own head. 0.5 was chosen on the v7 `p_notbad` scale and
#     was still being applied under v10. Measured on the q4_harvest ledger's 108
#     guard-passing rows: the v7-era floor admitted 75; the same 0.5 against the v10 rescore
#     admitted 57. The cut moved by 18 rows without a decision being made about it — which is
#     exactly what an unstamped floor does, and why every cut that remains lives in
#     `tools/emission/floors.py` carrying the head version it was set against.
#
# So a floor-admit row is admitted on guard ∧ distinct alone; the human does
# the quality pick downstream off the release sheet, which is what the rule always said.
# `FLOOR_PNOTBAD` was DELETED rather than set to 0.0 — a zero floor is still a floor, still
# reads as a policy somebody chose, and would still be re-tuned by the next person who found
# it. There is no machine badness cut on a floor-admit source.
FLOOR_ADMIT_SOURCES = frozenset({"q4_harvest", "human_q3plus"})


def source_tag_of(row: dict) -> str | None:
    """Durable per-row source tag: `mix_source` (newer supply producers) else the older
    `_source_tag` intake convention. None when untagged."""
    return row.get("mix_source") or row.get("_source_tag")


# auto_maxiter policy — IMPORTED from the owning module, never re-transcribed. This used to
# be a hand-copied mirror justified by "keep this module torch-free", and it went stale:
# it still carried base 500 / clamp 8000 after production raised them to 4000 / 67000 on
# 2026-07-31 (docs/design/auto_maxiter.md), so every Location this module minted for the
# emission intake got a cap 8x too low. The torch-free premise was wrong anyway —
# `production_pins` imports only math/sys/pathlib and defers `score_lib` to inside
# `make_scorer`. Re-exported under this module's own name because callers and tests reach
# it as `descriptor.auto_maxiter`; pinned by tools/scoring/test_maxiter_policy.py.
# Imported BARE (`from production_pins import`), not as `tools.scoring.production_pins`:
# the two spellings produce two distinct module objects, and the ~41 modules that reach the
# pins through `active_ckpt` all use the bare one. Matching them is what makes this the same
# function object rather than a second copy that merely agrees.
from production_pins import auto_maxiter  # noqa: E402,F401


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
    """Partition -> Rust render family. A DERIVED partition has no render family of its own,
    so it goes through `partitions.base_partition` first: `phoenix:classic` is the same Rust
    `phoenix` backend at a pinned parameter point, and raising on it would make the whole
    partition unrenderable the moment it became addressable on the cell axis."""
    partition = base_partition(partition)
    if partition == "mandelbrot" or partition in ("multibrot3", "multibrot4", "multibrot5"):
        return partition
    if partition == "phoenix":
        return "phoenix"
    if partition == "julia:mandelbrot":
        return "julia"
    if partition.startswith("julia:multibrot"):
        return "julia_" + partition.split(":", 1)[1]
    raise ValueError(f"unknown partition {partition!r}")


# --------------------------------------------------------------------------- #
# Cell identity vs clustering geometry — the two things `row["family"]` used to be.
#
# The emission cell's first axis is NAMED `fractal_type` and has always actually held
# `row["family"]`, i.e. the ledger's partition key. That was exact for the nine base
# partitions and wrong for exactly one: a classic-phoenix row carries `family == "phoenix"`,
# so `phoenix:classic` could not be addressed, weighted or seeded — the phoenix cell absorbed
# it. The fix is a READER-side re-key through `partitions.partition_of_row`; nothing on disk
# changes.
#
# But cell identity and clustering geometry are different questions, and only the first one
# moves. Clustering stays WITHIN the BASE partition, so a classic row is still compared
# against (and may still join) its varied-phoenix morphological neighbours exactly as today;
# what changes is which CELL the resulting cluster index names. So one cluster index k can
# surface as two cells, `phoenix#k` and `phoenix:classic#k`, which is the point: they are
# different supply with different release shares living in the same morphology.
# --------------------------------------------------------------------------- #
def cell_partition(row: dict) -> str:
    """The partition that keys a row's CELL identity (`phoenix:classic` included). Raises on
    a row whose family token is not a registered partition — a cell axis that can invent a
    key is a cell axis nothing downstream has a target, floor or census row for."""
    part = partition_of_row(row)
    if part is None:
        raise ValueError(
            f"row {row.get('id')!r} has family token "
            f"{(row.get('fractal_type') or row.get('family'))!r}, which is not a registered "
            f"partition (partitions.ALL_FAMS) — refusing to key a cell on it.")
    return part


def cluster_group(row: dict) -> str:
    """The morphology CLUSTERING group: the base partition. Identical to the old
    `row["family"]` grouping for every row, including classic phoenix."""
    return base_partition(cell_partition(row))


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
# Re-score sibling records (READER-RESOLVED; the original ledger is never rewritten).
# --------------------------------------------------------------------------- #
# A ledger's `p_good` is ONE HEAD's P(>=3), on a train-prior-calibrated scale. When the pin
# moves the number stays readable but stops meaning what the floors were set against, so it
# is re-derived. The re-score CANNOT be an in-place edit: a discovery ledger is the run's own
# record of what it found and what the head of the day said about it, and overwriting that
# erases the only evidence of the previous head's operating point. So a re-score is a SIBLING
# record, keyed by the ledger stem AND the head version:
#
#     data/discovery/campaign1/breadth/outcome_ledger.jsonl
#     data/discovery/campaign1/breadth/outcome_ledger.rescored_v10.jsonl
#
# and the READER overlays it (`resolve_rows`). Two properties earn the version in the name.
# (a) It cannot collide with the two existing `rescored.jsonl` files, which are RESUME STATE
# for their own producers and not overlays:
#         data/discovery/classic_phoenix/rescored.jsonl   (classic_phoenix_supply, 184 rows
#                                                          = one per coord, not per ledger row)
#         data/emission/q4_harvest/rescored.jsonl         (q4_harvest_ledger)
#     Worth naming the paths, because the resemblance misleads on sight: a 2026-08-06 census
#     read classic's as a stale-convention overlay and a cleanup pass proposed renaming it to
#     `outcome_ledger.rescored_v10.jsonl`. That would feed 184 resume rows to `resolve_rows`
#     as a rescore of a 24-row ledger AND break its producer's resume. Classic has no overlay
#     because it does not need one — it re-mints under the live pins every run and purges any
#     row not stamped with the active version, so it arrives current instead of being patched.
# (b) It is fail-correct across the NEXT flip in the sense that still applies: v12 looks for
# `rescored_v12.jsonl`, does not find it, and falls through to the v11 probabilities — which
# rank the pool worse rather than emptying it. That IS the change of 2026-08-09: the old
# behaviour was to reject every un-re-scored row outright, which read as "the corpus vanished"
# and cost a GPU pass to undo. The re-score is now an accuracy job, and what makes a flip
# CORRECT rather than merely non-fatal is re-scoring AND volume-matching the two floors
# together (`floors.GOOD_FLOOR` / `JUNK_FLOOR`; classifier_retrain_protocol.md section 5).
RESCORE_SUFFIX_FMT = ".rescored_{version}.jsonl"


def _active_scorer_version() -> str:
    """The live checkpoint's version token, for NAMING the sibling — never for judging a row.

    Resolved lazily so this module stays importable while a pin is mid-flip. It used to come
    from `corpus_common.active_scorer_version`, which sat beside the decode-version predicate
    family that was deleted on 2026-08-09; only the naming use survived the deletion."""
    from production_pins import ACTIVE_VERSION  # noqa: PLC0415
    return ACTIVE_VERSION


def rescore_path(ledger_path, version: str | None = None) -> Path:
    """The sibling re-score record for `ledger_path` under `version` (default: the ACTIVE
    head). Pure path arithmetic — says nothing about whether the file exists."""
    p = Path(ledger_path)
    v = version or _active_scorer_version()
    return p.with_name(p.stem + RESCORE_SUFFIX_FMT.format(version=v))


def load_rescored(ledger_path, version: str | None = None) -> dict:
    """`{id: re-scored row}` for `ledger_path`, or {} when no record exists for `version`.

    Absence is NOT an error here: a ledger with no re-score simply keeps its own rows, which
    the current-decode predicate then judges. The loud failure lives one level up, where a
    caller that needs a seeded/current population says so."""
    rp = rescore_path(ledger_path, version)
    if not rp.exists():
        return {}
    out = {}
    for line in rp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            out[row["id"]] = row
    return out


def resolve_rows(ledger_path, version: str | None = None) -> list:
    """The ledger's rows in ledger order, each overlaid with its current-version re-score.

    The overlay is a whole-row merge (`{**original, **rescored}`), not a decode-field splice:
    the re-score record carries complete rows, so a reader of the sibling alone sees the same
    thing the overlay produces. The ORIGINAL file is only ever read. A re-score row whose id
    is not in the ledger is ignored — the ledger defines the population."""
    over = load_rescored(ledger_path, version)
    rows = []
    for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        hit = over.get(row["id"])
        rows.append({**row, **hit} if hit else row)
    return rows


# --------------------------------------------------------------------------- #
# Admitted-location loader (current-decode ENFORCED).
# --------------------------------------------------------------------------- #
def guard_and_distinct(row: dict) -> bool:
    """The POPULATION half of admission: the run's own degenerate-outcome guard and its own
    morphology dedup. Neither is a head verdict — they are properties of the location and of
    the cloud it was found in — so they hold on EVERY intake path, including the read-time
    ranked one that takes no decode-version predicate (`ranked_intake`)."""
    return bool(row.get("guard_pass")) and bool(row.get("distinct"))


def load_admitted(ledger_path: Path, admit=None) -> list:
    """Yield admitted rows from a run-scoped ledger: guard_pass ∧ distinct ∧ (the row's raw
    P(>=3) clears `floors.GOOD_FLOOR`, OR the row is from a FLOOR_ADMIT source and bypasses
    the machine verdict entirely — see the block above `FLOOR_ADMIT_SOURCES`).

    Rows come through `resolve_rows`, so a ledger carrying a sibling re-score record for the
    ACTIVE head is judged on the RE-SCORED probability. Without one the original probability
    is read verbatim — which is a number on an older head's scale and therefore a worse
    estimate, not an inadmissible one. Until 2026-08-09 it WAS inadmissible: a decode-version
    predicate refused every row an older head had stamped, and the v10 flip consequently took
    this reader from ~1.4k rows to 16 with nothing going red.

    `admit` REPLACES the whole predicate (quality AND guard/distinct) with a caller-supplied
    `row -> bool`. It exists so the READ-TIME ranked intake (`ranked_intake.py`) shares this
    one reader — and therefore `resolve_rows`, the namespacing, the location dedup and the
    diagnostics — instead of growing a second union walker that could disagree with this one
    about what the population is."""
    rows = []
    for row in resolve_rows(ledger_path):
        if admit is not None:
            if admit(row):
                rows.append(row)
            continue
        if not guard_and_distinct(row):
            continue
        if source_tag_of(row) not in FLOOR_ADMIT_SOURCES and \
                not F.passes_good_floor(row.get("p_good")):
            continue
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# THE cross-ledger union (id namespaced per ledger, locations deduped by identity).
# --------------------------------------------------------------------------- #
# A ledger's `id` is RUN-SCOPED. `st_<fam>_<arm>_<seq>` is minted per campaign, so campaign1
# and campaign2 both hold `st_m_breadth_000039` naming DIFFERENT locations — 11 such pairs
# across the seven intake ledgers. A union keyed on the bare id therefore had to choose
# between silently dropping a distinct wallpaper and aborting, and it correctly aborted; the
# whole stage-2 intake has been unreachable since.
#
# The fix is at the READER and it is the standing normalize-at-the-reader pattern: row
# identity is NAMESPACED BY LEDGER at union time, so two run-scoped ids can never alias. No
# ledger row is rewritten and no prefixed COPY of a ledger is minted (the `c1__` scheme did
# that, wrote the copies to `scratch/`, and lost them).
#
# Deduplication does not disappear, it moves onto the axis that can actually carry it: a row
# is dropped iff its LOCATION IDENTITY (`loc_key`, unchanged) was already admitted by an
# earlier ledger. That is the genuine cross-ledger overlap case — 0 of it today, and it stays
# dedupable in principle rather than being traded away for the collision fix.
def loc_key(row: dict) -> tuple:
    """The location identity of a ledger row — what its id is SUPPOSED to name.
    THE one copy (`build_emission_diversity_v1` and `ledger_rescore` both import it; two
    unions that disagree about what "the same location" means are two different populations)."""
    return (str(row.get("outcome_cx")), str(row.get("outcome_cy")), str(row.get("outcome_fw")),
            str(row.get("julia_c_re")), str(row.get("julia_c_im")))


def ledger_namespace(ledger_path) -> str:
    """The id namespace for one ledger: its repo-relative path, slugified, `.jsonl` and the
    leading `data/` dropped. `data/discovery/campaign1/breadth/outcome_ledger.jsonl` ->
    `discovery_campaign1_breadth_outcome_ledger`.

    The whole path including the STEM, even though six of the seven ledgers are named
    `outcome_ledger` and the stem looks like noise there. Two ledgers in one directory is an
    ordinary thing (`outcome_ledger.jsonl` beside `outcome_ledger_v7_t45.jsonl` already exists
    in the tree), and a namespace that collides is a namespace that does not namespace. The
    ids are long; they are machine keys, and STABILITY is what they have to be — a namespace
    derived from the companion ledgers in the union would change a location's identity
    depending on what it was unioned with.

    Out-of-tree (a fixture) falls back to the parent directory name plus the stem."""
    p = Path(ledger_path).resolve()
    try:
        rel = p.relative_to(ROOT)
        parts = list(rel.parent.parts) + [rel.stem]
    except ValueError:
        parts = [p.parent.name, p.stem]
    if parts and parts[0] == "data":
        parts = parts[1:]
    slug = "_".join(x for x in parts if x)
    return slug or "ledger"


NS_SEP = "__"      # the `stage_first_release` `c1__` spelling, kept so the shape is familiar


def namespaced_id(namespace: str, row_id: str) -> str:
    return f"{namespace}{NS_SEP}{row_id}"


class LedgerNamespaceCollision(RuntimeError):
    """Two ledgers in one union slugged to the same id namespace, so namespacing would not
    actually separate their run-scoped ids. Refused rather than reported: this is the exact
    failure the namespacing exists to prevent, one level up."""


def load_union_admitted(ledger_paths, keep_row_id: bool = True, admit=None) -> tuple:
    """`(rows, diag)` — the admitted union over `ledger_paths`, in ledger order.

    `admit` is passed straight through to `load_admitted` (see there): the read-time ranked
    intake supplies its own predicate so both intake paths are THE same union reader.

    Each returned row is the ledger row with its `id` replaced by the ledger-namespaced id
    (the original kept under `_ledger_row_id`) plus `_source_ledger` / `_ledger_ns`. The
    ORIGINAL FILES ARE ONLY EVER READ.

    `diag` carries what a census wants to print: `n_union`, `per_ledger` admitted counts,
    `n_location_overlaps` (rows dropped because an earlier ledger already admitted the same
    `loc_key`) with a sample, and `n_id_collisions` — the number of bare ids that would have
    aliased two DIFFERENT locations under the old id-keyed union, reported so the fix stays
    visible instead of the count silently changing."""
    namespaces: dict = {}
    for lp in ledger_paths:
        ns = ledger_namespace(lp)
        if ns in namespaces and Path(namespaces[ns]).resolve() != Path(lp).resolve():
            raise LedgerNamespaceCollision(
                f"ledgers {namespaces[ns]} and {lp} both slug to id namespace {ns!r}; "
                f"namespacing would not separate their run-scoped ids.")
        namespaces[ns] = lp

    seen_loc: dict = {}          # loc_key -> (namespaced id, ledger label)
    bare_ids: dict = {}          # bare row id -> loc_key (collision census only)
    rows, overlaps, collisions = [], [], []
    per_ledger: dict = {}
    for lp in ledger_paths:
        lp = Path(lp)
        ns = ledger_namespace(lp)
        try:
            label = str(lp.resolve().relative_to(ROOT).as_posix())
        except ValueError:
            label = str(lp)
        n = 0
        for row in load_admitted(lp, admit=admit):
            rid, key = row["id"], loc_key(row)
            prev = bare_ids.get(rid)
            if prev is not None and prev != key:
                collisions.append(f"{rid} ({label})")
            bare_ids.setdefault(rid, key)
            if key in seen_loc:
                overlaps.append(f"{seen_loc[key][0]} vs {label}/{rid}")
                continue
            out = dict(row)
            out["id"] = namespaced_id(ns, rid)
            if keep_row_id:
                out["_ledger_row_id"] = rid
            out["_source_ledger"] = label
            out["_ledger_ns"] = ns
            seen_loc[key] = (out["id"], label)
            rows.append(out)
            n += 1
        per_ledger[label] = n
    return rows, {
        "n_union": len(rows), "per_ledger": per_ledger,
        "n_location_overlaps": len(overlaps), "overlap_sample": overlaps[:5],
        "n_id_collisions": len(collisions), "collision_sample": collisions[:5],
    }


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
    """`{location_id: embedding}` from EITHER seed layout.

    Two producers write morph embeddings and they disagree on the container. The emission
    driver and `stage_first_release` write ONE npz (`_save_embs` format: `ids` + `emb`); the
    discovery-side seed (`library_seed_v2`, and `deficit_scheduler`'s campaign-1 loader)
    writes one `<location_id>.npy` PER LOOK under a bulk directory. That difference is the
    whole reason stage 1 and stage 2 could not read each other's seed — so the reader learns
    both layouts rather than a converted copy being minted, which would be a second artifact
    to keep in step with the first. A directory reads per-look; a file reads npz."""
    p = Path(path)
    if p.is_dir():
        return {f.stem: np.load(f).astype(np.float32).reshape(-1) for f in sorted(p.glob("*.npy"))}
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=True)
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
    """location_id → morph cluster tag `<partition>#<k>`. Ledger order is the stable
    incremental order.

    Clustering runs WITHIN the BASE partition (`cluster_group`) — the within-family dedup
    convention, unchanged, so a classic-phoenix row is still compared against the varied
    phoenix medoids and its clustering geometry is exactly what it was. The TAG it comes out
    with is keyed by the row's own partition (`cell_partition`), so `phoenix:classic` is
    addressable on the cell axis and the `phoenix` cell no longer absorbs it.

    `library` is the existing library's medoids grouped the same way, `{base_partition:
    [(k, emb), ...]}` as returned by `library_medoids`. When given, each group's clustering is
    SEEDED with those medoids, so a new row that near-duplicates a library look joins the
    library's cluster index instead of founding a parallel one across the intake seam. When
    None (the pre-fix behaviour) the batch is deduplicated only against itself.

    Nothing already in the library moves: seeded clusters keep their key, seeded medoids are
    frozen, and only ids in `rows` appear in the returned map. `verify_library_unmoved` is the
    mechanical re-check the callers run."""
    lib = library or {}
    by_group: dict = {}
    part_of: dict = {}
    for row in rows:
        by_group.setdefault(cluster_group(row), []).append(row["id"])
        part_of[row["id"]] = cell_partition(row)
    tags = {}
    for group, ids in by_group.items():
        items = [(i, embs[i]) for i in ids if i in embs]
        assign = cluster_incremental(items, threshold, seed_medoids=lib.get(group))
        for i, k in assign.items():
            tags[i] = f"{part_of[i]}#{k}"
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
class SeedRekeyError(RuntimeError):
    """A library snapshot's cluster tag could not be re-keyed to partition cell identity
    without leaving its own base partition — i.e. the re-key would be a RE-CLUSTER."""


def seed_cluster_tags(meta: dict) -> dict:
    """`{location_id: "<partition>#<k>"}` for a library snapshot, re-keyed AT THE READ.

    A frozen seed artifact is never rewritten. The relit seed (`library_seed_v2`) tags looks
    family-keyed — `phoenix#k` for a classic-phoenix look too — which is precisely the state
    that makes `phoenix:classic` unseedable, and `verify_library_unmoved` raising on a pass
    that re-keys it is CORRECT behaviour, not an obstacle to route around. So the snapshot's
    own per-look record (`entries[<id>]["render"]`, the render block the look was made from)
    resolves the partition here, at read time.

    The re-key is deterministic and round-trips: `base_partition(new) == base_partition(old)`
    is asserted for every look, so the stored tag is recoverable from the re-keyed one and the
    cluster INDEX space (which is per base partition, the clustering group) is untouched.
    Anything else would be a re-cluster of committed library state and raises.

    A snapshot with no `entries` (the emission driver's own `intake.json`) passes through: its
    tags are already partition-keyed by `assign_morph_clusters`."""
    tags = meta.get("cluster_tags") or {}
    entries = meta.get("entries") or {}
    out = {}
    for loc_id, tag in tags.items():
        stored, _, k = tag.rpartition("#")
        render = (entries.get(loc_id) or {}).get("render")
        part = stored
        if render is not None:
            resolved = partition_of_row(render)
            if resolved is not None:
                part = resolved
        if base_partition(part) != base_partition(stored):
            raise SeedRekeyError(
                f"library look {loc_id!r} tagged {tag!r} re-keys to partition {part!r}, whose "
                f"base partition {base_partition(part)!r} differs from the stored tag's "
                f"{base_partition(stored)!r}. That is a re-cluster of committed library state, "
                f"not a re-key — refusing.")
        out[loc_id] = f"{part}#{k}"
    return out


def library_medoids(intake_path, embs_path) -> dict:
    """`{base_partition: [(k, medoid_emb), ...]}`, one medoid per existing library cluster,
    ordered by cluster key. Returns {} if either artifact is absent — the caller decides
    whether that is fatal (it MUST be loud: a silently-empty seed is exactly the un-deduped
    seam this seeding exists to close).

    Grouped by BASE partition because that is the clustering group (`cluster_group`): the
    cluster index space is shared by a base partition and everything derived off it, so
    `phoenix#3` and `phoenix:classic#3` are ONE cluster with one founding medoid, not two."""
    ip, ep = Path(intake_path), Path(embs_path)
    if not ip.exists() or not ep.exists():
        return {}
    tags = seed_cluster_tags(json.loads(ip.read_text(encoding="utf-8")))
    embs = load_embs(ep)
    founder: dict = {}                      # (group, k) → founding location id (snapshot order)
    for loc_id, tag in tags.items():
        part, _, k = tag.rpartition("#")
        gk = (base_partition(part), int(k))
        if gk not in founder and loc_id in embs:
            founder[gk] = loc_id
    by_group: dict = {}
    for (group, k), loc_id in founder.items():
        e = np.asarray(embs[loc_id], np.float32).reshape(-1)
        by_group.setdefault(group, []).append((k, e / (np.linalg.norm(e) + 1e-9)))
    return {f: sorted(v, key=lambda kv: kv[0]) for f, v in by_group.items()}


def library_assignments(intake_path) -> dict:
    """`{location_id: "<partition>#<k>"}` — the library's own cluster assignment (re-keyed at
    the read, see `seed_cluster_tags`), for the never-moved guard. {} if the snapshot is
    absent."""
    ip = Path(intake_path)
    if not ip.exists():
        return {}
    return seed_cluster_tags(json.loads(ip.read_text(encoding="utf-8")))


class LibraryRowMoved(RuntimeError):
    """An intake pass re-assigned a location that the library had already clustered. That
    rewrites committed library state (reachability, per-cell deficits, the release record's
    morph_cluster column), so it is refused rather than reported."""


# The released library the forward fix seeds against. This used to be
# `scratch/first_release` — the union snapshot `stage_first_release.py` assembles from the
# two committed intake passes (1387 locations / 1268 clusters). That directory is GONE and
# cannot come back: `scratch/` is the one class whose contract guarantees deletion, and both
# intake passes' own snapshots went with it. Pointing the DEFAULT at a disposable path is the
# same mistake campaign1's embeddings died of, so the default is now the durable relit seed
# (`data/emission/library_seed_v2`, 168 human->=3 looks over 9 partitions), which is the same
# artifact the discovery side's `deficit_scheduler.SEED_SOURCES` resolves — one seed, both
# stages. Any driver `--out` dir is still a valid `--library` (it writes the same two names).
DEFAULT_LIBRARY_DIR = ROOT / "data" / "emission" / "library_seed_v2"
LIBRARY_INTAKE_NAME = "intake.json"
LIBRARY_EMBS_NAME = "morph_embs.npz"


class LibrarySeedUnavailable(RuntimeError):
    """The library seed could not be loaded, so this intake would deduplicate against ITSELF
    ONLY while every downstream reader (cell reachability, per-cell deficits, the release
    record's `morph_cluster` column) assumes library-wide.

    This RAISES rather than warning. It used to print a note and continue, which is the exact
    shape that let campaign-1's seeding degrade silently: a printed line in a backgrounded
    run's log is not a decision anybody makes, and the run's numbers are on record as
    library-wide by the time anyone reads it."""


def _seed_registry():
    """`deficit_scheduler`, imported lazily and BARE (the spelling every other tools/ module
    uses), so stage 2 resolves seed paths through the same registry and the same
    scratch-class refusal the discovery side already owns — not a second copy of either."""
    for p in (ROOT / "tools" / "atlas",):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import deficit_scheduler                      # noqa: PLC0415
    return deficit_scheduler


def library_emb_source(library_dir) -> Path:
    """Where a library dir's morph embeddings actually live.

    Sibling `morph_embs.npz` when the snapshot is a driver/`stage_first_release` one; else
    the per-look bulk dir the SEED REGISTRY declares for that snapshot (the relit seed's 168
    `<loc_id>.npy` resolve out of tree through `paths.bulk`); else `<dir>/embs` for a
    hand-assembled pair. `load_embs` reads either layout, so no converted copy is minted."""
    d = Path(library_dir)
    npz = d / LIBRARY_EMBS_NAME
    if npz.exists():
        return npz
    ip = (d / LIBRARY_INTAKE_NAME).resolve()
    for _name, sip, sed in _seed_registry().SEED_SOURCES:
        if Path(sip).resolve() == ip:
            return Path(sed)
    return d / "embs"


def load_library_seed(library_dir=None) -> tuple:
    """`(medoids, prior_assignments, note)` for a library snapshot dir. FAIL-CLOSED.

    Raises `LibrarySeedUnavailable` when the snapshot is absent, unreadable, or holds no
    usable medoid — there is no "seed if you can" mode on the intake path. Both halves of the
    resolved pair are class-checked through `deficit_scheduler._refuse_scratch_class`, so a
    seed that a `rm -r scratch/*` could empty is refused while the mistake is still cheap."""
    d = Path(library_dir) if library_dir else DEFAULT_LIBRARY_DIR
    dsched = _seed_registry()
    ip = dsched._refuse_scratch_class("intake (library)", Path(d) / LIBRARY_INTAKE_NAME)
    ep = dsched._refuse_scratch_class("embeddings (library)", library_emb_source(d))
    med = library_medoids(ip, ep)
    if not med:
        missing = [str(p) for p in (ip, ep) if not p.exists()]
        why = ("missing " + ", ".join(missing) if missing else
               "the snapshot exists but yielded no usable medoid embedding "
               "(embedding dir empty, or no cluster_tags id has a vector)")
        raise LibrarySeedUnavailable(
            f"library seed UNAVAILABLE at {d}: {why}.\n"
            f"    intake     : {ip}\n"
            f"    embeddings : {ep}\n"
            f"An unseeded intake deduplicates against ITSELF ONLY and adds an un-deduped "
            f"seam that every per-cell deficit downstream is then denominated in. Rebuild "
            f"the seed (`uv run python tools/emission/library_seed_v2.py build` then "
            f"`embed`) or pass a --library whose snapshot exists.")
    prior = library_assignments(ip)
    n = sum(len(v) for v in med.values())
    return med, prior, (f"library seed: {n} medoids over {len(med)} types from {d} "
                        f"(embeddings: {ep})")


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
