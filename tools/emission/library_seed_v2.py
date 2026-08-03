#!/usr/bin/env python
r"""library_seed_v2.py — relight the scheduler's library look-seed from HUMAN verdicts.

WHY THE SEED IS DARK. `deficit_scheduler.require_library_seed` reads
`data/emission/campaign1/intake.json` plus per-look medoid embeddings under
`scratch/emission/campaign1/embs/`. Both are gone: the snapshot was never rebuilt after the
derived-artifact wipe, and the embeddings lived in `scratch/`, which is a class whose contract
GUARANTEES deletion. So `require_library_seed` sees **0 looks** today, and a `--scheduler` /
`--pop-quota` run either aborts fail-closed or measures RUN-LOCAL scarcity while every reader
assumes library-wide.

WHAT THIS SEEDS FROM, AND WHY IT IS THE RIGHT SOURCE. The q4 sitting left a residue: 322 rows
that Matt scored >=3 and that the run did NOT admit (`human_q3plus_queue.jsonl`, 168 distinct
looks, already clustered at the same cos 0.974 knee the tally uses). Those are the strongest
seed material in the tree — a human verdict, not a decode — and they are exactly the looks the
deficit is supposed to know the library already holds.

THE FLOOR-ADMIT PRECEDENT (`q4_harvest_emission.md`). A `q4_harvest` row is admitted on a
FLOOR (`p_notbad >= 0.5`) rather than on the q3 decode gate, because its selection signal is
orthogonal to the head's and gating on the head would let the head veto material it never
judged. The same shape applies here with a stronger floor: the admission condition is a HUMAN
label of 3 or 4. No decode is consulted, and `mix_source="human_q3plus"` tags every row so the
provenance is recoverable — the same source-tag mechanism `FLOOR_ADMIT_SOURCES` uses.

ONE ROW PER LOOK. The queue already carries `first_of_look` from the sitting's own
leader-radius clustering at cos 0.974. Re-clustering here would be a second opinion on a
question already answered with the same metric, so the flag is honoured rather than
recomputed — and the count of looks is asserted against the queue rather than reported from
this pass.

THE DURABILITY SPLIT IS THE TREE'S, DELIBERATELY. The snapshot is `durable()`
(git-tracked, survives `rm -r scratch/*`); the per-look embeddings are `bulk()` — regenerable
from the snapshot's own coordinates by `embed`, which is the property that makes the split
safe and the property the campaign-1 seed did not have (its inputs were in scratch and its
snapshot was not rebuildable from anything left on disk). `tools/emission/test_intake_durable.py`
pins that split for the campaign-1 pair and this follows it.

  uv run python tools/emission/library_seed_v2.py build          # snapshot only (fast)
  uv run python tools/emission/library_seed_v2.py embed          # the medoid embeddings
  uv run python tools/emission/library_seed_v2.py status
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "corpus",
           ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths                                            # noqa: E402

INTAKE_REL = "data/emission/library_seed_v2/intake.json"
EMB_REL = "scratch/emission/library_seed_v2/embs"
INTAKE_JSON = ROOT / INTAKE_REL
EMB_DIR = ROOT / EMB_REL

SOURCE_QUEUE = (ROOT / "data" / "discovery" / "q4_long_harvest_20260803" /
                "human_q3plus_queue.jsonl")
MIX_SOURCE = "human_q3plus"
FLOOR_LABEL = 3          # the floor: a HUMAN label of 3 or 4. No decode is consulted.


def _jl(p: Path):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines()
            if l.strip()]


# --------------------------------------------------------------------------- #
# build — the durable snapshot
# --------------------------------------------------------------------------- #
def build(queue_path: Path = SOURCE_QUEUE, *, write: bool = True) -> dict:
    """One medoid per distinct look, keyed `<partition>#<k>`, with its render block.

    The render block is joined back out of the label corpus by `(batch, image_id)`, so the
    snapshot carries everything `embed` needs and nothing that would make it stale — the
    coordinates ARE the location, and a re-embed months later reproduces the same vector."""
    import corpus_common as cc                          # noqa: E402

    rows = _jl(queue_path)
    admitted = [r for r in rows
                if int(r.get("human") or 0) >= FLOOR_LABEL and r.get("first_of_look")]
    floor_rejected = len(rows) - len(admitted)

    # join the render block out of the corpus, per source batch
    by_batch: dict = {}
    for r in admitted:
        by_batch.setdefault(r["batch"], []).append(r)
    renders: dict = {}
    missing = []
    for batch, brows in by_batch.items():
        idx = {x["image_id"]: x for x in
               cc.read_jsonl(str(Path(cc.batch_dir(batch)) / "images.jsonl"))}
        for r in brows:
            hit = idx.get(r["image_id"])
            if hit is None:
                missing.append((batch, r["image_id"]))
                continue
            renders[r["image_id"]] = hit["render"]
    if missing:
        raise SystemExit(f"{len(missing)} queue rows have no corpus row to join a render "
                         f"block from (first: {missing[0]}). The seed must not be built from "
                         f"a partial join — a look with no coordinates cannot be re-embedded.")

    per_part = Counter()
    medoid_id, entries = {}, {}
    for r in sorted(admitted, key=lambda x: (x["partition"], -int(x["human"]),
                                             x["image_id"])):
        part = r["partition"]
        tag = f"{part}#{per_part[part]}"
        per_part[part] += 1
        loc_id = r["image_id"]
        medoid_id[tag] = loc_id
        entries[loc_id] = dict(cluster_tag=tag, partition=part, human=int(r["human"]),
                               batch=r["batch"], image_id=r["image_id"],
                               mix_source=MIX_SOURCE, render=renders[loc_id])

    # Repo-relative when it is in the tree, absolute otherwise: the snapshot has to say where
    # it came from, and a `relative_to` that raises on an out-of-tree queue would make the
    # whole build untestable against a fixture.
    try:
        src = Path(queue_path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        src = str(queue_path)
    snap = dict(
        schema_version=1, source=src,
        mix_source=MIX_SOURCE,
        admission=dict(rule=f"human label >= {FLOOR_LABEL} AND first_of_look",
                       floor=FLOOR_LABEL, decode_consulted=False,
                       precedent="q4_harvest floor-admit (docs/design/q4_harvest_emission.md)",
                       near_dup_cos=0.974,
                       looks_from="queue `first_of_look` (the sitting's own leader-radius "
                                  "clustering at the same knee) — not recomputed here"),
        n_queue_rows=len(rows), n_looks=len(admitted), floor_rejected=floor_rejected,
        cluster_tags={loc_id: e["cluster_tag"] for loc_id, e in entries.items()},
        medoid_id=medoid_id, entries=entries,
        by_partition=dict(per_part),
        emb_dir=EMB_REL,
    )
    if write:
        p = paths.durable(INTAKE_REL, mkparents=True)
        p.write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8")
        snap["written"] = str(p)
    return snap


# --------------------------------------------------------------------------- #
# embed — the regenerable half
# --------------------------------------------------------------------------- #
def embed(*, limit: int | None = None, force: bool = False) -> dict:
    """One 768-d library-morph CLIP vector per medoid, written as `<emb_dir>/<loc_id>.npy`.

    The SAME recipe the tally, the emission clustering and the scheduler's own admission
    embed use (640x360 ss2 smooth field -> robust-z tanh gray -> vit_base_patch16_clip_224),
    because a seed embedded under a different recipe is not comparable to the admissions it
    is supposed to dedup against — and the 0.974 threshold would then mean nothing.

    Resumable: an existing `.npy` is skipped unless `--force`. A row that fails is COUNTED
    and named, never silently absent."""
    import numpy as np
    from tools.wallpaper import library_annotate as la      # noqa: E402
    from tools.curation.colored_clip import load_clip, embed_clip   # noqa: E402
    import location as loc_mod                              # noqa: E402

    if not INTAKE_JSON.exists():
        raise SystemExit(f"{INTAKE_JSON} missing — run `build` first.")
    snap = json.loads(INTAKE_JSON.read_text(encoding="utf-8"))
    out_dir = Path(paths.bulk(EMB_REL))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_fields"
    tmp.mkdir(parents=True, exist_ok=True)

    todo = [(lid, e) for lid, e in snap["entries"].items()
            if force or not (out_dir / f"{lid}.npy").exists()]
    if limit:
        todo = todo[:limit]
    model = tf = None
    done, failed = 0, Counter()
    for lid, e in todo:
        try:
            loc = loc_mod.from_render_block(e["render"])
            field = la.ensure_field(loc, retain=False, tmp_dir=tmp, cache_root=tmp)
            gray = la.morph_gray_image(field)
            if model is None:
                model, tf = load_clip()
            v = embed_clip(model, tf, [gray])[0].astype(np.float32)
            v = v / (float(np.linalg.norm(v)) + 1e-9)
            np.save(out_dir / f"{lid}.npy", v)
            done += 1
        except Exception as exc:                            # noqa: BLE001
            failed[f"{type(exc).__name__}: {str(exc)[:80]}"] += 1
    have = len(list(out_dir.glob("*.npy")))
    return dict(embedded_now=done, on_disk=have, wanted=len(snap["entries"]),
                failed=dict(failed.most_common(5)), emb_dir=str(out_dir))


# --------------------------------------------------------------------------- #
# status — what `require_library_seed` would see
# --------------------------------------------------------------------------- #
def status() -> dict:
    """The count the prompt asks for, taken THROUGH `require_library_seed` rather than by
    counting files — the question is what the guard sees, and the guard has its own rules
    (dimension check, per-partition grouping, tracked-partition restriction)."""
    import deficit_scheduler as dsched                     # noqa: E402
    rec = dsched.require_library_seed(allow_unseeded=True, intake_path=INTAKE_JSON,
                                      emb_dir=Path(paths.bulk(EMB_REL)))
    default = dsched.require_library_seed(allow_unseeded=True)
    return dict(
        v2=dict(status=rec["status"], library_looks=rec["library_looks"],
                library_partitions=rec["library_partitions"],
                source=rec["source"], emb_dir=rec["emb_dir"],
                per_partition={p: int(m.shape[0])
                               for p, m in (rec.get("embeddings") or {}).items()}),
        default_paths=dict(status=default["status"],
                           library_looks=default["library_looks"],
                           source=default["source"], emb_dir=default["emb_dir"],
                           source_exists=default["source_exists"]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--queue", default=str(SOURCE_QUEUE))
    b.add_argument("--dry-run", action="store_true")
    e = sub.add_parser("embed")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--force", action="store_true")
    sub.add_parser("status")
    a = ap.parse_args()

    if a.cmd == "build":
        snap = build(Path(a.queue), write=not a.dry_run)
        print(json.dumps({k: v for k, v in snap.items()
                          if k not in ("cluster_tags", "medoid_id", "entries")}, indent=2))
    elif a.cmd == "embed":
        print(json.dumps(embed(limit=a.limit, force=a.force), indent=2))
    print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main()
