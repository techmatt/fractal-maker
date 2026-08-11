r"""build_baserate_sheet.py — SHEET F: the BASE-RATE AUDIT correction sheet.

STATE §OPEN-1, RESHAPED BY MATT. The four-sheet table differs by **11x at >=3** between
sheet C (anchored, 46.4%) and sheet E (blind, 4.0%) drawn by the SAME rule, and `t_good`, the
mining gate and every downstream keeper rate were calibrated on anchored labels. The obvious
answer — buy a second, larger blind instrument — is not the one taken: blind labels are the
most expensive kind and sheet E already spent 150 of them on this population. What this sheet
buys instead is a STANDARD CORRECTION SHEET over a draw with no mining score in it, and it
answers two questions at once:

  (a) THE BASE RATE, on the population the mining gate actually sees. Nothing about the
      selection favours a mode, a score band or a look, so the realized tier mix is the
      roster's. The number is AGREEMENT-INFLATED — the page is anchored, see below — and that
      inflation is the one thing sheet E measures and this sheet does not; the two read
      together bound it on the same draw rule.
  (b) THE CUT. The page is ordered good->bad by mining v3's continuous readout and every row
      carries it, so Matt's labels arrive as a function of the score and the LABEL/SCORE
      CROSSOVER is a position on the page rather than a threshold sweep. Setting the mining
      cut at it is the follow-up prompt's job; this one only has to make it visible.

THE DRAW IS SHEET E'S, IMPORTED — NOT RESTATED, AND NOT PARAPHRASED.
`build_blind_mining_sheet` owns the population, the location draw, the palette draw, the
candidate deal, the screen and the near-dup/smooth-equivalence selection, and this module
calls those functions rather than carrying a second copy of them. That is not tidiness: "the
population is sheet E's" is the whole claim on which the C-vs-E-vs-F comparison rests, and a
paraphrase is a claim nothing checks. Everything sheet F declares of its own is a SPEC FIELD
(`SheetSpec` below) or lives in the write path.

  * `contested_modes=()` — the ONE draw-shaped difference, and it is a spec field on sheet E's
    own spec rather than a fork. Sheet E over-draws the four cells the staged arms contest; a
    base-rate read wants the opposite, so the whole page is one flat
    `apportion.deal_round_robin` over the active roster.
  * `exp_smoothing` stays excluded, inherited: measured ~100% smooth-equivalent on both
    batches with a table, and a label spent on a smooth twin buys nothing HERE either. It is
    a declared narrowing of "the gate's population" and it is what keeps F comparable to E.

WHERE THE MINING HEAD IS, AND WHERE IT IS NOT. Exactly one function in this module reaches a
checkpoint — `score_for_prefill`, called from `run_write` and from nowhere else, AFTER
`select` has already returned the drawn set. The draw cannot see a score because the score
does not exist until the rows are chosen and rendered. `test_baserate_sheet.py` enforces that
structurally with an AST walk (every mining-head token confined to that one function and the
writer, and no selection-path function calling it) rather than trusting this paragraph.

TRAIN-SIDE, 100%, and stamped per row (`classifier_retrain_protocol.md` §2b). A slice served
with the incumbent's suggestions measures agreement with the incumbent; it may never referee
one head against another and it never joins sheets D/E. That is a fact about how the LABELS
were elicited and it is independent of how good the draw is.

    uv run python -u tools/mining/build_baserate_sheet.py pool
    uv run python -u tools/mining/build_baserate_sheet.py estimate      # + the render bill
    uv run python -u tools/mining/build_baserate_sheet.py screen --limit 8    # smoke
    uv run python -u tools/mining/build_baserate_sheet.py screen > scratch/baserate_sheet/screen.log 2>&1
    uv run python -u tools/mining/build_baserate_sheet.py select
    uv run python -u tools/mining/build_baserate_sheet.py render --limit 6    # bounded E2E
    uv run python -u tools/mining/build_baserate_sheet.py render > scratch/baserate_sheet/render.log 2>&1
    uv run python -u tools/mining/build_baserate_sheet.py write
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "queries",
           ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import corpus_common as cc                                  # noqa: E402  engine launch defaults
import location as loc_mod                                  # noqa: E402
import partitions as PART                                   # noqa: E402  THE partition resolver
# SHEET E OWNS THE DRAW. Every population, location, palette, candidate, screen and selection
# function below is called out of this module, never copied into it — see the docstring.
from tools.mining import build_blind_mining_sheet as BE     # noqa: E402  THE draw
from tools.mining import build_mining_sheet as BMS          # noqa: E402  THE render paths
from tools.mining import build_rare_palette_sheet as RPS    # noqa: E402  embeddings store
from tools.mining import mining_pins as MP                  # noqa: E402  the pin (write only)
from tools.mining import mining_roster as MR                # noqa: E402  THE class vocabulary
from tools.mining import suggest_tier_mining as ST          # noqa: E402  the suggestion rule
from tools.palettes import hue_families as HF               # noqa: E402
from tools.scoring import batch_registry as BR              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                            # noqa: BLE001
    pass

CORPUS = ROOT / "data" / "render_mode_corpus"
SCREEN_GEOM = BE.SCREEN_GEOM
WORKERS = BE.WORKERS
ENGINE_THREADS = BE.ENGINE_THREADS

# THE ROW SHAPE, declared and asserted at write time — sheet E's mechanism, inverted. Sheet E
# asserts the ABSENCE of `suggested_tier` because that field is what puts the rig into
# correction mode; this sheet asserts its PRESENCE for the same reason, and asserts the row
# carries the complete join beside it. A correction row that lost `suggested_tier` would serve
# silently as a blind one and the sitting would cost several times what it was budgeted.
ROW_KEYS = ("image_id", "sheet_order", "render", "provenance", "label",
            "head_mining_v1", "pred", "p_ge3", "suggested_tier")


# =========================================================================== #
# The sheet spec — a frozen dataclass from the start (CLAUDE.md, "Writing a builder for one
# instance"). It is a SUPERSET of sheet E's field set, which is what lets `BE.universe`,
# `BE.run_screen`, `BE.select` and `BE.run_render` take it unchanged.
# =========================================================================== #
@dataclass(frozen=True)
class SheetSpec:
    key: str
    batch_id: str
    generator_version: str
    img_prefix: str
    id_salt: str
    target_rows: int
    n_locations: int
    max_rows_per_location: int
    oversample: float               # screen candidates per served row
    draw_seed: int
    location_caps: dict = field(default_factory=dict)
    classic_partition: str = PART.CLASSIC_PHOENIX
    # NO CELL IS FAVOURED. The empty tuple is the base-rate axis and `contested_per_mode` is
    # dead alongside it; both are kept so the spec stays substitutable for sheet E's.
    contested_modes: tuple = ()
    contested_per_mode: int = 0

    @property
    def batch_dir(self) -> Path:
        return CORPUS / "batches" / self.batch_id

    @property
    def work(self) -> Path:
        return ROOT / "scratch" / "baserate_sheet" / self.key

    @property
    def screen_log(self) -> Path:
        return self.work / "screen.jsonl"

    @property
    def embed_store(self) -> Path:
        return self.work / "screen_embeddings.npz"

    @property
    def labels_export(self) -> str:
        """What the page downloads and `merge_sitting --scores` READS — beside the sidecar,
        never under `scratch/`: a label export is the one artifact in the pipeline with no
        rebuild path."""
        return f"labels/scores_{self.batch_id}.json"

    @property
    def labels_sidecar(self) -> str:
        """What the MERGE writes. A different file from the export above; sheet D pointed
        both at the sidecar, which would have merged the destination into itself."""
        return f"labels/{self.generator_version}.json"

    @property
    def ui_url(self) -> str:
        # `tiers=3` is the render-mode scale and `order=file` honours the builder's stamped
        # good->bad sort. There is no `&correction` knob: the rig enters correction mode
        # because the ROWS carry `suggested_tier`, which is why its presence is asserted.
        return (f"tools/viz/wallpaper_label.html?corpus=render_mode_corpus&tiers=3"
                f"&order=file&batch={self.batch_id}")


SHEETS = {
    "f": SheetSpec(
        key="f",
        batch_id="2026-08-11_render_mode_baserate_audit_v1",
        generator_version="render_mode_baserate_audit_v1",
        img_prefix="mba",
        id_salt="render_mode_baserate_audit_v1/2026-08-11",
        target_rows=200,
        n_locations=200,
        max_rows_per_location=2,
        oversample=1.6,
        # A FRESH SEED, deliberately. Sheet E's 29 walks the same per-mode permutations of an
        # overlapping location pool; a shared seed would correlate which locations the two
        # sheets reach even though the freshness filter already makes them disjoint.
        draw_seed=41,
        # Sheet E's caps scaled to 200 rows. `phoenix:classic` costs ~54 s per keeper render
        # against ~8 s for mandelbrot and would otherwise own the bill; the cap is an upper
        # bound and its supply may already be spent, in which case the cell is simply absent.
        location_caps={"phoenix": 21, PART.CLASSIC_PHOENIX: 8},
    ),
}


def log(msg):
    print(msg, flush=True)


# =========================================================================== #
# 1. The render bill — sheet E's measured per-partition costs, projected onto THIS draw.
# =========================================================================== #
# WHERE THE BASIS COMES FROM, and why it is not sheet E's own constant. `BE.BILL_PROBE` points
# at `scratch/sittings27c/bill_probe.json`, which is gone — `scratch/` is disposable and was
# wiped. The same measurements survive INSIDE sheet E's committed `batch.json`, which is a
# tracked record carrying its own date and derivation, so that is the basis read here. This is
# the "a measurement that survives carries its date and the command that produced it" rule
# applied in the only direction left: the record outlived the scratch file it was written from.
BILL_BASIS_BATCH = "2026-08-11_render_mode_blind_v1"
BILL_SPEEDUP = BE.BILL_SPEEDUP


def _bill_means() -> tuple[dict, str]:
    """`({stage: {partition: mean_s}}, basis)` off the committed sheet-E batch record."""
    p = CORPUS / "batches" / BILL_BASIS_BATCH / "batch.json"
    if not p.exists():
        return {}, f"UNKNOWN — {p.relative_to(ROOT).as_posix()} is not on disk"
    doc = json.loads(p.read_text(encoding="utf-8")).get("render_bill", {})
    stages = doc.get("stages") or {}
    means = {name: {part: v.get("mean_s") for part, v in (s.get("per_partition") or {}).items()}
             for name, s in stages.items()}
    if not means:
        return {}, f"UNKNOWN — {BILL_BASIS_BATCH}'s batch.json carries no render_bill stages"
    return means, (f"data/render_mode_corpus/batches/{BILL_BASIS_BATCH}/batch.json "
                   f"(render_bill, measured 2026-08-10; its own scratch probe is wiped)")


def render_bill(spec: SheetSpec, candidates, twins) -> dict:
    """The bill, stated UP FRONT and per partition (CLAUDE.md, runtime discipline)."""
    means, basis = _bill_means()
    out = {"basis": basis,
           "speedup_assumed": BILL_SPEEDUP,
           "speedup_basis": "4 workers x 3 rayon threads at BELOW_NORMAL — sheet C's measured "
                            "screen speedup, not a core count",
           "n_screen_units": len(candidates) + len(twins),
           "n_keeper_units": spec.target_rows}
    if not means:
        out["status"] = basis
        return out
    screen_n = Counter(e["partition"] for e in candidates + twins)
    # The keeper draw is not known until `select`, so the bill projects the CANDIDATE mix
    # scaled to the served row count — stated, because a projection is not a measurement.
    scale = spec.target_rows / max(1, len(candidates))
    keep_n = {p: n * scale for p, n in Counter(e["partition"] for e in candidates).items()}
    stages = {}
    for name, counts in (("screen", screen_n), ("keeper", keep_n)):
        per, unpriced = {}, []
        for p, n in sorted(counts.items()):
            mean = means.get(name, {}).get(p)
            if mean is None:
                unpriced.append(p)
            per[p] = {"units": round(float(n), 1), "mean_s": mean,
                      "single_process_min": round(n * mean / 60.0, 2) if mean else None}
        total = sum(v["single_process_min"] or 0.0 for v in per.values())
        stages[name] = {"per_partition": per,
                        "single_process_min": round(total, 1),
                        "wall_min_at_4x": round(total / BILL_SPEEDUP, 1),
                        # NOT silently dropped: a partition sheet E never rendered has no
                        # measured cost here, and a bill that omits it reads as a free slice
                        # (CLAUDE.md, "no silent caps").
                        "unpriced_partitions": unpriced}
    out["stages"] = stages
    out["total_wall_min_at_4x"] = round(sum(s["wall_min_at_4x"] for s in stages.values()), 1)
    out["excluded_from_the_bill"] = (
        "the colored-CLIP embed pass over every screen unit and the ONE mining-v3 scoring "
        "pass over the keeper crops at write time — both GPU, both minutes, neither in the "
        "per-partition engine means above")
    out["projection_caveat"] = (
        "the keeper stage projects the CANDIDATE partition mix scaled to the served row "
        "count; the served mix is not known until `select` runs its filters. Reproject from "
        "the run's own observed rate rather than restating this "
        "(CLAUDE.md, 'projecting a long run's wall clock').")
    return out


def _selected(spec: SheetSpec, args):
    """Sheet E's universe + screen + selection, with THIS sheet's bill.

    Deliberately thin, and deliberately not `BE._selected`: the only difference is which
    render bill is attached, and everything that decides a ROW is sheet E's."""
    candidates, twins, loc_meta, uni = BE.universe(spec)
    screen = BE.load_screen(spec)
    if not screen:
        raise SystemExit("[select] no screen records — run `screen` first")
    emb = RPS.load_embeddings(spec)
    targets = dict(uni["mode_targets"]["rows_by_mode"])
    sel, rep = BE.select(spec, candidates, twins, emb, targets)
    by_key = {e["unit_key"]: e for e in candidates}
    return by_key, loc_meta, uni, sel, rep, render_bill(spec, candidates, twins)


# =========================================================================== #
# 2. THE ONLY FUNCTION IN THIS MODULE THAT REACHES A MINING CHECKPOINT.
# =========================================================================== #
def score_for_prefill(crop_paths: list[Path]) -> dict:
    """`{pred, p_ge2, p_ge3, score, passed, ckpt, head_version, threshold, k}` per crop.

    CALLED FROM `run_write` AND NOWHERE ELSE, on crops that are already chosen and already
    rendered. Confining every head symbol to one function is what makes "the draw is
    score-unconditioned" a checkable property of this file instead of a claim in its
    docstring: `test_baserate_sheet.py` walks the AST and fails if a mining token appears in
    any other function, or if any selection-path function calls this one.

    Refuses a head whose K disagrees with the suggestion rule rather than coercing — a cut on
    a CORN marginal sum is a point on one head's readout scale, and a K mismatch means the
    scale is not the one `ST.CUTS` was fitted on."""
    from tools.mining.mining_gate import MiningScorer         # noqa: PLC0415  (torch)

    scorer = MiningScorer(model_path=MP.ACTIVE_MINING_CKPT)
    if scorer.k != ST.K_TIERS:
        raise SystemExit(f"[write] head K={scorer.k} but the suggestion rule is written for "
                         f"K={ST.K_TIERS} — fix the rule, do not coerce.")
    log(f"[write] mining head {MP.HEAD_VERSION} (K={scorer.k}) on {scorer.device} · "
        f"scoring {len(crop_paths)} crops")
    scores = scorer.score_paths(crop_paths)
    pred = [ST.expected_tier([s.p_ge2, s.p_ge3]) for s in scores]
    return {
        "pred": pred,
        "tiers": [int(t) for t in ST.suggest_all(pred, ST.CUTS)],
        "rows": [{"p_ge2": s.p_ge2, "p_ge3": s.p_ge3, "score": s.score,
                  "passed": bool(s.passed)} for s in scores],
        "ckpt": MP.ACTIVE_MINING_CKPT, "head_version": MP.HEAD_VERSION,
        "gate_version": MP.MINING_GATE_VERSION, "threshold": scorer.threshold,
        "k": scorer.k,
    }


# =========================================================================== #
# 3. Write — v3 prefill, sorted good->bad, 100% train-side.
# =========================================================================== #
def provenance_block(spec, entry, rec, loc_meta, transfer_dropped) -> dict:
    p = entry["color_params"]
    return {
        "generator_version": spec.generator_version,
        "batch_id": spec.batch_id,
        "lineage": "mining_baserate_audit",
        "family": entry["family"],
        "partition": entry["partition"],
        "location_key": entry["location_key"],
        "render_mode": entry["mode"],
        "mode_kind": entry["kind"],
        # THE COMPLETE JOIN, and it is LAW for this sitting: `render` + `mode_params` +
        # `color_params` is exactly the tuple `render-one` needs, so every tracked row is
        # rebuildable without the crop and a re-render under a fresh id is still this row.
        "mode_params": dict(entry.get("mode_params", {})),
        "rolloff": MR.rolloff_token(entry["mode"]),
        "color_params": {
            "palette": entry["palette"],
            "palette_type": p.get("palette_type"), "palette_source": p.get("palette_source"),
            "reverse": p["reverse"], "log_premap": p["log_premap"], "gamma": p["gamma"],
            "phase": p["phase"], "n_cycles": p["n_cycles"],
            "transfer": p["transfer"], "transfer_gamma": p["transfer_gamma"],
            "interior_color": list(p.get("interior_color", [0.0, 0.0, 0.0])),
        },
        "transfer_dropped": transfer_dropped,
        "hue_family": entry["hue_family"],
        "palette_source_rule": "screened POOL draw — rare_palette_draw.PaletteDrawer against "
                               "the declared rare hue-family target, then palette_deficit.pick "
                               "within the family. No head proposes it and no head ranks it.",
        "bucket": rec["bucket"],
        "draw_order": entry["draw_order"],
        "cos_smooth": rec.get("cos_smooth"),
        "smooth_band": rec.get("band"),
        "screen_path": f"the keeper render paths at a SCORING-ONLY geometry "
                       f"{SCREEN_GEOM[0]}x{SCREEN_GEOM[1]}ss{SCREEN_GEOM[2]} "
                       f"(build_mining_sheet.render_one with `geom`) — embedded for the "
                       f"near-dup and smooth-equivalence filters, NEVER scored",
        # 100% TRAIN. Not a stratified assignment and nothing to re-derive: protocol 2b
        # disqualifies an anchored sheet from the eval side outright, so there is no split
        # decision to make and a seeded one would only look like there had been.
        "split_side": "train",
        "split_origin": "anchored_correction_2b",
        "source": {
            "corpus": "data/label_corpus",
            "human_score": entry["human_score"],
            "label_batches": loc_meta.get("label_batches"),
            "rule": "location label = MAX over its crops, resolved through "
                    "label_store.resolve_score with amendments applied. THE ONLY quality "
                    "condition on this row, and it is a HUMAN one — no mining score "
                    "conditioned the draw.",
        },
    }


def run_write(spec: SheetSpec, args):
    by_key, loc_meta, uni, selected, sel_rep, bill = _selected(spec, args)
    done = BE.load_ledger(spec)
    if not done:
        raise SystemExit("[write] no rendered units — run `render` first")
    live = [r for r in selected if r["unit_key"] in done]
    # FAIL BEFORE TRUNCATING (sheet E's rule, and its reason is unchanged): `images.jsonl` is
    # opened "w" below, so a run reaching that line with nothing to write REPLACES a good
    # sheet with an empty one — and an empty images.jsonl then feeds back into the next run's
    # own prior-batch freshness scan.
    if not live:
        raise SystemExit(
            f"[write] {len(selected)} selected, {len(done)} in the render ledger, 0 in both — "
            f"refusing to overwrite {spec.batch_dir / 'images.jsonl'} with an empty sheet. "
            f"The draw and the ledger disagree; re-run `select` and compare.")

    live.sort(key=lambda r: r["unit_key"])
    crops = spec.batch_dir / "crops"
    head = score_for_prefill([crops / f"{by_key[r['unit_key']]['image_id']}.jpg"
                              for r in live])

    rows = []
    for j, rec in enumerate(live):
        e = by_key[rec["unit_key"]]
        loc = loc_mod.from_render_block(e["render"])
        h = head["rows"][j]
        rows.append({
            "_unit_key": rec["unit_key"], "_crop_stem": e["image_id"],
            "render": BMS.render_block(e, loc),
            "provenance": provenance_block(
                spec, e, rec, loc_meta[e["location_key"]],
                bool(done[rec["unit_key"]]["transfer_dropped"])),
            # THE HUMAN SLOT stays null on a served row. A SUGGESTION IS NOT A LABEL: the
            # merge refuses to read `suggested_tier` as one, and an unreviewed suggestion
            # never leaves the page.
            "label": {"score": None, "labeler": None, "labeled_at": None},
            # The field NAME is the corpus schema's, not a version claim — `mining_corpus`,
            # `fresh_sheet_reads` and the v2/v3 reads all join on `head_mining_v1`, and the
            # block inside carries `head_version` for what actually scored it. Renaming it to
            # `head_mining_v3` would make this batch invisible to every reader in the tree.
            "head_mining_v1": {
                "pred": head["pred"][j], "p_ge2": h["p_ge2"], "p_ge3": h["p_ge3"],
                "score": h["score"],
                "ckpt": head["ckpt"], "head_version": head["head_version"],
                "gate_version": head["gate_version"],
                "would_pass_gate": h["passed"], "gate_threshold": head["threshold"],
            },
            "p_ge3": h["p_ge3"],
            "pred": head["pred"][j],
            "suggested_tier": head["tiers"][j],
        })

    # PRESENTATION — good -> bad by the CONTINUOUS score, descending; ties on the crop stem.
    # This is (b): the crossover between Matt's label and the head's rank is a POSITION on the
    # page, and the sheet is built so that reading it needs no threshold sweep.
    rows.sort(key=lambda r: (-r["pred"], r["_crop_stem"]))
    for i, r in enumerate(rows):
        r["sheet_order"] = i
        r["image_id"] = f"{spec.img_prefix}{i:04d}_{r['_crop_stem'][:8]}"
    assert len({r["image_id"] for r in rows}) == len(rows), "opaque ids collided"
    for r in rows:
        extra = set(r) - set(ROW_KEYS) - {"_unit_key", "_crop_stem"}
        assert not extra, f"{r['image_id']}: undeclared row field(s) {sorted(extra)}"
        missing = set(ROW_KEYS) - set(r)
        assert not missing, (f"{r['image_id']}: missing {sorted(missing)} — a correction row "
                             f"without `suggested_tier` serves BLIND and silently costs the "
                             f"sitting several times its budget (ROW_KEYS)")

    for r in rows:
        dst = crops / f"{r['image_id']}.jpg"
        if not dst.exists():
            shutil.copyfile(crops / f"{r['_crop_stem']}.jpg", dst)
    route = {r["image_id"]: {"unit_key": r.pop("_unit_key"), "crop_stem": r.pop("_crop_stem")}
             for r in rows}

    errors = []
    ep = spec.batch_dir / "_render_errors.json"
    if ep.exists():
        errors = json.loads(ep.read_text(encoding="utf-8"))
    # INCOMPLETE is DERIVED from the counts, never a flag: a bounded `--limit` run and a
    # killed run both produce a short batch and only one of them would have set one.
    incomplete = len(rows) < sel_rep["drawn_rows"]
    accounted = set(done) | {e["unit_key"] for e in errors}
    unaccounted = sorted(r["unit_key"] for r in selected if r["unit_key"] not in accounted)
    reg = BR.lookup(spec.batch_id, "mandelbrot")
    pred = [r["pred"] for r in rows]

    spec.batch_dir.mkdir(parents=True, exist_ok=True)
    with (spec.batch_dir / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (spec.batch_dir / "route.json").write_text(json.dumps(route, indent=1), encoding="utf-8")

    batch = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batch_id": spec.batch_id,
        "generator_version": spec.generator_version,
        "labeler": None,
        "n_rows": len(rows),
        "schema_note":
            "SHEET F — the BASE-RATE AUDIT correction sheet (state OPEN-1, reshaped by Matt). "
            "Sheet E's draw rule IMPORTED rather than restated — fresh (location, mode) pairs "
            "against every prior render-mode batch by key AND proximity, locations "
            "conditioned on HUMAN label quality alone, palettes a screened POOL draw, the "
            "canonical emission coloring at the frozen corpus pins — with the ONE difference "
            "that no mode cell is over-drawn: the whole page is a flat apportionment over the "
            "active roster, so the realized mix is the roster's. NO mining score is anywhere "
            "in the draw. The PAGE is anchored: mining v3's suggested tier is prefilled and "
            "the page is sorted good->bad by its continuous readout, so the label/score "
            "crossover is a position on the page. label.score is null on every row — a "
            "suggestion is not a label and the merge refuses to read one as one. TRAIN-SIDE, "
            "100%, per classifier_retrain_protocol.md 2b.",
        "sheet_incomplete": incomplete,
        "incomplete_note": (
            f"{len(rows)} of {sel_rep['drawn_rows']} drawn rows are present — this batch is a "
            f"BOUNDED or INTERRUPTED run and must not be treated as the full sheet. Re-run "
            f"`render` then `write`.") if incomplete else None,
        "registration": {
            "source": reg.source, "biased": reg.biased,
            "eval_eligible": reg.eval_eligible,
            "score_unconditioned": reg.score_unconditioned,
            "split": BR.split_of(reg),
            "registered_before_build": True,
            "owner": "tools/scoring/batch_registry.py",
            "why_the_flag_is_false":
                "`score_unconditioned` is the forced-eval-cascade exemption for eval "
                "INSTRUMENTS and the registry's own invariant forbids it on a train-side "
                "row; sheets C, D and E all carry False for the same reason. The fact it "
                "would have recorded — no mining head in the draw — is `draw_unconditioned` "
                "below, which is where a reader of a render-mode batch looks.",
        },
        # --- the property this sheet exists for -------------------------------------- #
        "draw_unconditioned": {
            "of": "the MINING head, at every stage of the selection",
            "held_by": "tools/mining/build_baserate_sheet.score_for_prefill is the only "
                       "function in the builder that reaches a checkpoint, it is called from "
                       "run_write alone, and it runs AFTER select has returned. Enforced by "
                       "an AST walk in test_baserate_sheet.py, not by this sentence.",
            "population": "imported from build_blind_mining_sheet (sheet E), so 'the same "
                          "draw rule' is a call graph rather than a paraphrase",
            "mode_axis": "flat apportion.deal_round_robin over the active roster — "
                         "contested_modes=(), so no cell is over-drawn",
            "palette": "screened POOL draw (rare_palette_draw against the declared rare "
                       "hue-family target, palette_deficit.pick within family)",
            "near_dup_tie_break": "DRAW ORDER — sheet C broke these ties by mining score; "
                                  "there is no score here to break them with",
            "not_unconditioned_on": "the HUMAN label corpus. Locations carry a human 4 "
                                    "(falling back to 3 where a partition is short), which is "
                                    "why the registration is biased -> train on the LOCATION "
                                    "axis and why no rate here is a LOCATION base rate. It is "
                                    "the rate over the strange-mode RENDERING of that gated "
                                    "population — which is what the mining gate sees.",
        },
        "anchored": {
            "prefill_head": head["head_version"],
            "prefill_ckpt": head["ckpt"],
            "prefill_ckpt_source": "tools/mining/mining_pins.ACTIVE_MINING_CKPT, read at "
                                   "write time — never a literal path",
            "presentation_sorted_on": "the continuous readout (pred), NOT the suggested tier",
            "cost": "labels returned on an anchored page are AGREEMENT-INFLATED. On the "
                    "mining sitting 892/960 = 0.929 came back equal to what was served, and "
                    "sheet A measured 815/960 = 0.849 on the wallpaper side. Every rate read "
                    "off this sheet is a CEILING; sheet E is the unanchored bound on the same "
                    "draw rule and the pair is what the audit reads.",
            "correction_rate_is_computable":
                "every row stamps the suggested_tier it was SERVED with, so agreement is "
                "labeled_score == suggested_tier over the merged sidecar — the first "
                "convergence datum on the flipped head (v3), and the reason the prefill "
                "source is frozen into this record rather than recomputed later.",
        },
        "suggested_tier_rule": ST.fit_derivation(ST.CUTS, pred, head["ckpt"],
                                                 head["head_version"]),
        "head": {
            "ckpt": head["ckpt"], "version": head["head_version"],
            "gate_version": head["gate_version"],
            "role": "PRE-LABEL ONLY — no gate, floor, threshold or pin is applied or moved "
                    "here, and would_pass_gate is stamped and gates nothing. Setting the "
                    "mining cut at the crossover is the FOLLOW-UP prompt's job.",
            "scorer": "tools/mining/mining_gate.MiningScorer (fp32, no autocast; marginal "
                      "p_ge = cumprod(sigmoid), NEVER the CORN conditional)",
            "deploy_transform": "classifier.data.Transform(train=False) — 384x224 bicubic "
                                "stretch + the checkpoint's own mean/std",
            "gate_threshold": head["threshold"],
        },
        "heads_read": {
            "mining": "AT WRITE TIME ONLY, on already-selected crops — the prefill and the "
                      "page order. Not in the draw, not in the screen, not in the selection.",
            "location": "NONE — quality is conditioned on the HUMAN label corpus.",
            "palette": "NONE — the palette is a screened POOL draw, not a head proposal.",
        },
        "universe": uni,
        "selection_report": sel_rep,
        "render_bill": bill,
        "split": {"rule": "EVERY row is train (protocol 2b: anchored correction-sheet labels "
                          "are TRAIN-SIDE ONLY). Not a stratified assignment and nothing to "
                          "re-derive — a seeded split here would only make it look as though "
                          "there had been a decision.",
                  "eval_rows": 0, "train_rows": len(rows),
                  "never": "this sheet must not join sheets D/E as an eval instrument, for "
                           "this head generation or any later one"},
        "seeds": {"draw_seed": spec.draw_seed, "id_salt": spec.id_salt,
                  "shuffle_seed": None,
                  "shuffle_note": "there is no shuffle — a correction sheet is SORTED, and "
                                  "the sort is a pure function of the head readout"},
        "render_defaults": {
            "width": BMS.W, "height": BMS.H, "ss": BMS.SS, "filter": BMS.FILT,
            "jpg_quality": BMS.JPG_Q, "interior_mode": "black", "composition": "center",
            "why_these_pins": "the July render-mode batches' own pins, which every other "
                              "batch in this corpus carries — a corpus whose parts differ in "
                              "geometry cannot be unioned, and the head pre-labeling this "
                              "sheet was trained on crops at these settings",
            "screen_geometry": list(SCREEN_GEOM),
        },
        "realized": {
            "rows_by_mode": dict(sorted(Counter(
                r["render"]["render_mode"] for r in rows).items())),
            "rows_by_kind": dict(sorted(Counter(
                r["provenance"]["mode_kind"] for r in rows).items())),
            "rows_by_partition": dict(sorted(Counter(
                r["provenance"]["partition"] for r in rows).items())),
            "rows_by_family": dict(sorted(Counter(
                r["render"]["fractal_type"] for r in rows).items())),
            "rows_by_hue_family": {f: sum(1 for r in rows
                                          if r["provenance"]["hue_family"] == f)
                                   for f in HF.FAMILIES},
            "rows_by_human_score": dict(sorted(Counter(
                r["provenance"]["source"]["human_score"] for r in rows).items())),
            "suggested_tier_hist": dict(sorted(Counter(
                r["suggested_tier"] for r in rows).items())),
            "would_pass_mining_gate": sum(1 for r in rows
                                          if r["head_mining_v1"]["would_pass_gate"]),
            "distinct_locations": len({r["provenance"]["location_key"] for r in rows}),
            "distinct_palettes": len({r["render"]["palette"] for r in rows}),
            "transfer_dropped_rows": sum(1 for r in rows
                                         if r["provenance"]["transfer_dropped"]),
            "cos_smooth": _cos_summary(rows),
            "pred": {"min": round(min(pred), 4), "max": round(max(pred), 4),
                     "p50": round(float(np.median(pred)), 4)},
            "p_ge3": {"min": round(min(r["p_ge3"] for r in rows), 4),
                      "max": round(max(r["p_ge3"] for r in rows), 4)},
        },
        "presentation": {
            "order": "sheet_order — DESCENDING pred (good -> bad), ties on the crop stem",
            "sorted_on": "the continuous head readout (pred), NOT the suggested tier",
            "contiguous": True,
            # sitting_builder.md 3: the FILE order IS the presentation order, so the page must
            # not reshuffle. Recorded here and served with `&order=file`.
            "presentation_order": "file",
            "image_id": "OPAQUE `<prefix><slot>_<hash8>` — slot is presentation position "
                        "(published anyway: the sheet is sorted), hash a salted digest of the "
                        "unit key, so the id encodes no mode, palette, family or band. "
                        "route.json maps it back.",
        },
        "labels_export": spec.labels_export,
        "labeling": {
            "ui": spec.ui_url,
            "mode": "CORRECTION — every row shows its suggested tier PREFILLED; Enter "
                    "confirms, 1-3 override. Only rows Matt acts on are exported; an "
                    "unreviewed suggestion never leaves the page as a label.",
            "bulk": "accept all remaining, and accept all BELOW THIS ROW — both behind a "
                    "confirm",
            "blind_rows": 0,
            "calibration_duplicates": 0,
            "export_download": "scores.json (the page's export button)",
            "save_export_as": spec.labels_export,
            "sidecar_written": spec.labels_sidecar,
            "merge": f"uv run python tools/wallpaper/merge_sitting.py "
                     f"--corpus render_mode_corpus --batch {spec.batch_id} "
                     f"--scores {spec.labels_export} --apply",
            "then": "the CUT is the follow-up prompt: read the label/score crossover off the "
                    "merged sidecar against `pred`/`p_ge3`. Nothing here moves a threshold.",
        },
        "not_yet_pooled": {
            "what": f"{spec.batch_id} is deliberately absent from "
                    f"near_dup_groups.BATCHES and mining_corpus.BATCH_TAG",
            "why": "it has no labels yet. Pooling an unlabeled batch into the training "
                   "corpus is a decision about a retrain, and it is made once the sidecar "
                   "exists — not as a side effect of building the sheet.",
        },
        "render_failures": errors,
        "run_status": {
            "drawn_rows": sel_rep["drawn_rows"], "rendered_rows": len(rows),
            "n_failures": len(errors), "n_unaccounted": len(unaccounted),
            "unaccounted_rows": unaccounted[:50],
            "unaccounted_note": "drawn but neither rendered nor failed — a bounded (--limit) "
                                "or interrupted run; re-run `render` then `write`",
        },
    }
    (spec.batch_dir / "batch.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print_summary(spec, batch)
    return batch


def _cos_summary(rows) -> dict:
    from tools.mining import smooth_equivalence as SE          # noqa: PLC0415
    cs = [r["provenance"]["cos_smooth"] for r in rows
          if r["provenance"]["cos_smooth"] is not None]
    if not cs:
        return {"n": 0}
    return {**SE.quantiles(cs),
            "bands": dict(Counter(r["provenance"]["smooth_band"] for r in rows))}


# =========================================================================== #
# Reporting.
# =========================================================================== #
def print_bill(bill):
    log("-" * 96)
    if bill.get("status"):
        log(f"RENDER BILL {bill['status']}")
        log("-" * 96)
        return
    log(f"RENDER BILL (basis {bill['basis']}, {bill['speedup_assumed']}x at 4 workers)")
    for name, s in bill["stages"].items():
        log(f"  {name:8} {s['single_process_min']:7.1f} min single-process -> "
            f"{s['wall_min_at_4x']:6.1f} min wall"
            + (f"   UNPRICED {s['unpriced_partitions']}" if s["unpriced_partitions"] else ""))
    log(f"  TOTAL    {bill['total_wall_min_at_4x']:.1f} min wall  "
        f"(+ GPU: {bill['excluded_from_the_bill']})")
    log("-" * 96)


def print_summary(spec, batch):
    r = batch["realized"]
    log("\n" + "=" * 96)
    log(f"SHEET F — {spec.batch_id}"
        + ("   *** INCOMPLETE ***" if batch["sheet_incomplete"] else ""))
    log("=" * 96)
    log(f"rows {batch['n_rows']} / drawn {batch['run_status']['drawn_rows']}  ·  failures "
        f"{batch['run_status']['n_failures']}")
    log(f"by mode:      {r['rows_by_mode']}")
    log(f"by kind:      {r['rows_by_kind']}")
    log(f"by partition: {r['rows_by_partition']}")
    log(f"human score:  {r['rows_by_human_score']}  ·  locations {r['distinct_locations']}  ·  "
        f"palettes {r['distinct_palettes']}")
    log(f"PREFILL {batch['anchored']['prefill_head']} tiers {r['suggested_tier_hist']}  ·  "
        f"pred {r['pred']}  ·  would-pass-gate {r['would_pass_mining_gate']}")
    log(f"SPLIT: 100% train (protocol 2b)  ·  sorted good->bad on pred, order=file")
    log(f"-> {spec.batch_dir}")
    log(f"-> serve: uv run python tools/viz/serve.py   then")
    log(f"   http://127.0.0.1:8010/{spec.ui_url}")


# =========================================================================== #
# Driver.
# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Base-rate audit correction sheet (sheet F).")
    ap.add_argument("stage", choices=("pool", "estimate", "screen", "select", "render", "write"))
    ap.add_argument("--sheet", default="f", choices=sorted(SHEETS))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap units this run. A short batch STAMPS itself sheet_incomplete "
                         "at write time.")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--dry-run", action="store_true", help="render: composition only")
    ap.add_argument("--unit-timeout", type=float, default=900.0)
    ap.add_argument("--wall-budget-s", type=float, default=4 * 3600.0)
    args = ap.parse_args(argv)

    if args.workers > WORKERS:
        raise SystemExit(f"[baserate] --workers {args.workers} exceeds the project process "
                         f"cap of {WORKERS} (CLAUDE.md).")
    missing = MR.missing_recipes()
    if missing:
        raise SystemExit(f"[baserate] roster/recipe mismatch: {missing}")

    spec = SHEETS[args.sheet]
    if not BR.is_registered(spec.batch_id):
        raise SystemExit(f"[baserate] {spec.batch_id} is NOT in the batch registry. Register "
                         f"it BEFORE building — an unregistered batch classifies fail-closed "
                         f"and its split story is lost.")
    spec.work.mkdir(parents=True, exist_ok=True)
    prio = cc.set_below_normal_priority()
    log(f"[baserate] {spec.batch_id} · priority {prio} · {args.workers} workers x "
        f"{ENGINE_THREADS} rayon threads")

    if args.stage == "pool":
        _pool, rep = BE.fresh_locations(spec)
        log(json.dumps(rep, indent=2))
        return 0
    if args.stage == "estimate":
        cands, twins, _lm, rep = BE.universe(spec)
        BE.print_universe(rep)
        bill = render_bill(spec, cands, twins)
        print_bill(bill)
        (spec.work / "universe.json").write_text(
            json.dumps({"universe": rep, "render_bill": bill}, indent=2), encoding="utf-8")
        log(f"-> {spec.work / 'universe.json'}")
        return 0
    if args.stage == "screen":
        BE.run_screen(spec, args)          # sheet E's screen: NO head is loaded in it
        return 0
    if args.stage == "select":
        _bk, _lm, _uni, _sel, rep, bill = _selected(spec, args)
        BE.print_composition(rep)
        print_bill(bill)
        (spec.work / "selection_report.json").write_text(
            json.dumps({"selection": rep, "render_bill": bill}, indent=2), encoding="utf-8")
        return 0
    if args.stage == "render":
        BE.run_render(spec, args)          # sheet E's render: NO head scores the result
        return 0
    if args.stage == "write":
        run_write(spec, args)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
