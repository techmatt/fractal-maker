"""Durable gate/release record — which locations were gated, which were released, out of what.

THE GAP THIS CLOSES. The emission driver decides twice: at GATE time (does this colorized
candidate clear its head's pool floor?) and at RELEASE time (does an eligible candidate win one
of N slots?). Both decisions were written only under `--out`, which is a `scratch/` path in every
real invocation. Campaign-2's emission output was wiped, so that stage has **no record for any
run** — and the loss is not recoverable by re-running, because the decision was taken against a
pool and a population that no longer exist. It has already cost two answers this month: whether
mb4's admissions actually ship, and campaign-2's run-only julia look denominator (the second is
the *population* half — you cannot recover a rate when the denominator is what was deleted).

Under the durability contract (`tools/paths.py`) that is squarely `durable()`: it records a
population that no longer exists and cannot be regenerated from anything else. So it is written
through `paths.durable()`, which ASSERTS at the write site that git would keep the path — an
unregistered log gets wiped by exactly the derived-artifact chain that ate the originals.

RELATION TO `tools/mining/gate_report.py`. That is the same shape, and this copies its pattern
(upsert-by-`key`, rewritten sorted, idempotent under re-run) rather than inventing a second one.
It is NOT a substitute for this: it records only the *strange* (mining-head) candidates, only the
*counterfactual* verdict of a gate that no longer acts, and nothing about the population. The
gate that actually cuts — the wallpaper head's pool/release floors on smooth — has never had a
durable record at all.

WHAT ACCUMULATION MEANS HERE. `key` is prefixed with `run_id`, so:
  * two DIFFERENT runs never collide — both survive in the file, which is the accumulation the
    record exists for; and
  * a re-run or `--resume` of the SAME run re-derives identical keys and upserts in place, so a
    resumed run does not double-count itself.
Rows are only ever added or replaced by their own run; nothing else is dropped.

NO RETRO-FILL. The past runs are gone and are not reconstructed here. The record starts at the
first run after this lands. A row invented for campaign-2 would look exactly like a measurement
and be worth less than the absent one it replaced.

Written by `tools/emission/build_emission_diversity_v1.py` at both decision points.
Read: `data/emission/release_records/<site>.jsonl` (per-decision) and
      `data/emission/release_records/<site>__runs.jsonl` (per-run population).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import paths  # noqa: E402   the durability-class declaration

RECORD_DIR_REL = "data/emission/release_records"
SCHEMA_VERSION = 1

STAGE_GATE = "gate"
STAGE_RELEASE = "release"


def record_path(site: str) -> Path:
    """The per-decision log. `durable()` raises if git would discard it."""
    return paths.durable(f"{RECORD_DIR_REL}/{site}.jsonl", mkparents=True)


def runs_path(site: str) -> Path:
    """The per-run population log — the denominator half of the record."""
    return paths.durable(f"{RECORD_DIR_REL}/{site}__runs.jsonl", mkparents=True)


def _key(run_id: str, stage: str, join_key: str) -> str:
    return "|".join((str(run_id), str(stage), str(join_key)))


def decision_row(*, run_id, stage, join_key, location_id, location, partition,
                 morph_cluster, decision, score=None, reason=None, head=None,
                 floor=None, style=None, palette=None) -> dict:
    """One gate-time or release-time decision.

    `join_key`  the candidate's identity within the run (location_id|style|palette) — the join
                back to the pool row and forward to whatever shipped.
    `partition` the family / cloud partition (mandelbrot, multibrot{3,4,5}, julia:*, phoenix).
    `decision`  gate: `admitted` | `rejected`; release: `selected` | `not_selected`.
    `score`     the head probability the decision was taken on, None when there wasn't one
                (a render error is a decision with a reason and no score — recording it as 0.0
                would make a crash indistinguishable from a bad wallpaper).
    `reason`    why, in the cases where the score alone does not say it.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "key": _key(run_id, stage, join_key),
        "run_id": str(run_id),
        "stage": stage,
        "join_key": str(join_key),
        "location_id": location_id,
        "location": location,
        "partition": partition,
        "morph_cluster": morph_cluster,
        "render_style": style,
        "palette": palette,
        "head": head,
        "decision": decision,
        "score": None if score is None else round(float(score), 6),
        "floor": None if floor is None else float(floor),
        "reason": reason,
    }


def run_row(*, run_id, site, out_dir, ledgers, counts, floors, ts=None) -> dict:
    """The population a run's decisions were taken out of.

    `counts` is the funnel — every stage's denominator, not just the survivors. Without it a
    later reader can count what passed and never learn what it passed out of, which is the exact
    shape of the campaign-2 julia-look question."""
    return {
        "schema_version": SCHEMA_VERSION,
        "key": str(run_id),
        "run_id": str(run_id),
        "site": site,
        "out_dir": out_dir,
        "ledgers": list(ledgers or []),
        "counts": dict(counts or {}),
        "floors": dict(floors or {}),
        "ts": ts,
    }


def _upsert(path: Path, rows) -> tuple[int, int]:
    """Merge `rows` into `path` by `key`, rewritten sorted by key. Returns (n_total, n_new).

    Same pattern as `mining.gate_report.write_gate_report`: idempotent on an unchanged run
    (byte-identical output), additive across runs (their keys differ by the run_id prefix)."""
    merged: dict = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                merged[r["key"]] = r
    before = set(merged)
    for r in rows:
        merged[r["key"]] = r
    ordered = [merged[k] for k in sorted(merged)]
    path.write_text("".join(json.dumps(r) + "\n" for r in ordered), encoding="utf-8")
    return len(ordered), len(set(merged) - before)


def write_decisions(site: str, rows) -> tuple[Path, int, int]:
    path = record_path(site)
    n_total, n_new = _upsert(path, rows)
    return path, n_total, n_new


def write_run(site: str, row: dict) -> tuple[Path, int, int]:
    path = runs_path(site)
    n_total, n_new = _upsert(path, [row])
    return path, n_total, n_new


def read_decisions(site: str, run_id: str | None = None) -> list:
    """Every recorded decision, optionally for one run. Read side for a later calibration
    pass; the record is useless if nothing can get at it without re-parsing by hand."""
    path = paths.durable(f"{RECORD_DIR_REL}/{site}.jsonl")
    if not Path(path).exists():
        return []
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if run_id is None or r["run_id"] == run_id]
