"""Mining-gate verdict log — durable, per strange candidate, paired with what happened.

Per strange candidate reaching a release, this records what the gate (`mining_v1`,
threshold 0.50 on marginal p_ge3) did to it and why, PAIRED with the actual outcome at
that site.

WHAT THE PAIRING IS WORTH DEPENDS ON WHETHER THE GATE ACTS, and it changed on 2026-08-06.
While the gate was REPORT-ONLY (prompts/mining_gate_report_only.md) it cut nothing, so a
`would_cut` row could still be SELECTED and `would_cut ∧ selected` accrued a labeled
false-cut count on every run — precision read off accumulated releases instead of a fresh
labeling session. The gate is ENFORCING again (prompts/mining_adoption_prompt.md,
`floors.MINING_RELEASE`), so selection now implies passing and BOTH outcome joins
(`would_cut ∧ selected`, and `would_cut_pool ∧ selected` since 0.25 < 0.50) are zero by
construction. Nothing here changes shape: the file is still written on every run and is
still the durable population record — every scored strange candidate, its p_ge3, and what
each floor did to it. What a calibration pass can no longer get for free is the labels; it
has to bring them.

Why here and not scratch/: the log must survive `rm -r scratch/*`, so it is committed under
`data/emission/` (NOT the disposable scratch/ tree where every existing per-candidate
score already lives and is silently wiped). See CLAUDE.md "Persistent-store convention".

Idempotent + accumulating: rows are UPSERT-by-`key` then rewritten sorted. A re-run over
an unchanged corpus re-derives identical rows -> the file is byte-identical (so a
deploy_tail no-op stays a no-op); a grown corpus adds new keys and preserves the rest.
Deterministic because the mining head is fp32/no-autocast (mining_gate.py) -> p_ge3 is
reproducible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GATE_LOG_DIR = ROOT / "data" / "emission" / "mining_gate_reports"

from tools.emission import emission_sinks as esinks  # noqa: E402  central sink-isolation


def gate_log_dir() -> Path:
    """WHERE this log lands, resolved through the sink binding rather than read off the
    module constant. Production (nothing bound) is `GATE_LOG_DIR` verbatim — the constant
    stays, and stays monkeypatchable, because tests and `durability_map` both name it. A
    smoke run binds an ephemeral root (`emission_sinks.use`) and this follows it, so a
    throwaway run cannot append rows to the accumulating production record."""
    if esinks.is_production():
        return GATE_LOG_DIR
    return esinks.record_root(ROOT) / esinks.MINING_GATE_REPORTS

# The gate version stamped into every durable row. IMPORTED from the pin (torch-free), never
# restated: this file used to carry its own `"mining_v1"` literal beside `mining_gate.py`'s,
# so a pin flip to v2 would have moved the checkpoint and left this log claiming v1 forever.
from tools.mining.mining_pins import MINING_GATE_VERSION  # noqa: E402


def gate_report_row(*, site, key, location, style, palette, p_ge3, release_threshold,
                    selected, selection_stage, pool_floor=None, pooled=None):
    """One gate verdict paired with the actual outcome, at BOTH cut sites.

    `p_ge3`      mining head marginal P(label>=3) (None on a render error).

    RELEASE site:
    `would_pass`/`would_cut` — the verdict against the production RELEASE threshold (0.50).
                 The field names are the report-only period's and are KEPT: a name change
                 would split every accumulated file into two schemas over a fact the
                 `release_threshold` beside it already carries.
    `selected`   what actually happened (shipped / kept).

    POOL site (a HARD cut — capacity ordering, not curation):
    `pool_floor` / `would_pass_pool` / `would_cut_pool` — the same verdict against the softer
                 pool-admission floor (0.25), and
    `pooled`     what actually happened AT THAT SITE (did the row enter the gated pool).

    BOTH `∧ selected` joins are now zero by construction (the release floor enforces, and
    clearing 0.50 implies clearing 0.25), so neither is a measurement any more — see the
    module docstring. The pairing is still recorded, for the reason it was worth recording
    at the pool site even while that floor acted: a row that records only "0.25 would have
    cut this" and never records what happened cannot be joined later, and either floor can
    move or go report-only again. `pooled=None` (site did not report an outcome) is preserved
    as null rather than coerced to False, so "not pooled" and "nobody said" stay
    distinguishable.
    """
    would_pass = p_ge3 is not None and float(p_ge3) >= float(release_threshold)
    row = {
        "site": site,
        "key": key,
        "location": location,
        "style": style,
        "palette": palette,
        "gate_version": MINING_GATE_VERSION,
        "p_ge3": None if p_ge3 is None else round(float(p_ge3), 6),
        "release_threshold": float(release_threshold),
        "would_pass": would_pass,
        "would_cut": not would_pass,
        "selected": bool(selected),
        "selection_stage": selection_stage,
    }
    if pool_floor is not None:
        wp_pool = (p_ge3 is not None and float(p_ge3) >= float(pool_floor))
        row["pool_floor"] = float(pool_floor)
        row["would_pass_pool"] = wp_pool
        row["would_cut_pool"] = not wp_pool
        row["pooled"] = None if pooled is None else bool(pooled)
    return row


def write_gate_report(site, rows):
    """Upsert `rows` (keyed by 'key') into data/emission/mining_gate_reports/<site>.jsonl,
    rewritten sorted by key.

    Returns `(path, n_total, n_would_cut, n_would_cut_selected, pool_counts)`, where
    `pool_counts` is the same pairing accrued at the POOL site — `{n_with_pool_site,
    n_would_cut_pool, n_would_cut_pool_pooled, n_would_cut_pool_selected}` — and is all zeros
    for a site that logs no pool floor (`deploy_tail` has no pool stage)."""
    log_dir = gate_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{site}.jsonl"
    merged: dict = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                merged[r["key"]] = r
    for r in rows:
        merged[r["key"]] = r
    ordered = [merged[k] for k in sorted(merged)]
    path.write_text("".join(json.dumps(r) + "\n" for r in ordered), encoding="utf-8")
    would_cut = [r for r in ordered if r.get("would_cut")]
    wc_selected = [r for r in would_cut if r.get("selected")]
    pool_rows = [r for r in ordered if r.get("pool_floor") is not None]
    # `would_cut_pool` is DERIVED at read time when absent: rows accrued before the pool
    # pairing landed carry `would_pass_pool` but not its complement, and a file is a MIX of
    # both formats after any partial re-run (the upsert preserves untouched keys). Trusting
    # the stored field alone would silently count every legacy row as "not would-cut".
    def _wc_pool(r):
        if "would_cut_pool" in r:
            return bool(r["would_cut_pool"])
        return not r.get("would_pass_pool", True)
    wc_pool = [r for r in pool_rows if _wc_pool(r)]
    pool_counts = {
        "n_with_pool_site": len(pool_rows),
        "n_would_cut_pool": len(wc_pool),
        "n_would_cut_pool_pooled": sum(1 for r in wc_pool if r.get("pooled")),
        "n_would_cut_pool_selected": sum(1 for r in wc_pool if r.get("selected")),
    }
    return path, len(ordered), len(would_cut), len(wc_selected), pool_counts
