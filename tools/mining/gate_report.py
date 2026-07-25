"""Report-only mining gate — durable would-cut log paired with the final selection.

The strange-mode mining gate (`mining_v1`, threshold 0.50 on marginal p_ge3) is
REPORT-ONLY: it no longer admits / cuts / keeps. Instead, per strange candidate
reaching a release, it records HERE what it WOULD have cut and why, PAIRED with the
actual selection outcome, so a future calibration pass reads labeled gate precision
straight off accumulated releases — the gate's verdict against the selection (and,
once eye-routing is joined, the human) verdict — instead of starting a fresh labeling
session. See prompts/mining_gate_report_only.md.

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE_LOG_DIR = ROOT / "data" / "emission" / "mining_gate_reports"

MINING_GATE_VERSION = "mining_v1"


def gate_report_row(*, site, key, location, style, palette, p_ge3, release_threshold,
                    selected, selection_stage, pool_floor=None):
    """One report-only gate verdict paired with the actual selection outcome.

    `p_ge3`      mining head marginal P(label>=3) (None on a render error).
    `would_pass`/`would_cut` — the counterfactual verdict against the production RELEASE
                 threshold the gate WOULD have used (0.50); the gate did NOT act on it.
    `selected`   what actually happened (shipped / kept). The join target a calibration
                 pass reads precision off (`would_cut` ∧ `selected` = a gate false-cut the
                 selection overrode). `pool_floor` (build site) records the softer
                 pool-admission counterfactual (0.25) too; it never acts either.
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
        row["pool_floor"] = float(pool_floor)
        row["would_pass_pool"] = (p_ge3 is not None and float(p_ge3) >= float(pool_floor))
    return row


def write_gate_report(site, rows):
    """Upsert `rows` (keyed by 'key') into data/emission/mining_gate_reports/<site>.jsonl,
    rewritten sorted by key. Returns (path, n_total, n_would_cut, n_would_cut_selected)."""
    GATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = GATE_LOG_DIR / f"{site}.jsonl"
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
    return path, len(ordered), len(would_cut), len(wc_selected)
