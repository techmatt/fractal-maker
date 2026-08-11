r"""eval_only.py — THE eval-only batch rule: a batch stamped `eval_only` is EVAL, always.

A blind slice bought to referee two heads is spent the moment it enters a train split, and
the failure is silent — the head trains on it, every later read off it is inflated, and
nothing is red. `classifier_retrain_protocol.md` §2a says a pooled run must either re-derive
the split globally or freeze one batch set as authority; **neither of those two fixes has
anything to say about an eval-only batch**, and both would happily place one on the train
side. So the pin is a THIRD constraint that outranks both, and it lives here rather than in
each split pass, because "unconditionally" across N split passes means one owner.

WHAT A BATCH DECLARES, and where. `batch.json` carries `eval_only: true` plus
`eval_only_note` (the reason, in the builder's words), and every row carries
`provenance.split_side == "eval"`. The declaration is the BATCH's; the row stamps are what a
loader that never opens `batch.json` still sees. Both are checked, because they are two
different ways to lose the property: `check_stamps` catches a batch whose rows disagree with
its own flag, `assert_eval` catches a split pass that overrode correct stamps.

    from tools.corpus.eval_only import eval_only_batches, eval_only_ids, assert_eval

    forced = eval_only_ids("wallpaper_corpus")             # {image_id: batch_id}
    assert_eval(side_by_image_id, forced, where="split_v4b")

`key_of` exists because the two corpora key their splits differently — the render-mode split
is over `provenance.location_key`, the wallpaper split over `image_id` / the c-inclusive
coordinate — and a rule that only speaks one of those dialects is a rule one corpus ignores.

Torch-free, numpy-free, stdlib only: a split guard that costs the GPU stack to import is a
guard someone routes around. The corpus list is IMPORTED from `merge_sitting.CORPORA` (the
existing owner of "which corpora exist, and what tier ceiling each has"), never restated.

    uv run pytest tools/corpus/test_eval_only.py -q
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.wallpaper.merge_sitting import CORPORA  # noqa: E402  THE corpus registry

KNOWN_CORPORA = tuple(sorted(CORPORA))


class EvalOnlyViolation(AssertionError):
    """Raised where the pin is broken. AssertionError so a split pass that catches nothing
    still dies, and a named subclass so a test can assert the reason rather than the text."""


@dataclass(frozen=True)
class EvalOnlyBatch:
    corpus: str
    batch_id: str
    reason: str
    n_rows: int

    @property
    def rel(self) -> str:
        return f"data/{self.corpus}/batches/{self.batch_id}"


def batches_dir(corpus: str, root: Path | None = None) -> Path:
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r} — known: {list(KNOWN_CORPORA)}. "
                         f"Add it to merge_sitting.CORPORA (the one registry) first.")
    return (root or ROOT) / "data" / corpus / "batches"


def _rows(bdir: Path) -> list[dict]:
    p = bdir / "images.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def eval_only_batches(corpus: str, root: Path | None = None) -> dict[str, EvalOnlyBatch]:
    """`{batch_id: EvalOnlyBatch}` — GLOBBED off the corpus tree, never a constant list.

    A hardcoded list is how the eighth batch silently stops being pinned (the same reason
    `build_blind_minibrot_sheet.prior_wallpaper_locations` globs its exclusion set). A batch
    that stamps `eval_only` without a reason is a violation, not a batch with an empty
    reason: the note is the only record of what the slice was bought for."""
    out: dict[str, EvalOnlyBatch] = {}
    bdir = batches_dir(corpus, root)
    if not bdir.exists():
        return out
    for bj in sorted(bdir.glob("*/batch.json")):
        doc = json.loads(bj.read_text(encoding="utf-8"))
        if not doc.get("eval_only"):
            continue
        reason = (doc.get("eval_only_note") or "").strip()
        if not reason:
            raise EvalOnlyViolation(
                f"{bj.parent.name} stamps eval_only with no `eval_only_note` — the reason "
                f"a slice may never train is the one thing a future retrain will need to "
                f"read, and it cannot be reconstructed from the flag.")
        out[bj.parent.name] = EvalOnlyBatch(corpus=corpus, batch_id=bj.parent.name,
                                            reason=reason, n_rows=len(_rows(bj.parent)))
    return out


def is_eval_only(corpus: str, batch_id: str, root: Path | None = None) -> bool:
    return batch_id in eval_only_batches(corpus, root)


def eval_only_ids(corpus: str, *, key_of=None, batch_ids=None,
                  root: Path | None = None) -> dict:
    """`{key: batch_id}` over every row of every eval-only batch.

    `key_of(row)` defaults to the `image_id`; pass e.g.
    `lambda r: r["provenance"]["location_key"]` for a split keyed on locations. A row whose
    key comes back None is dropped — a corpus whose rows carry no such key simply has
    nothing to pin under that dialect, which is different from having nothing to pin."""
    key_of = key_of or (lambda r: r["image_id"])
    out: dict = {}
    for bid, blk in eval_only_batches(corpus, root).items():
        if batch_ids is not None and bid not in batch_ids:
            continue
        for r in _rows(batches_dir(corpus, root) / bid):
            k = key_of(r)
            if k is not None:
                out[k] = blk.batch_id
    return out


def coord_key(row) -> tuple:
    """The c-INCLUSIVE coordinate key the wallpaper split already uses for disjointness
    (`train_wallpaper_v4.WRow.full_coord`), built straight off the render block.

    Keying the pin on the coordinate rather than the `image_id` is what makes it survive a
    RE-RENDER: sheet D excluded every prior location, but nothing stops a future batch from
    drawing a sheet-D location again under a new id, and pinning ids alone would let that
    copy train while the original sits in eval — the instrument spent by a row that never
    named it."""
    rd = row["render"]
    return (rd["cx"], rd["cy"], rd["fw"], rd["fractal_type"], rd.get("c_re"), rd.get("c_im"))


def check_stamps(corpus: str, root: Path | None = None) -> dict:
    """Every row of every eval-only batch stamps `provenance.split_side == "eval"`.

    Returns a report; `assert_stamps` is the raising form. This is the check that holds
    while no trainer has loaded the batch yet — the row stamps are the only thing a loader
    that never opens `batch.json` will see."""
    rep = {"corpus": corpus, "batches": {}, "n_batches": 0, "n_rows": 0, "violations": []}
    for bid, blk in eval_only_batches(corpus, root).items():
        rows = _rows(batches_dir(corpus, root) / bid)
        bad = [r["image_id"] for r in rows
               if (r.get("provenance") or {}).get("split_side") != "eval"]
        rep["batches"][bid] = {"n_rows": len(rows), "n_not_stamped_eval": len(bad),
                               "reason": blk.reason}
        rep["n_batches"] += 1
        rep["n_rows"] += len(rows)
        rep["violations"] += [{"batch": bid, "image_id": i} for i in bad[:20]]
    rep["ok"] = not rep["violations"]
    return rep


def assert_stamps(corpus: str, root: Path | None = None) -> dict:
    rep = check_stamps(corpus, root)
    if not rep["ok"]:
        raise EvalOnlyViolation(
            f"[{corpus}] {len(rep['violations'])} row(s) of an eval-only batch are not "
            f"stamped split_side=eval, e.g. {rep['violations'][:3]}")
    return rep


def pin(side: dict, forced, *, where: str) -> dict:
    """Force every forced key to "eval" IN PLACE and report what moved.

    The report is the point. A pin that silently corrects a split pass hides the fact that
    the pass wanted to train on a blind slice, and that fact is worth a line in a manifest:
    `n_moved > 0` means some other rule tried, and which rule is worth knowing."""
    forced = dict(forced) if isinstance(forced, dict) else {k: None for k in forced}
    moved = []
    for k in forced:
        if k not in side:
            continue
        if side[k] != "eval":
            moved.append({"key": str(k), "was": side[k], "batch": forced[k]})
            side[k] = "eval"
    return {"where": where, "n_forced_keys": len(forced),
            "n_present_in_split": sum(1 for k in forced if k in side),
            "n_moved_to_eval": len(moved), "moved": moved[:50],
            "rule": "eval_only batches are pinned to the eval side unconditionally "
                    "(classifier_retrain_protocol.md §2a); this outranks both the global "
                    "re-derivation and the frozen-authority fix"}


def assert_eval(side: dict, forced, *, where: str) -> dict:
    """Raise unless every forced key present in `side` is on the eval side.

    The check a split pass runs on the split it BUILT, not a claim the pin makes about
    itself (`verification_practice.md`): `pin` and `assert_eval` are deliberately separate
    so a pass that never called `pin` still dies here instead of passing silently."""
    bad = [{"key": str(k), "side": side[k], "batch": b}
           for k, b in (forced.items() if isinstance(forced, dict)
                        else ((k, None) for k in forced))
           if k in side and side[k] != "eval"]
    present = sum(1 for k in (forced if not isinstance(forced, dict) else forced)
                  if k in side)
    if bad:
        raise EvalOnlyViolation(
            f"[{where}] {len(bad)} key(s) of an EVAL-ONLY batch landed on the train side, "
            f"e.g. {bad[:3]}. A blind slice is spent the moment it trains — fix the split "
            f"pass, never the stamp.")
    return {"where": where, "n_forced_keys": len(forced), "n_present_in_split": present,
            "ok": True}
