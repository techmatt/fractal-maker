#!/usr/bin/env python
r"""maneuver_view_screen.py — the VIEW-level screen wired for a LIVE walk.

WHAT MOVED, AND WHY IT IS A DIFFERENT MODULE FROM `maneuver_screen.py`. That one screens
the **atom**: one 64x36 field per nucleus on its 4x frame, shared across every `k` row,
because the score describes the atom and not the picture (`minibrot_maneuvers.md` §3.1).
This one screens the **view**: one field per `(atom, k)` at the frame that is actually
pushed, scored by `view_screen.composite_v3`. The two are not variants of one measurement —
they answer different questions on different frames, and pooling them would be exactly the
cap/geometry error `orbital_field_metrics.md` §5 and §7 forbid. So they are separate
modules with separate caches, and a row records which one it came from (`screen_frame`).

THE COST, MEASURED BEFORE IT WAS COMMITTED TO. A view screen is ~3x the fields of an atom
screen (three `k` per nucleus, one field each, against one field shared). On the 840-batch
exploration run the atom screen cost 643 s of 24,846 s active (2.6%) for 5,627 fields; the
same population is 16,440 views, so the view screen prices at ~1,880 s, **~7.6% of active**.
That is the whole basis for keeping the k-set at three and for not ALSO running the atom
screen when this one is on: the two together would cross the 10% line the screen is
budgeted at. `[measured: state.json of data/discovery/maneuver_v14_exploration; 2026-08-01]`

BATCHING IS THE ONLY THING THAT MAKES IT AFFORDABLE, and the reason is that the cost is a
process spawn (~130 ms) around ~2 ms of compute. One pass per batch over every DISTINCT
view the batch enumerated, four concurrent engine processes, one thread each. A per-row
screen would pay the spawn once per k and once per repeat visit to the same nucleus.

RECORDING, NEVER A GATE — inherited from `maneuver_screen.py` and restated because it is
the property the run depends on: a view the 64 px geometry cannot reach comes back as a row
with `screened=False` and a named reason, and it is still a candidate. Nothing here refuses
a maneuver, and nothing here is allowed to raise.

THE FIELD RIDES ALONG. Every successful screen hands its raw f32 field to the run-local
`RunFieldCache`, so the population the run screened is on disk as arrays and the next
per-tile statistic is a numpy pass rather than a second engine pass over the whole run
(`view_field_cache.py`).
"""
from __future__ import annotations

import math
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "tools" / "orbital", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import field_metrics as fm           # noqa: E402
import maneuver_screen as msc        # noqa: E402  (the cap policy + the spawn-shaped pool)
import view_screen as vs             # noqa: E402  (the measures and composite_v3)
import view_fit as vfit              # noqa: E402  (the staged v1.1 fitted score)

SCREEN_WORKERS = msc.SCREEN_WORKERS  # concurrent engine PROCESSES (the CLAUDE.md cap)
SCREEN_THREADS = msc.SCREEN_THREADS  # per process: a 64x36 field wants exactly one


def view_key(atom_key, k) -> str:
    """The view's identity: `(atom, framing)`. Byte-identical to `view_field_cache.row_key`
    and to `steered_frontier`'s `man_visited` key, on purpose — the cache, the visited set
    and the screen must agree on what one candidate is."""
    return f"{atom_key}|{k}"


# The columns kept in the resumable checkpoint. NOT every measure: the full record rides
# `maneuvers.jsonl` (append-only, durable) and the raw field rides the field cache, so
# state.json only has to carry what SELECTION reads back after a resume. The alternative
# was measured against: the atom screen's full records are 3.7 MB of state.json at 5,627
# rows, and the view population is ~3x that with ~2x the columns — a per-batch checkpoint
# rewriting ~25 MB, 840 times, to re-derive numbers already on disk twice.
STATE_KEYS = ("screened", "screen_reason", "composite", "vetoed", "size_factor",
              "radial_range", "radial_rings", "interior_fraction",
              "band_coverage", "band_coverage_q25", "view_fw", fm.POLICY_KEY,
              # harvest v2: BOTH sourcing scores ride every screened row (see `_score`).
              "view_fit", "view_fit_p_notbad", "view_fit_model", "view_fit_reason")


def compact(rec: dict) -> dict:
    return {k: rec[k] for k in STATE_KEYS if k in rec}


def screen_view(cx, cy, fw, *, family: str = "mandelbrot", threads: int = SCREEN_THREADS,
                timeout: float = fm.FIELD_TIMEOUT_S, tmpdir=None):
    """`(record, field)` for one view at its own frame. Never raises; `field` may be None.

    Split from `view_screen.measure_view` for exactly one reason — that function throws the
    array away, and the array is what the field cache exists to keep. The measurement itself
    is `view_screen.measure_view_from_field`, called verbatim, so a row measured here and a
    row measured by the retrospective driver are the same computation on the same array.
    """
    meta = vs.view_frame_policy(fw)
    if not meta["screened"]:
        return meta, None
    try:
        with tempfile.TemporaryDirectory(dir=tmpdir) as td:
            field, _side = fm.dump_field(cx, cy, float(fw), meta["view_maxiter"],
                                         Path(td) / "f.bin", width=fm.SCREEN_W,
                                         height=fm.SCREEN_H, ss=fm.SCREEN_SS,
                                         family=family, threads=threads,
                                         timeout=max(1.0, float(timeout)))
    except Exception as e:
        return dict(meta, screened=False, screen_reason=f"dump_field:{str(e)[:120]}"), None
    return vs.measure_view_from_field(fw, field), field


class ViewScreenCache:
    """Run-scoped view screens keyed on `view_key`, with the composite already applied.

    Keyed on the VIEW and not on the atom — that is the entire difference from
    `maneuver_screen.ScreenCache`, and it is why the two caches cannot share state: the atom
    cache's key deliberately ignores `k` because its frame does not depend on it, and this
    one's frame is nothing but `k`.

    `params` is the reference-derived `ScreenParams` (veto + winsorization caps), resolved
    once per run rather than per row — re-reading `view_screen_refs.json` 16,000 times to
    re-derive the same four numbers is the shape of cost this module is trying to avoid.
    """

    def __init__(self, params, *, workers: int = SCREEN_WORKERS,
                 threads: int = SCREEN_THREADS, fields=None, fit_model=None):
        self.params = params
        self.by_key: dict[str, dict] = {}
        self.workers, self.threads = int(workers), int(threads)
        self.fields = fields                 # RunFieldCache | None
        # The staged `view_fit_v1.1` model. RECORDED, never a sort key: the pre-registered
        # adoption bar (delta-AP >= +0.1181 on labelled outcome) has not been read, and the
        # q4 sitting could not read it because NEITHER score existed on any row. That is the
        # gap this closes — carrying both columns is what makes the bar readable at the first
        # v2 sitting. `None` disables it; the column then says WHY it is absent rather than
        # silently not appearing.
        self.fit_model = fit_model
        self.n_hits = self.n_screened = self.n_fields_cached = 0
        self.n_view_fit = 0
        self.screen_s = 0.0

    def get(self, key: str):
        return self.by_key.get(key)

    def _view_fit(self, rec: dict, field, window_scale) -> dict:
        """`view_fit_v1.1` on this row. Never raises — a scoring failure must cost the column,
        never the candidate, exactly as a field-cache failure must not cost the screen.

        Two of the twelve features are not in the screen record and are supplied here:
        `falloff_rate`/`falloff_half` come from `view_fit.falloff_features(field)` on the SAME
        array the measures were taken from (so a cached re-score and a live score are the same
        computation), and `log10_size_rel` needs the ATOM's `window_scale`, which only a
        maneuver-originated view has. A row without one is not dropped: the model's own
        recorded `impute` median covers it and `view_fit_reason` names the substitution, so a
        later read can partition on it instead of discovering it as a distribution shift."""
        if self.fit_model is None:
            return dict(view_fit=None, view_fit_p_notbad=None, view_fit_model=None,
                        view_fit_reason="no_model_loaded")
        try:
            fw = float(rec.get("view_fw") or 0.0)
            reason = None
            feats = {
                "band_coverage": float(rec["band_coverage"]),
                "band_coverage_q25": float(rec["band_coverage_q25"]),
                "log1p_radial_range": math.log1p(max(0.0, float(rec["radial_range"]))),
                "log1p_radial_rings": math.log1p(max(0.0, float(rec["radial_rings"]))),
                "interior_fraction": float(rec["interior_fraction"]),
                "log10_fw": math.log10(fw) if fw > 0 else 0.0,
                "cap_headroom": (float(rec["cap_headroom"])
                                 if rec.get("cap_headroom") is not None else float("nan")),
                "clamped": 1.0 if rec.get("clamped") else 0.0,
                "composite_v3": float(rec["composite"]),
            }
            if window_scale and fw > 0:
                feats["log10_size_rel"] = math.log10(float(window_scale) / fw)
            else:
                feats["log10_size_rel"] = float("nan")
                reason = "imputed:log10_size_rel"
            if field is not None:
                ff = vfit.falloff_features(field)
                feats["falloff_rate"] = float(ff["falloff_rate"])
                feats["falloff_half"] = float(ff["falloff_half"])
            else:
                feats["falloff_rate"] = feats["falloff_half"] = float("nan")
                reason = ("imputed:falloff" if reason is None
                          else reason + "+falloff")
            return dict(view_fit=round(self.fit_model.score(feats), 6),
                        view_fit_p_notbad=round(self.fit_model.p_notbad(feats), 6),
                        view_fit_model=vfit.MODEL_ID_V11, view_fit_reason=reason)
        except Exception as e:                                   # noqa: BLE001
            return dict(view_fit=None, view_fit_p_notbad=None,
                        view_fit_model=vfit.MODEL_ID_V11,
                        view_fit_reason=f"error:{type(e).__name__}:{str(e)[:80]}")

    def _score(self, rec: dict, *, field=None, window_scale=None) -> dict:
        """Attach BOTH sourcing scores plus the terms a readout needs to explain them.

        `composite` is `composite_v3` — the LIVE sort key (`view_screen`'s module doc; v4 was
        measured and rejected). `view_fit` is the staged v1.1 fitted score, recorded beside
        it and NOT used to order anything: the pre-registered bar decides that, and it reads
        at a sitting's labels, not here. Recording both is the whole point — the q4 sitting
        recorded neither, so the bar was unreadable and NOT-ADOPT was the absence of evidence
        rather than a measured loss.

        An unscreened row gets `composite=None`, never a sentinel number: a sentinel would
        sort somewhere, and "we could not measure this" is not a position on the quality axis.
        Selection handles the None by sorting it last, which is a selection decision and
        belongs there, not here. `view_fit` is None there for the same reason AND because it
        takes `composite_v3` as one of its own twelve features.
        """
        if not rec.get("screened"):
            return dict(rec, composite=None, vetoed=None, size_factor=None,
                        view_fit=None, view_fit_p_notbad=None, view_fit_model=None,
                        view_fit_reason="unscreened")
        p = self.params
        out = dict(rec,
                   composite=round(float(vs.composite_v3(rec, p)), 6),
                   vetoed=bool(vs.is_vetoed(rec, p.veto)),
                   size_factor=round(float(vs.size_factor(rec, p)), 6))
        out.update(self._view_fit(out, field, window_scale))
        if out.get("view_fit") is not None:
            self.n_view_fit += 1
        return out

    def screen_many(self, jobs: list[dict], *, budget_s: float | None = None) -> dict:
        """`jobs` are `{view_key, cx, cy, fw, family, ...}`. Returns view_key -> record.

        `budget_s` bounds the WHOLE pass and each screen's own timeout is clamped to what is
        left of it, for the reason `ScreenCache.screen_many` states: the walk checks its cap
        BETWEEN batches, so anything unbounded inside one is outside the cap. A job the pass
        never reached comes back as an unscreened row with a named reason, never as a silent
        absence — a missing key downstream is indistinguishable from a key that was never
        proposed, and those are different facts.
        """
        todo, out = {}, {}
        for j in jobs:
            key = j["view_key"]
            hit = self.by_key.get(key)
            if hit is not None:
                self.n_hits += 1
                out[key] = hit
            elif key not in todo:
                todo[key] = j
        if not todo:
            return out
        t0 = time.time()
        items = list(todo.values())
        deadline = (t0 + float(budget_s)) if budget_s else None

        def one(j):
            if deadline is not None:
                left = deadline - time.time()
                if left <= 0:
                    return j, dict(screened=False, screen_reason="screen_budget_exhausted",
                                   **{fm.POLICY_KEY: msc.screen_policy_token()}), None
                per = min(fm.FIELD_TIMEOUT_S, max(1.0, left))
            else:
                per = fm.FIELD_TIMEOUT_S
            rec, field = screen_view(j["cx"], j["cy"], j["fw"],
                                     family=j.get("family", "mandelbrot"),
                                     threads=self.threads, timeout=per)
            return j, rec, field

        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(items)))) as ex:
            for j, rec, field in ex.map(one, items):
                key = j["view_key"]
                rec = self._score(rec, field=field, window_scale=j.get("window_scale"))
                self.by_key[key] = rec
                out[key] = rec
                self.n_screened += 1
                if field is not None and self.fields is not None:
                    # A field-cache failure must never fail a screen: the cache is an
                    # accelerator for later work, and losing a row of it costs one numpy
                    # pass, while raising here would cost the candidate.
                    try:
                        if self.fields.put(key, field, cx=str(j["cx"]), cy=str(j["cy"]),
                                           fw=float(j["fw"]),
                                           partition=j.get("family", "mandelbrot"),
                                           atom_key=j.get("atom_key"), k=j.get("k"),
                                           maxiter=rec.get("view_maxiter")):
                            self.n_fields_cached += 1
                    except Exception:
                        pass
        self.screen_s += time.time() - t0
        return out

    # -- checkpoint ---------------------------------------------------------- #
    def state_dict(self) -> dict:
        return dict(by_key={k: compact(v) for k, v in self.by_key.items()},
                    n_hits=self.n_hits, n_screened=self.n_screened,
                    n_fields_cached=self.n_fields_cached, n_view_fit=self.n_view_fit,
                    screen_s=round(self.screen_s, 3))

    def load_state(self, d: dict):
        self.by_key = dict(d.get("by_key") or {})
        self.n_hits = int(d.get("n_hits", 0))
        self.n_screened = int(d.get("n_screened", 0))
        self.n_fields_cached = int(d.get("n_fields_cached", 0))
        self.n_view_fit = int(d.get("n_view_fit", 0))
        self.screen_s = float(d.get("screen_s", 0.0))


def composite_sort_key(man: dict) -> tuple:
    """Descending-sort key for a maneuver node by `composite_v3`.

    UNSCREENED SORTS LAST AND IS NEVER EXCLUDED — the same contract the atom screen's quota
    sort keeps, and the reason a totally unscreenable population still behaves like the
    plain operator instead of stalling. `-inf` rather than 0.0 for the composite of an
    unscreened row, because a VETOED row scores in `[-1, 0)` and must still outrank a row
    that was never measured; conflating them would let the veto band act as a floor.
    """
    if not man.get("screened"):
        return (0, float("-inf"))
    c = man.get("composite")
    return (1, float("-inf") if c is None else float(c))
