# Minibrot roster v2 — the G-transfer readout (487 labels)

*2026-07-27. Batch `2026-07-26_minibrot_roster_v2`, 487 crops labeled 1–4, blind.*
Numbers reproduce with `uv run python tools/corpus/minibrot_roster_v2_readout.py`.
Companion sheets: `uv run python tools/corpus/minibrot_roster_v2_sheets.py`.

**The one question.** Stage-1 **G** was calibrated on non-minibrot fields. Does it mean
anything on minibrot fields — i.e. was the "positive arm" (the screen's accepts) actually a
positive arm? The prompt's prior: the anchor pass returned 8 minibrot accepts all scored 2,
none reaching 3, which read as *no*. With 487 labels the answer is **weak yes as a coarse gate,
firm no as a ranker** — and G is not the variable that actually predicts label here. **Degree is.**

Small n in most cells. These are counts, not rates; nothing is tested for significance.

## Join

487/487 rows join to the draw manifest (`data/minibrot_roster/batch_v1/draw.jsonl`) and to the
roster (`roster.jsonl`, for `log10|A|`). Every row carries arm, degree, period, band, log10|A|,
source atom, split. The only null field is **G on the 25 OOD-masked rows** — the screen never
assigned them a G (they were mask-excluded, not G-scored), which is expected, not a join failure.
G statistics below run over the 462 G-bearing rows.

## 1. Label distribution by arm

| arm | L1 | L2 | L3 | L4 | n | mean |
|---|---|---|---|---|---|---|
| accept | 60 | 129 | 46 | 2 | 237 | **1.96** |
| screen-reject (mask-surviving) | 76 | 123 | 26 | 0 | 225 | **1.78** |
| OOD-masked | 24 | 0 | 1 | 0 | 25 | **1.08** |
| **all** | 160 | 252 | 73 | 2 | 487 | 1.83 |

**Headline: accepts do outscore rejects — but only just, +0.25 mean** (1.96 vs 1.71 pooled over
both reject arms). The accept arm holds all the class-4s and a richer class-3 tail (46 vs 26).
The OOD-masked arm is nearly all 1s (mean 1.08) — masking removed genuine junk, so that part of
the screen works. So the accept/reject split is *not* noise, but the accept arm is dominated by
2s (129/237); "accept" buys you a mean-okay crop, not a good one.

## 2. G vs label

| metric | value | notes |
|---|---|---|
| Spearman(G, label), n=462 | **+0.152** | weak positive |
| AUC(G \| label ≥ 3) | **0.605** | weak |
| AUC(G \| label = 4) | 0.665 | n=2, noise |
| **within accepts**: Spearman | **+0.023** | flat |
| **within accepts**: AUC(label ≥ 3) | **0.511** | ≈ chance |

G range: accepts [1.40, 4.17], screen-rejects [−3.31, 1.39] — the cutoff sits ~1.40.

**G transfers as a threshold, then saturates.** Pooled AUC 0.605 comes almost entirely from the
accept/reject *boundary*; **above the cutoff G is dead** (within-accept AUC 0.511, Spearman 0.023).
So the screen's binary decision carries a little signal, but G's ordering among the crops it
accepts carries essentially none — you cannot use G to rank minibrot accepts.

**Is the weak signal just degree?** No — and this is the interesting part. Spearman(G, degree) =
−0.069 (G is not a degree proxy), and conditioning on degree *sharpens* G rather than killing it:

| degree | AUC(G \| label ≥ 3) | pos / neg |
|---|---|---|
| 2 | — | 0 / 118 (no L≥3 at deg 2) |
| 3 | 0.559 | 18 / 105 |
| 4 | **0.720** | 21 / 100 |
| 5 | 0.643 | 35 / 65 |

Within a fixed degree G recovers AUC ~0.56–0.72. Degree was a *suppressor*: pooling degrees
flattened G because the label signal from degree and from G are near-orthogonal. **So G is a real
but weak gate that saturates at its cutoff, and it is orthogonal to the strongest predictor.**

## 3. Class-4 count

Only **2**, and they are two windows of the **same atom** `d5_p11_029` (mb0095, mb0159) — degree 5,
period 11, band 10–12, both in the accept arm. Both are the same blue/orange double-spiral field
(see `minibrot_roster_v2_class4.png`). At this corpus size class-4 is essentially a point event;
no arm/degree/band breakdown is meaningful beyond "deg-5, accept."

## 4. Depth

**Degree is the label axis; period is a passenger.**

| | Spearman with label |
|---|---|
| degree | **+0.554** |
| period | +0.059 |
| log10\|A\| | +0.094 |
| period ↔ log10\|A\| (collinearity) | +0.522 |

Degree climbs monotonically: deg2 mean **1.22** (100/128 are 1s, zero L≥3) → deg3 1.87 → deg4 2.05 →
deg5 **2.27** (all the class-3/4 richness lives here). Period barely moves the label, and log10|A|
(collinear with period at 0.52) inherits period's non-effect.

**The designed diagnostic — within the period-matched eval slice (n=110), does period still predict
label?** Spearman(period, label) | eval = **−0.208**, i.e. *negative*. Once you hold the sampling
to the period-matched slice, period does not predict quality; if anything deeper is slightly worse.
The apparent "depth helps" is entirely degree riding along with the draw. **Depth per se is not the
knob — degree is.**

## 5. Within-atom variance — *the load-bearing one*

145 atoms (105 with ≥2 crops, 40 singletons).

- per-atom label range (max−min) over multi-crop atoms: mean **0.71**; 37 atoms constant, 61 span 1,
  7 span 2.
- mean within-atom variance **0.151**, between-atom variance **0.319**, grand variance 0.466.
- crude **ICC = 0.68** (atom explains ~2/3 of label variance).
- **25 of 105 multi-crop atoms straddle low (≤2) and high (≥3)** — e.g. `d3_p10_024: [1,3,3,3,3,3]`,
  `d3_p07_017: [1,2,2,2,3,3]`, `d4_p13_032: [1,2,2,2,2,3]`.

**Read:** the atom is the stronger unit (ICC 0.68) but **not** the whole story. A quarter of
multi-crop atoms produce both a good and a bad window, and even the two class-4s are two windows of
one atom that also yields lesser crops. **Quality is a property of the window as well as the atom** —
searching within a good minibrot's neighbourhood is a legitimate frame; the window-level task is not
degenerate. Practically: pick the atom by degree, then still hunt the window.

## 6. Reveal count

**0 revealed / 487 blind.** Every label was set blind; there is no revealed subset to contrast. The
blind-by-default labeling held.

## The sheets (Part C)

Three vivid-companion sheets in `docs/design/` (blue_orange, one map, read beside the rubric):

- **`minibrot_roster_v2_class4.png`** — the 2 class-4s (both windows of atom `d5_p11_029`).
- **`minibrot_roster_v2_hi_g_lo.png`** — top 24 by G of the 189 accepts I scored 1–2. **These
  share an obvious failure mode: a single sharp spiral or dendrite filament floating in a large
  flat blue field — high edge crispness, low compositional fill.** Skews to deg-2/deg-3. This is
  what high G rewards: local edge energy off a crisp isolated filament, which maxes out on lonely
  sparse dendrites, not on a dense composed wallpaper. *That is the thing to stop selecting for.*
- **`minibrot_roster_v2_sub_hi.png`** — the 27 sub-cutoff/OOD crops I scored 3. Visibly strong
  filigree and spiral fields the screen would have discarded — the direct evidence that harvesting
  outside the accepts pays.

---

## What this says (report only — no cutoff/draw changes made)

1. **G does not transfer as a ranker on minibrot fields.** It works as a coarse accept/reject gate
   (accepts +0.25, pooled AUC 0.60) but is flat above its own cutoff (within-accept AUC 0.51). The
   anchor pass's "8 accepts all scored 2" was not a fluke — the accept arm is a wall of 2s.
2. **The variable that predicts minibrot label is degree, not G and not depth.** Deg-5 holds all the
   richness; deg-2 is a floor of 1s. Period/log10|A| don't predict label once matched.
3. **Sub-cutoff crops are worth harvesting.** 26 screen-*rejects* (+1 OOD) were labeled 3 — visibly
   strong spiral/filigree fields the screen would have thrown away (`minibrot_roster_v2_sub_hi.png`).
   Hunting near minibrots *outside* the screen's accepts pays.
4. **Window and atom both matter.** ICC 0.68, but a quarter of atoms straddle good/bad — the search
   unit is the atom, the quality unit is still the window.

**Decision deferred to Matt with the sheets in hand:** what replaces G as the minibrot-field screen.
The evidence points at *degree as the primary draw axis + within-atom window search*, with G demoted
from ranker to (at most) a weak junk-gate. No recalibration, cutoff change, or draw change was made.
