# `prio_terms.jsonl` — park note (unprocessed, retained on purpose)

Short pointer so a future reader doesn't rediscover this file cold or reprocess it
blindly. **Not analysed here** — this only records what it is, where it lives, and why
it is kept.

## What it is

One row per **pushed candidate** — every node the walker expanded and scored into the
priority queue, **including the never-admitted majority** (most rows were never even
rendered, let alone admitted). Each row carries the priority-function inputs at push
time:

| field | meaning |
|---|---|
| `batch`, `node_id`, `root_id`, `partition`, `depth` | identity / provenance of the pushed node |
| `eord` | the **cheap** energy-order score (the `cheap_eord` that also appears in the harvest logs) |
| `gumbel` | best-first tie-break noise |
| `dup_pen`, `cos_max`, `nov_pen` | novelty / near-dup penalties |
| `depth_bonus` | depth-seeking term |
| `priority` | the composite the frontier actually sorted on |

## Where it lives (LFS-tracked, committed)

```
data/discovery/campaign1/breadth/prio_terms.jsonl
data/discovery/campaign2/breadth/prio_terms.jsonl
data/discovery/shakeout_legacy/prio_terms.jsonl
data/discovery/shakeout_recency/prio_terms.jsonl
data/discovery/steered_run2/prio_terms.jsonl
```

Committed via LFS (`.gitattributes`: `data/discovery/**/prio_terms.jsonl`), with a
matching `!`-exception in `.gitignore` — same treatment as `harvest_log.jsonl`.

## What it's for

The **steering-side counterpart to the harvest logs.** The harvest log records the
cheap→canonical→fate join only for candidates that cleared the `τ_h` cheap gate and got
a confirmation render — i.e. the *rendered* slice. `prio_terms` records the
priority-scoring inputs for the **entire pushed population**, the vast majority of which
was scored and dropped before ever being rendered. It is what lets the *steering /
priority* function (cheap `eord`, novelty and dedup penalties, depth bonus) be studied
against what the walker actually chose to expand.

It is retained because it **cannot be regenerated from a ledger of admissions**:
admissions are the tiny surviving tail; the pushed candidates and their cheap `eord` are
gone otherwise. Rescued and committed deliberately unprocessed — pick it up when there
is a steering question that needs it.
