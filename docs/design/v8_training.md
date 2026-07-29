# v8 training population

Train-split location counts from `data/v8/manifest.jsonl`, one row per fractal partition and
one column per quality class (1 = bad … 4 = exceptional wallpaper emission). Counts are
LOCATIONS, not crops — each expands to 24 cached augmentation tiles under the v8b recipe
(`data/v8/aug_roster.json`), so the tile count is 24x every number below. Class totals are
cross-checked against `build_metadata.population.class_{train,eval}` at emission.

| partition        |     1 |     2 |     3 |     4 | total |
|:-----------------|------:|------:|------:|------:|------:|
| mandelbrot       |  3165 |   588 |   289 |    83 |  4125 |
| julia            |   541 |   297 |   172 |    77 |  1087 |
| phoenix          |   354 |   136 |    29 |    54 |   573 |
| multibrot3       |   167 |   120 |    25 |     9 |   321 |
| multibrot4       |   127 |   136 |    36 |     5 |   304 |
| multibrot5       |   151 |    96 |    37 |    13 |   297 |
| julia_multibrot3 |    61 |    35 |    13 |    13 |   122 |
| julia_multibrot4 |    45 |    17 |    17 |    17 |    96 |
| julia_multibrot5 |    23 |    16 |    17 |    16 |    72 |
| **total**        | **4634** | **1441** | **635** | **287** | **6997** |

**eval split** (144 locations, all `julia_multibrot{3,4,5}`) — class 1: 23, class 2: 56, class 3: 43, class 4: 22; total 144.
