# Exposure / HDR — what the engine can already reach, and what no operator can

**Nothing is adopted beyond one mode.** This doc records the measurements and the reachability
map so a future exposure decision starts from them instead of re-measuring. The two open TODOs
at the end are **PENDING, owner Matt** — this doc does not decide them.

The subject is the blown-highlight failure on the strange render modes: large near-white
plateaus that read as flat. It splits into **two failure classes with opposite verdicts**, and
almost every wrong move in this area comes from treating them as one.

## 1. What exists in the engine

`render_modes::Rolloff` — a **luminance-domain, chroma-preserving** tone compression applied to
the final linear RGB immediately before downsample/sRGB. Four operators: `none` (exact identity,
early-returned, so byte-identical to the pre-rolloff path), `reinhard` (extended, white point =
strength), `aces` (Narkowicz, strength = pre-exposure), `soft_knee` (identity below knee
`k = strength`, `tanh` shoulder above, C¹ at the knee).

Luminance-domain is the load-bearing choice: it maps `L → L'` and rescales all three channels by
`L'/L`, preserving hue and chroma. A per-channel curve pulls the brightest channel down fastest
and desaturates the highlight *toward white* — the opposite of the goal.

**Adopted on exactly one mode**: `direct_trap_screen` at `soft_knee@0.35`, via
`tools/mining/mining_roster.ROLLOFF` (`rolloff_for` / `rolloff_token`; every other mode renders
`("none", 1.0)`, hence byte-identical). **And it works** — max severity **0.0004 over 209 rows**,
the lowest of all 16 modes (the roster's 15 plus the off-roster `smooth` baseline).

## 2. The reachability map — sharper than "gated to the screen operators"

`apply_rolloff` has exactly **two call sites**, each with its own gate
`[code: src/render_modes.rs — `mode == MergeMode::Screen` on the direct-trap accumulator;
`combine == Combine::Screen` on the composite path]`. Everything else never reaches the
function at all. Over the 15 roster modes:

| reachable by a rolloff | not reachable by ANY tone operator |
|---|---|
| `direct_trap_{screen, lines, ring}` (`merge_mode: screen`) | the 6 **pure-field** modes — `tia`, `stripe`, `exp_smoothing`, `gaussian_int`, `trap_circle`, `curv_linear`: the field render path has no rolloff call site |
| all 5 composites (`combine: screen`) — `smooth_{mean_angle, angle_min}`, `composite_c{7,13,17}_*` | **`direct_trap_multiply`** (`merge_mode: multiply`) |

So **8 of 15 modes are reachable and only 1 has it turned on**. "Reachable" and "adopted" are
different questions and the roster is where the second one is answered — turning a rolloff on for
the other seven is a config edit, not new code.

### `direct_trap_multiply` cannot be fixed by any tone operator, and is the worst mode measured

Mean severity **0.0907**, mean solid-clip **0.0336**, over **313 rows** — the worst in the
corpus. Rendering it with `rolloff soft_knee@0.35` is **bit-identical to base** (the merge-mode
gate). And even if the gate were lifted it would not help: the spec is
`{"merge_mode": "multiply", "start_color": "white"}`, so **its white is UNHIT BACKGROUND**, not a
saturated accumulator. There is nothing to compress. The only levers are opacity/threshold or a
different start colour.

## 3. Two failure classes, opposite verdicts

**Class A — solid white discs (achromatic clip).** `direct_trap_multiply` / `_lines`; solid-clip
up to 0.15. **43–60% of clipped pixels are exactly (255,255,255)** with std ≤ 1.5/255 inside the
mask. There is **no information in the file**: the largest in-mask chroma gain over all 8
exemplars × 13 PNG-side variants was **+0.0000**. Every PNG-side operator turns a white disc into
a grey disc. ⇒ **clipped-detail recovery FORCES the field side.**

**Class B — crushed-bright midtones.** Top-bin mass to 0.69 with near-zero true clip: the
composites, `curv_linear`, `smooth_angle_min`. Here a PNG-side CLAHE-style **global
redistribution does work**, and this is where the shipped release images sit.

Field-side wins come from **redistributing the whole field distribution** (`histeq`, robust-z),
**not from compressing its top** — these palettes are already near-white at mid-field values.

`[measured 2026-08-10, ~2,520 rendered render-mode images. The study lived in `scratch/` and is
gone; NO re-derivation command survives, so these numbers cannot be refreshed without rebuilding
the harness. Population caveat: the three render-mode batches that existed on that date hold
2,460 rows, so the 2,520 spans something wider — treat the per-mode row counts (209, 313) as the
reliable populations and the total as approximate.]`

## 4. `transform=histeq` — existing, parsed, and used by nothing

`Transform::Histeq` parses, round-trips through the params serializer, and is implemented as
`FieldNorm::Histeq` (a rank-fraction against the sorted valid field) — **and all 15 committed
specs are `"transform": "linear"`.** It was the single most effective field-side lever measured:
in-mask chroma **0.011 → 0.836** on a `composite_c7_smooth_trap_circle` trap-circle exemplar. It
needs **no new code**, and it composes with the existing `soft_knee` (they act at different
stages — field normalization vs final-colour compression).

## 5. Any field-side arm SPLITS — `--dump-field` refuses the direct family

`render-one --dump-field` errors on `direct_trap` with *"direct_trap is a colour-valued
composite, not a scalar"* `[code: src/render_one.rs]`. There is therefore **no pre-palette
scalar to renormalize** for the direct modes, and a field-side exposure arm necessarily splits
in two:

- **pure-field and composite modes** → scalar renormalization (`histeq` / robust-z) on the field;
- **direct / composite-colour modes** → render-side knobs only (rolloff, opacity/threshold,
  start colour).

Any proposal that reads as one uniform "renormalize the field" arm has not survived this.

## 6. ★ NEVER gate on clip share. Judge by IN-MASK CHROMA.

**Every compressive operator drives clip share to 0.000 by construction.** `soft_knee` is a
strict contraction with `L' < 1` for all finite `L`, so nothing it emits can clip — the statistic
is a property of the operator, not of the picture.

The measured instance: the in-engine rolloff takes `direct_trap_lines` from clip share **0.185 →
0.000 while the lozenge stays visibly white and flat.** At `k = 0.35` the operator's own ceiling
is `L' = 0.35 + 0.65·tanh(1) = 0.845`, i.e. **237/255** in sRGB — under any 8-bit clip threshold
by construction, at every strength, forever. The statistic passes; the picture does not.

**Use in-mask chroma** (mean chroma inside the blown mask), which is what the histeq exemplar
above is quoted in and what the class-A null result is quoted in. It can be zero when the file
holds no information, which is exactly the discrimination clip share cannot make.

*(The handoff record put the soft-knee ceiling at 239/255; 237 is what the shipped operator
computes at `k=0.35`. Same conclusion either way.)*

## 7. The two pending TODOs — PENDING, owner Matt

1. **Per-mode spec edits** — `histeq` where the field is scalar, widening the rolloff's *adopted*
   set beyond `direct_trap_screen`, and a different `start_color` for `direct_trap_multiply`.
   **Gated on a ~20-example verification judged by IN-MASK CHROMA, never clip share** (§6).
2. **A general continuous "auto-adjust" tone mapping applied ON THE PALETTE** — LUT-side, so it
   is resolution-independent and costs nothing per pixel.

One decision already depends on this and is explicitly waiting: two promoted strange modes
(`composite_c7_smooth_trap_circle`, `smooth_angle_min`) cleared **zero** at the emission gate
from 72 attempts. **Both stay ACTIVE — they are class B, not bad modes.** Fix the tone handling,
re-read the yield, *then* decide the demotion.

## 8. Related, separately owned

`direct_trap_screen`'s **saturation cap** (`DTS_SCREEN_*_CAP`, opacity ≤ 0.15 ∧ threshold ≤ 0.08,
enforced source-side and gated to cross+screen) is a different mechanism aimed at the same
failure from the other end: it stops the accumulator reaching unrecoverable `(1,1,1)` in the
first place, where the rolloff rescues the partially-blown tier. It is owned by the mode's own
spec `note` in `specs/direct_trap_screen.json`, with its measured blowout sweep. Palette-side
questions (which maps crush, the proposal-vs-release colour narrowing) belong to
`render_coloring_surface.md`, not here.
