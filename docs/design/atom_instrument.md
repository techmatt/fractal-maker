# The atom instrument `A` — size, orientation, required precision, f64-wall predictor

**What.** `deep_center_finder.atom_instrument(c, period, degree)` computes, from the
same recursion Newton already runs on a nucleus `(c₀, n)`, the **atom scaling
factor** `A`:

```
z_{k+1}  = z_k^d + c
z'_{k+1} = d·z_k^(d−1)·z'_k + 1          z'_0 = 0        (P_n'(c₀) = z'_n)
Lambda   = Π_{k=1..n−1} d·z_k^(d−1)                       (reduced multiplier; k=0 is z₀=0, dropped)
A        = Lambda^(1/(d−1)) · P_n'(c₀)
```

Locally the p-fold map conjugates to `w^d + C`; the embedded copy is the whole
multibrot pulled back by `δ = C/A`. So `A` is the atom's inverse linear scale and
orientation, and it yields three things the code did not have before, all at
essentially zero cost (one orbit pass, quantities Newton already forms):

1. **Default window scale `≈ 1/|A|`, rotation `≈ −arg A`** — a priori, before any
   render. The `(d−1)`-th root leaves `arg A` determined only **mod `2π/(d−1)`**:
   an orientation ambiguity (which of the `d−1` rotational copies), not an error.
   The instrument records the mpmath principal branch plus that ambiguity spacing.
2. **Required mpmath precision** — `⌈log₁₀|A|⌉ + guard` digits (floored at 50), the
   digits needed to localize a `~1/|A|` frame.
3. **An a-priori f64 pixel-spacing-wall predictor** — `f64_wall_margin_decades(width)`
   (below).

## Reconciliation with the empirical size law — they are the *same* quantity

`A` is **not** a second, independent estimate to cross-check `nucleus_size_estimate`
against. Measured over the existing d2 + d3/d4/d5 nuclei (12/degree, periods 3–15):

```
deg  n-range   |A|·|size_corrected|
 d2   3..15    1.00000000
 d3   3..15    1.00000000
 d4   3..10    1.00000000
 d5   3..15    1.00000000
```

`|A| ≡ 1/|size|` and `arg A ≡ −arg(size)` to full precision **at every n, n=3
included** — the identity is *exact, not asymptotic*. This is algebra, not luck: the
size code's `b`-sum times `Lambda` equals `P_n'(c₀)`, and its `Lambda^{d/(d−1)}`
denominator is `Lambda^{1/(d−1)}·Lambda`, so `A = Lambda^{1/(d−1)}·(b·Lambda) =
b·Lambda^{d/(d−1)} = 1/size`. Locked in `test_deep_center_finder_degree.py`
(`test_atom_A_equals_inverse_size`), computed via an independent orbit pass so the
identity is a genuine cross-check.

**On the size law's `d/(d−1)` correction.** Because `A` is derived from the
c-derivative and independently reproduces `1/size`, it *confirms* the corrected
exponent. Against the **old flat `λ²` law** (what the code used before the q4
multibrot transfer), `A` disagrees by exactly `|λ|^{(d−2)/(d−1)}`:

```
deg   old-λ² disagreement factor (over the same nuclei)
 d2   1.00×                       (d/(d−1)=2, no change)
 d3   4.1× … 499×                 (grows with |λ|, i.e. with depth)
 d4   6.6× … 390×
 d5   8.7× … 2497×
```

The "4–11×" the correction was described by is the factor at shallow/typical
periods; it grows without bound with depth. The fields support the **corrected**
law (interior-fraction ≈0.2–0.5 at `4·|size|`, matching d2 — see
`q4_multibrot_transfer.md`), i.e. they support `A`.

## Trustworthy period range

Two distinct claims, kept separate:

- **`A` vs the size *formula*:** exact at all `n` (identity above). There is no
  "small-n unreliability" between `A` and `size` — they are one quantity.
- **`A` vs the *true rendered atom extent*:** this is the asymptotic part. `A` is
  the linear term of a renormalization that also has an `O(δ²)` distortion; the
  embedded copy is a faithful minibrot only once it is well-separated from the
  parent's own structure. Empirically (transfer-study interior-fraction and the
  accepted fate-sheets) the `4·|A|⁻¹` framing is trustworthy from **n ≈ 4–5 upward**
  and tightest at the deep periods (p15); **n = 3** is borderline — the largest
  atoms, least self-similar, where the `O(δ²)` term is a non-trivial fraction of a
  `4/|A|` window. Use `A` as a hard scale/precision figure at all n, but treat the
  n≤3 *composition* suggestion as approximate.

## The f64 pixel-spacing-wall predictor

`render-one --dump-field-source f64` (and every multibrot field — multibrot has **no
perturbation backend**, so f64 is the *only* path) quantizes once pixel spacing drops
below `PERTURB_SPACING = 1e-13`. A default `k·|A|⁻¹` frame (k=4) at `width×ss` has
spacing `k/(|A|·width·ss)`, so the wall is crossed when

```
log₁₀|A|  >  log₁₀(k) − log₁₀(1e-13) − log₁₀(width·ss)
```

`f64_wall_margin_decades(width)` returns the signed headroom; **negative predicts an
f64 failure a priori, with no render attempt**. This is what the last d2 harvest
discovered *by failing* (8 of 116 render-failed): those 8 were ~2 minibrots whose
atoms sit deeper than the deepest survivor (`mb10_p21`, fw 7.26e-10 ≈ spacing 4e-13,
right at the wall), exactly the regime `log₁₀|A|` flags. The transfer study's
`SIZE_LO = 1e-10` band floor is this same wall expressed as a size cut; `A` makes it
an explicit, per-nucleus, pre-render figure (all 48 sourced nuclei carry margin ≥2.2
decades — the band already keeps them clear, and now we can *see* by how much).

## Where it is logged

`q4_multibrot_transfer.py` source records (`nuclei_d{2..5}.json`) and the screen-stage
`per_minibrot` blocks now carry `degree`, `period`/`n`, `log10_abs_A`, `arg_A`,
`required_dps`, `f64_wall_margin` as **covariates** so period-vs-quality can be read
later without re-running. Nothing keys off them yet.
