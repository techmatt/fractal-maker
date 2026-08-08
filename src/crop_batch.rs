//! `crop-batch` — the **extended-field crop executor** (v11 prep, plumbing only).
//!
//! ## What changes versus `v4-render-batch`
//!
//! The v8b..v10 augmentation recipe is `4 palettes × 3 geometries × 2 AA levels = 24`
//! tiles per location, and [`crate::v4_cache`] executes every one of those 24 as an
//! independent iterate→shade. The plan's own docs already note the waste: 4 palettes
//! share one escape-time field, and the executor does not exploit it. The geometry axis
//! is *also* nearly free, because every jittered crop is a sub-window of a slightly
//! larger view — and so is the AA axis, because both AA levels sample the same plane.
//!
//! This module renders **one iteration pass per location** over an **extended field**
//! (`--extend`, default 1.2× the canonical frame, i.e. +10% per side) and derives every
//! tile from it as *crop + resample + colormap*. At the default 512×288 / `field_ss 2`
//! that is 1230×692 = 851k subpixels per location against v8b's 8.85M (12 ss1 + 12 ss2
//! renders) — a **10.4× cut in sampled subpixels**, or 2.6× even against a hypothetical
//! executor that perfectly shared v8b's 6 distinct fields.
//!
//! ## Why one scalar field is sufficient
//!
//! Every family's cache tile is a pure per-subpixel function of the **smooth escape-time
//! scalar** (interior = `NaN`), through one of exactly two colour maps:
//!
//!  * **location profile** (Mandelbrot, quadratic Julia) — [`crate::generate::color_params`]
//!    is `channel = Smooth`, `interior = Black`, `de_shade = None`, `offset = 0`, so
//!    [`crate::coloring::shade`] reduces to `palette.lookup_linear((ν·density).rem_euclid(1))`.
//!    No trap channel, no DE, nothing frame-global.
//!  * **beautiful smooth** (Multibrot, Julia-multibrot `d ≥ 3`, Phoenix) — `v4-render-batch`
//!    routes these to `render_beautiful(beautiful(Smooth))`, which for exactly these
//!    families takes the fast path `render_smooth_f64_fast`: the same escape-time scalar,
//!    then a **frame-global** percentile stretch + transform + palette cycles
//!    ([`SmoothFieldColorer`], shared with that render so the chains cannot drift).
//!
//! The one genuinely global operation is the beautiful path's percentile stretch, so it is
//! rebuilt **per crop** over that crop's own subpixels — the population a whole-frame render
//! of that crop would have normalized over, not the extended field's.
//!
//! ## The AA axis, honestly
//!
//! The legacy `antialiased` tile is `ss2 + lanczos3`; the derived one is a lanczos3
//! minification at ratio `scale·field_ss` (1.8..2.2 at the default draw) from a **fractional
//! offset**. That is the same kernel and the same linear-light filtering, at a ratio the
//! random scale makes non-integer — a legitimate difference, not a defect.
//!
//! The legacy `aliased` tile is `ss1 + box`, i.e. a **point sample at each pixel centre**.
//! An ss2 field does not contain those points (its sub-cell centres sit at 0.25/0.75 of a
//! pixel, never 0.5), and no even `field_ss` ever does; an odd `field_ss` would, but only
//! for the identity crop, since a random shift breaks the alignment anyway. Running the box
//! kernel instead would average ≈`ratio` subpixels and produce a *second antialiased tile*,
//! destroying the axis. So the derived aliased tile is **nearest-neighbour** on the field:
//! a true point sample, displaced by at most half a field subpixel. That is the closest
//! honest scheme, and it is why the AA spec below names a *mode*, not a supersample factor.
//!
//! ## Two fan-out modes
//!
//! **Product** (the default, v8b's shape): `--geoms × --aa × palettes`, where the palettes
//! are `--palettes` plus `--draw-palettes` sampled once per LOCATION, and every (palette,
//! AA) cell shares the same geometries.
//!
//! **Independent** (`--tiles N`, the v11 recipe): N tiles per location, each one draw from
//! the joint distribution — its own palette (uniform over `--palette-pool`, with
//! replacement), geometry, AA level and JPEG quality, from its own seed slot. The axes are
//! then decorrelated across the corpus instead of crossed within a location. `--floor-
//! palette <name>:<count>` and `--floor-identity <n>` reserve the low slots so a guaranteed
//! minimum (the deploy-matched map, the labeler's map, the exact deploy composition) is
//! present on every location; the reservation comes out OF the N, never on top of it.
//!
//! The two cost the same: every tile is a full resample+colourize in both, so the choice is
//! about what the network sees, not about render budget. Independent mode does pay one more
//! percentile stretch per tile on the beautiful families (each tile owns its crop window,
//! so the normalization cache no longer has `--geoms` hits per entry).
//!
//! ## Determinism and replay
//!
//! Every draw (crop geometry, drawn palettes, JPG quality) comes from a `SplitMix64` seeded
//! by `(seed_tag, loc_id, slot index)` — no global RNG, no ordering dependence, so a tile is
//! a pure function of its location row and this module. The emitted manifest records each
//! tile's **realized** geometry in field-subpixel units (`src_x0`, `src_y0`, `ratio`), so
//! `--replay` regenerates a tile from its manifest row **without re-drawing anything**.
//!
//! Fields are **stream-and-discard**: rendered into a local `Vec<f32>`, cropped, dropped.
//! Nothing here writes into any field cache, so the frame-extension axis never has to enter
//! a cache key.
//!
//! Default output is `scratch/crop_batch/` (the generated-output convention); a real v11
//! build passes `--out-root`/`--manifest` explicitly.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use clap::Args;
use image::RgbImage;
use num_complex::Complex;
use rayon::prelude::*;

use crate::palette::{linear_to_srgb, Palette};
use crate::palette_pick::parse_colormaps;
use crate::probe::SplitMix64;
use crate::render::{self, DownsampleFilter, Frame};
use crate::render_modes::{self, ColoringParams, Family, Field, SmoothFieldColorer};
use crate::{ensure_parent_dir, hp, jsonl};

/// Escape radius for the **location-profile** families — identical to
/// render-one/present/enrich/v4-render-batch.
const BAILOUT: f64 = 1e6;

/// The location-profile smooth density ([`crate::generate::color_params`]`.density`).
/// Restated here only as the crop colourer's input; the value's owner is `generate`.
const PROFILE_DENSITY: f64 = 0.004;

// --------------------------------------------------------------------------- //
// Deterministic draw
// --------------------------------------------------------------------------- //

/// FNV-1a 64 over the seed tag — a stable, platform-independent namespace for the
/// per-location seeds (`DefaultHasher` is explicitly not stable across releases).
fn tag_hash(tag: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in tag.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Seed for one draw slot. `slot` namespaces the independent draws of a location
/// (crop geometry `g`, the palette draw, the per-tile quality draw), so adding a draw
/// never reshuffles the others.
fn slot_seed(tag: u64, loc_id: u64, slot: u64) -> u64 {
    let mut h = tag;
    for v in [loc_id, slot] {
        h ^= v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
        h ^= h >> 29;
    }
    h
}

/// Draw-slot namespaces. Disjoint by construction.
const SLOT_GEOM: u64 = 1_000;
const SLOT_PALETTE: u64 = 2_000;
const SLOT_QUALITY: u64 = 3_000;
/// Independent mode only: the per-tile AA coin and the per-tile palette draw. Separate
/// namespaces from `SLOT_PALETTE` (the product mode's per-LOCATION sample) so the two
/// modes cannot alias, and from each other so adding an axis never reshuffles the rest.
const SLOT_AA: u64 = 4_000;
const SLOT_TILE_PALETTE: u64 = 5_000;

// --------------------------------------------------------------------------- //
// Tile axes
// --------------------------------------------------------------------------- //

/// How a tile is reconstructed from the extended field.
///
/// `Point` is a nearest-neighbour point sample — the honest stand-in for the legacy
/// `ss1 + box` tile (see the module docs). `Filtered` is a real separable reconstruction
/// kernel at the crop's (non-integer) minification ratio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AaMode {
    Point,
    Filtered(DownsampleFilter),
}

impl AaMode {
    fn parse(s: &str) -> Result<AaMode, String> {
        match s {
            "point" => Ok(AaMode::Point),
            "box" => Ok(AaMode::Filtered(DownsampleFilter::Box)),
            "mitchell" => Ok(AaMode::Filtered(DownsampleFilter::Mitchell)),
            "lanczos3" => Ok(AaMode::Filtered(DownsampleFilter::Lanczos3)),
            other => Err(format!("unknown AA mode '{other}' (point|box|mitchell|lanczos3)")),
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            AaMode::Point => "point",
            AaMode::Filtered(DownsampleFilter::Box) => "box",
            AaMode::Filtered(DownsampleFilter::Mitchell) => "mitchell",
            AaMode::Filtered(DownsampleFilter::Lanczos3) => "lanczos3",
        }
    }
}

/// One AA level: the **label** carried into the cache manifest (`aliased` /
/// `antialiased`, the two-value vocabulary `classifier/data_v4.py` keys on) and the
/// reconstruction **mode** that realizes it. They are separate because with one field
/// there is no per-tile supersample factor left to name — see the module docs.
#[derive(Clone, Debug)]
struct AaLevel {
    label: String,
    mode: AaMode,
}

impl AaLevel {
    fn parse(s: &str) -> Result<AaLevel, String> {
        let (label, mode) = s.split_once(':').ok_or_else(|| {
            format!("--aa entry '{s}' must be <label>:<mode>, e.g. 'antialiased:lanczos3'")
        })?;
        if label.is_empty() {
            return Err(format!("--aa entry '{s}' has an empty label"));
        }
        Ok(AaLevel { label: label.to_string(), mode: AaMode::parse(mode)? })
    }
}

/// One crop geometry: a scale and a (magnitude, direction) shift of the canonical frame.
/// Geometry 0 is always the **identity** — dead centre, scale exactly 1.0 — so the deploy
/// composition is in front of the network in every (palette, AA) cell, exactly as v8b
/// requires. Shift magnitude is a fraction of the **canonical** frame width (not the
/// scaled slot width), which is what makes `extend ≥ scale_hi + 2·shift_max` the exact
/// containment condition.
#[derive(Clone, Copy, Debug)]
struct Geom {
    scale: f64,
    shift_frac: f64,
    angle: f64,
}

impl Geom {
    fn identity() -> Geom {
        Geom { scale: 1.0, shift_frac: 0.0, angle: 0.0 }
    }
    fn draw(seed: u64, lo: f64, hi: f64, shift_max: f64) -> Geom {
        let mut rng = SplitMix64(seed);
        Geom {
            scale: lo + rng.unit() * (hi - lo),
            shift_frac: rng.unit() * shift_max,
            angle: rng.unit() * std::f64::consts::TAU,
        }
    }
    /// Shift in canonical-frame-width fractions, `(dx, dy)`.
    fn offset(self) -> (f64, f64) {
        (self.shift_frac * self.angle.cos(), self.shift_frac * self.angle.sin())
    }
}

// --------------------------------------------------------------------------- //
// Field geometry
// --------------------------------------------------------------------------- //

/// The extended field's geometry for one location, plus the crop→field coordinate map.
///
/// Two properties are load-bearing.
///
/// **The margin is an equal PLANE distance on all four sides**, `(extend−1)/2` of the
/// canonical frame *width*, not of each axis's own extent. The crop shift is a single
/// magnitude in frame-width units with a uniform direction (v8b's convention, and the
/// prompt's "≤ 5% of canonical fw"), so a vertical displacement of `0.05·fw` is
/// `0.05·(W/H)·fh` = 8.9% of the frame height at 16:9. Padding each axis by a fraction of
/// *its own* extent under-pads the vertical by 22 subpixels at the default draw and lets
/// the tallest shifted crop run off the bottom of the field. Equal plane margin is also
/// the geometrically honest reading of "+10% per side" for a rotationally-uniform shift.
/// The consequence is a larger *relative* vertical extension (1.358 vs 1.201 at the
/// defaults), reported as `extend_y` rather than hidden.
///
/// **The pad is a whole number of subpixels**, so the extended sub-grid *contains* the
/// canonical sub-grid at an integer offset. That is not cosmetic: it makes the identity
/// crop land on the same sample positions the legacy tile used, up to the f64 rounding of
/// `fw_ext` — which is why the parity read on the identity framing is tight rather than a
/// resampling smear.
#[derive(Clone, Copy, Debug)]
struct FieldGeom {
    /// Canonical tile size in output pixels.
    out_w: u32,
    out_h: u32,
    /// Subpixels per canonical output pixel.
    field_ss: u32,
    /// Extended field dims in subpixels.
    sub_w: u32,
    sub_h: u32,
    /// Whole-subpixel pad per side — the SAME on all four (equal plane distance).
    pad_x: u32,
    pad_y: u32,
    /// The extended frame width actually rendered.
    fw_ext: f64,
}

impl FieldGeom {
    fn build(out_w: u32, out_h: u32, field_ss: u32, extend: f64, fw: f64) -> FieldGeom {
        let base_w = out_w * field_ss;
        let base_h = out_h * field_ss;
        // Rounded UP, so the realized margin is never below the requested one.
        let pad = (((extend - 1.0) / 2.0) * base_w as f64).ceil().max(0.0) as u32;
        let sub_w = base_w + 2 * pad;
        let sub_h = base_h + 2 * pad;
        FieldGeom {
            out_w,
            out_h,
            field_ss,
            sub_w,
            sub_h,
            pad_x: pad,
            pad_y: pad,
            fw_ext: fw * (sub_w as f64) / (base_w as f64),
        }
    }

    /// Realized extension factor on each axis (≥ the requested `--extend`).
    fn realized_extend(&self) -> (f64, f64) {
        (
            self.sub_w as f64 / (self.out_w * self.field_ss) as f64,
            self.sub_h as f64 / (self.out_h * self.field_ss) as f64,
        )
    }

    /// The frame handed to the iterate stage: the extended view at **one sample per
    /// subpixel** (`ss = 1`), so the field array is exactly `sub_w × sub_h`.
    fn frame(&self, center: Complex<f64>) -> Frame {
        Frame { center, frame_width: self.fw_ext, out_width: self.sub_w, out_height: self.sub_h }
    }

    /// A crop's window in **field-subpixel units**: `(src_x0, src_y0, ratio)`, where output
    /// pixel `d` is centred at `src_x0 + (d + 0.5)·ratio`. Identity maps to
    /// `(pad_x, pad_y, field_ss)` exactly.
    fn window(&self, g: Geom) -> (f64, f64, f64) {
        let ssf = self.field_ss as f64;
        let (dx, dy) = g.offset();
        let base_w = (self.out_w * self.field_ss) as f64;
        let base_h = (self.out_h * self.field_ss) as f64;
        // dx/dy are fractions of the canonical frame WIDTH, so both convert with base_w.
        let src_x0 = self.pad_x as f64 + dx * base_w + 0.5 * base_w * (1.0 - g.scale);
        let src_y0 = self.pad_y as f64 - dy * base_w + 0.5 * base_h * (1.0 - g.scale);
        (src_x0, src_y0, g.scale * ssf)
    }
}

// --------------------------------------------------------------------------- //
// Location rows
// --------------------------------------------------------------------------- //

/// One location parsed from the input JSONL.
struct Loc {
    loc_id: u64,
    cx: String,
    cy: String,
    fw: f64,
    kind: String,
    c: Option<(String, String)>,
    p: Option<(String, String)>,
    z1: Option<(String, String)>,
    /// The **canonical** frame's iteration cap (see `--maxiter`).
    maxiter: u32,
    /// The cap policy token the caller declares for this row, stamped through verbatim.
    maxiter_policy: String,
}

fn pair(line: &str, a: &str, b: &str) -> Option<(String, String)> {
    match (jsonl::field_str(line, a), jsonl::field_str(line, b)) {
        (Some(re), Some(im)) => Some((re, im)),
        _ => None,
    }
}

fn parse_loc(line: &str, idx: usize, args: &CropBatchArgs) -> Result<Loc, String> {
    let cx = jsonl::field_str(line, "cx").ok_or("missing cx")?;
    let cy = jsonl::field_str(line, "cy").ok_or("missing cy")?;
    let fw = jsonl::field_str(line, "fw")
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| jsonl::field_f64(line, "fw"))
        .ok_or("missing fw")?;
    let kind = jsonl::field_str(line, "fractal_type").unwrap_or_else(|| "mandelbrot".into());
    let loc_id = jsonl::field_usize(line, "loc_id").unwrap_or(idx) as u64;
    let maxiter = match jsonl::field_usize(line, "maxiter") {
        Some(0) => return Err("maxiter must be > 0".into()),
        Some(n) => n as u32,
        None => args.maxiter.ok_or(
            "row carries no `maxiter` and --maxiter was not passed. The extended field must \
             iterate at the CANONICAL frame's auto_maxiter (never re-derived from the extended \
             fw); supply it per row or pass the fallback explicitly",
        )?,
    };
    let maxiter_policy = jsonl::field_str(line, "maxiter_policy")
        .unwrap_or_else(|| args.maxiter_policy.clone());
    Ok(Loc {
        loc_id,
        cx,
        cy,
        fw,
        kind,
        c: pair(line, "c_re", "c_im"),
        p: pair(line, "p_re", "p_im"),
        z1: pair(line, "zm1_re", "zm1_im"),
        maxiter,
        maxiter_policy,
    })
}

impl Loc {
    /// `(center, family)` at the precision the canonical geometry justifies.
    fn resolve(&self, width: u32) -> Result<(Complex<f64>, Family), String> {
        let prec_bits = hp::prec_bits(width, self.fw);
        let f = |s: &str| -> Result<f64, String> { Ok(hp::to_f64(&hp::parse_decimal(s, prec_bits)?)) };
        let pf = |v: &Option<(String, String)>| -> Result<Option<Complex<f64>>, String> {
            match v {
                Some((re, im)) => Ok(Some(Complex::new(f(re)?, f(im)?))),
                None => Ok(None),
            }
        };
        let center = Complex::new(f(&self.cx)?, f(&self.cy)?);
        let family = Family::from_type_token(&self.kind, pf(&self.c)?, pf(&self.p)?, pf(&self.z1)?)?;
        Ok((center, family))
    }
}

// --------------------------------------------------------------------------- //
// Colouring
// --------------------------------------------------------------------------- //

/// A crop's colour map: raw smooth scalar → linear RGB, under one palette.
enum Colorizer<'a> {
    /// Location profile — `palette.lookup_linear((ν · density).rem_euclid(1))`, interior
    /// black. Byte-equal to [`crate::coloring::shade`] under
    /// [`crate::generate::color_params`] (channel `Smooth`, interior `Black`, offset 0,
    /// no DE shade, no trap).
    Profile { palette: &'a Palette, density: f64 },
    /// Beautiful smooth — the shared [`SmoothFieldColorer`], normalized over the crop.
    Beautiful { palette: &'a Palette, colorer: &'a SmoothFieldColorer },
}

impl Colorizer<'_> {
    #[inline]
    fn linear(&self, v: f32) -> [f64; 3] {
        match self {
            Colorizer::Profile { palette, density } => {
                if !v.is_finite() {
                    [0.0, 0.0, 0.0]
                } else {
                    palette.lookup_linear((v as f64 * density).rem_euclid(1.0))
                }
            }
            Colorizer::Beautiful { palette, colorer } => colorer.linear(v, palette),
        }
    }
}

#[inline]
fn encode(lin: [f64; 3], out: &mut Vec<u8>) {
    for c in lin {
        let v = linear_to_srgb(c.clamp(0.0, 1.0));
        out.push((v * 255.0 + 0.5) as u8);
    }
}

/// Resample one crop of `field` to `out_w × out_h` under `col`.
///
/// `Filtered` runs the same two separable linear-light passes as
/// [`render::shade_and_downsample_filtered`], on taps built by
/// [`render::build_taps_scaled`] at the crop's fractional origin and non-integer ratio.
/// `Point` takes the nearest field subpixel per output pixel.
fn resample_crop(
    field: &[f32],
    fg: &FieldGeom,
    src_x0: f64,
    src_y0: f64,
    ratio: f64,
    mode: AaMode,
    col: &Colorizer,
) -> RgbImage {
    let (ow, oh) = (fg.out_w as usize, fg.out_h as usize);
    let (sw, sh) = (fg.sub_w as usize, fg.sub_h as usize);

    let filter = match mode {
        AaMode::Point => {
            let mut pixels = Vec::with_capacity(ow * oh * 3);
            let rows: Vec<Vec<u8>> = (0..oh)
                .into_par_iter()
                .map(|y| {
                    let sy = (src_y0 + (y as f64 + 0.5) * ratio)
                        .floor()
                        .clamp(0.0, (sh - 1) as f64) as usize;
                    let mut out = Vec::with_capacity(ow * 3);
                    for x in 0..ow {
                        let sx = (src_x0 + (x as f64 + 0.5) * ratio)
                            .floor()
                            .clamp(0.0, (sw - 1) as f64) as usize;
                        encode(col.linear(field[sy * sw + sx]), &mut out);
                    }
                    out
                })
                .collect();
            for r in rows {
                pixels.extend_from_slice(&r);
            }
            return RgbImage::from_raw(fg.out_w, fg.out_h, pixels).expect("dims match");
        }
        AaMode::Filtered(f) => f,
    };

    let htaps = render::build_taps_scaled(ow, src_x0, ratio, sw, filter);
    let vtaps = render::build_taps_scaled(oh, src_y0, ratio, sh, filter);

    // Only the source rows the vertical kernel reaches are shaded.
    let r0 = vtaps.iter().map(|t| t.start).min().unwrap_or(0);
    let r1 = vtaps.iter().map(|t| t.start + t.w.len()).max().unwrap_or(0).min(sh);

    // Horizontal pass: sub_w → out_w, shading each needed source row on the fly.
    let inter: Vec<Vec<[f32; 3]>> = (r0..r1)
        .into_par_iter()
        .map(|r| {
            let base = r * sw;
            let mut row = vec![[0f32; 3]; ow];
            for (x, tap) in htaps.iter().enumerate() {
                let mut acc = [0.0f64; 3];
                for (k, &w) in tap.w.iter().enumerate() {
                    let p = col.linear(field[base + tap.start + k]);
                    acc[0] += w * p[0];
                    acc[1] += w * p[1];
                    acc[2] += w * p[2];
                }
                row[x] = [acc[0] as f32, acc[1] as f32, acc[2] as f32];
            }
            row
        })
        .collect();

    // Vertical pass: clamp, sRGB-encode.
    let out_rows: Vec<Vec<u8>> = (0..oh)
        .into_par_iter()
        .map(|y| {
            let tap = &vtaps[y];
            let mut out = Vec::with_capacity(ow * 3);
            for x in 0..ow {
                let mut acc = [0.0f64; 3];
                for (k, &w) in tap.w.iter().enumerate() {
                    let src = tap.start + k;
                    if src < r0 || src >= r1 {
                        continue;
                    }
                    let p = inter[src - r0][x];
                    acc[0] += w * p[0] as f64;
                    acc[1] += w * p[1] as f64;
                    acc[2] += w * p[2] as f64;
                }
                encode(acc, &mut out);
            }
            out
        })
        .collect();

    let mut pixels = Vec::with_capacity(ow * oh * 3);
    for r in out_rows {
        pixels.extend_from_slice(&r);
    }
    RgbImage::from_raw(fg.out_w, fg.out_h, pixels).expect("dims match")
}

/// The crop's valid (finite) field values — the population the beautiful path's global
/// percentile stretch normalizes over. Deliberately the **crop's** subpixels, not the
/// extended field's: a whole-frame render of this crop would have seen exactly these.
fn crop_valids(field: &[f32], fg: &FieldGeom, src_x0: f64, src_y0: f64, ratio: f64) -> Vec<f64> {
    let (sw, sh) = (fg.sub_w as usize, fg.sub_h as usize);
    let x_lo = src_x0.floor().max(0.0) as usize;
    let x_hi = ((src_x0 + ratio * fg.out_w as f64).ceil() as usize).min(sw);
    let y_lo = src_y0.floor().max(0.0) as usize;
    let y_hi = ((src_y0 + ratio * fg.out_h as f64).ceil() as usize).min(sh);
    let mut out = Vec::with_capacity((x_hi.saturating_sub(x_lo)) * (y_hi.saturating_sub(y_lo)));
    for y in y_lo..y_hi {
        for &v in &field[y * sw + x_lo..y * sw + x_hi] {
            if v.is_finite() {
                out.push(v as f64);
            }
        }
    }
    out
}

// --------------------------------------------------------------------------- //
// JSON helpers (hand-rolled, as everywhere else in this crate)
// --------------------------------------------------------------------------- //

/// Round-trippable JSON number. Rust's float `Display` never uses exponent notation, so
/// a deep-zoom `fw` would serialize as a 300-digit literal; `LowerExp` is the same
/// shortest-round-trip algorithm with an exponent, so tiny/huge magnitudes take that form.
fn num(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let a = x.abs();
    if (1e-4..1e15).contains(&a) {
        format!("{x}")
    } else {
        format!("{x:e}")
    }
}

fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// --------------------------------------------------------------------------- //
// Tile specs
// --------------------------------------------------------------------------- //

/// One tile's fully-realized recipe — everything `--replay` needs, and everything the
/// manifest row stamps. Nothing here is re-derived at render time.
struct TileSpec {
    tile: usize,
    geom_idx: usize,
    geom: Geom,
    aa_label: String,
    aa_mode: AaMode,
    palette: String,
    jpg_quality: u8,
    src_x0: f64,
    src_y0: f64,
    ratio: f64,
    out: String,
}

/// This location's palettes: the always-on list, plus `draw` names sampled without
/// replacement from the pool by the location's own seed.
fn location_palettes(args: &CropBatchArgs, tag: u64, loc_id: u64) -> Vec<String> {
    let mut out = args.palettes.clone();
    if args.draw_palettes > 0 && !args.palette_pool.is_empty() {
        let mut pool: Vec<String> = args.palette_pool.clone();
        let mut rng = SplitMix64(slot_seed(tag, loc_id, SLOT_PALETTE));
        let k = args.draw_palettes.min(pool.len());
        for _ in 0..k {
            let i = rng.below(pool.len());
            out.push(pool.swap_remove(i));
        }
    }
    out
}

/// Every tile of one location, in `(geometry, AA)` outer / palette inner order — the order
/// v8b emits, kept so a v11 plan diffs against a v10 plan slot-for-slot.
fn plan_tiles(args: &CropBatchArgs, aa: &[AaLevel], fg: &FieldGeom, loc: &Loc, tag: u64) -> Vec<TileSpec> {
    let palettes = location_palettes(args, tag, loc.loc_id);
    let mut out = Vec::with_capacity(args.geoms * aa.len() * palettes.len());
    let mut tile = 0usize;
    for g_idx in 0..args.geoms {
        let geom = if g_idx == 0 {
            Geom::identity()
        } else {
            Geom::draw(
                slot_seed(tag, loc.loc_id, SLOT_GEOM + g_idx as u64),
                args.scale_lo,
                args.scale_hi,
                args.shift_frac_max,
            )
        };
        let (src_x0, src_y0, ratio) = fg.window(geom);
        for level in aa {
            for pal in &palettes {
                let q = if args.jpg_quality_hi > args.jpg_quality_lo {
                    let mut rng = SplitMix64(slot_seed(tag, loc.loc_id, SLOT_QUALITY + tile as u64));
                    let span = (args.jpg_quality_hi - args.jpg_quality_lo) as usize + 1;
                    args.jpg_quality_lo + rng.below(span) as u8
                } else {
                    args.jpg_quality_lo
                };
                let name = format!(
                    "{pal}__g{g_idx}__s{:.4}__sh{:.4}__{}__q{q}.jpg",
                    geom.scale, geom.shift_frac, level.label
                );
                out.push(TileSpec {
                    tile,
                    geom_idx: g_idx,
                    geom,
                    aa_label: level.label.clone(),
                    aa_mode: level.mode,
                    palette: pal.clone(),
                    jpg_quality: q,
                    src_x0,
                    src_y0,
                    ratio,
                    out: format!("{}/{}/{}", args.out_root.trim_end_matches('/'), loc.loc_id, name),
                });
                tile += 1;
            }
        }
    }
    out
}

/// The parsed `--floor-palette` reservation: `(name, count)` pairs, in flag order.
///
/// A floor is drawn **from** the N tiles, never added to them — it reserves the first
/// `sum(count)` slots of the location's fan-out. That is the whole difference between a
/// floor and a bonus (`tools/apportion.py` makes the same distinction with `preseed`), and
/// it is why the reservation is applied before the uniform draw rather than after it.
fn parse_floor(specs: &[String]) -> Result<Vec<(String, usize)>, String> {
    let mut out = Vec::new();
    for s in specs {
        let (name, n) = s.rsplit_once(':').ok_or_else(|| {
            format!("--floor-palette entry '{s}' must be <palette>:<count>, e.g. 'blue_orange:2'")
        })?;
        let n: usize = n
            .parse()
            .map_err(|_| format!("--floor-palette entry '{s}': '{n}' is not a count"))?;
        if name.is_empty() || n == 0 {
            return Err(format!("--floor-palette entry '{s}' has an empty name or a zero count"));
        }
        out.push((name.to_string(), n));
    }
    Ok(out)
}

/// The floor's slot→palette expansion: `[twilight_shifted, twilight_shifted, blue_orange,
/// blue_orange]` for `twilight_shifted:2 blue_orange:2`.
fn floor_slots(floor: &[(String, usize)]) -> Vec<String> {
    floor.iter().flat_map(|(n, c)| std::iter::repeat(n.clone()).take(*c)).collect()
}

/// **Independent mode** (`--tiles N`): N tiles per location, each drawing its own palette,
/// geometry, AA level and JPEG quality from its own seed slot.
///
/// This is not the product fan-out with different numbers. Under the v8b product recipe a
/// location's tiles share `--geoms` distinct fields and every (palette, AA) cell sees the
/// same geometries; here each tile is one draw from the joint distribution, so the axes are
/// decorrelated across the corpus rather than crossed within a location. Both cost the same
/// — every tile is a full resample+colourize either way — so the choice is entirely about
/// what the network sees.
///
/// The **floor** occupies the low slots: `floor_slots` fixes the palette of the first
/// `F = Σ counts` tiles and `--floor-identity` fixes the geometry of the first `I` to the
/// exact identity framing. Everything above those indices is a free draw, and a floor
/// palette that is also in the pool can be drawn again — the floor is a minimum, not a quota.
fn plan_tiles_independent(
    args: &CropBatchArgs,
    aa: &[AaLevel],
    fg: &FieldGeom,
    loc: &Loc,
    tag: u64,
    n_tiles: usize,
    floor: &[String],
) -> Vec<TileSpec> {
    let mut out = Vec::with_capacity(n_tiles);
    for tile in 0..n_tiles {
        // -- palette: reserved by the floor, else uniform over the pool WITH replacement.
        let palette = match floor.get(tile) {
            Some(p) => p.clone(),
            None => {
                let mut rng = SplitMix64(slot_seed(tag, loc.loc_id, SLOT_TILE_PALETTE + tile as u64));
                args.palette_pool[rng.below(args.palette_pool.len())].clone()
            }
        };
        // -- geometry: the exact identity on the reserved low slots, else a fresh draw.
        let geom = if tile < args.floor_identity {
            Geom::identity()
        } else {
            Geom::draw(
                slot_seed(tag, loc.loc_id, SLOT_GEOM + tile as u64),
                args.scale_lo,
                args.scale_hi,
                args.shift_frac_max,
            )
        };
        let (src_x0, src_y0, ratio) = fg.window(geom);
        // -- AA: a uniform draw over the declared levels (two levels => the 50/50 coin).
        let level = {
            let mut rng = SplitMix64(slot_seed(tag, loc.loc_id, SLOT_AA + tile as u64));
            &aa[rng.below(aa.len())]
        };
        // -- quality: uniform integer in [lo, hi].
        let q = if args.jpg_quality_hi > args.jpg_quality_lo {
            let mut rng = SplitMix64(slot_seed(tag, loc.loc_id, SLOT_QUALITY + tile as u64));
            let span = (args.jpg_quality_hi - args.jpg_quality_lo) as usize + 1;
            args.jpg_quality_lo + rng.below(span) as u8
        } else {
            args.jpg_quality_lo
        };
        // The tile index leads the filename. In the product mode (palette, geom, ss) is a
        // uniqueness key; under independent draws it is not — two free slots may land on
        // the same palette and round to the same 4-dp scale/shift — so uniqueness is made
        // structural rather than left to the draw.
        let name = format!(
            "t{tile:02}__{palette}__s{:.4}__sh{:.4}__{}__q{q}.jpg",
            geom.scale, geom.shift_frac, level.label
        );
        out.push(TileSpec {
            tile,
            // Each tile owns its geometry here, so the beautiful path's per-crop
            // normalization cache is keyed on the tile — a shared key would hand tile B
            // the percentile stretch of tile A's window.
            geom_idx: tile,
            geom,
            aa_label: level.label.clone(),
            aa_mode: level.mode,
            palette,
            jpg_quality: q,
            src_x0,
            src_y0,
            ratio,
            out: format!("{}/{}/{}", args.out_root.trim_end_matches('/'), loc.loc_id, name),
        });
    }
    out
}

fn manifest_row(loc: &Loc, fg: &FieldGeom, t: &TileSpec, args: &CropBatchArgs,
                bailout: f64, profile: &str, incomplete: bool) -> String {
    let (ex, ey) = fg.realized_extend();
    let (dx, dy) = t.geom.offset();
    // The family constants, carried through as the ORIGINAL decimal strings (version-
    // invariant render keys) so a replay parses exactly what the location row supplied.
    let mut c_fields = String::new();
    for (re_key, im_key, v) in [
        ("c_re", "c_im", &loc.c),
        ("p_re", "p_im", &loc.p),
        ("zm1_re", "zm1_im", &loc.z1),
    ] {
        if let Some((re, im)) = v {
            c_fields.push_str(&format!(
                ",{}:{},{}:{}", jstr(re_key), jstr(re), jstr(im_key), jstr(im)));
        }
    }
    format!(
        "{{\"loc_id\":{loc_id},\"tile\":{tile},\"out\":{out},\
         \"render\":{{\"cx\":{cx},\"cy\":{cy},\"fw\":{fw},\"fractal_type\":{ft},\
         \"maxiter\":{mi},\"maxiter_policy\":{mip}{cf}}},\
         \"field\":{{\"profile\":{prof},\"field_ss\":{fss},\"sub_w\":{sw},\"sub_h\":{sh},\
         \"pad_x\":{px},\"pad_y\":{py},\"fw_ext\":{fwe},\"extend_x\":{ex},\"extend_y\":{ey},\
         \"bailout\":{bail}}},\
         \"crop\":{{\"geom\":{gi},\"scale\":{sc},\"shift_frac\":{sf},\"shift_angle\":{sa},\
         \"shift_dx\":{dx},\"shift_dy\":{dy},\"src_x0\":{sx0},\"src_y0\":{sy0},\"ratio\":{rt}}},\
         \"tile_geom\":{{\"w\":{tw},\"h\":{th}}},\
         \"aa\":{{\"level\":{al},\"mode\":{am}}},\
         \"palette\":{pal},\"jpg_quality\":{q},\"seed_tag\":{st},\"batch_incomplete\":{inc}}}",
        loc_id = loc.loc_id,
        tile = t.tile,
        out = jstr(&t.out),
        cx = jstr(&loc.cx),
        cy = jstr(&loc.cy),
        fw = jstr(&num(loc.fw)),
        ft = jstr(&loc.kind),
        mi = loc.maxiter,
        mip = jstr(&loc.maxiter_policy),
        cf = c_fields,
        prof = jstr(profile),
        fss = fg.field_ss,
        sw = fg.sub_w,
        sh = fg.sub_h,
        px = fg.pad_x,
        py = fg.pad_y,
        fwe = num(fg.fw_ext),
        ex = num(ex),
        ey = num(ey),
        bail = num(bailout),
        gi = t.geom_idx,
        sc = num(t.geom.scale),
        sf = num(t.geom.shift_frac),
        sa = num(t.geom.angle),
        dx = num(dx),
        dy = num(dy),
        sx0 = num(t.src_x0),
        sy0 = num(t.src_y0),
        rt = num(t.ratio),
        tw = fg.out_w,
        th = fg.out_h,
        al = jstr(&t.aa_label),
        am = jstr(t.aa_mode.as_str()),
        pal = jstr(&t.palette),
        q = t.jpg_quality,
        st = jstr(&args.seed_tag),
        inc = incomplete,
    )
}

// --------------------------------------------------------------------------- //
// Per-location execution
// --------------------------------------------------------------------------- //

/// Render one location's extended field and emit its tiles. Returns
/// `(tiles_written, tiles_skipped, field_secs, tile_secs)`.
fn run_location(
    loc: &Loc,
    tiles: &[TileSpec],
    fg: &FieldGeom,
    palettes: &HashMap<String, Palette>,
    width: u32,
    skip_existing: bool,
) -> Result<(u64, u64, f64, f64), String> {
    let (center, family) = loc.resolve(width)?;
    let profile = family.has_location_profile();
    let bailout = if profile { BAILOUT } else { render_modes::BEAUTIFUL_BAILOUT };

    let frame = fg.frame(center);
    let spacing = frame.pixel_size();
    if spacing <= 1e-13 {
        return Err(format!(
            "pixel spacing {spacing:.3e} inside f64 quantization (deep zoom) — crop-batch is \
             the shallow f64 cache path"
        ));
    }

    // --- the ONE iteration pass (stream-and-discard: local, dropped at return) ---
    let t0 = Instant::now();
    let (field, sub_w, sub_h) =
        render_modes::smooth_field_f64_supersampled(&frame, 1, loc.maxiter, family, bailout)?;
    debug_assert_eq!((sub_w, sub_h), (fg.sub_w, fg.sub_h));
    let field_secs = t0.elapsed().as_secs_f64();

    let cp = ColoringParams::beautiful(Field::Smooth);
    let t1 = Instant::now();
    let (mut wrote, mut skipped) = (0u64, 0u64);
    // Geometry-keyed cache of the beautiful normalization: palette-independent, so the
    // percentile stretch is built once per crop rather than once per tile.
    let mut norms: HashMap<usize, SmoothFieldColorer> = HashMap::new();

    for t in tiles {
        // Defence in depth behind the `--extend` precondition: `build_taps_scaled` clamps
        // at the source bounds, so a crop that ran off the field would emit an
        // edge-smeared tile rather than fail. A silently wrong training tile is exactly
        // the failure the recipe change exists to remove, so it is a hard error here.
        let (x1, y1) = (
            t.src_x0 + t.ratio * fg.out_w as f64,
            t.src_y0 + t.ratio * fg.out_h as f64,
        );
        if t.src_x0 < 0.0 || t.src_y0 < 0.0 || x1 > fg.sub_w as f64 || y1 > fg.sub_h as f64 {
            return Err(format!(
                "tile {} crop window [{:.2},{:.2}]x[{:.2},{:.2}] leaves the {}x{} field \
                 (scale {:.4}, shift {:.4}) — --extend is too small for the draw",
                t.tile, t.src_x0, x1, t.src_y0, y1, fg.sub_w, fg.sub_h,
                t.geom.scale, t.geom.shift_frac
            ));
        }
        if skip_existing && Path::new(&t.out).exists() {
            skipped += 1;
            continue;
        }
        let palette = palettes
            .get(&t.palette)
            .ok_or_else(|| format!("palette '{}' not built", t.palette))?;
        let img = if profile {
            let col = Colorizer::Profile {
                palette,
                density: PROFILE_DENSITY * palette.density_scale(),
            };
            resample_crop(&field, fg, t.src_x0, t.src_y0, t.ratio, t.aa_mode, &col)
        } else {
            let colorer = norms.entry(t.geom_idx).or_insert_with(|| {
                SmoothFieldColorer::new(crop_valids(&field, fg, t.src_x0, t.src_y0, t.ratio), &cp)
            });
            let col = Colorizer::Beautiful { palette, colorer };
            resample_crop(&field, fg, t.src_x0, t.src_y0, t.ratio, t.aa_mode, &col)
        };
        ensure_parent_dir(&t.out)?;
        render::save_jpeg(&img, Path::new(&t.out), t.jpg_quality)?;
        wrote += 1;
    }
    Ok((wrote, skipped, field_secs, t1.elapsed().as_secs_f64()))
}

// --------------------------------------------------------------------------- //
// Entry point
// --------------------------------------------------------------------------- //

pub fn run_crop_batch(args: &CropBatchArgs) -> Result<(), String> {
    if args.width == 0 || args.height == 0 {
        return Err("--width and --height must be > 0".into());
    }
    if args.field_ss == 0 {
        return Err("--field-ss must be > 0".into());
    }
    if args.geoms == 0 {
        return Err("--geoms must be > 0 (geometry 0 is the identity framing)".into());
    }
    if !(args.scale_lo > 0.0 && args.scale_hi >= args.scale_lo) {
        return Err("--scale-lo/--scale-hi must satisfy 0 < lo <= hi".into());
    }
    if args.shift_frac_max < 0.0 {
        return Err("--shift-frac-max must be >= 0".into());
    }
    if args.jpg_quality_lo == 0 || args.jpg_quality_hi > 100 || args.jpg_quality_hi < args.jpg_quality_lo
    {
        return Err("--jpg-quality-lo/--jpg-quality-hi must satisfy 1 <= lo <= hi <= 100".into());
    }
    // CONTAINMENT: the widest crop (`scale_hi`) displaced by the largest shift must fit
    // inside the extended field, on BOTH axes. The margin is an equal plane distance
    // `(extend−1)/2` of the frame WIDTH per side (see `FieldGeom`), the shift is
    // `shift_frac_max` of the frame width in any direction, and the scale overhang is
    // `(scale_hi−1)/2` of each axis's own extent — so the binding axis is the longer one:
    //
    //     extend  >=  1 + 2·shift_frac_max + max(1, H/W)·(scale_hi − 1)
    //
    // At the prompt's 512×288 / [0.90,1.10] / 5% that is 1 + 0.10 + 0.10 = 1.20 exactly:
    // the 1.2 figure and the draw bounds are one decision, with nothing to spare.
    // Validated, never clamped; the realized margin (whole subpixels, rounded up) is
    // never smaller than the flag asks for.
    let aspect = (args.height as f64 / args.width as f64).max(1.0);
    let need = 1.0 + 2.0 * args.shift_frac_max + aspect * (args.scale_hi - 1.0);
    // Tolerance, because the defaults sit EXACTLY on the bound and `1 + 0.1 + (1.1 − 1.0)`
    // is 1.2000000000000002 in binary — a bare `<` rejects the shipped configuration. The
    // slack is 1e-9 of a frame, ~0.001 subpixel at the cache geometry, and the whole-
    // subpixel pad rounds UP, so nothing admitted here can actually leave the field (the
    // per-crop window check in `run_location` is the real guarantee either way).
    if args.extend < need - 1e-9 {
        return Err(format!(
            "--extend {} cannot contain a scale-{} crop shifted by {} of the frame width at \
             {}x{}: needs >= {need}. Raise --extend or tighten the draw.",
            args.extend, args.scale_hi, args.shift_frac_max, args.width, args.height
        ));
    }

    let aa: Vec<AaLevel> = args.aa.iter().map(|s| AaLevel::parse(s)).collect::<Result<_, _>>()?;
    if aa.is_empty() {
        return Err("--aa needs at least one <label>:<mode> entry".into());
    }
    let floor_spec = parse_floor(&args.floor_palette)?;
    let floor = floor_slots(&floor_spec);
    match args.tiles {
        Some(n) => {
            // Independent mode owns the whole fan-out, so the product flags must not be
            // half-set: a run that passed both would silently render one of them.
            if n == 0 {
                return Err("--tiles must be > 0".into());
            }
            if args.palette_pool.is_empty() {
                return Err("--tiles needs --palette-pool (the per-tile palette draw)".into());
            }
            if args.draw_palettes > 0 {
                return Err("--draw-palettes is the product mode's per-LOCATION sample; \
                            --tiles draws a palette per TILE. Pass one or the other."
                    .into());
            }
            if floor.len() > n {
                return Err(format!(
                    "--floor-palette reserves {} of {n} tiles — the floor is drawn FROM the \
                     tiles, not added to them",
                    floor.len()
                ));
            }
            if args.floor_identity > n {
                return Err(format!("--floor-identity {} exceeds --tiles {n}", args.floor_identity));
            }
        }
        None => {
            if !floor.is_empty() || args.floor_identity > 0 {
                return Err("--floor-palette/--floor-identity only apply to --tiles \
                            (independent-draw) mode"
                    .into());
            }
            if args.palettes.is_empty() && args.draw_palettes == 0 {
                return Err(
                    "no palettes: pass --palettes and/or --palette-pool with --draw-palettes".into(),
                );
            }
        }
    }

    if let Some(replay) = &args.replay {
        return run_replay(args, replay);
    }

    // --- locations ---
    let text = std::fs::read_to_string(&args.locations)
        .map_err(|e| format!("read locations {}: {e}", args.locations))?;
    let mut locs: Vec<Loc> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        // `#` lines are the plan's own header. A bulk plan file resolves out-of-tree, away
        // from the module that wrote it, so it carries the command that rebuilds it —
        // which is only possible if the reader tolerates a comment.
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        locs.push(parse_loc(line, locs.len(), args).map_err(|e| format!("line {}: {e}", i + 1))?);
    }
    if locs.is_empty() {
        return Err("locations file is empty".into());
    }
    // BOUNDED END-TO-END: `--limit` runs the WHOLE path (field, crops, JPGs, manifest)
    // on the first N locations. It WRITES real files, so every row it emits carries
    // `batch_incomplete: true`, derived from the flag here rather than hardcoded — a
    // bounded run must stamp itself unusable, not merely be remembered as bounded.
    let incomplete = args.limit.is_some();
    if let Some(n) = args.limit {
        locs.truncate(n);
    }

    // --- colormap library once, each distinct palette once ---
    let cm_text = std::fs::read_to_string(&args.colormaps)
        .map_err(|e| format!("read {}: {e}", args.colormaps))?;
    let library = parse_colormaps(&cm_text).map_err(|e| format!("parse {}: {e}", args.colormaps))?;
    let mut names: Vec<String> = if args.tiles.is_some() {
        // Independent mode never applies `--palettes` (the product mode's always-on list);
        // building those palettes would only hide a typo in the floor.
        floor.clone()
    } else {
        args.palettes.clone()
    };
    names.extend(args.palette_pool.iter().cloned());
    names.sort();
    names.dedup();
    let mut palettes: HashMap<String, Palette> = HashMap::new();
    for n in &names {
        let cm = library
            .iter()
            .find(|c| &c.name == n)
            .ok_or_else(|| format!("palette '{n}' not in {}", args.colormaps))?;
        palettes.insert(
            n.clone(),
            Palette::from_srgb8_stops_mirrored(cm.name.clone(), &cm.stops, false, cm.mirror_needed),
        );
    }

    let tag = tag_hash(&args.seed_tag);
    let probe = FieldGeom::build(args.width, args.height, args.field_ss, args.extend, 1.0);
    let (ex, ey) = probe.realized_extend();
    let n_pal = args.palettes.len() + args.draw_palettes;
    let fanout = match args.tiles {
        Some(n) => format!(
            "{n} independent tiles/loc (pool {}, floor {:?}, identity {})",
            args.palette_pool.len(),
            floor_spec,
            args.floor_identity
        ),
        None => format!("{} geoms x {} AA x {n_pal} palettes", args.geoms, aa.len()),
    };
    eprintln!(
        "crop-batch: {} locations x [{}] = {} tiles; field {}x{} \
         (ss{}, pad {}/{}, realized extend {:.5}/{:.5}), q{}..{}{}",
        locs.len(),
        fanout,
        locs.len() * args.tiles.unwrap_or(args.geoms * aa.len() * n_pal),
        probe.sub_w,
        probe.sub_h,
        args.field_ss,
        probe.pad_x,
        probe.pad_y,
        ex,
        ey,
        args.jpg_quality_lo,
        args.jpg_quality_hi,
        if incomplete { "  [--limit: rows stamped batch_incomplete]" } else { "" },
    );

    ensure_parent_dir(&args.manifest)?;
    let mf = std::fs::File::create(&args.manifest)
        .map_err(|e| format!("create manifest {}: {e}", args.manifest))?;
    let manifest = Mutex::new(std::io::BufWriter::new(mf));

    let done = AtomicU64::new(0);
    let wrote = AtomicU64::new(0);
    let skipped = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let field_ms = AtomicU64::new(0);
    let tile_ms = AtomicU64::new(0);
    let start = Instant::now();
    let log_every = args.log_every.max(1) as u64;

    locs.par_iter().for_each(|loc| {
        let fg = FieldGeom::build(args.width, args.height, args.field_ss, args.extend, loc.fw);
        let tiles = match args.tiles {
            Some(n) => plan_tiles_independent(args, &aa, &fg, loc, tag, n, &floor),
            None => plan_tiles(args, &aa, &fg, loc, tag),
        };
        // The manifest row is DERIVED, not rendered — emitted even for a resumed
        // (skipped) tile, so a resumed run still produces a complete manifest.
        let profile_str = match loc.resolve(args.width) {
            Ok((_, f)) => {
                if f.has_location_profile() { "location" } else { "beautiful_smooth" }
            }
            Err(_) => "unknown",
        };
        let bailout = if profile_str == "location" { BAILOUT } else { render_modes::BEAUTIFUL_BAILOUT };
        {
            let mut w = manifest.lock().unwrap();
            for t in &tiles {
                let _ = writeln!(w, "{}", manifest_row(loc, &fg, t, args, bailout, profile_str, incomplete));
            }
        }
        match run_location(loc, &tiles, &fg, &palettes, args.width, !args.no_resume) {
            Ok((w, s, fs, ts)) => {
                wrote.fetch_add(w, Ordering::Relaxed);
                skipped.fetch_add(s, Ordering::Relaxed);
                field_ms.fetch_add((fs * 1000.0) as u64, Ordering::Relaxed);
                tile_ms.fetch_add((ts * 1000.0) as u64, Ordering::Relaxed);
            }
            Err(e) => {
                failed.fetch_add(1, Ordering::Relaxed);
                eprintln!("FAIL loc {}: {e}", loc.loc_id);
            }
        }
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % log_every == 0 || n as usize == locs.len() {
            let el = start.elapsed().as_secs_f64();
            eprintln!(
                "  [{n}/{}] {:.2} loc/s  elapsed {:.0}s  ETA {:.0}s  (wrote {}, skipped {}, failed {})",
                locs.len(),
                n as f64 / el.max(1e-9),
                el,
                (locs.len() as f64 - n as f64) / (n as f64 / el.max(1e-9)).max(1e-9),
                wrote.load(Ordering::Relaxed),
                skipped.load(Ordering::Relaxed),
                failed.load(Ordering::Relaxed),
            );
        }
    });

    manifest
        .into_inner()
        .unwrap()
        .flush()
        .map_err(|e| format!("flush manifest: {e}"))?;

    let el = start.elapsed().as_secs_f64();
    let nf = failed.load(Ordering::Relaxed);
    let nw = wrote.load(Ordering::Relaxed);
    let (fms, tms) = (field_ms.load(Ordering::Relaxed), tile_ms.load(Ordering::Relaxed));
    println!("=== crop-batch ===");
    println!("locations:  {}", locs.len());
    println!("manifest:   {}", args.manifest);
    println!("tiles:      {nw} written, {} skipped, {nf} location(s) failed",
             skipped.load(Ordering::Relaxed));
    println!("field cpu:  {:.1}s   tile cpu: {:.1}s   (summed over threads)",
             fms as f64 / 1000.0, tms as f64 / 1000.0);
    println!("wall:       {el:.1}s   ({:.3} s/location)", el / locs.len() as f64);
    println!("incomplete: {incomplete}");
    if nf > 0 {
        return Err(format!("{nf} location(s) failed"));
    }
    Ok(())
}

/// `--replay`: re-render tiles from an emitted manifest using the **recorded** geometry.
/// Nothing is re-drawn, so replay does not depend on the RNG, the palette pool, or the
/// draw parameters — only on the manifest row and the field it names.
fn run_replay(args: &CropBatchArgs, manifest_path: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(manifest_path)
        .map_err(|e| format!("read manifest {manifest_path}: {e}"))?;

    let cm_text = std::fs::read_to_string(&args.colormaps)
        .map_err(|e| format!("read {}: {e}", args.colormaps))?;
    let library = parse_colormaps(&cm_text).map_err(|e| format!("parse {}: {e}", args.colormaps))?;

    let root = args.replay_out_root.as_deref();
    let mut n = 0usize;
    // Group consecutive rows of one location so the field is iterated once, exactly as the
    // forward path does (the manifest is emitted location-major).
    let mut cur: Option<(u64, Vec<f32>, FieldGeom, bool, Loc)> = None;

    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ctx = |e: String| format!("manifest line {}: {e}", i + 1);
        let loc_id = jsonl::field_usize(line, "loc_id").ok_or_else(|| ctx("missing loc_id".into()))? as u64;
        let field_ss = jsonl::field_usize(line, "field_ss").ok_or_else(|| ctx("missing field_ss".into()))? as u32;
        let tw = jsonl::field_usize(line, "w").ok_or_else(|| ctx("missing tile_geom.w".into()))? as u32;
        let th = jsonl::field_usize(line, "h").ok_or_else(|| ctx("missing tile_geom.h".into()))? as u32;
        let pad_x = jsonl::field_usize(line, "pad_x").ok_or_else(|| ctx("missing pad_x".into()))? as u32;
        let pad_y = jsonl::field_usize(line, "pad_y").ok_or_else(|| ctx("missing pad_y".into()))? as u32;
        let fw = jsonl::field_str(line, "fw")
            .and_then(|s| s.parse::<f64>().ok())
            .ok_or_else(|| ctx("missing render.fw".into()))?;
        let src_x0 = jsonl::field_f64(line, "src_x0").ok_or_else(|| ctx("missing src_x0".into()))?;
        let src_y0 = jsonl::field_f64(line, "src_y0").ok_or_else(|| ctx("missing src_y0".into()))?;
        let ratio = jsonl::field_f64(line, "ratio").ok_or_else(|| ctx("missing ratio".into()))?;
        let mode = AaMode::parse(&jsonl::field_str(line, "mode").ok_or_else(|| ctx("missing aa.mode".into()))?)
            .map_err(ctx)?;
        let pal_name = jsonl::field_str(line, "palette").ok_or_else(|| ctx("missing palette".into()))?;
        let q = jsonl::field_usize(line, "jpg_quality").ok_or_else(|| ctx("missing jpg_quality".into()))? as u8;
        let out = jsonl::field_str(line, "out").ok_or_else(|| ctx("missing out".into()))?;

        let loc = Loc {
            loc_id,
            cx: jsonl::field_str(line, "cx").ok_or_else(|| ctx("missing cx".into()))?,
            cy: jsonl::field_str(line, "cy").ok_or_else(|| ctx("missing cy".into()))?,
            fw,
            kind: jsonl::field_str(line, "fractal_type").unwrap_or_else(|| "mandelbrot".into()),
            c: pair(line, "c_re", "c_im"),
            p: pair(line, "p_re", "p_im"),
            z1: pair(line, "zm1_re", "zm1_im"),
            maxiter: jsonl::field_usize(line, "maxiter").ok_or_else(|| ctx("missing maxiter".into()))? as u32,
            maxiter_policy: String::new(),
        };

        // The field geometry is taken from the RECORDED pads, not recomputed from
        // `--extend` — a replay must not depend on today's flags.
        let base_w = tw * field_ss;
        let base_h = th * field_ss;
        let fg = FieldGeom {
            out_w: tw,
            out_h: th,
            field_ss,
            sub_w: base_w + 2 * pad_x,
            sub_h: base_h + 2 * pad_y,
            pad_x,
            pad_y,
            fw_ext: fw * ((base_w + 2 * pad_x) as f64) / (base_w as f64),
        };

        let refresh = !matches!(&cur, Some((id, _, _, _, _)) if *id == loc_id);
        if refresh {
            let (center, family) = loc.resolve(tw).map_err(ctx)?;
            let profile = family.has_location_profile();
            let bail = if profile { BAILOUT } else { render_modes::BEAUTIFUL_BAILOUT };
            let (field, _, _) = render_modes::smooth_field_f64_supersampled(
                &fg.frame(center), 1, loc.maxiter, family, bail,
            )?;
            cur = Some((loc_id, field, fg, profile, loc));
        }
        let (_, field, fg, profile, _) = cur.as_ref().unwrap();

        let cm = library
            .iter()
            .find(|c| c.name == pal_name)
            .ok_or_else(|| format!("palette '{pal_name}' not in {}", args.colormaps))?;
        let palette =
            Palette::from_srgb8_stops_mirrored(cm.name.clone(), &cm.stops, false, cm.mirror_needed);

        let cp = ColoringParams::beautiful(Field::Smooth);
        let img = if *profile {
            let col = Colorizer::Profile {
                palette: &palette,
                density: PROFILE_DENSITY * palette.density_scale(),
            };
            resample_crop(field, fg, src_x0, src_y0, ratio, mode, &col)
        } else {
            let colorer =
                SmoothFieldColorer::new(crop_valids(field, fg, src_x0, src_y0, ratio), &cp);
            let col = Colorizer::Beautiful { palette: &palette, colorer: &colorer };
            resample_crop(field, fg, src_x0, src_y0, ratio, mode, &col)
        };

        // Mirror `<loc_id>/<slot>.jpg` under the alternate root — a FLAT root would collide
        // two locations' identically-named identity slots, and a byte-comparison keyed on
        // the basename would then silently compare 35 files where 36 were written.
        let dest = match root {
            Some(r) => format!(
                "{}/{}/{}",
                r.trim_end_matches('/'),
                loc_id,
                Path::new(&out)
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| format!("{n}.jpg"))
            ),
            None => out.clone(),
        };
        ensure_parent_dir(&dest)?;
        render::save_jpeg(&img, Path::new(&dest), q)?;
        n += 1;
    }

    println!("=== crop-batch (replay) ===");
    println!("manifest: {manifest_path}");
    println!("tiles:    {n}");
    Ok(())
}

// --------------------------------------------------------------------------- //
// Args
// --------------------------------------------------------------------------- //

/// `crop-batch` subcommand: one extended-field iteration pass per location, every tile
/// derived from it as crop + resample + colormap. See [`run_crop_batch`] and the module
/// docs for the AA-derivation and containment reasoning.
#[derive(Args, Debug)]
pub struct CropBatchArgs {
    /// Location JSONL: one row per LOCATION (not per tile) —
    /// `{loc_id, cx, cy, fw, fractal_type, c_re/c_im, p_re/p_im, zm1_re/zm1_im, maxiter,
    /// maxiter_policy}`. `cx`/`cy` are decimal strings; `maxiter` is the **canonical**
    /// frame's cap.
    #[arg(long, default_value = "data/v10/manifest.jsonl")]
    pub locations: String,

    /// Colormap library (selective-mirror loaded, as everywhere else).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub colormaps: String,

    /// Tile output root; tiles land at `<root>/<loc_id>/<slot>.jpg`.
    #[arg(long, default_value = "scratch/crop_batch/tiles")]
    pub out_root: String,

    /// Per-tile manifest JSONL (one row per tile, carrying its realized geometry).
    #[arg(long, default_value = "scratch/crop_batch/tiles.jsonl")]
    pub manifest: String,

    /// Canonical tile width (the cache geometry; v4..v10 is 512x288).
    #[arg(long, default_value_t = 512)]
    pub width: u32,

    /// Canonical tile height.
    #[arg(long, default_value_t = 288)]
    pub height: u32,

    /// Field subpixels per canonical output pixel. The default 2 matches the legacy
    /// `antialiased` tile's supersample, so the filtered derivation runs at ratio
    /// `scale·2 ∈ [1.8, 2.2]`. Raising it costs `field_ss²` in iteration.
    #[arg(long, default_value_t = 2)]
    pub field_ss: u32,

    /// Extended-field factor (1.2 = +10% per side). Must be `>= scale_hi +
    /// 2·shift_frac_max` or the widest shifted crop leaves the field; validated, not
    /// clamped. Realized to whole subpixels per side (rounded up).
    #[arg(long, default_value_t = 1.2)]
    pub extend: f64,

    /// Crop geometries per location, geometry 0 being the identity framing (dead centre,
    /// scale exactly 1.0 — the deploy composition, present in every (palette, AA) cell).
    #[arg(long, default_value_t = 3)]
    pub geoms: usize,

    /// Crop scale draw, lower bound.
    #[arg(long, default_value_t = 0.90)]
    pub scale_lo: f64,

    /// Crop scale draw, upper bound.
    #[arg(long, default_value_t = 1.10)]
    pub scale_hi: f64,

    /// Max centre shift as a fraction of the **canonical** frame width; magnitude is
    /// `U(0, max)` with a uniform direction.
    #[arg(long, default_value_t = 0.05)]
    pub shift_frac_max: f64,

    /// AA levels as `<label>:<mode>`, e.g. `aliased:point antialiased:lanczos3`. The
    /// label is the two-value cache-manifest vocabulary; the mode is the reconstruction
    /// (`point` = nearest-neighbour, the honest ss1 stand-in — see the module docs).
    #[arg(long, num_args = 1.., value_delimiter = ' ',
          default_values_t = ["aliased:point".to_string(), "antialiased:lanczos3".to_string()])]
    pub aa: Vec<String>,

    /// Palettes applied to **every** location (v8b's pinned pair: the deploy-matched
    /// scoring instrument and the map the labels were formed through).
    #[arg(long, num_args = 0.., value_delimiter = ' ',
          default_values_t = ["twilight_shifted".to_string(), "blue_orange".to_string()])]
    pub palettes: Vec<String>,

    /// Drawable pool for the per-location palette draw.
    #[arg(long, num_args = 0.., value_delimiter = ' ')]
    pub palette_pool: Vec<String>,

    /// How many palettes to draw per location from `--palette-pool` (seeded per location,
    /// without replacement).
    #[arg(long, default_value_t = 0)]
    pub draw_palettes: usize,

    /// **Independent-draw mode**: emit exactly N tiles per location, each drawing its own
    /// palette (uniform over `--palette-pool`, with replacement), geometry, AA level and
    /// JPEG quality from its own seed slot. Replaces the `geoms x AA x palettes` product;
    /// `--geoms`, `--palettes` and `--draw-palettes` do not apply.
    #[arg(long)]
    pub tiles: Option<usize>,

    /// Guaranteed floor inside `--tiles`, as `<palette>:<count>` — reserves the LOW tile
    /// slots. Drawn FROM the N tiles, never added to them.
    #[arg(long, num_args = 0.., value_delimiter = ' ')]
    pub floor_palette: Vec<String>,

    /// How many of the `--tiles` carry the exact identity framing (dead centre, scale 1.0
    /// — the deploy composition). Reserves the low slots, alongside `--floor-palette`.
    #[arg(long, default_value_t = 0)]
    pub floor_identity: usize,

    /// Per-tile JPEG quality draw, lower bound. `lo == hi` disables the draw (and is what
    /// reproduces the v4..v10 cache's flat q85).
    #[arg(long, default_value_t = 85)]
    pub jpg_quality_lo: u8,

    /// Per-tile JPEG quality draw, upper bound (inclusive).
    #[arg(long, default_value_t = 95)]
    pub jpg_quality_hi: u8,

    /// Iteration-cap fallback for a location row carrying no `maxiter`. There is no
    /// default: the cap must be the CANONICAL frame's `auto_maxiter`, which only the
    /// caller knows, and a silent flat cap is how v4..v8 iterated every tile at 8000.
    #[arg(long)]
    pub maxiter: Option<u32>,

    /// Cap-policy token stamped into rows that carry no `maxiter_policy` of their own
    /// (`tools/corpus/location.maxiter_policy_token()`; `""` is the legacy policy).
    #[arg(long, default_value = "")]
    pub maxiter_policy: String,

    /// Seed namespace for every per-location draw. Changing it reshuffles the whole
    /// fan-out; it is part of the recipe's identity.
    #[arg(long, default_value = "v11-crop")]
    pub seed_tag: String,

    /// Run the WHOLE path (field → crops → JPGs → manifest) on the first N locations.
    /// Every row it writes is stamped `batch_incomplete: true`.
    #[arg(long)]
    pub limit: Option<usize>,

    /// Re-render tiles from an emitted manifest using their **recorded** geometry
    /// (no draws). Ignores the location/draw flags.
    #[arg(long)]
    pub replay: Option<String>,

    /// Write replayed tiles here (flat) instead of over their recorded `out` paths.
    #[arg(long)]
    pub replay_out_root: Option<String>,

    /// Re-render tiles whose output already exists (default: skip, so a killed run
    /// resumes).
    #[arg(long, default_value_t = false)]
    pub no_resume: bool,

    /// Progress log cadence, in locations.
    #[arg(long, default_value_t = 200)]
    pub log_every: u32,
}
