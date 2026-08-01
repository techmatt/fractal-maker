//! Direct coverage for [`energy::occupancy`] — the detail-occupancy gate on the
//! live `--expand` path (`enrich.rs` and `guided_descend.rs` both call it with
//! [`energy::OCC_GX`]/[`energy::OCC_GY`]/[`energy::OCC_FLOOR`]).
//!
//! This is **not** a port of the deleted `tests/occupancy_parity.rs`. That test
//! asserted byte-parity against `score_complexity.py` / `complexity_scores.json`,
//! and neither the scorer nor the artifact exists any more — an assertion whose
//! reference is gone is not a weaker assertion, it is no assertion. What is worth
//! pinning today is the function's own contract, because every caller reads the
//! result as *"the fraction of the frame that carries detail"* and compares it to
//! a calibrated floor:
//!
//!  - it is a **fraction of tiles**, so it lands on a `k/(gx·gy)` lattice, is
//!    bounded in `[0, 1]`, and localizes — detail confined to half the frame
//!    reads as exactly half;
//!  - the floor comparison is **strict** (`mean > floor`), so a dead-flat frame
//!    reads 0 even at `floor = 0`. `guided_descend`'s Stage-2 cull is a `<`
//!    against this value, so the two strictnesses have to stay opposed;
//!  - it is **monotone non-increasing in `floor`** — the property that makes
//!    `OCC_FLOOR` a calibration knob rather than an arbitrary constant;
//!  - it is exactly the floor reduction of [`energy::tile_energy`], which is the
//!    documented relationship between the gate and `present`'s focus centroid.
//!    Two public functions sharing one primitive is a claim that can rot silently;
//!  - degenerate geometry returns 0 rather than panicking or dividing by zero.
//!
//! The live constants get their own case: `occupancy` drops the ragged remainder
//! (`w % gx`, `h % gy`), so at the gate resolution the tiling has to divide
//! evenly or the calibrated floor is being applied to a differently-shaped grid
//! than the one it was calibrated on.

use image::{Rgb, RgbImage};

use fractal_generator::energy::{self, OCC_FLOOR, OCC_GX, OCC_GY};

/// The gate's native resolution — the label geometry `enrich` renders at.
const W: u32 = 1280;
const H: u32 = 720;

const BLACK: Rgb<u8> = Rgb([0, 0, 0]);
const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
const GRAY: Rgb<u8> = Rgb([128, 128, 128]);

/// Solid fill — zero edge energy everywhere.
fn flat(w: u32, h: u32, c: Rgb<u8>) -> RgbImage {
    RgbImage::from_pixel(w, h, c)
}

/// Pixel-scale checkerboard over `x < split`, `fill` to the right of it. The
/// checkerboard saturates edge energy (OKLab ΔE ≈ 1 per neighbor); the fill has
/// none, so tiles either side of `split` are unambiguously occupied / not.
fn half_checker(w: u32, h: u32, split: u32, fill: Rgb<u8>) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        if x < split {
            if (x + y) % 2 == 0 { WHITE } else { BLACK }
        } else {
            fill
        }
    })
}

#[test]
fn a_flat_frame_occupies_nothing_and_the_floor_comparison_is_strict() {
    let img = flat(W, H, GRAY);
    assert_eq!(energy::occupancy(&img, OCC_GX, OCC_GY, OCC_FLOOR), 0.0);
    // floor 0.0 with a tile mean of exactly 0.0: `>` says empty, `>=` would say
    // full. The gate's whole job is rejecting empty frames, so this is the one
    // comparison it cannot get backwards.
    assert_eq!(energy::occupancy(&img, OCC_GX, OCC_GY, 0.0), 0.0);
}

#[test]
fn a_saturated_frame_occupies_every_tile() {
    let img = half_checker(W, H, W, BLACK);
    assert_eq!(energy::occupancy(&img, OCC_GX, OCC_GY, OCC_FLOOR), 1.0);
}

#[test]
fn detail_confined_to_half_the_frame_reads_as_exactly_half() {
    // The split is on a tile boundary (640 = 16 x 40px tiles), so no tile
    // straddles it: columns 0..15 are saturated, 16..31 are dead flat. The seam
    // edge at x = 639 falls inside tile 15, which is occupied either way.
    assert_eq!(W / OCC_GX as u32 * 16, 640);
    let img = half_checker(W, H, 640, GRAY);
    assert_eq!(energy::occupancy(&img, OCC_GX, OCC_GY, OCC_FLOOR), 0.5);
}

#[test]
fn occupancy_is_a_fraction_of_tiles_on_the_k_over_n_lattice() {
    // Not cosmetic: callers compare it to a floor calibrated as a *fraction*, so
    // the denominator must be the tile count and not the pixel count.
    let n = (OCC_GX * OCC_GY) as f64;
    for split in [0u32, 40, 200, 640, 1240, W] {
        let img = half_checker(W, H, split, GRAY);
        let occ = energy::occupancy(&img, OCC_GX, OCC_GY, OCC_FLOOR);
        assert!((0.0..=1.0).contains(&occ), "occupancy {occ} out of range");
        let k = occ * n;
        assert!((k - k.round()).abs() < 1e-9, "occupancy {occ} is not k/{n}");
    }
}

#[test]
fn occupancy_is_monotone_non_increasing_in_the_floor() {
    // A checkerboard whose CONTRAST ramps left-to-right, so tile mean energy
    // ramps with it. A plain colour gradient will not do: its tile means are all
    // within one decade of each other, so the sweep jumps 1 -> 0 in one step and
    // "monotone" holds vacuously. Here the floor crosses tile columns off one at
    // a time, which is what makes OCC_FLOOR a calibration knob at all.
    let img = RgbImage::from_fn(W, H, |x, y| {
        let a = ((x * 255) / (W - 1)) as u8;
        if (x + y) % 2 == 0 { Rgb([a, a, a]) } else { BLACK }
    });
    // The terminal floor is above the saturating value: a black/white neighbor
    // pair is ~1.0 of OKLab ΔE on each axis, so per-pixel energy tops out near
    // sqrt(2) and 2.0 is unreachable by construction.
    let floors = [0.0, 1e-4, 1e-3, 5e-3, OCC_FLOOR, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0];
    let mut prev = f64::INFINITY;
    let mut seen = Vec::new();
    for f in floors {
        let occ = energy::occupancy(&img, OCC_GX, OCC_GY, f);
        assert!(occ <= prev, "occupancy rose from {prev} to {occ} at floor {f}");
        prev = occ;
        seen.push(occ);
    }
    assert_eq!(prev, 0.0, "an unreachable floor must occupy nothing");
    // Non-vacuity: a two-valued sweep would satisfy monotonicity without ever
    // exercising the comparison against a partially-occupied frame.
    let distinct = {
        let mut v = seen.clone();
        v.dedup();
        v.len()
    };
    assert!(distinct >= 4, "sweep is degenerate ({distinct} distinct): {seen:?}");
    assert!(seen.iter().any(|&o| o > 0.0 && o < 1.0),
            "no floor left the frame partially occupied: {seen:?}");
}

#[test]
fn occupancy_is_exactly_the_floor_reduction_of_tile_energy() {
    // `tile_energy` is documented as the same primitive with the floor reduction
    // dropped, and `present`'s focus centroid depends on that being true. Checked
    // over several floors so it is the *reduction* being pinned, not one value.
    let img = RgbImage::from_fn(W, H, |x, y| {
        let v = (((x / 7) ^ (y / 5)) % 256) as u8;
        Rgb([v, 255 - v, (x % 256) as u8])
    });
    let means = energy::tile_energy(&img, OCC_GX, OCC_GY);
    assert_eq!(means.len(), OCC_GX * OCC_GY);
    for f in [0.0, 1e-4, OCC_FLOOR, 0.02, 0.1, 1.0] {
        let expect = means.iter().filter(|&&m| m > f).count() as f64 / means.len() as f64;
        assert_eq!(energy::occupancy(&img, OCC_GX, OCC_GY, f), expect,
                   "occupancy disagrees with tile_energy at floor {f}");
    }
}

#[test]
fn degenerate_geometry_returns_zero_rather_than_panicking() {
    let img = half_checker(W, H, W, BLACK);
    assert_eq!(energy::occupancy(&img, 0, OCC_GY, OCC_FLOOR), 0.0);
    assert_eq!(energy::occupancy(&img, OCC_GX, 0, OCC_FLOOR), 0.0);
    // A grid finer than the image: `w / gx == 0`, which is the div-by-zero.
    let tiny = half_checker(10, 10, 10, BLACK);
    assert_eq!(energy::occupancy(&tiny, OCC_GX, OCC_GY, OCC_FLOOR), 0.0);
    assert_eq!(energy::occupancy(&RgbImage::new(0, 0), OCC_GX, OCC_GY, OCC_FLOOR), 0.0);
}

#[test]
fn the_live_tiling_divides_the_gate_resolution_exactly() {
    // The ragged remainder is DROPPED, so an uneven grid would silently score a
    // sub-rectangle of the frame against a floor calibrated on the whole one.
    assert_eq!(W as usize % OCC_GX, 0, "{W} is not divisible by OCC_GX {OCC_GX}");
    assert_eq!(H as usize % OCC_GY, 0, "{H} is not divisible by OCC_GY {OCC_GY}");
    assert_eq!(W as usize / OCC_GX, 40);
    assert_eq!(H as usize / OCC_GY, 40);
}

#[test]
fn a_ragged_trailing_remainder_is_dropped_from_the_tiling() {
    // 1285 / 32 still floors to a 40px tile, so the tiled region is columns
    // 0..1279 in both images. The extra columns replicate column 1279, so they
    // add no seam edge either — any difference in the result would mean the
    // remainder was tiled rather than dropped.
    let base = half_checker(W, H, 640, GRAY);
    let padded = RgbImage::from_fn(W + 5, H, |x, y| *base.get_pixel(x.min(W - 1), y));
    assert_eq!(padded.width() as usize / OCC_GX, W as usize / OCC_GX);
    assert_eq!(energy::occupancy(&padded, OCC_GX, OCC_GY, OCC_FLOOR),
               energy::occupancy(&base, OCC_GX, OCC_GY, OCC_FLOOR));
}

