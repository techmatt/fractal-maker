//! `save_jpeg` must be atomic, and must overwrite.
//!
//! The augmentation-cache builder (`v4-render-batch`) resumes by SKIPPING any row whose
//! output already `exists()`. That makes a non-atomic write a correctness bug rather than
//! a lost-work bug: a process killed mid-encode leaves a short file, and the resume then
//! never rewrites it, so a truncated tile enters the training cache silently. `save_jpeg`
//! therefore encodes to a sibling temp file and renames over the destination.
//!
//! Killing a process mid-encode is not testable in-process, so these pin the two
//! observable properties the fix rests on:
//!   1. the rename REPLACES an existing destination (this is the platform-specific half —
//!      POSIX `rename(2)` replaces, and Windows only does because `fs::rename` maps to
//!      `MoveFileEx(MOVEFILE_REPLACE_EXISTING)`; a naive `CreateFile`-based port would not);
//!   2. no `.tmp` sibling survives a successful write.

use fractal_generator::render::save_jpeg;
use image::RgbImage;

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("fg_atomic_jpeg_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn solid(w: u32, h: u32, v: u8) -> RgbImage {
    RgbImage::from_pixel(w, h, image::Rgb([v, v, v]))
}

#[test]
fn save_jpeg_overwrites_an_existing_file() {
    let d = tmpdir("overwrite");
    let p = d.join("tile.jpg");

    save_jpeg(&solid(64, 36, 10), &p, 85).unwrap();
    let first = std::fs::read(&p).unwrap();

    // A second write to the SAME path must succeed and replace the bytes. On Windows this
    // is the whole point: rename-onto-existing is not universally allowed.
    save_jpeg(&solid(64, 36, 240), &p, 85).unwrap();
    let second = std::fs::read(&p).unwrap();

    assert_ne!(first, second, "second save_jpeg did not replace the file's contents");
    image::open(&p).expect("overwritten file is not a decodable image");
}

#[test]
fn save_jpeg_leaves_no_temp_sibling() {
    let d = tmpdir("notemp");
    save_jpeg(&solid(32, 18, 128), &d.join("tile.jpg"), 85).unwrap();

    let leftovers: Vec<_> = std::fs::read_dir(&d)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temp sibling(s) survived a successful write: {leftovers:?}");
}

#[test]
fn save_jpeg_creates_a_decodable_file_of_the_right_size() {
    let d = tmpdir("decode");
    let p = d.join("tile.jpg");
    save_jpeg(&solid(512, 288, 64), &p, 85).unwrap();
    let img = image::open(&p).unwrap().to_rgb8();
    assert_eq!((img.width(), img.height()), (512, 288));
}
