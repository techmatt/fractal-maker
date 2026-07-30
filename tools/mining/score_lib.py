"""Shared v3 scoring helpers for the biased mining harness.

Two scoring paths, both through the EXACT v3 deploy transform
(`classifier.data.Transform(train=False)` = 1280x720 -> 384x224 bicubic-stretch +
normalize), so scores stay on the trained distribution:

1. `Scorer.score_pils` / `score_paths` — score in-memory PIL images or JPGs on
   disk. Returns the CORN triple per frame: ordinal score in [0,2]
   (= sigma(l0)+sigma(l1) = `score_from_logits`), P(not-bad) = sigma(l0)
   (= P(label>=2)), P(good) = sigma(l1) (= P(label>=3)).

2. `run_enrich_score` — the in-memory Rust->Python bridge. Launches the frozen
   `enrich --mode score` subcommand (iterate-once at label geometry, present
   gates, recolor under the roster, stream raw RGB), scores every streamed frame,
   and returns {idx: {ki: (score, p_notbad, p_good)}} joined with the per-location
   gate verdict + palette list from the Rust --meta-out sidecar. No crops to disk.

The mining harness drives the same `enrich` machinery with *aggressive params*
(custom rosters, geometries) without touching any production default.

Sibling bridge: `tools/corpus/enrich_score.py` drives the same `enrich --mode
score` machinery for the labeling path, but is a *different contract* (a v2 CLI
writing scored.jsonl, not a v3-pinned library returning the CORN triple) —
deliberately not unified. The 16-byte stream header both parse (`HDR =
struct.Struct("<IIII")`) is owned by the Rust side (`src/enrich.rs`), not by
either script.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from classifier.data import Transform  # noqa: E402
from classifier.model import build_model  # noqa: E402

HDR = struct.Struct("<IIII")  # idx, ki, w, h  (little-endian u32 x4)
# The model the biased mining harness was calibrated against. NOT a default:
# `Scorer(model_path=...)` is required so no path can *silently* score with v3.
# The v5-intended callers (reframe/atlas/step0) always pass v5 explicitly via
# `make_scorer`; the two mining tools (harvest.py, calibrate_t2.py) pass this.
DEFAULT_V3 = "data/classifier/v3/model_best.pt"
BIN = "target/release/fractal-generator.exe"


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")


def corn_decode(p_notbad: float, p_good: float, t_good: float = 0.5,
                p_great: float | None = None, t_great: float = 0.5) -> int:
    """Canonical CORN hard-class decode -> {1, 2, 3} (bad / okay / good), or {1..4} when the
    head's third cutpoint is supplied (bad / okay / good / great).

    The ordinal sigmoids are the cumulative rank probabilities
    ``p_notbad = sigma(l0) = P(class >= 2)``, ``p_good = sigma(l1) = P(class >= 3)`` and — on a
    K=4 head (v8+) — ``p_great = sigma(l2) = P(class >= 4)``.
    Rank-consistent hard class = ``1 + #{cumulative probs >= threshold}``. This is NOT
    recoverable from the summed ``E[ord] = sum sigma(l_k)`` scalar (two frames with
    equal E[ord] can decode to different classes), so callers pass the
    probabilities and MUST NOT threshold the score. Single source of truth for the
    decode; reuse it, don't reimplement the >= threshold counting inline.

    ``p_great=None`` (the default) is the K=3 decode, BYTE-IDENTICAL to the historical
    two-probability form — every v5..v7-era caller stays put and can still only reach class 3.
    A K=4 caller passes the third probability and can reach class 4. ``t_great`` stays at its
    natural cutpoint 0.5 and gets NO per-family calibration (see
    data/v8/t_good_derivation.json ``no_class4_threshold``): only the q3 operating point is
    swept per partition, because only q3 gates admission.

    Note the rule COUNTS thresholds met rather than chaining them, which is how it has always
    worked — so a frame with ``p_great >= t_great`` but ``p_good < t_good`` decodes to 3, not 4.
    CORN's cumulative probabilities are not guaranteed monotone (see the monotonicity check in
    tools/v6/threshold_sweep.py), and counting degrades such a frame by one rank rather than
    promoting it on the strength of a cutpoint whose predecessor it failed.

    ``t_good`` is the q3 (rank-3) operating point on ``p_good``. It defaults to 0.5,
    which is BYTE-IDENTICAL to the historical decode — every existing caller stays put.
    Discovery sites opt in to a lower per-degree threshold (the v6 sweep knee) by
    passing ``t_good`` explicitly. The rank-2 gate on ``p_notbad`` stays fixed at 0.5:
    a class-3 outcome must still be not-bad, so lowering ``t_good`` below 0.5 can only
    turn a would-be class-2 into class-3, never resurrect a class-1 (the AND rule holds
    because ``p_notbad >= p_good`` is not guaranteed — see the monotonicity check in
    tools/v6/threshold_sweep.py — but a class-1 has ``p_notbad < 0.5`` and is capped at
    ``1 + 0 + 1 = 2`` regardless, i.e. it can reach class-2 but not class-3)."""
    cls = 1 + int(p_notbad >= 0.5) + int(p_good >= t_good)
    if p_great is not None:
        cls += int(p_great >= t_great)
    return cls


class Scorer:
    """CORN ordinal head + deploy transform, exposing the cumulative probabilities per frame.

    **K is read off the checkpoint** (``config["num_classes"]``), not assumed: v1..v7 are K=3
    (2 cutpoints), v8+ is K=4 (3 cutpoints). Building the head at the wrong K raises a
    state-dict shape mismatch on load rather than scoring wrongly, so a version flip cannot
    quietly degrade here — but it also means the K=3-shaped accessors below must stay honest
    about what they drop. ``score_pils``/``score_paths`` return the historical
    ``(score, p_notbad, p_good)`` triple for every K, where ``score`` is the FULL
    ``sum sigma(l_k)`` rank score in ``[0, K-1]`` (so a K=4 score is in [0,3], not [0,2]) and
    the third cutpoint is simply not surfaced. A caller that needs class-4 capability uses
    ``score_pils_k``/``score_paths_k``, which return every cumulative probability.
    """

    def __init__(self, model_path: str, device: str | None = None):
        self.device = pick_device(device)
        path = model_path if os.path.isabs(model_path) else str(ROOT / model_path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        if cfg["target"] != "ordinal":
            raise SystemExit(f"expected ordinal head, got target={cfg['target']!r}")
        self.k = int(cfg.get("num_classes", 3))    # tiers; the head emits k-1 cutpoint logits
        model = build_model(
            target=cfg["target"], drop_rate=cfg.get("drop_rate", 0.2),
            drop_path_rate=cfg.get("drop_path_rate", 0.1), pretrained=False,
            num_classes=self.k,
        )
        model.load_state_dict(ckpt["state_dict"])
        self.model = model.eval().to(self.device)
        self.transform = Transform(
            geometry=cfg["geometry"], interp=cfg["interpolation"],
            mean=tuple(cfg["mean"]), std=tuple(cfg["std"]), train=False,
        )
        self.cfg = cfg

    @torch.no_grad()
    def score_pils_k(self, imgs: list[Image.Image]):
        """Returns (score, P) where P is (N, k-1) cumulative probs [P(>=2), P(>=3), ...] and
        score = P.sum(axis=1) = ``score_from_logits``, in [0, k-1]."""
        x = torch.stack([self.transform(im) for im in imgs]).to(self.device)
        if self.device != "cpu":
            with torch.autocast(device_type=self.device.split(":")[0]):
                logits = self.model(x)
        else:
            logits = self.model(x)
        logits = logits.float().cpu()
        P = torch.sigmoid(logits).numpy()            # (N, k-1) cumulative rank probs
        return P.sum(axis=1), P

    def score_pils(self, imgs: list[Image.Image]):
        """Returns (score, p_notbad, p_good) numpy arrays, one row per image. K=3 shape; on a
        K=4 head the third cutpoint is dropped (use `score_pils_k` to keep it)."""
        score, P = self.score_pils_k(imgs)
        return score, P[:, 0], P[:, 1]

    def _score_buffered(self, paths, batch_size, fn):
        out = []
        buf: list[Image.Image] = []

        def flush():
            if not buf:
                return
            out.extend(fn(buf))
            buf.clear()

        for p in paths:
            with Image.open(p) as im:
                im.load()
                buf.append(im.convert("RGB"))
            if len(buf) >= batch_size:
                flush()
        flush()
        return out

    def score_paths(self, paths, batch_size: int = 64):
        """Score JPGs on disk. Returns list of (score, p_notbad, p_good)."""
        def fn(buf):
            s, nb, g = self.score_pils(buf)
            return list(zip(s.tolist(), nb.tolist(), g.tolist()))
        return self._score_buffered(paths, batch_size, fn)

    def score_paths_k(self, paths, batch_size: int = 64):
        """Score JPGs on disk, keeping every cutpoint. Returns list of
        ``(score, p_ge2, p_ge3, ...)`` tuples of length k (1 + the k-1 cumulative probs), so a
        K=4 head yields 4-tuples and a K=3 head yields the same 3-tuples `score_paths` does."""
        def fn(buf):
            s, P = self.score_pils_k(buf)
            return [(float(sv), *(float(v) for v in row)) for sv, row in zip(s.tolist(), P)]
        return self._score_buffered(paths, batch_size, fn)


def read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if not buf:
                return None
            raise EOFError(f"stream ended mid-record ({len(buf)}/{n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


def run_enrich_score(
    scorer: Scorer,
    pool_path: str,
    colormaps: str,
    *,
    k: int,
    seed: int = 0,
    width: int = 1280,
    height: int = 720,
    score_ss: int = 1,
    maxiter: int = 8000,
    black_cap: float = 0.30,
    occ_floor: float = 0.321,
    meta_out: str,
    batch_size: int = 96,
    bin_path: str = BIN,
    progress_every: int = 4096,
    frame_cb=None,
    log=print,
):
    """Stream every (location x roster-palette) frame through v3.

    Returns (scores, locs):
      scores: {idx: {ki: (score, p_notbad, p_good)}}
      locs:   list of per-location meta dicts (idx, cx, cy, fw, gated, gate_reason,
              black_fraction, occupancy, palettes[]) read from the Rust sidecar.
    """
    meta_abs = meta_out if os.path.isabs(meta_out) else str(ROOT / meta_out)
    os.makedirs(os.path.dirname(meta_abs), exist_ok=True)
    cmd = [
        str(ROOT / bin_path) if not os.path.isabs(bin_path) else bin_path,
        "enrich", "--mode", "score",
        "--pool", pool_path, "--colormaps", colormaps,
        "--k", str(k), "--seed", str(seed),
        "--width", str(width), "--height", str(height), "--score-ss", str(score_ss),
        "--maxiter", str(maxiter), "--black-cap", str(black_cap), "--occ-floor", str(occ_floor),
        "--meta-out", meta_out,
    ]
    log("launching: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=None, bufsize=0)

    scores: dict[int, dict[int, tuple]] = {}
    pend_imgs: list[Image.Image] = []
    pend_meta: list[tuple[int, int]] = []
    n_frames = 0

    def flush():
        nonlocal n_frames
        if not pend_imgs:
            return
        s, nb, g = scorer.score_pils(pend_imgs)
        for (idx, ki), a, b, c in zip(pend_meta, s.tolist(), nb.tolist(), g.tolist()):
            scores.setdefault(idx, {})[ki] = (a, b, c)
        n_frames += len(pend_imgs)
        pend_imgs.clear()
        pend_meta.clear()

    stream = proc.stdout
    while True:
        hdr = read_exact(stream, HDR.size)
        if hdr is None:
            break
        idx, ki, w, h = HDR.unpack(hdr)
        payload = read_exact(stream, w * h * 3)
        if payload is None:
            raise EOFError("EOF before image payload")
        arr = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)
        pil = Image.fromarray(arr, "RGB")
        if frame_cb is not None:
            frame_cb(idx, ki, pil)
        pend_imgs.append(pil)
        pend_meta.append((idx, ki))
        if len(pend_imgs) >= batch_size:
            flush()
            if progress_every and n_frames % progress_every == 0:
                log(f"  scored {n_frames} frames ({len(scores)} locations)...")
    flush()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"enrich --mode score exited {rc}")

    locs: list[dict] = []
    with open(meta_abs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or json.loads(line).get("kind") == "header":
                continue
            locs.append(json.loads(line))
    return scores, locs
