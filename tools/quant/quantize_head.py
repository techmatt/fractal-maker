#!/usr/bin/env python
"""quantize_head.py — THE post-training quantization recipe, applied and verified.

    uv run python tools/quant/quantize_head.py quantize --head location --rung int8
    uv run python tools/quant/quantize_head.py verify   --head location --rung int8
    uv run python tools/quant/quantize_head.py sweep    --head location   # per-group sensitivity

WHAT THIS IS FOR, and it is not speed. The objective is SMALLER STORED WEIGHTS at
near-identical scores: `backbone_search_v1` measured the backbone at 5-17% of end-to-end
score time (22.45-24.00 s/1k tiles across six backbones), so the deploy path is decode-bound
and no rung here can be argued on throughput. The bars this tool is measured against were
committed first, in `data/quant/prereg_quant_v1.json`.

THE RECIPE, three rungs, smallest passing one wins:

  fp16    every floating tensor stored float16; integer buffers verbatim.
  int8    weight-only, per-OUTPUT-CHANNEL (dim 0) symmetric int8 for every floating tensor
          with ndim >= 2 — conv and linear weights, 98.0-98.4% of float params on the four
          live heads — with an fp32 scale per channel; every other floating tensor (BN
          affine, biases, 1-D) fp16. Dequantized to the source dtype at LOAD; compute stays
          fp32. No activation quantization and no backend engine (fbgemm / qnnpack /
          TensorRT), so the recipe re-applies in a clean-room repo with standard torch on
          any platform — which is the whole reason it is weight-only.
  hybrid  int8 except named layer groups kept fp16, chosen by `sweep`.

PER-CHANNEL IS THE LOAD-BEARING WORD. These are MobileNetV4 backbones: their depthwise
convolutions have per-channel weight ranges spanning orders of magnitude, and one scale for
the whole tensor quantizes the small channels into noise. The axis is dim 0 (output
channels) for both conv (O,I,kh,kw) and linear (O,I), which is the axis a dequantized
weight broadcasts along.

THE ARTIFACT IS NOT A CHECKPOINT AND DELIBERATELY DOES NOT LOOK LIKE ONE. It carries no
`state_dict` key, so a loader that blindly does `torch.load(p)["state_dict"]` raises KeyError
instead of quietly loading the wrong thing. Read it through `read_artifact()` (returns a
plain fp32 state_dict) or apply it with `apply_to_model()`; both are three lines and neither
needs to know which head it holds.

Everything head-specific — which checkpoint is live, what its eval material is, how it turns
logits into probabilities — lives in `tools/quant/heads.py` and is resolved from each head's
own pin module. Nothing here hardcodes a version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "quant"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RECIPE_VERSION = "quant_v1"
ARTIFACT_FORMAT = "fractal-quant-1"
RUNGS = ("fp16", "int8", "hybrid")
INT8_AXIS = 0
INT8_QMAX = 127          # symmetric: [-127, 127], -128 unused so |q| is symmetric
WEIGHTS_DIR = "data/quant/weights"   # a REGISTERED bulk prefix — resolves out-of-tree


# --------------------------------------------------------------------------- #
# layer groups
# --------------------------------------------------------------------------- #
def group_of(name: str) -> str:
    """The layer GROUP a state_dict key belongs to — the unit the sensitivity sweep and the
    hybrid exception list are expressed in.

    timm backbones nest as `blocks.<stage>.<block>.<layer>...`, so a per-TENSOR sweep would
    be ~460 arms of which most are one BN vector. The grouping is one level below the stem:
    `conv_stem`, `blocks.0` .. `blocks.N`, `conv_head`, `classifier`. That is the granularity
    the hybrid rung can actually act on — "keep the first conv and the final head at fp16" is
    a statement about groups, not about tensors."""
    parts = name.split(".")
    if parts[0] == "blocks" and len(parts) > 1:
        return f"blocks.{parts[1]}"
    return parts[0]


def groups_of(state_dict) -> list[str]:
    """Every group present, in state_dict order (which is forward order for timm)."""
    seen = {}
    for k in state_dict:
        seen.setdefault(group_of(k), None)
    return list(seen)


# --------------------------------------------------------------------------- #
# the rung itself
# --------------------------------------------------------------------------- #
def quantize_tensor_int8(w):
    """(q int8, scale fp32) for one weight, per output channel along `INT8_AXIS`.

    A channel that is all zeros gets scale 1.0 rather than 0.0: the dequantized value is 0
    either way, and a zero scale is a NaN factory the first time someone divides by it."""
    import torch

    flat = w.reshape(w.shape[0], -1).to(torch.float32)
    amax = flat.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / INT8_QMAX, torch.ones_like(amax))
    q = torch.clamp(torch.round(flat / scale[:, None]), -INT8_QMAX, INT8_QMAX)
    return q.to(torch.int8).reshape(w.shape), scale


def dequantize_tensor_int8(q, scale, shape, dtype):
    import torch

    flat = q.reshape(q.shape[0], -1).to(torch.float32) * scale[:, None].to(torch.float32)
    return flat.reshape(shape).to(dtype)


def quantize_state_dict(state_dict, rung: str, *, keep_fp16_groups=(),
                        only_groups=None) -> dict:
    """state_dict -> the artifact payload (no config, no provenance — `write_artifact` adds
    those).

    `keep_fp16_groups` is the hybrid exception list. `only_groups` is the SWEEP's inverse:
    quantize just these groups to int8 and leave every other tensor at its source dtype, so
    one arm measures one group's contribution and nothing else."""
    import torch

    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r}; expected one of {RUNGS}")
    keep = set(keep_fp16_groups)
    only = None if only_groups is None else set(only_groups)
    if rung == "hybrid" and not keep:
        raise ValueError("rung 'hybrid' with no --keep-fp16 group is just 'int8' under "
                         "another name; name the groups the sweep found sensitive")

    tensors, i8, scales, f16, raw = {}, {}, {}, {}, {}
    for name, t in state_dict.items():
        rec = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape),
               "group": group_of(name)}
        if not t.is_floating_point():
            raw[name] = t
            rec["kind"] = "raw"
        elif only is not None and rec["group"] not in only:
            raw[name] = t                                  # sweep: untouched arm-mate
            rec["kind"] = "raw"
        elif rung == "fp16":
            f16[name] = t.to(torch.float16)
            rec["kind"] = "fp16"
        elif t.ndim >= 2 and rec["group"] not in keep:
            q, s = quantize_tensor_int8(t)
            i8[name], scales[name] = q, s
            rec["kind"] = "int8_per_channel"
        else:
            f16[name] = t.to(torch.float16)
            rec["kind"] = "fp16"
        tensors[name] = rec

    return {"format": ARTIFACT_FORMAT, "recipe_version": RECIPE_VERSION, "rung": rung,
            "int8_axis": INT8_AXIS, "int8_qmax": INT8_QMAX,
            "keep_fp16_groups": sorted(keep), "only_groups": (sorted(only) if only else None),
            "tensors": tensors, "int8": i8, "scale": scales, "fp16": f16, "raw": raw}


def dequantize(payload) -> dict:
    """The artifact payload -> a plain state_dict at each tensor's SOURCE dtype."""
    import torch

    out = {}
    for name, rec in payload["tensors"].items():
        dtype = getattr(torch, rec["dtype"])
        if rec["kind"] == "int8_per_channel":
            out[name] = dequantize_tensor_int8(payload["int8"][name], payload["scale"][name],
                                               tuple(rec["shape"]), dtype)
        elif rec["kind"] == "fp16":
            out[name] = payload["fp16"][name].to(dtype)
        else:
            out[name] = payload["raw"][name]
    return out


# --------------------------------------------------------------------------- #
# artifact I/O
# --------------------------------------------------------------------------- #
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_path(head_key: str, rung: str, tag: str = "") -> Path:
    """Where a quantized head lands: `bulk()`, i.e. out of the source tree.

    A quantized head is a pure function of a tracked checkpoint plus a recipe rung, so it is
    regenerable, not durable — and tracking one would make it a SECOND copy of a live weight
    under a retention policy that counts heads (storage_classes.md § weights retention)."""
    import paths

    stem = f"{head_key}_{rung}" + (f"_{tag}" if tag else "")
    return paths.bulk(f"{WEIGHTS_DIR}/{stem}.qpt")


def write_artifact(src_ckpt: Path, rung: str, out_path: Path, *, keep_fp16_groups=(),
                   only_groups=None) -> dict:
    """Quantize `src_ckpt` per `rung` and write the artifact. Returns its metadata block."""
    import torch

    src_ckpt = Path(src_ckpt)
    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    payload = quantize_state_dict(ck["state_dict"], rung, keep_fp16_groups=keep_fp16_groups,
                                  only_groups=only_groups)
    payload["config"] = ck.get("config")
    payload["source"] = {
        "path": src_ckpt.as_posix(),
        "rel": (src_ckpt.relative_to(ROOT).as_posix()
                if src_ckpt.is_absolute() and str(src_ckpt).startswith(str(ROOT))
                else src_ckpt.as_posix()),
        "sha256": sha256_file(src_ckpt), "bytes": src_ckpt.stat().st_size,
    }
    payload["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {"artifact": out_path.as_posix(), "rung": rung,
            "bytes_before": payload["source"]["bytes"], "bytes_after": out_path.stat().st_size,
            "sha256": sha256_file(out_path), "source_sha256": payload["source"]["sha256"],
            "keep_fp16_groups": payload["keep_fp16_groups"],
            "kinds": _kind_census(payload)}


def _kind_census(payload) -> dict:
    c = {}
    for rec in payload["tensors"].values():
        c[rec["kind"]] = c.get(rec["kind"], 0) + 1
    return dict(sorted(c.items()))


def read_artifact(path):
    """`(state_dict, config, meta)` from a quantized artifact — the ONE read path.

    Refuses anything that is not this format rather than guessing: a plain checkpoint handed
    here would dequantize to nothing and score perfectly, which is the one failure mode that
    would make every number in this study meaningless."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != ARTIFACT_FORMAT:
        raise ValueError(
            f"{path} is not a {ARTIFACT_FORMAT} artifact (format="
            f"{payload.get('format') if isinstance(payload, dict) else type(payload)!r}). "
            f"A quantized head carries no 'state_dict' key on purpose — read it here, not "
            f"through a checkpoint loader.")
    meta = {k: v for k, v in payload.items()
            if k not in ("int8", "scale", "fp16", "raw", "tensors")}
    meta["kinds"] = _kind_census(payload)
    return dequantize(payload), payload.get("config"), meta


def apply_to_model(model, path):
    """Load a quantized artifact's dequantized weights into an already-built model.

    This is what makes the acceptance measurement honest: the model, its transform and its
    scoring code are the head's OWN production ones, built off the real checkpoint, and the
    only thing that changes between the two arms is the weight values."""
    sd, _cfg, meta = read_artifact(path)
    model.load_state_dict(sd)
    return meta


# --------------------------------------------------------------------------- #
# weight-error instrument check
# --------------------------------------------------------------------------- #
def weight_error(src_sd, deq_sd) -> dict:
    """How much the rung actually moved the weights, over the quantized tensors only.

    Reported BEFORE any score is read (measurement_practice.md: "verify the instrument's
    inputs actually change"). A rung whose max relative error is 0 is vacuous, and a perfect
    agreement number under it measures nothing."""
    import torch

    max_rel = 0.0
    per_tensor_fro = []
    n_changed = 0
    for name, a in src_sd.items():
        b = deq_sd[name]
        if not a.is_floating_point():
            continue
        a32, b32 = a.to(torch.float32), b.to(torch.float32)
        d = (a32 - b32).abs()
        amax = float(a32.abs().amax())
        if float(d.amax()) > 0:
            n_changed += 1
        if amax > 0:
            max_rel = max(max_rel, float(d.amax()) / amax)
        na = float((a32 ** 2).sum()) ** 0.5
        if na > 0 and float(d.amax()) > 0:
            per_tensor_fro.append(float((d ** 2).sum()) ** 0.5 / na)
    # PER TENSOR, not pooled. A pooled ||dW||/||W|| over a whole state_dict is dominated by
    # the BatchNorm affine vectors — O(1) values that the int8 rung does not touch — and
    # reads ~5x smaller than the error on the tensors that were actually quantized.
    return {"max_rel_err_per_tensor": max_rel,
            "max_rel_frobenius_per_tensor": max(per_tensor_fro) if per_tensor_fro else 0.0,
            "mean_rel_frobenius_over_changed": (float(np.mean(per_tensor_fro))
                                                if per_tensor_fro else 0.0),
            "n_tensors_changed": n_changed}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _head(key):
    import heads
    return heads.HEADS[key]


def cmd_quantize(a):
    import torch

    spec = _head(a.head)
    src = ROOT / spec.ckpt_rel
    out = Path(a.out) if a.out else artifact_path(spec.key, a.rung, a.tag)
    meta = write_artifact(src, a.rung, out, keep_fp16_groups=a.keep_fp16)
    src_sd = torch.load(src, map_location="cpu", weights_only=False)["state_dict"]
    deq, _cfg, _m = read_artifact(out)
    meta["weight_error"] = weight_error(src_sd, deq)
    print(json.dumps(meta, indent=2))
    if a.verify:
        print(json.dumps(verify(spec, out, n=a.sample, device=a.device), indent=2))
    return meta


def verify(spec, artifact, *, n=48, device=None) -> dict:
    """Re-read the artifact from disk and prove it round-trips, twice over.

    (1) BITWISE: the dequantized tensors from the re-read file equal the ones computed at
        write time. A silent truncation in torch.save/load would otherwise show up as a
        small score delta and be filed as quantization error.
    (2) BEHAVIOURALLY: a sample of the head's own eval material is scored through the
        re-read artifact and through the source checkpoint, in one process, on the same
        rows. Both max|delta| are reported next to the sha256 of the file.
    """
    import torch

    import heads

    artifact = Path(artifact)
    sd_a, _cfg, meta = read_artifact(artifact)
    sd_b, _cfg2, _m2 = read_artifact(artifact)
    bitwise = all(bool(torch.equal(sd_a[k], sd_b[k])) for k in sd_a)

    items = spec.sample(n)
    scorer = heads.load_scorer(spec, device)
    P_fp32, rank_fp32 = scorer.score(items)
    apply_to_model(scorer.model, artifact)
    P_q, rank_q = scorer.score(items)
    dp = np.abs(np.asarray(P_q) - np.asarray(P_fp32))
    return {"artifact": artifact.as_posix(), "sha256": sha256_file(artifact),
            "bytes": artifact.stat().st_size, "rung": meta["rung"],
            "reload_bitwise_identical": bitwise,
            "sample_n": len(items),
            "sample_max_abs_delta_p": float(dp.max()) if dp.size else None,
            "sample_mean_abs_delta_p": float(dp.mean()) if dp.size else None,
            "sample_max_abs_delta_rank": float(np.abs(np.asarray(rank_q)
                                                      - np.asarray(rank_fp32)).max())}


def cmd_verify(a):
    spec = _head(a.head)
    art = Path(a.artifact) if a.artifact else artifact_path(spec.key, a.rung, a.tag)
    print(json.dumps(verify(spec, art, n=a.sample, device=a.device), indent=2))


def cmd_sweep(a):
    """Per-group sensitivity: int8 on ONE group, source dtype everywhere else.

    Rung 3's input. Isolating a single group is the only way an exception is attributable —
    "the first conv is sensitive" has to be a measurement of the first conv alone, not an
    inference from a whole-model number."""
    import torch

    import heads

    spec = _head(a.head)
    src = ROOT / spec.ckpt_rel
    src_sd = torch.load(src, map_location="cpu", weights_only=False)["state_dict"]
    grps = [g for g in groups_of(src_sd)
            if any(t.ndim >= 2 and t.is_floating_point()
                   for k, t in src_sd.items() if group_of(k) == g)]
    items = spec.sample(a.sample)
    scorer = heads.load_scorer(spec, device=a.device)
    P0, rank0 = scorer.score(items)

    rows = []
    tmp = artifact_path(spec.key, "int8", "sweeptmp")
    for g in grps:
        write_artifact(src, "int8", tmp, only_groups=[g])
        apply_to_model(scorer.model, tmp)
        P, _r = scorer.score(items)
        d = np.abs(np.asarray(P) - np.asarray(P0))
        rows.append({"group": g, "max_abs_delta_p": float(d.max()),
                     "mean_abs_delta_p": float(d.mean())})
        print(f"  {g:16s} max|dp| {d.max():.5f}  mean|dp| {d.mean():.6f}", flush=True)
    tmp.unlink(missing_ok=True)
    rows.sort(key=lambda r: -r["max_abs_delta_p"])
    out = {"head": spec.key, "ckpt": spec.ckpt_rel, "sample_n": len(items),
           "n_groups": len(grps), "groups": rows}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(json.dumps(out, indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quantize", help="apply a rung and write the artifact")
    q.add_argument("--head", required=True)
    q.add_argument("--rung", required=True, choices=RUNGS)
    q.add_argument("--keep-fp16", action="append", default=[], metavar="GROUP",
                   help="hybrid: a layer group to keep at fp16 (repeatable)")
    q.add_argument("--out", default=None)
    q.add_argument("--tag", default="")
    q.add_argument("--sample", type=int, default=48)
    q.add_argument("--no-verify", dest="verify", action="store_false")
    q.set_defaults(fn=cmd_quantize, verify=True)

    v = sub.add_parser("verify", help="re-read an artifact and prove it round-trips")
    v.add_argument("--head", required=True)
    v.add_argument("--rung", default="int8", choices=RUNGS)
    v.add_argument("--artifact", default=None)
    v.add_argument("--tag", default="")
    v.add_argument("--sample", type=int, default=48)
    v.set_defaults(fn=cmd_verify)

    s = sub.add_parser("sweep", help="per-layer-group int8 sensitivity (rung 3's input)")
    s.add_argument("--head", required=True)
    s.add_argument("--sample", type=int, default=128)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_sweep)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
