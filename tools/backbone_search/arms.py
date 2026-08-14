"""THE arm table for the backbone comparison, and the two path rules that go with it.

`prompts/backbone_search_v1.md`: train the v11 head recipe VERBATIM over a set of
timm-pretrained backbones and report quality / cost / per-partition slices against a
freshly-retrained control. One variable moves — the backbone — so everything a run needs
that is NOT the backbone comes off `data/classifier/v11/model_best.pt["config"]` at train
time and is asserted unchanged (`train_arm.py`). This file holds only what differs.

A frozen dataclass from the start, per CLAUDE.md's "writing a builder for one instance":
there are eight arms today and a ninth is a table line, not a refactor.

STORAGE, and it is two classes rather than one:
  * `records/` — per-arm config/metrics/history/log, the prereg, the results table and
    the figure. Small, and the only durable record of a run whose weights are thrown
    away, so they are TRACKED (`.gitignore` re-includes `/data/backbone_search/`).
  * `arms/`    — the weights. Staged arms stay untracked (the prompt says so, and the
    ACTIVE+PREVIOUS retention policy in storage_classes.md would refuse them anyway), so
    the family is registered in `artifacts.RELOCATED_PREFIXES` and is born OUT of the
    tree. In-tree it would be committed, not merely present, because the `!` above
    re-includes the whole subtree — the same trap `data/atlas/tau_h_rederive` names.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import paths  # noqa: E402

# The recipe source: v11's checkpoint config, read verbatim by every arm (control too).
V11_CKPT = ROOT / "data" / "classifier" / "v11" / "model_best.pt"

RECORDS_REL = "data/backbone_search/records"     # tracked
WEIGHTS_REL = "data/backbone_search/arms"        # bulk, relocated out-of-tree


@dataclass(frozen=True)
class ArmSpec:
    name: str                       # short id; the directory name and the table key
    timm_model: str                 # timm model.tag — the ONE thing that varies
    pretrain: str                   # what the tag's weights were trained on (table column)
    why: str                        # why this arm is in the set
    create_kwargs: dict = field(default_factory=dict)   # timm create-time kwargs
    is_control: bool = False
    # Gradient checkpointing: a MEMORY-TIME trade, not a recipe change — the gradients are
    # the same ones the unchecked graph produces, so the optimization is untouched and only
    # the arm's TRAIN WALL CLOCK stops being an architecture cost (it is stamped, and the
    # results table carries the flag). Set where the arm otherwise exceeds 8 GB at the
    # inherited batch size: without it effnetv2_s peaks at 8,494 MB and convnextv2_tiny at
    # 9,348 MB, and Windows spills to host memory rather than raising — which is why their
    # first smoke read 4.0 s and 28.9 s per step against the control's 0.14 s. The prompt's
    # instruction is "drop, don't shrink batch"; this drops NEITHER, because the batch and
    # the schedule are what it protects and checkpointing moves neither.
    grad_checkpointing: bool = False

    def weights_dir(self, seed: int) -> Path:
        return paths.bulk(f"{WEIGHTS_REL}/{self.name}/s{seed}")

    def record_dir(self, seed: int) -> Path:
        return paths.durable(f"{RECORDS_REL}/{self.name}/s{seed}")


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        name="mnv4_conv_medium", timm_model="mobilenetv4_conv_medium.e250_r384_in12k",
        pretrain="in12k, 384px", is_control=True,
        why="CONTROL — v5..v11's backbone, retrained fresh under these exact conditions "
            "so every delta separates backbone effect from retrain variance.",
    ),
    ArmSpec(
        name="mnv4_hybrid_medium", timm_model="mobilenetv4_hybrid_medium.ix_e550_r384_in1k",
        pretrain="in1k, 384px (MQA-hybrid)",
        why="The control's own family with attention blocks added at the same width — "
            "the cheapest read on whether attention buys anything here at all.",
    ),
    ArmSpec(
        name="mnv4_conv_large", timm_model="mobilenetv4_conv_large.e600_r384_in1k",
        pretrain="in1k, 384px",
        why="Same family, ~3x the compute: separates 'wrong architecture' from 'not "
            "enough capacity', which a cross-family win cannot.",
    ),
    ArmSpec(
        name="effnetv2_s", timm_model="tf_efficientnetv2_s.in21k_ft_in1k",
        pretrain="in21k -> in1k", grad_checkpointing=True,
        why="The strongest conventional conv baseline at this size, and a different "
            "pretrain corpus (in21k) than the control's in12k.",
    ),
    ArmSpec(
        name="fastvit_sa12", timm_model="fastvit_sa12.apple_dist_in1k",
        pretrain="in1k, distilled",
        why="THE mobile-transformer pick: a self-attention/conv hybrid whose train-time "
            "overparameterized blocks re-parameterize away at inference — this head runs "
            "over every ledger rescore and intake, so an arm whose deploy cost falls below "
            "its train cost is the one worth measuring in a throughput comparison, and "
            "distillation makes it the strongest of its class at this size. repvit_m1_5 was "
            "the first pick and was REPLACED: timm's RepVit takes no `drop_path_rate`, so "
            "running it would have silently dropped stochastic depth from the frozen recipe "
            "— a second moved variable, which the design law forbids.",
    ),
    ArmSpec(
        name="vit_small_p16", timm_model="vit_small_patch16_224.augreg_in21k_ft_in1k",
        pretrain="in21k -> in1k (AugReg)",
        create_kwargs={"img_size": (224, 384)},
        why="The small-ViT arm with strong pretraining. The deploy transform does NOT "
            "move: 224x384 is fed as-is and timm resamples the pretrained pos-embed to "
            "the 14x24 grid (both divide by patch 16). dinov2 was rejected for this slot "
            "— patch 14 does not divide 384, so it could only be fed by changing the "
            "deploy transform, which the design law forbids.",
    ),
)

ARMS_BY_NAME = {a.name: a for a in ARMS}
CONTROL = next(a for a in ARMS if a.is_control)
assert sum(a.is_control for a in ARMS) == 1, "exactly one control arm"

# Declared, then dropped BEFORE it ran. Kept here rather than deleted: the reason an arm was
# not measured is part of the study's record, and a bare absence would read as an oversight.
# Removing it from ARMS is also what STOPS it — an in-flight runner's plan is fixed at
# launch, so `train_arm --arm convnextv2_tiny` now fails at argparse in a second and the
# queue moves on, with no process surgery and a log line saying so.
DROPPED = (
    {"name": "convnextv2_tiny", "timm_model": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
     "when": "2026-08-14, queued but not yet started; no arm had been scored",
     "who": "Matt",
     "why": "COST-SUITABILITY, decided ahead of the measurement rather than after it. "
            "5.73 s/1k tiles GPU-only score cost against the control's 1.09 (5.3x) on a "
            "head that runs over every ledger rescore and intake, a 106 MB weight against "
            "34 MB (212 MB tracked under ACTIVE+PREVIOUS), and 4.7-5.7 h to train against "
            "0.9 h. It would need a large quality win to be adoptable at that price and "
            "there is no evidence predicting one.",
     "what_is_given_up": "It was the ONLY self-supervised-pretrain arm, so the study does "
                         "not test whether an FCMAE backbone reads fractal texture better "
                         "than a supervised one. That hypothesis is untested, not refuted. "
                         "convnextv2_nano.fcmae_ft_in22k_in1k_384 (15.6 M / ~60 MB, "
                         "projected ~2.3 h) is the cheap way to buy it later."},
)
