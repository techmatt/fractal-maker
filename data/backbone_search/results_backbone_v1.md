# Backbone comparison — results

Control: **mnv4_conv_medium**. Population: PRIMARY = 2190 eval locations that touch neither training nor the checkpoint pick (408 at label>=3). Deltas are paired cluster-bootstrap (B=5000, over split_group (the leakage-closure group the v11 holdout was drawn over — locations inside one are not independent)); a CI covering 0 is a TIE.

| arm | params M | pretrain | ckpt | train h | VRAM MB | score s/1k | AUC>=3 | delta vs control | AUC>=4 | AUC>=2 | exact | adj |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastvit_sa12_s0 | 10.559 | in1k, distilled | e16 | 1.29 | 4767 | 23.3 | 0.9513 | +0.0021 [-0.0047, +0.0095] TIE | 0.9322 | 0.9430 | 0.752 | 0.983 |
| mnv4_conv_medium_s0 | 8.438 | in12k, 384px | e35 | 0.86 | 2621 | 22.6 | 0.9492 | — (control) | 0.9203 | 0.9280 | 0.726 | 0.974 |
| mnv4_conv_medium_s2 | 8.438 | in12k, 384px | e31 | 0.90 | 2621 | 22.45 | 0.9461 | — (control) | 0.9187 | 0.9273 | 0.732 | 0.974 |
| vit_small_p16_s0 | 21.721 | in21k -> in1k (AugReg) | e38 | 1.64 | 3492 | 24.0 | 0.9432 | -0.0060 [-0.0149, +0.0025] TIE | 0.9208 | 0.9346 | 0.724 | 0.975 |
| mnv4_conv_medium_s1 | 8.438 | in12k, 384px | e28 | 0.87 | 2621 | 22.57 | 0.9428 | — (control) | 0.9149 | 0.9263 | 0.724 | 0.978 |
| mnv4_conv_large_s0 | 31.314 | in1k, 384px | e7 | 1.32 | 4513 | 22.81 | 0.9363 | -0.0129 [-0.0231, -0.0035] CONTROL | 0.9365 | 0.9293 | 0.692 | 0.975 |
| mnv4_hybrid_medium_s0 | 9.797 | in1k, 384px (MQA-hybrid) | e28 | 0.95 | 3134 | 22.56 | 0.9323 | -0.0169 [-0.0281, -0.0068] CONTROL | 0.8937 | 0.9344 | 0.740 | 0.978 |
| effnetv2_s_s0 | 20.181 | in21k -> in1k | e3 | 2.12 ⧗ | 1680 | 23.69 | 0.9218 | -0.0274 [-0.0407, -0.0149] CONTROL | 0.9209 | 0.9325 | 0.730 | 0.976 |

⧗ = gradient checkpointing (memory-time trade, identical gradients): the train-h column is not a clean architecture cost for that arm.

## Per-partition delta vs control — DESCRIPTIVE, unadjusted over 9 slices

| arm | julia:mandelbrot | julia:multibrot3 | julia:multibrot4 | julia:multibrot5 | mandelbrot | multibrot3 | multibrot4 | multibrot5 | phoenix |
|---|---|---|---|---|---|---|---|---|---|
| fastvit_sa12_s0 | -0.013 | — (3 pos) | -0.027 | -0.038 | -0.003 | +0.009 | +0.015 | +0.016 | +0.021 |
| vit_small_p16_s0 | -0.022 | — (3 pos) | -0.045 | -0.113* | -0.012* | +0.001 | -0.001 | +0.004 | +0.005 |
| mnv4_conv_large_s0 | -0.005 | — (3 pos) | -0.096* | -0.042 | -0.010* | -0.018 | -0.006 | +0.004 | -0.051* |
| mnv4_hybrid_medium_s0 | -0.027* | — (3 pos) | -0.105* | -0.064 | -0.003 | -0.002 | -0.012 | -0.014 | -0.007 |
| effnetv2_s_s0 | -0.014 | — (3 pos) | -0.038 | -0.074* | -0.002 | -0.018 | -0.024 | -0.042* | -0.020 |

## Pooled AUC>=3 vs EVERY control seed — the verdict's seed sensitivity

With one seed per arm, WHICH control run an arm is paired against is a coin flip. ROBUST = the same verdict against all three.

| arm | s0 | s1 | s2 | ROBUST | seed-dependent |
|---|---|---|---|---|---|
| fastvit_sa12_s0 | +0.0021 TIE | +0.0085 ARM | +0.0053 TIE | **TIE** | YES |
| vit_small_p16_s0 | -0.0060 TIE | +0.0003 TIE | -0.0029 TIE | **TIE** | no |
| mnv4_conv_large_s0 | -0.0129 CONTROL | -0.0065 TIE | -0.0097 TIE | **TIE** | YES |
| mnv4_hybrid_medium_s0 | -0.0169 CONTROL | -0.0106 TIE | -0.0138 CONTROL | **TIE** | YES |
| effnetv2_s_s0 | -0.0274 CONTROL | -0.0210 CONTROL | -0.0242 CONTROL | **CONTROL** | no |

## Round-2 seed bands — pooled PRIMARY AUC>=3

| arm | seeds | min | mean | max |
|---|---|---|---|---|
| fastvit_sa12 | 1 | 0.9513 | 0.9513 | 0.9513 |
| mnv4_conv_medium | 3 | 0.9428 | 0.9460 | 0.9492 |
| vit_small_p16 | 1 | 0.9432 | 0.9432 | 0.9432 |
| mnv4_conv_large | 1 | 0.9363 | 0.9363 | 0.9363 |
| mnv4_hybrid_medium | 1 | 0.9323 | 0.9323 | 0.9323 |
| effnetv2_s | 1 | 0.9218 | 0.9218 | 0.9218 |
