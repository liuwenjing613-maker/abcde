# Eval-Prior Validation Macro-F1 Experiments

Date: 2026-06-04

## Purpose

These experiments use the sampled eval-prior validation splits:

```text
artifacts/exp/eval_prior_val_splits/exact_public_with_replacement
```

Each validation fold has 382 rows sampled with replacement to match public leaderboard support:

| Class | Count |
|---|---:|
| 中度 | 292 |
| 正常 | 15 |
| 轻度 | 46 |
| 重度 | 14 |
| 非常严重 | 15 |

Goal: improve Macro-F1 on this eval-prior validation surface. This is intentionally **not** a clean independent validation estimate, because duplicated subjects are required to match public support. Treat it as a stress-test/calibration surface.

## Code Changes

Added/extended:

- `scripts/sample_eval_prior_val_splits.py`
  - Generates 5 eval-prior validation folds.
  - Stores `source_row`, `subject_id`, `repeat_count`, and `repeat_index`.
- `scripts/train_no_dass_eval_prior_splits.py`
  - Trains no-DASS DeepResidual on the generated split files.
  - Supports `--selection-metric macro_f1`.
  - Supports corrected ordinal/QWK checkpoint selection via `--selection-metric qwk`.
  - Supports training resampling via `--train-resample-prior none|eval_prior|balanced`.
  - Supports ordinal-aware auxiliary loss via `--ordinal-loss-weight`.
  - Supports `--torch-num-threads` to avoid excessive CPU use.
- `src/ccac/metrics.py`
  - Fixed ordinal severity mapping for QWK. Competition IDs are categorical:
    `0=中度, 1=正常, 2=轻度, 3=重度, 4=非常严重`.
  - True severity ranks are:
    `1=正常 < 2=轻度 < 0=中度 < 3=重度 < 4=非常严重`.

## Baseline Reference

Previous eval-prior split training used `robust_score` checkpoint selection:

| Output Dir | Selection | Macro-F1 | Macro-AUC | Robust | Accuracy |
|---|---|---:|---:|---:|---:|
| `artifacts/exp/no_dass_eval_prior_splits` | `robust_score` | 0.1073 | 0.5342 | 0.5342 | 0.1178 |

## Macro-F1 Selection Sweep

All runs:

- audio: `audio_wavlm_base`
- video: `video_clip_base`
- model: no-DASS DeepResidual
- split dir: `artifacts/exp/eval_prior_val_splits/exact_public_with_replacement`
- device: CUDA
- epochs: 80
- patience: 12
- batch size: 32
- `--torch-num-threads 4`
- checkpoint selection: `macro_f1`

| Run | Selection | Macro-F1 | Accuracy | Macro-AUC | Robust | Focal Gamma | Class Weight Power | Train Resample |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `mf1_base_g2_cw1` | `macro_f1` | **0.2354** | 0.5890 | 0.5232 | 0.5232 | 2.0 | 1.0 | none |
| `mf1_g1_cw1` | `macro_f1` | 0.2167 | 0.4822 | 0.5288 | 0.5288 | 1.0 | 1.0 | none |
| `mf1_resample_eval_g2_cw0` | `macro_f1` | 0.2141 | 0.3791 | 0.5534 | 0.5534 | 2.0 | 0.0 | eval_prior |
| `mf1_resample_eval_g2_cw1` | `macro_f1` | 0.2091 | 0.4450 | 0.5444 | 0.5444 | 2.0 | 1.0 | eval_prior |
| `mf1_g2_cw2` | `macro_f1` | 0.1907 | 0.7215 | 0.5064 | 0.2532 | 2.0 | 2.0 | none |
| `mf1_resample_balanced_g2_cw0` | `macro_f1` | 0.1528 | 0.2283 | 0.5401 | 0.5401 | 2.0 | 0.0 | balanced |
| `mf1_g2_cw05` | `macro_f1` | 0.1142 | 0.2660 | 0.5337 | 0.2668 | 2.0 | 0.5 | none |
| `mf1_ce_cw0` | `macro_f1` | 0.0769 | 0.0874 | 0.5189 | 0.5189 | 0.0 | 0.0 | none |
| `mf1_g2_cw0` | `macro_f1` | 0.0626 | 0.0670 | 0.5308 | 0.5308 | 2.0 | 0.0 | none |

## Post-Hoc Calibration Checks

Using the best run `mf1_base_g2_cw1`:

| Post-Hoc Method | Macro-F1 | Accuracy | Notes |
|---|---:|---:|---|
| Raw argmax | 0.2354 | 0.5890 | Best training-side result |
| Coordinate per-class threshold search | 0.2364 | 0.5890 | Tiny gain; threshold `[1.0, 1.5, 1.0, 1.0, 1.0]` |
| Prior/temperature sweep | 0.2354 | 0.5890 | No correction beat raw argmax |

## Ordinal-Aware Checks

Motivation: the public label IDs are not severity values. We tested whether using the true severity order helps:

```text
class id 1 -> rank 0  正常
class id 2 -> rank 1  轻度
class id 0 -> rank 2  中度
class id 3 -> rank 3  重度
class id 4 -> rank 4  非常严重
```

Added an auxiliary expected-rank distance term:

```text
loss = focal_loss + ordinal_loss_weight * mean(((E[rank] - true_rank) / 4)^2)
```

All runs keep the same base setting as `mf1_base_g2_cw1` unless noted.

| Run | Selection | Ordinal Loss Weight | Macro-F1 | Accuracy | Macro-AUC | Corrected QWK | Robust |
|---|---|---:|---:|---:|---:|---:|---:|
| `mf1_base_g2_cw1` | `macro_f1` | 0.00 | **0.2354** | 0.5890 | 0.5232 | 0.0520 | 0.5232 |
| `mf1_ord005_g2_cw1` | `macro_f1` | 0.05 | 0.2250 | 0.5995 | **0.5419** | 0.0541 | **0.5419** |
| `mf1_ord010_g2_cw1` | `macro_f1` | 0.10 | 0.2079 | 0.5717 | 0.5291 | 0.0301 | 0.5291 |
| `mf1_ord020_g2_cw1` | `macro_f1` | 0.20 | 0.2167 | 0.4932 | 0.5270 | 0.0729 | 0.5270 |
| `qwk_base_g2_cw1` | `qwk` | 0.00 | 0.1775 | 0.2586 | 0.5390 | **0.0762** | 0.5390 |

Interpretation:

1. Correcting the ordinal mapping is necessary for analysis, because raw class IDs do not reflect severity.
2. The light ordinal loss (`0.05`) improves Macro-AUC/robust score and slightly improves corrected QWK, but it does not beat the best Macro-F1 checkpoint.
3. Stronger ordinal loss and direct QWK checkpoint selection trade off too much leaderboard Macro-F1.
4. For the current public metric, keep `mf1_base_g2_cw1` as the submission candidate; use ordinal-aware metrics as diagnostics rather than the primary selector.

## Interpretation

1. Switching checkpoint selection from `robust_score` to `macro_f1` is the largest improvement on this eval-prior validation surface: `0.1073 -> 0.2354`.
2. The best training-side configuration remains the original focal/weight setup: `focal_gamma=2.0`, `class_weight_power=1.0`, no train resampling.
3. Training resampling to public prior helps compared with unweighted losses, but it does not beat class-weighted focal loss.
4. Balanced resampling is worse than eval-prior resampling for this public-prior validation surface.
5. Simple threshold/prior post-processing barely helps once the checkpoint is selected by Macro-F1.
6. Ordinal information is real, but direct ordinal optimization did not improve eval-prior Macro-F1 in this sweep.

## Current Best

Use:

```bash
conda run -n ccac_maps env PYTHONPATH=src python -u scripts/train_no_dass_eval_prior_splits.py \
  --dataset-path datasets \
  --split-dir artifacts/exp/eval_prior_val_splits/exact_public_with_replacement \
  --output-dir artifacts/exp/eval_prior_macro_f1_sweep/mf1_base_g2_cw1 \
  --audio-feature-name audio_wavlm_base \
  --video-feature-name video_clip_base \
  --device cuda \
  --num-folds 5 \
  --epochs 80 \
  --patience 12 \
  --batch-size 32 \
  --focal-gamma 2.0 \
  --class-weight-power 1.0 \
  --selection-metric macro_f1 \
  --train-resample-prior none \
  --torch-num-threads 4 \
  --num-workers 0
```

Best eval-prior Macro-F1 so far: **0.2354**.
