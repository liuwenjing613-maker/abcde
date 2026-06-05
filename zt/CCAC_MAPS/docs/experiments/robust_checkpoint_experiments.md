# Robust Checkpoint Experiment Log

Date: 2026-06-04

## Purpose

This log tracks experiments whose **training-time best checkpoint selection uses the new `robust_score` metric** instead of legacy `macro_f1`.

The goal is to separate two things:

- model discrimination quality, primarily `macro_auc`
- collapse avoidance, via `min_class_recall` and the `robust_score` coverage penalty

## Code Change Summary

The following training paths now select `best_model.pt` by:

```text
selection_metric = robust_score
```

Updated paths:

- `src/ccac/baselines/anxiety_baseline.py`
- `src/ccac/baselines/dass_baseline.py`
- `src/ccac/experiments/deep_residual.py`
- `src/ccac/experiments/knowledge_distillation.py`
- `scripts/train_no_dass.py`
- `scripts/exp_nd16_multiteacher.py`

Each saved checkpoint/fold metric now records:

- `selection_metric`
- `selection_score`
- `macro_auc`
- `qwk`
- `min_class_recall`
- `robust_score`

## Environment Note

The Codex sandbox used for the code edit could not see CUDA:

```text
torch.cuda.is_available() = False
torch.cuda.device_count() = 0
warning: Can't initialize NVML
```

Because of that, attempted sandbox runs fell back to CPU and were stopped/cleaned up. CPU runs are not used as experiment evidence here.

Formal runs below were executed outside the sandbox in the `ccac_maps` conda environment, where CUDA was visible:

```text
torch.cuda.is_available() = True
torch.cuda.device_count() = 1
GPU = NVIDIA GeForce RTX 4090 D
```

## Recommended Formal Runs

Run these on the normal GPU environment where CUDA is visible.

### 1. ND-3 Robust Checkpoint

Purpose: re-run the current public-score-leading learned base model with best epoch selected by `robust_score`.

```bash
PYTHONPATH=src python scripts/train_no_dass.py \
  --dataset-path datasets \
  --output-dir artifacts/exp/nd3_focal_g2_robust_ckpt \
  --audio-feature-name audio_wavlm_base \
  --video-feature-name video_clip_base \
  --device cuda \
  --num-folds 5 \
  --epochs 80 \
  --patience 12 \
  --batch-size 32 \
  --focal-gamma 2.0 \
  --class-weight-power 1.0
```

Compare against historical ND-3:

- historical output: `artifacts/exp/nd3_focal_g2`
- historical checkpoint selection: legacy Macro-F1
- historical OOF Macro-F1: about `0.264`
- historical public calibrated Macro-F1: `0.1805`

### 2. ND-11 Robust Checkpoint

Purpose: re-run the most defensible submitted model, DASS teacher to no-DASS student, with best epoch selected by `robust_score`.

```bash
PYTHONPATH=src python scripts/exp_nd11_distillation.py \
  --dataset-path datasets \
  --output-dir artifacts/exp/nd11_distillation_robust_ckpt \
  --teacher-oof artifacts/dass/focal_g1/oof_predictions.csv \
  --audio-feature-name audio_wavlm_base \
  --video-feature-name video_clip_base \
  --device cuda \
  --alpha 0.9 \
  --temperature 3.0 \
  --student-focal-gamma 2.0 \
  --num-folds 5 \
  --epochs 80 \
  --patience 12 \
  --batch-size 32
```

Compare against historical ND-11:

- historical output: `artifacts/exp/nd11_distillation`
- historical checkpoint selection: legacy Macro-F1
- historical OOF Macro-F1: about `0.329`
- historical public calibrated Macro-F1: `0.1769`

### 3. ND-16 Robust Checkpoint

Purpose: re-run the best no-DASS macro-AUC candidate with best epoch selected by `robust_score`.

```bash
PYTHONPATH=src python scripts/exp_nd16_multiteacher.py \
  --dataset-path datasets \
  --output-dir artifacts/exp/nd16_multiteacher_robust_ckpt \
  --teacher-oofs artifacts/dass/focal_g1/oof_predictions.csv artifacts/exp/transformer/oof_predictions.csv \
  --audio-feature-name audio_wavlm_base \
  --video-feature-name video_clip_base \
  --device cuda \
  --alpha 0.9 \
  --temperature 3.0 \
  --student-focal-gamma 2.0 \
  --num-folds 5 \
  --epochs 80 \
  --patience 12 \
  --batch-size 32
```

Compare against historical ND-16:

- historical output: `artifacts/exp/nd16_multiteacher`
- historical checkpoint selection: legacy Macro-F1
- historical re-evaluated macro-AUC: about `0.759`
- historical Macro-F1: about `0.308`

## Result Table

| Experiment | Output Dir | Selection Metric | Macro-F1 | Macro-AUC | QWK | Min Recall | Robust Score | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| ND-3 robust ckpt | `artifacts/exp/nd3_focal_g2_robust_ckpt` | `robust_score` | 0.2464 | 0.6059 | 0.0277 | 0.1296 | 0.6059 | Formal GPU run completed 2026-06-04 |
| ND-11 robust ckpt | `artifacts/exp/nd11_distillation_robust_ckpt` | `robust_score` | 0.2308 | 0.7451 | 0.0045 | 0.0000 | 0.3725 | Formal GPU run completed 2026-06-04 |
| ND-16 robust ckpt | `artifacts/exp/nd16_multiteacher_robust_ckpt` | `robust_score` | 0.2879 | 0.7851 | 0.0508 | 0.0167 | 0.3925 | Formal GPU run completed 2026-06-04; values from `summary.json` |

## Eval-Prior Validation Split Probe

Purpose: create validation splits that mimic the public leaderboard support, then train a no-DASS model using those splits to see whether selection under public-like class prior changes behavior.

Split generation:

```bash
python scripts/sample_eval_prior_val_splits.py \
  --labels-csv datasets/train_val/labels.csv \
  --output-dir artifacts/exp/eval_prior_val_splits/exact_public_with_replacement \
  --mode exact_public_with_replacement \
  --num-folds 5 \
  --seed 42
```

Each fold has 382 validation rows with replacement:

| Class | Count |
|---|---:|
| 中度 | 292 |
| 正常 | 15 |
| 轻度 | 46 |
| 重度 | 14 |
| 非常严重 | 15 |

Training command:

```bash
conda run -n ccac_maps env PYTHONPATH=src python -u scripts/train_no_dass_eval_prior_splits.py \
  --dataset-path datasets \
  --split-dir artifacts/exp/eval_prior_val_splits/exact_public_with_replacement \
  --output-dir artifacts/exp/no_dass_eval_prior_splits \
  --audio-feature-name audio_wavlm_base \
  --video-feature-name video_clip_base \
  --device cuda \
  --num-folds 5 \
  --epochs 80 \
  --patience 12 \
  --batch-size 32 \
  --focal-gamma 2.0 \
  --class-weight-power 1.0 \
  --torch-num-threads 4 \
  --num-workers 0
```

Result:

| Output Dir | Selection Metric | Eval Rows | Macro-F1 | Macro-AUC | QWK | Min Recall | Robust Score | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `artifacts/exp/no_dass_eval_prior_splits` | `robust_score` | 1910 | 0.1073 | 0.5342 | -0.0024 | 0.0726 | 0.5342 | Eval-prior sampled validation, with replacement |

Interpretation: training/selection on public-prior-like validation improves the validation Macro-F1 compared with collapsed raw submissions, but this is not an independent validation set because the public-prior split uses replacement and repeated subjects. Treat it as a calibration/threshold stress-test surface, not as evidence of generalization.

## Verification Performed

Code-level verification passed:

```bash
python -m py_compile \
  src/ccac/baselines/anxiety_baseline.py \
  src/ccac/baselines/dass_baseline.py \
  src/ccac/experiments/deep_residual.py \
  src/ccac/experiments/knowledge_distillation.py \
  scripts/train_no_dass.py \
  scripts/exp_nd16_multiteacher.py

PYTHONPATH=src pytest -q tests
```

Test result:

```text
3 passed
```
