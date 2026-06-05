# CCAC_MAPS Current Project Situation

Date: 2026-06-02  
Project path: `/home/zt/Desktop/emotion_analysis/CCAC_MAPS`

## One-Line Status

The project has moved from baseline reproduction into distribution-shift diagnosis and no-DASS model development. The main lesson is clear: DASS history is powerful offline but unavailable in the real test file, so valid submissions must be audio-video only or use DASS only as a teacher for distillation.

## Repository Snapshot

- Core package: `src/ccac/`
  - `baselines/anxiety_baseline.py`: official AV baseline, now extended with robust metric hooks.
  - `baselines/dass_baseline.py`: DASS/focal infrastructure still used by older teacher/offline experiments.
  - `experiments/`: deep residual, transformer, TCN, SWA, LDAM, knowledge distillation, feature experiments.
  - `metrics.py`: new metric framework with macro-AUC, QWK, min class recall, robust score.
- Experiment scripts: `scripts/`
  - Important valid no-DASS scripts: `exp_nd11_distillation.py`, `exp_nd16_multiteacher.py`, `exp_nd15_distill_transformer.py`, `exp_nd13_tcn.py`, `exp_final.py`, `train_no_dass.py`.
  - `run_final.sh` currently runs `scripts/exp_basic_features.py` into `artifacts/exp/final_wider` with no DASS, `audio_wavlm_base + video_clip_base`, wider hidden dim 320, 8 heads, 4 residual blocks, and basic features enabled.
- Main logs:
  - `docs/experiments/submission_log.md`: public submission diagnosis.
  - `docs/experiments/no_dass_experiments.md`: valid no-DASS experiment history.
  - `docs/archive/experiment_log.md`: older DASS-heavy exploration, useful but partially invalid for test because DASS is not available.
  - `docs/metric_analysis.md`: why Macro-F1 is misleading here and why macro-AUC/coverage are needed.
- Saved artifacts:
  - Baselines: `artifacts/baselines/`
  - DASS/offline teacher runs: `artifacts/dass/`
  - Main experiments: `artifacts/exp/`
  - Feature sweeps: `artifacts/sweep/`
  - Submitted zips: `submissions_v3/`

## Data Reality

Training data has 1527 subjects with full labels and DASS history. Test has 1909 subjects, but `test/subjects.csv` contains only identifiers; it does not contain T1-T3 DASS scores/levels.

This makes all direct DASS-input models invalid for real test inference. They can be used as offline teachers, but not as final students unless the student takes no DASS at inference.

The public leaderboard subset appears to be 382 subjects. From CodaBench support counts, its class prior is almost inverted relative to training:

| Class | Train Count / Share | Public Count / Share |
|---|---:|---:|
| 正常 | 1169 / 76.5% | 15 / 3.9% |
| 中度 | 186 / 12.2% | 292 / 76.4% |
| 轻度 | 58 / 3.8% | 46 / 12.0% |
| 重度 | 54 / 3.5% | 14 / 3.7% |
| 非常严重 | 60 / 3.9% | 15 / 3.9% |

This inversion dominates public Macro-F1. A model that learns the train majority class, `正常`, can look decent in OOF and fail publicly.

## Public Submission Situation

Recorded public submissions:

| File | Method | Public Macro-F1 | Diagnosis |
|---|---|---:|---|
| `sub_nd3_calibrated.zip` | DeepResidual gamma=2.0 + test prior calibration | 0.1805 | Best public score, but mostly a prior/distribution bet. |
| `sub_nd11_distillation_calibrated.zip` | DASS teacher -> no-DASS student + prior calibration | 0.1769 | Best real learned model among submitted runs; calibration helps. |
| `sub1_deep_residual.zip` | Direct DASS model | 0.0908 | Invalid test condition; DASS missing, model collapses. |
| `sub_baseline_nodass.zip` | Official AV BiGRU baseline | 0.0486 | Learns train majority `正常`. |
| `sub_nd11_distillation_raw.zip` | KD without calibration | 0.0328 | Predicts train prior; calibration is essential for public subset. |

Important interpretation: public Macro-F1 currently rewards matching the public prior more than robust discrimination. The best public score is not necessarily the best model.

## Offline Experiment Situation

### Invalid For Direct Test But Useful As Teachers

- Direct DASS models reached about 0.36 OOF Macro-F1.
- Best saved DASS/offline-style artifacts include:
  - `artifacts/exp/basic_features`: OOF Macro-F1 0.3649.
  - `artifacts/exp/deep_residual`: OOF Macro-F1 0.3628.
  - `artifacts/exp/final`: OOF Macro-F1 0.3623.
- These runs prove DASS is highly informative, but direct use at test is not allowed by the actual test feature file.

### Valid No-DASS Line

The valid path is no-DASS inference. Main progression:

| Experiment | Method | OOF Macro-F1 | Notes |
|---|---:|---:|---|
| ND-1 | Official BiGRU AV baseline | 0.232 | Strongly biased to `正常`. |
| ND-3 | DeepResidual + Focal gamma=2.0 | 0.264 | Better minority coverage; public calibrated score 0.1805. |
| ND-9 | Transformer + Focal gamma=2.0 | about 0.274 | Slight OOF gain over ND-3. |
| ND-12 | DeepResidual + Transformer + BiGRU ensemble | about 0.287 | Ensemble helps but does not beat distillation. |
| ND-11 | Single-teacher KD, DASS teacher -> no-DASS DeepResidual student | 0.3293 | Best no-DASS OOF Macro-F1 and best learned submitted model. |
| ND-15 | Transformer student KD | about 0.310 | Worse than DeepResidual student. |
| ND-16 | Multi-teacher KD | 0.3076 legacy OOF MF1 | Worse by Macro-F1, but best by macro-AUC. |

## Metric Situation

`docs/metric_analysis.md` and `src/ccac/metrics.py` introduce a better evaluation frame:

- `macro_auc`: threshold-independent discrimination, primary for offline model quality.
- `qwk`: ordinal quality for anxiety levels.
- `min_class_recall`: collapse detection.
- `robust_score`: macro-AUC with a coverage penalty.

Historical reevaluation in `artifacts/exp/metrics_reevaluation.json` shows:

| Rank By AUC | Experiment | Macro-F1 | Macro-AUC | QWK | Min Recall |
|---:|---|---:|---:|---:|---:|
| 1 | `nd16_multiteacher` | 0.298 | 0.759 | 0.056 | 0.019 |
| 2 | `basic_features` | 0.358 | 0.753 | 0.225 | 0.167 |
| 3 | `final` | 0.359 | 0.751 | 0.203 | 0.259 |
| 4 | `nd11_distillation` | 0.306 | 0.745 | 0.077 | 0.019 |
| 5 | `nd15_distill_transformer` | 0.295 | 0.739 | 0.045 | 0.019 |

Key nuance: DASS experiments naturally rank high in offline AUC but are not test-valid. Among no-DASS candidates, ND-16 has the best AUC, while ND-11 has better legacy Macro-F1 and an actual public calibrated submission.

## Current Best Choices

For leaderboard Macro-F1 under the known public prior:

1. `sub_nd3_calibrated.zip` is the best recorded public result at 0.1805, but it is mostly a distribution-prior strategy.
2. `sub_nd11_distillation_calibrated.zip` is nearly tied at 0.1769 and is more defensible because it comes from a no-DASS distilled student.

For model quality under valid test features:

1. ND-11 is the safest current candidate: DASS teacher -> no-DASS DeepResidual student, `audio_wavlm_base + video_clip_base`, alpha 0.9, temperature 3.0, student focal gamma 2.0.
2. ND-16 is worth revisiting because its macro-AUC is best among reevaluated experiments, even though its argmax Macro-F1 is lower.
3. Any future work should separate discrimination quality from calibration/public-prior betting.

## What Is Probably Wrong Or Risky

- Do not submit direct DASS-input models as final solutions. Test does not provide DASS.
- Do not trust OOF Macro-F1 alone. It is trained/evaluated under the train distribution, which is opposite to public distribution.
- Do not over-interpret the best public score. Public subset prior leakage can make a weak discriminator score well.
- Be careful with old docs. `docs/archive/experiment_log.md` and `docs/archive/baseline_improvement_notes.md` include DASS-first recommendations that were later invalidated by the test file structure.
- The working tree is dirty and contains untracked/modified files. Avoid reverting unrelated changes.

## Suggested Next Step

The most sensible next experiment is a calibration-focused pass over ND-11 and ND-16:

1. Keep inference no-DASS.
2. Compare ND-11 vs ND-16 using macro-AUC, QWK, and per-class recalls.
3. Tune only decision/calibration rules on OOF, then separately prepare public-prior-calibrated submission variants.
4. If submitting, prefer naming zips clearly as either `raw`, `oof_calibrated`, or `public_prior_calibrated` so the experiment log stays interpretable.

## Commands Of Interest

Reevaluate historical experiments:

```bash
PYTHONPATH=src python scripts/reevaluate_experiments.py
```

Train ND-11:

```bash
PYTHONPATH=src python scripts/exp_nd11_distillation.py \
  --dataset-path datasets \
  --output-dir artifacts/exp/nd11_distillation \
  --teacher-oof artifacts/dass/focal_g1/oof_predictions.csv \
  --audio-feature-name audio_wavlm_base \
  --video-feature-name video_clip_base \
  --device cuda
```

Current `run_final.sh` path:

```bash
bash run_final.sh
```
