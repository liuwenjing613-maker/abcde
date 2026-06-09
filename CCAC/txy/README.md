# CCAC txy — AdoDAS 迁移工作区

将 AdoDAS A1 中验证过的思想迁移到 CCAC_MAPS 赛道一（T1/T2/T3 → T4 焦虑五分类）：

- **participant-level** 聚合（`anon_school + anon_class + anon_person`）
- **school/class group split** 防泄漏
- **clip attention + stage temporal encoder**（`LabelWiseSession` → `StageWiseLongitudinal`）
- **历史 DASS tabular anchor + 多模态 residual**
- **macro-F1 导向 checkpoint + class logit bias 校准**
- **class-wise logit ensemble**

## 目录结构

```text
txy/
├── src/txy/           # 核心库
├── scripts/           # 实验入口（对应你文档里的实验 1-6）
├── configs/           # 默认配置
├── artifacts/         # 训练输出（运行后生成）
└── tests/
```

## 安装

```bash
cd /home/adodas/CCAC/txy
pip install -r requirements.txt
```

## 实验顺序

### 实验 1：HistoryTabular（历史量表强 baseline）

```bash
PYTHONPATH=src python scripts/train_history_tabular.py \
  --dataset-path /home/adodas/dataset_ccac \
  --output-dir artifacts/history_tabular \
  --model-type lightgbm
```

### 实验 2：MultimodalOnly（纯音视频）

```bash
PYTHONPATH=src python scripts/train_stagewise.py \
  --dataset-path /home/adodas/dataset_ccac \
  --output-dir artifacts/multimodal_only \
  --multimodal-only
```

### 实验 3：History + Multimodal

```bash
PYTHONPATH=src python scripts/train_stagewise.py \
  --dataset-path /home/adodas/dataset_ccac \
  --output-dir artifacts/stagewise
```

### 实验 4：Residual fusion（推荐主线）

```bash
PYTHONPATH=src python scripts/train_residual.py \
  --dataset-path /home/adodas/dataset_ccac \
  --output-dir artifacts/residual \
  --alpha 0.25
```

公式：`logits = logits_tabular + alpha * logits_multimodal`

### 实验 5：Ordinal model

```bash
PYTHONPATH=src python scripts/train_ordinal.py \
  --dataset-path /home/adodas/dataset_ccac \
  --output-dir artifacts/ordinal
```

### 实验 6：Class-wise ensemble

```bash
PYTHONPATH=src python scripts/ensemble_predictions.py \
  --inputs artifacts/history_tabular/oof_predictions.csv \
              artifacts/stagewise/oof_predictions.csv \
              artifacts/residual/oof_predictions.csv \
  --weights 0.60 0.25 0.15 \
  --output artifacts/ensemble/oof_predictions.csv
```

## 与 AdoDAS 的对应关系

| AdoDAS | CCAC txy |
|---|---|
| `GroupedParticipantDataset` | `LongitudinalPersonDataset` |
| session attention | `ClipAttentionPool` + `GRU` stage encoder |
| `MTCNBackbone` + adapters | `FeatureAdapter` + `GatedFusion` |
| D/A/S BCE | 5-class CE / ordinal BCE |
| `0.5` threshold + bias | `search_class_bias` on val |
| label-wise transplant | `class_wise_blend` |
| `manifests_internal/split_*` | `build_group_folds(group_by=school_class)` |

## 输出文件

每次训练会生成：

- `fold_metrics.csv` — 每折 macro-F1 / balanced accuracy
- `oof_predictions.csv` — person-level OOF logits
- `test_predictions.csv` — 提交格式：`anon_school, anon_class, anon_person, label`
- `best_macro_f1.pt` — 按 macro-F1 保存的 checkpoint

## 测试

```bash
PYTHONPATH=src pytest -q tests
```
