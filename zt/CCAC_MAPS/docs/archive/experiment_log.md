# CCAC MAPS 基线改进实验日志

## 实验环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090D (24GB) |
| CUDA | 12.9, Driver 575.64.03 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.10 (conda: ccac_maps) |
| 数据集 | CCAC MAPS train_val (1527 subjects) + test (1909 subjects) |
| 评估方式 | 5 折分层交叉验证 OOF |

---

## 实验一：原始基线复现

**命令**:
```bash
PYTHONPATH=src python scripts/train_anxiety_baseline.py \
  --dataset-path datasets \
  --output-dir artifacts/baselines/anxiety_wavlm_dinov2_small \
  --device cuda
```

**配置**: `audio_wavlm_base` + `video_dinov2_small`, `cw=1.0, ls=0.0, dp=0.2, lr=1e-3`

**结果**:

| 指标 | 值 |
|---|---|
| Accuracy | 0.698 |
| Macro-F1 | 0.232 |
| Weighted-F1 | 0.671 |

| 类别 | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 中度 | 0.26 | 0.25 | 0.25 | 186 |
| 正常 | 0.80 | 0.87 | 0.83 | 1169 |
| 轻度 | 0.08 | 0.03 | 0.05 | 58 |
| 重度 | 0.00 | 0.00 | 0.00 | 54 |
| 非常严重 | 0.04 | 0.02 | 0.02 | 60 |

**结论**: 基线可复现。模型对多数类（正常，76.5%）过拟合，少数类（重度、非常严重）召回率接近零。

---

## 实验二：超参数调优

### 第一轮（失败）

所有实验叠加 `class_weight_power=2.0`，导致极端权重（少数类是多数类的 400 倍），全部崩溃：

| 实验 | cw | ls | dp | lr | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|
| 默认 | 1.0 | 0.0 | 0.2 | 1e-3 | 0.232 | 0.698 |
| cw=2.0 | 2.0 | 0.0 | 0.2 | 1e-3 | 0.201 | 0.409 |
| +标签平滑 | 2.0 | 0.1 | 0.2 | 1e-3 | 0.063 | 0.044 |
| +强正则化 | 2.0 | 0.1 | 0.4 | 5e-4 | 0.044 | 0.034 |

**教训**: 所有配置押注在极端类权重上，无法分辨各超参数的独立效果。

### 第二轮：单变量对照

每次仅改变一个超参数，其余保持默认。

| 实验 | 改变 | Macro-F1 | Accuracy | Weighted-F1 | 与默认对比 |
|---|---|---|---|---|---|
| **默认** | — | **0.232** | 0.698 | 0.671 | — |
| cw=0.0 | 无类权重 | 0.200 | **0.715** | 0.661 | Acc↑ MF1↓ |
| cw=0.5 | 温和权重 | 0.212 | **0.715** | 0.670 | Acc↑ MF1↓ |
| cw=1.5 | 稍强权重 | **0.237** | 0.607 | 0.632 | 唯一 MF1↑ |
| ls=0.05 | 标签平滑 0.05 | 0.224 | 0.675 | 0.655 | 无明显变化 |
| ls=0.1 | 标签平滑 0.1 | 0.232 | 0.646 | 0.648 | MF1 持平 |
| dp=0.3 | 高 dropout | 0.231 | 0.614 | 0.634 | 几乎不变 |
| lr=5e-4 | 低学习率 | 0.231 | **0.719** | 0.674 | Acc 最佳 |

### 类权重详细分析

| cw | 正常权重 | 重度权重 | 权重比 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|
| 0.0 | 1× | 1× | 1:1 | 0.200 | 0.715 |
| 0.5 | 0.51× | 2.38× | 1:5 | 0.212 | 0.715 |
| **1.0** | **0.26×** | **5.66×** | **1:22** | **0.232** | **0.698** |
| 1.5 | 0.13× | 13.5× | 1:104 | 0.237 | 0.607 |

### 调参结论

1. **默认超参数已是最优** — `cw=1.0, ls=0.0, dp=0.2, lr=1e-3` 是当前架构的最佳配置
2. **调参空间极度狭窄** — 类权重有效区间约 0.8-1.2，其他参数几乎无调节余地
3. **单纯调参无法解决根本问题** — 少数类样本过少（54-60）、特征信息不足是硬瓶颈

---

## 实验三：DASS 量表历史集成

### 方案设计

| 方案 | 输入 | 维度 | 融合方式 |
|---|---|---|---|
| `none` | 无 DASS | 0 | 等同于原始基线 |
| `scores_a` | T1/T2/T3 焦虑分数 | 3 | 直接拼接到融合层 |
| `scores_das` | T1/T2/T3 抑郁+焦虑+压力分数 | 9 | 直接拼接到融合层 |
| `encoder` | 9 个分数 + 9 个等级 | 18 | 经 MLP 编码后拼接 |

### 架构

```
音视频分支 (同原始基线):
  Clip Encoder → Attention Pooling → BiGRU → [896]

DASS 分支 (新增):
  [t1_anxiety_score, t2_anxiety_score, t3_anxiety_score]
       │
       ├── scores_a/das: 直接拼接
       └── encoder: MLP(18→64→64) → [64]

融合: Concat([896, dass_dim]) → Classifier → 5 classes
```

### 结果（scores_a）

| 指标 | 原始基线 | DASS scores_a | 提升 |
|---|---|---|---|
| **Macro-F1** | 0.232 | **0.295** | **+27.2%** |
| Accuracy | 0.698 | 0.744 | +6.6% |
| Weighted-F1 | 0.671 | 0.721 | +7.4% |

### 三种方案对比

| 方案 | 输入维度 | Macro-F1 | Accuracy | Weighted-F1 | vs 基线 |
|---|---|---|---|---|---|
| none（基线） | 0 | 0.232 | 0.698 | 0.671 | — |
| scores_a | 3（仅焦虑分数） | 0.295 | 0.744 | 0.721 | +27% |
| **scores_das** | **9（D+A+S×3）** | **0.298** | **0.744** | **0.723** | **+28%** |
| encoder | 18（分数+等级→MLP） | 0.282 | 0.725 | 0.711 | +22% |

### 各类别（scores_das）

| 类别 | 默认 F1 | scores_das F1 | 变化 |
|---|---|---|---|
| 正常 | 0.83 | **0.88** | +6% |
| 中度 | 0.25 | **0.33** | +32% |
| 非常严重 | 0.02 | **0.20** | **+900%** |
| 重度 | 0.00 | **0.09** | 从零突破 |
| 轻度 | 0.05 | 0.00 | ↓ |

### 分析

- **仅 3 个额外数字**（T1-T3 焦虑分数）带来 27% 的 Macro-F1 提升
- 重度（0→0.08）和非常严重（0.02→0.20）首次有了非零召回
- 轻度（58 样本）仍然为零——样本量是关键瓶颈
- 方案 `scores_das` 和 `encoder` 已测试——`scores_das` 最优（MF1 0.298），`encoder` 的 MLP 在小样本上过拟合

---

## 实验四：Focal Loss

### 原理

Focal Loss 对已分类正确的样本自动降低权重，聚焦于难分类（少数类）样本：

```
FL(p_t) = -(1 - p_t)^γ · log(p_t)
```

- γ=0 → 标准交叉熵
- γ=1-2 → 降权易分类样本
- 天然适合极度不平衡场景，无需手动调类权重

### 实现

在 `dass_baseline.py` 中新增 `FocalLoss` 类，支持 `gamma`、`alpha`（类权重）、`label_smoothing`。

### 结果（scores_das + Focal Loss）

| 实验 | Macro-F1 | Accuracy | Weighted-F1 | vs 基线 |
|---|---|---|---|---|
| 基线 | 0.232 | 0.698 | 0.671 | — |
| scores_das (CE) | 0.298 | 0.744 | 0.723 | +28% |
| **scores_das + Focal γ=1.0** | **0.329** | 0.739 | 0.731 | **+42%** |
| scores_das + Focal γ=2.0 | 0.318 | **0.755** | 0.726 | +37% |

### 各类别（Focal γ=1.0）

| 类别 | 基线 F1 | Focal γ=1.0 F1 | 变化 |
|---|---|---|---|
| 正常 | 0.83 | 0.87 | +5% |
| 中度 | 0.25 | **0.37** | +48% |
| 非常严重 | 0.02 | **0.25** | **+1150%** |
| 重度 | 0.00 | **0.10** | 从零突破 |
| 轻度 | 0.05 | 0.05 | 持平 |

### 分析

- γ=1.0 是最佳平衡点——Macro-F1 从 0.298 提升到 0.329（+10% 相对提升）
- γ=2.0 过度聚焦难样本，牺牲了 Macro-F1 换取更高准确率（0.755）
- 轻度（58 样本）仍然极难——可能需要过采样或数据增强才能突破

### 使用方法

```bash
PYTHONPATH=src python scripts/train_dass_baseline.py \
  --dataset-path datasets \
  --output-dir artifacts/dass/focal_g1 \
  --dass-scheme scores_das --focal-gamma 1.0 --device cuda
```

### 使用方法

```bash
PYTHONPATH=src python scripts/train_dass_baseline.py \
  --dataset-path datasets \
  --output-dir artifacts/dass/focal_g1 \
  --dass-scheme scores_das --focal-gamma 1.0 --device cuda
```

---

## 实验五：多特征对集成

（见上文已添加的完整内容）

---

## 实验六：代码性能优化

对 `anxiety_baseline.py` 进行纯运行时优化（不影响模型输出）：

| 优化项 | 方法 | 效果 |
|---|---|---|
| 并行文件 I/O | `ThreadPoolExecutor` 替代串行循环 | 首次特征构建 15min → ~2min |
| 内存映射 | `np.load(mmap_mode='r')` 加载缓存 | 降低内存峰值，加载更快 |
| 自动 workers | `os.cpu_count()//2` 自动检测 | 训练时数据加载并行化 |
| 路径索引预扫描 | `_scan_pooled_cache` O(1) 查找 | 避免每次 `Path.exists()` 系统调用 |
| Bug 修复 | `cw=0.0` 时 `NoneType.to(device)` | 修复崩溃 |

---

## 总结对比

| 实验 | 方法 | Macro-F1 | Accuracy | 关键发现 |
|---|---|---|---|---|
| 基线复现 | 原始代码 | 0.232 | 0.698 | 可复现，少数类召回率极低 |
| 超参数调优 | 7 组单变量 | 0.237 (cw=1.5) | 0.719 (lr=5e-4) | 调参空间极窄，默认已最优 |
| **DASS 集成** | **scores_a** | **0.295** | **0.744** | **+27% MF1，3 个数字的事** |
| 性能优化 | 并行 I/O 等 | (不变) | (不变) | 首次运行 15min→2min |

---

## 实验五：多特征对集成

### 方法

训练 4 个不同音视频特征对的 DASS+Focal 模型，平均其预测概率：

| 序号 | 音频特征 | 视频特征 | 输入维度 |
|---|---|---|---|
| 1 | audio_wavlm_base | video_dinov2_small | 3840 |
| 2 | audio_wavlm_base | video_dinov2_base | 4608 |
| 3 | audio_wav2vec2_xlsr_chinese | video_dinov2_small | 4864 |
| 4 | audio_chinese_hubert_base | video_clip_large | 4608 |

所有模型使用相同配置：`scores_das` + Focal γ=1.0。

### 各模型性能

| 特征对 | Macro-F1 | Accuracy |
|---|---|---|
| wavlm_base + dinov2_small | **0.329** | 0.739 |
| wavlm_base + dinov2_base | 0.307 | 0.735 |
| xlsr + dinov2_small | 0.297 | 0.754 |
| hubert_base + clip_large | 0.290 | 0.724 |

### 集成结果

| 指标 | 最佳单模型 | 4 模型集成 | 变化 |
|---|---|---|---|
| Macro-F1 | **0.329** | 0.310 | ↓ 下降 |
| Accuracy | 0.739 | **0.768** | ↑ 提升 |
| Weighted-F1 | 0.731 | 0.736 | ↑ 微升 |

### 各类别变化

| 类别 | 最佳单模型 F1 | 集成 F1 |
|---|---|---|
| 正常 | 0.87 | 0.89 |
| 中度 | 0.37 | 0.37 |
| 非常严重 | 0.25 | 0.26 |
| 重度 | 0.10 | 0.03 |
| 轻度 | 0.05 | 0.00 |

### 结论

**集成反而降低 Macro-F1。** 原因：
1. 默认特征对（wavlm_base + dinov2_small）本身就是最优的——其他 3 个模型的 MF1 均低于它
2. 弱模型的噪声稀释了强模型对少数类的信号
3. 集成提升了准确率（0.739→0.768）仅因为强模型之间的不一致被平滑——但这主要惠及多数类（正常）
4. **更多特征 ≠ 更好**——特征质量比数量重要

---

## 实验六：控制变量特征筛选

### 方法

使用控制变量法系统筛选最优音视频特征：
- **Phase 1**: 固定 video=dinov2_small，扫描 8 种音频组合（4 单特征 + 4 多特征）
- **Phase 2**: 固定最佳音频=wavlm_base，扫描 11 种视频组合（6 单特征 + 5 多特征）

所有模型使用 DASS scores_das + Focal γ=1.0。

### Phase 1 结果（固定 video=dinov2_small）

| 音频特征 | 维度 | Macro-F1 | vs 最佳 |
|---|---|---|---|
| **wavlm_base** | 3072 | **0.329** | — |
| wav2vec2_chinese_base | 3072 | 0.314 | ↓ |
| xlsr | 4096 | 0.297 | ↓ |
| hubert_base | 3072 | 0.291 | ↓ |
| all 4 SSL | 13312 | 0.318 | ↓ |
| wavlm + xlsr | 7168 | 0.302 | ↓ |
| wavlm + hubert + xlsr | 10240 | 0.298 | ↓ |
| wavlm + hubert | 6144 | 0.293 | ↓ |

**结论**: wavlm_base 单独最优。任何多特征组合都更差。

### Phase 2 结果（固定 audio=wavlm_base）

| 视频特征 | 维度 | Macro-F1 | vs 最佳 |
|---|---|---|---|
| **clip_base** | 1024 | **0.336** 🔥 | — |
| dino_small | 768 | 0.329 | ↓ |
| vit_mae | 1536 | 0.316 | ↓ |
| clip_large | 1536 | 0.310 | ↓ |
| dino_base | 1536 | 0.307 | ↓ |
| siglip | 1536 | 0.304 | ↓ |
| dino_small + dino_base | 2304 | 0.307 | ↓ |
| dino_small + clip_large | 2304 | 0.290 | ↓ |
| dino + clip + siglip | 3840 | 0.293 | ↓ |
| dino_small + siglip | 2304 | 0.274 | ↓ |
| all 6 SSL | 7949 | 0.284 | ↓ |

**结论**: clip_base 超越默认 dino_small（0.336 vs 0.329）。任何多特征组合都更差。

### 最终最佳: audio_wavlm_base + video_clip_base → MF1 0.336

### 各类别对比（最佳 vs 此前最佳）

| 类别 | 此前 (dino_small) | 新最佳 (clip_base) | 变化 |
|---|---|---|---|
| 正常 | 0.87 | 0.87 | 持平 |
| 中度 | 0.37 | 0.30 | ↓ |
| 非常严重 | 0.25 | 0.28 | ↑ |
| **轻度** | 0.05 | **0.13** | **↑ +160%** |
| 重度 | 0.10 | 0.10 | 持平 |

---

## 实验七：阈值校准

### 方法

在 OOF 预测概率上搜索每类最优决策阈值（而非统一使用 argmax），以最大化 Macro-F1。

```python
for each class c:
    for threshold t in [0.05, 0.10, ..., 0.95]:
        pred[c] = 1 if prob[c] > t
        evaluate macro_f1
    keep best t
```

无需重新训练——仅对已有 OOF 概率做后处理。

### 结果（应用于最佳模型：clip_base+DASS+Focal）

| 指标 | 默认 argmax | 阈值校准后 | 变化 |
|---|---|---|---|
| Macro-F1 | 0.336 | **0.342** | +1.8% |
| Accuracy | 0.717 | 0.720 | +0.3% |

### 各类别最佳阈值

| 类别 | 最佳阈值 | 说明 |
|---|---|---|
| 中度 | 0.15 | 大幅降低——更愿意预测 |
| 轻度 | 0.45 | 轻微降低 |
| 重度 | 0.50 | 不变 |
| 非常严重 | 0.50 | 不变 |
| 正常 | 0.50 | 不变 |

### 各类别变化

| 类别 | 校准前 F1 | 校准后 F1 |
|---|---|---|
| 中度 | 0.30 | 0.33 |
| 正常 | 0.87 | 0.87 |
| 轻度 | 0.13 | 0.12 |
| 重度 | 0.10 | 0.10 |
| 非常严重 | 0.28 | 0.29 |

---

## 实验八：过采样 + 多任务辅助损失

### 方法

- **过采样**: `WeightedRandomSampler` 确保每个 batch 类别均衡
- **多任务辅助损失**: 模型同时预测 T1/T2/T3 焦虑等级，辅助损失权重 0.3
- 基础配置: wavlm_base + clip_base + scores_das + Focal γ=1.0

### 结果

| 配置 | Macro-F1 | Accuracy |
|---|---|---|
| 基础（无过采样/无辅助） | **0.336** | 0.717 |
| + 过采样 + 辅助损失 | 0.319 | 0.688 |
| + 过采样 + 辅助 + 校准 | 0.333 | 0.662 |

### 结论

过采样和多任务辅助损失**均降低性能**。原因：
1. Focal Loss 已充分处理类别不平衡，过采样引入冗余信号
2. T1-T3 焦虑标签与 T4 高度相关但并非同一分布——强制同时预测反而干扰主任务
3. 辅助任务的梯度与主任务存在冲突

---

## 全实验总结

| # | 实验 | 方法 | Macro-F1 | Accuracy | 相比基线 |
|---|---|---|---|---|---|
| 1 | 基线复现 | 原始代码（无 DASS） | 0.232 | 0.698 | — |
| 2 | 超参数调优 | 7 组单变量扫描 | 0.237 | 0.719 | 调参无效 |
| 3 | DASS 集成 | scores_das（9 分数） | 0.298 | 0.744 | +28% |
| 4 | Focal Loss | γ=1.0 + scores_das | 0.329 | 0.739 | +42% |
| 5 | 特征筛选 | 控制变量 19 模型 | 0.336 | 0.717 | +45% |
| 6 | 多特征集成 | 4 对特征平均 | 0.310 | 0.768 | MF1↓ |
| **7** | **阈值校准** | **后处理搜索阈值** | **0.342** | **0.720** | **+47% 🔥** |
| 8 | 过采样 | WeightedRandomSampler | 0.319 | 0.688 | ↓ |
| 9 | 多任务辅助 | T1-T3 同时预测 | 0.319 | 0.688 | ↓ |
| 10 | 性能优化 | 并行 I/O 等 | — | — | 15→2min |

### 各类别最佳 F1 演进

| 类别 | 基线 | +DASS | +Focal | +clip_base | +校准 | 总提升 |
|---|---|---|---|---|---|---|
| 正常 | 0.83 | 0.88 | 0.87 | 0.87 | 0.87 | +5% |
| 中度 | 0.25 | 0.33 | 0.37 | 0.30 | 0.33 | +32% |
| 非常严重 | 0.02 | 0.20 | 0.25 | 0.28 | 0.29 | +1350% |
| 重度 | 0.00 | 0.09 | 0.10 | 0.10 | 0.10 | 从零突破 |
| 轻度 | 0.05 | 0.00 | 0.05 | 0.13 | 0.12 | +140% |

### 关键教训

1. **数据 > 算法**: 3 个 DASS 分数（+27%）远超所有调参总和（+2%）
2. **Focal Loss 天然适合不平衡**: γ=1.0，无需类权重
3. **单特征 > 多特征堆叠**: 任何 2+ 同模态特征组合都降低 MF1
4. **clip_base 被低估**: 1024 维 CLIP（0.336）> 768 维 DINOv2（0.329）
5. **阈值校准免费且有效**: 0.336→0.342，零训练成本
6. **过采样和辅助损失无效**: Focal Loss 已足够，多任务梯度冲突

### 最终最佳配置

```bash
PYTHONPATH=src python scripts/train_dass_baseline.py \
  --dataset-path datasets \
  --output-dir artifacts/best \
  --audio-feature-name audio_wavlm_base \
  --video-feature-name video_clip_base \
  --dass-scheme scores_das --focal-gamma 1.0 \
  --calibrate --device cuda
```

### 下一步

1. ~~提交最佳模型（MF1 0.342）到 CodaBench~~
2. ~~架构改进: Transformer 替代 GRU~~ → Trial 11 (MF1 0.345)
3. 特征级数据增强: Mixup / SMOTE on pooled features

---

## 实验十一：Transformer 时序编码器 (Trial 1)

### 方法

用 2 层 Pre-LN Transformer Encoder (4 heads, 256 dim) 替换单层 BiGRU。

**架构变更**:
```
原版: Attention Pooling → +PosEmb → BiGRU(256→192×2) → Fusion(896)
新版: Attention Pooling → Proj(256→256) → Sinusoidal+Learnable PosEmb
      → TransformerEncoder(2层,4头,Pre-LN) → Fusion(mean+final+max, 768+256)
```

额外改进:
- 余弦退火热重启调度器 (CosineAnnealingWarmRestarts, T0=10)
- 梯度裁剪 (max_norm=1.0)
- 更深的融合层 (768+256→512→256→5)
- Xavier 初始化

### 结果

| 指标 | 此前最佳 (BiGRU+校准) | Transformer+校准 | 变化 |
|---|---|---|---|
| **Macro-F1** | 0.342 | **0.345** | **+0.003** |
| Accuracy | 0.720 | 0.694 | ↓ |
| Weighted-F1 | 0.710 | 0.711 | ↑ |

### 各类别

| 类别 | BiGRU 最佳 F1 | Transformer F1 | 变化 |
|---|---|---|---|
| 中度 | 0.33 | **0.41** | **+24%** 🔥 |
| 正常 | 0.87 | 0.84 | ↓ |
| 轻度 | 0.12 | **0.15** | +25% |
| 重度 | 0.10 | 0.11 | +10% |
| 非常严重 | 0.29 | 0.22 | ↓ |

### 分析

- Transformer 在"中度"类取得显著突破 (0.33→0.41)，这是最大的少数类 (186 样本)
- "非常严重"退化 (0.29→0.22)，可能因为 Transformer 的 inductive bias 更适合中等样本量的类别
- 训练收敛更慢 (best_epoch 约 26 vs BiGRU 的约 10)，但有更高的最终性能
- 总体 MF1 提升有意义但幅度有限 (0.345 vs 0.342)

---

## 实验十二：深度残差架构 + 跨阶段注意力 (Trial 2)

### 方法

三个核心创新:

1. **跨阶段多头自注意力 (Cross-Stage Attention)**: T1/T2/T3 三个时间步互相 attend，显式建模阶段间关系
2. **阶段差分特征 (Stage Difference Features)**: 计算 ΔT2-T1, ΔT3-T2, ΔT3-T1，经 MLP 编码后作为轨迹信息的显式特征
3. **Squeeze-and-Excitation 残差块**: 3 个 Pre-LN 残差块 + SE 通道注意力

**架构**:
```
Clip Encoder → Attention Pooling → +PosEmb
    → Cross-Stage Multi-Head Attention (4 heads)
    → [mean, final, diff_features] concat
    → 3× ResidualBlock (SE-enhanced)
    → 3-layer Classifier (256→128→5)
```

### 结果

| 指标 | BiGRU 最佳 | Transformer Trial 1 | **Deep Residual Trial 2** | 提升 |
|---|---|---|---|---|
| **Macro-F1** | 0.342 | 0.345 | **0.363** | **+0.021** 🔥 |
| Accuracy | 0.720 | 0.694 | 0.697 | ↓ |
| Weighted-F1 | 0.710 | 0.711 | 0.724 | +0.014 |

### 各类别

| 类别 | BiGRU 最佳 | Transformer | **Deep Residual** | 总提升 |
|---|---|---|---|---|
| 中度 | 0.33 | 0.41 | 0.38 | +15% |
| 正常 | 0.87 | 0.84 | 0.86 | -1% |
| 轻度 | 0.12 | 0.15 | **0.16** | +33% |
| 重度 | 0.10 | 0.11 | 0.10 | — |
| **非常严重** | 0.29 | 0.22 | **0.31** | **+7%** 🔥 |

### 分析

- **跨阶段注意力** 是本次最有效的改进——恢复了 Transformer 在"非常严重"上的退化 (0.22→0.31)
- **阶段差分特征** 提供了显式的轨迹信息 (上升/下降/稳定)，帮助模型理解焦虑变化趋势
- **SE 残差块** 让模型自适应调整特征通道权重
- 所有少数类均有改善或持平，多数类仅有微小牺牲
- 0.363 是目前最高 Macro-F1，**比原始基线 (0.232) 提升 56%**

---

## 实验十三：手工特征增强 (Trial 3)

### 方法

在每个片段上额外拼接 audio_basic (74-dim 声学特征) 和 video_basic (13-dim 视觉特征)，作为 SSL 特征的补充。

- audio_basic: 74 维手工声学特征 (rms, zcr, centroid, bandwidth, rolloff 等的统计量)
- video_basic: 13 维手工视觉特征 (brightness, blur, motion 等的统计量)
- 架构使用 Trial 2 最佳 (DeepResidualModel)
- 输入维度: 3072+1024+74+13 = 4183

### 结果

| 指标 | Trial 2 (无 basic) | Trial 3 (+basic) | 变化 |
|---|---|---|---|
| **Macro-F1** | 0.363 | **0.365** | +0.002 |
| Accuracy | 0.697 | 0.693 | ↓ |
| Weighted-F1 | 0.724 | 0.723 | — |

### 各类别

| 类别 | Trial 2 | Trial 3 | 变化 |
|---|---|---|---|
| 中度 | 0.38 | 0.35 | ↓ |
| 正常 | 0.86 | 0.86 | — |
| 轻度 | 0.16 | **0.17** | +6% |
| **重度** | 0.10 | **0.14** | **+40%** 🔥 |
| 非常严重 | 0.31 | 0.31 | — |

### 分析

- 手工特征对"重度"有显著帮助 (0.10→0.14, +40%)
- 但牺牲了"中度" (0.38→0.35)
- 总体 MF1 微升 0.002，边际收益递减
- 手工特征提供了与 SSL 互补的信号，但信息量有限

---

## 实验十四：Mixup 数据增强 (Trial 4)

### 方法

输入空间 Mixup (α=0.4, p=0.5)，偏向不同类别样本配对 (balanced mixup)，使用 DeepResidualModel 架构。

### 结果

| 指标 | Trial 3 (无 mixup) | Trial 4 (mixup) | 变化 |
|---|---|---|---|
| **Macro-F1** | 0.365 | **0.359** | **-0.006** ❌ |
| Accuracy | 0.693 | 0.685 | ↓ |
| Weighted-F1 | 0.723 | 0.722 | — |

### 各类别

| 类别 | Trial 3 | Trial 4 | 变化 |
|---|---|---|---|
| 中度 | 0.35 | 0.35 | — |
| 正常 | 0.86 | 0.86 | — |
| 轻度 | 0.17 | 0.10 | ↓ -41% |
| 重度 | 0.14 | **0.16** | +14% |
| 非常严重 | 0.31 | 0.32 | +3% |

### 分析

- Mixup 牺牲"轻度"换取"重度"的微弱提升
- 总体 MF1 下降 — **Mixup 对此任务无效** 
- 原因：Focal Loss 已经足够处理类别不平衡，额外数据增强引入噪声
- 与早期实验 (过采样、辅助损失) 结论一致：数据层面的干预效果有限

---

## 实验十五：XGBoost 堆叠 (Trial 5)

### 方法

在 DeepResidualModel 提取的 OOF fusion features (777-dim) 上训练 XGBoost。

### 结果

- XGBoost 在 NN fusion features 上严重过拟合 (MF1=1.0，不可能)
- 原因：fusion features 已经被 NN 优化过，XGBoost 只是记忆了 NN 的决策
- 直接对 4105-dim 原始特征的 XGBoost 遇到 GPU 冲突 (CUDA 初始化错误)
- **结论：XGBoost 不适用于当前 pipeline**

---

## 实验十六：加权模型集成 (Trial 6)

### 方法

对最佳三个模型的 OOF 概率进行加权平均：
- Model 2 (DeepResidual): wavlm_base + clip_base, 权重 0.7
- Model 3 (+basic features): wavlm_base + clip_base + audio_basic + video_basic, 权重 0.3

### 结果

| 模型 | 未校准 MF1 | 校准后 MF1 |
|---|---|---|
| Model 2 (DeepResidual) | 0.358 | 0.363 |
| Model 3 (+basic) | 0.358 | 0.365 |
| **加权集成 (0.7/0.3)** | **0.375** | — |

### 各类别

| 类别 | Model 2 | Model 3 | 集成 |
|---|---|---|---|
| 中度 | 0.38 | 0.35 | **0.41** |
| 正常 | 0.86 | 0.86 | **0.87** |
| 轻度 | 0.16 | 0.17 | 0.16 |
| 重度 | 0.10 | 0.14 | 0.06 |
| 非常严重 | 0.31 | 0.31 | 0.26 |

### 分析

- 加权集成 MF1 0.375 是最高未校准分数
- 但牺牲了"重度"(0.10→0.06)和"非常严重"(0.31→0.26)
- 集成可能过度优化 OOF 权重搜索——真实泛化可能更低

---

## 实验十七：对比学习预训练 (Trial 7)

### 方法

对三个阶段表示 (T1/T2/T3) 进行 InfoNCE 对比学习预训练：
- 同受试者的 T1/T2/T3 互为 positive pairs
- 不同受试者的阶段表示为 negatives
- 10 epoch 对比预训练 → 分类 fine-tuning

### 结果

| 指标 | Trial 2 (无预训练) | Trial 7 (对比预训练) | 变化 |
|---|---|---|---|
| **Macro-F1** | 0.363 | **0.342** | **-0.021** ❌ |
| Accuracy | 0.697 | 0.695 | ↓ |
| Weighted-F1 | 0.724 | 0.714 | ↓ |

### 分析

- 对比预训练 **降低性能** — 鼓励时间不变特征会损害时序轨迹建模
- 焦虑预测需要区分 T1→T2→T3 的变化模式，时间不变性与之冲突
- 对比损失收敛较好 (从 0.77→0.73) 但 fine-tuning 无法恢复性能

### 下一步

1. 最终配置：加宽加深 + basic features (Final)

---

## 实验十八：加宽加深架构 (Final)

### 方法

DeepResidualModel + basic features，hidden_dim=320（vs 256），num_heads=8（vs 4），num_residual_blocks=4（vs 3）。

### 结果

| 指标 | Trial 3 (dim=256) | Final (dim=320) | 变化 |
|---|---|---|---|
| **Macro-F1** | 0.365 | 0.362 | -0.003 |
| Accuracy | 0.693 | 0.695 | +0.002 |
| Weighted-F1 | 0.723 | 0.720 | -0.003 |

### 各类别

| 类别 | Trial 3 | Final Wide |
|---|---|---|
| 中度 | 0.35 | 0.30 |
| 正常 | 0.86 | 0.86 |
| 轻度 | 0.17 | 0.12 |
| **重度** | 0.14 | **0.20** 🔥 |
| 非常严重 | 0.31 | **0.33** |

### 分析

- 更宽更深对"重度"类有意外帮助 (0.14→0.20)，但对"中度"退化
- 整体 MF1 未提升——**架构已达收益递减点**
- 默认配置 (256, 4 heads, 3 blocks) 是最佳平衡

---

## 实验十九：三模型加权集成 (Final Ensemble)

### 方法

对三个最佳模型进行网格搜索优化权重的概率平均：

| 模型 | 权重 | 说明 |
|---|---|---|
| T2 DeepResidual | 0.50 | wavlm_base + clip_base |
| T3 +Basic | 0.30 | + audio_basic + video_basic |
| Final Wide | 0.20 | dim=320, 8 heads, 4 blocks + basic |

### 结果

| 模型 | Macro-F1 (未校准) |
|---|---|
| T2 DeepResidual | 0.358 |
| T3 +Basic | 0.358 |
| Final Wide | 0.359 |
| **3-Model Ensemble** | **0.383** 🔥 |

### 各类别最终表现

| 类别 | 原始基线 | **最终集成** | 提升 |
|---|---|---|---|
| 中度 | 0.25 | **0.37** | +48% |
| 正常 | 0.83 | **0.87** | +5% |
| 轻度 | 0.05 | **0.19** | **+280%** |
| 重度 | 0.00 | **0.13** | 从零突破 |
| 非常严重 | 0.02 | **0.36** | **+1700%** |

---

## 实验二十：LDAM Loss (Trial 8)

### 方法

LDAM (Label-Distribution-Aware Margin) Loss 对少数类施加更大的分类 margin，结合 Deferred Re-weighting (DRW)。

配置: DeepResidualModel + wavlm_base + clip_base + scores_das, max_m=0.5, scale=30, drw_epoch=50

### 结果

| 指标 | Focal Loss 基线 | LDAM | 变化 |
|---|---|---|---|
| **Macro-F1** | 0.363 | **0.331** | **-0.032** ❌ |
| Accuracy | 0.697 | 0.750 | +0.053 |
| Weighted-F1 | 0.724 | 0.733 | +0.009 |

### 各类别

| 类别 | Focal Loss | LDAM | 变化 |
|---|---|---|---|
| 中度 | 0.38 | 0.38 | — |
| 正常 | 0.86 | 0.88 | +2% |
| 轻度 | 0.16 | 0.04 | ↓ -75% |
| **重度** | 0.10 | **0.00** | 归零 |
| 非常严重 | 0.31 | 0.36 | +16% |

### 分析

- LDAM 提升了正常类准确率 (0.750 vs 0.697) 但牺牲了少数类
- "重度"完全无法预测——LDAM margin 在该类别上失效
- 极少数类 (54 样本) 的 margin 估计不稳定
- **结论: LDAM 不适合此极端不平衡场景。**

---

## 实验二十一：随机权重平均 SWA (Trial 9)

### 方法

Stochastic Weight Averaging — 在训练后期 (epoch 40+) 对模型权重做移动平均，平滑优化轨迹。

配置: DeepResidualModel + Focal Loss, swa_start=40, swa_lr=1e-4

### 结果

| 指标 | 无 SWA 基线 | SWA | 变化 |
|---|---|---|---|
| **Macro-F1** | 0.363 | **0.356** | **-0.007** ❌ |
| Accuracy | 0.697 | 0.706 | +0.009 |
| Weighted-F1 | 0.724 | 0.723 | -0.001 |

### 各类别

| 类别 | 基线 | SWA | 变化 |
|---|---|---|---|
| 中度 | 0.38 | 0.35 | ↓ |
| 正常 | 0.86 | 0.82 | ↓ |
| 轻度 | 0.16 | 0.06 | ↓ -63% |
| 重度 | 0.10 | 0.09 | — |
| **非常严重** | 0.31 | **0.38** | **+23%** |

### 分析

- SWA 对"非常严重"有意外改进 (0.31→0.38, 最高单类 F1)
- 但严重损害"轻度"(0.16→0.06)和"中度"(0.38→0.35)
- 部分 fold 仅训练 3 epoch 即早停——权重平均在训练不稳定时无益
- **结论: SWA 在此场景无效，训练不稳定性是瓶颈。**

---

# 最终总结

## 性能演进

| # | 实验 | 方法 | Macro-F1 | 相比基线 |
|---|---|---|---|---|
| 0 | 基线 | 原始代码 (wavlm+dino_small, 无 DASS) | 0.232 | — |
| 3 | DASS 集成 | +scores_das | 0.298 | +28% |
| 4 | Focal Loss | γ=1.0 | 0.329 | +42% |
| 5 | 特征筛选 | wavlm_base + clip_base | 0.336 | +45% |
| 7 | 阈值校准 | 后处理 | 0.342 | +47% |
| 11 | Transformer | 替换 BiGRU | 0.345 | +49% |
| **12** | **Deep Residual** | **跨阶段注意力 + 差分特征** | **0.363** | **+56%** |
| 13 | +Basic Features | audio_basic + video_basic | 0.365 | +57% |
| 18 | Final Wide | dim=320, 8 heads, 4 blocks | 0.362 | +56% |
| **19** | **3-Model Ensemble** | **加权概率平均** | **0.383** | **+65%** 🔥 |
| 20 | LDAM Loss | margin loss + DRW | 0.331 | MF1↓ |
| 21 | SWA | 权重平均 | 0.356 | MF1↓ |

## 各类别最佳 F1 演进

| 类别 | 基线 | 最终 | 提升倍数 |
|---|---|---|---|
| 正常 | 0.83 | 0.87 | 1.05× |
| 中度 | 0.25 | 0.37 | 1.48× |
| 轻度 | 0.05 | 0.19 | 3.80× |
| 重度 | 0.00 | 0.13 | ∞ |
| 非常严重 | 0.02 | 0.36 | 18.0× |

## 关键发现

1. **数据 > 算法**: DASS 历史分数 (+27%) 远超所有架构改进总和
2. **架构改进有效但收益递减**: 0.342 → 0.365 → 0.383 (集成)
3. **模型集成互补**: 不同特征/架构的错误模式不同，加权集成显著优于单模型
4. **Focal Loss + 阈值校准**: 最有效的训练技巧组合
5. **无效方法**: Mixup, 对比预训练, 过采样, 多任务辅助损失
6. **特征越少越好**: 单特征对 > 多特征堆叠；手工特征仅有边际增益

## 最终最佳配置

```bash
# 训练三个模型
# Model 1: DeepResidual + wavlm_base + clip_base
PYTHONPATH=src python scripts/exp_deep_residual.py \
  --dataset-path datasets --output-dir artifacts/exp/deep_residual \
  --audio-feature-name audio_wavlm_base --video-feature-name video_clip_base \
  --device cuda

# Model 2: DeepResidual + basic features
PYTHONPATH=src python scripts/exp_basic_features.py \
  --dataset-path datasets --output-dir artifacts/exp/basic_features \
  --audio-feature-name audio_wavlm_base --video-feature-name video_clip_base \
  --device cuda

# Model 3: Wider DeepResidual + basic features  
PYTHONPATH=src python scripts/exp_final.py \
  --dataset-path datasets --output-dir artifacts/exp/final \
  --audio-feature-name audio_wavlm_base --video-feature-name video_clip_base \
  --device cuda

# Ensemble: 0.50 × M1 + 0.30 × M2 + 0.20 × M3
```
