# CCAC MAPS 基线改进方向

## 当前基线概况

- **模型**: `LongitudinalAnxietyModel`，约 166 万参数
- **输入**: `audio_wavlm_base` (3072维) + `video_dinov2_small` (768维) = 3840维
- **架构**: Clip Encoder (共享MLP) → 注意力池化 (阶段内4个片段融合) → 双向GRU (T1→T2→T3) → 分类器
- **训练**: 5折分层交叉验证，类别加权交叉熵损失，AdamW优化器，早停 patience=12
- **推理**: 5折模型概率平均集成

## 基线结果 (OOF)

| 指标 | 值 |
|---|---|
| 准确率 (Accuracy) | 69.8% |
| 宏观 F1 (Macro F1) | 0.232 |
| 加权 F1 (Weighted F1) | 0.671 |

### 各类别表现

| 类别 | 样本数 | 占比 | 精确率 | 召回率 | F1 |
|---|---|---|---|---|---|
| 正常 | 1169 | 76.5% | 0.80 | 0.87 | 0.83 |
| 中度 | 186 | 12.2% | 0.26 | 0.25 | 0.25 |
| 轻度 | 58 | 3.8% | 0.08 | 0.03 | 0.05 |
| 重度 | 54 | 3.5% | 0.00 | 0.00 | 0.00 |
| 非常严重 | 60 | 3.9% | 0.04 | 0.02 | 0.02 |

**核心问题**: 模型对少数类（轻度、重度、非常严重）几乎无预测能力，严重倾向于预测"正常"。

---

## 可用数据资源

### DASS-21 量表历史（labels.csv，当前基线未使用）(test set not availablle, wrong idea!!!!!!)

| 时间点 | 可用字段 |
|---|---|
| T1 | `t1_depression_score`, `t1_depression_level`, `t1_anxiety_score`, `t1_anxiety_level`, `t1_stress_score`, `t1_stress_level` |
| T2 | `t2_depression_score`, `t2_depression_level`, `t2_anxiety_score`, `t2_anxiety_level`, `t2_stress_score`, `t2_stress_level` |
| T3 | `t3_depression_score`, `t3_depression_level`, `t3_anxiety_score`, `t3_anxiety_level`, `t3_stress_score`, `t3_stress_level` |
| T4（目标） | `t4_anxiety_level` |

> 总共 18 列 DASS 历史 + 1 列目标标签。基线只用 T4 焦虑等级，完全忽略 T1-T3 的所有分数和等级。

### 音频特征（5种）

| 特征名 | 来源模型 | 池化后维度 | 有时序序列 |
|---|---|---|---|
| `audio_basic` | 手工声学特征 | 74 | ✅ `sequence.npz` (T, 18) |
| `audio_wavlm_base` | microsoft/wavlm-base | 3072 | ❌ |
| `audio_chinese_hubert_base` | TencentGameMate/chinese-hubert-base | 3072 | ❌ |
| `audio_wav2vec2_chinese_base` | TencentGameMate/chinese-wav2vec2-base | 3072 | ❌ |
| `audio_wav2vec2_xlsr_chinese` | jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn | 4096 | ❌ |

### 视频特征（7种）

| 特征名 | 来源模型 | 池化后维度 | 有时序序列 |
|---|---|---|---|
| `video_basic` | 手工特征（亮度/模糊/运动） | 13 | ✅ `sequence.npz` (T, 3) |
| `video_dinov2_small` | facebook/dinov2-small | 768 | ❌ |
| `video_dinov2_base` | facebook/dinov2-base | 1536 | ❌ |
| `video_clip_base` | openai/clip-vit-base-patch32 | 1024 | ❌ |
| `video_clip_large` | openai/clip-vit-large-patch14 | 1536 | ❌ |
| `video_siglip_base` | google/siglip-base-patch16-224 | 1536 | ❌ |
| `video_vit_mae_base` | facebook/vit-mae-base | 1536 | ❌ |

> 当前基线默认仅用 1 种音频 + 1 种视频特征。共 12 种特征可用。

---

## 改进方向（按优先级排序）

### 1. 🔴 利用 DASS 历史分数（最高优先级）

**当前状态**: T1-T3 的焦虑分数、抑郁分数、压力分数完全未使用。

**理由**: DASS 历史是 T4 最强的先验信息。一个在 T1-T3 焦虑分数持续升高的人，T4 大概率也高。

**实现方案**:

- **方案 A（最简）**: 将 3 个焦虑分数 `[t1_anxiety_score, t2_anxiety_score, t3_anxiety_score]` 作为额外特征，拼接在融合层
- **方案 B**: 加入全部 9 个分数（焦虑 + 抑郁 + 压力 × 3 时间点）
- **方案 C**: 将分数和等级编码后，通过独立 MLP 编码器处理，再与音视频特征融合

**预期收益**: 宏观 F1 提升 5-10 个百分点

---

### 2. 🔴 解决类别不平衡

**当前状态**: "正常"占 76.5%，模型对重度/非常严重的召回率为 0。

**可行方法**:

| 方法 | 说明 | 实现难度 |
|---|---|---|
| 增强类别权重 | `--class-weight-power 2.0` 或更高（已有 CLI 参数） | 极低 |
| Focal Loss | 自动降低易分类样本（正常类）的损失权重 | 低 |
| 过采样 | 在训练 batch 中重复少数类样本 | 低 |
| 标签平滑 | `--label-smoothing 0.1`（已有 CLI 参数） | 极低 |
| 阈值调整 | 在 OOF 预测上搜索最优决策阈值 | 低 |
| 类别平衡采样 | 每个 batch 强制各类别等量 | 低 |

**预期收益**: 少数类召回率从 0 提升到可接受水平

---

### 3. 🟡 多特征融合

**当前状态**: 仅用 1 对特征。有 5×7=35 种音视频组合可用。

**方案**:

- **后期融合（最简）**: 训练多个不同特征对的模型，投票/平均集成
- **早期融合**: 将多个池化向量直接拼接，增大输入维度
- **中期融合**: 各特征独立编码后，在注意力池化层融合

**预期收益**: 集成通常带来 2-5% 的稳定提升

---

### 4. 🟡 多任务辅助损失

**当前状态**: 仅预测 T4 焦虑等级。

**方案**: 让共享编码器同时预测 T1、T2、T3 的焦虑等级：

```
                    ┌→ 分类器_T4 → T4焦虑（主任务）
共享编码器 ────┼→ 分类器_T1 → T1焦虑（辅助）
                    ├→ 分类器_T2 → T2焦虑（辅助）
                    └→ 分类器_T3 → T3焦虑（辅助）
```

**理由**: 迫使编码器学习跟踪焦虑轨迹的表示，而不仅仅是终点分类。T1-T3 标签天然存在，无需额外标注。

**预期收益**: 更好的泛化，间接改善少数类

---

### 5. 🟢 利用时序序列特征

**当前状态**: `audio_basic` 和 `video_basic` 有帧级别 `sequence.npz`，但基线只用 `pooled.npy`。

**方案**:

- 修改 `_build_release_features` 读取 `sequence.npz`
- 在 Clip Encoder 之前增加时序编码器（如 1D CNN 或小型 Transformer）
- 可用的时序特征：18 维音频帧特征 + 3 维视频帧特征

**理由**: 捕捉片段内的动态变化（语速、停顿、表情变化、身体运动）

**预期收益**: 中等，但需要架构改动

---

### 6. 🟢 架构升级

| 组件 | 当前 | 可升级为 |
|---|---|---|
| 时序建模 | 单层 BiGRU | Transformer Encoder |
| 片段聚合 | 标量注意力 | 多头注意力 |
| 位置编码 | 可学习 3×256 | 正弦编码 + 可学习残差 |
| 正则化 | Dropout 0.2 | DropPath + 更大的 Dropout |

**预期收益**: 边际提升，主要是工程优化

---

## 实施路线建议

### 第一阶段（1-2 小时，预期提升最大）

1. 添加 DASS 焦虑分数 `[t1_score, t2_score, t3_score]` 作为输入特征
2. 调高 `class_weight_power` 至 2.0-3.0
3. 启用 `label_smoothing=0.1`

### 第二阶段（半天）

4. 实现 Focal Loss 替代交叉熵
5. 多特征对集成（训练 3-5 个不同特征组合的模型）
6. 添加抑郁和压力分数

### 第三阶段（1-2 天）

7. 实现多任务辅助损失
8. 利用 `audio_basic` 和 `video_basic` 的时序序列
9. 架构改进（Transformer 替换 GRU 等）

---

## 关键代码位置

| 文件 | 说明 |
|---|---|
| `src/ccac/baselines/anxiety_baseline.py` | 完整基线代码（639行） |
| `scripts/train_anxiety_baseline.py` | 训练入口脚本 |
| `src/ccac/baselines/anxiety_baseline.py:317-346` | `_load_release_train_val` — 数据加载与缓存 |
| `src/ccac/baselines/anxiety_baseline.py:418-439` | `_build_release_features` — 特征构建（需改动以加入 DASS） |
| `src/ccac/baselines/anxiety_baseline.py:63-111` | `LongitudinalAnxietyModel` — 模型架构 |
| `src/ccac/baselines/anxiety_baseline.py:170-310` | `train_anxiety_baseline` — 训练主循环 |
| `src/ccac/baselines/anxiety_baseline.py:545-553` | `_class_weights` — 类别权重计算 |
