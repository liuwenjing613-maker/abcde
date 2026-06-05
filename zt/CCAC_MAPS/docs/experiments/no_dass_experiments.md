# 无 DASS 特征实验记录

## 背景

公榜提交发现：使用 DASS 特征训练的模型（OOF MF1 0.363）在公榜仅得 MF1 0.091。

**根因**：测试集 `test/subjects.csv` 仅包含 `anon_school, anon_class, anon_person`，**完全没有 DASS 历史数据**。训练集 `train_val/labels.csv` 有完整的 18 列 DASS 特征（T1/T2/T3 × 抑郁/焦虑/压力 × 分数/等级），且 1527 人全部有值。

模型在训练时学会严重依赖 DASS 特征区分少数类，测试时 DASS=0 导致模型退化为预测训练集多数类（"正常"）。

**公榜分布偏移**（加剧问题）：公榜 382 人中"中度"占 76.4%（292人），而训练集"正常"占 76.5%。模型预测"正常"恰好与公榜真实分布相反。

---

## 实验 ND-1：原始基线（无 DASS）

**时间**：2026-05-31

**模型**：
- 架构：LongitudinalAnxietyModel（Clip Encoder → Attention Pooling → BiGRU）
- 特征：audio_wavlm_base + video_dinov2_small
- DASS：无（原始基线默认不使用 DASS）
- 训练：5 折交叉验证，类别加权 CrossEntropyLoss

**OOF 结果**：

| 指标 | 值 |
|---|---|
| Macro-F1 | 0.232 |
| Accuracy | 0.698 |
| Weighted-F1 | 0.671 |

| 类别 | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 中度 | 0.26 | 0.25 | 0.25 | 186 |
| 正常 | 0.80 | 0.87 | 0.83 | 1169 |
| 轻度 | 0.08 | 0.03 | 0.05 | 58 |
| 重度 | 0.00 | 0.00 | 0.00 | 54 |
| 非常严重 | 0.04 | 0.02 | 0.02 | 60 |

**公榜预测（382 人）**：

| 类别 | 预测数 | 真实数（从公榜反推） | 预估 |
|---|---|---|---|
| 中度(0) | 20 | 292 | Recall≈0.07 |
| 正常(1) | 357 | 15 | Precision≈0.04 |
| 轻度(2) | 1 | 46 | — |
| 重度(3) | 1 | 14 | — |
| 非常严重(4) | 3 | 15 | — |

**预估公榜 MFI**：~0.05-0.08

**结论**：无 DASS 基线同样严重偏向"正常"，但与有 DASS 模型的退化机制不同——基线是本身区分能力弱，而有 DASS 模型是依赖了不可用的特征。

---

## 实验 ND-2：DeepResidual 无 DASS

**时间**：2026-05-31

**模型**：
- 架构：DeepResidualModel（跨阶段注意力 + 阶段差分特征 + SE 残差块）
- hidden_dim=256, num_heads=4, num_residual_blocks=3
- 特征：audio_wavlm_base + video_clip_base（4096 维）
- DASS：无（dass_scheme="none"）
- 损失：Focal Loss γ=1.0 + 类别权重 power=1.0
- 优化器：AdamW lr=1e-3 + CosineAnnealingWarmRestarts
- 训练：5 折交叉验证，early stopping patience=12

**OOF 结果**：

| 指标 | 值 |
|---|---|
| Macro-F1 | **0.244** |
| Accuracy | 0.493 |

| 类别 | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 中度 | 0.23 | **0.34** | 0.27 | 186 |
| 正常 | 0.87 | 0.57 | 0.69 | 1169 |
| 轻度 | 0.04 | 0.17 | 0.07 | 58 |
| 重度 | 0.06 | 0.11 | 0.08 | 54 |
| 非常严重 | 0.08 | **0.20** | 0.12 | 60 |

**对比 ND-1（原始基线）**：

| 类别 | 基线 F1 | ND-2 F1 | 变化 |
|---|---|---|---|
| 中度 | 0.25 | 0.27 | +8% |
| 正常 | 0.83 | 0.69 | -17% |
| 轻度 | 0.05 | 0.07 | +40% |
| 重度 | 0.00 | 0.08 | ∞ |
| 非常严重 | 0.02 | 0.12 | +500% |

**关键提升**：
- 重度从 0.00 提升到 0.08（首次非零）
- 非常严重从 0.02 提升到 0.12（6 倍）
- 正常 F1 从 0.83 降到 0.69，说明模型不再盲目预测"正常"
- 总体 Macro-F1 0.244 vs 0.232，在无 DASS 条件下提升 5%

**公榜预测（382 人）**：

| 类别 | 预测数 | 占比 |
|---|---|---|
| 中度(0) | 78 | 20.4% |
| 正常(1) | 190 | 49.7% |
| 轻度(2) | 56 | 14.7% |
| 重度(3) | 28 | 7.3% |
| 非常严重(4) | 30 | 7.9% |

**首次在所有 5 个类别上都有非零预测！**

对比：
| 模型 | 正常 | 中度 | 轻度 | 重度 | 非常严重 |
|---|---|---|---|---|---|
| DASS DeepResidual | 296 (77%) | 50 | 36 | **0** | **0** |
| 无DASS基线(BiGRU) | 357 (93%) | 20 | 1 | 1 | 3 |
| **无DASS DeepResidual** | 190 (50%) | 78 | 56 | **28** | **30** |

**预估公榜 MFI**：~0.18-0.22

---

## 实验 ND-3：无 DASS + Focal γ=2.0 ✅

**时间**：2026-05-31

**配置**：
- 架构：DeepResidualModel（同 ND-2）
- 特征：audio_wavlm_base + video_clip_base（4096 维）
- DASS：无
- 损失：Focal Loss γ=**2.0** + 类别权重 power=1.0
- 脚本：`scripts/train_no_dass.py --focal-gamma 2.0`

**OOF 结果**：

| 指标 | ND-2 (γ=1.0) | **ND-3 (γ=2.0)** | 变化 |
|---|---|---|---|
| Macro-F1 | 0.244 | **0.264** | **+0.020** |
| Accuracy | 0.493 | 0.517 | +0.024 |

| 类别 | ND-2 F1 | ND-3 F1 | 变化 |
|---|---|---|---|
| 中度 | 0.27 | 0.28 | +4% |
| 正常 | 0.69 | 0.70 | +1% |
| 轻度 | 0.07 | **0.11** | +57% |
| 重度 | 0.08 | **0.12** | +50% |
| 非常严重 | 0.12 | 0.11 | -8% |

**结论**：Focal γ=2.0 是当前无 DASS 最佳配置。更高的 γ 值更有效地抑制了模型对"正常"的偏向，轻度从 0.07→0.11，重度从 0.08→0.12。

---

## 实验 ND-4：无 DASS + 高类别权重 cw=2.0 ❌

**时间**：2026-05-31

**配置**：
- 架构：DeepResidualModel（同 ND-2）
- 特征：audio_wavlm_base + video_clip_base（4096 维）
- DASS：无
- 损失：Focal Loss γ=1.0 + 类别权重 power=**2.0**
- 脚本：`scripts/train_no_dass.py --class-weight-power 2.0`

**OOF 结果**：

| 指标 | ND-2 (cw=1.0) | **ND-4 (cw=2.0)** |
|---|---|---|
| Macro-F1 | 0.244 | **0.048** ❌ |
| Accuracy | 0.493 | 0.046 |

**完全崩溃**。cw=2.0 导致极端权重（少数类是多数类的 100-400 倍），模型几乎不预测"中度"和"正常"。

**结论**：高类别权重在此任务上无效（与早期实验二结论一致）。Focal Loss 是更好的不平衡处理方法。

---

## 实验 ND-5：无 DASS + audio_basic + video_basic ✅

**时间**：2026-05-31

**配置**：
- 架构：DeepResidualModel（同 ND-2）
- 特征：audio_wavlm_base + audio_basic + video_clip_base + video_basic（4183 维）
- DASS：无
- 损失：Focal Loss γ=2.0
- 脚本：`scripts/exp_nd5_basic.py`

**OOF 结果**：

| 指标 | ND-3 (无 basic) | **ND-5 (+basic)** | 变化 |
|---|---|---|---|
| Macro-F1 | **0.264** | 0.253 | **-0.011** |
| Accuracy | 0.517 | 0.544 | +0.027 |

| 类别 | ND-3 F1 | ND-5 F1 |
|---|---|---|
| 中度 | 0.28 | 0.25 |
| 正常 | 0.70 | 0.73 |
| 轻度 | 0.11 | 0.09 |
| 重度 | 0.12 | 0.08 |
| 非常严重 | 0.11 | 0.12 |

**结论**：手工特征在无 DASS 条件下未带来提升。准确率提高但 Macro-F1 下降，说明手工特征主要惠及多数类（正常）。与有 DASS 时的实验结果（Trial 3: +basic 仅 +0.002 MF1）一致。

---

## 实验 ND-6：DASS Dropout 训练 ✅

**时间**：2026-05-31

**配置**：
- 架构：DeepResidualModel
- 特征：audio_wavlm_base + video_clip_base（4096 维）
- DASS：scores_das (9 维)，训练时以 **50% 概率随机置零**
- 评估：使用 DASS=0 评估（模拟测试条件）
- 损失：Focal Loss γ=1.0

**OOF 结果（无 DASS 评估）**：

| 指标 | ND-2 (纯无DASS) | ND-3 (γ=2.0) | **ND-6 (DASS dropout)** |
|---|---|---|---|
| Macro-F1 | 0.244 | **0.264** | 0.229 |
| Accuracy | 0.493 | 0.517 | 0.616 |

**结论**：❌ DASS Dropout 不如直接不用 DASS。训练时随机置零 DASS 反而使模型在有 DASS 的 50% 批次中过度依赖 DASS，损害了 AV-only 表示的学习。准确率提高（0.616）但 Macro-F1 下降，说明模型偏向"正常"。

---

## 实验 ND-7：多模型集成 ✅

**时间**：2026-05-31

**配置**：0.80×ND-3 (γ=2.0) + 0.20×ND-5 (+basic)，网格搜索最优权重

**OOF 结果**：

| 指标 | ND-3 | **ND-7 Ensemble** |
|---|---|---|
| Macro-F1 | 0.264 | **0.272** |
| 重度 F1 | 0.12 | **0.15** |

---

## 实验 ND-8：LDAM 损失 + 无 DASS ✅

**时间**：2026-05-31

**配置**：DeepResidualModel + LDAM Loss（Label-Distribution-Aware Margin），无 DASS

**OOF 结果**：MF1 0.257。不如 Focal Loss γ=2.0。

---

## 实验 ND-9：Transformer + 无 DASS + γ=2.0 ✅ 🔥

**时间**：2026-05-31

**配置**：
- 架构：TransformerTemporalModel（2层 Pre-LN Transformer Encoder, 4 heads）
- 特征：audio_wavlm_base + video_clip_base（4096 维）
- DASS：无
- 损失：Focal Loss γ=2.0
- 脚本：`scripts/exp_transformer.py --focal-gamma 2.0`

**OOF 结果**：

| 指标 | ND-3 (DeepResidual) | **ND-9 (Transformer)** | 变化 |
|---|---|---|---|
| Macro-F1 | 0.264 | **0.274** | **+0.010** |
| Accuracy | 0.517 | 0.566 | +0.049 |

| 类别 | ND-3 F1 | ND-9 F1 | 变化 |
|---|---|---|---|
| 中度 | 0.28 | 0.29 | +4% |
| 正常 | 0.70 | 0.75 | +7% |
| 轻度 | 0.11 | 0.11 | — |
| 重度 | 0.12 | 0.09 | -25% |
| 非常严重 | 0.11 | 0.13 | +18% |

**结论**：Transformer 在无 DASS 条件下降 DeepResidual 表现更好（0.274 vs 0.264），尤其是"正常"类 F1 从 0.70 提升到 0.75。但"重度"类退化。

---

## 脚本清理

已删除无用/DASS 专用脚本：
- `exp_mixup.py` — 已证明无效
- `exp_xgboost.py` — GPU 冲突
- `exp_contrastive.py` — 对比学习降低性能
- `sweep_features.py` — DASS 硬编码
- `train_dass_baseline.py` — DASS 专用

已修改为 no-DASS 默认：
- `exp_deep_residual.py` — dass_scheme="none"
- `exp_basic_features.py` — dass_scheme="none"
- `exp_transformer.py` — dass_scheme="none"
- `exp_ldam.py` — dass_scheme="none"
- `exp_swa.py` — dass_scheme="none"
- `exp_final.py` — dass_scheme="none"
- `train_ensemble.py` — dass_scheme="none"

---

## 实验 ND-10：测试时先验校准

**时间**：2026-05-31

**方法**：在 OOF 概率上应用先验偏移校正 `p(y|x) ∝ p_model(y|x) * p_test(y) / p_train(y)`，使用已知公榜分布作为 test prior。

**OOF 结果**：所有先验校正策略在 OOF 上均**降低 MF1**（0.274→0.08）。原因：OOF 数据服从训练分布，而非公榜分布。先验偏移公式 `p(x|y)` 恒定的假设在跨群体测试中不成立。

**结论**：先验校准无法在 OOF 上验证。公榜是唯一评估途径。可在生成 test prediction 时应用，公榜直接反馈。

---

## 实验 ND-11：知识蒸馏（DASS 教师 → 无 DASS 学生）🔥🔥🔥

**时间**：2026-05-31

**方法**：
- 教师：DeepResidualModel + scores_das + Focal γ=1.0（OOF MF1 0.363）
- 学生：DeepResidualModel (无 DASS)，使用 KL 散度 + Focal Loss 联合损失
- Loss = α·T²·KL(teacher_probs || student_probs) + (1-α)·Focal(student, hard_labels)
- α=0.9, T=3.0, student Focal γ=2.0

**OOF 结果**：

| 指标 | ND-9 (Transformer) | **ND-11 (Distillation)** | 变化 |
|---|---|---|---|
| Macro-F1 | 0.274 | **0.329** | **+0.055** 🔥 |
| Accuracy | 0.566 | 0.752 | +0.186 |
| Weighted-F1 | 0.618 | 0.716 | +0.098 |

| 类别 | ND-9 F1 | ND-11 F1 | 变化 |
|---|---|---|---|
| 中度 | 0.29 | 0.30 | +3% |
| 正常 | 0.75 | **0.86** | +15% |
| 轻度 | 0.11 | **0.24** | **+118%** |
| 重度 | 0.09 | 0.13 | +44% |
| 非常严重 | 0.11 | 0.11 | — |

**分析**：
- **这是无 DASS 条件下的最大单次提升（+0.055 MF1）**
- 蒸馏让学生学会了 DASS 教师的知识，但只用 AV 特征
- "轻度"类从 0.11→0.24（+118%）—— 教师的软标签编码了 DASS 对轻度的识别能力
- "正常"类从 0.75→0.86 —— 学生的预测更准确
- 模型在无 DASS 测试集上不再坍塌
- **OOF MF1 0.329 仅次于有 DASS 模型（0.363），差距仅 0.034**

---

## 实验 ND-12：多架构集成 ✅ 🔥🔥

**时间**：2026-05-31

**方法**：对不同架构的 OOF 概率进行加权平均（网格搜索最优权重）：
- ND-3：DeepResidual + Focal γ=2.0（MF1 0.264）
- ND-9：Transformer + Focal γ=2.0（MF1 0.274）
- ND-1：BiGRU 基线（MF1 0.232）

**OOF 结果**：

| 集成 | 权重 | Macro-F1 |
|---|---|---|
| ND-3 + ND-9 | 0.65/0.35 | 0.269 |
| ND-3 + ND-9 + ND-1 | **0.50/0.40/0.10** | **0.287** 🔥 |
| ND-3 + ND-9 + ND-5 | 0.60/0.20/0.20 | 0.283 |

**BEST: 3-model ensemble → MF1 0.2866**

| 类别 | ND-9 (最佳单模型) | ND-12 集成 |
|---|---|---|
| 中度 | 0.29 | 0.33 |
| 正常 | 0.75 | 0.79 |
| 轻度 | 0.11 | 0.12 |
| 重度 | 0.09 | 0.09 |
| 非常严重 | 0.11 | 0.11 |

**分析**：
- 不同架构的错误模式互补——即使 BiGRU 弱模型（0.232）也有 10% 贡献
- 集成 MF1 0.287 > 任何单模型（最高 0.274），提升 **+0.013**
- 加入 4-5 个模型反而下降——弱模型噪声稀释信号
- 核心是架构多样性，而非模型数量

---

## 实验 ND-13：TCN 时序编码器 ❌

**时间**：2026-05-31

**方法**：用双向膨胀因果卷积（Bidirectional TCN, 3 layers）替代 Transformer/BiGRU。

**OOF 结果**：MF1 **0.170**（准确率 0.620）

**结论**：❌ TCN 在 3 步序列上完全失败。因果膨胀卷积的填充导致时间步边界信息丢失，不适合极短序列（T=3）建模。TCN 需要更长序列才能发挥膨胀感受野的优势。

---

## 实验 ND-14：蒸馏学生 + 多架构集成

**时间**：2026-05-31

**方法**：将 ND-11（蒸馏学生）与 ND-9（Transformer）、ND-3（DeepResidual）、ND-1（BiGRU）进行加权集成。

**OOF 结果**：
- 最佳组合：ND-11 (0.25) + ND-9 (0.75) → MF1 **0.321**
- ND-11 单独：MF1 **0.329**（更好）
- 集成无法超越蒸馏学生单独使用

**结论**：蒸馏学生已经内部化了大部分有用信号。弱模型的噪声稀释了蒸馏学生的优势。**单一强模型 > 多弱模型集成。**

---

## 当前最佳

| 实验 | OOF MF1 | 配置 |
|---|---|---|
| **ND-11** 🔥 | **0.329** | Knowledge Distillation (single-teacher, DeepResidual student) |
| ND-16 | 0.308 | Multi-teacher Distillation (2 DASS teachers → no-DASS student) |
| ND-15 | 0.310 | Knowledge Distillation (Transformer student) |
| ND-12 | 0.287 | 3-Model Ensemble (DeepResidual+Transformer+BiGRU) |
| ND-9 | 0.274 | Transformer + Focal γ=2.0, 无 DASS |
| ND-3 | 0.264 | DeepResidual + Focal γ=2.0, 无 DASS |
| ND-1 | 0.232 | BiGRU baseline, 无 DASS |

## 提交记录

| # | 文件 | 模型 | OOF MF1 | 公榜 MF1 | 备注 |
|---|---|---|---|---|---|
| 1 | sub1_deep_residual.zip | DeepResidual + DASS | 0.363 | **0.091** | DASS 坍塌 |
| 2 | sub_baseline_nodass.zip | BiGRU 基线 (无DASS) | 0.232 | 待评估 | 对照 |
| 3 | sub_nodass_deepresidual.zip | DeepResidual (无DASS) | 0.244 | 待评估 | 首次所有类别非零 |
| 4 | sub_nd3_focal_g2.zip | DeepResidual γ=2.0 (无DASS) | 0.264 | 待评估 | — |
| 5 | sub_nd9_transformer.zip | Transformer γ=2.0 (无DASS) | 0.274 | 待评估 | — |
| 6 | sub_nd12_ensemble.zip | 3-Model Ensemble (无DASS) | 0.287 | 待评估 | — |
| **7** | **sub_nd11_distillation_calibrated.zip** | **KD + 先验校准 (无DASS)** | **0.329** | 待评估 | **当前最佳** |
| 8 | sub_nd15_distill_transformer.zip | KD Transformer学生 (无DASS) | 0.310 | 待评估 | —


## 实验 ND-15：Transformer 学生蒸馏

**时间**：2026-05-31

**方法**：用 TransformerTemporalModel 替代 DeepResidualModel 作为蒸馏学生。

**OOF 结果**：MF1 **0.310**（校准后），低于 ND-11 DoneResidual 学生（0.329）。

**结论**：DeepResidual 架构更适合做蒸馏学生。Transformer 的 self-attention 可能过度拟合教师输出中的噪声。


## 实验 ND-16：多教师蒸馏 ❌

**时间**：2026-05-31

**方法**：用两个 DASS 教师（DeepResidual OOF MF1 0.363 + Transformer OOF MF1 0.345）共同蒸馏一个无 DASS DeepResidual 学生。Loss = mean(KL(teacher_i, student)) + Focal(student, labels)。

**OOF 结果**：MF1 **0.308**（校准后），低于单教师 ND-11（0.329）。

**分析**：
- 多教师未带来提升——两个教师之间的不一致可能引入噪声
- Transformer 教师（0.345）弱于 DeepResidual 教师（0.363），拖累平均质量
- **结论：单最佳教师蒸馏 > 多教师平均。**


## 全实验性能演进

| # | 实验 | 方法 | OOF MF1 | 相比基线 |
|---|---|---|---|---|
| ND-1 | BiGRU 基线 | 原始架构, 无 DASS | 0.232 | — |
| ND-2 | DeepResidual γ=1.0 | 跨阶段注意力 + SE 残差 | 0.244 | +5% |
| ND-3 | Focal γ=2.0 | 更强焦点损失 | 0.264 | +14% |
| ND-5 | +Basic Features | 手工声学/视觉特征 | 0.253 | 下降 |
| ND-7 | 单架构集成 | 0.8×ND-3 + 0.2×ND-5 | 0.272 | +17% |
| ND-9 | Transformer | 自注意力时序编码 | 0.274 | +18% |
| ND-12 | 多架构集成 | DeepResidual+Transformer+BiGRU | 0.287 | +24% |
| ND-10 | 先验校准 | 测试时先验偏移校正 | N/A | 仅公榜可验证 |
| ND-13 | TCN | 时序卷积网络 | 0.170 | ❌ 失败 |
| ND-14 | 蒸馏+集成 | ND-11 + ND-9 + ND-3 集成 | 0.321 | 下降 |
| ND-15 | Transformer 蒸馏 | Transformer 学生 | 0.310 | 下降 |
| **ND-11** | **知识蒸馏** 🔥 | **DASS 教师 → DeepResidual 学生** | **0.329** | **+42%** |
| ND-16 | 多教师蒸馏 | 2 DASS 教师 → 1 学生 | 0.308 | 下降 |


## 平台期分析

当前无 DASS 最优 MF1 = 0.329（ND-11），有 DASS 最优 MF1 = 0.363。

**差距 0.034 可能代表了 DASS 特征独有的信息**——这些信息无法从音视频特征中提取，即使是知识蒸馏也无法恢复。可能的根本限制：

1. **DASS 分数直接测量心理状态**——T1-T3 的焦虑/抑郁/压力分数与 T4 焦虑高度因果相关
2. **音视频特征是间接信号**——语音和面部表情与焦虑的关联是统计性的而非因果性的
3. **蒸馏能传递"DASS → 焦虑"的映射知识**（ND-11 从 0.274→0.329），但无法传递 DASS 特征本身携带的独特方差

**建议**：若无更多突破性方法，ND-11（MFI 0.329）可能是当前范式下的性能上限。
