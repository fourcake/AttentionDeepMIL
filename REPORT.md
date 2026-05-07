# iMiGUE ABMIL Baseline 实验报告

**日期**: 2026-05-06 ~ 2026-05-07
**分支**: `dev/imigue-baseline`
**代码**: https://github.com/fourcake/AttentionDeepMIL
**服务器**: xingke@10.50.0.31, 8×A800 80GB

---

## 一、研究背景

### 1.1 研究问题

网球运动员赛后新闻发布会中的微动作（micro-gesture）能否通过弱监督方式预测比赛结果（Win/Lose）？

核心假设：微动作中的心理信号（如自我接触、重复性行为等）与比赛结果存在关联，可通过注意力机制（ABMIL）在弱标签（bag-level Win/Lose）下自动发现关键实例。

### 1.2 方法论

- **框架**: Multiple Instance Learning (MIL)
  - Bag = 一个视频（一名运动员的整段发布会）
  - Instance = 一个微动作段（标注的 start_frame:end_frame）
  - Bag label = Win (1) / Lose (0)
- **模型**: Attention-based Deep MIL (Ilse et al., ICML 2018)
- **目标**: AAAI 2027 投稿，10 周时间线

---

## 二、数据

### 2.1 数据集：iMiGUE

| 项 | 值 |
|---|---|
| 总视频数 | 359 |
| 总被试数 | 72 |
| 总标注数 | 18,498 |
| 微动作类别 | 31 类（class 1-31） |
| 非微动作 | class 99，占 5,715 条（31%） |
| 骨骼来源 | OpenPose 137-keypoint 全身骨骼 |
| 每帧维度 | 411 = 137 keypoint × 3 (x, y, confidence) |
| 骨骼格式 | xlsx → 已转换为 .npy（359 个文件） |
| 典型骨骼形状 | (9947, 411) float32，约 9947 帧/视频 |
| 空文件 | 视频 347、348 的 xlsx 为空（0行0列） |

### 2.2 CSV 字段

```
video_id, label_id, start_time, start_frame, end_time, end_frame,
class, subject_id, subject_gender, subject_nationality, win_or_lose
```

### 2.3 官方数据划分

| Split | 视频数 | Win | Lose | 被试数 | 说明 |
|---|---|---|---|---|---|
| Train | 245 | 192 | 53 | 37 | Win:Lose ≈ 3.6:1 |
| Val（官方） | 10 | 10 | 0 | 5 | **不可用**：全 Win，5 被试与 Train 重叠 |
| Test | 104 | 56 | 48 | 35 | 被试与 Train 完全独立 |

**官方 Val 不可用的原因**：
- 标签全为 Win，Balanced Accuracy 恒为 1.0，无法用于模型选择
- 5 个被试与 Train 重叠，存在数据泄露风险

### 2.4 实际使用的数据划分

从 Train 的 245 个视频中，按被试划 5-fold Subject-Disjoint CV：

| Fold | Train | Val | Val Win | Val Lose |
|---|---|---|---|---|
| 0 | 208 | 35 | 26 | 9 |
| 1 | 185 | 58 | 43 | 15 |
| 2 | 202 | 41 | 30 | 11 |
| 3 | 189 | 54 | 40 | 14 |
| 4 | 188 | 55 | 41 | 14 |

Test 始终固定为 104 个视频（56 Win, 48 Lose），被试与所有 Train/Val fold 完全独立。

**排除的视频**: 347, 348（空骨骼文件，默认排除）

---

## 三、特征工程

### 3.1 输入特征构造流程

每个微动作实例（instance）的特征构造：

```
原始骨骼 (T_video, 411)
    ↓ 1. 按 start_frame:end_frame 切片
(T_inst, 411)
    ↓ 2. 保存原始零帧 mask（归一化前）
raw_valid = skel.any(axis=1)
    ↓ 3. Z-normalize（per-video，仅非零帧的 mean/std）
(T_inst, 411)  normalized
    ↓ 4. 用 raw_valid mask 过滤 OpenPose 失败帧
(T_valid, 411)
    ↓ 5. Pooling over T_valid（默认 mean-pool）
(411,)
    ↓ 6. 拼接 class one-hot (32维: 31类微动作 + class 99)
(443,)
    ↓ 7. Pad 到 max_bag_size=256，用 attention mask 屏蔽 padding
(256, 443) + (256,) mask
```

### 3.2 特征维度

| 部分 | 维度 | 说明 |
|---|---|---|
| 骨骼特征 | 411 | 137 keypoint × 3 的 pooled 值 |
| Class one-hot | 32 | 31 类微动作 + 1 类 class 99 |
| **总输入** | **443** | 默认 mean-pool 配置 |

### 3.3 Bag 大小统计（去 class 99 后）

| 统计 | 值 |
|---|---|
| 平均实例数 | 35.14 |
| 中位数 | 27 |
| 最大值 | 193 |
| 超过 256 的视频 | 0 |

---

## 四、网络结构

### 4.1 模型：ABMIL_iMiGUE

基于 Ilse et al. ICML 2018 的 Attention-based Deep MIL，将原版的 CNN 特征提取器替换为 MLP（输入是向量而非图像）。

```
输入: (batch, max_bag_size=256, input_dim=443)
    │
    ▼
Feature Extractor (2层 MLP):
    Linear(443, 256) → ReLU → Dropout(0.3)
    Linear(256, 256) → ReLU → Dropout(0.3)
    输出: (batch, 256, 256)
    │
    ▼
Attention Mechanism:
    Linear(256, 128) → Tanh
    Linear(128, 1)
    Mask padding positions → Softmax over instances
    输出: (batch, 1, 256) — attention weights A
    │
    ▼
Aggregation:
    Z = A × H  (加权求和)
    输出: (batch, 256)
    │
    ▼
Classifier:
    Linear(256, 1) → BCEWithLogitsLoss
    输出: (batch, 1) — bag-level Win logit
```

### 4.2 超参数配置

| 参数 | 默认值 | 小模型 |
|---|---|---|
| hidden_dim | 256 | 32 |
| attention_dim | 128 | 16 |
| dropout | 0.3 | 0.5 |
| lr | 5e-4 | 5e-4 |
| weight_decay | 1e-4 | 1e-4 |
| batch_size | 16 | 16 |
| max_bag_size | 256 | 256 |
| epochs | 80 | 80 |
| 参数量 | 212,738 | 8,257 |

### 4.3 模型变体

| 变体 | Attention 机制 | 说明 |
|---|---|---|
| ABMIL_iMiGUE | V^T × tanh(W × h_i) | 标准 attention |
| GatedABMIL_iMiGUE | (V × tanh) ⊙ (U × sigmoid) | 门控 attention |

### 4.4 训练策略

| 项 | 实现 |
|---|---|
| 损失函数 | BCEWithLogitsLoss（数值稳定） |
| 类别平衡 | WeightedRandomSampler（oversample Lose 类） |
| 优化器 | Adam |
| 学习率调度 | ReduceLROnPlateau（factor=0.5, patience=5） |
| Best model 选择 | 最高 Val Balanced Accuracy |
| 输出目录 | timestamped，每次运行独立目录 |

---

## 五、实验结果

### 5.1 Sanity Checks（管道验证）

| 实验 | 目的 | 结果 | 结论 |
|---|---|---|---|
| MNIST Toy | 验证 ABMIL 代码正确性 | TestErr=8-10%, digit-9 权重>0.9 | ✅ 代码正确 |
| 10-video overfit | 验证模型能学习 | 5W+5L, 20 epoch 内 acc=1.0 | ✅ 模型容量足够 |
| Label permutation | 验证无数据泄露 | 随机标签下 TestBalAcc ≈ 0.50 | ✅ 无泄露 |

**说明**: 三个 sanity check 全部通过，确认：
- ABMIL 代码在 PyTorch 2.6 下正常工作
- 模型有足够容量记忆小数据集
- 数据管道无泄露，随机标签无法被学习

### 5.2 Baselines

| 实验 | 方法 | TestBalAcc | TestAUC | 说明 |
|---|---|---|---|---|
| Majority baseline | 全预测 Win | 0.500 | 0.500 | 下界 |
| LR (bag mean-pool) | Logistic Regression | 0.537 | 0.582 | C=0.01, 最佳 |
| Class-only ABMIL | 仅 class one-hot, 无骨骼 | 0.500 | — | 32-dim 输入 |

**说明**:
- LR baseline 的 AUC=0.582 是目前所有实验中最高的，但仍属弱信号
- Class-only ABMIL 完全随机，说明微动作类别分布本身不含 Win/Lose 信号
- 信号必须来自骨骼坐标的变化，而非"做了哪种微动作"

### 5.3 ABMIL 主实验（5-fold Subject CV）

**配置**: hidden=256, attn=128, dropout=0.3, lr=5e-4, epochs=80, max_bag=256, 去 class 99

| Fold | Train | Val | ValBalAcc | TestBalAcc | TestAUC | TestF1 |
|---|---|---|---|---|---|---|
| 0 | 208 | 35 | 0.571 | 0.530 | 0.513 | 0.647 |
| 1 | 185 | 58 | 0.698 | 0.494 | 0.483 | 0.648 |
| 2 | 202 | 41 | 0.589 | 0.509 | 0.507 | 0.637 |
| 3 | 189 | 54 | 0.598 | **0.570** | **0.534** | **0.672** |
| 4 | 188 | 55 | 0.514 | 0.494 | 0.507 | 0.648 |
| **Mean** | | | **0.594** | **0.519** | **0.509** | **0.650** |

**典型训练曲线（Fold 0）**:

| Epoch | TrainLoss | ValLoss | ValBalAcc |
|---|---|---|---|
| 1 | 0.692 | 0.680 | 0.500 |
| 10 | 0.482 | 0.696 | **0.571** ← best |
| 20 | 0.330 | 0.699 | 0.457 |
| 40 | 0.171 | 0.863 | 0.496 |
| 80 | 0.163 | 0.890 | 0.496 |

**现象**: 训练 loss 持续下降，但 val loss 从 epoch 10 起上升 → 严重过拟合

**混淆矩阵（5-fold 累计 Test）**:

|  | 预测 Win | 预测 Lose |
|---|---|---|
| 实际 Win | TP ≈ 44 | FN ≈ 12 |
| 实际 Lose | FP ≈ 36 | TN ≈ 12 |

模型大量预测 Win（TP+FP >> TN+FN），Lose 类的 recall ≈ 0.25。

### 5.4 消融实验

#### A. 模型大小

| 配置 | 参数量 | ValBalAcc | TestBalAcc | TestAUC |
|---|---|---|---|---|
| hidden=256, attn=128 | 212,738 | 0.571 | 0.530 | 0.513 |
| hidden=32, attn=16 | 8,257 | **0.848** | 0.500 | 0.507 |

**结论**: 小模型在 val 上更好（0.848 vs 0.571），但 test 上完全随机。说明 val 和 test 分布不一致，且模型大小不是瓶颈。

#### B. Attention 机制

| 模型 | ValBalAcc | TestBalAcc | TestAUC |
|---|---|---|---|
| Standard Attention | 0.571 | 0.530 | 0.513 |
| Gated Attention | 0.699 | 0.528 | 0.547 |

**结论**: Gated attention 略好但差异不显著，信号太弱时 attention 机制选择影响不大。

#### C. Class 99 处理

| 设置 | ValBalAcc | TestBalAcc | TestAUC |
|---|---|---|---|
| 去 class 99（默认） | 0.571 | 0.530 | 0.513 |
| 含 class 99 | 0.609 | 0.512 | 0.524 |

**结论**: 去/留 class 99 差异不大。Class 99 不是噪声源，但也不提供有用信号。

#### D. Pooling 策略（最新）

**配置**: fold 0, hidden=256, attn=128, dropout=0.3, epochs=80

| Pooling | 输入维度 | ValBalAcc | TestBalAcc | TestAUC |
|---|---|---|---|---|
| mean | 443 | 0.716 | 0.542 | 0.546 |
| **max** | **443** | **0.737** | **0.569** | **0.579** |
| mean_std | 854 | 0.682 | 0.534 | 0.532 |
| mean_max | 854 | 0.645 | 0.537 | 0.551 |
| mean_std_max | 1265 | 0.592 | 0.558 | 0.583 |

**结论**:
- **max-pool 优于 mean-pool**（TestBalAcc 0.569 vs 0.542, AUC 0.579 vs 0.546）
- mean_std_max 的 AUC 最高（0.583），但 ValBalAcc 最低
- 更多统计量不总是更好：mean+std 和 mean+max 均不如纯 max
- **max-pool 能保留微动作段内的峰值激活**（如最大手-脸距离），比 mean-pool 更有意义

---

## 六、结果汇总

| # | 实验 | TestBalAcc | TestAUC | 结论 |
|---|---|---|---|---|
| 1 | Majority baseline | 0.500 | 0.500 | 下界 |
| 2 | LR (bag mean-pool) | 0.537 | 0.582 | 弱信号基线 |
| 3 | Class-only ABMIL | 0.500 | — | 类别分布无信号 |
| 4 | ABMIL 5-fold mean | 0.519 | 0.509 | 接近随机 |
| 5 | ABMIL small (32d) | 0.500 | 0.507 | Val 过拟合, Test 随机 |
| 6 | Gated Attention | 0.528 | 0.547 | 略好，不显著 |
| 7 | 含 class 99 | 0.512 | 0.524 | 无显著影响 |
| 8 | Label permutation | 0.500 | — | 无泄露 |
| 9 | **ABMIL max-pool** | **0.569** | **0.579** | **当前最佳** |
| 10 | ABMIL mean_std_max | 0.558 | 0.583 | AUC 最高 |

---

## 七、关键发现

### 7.1 确认的事实

1. **管道正确**: Sanity checks 全部通过（MNIST、过拟合、标签置换）
2. **信号极弱**: 所有实验的 TestBalAcc 在 0.50-0.57 之间，AUC 在 0.50-0.58 之间
3. **Val-Test 不一致**: ValBalAcc 可达 0.85，但 TestBalAcc 只有 0.50，说明 val fold 不代表 test 分布
4. **模型偏向 Win 类**: 混淆矩阵显示 TP+FP >> TN+FN，Lose 类 recall ≈ 0.25
5. **Class-only 无信号**: 仅用微动作类别标签（不含骨骼）完全随机

### 7.2 Pooling 策略的影响

- **max-pool 比 mean-pool 好**: TestBalAcc 0.569 vs 0.542, AUC 0.579 vs 0.546
- 这初步验证了"mean-pool 丢失信息"的假设
- 但提升幅度有限（+2.7% BalAcc），不是根本性瓶颈

### 7.3 根本瓶颈分析

信号弱的原因是多因素叠加：

| 因素 | 影响 | 可改进性 |
|---|---|---|
| Win/Lose 是极弱的情感代理标签 | 根本性 | 低（标签固定） |
| Mean-pool 丢失时序信息 | 重要 | 中（已验证 max-pool 更好） |
| OpenPose 坐标区分力不足 | 重要 | 中（可用相对距离特征） |
| 训练样本少（~200） | 限制 | 低（数据集固定） |
| Val-Test 分布不一致 | 干扰评估 | 中（可增大 val fold 或用全 train） |

---

## 八、下一步方向

### 8.1 短期（本周）

| 方向 | 具体做法 | 预期 |
|---|---|---|
| **相对距离特征** | 手-脸距离、手-手距离、手-身体距离 | 直接编码"自我接触"先验 |
| **运动特征** | 帧间差分的 mean/std（速度）、加速度 | 捕捉动态信息 |
| **更好的 pooling** | max-pool（已验证更好）+ 相对距离 | 组合使用 |
| **5-fold 集成** | 用全部 5 折的预测做集成 | 减少单折方差 |

### 8.2 中期（下周）

| 方向 | 具体做法 | 预期 |
|---|---|---|
| **心理先验 soft constraint** | 自接触先验作为 attention 正则 | 改善证据定位 |
| **时序模型** | 用 LSTM/Transformer 替代 mean-pool | 保留完整时序 |
| **RGB 特征** | VideoMAE embedding 补充/替代骨骼 | 更强的视觉信号 |

### 8.3 决策标准

| 结果 | 决策 |
|---|---|
| TestBalAcc ≥ 0.62 或 AUC ≥ 0.65 | 继续骨骼路线 + 心理先验 |
| TestBalAcc 0.58-0.62 | 继续，但论文框架为"弱代理 + 证据定位" |
| TestBalAcc ≤ 0.58 且 AUC ≤ 0.60 | 转向 RGB+VideoMAE 或改变论文故事 |

当前最佳（max-pool, TestBalAcc=0.569, AUC=0.579）处于第二档边缘，仍有改进空间。

---

## 九、代码与输出

### 9.1 Git 提交历史

```
fba16b9 docs: update experiment log with 5-fold + ablation results
5113195 chore: remove diagnosis doc from repo
cbf75dd fix: baseline pipeline overhaul + diagnosis document
bf4ce1f docs: add experiment log for iMiGUE ABMIL baseline
072a698 fix: feature normalization + proper val split
d4d611b fix: class imbalance handling + empty skeleton handling
14c7e70 perf: use numpy skeleton format + preprocessing script
67c358d feat: iMiGUE ABMIL dataloader + model + training script
9ee22c5 feat: MNIST toy experiment logs (attention + gated_attention)
d250ad5 fix: PyTorch 2.6 + NumPy 1.24+ compatibility
```

### 9.2 文件结构

```
/home/xingke/MIL/AttentionDeepMIL/
├── model.py              # 原始 ABMIL (MNIST CNN 版)
├── main.py               # 原始 MNIST 训练脚本
├── dataloader.py          # 原始 MNIST bag dataloader
├── mnist_bags_loader.py   # 原始 MNIST bag loader
├── model_imigue.py        # iMiGUE ABMIL (MLP 版, BCEWithLogitsLoss)
├── imigue_dataset.py      # iMiGUE MIL bag dataset (含 class99 过滤, 零帧 mask)
├── train_imigue.py        # iMiGUE 训练脚本 (timestamped 输出, 逐视频预测)
├── convert_xlsx_to_npy.py # xlsx→npy 预处理
├── EXPERIMENT_LOG.md      # 实验日志（服务器本地）
├── IMIGUE_BASELINE_DIAGNOSIS_20260506.md  # 诊断文档（服务器本地）
└── outputs/               # 所有实验输出（timestamped 目录, 不进 git）
```

### 9.3 输出目录结构

每次运行生成独立目录，如 `outputs/attention_20260507_030525_seed42_fold0_no99_bag256/`：

```
├── results.json          # 结构化结果（参数、指标、混淆矩阵）
├── results.txt           # 人类可读摘要
├── test_predictions.json # 逐视频预测（video_id, y_true, y_pred, y_prob）
├── test_attention.npz    # attention 权重 + 视频 ID + 标签
└── best_model.pth        # 模型 checkpoint
```
