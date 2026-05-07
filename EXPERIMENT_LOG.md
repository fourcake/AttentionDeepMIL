# iMiGUE ABMIL Experiment Log

**日期**: 2026-05-06 ~ 2026-05-07
**分支**: dev/imigue-baseline
**代码**: /home/xingke/MIL/AttentionDeepMIL
**数据**: /data-store/xingke/iMiGUE

---

## 1. 实验环境

| 项 | 值 |
|---|---|
| GPU | 8x A800 80GB |
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu118 |
| 输入特征 | 443-dim = 411 (OpenPose 137-keypoint × 3, mean-pool) + 32 (class one-hot) |
| 标签 | Win (1) / Lose (0), 二分类 |
| Bag | 视频, 包含多个微动作实例 |
| Instance | 微动作段 (class, start_frame, end_frame) |

## 2. 数据划分

| Split | 视频数 | Win | Lose | 说明 |
|---|---|---|---|---|
| Train | ~188-208 | ~150-166 | ~38-44 | 按被试 5-fold CV |
| Val | ~35-58 | ~26-43 | ~9-15 | 被试不重叠 |
| Test | 104 | 56 | 48 | 完全独立 |

排除: 视频 347, 348 (空骨骼文件)
默认: 排除 class 99 (非微动作, 31% 实例)

## 3. 网络结构

ABMIL (Ilse et al. ICML 2018) 适配版:

输入 (batch, max_bag, 443)
  → MLP: Linear(443,256)→ReLU→Dropout→Linear(256,256)→ReLU→Dropout
  → Attention: Linear(256,128)→Tanh→Linear(256,1), Mask→Softmax
  → Aggregation: Z = A × H (加权求和)
  → Classifier: Linear(256,1)→BCEWithLogitsLoss
  → 输出: bag-level Win 概率 + attention 权重

参数量: 212,738 (hidden=256) 或 8,257 (hidden=32)

## 4. 实验结果

### 4.1 Sanity Checks

| 实验 | 结果 | 结论 |
|---|---|---|
| MNIST Toy | TestErr=8-10%, digit-9 权重>0.9 | ABMIL 代码正确 |
| 10-video overfit | 20 epoch 内 acc=1.0 | 模型能记忆小数据集 |
| Label permutation | TestBalAcc ≈ 0.50 | 无数据泄露 |

### 4.2 Baselines

| 实验 | TestBalAcc | TestAUC | 说明 |
|---|---|---|---|
| Logistic Regression (bag mean-pool) | 0.537 | 0.582 | C=0.01, 最佳 |
| Class-only ABMIL (无骨骼) | 0.500 | — | 32-dim class one-hot |

### 4.3 ABMIL 主实验

| 实验 | ValBalAcc | TestBalAcc | TestAUC | TestF1 |
|---|---|---|---|---|
| 5-fold mean | 0.594 | 0.519 | 0.509 | 0.650 |
| Fold 0 | 0.571 | 0.530 | 0.513 | 0.647 |
| Fold 1 | 0.698 | 0.494 | 0.483 | 0.648 |
| Fold 2 | 0.589 | 0.509 | 0.507 | 0.637 |
| Fold 3 | 0.598 | 0.570 | 0.534 | 0.672 |
| Fold 4 | 0.514 | 0.494 | 0.507 | 0.648 |

配置: hidden=256, attn=128, dropout=0.3, lr=5e-4, epochs=80

### 4.4 Ablations

| 实验 | ValBalAcc | TestBalAcc | TestAUC |
|---|---|---|---|
| Small model (hidden=32, dropout=0.5) | 0.848 | 0.500 | 0.507 |
| Gated Attention | 0.699 | 0.528 | 0.547 |
| With Class 99 (含非微动作) | 0.609 | 0.512 | 0.524 |

### 4.5 关键观察

1. 所有实验 TestBalAcc ≈ 0.50-0.57, AUC ≈ 0.50-0.55
2. 模型大量预测 Win (TP+FP >> TN+FN)
3. Val 和 Test 严重不一致 (Val=0.85 但 Test=0.50)
4. Class-only ABMIL 完全随机 → 微动作分布无 Win/Lose 信号
5. Label permutation ≈ 0.50 → 无数据泄露
6. 10-video overfit = 1.0 → 模型容量足够

## 5. 诊断

Win/Lose 作为弱监督标签, 在 mean-pooled 骨骼特征上信号极弱。

可能原因:
1. Mean-pool 丢失时序信息 (速度、轨迹)
2. Win/Lose 是极弱的情感代理 (心理状态与比赛结果关联弱)
3. OpenPose 坐标本身区分力不足 (需要相对距离、变化率)

## 6. 下一步

1. 运动特征: 速度 (帧间差分)、手-脸距离、手-手距离
2. 更好的 pooling: mean+std+max concat, 或时序模型 (LSTM/Transformer)
3. RGB 特征: VideoMAE embedding 替代/补充骨骼
4. 心理先验: 自接触先验 (手-脸距离阈值) 作为 attention 约束
