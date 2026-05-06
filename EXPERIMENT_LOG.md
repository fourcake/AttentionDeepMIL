# iMiGUE ABMIL Baseline Experiment Log

**日期**: 2026-05-06
**分支**: dev/imigue-baseline
**目标**: 跑通 ABMIL 在 iMiGUE 上的 baseline，验证 Win/Lose 预测是否可行

---

## 1. 环境

| 项目 | 配置 |
|---|---|
| 服务器 | xingke@10.50.0.31 |
| GPU | 8x A800 80GB |
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu118 |
| 代码目录 | /home/xingke/MIL/AttentionDeepMIL |
| 数据目录 | /data-store/xingke/iMiGUE |

## 2. 数据概况

### 2.1 官方划分

| Split | 视频数 | Win | Lose | 被试数 |
|---|---|---|---|---|
| Train | 245 | 192 | 53 | 37 |
| Val (官方) | 10 | 10 | 0 | 5 (与 Train 重叠 5 个) |
| Test | 104 | 56 | 48 | 35 (与 Train/Test 完全独立) |

**问题**: 官方 Val 集全为 Win 标签，且与 Train 有被试重叠，不可用。

### 2.2 实际使用的 Val 划分

从 Train 中按被试划出 5-fold CV 的第 0 折作为 Val：

| Split | 视频数 | Win | Lose |
|---|---|---|---|
| Train (新) | 210 | 166 | 44 |
| Val (新) | 35 | 26 | 9 |
| Test | 104 | 56 | 48 |

### 2.3 骨骼数据

- 来源: OpenPose 137-keypoint 全身骨骼
- 格式: xlsx (每行=1帧, 411列 = 137 keypoint × 3)
- 已转换为 numpy .npy 格式加速加载 (359 个文件)
- **数据问题**: 视频 347, 348 的 xlsx 为空文件 (0行0列)

### 2.4 特征工程

每个微动作实例:
1. 从骨骼 xlsx 按 `start_frame:end_frame` 切片 → (T_inst, 411)
2. 过滤全零帧
3. Z-normalize (per-video, 非零帧的 mean/std)
4. Mean-pool over T_inst → (411,)
5. 拼接 class one-hot (32维) → (443,)
6. Pad 到 max_bag_size=64, 用 attention mask 屏蔽 padding

## 3. 实验结果

### 3.1 MNIST Toy (验证 ABMIL 代码正确性)

| 模型 | Train Loss | Train Error | Test Loss | Test Error |
|---|---|---|---|---|
| Attention | 0.66 → 0.0001 | 37.5% → 0% | 0.66 | 10% |
| Gated Attention | 0.69 → 0.0009 | 37.5% → 0% | 0.35 | 8% |

Attention 权重正确集中在 digit-9 实例上 (权重 >0.9)。

### 3.2 Bag-Level Logistic Regression Baseline

直接将 bag 内所有实例 mean-pool 成一个 443 维向量，用 Logistic Regression 分类:

| C | ValBalAcc | TestBalAcc | TestAUC |
|---|---|---|---|
| 0.01 | 0.6816 | 0.5372 | 0.5026 |
| 0.1 | 0.5662 | 0.5060 | 0.4985 |
| 1.0 | 0.6795 | 0.5119 | 0.5618 |
| 10.0 | 0.4957 | 0.5357 | 0.5818 |

**结论**: 信号极弱。Test balanced accuracy ~0.54, AUC ~0.58，仅略高于随机 (0.50)。

### 3.3 ABMIL (Attention Model)

配置: hidden_dim=128, attention_dim=64, dropout=0.3, lr=5e-4, batch_size=16

| Epoch | TrainLoss | TrainErr | ValLoss | ValErr | ValBalAcc |
|---|---|---|---|---|---|
| 1 | 0.6864 | 49.1% | 0.6972 | 47.9% | 0.5171 |
| 10 | 0.6092 | 31.3% | 0.6689 | 54.2% | 0.4594 |
| 20 | 0.4824 | 20.5% | 0.9123 | 56.3% | 0.4038 |
| 30 | 0.3942 | 15.2% | 1.2585 | 49.3% | 0.5299 |
| 60 | 0.2756 | 6.7% | 1.5730 | 64.6% | 0.3269 |

**结论**: 严重过拟合。训练误差持续下降，但验证损失从 epoch 10 开始上升。
模型参数 81,794，训练样本仅 210 — 参数量/样本量比过高。

### 3.4 早期失败尝试

| 尝试 | 问题 | 结果 |
|---|---|---|
| 未修复的原始 ABMIL 代码 | PyTorch 2.6 不兼容 (`loss.data[0]`, `np.int`) | 崩溃 |
| 使用官方 Val 集 | 全 Win 标签，ValBalAcc=1.0 无意义 | Best model 选在 epoch 1 |
| pos_weight=0.276 (N_lose/N_win) | 少数类权重反而降低 | 模型全部预测 Win，TestBalAcc=0.50 |
| 未 normalize 特征 | 原始骨骼坐标方差大 | 训练不收敛 |

## 4. 关键发现

1. **Win/Lose 是极弱的代理标签**: 即使最简单的 logistic regression 也几乎无法区分，说明微动作与比赛结果之间的关联非常微弱。
2. **Mean-pool 丢失时序信息**: 每个 2.5 秒的微动作被压缩成一个静态向量，丢失了运动轨迹的动态特征。
3. **过拟合是主要瓶颈**: 210 个训练样本对 82K 参数的模型来说太少，需要更强的正则化或更简单的模型。
4. **Class 99 (非微动作) 占比大**: 5715/18498 = 31% 的实例是 class 99，可能稀释了有效信号。

## 5. 下一步方向

### 5.1 短期 (本周)

- [ ] 排除 class 99 实例，只用 31 类微动作
- [ ] 尝试更简单的模型: 1 层 MLP (32 dims) + dropout 0.5
- [ ] 尝试不同的 pooling: max-pool, mean+max concat, std-pool
- [ ] 5-fold subject-CV 获得更可靠的评估
- [ ] 特征工程: 关键点间距离 (手-脸距离等) 作为额外特征

### 5.2 中期 (下周)

- [ ] 加入心理先验软约束 (自我接触、重复性)
- [ ] 尝试 RGB 特征 (VideoMAE) 替代或补充骨骼特征
- [ ] 分析哪些微动作类别最具区分力

### 5.3 长期

- [ ] 如果骨骼特征确实信号太弱，考虑转向 RGB + VideoMAE 路线
- [ ] 跨数据集迁移 (iMiGUE → SMG)

## 6. Git 提交记录

```
072a698 fix: feature normalization + proper val split
d4d611b fix: class imbalance handling + empty skeleton handling
14c7e70 perf: use numpy skeleton format + preprocessing script
67c358d feat: iMiGUE ABMIL dataloader + model + training script
9ee22c5 feat: MNIST toy experiment logs (attention + gated_attention)
d250ad5 fix: PyTorch 2.6 + NumPy 1.24+ compatibility
```

## 7. 文件结构

```
/home/xingke/MIL/AttentionDeepMIL/
├── .gitignore
├── model.py              # 原始 ABMIL (MNIST CNN 版)
├── main.py               # 原始 MNIST 训练脚本
├── dataloader.py          # 原始 MNIST bag dataloader
├── mnist_bags_loader.py   # 原始 MNIST bag loader
├── model_imigue.py        # iMiGUE ABMIL (MLP 版)
├── imigue_dataset.py      # iMiGUE MIL bag dataset
├── train_imigue.py        # iMiGUE 训练脚本
├── convert_xlsx_to_npy.py # xlsx→npy 预处理
├── logs/                  # MNIST toy 日志
└── outputs/               # iMiGUE 训练结果
```
