# iMiGUE ABMIL Baseline Diagnosis and Next Steps

Date: 2026-05-06
Remote repo: `/home/xingke/MIL/AttentionDeepMIL`
Data: `/data-store/xingke/iMiGUE`

## 1. Bottom line

The current bad result should not yet be interpreted as "iMiGUE skeleton has no signal".
The first conclusion is more basic: the present baseline has several evaluation and data-pipeline issues that can easily suppress or distort the result.

My current diagnosis:

1. The code is good enough to prove that the ABMIL training loop can run.
2. The current iMiGUE result is not yet a trustworthy scientific negative result.
3. Before adding psychological priors or Agent logic, fix the dataset/evaluation pipeline and run a small set of controlled baselines.
4. If the fixed, leakage-free, no-truncation skeleton/class baselines still stay near `Test balanced accuracy <= 0.58` and `AUC <= 0.60`, then treat Win/Lose as a weak proxy label and pivot the paper toward attribution reliability, prior validation, or RGB/multimodal evidence rather than raw skeleton-only classification.

## 2. Code issues found

### 2.1 `--include_class99` does not do what it says

Location: `/home/xingke/MIL/AttentionDeepMIL/train_imigue.py:160`

The parser defines:

```python
parser.add_argument('--include_class99', action='store_true')
```

but when `args.include_class99` is false, the code only reads the CSV and prints a message. It never filters class 99.

Current behavior: class 99 is always included.

Why this matters:

- Class 99 has 5,715 instances, about 31% of all annotations.
- It is not necessarily bad to include it, but it must be an explicit ablation.
- The current experiment log says "`include_class99=False`", but the actual dataset still contains class 99.

Fix:

- Add `include_class99` to `iMiGUEDataset`.
- Filter `sub = sub[sub["class"] != 99]` when `include_class99=False`.
- Report bag-size statistics after filtering.
- Keep a separate `class99_as_soft_negative` experiment later; do not mix it into the first baseline.

### 2.2 All-zero frame filtering is broken by normalization order

Location: `/home/xingke/MIL/AttentionDeepMIL/imigue_dataset.py:91`

The dataset first normalizes the full skeleton array:

```python
skel = (skel - skel_mean) / skel_std
```

Then it tries to detect zero frames inside each segment:

```python
valid = seg.any(axis=1)
```

After normalization, originally all-zero OpenPose failure frames are no longer zero. They become `(-mean / std)`, so `seg.any(axis=1)` becomes true and the failure frames are kept.

Why this matters:

- The original kickoff document already identified leading all-zero frames and scattered OpenPose failures.
- The code intends to filter them, but currently does not after normalization.
- This can inject artificial "negative mean skeleton" patterns into instance features.

Fix:

- Compute a raw valid-frame mask before normalization: `raw_valid = skel.any(axis=1)`.
- Use that raw mask when filtering each segment.
- Or filter zero frames before applying z-score.
- Keep confidence channels separate from coordinate channels if possible.

### 2.3 Chronological truncation silently drops later evidence

Location: `/home/xingke/MIL/AttentionDeepMIL/imigue_dataset.py:136`

The dataset sorts instances by `start_frame`, then if a bag has more than `max_bag_size`, it keeps only the first 64:

```python
features = features[:self.max_bag_size]
```

Measured with the CSV:

- With all instances: 97 videos have more than 64 instances.
- Without class 99: 47 videos still have more than 64 micro-action instances.
- Max bag size is 282 with all instances and 193 without class 99.

Why this matters:

- ABMIL is supposed to find sparse evidence anywhere in the video.
- Current truncation biases the model toward the beginning of videos.
- Prefix experiments are a later research question; the baseline should not accidentally become a prefix model.

Fix:

- For baseline, use dynamic padding to the max bag length in the batch.
- Or set `max_bag_size=256` and only drop the few videos above that, with a logged warning.
- At minimum, log `num_instances_raw`, `num_instances_used`, and truncation ratio per split.

### 2.4 `--cv_folds` is ignored and only fold 0 is used

Location: `/home/xingke/MIL/AttentionDeepMIL/train_imigue.py:128` and `/home/xingke/MIL/AttentionDeepMIL/train_imigue.py:150`

The CLI exposes `--cv_folds`, but the script always builds 5 folds and always uses fold 0:

```python
folds = build_subject_cv_folds(train_ids, csv_path, n_folds=5)
new_train_ids, new_val_ids = folds[0]
```

Why this matters:

- One fold is too noisy for a 245-video training pool.
- The fold construction is subject-disjoint, but not label-stratified by subject.
- A lucky or unlucky fold can select a misleading epoch.

Fix:

- Add `--cv_fold`.
- Run all 5 folds for validation and choose hyperparameters by mean validation balanced accuracy.
- Keep the official 104-video test set untouched and evaluate it only after hyperparameters are fixed.

### 2.5 Result files can become stale or ambiguous

Location: `/home/xingke/MIL/AttentionDeepMIL/outputs/results_attention.txt`

The current result file reports `Best Val Balanced Accuracy: 1.0000`, while the experiment log reports the more recent subject-split run with poor validation balanced accuracy.

Why this matters:

- Reusing `outputs/best_attention.pth` and `outputs/results_attention.txt` across experiments makes it hard to know which code/config produced which number.

Fix:

- Write each run to a timestamped directory, for example `outputs/imigue_abmil_YYYYMMDD_HHMMSS_seed42_fold0_no99_bag256/`.
- Save config, git commit, fold IDs, predictions, probabilities, labels, video IDs, and attention weights.

### 2.6 Model is larger than needed for first signal detection

Location: `/home/xingke/MIL/AttentionDeepMIL/model_imigue.py:23`

The current default model is a 2-layer MLP plus attention. With `hidden_dim=256`, it has about 213K trainable parameters. The experiment log also tried a smaller 82K model, but even that is large for 192-210 training videos.

This is not a correctness bug, but it explains the observed pattern:

- Training error decreases.
- Validation loss rises.
- Validation balanced accuracy is unstable or declines.

Fix:

- First run a deliberately small model: `hidden_dim=32`, `attention_dim=16`, dropout `0.3-0.5`.
- Add a purely linear/logistic bag baseline and class-count baseline.
- Use `BCEWithLogitsLoss` instead of `Sigmoid + BCELoss` for numerical stability.

## 3. Data and evaluation observations

Official split:

| Split | Videos | Subjects | Win | Lose | Notes |
|---|---:|---:|---:|---:|---|
| Train | 245 | 37 | 192 | 53 | imbalanced |
| Official val | 10 | 5 | 10 | 0 | unusable |
| Test | 104 | 35 | 56 | 48 | subject-disjoint and balanced enough |

Subject overlap:

| Pair | Overlap |
|---|---:|
| Train / official val | 5 subjects |
| Train / test | 0 subjects |
| Official val / test | 0 subjects |

Bag-size statistics over official train + test:

| Setting | Mean | Median | Max | Videos >64 |
|---|---:|---:|---:|---:|
| All instances | 50.78 | 38 | 282 | 97 |
| Drop class 99 | 35.14 | 27 | 193 | 47 |

Class 99 is common in both labels:

| Label | Mean class99/video | Median | Nonzero fraction |
|---|---:|---:|---:|
| Win | 16.26 | 9 | 0.891 |
| Lose | 15.06 | 6 | 0.881 |

Interpretation:

- Class 99 count alone is unlikely to be a strong label discriminator.
- Including class 99 may still help as an explicit negative/evidence-suppression prior, but it should not be silently included in the first ABMIL baseline.
- The current truncation interacts badly with class 99 because early non-MG segments can consume the 64-instance budget.

Empty skeleton files:

| Video | Split | Label | Subject | Instances |
|---:|---|---|---:|---:|
| 347 | train | Win | 48 | 36 |
| 348 | train | Win | 30 | 43 |

These two videos should be excluded from skeleton-only training unless RGB features are used.

## 4. What the bad result probably means

The poor result likely comes from several stacked causes:

1. Win/Lose is a weak proxy for affect.
2. Mean-pooled skeleton coordinates are a weak representation of motion.
3. Current zero-frame filtering is ineffective after normalization.
4. Current truncation removes later evidence from many videos.
5. The model is too flexible relative to the number of training videos.
6. Validation is a single unstratified subject fold, so model selection is noisy.

The important distinction:

- It is plausible that iMiGUE Win/Lose is genuinely weak for skeleton-only affect prediction.
- It is premature to conclude that before fixing the pipeline and running controlled baselines.

## 5. Immediate next steps

### Step 0: Fix the baseline pipeline first

Implement these before any new research method:

1. Make `include_class99` real.
2. Preserve raw zero-frame masks before normalization.
3. Remove chronological first-64 truncation.
4. Add `--cv_fold` and run all 5 subject folds.
5. Exclude empty skeleton videos 347 and 348 from skeleton-only experiments.
6. Save per-run outputs in timestamped directories.
7. Save per-video predictions and probabilities, not just aggregate metrics.

Expected time: 0.5-1 day.

### Step 1: Run sanity baselines

Run these in this order:

| Baseline | Purpose |
|---|---|
| Majority / balanced random | metric sanity |
| Class-count logistic regression | tests whether action distribution predicts Win/Lose |
| Class-sequence ABMIL only | tests whether annotation labels carry signal |
| Skeleton-only logistic / ABMIL | tests whether pose adds signal |
| Skeleton + class ABMIL | current intended baseline |
| Label permutation test | verifies no leakage |
| 10-video overfit test | verifies training capacity |

Minimum outputs to save:

- balanced accuracy
- AUC
- F1
- confusion matrix
- prediction probabilities
- video IDs
- fold IDs

Do not move to psychological priors until these are clean.

### Step 2: Replace mean-only instance features

Mean pooling over 2.5 seconds loses almost all motion information. Add simple, cheap motion features before using VideoMAE:

1. Mean + std + min + max of selected keypoints.
2. Velocity mean/std from frame differences.
3. Confidence-weighted pooling.
4. Hand-face distance: wrist to nose/face landmarks.
5. Hand-body distance: wrist to neck/torso/hip.
6. Hand-hand distance and crossing/symmetry features.
7. Segment duration and normalized timestamp.
8. Repetition features: same class count in local window and whole video.

This is more aligned with the research claim than raw 411-dimensional coordinate means.

### Step 3: Only then add psychological priors

After the fixed baseline is stable, add priors as both features and attention regularizers:

| Prior | First implementation |
|---|---|
| Self-contact | min wrist-to-face distance per segment |
| Repetition | local same-class frequency + trajectory similarity |
| Non-task motion | defer until VAD is available; do not block W1 |
| Class 99 | attention suppression / soft negative prior |

Keep these ablations separate:

1. no prior
2. random prior
3. self-contact only
4. repetition only
5. self-contact + repetition
6. class99 suppression

The core claim should be evidence quality improvement, not necessarily raw accuracy.

### Step 4: Decide whether to continue skeleton-only

Use this decision rule after the fixed baselines:

| Result | Decision |
|---|---|
| Test balanced acc >= 0.62 or AUC >= 0.65 | Continue skeleton-only + priors |
| Test balanced acc 0.58-0.62 | Continue, but frame as weak proxy / attribution study |
| Test balanced acc <= 0.58 and AUC <= 0.60 | Add RGB/VideoMAE or shift away from classification as the main claim |

If skeleton-only remains weak, the paper can still be viable, but the story changes:

- "Win/Lose is a weak affect proxy."
- "Raw micro-action classification is insufficient."
- "Psychological priors improve evidence localization more than video-level accuracy."
- "Counterfactual evidence evaluation reveals limits of weak supervision."

## 6. Recommended experiment order for the next 3 days

Day 1:

1. Patch dataset and training script issues.
2. Run a 10-video overfit test.
3. Run class-count logistic regression and class-only ABMIL.

Day 2:

1. Run fixed no99 skeleton+class ABMIL on 5 subject folds.
2. Save per-fold predictions and confusion matrices.
3. Compare `max_bag_size=64`, dynamic/full, and `max_bag_size=256`.

Day 3:

1. Add motion/distance features.
2. Run skeleton+class+motion ABMIL.
3. Inspect top-attention segments qualitatively for 10 Win and 10 Lose videos.

Only after this should you spend time on LangGraph Agent, VAD, or VideoMAE.

## 7. Paper-strategy implication

Do not pitch the current result as "ABMIL failed".

A safer formulation is:

> The first iMiGUE skeleton-only ABMIL pilot exposes the difficulty of weak proxy affect labels: naive mean-pooled pose features overfit and do not generalize. We therefore first establish a leakage-free, no-truncation MIL protocol and then evaluate whether psychologically grounded priors improve evidence localization under this weak supervision.

This keeps the research direction alive while being honest about the current result.

