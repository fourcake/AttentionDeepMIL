"""iMiGUE dataset for Multiple Instance Learning.

Each video is a bag containing multiple micro-action instances.
Bag label: Win (1) / Lose (0) from CSV column 'win_or_lose'.
Instance features: mean-pooled OpenPose 137-keypoint skeleton (411-dim)
                   + class one-hot (32-dim) = 443-dim.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class iMiGUEDataset(Dataset):
    def __init__(self, csv_path, skeleton_root, video_ids, max_bag_size=256,
                 num_classes=32, filter_zero_frames=True, cache_skeletons=False,
                 skeleton_npy_dir=None, normalize_features=True,
                 include_class99=True, exclude_empty_skeletons=True):
        """
        Args:
            csv_path: Path to labels_20200831.csv
            skeleton_root: Path to mg_skeleton_only/ (fallback xlsx)
            video_ids: List of video IDs for this split
            max_bag_size: Max number of instances per bag (pad to this)
            num_classes: Number of class bins (31 MG + 1 for class 99)
            filter_zero_frames: Filter all-zero frames before mean pooling
            cache_skeletons: Cache skeleton data in memory
            skeleton_npy_dir: Path to mg_skeleton_npy/ (fast numpy format, preferred)
            normalize_features: Z-normalize skeleton features per-video
            include_class99: If False, filter out class 99 (non-MG) instances
            exclude_empty_skeletons: If True, exclude videos 347, 348 (empty skeleton files)
        """
        self.skeleton_root = skeleton_root
        self.skeleton_npy_dir = skeleton_npy_dir
        self.video_ids = list(video_ids)
        self.max_bag_size = max_bag_size
        self.num_classes = num_classes
        self.filter_zero_frames = filter_zero_frames
        self.cache_skeletons = cache_skeletons
        self.normalize_features = normalize_features
        self.include_class99 = include_class99
        self._skeleton_cache = {}

        # Videos with empty skeleton files (0-row xlsx)
        self.empty_skeleton_videos = {347, 348}

        # Read and group annotations
        df = pd.read_csv(csv_path)
        self.bags = {}
        skipped_empty = 0
        skipped_no_instances = 0

        for vid in self.video_ids:
            # Optionally exclude videos with empty skeleton files
            if exclude_empty_skeletons and vid in self.empty_skeleton_videos:
                skipped_empty += 1
                continue

            sub = df[df['video_id'] == vid].sort_values('start_frame')

            # Filter class 99 if requested
            if not self.include_class99:
                sub = sub[sub['class'] != 99]

            if len(sub) == 0:
                skipped_no_instances += 1
                continue

            self.bags[vid] = {
                'instances': sub[['class', 'start_frame', 'end_frame']].values,
                'label': 1 if sub.iloc[0]['win_or_lose'] == 'Win' else 0,
                'subject_id': sub.iloc[0]['subject_id'],
                'num_instances_raw': len(sub),
            }

        self.video_ids = list(self.bags.keys())

        # Log statistics
        if skipped_empty > 0:
            print(f'  Skipped {skipped_empty} videos with empty skeleton files')
        if skipped_no_instances > 0:
            print(f'  Skipped {skipped_no_instances} videos with no instances after filtering')
        if not self.include_class99:
            n_filtered = sum(1 for _ in df[df['class'] == 99]['video_id'].unique()
                             if _ in video_ids)
            print(f'  Class 99 filtered: {n_filtered} videos had class 99 instances removed')

        # Report bag size distribution
        bag_sizes = [b['num_instances_raw'] for b in self.bags.values()]
        if bag_sizes:
            over_limit = sum(1 for s in bag_sizes if s > self.max_bag_size)
            print(f'  Bag sizes: mean={np.mean(bag_sizes):.1f}, median={np.median(bag_sizes):.0f}, '
                  f'max={max(bag_sizes)}, over {self.max_bag_size}={over_limit}')

    def __len__(self):
        return len(self.video_ids)

    def _load_skeleton(self, vid):
        if self.cache_skeletons and vid in self._skeleton_cache:
            return self._skeleton_cache[vid]

        # Try fast numpy format first
        if self.skeleton_npy_dir:
            npy_path = os.path.join(self.skeleton_npy_dir, f'{vid:04d}.npy')
            if os.path.exists(npy_path):
                skel = np.load(npy_path)  # (T, 411) float32
            else:
                # Fallback to xlsx
                path = os.path.join(self.skeleton_root, f'{vid:04d}', f'{vid:04d}_2.xlsx')
                skel = pd.read_excel(path, header=None).values
        else:
            path = os.path.join(self.skeleton_root, f'{vid:04d}', f'{vid:04d}_2.xlsx')
            skel = pd.read_excel(path, header=None).values  # (T, 411)

        # Handle empty skeleton files (videos 347, 348 have empty xlsx)
        if skel.ndim < 2 or skel.shape[0] == 0 or skel.shape[1] == 0:
            skel = np.zeros((1, 411), dtype=np.float32)

        if self.cache_skeletons:
            self._skeleton_cache[vid] = skel
        return skel

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        bag = self.bags[vid]

        skel = self._load_skeleton(vid)  # (T, 411)
        instances = bag['instances']     # (N, 3): class, start_frame, end_frame

        skel_dim = skel.shape[1] if skel.ndim == 2 else 411
        feat_dim = skel_dim + self.num_classes  # typically 443

        # Compute raw zero-frame mask BEFORE normalization.
        # Originally all-zero OpenPose failure frames become (-mean/std) after
        # z-score, which makes seg.any() always True and disables filtering.
        raw_valid_mask = skel.any(axis=1) if skel.ndim == 2 else np.ones(skel.shape[0], dtype=bool)

        # Per-video normalization: z-score on non-zero skeleton data
        if self.normalize_features and skel.shape[0] > 1:
            if raw_valid_mask.sum() > 1:
                skel_mean = skel[raw_valid_mask].mean(axis=0)
                skel_std = skel[raw_valid_mask].std(axis=0)
                skel_std[skel_std < 1e-6] = 1.0  # avoid division by zero
                skel = (skel - skel_mean) / skel_std

        features = []
        for cls, sf, ef in instances:
            sf, ef = int(sf), int(ef)

            # Clamp to valid range
            sf = max(0, min(sf, skel.shape[0] - 1))
            ef = max(sf + 1, min(ef, skel.shape[0]))
            seg = skel[sf:ef, :]  # (T_inst, skel_dim)

            # Use the RAW mask (pre-normalization) to filter OpenPose failure frames.
            # After normalization, zero frames become non-zero (-mean/std),
            # so checking seg.any() would incorrectly keep them.
            if self.filter_zero_frames and seg.shape[0] > 0:
                seg_valid = raw_valid_mask[sf:ef]
                if seg_valid.any():
                    seg = seg[seg_valid]

            if seg.shape[0] == 0:
                # Empty segment: use zero vector
                skel_feat = np.zeros(skel_dim, dtype=np.float32)
            else:
                skel_feat = seg.mean(axis=0).astype(np.float32)

            # Replace NaN with 0
            skel_feat = np.nan_to_num(skel_feat, nan=0.0)

            # Class one-hot: classes 1-31 -> bins 0-30, class 99 -> bin 31
            class_oh = np.zeros(self.num_classes, dtype=np.float32)
            bin_idx = (cls - 1) if 1 <= cls < self.num_classes else (self.num_classes - 1)
            class_oh[int(bin_idx)] = 1.0
            feat = np.concatenate([skel_feat, class_oh])  # (feat_dim,)

            assert feat.shape[0] == feat_dim, f'Feature dim mismatch: {feat.shape[0]} != {feat_dim}'
            features.append(feat)

        # Handle empty bag (no instances)
        if len(features) == 0:
            features.append(np.zeros(feat_dim, dtype=np.float32))

        # Truncate if too many instances (with warning)
        N = len(features)
        truncated = False
        if N > self.max_bag_size:
            features = features[:self.max_bag_size]
            N = self.max_bag_size
            truncated = True

        # Pad to max_bag_size
        if N < self.max_bag_size:
            features.extend([np.zeros(feat_dim, dtype=np.float32)] * (self.max_bag_size - N))

        mask = np.zeros(self.max_bag_size, dtype=bool)
        mask[:N] = True

        return {
            'features': torch.FloatTensor(np.stack(features)),  # (max_bag, 443)
            'mask': torch.BoolTensor(mask),                     # (max_bag,)
            'label': torch.LongTensor([bag['label']]),           # (1,)
            'subject_id': bag['subject_id'],
            'video_id': vid,
            'num_instances': N,
            'num_instances_raw': bag['num_instances_raw'],
            'truncated': truncated,
        }


def get_split_ids(dataset_root):
    """Get official train/val/test video IDs from directory structure."""
    train_dir = os.path.join(dataset_root, 'iMiGUE_RGB_Phase1', 'imigue_rgb_train')
    val_dir = os.path.join(dataset_root, 'iMiGUE_RGB_Phase1', 'imigue_rgb_validate')
    test_dir = os.path.join(dataset_root, 'iMiGUE_RGB_Phase2', 'imigue_rgb_test')

    train_ids = sorted([int(d) for d in os.listdir(train_dir)
                        if os.path.isdir(os.path.join(train_dir, d))])
    val_ids = sorted([int(d) for d in os.listdir(val_dir)
                      if os.path.isdir(os.path.join(val_dir, d))])
    test_ids = sorted([int(d) for d in os.listdir(test_dir)
                       if os.path.isdir(os.path.join(test_dir, d))])

    return train_ids, val_ids, test_ids


def build_subject_cv_folds(train_ids, csv_path, n_folds=5):
    """Build subject-disjoint cross-validation folds from training set.

    Returns list of (train_fold_ids, val_fold_ids) tuples.
    """
    df = pd.read_csv(csv_path)
    train_df = df[df['video_id'].isin(train_ids)]
    subjects = train_df.groupby('subject_id')['video_id'].first().index.tolist()
    np.random.seed(42)
    np.random.shuffle(subjects)

    fold_size = len(subjects) // n_folds
    folds = []
    for i in range(n_folds):
        val_subjects = set(subjects[i * fold_size:(i + 1) * fold_size]) if i < n_folds - 1 else set(subjects[i * fold_size:])
        train_fold = [v for v in train_ids
                      if df[df['video_id'] == v].iloc[0]['subject_id'] not in val_subjects]
        val_fold = [v for v in train_ids
                    if df[df['video_id'] == v].iloc[0]['subject_id'] in val_subjects]
        folds.append((train_fold, val_fold))

    return folds


if __name__ == '__main__':
    dataset_root = '/data-store/xingke/iMiGUE'
    csv_path = os.path.join(dataset_root, 'Label', 'labels_20200831.csv')
    skeleton_root = os.path.join(dataset_root, 'mg_skeleton_only')
    npy_dir = os.path.join(dataset_root, 'mg_skeleton_npy')

    train_ids, val_ids, test_ids = get_split_ids(dataset_root)
    print(f'Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}')

    # Test with class 99 included
    print('\n--- With class 99 ---')
    ds = iMiGUEDataset(csv_path, skeleton_root, train_ids[:5], skeleton_npy_dir=npy_dir,
                       include_class99=True)

    # Test with class 99 excluded
    print('\n--- Without class 99 ---')
    ds_no99 = iMiGUEDataset(csv_path, skeleton_root, train_ids[:5], skeleton_npy_dir=npy_dir,
                            include_class99=False)

    sample = ds[0]
    print(f'\nFeatures shape: {sample["features"].shape}')
    print(f'Mask shape: {sample["mask"].shape}')
    print(f'Label: {sample["label"]}')
    print(f'Video ID: {sample["video_id"]}')
    print(f'Num instances (raw): {sample["num_instances_raw"]}')
    print(f'Num instances (used): {sample["num_instances"]}')
    print(f'Truncated: {sample["truncated"]}')
