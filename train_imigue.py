"""Training script for iMiGUE ABMIL baseline.

Usage:
    # Standard attention (default: exclude class 99, fold 0)
    python train_imigue.py --model attention --epochs 50

    # Gated attention
    python train_imigue.py --model gated_attention --epochs 50

    # Include class 99 as soft negative (ablation)
    python train_imigue.py --model attention --include_class99

    # Select CV fold
    python train_imigue.py --model attention --cv_fold 2

    # Custom hyperparams
    python train_imigue.py --lr 0.001 --hidden_dim 512 --attention_dim 256 --dropout 0.2

    # Small model for signal detection
    python train_imigue.py --hidden_dim 32 --attention_dim 16 --dropout 0.5
"""

import argparse
import json
import os
import time
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from imigue_dataset import iMiGUEDataset, get_split_ids, build_subject_cv_folds
from model_imigue import ABMIL_iMiGUE, GatedABMIL_iMiGUE


def collate_fn(batch):
    """Collate variable-length bags into a batch."""
    return {
        'features': torch.stack([s['features'] for s in batch]),
        'mask': torch.stack([s['mask'] for s in batch]),
        'label': torch.stack([s['label'] for s in batch]),
        'subject_id': [s['subject_id'] for s in batch],
        'video_id': [s['video_id'] for s in batch],
        'num_instances': [s['num_instances'] for s in batch],
        'num_instances_raw': [s['num_instances_raw'] for s in batch],
    }


def compute_balanced_accuracy(y_true, y_pred):
    """Balanced accuracy: average of per-class recall."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accs = []
    for c in [0, 1]:
        mask = y_true == c
        if mask.sum() > 0:
            accs.append((y_pred[mask] == c).mean())
    return np.mean(accs) if accs else 0.0


def compute_auc(y_true, y_prob):
    """Compute AUC-ROC using the trapezoidal rule (no sklearn dependency)."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    # Sort by descending probability
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    # Trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return auc


def compute_f1(y_true, y_pred):
    """Compute F1 score for binary classification."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss = 0.0
    total_error = 0.0
    n = 0

    for batch in loader:
        x = batch['features'].to(device)
        y = batch['label'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()
        loss, _ = model.calculate_loss(x, y, mask, pos_weight=pos_weight)
        error, _ = model.calculate_error(x, y, mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_error += error
        n += 1

    return total_loss / n, total_error / n


@torch.no_grad()
def evaluate(model, loader, device, pos_weight):
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    video_ids_all = []
    attn_weights_all = []
    n = 0

    for batch in loader:
        x = batch['features'].to(device)
        y = batch['label'].to(device)
        mask = batch['mask'].to(device)

        loss, attn = model.calculate_loss(x, y, mask, pos_weight=pos_weight)
        error, y_hat = model.calculate_error(x, y, mask)

        # Get probabilities from logits
        y_logit, _, _ = model.forward(x, mask)
        y_prob = torch.sigmoid(y_logit)

        total_loss += loss.item()
        total_error += error
        y_true_all.extend(y.cpu().numpy().flatten().tolist())
        y_pred_all.extend(y_hat.cpu().numpy().flatten().tolist())
        y_prob_all.extend(y_prob.cpu().numpy().flatten().tolist())
        video_ids_all.extend(batch['video_id'])
        # Store per-sample attention weights: (attn_branches, max_bag)
        attn_weights_all.extend(attn.cpu().numpy())
        n += 1

    avg_loss = total_loss / n
    avg_error = total_error / n
    bal_acc = compute_balanced_accuracy(y_true_all, y_pred_all)
    auc = compute_auc(y_true_all, y_prob_all)
    f1 = compute_f1(y_true_all, y_pred_all)

    return avg_loss, avg_error, bal_acc, auc, f1, y_true_all, y_pred_all, y_prob_all, video_ids_all, attn_weights_all


def main():
    parser = argparse.ArgumentParser(description='iMiGUE ABMIL Training')
    # Data
    parser.add_argument('--dataset_root', type=str, default='/data-store/xingke/iMiGUE')
    parser.add_argument('--max_bag_size', type=int, default=256)
    parser.add_argument('--include_class99', action='store_true',
                        help='Include class 99 (non-MG) instances in bags')
    # Model
    parser.add_argument('--model', type=str, default='attention',
                        choices=['attention', 'gated_attention'])
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--attention_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    # Validation
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of CV folds to build')
    parser.add_argument('--cv_fold', type=int, default=0,
                        help='Which CV fold to use as validation (0-indexed)')
    # Misc
    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument('--log_interval', type=int, default=5)

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    class_tag = 'all' if args.include_class99 else 'no99'
    run_name = f'{args.model}_{timestamp}_seed{args.seed}_fold{args.cv_fold}_{class_tag}_bag{args.max_bag_size}'
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f'Output directory: {save_dir}')

    # Data paths
    csv_path = os.path.join(args.dataset_root, 'Label', 'labels_20200831.csv')
    skeleton_root = os.path.join(args.dataset_root, 'mg_skeleton_only')
    npy_dir = os.path.join(args.dataset_root, 'mg_skeleton_npy')

    # Get splits
    train_ids, val_ids, test_ids = get_split_ids(args.dataset_root)

    # Build subject-disjoint CV folds from training data
    # (official val set is useless: all Win, subjects overlap with train)
    from imigue_dataset import build_subject_cv_folds
    folds = build_subject_cv_folds(train_ids, csv_path, n_folds=args.cv_folds)
    fold_idx = min(args.cv_fold, len(folds) - 1)
    new_train_ids, new_val_ids = folds[fold_idx]
    print(f'Using CV fold {fold_idx}/{len(folds)-1}')
    print(f'Splits: train={len(new_train_ids)}, val={len(new_val_ids)}, test={len(test_ids)}')
    train_ids = new_train_ids
    val_ids = new_val_ids

    # Compute class counts for balanced sampling
    import pandas as pd
    df = pd.read_csv(csv_path)
    train_labels = df[df['video_id'].isin(train_ids)].groupby('video_id')['win_or_lose'].first()
    n_win = (train_labels == 'Win').sum()
    n_lose = (train_labels == 'Lose').sum()
    pos_weight = None  # handled by WeightedRandomSampler
    print(f'Train label distribution: Win={n_win}, Lose={n_lose}')

    # Build datasets
    train_ds = iMiGUEDataset(csv_path, skeleton_root, train_ids,
                             max_bag_size=args.max_bag_size, skeleton_npy_dir=npy_dir,
                             include_class99=args.include_class99)
    val_ds = iMiGUEDataset(csv_path, skeleton_root, val_ids,
                           max_bag_size=args.max_bag_size, skeleton_npy_dir=npy_dir,
                           include_class99=args.include_class99)
    test_ds = iMiGUEDataset(csv_path, skeleton_root, test_ids,
                            max_bag_size=args.max_bag_size, skeleton_npy_dir=npy_dir,
                            include_class99=args.include_class99)

    # Balanced sampling: oversample Lose class to match Win class
    train_sample_weights = []
    for vid in train_ds.video_ids:
        label = train_ds.bags[vid]['label']
        train_sample_weights.append(1.0 / n_win if label == 1 else 1.0 / n_lose)
    sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_ds.video_ids), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Build model
    if args.model == 'attention':
        model = ABMIL_iMiGUE(input_dim=443, hidden_dim=args.hidden_dim,
                             attention_dim=args.attention_dim, dropout=args.dropout)
    else:
        model = GatedABMIL_iMiGUE(input_dim=443, hidden_dim=args.hidden_dim,
                                   attention_dim=args.attention_dim, dropout=args.dropout)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {args.model}, Parameters: {param_count:,}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Save run config
    results = {
        'args': vars(args),
        'run_name': run_name,
        'param_count': param_count,
        'train_size': len(train_ds.video_ids),
        'val_size': len(val_ds.video_ids),
        'test_size': len(test_ds.video_ids),
        'train_win': int(n_win),
        'train_lose': int(n_lose),
    }

    # Training loop
    best_val_bal_acc = 0.0

    print(f'\n{"Epoch":>5} {"TrainLoss":>10} {"TrainErr":>10} {"ValLoss":>10} '
          f'{"ValErr":>10} {"ValBalAcc":>10} {"ValAUC":>8} {"Time":>8}')
    print('-' * 78)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_err = train_epoch(model, train_loader, optimizer, device, pos_weight)
        val_loss, val_err, val_bal_acc, val_auc, val_f1, _, _, _, _, _ = \
            evaluate(model, val_loader, device, pos_weight)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        if epoch % args.log_interval == 0 or epoch == 1:
            print(f'{epoch:5d} {train_loss:10.4f} {train_err:10.4f} {val_loss:10.4f} '
                  f'{val_err:10.4f} {val_bal_acc:10.4f} {val_auc:8.4f} {elapsed:7.1f}s')

        # Save best model
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_bal_acc': val_bal_acc,
                'val_loss': val_loss,
                'args': vars(args),
            }, save_path)

    # Final evaluation on test set
    print(f'\nBest validation balanced accuracy: {best_val_bal_acc:.4f}')
    print('Loading best model for test evaluation...')

    ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_err, test_bal_acc, test_auc, test_f1, \
        test_y_true, test_y_pred, test_y_prob, test_video_ids, test_attn = \
        evaluate(model, test_loader, device, pos_weight)

    print(f'\nTest Results:')
    print(f'  Loss: {test_loss:.4f}')
    print(f'  Error: {test_err:.4f}')
    print(f'  Balanced Accuracy: {test_bal_acc:.4f}')
    print(f'  AUC: {test_auc:.4f}')
    print(f'  F1: {test_f1:.4f}')

    # Confusion matrix
    y_true = np.array(test_y_true)
    y_pred = np.array(test_y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    print(f'\nConfusion Matrix:')
    print(f'  TP={tp} FP={fp} TN={tn} FN={fn}')

    # Save results
    results.update({
        'best_epoch': ckpt['epoch'],
        'best_val_bal_acc': float(best_val_bal_acc),
        'test_loss': float(test_loss),
        'test_error': float(test_err),
        'test_bal_acc': float(test_bal_acc),
        'test_auc': float(test_auc),
        'test_f1': float(test_f1),
        'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
    })

    # Save JSON results
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save per-video predictions
    predictions_path = os.path.join(save_dir, 'test_predictions.json')
    predictions = []
    for i, vid in enumerate(test_video_ids):
        predictions.append({
            'video_id': int(vid),
            'y_true': int(test_y_true[i]),
            'y_pred': int(test_y_pred[i]),
            'y_prob': float(test_y_prob[i]),
        })
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    # Save attention weights
    attn_path = os.path.join(save_dir, 'test_attention.npz')
    # Stack into (n_test, attn_branches, max_bag_size)
    attn_array = np.stack(test_attn)
    np.savez(attn_path,
             attention=attn_array,
             video_ids=np.array(test_video_ids),
             y_true=np.array(test_y_true),
             y_pred=np.array(test_y_pred))

    # Also save human-readable text summary
    txt_path = os.path.join(save_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write(f'Run: {run_name}\n')
        f.write(f'Model: {args.model}\n')
        f.write(f'Args: {vars(args)}\n')
        f.write(f'Parameters: {param_count:,}\n')
        f.write(f'Train size: {len(train_ds.video_ids)}\n')
        f.write(f'Val size: {len(val_ds.video_ids)}\n')
        f.write(f'Test size: {len(test_ds.video_ids)}\n')
        f.write(f'\nBest Val Balanced Accuracy: {best_val_bal_acc:.4f}\n')
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Test Error: {test_err:.4f}\n')
        f.write(f'Test Balanced Accuracy: {test_bal_acc:.4f}\n')
        f.write(f'Test AUC: {test_auc:.4f}\n')
        f.write(f'Test F1: {test_f1:.4f}\n')
        f.write(f'\nConfusion Matrix:\n')
        f.write(f'  TP={tp} FP={fp} TN={tn} FN={fn}\n')

    print(f'\nAll outputs saved to {save_dir}/')
    print(f'  results.json        — structured results')
    print(f'  results.txt         — human-readable summary')
    print(f'  test_predictions.json — per-video predictions and probabilities')
    print(f'  test_attention.npz  — attention weights')
    print(f'  best_model.pth      — model checkpoint')


if __name__ == '__main__':
    main()
