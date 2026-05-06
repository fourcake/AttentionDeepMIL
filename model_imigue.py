"""ABMIL model adapted for iMiGUE skeleton features.

Input: bag of feature vectors (N x 443), where 443 = 411 (skeleton) + 32 (class one-hot)
Output: bag-level binary prediction (Win/Lose) + attention weights over instances

Based on: Ilse, Tomczak & Welling, "Attention-based Deep Multiple Instance Learning", ICML 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL_iMiGUE(nn.Module):
    def __init__(self, input_dim=443, hidden_dim=256, attention_dim=128,
                 attention_branches=1, dropout=0.1):
        super().__init__()
        self.M = hidden_dim
        self.L = attention_dim
        self.ATTENTION_BRANCHES = attention_branches

        # Feature extractor: MLP instead of CNN (input is vectors, not images)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention mechanism: V^T * tanh(W * h_i)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, attention_branches),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * attention_branches, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, max_bag_size, input_dim) — padded bag features
            mask: (batch, max_bag_size) — True for real instances, False for padding
        Returns:
            y_prob: (batch, 1) — bag-level probability
            y_hat: (batch, 1) — bag-level prediction (0 or 1)
            A: (batch, attention_branches, max_bag_size) — attention weights
        """
        batch_size, max_bag, input_dim = x.shape

        # Flatten for MLP: (batch * max_bag, input_dim)
        H = x.view(-1, input_dim)
        H = self.feature_extractor(H)  # (batch * max_bag, hidden_dim)
        H = H.view(batch_size, max_bag, self.M)  # (batch, max_bag, hidden_dim)

        # Attention: (batch, max_bag, attn_branches)
        A = self.attention(H)

        # Mask padding positions before softmax
        if mask is not None:
            # Set padding positions to large negative value
            mask_expanded = mask.unsqueeze(-1)  # (batch, max_bag, 1)
            A = A.masked_fill(~mask_expanded, float('-inf'))

        A = A.permute(0, 2, 1)  # (batch, attn_branches, max_bag)
        A = F.softmax(A, dim=2)  # softmax over instances

        # Replace NaN (from all-padding softmax) with 0
        A = torch.nan_to_num(A, nan=0.0)

        # Aggregate: weighted sum of hidden representations
        # (batch, attn_branches, max_bag) x (batch, max_bag, hidden_dim)
        Z = torch.bmm(A, H)  # (batch, attn_branches, hidden_dim)
        Z = Z.view(batch_size, -1)  # (batch, attn_branches * hidden_dim)

        # Classify
        y_prob = self.classifier(Z)  # (batch, 1)
        y_hat = (y_prob >= 0.5).float()

        return y_prob, y_hat, A

    def calculate_loss(self, x, y, mask=None, pos_weight=None):
        """Binary cross-entropy loss.

        Args:
            x: bag features
            y: bag label (0 or 1)
            mask: attention mask
            pos_weight: weight for positive class (for imbalanced data)
        """
        y = y.float()
        y_prob, _, A = self.forward(x, mask)
        y_prob = torch.clamp(y_prob, min=1e-5, max=1.0 - 1e-5)

        if pos_weight is not None:
            loss = F.binary_cross_entropy(y_prob, y, reduction='none')
            weight = torch.where(y == 1, pos_weight, 1.0)
            loss = (loss * weight).mean()
        else:
            loss = F.binary_cross_entropy(y_prob, y)

        return loss, A

    def calculate_error(self, x, y, mask=None):
        """Classification error rate."""
        y = y.float()
        _, y_hat, _ = self.forward(x, mask)
        error = 1.0 - y_hat.eq(y).float().mean().item()
        return error, y_hat


class GatedABMIL_iMiGUE(nn.Module):
    """Gated attention variant (Ilse et al. 2018)."""

    def __init__(self, input_dim=443, hidden_dim=256, attention_dim=128,
                 attention_branches=1, dropout=0.1):
        super().__init__()
        self.M = hidden_dim
        self.L = attention_dim
        self.ATTENTION_BRANCHES = attention_branches

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attention_dim, attention_branches)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * attention_branches, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        batch_size, max_bag, input_dim = x.shape

        H = x.view(-1, input_dim)
        H = self.feature_extractor(H)
        H = H.view(batch_size, max_bag, self.M)

        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_w(A_V * A_U)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            A = A.masked_fill(~mask_expanded, float('-inf'))

        A = A.permute(0, 2, 1)
        A = F.softmax(A, dim=2)
        A = torch.nan_to_num(A, nan=0.0)

        Z = torch.bmm(A, H)
        Z = Z.view(batch_size, -1)

        y_prob = self.classifier(Z)
        y_hat = (y_prob >= 0.5).float()

        return y_prob, y_hat, A

    def calculate_loss(self, x, y, mask=None, pos_weight=None):
        y = y.float()
        y_prob, _, A = self.forward(x, mask)
        y_prob = torch.clamp(y_prob, min=1e-5, max=1.0 - 1e-5)

        if pos_weight is not None:
            loss = F.binary_cross_entropy(y_prob, y, reduction='none')
            weight = torch.where(y == 1, pos_weight, 1.0)
            loss = (loss * weight).mean()
        else:
            loss = F.binary_cross_entropy(y_prob, y)

        return loss, A

    def calculate_error(self, x, y, mask=None):
        y = y.float()
        _, y_hat, _ = self.forward(x, mask)
        error = 1.0 - y_hat.eq(y).float().mean().item()
        return error, y_hat
