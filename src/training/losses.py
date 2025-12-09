"""
Loss Functions for World Model Training

Includes reconstruction loss, prediction loss, and regularization terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class WorldModelLoss(nn.Module):
    """
    Combined loss for training the world model.

    Components:
        - Reconstruction loss: how well we reconstruct current states
        - Prediction loss: how well we predict next states
        - Existence loss: binary cross-entropy for agent existence
        - Regularization: optional L2 or KL divergence
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        pred_weight: float = 1.0,
        existence_weight: float = 0.1,
        l2_weight: float = 0.0,
        huber_delta: float = 1.0
    ):
        """
        Initialize loss function.

        Args:
            recon_weight: Weight for reconstruction loss
            pred_weight: Weight for prediction loss
            existence_weight: Weight for existence prediction
            l2_weight: Weight for L2 regularization
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__()

        self.recon_weight = recon_weight
        self.pred_weight = pred_weight
        self.existence_weight = existence_weight
        self.l2_weight = l2_weight
        self.huber_delta = huber_delta

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Args:
            predictions: Dictionary from model forward pass
                - reconstructed_states: [B, T, K, F]
                - predicted_states: [B, T, K, F]
                - existence_logits: [B, T, K]
                - predicted_existence: [B, T, K]
            targets: Dictionary with ground truth
                - states: [B, T, K, F]
                - masks: [B, T, K]

        Returns:
            Dictionary of losses
        """
        states = targets['states']
        masks = targets['masks']

        # Reconstruction loss
        recon_loss = self._masked_huber_loss(
            predictions['reconstructed_states'],
            states,
            masks
        )

        # One-step prediction loss (shift by 1 timestep)
        pred_states = predictions['predicted_states'][:, :-1]  # [B, T-1, K, F]
        target_states = states[:, 1:]  # [B, T-1, K, F]
        target_masks = masks[:, 1:]    # [B, T-1, K]

        pred_loss = self._masked_huber_loss(
            pred_states,
            target_states,
            target_masks
        )

        # Existence prediction loss
        existence_loss = self._existence_loss(
            predictions['existence_logits'],
            masks
        )

        # Total loss
        total_loss = (
            self.recon_weight * recon_loss +
            self.pred_weight * pred_loss +
            self.existence_weight * existence_loss
        )

        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'prediction': pred_loss,
            'existence': existence_loss
        }

    def _masked_huber_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Huber loss with masking.

        Args:
            pred: [B, T, K, F] predictions
            target: [B, T, K, F] targets
            mask: [B, T, K] binary mask

        Returns:
            Scalar loss
        """
        # Huber loss
        diff = pred - target
        loss = F.huber_loss(pred, target, reduction='none', delta=self.huber_delta)  # [B, T, K, F]

        # Apply mask
        mask_expanded = mask.unsqueeze(-1)  # [B, T, K, 1]
        masked_loss = loss * mask_expanded

        # Average over valid elements
        total_loss = masked_loss.sum()
        num_valid = mask_expanded.sum().clamp(min=1)

        return total_loss / num_valid

    def _masked_mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Alternative: MSE loss with masking."""
        diff = (pred - target) ** 2  # [B, T, K, F]

        mask_expanded = mask.unsqueeze(-1)
        masked_loss = diff * mask_expanded

        total_loss = masked_loss.sum()
        num_valid = mask_expanded.sum().clamp(min=1)

        return total_loss / num_valid

    def _existence_loss(
        self,
        logits: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Binary cross-entropy for agent existence prediction.

        Args:
            logits: [B, T, K] existence logits
            target_masks: [B, T, K] ground truth masks

        Returns:
            Scalar loss
        """
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target_masks,
            reduction='mean'
        )
        return loss


class RolloutLoss(nn.Module):
    """
    Loss for multi-step rollout evaluation.

    Computes losses at multiple future horizons.
    """

    def __init__(
        self,
        horizons: list = [1, 3, 5, 10],
        loss_type: str = 'huber'
    ):
        super().__init__()
        self.horizons = horizons
        self.loss_type = loss_type

    def forward(
        self,
        rollout_states: torch.Tensor,
        ground_truth: torch.Tensor,
        masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rollout losses at different horizons.

        Args:
            rollout_states: [B, T_rollout, K, F] predicted states
            ground_truth: [B, T_rollout, K, F] ground truth
            masks: [B, T_rollout, K] masks

        Returns:
            Dictionary of losses per horizon
        """
        losses = {}

        for h in self.horizons:
            if h > rollout_states.shape[1]:
                continue

            pred_h = rollout_states[:, :h]
            gt_h = ground_truth[:, :h]
            mask_h = masks[:, :h]

            if self.loss_type == 'huber':
                loss_h = self._masked_huber(pred_h, gt_h, mask_h)
            else:
                loss_h = self._masked_mse(pred_h, gt_h, mask_h)

            losses[f'horizon_{h}'] = loss_h

        # Average across all horizons
        losses['average'] = torch.stack(list(losses.values())).mean()

        return losses

    def _masked_huber(self, pred, target, mask):
        """Masked Huber loss."""
        loss = F.huber_loss(pred, target, reduction='none')
        mask_expanded = mask.unsqueeze(-1)
        masked_loss = (loss * mask_expanded).sum()
        num_valid = mask_expanded.sum().clamp(min=1)
        return masked_loss / num_valid

    def _masked_mse(self, pred, target, mask):
        """Masked MSE loss."""
        loss = (pred - target) ** 2
        mask_expanded = mask.unsqueeze(-1)
        masked_loss = (loss * mask_expanded).sum()
        num_valid = mask_expanded.sum().clamp(min=1)
        return masked_loss / num_valid


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning better latent representations.

    Pulls together latents from similar scenes, pushes apart dissimilar ones.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        latents: torch.Tensor,
        scene_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            latents: [B, T, D] latent representations
            scene_ids: [B] scene identifiers

        Returns:
            Scalar contrastive loss
        """
        # Average pool over time
        latent_mean = latents.mean(dim=1)  # [B, D]

        # L2 normalize
        latent_norm = F.normalize(latent_mean, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(latent_norm, latent_norm.t()) / self.temperature  # [B, B]

        # Create positive mask (same scene)
        scene_ids = scene_ids.unsqueeze(0)  # [1, B]
        positive_mask = (scene_ids == scene_ids.t()).float()  # [B, B]

        # Remove self-similarity
        positive_mask.fill_diagonal_(0)

        # Compute InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        exp_sim.fill_diagonal_(0)

        # Positive pairs
        pos_sim = (exp_sim * positive_mask).sum(dim=1)

        # All pairs
        all_sim = exp_sim.sum(dim=1)

        # Loss: -log(pos / all)
        loss = -torch.log((pos_sim / all_sim.clamp(min=1e-8)) + 1e-8).mean()

        return loss


if __name__ == '__main__':
    # Test loss functions
    B, T, K, F = 4, 10, 20, 6

    # Create dummy data
    predictions = {
        'reconstructed_states': torch.randn(B, T, K, F),
        'predicted_states': torch.randn(B, T, K, F),
        'existence_logits': torch.randn(B, T, K),
        'predicted_existence': torch.randn(B, T, K)
    }

    targets = {
        'states': torch.randn(B, T, K, F),
        'masks': torch.randint(0, 2, (B, T, K)).float()
    }

    # Test loss
    loss_fn = WorldModelLoss()
    losses = loss_fn(predictions, targets)

    print("Losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
