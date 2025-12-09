"""
World Model

High-level world model that combines encoder, dynamics, and decoder.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoder import MultiAgentEncoder
from .dynamics import LatentDynamics
from .decoder import StateDecoder


class WorldModel(nn.Module):
    """
    Complete world model for multi-agent trajectory prediction.

    Architecture:
        states [B, T, K, F] -> Encoder -> latent [B, T, D]
        latent [B, T, D] -> Dynamics -> next_latent [B, T, D]
        next_latent [B, T, D] -> Decoder -> predicted_states [B, T, K, F]
    """

    def __init__(
        self,
        input_dim: int = 6,
        max_agents: int = 50,
        encoder_hidden: int = 128,
        latent_dim: int = 256,
        dynamics_hidden: int = 512,
        dynamics_type: str = 'gru',
        n_encoder_layers: int = 2,
        n_dynamics_layers: int = 2,
        n_decoder_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize world model.

        Args:
            input_dim: Number of features per agent (F)
            max_agents: Maximum number of agents (K)
            encoder_hidden: Hidden dimension for encoder
            latent_dim: Latent space dimension (D)
            dynamics_hidden: Hidden dimension for dynamics
            dynamics_type: Type of dynamics model ('gru', 'lstm', 'transformer')
            n_encoder_layers: Number of encoder layers
            n_dynamics_layers: Number of dynamics layers
            n_decoder_layers: Number of decoder layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.max_agents = max_agents
        self.latent_dim = latent_dim

        # Encoder: states -> latent
        self.encoder = MultiAgentEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden,
            latent_dim=latent_dim,
            n_layers=n_encoder_layers,
            dropout=dropout
        )

        # Dynamics: latent -> next_latent
        self.dynamics = LatentDynamics(
            latent_dim=latent_dim,
            hidden_dim=dynamics_hidden,
            n_layers=n_dynamics_layers,
            dropout=dropout,
            model_type=dynamics_type
        )

        # Decoder: latent -> states
        self.decoder = StateDecoder(
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden,
            output_dim=input_dim,
            max_agents=max_agents,
            n_layers=n_decoder_layers,
            dropout=dropout
        )

    def forward(
        self,
        states: torch.Tensor,
        masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode, predict dynamics, decode.

        Args:
            states: [B, T, K, F] input states
            masks: [B, T, K] binary mask

        Returns:
            Dictionary containing:
                - latent: [B, T, D] encoded latent states
                - predicted_latent: [B, T, D] predicted latent from dynamics
                - reconstructed_states: [B, T, K, F] reconstructed states
                - predicted_states: [B, T, K, F] one-step predicted states
                - existence_logits: [B, T, K] agent existence predictions
        """
        # Encode to latent
        latent = self.encoder(states, masks)  # [B, T, D]

        # Predict next latent via dynamics
        predicted_latent, _ = self.dynamics(latent)  # [B, T, D]

        # Decode current latent (reconstruction)
        reconstructed_states, existence_logits = self.decoder(latent)

        # Decode predicted latent (one-step prediction)
        predicted_states, predicted_existence = self.decoder(predicted_latent)

        return {
            'latent': latent,
            'predicted_latent': predicted_latent,
            'reconstructed_states': reconstructed_states,
            'predicted_states': predicted_states,
            'existence_logits': existence_logits,
            'predicted_existence': predicted_existence
        }

    def encode(
        self,
        states: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """Encode states to latent."""
        return self.encoder(states, masks)

    def predict_next(
        self,
        latent: torch.Tensor,
        hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Predict next latent state."""
        return self.dynamics(latent, hidden)

    def decode(
        self,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent to states."""
        return self.decoder(latent)

    def rollout(
        self,
        initial_states: torch.Tensor,
        initial_masks: torch.Tensor,
        n_steps: int,
        teacher_forcing: bool = False,
        ground_truth_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressively rollout future states.

        Args:
            initial_states: [B, T_context, K, F] initial states for context
            initial_masks: [B, T_context, K] masks
            n_steps: Number of steps to predict
            teacher_forcing: Whether to use ground truth during rollout
            ground_truth_states: [B, T_context + n_steps, K, F] (if teacher forcing)

        Returns:
            Dictionary with rollout predictions
        """
        B = initial_states.shape[0]
        device = initial_states.device

        # Encode initial states
        latent = self.encoder(initial_states, initial_masks)  # [B, T_context, D]

        # Initialize RNN hidden state
        _, hidden = self.dynamics(latent)

        # Storage for rollout
        predicted_states_list = []
        predicted_masks_list = []
        latent_list = [latent]

        # Last latent state
        current_latent = latent[:, -1:]  # [B, 1, D]

        for step in range(n_steps):
            # Predict next latent
            next_latent, hidden = self.dynamics(current_latent, hidden)

            # Decode to states
            next_states, existence_logits = self.decoder(next_latent)

            # Convert existence to mask
            next_masks = (torch.sigmoid(existence_logits) > 0.5).float()

            predicted_states_list.append(next_states)
            predicted_masks_list.append(next_masks)
            latent_list.append(next_latent)

            # Teacher forcing: use ground truth
            if teacher_forcing and ground_truth_states is not None:
                context_len = initial_states.shape[1]
                gt_states = ground_truth_states[:, context_len + step:context_len + step + 1]
                gt_masks = initial_masks[:, :1]  # Placeholder
                current_latent = self.encoder(gt_states, gt_masks)
            else:
                current_latent = next_latent

        # Stack predictions
        predicted_states = torch.cat(predicted_states_list, dim=1)  # [B, n_steps, K, F]
        predicted_masks = torch.cat(predicted_masks_list, dim=1)    # [B, n_steps, K]
        latents = torch.cat(latent_list, dim=1)                     # [B, T_context + n_steps, D]

        return {
            'predicted_states': predicted_states,
            'predicted_masks': predicted_masks,
            'latents': latents
        }


if __name__ == '__main__':
    # Test world model
    B, T, K, F = 4, 20, 30, 6

    states = torch.randn(B, T, K, F)
    masks = torch.randint(0, 2, (B, T, K)).float()

    model = WorldModel(
        input_dim=F,
        max_agents=K,
        latent_dim=256,
        dynamics_type='gru'
    )

    # Forward pass
    output = model(states, masks)

    print("World Model Output:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Test rollout
    context_len = 10
    initial_states = states[:, :context_len]
    initial_masks = masks[:, :context_len]

    rollout_output = model.rollout(
        initial_states=initial_states,
        initial_masks=initial_masks,
        n_steps=10
    )

    print("\nRollout Output:")
    for key, value in rollout_output.items():
        print(f"  {key}: {value.shape}")
