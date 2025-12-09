"""
State Decoder

Decodes latent representations back to multi-agent states.
"""

import torch
import torch.nn as nn
from typing import Tuple


class StateDecoder(nn.Module):
    """
    Decoder from latent space to multi-agent states.

    Takes latent [B, T, D] and produces states [B, T, K, F]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 6,
        max_agents: int = 50,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize state decoder.

        Args:
            latent_dim: Latent dimension (D)
            hidden_dim: Hidden layer dimension
            output_dim: Output features per agent (F)
            max_agents: Maximum number of agents (K)
            n_layers: Number of MLP layers
            dropout: Dropout probability
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_agents = max_agents

        # Latent to hidden
        layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]

        for _ in range(n_layers - 1):
            layers.extend([
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])

        self.mlp = nn.Sequential(*layers)

        # Output layer: predict all agents at once
        self.output_layer = nn.Linear(hidden_dim, max_agents * output_dim)

        # Optional: predict existence probability for each agent
        self.existence_layer = nn.Linear(hidden_dim, max_agents)

    def forward(
        self,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent to states.

        Args:
            latent: [B, T, D] latent representations

        Returns:
            states: [B, T, K, F] predicted states
            existence_logits: [B, T, K] logits for agent existence
        """
        B, T, D = latent.shape

        # Reshape for processing
        latent_flat = latent.view(B * T, D)

        # Pass through MLP
        hidden = self.mlp(latent_flat)  # [B*T, H]

        # Predict states
        states_flat = self.output_layer(hidden)  # [B*T, K*F]
        states = states_flat.view(B, T, self.max_agents, self.output_dim)

        # Predict existence
        existence_logits = self.existence_layer(hidden)  # [B*T, K]
        existence_logits = existence_logits.view(B, T, self.max_agents)

        return states, existence_logits


class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive decoder that generates agents one-by-one.
    Useful for variable number of agents.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 6,
        max_agents: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_agents = max_agents

        # Initial state embedding
        self.start_token = nn.Parameter(torch.randn(1, output_dim))

        # RNN for autoregressive generation
        self.rnn = nn.GRU(
            input_size=latent_dim + output_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )

        # Output head
        self.output_head = nn.Linear(hidden_dim, output_dim)

        # Stop token predictor
        self.stop_predictor = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        latent: torch.Tensor,
        target_states: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressively generate agent states.

        Args:
            latent: [B, T, D] latent representations
            target_states: [B, T, K, F] ground truth (for teacher forcing)
            teacher_forcing_ratio: Probability of using ground truth

        Returns:
            states: [B, T, K, F] generated states
            stop_probs: [B, T, K] probability of stopping
        """
        B, T, D = latent.shape
        device = latent.device

        # Storage
        all_states = []
        all_stop_probs = []

        # Process each timestep
        for t in range(T):
            latent_t = latent[:, t]  # [B, D]

            states_t = []
            stop_probs_t = []

            # Start with start token
            prev_state = self.start_token.expand(B, -1)  # [B, F]

            hidden = None

            for k in range(self.max_agents):
                # Concatenate latent and previous state
                rnn_input = torch.cat([latent_t, prev_state], dim=-1)  # [B, D+F]
                rnn_input = rnn_input.unsqueeze(1)  # [B, 1, D+F]

                # RNN step
                rnn_out, hidden = self.rnn(rnn_input, hidden)
                rnn_out = rnn_out.squeeze(1)  # [B, H]

                # Predict next state
                next_state = self.output_head(rnn_out)  # [B, F]
                states_t.append(next_state)

                # Predict stop probability
                stop_logit = self.stop_predictor(rnn_out)  # [B, 1]
                stop_probs_t.append(torch.sigmoid(stop_logit.squeeze(-1)))

                # Teacher forcing
                if target_states is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    prev_state = target_states[:, t, k]
                else:
                    prev_state = next_state

            # Stack agents
            states_t = torch.stack(states_t, dim=1)  # [B, K, F]
            stop_probs_t = torch.stack(stop_probs_t, dim=1)  # [B, K]

            all_states.append(states_t)
            all_stop_probs.append(stop_probs_t)

        # Stack timesteps
        states = torch.stack(all_states, dim=1)  # [B, T, K, F]
        stop_probs = torch.stack(all_stop_probs, dim=1)  # [B, T, K]

        return states, stop_probs


if __name__ == '__main__':
    # Test decoder
    B, T, D = 4, 10, 256
    K, F = 20, 6

    latent = torch.randn(B, T, D)

    # Test standard decoder
    decoder = StateDecoder(
        latent_dim=D,
        hidden_dim=128,
        output_dim=F,
        max_agents=K
    )

    states, existence = decoder(latent)
    print(f"Decoded states shape: {states.shape}")  # [B, T, K, F]
    print(f"Existence logits shape: {existence.shape}")  # [B, T, K]

    # Test autoregressive decoder
    ar_decoder = AutoregressiveDecoder(
        latent_dim=D,
        output_dim=F,
        max_agents=K
    )

    states_ar, stop_probs = ar_decoder(latent)
    print(f"AR states shape: {states_ar.shape}")  # [B, T, K, F]
    print(f"Stop probs shape: {stop_probs.shape}")  # [B, T, K]
