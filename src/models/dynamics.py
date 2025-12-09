"""
Latent Dynamics Model

Models temporal evolution in latent space.
Can use GRU, LSTM, or Transformer for temporal modeling.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LatentDynamics(nn.Module):
    """
    Temporal dynamics model in latent space.

    Takes latent sequence [B, T, D] and predicts next latent states.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        model_type: str = 'gru'
    ):
        """
        Initialize latent dynamics model.

        Args:
            latent_dim: Dimension of latent space (D)
            hidden_dim: Hidden dimension for RNN/Transformer
            n_layers: Number of layers
            dropout: Dropout probability
            model_type: 'gru', 'lstm', or 'transformer'
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type

        if model_type == 'gru':
            self.rnn = nn.GRU(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
        elif model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Output projection back to latent space
        if model_type in ['gru', 'lstm']:
            self.output_proj = nn.Linear(hidden_dim, latent_dim)
        # For transformer, output is already in latent_dim

    def forward(
        self,
        latent: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Forward pass through dynamics model.

        Args:
            latent: [B, T, D] latent states
            hidden: Optional hidden state for RNNs

        Returns:
            output: [B, T, D] predicted latent states
            hidden: Updated hidden state (for RNNs)
        """
        if self.model_type == 'transformer':
            output = self.transformer(latent)
            return output, None

        else:  # GRU or LSTM
            if hidden is not None:
                rnn_out, hidden = self.rnn(latent, hidden)
            else:
                rnn_out, hidden = self.rnn(latent)

            output = self.output_proj(rnn_out)  # [B, T, D]
            return output, hidden

    def step(
        self,
        latent_t: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Single-step prediction (for autoregressive rollout).

        Args:
            latent_t: [B, 1, D] or [B, D] current latent state
            hidden: Hidden state from previous step

        Returns:
            next_latent: [B, D] predicted next latent
            hidden: Updated hidden state
        """
        if latent_t.dim() == 2:
            latent_t = latent_t.unsqueeze(1)  # [B, D] -> [B, 1, D]

        if self.model_type == 'transformer':
            # For transformer, we need full history - use caching for efficiency
            output = self.transformer(latent_t)
            return output.squeeze(1), None

        else:  # GRU or LSTM
            rnn_out, hidden = self.rnn(latent_t, hidden)
            output = self.output_proj(rnn_out)
            return output.squeeze(1), hidden


class RSSMDynamics(nn.Module):
    """
    Recurrent State-Space Model (RSSM) dynamics.

    Separates deterministic and stochastic components.
    Used in Dreamer-style world models.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        stochastic_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim

        # Deterministic state (recurrent)
        self.rnn = nn.GRU(
            input_size=latent_dim + stochastic_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # Stochastic state (prior)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stochastic_dim * 2)  # mean and logvar
        )

        # Stochastic state (posterior, using observation)
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stochastic_dim * 2)
        )

    def forward(
        self,
        latent: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        RSSM forward pass with both prior and posterior.

        Args:
            latent: [B, T, D] observed latent states
            hidden: [n_layers, B, H] initial hidden state

        Returns:
            reconstructed: [B, T, D] reconstructed latent
            info: Dictionary with prior/posterior distributions
        """
        B, T, D = latent.shape

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(
                self.rnn.num_layers, B, self.hidden_dim,
                device=latent.device
            )

        # Storage for outputs
        stochastic_states = []
        prior_dists = []
        posterior_dists = []

        # Process sequence
        for t in range(T):
            latent_t = latent[:, t]  # [B, D]

            # Posterior (using observation)
            post_input = torch.cat([hidden[-1], latent_t], dim=-1)
            post_params = self.posterior(post_input)
            post_mean, post_logvar = torch.chunk(post_params, 2, dim=-1)

            # Sample stochastic state
            stoch = self._sample_gaussian(post_mean, post_logvar)
            stochastic_states.append(stoch)

            # Update deterministic state
            rnn_input = torch.cat([latent_t, stoch], dim=-1).unsqueeze(1)
            _, hidden = self.rnn(rnn_input, hidden)

            # Prior (for KL loss)
            prior_params = self.prior(hidden[-1])
            prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

            prior_dists.append((prior_mean, prior_logvar))
            posterior_dists.append((post_mean, post_logvar))

        # Stack outputs
        stochastic = torch.stack(stochastic_states, dim=1)  # [B, T, S]

        info = {
            'prior_dists': prior_dists,
            'posterior_dists': posterior_dists,
            'stochastic': stochastic,
            'hidden': hidden
        }

        return stochastic, info

    def _sample_gaussian(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


if __name__ == '__main__':
    # Test dynamics model
    B, T, D = 4, 10, 256

    latent = torch.randn(B, T, D)

    # Test GRU dynamics
    dynamics = LatentDynamics(
        latent_dim=D,
        hidden_dim=512,
        model_type='gru'
    )

    output, hidden = dynamics(latent)
    print(f"GRU output shape: {output.shape}")  # [B, T, D]

    # Test single step
    next_latent, hidden = dynamics.step(latent[:, 0], hidden)
    print(f"Step output shape: {next_latent.shape}")  # [B, D]
