"""
=============================================================================
IDENTITY SEMANTICS ENCODER (VAE Component)
=============================================================================

WHAT IS THIS FILE?
------------------
This file implements a VAE (Variational Autoencoder) that creates unique
"identity signatures" for each agent. Think of it like giving each sensor
a "personality" or "fingerprint" that helps the mixer understand who is who.

WHAT IS A VAE?
--------------
A VAE is a type of neural network that:
1. ENCODES: Compresses data into a smaller representation
2. DECODES: Reconstructs the data from the compressed form

It's like summarizing a long book into a short abstract, then trying to
recreate the book from the abstract. Good summaries = low reconstruction error.

WHY DO WE NEED THIS?
--------------------
In multi-agent RL, each agent has different roles and behaviors.
The VAE learns to create unique "identity semantics" (z_i) for each agent,
which helps the mixer network understand:
- Which agents should cooperate
- Which agents are more important in different situations
- How to assign credit for good/bad outcomes

ARCHITECTURE DIAGRAM:
---------------------
                    ┌─────────────────────────┐
    Hidden State    │                         │
    (from GRU)  ────►│    ENCODER Network     │────► Mean, LogVar
    [64-dim]        │    (compresses data)    │      (VAE parameters)
                    └─────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   REPARAMETERIZATION    │
                    │  z = mean + std × noise │────► z (identity semantic)
                    │   (allows backprop)     │      [16-dim]
                    └─────────────────────────┘
                                │
                        ┌───────┴───────┐
                        ▼               ▼
            ┌───────────────────┐ ┌───────────────────┐
            │   DECODER         │ │   MASK GENERATOR  │
            │ (reconstructs     │ │ (creates attention│
            │  hidden state)    │ │  mask for mixer)  │
            └───────────────────┘ └───────────────────┘
                    │                       │
                    ▼                       ▼
            Reconstruction           Semantic Mask
            (for training loss)      (for interpretability)

VAE LOSS FUNCTION:
------------------
L = MSE(reconstruction, original) + β × KL_divergence

- MSE: Measures how well we reconstruct the input (smaller = better)
- KL: Measures how close z is to a normal distribution (regularization)
- β = 0.1: Balance between reconstruction and regularization
"""

from typing import Tuple

import torch
import torch.nn as nn


class IdentitySemanticsEncoder(nn.Module):
    """
    VAE-based encoder that creates unique identity representations for agents.

    Each agent's GRU hidden state is compressed into a "semantic" representation
    that captures the agent's unique role and behavior patterns.

    Parameters:
    -----------
    input_dim : int
        Dimension of input (GRU hidden state, typically 64)
    hidden_dim : int
        Hidden layer size in encoder/decoder (default: 32)
    latent_dim : int
        Size of the identity semantic vector z (default: 16)

    Example:
    --------
    >>> encoder = IdentitySemanticsEncoder(input_dim=64)
    >>> hidden_state = torch.randn(5, 64)  # 5 agents, 64-dim hidden
    >>> z, mask, recon, mean, logvar = encoder(hidden_state)
    >>> print(z.shape)  # torch.Size([5, 16])
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()

        # Store configuration
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # =============================================
        # ENCODER: Compress hidden state → (mean, log_var)
        # =============================================
        # The encoder outputs two things:
        # - mean: The center of the distribution
        # - log_var: The spread of the distribution (in log space for stability)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 64 → 32
            nn.ReLU(),
            nn.Linear(
                hidden_dim, latent_dim * 2
            ),  # 32 → 32 (16 for mean, 16 for log_var)
        )

        # =============================================
        # DECODER: Reconstructs hidden state from z
        # =============================================
        # Used for training: we want the reconstruction to match the original
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # 16 → 32
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # 32 → 64
        )

        # =============================================
        # MASK GENERATOR: Creates attention mask from z
        # =============================================
        # The mask helps the mixer focus on relevant features
        # Sigmoid output ensures values are between 0 and 1
        self.mask_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # 16 → 32
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # 32 → 64
            nn.Sigmoid(),  # Output in range [0, 1]
        )

    def encode(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode hidden state to VAE parameters.

        Parameters:
        -----------
        h : torch.Tensor
            Hidden state from GRU, shape [batch, input_dim]

        Returns:
        --------
        mean : torch.Tensor
            Mean of the latent distribution, shape [batch, latent_dim]
        log_var : torch.Tensor
            Log variance of the distribution, shape [batch, latent_dim]
        """
        # Pass through encoder network
        h_enc = self.encoder(h)  # [batch, latent_dim * 2]

        # Split into mean and log_var
        mean, log_var = h_enc.chunk(2, dim=-1)  # Each: [batch, latent_dim]

        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.

        WHY WE NEED THIS:
        We can't backpropagate through random sampling directly.
        The trick: z = mean + std × noise
        This way, the randomness is "outside" the gradient path.

        Parameters:
        -----------
        mean : torch.Tensor
            Mean of the latent distribution
        log_var : torch.Tensor
            Log variance of the distribution

        Returns:
        --------
        z : torch.Tensor
            Sampled latent vector (identity semantic)
        """
        # Convert log_var to standard deviation
        # std = sqrt(var) = sqrt(exp(log_var)) = exp(0.5 * log_var)
        std = torch.exp(0.5 * log_var)

        # Sample random noise from standard normal distribution
        eps = torch.randn_like(std)

        # Reparameterization: z = mean + std × noise
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector back to hidden state space.

        Parameters:
        -----------
        z : torch.Tensor
            Latent vector (identity semantic), shape [batch, latent_dim]

        Returns:
        --------
        reconstruction : torch.Tensor
            Reconstructed hidden state, shape [batch, input_dim]
        """
        return self.decoder(z)

    def get_mask(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate attention mask from latent vector.

        Parameters:
        -----------
        z : torch.Tensor
            Latent vector, shape [batch, latent_dim]

        Returns:
        --------
        mask : torch.Tensor
            Attention mask, shape [batch, input_dim], values in [0, 1]
        """
        return self.mask_generator(z)

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through the VAE.

        Parameters:
        -----------
        h : torch.Tensor
            Hidden state from GRU, shape [batch, input_dim]

        Returns:
        --------
        z : torch.Tensor
            Identity semantic (latent vector), shape [batch, latent_dim]
        mask : torch.Tensor
            Semantic mask, shape [batch, input_dim]
        recon : torch.Tensor
            Reconstructed hidden state, shape [batch, input_dim]
        mean : torch.Tensor
            VAE mean parameter, shape [batch, latent_dim]
        log_var : torch.Tensor
            VAE log variance parameter, shape [batch, latent_dim]
        """
        # Step 1: Encode to get distribution parameters
        mean, log_var = self.encode(h)

        # Step 2: Sample z using reparameterization trick
        z = self.reparameterize(mean, log_var)

        # Step 3: Decode to get reconstruction (for training loss)
        recon = self.decode(z)

        # Step 4: Generate mask (for attention mechanism)
        mask = self.get_mask(z)

        return z, mask, recon, mean, log_var
