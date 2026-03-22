"""
=============================================================================
NA²Q MIXER COMPONENT
=============================================================================

WHAT IS THIS FILE?
------------------
This file implements the "Mixer" - the component that combines all individual
agent Q-values into a single team Q-value. This is crucial for multi-agent
reinforcement learning because agents need to cooperate.

THE CREDIT ASSIGNMENT PROBLEM:
------------------------------
Imagine a soccer team scores a goal. The team gets +1 reward, but which
players contributed? The striker who scored? The midfielder who passed?
The defender who started the play?

This is the "credit assignment problem" - figuring out who deserves credit.
The mixer solves this using attention mechanisms.

WHAT IS GAM (Generalized Additive Model)?
-----------------------------------------
GAM decomposes the total Q-value into interpretable parts:

    Q_total = α₁×f₁(Q₁) + α₂×f₂(Q₂) + ... + αₙ×fₙ(Qₙ)       <- Individual
            + αᵢⱼ×fᵢⱼ(Qᵢ,Qⱼ) + ...                            <- Pairwise
            + bias(state)                                       <- Base value

Where:
- f_i(Q_i) = "shape function" that transforms individual Q-values
- f_ij(Q_i, Q_j) = pairwise function capturing interactions between agents
- α = attention weights (learned, sum to 1)
- bias = state-dependent baseline

WHY IS THIS BETTER THAN JUST SUM?
----------------------------------
Simple sum: Q_total = Q₁ + Q₂ + Q₃
- Pros: Simple
- Cons: Can't capture interactions, no credit assignment

GAM:
- Pros: Captures interactions, provides interpretable credit assignment
- Cons: More complex

ARCHITECTURE DIAGRAM:
---------------------

Individual Agent Q-values: [Q₁, Q₂, Q₃, Q₄, Q₅]
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    ┌───────────────┐                   ┌───────────────┐
    │ ORDER-1 SHAPES│                   │ ORDER-2 SHAPES│
    │ f₁(Q₁) → val₁ │                   │ f₁₂(Q₁,Q₂)→  │
    │ f₂(Q₂) → val₂ │                   │ f₁₃(Q₁,Q₃)→  │
    │ ...           │                   │ f₂₃(Q₂,Q₃)→  │
    └───────────────┘                   │ ...          │
            │                           └───────────────┘
            └─────────────────┬─────────────────┘
                              │ All shape outputs
                              ▼
                    ┌───────────────────┐
    State ─────────►│ ATTENTION NETWORK │
    Semantics ─────►│ (weights α₁..αₖ) │
                    └───────────────────┘
                              │ Attention weights (sum to 1)
                              ▼
                    ┌───────────────────┐
                    │ WEIGHTED SUM      │
                    │ Σ αₖ × fₖ(...)   │
                    └───────────────────┘
                              │
                              + bias(state)
                              │
                              ▼
                        Q_total (team value)
"""

from itertools import combinations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeFunction(nn.Module):
    """
    A "Shape Function" that transforms Q-value inputs.

    Shape functions are small neural networks that learn non-linear
    transformations of Q-values. They're called "shape" functions because
    they can learn arbitrary shapes (like curves or steps).

    KEY PROPERTY: Non-negative weights
    The weights are forced to be positive (using abs()) to ensure
    monotonicity - higher Q-values should mean higher contributions.

    Parameters:
    -----------
    input_dim : int
        Number of Q-values as input (1 for individual, 2 for pairwise)
    """

    def __init__(self, input_dim: int):
        super().__init__()

        # Small network: input → 8 → 4 → 1
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)

        # ELU activation (like ReLU but smoother for negative values)
        self.elu = nn.ELU()

        # Initialize weights with small values for stability
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform Q-value(s) through the shape function.

        Note: We use .abs() on weights to ensure non-negativity.
        This guarantees that higher Q-values → higher outputs.
        """
        # Apply absolute value to weights for non-negativity constraint
        x = self.elu(F.linear(x, self.fc1.weight.abs(), self.fc1.bias))
        x = self.elu(F.linear(x, self.fc2.weight.abs(), self.fc2.bias))
        x = F.linear(x, self.fc3.weight.abs(), self.fc3.bias)
        return x


class NA2QMixer(nn.Module):
    """
    NA²Q Mixer with GAM-based Value Decomposition.

    Combines individual agent Q-values into a team Q-value using:
    - Individual shape functions (one per agent)
    - Pairwise shape functions (one per agent pair)
    - Attention mechanism for credit assignment

    Formula:
        Q_total = Σₖ (αₖ × fₖ(inputs)) + bias(state)

    Where:
        - αₖ = attention weight for shape function k
        - fₖ = shape function k (can be individual or pairwise)
        - bias = learned state-dependent baseline

    Parameters:
    -----------
    n_agents : int
        Number of agents (e.g., 5 sensors)
    state_dim : int
        Dimension of global state
    latent_dim : int
        Dimension of identity semantics from VAE (default: 16)
    attention_hidden_dim : int
        Hidden dimension for attention network (default: 64)

    Example:
    --------
    >>> mixer = NA2QMixer(n_agents=5, state_dim=25)
    >>> q_values = torch.randn(32, 5)  # 32 batches, 5 agents
    >>> state = torch.randn(32, 25)
    >>> semantics = torch.randn(32, 5, 16)
    >>> q_total, attention, shapes = mixer(q_values, state, semantics)
    >>> print(q_total.shape)  # torch.Size([32, 1])
    """

    def __init__(
        self, n_agents: int, state_dim: int, latent_dim: int = 16, attention_hidden_dim: int = 64
    ):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        # =============================================
        # Count shape functions
        # =============================================
        # Order-1: One per agent (n agents)
        # Order-2: One per pair (n choose 2 = n×(n-1)/2 pairs)
        self.n_order1 = n_agents
        self.n_order2 = n_agents * (n_agents - 1) // 2
        self.n_shape_functions = self.n_order1 + self.n_order2

        # Generate all pairwise indices
        # For 5 agents: [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        self.pairwise_indices = list(combinations(range(n_agents), 2))

        # =============================================
        # Order-1 Shape Functions (Individual)
        # =============================================
        # Each takes a single Q-value and outputs a contribution
        self.order1_shapes = nn.ModuleList(
            [ShapeFunction(input_dim=1) for _ in range(self.n_order1)]
        )

        # =============================================
        # Order-2 Shape Functions (Pairwise)
        # =============================================
        # Each takes TWO Q-values and outputs their interaction
        self.order2_shapes = nn.ModuleList(
            [ShapeFunction(input_dim=2) for _ in range(self.n_order2)]
        )

        # =============================================
        # Attention Network (Credit Assignment)
        # =============================================
        # Encodes global state
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, attention_hidden_dim), nn.ReLU())

        # Encodes agent semantics (identities)
        self.semantic_encoder = nn.Sequential(
            nn.Linear(latent_dim * n_agents, attention_hidden_dim), nn.ReLU()
        )

        # Combines state and semantics → attention weights
        self.attention_net = nn.Sequential(
            nn.Linear(attention_hidden_dim * 2, attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(attention_hidden_dim, self.n_shape_functions),
        )

        # =============================================
        # State-Dependent Bias
        # =============================================
        # Baseline value that depends on state
        self.bias_net = nn.Sequential(
            nn.Linear(state_dim, attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(attention_hidden_dim, 1),
        )

    def forward(
        self, agent_q_values: torch.Tensor, state: torch.Tensor, agent_semantics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute team Q-value from individual Q-values.

        Parameters:
        -----------
        agent_q_values : torch.Tensor
            Individual Q-values for chosen actions, shape [batch, n_agents]
        state : torch.Tensor
            Global state, shape [batch, state_dim]
        agent_semantics : torch.Tensor
            Identity semantics from VAE, shape [batch, n_agents, latent_dim]

        Returns:
        --------
        q_total : torch.Tensor
            Team Q-value, shape [batch, 1]
        attention_weights : torch.Tensor
            Credit assignment weights, shape [batch, n_shape_functions]
        shape_outputs : torch.Tensor
            Raw shape function outputs, shape [batch, n_shape_functions]
        """
        batch_size = agent_q_values.size(0)

        # =============================================
        # Step 1: Compute Shape Function Outputs
        # =============================================
        shape_outputs = []

        # Order-1: Individual contributions f_i(Q_i)
        for i in range(self.n_order1):
            q_i = agent_q_values[:, i : i + 1]  # Shape [batch, 1]
            f_i = self.order1_shapes[i](q_i)
            shape_outputs.append(f_i)

        # Order-2: Pairwise interactions f_ij(Q_i, Q_j)
        for idx, (i, j) in enumerate(self.pairwise_indices):
            q_i = agent_q_values[:, i : i + 1]
            q_j = agent_q_values[:, j : j + 1]
            q_ij = torch.cat([q_i, q_j], dim=-1)  # [batch, 2]
            f_ij = self.order2_shapes[idx](q_ij)
            shape_outputs.append(f_ij)

        # Stack all outputs: [batch, n_shape_functions]
        shape_outputs = torch.cat(shape_outputs, dim=-1)

        # =============================================
        # Step 2: Compute Attention Weights
        # =============================================
        # Encode global state
        state_encoding = self.state_encoder(state)  # [batch, 64]

        # Encode agent semantics (flatten first)
        semantics_flat = agent_semantics.view(batch_size, -1)  # [batch, n_agents × latent_dim]
        semantic_encoding = self.semantic_encoder(semantics_flat)  # [batch, 64]

        # Combine state and semantic encodings
        combined = torch.cat([state_encoding, semantic_encoding], dim=-1)  # [batch, 128]

        # Compute attention logits and apply softmax
        attention_logits = self.attention_net(combined)  # [batch, n_shape_functions]
        attention_weights = F.softmax(attention_logits, dim=-1)  # Sum to 1

        # =============================================
        # Step 3: Compute Q_total
        # =============================================
        # Weighted sum of shape function outputs
        weighted_sum = (attention_weights * shape_outputs).sum(dim=-1, keepdim=True)

        # Add state-dependent bias
        bias = self.bias_net(state)  # [batch, 1]

        # Final Q_total
        q_total = weighted_sum + bias  # [batch, 1]

        return q_total, attention_weights, shape_outputs

    def get_individual_contributions(
        self, attention_weights: torch.Tensor, shape_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get interpretable individual and pairwise contributions.

        This is useful for understanding which agents contributed most
        to the team Q-value.

        Parameters:
        -----------
        attention_weights : torch.Tensor
            Credit assignment weights, shape [batch, n_shape_functions]
        shape_outputs : torch.Tensor
            Raw shape outputs, shape [batch, n_shape_functions]

        Returns:
        --------
        individual_contribs : torch.Tensor
            Contribution from each agent, shape [batch, n_agents]
        pairwise_contribs : torch.Tensor
            Contribution from each pair, shape [batch, n_pairs]
        """
        # Individual contributions (first n_order1 elements)
        individual_contribs = (
            attention_weights[:, : self.n_order1] * shape_outputs[:, : self.n_order1]
        )

        # Pairwise contributions (remaining elements)
        pairwise_contribs = (
            attention_weights[:, self.n_order1 :] * shape_outputs[:, self.n_order1 :]
        )

        return individual_contribs, pairwise_contribs
