"""
=============================================================================
AGENT Q-NETWORK COMPONENT
=============================================================================

WHAT IS THIS FILE?
------------------
This file implements the "brain" of each sensor/camera. Each sensor has its
own Q-Network that decides what action to take (rotate left, stay, rotate right).

WHAT IS A Q-NETWORK?
--------------------
A Q-Network is a neural network that learns to estimate "how good" each action is.
- Input: What the sensor currently sees (observation)
- Output: A score for each possible action (Q-values)
- The sensor picks the action with the highest score

WHY USE A GRU (Memory)?
-----------------------
A GRU (Gated Recurrent Unit) is a type of neural network with "memory".
It remembers what happened in previous timesteps, which helps because:
- Targets move in patterns (knowing history helps predict future)
- Coordination requires remembering what other sensors did

ARCHITECTURE DIAGRAM:
---------------------
                    ┌─────────────────────────┐
    Observation ────►│                         │
    (what sensor    │     First Layer (FC)    │────►
    sees)           │   64 neurons, ReLU      │
                    └─────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
    Previous ──────►│                         │
    Hidden State    │      GRU Memory Cell    │────► New Hidden State
    (memory)        │   64-dim hidden state   │      (updated memory)
                    └─────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │                         │
                    │    Output Layer (FC)    │────► Q-values
                    │   3 outputs (L,S,R)     │      [score for each action]
                    └─────────────────────────┘
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentQNetwork(nn.Module):
    """
    Individual Agent Q-Network with GRU memory.

    This network takes an observation and outputs Q-values for each action.
    The GRU provides memory over time, allowing the agent to learn patterns.

    WHAT DOES IT DO?
    ----------------
    1. Takes the current observation (what the sensor sees)
    2. Combines it with what action it took last time
    3. Updates its memory (hidden state) using GRU
    4. Outputs Q-values predicting future rewards for each action

    The sensor then picks the action with the highest Q-value.

    Parameters:
    -----------
    obs_dim : int
        Size of observation vector (e.g., 24 for 6 targets × 4 features)
    n_actions : int
        Number of possible actions (3: LEFT, STAY, RIGHT)
    hidden_dim : int
        Size of the first hidden layer (default: 64)
    rnn_hidden_dim : int
        Size of the GRU memory (default: 64)

    Example:
    --------
    >>> network = AgentQNetwork(obs_dim=24, n_actions=3)
    >>> hidden = network.init_hidden(batch_size=1)
    >>> obs = torch.randn(1, 24)  # Random observation
    >>> q_values, new_hidden = network(obs, hidden)
    >>> print(q_values.shape)  # torch.Size([1, 3])
    """

    def __init__(
        self, obs_dim: int, n_actions: int, hidden_dim: int = 64, rnn_hidden_dim: int = 64
    ):
        super().__init__()

        # Store configuration
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # =============================================
        # Layer 1: Input Processing
        # =============================================
        # Combines observation and previous action
        # Input size: obs_dim + n_actions (e.g., 24 + 3 = 27)
        # Output size: hidden_dim (e.g., 64)
        self.fc1 = nn.Linear(obs_dim + n_actions, hidden_dim)

        # =============================================
        # Layer 2: GRU Memory Cell
        # =============================================
        # This is the "memory" that remembers past observations
        # It takes the processed input and updates its internal state
        self.gru = nn.GRUCell(hidden_dim, rnn_hidden_dim)

        # =============================================
        # Layer 3: Output Q-values
        # =============================================
        # Converts GRU output to Q-values for each action
        self.fc_q = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """
        Initialize the hidden state (memory) to zeros.

        This should be called at the start of each episode to
        reset the agent's memory.

        Parameters:
        -----------
        batch_size : int
            Number of parallel environments/agents

        Returns:
        --------
        hidden : torch.Tensor
            Zero-initialized hidden state of shape [batch_size, rnn_hidden_dim]
        """
        return torch.zeros(batch_size, self.rnn_hidden_dim)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Parameters:
        -----------
        obs : torch.Tensor
            Current observation, shape [batch_size, obs_dim]
        hidden_state : torch.Tensor
            Previous hidden state (memory), shape [batch_size, rnn_hidden_dim]
        prev_action : torch.Tensor, optional
            Previous action taken (for conditioning).
            Can be either:
            - Action indices, shape [batch_size]
            - One-hot encoded, shape [batch_size, n_actions]
            If None, uses zeros (for first timestep)

        Returns:
        --------
        q_values : torch.Tensor
            Q-value for each action, shape [batch_size, n_actions]
            Higher values = better actions
        new_hidden : torch.Tensor
            Updated hidden state (memory), shape [batch_size, rnn_hidden_dim]
        """
        batch_size = obs.size(0)

        # =============================================
        # Step 1: Prepare previous action
        # =============================================
        if prev_action is None:
            # First timestep: no previous action, use zeros
            prev_action = torch.zeros(batch_size, self.n_actions, device=obs.device)
        elif prev_action.dim() == 1:
            # Convert action index to one-hot encoding
            # Example: action=2 becomes [0, 0, 1]
            prev_action = F.one_hot(prev_action.long(), num_classes=self.n_actions).float()

        # =============================================
        # Step 2: Combine observation and previous action
        # =============================================
        # This helps the agent learn action-dependent patterns
        x = torch.cat([obs, prev_action], dim=-1)  # [batch, obs_dim + n_actions]

        # =============================================
        # Step 3: First layer processing (with ReLU activation)
        # =============================================
        # ReLU converts negative values to 0, keeps positive values
        x = F.relu(self.fc1(x))  # [batch, hidden_dim]

        # =============================================
        # Step 4: Update memory using GRU
        # =============================================
        # The GRU combines current input with past memory
        hidden_state = self.gru(x, hidden_state)  # [batch, rnn_hidden_dim]

        # =============================================
        # Step 5: Compute Q-values for each action
        # =============================================
        q_values = self.fc_q(hidden_state)  # [batch, n_actions]

        return q_values, hidden_state
