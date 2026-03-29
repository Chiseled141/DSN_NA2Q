"""
=============================================================================
NA²Q AGENT - Training and Inference Wrapper
=============================================================================

WHAT IS THIS FILE?
------------------
This file is the "boss" that coordinates everything. It wraps the NA²Q neural
network and handles:
1. Action selection (deciding what each camera should do)
2. Training (updating neural network weights)
3. Saving/loading models (checkpoints)

THINK OF IT LIKE A COACH:
- The neural networks (in na2q_model.py) are the "players"
- This agent is the "coach" that tells them when to train and how to play

KEY CONCEPTS EXPLAINED:
-----------------------

1. EPSILON-GREEDY EXPLORATION
   When learning, agents need to balance:
   - EXPLORATION: Try random actions to discover new strategies
   - EXPLOITATION: Use what we already learned (pick best action)

   ε-greedy: With probability ε, pick random action; otherwise pick best action

   Example: ε = 0.3 means 30% random, 70% best

   Over time, ε decreases so the agent explores less and exploits more.

2. DOUBLE DQN (Deep Q-Network)
   Problem: Regular DQN overestimates Q-values (too optimistic)
   Solution: Use TWO networks:
   - Online Network: Picks the best action
   - Target Network: Estimates the Q-value for that action

   This gives more stable, realistic Q-value estimates.

3. TARGET NETWORK
   A slowly-updated copy of the main network.
   Why? Training is unstable if targets keep changing.
   Solution: Update target network slowly (soft update with τ=0.001)

4. OPTIMIZERS
   - RMSprop: For Q-learning (stable for RL, used in original DQN)
   - Adam: For VAE (good for autoencoders)

ARCHITECTURE OVERVIEW:
----------------------
┌─────────────────────────────────────────────────────────────────────┐
│                          NA²Q AGENT                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────┐    ┌─────────────────────────┐        │
│  │    ONLINE MODEL         │    │    TARGET MODEL         │        │
│  │  (learns and updates)   │    │  (stable Q targets)     │        │
│  └───────────┬─────────────┘    └───────────┬─────────────┘        │
│              │                              │                        │
│              │ Q-values for actions         │ Q-values for targets   │
│              │                              │                        │
│  ┌───────────▼─────────────────────────────▼───────────────┐       │
│  │                    TD LOSS CALCULATION                    │       │
│  │   Loss = (Q_predicted - Q_target)²                        │       │
│  │   Q_target = reward + γ × Target_Q(next_state, best_act)  │       │
│  └───────────────────────────┬──────────────────────────────┘       │
│                              │                                       │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │                    BACKPROPAGATION                         │      │
│  │   Update weights to minimize loss                          │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

from .na2q_model import NA2Q

# =============================================================================
# NA²Q AGENT CLASS
# =============================================================================


class NA2QAgent:
    """
    NA²Q Agent for training and inference.

    This class wraps the NA²Q neural network model and provides:
    - Action selection (ε-greedy for exploration)
    - Training step (update weights using TD learning)
    - Model saving/loading (checkpoints)
    - Interpretability (visualize agent contributions)

    Parameters:
    -----------
    n_agents : int
        Number of agents (sensors/cameras)
    obs_dim : int
        Observation dimension per agent
    state_dim : int
        Global state dimension (used by mixer)
    n_actions : int
        Number of possible actions (3: LEFT, STAY, RIGHT)
    hidden_dim : int
        Hidden layer size in Q-network (default: 64)
    rnn_hidden_dim : int
        GRU memory size (default: 64)
    semantic_hidden_dim : int
        VAE hidden size (default: 32)
    latent_dim : int
        VAE latent dimension / identity semantic size (default: 16)
    attention_hidden_dim : int
        Mixer attention hidden size (default: 64)
    lr : float
        Learning rate (default: 5e-4 = 0.0005)
    gamma : float
        Discount factor - how much to value future rewards (default: 0.99)
        γ close to 1 = think long-term, γ close to 0 = think short-term
    epsilon_start : float
        Initial exploration rate (default: 1.0 = 100% random)
    epsilon_end : float
        Final exploration rate (default: 0.05 = 5% random)
    epsilon_decay : int
        Episodes until epsilon reaches epsilon_end (default: 50000)
    target_update_interval : int
        How often to hard-update target network (legacy, we use soft update now)
    vae_loss_weight : float
        Weight for VAE loss in total loss (default: 0.1)
    device : str
        "cuda", "mps", or "cpu"
    use_amp : bool
        Use Automatic Mixed Precision for faster training on GPU

    Example:
    --------
    >>> agent = NA2QAgent(n_agents=5, obs_dim=24, state_dim=25, n_actions=3)
    >>> agent.init_hidden(batch_size=1)
    >>> actions = agent.select_actions(observations, avail_actions=avail)
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        semantic_hidden_dim: int = 32,
        latent_dim: int = 16,
        attention_hidden_dim: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        target_update_interval: int = 200,
        vae_loss_weight: float = 0.1,
        device: str = "cpu",
        use_amp: bool = True,
    ):
        """Initialize the agent with all hyperparameters."""

        # =============================================
        # Store configuration
        # =============================================
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma  # Discount factor: 0.99 means future rewards are worth 99% of immediate

        # Exploration parameters (ε-greedy)
        self.epsilon = epsilon_start  # Current exploration rate
        self.epsilon_end = epsilon_end  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Episodes to decay over

        self.target_update_interval = target_update_interval
        self.vae_loss_weight = vae_loss_weight
        self.device = torch.device(device)

        # AMP (Automatic Mixed Precision) only works on CUDA GPUs
        self.use_amp = use_amp and self.device.type == "cuda"

        # =============================================
        # Create Neural Networks
        # =============================================

        # ONLINE MODEL: The main model that learns and makes decisions
        self.model = NA2Q(
            n_agents,
            obs_dim,
            state_dim,
            n_actions,
            hidden_dim,
            rnn_hidden_dim,
            semantic_hidden_dim,
            latent_dim,
            attention_hidden_dim,
        ).to(self.device)

        # TARGET MODEL: A stable copy used for computing TD targets
        # Why? If we used the online model for targets, the targets would
        # keep changing, making training unstable (like chasing a moving target)
        self.target_model = NA2Q(
            n_agents,
            obs_dim,
            state_dim,
            n_actions,
            hidden_dim,
            rnn_hidden_dim,
            semantic_hidden_dim,
            latent_dim,
            attention_hidden_dim,
        ).to(self.device)

        # Copy weights from online model to target model
        self.target_model.load_state_dict(self.model.state_dict())

        # Freeze target model (we update it manually, not through backprop)
        for param in self.target_model.parameters():
            param.requires_grad = False

        # =============================================
        # Create Optimizers
        # =============================================

        # Separate parameters for Q-learning and VAE
        # Q-learning: Agent Q-network + Mixer
        q_params = list(self.model.agent_q_network.parameters()) + list(
            self.model.mixer.parameters()
        )

        # VAE: Semantics encoder only
        vae_params = list(self.model.semantics_encoder.parameters())

        # RMSprop for Q-learning (stable for RL, used in original DQN paper)
        # - alpha=0.99: Decay rate for running average of squared gradients
        # - eps=1e-5: Small constant for numerical stability
        self.q_optimizer = torch.optim.RMSprop(q_params, lr=lr, alpha=0.99, eps=1e-5)

        # Adam for VAE (good for autoencoders)
        # - betas=(0.9, 0.999): Decay rates for first and second moment estimates
        self.vae_optimizer = torch.optim.Adam(vae_params, lr=lr, betas=(0.9, 0.999))

        # Gradient scaler for mixed precision training
        self.scaler = GradScaler(enabled=self.use_amp)

        # =============================================
        # Training State
        # =============================================
        self.train_step = 0  # Number of training steps taken
        self.hidden_states = None  # GRU hidden states (memory)

    def init_hidden(self, batch_size: int = 1):
        """
        Initialize hidden states (memory) for all agents.

        This should be called at the start of each episode to reset
        the agent's memory.

        Parameters:
        -----------
        batch_size : int
            Number of parallel environments (usually 1)
        """
        self.hidden_states = self.model.init_hidden(batch_size).to(self.device)

    # =========================================================================
    # ACTION SELECTION
    # =========================================================================

    def select_actions(
        self,
        observations: np.ndarray,
        prev_actions: Optional[np.ndarray] = None,
        avail_actions: Optional[np.ndarray] = None,
        evaluate: bool = False,
    ) -> np.ndarray:
        """
        Select actions for all agents using ε-greedy policy.

        ε-GREEDY EXPLAINED:
        - With probability ε: Pick a RANDOM action (explore)
        - With probability 1-ε: Pick the BEST action (exploit)

        During training, ε starts high (more exploration) and decreases
        over time (more exploitation of learned knowledge).

        Parameters:
        -----------
        observations : np.ndarray
            Observations for all agents, shape [n_agents, obs_dim]
        prev_actions : np.ndarray, optional
            Previous actions (for action-conditioned policies)
        avail_actions : np.ndarray, optional
            Mask of available actions, shape [n_agents, n_actions]
            1 = available, 0 = not available
        evaluate : bool
            If True, always pick best action (no exploration)
            Used during testing/evaluation

        Returns:
        --------
        actions : np.ndarray
            Selected actions for each agent, shape [n_agents]

        Example:
        --------
        >>> actions = agent.select_actions(obs, evaluate=False)
        >>> print(actions)  # e.g., [0, 2, 1, 1, 0] for 5 agents
        """
        # No gradient computation needed for action selection
        with torch.no_grad():
            # =============================================
            # Step 1: Prepare inputs
            # =============================================

            # Convert numpy to tensor and add batch dimension
            obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
            # Shape: [1, n_agents, obs_dim]

            # Prepare previous actions if provided
            prev_actions_tensor = None
            if prev_actions is not None:
                prev_actions_tensor = (
                    torch.LongTensor(prev_actions).unsqueeze(0).to(self.device)
                )

            # Initialize hidden states if not already done
            if self.hidden_states is None:
                self.init_hidden(1)

            # =============================================
            # Step 2: Get Q-values from network
            # =============================================

            q_values, self.hidden_states = self.model.get_q_values(
                obs_tensor, self.hidden_states, prev_actions_tensor
            )
            q_values = q_values.squeeze(0)  # Remove batch dim: [n_agents, n_actions]

            # =============================================
            # Step 3: Mask unavailable actions
            # =============================================

            if avail_actions is not None:
                avail_mask = torch.FloatTensor(avail_actions).to(self.device)
                # Set Q-value to -inf for unavailable actions
                # This ensures they're never selected by argmax
                q_values[avail_mask == 0] = -float("inf")

            # =============================================
            # Step 4: ε-greedy action selection
            # =============================================

            # If evaluating OR random > epsilon: pick best action
            # If training AND random < epsilon: pick random action
            if not evaluate and np.random.random() < self.epsilon:
                # EXPLORE: Random action
                if avail_actions is not None:
                    # Only pick from available actions
                    actions = []
                    for i in range(self.n_agents):
                        avail = np.where(avail_actions[i] == 1)[0]
                        actions.append(np.random.choice(avail))
                    actions = np.array(actions)
                else:
                    actions = np.random.randint(0, self.n_actions, self.n_agents)
            else:
                # EXPLOIT: Best action according to Q-values
                actions = q_values.argmax(dim=-1).cpu().numpy()

        return actions

    # =========================================================================
    # TRAINING
    # =========================================================================

    def set_episode_count(self, episode: int):
        """
        Set current episode count for epsilon decay.

        This is called by the trainer to update exploration rate.

        Parameters:
        -----------
        episode : int
            Current episode number
        """
        self.episode_count = episode

    def update_epsilon(self):
        """
        Update exploration rate (ε) based on episode count.

        EPSILON DECAY FORMULA:
        ε = ε_end + (1.0 - ε_end) × (1 - episode / decay_episodes)

        This creates a linear decay from ε_start to ε_end over decay_episodes.

        Example (with default values):
        - Episode 0: ε = 1.0 (100% random)
        - Episode 25000: ε = 0.525 (52.5% random)
        - Episode 50000+: ε = 0.05 (5% random)
        """
        episode = getattr(self, "episode_count", 0)

        if episode < self.epsilon_decay:
            # Linear decay: high exploration → low exploration
            progress = episode / self.epsilon_decay  # 0.0 to 1.0
            self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * (
                1.0 - progress
            )
        else:
            # After decay period, stay at minimum
            self.epsilon = self.epsilon_end

    def soft_update_target(self, tau: float = 0.005):
        """
        Soft update target network using Polyak averaging.

        Instead of copying weights exactly, we do a weighted average:
        target = τ × online + (1 - τ) × target

        With τ = 0.001, target network updates very slowly,
        providing stable targets for TD learning.

        POLYAK AVERAGING EXPLAINED:
        - τ close to 0: Target changes very slowly (more stable)
        - τ close to 1: Target closely follows online (less stable)
        - τ = 1.0: Hard update (exact copy)

        Parameters:
        -----------
        tau : float
            Interpolation factor (default: 0.005)
        """
        for target_param, online_param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            # Blend target and online weights
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def train_step_fn(self, batch: dict) -> dict:
        """
        Perform one training step on a batch of experience.

        TRAINING STEP OVERVIEW:
        1. Unpack batch data (observations, actions, rewards, etc.)
        2. Forward pass through online model → Q_total
        3. Forward pass through target model → target Q_total
        4. Compute TD loss: (Q_total - target)²
        5. Add VAE loss for identity semantics
        6. Backpropagate and update weights
        7. Soft update target network

        TD LEARNING EXPLAINED:
        ----------------------
        The "Temporal Difference" error is:

            TD_error = Q(s, a) - [r + γ × max_a' Q(s', a')]
                       ↑           ↑        ↑
                    Predicted   Reward   Future value

        We want this error to be zero, so we minimize (TD_error)²

        DOUBLE DQN:
        -----------
        To reduce overestimation, we use two networks:
        1. Online model picks the best next action: a' = argmax Q_online(s')
        2. Target model estimates its value: Q_target(s', a')

        Parameters:
        -----------
        batch : dict
            Contains: observations, actions, rewards, states,
                     next_observations, next_states, dones, avail_actions
            Each has shape [batch_size, seq_len, ...]

        Returns:
        --------
        metrics : dict
            Training metrics (loss, td_loss, vae_loss, epsilon)
        """
        device = self.device

        # =============================================
        # Step 1: Unpack batch data
        # =============================================

        # Convert to tensors and move to device
        observations = torch.as_tensor(
            batch["observations"], device=device, dtype=torch.float32
        )
        actions = torch.as_tensor(batch["actions"], device=device, dtype=torch.long)
        rewards = torch.as_tensor(batch["rewards"], device=device, dtype=torch.float32)
        states = torch.as_tensor(batch["states"], device=device, dtype=torch.float32)
        next_observations = torch.as_tensor(
            batch["next_observations"], device=device, dtype=torch.float32
        )
        next_states = torch.as_tensor(
            batch["next_states"], device=device, dtype=torch.float32
        )
        dones = torch.as_tensor(batch["dones"], device=device, dtype=torch.float32)
        avail_actions = torch.as_tensor(
            batch["avail_actions"], device=device, dtype=torch.float32
        )

        batch_size = observations.size(0)
        seq_len = observations.size(1)
        n_agents = self.n_agents

        # =============================================
        # Step 2: Initialize hidden states
        # =============================================

        # Fresh hidden states for each sequence in the batch
        hidden = self.model.init_hidden(batch_size).to(device)
        target_hidden = self.target_model.init_hidden(batch_size).to(device)

        # Accumulators for losses
        q_totals = []  # Q_total for each timestep
        target_q_totals = []  # Target Q_total for each timestep
        vae_losses = []  # VAE loss for each timestep

        # =============================================
        # Step 3: Process sequence (unroll through time)
        # =============================================

        device_type = "cuda" if device.type == "cuda" else "cpu"

        with torch.amp.autocast(device_type, enabled=self.use_amp):
            for t in range(seq_len):
                # Get data for timestep t
                obs_t = observations[:, t]  # [batch, n_agents, obs_dim]
                state_t = states[:, t]  # [batch, state_dim]
                actions_t = actions[:, t]  # [batch, n_agents]

                # Handle episode boundaries (reset hidden when done)
                done_t = (
                    dones[:, t]
                    if dones.dim() == 2
                    else torch.zeros(batch_size, device=device)
                )
                prev_done = (
                    dones[:, t - 1]
                    if t > 0 and dones.dim() == 2
                    else torch.zeros(batch_size, device=device)
                )

                # Reset hidden states for episodes that ended
                prev_done_expanded = prev_done.repeat_interleave(n_agents).view(
                    batch_size * n_agents, 1
                )
                hidden = hidden * (1 - prev_done_expanded.expand_as(hidden).float())
                target_hidden = target_hidden * (
                    1 - prev_done_expanded.expand_as(target_hidden).float()
                )

                # Previous actions (for action-conditioned network)
                prev_actions_t = (
                    actions[:, t - 1] if t > 0 else torch.zeros_like(actions_t)
                )

                # =============================================
                # Forward pass through ONLINE model
                # =============================================
                result = self.model(obs_t, hidden, state_t, actions_t, prev_actions_t)

                q_totals.append(result["q_total"])  # Team Q-value
                vae_losses.append(result["vae_loss"])  # VAE reconstruction loss
                hidden = result["hidden_states"]  # Update hidden state

                # =============================================
                # Compute TARGET Q-value (Double DQN)
                # =============================================
                with torch.no_grad():
                    next_obs_t = next_observations[:, min(t, seq_len - 1)]
                    next_state_t = next_states[:, min(t, seq_len - 1)]

                    # Reset target hidden for done episodes
                    done_t_expanded = done_t.repeat_interleave(n_agents).view(
                        batch_size * n_agents, 1
                    )
                    target_hidden = target_hidden * (
                        1 - done_t_expanded.expand_as(target_hidden).float()
                    )

                    # DOUBLE DQN:
                    # 1. Online model selects best action
                    online_q_values, _ = self.model.get_q_values(
                        next_obs_t, hidden.detach(), actions_t
                    )
                    next_actions = online_q_values.argmax(dim=-1)

                    # 2. Target model evaluates that action
                    target_result = self.target_model(
                        next_obs_t, target_hidden, next_state_t, next_actions, actions_t
                    )
                    target_q_totals.append(target_result["q_total"])
                    target_hidden = target_result["hidden_states"]

            # Stack across time dimension
            q_totals = torch.stack(q_totals, dim=1).squeeze(-1)  # [batch, seq_len]
            target_q_totals = torch.stack(target_q_totals, dim=1).squeeze(
                -1
            )  # [batch, seq_len]
            vae_loss = torch.stack(vae_losses).mean()  # Scalar

        # =============================================
        # Step 4: Compute TD Loss
        # =============================================

        # Ensure rewards/dones have correct shape
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(0)
        if dones.dim() == 1:
            dones = dones.unsqueeze(0)

        if rewards.size(0) != target_q_totals.size(0):
            rewards = rewards.expand(target_q_totals.size(0), -1)
            dones = dones.expand(target_q_totals.size(0), -1)

        # TD TARGET: r + γ × (1 - done) × Q_target(s', a')
        # If done=1, we don't add future value (episode ended)
        targets = rewards + self.gamma * (1 - dones) * target_q_totals

        # Clip values for numerical stability
        q_totals_clipped = torch.clamp(q_totals, -100.0, 400.0)
        targets_clipped = torch.clamp(targets.detach(), -100.0, 400.0)

        # Smooth L1 loss (Huber loss) - less sensitive to outliers than MSE
        td_loss = F.smooth_l1_loss(q_totals_clipped, targets_clipped)

        # =============================================
        # Step 5: Compute Total Loss
        # =============================================

        # Warmup VAE loss (start small, gradually increase)
        warmup_steps = 15000
        vae_weight = self.vae_loss_weight * min(1.0, self.train_step / warmup_steps)

        total_loss = td_loss + vae_weight * vae_loss

        # =============================================
        # Step 6: Backpropagation and Optimization
        # =============================================

        # Zero gradients from previous step
        self.q_optimizer.zero_grad(set_to_none=True)
        self.vae_optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            # Mixed precision training (faster on GPU)
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.q_optimizer)
            self.scaler.unscale_(self.vae_optimizer)

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.scaler.step(self.q_optimizer)
            self.scaler.step(self.vae_optimizer)
            self.scaler.update()
        else:
            # Standard training
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.q_optimizer.step()
            self.vae_optimizer.step()

        # =============================================
        # Step 7: Update counters and target network
        # =============================================

        self.train_step += 1

        # Soft update target network (τ = 0.001 for stability)
        self.soft_update_target(tau=0.001)

        # Update exploration rate
        self.update_epsilon()

        # Return training metrics
        return {
            "loss": total_loss.item(),
            "td_loss": td_loss.item(),
            "vae_loss": vae_loss.item(),
            "epsilon": self.epsilon,
        }

    # =========================================================================
    # CHECKPOINTING (Save/Load Models)
    # =========================================================================

    def save(self, path: str):
        """
        Save model checkpoint to file.

        Saves everything needed to resume training or evaluate:
        - Model weights (online and target)
        - Optimizer states
        - Training step count
        - Current epsilon

        Parameters:
        -----------
        path : str
            File path to save checkpoint (e.g., "best_model.pt")
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "q_optimizer_state_dict": self.q_optimizer.state_dict(),
                "vae_optimizer_state_dict": self.vae_optimizer.state_dict(),
                "train_step": self.train_step,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str):
        """
        Load model checkpoint from file.

        Note: Epsilon is NOT restored - it will be recalculated
        based on the current episode count. This allows resuming
        training with proper exploration.

        Parameters:
        -----------
        path : str
            File path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])

        # Load optimizer states
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.vae_optimizer.load_state_dict(checkpoint["vae_optimizer_state_dict"])

        # Load training step
        self.train_step = checkpoint["train_step"]

        # Note: epsilon NOT restored - will be recalculated from episode count

    # =========================================================================
    # INTERPRETABILITY (Understand Agent Behavior)
    # =========================================================================

    def get_interpretable_contributions(
        self, observations: np.ndarray, state: np.ndarray, actions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get interpretable contributions for visualization.

        This method shows HOW much each agent contributed to the team decision,
        which is useful for understanding and debugging agent behavior.

        Returns:
        --------
        dict with:
            - individual_contribs: How much each agent contributed [n_agents]
            - pairwise_contribs: How much each pair contributed [n_pairs]
            - attention_weights: Credit assignment weights [n_shape_functions]
            - masks: Semantic masks from VAE [n_agents, hidden_dim]
            - q_total: Final team Q-value (scalar)

        Example:
        --------
        >>> contribs = agent.get_interpretable_contributions(obs, state, actions)
        >>> print("Agent 0 contribution:", contribs['individual_contribs'][0])
        >>> print("Team Q-value:", contribs['q_total'])
        """
        with torch.no_grad():
            # Prepare inputs
            obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            actions_tensor = torch.LongTensor(actions).unsqueeze(0).to(self.device)

            # Initialize hidden if needed
            if self.hidden_states is None:
                self.init_hidden(1)

            # Forward pass
            result = self.model(
                obs_tensor, self.hidden_states, state_tensor, actions_tensor
            )

            # Extract and return contributions
            return {
                "individual_contribs": result["individual_contribs"]
                .squeeze(0)
                .cpu()
                .numpy(),
                "pairwise_contribs": result["pairwise_contribs"]
                .squeeze(0)
                .cpu()
                .numpy(),
                "attention_weights": result["attention_weights"]
                .squeeze(0)
                .cpu()
                .numpy(),
                "masks": result["masks"].squeeze(0).cpu().numpy(),
                "q_total": result["q_total"].squeeze().cpu().numpy(),
            }
