"""
=============================================================================
HiT-MAC NEURAL NETWORK MODELS
=============================================================================

WHAT IS THIS FILE?
------------------
This file contains the core neural network architectures for HiT-MAC:
- A3C_Single: The "Executor" - controls individual sensors
- A3C_Multi: The "Coordinator" - assigns targets to sensors

THE HIERARCHICAL ARCHITECTURE:
------------------------------
HiT-MAC uses a two-level hierarchy:

    Level 1: COORDINATOR (A3C_Multi)
    ┌─────────────────────────────────────────────────────────┐
    │  "Which sensor should track which target?"              │
    │                                                         │
    │  Input: All sensor-target observations                  │
    │  Output: Binary assignments (sensor i tracks target j)  │
    │  Credit: Uses Shapley values for fair reward distribution│
    └─────────────────────────────────────────────────────────┘
                           │
                           ▼
    Level 2: EXECUTOR (A3C_Single)
    ┌─────────────────────────────────────────────────────────┐
    │  "How should sensor i orient to track its targets?"     │
    │                                                         │
    │  Input: Single sensor's view of targets                 │
    │  Output: Discrete orientation action                    │
    │  Focus: Individual sensor control                       │
    └─────────────────────────────────────────────────────────┘

WHAT IS A3C?
------------
A3C = Asynchronous Advantage Actor-Critic

    Actor:  Decides WHAT ACTION to take (policy)
    Critic: Evaluates HOW GOOD the state is (value function)
    
    The "advantage" = actual reward - expected value
    If positive: action was better than expected → encourage it
    If negative: action was worse than expected → discourage it

SHAPLEY VALUES (Optional):
-------------------------
Shapley values fairly distribute credit among agents:
- How much did each agent contribute to team success?
- Computed by looking at marginal contributions to coalitions
- Prevents free-rider problem (agents that don't help but share reward)

Based on: "Learning Multi-Agent Coordination for Enhancing Target Coverage
in Directional Sensor Networks" (NeurIPS 2020)
"""

from __future__ import division
import torch
import numpy as np
import torch.nn as nn
from gym import spaces
import torch.nn.functional as F
from torch.autograd import Variable

from hitmac.utils import norm_col_init, weights_init
from hitmac.perception import NoisyLinear, BiRNN, AttentionLayer


# =============================================================================
# MODEL FACTORY
# =============================================================================

def build_model(obs_space, action_space, args, device):
    """
    Build the appropriate model based on args.model name.
    
    Model naming convention:
    - 'single-*': Uses A3C_Single (executor)
    - 'multi-*': Uses A3C_Multi (coordinator)
    - '*-att': Uses attention mechanism
    - '*-shap': Uses Shapley value critic
    - '*-ns': Uses noisy layers for exploration
    
    Example model names:
    - 'single-att': Executor with attention
    - 'multi-att-shap': Coordinator with attention and Shapley values
    
    Parameters:
    -----------
    obs_space : gym.Space
        Observation space(s)
    action_space : gym.Space
        Action space(s)
    args : argparse.Namespace
        Must contain: model (str), lstm_out (int)
    device : torch.device
        CPU or GPU
    
    Returns:
    --------
    model : nn.Module
        Initialized model in training mode
    """
    name = args.model

    if 'single' in name:
        model = A3C_Single(obs_space, action_space, args, device)
    elif 'multi' in name:
        model = A3C_Multi(obs_space, action_space, args, device)
    else:
        raise ValueError(f"Unknown model type: {name}. Use 'single-*' or 'multi-*'")

    model.train()
    return model


# =============================================================================
# ACTION SAMPLING
# =============================================================================

def wrap_action(self, action):
    """Scale action from [-1, 1] to environment action range."""
    action = np.squeeze(action)
    out = action * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2.0
    return out


def sample_action(mu_multi, sigma_multi, device, test=False):
    """
    Sample discrete action from policy logits.
    
    HOW IT WORKS:
    -------------
    1. Convert logits to probabilities with softmax
    2. Either:
       - Test mode: Pick action with highest probability (greedy)
       - Train mode: Sample from probability distribution (exploration)
    3. Compute entropy for exploration bonus
    
    Parameters:
    -----------
    mu_multi : torch.Tensor
        Policy logits (unnormalized log-probabilities)
    sigma_multi : torch.Tensor
        Unused (kept for continuous action interface compatibility)
    device : torch.device
        CPU or GPU
    test : bool
        If True, use greedy action selection
    
    Returns:
    --------
    action_env : numpy array
        Selected actions for environment
    entropy : torch.Tensor
        Policy entropy (higher = more exploration)
    log_prob : torch.Tensor
        Log probability of selected actions (for policy gradient)
    """
    logit = mu_multi
    
    # Softmax: convert logits to probabilities (sum to 1)
    prob = F.softmax(logit, dim=-1)
    
    # Log softmax for numerical stability
    log_prob = F.log_softmax(logit, dim=-1)
    
    # Entropy = -sum(p * log(p))
    # High entropy = uncertain/exploring, low entropy = confident
    entropy = -(log_prob * prob).sum(-1, keepdim=True)
    
    if test:
        # Greedy: pick action with highest probability
        action = prob.max(-1)[1].data
        action_env = action.cpu().numpy()
    else:
        # Stochastic: sample from distribution
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        action_env = action.squeeze(0)

    return action_env, entropy, log_prob


# =============================================================================
# VALUE NETWORK (Critic)
# =============================================================================

class ValueNet(nn.Module):
    """
    Value network for critic in A3C.
    
    WHAT IS THIS?
    -------------
    Estimates V(s) = expected future reward from state s.
    Used to compute advantage: A(s,a) = Q(s,a) - V(s)
    
    Think of it as asking: "How good is this situation, regardless of action?"
    
    Parameters:
    -----------
    input_dim : int
        Feature dimension from encoder
    head_name : str
        If contains 'ns', uses noisy layers for exploration
    num : int
        Output dimension (typically 1 for single value)
    """
    
    def __init__(self, input_dim, head_name, num=1):
        super(ValueNet, self).__init__()
        
        # Use noisy layer if 'ns' in head_name
        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            # Initialize with small weights for stable training
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

    def forward(self, x):
        """Compute state value estimate."""
        value = self.critic_linear(x)
        return value

    def sample_noise(self):
        """Sample new noise for exploration."""
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        """Remove noise for deterministic evaluation."""
        if self.noise:
            self.critic_linear.remove_noise()


# =============================================================================
# SHAPLEY VALUE CRITIC (AMC Value Network)
# =============================================================================

class AMCValueNet(nn.Module):
    """
    Attention-based Multi-agent Credit (AMC) Value Network.
    
    WHAT IS THIS?
    -------------
    Uses Shapley value-inspired credit assignment to fairly distribute
    rewards among agents based on their MARGINAL CONTRIBUTIONS.
    
    SHAPLEY VALUES EXPLAINED:
    -------------------------
    Imagine a team project. How much did each person contribute?
    
    Shapley's idea: Look at all possible orderings of team members.
    For each ordering, see what each person adds when they join.
    Average these marginal contributions = fair credit.
    
    MARGINAL CONTRIBUTION DIAGRAM:
    ------------------------------
    Coalition: {}        → Add Agent 1 → {A1}     (contribution: V({A1}) - V({}))
    Coalition: {A1}      → Add Agent 2 → {A1,A2}  (contribution: V({A1,A2}) - V({A1}))
    Coalition: {A1,A2}   → Add Agent 3 → {A1,A2,A3} (contribution: V({A1,A2,A3}) - V({A1,A2}))
    
    Total value = sum of all marginal contributions
    
    ATTENTION-BASED APPROXIMATION:
    ------------------------------
    Computing exact Shapley values is O(n!) - infeasible for many agents.
    We approximate by:
    1. Processing agents in order
    2. Using attention to aggregate coalition context
    3. Computing marginal contribution given context
    
    Parameters:
    -----------
    input_dim : int
        Feature dimension from encoder
    head_name : str
        Model configuration string
    num : int
        Output dimension (typically 1)
    device : torch.device
        CPU or GPU
    """
    
    def __init__(self, input_dim, head_name, num=1, device=torch.device('cpu')):
        super(AMCValueNet, self).__init__()
        self.head_name = head_name
        self.device = device

        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        elif 'onlyJ' in head_name:
            # Joint value only (no coalition context)
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)
        else:
            # Full Shapley: uses 2*input_dim (coalition context + agent feature)
            self.noise = False
            self.critic_linear = nn.Linear(2 * input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

            # Attention for aggregating coalition features
            self.attention = AttentionLayer(input_dim, input_dim, device)
            
        self.feature_dim = input_dim

    def forward(self, x, goal):
        """
        Compute value using Shapley-like marginal contributions.
        
        Parameters:
        -----------
        x : torch.Tensor
            Agent features [n_agents, feature_dim]
        goal : torch.Tensor
            Action goals (currently unused)
        
        Returns:
        --------
        value : torch.Tensor
            Aggregated value estimate [1]
        """
        _, feature_dim = x.shape
        value = []

        coalition = x.view(-1, feature_dim)
        n = coalition.shape[0]

        # =============================================
        # First agent: no coalition context yet
        # =============================================
        feature = torch.zeros([self.feature_dim]).to(self.device)
        value.append(self.critic_linear(torch.cat([feature, coalition[0]])))
        
        # =============================================
        # Subsequent agents: marginal contribution given previous agents
        # =============================================
        for j in range(1, n):
            # Use attention to get coalition summary
            _, feature = self.attention(coalition[:j].unsqueeze(0))
            # Marginal contribution: Value(coalition + agent_j) - Value(coalition)
            value.append(self.critic_linear(torch.cat([feature.squeeze(), coalition[j]])))

        # Sum all marginal contributions
        value = torch.cat(value).sum()

        return value.unsqueeze(0)

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.critic_linear.remove_noise()


# =============================================================================
# POLICY NETWORK (Actor)
# =============================================================================

class PolicyNet(nn.Module):
    """
    Policy network for discrete action selection.
    
    WHAT IS THIS?
    -------------
    The "actor" in actor-critic. Decides what action to take.
    
    Outputs logits (unnormalized scores) which are:
    1. Converted to probabilities with softmax
    2. Used to sample actions during training
    3. Used to select greedy action during testing
    
    Parameters:
    -----------
    input_dim : int
        Feature dimension from encoder
    action_space : gym.Space
        Discrete action space
    head_name : str
        If contains 'ns', uses noisy layers
    device : torch.device
        CPU or GPU
    """
    
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNet, self).__init__()
        self.head_name = head_name
        self.device = device
        num_outputs = action_space.n  # Number of discrete actions

        if 'ns' in head_name:
            self.noise = True
            self.actor_linear = NoisyLinear(input_dim, num_outputs, sigma_init=0.017)
        else:
            self.noise = False
            self.actor_linear = nn.Linear(input_dim, num_outputs)
            self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.1)
            self.actor_linear.bias.data.fill_(0)

    def forward(self, x, test=False):
        """
        Select action from policy.
        
        Parameters:
        -----------
        x : torch.Tensor
            Feature vector [batch, feature_dim]
        test : bool
            If True, use greedy selection
        
        Returns:
        --------
        action : numpy array
            Selected actions
        entropy : torch.Tensor
            Policy entropy
        log_prob : torch.Tensor
            Log probability of actions
        """
        # ReLU to ensure positive logits
        mu = F.relu(self.actor_linear(x))
        sigma = torch.ones_like(mu)  # Unused, for interface compatibility
        action, entropy, log_prob = sample_action(mu, sigma, self.device, test)
        return action, entropy, log_prob

    def sample_noise(self):
        if self.noise:
            self.actor_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.actor_linear.remove_noise()


# =============================================================================
# ENCODER NETWORKS
# =============================================================================

class EncodeBiRNN(torch.nn.Module):
    """
    Bidirectional RNN encoder for sequence features.
    
    Processes observation sequences and outputs both:
    - Per-timestep features
    - Global summary feature
    """
    
    def __init__(self, dim_in, lstm_out=128, head_name='birnn_lstm', device=None):
        super(EncodeBiRNN, self).__init__()
        self.head_name = head_name

        # BiRNN with half the output size (because bidirectional doubles it)
        self.encoder = BiRNN(dim_in, int(lstm_out / 2), 1, device, 'gru')

        self.feature_dim = self.encoder.feature_dim
        self.global_feature_dim = self.encoder.feature_dim
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = inputs
        cn, hn = self.encoder(x)

        feature = cn  # All hidden states [batch, seq_len, features]
        global_feature = hn.permute(1, 0, 2).reshape(-1)  # Final hidden state

        return feature, global_feature


class EncodeLinear(torch.nn.Module):
    """
    Simple MLP encoder for observation features.
    
    Two-layer MLP with ReLU activations.
    Used as the first encoding stage in A3C_Multi.
    """
    
    def __init__(self, dim_in, dim_out=32, head_name='lstm', device=None):
        super(EncodeLinear, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(inplace=True)
        )

        self.head_name = head_name
        self.feature_dim = dim_out
        self.train()

    def forward(self, inputs):
        x = inputs
        feature = self.features(x)
        return feature


# =============================================================================
# A3C_SINGLE: EXECUTOR MODEL
# =============================================================================

class A3C_Single(torch.nn.Module):
    """
    Single-agent Executor model for HiT-MAC.
    
    WHAT IS THIS?
    -------------
    The "low-level" controller that controls individual sensor orientation.
    Each sensor has its own executor that decides how to orient.
    
    ARCHITECTURE DIAGRAM:
    ---------------------
    ┌─────────────────────────────────────────────────────────┐
    │                      A3C_Single                         │
    │                                                         │
    │   Observation      ┌───────────────┐                    │
    │   [n_targets,      │   Attention   │    Feature         │
    │    obs_dim]   ────▶│    Encoder    │────▶[lstm_out]     │
    │                    └───────────────┘                    │
    │                           │                             │
    │              ┌────────────┴────────────┐                │
    │              ▼                         ▼                │
    │       ┌──────────┐              ┌──────────┐            │
    │       │  Actor   │              │  Critic  │            │
    │       │ (Policy) │              │ (Value)  │            │
    │       └────┬─────┘              └────┬─────┘            │
    │            ▼                         ▼                  │
    │         Action                     V(s)                 │
    │       (discrete)                 (scalar)               │
    └─────────────────────────────────────────────────────────┘
    
    Parameters:
    -----------
    obs_space : list of gym.Space
        Observation spaces for each sensor
    action_spaces : list of gym.Space
        Action spaces for each sensor
    args : argparse.Namespace
        Must contain: model (str), lstm_out (int)
    device : torch.device
        CPU or GPU
    """
    
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Single, self).__init__()
        self.n = len(obs_space)  # Number of sensors
        obs_dim = obs_space[0].shape[1]  # Observation dimension

        lstm_out = args.lstm_out
        head_name = args.model

        self.head_name = head_name

        # Attention encoder: processes observation → feature vector
        self.encoder = AttentionLayer(obs_dim, lstm_out, device)
        
        # Critic: estimates state value
        self.critic = ValueNet(lstm_out, head_name, 1)
        
        # Actor: outputs action probabilities
        self.actor = PolicyNet(lstm_out, action_spaces[0], head_name, device)

        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        """
        Forward pass through executor.
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Observations [batch, n_targets, obs_dim]
        test : bool
            If True, use greedy action selection
        
        Returns:
        --------
        values : torch.Tensor
            State value estimates [batch, 1]
        actions : numpy array
            Selected actions
        entropies : torch.Tensor
            Policy entropies
        log_probs : torch.Tensor
            Log probabilities of selected actions
        """
        data = Variable(inputs, requires_grad=True)
        
        # Attention encoder: [batch, n_targets, obs_dim] → [batch, lstm_out]
        _, feature = self.encoder(data)

        # Actor: feature → action
        actions, entropies, log_probs = self.actor(feature, test)
        
        # Critic: feature → value
        values = self.critic(feature)

        return values, actions, entropies, log_probs

    def sample_noise(self):
        """Sample new noise for exploration."""
        self.actor.sample_noise()
        self.critic.sample_noise()

    def remove_noise(self):
        """Remove noise for evaluation."""
        self.actor.remove_noise()
        self.critic.remove_noise()


# =============================================================================
# A3C_MULTI: COORDINATOR MODEL
# =============================================================================

class A3C_Multi(torch.nn.Module):
    """
    Multi-agent Coordinator model for HiT-MAC.
    
    WHAT IS THIS?
    -------------
    The "high-level" controller that assigns targets to sensors.
    It sees all sensors and all targets, then decides the assignment.
    
    ARCHITECTURE DIAGRAM:
    ---------------------
    ┌─────────────────────────────────────────────────────────────┐
    │                       A3C_Multi                             │
    │                                                             │
    │   Observations         ┌───────────┐      ┌───────────┐    │
    │   [n_agents,           │  Linear   │      │ Attention │    │
    │    n_targets,   ──────▶│  Encoder  │─────▶│   Layer   │    │
    │    pose_dim]           └───────────┘      └─────┬─────┘    │
    │                                                 │          │
    │                        ┌────────────────────────┤          │
    │                        │                        │          │
    │                        ▼                        ▼          │
    │                 ┌──────────┐           ┌────────────────┐  │
    │                 │  Actor   │           │    Critic      │  │
    │                 │ (Policy) │           │ (Shapley/Std)  │  │
    │                 └────┬─────┘           └───────┬────────┘  │
    │                      ▼                         ▼           │
    │               Assignments                    V(s)          │
    │           [n_agents, n_targets]            (scalar)        │
    └─────────────────────────────────────────────────────────────┘
    
    KEY FEATURES:
    -------------
    1. Sees ALL agents and ALL targets (global view)
    2. Uses attention to aggregate features
    3. Optional Shapley value critic for fair credit assignment
    4. Binary output: sensor i tracks target j (yes/no)
    
    Parameters:
    -----------
    obs_space : gym.Space
        Observation space with shape (n_agents, n_targets, pose_dim)
    action_spaces : list of gym.Space
        Action spaces (unused, binary assignment hardcoded)
    args : argparse.Namespace
        Must contain: model (str), lstm_out (int)
    device : torch.device
        CPU or GPU
    """
    
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Multi, self).__init__()
        self.num_agents, self.num_targets, self.pose_dim = obs_space.shape

        lstm_out = args.lstm_out
        head_name = args.model
        self.head_name = head_name

        # Linear encoder: pose_dim → lstm_out
        self.encoder = EncodeLinear(self.pose_dim, lstm_out, head_name, device)
        feature_dim = self.encoder.feature_dim

        # Attention layer: aggregates all features
        self.attention = AttentionLayer(feature_dim, lstm_out, device)
        feature_dim = self.attention.feature_dim

        # Actor: binary decisions (track target or not)
        self.actor = PolicyNet(feature_dim, spaces.Discrete(2), head_name, device)

        # Critic: Shapley value-based or standard
        if 'shap' in head_name:
            # Fair credit assignment using Shapley values
            self.ShapleyVcritic = AMCValueNet(feature_dim, head_name, 1, device)
        else:
            # Standard value network
            self.critic = ValueNet(feature_dim, head_name, 1)

        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        """
        Forward pass through coordinator.
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Observations [n_agents, n_targets, pose_dim]
        test : bool
            If True, use greedy action selection
        
        Returns:
        --------
        values : torch.Tensor
            State value estimates [1]
        actions : torch.Tensor
            Target assignments [n_agents, n_targets, 1]
        entropies : torch.Tensor
            Policy entropies
        log_probs : torch.Tensor
            Log probabilities of actions
        """
        pos_obs = inputs

        # Step 1: Encode each observation independently
        feature_target = Variable(pos_obs, requires_grad=True)
        feature_target = self.encoder(feature_target)

        # Step 2: Reshape for attention [1, n_agents*n_targets, feature_dim]
        feature_target = feature_target.reshape(-1, self.encoder.feature_dim).unsqueeze(0)
        
        # Step 3: Apply attention
        feature, global_feature = self.attention(feature_target)
        feature = feature.squeeze()

        # Step 4: Actor decides assignments
        actions, entropies, log_probs = self.actor(feature, test)
        actions = actions.reshape(self.num_agents, self.num_targets, -1)

        # Step 5: Critic estimates value
        if 'shap' not in self.head_name:
            # Standard critic uses global feature
            values = self.critic(global_feature)
        else:
            # Shapley critic uses per-agent features
            values = self.ShapleyVcritic(feature, actions)

        return values, actions, entropies, log_probs

    def sample_noise(self):
        """Sample new noise for exploration."""
        self.actor.sample_noise()

    def remove_noise(self):
        """Remove noise for evaluation."""
        self.actor.remove_noise()
