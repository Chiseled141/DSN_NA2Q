"""
=============================================================================
HiT-MAC PERCEPTION MODULES
=============================================================================

WHAT IS THIS FILE?
------------------
This file contains the core neural network building blocks for HiT-MAC:
- NoisyLinear: Adds learned noise for exploration (no epsilon-greedy needed!)
- BiRNN: Bidirectional RNN for sequence encoding
- AttentionLayer: Scaled dot-product attention (like in Transformers)

WHY DO WE NEED THESE?
---------------------
In multi-agent RL, sensors need to understand their environment and
coordinate with each other. These modules help by:

1. NoisyLinear → Better exploration without manual epsilon tuning
2. BiRNN → Process sequences of observations over time
3. Attention → Focus on the most important targets/agents

ATTENTION MECHANISM DIAGRAM:
----------------------------
                    Input Features
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
        ┌─────┐       ┌─────┐       ┌─────┐
        │  Q  │       │  K  │       │  V  │
        │Query│       │ Key │       │Value│
        └──┬──┘       └──┬──┘       └──┬──┘
           │             │             │
           └──────┬──────┘             │
                  ▼                    │
           ┌───────────┐               │
           │  Q × K^T  │               │
           │ (scores)  │               │
           └─────┬─────┘               │
                 ▼                     │
           ┌───────────┐               │
           │  Softmax  │               │
           │ (weights) │               │
           └─────┬─────┘               │
                 │                     │
                 └────────┬────────────┘
                          ▼
                   ┌─────────────┐
                   │ Weights × V │
                   │  (output)   │
                   └─────────────┘

The attention mechanism learns to focus on important features by:
1. Creating Query, Key, Value projections from input
2. Computing similarity scores between Query and Keys
3. Using softmax to get attention weights (sum to 1)
4. Weighted sum of Values gives the output
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn import Parameter

# =============================================================================
# NOISY LINEAR LAYER
# =============================================================================


class NoisyLinear(nn.Linear):
    """
    Linear layer with learnable noise for exploration.

    WHAT IS THIS?
    -------------
    Instead of using epsilon-greedy (random actions with probability ε),
    NoisyLinear adds learnable noise directly to network weights.
    The network learns WHEN to explore vs exploit!

    HOW IT WORKS:
    -------------
    Normal linear:    y = Wx + b
    Noisy linear:     y = (W + σ_w × ε_w)x + (b + σ_b × ε_b)

    Where:
    - W, b: Learned weights and biases
    - σ_w, σ_b: Learned noise scales (how much to explore)
    - ε_w, ε_b: Random noise (sampled each forward pass)

    Parameters:
    -----------
    in_features : int
        Input dimension
    out_features : int
        Output dimension
    sigma_init : float
        Initial noise scale (default: 0.017)

    Example:
    --------
    >>> layer = NoisyLinear(64, 32)
    >>> layer.sample_noise()  # Sample new noise
    >>> output = layer(input)  # Forward with noise
    >>> layer.remove_noise()   # Disable noise for evaluation
    """

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)

        # =============================================
        # Noise scale parameters (LEARNED)
        # =============================================
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = Parameter(torch.Tensor(out_features))

        # =============================================
        # Noise samples (NOT learned, just random)
        # =============================================
        # register_buffer: saved with model but not trained
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with uniform distribution, noise scales with constant."""
        if hasattr(self, "sigma_weight"):
            # Uniform initialization for weights
            bound = math.sqrt(3 / self.in_features)
            init.uniform_(self.weight, -bound, bound)
            init.uniform_(self.bias, -bound, bound)

            # Constant initialization for noise scales
            init.constant_(self.sigma_weight, self.sigma_init)
            init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        """
        Forward pass with noisy weights.

        y = (W + σ_w × ε_w)x + (b + σ_b × ε_b)
        """
        noisy_weight = self.weight + self.sigma_weight * Variable(self.epsilon_weight)
        noisy_bias = self.bias + self.sigma_bias * Variable(self.epsilon_bias)
        return F.linear(input, noisy_weight, noisy_bias)

    def sample_noise(self):
        """Sample new random noise for exploration."""
        self.epsilon_weight.data.copy_(torch.randn(self.out_features, self.in_features))
        self.epsilon_bias.data.copy_(torch.randn(self.out_features))

    def remove_noise(self):
        """Set noise to zero for deterministic evaluation."""
        self.epsilon_weight.data.zero_()
        self.epsilon_bias.data.zero_()


# =============================================================================
# BIDIRECTIONAL RNN ENCODER
# =============================================================================


class BiRNN(torch.nn.Module):
    """
    Bidirectional RNN encoder (LSTM or GRU).

    WHAT IS THIS?
    -------------
    Processes sequences in BOTH directions (forward and backward),
    then concatenates the hidden states. This captures both:
    - Past context (what happened before)
    - Future context (what happens after)

    DIAGRAM:
    --------
    Input sequence: [x₁, x₂, x₃, x₄, x₅]

    Forward RNN:   x₁ → x₂ → x₃ → x₄ → x₅ → h_forward
    Backward RNN:  x₁ ← x₂ ← x₃ ← x₄ ← x₅ → h_backward

    Output: [h_forward; h_backward]  (concatenated)

    Parameters:
    -----------
    input_size : int
        Dimension of input features
    hidden_size : int
        Hidden state dimension (output will be 2× this)
    num_layers : int
        Number of stacked RNN layers
    device : torch.device
        CPU or GPU
    head_name : str
        Contains 'lstm' for LSTM, else uses GRU
    """

    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # =============================================
        # Choose RNN type based on head_name
        # =============================================
        # LSTM: Has both hidden state (h) and cell state (c)
        # GRU: Only has hidden state, simpler and often faster
        if "lstm" in head_name:
            self.lstm = True
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,  # Input shape: [batch, seq, features]
                bidirectional=True,  # Process both directions
            ).to(device)
        else:
            self.lstm = False
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
            ).to(device)

        # Output dimension is 2× hidden_size (forward + backward)
        self.feature_dim = hidden_size * 2

    def forward(self, x, state=None):
        """
        Process sequence through bidirectional RNN.

        Parameters:
        -----------
        x : torch.Tensor
            Input sequence [batch, seq_len, input_size]

        Returns:
        --------
        out : torch.Tensor
            All hidden states [batch, seq_len, hidden_size * 2]
        hn : torch.Tensor
            Final hidden states [num_layers * 2, batch, hidden_size]
        """
        # Initialize hidden states to zeros
        # Factor of 2 for bidirectional
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        if self.lstm:
            out, (hn, _) = self.rnn(x, (h0, c0))
        else:
            out, hn = self.rnn(x, h0)

        return out, hn


# =============================================================================
# UNIDIRECTIONAL RNN ENCODER
# =============================================================================


class RNN(torch.nn.Module):
    """
    Unidirectional RNN encoder (LSTM or GRU).

    Same as BiRNN but only processes in forward direction.
    Faster but less contextual understanding.
    """

    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        if "lstm" in head_name:
            self.lstm = True
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        else:
            self.lstm = False
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)

        self.feature_dim = hidden_size

    def forward(self, x, state=None):
        """Process sequence through unidirectional RNN."""
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        if self.lstm:
            out, (hn, _) = self.rnn(x, (h0, c0))
        else:
            out, hn = self.rnn(x, h0)

        return out, hn


def xavier_init(layer):
    """Initialize layer with Xavier uniform for stable gradients."""
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.constant_(layer.bias, 0)
    return layer


# =============================================================================
# ATTENTION LAYER
# =============================================================================


class AttentionLayer(torch.nn.Module):
    """
    Scaled dot-product attention mechanism.

    WHAT IS THIS?
    -------------
    Attention allows the model to focus on the most relevant parts
    of the input. Like how you focus on important words when reading.

    THE Q-K-V PARADIGM:
    -------------------
    - Query (Q): "What am I looking for?"
    - Key (K): "What do I have to offer?"
    - Value (V): "What information do I contain?"

    Attention score = how well Query matches each Key
    Output = weighted sum of Values based on attention scores

    MATH:
    -----
    Attention(Q, K, V) = softmax(Q × K^T) × V

    Parameters:
    -----------
    feature_dim : int
        Input feature dimension
    weight_dim : int
        Output feature dimension (Q, K, V projection size)
    device : torch.device
        CPU or GPU

    Example:
    --------
    >>> attention = AttentionLayer(feature_dim=64, weight_dim=128, device='cpu')
    >>> x = torch.randn(1, 10, 64)  # batch=1, seq_len=10, features=64
    >>> attended, global_feat = attention(x)
    >>> print(attended.shape)  # [1, 10, 128]
    >>> print(global_feat.shape)  # [1, 128]
    """

    def __init__(self, feature_dim, weight_dim, device):
        super(AttentionLayer, self).__init__()
        self.in_dim = feature_dim
        self.device = device

        # =============================================
        # Q, K, V projection layers
        # =============================================
        # Each transforms input to weight_dim dimensions
        self.Q = xavier_init(nn.Linear(self.in_dim, weight_dim))  # Query
        self.K = xavier_init(nn.Linear(self.in_dim, weight_dim))  # Key
        self.V = xavier_init(nn.Linear(self.in_dim, weight_dim))  # Value

        self.feature_dim = weight_dim

    def forward(self, x):
        """
        Compute attention over input sequence.

        Parameters:
        -----------
        x : torch.Tensor
            Input features [batch_size, sequence_len, feature_dim]

        Returns:
        --------
        z : torch.Tensor
            Attended features [batch_size, sequence_len, weight_dim]
        global_feature : torch.Tensor
            Sum of attended features [batch_size, weight_dim]
        """
        # Step 1: Project input to Q, K, V
        # tanh activation for bounded values
        q = torch.tanh(self.Q(x))  # [batch, seq, weight_dim]
        k = torch.tanh(self.K(x))  # [batch, seq, weight_dim]
        v = torch.tanh(self.V(x))  # [batch, seq, weight_dim]

        # Step 2: Compute attention scores
        # Q × K^T gives [batch, seq, seq] similarity matrix
        scores = torch.bmm(q, k.permute(0, 2, 1))

        # Step 3: Softmax to get attention weights (sum to 1)
        weights = F.softmax(scores, dim=2)

        # Step 4: Weighted sum of values
        z = torch.bmm(weights, v)  # [batch, seq, weight_dim]

        # Step 5: Global feature is sum across sequence
        global_feature = z.sum(dim=1)  # [batch, weight_dim]

        return z, global_feature
