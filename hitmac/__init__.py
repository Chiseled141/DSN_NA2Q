"""
HiT-MAC: Hierarchical Twin-Actor Multi-Agent Coordination

A hierarchical multi-agent reinforcement learning algorithm for
Directional Sensor Networks, featuring coordinator and executor policies.

Paper: "Learning Multi-Agent Coordination for Enhancing Target Coverage
in Directional Sensor Networks" (NeurIPS 2020)
"""

from hitmac.models import build_model, A3C_Single, A3C_Multi
from hitmac.perception import AttentionLayer, BiRNN, NoisyLinear

__all__ = [
    'build_model',
    'A3C_Single',
    'A3C_Multi',
    'AttentionLayer',
    'BiRNN',
    'NoisyLinear',
]
