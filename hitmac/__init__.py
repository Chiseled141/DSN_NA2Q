"""
=============================================================================
HiT-MAC: HIERARCHICAL TWIN-ACTOR MULTI-AGENT COORDINATION
=============================================================================

WHAT IS THIS PACKAGE?
---------------------
HiT-MAC (Hierarchical Twin-Actor Multi-Agent Coordination) is a hierarchical
reinforcement learning approach for multi-agent coordination problems.

Uses the same DSNEnv environment as NA²Q for fair algorithm comparison.

ARCHITECTURE:
-------------
    ┌─────────────────────────────────────────────────────────┐
    │                    COORDINATOR                           │
    │  High-level policy that assigns targets to sensors       │
    │  Uses attention over all agents for global coordination  │
    └────────────────────────┬────────────────────────────────┘
                             │ Target assignments
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │                     EXECUTOR                             │
    │  Low-level policy that controls individual sensors       │
    │  Trained to track assigned targets efficiently           │
    └─────────────────────────────────────────────────────────┘

KEY COMPONENTS:
---------------
- models.py:       A3C models with attention (Single/Multi-agent)
- train.py:        A3C training worker process
- test.py:         Evaluation and checkpoint saving
- player.py:       Agent class for episode collection
- perception.py:   CNN-based feature extraction
- shared_optim.py: Shared memory optimizers for multi-process training

USAGE:
------
    # Train on scenario 1
    python -m hitmac.main --mode train --scenario 1
    
    # Evaluate trained model
    python -m hitmac.main --mode test --scenario 1

REFERENCE:
----------
Based on NeurIPS 2020 paper on hierarchical multi-agent coordination.
"""

from hitmac.models import A3C_Single, A3C_Multi, build_model

__all__ = ['A3C_Single', 'A3C_Multi', 'build_model']
