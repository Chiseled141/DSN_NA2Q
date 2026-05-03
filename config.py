"""
Configuration for NA²Q.

Contains both scenario presets (environment parameters) and training presets (hyperparameters).
"""

import os
from copy import deepcopy
from typing import Dict

# =============================================================================
# Scenario Presets (Environment Parameters)
# =============================================================================

SCENARIO_PRESETS: Dict[int, Dict] = {
    # -------------------------------------------------------------------------
    # Scenario 1: Small-scale (for quick testing and debugging)
    # -------------------------------------------------------------------------
    1: {
        "grid_size": 3,
        "n_sensors": 5,
        "n_targets": 6,
        "cell_size": 20.0,
        "sensing_range": 18.0,
        "fov_angle": 90.0,  # degrees
        "rotation_step": 5.0,  # degrees per action
        "max_steps": 100,
        "target_speed_range": (0.3, 0.7),
    },
    # -------------------------------------------------------------------------
    # Scenario 2: Large-scale (swarm-level coordination)
    # -------------------------------------------------------------------------
    2: {
        "grid_size": 10,
        "n_sensors": 50,
        "n_targets": 60,
        "cell_size": 20.0,
        "sensing_range": 28.0,
        "fov_angle": 90.0,
        "rotation_step": 5.0,
        "max_steps": 100,
        "target_speed_range": (0.3, 0.7),
    },
    # -------------------------------------------------------------------------
    # Scenario 3: Medium-scale (optional custom scenario)
    # -------------------------------------------------------------------------
    3: {
        "grid_size": 5,
        "n_sensors": 15,
        "n_targets": 20,
        "cell_size": 20.0,
        "sensing_range": 28.0,
        "fov_angle": 90.0,
        "rotation_step": 5.0,
        "max_steps": 100,
        "target_speed_range": (0.3, 0.7),
    },
}


def get_scenario_config(scenario: int = 1) -> Dict:
    """Get scenario configuration by ID."""
    return deepcopy(SCENARIO_PRESETS.get(scenario, SCENARIO_PRESETS[1]))


# =============================================================================
# Training Presets (Hyperparameters)
# =============================================================================

TRAINING_PRESETS: Dict[int, Dict] = {
    # -------------------------------------------------------------------------
    # Scenario 1: Small-scale
    # -------------------------------------------------------------------------
    1: {
        "device": None,  # Auto-detects in main.py
        "num_envs": 1,
        "episodes": 25000,
        "batch_size": 256,
        "lr": 3.0e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 5000,
        "target_update": 200,
        "eval_interval": 500,
        "eval_episodes": 20,
        "save_interval": 2000,
        "buffer_capacity": 50000,
        "chunk_length": 10,
        "updates_per_step": 1,
        "learning_starts": 100,
        "no_amp": False,
    },
    # -------------------------------------------------------------------------
    # Scenario 2: Large-scale
    # -------------------------------------------------------------------------
    2: {
        "device": None,
        "num_envs": 1,
        "episodes": 5000,
        "batch_size": 256,
        "lr": 3.0e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 1000,
        "target_update": 200,
        "eval_interval": 500,
        "eval_episodes": 5,
        "save_interval": 250,
        "buffer_capacity": 20000,
        "chunk_length": 25,
        "updates_per_step": 1,
        "learning_starts": 50,
        "no_amp": False,
    },
    # -------------------------------------------------------------------------
    # Scenario 3: Medium-scale
    # -------------------------------------------------------------------------
    3: {
        "device": None,
        "num_envs": 1,
        "episodes": 20000,
        "batch_size": 256,
        "lr": 3.0e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 5000,
        "target_update": 200,
        "eval_interval": 2000,
        "eval_episodes": 10,
        "save_interval": 2000,
        "buffer_capacity": 50000,
        "chunk_length": 25,
        "updates_per_step": 1,
        "learning_starts": 100,
        "no_amp": False,
    },
}


def get_training_config(scenario: int = 1) -> Dict:
    """Get training configuration by ID."""
    return deepcopy(TRAINING_PRESETS.get(scenario, TRAINING_PRESETS[1]))


# =============================================================================
# HiT-MAC Training Presets
# =============================================================================

# Calculate safe worker count: cap at 16 to avoid OOM on high-core machines
_safe_workers = min(16, max(4, (os.cpu_count() or 6) - 2))

HITMAC_TRAINING_PRESETS: Dict[str, Dict] = {
    # -------------------------------------------------------------------------
    # Executor Training (Single-agent control with attention)
    # -------------------------------------------------------------------------
    "executor": {
        "env": "Pose-v0",
        "model": "single-att",
        "lr": 0.0005,
        "gamma": 0.9,
        "tau": 1.0,
        "entropy": 0.01,
        "num_steps": 20,
        "max_step": 5000000,
        "lstm_out": 128,
        "workers": _safe_workers,
        "optimizer": "Adam",
        "test_eps": 1,
    },
    # -------------------------------------------------------------------------
    # Coordinator Training (Multi-agent with Shapley value attribution)
    # -------------------------------------------------------------------------
    "coordinator": {
        "env": "Pose-v1",
        "model": "multi-att-shap",
        "lr": 0.0005,
        "gamma": 0.9,
        "tau": 1.0,
        "entropy": 0.01,
        "num_steps": 20,
        "max_step": 5000000,
        "lstm_out": 128,
        "workers": _safe_workers,
        "optimizer": "Adam",
        "test_eps": 1,
    },
    # -------------------------------------------------------------------------
    # Coordinator Training (without Shapley values, faster but less accurate)
    # -------------------------------------------------------------------------
    "coordinator-fast": {
        "env": "Pose-v1",
        "model": "multi-att",
        "lr": 0.0005,
        "gamma": 0.9,
        "tau": 1.0,
        "entropy": 0.01,
        "num_steps": 20,
        "max_step": 2000000,
        "lstm_out": 128,
        "workers": min(
            _safe_workers, max(4, (os.cpu_count() or 4) - 1)
        ),  # Keep at least 4, leave 1 for test
        "optimizer": "Adam",
        "test_eps": 1,
    },
}


def get_hitmac_config(mode: str = "executor") -> Dict:
    """Get HiT-MAC training configuration by mode.

    Args:
        mode: One of 'executor', 'coordinator', or 'coordinator-fast'

    Returns:
        Configuration dictionary for HiT-MAC training
    """
    return deepcopy(
        HITMAC_TRAINING_PRESETS.get(mode, HITMAC_TRAINING_PRESETS["executor"])
    )


# =============================================================================
# HiT-MAC Scenario-Specific Training Presets
# =============================================================================

HITMAC_SCENARIO_PRESETS: Dict[int, Dict] = {
    # -------------------------------------------------------------------------
    # Scenario 1: Small-scale (5 sensors, 6 targets)
    # -------------------------------------------------------------------------
    1: {
        "model": "single-att",
        "optimizer": "Adam",
        "lr": 0.0005,
        "gamma": 0.9,
        "tau": 1.0,
        "entropy": 0.01,
        "num_steps": 20,
        # 2,500,000 steps per phase × 2 phases = 5,000,000 total env steps
        # Matches NA2Q S1 (50,000 episodes × 100 steps/ep = 5,000,000 env steps)
        "max_step": 2500000,
        "save_interval": 100000,  # save every 4% — catches best before divergence
        "lstm_out": 128,
        "workers": 6,
        "norm_reward": True,
        "train_mode": 1,
        "test_eps": 5,
    },
    # -------------------------------------------------------------------------
    # Scenario 2: Large-scale (50 sensors, 60 targets)
    # -------------------------------------------------------------------------
    2: {
        "model": "single-att",
        "optimizer": "Adam",
        "lr": 0.0003,
        "gamma": 0.9,
        "tau": 1.0,
        "entropy": 0.01,
        "num_steps": 40,
        # Matches NA2Q S2 (5,000 episodes × 100 steps/ep = 500,000 env steps)
        "max_step": 500000,
        "save_interval": 50000,
        "lstm_out": 256,
        "workers": min(
            _safe_workers + 2, max(6, (os.cpu_count() or 8) - 1)
        ),
        "norm_reward": True,
        "train_mode": 1,
        "test_eps": 5,
    },
    # -------------------------------------------------------------------------
    # Scenario 3: Medium-scale (15 sensors, 20 targets)
    # -------------------------------------------------------------------------
    3: {
        "model": "single-att",
        "optimizer": "Adam",
        "lr": 0.0004,
        "gamma": 0.9,
        "tau": 1.0,
        "entropy": 0.01,
        "num_steps": 40,
        "max_step": 4000000,  # ~40,000 episodes
        "save_interval": 400000,
        "lstm_out": 128,
        "workers": _safe_workers,
        "norm_reward": True,
        "train_mode": 1,
        "test_eps": 5,
    },
}


def get_hitmac_training_config(scenario: int = 1) -> Dict:
    """Get HiT-MAC scenario-specific training configuration.

    Args:
        scenario: Scenario ID (1, 2, or 3)

    Returns:
        Configuration dictionary for HiT-MAC training in that scenario
    """
    return deepcopy(HITMAC_SCENARIO_PRESETS.get(scenario, HITMAC_SCENARIO_PRESETS[1]))
