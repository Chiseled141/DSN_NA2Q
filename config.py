"""
Configuration for NA²Q.

Contains both scenario presets (environment parameters) and training presets (hyperparameters).
"""

from typing import Dict
from copy import deepcopy

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
        "fov_angle": 90.0,           # degrees
        "rotation_step": 5.0,        # degrees per action
        "max_steps": 100,
        "target_speed_range": (0.3, 0.7),
        "dynamic_population": False,
        "spawn_rate": 0.0,
        "despawn_at_edge": False,
        "min_dwell_time": 100,
        "max_dwell_time": 100,
    },
    
    # -------------------------------------------------------------------------
    # Scenario 2: Large-scale (swarm-level coordination)
    # -------------------------------------------------------------------------
    2: {
        "grid_size": 10,
        "n_sensors": 50,
        "n_targets": 60,
        "cell_size": 20.0,
        "sensing_range": 18.0,
        "fov_angle": 90.0,
        "rotation_step": 5.0,
        "max_steps": 100,
        "target_speed_range": (0.3, 0.7),
        "dynamic_population": True,
        "spawn_rate": 0.05,
        "despawn_at_edge": True,
        "min_dwell_time": 30,
        "max_dwell_time": 80,
    },
    
    # -------------------------------------------------------------------------
    # Scenario 3: Medium-scale (optional custom scenario)
    # -------------------------------------------------------------------------
    3: {
        "grid_size": 5,
        "n_sensors": 15,
        "n_targets": 20,
        "cell_size": 20.0,
        "sensing_range": 18.0,
        "fov_angle": 90.0,
        "rotation_step": 5.0,
        "max_steps": 100,
        "target_speed_range": (0.3, 0.7),
        "dynamic_population": True,
        "spawn_rate": 0.10,
        "despawn_at_edge": True,
        "min_dwell_time": 20,
        "max_dwell_time": 60,
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
        "device": "cuda",
        "num_envs": 1,
        "episodes": 15000,
        "batch_size": 256,
        "lr": 3.0e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 7500,
        "target_update": 200,
        "eval_interval": 2000,
        "eval_episodes": 20,
        "save_interval": 5000,
        "buffer_capacity": 100000,
        "chunk_length": 25,
        "updates_per_step": 1,
        "learning_starts": 100,
        "no_amp": False,
    },
    
    # -------------------------------------------------------------------------
    # Scenario 2: Large-scale
    # -------------------------------------------------------------------------
    2: {
        "device": "cuda",
        "num_envs": 1,
        "episodes": 10000,
        "batch_size": 256,
        "lr": 3.0e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 1000,
        "target_update": 200,
        "eval_interval": 2000,
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
        "device": "cuda",
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
        "max_step": 1500000,
        "lstm_out": 128,
        "workers": 6,
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
        "max_step": 1500000,
        "lstm_out": 128,
        "workers": 6,
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
        "workers": 4,
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
    return deepcopy(HITMAC_TRAINING_PRESETS.get(mode, HITMAC_TRAINING_PRESETS["executor"]))

