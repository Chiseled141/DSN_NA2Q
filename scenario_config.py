"""
Scenario configuration for DSN environments.

Defines environment parameters (grid size, sensors, targets, physics) 
separately from training hyperparameters.
"""

from typing import Dict, Tuple
from copy import deepcopy


# =============================================================================
# Scenario Presets
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
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_scenario_config(scenario: int = 1) -> Dict:
    """Get scenario configuration by ID.
    
    Args:
        scenario: Scenario ID (1, 2, or 3)
    
    Returns:
        Dict with environment parameters
    """
    preset = SCENARIO_PRESETS.get(scenario, SCENARIO_PRESETS[1])
    return deepcopy(preset)


def format_scenario_config(scenario: int = 1) -> str:
    """Return human-readable scenario config string."""
    cfg = get_scenario_config(scenario)
    lines = [f"Scenario {scenario} Configuration:"]
    lines.append(f"  Grid: {cfg['grid_size']}×{cfg['grid_size']} ({cfg['grid_size'] * cfg['cell_size']:.0f}×{cfg['grid_size'] * cfg['cell_size']:.0f} units)")
    lines.append(f"  Sensors: {cfg['n_sensors']}")
    lines.append(f"  Targets: {cfg['n_targets']}")
    lines.append(f"  Cell size: {cfg['cell_size']}")
    lines.append(f"  Sensing range: {cfg['sensing_range']}")
    lines.append(f"  FoV angle: {cfg['fov_angle']}°")
    lines.append(f"  Rotation step: {cfg['rotation_step']}°")
    lines.append(f"  Max steps: {cfg['max_steps']}")
    lines.append(f"  Target speed: {cfg['target_speed_range']}")
    return "\n".join(lines)


def list_scenarios() -> str:
    """List all available scenarios."""
    lines = ["Available Scenarios:"]
    for sid, cfg in SCENARIO_PRESETS.items():
        lines.append(f"  {sid}: {cfg['grid_size']}×{cfg['grid_size']} grid, {cfg['n_sensors']} sensors, {cfg['n_targets']} targets")
    return "\n".join(lines)
