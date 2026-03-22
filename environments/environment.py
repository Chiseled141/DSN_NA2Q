"""
=============================================================================
DIRECTIONAL SENSOR NETWORK (DSN) ENVIRONMENT
=============================================================================

WHAT IS THIS FILE?
------------------
This file creates a "game world" where AI agents learn to control security
cameras (sensors) to track moving targets (like people walking around).

Think of it like a video game:
- The SENSORS are security cameras that can rotate left/right
- The TARGETS are people moving randomly in the area
- The GOAL is to have at least one camera watching each person

HOW IT WORKS:
1. The environment is a grid (like a chessboard)
2. Cameras are placed at fixed positions on the grid
3. Each camera has a "field of view" (FoV) - a pie-shaped area it can see
4. Targets move around randomly
5. The AI learns to rotate cameras to track as many targets as possible

KEY CLASSES:
- DSNEnv: The main environment class (the "game")

USAGE:
    from environments.environment import make_env

    env = make_env(scenario=1)  # Create environment
    obs, info = env.reset()      # Start a new episode
    obs, reward, done, truncated, info = env.step([0, 1, 2, 1, 0])  # Take actions
"""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.patches import Wedge

from config import get_scenario_config

# =============================================================================
# CONSTANTS (These are like the "rules" of the game)
# =============================================================================

# Action definitions - what each camera can do
ACTION_TURN_LEFT = 0  # Rotate camera counter-clockwise
ACTION_STAY = 1  # Don't move
ACTION_TURN_RIGHT = 2  # Rotate camera clockwise


# =============================================================================
# DSN ENVIRONMENT CLASS
# =============================================================================


class DSNEnv(gym.Env):
    """
    Directional Sensor Network Environment for Target Tracking.

    This is the "game" that AI agents learn to play. Each episode:
    1. Targets start at random positions with random velocities
    2. Sensors (cameras) try to track targets by rotating
    3. The episode ends after max_steps (e.g., 100 steps)
    4. The AI gets rewards based on how many targets are tracked

    ATTRIBUTES (Important Variables):
    ---------------------------------
    - n_sensors: Number of cameras (e.g., 5)
    - n_targets: Number of moving targets (e.g., 6)
    - grid_size: Size of the grid (e.g., 3x3)
    - sensing_range: How far each camera can see
    - fov_angle: The camera's viewing angle (e.g., 90 degrees = quarter circle)
    - rotation_step: How much a camera rotates per action (e.g., 5 degrees)

    OBSERVATIONS:
    -------------
    Each camera sees information about all targets:
    - Distance to each target (normalized 0-1)
    - Angle to each target (normalized -1 to 1)
    - Camera ID and target ID (for identification)

    REWARDS:
    --------
    - Base reward = coverage_rate (0 to 1)
    - Bonus for high coverage (50%, 80%, 100%)
    - Extra bonus for pointing at targets
    """

    # Gymnasium metadata
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Action constants (accessible from outside)
    ACTION_TURN_LEFT = 0
    ACTION_STAY = 1
    ACTION_TURN_RIGHT = 2

    # =========================================================================
    # INITIALIZATION (Setup the game)
    # =========================================================================

    def __init__(
        self,
        scenario: int = 1,  # Which scenario to use (1 = small, 2 = large)
        n_sensors: Optional[int] = None,  # Override number of sensors
        n_targets: Optional[int] = None,  # Override number of targets
        grid_size: Optional[int] = None,  # Override grid size
        cell_size: Optional[float] = None,  # Size of each grid cell (in units)
        sensing_range: Optional[float] = None,  # How far cameras can see
        fov_angle: Optional[float] = None,  # Camera field of view in degrees
        rotation_step: Optional[float] = None,  # Degrees per rotation action
        max_steps: Optional[int] = None,  # Episode length
        target_speed_range: Optional[Tuple[float, float]] = None,  # Min/max target speed
        render_mode: Optional[str] = None,  # "human" or "rgb_array"
        seed: Optional[int] = None,  # Random seed for reproducibility
    ):
        """
        Initialize the environment.

        Parameters:
        -----------
        scenario : int
            Which scenario preset to use:
            - 1: Small (3x3 grid, 5 sensors, 6 targets) - good for testing
            - 2: Large (10x10 grid, 50 sensors, 60 targets) - realistic

        All other parameters are optional and will override scenario defaults.
        """
        super().__init__()

        # Load default settings from config file
        config = get_scenario_config(scenario)

        # Store configuration (use provided values or defaults from config)
        self.scenario = scenario
        self.grid_size = grid_size or config["grid_size"]
        self.n_sensors = n_sensors or config["n_sensors"]
        self.n_targets = n_targets or config["n_targets"]
        self.cell_size = cell_size or config["cell_size"]
        self.sensing_range = sensing_range or config["sensing_range"]

        # Convert angles from degrees to radians (computers prefer radians)
        # Radians = Degrees × (π / 180)
        self.fov_angle = np.radians(fov_angle or config["fov_angle"])
        self.rotation_step = np.radians(rotation_step or config["rotation_step"])

        self.max_steps = max_steps or config["max_steps"]
        self.target_speed_range = target_speed_range or config["target_speed_range"]
        self.render_mode = render_mode

        # Curriculum learning settings (start easy, get harder)
        self.difficulty_level = 1.0  # 0.0 = easy, 1.0 = hard
        self.use_realistic_obs = False  # If True, cameras can only see targets in their FoV

        # Calculate total field size
        self.field_size = self.grid_size * self.cell_size

        # Random number generator (for reproducibility)
        self.np_random = np.random.default_rng(seed)

        # =================================
        # Define Action and Observation Spaces
        # =================================

        # Each camera has 3 possible actions
        self.n_actions = 3  # LEFT, STAY, RIGHT
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation: for each target, we observe 4 values
        # [sensor_id, target_id, distance, angle]
        self.obs_per_target = 4
        self.obs_dim = self.n_targets * self.obs_per_target
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Global state dimension (used for training the mixer)
        # = sensor positions (x, y, angle) + target positions (x, y)
        self.state_dim = self.n_sensors * 3 + self.n_targets * 2

        # =================================
        # Initialize State Variables
        # =================================

        self.sensor_positions = None  # Where each camera is located [n_sensors, 2]
        self.sensor_angles = None  # Which direction each camera faces [n_sensors]
        self.target_positions = None  # Where each target is [n_targets, 2]
        self.target_velocities = None  # How each target is moving [n_targets, 2]
        self.goal_map = None  # Which sensor tracks which target [n_sensors, n_targets]
        self.current_step = 0  # Current timestep in the episode

        # Rendering (for visualization)
        self.fig = None
        self.ax = None

    # =========================================================================
    # SENSOR PLACEMENT (Where to put the cameras)
    # =========================================================================

    def _initialize_sensor_positions(self) -> np.ndarray:
        """
        Place sensors on the grid based on the scenario.

        Returns:
            positions: Array of shape [n_sensors, 2] with (x, y) coordinates
        """
        if self.scenario == 1:
            return self._get_scenario1_positions()
        else:
            return self._get_scenario2_positions()

    def _get_scenario1_positions(self) -> np.ndarray:
        """
        Scenario 1: Fixed sensor positions at cells 1, 3, 5, 7, 9.

        The 3x3 grid is numbered like this:

            7 | 8 | 9
           -----------
            4 | 5 | 6
           -----------
            1 | 2 | 3

        We place sensors at the corners (1,3,7,9) and center (5).
        """
        # Map cell numbers to (column, row) positions
        cell_centers = {
            1: (0, 0),  # Bottom-left
            3: (2, 0),  # Bottom-right
            5: (1, 1),  # Center
            7: (0, 2),  # Top-left
            9: (2, 2),  # Top-right
        }

        positions = []
        for cell_num in [1, 3, 5, 7, 9]:
            col, row = cell_centers[cell_num]
            # Calculate center of the cell
            x = (col + 0.5) * self.cell_size
            y = (row + 0.5) * self.cell_size
            positions.append([x, y])

        return np.array(positions[: self.n_sensors])

    def _get_scenario2_positions(self) -> np.ndarray:
        """
        Scenario 2: Random sensor placement with 50% probability per cell.

        This simulates a more realistic scenario where sensors are
        randomly distributed across the area.
        """
        positions = []

        # Try to place a sensor in each cell with 50% probability
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.np_random.random() < 0.5:  # 50% chance
                    x = (col + 0.5) * self.cell_size
                    y = (row + 0.5) * self.cell_size
                    positions.append([x, y])

        positions = np.array(positions) if positions else np.zeros((0, 2))

        # Ensure we have exactly n_sensors
        if len(positions) > self.n_sensors:
            # Too many: randomly select n_sensors
            indices = self.np_random.choice(len(positions), self.n_sensors, replace=False)
            positions = positions[indices]
        elif len(positions) < self.n_sensors:
            # Too few: add random positions
            while len(positions) < self.n_sensors:
                col = self.np_random.integers(0, self.grid_size)
                row = self.np_random.integers(0, self.grid_size)
                x = (col + 0.5) * self.cell_size
                y = (row + 0.5) * self.cell_size
                new_pos = np.array([[x, y]])
                positions = np.vstack([positions, new_pos]) if len(positions) > 0 else new_pos

        return positions[: self.n_sensors]

    # =========================================================================
    # TARGET MANAGEMENT (The moving objects to track)
    # =========================================================================

    def set_curriculum_difficulty(self, level: float):
        """
        Set the difficulty level for curriculum learning.

        Curriculum learning means we start easy and gradually increase difficulty:
        - Level 0-49%: Global observations (cameras can see ALL targets)
        - Level 50-100%: Realistic observations (cameras only see targets in their FoV)

        Parameters:
        -----------
        level : float
            Difficulty level from 0.0 (easiest) to 1.0 (hardest)
        """
        self.difficulty_level = np.clip(level, 0.0, 1.0)
        self.use_realistic_obs = level >= 0.5

    def _initialize_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize targets at random positions with random velocities.

        Returns:
            positions: [n_targets, 2] - (x, y) positions
            velocities: [n_targets, 2] - (vx, vy) velocities
        """
        # Random positions within the field (with small margin from edges)
        margin = self.cell_size * 0.1
        positions = self.np_random.uniform(margin, self.field_size - margin, (self.n_targets, 2))

        # Random speeds and directions
        speeds = self.np_random.uniform(
            self.target_speed_range[0], self.target_speed_range[1], self.n_targets
        )
        angles = self.np_random.uniform(0, 2 * np.pi, self.n_targets)

        # Convert speed + angle to (vx, vy) velocity
        # vx = speed × cos(angle)
        # vy = speed × sin(angle)
        velocities = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1)

        return positions, velocities

    def _update_targets(self):
        """
        Update target positions for one timestep (vectorized).

        This includes:
        1. Adding velocity to position (movement)
        2. Bouncing off walls (reflection)
        3. Random direction changes (10% chance per target)
        """
        # Add some speed variation (±20%) for realism
        speed_variation = 1 + 0.2 * self.np_random.random(self.n_targets)
        varied_velocities = self.target_velocities * speed_variation[:, np.newaxis]

        # Update positions
        self.target_positions += varied_velocities

        # Bounce off walls — vectorized
        mask_low = self.target_positions < 0
        mask_high = self.target_positions > self.field_size
        # Reflect positions
        self.target_positions = np.where(mask_low, -self.target_positions, self.target_positions)
        self.target_positions = np.where(
            mask_high, 2 * self.field_size - self.target_positions, self.target_positions
        )
        # Reverse velocities at boundaries
        bounce_mask = mask_low | mask_high
        self.target_velocities = np.where(
            bounce_mask, -self.target_velocities, self.target_velocities
        )

        # Random direction changes (10% chance per target) — vectorized
        change_mask = self.np_random.random(self.n_targets) < 0.1
        n_changes = np.sum(change_mask)
        if n_changes > 0:
            new_angles = self.np_random.uniform(0, 2 * np.pi, n_changes)
            speeds = np.linalg.norm(self.target_velocities[change_mask], axis=1)
            self.target_velocities[change_mask, 0] = speeds * np.cos(new_angles)
            self.target_velocities[change_mask, 1] = speeds * np.sin(new_angles)

    # =========================================================================
    # CORE ENVIRONMENT METHODS (The main game loop)
    # =========================================================================

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Reset the environment to start a new episode.

        This is called at the beginning of each training episode.

        Returns:
            observations: List of observations, one for each sensor
            info: Dictionary with extra information
        """
        # Set random seed if provided
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Initialize sensor positions
        self.sensor_positions = self._initialize_sensor_positions()

        # Initialize targets
        self.target_positions, self.target_velocities = self._initialize_targets()

        # Initialize sensor angles (point each sensor at nearest target) — vectorized
        diffs = (
            self.target_positions[np.newaxis, :, :] - self.sensor_positions[:, np.newaxis, :]
        )  # [n_s, n_t, 2]
        distances = np.linalg.norm(diffs, axis=2)  # [n_s, n_t]
        nearest_idx = np.argmin(distances, axis=1)  # [n_s]
        nearest_diffs = diffs[np.arange(self.n_sensors), nearest_idx]  # [n_s, 2]
        self.sensor_angles = np.arctan2(nearest_diffs[:, 1], nearest_diffs[:, 0])  # [n_s]

        # Initialize tracking map (all zeros = no tracking yet)
        self.goal_map = np.zeros((self.n_sensors, self.n_targets), dtype=np.int32)

        # Reset step counter
        self.current_step = 0

        return self._get_observations(), self._get_info()

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], float, bool, bool, dict]:
        """
        Take one step in the environment.

        This is the main game loop:
        1. Apply actions (rotate sensors)
        2. Move targets
        3. Update tracking status
        4. Calculate reward

        Parameters:
        -----------
        actions : List[int]
            One action per sensor: 0=LEFT, 1=STAY, 2=RIGHT

        Returns:
        --------
        observations : List[np.ndarray]
            New observations for each sensor
        reward : float
            Team reward for this step
        terminated : bool
            True if episode ended naturally (never in this env)
        truncated : bool
            True if episode hit max_steps
        info : dict
            Extra information (coverage rate, etc.)
        """
        # Validate actions — flatten to 1D to handle both list and tensor inputs
        actions_arr = np.asarray(actions, dtype=np.int64).ravel()
        if len(actions_arr) != self.n_sensors:
            raise ValueError(f"Expected {self.n_sensors} actions, got {len(actions_arr)}")

        # Step 1: Apply rotations — vectorized
        rotation = np.where(
            actions_arr == self.ACTION_TURN_LEFT,
            -self.rotation_step,
            np.where(actions_arr == self.ACTION_TURN_RIGHT, self.rotation_step, 0.0),
        )
        self.sensor_angles = (self.sensor_angles + rotation) % (2 * np.pi)

        # Step 2: Move targets
        self._update_targets()

        # Step 3: Update tracking status
        self._update_goal_map()

        # Step 4: Increment step counter
        self.current_step += 1

        # Step 5: Calculate reward
        reward, info = self._calculate_reward()

        # Check if episode is done
        terminated = False  # This environment never terminates early
        truncated = self.current_step >= self.max_steps

        return self._get_observations(), reward, terminated, truncated, info

    # =========================================================================
    # TRACKING LOGIC (Which sensors see which targets)
    # =========================================================================

    def _is_target_tracked(self, sensor_idx: int, target_idx: int) -> bool:
        """
        Check if a target is within a sensor's field of view.

        A target is tracked if:
        1. It's within sensing_range distance
        2. It's within the FoV angle cone

        Parameters:
        -----------
        sensor_idx : int
            Index of the sensor (0 to n_sensors-1)
        target_idx : int
            Index of the target (0 to n_targets-1)

        Returns:
        --------
        bool : True if target is tracked by this sensor
        """
        # Vector from sensor to target
        diff = self.target_positions[target_idx] - self.sensor_positions[sensor_idx]

        # Distance to target
        distance = np.linalg.norm(diff)

        # Check distance constraint
        if distance > self.sensing_range:
            return False

        # Angle from sensor to target
        angle_to_target = np.arctan2(diff[1], diff[0])

        # Angle difference from sensor's facing direction
        angle_diff = self._normalize_angle(angle_to_target - self.sensor_angles[sensor_idx])

        # Check if within FoV cone (half the FoV on each side)
        return abs(angle_diff) <= self.fov_angle / 2

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to range [-π, π].

        This is needed because angles wrap around:
        - 350° and -10° are the same angle
        - This function converts any angle to the range -180° to 180° (in radians)

        Parameters:
        -----------
        angle : float
            Any angle in radians

        Returns:
        --------
        float : Angle in range [-π, π]
        """
        # Use modulo to wrap angle, then shift to [-π, π]
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _update_goal_map(self):
        """
        Update the goal map showing which sensors track which targets (vectorized).

        Uses broadcasting to compute all sensor-target distances and angles at once.
        """
        # diffs: [n_sensors, n_targets, 2]
        diffs = self.target_positions[np.newaxis, :, :] - self.sensor_positions[:, np.newaxis, :]
        # distances: [n_sensors, n_targets]
        distances = np.linalg.norm(diffs, axis=2)
        # angles: [n_sensors, n_targets]
        angles_to_targets = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])
        angle_diffs = (angles_to_targets - self.sensor_angles[:, np.newaxis] + np.pi) % (
            2 * np.pi
        ) - np.pi

        in_range = distances <= self.sensing_range
        in_fov = np.abs(angle_diffs) <= self.fov_angle / 2
        self.goal_map = (in_range & in_fov).astype(np.int32)

    # =========================================================================
    # REWARD CALCULATION (How well are we doing?)
    # =========================================================================

    def _calculate_reward(self) -> Tuple[float, dict]:
        """
        Calculate the team reward based on coverage (vectorized).

        Returns:
            reward: Float reward value
            info: Dictionary with tracking statistics
        """
        # Check which targets are tracked by at least one sensor
        targets_tracked = np.any(self.goal_map, axis=0)  # [n_targets]
        n_tracked = np.sum(targets_tracked)
        coverage_rate = n_tracked / self.n_targets

        # Base reward = coverage rate
        reward = float(coverage_rate)

        # Bonuses for high coverage
        if n_tracked == self.n_targets:
            reward += 1.0
        elif coverage_rate >= 0.8:
            reward += 0.6
        elif coverage_rate >= 0.5:
            reward += 0.2

        # Centering bonus — vectorized
        # diffs: [n_sensors, n_targets, 2]
        diffs = self.target_positions[np.newaxis, :, :] - self.sensor_positions[:, np.newaxis, :]
        # distances: [n_sensors, n_targets]
        dist_all = np.linalg.norm(diffs, axis=2)
        # nearest target index per sensor: [n_sensors]
        nearest_idx = np.argmin(dist_all, axis=1)
        # Get the diff to the nearest target for each sensor
        nearest_diffs = diffs[np.arange(self.n_sensors), nearest_idx]  # [n_sensors, 2]
        angles_to_nearest = np.arctan2(nearest_diffs[:, 1], nearest_diffs[:, 0])
        angle_diffs = (angles_to_nearest - self.sensor_angles + np.pi) % (2 * np.pi) - np.pi
        centering_bonus = float(np.mean(np.cos(angle_diffs)))

        reward += centering_bonus

        info = {
            "n_tracked": int(n_tracked),
            "coverage_rate": float(coverage_rate),
            "goal_map": self.goal_map.copy(),
        }

        return reward, info

    # =========================================================================
    # OBSERVATIONS (What each sensor sees)
    # =========================================================================

    def _get_observations(self) -> List[np.ndarray]:
        """
        Get observations for all sensors (vectorized).

        Computes all sensor-target distances and angles using broadcasting,
        then sorts and formats per-sensor observations.

        Returns:
            List of observation arrays, one per sensor
        """
        n_s = self.n_sensors
        n_t = self.n_targets
        diag = self.field_size * np.sqrt(2)

        # Precompute all diffs, distances, angles: [n_sensors, n_targets]
        diffs = (
            self.target_positions[np.newaxis, :, :] - self.sensor_positions[:, np.newaxis, :]
        )  # [n_s, n_t, 2]
        distances = np.linalg.norm(diffs, axis=2)  # [n_s, n_t]
        angles_to_targets = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])  # [n_s, n_t]
        angle_diffs = (angles_to_targets - self.sensor_angles[:, np.newaxis] + np.pi) % (
            2 * np.pi
        ) - np.pi  # [n_s, n_t]

        # Precompute normalized IDs
        sensor_ids_norm = np.arange(n_s) / max(n_s - 1, 1)  # [n_s]
        target_ids_norm = np.arange(n_t) / max(n_t - 1, 1)  # [n_t]

        # Normalized distances and angles
        dist_norm = distances / diag  # [n_s, n_t]
        angle_norm = angle_diffs / np.pi  # [n_s, n_t]

        if self.use_realistic_obs:
            # Visibility mask
            in_range = distances <= self.sensing_range
            in_fov = np.abs(angle_diffs) <= self.fov_angle / 2
            visible = in_range & in_fov  # [n_s, n_t]
            # Zero out hidden targets
            dist_norm = np.where(visible, dist_norm, 0.0)
            angle_norm = np.where(visible, angle_norm, 0.0)
            # Sort keys: visible targets by distance, hidden get inf
            sort_keys = np.where(visible, distances, np.inf)  # [n_s, n_t]
        else:
            sort_keys = distances  # [n_s, n_t]

        # Sort indices per sensor
        sort_order = np.argsort(sort_keys, axis=1)  # [n_s, n_t]

        observations = []
        for i in range(n_s):
            order = sort_order[i]  # [n_t]
            # Build observation: [sensor_id, target_id, dist, angle] per target
            obs = np.empty((n_t, 4), dtype=np.float32)
            obs[:, 0] = sensor_ids_norm[i]
            obs[:, 1] = target_ids_norm[order]
            obs[:, 2] = dist_norm[i, order]
            obs[:, 3] = angle_norm[i, order]
            observations.append(obs.ravel())

        return observations

    def _get_agent_observation(self, sensor_idx: int) -> np.ndarray:
        """
        Get observation for a single sensor (used by external callers).
        Falls back to the vectorized batch method.
        """
        return self._get_observations()[sensor_idx]

    def get_state(self) -> np.ndarray:
        """
        Get global state for centralized training.

        The state includes ALL information about the environment:
        - Each sensor: x, y, angle (normalized)
        - Each target: x, y (normalized)

        This is used by the "mixer" network during training.
        Only the trainer sees this; individual sensors don't.

        Returns:
            state: Array of shape [state_dim]
        """
        state = []

        # Sensor information
        for i in range(self.n_sensors):
            pos = self.sensor_positions[i] / self.field_size  # Normalize to [0, 1]
            angle = self.sensor_angles[i] / (2 * np.pi)  # Normalize to [0, 1]
            state.extend([pos[0], pos[1], angle])

        # Target information
        for j in range(self.n_targets):
            pos = self.target_positions[j] / self.field_size
            state.extend([pos[0], pos[1]])

        return np.array(state, dtype=np.float32)

    def _get_info(self) -> dict:
        """Get environment info dictionary."""
        return {
            "step": self.current_step,
            "sensor_positions": self.sensor_positions.copy(),
            "sensor_angles": self.sensor_angles.copy(),
            "target_positions": self.target_positions.copy(),
            "target_velocities": self.target_velocities.copy(),
            "goal_map": self.goal_map.copy() if self.goal_map is not None else None,
        }

    def get_avail_actions(self) -> List[np.ndarray]:
        """
        Get available actions for each sensor.

        In this environment, all actions are always available.

        Returns:
            List of availability masks, one per sensor
        """
        # Pre-build once and return views (avoids creating new arrays each call)
        avail = np.ones((self.n_sensors, self.n_actions), dtype=np.float32)
        return [avail[i] for i in range(self.n_sensors)]

    # =========================================================================
    # RENDERING (Visualization)
    # =========================================================================

    def render(self):
        """
        Render the current state of the environment.

        This creates a visualization showing:
        - The grid layout
        - Sensor positions and their FoV cones
        - Target positions (green = tracked, red = untracked)
        - Coverage statistics
        """
        if self.render_mode is None:
            return

        # Initialize figure if needed
        if self.fig is None:
            plt.style.use("default")
            self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 16), facecolor="#ffffff")
            self.ax.set_facecolor("#f8f9fa")

        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("#f8f9fa")

        self.ax.clear()
        self.ax.set_xlim(-15, self.field_size + 15)
        self.ax.set_ylim(-15, self.field_size + 15)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("#f8f9fa")

        # Calculate coverage
        targets_tracked = (
            np.any(self.goal_map, axis=0) if self.goal_map is not None else np.zeros(self.n_targets)
        )
        n_tracked = int(np.sum(targets_tracked))
        coverage_pct = 100 * n_tracked / self.n_targets

        # Title
        self.ax.set_title(
            f"NA²Q Sensor Network Simulation\n"
            f"Step {self.current_step:03d} | Coverage: {coverage_pct:.0f}%",
            fontsize=16,
            fontweight="bold",
            color="#1a1a1a",
            pad=15,
        )

        # Draw grid
        for i in range(self.grid_size + 1):
            alpha = 0.8 if i % 2 == 0 else 0.5
            linewidth = 1.5 if i % 2 == 0 else 1.0
            self.ax.axhline(y=i * self.cell_size, color="#94a3b8", linewidth=linewidth, alpha=alpha)
            self.ax.axvline(x=i * self.cell_size, color="#94a3b8", linewidth=linewidth, alpha=alpha)

        # Field boundary
        self.ax.add_patch(
            plt.Rectangle(
                (0, 0),
                self.field_size,
                self.field_size,
                fill=False,
                edgecolor="#3b82f6",
                linewidth=2.5,
                alpha=0.9,
            )
        )
        self.ax.add_patch(
            plt.Rectangle(
                (-0.3, -0.3),
                self.field_size + 0.6,
                self.field_size + 0.6,
                fill=False,
                edgecolor="#93c5fd",
                linewidth=1,
                alpha=0.5,
            )
        )

        # Color palette for sensors
        sensor_colors = [
            "#ef4444",
            "#3b82f6",
            "#10b981",
            "#f59e0b",
            "#8b5cf6",
            "#ec4899",
            "#06b6d4",
            "#84cc16",
            "#f97316",
            "#6366f1",
        ]

        # Draw sensors
        for i in range(self.n_sensors):
            pos = self.sensor_positions[i]
            angle = np.degrees(self.sensor_angles[i])
            color = sensor_colors[i % len(sensor_colors)]

            # FoV wedge
            wedge_outer = Wedge(
                pos,
                self.sensing_range,
                angle - np.degrees(self.fov_angle) / 2,
                angle + np.degrees(self.fov_angle) / 2,
                alpha=0.2,
                color=color,
            )
            wedge_inner = Wedge(
                pos,
                self.sensing_range * 0.7,
                angle - np.degrees(self.fov_angle) / 2,
                angle + np.degrees(self.fov_angle) / 2,
                alpha=0.15,
                color=color,
            )
            self.ax.add_patch(wedge_outer)
            self.ax.add_patch(wedge_inner)

            # FoV edge lines
            fov_half = np.degrees(self.fov_angle) / 2
            for edge_angle in [angle - fov_half, angle + fov_half]:
                rad = np.radians(edge_angle)
                end_x = pos[0] + self.sensing_range * np.cos(rad)
                end_y = pos[1] + self.sensing_range * np.sin(rad)
                self.ax.plot([pos[0], end_x], [pos[1], end_y], color=color, alpha=0.5, linewidth=1)

            # Sensor marker
            self.ax.plot(
                pos[0],
                pos[1],
                "o",
                color=color,
                markersize=14,
                markeredgecolor="white",
                markeredgewidth=2.5,
                zorder=10,
            )
            self.ax.plot(
                pos[0] + 0.3, pos[1] - 0.3, "o", color="#00000020", markersize=14, zorder=9
            )

        # Draw targets
        for j in range(self.n_targets):
            pos = self.target_positions[j]
            tracked = targets_tracked[j] if len(targets_tracked) > j else False

            if tracked:
                # Tracked: green
                self.ax.plot(
                    pos[0],
                    pos[1],
                    "o",
                    color="#22c55e",
                    markersize=16,
                    markeredgecolor="#166534",
                    markeredgewidth=2,
                    zorder=11,
                )
                self.ax.plot(
                    pos[0], pos[1], "o", color="#22c55e", markersize=24, alpha=0.2, zorder=8
                )
            else:
                # Untracked: red
                self.ax.plot(
                    pos[0],
                    pos[1],
                    "o",
                    color="#ef4444",
                    markersize=16,
                    markeredgecolor="#991b1b",
                    markeredgewidth=2,
                    zorder=11,
                )
                self.ax.plot(
                    pos[0], pos[1], "o", color="#ef4444", markersize=24, alpha=0.15, zorder=8
                )

        # Coverage meter
        meter_y = -2
        meter_width = self.field_size * 0.8
        meter_x = (self.field_size - meter_width) / 2

        self.ax.add_patch(
            plt.Rectangle(
                (meter_x, meter_y - 0.3), meter_width, 0.6, color="#e2e8f0", alpha=0.9, zorder=12
            )
        )
        progress_width = meter_width * (coverage_pct / 100)
        bar_color = (
            "#22c55e" if coverage_pct >= 70 else "#f59e0b" if coverage_pct >= 40 else "#ef4444"
        )
        self.ax.add_patch(
            plt.Rectangle(
                (meter_x, meter_y - 0.25),
                progress_width,
                0.5,
                color=bar_color,
                alpha=0.9,
                zorder=13,
            )
        )

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#22c55e",
                markersize=12,
                markeredgecolor="#166534",
                markeredgewidth=1.5,
                label=f"Tracked ({n_tracked})",
                linestyle="None",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#ef4444",
                markersize=12,
                markeredgecolor="#991b1b",
                markeredgewidth=1.5,
                label=f"Untracked ({self.n_targets - n_tracked})",
                linestyle="None",
            ),
        ]
        legend = self.ax.legend(
            handles=legend_elements,
            loc="upper right",
            facecolor="#ffffff",
            edgecolor="#e2e8f0",
            labelcolor="#1a1a1a",
            fontsize=12,
            framealpha=0.95,
        )
        legend.get_frame().set_linewidth(1.5)

        # Axis labels
        grid_ticks = [i * self.cell_size for i in range(self.grid_size + 1)]
        self.ax.set_xticks(grid_ticks)
        self.ax.set_yticks(grid_ticks)
        self.ax.set_xticklabels(
            [f"{int(t)}" for t in grid_ticks], fontsize=18, fontweight="bold", color="#1a1a1a"
        )
        self.ax.set_yticklabels(
            [f"{int(t)}" for t in grid_ticks], fontsize=18, fontweight="bold", color="#1a1a1a"
        )
        self.ax.set_xlabel(
            "X Coordinate", fontsize=20, fontweight="bold", color="#1a1a1a", labelpad=12
        )
        self.ax.set_ylabel(
            "Y Coordinate", fontsize=20, fontweight="bold", color="#1a1a1a", labelpad=12
        )
        self.ax.tick_params(axis="both", which="major", length=8, width=2, colors="#1a1a1a")

        for spine in self.ax.spines.values():
            spine.set_color("#3b82f6")
            spine.set_linewidth(2)

        plt.tight_layout()

        if self.render_mode == "human":
            plt.pause(0.1)
            plt.draw()
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            return buf[:, :, [1, 2, 3]]

    def close(self):
        """Clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# =============================================================================
# FACTORY FUNCTION (Easy way to create the environment)
# =============================================================================


def make_env(scenario: int = 1, **kwargs) -> DSNEnv:
    """
    Create a DSN environment.

    This is the recommended way to create the environment.

    Parameters:
    -----------
    scenario : int
        Which scenario to use:
        - 1: Small (3x3 grid, 5 sensors, 6 targets) - good for testing
        - 2: Large (10x10 grid, 50 sensors, 60 targets) - realistic
    **kwargs :
        Any additional parameters to pass to DSNEnv

    Returns:
    --------
    env : DSNEnv
        The created environment

    Example:
    --------
    >>> env = make_env(scenario=1)
    >>> obs, info = env.reset()
    >>> print(f"Number of sensors: {env.n_sensors}")
    >>> print(f"Number of targets: {env.n_targets}")
    """
    return DSNEnv(scenario=scenario, **kwargs)
