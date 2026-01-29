"""
Visualization utilities for NA²Q training results.

Includes:
- Training metrics plots (rewards, coverage, loss)
- Video generation of trained agents
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


# Use non-interactive backend for saving figures
plt.switch_backend('Agg')

from na2q.utils import get_device


def smooth_curve(values: List[float], window: int = 10) -> np.ndarray:
    """Smooth a curve using moving average with expanding window at start."""
    if len(values) < window:
        return np.array(values)
    
    values_arr = np.array(values)
    smoothed = np.zeros(len(values))
    
    # Use expanding window for the beginning (avoids spike)
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        smoothed[i] = np.mean(values_arr[start_idx:i+1])
    
    return smoothed


def plot_training_results(exp_dir: str, window: int = 50, history_dir: Optional[str] = None, 
                          media_dir: Optional[str] = None, scenario: int = 1, algorithm_name: str = "NA²Q"):
    """
    Generate professional dark-themed training dashboard.
    
    Creates:
    - {algo}_train_dashboard.png: 4-panel dark theme dashboard (Reward, Coverage, Loss, Epsilon)
    - {algo}_train_coverage.png: Coverage over time
    - {algo}_train_losses.png: Loss curves
    """
    history_dir = history_dir or os.path.join(exp_dir, "checkpoints")
    media_dir = media_dir or exp_dir # Save directly in result folder
    os.makedirs(media_dir, exist_ok=True)
    
    # Clean algorithm name for filenames (lowercase, no spaces/special chars)
    algo_prefix = algorithm_name.lower().replace("²", "2").replace("-", "").replace(" ", "")
    
    # Find history file
    history_path = os.path.join(history_dir, "training_history.npz")
    if not os.path.exists(history_path):
        history_path = os.path.join(exp_dir, "training_history.npz")
    
    if not os.path.exists(history_path):
        print(f"Warning: No training history found at {history_path}")
        return
    
    data = np.load(history_path)
    
    rewards = data["episode_rewards"] if "episode_rewards" in data else []
    coverages = data["coverage_rates"] if "coverage_rates" in data else []
    losses = data["losses"] if "losses" in data else []
    
    if len(rewards) == 0:
        print("Warning: Empty training history")
        return
    
    episodes = np.arange(1, len(rewards) + 1)
    coverage_pct = np.array(coverages) * 100
    
    # =========================================================================
    # Light Theme Dashboard
    # =========================================================================
    
    # Light color palette
    LIGHT_BG = '#ffffff'
    PANEL_BG = '#f8f9fa'
    GRID_COLOR = '#e0e0e0'
    TEXT_COLOR = '#333333'
    TITLE_COLOR = '#1a1a1a'
    
    # Chart colors
    REWARD_COLOR = '#2563eb'      # Blue
    COVERAGE_COLOR = '#16a34a'    # Green
    LOSS_COLOR = '#dc2626'        # Red
    
    # Create figure with light background
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=LIGHT_BG)
    fig.patch.set_facecolor(LIGHT_BG)
    
    # Main title
    fig.suptitle(f'{algorithm_name} Training Dashboard - Scenario {scenario}', 
                 fontsize=18, fontweight='bold', color=TITLE_COLOR, y=0.98)
    
    def style_axis(ax, title, xlabel, ylabel, color):
        """Apply light theme styling to an axis."""
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, fontsize=14, fontweight='bold', color=TITLE_COLOR, pad=10)
        ax.set_xlabel(xlabel, fontsize=12, color=TEXT_COLOR, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=12, color=TEXT_COLOR, labelpad=8)
        ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=10, labelcolor=TEXT_COLOR)
        ax.grid(True, alpha=0.5, color=GRID_COLOR, linestyle='-')
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
            spine.set_linewidth(0.5)
    
    # -------------------------------------------------------------------------
    # 1. Episode Reward (Top-Left)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    smoothed_rewards = smooth_curve(list(rewards), window)
    ax1.plot(episodes, smoothed_rewards, color=REWARD_COLOR, linewidth=2.5, label=f'Smoothed (w={window})')
    style_axis(ax1, 'Episode Reward', 'Episode', 'Reward', REWARD_COLOR)
    ax1.legend(loc='lower right', facecolor=PANEL_BG, edgecolor=GRID_COLOR, 
               labelcolor=TEXT_COLOR, fontsize=9)
    
    # -------------------------------------------------------------------------
    # 2. Coverage Rate (Top-Right)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    smoothed_coverage = smooth_curve(list(coverage_pct), window)
    ax2.plot(episodes, smoothed_coverage, color=COVERAGE_COLOR, linewidth=2.5, label=f'Smoothed (w={window})')
    style_axis(ax2, 'Coverage Rate (%)', 'Episode', 'Coverage %', COVERAGE_COLOR)
    ax2.set_ylim(0, 110)
    ax2.axhline(y=100, color='#cccccc', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)
    
    # -------------------------------------------------------------------------
    # 3. Training Loss (Bottom-Left)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    if len(losses) > 0:
        clean_losses = [l for l in losses if l > 0]
        # Match steps to episodes approx
        loss_x = np.linspace(1, len(rewards), len(clean_losses))
        ax3.plot(loss_x, clean_losses, alpha=0.2, color=LOSS_COLOR, linewidth=0.5)
        
        if len(clean_losses) > window:
            smooth_loss = np.convolve(clean_losses, np.ones(window)/window, mode='valid')
            smooth_x = np.linspace(1, len(rewards), len(smooth_loss))
            ax3.plot(smooth_x, smooth_loss, color=LOSS_COLOR, linewidth=2.5, label='Loss')
            
    style_axis(ax3, 'Training Loss', 'Episode', 'Loss', LOSS_COLOR)
    ax3.legend(loc='upper right', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)
    
    # -------------------------------------------------------------------------
    # 4. Training Summary (Bottom-Right)
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    TRAINING SUMMARY
    ---------------
    Algorithm: {algorithm_name}
    Scenario: {scenario}
    Total Episodes: {len(rewards):,}
    
    Reward:
      Highest: {np.max(rewards):.2f}
      Lowest:  {np.min(rewards):.2f}
      Mean:    {np.mean(rewards):.2f}
    
    Coverage:
      Highest: {np.max(coverages)*100:.1f}%
      Lowest:  {np.min(coverages)*100:.1f}%
      Mean:    {np.mean(coverages)*100:.1f}%
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace', color=TEXT_COLOR,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f9ff', edgecolor=GRID_COLOR, alpha=0.8))
    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    dashboard_path = os.path.join(media_dir, f"{algo_prefix}_train_dashboard.png")
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG, edgecolor='none')
    plt.close()
    print(f"Saved: {dashboard_path}")
    
    # =========================================================================
    # Coverage Ratio Chart (Separate) - Clean visualization
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=LIGHT_BG)
    ax.set_facecolor(PANEL_BG)
    
    # Use larger smoothing window for cleaner chart
    smooth_window = max(100, window * 2)
    smoothed_coverage = smooth_curve(list(coverage_pct), smooth_window)
    
    # Plot smoothed line only (clean look)
    ax.plot(episodes, smoothed_coverage, color=COVERAGE_COLOR, linewidth=2.5, label=f'Smoothed (w={smooth_window})')
    
    # Mean line
    ax.axhline(y=np.mean(coverage_pct), color='#f59e0b', linestyle='--', linewidth=2,
               alpha=0.8, label=f'Mean: {np.mean(coverage_pct):.1f}%')
    
    ax.set_xlabel('Episode', fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel('Coverage Rate (%)', fontsize=12, color=TEXT_COLOR)
    ax.set_title(f'{algorithm_name} Target Coverage - Scenario {scenario}', fontsize=14, fontweight='bold', color=TITLE_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.set_ylim(0, 105)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    
    coverage_path = os.path.join(media_dir, f"{algo_prefix}_train_coverage.png")
    plt.savefig(coverage_path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close()
    print(f"Saved: {coverage_path}")
    
    # =========================================================================
    # Training Losses Chart (Separate)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=LIGHT_BG)
    ax.set_facecolor(PANEL_BG)
    
    if len(losses) > 0:
        clean_losses = [l for l in losses if l > 0]
        loss_x = np.linspace(1, len(rewards), len(clean_losses))
        ax.plot(loss_x, clean_losses, alpha=0.4, color=LOSS_COLOR, linewidth=0.5)
        
        if len(clean_losses) > window:
            smooth_loss = np.convolve(clean_losses, np.ones(window)/window, mode='valid')
            smooth_x = np.linspace(1, len(rewards), len(smooth_loss))
            ax.plot(smooth_x, smooth_loss, color=LOSS_COLOR, linewidth=2.5, label='Total Loss')
        
        # Mean line
        mean_loss = np.mean(clean_losses)
        ax.axhline(y=mean_loss, color='#f59e0b', linestyle='--', linewidth=2,
                   alpha=0.8, label=f'Mean: {mean_loss:.4f}')
    
    ax.set_xlabel('Episode', fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel('Loss', fontsize=12, color=TEXT_COLOR)
    ax.set_title(f'{algorithm_name} Training Loss - Scenario {scenario}', fontsize=14, fontweight='bold', color=TITLE_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(True, alpha=0.3, color=GRID_COLOR)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    
    loss_path = os.path.join(media_dir, f"{algo_prefix}_train_losses.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight', facecolor=LIGHT_BG)
    plt.close()
    print(f"Saved: {loss_path}")


def plot_test_results(results: dict, output_path: str, scenario: int = 1):
    """
    Generate test results chart.
    
    Creates:
    - test_results.png: Bar chart with coverage distribution and summary stats
    """
    episode_rewards = results.get("episode_rewards", [])
    coverage_rates = results.get("coverage_rates", [])
    
    if len(episode_rewards) == 0:
        print("Warning: No test results to plot")
        return
    
    coverage_pct = np.array(coverage_rates) * 100
    n_test_episodes = len(episode_rewards)
    
    # Try to get training episodes count from training_history
    training_episodes = "N/A"
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        history_path = os.path.join(script_dir, f"Scenario {scenario} Result", "checkpoints", "training_history.npz")
        if os.path.exists(history_path):
            data = np.load(history_path)
            training_episodes = len(data.get("episode_rewards", []))
    except:
        pass
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'NA²Q Test Results - Scenario {scenario}', fontsize=14, fontweight='bold')
    
    # Left: Coverage distribution histogram
    ax1 = axes[0]
    ax1.hist(coverage_pct, bins=min(20, n_test_episodes), color='#28A745', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(coverage_pct), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(coverage_pct):.1f}%')
    ax1.set_xlabel('Coverage Rate (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Coverage Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # Right: Summary statistics
    ax2 = axes[1]
    ax2.axis('off')
    
    summary_text = f"""
    Test Results Summary
    ────────────────────────────────
    Model trained on: {training_episodes} episodes
    Test episodes: {n_test_episodes}
    
    Coverage Rate:
      Mean:    {np.mean(coverage_pct):6.2f}%
      Std:     {np.std(coverage_pct):6.2f}%
      Best:    {np.max(coverage_pct):6.2f}%
      Worst:   {np.min(coverage_pct):6.2f}%
    
    Episode Reward:
      Mean:    {np.mean(episode_rewards):8.3f}
      Std:     {np.std(episode_rewards):8.3f}
      Best:    {np.max(episode_rewards):8.3f}
      Worst:   {np.min(episode_rewards):8.3f}
    """
    
    ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_video(
    model_path: str,
    scenario: int = 1,
    output_path: str = "results/demo.gif",
    duration: int = 20,
    fps: int = 10,
    device: Optional[str] = None,
    seed: int = 42,
    single_episode: bool = True
):
    """
    Generate video of trained agent.
    
    Creates GIF showing:
    - Grid layout
    - Sensor positions and FoV
    - Target movements
    - Tracking status
    
    Args:
        model_path: Path to trained model checkpoint
        scenario: Environment scenario (1 or 2)
        output_path: Output path for GIF
        duration: Video duration in seconds (ignored if single_episode=True)
        fps: Frames per second for output GIF
        device: Device to run on (cuda/cpu)
        seed: Random seed
        single_episode: If True, run exactly 1 episode (default). If False, run until duration.
    """
    try:
        import imageio
    except ImportError:
        print("Error: imageio required for video generation. Install with: pip install imageio")
        return
    
    from environments.environment import make_env
    from na2q.models import NA2QAgent
    
    # Auto-detect device if not specified
    device = get_device(device)
    
    # Create environment with rgb_array rendering
    env = make_env(scenario=scenario, render_mode="rgb_array", seed=seed)
    # Enforce realistic observations
    env.set_curriculum_difficulty(1.0)
    print("Video Generation: Realistic Observations Enabled (Limited FoV)")

    
    # Create and load agent (use hidden_dim=128 to match trainer)
    agent = NA2QAgent(
        n_agents=env.n_sensors,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=128,
        rnn_hidden_dim=128,
        attention_hidden_dim=128,
        device=device
    )
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path} (best model)")
    else:
        print(f"Warning: Model not found at {model_path}, using random policy")
    
    # Determine how many frames to capture
    if single_episode:
        max_frames = env.max_steps  # 1 episode = max_steps frames
        print(f"Generating 1 test episode ({max_frames} steps) at {fps} FPS")
    else:
        max_frames = duration * fps
        print(f"Generating {duration}s video at {fps} FPS ({max_frames} frames)")
    
    print(f"Scenario {scenario}: {env.grid_size}×{env.grid_size} grid, {env.n_sensors} sensors, {env.n_targets} targets")
    
    frames = []
    frame_count = 0
    episode_count = 0
    total_reward = 0.0
    
    while frame_count < max_frames:
        obs_list, info = env.reset()
        observations = np.stack(obs_list)
        agent.init_hidden(1)
        prev_actions = np.zeros(env.n_sensors, dtype=np.int64)
        episode_count += 1
        
        done, truncated = False, False
        episode_reward = 0.0
        
        while not done and not truncated and frame_count < max_frames:
            # Render frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                frame_count += 1
            
            # Take action
            avail_actions = np.stack(env.get_avail_actions())
            actions = agent.select_actions(observations, prev_actions, avail_actions, evaluate=True)
            next_obs_list, reward, done, truncated, info = env.step(actions.tolist())
            observations = np.stack(next_obs_list)
            prev_actions = actions
            episode_reward += reward
        
        total_reward += episode_reward
        
        # Stop after 1 episode if single_episode mode
        if single_episode:
            coverage = info.get('coverage_rate', 0) * 100
            print(f"Episode completed: Reward={episode_reward:.2f}, Final Coverage={coverage:.1f}%")
            break
    
    env.close()
    
    # Save video
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    if frames:
        # Calculate frame duration for target FPS
        frame_duration = 1.0 / fps
        imageio.mimsave(output_path, frames, duration=frame_duration)
        print(f"Saved video to {output_path} ({len(frames)} frames)")
    else:
        print("Warning: No frames captured")

