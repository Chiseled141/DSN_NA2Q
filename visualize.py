"""Visualization utilities for NA²Q training results."""

import os
import textwrap
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

from na2q.utils import get_device

# Color palette
_BG = "#ffffff"
_PANEL = "#f8f9fa"
_GRID = "#e0e0e0"
_TEXT = "#333333"
_TITLE = "#1a1a1a"
_BLUE = "#2563eb"
_GREEN = "#16a34a"
_RED = "#dc2626"
_AMBER = "#f59e0b"


def smooth_curve(values: List[float], window: int = 10) -> np.ndarray:
    """Smooth a curve using an expanding moving average."""
    if len(values) < window:
        return np.array(values)
    values_arr = np.array(values)
    smoothed = np.zeros(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed[i] = np.mean(values_arr[start : i + 1])
    return smoothed


def _style_axis(ax, title, xlabel, ylabel):
    ax.set_facecolor(_PANEL)
    ax.set_title(title, fontsize=14, fontweight="bold", color=_TITLE, pad=10)
    ax.set_xlabel(xlabel, fontsize=12, color=_TEXT, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12, color=_TEXT, labelpad=8)
    ax.tick_params(axis="both", colors=_TEXT, labelsize=10, labelcolor=_TEXT)
    ax.grid(True, alpha=0.5, color=_GRID, linestyle="-")
    for spine in ax.spines.values():
        spine.set_color(_GRID)
        spine.set_linewidth(0.5)


def _legend(ax):
    ax.legend(
        loc="lower right",
        facecolor=_PANEL,
        edgecolor=_GRID,
        labelcolor=_TEXT,
        fontsize=9,
    )


def plot_training_results(
    exp_dir: str,
    window: int = 50,
    history_dir: Optional[str] = None,
    media_dir: Optional[str] = None,
    scenario: int = 1,
    algorithm_name: str = "NA²Q",
):
    """Generate training dashboard and individual metric charts.

    Saves:
    - {algo}_train_dashboard.png  — 4-panel summary
    - {algo}_train_coverage.png   — coverage over time
    - {algo}_train_losses.png     — loss curves
    """
    history_dir = history_dir or os.path.join(exp_dir, "checkpoints")
    media_dir = media_dir or exp_dir
    os.makedirs(media_dir, exist_ok=True)

    algo_prefix = (
        algorithm_name.lower().replace("²", "2").replace("-", "").replace(" ", "")
    )

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

    # Dashboard (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=_BG)
    fig.patch.set_facecolor(_BG)
    fig.suptitle(
        f"{algorithm_name} Training Dashboard — Scenario {scenario}",
        fontsize=18,
        fontweight="bold",
        color=_TITLE,
        y=0.98,
    )

    # Reward
    ax1 = axes[0, 0]
    ax1.plot(
        episodes,
        smooth_curve(list(rewards), window),
        color=_BLUE,
        linewidth=2.5,
        label=f"Smoothed (w={window})",
    )
    _style_axis(ax1, "Episode Reward", "Episode", "Reward")
    _legend(ax1)

    # Coverage
    ax2 = axes[0, 1]
    ax2.plot(
        episodes,
        smooth_curve(list(coverage_pct), window),
        color=_GREEN,
        linewidth=2.5,
        label=f"Smoothed (w={window})",
    )
    _style_axis(ax2, "Coverage Rate (%)", "Episode", "Coverage %")
    ax2.set_ylim(0, 110)
    ax2.axhline(y=100, color=_GRID, linestyle="--", alpha=0.5)
    _legend(ax2)

    # Loss
    ax3 = axes[1, 0]
    if len(losses) > 0:
        clean_losses = [l for l in losses if l > 0]
        loss_x = np.linspace(1, len(rewards), len(clean_losses))
        ax3.plot(loss_x, clean_losses, alpha=0.2, color=_RED, linewidth=0.5)
        if len(clean_losses) > window:
            smooth_loss = np.convolve(
                clean_losses, np.ones(window) / window, mode="valid"
            )
            smooth_x = np.linspace(1, len(rewards), len(smooth_loss))
            ax3.plot(smooth_x, smooth_loss, color=_RED, linewidth=2.5, label="Loss")
    _style_axis(ax3, "Training Loss", "Episode", "Loss")
    ax3.legend(
        loc="upper right",
        facecolor=_PANEL,
        edgecolor=_GRID,
        labelcolor=_TEXT,
        fontsize=9,
    )

    # Summary text panel
    ax4 = axes[1, 1]
    ax4.axis("off")
    summary = textwrap.dedent(
        f"""\
        TRAINING SUMMARY
        ────────────────────────────────
        Algorithm : {algorithm_name}
        Scenario  : {scenario}
        Episodes  : {len(rewards):,}

        Reward
          Highest : {np.max(rewards):.2f}
          Lowest  : {np.min(rewards):.2f}
          Mean    : {np.mean(rewards):.2f}

        Coverage
          Highest : {np.max(coverages) * 100:.1f}%
          Lowest  : {np.min(coverages) * 100:.1f}%
          Mean    : {np.mean(coverages) * 100:.1f}%
    """
    )
    ax4.text(
        0.1,
        0.5,
        summary,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
        color=_TEXT,
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="#f0f9ff", edgecolor=_GRID, alpha=0.8
        ),
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    dashboard_path = os.path.join(media_dir, f"{algo_prefix}_train_dashboard.png")
    plt.savefig(
        dashboard_path, dpi=150, bbox_inches="tight", facecolor=_BG, edgecolor="none"
    )
    plt.close()
    print(f"Saved: {dashboard_path}")

    # Coverage chart
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
    ax.set_facecolor(_PANEL)
    smooth_window = max(100, window * 2)
    ax.plot(
        episodes,
        smooth_curve(list(coverage_pct), smooth_window),
        color=_GREEN,
        linewidth=2.5,
        label=f"Smoothed (w={smooth_window})",
    )
    ax.axhline(
        y=np.mean(coverage_pct),
        color=_AMBER,
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"Mean: {np.mean(coverage_pct):.1f}%",
    )
    ax.set_xlabel("Episode", fontsize=12, color=_TEXT)
    ax.set_ylabel("Coverage Rate (%)", fontsize=12, color=_TEXT)
    ax.set_title(
        f"{algorithm_name} Target Coverage — Scenario {scenario}",
        fontsize=14,
        fontweight="bold",
        color=_TITLE,
    )
    ax.tick_params(colors=_TEXT)
    ax.grid(True, alpha=0.3, color=_GRID)
    ax.set_ylim(0, 105)
    ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    coverage_path = os.path.join(media_dir, f"{algo_prefix}_train_coverage.png")
    plt.savefig(coverage_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"Saved: {coverage_path}")

    # Loss chart
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
    ax.set_facecolor(_PANEL)
    if len(losses) > 0:
        clean_losses = [l for l in losses if l > 0]
        loss_x = np.linspace(1, len(rewards), len(clean_losses))
        ax.plot(loss_x, clean_losses, alpha=0.4, color=_RED, linewidth=0.5)
        if len(clean_losses) > window:
            smooth_loss = np.convolve(
                clean_losses, np.ones(window) / window, mode="valid"
            )
            smooth_x = np.linspace(1, len(rewards), len(smooth_loss))
            ax.plot(
                smooth_x, smooth_loss, color=_RED, linewidth=2.5, label="Total Loss"
            )
        mean_loss = np.mean(clean_losses)
        ax.axhline(
            y=mean_loss,
            color=_AMBER,
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Mean: {mean_loss:.4f}",
        )
    ax.set_xlabel("Episode", fontsize=12, color=_TEXT)
    ax.set_ylabel("Loss", fontsize=12, color=_TEXT)
    ax.set_title(
        f"{algorithm_name} Training Loss — Scenario {scenario}",
        fontsize=14,
        fontweight="bold",
        color=_TITLE,
    )
    ax.tick_params(colors=_TEXT)
    ax.grid(True, alpha=0.3, color=_GRID)
    ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    loss_path = os.path.join(media_dir, f"{algo_prefix}_train_losses.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"Saved: {loss_path}")


def plot_test_results(results: dict, output_path: str, scenario: int = 1):
    """Generate test results chart with coverage distribution and summary stats."""
    episode_rewards = results.get("episode_rewards", [])
    coverage_rates = results.get("coverage_rates", [])

    if len(episode_rewards) == 0:
        print("Warning: No test results to plot")
        return

    coverage_pct = np.array(coverage_rates) * 100
    n_test_episodes = len(episode_rewards)

    training_episodes = "N/A"
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        history_path = os.path.join(
            script_dir,
            f"Scenario {scenario} Result",
            "checkpoints",
            "training_history.npz",
        )
        if os.path.exists(history_path):
            data = np.load(history_path)
            training_episodes = len(data.get("episode_rewards", []))
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"NA²Q Test Results — Scenario {scenario}", fontsize=14, fontweight="bold"
    )

    # Coverage histogram
    ax1 = axes[0]
    ax1.hist(
        coverage_pct,
        bins=min(20, n_test_episodes),
        color=_GREEN,
        alpha=0.7,
        edgecolor="black",
    )
    ax1.axvline(
        np.mean(coverage_pct),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(coverage_pct):.1f}%",
    )
    ax1.set_xlabel("Coverage Rate (%)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Coverage Distribution", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)

    # Summary text panel
    ax2 = axes[1]
    ax2.axis("off")
    summary = textwrap.dedent(
        f"""\
        TEST RESULTS SUMMARY
        ────────────────────────────────
        Trained on : {training_episodes} episodes
        Test eps   : {n_test_episodes}

        Coverage Rate
          Mean    : {np.mean(coverage_pct):6.2f}%
          Std     : {np.std(coverage_pct):6.2f}%
          Best    : {np.max(coverage_pct):6.2f}%
          Worst   : {np.min(coverage_pct):6.2f}%

        Episode Reward
          Mean    : {np.mean(episode_rewards):8.3f}
          Std     : {np.std(episode_rewards):8.3f}
          Best    : {np.max(episode_rewards):8.3f}
          Worst   : {np.min(episode_rewards):8.3f}
    """
    )
    ax2.text(
        0.1,
        0.5,
        summary,
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="#f0fff0", edgecolor=_GRID, alpha=0.8
        ),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
    single_episode: bool = True,
):
    """Generate a GIF of the trained agent in the environment.

    Args:
        model_path: Path to trained model checkpoint.
        scenario: Environment scenario (1 or 2).
        output_path: Output path for GIF.
        duration: Video duration in seconds (ignored when single_episode=True).
        fps: Frames per second.
        device: Device override (cuda/cpu); auto-detected if None.
        seed: Random seed.
        single_episode: Run exactly one episode if True, else run until duration.
    """
    try:
        import imageio
    except ImportError:
        print("Error: imageio required. Install with: pip install imageio")
        return

    from environments.environment import make_env
    from na2q.models import NA2QAgent

    device = get_device(device)
    env = make_env(scenario=scenario, render_mode="rgb_array", seed=seed)
    env.set_curriculum_difficulty(1.0)
    print("Video Generation: Realistic Observations Enabled (Limited FoV)")

    agent = NA2QAgent(
        n_agents=env.n_sensors,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=128,
        rnn_hidden_dim=128,
        attention_hidden_dim=128,
        device=device,
    )

    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using random policy")

    max_frames = env.max_steps if single_episode else duration * fps
    print(
        f"Scenario {scenario}: {env.grid_size}×{env.grid_size} grid, "
        f"{env.n_sensors} sensors, {env.n_targets} targets"
    )
    print(
        f"Generating {'1 episode' if single_episode else f'{duration}s'} "
        f"at {fps} FPS ({max_frames} frames)"
    )

    frames = []
    frame_count = 0

    while frame_count < max_frames:
        obs_list, info = env.reset()
        observations = np.stack(obs_list)
        agent.init_hidden(1)
        prev_actions = np.zeros(env.n_sensors, dtype=np.int64)
        done, truncated = False, False
        episode_reward = 0.0

        while not done and not truncated and frame_count < max_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                frame_count += 1
            avail_actions = np.stack(env.get_avail_actions())
            actions = agent.select_actions(
                observations, prev_actions, avail_actions, evaluate=True
            )
            next_obs_list, reward, done, truncated, info = env.step(actions.tolist())
            observations = np.stack(next_obs_list)
            prev_actions = actions
            episode_reward += reward

        if single_episode:
            coverage = info.get("coverage_rate", 0) * 100
            print(f"Episode: reward={episode_reward:.2f}, coverage={coverage:.1f}%")
            break

    env.close()

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    if frames:
        imageio.mimsave(output_path, frames, duration=1.0 / fps)
        print(f"Saved: {output_path} ({len(frames)} frames)")
    else:
        print("Warning: No frames captured")
