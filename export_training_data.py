"""
Export training data to JSON for the static website.

Run this AFTER training is complete:
    python export_training_data.py --scenario 1

This creates website/data/training_data.json with your real training metrics.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np

from config import get_hitmac_training_config


def export_training_data(scenario: int = 1):
    """Export training history to JSON for website dashboard."""

    # Paths
    na2q_path = os.path.join("na2q", "checkpoints", "training_history.npz")
    hitmac_path = os.path.join("hitmac", "checkpoints", "training_history.npz")
    output_dir = "docs/data"
    output_path = os.path.join(output_dir, "training_data.js")  # Changed to .js

    export_data = {}

    # Each HiT-MAC worker logs one entry per episode (when player.done).
    # With N_WORKERS parallel workers each running the full max_step budget,
    # the log has N_WORKERS × (max_step / max_steps_per_episode) entries.
    # Grouping by N_WORKERS converts to effective wall-clock episode count,
    # making the x-axis comparable with NA2Q (single-process, one entry per episode).
    hitmac_cfg = get_hitmac_training_config(scenario)
    N_WORKERS = hitmac_cfg.get("workers", 4)

    # helper to process NA2Q history (already episode-based)
    def process_na2q_history(path, max_episodes=None):
        if not os.path.exists(path):
            print(f"⚠️  NA2Q history not found at: {path}")
            return None, None

        print(f"📂 Loading NA2Q history from: {path}")
        data = np.load(path, allow_pickle=True)

        # Extract metrics (already episode-based, one entry per episode)
        n = len(data["episode_rewards"])
        if max_episodes is not None:
            n = min(n, max_episodes)

        episodes = list(range(n))
        rewards = data["episode_rewards"][:n].tolist()
        coverage = data["coverage_rates"][:n].tolist() if "coverage_rates" in data else []
        losses = data["losses"][:n].tolist() if "losses" in data else []
        durations = data["episode_durations"][:n].tolist() if "episode_durations" in data else [0] * n
        cumulative_time = np.cumsum(durations).tolist()

        epsilon_decay = 7500 if scenario == 1 else 1000
        epsilon = [max(0.05, 1.0 - ep / epsilon_decay) for ep in episodes]

        # Full data (dashboard) — up to 5000 points
        full_rate = max(1, n // 5000)
        full_data = {
            "episodes": episodes[::full_rate],
            "rewards": [round(r, 3) for r in rewards[::full_rate]],
            "coverage": [round(c * 100, 1) if c <= 1 else round(c, 1) for c in coverage[::full_rate]] if coverage else [],
            "loss": [round(l, 4) for l in losses[::full_rate]] if losses else [],
            "epsilon": [round(e, 3) for e in epsilon[::full_rate]],
            "time": [round(t, 1) for t in cumulative_time[::full_rate]],
        }

        # Sampled data (comparison chart) — ~300 points
        sample_rate = max(1, n // 300)
        sampled_data = {
            "episodes": episodes[::sample_rate],
            "rewards": [round(r, 3) for r in rewards[::sample_rate]],
            "coverage": [round(c * 100, 1) if c <= 1 else round(c, 1) for c in coverage[::sample_rate]] if coverage else [],
            "loss": [round(l, 4) for l in losses[::sample_rate]] if losses else [],
            "epsilon": [round(e, 3) for e in epsilon[::sample_rate]],
            "time": [round(t, 1) for t in cumulative_time[::sample_rate]],
        }

        print(f"   NA2Q: {n} episodes")
        return sampled_data, full_data

    # helper to process HiT-MAC history
    # The npz stores one entry per episode per worker (logged when player.done).
    # With N_WORKERS parallel workers, there are N_WORKERS × ep_per_worker entries.
    # Grouping by N_WORKERS converts to effective wall-clock episodes, matching NA2Q.
    def process_hitmac_history(path, max_episodes=None):
        if not os.path.exists(path):
            print(f"⚠️  HiT-MAC history not found at: {path}")
            return None, None

        print(f"📂 Loading HiT-MAC history from: {path}")
        data = np.load(path, allow_pickle=True)

        raw_coverage = data["coverage_rates"] if "coverage_rates" in data else []
        raw_rewards = data["episode_rewards"] if "episode_rewards" in data else []

        if len(raw_coverage) == 0:
            print(f"⚠️  HiT-MAC has no coverage data")
            return None, None

        # Group every N_WORKERS entries → one effective wall-clock episode
        n_raw = len(raw_coverage)
        num_episodes = n_raw // N_WORKERS
        if max_episodes is not None:
            num_episodes = min(num_episodes, max_episodes)

        print(f"   {n_raw} raw entries → {num_episodes} episodes (grouped by {N_WORKERS} workers)")

        episodes, avg_rewards, avg_coverage = [], [], []
        raw_rewards_arr = np.array(raw_rewards) if len(raw_rewards) > 0 else np.zeros(n_raw)
        raw_coverage_arr = np.array(raw_coverage)

        for ep in range(num_episodes):
            s, e = ep * N_WORKERS, (ep + 1) * N_WORKERS
            episodes.append(ep)
            avg_rewards.append(float(np.mean(raw_rewards_arr[s:e])))
            avg_coverage.append(float(np.mean(raw_coverage_arr[s:e])))

        epsilon_decay = 8000
        epsilon = [max(0.1, 1.0 - ep / epsilon_decay) for ep in episodes]

        # Full data (dashboard) — up to 5000 points
        full_rate = max(1, num_episodes // 5000)
        full_data = {
            "episodes": episodes[::full_rate],
            "rewards": [round(r, 3) for r in avg_rewards[::full_rate]],
            "coverage": [round(c * 100, 1) if c <= 1 else round(c, 1) for c in avg_coverage[::full_rate]],
            "loss": [],
            "epsilon": [round(e, 3) for e in epsilon[::full_rate]],
            "time": [],
        }

        # Sampled data (comparison chart) — ~300 points
        sample_rate = max(1, num_episodes // 300)
        sampled_data = {
            "episodes": episodes[::sample_rate],
            "rewards": [round(r, 3) for r in avg_rewards[::sample_rate]],
            "coverage": [round(c * 100, 1) if c <= 1 else round(c, 1) for c in avg_coverage[::sample_rate]],
            "loss": [],
            "epsilon": [round(e, 3) for e in epsilon[::sample_rate]],
            "time": [],
        }

        print(f"   HiT-MAC: {num_episodes} episodes")
        return sampled_data, full_data

    # First pass — determine HiT-MAC episode count, then align NA2Q to match
    hitmac_episodes_available = (
        np.load(hitmac_path, allow_pickle=True)["coverage_rates"].shape[0] // N_WORKERS
        if os.path.exists(hitmac_path) else None
    )
    na2q_episodes_available = (
        np.load(na2q_path, allow_pickle=True)["episode_rewards"].shape[0]
        if os.path.exists(na2q_path) else None
    )

    shared_episodes = None
    if hitmac_episodes_available and na2q_episodes_available:
        shared_episodes = min(hitmac_episodes_available, na2q_episodes_available)
        print(f"   Shared episode limit: {shared_episodes} (NA2Q={na2q_episodes_available}, HiT-MAC={hitmac_episodes_available})")

    na2q_sampled, na2q_full = process_na2q_history(na2q_path, max_episodes=shared_episodes)
    hitmac_sampled, hitmac_full = process_hitmac_history(hitmac_path, max_episodes=shared_episodes)

    if na2q_sampled:
        export_data[f"scenario{scenario}"] = na2q_sampled
        export_data[f"scenario{scenario}_full"] = na2q_full

    if hitmac_sampled:
        export_data[f"hitmac_scenario{scenario}"] = hitmac_sampled
        export_data[f"hitmac_scenario{scenario}_full"] = hitmac_full

    # Metadata
    export_data["metadata"] = {
        "scenario": scenario,
        "na2q_episodes": len(na2q_full["episodes"]) if na2q_full else 0,
        "hitmac_episodes": len(hitmac_full["episodes"]) if hitmac_full else 0,
        "note": f"Both datasets are episode-based. HiT-MAC: {N_WORKERS} parallel workers, entries grouped by worker count to align x-axis with NA2Q.",
        "exported_at": datetime.now().isoformat(),
    }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save as JS file with global variable
    # This allows loading locally without CORS issues (unlike fetch)
    json_str = json.dumps(export_data, indent=2)
    js_content = f"window.trainingData = {json_str};\n"

    with open(output_path, "w") as f:
        f.write(js_content)

    print(f"✅ Exported training data to: {output_path}")
    if na2q_full:
        print(
            f"   NA2Q: {len(na2q_full['episodes'])} episodes (sampled: {len(na2q_sampled['episodes'])})"
        )
    if hitmac_full:
        print(
            f"   HiT-MAC: {len(hitmac_full['episodes'])} episodes (sampled: {len(hitmac_sampled['episodes'])})"
        )
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")


def export_demo_gif(scenario: int = 1):
    """Copy the demo GIF to website assets."""

    import shutil

    # NA2Q GIF
    gif_path = os.path.join("Result", f"Scenario{scenario}", f"na2q_scenario{scenario}_demo.gif")
    output_dir = "docs/assets/images"
    output_path = os.path.join(output_dir, f"scenario{scenario}_demo.gif")

    if not os.path.exists(gif_path):
        gif_path = os.path.join("Result", f"Scenario{scenario}", f"scenario{scenario}_demo.gif")

    if os.path.exists(gif_path):
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(gif_path, output_path)
        print(f"✅ Copied demo GIF to: {output_path}")
    else:
        print(f"⚠️  Demo GIF not found. Generate it with:")
        print(f"   python -m na2q.main --mode video --scenario {scenario}")


def main():
    parser = argparse.ArgumentParser(description="Export training data for website")
    parser.add_argument(
        "--scenario", type=int, default=1, help="Scenario to export (1 or 2)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Export all available scenarios"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("NA²Q Training Data Export")
    print("=" * 50)

    if args.all:
        for s in [1, 2]:
            export_training_data(s)
            export_demo_gif(s)
    else:
        export_training_data(args.scenario)
        export_demo_gif(args.scenario)

    print("\n📌 Next steps:")
    print("   1. Open docs/index.html to preview")
    print("   2. Host the 'docs' folder on GitHub Pages or any static host")


if __name__ == "__main__":
    main()
