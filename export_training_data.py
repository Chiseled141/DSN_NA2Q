"""
Export training data to JSON for the static website.

Run this AFTER training is complete:
    python export_training_data.py --scenario 1

This creates website/data/training_data.json with your real training metrics.
"""

import json
import numpy as np
import argparse
import os

def export_training_data(scenario: int = 1):
    """Export training history to JSON for website dashboard."""
    
    # Paths
    result_dir = f"Scenario {scenario} Result"
    history_path = os.path.join(result_dir, "checkpoints", "training_history.npz")
    output_dir = "docs/data"
    output_path = os.path.join(output_dir, "training_data.json")
    
    # Check if training data exists
    if not os.path.exists(history_path):
        print(f"❌ Training data not found at: {history_path}")
        print(f"   Run training first: python -m na2q.main --mode train --scenario {scenario}")
        return
    
    # Load training history
    print(f"📂 Loading training history from: {history_path}")
    data = np.load(history_path, allow_pickle=True)
    
    # Extract metrics
    episodes = list(range(len(data['episode_rewards'])))
    rewards = data['episode_rewards'].tolist()
    coverage = data['coverage_rates'].tolist() if 'coverage_rates' in data else []
    losses = data['losses'].tolist() if 'losses' in data else []
    
    # Calculate epsilon decay (based on config)
    epsilon_decay = 7500 if scenario == 1 else 1000
    epsilon = [max(0.05, 1.0 - ep / epsilon_decay) for ep in episodes]
    
    # Sample every N episodes to reduce file size
    sample_rate = max(1, len(episodes) // 300)
    
    # Create export data
    export_data = {
        f"scenario{scenario}": {
            "episodes": episodes[::sample_rate],
            "rewards": [round(r, 3) for r in rewards[::sample_rate]],
            "coverage": [round(c * 100, 1) if c <= 1 else round(c, 1) for c in coverage[::sample_rate]] if coverage else [],
            "loss": [round(l, 4) for l in losses[::sample_rate]] if losses else [],
            "epsilon": [round(e, 3) for e in epsilon[::sample_rate]]
        },
        "metadata": {
            "scenario": scenario,
            "total_episodes": len(episodes),
            "final_coverage": round(coverage[-1] * 100, 1) if coverage else 0,
            "best_reward": round(max(rewards), 2),
            "exported_at": __import__('datetime').datetime.now().isoformat()
        }
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing data if present (to merge multiple scenarios)
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing = json.load(f)
            existing.update(export_data)
            export_data = existing
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✅ Exported training data to: {output_path}")
    print(f"   Episodes: {len(episodes)}")
    print(f"   Sampled: {len(episodes[::sample_rate])} points")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")

def export_demo_gif(scenario: int = 1):
    """Copy the demo GIF to website assets."""
    
    import shutil
    
    gif_path = f"Scenario {scenario} Result/na2q_scenario{scenario}_demo.gif"
    output_dir = "docs/assets/images"
    output_path = os.path.join(output_dir, f"scenario{scenario}_demo.gif")
    
    if not os.path.exists(gif_path):
        # Try alternative path
        gif_path = f"Scenario {scenario} Result/scenario{scenario}_demo.gif"
    
    if os.path.exists(gif_path):
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(gif_path, output_path)
        print(f"✅ Copied demo GIF to: {output_path}")
    else:
        print(f"⚠️  Demo GIF not found. Generate it with:")
        print(f"   python -m na2q.main --mode video --scenario {scenario}")

def main():
    parser = argparse.ArgumentParser(description='Export training data for website')
    parser.add_argument('--scenario', type=int, default=1, help='Scenario to export (1 or 2)')
    parser.add_argument('--all', action='store_true', help='Export all available scenarios')
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
