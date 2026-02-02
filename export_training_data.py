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
    na2q_path = os.path.join("na2q", "checkpoints", "training_history.npz")
    hitmac_path = os.path.join("hitmac", "checkpoints", "training_history.npz")
    output_dir = "docs/data"
    output_path = os.path.join(output_dir, "training_data.js")  # Changed to .js
    
    export_data = {}
    
    # Configuration
    MAX_EPISODES = 15000  # Limit both algorithms to 15,000 episodes
    HITMAC_STEPS_PER_EPISODE = 100  # Convert HiT-MAC steps to episodes
    
    # helper to process NA2Q history (already episode-based)
    def process_na2q_history(path):
        if not os.path.exists(path):
            print(f"⚠️  NA2Q history not found at: {path}")
            return None
            
        print(f"📂 Loading NA2Q history from: {path}")
        data = np.load(path, allow_pickle=True)
        
        # Extract metrics (already episode-based)
        total_episodes = min(len(data['episode_rewards']), MAX_EPISODES)
        episodes = list(range(total_episodes))
        rewards = data['episode_rewards'][:total_episodes].tolist()
        coverage = data['coverage_rates'][:total_episodes].tolist() if 'coverage_rates' in data else []
        losses = data['losses'][:total_episodes].tolist() if 'losses' in data else []
        durations = data['episode_durations'][:total_episodes].tolist() if 'episode_durations' in data else [0] * total_episodes
        cumulative_time = np.cumsum(durations).tolist()
        
        # Calculate epsilon decay (based on config)
        epsilon_decay = 7500 if scenario == 1 else 1000
        epsilon = [max(0.05, 1.0 - ep / epsilon_decay) for ep in episodes]
        
        # Sample every N episodes to reduce file size
        sample_rate = max(1, len(episodes) // 300)
        
        return {
            "episodes": episodes[::sample_rate],
            "rewards": [round(r, 3) for r in rewards[::sample_rate]],
            "coverage": [round(c * 100, 1) if c <= 1 else round(c, 1) for c in coverage[::sample_rate]] if coverage else [],
            "loss": [round(l, 4) for l in losses[::sample_rate]] if losses else [],
            "epsilon": [round(e, 3) for e in epsilon[::sample_rate]],
            "time": [round(t, 1) for t in cumulative_time[::sample_rate]]
        }
    
    # helper to process HiT-MAC history (step-based, needs conversion)
    def process_hitmac_history(path):
        if not os.path.exists(path):
            print(f"⚠️  HiT-MAC history not found at: {path}")
            return None
            
        print(f"📂 Loading HiT-MAC history from: {path}")
        data = np.load(path, allow_pickle=True)
        
        # HiT-MAC stores step-based data, convert to episodes (100 steps = 1 episode)
        raw_coverage = data['coverage_rates'] if 'coverage_rates' in data else []
        raw_rewards = data['episode_rewards'] if 'episode_rewards' in data else []
        
        if len(raw_coverage) == 0:
            print(f"⚠️  HiT-MAC has no coverage data")
            return None
        
        # Convert steps to episodes by averaging every 100 steps
        num_steps = len(raw_coverage)
        num_episodes = min(num_steps // HITMAC_STEPS_PER_EPISODE, MAX_EPISODES)
        
        print(f"   Converting {num_steps} steps → {num_episodes} episodes (100 steps/episode)")
        
        episodes = []
        avg_rewards = []
        avg_coverage = []
        
        for ep in range(num_episodes):
            start_idx = ep * HITMAC_STEPS_PER_EPISODE
            end_idx = start_idx + HITMAC_STEPS_PER_EPISODE
            
            # Average coverage for this "episode" (100 steps)
            ep_coverage = raw_coverage[start_idx:end_idx]
            avg_cov = np.mean(ep_coverage) if len(ep_coverage) > 0 else 0
            
            # Sum rewards for this "episode" (100 steps)
            if len(raw_rewards) > 0:
                ep_rewards = raw_rewards[start_idx:end_idx] if end_idx <= len(raw_rewards) else raw_rewards[start_idx:]
                ep_reward = np.sum(ep_rewards) if len(ep_rewards) > 0 else 0
            else:
                ep_reward = avg_cov  # Use coverage as proxy if no reward data
            
            episodes.append(ep)
            avg_coverage.append(avg_cov)
            avg_rewards.append(ep_reward)
        
        # Calculate epsilon decay for HiT-MAC
        epsilon_decay = 8000
        epsilon = [max(0.1, 1.0 - ep / epsilon_decay) for ep in episodes]
        
        # Sample every N episodes to reduce file size
        sample_rate = max(1, len(episodes) // 300)
        
        return {
            "episodes": episodes[::sample_rate],
            "rewards": [round(r, 3) for r in avg_rewards[::sample_rate]],
            "coverage": [round(c * 100, 1) if c <= 1 else round(c, 1) for c in avg_coverage[::sample_rate]],
            "loss": [],  # HiT-MAC doesn't log loss in the same way
            "epsilon": [round(e, 3) for e in epsilon[::sample_rate]],
            "time": []
        }

    # Process NA2Q
    na2q_data = process_na2q_history(na2q_path)
    if na2q_data:
        export_data[f"scenario{scenario}"] = na2q_data
        
    # Process HiT-MAC
    hitmac_data = process_hitmac_history(hitmac_path)
    if hitmac_data:
        export_data[f"hitmac_scenario{scenario}"] = hitmac_data
    
    # Metadata
    export_data["metadata"] = {
        "scenario": scenario,
        "total_episodes": MAX_EPISODES,
        "hitmac_steps_per_episode": HITMAC_STEPS_PER_EPISODE,
        "note": "HiT-MAC data converted from steps to episodes (100 steps = 1 episode, using average coverage)",
        "exported_at": __import__('datetime').datetime.now().isoformat()
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JS file with global variable
    # This allows loading locally without CORS issues (unlike fetch)
    json_str = json.dumps(export_data, indent=2)
    js_content = f"window.trainingData = {json_str};\n"
    
    with open(output_path, 'w') as f:
        f.write(js_content)
    
    print(f"✅ Exported training data to: {output_path}")
    if na2q_data:
        print(f"   NA2Q Episodes: {len(na2q_data['episodes'])}")
    if hitmac_data:
        print(f"   HiT-MAC Episodes: {len(hitmac_data['episodes'])}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")

def export_demo_gif(scenario: int = 1):
    """Copy the demo GIF to website assets."""
    
    import shutil
    
    # NA2Q GIF
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
