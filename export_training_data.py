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
    
    # helper to process a single history file
    def process_history(path, key_prefix):
        if not os.path.exists(path):
            print(f"⚠️  {key_prefix} history not found at: {path}")
            return None
            
        print(f"📂 Loading {key_prefix} history from: {path}")
        data = np.load(path, allow_pickle=True)
        
        # Extract metrics
        episodes = list(range(len(data['episode_rewards'])))
        rewards = data['episode_rewards'].tolist()
        coverage = data['coverage_rates'].tolist() if 'coverage_rates' in data else []
        losses = data['losses'].tolist() if 'losses' in data else []
        durations = data['episode_durations'].tolist() if 'episode_durations' in data else [0] * len(episodes)
        cumulative_time = np.cumsum(durations).tolist()
        
        # Calculate epsilon decay (based on config)
        epsilon_decay = 7500 if scenario == 1 else 1000
        # HiT-MAC might have different decay, but using same formula for viz consistency if not logged
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

    # Process NA2Q
    na2q_data = process_history(na2q_path, "NA2Q")
    if na2q_data:
        export_data[f"scenario{scenario}"] = na2q_data
        
    # Process HiT-MAC
    hitmac_data = process_history(hitmac_path, "HiT-MAC")
    if hitmac_data:
        export_data[f"hitmac_scenario{scenario}"] = hitmac_data
    
    # Metadata
    export_data["metadata"] = {
        "scenario": scenario,
        "total_episodes": len(na2q_data["episodes"]) if na2q_data else (len(hitmac_data["episodes"]) if hitmac_data else 0),
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
