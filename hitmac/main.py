"""
=============================================================================
HiT-MAC: Hierarchical Twin-Actor Multi-Agent Coordination
=============================================================================

A hierarchical A3C approach to multi-agent reinforcement learning.
Uses the same DSNEnv environment as NA²Q for fair algorithm comparison.

OUTPUT STRUCTURE:
-----------------
    hitmac/
    └── checkpoints/
        ├── best.pt                 # Best model checkpoint
        ├── latest.pt               # Latest model checkpoint
        └── training_history.npz    # Training metrics
    
    Scenario X Result/              # Results folder (charts & GIFs)
        ├── hitmac_scenario1_demo.gif
        └── hitmac_train_dashboard.png

Usage:
    python -m hitmac.main --mode train --scenario 1
    python -m hitmac.main --mode test --scenario 1
"""

import argparse
import os
import sys


# =============================================================================
# Argument Classes (module-level for pickle compatibility)
# =============================================================================

class TrainArgs:
    """Training arguments for worker processes (pickle-compatible)."""
    def __init__(self, env, scenario, seed, workers, num_steps, max_step,
                 test_eps, gamma, tau, entropy, lstm_out, lr, log_dir,
                 checkpoints_dir, results_dir, gpu_ids, model, norm_reward,
                 render, render_save, sleep_time, fix, train_mode, optimizer):
        self.env = env
        self.scenario = scenario
        self.seed = seed
        self.workers = workers
        self.num_steps = num_steps
        self.max_step = max_step
        self.test_eps = test_eps
        self.gamma = gamma
        self.tau = tau
        self.entropy = entropy
        self.lstm_out = lstm_out
        self.lr = lr
        self.log_dir = log_dir
        self.checkpoints_dir = checkpoints_dir
        self.results_dir = results_dir
        self.gpu_ids = gpu_ids
        self.model = model
        self.norm_reward = norm_reward
        self.render = render
        self.render_save = render_save
        self.sleep_time = sleep_time
        self.fix = fix
        self.train_mode = train_mode
        self.optimizer = optimizer


class ModelArgs:
    """Model arguments (pickle-compatible)."""
    def __init__(self, model, lstm_out):
        self.model = model
        self.lstm_out = lstm_out


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="HiT-MAC: Hierarchical Multi-Agent Coordination for DSN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hitmac.main --mode train --scenario 1
  python -m hitmac.main --mode test --scenario 1
"""
    )
    
    # Mode
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test"])
    
    # Environment
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    
    # Training
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel training workers")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor")
    parser.add_argument("--entropy", type=float, default=None,
                        help="Entropy regularization coefficient")
    parser.add_argument("--lstm-out", type=int, default=None,
                        help="LSTM/attention output size")
    
    # Evaluation  
    parser.add_argument("--test-episodes", type=int, default=10,
                        help="Number of test episodes")
    
    # Paths
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--results-dir", type=str, default="Result",
                        help="Results directory suffix")
    
    # Hardware
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1)
    
    # Misc
    parser.add_argument("--render", action="store_true")
    
    return parser.parse_args()


# =============================================================================
# Train Mode
# =============================================================================

def run_train(args):
    """Run HiT-MAC training."""
    import torch
    import torch.multiprocessing as mp
    from datetime import datetime
    
    from environments.environment import DSNEnv
    from hitmac.models import build_model
    from hitmac.train import train
    from hitmac.test import test
    from hitmac.shared_optim import SharedAdam
    from config import get_hitmac_config
    
    # Load config defaults (coordinator mode is default for multi-agent)
    # We use "coordinator" presets as the baseline
    config = get_hitmac_config("coordinator")
    
    # Apply defaults if args are not provided
    if args.workers is None: args.workers = config.get("workers", 4)
    if args.max_steps is None: args.max_step = config.get("max_step", 500000) # Note: config uses max_step, args uses max_steps
    else: args.max_step = args.max_steps
    
    if args.lr is None: args.lr = config.get("lr", 0.0005)
    if args.gamma is None: args.gamma = config.get("gamma", 0.9)
    if args.entropy is None: args.entropy = config.get("entropy", 0.01)
    if args.lstm_out is None: args.lstm_out = config.get("lstm_out", 128)
    
    print(f"HiT-MAC Training Config:")
    print(f"  Max Steps: {args.max_step} (approx {args.max_step // 100} episodes)")
    print(f"  Workers: {args.workers}")
    print(f"  LR: {args.lr}")
    
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
        torch.manual_seed(args.seed)
    
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set
    
    # Create environment to get dimensions
    env = DSNEnv(scenario=args.scenario)
    
    # Build model
    print(f"Building model for {env.n_sensors} agents, {env.n_targets} targets...")
    
    model_args = ModelArgs("single-att", args.lstm_out)
    shared_model = build_model(
        env.n_sensors, env.n_targets, env.n_actions,
        model_args, device
    )
    shared_model = shared_model.to(device)
    shared_model.share_memory()
    env.close()
    
    # Setup optimizer
    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr, amsgrad=True)
    optimizer.share_memory()
    
    # Setup directory structure
    hitmac_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(hitmac_dir, "checkpoints")
    
    # Results (charts, GIFs) go in Result/ScenarioX/
    results_dir = os.path.join("Result", f"Scenario{args.scenario}")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create args object for workers (module-level class for pickle)
    train_args = TrainArgs(
        env=f"Pose-v{args.scenario}",
        scenario=args.scenario,
        seed=args.seed,
        workers=args.workers,
        num_steps=20,
        max_step=args.max_steps,
        test_eps=args.test_episodes,
        gamma=args.gamma,
        tau=1.0,
        entropy=args.entropy,
        lstm_out=args.lstm_out,
        lr=args.lr,
        log_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
        results_dir=results_dir,
        gpu_ids=[-1] if args.device != "cuda" else [0],
        model="single-att",
        norm_reward=False,
        render=args.render,
        render_save=False,
        sleep_time=0,
        fix=False,
        train_mode=1,
        optimizer="Adam"
    )
    
    # Launch processes
    print(f"\nStarting training with {args.workers} workers...")
    print(f"Checkpoints: {checkpoints_dir}")
    print(f"Results: {results_dir}")
    
    processes = []
    manager = mp.Manager()
    train_modes = manager.list()
    n_iters = manager.list()
    
    # Shared lists for training history (matches NA2Q format)
    # These will be populated by worker processes as they finish episodes
    episode_rewards = manager.list()
    coverage_rates = manager.list()
    episode_durations = manager.list()
    
    # Start test process
    p = mp.Process(target=test, args=(train_args, shared_model, optimizer, train_modes, n_iters, 
                                     episode_rewards, coverage_rates, episode_durations))
    p.start()
    processes.append(p)
    
    # Start training workers
    for rank in range(args.workers):
        p = mp.Process(target=train, args=(rank, train_args, shared_model, optimizer, train_modes, n_iters,
                                          episode_rewards, coverage_rates, episode_durations))
        p.start()
        processes.append(p)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    print("\nTraining complete!")
    print(f"Best model: {os.path.join(checkpoints_dir, 'best.pt')}")
    print(f"Training history: {os.path.join(checkpoints_dir, 'training_history.npz')}")
    
    # Generate charts
    print("\nGenerating training visualizations...")
    try:
        from visualize import plot_training_results
        plot_training_results(
            exp_dir=results_dir,  # Fallback
            history_dir=checkpoints_dir,
            media_dir=results_dir,
            scenario=args.scenario,
            algorithm_name="HiT-MAC"
        )
    except Exception as e:
        print(f"Warning: Failed to generate charts: {e}")
    
    return {"checkpoints_dir": checkpoints_dir, "results_dir": results_dir}


# =============================================================================
# Test Mode
# =============================================================================

def run_test(args):
    """Run HiT-MAC evaluation."""
    import torch
    import numpy as np
    
    from environments.environment import DSNEnv
    from hitmac.models import build_model
    
    # Setup device
    device = torch.device(args.device if args.device else "cpu")
    
    # Checkpoints directory inside hitmac/
    hitmac_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(hitmac_dir, "checkpoints")
    
    # Find model
    model_path = args.model
    if model_path is None:
        candidates = [
            os.path.join(checkpoints_dir, "best.pt"),
            os.path.join(checkpoints_dir, "latest.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
    
    if model_path is None or not os.path.exists(model_path):
        print(f"Error: No HiT-MAC model found in {checkpoints_dir}")
        print("Train first with: python -m hitmac.main --mode train --scenario 1")
        return
    
    print(f"Loading model from: {model_path}")
    
    # Create environment
    env = DSNEnv(scenario=args.scenario)
    
    model_args = ModelArgs("single-att", args.lstm_out)
    
    # Build and load model
    model = build_model(
        env.n_sensors, env.n_targets, env.n_actions,
        model_args, device
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    # Run evaluation
    print(f"\nRunning {args.test_episodes} test episodes...")
    
    rewards = []
    coverages = []
    
    for ep in range(args.test_episodes):
        obs_list, info = env.reset()
        obs = np.array(obs_list, dtype=np.float32).reshape(env.n_sensors, env.n_targets, 4)
        state = torch.from_numpy(obs).float()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                value, actions, _, _ = model(state)
            
            obs_list, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            obs = np.array(obs_list, dtype=np.float32).reshape(env.n_sensors, env.n_targets, 4)
            state = torch.from_numpy(obs).float()
            episode_reward += reward
        
        coverage = info.get('coverage_rate', 0)
        rewards.append(episode_reward)
        coverages.append(coverage)
        
        if args.render or len(rewards) <= 3:
            print(f"  Episode {ep+1}: Reward={episode_reward:.2f}, Coverage={coverage*100:.1f}%")
    
    env.close()
    
    print(f"\n{'='*40}")
    print(f"Results over {args.test_episodes} episodes:")
    print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean Coverage: {np.mean(coverages)*100:.1f}% ± {np.std(coverages)*100:.1f}%")
    print(f"{'='*40}")
    
    return {"rewards": rewards, "coverages": coverages}


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    args = parse_args()
    
    # Banner
    print("=" * 60)
    print("HiT-MAC: Hierarchical Twin-Actor Multi-Agent Coordination")
    print("Applied to Directional Sensor Network")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Scenario: {args.scenario}")
    print(f"Device: {args.device or 'auto'}")
    print("=" * 60)
    
    # Dispatch
    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
