# NA²Q - Neural Attention Additive Q-Learning for Directional Sensor Networks.


import argparse
import os
import sys

# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="NA²Q: Multi-Agent Q-Learning on DSN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m na2q.main --mode train --scenario 1
  python -m na2q.main --mode test --scenario 1
""",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "quick-test"],
    )

    # Environment
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max-steps", type=int, default=100)

    # Training (defaults from train_config.py)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--epsilon-decay", type=int, default=None)
    parser.add_argument("--target-update", type=int, default=None)
    parser.add_argument("--buffer-capacity", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)

    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--test-episodes", type=int, default=10)

    # Paths
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-dir", type=str, default="Result")

    # Hardware
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Misc
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


# =============================================================================
# Train Mode
# =============================================================================


def run_train(args):
    """Run training mode."""
    from config import get_training_config
    from na2q.engine.trainer import Trainer

    # Load base config
    config = get_training_config(args.scenario)

    # Apply defaults to args for display/overrides if not provided
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    # Print config
    print(f"Training config (Scenario {args.scenario}) - from config.py:")
    print(f"  episodes        : {args.episodes}")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  lr              : {args.lr}")
    print(f"  gamma           : {args.gamma}")
    print(f"  epsilon_decay   : {args.epsilon_decay}")
    print(f"  num_envs        : {args.num_envs}")

    exp_name = args.exp_name or f"scenario{args.scenario}"

    # Update config with args (in case args were provided on CLI)
    config.update(
        {
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "buffer_capacity": args.buffer_capacity,
            "n_episodes": args.episodes,  # Trainer expects n_episodes
            "log_dir": ".",
            "exp_name": f"Scenario {args.scenario} Result",
            "use_amp": not args.no_amp,
            "learning_starts": getattr(args, "learning_starts", 5000),
            "target_update_interval": args.target_update,
            "scenario": args.scenario,
            "resume": args.resume,
        }
    )

    # Train
    trainer = Trainer(config)
    result = trainer.train()

    print(f"\nTraining completed!")
    print(f"  Best model: {result['best_model_path']}")

    return result


# =============================================================================
# Test Mode
# =============================================================================


def run_test(args):
    """Run test/evaluation mode."""
    from na2q.test import test

    class TestArgs:
        def __init__(self, args):
            na2q_dir = os.path.dirname(os.path.abspath(__file__))
            self.model = args.model or os.path.join(
                na2q_dir, "checkpoints", "best_model.pt"
            )
            self.scenario = args.scenario
            self.episodes = args.test_episodes
            self.max_steps = args.max_steps
            self.render = args.render
            self.device = args.device
            self.seed = args.seed
            self.verbose = args.verbose
            self.hidden_dim = 128  # Default hidden dimension

    return test(TestArgs(args))


# =============================================================================
# Quick Test Mode
# =============================================================================


def run_quick_test(args):
    """Run quick test to verify everything works."""
    from na2q.test import run_quick_test

    run_quick_test()


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    import torch

    from na2q.utils import get_device

    args = parse_args()

    # Auto-detect device
    device = get_device(args.device)
    args.device = device

    # Banner
    print("=" * 60)
    print("NA²Q: Neural Attention Additive Q-Learning")
    print("Applied to Directional Sensor Network")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Scenario: {args.scenario}")
    print(f"Device: {device}")
    if device == "cuda":
        if args.gpu_id is not None:
            torch.cuda.set_device(args.gpu_id)
        print(f"  CUDA Device: {torch.cuda.get_device_name()}")
    print("=" * 60)

    # Dispatch
    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    elif args.mode == "quick-test":
        run_quick_test(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
