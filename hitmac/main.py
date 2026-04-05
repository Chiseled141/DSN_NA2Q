"""HiT-MAC: Hierarchical Twin-Actor Multi-Agent Coordination for DSN."""

import argparse
import os


class TrainArgs:
    """Training arguments for worker processes (pickle-compatible)."""

    def __init__(
        self,
        env,
        scenario,
        seed,
        workers,
        num_steps,
        max_step,
        test_eps,
        gamma,
        tau,
        entropy,
        lstm_out,
        lr,
        log_dir,
        checkpoints_dir,
        results_dir,
        gpu_ids,
        model,
        norm_reward,
        render,
        render_save,
        sleep_time,
        fix,
        train_mode,
        optimizer,
        start_step=0,
        save_interval=500000,
    ):
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
        self.start_step = start_step
        self.save_interval = save_interval


class ModelArgs:
    def __init__(self, model, lstm_out):
        self.model = model
        self.lstm_out = lstm_out


def parse_args():
    parser = argparse.ArgumentParser(
        description="HiT-MAC: Hierarchical Multi-Agent Coordination for DSN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hitmac.main --mode train --scenario 1
  python -m hitmac.main --mode test --scenario 1
""",
    )

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel training workers"
    )
    parser.add_argument(
        "--max-steps",
        default=None,
        type=int,
        metavar="N",
        help="number of training steps (default: from config)",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor")
    parser.add_argument(
        "--entropy", type=float, default=None, help="Entropy regularization coefficient"
    )
    parser.add_argument(
        "--lstm-out", type=int, default=None, help="LSTM/attention output size"
    )
    parser.add_argument(
        "--test-episodes", type=int, default=10, help="Number of test episodes"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--results-dir", type=str, default="Result", help="Results directory suffix"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from latest checkpoint"
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true")

    return parser.parse_args()


def run_train(args):
    import torch
    import torch.multiprocessing as mp

    from config import get_hitmac_training_config
    from environments.environment import DSNEnv
    from hitmac.models import build_model
    from hitmac.shared_optim import SharedAdam
    from hitmac.test import test
    from hitmac.train import train

    config = get_hitmac_training_config(args.scenario)

    if args.workers is None:
        args.workers = config.get("workers", 4)
    if args.max_steps is None:
        args.max_step = config.get(
            "max_step", 500000
        )  # Note: config uses max_step, args uses max_steps
    else:
        args.max_step = args.max_steps

    if args.lr is None:
        args.lr = config.get("lr", 0.0005)
    if args.gamma is None:
        args.gamma = config.get("gamma", 0.9)
    if args.entropy is None:
        args.entropy = config.get("entropy", 0.01)
    if args.lstm_out is None:
        args.lstm_out = config.get("lstm_out", 128)

    print(f"HiT-MAC Training Config:")
    print(f"  Max Steps: {args.max_step} (approx {args.max_step // 100} episodes)")
    print(f"  Workers: {args.workers}")
    print(f"  LR: {args.lr}")

    os.environ["OMP_NUM_THREADS"] = "1"

    use_cuda = torch.cuda.is_available() if args.device != "cpu" else False
    if use_cuda:
        device = torch.device("cuda")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
        torch.manual_seed(args.seed)

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    env = DSNEnv(scenario=args.scenario)
    print(f"Building model for {env.n_sensors} agents, {env.n_targets} targets...")

    model_args = ModelArgs("single-att", args.lstm_out)
    shared_model = build_model(
        env.n_sensors, env.n_targets, env.n_actions, model_args, device
    )
    shared_model = shared_model.to(device)
    shared_model.share_memory()
    env.close()

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr, amsgrad=True)
    optimizer.share_memory()

    hitmac_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(hitmac_dir, "checkpoints")
    results_dir = os.path.join("Result", f"Scenario{args.scenario}")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    start_step = 0
    history_to_restore = {}
    if args.resume:
        import numpy as np

        latest_path = os.path.join(checkpoints_dir, "latest.pt")
        if os.path.exists(latest_path):
            print(f"Resuming from checkpoint: {latest_path}")
            resume_ckpt = torch.load(latest_path, map_location=device)
            shared_model.load_state_dict(resume_ckpt["model"])
            if resume_ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
            start_step = resume_ckpt.get("step", 0)
            # Load training history from .npz (not from .pt — history is not stored there)
            npz_path = os.path.join(checkpoints_dir, "training_history.npz")
            if os.path.exists(npz_path):
                npz = np.load(npz_path)
                history_to_restore = {
                    "episode_rewards": npz["episode_rewards"].tolist(),
                    "coverage_rates": npz["coverage_rates"].tolist(),
                    "episode_durations": npz["episode_durations"].tolist(),
                }
            print(f"  Resuming from step {start_step} / {args.max_step}")
        else:
            print(
                f"Warning: --resume specified but no checkpoint found at {latest_path}"
            )

    # Create args object for workers (module-level class for pickle)
    train_args = TrainArgs(
        env=f"Pose-v{args.scenario}",
        scenario=args.scenario,
        seed=args.seed,
        workers=args.workers,
        num_steps=20,
        max_step=args.max_step,
        test_eps=args.test_episodes,
        gamma=args.gamma,
        tau=1.0,
        entropy=args.entropy,
        lstm_out=args.lstm_out,
        lr=args.lr,
        log_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
        results_dir=results_dir,
        gpu_ids=[0] if use_cuda else [-1],
        model="single-att",
        norm_reward=config.get("norm_reward", False),
        render=args.render,
        render_save=False,
        sleep_time=0,
        fix=False,
        train_mode=1,
        optimizer="Adam",
        start_step=start_step,
        save_interval=config.get("save_interval", 500000),
    )

    print(f"\nStarting training with {args.workers} workers...")
    print(f"Checkpoints: {checkpoints_dir}")
    print(f"Results: {results_dir}")

    processes = []
    manager = mp.Manager()
    train_modes = manager.list(["" for _ in range(args.workers)])
    n_iters = manager.list([0 for _ in range(args.workers)])
    episode_rewards = manager.list()
    coverage_rates = manager.list()
    episode_durations = manager.list()

    episode_rewards.extend(history_to_restore.get("episode_rewards", []))
    coverage_rates.extend(history_to_restore.get("coverage_rates", []))
    episode_durations.extend(history_to_restore.get("episode_durations", []))

    p = mp.Process(
        target=test,
        args=(
            train_args,
            shared_model,
            optimizer,
            train_modes,
            n_iters,
            episode_rewards,
            coverage_rates,
            episode_durations,
            train_args.start_step,
        ),
    )
    p.start()
    processes.append(p)

    for rank in range(args.workers):
        p = mp.Process(
            target=train,
            args=(
                rank,
                train_args,
                shared_model,
                optimizer,
                train_modes,
                n_iters,
                episode_rewards,
                coverage_rates,
                episode_durations,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nTraining complete!")
    print(f"Best model: {os.path.join(checkpoints_dir, 'best.pt')}")
    print(f"Training history: {os.path.join(checkpoints_dir, 'training_history.npz')}")

    return {"checkpoints_dir": checkpoints_dir, "results_dir": results_dir}


def run_test(args):
    import numpy as np
    import torch

    from config import get_hitmac_training_config
    from environments.environment import DSNEnv
    from hitmac.models import build_model

    config = get_hitmac_training_config(args.scenario)
    if args.lstm_out is None:
        args.lstm_out = config.get("lstm_out", 128)

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hitmac_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(hitmac_dir, "checkpoints")

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

    env = DSNEnv(scenario=args.scenario)
    model_args = ModelArgs("single-att", args.lstm_out)
    model = build_model(env.n_sensors, env.n_targets, env.n_actions, model_args, device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    print(f"\nRunning {args.test_episodes} test episodes...")

    rewards = []
    coverages = []

    for ep in range(args.test_episodes):
        obs_list, info = env.reset()
        obs = np.array(obs_list, dtype=np.float32).reshape(
            env.n_sensors, env.n_targets, 4
        )
        state = torch.from_numpy(obs).float()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                value, actions, _, _ = model(state)

            actions_list = actions.cpu().tolist() if isinstance(actions, torch.Tensor) else actions.tolist()
            obs_list, reward, terminated, truncated, info = env.step(actions_list)
            done = terminated or truncated
            obs = np.array(obs_list, dtype=np.float32).reshape(
                env.n_sensors, env.n_targets, 4
            )
            state = torch.from_numpy(obs).float()
            episode_reward += reward

        coverage = info.get("coverage_rate", 0)
        rewards.append(episode_reward)
        coverages.append(coverage)

        if args.render or len(rewards) <= 3:
            print(
                f"  Episode {ep+1}: Reward={episode_reward:.2f}, Coverage={coverage*100:.1f}%"
            )

    env.close()

    print(f"\n{'='*40}")
    print(f"Results over {args.test_episodes} episodes:")
    print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(
        f"  Mean Coverage: {np.mean(coverages)*100:.1f}% ± {np.std(coverages)*100:.1f}%"
    )
    print(f"{'='*40}")

    return {"rewards": rewards, "coverages": coverages}


def main():
    args = parse_args()

    print("=" * 60)
    print("HiT-MAC: Hierarchical Twin-Actor Multi-Agent Coordination")
    print("Applied to Directional Sensor Network")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Scenario: {args.scenario}")
    print(f"Device: {args.device or 'auto'}")
    print("=" * 60)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)


if __name__ == "__main__":
    main()
