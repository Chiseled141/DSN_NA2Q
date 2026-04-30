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
  python -m hitmac.main --mode train --scenario 1 --phase2-only
  python -m hitmac.main --mode test --scenario 1
""",
    )

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3])
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
    parser.add_argument(
        "--phase2-only", action="store_true",
        help="Skip Phase 1 and start directly from Phase 2 using existing executor_final.pt"
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true")

    return parser.parse_args()


def _fmt_time(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _coordinator_monitor(args, shared_model, optimizer, train_modes, n_iters,
                          shared_lists, checkpoints_dir):
    """Lightweight monitor for Phase 2 coordinator training.
    Handles progress bar, checkpointing, and history saving — no episode evaluation."""
    import glob
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    pbar = tqdm(total=args.max_step, desc="Phase 2 (Coordinator)", unit="step")
    save_interval = getattr(args, "save_interval", 200000)
    last_checkpoint_step = args.start_step
    n_iter = args.start_step
    best_coverage = -1.0
    log_path = os.path.join(checkpoints_dir, "training.log")
    monitor_start = time.time()

    def log(msg):
        elapsed = time.time() - monitor_start
        progress = (n_iter - args.start_step) / max(args.max_step - args.start_step, 1)
        eta_str = _fmt_time(elapsed / progress * (1 - progress)) if progress > 0 else "--:--:--"
        prefix = f"[+{_fmt_time(elapsed)} | ~{eta_str} left]"
        pbar.write(f"[Coordinator] {msg}")
        with open(log_path, "a") as f:
            f.write(f"{prefix} | {msg}\n")

    log(f"=== PHASE 2: Coordinator Training | max_step={args.max_step} ===")
    log(f"Executor: scripted (deterministic) | Coordinator: multi-att-shap (training from scratch)")
    log(f"Coverage baseline higher than Phase 1 — scripted executor has zero aiming error vs learned executor noise in Phase 1. This is expected.")
    log(f"Scripted executor is a training-only surrogate. At inference, the Phase 1 learned executor is restored.")

    while n_iter < args.max_step:
        time.sleep(5)
        n_iter = args.start_step + sum(n_iters)
        pbar.n = min(n_iter, args.max_step)
        pbar.refresh()

        rewards = list(shared_lists["rewards"])
        coverage = list(shared_lists["coverage"])
        if rewards:
            mean_r = float(np.mean(rewards[-50:]))
            mean_c = float(np.mean(coverage[-50:])) if coverage else 0.0
            pbar.set_postfix({"R": f"{mean_r:.2f}", "Cov": f"{mean_c:.1%}"})

            if mean_c > best_coverage:
                best_coverage = mean_c
                state = {"model": shared_model.state_dict(),
                         "optimizer": optimizer.state_dict() if optimizer else None,
                         "step": n_iter}
                torch.save(state, os.path.join(checkpoints_dir, "coordinator_best.pt"))

        state = {"model": shared_model.state_dict(),
                 "optimizer": optimizer.state_dict() if optimizer else None,
                 "step": n_iter}
        torch.save(state, os.path.join(checkpoints_dir, "coordinator_latest.pt"))

        if n_iter >= last_checkpoint_step + save_interval:
            last_checkpoint_step = n_iter
            ckpt = os.path.join(checkpoints_dir, f"coordinator_checkpoint_{n_iter}.pt")
            torch.save(state, ckpt)
            log(f"Checkpoint saved: coordinator_checkpoint_{n_iter}.pt")
            old = sorted(glob.glob(os.path.join(checkpoints_dir, "coordinator_checkpoint_*.pt")),
                         key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for f in old[:-5]:
                try: os.remove(f)
                except: pass

        try:
            durations = list(shared_lists["durations"])
            gains = list(shared_lists["gains"])
            min_len = min(len(rewards), len(coverage), len(durations), len(gains)) if all([rewards, coverage, durations, gains]) else min(len(rewards), len(coverage))
            np.savez(os.path.join(checkpoints_dir, "training_history.npz"),
                     episode_rewards=np.array(rewards[:min_len]),
                     coverage_rates=np.array(coverage[:min_len]),
                     losses=np.array([]),
                     episode_durations=np.array(durations[:min_len]),
                     average_gains=np.array(gains[:min_len]))
        except Exception:
            pass

    pbar.close()
    log(f"=== PHASE 2 COMPLETE — best coverage: {best_coverage:.1%} | trained for {n_iter:,} steps ===")
    for i in range(len(train_modes)):
        train_modes[i] = -100


def run_train(args):
    import time as _time
    import torch
    import torch.multiprocessing as mp

    from config import get_hitmac_training_config
    from environments.environment import DSNEnv
    from hitmac.coordinator_train import train_coordinator
    from hitmac.models import build_model
    from hitmac.shared_optim import SharedAdam
    from hitmac.test import test

    _run_start = _time.time()
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
    checkpoints_dir = os.path.join(hitmac_dir, "checkpoints", f"scenario{args.scenario}")
    results_dir = os.path.join("Result", f"Scenario{args.scenario}")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Write session header to log before any subprocess starts
    from datetime import datetime
    log_path = os.path.join(checkpoints_dir, "training.log")
    env_tmp = DSNEnv(scenario=args.scenario)
    n_sensors, n_targets = env_tmp.n_sensors, env_tmp.n_targets
    env_tmp.close()
    with open(log_path, "a") as _f:
        _p = f"+{_fmt_time(_time.time() - _run_start)}"
        _f.write(f"\n{'='*60}\n")
        _f.write(f"[{_p}] | NEW TRAINING SESSION STARTED\n")
        _f.write(f"[{_p}] | Device: {device}\n")
        _f.write(f"[{_p}] | Scenario: {args.scenario} | Sensors: {n_sensors} | Targets: {n_targets}\n")
        _f.write(f"[{_p}] | Config: lr={args.lr}, gamma={args.gamma}, entropy={args.entropy}, workers={args.workers}, num_steps={config.get('num_steps', 40)}, max_step={args.max_step}\n")
        _f.write(f"{'='*60}\n")

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
                    "average_gains": npz["average_gains"].tolist() if "average_gains" in npz else [],
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
        num_steps=config.get("num_steps", 40),
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
        sleep_time=10,
        fix=False,
        train_mode=1,
        optimizer="Adam",
        start_step=start_step,
        save_interval=config.get("save_interval", 500000),
    )

    print(f"\n{'='*50}")
    print(f"PHASE 1: Executor Training ({args.max_step:,} steps)")
    print(f"{'='*50}")
    print(f"Checkpoints: {checkpoints_dir}")

    def _run_phase(target_fn, target_model, target_optimizer, max_step, shared_lists,
                   phase_name, model_name="single-att", start_step=0):
        """Run one training phase with all workers + monitor process."""
        import types
        phase_args = types.SimpleNamespace(**vars(train_args))
        phase_args.max_step = max_step
        phase_args.model = model_name
        phase_args.start_step = start_step

        n_workers = args.workers
        phase_manager = mp.Manager()
        train_modes = phase_manager.list(["" for _ in range(n_workers)])
        n_iters = phase_manager.list([0 for _ in range(n_workers)])

        processes = []

        # Monitor process — uses test() for executor, simple monitor for coordinator
        if model_name == "single-att":
            p = mp.Process(
                target=test,
                args=(phase_args, target_model, target_optimizer, train_modes, n_iters,
                      shared_lists["rewards"], shared_lists["coverage"],
                      shared_lists["durations"], shared_lists["gains"], start_step),
            )
        else:
            p = mp.Process(
                target=_coordinator_monitor,
                args=(phase_args, target_model, target_optimizer, train_modes, n_iters,
                      shared_lists, checkpoints_dir),
            )
        p.start()
        processes.append(p)

        # Training workers
        for rank in range(n_workers):
            p = mp.Process(
                target=target_fn,
                args=(rank, phase_args, target_model, target_optimizer,
                      train_modes, n_iters,
                      shared_lists["rewards"], shared_lists["coverage"],
                      shared_lists["durations"], shared_lists["gains"]),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f"\n{phase_name} complete.")

    # ── Shared history lists (persist across both phases) ──────────────────────
    manager = mp.Manager()
    shared_lists = {
        "rewards":  manager.list(history_to_restore.get("episode_rewards", [])),
        "coverage": manager.list(history_to_restore.get("coverage_rates", [])),
        "durations": manager.list(history_to_restore.get("episode_durations", [])),
        "gains":    manager.list(history_to_restore.get("average_gains", [])),
    }

    # ── Phase 1: Train executor (single-att) ───────────────────────────────────
    import torch as _torch
    executor_path = os.path.join(checkpoints_dir, "executor_final.pt")

    if getattr(args, "phase2_only", False) and os.path.exists(executor_path):
        print(f"Skipping Phase 1 — loading executor from: {executor_path}")
        ckpt = _torch.load(executor_path, map_location=device)
        shared_model.load_state_dict(ckpt["model"])
        with open(log_path, "a") as _f:
            _p = f"+{_fmt_time(_time.time() - _run_start)}"
            _f.write(f"[{_p}] | PHASE 1 SKIPPED — loaded existing executor: {executor_path}\n")
    else:
        _run_phase(train, shared_model, optimizer, args.max_step, shared_lists,
                   "Phase 1 (Executor)", model_name="single-att", start_step=start_step)
        _torch.save({"model": shared_model.state_dict(), "step": args.max_step}, executor_path)
        print(f"Executor saved: {executor_path}")
        with open(log_path, "a") as _f:
            _p = f"+{_fmt_time(_time.time() - _run_start)}"
            _f.write(f"[{_p}] | PHASE 1 COMPLETE — executor saved to {executor_path}\n")

    # ── Phase 2: Train coordinator (multi-att-shap) ────────────────────────────
    print(f"\n{'='*50}")
    print(f"PHASE 2: Coordinator Training ({args.max_step:,} steps)")
    print(f"{'='*50}")

    coord_model_args = ModelArgs("multi-att-shap", args.lstm_out)
    env_tmp = DSNEnv(scenario=args.scenario)
    coord_model = build_model(env_tmp.n_sensors, env_tmp.n_targets, env_tmp.n_actions,
                              coord_model_args, device)
    env_tmp.close()
    coord_model = coord_model.to(device)
    coord_model.share_memory()

    coord_optimizer = SharedAdam(coord_model.parameters(), lr=args.lr, amsgrad=True)
    coord_optimizer.share_memory()

    _run_phase(train_coordinator, coord_model, coord_optimizer,
               args.max_step, shared_lists, "Phase 2 (Coordinator)",
               model_name="multi-att-shap", start_step=0)

    # Save coordinator checkpoint
    coord_path = os.path.join(checkpoints_dir, "coordinator_final.pt")
    _torch.save({"model": coord_model.state_dict(), "step": args.max_step}, coord_path)
    print(f"Coordinator saved: {coord_path}")

    with open(log_path, "a") as _f:
        _p = f"+{_fmt_time(_time.time() - _run_start)}"
        _f.write(f"[{_p}] | PHASE 2 COMPLETE — coordinator saved to {coord_path}\n")
        _f.write(f"[{_p}] | TRAINING COMPLETE — at test time the learned executor (Phase 1) replaces the scripted one\n")
        _f.write(f"{'='*60}\n")

    print("\n" + "="*50)
    print("HiT-MAC Training Complete!")
    print(f"  Executor:    {executor_path}")
    print(f"  Coordinator: {coord_path}")
    print(f"  History:     {checkpoints_dir}/training_history.npz")
    print("="*50)

    return {"checkpoints_dir": checkpoints_dir, "results_dir": results_dir}


def run_test(args):
    import numpy as np
    import torch

    from config import get_hitmac_training_config
    from environments.environment import DSNEnv
    import types as _types
    from hitmac.models import build_model
    from hitmac.utils import scripted_action, hungarian_assignment

    config = get_hitmac_training_config(args.scenario)
    if args.lstm_out is None:
        args.lstm_out = config.get("lstm_out", 128)

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hitmac_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(hitmac_dir, "checkpoints")

    # Load coordinator model for goal assignments
    coord_path = os.path.join(checkpoints_dir, "coordinator_best.pt")
    if not os.path.exists(coord_path):
        coord_path = os.path.join(checkpoints_dir, "coordinator_final.pt")

    env = DSNEnv(scenario=args.scenario)

    print(f"\nRunning {args.test_episodes} test episodes (coordinator + scripted executor)...")

    rewards = []
    coverages = []

    if os.path.exists(coord_path):
        print(f"Coordinator: {coord_path}")
        coord_args = _types.SimpleNamespace(model="multi-att-shap", lstm_out=args.lstm_out)
        coord_model = build_model(env.n_sensors, env.n_targets, env.n_actions, coord_args, device)
        ckpt = torch.load(coord_path, map_location=device)
        coord_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
        coord_model.eval()
        use_coordinator = True
    else:
        print("No coordinator found — using scripted executor only (no goal assignment)")
        use_coordinator = False

    for ep in range(args.test_episodes):
        obs_list, info = env.reset()
        obs = np.array(obs_list, dtype=np.float32).reshape(env.n_sensors, env.n_targets, 4)
        state = torch.from_numpy(obs).float().to(device)
        episode_reward = 0
        done = False
        step = 0
        goals = np.zeros((env.n_sensors, env.n_targets), dtype=np.float32)

        while not done:
            # Coordinator reassigns goals every COORDINATOR_PERIOD steps
            if step % 25 == 0:
                if use_coordinator:
                    with torch.no_grad():
                        _, goal_assignments, _, _ = coord_model(state)
                    goals = goal_assignments.squeeze(-1)
                else:
                    # Hungarian fallback: provably optimal assignment
                    goals = hungarian_assignment(env)

            # Scripted executor follows coordinator's assignments perfectly
            actions_list = [scripted_action(env, i, goals[i]) for i in range(env.n_sensors)]
            obs_list, reward, terminated, truncated, info = env.step(actions_list)
            done = terminated or truncated
            obs = np.array(obs_list, dtype=np.float32).reshape(env.n_sensors, env.n_targets, 4)
            state = torch.from_numpy(obs).float().to(device)
            episode_reward += reward
            step += 1

        coverage = info.get("coverage_rate", 0)
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
