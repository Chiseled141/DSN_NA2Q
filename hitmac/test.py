"""
=============================================================================
HiT-MAC TESTING PROCESS
=============================================================================

WHAT IS THIS FILE?
------------------
Implements the A3C test/evaluation process. Runs in parallel with training
workers, periodically evaluating the shared model and saving checkpoints.

OUTPUT:
-------
- hitmac/checkpoints/best.pt              - Best model checkpoint
- hitmac/checkpoints/latest.pt            - Latest model checkpoint
- hitmac/checkpoints/training_history.npz - Training metrics

KEY RESPONSIBILITIES:
---------------------
1. Load weights from shared model
2. Run evaluation episodes
3. Track best performance
4. Save model checkpoints and training history
"""

import glob
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from environments.environment import DSNEnv


def _fmt_time(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
from hitmac.models import build_model
from hitmac.player import Agent

# Optional dependencies
try:
    from tensorboardX import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    try:
        from setproctitle import setproctitle as ptitle
    except ImportError:

        def ptitle(name):
            pass

except ImportError:

    def ptitle(title):
        pass


def test(
    args,
    shared_model,
    optimizer,
    train_modes,
    n_iters,
    episode_rewards=None,
    coverage_rates=None,
    episode_durations=None,
    average_gains=None,
    start_step=0,
):
    """Test process for A3C - evaluates and saves best models.

    Args:
        args: Test arguments (must include checkpoints_dir)
        shared_model: Shared model to evaluate
        optimizer: Optimizer (for checkpoint saving)
        train_modes: Shared list for coordination
        n_iters: Shared list tracking training iterations
        episode_rewards: Shared list for episode rewards (optional)
        coverage_rates: Shared list for coverage rates (optional)
    """
    ptitle("Test Agent")
    n_iter = start_step

    # Disable TensorBoard to keep Result folder clean
    writer = None

    gpu_id = args.gpu_ids[-1]

    # Initialize tqdm progress bar
    pbar = tqdm(total=args.max_step, desc="HiT-MAC Training", unit="step")

    # Get checkpoints directory (inside hitmac/)
    checkpoints_dir = getattr(
        args, "checkpoints_dir", os.path.join(os.path.dirname(__file__), "checkpoints")
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Log file — saved to hitmac/checkpoints/training.log
    log_path = os.path.join(checkpoints_dir, "training.log")

    def write_log(msg):
        elapsed = time.time() - start_time
        progress = n_iter / max(args.max_step, 1)
        eta_str = _fmt_time(elapsed / progress * (1 - progress)) if progress > 0 else "--:--:--"
        prefix = f"[+{_fmt_time(elapsed)} | ~{eta_str} left]"
        with open(log_path, "a") as f:
            f.write(f"{prefix} | {msg}\n")

    # Use pbar.write for logging to avoid breaking the bar; also write to file
    def log_info(msg):
        pbar.write(f"[HiT-MAC] {msg}")
        write_log(msg)

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + str(gpu_id))
    else:
        device = torch.device("cpu")

    write_log(f"=== PHASE 1: Executor Training (learned policy, max_step={args.max_step}) ===")
    write_log(f"WHAT IS HAPPENING: Each agent learns to physically aim its sensor at assigned targets")
    write_log(f"Device: {device} | lr={args.lr} | gamma={args.gamma} | entropy={args.entropy} | workers={getattr(args, 'workers', '?')} | num_steps={args.num_steps}")

    # Create DSNEnv directly
    env = DSNEnv(scenario=getattr(args, "scenario", 1), seed=args.seed)
    start_time = time.time()
    count_eps = 0

    # Use shared lists if available (for training history)
    # If not provided (standalone test), use local lists
    if episode_rewards is None:
        episode_rewards = []
    if coverage_rates is None:
        coverage_rates = []

    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.model = build_model(
        env.n_sensors, env.n_targets, env.n_actions, args, device
    )
    player.model = player.model.to(device)
    player.model.eval()
    max_score = -100
    save_interval = getattr(args, "save_interval", 500000)
    last_checkpoint_step = start_step

    while n_iter < args.max_step:
        player.reset()
        reward_sum = np.zeros(player.num_agents)
        reward_sum_ep = np.zeros(player.num_agents)
        len_sum = 0

        AG = 0
        reward_sum_list = []
        coverage_sum_list = []
        fps_all = []

        # Sync model
        player.model.load_state_dict(shared_model.state_dict())
        player.model.eval()

        # Test loop
        for i_eps in range(args.test_eps):
            player.reset()
            reward_sum_ep = np.zeros(player.num_agents)
            fps_counter = 0
            t0 = time.time()

            while True:
                player.action_test()
                reward_sum_ep += player.reward_org
                fps_counter += 1

                if player.done:
                    # Metrics
                    episode_coverage = player.info.get("coverage_rate", 0)
                    AG += episode_coverage

                    # Accumulate
                    reward_sum += reward_sum_ep
                    reward_sum_list.append(reward_sum_ep[0])
                    coverage_sum_list.append(episode_coverage)

                    len_sum += player.eps_len
                    fps = fps_counter / (time.time() - t0 + 1e-8)

                    n_iter = start_step + sum(n_iters)

                    # Update progress bar
                    pbar.n = n_iter
                    pbar.refresh()

                    # Update description with simple stats
                    pbar.set_postfix(
                        {
                            "R": f"{np.mean(reward_sum_list) if reward_sum_list else 0:.2f}",
                            "Cov": f"{np.mean(coverage_sum_list) if coverage_sum_list else 0:.1%}",
                        }
                    )

                    fps_all.append(fps)
                    break

        # Compute statistics
        ave_AG = AG / args.test_eps
        ave_reward_sum = reward_sum / args.test_eps
        len_mean = len_sum / args.test_eps
        reward_step = reward_sum / (len_sum + 1e-8)
        mean_reward = np.mean(reward_sum_list)
        std_reward = np.std(reward_sum_list)
        mean_coverage = np.mean(coverage_sum_list)

        # No longer log own test metrics to training history
        # History is now populated by workers!

        # Simplified logging similar to NA2Q
        log_info(
            f"Step {n_iter}/{args.max_step} | "
            f"Reward: {mean_reward:.2f} | "
            f"Coverage: {mean_coverage:.1%}"
        )

        # Log to TensorBoard if enabled (for test metrics only)
        if writer is not None:
            writer.add_scalar("test/mean_reward", mean_reward, n_iter)
            writer.add_scalar("test/mean_coverage", mean_coverage, n_iter)

        # Save model to hitmac/checkpoints/
        # Training history is saved separately in .npz — do NOT embed it here
        # (embedding it bloats the .pt file to 100s of MB after long runs)
        state_to_save = {
            "model": player.model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "step": n_iter,
        }
        torch.save(state_to_save, os.path.join(checkpoints_dir, "latest.pt"))

        # Periodic checkpoint (every save_interval steps)
        if n_iter >= last_checkpoint_step + save_interval:
            last_checkpoint_step = n_iter
            ckpt_path = os.path.join(checkpoints_dir, f"checkpoint_{n_iter}.pt")
            torch.save(state_to_save, ckpt_path)
            log_info(f"Checkpoint saved: checkpoint_{n_iter}.pt")

            # Keep only the last 10 periodic checkpoints
            old_ckpts = sorted(
                glob.glob(os.path.join(checkpoints_dir, "checkpoint_*.pt")),
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            for f in old_ckpts[:-10]:
                try:
                    os.remove(f)
                except Exception:
                    pass

        # Save best model if improved
        if ave_reward_sum[0] >= max_score:
            prev_score = max_score
            max_score = ave_reward_sum[0]
            torch.save(state_to_save, os.path.join(checkpoints_dir, "best.pt"))
            delta = f" (+{max_score - prev_score:.2f})" if prev_score > -100 else " (first best)"
            log_info(
                f"New best model saved (reward: {ave_reward_sum[0]:.2f}{delta}, coverage: {mean_coverage:.1%})"
            )

        # Save training history as .npz (matches NA2Q format)
        # Use metrics from the SHARED lists collected from WORKERS
        try:
            # Handle potential concurrency issues with a quick snapshot
            current_rewards = list(episode_rewards)
            current_coverage = list(coverage_rates)
            current_durations = list(episode_durations) if episode_durations else []
            current_gains = list(average_gains) if average_gains else []

            # Align lengths (workers append asynchronously, so lengths may differ)
            lengths = [len(current_rewards), len(current_coverage)]
            if current_durations:
                lengths.append(len(current_durations))
            if current_gains:
                lengths.append(len(current_gains))
            min_len = min(lengths)
            current_rewards = current_rewards[:min_len]
            current_coverage = current_coverage[:min_len]
            current_durations = current_durations[:min_len]
            current_gains = current_gains[:min_len]

            np.savez(
                os.path.join(checkpoints_dir, "training_history.npz"),
                episode_rewards=np.array(current_rewards),
                coverage_rates=np.array(current_coverage),
                losses=np.array([]),
                episode_durations=np.array(current_durations),
                average_gains=np.array(current_gains),
            )
        except Exception as e:
            print(f"Warning: Failed to save history: {e}")

        time.sleep(args.sleep_time)

    pbar.close()
    log_info(f"=== PHASE 1 COMPLETE — best reward: {max_score:.2f} | trained for {n_iter:,} steps ===")
    log_info(f"NOTE: Executor (learned sensor control) is now frozen. Phase 2 will train the coordinator using a scripted executor as a temporary stand-in.")
    # Signal all workers to stop (they check train_modes[rank] == -100)
    for i in range(len(train_modes)):
        train_modes[i] = -100
    if writer is not None:
        writer.close()
