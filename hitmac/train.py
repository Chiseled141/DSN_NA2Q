"""
=============================================================================
HiT-MAC TRAINING WORKER
=============================================================================

WHAT IS THIS FILE?
------------------
Implements a single A3C training worker process. Multiple workers run in
parallel, each with their own environment copy, collecting trajectories
and syncing gradients to a shared model.

A3C PARALLELIZATION:
--------------------
    ┌─────────────────────────────────────────────────────┐
    │                   SHARED MODEL                      │
    │                 (in CPU memory)                     │
    └─────────┬──────────────┬──────────────┬─────────────┘
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │ Worker 0  │  │ Worker 1  │  │ Worker N  │
        │  (GPU 0)  │  │  (GPU 1)  │  │  (CPU)    │
        │           │  │           │  │           │
        │ Local Env │  │ Local Env │  │ Local Env │
        └───────────┘  └───────────┘  └───────────┘

Each worker:
1. Syncs weights from shared model
2. Collects trajectory in local environment
3. Computes gradients locally
4. Applies gradients to shared model (async)
"""

import time

import numpy as np
import torch
import torch.optim as optim

try:
    from setproctitle import setproctitle as ptitle
except ImportError:

    def ptitle(name):
        pass


from environments.environment import DSNEnv
from hitmac.models import build_model
from hitmac.player import Agent

# Optional tensorboardX for logging
try:
    from tensorboardX import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def train(
    rank,
    args,
    shared_model,
    optimizer,
    train_modes,
    n_iters,
    episode_rewards=None,
    coverage_rates=None,
    episode_durations=None,
    average_gains=None,
    env=None,
):
    """Training worker process for A3C.

    Args:
        rank: Worker ID
        args: Training arguments
        shared_model: Shared model in CPU memory
        optimizer: Shared optimizer
        train_modes: Shared list for training mode coordination
        n_iters: Shared list for iteration tracking
        episode_rewards: Shared list for episode rewards (optional)
        coverage_rates: Shared list for coverage rates (optional)
        env: Optional pre-created environment
    """
    n_iter = 0

    # Disable TensorBoard to keep Result folder clean
    writer = None

    ptitle("Training Agent: {}".format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    training_mode = args.train_mode

    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
        device = torch.device("cuda:" + str(gpu_id))
        if len(args.gpu_ids) > 1:
            device_share = torch.device("cpu")
        else:
            device_share = torch.device("cuda:" + str(args.gpu_ids[-1]))
    else:
        device = device_share = torch.device("cpu")

    # Create DSNEnv directly
    if env is None:
        env = DSNEnv(scenario=getattr(args, "scenario", 1))

    params = list(shared_model.parameters())
    if optimizer is None:
        if args.optimizer == "RMSprop":
            optimizer = optim.RMSprop(params, lr=args.lr)
        elif args.optimizer == "Adam":
            optimizer = optim.Adam(params, lr=args.lr)

    # Seed via reset
    seed = args.seed if args.fix else (rank % (args.seed + 1))

    player = Agent(None, env, args, None, device)
    player.rank = rank
    player.gpu_id = gpu_id

    # Build local model
    player.model = build_model(
        env.n_sensors, env.n_targets, env.n_actions, args, device
    )
    player.model = player.model.to(device)
    player.model.train()

    player.reset()
    episode_start_time = time.time()
    reward_sum = torch.zeros(player.num_agents).to(device)
    reward_sum_org = np.zeros(player.num_agents)
    ave_reward = np.zeros(2)
    ave_reward_longterm = np.zeros(2)
    count_eps = 0

    while True:
        # Sync to shared model
        player.model.load_state_dict(shared_model.state_dict())

        if player.done:
            # Log episode metrics to shared lists
            if episode_rewards is not None:
                episode_rewards.append(reward_sum_org[0])
            if coverage_rates is not None:
                if player.info:
                    coverage_rates.append(player.info.get("coverage_rate", 0.0))
                else:
                    coverage_rates.append(0.0)

            if episode_durations is not None:
                episode_durations.append(time.time() - episode_start_time)

            if average_gains is not None and player.info:
                coverage = player.info.get("coverage_rate", 0.0)
                cost = player.eps_rotation_count / max(player.eps_len * player.num_agents, 1)
                ag = coverage / (cost + 1e-8) if cost > 0 else 0.0
                average_gains.append(ag)

            player.reset()
            episode_start_time = time.time()
            reward_sum = torch.zeros(player.num_agents).to(device)
            reward_sum_org = np.zeros(player.num_agents)
            count_eps += 1

        player.update_rnn_hidden()
        t0 = time.time()

        # Collect trajectory
        for s_i in range(args.num_steps):
            player.action_train()
            reward_sum += player.reward
            reward_sum_org += player.reward_org

            if player.done:
                if writer is not None:
                    for i, r_i in enumerate(reward_sum):
                        writer.add_scalar("train/reward_" + str(i), r_i, player.n_steps)
                        if args.norm_reward:
                            writer.add_scalar(
                                "train/reward_org_" + str(i),
                                reward_sum_org[i].sum(),
                                player.n_steps,
                            )
                break

        fps = s_i / (time.time() - t0 + 1e-8)

        # Optimize
        policy_loss, value_loss, entropies = player.optimize(
            params, optimizer, shared_model, training_mode, device_share
        )

        if writer is not None:
            writer.add_scalar(
                "train/policy_loss_sum", policy_loss.sum(), player.n_steps
            )
            writer.add_scalar("train/value_loss_sum", value_loss.sum(), player.n_steps)
            writer.add_scalar("train/entropies_sum", entropies.sum(), player.n_steps)
            writer.add_scalar(
                "train/ave_reward",
                ave_reward[0] - ave_reward_longterm[0],
                player.n_steps,
            )
            writer.add_scalar("train/mode", training_mode, player.n_steps)
            writer.add_scalar("train/fps", fps, player.n_steps)

        n_iter += 1
        n_iters[rank] = n_iter

        # Check for termination signal
        if train_modes[rank] == -100:
            env.close()
            break
