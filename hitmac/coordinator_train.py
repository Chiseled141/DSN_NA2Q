"""
HiT-MAC Coordinator Training Worker (Phase 2).

The coordinator (A3C_Multi) is trained after the executor.
It uses a scripted executor for primitive actions (75 FPS vs 25 FPS with learned executor).

Paper Section 3.3:
- Every k=10 steps, coordinator assigns goals to executors
- Scripted executor takes primitive actions to track assigned targets
- Team reward = coverage_rate (Eq. 3), or -0.1 if nothing covered
"""

import time

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

from environments.environment import DSNEnv
from hitmac.models import build_model
from hitmac.player import ensure_shared_grads
from hitmac.utils import scripted_action

try:
    from setproctitle import setproctitle as ptitle
except ImportError:
    def ptitle(name): pass

COORDINATOR_PERIOD = 10  # k steps between coordinator calls (paper Section 3.3)


def train_coordinator(
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
):
    """Coordinator training worker (A3C_Multi with scripted executor)."""
    ptitle(f"Coordinator Worker: {rank}")
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank + 100)

    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
    device_share = torch.device("cpu") if len(args.gpu_ids) > 1 else device

    env = DSNEnv(scenario=getattr(args, "scenario", 1))
    n_sensors = env.n_sensors
    n_targets = env.n_targets

    # Build coordinator model (A3C_Multi with Shapley critic)
    import types
    coord_args = types.SimpleNamespace(model="multi-att-shap", lstm_out=args.lstm_out)
    local_model = build_model(n_sensors, n_targets, env.n_actions, coord_args, device)
    local_model.to(device)
    local_model.train()

    params = list(shared_model.parameters())
    if optimizer is None:
        optimizer = optim.Adam(params, lr=args.lr)

    n_iter = 0
    obs_list, _ = env.reset()
    state = _obs_to_tensor(obs_list, n_sensors, n_targets, device)

    episode_start_time = time.time()
    eps_rotation_count = 0
    eps_len = 0
    done = False
    _last_coverage = 0.0
    reward_sum_ep = 0.0

    # Trajectory buffers
    values, log_probs, rewards, entropies = [], [], [], []

    while True:
        local_model.load_state_dict(shared_model.state_dict())

        if done:
            coverage = _last_coverage
            if episode_rewards is not None:
                episode_rewards.append(reward_sum_ep)
            if coverage_rates is not None:
                coverage_rates.append(coverage)
            if episode_durations is not None:
                episode_durations.append(time.time() - episode_start_time)
            if average_gains is not None:
                cost = eps_rotation_count / max(eps_len * n_sensors, 1)
                ag = coverage / (cost + 1e-8) if cost > 0 else 0.0
                average_gains.append(ag)

            obs_list, _ = env.reset()
            state = _obs_to_tensor(obs_list, n_sensors, n_targets, device)
            episode_start_time = time.time()
            eps_rotation_count = 0
            eps_len = 0
            done = False
            _last_coverage = 0.0
            reward_sum_ep = 0.0
            values, log_probs, rewards, entropies = [], [], [], []

        # --- Coordinator step ---
        value, goal_assignments, entropy, log_prob = local_model(state)
        # goal_assignments: [n_sensors, n_targets, 1]
        goals = goal_assignments.squeeze(-1).detach().cpu().numpy()  # [n_sensors, n_targets]

        # Run k=COORDINATOR_PERIOD env steps with scripted executor
        step_rewards = []
        _last_coverage = 0.0
        for _ in range(COORDINATOR_PERIOD):
            actions = [scripted_action(env, i, goals[i]) for i in range(n_sensors)]
            obs_list, _, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            n_tracked = info.get("n_tracked", 0)
            coverage = info.get("coverage_rate", 0.0)
            _last_coverage = coverage
            team_reward = coverage if n_tracked > 0 else -0.1
            step_rewards.append(team_reward)

            eps_rotation_count += sum(1 for a in actions if a != 1)
            eps_len += 1

            if done:
                break

        # Coordinator reward = mean team reward over k steps
        coord_reward = float(np.mean(step_rewards))
        reward_sum_ep += coord_reward
        values.append(value)
        log_probs.append(log_prob)
        rewards.append(torch.tensor([[coord_reward]] * 1).float().to(device))
        entropies.append(entropy)

        state = _obs_to_tensor(obs_list, n_sensors, n_targets, device)

        n_iter += 1
        n_iters[rank] = n_iter

        # Update after num_steps coordinator decisions
        if len(rewards) >= args.num_steps or done:
            _update(
                local_model, shared_model, optimizer,
                values, log_probs, rewards, entropies,
                done, state, args, device, device_share, params
            )
            values, log_probs, rewards, entropies = [], [], [], []

        if train_modes[rank] == -100:
            env.close()
            break


def _obs_to_tensor(obs_list, n_sensors, n_targets, device):
    obs = np.array(obs_list, dtype=np.float32).reshape(n_sensors, n_targets, 4)
    return torch.from_numpy(obs).float().to(device)


def _update(local_model, shared_model, optimizer,
            values, log_probs, rewards, entropies,
            done, state, args, device, device_share, params):
    """A3C update for coordinator."""
    R = torch.zeros(1, 1).to(device)
    if not done:
        value, *_ = local_model(Variable(state))
        R = value.data

    values.append(Variable(R))
    policy_loss = torch.zeros(1, 1).to(device)
    value_loss = torch.zeros(1, 1).to(device)
    gae = torch.zeros(1, 1).to(device)

    for i in reversed(range(len(rewards))):
        R = args.gamma * R + rewards[i]
        advantage = R - values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
        gae = gae * args.gamma * args.tau + delta_t

        # log_probs[i]: [n_sensors*n_targets, 1] — sum over all assignments
        policy_loss = policy_loss - (log_probs[i].sum() * Variable(gae)) - (
            float(args.entropy) * entropies[i].sum()
        )

    local_model.zero_grad()
    loss = policy_loss + 0.5 * value_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 50)
    ensure_shared_grads(local_model, shared_model, device, device_share)
    optimizer.step()
