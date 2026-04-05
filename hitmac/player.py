"""
=============================================================================
HiT-MAC AGENT (Player Utility)
=============================================================================

WHAT IS THIS FILE?
------------------
Contains the Agent class that manages the interaction between the policy
network and the environment during A3C (Asynchronous Advantage Actor-Critic)
training.

A3C TRAINING LOOP:
------------------
    ┌─────────────┐
    │   Agent     │
    │  (this file)│
    └──────┬──────┘
           │
    ┌──────▼──────┐     ┌─────────────┐
    │   Model     │◄────│   Shared    │
    │  (local)    │     │    Model    │
    └──────┬──────┘     └──────▲──────┘
           │                   │
    ┌──────▼──────┐     ┌──────┴──────┐
    │ Environment │     │  Gradients  │
    │   Step      │     │   (sync)    │
    └──────┬──────┘     └─────────────┘
           │
    ┌──────▼──────┐
    │  Collect    │
    │ Trajectory  │ → values, log_probs, rewards, entropies
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Optimize   │ → Compute GAE, backprop, update shared model
    └─────────────┘

KEY METHODS:
------------
- action_train(): Take action, store trajectory data
- action_test(): Greedy action for evaluation
- optimize(): Compute loss and sync gradients to shared model
"""

import logging

import numpy as np
import torch
from torch.autograd import Variable


def setup_logger(logger_name, log_file, level=logging.INFO):
    """Setup a logger with file and stream handlers."""
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    fileHandler = logging.FileHandler(log_file, mode="w")
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)
    return l


def ensure_shared_grads(model, shared_model, device, device_share):
    """Copy gradients from local model to shared model.

    Handles the case where models are on different devices.
    """
    diff_device = device != device_share
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if param.grad is None:
            continue
        if diff_device:
            shared_param._grad = param.grad.to(device_share)
        else:
            shared_param._grad = param.grad


class Agent(object):
    """Agent for HiT-MAC that manages episode collection and updates.

    Works directly with DSNEnv (same environment as NA2Q).

    Handles:
    - Environment interaction
    - Action selection (train and test modes)
    - Trajectory storage
    - Policy optimization with GAE
    """

    def __init__(self, model, env, args, state, device):
        self.model = model
        self.env = env

        # Environment dimensions from DSNEnv
        self.num_agents = env.n_sensors
        self.num_targets = env.n_targets
        self.state_dim = env.n_targets * 4  # 4 features per target

        # Action space
        self.dim_action = 1
        self.continuous = False

        self.eps_len = 0
        self.eps_num = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.rewards_eps = []
        self.done = True
        self.info = None
        self.reward = 0
        self.device = device
        self.lstm_out = args.lstm_out
        self.reward_mean = None
        self.reward_std = 1
        self.num_steps = 0
        self.n_steps = 0
        self.vk = 0
        self.state = state
        self.hxs = torch.zeros(self.num_agents, self.lstm_out).to(device)
        self.cxs = torch.zeros(self.num_agents, self.lstm_out).to(device)
        self.rank = 0
        self.rotation = 0
        self.eps_rotation_count = 0

    def action_train(self):
        """Take a training action and store trajectory data."""
        self.n_steps += 1
        value_multi, actions, entropy, log_prob = self.model(
            Variable(self.state, requires_grad=True)
        )

        # DSNEnv returns 5-tuple: (obs_list, reward, terminated, truncated, info)
        actions_list = actions.cpu().tolist() if isinstance(actions, torch.Tensor) else actions.tolist()
        obs_list, reward, terminated, truncated, self.info = self.env.step(actions_list)
        self.done = terminated or truncated

        # Reshape observations for model: [n_agents, n_targets, 4]
        state_multi = self._reshape_obs(obs_list)
        self.state = torch.from_numpy(state_multi).float().to(self.device)

        # Paper Eq. 3: team reward = coverage_rate, or -0.1 if nothing covered
        n_tracked = self.info.get("n_tracked", 0)
        coverage_rate = self.info.get("coverage_rate", 0.0)
        team_reward = coverage_rate if n_tracked > 0 else -0.1

        # Paper Eq. 4: executor reward = angle-error reward per assigned target
        # obs: [sensor_id, target_id, dist_norm, angle_norm]  angle_norm in [-1,1]
        angles = state_multi[:, :, 3] * np.pi  # [n_agents, n_targets] in radians
        dist_norm = state_multi[:, :, 2]
        diag = self.env.field_size * np.sqrt(2)
        distances = dist_norm * diag
        alpha_max = self.env.fov_angle / 2   # half-FOV
        rho_max = self.env.sensing_range
        in_range = (distances <= rho_max) & (np.abs(angles) <= alpha_max)
        # r_i,j = 1 - |α|/α_max if in FoV range, else -1
        r_ij = np.where(in_range, 1.0 - np.abs(angles) / alpha_max, -1.0).astype(np.float32)

        # Goal: targets within sensing range (paper's goal generation strategy, Appendix A)
        goal_map = (distances <= rho_max).astype(np.float32)
        goal_count = np.maximum(goal_map.sum(axis=1), 1.0)
        base_reward = (r_ij * goal_map).sum(axis=1) / goal_count

        # Power cost: β=0.01, cost=1 if rotating, 0 if Stay (action 1)
        beta = 0.01
        cost = np.array([0.0 if a == 1 else 1.0 for a in actions_list], dtype=np.float32)
        reward_multi = base_reward - beta * cost + team_reward * 0.1

        self.reward_org = reward_multi.copy()
        self.eps_rotation_count += sum(1 for a in actions_list if a != 1)
        if self.args.norm_reward:
            reward_multi = self.reward_normalizer(reward_multi)
        self.reward = torch.tensor(reward_multi).float().to(self.device)
        self.eps_len += 1
        self.values.append(value_multi)
        self.entropies.append(entropy)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward.unsqueeze(1))

    def action_test(self):
        """Take a greedy test action."""
        with torch.no_grad():
            value_multi, actions, entropy, log_prob = self.model(
                Variable(self.state), True
            )

        # DSNEnv returns 5-tuple
        actions_list = actions.cpu().tolist() if isinstance(actions, torch.Tensor) else actions.tolist()
        obs_list, reward, terminated, truncated, self.info = self.env.step(actions_list)
        self.done = terminated or truncated

        state_multi = self._reshape_obs(obs_list)
        self.state = torch.from_numpy(state_multi).float().to(self.device)

        # Paper Eq. 4: same executor reward as training
        angles = state_multi[:, :, 3] * np.pi
        dist_norm = state_multi[:, :, 2]
        diag = self.env.field_size * np.sqrt(2)
        distances = dist_norm * diag
        alpha_max = self.env.fov_angle / 2
        rho_max = self.env.sensing_range
        in_range = (distances <= rho_max) & (np.abs(angles) <= alpha_max)
        r_ij = np.where(in_range, 1.0 - np.abs(angles) / alpha_max, -1.0).astype(np.float32)
        goal_map = (distances <= rho_max).astype(np.float32)
        goal_count = np.maximum(goal_map.sum(axis=1), 1.0)
        n_tracked = self.info.get("n_tracked", 0)
        coverage_rate = self.info.get("coverage_rate", 0.0)
        team_reward = coverage_rate if n_tracked > 0 else -0.1
        reward_multi = (r_ij * goal_map).sum(axis=1) / goal_count + team_reward * 0.1
        self.reward_org = reward_multi.copy()
        self.eps_rotation_count += sum(1 for a in actions_list if a != 1)
        self.rotation = sum(1 for a in actions_list if a != 1)
        self.eps_len += 1

    def reset(self):
        """Reset for new episode."""
        # DSNEnv returns 2-tuple: (obs_list, info)
        obs_list, info = self.env.reset()
        state = self._reshape_obs(obs_list)
        self.state = torch.from_numpy(state).float().to(self.device)

        self.eps_len = 0
        self.eps_num += 1
        self.eps_rotation_count = 0
        self.reset_rnn_hidden()
        self.model.sample_noise()

    def _reshape_obs(self, obs_list):
        """Reshape flat observations to [n_agents, n_targets, 4]."""
        obs_array = np.array(obs_list, dtype=np.float32)
        return obs_array.reshape(self.num_agents, self.num_targets, 4)

    def clean_buffer(self, done):
        """Clear trajectory buffers after optimization."""
        self.values = []
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.obs_tracker = []
        if done:
            self.rewards_eps = []
        return self

    def reward_normalizer(self, reward):
        """Online reward normalization using Welford's algorithm."""
        reward = np.array(reward)
        self.num_steps += 1
        if self.num_steps == 1:
            self.reward_mean = reward
            self.vk = 0
            self.reward_std = 1
        else:
            delt = reward - self.reward_mean
            self.reward_mean = self.reward_mean + delt / self.num_steps
            self.vk = self.vk + delt * (reward - self.reward_mean)
            self.reward_std = np.sqrt(self.vk / (self.num_steps - 1))
        reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return reward

    def reset_rnn_hidden(self):
        """Reset RNN hidden states."""
        self.cxs = Variable(torch.zeros(self.num_agents, self.lstm_out).to(self.device))
        self.hxs = Variable(torch.zeros(self.num_agents, self.lstm_out).to(self.device))

    def update_rnn_hidden(self):
        """Detach RNN hidden states for next forward pass."""
        self.cxs = Variable(self.cxs.data)
        self.hxs = Variable(self.hxs.data)

    def optimize(self, params, optimizer, shared_model, training_mode, device_share):
        """Perform A3C optimization step with GAE.

        Computes policy and value losses using Generalized Advantage Estimation,
        backpropagates gradients to shared model, and updates parameters.

        Returns:
            policy_loss: Policy gradient loss
            value_loss: Value function loss
            entropies: Policy entropy for exploration bonus
        """
        R = torch.zeros(len(self.rewards[0]), 1).to(self.device)
        if not self.done:
            # Bootstrap value from current state
            state = self.state
            value_multi, *others = self.model(Variable(state, requires_grad=True))
            for i in range(len(self.rewards[0])):
                R[i][0] = value_multi[i].data

        self.values.append(Variable(R).to(self.device))

        batch_size = len(self.entropies[0][0])
        policy_loss = torch.zeros(batch_size, 1).to(self.device)
        value_loss = torch.zeros(1, 1).to(self.device)
        entropies = torch.zeros(batch_size, self.dim_action).to(self.device)
        w_entropies = float(self.args.entropy)

        R = Variable(R, requires_grad=True).to(self.device)
        gae = torch.zeros(1, 1).to(self.device)

        # Compute losses backward through trajectory
        for i in reversed(range(len(self.rewards))):
            R = self.args.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = (
                self.rewards[i]
                + self.args.gamma * self.values[i + 1].data
                - self.values[i].data
            )
            gae = gae * self.args.gamma * self.args.tau + delta_t
            policy_loss = (
                policy_loss
                - (self.log_probs[i] * Variable(gae))
                - (w_entropies * self.entropies[i])
            )
            entropies += self.entropies[i].sum()

        # Backprop and update
        self.model.zero_grad()
        loss = policy_loss.sum() + 0.5 * value_loss.sum()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(params, 50)
        ensure_shared_grads(self.model, shared_model, self.device, device_share)
        optimizer.step()

        self.clean_buffer(self.done)

        return policy_loss, value_loss, entropies
