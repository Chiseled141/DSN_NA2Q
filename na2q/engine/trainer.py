"""
NA²Q Trainer - Main Training Engine.

Handles training loop, logging, evaluation, and checkpointing.
"""

import glob
import os
import time
from typing import Dict


def _fmt_time(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

import numpy as np
import torch
from tqdm import tqdm

from environments.environment import make_env
from na2q.engine.collector import collect_episode
from na2q.models.agent import NA2QAgent
from na2q.utils import EpisodeReplayBuffer, Logger, MetricsTracker, get_device

# =============================================================================
# Trainer Class
# =============================================================================


class Trainer:
    """Manages the training process for NA²Q."""

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, config: Dict):
        self.config = config
        self.device = get_device(config.get("device"))

        # Directories
        # Checkpoints go in na2q/checkpoints/scenario{N}/ or scenario{N}/run{K}/
        scenario = config.get("scenario", 1)
        run_id   = config.get("run_id")
        na2q_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = os.path.join(na2q_dir, "checkpoints", f"scenario{scenario}")
        self.checkpoints_dir = os.path.join(base, f"run{run_id}") if run_id is not None else base
        self.history_dir = self.checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Results (charts, GIFs) go in Result/ScenarioX/
        self.exp_dir = os.path.join("Result", f"Scenario{scenario}")
        self.media_dir = self.exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)

        # Logger — log file saved to na2q/checkpoints/training.log
        self.logger = Logger(
            self.checkpoints_dir,
            experiment_name="",
            use_tensorboard=False,
            write_log_file=True,
        )
        self.tracker = MetricsTracker()

        # Environment
        self.scenario = config.get("scenario", 1)
        self.num_envs = config.get("num_envs", 1)
        self.seed = config.get("seed", 42)
        self.max_steps = config.get("max_steps", 100)
        self.env_kwargs = config.get("env_kwargs", {})
        self._setup_environments()
        self._setup_agent()
        self._setup_buffer()

        # State
        self.start_episode = 0
        self.training_history = {
            "episode_rewards": [],
            "coverage_rates": [],
            "losses": [],
            "episode_durations": [],
        }

    def _setup_environments(self):
        self.env = make_env(
            scenario=self.scenario,
            max_steps=self.max_steps,
            seed=self.seed,
            **self.env_kwargs,
        )
        self.n_agents = self.env.n_sensors
        self.obs_dim = self.env.obs_dim
        self.state_dim = self.env.state_dim
        self.n_actions = self.env.n_actions

        self.eval_env = make_env(
            scenario=self.scenario,
            max_steps=self.max_steps,
            seed=self.seed + 1000,
            **self.env_kwargs,
        )

    def _setup_agent(self):
        self.agent = NA2QAgent(
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            lr=self.config.get("lr", 5e-4),
            gamma=self.config.get("gamma", 0.99),
            epsilon_start=self.config.get("epsilon_start", 1.0),
            epsilon_end=self.config.get("epsilon_end", 0.05),
            epsilon_decay=self.config.get("epsilon_decay", 10000),
            target_update_interval=self.config.get("target_update_interval", 200),
            device=self.device,
            use_amp=self.config.get("use_amp", True),
            hidden_dim=128,
            rnn_hidden_dim=128,
            attention_hidden_dim=128,
        )

    def _setup_buffer(self):
        buffer_capacity = self.config.get("buffer_capacity", 8000)
        self.buffer = EpisodeReplayBuffer(
            capacity=buffer_capacity,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            max_episode_length=self.max_steps,
            chunk_length=self.config.get("chunk_length", 100),
        )

    def _transfer_weights(self, path: str):
        if not os.path.exists(path):
            print(f"  Warning: transfer checkpoint not found: {path}")
            return
        print(f"  Transfer learning from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        transferred = 0
        for model in (self.agent.model, self.agent.target_model):
            key = "model_state_dict" if model is self.agent.model else "target_model_state_dict"
            src = checkpoint.get(key, {})
            own = model.state_dict()
            compatible = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
            own.update(compatible)
            model.load_state_dict(own)
            transferred = len(compatible)
        skipped = len(checkpoint.get("model_state_dict", {})) - transferred
        print(f"  Transferred {transferred} layers, skipped {skipped} (size mismatch)")

    def _resume_training(self):
        checkpoint_path = os.path.join(self.checkpoints_dir, "final_model.pt")
        if os.path.exists(checkpoint_path):
            print(f"  Resuming from: {checkpoint_path}")
            self.agent.load(checkpoint_path)
            history_path = os.path.join(self.checkpoints_dir, "training_history.npz")
            if os.path.exists(history_path):
                old_history = np.load(history_path)
                self.training_history = {
                    "episode_rewards": list(old_history["episode_rewards"]),
                    "coverage_rates": list(old_history["coverage_rates"]),
                    "losses": list(old_history["losses"]),
                    "episode_durations": (
                        list(old_history["episode_durations"])
                        if "episode_durations" in old_history
                        else []
                    ),
                }
                self.start_episode = len(self.training_history["episode_rewards"])
                print(f"  Continuing from episode {self.start_episode}")
        else:
            print(f"  Warning: No checkpoint found, starting fresh")

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------

    def _time_prefix(self, done_episodes: int) -> str:
        elapsed = time.time() - self.train_start_time
        done = done_episodes - self.start_episode
        progress = done / max(self.n_episodes, 1)
        if progress > 0:
            eta = elapsed / progress * (1 - progress)
            return f"[+{_fmt_time(elapsed)} | ~{_fmt_time(eta)} left]"
        return f"[+{_fmt_time(elapsed)} | ~--:--:-- left]"

    def train(self) -> Dict:
        """Run the training loop."""
        self.train_start_time = time.time()
        n_episodes = self.config.get("n_episodes", 2000)
        self.n_episodes = n_episodes

        if self.config.get("resume"):
            self._resume_training()
        elif self.config.get("transfer_from"):
            self._transfer_weights(self.config["transfer_from"])

        self.logger.log_message(f"[+00:00:00 | ~--:--:-- left] Device: {self.device}")
        batch_size = self.config.get("batch_size", 32)
        learning_starts = self.config.get("learning_starts", 5000)
        eval_interval = self.config.get("eval_interval", 50)
        save_interval = self.config.get("save_interval", 100)

        target_episodes = self.start_episode + n_episodes
        pbar = tqdm(
            total=target_episodes,
            initial=self.start_episode,
            desc="Training",
            unit="ep",
        )

        total_episodes = self.start_episode
        session_episodes = self.start_episode  # Continue epsilon/curriculum from where we left off
        last_eval_episode = self.start_episode
        last_save_episode = self.start_episode
        best_eval_reward = -float("inf")
        current_loss = 0.0

        try:
            while total_episodes < target_episodes:
                # Collect episode
                start_time = time.time()
                episode, info = collect_episode(self.env, self.agent, self.max_steps)
                duration = time.time() - start_time
                self._process_episode(episode, info, total_episodes, duration)
                total_episodes += 1
                pbar.update(1)

                # Update epsilon based on session episode count
                self.agent.set_episode_count(session_episodes)

                # Curriculum
                session_episodes += 1
                self._update_curriculum(session_episodes, n_episodes)

                # Training updates
                updates_per_step = self.config.get("updates_per_step", 1)
                for _ in range(updates_per_step):
                    loss = self._training_step(
                        batch_size, learning_starts, total_episodes
                    )
                    if loss is not None:
                        current_loss = loss
                    self.training_history["losses"].append(loss)

                # Log after training so loss reflects this episode's update
                self._log_progress(total_episodes - 1, current_loss)

                # Progress bar
                pbar.set_postfix(
                    {
                        "R": f"{self.tracker.get_mean('reward'):.2f}",
                        "C": f"{self.tracker.get_mean('coverage'):.1%}",
                        "ε": f"{self.agent.epsilon:.3f}",
                        "L": f"{current_loss:.3f}",
                    }
                )

                # Evaluation
                if total_episodes >= last_eval_episode + eval_interval:
                    last_eval_episode = total_episodes
                    best_eval_reward = self._evaluate_and_save(
                        total_episodes, best_eval_reward
                    )

                # Checkpoint
                if total_episodes >= last_save_episode + save_interval:
                    last_save_episode = total_episodes
                    self._save_checkpoint(total_episodes, n_episodes)

                # GPU cleanup
                if self.device == "cuda" and total_episodes % 1000 == 0:
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            self._handle_interrupt()

        self._final_save(best_eval_reward)
        self._print_training_report(total_episodes, best_eval_reward)
        self._cleanup()

        return {
            "exp_dir": self.exp_dir,
            "best_model_path": os.path.join(self.checkpoints_dir, "best_model.pt"),
            "training_history": self.training_history,
            "best_eval_reward": best_eval_reward,
            "total_episodes": total_episodes,
        }

    def _log_progress(self, total_episodes, current_loss):
        prefix = self._time_prefix(total_episodes)
        self.logger.log_message(
            f"{prefix} [Ep {total_episodes}] Reward: {self.tracker.get_mean('reward'):.2f} | "
            f"Coverage: {self.tracker.get_mean('coverage'):.1%} | "
            f"Eps: {self.agent.epsilon:.3f} | Loss: {current_loss:.4f}"
        )

    def _update_curriculum(self, session_episodes, total_session_episodes):
        if session_episodes % 100 != 0:
            return
        # Curriculum based on session progress, not total episodes
        ramp_episodes = total_session_episodes / 2.0
        curriculum_level = min(1.0, session_episodes / ramp_episodes)
        self.env.set_curriculum_difficulty(curriculum_level)
        self.eval_env.set_curriculum_difficulty(curriculum_level)

    # -------------------------------------------------------------------------
    # Episode Processing
    # -------------------------------------------------------------------------

    def _process_episode(self, episode, info, total_episodes, duration=0.0):
        self.buffer.add_episode(episode)
        reward = sum(episode["rewards"])
        coverage = info.get("coverage_rate", 0)

        self.tracker.add("reward", reward)
        self.tracker.add("coverage", coverage)
        self.training_history["episode_rewards"].append(reward)
        self.training_history["coverage_rates"].append(coverage)
        self.training_history["episode_durations"].append(duration)

        self.logger.log_scalars(
            {
                "episode/reward": reward,
                "episode/coverage_rate": coverage,
                "episode/duration": duration,
            },
            total_episodes,
        )

    # -------------------------------------------------------------------------
    # Training Step
    # -------------------------------------------------------------------------

    def _training_step(self, batch_size, learning_starts, total_episodes):
        can_train = len(self.buffer) > 0 and total_episodes >= learning_starts
        if not can_train:
            return 0.0

        batch = self.buffer.sample(batch_size)
        train_info = self.agent.train_step_fn(batch)
        loss = train_info["loss"]

        self.logger.log_scalar("train/loss", loss, self.agent.train_step)
        self.tracker.add("loss", loss)
        return loss

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def _evaluate_and_save(self, episode, best_reward):
        n_eval = self.config.get("eval_episodes", 20)
        avg_reward, avg_coverage = self._run_evaluation(n_eval)

        self.logger.log_scalars(
            {"eval/reward_mean": avg_reward, "eval/coverage_mean": avg_coverage},
            episode,
        )

        self.logger.log_message(
            f"\n{self._time_prefix(episode)} [Eval] Episode {episode}: Reward={avg_reward:.2f}, Coverage={avg_coverage:.1%}"
        )

        if avg_reward > best_reward:
            best_reward = avg_reward
            self.agent.save(os.path.join(self.checkpoints_dir, "best_model.pt"))
            self.logger.log_message("  -> New best model saved!")

        return best_reward

    def _run_evaluation(self, n_episodes=20):
        rewards, coverages = [], []
        for _ in range(n_episodes):
            obs_list, _ = self.eval_env.reset()
            observations = np.stack(obs_list)
            self.agent.init_hidden(1)

            episode_reward = 0
            done, truncated = False, False
            prev_actions = np.zeros(self.n_agents, dtype=np.int64)

            while not done and not truncated:
                avail_actions = np.stack(self.eval_env.get_avail_actions())
                actions = self.agent.select_actions(
                    observations, prev_actions, avail_actions, evaluate=True
                )
                next_obs, reward, done, truncated, info = self.eval_env.step(
                    actions.tolist()
                )
                observations = np.stack(next_obs)
                episode_reward += reward
                prev_actions = actions

            rewards.append(episode_reward)
            coverages.append(info.get("coverage_rate", 0))

        return np.mean(rewards), np.mean(coverages)

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def _save_checkpoint(self, episode, max_episodes):
        path = os.path.join(self.checkpoints_dir, f"checkpoint_{episode}.pt")
        self.agent.save(path)

        # Cleanup old checkpoints
        if max_episodes > 2000:
            files = sorted(
                glob.glob(os.path.join(self.checkpoints_dir, "checkpoint_*.pt")),
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            for f in files[:-10]:
                if "best" not in f and "final" not in f:
                    try:
                        os.remove(f)
                    except:
                        pass

    def _print_training_report(self, total_episodes, best_eval_reward):
        """Print and log detailed training report after completion."""
        rewards = self.training_history["episode_rewards"]
        coverages = self.training_history["coverage_rates"]
        losses = [l for l in self.training_history["losses"] if l > 0]

        loss_lines = (
            f"    Final (last 100): {np.mean(losses[-100:]):.4f}\n"
            f"    Minimum:          {np.min(losses):.4f}"
            if losses
            else "    No training updates"
        )

        report = (
            f"\n{'=' * 70}\n"
            f"                    NA²Q TRAINING REPORT\n"
            f"{'=' * 70}\n"
            f"  Scenario:           {self.scenario}\n"
            f"  Total Episodes:     {total_episodes:,}\n"
            f"  Parallel Envs:      {self.num_envs}\n"
            f"{'-' * 70}\n"
            f"  REWARD STATISTICS\n"
            f"    Final (last 100): {np.mean(rewards[-100:]):.2f}\n"
            f"    Best Episode:     {np.max(rewards):.2f}\n"
            f"    Overall Mean:     {np.mean(rewards):.2f}\n"
            f"    Std Deviation:    {np.std(rewards):.2f}\n"
            f"{'-' * 70}\n"
            f"  COVERAGE STATISTICS\n"
            f"    Final (last 100): {np.mean(coverages[-100:]) * 100:.1f}%\n"
            f"    Best Episode:     {np.max(coverages) * 100:.1f}%\n"
            f"    Overall Mean:     {np.mean(coverages) * 100:.1f}%\n"
            f"    Std Deviation:    {np.std(coverages) * 100:.1f}%\n"
            f"{'-' * 70}\n"
            f"  TRAINING LOSS\n"
            f"{loss_lines}\n"
            f"{'-' * 70}\n"
            f"  MODEL SAVED\n"
            f"    Best Model:       {self.checkpoints_dir}/best_model.pt\n"
            f"    Final Model:      {self.checkpoints_dir}/final_model.pt\n"
            f"    History:          {self.history_dir}/training_history.npz\n"
            f"    Log:              {self.checkpoints_dir}/training.log\n"
            f"{'=' * 70}\n"
        )
        self.logger.log_message(report)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def _handle_interrupt(self):
        self.logger.log_message("\nTraining interrupted!")
        self.agent.save(os.path.join(self.checkpoints_dir, "interrupted_model.pt"))

    def _final_save(self, best_reward):
        final_path = os.path.join(self.checkpoints_dir, "final_model.pt")
        self.agent.save(final_path)

        best_path = os.path.join(self.checkpoints_dir, "best_model.pt")
        if not os.path.exists(best_path):
            import shutil

            shutil.copy(final_path, best_path)

        np.savez(
            os.path.join(self.history_dir, "training_history.npz"),
            episode_rewards=np.array(self.training_history["episode_rewards"]),
            coverage_rates=np.array(self.training_history["coverage_rates"]),
            losses=np.array(self.training_history["losses"]),
            episode_durations=np.array(self.training_history["episode_durations"]),
        )
        self.logger.close()

    def _cleanup(self):
        if getattr(self, "env", None):
            self.env.close()
        if self.eval_env:
            self.eval_env.close()
