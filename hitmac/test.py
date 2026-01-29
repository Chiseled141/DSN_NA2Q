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

from __future__ import division
import os
import time
import torch
import logging
import numpy as np

from hitmac.models import build_model
from hitmac.player import Agent, setup_logger
from environments.environment import DSNEnv

# Optional dependencies
try:
    from tensorboardX import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    from setproctitle import setproctitle as ptitle
except ImportError:
    def ptitle(title):
        pass


def test(args, shared_model, optimizer, train_modes, n_iters):
    """Test process for A3C - evaluates and saves best models.
    
    Args:
        args: Test arguments (must include checkpoints_dir)
        shared_model: Shared model to evaluate
        optimizer: Optimizer (for checkpoint saving)
        train_modes: Shared list for coordination
        n_iters: Shared list tracking training iterations
    """
    ptitle('Test Agent')
    n_iter = 0
    
    # Disable TensorBoard to keep Result folder clean
    writer = None
        
    gpu_id = args.gpu_ids[-1]
    
    # Use print-based logging instead of file logging
    def log_info(msg):
        print(f"[HiT-MAC] {msg}")

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    # Create DSNEnv directly
    env = DSNEnv(scenario=getattr(args, 'scenario', 1), seed=args.seed)
    start_time = time.time()
    count_eps = 0
    
    # Get checkpoints directory (inside hitmac/)
    checkpoints_dir = getattr(args, 'checkpoints_dir', 
                              os.path.join(os.path.dirname(__file__), 'checkpoints'))
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Training history (matches NA2Q format)
    training_history = {
        "episode_rewards": [],
        "coverage_rates": [],
        "losses": []
    }

    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.model = build_model(
        env.n_sensors, env.n_targets, env.n_actions,
        args, device
    ).to(device)
    player.model.eval()
    max_score = -100

    while True:
        AG = 0
        reward_sum = np.zeros(player.num_agents)
        reward_sum_list = []
        coverage_sum_list = []
        len_sum = 0
        
        for i_episode in range(args.test_eps):
            player.model.load_state_dict(shared_model.state_dict())
            player.reset()
            reward_sum_ep = np.zeros(player.num_agents)
            rotation_sum_ep = 0
            episode_coverage = 0

            fps_counter = 0
            t0 = time.time()
            count_eps += 1
            fps_all = []
            
            while True:
                player.action_test()
                fps_counter += 1
                reward_sum_ep += player.reward
                rotation_sum_ep += player.rotation
                
                # Track coverage from info
                if player.info:
                    episode_coverage = player.info.get('coverage_rate', 0)
                
                if player.done:
                    if rotation_sum_ep > 0:
                        AG += reward_sum_ep[0] / rotation_sum_ep * player.num_agents
                    reward_sum += reward_sum_ep
                    reward_sum_list.append(reward_sum_ep[0])
                    coverage_sum_list.append(episode_coverage)
                    len_sum += player.eps_len
                    fps = fps_counter / (time.time() - t0 + 1e-8)
                    
                    n_iter = sum(n_iters)

                    if writer is not None:
                        for i, r_i in enumerate(reward_sum_ep):
                            writer.add_scalar('test/reward' + str(i), r_i, n_iter)
                        writer.add_scalar('test/fps', fps, n_iter)
                        writer.add_scalar('test/eps_len', player.eps_len, n_iter)
                        writer.add_scalar('test/coverage', episode_coverage, n_iter)

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
        
        # Record to training history
        training_history["episode_rewards"].append(mean_reward)
        training_history["coverage_rates"].append(mean_coverage)
        training_history["losses"].append(0)  # A3C doesn't have centralized loss

        log_info(
            "Time {0}, ave eps reward {1}, ave eps length {2}, reward step {3}, FPS {4}, "
            "mean reward {5}, std reward {6}, coverage {7:.1%}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                np.around(ave_reward_sum, decimals=2), np.around(len_mean, decimals=2),
                np.around(reward_step, decimals=2), np.around(np.mean(fps_all), decimals=2),
                mean_reward, std_reward, mean_coverage
            ))

        # Save model to hitmac/checkpoints/
        # Always save latest model
        state_to_save = {
            "model": player.model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "training_history": training_history,
        }
        torch.save(state_to_save, os.path.join(checkpoints_dir, 'latest.pt'))
        
        # Save best model if improved
        if ave_reward_sum[0] >= max_score:
            print(f'Saving best model (reward: {ave_reward_sum[0]:.2f}, coverage: {mean_coverage:.1%})')
            max_score = ave_reward_sum[0]
            torch.save(state_to_save, os.path.join(checkpoints_dir, 'best.pt'))
        
        # Save training history as .npz (matches NA2Q format)
        np.savez(
            os.path.join(checkpoints_dir, "training_history.npz"),
            episode_rewards=np.array(training_history["episode_rewards"]),
            coverage_rates=np.array(training_history["coverage_rates"]),
            losses=np.array(training_history["losses"])
        )

        time.sleep(getattr(args, 'sleep_time', 0))
        
        # Check termination
        if n_iter > args.max_step:
            env.close()
            for id in range(0, args.workers):
                train_modes[id] = -100
            
            # Final summary
            print("\n" + "=" * 50)
            print("HiT-MAC Training Complete!")
            print("=" * 50)
            print(f"Best Reward: {max_score:.2f}")
            print(f"Final Coverage: {mean_coverage:.1%}")
            print(f"Checkpoints: {checkpoints_dir}")
            print(f"  - best.pt")
            print(f"  - training_history.npz")
            print("=" * 50)
            break
