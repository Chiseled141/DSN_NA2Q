
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
    try:
        from setproctitle import setproctitle as ptitle
    except ImportError:
        def ptitle(name): pass
except ImportError:
    def ptitle(title):
        pass


def test(args, shared_model, optimizer, train_modes, n_iters, episode_rewards=None, coverage_rates=None, episode_durations=None):
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
    
    # Use shared lists if available (for training history)
    # If not provided (standalone test), use local lists
    if episode_rewards is None:
        episode_rewards = []
    if coverage_rates is None:
        coverage_rates = []

    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.model = build_model(
        env.n_sensors, env.n_targets, env.n_actions,
        args, device
    )

    player.model.eval()
    max_score = -100

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
                    episode_coverage = player.info.get('coverage_rate', 0)
                    AG += episode_coverage
                    
                    # Accumulate
                    reward_sum += reward_sum_ep
                    reward_sum_list.append(reward_sum_ep[0])
                    coverage_sum_list.append(episode_coverage)
                    
                    len_sum += player.eps_len
                    fps = fps_counter / (time.time() - t0 + 1e-8)
                    
                    n_iter = sum(n_iters)

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

        log_info(
            "Time {0}, ave eps reward {1}, ave eps length {2}, reward step {3}, FPS {4}, "
            "mean reward {5}, std reward {6}, coverage {7:.1%}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                np.around(ave_reward_sum, decimals=2), np.around(len_mean, decimals=2),
                np.around(reward_step, decimals=2), np.around(np.mean(fps_all), decimals=2),
                mean_reward, std_reward, mean_coverage
            ))
            
        # Log to TensorBoard if enabled (for test metrics only)
        if writer is not None:
             writer.add_scalar('test/mean_reward', mean_reward, n_iter)
             writer.add_scalar('test/mean_coverage', mean_coverage, n_iter)

        # Save model to hitmac/checkpoints/
        # Always save latest model
        state_to_save = {
            "model": player.model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "training_history": {
                "episode_rewards": list(episode_rewards),
                "coverage_rates": list(coverage_rates),
                "losses": [] # A3C doesn't have centralized loss
            }
        }
        torch.save(state_to_save, os.path.join(checkpoints_dir, 'latest.pt'))
        
        # Save best model if improved
        if ave_reward_sum[0] >= max_score:
            print(f'Saving best model (reward: {ave_reward_sum[0]:.2f}, coverage: {mean_coverage:.1%})')
            max_score = ave_reward_sum[0]
            torch.save(state_to_save, os.path.join(checkpoints_dir, 'best.pt'))
        
        # Save training history as .npz (matches NA2Q format)
        # Use metrics from the SHARED lists collected from WORKERS
        try:
            # Handle potential concurrency issues with a quick snapshot
            current_rewards = list(episode_rewards)
            current_coverage = list(coverage_rates)
            
            np.savez(
                os.path.join(checkpoints_dir, "training_history.npz"),
                episode_rewards=np.array(current_rewards),
                coverage_rates=np.array(current_coverage),
                losses=np.array([]),
                episode_durations=np.array(list(episode_durations) if episode_durations else [])
            )
        except Exception as e:
            print(f"Warning: Failed to save history: {e}")

        time.sleep(args.sleep_time)

    if writer is not None:
        writer.close()
