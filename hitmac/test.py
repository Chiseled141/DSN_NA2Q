"""
HiT-MAC Testing Process.

Evaluates trained models and saves checkpoints based on performance.
"""

from __future__ import division
import os
import time
import torch
import logging
import numpy as np

from hitmac.models import build_model
from hitmac.player import Agent, setup_logger
from hitmac.environment import create_env

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
        args: Test arguments
        shared_model: Shared model to evaluate
        optimizer: Optimizer (for checkpoint saving)
        train_modes: Shared list for coordination
        n_iters: Shared list tracking training iterations
    """
    ptitle('Test Agent')
    n_iter = 0
    
    if HAS_TENSORBOARD:
        writer = SummaryWriter(os.path.join(args.log_dir, 'Test'))
    else:
        writer = None
        
    gpu_id = args.gpu_ids[-1]
    log = {}
    
    # Setup logging
    os.makedirs(os.path.join(args.log_dir, 'logger'), exist_ok=True)
    setup_logger('{}_log'.format(args.env),
                 os.path.join(args.log_dir, 'logger', 'log.txt'))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    
    # Log arguments
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    env = create_env(args.env, args)
    env.seed(args.seed)
    start_time = time.time()
    count_eps = 0

    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.model = build_model(
        player.env.observation_space, 
        player.env.action_space, 
        args, 
        device
    ).to(device)
    player.model.eval()
    max_score = -100

    while True:
        AG = 0
        reward_sum = np.zeros(player.num_agents)
        reward_sum_list = []
        len_sum = 0
        
        for i_episode in range(args.test_eps):
            player.model.load_state_dict(shared_model.state_dict())
            player.reset()
            reward_sum_ep = np.zeros(player.num_agents)
            rotation_sum_ep = 0

            fps_counter = 0
            t0 = time.time()
            count_eps += 1
            fps_all = []
            
            while True:
                player.action_test()
                fps_counter += 1
                reward_sum_ep += player.reward
                rotation_sum_ep += player.rotation
                
                if player.done:
                    if rotation_sum_ep > 0:
                        AG += reward_sum_ep[0] / rotation_sum_ep * player.num_agents
                    reward_sum += reward_sum_ep
                    reward_sum_list.append(reward_sum_ep[0])
                    len_sum += player.eps_len
                    fps = fps_counter / (time.time() - t0 + 1e-8)
                    
                    n_iter = sum(n_iters)

                    if writer is not None:
                        for i, r_i in enumerate(reward_sum_ep):
                            writer.add_scalar('test/reward' + str(i), r_i, n_iter)
                        writer.add_scalar('test/fps', fps, n_iter)
                        writer.add_scalar('test/eps_len', player.eps_len, n_iter)

                    fps_all.append(fps)
                    break

        # Compute statistics
        ave_AG = AG / args.test_eps
        ave_reward_sum = reward_sum / args.test_eps
        len_mean = len_sum / args.test_eps
        reward_step = reward_sum / (len_sum + 1e-8)
        mean_reward = np.mean(reward_sum_list)
        std_reward = np.std(reward_sum_list)

        log['{}_log'.format(args.env)].info(
            "Time {0}, ave eps reward {1}, ave eps length {2}, reward step {3}, FPS {4}, "
            "mean reward {5}, std reward {6}, AG {7}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                np.around(ave_reward_sum, decimals=2), np.around(len_mean, decimals=2),
                np.around(reward_step, decimals=2), np.around(np.mean(fps_all), decimals=2),
                mean_reward, std_reward, np.around(ave_AG, decimals=2)
            ))

        # Save model
        os.makedirs(args.log_dir, exist_ok=True)
        if ave_reward_sum[0] >= max_score:
            print(f'Saving best model (reward: {ave_reward_sum[0]:.2f})')
            max_score = ave_reward_sum[0]
            model_dir = os.path.join(args.log_dir, 'best.pt')
        else:
            model_dir = os.path.join(args.log_dir, 'latest.pt')
            
        state_to_save = {
            "model": player.model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None
        }
        torch.save(state_to_save, model_dir)

        time.sleep(getattr(args, 'sleep_time', 0))
        
        # Check termination
        if n_iter > args.max_step:
            env.close()
            for id in range(0, args.workers):
                train_modes[id] = -100
            break
