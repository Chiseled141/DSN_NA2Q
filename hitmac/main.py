"""
HiT-MAC Main Entry Point.

Hierarchical Twin-Actor Multi-Agent Coordination for Directional Sensor Networks.

Usage:
    # Train executor (single-agent control)
    python -m hitmac.main --env Pose-v0 --model single-att --workers 6

    # Train coordinator (multi-agent coordination with Shapley values)
    python -m hitmac.main --env Pose-v1 --model multi-att-shap --workers 6

    # Evaluate trained model
    python -m hitmac.main --env Pose-v1 --model multi-att-shap --workers 0 \\
        --load-coordinator-dir trainedModel/best_coordinator.pt \\
        --load-executor-dir trainedModel/best_executor.pt
"""

from __future__ import print_function, division
import os
import time
import torch
import argparse
from datetime import datetime
import torch.multiprocessing as mp

from hitmac.test import test
from hitmac.train import train
from hitmac.models import build_model
from hitmac.environment import create_env
from hitmac.shared_optim import SharedRMSprop, SharedAdam

os.environ["OMP_NUM_THREADS"] = "1"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='HiT-MAC: Hierarchical Twin-Actor Multi-Agent Coordination')
    
    # Learning
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                        help='discount factor for rewards (default: 0.9)')
    parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                        help='parameter for GAE (default: 1.00)')
    parser.add_argument('--entropy', type=float, default=0.01, metavar='T',
                        help='entropy regularization coefficient (default: 0.01)')
    parser.add_argument('--grad-entropy', type=float, default=1.0, metavar='T',
                        help='gradient entropy coefficient (default: 1.0)')
    
    # Training
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--workers', type=int, default=1, metavar='W',
                        help='number of training processes (default: 1)')
    parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                        help='number of forward steps in A3C (default: 20)')
    parser.add_argument('--test-eps', type=int, default=1, metavar='M',
                        help='number of test episodes (default: 1)')
    parser.add_argument('--max-step', type=int, default=5000000, metavar='LO',
                        help='max training steps (default: 5000000)')
    
    # Environment
    parser.add_argument('--env', default='Pose-v0', metavar='ENV',
                        help='environment: Pose-v0 (executor), Pose-v1 (coordinator)')
    
    # Model
    parser.add_argument('--model', default='single-att', metavar='M',
                        help='model type: single-att, multi-att, multi-att-shap')
    parser.add_argument('--lstm-out', type=int, default=128, metavar='LO',
                        help='LSTM/attention output size (default: 128)')
    
    # Optimizer
    parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                        help='optimizer: Adam or RMSprop')
    parser.add_argument('--amsgrad', default=True, metavar='AM',
                        help='Adam optimizer amsgrad parameter')
    parser.add_argument('--shared-optimizer', dest='shared_optimizer', action='store_true',
                        help='use shared optimizer statistics')
    
    # Checkpoints
    parser.add_argument('--load-coordinator-dir', default=None, metavar='LMD',
                        help='path to load coordinator model')
    parser.add_argument('--load-executor-dir', default=None, metavar='LMD',
                        help='path to load executor model')
    parser.add_argument('--log-dir', default='logs/', metavar='LG',
                        help='folder to save logs and checkpoints')
    
    # Hardware
    parser.add_argument('--gpu-ids', type=int, default=-1, nargs='+',
                        help='GPUs to use [-1 CPU only] (default: -1)')
    
    # Misc
    parser.add_argument('--norm-reward', dest='norm_reward', action='store_true',
                        help='normalize rewards')
    parser.add_argument('--render', dest='render', action='store_true',
                        help='render environment')
    parser.add_argument('--render-save', dest='render_save', action='store_true',
                        help='save rendered frames')
    parser.add_argument('--fix', dest='fix', action='store_true',
                        help='fix random seed across workers')
    parser.add_argument('--train-mode', type=int, default=-1, metavar='TM',
                        help='training mode')
    parser.add_argument('--input-size', type=int, default=80, metavar='IS',
                        help='input image size (default: 80)')
    parser.add_argument('--sleep-time', type=int, default=0, metavar='LO',
                        help='sleep time between process starts (default: 0)')
    
    return parser.parse_args()


def main():
    """Main entry point for HiT-MAC training and evaluation."""
    args = parse_args()
    args.shared_optimizer = True
    
    # Setup device
    if args.gpu_ids == -1:
        torch.manual_seed(args.seed)
        args.gpu_ids = [-1]
        device_share = torch.device('cpu')
        mp.set_start_method('spawn')
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn', force=True)
        if len(args.gpu_ids) > 1:
            device_share = torch.device('cpu')
        else:
            device_share = torch.device('cuda:' + str(args.gpu_ids[-1]))

    # Print banner
    print("=" * 60)
    print("HiT-MAC: Hierarchical Twin-Actor Multi-Agent Coordination")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Device: {device_share}")
    print("=" * 60)

    # Create environment to get observation/action spaces
    env = create_env(args.env, args)
    
    # Build shared model
    shared_model = build_model(env.observation_space, env.action_space, args, device_share)
    shared_model = shared_model.to(device_share)
    shared_model.share_memory()
    env.close()
    del env

    # Load pretrained coordinator if specified
    if args.load_coordinator_dir is not None:
        print(f"Loading coordinator from: {args.load_coordinator_dir}")
        saved_state = torch.load(
            args.load_coordinator_dir,
            map_location=lambda storage, loc: storage
        )
        if args.load_coordinator_dir.endswith('.pt'):
            shared_model.load_state_dict(saved_state['model'], strict=False)
        else:
            shared_model.load_state_dict(saved_state)

    # Setup optimizer
    params = shared_model.parameters()
    if args.shared_optimizer:
        print('Using shared optimizer')
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(params, lr=args.lr)
        elif args.optimizer == 'Adam':
            optimizer = SharedAdam(params, lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    # Setup logging directory
    current_time = datetime.now().strftime('%b%d_%H-%M')
    args.log_dir = os.path.join(args.log_dir, args.env, current_time)
    os.makedirs(args.log_dir, exist_ok=True)

    # Launch processes
    processes = []
    manager = mp.Manager()
    train_modes = manager.list()
    n_iters = manager.list()

    # Start test process
    p = mp.Process(target=test, args=(args, shared_model, optimizer, train_modes, n_iters))
    p.start()
    processes.append(p)
    time.sleep(args.sleep_time)

    # Start training workers
    for rank in range(0, args.workers):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer, train_modes, n_iters))
        p.start()
        processes.append(p)
        time.sleep(args.sleep_time)

    # Wait for all processes
    for p in processes:
        time.sleep(args.sleep_time)
        p.join()

    print("Training complete!")


if __name__ == '__main__':
    main()
