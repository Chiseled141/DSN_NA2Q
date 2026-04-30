"""
Multi-seed experiment runner for NA²Q research paper.

Runs each algorithm × scenario × seed sequentially (no parallelism — safe on low-RAM machines).
Results land in checkpoints/scenario{N}/run{K}/ for each run K.

Usage:
    # Full experiment: NA2Q + HiT-MAC, scenario 1, 5 seeds
    python run_experiments.py --scenario 1 --algorithms na2q hitmac --seeds 5

    # Quick check: 2 seeds only
    python run_experiments.py --scenario 1 --algorithms na2q --seeds 2

    # Single run (equivalent to normal training)
    python run_experiments.py --scenario 1 --algorithms na2q --seeds 1
"""

import argparse
import subprocess
import sys
import os


ALGORITHMS = ["na2q", "hitmac"]


def run_na2q(scenario: int, run_id: int, seed: int):
    cmd = [
        sys.executable, "-m", "na2q.main",
        "--mode", "train",
        "--scenario", str(scenario),
        "--run-id", str(run_id),
        "--seed", str(seed),
    ]
    print(f"\n{'='*60}")
    print(f"NA²Q | scenario {scenario} | run {run_id} | seed {seed}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def run_hitmac(scenario: int, run_id: int, seed: int):
    cmd = [
        sys.executable, "-m", "hitmac.main",
        "--mode", "train",
        "--scenario", str(scenario),
        "--run-id", str(run_id),
        "--seed", str(seed),
    ]
    print(f"\n{'='*60}")
    print(f"HiT-MAC | scenario {scenario} | run {run_id} | seed {seed}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS,
                        choices=ALGORITHMS, help="Algorithms to train")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of independent runs (seeds)")
    parser.add_argument("--start-seed", type=int, default=42,
                        help="Base random seed (incremented per run)")
    args = parser.parse_args()

    total = len(args.algorithms) * args.seeds
    done = 0

    for run_id in range(args.seeds):
        seed = args.start_seed + run_id
        for algo in args.algorithms:
            done += 1
            print(f"\n[{done}/{total}] {algo.upper()} | scenario {args.scenario} "
                  f"| run {run_id} | seed {seed}")
            if algo == "na2q":
                run_na2q(args.scenario, run_id, seed)
            elif algo == "hitmac":
                run_hitmac(args.scenario, run_id, seed)

    print(f"\n{'='*60}")
    print(f"All {total} runs complete.")
    print(f"Aggregate results with:")
    print(f"  python aggregate_results.py --scenario {args.scenario} --seeds {args.seeds}")
    print(f"Then export to dashboard:")
    print(f"  python export_training_data.py --scenario {args.scenario}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
