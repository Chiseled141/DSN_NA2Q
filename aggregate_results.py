"""
Aggregate multi-seed results into mean ± std for dashboard and paper tables.

Reads checkpoints/scenario{N}/run{K}/training_history.npz for each K,
interpolates all runs onto a common episode grid, then computes mean ± std.
Writes an aggregated npz that export_training_data.py can read.

Usage:
    python aggregate_results.py --scenario 1 --seeds 5
    python aggregate_results.py --scenario 1 --seeds 5 --algorithms na2q hitmac
"""

import argparse
import os

import numpy as np


def load_run(path: str):
    """Load training_history.npz from a single run directory."""
    npz_path = os.path.join(path, "training_history.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    return {
        "rewards":   np.array(data["episode_rewards"]),
        "coverage":  np.array(data["coverage_rates"]),
    }


def interpolate_to_grid(values: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 1-D array to target_len points via linear interpolation."""
    src_x = np.linspace(0, 1, len(values))
    dst_x = np.linspace(0, 1, target_len)
    return np.interp(dst_x, src_x, values)


def aggregate(algo: str, scenario: int, n_seeds: int, base_dir: str):
    """Load all runs for an algorithm, interpolate, return mean ± std arrays."""
    runs_rewards  = []
    runs_coverage = []

    for k in range(n_seeds):
        run_dir = os.path.join(base_dir, f"run{k}")
        run = load_run(run_dir)
        if run is None:
            print(f"  ⚠  {algo} run{k} not found — skipping")
            continue
        runs_rewards.append(run["rewards"])
        runs_coverage.append(run["coverage"])
        print(f"  {algo} run{k}: {len(run['rewards'])} episodes")

    if not runs_rewards:
        print(f"  No completed runs found for {algo} scenario {scenario}")
        return None

    # Interpolate all runs to the length of the shortest run
    min_len = min(len(r) for r in runs_rewards)
    rew_grid  = np.stack([interpolate_to_grid(r, min_len) for r in runs_rewards])
    cov_grid  = np.stack([interpolate_to_grid(c, min_len) for c in runs_coverage])

    return {
        "episodes":       np.arange(min_len),
        "reward_mean":    rew_grid.mean(axis=0),
        "reward_std":     rew_grid.std(axis=0),
        "reward_min":     rew_grid.min(axis=0),
        "reward_max":     rew_grid.max(axis=0),
        "coverage_mean":  cov_grid.mean(axis=0),
        "coverage_std":   cov_grid.std(axis=0),
        "coverage_min":   cov_grid.min(axis=0),
        "coverage_max":   cov_grid.max(axis=0),
        "n_seeds":        len(runs_rewards),
    }


def print_summary(name: str, agg: dict):
    """Print paper-table style summary for the last 500 episodes."""
    n = len(agg["coverage_mean"])
    tail = agg["coverage_mean"][max(0, n - 500):]
    tail_std = agg["coverage_std"][max(0, n - 500):]
    mean_cov  = tail.mean() * 100 if tail.max() <= 1.0 else tail.mean()
    mean_std  = tail_std.mean() * 100 if tail_std.max() <= 1.0 else tail_std.mean()
    best      = agg["coverage_mean"].max() * 100 if agg["coverage_mean"].max() <= 1.0 else agg["coverage_mean"].max()
    print(f"\n  {name}")
    print(f"    Seeds:           {agg['n_seeds']}")
    print(f"    Final coverage:  {mean_cov:.1f}% ± {mean_std:.1f}%  (last 500 eps mean)")
    print(f"    Best mean cov:   {best:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--algorithms", nargs="+", default=["na2q", "hitmac"],
                        choices=["na2q", "hitmac"])
    args = parser.parse_args()

    print(f"Aggregating scenario {args.scenario} — {args.seeds} seeds")
    print("=" * 50)

    for algo in args.algorithms:
        if algo == "na2q":
            base = os.path.join("na2q", "checkpoints", f"scenario{args.scenario}")
        else:
            base = os.path.join("hitmac", "checkpoints", f"scenario{args.scenario}")

        print(f"\n{algo.upper()}: loading from {base}/run*/")
        agg = aggregate(algo, args.scenario, args.seeds, base)
        if agg is None:
            continue

        # Save aggregated npz back into the scenario base dir so
        # export_training_data.py can optionally pick it up
        out_path = os.path.join(base, "aggregated.npz")
        np.savez(out_path, **agg)
        print(f"  Saved → {out_path}")
        print_summary(algo.upper(), agg)

    print(f"\n{'='*50}")
    print("Done. Update the dashboard with:")
    print(f"  python export_training_data.py --scenario {args.scenario}")


if __name__ == "__main__":
    main()
