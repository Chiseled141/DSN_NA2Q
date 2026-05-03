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

    def _arr(key, default=None):
        return np.array(data[key]) if key in data else default

    def _scalar(key):
        return float(data[key]) if key in data and data[key].ndim == 0 else None

    return {
        "rewards":              _arr("episode_rewards"),
        "coverage":             _arr("coverage_rates"),
        "wall_clock_times":     _arr("wall_clock_times"),
        "eval_episodes":        _arr("eval_episodes"),
        "eval_coverages":       _arr("eval_coverages"),
        "eval_rewards":         _arr("eval_rewards"),
        "random_baseline_mean": _scalar("random_baseline_mean"),
        "random_baseline_std":  _scalar("random_baseline_std"),
        "greedy_baseline_mean": _scalar("greedy_baseline_mean"),
        "greedy_baseline_std":  _scalar("greedy_baseline_std"),
    }


def interpolate_to_grid(values: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 1-D array to target_len points via linear interpolation."""
    src_x = np.linspace(0, 1, len(values))
    dst_x = np.linspace(0, 1, target_len)
    return np.interp(dst_x, src_x, values)


def aggregate(algo: str, scenario: int, n_seeds: int, base_dir: str, phase1_only: bool = False):
    """Load all runs for an algorithm, interpolate, return mean ± std arrays."""
    runs_rewards  = []
    runs_coverage = []
    runs_wct      = []
    rand_means, greedy_means = [], []

    for k in range(n_seeds):
        run_dir = os.path.join(base_dir, f"run{k}")
        run = load_run(run_dir)
        if run is None:
            print(f"  ⚠  {algo} run{k} not found — skipping")
            continue
        rewards  = run["rewards"]
        coverage = run["coverage"]
        # For HiT-MAC, Phase 1 = first half of episodes (Phase 2 degrades performance)
        if phase1_only and len(rewards) > 0:
            half = len(rewards) // 2
            rewards  = rewards[:half]
            coverage = coverage[:half]
        runs_rewards.append(rewards)
        runs_coverage.append(coverage)
        if run["wall_clock_times"] is not None and len(run["wall_clock_times"]) > 0:
            wct = run["wall_clock_times"]
            if phase1_only:
                wct = wct[:len(wct) // 2]
            runs_wct.append(wct)
        if run["random_baseline_mean"] is not None:
            rand_means.append(run["random_baseline_mean"])
        if run["greedy_baseline_mean"] is not None:
            greedy_means.append(run["greedy_baseline_mean"])
        print(f"  {algo} run{k}: {len(rewards)} episodes (phase1_only={phase1_only})")

    if not runs_rewards:
        print(f"  No completed runs found for {algo} scenario {scenario}")
        return None

    # Interpolate all runs to the length of the shortest run
    min_len  = min(len(r) for r in runs_rewards)
    rew_grid = np.stack([interpolate_to_grid(r, min_len) for r in runs_rewards])
    cov_grid = np.stack([interpolate_to_grid(c, min_len) for c in runs_coverage])

    result = {
        "episodes":      np.arange(min_len),
        "reward_mean":   rew_grid.mean(axis=0),
        "reward_std":    rew_grid.std(axis=0),
        "reward_min":    rew_grid.min(axis=0),
        "reward_max":    rew_grid.max(axis=0),
        "coverage_mean": cov_grid.mean(axis=0),
        "coverage_std":  cov_grid.std(axis=0),
        "coverage_min":  cov_grid.min(axis=0),
        "coverage_max":  cov_grid.max(axis=0),
        "n_seeds":       len(runs_rewards),
    }

    # Wall-clock time
    valid_wct = [w for w in runs_wct if len(w) > 0]
    if valid_wct:
        wct_grid = np.stack([interpolate_to_grid(w, min_len) for w in valid_wct])
        result["wall_clock_mean"] = wct_grid.mean(axis=0)

    # Baselines (mean across seeds)
    if rand_means:
        result["random_baseline_mean"] = float(np.mean(rand_means))
        result["random_baseline_std"]  = float(np.std(rand_means))
    if greedy_means:
        result["greedy_baseline_mean"] = float(np.mean(greedy_means))
        result["greedy_baseline_std"]  = float(np.std(greedy_means))

    # Coverage at fixed milestones (1k, 5k, 10k episodes)
    milestones = [1000, 5000, 10000]
    for m in milestones:
        if m <= min_len:
            result[f"coverage_at_{m}ep"] = float(cov_grid[:, m - 1].mean())

    return result


def print_summary(name: str, agg: dict):
    """Print paper-table style summary for the last 500 episodes."""
    cov  = agg["coverage_mean"]
    std  = agg["coverage_std"]
    n    = len(cov)
    sc   = 100.0 if cov.max() <= 1.0 else 1.0

    tail_cov = cov[max(0, n - 500):] * sc
    tail_std = std[max(0, n - 500):] * sc
    best     = cov.max() * sc

    # Convergence: first episode where 100-ep rolling mean >= 50%
    conv_ep = None
    for i in range(99, n):
        if cov[i - 99: i + 1].mean() * sc >= 50.0:
            conv_ep = i
            break

    print(f"\n  {name}")
    print(f"    Seeds:              {agg['n_seeds']}")
    print(f"    Final coverage:     {tail_cov.mean():.1f}% ± {tail_std.mean():.1f}%")
    print(f"    Best mean coverage: {best:.1f}%")
    print(f"    Ep. to 50% cov:     {conv_ep if conv_ep else 'N/A'}")

    if "random_baseline_mean" in agg:
        rb = agg["random_baseline_mean"] * 100 if agg["random_baseline_mean"] < 1.0 else agg["random_baseline_mean"]
        print(f"    Random baseline:    {rb:.1f}%")
    if "greedy_baseline_mean" in agg:
        gb = agg["greedy_baseline_mean"] * 100 if agg["greedy_baseline_mean"] < 1.0 else agg["greedy_baseline_mean"]
        print(f"    Greedy baseline:    {gb:.1f}%")
    for m in [1000, 5000, 10000]:
        key = f"coverage_at_{m}ep"
        if key in agg:
            v = agg[key] * 100 if agg[key] < 1.0 else agg[key]
            print(f"    Coverage @ {m:>5}ep: {v:.1f}%")


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
        agg = aggregate(algo, args.scenario, args.seeds, base,
                        phase1_only=(algo == "hitmac"))
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
