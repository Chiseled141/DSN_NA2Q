# Report Preparation Guide

This file documents what data exists, what still needs to run, and how to generate
all figures and tables for the paper.

---

## Current Status

| Experiment | Status | Location |
|---|---|---|
| NA²Q — Scenario 1 (5 seeds) | ✅ Done | `na2q/checkpoints/scenario1/` |
| HiT-MAC — Scenario 1 (5 seeds) | ✅ Done | `hitmac/checkpoints/scenario1/run{0-4}/` |
| NA²Q — Scenario 2 (1 seed) | ⏳ Not yet trained | — |
| HiT-MAC — Scenario 2 (1 seed) | ⏳ Not yet trained | — |

---

## Training Commands

```bash
# Scenario 2 — run both on a GPU machine
python3 main.py --algorithm na2q   --mode train --scenario 2
python3 main.py --algorithm hitmac --mode train --scenario 2
```

Expected time on GPU: ~30–60 min each.

---

## After Training Completes

### Step 1 — Aggregate scenario 1 results (NA²Q, 5 seeds)

NA²Q scenario 1 was run as a single long run, not with `run_experiments.py`.
The `training_history.npz` is already at `na2q/checkpoints/scenario1/`.
No aggregation needed — `export_training_data.py` reads it directly.

### Step 2 — Export dashboard data

```bash
# After scenario 2 training is done:
python3 export_training_data.py --scenario 1
python3 export_training_data.py --scenario 2
```

This writes `docs/data/training_data.js` (loaded by the web dashboard).

### Step 3 — Check the dashboard

Open `docs/index.html` in a browser to verify all charts look correct before
copying numbers into the report.

---

## Key Numbers for the Report

Run this to get all table values:

```bash
python3 -c "
import numpy as np

def summarize(path, label, n_tail=500):
    try:
        d = np.load(path, allow_pickle=True)
        if 'coverage_mean' in d:   # aggregated
            cov = d['coverage_mean']
            std = d['coverage_std']
            scale = 100 if cov.max() <= 1 else 1
            print(f'{label}: {cov[-n_tail:].mean()*scale:.1f}% ± {std[-n_tail:].mean()*scale:.1f}% (best {cov.max()*scale:.1f}%, {int(d[\"n_seeds\"])} seeds)')
        else:
            cov = d['coverage_rates']
            scale = 100 if cov.max() <= 1 else 1
            rb = float(d.get('random_baseline_mean', 0)) * scale
            gb = float(d.get('greedy_baseline_mean', 0)) * scale
            print(f'{label}: {cov[-n_tail:].mean()*scale:.1f}% (best {cov.max()*scale:.1f}%) | random={rb:.1f}% greedy={gb:.1f}%')
    except: print(f'{label}: NOT FOUND')

summarize('na2q/checkpoints/scenario1/training_history.npz',      'NA2Q   S1 (1 seed) ')
summarize('hitmac/checkpoints/scenario1/aggregated.npz',          'HiT-MAC S1 (5 seeds)')
summarize('na2q/checkpoints/scenario2/training_history.npz',      'NA2Q   S2 (1 seed) ')
summarize('hitmac/checkpoints/scenario2/training_history.npz',    'HiT-MAC S2 (1 seed) ')
"
```

---

## Report Structure (Suggested)

### Abstract
- Multi-agent sensor coverage problem
- NA²Q achieves X% vs HiT-MAC Y% on scenario 1
- Scales to 50-agent scenario 2

### 1. Introduction
- Directional sensor networks (DSN)
- Target tracking coordination challenge
- Contributions: NA²Q algorithm + large-scale evaluation

### 2. Problem Formulation
- Grid environment: `n_sensors` agents, `n_targets` targets
- Observation: `[sensor_id, target_id, dist_norm, angle_norm]` × n_targets per agent
- Action space: Discrete(3) — TurnLeft / Stay / TurnRight
- Reward: coverage rate with angle-centering bonus

### 3. Methods
- **NA²Q**: Neural Attention Additive Q-Learning + QMIX mixer with Shapley values
- **HiT-MAC**: A3C executor with attention encoder, single-phase training

### 4. Experiments

**Table 1 — Scenario 1 (5 sensors, 6 targets, 3×3 grid)**

| Method | Final Coverage (last 500 ep) | Best Coverage | Baseline |
|---|---|---|---|
| Random | ~31% | — | — |
| Greedy (Hungarian) | ~65% | — | — |
| HiT-MAC (5 seeds) | X% ± Y% | Z% | — |
| NA²Q (1 seed) | X% | Z% | — |

**Table 2 — Scenario 2 (50 sensors, 60 targets, 10×10 grid)**

| Method | Final Coverage | Best Coverage | Random BL | Greedy BL |
|---|---|---|---|---|
| HiT-MAC (1 seed) | X% | Z% | 56.0% | 80.1% |
| NA²Q (1 seed) | X% | Z% | 56.0% | 80.1% |

**Figure 1** — Learning curves (coverage vs episodes) for scenario 1, both algorithms  
**Figure 2** — Learning curves for scenario 2  
**Figure 3** — Comparison: scenario 1 vs scenario 2 scalability

### 5. Results & Discussion
- Scenario 1: Which algorithm converges faster / reaches higher coverage
- Scenario 2: Both algorithms scale — cite that 1 seed is used due to compute cost
  > *"Due to the computational cost of scenario 2 (50 sensors, 60 targets), we report single-seed results. Scenario 1 uses 5 seeds to establish statistical significance."*

### 6. Conclusion

---

## Figures from the Dashboard

The web dashboard at `docs/index.html` already renders:
- Coverage curves per scenario
- Reward curves
- Baseline comparison lines

To export as images: open in browser → right-click chart → Save image, or use
the browser's screenshot tool.

---

## Notes for Collaborators

- All training history is in `**/checkpoints/scenario{N}/training_history.npz`
- Keys: `episode_rewards`, `coverage_rates`, `random_baseline_mean`, `greedy_baseline_mean`
- HiT-MAC S1 uses `aggregated.npz` (mean ± std across 5 seeds) — use `coverage_mean` / `coverage_std`
- NA²Q S1 is a single run — use `coverage_rates` directly
- Scenario 2: 1 seed each, justify in paper with compute argument
- Do NOT re-run scenario 1 — existing 5-seed results are final
