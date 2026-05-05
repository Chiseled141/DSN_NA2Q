# Result/

All figures used in the SS2 report, organised by scenario and algorithm.

```
Result/
├── scenario_1/
│   ├── na2q/                NA²Q-only Scenario 1 charts
│   ├── hitmac/              HiT-MAC-only Scenario 1 charts
│   └── combined/            Both algorithms overlaid
└── scenario_2/
    ├── hitmac/              HiT-MAC-only Scenario 2 charts
    ├── combined/            Scenario 2 cross-algorithm view
    └── na2q/                NA²Q not trained on S2 (see NOT_TRAINED.txt)
```

---

## scenario_1/

### na2q/
| File | Description |
|---|---|
| `coverage_curve.png` | NA²Q coverage learning curve (1 seed, 25,000 episodes) |
| `reward_curve.png` | NA²Q episode-reward curve |

### hitmac/
| File | Description |
|---|---|
| `coverage_curve.png` | HiT-MAC coverage curve — 5-seed mean ± SD |
| `reward_curve.png` | HiT-MAC episode-reward curve — 5-seed mean ± SD |
| `perseed_with_collapse.png` | All 5 seeds shown individually, including the seed-4 A3C collapse |

### combined/
| File | Report Figure | Description |
|---|:---:|---|
| `coverage_curves_both.png` | Fig 4 | NA²Q + HiT-MAC coverage on the same axes |
| `reward_curves_both.png` | Fig 5 | NA²Q + HiT-MAC reward curves |
| `sample_efficiency.png` | Fig 6 | Episodes-to-threshold bar chart |
| `coverage_distribution.png` | Fig 7 | Late-training per-episode coverage histogram |
| `wallclock.png` | Fig 8 | Coverage vs cumulative wall-clock time on GPU |

---

## scenario_2/

> NA²Q was not trained on Scenario 2. See `na2q/NOT_TRAINED.txt`.

### hitmac/
| File | Description |
|---|---|
| `coverage_curve.png` | HiT-MAC S2 coverage (1 seed, ~5,000 episodes) |
| `reward_curve.png` | HiT-MAC S2 reward curve |

### combined/
| File | Report Figure | Description |
|---|:---:|---|
| `coverage_curve_hitmac_only.png` | Fig 9 | HiT-MAC S2 coverage (NA²Q not available) |

---

## Source Data

All charts are generated from `.npz` training-history files:

| File | Used for |
|---|---|
| `na2q/checkpoints/scenario1/training_history.npz` | NA²Q S1 curves |
| `hitmac/checkpoints/scenario1/aggregated.npz` | HiT-MAC S1 mean ± SD (5 seeds) |
| `hitmac/checkpoints/scenario2/training_history.npz` | HiT-MAC S2 curves |

Smoothing window: **500 episodes** for Scenario 1, **200 episodes** for Scenario 2.
