# DSN — NA²Q vs HiT-MAC

> **Multi-Agent Reinforcement Learning for Directional Sensor Networks**
>
> A comparative study of two cooperative MARL approaches — **NA²Q** (Neural Attention Additive Q-Learning, value decomposition) and **HiT-MAC** (Hierarchical Twin-Actor Multi-Agent Coordination, A3C-based) — on cooperative coverage tasks in Directional Sensor Networks (DSNs).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Scenarios](#scenarios)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training](#training)
- [Results](#results)
- [Dashboard](#dashboard)
- [Citation](#citation)

---

## Overview

Directional Sensor Networks consist of sensors with limited fields of view that must coordinate to maximise target coverage. This project benchmarks two fundamentally different MARL paradigms on this task across two environment scales.

Key findings and known training issues are documented in [`FINDINGS.md`](FINDINGS.md).

---

## Algorithms

| Algorithm | Paradigm | Architecture |
|-----------|----------|--------------|
| **NA²Q** | Value Decomposition (CTDE) | Attention-based additive Q-network with RNN |
| **HiT-MAC** | Hierarchical A3C | Coordinator + Executor with shared-memory async training |

---

## Scenarios

| | Scenario 1 | Scenario 2 |
|---|:---:|:---:|
| **Sensors** | 5 | 50 |
| **Targets** | 6 | 60 |
| **Grid** | Small | Large |
| **Runs (seeds)** | 5 | 1 |

---

## Project Structure

```
DSN_NA2Q/
├── environments/          # DSN Gymnasium environment
├── na2q/                  # NA²Q algorithm
│   ├── models/            # Q-network, attention, mixer
│   ├── engine/            # Trainer & data collector
│   ├── utils/             # Logger, replay buffer
│   ├── main.py            # Training entry point
│   └── checkpoints/       # Saved models & training history
├── hitmac/                # HiT-MAC algorithm
│   ├── models.py          # Coordinator + Executor networks
│   ├── train.py           # A3C worker training loop
│   ├── coordinator_train.py
│   ├── main.py            # Training entry point
│   └── checkpoints/       # Saved models & training history
├── docs/                  # Static results dashboard (HTML)
├── Result/                # Exported figures & plots
├── config.py              # Shared hyperparameter configuration
├── requirements.txt       # Python dependencies
├── FINDINGS.md            # Training notes & known issues
└── README.md
```

---

## Setup

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1

### CPU (Local)

```bash
git clone https://github.com/Chiseled141/DSN_NA2Q.git
cd DSN_NA2Q
pip install -r requirements.txt
```

### GPU (CUDA 12.1)

```bash
git clone https://github.com/Chiseled141/DSN_NA2Q.git
cd DSN_NA2Q
# Install PyTorch with CUDA support first
pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu121
# Then install remaining dependencies
pip install -r requirements.txt
```

---

## Training

### NA²Q

```bash
# Scenario 1
python -m na2q.main --mode train --scenario 1

# Scenario 2
python -m na2q.main --mode train --scenario 2 --device cuda
```

### HiT-MAC

```bash
# Scenario 1
python -m hitmac.main --mode train --scenario 1

# Scenario 2
python -m hitmac.main --mode train --scenario 2
```

### Checkpoint Structure

```
na2q/checkpoints/scenario{N}/
  best_model.pt          # Best checkpoint by validation reward
  final_model.pt         # Final checkpoint at end of training
  training_history.npz   # Full training metrics (rewards, coverage)
  training.log           # Human-readable training log

hitmac/checkpoints/scenario{N}/
  best.pt                # Best checkpoint by validation reward
  latest.pt              # Most recent checkpoint
  executor_final.pt      # Frozen executor after Phase 1
  training_history.npz   # Full training metrics
  aggregated.npz         # Multi-seed mean ± std (Scenario 1 only)
  training.log           # Human-readable training log
```

---

## Results

Figures and exported result data are saved under `Result/`:

```
Result/
├── scenario_1/
│   ├── na2q/            # Coverage & reward curves
│   └── hitmac/          # Coverage, reward & per-seed plots
├── scenario_2/
│   ├── na2q/
│   └── hitmac/
└── shared/              # Cross-algorithm comparison figures
```

---

## Dashboard

An interactive results dashboard is available in `docs/`:

```bash
cd docs && python -m http.server 8080
# Then open http://localhost:8080 in your browser
```

---

## Citation

If you use this codebase or build upon these results, please cite the original works:

```bibtex
@inproceedings{liu2023na2q,
  title={{NA$^2$Q}: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning},
  author={Liu, Zichuan and Zhu, Yuanyang and Chen, Chunlin},
  booktitle={ICML},
  year={2023}
}

@article{xu2020learning,
  title={Learning Multi-Agent Coordination for Enhancing Target Coverage in Directional Sensor Networks},
  author={Xu, Jing and Zhong, Fangwei and Wang, Yizhou},
  journal={NeurIPS},
  year={2020}
}
```
