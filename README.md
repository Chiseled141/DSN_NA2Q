# DSN Lab

Multi-agent RL algorithms for Directional Sensor Networks.

## Algorithms

- **NA²Q** — Neural Attention Additive Q-Learning
- **HiT-MAC** — Hierarchical Multi-Agent Coordination (NeurIPS 2020)

## Quick Start

```bash
pip install -r requirements.txt

# NA²Q
python -m na2q.main --mode train --scenario 1

# HiT-MAC
python -m hitmac.main --env Pose-v0 --model single-att --workers 6
```

## Config

See `config.py` for training settings.
