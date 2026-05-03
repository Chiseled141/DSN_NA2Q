# DSN — NA²Q vs HiT-MAC

Multi-agent reinforcement learning for Directional Sensor Networks. Compares NA²Q (value decomposition) against HiT-MAC (hierarchical A3C) on cooperative coverage tasks.

See `FINDINGS.md` for known training issues and result notes.

---

## Setup

### Local (CPU)
```bash
git clone https://github.com/Chiseled141/DSN_NA2Q.git
cd DSN_NA2Q
pip install -r requirements.txt
```

### Cloud GPU
```bash
git clone https://github.com/Chiseled141/DSN_NA2Q.git
cd DSN_NA2Q
bash setup_cloud.sh
```

---

## Training

### Scenario 1 — 5 seeds (for report)
```bash
python run_experiments.py --scenario 1 --algorithms na2q hitmac --seeds 5
```

### Scenario 2 — single run
```bash
python -m na2q.main  --mode train --scenario 2 --device cuda
python -m hitmac.main --mode train --scenario 2
```

### Single run (test / debug)
```bash
python -m na2q.main  --mode train --scenario 1 --device cuda
python -m hitmac.main --mode train --scenario 1
```

---

## After Training

```bash
# Aggregate 5 seeds into mean ± std
python aggregate_results.py --scenario 1 --seeds 5

# Export to dashboard
python export_training_data.py --scenario 1
python export_training_data.py --scenario 2
```

---

## Dashboard

```bash
cd docs && python -m http.server 8080
# open http://localhost:8080
```

---

## Scenarios

| | Scenario 1 | Scenario 2 |
|---|---|---|
| Sensors | 5 | 50 |
| Targets | 6 | 60 |

---

## Checkpoint Structure

```
na2q/checkpoints/scenario{N}/
  training_history.npz     # single run
  run0/ … run4/            # multi-seed runs
  aggregated.npz           # mean ± std (after aggregate_results.py)

hitmac/checkpoints/scenario{N}/
  training_history.npz
  executor_final.pt
  run0/ … run4/
  aggregated.npz
```

---

## Citation

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
