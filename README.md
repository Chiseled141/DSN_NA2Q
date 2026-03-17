# DSN Lab

Multi-agent reinforcement learning for Directional Sensor Networks — coordinate PTZ cameras to track targets and maximize coverage.

**[View Results & Documentation](https://chiseled141.github.io/DSN_NA2Q/)**

---

## Install

```bash
git clone https://github.com/Chiseled141/DSN_NA2Q.git
cd DSN_NA2Q
pip install -r requirements.txt
```

---

## Train

**NA²Q**
```bash
python main.py -a na2q --mode train --scenario 1
```

**HiT-MAC**
```bash
python main.py -a hitmac --mode train --scenario 1
```

**Resume HiT-MAC from checkpoint**
```bash
python main.py -a hitmac --mode train --scenario 1 --resume
```

**Run in background**
```bash
nohup python main.py -a hitmac --mode train --scenario 1 > train.log 2>&1 &
```

---

## Test

```bash
python main.py -a na2q --mode test --scenario 1
python main.py -a hitmac --mode test --scenario 1
```

---

## Scenarios

| Scenario | Sensors | Targets | Grid |
| :--- | :--- | :--- | :--- |
| 1 (small) | 5 | 6 | 3×3 |
| 2 (large) | 50 | 60 | 10×10 |

---

## Output Files

| File | Location | What it is |
| :--- | :--- | :--- |
| `best.pt` | `[algo]/checkpoints/` | Best model (highest reward) |
| `latest.pt` | `[algo]/checkpoints/` | Most recent model |
| `training_history.npz` | `[algo]/checkpoints/` | Rewards, coverage, durations |
| Charts & GIFs | `Result/ScenarioX/` | Training plots and replays |

---

## Web Dashboard

```bash
cd docs && python -m http.server 8000
# open http://localhost:8000
```

Side-by-side NA²Q vs HiT-MAC comparison, training charts, and live benchmark.

---

## Citation

```bibtex
@article{xu2020learning,
  title={Learning Multi-Agent Coordination for Enhancing Target Coverage in Directional Sensor Networks},
  author={Xu, Jing and Zhong, Fangwei and Wang, Yizhou},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@inproceedings{liu2023na2q,
  title={{NA$^2$Q}: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning},
  author={Liu, Zichuan and Zhu, Yuanyang and Chen, Chunlin},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  pages={22539--22558},
  year={2023},
  publisher={PMLR}
}
```
