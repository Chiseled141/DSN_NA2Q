# DSN Lab

Multi-agent reinforcement learning for Directional Sensor Networks — coordinate cameras to track targets and maximize coverage.

## Setup

```bash
git clone https://github.com/Chiseled141/DSN_NA2Q.git
cd DSN_NA2Q
pip install -r requirements.txt
```

## Usage

```bash
# Train
python main.py -a na2q   --mode train --scenario 1
python main.py -a hitmac --mode train --scenario 1

# Resume from checkpoint
python main.py -a na2q   --mode train --scenario 1 --resume
python main.py -a hitmac --mode train --scenario 1 --resume

# Test
python main.py -a na2q   --mode test --scenario 1
python main.py -a hitmac --mode test --scenario 1

# Run in background
nohup python main.py -a hitmac --mode train --scenario 1 > train.log 2>&1 &
```

## Scenarios

| | Scenario 1 | Scenario 2 |
|---|---|---|
| Sensors | 5 | 50 |
| Targets | 6 | 60 |
| Grid | 3×3 | 10×10 |

## Output

| File | Location |
|---|---|
| `best_model.pt` / `best.pt` | `[algo]/checkpoints/` |
| `training_history.npz` | `[algo]/checkpoints/` |
| `training.log` | `[algo]/checkpoints/` |
| Charts & GIFs | `Result/ScenarioX/` |

## Web Dashboard

```bash
cd docs && python -m http.server 8000
# open http://localhost:8000
```

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
