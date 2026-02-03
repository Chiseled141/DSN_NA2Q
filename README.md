# DSN Lab

**Multi-agent Reinforcement Learning for Directional Sensor Networks.**  
Coordinate PTZ cameras to track targets and maximize coverage.

**[View Full Documentation & Results](https://chiseled141.github.io/DSN_NA2Q/)**

## Quick Start

### 1. Install
**Clone this specific branch (`phu`):**
```bash
git clone -b phu https://github.com/Chiseled141/DSN_NA2Q.git
cd DSN_NA2Q
pip install -r requirements.txt
```

### 2. Run Algorithms

**Unified CLI** (choose algorithm with `-a`):
```bash
# NA²Q
python main.py -a na2q --mode train --scenario 1
python main.py -a na2q --mode video --scenario 1

# HiT-MAC
python main.py -a hitmac --mode train --scenario 1
python main.py -a hitmac --mode test --scenario 1
```

**Or run directly:**
```bash
python -m na2q.main --mode train --scenario 1
python -m hitmac.main --mode train --scenario 1
```

## Project Structure
```
DSN_NA2Q/
├── main.py                 # Unified CLI entry point
├── na2q/                   # NA²Q Algorithm
│   ├── checkpoints/        # Saved models & history
│   └── main.py
├── hitmac/                 # HiT-MAC Algorithm
│   ├── checkpoints/        # Saved models & history
│   └── main.py
├── environments/           # Shared DSN Environment
└── Result/                 # Training Results (Shared)
    └── Scenario1/
        ├── na2q_train_dashboard.png
        ├── hitmac_train_dashboard.png
        └── ...
```

## Results & Artifacts

| Artifact | Location | Description |
| :--- | :--- | :--- |
| **Charts** | `Result/ScenarioX/` | Dashboard plots (Rewards, Coverage, Loss) with algorithm prefix. |
| **GIFs** | `Result/ScenarioX/` | Video demonstrations of trained agents. |
| **Models** | `[algo]/checkpoints/` | `best.pt` (highest reward) and `latest.pt` (most recent). |
| **History** | `[algo]/checkpoints/` | `training_history.npz` containing raw metrics. |

## Web Dashboard

The project includes a state-of-the-art **Interactive Web Interface** for visualizing results and understanding the algorithms.

**Launch the dashboard:**
```bash
cd docs
python3 -m http.server 8000
```
Then visit: `http://localhost:8000`

### Key Features
- **Professional Analytics**: Real-time interactive charts with 10k+ episode tracking.
- **Unified Comparison**: Side-by-side performance metrics for **NA²Q** vs **HiT-MAC**.
- **Code Deep Dive**: IDE-style code walkthroughs explaining the internal logic (Agent, Q-Network, Mixer).
- **Precise Metrics**: Data labels and tooltips with 1-decimal precision and percent indicators.

## Citation
This repository implements and builds upon the research and codebases of the **HiT-MAC** and **NA²Q** algorithms, as well as the **Directional Sensor Network (DSN)** environment. If you use this work, please cite the original papers:

```bibtex
@article{xu2020learning,
  title={Learning Multi-Agent Coordination for Enhancing Target Coverage in Directional Sensor Networks},
  author={Xu, Jing and Zhong, Fangwei and Wang, Yizhou},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@inproceedings{liu2023na2q,
  title = {{NA$^2$Q}: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning},
  author = {Liu, Zichuan and Zhu, Yuanyang and Chen, Chunlin},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {22539--22558},
  year = {2023},
  volume = {202},
  series = {Proceedings of Machine Learning Research},
  month = {23--29 Jul},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v202/liu23be.html},
}
```


