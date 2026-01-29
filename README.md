# DSN Lab

**Multi-agent Reinforcement Learning for Directional Sensor Networks.**  
Coordinate PTZ cameras to track targets and maximize coverage.

👉 **[View Full Documentation & Results](https://chiseled141.github.io/DSN_NA2Q/)**

## 🚀 Quick Start

### 1. Install
```bash
git clone https://github.com/Chiseled141/DSN_NA2Q.git
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

## 📂 Project Structure

*   `na2q/` — NA²Q algorithm implementation
*   `hitmac/` — HiT-MAC algorithm implementation
*   `environments/` — Shared DSN environment
*   `docs/` — Documentation website

---
&copy; 2026 DSN Lab Research Project.
