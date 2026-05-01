#!/usr/bin/env bash
# Cloud GPU setup script for DSN_NA2Q
# Run once after cloning on your cloud instance (Colab, RunPod, Lambda, vast.ai, etc.)
#
#   bash setup_cloud.sh

set -e

echo "=== DSN_NA2Q Cloud Setup ==="

# 1. Install CUDA-compatible PyTorch first
echo "[1/3] Installing PyTorch (CUDA 12.1)..."
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121 -q

# 2. Install remaining dependencies
echo "[2/3] Installing other dependencies..."
pip install numpy scipy gymnasium tqdm matplotlib tensorboard jinja2 -q

# 3. Quick sanity check
echo "[3/3] Sanity check..."
python - <<'EOF'
import torch, numpy, gymnasium, tqdm
print(f"  torch  : {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
print(f"  numpy  : {numpy.__version__}")
print(f"  gym    : {gymnasium.__version__}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "=== Setup complete. Run experiments with: ==="
echo "  python run_experiments.py --scenario 1 --algorithms na2q hitmac --seeds 5"
echo ""
echo "=== After training, export results: ==="
echo "  python aggregate_results.py --scenario 1 --seeds 5"
echo "  python export_training_data.py --scenario 1"
echo "  python generate_report.py --scenario 1 --seeds 5"
