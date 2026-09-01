#!/bin/bash
# ============================================================================
# Environment setup for AutoMLR
#
# Creates (or updates) a conda environment with all dependencies.
# Usage:
#   bash setup_env.sh                         # creates 'automlr' env
#   ENV_NAME=myenv bash setup_env.sh          # custom env name
#   CONDA_PREFIX=/scratch/.../envs bash setup_env.sh  # custom location
# ============================================================================

set -eo pipefail

ENV_NAME="${ENV_NAME:-automlr}"
CONDA_PREFIX="${CONDA_PREFIX:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== AutoMLR environment setup ==="

# Load conda if on SLURM cluster
if command -v module &>/dev/null; then
  module load anaconda3/2024.10-1 2>/dev/null || true
fi
if [[ -f /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh ]]; then
  source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
fi

# Create or reuse environment
if [[ -n "$CONDA_PREFIX" ]]; then
  ENV_PATH="$CONDA_PREFIX/$ENV_NAME"
  if [[ ! -d "$ENV_PATH" ]]; then
    echo "Creating conda env at $ENV_PATH ..."
    conda create -p "$ENV_PATH" python=3.10 -y
  fi
  conda activate "$ENV_PATH"
else
  if ! conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Creating conda env '$ENV_NAME' ..."
    conda create -n "$ENV_NAME" python=3.10 -y
  fi
  conda activate "$ENV_NAME"
fi

echo "Installing PyTorch (CUDA 12.1) ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing remaining dependencies ..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=== Setup complete ==="
echo "Activate with:  conda activate $ENV_NAME"
echo "Verify:         python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
