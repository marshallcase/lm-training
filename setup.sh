#!/bin/bash
# setup.sh - Complete environment setup for LM training project
set -e  # Exit on any error

# Default settings
ENV_NAME="lm-training"
PYTHON_VERSION="3.10"
WITH_GPU=false
WITH_DEV=false
WITH_OPTIM=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --gpu)
      WITH_GPU=true
      shift
      ;;
    --dev)
      WITH_DEV=true
      shift
      ;;
    --optim)
      WITH_OPTIM=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --name NAME       Name of conda environment (default: lm-training)"
      echo "  --python VERSION  Python version (default: 3.10)"
      echo "  --gpu             Install GPU-enabled PyTorch"
      echo "  --dev             Install development dependencies"
      echo "  --optim           Install optimization libraries"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for available options"
      exit 1
      ;;
  esac
done

echo "========== Setting up environment: $ENV_NAME =========="
echo "Python version: $PYTHON_VERSION"
echo "GPU support: $WITH_GPU"
echo "Development packages: $WITH_DEV"
echo "Optimization packages: $WITH_OPTIM"
echo "==================================================="

# Create conda environment
echo "Creating conda environment..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

# Define activation command and CUDA check for the different shells
if [[ "$SHELL" == */zsh ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    ACTIVATE_CMD="conda activate $ENV_NAME"
elif [[ "$SHELL" == */bash ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    ACTIVATE_CMD="conda activate $ENV_NAME"
else
    ACTIVATE_CMD="source activate $ENV_NAME"
fi

# Activate the environment
echo "Activating environment..."
eval "$ACTIVATE_CMD"

# Install PyTorch based on GPU preference
echo "Installing PyTorch..."
if [ "$WITH_GPU" = true ]; then
    # For CUDA 11.8 (adjust as needed for your CUDA version)
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
else
    conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
fi

# Install common conda packages
echo "Installing common packages via conda..."
conda install -y -c conda-forge ipython jupyter

# Set up installation flags for setup.py
SETUP_OPTS=""
if [ "$WITH_DEV" = true ] && [ "$WITH_OPTIM" = true ]; then
    SETUP_OPTS="[dev,optim]"
elif [ "$WITH_DEV" = true ]; then
    SETUP_OPTS="[dev]"
elif [ "$WITH_OPTIM" = true ]; then
    SETUP_OPTS="[optim]"
fi

# Install the package
echo "Installing the package in development mode..."
pip install -e ."$SETUP_OPTS"

# Install datasets - required according to the conda_setup.py
echo "Installing datasets package..."
pip install "datasets>=2.12.0"

# We're removing the spaCy model download as it's not required
# for the core functionality of the language model training

echo ""
echo "========== Environment setup complete =========="
echo "Environment: $ENV_NAME"
echo ""
echo "To activate this environment, run:"
echo "    $ACTIVATE_CMD"
echo ""
echo "To test your installation, try running:"
echo "    python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")"'
echo ""