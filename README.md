# LM Training Framework

A modular framework for training transformer-based language models in PyTorch.

## Features

- Configurable transformer architecture
- Training and evaluation pipelines
- Data preprocessing utilities
- Support for causal and masked language modeling

## Quick Start

### Environment Setup

The easiest way to set up your environment is to use the included `setup.sh` script:

```bash
# Make the script executable
chmod +x setup.sh

# Basic setup with CPU
./setup.sh

# Setup with GPU support
./setup.sh --gpu

# Setup with development tools
./setup.sh --dev

# Setup with optimization libraries
./setup.sh --gpu --optim

# Customize environment name and Python version
./setup.sh --name custom-env --python 3.11 --gpu
```

### Manual Setup

If you prefer to set up manually:

1. Create a conda environment:
   ```bash
   conda create -y -n lm-training python=3.10
   conda activate lm-training
   ```

2. Install PyTorch (with or without GPU support):
   ```bash
   # With GPU
   conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # Without GPU
   conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
   ```

3. Install the package:
   ```bash
   # Basic installation
   pip install -e .
   
   # With development tools
   pip install -e ".[dev]"
   
   # With optimization libraries
   pip install -e ".[optim]"
   
   # With all extras
   pip install -e ".[all]"
   ```

## Training a Model

To train a small language model on WikiText:

1. Prepare the dataset:
   ```bash
   python scripts/prepare_wikitext_hf.py --config wikitext-2-v1
   ```

2. Train the model:
   ```bash
   python examples/train_small_lm.py --config_path examples/configs/wikitext_small.json
   ```

## Configuration

Model and training configurations are defined in JSON files. See `examples/configs/wikitext_small.json` for an example.

## Project Structure

- `lmtraining/`: Main package directory
  - `models/`: Transformer model components
  - `data/`: Data loading and preprocessing utilities
  - `training/`: Training and evaluation utilities
- `examples/`: Example scripts and configurations
- `scripts/`: Utility scripts for data preparation and other tasks

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See `setup.py` for a complete list of dependencies

## License

MIT