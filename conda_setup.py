#!/usr/bin/env python
import argparse
import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True)
    process.communicate()
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}")
        sys.exit(process.returncode)

def setup_conda_env(env_name, python_version, gpu, dev, optim):
    # Create conda environment
    run_command(f"conda create -y -n {env_name} python={python_version}")
    
    # Activate environment in current script
    activate_cmd = f"conda activate {env_name}"
    print(f"To activate your environment, run: {activate_cmd}")
    
    # Determine PyTorch installation command based on GPU presence
    if gpu:
        # For CUDA 11.8 (adjust as needed for your CUDA version)
        pytorch_install = "conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    else:
        pytorch_install = "conda install -y pytorch torchvision torchaudio cpuonly -c pytorch"
    
    print("\nPlease run the following commands in your terminal:")
    print(f"1. {activate_cmd}")
    print(f"2. {pytorch_install}")
    
    # Install package in development mode
    setup_options = []
    if dev:
        setup_options.append("dev")
    if optim:
        setup_options.append("optim")
    
    setup_cmd = "pip install -e ."
    if setup_options:
        setup_cmd = f"pip install -e .[{','.join(setup_options)}]"
    
    print(f"3. {setup_cmd}")
    
    # Install additional conda packages that might be better via conda than pip
    print("4. conda install -y -c conda-forge ipython jupyter")
    
    # Optional: Install spaCy models
    if "spacy" in open("setup.py").read():
        print("5. python -m spacy download en_core_web_sm")

    print("6. pip install datasets>=2.12.0")
    
    print("\nAfter running these commands, your environment will be ready!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup conda environment for language model training")
    parser.add_argument("--name", default="lm-training", help="Name of the conda environment")
    parser.add_argument("--python", default="3.10", help="Python version")
    parser.add_argument("--gpu", action="store_true", help="Install GPU-enabled PyTorch")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--optim", action="store_true", help="Install optimization libraries")
    
    args = parser.parse_args()
    setup_conda_env(args.name, args.python, args.gpu, args.dev, args.optim)