# gpu_diagnostics.py
import torch
import os
import sys

def check_gpu():
    print("\n===== PyTorch GPU Diagnostic Report =====")
    
    # Check PyTorch installation and version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("CUDA is not available. Please check your PyTorch installation.")
        return False
    
    # Check CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
    
    # Check number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    
    if gpu_count == 0:
        print("No GPUs detected by PyTorch.")
        return False
    
    # Display GPU information
    print("\nGPU Information:")
    for i in range(gpu_count):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Compute capability: {torch.cuda.get_device_capability(i)}")
        print(f"  - Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"  - Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    
    # Test creating a tensor on GPU and performing an operation
    print("\nCreating test tensor on GPU...")
    try:
        x = torch.ones(1000, 1000, device="cuda")
        y = x + x
        result = y.sum().item()
        print(f"Test tensor operation successful. Sum: {result}")
    except Exception as e:
        print(f"Error creating tensor on GPU: {e}")
        return False
    
    # Show available memory after test
    print(f"Memory allocated after test: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    return True

if __name__ == "__main__":
    check_gpu()