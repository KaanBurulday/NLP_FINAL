# import torch
import torch_directml

# print(dml)

import torch

# Create a tensor on the DirectML device
# x = torch.tensor([1, 2, 3], device=dml)
# print(x.device)  # Output: privateuseone:0
dev = torch.device(0)
print(dev)
try:
    dml = torch_directml.device()

    # Test tensor creation on the DML device
    x = torch.randn(3, 3, device=dml)
    print(f"Tensor on DML device: \n{x}")
except RuntimeError as e:
    print(f"Error: {e}")


def check_amd_gpu():
    if torch.cuda.is_available():
        # Check for CUDA GPUs
        print("CUDA is available, which is typically for NVIDIA GPUs.")
    elif torch.backends.mps.is_available():
        # Check for Metal Performance Shaders (MPS), typically for macOS
        print("MPS is available. AMD GPUs may use this on macOS.")
    elif torch.backends.hip.is_available():
        # Check for ROCm/HIP support for AMD GPUs
        print("ROCm/HIP is available. This indicates AMD GPU support.")
        print(f"Number of AMD GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("No GPU is detected, or it's not supported by PyTorch.")

check_amd_gpu()