import torch

# Check for DirectML support
try:
    device = torch.device('dml')  # DirectML device
    print(f"Using device: {device}")

    # Test a simple tensor computation on the device
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = x + 1
    print(y)
except RuntimeError as e:
    print(f"Error: {e}")