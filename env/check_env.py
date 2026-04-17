"""Verify hardware, PyTorch, cuDNN are set up correctly for Blackwell (sm_120)."""
import sys
import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA (PyTorch-linked): {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Check driver and wheel.")
    sys.exit(1)

print(f"Device: {torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability(0)
print(f"Compute capability: sm_{cap[0]}{cap[1]}")

if cap != (12, 0):
    print(f"WARNING: Expected sm_120 (Blackwell). Got sm_{cap[0]}{cap[1]}.")

x = torch.randn(1024, 1024, device='cuda')
y = torch.matmul(x, x.T)
torch.cuda.synchronize()
print(f"Matmul smoke test: output shape {y.shape}, max = {y.max().item():.2f}")

import torch.nn.functional as F
a = torch.randn(16, 64, 56, 56, device='cuda')
w = torch.randn(128, 64, 3, 3, device='cuda')
b = F.conv2d(a, w, padding=1)
torch.cuda.synchronize()
print(f"cuDNN conv smoke test: output shape {b.shape}")

print("\nAll checks passed.")
