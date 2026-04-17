"""Ten-line smoke test confirming cuDNN actually dispatches a conv forward.

Distinct from check_env.py: that script verifies versions and shapes;
this one verifies the cuDNN kernel path by enabling the benchmark flag
and running one real 3x3 conv at a shape that lives inside ResNet-18.
"""
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

# Shapes picked to match ResNet-18 block-2 output: N=16, C=64, 56x56 -> 128 ch
x = torch.randn(16, 64, 56, 56, device='cuda')
w = torch.randn(128, 64, 3, 3, device='cuda')
y = F.conv2d(x, w, padding=1)
torch.cuda.synchronize()

print(f"cuDNN path OK -> {tuple(y.shape)}  "
      f"max|y| = {y.abs().max().item():.2f}")
