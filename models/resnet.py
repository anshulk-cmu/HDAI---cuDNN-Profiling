"""ResNet-18 wrapper for the cuDNN profiling study."""
import torch
import torchvision.models as tvm


def get_model():
    return tvm.resnet18(
        weights=tvm.ResNet18_Weights.IMAGENET1K_V1
    ).eval().cuda()


def get_input(batch=32):
    return torch.randn(batch, 3, 224, 224, device='cuda')
