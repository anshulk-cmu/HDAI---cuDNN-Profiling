"""MobileNetV3-Small wrapper — depthwise-conv reference point for the zoo."""
import torch
import torchvision.models as tvm


def get_model():
    return tvm.mobilenet_v3_small(
        weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    ).eval().cuda()


def get_input(batch=32):
    return torch.randn(batch, 3, 224, 224, device='cuda')
