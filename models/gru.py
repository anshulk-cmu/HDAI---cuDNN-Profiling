"""Tiny GRU — memory-bound RNN reference point for the zoo."""
import torch
import torch.nn as nn


class TinyGRU(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


def get_model():
    return TinyGRU().eval().cuda()


def get_input(batch=32, seq=100):
    return torch.randn(batch, seq, 64, device='cuda')
