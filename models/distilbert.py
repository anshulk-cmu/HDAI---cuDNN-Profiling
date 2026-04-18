"""DistilBERT-base wrapper — cuBLAS/matmul reference point for the zoo."""
import torch
from transformers import DistilBertModel


def get_model():
    return DistilBertModel.from_pretrained(
        'distilbert-base-uncased'
    ).eval().cuda()


def get_input(batch=8, seq=128):
    # Fake token IDs in a valid range for distilbert-base-uncased's 30522-token vocab.
    return torch.randint(0, 30000, (batch, seq), device='cuda')
