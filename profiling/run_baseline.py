"""Baseline profile of a single model. Saves chrome trace + top-kernel table."""
import argparse
import os
import torch
from torch.profiler import profile, ProfilerActivity, schedule


def load_model_and_input(name, batch=32):
    if name == 'resnet18':
        from models.resnet import get_model, get_input
        return get_model(), get_input(batch)
    raise ValueError(f"Unknown model: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--benchmark', action='store_true', default=True)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = args.benchmark
    model, x = load_model_and_input(args.model, args.batch)

    for _ in range(30):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    os.makedirs('results/traces', exist_ok=True)
    trace_path = f'results/traces/{args.model}_baseline.json'

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        schedule=schedule(wait=1, warmup=2, active=10, repeat=1),
    ) as prof:
        for _ in range(15):
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    prof.export_chrome_trace(trace_path)
    print(f"\nTrace saved to {trace_path}")


if __name__ == '__main__':
    main()
