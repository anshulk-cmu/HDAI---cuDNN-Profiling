"""Baseline profile of a single model.

Produces two artefacts per run:
  1. A chrome-trace JSON from PyTorch Profiler (10 iterations).
  2. A latency distribution (mean, std, min, max) over >= 5 CUDA-event-timed trials.

The trace filename encodes (model, batch, benchmark state) so multiple
configurations can coexist in results/traces/ without overwriting each other.
"""
import argparse
import os
import statistics
import torch
from torch.profiler import profile, ProfilerActivity, schedule

try:
    import nvtx
except ImportError:
    from contextlib import nullcontext
    class _NvtxShim:
        def annotate(self, *a, **k):
            return nullcontext()
    nvtx = _NvtxShim()


def _load_resnet18(batch):
    from models.resnet import get_model, get_input
    return get_model(), get_input(batch)


def _load_mobilenetv3(batch):
    from models.mobilenet import get_model, get_input
    return get_model(), get_input(batch)


def _load_distilbert(batch):
    from models.distilbert import get_model, get_input
    return get_model(), get_input(batch)


def _load_gru(batch):
    from models.gru import get_model, get_input
    return get_model(), get_input(batch)


MODEL_LOADERS = {
    'resnet18':    _load_resnet18,
    'mobilenetv3': _load_mobilenetv3,
    'distilbert':  _load_distilbert,
    'gru':         _load_gru,
}


# Per-model default batch when --batch is not explicitly provided on the CLI.
# DistilBERT's smaller default reflects its 66M-parameter footprint under the
# 12GB VRAM cap (brief §1.3); the other three scale fine at 32.
DEFAULT_BATCH = {
    'resnet18':    32,
    'mobilenetv3': 32,
    'distilbert':  8,
    'gru':         32,
}


def load_model_and_input(name, batch):
    if name not in MODEL_LOADERS:
        raise ValueError(
            f"Unknown model: {name!r}. Known: {sorted(MODEL_LOADERS)}")
    return MODEL_LOADERS[name](batch)


def _print_flags(args):
    cap = torch.cuda.get_device_capability(0)
    print(f"[flags] device               = {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    print(f"[flags] cudnn.version        = {torch.backends.cudnn.version()}")
    print(f"[flags] cudnn.enabled        = {torch.backends.cudnn.enabled}")
    print(f"[flags] cudnn.benchmark      = {torch.backends.cudnn.benchmark}")
    print(f"[flags] cudnn.allow_tf32     = {torch.backends.cudnn.allow_tf32}")
    print(f"[flags] matmul.allow_tf32    = {torch.backends.cuda.matmul.allow_tf32}")
    print(f"[flags] model/batch          = {args.model} / bs{args.batch}")
    print(f"[flags] trials x iters       = {args.trials} x {args.iters_per_trial}")
    print(f"[flags] warmup               = {args.warmup}")


def time_trials(model, x, n_trials=7, iters_per_trial=50):
    """Return per-trial mean-ms-per-iter as a list of length n_trials.

    Each trial runs iters_per_trial forwards under a single pair of
    cuda.Event markers, then divides. Warm-up is handled by the caller.
    """
    per_trial_means = []
    for _ in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters_per_trial):
            with torch.no_grad():
                _ = model(x)
        end.record()
        torch.cuda.synchronize()
        per_trial_means.append(start.elapsed_time(end) / iters_per_trial)
    return per_trial_means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--batch', type=int, default=None,
                    help="Batch size. Defaults to DEFAULT_BATCH[model] if omitted.")
    ap.add_argument(
        '--benchmark', action=argparse.BooleanOptionalAction, default=True,
        help="Enable torch.backends.cudnn.benchmark (default: on). "
             "Use --no-benchmark to disable.")
    ap.add_argument('--trials', type=int, default=7)
    ap.add_argument('--iters-per-trial', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=30)
    args = ap.parse_args()

    if args.batch is None:
        args.batch = DEFAULT_BATCH[args.model]

    torch.backends.cudnn.benchmark = args.benchmark
    _print_flags(args)
    model, x = load_model_and_input(args.model, args.batch)

    warmup_start = torch.cuda.Event(enable_timing=True)
    warmup_end = torch.cuda.Event(enable_timing=True)
    with nvtx.annotate("warmup", color="grey"):
        warmup_start.record()
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = model(x)
        warmup_end.record()
        torch.cuda.synchronize()
    warmup_ms = warmup_start.elapsed_time(warmup_end)
    print(f"[warmup] total_ms = {warmup_ms:.2f}  "
          f"({args.warmup} iters, {warmup_ms / args.warmup:.2f} ms/iter avg)")

    with nvtx.annotate("cuda_event_timing", color="blue"):
        trial_means = time_trials(
            model, x,
            n_trials=args.trials,
            iters_per_trial=args.iters_per_trial,
        )
    mean_ms = statistics.mean(trial_means)
    std_ms = statistics.stdev(trial_means) if len(trial_means) > 1 else 0.0
    min_ms = min(trial_means)
    max_ms = max(trial_means)

    print(f"\nLatency (ms / iter at batch {args.batch}, "
          f"{args.trials} trials x {args.iters_per_trial} iters, "
          f"benchmark={args.benchmark}):")
    print(f"  mean = {mean_ms:.3f}  std = {std_ms:.3f}  "
          f"min = {min_ms:.3f}  max = {max_ms:.3f}")
    print(f"  per-trial: " + "  ".join(f"{t:.3f}" for t in trial_means))
    throughput = args.batch / (mean_ms / 1000.0)
    print(f"  throughput = {throughput:.1f} samples/sec")

    os.makedirs('results/traces', exist_ok=True)
    tag_bench = 'benchOn' if args.benchmark else 'benchOff'
    trace_path = (f'results/traces/{args.model}_baseline'
                  f'_bs{args.batch}_{tag_bench}.json')

    with nvtx.annotate("profiler_capture", color="green"):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=schedule(wait=1, warmup=2, active=10, repeat=1),
        ) as prof:
            for i in range(15):
                with nvtx.annotate(f"iter_{i:02d}", color="cyan"):
                    with torch.no_grad():
                        _ = model(x)
                    torch.cuda.synchronize()
                prof.step()

    print("\n" + prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25))
    prof.export_chrome_trace(trace_path)
    print(f"\nTrace saved to {trace_path}")


if __name__ == '__main__':
    main()
