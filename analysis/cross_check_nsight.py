"""Cross-check PyTorch-Profiler-aggregate GPU time vs Nsight-aggregate GPU time.

For each of the four baseline models, reads:
  - results/traces/<model>_baseline_bs<B>_benchOn.json  (PyTorch Profiler)
  - results/nsys/stats/<model>_kern_sum_cuda_gpu_kern_sum.csv  (Nsight Systems)

Computes total GPU-kernel time from each source and prints a comparison table.
The PyTorch run profiles 10 iters (profiler window with schedule active=10).
The Nsight run captures the full process, so kern_sum aggregates every kernel
from every forward pass: 10 warmup + 40 timing (2 trials x 20 iters) + 15
profiler = 65 iters total. We normalise each source by its own iter count.
"""
import csv
import os
import sys
from analysis.parse_trace import load_trace, aggregate_by_name

MODELS = [
    ("resnet18",    "results/traces/resnet18_baseline_bs32_benchOn.json",    10),
    ("mobilenetv3", "results/traces/mobilenetv3_baseline_bs32_benchOn.json", 10),
    ("distilbert",  "results/traces/distilbert_baseline_bs8_benchOn.json",   10),
    ("gru",         "results/traces/gru_baseline_bs32_benchOn.json",         10),
]

# Nsight capture config from profiling/run_nsight.sh: 10 warmup + 2*20 timing
# iters + 15 profiler iters = 65 forward passes observed in kern_sum totals.
NSYS_ITERS = 65


def pytorch_total_us(trace_path):
    per = aggregate_by_name(load_trace(trace_path))
    return sum(us for us, _ in per.values())


def nsight_total_ns(stats_csv):
    total_ns = 0.0
    with open(stats_csv) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            total_ns += float(row["Total Time (ns)"])
    return total_ns


def main():
    print(f"{'model':14s}  {'pyt ms/it':>10s}  {'nsys ms/it':>11s}  {'delta %':>8s}")
    print("-" * 50)
    worst = 0.0
    for m, trace, pyt_iters in MODELS:
        stats_csv = f"results/nsys/stats/{m}_kern_sum_cuda_gpu_kern_sum.csv"
        if not os.path.exists(stats_csv):
            print(f"{m:14s}  (missing {stats_csv})")
            continue
        pyt_ms_per_iter = (pytorch_total_us(trace) / 1000.0) / pyt_iters
        nsys_ms_per_iter = (nsight_total_ns(stats_csv) / 1e6) / NSYS_ITERS
        delta_pct = 100 * (nsys_ms_per_iter - pyt_ms_per_iter) / pyt_ms_per_iter
        worst = max(worst, abs(delta_pct))
        print(f"{m:14s}  {pyt_ms_per_iter:10.3f}  {nsys_ms_per_iter:11.3f}  {delta_pct:+7.2f}%")
    print("-" * 50)
    print(f"worst |delta|: {worst:.2f}%")
    if worst > 20.0:
        print("FAIL: Nsight vs PyTorch Profiler disagreement exceeds 20%.")
        sys.exit(1)
    print("OK (within 20% tolerance; target is < 10%)")


if __name__ == "__main__":
    main()
