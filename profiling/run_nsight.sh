#!/usr/bin/env bash
# Capture Nsight Systems timelines for the four baseline models.
# Writes results/nsys/{model}.nsys-rep plus a small text summary per model.
#
# Requires: Nsight Systems 2025.x+ (we look for nsys on PATH first, then
# fall back to the default Windows install location for 2026.2.1).
# Requires: hdai conda env activated (torch + cu128 wheel).

set -euo pipefail

# Resolve nsys. Prefer PATH; fall back to known Windows install path.
if command -v nsys >/dev/null 2>&1; then
    NSYS=nsys
elif [ -x "/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.2.1/target-windows-x64/nsys.exe" ]; then
    NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.2.1/target-windows-x64/nsys.exe"
else
    echo "ERROR: nsys not on PATH and 2026.2.1 default path missing." >&2
    echo "       Add .../Nsight Systems 2026.x/target-windows-x64/ to PATH or edit NSYS above." >&2
    exit 1
fi

python -c "import torch; assert torch.cuda.is_available()" || {
    echo "ERROR: torch.cuda.is_available() is False. Activate the hdai conda env first." >&2
    exit 1
}

mkdir -p results/nsys results/nsys/stats

MODELS=(resnet18 mobilenetv3 distilbert gru)

for m in "${MODELS[@]}"; do
    echo ""
    echo "=================================================================="
    echo "Capturing $m"
    echo "=================================================================="

    # Short capture: --trials 2 --iters-per-trial 20 --warmup 10 keeps the
    # .nsys-rep small while preserving all three NVTX bands. Latency numbers
    # from this run are discarded; the committed baseline CSV comes from the
    # full 7x50 + 30 warmup runs captured in Phase 3 and left untouched.
    "$NSYS" profile \
        -t cuda,cuDNN,cublas,nvtx \
        -s none \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --stats=false \
        -o "results/nsys/${m}" \
        python -m profiling.run_baseline --model "${m}" \
            --trials 2 --iters-per-trial 20 --warmup 10

    # Emit kernel + API summary CSVs alongside the binary report.
    # --force-export=true is needed for the second stats call because
    # Nsight 2026.2.1 refuses to reuse a just-created SQLite without it.
    "$NSYS" stats --report cuda_gpu_kern_sum --format csv \
        --force-export=true --force-overwrite=true \
        --output "results/nsys/stats/${m}_kern_sum" \
        "results/nsys/${m}.nsys-rep" || true

    "$NSYS" stats --report cuda_api_sum --format csv \
        --force-export=true --force-overwrite=true \
        --output "results/nsys/stats/${m}_api_sum" \
        "results/nsys/${m}.nsys-rep" || true

    size_mb=$(du -m "results/nsys/${m}.nsys-rep" | cut -f1)
    echo "  .nsys-rep size: ${size_mb} MB"
    if [ "$size_mb" -gt 50 ]; then
        echo "  WARNING: exceeds 50MB; consider gitignoring this file"
    fi
done

# Nsight's short runs also overwrite results/traces/*.json as a side effect of
# run_baseline.py's profiler block. Restore the committed Phase-3 traces so
# the baseline CSV (which points at them) stays consistent.
echo ""
echo "Restoring committed Phase-3 chrome traces..."
git checkout HEAD -- results/traces/ results/tables/baseline_breakdown.csv

echo ""
echo "=================================================================="
echo "All captures complete. Artefacts:"
ls -lh results/nsys/*.nsys-rep
ls -lh results/nsys/stats/
echo "=================================================================="
