"""Render the Phase-2 (ResNet-18) analysis plots.

Two charts are produced:
  - resnet18_kernel_breakdown.png : horizontal bar of CUDA-time share by
      coarse kernel category (conv / norm / pool / elementwise / ...).
  - resnet18_conv_algorithms.png  : breakdown of only the convolution
      kernels into the named cuDNN/CUTLASS algorithm variants.

Saved to results/plots/; copied nowhere else. The writeup links to
these paths relative to the repo root.
"""
import os
import re
import matplotlib
matplotlib.use('Agg')  # no display on the profiling host
import matplotlib.pyplot as plt

from analysis.parse_trace import aggregate_by_name, load_trace
from analysis.classify_kernels import (
    aggregate_by_category, classify, CATEGORY_ORDER,
)

PLOTS_DIR = 'results/plots'
TRACE = 'results/traces/resnet18_baseline_bs32_benchOn.json'
TRACE_LEGACY = 'results/traces/resnet18_baseline.json'  # pre-fix filename


def _pick_trace_path():
    if os.path.exists(TRACE):
        return TRACE
    if os.path.exists(TRACE_LEGACY):
        return TRACE_LEGACY
    raise FileNotFoundError(
        f"Neither {TRACE} nor {TRACE_LEGACY} exists. "
        "Run `python -m profiling.run_baseline --model resnet18` first.")


def plot_category_breakdown(per_name, out_path):
    cat_totals = aggregate_by_category(per_name)
    total_ms = sum(v[0] for v in cat_totals.values()) / 1000.0
    rows = [(c, cat_totals[c][0] / 1000.0)
            for c in CATEGORY_ORDER if c in cat_totals]
    rows.sort(key=lambda kv: -kv[1])

    labels = [c for c, _ in rows]
    values_ms = [v for _, v in rows]
    pcts = [100 * v / total_ms for v in values_ms]

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    y = range(len(labels))
    ax.barh(y, values_ms, color='#3b7dd8')
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('CUDA time (ms, 10 profiled iterations)')
    ax.set_title(
        f'ResNet-18 kernel-time breakdown  '
        f'(batch 32, FP32+TF32, total = {total_ms:.2f} ms)',
        fontsize=11)
    for i, (v, p) in enumerate(zip(values_ms, pcts)):
        ax.text(v, i, f'  {v:.2f} ms  ({p:.1f}%)',
                va='center', fontsize=9)
    ax.set_xlim(0, max(values_ms) * 1.22)
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_conv_algorithms(per_name, out_path):
    """Bar chart over all kernels classified as conv_* (excluding the
    wrapper aten::cudnn_convolution, which isn't a kernel)."""
    conv_items = [(name, us / 1000.0)
                  for name, (us, _) in per_name.items()
                  if classify(name).startswith('conv_')]
    conv_items.sort(key=lambda kv: -kv[1])
    total_ms = sum(v for _, v in conv_items)
    if not conv_items:
        print("No conv_* kernels found; skipping conv algorithm plot.")
        return

    def shorten(name):
        if 'cutlass_tensorop_s1688fprop' in name:
            return 'cutlass_tensorop_s1688fprop_tf32 (TC)'
        if 'xmma_fprop_implicit_gemm_tf32' in name:
            # Two variants differ only in output layout (nchw vs nhwc).
            if 'nhwckrsc_nhwc' in name:
                return 'sm80_xmma_fprop_implicit_gemm_tf32 nhwc-out (TC)'
            return 'sm80_xmma_fprop_implicit_gemm_tf32 nchw-out (TC)'
        if 'implicit_convolve_sgemm' in name:
            # Extract first few Li<digits>E template args so the two
            # SIMT variants render with distinguishable labels.
            nums = re.findall(r'Li(\d+)E', name)
            tag = ','.join(nums[:6]) if nums else '?'
            return f'implicit_convolve_sgemm<{tag}> (SIMT)'
        if 'cutlass_80_simt_sgemm' in name:
            return 'cutlass_80_simt_sgemm (SIMT FC)'
        return name[:60]

    labels = [shorten(n) for n, _ in conv_items]
    values = [v for _, v in conv_items]
    pcts = [100 * v / total_ms for v in values]

    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    y = range(len(labels))
    colours = ['#d8553b' if '(TC)' in l else '#3b7dd8' for l in labels]
    ax.barh(y, values, color=colours)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('CUDA time (ms)')
    ax.set_title(
        f'ResNet-18 convolution kernels by algorithm '
        f'(total conv = {total_ms:.2f} ms)',
        fontsize=11)
    for i, (v, p) in enumerate(zip(values, pcts)):
        ax.text(v, i, f'  {v:.2f} ms  ({p:.1f}%)',
                va='center', fontsize=9)
    ax.set_xlim(0, max(values) * 1.30)
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    trace_path = _pick_trace_path()
    print(f"Using trace: {trace_path}")
    per_name = aggregate_by_name(load_trace(trace_path))

    out_breakdown = os.path.join(PLOTS_DIR, 'resnet18_kernel_breakdown.png')
    out_conv = os.path.join(PLOTS_DIR, 'resnet18_conv_algorithms.png')
    plot_category_breakdown(per_name, out_breakdown)
    plot_conv_algorithms(per_name, out_conv)
    print(f"Wrote {out_breakdown}")
    print(f"Wrote {out_conv}")


if __name__ == '__main__':
    main()
