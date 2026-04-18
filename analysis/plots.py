"""Analysis plots for the cuDNN profiling study.

Produces eight PNG artefacts under results/plots/:

Per-model kernel-category breakdowns (horizontal bar, one per model):
  - resnet18_kernel_breakdown.png
  - mobilenetv3_kernel_breakdown.png
  - distilbert_kernel_breakdown.png
  - gru_kernel_breakdown.png

ResNet-18-specific deep-dive (conv algorithms, TC vs SIMT):
  - resnet18_conv_algorithms.png

Cross-model comparison plots (all four models on one figure):
  - cross_model_category_stacked.png     horizontal stacked bar, category shares
  - cross_model_latency_throughput.png   paired-axis bar, ms and samples/sec
  - cross_model_tc_share.png             bar of TF32-Tensor-Core share per model

All numbers except latency/throughput come straight from the chrome-trace JSONs.
Latency/throughput are the multi-trial means captured in execution_log_3.md;
they are kept in MODEL_LATENCY so this module stays self-contained.
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

# (display_name, trace_path) — order sets the y-axis order on cross-model plots.
MODELS = [
    ('ResNet-18',         'results/traces/resnet18_baseline_bs32_benchOn.json'),
    ('MobileNetV3-Small', 'results/traces/mobilenetv3_baseline_bs32_benchOn.json'),
    ('DistilBERT-base',   'results/traces/distilbert_baseline_bs8_benchOn.json'),
    ('Tiny GRU',          'results/traces/gru_baseline_bs32_benchOn.json'),
]

# Keyed by short model id (= first token of display name, lowercased) — matches
# the filename stem used by run_baseline.py (resnet18/mobilenetv3/distilbert/gru).
MODEL_LATENCY = {
    'resnet18':    {'batch': 32, 'mean_ms': 11.710, 'std_ms': 0.609,
                    'samples_per_sec': 2732.7},
    'mobilenetv3': {'batch': 32, 'mean_ms':  3.006, 'std_ms': 0.180,
                    'samples_per_sec': 10644.2},
    'distilbert':  {'batch':  8, 'mean_ms': 12.355, 'std_ms': 0.436,
                    'samples_per_sec': 647.5},
    'gru':         {'batch': 32, 'mean_ms':  0.252, 'std_ms': 0.010,
                    'samples_per_sec': 127003.4},
}

# Short id used in filenames, keyed by display name.
MODEL_ID = {
    'ResNet-18':         'resnet18',
    'MobileNetV3-Small': 'mobilenetv3',
    'DistilBERT-base':   'distilbert',
    'Tiny GRU':          'gru',
}

# Category colours kept consistent across all plots. Order matters for stacking.
CATEGORY_COLOURS = {
    'conv_implicit_gemm':  '#d8553b',
    'conv_depthwise':      '#a63720',
    'conv_winograd':       '#e6a23c',
    'conv_backward':       '#f5c871',
    'matmul_tensor_core':  '#b24df0',
    'matmul_fp32':         '#3b7dd8',
    'fused_attention':     '#7a3fc4',
    'norm':                '#2ba87b',
    'layout_convert':      '#888888',
    'pool':                '#6fa3ef',
    'rnn':                 '#de6dae',
    'softmax':             '#c4b94a',
    'elementwise':         '#9aa8ba',
    'reduce':              '#6c7580',
    'embed_gather':        '#a0a0a0',
    'other':               '#d0d0d0',
}


# --------------------------------------------------------------------------- #
# Per-model plots
# --------------------------------------------------------------------------- #

def plot_category_breakdown(per_name, display_name, out_path):
    cat_totals = aggregate_by_category(per_name)
    total_ms = sum(v[0] for v in cat_totals.values()) / 1000.0
    rows = [(c, cat_totals[c][0] / 1000.0)
            for c in CATEGORY_ORDER if c in cat_totals]
    rows.sort(key=lambda kv: -kv[1])

    labels = [c for c, _ in rows]
    values_ms = [v for _, v in rows]
    pcts = [100 * v / total_ms for v in values_ms]
    colours = [CATEGORY_COLOURS.get(c, '#888') for c in labels]

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    y = range(len(labels))
    ax.barh(y, values_ms, color=colours)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('CUDA time (ms, 10 profiled iterations)')
    ax.set_title(
        f'{display_name} kernel-time breakdown '
        f'(total = {total_ms:.2f} ms)',
        fontsize=11)
    for i, (v, p) in enumerate(zip(values_ms, pcts)):
        ax.text(v, i, f'  {v:.2f} ms  ({p:.1f}%)',
                va='center', fontsize=9)
    ax.set_xlim(0, max(values_ms) * 1.25)
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=140, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)


def plot_conv_algorithms(per_name, out_path):
    """ResNet-18-specific bar chart over its conv kernel variants."""
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
            if 'nhwckrsc_nhwc' in name:
                return 'sm80_xmma_fprop_implicit_gemm_tf32 nhwc-out (TC)'
            return 'sm80_xmma_fprop_implicit_gemm_tf32 nchw-out (TC)'
        if 'implicit_convolve_sgemm' in name:
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
    fig.savefig(out_path, dpi=140, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Cross-model plots
# --------------------------------------------------------------------------- #

def plot_cross_model_stacked(model_category_shares, out_path):
    """model_category_shares: list of (display_name, {category: share_pct}).

    Horizontal stacked bar: one bar per model, coloured by category share.
    """
    all_categories = [c for c in CATEGORY_ORDER
                      if any(c in shares for _, shares in model_category_shares)]

    names = [n for n, _ in model_category_shares]
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    left = [0.0] * len(names)
    for cat in all_categories:
        widths = [shares.get(cat, 0.0) for _, shares in model_category_shares]
        bars = ax.barh(
            names, widths, left=left,
            color=CATEGORY_COLOURS.get(cat, '#888'),
            label=cat,
            edgecolor='white', linewidth=0.5,
        )
        for i, (w, l) in enumerate(zip(widths, left)):
            if w >= 4.0:  # only label slices >=4% for readability
                ax.text(l + w / 2, i, f'{w:.0f}%',
                        ha='center', va='center', fontsize=8, color='black')
        left = [l + w for l, w in zip(left, widths)]

    ax.set_xlabel('Share of total GPU kernel time (%)')
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_title('Cross-model GPU time by kernel category '
                 '(baseline, batch defaults, FP32+TF32, benchmark=True)',
                 fontsize=11)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.42),
              ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(out_path, dpi=140, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)


def plot_cross_model_latency_throughput(out_path):
    """Dual-panel: latency bar (left) + throughput bar (right).

    Latency uses batch-normalised ms/iter (smaller = faster).
    Throughput uses samples/sec on a log scale (bigger = faster).
    """
    names = [n for n, _ in MODELS]
    ids = [MODEL_ID[n] for n in names]
    lat_mean = [MODEL_LATENCY[i]['mean_ms'] for i in ids]
    lat_std = [MODEL_LATENCY[i]['std_ms'] for i in ids]
    thr = [MODEL_LATENCY[i]['samples_per_sec'] for i in ids]
    batches = [MODEL_LATENCY[i]['batch'] for i in ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.2))

    # Panel A: latency
    colours = ['#d8553b', '#e6a23c', '#3b7dd8', '#2ba87b']
    ax1.bar(names, lat_mean, yerr=lat_std, capsize=6, color=colours,
            edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Latency (ms / iteration)')
    ax1.set_title('Per-iteration latency (mean ± std over 7 trials × 50 iters)',
                  fontsize=10)
    for i, (v, s, b) in enumerate(zip(lat_mean, lat_std, batches)):
        ax1.text(i, v + s + max(lat_mean) * 0.02,
                 f'{v:.2f}±{s:.2f} ms\nbatch={b}',
                 ha='center', va='bottom', fontsize=8)
    ax1.set_ylim(0, max(lat_mean) * 1.35)
    ax1.tick_params(axis='x', rotation=15)

    # Panel B: throughput (log scale)
    ax2.bar(names, thr, color=colours, edgecolor='black', linewidth=0.5)
    ax2.set_yscale('log')
    ax2.set_ylabel('Throughput (samples / sec, log scale)')
    ax2.set_title('Per-sample throughput (higher = faster)',
                  fontsize=10)
    for i, v in enumerate(thr):
        ax2.text(i, v * 1.15, f'{v:,.0f}',
                 ha='center', va='bottom', fontsize=9)
    ax2.set_ylim(100, max(thr) * 3)
    ax2.tick_params(axis='x', rotation=15)

    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_path, dpi=140, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)


def plot_cross_model_tc_share(tc_shares, out_path):
    """tc_shares: list of (display_name, tc_pct)."""
    names = [n for n, _ in tc_shares]
    vals = [v for _, v in tc_shares]
    colours = ['#d8553b' if v >= 30 else '#e6a23c' if v >= 10 else '#3b7dd8'
               for v in vals]

    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    bars = ax.bar(names, vals, color=colours,
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('TF32 Tensor-Core share (% of total CUDA time)')
    ax.set_title('Tensor-Core engagement across the model zoo '
                 '(higher = more GPU time in TC-TF32 kernels)',
                 fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.03, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=10)
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_path, dpi=140, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def tensor_core_share_pct(per_name):
    """Fraction of total kernel time in TC kernels (tensorop / xmma / hmma)."""
    total_us = sum(us for us, _ in per_name.values())
    if total_us <= 0:
        return 0.0
    tc_us = 0.0
    for name, (us, _) in per_name.items():
        low = name.lower()
        if ('tensorop' in low or 'xmma' in low or 'hmma' in low
                or 'bmma' in low or 'imma' in low or 's1688' in low
                or 's16816' in low):
            tc_us += us
    return 100.0 * tc_us / total_us


def category_share_pct(per_name):
    """{category -> share_pct} summing to 100 across all categories present."""
    cat_totals = aggregate_by_category(per_name)
    total_us = sum(v[0] for v in cat_totals.values())
    if total_us <= 0:
        return {}
    return {c: 100.0 * v[0] / total_us for c, v in cat_totals.items()}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load each trace once; reuse for per-model and cross-model plots.
    per_model = {}
    missing = []
    for display_name, path in MODELS:
        if not os.path.exists(path):
            missing.append((display_name, path))
            continue
        per_model[display_name] = aggregate_by_name(load_trace(path))
        print(f"Loaded {display_name}: {path}")

    if missing:
        names = ', '.join(n for n, _ in missing)
        raise FileNotFoundError(
            f"Missing traces for: {names}. "
            "Run `python -m profiling.run_baseline --model <name>` first.")

    # Per-model plots.
    for display_name, per_name in per_model.items():
        mid = MODEL_ID[display_name]
        out = os.path.join(PLOTS_DIR, f'{mid}_kernel_breakdown.png')
        plot_category_breakdown(per_name, display_name, out)
        print(f"Wrote {out}")

    # ResNet-18-specific conv-algorithms deep-dive (kept for parity with log_2).
    resnet_out = os.path.join(PLOTS_DIR, 'resnet18_conv_algorithms.png')
    plot_conv_algorithms(per_model['ResNet-18'], resnet_out)
    print(f"Wrote {resnet_out}")

    # Cross-model stacked category bar.
    model_category_shares = [
        (name, category_share_pct(per_model[name])) for name, _ in MODELS
    ]
    stacked_out = os.path.join(PLOTS_DIR, 'cross_model_category_stacked.png')
    plot_cross_model_stacked(model_category_shares, stacked_out)
    print(f"Wrote {stacked_out}")

    # Cross-model latency + throughput.
    lt_out = os.path.join(PLOTS_DIR, 'cross_model_latency_throughput.png')
    plot_cross_model_latency_throughput(lt_out)
    print(f"Wrote {lt_out}")

    # Cross-model Tensor-Core engagement.
    tc_shares = [(name, tensor_core_share_pct(per_model[name]))
                 for name, _ in MODELS]
    tc_out = os.path.join(PLOTS_DIR, 'cross_model_tc_share.png')
    plot_cross_model_tc_share(tc_shares, tc_out)
    print(f"Wrote {tc_out}")

    print("\nTensor-Core shares (for inclusion in log_3):")
    for name, share in tc_shares:
        print(f"  {name:22s} {share:5.2f}%")


if __name__ == '__main__':
    main()
