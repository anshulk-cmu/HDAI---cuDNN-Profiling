"""Emit the Phase-4 cross-model summary CSV.

Reads the four committed baseline traces under results/traces/, classifies
every kernel event via analysis.classify_kernels, aggregates by category,
and writes results/tables/baseline_breakdown.csv — the centerpiece
Phase-4 deliverable per docs/brief.md Phase 4.

Latency/throughput columns come from MODEL_LATENCY in analysis.plots
(single source of truth; synchronised with the multi-trial run captured
in docs/execution_log_3.md).

The CSV schema is stable — future phases extend columns, never rename.
Column meanings:
  model                       display name
  batch                       per-model default batch size
  latency_ms_mean/std         7 trials x 50 iters CUDA-event timing
  throughput_samples_per_sec  batch / (latency_mean / 1000)
  total_cuda_ms               sum of all GPU-kernel durations in the
                              10-iteration profiler window
  events                      number of GPU kernel events
  events_per_iter             events / 10
  <category>_pct              % of total_cuda_ms in that classifier
                              category (one column per CATEGORY_ORDER
                              entry)
  tc_total_pct                % of total_cuda_ms in any TC-eligible
                              kernel (tensorop/xmma/hmma/s1688/s16816),
                              computed independently of the classifier
"""
import csv
import os

from analysis.parse_trace import aggregate_by_name, load_trace
from analysis.classify_kernels import (
    CATEGORY_ORDER, aggregate_by_category,
)
from analysis.plots import (
    MODELS, MODEL_ID, MODEL_LATENCY, tensor_core_share_pct,
)

TABLES_DIR = 'results/tables'
OUT_PATH = os.path.join(TABLES_DIR, 'baseline_breakdown.csv')
PROFILED_ITERS = 10  # matches run_baseline.py's profiler schedule active=10


def summarise_trace(path):
    trace = load_trace(path)
    per_name = aggregate_by_name(trace)
    cats = aggregate_by_category(per_name)
    total_us = sum(us for us, _ in per_name.values())
    events = sum(n for _, n in per_name.values())
    cat_pct = ({c: 100.0 * cats.get(c, (0.0, 0))[0] / total_us
                for c in CATEGORY_ORDER}
               if total_us else {c: 0.0 for c in CATEGORY_ORDER})
    return {
        'total_cuda_ms': total_us / 1000.0,
        'events': events,
        'cat_pct': cat_pct,
        'tc_total_pct': tensor_core_share_pct(per_name),
    }


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    fieldnames = [
        'model', 'batch', 'latency_ms_mean', 'latency_ms_std',
        'throughput_samples_per_sec',
        'total_cuda_ms', 'events', 'events_per_iter',
    ] + [f'{c}_pct' for c in CATEGORY_ORDER] + ['tc_total_pct']

    rows = []
    for display_name, path in MODELS:
        mid = MODEL_ID[display_name]
        lat = MODEL_LATENCY[mid]
        s = summarise_trace(path)
        row = {
            'model': display_name,
            'batch': lat['batch'],
            'latency_ms_mean': round(lat['mean_ms'], 3),
            'latency_ms_std': round(lat['std_ms'], 3),
            'throughput_samples_per_sec': round(lat['samples_per_sec'], 1),
            'total_cuda_ms': round(s['total_cuda_ms'], 3),
            'events': s['events'],
            'events_per_iter': round(s['events'] / PROFILED_ITERS, 1),
            'tc_total_pct': round(s['tc_total_pct'], 2),
        }
        for c in CATEGORY_ORDER:
            row[f'{c}_pct'] = round(s['cat_pct'].get(c, 0.0), 2)
        rows.append(row)

    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {OUT_PATH}')

    # Also print a compact markdown-friendly table to stdout for the writeup.
    print()
    header = ['Model', 'Batch', 'Lat (ms)', 'Thru (samp/s)',
              'Conv%', 'Matmul%', 'Norm%', 'Elem%', 'Other%', 'TC%']
    sep = ['---'] * len(header)
    print('| ' + ' | '.join(header) + ' |')
    print('| ' + ' | '.join(sep) + ' |')
    for r in rows:
        conv = (r['conv_implicit_gemm_pct']
                + r['conv_depthwise_pct']
                + r['conv_winograd_pct']
                + r['conv_backward_pct'])
        matmul = (r['matmul_tensor_core_pct']
                  + r['matmul_fp32_pct']
                  + r['fused_attention_pct'])
        other = (r['other_pct']
                 + r['embed_gather_pct']
                 + r['layout_convert_pct']
                 + r['pool_pct']
                 + r['rnn_pct']
                 + r['softmax_pct']
                 + r['reduce_pct'])
        cells = [
            r['model'], str(r['batch']),
            f"{r['latency_ms_mean']:.2f} ± {r['latency_ms_std']:.2f}",
            f"{r['throughput_samples_per_sec']:,.0f}",
            f"{conv:.1f}", f"{matmul:.1f}", f"{r['norm_pct']:.1f}",
            f"{r['elementwise_pct']:.1f}", f"{other:.1f}",
            f"{r['tc_total_pct']:.1f}",
        ]
        print('| ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
