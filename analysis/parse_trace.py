"""Parse a PyTorch-profiler chrome-trace JSON into a per-kernel table.

Used by analysis/plots.py; kept separate so the trace -> dataframe step
can be imported in isolation by later analysis scripts.
"""
import json
from collections import defaultdict


def load_trace(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def kernel_events(trace):
    """Return only the GPU-kernel events from a chrome trace.

    PyTorch profiler marks GPU kernels with cat == 'kernel'. Python-side
    aten::* ops use cat in {'cpu_op', 'python_function'} and are excluded.
    """
    for ev in trace.get('traceEvents', []):
        if ev.get('cat') != 'kernel':
            continue
        if 'dur' not in ev or 'name' not in ev:
            continue
        yield ev


def aggregate_by_name(trace):
    """Sum {name -> (total_us, n_calls)} across all GPU-kernel events."""
    totals = defaultdict(lambda: [0.0, 0])
    for ev in kernel_events(trace):
        t = totals[ev['name']]
        t[0] += ev['dur']
        t[1] += 1
    return {k: (v[0], v[1]) for k, v in totals.items()}


def summarise(trace, top_n=25):
    agg = aggregate_by_name(trace)
    total_us = sum(v[0] for v in agg.values())
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])[:top_n]
    return total_us, [(name, us, n, 100 * us / total_us if total_us else 0.0)
                      for name, (us, n) in rows]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('trace', help="Path to chrome-trace JSON.")
    ap.add_argument('--top', type=int, default=25)
    args = ap.parse_args()

    trace = load_trace(args.trace)
    total_us, rows = summarise(trace, top_n=args.top)
    print(f"Total GPU-kernel time: {total_us/1000:.3f} ms  "
          f"across {sum(1 for _ in kernel_events(trace))} events")
    print(f"{'%':>6}  {'ms':>9}  {'calls':>6}  name")
    for name, us, n, pct in rows:
        short = name if len(name) <= 90 else name[:87] + '...'
        print(f"{pct:6.2f}  {us/1000:9.3f}  {n:6d}  {short}")


if __name__ == '__main__':
    main()
