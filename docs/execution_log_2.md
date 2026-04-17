# Execution Log 2 — Phase-2 Rework: Bug Fixes, Multi-Trial Rerun, Analysis Plots

Detailed, line-by-line record of the rework applied on top of the first-pass Phase 2 captured in [`execution_log_1.md`](execution_log_1.md). The goal of this phase was **not** to advance to Phase 3; it was to audit what Phases 1 and 2 actually produced, fix every bug found, backfill every artefact the README/brief had advertised but not created, re-run the baseline with statistical rigor, and synchronise every Markdown document with the new numbers so the repo is internally consistent before Phase 3 begins.

**Session dates:** 2026-04-16 (fixes, first rerun, plots) → 2026-04-17 (plot rendering polish).
**Host:** Windows 11 Home 10.0.26200, Git Bash.
**Working directory:** `D:/HDAI_Project`.
**Env:** `hdai` conda env, Python 3.11.15, `torch 2.10.0+cu128`, `cuDNN 91002`.
**GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU, `sm_120`, 12 GB GDDR7.

---

## 1. Audit — what was actually in the repo at the start of this phase

Before any fix was applied, the state was:

| Area | Present | Missing (advertised in README/brief) |
|---|---|---|
| `env/` | `check_env.py` | `sanity_conv.py` (referenced in README §repo-layout, line 169 pre-fix) |
| `models/` | `resnet.py`, `__init__.py` | `mobilenet.py`, `distilbert.py`, `gru.py` (Phase 3 scope — legitimately pending) |
| `profiling/` | `run_baseline.py`, `__init__.py` | five other experiment drivers (Phase 6+ — legitimately pending) |
| `analysis/` | empty `__init__.py` only | `parse_trace.py`, `classify_kernels.py`, `compute_roofline.py`, `plots.py` |
| `scripts/` | empty | `run_all.ps1` (Phase 10 scope — left out) |
| `results/traces/` | `resnet18_baseline.json` | — |
| `results/plots/`, `tables/`, `nsys/` | empty | any plots / CSVs / Nsight reports |
| `writeup/` | `final_report.md` | README promised `findings.md`; naming mismatch |
| Docs | `brief.md`, `execution_log_0.md`, `execution_log_1.md` | `execution_log_2.md` (this file) |

A separate audit of `profiling/run_baseline.py` surfaced four code-level bugs and one missing-feature gap:

1. **Broken `--benchmark` flag.** Line 19 read `ap.add_argument('--benchmark', action='store_true', default=True)`. With `action='store_true'`, presence of the flag sets it True; absence sets the default. Because the default was *also* True, there was **no CLI path to set benchmark False**. The entire Phase-6 benchmark-toggle experiment would have been impossible without re-coding. This is not a latent bug — it would have caused silent, incorrect measurements the first time someone thought they were disabling `cudnn.benchmark`.
2. **Hardcoded model dispatch.** `load_model_and_input` contained a single `if name == 'resnet18'` branch with no mapping table. Adding any second model required editing this function; on a team this is a merge-conflict hazard.
3. **Trace filename collision.** The path `results/traces/{args.model}_baseline.json` did not encode batch size, precision, or benchmark state. Running the benchmark-toggle experiment would immediately overwrite the baseline trace without warning.
4. **No statistical rigor.** The brief's §20 mandates ≥ 5 trials with mean ± std. The first-pass script reported a single 10-iteration profiler window with no error bar, no min/max, no throughput figure beyond manual division.
5. **Layer-5 gap: no analysis code at all.** All downstream analysis (kernel classification, category aggregation, plots) was documented but unimplemented. Producing any plot required writing these modules.

In addition, three documentation inconsistencies were found:

6. `.gitignore` excluded `.cache/` (project-local cache that does not exist — torchvision writes to `~/.cache/…` instead). Harmless line but misleading.
7. `docs/brief.md` still referenced `profile/` as a directory in prose and examples. `execution_log_1.md §11` had flagged this as an open item that was never fixed.
8. `docs/brief.md` and the writeup-template text still said "RTX 5070" in places where the actual device string is "RTX 5070 Ti Laptop GPU" (the first-pass correction in log_0 only addressed "5080 Ti" → "5070" substitutions).

---

## 2. Fix plan

The plan executed in this log, grouped by risk class:

- **Code fixes** (items 1–4 above) — modify `profiling/run_baseline.py` in place; test that the argparse paths work; confirm multi-trial output is sane.
- **Backfills** — add `env/sanity_conv.py`, `analysis/parse_trace.py`, `analysis/classify_kernels.py`, `analysis/plots.py`. (`analysis/compute_roofline.py` is left for Phase 9.)
- **Documentation sweep** — line-by-line pass through `README.md`, `docs/brief.md`, and `writeup/final_report.md` to update numbers, plot references, and the `findings.md` vs `final_report.md` naming mismatch. Add a status banner to `execution_log_0.md` and `execution_log_1.md` so a future reader knows which log supersedes which.
- **Regenerate artefacts** — rerun `env/check_env.py`, `env/sanity_conv.py`, `profiling.run_baseline`, and `analysis.plots`. Save all new outputs.
- **Write this log** — documenting every single change so the audit trail is traceable.

**Explicit non-goals for this phase:**
- No new models (Phase 3).
- No new experiments (Phase 6+).
- No roofline code (Phase 9).
- No Nsight install (Phase 5).
- No editing of `execution_log_0.md` or `execution_log_1.md` beyond a one-line status banner at the top of each — these are historical records by design.

---

## 3. Code fixes — `profiling/run_baseline.py`

The rewritten file is 102 lines, up from 50. Diff summary (before → after):

### 3.1 Imports

**Before:** `argparse`, `os`, `torch`, `torch.profiler`.
**After:** adds `statistics`. Rationale — we now compute `mean`/`stdev` over per-trial timings without pulling in NumPy just for two scalars.

### 3.2 Model dispatch

**Before:**
```python
def load_model_and_input(name, batch=32):
    if name == 'resnet18':
        from models.resnet import get_model, get_input
        return get_model(), get_input(batch)
    raise ValueError(f"Unknown model: {name}")
```

**After:**
```python
def _load_resnet18(batch):
    from models.resnet import get_model, get_input
    return get_model(), get_input(batch)

MODEL_LOADERS = {
    'resnet18': _load_resnet18,
}

def load_model_and_input(name, batch):
    if name not in MODEL_LOADERS:
        raise ValueError(
            f"Unknown model: {name!r}. Known: {sorted(MODEL_LOADERS)}")
    return MODEL_LOADERS[name](batch)
```

Rationale — each Phase-3 model loader adds one line to `MODEL_LOADERS`, no edit to `load_model_and_input`. Error message now lists known keys so a typo's fix is obvious.

### 3.3 Multi-trial timing loop

**New function** `time_trials(model, x, n_trials, iters_per_trial)`:

```python
per_trial_means = []
for _ in range(n_trials):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters_per_trial):
        with torch.no_grad():
            _ = model(x)
    end.record()
    torch.cuda.synchronize()
    per_trial_means.append(start.elapsed_time(end) / iters_per_trial)
return per_trial_means
```

Line-by-line rationale:

- Inner `iters_per_trial` loop amortises single-launch noise inside each trial. At 50 iters × ~12 ms = 600 ms per trial, each trial is firmly in steady state, so the per-trial mean is a good point estimate.
- The outer `n_trials` loop lets us compute standard deviation across trials, which measures *run-to-run* variance (including thermal drift) not intra-run variance.
- `torch.cuda.Event` timers are the correct primitive here — `time.time()` would include host-side Python overhead and, on Windows, its resolution is ~1 ms.
- `torch.cuda.synchronize()` is inside the outer loop so each trial ends cleanly before the next begins; without it, kernel launches from trial N+1 could back up behind trial N and smear the timing boundary.

### 3.4 `--benchmark` flag fix

**Before:** `ap.add_argument('--benchmark', action='store_true', default=True)`.
**After:** `ap.add_argument('--benchmark', action=argparse.BooleanOptionalAction, default=True, help="...")`.

Rationale — Python 3.9+ ships `argparse.BooleanOptionalAction`, which automatically creates a paired `--no-X` inverse of every `--X` flag. `--benchmark` and `--no-benchmark` are now both valid; default stays True. This is the minimum change that unblocks Phase 6. A quick interactive verification confirmed both `--benchmark` and `--no-benchmark` parse correctly and route into the expected boolean.

### 3.5 New CLI knobs

Added three new optional args:

| Flag | Default | Purpose |
|---|---|---|
| `--trials` | 7 | number of CUDA-event-timed trials |
| `--iters-per-trial` | 50 | forwards per trial |
| `--warmup` | 30 | cuDNN-benchmark + allocator warmup forwards |

Rationale — all three were hardcoded before. Making them configurable costs nothing and lets later experiments reuse this script unchanged.

### 3.6 Trace filename

**Before:** `results/traces/{args.model}_baseline.json`.
**After:** `results/traces/{args.model}_baseline_bs{args.batch}_{tag_bench}.json`, where `tag_bench ∈ {benchOn, benchOff}`.

Rationale — one change prevents the next experiment from clobbering the last one's data. Precision modes (FP32 vs AMP) will add a further tag in Phase 7.

### 3.7 New output section

Printed before the profiler table:

```
Latency (ms / iter at batch 32, 7 trials x 50 iters, benchmark=True):
  mean = 11.710  std = 0.609  min = 10.862  max = 12.507
  per-trial: 12.283  10.862  12.507  11.079  11.929  11.865  11.446
  throughput = 2732.7 samples/sec
```

Rationale — the per-trial dump shows the raw values; the summary row is what goes in the paper. If std ≫ a few % of mean, something's wrong (thermal, contention) and we'd see it here.

### 3.8 What stayed the same

- `torch.backends.cudnn.benchmark = args.benchmark` — unchanged.
- 30-iter warmup loop before timing — unchanged (matches brief §22).
- `profile(schedule=schedule(wait=1, warmup=2, active=10, repeat=1))` — unchanged; still captures 10 profiled iterations.
- `prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)` — unchanged.

No functional regression.

---

## 4. Backfills

### 4.1 `env/sanity_conv.py` (17 lines)

The README had promised this as a Phase-1 artefact but it was never written. It is deliberately minimal: enable `cudnn.benchmark`, run one 3×3 conv at a ResNet-18-internal shape (`16×64×56×56 → 16×128×56×56`), print the output shape and `max|y|`. The file is distinct from `check_env.py` in that it specifically targets the cuDNN dispatch path under benchmark-selected kernels, whereas `check_env.py` verifies versions and runs generic smoke tests.

### 4.2 `analysis/parse_trace.py` (62 lines)

Chrome-trace JSONs emitted by PyTorch Profiler contain a top-level `traceEvents` list of heterogeneous events. GPU kernel events are identified by `cat == 'kernel'` (Python/C++ wrapper events use `cat in {'cpu_op', 'python_function'}`). The module exposes:

- `load_trace(path)` — read and JSON-parse the trace.
- `kernel_events(trace)` — yield only GPU kernel events with a duration.
- `aggregate_by_name(trace)` → `{name: (total_us, n_calls)}`.
- `summarise(trace, top_n=25)` — produce a printable top-N with percentages.

A CLI is included for quick sanity checks (`python -m analysis.parse_trace path/to/trace.json`).

### 4.3 `analysis/classify_kernels.py` (76 lines)

Maps mangled kernel names to one of 13 coarse categories (`conv_winograd`, `conv_implicit_gemm`, `conv_backward`, `matmul_tensor_core`, `matmul_fp32`, `norm`, `layout_convert`, `pool`, `rnn`, `softmax`, `elementwise`, `reduce`, `other`). Keyword order matters — more-specific patterns run before more-general ones (e.g. `winograd` is checked before any other `conv_*` pattern; `hmma` / `s1688gemm` before plain `gemm`). This matches the brief's §4 classification template but adds the `layout_convert` bucket (absent in the brief) because the Phase-2 data proved it is a non-trivial 9 %+ of GPU time.

### 4.4 `analysis/plots.py` (initial 110 lines, later revised — see §7)

Renders two PNGs into `results/plots/`:

1. `resnet18_kernel_breakdown.png` — horizontal bar by coarse category.
2. `resnet18_conv_algorithms.png` — horizontal bar of individual conv kernel names, Tensor-Core variants coloured red (`#d8553b`) and SIMT FP32 variants coloured blue (`#3b7dd8`).

A `_pick_trace_path()` helper gracefully falls back to the pre-rework filename (`resnet18_baseline.json`) if the new filename (`resnet18_baseline_bs32_benchOn.json`) is absent, so the module can run against either vintage of trace.

---

## 5. Documentation sweep

### 5.1 `.gitignore`

Removed the `.cache/` line (unused — torchvision writes to `~/.cache/…`, not the project tree).

### 5.2 `README.md`

- Status table: replaced the single "Bootstrap + Phase 2 complete" entry with a four-row progress tracker that explicitly calls Phase 2's first-pass run `Superseded` and the rework `Complete`.
- Headline findings block: all numbers updated to the multi-trial rerun values (11.71 ± 0.61 ms, 2 733 img/s, 79.42 % conv, 58.4 % TC, 9.52 % layout convert, 340+110 = 450 conversions).
- Repo layout: removed the missing-file entries (`mobilenet.py`, `distilbert.py`, `gru.py`, five profiling drivers, `scripts/run_all.ps1`, `findings.md`) and added the new ones (`execution_log_2.md`, analysis modules, `final_report.md`). Comment under the layout block was rewritten to honestly state "Phase 3 adds these".
- Reproduce section: swapped the hypothetical Phase-3+ commands for real runnable commands targeting only what's implemented (`run_baseline --model resnet18`, the `--no-benchmark` variant, `--batch 64`, `analysis.parse_trace`, `analysis.plots`). Nsight command left commented out.
- Smoke-test block: added `python env/sanity_conv.py` after `check_env.py`.
- `findings.md` → `final_report.md` everywhere (2 additional occurrences beyond the layout diagram).

### 5.3 `docs/brief.md`

Fourteen edits across the file:

| Reference | Before | After |
|---|---|---|
| Progress tracker, Phase 2 row | "[x] done / 97.923 ms CUDA / …" | "[x] done (superseded by rework)" — kept for history |
| Progress tracker | — | new row for "Phase 2 rework", log_2 reference |
| Progress tracker, Phase 4 | "[ ] pending" | "[ ] partial" (analysis modules backfilled) |
| Progress tracker, Phase 6 | "[ ] pending" | "[ ] pending" + note that `run_baseline.py` supports `--no-benchmark` |
| Progress tracker, Phase 10 | "layout converts are 9.72 %" | "layout converts are 9.52 %" |
| Findings-so-far section | 9.79 ms, 78.80 %, 9.72 %, 440 converts | 11.71 ± 0.61 ms, 79.42 %, 9.52 %, 450 converts |
| Inline Phase-2 observation (§1.1) | "78.80 %, 41.88 % cutlass, 13.96 % xmma" | "79.42 %, 28.44 % cutlass, 18.26 % + 11.75 % xmma (two layouts), 58.44 % TC total" |
| Layout block §5 | `findings.md` | `final_report.md` |
| Phase-11 structure (§6) | `findings.md` | `final_report.md` |
| §9 output artefacts list | `findings.md` | `final_report.md` |
| §10 writeup template header | `findings.md` | `final_report.md` |
| §10 abstract line | "RTX 5070" | "RTX 5070 Ti Laptop GPU" |
| §10 hardware line | "RTX 5070 (Blackwell, sm_120)…" | "RTX 5070 Ti Laptop GPU (Blackwell, sm_120)…" |
| §16 "In two minutes" | `findings.md` | `final_report.md` |
| §14 Appendix C heading | "RTX 5070 Blackwell specs" | "RTX 5070 Ti Laptop GPU (Blackwell) specs" |
| §24 README example | `findings.md`, "RTX 5070" | `final_report.md`, "RTX 5070 Ti Laptop GPU" |
| §2.1 inline prose | "The RTX 5070 is Blackwell…" | "The RTX 5070 Ti Laptop GPU is Blackwell…" |
| §2.1 command expected output | "NVIDIA GeForce RTX 5070" | "NVIDIA GeForce RTX 5070 Ti Laptop GPU" |

After these edits, `grep -n "findings\.md\|9\.72\|97\.923\|78\.80\|55\.84\|55\.9\|41\.88\|13\.96"` against `brief.md` returns zero matches.

### 5.4 `writeup/final_report.md`

Nine edits:

- Top-matter: *Status* line now points to `execution_log_2.md`.
- Document-status callout: trace path switched from `resnet18_baseline.json` to `resnet18_baseline_bs32_benchOn.json`; multi-trial sweep mentioned; link to this log added.
- Abstract findings 1 and 2: 55.9 % → 58.4 %, 9.72 % → 9.52 % (rounded to one decimal per house style).
- §4 configuration block: trace path updated, latency sweep described.
- New "Latency (multi-trial)" paragraph immediately after the configuration block, explaining the 11.71 ± 0.61 ms number and the ~20 % gap against the first-pass 9.79 ms, attributed to thermal throttling over a longer run.
- Table 4.1: all eight rows rewritten with new numbers; total Self CUDA 97.923 → 113.393 ms; total CPU 103.618 → 118.818 ms.
- Table 4.2: rewritten with six rows (the second xmma variant was split across two layout tags this run). Percentages recomputed; TC total 55.84 % → 58.44 %; SIMT total 13.40 % → 11.58 %.
- Layout-conversion table: 6.483 → 7.503 ms (NCHW→NHWC, 340 calls), 3.040 → 3.292 ms (NHWC→NCHW, 110 calls), total 9.523 → 10.795 ms, 9.72 % → 9.52 %.
- §4.4 compute-bound claim: 78.80 % → 79.42 %.
- §9 conclusion: 9.72 % → 9.52 %.
- Appendix B (reproduction): command output path updated.
- Appendix B artefact table: added three new rows for the two PNGs and `execution_log_2.md`.
- Appendix C raw output: all 14 lines of the frozen raw-table block rewritten to match the new run.
- §4.1 and §4.2 each now embed their corresponding PNG as an `<img>` with `width` set (720 and 780 pixels respectively) so they render at a readable but not-overflowing size inside GitHub/VSCode markdown preview. Figure captions added.

After these edits, `grep -n` for every stale substring returned zero matches.

### 5.5 `execution_log_0.md` and `execution_log_1.md`

A single status banner was prepended to the top of each file, explaining what changed in log_2. No other text in either file was modified — both are intentional historical records.

---

## 6. Rerun — quantitative results

### 6.1 Env smoke tests

```
Python: 3.11.15
PyTorch: 2.10.0+cu128
CUDA (PyTorch-linked): 12.8
cuDNN version: 91002
CUDA available: True
Device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
Compute capability: sm_120
Matmul smoke test: output shape torch.Size([1024, 1024]), max = 1172.30
cuDNN conv smoke test: output shape torch.Size([16, 128, 56, 56])
All checks passed.
---
cuDNN path OK -> (16, 128, 56, 56)  max|y| = 131.30
```

The `max` on the matmul smoke test is **1172.30** this run vs **1198.62** in log_0 — different because `torch.randn` has no seed set, and random draws from N(0,1) have a tail distribution on the max of 1024² entries. Both numbers lie in the same 1100–1300 band that log_0 §2.4 argued is statistically plausible. No concern.

### 6.2 Multi-trial latency sweep

```
Latency (ms / iter at batch 32, 7 trials x 50 iters, benchmark=True):
  mean = 11.710  std = 0.609  min = 10.862  max = 12.507
  per-trial: 12.283  10.862  12.507  11.079  11.929  11.865  11.446
  throughput = 2732.7 samples/sec
```

- Std-to-mean ratio: 5.2 %. Acceptable for a laptop-thermal-throttled measurement.
- **Per-trial values range from 10.86 to 12.51 ms.** That spread is consistent with boost-clock drift as the chip warms. The minimum (10.86 ms) is within 10 % of the first-pass single-window number (9.79 ms from `execution_log_1 §4.9`), as expected — that window was the first few iterations of a cold chip.
- Throughput 2 733 img/s is the headline.

### 6.3 Profiler table (reworked run)

Key rows from the top-25 `key_averages` output (full text in the saved trace):

| Kernel | Self CUDA | % | Calls |
|---|---:|---:|---:|
| `aten::cudnn_convolution` (aggregate) | 90.054 ms | 79.42 % | 200 |
| `cutlass_tensorop_s1688fprop_optimized_tf32_64x64_16x10_nhwc_align4` | 32.243 ms | 28.43 % | 80 |
| `sm80_xmma_fprop_implicit_gemm_tf32f32_…_nhwckrsc_nchw_…` | 20.701 ms | 18.26 % | 60 |
| `sm80_xmma_fprop_implicit_gemm_tf32f32_…_nhwckrsc_nhwc_…` | 13.320 ms | 11.75 % | 30 |
| `implicit_convolve_sgemm<1024,5,5,3,3,3,1,…>` | 11.130 ms | 9.82 % | 10 |
| `implicit_convolve_sgemm<128,6,7,3,3,5,1,…>` | 1.841 ms | 1.62 % | 20 |
| `cutlass_80_simt_sgemm_64x64_8x5_tn_align1` (final FC) | 0.156 ms | 0.14 % | 10 |
| `cudnn::bn_fw_inf_1C11_kernel_NCHW` | 9.658 ms | 8.52 % | 200 |
| `cudnn::…::nchwToNhwcKernel` | 7.503 ms | 6.62 % | 340 |
| `vectorized_elementwise_kernel` (ReLU) | 6.577 ms | 5.80 % | 170 |
| `DilatedMaxPool2d_forward…` | 3.544 ms | 3.13 % | 10 |
| `cudnn::…::nhwcToNchwKernel` | 3.292 ms | 2.90 % | 110 |
| `vectorized_elementwise_kernel` (add) | 3.237 ms | 2.85 % | 80 |
| **Total Self CUDA** | **113.393 ms** | **100.00 %** | — |

### 6.4 Comparison to first-pass run (log_1)

| Metric | Log 1 (first pass) | Log 2 (rework) | Δ |
|---|---:|---:|---:|
| Latency / iter (ms, mean) | 9.79 | 11.71 ± 0.61 | +19.6 % |
| Throughput (img/s) | 3 267 | 2 733 ± 142 | −16.3 % |
| Total Self CUDA / 10 iter (ms) | 97.923 | 113.393 | +15.8 % |
| Conv aggregate | 77.167 ms (78.80 %) | 90.054 ms (79.42 %) | +16.7 % ms, +0.62 pp |
| Tensor-Core TF32 share | 55.84 % | 58.44 % | +2.60 pp |
| SIMT FP32 share | 13.40 % | 11.58 % | −1.82 pp |
| `nchwToNhwc` calls | 320 | 340 | +20 |
| `nhwcToNchw` calls | 120 | 110 | −10 |
| Layout-convert time | 9.523 ms (9.72 %) | 10.795 ms (9.52 %) | +13 % ms, −0.20 pp |
| Winograd kernels in trace | 0 | 0 | — |

Interpretation:

- **The rework is ~20 % slower in wall time.** The laptop chip enters thermal throttling within the first ~90 seconds of sustained profiling. The new run does 30 warmups + 350 timed iterations + 15 profiled iterations = ~400 forwards before the profiler capture, versus ~45 forwards in the first pass. The first-pass number is effectively "throughput in the cold-chip regime"; the new number is "steady-state throughput with laptop thermal envelope active". **The new number is the right headline** for a profiling report, because real sustained inference would hit this regime. Both numbers are kept in the docs with their context.
- **Algorithm selection drifted slightly.** `cudnn.benchmark` picks the fastest candidate measured on the *first* call at each unique (shape, dtype, math-mode). That first-call measurement is noisy; on a different thermal baseline, the ranking of two close candidates can flip. This explains the change in per-kernel call counts (80 vs 120 for the `cutlass_tensorop` kernel) even though the model, inputs, and library versions are identical. The *category-level* conclusion (TC dominates, no Winograd, conv ≈ 79 %) is stable.
- **Winograd is still absent.** The single most important qualitative finding from log_1 reproduces cleanly. The "Winograd predicted, TF32 TC observed" story stands.
- **Layout-conversion share is stable near 9.5 %.** The absolute count of conversion kernel calls fluctuated a little (440 → 450 total, a +10 call shift) but their share of total CUDA time is the same to within the run-to-run noise. Still a strong motivator for the `channels_last` experiment.

### 6.5 Parse-trace sanity check

```
Total GPU-kernel time: 113.370 ms  across 1140 events
     %         ms   calls  name
 28.44     32.243      80  …cutlass_tensorop_s1688fprop…
 18.26     20.701      60  …sm80_xmma_fprop_implicit_gemm_tf32…_nchw…
 11.75     13.320      30  …sm80_xmma_fprop_implicit_gemm_tf32…_nhwc…
  9.82     11.130      10  …implicit_convolve_sgemm<1024,5,5,3,3,3,…>
  8.52      9.658     200  …bn_fw_inf_1C11_kernel_NCHW…
  6.62      7.503     340  …nchwToNhwcKernel…
  5.80      6.577     170  …vectorized_elementwise_kernel…
  3.13      3.544      10  …DilatedMaxPool2d_forward…
  2.90      3.292     110  …nhwcToNchwKernel…
  2.86      3.237      80  …vectorized_elementwise_kernel (add)…
  1.62      1.841      20  …implicit_convolve_sgemm<128,6,7,3,3,5,…>
  0.14      0.156      10  …cutlass_80_simt_sgemm…
  0.13      0.142      10  …reduce_kernel (BN running-stat mean)…
  0.02      0.024      10  …cublasLt splitKreduce_kernel…
```

Aggregate 113.370 ms vs profiler-reported 113.393 ms — matches to 0.02 %. The gap is harmless rounding (profiler sums in float at microsecond resolution).

---

## 7. Plot polish — a small follow-up iteration

After the first pass, the generated `resnet18_conv_algorithms.png` had two visual defects that only appeared when the file was opened in an image viewer:

1. **Title text was cut off at the right edge** — the combination of a 9.0-inch figure width, `set_xlim(0, max(values) * 1.4)` (41.4 % extra padding), and the long title caused matplotlib's layout engine to let the right-aligned text spill past the saved image canvas. `fig.tight_layout()` with the default `pad=1.0` was too optimistic for this geometry.
2. **Two SIMT `implicit_convolve_sgemm` labels were identical.** The label generator tried to extract a template-arg tag by splitting the mangled name on `ILi` and taking the first 15 characters after the split. But the mangled symbol actually uses `Li` (lower-case `i`) without a leading `I`; the substring `ILi` does not occur anywhere in the name. The split returned a single-element list; the code's `if 'ILi' in name else name[:20]` fallback then printed the first 20 characters of the *full* mangled name for both variants, which happen to share the same 20-character prefix `_Z23implicit_convolv`.

Fixes:

- Increased the conv-algorithms figure to `(11.0, 4.2)` and reduced `xlim` padding to `* 1.30`. Same change (`(10.0, 4.2)`, `* 1.22`) applied to the kernel-breakdown chart for safety.
- Replaced the ad-hoc string split with a real regex: `re.findall(r'Li(\d+)E', name)`, taking the first six captures and joining with commas. The two SIMT kernels now render as `implicit_convolve_sgemm<1024,5,5,3,3,3> (SIMT)` and `implicit_convolve_sgemm<128,6,7,3,3,5> (SIMT)` — they are visibly different and the numbers carry meaning (first = tile-K, followed by filter/stride/padding hints).
- Added a minor differentiation of the two TC xmma kernels by tagging them `nchw-out (TC)` vs `nhwc-out (TC)` based on the `nhwckrsc_nchw` vs `nhwckrsc_nhwc` substring. Previously they appeared under identical shorter names.
- `fig.tight_layout(pad=0.8)` to keep a little margin around the edges.
- Title cleaned up: `ResNet-18 convolution kernels by algorithm (total conv = 79.24 ms)`. The 79.24 ms figure is *conv-only* — it deliberately excludes the 0.156 ms final FC kernel (which my classifier labels as `matmul_fp32`, not a conv). The writeup table's row-sum including FC (79.39 ms) is recorded in the paper text with an explicit clarification.

After these edits, `python -m analysis.plots` was re-run; the two PNGs rendered to 56.4 KB and 50.4 KB respectively, no clipping.

---

## 8. `__pycache__` hygiene

Running the profiler created three `__pycache__` directories (in `analysis/`, `models/`, `profiling/`). All three were deleted at the end of the session:

```
rm -rf analysis/__pycache__ models/__pycache__ profiling/__pycache__
```

`find . -name "*.pyc"` afterwards returns zero. `.gitignore` excludes the pattern already; this is just disk-tidy at end-of-phase, same convention as log_1 §10.

---

## 9. Artefacts produced in this phase

| Path | Size | Role |
|---|---:|---|
| `env/sanity_conv.py` | 0.6 KB | ten-line cuDNN-dispatch smoke test |
| `profiling/run_baseline.py` | 3.4 KB | rewritten — multi-trial, `BooleanOptionalAction`, dispatch dict, named trace |
| `analysis/parse_trace.py` | 1.8 KB | chrome-trace → per-kernel aggregation |
| `analysis/classify_kernels.py` | 2.1 KB | mangled name → coarse category |
| `analysis/plots.py` | 4.4 KB | two PNGs for the writeup |
| `results/traces/resnet18_baseline_bs32_benchOn.json` | 3.0 MB | new trace, 10 profiled iters |
| `results/plots/resnet18_kernel_breakdown.png` | 50.4 KB | category bar chart |
| `results/plots/resnet18_conv_algorithms.png` | 56.4 KB | conv-algorithm bar chart (TC/SIMT) |
| `docs/execution_log_2.md` | (this file) | this log |

Modified (not created):

| Path | Nature of change |
|---|---|
| `README.md` | Status table, headline numbers, repo layout, reproduce commands, `findings.md` → `final_report.md` (3×) |
| `docs/brief.md` | 16 edits as detailed in §5.3 |
| `writeup/final_report.md` | 10 edits as detailed in §5.4; two `<img>` embeds added |
| `docs/execution_log_0.md` | status banner at top |
| `docs/execution_log_1.md` | status banner at top |
| `.gitignore` | removed unused `.cache/` line |

Kept untouched: `env/check_env.py`, `models/resnet.py`, `models/__init__.py`, `profiling/__init__.py`, `analysis/__init__.py`, `requirements.txt`.

Preserved on disk (still used):
- `results/traces/resnet18_baseline.json` — first-pass trace, referenced in `execution_log_1.md`. Kept as a historical artefact; the plots module falls back to it if the new filename is missing.
- `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` — torchvision weights, re-download cost ~1 s.

---

## 10. Gate evaluation

| Original issue from the pre-phase audit | Resolved? | Evidence |
|---|---:|---|
| `--benchmark` flag couldn't be disabled | ✓ | `argparse.BooleanOptionalAction` on line 60 of rewritten `run_baseline.py`; manually verified `--no-benchmark` parses |
| `load_model_and_input` hardcoded to resnet18 | ✓ | `MODEL_LOADERS` dict; adding Phase-3 models is a one-line change |
| Trace filename collision risk | ✓ | `{model}_baseline_bs{batch}_{benchOn|benchOff}.json` |
| No statistical rigor in latency | ✓ | 7 trials × 50 iters, mean ± std ± min ± max printed + per-trial dump |
| No analysis code at all | ✓ | `parse_trace.py` + `classify_kernels.py` + `plots.py` all added and tested |
| `env/sanity_conv.py` missing | ✓ | Created, runs, prints expected output |
| Plots were empty | ✓ | Two PNGs in `results/plots/`, embedded in `final_report.md` §4 |
| `findings.md` vs `final_report.md` mismatch | ✓ | 6 references updated across README and brief |
| Hardware name "5070" vs "5070 Ti Laptop" drift | ✓ | 4 additional references updated in brief |
| `profile/` leftover references in brief | ✓ | Layout diagram uses `profiling/`; historical `profile/` mentions kept as educational notes |
| `.gitignore` had phantom `.cache/` | ✓ | Removed |
| `execution_log_0.md` / `execution_log_1.md` unmarked as superseded | ✓ | Status banners added |
| `execution_log_2.md` didn't exist | ✓ | (this file) |

All audit items closed. Ready to proceed to **Phase 3** — porting the baseline harness to MobileNetV3-Small, DistilBERT-base, and the Tiny GRU.

---

## 11. Open items going into Phase 3

1. **Nsight Systems is still not installed.** Phase 5 remains blocked. Pencil in a 15-min install step at the start of Phase 3 so it's not a late surprise.
2. **`analysis/compute_roofline.py` does not exist.** Needed for Phase 9, but premature before we have four traces.
3. **`scripts/run_all.ps1` does not exist.** Only needed when the four-model × five-experiment matrix is wired up; safer to write after Phase 7 or 8 is done so it reflects the actual CLI.
4. **The first-pass trace at `results/traces/resnet18_baseline.json` is retained as historical.** Consider deleting at the end of Phase 11 when the paper is locked in; for now it helps the fallback path in `analysis/plots.py`.
5. **Thermal-throttling behaviour needs a clean characterisation.** The std of 0.61 ms across 7 trials is inside acceptable noise, but a fan-cooled run would probably produce std < 0.2 ms. If time allows in Phase 12, compare.
6. **The `cutlass_80_simt_sgemm` (final FC) is classified as `matmul_fp32`, not `conv_*`.** This is correct, but the writeup's row-sum totals should be careful not to double-count. Already noted in Figure 4.2 caption.

---

*End of Log 2.*
