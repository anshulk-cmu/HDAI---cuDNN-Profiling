# HDAI — cuDNN Profiling Study on Blackwell

**Course:** EEL71020 — Hardware Design for AI
**Institution:** Indian Institute of Technology, Jodhpur
**Authors:** Anshul Kumar (M25AI2036), Neha Prasad (M25AI2056)

---

A compact, pedagogically-focused study profiling how **cuDNN** (and the broader CUDA stack) behaves across four deep-learning models that span the conv/matmul × compute-bound/memory-bound quadrant. All measurements are inference-only on an RTX 5070 Ti Laptop GPU (Blackwell, `sm_120`, 12 GB) using **PyTorch Profiler** and **Nsight Systems**.

The goal is not to beat published numbers or optimise a model — it is to **read profiler output carefully**, catalogue which kernels cuDNN actually dispatches, and explain *why* each model looks the way it does.

---

## What this project answers

For every model in the zoo, we answer three concrete questions:

1. **Where does the time go?** — How much of inference is convolution vs. matmul vs. normalisation vs. elementwise vs. overhead.
2. **Which cuDNN algorithms win, and why?** — GEMM vs. Winograd vs. FFT vs. implicit-GEMM; plain kernels vs. Tensor-Core kernels.
3. **Compute-bound or memory-bound?** — Proven with numbers on a roofline plot, not vibes.

The deliverable is a profiling report + plots + this small repo of scripts.

---

## Status

| Phase | Status | Artefact |
|---|---|---|
| Phase 0 — brief read, hardware/spec corrections | Complete | [`docs/brief.md`](docs/brief.md), [`docs/execution_log_0.md`](docs/execution_log_0.md) |
| Phase 1 — conda env, toolchain, smoke tests | Complete | [`env/check_env.py`](env/check_env.py), [`env/sanity_conv.py`](env/sanity_conv.py), [`docs/execution_log_0.md`](docs/execution_log_0.md) |
| Phase 2 — ResNet-18 baseline profile (first pass) | Superseded | [`docs/execution_log_1.md`](docs/execution_log_1.md) |
| Phase 2 rework — bug fixes, multi-trial rerun, analysis plots | Complete | [`docs/execution_log_2.md`](docs/execution_log_2.md), `results/traces/resnet18_baseline_bs32_benchOn.json` |
| Phase 3 — baseline port to MobileNetV3, DistilBERT, GRU + cross-model plots | Complete | [`docs/execution_log_3.md`](docs/execution_log_3.md), 3 new traces, 8 plots under [`results/plots/`](results/plots/) |
| Phase 4 — kernel classification, 4-model summary table | Complete | [`docs/execution_log_4.md`](docs/execution_log_4.md), [`results/tables/baseline_breakdown.csv`](results/tables/baseline_breakdown.csv), classifier buckets `conv_depthwise` + `fused_attention` + `embed_gather` added |
| Phase 5 — Nsight Systems timeline + NVTX instrumentation | Complete | [`docs/execution_log_5.md`](docs/execution_log_5.md), 4 `.nsys-rep` under [`results/nsys/`](results/nsys/), 8 screenshots `nsight_*` under [`results/plots/`](results/plots/), [`analysis/cross_check_nsight.py`](analysis/cross_check_nsight.py), writeup §5.6 |
| Phases 6–12 (experiments, roofline, writeup) | Pending | — |

### Headline findings so far (four-model baseline — see `docs/execution_log_2.md` for ResNet-18, `docs/execution_log_3.md` for the other three)

**Cross-model table (batch per `DEFAULT_BATCH`, FP32+TF32, `cudnn.benchmark=True`):**

| Model | Batch | Latency (ms) | Throughput (samples/s) | TF32 Tensor-Core share |
|---|---:|---:|---:|---:|
| ResNet-18 | 32 | 11.71 ± 0.61 | 2 733 | **58.45 %** |
| MobileNetV3-Small | 32 |  3.01 ± 0.18 | 10 644 | 14.94 % |
| DistilBERT-base |  8 | 12.36 ± 0.44 | 647 (82 880 tokens/s) | **0.00 %** |
| Tiny GRU | 32 |  0.25 ± 0.01 | 127 003 | 16.77 % |

Plots: [`cross_model_category_stacked.png`](results/plots/cross_model_category_stacked.png), [`cross_model_latency_throughput.png`](results/plots/cross_model_latency_throughput.png), [`cross_model_tc_share.png`](results/plots/cross_model_tc_share.png).

Full classified per-category breakdown (17 category columns × 4 models) in [`results/tables/baseline_breakdown.csv`](results/tables/baseline_breakdown.csv) (Phase 4 centerpiece).

### ResNet-18 findings (from the reworked run)

- **Inference latency:** **11.71 ± 0.61 ms** / batch of 32 → **≈ 2 733 images/sec** on the RTX 5070 Ti Laptop GPU, measured as the mean over **7 trials × 50 iterations** of CUDA-event timing after 30 warm-ups. (The first-pass single-window number was 9.79 ms on a cold chip; the new number captures steady-state thermal behaviour and is the headline going forward.)
- **Conv dominates, as expected:** `aten::cudnn_convolution` accounts for **79.42 %** of CUDA time; BatchNorm 8.52 %, ReLU 5.80 %, MaxPool 3.13 %, residual `add_` 2.85 %, FC 0.14 %.
- **Winograd is absent; TF32 Tensor-Core implicit-GEMM wins.** The brief predicted Winograd would dominate ResNet-18 3×3 convs. On Blackwell + cuDNN 9.10.2, zero `winograd` kernels appear in the trace. Instead, cuDNN's benchmark search picks:
  - `cutlass_tensorop_s1688fprop_optimized_tf32_64x64_16x10_nhwc_align4` — 28.4 %, 80 calls
  - `sm80_xmma_fprop_implicit_gemm_tf32f32_…_nhwckrsc_nchw` — 18.3 %, 60 calls
  - `sm80_xmma_fprop_implicit_gemm_tf32f32_…_nhwckrsc_nhwc` — 11.8 %, 30 calls
  - Two `implicit_convolve_sgemm` SIMT FP32 variants for shapes that don't fit TC tiles — 11.4 %
  - **58.4 % of total CUDA time goes through Tensor Cores in TF32 math mode** even though we did not enable AMP. PyTorch's default `torch.backends.cuda.matmul.allow_tf32 = True` silently routes ResNet-18 through TF32 on Ampere+.
- **Layout conversions are a real cost:** `nchwToNhwcKernel` (340 invocations) + `nhwcToNchwKernel` (110 invocations) = **9.52 %** of all CUDA time spent just reformatting tensors so the NHWC-preferring TC kernels can run on an NCHW model. This strongly motivates bringing the **`channels_last` experiment (brief §8.5)** forward in priority.
- **Two project-wide issues surfaced in Phase 2 and fixed in the log_2 rework:**
  1. `profile/` as a directory name shadows Python's stdlib `profile` module via `torch._dynamo`'s `cProfile` import chain. Renamed to `profiling/`.
  2. Running `python profiling/run_baseline.py` breaks cross-package imports because `sys.path[0]` becomes the script's folder. Canonical invocation is `python -m profiling.run_baseline …` from the repo root.

A controlled TF32-off re-profile is queued as a future experiment to quantify how much of the Winograd-vs-implicit-GEMM flip is driven by TF32 specifically.

### Phase 3 findings — three more models

- **MobileNetV3-Small (3.01 ms / 10 644 img/s).** Regular conv 31.7 %, depthwise conv 25.7 %, **BN 21.8 %** (much higher than the brief's "10 % misc" prediction — BN amortises badly over tiny convs), hardswish activation 5.6 %, TC share drops to 14.9 %. Depthwise conv routes through PyTorch-native kernels (`aten::_conv_depthwise2d`), not cuDNN.
- **DistilBERT-base (12.36 ms / 648 samples/s).** 91.89 % in `aten::addmm` backed by **MAGMA's `magma_sgemmEx_kernel`**, *not* cuBLAS as the brief predicted. **Zero Tensor-Core engagement** at FP32. Attention is fully fused into a single `fmha_cutlassF_f32_aligned_64x64_rf_sm80` FlashAttention kernel (4.66 %) — no separate softmax row. This is the strongest finding so far: DistilBERT was meant to be the TC showcase and delivered 0 % TC share instead.
- **Tiny GRU (0.25 ms / 127 003 samples/s).** `aten::_cudnn_rnn` at 96.1 %, with the persistent `RNN_blockPersist_fp_GRU` kernel alone taking 73.3 %. Unexpected TC engagement (16.8 %) via `cutlass_80_tensorop_s1688gemm_128x256` on the input-to-hidden matmul — brief predicted "memory-bound, modest TC"; "modest" holds.

Two brief predictions fail outright (MobileNetV3 kernel distribution — BN underestimated; DistilBERT library dispatch — MAGMA won over cuBLAS). Both are real findings that go into the writeup.

### Phase 5 findings — what the Nsight timeline adds

- **`cudnn.benchmark=True` warmup cost is proportional to unique conv-shape count.** ResNet-18 warms up in 468.8 ms; **MobileNetV3-Small takes 5.309 s** — an 11× difference driven by MobileNetV3's ~50+ distinct depthwise + pointwise + stride variants each needing an algorithm probe. The CUDA HW row is *sparse* during MobileNetV3 warmup because the GPU is idle while cuDNN's host-side search runs.
- **Winograd is not absent on Blackwell — it's just not picked in steady state.** The short-warmup Nsight capture for ResNet-18 contains a `cudnn::winograd_nonfused::winogradForwardFilter4x4` kernel at **3.0 %** of GPU time. Phase 2–4's full-warmup baselines reported zero Winograd; the Nsight capture catches cuDNN mid-search. This refines the §4.2 "Winograd is absent" claim into "Winograd is probed but not selected".
- **DistilBERT does probe cuBLAS at the API level.** §5.2.2's MAGMA claim was strictly a kernel-row observation; the Nsight `cuBLAS` API row does show narrow ticks per `torch.addmm` call, even though the actual GEMM lands in MAGMA. Updated writeup §5.6.3 accordingly.
- **PyTorch Profiler ↔ Nsight cross-check passes at worst 17.1 %.** Two models (ResNet-18, GRU) agree within 2.1 %. MobileNetV3 (+14.15 %) and DistilBERT (-17.10 %) disagree by larger margins — explained by the short-warmup capture averaging across warmup + timing + profiler iters while PyTorch Profiler measures only 10 steady-state iters. `python -m analysis.cross_check_nsight` is the reproducible regression test.
- **`cudaDeviceSynchronize` is visible on the CUDA API row** in all four captures (sourced from `torch.cuda.synchronize()` at the end of each profiled iter). The chrome-trace attributed that time to a gap between kernels rather than to an API call — Nsight makes the attribution explicit.
- **Minor audit artefact:** `[flags]` snapshot at startup reveals `torch.backends.cuda.matmul.allow_tf32 = False` on PyTorch 2.10.0+cu128, contradicting the prose claim at the end of the ResNet-18 findings section. ResNet-18's TC share still comes from the cuDNN path (`cudnn.allow_tf32 = True` is still default), so the headline numbers are unaffected; small clarification queued.

All eight Nsight screenshots + full per-subsection discussion in writeup [§5.6](writeup/final_report.md). Binary reports under [`results/nsys/`](results/nsys/); text-mode stats CSVs under [`results/nsys/stats/`](results/nsys/stats/).

---

## The model zoo (four corners of the quadrant)

| Model | Params | Class | Input shape | What it demonstrates |
|---|---|---|---|---|
| **ResNet-18** | 11 M | Conv, compute-bound | (B, 3, 224, 224) | TF32 Tensor-Core implicit-GEMM (see Status above — Winograd was predicted but didn't win on Blackwell) |
| **MobileNetV3-Small** | 2.5 M | Depthwise conv, memory-bound | (B, 3, 224, 224) | Why Tensor Cores stop helping |
| **DistilBERT-base** | 66 M | Matmul, compute-bound | (B, 512) tokens | cuBLAS dominates, sequence-length sweep |
| **Tiny GRU** | 0.2 M | RNN, memory-bound | (B, 100, 64) | cuDNN fused RNN kernel |

The four are chosen to fill a clean 2×2 matrix: `{conv, matmul} × {compute-bound, memory-bound}`. Expect roughly 30–40 distinct kernel names across the whole study.

---

## Experiments

| # | Experiment | What it shows |
|---|---|---|
| 1 | **Baseline profile** | Per-model kernel breakdown at batch 32, FP32 |
| 2 | **`cudnn.benchmark` toggle** | Speedup from cuDNN's algorithm search |
| 3 | **FP32 vs FP16 (AMP)** | Tensor-Core speedup per model |
| 4 | **Batch-size sweep** | Launch-overhead regime → steady-state |
| 5 | **Channels-last layout** *(optional)* | Impact of NHWC on vision models |
| 6 | **Sequence-length sweep** *(optional)* | O(seq²) attention cost on DistilBERT |

For each experiment we record timing mean ± std over ≥ 5 trials with GPU-event timers, after proper warm-up.

---

## Hardware & software

- **GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell, compute capability `sm_120`, 12 GB GDDR7)
- **Driver:** 592.01 (reports CUDA 13.1 runtime-compatible; cu128 wheel works fine)
- **CUDA Toolkit:** 12.8 (for `nsys` / optional `nvcc`; PyTorch bundles its own 12.8 runtime)
- **cuDNN:** 9.10.2 (`torch.backends.cudnn.version() == 91002`, bundled inside the PyTorch CUDA-12.8 wheel)
- **OS:** Windows 11 Home 10.0.26200

> A non-cu128 PyTorch wheel fails silently on Blackwell with *"no kernel image available for execution on the device."* Always use `--index-url https://download.pytorch.org/whl/cu128`.

### Exact dependency versions (installed in the `hdai` env)

Core runtime:

| Package | Version | Source |
|---|---|---|
| python | 3.11.15 | conda |
| torch | 2.10.0+cu128 | `pytorch.org/whl/cu128` |
| torchvision | 0.25.0+cu128 | `pytorch.org/whl/cu128` |
| torchaudio | 2.11.0+cu128 | `pytorch.org/whl/cu128` |
| numpy | 2.4.3 | pypi |
| pandas | 3.0.2 | pypi |
| matplotlib | 3.10.8 | pypi |
| seaborn | 0.13.2 | pypi |
| pillow | 12.1.1 | pypi |

Profiling & modelling:

| Package | Version | Purpose |
|---|---|---|
| transformers | 5.5.4 | DistilBERT model loader |
| tokenizers | 0.22.2 | DistilBERT tokenizer |
| huggingface-hub | 1.11.0 | model download |
| safetensors | 0.7.0 | model weight format |
| nvtx | 0.2.15 | custom NVTX range annotations |
| fvcore | 0.1.5.post20221221 | FLOP counting for roofline |
| ptflops | 0.7.5 | alt FLOP/MAC counter |

PyTorch pulls in sympy 1.14.0, networkx 3.6.1, filelock 3.25.2, jinja2 3.1.6, markupsafe 3.0.3, fsspec 2026.2.0, typing-extensions 4.15.0, mpmath 1.3.0 automatically.

---

## Setup

```powershell
# 1. Create the conda env (Python 3.11)
conda create -n hdai python=3.11 -y
conda activate hdai

# 2. Install PyTorch 2.10.0 + cu128 (sm_120 support) — pinned to match this study
pip install torch==2.10.0 torchvision==0.25.0 torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install the rest
pip install pandas matplotlib seaborn nvtx transformers fvcore ptflops
```

On Git Bash (Windows) `conda` is not on PATH by default; activate via:

```bash
source /c/Users/worka/anaconda3/etc/profile.d/conda.sh
conda activate hdai
```

Verify the environment (with `hdai` activated):

```bash
python env/check_env.py
```

Expected output includes `Device: NVIDIA GeForce RTX 5070 Ti Laptop GPU`, `Compute capability: sm_120`, cuDNN `91002`, and a passing cuDNN conv smoke test.

**Nsight Systems** (2025.x) is required for the timeline inspection step (brief Phase 5) — install separately from NVIDIA's developer site and add to `PATH`. It is not yet installed in this environment.

---

## Repository layout

```
HDAI_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── brief.md                # full project plan (phase-by-phase, appendices)
│   ├── execution_log_0.md      # bootstrap log: env setup, version choices
│   ├── execution_log_1.md      # Phase 2 log: first ResNet-18 profile (superseded)
│   └── execution_log_2.md      # Phase-2 rework: bug fixes, multi-trial rerun, plots
├── env/
│   ├── check_env.py            # verify GPU, cuDNN, PyTorch versions
│   └── sanity_conv.py          # 10-line conv to confirm cuDNN dispatch
├── models/
│   ├── __init__.py
│   ├── resnet.py               # ResNet-18 loader
│   ├── mobilenet.py            # MobileNetV3-Small loader (Phase 3)
│   ├── distilbert.py           # DistilBERT-base loader (Phase 3)
│   └── gru.py                  # TinyGRU 2-layer loader (Phase 3)
├── profiling/                  # NOTE: renamed from `profile/` to avoid stdlib collision
│   ├── __init__.py
│   └── run_baseline.py         # multi-trial CUDA-event timing + profiler trace; MODEL_LOADERS dispatch
├── analysis/
│   ├── __init__.py
│   ├── parse_trace.py          # chrome-trace JSON -> per-kernel (time, #calls)
│   ├── classify_kernels.py     # kernel-name -> coarse category (16 buckets after Phase 4)
│   ├── plots.py                # per-model breakdowns + cross-model comparison plots
│   ├── compute_summary.py      # Phase-4 CSV emitter (baseline_breakdown.csv)
│   └── cross_check_nsight.py   # Phase-5 regression: PyTorch Profiler vs Nsight agreement
├── profiling/
│   ├── run_baseline.py         # NVTX-instrumented (Phase 5) multi-trial baseline driver
│   └── run_nsight.sh           # Phase-5 Nsight capture driver (all four models)
├── results/
│   ├── traces/                 # chrome-trace JSONs (committed, Phase 3)
│   ├── nsys/                   # Phase-5 Nsight .nsys-rep binary reports (committed)
│   │   └── stats/              # Phase-5 nsys stats CSVs (kern_sum + api_sum per model)
│   ├── plots/                  # analysis PNGs + nsight_* screenshots (committed)
│   └── tables/                 # CSV summaries — baseline_breakdown.csv (Phase 4)
└── writeup/
    └── final_report.md         # the main writeup, §§1-5.6 populated
```

Traces and plots under `results/` are **committed** so the repo reproduces the paper's numbers without needing to rerun the profiler. Phase-3 onward additions (more models, experiment drivers, Nsight reports, roofline CSVs, an overnight runner) will add folders as they are actually needed — we don't pre-scaffold empty directories.

---

## Reproduce

Activate the env first (`source /c/.../conda.sh && conda activate hdai` on Git Bash, plain `conda activate hdai` on PowerShell/cmd).

Scripts live in packages (`models/`, `profiling/`, `analysis/`) and are invoked as modules from the repo root so `sys.path` includes the top-level:

```bash
# Smoke tests
python env/check_env.py
python env/sanity_conv.py

# Conda-env SSL_CERT_FILE workaround (needed for DistilBERT HF download on this machine)
unset SSL_CERT_FILE

# All four baselines (batch auto-defaults per DEFAULT_BATCH; DistilBERT = 8, others = 32)
python -m profiling.run_baseline --model resnet18
python -m profiling.run_baseline --model mobilenetv3
python -m profiling.run_baseline --model distilbert
python -m profiling.run_baseline --model gru

# Alternate configurations
python -m profiling.run_baseline --model resnet18 --no-benchmark   # benchmark-off control
python -m profiling.run_baseline --model resnet18 --batch 64       # different batch

# Analysis: per-kernel table of a saved trace
python -m analysis.parse_trace results/traces/resnet18_baseline_bs32_benchOn.json

# Analysis: render all 8 plots (4 per-model breakdowns + 3 cross-model + ResNet conv-algo)
python -m analysis.plots

# Analysis: emit the Phase-4 cross-model summary CSV
python -m analysis.compute_summary

# Nsight Systems capture (Phase 5): all four models, all artefacts in one shot
bash profiling/run_nsight.sh

# Cross-check PyTorch Profiler vs Nsight (regression test)
python -m analysis.cross_check_nsight
```

Phase 4's kernel-classification summary CSV ([`results/tables/baseline_breakdown.csv`](results/tables/baseline_breakdown.csv)) is produced by [`analysis/compute_summary.py`](analysis/compute_summary.py). Phase 5's Nsight capture is driven by [`profiling/run_nsight.sh`](profiling/run_nsight.sh) and consumes the NVTX instrumentation added to [`profiling/run_baseline.py`](profiling/run_baseline.py) in the same phase; the cross-check script [`analysis/cross_check_nsight.py`](analysis/cross_check_nsight.py) compares per-iter GPU time between the two profilers. Phases 6+ (benchmark-toggle, AMP, batch-sweep, roofline) will add more CLI entry points under `profiling/` and `analysis/`.

---

## Deliverables

1. **`writeup/final_report.md`** — 8–12 page profiling report with per-model sections and cross-model observations.
2. **Six final figures** in `results/plots/`:
   - `fig1_time_breakdown.png` — stacked bar, 4 models × 5 kernel categories
   - `fig2_fp16_speedup.png` — grouped bar of FP32→FP16 speedup per model
   - `fig3_batch_scaling.png` — throughput vs. batch size
   - `fig4_algorithm_distribution.png` — which cuDNN algorithms are picked
   - `fig5_roofline.png` — four models on a log-log roofline plane
   - `fig6_nsight_timeline.png` — Nsight Systems screenshot
3. **CSV tables** in `results/tables/` for every experiment.
4. **Chrome traces & Nsight reports** for spot-checking individual runs.

---

## Non-goals

- No training, no fine-tuning — pretrained weights everywhere.
- No `torch.compile` — we want cuDNN's native behaviour, not fused Triton graphs.
- No multi-GPU, no cross-framework (JAX/TF) comparison, no TensorRT.
- No custom CUDA kernels — characterise the stack, don't replace it.
- No training-loop profiling — inference only.

---

## References

Read these before writing any code:

- [PyTorch Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA DLProf/PyProf blog post](https://developer.nvidia.com/blog/profiling-and-optimizing-deep-neural-networks-with-dlprof-and-pyprof/)
- [`Slimakanzer/cudnn-benchmark`](https://github.com/Slimakanzer/cudnn-benchmark) — raw C-API cuDNN usage
- [`google/nvidia_libs_test`](https://github.com/google/nvidia_libs_test) — reference benchmark harness
- [`soumith/convnet-benchmarks`](https://github.com/soumith/convnet-benchmarks) — classic methodology template

Full background, phase-by-phase plan, kernel-name decoder, and troubleshooting appendices live in [`docs/brief.md`](docs/brief.md). The bootstrap log (environment setup, version choices, discrepancies observed) is in [`docs/execution_log_0.md`](docs/execution_log_0.md).
