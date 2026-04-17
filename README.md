# HDAI — cuDNN Profiling Study on Blackwell

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

## The model zoo (four corners of the quadrant)

| Model | Params | Class | Input shape | What it demonstrates |
|---|---|---|---|---|
| **ResNet-18** | 11 M | Conv, compute-bound | (B, 3, 224, 224) | Winograd, clean Tensor-Core speedup |
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

Verify the environment:

```powershell
python env/check_env.py
```

Expected output includes `Device: NVIDIA GeForce RTX 5070 Ti Laptop GPU`, `Compute capability: sm_120`, cuDNN `91002`, and a passing cuDNN conv smoke test.

**Nsight Systems** (2025.x) is required for the timeline inspection step (brief Hour 5) — install separately from NVIDIA's developer site and add to `PATH`. It is not yet installed in this environment.

---

## Repository layout

```
HDAI_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── brief.md                # full project plan (hour-by-hour, appendices)
│   └── execution_log_0.md      # bootstrap log: every step taken so far
├── env/
│   ├── check_env.py            # verify GPU, cuDNN, PyTorch versions
│   └── sanity_conv.py          # 10-line conv to confirm cuDNN path
├── models/
│   ├── __init__.py
│   ├── resnet.py
│   ├── mobilenet.py
│   ├── distilbert.py
│   └── gru.py
├── profile/
│   ├── __init__.py
│   ├── run_baseline.py
│   ├── run_benchmark_toggle.py
│   ├── run_amp.py
│   ├── run_batch_sweep.py
│   ├── run_channels_last.py
│   └── run_seq_sweep.py
├── analysis/
│   ├── __init__.py
│   ├── parse_trace.py
│   ├── classify_kernels.py
│   ├── compute_roofline.py
│   └── plots.py
├── scripts/
│   └── run_all.ps1             # overnight batch runner (brief §25)
├── results/
│   ├── traces/                 # chrome-trace JSONs (gitignored)
│   ├── nsys/                   # Nsight Systems reports (gitignored)
│   ├── tables/                 # CSV summaries
│   └── plots/                  # final PNGs (committed)
└── writeup/
    ├── findings.md
    └── plots/
```

Large binaries (`results/traces/*.json`, `results/nsys/*.nsys-rep`) are gitignored and regeneratable. Currently only `README.md`, `requirements.txt`, `.gitignore`, the two docs under `docs/`, and empty `__init__.py` placeholders exist — scripts are written in brief Hours 2–10.

---

## Reproduce

```powershell
# Smoke test
python env/check_env.py

# Run all baselines
python profile/run_baseline.py --model resnet18
python profile/run_baseline.py --model mobilenetv3
python profile/run_baseline.py --model distilbert
python profile/run_baseline.py --model gru

# Experiments
python profile/run_benchmark_toggle.py --model resnet18
python profile/run_amp.py             --model resnet18
python profile/run_batch_sweep.py     --model resnet18 --batches 1,4,16,64,256

# Nsight capture
nsys profile -t cuda,cudnn,cublas,nvtx -o results/nsys/resnet18 ^
    python profile/run_baseline.py --model resnet18

# Analysis + plots
python analysis/classify_kernels.py results/traces/resnet18_baseline.json
python analysis/plots.py
```

An overnight-style runner that does all four models × all experiments lives at `scripts/run_all.ps1`.

---

## Deliverables

1. **`writeup/findings.md`** — 8–12 page profiling report with per-model sections and cross-model observations.
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

Full background, hour-by-hour plan, kernel-name decoder, and troubleshooting appendices live in [`docs/brief.md`](docs/brief.md). The bootstrap log (environment setup, version choices, discrepancies observed) is in [`docs/execution_log_0.md`](docs/execution_log_0.md).
