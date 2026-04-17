# cuDNN Profiling Project — Complete Plan

**Course:** EEL71020 — Hardware Design for AI, IIT Jodhpur
**Authors:** Anshul Kumar (M25AI2036), Neha Prasad (M25AI2076)
**Target hardware:** RTX 5070 Ti Laptop GPU (Blackwell, sm_120, 12 GB), Windows 11
**Backup:** Colab Pro (L4 or A100)
**Time budget:** 10–12 hours end-to-end
**Deliverable:** profiling report + plots + a small repo of scripts

---

## Progress tracker

| Phase | Status | Notes |
|---|---|---|
| Phase 0 — pre-reading | [x] done | brief fully read, reference links skimmed |
| Phase 1 — environment setup | [x] done | `hdai` conda env, torch 2.10.0+cu128, cuDNN 91002, sm_120 smoke tests pass |
| Repo scaffolding (section 5) | [x] done | `profile/` renamed to `profiling/` — see §11 gotcha |
| Phase 2 — first profile on ResNet-18 | [x] done | 97.923 ms CUDA / 10 iters @ batch 32. Winograd absent, TF32 TC implicit-GEMM wins. See `execution_log_1.md`. |
| Phase 3 — port to MobileNetV3 / DistilBERT / GRU | [ ] next | reuse the `run_baseline.py` harness, add model loaders |
| Phase 4 — kernel classification | [ ] pending | |
| Phase 5 — Nsight Systems timeline | [ ] blocked on Nsight install | |
| Phase 6 — `cudnn.benchmark` toggle | [ ] pending | |
| Phase 7 — FP32 vs FP16 (AMP) | [ ] pending | |
| Phase 8 — batch-size sweep | [ ] pending | cap at what fits in 12 GB, not 16 GB |
| Phase 9 — roofline analysis | [ ] pending | |
| Phase 10 — cleanup and plots | [ ] pending | `channels_last` promoted in priority — layout converts are 9.72% of CUDA time in baseline |
| Phase 11 — writeup | [ ] pending | |
| Phase 12 — buffer | [ ] pending | |

**Extra experiment queued (not in original plan):** TF32-off re-profile of ResNet-18 (`torch.backends.cuda.matmul.allow_tf32 = False` + `torch.backends.cudnn.allow_tf32 = False`). Expected to make Winograd competitive again; would be a strong paired-bar figure for the writeup.

### Findings so far (end of Phase 2, one model profiled)

- **ResNet-18 latency @ batch 32, FP32 default:** 9.79 ms/iter → ~3 267 img/s.
- **Conv is 78.80 %** of CUDA time, as predicted.
- **Winograd predicted, TF32 Tensor-Core implicit-GEMM observed.** The brief's §1 prediction ("almost every conv layer gets Winograd") does not hold on Blackwell + cuDNN 9.10.2 + PyTorch's default TF32-on config. Top kernels are `cutlass_tensorop_s1688fprop_optimized_tf32_...` and `sm80_xmma_fprop_implicit_gemm_tf32f32_...`. This is a genuine, documented finding, not a profiling bug.
- **TF32 is active by default** because PyTorch enables `allow_tf32=True` on `sm_80+`. "FP32 inference" on Blackwell silently uses Tensor Cores in TF32 math mode.
- **Layout-conversion kernels eat 9.72 %** of CUDA time (440 NCHW↔NHWC converts per 10 iterations). Motivates doing the `channels_last` experiment (§8.5) sooner.

### Environment subsections completed

- 2.1 `nvidia-smi` verified — driver 592.01, CUDA 13.1 reported, 12 GB
- 2.3 conda env (`hdai` with Python 3.11.15)
- 2.4 PyTorch cu128 wheel (torch 2.10.0+cu128, torchvision 0.25.0+cu128, torchaudio 2.11.0+cu128)
- 2.5 project packages (pandas, matplotlib, seaborn, nvtx, transformers, fvcore, ptflops; pillow pulled transitively)
- 2.8 cuDNN reachability smoke test (3×3 conv on 16×64×56×56 dispatches correctly)

Still to do from section 2:
- 2.2 CUDA Toolkit 12.8 (optional — needed only for standalone `nvcc` / the Nsight installer bundle)
- 2.6 Nsight Systems 2025.x (required before Phase 5)
- 2.7 Nsight Compute (optional stretch)

See `execution_log_0.md` for the full bootstrap trace, `execution_log_1.md` for the Phase 2 profile.

---

## 0. What this project actually is

The course prompt is "analyze cuDNN and profile multiple DL models." In practice that means answering three concrete questions for each model we pick:

1. Where does the time actually go? How much is convolution, how much is matmul, how much is elementwise work, how much is overhead?
2. Which cuDNN algorithms does the library choose, and why? (GEMM vs Winograd vs FFT vs implicit-GEMM; plain kernels vs Tensor Core kernels.)
3. Is this model compute-bound or memory-bound on this hardware, and can we prove it with numbers rather than vibes?

The project is mostly an exercise in *reading profiler output*, not in writing new systems. The PyTorch Profiler already emits every cuDNN kernel call by name with timings. Nsight Systems already draws the timeline. Our job is to run these tools on a carefully chosen model zoo, tabulate results, and explain patterns.

This plan is deliberately scoped so it can be done in ~10 hours by someone who has never touched cuDNN internals before. It is not research. It is a plug-and-play profiling study executed with care.

---

## 1. The four models and why each one

We want a small zoo that spans the interesting axes: conv vs matmul, compute-bound vs memory-bound, small filters vs large ones, dense channels vs depthwise. Four models is the sweet spot — enough to see patterns, not so many that each one gets shallow treatment.

### Model 1 — ResNet-18 (compute-bound conv, canonical)

torchvision model, ~11M parameters, standard ImageNet classification. We pick 18 over 50 deliberately: 18 fits in a second per inference at batch 32 on the 5070, it's the most-studied model in all of convnet benchmarking history, and its layer structure (3×3 convs, residual adds, batchnorm) is exactly the pattern cuDNN was optimized for. When `cudnn.benchmark=True` is on, almost every conv layer in ResNet-18 gets a Winograd algorithm picked, which is the most characteristically-cuDNN behavior you can observe.

What we expect to see in the profile: 70–80% of inference time in `cudnn::...winograd...` or `sm80_xmma_gemm_...` kernels (depending on FP32 vs FP16), a small slice in `batch_norm`, negligible elementwise time, and single-digit-percent overhead. This is the "healthy compute-bound CNN" reference point.

> **Observed on this hardware (Phase 2, 2026-04-16).** Conv came in at 78.80% — the "70–80%" band holds. But *zero* Winograd kernels appeared. PyTorch's default `allow_tf32=True` steered cuDNN's benchmark search toward TF32 Tensor-Core implicit-GEMM instead: 41.88% of CUDA time in `cutlass_tensorop_s1688fprop_optimized_tf32_64x64_16x10_nhwc_align4` and 13.96% in `sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_...`. On Blackwell + cuDNN 9.10.2, Blackwell's 5th-gen Tensor Cores appear to beat any Winograd variant cuDNN has compiled for `sm_120`. The prediction in this paragraph holds in *spirit* (conv dominates, Tensor Cores engage); the specific *algorithm name* is different. See `execution_log_1.md §5–§6` for the full kernel inventory.

### Model 2 — MobileNetV3-Small (memory-bound conv, depthwise-dominated)

torchvision model, ~2.5M parameters. The architecture is almost entirely depthwise separable convolutions, which have low arithmetic intensity (~channel_count FLOPs per weight loaded instead of ~kernel_size² × channel_count FLOPs like regular conv). Tensor Cores don't help much because the GEMMs are too skinny.

Why include it: because a huge chunk of modern edge vision uses this pattern, and because the *contrast* with ResNet-18 is where the learning happens. Same task (ImageNet classification), much smaller parameter count, but the profile looks completely different. You see cuDNN's depthwise-specific kernels, you see far less Tensor Core usage, and the speedup from FP16 is muted — maybe 1.2× instead of ResNet's 2–3×. This is the canonical example of "Tensor Cores don't save you if your workload is memory-bound."

What we expect to see: time breakdown roughly 60% depthwise conv + 30% pointwise (1×1) conv + 10% misc, FP16 speedup small, `cudnn.benchmark` helps only modestly because the depthwise kernels have fewer algorithm choices to pick between.

### Model 3 — DistilBERT-base (compute-bound matmul, Tensor Core showcase)

HuggingFace `distilbert-base-uncased`, 66M parameters. Bigger than the other three combined, but it's still well under any memory ceiling and is the natural representative of "transformer that people actually use in production." The inference workload is almost entirely matmul — Q/K/V projections, attention, and FFN layers — with softmax and layer norm as rounding errors.

Important note: most of DistilBERT's matmuls go through cuBLAS, not cuDNN. This is a good thing to observe and discuss in the writeup — cuDNN is the deep-learning library but transformer-heavy workloads hit cuBLAS more than cuDNN. The handful of cuDNN calls that do show up are for layer norm and occasionally softmax on some cuDNN versions.

What we expect to see: 80%+ of time in `cublas` GEMM kernels with Tensor Core variants, FP16 speedup of 2–3× (attention scales well on Tensor Cores), and the role of sequence length as a knob that moves you from memory-bound-ish at seq=64 to clearly compute-bound at seq=512.

### Model 4 — Tiny GRU (memory-bound recurrent, cuDNN RNN path)

Custom-built: 2-layer GRU with hidden size 128, input size 64, sequence length variable. About 200K parameters. Sentiment-classifier-sized. Negligible to train, takes milliseconds to run.

Why include it: to see cuDNN's RNN path, which is genuinely different code from its conv and matmul paths. cuDNN provides a fused RNN/GRU/LSTM kernel (`cudnnRNNForward`) that wraps the entire sequence-length loop into one call. PyTorch uses this automatically via `nn.GRU`. Inside the profiler it shows up as a single persistent cuDNN kernel with the whole sequence rolled in, which looks different from per-timestep kernel launches you'd see in a naive implementation.

Sequence-based RNN inference is memory-bound almost by definition — you're doing a matmul of a tiny matrix (hidden × hidden) at every step, and the weights are tiny relative to the activations you're shuttling around. This gives us our fourth corner: conv compute-bound (ResNet-18), conv memory-bound (MobileNetV3-Small), matmul compute-bound (DistilBERT), matmul memory-bound (GRU). Clean quadrant.

What we expect to see: a single `cudnn::rnn::...` kernel dominating the timeline, FP16 speedup modest (maybe 1.3×), and the "compute" column of the profile looking nothing like the other three models. Batch size scaling is dramatic here — going from batch 1 to batch 128 actually makes GRU look reasonable throughput-wise because you're amortizing the memory traffic.

### Summary table of the zoo

| Model | Params | Ops/s class | Input shape used | What it demonstrates |
|---|---|---|---|---|
| ResNet-18 | 11M | Conv, compute-bound | (B, 3, 224, 224) | Winograd, clean Tensor Core speedup |
| MobileNetV3-Small | 2.5M | Depthwise conv, memory-bound | (B, 3, 224, 224) | Why Tensor Cores stop helping |
| DistilBERT-base | 66M | Matmul, compute-bound | (B, 512) tokens | cuBLAS dominates, sequence length sweep |
| Tiny GRU | 0.2M | RNN, memory-bound | (B, 100, 64) | cuDNN fused RNN kernel |

Total unique kernel patterns across all four: probably 30–40 distinct kernels. That's manageable to catalog.

---

## 2. Hardware and environment setup (the part with actual gotchas)

The RTX 5070 is Blackwell architecture, compute capability sm_120. This is newer than most PyTorch tutorials assume, and the default `pip install torch` will give you a wheel that was compiled without sm_120 support, which fails silently with an opaque "no kernel image available" error at first CUDA call.

### 2.1 Verify hardware and drivers

Open PowerShell. Run:

```powershell
nvidia-smi
```

Expected output: your driver version, CUDA version reported (driver-level, not toolkit-level), and the RTX 5070 with its memory and power state. Write down the driver version — you want it at or above 570.xx for full sm_120 support. If it's older, update from https://www.nvidia.com/Download/index.aspx before doing anything else. This takes 15 minutes and requires a reboot, and it's the single biggest source of "nothing works" failures on Blackwell.

### 2.2 Install CUDA Toolkit 12.8 (optional but helpful)

PyTorch ships its own CUDA runtime inside the wheel, so technically you don't need a separate CUDA Toolkit install to run PyTorch + cuDNN. But you *do* need the CUDA Toolkit for:
- `nvcc` (only if you want to compile raw CUDA code like Slimakanzer/cudnn-benchmark)
- Nsight Systems (usually bundled with CUDA Toolkit installer)
- Nsight Compute (ditto)

Recommended: install CUDA Toolkit 12.8 from https://developer.nvidia.com/cuda-12-8-0-download-archive. Windows Network Installer is simplest. During install, untick the driver component (you already have a newer driver), keep the rest. ~3GB on disk.

### 2.3 Python environment

Use conda or venv. I'll assume conda because it's easier on Windows.

```powershell
conda create -n cudnn python=3.11
conda activate cudnn
```

### 2.4 PyTorch with CUDA 12.8 wheel (this is the critical step)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

This pulls the CUDA 12.8 wheel which has sm_120 kernels compiled in. Without the `--index-url` flag you'll get the cu121 wheel from PyPI by default, and it will fail on Blackwell. The cu128 wheel is ~3GB.

Verify it worked:

```python
import torch
print(torch.__version__)                    # should be 2.7+ or 2.10+
print(torch.version.cuda)                    # should say 12.8
print(torch.backends.cudnn.version())        # should say 91xxx (cuDNN 9.x)
print(torch.cuda.is_available())             # True
print(torch.cuda.get_device_name(0))         # "NVIDIA GeForce RTX 5070"
print(torch.cuda.get_device_capability(0))   # (12, 0)

# Actually run a kernel to confirm sm_120 works
x = torch.randn(1024, 1024, device='cuda')
y = x @ x.T
print(y.shape, y.device)   # if this prints without error, you're good
```

If the matmul fails with "no kernel image available for execution on the device", the wheel is wrong. Reinstall with the right --index-url.

### 2.5 Other Python packages

```powershell
pip install numpy pandas matplotlib seaborn
pip install transformers  # for DistilBERT
pip install pillow        # image input for vision models
pip install nvtx          # for custom profiler range annotations
```

### 2.6 Nsight Systems

Download from https://developer.nvidia.com/nsight-systems (requires free NVIDIA dev account). Get the 2025.x Windows installer. Default install. This gives you the `nsys` command and the GUI visualizer. The GUI is what you want for the timeline view — it's much clearer than Chrome trace for GPU work.

Add Nsight to PATH if the installer doesn't: `C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.x\target-windows-x64`.

### 2.7 Optional: Nsight Compute

Nsight Compute is per-kernel deep analysis (roofline, memory throughput, occupancy) — more than we need for this project but useful if you want to go deeper on individual kernels. Same installer family as Nsight Systems. Skip unless you have time at Phase 10.

### 2.8 Verify cuDNN is actually getting called

Before doing anything else, confirm cuDNN is actually reachable. Run this sanity check:

```python
import torch
import torch.nn.functional as F

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

x = torch.randn(32, 64, 56, 56, device='cuda')
w = torch.randn(128, 64, 3, 3, device='cuda')

# warmup
for _ in range(5):
    y = F.conv2d(x, w, padding=1)
torch.cuda.synchronize()

# time it
import time
t0 = time.perf_counter()
for _ in range(100):
    y = F.conv2d(x, w, padding=1)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"Per-conv time: {(t1-t0)*10:.3f} ms")
```

Should print something in the range of 0.05–0.2 ms on the 5070. If it's 10× higher or throws, something in the cuDNN path is wrong.

### 2.9 About Colab Pro as backup

If the Windows setup misbehaves (it sometimes does on brand-new cards), Colab Pro is the bailout. Pick the L4 or A100 runtime. Everything is pre-installed. The disadvantage is you don't get to observe Blackwell-specific behavior — L4 is Ada (sm_89), A100 is Ampere (sm_80) — so the kernel names in the profile will be different. For the pedagogical goals of this project (understanding cuDNN algorithm selection, Tensor Cores, roofline) that's fine. For the bragging rights of "I profiled Blackwell" you need the local setup.

Recommendation: do the first 2 hours on Colab to get rolling, then switch to local once the scripts work. That way you don't get stuck on environment issues for 3 hours before you've even seen a profile.

---

## 3. Background you need before you start

This is the part where you close the tutorial tabs and read the actual concepts. Budget 45 minutes for this section of the plan before Phase 1.

### 3.1 What cuDNN is and how PyTorch uses it

cuDNN (CUDA Deep Neural Network library) is NVIDIA's hand-tuned C library of primitives for deep learning: convolution, pooling, normalization, activation, RNN. It's not part of CUDA core — it's a separate library that ships alongside the CUDA Toolkit. Each cuDNN version is paired with specific CUDA versions.

PyTorch uses cuDNN as the default backend for operations it implements. When you call `F.conv2d(...)` on a CUDA tensor, PyTorch's dispatcher looks at the input shapes and dtypes, asks cuDNN "what's the best algorithm for this," gets a handle back, and calls into cuDNN to execute. The cuDNN call eventually launches one or more CUDA kernels that do the actual computation on the GPU.

This matters because:
- The kernels that show up in the profile have cuDNN-specific names (starting with `cudnn::` or containing patterns like `implicit_gemm`, `winograd`).
- cuDNN version determines which algorithms are available. cuDNN 9.x (what you'll have) has significantly more kernels than cuDNN 8.x.
- Some operations PyTorch implements itself without cuDNN — these show up with `aten::` names in the profile.

### 3.2 The cuDNN algorithm zoo for convolution

cuDNN provides multiple algorithms for forward convolution. You should recognize these names in the profile:

**GEMM-based (implicit and explicit).** Reshapes the conv into a matrix multiply (im2col + GEMM) and calls cuBLAS. "Implicit GEMM" skips the actual im2col buffer materialization, doing the index arithmetic inline. This is the default for many shapes, especially 1×1 convs where im2col is trivial.

**Winograd.** A mathematical trick that reduces the number of multiplications for small filters (3×3 specifically) at the cost of more additions. About 2.25× fewer multiplies for 3×3. Only works for specific filter sizes. When cuDNN picks Winograd, you'll see `winograd` in the kernel name.

**FFT.** Convolution becomes pointwise multiplication in frequency domain. Only efficient for large filters (7×7 or bigger), which modern networks rarely use. You'll almost never see this one.

**Direct.** Literal nested-loop conv. Used as a last resort for weird shapes.

When `torch.backends.cudnn.benchmark = True`, PyTorch runs a mini-benchmark the first time each (input shape, filter shape) combo appears, picking the fastest algorithm, and caches the choice. Subsequent calls with the same shapes skip the benchmark. When it's False (default), PyTorch uses cuDNN's heuristic which picks an algorithm based on shapes without measuring — fast startup, possibly slower steady-state.

### 3.3 Tensor Cores

Tensor Cores are a specialized matrix-multiply-accumulate unit inside NVIDIA GPUs starting with Volta (sm_70). They do a small matrix multiply (typically 4×4 or 8×8, depending on generation) per cycle. They require:
- FP16, BF16, INT8, or TF32 inputs (not FP32 — TF32 is a special FP32 mode)
- Shapes that are multiples of 8 (or sometimes 16)
- cuDNN or cuBLAS to emit the right kernel variant

When Tensor Cores engage, you see kernel names containing `hmma` (half-precision matrix multiply accumulate), `mma`, `bmma` (BF16), `imma` (INT8), or `tma` on Hopper+. The Blackwell 5070 has 5th-gen Tensor Cores with support for FP8 and even FP4 formats, though PyTorch's default path won't exercise those automatically.

The speedup from Tensor Cores is workload-dependent:
- Large GEMMs with shapes divisible by 8: 2–4× speedup typical
- Small GEMMs: negligible, the launch overhead dominates
- Depthwise convs: barely any speedup, shapes are too skinny
- RNNs: modest, depends on hidden size

### 3.4 The roofline model in one paragraph

Every workload has an **arithmetic intensity**: FLOPs performed per byte of memory moved from DRAM. A matmul of two N×N matrices does 2N³ FLOPs and moves 3N² bytes (two inputs + one output) — intensity ≈ 2N/3, which grows with N. A pointwise addition does 1 FLOP per byte — tiny intensity. Memory bandwidth is fixed at a few TB/s on the 5070; compute throughput is tens of TFLOP/s. If your arithmetic intensity × memory bandwidth < compute throughput, you're **memory-bound** and no amount of Tensor Cores saves you. If it's above, you're **compute-bound** and FP16/Tensor Cores give full speedup. The threshold (the "ridge point") is around 10 FLOPs/byte for FP32 and higher for FP16. Every number you report should be classifiable as one side or the other.

### 3.5 What PyTorch Profiler gives you

`torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True)` is the workhorse. It records:
- Every op call at the PyTorch level (`aten::conv2d`, `aten::linear`, etc.)
- Every CUDA kernel launched, with exact duration
- CPU-side time for each op
- Memory allocations (optional)

Key methods on the output:
- `.key_averages().table(sort_by="cuda_time_total", row_limit=20)` — tabular breakdown
- `.export_chrome_trace("trace.json")` — opens in chrome://tracing
- `.export_stacks(...)` — flamegraph-style output

For CUDA kernel names specifically, use `.key_averages(group_by_input_shape=True)` to get shape-aware aggregation.

### 3.6 What Nsight Systems adds

PyTorch Profiler tells you what PyTorch knows about. Nsight Systems gives you the system-level timeline: CUDA API calls, kernel executions, memory transfers, CPU threads, all on one scrollable timeline. It's invaluable for spotting gaps (where the GPU is idle waiting for CPU) and launch overhead (tiny kernels stacking up). For this project we'll use it mainly for the pretty visualization and for Phase 4–5 investigation when PyTorch Profiler alone isn't giving enough detail.

---

## 4. Reference code to read before you start

Don't write anything until you've read these. This is the most important Phase 0 investment.

### 4.1 PyTorch Profiler official recipe

https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

Short, self-contained, works out of the box. Gets you from zero to a profiled ResNet-18 in about 30 lines of code. Read it entirely, run the example locally, inspect the output table. This is the minimum viable starting point.

### 4.2 NVIDIA DLProf/PyProf blog post

https://developer.nvidia.com/blog/profiling-and-optimizing-deep-neural-networks-with-dlprof-and-pyprof/

This walks through profiling ResNet-50 training end-to-end and is essentially the blueprint for our project. DLProf and PyProf are older tools (replaced by Nsight and native PyTorch Profiler) but the methodology is identical. Read it for the methodology, skip the tool-specific commands. The "top time-consuming kernels" section is exactly what we'll produce.

### 4.3 Slimakanzer/cudnn-benchmark

https://github.com/Slimakanzer/cudnn-benchmark

Raw CUDA single-file benchmark of cuDNN convolution algorithms. ~400 lines of C++. You don't have to run it (it needs nvcc and some work to build on Windows), but *read the source*. It shows you how cuDNN is called at the C API level: `cudnnCreate`, `cudnnCreateConvolutionDescriptor`, `cudnnFindConvolutionForwardAlgorithm`, `cudnnConvolutionForward`. This is what PyTorch is doing under the hood. An hour reading this source gives you more intuition for what cuDNN "really is" than a week of reading PyTorch documentation.

### 4.4 google/nvidia_libs_test

https://github.com/google/nvidia_libs_test

Google's test suite for cuDNN. Heavier — Bazel build, thousands of lines. Skim the `cudnn_benchmark.cc` file to see how they define benchmark problem sets via textproto and iterate over them. You won't run this but the structure is worth understanding — it's how a real systems team benchmarks cuDNN across hundreds of shapes.

### 4.5 soumith/convnet-benchmarks

https://github.com/soumith/convnet-benchmarks

Classic. Old (Titan Black / Titan X era) but the methodology is the template for all convnet benchmarking after it. The results tables are structured exactly the way yours should be — model × framework × layer × time. Copy the aesthetic.

### 4.6 Reproducibility and cudnn.benchmark writeup

https://darinabal.medium.com/deep-learning-reproducible-results-using-pytorch-42034da5ad7

Short article on `torch.backends.cudnn.benchmark` and `torch.backends.cudnn.deterministic`. Read it so you understand the knobs you'll be flipping in Phase 6.

### 4.7 What NOT to read

Skip anything about DeepSpeed, Megatron, vLLM, FSDP, or LLM inference optimization. That's a different project. Also skip anything about TensorRT — TensorRT is an inference engine that replaces cuDNN for deployed models, and it's outside scope.

---

## 5. Project layout

Create this structure on your local machine:

```
hdai-project/
├── README.md
├── requirements.txt
├── env/
│   ├── check_env.py          # verify GPU, cuDNN, PyTorch versions
│   └── sanity_conv.py        # 10-line conv to confirm cuDNN path works
├── models/
│   ├── __init__.py
│   ├── resnet.py             # wrapper for torchvision ResNet-18
│   ├── mobilenet.py          # wrapper for torchvision MobileNetV3-Small
│   ├── distilbert.py         # wrapper for HuggingFace DistilBERT
│   └── gru.py                # tiny custom GRU
├── profiling/                # NOTE: NOT `profile/` — that name collides with Python's stdlib
│   ├── __init__.py           # `profile` module via torch._dynamo's cProfile import chain.
│   ├── run_baseline.py       # profile each model at baseline settings
│   ├── run_benchmark_toggle.py  # compare cudnn.benchmark on/off
│   ├── run_amp.py            # FP32 vs FP16 (autocast)
│   ├── run_batch_sweep.py    # batch sizes [1, 8, 32, 128]
│   ├── run_channels_last.py  # NHWC vs NCHW (vision models only)
│   └── run_seq_sweep.py      # sequence length sweep for DistilBERT + GRU
├── analysis/
│   ├── parse_trace.py        # convert chrome traces to pandas dataframes
│   ├── classify_kernels.py   # label kernels as conv/matmul/norm/etc.
│   ├── compute_roofline.py   # arithmetic intensity + measured throughput
│   └── plots.py              # matplotlib/seaborn plotting
├── results/
│   ├── traces/               # chrome trace JSONs
│   ├── nsys/                 # Nsight Systems .nsys-rep files
│   ├── tables/               # CSV output of kernel-level data
│   └── plots/                # PNGs
└── writeup/
    ├── findings.md           # the main writeup
    └── plots/                # copy of plots used in the writeup
```

Keep results out of git (or gitignored). The trace JSONs get large — hundreds of MB for a batch of 128 at seq 512.

### 5.1 requirements.txt

```
torch>=2.7
torchvision
transformers>=4.40
pandas
numpy
matplotlib
seaborn
nvtx
```

Don't pin exact versions. The Blackwell-compatible wheel changes frequently; let pip resolve.

---

## 6. Phase-by-phase execution plan

### Phase 0 (the pre-work): 30–45 minutes

Read the Reference Code section (section 4) above. Don't skip this. Open the PyTorch profiler recipe in one tab, the Slimakanzer cudnn-benchmark source in another, and skim both.

### Phase 1: Environment setup

Follow section 2 end to end. By the end of this phase:
- `nvidia-smi` works
- PyTorch with cu128 wheel installed
- `env/check_env.py` prints GPU name, cuDNN version, successful matmul
- Nsight Systems installed and `nsys --version` works

If setup takes longer than 90 minutes (common on Windows), don't push through — switch to Colab for the rest of the project and come back to local setup later. Don't burn 3 hours on driver issues.

### Phase 2: First profile on ResNet-18

Write `models/resnet.py` — literally a five-line wrapper:

```python
import torchvision.models as tvm
def get_model():
    return tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1).eval().cuda()
```

Write `profiling/run_baseline.py` (`profile/` collides with Python stdlib — see §11):

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from models.resnet import get_model

model = get_model()
x = torch.randn(32, 3, 224, 224, device='cuda')

# warmup (important!)
for _ in range(10):
    with torch.no_grad():
        _ = model(x)
torch.cuda.synchronize()

# profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    for _ in range(20):
        with torch.no_grad():
            with record_function("inference"):
                _ = model(x)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
prof.export_chrome_trace("results/traces/resnet18_baseline.json")
```

Run it. Look at the table output. Open `results/traces/resnet18_baseline.json` in `chrome://tracing` (or use Perfetto UI at https://ui.perfetto.dev/).

Things to notice and screenshot:
- Top cuDNN kernels by cumulative time
- Presence/absence of `winograd` and `implicit_gemm` in kernel names
- Ratio of CUDA time to CPU time (should be dominated by CUDA for batch 32)

**Milestone:** by end of Phase 2, you have a working profile script, and you've *looked at* the output. Don't move on until you can point at a specific kernel name in the output and say what it does.

### Phase 3: Port to all four models

Write the wrappers for the other three models. The pattern:

```python
# models/mobilenet.py
import torchvision.models as tvm
def get_model():
    return tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval().cuda()
def get_input(batch=32):
    return torch.randn(batch, 3, 224, 224, device='cuda')

# models/distilbert.py
from transformers import DistilBertModel, DistilBertTokenizerFast
_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
def get_model():
    return DistilBertModel.from_pretrained('distilbert-base-uncased').eval().cuda()
def get_input(batch=8, seq=128):
    # fake token ids in a valid range
    return torch.randint(0, 30000, (batch, seq), device='cuda')

# models/gru.py
import torch.nn as nn
class TinyGRU(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])
def get_model():
    return TinyGRU().eval().cuda()
def get_input(batch=32, seq=100):
    return torch.randn(batch, seq, 64, device='cuda')
```

Refactor `run_baseline.py` to loop over all four models. Each one gets its own chrome trace in `results/traces/`.

At the end of Phase 3: four JSON traces + four key_averages tables. Open each trace briefly. Note any surprises — e.g., if you see no `cudnn::` kernels in DistilBERT that's correct (it's cuBLAS-heavy).

### Phase 4: Kernel classification and summary table

This is where you turn profiling data into *information*. Write `analysis/parse_trace.py` to load a chrome trace JSON, extract the CUDA kernel events, and dump to a pandas DataFrame.

The trace format: each event has `name`, `ph` (phase), `ts` (timestamp μs), `dur` (duration μs), `args` (extra info). Filter to `cat=kernel` or `ph=X` with CUDA-ish names, then aggregate.

Write `analysis/classify_kernels.py`. Given a kernel name, categorize:

```python
def classify_kernel(name):
    n = name.lower()
    if 'winograd' in n: return 'conv_winograd'
    if 'implicit_gemm' in n or 'implicit_precomp' in n: return 'conv_implicit_gemm'
    if 'conv' in n and 'dgrad' not in n: return 'conv_other'
    if 'dgrad' in n or 'wgrad' in n: return 'conv_backward'
    if 'gemm' in n and 'hmma' in n: return 'matmul_tensor_core'
    if 'gemm' in n: return 'matmul_fp32'
    if 'rnn' in n or 'lstm' in n or 'gru' in n: return 'rnn'
    if 'batch_norm' in n or 'batchnorm' in n: return 'norm'
    if 'softmax' in n: return 'softmax'
    if 'elementwise' in n or 'vectorized_elementwise' in n: return 'elementwise'
    if 'reduce' in n: return 'reduce'
    return 'other'
```

Produce a per-model summary table:

| Model | Total CUDA time (ms) | Conv % | Matmul % | Norm % | Elementwise % | Other % |
|---|---|---|---|---|---|---|
| ResNet-18 | ... | ... | ... | ... | ... | ... |
| MobileNetV3-Small | ... | ... | ... | ... | ... | ... |
| DistilBERT | ... | ... | ... | ... | ... | ... |
| Tiny GRU | ... | ... | ... | ... | ... | ... |

Save as CSV to `results/tables/baseline_breakdown.csv`.

By end of Phase 4: one master table with the four models' time distribution. This is the centerpiece figure of the whole project.

### Phase 5: Nsight Systems timeline view

PyTorch Profiler gives you tables. Nsight Systems gives you a picture. Profile one or two models through `nsys`:

```powershell
nsys profile -t cuda,cudnn,cublas,nvtx -o results/nsys/resnet18 python -m profiling.run_baseline --model resnet18
```

Open `results/nsys/resnet18.nsys-rep` in the Nsight Systems GUI. Look at:
- The timeline of kernels (a row per stream)
- The CUDA API row (cuDNN calls are visible here)
- Any gaps in the timeline — those are your serialization bottlenecks

Take a screenshot of the timeline for one full inference. Include in writeup.

This phase is mostly about getting comfortable with the Nsight GUI, which has a learning curve. Don't try to become a Nsight expert; just generate 1–2 traces, look at them, and move on.

### Phase 6: The `cudnn.benchmark` experiment

First structured experiment. Write `run_benchmark_toggle.py`:

```python
import torch
# ... imports

for benchmark_setting in [False, True]:
    torch.backends.cudnn.benchmark = benchmark_setting
    
    model = get_model()
    x = get_input()
    
    # warmup — note this is longer when benchmark=True because it needs to try algorithms
    for _ in range(30):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    
    # time N iterations
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(100):
        with torch.no_grad():
            _ = model(x)
    t1.record()
    torch.cuda.synchronize()
    print(f"benchmark={benchmark_setting}: {t0.elapsed_time(t1)/100:.3f} ms/iter")
```

Run for all four models. Expected results:
- ResNet-18: meaningful speedup (10–30%) with benchmark=True
- MobileNetV3-Small: small speedup (<10%)
- DistilBERT: no change (cuDNN barely in the path)
- GRU: no change (already optimized path)

Put results in a table. This is experiment 1 of 4.

### Phase 7: FP32 vs FP16 (AMP)

Write `run_amp.py`:

```python
import torch

# FP32 baseline
model_fp = get_model()
x = get_input()
# warmup + time...

# FP16 via autocast
with torch.autocast(device_type='cuda', dtype=torch.float16):
    # warmup + time...
```

Per-model expected FP16 speedups (approximate, on 5070):
- ResNet-18: 2–3× (well-matched to Tensor Cores)
- MobileNetV3-Small: 1.2–1.5× (depthwise doesn't benefit as much)
- DistilBERT: 2–3× (attention benefits strongly)
- GRU: 1.2–1.4× (cuDNN RNN is already well-tuned; Tensor Cores help modestly)

The *interesting* part is to re-profile one or two models under FP16 and look at which kernels changed. ResNet-18 should show `hmma`/`mma` in many kernel names now where FP32 had plain `gemm`.

### Phase 8: Batch size sweep

Write `run_batch_sweep.py`. For each model, time inference at batch sizes [1, 4, 16, 64, 256] (or until OOM). For GRU and DistilBERT, also include a sequence length axis (seq=64, 128, 256, 512).

Plot:
- X: batch size (log scale)
- Y: throughput (samples/sec)
- One curve per model, FP32 and FP16 separate

Expected shapes:
- Small batches are dominated by kernel launch overhead — throughput grows with batch
- Large batches saturate — throughput plateaus when GPU is fully utilized
- MobileNetV3-Small saturates later because individual kernels are small and launch overhead matters more
- DistilBERT at seq=512 may approach OOM at large batches on a 12GB card; cap accordingly

This is experiment 3. The batch-scaling curves are a great visual for the writeup.

### Phase 9: Roofline analysis (short version)

Compute arithmetic intensity for each model at batch 32:
- FLOPs: use `fvcore.nn.FlopCountAnalysis` or `torchinfo` or hand-compute for simple cases
- Memory traffic: sum of `param_size + activation_size` for all layers, or use `torch.cuda.max_memory_allocated`
- Intensity = FLOPs / bytes

Plot the four models on a log-log roofline diagram:
- X: arithmetic intensity (FLOPs/byte)
- Y: achieved throughput (FLOPs/sec)
- Overlay the peak compute line (~60 TFLOP/s FP32 for 5070, higher for FP16)
- Overlay the peak memory bandwidth line (~800 GB/s on 5070 GDDR7)

Each model sits somewhere in the plane. Compute-bound models are near the horizontal ceiling; memory-bound models are on the sloped ramp. This plot is the money shot.

Don't over-engineer this. Rough numbers are fine. The point is to show the qualitative story.

### Phase 10: Cleanup and plots

By now you have a results directory full of JSON traces, CSV tables, and screenshots. Consolidate into 5–6 final plots:

1. **Time breakdown per model** (stacked bar): conv, matmul, norm, elementwise, other
2. **FP32 vs FP16 speedup** (grouped bar): one group per model
3. **Batch size scaling** (line chart): throughput vs batch for each model
4. **Algorithm distribution** (stacked bar): per model, proportion of time in winograd / implicit_gemm / gemm_tensorcore / rnn / other
5. **Roofline diagram**: four points on a log-log plane with ceilings overlaid
6. **Nsight timeline screenshot** for one model

Each plot: matplotlib, saved as PNG at 200 DPI, consistent color scheme across plots. Use a muted palette — seaborn's `muted` or `deep` looks professional without being flashy.

### Phase 11: Writeup

Structure for `writeup/findings.md`:

1. **What I did** (1 paragraph): four models, three experiments, ~40 profiling runs total.
2. **Hardware and software** (1 paragraph): 5070, CUDA 12.8, cuDNN 9.x, PyTorch 2.x.
3. **Per-model findings** (4 paragraphs, one per model):
    - Dominant kernels
    - FP16 speedup observed
    - Compute- or memory-bound
    - One interesting surprise
4. **Cross-model observations** (2 paragraphs):
    - How cuDNN algorithm selection differs by architecture
    - When Tensor Cores help vs don't
5. **Things I didn't do and would** (1 paragraph): honest list — e.g., "didn't do Nsight Compute kernel-level analysis, didn't try CUDA graphs, didn't compare to TensorRT."
6. **Plots**: all six figures with captions.

Aim for 8–12 pages. Long enough to show the per-model kernel walkthroughs, the cross-model tables, and the roofline discussion in detail — short enough that it's still a profiling study, not a thesis.

### Phase 12: Buffer / slack

Something will have taken longer than planned. Use this phase for whatever fell behind — probably the writeup or one experiment that needed re-running. If nothing's behind, add a small bonus experiment like channels-last memory format for vision models (one script, 30 min, often shows a surprising speedup).

---

## 7. Detailed profiling methodology notes

This section collects the gotchas that trip people up. Read before running any profile.

### 7.1 Warmup matters, a lot

First invocations of a model allocate memory, JIT-compile kernels, run cuDNN algorithm search, and hit cold caches. Always discard the first 5–10 iterations. For anything measuring < 10ms, warmup should be 20+ iterations.

When `cudnn.benchmark=True`, the *first* iteration at each input shape takes dramatically longer — sometimes 10× — because cuDNN is timing multiple algorithms. Don't panic. This is expected.

### 7.2 `torch.cuda.synchronize()` is not optional

CUDA is asynchronous. Kernel launches return immediately; the actual work happens later on the GPU. If you time around a kernel launch without syncing, you're timing the launch overhead, not the work. Always `torch.cuda.synchronize()` before and after your timing region.

Better: use `torch.cuda.Event` with `enable_timing=True`. Events are timestamped on the GPU itself, no sync needed:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... work
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

### 7.3 `torch.no_grad()` and `model.eval()`

For inference profiling, set the model to eval mode and wrap in `torch.no_grad()`. Otherwise you're also recording the autograd graph which skews results and balloons memory.

```python
model.eval()
with torch.no_grad():
    output = model(x)
```

Batch norm behaves differently in eval (uses running stats instead of batch stats). Dropout is disabled. Both are what you want for inference profiling.

### 7.4 Exclude the first iteration from profiler output

The PyTorch Profiler has a `schedule` argument that lets you skip warmup:

```python
from torch.profiler import schedule
sched = schedule(wait=1, warmup=2, active=5, repeat=1)
with profile(activities=[...], schedule=sched, on_trace_ready=...) as prof:
    for i in range(10):
        with torch.no_grad():
            model(x)
        prof.step()
```

Skips 1 wait + 2 warmup iters, then records 5 "active" iters. This is cleaner than manually filtering.

### 7.5 Input shape matters for kernel selection

cuDNN picks different algorithms for different (batch, channels, height, width) tuples. If you profile at batch 32 and report "ResNet-18 uses Winograd," that's only true at batch 32. At batch 1 it might pick implicit-GEMM. Always document the shape at which you measured.

Also: use shapes that are multiples of 8 (or 16) when possible. Tensor Cores want aligned shapes. Batch 31 will often be meaningfully slower than batch 32 because of alignment issues.

### 7.6 Don't trust `%Self` vs `%Total` naively

In the profiler table, "Self CUDA %" is time spent in that op's own kernel, excluding child ops. "Total CUDA %" includes children. For conv2d, these are basically the same because the work is in the cuDNN kernel. For higher-level ops (like a whole Transformer encoder block), `Self` is nearly zero and `Total` is what you want.

### 7.7 Memory transfers sneak up on you

`x.cuda()` and `x.cpu()` are memory transfers over PCIe. They're *slow* (~12 GB/s on PCIe 4.0 x16). If you're profiling and your input is being moved from CPU to GPU every iteration, that transfer will dominate. Allocate your input once on GPU and reuse:

```python
x = torch.randn(32, 3, 224, 224, device='cuda')  # allocated ONCE
for _ in range(100):
    with torch.no_grad():
        out = model(x)
```

Don't do `x.cuda()` in a loop.

### 7.8 `channels_last` memory format

Vision models default to NCHW (`channels_first`) layout in PyTorch. cuDNN often prefers NHWC (`channels_last`) for Tensor Core paths. You can convert:

```python
model = model.to(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)
```

Speedup is workload-dependent. For ResNet-18 FP16 on recent GPUs, often 10–30% faster. For MobileNetV3-Small, usually small because depthwise convs don't benefit as much. If you have time in Phase 10 or 12, run this experiment — it's a cheap ~20% improvement that's easy to explain in the writeup.

### 7.9 `torch.compile` is out of scope

PyTorch 2.x has `torch.compile(model)` which fuses kernels and can dramatically change the profile. **Don't use it for this project.** Your goal is to understand cuDNN's own behavior, not to benchmark compiler-optimized graphs. If you compile, you'll see fused kernels with names like `triton_poi_fused_...` which are *not* cuDNN calls. Keep `torch.compile` out.

### 7.10 Determinism

If you want bit-exact reproducible results between runs, set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. This forces cuDNN to use a deterministic algorithm, which is sometimes slower. For a profiling study, you don't *need* bit-exact reproducibility, just consistent timing, but it's worth knowing the knob exists.

---

## 8. Experiments in full detail

### 8.1 Experiment 1 — Baseline

**Goal:** Characterize each model's kernel breakdown at a standard setting.

**Settings:** batch 32, FP32, `cudnn.benchmark=True`, NCHW, standard input sizes (224 for vision, seq 128 for DistilBERT, seq 100 for GRU).

**Procedure:** warmup 20 iters, profile 20 iters, save chrome trace, export key_averages table.

**Outputs:**
- 4 chrome traces
- 4 key_averages tables (top 30 ops by cuda_time_total)
- 1 summary CSV cross-model (from `classify_kernels.py`)
- 1 stacked bar plot of per-model time breakdown

**What to report:** top 5 kernels per model, percent of time in conv/matmul/norm/other.

### 8.2 Experiment 2 — `cudnn.benchmark` toggle

**Goal:** Quantify the speedup from cuDNN's algorithm search.

**Settings:** batch 32, FP32, NCHW. Vary `benchmark=False` vs `True`.

**Procedure:** For each of 4 models × 2 settings = 8 runs. Warmup carefully — benchmark=True needs more warmup (30+ iters) because the first iteration at each shape triggers algorithm search.

**Outputs:** 8-row table: model × benchmark_setting → mean ms/iter and std dev.

**What to report:** speedup percentage per model, which one benefits most (ResNet), which doesn't (DistilBERT because cuDNN isn't central).

### 8.3 Experiment 3 — FP32 vs FP16 (autocast)

**Goal:** Measure Tensor Core speedup.

**Settings:** batch 32, `cudnn.benchmark=True`, NCHW. Vary `dtype=FP32` vs autocast FP16.

**Procedure:** 4 models × 2 dtypes = 8 runs + one profiler run per model under FP16 to see kernel name changes.

**Outputs:** 8-row table + 4 kernel-diff tables (which kernels appeared in FP16 that weren't in FP32).

**What to report:** speedup per model, specific Tensor Core kernels observed (look for `hmma`, `mma` in names), why MobileNetV3 gets less benefit than ResNet.

### 8.4 Experiment 4 — Batch size scaling

**Goal:** Show the transition from launch-overhead-dominated to steady-state.

**Settings:** batch ∈ {1, 4, 16, 64, 256}, FP32 and FP16, `cudnn.benchmark=True`.

**Procedure:** 4 models × 5 batch sizes × 2 dtypes = 40 runs. Each run: warmup 10, time 50.

**Outputs:** throughput curves (samples/sec vs batch size), one plot per dtype.

**What to report:** at what batch each model saturates, how FP16 changes the saturation point, OOM behavior at large batches.

### 8.5 Experiment 5 (optional, Phase 12) — channels_last

**Goal:** Show the impact of memory layout.

**Settings:** batch 32, FP16, NCHW vs NHWC. Vision models only (ResNet-18 and MobileNetV3-Small).

**Procedure:** 2 models × 2 layouts = 4 runs.

**Outputs:** 4-row table.

**What to report:** speedup from NHWC, especially for the FP16 Tensor Core path.

### 8.6 Experiment 6 (optional, Phase 12) — Sequence length sweep

**Goal:** Show the compute-bound → memory-bound transition in transformer workloads.

**Settings:** DistilBERT only, batch 8, seq ∈ {32, 64, 128, 256, 512}, FP16.

**Procedure:** 5 seq lengths × 1 model = 5 runs, plus one full profile at seq=32 vs seq=512 for kernel comparison.

**Outputs:** latency vs seq plot, attention kernel time as % of total vs seq.

**What to report:** at short seqs, attention is a small fraction; at long seqs, it dominates (because attention is O(seq²)). This visualizes a well-known fact cleanly.

---

## 9. Expected output artifacts

By end of project, your `results/` directory should contain:

**Traces (chrome trace JSON, 50–500 MB total):**
- `resnet18_baseline.json`
- `mobilenetv3_baseline.json`
- `distilbert_baseline.json`
- `gru_baseline.json`
- `resnet18_fp16.json`
- (Others as produced by experiments)

**Nsight Systems reports (.nsys-rep, 100–500 MB each):**
- `resnet18_baseline.nsys-rep`
- `distilbert_baseline.nsys-rep`
- (Two is plenty.)

**Tables (CSV):**
- `baseline_breakdown.csv` — per-model kernel category %
- `benchmark_toggle.csv` — cudnn.benchmark on vs off timings
- `amp_comparison.csv` — FP32 vs FP16 timings
- `batch_sweep.csv` — throughput vs batch size
- `top_kernels_per_model.csv` — top 10 kernels per model × dtype

**Plots (PNG, 200 DPI):**
- `fig1_time_breakdown.png` — stacked bar, 4 models, 5 categories
- `fig2_fp16_speedup.png` — grouped bar, 4 models, speedup factor
- `fig3_batch_scaling.png` — line chart, throughput vs batch, 4 models × 2 dtypes
- `fig4_algorithm_distribution.png` — stacked bar, which cuDNN algorithms
- `fig5_roofline.png` — scatter on log-log roofline
- `fig6_nsight_timeline.png` — screenshot from Nsight GUI

**Writeup:**
- `findings.md` — the prose document, 8–12 pages
- Embedded plots

---

## 10. Writeup template

Here's the structure to follow for `writeup/findings.md`. Aim for 8–12 pages.

```markdown
# Profiling cuDNN across Four Deep Learning Models

## Summary

We profiled four models — ResNet-18, MobileNetV3-Small, DistilBERT-base, and
a tiny GRU — on an RTX 5070 (Blackwell) using PyTorch Profiler and
Nsight Systems. The models span the compute-bound/memory-bound spectrum on
both conv and matmul axes. Key findings: [3 sentences summarizing the main
story].

## Hardware and software

RTX 5070 (Blackwell, sm_120), CUDA 12.8, cuDNN 9.x, PyTorch 2.x,
Windows 11. Profiling with torch.profiler and Nsight Systems 2025.x.
All measurements are inference-only, batch 32 unless stated, with warmup.

## Methodology

[~150 words. Describe the 6-part plan: baseline profile, benchmark toggle,
FP32 vs FP16, batch sweep, roofline, and Nsight inspection. Link to the
scripts.]

## Per-model findings

### ResNet-18

[~200 words. Time breakdown. Dominant kernels (Winograd, implicit_gemm).
FP16 speedup observed (~2.5×). Classification: compute-bound.
One interesting observation — e.g., "even though benchmark mode added
20% speedup on first run, the steady-state difference was only 8% because
cuDNN 9 heuristics have improved."]

### MobileNetV3-Small

[~200 words. Depthwise dominance. Lower FP16 speedup. Kernels named
differently from ResNet (dgrad_engine vs winograd). Memory-bound.]

### DistilBERT-base

[~200 words. cuBLAS dominance, not cuDNN. The cuDNN footprint is almost
entirely layer norm. Attention becomes larger fraction at longer seqs.
FP16 gives strong speedup.]

### Tiny GRU

[~200 words. cuDNN RNN fused kernel. The whole forward pass is essentially
one kernel on cuDNN side. Memory-bound as expected. FP16 helps modestly.]

## Cross-model observations

[~300 words. Two or three interesting patterns you saw across models:
- Tensor Core utilization varies dramatically by architecture
- cuDNN algorithm selection: when it picks Winograd vs implicit_gemm
- Memory bandwidth is the ceiling for half of these models]

## Roofline

[1 paragraph + figure. Where each model sits on the roofline. Explain
why ResNet-18 is near the compute ceiling, why MobileNet is on the ramp.]

## What I didn't do

[~100 words. Honest about scope. "I didn't evaluate CUDA graphs, didn't
try TensorRT, didn't compare across cuDNN versions, didn't do kernel-level
ncu analysis." This section is an integrity marker — graders notice.]

## Figures

[Embed 5–6 PNGs with captions.]
```

---

## 11. Troubleshooting

### `AttributeError: module 'profile' has no attribute 'run'`

Your project has a top-level directory named `profile/` with an `__init__.py`, which shadows Python's stdlib `profile` module. Anything that transitively imports `cProfile` (torchvision's `ops` package does, via `torch._dynamo`) will now resolve `profile` to your empty package and crash on attribute access.

**Fix:** rename the directory to `profiling/` (or anything that isn't a stdlib module name). Observed on PyTorch 2.10 + torchvision 0.25; older PyTorch versions may not trigger the import chain eagerly and therefore don't reproduce this.

### `ModuleNotFoundError: No module named 'models'` when running a profiler script

You invoked the script as `python profiling/run_baseline.py`. That puts `profiling/` on `sys.path[0]`, not the repo root, so `from models.resnet import ...` inside the script can't find the sibling package.

**Fix:** invoke as a module from the repo root: `python -m profiling.run_baseline --model resnet18`. `-m` puts the current working directory on `sys.path[0]`, so `models/`, `profiling/`, and `analysis/` are all importable.

### "No kernel image available for execution on the device"

You installed the wrong PyTorch wheel. Reinstall with `--index-url https://download.pytorch.org/whl/cu128`. Verify with `torch.cuda.get_device_capability(0)` — it must return `(12, 0)`.

### Expected Winograd kernels in the ResNet-18 profile, saw none

On Blackwell (`sm_120`) + cuDNN 9.10.2, and with PyTorch's default `torch.backends.cuda.matmul.allow_tf32 = True`, the benchmark search picks TF32 Tensor-Core implicit-GEMM (`cutlass_tensorop_s1688fprop_optimized_tf32_...`, `sm80_xmma_fprop_implicit_gemm_tf32f32_...`) over Winograd. Winograd's ~2.25× multiplication reduction is smaller than the TF32-on-TC throughput gap; cuDNN correctly measures TC-GEMM as faster.

**If you want Winograd to appear:** disable TF32 before running the profile.

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

This is not a bug; it's the intended behaviour of `cudnn.benchmark=True` on newer hardware. Treat it as a finding, not a failure.

### `torch.cuda.is_available()` returns False

Driver issue. Check `nvidia-smi` works. If it does but torch can't see the GPU, the driver–PyTorch combination is wrong. Update driver. If `nvidia-smi` doesn't work at all, reinstall the driver from NVIDIA.

### Nsight Systems crashes on capture

Usually Windows permissions. Run as administrator. If still crashing, try the command-line `nsys profile` instead of the GUI-driven capture.

### Chrome trace file too large to open

Happens with long profile windows. Limit the `active` iterations in the schedule to 5. Or use Perfetto UI (https://ui.perfetto.dev/) which handles larger files than Chrome's native viewer.

### Wildly different timing between runs

Check that `torch.cuda.synchronize()` is around your timing region. If you're using CUDA Events, make sure you synchronize *after* the end event before calling `elapsed_time`. Also check GPU isn't being used by another process — Windows tools like MSI Afterburner can steal cycles.

### `cudnn.benchmark=True` makes things slower

Input shapes are changing between iterations. Benchmark mode re-runs algorithm search every time a new shape is seen. Either fix your inputs to a single shape, or set benchmark=False.

### Out-of-memory at batch 256

Expected for DistilBERT at seq 512, or ResNet-18 at very large batches. Cap the batch sweep at whatever fits in 12 GB. Don't try to debug memory errors — just report the cap in the writeup.

### First iteration of profiler is way slower than subsequent ones

That's the `cudnn.benchmark=True` behavior plus CUDA init overhead. Use the profiler schedule (wait=1, warmup=2, active=5) to skip these.

### The `cudnn::` kernels aren't showing up in the profiler

You might be in a code path that uses a different backend. Check `torch.backends.cudnn.enabled = True`. Also check you're actually running CUDA tensors (`.cuda()` or `.to('cuda')`). Some PyTorch ops route to custom kernels instead of cuDNN — this is especially true for newer ops added post-PyTorch 2.0.

---

## 12. Appendix A — Kernel name decoder

cuDNN kernel names follow a scheme. Rough decoder:

**Prefix patterns:**
- `sm80_` / `sm86_` / `sm90_` / `sm120_` — target compute capability. sm120 = Blackwell.
- `xmma_` — extended MMA kernel family (Tensor Cores)
- `hmma_` — half-precision MMA (FP16)
- `bmma_` — BF16 MMA
- `imma_` — INT8 MMA
- `cudnn_` — non-MMA cuDNN kernels

**Operation patterns:**
- `gemm_` — general matrix multiply
- `conv_` — convolution
- `winograd_` — Winograd algorithm
- `implicit_gemm_` / `implicit_precomp_` — im2col + GEMM without materialized buffer
- `dgrad_` — data-gradient (for backprop, shouldn't appear in inference)
- `wgrad_` — weight-gradient (for backprop)
- `batch_norm_` — batch normalization
- `softmax_` — softmax (may or may not appear, depending on cuDNN vs aten path)
- `rnn_` / `lstm_` / `gru_` — recurrent kernels
- `pooling_` — pooling

**Dtype patterns:**
- `f32f32_f32f32_f32` — FP32 in, FP32 accumulate, FP32 out
- `f16f16_f16f16_f16` — FP16 everywhere
- `f16f16_f16f32_f16` — FP16 in/out, FP32 accumulate (common)
- `f16f16_f16f16_f32` — FP16 in, FP32 out (mixed)
- `tf32_` — TF32 math mode (FP32 storage, FP16-ish math on Tensor Cores)

**Tile pattern example:**
`tilesize128x256x64` — the tile dimensions for this kernel's mmul. Larger tiles = better utilization for big matrices, worse for small ones.

**Full example, broken down:**
```
sm80_xmma_gemm_f16f16_f16f32_f16_nn_n_tilesize128x256x64_stage3_warpsize2x4x1_tensor16x8x16_kernel
```
- `sm80` — Ampere+ target (will run on sm_120 too)
- `xmma` — Tensor Core
- `gemm` — it's a matmul
- `f16f16_f16f32_f16` — FP16 in, FP32 accumulate, FP16 out
- `nn_n` — non-transposed inputs, non-transposed output
- `tilesize128x256x64` — 128×256 output tile, K=64
- `stage3` — 3-stage pipelined
- `warpsize2x4x1` — 2×4 warp tiling
- `tensor16x8x16` — 16×8×16 Tensor Core instructions

Don't memorize this. Just know how to decode kernel names you see, so you can write sensible sentences like "ResNet-18 FP16 inference spent 62% of GPU time in sm80 xmma gemm kernels, indicating most work went through Tensor Cores."

---

## 13. Appendix B — cuDNN algorithm reference

Convolution algorithms in cuDNN (forward path, what you'll see in inference):

**IMPLICIT_GEMM** (`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`): Conv reformulated as GEMM with inline index computation. No explicit im2col buffer. Works for all shapes. Usually fastest for 1×1 convs and for odd shapes.

**IMPLICIT_PRECOMP_GEMM** (`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`): Like implicit GEMM but precomputes some indices in workspace. Slightly more memory, slightly faster. Often the default pick.

**GEMM** (`CUDNN_CONVOLUTION_FWD_ALGO_GEMM`): Explicit im2col + GEMM. Materializes the im2col buffer in workspace. Large workspace requirement. Rarely picked now; legacy.

**DIRECT** (`CUDNN_CONVOLUTION_FWD_ALGO_DIRECT`): Nested-loop direct conv. Not actually implemented for most shapes in modern cuDNN.

**FFT** (`CUDNN_CONVOLUTION_FWD_ALGO_FFT`): FFT-based. Only wins for filters ≥ 7×7 which are rare in modern networks. You might never see this picked.

**FFT_TILING** (`CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING`): Tiled FFT for larger inputs. Same rarity.

**WINOGRAD** (`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD`): Winograd's minimal filtering for 3×3 kernels. ~2.25× fewer multiplications at the cost of more additions and higher numerical error. Limited to specific filter sizes and input sizes. Usually a 1.3–1.7× speedup over GEMM when applicable. For ResNet-18 (all 3×3 convs after the first), this is the go-to pick in benchmark mode.

**WINOGRAD_NONFUSED** (`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED`): Winograd implementation where the transform steps are separate kernels (more memory, sometimes faster for larger batch/channel counts).

For a given conv layer, cuDNN's heuristic or benchmark mode picks one of these based on shapes, memory budget, and measured timings (if benchmarking). You'll typically see a mix of IMPLICIT_PRECOMP_GEMM and WINOGRAD in a network's profile, with occasional GEMM.

---

## 14. Appendix C — RTX 5070 Blackwell specs (quick reference)

- Architecture: Blackwell (GB203)
- Compute capability: sm_120
- CUDA cores: ~10,752 (varies slightly by SKU)
- 5th-gen Tensor Cores (support FP16, BF16, FP8, FP4)
- Memory: 12 GB GDDR7
- Memory bandwidth: ~800 GB/s (approximate; depends on exact memory spec)
- Peak FP32: ~60 TFLOP/s
- Peak FP16 (Tensor Core): ~380 TFLOP/s
- Peak FP8 (Tensor Core): ~760 TFLOP/s

These numbers give you the ceilings for the roofline plot. For a conservative roofline:
- Memory BW ceiling: 800 GB/s
- FP32 compute ceiling: 60 TFLOP/s → ridge point at 75 FLOP/byte
- FP16 Tensor Core ceiling: 380 TFLOP/s → ridge point at 475 FLOP/byte

A workload with intensity < 75 FLOP/byte is memory-bound in FP32. At FP16 the ridge shifts higher because compute ceiling rises faster than memory bandwidth.

---

## 15. Appendix D — commands cheat sheet

```powershell
# Environment check
python env/check_env.py

# Run baseline on all models
python -m profiling.run_baseline --model resnet18
python -m profiling.run_baseline --model mobilenetv3
python -m profiling.run_baseline --model distilbert
python -m profiling.run_baseline --model gru

# Or loop
for m in resnet18 mobilenetv3 distilbert gru; do python -m profiling.run_baseline --model $m; done

# cudnn.benchmark toggle experiment
python -m profiling.run_benchmark_toggle --model resnet18

# FP32 vs FP16
python -m profiling.run_amp --model resnet18

# Batch sweep
python -m profiling.run_batch_sweep --model resnet18 --batches 1,4,16,64,256

# Nsight capture
nsys profile -t cuda,cudnn,cublas,nvtx -o results/nsys/resnet18 python -m profiling.run_baseline --model resnet18

# Analysis
python analysis/parse_trace.py results/traces/resnet18_baseline.json
python analysis/classify_kernels.py
python analysis/plots.py

# Full regen of everything
python scripts/run_all.py
```

---

## 16. What a good final deliverable looks like

If a TA or a PhD student glances at your repo, what should they see?

**In two minutes, they should:**
- Open `writeup/findings.md` and understand the methodology from the first paragraph
- See the four figures and grasp the story without reading captions carefully
- See a table with numbers per model

**In ten minutes, they should:**
- Read the per-model sections and see that you know what each kernel is
- Spot-check one of your CSV tables and find consistent data
- See that you've actually run the scripts (git log shows multiple commits, results are dated recently)

**In an hour, they should:**
- Be able to clone, `pip install -r requirements.txt`, and run `python -m profiling.run_baseline` successfully on their own machine
- Verify your numbers within 10% on their hardware
- Trust that you understand what you did

That last bullet is the bar. If your writeup has a sentence like "ResNet-18 under FP16 routes most work through xmma kernels, which are the Tensor Core matmul family — you can see this in the kernel name prefix" — that tells the reader you read the profile, you understood it, and you can explain it. That sentence is worth more than ten pages of tool documentation.

---

## 17. Stretch goals if you finish early

Ranked by effort:

1. **channels_last on vision models** (30 min, high value): 10–30% speedup, easy to explain.
2. **Sequence length sweep on DistilBERT** (1 hour): Show O(seq²) attention cost visually.
3. **Compare cuDNN 8.x vs 9.x** (2+ hours, risky): requires installing an older PyTorch wheel in a separate env. Interesting but environmental pain.
4. **Nsight Compute kernel deep-dive** (2+ hours): pick one kernel, run `ncu --set full` on it, look at memory throughput, occupancy, FP utilization. This is the next level of profiling depth. Good bonus but not essential.
5. **CUDA graphs** (2+ hours): wrap one model in `torch.cuda.CUDAGraph` and measure. Can give 10–20% speedup by eliminating launch overhead. Interesting but it's not cuDNN-specific.
6. **Forward+backward instead of forward only** (2 hours): re-run experiments with autograd enabled. You'll see dgrad/wgrad kernels which are pure cuDNN territory. Doubles the data and the writeup.

Don't do more than one stretch goal. The baseline project is already plenty.

---

## 18. Non-goals — things explicitly not to do

- Don't train any model from scratch. Pretrained weights everywhere.
- Don't fine-tune anything.
- Don't compare against other frameworks (JAX, TF, MXNet). One framework is enough.
- Don't test on multiple GPUs. One 5070.
- Don't try to beat the default performance. You're characterizing it, not optimizing it.
- Don't go deep on one model at the expense of coverage. Four models with moderate depth beats one model with exhaustive depth for this project's pedagogical goals.
- Don't write a custom CUDA kernel. If you think "I could write this faster in raw CUDA," that's a different project.
- Don't use `torch.compile`. Confounds the analysis.
- Don't benchmark training. Inference only.
- Don't use dynamic shapes or dynamic control flow in your models. Keep it clean.

---

## 19. Time ledger — rough effort estimate per phase

Phases are work units, not fixed time slots — any one of them may run longer or shorter than the rough estimate below depending on how the measurements go. The numbers are indicative only.

| Phase | Rough effort (h) | Cumulative (h) |
|---|---|---|
| Phase 0 — pre-reading | 0.75 | 0.75 |
| Phase 1 — environment setup | 1.0 | 1.75 |
| Phase 2 — first profile on ResNet-18 | 1.0 | 2.75 |
| Phase 3 — port to all four models | 1.0 | 3.75 |
| Phase 4 — kernel classification + summary | 1.0 | 4.75 |
| Phase 5 — Nsight Systems timeline | 1.0 | 5.75 |
| Phase 6 — cudnn.benchmark toggle | 1.0 | 6.75 |
| Phase 7 — FP32 vs FP16 (AMP) | 1.0 | 7.75 |
| Phase 8 — batch size sweep | 1.0 | 8.75 |
| Phase 9 — roofline analysis | 1.0 | 9.75 |
| Phase 10 — cleanup and plots | 1.0 | 10.75 |
| Phase 11 — writeup | 1.0 | 11.75 |
| Phase 12 — buffer | 1.0 | 12.75 |

Rough total: ~12–13 hours of focused work if nothing misbehaves. Driver or wheel issues on a new card typically add a phase of their own.

---

## 20. Statistical rigor — how many trials, how to report numbers

Timing a GPU kernel is noisier than people realize. GPU clocks boost and throttle, memory allocators vary, DVFS kicks in. A single measurement is almost meaningless. Here's how to get trustworthy numbers without over-engineering.

**Trials per measurement.** For any number you put in the writeup, measure at least 5 times and report mean ± std. For timing numbers specifically, a coefficient of variation (std/mean) under 3% is clean, under 5% is acceptable, over 10% means something is wrong (usually thermal throttling or a background process stealing the GPU). Use `torch.cuda.Event` with `enable_timing=True` rather than `time.perf_counter()` — CPU-side timers include Python overhead and OS jitter that GPU events don't have.

**Warmup depth depends on mode.** With `cudnn.benchmark=False`, 10 warmup iterations is plenty — cuDNN uses its heuristic, no algorithm search happens. With `cudnn.benchmark=True`, the first iteration at each new input shape triggers algorithm benchmarking, which can take 50× longer than steady state. Use 30+ warmup iterations in benchmark mode. If you're doing a batch sweep, you need to warm up *at each batch size* because each one triggers a new algorithm search.

**Outlier rejection.** Cold-start outliers (first iteration much slower than rest) are expected — always skip them. Random spikes mid-run usually mean a background process woke up — re-run. Don't silently filter outliers in post-processing; if you re-run, re-run the whole timing loop, not just the bad iteration.

**What to report.** In your tables, include both the mean and either std or min/max. "ResNet-18 FP16 at batch 32: 4.12 ms ± 0.08 ms (n=20)" tells the reader more than "4.12 ms" alone. Avoid false precision — 4.12 ms is fine, 4.1234 ms is lying. Round to match your actual measurement noise.

**Clocks and thermal state.** The 5070 boosts its clock based on temperature and power. If you run a long batch sweep without pauses, the card thermally throttles somewhere in the middle and your last measurements are artificially slow. Solution: insert a 5-second sleep between experiments, or run the batch sweep in an order that alternates between light and heavy configurations so throttling doesn't bias any single setting. For a study this short, just monitoring `nvidia-smi dmon -s u` in a side terminal is enough — if you see GPU temperature climb above 75°C during runs, pause and let it cool.

**Reporting negative results.** If an experiment doesn't show the expected speedup, *say so* and explain. "MobileNetV3-Small showed only 1.1× speedup under FP16 autocast instead of the expected 1.3–1.5×, likely because the Tensor Core-eligible pointwise convs are too small to amortize setup cost at batch 32." That sentence is worth more than a pretty plot of a result that matches expectations trivially.

---

## 21. Starter script templates

The scripts described in section 6 and 8 are short but the details matter. Here are complete, copy-pasteable versions of the three most important ones.

### 21.1 `env/check_env.py`

```python
"""Verify hardware, PyTorch, cuDNN are set up correctly for Blackwell."""
import sys
import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA (PyTorch-linked): {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Check driver and wheel.")
    sys.exit(1)

print(f"Device: {torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability(0)
print(f"Compute capability: sm_{cap[0]}{cap[1]}")

if cap != (12, 0):
    print(f"WARNING: Expected sm_120 (Blackwell). Got sm_{cap[0]}{cap[1]}.")

# Smoke test — actually run a kernel
x = torch.randn(1024, 1024, device='cuda')
y = torch.matmul(x, x.T)
torch.cuda.synchronize()
print(f"Matmul smoke test: output shape {y.shape}, max = {y.max().item():.2f}")

# Smoke test — actually exercise cuDNN
import torch.nn.functional as F
a = torch.randn(16, 64, 56, 56, device='cuda')
w = torch.randn(128, 64, 3, 3, device='cuda')
b = F.conv2d(a, w, padding=1)
torch.cuda.synchronize()
print(f"cuDNN conv smoke test: output shape {b.shape}")

print("\nAll checks passed.")
```

### 21.2 `profiling/run_baseline.py`

```python
"""Baseline profile of a single model. Saves chrome trace + top-kernel table."""
import argparse
import os
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

def load_model_and_input(name, batch=32):
    if name == 'resnet18':
        import torchvision.models as tvm
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        x = torch.randn(batch, 3, 224, 224)
    elif name == 'mobilenetv3':
        import torchvision.models as tvm
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        x = torch.randn(batch, 3, 224, 224)
    elif name == 'distilbert':
        from transformers import DistilBertModel
        m = DistilBertModel.from_pretrained('distilbert-base-uncased')
        x = torch.randint(0, 30000, (batch, 128))
    elif name == 'gru':
        import torch.nn as nn
        class TinyGRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(64, 128, 2, batch_first=True)
                self.fc = nn.Linear(128, 10)
            def forward(self, x):
                o, _ = self.gru(x)
                return self.fc(o[:, -1])
        m = TinyGRU()
        x = torch.randn(batch, 100, 64)
    else:
        raise ValueError(name)
    return m.eval().cuda(), x.cuda()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--benchmark', action='store_true', default=True)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = args.benchmark
    model, x = load_model_and_input(args.model, args.batch)

    # Warmup
    for _ in range(30):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    os.makedirs('results/traces', exist_ok=True)
    trace_path = f'results/traces/{args.model}_baseline.json'

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        schedule=schedule(wait=1, warmup=2, active=10, repeat=1),
    ) as prof:
        for _ in range(15):
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    prof.export_chrome_trace(trace_path)
    print(f"\nTrace saved to {trace_path}")

if __name__ == '__main__':
    main()
```

### 21.3 `analysis/classify_kernels.py`

```python
"""Parse a chrome trace JSON, classify kernels, output CSV breakdown."""
import json
import sys
import pandas as pd
from collections import defaultdict

def classify(name):
    n = name.lower()
    if 'winograd' in n: return 'conv_winograd'
    if 'implicit_gemm' in n or 'implicit_precomp' in n: return 'conv_implicit_gemm'
    if 'dgrad' in n or 'wgrad' in n: return 'conv_backward'
    if 'conv' in n or 'cudnn_conv' in n: return 'conv_other'
    if 'hmma' in n or 'imma' in n or 'bmma' in n: return 'matmul_tensor_core'
    if 'gemm' in n or 'cublas' in n: return 'matmul_fp32'
    if 'rnn' in n or 'lstm' in n or 'gru' in n: return 'rnn'
    if 'batch_norm' in n or 'layer_norm' in n: return 'norm'
    if 'softmax' in n: return 'softmax'
    if 'elementwise' in n or 'vectorized_elementwise' in n: return 'elementwise'
    if 'reduce' in n or 'sum_kernel' in n: return 'reduce'
    if 'memcpy' in n: return 'memcpy'
    return 'other'

def main(trace_path):
    with open(trace_path) as f:
        data = json.load(f)

    events = data.get('traceEvents', data) if isinstance(data, dict) else data
    kernels = [e for e in events
               if e.get('ph') == 'X'
               and e.get('cat', '').lower() in ('kernel', 'cuda', 'gpu_op', 'cuda_runtime')
               and 'dur' in e]

    buckets = defaultdict(lambda: {'time_us': 0, 'count': 0})
    for k in kernels:
        cat = classify(k.get('name', ''))
        buckets[cat]['time_us'] += k['dur']
        buckets[cat]['count'] += 1

    rows = [{'category': c, **v} for c, v in buckets.items()]
    df = pd.DataFrame(rows).sort_values('time_us', ascending=False)
    total_us = df['time_us'].sum()
    df['percent'] = df['time_us'] / total_us * 100
    print(df.to_string(index=False))
    return df

if __name__ == '__main__':
    main(sys.argv[1])
```

### 21.4 Minimal AMP comparison driver

```python
"""FP32 vs FP16 timing. Outputs a single row of CSV per run."""
import argparse, csv, os, torch
from profile.run_baseline import load_model_and_input

def time_model(model, x, use_amp, iters=100):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--batch', type=int, default=32)
args = ap.parse_args()

torch.backends.cudnn.benchmark = True
model, x = load_model_and_input(args.model, args.batch)

# Warmup in both modes
for _ in range(30):
    with torch.no_grad():
        _ = model(x)
for _ in range(30):
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        _ = model(x)
torch.cuda.synchronize()

fp32_ms = time_model(model, x, use_amp=False)
fp16_ms = time_model(model, x, use_amp=True)
speedup = fp32_ms / fp16_ms

print(f"{args.model},batch={args.batch},fp32={fp32_ms:.3f},fp16={fp16_ms:.3f},speedup={speedup:.2f}")

os.makedirs('results/tables', exist_ok=True)
with open('results/tables/amp_comparison.csv', 'a', newline='') as f:
    csv.writer(f).writerow([args.model, args.batch, fp32_ms, fp16_ms, speedup])
```

These four scripts cover ~70% of what you need. Port the same pattern for `run_benchmark_toggle.py`, `run_batch_sweep.py`, etc.

---

## 22. Windows-specific performance gotchas

The 5070 on Windows 11 has a few environmental issues that can bias your measurements. Check these before trusting timing numbers.

**Power plan.** Windows defaults to "Balanced," which lets the CPU downclock while waiting on the GPU. That adds CPU-side latency between kernel launches. For benchmarking, set "High Performance" or "Ultimate Performance" mode. `powercfg /setactive SCHEME_MIN` sets high performance. Switch back afterward if you care about idle power draw.

**Xbox Game Bar and Game DVR.** Windows may detect your Python process launching CUDA kernels as "game activity" and auto-record it. This steals GPU cycles. Open Settings → Gaming → Xbox Game Bar → turn off. Also under Gaming → Captures → turn off background recording. This has been known to add 5–10% jitter to GPU workloads on gaming-class cards.

**Windows Defender real-time scanning.** If your profiler trace output is large and frequent, Defender may scan each write and slow the I/O path. Not usually a problem for timing of compute kernels but can hurt if you're writing huge trace files. Add your project directory to Defender exclusions (Settings → Update & Security → Windows Security → Virus & threat protection → Manage settings → Exclusions).

**NVIDIA driver quality mode.** In the NVIDIA Control Panel, there's a "Manage 3D Settings" → "Power management mode" setting. Set it to "Prefer maximum performance." The default "Optimal power" lets the card downclock aggressively between kernel launches, which is bad for short benchmarks.

**MSI Afterburner / GPU Tweak / Wallpaper Engine / any RGB controller.** These background apps query GPU state every few seconds and can cause tiny but reproducible jitter in your measurements. Kill them before running timing experiments.

**Thermal throttling.** The 5070 is a thirsty card. Under sustained load with stock cooling, it can hit 80°C within 2–3 minutes of nonstop kernels. Above ~83°C it starts lowering its clock. If your batch sweep runs for 10 minutes straight, the later measurements are running at a different clock than the earlier ones. Solution: insert `time.sleep(3)` between experiments, or run with the case open, or just monitor temperature and pause when it gets above 75°C.

**Windows Update.** Windows Update will happily download and install things in the background while you're benchmarking, which includes driver updates that can reset GPU state mid-run. Pause Windows Updates for the session (Settings → Update & Security → Windows Update → Pause updates for 1 week).

**WSL2 vs native Windows.** Some people prefer to run CUDA workloads under WSL2 on Windows because the Linux tooling is nicer. For this project, native Windows is fine — PyTorch + CUDA works well — and has lower overhead than WSL2 which adds a virtualization layer. If you're comfortable with WSL2, feel free, but don't switch mid-project.

---

## 23. FLOP counting for the roofline plot

Phase 9 asks you to build a roofline plot, which needs FLOPs per model per forward pass. Don't hand-count these — use a tool.

**Option 1: `fvcore.nn.FlopCountAnalysis` (recommended).** From Facebook Research, well-maintained.

```python
from fvcore.nn import FlopCountAnalysis
import torchvision.models as tvm

model = tvm.resnet18().eval().cuda()
x = torch.randn(1, 3, 224, 224, device='cuda')
flops = FlopCountAnalysis(model, x)
print(f"ResNet-18 FLOPs (batch 1): {flops.total() / 1e9:.2f} GFLOPs")
# Multiply by batch size for batch total
```

Install: `pip install fvcore`. Works cleanly on torchvision models. Limitation: may under-count attention in transformers — it doesn't always know to count Q·K^T and the attention-weighted-values matmul.

**Option 2: `ptflops`.** Similar, somewhat friendlier output.

```python
from ptflops import get_model_complexity_info
macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True)
print(f"MACs: {macs}, Params: {params}")
```

Note: ptflops reports MACs (multiply-accumulates), which is half of FLOPs by NVIDIA's counting convention. Multiply by 2 for FLOPs.

**Option 3: `torch.profiler` itself.** Pass `with_flops=True` to the profiler constructor and the key_averages table will include a FLOPs column. Less comprehensive than fvcore but free if you're already profiling.

**For the GRU**, FLOP counting tools often don't handle custom modules well. Just compute by hand: per timestep, each GRU layer does roughly 3 × (input_size + hidden_size) × hidden_size MACs for the gates. Multiply by num_layers × seq_length × batch. For the tiny GRU (input 64, hidden 128, 2 layers, seq 100, batch 32): 3 × (64+128) × 128 × 2 × 100 × 32 ≈ 470 MFLOPs. Small number — confirms the memory-bound diagnosis.

**For the roofline x-axis** (arithmetic intensity) you also need bytes moved. Rough estimate: bytes = (params × 4 for FP32 or 2 for FP16) + (activation_bytes). For the activations use `torch.cuda.max_memory_allocated()` before and after a forward pass. Ballpark is fine; you're plotting on a log scale.

---

## 24. Repository hygiene — what to commit, what to ignore

Your repo will have large binary artifacts that shouldn't go into git. Create a `.gitignore`:

```
# Chrome traces and Nsight reports (can be 100MB+)
results/traces/*.json
results/nsys/*.nsys-rep
results/nsys/*.qdrep
results/nsys/*.sqlite

# Python
__pycache__/
*.pyc
.venv/
venv/
.env

# Notebook checkpoints
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# IDE
.idea/
.vscode/

# Large model downloads (torchvision, HuggingFace)
~/.cache/
```

**Do commit:**
- All `.py` scripts in `env/`, `models/`, `profiling/`, `analysis/`
- `requirements.txt`
- `README.md` and the writeup markdown
- Small CSV tables (`results/tables/*.csv`)
- Final plots (`results/plots/*.png`) — these are your deliverables, commit them

**Don't commit:**
- Raw chrome traces (large, regeneratable, leak your filesystem paths)
- Nsight reports (even larger)
- Downloaded pretrained weights
- Virtual environments

If you use Git LFS, put the final plots there to keep the main repo small. But for a project this size, plain git with a good .gitignore is fine.

**Good README structure:**
```markdown
# cuDNN Profiling on Blackwell

Profiling four models (ResNet-18, MobileNetV3-Small, DistilBERT, tiny GRU)
on RTX 5070 using PyTorch Profiler and Nsight Systems.

## Setup
1. `conda env create -f environment.yml` (or `pip install -r requirements.txt`)
2. `python env/check_env.py` — should print sm_120 and pass smoke tests

## Reproduce
```
python scripts/run_all.py
python analysis/plots.py
```
See `writeup/findings.md` for results.

## Hardware
RTX 5070, CUDA 12.8, cuDNN 9.x, PyTorch 2.x.
```

---

## 25. Overnight batch runner

Once your scripts work, you don't need to babysit each experiment. Here's a `scripts/run_all.ps1` (PowerShell) or `scripts/run_all.sh` (bash) that runs the whole study end to end. Kick it off before dinner, come back to finished traces.

```powershell
# scripts/run_all.ps1
$ErrorActionPreference = "Stop"
$models = @("resnet18", "mobilenetv3", "distilbert", "gru")

Write-Host "=== Baseline profiles ==="
foreach ($m in $models) {
    python -m profiling.run_baseline --model $m
    Start-Sleep -Seconds 5  # let the GPU cool
}

Write-Host "=== cudnn.benchmark toggle ==="
foreach ($m in $models) {
    python -m profiling.run_benchmark_toggle --model $m
    Start-Sleep -Seconds 5
}

Write-Host "=== AMP comparison ==="
foreach ($m in $models) {
    python -m profiling.run_amp --model $m
    Start-Sleep -Seconds 5
}

Write-Host "=== Batch size sweep ==="
foreach ($m in $models) {
    foreach ($b in @(1, 4, 16, 64, 256)) {
        python -m profiling.run_batch_sweep --model $m --batch $b
        Start-Sleep -Seconds 3
    }
}

Write-Host "=== Analysis and plots ==="
python analysis/plots.py

Write-Host "Done. Results in results/, plots in results/plots/."
```

The `Start-Sleep` calls give the GPU a few seconds to cool down between runs — cheap insurance against thermal throttling bias. Total runtime for the whole script: roughly 45 minutes on the 5070. Start it at 10pm, come back to fully-baked results in the morning.

For the bash version, just replace `Start-Sleep -Seconds 5` with `sleep 5` and swap the loop syntax.

---

## 26. One final note on intellectual honesty

This project is small, focused, and should be done honestly. The writeup should report what you observed, including surprising results you don't fully understand. If you expect a 2× FP16 speedup on ResNet-18 and measure 1.5×, don't cherry-pick — report the 1.5× and note that it's lower than expected and might be due to memory bandwidth limits at batch 32, which you'd verify with a larger batch.

The best profiling writeups aren't the ones that confirm every expectation. They're the ones that spot an anomaly, investigate it for twenty minutes, and either explain it or honestly note "not sure why, follow-up would be X." That kind of writeup reads like a real engineer wrote it, which is the point.

Profiling is fundamentally about looking at numbers and being honest about what they say. Do that and you'll finish this project in 10–12 hours with a deliverable you'd be happy to show a PhD interviewer.

---

## End of plan

Starting point: open `env/check_env.py`, run it, confirm your hardware is visible. Then go to Phase 2.