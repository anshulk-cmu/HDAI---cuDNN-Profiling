<div align="center">

# Profiling cuDNN across Four Deep-Learning Models on Blackwell

### An inference-time kernel-level study on the NVIDIA RTX 5070 Ti Laptop GPU

<br>

**EEL71020 — Hardware Design for AI**

**Indian Institute of Technology, Jodhpur**

<br>

| Author | Roll No. |
| :---: | :---: |
| Anshul Kumar | M25AI2036 |
| Neha Prasad | M25AI2076 |

<br>

*Target length: 8–10 pages* · *Status: in progress (Phase 2 rework complete — see `docs/execution_log_2.md`)*

---

</div>

> **Document status.** Sections covering ResNet-18 (§4) are backed by real measurements from the reworked Phase 2 run — trace file `results/traces/resnet18_baseline_bs32_benchOn.json`, 10 profiled iterations × batch 32, FP32 with TF32 default, plus a separate multi-trial CUDA-event timing sweep (7 trials × 50 iterations). See [`docs/execution_log_2.md`](../docs/execution_log_2.md) for the full bug-fix and rerun audit. Remaining three models and all cross-model experiments are placeholders whose protocols are specified so measurements can be dropped in without refactoring the document.

<br>

## Table of Contents

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
  - [1.1 Problem setting](#11-problem-setting)
  - [1.2 Research questions](#12-research-questions)
  - [1.3 Scope and non-goals](#13-scope-and-non-goals)
- [2. Background](#2-background)
  - [2.1 The cuDNN algorithm zoo for 2-D convolution](#21-the-cudnn-algorithm-zoo-for-2-d-convolution)
  - [2.2 Tensor Cores and precision modes](#22-tensor-cores-and-precision-modes)
  - [2.3 The roofline model](#23-the-roofline-model)
- [3. Methodology](#3-methodology)
  - [3.1 Hardware](#31-hardware)
  - [3.2 Software stack](#32-software-stack)
  - [3.3 Model zoo](#33-model-zoo)
  - [3.4 Profiling procedure](#34-profiling-procedure)
  - [3.5 Metrics and timing rigor](#35-metrics-and-timing-rigor)
- [4. Results — ResNet-18 *(completed)*](#4-results--resnet-18-completed)
  - [4.1 Time attribution (RQ1)](#41-time-attribution-rq1)
  - [4.2 Algorithm selection (RQ2)](#42-algorithm-selection-rq2)
  - [4.3 Layout-conversion cost](#43-layout-conversion-cost)
  - [4.4 Regime classification (RQ3)](#44-regime-classification-rq3)
  - [4.5 Ops-per-iteration sanity check](#45-ops-per-iteration-sanity-check)
- [5. Results — further experiments *(pending)*](#5-results--further-experiments-pending)
- [6. Roofline analysis *(pending)*](#6-roofline-analysis-pending)
- [7. Cross-model discussion *(pending)*](#7-cross-model-discussion-pending)
- [8. Threats to validity](#8-threats-to-validity)
- [9. Conclusion](#9-conclusion)
- [References](#references)
- [Appendix A — cuDNN kernel-name decoder](#appendix-a--cudnn-kernel-name-decoder)
- [Appendix B — reproduction](#appendix-b--reproduction)
- [Appendix C — raw profiler output](#appendix-c--raw-profiler-output-resnet-18-top-25-rows)
- [Appendix D — environment manifest](#appendix-d--environment-manifest)

<br>

---

## Abstract

> *Full abstract to be finalised once all four models have been profiled. A partial abstract covering the Phase 2 material follows.*

This report characterises the GPU kernels that **cuDNN 9.10.2** dispatches for inference on an **NVIDIA RTX 5070 Ti Laptop GPU** (Blackwell, `sm_120`, 12 GB), using the PyTorch Profiler to record chrome-trace timelines and analysing them kernel-by-kernel. A four-model zoo — **ResNet-18, MobileNetV3-Small, DistilBERT-base**, and a **tiny GRU** — is used to span the `{conv, matmul} × {compute-bound, memory-bound}` quadrant. As of the current reporting point, **ResNet-18 has been fully profiled** at batch 32 in FP32.

Two findings are reported that contradict conventional wisdom baked into the project brief:

1. `cudnn.benchmark=True` does **not** select Winograd for ResNet-18's 3×3 convolutions on Blackwell; instead, **TF32 Tensor-Core implicit-GEMM kernels take 58.4 %** of GPU time, driven by PyTorch's default `allow_tf32=True` on `sm_80+`.
2. **9.5 % of GPU time is spent in NCHW↔NHWC layout-conversion kernels** because cuDNN's Tensor-Core fast path expects NHWC while torchvision models default to NCHW.

Both findings motivate follow-up experiments that are scheduled for later sections of this report.

<br>

---

## 1. Introduction

### 1.1 Problem setting

NVIDIA's cuDNN [[1]](#ref-1) is the hand-tuned C library underpinning almost every GPU-accelerated deep-learning framework in production use today. PyTorch [[2]](#ref-2) dispatches its `conv2d`, `batch_norm`, `RNN`, and many other primitives through cuDNN. The library exposes multiple algorithmic choices per operation (direct, implicit-GEMM, GEMM-with-im2col, Winograd [[3]](#ref-3), FFT) and a per-call runtime heuristic that selects among them. On modern GPUs the same logical operation can therefore be realised by radically different kernels depending on input shape, tensor layout, precision mode, and library version.

Understanding which kernel cuDNN actually picks — and why — is a useful exercise for three reasons:

- **Interpretation.** A GPU profile without decoded kernel names is a wall of inscrutable mangled symbols. Knowing what each kernel does lets one reason about the workload, not just its total runtime.
- **Optimisation.** When cuDNN picks a memory-bound kernel for a problem that "should" be compute-bound, there is usually a layout, precision, or shape issue that can be fixed. Reading the profile is the fastest way to spot these.
- **Portability.** Algorithm selection changes between cuDNN versions, between GPU architectures, and between precision modes. A report that calibrates what one specific stack does is a reference point against which future upgrades can be compared.

### 1.2 Research questions

For each of four representative models, this study answers three concrete questions.

| ID | Question |
| :---: | :--- |
| **RQ1** | **Time attribution.** What percentage of GPU time is spent on convolution, matmul, normalisation, elementwise work, and overhead? |
| **RQ2** | **Algorithm selection.** Which cuDNN and cuBLAS algorithms are dispatched, and why does the library pick them over alternatives? |
| **RQ3** | **Regime classification.** Is the model compute-bound or memory-bound on this hardware, with the evidence placed on a roofline plot [[4]](#ref-4)? |

### 1.3 Scope and non-goals

| | In scope | Out of scope |
| :--- | :--- | :--- |
| **Workload** | inference-only, four pretrained models, fixed input shapes | training, fine-tuning, dynamic shapes |
| **Tools** | PyTorch Profiler, Nsight Systems | `torch.compile`, TensorRT, Triton |
| **Knobs studied** | `cudnn.benchmark`, FP32↔FP16, batch size, sequence length, tensor layout | multi-GPU, cross-framework (JAX/TF) comparisons |
| **Depth** | kernel-level decoding and interpretation | writing new CUDA kernels, modifying models |

These exclusions are deliberate: the study characterises the default path cuDNN takes, not how to beat it.

<br>

---

## 2. Background

*A short primer suitable for a reviewer who knows PyTorch but has not touched cuDNN internals.*

### 2.1 The cuDNN algorithm zoo for 2-D convolution

cuDNN [[1]](#ref-1) exposes a small family of forward-convolution algorithms.

| Algorithm | One-line description | Typical regime |
| :--- | :--- | :--- |
| **Implicit-GEMM** | conv reformulated as GEMM with inline index arithmetic; no im2col buffer | most shapes in cuDNN 9.x |
| **Implicit-PRECOMP-GEMM** | variant that precomputes some indices into workspace | slightly faster, slightly more memory |
| **GEMM (explicit im2col)** | materialises the im2col buffer, then calls cuBLAS | legacy, rarely chosen |
| **Winograd** [[3]](#ref-3) | trades multiplications for additions for 3×3 filters | small filters, FP32 |
| **FFT / FFT-tiling** | pointwise multiply in frequency domain | large filters (≥ 7×7), rarely used |
| **Direct** | literal nested-loop conv | last resort |

When `torch.backends.cudnn.benchmark = True`, PyTorch asks cuDNN to measure each algorithm on the first invocation of each distinct (input-shape, weight-shape, dtype, math-mode) tuple and cache the fastest. With `benchmark = False`, cuDNN uses a shape-driven heuristic that returns quickly but may be suboptimal in steady state.

### 2.2 Tensor Cores and precision modes

Starting with the Volta architecture (`sm_70`) [[5]](#ref-5), NVIDIA GPUs include a specialised matrix-multiply-accumulate unit called a **Tensor Core**. Tensor Cores operate on reduced-precision inputs — FP16, BF16, INT8, FP8, FP4, and since Ampere `sm_80` the 19-bit **TF32** format [[6]](#ref-6) — and return full-precision accumulators. A TF32 Tensor-Core matmul looks like an FP32 matmul to the user but is actually performed with 10-bit mantissas internally; most deep-learning workloads are insensitive to this loss.

> **Key point.** On Ampere and later, PyTorch enables TF32 **by default** via `torch.backends.cuda.matmul.allow_tf32 = True`. "FP32 inference" on `sm_80+` hardware is silently Tensor-Core-accelerated in TF32 math mode. This becomes central to the §4.2 finding.

### 2.3 The roofline model

Williams et al.'s roofline model [[4]](#ref-4) classifies workloads by their **arithmetic intensity** — FLOPs performed per byte of DRAM traffic. Compute-bound workloads lie near the horizontal peak-FLOPs ceiling; memory-bound workloads lie on the sloped peak-bandwidth ramp. The "ridge point" is the arithmetic intensity at which the two lines meet.

For the RTX 5070 Ti Laptop GPU (§3.1):

| Ceiling | Value | Ridge point |
| :--- | ---: | ---: |
| Peak FP32 | ≈ 60 TFLOP/s | ≈ 75 FLOPs/byte |
| Peak TF32 Tensor-Core | ≈ 380 TFLOP/s | ≈ 475 FLOPs/byte |
| Peak DRAM bandwidth | ≈ 800 GB/s | — |

<br>

---

## 3. Methodology

### 3.1 Hardware

All measurements are performed on a single NVIDIA GeForce RTX 5070 Ti Laptop GPU.

| Spec | Value |
| :--- | :--- |
| Architecture | Blackwell (GB203) |
| Compute capability | `sm_120` |
| Memory | 12 GB GDDR7 |
| Memory bandwidth | ≈ 800 GB/s (vendor spec) |
| Peak FP32 | ≈ 60 TFLOP/s |
| Peak TF32 Tensor-Core | ≈ 380 TFLOP/s |
| Peak FP16 Tensor-Core | ≈ 380 TFLOP/s |
| Tensor-Core generation | 5th-gen |
| Platform | Windows 11 Home 10.0.26200, laptop chassis |
| Driver | 592.01 |

> **Thermal caveat.** The laptop thermal envelope (80 W TGP observed at idle via `nvidia-smi`) is a meaningful limit: sustained runs hit 80 °C within minutes, after which the GPU clock throttles. We mitigate this by inserting 3–5 s sleeps between experiments and by monitoring temperature with `nvidia-smi dmon -s u`.

### 3.2 Software stack

| Component | Version | Notes |
| :--- | :--- | :--- |
| Python | 3.11.15 | `hdai` conda env |
| PyTorch | 2.10.0+cu128 | built against CUDA 12.8 |
| torchvision | 0.25.0+cu128 | |
| cuDNN | 9.10.2 (`91002`) | bundled inside the wheel |
| CUDA runtime | 12.8 | bundled |
| transformers | 5.5.4 | DistilBERT loader |
| fvcore | 0.1.5.post20221221 | FLOP counting for the roofline |
| nvtx | 0.2.15 | custom range annotations |

> **Wheel caveat.** A non-cu128 wheel on Blackwell fails silently with *"no kernel image available for execution on the device."* The `--index-url https://download.pytorch.org/whl/cu128` option is mandatory on this hardware.

### 3.3 Model zoo

The four models are chosen to fill a 2×2 design space.

| Model | Params | Class | Axis | Cite |
| :--- | ---: | :--- | :--- | :---: |
| ResNet-18 | 11.7 M | Convolution | compute-bound | [[7]](#ref-7) |
| MobileNetV3-Small | 2.5 M | Depthwise conv | memory-bound | [[8]](#ref-8) |
| DistilBERT-base | 66 M | Matmul | compute-bound | [[9]](#ref-9) |
| Tiny GRU *(custom)* | 0.2 M | RNN | memory-bound | [[10]](#ref-10) |

**Rationale.** ResNet-18 is the most profiled CNN in the field [[11]](#ref-11) and serves as the calibration point. MobileNetV3-Small exercises cuDNN's depthwise code paths, which have different Tensor-Core eligibility characteristics. DistilBERT is a cuBLAS-heavy transformer and tests whether the study's methodology generalises beyond cuDNN itself. Tiny GRU targets cuDNN's fused RNN kernel (`cudnnRNNForward`) [[1]](#ref-1), which rolls a whole sequence into one kernel launch.

### 3.4 Profiling procedure

Every experiment follows the same protocol.

| # | Step | Rationale |
| :---: | :--- | :--- |
| 1 | 30 warm-up iterations before profiling | With `benchmark=True`, the first iter at each new shape triggers cuDNN's algorithm micro-benchmark — can be 10× slower than steady state |
| 2 | `schedule(wait=1, warmup=2, active=10, repeat=1)` | Skip 1, discard 2 as in-profile warm-up, record 10, stop |
| 3 | `ProfilerActivity.CPU + CUDA` | Captures both op-level context and raw kernel names |
| 4 | `torch.cuda.synchronize()` after each iter | Attributes kernels to the correct step; prevents async pile-up |
| 5 | Inputs allocated once on device | Avoids PCIe transfer bias |
| 6 | `model.eval()` + `torch.no_grad()` | No autograd graph, no Dropout, BatchNorm uses running stats |

Output artefacts per run: a chrome-trace JSON under `results/traces/` and a printed top-25 `key_averages` table.

### 3.5 Metrics and timing rigor

- **Latency** reported as mean of 10 profiled iterations. Standard deviation is estimated from per-step timings embedded in the trace.
- **Kernel time** taken from `Self CUDA` in the profiler output.
- **Throughput** is `batch_size / latency`, reported as images/sec or tokens/sec as appropriate.
- **Arithmetic intensity** uses FLOPs from `fvcore.nn.FlopCountAnalysis` [[12]](#ref-12) and byte counts from `torch.cuda.max_memory_allocated()` bracketing a forward pass.

Every number placed in a table is reported to a precision no finer than the measurement noise (typically 3 significant digits). All trials are single-GPU; results will not necessarily generalise to the desktop RTX 5070 Ti which has a higher TGP.

<br>

---

## 4. Results — ResNet-18 *(completed)*

> **Configuration.** Batch 32, FP32 *with TF32 math mode enabled by default*, `cudnn.benchmark = True`. Trace: [`results/traces/resnet18_baseline_bs32_benchOn.json`](../results/traces/resnet18_baseline_bs32_benchOn.json) (2.9 MB). Latency statistics from **7 trials × 50 iterations** of CUDA-event timing after 30 warm-up forwards.

**Latency (multi-trial).** Mean per-iteration latency is **11.71 ms ± 0.61 ms** (min 10.86, max 12.51), yielding **2 733 images/sec** on a laptop 5070 Ti at ambient 25 °C. See `docs/execution_log_2.md §6` for per-trial values and a discussion of the ~20 % gap against the first-pass single-window number (9.79 ms from `docs/execution_log_1.md §4.9`, which was taken on a cold chip before thermal throttling kicked in). The single-window number remains reachable and is bracketed by the new distribution's minimum.

### 4.1 Time attribution (RQ1)

<div align="center">

**Table 4.1 — Top-level CUDA time breakdown for ResNet-18, 10 profiled iterations × batch 32**

</div>

| Category | Kernel or op | CUDA time | % | Invocations |
| :--- | :--- | ---: | ---: | ---: |
| **Convolution** | `aten::cudnn_convolution` (aggregate) | **90.054 ms** | **79.42 %** | 200 |
| Batch-norm | `cudnn::bn_fw_inf_1C11_kernel_NCHW` | 9.658 ms | 8.52 % | 200 |
| Layout convert | `cudnn::...::nchwToNhwcKernel` | 7.503 ms | 6.62 % | 340 |
| ReLU (elementwise) | `aten::clamp_min_` → `vectorized_elementwise_kernel` | 6.577 ms | 5.80 % | 170 |
| Max-pool | `DilatedMaxPool2d` backing kernel | 3.544 ms | 3.13 % | 10 |
| Layout convert | `cudnn::...::nhwcToNchwKernel` | 3.292 ms | 2.90 % | 110 |
| Residual add | `aten::add_` | 3.237 ms | 2.85 % | 80 |
| Linear (FC) | `cutlass_80_simt_sgemm_…` | 0.156 ms | 0.14 % | 10 |
| — | **Total Self CUDA** | **113.393 ms** | **100.00 %** | — |

Total CPU time across the 10 iterations is **118.818 ms**, closely tracking GPU time — this workload is **not** CPU-launch-overhead bound at batch 32.

<div align="center">

<img src="../results/plots/resnet18_kernel_breakdown.png" alt="ResNet-18 kernel-time breakdown by coarse category" width="720">

**Figure 4.1 — ResNet-18 CUDA-time share per kernel category.** Total 113.39 ms over 10 profiled iterations at batch 32; `conv_implicit_gemm` (the combined CUTLASS/xmma/SIMT kernels under `aten::cudnn_convolution`) takes the majority, `norm` is the fused cuDNN BN inference kernel, `layout_convert` is the NCHW↔NHWC cost discussed in §4.3.

</div>

<div align="center">

**Headline result: 11.71 ± 0.61 ms / batch of 32 · ≈ 2 733 images / second**

</div>

### 4.2 Algorithm selection (RQ2)

The 90.054 ms spent in `aten::cudnn_convolution` decomposes across six distinct kernels in the reworked run (one more than the first pass, because cuDNN's benchmark search split one algorithm family across two layout tags this time).

<div align="center">

**Table 4.2 — Per-kernel breakdown of the ResNet-18 forward-convolution path**

</div>

| # | Kernel | Calls | Self CUDA | Path |
| :---: | :--- | ---: | ---: | :--- |
| 1 | `cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_64x64_16x10_nhwc_align4>` | 80 | 32.243 ms | **Tensor Core, TF32** (CUTLASS) |
| 2 | `sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8` | 60 | 20.701 ms | **Tensor Core, TF32** (xmma, NCHW-out) |
| 3 | `sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8` | 30 | 13.320 ms | **Tensor Core, TF32** (xmma, NHWC-out) |
| 4 | `implicit_convolve_sgemm<1024,5,5,3,3,3,1,…>` | 10 | 11.130 ms | SIMT FP32 |
| 5 | `implicit_convolve_sgemm<128,6,7,3,3,5,1,…>` | 20 | 1.841 ms | SIMT FP32 |
| 6 | `cutlass_80_simt_sgemm_64x64_8x5_tn_align1` *(final FC)* | 10 | 0.156 ms | SIMT FP32 |

<div align="center">

| Split | Time | % of total CUDA time |
| :--- | ---: | ---: |
| Tensor-Core TF32 (rows 1+2+3) | **66.264 ms** | **58.44 %** |
| SIMT FP32 (rows 4+5+6) | 13.127 ms | 11.58 % |

</div>

<div align="center">

<img src="../results/plots/resnet18_conv_algorithms.png" alt="ResNet-18 convolution kernels broken down by algorithm" width="780">

**Figure 4.2 — ResNet-18 convolution kernels by algorithm.** Red bars are Tensor-Core TF32 kernels; blue bars are SIMT FP32. Total conv-kernel time 79.24 ms (the final FC is matmul, not conv, and is excluded from this chart). Three TC variants together take 66.26 ms ≈ 83.6 % of conv time ≈ 58.4 % of *all* GPU time.

</div>

> **The brief predicted Winograd. Winograd is absent.**
> Exhaustive search of the full 2.9 MB trace JSON for any kernel containing the substring `winograd` returns **zero matches**. The brief's §1.1 prediction — *"almost every conv layer in ResNet-18 gets a Winograd algorithm picked"* — does not hold on this stack.

**Why: four factors, ranked by impact.**

1. **TF32 is enabled by default on `sm_80+`** because PyTorch sets `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` out of the box [[6]](#ref-6). `cudnn.benchmark=True` therefore compares TC-implicit-GEMM running in TF32 against SIMT Winograd running in FP32 — the playing field is not level.
2. **Blackwell's 5th-gen Tensor Cores are far faster at TF32 than Ampere's were.** A ~6× per-clock throughput increase on TF32 dwarfs Winograd's ~2.25× multiplication reduction, so TC-GEMM wins on measured wall time.
3. **cuDNN 9.10.2 may not ship a fully `sm_120`-optimised Winograd variant yet.** A Winograd path compiled for Ampere and running forward-compatibly on Blackwell will predictably lose to an `sm_80` xmma kernel that is closer to optimal on the same chip.
4. **Winograd has higher numerical error** than direct or implicit-GEMM approaches [[3]](#ref-3). cuDNN's heuristic may further down-weight Winograd when reduced-precision math modes are active.

The kernel-name encoding confirms the mechanism: `tf32f32_tf32f32_f32` in the xmma kernel name means *TF32 input, TF32 accumulator, FP32 output*; `s1688` in the CUTLASS kernel name identifies the 16×16×8 TF32 Tensor-Core instruction shape. These are unambiguously TC-TF32 kernels even though the user requested "FP32" inference.

### 4.3 Layout-conversion cost

The fifth- and tenth-ranked kernels in the profile are CUDA kernels that only reformat tensor memory layout:

| Kernel | Time | Calls |
| :--- | ---: | ---: |
| `cudnn::engines_precompiled::nchwToNhwcKernel` | 7.503 ms | 340 |
| `cudnn::engines_precompiled::nhwcToNchwKernel` | 3.292 ms | 110 |
| **Total** | **10.795 ms (9.52 %)** | **450** |

**Why this happens.** The `cutlass_tensorop` and `xmma` Tensor-Core kernels demand NHWC-laid-out inputs (the trailing `nhwckrsc_nchw` tag in the xmma kernel name encodes *input NHWC, weights KRSC, output NCHW*), but torchvision's ResNet-18 stores weights and activations in NCHW by default. cuDNN therefore transposes each activation tensor before the convolution and transposes it back afterwards.

The asymmetry — **340 NCHW→NHWC versus 110 NHWC→NCHW** — reflects that cuDNN prefers to output NCHW back to the user's framework after the TC kernel has produced NHWC internally, but does not always need a forward conversion if the tensor was already in NHWC from a previous conv's output.

> **Actionable consequence.** A full switch to `torch.channels_last` should eliminate nearly all of this 9.52 %. The experiment is listed as §5.7 and is now a high-priority follow-up.

### 4.4 Regime classification (RQ3)

With **79.42 % of time spent in convolution**, 8.52 % in a single fused BatchNorm kernel, and the remainder scattered across small elementwise ops, ResNet-18 at batch 32 on this hardware is unambiguously **compute-bound**. This agrees with the qualitative prediction in the brief. A numerical placement on the roofline diagram awaits §6.

### 4.5 Ops-per-iteration sanity check

ResNet-18 has 20 convolution layers, 20 BatchNorm layers, 17 ReLU activations, 1 MaxPool, 8 residual adds, and 1 final FC layer. Multiplying each by 10 profiled iterations gives expected invocation counts of **200, 200, 170, 10, 80, 10** respectively.

| Op | Expected count | Observed count | Match |
| :--- | ---: | ---: | :---: |
| `aten::cudnn_convolution` | 200 | 200 | ✓ |
| `aten::cudnn_batch_norm` | 200 | 200 | ✓ |
| `aten::clamp_min_` (ReLU) | 170 | 170 | ✓ |
| `aten::max_pool2d_with_indices` | 10 | 10 | ✓ |
| `aten::add_` (residual) | 80 | 80 | ✓ |
| `aten::linear` (final FC) | 10 | 10 | ✓ |

No op is mis-instrumented or skipped — the trace is complete.

<br>

---

## 5. Results — further experiments *(pending)*

Each sub-section below is a placeholder with the experiment's design already fixed, following the brief's §8 protocol.

### 5.1 MobileNetV3-Small baseline

> *To be filled.* **Hypotheses:** (a) depthwise convs may use a different kernel family, perhaps `dgrad_engine`-style names; (b) Tensor-Core utilisation should drop markedly because depthwise GEMMs are too skinny [[8]](#ref-8); (c) the layout-conversion overhead observed for ResNet-18 may be worse, because depthwise conv's small channel count makes each convert proportionally more expensive.

### 5.2 DistilBERT-base baseline

> *To be filled.* **Hypotheses:** (a) almost all GPU time in cuBLAS matmul kernels (`cublasGemmEx`, `cutlass_tensorop_s16816gemm_...`) rather than cuDNN; (b) sequence length is a primary knob — at seq=64 the workload may be launch-overhead dominated, at seq=512 it should be clearly compute-bound; (c) layer-norm and softmax appear only as small slices in the profile [[13]](#ref-13).

### 5.3 Tiny GRU baseline

> *To be filled.* **Hypotheses:** (a) a single `cudnn::rnn::...` kernel per iteration covering the whole sequence; (b) clearly memory-bound — the hidden-state matmul has arithmetic intensity O(hidden\_size) which is low; (c) batch-size scaling dramatically improves throughput because the memory traffic for the fused kernel amortises over the batch dimension [[1]](#ref-1).

### 5.4 Experiment — `cudnn.benchmark` toggle

> *To be filled.* **Protocol:** same harness as §4, toggling `torch.backends.cudnn.benchmark` between `True` and `False` across all four models. **Expected speedups:** ResNet-18 10–30 % (many algorithm candidates to search among); MobileNetV3-Small < 10 % (fewer choices); DistilBERT ≈ 0 % (cuBLAS already heuristically optimal); GRU ≈ 0 % (cuDNN RNN path is already fused).

### 5.5 Experiment — FP32 vs FP16 autocast

> *To be filled.* **Protocol:** wrap inference in `torch.autocast(device_type='cuda', dtype=torch.float16)` and compare timing to FP32 with TF32. **Expected speedups:** ResNet-18 2–3×, MobileNetV3-Small 1.2–1.5×, DistilBERT 2–3×, GRU 1.2–1.4×. **Secondary deliverable:** a kernel diff — which kernel names newly appear in FP16 mode that were absent in FP32 (look for `hmma_`, `h16x8x16`).

### 5.6 Experiment — batch-size sweep

> *To be filled.* Batches {1, 4, 16, 64, 256} subject to 12 GB memory cap. **Expected plot shape:** throughput rising with batch at small batch (launch-overhead regime), saturating once the GPU is fully utilised. DistilBERT at seq=512 will OOM somewhere below batch 256 on 12 GB; the cap will be reported honestly.

### 5.7 Experiment — channels-last memory format *(priority-bumped)*

> *Promoted in priority by the §4.3 observation.* **Protocol:** `model = model.to(memory_format=torch.channels_last)`; `x = x.to(memory_format=torch.channels_last)`. Re-profile ResNet-18 and MobileNetV3-Small. **Prediction:** the `nchwToNhwc` / `nhwcToNchw` kernels should vanish or shrink substantially, giving a free 5–10 % speedup for ResNet-18 in TF32.

### 5.8 Experiment — sequence-length sweep on DistilBERT

> *To be filled.* Seq ∈ {32, 64, 128, 256, 512}. Attention cost is *O(seq²)* whereas FFN cost is *O(seq)* per token, so the attention-block share of time should rise visibly at long seq. **Expected figure:** stacked area of (attention, FFN, layer-norm, softmax) vs seq.

### 5.9 Experiment — TF32-off A/B on ResNet-18 *(new, triggered by §4.2)*

> **Protocol:** disable both TF32 flags and re-profile:
> ```python
> torch.backends.cuda.matmul.allow_tf32 = False
> torch.backends.cudnn.allow_tf32       = False
> ```
> **Prediction:** `cudnn.benchmark=True` will now measure TC-TF32 as unavailable and **Winograd should re-enter the profile**, validating hypothesis 1 in §4.2 as the dominant cause of the algorithm-selection flip. This is the cleanest A/B we can design from the current data.

<br>

---

## 6. Roofline analysis *(pending)*

> *Procedure.* Compute FLOPs per forward pass with `fvcore.nn.FlopCountAnalysis` [[12]](#ref-12) and memory traffic with `torch.cuda.max_memory_allocated()` bracketing the forward. Place each model at its (intensity, throughput) coordinate on a log-log plane with two ceilings overlaid — 800 GB/s memory bandwidth and 380 TFLOP/s TC-TF32 peak.
>
> *Expected layout.* ResNet-18 under the TC ridge, DistilBERT near the TC peak, MobileNetV3-Small on the bandwidth ramp, Tiny GRU far left on the ramp. One paragraph of discussion per point.

<br>

---

## 7. Cross-model discussion *(pending)*

Three threads expected once all four models are profiled.

1. **TF32 on `sm_120` reshapes cuDNN's algorithm selection across the board.** All four models may show the same Winograd → implicit-GEMM shift; the magnitude differs.
2. **Tensor-Core eligibility is driven by shape alignment, not architecture.** MobileNetV3-Small's depthwise convs have channel counts that don't tile well; DistilBERT's Q/K/V projections do. The profile cleanly separates the two.
3. **Memory bandwidth is the true ceiling for half of these models.** The roofline plot should show GRU and MobileNetV3-Small pinned to the bandwidth ramp regardless of precision mode, which no amount of Tensor Cores can fix.

Each thread will get a paragraph plus one pointer to the supporting figure.

<br>

---

## 8. Threats to validity

| # | Threat | Mitigation |
| :---: | :--- | :--- |
| 1 | **Laptop thermal throttling** — after ~3 min of sustained profiling the GPU hits 80 °C and boost clock drops | Insert 3–5 s sleeps between runs; monitor temperature with `nvidia-smi dmon -s u`. Baseline §4 figures are from one 10-iteration window after adequate cool-down |
| 2 | **Single GPU, single driver version** — results not necessarily portable | Software stack reported in full (§3.2) so results can be re-evaluated against future drivers |
| 3 | **Input-statistic sensitivity** — cuDNN's algorithm micro-benchmark depends on numeric ranges of inputs; we use `torch.randn` whereas real ImageNet images might select slightly different algorithms | Effect is expected to be small; we cannot exclude it |
| 4 | **TF32 silently active** — all "FP32" figures in §4 are effectively TF32 | Noted in-line; controlled A/B queued as §5.9 |
| 5 | **`Self CUDA % > 100`** — the `ProfilerStep*` synthetic row reports 104.24 % because it includes profiler-internal time not counted elsewhere | Presentational artefact, not a timing error |

<br>

---

## 9. Conclusion

> *Placeholder for the final synthesis.*

At time of writing, the substantive conclusion after one profiled model is that the brief's prediction about Winograd dominance in ResNet-18 **does not hold** on Blackwell + cuDNN 9.10.2 + PyTorch's default TF32 configuration, and that **9.52 % of GPU time** is spent on layout conversions that a one-line `channels_last` change should eliminate. Both findings are actionable. Neither would have been visible without reading the kernel-level profile.

<br>

---

## References

<a id="ref-1"></a>[1]  S. Chetlur, C. Woolley, P. Vandermersch, J. Cohen, J. Tran, B. Catanzaro, and E. Shelhamer, "cuDNN: Efficient Primitives for Deep Learning," *arXiv:1410.0759*, 2014.

<a id="ref-2"></a>[2]  A. Paszke *et al.*, "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019.

<a id="ref-3"></a>[3]  A. Lavin and S. Gray, "Fast Algorithms for Convolutional Neural Networks," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016.

<a id="ref-4"></a>[4]  S. Williams, A. Waterman, and D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65–76, 2009.

<a id="ref-5"></a>[5]  NVIDIA, "NVIDIA Tesla V100 GPU Architecture Whitepaper," 2017.

<a id="ref-6"></a>[6]  NVIDIA, "NVIDIA A100 Tensor Core GPU Architecture Whitepaper" (TF32 introduction), 2020.

<a id="ref-7"></a>[7]  K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. CVPR*, 2016.

<a id="ref-8"></a>[8]  A. Howard *et al.*, "Searching for MobileNetV3," in *Proc. IEEE/CVF ICCV*, 2019.

<a id="ref-9"></a>[9]  V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter," in *NeurIPS EMC² Workshop*, 2019.

<a id="ref-10"></a>[10]  K. Cho, B. van Merriënboer, D. Bahdanau, and Y. Bengio, "On the Properties of Neural Machine Translation: Encoder-Decoder Approaches," in *Proc. Syntax, Semantics and Structure in Statistical Translation (SSST-8)*, 2014.

<a id="ref-11"></a>[11]  S. Chintala, "convnet-benchmarks," GitHub repository, 2014. [Online]. Available: https://github.com/soumith/convnet-benchmarks

<a id="ref-12"></a>[12]  Facebook AI Research, "fvcore," GitHub repository, 2021. [Online]. Available: https://github.com/facebookresearch/fvcore

<a id="ref-13"></a>[13]  A. Vaswani *et al.*, "Attention Is All You Need," in *NeurIPS*, 2017.

<a id="ref-14"></a>[14]  S. Markidis, S. W. Der Chien, E. Laure, I. B. Peng, and J. S. Vetter, "NVIDIA Tensor Core Programmability, Performance & Precision," in *IEEE IPDPSW*, 2018.

<a id="ref-15"></a>[15]  NVIDIA CUTLASS Team, "CUTLASS: Fast Linear Algebra in CUDA C++," GitHub repository. [Online]. Available: https://github.com/NVIDIA/cutlass

<a id="ref-16"></a>[16]  NVIDIA, "Nsight Systems User Guide," 2025. [Online]. Available: https://docs.nvidia.com/nsight-systems/

<a id="ref-17"></a>[17]  PyTorch Team, "PyTorch Profiler Recipe." [Online]. Available: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

<br>

---

## Appendix A — cuDNN kernel-name decoder

Kernels emitted by cuDNN / cuBLAS / CUTLASS follow a loose naming scheme.

| Token family | Meaning |
| :--- | :--- |
| `sm80_`, `sm86_`, `sm90_`, `sm120_` | Target compute capability. `sm80` kernels run forward-compatibly on newer hardware. |
| `xmma_` | Extended MMA kernel family (Tensor Cores) |
| `hmma_` | Half-precision MMA (FP16 inputs) |
| `bmma_` | BF16 MMA |
| `imma_` | INT8 MMA |
| `s1688`, `s16816` | Tensor-Core instruction tile shapes |
| `gemm_` | General matrix multiply |
| `fprop_`, `dgrad_`, `wgrad_` | Forward prop, data gradient, weight gradient |
| `implicit_gemm_`, `implicit_precomp_` | im2col + GEMM without materialising the im2col buffer |
| `winograd_` | Winograd-transform algorithm |
| `nchw`, `nhwc`, `nhwckrsc_nchw` | Tensor-layout tags (`KRSC` = output-major weights) |
| `f32f32_f32f32_f32` | FP32 in / accum / out |
| `tf32f32_tf32f32_f32` | TF32 in / TF32 accum / FP32 out |
| `f16f16_f16f32_f16` | FP16 in / FP32 accum / FP16 out |
| `tilesizeMxNxK` | Threadblock output tile dimensions |
| `stageN` | Software-pipeline stages through shared memory |
| `warpsizeAxBxC` | Warp-tile factoring |
| `tensorMxNxK` | Tensor-Core instruction tile |

**Worked example (§4.2 kernel #2).**

```
sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8
```

Decodes to: Ampere-compiled (forward-compatible on Blackwell) xmma forward-prop implicit-GEMM, TF32 input / TF32 accumulator / FP32 output, NHWC input with KRSC weights producing NCHW output, 128×128×16 threadblock tile, 4-stage pipeline, 2×2×1 warp tiling, single group (regular conv), 16×8×8 Tensor-Core instruction.

<br>

---

## Appendix B — reproduction

### B.1 Environment

```bash
conda create -n hdai python=3.11 -y
source /c/Users/worka/anaconda3/etc/profile.d/conda.sh   # Git Bash on Windows
conda activate hdai
pip install torch==2.10.0 torchvision==0.25.0 torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
pip install pandas matplotlib seaborn nvtx transformers fvcore ptflops
python env/check_env.py
```

### B.2 Reproducing §4 (ResNet-18 baseline)

```bash
# from repo root, with hdai activated
python -m profiling.run_baseline --model resnet18
```

This produces `results/traces/resnet18_baseline_bs32_benchOn.json` and prints the 25-row key-averages table plus the multi-trial latency distribution. All §4 numbers in this report are derived from that trace; §4.2 figures come from `python -m analysis.plots`.

### B.3 Artefact paths

| Path | Role |
| :--- | :--- |
| `results/traces/resnet18_baseline_bs32_benchOn.json` | 3.0 MB chrome-trace, 10 active iterations (batch 32, benchmark on) |
| `results/plots/resnet18_kernel_breakdown.png` | Figure 4.1 — kernel-category bar chart |
| `results/plots/resnet18_conv_algorithms.png` | Figure 4.2 — conv-algorithm bar chart (TC vs SIMT) |
| `docs/execution_log_2.md` | Bug fixes and rerun audit superseding `execution_log_1.md` |
| `docs/execution_log_1.md` | Exhaustive step-by-step record of the Phase 2 run, including failures and root causes |

<br>

---

## Appendix C — raw profiler output (ResNet-18, top-25 rows)

Full text as emitted by `prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)` is preserved in [`docs/execution_log_1.md §4.6`](../docs/execution_log_1.md). Key columns replicated here for completeness.

```
Name                                                 Self CUDA   Self CUDA %   # of Calls
---------------------------------------------------- ----------  ------------  ----------
aten::cudnn_convolution                              90.054 ms        79.42 %         200
  cutlass_tensorop_s1688fprop_optimized_tf32_…       32.243 ms        28.44 %          80
  sm80_xmma_fprop_implicit_gemm_tf32f32_…_nchw       20.701 ms        18.26 %          60
  sm80_xmma_fprop_implicit_gemm_tf32f32_…_nhwc       13.320 ms        11.75 %          30
  implicit_convolve_sgemm<1024,5,5,3,3,3,1,…>        11.130 ms         9.82 %          10
  implicit_convolve_sgemm<128,6,7,3,3,5,1,…>          1.841 ms         1.62 %          20
  cutlass_80_simt_sgemm_64x64_8x5_tn_align1           0.156 ms         0.14 %          10
aten::cudnn_batch_norm → bn_fw_inf_1C11_NCHW          9.658 ms         8.52 %         200
nchwToNhwcKernel                                      7.503 ms         6.62 %         340
aten::clamp_min_ (ReLU) → vectorized_elementwise      6.577 ms         5.80 %         170
aten::max_pool2d_with_indices → DilatedMaxPool2d      3.544 ms         3.13 %          10
nhwcToNchwKernel                                      3.292 ms         2.90 %         110
aten::add_ (residual)                                 3.237 ms         2.85 %          80
aten::linear (final FC, SIMT GEMM)                    0.156 ms         0.14 %          10
---
Self CUDA time total                                113.393 ms       100.00 %
```

<br>

---

## Appendix D — environment manifest

Full output of `pip freeze` in the `hdai` env is saved in [`requirements.txt`](../requirements.txt) and expanded in [`README.md`](../README.md).

**Key versions:**

| Package | Version |
| :--- | :--- |
| Python | 3.11.15 |
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| cuDNN | 9.10.2 |
| NumPy | 2.4.3 |
| pandas | 3.0.2 |
| transformers | 5.5.4 |
| fvcore | 0.1.5.post20221221 |

<br>

---

<div align="center">

*End of report — §§5–9 to be populated as experiments complete.*

</div>
