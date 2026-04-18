# cuDNN Profiling Project — Complete Plan

**Course:** EEL71020 — Hardware Design for AI, IIT Jodhpur
**Authors:** Anshul Kumar (M25AI2036), Neha Prasad (M25AI2056)
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
| Phase 2 — first profile on ResNet-18 | [x] done (superseded by rework) | 97.923 ms CUDA / 10 iters @ batch 32 on cold chip. See `execution_log_1.md`. |
| Phase 2 rework — bug fixes, multi-trial rerun, plots | [x] done | 11.71 ± 0.61 ms mean / 2 733 img/s (7 trials × 50 iters). Bugs in `run_baseline.py` fixed. Plots in `results/plots/`. See `execution_log_2.md`. |
| Phase 3 — port to MobileNetV3 / DistilBERT / GRU | [x] done | 4 baselines complete. MobileNetV3 @ 3.01 ms (TC 15%, BN 22%), DistilBERT @ 12.36 ms (**0% TC — MAGMA dispatch, not cuBLAS**), GRU @ 0.25 ms (persistent RNN + TC input matmul). See `execution_log_3.md`. |
| Phase 4 — kernel classification | [x] done | Classifier gains `conv_depthwise`, `fused_attention`, `embed_gather` buckets (zero `other` leakage on all 4 traces). `results/tables/baseline_breakdown.csv` emitted by new `analysis/compute_summary.py`. Cross-model summary § 5.5 added to writeup. See `execution_log_4.md`. |
| Phase 5 — Nsight Systems timeline | [x] done | Nsight 2026.2.1 installed, 4 `.nsys-rep` captured, 8 screenshots, `analysis/cross_check_nsight.py` agreement ≤17 %. NVTX ranges + `_print_flags()` added to `run_baseline.py`. Winograd-in-warmup finding + MobileNetV3 5.3 s algo-search cost documented in writeup §5.6. See `execution_log_5.md`. |
| Phase 6 — `cudnn.benchmark` toggle | [ ] pending | `run_baseline.py` now supports `--no-benchmark` (log_2 fix) |
| Phase 7 — FP32 vs FP16 (AMP) | [ ] pending | |
| Phase 8 — batch-size sweep | [ ] pending | cap at what fits in 12 GB, not 16 GB |
| Phase 9 — roofline analysis | [ ] pending | |
| Phase 10 — cleanup and plots | [ ] pending | `channels_last` promoted in priority — layout converts are 9.52% of CUDA time in baseline |
| Phase 11 — writeup | [ ] pending | |
| Phase 12 — buffer | [ ] pending | |

**Extra experiment queued (not in original plan):** TF32-off re-profile of ResNet-18. On PyTorch 2.10.0+cu128 the Phase 5 `[flags]` snapshot already shows `torch.backends.cuda.matmul.allow_tf32 = False` out of the box, so the experiment reduces to flipping **`torch.backends.cudnn.allow_tf32 = False`** (currently True, and the actual route through which ResNet-18's 58.45 % TC share is dispatched). Phase 5 already surfaced a partial answer: under short warmup, Winograd *does* reappear in the Nsight capture at 3.0 % (§ Phase-5 row above) — so cuDNN's benchmark search considers it, it just loses to TF32 TC implicit-GEMM in steady state. A controlled TF32-off A/B would be the clean paired-bar figure for the writeup.

### Findings so far (end of Phase 5, four models profiled, classifier wired, Nsight timelines captured)

Per-phase logs: [`execution_log_2.md`](execution_log_2.md) (ResNet-18 rework), [`execution_log_3.md`](execution_log_3.md) (three-model port), [`execution_log_4.md`](execution_log_4.md) (classifier + cross-model CSV), [`execution_log_5.md`](execution_log_5.md) (Nsight + NVTX + cross-check).

**Four-model latency/TC summary (batch per DEFAULT_BATCH, FP32+TF32 via cuDNN path, `cudnn.benchmark=True`):**

| Model | Batch | Latency (ms) | Throughput (samples/s) | TC share |
|---|---:|---:|---:|---:|
| ResNet-18 | 32 | 11.71 ± 0.61 | 2 733 | 58.45 % |
| MobileNetV3-Small | 32 |  3.01 ± 0.18 | 10 644 | 14.94 % |
| DistilBERT-base |  8 | 12.36 ± 0.44 | 647 | **0.00 %** |
| Tiny GRU | 32 |  0.25 ± 0.01 | 127 003 | 16.77 % |

Full 25-column per-category breakdown in [`results/tables/baseline_breakdown.csv`](../results/tables/baseline_breakdown.csv) (Phase 4 centerpiece).

**Key findings (Phases 2–4):**

- **ResNet-18:** 79.42 % conv; brief predicted Winograd, observed TF32 Tensor-Core implicit-GEMM in full-warmup baseline; layout-converts eat 9.52 % of CUDA time (motivates `channels_last` sooner).
- **MobileNetV3-Small:** BN unexpectedly large (21.83 %) because it amortises badly over tiny convs; depthwise conv (25.71 %) routes through PyTorch-native kernels, not cuDNN; TC share drops to 15 % (only pointwise 1×1 convs are TC-eligible).
- **DistilBERT-base:** 91.89 % of time in `aten::addmm` backed by **MAGMA** (not cuBLAS as brief predicted); **0 % Tensor-Core engagement** at FP32; attention fully fused via `fmha_cutlassF_f32_aligned_64x64_rf_sm80` FlashAttention. Strongest finding of the study so far — dispatcher routed matmuls to a non-TC library.
- **Tiny GRU:** cuDNN's persistent-RNN kernel (`RNN_blockPersist_fp_GRU`) dominates at 73.33 %; 16.77 % of time in `cutlass_80_tensorop_s1688gemm_128x256` on the input-to-hidden matmul (partial TC engagement).
- **Two TF32 flags, not one.** PyTorch 2.10.0+cu128 defaults `torch.backends.cudnn.allow_tf32 = True` but `torch.backends.cuda.matmul.allow_tf32 = False` (Phase 5 `[flags]` snapshot). "FP32 inference" therefore means different things for the cuDNN conv path vs the aten matmul path, and the brief's original "one flag" framing is imprecise.

**Additional findings from Phase 5 (Nsight timeline):**

- **cuDNN `benchmark=True` warmup is proportional to unique conv-shape count.** ResNet-18 warms up in 468.8 ms; MobileNetV3-Small takes **5.309 s** — an 11× gap driven by MobileNetV3's ~50+ distinct conv shapes, each needing an algorithm probe. During the long MobileNetV3 warmup the CUDA HW row is visibly sparse (GPU idle) while the host-side cuDNN search runs.
- **Winograd is not absent on Blackwell — it loses in steady state.** ResNet-18's short-warmup (10-iter) Nsight capture contains `cudnn::winograd_nonfused::winogradForwardFilter4x4` at 3.0 % of GPU time. cuDNN probes Winograd during algorithm search, then picks TF32 TC implicit-GEMM at convergence.
- **DistilBERT does probe cuBLAS at the API level** (visible as narrow ticks on the `cuBLAS` row in the Nsight trace) even though MAGMA runs the kernel. The §5.2.2 MAGMA finding is a kernel-row observation, not an API-row one.
- **PyTorch Profiler ↔ Nsight cross-check** passes at worst 17.1 % per-iter disagreement; ResNet-18 and GRU agree within 2.1 %. [`analysis/cross_check_nsight.py`](../analysis/cross_check_nsight.py) is the reproducible regression test.

### Environment subsections completed

- 2.1 `nvidia-smi` verified — driver 592.01, CUDA 13.1 reported, 12 GB
- 2.3 conda env (`hdai` with Python 3.11.15)
- 2.4 PyTorch cu128 wheel (torch 2.10.0+cu128, torchvision 0.25.0+cu128, torchaudio 2.11.0+cu128)
- 2.5 project packages (pandas, matplotlib, seaborn, nvtx, transformers, fvcore, ptflops; pillow pulled transitively)
- 2.6 **Nsight Systems 2026.2.1** installed at `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.2.1\` (Phase 5 unblocked). `nsys.exe` (CLI) + `nsys-ui.exe` (GUI) both verified working. Not on PATH; [`profiling/run_nsight.sh`](../profiling/run_nsight.sh) uses absolute-path fallback.
- 2.8 cuDNN reachability smoke test (3×3 conv on 16×64×56×56 dispatches correctly)

Still to do / deliberately skipped from section 2:
- 2.2 CUDA Toolkit 12.8 — **not installed, not needed.** The cu128 PyTorch wheel bundles the full CUDA 12.8 runtime; `nsys` ships with Nsight Systems (§2.6) independently of any Toolkit install. A Toolkit install would only be needed if we wanted to build raw-C cuDNN benchmarks or use standalone `nvcc`.
- 2.7 Nsight Compute (optional stretch) — not installed. Not required for any phase 1–12 deliverable.

See `execution_log_0.md` for the full bootstrap trace, `execution_log_1.md` for the first-pass Phase 2 profile, `execution_log_2.md` for the reworked Phase 2 (bug fixes, multi-trial rerun, analysis plots), `execution_log_3.md` for the three-model port, `execution_log_4.md` for the classifier + cross-model CSV, and `execution_log_5.md` for Nsight capture + NVTX + cross-check.

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

> **Observed on this hardware (Phase 2 rework, 2026-04-17).** Conv came in at 79.42% — the "70–80%" band holds. But *zero* Winograd kernels appeared. PyTorch's default `allow_tf32=True` steered cuDNN's benchmark search toward TF32 Tensor-Core implicit-GEMM instead: 28.44% of CUDA time in `cutlass_tensorop_s1688fprop_optimized_tf32_64x64_16x10_nhwc_align4`, 18.26% in `sm80_xmma_fprop_implicit_gemm_tf32f32_..._nhwckrsc_nchw`, and 11.75% in a second `...nhwckrsc_nhwc` variant of the same xmma kernel (total 58.44% through Tensor Cores). On Blackwell + cuDNN 9.10.2, Blackwell's 5th-gen Tensor Cores appear to beat any Winograd variant cuDNN has compiled for `sm_120`. The prediction in this paragraph holds in *spirit* (conv dominates, Tensor Cores engage); the specific *algorithm name* is different. See `execution_log_2.md §6` for the full kernel inventory.

### Model 2 — MobileNetV3-Small (memory-bound conv, depthwise-dominated)

torchvision model, ~2.5M parameters. The architecture is almost entirely depthwise separable convolutions, which have low arithmetic intensity (~channel_count FLOPs per weight loaded instead of ~kernel_size² × channel_count FLOPs like regular conv). Tensor Cores don't help much because the GEMMs are too skinny.

Why include it: because a huge chunk of modern edge vision uses this pattern, and because the *contrast* with ResNet-18 is where the learning happens. Same task (ImageNet classification), much smaller parameter count, but the profile looks completely different. You see cuDNN's depthwise-specific kernels, you see far less Tensor Core usage, and the speedup from FP16 is muted — maybe 1.2× instead of ResNet's 2–3×. This is the canonical example of "Tensor Cores don't save you if your workload is memory-bound."

What we expect to see: time breakdown roughly 60% depthwise conv + 30% pointwise (1×1) conv + 10% misc, FP16 speedup small, `cudnn.benchmark` helps only modestly because the depthwise kernels have fewer algorithm choices to pick between.

> **Observed on this hardware (Phase 3, 2026-04-18).** Decomposition is *different from the prediction*: 37.6 % convolution (11.88 % `conv_implicit_gemm` pointwise + 25.71 % `conv_depthwise` dispatched through PyTorch-native `aten::_conv_depthwise2d`, not cuDNN), **21.83 % BatchNorm** (vs predicted ~10 % "misc" — BN amortises badly over tiny convs), 21.1 % matmul (15.52 % elementwise + pointwise helpers). TC share 14.94 % (only the 1×1 pointwise convs hit TC). Phase 5 Nsight finding: MobileNetV3's `cudnn.benchmark=True` **warmup is 5.309 s** — ~11× longer than ResNet-18's — because ~50+ distinct depthwise/pointwise/stride shapes each trigger an algorithm search. See `execution_log_3.md §3.1–3.2` and `execution_log_5.md §6`.

### Model 3 — DistilBERT-base (compute-bound matmul, Tensor Core showcase)

HuggingFace `distilbert-base-uncased`, 66M parameters. Bigger than the other three combined, but it's still well under any memory ceiling and is the natural representative of "transformer that people actually use in production." The inference workload is almost entirely matmul — Q/K/V projections, attention, and FFN layers — with softmax and layer norm as rounding errors.

Important note: most of DistilBERT's matmuls go through cuBLAS, not cuDNN. This is a good thing to observe and discuss in the writeup — cuDNN is the deep-learning library but transformer-heavy workloads hit cuBLAS more than cuDNN. The handful of cuDNN calls that do show up are for layer norm and occasionally softmax on some cuDNN versions.

What we expect to see: 80%+ of time in `cublas` GEMM kernels with Tensor Core variants, FP16 speedup of 2–3× (attention scales well on Tensor Cores), and the role of sequence length as a knob that moves you from memory-bound-ish at seq=64 to clearly compute-bound at seq=512.

> **Observed on this hardware (Phase 3, 2026-04-18).** The library prediction is *wrong*: `aten::addmm` routes to **MAGMA's `magma_sgemmEx_kernel`** (91.89 % of GPU time), not cuBLAS. The Tensor Core prediction is also wrong at FP32: **0.00 % TC engagement** at batch 8 / seq 128. Attention is fully fused into one `fmha_cutlassF_f32_aligned_64x64_rf_sm80` FlashAttention kernel (4.66 %) — no separate softmax row, no separate Q·Kᵀ matmul row. Phase 5 refinement: the `cuBLAS` *API row* in Nsight *is* populated (narrow ticks per `torch.addmm` call — torch probes cuBLAS at dispatch time) but execution lands in MAGMA, so the "no cuBLAS" claim holds at the *kernel* level, not the *API* level. FP16 speedup is Phase-7 work; the seq-sweep is Phase 12 optional. See `execution_log_3.md §3.3–3.4` and `execution_log_5.md §6.2.3`.

### Model 4 — Tiny GRU (memory-bound recurrent, cuDNN RNN path)

Custom-built: 2-layer GRU with hidden size 128, input size 64, sequence length variable. About 200K parameters. Sentiment-classifier-sized. Negligible to train, takes milliseconds to run.

Why include it: to see cuDNN's RNN path, which is genuinely different code from its conv and matmul paths. cuDNN provides a fused RNN/GRU/LSTM kernel (`cudnnRNNForward`) that wraps the entire sequence-length loop into one call. PyTorch uses this automatically via `nn.GRU`. Inside the profiler it shows up as a single persistent cuDNN kernel with the whole sequence rolled in, which looks different from per-timestep kernel launches you'd see in a naive implementation.

Sequence-based RNN inference is memory-bound almost by definition — you're doing a matmul of a tiny matrix (hidden × hidden) at every step, and the weights are tiny relative to the activations you're shuttling around. This gives us our fourth corner: conv compute-bound (ResNet-18), conv memory-bound (MobileNetV3-Small), matmul compute-bound (DistilBERT), matmul memory-bound (GRU). Clean quadrant.

What we expect to see: a single `cudnn::rnn::...` kernel dominating the timeline, FP16 speedup modest (maybe 1.3×), and the "compute" column of the profile looking nothing like the other three models. Batch size scaling is dramatic here — going from batch 1 to batch 128 actually makes GRU look reasonable throughput-wise because you're amortizing the memory traffic.

> **Observed on this hardware (Phase 3, 2026-04-18).** Prediction basically holds: `aten::_cudnn_rnn` at 96.1 %, with the persistent `RNN_blockPersist_fp_GRU` kernel alone at 73.33 %. Unexpected bonus: 16.77 % TC engagement via `cutlass_80_tensorop_s1688gemm_128x256_16x3_tn_align4` on the input-to-hidden matmul — brief predicted "modest" TC; "modest" holds. Phase 5 Nsight finding: at `iter_05 = 409.917 µs` the timeline shows **visible idle gaps between iterations**, directly confirming the launch-overhead-bound signature the brief predicts. FP16 speedup and batch sweep remain Phase 7/8 work. See `execution_log_3.md §3.5–3.6` and `execution_log_5.md §6.2.4`.

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

The RTX 5070 Ti Laptop GPU is Blackwell architecture, compute capability sm_120. This is newer than most PyTorch tutorials assume, and the default `pip install torch` will give you a wheel that was compiled without sm_120 support, which fails silently with an opaque "no kernel image available" error at first CUDA call.

### 2.1 Verify hardware and drivers

Open PowerShell. Run:

```powershell
nvidia-smi
```

Expected output: your driver version, CUDA version reported (driver-level, not toolkit-level), and the RTX 5070 Ti Laptop GPU with its memory and power state. Write down the driver version — you want it at or above 570.xx for full sm_120 support. If it's older, update from https://www.nvidia.com/Download/index.aspx before doing anything else. This takes 15 minutes and requires a reboot, and it's the single biggest source of "nothing works" failures on Blackwell.

### 2.2 Install CUDA Toolkit 12.8 (skipped — not needed for this project)

PyTorch ships its own CUDA runtime inside the wheel, so you don't need a separate CUDA Toolkit install to run PyTorch + cuDNN. The only reasons to install one:
- `nvcc` (only if you want to compile raw CUDA code like Slimakanzer/cudnn-benchmark)
- Nsight Systems / Nsight Compute — but these ship as **standalone installers** on the NVIDIA developer site; the CUDA Toolkit installer is not the only route.

> **Decision on this project (Phase 5, 2026-04-18):** the CUDA Toolkit was *not* installed. Nsight Systems 2026.2.1 was installed standalone from https://developer.nvidia.com/nsight-systems. All four models profile correctly with the cu128-bundled runtime. If you follow this plan fresh, you can skip §2.2 entirely — just grab Nsight Systems from its own download page.

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
print(torch.cuda.get_device_name(0))         # "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
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

Download from https://developer.nvidia.com/nsight-systems (requires free NVIDIA dev account). On this machine we installed **Nsight Systems 2026.2.1** (2025.x also works fine if you prefer an earlier release). Default install. Gives you `nsys.exe` (CLI) and `nsys-ui.exe` (GUI).

Install path on this machine: `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.2.1\`
- CLI binary: `target-windows-x64\nsys.exe`
- GUI binary: `host-windows-x64\nsys-ui.exe`

Adding to PATH is **optional** — [`profiling/run_nsight.sh`](../profiling/run_nsight.sh) resolves `nsys` via `command -v` first and falls back to the absolute path above, so scripts work either way. If you prefer PATH for interactive use: `setx PATH "%PATH%;C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.2.1\target-windows-x64"` from a fresh PowerShell, then open a new shell.

Sanity check: `"/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.2.1/target-windows-x64/nsys.exe" --version` should print `NVIDIA Nsight Systems version 2026.2.1.210-...`.

Nsight 2026.x **case-sensitivity gotcha** surfaced in Phase 5: `-t cuDNN` (capital D, capital NN) is required; lowercase `cudnn` errors out with "Illegal --trace argument". Similarly the second invocation of `nsys stats` on the same `.nsys-rep` needs `--force-export=true`. Both are already baked into `run_nsight.sh`.

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

PyTorch Profiler tells you what PyTorch knows about. Nsight Systems gives you the system-level timeline: CUDA API calls, kernel executions, memory transfers, CPU threads, all on one scrollable timeline. It's invaluable for spotting gaps (where the GPU is idle waiting for CPU) and launch overhead (tiny kernels stacking up). For this project we use it mainly for the timeline visualisation and for Phase 4–5 investigation when PyTorch Profiler alone isn't giving enough detail.

> **Confirmed in Phase 5.** Nsight surfaced three findings that were invisible in the chrome-trace: (1) MobileNetV3's 5.3 s cuDNN algorithm-search warmup, (2) Winograd kernels probed-but-not-picked on ResNet-18 under short warmup, (3) idle gaps between iterations on GRU. All three required the API-row + host-row view that Nsight provides and PyTorch Profiler collapses. See writeup §5.6.

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

This is the planned layout. **Asterisks (✱) mark files that exist as of Phase 5; the rest are Phase-6+ work.** The canonical repo-layout snapshot with current file descriptions is in [`README.md`](../README.md) — this section keeps the original plan for historical reference.

```
HDAI_Project/
├── README.md                                 ✱
├── requirements.txt                          ✱
├── .gitignore                                ✱  (Nsight .sqlite ignored; results/ otherwise committed)
├── docs/
│   ├── brief.md                              ✱  (this file)
│   ├── execution_log_0.md                    ✱  (env bootstrap)
│   ├── execution_log_1.md                    ✱  (Phase 2 first pass, superseded)
│   ├── execution_log_2.md                    ✱  (Phase 2 rework)
│   ├── execution_log_3.md                    ✱  (Phase 3: three-model port)
│   ├── execution_log_4.md                    ✱  (Phase 4: classifier + CSV)
│   └── execution_log_5.md                    ✱  (Phase 5: Nsight + NVTX + cross-check)
├── env/
│   ├── check_env.py                          ✱
│   └── sanity_conv.py                        ✱
├── models/
│   ├── __init__.py                           ✱
│   ├── resnet.py                             ✱
│   ├── mobilenet.py                          ✱
│   ├── distilbert.py                         ✱
│   └── gru.py                                ✱
├── profiling/                                    (NOT `profile/` — stdlib collision; see §11)
│   ├── __init__.py                           ✱
│   ├── run_baseline.py                       ✱  (multi-trial; NVTX-instrumented in Phase 5)
│   ├── run_nsight.sh                         ✱  (Phase-5 Nsight capture driver, all four models)
│   ├── run_benchmark_toggle.py                  (Phase 6 — not yet written)
│   ├── run_amp.py                               (Phase 7)
│   ├── run_batch_sweep.py                       (Phase 8)
│   ├── run_channels_last.py                     (Phase 12 optional)
│   └── run_seq_sweep.py                         (Phase 12 optional)
├── analysis/
│   ├── __init__.py                           ✱
│   ├── parse_trace.py                        ✱  (chrome-trace → per-kernel table)
│   ├── classify_kernels.py                   ✱  (16 buckets after Phase 4)
│   ├── plots.py                              ✱  (8 PNGs: 4 per-model + 3 cross-model + 1 algo)
│   ├── compute_summary.py                    ✱  (Phase-4 emitter of baseline_breakdown.csv)
│   ├── cross_check_nsight.py                 ✱  (Phase-5 regression: PyTorch Profiler vs Nsight)
│   └── compute_roofline.py                      (Phase 9 — not yet written)
├── results/
│   ├── traces/                               ✱  chrome-trace JSONs, 4 files committed
│   ├── nsys/                                 ✱  Phase-5 .nsys-rep (4 files) + run logs
│   │   └── stats/                            ✱  kern_sum + api_sum CSVs per model (8 files)
│   ├── tables/                               ✱  baseline_breakdown.csv (Phase 4)
│   └── plots/                                ✱  16 PNGs: 8 analysis + 8 nsight_* screenshots
└── writeup/
    └── final_report.md                       ✱  §§1–5.6 populated (Phases 1–5)
```

**Decision on committing artefacts:** unlike the original "keep results out of git" stance, we deliberately **commit** chrome traces, Nsight reports, PNGs, and CSVs under `results/`. The tradeoff: repo size grows (~25 MB currently) but the writeup's numbers reproduce without anyone having to re-run the profiler. Only regenerable Nsight SQLite side-files (`results/nsys/*.sqlite`) are gitignored. See the commented block in [`.gitignore`](../.gitignore) lines 20–25 for the rationale. This reverses the "small repo, regenerate" advice of the pre-Phase-2 draft; turns out chrome traces on a 12 GB card at batch 32 come out at ~3 MB apiece (not hundreds of MB as originally feared), and Nsight short-capture `.nsys-rep` at `--trials 2 --iters-per-trial 20 --warmup 10` stays under 1 MB each.

### 5.1 requirements.txt

Original draft:

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

**What we actually pinned (Phase 1, updated through Phase 5):** see the committed [`requirements.txt`](../requirements.txt). We deliberately *did* pin exact versions because the cu128 Blackwell path is fragile and we wanted the same numbers to reproduce months later. Notable adds beyond the original stub: `fvcore` / `ptflops` (FLOP counting, staged for Phase 9 roofline), `pillow` / `tokenizers` / `safetensors` / `huggingface-hub` (transitive from DistilBERT), `nvtx==0.2.15` (pinned through Phase 0, finally *exercised* in Phase 5's `run_baseline.py` NVTX ranges). Torch itself is installed from `pytorch.org/whl/cu128` — not PyPI — because the default PyPI wheel does not carry sm_120 kernels and fails silently on Blackwell.

---

## 6. Phase-by-phase execution plan

### Phase 0 (the pre-work): 30–45 minutes

Read the Reference Code section (section 4) above. Don't skip this. Open the PyTorch profiler recipe in one tab, the Slimakanzer cudnn-benchmark source in another, and skim both.

### Phase 1: Environment setup

Follow section 2 end to end. By the end of this phase:
- `nvidia-smi` works ✓ (Phase 1)
- PyTorch with cu128 wheel installed ✓ (Phase 1, torch 2.10.0+cu128)
- `env/check_env.py` prints GPU name, cuDNN version, successful matmul ✓ (Phase 1)
- Nsight Systems installed and `nsys --version` works ✓ (Phase 5, version 2026.2.1.210; install was delayed from Phase 1 to Phase 5 — everything else could proceed without it)

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

Run it. Look at the table output. (Filename after the Phase-2 rework is actually `results/traces/resnet18_baseline_bs32_benchOn.json` — encoded batch + benchmark state to allow multiple configurations to coexist.) Open in `chrome://tracing` (or use Perfetto UI at https://ui.perfetto.dev/).

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

Produce a per-model summary table.

**Actually produced (Phase 4, committed in [`results/tables/baseline_breakdown.csv`](../results/tables/baseline_breakdown.csv)):** 25 columns — model, batch, latency mean/std, throughput, total CUDA ms, event count, 17 per-category percentage columns, and a cross-cutting `tc_total_pct` column. Condensed rendering:

| Model | Batch | Lat (ms) | Thru (samp/s) | Conv % | Matmul % | Norm % | Elem % | Other % | TC % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | 32 | 11.71 ± 0.61 | 2,733 | 69.9 | 0.1 | 8.5 | 8.7 | 12.8 | 58.5 |
| MobileNetV3-Small | 32 | 3.01 ± 0.18 | 10,644 | 37.6 | 21.1 | 21.8 | 15.5 | 4.0 | 14.9 |
| DistilBERT-base | 8 | 12.36 ± 0.44 | 648 | 0.0 | 96.5 | 1.5 | 1.9 | 0.1 | 0.0 |
| Tiny GRU | 32 | 0.25 ± 0.01 | 127,003 | 0.0 | 18.5 | 0.0 | 1.5 | 80.0 | 16.8 |

All rows sum to 100 ± 0.01 pp (classifier coverage is 100 % after the Phase-4 `conv_depthwise` / `fused_attention` / `embed_gather` rule additions — `other_pct = 0.00` for every model). See `execution_log_4.md` for the classifier-bug audit that surfaced those three new buckets.

By end of Phase 4: master table + regenerator script ([`analysis/compute_summary.py`](../analysis/compute_summary.py)). Centerpiece figure of the whole project.

### Phase 5: Nsight Systems timeline view

PyTorch Profiler gives you tables. Nsight Systems gives you a picture. Profile each model through `nsys`.

> **Phase 5 executed (2026-04-18, see `execution_log_5.md`):**
> - Capture driver: [`profiling/run_nsight.sh`](../profiling/run_nsight.sh) — captures all four models in one pass with shortened parameters (`--trials 2 --iters-per-trial 20 --warmup 10`) to keep each `.nsys-rep` under 1 MB.
> - Trace selection: `-t cuda,cuDNN,cublas,nvtx -s none --cuda-memory-usage=true`. Note case-sensitive `cuDNN` (Nsight 2026.x rejects lowercase `cudnn`).
> - NVTX instrumentation: [`profiling/run_baseline.py`](../profiling/run_baseline.py) gained `_print_flags()` + four NVTX `annotate` wrappers (`warmup` / `cuda_event_timing` / `profiler_capture` + per-iter `iter_NN`). Zero overhead when not under `nsys`.
> - Screenshots: 8 PNGs (overview + one-inference per model) at [`results/plots/nsight_*.png`](../results/plots/).
> - Cross-check: [`analysis/cross_check_nsight.py`](../analysis/cross_check_nsight.py) agrees with the PyTorch Profiler per-iter numbers within 17.1 % worst-case (ResNet-18 and GRU within 2.1 %).
> - Writeup: full §5.6 "Timeline view via Nsight Systems" with figures 5.6.1a–5.6.4b and sub-findings §6.1–6.5.

Original plan for reference:

```bash
nsys profile -t cuda,cuDNN,cublas,nvtx -o results/nsys/resnet18 python -m profiling.run_baseline --model resnet18
```

Open `results/nsys/resnet18.nsys-rep` in the Nsight Systems GUI. Look at:
- The timeline of kernels (a row per stream)
- The CUDA API row (cuDNN calls are visible here)
- Any gaps in the timeline — those are your serialization bottlenecks

Take a screenshot of the timeline for one full inference. Include in writeup.

This phase is mostly about getting comfortable with the Nsight GUI, which has a learning curve. Don't try to become a Nsight expert; just generate 1–2 traces, look at them, and move on. *(In practice, all four were captured — 4× the plan's scope — because the incremental cost was near zero once the first one worked.)*

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

Structure for `writeup/final_report.md`:

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

**As of Phase 5 (committed), the `results/` directory contains:**

**Chrome traces (committed, PyTorch Profiler, 2.9–5.3 MB each — far smaller than the original estimate):**
- `resnet18_baseline.json` (legacy first-pass, Phase 2)
- `resnet18_baseline_bs32_benchOn.json`
- `mobilenetv3_baseline_bs32_benchOn.json`
- `distilbert_baseline_bs8_benchOn.json`
- `gru_baseline_bs32_benchOn.json`
- FP16 variants, batch-sweep variants, etc. — *pending (Phases 7/8)*

**Nsight Systems reports (committed, 181 KB – 992 KB each — far smaller than the original estimate):**
- `resnet18.nsys-rep` (640 KB)
- `mobilenetv3.nsys-rep` (992 KB)
- `distilbert.nsys-rep` (367 KB)
- `gru.nsys-rep` (181 KB)

Plus stats CSVs (`results/nsys/stats/{model}_{kern,api}_sum_*.csv` — 8 files) regenerable from the `.nsys-rep` via `nsys stats`. The `results/nsys/*.sqlite` side-files are Nsight export artefacts and are gitignored.

**Tables (CSV) — committed so far:**
- `baseline_breakdown.csv` — per-model kernel category % (Phase 4 centerpiece; 25 columns × 4 rows)

**Pending (Phases 6–8):**
- `benchmark_toggle.csv` — cudnn.benchmark on vs off timings
- `amp_comparison.csv` — FP32 vs FP16 timings
- `batch_sweep.csv` — throughput vs batch size
- `top_kernels_per_model.csv` — top 10 kernels per model × dtype

**Plots (PNG, matplotlib) — 16 committed:**

*Phase 3/4 analysis PNGs (produced by `analysis/plots.py`):*
- `resnet18_kernel_breakdown.png` / `mobilenetv3_kernel_breakdown.png` / `distilbert_kernel_breakdown.png` / `gru_kernel_breakdown.png`
- `resnet18_conv_algorithms.png`
- `cross_model_category_stacked.png` / `cross_model_latency_throughput.png` / `cross_model_tc_share.png`

*Phase 5 Nsight screenshots (manual from `nsys-ui.exe`):*
- `nsight_{model}_overview.png` / `nsight_{model}_one_inference.png` for all four models

**Pending (Phases 7/8/9/10):**
- `fp16_speedup.png` (grouped bar)
- `batch_scaling.png` (line chart)
- `roofline.png` (log-log)

(The original plan named these `fig1_..._png` through `fig6_..._png`; actual filenames are descriptive rather than numbered.)

**Writeup:**
- `final_report.md` — §§1–5.6 populated, §§6–9 roofline / cross-model / threats / conclusion scaffolded with Phase-3/4/5 findings. Roofline pass awaits Phase 9.
- Embedded plots (16 PNGs referenced so far).

---

## 10. Writeup template

Here's the structure to follow for `writeup/final_report.md`. Aim for 8–12 pages.

```markdown
# Profiling cuDNN across Four Deep Learning Models

## Summary

We profiled four models — ResNet-18, MobileNetV3-Small, DistilBERT-base, and
a tiny GRU — on an RTX 5070 Ti Laptop GPU (Blackwell) using PyTorch Profiler and
Nsight Systems. The models span the compute-bound/memory-bound spectrum on
both conv and matmul axes. Key findings: [3 sentences summarizing the main
story].

## Hardware and software

RTX 5070 Ti Laptop GPU (Blackwell, sm_120), CUDA 12.8 (cu128 PyTorch wheel; no separate Toolkit), cuDNN 9.10.2, PyTorch 2.10.0+cu128, Windows 11. Profiling with `torch.profiler` and Nsight Systems 2026.2.1. All measurements are inference-only, batch 32 except DistilBERT (batch 8 for 12 GB VRAM), with warmup.

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

On Blackwell (`sm_120`) + cuDNN 9.10.2, the benchmark search picks TF32 Tensor-Core implicit-GEMM (`cutlass_tensorop_s1688fprop_optimized_tf32_...`, `sm80_xmma_fprop_implicit_gemm_tf32f32_...`) over Winograd *in steady state*. The routing is via `torch.backends.cudnn.allow_tf32 = True` (default) — on PyTorch 2.10.0+cu128 the parallel `torch.backends.cuda.matmul.allow_tf32` flag actually defaults to `False`, but the cuDNN conv path does its own TF32 decision and engages Tensor Cores regardless. Winograd's ~2.25× multiplication reduction is smaller than the TF32-on-TC throughput gap; cuDNN correctly measures TC-GEMM as faster.

**Phase 5 refinement.** Winograd is *not absent* on this hardware — it's just not picked at convergence. The Nsight short-warmup (10-iter) capture for ResNet-18 contains `cudnn::winograd_nonfused::winogradForwardFilter4x4` at 3.0 % of GPU time, meaning cuDNN *did* probe Winograd during algorithm search. With the full 30-iter warmup of the production baseline, cuDNN has converged on TF32 TC and dropped Winograd from the steady-state kernel mix.

**If you want Winograd to *win* in steady state:** disable TF32 on the cuDNN path before running the profile.

```python
torch.backends.cudnn.allow_tf32 = False   # this is the flag that actually matters on PyTorch 2.10+cu128
torch.backends.cuda.matmul.allow_tf32 = False  # already False by default on this wheel; harmless to set
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

## 14. Appendix C — RTX 5070 Ti Laptop GPU (Blackwell) specs (quick reference)

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

Uses `bash` (Git Bash) conventions where relevant. On PowerShell the `for m in ...; do ... done` lines become `foreach ($m in @("resnet18", ...)) { ... }`; everything else is identical.

```bash
# Environment check
python env/check_env.py

# Run baseline on all models (--benchmark defaults True; use --no-benchmark for the control run)
python -m profiling.run_baseline --model resnet18
python -m profiling.run_baseline --model mobilenetv3
python -m profiling.run_baseline --model distilbert
python -m profiling.run_baseline --model gru

# Or loop (bash)
for m in resnet18 mobilenetv3 distilbert gru; do python -m profiling.run_baseline --model $m; done

# cudnn.benchmark toggle experiment — Phase 6, script not yet written
# python -m profiling.run_benchmark_toggle --model resnet18

# FP32 vs FP16 — Phase 7, script not yet written
# python -m profiling.run_amp --model resnet18

# Batch sweep — Phase 8, script not yet written
# python -m profiling.run_batch_sweep --model resnet18 --batches 1,4,16,64,256

# Nsight capture — single model (note capital cuDNN; lowercase rejected by Nsight 2026.x)
nsys profile -t cuda,cuDNN,cublas,nvtx -o results/nsys/resnet18 python -m profiling.run_baseline --model resnet18

# Nsight capture — all four models + stats CSVs in one shot (Phase 5)
bash profiling/run_nsight.sh

# Analysis
python -m analysis.parse_trace results/traces/resnet18_baseline_bs32_benchOn.json   # top-N kernel table
python -m analysis.plots                       # regenerate all 8 analysis PNGs
python -m analysis.compute_summary             # Phase-4 cross-model CSV
python -m analysis.cross_check_nsight          # Phase-5 PyTorch-Profiler vs Nsight regression
```

Note: `analysis/classify_kernels.py` is a library module, not a CLI — import its `classify()` / `aggregate_by_category()` / `CATEGORY_ORDER` from other scripts. A full overnight-runner shell wrapper is sketched in §25 but not currently committed; Phase 5 shipped `profiling/run_nsight.sh` as the only driver script to date.

---

## 16. What a good final deliverable looks like

If a TA or a PhD student glances at your repo, what should they see?

**In two minutes, they should:**
- Open `writeup/final_report.md` and understand the methodology from the first paragraph
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

**This is the early (pre-Phase-2-rework) starter template.** The committed file at [`profiling/run_baseline.py`](../profiling/run_baseline.py) has diverged substantially since and now includes: multi-trial CUDA-event timing (7 trials × 50 iters in a `time_trials` helper — Phase 2 rework), `BooleanOptionalAction` for `--benchmark`, the `_print_flags()` flag-state dumper (Phase 5), and four NVTX `annotate` context managers wrapping the warmup / timing / profiler phases (Phase 5). Refer to the committed file for the current canonical source. The template below is kept for historical context only.

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

**Early template.** The committed [`analysis/classify_kernels.py`](../analysis/classify_kernels.py) now has 16 category buckets (including `conv_depthwise`, `fused_attention`, `embed_gather` added in Phase 4), explicit keyword-precedence ordering, and separates the classifier library from any CLI — CSV emission moved to [`analysis/compute_summary.py`](../analysis/compute_summary.py). Use the real files; this stub only shows the idea.

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

**Template for Phase 7; not yet implemented.** The `from profile.run_baseline` import below is a pre-rework artifact — the actual package is `profiling.run_baseline` (renamed in Phase 2 rework to dodge a stdlib collision; see §11). When we write `profiling/run_amp.py`, the real import will be `from profiling.run_baseline import load_model_and_input`. Also: single-window timing is acceptable for a quick A/B but the Phase 6+ scripts should reuse `profiling.run_baseline.time_trials` to stay consistent with the baseline methodology (7×50 trials, multi-trial mean ± std).

```python
"""FP32 vs FP16 timing. Outputs a single row of CSV per run."""
import argparse, csv, os, torch
from profile.run_baseline import load_model_and_input  # historical typo; real import is profiling.run_baseline

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

**Revised decision (Phase 2 rework onward):** we commit the whole `results/` tree except Nsight's regenerable SQLite side-files. The committed [`.gitignore`](../.gitignore) implements this; the empirical basis for reversing the original "don't commit traces" stance is that chrome traces at batch 32 come in at ~3 MB each and short-capture `.nsys-rep` at `--trials 2 --iters-per-trial 20 --warmup 10` stay under 1 MB each — far from the "100 MB+" worst-case this section originally feared. The total committed `results/` footprint at end of Phase 5 is ~25 MB, which is a reasonable price to pay for the writeup reproducing without anyone re-running the profiler.

Current `.gitignore` (abbreviated):

```
# Python / IDE / OS — standard
__pycache__/
*.pyc
.venv/  venv/  .env
.ipynb_checkpoints/
.DS_Store  Thumbs.db
.idea/  .vscode/

# Nsight side-files (regenerable from .nsys-rep via nsys stats)
results/nsys/*.sqlite

# results/traces/*.json, results/nsys/*.nsys-rep, results/plots/*.png,
# results/tables/*.csv, results/nsys/stats/*.csv are deliberately COMMITTED
# so the writeup reproduces without re-running the profiler.
```

**Do commit (current practice):**
- All `.py` scripts in `env/`, `models/`, `profiling/`, `analysis/`
- `profiling/run_nsight.sh` (the only shell script)
- `requirements.txt`
- `README.md`, `docs/*.md`, `writeup/final_report.md`
- `results/tables/*.csv`
- `results/plots/*.png` (16 of them as of Phase 5)
- `results/traces/*.json` (chrome traces, ~3 MB each)
- `results/nsys/*.nsys-rep` (Nsight reports, <1 MB each at short-capture params)
- `results/nsys/stats/*.csv` (`nsys stats` outputs)

**Don't commit:**
- `results/nsys/*.sqlite` (regenerable, large relative to .nsys-rep)
- Downloaded pretrained weights (they land in the HF / torchvision cache dirs)
- Virtual envs

**Good README structure:** the committed [`../README.md`](../README.md) is the canonical one; it's expanded well past the stub below as the study has progressed through Phases 1–5. The stub is preserved for historical context.

```markdown
# cuDNN Profiling on Blackwell

Profiling four models (ResNet-18, MobileNetV3-Small, DistilBERT, tiny GRU)
on RTX 5070 Ti Laptop GPU using PyTorch Profiler and Nsight Systems.

## Setup
1. `pip install -r requirements.txt`
2. `python env/check_env.py` — should print sm_120 and pass smoke tests

## Reproduce (Phase 5 state)
```
for m in resnet18 mobilenetv3 distilbert gru; do python -m profiling.run_baseline --model $m; done
python -m analysis.plots
python -m analysis.compute_summary
bash profiling/run_nsight.sh                  # Phase 5
python -m analysis.cross_check_nsight         # Phase 5 regression
```
See `writeup/final_report.md` for results.

## Hardware
RTX 5070 Ti Laptop GPU, cu128 PyTorch wheel, cuDNN 9.10.2, Nsight Systems 2026.2.1.
```

---

## 25. Overnight batch runner

Once your scripts work, you don't need to babysit each experiment. Here's the plan for a `scripts/run_all.ps1` (PowerShell) or `scripts/run_all.sh` (bash) that runs the whole study end to end. Kick it off before dinner, come back to finished traces.

> **Current state:** not yet written. The only committed runner script as of Phase 5 is [`profiling/run_nsight.sh`](../profiling/run_nsight.sh) which batches the four Nsight captures. The full overnight runner becomes worthwhile once Phase 6–8 scripts land (benchmark-toggle, AMP, batch-sweep) — it would wrap them with the `Start-Sleep -Seconds 5` thermal pause below.

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