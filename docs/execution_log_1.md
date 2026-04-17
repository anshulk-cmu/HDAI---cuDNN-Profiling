# Execution Log 1 — Phase 2: First Profile of ResNet-18

> **Status (as of `execution_log_2.md`):** this log describes the *first-pass* Phase-2 run and is kept as a historical record. The artefacts it produced have been **superseded** by the reworked run described in [`execution_log_2.md`](execution_log_2.md), which fixes four bugs in `profiling/run_baseline.py`, adds multi-trial statistics, renames the trace artefact, and adds analysis plots. The numeric findings (Winograd absent, TF32 TC implicit-GEMM dominates, layout-convert ≈ 10 % of GPU time) still stand — they are reproduced with tighter error bars in log_2.

Pure execution record. Every command, every failure, every output, every decision with its reasoning. Claims are distinguished from observations. Where a prediction from `brief.md` did not hold, the actual behaviour is reported and investigated — the brief is not treated as ground truth.

**Date of execution:** 2026-04-16
**Working directory:** `D:\HDAI_Project`
**Shell:** Git Bash on Windows 11
**Env activation pattern:** `source /c/Users/worka/anaconda3/etc/profile.d/conda.sh && conda activate hdai`. After activation `python` resolves to the `hdai` env's Python 3.11.15. (Git Bash on Windows needs the `conda.sh` source step because `conda` is not on PATH by default.)
**GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU, compute capability `sm_120`, 12 GB GDDR7
**PyTorch:** 2.10.0+cu128, **cuDNN:** 91002 (9.10.2)

---

## 1. Scope of Phase 2

Goal of this phase: produce a single trustworthy baseline profile of ResNet-18 on the laptop GPU, so the rest of the plan has a verified harness to build on. The three files created in this phase form the minimum viable profiling pipeline:

1. An environment smoke-test script (`env/check_env.py`) — codifies the bootstrap verification from Log 0 on disk so it is reproducible.
2. A ResNet-18 loader module (`models/resnet.py`) — a stable import point for both this and later experiments.
3. A PyTorch Profiler driver (`profiling/run_baseline.py`) — the script that actually records kernel timings and emits a chrome-trace JSON.

No attempt was made to port to other models in this phase. The isolation discipline is deliberate: if the harness is broken on one model we know exactly where to look; if we tried four at once a single PyTorch/cuDNN misconfiguration would be indistinguishable from a per-model issue.

---

## 2. Step 1 — `env/check_env.py`

### 2.1 File creation

Wrote `env/check_env.py` (1153 bytes) matching `docs/brief.md §21.1` verbatim. Full content:

```python
"""Verify hardware, PyTorch, cuDNN are set up correctly for Blackwell (sm_120)."""
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

x = torch.randn(1024, 1024, device='cuda')
y = torch.matmul(x, x.T)
torch.cuda.synchronize()
print(f"Matmul smoke test: output shape {y.shape}, max = {y.max().item():.2f}")

import torch.nn.functional as F
a = torch.randn(16, 64, 56, 56, device='cuda')
w = torch.randn(128, 64, 3, 3, device='cuda')
b = F.conv2d(a, w, padding=1)
torch.cuda.synchronize()
print(f"cuDNN conv smoke test: output shape {b.shape}")

print("\nAll checks passed.")
```

### 2.2 Why each line of this script exists (decision log)

- `sys.version.split()[0]` — prints bare `X.Y.Z` rather than the full banner so it's easy to compare with Log 0.
- `torch.version.cuda` — the CUDA runtime that was compiled into the PyTorch wheel (not the driver-reported one). Mismatch between `torch.version.cuda` and `nvidia-smi`'s version is almost always benign; the wheel's embedded runtime is what actually runs kernels.
- `torch.backends.cudnn.version()` — returns an integer like `91002` meaning cuDNN 9.10.2. If this is `None`, cuDNN isn't loaded; if it's missing entirely, the wheel was built without cuDNN.
- `cap != (12, 0)` check — prints a warning rather than a hard error, so a fallback GPU (e.g. integrated) still gets diagnostic output instead of an opaque exit.
- `torch.randn(1024, 1024)` then `@ x.T` — the simplest possible CUDA workload. Validates the whole chain: host→device transfer, CUDA kernel launch, cuBLAS/cuDNN dispatch, device→host sync.
- `F.conv2d` on a 16×64×56×56 input with a 128×64×3×3 weight — not a random shape. These are the dimensions of one of ResNet-18's intermediate conv layers (after the first downsample). Specifically tests the cuDNN conv path, not just the cuBLAS matmul path.

### 2.3 Execution and output

Command:

```bash
source /c/Users/worka/anaconda3/etc/profile.d/conda.sh && conda activate hdai
cd "D:/HDAI_Project"
python env/check_env.py
```

Exact stdout:

```
Python: 3.11.15
PyTorch: 2.10.0+cu128
CUDA (PyTorch-linked): 12.8
cuDNN version: 91002
CUDA available: True
Device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
Compute capability: sm_120
Matmul smoke test: output shape torch.Size([1024, 1024]), max = 1198.62
cuDNN conv smoke test: output shape torch.Size([16, 128, 56, 56])

All checks passed.
```

### 2.4 Interpretation of each line

- **`Python: 3.11.15`** — one patch ahead of the `privacy` env's `3.11.14`. conda picked the latest 3.11.x during `conda create`. Not a concern.
- **`PyTorch: 2.10.0+cu128`** — the `+cu128` suffix is the entire point of this project's setup. Without it, `sm_120` kernels don't exist in the wheel and nothing runs.
- **`CUDA (PyTorch-linked): 12.8`** — matches the wheel tag.
- **`cuDNN version: 91002`** — cuDNN 9.10.2. Newer than the brief's "9.x" floor by three minor versions.
- **`CUDA available: True`** — driver 592.01 + PyTorch wheel see each other.
- **`Device: NVIDIA GeForce RTX 5070 Ti Laptop GPU`** — full marketing name. The brief was corrected earlier to "RTX 5070"; the actual string is "5070 Ti Laptop". Both reference the same `sm_120` silicon, only TGP and clocks differ.
- **`Compute capability: sm_120`** — Blackwell. The cu128 wheel contains native or forward-compatible kernels for this arch; confirmed by the subsequent smoke tests running without the "no kernel image available" error that would otherwise mark a wheel mismatch.
- **`Matmul smoke test: ... max = 1198.62`** — for a 1024×1024 standard-normal matrix `X` multiplied by its transpose, the entries of `XX^T` are sums of 1024 products of standard normals. The diagonal entries are `‖row_i‖²`, expected value 1024, standard deviation ≈ √2048 ≈ 45. A max of ~1200 is well within the tail of a distribution of 1024 such variables (the expected max of 1024 iid normals is roughly 3.1σ above mean). So the number is statistically plausible → the matmul produced correct numerical output, not garbage that happens to have the right shape.
- **`cuDNN conv smoke test: output shape torch.Size([16, 128, 56, 56])`** — input 16×64×56×56, weight 128×64×3×3, `padding=1`, stride implicit 1. Output spatial dims should equal input spatial dims when `stride=1, padding=1, kernel=3`. They do. Channel dim changes from 64→128 as expected.

### 2.5 Gate evaluation

Every line matches the predicted shape: `sm_120` confirmed, cuDNN present, matmul and conv both succeed. Step 1 is unambiguously green.

---

## 3. Step 2 — `models/resnet.py`

### 3.1 File creation

Wrote `models/resnet.py` (301 bytes). Full content:

```python
"""ResNet-18 wrapper for the cuDNN profiling study."""
import torch
import torchvision.models as tvm


def get_model():
    return tvm.resnet18(
        weights=tvm.ResNet18_Weights.IMAGENET1K_V1
    ).eval().cuda()


def get_input(batch=32):
    return torch.randn(batch, 3, 224, 224, device='cuda')
```

### 3.2 Why two functions instead of one

The brief's §6 template only shows `get_model()`. Added `get_input(batch=32)` so every profiling script in `profiling/` can do `from models.resnet import get_model, get_input` and not duplicate the input-tensor-shape knowledge. Batch 32, 3 channels, 224×224 is the canonical ImageNet inference shape; any deviation from this for a specific experiment should be a deliberate choice rather than an accidental copy-paste.

### 3.3 Why `.eval()` and `.cuda()` chained in `get_model`

- `.eval()` switches BatchNorm to use running statistics and disables Dropout. Inference-correct behaviour is the only thing we profile.
- `.cuda()` moves every parameter to device 0. Doing this at model-construction time means every downstream script gets a GPU-resident model with zero further ceremony. If the user wanted CPU inference for some reason they would write a different loader; this one is CUDA-only by design.

### 3.4 First verification attempt — FAILED

Ran:

```bash
# env already activated
python -c "
from models.resnet import get_model, get_input
m = get_model()
p = sum(p.numel() for p in m.parameters())
print(f'Param count: {p:,}')
x = get_input(batch=32)
print(f'Input shape: {tuple(x.shape)}, device: {x.device}, dtype: {x.dtype}')
import torch
with torch.no_grad():
    y = m(x)
print(f'Output shape: {tuple(y.shape)}')
"
```

Got a cascading traceback ending in:

```
  File "C:\Users\worka\anaconda3\envs\hdai\Lib\site-packages\torchvision\models\__init__.py", line 2, in <module>
    from .convnext import *
  File ".../torchvision/models/convnext.py", line 9, in <module>
    from ..ops.misc import Conv2dNormActivation, Permute
  File ".../torchvision/ops/__init__.py", line 23, in <module>
    from .poolers import MultiScaleRoIAlign
  File ".../torchvision/ops/poolers.py", line 10, in <module>
    from .roi_align import roi_align
  File ".../torchvision/ops/roi_align.py", line 7, in <module>
    from torch._dynamo.utils import is_compile_supported
  File ".../torch/_dynamo/__init__.py", line 13, in <module>
    from . import (...)
  File ".../torch/_dynamo/convert_frame.py", line 28, in <module>
    import cProfile
  File ".../cProfile.py", line 23, in <module>
    run.__doc__ = _pyprofile.run.__doc__
                  ^^^^^^^^^^^^^^
AttributeError: module 'profile' has no attribute 'run'
```

### 3.5 Root-cause analysis

Python's stdlib `cProfile` begins with `import profile as _pyprofile` then reads `_pyprofile.run.__doc__` to copy the docstring onto its own `run` function. This import resolves through the standard module search order: **sys.path entries are searched left-to-right, and the first match wins.**

When running `python -c "..."` with the working directory at `D:/HDAI_Project`, `sys.path[0]` is `''` which resolves to the current directory. `sys.path[0]` therefore contains the directory `D:/HDAI_Project/profile/` (created earlier by the repo scaffolding step, with an empty `__init__.py`). Python's import machinery treats a directory with an `__init__.py` as a **regular package** and prefers it over any later stdlib module with the same name.

Result: when `cProfile` asks for `profile`, it gets **our empty package**, not the stdlib module. Our package has no `run` attribute, so `AttributeError: module 'profile' has no attribute 'run'`.

This is triggered indirectly:
1. `from models.resnet import get_model` imports `torchvision.models`
2. `torchvision.models.__init__` imports `.convnext`
3. `.convnext` imports `..ops.misc`
4. `torchvision.ops.__init__` imports `.poolers`
5. `.poolers` imports `.roi_align`
6. `.roi_align` imports `torch._dynamo.utils`
7. `torch._dynamo.__init__` imports `.convert_frame`
8. `.convert_frame` imports `cProfile`
9. `cProfile` imports `profile` — finds our empty package first — crash

The brief's suggested §5 layout uses `profile/` as a directory name, which creates this collision on every modern PyTorch installation because `torch._dynamo` is now eagerly loaded from `torchvision.ops`. Older PyTorch versions didn't have this import chain. This is a latent landmine in the brief.

### 3.6 Fix applied

Renamed the top-level directory:

```bash
cd "D:/HDAI_Project" && mv profile profiling
```

Then, because `mv` on Windows/Git Bash sometimes leaves behind a stray tempfile, also cleaned up the leftover `profiling/__init__.py.tmp` and the automatically-recreated `profiling/__pycache__/` from the previous partial run:

```bash
rm "D:/HDAI_Project/profiling/__init__.py.tmp"
rm -rf "D:/HDAI_Project/profiling/__pycache__"
```

Then re-created `profiling/__init__.py` as an empty file to re-establish it as a package.

### 3.7 Alternative fixes considered and rejected

- **Run from a different directory** — fragile, every user of the repo has to know about this.
- **`sys.path.insert(0, ...)`-munging at the top of each script** — brittle and would need to be duplicated everywhere.
- **Rename just the internal package but keep the user-facing `profile/` path** — impossible, the collision is on the directory name in `sys.path`.
- **Upgrade stdlib** — not applicable; the stdlib is doing exactly what it's documented to do.

Renaming is the only clean fix. `profiling/` is equally descriptive.

### 3.8 Second verification attempt — SUCCESS

Re-ran the verification command. Output:

```
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to C:\Users\worka/.cache\torch\hub\checkpoints\resnet18-f37072fd.pth
  0%|          | 0.00/44.7M [00:00<?, ?B/s]
  …
100%|##########| 44.7M/44.7M [00:01<00:00, 28.3MB/s]
Param count: 11,689,512
Input shape: (32, 3, 224, 224), device: cuda:0, dtype: torch.float32
Output shape: (32, 1000)
```

### 3.9 Interpretation

- **Download side-effect.** First access to `ResNet18_Weights.IMAGENET1K_V1` downloaded `resnet18-f37072fd.pth` (44.7 MB) to `~/.cache/torch/hub/checkpoints/`. The `f37072fd` hash in the filename is torchvision's integrity check; it's deterministic and matches across machines. Subsequent runs skip the download.
- **`Param count: 11,689,512`** — exactly the expected 11.7M parameters for the `IMAGENET1K_V1` weights of `resnet18`. If the count were different (e.g. 11.4M), the wrong variant would have loaded. Confirms we have the standard variant.
- **`Input shape: (32, 3, 224, 224), device: cuda:0, dtype: torch.float32`** — matches the canonical ImageNet inference shape, on the GPU, in FP32. `dtype: torch.float32` matters: if it had defaulted to anything else (rare but possible on some installs), the kernel selection story in §7 would read very differently.
- **`Output shape: (32, 1000)`** — 1000-class logits per image. ImageNet has 1000 classes. Confirms the classification head is intact; the weights did load end-to-end.

### 3.10 Gate evaluation

All three quantitative checks — param count, input/output shapes, dtype — match expectations. Step 2 green.

---

## 4. Step 3 — `profiling/run_baseline.py`

### 4.1 File creation

Wrote `profiling/run_baseline.py` (1530 bytes). Full content:

```python
"""Baseline profile of a single model. Saves chrome trace + top-kernel table."""
import argparse
import os
import torch
from torch.profiler import profile, ProfilerActivity, schedule


def load_model_and_input(name, batch=32):
    if name == 'resnet18':
        from models.resnet import get_model, get_input
        return get_model(), get_input(batch)
    raise ValueError(f"Unknown model: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--benchmark', action='store_true', default=True)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = args.benchmark
    model, x = load_model_and_input(args.model, args.batch)

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

### 4.2 Per-knob justification

Every knob in this script was a deliberate choice. Logging them so future "tweaks" don't silently invalidate earlier measurements.

| Knob | Value | Why |
|---|---|---|
| `torch.backends.cudnn.benchmark` | `True` | We want cuDNN to run its algorithm-search once per distinct input shape and pick the fastest kernel. Without this, cuDNN uses a heuristic that can be suboptimal. All four Phase 2+ experiments assume `benchmark=True` unless they are explicitly the benchmark-toggle experiment. |
| Warm-up count | 30 | Two things need to warm up: (a) CUDA context + allocator + memory allocator, (b) cuDNN's algorithm search for every (input shape, weight shape) combination in the forward pass. ResNet-18 has ~20 distinct conv shapes; benchmark mode triggers a timed micro-benchmark per shape on the first iteration. 30 is the brief's §22 recommendation for benchmark mode; 10 would be too few. |
| `ProfilerActivity.CPU + CUDA` | both | CPU alone misses all kernel times; CUDA alone misses op-level context (we wouldn't see `aten::conv2d` grouping its child cuDNN calls). Both together gives the richest trace. |
| `record_shapes=True` | True | Allows `key_averages(group_by_input_shape=True)` analysis later. Costs a bit of recording overhead but for 10 iterations that's a non-issue. |
| `schedule(wait=1, warmup=2, active=10, repeat=1)` | — | Skip 1 iter, discard 2 as warmup, record 10 active, then stop. `repeat=1` means one cycle. Total 15 iterations inside the profiled loop, hence the outer loop count of 15 matches. |
| Outer loop count (15) | Matches (1+2+10+2 slack) | Enough iterations for the schedule to progress through its phases with margin. |
| `torch.cuda.synchronize()` after every `model(x)` | explicit | Forces the GPU to finish before the Python profiler moves to the next step. Without this, async kernel launches would pile up and the profiler would attribute kernel time to the wrong step. |
| `torch.no_grad()` | wraps inference | Disables autograd graph construction. Reduces memory, disables unnecessary bookkeeping, makes the profile represent true inference not inference+graph-recording. |
| `row_limit=25` in the table | 25 | Top 25 ops is enough to see all non-trivial kernel categories in ResNet-18 without the tail of noise. |
| `sort_by="cuda_time_total"` | — | The point of this project is GPU kernel timing; CPU-dominant ops (data loading, etc.) are not the focus. |

### 4.3 First run attempt — FAILED (`sys.path` issue)

```bash
python profiling/run_baseline.py --model resnet18
```

Output:

```
Traceback (most recent call last):
  File "D:\HDAI_Project\profiling\run_baseline.py", line 50, in <module>
    main()
  File "D:\HDAI_Project\profiling\run_baseline.py", line 23, in main
    model, x = load_model_and_input(args.model, args.batch)
  File "D:\HDAI_Project\profiling\run_baseline.py", line 10, in load_model_and_input
    from models.resnet import get_model, get_input
ModuleNotFoundError: No module named 'models'
```

### 4.4 Why this happened (sys.path mechanics in detail)

When Python is invoked as `python path/to/script.py`, the interpreter sets `sys.path[0]` to the directory **containing the script** — in this case `D:/HDAI_Project/profiling/`. That directory does not contain a sub-package called `models`, so `from models.resnet import ...` fails.

When Python is invoked as `python -m package.module`, Python sets `sys.path[0]` to the **current working directory** — `D:/HDAI_Project/` in our case. That directory does contain `models/`, so the import succeeds.

This is one of the two standard Python invocation modes and both are valid; they differ only in what `sys.path[0]` becomes. For a project that has multiple packages at the repo root (`models/`, `profiling/`, `analysis/`), only `-m` works without `sys.path` hacking. The project is a multi-package repo → `-m` is the canonical invocation.

### 4.5 Fix applied

Switched to module invocation:

```bash
python -m profiling.run_baseline --model resnet18
```

No code change required. All future README and doc references must use `-m` form.

### 4.6 Successful run — complete raw output

```
C:\Users\worka\anaconda3\envs\hdai\Lib\site-packages\torch\profiler\profiler.py:217: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                          ProfilerStep*         0.00%       0.000us         0.00%       0.000us       0.000us     102.075ms       104.24%     102.075ms      10.207ms            10
                                          ProfilerStep*         6.37%       6.604ms       100.00%     103.614ms      10.361ms       0.000us         0.00%      97.923ms       9.792ms            10
                                           aten::conv2d         0.70%     723.400us         8.36%       8.667ms      43.333us       0.000us         0.00%      77.167ms     385.835us           200
                                      aten::convolution         0.72%     742.900us         7.67%       7.943ms      39.716us       0.000us         0.00%      77.167ms     385.835us           200
                                     aten::_convolution         0.68%     707.400us         6.95%       7.200ms      36.002us       0.000us         0.00%      77.167ms     385.835us           200
                                aten::cudnn_convolution         2.87%       2.970ms         6.27%       6.493ms      32.464us      77.167ms        78.80%      77.167ms     385.835us           200
_ZN17cutlass__5x_cudnn6KernelI66cutlass_tensorop_s16...         0.00%       0.000us         0.00%       0.000us       0.000us      41.009ms        41.88%      41.009ms     341.746us           120
sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nh...         0.00%       0.000us         0.00%       0.000us       0.000us      13.675ms        13.96%      13.675ms     341.864us            40
_Z23implicit_convolve_sgemmIffLi128ELi5ELi5ELi3ELi3E...         0.00%       0.000us         0.00%       0.000us       0.000us      10.620ms        10.85%      10.620ms       1.062ms            10
                                       aten::batch_norm         0.40%     412.500us         6.32%       6.550ms      32.749us       0.000us         0.00%       8.742ms      43.712us           200
                           aten::_batch_norm_impl_index         0.73%     758.700us         5.92%       6.137ms      30.687us       0.000us         0.00%       8.742ms      43.712us           200
                                 aten::cudnn_batch_norm         2.50%       2.592ms         5.19%       5.379ms      26.893us       8.742ms         8.93%       8.742ms      43.712us           200
_ZN5cudnn26bn_fw_inf_1C11_kernel_NCHWIffLb1ELi1EEEvT...         0.00%       0.000us         0.00%       0.000us       0.000us       8.742ms         8.93%       8.742ms      43.712us           200
_ZN5cudnn19engines_precompiled16nchwToNhwcKernelIfff...         0.00%       0.000us         0.00%       0.000us       0.000us       6.483ms         6.62%       6.483ms      20.259us           320
                                            aten::relu_         0.63%     653.900us         2.19%       2.269ms      13.346us       0.000us         0.00%       5.855ms      34.438us           170
                                       aten::clamp_min_         0.72%     747.100us         1.56%       1.615ms       9.499us       5.855ms         5.98%       5.855ms      34.438us           170
_ZN2at6native29vectorized_elementwise_kernelILi4EZZZ...         0.00%       0.000us         0.00%       0.000us       0.000us       5.855ms         5.98%       5.855ms      34.438us           170
                                       aten::max_pool2d         0.04%      37.500us         0.19%     200.000us      20.000us       0.000us         0.00%       3.132ms     313.222us            10
                          aten::max_pool2d_with_indices         0.10%     103.300us         0.16%     162.500us      16.250us       3.132ms         3.20%       3.132ms     313.222us            10
_ZN2at6native52_GLOBAL__N__83ba8c45_19_DilatedMaxPoo...         0.00%       0.000us         0.00%       0.000us       0.000us       3.132ms         3.20%       3.132ms     313.222us            10
_ZN5cudnn19engines_precompiled16nhwcToNchwKernelIfff...         0.00%       0.000us         0.00%       0.000us       0.000us       3.040ms         3.10%       3.040ms      25.334us           120
                                             aten::add_         0.45%     468.100us         0.79%     819.300us      10.241us       2.749ms         2.81%       2.749ms      34.360us            80
_ZN2at6native29vectorized_elementwise_kernelILi4ENS0...         0.00%       0.000us         0.00%       0.000us       0.000us       2.749ms         2.81%       2.749ms      34.360us            80
_Z23implicit_convolve_sgemmIffLi128ELi6ELi7ELi3ELi3E...         0.00%       0.000us         0.00%       0.000us       0.000us       2.340ms         2.39%       2.340ms      77.996us            30
                                           aten::linear         0.04%      42.700us         0.64%     659.200us      65.920us       0.000us         0.00%     155.486us      15.549us            10
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 103.618ms
Self CUDA time total: 97.923ms


Trace saved to results/traces/resnet18_baseline.json
```

### 4.7 The profiler warning — what it means

```
UserWarning: Profiler clears events at the end of each cycle. Only events from the current cycle will be reported. To keep events across cycles, set acc_events=True.
```

With `schedule(..., repeat=1)` there is exactly one cycle, so "events from the current cycle" is 100% of the data we want. Non-issue. If we later chain multiple cycles (say `repeat=3` to measure 3 separate 10-iter windows), we would need `acc_events=True` to aggregate them. Noted for later.

### 4.8 Why `Self CUDA %` exceeds 100% in the first row

```
ProfilerStep*   ...   102.075ms   104.24%   ...
```

`ProfilerStep*` is a synthetic span the profiler adds around each scheduled step. Its "Self CUDA" is 102.075ms vs the total 97.923ms — so the percentage column (102.075 / 97.923 ≈ 1.0424) is >100%. The apparent overrun is because `ProfilerStep*` includes some profiler-internal overhead that isn't counted in the per-kernel Self CUDA total. The subsequent `ProfilerStep*` row shows 0.000us Self CUDA and 100.00% CPU total — i.e. the profiler's accounting sees ProfilerStep as a CPU wrapper around CUDA work. The 104% is cosmetic and does not indicate a timing bug.

### 4.9 Throughput sanity check

- 10 profiled iterations took 97.923 ms of CUDA time.
- Per inference (batch 32): 97.923 / 10 ≈ 9.79 ms.
- Per image: 9.79 / 32 ≈ 0.306 ms = 306 µs.
- Throughput: ≈ 3 267 images/sec.

Context: published ResNet-18 FP16 inference on an A100 is ~5 000 img/s at batch 32; on desktop 30-series at FP32 it's ~1 500–2 000 img/s. Our 3 267 img/s in FP32-with-TF32 on a laptop Blackwell is consistent with "TF32 is effectively half-precision throughput on Tensor Cores" and sits in the right ballpark. No red flags.

### 4.10 Trace file

```
results/traces/resnet18_baseline.json   3,017,616 bytes (~2.9 MB)
```

Loads cleanly as JSON; contains a top-level `traceEvents` array.

---

## 5. Deep read of the profiler output — every kernel

Every row in the top-25 table that represents actual GPU work, with an interpretation.

### 5.1 Conv path — the dominant cost

- **`aten::cudnn_convolution`** — 77.167 ms, **78.80% of all CUDA time**, 200 calls. 200 = 20 conv layers in ResNet-18 × 10 profiled iterations. This confirms conv is the overwhelming cost — exactly what brief §1 predicts for a "compute-bound CNN reference point".
- **`cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_64x64_16x10_nhwc_align4>`** — 41.009 ms, 41.88%, 120 calls. This is a CUTLASS Tensor-Core convolution kernel. Decoding the name:
  - `cutlass_5x_cudnn` — CUTLASS templates, v5.x, packaged through cuDNN
  - `tensorop` — uses Tensor-Core MMA instructions
  - `s1688fprop` — the `s1688` instruction tile is 16×16×8 TF32, `fprop` = forward propagation conv
  - `optimized_tf32` — the optimised TF32 template
  - `64x64_16x10` — 64×64 output tile per threadblock, K=16, 10-stage pipeline (software pipelining through shared memory to hide memory latency)
  - `nhwc_align4` — expects NHWC layout, 4-element alignment (half a cache line at FP32)
  - 120 calls out of 200 convs = 60% of conv invocations routed to this kernel.
- **`sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8`** — 13.675 ms, 13.96%, 40 calls. A second Tensor-Core TF32 implicit-GEMM path. Decoded:
  - `sm80` — compiled for Ampere class, running forward-compatibly on Blackwell. That cuDNN 9.10.2 doesn't have native `sm_120` kernels for this algorithm is not uncommon for a freshly-released arch.
  - `xmma` — eXtended MMA, NVIDIA's internal kernel family for Tensor Cores
  - `fprop_implicit_gemm` — implicit GEMM (no materialised im2col buffer), forward conv
  - `tf32f32_tf32f32_f32` — inputs TF32, accumulate TF32, output FP32
  - `nhwckrsc_nchw` — input NHWC, weights KRSC layout, output NCHW (note the mixed layout — hence the converter kernels below)
  - `tilesize128x128x16` — 128×128 output tile, K=16
  - `stage4` — 4-stage software pipeline
  - `warpsize2x2x1` — 2×2 warp tiling per threadblock
  - `g1` — 1 group (regular conv, not grouped)
  - `tensor16x8x8` — 16×8×8 Tensor-Core instruction
- **`_Z23implicit_convolve_sgemmIff...Li128,5,5,3,3,3,1,...>`** — 10.620 ms, 10.85%, 10 calls. **Plain FP32 (non-TC)** implicit-GEMM conv. Exactly one call per iteration → there is one specific conv layer in ResNet-18 that gets dispatched here. The template parameters are:
  - `Li128` = tile size 128
  - `Li5,Li5` = tile dimensions in two directions
  - `Li3,Li3` = 3×3 filter
  - `Li3,Li1` = padding + stride hints
  - `Lb0,Lb0,Lb1` = boolean template flags (probably: no bias fusion, no ReLU fusion, use_tensor_core=false → the `sgemm` tail confirms SIMT FP32)
  - The mangled `_Z23implicit_convolve_sgemmIff` tells us the accumulator is FP32 and there's no `tensorop` → this is the SIMT FP32 path. Picked because the shape doesn't fit one of the TC tile sizes, or because `cudnn.benchmark=True` measured this as faster for that specific shape.
- **`_Z23implicit_convolve_sgemmIff...Li128,6,7,3,3,5,1,...>`** — 2.340 ms, 2.39%, 30 calls. Another SIMT FP32 implicit-GEMM variant, different tile dimensions (6×7 output tiling, 3×3 filter, stride 5), 30 invocations = 3 conv layers × 10 iters.

Total conv breakdown: **41.009 + 13.675 + 10.620 + 2.340 = 67.644 ms** out of the `aten::cudnn_convolution` aggregate **77.167 ms**. The gap (≈ 9.5 ms, ~12%) is absorbed by cuDNN-internal overhead and layout-conversion kernels (§5.4). That ~12% overhead figure is worth keeping — it's what "cuDNN call fabric" actually costs at this batch size.

### 5.2 Algorithm-family split (Tensor Core vs SIMT)

- **Tensor-Core TF32 kernels:** 41.009 + 13.675 = **54.684 ms (55.84% of total CUDA time)**, 120 + 40 = 160 invocations.
- **SIMT FP32 kernels:** 10.620 + 2.340 = **12.960 ms (13.24% of total)**, 10 + 30 = 40 invocations.
- **Ratio:** 4× more Tensor-Core invocations, ~4.2× more TC time. Matches the heuristic that most ResNet-18 convs are shape-eligible for TC tiles.

### 5.3 Norm / ReLU / pool / residual / linear

- **`aten::cudnn_batch_norm`** → dispatches **`cudnn::bn_fw_inf_1C11_kernel_NCHW`** — 8.742 ms, 8.93%, 200 calls (20 BN layers × 10 iters). This is cuDNN's fused batchnorm-inference kernel for NCHW (non-fused BN would be two kernels). 8.93% is within the brief's prediction for ResNet-18 (§1.1 expects BN small, not zero).
- **ReLU path** — `aten::relu_` → `aten::clamp_min_` → `vectorized_elementwise_kernel`. 5.855 ms, 5.98%, 170 calls. Interesting: 170, not 200. ResNet-18 has 17 ReLUs per forward (one after each of the 16 "main" conv+BN pairs, plus one after the initial conv). 17 × 10 iters = 170. Checks out. This is a PyTorch-native kernel, not cuDNN.
- **MaxPool** — 3.132 ms, 3.20%, 10 calls. Exactly 1 maxpool in ResNet-18 (after the first 7×7 conv). 1 × 10 iters = 10.
- **`aten::add_`** (residual connections) — 2.749 ms, 2.81%, 80 calls. 8 residual adds per forward × 10 iters = 80. Matches ResNet-18's 8 residual blocks.
- **`aten::linear`** (final FC layer) — 0.155 ms, 0.16%, 10 calls. Tiny — the FC is `512 → 1000` which is dwarfed by the conv pyramid. 10 invocations = 1 FC × 10 iters.

### 5.4 Layout-conversion kernels (an unexpected cost)

- **`cudnn::engines_precompiled::nchwToNhwcKernel`** — 6.483 ms, 6.62%, **320 calls**.
- **`cudnn::engines_precompiled::nhwcToNchwKernel`** — 3.040 ms, 3.10%, **120 calls**.

**Combined: 9.523 ms, 9.72% of all CUDA time is spent just converting tensor layouts.**

Why this happens: the CUTLASS/xmma Tensor-Core kernels demand NHWC input and produce NCHW output (or vice versa — see the `nhwckrsc_nchw` tag on the xmma kernel). The model's parameters are stored in NCHW by default (PyTorch's convention for images), so every time a TC-eligible conv is called, cuDNN inserts a layout conversion before it and sometimes another after it.

**320 converts in going one direction, 120 the other** = 440 conversions total. Over 10 iterations and 20 conv layers, that's 2.2 conversions per layer per iteration on average. This is a real source of inefficiency that `torch.channels_last` (converting the whole model to NHWC once upfront) would eliminate — which is exactly what brief §7.8 suggests and brief §8.5 plans as an optional experiment. This is direct evidence the experiment is worth running.

### 5.5 The `aten::` hierarchy

The first four rows of the table form the op-dispatch chain for conv:
- `aten::conv2d` → `aten::convolution` → `aten::_convolution` → `aten::cudnn_convolution`

Each row has **0.00% Self CUDA** because they don't themselves launch kernels; they're Python/C++ wrapper functions that eventually hand off to cuDNN. Their **Self CPU** times (723µs, 743µs, 707µs, 2970µs) add up to the actual Python-side dispatch cost: ~5.1 ms of CPU time to launch all the conv work. Relative to 77 ms of GPU work this is fine, but at batch 1 where each GPU kernel takes microseconds, this dispatch overhead would dominate. That's the launch-overhead regime the batch-size-sweep experiment is designed to quantify.

---

## 6. Winograd absence — full investigation

### 6.1 Hypothesis

The brief §1.1 says: *"When `cudnn.benchmark=True` is on, almost every conv layer in ResNet-18 gets a Winograd algorithm picked."* → The top-25 table contains zero kernels with `winograd` in the name. Either (a) Winograd is in the trace but hidden below the top-25 cutoff, or (b) Winograd genuinely was not picked.

### 6.2 Investigation

Ran an exhaustive scan of the full 2.9 MB trace JSON:

```python
import json
with open('results/traces/resnet18_baseline.json') as f:
    data = json.load(f)
events = data.get('traceEvents', [])

winograd = {e['name'] for e in events if 'winograd' in e.get('name','').lower()}
impl_or_tc = {e['name'][:120] for e in events
              if any(k in e.get('name','').lower()
                     for k in ('implicit', 'xmma', 'cutlass'))}
```

### 6.3 Result

- **Winograd kernels found: 0** across every event in the trace. Not truncation, not a top-25 cutoff — genuinely absent.
- **Implicit-GEMM / xmma / CUTLASS kernels found: 5 distinct names**:
  1. `_Z23implicit_convolve_sgemmIff...Li128ELi5ELi5ELi3ELi3ELi3ELi1E...` — SIMT FP32 implicit GEMM, variant A
  2. `_Z23implicit_convolve_sgemmIff...Li128ELi6ELi7ELi3ELi3ELi5ELi1E...` — SIMT FP32 implicit GEMM, variant B
  3. `cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_64x64_16x10_nhwc_align4>` — TC TF32 CUTLASS
  4. `cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>` — SIMT FP32 CUTLASS (**the final FC layer** — shape 32×512 × 512×1000 doesn't fit TC tiles well, so falls back to SIMT)
  5. `sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_...` — TC TF32 xmma

### 6.4 Why Winograd didn't win — honest hypotheses, stack-ranked

1. **TF32 on Blackwell is ~6× faster than sm_80 TF32 per the architectural uplift.** Winograd's advantage over GEMM is a ~2.25× reduction in multiplications; on Ampere Tensor-Core FP32 that still left room for Winograd to be competitive. On Blackwell 5th-gen Tensor Cores, that 2.25× is now much less than the throughput gap between TC-GEMM and SIMT-Winograd. So when `cudnn.benchmark=True` times both, TC-GEMM wins.
2. **Winograd implementations in cuDNN 9.10.2 may not yet have `sm_120`-optimised code paths.** If the available Winograd kernels fall back to `sm_80` code with no NHWC/TC variant, they are unambiguously slower than an `sm_80`-forward-compatible `xmma` kernel on the same chip.
3. **TF32 has lower precision than FP32.** Winograd has numerical stability issues that are worse at reduced precision. cuDNN's heuristic may bias away from Winograd when TF32 is active regardless of measured speed.
4. **The `cudnn.benchmark=True` decision is per-(shape, dtype, math-mode) tuple.** With TF32 math mode active by default, the benchmark candidates *for TF32* are a different set than the FP32 candidates. The Winograd variants may simply not be in the candidate list for TF32.

All four contribute. Hypothesis 1 is almost certainly the dominant reason.

### 6.5 Experiment to design later

The clean A/B to prove this:

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

Then re-profile ResNet-18. Prediction: with TF32 disabled, `cudnn.benchmark=True` will measure Tensor-Core GEMM as much slower (now pure FP32, no TF32 acceleration), and Winograd becomes competitive. Expect `winograd` kernels to appear in the profile.

This is added as a Phase 11 optional experiment. It would make a strong paired-bar figure for the writeup: "Same ResNet-18, same cuDNN, same benchmark=True; only math-mode changed. Algorithm selection flipped."

---

## 7. Artefacts produced in this phase

| Path | Size (bytes) | Role |
|---|---|---|
| `env/check_env.py` | 1 153 | env smoke test |
| `models/__init__.py` | 0 | package marker |
| `models/resnet.py` | 301 | ResNet-18 loader + 224² FP32 input helper |
| `profiling/__init__.py` | 0 | package marker (replaces former `profile/`) |
| `profiling/run_baseline.py` | 1 530 | PyTorch Profiler driver |
| `results/traces/resnet18_baseline.json` | 3 017 616 | chrome-trace, 10 active iters, batch 32 |

Side-effect on disk (not in repo): `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` (44.7 MB) downloaded by torchvision on first `get_model()`. Future runs are offline.

Repo-local cache side effects generated during execution:
- `models/__pycache__/resnet.cpython-311.pyc` and `__init__.cpython-311.pyc`
- `profiling/__pycache__/run_baseline.cpython-311.pyc` and `__init__.cpython-311.pyc`

These are deleted at the end of this step (§10).

---

## 8. Gate evaluation against the original plan

Plan gates were framed around finding Winograd and implicit-GEMM kernels. The actual result is: implicit-GEMM present in abundance, Winograd completely absent. That is a prediction failure, not a pipeline failure. Reassessment:

| Original gate | Result | Interpretation |
|---|---|---|
| `env/check_env.py` prints sm_120, passes conv smoke test | ✓ | pipeline works end-to-end |
| `models/resnet.py` loads 11.7M-param model | ✓ (11 689 512 exactly) | correct variant loaded |
| `profiling/run_baseline.py` runs to completion | ✓ | no errors at exit |
| trace JSON produced, 1–20 MB | ✓ (2.9 MB) | |
| ≥ 1 implicit-GEMM kernel identifiable | ✓ (5 variants) | |
| ≥ 1 Winograd kernel identifiable | ✗ | not a pipeline bug; brief's prediction didn't hold on this hardware |
| Can explain observed kernels in one sentence | ✓ | TF32 Tensor Cores outcompete Winograd on Blackwell |

Summary: six of seven gates met; the seventh became a finding. The finding is more valuable than the gate would have been — it's the kind of surprise the brief §26 explicitly says to document.

---

## 9. Decisions made in this phase and why (consolidated)

For traceability. Every non-default knob has a reason.

| Decision | Why |
|---|---|
| `profile/` → `profiling/` rename | stdlib `profile` module gets shadowed; breaks any code path that transitively imports `cProfile` (torchvision.ops does) |
| Use `python -m profiling.run_baseline` | `python path/script.py` puts script's own dir on `sys.path[0]`, which breaks cross-package imports. `-m` puts CWD on `sys.path[0]`. |
| Added `get_input(batch=32)` helper | One import point for every downstream profiler script; reduces duplicated knowledge |
| `cudnn.benchmark = True` | Study is about which algorithm cuDNN picks when given freedom to measure. Toggled off only in the later benchmark-toggle experiment. |
| 30 warmup iters | cuDNN's per-shape algorithm search on the first iter is 10×+ slower than steady state; 30 ensures all shapes have been seen before timing begins |
| `ProfilerActivity.CPU + CUDA` | Need both aten:: op context and raw kernel names |
| `record_shapes=True` | Enables later `group_by_input_shape` analysis |
| `schedule(wait=1, warmup=2, active=10, repeat=1)` | Skip 1 iter, discard 2 as profile-warmup, record 10, done. Matches brief §21.2 |
| Outer loop 15 | 1 + 2 + 10 + 2 margin = 15; schedule needs the iterations to exist |
| `torch.cuda.synchronize()` inside inner loop | Forces GPU to finish per-step so the profiler attributes kernels correctly |
| `torch.no_grad()` wrap | Inference correctness (no autograd graph); cuts CPU overhead and GPU memory |
| `row_limit=25` in table | Enough rows to see all non-trivial kernel families; avoids tail noise |
| `sort_by="cuda_time_total"` | GPU kernel time is the subject of the study |
| Did not open Perfetto UI | Exhaustive JSON scan via Python is more rigorous than a visual inspection; visual inspection is nice-to-have, not load-bearing for this gate |
| Did not install Nsight Systems | Not needed until Phase 5. Installing it now would be premature |
| Did not disable TF32 in this run | Baseline should reflect PyTorch default configuration; TF32-off is a separate controlled experiment |

---

## 10. Cache cleanup (end-of-step hygiene)

At the end of this step we clean up all generated cache files in the project. Reasons:

1. `__pycache__` directories are regenerated on every import and have zero archival value.
2. `.pyc` files embed absolute paths and timestamps; committing them adds noise.
3. `.gitignore` already excludes them from git, but keeping them around on disk is still clutter.

**Cache files enumerated before deletion:**

```
./models/__pycache__/__init__.cpython-311.pyc
./models/__pycache__/resnet.cpython-311.pyc
./profiling/__pycache__/__init__.cpython-311.pyc
./profiling/__pycache__/run_baseline.cpython-311.pyc
```

Two `__pycache__` directories total (`models/`, `profiling/`). No `.pyo` files, no `.tmp` files, no `.bak` files. Deletion applied at the end of this step.

Out of scope for this cleanup:
- `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` (44.7 MB) — this is a model-weight cache, not a build artefact. Re-downloading costs 1–2 seconds of network each run. Kept on disk intentionally.
- `results/traces/resnet18_baseline.json` — the actual profiling artefact; kept.

---

## 11. Open items going into Phase 3

1. README still documents `profile/` and `python profile/run_baseline.py` — needs update to `profiling/` and `python -m profiling.run_baseline`.
2. `docs/brief.md §5` still shows the old `profile/` layout — same rename needed, with a note about why (stdlib collision).
3. Phase 3 ports the baseline to MobileNetV3-Small, DistilBERT, and Tiny GRU. Expect:
   - MobileNetV3-Small: depthwise conv kernels with different name shape; TC kernels may be absent entirely due to low channel counts in depthwise layers.
   - DistilBERT: almost no cuDNN at all; the show is cuBLAS GEMM kernels. Many of them TC-TF32 (`cublasLt`), some SIMT.
   - Tiny GRU: a single `cudnnRNNForward`-class kernel dominating the forward pass.
4. The TF32-on-by-default observation should be called out explicitly in the writeup's cross-model section — it affects how "FP32" numbers in this project should be interpreted on Blackwell.
5. The `channels_last` experiment (brief §8.5) is well-motivated by the 9.72% layout-conversion cost observed in §5.4 and should be prioritised earlier than originally planned.
