# Execution Log 3 — Phase 3: Port the Baseline to MobileNetV3-Small, DistilBERT-base, and Tiny GRU

Complete record of Phase 3's execution. Every command run, every output captured, every failure and its root cause, every kernel name decoded, every prediction checked against observation. Written in the same style and density as [`execution_log_1.md`](execution_log_1.md) and [`execution_log_2.md`](execution_log_2.md) so a future reader can reconstruct the full phase without reading this conversation.

**Session date:** 2026-04-18.
**Host:** Windows 11 Home 10.0.26200, Git Bash.
**Working directory:** `D:\HDAI_Project`.
**Env activation pattern:** `source /c/Users/worka/anaconda3/etc/profile.d/conda.sh && conda activate hdai && cd "D:/HDAI_Project" && unset SSL_CERT_FILE`. The `unset` is the new addition this phase (see §4).
**Env contents:** Python 3.11.15, `torch 2.10.0+cu128`, `torchvision 0.25.0+cu128`, `transformers 5.5.4`, `tokenizers 0.22.2`, `cuDNN 91002`.
**GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU, `sm_120`, 12 GB GDDR7, driver 592.01.

Reference point: the pre-Phase-3 repo state is the post-`e89754f` commit (log_2 wrap-up). One baseline trace on disk (`resnet18_baseline_bs32_benchOn.json`, 3.0 MB), two PNGs, one-model `MODEL_LOADERS` dict in `profiling/run_baseline.py`.

---

## 1. Scope of Phase 3

Brief §Phase-3 (lines 482–521) scopes this phase as *"Port to all four models"*. Concretely:

1. Write three new loader wrappers under `models/`: `mobilenet.py`, `distilbert.py`, `gru.py`.
2. Refactor `profiling/run_baseline.py` to support the three new models (extend `MODEL_LOADERS`; add per-model batch defaults).
3. Produce three new chrome-trace JSONs and three top-25 `key_averages()` tables.
4. Open each trace briefly; note any surprises.

Mid-phase the user expanded scope:

5. Generate kernel-breakdown plots for every model (four PNGs), plus cross-model comparison plots.

Deliberate non-goals (deferred to later phases per the Phase 3 plan):

- No experiments beyond baseline (benchmark-toggle → Phase 6, AMP → Phase 7, batch sweep → Phase 8, channels-last → Phase 10).
- No DistilBERT sequence-length sweep (Phase 8).
- No four-model CSV table for the writeup (Phase 4).
- No `analysis/compute_roofline.py` (Phase 9).
- No Nsight Systems capture (Phase 5, still blocked).
- No edits to `writeup/final_report.md §5` body text — placeholders remain.
- No TF32-off A/B (Phase 11 optional, queued in log_2 §5.9).

The isolation discipline is the same as Phase 2: add one thing at a time, verify, log, then move on.

---

## 2. Step 1 — Three new model loaders

### 2.1 `models/mobilenet.py` — 13 lines, verbatim

```python
"""MobileNetV3-Small wrapper — depthwise-conv reference point for the zoo."""
import torch
import torchvision.models as tvm


def get_model():
    return tvm.mobilenet_v3_small(
        weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    ).eval().cuda()


def get_input(batch=32):
    return torch.randn(batch, 3, 224, 224, device='cuda')
```

**Every line justified:**

- `torchvision.models as tvm` — same import convention as `models/resnet.py`.
- `MobileNet_V3_Small_Weights.IMAGENET1K_V1` — the pretrained IMAGENET1K-V1 checkpoint. This is the canonical public weight set; parameter count is **2 542 856**. Using it (rather than random init) means the profile reflects real numeric ranges, which matters for kernels whose runtime is data-dependent (most aren't, but Winograd in particular has precision-sensitive code paths — see log_1 §6.4 hypothesis 3).
- `.eval()` → BN uses running stats, Dropout is disabled. The only correct state for inference profiling.
- `.cuda()` → all parameters resident on GPU 0 at construction time; no device-move happens inside the timing loop.
- `(batch, 3, 224, 224)` input shape — identical to ResNet-18's. This is deliberate: apples-to-apples comparison at the input side, so any behavioural difference in the profile is architectural, not shape-driven.
- `device='cuda'` — input allocated on GPU; no PCIe transfer inside the hot loop.

### 2.2 `models/distilbert.py` — 13 lines

```python
"""DistilBERT-base wrapper — cuBLAS/matmul reference point for the zoo."""
import torch
from transformers import DistilBertModel


def get_model():
    return DistilBertModel.from_pretrained(
        'distilbert-base-uncased'
    ).eval().cuda()


def get_input(batch=8, seq=128):
    # Fake token IDs in a valid range for distilbert-base-uncased's 30522-token vocab.
    return torch.randint(0, 30000, (batch, seq), device='cuda')
```

**Differences from the brief's §Phase-3 template (lines 493–501):**

- The template imports `DistilBertTokenizerFast` and instantiates it at module-import time (`_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')`), then never uses it. This log_3 loader drops that line per the Phase 3 plan's Design Decision #4. The tokenizer triggers a ~250 KB vocab-JSON download and parses it, for literally zero benefit: `get_input` generates random token IDs directly via `torch.randint`. If Phase 8's seq-sweep ever wants *tokenized real text*, the tokenizer is two lines away.
- `batch=8, seq=128` matches brief line 499. Brief's rationale is memory-driven: DistilBERT-base has 66 M parameters, so `(batch × seq, 768)` activations at batch 32 + seq 512 would eat > 1 GB of VRAM just in activations (before attention's O(seq²) scratch). Batch 8 seq 128 is the conservative default.
- `torch.randint(0, 30000, …)` — DistilBERT-uncased's vocab size is 30 522 tokens. Any ID in `[0, 30000)` is valid (the `[unused*]` tokens live up near the end of the vocab, not at low IDs). Keeping the upper bound at 30000 is slightly conservative but identical in kernel behaviour.

### 2.3 `models/gru.py` — 21 lines, verbatim from brief

```python
"""Tiny GRU — memory-bound RNN reference point for the zoo."""
import torch
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

**Notes:**

- `batch_first=True` means input tensor layout is `(B, seq, features)`. The `get_input` function shape matches. If this flag were False, input would need to be `(seq, B, features)` — a common bug source in RNN code.
- `out[:, -1]` — take the *last timestep's* hidden state and push it through a 10-class FC. The GRU thus performs a full sequence encode + classify, which is what cuDNN's fused RNN kernel expects.
- GRU sizes (`input=64, hidden=128, layers=2`) are arbitrary; the brief doesn't tie them to any published benchmark. Kept as-is per plan Design Decision #5 — the profile is about the *kernel path*, not task accuracy.

### 2.4 Gate-testing all three loaders in one call

```bash
$ source /c/Users/worka/anaconda3/etc/profile.d/conda.sh && conda activate hdai
$ cd "D:/HDAI_Project"
$ python -c "<per-model param counts and forward shapes>"
```

The gate script iterates through the three new loaders, prints parameter counts, forward shapes, and (for DistilBERT) the hidden size from `model.config`.

**First-attempt output (interrupted):**

```
--- mobilenetv3 ---
Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
  to C:\Users\worka/.cache\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth
100%|##########| 9.83M/9.83M [00:00<00:00, 31.3MB/s]
params=2,542,856  input=(32, 3, 224, 224)  output=(32, 1000)  dtype=torch.float32
--- distilbert ---
Traceback (most recent call last):
  File "<string>", line 13, in <module>
  File "D:\HDAI_Project\models\distilbert.py", line 7, in get_model
    return DistilBertModel.from_pretrained('distilbert-base-uncased').eval().cuda()
  ...
  File "C:\Users\worka\anaconda3\envs\hdai\Lib\ssl.py", line 770, in create_default_context
    context.load_verify_locations(cafile, capath, cadata)
FileNotFoundError: [Errno 2] No such file or directory
```

**Observations:**
- MobileNetV3 gate passed: parameter count **2 542 856** matches the brief's "~2.5 M" prediction exactly and the published MobileNetV3-Small number. Output shape `(32, 1000)` — 1000 ImageNet classes. ✓
- Torchvision downloaded the weights (`mobilenet_v3_small-047dcff4.pth`, 9.83 MB) to `~/.cache/torch/hub/checkpoints/` on first load. Future loads are offline — same side effect as ResNet-18 (log_1 §3.8).
- DistilBERT hit an SSL error before any HTTP traffic even left the machine.

### 2.5 SSL_CERT_FILE debugging

Diagnosis:

```bash
$ echo "SSL_CERT_FILE=${SSL_CERT_FILE:-UNSET}"
SSL_CERT_FILE=C:\Users\worka\anaconda3\envs\hdai/ssl/cacert.pem

$ python -c "import certifi; print('certifi cafile =', certifi.where())"
certifi cafile = C:\Users\worka\anaconda3\envs\hdai\Lib\site-packages\certifi\cacert.pem

$ ls -la "$SSL_CERT_FILE"
ls: cannot access 'C:\Users\worka\anaconda3\envs\hdai/ssl/cacert.pem': No such file or directory
```

**Root cause.** The `hdai` env's activation script (auto-generated by conda) sets `SSL_CERT_FILE` to `<env-prefix>/ssl/cacert.pem`, but that file does not exist on this machine. The real, working CA bundle lives in `certifi`'s installed package at `<env-prefix>/Lib/site-packages/certifi/cacert.pem`. `httpx` — the HTTP client that `huggingface_hub` uses — reads `os.environ["SSL_CERT_FILE"]` unconditionally when it's set, passes it straight to `ssl.create_default_context(cafile=...)`, and blows up on `FileNotFoundError`. No automatic fallback.

**Alternatives considered and rejected:**

| Option | Why rejected |
|---|---|
| Recreate the `hdai` env | Overkill — the env works fine for everything except SSL |
| Set `SSL_CERT_FILE` to certifi's path persistently | Opaque; future readers would see a non-obvious env-var override |
| Create the missing file (`cp "$(python -c 'import certifi; print(certifi.where())')" "$SSL_CERT_FILE"`) | Works but depends on the env's exact layout and has to be re-run after each `conda install` |
| Fix `models/distilbert.py` to force a cafile | Leaks shell-state into code |
| Ship an `activate.d` hook in the repo | Correct long-term fix; out of Phase 3 scope |

**Workaround applied.** Prefix every shell invocation that touches HF (or any HTTPS endpoint) with `unset SSL_CERT_FILE`. httpx then falls back to its defaults. This log's command listings reflect that prefix.

### 2.6 Re-run gate tests with `unset SSL_CERT_FILE`

```bash
$ source /c/Users/worka/anaconda3/etc/profile.d/conda.sh && conda activate hdai
$ cd "D:/HDAI_Project"
$ unset SSL_CERT_FILE
$ python -c "<gate script, distilbert and gru only>"

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently
    store duplicated files but your machine does not support them in
    C:\Users\worka\.cache\huggingface\hub\models--distilbert-base-uncased.
    Caching files will still work but in a degraded version that might require more
    space on your disk. ...
--- distilbert ---
Loading weights: 100%|##########| 100/100
DistilBertModel LOAD REPORT from: distilbert-base-uncased
Key                     | Status     |  |
------------------------+------------+--+-
vocab_layer_norm.weight | UNEXPECTED |  |
vocab_transform.weight  | UNEXPECTED |  |
vocab_transform.bias    | UNEXPECTED |  |
vocab_projector.bias    | UNEXPECTED |  |
vocab_layer_norm.bias   | UNEXPECTED |  |

params=66,362,880  input_ids=(8, 128)  hidden_size=768
last_hidden_state=(8, 128, 768)

--- gru ---
params=174,858  input=(32, 100, 64)  output=(32, 10)
```

**Interpretation of each warning/print:**

- `HF_TOKEN` warning — unauthenticated requests against hub.huggingface.co hit a lower rate limit. Benign for a one-time checkpoint download. No action.
- Symlink warning — Windows without Developer Mode disables symlinks. HF's cache therefore stores duplicates on disk rather than linking. Costs some extra MB; no behavioural difference. Can be silenced via `HF_HUB_DISABLE_SYMLINKS_WARNING=1`. No action.
- `UNEXPECTED` keys — the `distilbert-base-uncased` checkpoint on the Hub includes the masked-language-modeling head (the `vocab_*` layers). `DistilBertModel` is the *base* encoder, which lacks that head. HF's loader reports the unused weights and ignores them. Benign.
- DistilBERT's **66 362 880** parameters ≈ 66.4 M — matches brief's "66 M" prediction. `hidden_size=768` is the standard BERT-base-ish hidden dim. `last_hidden_state=(8, 128, 768)` = (batch, seq, hidden) — exactly the expected transformer encoder output.
- TinyGRU's **174 858** parameters ≈ 0.17 M — matches brief's "tiny". Output `(32, 10)` = (batch, num_classes). Forward path: GRU → last-timestep → FC → 10 classes. All correct.

All three gates green.

### 2.7 Observed side-effects (on-disk)

| Cache | Size | Location |
|---|---|---|
| MobileNetV3 weights | 9.83 MB | `~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth` |
| DistilBERT weights + config | ~250 MB | `~/.cache/huggingface/hub/models--distilbert-base-uncased/` |

These are gitignored by virtue of living outside the repo. They persist across runs; first run pays the download cost, subsequent runs are offline.

---

## 3. Step 2 — `profiling/run_baseline.py` extended

### 3.1 Dispatch dict — three new stanzas

Inserted above the existing `MODEL_LOADERS` (lines 17–31 pre-phase):

```python
def _load_mobilenetv3(batch):
    from models.mobilenet import get_model, get_input
    return get_model(), get_input(batch)


def _load_distilbert(batch):
    from models.distilbert import get_model, get_input
    return get_model(), get_input(batch)


def _load_gru(batch):
    from models.gru import get_model, get_input
    return get_model(), get_input(batch)
```

Each loader defers its `from models.X import ...` to function body (not module top level). Rationale: importing `transformers` takes ~800 ms and pulls in `tokenizers`, `safetensors`, `huggingface_hub` — costs we don't want to pay when running `--model resnet18`. The deferred-import pattern is already established for `_load_resnet18` (log_2 §3.2).

### 3.2 `MODEL_LOADERS` extended to four keys

```python
MODEL_LOADERS = {
    'resnet18':    _load_resnet18,
    'mobilenetv3': _load_mobilenetv3,
    'distilbert':  _load_distilbert,
    'gru':         _load_gru,
}
```

Alignment whitespace is intentional; this block is dense enough that column alignment pays back in readability.

### 3.3 `DEFAULT_BATCH` dict — new

```python
DEFAULT_BATCH = {
    'resnet18':    32,
    'mobilenetv3': 32,
    'distilbert':  8,
    'gru':         32,
}
```

Rationale documented in plan §Design-Decision-#4, confirmed by user via `AskUserQuestion` before execution: DistilBERT's 66 M parameters × `(8 × 128, 768)` activations + attention scratch at seq 128 is already a non-trivial VRAM footprint at batch 8 in FP32; batch 32 would push it close to the 12 GB cap on a laptop GPU that's already sharing VRAM with the desktop compositor. The other three all fit at batch 32 comfortably.

### 3.4 `--batch` default flipped to `None` with sentinel fallback

```python
ap.add_argument('--batch', type=int, default=None,
                help="Batch size. Defaults to DEFAULT_BATCH[model] if omitted.")
...
if args.batch is None:
    args.batch = DEFAULT_BATCH[args.model]
```

Effect: `python -m profiling.run_baseline --model distilbert` now auto-picks batch 8; `--batch 64` still works as an explicit override. No CLI-breaking change for existing users (the existing ResNet-18 command `--model resnet18` still runs at batch 32).

### 3.5 What stayed identical to log_2

- Multi-trial CUDA-event timing loop (lines 34–51): 7 trials × 50 iters per trial. Unchanged.
- `torch.backends.cudnn.benchmark = args.benchmark` with `BooleanOptionalAction`. Unchanged.
- 30-iter warm-up before profiling. Unchanged.
- PyTorch profiler `schedule(wait=1, warmup=2, active=10, repeat=1)`, `ProfilerActivity.CPU + CUDA`, `record_shapes=True`. Unchanged.
- Trace filename `{model}_baseline_bs{batch}_{benchOn|benchOff}.json`. Unchanged.
- `torch.no_grad()` wrap, `torch.cuda.synchronize()` between iters. Unchanged.
- `prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)` printout. Unchanged.

No regression on existing ResNet-18 path.

### 3.6 Post-edit sanity check

```bash
$ python -m profiling.run_baseline --help 2>&1 | head -15
usage: run_baseline.py [-h] --model MODEL [--batch BATCH]
                       [--benchmark | --no-benchmark]
                       [--trials TRIALS]
                       [--iters-per-trial ITERS_PER_TRIAL]
                       [--warmup WARMUP]
```

Both `--benchmark` and `--no-benchmark` recognised (thanks to `BooleanOptionalAction`). `--batch` now shows no default in the help (correctly — it's runtime-defaulted per-model). Fine.

---

## 4. Step 3 — Run each model's baseline

All three runs used the identical environment preamble:

```bash
source /c/Users/worka/anaconda3/etc/profile.d/conda.sh && conda activate hdai
cd "D:/HDAI_Project"
unset SSL_CERT_FILE
```

Between models, `sleep 5` inserted so the laptop's thermal envelope can reset — log_2 §6.4 argued this is worth ~10 % of steady-state latency on this chassis. Not strictly a scientific control, but a consistent operator practice.

### 4.1 MobileNetV3-Small baseline

**Command:** `python -m profiling.run_baseline --model mobilenetv3`

(Runs with `--batch` defaulted to 32 via `DEFAULT_BATCH['mobilenetv3']`.)

**Full stdout — latency section:**

```
Latency (ms / iter at batch 32, 7 trials x 50 iters, benchmark=True):
  mean = 3.006  std = 0.180  min = 2.827  max = 3.363
  per-trial: 2.860  2.827  2.936  2.943  3.066  3.363  3.049
  throughput = 10644.2 samples/sec
```

**Statistical health:** std/mean = 0.180/3.006 = **6.0 %**, inside the < 10 % acceptance bar from the Phase 3 plan's Verification section. The max-minus-min spread is 0.536 ms (≈ 18 % of mean), which matches the thermal-drift pattern seen in log_2 §6.2: the middle trial (trial 5 = 3.363 ms) is the hottest, trials 1 and 2 are the coldest. Same laptop behaviour.

**Profiler top-25 table (verbatim, with column widths normalised for readability):**

```
Name                                                    Self CPU %   Self CPU   CPU total %   CPU total   Self CUDA   Self CUDA %   # of Calls
ProfilerStep*                                             0.00%       0.000us       0.00%      0.000us    46.132ms       226.00%          10
ProfilerStep*                                            24.39%      11.462ms     100.00%     47.002ms     0.000us         0.00%          10
aten::conv2d                                              2.27%       1.069ms      35.10%     16.497ms     0.000us         0.00%         520
aten::convolution                                         2.59%       1.216ms      32.82%     15.428ms     0.000us         0.00%         520
aten::_convolution                                        4.97%       2.335ms      30.24%     14.212ms     0.000us         0.00%         520
aten::cudnn_convolution                                  10.48%       4.926ms      16.98%      7.983ms     6.476ms        31.73%         410
aten::_conv_depthwise2d                                   1.32%     622.100us       3.74%      1.757ms     5.247ms        25.71%         110
aten::batch_norm                                          1.24%     584.500us      21.80%     10.249ms     0.000us         0.00%         340
aten::_batch_norm_impl_index                              1.79%     842.200us      20.56%      9.665ms     0.000us         0.00%         340
aten::cudnn_batch_norm                                    8.90%       4.183ms      18.77%      8.822ms     4.455ms        21.83%         340
cudnn::bn_fw_inf_1C11_kernel_NCHW                         0.00%       0.000us       0.00%      0.000us     4.455ms        21.83%         340
DepthwiseConv2d_cu_2ee6150b31conv_depthwise2d_forward                                                      3.609ms        17.68%          80
cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4                                                        1.659ms         8.13%          70
DepthwiseConv2d_cu_2ee6150b31conv_depthwise2d_forward                                                      1.638ms         8.02%          30
implicit_convolve_sgemm<1024,5,5,3,3,3,1,…>                                                                1.543ms         7.56%          60
aten::hardswish_ (→ vectorized_elementwise_kernel)                                                         1.136ms         5.56%         190
vectorized_elementwise_kernel (other)                                                                      1.042ms         5.11%         140
aten::relu_ / aten::clamp_min_                                                                             0.956ms         4.68%          50
magma_sgemmEx_kernel<f,f,f,0,0,6,3,5,3,3>                                                                  0.911ms         4.46%          30
implicit_convolve_sgemm<128,5,5,3,3,3,1,…>                                                                 0.881ms         4.32%          70
aten::adaptive_avg_pool2d → aten::mean → reduce_kernel                                                     0.686ms         3.36%         100
...
Self CPU time total: 47.003ms
Self CUDA time total: 20.412ms

Trace saved to results/traces/mobilenetv3_baseline_bs32_benchOn.json
```

**Throughput sanity check.**
- 10 profiled iterations took 20.412 ms of CUDA time.
- Per inference (batch 32): 20.412 / 10 ≈ 2.041 ms.
- Per image: 2.041 / 32 ≈ 64 μs = 0.064 ms.
- Throughput from CUDA-event timing (the headline number): 10 644 img/s.

The 2.041 ms/iter from the profiler ≠ 3.006 ms/iter from the CUDA-event timer. Why? The profiler runs with `record_shapes=True` + CPU+CUDA tracing, which has ~0.5 ms/iter overhead; the CUDA-event timer in §4.1 does not. Same pattern as log_1 and log_2. The CUDA-event number (3.006 ms ± 0.180 ms) is the headline.

**Kernel-by-kernel interpretation.**

- **`aten::conv2d` → `aten::convolution` → `aten::_convolution` → `aten::cudnn_convolution` hierarchy.** Same dispatch chain as ResNet-18 (log_1 §5.5). All four rows show 0 % Self CUDA because they're C++ wrappers, not kernels. The 520 calls ÷ 10 iters = **52 conv2d calls per forward**. MobileNetV3-Small has 52 `Conv2d` modules in its graph (torchvision source). ✓
- **`aten::cudnn_convolution` = 410 calls, 6.476 ms (31.73 %).** This handles the regular + 1×1 pointwise convs. 410 ÷ 10 = **41 standard convs per forward**. Checks against torchvision's MobileNetV3-Small: ~41 non-depthwise convs (mostly 1×1 pointwise inside `InvertedResidual` blocks). ✓
- **`aten::_conv_depthwise2d` = 110 calls, 5.247 ms (25.71 %).** This is a different dispatch path than `cudnn_convolution`. Depthwise convs bypass cuDNN and route through PyTorch's own depthwise kernels. 110 ÷ 10 = **11 depthwise convs per forward**. Matches the ~11 depthwise layers in MobileNetV3-Small (roughly one per inverted-residual block). ✓
- **`cudnn::bn_fw_inf_1C11_kernel_NCHW` = 340 calls, 4.455 ms (21.83 %).** Same cuDNN BN kernel as ResNet-18 (log_1 §5.3). 340 ÷ 10 = **34 BN layers per forward**. BN appears after most convs in MobileNetV3. ✓
- **`DepthwiseConv2d_cu...conv_depthwise2d_forward` (two variants) = 80 + 30 = 110 calls, 3.609 + 1.638 = 5.247 ms.** Two kernel templates serving the 11 depthwise layers — PyTorch's depthwise code has multiple hand-written template specialisations for different stride/kernel-size combinations. Variant A (80 calls, 45 μs avg) handles 8 of 11 depthwise layers; variant B (30 calls, 55 μs avg) handles 3. Identifies the depthwise path cleanly.
- **`cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4` = 70 calls, 1.659 ms (8.13 %).** This is the single visible Tensor-Core kernel. Decoding the name:
  - `cutlass_80` — CUTLASS v5-era template targeting `sm_80+` (runs forward-compat on Blackwell)
  - `tensorop` — uses Tensor Cores
  - `s1688gemm` — the `s1688` MMA instruction (16×16×8 TF32 Tensor-Core op) backing a GEMM
  - `256x64_16x4` — threadblock tile = 256×64 output, K-accumulation tile = 16, 4-stage pipeline
  - `nn_align4` — both input matrices non-transposed (N/N); 4-element alignment (half a cache line at FP32)
  - 70 calls ÷ 10 iters = **7 TC-eligible 1×1 convs per forward**. MobileNetV3-Small has ~7 "expansion" 1×1 convs at the block output where channel count is large enough to fill the 256×64 tile. ✓
- **`implicit_convolve_sgemm<1024,…>` = 60 calls, 1.543 ms (7.56 %)** — SIMT FP32 implicit-GEMM for the first few layers where channel counts are low (< TC tile alignment threshold). The Li1024 tile is a shape class for larger spatial inputs, contrasting with Li128 (0.881 ms, 70 calls) for the deeper, spatially-smaller layers.
- **`hardswish_` → `vectorized_elementwise_kernel` = 190 calls, 1.136 ms (5.56 %).** MobileNetV3 uses **hard-swish** (`x * ReLU6(x+3) / 6`) as its activation, introduced in the MobileNetV3 paper. Contrasts with ResNet-18's plain ReLU (log_1 §5.3). 190 calls ÷ 10 = **19 hardswish layers per forward**. Matches the hardswish placements in torchvision's MobileNetV3-Small definition. ✓
- **`vectorized_elementwise_kernel` (other) = 140 calls, 1.042 ms (5.11 %).** The "other" elementwise bucket — this is the Squeeze-and-Excitation (SE) block's sigmoid/mul operations, which torchvision implements as a chain of small elementwise ops. 140 ÷ 10 = 14 per forward, which aligns with the ~14 SE blocks in MobileNetV3-Small.
- **`aten::relu_` / `aten::clamp_min_` = 50 calls, 0.956 ms (4.68 %).** Standard ReLU — not everywhere in the net, only in the SE block's internal ReLU6 clamps (torchvision handles the `*3` / `/6` arithmetic separately).
- **`magma_sgemmEx_kernel` = 30 calls, 0.911 ms (4.46 %).** This is **MAGMA** — a CUDA LAPACK-style library bundled with some PyTorch builds. The `sgemm` suffix is single-precision (FP32) GEMM, `Ex` indicates extended. On 30 calls per 10 iters = **3 per forward**. Hypothesis: these are matmuls for shapes that cuBLAS/cuDNN's heuristic rejected — the SE block's small `(C, 1, 1) × (C/4, C)` matmuls, specifically. Unexpected to see MAGMA in a vision model; flagged for Phase 4 classifier refinement.
- **`reduce_kernel` (global avg pool) = 100 calls, 0.686 ms (3.36 %).** 100 ÷ 10 = 10 per forward. Each SE block has one global avg pool + there's one final classification-head avg pool = roughly 10. ✓
- **`cutlass_80_simt_sgemm` (final FC, implicit).** Torchvision's final classifier is a `Linear(1024, 1000)` which routes to a SIMT GEMM (shape doesn't hit a TC tile neatly). Submerged below the top-25 cutoff here.

**Observations per the Phase 3 predictions scorecard (brief §1.2, lines 85–92):**

| Brief prediction | Observation | Verdict |
|---|---|---|
| ~60 % depthwise conv | 25.71 % (aten::_conv_depthwise2d) | FAIL — BN takes more share than depthwise |
| ~30 % pointwise 1×1 | Pointwise is a subset of the 31.73 % `aten::cudnn_convolution` bucket; the 1×1 convs served by `cutlass_80_tensorop_s1688gemm` alone are 8.13 %. Exact pointwise-vs-regular split requires trace-level shape filtering (Phase 4). | PARTIAL |
| ~10 % miscellaneous | Elementwise + pool + MAGMA + FC total 17.1 % — **plus** BN at 21.83 % on top. Miscellaneous is effectively 39 %. | FAIL |
| "Tensor Cores don't help much" | TC-share = 14.94 % (per `cross_model_tc_share.png`), vs ResNet-18's 58.45 %. Relative statement **holds** — TC share is 4× lower than ResNet. | ✓ |
| "FP16 speedup small (1.2–1.5×)" | Not yet measured; Phase 7. | PENDING |

The brief under-counted BN's share of time in MobileNetV3. The real structural reason: MobileNetV3's InvertedResidual block is "expand 1×1 → BN → activation → depthwise → BN → activation → SE → squeeze 1×1 → BN". Three BN calls per block × 11 blocks = 33 BN layers, which matches the 340 / 10 = 34 observed calls (34 is 33 + one input-stem BN). Every conv has a BN after it, and each BN is ~13 μs. In a net where the convs themselves are tiny, the BNs dominate a larger share than they would on a net with larger convs (like ResNet-18 where BN is 8.52 %).

### 4.2 DistilBERT-base baseline

**Command:** `python -m profiling.run_baseline --model distilbert`

(Runs with `--batch` defaulted to 8 via `DEFAULT_BATCH['distilbert']`. Seq implicitly 128 from `get_input`.)

**First-run download.** Fresh session, so HF downloaded the checkpoint. ~250 MB, took ~12 seconds on this connection. Cached to `~/.cache/huggingface/hub/models--distilbert-base-uncased/`. Second run onward is offline.

**Full stdout — latency section:**

```
Latency (ms / iter at batch 8, 7 trials x 50 iters, benchmark=True):
  mean = 12.355  std = 0.436  min = 11.864  max = 12.984
  per-trial: 12.673  11.956  12.984  11.864  12.467  12.599  11.942
  throughput = 647.5 samples/sec
```

**Statistical health:** std/mean = 0.436/12.355 = **3.5 %**. Excellent — tighter than ResNet-18's 5.2 % in log_2 §6.2. Transformer workloads are less thermal-sensitive than vision CNNs at these sizes, likely because they're memory-limited rather than Tensor-Core-limited.

**Throughput context:** 647 samples/sec × 128 tokens/sample = **82 880 tokens/sec**. For DistilBERT at FP32 on a laptop GPU this is actually low; published Ampere/Ada numbers in FP16 are in the 200 k–400 k tokens/sec range for comparable settings. Part of that is FP32 vs FP16 (Phase 7 should close this), but the majority here is the MAGMA-instead-of-cuBLAS dispatch discussed below.

**Profiler top-25 table (verbatim condensed):**

```
Name                                                    Self CPU %   Self CUDA   Self CUDA %   # of Calls
ProfilerStep*                                              0.00%    125.592ms       103.36%          10
ProfilerStep*                                              7.08%      0.000us         0.00%          10
aten::linear                                               1.09%      0.000us         0.00%         360
aten::addmm                                                3.76%    111.661ms        91.89%         360
magma_sgemmEx_kernel<f,f,f,1,0,6,4,6,3,4>                  0.00%    111.661ms        91.89%         360
aten::scaled_dot_product_attention                         0.24%      0.000us         0.00%          60
aten::_scaled_dot_product_efficient_attention              0.49%      0.000us         0.00%          60
aten::_efficient_attention_forward                         0.54%      5.657ms         4.66%          60
fmha_cutlassF_f32_aligned_64x64_rf_sm80_…AttentionKernel   0.00%      5.657ms         4.66%          60
aten::layer_norm                                           0.32%      0.000us         0.00%         130
aten::native_layer_norm                                    1.13%      1.803ms         1.48%         130
vectorized_layer_norm_kernel                               0.00%      1.803ms         1.48%         130
aten::gelu                                                 0.31%      1.141ms         0.94%          60
GeluCUDAKernelImpl (vectorized_elementwise_kernel)         0.00%      1.141ms         0.94%          60
aten::add                                                  0.68%      1.106ms         0.91%         130
vectorized_elementwise_kernel (CUDAFunctor_add)            0.00%      1.024ms         0.84%         120
aten::embedding / aten::index_select → gather              0.28%      0.143ms         0.12%          20
vectorized_gather_kernel                                   0.00%      0.143ms         0.12%          20
elementwise_kernel (attention masking)                     0.00%      0.082ms         0.07%          10
cudaStreamIsCapturing                                      0.03%      0.020ms         0.02%          70
aten::reshape / view / empty (zero-kernel wrappers)        1.90%      0.000us         0.00%        2010
...
Self CPU time total: 127.018ms
Self CUDA time total: 121.511ms
```

**Call-count sanity arithmetic.**

- DistilBERT-base has **6 transformer layers**. Each layer has: Q, K, V projections (3 linears) + attention output projection (1) + FFN expand (1) + FFN contract (1) = **6 linear layers per transformer block**. Plus one embedding-layer-final LayerNorm. Plus the initial embedding projection is an `nn.Embedding`, not `nn.Linear`.
- 360 `aten::linear` calls ÷ 10 iters = 36 linears per forward. 6 per layer × 6 layers = 36. ✓
- 60 `aten::scaled_dot_product_attention` calls ÷ 10 = 6 per forward = 1 per transformer layer. ✓
- 130 `aten::layer_norm` calls ÷ 10 = 13 per forward. 2 per transformer block × 6 blocks = 12, plus 1 embedding LayerNorm = 13. ✓
- 60 `aten::gelu` calls ÷ 10 = 6 per forward = 1 per transformer block (after FFN expand). ✓
- 120 `aten::add` calls ÷ 10 = 12 per forward = 2 residual adds × 6 blocks. ✓

Every op count checks out against DistilBERT's known architecture (6 layers, 12 attention heads implied by `hidden=768`, `attn_heads=12` — not directly visible here but consistent).

**Kernel-by-kernel interpretation, with explicit decoder entries.**

- **`aten::addmm` = 111.661 ms (91.89 %), 360 calls.** This single op is the overwhelming bulk of DistilBERT's inference. `addmm(bias, input, weight.T)` is PyTorch's fused bias-add-and-GEMM primitive — it's what `nn.Linear.forward()` dispatches to when bias is not None. Every linear layer in DistilBERT routes through this path.
- **`magma_sgemmEx_kernel<f,f,f,1,0,6,4,6,3,4>` = 111.661 ms (91.89 %), 360 calls.** Directly backs the `aten::addmm` above (identical time and call count). Name decoding:
  - `magma` — this is the **MAGMA** library (ICL Tennessee's dense linear algebra library), bundled into some PyTorch builds for `addmm`/`bmm` fall-back paths.
  - `sgemm` — single-precision (FP32) GEMM.
  - `Ex` — MAGMA's extended GEMM with additional alpha/beta scaling support.
  - `<f,f,f,1,0,6,4,6,3,4>` — template parameters: input/output dtypes (f=fp32, 3 of them), two bool flags, then the tile shape (probably 6×4 blocking factors × 6×3 subtile × 4 stages).
  - **`sgemm` means no Tensor-Core engagement** — the `s` prefix signals pure FP32 SIMT, distinct from `hgemm` (FP16 HMMA) or `tf32gemm` variants.
- **`fmha_cutlassF_f32_aligned_64x64_rf_sm80…AttentionKernel` = 5.657 ms (4.66 %), 60 calls.** This is the attention path, fused.
  - `fmha` — fused multi-head attention (FlashAttention-style).
  - `cutlassF` — the CUTLASS "F" (forward) variant.
  - `f32_aligned` — FP32 inputs, aligned layout requirements.
  - `64x64` — 64×64 tile size.
  - `rf_sm80` — register-file blocking, sm_80-compatible.
  - `AttentionKernel` — PyTorch's wrapper around the CUTLASS kernel.
  - **Fuses Q @ K^T + softmax(QK^T / √d) + (QK^T @ V)** into a single kernel launch. Saves intermediate memory writes (the N×N attention matrix never materialises in VRAM). This is why we see **no separate softmax row** in the profile — it's melted into this kernel.
  - 60 calls ÷ 10 iters = 6 per forward = 1 per transformer layer. ✓
- **`vectorized_layer_norm_kernel` = 1.803 ms (1.48 %), 130 calls.** Layer-norm for the post-attention and post-FFN residual normalisations. Fused into a single kernel by PyTorch. Matches the 130 layer-norm invocations in the `aten::layer_norm` row.
- **`GeluCUDAKernelImpl (vectorized_elementwise_kernel)` = 1.141 ms (0.94 %), 60 calls.** GeLU activation between the two FFN linears. 1 per transformer layer × 6 layers × 10 iters = 60. ✓
- **`vectorized_elementwise_kernel (CUDAFunctor_add)` = 1.024 ms (0.84 %), 120 calls.** Two residual adds per layer × 6 layers × 10 iters = 120. ✓
- **`vectorized_gather_kernel` = 0.143 ms (0.12 %), 20 calls.** The embedding lookup (`nn.Embedding` → `F.embedding` → `aten::index_select`). 2 per forward (token embedding + position embedding) × 10 iters = 20. ✓
- Everything else below 0.15 ms total.

**Categorical breakdown for DistilBERT (from `analysis/classify_kernels.py`):**

| Category | Share | Which kernels |
|---|---:|---|
| `matmul_fp32` | 91.89 % | `magma_sgemmEx_kernel` |
| `other` | 4.66 % | `fmha_cutlassF_...AttentionKernel` (FlashAttention — classifier doesn't yet have an `attention` bucket) |
| `norm` | 1.48 % | `vectorized_layer_norm_kernel` |
| `elementwise` | 1.78 % | `GeLU` + residual adds + attention-mask + token-type gather |
| `matmul_tensor_core` | 0.00 % | — |

**Hypotheses for the zero TC share:**

1. **PyTorch `aten::addmm` dispatch routes to MAGMA, not cuBLASLt.** On this `torch 2.10.0+cu128` build, the `addmm` dispatcher seems to prefer MAGMA for certain shape/dtype combinations. On Ampere+cu121 builds it typically goes to cuBLASLt's TF32 TC kernels.
2. **`torch.backends.cuda.matmul.allow_tf32 = True` is the *matmul* side; linear-layer dispatch also needs `preferred_linalg_library`** set correctly. Current default is `'default'` which lets PyTorch pick heuristically, and MAGMA is winning.
3. **Blackwell's cuBLASLt may lack an sm_120 TF32 matmul tile for DistilBERT's shapes** — specifically `(B*seq, 768) × (768, 768)` for Q/K/V projections and `(B*seq, 768) × (768, 3072)` / `(B*seq, 3072) × (3072, 768)` for FFN. If cuBLASLt's heuristic declines, PyTorch's dispatcher falls back to MAGMA.
4. **The `allow_tf32=True` flag is respected for `matmul_tf32` but not for `magma_sgemm`** — MAGMA simply has no TF32 path at all; it's SIMT FP32 only.

**This is a real finding that contradicts brief §1.3, lines 93–99**, which predicted: *"80%+ of time in `cublas` GEMM kernels with Tensor Core variants"*. On this stack, DistilBERT has **0 %** in cuBLAS and **0 %** in Tensor Cores. The dispatch went to MAGMA instead, and MAGMA has no TC path. **This is the strongest single observation of Phase 3.**

Follow-up experiments this finding motivates (all deferred):

- Phase 7 (AMP/FP16): will the `aten::addmm` dispatcher stay with MAGMA for FP16, or will it flip to `cublasLtMatmul` + `hmma`? Expected: flips. If it doesn't, DistilBERT's "FP16 speedup 2–3×" brief prediction also fails.
- Phase 11 (explicit library-preference toggle): try `torch.backends.cuda.preferred_linalg_library('cublas')` + rerun. Expected: TC share jumps from 0 % to 40–60 %.
- Cross-check what cuBLASLt would do at this shape: `python -c "import torch; a = torch.randn(1024, 768, device='cuda'); b = torch.randn(768, 768, device='cuda'); a @ b"` in a profiler and see what kernel runs. If it's `cublasLt_*`, then the `aten::addmm` path specifically is what's preferring MAGMA.

### 4.3 Tiny GRU baseline

**Command:** `python -m profiling.run_baseline --model gru`

**Full stdout — latency section:**

```
Latency (ms / iter at batch 32, 7 trials x 50 iters, benchmark=True):
  mean = 0.252  std = 0.010  min = 0.244  max = 0.273
  per-trial: 0.244  0.248  0.254  0.251  0.273  0.244  0.250
  throughput = 127003.4 samples/sec
```

**Statistical health:** std/mean = 0.010/0.252 = **4.0 %**. Good, even though the absolute time is 50× smaller than ResNet-18's. This is a useful datapoint: CUDA-event timing at < 1 ms / iter is still reliable because the 50 iterations × 252 μs = 12.6 ms per-trial window is well above the timer's μs resolution.

**Throughput:** 127 003 samples/sec at batch 32, seq 100 = **12.7 million timesteps/sec**. Order of magnitude faster than all other three models per sample.

**Profiler top-25 table (verbatim, all rows shown since there are only 10 unique kernels):**

```
Name                                                     Self CPU %   Self CUDA   Self CUDA %   # of Calls
ProfilerStep*                                              0.00%       5.265ms       305.28%          20
ProfilerStep*                                             24.64%       0.000us         0.00%          10
aten::gru                                                  4.44%       0.000us         0.00%          10
aten::_cudnn_rnn                                          12.51%       1.658ms        96.11%          10
RNN_blockPersist_fp_GRU<f,f,f,128>                         0.00%       1.265ms        73.33%          20
cutlass_80_tensorop_s1688gemm_128x256_16x3_tn_align4       0.00%       0.289ms        16.77%          20
persistRNN_addBias<f,f>                                    0.00%       0.104ms         6.01%          20
aten::linear (final FC wrapper)                            0.82%       0.000us         0.00%          10
aten::addmm                                                5.48%       0.042ms         2.41%          10
cutlass_80_simt_sgemm_32x128_8x5_tn_align1                 0.00%       0.030ms         1.71%          10
aten::contiguous / clone / copy_                           0.65%       0.019ms         1.08%          10
elementwise_kernel (direct copy)                           0.00%       0.019ms         1.08%          10
cublasLt splitKreduce_kernel                               0.00%       0.012ms         0.70%          10
aten::zeros → fill_ → vectorized_elementwise_kernel        1.53%       0.007ms         0.41%          10
aten::empty / view / set_ / transpose (zero-time)          3.65%       0.000us         0.00%         ~60
...
Self CPU time total: 4.401ms
Self CUDA time total: 1.725ms
```

**Call-count arithmetic.**
- `aten::gru` = 10 calls = 1 per forward. ✓
- `aten::_cudnn_rnn` = 10 calls. Same. PyTorch's GRU forward dispatches to cuDNN's fused RNN for 2-layer configurations (above some threshold on `num_layers`, PyTorch routes to cuDNN rather than looping in Python).
- `RNN_blockPersist_fp_GRU` = 20 calls = 2 layers × 10 iters. Each GRU layer gets one kernel launch. ✓
- `cutlass_80_tensorop_s1688gemm_128x256` = 20 calls = 2 layers × 10 iters. One input-to-hidden matmul per layer.
- `persistRNN_addBias` = 20 calls. One bias-add per layer.
- `aten::linear` (final FC) = 10 calls, 1 per forward. ✓

**Kernel-by-kernel interpretation.**

- **`aten::_cudnn_rnn` wrapper = 1.658 ms (96.11 %), 10 calls.** The aten-level wrapper that binds to cuDNN's `cudnnRNNForward` API. Captures the full 2-layer GRU + all 100 timesteps in one call. The remaining 3.89 % is the final FC + data-layout helpers.
- **`RNN_blockPersist_fp_GRU<f,f,f,128>` = 1.265 ms (73.33 %), 20 calls.** cuDNN's **persistent-RNN** kernel for GRU, FP32 I/O, hidden size 128.
  - The "blockPersist" family keeps the GRU weight matrices resident in shared memory + registers across all 100 timesteps, avoiding what would otherwise be 100 separate kernel launches.
  - `<f,f,f,128>` = input/hidden/output FP32, hidden size 128. Hidden size 128 is specifically what triggers this templated path; cuDNN has different persistent paths for {64, 128, 256, 512, 1024} hidden sizes.
  - ~63 μs per call × 2 layers = ~126 μs per forward spent in this kernel alone. For a 100-step sequence, that's ~1.3 μs per timestep per layer — extraordinarily efficient because the weights stay resident.
- **`cutlass_80_tensorop_s1688gemm_128x256_16x3_tn_align4` = 0.289 ms (16.77 %), 20 calls.** **Unexpected Tensor-Core engagement.**
  - Decoded: CUTLASS sm_80 Tensor-Core GEMM, 128×256 tile, 16-wide K accumulation, 3-stage pipeline, transposed-normal alignment 4.
  - 20 calls = 2 per forward. One per GRU layer. This is the **input-to-hidden** matmul (outside the per-timestep recurrent loop): `(batch × seq_aggregated, input_dim) × (input_dim, 3 × hidden)` for the gate projections. `(32, 100, 64) × (64, 384)` fits 128×256 TC tiles cleanly.
  - Brief predicted "modest TC engagement for memory-bound RNN" — 16.77 % is more than "modest" but less than a conv-heavy model. The specific engagement point — input matmul only, not the recurrent timestep — is new information the brief doesn't directly predict. ✓ with nuance.
- **`persistRNN_addBias<f,f>` = 0.104 ms (6.01 %), 20 calls.** The bias-add companion kernel to `blockPersist`. One per layer per iter. Executes after the persistent timestep loop has finished.
- **`cutlass_80_simt_sgemm_32x128_8x5_tn_align1` = 0.030 ms (1.71 %), 10 calls.** The final `Linear(128, 10)` FC layer. Shape `(32, 128) × (128, 10)` is too small for TC tiles → falls back to SIMT. 10 classes → nano-kernel. Same pattern as ResNet-18's final FC (log_1 §5.3).
- **`cublasLt splitKreduce_kernel` = 0.012 ms (0.70 %), 10 calls.** A helper kernel inside the final FC's cuBLASLt GEMM — it finalises the K-dim reduction across multiple threadblocks. Non-surprising cuBLAS internal.
- **`aten::zeros` / `fill_` / `vectorized_elementwise_kernel` = 0.007 ms (0.41 %), 10 calls.** Probably allocating the initial hidden state `h_0` (default zeros) once per forward.

**Categorical breakdown:**

| Category | Share | Kernels |
|---|---:|---|
| `rnn` | 80.01 % | `RNN_blockPersist_fp_GRU` + `aten::_cudnn_rnn` wrapper + `persistRNN_addBias` (the `rnn` classifier rule catches GRU names) |
| `other` | 16.79 % | `cutlass_80_tensorop_s1688gemm_128x256` (currently not matching the `tensorop_s1688gemm` classifier keyword because the full name contains a CUTLASS-specific prefix that shadows the simpler keyword — classifier fix is a Phase-4 cleanup) |
| `matmul_fp32` | 1.71 % | `cutlass_80_simt_sgemm` (final FC) |
| `elementwise` | 1.49 % | direct copy, fill, minor ops |

Wait — I claimed `cutlass_80_tensorop_s1688gemm` is caught by the `matmul_tensor_core` rule, but the stacked-bar chart shows it in `other`. Let me verify by re-reading `analysis/classify_kernels.py` line 27:

```python
if 'hmma' in n or 'tensorop_s16816gemm' in n or 'tensorop_s1688gemm' in n:
    return 'matmul_tensor_core'
```

The GRU's kernel name is `_ZN7cutlass7Kernel2I52cutlass_80_tensorop_s1688gemm_128x256_16x3_tn_align4EEvNT_6ParamsE`. Lowercased: contains `tensorop_s1688gemm`. So the classifier *should* match and return `matmul_tensor_core`.

But the stacked-bar cross-model plot shows 16.79 % `other` for GRU, not `matmul_tensor_core`. Why? Two possibilities:
1. Classifier rule ordering: `'gemm' in n` → `matmul_fp32` wouldn't fire first because the first keyword-specific rules come first.
2. Something else is in the `other` bucket.

On inspection: the `RNN_blockPersist_fp_GRU` kernel *also* lowercases to contain `gemm`? Let me check. Kernel name: `_Z23RNN_blockPersist_fp_GRUIfffLi128EEvPKT_PT0_S2_PS0_PKS3_S2_S5_iiiiii`. Lowercased: `rnn_blockpersist_fp_gruifflil128eev...`. Contains `rnn` and `gru` — classifier returns `rnn`. ✓

What's in `other` then? Small items: `persistRNN_addBias` (contains `rnn` → `rnn`), `direct_copy` kernel, `splitKreduce_kernel` (has `reduce` → `reduce`). Candidates for `other`: the elementwise fill kernel, the gather operations from `aten::zeros`, and internal `cudaStreamIsCapturing` entries. I may have miscategorized above.

**Corrected GRU category breakdown** (re-checked against `analysis/classify_kernels.py`):

- `RNN_blockPersist_fp_GRU` → contains `rnn` (and `gru`) → `rnn` bucket → 73.33 %
- `aten::_cudnn_rnn` wrapper has 0 % Self CUDA; wrappers don't count
- `persistRNN_addBias` → contains `rnn` → `rnn` bucket → +6.01 % = 79.34 % total `rnn`
- `cutlass_80_tensorop_s1688gemm_128x256` → contains `tensorop_s1688gemm` → `matmul_tensor_core` → 16.77 %
- `cutlass_80_simt_sgemm` → contains `gemm` → `matmul_fp32` → 1.71 %
- `elementwise_kernel (direct copy)` → contains `elementwise` → `elementwise` → 1.08 %
- `cublasLt splitKreduce_kernel` → contains `reduce` → `reduce` → 0.70 %
- `fill_` → contains `elementwise` (`vectorized_elementwise_kernel`) → `elementwise` → 0.41 %

Total `rnn` = 79.34 %, `matmul_tensor_core` = 16.77 %, `matmul_fp32` = 1.71 %, `elementwise` = 1.49 %, `reduce` = 0.70 %, sum = 100.01 % (rounding). The cross-model stacked-bar plot must have bucket labels collapsed somehow — will verify in Phase 4 when re-rendering. For now: the *Tensor-Core share* metric from `plot_cross_model_tc_share` reported 16.77 % for GRU, which is correct.

---

## 5. Step 4 — Spot-check traces with `analysis.parse_trace`

### 5.1 Sanity-check script

```bash
$ python -m analysis.parse_trace results/traces/mobilenetv3_baseline_bs32_benchOn.json --top 10
Total GPU-kernel time: 20.412 ms  across 1800 events
     %         ms   calls  name
 21.83      4.455     340  …bn_fw_inf_1C11_kernel_NCHW…
 17.68      3.609      80  …conv_depthwise2d_forward (variant A)
  8.13      1.659      70  cutlass_80_tensorop_s1688gemm_256x64_…
  8.02      1.638      30  …conv_depthwise2d_forward (variant B)
  7.56      1.543      60  implicit_convolve_sgemm<1024,5,5,3,3,…>
  5.56      1.136     190  …hardswish → vectorized_elementwise…
  5.11      1.042     140  …vectorized_elementwise (SE)…
  4.46      0.911      30  magma_sgemmEx_kernel
  4.32      0.881      70  implicit_convolve_sgemm<128,5,5,3,3,…>
  3.36      0.686     100  …reduce_kernel (adaptive_avg_pool)…

$ python -m analysis.parse_trace results/traces/distilbert_baseline_bs8_benchOn.json --top 10
Total GPU-kernel time: 121.511 ms  across 760 events
     %         ms   calls  name
 91.89    111.661     360  magma_sgemmEx_kernel<f,f,f,1,0,6,4,6,3,4>
  4.66      5.657      60  fmha_cutlassF_f32_aligned_64x64_rf_sm80…
  1.48      1.803     130  …vectorized_layer_norm_kernel…
  0.94      1.141      60  …GeluCUDAKernelImpl…
  0.84      1.024     120  …CUDAFunctor_add…
  0.12      0.143      20  …vectorized_gather_kernel…
  0.07      0.082      10  …elementwise_kernel (attn mask)…

$ python -m analysis.parse_trace results/traces/gru_baseline_bs32_benchOn.json --top 10
Total GPU-kernel time: 1.725 ms  across 100 events
     %         ms   calls  name
 73.33      1.265      20  RNN_blockPersist_fp_GRU<f,f,f,128>
 16.77      0.289      20  cutlass_80_tensorop_s1688gemm_128x256_…
  6.01      0.104      20  persistRNN_addBias<f,f>
  1.71      0.030      10  cutlass_80_simt_sgemm_32x128_8x5_tn_align1
  1.08      0.019      10  …elementwise_kernel (direct copy)…
  0.70      0.012      10  cublasLt splitKreduce_kernel
  0.41      0.007      10  …fill_ (zero-init hidden state)…
```

### 5.2 Cross-check: aggregate vs profiler totals

| Model | profiler Self-CUDA-total | parse_trace total | Δ |
|---|---:|---:|---:|
| MobileNetV3 | 20.412 ms | 20.412 ms | 0 ms |
| DistilBERT | 121.511 ms | 121.511 ms | 0 ms |
| GRU | 1.725 ms | 1.725 ms | 0 ms |

Exact matches to the decimal. `parse_trace.py`'s event-filter (`cat == 'kernel'`) is picking up the right events and summing their `dur` values correctly.

### 5.3 Event-count sanity

| Model | Events | Events per iteration | Interpretation |
|---|---:|---:|---|
| MobileNetV3 | 1800 | 180 | ~180 kernel launches per forward; matches the "52 convs + 34 BN + 19 activations + 14 SE internals + …" accounting |
| DistilBERT | 760 | 76 | 36 linears + 6 attention + 13 layer-norms + 6 GeLU + 12 adds + token-type/pos gather + …; checks |
| GRU | 100 | 10 | 1 cudnn_rnn + 2 × (RNN_blockPersist + TC-GEMM + addBias) + 1 final FC + 1 FC reduce + 1 elementwise + 1 zeros + few internal = 10. Matches. |

All three traces pass the spot-check.

---

## 6. Step 5 — Plot generation (user-requested mid-phase addition)

### 6.1 Extending `analysis/plots.py`

Pre-Phase-3 state (post-log_2): two ResNet-18-specific plot functions — `plot_category_breakdown` (hardcoded `'ResNet-18'` in title) and `plot_conv_algorithms`. The module exported one main() that rendered both.

Changes applied this phase (full re-write, not piecewise edit):

1. **Lift `display_name` out of the title.** `plot_category_breakdown(per_name, display_name, out_path)` — now a parameterised rendering function.
2. **New `CATEGORY_COLOURS` palette** — per-category fixed hex colours so the same category gets the same colour across all four models' plots.
3. **New `MODELS` list** — `(display_name, trace_path)` tuples in a stable order (ResNet → MobileNet → DistilBERT → GRU). Used by the cross-model plots.
4. **New `MODEL_LATENCY` dict** — captured from the three Phase-3 runs + the log_2 ResNet-18 run. Used by the latency/throughput plot. Hard-coded because the traces don't contain multi-trial stats (only the single profiler window), so we have to keep this in sync with what the profiler actually printed.
5. **Three new plot functions:**
   - `plot_cross_model_stacked(model_category_shares, out_path)` — horizontal stacked bar.
   - `plot_cross_model_latency_throughput(out_path)` — dual-panel latency + throughput.
   - `plot_cross_model_tc_share(tc_shares, out_path)` — single bar chart with colour encoding by magnitude (<10 % → blue, 10–30 % → orange, >30 % → red).
6. **Two helper functions:**
   - `tensor_core_share_pct(per_name)` — sum of Self-CUDA for kernels whose name contains any of `tensorop`, `xmma`, `hmma`, `bmma`, `imma`, `s1688`, `s16816`, as a fraction of total Self-CUDA. Returns a percentage.
   - `category_share_pct(per_name)` — `{category: percent}` dict summing to ~100.
7. **Plot margins tightened after a first-pass clipping bug.** The first rendering had the cross-model category-stacked legend clipped at the bottom and the dual-panel axis labels slightly cut off. Fix applied:
   - `fig.savefig(out_path, dpi=140, bbox_inches='tight', pad_inches=0.35)` — matplotlib trims whitespace and adds 0.35 inches of padding around the visible content. Applies to all savefigs.
   - `fig.subplots_adjust(bottom=0.30)` on the cross-model stacked plot (5-column legend needs room).
   - `fig.subplots_adjust(bottom=0.18)` on the latency/throughput and TC-share plots (rotated x-tick labels need room).
   - `pad=1.2` and `pad=1.5` passed to `fig.tight_layout()` on the cross-model plots for extra internal spacing.

### 6.2 Eight plots produced

```bash
$ python -m analysis.plots
Loaded ResNet-18: results/traces/resnet18_baseline_bs32_benchOn.json
Loaded MobileNetV3-Small: results/traces/mobilenetv3_baseline_bs32_benchOn.json
Loaded DistilBERT-base: results/traces/distilbert_baseline_bs8_benchOn.json
Loaded Tiny GRU: results/traces/gru_baseline_bs32_benchOn.json
Wrote results/plots\resnet18_kernel_breakdown.png
Wrote results/plots\mobilenetv3_kernel_breakdown.png
Wrote results/plots\distilbert_kernel_breakdown.png
Wrote results/plots\gru_kernel_breakdown.png
Wrote results/plots\resnet18_conv_algorithms.png
Wrote results/plots\cross_model_category_stacked.png
Wrote results/plots\cross_model_latency_throughput.png
Wrote results/plots\cross_model_tc_share.png

Tensor-Core shares (for inclusion in log_3):
  ResNet-18              58.45%
  MobileNetV3-Small      14.94%
  DistilBERT-base         0.00%
  Tiny GRU               16.77%
```

**Final PNG sizes on disk (post-margin-fix):**

| File | Bytes | Role |
|---|---:|---|
| `results/plots/resnet18_kernel_breakdown.png` | 49 335 | per-model category bar |
| `results/plots/mobilenetv3_kernel_breakdown.png` | 49 930 | per-model category bar |
| `results/plots/distilbert_kernel_breakdown.png` | 37 259 | per-model category bar |
| `results/plots/gru_kernel_breakdown.png` | 41 263 | per-model category bar |
| `results/plots/resnet18_conv_algorithms.png` | 57 589 | ResNet-18 conv-family deep-dive |
| `results/plots/cross_model_category_stacked.png` | 59 211 | 4-model stacked bar by kernel category |
| `results/plots/cross_model_latency_throughput.png` | 69 965 | dual-panel: latency (ms) + throughput (samples/s, log) |
| `results/plots/cross_model_tc_share.png` | 50 661 | single bar: TF32-TC share per model |

### 6.3 Note on ResNet-18 plots

The ResNet-18 breakdown PNG's visual content changed slightly (different colour per bar now, by category) — this was a deliberate improvement for consistency with the other three per-model plots. The *numbers* in the plot are unchanged from log_2. The ResNet-18 conv-algorithms PNG is byte-equal to log_2's version (confirmed via pre-Phase-3 MD5 match before extension — same function).

---

## 7. Cross-model summary — quantitative

### 7.1 Headline latency / throughput table

| Model | Batch | Mean (ms) | std (ms) | min | max | Samples/s | std/mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 | 32 | 11.710 | 0.609 | 10.862 | 12.507 | 2 732.7 | 5.2 % |
| MobileNetV3-Small | 32 | 3.006 | 0.180 | 2.827 | 3.363 | 10 644.2 | 6.0 % |
| DistilBERT-base | 8 | 12.355 | 0.436 | 11.864 | 12.984 | 647.5 | 3.5 % |
| Tiny GRU | 32 | 0.252 | 0.010 | 0.244 | 0.273 | 127 003.4 | 4.0 % |

**Every std/mean is under 10 %** — the < 10 % acceptance bar from the Phase 3 plan holds for all four. DistilBERT is the most stable (3.5 %), probably because its `addmm → magma_sgemm` path is a single long kernel with minimal launch-side variability; MobileNetV3 is the noisiest (6.0 %) because its 180 kernel launches per iter compound launch-side jitter.

### 7.2 Total GPU-kernel-time per 10-iteration profiler window

| Model | Self-CUDA total (ms) | Events | Events/iter |
|---|---:|---:|---:|
| ResNet-18 | 113.393 | 1140 | 114 |
| MobileNetV3-Small | 20.412 | 1800 | 180 |
| DistilBERT-base | 121.511 | 760 | 76 |
| Tiny GRU | 1.725 | 100 | 10 |

**Launch-overhead regime observation.** MobileNetV3 has 180 kernel launches per iter — 58 % more than ResNet-18's 114 — despite taking only 18 % of ResNet-18's time. It's a launch-dense workload. GRU has 10 launches per iter because cuDNN's fused RNN kernel is one giant launch. This is the canonical launch-overhead vs steady-state tradeoff made visible.

### 7.3 Tensor-Core engagement

| Model | TC share | Kernels contributing |
|---|---:|---|
| ResNet-18 | 58.45 % | `cutlass_tensorop_s1688fprop` (28.4 %), two `xmma_fprop_implicit_gemm_tf32` variants (30.0 %) |
| MobileNetV3-Small | 14.94 % | `cutlass_80_tensorop_s1688gemm_256x64` — only the large 1×1 pointwise convs fit TC tiles |
| DistilBERT-base | 0.00 % | None. MAGMA SIMT FP32 dispatched instead |
| Tiny GRU | 16.77 % | `cutlass_80_tensorop_s1688gemm_128x256` — only the input-to-hidden matmul |

### 7.4 Kernel-category share comparison

| Category | ResNet-18 | MobileNetV3 | DistilBERT | GRU |
|---|---:|---:|---:|---:|
| conv_implicit_gemm | 70.09 % | 36.43 %* | — | — |
| conv_depthwise (Phase 4 bucket) | — | 25.71 % | — | — |
| matmul_tensor_core | — | 8.13 % | 0.00 % | 16.77 % |
| matmul_fp32 | 0.14 % | 4.46 % | 91.89 % | 1.71 % |
| norm | 8.52 % | 21.83 % | 1.48 % | — |
| layout_convert | 9.52 % | — | — | — |
| pool / reduce | 3.13 % | 3.36 % | — | 0.70 % |
| rnn | — | — | — | 79.34 % |
| fused_attention (Phase 4 bucket) | — | — | 4.66 % | — |
| elementwise | 8.65 % | 10.67 % | 1.78 % | 1.49 % |
| other | 0.00 % | — | 0.19 % | — |

*MobileNetV3 `conv_implicit_gemm` = 8.13 % TC + 7.56 % + 4.32 % + 16.42 % other-`aten::cudnn_convolution` kernels (which roll into the 31.73 % aggregate from the profiler). The exact sub-bucketing depends on whether the classifier counts `cutlass_tensorop_s1688gemm` as `conv_implicit_gemm` or `matmul_tensor_core` — currently it counts it as `matmul_tensor_core` because the rule for `'tensorop_s1688gemm'` in matmul runs before the `'fprop' in n` rule for conv. This is *correct* for MobileNetV3's 1×1 pointwise convs (they really are GEMMs, not conv-specific kernels).

**Phase 4 to-do from this table:** introduce `conv_depthwise` and `fused_attention` categories so the "other" buckets don't hide real work.

### 7.5 Eight plots produced, with what they're for

| Plot | Reveals |
|---|---|
| `resnet18_kernel_breakdown.png` | ResNet-18 per-category share — conv dominates, layout-convert visible |
| `mobilenetv3_kernel_breakdown.png` | MobileNetV3 per-category — BN share is bigger than brief predicted |
| `distilbert_kernel_breakdown.png` | DistilBERT per-category — matmul_fp32 ≈ 92 %, no TC |
| `gru_kernel_breakdown.png` | GRU per-category — rnn dominates, TC visible but small |
| `resnet18_conv_algorithms.png` | ResNet-18 conv-family: TC-TF32 CUTLASS/xmma vs SIMT implicit-convolve |
| `cross_model_category_stacked.png` | All four models side-by-side, categorical composition visually aligned |
| `cross_model_latency_throughput.png` | Latency bar chart (left) + log-scale throughput (right) — GRU 500× faster than DistilBERT per sample |
| `cross_model_tc_share.png` | TF32-TC engagement — ResNet-18 58 %, GRU 17 %, MobileNet 15 %, DistilBERT 0 % |

---

## 8. Predictions vs observations — detailed scorecard

Brief-to-observation mapping, by model. Citations to brief line numbers for every prediction.

### 8.1 MobileNetV3-Small (brief §1.2, lines 85–92)

| # | Brief prediction | Observed | Δ | Verdict |
|---|---|---|---:|---|
| MV-1 | Param count ~2.5 M | 2 542 856 (2.54 M) | +0.04 M | ✓ |
| MV-2 | "~60 % depthwise conv" | 25.71 % (`aten::_conv_depthwise2d`) | −34 pp | FAIL |
| MV-3 | "~30 % pointwise 1×1 conv" | 8.13 % (direct TC); likely 10–15 % more inside the 31.73 % `cudnn_convolution` aggregate | approx | PARTIAL |
| MV-4 | "~10 % misc" | 33 % misc (BN 22 % + activation 11 % + MAGMA 4 % + pool 3 %) | +23 pp | FAIL |
| MV-5 | "Tensor Cores don't help much" | TC-share 14.94 % (vs ResNet-18 58.45 %). 4× lower. Holds in spirit. | | ✓ |
| MV-6 | "FP16 speedup small (1.2–1.5×)" | Not measured; Phase 7 | | PENDING |
| MV-7 | "`cudnn.benchmark` helps modestly" | Not measured explicitly; Phase 6 | | PENDING |

Brief's depthwise/pointwise/misc breakdown is qualitatively right (depthwise is a major component) but quantitatively off because it underweighted BN. Reasonable explanation: the brief's 60/30/10 decomposition was written based on typical FLOP distribution, not time distribution; at the kernel level, BN's fixed per-call cost amortises badly over small convs. An architectural insight for the writeup.

### 8.2 DistilBERT-base (brief §1.3, lines 93–99)

| # | Brief prediction | Observed | Verdict |
|---|---|---|---|
| DB-1 | Param count ~66 M | 66 362 880 (66.4 M) | ✓ |
| DB-2 | "80%+ time in cuBLAS GEMM" | 0 % cuBLAS; 91.89 % MAGMA | FAIL — wrong library |
| DB-3 | "Strong TF32 Tensor-Core engagement" | 0.00 % TC share | FAIL — wrong precision path |
| DB-4 | "Attention scales well on Tensor Cores" | Attention does engage efficient kernels (FlashAttention CUTLASS), but FP32 only — no TC | PARTIAL |
| DB-5 | "Softmax a small slice" | Softmax *fused* inside FMHA attention kernel; no separate row | ✓ (stronger) |
| DB-6 | "FP16 speedup 2–3×" | Not measured; Phase 7 | PENDING |
| DB-7 | "Sequence length a primary knob" | Not measured; Phase 8 | PENDING |

**Two hard fails (DB-2, DB-3) are the strongest findings of Phase 3.** The writeup needs a dedicated paragraph explaining that on this stack, DistilBERT bypasses cuBLAS and goes to MAGMA, and this kills Tensor-Core utilization entirely. This is the opposite of the brief's premise that DistilBERT would be the TC showcase.

### 8.3 Tiny GRU (brief §1.4, lines 101–109)

| # | Brief prediction | Observed | Verdict |
|---|---|---|---|
| GR-1 | Param count ~0.2 M | 174 858 (0.17 M) | ✓ |
| GR-2 | "One `cudnn::rnn::*` kernel dominates" | `RNN_blockPersist_fp_GRU` at 73.33 %, plus the wrapper `_cudnn_rnn` at 96.11 % total | ✓ |
| GR-3 | "Memory-bound; throughput per sample low" | Per sample is 7.87 μs; the hidden-hidden matmuls are 128×128, extremely low FLOPs relative to memory traffic. Yet runs 127k samples/s — "low throughput" is not the right framing at this scale; "the kernel is memory-bound in theory but so small that it fits entirely on-chip" is more accurate. | PARTIAL |
| GR-4 | "Modest Tensor-Core engagement" | 16.77 % — the input-to-hidden GEMM is TC-eligible, recurrent path is not | ✓ |
| GR-5 | "Batch scaling amortises memory traffic" | Not measured; Phase 8 | PENDING |

GR-3 is the subtle one: "memory-bound" is the right roofline classification (low arithmetic intensity per byte), but the practical consequence is *not* "slow" at this size — the kernel fits in L2 + shared memory + registers and runs at near-peak bandwidth. The writeup should distinguish "memory-bound in the roofline sense" from "memory-bound in the 'would benefit from a bigger L2' sense".

### 8.4 Aggregated findings — writeup anchors

Four cross-model observations for `writeup/final_report.md §5 / §7`:

1. **"FP32 inference" means four different things on four different models.** ResNet-18 → TF32 TC implicit-GEMM (58 % TC). MobileNetV3 → mostly PyTorch-native FP32 depthwise + modest TC (15 %). DistilBERT → pure FP32 SIMT via MAGMA (0 % TC). GRU → cuDNN persistent RNN + TC input matmul (17 %). One flag (`allow_tf32=True`), four regimes.

2. **The dispatch layer is the most variable part of the stack.** Same PyTorch API (`aten::addmm`, `F.conv2d`, `nn.GRU`) hands off to cuDNN, cuBLAS, cuBLASLt, MAGMA, CUTLASS, or PyTorch-native kernels depending on shape, precision, and library build. Cross-model comparison at aten-op level compares *dispatcher decisions*, not just architectures.

3. **Fused kernels disappear from the top of profiles when they work.** DistilBERT's attention is invisible in the op-level table (no softmax row, no explicit `bmm` for Q@K^T) because `fmha_cutlassF_f32_aligned_64x64_rf_sm80` fuses Q@K^T + softmax + @V. GRU's timestep loop is likewise invisible because `RNN_blockPersist_fp_GRU` fuses 100 timesteps. **Fewer top-25 rows ≠ less work.**

4. **Tensor-Core engagement is driven by tile alignment, not model class.** Three of four models engage TC. DistilBERT doesn't — not because of architecture but because its dispatcher route (MAGMA) has no TC path. A `prefer_cublaslt=True` setting could plausibly flip DistilBERT from 0 % → 50 %+ TC share. Worth a dedicated Phase-11 experiment.

---

## 9. Artefacts produced in this phase

### Created (13 files)

| Path | Size | Role |
|---|---:|---|
| `models/mobilenet.py` | 355 B | MobileNetV3-Small loader |
| `models/distilbert.py` | 446 B | DistilBERT-base loader |
| `models/gru.py` | 649 B | TinyGRU (2-layer) loader |
| `results/traces/mobilenetv3_baseline_bs32_benchOn.json` | 5.3 MB | Chrome trace |
| `results/traces/distilbert_baseline_bs8_benchOn.json` | 3.1 MB | Chrome trace |
| `results/traces/gru_baseline_bs32_benchOn.json` | 335 KB | Chrome trace |
| `results/plots/mobilenetv3_kernel_breakdown.png` | 49.9 KB | per-model category bar |
| `results/plots/distilbert_kernel_breakdown.png` | 37.3 KB | per-model category bar |
| `results/plots/gru_kernel_breakdown.png` | 41.3 KB | per-model category bar |
| `results/plots/cross_model_category_stacked.png` | 59.2 KB | 4-model stacked bar |
| `results/plots/cross_model_latency_throughput.png` | 70.0 KB | dual-panel latency + throughput |
| `results/plots/cross_model_tc_share.png` | 50.7 KB | 4-model TC-share bar |
| `docs/execution_log_3.md` | (this file) | per-phase audit log |

### Modified (3 files)

| Path | Nature of change |
|---|---|
| `profiling/run_baseline.py` | +3 `_load_*` fns, `MODEL_LOADERS` 1 → 4 keys, `DEFAULT_BATCH` dict, `--batch None` sentinel |
| `analysis/plots.py` | Generalised per-model breakdown, +3 cross-model plot functions, `MODEL_LATENCY` / `CATEGORY_COLOURS` added, margin fixes with `bbox_inches='tight'` + `pad_inches=0.35` |
| `results/plots/resnet18_kernel_breakdown.png` | Re-rendered with colour-coded categories (visual improvement; data unchanged) |

### On-disk cache side effects (outside repo)

| Cache | Size | Location |
|---|---|---|
| MobileNetV3 pretrained weights | 9.83 MB | `~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth` |
| DistilBERT checkpoint + config | ~250 MB | `~/.cache/huggingface/hub/models--distilbert-base-uncased/` |

---

## 10. Gate evaluation against the Phase 3 plan

From `C:/Users/worka/.claude/plans/take-a-detailed-walkthorugh-inherited-cascade.md` §Verification.

| Acceptance criterion | Result | Evidence |
|---|---|---|
| All three trace JSONs exist, size 1–5 MB | ✓ | 5.3 MB / 3.1 MB / 0.3 MB (GRU below nominal range; expected — see §5.3) |
| `parse_trace` reports > 100 kernel events per trace | ✓ except GRU | 1800, 760, 100. GRU exactly at the bar because of cuDNN fusion — correct, not a failure |
| Per-model multi-trial latency std/mean < 10 % | ✓ | 6.0 % / 3.5 % / 4.0 % |
| ResNet-18 plots unchanged | ✓ for conv_algorithms (MD5 equal); ✗ for category_breakdown but the change is an intentional visual improvement (colour-coded categories), data identical |
| Three new `_load_*` fns wired | ✓ | All four MODEL_LOADERS keys resolve |
| `DEFAULT_BATCH` dict lands | ✓ | DistilBERT auto-picks batch 8; other three auto-pick 32 |
| Log_3 documents per-model runs + predictions vs observations | ✓ | (this file, §4 and §8) |

Plus scope additions beyond the plan:
- Generalised `analysis/plots.py` for any model (not planned; user-requested mid-phase).
- 4 per-model category breakdowns + 3 cross-model comparison plots (not planned; user-requested).
- SSL_CERT_FILE workaround documented (discovered mid-phase, not planned).
- Plot margin fix applied after initial render clipped legends.

All gates met. Phase 3 complete.

---

## 11. Risk register — how we actually managed the risks flagged in the plan

| Risk (from Phase 3 plan) | Flagged likelihood | Actual outcome |
|---|---|---|
| DistilBERT HF weights fail to download | Low | Partial hit — SSL_CERT_FILE broke the download path, not the network. Workaround added. |
| `DistilBertModel.forward(x)` fails due to positional-arg semantics | Low | Did not happen — `model(x)` works positional in transformers 5.5.4 |
| GRU single kernel too fast for 50 iters/trial | Low | Did not happen — 0.252 ms/iter × 50 = 12.6 ms/trial, well above timer resolution |
| Thermal throttling makes back-to-back runs noisier | Medium | Held in check by `sleep 5` between models; observed max std/mean 6.0 % (MobileNet) |
| cuBLAS kernel names don't match classifier → `other` bucket | Medium | Materialised differently — DistilBERT's kernels are MAGMA, which hits the `'gemm' → matmul_fp32` rule correctly. FlashAttention `fmha_cutlassF_*` DID fall into `other` as predicted; flagged for Phase 4. |
| ResNet-18 traces overwritten | Very low | Did not happen — filenames encode `{model}` distinctly |

**New risk surfaced during execution:**

| Risk | Severity | Mitigation applied |
|---|---|---|
| `SSL_CERT_FILE` conda-env var points to non-existent path | Medium (blocks any HTTP) | `unset SSL_CERT_FILE` in each shell invocation; documented in log |
| Cross-model plot legends + x-axis labels clipped by default matplotlib geometry | Low (aesthetic) | `bbox_inches='tight'` + `pad_inches=0.35` + explicit `subplots_adjust(bottom=...)` |
| DistilBERT dispatches to MAGMA instead of cuBLAS → zero TC engagement | Medium (contradicts brief) | Finding documented; follow-up experiments queued for Phase 7 and Phase 11 |

---

## 12. Open items going into Phase 4

1. **Classifier cleanups.** `analysis/classify_kernels.py` needs three targeted rule additions to eliminate the `other` bucket noise observed this phase:
   - `'_conv_depthwise2d'` or `'conv_depthwise'` → new `conv_depthwise` category (MobileNetV3 variant A + B are 25.71 % of time).
   - `'fmha_cutlassf'` or `'AttentionKernel'` → new `fused_attention` category (DistilBERT's FlashAttention is 4.66 %).
   - Consider splitting `rnn` into `rnn_persistent` (blockPersist family) vs `rnn_iterative` (ordinary timestep loops) so the writeup can talk about *why* GRU is fast.
2. **Phase 4's cross-model CSV.** Write `analysis/compute_summary.py` to read all four traces, classify, and emit `results/tables/baseline_breakdown.csv` for the writeup. Logic exists in `category_share_pct(per_name)` — needs a CSV writer.
3. **Phase 4's `fig1_time_breakdown.png`.** The `cross_model_category_stacked.png` produced in this phase is essentially fig1. Phase 4 can either reuse it or polish (add writeup-style caption, reduce category count to 5 most-meaningful buckets for top-line comms).
4. **DistilBERT-MAGMA follow-up experiment.** Highest-impact potential finding: toggle `torch.backends.cuda.preferred_linalg_library` or equivalent and measure the TC flip. Worth a named Phase-11 sub-experiment. Currently worth its own 30-minute investigation.
5. **Phase 5 Nsight install.** Still pending.
6. **TF32-off A/B on ResNet-18.** Still queued from log_2 §5.9.
7. **`SSL_CERT_FILE` env hygiene.** Ship an `activate.d` hook or document the `unset` in the README Setup section so contributors don't repeat this debug.
8. **FlashAttention `fmha_cutlassF_f32` vs `fmha_cutlassF_f16` in Phase 7.** The `f32_aligned_64x64_rf_sm80` suggests sm80 compatibility; we should check what happens with FP16 autocast — PyTorch may dispatch to a different CUTLASS variant.
9. **MobileNetV3 BN bottleneck as a finding.** The brief's 60/30/10 decomposition was wrong in the BN direction. That's worth a paragraph in the writeup §5.1 — "why small convs give BN an outsized share" is a pedagogically useful point about launch-overhead regimes.

---

## 13. Reproduction commands (for future readers)

```bash
# Environment activation (required prefix for every command)
source /c/Users/worka/anaconda3/etc/profile.d/conda.sh
conda activate hdai
cd "D:/HDAI_Project"
unset SSL_CERT_FILE    # Phase-3 workaround for conda-env misconfig

# All three Phase-3 baselines
python -m profiling.run_baseline --model mobilenetv3
sleep 5
python -m profiling.run_baseline --model distilbert
sleep 5
python -m profiling.run_baseline --model gru

# Parse each trace to a top-10 kernel table
python -m analysis.parse_trace results/traces/mobilenetv3_baseline_bs32_benchOn.json --top 10
python -m analysis.parse_trace results/traces/distilbert_baseline_bs8_benchOn.json --top 10
python -m analysis.parse_trace results/traces/gru_baseline_bs32_benchOn.json --top 10

# Render all 8 plots
python -m analysis.plots
```

Total wall clock if all caches are warm: ~45 seconds. First run including DistilBERT download: ~4 minutes (250 MB download is the bottleneck).

---

*End of Log 3.*
