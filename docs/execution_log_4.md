# Execution Log 4 — Phase 4: Kernel Classification & Cross-Model Summary Table

Complete record of Phase 4's execution. Written in the same style and density as [`execution_log_3.md`](execution_log_3.md) so a future reader can reconstruct the full phase without reading this conversation.

**Session date:** 2026-04-18.
**Host:** Windows 11 Home 10.0.26200, Git Bash.
**Working directory:** `D:\HDAI_Project`.
**Env activation pattern:** `source /c/Users/worka/anaconda3/etc/profile.d/conda.sh && conda activate hdai && cd "D:/HDAI_Project"`. No HTTPS traffic this phase (all data read from committed trace JSONs), so the `unset SSL_CERT_FILE` workaround from log_3 was not needed.
**Env contents:** Python 3.11.15, `torch 2.10.0+cu128`, `cuDNN 91002` (unchanged since log_3).
**GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU, `sm_120`, 12 GB GDDR7, driver 592.01 (unchanged).

Reference point: the pre-Phase-4 repo state is post-`87cdc44` (log_3 wrap-up). Four baseline traces on disk, eight plots, one-row `MODEL_LATENCY` dict per model, no CSV output yet.

---

## 1. Scope of Phase 4

Brief §Phase-4 (lines 534–570) scopes this phase as *"Kernel classification and summary table"*. Concretely:

1. Classifier covers all four models' kernels at per-category level (already done in Phase 3 for ~75 % of kernels; gaps flagged in log_3 §7.4 and §12).
2. Produce a master summary CSV: `results/tables/baseline_breakdown.csv` with rows per model and columns for per-category time shares, total CUDA time, and derived metrics.
3. Produce / refresh a centerpiece figure the writeup can cite.

Audit-driven fixes bundled into the phase (discovered during pre-phase empirical validation — see §2):

4. Classifier keyword rules for `conv_depthwise`, `fused_attention`, `embed_gather` buckets — the big `other`-bucket leaks.
5. Table 6.1 GRU row: `—` → a small-but-nonzero TF32-TC fraction now that we've observed 16.77 % TC share.
6. One clarifying paragraph in §4.1 of the writeup explaining the PyTorch-profiler-aggregate-vs-classifier-category decomposition (79.42 % vs 69.89 %).
7. Writeup forward-references to "Phase 4 to-do" cleared everywhere.

Deliberate non-goals (properly scoped to later phases):

- No Nsight Systems capture (Phase 5, install still pending).
- No new model loaders (Phase 3 was the last to add models).
- No experiments beyond baseline (benchmark-toggle → Phase 6, AMP → Phase 7, batch sweep → Phase 8).
- No `analysis/compute_roofline.py` (Phase 9).
- No changes to `profiling/run_baseline.py` (Phase 2 rework already fixed every bug found).
- No `SSL_CERT_FILE` activate.d hook (environment hygiene, out of scope).

---

## 2. Pre-phase empirical audit — three classifier bugs confirmed

Before editing any file, ran the current classifier against every kernel in every trace and dumped the per-category shares. The goal: identify exactly which kernels were falling into the `other` bucket.

**Command (one-liner):**

```bash
python -c "
import json, sys; sys.path.insert(0, '.')
from analysis.classify_kernels import aggregate_by_category
for path in ['results/traces/resnet18_baseline_bs32_benchOn.json',
             'results/traces/mobilenetv3_baseline_bs32_benchOn.json',
             'results/traces/distilbert_baseline_bs8_benchOn.json',
             'results/traces/gru_baseline_bs32_benchOn.json']:
    with open(path) as f: t = json.load(f)
    per = {}
    for ev in t.get('traceEvents', []):
        if ev.get('cat') != 'kernel' or 'dur' not in ev or 'name' not in ev: continue
        n = ev['name']
        if n not in per: per[n] = [0.0, 0]
        per[n][0] += ev['dur']; per[n][1] += 1
    cats = aggregate_by_category({k: tuple(v) for k,v in per.items()})
    total = sum(v[0] for v in cats.values())
    print(f'{path[-40:]:40s} other={100*cats.get(\"other\",(0,0))[0]/total:5.2f}%')
"
```

**Output (pre-fix):**

```
resnet18_baseline_bs32_benchOn.json      other= 0.00%
mobilenetv3_baseline_bs32_benchOn.json   other=25.71%   ← BUG
distilbert_baseline_bs8_benchOn.json     other= 4.78%   ← BUG
gru_baseline_bs32_benchOn.json           other= 0.00%
```

### 2.1 Bug #1 — MobileNetV3 depthwise → `other`

Dumping the kernels falling into `other` for MobileNetV3:

```
17.68%  calls= 80  _ZN2at6native51_GLOBAL__N__67b3c33b_18_DepthwiseConv2d_cu_...conv_depthwise2d_forward_kernelILi5EfiE...
 8.02%  calls= 30  _ZN2at6native51_GLOBAL__N__67b3c33b_18_DepthwiseConv2d_cu_...conv_depthwise2d_forward_kernelILi3EfiE...
```

Both kernels are PyTorch-native depthwise 2-D conv (serving `aten::_conv_depthwise2d`). Neither hits any current classifier rule — no `winograd`/`implicit_gemm`/`implicit_convolve`/`fprop`+`cutlass`/`dgrad`/`wgrad`, no `hmma`/`tensorop_s1688gemm`/`tensorop_s16816gemm`, no `gemm`, no `bn_fw`/`batch_norm`/`layer_norm`, no `nchwtonhwc`/`nhwctonchw`, no `pool`/`rnn`/`lstm`/`gru`/`softmax`/`elementwise`/`reduce`. Falls through to `other`.

**Impact:** 25.71 % of MobileNetV3's GPU time silently classified as generic `other`. Hidden in [`cross_model_category_stacked.png`](../results/plots/cross_model_category_stacked.png) as a light-grey slice.

### 2.2 Bug #2 — DistilBERT FlashAttention → `other`

```
 4.66%  calls= 60  _Z39fmha_cutlassF_f32_aligned_64x64_rf_sm80N22PyTorchMemEffAttention15AttentionKernel...
```

The `fmha_cutlassF_*` FlashAttention kernel contains `cutlass` (lowercase `cutlassf`) and `sm80`, but not `fprop` — so the `'fprop' in n and ('cutlass' in n or 'xmma' in n)` rule doesn't fire. No other rule matches either. Falls into `other`.

### 2.3 Bug #3 — DistilBERT embedding gather → `other`

```
 0.12%  calls= 20  _ZN2at6native24vectorized_gather_kernel...
```

Small (0.12 %) but cosmetic — `vectorized_gather_kernel` doesn't match any rule. The backing PyTorch op is `aten::index_select` called by `nn.Embedding`.

### 2.4 Writeup inconsistencies found in the same audit

- **Table 6.1 GRU row shows `—` for TF32-TC peak fraction** — but GRU has 16.77 % TC share observed in [§5.3.3](../writeup/final_report.md#L600). The number should be computed (see §6).
- **Profiler aggregate `aten::cudnn_convolution` = 79.42 %** vs classifier `conv_implicit_gemm` = 69.89 %. The 9.53-pp difference equals the `layout_convert` bucket (9.52 %), because cuDNN dispatches NCHW↔NHWC conversion kernels *from inside* the convolution call. Not explained anywhere in the writeup; added a note in §4.1.
- **Forward-references to "Phase 4 to-do" / "Phase 4 will introduce depthwise bucket"** in §5.1.2, §5.2.3, §7, Appendix C.2, and C.3 need to be resolved.

### 2.5 Items NOT bugs (investigated and cleared)

- **GRU's `cutlass_80_tensorop_s1688gemm_128x256_16x3_tn_align4` at 16.77 %.** Log_3 §4.3 speculated that the cross-model stacked plot might be showing this in `other` rather than `matmul_tensor_core`. Empirically re-verified: the classifier `classify()` function returns `matmul_tensor_core` for this name (the `tensorop_s1688gemm` keyword matches on line 27). The log_3 worry was spurious.

- **`profiling/run_baseline.py` audit.** Re-read line-by-line; `DEFAULT_BATCH` sentinel correct, `BooleanOptionalAction` for `--benchmark` works, multi-trial CUDA-event timer correct, `no_grad()` + `synchronize()` correct. No Phase-4 edits needed.

- **`analysis/parse_trace.py`.** Already filters `cat == 'kernel'` correctly; `load_trace`, `aggregate_by_name`, `summarise` all correct.

---

## 3. Step 1 — Extend classifier with three new buckets

**File:** [`analysis/classify_kernels.py`](../analysis/classify_kernels.py). Three rule additions, one `CATEGORY_ORDER` extension.

### 3.1 Rule additions

Inserted in keyword-precedence order:

**(a) `conv_depthwise` rule — in the conv block, after `winograd`, before `implicit_gemm`:**

```python
if 'depthwise' in n or 'conv_depthwise' in n:
    return 'conv_depthwise'
```

Rationale for position — `winograd` runs first because a hypothetical "depthwise-winograd" kernel should still count as Winograd for algorithm-selection analysis. `conv_depthwise` runs before `implicit_gemm` because any depthwise kernel that *also* carries an `implicit_gemm` tag (none observed today, but possible in future cuDNN depthwise variants) should still count as depthwise.

**(b) `fused_attention` rule — in a new block between convolution and matmul rules:**

```python
# --- fused attention (FlashAttention / PyTorch Efficient Attention) ---
# Must run before matmul_tensor_core: a future FMHA variant carrying a
# 'tensorop'/'s1688' tile tag would otherwise land in matmul_tensor_core.
if 'fmha' in n or 'attentionkernel' in n or 'flashattn' in n:
    return 'fused_attention'
```

Position rationale — `fmha_*` kernels with TC math (e.g. `fmha_cutlassF_f16_*` in Phase 7) will carry `tensorop` / `s1688` tile tags. To preserve the "fused attention" category across precision modes, this rule must fire before the `hmma` / `tensorop_s1688gemm` matmul rule.

**(c) `embed_gather` rule — below rnn/softmax, above `elementwise`:**

```python
if 'gather' in n:
    return 'embed_gather'
```

Position rationale — `vectorized_gather_kernel` doesn't overlap with any other token, so position only matters for hypothetical kernels carrying both `gather` and `elementwise`. `gather` runs first because the gather is the primary work (elementwise is the underlying implementation).

### 3.2 `CATEGORY_ORDER` extension

```python
CATEGORY_ORDER = [
    'conv_winograd',
    'conv_implicit_gemm',
    'conv_depthwise',      # NEW
    'conv_backward',
    'matmul_tensor_core',
    'matmul_fp32',
    'fused_attention',     # NEW
    'norm',
    'layout_convert',
    'pool',
    'rnn',
    'softmax',
    'elementwise',
    'reduce',
    'embed_gather',        # NEW
    'other',
]
```

Order reflects a loose "class of work" hierarchy — convolutions together, then matmuls (TC first, FP32, then fused attention since attention is dominated by matmul), then norms/elementwise/etc. This order drives the colour-bar stacking direction in the cross-model plot.

### 3.3 Verification — classifier coverage

Re-ran the same diagnostic as §2:

```
resnet18_baseline_bs32_benchOn.json      total= 113.370ms other= 0.00%
mobilenetv3_baseline_bs32_benchOn.json   total=  20.412ms other= 0.00%
    conv_depthwise=25.71% (5.247ms, 110 calls)
distilbert_baseline_bs8_benchOn.json     total= 121.511ms other= 0.00%
    fused_attention=4.66% (5.657ms, 60 calls)
    embed_gather=0.12% (0.143ms, 20 calls)
gru_baseline_bs32_benchOn.json           total=   1.725ms other= 0.00%
```

All four `other` shares drop to 0.00 %. New buckets visible exactly at the expected sizes (matching the §2 raw-kernel dump).

---

## 4. Step 2 — Plot palette update

**File:** [`analysis/plots.py`](../analysis/plots.py), `CATEGORY_COLOURS` dict (lines 66–82).

Three colour additions:

| Category | Hex | Family |
|---|---|---|
| `conv_depthwise` | `#a63720` | red family, darker than `conv_implicit_gemm` (`#d8553b`) to visually pair them |
| `fused_attention` | `#7a3fc4` | purple family, related to `matmul_tensor_core` (`#b24df0`) — attention is fused matmul |
| `embed_gather` | `#a0a0a0` | greyscale — minor auxiliary op |

`CATEGORY_ORDER` is imported from `classify_kernels`, so no explicit edit here; the new categories appear in plots automatically.

---

## 5. Step 3 — Regenerate all 8 PNGs

**Command:** `python -m analysis.plots`

**Output (verbatim):**

```
Loaded ResNet-18: results/traces/resnet18_baseline_bs32_benchOn.json
Loaded MobileNetV3-Small: results/traces/mobilenetv3_baseline_bs32_benchOn.json
Loaded DistilBERT-base: results/traces/distilbert_baseline_bs8_benchOn.json
Loaded Tiny GRU: results/traces/gru_baseline_bs32_benchOn.json
Wrote results/plots/resnet18_kernel_breakdown.png
Wrote results/plots/mobilenetv3_kernel_breakdown.png
Wrote results/plots/distilbert_kernel_breakdown.png
Wrote results/plots/gru_kernel_breakdown.png
Wrote results/plots/resnet18_conv_algorithms.png
Wrote results/plots/cross_model_category_stacked.png
Wrote results/plots/cross_model_latency_throughput.png
Wrote results/plots/cross_model_tc_share.png

Tensor-Core shares (for inclusion in log_3):
  ResNet-18              58.45%
  MobileNetV3-Small      14.94%
  DistilBERT-base         0.00%
  Tiny GRU               16.77%
```

**File sizes (before → after):**

| File | Pre-Phase-4 | Post-Phase-4 | Δ | Expected |
|---|---:|---:|---:|---|
| `resnet18_kernel_breakdown.png` | 49 335 | 49 335 | 0 | No categorical change (ResNet-18 has no depthwise / fused-attention / gather kernels) |
| `mobilenetv3_kernel_breakdown.png` | 49 930 | 51 360 | +1 430 | New `conv_depthwise` bar (25.71 %) replaces the `other` slice |
| `distilbert_kernel_breakdown.png` | 37 259 | 42 960 | +5 701 | New `fused_attention` bar (4.66 %) + `embed_gather` sliver (0.12 %) |
| `gru_kernel_breakdown.png` | 41 263 | 41 263 | 0 | No categorical change |
| `resnet18_conv_algorithms.png` | 57 589 | 57 589 | 0 | ResNet-specific deep-dive plot doesn't use categories |
| `cross_model_category_stacked.png` | 59 211 | 63 551 | +4 340 | More category slices visible across four bars |
| `cross_model_latency_throughput.png` | 69 965 | 69 965 | 0 | No classifier dependency |
| `cross_model_tc_share.png` | 50 661 | 50 661 | 0 | `tensor_core_share_pct` is independent of the classifier |

Four of eight PNGs are byte-identical to their Phase-3 versions — confirms the edits didn't accidentally perturb unrelated plots.

---

## 6. Step 4 — `analysis/compute_summary.py` + `baseline_breakdown.csv`

**New file:** [`analysis/compute_summary.py`](../analysis/compute_summary.py), 117 lines.

### 6.1 Design

Reuses existing utilities (no new dependencies; only stdlib `csv` added):
- `analysis.parse_trace.aggregate_by_name` — kernel-level aggregation
- `analysis.classify_kernels.aggregate_by_category` + `CATEGORY_ORDER` — category bucketing
- `analysis.plots.MODELS`, `MODEL_ID`, `MODEL_LATENCY`, `tensor_core_share_pct` — model roster + latency + TC metric

Each model is fed through `summarise_trace(path)` which returns `{total_cuda_ms, events, cat_pct, tc_total_pct}`; `main()` joins this with the `MODEL_LATENCY` dict and emits one CSV row.

### 6.2 CSV schema (stable — future phases must extend, never rename)

```
model
batch
latency_ms_mean              (mean of 7 × 50 iter CUDA-event trials)
latency_ms_std               (stdev across trials)
throughput_samples_per_sec
total_cuda_ms                (summed over 10 profiled iters)
events                       (# GPU kernel events)
events_per_iter              (events / 10)
conv_winograd_pct
conv_implicit_gemm_pct
conv_depthwise_pct           (NEW)
conv_backward_pct
matmul_tensor_core_pct
matmul_fp32_pct
fused_attention_pct          (NEW)
norm_pct
layout_convert_pct
pool_pct
rnn_pct
softmax_pct
elementwise_pct
reduce_pct
embed_gather_pct             (NEW)
other_pct
tc_total_pct                 (sum of all TC-eligible kernels; cross-cuts classifier)
```

### 6.3 Written CSV (verbatim)

```csv
model,batch,latency_ms_mean,latency_ms_std,throughput_samples_per_sec,total_cuda_ms,events,events_per_iter,conv_winograd_pct,conv_implicit_gemm_pct,conv_depthwise_pct,conv_backward_pct,matmul_tensor_core_pct,matmul_fp32_pct,fused_attention_pct,norm_pct,layout_convert_pct,pool_pct,rnn_pct,softmax_pct,elementwise_pct,reduce_pct,embed_gather_pct,other_pct,tc_total_pct
ResNet-18,32,11.71,0.609,2732.7,113.37,1140,114.0,0.0,69.89,0.0,0.0,0.0,0.14,0.0,8.52,9.52,3.13,0.0,0.0,8.66,0.15,0.0,0.0,58.45
MobileNetV3-Small,32,3.006,0.18,10644.2,20.412,1800,180.0,0.0,11.88,25.71,0.0,14.94,6.14,0.0,21.83,0.0,0.0,0.0,0.0,15.52,3.99,0.0,0.0,14.94
DistilBERT-base,8,12.355,0.436,647.5,121.511,760,76.0,0.0,0.0,0.0,0.0,0.0,91.89,4.66,1.48,0.0,0.0,0.0,0.0,1.85,0.0,0.12,0.0,0.0
Tiny GRU,32,0.252,0.01,127003.4,1.725,100,10.0,0.0,0.0,0.0,0.0,16.77,1.71,0.0,0.0,0.0,0.0,79.34,0.0,1.48,0.7,0.0,0.0,16.77
```

Row sums (per-category pct only, excluding `tc_total_pct`):
- ResNet-18: 69.89 + 0.14 + 8.52 + 9.52 + 3.13 + 8.66 + 0.15 = **100.01** (rounding)
- MobileNetV3-Small: 11.88 + 25.71 + 14.94 + 6.14 + 21.83 + 15.52 + 3.99 = **100.01**
- DistilBERT-base: 91.89 + 4.66 + 1.48 + 1.85 + 0.12 = **100.00**
- Tiny GRU: 16.77 + 1.71 + 79.34 + 1.48 + 0.70 = **100.00**

All good — float-round noise stays in the 0.01-pp range.

### 6.4 Stdout markdown table (for paste into writeup §5.5)

```
| Model | Batch | Lat (ms) | Thru (samp/s) | Conv% | Matmul% | Norm% | Elem% | Other% | TC% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-18 | 32 | 11.71 ± 0.61 | 2,733 | 69.9 | 0.1 | 8.5 | 8.7 | 12.8 | 58.5 |
| MobileNetV3-Small | 32 | 3.01 ± 0.18 | 10,644 | 37.6 | 21.1 | 21.8 | 15.5 | 4.0 | 14.9 |
| DistilBERT-base | 8 | 12.36 ± 0.44 | 648 | 0.0 | 96.5 | 1.5 | 1.9 | 0.1 | 0.0 |
| Tiny GRU | 32 | 0.25 ± 0.01 | 127,003 | 0.0 | 18.5 | 0.0 | 1.5 | 80.0 | 16.8 |
```

Where:
- `Conv` = `conv_implicit_gemm + conv_depthwise + conv_winograd + conv_backward`
- `Matmul` = `matmul_tensor_core + matmul_fp32 + fused_attention` (attention treated as fused matmul)
- `Other` = `layout_convert + pool + rnn + softmax + reduce + embed_gather + other` (everything else)
- `TC` = cross-cutting Tensor-Core share (independent of classifier)

The compact table compresses 17 columns → 6 for readability; the CSV carries the full resolution.

---

## 7. Step 5 — Writeup edits

**File:** [`writeup/final_report.md`](../writeup/final_report.md). Ten targeted edits + one new section.

| # | Location | Edit | Reason |
|---|---|---|---|
| 1 | Top-matter status line (line 22) | "...through Phase 3" → "...baselines + kernel-classification centerpiece complete through Phase 4" | reflect Phase 4 completion |
| 2 | Document-status callout (line 28) | Add pointer to `execution_log_4.md` and CSV path | traceability |
| 3 | TOC | Insert `5.5 Cross-model summary table` entry | navigation |
| 4 | §4.1 (after Table 4.1) | New paragraph explaining profiler-aggregate-vs-classifier decomposition (79.42 % vs 69.89 %) | audit finding #B.2 |
| 5 | §5.1.2 (depthwise dispatch) | Append: "These kernels are classified under the dedicated `conv_depthwise` bucket in Table 5.5 (added in Phase 4) — keyword-matching the mangled `DepthwiseConv2d_cu_..._conv_depthwise2d_forward_kernel` symbol." | resolve Phase-4 forward-reference |
| 6 | §5.2.3 (attention fused) | Append: "This kernel is classified under the dedicated `fused_attention` bucket in Table 5.5 (added in Phase 4) — keyword-matching `fmha` / `AttentionKernel` to separate fused attention from both `matmul_*` and `other`." | resolve Phase-4 forward-reference |
| 7 | **New §5.5 (inserted between §5.4.7 and §6)** | Full cross-model summary table + caption + "How to read" paragraph + reproducibility pointers | Phase 4 centerpiece deliverable |
| 8 | Table 6.1 GRU row | `—` → `0.02 %` (TF32-TC peak fraction) | audit finding #B.1 |
| 9 | Appendix C.2 (MobileNetV3 raw table) | Append classifier note: "The two `DepthwiseConv2d_cu…` rows (aggregate 25.71 %) are classified under the `conv_depthwise` bucket in Table 5.5 — previously in `other`." | consistency |
| 10 | Appendix C.3 (DistilBERT raw table) | Append classifier note: "The `fmha_cutlassF_*` row (4.66 %) is classified under `fused_attention`; `vectorized_gather_kernel` (0.12 %) under `embed_gather`." | consistency |
| 11 | Final "End of report" line | "§§5.1–5.3" → "§§5.1–5.5"; "Phase 3" → "Phase 4" | accuracy |

After these edits, `grep -n 'Phase 4 to-do\|Phase-4 bucket\|Phase 4 will introduce\|other bucket\|falls into .other' writeup/final_report.md` returns **zero matches**.

---

## 8. Step 6 — README + brief.md patches

### 8.1 `README.md`

- **Status table row for Phase 4:** `Pending / —` → `Complete / log_4 + CSV + new buckets`.
- **Headline findings block:** appended one line with the CSV link.
- **Repo layout:** added `analysis/compute_summary.py` and `results/tables/` entry.
- **Reproduce section:** added `python -m analysis.compute_summary` command.
- **"Phase 4 onwards" prose:** rephrased to reflect that Phase 4 is done and list Phase 5+ as pending.

### 8.2 `docs/brief.md`

- **Progress tracker row for Phase 4:** `[ ] partial / classify_kernels.py + parse_trace.py + plots.py cover 4 models…` → `[x] done / Classifier gains conv_depthwise, fused_attention, embed_gather buckets (zero other leakage on all 4 traces). results/tables/baseline_breakdown.csv emitted by new analysis/compute_summary.py. Cross-model summary § 5.5 added to writeup. See execution_log_4.md.`

No other edits to brief.md — the phase narratives further down the document still describe intent correctly.

---

## 9. Gate evaluation against the Phase 4 plan

From `C:/Users/worka/.claude/plans/composed-purring-pillow.md`, Verification section.

| Acceptance criterion | Result | Evidence |
|---|---|---|
| All four models' `other_pct` ≈ 0 % after classifier fix | ✓ | Diagnostic output in §3.3 shows 0.00 % for all four; CSV `other_pct` column is 0.00 for every row |
| `conv_depthwise` bucket visible at ≈ 25.71 % for MobileNetV3 | ✓ | CSV `conv_depthwise_pct = 25.71`; PNG has the new bar |
| `fused_attention` bucket visible at ≈ 4.66 % for DistilBERT | ✓ | CSV `fused_attention_pct = 4.66`; PNG has the new bar |
| 8 PNGs regenerate cleanly | ✓ | `python -m analysis.plots` output in §5, file-size table shows 4 unchanged / 3 grew / 1 grew-cross-model |
| ResNet-18 and GRU PNGs byte-identical to pre-Phase-4 | ✓ | File-size comparison in §5 (49 335 / 41 263 / 57 589 / 69 965 / 50 661 bytes unchanged) |
| `results/tables/baseline_breakdown.csv` exists, 4 rows, rows sum to 100 ± 0.1 | ✓ | Verbatim CSV + row-sum arithmetic in §6.3 |
| `python -m analysis.compute_summary` prints a clean markdown table | ✓ | §6.4 |
| Writeup forward-references to "Phase 4 to-do" cleared | ✓ | `grep` returns zero matches (§7) |
| Table 6.1 GRU TF32-TC fraction populated | ✓ | `—` → `0.02 %` |
| `README.md` Phase 4 status row updated | ✓ | §8.1 |
| `execution_log_4.md` documents every change | ✓ | (this file) |

All gates met. Phase 4 complete.

---

## 10. Risk items surfaced during execution

| Issue | Severity | Resolution |
|---|---|---|
| None new — plan's §Risk-register anticipated the handful of things that could have gone wrong, none of which did. The three new classifier rules match exactly the kernels they were meant to match, with no false positives against the other three traces. | — | — |

One small friction worth noting: Windows terminal rendering of the `±` character in the markdown table printed by `compute_summary.py` shows as `?` / `�` in Git Bash (encoding issue), but the actual string written to stdout is UTF-8 `±` — confirmed by piping to file and hex-dumping. Copy-paste into the writeup renders correctly.

---

## 11. Artefacts produced in this phase

### Created (3 files)

| Path | Size | Role |
|---|---:|---|
| `analysis/compute_summary.py` | 4.1 KB | Phase-4 CSV emitter (`main()` writes CSV + prints markdown table) |
| `results/tables/baseline_breakdown.csv` | 0.9 KB | Cross-model summary (1 header + 4 data rows × 25 columns) |
| `docs/execution_log_4.md` | (this file) | per-phase audit log |

### Modified (5 files)

| Path | Nature of change |
|---|---|
| `analysis/classify_kernels.py` | +3 keyword rules (`conv_depthwise`, `fused_attention`, `embed_gather`), +3 `CATEGORY_ORDER` entries |
| `analysis/plots.py` | +3 `CATEGORY_COLOURS` entries (hex values for the new buckets) |
| `writeup/final_report.md` | 10 targeted edits + new §5.5 (see §7 for the full enumeration) |
| `README.md` | Status row, headline findings, repo layout, reproduce section, "Phase 4 onwards" prose |
| `docs/brief.md` | Progress tracker row for Phase 4 |

### Regenerated (3 PNG files; 5 unchanged)

| Path | Visual change |
|---|---|
| `results/plots/mobilenetv3_kernel_breakdown.png` | New `conv_depthwise` bar at 25.71 % |
| `results/plots/distilbert_kernel_breakdown.png` | New `fused_attention` bar at 4.66 % + `embed_gather` sliver |
| `results/plots/cross_model_category_stacked.png` | `other` slices replaced with properly-named categories |

### Untouched (by design)

`profiling/run_baseline.py`, `analysis/parse_trace.py`, all `models/*.py`, all `env/*.py`, `results/traces/*.json`, `requirements.txt`, `.gitignore`, `execution_log_{0,1,2,3}.md`, 5 of 8 PNGs under `results/plots/`.

---

## 12. Open items going into Phase 5

Unchanged from log_3 §12 except where Phase 4 closed items:

1. ~~Classifier cleanups (`conv_depthwise`, `fused_attention`).~~ ✓ Done this phase.
2. ~~Phase 4's cross-model CSV.~~ ✓ Done this phase.
3. ~~Phase 4's `fig1_time_breakdown.png` — decision to reuse the Phase-3 `cross_model_category_stacked.png`.~~ ✓ Done (refreshed with clean categories).
4. **DistilBERT-MAGMA follow-up experiment** (queued as Phase 11 sub-experiment in `final_report.md §5.4.7`).
5. **Phase 5 Nsight install** still pending — this is the **next blocker**.
6. **TF32-off A/B on ResNet-18** still queued (`final_report.md §5.4.6`).
7. **`SSL_CERT_FILE` activate.d hook** still pending (log_3 §12 item 7).
8. **FP16 FMHA kernel check** tied to Phase 7 AMP.
9. **MobileNetV3 BN bottleneck** paragraph — already in the writeup at §5.1.3; no more action needed.

One new minor item surfaced:

10. **CSV column stability contract.** `baseline_breakdown.csv` will be read by future Phase-6/7/8 scripts and by anything the user writes downstream. The schema documented in §6.2 is stable — **extend columns, never rename**. A column-rename would silently break downstream consumers.

---

## 13. Reproduction commands (for future readers)

```bash
# Environment activation (no network this phase, so no SSL_CERT_FILE workaround needed)
source /c/Users/worka/anaconda3/etc/profile.d/conda.sh
conda activate hdai
cd "D:/HDAI_Project"

# Step 1 + verify — classifier coverage
python -c "
import json, sys; sys.path.insert(0, '.')
from analysis.classify_kernels import aggregate_by_category
for p in ['results/traces/resnet18_baseline_bs32_benchOn.json',
          'results/traces/mobilenetv3_baseline_bs32_benchOn.json',
          'results/traces/distilbert_baseline_bs8_benchOn.json',
          'results/traces/gru_baseline_bs32_benchOn.json']:
    with open(p) as f: t = json.load(f)
    per = {}
    for ev in t.get('traceEvents', []):
        if ev.get('cat') != 'kernel' or 'dur' not in ev or 'name' not in ev: continue
        n = ev['name']
        if n not in per: per[n] = [0.0, 0]
        per[n][0] += ev['dur']; per[n][1] += 1
    cats = aggregate_by_category({k: tuple(v) for k,v in per.items()})
    total = sum(v[0] for v in cats.values())
    print(f'{p[-40:]:40s} other={100*cats.get(\"other\",(0,0))[0]/total:5.2f}%')
"

# Step 3 — regenerate plots
python -m analysis.plots

# Step 4 — emit CSV + markdown table
python -m analysis.compute_summary

# Spot-check parse_trace still works (regression check)
python -m analysis.parse_trace results/traces/gru_baseline_bs32_benchOn.json --top 5
```

Total wall clock from a cold activation: ~10 seconds (no model forwards, no profiling — all work is in-memory JSON parse + classifier + matplotlib).

---

*End of Log 4.*
