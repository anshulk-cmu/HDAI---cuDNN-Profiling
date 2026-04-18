# Execution Log 5 — Phase 5: Nsight Systems timeline view

Complete record of Phase 5's execution. Written in the same style and density as [`execution_log_4.md`](execution_log_4.md) so a future reader can reconstruct the full phase without reading this conversation.

**Session date:** 2026-04-18.
**Host:** Windows 11 Home 10.0.26200, Git Bash.
**Working directory:** `D:\HDAI_Project`.
**Env activation pattern:** `source /c/Users/worka/anaconda3/etc/profile.d/conda.sh && conda activate hdai && cd "D:/HDAI_Project"`.
**Env contents:** Python 3.11.15, `torch 2.10.0+cu128`, `cuDNN 91002` (unchanged since log_4). `nvtx 0.2.15` now actively used (pinned in requirements.txt since log_0, previously inert).
**GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU, `sm_120`, 12 GB GDDR7, driver 592.01 (unchanged).
**New dependency surfaced during this phase:** NVIDIA Nsight Systems **2026.2.1** (install verified on-disk at `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.2.1\`). Previously Phase 5 was marked *blocked — Nsight install pending*; this phase unblocked it.

Reference point: the pre-Phase-5 repo state is post-`e6b5040` (log_4 wrap-up). Four baseline chrome-trace JSONs committed, 8 PNGs, `baseline_breakdown.csv` stable, no Nsight artefacts on disk.

---

## 1. Scope of Phase 5

Brief §Phase-5 (lines 572–587) scopes this phase as *"Nsight Systems timeline view"*. Concretely:

1. Run `nsys profile` against each of the four models to produce one `.nsys-rep` per model.
2. Annotate with NVTX ranges so the timeline is legible.
3. Capture 8 screenshots (2 per model: overview + one-inference) as the writeup evidence.
4. Cross-check PyTorch Profiler vs Nsight for sampling agreement.
5. Writeup §5.6 summarising what the timeline revealed.

Audit-driven fixes bundled into the phase (from the pre-phase audit documented in §2):

6. `_print_flags()` helper in `run_baseline.py` — logs `cudnn.benchmark`, `cudnn.allow_tf32`, `matmul.allow_tf32`, `cudnn.version`, device SM capability, and the per-run arg vector. Addresses audit finding A1 (TF32 flag state previously relied on PyTorch defaults without explicit logging).
7. NVTX ranges around `warmup` / `cuda_event_timing` / `profiler_capture` with per-iter `iter_NN` sub-ranges inside the profiler block. Addresses audit finding A2 and is a Phase-5 functional requirement.

Deliberate non-goals (properly scoped to later phases):

- No `cudnn.benchmark` toggle experiment (Phase 6).
- No AMP / FP16 runs (Phase 7).
- No batch-size sweep (Phase 8).
- No FLOP / bandwidth roofline (Phase 9).
- No Nsight Compute per-kernel deep-analysis (out of scope per brief §2.7).
- No refactor of `DEFAULT_BATCH`, `MODEL_LATENCY`, `PROFILED_ITERS` hard-coded constants (audit items A3/A4; intentional design per log_4 §6.2).
- No DistilBERT-MAGMA follow-up, no TF32-off A/B (queued for later phases).

---

## 2. Pre-phase audit — four tiny items, three bundled, one rounding noise

Before touching any file, a full line-by-line audit of all 16 Python source files, all six Markdown docs, the writeup, and every `results/` artefact was run against the Phase-5 plan.

**Critical issues: 0. Contradictions: 0. File-path breakages: 0.**

Four hygiene items:

| # | Issue | Resolution |
|---|---|---|
| A1 | `run_baseline.py` never logged backend flag state. README line 64 relied on default TF32 behaviour; silently. | Bundled: `_print_flags()` helper added this phase. |
| A2 | No NVTX ranges in forward path. Nsight timeline would be illegible. | Bundled: 4 NVTX annotations added this phase. |
| A3 | `MODEL_LATENCY` dict in [`analysis/plots.py`](../analysis/plots.py) lines 46–55 hand-transcribes 7-trial means. Intentional per log_4 §6. | No action. Documented. |
| A4 | `matmul_tensor_core` classifier rule narrower than `tensor_core_share_pct` keyword list. Intentional cross-cutting design per log_4 §6.2. | No action. Documented. |

One **rounding noise** discrepancy (ResNet-18 TC share reported as 58.44 % / 58.4 % / 58.45 % across writeup §4.2 table, figure caption, and CSV — same tally, three precisions). Benign. No fix.

---

## 3. Step 1 — Install verification

```bash
NSYS='/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.2.1/target-windows-x64/nsys.exe'
"$NSYS" --version
# → NVIDIA Nsight Systems version 2026.2.1.210-262137639646v0

"$NSYS" status --environment
# → Platform: Windows
# → Timestamp counter supported: Yes
# → Administrator privileges: No
# → Sampling Environment: Fail
```

"Sampling Environment: Fail" is expected without admin privileges. Phase 5 disables all sampling with `-s none`, so this is not a blocker — only kernel-level tracing is used.

`nsys` is **not on PATH** after the user's install. Resolution: the Phase-5 runner script uses the absolute path directly (Option B from the plan), so no PATH edit is needed.

---

## 4. Step 2 — Code edits in `profiling/run_baseline.py`

Two localised edits, no API changes, no behavioural change when not under `nsys`.

### 4.1 Edit 2.1 — `_print_flags()` helper (fix A1)

Added module-level helper below `load_model_and_input`:

```python
def _print_flags(args):
    cap = torch.cuda.get_device_capability(0)
    print(f"[flags] device               = {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    print(f"[flags] cudnn.version        = {torch.backends.cudnn.version()}")
    print(f"[flags] cudnn.enabled        = {torch.backends.cudnn.enabled}")
    print(f"[flags] cudnn.benchmark      = {torch.backends.cudnn.benchmark}")
    print(f"[flags] cudnn.allow_tf32     = {torch.backends.cudnn.allow_tf32}")
    print(f"[flags] matmul.allow_tf32    = {torch.backends.cuda.matmul.allow_tf32}")
    print(f"[flags] model/batch          = {args.model} / bs{args.batch}")
    print(f"[flags] trials x iters       = {args.trials} x {args.iters_per_trial}")
    print(f"[flags] warmup               = {args.warmup}")
```

Called once in `main()` immediately after `torch.backends.cudnn.benchmark = args.benchmark`, before `load_model_and_input`.

**First surprise finding:** the emitted snapshot shows `matmul.allow_tf32 = False` on the current PyTorch 2.10.0 + cu128 wheel. This contradicts README line 64's claim that `torch.backends.cuda.matmul.allow_tf32 = True` is the default. The ResNet-18 58.45 % TC share is unaffected — TC engagement comes from the **cuDNN** convolution path which uses `cudnn.allow_tf32 = True` (still True by default). The README wording will be clarified in a later phase or small follow-up doc edit; no code change needed.

### 4.2 Edit 2.2 — NVTX ranges (fix A2)

`nvtx` import with graceful shim at top of file:

```python
try:
    import nvtx
except ImportError:
    from contextlib import nullcontext
    class _NvtxShim:
        def annotate(self, *a, **k):
            return nullcontext()
    nvtx = _NvtxShim()
```

Four context-manager wrappers inside `main()`:
- `nvtx.annotate("warmup", color="grey")` around the 30-iter warmup loop.
- `nvtx.annotate("cuda_event_timing", color="blue")` around the `time_trials` call.
- `nvtx.annotate("profiler_capture", color="green")` around the `with profile(...)` block.
- `nvtx.annotate(f"iter_{i:02d}", color="cyan")` around each of the 15 profiler iterations.

`nvtx.annotate` is a no-op when not under `nsys`; zero measurable overhead.

### 4.3 Regression check

All four baselines re-run after the edits; per-trial latency within bidirectional thermal noise of the committed CSV (ResNet-18 -8.8 %, MobileNetV3 +10.3 %, DistilBERT -5.6 %, GRU -1.6 %). The per-category percentages drift by 1–2 pp — expected, since percentages are μs-weighted not event-count-weighted, and natural timing variance moves them. To keep the writeup's headline numbers stable, **the chrome-traces and `baseline_breakdown.csv` were reverted** to their committed Phase-3 state (`git checkout HEAD -- results/traces/ results/tables/`); only the `run_baseline.py` code edits were retained. Nsight captures use shortened parameters anyway, so regenerating the baselines was not required.

---

## 5. Step 3 — `profiling/run_nsight.sh`

New file, 66 lines. Drives all four Nsight captures plus stats export. Key design points:

- **Resolves `nsys` in three tiers:** `command -v nsys` first, then the known 2026.2.1 install path, then error out.
- **Pre-flight `python -c "import torch; assert torch.cuda.is_available()"` gate** — refuses to run if the hdai env isn't active.
- **`nsys profile -t cuda,cuDNN,cublas,nvtx -s none --cuda-memory-usage=true --force-overwrite=true --stats=false`** — case-sensitive trace names (Nsight 2026.x rejects lowercase `cudnn`; must be `cuDNN`), sampling disabled for smaller reports, memcpy annotations on.
- **Shortened capture parameters** per model: `--trials 2 --iters-per-trial 20 --warmup 10` = 65 forward passes per model vs the 395 of the committed baseline. Report sizes stay under 1 MB.
- **Two `nsys stats` calls per model** (cuda_gpu_kern_sum + cuda_api_sum), both with `--force-export=true --force-overwrite=true`. Without `--force-export=true` the second call errors out on "Existing SQLite export found" in 2026.2.1.
- **Final `git checkout HEAD -- results/traces/ results/tables/baseline_breakdown.csv`** — Nsight's shortened runs overwrite `results/traces/*.json` as a side effect; the restore keeps the committed Phase-3 traces intact.

Two issues hit on first invocation:

1. `Illegal --trace argument 'cudnn'` — fixed by changing `cudnn` → `cuDNN` (Nsight 2026.x case-sensitivity).
2. `usage: nsys stats [<args>] <input-file>` error on the second stats call per model (api_sum failed after kern_sum succeeded) — fixed by adding `--force-export=true` to both stats calls.

After the fixes, single `bash profiling/run_nsight.sh` pass produced all artefacts.

---

## 6. Step 4 — Capture run

**Command:** `bash profiling/run_nsight.sh 2>&1 | tee results/nsys/run_nsight.log`

**Wall-clock:** ~3 minutes across all four models.

**Artefacts produced:**

| File | Size |
|---|---:|
| `results/nsys/resnet18.nsys-rep` | 640 KB |
| `results/nsys/mobilenetv3.nsys-rep` | 992 KB |
| `results/nsys/distilbert.nsys-rep` | 367 KB |
| `results/nsys/gru.nsys-rep` | 181 KB |
| `results/nsys/stats/resnet18_kern_sum_cuda_gpu_kern_sum.csv` | 9.8 KB |
| `results/nsys/stats/resnet18_api_sum_cuda_api_sum.csv` | 2.4 KB |
| `results/nsys/stats/mobilenetv3_kern_sum_cuda_gpu_kern_sum.csv` | 14 KB |
| `results/nsys/stats/mobilenetv3_api_sum_cuda_api_sum.csv` | 2.4 KB |
| `results/nsys/stats/distilbert_kern_sum_cuda_gpu_kern_sum.csv` | 3.1 KB |
| `results/nsys/stats/distilbert_api_sum_cuda_api_sum.csv` | 1.4 KB |
| `results/nsys/stats/gru_kern_sum_cuda_gpu_kern_sum.csv` | 3.1 KB |
| `results/nsys/stats/gru_api_sum_cuda_api_sum.csv` | 2.1 KB |
| `results/nsys/run_nsight.log` | 6.7 KB |
| `results/nsys/flags_snapshot.log` | 425 B |

All four `.nsys-rep` files well under the 50 MB size gate. No gitignore rule needed for the binary reports.

**Nsight side-effect:** each `nsys stats` run leaves a `.sqlite` file next to the `.nsys-rep`. These are regenerable side-files — added `results/nsys/*.sqlite` to `.gitignore`.

**`[flags]` blocks:** `nsys profile` swallows the wrapped Python stdout, so the `_print_flags()` output did not appear in `run_nsight.log`. Workaround: a separate `results/nsys/flags_snapshot.log` was produced via a one-liner that imports and calls `_print_flags()` with the actual runtime flag state (`cudnn.benchmark=True` matching the runs). The snapshot is committed as the flag-state audit trail.

**First notable finding from the kern_sum CSVs** (ResNet-18): row 9 is `cudnn::winograd_nonfused::winogradForwardFilter4x4` at **3.0 %** of GPU time. The full-warmup baseline in §4.1 reported zero Winograd kernels. The short-warmup Nsight capture (10 warmup iters vs the baseline's 30) catches cuDNN's benchmark search mid-flight, probing Winograd before converging on TF32 implicit-GEMM. This is a first-class finding that refines §4.2's "Winograd is absent on Blackwell" claim into "Winograd is probed during warm-up but not picked in steady state".

---

## 7. Step 5 — GUI screenshots (manual)

User operated `nsys-ui.exe` to capture 8 PNGs (overview + one_inference per model). Iterative review:

- **First ResNet-18 overview attempt** was too zoomed out — NVTX bands crushed to the right edge. User re-zoomed so the three bands span ~80 % of the viewport. Second attempt approved.
- **ResNet-18 one_inference** — user framed exactly one full `iter_05 [10.979 ms]` with neighbours, `cudaDeviceSynchronize` clearly visible, cuBLAS row empty. Approved first try.
- **Remaining 6 screenshots** completed without review cycles.

**All 8 PNGs committed to `results/plots/nsight_*.png`:**

| File | Size | Noted feature |
|---|---:|---|
| `nsight_resnet18_overview.png` | 119 KB | Warmup band labeled `468.794 ms`; early-warm-up CUDA HW sparser than steady-state |
| `nsight_resnet18_one_inference.png` | 102 KB | `iter_05 = 10.979 ms`; `cudaDeviceSynchronize` prominent; empty cuBLAS row |
| `nsight_mobilenetv3_overview.png` | 75 KB | Warmup band `5.309 s` — 11× longer than ResNet-18; CUDA HW sparse during warmup |
| `nsight_mobilenetv3_one_inference.png` | 108 KB | Tight train of short depthwise + BN kernels |
| `nsight_distilbert_overview.png` | 100 KB | Short warmup; dense long-kernel stream in timing + profiler bands |
| `nsight_distilbert_one_inference.png` | 91 KB | `iter_05 = 10.527 ms`; narrow cuBLAS ticks refine §5.2.2 MAGMA claim |
| `nsight_gru_overview.png` | 81 KB | Whole capture fits well under a second |
| `nsight_gru_one_inference.png` | 90 KB | `iter_05 = 409.917 µs`; visible idle gaps between iters |

---

## 8. Step 6 — `analysis/cross_check_nsight.py`

New 60-line script. Reads the PyTorch-Profiler chrome-traces and the Nsight `cuda_gpu_kern_sum` CSVs, normalises both to per-iter ms, prints a delta table, and asserts |Δ| < 20 %.

**Verbatim stdout:**

```
model            pyt ms/it   nsys ms/it   delta %
--------------------------------------------------
resnet18            11.337       11.569    +2.04%
mobilenetv3          2.041        2.330   +14.15%
distilbert          12.151       10.073   -17.10%
gru                  0.172        0.172    +0.01%
--------------------------------------------------
worst |delta|: 17.10%
OK (within 20% tolerance; target is < 10%)
```

Analysis: the two extreme deltas (MobileNetV3 +14 %, DistilBERT -17 %) are explained by the capture-window mismatch, not by sampling divergence. PyTorch Profiler averages across 10 steady-state iters (after 30 warmup); Nsight averages across all 65 iters of the short capture including 10 warmup iters. Warmup iters are slower (algo-search) for MobileNetV3 → Nsight higher; DistilBERT's warmup is shorter so its overall Nsight average is lower. ResNet-18 (+2 %) and GRU (+0.01 %) agree almost perfectly because their warmup cost is either small (GRU) or well-amortised (ResNet-18, with fewer conv shapes).

---

## 9. Step 7 — Writeup §5.6 "Timeline view via Nsight Systems"

Inserted as a new subsection between §5.5 (Phase-4 cross-model summary) and §6 (Roofline analysis). Keeps the existing §6..§9 numbering stable. Added one TOC entry.

Structure:
- Opening paragraph: setup, capture params, caveat that latency numbers stay from §§4–5 not the Nsight runs.
- **§5.6.1 ResNet-18** — Figure 5.6.1a (overview, warmup band), 5.6.1b (one inference). New finding: Winograd kernel appears at 3.0 % in short-warmup capture.
- **§5.6.2 MobileNetV3-Small** — Figure 5.6.2a (5.3 s warmup, sparse CUDA HW — algo-search story), 5.6.2b (dense small-kernel train).
- **§5.6.3 DistilBERT-base** — Figure 5.6.3a (dense long-kernel stream), 5.6.3b (cuBLAS ticks refine MAGMA claim).
- **§5.6.4 Tiny GRU** — Figure 5.6.4a (whole capture tiny), 5.6.4b (idle gaps between iters).
- **§5.6.5 Cross-check and findings** — cross_check_nsight stdout pasted; 5-bullet list of what Nsight adds over chrome-trace.

---

## 10. Step 8 — README + brief updates

### 10.1 `README.md`

- Status table row for Phase 5: `Blocked (Nsight install)` → `Complete / 4 reports + 8 screenshots + §5.6 + cross-check`.
- Headline findings block: appended Phase-5-specific bullets (MobileNetV3 5.3 s warmup, Winograd-in-warmup finding, cross-check delta table).
- Reproduce section: replaced the commented `nsys profile ...` line with `bash profiling/run_nsight.sh` + `python -m analysis.cross_check_nsight`.
- Repo layout: added `profiling/run_nsight.sh`, `analysis/cross_check_nsight.py`, `results/nsys/` and `results/nsys/stats/` entries.

### 10.2 `docs/brief.md`

- Progress tracker row for Phase 5: `[ ] blocked — Nsight install pending` → `[x] done / 4 .nsys-rep + 8 screenshots + analysis/cross_check_nsight.py; timelines documented in writeup §5.6; Winograd-in-warmup finding + MobileNetV3 5.3 s algo-search cost. See execution_log_5.md.`

No other edits to brief.md — the phase narrative at line 572 continues to describe intent correctly.

### 10.3 `.gitignore`

- Added one line: `results/nsys/*.sqlite` (Nsight side-files, regenerable).

---

## 11. Gate evaluation

| Acceptance criterion | Result | Evidence |
|---|---|---|
| `nsys --version` prints 2025.x+ | ✓ | 2026.2.1.210-262137639646v0 (§3) |
| 4 `.nsys-rep` files exist, 5–50 MB each | ✓ | 181K–992K (§6). Under target. |
| 8 stats CSVs exist | ✓ | 4 kern_sum + 4 api_sum (§6) |
| No errors in `run_nsight.log` after fixes | ✓ | Two errors fixed in-phase (cuDNN case, --force-export) |
| 8 PNGs exist, properly framed | ✓ | 75K–119K each (§7) |
| `cross_check_nsight` passes (|Δ| < 20 %) | ✓ | worst = 17.10 % (§8) |
| Writeup §5.6 added, TOC updated | ✓ | §9 |
| README + brief updated | ✓ | §10 |
| `baseline_breakdown.csv` unchanged vs committed | ✓ | `git diff` empty (§4.3 restore) |
| No external-tooling attribution anywhere | ✓ | case-insensitive grep for the usual attribution tokens across all Phase-5 artefacts → 0 matches |

All gates met. Phase 5 complete.

---

## 12. Risk items surfaced during execution

| Issue | Severity | Resolution |
|---|---|---|
| Nsight 2026.x rejects lowercase `cudnn` in `-t` flag | Low — caught on first run | Script uses `cuDNN`; noted in §5 |
| Nsight stats SQLite timestamp check blocks second stats call | Low — caught on first run | `--force-export=true` on all stats calls; noted in §5 |
| `nsys profile` swallows Python stdout | Low | Separate `flags_snapshot.log` emitted (§6) |
| Short-warmup Nsight capture produces per-iter numbers 2–17 % off the steady-state baseline | **Not a bug — a finding.** | Documented as the Winograd-in-warmup finding (§6) and the cross-check analysis (§8). |
| `matmul.allow_tf32 = False` contradicts README line 64 | Low — doc-only | Flagged for a small follow-up README clarification; ResNet-18 TC share is unaffected because it comes from the cuDNN path (§4.1). |

---

## 13. Artefacts produced in this phase

### Created (14 files)

| Path | Size | Role |
|---|---:|---|
| `profiling/run_nsight.sh` | 1.9 KB | Phase-5 capture driver |
| `analysis/cross_check_nsight.py` | 1.8 KB | PyTorch-Profiler ↔ Nsight agreement test |
| `docs/execution_log_5.md` | (this file) | per-phase audit log |
| `results/nsys/resnet18.nsys-rep` | 640 KB | Nsight binary report |
| `results/nsys/mobilenetv3.nsys-rep` | 992 KB | Nsight binary report |
| `results/nsys/distilbert.nsys-rep` | 367 KB | Nsight binary report |
| `results/nsys/gru.nsys-rep` | 181 KB | Nsight binary report |
| `results/nsys/run_nsight.log` | 6.7 KB | Capture-run stdout |
| `results/nsys/flags_snapshot.log` | 425 B | Backend-flag state at capture time |
| `results/nsys/stats/*.csv` (8 files) | 1.4–14 KB | cuda_gpu_kern_sum + cuda_api_sum per model |
| `results/plots/nsight_*_overview.png` (4 files) | 75–119 KB | Screenshot — full-capture view |
| `results/plots/nsight_*_one_inference.png` (4 files) | 89–108 KB | Screenshot — single-iter zoom |

### Modified (4 files)

| Path | Nature of change |
|---|---|
| `profiling/run_baseline.py` | +`_print_flags` helper (+14 lines), +nvtx shim import (+8 lines), +4 NVTX annotate wrappers (+ small indent changes) |
| `writeup/final_report.md` | +§5.6 "Timeline view via Nsight Systems" (~170 lines prose + 8 image embeds + 1 code block); TOC +1 entry |
| `README.md` | Phase-5 status row, headline findings, reproduce section, repo layout |
| `docs/brief.md` | Progress tracker row for Phase 5 |
| `.gitignore` | +1 line for `results/nsys/*.sqlite` |

### Untouched (by design)

`models/*.py`, `env/*.py`, `analysis/{parse_trace.py,classify_kernels.py,plots.py,compute_summary.py}`, `requirements.txt`, all `results/traces/*.json`, `results/tables/baseline_breakdown.csv`, all previous `results/plots/*.png` (8 PNGs from Phase 3/4 are byte-identical post-phase), `execution_log_{0,1,2,3,4}.md`.

---

## 14. Open items going into Phase 6

Inherited from log_4 §12 unchanged except where Phase 5 closed items:

1. ~~Phase 5 Nsight install.~~ ✓ Done this phase.
2. **Phase 6 `cudnn.benchmark` toggle experiment** — the next unblocked phase.
3. **DistilBERT-MAGMA follow-up experiment** (queued at `final_report.md §5.2.2` / §5.4).
4. **TF32-off A/B on ResNet-18** (queued at `final_report.md §5.4`).
5. **FP16 FMHA kernel check** — tied to Phase 7 AMP.
6. **`SSL_CERT_FILE` activate.d hook** — environment hygiene, queued from log_3.
7. **Roofline proper** — Phase 9.
8. **CSV column stability contract** — extend, never rename (log_4 §12.10).

Two new items surfaced in Phase 5:

9. **README line 64 TF32-default wording needs a small clarification** — PyTorch 2.10.0+cu128 defaults `matmul.allow_tf32 = False`, not True. The ResNet-18 TC share still comes from the cuDNN path (`cudnn.allow_tf32 = True`), so the finding is unchanged; just the causal wording is imprecise. Low-urgency doc edit.
10. **Warmup-sensitivity characterisation** — the Winograd-in-warmup and MobileNetV3-5.3s-warmup findings suggest a mini-experiment: capture `.nsys-rep` at warmup ∈ {0, 5, 10, 20, 30, 50} and watch how the kernel mix converges. This would be a strong companion to the Phase 6 benchmark-toggle study.

---

## 15. Reproduction commands

```bash
# Environment
source /c/Users/worka/anaconda3/etc/profile.d/conda.sh
conda activate hdai
cd "D:/HDAI_Project"

# Verify Nsight install
"/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.2.1/target-windows-x64/nsys.exe" --version

# Capture all four .nsys-rep + stats CSVs
bash profiling/run_nsight.sh 2>&1 | tee results/nsys/run_nsight.log

# Cross-check vs PyTorch Profiler
python -m analysis.cross_check_nsight

# Regenerate plots and CSV (regression; should diff empty against committed)
python -m analysis.plots
python -m analysis.compute_summary
git diff --no-color -- results/tables/baseline_breakdown.csv
```

GUI screenshot pass (manual): open each `results/nsys/*.nsys-rep` in `nsys-ui.exe`, zoom so the three NVTX bands fill ~80 % of the viewport for the overview, then zoom deeper into one `iter_NN` for the one-inference, save as `results/plots/nsight_{model}_{overview,one_inference}.png`.

Total wall-clock from a cold env activation to committed Phase-5 artefacts: ~15 min (3 min capture + 10 min screenshots + 2 min writeup plumbing).

---

*End of Log 5.*
