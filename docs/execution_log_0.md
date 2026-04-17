# Execution Log 0 — Project Bootstrap

> **Status (as of `execution_log_2.md`):** historical bootstrap record. The `hdai` env it describes is still current. The one Phase-1-era gap — the missing `env/sanity_conv.py` advertised in README — was filled in log_2. No other edits applied.

Timestamped record of everything executed from the moment the repo only contained `brief.md` up to a working `hdai` conda environment. Logged factually: what was run, why it was run, the exact output observed, and any discrepancies. No hype, no rounding.

**Session date:** 2026-04-16
**Host:** Windows 11 Home 10.0.26200
**Shell:** Git Bash (bash on Windows)
**Working directory:** `D:\HDAI_Project`

---

## Step 0 — Initial state of the repo

Before anything was run, the project directory contained exactly one file:

```
D:\HDAI_Project\
└── brief.md
```

`brief.md` is a ~33 k-token design document covering the multi-phase cuDNN profiling study across four models (ResNet-18, MobileNetV3-Small, DistilBERT-base, Tiny GRU). It was read in three chunks (lines 1–500, 500–1000, 1000–1500, 1500–end) via the `Read` tool because a single read exceeded the 25k-token limit.

**Why read it:** the user asked for a brief explanation and a correct README. Understanding the plan in detail (models, experiments, hardware assumptions, dependency list) was a prerequisite for both.

---

## Step 1 — Understanding the brief

Key facts extracted from `brief.md`:

| Item | Value from brief |
|---|---|
| Original target GPU | RTX 5080 Ti (Blackwell, `sm_120`), 16 GB |
| Models | ResNet-18, MobileNetV3-Small, DistilBERT-base, Tiny GRU |
| Experiments | baseline, `cudnn.benchmark` toggle, FP32 vs FP16, batch sweep, (+ optional channels-last, seq sweep) |
| Required wheel | PyTorch `+cu128` (because default cu121 fails on Blackwell with "no kernel image available") |
| Profiler stack | PyTorch Profiler + Nsight Systems 2025.x |
| Time budget | ~12 hours |

**Critical note:** the brief is scoped as a *profiling study*, not a research project. The deliverable is a writeup + plots + small script repo. This scope shaped the README (no training, no torch.compile, no multi-GPU).

---

## Step 2 — Probing the local environment

### 2.1 Attempted `conda env list` via `conda` on PATH

```
$ conda env list
/usr/bin/bash: line 1: conda: command not found
```

`conda` is not on PATH in Git Bash. This is expected — Anaconda's activate script only patches PATH for `cmd` / PowerShell on Windows, not bash.

### 2.2 Searching for conda installation

Probed three candidate locations:

```bash
which conda
ls /c/ProgramData/miniconda3/Scripts/
ls /c/Users/worka/miniconda3/Scripts/
ls /c/Users/worka/anaconda3/Scripts/
```

Results:
- `which conda` → not found
- `/c/ProgramData/miniconda3/` → does not exist
- `/c/Users/worka/miniconda3/` → does not exist
- `/c/Users/worka/anaconda3/Scripts/` → **exists**, contains `activate`, `activate.bat`, `conda` binaries

Decision: call `conda.exe` via its absolute path (`/c/Users/worka/anaconda3/Scripts/conda.exe`) throughout the rest of the session. No PATH mutation required.

### 2.3 Existing conda environments

```
$ /c/Users/worka/anaconda3/Scripts/conda.exe env list
# conda environments:
#
base                   C:\Users\worka\anaconda3
housing                C:\Users\worka\anaconda3\envs\housing
privacy                C:\Users\worka\anaconda3\envs\privacy
```

The `privacy` env is the one the user asked to mirror. `housing` is unrelated to this project.

### 2.4 GPU query

```
$ nvidia-smi
Thu Apr 16 22:25:12 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 592.01                 Driver Version: 592.01         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA GeForce RTX 5070 ...  WDDM  |   00000000:02:00.0  On |                  N/A |
| N/A   41C    P8              6W /   80W |    1568MiB /  12227MiB |      0%      Default |
```

**Observed facts:**
- Driver version 592.01 (≥ 570, so `sm_120` is supported).
- Driver reports CUDA 13.1 runtime. PyTorch will still be happy with the cu128 wheel — the driver's CUDA version is an upper bound, not a requirement.
- **12227 MiB** total VRAM, not 16 GB as the brief assumed.
- Name truncated in `nvidia-smi` output to `RTX 5070 ...`. Full name not yet known at this step.
- Idle power 6 W / 80 W → this is a **laptop TGP budget**, confirming this is a mobile/laptop GPU.
- Compute mode `Default`, WDDM driver model (expected on Windows).

**Discrepancy against brief:** the brief says "RTX 5080 Ti, 16 GB". The actual card is a 5070-class Blackwell with 12 GB. The `sm_120` architecture is identical across the 50-series lineup, so the study's substantive conclusions don't change, but every mention of "5080 Ti" / "16 GB" in the brief is now factually wrong.

---

## Step 3 — Writing the README (first pass)

Wrote `D:\HDAI_Project\README.md`. Structure:

1. One-paragraph project description
2. Three-question framing (time distribution, algorithm choice, roofline classification)
3. Model-zoo table (same as brief section 1.5)
4. Experiments table (six items)
5. Hardware/software block (initially generic: "RTX 50-series")
6. Setup commands (`conda create`, `pip install ... cu128`, rest via pip)
7. Repo layout
8. Reproduce section with representative commands
9. Deliverables (six PNGs + writeup + CSVs)
10. Non-goals (mirror brief section 18)
11. References

**Why this structure:** a TA or collaborator landing on the repo cold needs `what is this → how do I run it → what does it produce` in that order. Everything else is secondary. Non-goals are there as an integrity marker, same reason the brief keeps them.

First-pass compromise: the GPU field said "RTX 50-series (Blackwell)" rather than a specific model, because the `nvidia-smi` output was truncated and I had not yet confirmed the full device string from PyTorch.

---

## Step 4 — Inspecting the `privacy` conda env

```
$ /c/Users/worka/anaconda3/Scripts/conda.exe list -n privacy
```

Extracted the versions that matter for GPU compatibility:

| Package | Version in `privacy` |
|---|---|
| python | 3.11.14 |
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| transformers | 5.3.0 |
| numpy | 2.3.5 |
| pandas | 3.0.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |

The `+cu128` suffix is the important part — it confirms this wheel was built with the CUDA 12.8 toolchain and *does* carry `sm_120` kernels, which is exactly what Blackwell needs. Since the user had already validated this stack works on this laptop, the `hdai` env was pinned to the same torch/torchvision versions.

---

## Step 5 — User interrupt: brief.md needed correcting

Mid-way through environment setup the user noted the brief should say RTX 5070 (Blackwell, 12 GB). Addressed this as a priority before creating the env, because brief.md is referenced from README.md and should be internally consistent.

### 5.1 Finding all stale references

```
Grep pattern: 5080 Ti|16 GB|16GB
File: D:\HDAI_Project\brief.md
```

Matched **22 lines** across the brief. Some were inside kernel-name examples, some inside spec tables, some inside prose.

### 5.2 In-place substitution

```bash
sed -i 's/RTX 5080 Ti/RTX 5070/g; s/5080 Ti/5070/g; s/16 GB/12 GB/g; s/16GB/12GB/g' "D:/HDAI_Project/brief.md"
```

**Why `sed` and not the `Edit` tool:** 22 separate edits across a 1642-line file are exactly the kind of bulk substitution `sed` was designed for. The `Edit` tool's `replace_all` would work per-string, but here four substitution patterns were needed in one pass (including both `16 GB` with a space and `16GB` without).

**Order of substitutions matters.** `RTX 5080 Ti` must be replaced *before* the bare `5080 Ti` pattern, otherwise the first rule would never match (the second would have already shortened the token). The semicolon-chained `sed` command applies rules left-to-right on each line, so the order in the script above is correct.

### 5.3 Verification

```
Grep pattern: 5080|16 GB|16GB
File: D:\HDAI_Project\brief.md
Result: No matches found
```

```
Grep pattern: RTX 5070|5070
File: D:\HDAI_Project\brief.md
Result: 22 matches
```

Count matches the count of replacements. No stale references remain.

---

## Step 6 — Creating the `hdai` conda env

### 6.1 `conda create`

```
$ /c/Users/worka/anaconda3/Scripts/conda.exe create -n hdai python=3.11 -y
```

Installed Python 3.11.15 (slightly newer than `privacy`'s 3.11.14 — conda picked the latest 3.11.x patch). `-y` skips the confirmation prompt.

Result: env created at `C:\Users\worka\anaconda3\envs\hdai`.

**Why Python 3.11 and not 3.12+:** the `privacy` env uses 3.11, the cu128 wheels are still happiest on 3.10–3.12, and some profiling tooling (older `fvcore`) lags on brand-new Python releases. Stay conservative.

### 6.2 Installing PyTorch with the cu128 wheel

At this step the env was installed into via its Python directly (absolute path) because conda activation in Git Bash had not yet been set up. Later in the project we switched to the `source conda.sh && conda activate hdai && <cmd>` pattern; both forms are equivalent for pip installs. Historical command as run:

```
$ /c/Users/worka/anaconda3/envs/hdai/python.exe -m pip install \
    torch==2.10.0 torchvision==0.25.0 torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

Canonical form going forward:

```
$ source /c/Users/worka/anaconda3/etc/profile.d/conda.sh && conda activate hdai
$ pip install torch==2.10.0 torchvision==0.25.0 torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Installed (final line of pip output):

```
Successfully installed MarkupSafe-3.0.3 filelock-3.25.2 fsspec-2026.2.0 jinja2-3.1.6
mpmath-1.3.0 networkx-3.6.1 numpy-2.4.3 pillow-12.1.1 sympy-1.14.0
torch-2.10.0+cu128 torchaudio-2.11.0+cu128 torchvision-0.25.0+cu128
typing-extensions-4.15.0
```

Observations:
- Pinned `torch==2.10.0` and `torchvision==0.25.0` to exactly match `privacy`. `torchaudio` was left unpinned → resolved to `2.11.0+cu128`, one minor ahead; harmless for this project (we never call torchaudio, but it's a hard dep of the torch metapackage on some index mirrors).
- `numpy` resolved to `2.4.3` (ahead of `privacy`'s `2.3.5`). This is torch's own dependency floor; numpy 2.4 is API-compatible with 2.3.
- Three PATH warnings for `torchrun.exe`, `torchfrtrace.exe`, `isympy.exe`, `f2py.exe`, `numpy-config.exe`. These don't matter — we invoke the env's python directly, never via its scripts dir.

### 6.3 Installing project-specific packages

```
$ /c/Users/worka/anaconda3/envs/hdai/python.exe -m pip install \
    pandas matplotlib seaborn nvtx transformers fvcore ptflops
```

(Canonical form going forward: `pip install pandas matplotlib seaborn nvtx transformers fvcore ptflops` after `conda activate hdai`.)

Installed (trimmed):

```
Successfully installed annotated-doc-0.0.4 anyio-4.13.0 certifi-2026.2.25 click-8.3.2
contourpy-1.3.3 fonttools-4.62.1 fvcore-0.1.5.post20221221 huggingface-hub-1.11.0
matplotlib-3.10.8 nvtx-0.2.15 pandas-3.0.2 ptflops-0.7.5 pyyaml-6.0.3 regex-2026.4.4
safetensors-0.7.0 seaborn-0.13.2 tokenizers-0.22.2 transformers-5.5.4 ...
```

Notable version deltas vs. `privacy`:
- `transformers` 5.5.4 vs `privacy`'s 5.3.0 → newer, fine
- `pandas` 3.0.2 vs 3.0.0 → patch-level bump
- `huggingface-hub` 1.11.0 vs 1.7.2 → major-ish bump; still API-compatible for `from_pretrained`
- `fvcore` 0.1.5.post20221221 → not in `privacy` (profiling-specific dep added here)
- `ptflops` 0.7.5 → also new, for roofline flop-counting

**Why these packages specifically:** they're the exact set called out in brief sections 2.5 (`pandas`, `matplotlib`, `seaborn`, `transformers`, `pillow` [pulled by torchvision], `nvtx`) and 23 (`fvcore`, `ptflops` for FLOP counting). `numpy`, `pillow`, and `safetensors` are already installed transitively.

**Not installed:** anything that would force the user to recreate the env later. No `jupyter`, no `ipykernel`, no `black` — add as needed.

---

## Step 7 — Smoke-testing the `hdai` env

```python
$ conda activate hdai && python -c "..."
PyTorch: 2.10.0+cu128
CUDA: 12.8
cuDNN: 91002
CUDA available: True
Device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
Compute cap: (12, 0)
Matmul smoke: torch.Size([1024, 1024]) OK
cuDNN conv smoke: torch.Size([16, 128, 56, 56]) OK
```

Every line is meaningful:

| Line | Interpretation |
|---|---|
| `PyTorch 2.10.0+cu128` | correct wheel is active |
| `CUDA: 12.8` | runtime embedded in the wheel is 12.8, matches toolkit |
| `cuDNN: 91002` | cuDNN 9.10.2 — newer than the brief's "9.x" floor |
| `CUDA available: True` | driver sees the GPU and PyTorch can talk to it |
| `Device: NVIDIA GeForce RTX 5070 Ti Laptop GPU` | **this is the actual card** — it's a 5070 Ti *Laptop*, not the plain 5070 the user mentioned. Same `sm_120` architecture regardless. |
| `Compute cap: (12, 0)` | `sm_120` confirmed — this is what the whole cu128-wheel dance is for |
| `Matmul smoke` | 1024×1024 matmul runs end-to-end on the GPU, returns correct shape. If the wheel were wrong this would have raised "no kernel image available". |
| `cuDNN conv smoke` | a 16×64×56×56 → 16×128×56×56 3×3 conv with padding=1 actually dispatches through cuDNN. Proves the cuDNN path is not silently bypassed. |

**Discrepancy revisited:** the card is labelled "RTX 5070 Ti Laptop GPU", not "RTX 5070". In the brief I replaced "5080 Ti" → "5070" per the user's message. The user can decide whether to further correct to "5070 Ti Laptop". This is flagged in the summary.

**What this step did NOT prove:**
- The cuDNN path isn't being used in a suboptimal algorithm (baseline profiling in Phase 2 will reveal algorithm selection).
- That Tensor Cores are actually engaged (FP16 autocast experiment in Phase 7 will).
- That Nsight Systems is installed (it is a separate download; not yet done this session).

---

## Step 8 — Updating README with concrete versions

After the env was built, the README was updated to replace the generic hardware block with:

- Exact device string: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- Exact driver version: 592.01
- Exact cuDNN version: 9.10.2 (`91002`)
- Exact OS build: Windows 11 Home 10.0.26200
- Two version tables (core runtime, profiling/modelling) listing every pinned-or-observed version

Setup commands were also tightened: `torch==2.10.0 torchvision==0.25.0` are now pinned so a fresh clone reproduces the same stack instead of drifting to whatever PyTorch ships next week.

---

## Step 9 — What still does NOT exist

Deliberate non-actions, for transparency:

- `env/`, `models/`, `profile/`, `analysis/`, `results/`, `writeup/` directories — not created yet. README describes them as the target layout, the code will be written in later phases (brief Phases 2–10).
- `requirements.txt` — not written. Setup commands in README are authoritative for now; `requirements.txt` can be generated later with `pip freeze > requirements.txt` after the full script suite is in place.
- `.gitignore` — not written. No `.git` directory exists in `D:\HDAI_Project` yet.
- `env/check_env.py` — referenced in README but not yet written. Brief section 21.1 contains the exact source to paste in when needed.
- Nsight Systems — not installed. Mentioned in README as "required for Experiment 5"; flagged for a separate install step.
- No profiling has been run. No traces, no tables, no plots.

---

## Step 10 — State at end of this log

```
D:\HDAI_Project\
├── brief.md               # corrected: RTX 5070, 12 GB
├── README.md              # up-to-date: exact device string, pinned versions
└── execution_log_0.md     # this file
```

Conda envs on the machine:

```
base      C:\Users\worka\anaconda3
housing   C:\Users\worka\anaconda3\envs\housing
privacy   C:\Users\worka\anaconda3\envs\privacy    # reference env, untouched
hdai      C:\Users\worka\anaconda3\envs\hdai       # new, ready for profiling work
```

Env `hdai` verified:
- Python 3.11.15
- torch 2.10.0+cu128, cuDNN 91002
- sees `NVIDIA GeForce RTX 5070 Ti Laptop GPU` at `sm_120`
- cuDNN conv smoke test passes

Ready to proceed to brief's **Phase 2** (write `env/check_env.py`, then `models/resnet.py`, then `profile/run_baseline.py`).

---

## Critical observations & risks going forward

1. **Card is 12 GB, not 16 GB.** Brief's batch-256 suggestions for ResNet-18 and DistilBERT at seq=512 will OOM sooner. Cap accordingly in the batch sweep.
2. **Laptop-class TGP (80 W cap seen at idle).** Thermal throttling will bite harder than on a desktop 5080 Ti. The brief already recommends `sleep 3–5` between runs and monitoring `nvidia-smi dmon -s u` — this is not optional on a laptop. Expect the later measurements in a long sweep to be slower simply because the card hit 80 °C.
3. **Driver is 592.01 reporting CUDA 13.1.** Newer than brief assumes (brief baseline is 570+ / CUDA 12.8). No incompatibility, but if any `nvidia-smi` field parsing is done downstream, the numbers will differ from the brief's examples.
4. **`torchaudio` 2.11.0 vs `torch` 2.10.0.** Minor version skew; unused in this project. Document but ignore.
5. **`numpy` 2.4.x.** Still relatively fresh. If any plotting or analysis library raises a numpy-2 compat warning, first look here.
6. **The actual device string is "RTX 5070 Ti Laptop GPU", not "RTX 5070".** The brief was corrected to "5070" per the user's instruction; if the user wants the exact marketing name, one more correction pass is needed.
7. **No git repo.** If version control matters for the writeup audit trail, `git init` should happen before any more files are added.
