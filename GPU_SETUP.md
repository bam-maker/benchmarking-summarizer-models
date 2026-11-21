# RTX 4090 setup and four-model inference

The setup and historical measurements below are preserved. The current simple
demo automatically selects its device and processes full articles, without
device, prefix or timing controls. See [FULL_ARTICLE.md](./FULL_ARTICLE.md) for
the current app; fixed-budget benchmark comparisons are unchanged.

This pass adds and verifies GPU inference for the four checkpoint backends.
It does not train/fine-tune models or claim to complete the broader dataset
evaluation, adaptive-length, chunking, or presentation work in the attached brief.
The existing OpenNMT code revision, checkpoints, vocabularies and two compatibility
fixes are preserved; the original kit scripts were not modified.

## Verified environment

- Windows 11 Pro x64, Python 3.11.9 in the existing project `.venv`.
- NVIDIA GeForce RTX 4090, 24,564 MiB VRAM; NVIDIA driver 621.24.
- PyTorch **2.5.1+cu124**, bundled CUDA runtime **12.4**.
- `torch.cuda.is_available()` is true; a real CUDA matrix multiplication passed.
- torchtext 0.6.0 and NumPy 1.26.4 remain unchanged.
- Transformers 4.46.3, tokenizers 0.20.3, safetensors 0.6.2,
  huggingface-hub 0.36.0. The Hub version satisfies both Transformers and the
  existing Gradio 5.49.1; `pip check` passes.

PyTorch's [official previous-version instructions](https://pytorch.org/get-started/previous-versions/#v251)
publish the CUDA 12.4 wheel for Windows. No separate CUDA toolkit installation,
driver update, WSL changes, or administrative symlink permissions were needed.

## Install / reproduce

Run from the repository root using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Only needed if the legacy checkpoint/code files are missing:
.\.venv\Scripts\python.exe OpenNMT_Verified_Setup\prepare.py

# Prefetch only the revision-pinned HF inference files:
.\.venv\Scripts\python.exe download_models.py BART PEGASUS

.\.venv\Scripts\python.exe app.py
```

The CUDA wheel also runs on CPU; selecting CPU does **not** require reinstalling
PyTorch. A CPU-only rollback/install is available if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

Root requirements use `torch==2.5.1` (accepts either local build) to avoid replacing
the chosen build when installing the remaining dependencies. Original OpenNMT-py
must still come from the pinned legacy checkout, not the current PyPI package.
Weights, downloaded code, virtual environments and caches remain Git-ignored.
Allow around 20 GB including cached wheels and the four models' files.

## Device selection, memory, and timing

The backend API offers Auto, CPU and CUDA; the current demo uses Auto. Auto uses CUDA when
available and reports fallback to CPU when unavailable or when CUDA runs out of
memory during activation or generation. Explicit CUDA instead fails visibly; it
never inserts a CPU result into a CUDA comparison. A failed device transfer is
rolled back to CPU so it cannot leave a partially moved cached model.

Models/tokenizers load lazily once per process and are cached in CPU RAM.
Only the active model resides on the GPU. Returning to a previously selected
model transfers the same cached model object rather than loading another copy.
Caching all four consumes several GB of host RAM; this machine has about 63 GB.
Process-level serialization protects GPU transfers and inference, including direct
Python calls outside the Gradio queue. No model substitution or external API is used.

The optional `SUMMARIZER_DEVICE=cpu` environment variable makes Auto select CPU.
An explicit UI/benchmark device takes precedence. Example direct call:

```python
from summarizer import summarize
result = summarize(article, "BART", "Medium", device="cuda")
print(result.device, result.load_seconds, result.seconds)
```

Loading time includes cache lookup, initial disk/tokenizer/model loading, and
device/offload transfers; a prefetched warm request normally has near-zero loading
time. First-use download, if necessary, also belongs to loading, not generation.
CUDA context startup is outside the measured model-loading window. Generation
includes dataset/tensor construction, decoding and cleanup, excluding model load,
tokenization, queue waiting and browser/network overhead. CUDA is synchronized
before/after the timed regions. If Auto retries on CPU, the failed CUDA attempt
is reported separately rather than silently included in successful CPU latency.

All four backends use FP32, batch size 1, beam size 4, two CPU threads, deterministic
decoding and trigram repetition controls. TF32 is disabled for CUDA measurements.
Length choices remain approximate token budgets, **not word-count guarantees**;
different models use different tokenizers and generation defaults. OpenNMT's
tested budgets/behavior are preserved. HF uses `min_new_tokens` and
`max_new_tokens`, no-repeat trigrams and EOS stopping. Forced EOS at the cap is
disabled; BART's initial EOS-valued decoder-start token is excluded from termination
checks. PEGASUS's `<n>` sentence separator is displayed as a newline.

## Exact added checkpoints

The immutable HF revisions and weight filenames are in [model_manifest.json](./model_manifest.json).

| Backend | Original source | Revision | Instantiated parameters |
|---|---|---|---:|
| BART | [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) | `37f520fa929c961707657b28798b30c003dd100b` | 406,290,432 |
| PEGASUS | [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail) | `40d588fdab0cc077b80d950b300bf66ad3c75b92` | 570,797,056 |

BART uses 12 encoder / 12 decoder layers, hidden width 1024 and its original
byte-level BPE tokenizer; PEGASUS uses 16 / 16, width 1024 and its original
SentencePiece tokenizer. Both preserve input case and have 1024-token encoder
limits including special tokens. Over-limit input is rejected unless the user
explicitly approves a supported prefix; the full article remains in the textbox.

PEGASUS's total includes 2,097,152 **fixed sinusoidal positional values**; its
learned parameter count is 568,699,904. Transformers may print a generic
missing-weights / TRAIN warning for its two positional tables. In the pinned
implementation these are deterministically reconstructed using sin/cos with
`requires_grad=False`, not random learned parameters. The adapter validates load
information and accepts only these known fixed missing tables; loading information
is saved in the speed reports. There is no training or fine-tuning.

The original two checkpoints, hashes and architecture details remain documented
in [README.md](./README.md) and the [kit manifest](./OpenNMT_Verified_Setup/checkpoint_manifest.json).
HF cache files live under `.cache/huggingface/`. Windows may warn that symlinks
are unavailable; the normal copy/move fallback is used. Serial downloads avoid
a concurrent symlink-capability-probe race found in the initial PEGASUS download.

## Focused verification and fair speed comparisons

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests\smoke_cpu.py
.\.venv\Scripts\python.exe tests\smoke_devices.py --device cuda

# One device per complete four-model run; no automatic fallback:
.\.venv\Scripts\python.exe benchmark_speed.py --device cuda
.\.venv\Scripts\python.exe benchmark_speed.py --device cpu
```

The speed runner uses the **same synthetic bus-news article** for all four models,
Medium settings, one warm-up and three timed repetitions. It verifies actual tensor
device, FP32, real nonempty summaries, model identity reuse and at most one GPU
resident. Reports record source text/hash, checkpoint revisions, parameter counts,
load information, cold loading and cold generation, and warmed median/p95 latency.
Failures are recorded and produce a nonzero exit, not dropped from the model set.

Actual measurements are in [CUDA results](./reports/speed_cuda.json) and
[CPU results](./reports/speed_cpu.json). These are speed/integration checks on **one
fixture**, not a CNN/DailyMail quality benchmark. With only three measurements,
p95 is descriptive, not a statistically stable benchmark estimate. CPU and CUDA
runs are kept in separate reports and must not be mixed into one model ranking.

### Measured RTX 4090 result

All four models passed real inference on both CUDA and forced CPU. On this single
fixture, the same-device CUDA run measured:

| Model | Initial loading / transfer (s) | Warm generation median (s) | Warm generation p95 (s) |
|---|---:|---:|---:|
| LSTM | 0.546 | 0.276 | 0.278 |
| Transformer | 1.186 | 0.638 | 0.676 |
| BART | 2.168 | 0.471 | 0.475 |
| PEGASUS | 3.312 | 0.562 | 0.603 |

Additional checks passed: 17 unit checks, the six original OpenNMT CPU interface
cases, and eight real CUDA interface cases (all four models at Short and Detailed),
then a CPU-to-CUDA switch. Exactly two OpenNMT and two HF model objects were
constructed across that interface run, demonstrating reuse after model switches.
CPU fallback on CUDA unavailability and out-of-memory conditions was tested with
injected unit-test failures; physical GPU memory was not deliberately exhausted.
Lint, editor diagnostics and dependency validation passed. Legacy library warnings
remain visible in the terminal; no model training was performed.
