# Project status: development paused and saved

## Offline animated presentation, November 22, 2026

The [browser deck](presentation/web/README.md) contains 16 main slides and 14
appendix slides. Its [15-minute speaking script](presentation/web/SPEAKING_SCRIPT.md)
includes the live-demo switch and an offline fallback. The deck runs from a
loopback static server on port 8008, independently of Gradio on port 7860.
Reveal.js, fonts, diagrams and reproduced Riverton outputs are included locally.

The main charts and backup tables use the saved 50-article results. The
[static PDF](presentation/web/News_Summarization.pdf) shows the final state of
each slide. Windows Chrome testing covered 52 slide and fragment states,
reduced motion, presenter controls and offline navigation.
[Preview](presentation/web/contact-sheet.jpg) · [Validation](presentation/web/VALIDATION.md)

## Redesigned presentation, November 22, 2026

The [revised editable PowerPoint](presentation/Comparing_Neural_Models_for_News_Summarization.pptx)
and [PDF](presentation/Comparing_Neural_Models_for_News_Summarization.pdf) contain
31 slides, with 16 main slides, two references slides and 13 backups. The
[two-presenter script](presentation/SPEAKING_SCRIPT.md) totals 15 minutes including
the brief demo. Original PowerPoint and PDF files are preserved with `_v1`
filenames in the same folder.

The redesign uses warm-white slides, separate model architecture diagrams,
actual saved dataset and tokenizer examples, a current app screenshot captured
without submission, and native editable benchmark charts. The full-article and
shared-prefix tables use the existing frozen 50-article values. Exact saved
Riverton outputs and the documented Ubuntu launch checklist remain in backups.
No course guideline was found in the available files, so the supplied redesign
brief governs the structure.

All 31 slides were rendered with PowerPoint and visually reviewed. Source,
assets, previews, validation notes and an evidence-hash manifest are in
[presentation/](presentation/README.md). This work ran no inference, training or
benchmark and preserved application code, settings, models and the application
environment. The older development status below remains historical context.

## Application and evaluation status

Current status: November 22, 2026 (America/Chicago). The natural-length demo and
50-article comparison are complete. The Ubuntu CPU setup and focused laptop
diagnosis have now been documented. This save contains the working source, tests,
documentation and selected measured reports. No application features, generation
settings or benchmarks were changed or rerun during this save. Earlier
development/audit reports retain their historical configurations; they are not
current production results.

## Current application

The polished local Gradio UI has Article input, a LSTM / Transformer / BART /
PEGASUS model dropdown, Summarize, read-only Summary, Clear, Copy, compact progress,
input/output word counts, compression, processing time and measured generation
throughput. There is no summary-length selector or percentage target in the demo.

Production explicitly uses **natural-checkpoint-v1**: model-specific checkpoint
search settings, deterministic decoding and natural EOS stopping. Initial output
ceilings are 100 tokens for each OpenNMT model, 141 new tokens for BART and 127 for
PEGASUS. Minimum lengths are bounded by source tokens divided by eight and each
model's native floor. Missing EOS permits bounded retries to hard output caps
of 256 for OpenNMT and 512 for HF models. These are safeguards, not word promises.
Experimental adaptive-length controls remain available outside the demo; their
historical results must not be pooled with natural-policy outputs.

Articles fitting the supported input window use one source pass. Longer articles
use sentence-aware chunks, safe splitting of oversized sentences, complete source
coverage tracking, source-order merging, exact-sentence deduplication and bounded
consolidation with progress checks. Workload/depth/memory/time safeguards remain.
A failed full-article request returns a clear error, not a partial final summary.
Models load lazily and are cached; inactive models move off GPU. Auto prefers the
existing RTX 4090 CUDA configuration on the desktop, with explicitly reported CPU
fallback. The Ubuntu laptop has no supported CUDA GPU and uses the existing CPU path.

## Ubuntu laptop CPU setup and offline class demo

The Ubuntu laptop has an Intel Core i7-1195G7, about 15 GiB RAM and Intel Iris Xe
graphics; PyTorch reports no CUDA device. A fresh Linux `.venv` uses Python 3.11.16
and PyTorch 2.5.1+cpu. The verified dependency pins were reused, including
`sentencepiece==0.2.3` for PEGASUS. The pinned OpenNMT checkout and exact original
checkpoints were recovered, with original SHA-256 hashes verified; Linux inference
copies were checked against their original tensors and vocabularies. Both Hugging
Face models use the revisions in [model_manifest.json](model_manifest.json), with
their cached snapshot hashes checked against the saved verification report.
Environments, weights, caches, downloaded dependencies, datasets and credentials
remain Git-ignored.

The Gradio app started on localhost, and all four models completed real CPU
inference using cached resources. Subsequent direct checks on the supplied
Riverton article ran with Hugging Face offline mode enabled and all four ended
normally. For an offline classroom relaunch from the repository root, after
stopping any existing process on port 7860, use exactly:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python app.py --port 7860
```

Open <http://127.0.0.1:7860/>. The first request per model includes loading time;
the larger BART and PEGASUS models require more CPU RAM and patience. A fresh clone
still needs internet once for dependencies, the pinned third-party checkout and
weights; the prepared laptop's cached resources support offline inference. See
the [Ubuntu setup instructions](README.md#ubuntu-laptop-cpu-setup-and-offline-launch)
for fresh-environment recovery. No install or inference was run while preparing
this Git save.

## Completed measured 50-article benchmark

Source: CNN/DailyMail 3.0.0 official test split; the same first 50 IDs from the
previously frozen random seed-42 sample for every system. Five systems, two tracks,
**500 evaluations completed, zero failed requests**. Native natural policy remained
frozen. Metrics below are actual ROUGE F1 x100 with Porter stemming; Lsum uses
consistent sentence segmentation. No hypothetical target scores are included.

Full-article application:

| System | Articles | R-1 | R-2 | R-L | R-Lsum | Mean words | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 50 | 40.95 | 20.34 | 28.71 | 38.22 | 55.0 | 0 |
| Transformer | 50 | 38.75 | 17.35 | 26.99 | 36.09 | 47.4 | 0 |
| BART | 50 | 45.21 | 22.50 | 30.88 | 42.73 | 77.4 | 0 |
| PEGASUS | 50 | 43.81 | 22.43 | 30.92 | 40.91 | 68.9 | 0 |
| Lead-3 | 50 | 40.36 | 18.28 | 24.91 | 36.30 | 74.4 | 0 |

Shared-prefix comparison (identical original text within 400 OpenNMT tokens and HF capacities):

| System | Articles | R-1 | R-2 | R-L | R-Lsum | Mean words | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 50 | 38.14 | 18.33 | 26.82 | 35.53 | 50.2 | 0 |
| Transformer | 50 | 39.09 | 18.61 | 28.18 | 35.95 | 46.7 | 0 |
| BART | 50 | 43.83 | 20.71 | 30.37 | 40.73 | 62.5 | 0 |
| PEGASUS | 50 | 42.59 | 21.34 | 30.24 | 39.88 | 53.9 | 0 |
| Lead-3 | 50 | 40.36 | 18.28 | 24.91 | 36.30 | 74.4 | 0 |

Measured results, IDs and predictions: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md),
[metrics](reports/pilot50/results.json) and
[outputs/references](reports/pilot50/predictions.csv). Lead-3 extracts the first
three sentences. Both tracks use the original full highlights; references can
mention facts outside the common prefix. Keep the two tracks separate.
BART leads most overlap metrics on this small sample; this establishes no factual
quality winner. GPU/desktop contention, including Fortnite, was observed, so saved
local timings do not establish a controlled speed ranking.

## Verification already completed

- Four-backend identity/loading checks: official OpenNMT original hashes and
  inference-copy tensors/vocabularies agree; pinned HF revisions and loading keys
  recorded. PEGASUS missing positional tables equal fixed deterministic sinusoids.
  Actual evaluation/inference mode, HF attention masks, CUDA FP32, input delivery
  and cached model identity were checked. See
  [model verification](reports/final_model_verification.json).
- Saved natural-policy checks: all four models completed 81-word and multi-chunk
  1,354-word examples with full coverage and EOS. Three five-word cases completed;
  LSTM's empty five-word decode was explicitly rejected. Token/time accounting,
  model reuse and UI routing were checked. See
  [natural-policy verification](reports/natural_policy_verification.json).
- Prior focused checks passed: 10 UI/statistics, 20 pipeline, 6 natural-policy and
  4 evaluation-protocol tests. Rendered Chrome checks passed at 320/390/760/1280px
  for controls, progress, statistics, Clear/Copy, stale-result invalidation and
  responsive layout. These are recorded earlier checks, not a new save-time run.
- Earlier 40-article development evaluation and five-case factual audit are retained
  under reports/. Historical smoke scripts with length inputs describe the older
  interface; use smoke_natural_policy.py for current real-backend UI routing checks.

## Known limitations

Models can omit the main event, reverse relationships, invent an entity, repeat
claims or produce incomplete text despite EOS. More words, source delivery, ROUGE
and passing software tests do not verify factual accuracy. No human factuality
ratings or population factual-error rates have been measured. Blank review sheets
and AI-assisted observations are labeled separately in the saved reports.

The earlier fictional Riverton fixture is separate from CNN/DailyMail. In that
older case, PEGASUS omits lifting the advisory in raw direct inference and the app
despite complete one-pass input delivery; BART retains it. No implementation fix
was supported. See [earlier Riverton diagnosis](reports/RIVERTON_DIAGNOSIS.md).

The **latest, different Riverton classroom article** is recorded verbatim in the
[laptop model diagnosis](reports/LAPTOP_MODEL_DIAGNOSIS.md). The supplied LSTM and
Transformer display strings were exactly identical, but direct inference and a
fresh live Gradio LSTM request produced a distinct LSTM summary. Its raw output
differs from Transformer's, and each selection resolved to its intended model;
the original duplicate display's browser-level cause remains unresolved because
that response and selected value were not captured. On this article BART's raw
summary omitted the lifted advisory, while PEGASUS included it but omitted the
repair/testing evidence. This single example does not establish a general model
ranking or justify a production change. The completed **50-article benchmark and
its saved predictions remain unchanged**.

OpenNMT still uses an NLTK Treebank approximation of historical Stanford
preprocessing; its 1024-token app
window exceeds the 400-token training convention. Chunk merging can lose global
relationships and repeat paraphrases. Tiny inputs can yield empty decodes.
The time guard is cooperative; it cannot interrupt an in-flight model call.

## Interrupted 1,000-article study: INCOMPLETE

The user stopped the larger run and chose the 50-article comparison instead.
It is not a completed final benchmark. Existing saved attempts per track are:

| System | Common-input files | Full-article files |
|---|---:|---:|
| LSTM | 1000 | 1000 |
| Transformer | 407 | 406 |
| BART | 50 | 50 |
| PEGASUS | 50 | 50 |
| Lead-3 | 50 | 50 |

Counts include the later completed 50-article subset. LSTM's larger full-article
run has one empty-chunk failure outside that subset. The larger report/aggregates
are earlier partial snapshots and do not necessarily reflect every saved file.
Do not pool their unequal sample counts or call them final results.

Exact sources/prefixes remain locally in ignored data/final_test_1000/; exact
per-call inputs, raw outputs, diagnostics, predictions, logs and code archive
remain in ignored test-results/final_test_1000/. The large partial prediction
export is also ignored at reports/final_test_1000/predictions_and_scores.csv.
Small frozen configuration, snapshot reports and interruption notes are retained.
Nothing was deleted. Do not resume the larger run without a new instruction.

## Restart the app and resume work on Windows

From PowerShell, using the existing environment (no reinstall or activation):

```powershell
Set-Location -LiteralPath 'C:\Users\selam\OneDrive\Documents\benchmarking-summarizer-models'
$env:SUMMARIZER_DEVICE = 'auto'
.\.venv\Scripts\python.exe .\app.py --processing-mode full
```

Open http://127.0.0.1:7860/. Stop the existing app with Ctrl+C in its terminal before
restarting on the same port. Use `$env:SUMMARIZER_DEVICE = 'cpu'` for CPU only;
restore `auto` to prefer CUDA. The app is local only; public sharing is disabled.

To resume ordinary project work, inspect the saved state and results first:

```powershell
Set-Location -LiteralPath 'C:\Users\selam\OneDrive\Documents\benchmarking-summarizer-models'
git status --short
git log -1 --oneline
Get-Content .\PROJECT_STATUS.md
Get-Content .\BENCHMARK_RESULTS.md
```

To regenerate only the small report from saved predictions (no model inference):

```powershell
.\.venv\Scripts\python.exe .\tests\report_pilot50.py
```

For a separately authorized later resumption of the incomplete larger study, the
exact command is `.\.venv\Scripts\python.exe .\tests\final_benchmark.py run`.
It validates frozen code/environment/input/model hashes and reuses matching saved
attempts. Failed attempts are preserved. It requires the local ignored sources,
legacy checkout and weights; do not run it as part of routine restart. Frozen code
is archived locally at test-results/final_test_1000/frozen_code.zip.

## Checkpoint storage and recovery

Weights are intentionally local and Git-ignored; Git contains the source manifests
and download/conversion scripts, not model weights or third-party checkouts.
Paths below are relative to this repository root:

| Backend | Local storage |
|---|---|
| LSTM | `OpenNMT_Verified_Setup/lstm.pt` (original), `OpenNMT_Verified_Setup/lstm_inference.pt` (inference copy) |
| Transformer | `OpenNMT_Verified_Setup/transformer.pt` (original), `OpenNMT_Verified_Setup/transformer_inference.pt` (inference copy) |
| BART | `.cache/huggingface/hub/models--facebook--bart-large-cnn/snapshots/37f520fa929c961707657b28798b30c003dd100b/` |
| PEGASUS | `.cache/huggingface/hub/models--google--pegasus-cnn_dailymail/snapshots/40d588fdab0cc077b80d950b300bf66ad3c75b92/` |

The HF Hub may store weight bytes in its cache's `blobs` directory instead of
directly in a snapshot, depending on Windows symlink support; preserve the whole
cache when backing it up. The ignored legacy code is under
`OpenNMT_Verified_Setup/legacy/`, pinned to OpenNMT-py 0.2.1 commit
`6db7ec1b3afae28c257a0416dd947022c3b12bc0`.

To recover missing files later (network access and disk space required), run:

```powershell
Set-Location -LiteralPath "C:\Users\selam\OneDrive\Documents\benchmarking-summarizer-models"
.\.venv\Scripts\python.exe .\OpenNMT_Verified_Setup\prepare.py
.\.venv\Scripts\python.exe .\download_models.py BART PEGASUS
```

The first command restores the pinned official legacy checkout, applies tested
compatibility patches, verifies SHA-256 hashes of the original checkpoints and
recreates inference copies. The second fetches the exact HF revisions above.
These recovery commands were not rerun for this save. Official OpenNMT checkpoint
URLs, sizes and hashes are in [checkpoint_manifest.json](./OpenNMT_Verified_Setup/checkpoint_manifest.json);
HF identifiers, revisions and weight filenames are in [model_manifest.json](./model_manifest.json).
Only deserialize these verified trusted sources, not arbitrary checkpoints.

## Remaining work and save boundaries

Development is paused. Optional next work is source-grounded human review and the
class presentation, using the completed rough comparison and its limitations.
No further large benchmark or generation tuning is required for the current scope.

Git contains application/setup/download code, tests, documentation, synthetic
fixtures and selected small reports. Credentials, virtual environments, model
weights, caches, downloaded dependencies/datasets and bulk intermediates remain
ignored and local. Back these up separately if moving computers. This save does
not alter checkpoints, environments, UI or generation settings.
