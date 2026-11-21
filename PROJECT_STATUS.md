# Project status: saved working application

Snapshot: November 20, 2026 (Windows local date). Feature development is paused.
This snapshot records completed work and previously observed results; saving it
does not rerun model tests, install packages, train models or change benchmarks.

Repository: <https://github.com/bam-maker/benchmarking-summarizer-models>

## Completed features

- Four existing CNN/DailyMail-trained inference backends: OpenNMT attention/copying
  LSTM, OpenNMT copying Transformer, BART-large-CNN and PEGASUS-CNN/DailyMail.
  No training or fine-tuning was performed.
- Native Windows Python 3.11.9 environment with PyTorch 2.5.1+cu124 and CUDA 12.4
  on an RTX 4090. Automatic GPU preference, CPU fallback, lazy per-process model
  reuse and offloading of inactive models to CPU RAM are implemented.
- Full-article processing is the default: sentence-aware tokenizer-bounded chunks,
  safe oversized-sentence/word splitting, source-order processing, intermediate
  deduplication and consolidation. Source spans, coverage, calls and passes are
  tracked internally. Consolidation must shrink both words and tokens.
- Configurable input/workload/time/depth safeguards reject incomplete jobs rather
  than presenting a partial summary as complete. Prefix mode remains a CLI/config
  alternative and explicitly reports incomplete coverage.
- Adaptive Short/Medium/Detailed preferences start at 10%/15%/22% of the complete
  original article's words. Token budgets are model-specific and bounded; EOS,
  repetition controls and bounded completion retries are used. Text is not cut
  mid-sentence just to meet a word count.
- A responsive Gradio demo shows only Article, horizontal model selection, a
  three-position length slider (Medium default), Summarize and read-only Summary.
  Chunk/consolidation progress and concise notices are transient; detailed
  diagnostics and separate loading/generation times are logged.
- Setup/download scripts, pinned checkpoint manifests, tests and original
  fixed-budget speed benchmark scripts/reports are retained separately.

See [README](./README.md), [full-article design and limits](./FULL_ARTICLE.md) and
[GPU setup and historical speed checks](./GPU_SETUP.md).

## Actual verification results already completed

- 38 unit tests passed, including coverage, oversized Unicode input, length-budget
  selection, reduction/termination, safeguards, EOS retries, simulated CUDA memory
  recovery, backend reuse and simplified UI wiring.
- All four models generated real CUDA summaries for all three preferences on a
  1,200-word synthetic long article: 12 successful cases. Four additional
  single-oversized-sentence cases passed. Every source character was covered;
  every consolidation stage reduced words and tokens. Four already-concise input
  cases explicitly returned the original text without model generation.
- All 12 model/length combinations passed through the real Gradio API. Reselecting
  a model reused the same backend object, and at most one model resided on GPU.
- Real LSTM CPU inference passed through the new full-article path. CUDA OOM
  recovery was simulated, not intentionally induced on the physical GPU.
- Browser checks at 390px and 1280px widths passed: horizontal model choices,
  all three changing length labels, read-only output, no horizontal overflow,
  real long-article generation and chunk/consolidation progress. A limit failure
  preserved all 120,001 input characters and cleared the stale summary.
- Ruff lint/format checks and `pip check` passed. Editor diagnostics reported no
  errors. Non-failing legacy PyTorch/PEGASUS and Windows Gradio resource warnings
  were observed; the environment was not upgraded to silence them.

Saved small reports (copied from existing results, not regenerated for this save):

- [Full-article verification](./reports/full_article_verification.json): 16 real
  generation cases with outputs, coverage, reductions and separate timings.
- [CUDA Gradio verification](./reports/cuda_interface_verification.json): all 12
  model/length API outputs and model-reuse result.
- [Original CPU speed report](./reports/speed_cpu.json) and
  [original CUDA speed report](./reports/speed_cuda.json): historical fixed-budget
  synthetic-fixture checks, not quality scores or multi-pass application timings.
- [Original setup-kit smoke results](./OpenNMT_Verified_Setup/smoke_results.json):
  earlier hosted-CPU checks; the kit's README describes that historical environment.

## Known limitations

- Lengths are preferences, not guarantees. On the repetitive 1,200-word fixture,
  targets were 120/180/264 words, but outputs were 18 words for LSTM, 28 for
  Transformer, 40 for BART and 29 for PEGASUS across all three settings. Early EOS
  can produce identical outputs despite different requested budgets.
- Complete source coverage means every source chunk was processed, not that the
  summary retains every fact. Hierarchical abstraction can omit facts, conflate
  details or introduce unsupported claims. Quality has not been benchmarked on a
  held-out CNN/DailyMail sample; no quality ranking is established.
- English sentence segmentation and legacy PTB-style normalization are practical
  approximations, not exact reproduction of the original preprocessing.
- Default limits are 120,000 characters, 12,000 words, 64 chunks per stage,
  96 model calls, 64,000 total input tokens, four consolidation passes and
  180 seconds. The time guard is cooperative and cannot interrupt an in-flight
  model call. Jobs can be rejected for non-progress, absent EOS or workload limits.
- Tiny inputs may be returned unchanged with a notice. An empty decode for a
  tiny intermediate tail preserves that source fragment for final consolidation;
  retained fragments are recorded rather than silently discarded.
- First use of each model incurs loading delay; cached inactive models still
  consume CPU RAM. The supported runtime is the pinned existing environment,
  not arbitrary newer OpenNMT/PyTorch releases.

## Restart on this Windows computer

Use PowerShell. No activation or reinstall is needed for the existing environment:

```powershell
Set-Location -LiteralPath "C:\Users\selam\OneDrive\Documents\benchmarking-summarizer-models"
$env:SUMMARIZER_DEVICE = "auto"
.\.venv\Scripts\python.exe .\app.py --processing-mode full
```

Open <http://127.0.0.1:7860/>. Stop with Ctrl+C in the app's terminal, then run the
same commands to restart. If port 7860 is already occupied by this app, use its
existing terminal to stop it before launching a second copy. The app is local
only (`127.0.0.1`, no public sharing).

Optional CPU-only restart:

```powershell
Set-Location -LiteralPath "C:\Users\selam\OneDrive\Documents\benchmarking-summarizer-models"
$env:SUMMARIZER_DEVICE = "cpu"
.\.venv\Scripts\python.exe .\app.py --processing-mode full
```

Set `SUMMARIZER_DEVICE` back to `auto` to prefer the GPU again. Processing defaults
are in [processing_config.json](./processing_config.json). A fresh clone needs
the pinned environment and ignored checkpoints restored using the documented
[installation](./README.md) / [CUDA setup](./GPU_SETUP.md); do not install the CPU
PyTorch wheel over the current functioning CUDA environment.

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

## Remaining work (not started in this save)

1. Quality benchmark: select and freeze an appropriate held-out CNN/DailyMail
   sample, evaluate all systems on the same articles/references, record preprocessing
   and decoding policies, and assess factuality, coverage, repetition and length
   behavior alongside chosen automatic metrics. Keep speed comparisons on the
   same device and separate loading from generation time. Do not treat the
   existing synthetic execution checks as quality evidence.
2. Presentation: after the quality benchmark, prepare a concise account of the
   project, system differences, evaluation results, demo and limitations. No
   presentation has been created yet.

## Git/save boundaries

Application code, reproducible setup/download scripts, tests, documentation,
manifests, a small synthetic fixture and the small reports above belong in Git.
Virtual environments, checkpoints, downloaded dependencies, datasets, caches,
credentials and large intermediates are excluded by [.gitignore](./.gitignore).
Local generated test outputs remain ignored; only the selected report snapshots
are saved under [reports](./reports/). No dataset corpus or secret is required to
restart the existing demo. Trained weights and the local environment must be
backed up separately or recreated from the documented sources.
