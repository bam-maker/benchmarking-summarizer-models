# CNN/DailyMail benchmark on 250 articles

All five systems were evaluated on the same first 250 IDs from the existing frozen seed-42 CNN/DailyMail 3.0.0 test sample. Both tracks are complete. The original 50-article pilot is nested in these 250 articles. Its comparison is descriptive, and the two sample sizes share observations.

Scores are mean per-article ROUGE F1 ×100 with Porter stemming, using the original reference highlights. ROUGE-L is sequence LCS; ROUGE-Lsum uses the existing sentence-aware union-LCS segmentation. Attempted failures retain empty outputs and zero ROUGE. Output words use whitespace splitting. Input coverage measures processed characters and supplied/original source lengths.

## Full-article application

| System | Complete | Failed | Pending | R-1 | R-2 | R-L | R-Lsum | Mean words | Reused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 250 | 0 | 0 | 39.17 | 17.60 | 26.56 | 36.21 | 57.24 | 250 |
| Transformer | 250 | 0 | 0 | 38.25 | 16.35 | 26.02 | 35.37 | 50.20 | 250 |
| BART | 250 | 0 | 0 | 43.25 | 20.30 | 29.24 | 40.47 | 78.37 | 50 |
| PEGASUS | 250 | 0 | 0 | 42.15 | 19.64 | 28.79 | 39.21 | 71.50 | 50 |
| Lead-3 | 250 | 0 | 0 | 41.32 | 18.24 | 25.48 | 37.14 | 75.80 | 50 |
## Shared-prefix comparison

| System | Complete | Failed | Pending | R-1 | R-2 | R-L | R-Lsum | Mean words | Reused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 250 | 0 | 0 | 38.72 | 17.38 | 26.62 | 35.81 | 48.69 | 250 |
| Transformer | 250 | 0 | 0 | 38.59 | 17.14 | 26.64 | 35.42 | 45.96 | 250 |
| BART | 250 | 0 | 0 | 43.64 | 20.40 | 29.77 | 40.53 | 59.12 | 50 |
| PEGASUS | 250 | 0 | 0 | 42.62 | 19.99 | 29.54 | 39.72 | 54.47 | 50 |
| Lead-3 | 250 | 0 | 0 | 41.32 | 18.24 | 25.48 | 37.14 | 75.80 | 50 |

## Descriptive change from the nested pilot

Each cell is the 250-article mean minus the 50-article mean in score points.

| Track | System | ΔR-1 | ΔR-2 | ΔR-L | ΔR-Lsum |
|---|---|---:|---:|---:|---:|
| common_input | LSTM | +0.58 | -0.95 | -0.20 | +0.28 |
| common_input | Transformer | -0.50 | -1.47 | -1.54 | -0.53 |
| common_input | BART | -0.19 | -0.31 | -0.61 | -0.20 |
| common_input | PEGASUS | +0.03 | -1.34 | -0.70 | -0.16 |
| common_input | Lead-3 | +0.96 | -0.04 | +0.57 | +0.84 |
| full_article | LSTM | -1.78 | -2.74 | -2.14 | -2.01 |
| full_article | Transformer | -0.50 | -1.00 | -0.96 | -0.72 |
| full_article | BART | -1.95 | -2.19 | -1.64 | -2.26 |
| full_article | PEGASUS | -1.66 | -2.80 | -2.14 | -1.70 |
| full_article | Lead-3 | +0.96 | -0.04 | +0.57 | +0.84 |

## Provenance and interpretation

The production code, preprocessing, native decoding, chunking and scoring match the original freeze. The requirements-file difference is the addition of a sentencepiece==0.2.3 pin. That version was already in the frozen installed dependency inventory. Every installed version matches. All four checkpoint identities were checked, and the exact prepared text hashes and parent fingerprint were checked for each reused prediction. The reuse manifest lists every original path and raw-record SHA-256.

1,300 existing records were reused. These include 250 per track for LSTM and Transformer from the interrupted larger run, and 50 per track for BART, PEGASUS and Lead-3 from the pilot. The run generated 400 additional BART summaries and 400 PEGASUS summaries, and extracted 400 additional Lead-3 summaries. Reused records are excluded from new-generation rate measurements.

All neural records use CUDA on the same NVIDIA GeForce RTX 4090, FP32 with TF32 disabled, batch size one. The prepared PC had about 302 GiB of free disk and 22 GiB free GPU memory at preflight. Desktop, Epic overlay and a separate Gradio process were present. Historical timings and current timings come from different sessions. They describe operational conditions and support progress estimates. A controlled speed ranking was not performed.

Full article uses the original input and the existing natural-checkpoint-v1 application policy. Shared prefix gives all systems identical original text up to 400 OpenNMT tokens and within both HF capacities. Both tracks retain full original references, including facts outside the prefix. Complete input processing and high ROUGE leave factual coverage to examine through source-based reading. Human ratings, statistical significance and factual-accuracy percentages were not measured.

## Reproduce and resume

Use the prepared repository with its existing .venv, pinned checkpoints, original 1,000-row prepared source cache, historical raw prediction cache and original freeze metadata. Raw traces and complete article corpus stay in ignored local storage. Committed artifacts contain IDs/hashes, compact predictions with original references, and configurations.

From the repository root in Windows PowerShell, run each line separately.

~~~powershell
.\.venv\Scripts\python.exe tests\benchmark250.py prepare
.\.venv\Scripts\python.exe -u tests\benchmark250.py run
.\.venv\Scripts\python.exe tests\benchmark250.py status
.\.venv\Scripts\python.exe tests\benchmark250.py report
~~~

The same commands on an identically prepared Ubuntu CUDA system use .venv/bin/python and forward-slash paths. The freeze checks hardware, installed versions and checkpoint/input hashes. This session was executed on Windows CUDA.

Predictions are saved atomically after each article/track. Run again to resume missing cases; attempted failures are retained. A run.lock prevents two writers. After an abrupt process termination, verify that the PID recorded in test-results/benchmark250/run.lock is no longer running before removing that specific stale lock.

The original pilot50 and interrupted final_test_1000 artifacts remain separate. A fresh clone needs the prepared datasets and model cache to generate, while it can read the committed results immediately. Presentation revisions consume results only after all 2,500 requested evaluations have records.

[Metrics](results.json) · [Predictions and references](predictions.csv) · [Frozen configuration](frozen_config.json) · [Article IDs and hashes](article_ids.json) · [Reuse manifest](reuse_manifest.json)

[Timing and completion counts](timing_summary.json) · [Export validation](validation.json)
