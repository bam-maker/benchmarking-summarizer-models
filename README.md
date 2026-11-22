# CNN/DailyMail summarizer comparison

A native Windows, Python 3.11 class project comparing four existing summarizers:
OpenNMT's attention/copying LSTM and copying Transformer, plus BART-large-CNN
and PEGASUS-CNN/DailyMail as stronger candidates (not assumed quality winners).
This is integration of pretrained CNN/DailyMail systems, not training, fine-tuning,
or a controlled architecture-only experiment. CPU and RTX 4090 CUDA execution
are supported. See [GPU setup, four-model verification and speed results](./GPU_SETUP.md)
for the current GPU workflow, exact HF revisions, device/memory policy, and
separate loading/generation measurements. The original OpenNMT setup is retained.

The current demo automatically summarizes full articles with the named
`natural-checkpoint-v1` policy. The News Summarizer interface uses a model
dropdown, balanced Article/Summary panels, Clear/Copy actions, a live
input word count and compact result statistics. See
[full-article behavior, limits and checks](./FULL_ARTICLE.md).
The existing environment is ready; no dependencies or checkpoints need reinstalling.

The previous adaptive-length implementation remains available for experiments
through `summarize_article(article, model, "Short"/"Medium"/"Detailed")`; the demo
instead explicitly selects `generation_policy="natural-checkpoint-v1"` and has
no length selection or percentage output target. Keep its results separate from
all previous adaptive-length results. The earlier
[quality diagnosis](./FULL_ARTICLE.md#focused-quality-diagnosis-november-21)
removes unnecessary compression and restores model-specific summarization search.
Experimental length preferences start at 15% / 25% / 35%, without forcing those counts.
OpenNMT's app input window is 1024 tokens (position/work bounded); its original
400-token training/fixed-benchmark policy is retained separately. Read the
[before/after review](./reports/quality_regression.json): known model salience and
factual-relationship failures remain; successful execution is not faithful-summary
certification. A frozen 40-article validation evaluation now has
[results and a source/reference review sheet](./reports/validation_40_review.html).
All four models completed all inputs, but substantive factual errors remain;
no default quality winner is established. See the
[protocol and limitations](./FULL_ARTICLE.md#frozen-40-article-validation-evaluation).
The completed 50-article test comparison is in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).
The larger 1,000-article run was interrupted and remains incomplete.

## CPU installation on Windows (PowerShell)

### Current production policy and launch

Reuse the existing environment:

```powershell
.\.venv\Scripts\python.exe app.py --processing-mode full
```

Open http://127.0.0.1:7860/. Auto prefers the existing CUDA GPU with the existing
CPU fallback. Stop the prior app terminal before restarting on the same port.

Natural policy uses checkpoint search settings and EOS. Initial output ceilings
are100 tokens for OpenNMT,141 new tokens for BART,127 for PEGASUS. Minimums are
bounded by source tokens divided by8 and the native floor (35/56/32 respectively;
HF converts its minimum to new tokens by subtracting the start token). These are
decoding limits, not word targets. Cap-without-EOS retries retain the bounded
256 legacy/512 HF hard limits and existing workload checks. Nothing is padded or
cut to meet an article percentage. Very tiny inputs may yield empty model output;
the app reports that explicitly rather than returning a fabricated summary.

Fitting articles use one source pass. Long articles retain sentence-aware chunks,
complete source coverage, ordered exact-sentence deduplication, and strict shrinking
consolidation if the merge exceeds the per-pass window. Native chunk budgets
retain model-specific headroom. Existing character/word/token/call/time/depth limits
and GPU memory handling remain; no new article cutoff was added. Processing failures
return no partial final summary. Processing all text does not guarantee factual
coverage. This UI simplification does not fix known model factuality limitations.

Statistics retain actual output words, compression against the complete input,
server processing time, and actual summed decoded tokens divided by measured
generation time (excluding loading/start/EOS). Separate verification:
[natural policy checks](reports/natural_policy_verification.json). The preceding
adaptive app/configuration is recoverable from
`reports/model_capabilities/application_snapshot.zip`; frozen historical evaluation
drivers intentionally reject changed generation hashes until that snapshot is
restored in a separate checkout. Original results/configurations are unchanged.

### Fresh CPU environment only

Run these commands from the repository root. Python 3.11 must be installed; the
existing project environment can be reused. No activation-policy changes or WSL
are necessary.

This installation block is for a fresh CPU-only environment. Skip the entire
block when using the existing working environment; installing the CPU wheel over
it would replace the working CUDA build. For a fresh GPU setup, use the separate
[GPU instructions](./GPU_SETUP.md).

```powershell
# Only if .venv does not already exist:
py -3.11 -m venv .venv

.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe OpenNMT_Verified_Setup\prepare.py
.\.venv\Scripts\python.exe -m pip check
```

Preparation requires Git, internet access, and roughly 6 GB of free space. Original
checkpoint downloads total 1,475,929,770 bytes (about 1.48 GB); inference copies
and third-party source are also stored locally. The tested setup scripts are
reused from [OpenNMT_Verified_Setup](./OpenNMT_Verified_Setup/README.md), originally
supplied as a ZIP. Do **not** install current `OpenNMT-py` over its legacy checkout.

The root [requirements](./requirements.txt) reuse the kit's pinned compatibility
dependencies plus Gradio 5.49.1, NLTK 3.9.2 and compatible HF libraries listed in
[GPU_SETUP.md](./GPU_SETUP.md). PyTorch remains version 2.5.1, accepting either
CPU or CUDA builds. To prefetch the added models, run the project interpreter
with [download_models.py](./download_models.py) and arguments `BART PEGASUS`.
Allow about 20 GB including all weights, inference copies and wheel caches. Model weights,
the downloaded OpenNMT checkout, caches, and virtual environments are Git-ignored.

## Launch

```powershell
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:7860>. Stop with Ctrl+C. The app binds to localhost and does
not create a public sharing link. Paste an English news article, select one of
the four models, choose Short / Medium / Detailed in the length dropdown, and click
**Summarize**. Medium is the default. GPU is preferred automatically; the current
demo has no device, truncation or technical-settings controls. Input and output
appear side by side on desktop and stack on mobile. Clear resets the article and
results; Copy is enabled only for a completed result. Editing the article or either
selection clears stale results. Controls are disabled while processing.

Each model is loaded lazily once per process and reused for later requests. The
first request for each model includes loading delay. Only the active model stays
on GPU; other cached models are offloaded to CPU RAM. Loading/device-transfer time
is logged separately; it is included in total request processing time but excluded
from generation throughput. The compact statistics show output words, "Shorter by"
(`100 * (1 - summary_words / article_words)`), processing time, and measured
generation tokens/second when available. Hover/focus the labels for measurement
definitions. Multi-pass requests use **Pipeline generation tok/s**: all actual
decoded output-token IDs across chunks, consolidation and completion retries,
divided by summed generation time. Decoder-start/EOS IDs are excluded. Missing
measurements, no generation or a failed CUDA attempt produce an unavailable rate,
not an estimate. Model tokenizers differ, so rates are not directly equivalent.

Generation time measures
dataset construction, CPU/CUDA decoding, and output cleanup, excluding model loading,
input tokenization, queue waiting, and browser/network latency. Requests are
queued one at a time; process and per-model locks also protect direct calls.
CUDA timings are synchronized. Explicit CUDA never silently falls back during
benchmarking; Auto reports CPU fallback if CUDA is unavailable or out of memory.

## Original fixed-budget backend policy (not the current demo)

This section documents the preserved [summarizer.py](./summarizer.py) backend API
and [benchmark_speed.py](./benchmark_speed.py), whose methodology is unchanged.
The updated application instead uses [article_summarizer.py](./article_summarizer.py)
and [processing_config.json](./processing_config.json); see [FULL_ARTICLE.md](./FULL_ARTICLE.md).

The original CNN/DailyMail setup truncated training inputs to 400 tokens. Here,
OpenNMT articles over 400 model-input tokens are **rejected by default**. An explicit
API argument permits summarizing only the first 400 tokens; the notices report the
total, used, and excluded token counts. The original article remains in the input
textbox. There is no hidden chunking, article replacement, or sentence deletion.
Tokens are not words: punctuation and split contractions also count.

Raw text is normalized for curly quotes, split at simple sentence boundaries,
PTB-style tokenized using NLTK's `TreebankWordTokenizer`, and lowercased, with
parentheses mapped to the checkpoint's bracket spelling. No NLTK corpus downloads
are needed. This is a practical approximation to the original Stanford/PTB
preprocessing, **not a claim of exact CoreNLP parity**; abbreviations and unusual
punctuation can differ. Both models use this same preprocessing and their own
saved vocabularies and copying machinery. Newlines remain part of the same
article, not separate examples. A source-consistency check catches additional
filtering or truncation inside OpenNMT.

BART and PEGASUS instead use their own original tokenizers, preserving case, with
1024-token encoder limits including special tokens. Their over-limit inputs are
also rejected unless supported-prefix truncation is explicitly approved. Their
tokens must not be equated with legacy tokens or words. These backend capacities
apply per chunk in the updated full-article app; there is no prefix-only default
or truncation checkbox in the current demo.

| Choice | Minimum decoding setting | Maximum decoding steps |
|---|---:|---:|
| Short | 10 | 35 |
| Medium | 30 | 65 |
| Detailed | 55 | 100 |

These are the verified kit's approximate decoding budgets, **not promised word
counts**. The legacy minimum setting includes its initial beam position; maximum
steps can include EOS. Sentence tags and punctuation consume steps. Models may
produce the same summary for multiple choices or end before the maximum; longer
does not guarantee better content. All presets use beam size 4, batch size 1,
two PyTorch CPU threads, and trigram blocking with `.`, `<t>`, `</t>` exempted.
No additional coverage or length penalties are introduced over the kit defaults.

Sentence tags are removed and tokens detokenized for display. Actual word count
is `len(summary.split())` on that displayed text. Output-limit warnings inspect
the selected hypothesis's actual EOS token, before it is removed by OpenNMT's
translation builder. If EOS is absent, the app warns that the output hit its cap
and may be unfinished. A separate missing-final-punctuation heuristic also warns
about possibly unfinished sentences. Text is never silently cut to a word target
or repaired by inventing an ending. EOS/punctuation do not establish factual
accuracy or grammatical completeness. These older models can repeat, omit key
facts, or hallucinate.

## Checkpoint and code provenance

Official documentation: [OpenNMT CNN/DailyMail summarization](https://opennmt.net/OpenNMT-py/examples/summary/Summarization.html).
The OpenNMT backends use the two complete pretrained systems, not the bottom-up content
selector or a separately trained coverage model.

| Model | Saved architecture | Unique parameters |
|---|---|---:|
| LSTM | One-layer bidirectional LSTM encoder, one-layer LSTM decoder, 128-dimensional embeddings, hidden size 512, additive attention and copying; no learned coverage attention | 42,789,205 |
| Transformer | Four encoder and four decoder layers, hidden size 512, shared source/target embeddings, positional encoding and copying | 81,468,757 |

Original, official checkpoint URLs (not re-hosted copies):

- LSTM: [ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20.pt](https://s3.amazonaws.com/opennmt-models/Summary/ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20.pt)
  — 363,581,349 bytes; SHA-256 `7e0092b066014af8b02c1221853ad48f0ee0db2182787baeb40abd9df2331d2e`.
- Transformer: [sum_transformer_model_acc_57.25_ppl_9.22_e16.pt](https://s3.amazonaws.com/opennmt-models/sum_transformer_model_acc_57.25_ppl_9.22_e16.pt)
  — 1,112,348,421 bytes; SHA-256 `683ef35e72d47cb33043e5db243b78a1386dd8a57e079c84b7642fba126c6605`.

These are recorded in the supplied [checkpoint manifest](./OpenNMT_Verified_Setup/checkpoint_manifest.json).
[prepare.py](./OpenNMT_Verified_Setup/prepare.py) clones official
[OpenNMT-py tag 0.2.1](https://github.com/OpenNMT/OpenNMT-py/tree/0.2.1), checks
commit `6db7ec1b3afae28c257a0416dd947022c3b12bc0`, and applies the kit's two
compatibility fixes: Boolean attention-mask inversion and integer beam-index
division. It downloads each original checkpoint, verifies its size and SHA-256,
then invokes the supplied [converter](./OpenNMT_Verified_Setup/convert_checkpoints.py).
The converter removes unused legacy optimizer state and writes `lstm_inference.pt`
and `transformer_inference.pt`; model/generator tensors and vocabularies are
retained and loaded strictly by OpenNMT. It deserializes these verified official
checkpoints; it is not an arbitrary checkpoint upload feature. Original upstream
code and model licensing still apply.

## Focused checks

```powershell
# Small deterministic checks; do not load model weights:
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Original fixed-budget CPU backend regression:
.\.venv\Scripts\python.exe tests\smoke_cpu.py

# Full-article checks using all four cached real models:
.\.venv\Scripts\python.exe tests\smoke_full_article.py

# Simple demo API and all model/length selections (temporary port 7861):
.\.venv\Scripts\python.exe tests\smoke_devices.py --device cuda
```

The CPU regression checks the six original OpenNMT fixed-budget cases and model
reuse independently of the demo. Full-article checks use a 1,200-word synthetic
article, all 12 model/preference pairs, oversized sentences and concise inputs;
actual outputs and coverage diagnostics are saved to ignored
`test-results/full_article.json`. The simplified API check saves
`test-results/cuda_interface.json`. Unit checks cover reduction, workload/depth/time
limits, Unicode splitting, adaptive targets, cache reuse and UI error clearing.
The kit's bundled [smoke_results.json](./OpenNMT_Verified_Setup/smoke_results.json)
contains its **earlier Linux** results, not this computer's measurements.

These are execution/integration checks, not a ROUGE evaluation or proof of summary
quality. Compare the systems on the same held-out articles for a class evaluation.

### Original native Windows CPU run (before the CUDA extension)

The implementation passed on Windows 11, Python 3.11.9, PyTorch 2.5.1+cpu,
and Gradio 5.49.1: 11 unit checks, all six live HTTP model/length combinations,
explicit truncation and blank-input checks, a forced output cap, and exactly two
model constructions (one per model). `pip check` and the application lint check
also passed. Actual word counts on the synthetic fixture were:

| Model | Short | Medium | Detailed |
|---|---:|---:|---:|
| LSTM | 16 | 23 | 47 |
| Transformer | 28 | 28 | 50 |

Generation took about 0.46-1.95 seconds on that run, excluding model loading.
Transformer Short and Medium were identical; length presets are not guarantees.
Legacy PyTorch/torchtext warnings can appear in the terminal; the checks above
completed successfully without changing the kit's inference behavior.
## Small ROUGE comparison (latest scope)

The user reduced the scope to a rough comparison. See
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for **50 fixed CNN/DailyMail test
articles**, the four models and Lead-3, with ROUGE scores. Shared-prefix and
full-article results remain separate. The 1,000-article run was stopped; its saved
partial results and historical development results are preserved, not presented
as a completed final benchmark.

[Riverton diagnosis](reports/RIVERTON_DIAGNOSIS.md) and
[model integrity evidence](reports/final_model_verification.json) distinguish
verified source delivery from model factual quality. The synthetic advisory
omission persists in direct PEGASUS inference; no generation change was justified.
The UI and existing environment are preserved.

Reproduce the small comparison from this repository in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe tests\final_benchmark.py run --limit 50
.\.venv\Scripts\python.exe tests\report_pilot50.py
```

The runner rejects incompatible code, dependencies, weights or inputs instead of
pooling results. Small-comparison metrics, IDs, outputs and references are under
`reports/pilot50/`. Local exact inputs/traces remain under ignored `data/` and
`test-results/`; the interrupted larger study remains under `reports/final_test_1000/`.
Do not resume the larger study without a new request. No human factuality rate or
factual-quality winner is established by ROUGE. GPU contention limits timing comparisons.
