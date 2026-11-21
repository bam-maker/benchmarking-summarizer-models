# Full-article application

The update reuses all four cached pretrained backends, the existing Python 3.11
environment, automatic CUDA/CPU fallback and lazy model cache. No dependencies or
checkpoints were reinstalled. Benchmark scripts, their fixed-budget policy and
saved comparison reports are separate and unchanged.

## Launch

```powershell
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:7860>. The only main elements are Article, a horizontal
four-model selector, a three-position Short/Medium/Detailed slider (Medium by
default), Summarize and read-only Summary. A transient progress indicator reports
chunks completed and consolidation passes. Brief warnings/errors do not add
technical panels. A failed request clears the previous Summary rather than
leaving stale or partial output visible.

GPU is preferred automatically. Models/tokenizers remain lazily cached; only the
active model resides on GPU, with inactive models in CPU RAM. Existing CPU
fallback is preserved. `SUMMARIZER_DEVICE=cpu` can force CPU outside the demo.

## Complete source processing

- A corpus-free Punkt sentence segmenter, with common English abbreviations,
  identifies preferred boundaries. Sentence segmentation remains heuristic.
- Contiguous, non-overlapping spans of the original string are packed using the
  selected backend's tokenizer. OpenNMT's capacity is 400 tokens; BART/PEGASUS
  allow 1024 including special tokens. Actual inference preparation never truncates.
- Oversized sentences split at verified fitting word boundaries. Enormous words
  or URLs fall back to Unicode codepoint boundaries, never partial UTF-8 bytes.
- Concatenating the planned chunks must reconstruct the complete source exactly,
  including whitespace. Every original chunk is processed in source order.
- Intermediate summaries are joined in order. Only exact duplicate complete
  output sentences are removed, keeping the first. Source text is not deduplicated.
- Intermediate text is re-chunked and consolidated as necessary. Every reduction
  stage must strictly shrink both word count and backend-token count, including
  the final consolidation. A failure to shrink aborts, not loops indefinitely.

A tiny tail of an oversized sentence can make an older news model emit only EOS.
Its source is still presented to the model; if an intermediate decode is empty
and the fragment is at most 12 words/240 characters, its exact text is retained
for consolidation and recorded as a retained fragment, not called a generated
summary. Other empty chunk/final results abort. No source suffix is silently lost.

Internal diagnostics track original/covered characters, exact chunk spans,
chunk counts, model calls, total token work, reduction stages and passes. All
source chunks must have been processed before a full result is returned.
Coverage is a **processing guarantee**, not proof that every fact survives
abstractive summarization. Whole concise-input passthrough is explicitly labeled.

## Adaptive length

The final requested word target is based on the **complete original article**:

`min(article_words, max(1, round(article_words * ratio)))`

| Choice | Starting ratio |
|---|---:|
| Short | 10% |
| Medium | 15% |
| Detailed | 22% |

Words are whitespace-delimited, consistent with the existing word-count policy.
For example, 1,200 words request approximately 120 / 180 / 264 words. These are
configurable initial preferences, not calibrated or guaranteed counts.

Model-specific token/word estimates start at 1.6 for OpenNMT and PEGASUS, 1.5
for BART. The selected tokenizer's observed source token/word ratio can raise
that estimate; configurable slack allows for subwords, punctuation and sentence
tags. This approximates output tokenization; it does not equate tokens and words.
Chunk targets allocate detail proportionally and preserve intermediate context.
Consolidation targets are also bounded by current text size to ensure reduction.
Minimum-length requirements are modest and reduced for short inputs; EOS may
stop generation early without padding or invented details.

Application output caps default to 256 tokens for OpenNMT and 512 for BART/PEGASUS.
Actual decoder positional limits are also enforced. The LSTM has no positional
table limit, so its configurable work cap bounds decoding. OpenNMT was trained
with a 100-token target convention: a larger application budget is not evidence
of validated long-summary quality. Its benchmark settings remain unchanged.

No generated sentence is post-trimmed to reach a word count or repaired with an
invented ending. A missing EOS at the soft budget triggers a bounded retry with
more decoding room, up to the configurable hard cap and architecture limit. All
retries count against total workload/time. If EOS is still missing at that cap,
the job fails without returning unfinished text. Missing final punctuation after
natural EOS and large length misses are reported honestly. Inputs of at most 12
words and 240 characters are returned
unchanged with an explicit already-concise notice, without model generation.

## Configuration and safeguards

Edit [processing_config.json](./processing_config.json), use `--config custom.json`,
or select a file via `SUMMARIZER_CONFIG`. Use the supplied schema for custom files.

| Limit | Default |
|---|---:|
| Original characters / words | 120,000 / 12,000 |
| Source chunks / chunks per stage | 64 |
| Model calls, including retries | 96 |
| Total input tokens over all calls/retries | 64,000 |
| Consolidation passes after the initial chunk stage | 4 |
| Processing time | 180 seconds |

Limits are checked before work and around each model call. The time guard is
cooperative: it cannot interrupt an in-flight PyTorch kernel or blocking model
load. One bounded call can overrun the deadline, but the result is then rejected
and no further work/final output is returned. Work caps and strict reduction
checks independently prevent unbounded processing; CPU recovery cannot bypass them.

If a guard, generation failure or non-progress stage prevents completion, no
partial final summary is returned. Intermediate outputs remain internal and the
complete Article field remains unchanged. Detailed diagnostics go to the local
process log without article bodies; loading and generation times stay separate.

Prefix-only processing is an explicit faster alternative outside the demo:

```powershell
.\.venv\Scripts\python.exe app.py --processing-mode prefix
```

It can also be selected with `processing_mode` in configuration or
`SUMMARIZER_PROCESSING_MODE=prefix`. Prefix results are labeled with used/original
word counts and have incomplete-coverage diagnostics. Full processing is default.

## Focused verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests\smoke_full_article.py
.\.venv\Scripts\python.exe tests\smoke_devices.py --device cuda
```

The real-model check uses a 1,200-word synthetic article, all 12 model/preference
pairs, four single-oversized-sentence variants and four explicit concise-input
cases. It checks coverage, strict reduction, termination, adaptive targets and
model identity reuse. Actual summaries and diagnostics are saved to ignored
`test-results/full_article.json`. The API check covers all 12 real model/slider
selections in the one-output demo. Unit checks also simulate failed reduction,
empty-fragment recovery and all safeguards. These are focused execution checks,
not quality evaluation or benchmark tuning.

The original [benchmark_speed.py](./benchmark_speed.py), fixed-budget
[summarizer.summarize](./summarizer.py) and saved speed reports are intentionally
separate. Historical fixed-budget timings do not measure this multi-pass app.

### Results of this update

Verified locally on Windows with the existing Python 3.11 environment and RTX 4090:

- 38 unit checks passed, including complete coverage, oversized Unicode words,
  strict reduction, all workload/depth/time guards, EOS retries, CUDA recovery,
  model reuse and the five-component interface.
- All four models generated real CUDA summaries for every length preference on
  the 1,200-word fixture (12 cases), plus four oversized-sentence cases. Every
  source character was covered and every consolidation stage strictly reduced
  words and tokens. Four already-concise cases explicitly returned their input.
- All 12 model/length combinations passed through the real Gradio API. Cached
  model identity survived reselection, with at most one model resident on GPU.
- Real LSTM inference also passed on CPU through the new full-article path.
  CUDA out-of-memory recovery was simulated, not deliberately induced on hardware.
- Browser checks at 390px and 1280px widths passed: horizontal model choices,
  all three live length labels, read-only output, no horizontal overflow, real
  long-article generation and visible chunk/consolidation progress. An over-limit
  request preserved all 120,001 input characters, displayed the processing-limit
  notice and cleared the previous summary.

Remaining limitations: models may stop far short of a preference and produce
identical summaries across length choices. All four undershot targets on the
repetitive synthetic long fixture; successful coverage is not calibrated length
control. Hierarchical summarization can lose facts or introduce unsupported claims;
source-order processing does not prove factual accuracy or complete fact coverage.
