# Full-article application

The update reuses all four cached pretrained backends, the existing Python 3.11
environment, automatic CUDA/CPU fallback and lazy model cache. No dependencies or
checkpoints were reinstalled. Benchmark scripts, their fixed-budget policy and
saved comparison reports are separate and unchanged.

## Launch

```powershell
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:7860>. News Summarizer has native model and length dropdowns
(LSTM and Medium by default), balanced Article and read-only Summary panels,
Summarize, Clear and Copy summary. Panels stack on mobile. A transient progress indicator reports
chunks completed and consolidation passes. Brief warnings/errors do not add
technical panels. A failed request clears the previous Summary rather than
leaving stale or partial output visible. Edits to input or selections invalidate
old results; processing disables editing, Clear and duplicate submissions.

Below Summary, output words and "Shorter by" use the complete original input's
whitespace word count. Processing time includes model loading, preprocessing and
consolidation, but not Gradio queue wait or browser/network time. Throughput uses
the existing backends' actual decoded token-ID counts (excluding start/EOS), not
word estimates. Single-pass generation tokens/second excludes loading; pipeline
generation tokens/second sums tokens and synchronized backend generation times
over every chunk, consolidation pass and completion retry. Intermediate or
discarded retry outputs count as work. Tokenizers differ across models, so rates
are not directly equivalent. No-generation and incompletely measured/OOM requests
show an unavailable rate. These additions are observability only: model weights,
token budgets, decoding, CPU fallback and benchmark methodology are unchanged.

GPU is preferred automatically. Models/tokenizers remain lazily cached; only the
active model resides on GPU, with inactive models in CPU RAM. Existing CPU
fallback is preserved. `SUMMARIZER_DEVICE=cpu` can force CPU outside the demo.

## Complete source processing

- A corpus-free Punkt sentence segmenter, with common English abbreviations,
  identifies preferred boundaries. Sentence segmentation remains heuristic.
- Contiguous, non-overlapping spans of the original string are packed using the
  selected backend's tokenizer. The app uses a bounded 1024-token OpenNMT work
  window, capped by encoder positional capacity. The 400-token training convention
  remains the fixed-budget API/benchmark policy, not a hard network limit.
  BART/PEGASUS allow 1024 including special tokens. Actual inference never truncates.
- Oversized sentences split at verified fitting word boundaries. Enormous words
  or URLs fall back to Unicode codepoint boundaries, never partial UTF-8 bytes.
- Concatenating the planned chunks must reconstruct the complete source exactly,
  including whitespace. Every original chunk is processed in source order.
- Intermediate summaries are joined in order. Only exact duplicate complete
  output sentences are removed, keeping the first. Source text is not deduplicated.
- Intermediate text is re-chunked and consolidated as necessary. Every reduction
  stage must strictly shrink both word count and backend-token count, including
  the final consolidation. A failure to shrink aborts, not loops indefinitely.
- If the source-order, exact-deduplicated intermediate merge already fits the
  word preference and input window, it is returned without another lossy decode.
  Consolidation passes count additional generative passes, not this simple merge.

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
| Short | 15% |
| Medium | 25% |
| Detailed | 35% |

Words are whitespace-delimited, consistent with the existing word-count policy.
For example, 1,200 words request approximately 180 / 300 / 420 words. These are
configurable initial preferences, not calibrated or guaranteed counts.

Model-specific token/word estimates start at 1.6 for OpenNMT and PEGASUS, 1.5
for BART. The selected tokenizer's observed source token/word ratio can raise
that estimate; configurable slack allows for subwords, punctuation and sentence
tags. This approximates output tokenization; it does not equate tokens and words.
Chunk targets allocate detail proportionally and preserve intermediate context.
Consolidation targets are also bounded by current text size to ensure reduction.
The app uses checkpoint-specific HF search settings, and the pinned OpenNMT
summarization recipe (beam 5, Wu length penalty alpha 0.9, summary coverage beta
5, stepwise penalty and existing trigram blocking). The original benchmark API
keeps its legacy profile. Request-specific translator/source settings are restored
even on failure, so cached models do not leak settings into later callers.
Minimum lengths use the documented model floor, bounded to at most one quarter
of the source's tokens and below its maximum for short inputs. Native maximums
provide decoding headroom, not required output length. EOS may stop early without
padding or invented details.

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
| App OpenNMT input window (configurable, positional-limit bounded) | 1024 tokens |
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

For UI-only changes, use the fast unit checks and optional installed-Chrome check:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests\check_ui_browser.py
```

The November 21 UI redesign passed 45 unit checks and rendered checks at
320/390/760/1280px, including actual clipboard round-trip, native dropdown
keyboard selection/Escape/reopening, busy/duplicate-submission states, progress,
error preservation, Clear, stale-result invalidation and action-button clipping.
Browser checks use deterministic test-only decoders on a temporary localhost
server; they do not load weights or claim model/benchmark performance. Screenshots
and `ui_browser.json` are generated under ignored `test-results/`. Gradio 5's
readonly-dropdown reopening after Escape has a small delegated UI workaround;
the native options and keyboard selection remain in use.

Live LSTM CUDA inference on a short article also passed through the updated app:
output words/compression were recomputed, and displayed generation throughput
matched logged actual token counts divided by synchronized generation time.
This is a focused integration check, not a rerun of the model benchmarks.
The Gradio client returns summary/statistics/status; the fourth UI output is a
Copy-button update, which the client omits from its returned values.

The separate real-model smoke checks remain available, but were not repeated
for this UI redesign:

```powershell
.\.venv\Scripts\python.exe tests\smoke_full_article.py
.\.venv\Scripts\python.exe tests\smoke_devices.py --device cuda
```

The real-model check uses a 1,200-word synthetic article, all 12 model/preference
pairs, four single-oversized-sentence variants and four explicit concise-input
cases. It checks coverage, strict reduction, termination, adaptive targets and
model identity reuse. Actual summaries and diagnostics are saved to ignored
`test-results/full_article.json`. The API smoke script is adapted to dropdown
labels and the four-result UI response (summary, statistics, status, Copy state).
Unit checks also simulate failed reduction,
empty-fragment recovery and all safeguards. These are focused execution checks,
not quality evaluation or benchmark tuning.

The original [benchmark_speed.py](./benchmark_speed.py), fixed-budget
[summarizer.summarize](./summarizer.py) and saved speed reports are intentionally
separate. Historical fixed-budget timings do not measure this multi-pass app.

### Results of this update

Original saved verification on Windows with the existing Python 3.11 environment
and RTX 4090 (not repeated for the UI-only redesign):

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

## Focused quality diagnosis (November 21)

The user-pasted Huang article was treated as **unverified source text**, not news
verified by outside knowledge. It contains 518 words: OpenNMT 580 input tokens,
BART 654 (including two special tokens), PEGASUS 608 (including one). All four
now read it once, without chunking or consolidation; both legacy encoders were
also tested directly at 580 tokens before relaxing the old 400-token app policy.
That larger window is a bounded inference choice, not retraining or a promise of
quality beyond the original training convention.

Before/after Medium lengths were LSTM 17/46, Transformer 12/47, BART 53/52,
PEGASUS 25/68 words; the preference is now about 130 words, not a forced minimum.
PEGASUS's restored checkpoint search recovers the supplied main claim, whereas
raising only its minimum length did not. BART and the OpenNMT models still fail
to select that event reliably in direct inference: those limitations are **not
solved** by allowing more words. Repeated ideas and entity/negation/attribution
errors remain on other examples; a longer summary is not a factuality guarantee.

The fixed development check uses that input, CNN/DailyMail validation rows 0/1
(with their existing highlights), and one short synthetic uncertainty fixture.
The 1,324-word validation article genuinely exceeds every 1024-token input window
and is processed in two complete chunks. No final test split or large benchmark
was used. All 16 final cases cover the exact source and end via EOS; fitting
inputs use one model call. Manual source/reference review, not ROUGE or keyword
counts presented as factuality certification, records both improvements and failures.

[quality_regression.json](./reports/quality_regression.json) contains the detailed
review, settings and before/after comparisons. Exact inputs, chunks, raw output
IDs, pre-cleanup text, displayed results and direct/ablation controls are local
and ignored under `test-results/quality_regression/`. `before/` is the baseline;
`after_capacity/` is the final run; `after/` and `after_final/` preserve earlier
diagnostic candidates. The driver is [diagnose_quality.py](./tests/diagnose_quality.py).
54 unit checks cover the actual fixes; none hardcode a neural output sentence as
the only acceptable summary. The UI, checkpoints, dependencies and fixed-budget
benchmark methodology were not changed by this quality pass.

## Frozen 40-article validation evaluation

This November 21 evaluation keeps the demonstrated fixes and all generation
settings unchanged. It evaluates Medium on the same 40 CNN/DailyMail 3.0.0
**validation** articles for every model: fixed rows 2-41, selected before
generation. Rows 0/1 used in development and the Jensen example are excluded.
The sample is consecutive, not random or population-representative. No final
test data, training, checkpoint replacement or further parameter tuning is used.

The exact diagnosis input is local `test-results/quality_regression/huang_article.txt`:
518 words by `len(text.split())`, 3,066 characters including its final newline.
Hyphenated words count once; a standalone dash also counts once. Opening sentence:
"Jensen Huang, known for sporting a black leather jacket to shareholder events,
appeared to have it all." Closing sentence: "It now creates the key processing
chips used to power the growing AI technology." Its UTF-8 SHA-256 is
`fd48270a2513baf53a5598394b934d8a9d8c4c016f5758373f936bd0bbf926fd`.
It is unverified user-supplied text, not verified news, and is not scored here.

### Protocol and saved evidence

- [Manifest](./reports/validation_40_manifest.json): exact IDs, article/reference
  hashes, generation-source/configuration hashes, checkpoint provenance and
  sampling policy. Source word counts range from 81 to 1,354.
- [Detailed results](./reports/validation_40_results.json): all 160 per-case
  scores, word counts/compression, original model-specific input tokens, capacity,
  chunk counts, source coverage, requested word targets, actual per-call token
  limits/decoding overrides, output tokens, EOS, consolidation counts and timings.
  HF checkpoint generation configurations and explicit overrides are separate.
- [Offline review sheet](./reports/validation_40_review.html): each complete
  original article, original reference and all four generated summaries, with
  160 **AI-assisted** source-comparison notes. Every human-review field remains
  blank/not reviewed in the [editable CSV](./reports/validation_40_review.csv).
  The [AI notes](./reports/validation_40_ai_review.json) are not human ratings.
- Exact dataset snapshot: ignored `data/validation_40/sources.json`. Exact
  decoder inputs, chunk contents, raw output IDs, pre-cleanup and displayed text:
  ignored `test-results/validation_40/cases/`. Loading/environment records are
  under `test-results/validation_40/`. Report generation reuses these traces.

All runs used the same RTX 4090 CUDA device and float32; CPU fallback was not
mixed into timings. Models were loaded once and reused; only the active model
occupied GPU memory. Synchronized generation time is summed across chunk calls;
request wall time is recorded separately. Loading/transfer was measured separately
in this evaluation process. There was no explicit warmup or repeated latency
trial, so first-use kernel costs and machine load can affect these timings.

All four input windows were bounded at 1,024 tokenizer tokens. OpenNMT uses
legacy PTB-style lowercase preprocessing; BART/PEGASUS preserve case with their
own tokenizers. OpenNMT's original training/fixed-benchmark source convention
remains 400 tokens; running the app at 1,024 is a capacity policy, not evidence
that long-input model quality is validated. Frozen Medium targets start at 25%
of original words (20-339 words here). Actual per-call maxima ranged 100-256
tokens for legacy models, 141-512 for BART and 127-512 for PEGASUS. Checkpoint
minimum lengths are bounded for short inputs; exact actual settings are saved.
Budgets are preferences: EOS can yield much shorter summaries.

ROUGE uses Google's `rouge-score==0.1.2`, Porter stemming, no stopword removal,
and mean per-article F1 multiplied by 100. ROUGE-L is **flat LCS**, not ROUGE-Lsum;
these numbers should not be compared directly with published Lsum results.
CSV per-case ROUGE is on 0-1 scale; aggregate JSON and HTML are on 0-100 scale.
Failed generations would contribute zero to the primary 40-case mean; none failed.
Repetition is excess repeated trigrams divided by total trigrams after casefolding
and regex word tokenization. Exact repeated sentences are a separate heuristic.
Neither metric measures factual accuracy.

### Observations and limits

| Model | R1 / R2 / RL F1 | Mean words (range) | Mean shorter by | Mean generation s | Load/transfer s |
|---|---|---|---:|---:|---:|
| LSTM | 35.45 / 16.64 / 26.11 | 41.7 (21-127) | 86.8% | 0.553 | 0.486 |
| Transformer | 39.67 / 20.34 / 31.32 | 36.3 (20-85) | 87.5% | 0.673 | 1.146 |
| BART | 39.42 / 20.02 / 29.89 | 51.0 (25-102) | 82.5% | 0.785 | 1.805 |
| PEGASUS | 39.09 / 20.28 / 30.15 | 39.0 (20-81) | 86.3% | 0.978 | 3.141 |

160/160 completed with full source coverage, no input truncation and EOS; every
model had 38 one-chunk inputs and two two-chunk inputs. The latter used the
existing source-order merge; no additional generative consolidation was needed.
There were zero failed jobs, zero maximum-length unfinished outputs and no exact
duplicate sentences. Mean excess-trigram rates were 0%, 0%, 0.09%, 0.63% in table
order; PEGASUS had one output above 10%. Missing-final-punctuation flags were
3/0/2/2, but punctuation/EOS cannot establish grammatical or semantic completeness:
LSTM row 34 ends with the incomplete clause "after the car he was riding."

Representative qualitative observations below are **AI-assisted comparisons**,
not human factuality scores; the sheet preserves full source and reference evidence:

- Row 2 BART correctly selects Gomis's collapse, reported wellbeing and Swansea
  manager's assessment. This is a plausible useful summary, not blanket validation.
- Row 19 Transformer says "real madrid beat athletic bilbao 1-0 to move up to
  eighth in ligue 1" and gives Bale a header; the source says Real lost, Bilbao
  moved eighth, the league was La Liga and Aduriz scored. These errors already
  appear in raw single-pass output before cleanup.
- Row 21, only 81 source words, describes a car bomb killing seven and injuring
  23. All four omit the main bombing event; BART outputs just the time/location
  and that identity/motive were unavailable. Complete coverage did not preserve
  the main event, and an increased ceiling did not prevent early EOS.
- Row 30 PEGASUS repeats the Cambodia-flight-location claim twice in near-identical
  sentences. Exact-sentence duplication misses it; repeated trigrams flag it.
- Row 35 BART invents author "Frida Ghitis" although the highlights name Donna
  Brazile; other backends also misattribute this opinion article. BART's wrong
  author is already in raw single-pass output and survives cleanup.
- Row 38 LSTM/Transformer risk conflating Oklahoma medical and Florida military
  helicopter crashes; Transformer attaches the military death toll to the medical
  flight context. Row 41 LSTM attributes a researcher's quotation to Pope Francis.

Two CNN Student News template pages, rows 10/14, share identical highlights.
Models reproduce template material, including reference language not explicitly
present in the source, rather than episode events. This raises a dataset/reference
concern; it does **not** establish memorization or train/validation leakage.
Primary results retain all 40. A clearly labeled **post-hoc** 38-page sensitivity
has R1/R2/RL: LSTM 32.61/12.92/22.78; Transformer 36.59/16.29/27.80;
BART 36.23/15.82/26.20; PEGASUS 35.89/16.09/26.48. References can also omit a
main event or contain a claim absent from the supplied article. They are overlap
targets, not exhaustive factuality oracles.

The 2,000 bootstrap resamples (seed 20261121) quantify variation within these
40 articles, not population or factuality confidence. BART-minus-Transformer
ROUGE-L difference is -1.43, interval [-5.36, 2.09]; BART-minus-PEGASUS -0.26,
[-3.11, 2.64]. There is no stable quality-default recommendation, and the UI
default has not been changed. Transformer leads lexical scores but makes serious
entity/relationship errors. More output words are not a quality guarantee.

Evidence points to decoder salience, entity and attribution limitations for
several failures: they arise in raw, fitting, single-pass generation, independent
of consolidation and final cleanup. It does not isolate checkpoint behavior
from every preprocessing/search choice. OpenNMT preprocessing approximation and
its expanded input window remain possible contributors requiring controlled
evidence. Only two articles exercise multi-chunk processing; no new generative
consolidation-depth evidence is provided. No Short/Detailed comparison was run.
Human rubric review of all four summaries against each source remains necessary;
no factuality pass rate or fabricated human score is reported.

### Reproduce or inspect without repeating setup

Existing results require no inference rerun. From repository-root PowerShell:

```powershell
.\.venv\Scripts\python.exe tests\evaluate_validation.py report
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_validation_report.py -v
Start-Process .\reports\validation_40_review.html
```

Only if the isolated evaluation metric folder is missing (the application
environment remains unchanged):

```powershell
.\.venv\Scripts\python.exe -m pip install --target .cache\eval-metrics --no-deps --use-pep517 rouge-score==0.1.2 absl-py==2.3.1
```

Source recovery, if missing, downloads only the same 40 validation rows and
requires exact ID/content/snapshot hash matches to the existing manifest:

```powershell
.\.venv\Scripts\python.exe tests\evaluate_validation.py prepare
```

For a future explicit rerun with the exact frozen application/configuration and
existing checkpoints, `tests\evaluate_validation.py run` generates missing case
traces only. It refuses changed generation hashes and skips an already complete
run. The current report and review sheet are self-contained saved deliverables;
regenerating the report also needs the ignored raw traces. Six focused metric,
paired-ID, hash, coverage/settings and review-separation audit tests passed, plus
lint/format checks of the evaluation files. No model setup was repeated and the
app/UI/generation files were not edited during this evaluation.

## Controlled factual-error audit

Five cases from that validation sample were frozen: reversed match result (row
19), invented author (35), missed bombing (21), repeated PEGASUS claim (30) and
the two-chunk Runcible article (39). All four models are included. This is a
targeted diagnosis, not a quality-rate estimate or parameter sweep. The
[manifest](./reports/factual_audit_manifest.json) protects app/configuration and
original validation evidence by SHA-256.

[Review report](./reports/factual_audit_review.html),
[structured comparisons](./reports/factual_audit_results.json), and
[summary CSV](./reports/factual_audit_summary.csv) preserve all findings separately
from the original results. Exact sources, references, prepared tokens/tensors,
raw output IDs/text, displayed outputs and effective decoding settings are local
under ignored `test-results/factual_audit/sources/` and `controls/`, linked in the
HTML. No existing evaluation files were overwritten.

The direct OpenNMT path calls pinned `Translator.translate` independently of the
app generation adapter, retaining checkpoint vocabularies and dynamic copying.
HF calls its checkpoint tokenizer and `model.generate` directly. All 172 calls
used RTX 4090 CUDA float32 and emitted EOS. All 24 direct app-setting raw outputs
matched saved IDs; all 20 merged summaries matched. These are software checks,
not factuality evidence.

Each HF decoding edge changes one factor: checkpoint defaults, then forced EOS
disabled, then adaptive maximum, then adaptive minimum; a minimum-only branch
is also saved. Legacy controls vary the 35/100 recipe's maximum and minimum
separately. Its pinned command uses beam 5, summary coverage beta 5, Wu alpha
0.9, stepwise penalty and trigram blocking with boundary exceptions. The pinned
prose says alpha 0.8 while the explicit command says 0.9; today's website uses
beam 10. This audit follows the pinned explicit command, matching the app.

Original CNN/DailyMail preparation used Stanford CoreNLP PTB tokenization then
lowercasing, original-line missing-period repairs, and target sentence tags.
OpenNMT uses the 400-source/100-target convention with shared/dynamic vocabulary;
`<t>` tags belong on targets, not articles. The upstream data README names
CoreNLP 3.7.0. That tokenizer and a portable Java runtime were downloaded only to
ignored `.cache/factual-audit-tools/`; no Python packages, PATH, weights or system
installation changed. URLs and publisher checksums are recorded. Flattened HF
articles lack original `.story` line boundaries, so exact historical caption
period repair cannot be recovered.

Stanford differs from the NLTK approximation at quoted sentence endings and
abbreviations: `front.` versus `front` + `.`, and `p.m.`, `Mo.`, `Det.`. Fixed-source,
fixed-decoding swaps change some wording, but do not consistently remove errors.
The separate 400/1024 comparisons share token streams and decoding, while
explicitly varying available prefix content. They record omitted tokens and are
partial unless the full source fits. In the match article NLTK400 changes
Transformer's result reversal but leaves the wrong goalscorer; Stanford400
reverses the result again. LSTM is correct with expanded input and wrong at400
under both tokenizers. This does not support a global window rollback.

BART invents the same author under every decoding control. The source has no
byline naming either that author or reference author Donna Brazile. All four
still omit the main bombing event with documented defaults; raising BART's
minimum adds secondary detail without recovering it. PEGASUS repeats the same
claim with unchanged checkpoint defaults and naturally ends below the native
cap. In row39, Transformer omits Monohm/Runcible in the first raw chunk output;
merging does not introduce that omission and no extra generative consolidation
occurs. These observations do not prove that chunk boundaries or deeper
hierarchical consolidation are optimal or faithful.

Template rows10/14 are retained: March10 solar-plane/crater/Everest introductions
and March9 Selma/MH370/International Women's Day/unemployment introductions
precede common transcript/roll-call boilerplate. Identical highlights emphasize
that boilerplate rather than the differing episode topics. All models reproduce
template-like language, including a reading/vocabulary instruction absent from
the articles. This raises a reference-artifact concern, not proof of memorization
or leakage. Original40 scores and the labeled post-hoc38 sensitivity remain
byte-for-byte unchanged; no example is removed for poor model performance.

**No production change is supported or proposed.** No candidate is promoted;
reserved validation rows42-47 are therefore not run as candidate validation.
Any future proposal needs an independent small-sample check before adoption.
Six focused tests check one-factor edges, source/ID equality, window labels,
evidence preservation and review separation; none requires an exact generated
sentence. All qualitative judgments are AI-assisted; no human ratings are claimed.

Inspect saved evidence without additional inference:

```powershell
.\.venv\Scripts\python.exe tests\report_factual_audit.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_factual_audit.py -v
Start-Process .\reports\factual_audit_review.html
```

For explicit reproduction only: `tests/prepare_factual_audit.py` recovers isolated
tools (network required); `tests/audit_factual_failures.py prepare` verifies the
frozen protocol; `tests/audit_factual_failures.py run` generates missing controls
and preserves/skips existing files. Regenerating the report requires ignored raw
traces. Saved HTML/JSON/CSV are directly readable. The UI, training state,
checkpoints and final-test benchmark remain unchanged; nothing was pushed.
