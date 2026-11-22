# Final CNN/DailyMail benchmark

**IN PROGRESS / PARTIAL: 2,274 of 10,000 system/article/track evaluations attempted.**
Each completed track targets the same 1,000 randomly sampled official CNN/DailyMail
3.0.0 test articles, seed 42. This is a complete-system comparison, not an isolated
architecture experiment. Production policy: **natural-checkpoint-v1**.

No training, checkpoint replacement, dependency reinstall, UI change or test-based
generation tuning was performed. Riverton remained separate synthetic diagnostic
data. Its PEGASUS omission appears in raw direct inference with the complete input;
[the diagnosis](reports/RIVERTON_DIAGNOSIS.md) explains why no generation fix was justified.

## A. Common-input comparison

| System | Attempted / 1000 | Failed | R-1 | R-2 | R-L | R-Lsum | Words | Median / p95 seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 1000 | 0 | 38.67 | 17.15 | 26.72 | 35.71 | 48.7 | 1.105 / 2.473 |
| Transformer | 122 | 0 | 38.91 | 18.01 | 27.45 | 35.60 | 46.3 | 1.599 / 3.127 |
| BART | 5 | 0 | 43.60 | 20.59 | 29.11 | 40.48 | 59.4 | 0.993 / 1.055 |
| PEGASUS | 5 | 0 | 41.64 | 22.81 | 31.23 | 37.83 | 42.8 | 1.164 / 1.386 |
| Lead-3 | 5 | 0 | 38.35 | 18.40 | 23.41 | 32.63 | 79.0 | 0.000228 / 0.000296 |

Each system receives the **same exact original text prefix**, bounded by 400
production OpenNMT tokens and each HF tokenizer's 1024-token capacity including
specials. OpenNMT retains its existing approximate PTB preprocessing. References
are the complete original highlights, so they may describe omitted source text.
This is explicitly a prefix diagnostic/comparison, not full-article coverage.

## B. Full-article application comparison

| System | Attempted / 1000 | Failed | R-1 | R-2 | R-L | R-Lsum | Words | Median / p95 seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 1000 | 1 | 38.71 | 16.91 | 26.37 | 35.75 | 58.0 | 1.180 / 3.190 |
| Transformer | 122 | 0 | 39.17 | 17.28 | 26.66 | 36.14 | 49.4 | 1.718 / 3.727 |
| BART | 5 | 0 | 39.62 | 19.34 | 26.13 | 36.03 | 64.2 | 1.094 / 1.258 |
| PEGASUS | 5 | 0 | 38.67 | 20.54 | 28.55 | 34.58 | 55.8 | 1.269 / 1.885 |
| Lead-3 | 5 | 0 | 38.35 | 18.40 | 23.41 | 32.63 | 79.0 | 0.000354 / 0.000444 |

All original text is submitted to the production coverage-checked pipeline.
Completed neural requests cover every source character; aborted requests are
reported as failures, never partial completed summaries. Lead-3 selects the first
three sentences using the same sentence segmenter. It reads full input but does
not claim to represent later facts. Source delivery is not factual coverage.

## Measurement and denominators

ROUGE-1/2/L/Lsum are F1 ×100, rouge-score 0.1.2 with Porter stemming. Lsum receives
newline-separated sentences from the same deterministic Punkt segmenter for both
reference and output. Failed/empty final summaries score **zero** and remain in
all attempted-case quality means. Pending cases are excluded and explicitly shown.
Words use whitespace splitting. Compression is summary words / original article
words, averaged per case; supplied-input compression is also saved. Repetition is
excess duplicate lowercase regex-word trigrams divided by all output trigrams.

Timing above is median/p95 **completed-request generation time**, summed across
all decode calls and retries, synchronized on the RTX 4090; excludes model loading,
queueing and input tokenization. All neural systems use CUDA FP32, TF32 disabled,
batch size one and one excluded fixed warmup per process. Full request wall times
and all-attempt latency statistics are saved separately. Lead-3 timing is **CPU
sentence extraction**, not a GPU/model speed comparison. This is a local Windows
desktop measurement, not an isolated hardware throughput study.

**GPU contention limitation:** a spot check found Fortnite and other desktop GPU
contexts present alongside Python. Per-process utilization/memory was unavailable
under Windows WDDM, so contention cannot be quantified or corrected. These times
are observed local latencies, **not evidence for a controlled speed ranking**.
No unrelated application was stopped. See
[hardware observation](reports/final_test_1000/hardware_observation.json).

BERTScore was omitted: its package and scorer weights were not installed/cached;
the working environment was preserved. ROUGE is reference similarity, **not proof
of factual correctness**. Neither longer output nor successful execution establishes
faithfulness. Native EOS does not establish grammatical or factual completeness.

## Coverage, repetition and termination

| Track | System | Compression % | Repeated trigrams % | Supplied word coverage % | Mean chunks* | Max chunks* | Extra consolidation passes* | Empty/failed | Cap failures | Punctuation flags |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| common_input | LSTM | 9.2 | 0.02 | 63.1 | 1.00 | 1 | 0.000 | 0 | 0 | 20 |
| common_input | Transformer | 8.2 | 0.04 | 62.1 | 1.00 | 1 | 0.000 | 0 | 0 | 5 |
| common_input | BART | 12.9 | 0.00 | 75.1 | 1.00 | 1 | 0.000 | 0 | 0 | 0 |
| common_input | PEGASUS | 10.3 | 1.28 | 75.1 | 1.00 | 1 | 0.000 | 0 | 0 | 0 |
| common_input | Lead-3 | 17.4 | 0.61 | 75.1 | 1.00 | 1 | 0.000 | 0 | 0 | 0 |
| full_article | LSTM | 10.0 | 0.08 | 100.0 | 1.22 | 3 | 0.000 | 1 | 0 | 16 |
| full_article | Transformer | 8.3 | 0.15 | 100.0 | 1.18 | 2 | 0.000 | 0 | 0 | 5 |
| full_article | BART | 13.9 | 0.00 | 100.0 | 1.20 | 2 | 0.000 | 0 | 0 | 0 |
| full_article | PEGASUS | 11.9 | 1.28 | 100.0 | 1.20 | 2 | 0.000 | 0 | 0 | 0 |
| full_article | Lead-3 | 17.4 | 0.61 | 100.0 | 1.00 | 1 | 0.000 | 0 | 0 | 0 |

*Completed requests only. Extra passes exclude source-order concatenation/exact-sentence deduplication. Punctuation flags are an automated heuristic, not human grammar ratings. Recovered maximum-length attempts are recorded separately from failed final outputs. Failure-stage coverage can be unknown; traces are preserved.

## Parameters and loading

| Model | Parameters | Cold load/transfer observations (seconds) |
|---|---:|---|
| LSTM | 42,789,205 | 0.500, 0.499 |
| Transformer | 81,468,757 | 1.169, 1.495 |
| BART | 406,290,432 | 1.826 |
| PEGASUS | 570,797,056 | 3.316 |
| Lead-3 | 0 | N/A |

PEGASUS includes 2,097,152 fixed sinusoidal positional values; learned count 568,699,904. Load observations include process-start initialization/device transfer and filesystem cache variation; operational/resume processes are listed rather than folded into decode latency.

## Paired uncertainty

[All paired ROUGE differences](reports/final_test_1000/paired_bootstrap.json) use 2,000 article-level paired bootstrap replicates, seed 42, 95% percentile intervals. Failures retain zero scores. These are pointwise intervals without multiple-comparison correction.

| Track | Difference (first − second) | R-Lsum difference | 95% CI | Paired n |
|---|---|---:|---|---:|
| common_input | LSTM − Transformer | 0.11 | [-1.87, 2.13] | 122 |
| common_input | LSTM − BART | -1.96 | [-9.97, 4.68] | 5 |
| common_input | LSTM − PEGASUS | 0.69 | [-6.31, 10.24] | 5 |
| common_input | LSTM − Lead-3 | 5.89 | [1.83, 9.93] | 5 |
| common_input | Transformer − BART | -1.43 | [-8.31, 4.80] | 5 |
| common_input | Transformer − PEGASUS | 1.22 | [-8.57, 15.33] | 5 |
| common_input | Transformer − Lead-3 | 6.43 | [3.08, 10.21] | 5 |
| common_input | BART − PEGASUS | 2.65 | [-8.27, 13.74] | 5 |
| common_input | BART − Lead-3 | 7.86 | [2.30, 13.59] | 5 |
| common_input | PEGASUS − Lead-3 | 5.20 | [-6.70, 15.26] | 5 |
| full_article | LSTM − Transformer | 0.98 | [-0.84, 2.64] | 122 |
| full_article | LSTM − BART | 1.75 | [-1.64, 4.39] | 5 |
| full_article | LSTM − PEGASUS | 3.20 | [-5.53, 11.94] | 5 |
| full_article | LSTM − Lead-3 | 5.15 | [1.04, 9.27] | 5 |
| full_article | Transformer − BART | 3.10 | [-0.48, 6.57] | 5 |
| full_article | Transformer − PEGASUS | 4.55 | [-4.03, 16.64] | 5 |
| full_article | Transformer − Lead-3 | 6.51 | [1.60, 10.65] | 5 |
| full_article | BART − PEGASUS | 1.45 | [-5.29, 10.59] | 5 |
| full_article | BART − Lead-3 | 3.40 | [-1.35, 9.78] | 5 |
| full_article | PEGASUS − Lead-3 | 1.95 | [-9.09, 12.35] | 5 |

## Review and interpretation

The [blinded review packet](reports/final_test_1000/blinded_review.html) uses 20
random frozen sample IDs (seed 142), with shuffled anonymous candidates for both
tracks. [Human rating fields](reports/final_test_1000/human_review_blank.csv) are
blank. **No human factuality ratings or population factual-error rates have been
measured; no factual-quality winner is established.** Candidate style can partially
reveal identity. The separate key is for unblinding after review.

Main-event coverage, entity/relationship errors, unsupported claims, attribution
and uncertainty require source-grounded review. Automatic repetition, punctuation,
EOS and overlap metrics cannot substitute for it. Riverton and the earlier five
known development failures are diagnostic evidence, not estimates of test accuracy.

The released systems differ in size, training/pretraining, input convention and
decoding. OpenNMT preprocessing approximates historical Stanford CoreNLP and its
full-article window exceeds the 400-token training convention. Concatenated chunk
summaries may miss global relationships or repeat paraphrases even with complete
source delivery. Full-article and common-prefix outputs must not be pooled.

[Illustrative outputs and all request failures](reports/final_test_1000/illustrative_examples.html)
include lowest, median and highest ROUGE-Lsum examples under a transparent selection
rule. These are diagnostic examples, not representative factuality judgments.

## Reproduce and resume

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe tests\fetch_final_sources.py
.\.venv\Scripts\python.exe tests\final_benchmark.py prepare
.\.venv\Scripts\python.exe tests\final_benchmark.py run
.\.venv\Scripts\python.exe tests\report_final_benchmark.py
```

The run command validates code, dependency versions, model files, source hashes and
the complete frozen fingerprint before reusing any prediction. Existing failed
attempts are preserved rather than silently retried/removed. The first five IDs
were the operational run and remain included. No generation tuning followed.
An incompatible environment fails validation; recover the recorded environment
and `test-results/final_test_1000/frozen_code.zip` rather than overwrite results.

Local exact sources and shared prefixes: `data/final_test_1000/prepared_sources.json`.
Exact per-call inputs, raw outputs/IDs, decoded/displayed text, budgets, coverage,
timing and scores: `test-results/final_test_1000/predictions/<track>/<model>/<id>.json`.
These bulk files remain ignored by Git. Portable summaries and predictions are in
[predictions_and_scores.csv](reports/final_test_1000/predictions_and_scores.csv),
[aggregate metrics](reports/final_test_1000/aggregate_metrics.json), and
[frozen configuration/IDs](reports/final_test_1000/frozen_config.json).

![ROUGE-Lsum](reports/final_test_1000/rouge_comparison.svg)

![Generation latency](reports/final_test_1000/latency_comparison.svg)
